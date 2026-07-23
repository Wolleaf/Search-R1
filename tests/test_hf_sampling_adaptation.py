import contextlib
from types import SimpleNamespace

import pytest
import torch

from verl import DataProto
from verl.workers.rollout.hf_rollout import (
    HFRollout,
    _PresencePenaltyLogitsProcessor,
)


class _Config(dict):

    def __getattr__(self, name):
        return self[name]


class _GenerateRecorder(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.generate_kwargs = None

    def generate(self, input_ids, max_new_tokens, **kwargs):
        self.generate_kwargs = kwargs
        suffix = torch.full(
            (input_ids.size(0), max_new_tokens),
            7,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        return SimpleNamespace(sequences=torch.cat((input_ids, suffix), dim=1))


def _prompts(meta_info=None):
    metadata = {'eos_token_id': 9, 'pad_token_id': 0}
    metadata.update(meta_info or {})
    return DataProto.from_dict(
        tensors={
            'input_ids': torch.tensor([[0, 0, 4]], dtype=torch.long),
            'attention_mask': torch.tensor([[0, 0, 1]], dtype=torch.long),
            'position_ids': torch.tensor([[0, 0, 0]], dtype=torch.long),
        },
        meta_info=metadata,
    )


def _config(**overrides):
    config = _Config(
        do_sample=True,
        response_length=2,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
    )
    config.update(overrides)
    return config


def _run_minibatch(monkeypatch, config, meta_info=None):
    module = _GenerateRecorder()
    rollout = HFRollout(module=module, config=config)
    monkeypatch.setattr(
        torch,
        'autocast',
        lambda *args, **kwargs: contextlib.nullcontext(),
    )
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    rollout._generate_minibatch(_prompts(meta_info))
    return module.generate_kwargs


def test_presence_penalty_ignores_prompt_and_counts_generated_token_once():
    processor = _PresencePenaltyLogitsProcessor(
        penalty=2.0,
        prompt_length=3,
    )
    input_ids = torch.tensor([
        [0, 5, 6, 1, 1, 3],
        [4, 1, 2, 2, 4, 4],
    ])
    scores = torch.zeros((2, 8))

    actual = processor(input_ids, scores)

    expected = torch.zeros_like(scores)
    expected[0, [1, 3]] = -2.0
    expected[1, [2, 4]] = -2.0
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(scores, torch.zeros_like(scores))


def test_presence_penalty_resets_when_the_next_turn_becomes_prompt():
    scores = torch.zeros((1, 8))
    first_turn = _PresencePenaltyLogitsProcessor(2.0, prompt_length=2)
    second_turn = _PresencePenaltyLogitsProcessor(2.0, prompt_length=3)
    input_ids = torch.tensor([[5, 6, 4]])

    assert first_turn(input_ids, scores)[0, 4].item() == -2.0
    assert second_turn(input_ids, scores) is scores


def test_zero_presence_penalty_is_an_identity():
    processor = _PresencePenaltyLogitsProcessor(0.0, prompt_length=1)
    scores = torch.randn((2, 10))

    actual = processor(torch.tensor([[1, 2], [3, 4]]), scores)

    assert actual is scores


@pytest.mark.parametrize('penalty', [-2.0, 2.0])
def test_presence_penalty_accepts_documented_boundaries(penalty):
    _PresencePenaltyLogitsProcessor(penalty, prompt_length=0)


@pytest.mark.parametrize(
    'penalty',
    [-2.01, 2.01, float('nan'),
     float('inf'), True, '2.0'],
)
def test_presence_penalty_rejects_invalid_values(penalty):
    with pytest.raises(ValueError, match='presence_penalty'):
        _PresencePenaltyLogitsProcessor(penalty, prompt_length=0)


def test_zero_sampling_defaults_preserve_the_generate_call(monkeypatch):
    kwargs = _run_minibatch(monkeypatch, _config())
    generation_config = kwargs['generation_config']

    assert 'logits_processor' not in kwargs
    assert generation_config.top_k == 0
    assert generation_config.min_p is None
    assert generation_config.repetition_penalty == 1.0


def test_hf_rollout_passes_non_default_sampling_adapters(monkeypatch):
    kwargs = _run_minibatch(
        monkeypatch,
        _config(
            top_k=20,
            min_p=0.1,
            presence_penalty=2.0,
            repetition_penalty=1.1,
        ),
    )
    generation_config = kwargs['generation_config']
    processors = kwargs['logits_processor']

    assert generation_config.top_k == 20
    assert generation_config.min_p == 0.1
    assert generation_config.repetition_penalty == 1.1
    assert len(processors) == 1
    assert isinstance(processors[0], _PresencePenaltyLogitsProcessor)
    assert processors[0].prompt_length == 3
    assert processors[0].penalty == 2.0


def test_sampling_metadata_overrides_rollout_defaults(monkeypatch):
    kwargs = _run_minibatch(
        monkeypatch,
        _config(),
        meta_info={
            'top_k': 20,
            'min_p': 0.2,
            'presence_penalty': -1.5,
            'repetition_penalty': 1.2,
        },
    )
    generation_config = kwargs['generation_config']
    processor = kwargs['logits_processor'][0]

    assert generation_config.top_k == 20
    assert generation_config.min_p == 0.2
    assert generation_config.repetition_penalty == 1.2
    assert processor.penalty == -1.5


@pytest.mark.parametrize(
    ('field', 'value', 'message'),
    [
        ('min_p', -0.1, 'min_p'),
        ('min_p', True, 'min_p'),
        ('presence_penalty', False, 'presence_penalty'),
        ('repetition_penalty', 0.0, 'repetition_penalty'),
        ('repetition_penalty', float('nan'), 'repetition_penalty'),
    ],
)
def test_hf_rollout_rejects_invalid_sampling_config(monkeypatch, field, value,
                                                    message):
    with pytest.raises(ValueError, match=message):
        _run_minibatch(monkeypatch, _config(**{field: value}))
