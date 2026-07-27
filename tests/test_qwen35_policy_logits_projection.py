from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import verl.workers.actor.dp_actor as dp_actor_module
from verl.workers.actor.dp_actor import (DataParallelPPOActor,
                                         _policy_logit_projection_indices)


class _Config(dict):

    def __getattr__(self, name):
        return self[name]


class _FakeCausalLM(torch.nn.Module):

    def __init__(self, hidden_states, lm_head_weight, model_type):
        super().__init__()
        self.hidden_states = torch.nn.Parameter(hidden_states.clone())
        self.lm_head = torch.nn.Linear(hidden_states.size(1),
                                       lm_head_weight.size(0),
                                       bias=False,
                                       dtype=hidden_states.dtype)
        self.lm_head.weight.data.copy_(lm_head_weight)
        self.config = SimpleNamespace(model_type=model_type)
        self.last_logits_to_keep = None

    def forward(self, input_ids, logits_to_keep=0, **_kwargs):
        batch_size = input_ids.size(0)
        hidden_states = self.hidden_states.unsqueeze(0).repeat(
            batch_size, 1, 1)
        if isinstance(logits_to_keep, torch.Tensor):
            self.last_logits_to_keep = logits_to_keep.detach().clone()
            hidden_states = hidden_states.index_select(1, logits_to_keep)
        else:
            self.last_logits_to_keep = None
        logits = self.lm_head(hidden_states)
        return SimpleNamespace(logits=logits)


class _FakeFSDP(torch.nn.Module):

    def __init__(self, module):
        super().__init__()
        self._fsdp_wrapped_module = module

    def forward(self, *args, **kwargs):
        return self._fsdp_wrapped_module(*args, **kwargs)


def _make_actor(module):
    config = _Config(
        use_remove_padding=False,
        state_masking=True,
        ulysses_sequence_parallel_size=1,
    )
    with patch.object(torch, 'compile', side_effect=lambda fn, **_kwargs: fn):
        return DataParallelPPOActor(config=config, actor_module=module)


def _naive_log_probs(logits, labels):
    return torch.gather(torch.log_softmax(logits, dim=-1), -1,
                        labels.unsqueeze(-1)).squeeze(-1)


def test_policy_projection_maps_labels_to_predecessor_logits():
    input_ids = torch.zeros((2, 9), dtype=torch.long)
    responses = torch.zeros((2, 5), dtype=torch.long)
    loss_mask = torch.tensor([
        [0, 1, 0, 0, 1],
        [0, 0, 1, 0, 0],
    ])

    response_indices, logit_indices = _policy_logit_projection_indices(
        input_ids, responses, loss_mask)

    assert response_indices.tolist() == [1, 2, 4]
    assert logit_indices.tolist() == [4, 5, 7]


def test_policy_projection_rejects_an_empty_policy_mask():
    with pytest.raises(ValueError, match='at least one policy token'):
        _policy_logit_projection_indices(
            torch.zeros((1, 5), dtype=torch.long),
            torch.zeros((1, 3), dtype=torch.long),
            torch.zeros((1, 3), dtype=torch.long),
        )


@pytest.mark.parametrize('model_type', ['qwen3_5', 'qwen3_5_text'])
def test_qwen35_config_is_detected_through_fsdp_wrapper(model_type):
    module = _FakeCausalLM(
        torch.randn(5, 3),
        torch.randn(7, 3),
        model_type,
    )

    actor = _make_actor(_FakeFSDP(module))

    assert actor.use_qwen35_policy_logits is True


def test_qwen35_projection_preserves_masked_values_and_gradients():
    torch.manual_seed(7)
    sequence_length = 9
    response_length = 5
    vocab_size = 11
    hidden_size = 4
    hidden_states = torch.randn(sequence_length,
                                hidden_size,
                                dtype=torch.float64)
    lm_head_weight = torch.randn(vocab_size,
                                 hidden_size,
                                 dtype=torch.float64)
    qwen_module = _FakeCausalLM(hidden_states, lm_head_weight,
                                'qwen3_5_text')
    generic_module = _FakeCausalLM(hidden_states, lm_head_weight, 'generic')
    qwen_actor = _make_actor(_FakeFSDP(qwen_module))
    generic_actor = _make_actor(generic_module)

    responses = torch.tensor([
        [2, 4, 1, 8, 3],
        [5, 3, 9, 1, 6],
    ])
    loss_mask = torch.tensor([
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
    ])
    micro_batch = {
        'input_ids': torch.arange(sequence_length).repeat(2, 1),
        'attention_mask': torch.ones((2, sequence_length), dtype=torch.long),
        'position_ids': torch.arange(sequence_length).repeat(2, 1),
        'responses': responses,
        'loss_mask': loss_mask,
    }

    with patch.object(dp_actor_module.torch,
                      'autocast',
                      return_value=nullcontext()), patch.object(
                          dp_actor_module,
                          'logprobs_from_logits',
                          side_effect=_naive_log_probs):
        full_entropy, full_log_probs = generic_actor._forward_micro_batch(
            micro_batch, temperature=1.0)
        sparse_entropy, sparse_log_probs = qwen_actor._forward_micro_batch(
            micro_batch, temperature=1.0)

    policy_mask = loss_mask.bool()
    assert qwen_module.last_logits_to_keep.tolist() == [4, 5, 6, 7]
    assert generic_module.last_logits_to_keep is None
    torch.testing.assert_close(sparse_entropy[policy_mask],
                               full_entropy[policy_mask])
    torch.testing.assert_close(sparse_log_probs[policy_mask],
                               full_log_probs[policy_mask])
    assert torch.count_nonzero(sparse_entropy[:, 0]) == 0
    assert torch.count_nonzero(sparse_log_probs[:, 0]) == 0

    full_loss = (full_entropy[policy_mask].mean() +
                 full_log_probs[policy_mask].mean())
    sparse_loss = (sparse_entropy[policy_mask].mean() +
                   sparse_log_probs[policy_mask].mean())
    full_loss.backward()
    sparse_loss.backward()
    torch.testing.assert_close(qwen_module.hidden_states.grad,
                               generic_module.hidden_states.grad)
    torch.testing.assert_close(qwen_module.lm_head.weight.grad,
                               generic_module.lm_head.weight.grad)
