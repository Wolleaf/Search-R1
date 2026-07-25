import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from search_r1.trajectory_trace import TraceJsonlWriter
from verl import DataProto
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    _get_eval_group_size,
    _prepare_validation_batch,
    _validation_meta_info,
)


def _eval_record():
    return {
        'sample_id': 'hotpotqa:train:17',
        'source_index': 17,
        'question': 'Which city is the capital?',
        'gold_answers': ['Paris'],
        'raw_trajectory': '<answer>Paris</answer>',
        'raw_generations': [{
            'turn': 0,
            'raw_text': '<answer>Paris</answer>',
            'raw_token_ids': [10, 11, 12, 13],
            'raw_token_count': 4,
            'action_text': '<answer>Paris</answer>',
            'action_token_ids': [10, 11, 12, 13],
            'action_token_count': 4,
            'boundary': 'answer',
            'tail_dropped': False,
            'raw_clipped': False,
        }],
        'turns': [],
        'extracted_answer': 'Paris',
        'em': 1,
        'executed_search_count': 0,
        'max_action_budget': 4,
        'action_count': 1,
        'policy_token_count': 4,
        'observation_token_count': 0,
        'observation_policy_token_count': 0,
        'info_mask_consistent': True,
        'posthoc_utility': 1.0,
        'response_tokens': 4,
        'response_clipped': False,
        'turns_used': 0,
        'invalid_action_count': 0,
    }


def test_ppo_config_defaults_to_single_validation_trajectory():
    config = OmegaConf.load(
        Path(__file__).parents[1] / 'verl/trainer/config/ppo_trainer.yaml')

    assert config.data.eval_group_size == 1


def test_default_eval_group_size_preserves_legacy_batch_and_greedy_mode():
    config = OmegaConf.create({'data': {}})
    tokenizer = SimpleNamespace(eos_token_id=1, pad_token_id=0)
    batch = DataProto.from_dict(
        tensors={'input_ids': torch.tensor([[1], [2]])},
        non_tensors={'index': np.array([10, 20], dtype=object)},
    )

    group_size = _get_eval_group_size(config)

    assert group_size == 1
    assert _prepare_validation_batch(batch, group_size) is batch
    assert 'group_slot' not in batch.non_tensor_batch
    assert _validation_meta_info(tokenizer, group_size)['do_sample'] is False


@pytest.mark.parametrize('group_size', [0, -1, True, 1.5, '5'])
def test_eval_group_size_rejects_invalid_values(group_size):
    config = OmegaConf.create({'data': {'eval_group_size': group_size}})

    with pytest.raises(ValueError, match='positive integer'):
        _get_eval_group_size(config)


def test_grouped_validation_repeats_each_prompt_with_aligned_slots():
    tokenizer = SimpleNamespace(eos_token_id=1, pad_token_id=0)
    batch = DataProto.from_dict(
        tensors={'input_ids': torch.tensor([[1], [2]])},
        non_tensors={'index': np.array([10, 20], dtype=object)},
    )

    repeated = _prepare_validation_batch(batch, 5)

    assert repeated.batch['input_ids'].flatten().tolist() == [1] * 5 + [2] * 5
    assert repeated.non_tensor_batch['index'].tolist() == [10] * 5 + [20] * 5
    assert repeated.non_tensor_batch['group_slot'].tolist() == list(
        range(5)) * 2
    assert repeated.non_tensor_batch['group_size'].tolist() == [5] * 10
    assert _validation_meta_info(tokenizer, 5)['do_sample'] is True


def test_grouped_eval_trace_records_have_unique_slot_identity(tmp_path):
    trainer = object.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create({
        'data': {
            'eval_group_size': 5
        },
        'trainer': {
            'trace_checkpoint_digest': 'a' * 64,
        },
    })
    trainer.eval_trace_writer = TraceJsonlWriter(
        tmp_path / 'eval_predictions.jsonl',
        record_type='eval',
        expected_rows=5,
        run_id='probe',
        stage='grouped-probe',
    )
    trainer._common_trace_records = lambda batch: [
        _eval_record() for _ in range(5)
    ]
    batch = SimpleNamespace(
        non_tensor_batch={
            'group_slot': np.arange(5, dtype=object),
            'group_size': np.full(5, 5, dtype=object),
        })

    trainer._append_eval_traces(batch)
    trainer.eval_trace_writer.finalize()
    records = [
        json.loads(line)
        for line in (tmp_path / 'eval_predictions.jsonl').read_text(
            encoding='utf-8').splitlines()
    ]

    assert [record['group_slot'] for record in records] == list(range(5))
    assert {record['group_uid'] for record in records} == {'hotpotqa:train:17'}
    assert {record['group_size'] for record in records} == {5}
    assert len({record['record_id'] for record in records}) == 5


def test_grouped_eval_trace_writer_reserves_repeated_row_count(tmp_path):
    trainer = object.__new__(RayPPOTrainer)
    trainer.val_dataloader = [None, None]
    trainer.config = OmegaConf.create({
        'data': {
            'val_batch_size': 8,
            'eval_group_size': 5
        },
        'trainer': {
            'val_only': True,
            'experiment_name': 'grouped-probe',
            'trace_output_dir': str(tmp_path),
            'trace_stage': 'grouped-probe',
            'trace_run_id': 'probe',
            'trace_checkpoint_digest': 'a' * 64,
        },
    })
    trainer.train_trace_writer = None
    trainer.eval_trace_writer = None

    trainer._init_trace_writer()

    assert trainer.eval_trace_writer.expected_rows == 80
    trainer.eval_trace_writer.close()
