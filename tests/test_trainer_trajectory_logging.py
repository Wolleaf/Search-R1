import json
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import torch

from search_r1.trajectory_trace import TraceJsonlWriter
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, _compute_group_metrics


class _Tokenizer:

    def decode(self, token_ids):
        prompt = (
            '<|im_start|>user\nUse <answer> Beijing </answer>. '
            'Question: What is the capital of France?\n<|im_end|>\n'
            '<|im_start|>assistant\n')
        if len(token_ids) <= 2:
            return prompt
        return prompt + (
            '<think>I should verify it.</think><search>France capital</search>'
            '<information>Doc 1 says Paris.</information>'
            '<think>The evidence is clear.</think><answer>Paris</answer>')


def _object_array(values):
    output = np.empty(len(values), dtype=object)
    output[:] = values
    return output


def test_training_trace_integration_keeps_group_fields_aligned(tmp_path):
    batch_size = 5
    tensors = {
        'prompts': torch.tensor([[1, 2]] * batch_size),
        'responses': torch.tensor([[3, 4, 5]] * batch_size),
        'attention_mask': torch.ones(batch_size, 5, dtype=torch.long),
        'executed_search_count': torch.ones(batch_size, dtype=torch.long),
        'sequence_em_scores': torch.tensor([1, 0, 0, 0, 0], dtype=torch.float32),
        'sequence_train_rewards': torch.tensor([0.975, 0, 0, 0, 0]),
        'sequence_posthoc_utilities': torch.tensor([0.975, -0.025, -0.025, -0.025, -0.025]),
        'advantages': torch.tensor([[1.7888] * 3] + [[-0.4472] * 3] * 4),
    }
    retrieval = [[{
        'turn': 0,
        'query': 'France capital',
        'documents': [{
            'document_id': '7',
            'score': 2.5,
            'document': {'contents': 'France\nParis is the capital.'},
        }],
    }] for _ in range(batch_size)]
    generations = [[{
        'turn': 0,
        'text': '<search>France capital</search>',
        'token_count': 3,
        'clipped': False,
    }] for _ in range(batch_size)]
    batch = DataProto.from_dict(
        tensors=tensors,
        non_tensors={
            'data_source': np.array(['nq'] * batch_size, dtype=object),
            'index': np.array([17] * batch_size, dtype=object),
            'uid': np.array([17] * batch_size, dtype=object),
            'group_slot': np.arange(batch_size, dtype=object),
            'extra_info': _object_array([
                {'split': 'train', 'index': 17} for _ in range(batch_size)
            ]),
            'reward_model': _object_array([{
                'ground_truth': {'target': ['Paris']},
            } for _ in range(batch_size)]),
            'retrieval_events': _object_array(retrieval),
            'generation_events': _object_array(generations),
        },
    )

    trainer = object.__new__(RayPPOTrainer)
    trainer.tokenizer = _Tokenizer()
    trainer.global_steps = 1
    trainer.config = OmegaConf.create({
        'data': {'max_prompt_length': 32},
        'algorithm': {'cost_reward_mode': 'correct_only', 'cost_lambda': 0.1},
        'actor_rollout_ref': {'rollout': {'n_agent': 5}},
        'trainer': {'trace_parent_checkpoint_digest': 'a' * 64},
        'max_turns': 4,
    })
    trainer.train_trace_writer = TraceJsonlWriter(
        tmp_path / 'train_trajectories.jsonl',
        record_type='train',
        expected_rows=5,
        run_id='unit-train',
        stage='cost_aware_gated',
    )

    trainer._append_train_traces(batch)
    trainer.train_trace_writer.finalize()
    records = [json.loads(line) for line in (
        tmp_path / 'train_trajectories.jsonl').read_text(
            encoding='utf-8').splitlines()]

    assert [record['group_slot'] for record in records] == list(range(5))
    assert all(record['group_correct_count'] == 1 for record in records)
    assert records[0]['question'] == 'What is the capital of France?'
    assert records[0]['retrieval_events'][0]['documents'][0]['document_id'] == '7'
    assert records[0]['parent_checkpoint_digest'] == 'a' * 64

    metrics = _compute_group_metrics(batch)
    assert metrics['env/group/all_wrong_ratio'] == 0.0
    assert metrics['env/group/correct_count_1_ratio'] == 1.0
    assert metrics['env/search_count/correct_mean'] == 1.0
    assert metrics['env/search_count/wrong_mean'] == 1.0
