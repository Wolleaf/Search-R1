import numpy as np
import pytest
import torch

from verl import DataProto
from verl.trainer import main_ppo


class _Tokenizer:

    def decode(self, sequence):
        return 'correct' if sequence[-1].item() == 11 else 'incorrect'


def _data(search_counts=None):
    tensors = {
        'prompts': torch.tensor([[1], [2]]),
        'responses': torch.tensor([[10, 11], [20, 22]]),
        'attention_mask': torch.ones(2, 3, dtype=torch.long),
    }
    if search_counts is not None:
        tensors['executed_search_count'] = torch.tensor(search_counts)
    return DataProto.from_dict(
        tensors=tensors,
        non_tensors={
            'reward_model': np.array([
                {'ground_truth': {'target': ['unused']}},
                {'ground_truth': {'target': ['unused']}},
            ], dtype=object),
            'data_source': np.array(['nq', 'nq'], dtype=object),
        },
    )


@pytest.fixture(autouse=True)
def _deterministic_em(monkeypatch):
    def score(solution_str, **_):
        return float(solution_str == 'correct')

    monkeypatch.setattr(main_ppo, '_select_rm_score_fn', lambda _: score)


def test_zero_lambda_preserves_em_reward_without_search_tensor():
    data = _data()
    reward = main_ppo.RewardManager(
        tokenizer=_Tokenizer(), num_examine=0, cost_lambda=0.0, max_searches=4)(data)

    assert reward.sum(-1).tolist() == [1.0, 0.0]
    assert data.batch['sequence_em_scores'].tolist() == [1.0, 0.0]
    assert data.batch['sequence_search_costs'].tolist() == [0.0, 0.0]


def test_cost_reward_is_normalized_by_max_searches():
    data = _data(search_counts=[4, 1])
    reward = main_ppo.RewardManager(
        tokenizer=_Tokenizer(), num_examine=0, cost_lambda=0.1, max_searches=4)(data)

    assert reward.sum(-1).tolist() == pytest.approx([0.9, -0.025])
    assert data.batch['sequence_em_scores'].tolist() == [1.0, 0.0]
    assert data.batch['sequence_search_costs'].tolist() == pytest.approx([0.1, 0.025])


def test_search_cost_stays_aligned_after_batch_reorder():
    data = _data(search_counts=[4, 0])
    data.reorder(torch.tensor([1, 0]))

    reward = main_ppo.RewardManager(
        tokenizer=_Tokenizer(), num_examine=0, cost_lambda=0.1, max_searches=4)(data)

    assert reward.sum(-1).tolist() == pytest.approx([0.0, 0.9])


def test_nonzero_lambda_requires_aligned_search_count():
    manager = main_ppo.RewardManager(
        tokenizer=_Tokenizer(), num_examine=0, cost_lambda=0.1, max_searches=4)

    with pytest.raises(KeyError, match='executed_search_count'):
        manager(_data())

    with pytest.raises(ValueError, match='must have shape'):
        manager(_data(search_counts=[[0], [1]]))
