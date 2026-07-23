import numpy as np
import pytest
import torch

from verl import DataProto
from verl.trainer import main_ppo
from verl.trainer.ppo import core_algos


class _Tokenizer:

    def decode(self, sequence):
        return 'correct' if sequence[-1].item() == 11 else 'incorrect'


def _data(search_counts=None, final_answers=None):
    tensors = {
        'prompts': torch.tensor([[1], [2]]),
        'responses': torch.tensor([[10, 11], [20, 22]]),
        'attention_mask': torch.ones(2, 3, dtype=torch.long),
    }
    if search_counts is not None:
        tensors['executed_search_count'] = torch.tensor(search_counts)
    non_tensors = {
        'reward_model': np.array([
            {'ground_truth': {'target': ['unused']}},
            {'ground_truth': {'target': ['unused']}},
        ], dtype=object),
        'data_source': np.array(['nq', 'nq'], dtype=object),
    }
    if final_answers is not None:
        non_tensors['final_answer'] = np.array(final_answers, dtype=object)
    return DataProto.from_dict(
        tensors=tensors,
        non_tensors=non_tensors,
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
    assert data.batch['sequence_train_rewards'].tolist() == pytest.approx([0.9, -0.025])
    assert data.batch['sequence_posthoc_utilities'].tolist() == pytest.approx([0.9, -0.025])


def test_correct_only_cost_is_gated_by_em_but_keeps_posthoc_utility():
    data = _data(search_counts=[4, 1])
    reward = main_ppo.RewardManager(
        tokenizer=_Tokenizer(),
        num_examine=0,
        cost_lambda=0.1,
        max_searches=4,
        cost_reward_mode='correct_only',
    )(data)

    assert reward.sum(-1).tolist() == pytest.approx([0.9, 0.0])
    assert data.batch['sequence_train_rewards'].tolist() == pytest.approx([0.9, 0.0])
    assert data.batch['sequence_posthoc_utilities'].tolist() == pytest.approx([0.9, -0.025])


def test_unknown_cost_reward_mode_is_rejected():
    with pytest.raises(ValueError, match='cost_reward_mode'):
        main_ppo.RewardManager(
            tokenizer=_Tokenizer(), num_examine=0, cost_reward_mode='unknown')

    with pytest.raises(ValueError, match='finite'):
        main_ppo.RewardManager(
            tokenizer=_Tokenizer(), num_examine=0, cost_lambda=float('nan'))


def test_correct_only_all_wrong_group_has_zero_advantage():
    data = _data(search_counts=[0, 4])
    data.batch['responses'][0, -1] = 12
    reward = main_ppo.RewardManager(
        tokenizer=_Tokenizer(),
        num_examine=0,
        cost_lambda=0.1,
        max_searches=4,
        cost_reward_mode='correct_only',
    )(data)

    advantages, _ = core_algos.compute_grpo_outcome_advantage(
        token_level_rewards=reward,
        eos_mask=torch.ones_like(reward),
        index=np.array(['same-question', 'same-question'], dtype=object),
    )

    assert reward.sum(-1).tolist() == [0.0, 0.0]
    assert torch.count_nonzero(advantages).item() == 0


def test_search_count_cannot_exceed_the_execution_budget():
    manager = main_ppo.RewardManager(
        tokenizer=_Tokenizer(), num_examine=0, cost_lambda=0.1, max_searches=4)

    with pytest.raises(ValueError, match='exceeds'):
        manager(_data(search_counts=[0, 5]))


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


def test_native_reward_uses_aligned_plain_final_answer():
    data = _data(search_counts=[1, 4], final_answers=['Paris', 'London'])
    data.non_tensor_batch['reward_model'] = np.array([
        {'ground_truth': {'target': ['paris']}},
        {'ground_truth': {'target': ['Paris']}},
    ], dtype=object)
    data.reorder(torch.tensor([1, 0]))

    reward = main_ppo.RewardManager(
        tokenizer=_Tokenizer(),
        num_examine=0,
        cost_lambda=0.1,
        max_searches=4,
        tool_protocol='qwen35_native',
    )(data)

    assert reward.sum(-1).tolist() == pytest.approx([-0.1, 0.975])
    assert data.batch['sequence_em_scores'].tolist() == [0.0, 1.0]


def test_native_unfinished_answer_scores_zero_without_fabricating_xml():
    data = _data(search_counts=[0, 0], final_answers=[None, ''])

    reward = main_ppo.RewardManager(
        tokenizer=_Tokenizer(),
        num_examine=0,
        tool_protocol='qwen35_native',
    )(data)

    assert reward.sum(-1).tolist() == [0.0, 0.0]


def test_native_reward_requires_environment_final_answer():
    manager = main_ppo.RewardManager(
        tokenizer=_Tokenizer(),
        num_examine=0,
        tool_protocol='qwen35_native',
    )

    with pytest.raises(KeyError, match='final_answer'):
        manager(_data(search_counts=[0, 0]))
