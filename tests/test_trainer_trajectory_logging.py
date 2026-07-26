import json
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from search_r1.trajectory_trace import TraceJsonlWriter
from search_r1.llm_agent.tool_protocol import QWEN35_PROMPT_VERSION
from verl import DataProto
from verl.trainer.ppo.ray_trainer import (RayPPOTrainer,
                                          _compute_group_metrics,
                                          _event_aligned_trace_turns,
                                          _validate_qwen35_native_training_batch,
                                          _validate_qwen35_native_training_contract)


class _Tokenizer:

    def __init__(self, decoded_query='France capital', trailing_output=''):
        self.decoded_query = decoded_query
        self.trailing_output = trailing_output

    def decode(self, token_ids):
        prompt = (
            '<|im_start|>user\nUse <answer> Beijing </answer>. '
            'Question: What is the capital of France?\n<|im_end|>\n'
            '<|im_start|>assistant\n')
        if len(token_ids) <= 2:
            return prompt
        return prompt + (
            f'<think>I should verify it.</think><search>{self.decoded_query}</search>'
            '<information>Doc 1 says Paris.</information>'
            '<think>The evidence is clear.</think><answer>Paris</answer>'
            f'{self.trailing_output}')


def _object_array(values):
    output = np.empty(len(values), dtype=object)
    output[:] = values
    return output


def test_native_protocol_requires_search_agent_loop_at_trainer_setup():
    config = OmegaConf.create({
        'tool_protocol': 'qwen35_native',
        'do_search': False,
    })

    with pytest.raises(ValueError, match=(
            'qwen35_native requires do_search=true')):
        RayPPOTrainer(
            config=config,
            tokenizer=None,
            role_worker_mapping={},
            resource_pool_manager=None,
        )


def _native_training_config(variant='reproduce'):
    variant_contracts = {
        'smoke': (2, 0.0, 'linear'),
        'reproduce': (60, 0.0, 'linear'),
        'control': (20, 0.0, 'linear'),
        'cost_aware_gated': (20, 0.10, 'correct_only'),
    }
    steps, cost_lambda, cost_mode = variant_contracts[variant]
    return OmegaConf.create({
        'tool_protocol': 'qwen35_native',
        'do_search': True,
        'qwen35_prompt_version': QWEN35_PROMPT_VERSION,
        'trainer': {
            'val_only': False,
            'native_training_variant': variant,
            'total_training_steps': steps,
        },
        'algorithm': {
            'adv_estimator': 'grpo',
            'cost_lambda': cost_lambda,
            'cost_reward_mode': cost_mode,
        },
        'actor_rollout_ref': {
            'actor': {
                'state_masking': True,
                'ppo_mini_batch_size': 40,
                'ppo_micro_batch_size': 2,
            },
            'rollout': {
                'name': 'hf',
                'do_sample': True,
                'temperature': 1.0,
                'top_p': 1.0,
                'top_k': 0,
                'min_p': 0.0,
                'presence_penalty': 0.0,
                'repetition_penalty': 1.0,
                'n': 1,
                'n_agent': 5,
                'log_prob_micro_batch_size': 2,
            },
            'ref': {
                'log_prob_micro_batch_size': 2,
            },
        },
        'data': {
            'train_batch_size': 8,
            'return_raw_chat': True,
            'max_prompt_length': 4096,
            'max_response_length': 500,
            'max_start_length': 1024,
            'max_obs_length': 500,
        },
        'max_turns': 4,
        'retriever': {
            'topk': 3,
        },
    })


@pytest.mark.parametrize('variant', [
    'smoke', 'reproduce', 'control', 'cost_aware_gated'
])
def test_native_protocol_accepts_registered_training_contract(variant):
    _validate_qwen35_native_training_contract(
        _native_training_config(variant))


def test_native_val_only_does_not_require_training_contract():
    config = OmegaConf.create({
        'tool_protocol': 'qwen35_native',
        'do_search': True,
        'trainer': {
            'val_only': True,
        },
    })

    _validate_qwen35_native_training_contract(config)


@pytest.mark.parametrize(('path', 'value'), [
    ('qwen35_prompt_version', 'qwen35-native-search-v1'),
    ('trainer.native_training_variant', 'cost_aware'),
    ('algorithm.adv_estimator', 'gae'),
    ('actor_rollout_ref.rollout.name', 'vllm'),
    ('actor_rollout_ref.rollout.do_sample', False),
    ('actor_rollout_ref.rollout.temperature', 0.9),
    ('actor_rollout_ref.rollout.top_p', 0.95),
    ('actor_rollout_ref.rollout.top_k', 20),
    ('actor_rollout_ref.rollout.min_p', 0.1),
    ('actor_rollout_ref.rollout.presence_penalty', 2.0),
    ('actor_rollout_ref.rollout.repetition_penalty', 1.1),
    ('actor_rollout_ref.rollout.n', 2),
    ('actor_rollout_ref.rollout.n_agent', 4),
    ('actor_rollout_ref.actor.state_masking', False),
    ('actor_rollout_ref.actor.ppo_mini_batch_size', 20),
    ('actor_rollout_ref.actor.ppo_micro_batch_size', 1),
    ('actor_rollout_ref.rollout.log_prob_micro_batch_size', 1),
    ('actor_rollout_ref.ref.log_prob_micro_batch_size', 1),
    ('data.train_batch_size', 4),
    ('data.return_raw_chat', False),
    ('data.max_prompt_length', 4036),
    ('data.max_response_length', 384),
    ('data.max_start_length', 512),
    ('data.max_obs_length', 384),
    ('trainer.total_training_steps', 59),
    ('max_turns', 3),
    ('retriever.topk', 5),
])
def test_native_protocol_rejects_training_contract_drift(path, value):
    config = _native_training_config()
    OmegaConf.update(config, path, value, merge=False)

    with pytest.raises(ValueError, match=path.replace('.', r'\.')):
        _validate_qwen35_native_training_contract(config)


def test_native_protocol_binds_reward_to_training_variant():
    config = _native_training_config('cost_aware_gated')
    config.algorithm.cost_lambda = 0.0

    with pytest.raises(ValueError, match=r'algorithm\.cost_lambda'):
        _validate_qwen35_native_training_contract(config)


def _native_training_batch():
    responses = torch.tensor([[10, 11, 12], [20, 21, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1],
                                   [1, 1, 1, 1, 0]])
    info_mask = torch.tensor([[1, 1, 1, 0, 1],
                              [1, 1, 1, 1, 0]])
    loss_mask = info_mask[:, -responses.shape[1]:].clone()
    return DataProto.from_dict({
        'responses': responses,
        'old_log_probs': torch.tensor([[-1.0, -2.0, -3.0],
                                       [-1.5, -2.5, 0.0]]),
        'advantages': torch.tensor([[1.0, 0.0, -1.0],
                                    [0.5, -0.5, 0.0]]),
        'token_level_rewards': torch.tensor([[0.0, 0.0, 1.0],
                                             [0.0, 0.0, 0.0]]),
        'attention_mask': attention_mask,
        'info_mask': info_mask,
        'loss_mask': loss_mask,
    })


def test_native_training_batch_contract_returns_auditable_metrics():
    metrics = _validate_qwen35_native_training_batch(_native_training_batch())

    assert metrics['native_batch/contract_valid'] == 1.0
    assert metrics['native_batch/trajectories'] == 2.0
    assert metrics['native_batch/response_width'] == 3.0
    assert metrics['native_batch/policy_tokens'] == 4.0
    assert metrics['native_batch/policy_tokens_min_per_trajectory'] == 2.0
    assert metrics['native_batch/old_log_prob_finite_ratio'] == 1.0
    assert metrics['native_batch/advantage_finite_ratio'] == 1.0
    assert metrics['native_batch/reward_finite_ratio'] == 1.0
    assert metrics['native_batch/nonzero_advantage_tokens'] == 4.0


@pytest.mark.parametrize('key', [
    'old_log_probs', 'advantages', 'token_level_rewards', 'loss_mask'
])
def test_native_training_batch_contract_rejects_response_shape_drift(key):
    batch = _native_training_batch()
    batch.batch[key] = batch.batch[key][:, :-1]

    with pytest.raises(ValueError, match=key):
        _validate_qwen35_native_training_batch(batch)


def test_native_training_batch_contract_rejects_mask_shape_drift():
    batch = _native_training_batch()
    batch.batch['info_mask'] = batch.batch['info_mask'][:, :-1]

    with pytest.raises(ValueError, match='attention_mask and info_mask'):
        _validate_qwen35_native_training_batch(batch)


def test_native_training_batch_contract_rejects_info_loss_mask_drift():
    batch = _native_training_batch()
    batch.batch['loss_mask'][0, 1] = 1

    with pytest.raises(ValueError, match='loss_mask must equal'):
        _validate_qwen35_native_training_batch(batch)


def test_native_training_batch_contract_rejects_policy_outside_response():
    batch = _native_training_batch()
    batch.batch['attention_mask'][0, -1] = 0

    with pytest.raises(ValueError, match='not a subset'):
        _validate_qwen35_native_training_batch(batch)


def test_native_training_batch_contract_rejects_empty_policy_trajectory():
    batch = _native_training_batch()
    batch.batch['info_mask'][1, -3:] = 0
    batch.batch['loss_mask'][1] = 0

    with pytest.raises(ValueError, match='nonempty for every trajectory'):
        _validate_qwen35_native_training_batch(batch)


@pytest.mark.parametrize(('key', 'value'), [
    ('old_log_probs', float('nan')),
    ('advantages', float('inf')),
    ('token_level_rewards', float('-inf')),
])
def test_native_training_batch_contract_rejects_nonfinite_policy_values(
        key, value):
    batch = _native_training_batch()
    batch.batch[key][0, 0] = value

    with pytest.raises(ValueError, match=f'non-finite {key}'):
        _validate_qwen35_native_training_batch(batch)


def test_native_training_batch_records_zero_advantage_without_aborting():
    batch = _native_training_batch()
    batch.batch['advantages'].zero_()

    metrics = _validate_qwen35_native_training_batch(batch)
    assert metrics['native_batch/nonzero_advantage_tokens'] == 0.0
    assert metrics['native_batch/advantage_abs_max'] == 0.0


@pytest.mark.parametrize(('decoded_query', 'trailing_output',
                          'action_overshoot'), [
    ('France capital', '', ''),
    ('France  capital', '', '.'),
    ('France capital',
     '<answer>Paris, France</answer><search>France capital</search>', ''),
])
def test_training_trace_integration_keeps_group_fields_aligned(
        tmp_path, decoded_query, trailing_output, action_overshoot):
    batch_size = 5
    tensors = {
        'prompts': torch.tensor([[1, 2]] * batch_size),
        'responses': torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10, 11]] * batch_size),
        'attention_mask': torch.ones(batch_size, 11, dtype=torch.long),
        'info_mask': torch.tensor(
            [[1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1]] * batch_size),
        'action_count': torch.full((batch_size,), 2, dtype=torch.long),
        'executed_search_count': torch.ones(batch_size, dtype=torch.long),
        'sequence_em_scores': torch.tensor([1, 0, 0, 0, 0], dtype=torch.float32),
        'sequence_train_rewards': torch.tensor([0.975, 0, 0, 0, 0]),
        'sequence_posthoc_utilities': torch.tensor([0.975, -0.025, -0.025, -0.025, -0.025]),
        'advantages': torch.tensor([[1.7888] * 9] + [[-0.4472] * 9] * 4),
    }
    retrieval = [[{
        'turn': 0,
        'query': 'France capital',
        'documents': [{
            'document_id': '7',
            'score': 2.5,
            'document': {'contents': 'France\nParis is the capital.'},
        }],
        'observation': 'Doc 1 says Paris.',
    }] for _ in range(batch_size)]
    search_text = (
        'I need evidence.</think><tool_call><function=search>'
        '<parameter=query>France capital'
        '</parameter></function></tool_call>')
    answer_text = (
        'The evidence is clear.</think><answer>Paris</answer>'
        f'{action_overshoot}')
    generations = [[{
        'turn': 0,
        'text': search_text,
        'raw_text': search_text,
        'raw_token_ids': [30, 31, 32],
        'raw_token_count': 3,
        'action_token_ids': [30, 31, 32],
        'boundary': 'tool_call',
        'tail_dropped': False,
        'raw_clipped': False,
        'token_count': 3,
        'clipped': False,
        'valid_action': True,
        'done': False,
        'executed_search': True,
        'action': 'search',
        'content': 'France capital',
        'parse_error': None,
        'reasoning_prefix': 'I need evidence.',
    }, {
        'turn': 1,
        'text': answer_text,
        'raw_text': answer_text + trailing_output,
        'raw_token_ids': ([40, 41, 42, 43] + ([44] if trailing_output else [])),
        'raw_token_count': 4 + int(bool(trailing_output)),
        'action_token_ids': [40, 41, 42, 43],
        'boundary': 'answer',
        'tail_dropped': bool(trailing_output),
        'raw_clipped': False,
        'token_count': 4,
        'clipped': False,
        'valid_action': True,
        'done': True,
        'executed_search': False,
        'action': 'answer',
        'content': 'Paris',
        'parse_error': None,
        'reasoning_prefix': 'The evidence is clear.',
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
    trainer.tokenizer = _Tokenizer(decoded_query)
    trainer.global_steps = 1
    trainer.config = OmegaConf.create({
        'tool_protocol': 'qwen35_native',
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
    executed_turns = [
        turn for turn in records[0]['turns']
        if turn['retrieval_executed']
    ]
    assert len(executed_turns) == 1
    assert executed_turns[0]['search_query'] == 'France capital'
    assert executed_turns[0]['observation'] == 'Doc 1 says Paris.'
    assert decoded_query in records[0]['raw_trajectory']
    if trailing_output:
        assert records[0]['turns'][-1]['action'] == 'answer'
        assert trailing_output in records[0]['raw_generations'][-1]['raw_text']
        assert trailing_output not in records[0]['raw_trajectory']
    if action_overshoot:
        generation = records[0]['raw_generations'][-1]
        assert generation['action_text'].endswith('</answer>.')
        assert generation['boundary'] == 'answer'
        assert generation['action_token_ids'] == [40, 41, 42, 43]
    assert records[0]['max_action_budget'] == 4
    assert records[0]['action_count'] == 2
    assert records[0]['policy_token_count'] == 7
    assert records[0]['observation_token_count'] == 2
    assert records[0]['observation_policy_token_count'] == 0
    assert records[0]['info_mask_consistent'] is True

    metrics = _compute_group_metrics(batch)
    assert metrics['env/group/all_wrong_ratio'] == 0.0
    assert metrics['env/group/correct_count_1_ratio'] == 1.0
    assert metrics['env/search_count/correct_mean'] == 1.0
    assert metrics['env/search_count/wrong_mean'] == 1.0


def test_legacy_trace_keeps_v1_shape_without_fake_raw_sampled_tokens():
    batch = DataProto.from_dict(
        tensors={
            'prompts': torch.tensor([[1, 2]]),
            'responses': torch.tensor([[3, 4]]),
            'attention_mask': torch.ones(1, 4, dtype=torch.long),
            'executed_search_count': torch.zeros(1, dtype=torch.long),
            'sequence_em_scores': torch.ones(1),
            'sequence_posthoc_utilities': torch.ones(1),
        },
        non_tensors={
            'data_source': np.array(['nq'], dtype=object),
            'index': np.array([17], dtype=object),
            'extra_info': _object_array([{'split': 'train', 'index': 17}]),
            'reward_model': _object_array([{
                'ground_truth': {'target': ['Paris']},
            }]),
            'retrieval_events': _object_array([[]]),
            'generation_events': _object_array([[{
                'turn': 0,
                'text': '<answer>Paris</answer>',
                'token_count': 2,
                'clipped': False,
                'valid_action': True,
                'done': True,
                'executed_search': False,
            }]]),
        },
    )
    trainer = object.__new__(RayPPOTrainer)
    trainer.tokenizer = _Tokenizer()
    trainer.config = OmegaConf.create({
        'tool_protocol': 'legacy_xml',
        'data': {'max_prompt_length': 32},
        'algorithm': {'cost_reward_mode': 'linear', 'cost_lambda': 0.0},
        'max_turns': 4,
    })

    record = trainer._common_trace_records(batch)[0]

    assert record['schema_version'] == 1
    assert record['max_searches'] == 4
    assert 'raw_generations' not in record
    assert 'action_count' not in record
    assert 'max_action_budget' not in record


@pytest.mark.parametrize(('protocol', 'expected_schema_version'), [
    ('legacy_xml', 1),
    ('qwen35_native', 3),
])
def test_trainer_selects_trace_writer_schema_by_protocol(
        tmp_path, protocol, expected_schema_version):
    trainer = object.__new__(RayPPOTrainer)
    trainer.val_dataloader = [None]
    trainer.eval_trace_writer = None
    trainer.train_trace_writer = None
    trainer.config = OmegaConf.create({
        'tool_protocol': protocol,
        'data': {
            'val_batch_size': 1,
            'eval_group_size': 1,
        },
        'trainer': {
            'trace_output_dir': str(tmp_path),
            'trace_stage': 'schema-test',
            'trace_run_id': 'schema-test',
            'trace_checkpoint_digest': 'a' * 64,
            'experiment_name': 'schema-test',
            'val_only': True,
        },
    })

    trainer._init_trace_writer()

    assert trainer.eval_trace_writer.schema_version == expected_schema_version
    trainer.eval_trace_writer.close()


def test_trace_turns_respect_generation_boundaries_for_unclosed_tags():
    generation_events = [{
        'turn': 0,
        'text': '<think>unfinished',
        'valid_action': False,
        'executed_search': False,
    }, {
        'turn': 1,
        'text': '<search>France capital</search></think>',
        'valid_action': True,
        'executed_search': True,
    }]
    retrieval_events = [{
        'turn': 1,
        'query': 'France capital',
        'documents': [{'document_id': '7'}],
    }]

    turns = _event_aligned_trace_turns(generation_events, retrieval_events, 1)

    executed_turns = [turn for turn in turns if turn['retrieval_executed']]
    assert len(executed_turns) == 1
    assert executed_turns[0]['generation_turn'] == 1
    assert executed_turns[0]['retrieved_docs'] == [{'document_id': '7'}]


def test_trace_turns_use_the_post_truncation_visible_observation():
    generation_events = [{
        'turn': 0,
        'text': '<search>France capital</search>',
        'valid_action': True,
        'executed_search': True,
    }]
    retrieval_events = [{
        'turn': 0,
        'query': 'France capital',
        'documents': [{
            'document_id': '7'
        }],
        'observation': 'visible text followed by hidden evidence',
        'visible_observation': 'visible text',
    }]

    turns = _event_aligned_trace_turns(generation_events, retrieval_events, 1)
    executed_turn = next(turn for turn in turns
                         if turn['retrieval_executed'])

    assert executed_turn['observation'] == 'visible text'


def test_trace_turns_reject_generation_retrieval_turn_mismatch():
    generation_events = [{
        'turn': 0,
        'text': '<search>France capital</search>',
        'valid_action': True,
        'executed_search': True,
    }]
    retrieval_events = [{
        'turn': 1,
        'query': 'France capital',
        'documents': [],
    }]

    with pytest.raises(ValueError, match='turns are not aligned'):
        _event_aligned_trace_turns(generation_events, retrieval_events, 1)


def test_trace_turns_preserve_environment_action_nested_in_think():
    generation_events = [{
        'turn': 0,
        'text': '<think>try <search>France capital</search></think>',
        'valid_action': True,
        'executed_search': True,
    }]
    retrieval_events = [{
        'turn': 0,
        'query': 'France capital',
        'documents': [{'document_id': '7'}],
    }]

    turns = _event_aligned_trace_turns(generation_events, retrieval_events, 1)

    executed_turns = [turn for turn in turns if turn['retrieval_executed']]
    assert len(executed_turns) == 1
    assert executed_turns[0]['synthetic_environment_action'] is True
    assert executed_turns[0]['search_query'] == 'France capital'


def test_trace_turns_keep_forced_final_search_unexecuted():
    generation_events = [{
        'turn': 4,
        'text': '<search>France capital</search>',
        'valid_action': True,
        'executed_search': False,
    }]

    turns = _event_aligned_trace_turns(generation_events, [], 0)

    assert len(turns) == 1
    assert turns[0]['action'] == 'search'
    assert turns[0]['environment_action'] is True
    assert turns[0]['retrieval_executed'] is False


def test_trace_turns_bind_multiple_searches_without_using_trailing_search():
    generation_events = [{
        'turn': 0,
        'text': '<search>first query</search>',
        'valid_action': True,
        'executed_search': True,
    }, {
        'turn': 1,
        'text': '<search>second query</search>',
        'valid_action': True,
        'executed_search': True,
    }, {
        'turn': 2,
        'text': '<answer>done</answer><search>unused query</search>',
        'valid_action': True,
        'executed_search': False,
    }]
    retrieval_events = [{
        'turn': 0,
        'query': 'first query',
        'documents': [{'document_id': '1'}],
    }, {
        'turn': 1,
        'query': 'second query',
        'documents': [{'document_id': '2'}],
    }]

    turns = _event_aligned_trace_turns(generation_events, retrieval_events, 2)

    executed_turns = [turn for turn in turns if turn['retrieval_executed']]
    assert [turn['search_query'] for turn in executed_turns] == [
        'first query', 'second query'
    ]
    assert turns[-1]['search_query'] == 'unused query'
    assert turns[-1]['retrieval_executed'] is False


def test_trace_turns_use_native_normalized_actions_without_xml_reparse():
    generation_events = [{
        'turn': 0,
        'text': ('checking\n<tool_call><function=search><parameter=query>'
                 'France capital</parameter></function></tool_call>'),
        'action': 'search',
        'content': 'France capital',
        'parse_error': None,
        'reasoning_prefix': 'checking',
        'valid_action': True,
        'done': False,
        'executed_search': True,
    }, {
        'turn': 1,
        'text': 'Paris',
        'action': 'answer',
        'content': 'Paris',
        'parse_error': None,
        'reasoning_prefix': '',
        'valid_action': True,
        'done': True,
        'executed_search': False,
    }]
    retrieval_events = [{
        'turn': 0,
        'query': 'France capital',
        'documents': [{'document_id': '7'}],
        'visible_observation': 'Paris is the capital.',
    }]

    turns = _event_aligned_trace_turns(generation_events, retrieval_events, 1)

    assert [turn['action'] for turn in turns] == ['search', 'answer']
    assert turns[0]['think'] == 'checking'
    assert turns[0]['search_query'] == 'France capital'
    assert turns[0]['retrieval_executed'] is True
    assert turns[1]['answer'] == 'Paris'


def test_trace_turns_keep_native_parse_error_invalid():
    events = [{
        'turn': 0,
        'text': '{"name":"search"}',
        'action': None,
        'content': '',
        'parse_error': 'json_tool_call_not_supported',
        'reasoning_prefix': '',
        'valid_action': False,
        'done': False,
        'executed_search': False,
    }]

    turns = _event_aligned_trace_turns(events, [], 0)

    assert len(turns) == 1
    assert turns[0]['action'] == 'invalid'
    assert turns[0]['valid_action'] is False
