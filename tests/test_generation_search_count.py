from types import SimpleNamespace

import torch

from search_r1.llm_agent import generation
from search_r1.llm_agent.generation import LLMGenerationManager
from search_r1.llm_agent.tensor_helper import TensorConfig, TensorHelper
from verl import DataProto


def _manager():
    manager = object.__new__(LLMGenerationManager)
    manager.tokenizer = SimpleNamespace(pad_token_id=0)
    manager.tensor_fn = TensorHelper(
        TensorConfig(
            pad_token_id=0,
            max_prompt_length=8,
            max_obs_length=8,
            max_start_length=8,
        ))
    return manager


def test_execute_predictions_counts_only_active_retrievals():
    manager = _manager()
    seen_queries = []

    def batch_search(queries):
        seen_queries.extend(queries)
        manager._last_batch_search_metadata = [None] * len(queries)
        return ['result'] * len(queries)

    manager.batch_search = batch_search
    _, _, _, executed_search = manager.execute_predictions(
        ['<search>active</search>', '<answer>done</answer>', '<search>inactive</search>'],
        pad_token='<pad>',
        active_mask=torch.tensor([True, True, False]),
    )

    assert seen_queries == ['active']
    assert executed_search == [1, 0, 0]


def test_execute_predictions_preserves_raw_retrieval_metadata_out_of_band():
    manager = _manager()
    documents = [{
        'document_id': '42',
        'score': 3.5,
        'document': {'contents': 'Title\nPassage'},
    }]

    def batch_search(_):
        manager._last_batch_search_metadata = [documents]
        return ['Doc 1(Title: Title) Passage\n']

    manager.batch_search = batch_search
    manager.execute_predictions(
        ['<search>query</search>'],
        pad_token='<pad>',
        active_mask=torch.tensor([True]),
    )

    assert manager._last_execution_retrieval_events == [{
        'query': 'query',
        'documents': documents,
        'observation': 'Doc 1(Title: Title) Passage',
    }]


def test_explicitly_disabled_search_is_not_counted_or_executed():
    manager = _manager()
    manager.batch_search = lambda _: (_ for _ in ()).throw(AssertionError('retriever was called'))

    observations, dones, valid_actions, executed_search = manager.execute_predictions(
        ['<search>not executed</search>'],
        pad_token='<pad>',
        active_mask=torch.tensor([True]),
        do_search=False,
    )

    assert observations == ['\n\n<information></information>\n\n']
    assert dones == [0]
    assert valid_actions == [1]
    assert executed_search == [0]
    assert manager._last_execution_retrieval_events == [None]


def test_search_count_is_a_reorderable_batch_tensor():
    manager = _manager()
    output = manager._compose_final_output(
        left_side={'input_ids': torch.tensor([[1], [2]])},
        right_side={
            'responses': torch.tensor([[3, 0], [4, 5]]),
            'responses_with_info_mask': torch.tensor([[3, 0], [4, 5]]),
        },
        meta_info={},
        action_count=torch.tensor([1, 2]),
        executed_search_count=torch.tensor([0, 2]),
    )

    output.reorder(torch.tensor([1, 0]))
    assert output.batch['action_count'].tolist() == [2, 1]
    assert output.batch['executed_search_count'].tolist() == [2, 0]


def test_retrieval_metadata_stays_aligned_after_batch_reorder():
    manager = _manager()
    output = manager._compose_final_output(
        left_side={'input_ids': torch.tensor([[1], [2]])},
        right_side={
            'responses': torch.tensor([[3], [4]]),
            'responses_with_info_mask': torch.tensor([[3], [4]]),
        },
        meta_info={},
        action_count=torch.tensor([1, 1]),
        executed_search_count=torch.tensor([1, 0]),
        retrieval_events=[
            [{'turn': 0, 'query': 'first', 'documents': []}],
            [],
        ],
    )

    output.reorder(torch.tensor([1, 0]))
    assert output.non_tensor_batch['retrieval_events'].tolist() == [
        [],
        [{'turn': 0, 'query': 'first', 'documents': []}],
    ]


def test_retrieval_trace_records_post_truncation_visible_observation():

    class WordTokenizer:

        pad_token_id = 0
        pad_token = '<pad>'

        def __init__(self):
            self._token_ids = {}
            self._tokens = {}

        def _id(self, token):
            if token not in self._token_ids:
                token_id = len(self._token_ids) + 100
                self._token_ids[token] = token_id
                self._tokens[token_id] = token
            return self._token_ids[token]

        def __call__(self, values, **_kwargs):
            rows = [[self._id(token) for token in value.split()]
                    for value in values]
            width = max(map(len, rows))
            return {
                'input_ids':
                torch.tensor([
                    row + [self.pad_token_id] * (width - len(row))
                    for row in rows
                ])
            }

        def batch_decode(self, rows, skip_special_tokens=True):
            assert skip_special_tokens
            return [
                ' '.join(self._tokens[int(token)] for token in row
                         if int(token) != self.pad_token_id) for row in rows
            ]

    manager = object.__new__(LLMGenerationManager)
    manager.tokenizer = WordTokenizer()
    manager.config = SimpleNamespace(num_gpus=1,
                                     max_turns=1,
                                     max_start_length=8,
                                     max_prompt_length=512,
                                     max_response_length=8,
                                     max_obs_length=500)
    manager.tensor_fn = TensorHelper(
        TensorConfig(pad_token_id=0,
                     max_prompt_length=512,
                     max_obs_length=500,
                     max_start_length=8))

    class WorkerGroup:

        def __init__(self):
            self.calls = 0

        def generate_sequences(self, batch):
            self.calls += 1
            token = 11 if self.calls == 1 else 12
            return DataProto.from_dict(
                {'responses': torch.full((len(batch), 1), token)},
                meta_info=batch.meta_info.copy())

    manager.actor_rollout_wg = WorkerGroup()
    manager._postprocess_responses = lambda responses: (responses, [
        '<search>topic</search>'
        if int(responses[0, 0]) == 11 else '<answer>done</answer>'
    ])
    full_observation = ' '.join(['visible'] * 500 + ['HIDDEN'])
    documents = [{
        'document_id': '42',
        'document': {
            'contents': 'Visible title\nEvidence'
        },
    }]

    def batch_search(queries):
        manager._last_batch_search_metadata = [documents for _ in queries]
        return [full_observation for _ in queries]

    manager.batch_search = batch_search
    gen_batch = DataProto.from_dict(
        {
            'input_ids': torch.tensor([[1, 2]]),
            'attention_mask': torch.ones(1, 2, dtype=torch.long),
            'position_ids': torch.tensor([[0, 1]]),
        },
        meta_info={
            'do_sample': True,
            'validate': True
        })

    output = manager.run_llm_loop(gen_batch,
                                  gen_batch.batch['input_ids'].clone())
    event = output.non_tensor_batch['retrieval_events'][0][0]

    assert 'HIDDEN' in event['observation']
    assert 'HIDDEN' not in event['visible_observation']
    assert len(event['visible_observation'].split()) == 500


def test_legacy_retokenized_responses_and_observations_follow_rollout_device():

    class Tokenizer:

        pad_token_id = 0

        def __call__(self, _values, **_kwargs):
            return {'input_ids': torch.tensor([[4, 5]])}

        def batch_decode(self, rows, skip_special_tokens=True):
            assert skip_special_tokens
            return ['<answer>done</answer>'] * len(rows)

    manager = _manager()
    manager.tokenizer = Tokenizer()
    manager.config = SimpleNamespace(
        no_think_rl=False,
        tool_protocol='legacy_xml',
        max_obs_length=1,
    )
    generated = torch.empty((1, 2), dtype=torch.long, device='meta')

    response_ids, _ = manager._postprocess_responses(generated)
    observation_ids, _ = manager._process_next_obs(
        ['retrieved document'], device=generated.device)

    assert response_ids.device == generated.device
    assert observation_ids.device == generated.device
    assert observation_ids.shape == (1, 1)
    combined = manager.tensor_fn.concatenate_with_padding([
        generated, response_ids, observation_ids
    ])
    assert combined.device == generated.device


def test_odd_active_validation_batch_keeps_deterministic_sampling_metadata():
    manager = _manager()
    manager.tokenizer.pad_token = '<pad>'
    manager.config = SimpleNamespace(
        num_gpus=2,
        max_turns=1,
        max_start_length=8,
        max_prompt_length=8,
        max_obs_length=8,
    )
    captured = {}

    class WorkerGroup:

        def generate_sequences(self, batch):
            captured['batch_size'] = len(batch)
            captured['meta_info'] = batch.meta_info.copy()
            responses = torch.arange(1, len(batch) + 1).unsqueeze(1)
            return DataProto.from_dict({'responses': responses})

    manager.actor_rollout_wg = WorkerGroup()
    manager._postprocess_responses = lambda responses: (
        responses,
        ['<answer>x</answer>'] * len(responses),
    )
    manager._process_next_obs = lambda observations, device=None: (
        torch.empty((len(observations), 0), dtype=torch.long, device=device),
        [''] * len(observations),
    )
    manager.execute_predictions = lambda predictions, pad_token, active_mask, do_search=False: (
        [''] * len(predictions),
        [1] * len(predictions),
        [1] * len(predictions),
        [0] * len(predictions),
    )
    manager._last_execution_retrieval_events = [None, None, None]
    gen_batch = DataProto.from_dict(
        {
            'input_ids': torch.tensor([[1, 2], [3, 4], [5, 6]]),
            'attention_mask': torch.ones(3, 2, dtype=torch.long),
            'position_ids': torch.tensor([[0, 1], [0, 1], [0, 1]]),
        },
        meta_info={'do_sample': False, 'validate': True},
    )

    output = manager.run_llm_loop(gen_batch, gen_batch.batch['input_ids'].clone())

    assert captured == {
        'batch_size': 4,
        'meta_info': {'do_sample': False, 'validate': True},
    }
    assert output.batch['responses'].squeeze(1).tolist() == [1, 2, 3]


def test_retriever_request_has_a_bounded_timeout(monkeypatch):
    manager = _manager()
    manager.config = SimpleNamespace(search_url='http://127.0.0.1:8000/retrieve', topk=3)
    captured = {}

    class Response:

        def raise_for_status(self):
            captured['status_checked'] = True

        def json(self):
            return {'result': []}

    def post(url, json, timeout):
        captured.update(url=url, payload=json, timeout=timeout)
        return Response()

    monkeypatch.setattr(generation.requests, 'post', post)

    assert manager._batch_search(['query']) == {'result': []}
    assert captured == {
        'url': 'http://127.0.0.1:8000/retrieve',
        'payload': {'queries': ['query'], 'topk': 3, 'return_scores': True},
        'timeout': 60,
        'status_checked': True,
    }
