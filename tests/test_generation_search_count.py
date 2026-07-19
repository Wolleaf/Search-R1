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
        return ['result'] * len(queries)

    manager.batch_search = batch_search
    _, _, _, executed_search = manager.execute_predictions(
        ['<search>active</search>', '<answer>done</answer>', '<search>inactive</search>'],
        pad_token='<pad>',
        active_mask=torch.tensor([True, True, False]),
    )

    assert seen_queries == ['active']
    assert executed_search == [1, 0, 0]


def test_forced_final_search_is_not_counted_or_executed():
    manager = _manager()
    manager.batch_search = lambda _: (_ for _ in ()).throw(AssertionError('retriever was called'))

    observations, _, _, executed_search = manager.execute_predictions(
        ['<search>not executed</search>'],
        pad_token='<pad>',
        active_mask=torch.tensor([True]),
        do_search=False,
    )

    assert observations == ['\n\n<information></information>\n\n']
    assert executed_search == [0]


def test_search_count_is_a_reorderable_batch_tensor():
    manager = _manager()
    output = manager._compose_final_output(
        left_side={'input_ids': torch.tensor([[1], [2]])},
        right_side={
            'responses': torch.tensor([[3, 0], [4, 5]]),
            'responses_with_info_mask': torch.tensor([[3, 0], [4, 5]]),
        },
        meta_info={},
        executed_search_count=torch.tensor([0, 2]),
    )

    output.reorder(torch.tensor([1, 0]))
    assert output.batch['executed_search_count'].tolist() == [2, 0]


def test_odd_active_validation_batch_keeps_deterministic_sampling_metadata():
    manager = _manager()
    manager.tokenizer.pad_token = '<pad>'
    manager.config = SimpleNamespace(
        num_gpus=2,
        max_turns=0,
        max_start_length=8,
        max_prompt_length=8,
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
    manager.execute_predictions = lambda predictions, pad_token, active_mask, do_search=False: (
        [''] * len(predictions),
        [1] * len(predictions),
        [1] * len(predictions),
        [0] * len(predictions),
    )
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
