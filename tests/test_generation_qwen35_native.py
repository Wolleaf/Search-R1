import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from search_r1.llm_agent.generation import LLMGenerationManager
from search_r1.llm_agent.tool_protocol import (QWEN35_RETRY_PROMPT,
                                               qwen35_messages, qwen35_tools)
from verl import DataProto


class _CharTokenizer:

    pad_token_id = 0
    eos_token_id = 1
    pad_token = "<pad>"
    _assistant_end = "</assistant>"

    def __init__(self):
        self._char_to_id = {}
        self._id_to_char = {}

    def _encode(self, text):
        output = []
        index = 0
        while index < len(text):
            if text.startswith(self._assistant_end, index):
                output.append(self.eos_token_id)
                index += len(self._assistant_end)
                continue
            char = text[index]
            if char not in self._char_to_id:
                token_id = len(self._char_to_id) + 10
                self._char_to_id[char] = token_id
                self._id_to_char[token_id] = char
            output.append(self._char_to_id[char])
            index += 1
        return output

    def __call__(self, values, return_tensors=None, padding=None, **_kwargs):
        single = isinstance(values, str)
        values = [values] if single else list(values)
        rows = [self._encode(value) for value in values]
        if return_tensors == "pt":
            width = max((len(row) for row in rows), default=0)
            input_ids = torch.full((len(rows), width), self.pad_token_id)
            for index, row in enumerate(rows):
                if row:
                    input_ids[index, :len(row)] = torch.tensor(row)
            return {"input_ids": input_ids}
        return {"input_ids": rows[0] if single else rows}

    def decode(self, token_ids, skip_special_tokens=False, **_kwargs):
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        output = []
        for token_id in token_ids:
            token_id = int(token_id)
            if token_id == self.pad_token_id:
                continue
            if token_id == self.eos_token_id:
                if not skip_special_tokens:
                    output.append(self._assistant_end)
                continue
            output.append(self._id_to_char[token_id])
        return "".join(output)

    def batch_decode(self, rows, skip_special_tokens=False, **kwargs):
        return [
            self.decode(row, skip_special_tokens=skip_special_tokens, **kwargs)
            for row in rows
        ]

    def apply_chat_template(self, messages, tools, enable_thinking,
                            add_generation_prompt, tokenize):
        assert enable_thinking is False
        assert tokenize is False
        rendered = "<tools>" + json.dumps(
            tools, sort_keys=True, separators=(",", ":")) + "</tools>"
        assistant_prefix = "<assistant><think></think>"
        for message in messages:
            if message["role"] == "system":
                rendered += f'<system>{message["content"]}</system>'
            elif message["role"] == "user":
                rendered += f'<user>{message["content"]}</user>'
            elif message["role"] == "assistant":
                rendered += assistant_prefix + message[
                    "content"] + "</assistant>"
            elif message["role"] == "tool":
                rendered += f'<tool>{message["content"]}</tool>'
            else:
                raise AssertionError(message["role"])
        if add_generation_prompt:
            rendered += assistant_prefix
        return rendered


class _TrimNoncanonicalCharTokenizer(_CharTokenizer):

    noncanonical_token_id = 2

    def decode(self, token_ids, skip_special_tokens=False, **_kwargs):
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        output = []
        for token_id in token_ids:
            token_id = int(token_id)
            if token_id == self.pad_token_id:
                continue
            if token_id == self.eos_token_id:
                if not skip_special_tokens:
                    output.append(self._assistant_end)
                continue
            if token_id == self.noncanonical_token_id:
                output.append("France")
            else:
                output.append(self._id_to_char[token_id])
        return "".join(output)

    def apply_chat_template(self, messages, tools, enable_thinking,
                            add_generation_prompt, tokenize):
        assert enable_thinking is False
        assert tokenize is False
        rendered = "<tools>" + json.dumps(
            tools, sort_keys=True, separators=(",", ":")) + "</tools>"
        assistant_prefix = "<assistant><think></think>"
        for message in messages:
            content = message["content"].strip()
            if message["role"] == "system":
                rendered += f"<system>{content}</system>"
            elif message["role"] == "user":
                rendered += f"<user>{content}</user>"
            elif message["role"] == "assistant":
                rendered += assistant_prefix + content + "</assistant>"
            elif message["role"] == "tool":
                rendered += f"<tool>{content}</tool>"
            else:
                raise AssertionError(message["role"])
        if add_generation_prompt:
            rendered += assistant_prefix
        return rendered


def _config(**overrides):
    values = {
        "max_turns": 1,
        "max_start_length": 1024,
        "max_prompt_length": 2048,
        "max_response_length": 512,
        "max_obs_length": 256,
        "num_gpus": 1,
        "no_think_rl": False,
        "search_url": "unused",
        "topk": 3,
        "tool_protocol": "qwen35_native",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _generation_batch(tokenizer, raw_messages):
    rendered = [
        tokenizer.apply_chat_template(
            messages,
            tools=qwen35_tools(),
            enable_thinking=False,
            add_generation_prompt=True,
            tokenize=False,
        ) for messages in raw_messages
    ]
    rows = [
        tokenizer(text, add_special_tokens=False)["input_ids"]
        for text in rendered
    ]
    width = max(map(len, rows))
    input_ids = torch.full((len(rows), width), tokenizer.pad_token_id)
    attention_mask = torch.zeros_like(input_ids)
    for index, row in enumerate(rows):
        input_ids[index, -len(row):] = torch.tensor(row)
        attention_mask[index, -len(row):] = 1
    position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask
    return DataProto.from_dict({
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }), rows


def _response_tensor(tokenizer, texts):
    rows = [
        tokenizer(text, add_special_tokens=False)["input_ids"] +
        [tokenizer.eos_token_id] for text in texts
    ]
    width = max(map(len, rows)) + 2
    output = torch.full((len(rows), width), tokenizer.pad_token_id)
    for index, row in enumerate(rows):
        output[index, :len(row)] = torch.tensor(row)
    return output


class _ScriptedWorker:

    def __init__(self, tokenizer, turns):
        self.tokenizer = tokenizer
        self.turns = list(turns)
        self.prompts = []

    def generate_sequences(self, batch):
        self.prompts.append(batch.batch["input_ids"].clone())
        texts = self.turns.pop(0)
        assert len(texts) == len(batch)
        return DataProto.from_dict(
            {"responses": _response_tensor(self.tokenizer, texts)},
            meta_info=batch.meta_info.copy(),
        )


class _RawScriptedWorker:

    def __init__(self, tokenizer, turns):
        self.tokenizer = tokenizer
        self.turns = list(turns)
        self.prompts = []

    def generate_sequences(self, batch):
        self.prompts.append(batch.batch["input_ids"].clone())
        rows = self.turns.pop(0)
        assert len(rows) == len(batch)
        width = max(map(len, rows))
        responses = torch.full((len(rows), width),
                               self.tokenizer.pad_token_id)
        for index, row in enumerate(rows):
            responses[index, :len(row)] = torch.tensor(row)
        return DataProto.from_dict(
            {"responses": responses}, meta_info=batch.meta_info.copy())


def _native_manager(tokenizer, worker, **config_overrides):
    manager = LLMGenerationManager(
        tokenizer=tokenizer,
        actor_rollout_wg=worker,
        config=_config(**config_overrides),
    )
    documents = [{
        "document_id": "7",
        "document": {
            "contents": "France\nParis is the capital."
        },
    }]

    def batch_search(queries):
        assert queries == ["capital of France"]
        manager._last_batch_search_metadata = [documents]
        return ["Doc 1 says Paris is the capital of France."]

    manager.batch_search = batch_search
    return manager


def test_native_postprocess_preserves_sampled_tokens_without_action_rewrite():
    tokenizer = _CharTokenizer()
    manager = LLMGenerationManager(
        tokenizer=tokenizer,
        actor_rollout_wg=None,
        config=_config(),
    )
    raw = "  <tool_call>malformed but sampled</tool_call>  "
    generated = _response_tensor(tokenizer, [raw])

    response_ids, response_text = manager._postprocess_responses(generated)

    expected = (tokenizer(raw, add_special_tokens=False)["input_ids"] +
                [tokenizer.eos_token_id])
    assert response_ids[0, :len(expected)].tolist() == expected
    assert response_ids.shape[1] == len(expected)
    assert response_text == [raw]


def test_native_loop_keeps_tokens_masks_actions_and_reorder_alignment():
    tokenizer = _CharTokenizer()
    search = ("<tool_call>\n<function=search>\n<parameter=query>\n"
              "capital of France\n</parameter>\n</function>\n</tool_call>")
    worker = _ScriptedWorker(tokenizer, [[search, "Lyon"], ["Paris"]])
    manager = _native_manager(tokenizer, worker)
    raw_messages = [
        qwen35_messages("Capital of France?"),
        qwen35_messages("Capital of Gaul?")
    ]
    gen_batch, prompt_rows = _generation_batch(tokenizer, raw_messages)

    output = manager.run_llm_loop(
        gen_batch,
        gen_batch.batch["input_ids"].clone(),
        raw_messages=raw_messages,
    )

    search_ids = (tokenizer(search, add_special_tokens=False)["input_ids"] +
                  [tokenizer.eos_token_id])
    answer_ids = (tokenizer("Paris", add_special_tokens=False)["input_ids"] +
                  [tokenizer.eos_token_id])
    second_prompt = worker.prompts[1][0].tolist()
    suffix_ids = second_prompt[len(prompt_rows[0]) + len(search_ids):]
    assert second_prompt == prompt_rows[0] + search_ids + suffix_ids
    assert "<tool>Doc 1 says Paris" in tokenizer.decode(suffix_ids)

    response_width = output.batch["responses"].shape[-1]
    row = output.batch["responses"][0]
    expected_response = search_ids + suffix_ids + answer_ids
    assert row[:len(expected_response)].tolist() == expected_response
    response_loss_mask = output.batch["info_mask"][0, -response_width:]
    expected_mask = ([1] * len(search_ids) + [0] * len(suffix_ids) +
                     [1] * len(answer_ids))
    assert response_loss_mask[:len(expected_mask)].tolist() == expected_mask
    assert response_loss_mask[len(search_ids) - 1].item() == 1
    assert response_loss_mask[len(expected_response) - 1].item() == 1

    assert output.batch["executed_search_count"].tolist() == [1, 0]
    assert output.non_tensor_batch["final_answer"].tolist() == [
        "Paris", "Lyon"
    ]
    assert [
        event["action"]
        for event in output.non_tensor_batch["generation_events"][0]
    ] == ["search", "answer"]
    assert output.non_tensor_batch["generation_events"][0][0][
        "content"] == "capital of France"
    assert output.non_tensor_batch["generation_events"][0][0][
        "parse_error"] is None
    assert [
        action["action"]
        for action in output.non_tensor_batch["parsed_actions"][0]
    ] == ["search", "answer"]
    assert output.non_tensor_batch["retrieval_events"][0][0][
        "visible_observation"] == "Doc 1 says Paris is the capital of France."

    output.reorder(torch.tensor([1, 0]))
    assert output.non_tensor_batch["final_answer"].tolist() == [
        "Lyon", "Paris"
    ]
    assert output.batch["executed_search_count"].tolist() == [0, 1]
    assert output.non_tensor_batch["parsed_actions"][1][0][
        "content"] == "capital of France"


@pytest.mark.parametrize("sampled_eos", [True, False])
def test_native_loop_preserves_trimmed_noncanonical_sample_and_mask(
        sampled_eos):
    tokenizer = _TrimNoncanonicalCharTokenizer()
    search = ("  <tool_call><function=search><parameter=query>"
              "capital of France</parameter></function></tool_call>  ")
    before, marker, after = search.partition("France")
    assert marker
    search_ids = (
        tokenizer(before, add_special_tokens=False)["input_ids"] +
        [tokenizer.noncanonical_token_id] +
        tokenizer(after, add_special_tokens=False)["input_ids"])
    if sampled_eos:
        search_ids.append(tokenizer.eos_token_id)
    answer_ids = (tokenizer("Paris", add_special_tokens=False)["input_ids"] +
                  [tokenizer.eos_token_id])
    worker = _RawScriptedWorker(tokenizer, [[search_ids], [answer_ids]])
    manager = _native_manager(tokenizer, worker)
    raw_messages = [qwen35_messages("Capital of France?")]
    gen_batch, prompt_rows = _generation_batch(tokenizer, raw_messages)

    output = manager.run_llm_loop(
        gen_batch,
        gen_batch.batch["input_ids"].clone(),
        raw_messages=raw_messages,
    )

    second_prompt = worker.prompts[1][0].tolist()
    suffix_ids = second_prompt[len(prompt_rows[0]) + len(search_ids):]
    expected = search_ids + suffix_ids + answer_ids
    response_row = output.batch["responses"][0]
    assert response_row[:len(expected)].tolist() == expected
    loss_mask = output.batch["info_mask"][
        0, -output.batch["responses"].shape[-1]:]
    assert loss_mask[:len(search_ids)].tolist() == [1] * len(search_ids)
    assert loss_mask[len(search_ids):len(search_ids) +
                     len(suffix_ids)].tolist() == [0] * len(suffix_ids)
    assert loss_mask[len(search_ids) + len(suffix_ids):len(expected)].tolist(
    ) == [1] * len(answer_ids)
    if sampled_eos:
        assert suffix_ids[0] != tokenizer.eos_token_id
    else:
        assert suffix_ids[0] == tokenizer.eos_token_id


def test_native_invalid_action_uses_native_retry_and_preserves_mask():
    tokenizer = _CharTokenizer()
    invalid = ("<tool_call><function=search><parameter=query>x</parameter>"
               "trailing</function></tool_call>")
    worker = _ScriptedWorker(tokenizer, [[invalid], ["Paris"]])
    manager = _native_manager(tokenizer, worker)
    raw_messages = [qwen35_messages("Capital of France?")]
    gen_batch, prompt_rows = _generation_batch(tokenizer, raw_messages)

    output = manager.run_llm_loop(
        gen_batch,
        gen_batch.batch["input_ids"].clone(),
        raw_messages=raw_messages,
    )

    invalid_ids = (tokenizer(invalid, add_special_tokens=False)["input_ids"] +
                   [tokenizer.eos_token_id])
    retry_prompt = tokenizer.decode(worker.prompts[1][0])
    assert QWEN35_RETRY_PROMPT in retry_prompt
    assert "<information>" not in retry_prompt
    events = output.non_tensor_batch["generation_events"][0]
    assert events[0]["action"] is None
    assert events[0]["parse_error"] == "malformed_tool_call"
    assert events[1]["action"] == "answer"
    assert output.non_tensor_batch["final_answer"].tolist() == ["Paris"]

    suffix_length = len(worker.prompts[1][0]) - len(
        prompt_rows[0]) - len(invalid_ids)
    response_width = output.batch["responses"].shape[-1]
    loss_mask = output.batch["info_mask"][0, -response_width:]
    assert loss_mask[:len(invalid_ids)].tolist() == [1] * len(invalid_ids)
    assert loss_mask[len(invalid_ids) - 1].item() == 1
    assert loss_mask[len(invalid_ids):len(invalid_ids) +
                     suffix_length].sum() == 0


def test_native_conversations_copy_raw_messages_and_require_alignment():
    tokenizer = _CharTokenizer()
    worker = _ScriptedWorker(tokenizer, [])
    manager = _native_manager(tokenizer, worker)
    raw_messages = [qwen35_messages("Original?")]
    gen_batch, prompt_rows = _generation_batch(tokenizer, raw_messages)

    conversations = manager._prepare_native_conversations(
        gen_batch, raw_messages)
    raw_messages[0][1]["content"] = "Question: Mutated?\n"

    assert conversations[0].messages[1]["content"] == "Question: Original?\n"
    try:
        manager._prepare_native_conversations(gen_batch, [])
    except ValueError as error:
        assert "batch-aligned" in str(error)
    else:
        raise AssertionError("misaligned raw messages must fail")

    short_manager = _native_manager(tokenizer,
                                    worker,
                                    max_start_length=len(prompt_rows[0]) - 1)
    try:
        short_manager._prepare_native_conversations(gen_batch, raw_messages)
    except ValueError as error:
        assert "max_start_length" in str(error)
    else:
        raise AssertionError("oversized native prompts must fail")


def test_native_capacity_counts_only_the_policy_right_side():
    tokenizer = _CharTokenizer()
    exact_capacity = 4 * (500 + 384) + 500

    LLMGenerationManager(
        tokenizer=tokenizer,
        actor_rollout_wg=None,
        config=_config(
            max_turns=4,
            max_start_length=10_000,
            max_prompt_length=exact_capacity,
            max_response_length=500,
            max_obs_length=384,
        ),
    )

    with pytest.raises(ValueError, match=(
            "right-side capacity requires 4036 tokens.*4035")):
        LLMGenerationManager(
            tokenizer=tokenizer,
            actor_rollout_wg=None,
            config=_config(
                max_turns=4,
                max_start_length=1,
                max_prompt_length=exact_capacity - 1,
                max_response_length=500,
                max_obs_length=384,
            ),
        )


def test_native_rolling_context_keeps_initial_left_and_policy_right_side():
    tokenizer = _CharTokenizer()
    manager = LLMGenerationManager(
        tokenizer=tokenizer,
        actor_rollout_wg=None,
        config=_config(
            max_turns=1,
            max_start_length=4,
            max_prompt_length=8,
            max_response_length=3,
            max_obs_length=2,
        ),
    )
    rollings = DataProto.from_dict({
        "input_ids": torch.tensor([[10, 11, 12, 13]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "position_ids": torch.tensor([[0, 1, 2, 3]]),
    })
    response = torch.tensor([[20, 21, tokenizer.eos_token_id]])
    observation = torch.tensor([[22, 23]])

    next_rollings = manager._update_rolling_state(
        rollings, response, observation)
    right_side = manager._update_right_side(
        {
            "responses": torch.empty((1, 0), dtype=torch.long),
            "responses_with_info_mask": torch.empty((1, 0), dtype=torch.long),
        },
        response,
        observation,
    )

    assert next_rollings.batch["input_ids"].tolist() == [[
        10, 11, 12, 13, 20, 21, tokenizer.eos_token_id, 22, 23
    ]]
    assert right_side["responses"].tolist() == [[
        20, 21, tokenizer.eos_token_id, 22, 23
    ]]

    with pytest.raises(RuntimeError, match="policy right side exceeded"):
        manager._update_right_side(
            {
                "responses": torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17]]),
                "responses_with_info_mask": torch.tensor(
                    [[10, 11, 12, 13, 14, 15, 16, 17]]),
            },
            torch.tensor([[18]]),
        )


def test_legacy_rolling_context_still_uses_prompt_length_cap():
    tokenizer = _CharTokenizer()
    manager = LLMGenerationManager(
        tokenizer=tokenizer,
        actor_rollout_wg=None,
        config=_config(
            tool_protocol="legacy_xml",
            max_start_length=4,
            max_prompt_length=8,
            max_response_length=3,
            max_obs_length=2,
        ),
    )
    rollings = DataProto.from_dict({
        "input_ids": torch.tensor([[10, 11, 12, 13]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "position_ids": torch.tensor([[0, 1, 2, 3]]),
    })

    next_rollings = manager._update_rolling_state(
        rollings,
        torch.tensor([[20, 21, tokenizer.eos_token_id]]),
        torch.tensor([[22, 23]]),
    )

    assert next_rollings.batch["input_ids"].tolist() == [[
        11, 12, 13, 20, 21, tokenizer.eos_token_id, 22, 23
    ]]


def test_native_rolling_context_keeps_full_validated_capacity():
    tokenizer = _CharTokenizer()
    max_start_length = 1024
    max_prompt_length = 4 * (500 + 384) + 500
    manager = LLMGenerationManager(
        tokenizer=tokenizer,
        actor_rollout_wg=None,
        config=_config(
            max_turns=4,
            max_start_length=max_start_length,
            max_prompt_length=max_prompt_length,
            max_response_length=500,
            max_obs_length=384,
        ),
    )
    initial = torch.arange(2, max_start_length + 2).unsqueeze(0)
    rollings = DataProto.from_dict({
        "input_ids": initial,
        "attention_mask": torch.ones_like(initial),
        "position_ids": torch.arange(max_start_length).unsqueeze(0),
    })
    expected = initial

    for turn in range(4):
        response = torch.full((1, 500), 10 + turn * 2)
        observation = torch.full((1, 384), 11 + turn * 2)
        rollings = manager._update_rolling_state(
            rollings, response, observation)
        expected = torch.cat((expected, response, observation), dim=1)

    final_response = torch.full((1, 500), 30)
    rollings = manager._update_rolling_state(
        rollings,
        final_response,
        torch.empty((1, 0), dtype=torch.long),
    )
    expected = torch.cat((expected, final_response), dim=1)

    assert expected.shape[1] == max_start_length + max_prompt_length
    assert torch.equal(rollings.batch["input_ids"], expected)
