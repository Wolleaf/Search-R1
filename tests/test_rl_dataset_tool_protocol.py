import copy

import numpy as np
import pandas as pd
import pytest
import torch

from search_r1.llm_agent.tool_protocol import (LEGACY_XML, QWEN35_NATIVE,
                                               ProtocolError,
                                               Qwen35Conversation,
                                               qwen35_messages, qwen35_tools)
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn


class RecordingTokenizer:
    pad_token_id = 0

    def __init__(self, chat_template="test-template"):
        self.chat_template = chat_template
        self.template_calls = []
        self.rendered_prompts = []

    @staticmethod
    def _messages(chat):
        return chat.tolist() if hasattr(chat, "tolist") else list(chat)

    @staticmethod
    def _token_ids(text):
        return [(ord(character) % 251) + 1 for character in text]

    def apply_chat_template(self, chat, **kwargs):
        messages = self._messages(chat)
        self.template_calls.append({
            "messages": copy.deepcopy(messages),
            "kwargs": copy.deepcopy(kwargs),
        })
        prefix = "native" if "tools" in kwargs else "legacy"
        rendered = prefix + "|" + "|".join(message["content"]
                                           for message in messages)
        self.rendered_prompts.append(rendered)
        return rendered

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        del return_tensors, add_special_tokens
        input_ids = torch.tensor([self._token_ids(text)], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }


def _dataset(monkeypatch, tokenizer, chats, **kwargs):
    monkeypatch.setattr(RLHFDataset, "_download", lambda self: None)

    def read_rows(dataset):
        dataset.dataframe = pd.DataFrame({
            "prompt":
            chats,
            "extra_info": [{
                "index": index
            } for index in range(len(chats))],
        })

    monkeypatch.setattr(RLHFDataset, "_read_files_and_tokenize", read_rows)
    return RLHFDataset(
        parquet_files=["unused.parquet"],
        tokenizer=tokenizer,
        max_prompt_length=1024,
        **kwargs,
    )


def _active_token_ids(item):
    return item["input_ids"][item["attention_mask"].bool()].tolist()


def test_default_legacy_rendering_is_unchanged(monkeypatch):
    chat = np.array([{
        "role": "user",
        "content": "Question: Where is Paris?",
    }],
                    dtype=object)
    tokenizer = RecordingTokenizer()
    dataset = _dataset(monkeypatch, tokenizer, [chat])

    item = dataset[0]

    assert dataset.tool_protocol == LEGACY_XML
    assert tokenizer.template_calls == [{
        "messages": chat.tolist(),
        "kwargs": {
            "add_generation_prompt": True,
            "tokenize": False,
        },
    }]
    assert _active_token_ids(item) == tokenizer._token_ids(
        "legacy|Question: Where is Paris?")
    assert "raw_prompt" not in item


def test_legacy_return_raw_chat_still_uses_original_messages(monkeypatch):
    chat = np.array([{
        "role": "user",
        "content": "Question: Where is Paris?",
    }],
                    dtype=object)
    dataset = _dataset(
        monkeypatch,
        RecordingTokenizer(),
        [chat],
        return_raw_chat=True,
    )

    assert dataset[0]["raw_prompt"] == chat.tolist()


def test_native_uses_qwen_template_and_always_returns_raw_prompt(monkeypatch):
    expected_chat = qwen35_messages("Where is Paris?")
    chat = np.array(copy.deepcopy(expected_chat), dtype=object)
    tokenizer = RecordingTokenizer(chat_template=None)
    dataset = _dataset(
        monkeypatch,
        tokenizer,
        [chat],
        tool_protocol=QWEN35_NATIVE,
        return_raw_chat=False,
    )

    item = dataset[0]

    assert item["raw_prompt"] == expected_chat
    assert tokenizer.template_calls[0] == {
        "messages": chat.tolist(),
        "kwargs": {
            "tools": qwen35_tools(),
            "enable_thinking": True,
            "add_generation_prompt": True,
            "tokenize": False,
        },
    }
    assert _active_token_ids(item) == tokenizer._token_ids(
        tokenizer.rendered_prompts[0])

    # The rollout manager validates this exact unpadded token prefix.
    Qwen35Conversation(tokenizer, item["raw_prompt"], _active_token_ids(item))
    chat[0]["content"] = "mutated after loading"
    assert item["raw_prompt"] == expected_chat


def test_native_raw_prompts_remain_batch_aligned(monkeypatch):
    chats = [
        np.array(qwen35_messages(f"Item {index}?"), dtype=object)
        for index in range(2)
    ]
    dataset = _dataset(
        monkeypatch,
        RecordingTokenizer(),
        chats,
        tool_protocol=QWEN35_NATIVE,
    )

    batch = collate_fn([dataset[0], dataset[1]])

    assert len(batch["raw_prompt"]) == 2
    assert batch["raw_prompt"][0].tolist() == chats[0].tolist()
    assert batch["raw_prompt"][1].tolist() == chats[1].tolist()


def test_dataset_rejects_unknown_tool_protocol_before_loading(monkeypatch):
    monkeypatch.setattr(
        RLHFDataset,
        "_download",
        lambda self: pytest.fail("invalid protocol reached dataset loading"),
    )

    with pytest.raises(ProtocolError, match="tool_protocol must be one of"):
        RLHFDataset(
            parquet_files=["unused.parquet"],
            tokenizer=RecordingTokenizer(),
            tool_protocol="native",
        )


def test_native_rejects_noncanonical_message_shape(monkeypatch):
    chat = np.array([{
        "role": "system",
        "content": "custom policy",
    }, {
        "role": "user",
        "content": qwen35_messages("Question?")[0]["content"],
    }],
                    dtype=object)
    dataset = _dataset(
        monkeypatch,
        RecordingTokenizer(),
        [chat],
        tool_protocol=QWEN35_NATIVE,
    )

    with pytest.raises(ProtocolError, match="exactly one user message"):
        dataset[0]
