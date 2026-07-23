import json

import pytest

from search_r1.llm_agent.tool_protocol import (
    LEGACY_XML,
    QWEN35_NATIVE,
    ParsedAction,
    ProtocolError,
    Qwen35Conversation,
    parse_action,
    qwen35_messages,
    qwen35_tool_schema_sha256,
    qwen35_tools,
    render_qwen35_prompt,
    validate_presence_penalty,
    validate_qwen35_messages,
)


class _CharacterTokenizer:
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": [ord(character) for character in text]}

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(int(token_id)) for token_id in token_ids)

    def apply_chat_template(self, messages, tools, enable_thinking,
                            add_generation_prompt, tokenize):
        assert enable_thinking is False
        assert add_generation_prompt is True
        assert tokenize is False
        rendered = "TOOLS:" + json.dumps(
            tools, sort_keys=True, separators=(",", ":")) + "\n"
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "tool":
                rendered += ("<|im_start|>user\n<tool_response>\n" + content +
                             "\n</tool_response><|im_end|>\n")
            elif role == "assistant":
                rendered += ("<|im_start|>assistant\n<think>\n\n</think>\n\n" +
                             content + "<|im_end|>\n")
            else:
                rendered += (f"<|im_start|>{role}\n{content}<|im_end|>\n")
        rendered += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return rendered


class _LastQueryAwareCharacterTokenizer(_CharacterTokenizer):
    """Model the Qwen template behavior that controls assistant think tags."""

    def apply_chat_template(self, messages, tools, enable_thinking,
                            add_generation_prompt, tokenize):
        assert enable_thinking is False
        assert add_generation_prompt is True
        assert tokenize is False
        last_query_index = -1
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            content = message["content"].strip()
            is_tool_response = (content.startswith("<tool_response>")
                                and content.endswith("</tool_response>"))
            if message["role"] == "user" and not is_tool_response:
                last_query_index = index
                break

        rendered = "TOOLS:" + json.dumps(
            tools, sort_keys=True, separators=(",", ":")) + "\n"
        for index, message in enumerate(messages):
            role = message["role"]
            content = message["content"]
            if role == "tool":
                rendered += ("<|im_start|>user\n<tool_response>\n" + content +
                             "\n</tool_response><|im_end|>\n")
            elif role == "assistant":
                rendered += "<|im_start|>assistant\n"
                if index > last_query_index:
                    rendered += "<think>\n\n</think>\n\n"
                rendered += content + "<|im_end|>\n"
            else:
                rendered += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        rendered += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return rendered


class _TrimNoncanonicalTokenizer:
    """Exercise Qwen's content trim with a sampled non-canonical token."""

    pad_token_id = 0
    eos_token_id = 1
    noncanonical_token_id = 2
    _eos_text = "<|im_end|>"

    def _encode(self, text):
        output = []
        index = 0
        while index < len(text):
            if text.startswith(self._eos_text, index):
                output.append(self.eos_token_id)
                index += len(self._eos_text)
            else:
                output.append(ord(text[index]) + 10)
                index += 1
        return output

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": self._encode(text)}

    def decode(self, token_ids, skip_special_tokens=True):
        output = []
        for token_id in token_ids:
            token_id = int(token_id)
            if token_id == self.eos_token_id:
                if not skip_special_tokens:
                    output.append(self._eos_text)
            elif token_id == self.noncanonical_token_id:
                output.append("France")
            else:
                output.append(chr(token_id - 10))
        return "".join(output)

    def apply_chat_template(self, messages, tools, enable_thinking,
                            add_generation_prompt, tokenize):
        assert enable_thinking is False
        assert add_generation_prompt is True
        assert tokenize is False
        rendered = "TOOLS:" + json.dumps(
            tools, sort_keys=True, separators=(",", ":")) + "\n"
        for message in messages:
            role = message["role"]
            content = message["content"].strip()
            if role == "tool":
                rendered += ("<|im_start|>user\n<tool_response>\n" + content +
                             "\n</tool_response><|im_end|>\n")
            elif role == "assistant":
                rendered += ("<|im_start|>assistant\n<think>\n\n</think>\n\n" +
                             content + "<|im_end|>\n")
            else:
                rendered += (f"<|im_start|>{role}\n{content}"
                             "<|im_end|>\n")
        rendered += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return rendered


@pytest.mark.parametrize(
    ("text", "action", "content", "error"),
    [
        (
            "<tool_call>\n<function=search>\n<parameter=query>\n"
            "capital of France\n</parameter>\n</function>\n</tool_call>",
            "search",
            "capital of France",
            None,
        ),
        (
            "I will verify.\n<tool_call><function=search>"
            "<parameter=query>France capital</parameter></function></tool_call>",
            "search",
            "France capital",
            None,
        ),
        ("Paris", "answer", "Paris", None),
        ("  Paris, France  ", "answer", "Paris, France", None),
        ("", None, "", "empty_response"),
        (
            "<tool_call><function=search><parameter=query>query</parameter>"
            "</function></tool_call>",
            None,
            "",
            "placeholder_search_query",
        ),
        (
            "<tool_call><function=search><parameter=q>France</parameter>"
            "</function></tool_call>",
            None,
            "",
            "invalid_search_parameter",
        ),
        (
            "<tool_call><function=finish><parameter=query>Paris</parameter>"
            "</function></tool_call>",
            None,
            "",
            "unknown_tool",
        ),
        (
            "<tool_call><function=search><parameter=query>France</parameter>"
            "</function></tool_call> trailing",
            None,
            "",
            "malformed_tool_call",
        ),
        ("<search>France</search>", None, "",
         "multiple_or_unbalanced_tool_calls"),
        (
            '{"name":"search","arguments":{"query":"France"}}',
            None,
            "",
            "json_tool_call_not_supported",
        ),
    ],
)
def test_native_parser_is_strict(text, action, content, error):
    parsed = parse_action(text, QWEN35_NATIVE)

    assert parsed.action == action
    assert parsed.content == content
    assert parsed.error == error


def test_legacy_parser_keeps_historical_first_match_behavior():
    parsed = parse_action(
        "reason <search>first</search><answer>later</answer>", LEGACY_XML)

    assert parsed == ParsedAction("search", "first")


def test_tool_schema_copy_and_digest_are_stable():
    first = qwen35_tools()
    digest = qwen35_tool_schema_sha256()
    first[0]["function"]["name"] = "changed"

    assert qwen35_tools()[0]["function"]["name"] == "search"
    assert qwen35_tool_schema_sha256() == digest
    assert len(digest) == 64


def test_native_message_contract_preserves_question_text():
    messages = qwen35_messages("Who wrote Hamlet?")

    assert messages[1]["content"] == "Question: Who wrote Hamlet?\n"
    assert validate_qwen35_messages(messages) == messages
    with pytest.raises(ProtocolError, match="system contract"):
        validate_qwen35_messages([{
            "role": "system",
            "content": "different"
        }, messages[1]])


def test_native_conversation_preserves_prefix_and_masks_complete_wrapper():
    tokenizer = _CharacterTokenizer()
    messages = qwen35_messages("Where is the Eiffel Tower?")
    initial = render_qwen35_prompt(tokenizer, messages)
    initial_ids = tokenizer(initial, add_special_tokens=False)["input_ids"]
    conversation = Qwen35Conversation(tokenizer, messages, initial_ids)
    response = ("<tool_call><function=search><parameter=query>"
                "Eiffel Tower location</parameter></function></tool_call>")
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]

    followup = conversation.append_followup(
        response,
        parse_action(response, QWEN35_NATIVE),
        "Paris " * 100,
        max_obs_length=160,
        response_token_ids=response_ids,
    )

    suffix = tokenizer.decode(followup.token_ids)
    assert len(followup.token_ids) <= 160
    assert suffix.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert "<tool_response>" in suffix
    assert "</tool_response>" in suffix
    assert conversation.prompt_token_ids == initial_ids + response_ids + list(
        followup.token_ids)


def test_native_conversation_uses_complete_invalid_retry_wrapper():
    tokenizer = _LastQueryAwareCharacterTokenizer()
    messages = qwen35_messages("Capital of France?")
    initial = render_qwen35_prompt(tokenizer, messages)
    initial_ids = tokenizer(initial, add_special_tokens=False)["input_ids"]
    conversation = Qwen35Conversation(tokenizer, messages, initial_ids)
    response = "<tool_call>broken"
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]

    followup = conversation.append_followup(
        response,
        parse_action(response, QWEN35_NATIVE),
        "",
        max_obs_length=300,
        response_token_ids=response_ids,
    )

    suffix = tokenizer.decode(followup.token_ids)
    sampled_prefix = initial_ids + response_ids
    actual_prefix = conversation.prompt_token_ids[:len(sampled_prefix)]
    assert actual_prefix == sampled_prefix
    assert "<tool_response>" in suffix
    assert "</tool_response>" in suffix
    assert "previous response was not executable" in suffix
    assert suffix.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


@pytest.mark.parametrize("sampled_eos", [True, False])
def test_native_conversation_preserves_trimmed_noncanonical_sample(
        sampled_eos):
    tokenizer = _TrimNoncanonicalTokenizer()
    messages = qwen35_messages("Capital of France?")
    initial = render_qwen35_prompt(tokenizer, messages)
    initial_ids = tokenizer(initial, add_special_tokens=False)["input_ids"]
    conversation = Qwen35Conversation(tokenizer, messages, initial_ids)
    response = (
        "  <tool_call><function=search><parameter=query>capital France"
        "</parameter></function></tool_call>  ")
    before, marker, after = response.partition("France")
    assert marker
    response_ids = (
        tokenizer(before, add_special_tokens=False)["input_ids"] +
        [tokenizer.noncanonical_token_id] +
        tokenizer(after, add_special_tokens=False)["input_ids"])
    if sampled_eos:
        response_ids.append(tokenizer.eos_token_id)

    followup = conversation.append_followup(
        response,
        parse_action(response, QWEN35_NATIVE),
        "Paris is the capital of France.",
        max_obs_length=400,
        response_token_ids=response_ids,
    )

    suffix_ids = list(followup.token_ids)
    assert conversation.prompt_token_ids == initial_ids + response_ids + suffix_ids
    assert conversation.prompt_token_ids[len(initial_ids):len(initial_ids) +
                                         len(response_ids)] == response_ids
    canonical_response_ids = tokenizer(
        response, add_special_tokens=False)["input_ids"]
    assert response_ids[:len(canonical_response_ids)] != canonical_response_ids
    if sampled_eos:
        assert suffix_ids[0] != tokenizer.eos_token_id
    else:
        assert suffix_ids[0] == tokenizer.eos_token_id


def test_native_conversation_rejects_noncanonical_initial_tokens():
    with pytest.raises(ProtocolError, match="dataset prompt tokens"):
        Qwen35Conversation(_CharacterTokenizer(), qwen35_messages("Question?"),
                           [1])


@pytest.mark.parametrize(("value", "expected"), [(0, 0.0), (2, 2.0),
                                                 ("1", 1.0)])
def test_presence_penalty_validation(value, expected):
    assert validate_presence_penalty(value) == expected


@pytest.mark.parametrize("value", [True, -0.1, 2.1, float("nan"), "bad"])
def test_invalid_presence_penalty_is_rejected(value):
    with pytest.raises(ValueError, match="presence_penalty"):
        validate_presence_penalty(value)
