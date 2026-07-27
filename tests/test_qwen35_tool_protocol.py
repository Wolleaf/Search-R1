import hashlib
import json

import pytest

from search_r1.llm_agent.tool_protocol import (
    LEGACY_XML,
    QWEN35_NATIVE,
    QWEN35_PROMPT_VERSION,
    QWEN35_REASONING_CONTINUATION,
    QWEN35_REASONING_FULL,
    QWEN35_RETRY_PROMPT,
    QWEN35_TERMINAL_PROMPT,
    QWEN35_TERMINAL_PROMPT_SHA256,
    QWEN35_TERMINAL_PROMPT_VERSION,
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
        assert enable_thinking is True
        assert add_generation_prompt is True
        assert tokenize is False
        rendered = "TOOLS:" + json.dumps(
            tools, sort_keys=True, separators=(",", ":")) + "\n"
        last_query_index = max(index for index, message in enumerate(messages)
                               if message["role"] == "user")
        for index, message in enumerate(messages):
            role = message["role"]
            content = message["content"]
            if role == "tool":
                rendered += ("<|im_start|>user\n<tool_response>\n" + content +
                             "\n</tool_response><|im_end|>\n")
            elif role == "assistant":
                rendered += "<|im_start|>assistant\n"
                if index > last_query_index:
                    reasoning = message.get("reasoning_content", "").strip()
                    rendered += ("<think>\n" + reasoning +
                                 "\n</think>\n\n")
                rendered += content.strip() + "<|im_end|>\n"
            else:
                rendered += (f"<|im_start|>{role}\n{content}<|im_end|>\n")
        rendered += "<|im_start|>assistant\n<think>\n"
        return rendered


class _LastQueryAwareCharacterTokenizer(_CharacterTokenizer):
    """Model the Qwen template behavior that controls assistant think tags."""

    def apply_chat_template(self, messages, tools, enable_thinking,
                            add_generation_prompt, tokenize):
        assert enable_thinking is True
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
        rendered += "<|im_start|>assistant\n<think>\n"
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
        assert enable_thinking is True
        assert add_generation_prompt is True
        assert tokenize is False
        rendered = "TOOLS:" + json.dumps(
            tools, sort_keys=True, separators=(",", ":")) + "\n"
        last_query_index = max(index for index, message in enumerate(messages)
                               if message["role"] == "user")
        for index, message in enumerate(messages):
            role = message["role"]
            content = message["content"].strip()
            if role == "tool":
                rendered += ("<|im_start|>user\n<tool_response>\n" + content +
                             "\n</tool_response><|im_end|>\n")
            elif role == "assistant":
                rendered += "<|im_start|>assistant\n"
                if index > last_query_index:
                    reasoning = message.get("reasoning_content", "").strip()
                    rendered += ("<think>\n" + reasoning +
                                 "\n</think>\n\n")
                rendered += content + "<|im_end|>\n"
            else:
                rendered += (f"<|im_start|>{role}\n{content}"
                             "<|im_end|>\n")
        rendered += "<|im_start|>assistant\n<think>\n"
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
        ("<answer>Paris</answer>", "answer", "Paris", None),
        ("  <answer>Paris, France</answer>  ", "answer", "Paris, France", None),
        ("Paris", None, "", "missing_native_action"),
        ("", None, "", "empty_response"),
        (
            "<tool_call><function=search><parameter=query>query</parameter>"
            "</function></tool_call>",
            "search",
            "query",
            None,
        ),
        (
            "<tool_call><function=search><parameter=query>and</parameter>"
            "</function></tool_call>",
            "search",
            "and",
            None,
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
            "search",
            "France",
            None,
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
    response = text if not text else "</think>\n" + text
    parsed = parse_action(
        response,
        QWEN35_NATIVE,
        qwen35_reasoning_mode=QWEN35_REASONING_CONTINUATION,
    )

    assert parsed.action == action
    assert parsed.content == content
    assert parsed.error == error


@pytest.mark.parametrize(
    ("prefix", "reasoning_mode", "expected_prefix"),
    [
        ("I will verify.\n</think>\n", QWEN35_REASONING_CONTINUATION,
         "I will verify."),
        ("</think>\n", QWEN35_REASONING_CONTINUATION, ""),
        ("Useful nonempty reasoning.\n</think>\n",
         QWEN35_REASONING_CONTINUATION, "Useful nonempty reasoning."),
        ("<think>Useful nonempty reasoning.</think>\n",
         QWEN35_REASONING_FULL, "Useful nonempty reasoning."),
        ("</think>\nI will verify.\n", QWEN35_REASONING_CONTINUATION,
         "I will verify."),
        ("Reason first.\n</think>\nI will verify.\n",
         QWEN35_REASONING_CONTINUATION,
         "Reason first.\nI will verify."),
    ],
)
@pytest.mark.parametrize(
    ("action_text", "action", "content"),
    [
        ("<answer>Paris</answer>", "answer", "Paris"),
        (
            "<tool_call><function=search><parameter=query>France capital"
            "</parameter></function></tool_call>",
            "search",
            "France capital",
        ),
    ],
)
def test_native_parser_shares_safe_prefix(prefix, reasoning_mode,
                                          expected_prefix, action_text, action,
                                          content):
    parsed = parse_action(
        prefix + action_text,
        QWEN35_NATIVE,
        qwen35_reasoning_mode=reasoning_mode,
    )

    assert parsed == ParsedAction(action, content, prefix=expected_prefix)


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ("<answer></answer>", "empty_answer"),
        ("<answer>   </answer>", "empty_answer"),
        (
            "<answer>Paris <answer>France</answer></answer>",
            "multiple_or_unbalanced_answers",
        ),
        (
            "Reason about <search>France</search>.\n<answer>Paris</answer>",
            "invalid_action_prefix",
        ),
        ("<answer><search>France</search></answer>", "nested_protocol_marker"),
    ],
)
def test_native_parser_rejects_invalid_terminal_answers(text, error):
    parsed = parse_action(
        "</think>\n" + text,
        QWEN35_NATIVE,
        qwen35_reasoning_mode=QWEN35_REASONING_CONTINUATION,
    )

    assert parsed == ParsedAction(None, "", error)


@pytest.mark.parametrize(
    ("reasoning", "action_text", "expected"),
    [
        (
            "A discarded <answer>guess</answer> is only reasoning.",
            "<tool_call><function=search><parameter=query>Passaic County"
            "</parameter></function></tool_call>",
            ParsedAction(
                "search",
                "Passaic County",
                prefix=(
                    "A discarded <answer>guess</answer> is only reasoning."
                ),
            ),
        ),
        (
            "Do not emit this opening marker: <answer>",
            "<tool_call><function=search><parameter=query>New Jersey county"
            "</parameter></function></tool_call>",
            ParsedAction(
                "search",
                "New Jersey county",
                prefix="Do not emit this opening marker: <answer>",
            ),
        ),
        (
            "The token <answer> appeared while reasoning.",
            "<answer>Passaic County</answer>",
            ParsedAction(
                "answer",
                "Passaic County",
                prefix="The token <answer> appeared while reasoning.",
            ),
        ),
        (
            "A stray </tool_call> belongs to reasoning.",
            "<answer>Passaic County</answer>",
            ParsedAction(
                "answer",
                "Passaic County",
                prefix="A stray </tool_call> belongs to reasoning.",
            ),
        ),
    ],
)
def test_native_parser_ignores_protocol_markers_inside_reasoning(
        reasoning, action_text, expected):
    assert parse_action(
        reasoning + "\n</think>\n" + action_text,
        QWEN35_NATIVE,
        qwen35_reasoning_mode=QWEN35_REASONING_CONTINUATION,
    ) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "</think>\n<answer>Paris</answer> trailing audit text",
            ParsedAction("answer", "Paris"),
        ),
        (
            "</think>\n<answer>Paris</answer><answer>Lyon</answer>",
            ParsedAction("answer", "Paris"),
        ),
        (
            "</think>\n<answer>Paris</answer>"
            "<tool_call><function=search><parameter=query>Lyon"
            "</parameter></function></tool_call>",
            ParsedAction("answer", "Paris"),
        ),
        (
            "</think>\n<tool_call><function=search><parameter=query>France"
            "</parameter></function></tool_call><answer>Paris</answer>",
            ParsedAction("search", "France"),
        ),
    ],
)
def test_native_parser_selects_first_complete_action_and_ignores_raw_tail(
        text, expected):
    assert parse_action(
        text,
        QWEN35_NATIVE,
        qwen35_reasoning_mode=QWEN35_REASONING_CONTINUATION,
    ) == expected


@pytest.mark.parametrize(
    "text",
    [
        "</think>prose<answer>Paris</answer>",
        "</think><answer>Paris</answer> garbage",
        ("</think><answer>Paris</answer>"
         "<tool_call><function=search><parameter=query>Lyon"
         "</parameter></function></tool_call>"),
    ],
)
def test_terminal_answer_only_rejects_prose_or_raw_tail(text):
    parsed = parse_action(
        text,
        QWEN35_NATIVE,
        qwen35_reasoning_mode=QWEN35_REASONING_CONTINUATION,
        qwen35_answer_only=True,
    )

    assert parsed.action == "answer"
    assert parsed.content == "Paris"
    assert parsed.error == "invalid_terminal_answer_format"
    assert parsed.valid is False


def test_terminal_answer_only_accepts_reasoning_and_action_whitespace():
    parsed = parse_action(
        "Useful reasoning.</think>\n \t<answer>Paris</answer>\r\n\t",
        QWEN35_NATIVE,
        qwen35_reasoning_mode=QWEN35_REASONING_CONTINUATION,
        qwen35_answer_only=True,
    )

    assert parsed == ParsedAction(
        "answer", "Paris", prefix="Useful reasoning.")


def test_terminal_answer_only_rejects_search_with_auditable_request():
    parsed = parse_action(
        "Need evidence.</think>\n<tool_call><function=search>"
        "<parameter=query>France capital</parameter></function></tool_call>",
        QWEN35_NATIVE,
        qwen35_reasoning_mode=QWEN35_REASONING_CONTINUATION,
        qwen35_answer_only=True,
    )

    assert parsed == ParsedAction(
        "search",
        "France capital",
        error="search_disallowed_after_budget",
        prefix="Need evidence.",
    )
    assert parsed.valid is False


def test_native_parser_keeps_malformed_first_close_as_selected_boundary():
    response = (
        "Reasoning.\n</think>\n</answer>"
        "<tool_call><function=search><parameter=query>France capital"
        "</parameter></function></tool_call>"
    )

    assert parse_action(
        response,
        QWEN35_NATIVE,
        qwen35_reasoning_mode=QWEN35_REASONING_CONTINUATION,
    ) == ParsedAction(None, "", "multiple_or_unbalanced_answers")


def test_native_parser_rejects_second_think_in_production_continuation():
    response = (
        "First thought.\n</think>\n<think>Second thought.</think>\n"
        "<answer>Paris</answer>"
    )

    assert parse_action(
        response,
        QWEN35_NATIVE,
        qwen35_reasoning_mode=QWEN35_REASONING_CONTINUATION,
    ) == ParsedAction(None, "", "invalid_thinking_prefix")


def test_native_parser_accepts_explicit_full_thinking_mode():
    response = "<think>Useful reasoning.</think>\n<answer>Paris</answer>"

    assert parse_action(
        response,
        QWEN35_NATIVE,
        qwen35_reasoning_mode=QWEN35_REASONING_FULL,
    ) == ParsedAction("answer", "Paris", prefix="Useful reasoning.")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            '</think>\n<answer>{"name":"search","arguments":{"query":"France"}}'
            "</answer>",
            ParsedAction(
                "answer",
                '{"name":"search","arguments":{"query":"France"}}',
            ),
        ),
        (
            "</think>\n<tool_call><function=search><parameter=query>"
            '{"query":"France","operator":"and"}'
            "</parameter></function></tool_call>",
            ParsedAction("search", '{"query":"France","operator":"and"}'),
        ),
        (
            '{"name":"search","arguments":{"query":"France"}}\n'
            "</think>\n\n<answer>Paris</answer>",
            ParsedAction(
                "answer",
                "Paris",
                prefix='{"name":"search","arguments":{"query":"France"}}',
            ),
        ),
    ],
)
def test_native_parser_does_not_interpret_valid_payloads_as_json_tool_calls(
        text, expected):
    assert parse_action(
        text,
        QWEN35_NATIVE,
        qwen35_reasoning_mode=QWEN35_REASONING_CONTINUATION,
    ) == expected


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
    messages = qwen35_messages("  Who   wrote\nHamlet  ")

    original_user = (
        "Answer the given question. You must conduct reasoning inside <think> "
        "and </think> first every time you get new information. After "
        "reasoning, if you find you lack some knowledge, you can call a search "
        "engine by <search> query </search> and it will return the top searched "
        "results between <information> and </information>. You can search as "
        "many times as your want. If you find no further external knowledge "
        "needed, you can "
        "directly provide the answer inside <answer> and </answer>, without "
        "detailed illustrations. For example, <answer> Beijing </answer>. "
        "Question: Who wrote Hamlet?\n"
    )
    expected_user = original_user.replace(
        "call a search engine by <search> query </search> and",
        "call the available search tool with a query, and",
    ).replace(
        "between <information> and </information>",
        "in a tool response",
    ).replace("as your want", "as you want")
    assert QWEN35_PROMPT_VERSION == (
        "qwen35-native-search-v4-terminal-answer-only")
    assert messages == [{"role": "user", "content": expected_user}]
    assert validate_qwen35_messages(messages) == messages
    with pytest.raises(ProtocolError, match="exactly one user"):
        validate_qwen35_messages([{
            "role": "system",
            "content": "custom policy"
        }, messages[0]])


def test_native_contract_contains_no_added_search_policy():
    assert QWEN35_RETRY_PROMPT == "My action is not correct. Let me rethink."
    contract = json.dumps(qwen35_messages("Question")) + json.dumps(
        qwen35_tools()) + QWEN35_RETRY_PROMPT
    for forbidden in ("at least once", "at most four", "exactly one search",
                      "call search once", "specific factual"):
        assert forbidden not in contract.casefold()


def test_terminal_answer_only_prompt_contract_is_versioned_and_hashed():
    expected = (
        "The search budget is exhausted. You must not call the search tool "
        "again. Using only the question and information already available, "
        "give your best answer even if uncertain. After reasoning, output "
        "exactly one concise final answer inside <answer> and </answer>, with "
        "no text after </answer>.")

    assert QWEN35_TERMINAL_PROMPT_VERSION == "qwen35-terminal-answer-v1"
    assert QWEN35_TERMINAL_PROMPT == expected
    assert QWEN35_TERMINAL_PROMPT_SHA256 == (
        "afc18b79afaafccece6927aec5ccd7898ef2ae17766bce7ffda244eef388d7f2")
    assert hashlib.sha256(expected.encode("utf-8")).hexdigest() == (
        QWEN35_TERMINAL_PROMPT_SHA256)


def test_native_conversation_preserves_prefix_and_masks_complete_wrapper():
    tokenizer = _CharacterTokenizer()
    messages = qwen35_messages("Where is the Eiffel Tower?")
    initial = render_qwen35_prompt(tokenizer, messages)
    initial_ids = tokenizer(initial, add_special_tokens=False)["input_ids"]
    conversation = Qwen35Conversation(tokenizer, messages, initial_ids)
    response = ("I need current evidence.\n</think>\n\n"
                "<tool_call><function=search><parameter=query>"
                "Eiffel Tower location</parameter></function></tool_call>")
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]

    followup = conversation.append_followup(
        response,
        parse_action(
            response,
            QWEN35_NATIVE,
            qwen35_reasoning_mode=QWEN35_REASONING_CONTINUATION,
        ),
        "Paris " * 100,
        max_obs_length=160,
        response_token_ids=response_ids,
    )

    suffix = tokenizer.decode(followup.token_ids)
    assert len(followup.token_ids) <= 160
    assert suffix.endswith("<|im_start|>assistant\n<think>\n")
    assert "<tool_response>" in suffix
    assert "</tool_response>" in suffix
    assert conversation.prompt_token_ids == initial_ids + response_ids + list(
        followup.token_ids)
    assert conversation.messages[1] == {
        "role": "assistant",
        "content": ("<tool_call><function=search><parameter=query>"
                    "Eiffel Tower location</parameter></function></tool_call>"),
        "reasoning_content": "I need current evidence.",
    }
    assert conversation.messages[2]["role"] == "tool"


def test_native_conversation_uses_complete_invalid_retry_wrapper():
    tokenizer = _LastQueryAwareCharacterTokenizer()
    messages = qwen35_messages("Capital of France?")
    initial = render_qwen35_prompt(tokenizer, messages)
    initial_ids = tokenizer(initial, add_special_tokens=False)["input_ids"]
    conversation = Qwen35Conversation(tokenizer, messages, initial_ids)
    response = "</think>\n<tool_call>broken"
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]

    followup = conversation.append_followup(
        response,
        parse_action(
            response,
            QWEN35_NATIVE,
            qwen35_reasoning_mode=QWEN35_REASONING_CONTINUATION,
        ),
        "",
        max_obs_length=300,
        response_token_ids=response_ids,
    )

    suffix = tokenizer.decode(followup.token_ids)
    sampled_prefix = initial_ids + response_ids
    actual_prefix = conversation.prompt_token_ids[:len(sampled_prefix)]
    assert actual_prefix == sampled_prefix
    assert "<tool_response>" not in suffix
    assert "</tool_response>" not in suffix
    assert QWEN35_RETRY_PROMPT in suffix
    assert "<|im_start|>user\n" in suffix
    assert suffix.endswith("<|im_start|>assistant\n<think>\n")
    assert conversation.messages[-1] == {
        "role": "user",
        "content": QWEN35_RETRY_PROMPT,
    }


def test_native_consecutive_invalid_retries_do_not_rewrite_history():
    tokenizer = _LastQueryAwareCharacterTokenizer()
    messages = qwen35_messages("Capital of France?")
    initial = render_qwen35_prompt(tokenizer, messages)
    initial_ids = tokenizer(initial, add_special_tokens=False)["input_ids"]
    conversation = Qwen35Conversation(tokenizer, messages, initial_ids)
    first = "</think>\n<tool_call>broken"
    first_ids = tokenizer(first, add_special_tokens=False)["input_ids"]
    conversation.append_followup(
        first,
        parse_action(
            first,
            QWEN35_NATIVE,
            qwen35_reasoning_mode=QWEN35_REASONING_CONTINUATION,
        ),
        "",
        max_obs_length=300,
        response_token_ids=first_ids,
    )
    before_second = list(conversation.prompt_token_ids)
    first_assistant = dict(conversation.messages[1])
    second = "</think>\nstill not a valid action"
    second_ids = tokenizer(second, add_special_tokens=False)["input_ids"]

    followup = conversation.append_followup(
        second,
        parse_action(
            second,
            QWEN35_NATIVE,
            qwen35_reasoning_mode=QWEN35_REASONING_CONTINUATION,
        ),
        "",
        max_obs_length=300,
        response_token_ids=second_ids,
    )

    assert conversation.messages[1] == first_assistant
    assert conversation.prompt_token_ids == (
        before_second + second_ids + list(followup.token_ids))
    assert render_qwen35_prompt(tokenizer, conversation.messages) == (
        tokenizer.decode(conversation.prompt_token_ids))


def test_native_user_retry_does_not_rewrite_prior_search_tokens():
    tokenizer = _CharacterTokenizer()
    messages = qwen35_messages("Who wrote Hamlet?")
    initial = render_qwen35_prompt(tokenizer, messages)
    initial_ids = tokenizer(initial, add_special_tokens=False)["input_ids"]
    conversation = Qwen35Conversation(tokenizer, messages, initial_ids)
    search = (
        "I should verify.\n</think>\n\n<tool_call><function=search>"
        "<parameter=query>Hamlet author</parameter></function></tool_call>")
    search_ids = tokenizer(search, add_special_tokens=False)["input_ids"]
    conversation.append_followup(
        search,
        parse_action(
            search,
            QWEN35_NATIVE,
            qwen35_reasoning_mode=QWEN35_REASONING_CONTINUATION,
        ),
        "William Shakespeare wrote Hamlet.",
        max_obs_length=200,
        response_token_ids=search_ids,
    )
    before_retry = list(conversation.prompt_token_ids)
    invalid = "</think>\n<tool_call>broken"
    invalid_ids = tokenizer(invalid, add_special_tokens=False)["input_ids"]

    retry = conversation.append_followup(
        invalid,
        parse_action(
            invalid,
            QWEN35_NATIVE,
            qwen35_reasoning_mode=QWEN35_REASONING_CONTINUATION,
        ),
        "",
        max_obs_length=200,
        response_token_ids=invalid_ids,
    )

    assert conversation.prompt_token_ids == (
        before_retry + invalid_ids + list(retry.token_ids))
    assert render_qwen35_prompt(tokenizer, conversation.messages) == (
        tokenizer.decode(conversation.prompt_token_ids))


def test_native_terminal_prompt_materializes_search_reasoning_history():
    tokenizer = _LastQueryAwareCharacterTokenizer()
    messages = qwen35_messages("Capital of France?")
    initial = render_qwen35_prompt(tokenizer, messages)
    initial_ids = tokenizer(initial, add_special_tokens=False)["input_ids"]
    conversation = Qwen35Conversation(tokenizer, messages, initial_ids)
    first_search = (
        "I should identify the country.\n</think>\n\n"
        "<tool_call><function=search><parameter=query>France country"
        "</parameter></function></tool_call>")
    first_ids = tokenizer(first_search,
                          add_special_tokens=False)["input_ids"]
    conversation.append_followup(
        first_search,
        parse_action(
            first_search,
            QWEN35_NATIVE,
            qwen35_reasoning_mode=QWEN35_REASONING_CONTINUATION,
        ),
        "France is a country in Europe.",
        max_obs_length=300,
        response_token_ids=first_ids,
    )
    before_terminal = list(conversation.prompt_token_ids)
    terminal_search = (
        "I still need its capital.\n</think>\n\n"
        "<tool_call><function=search><parameter=query>capital of France"
        "</parameter></function></tool_call>")
    terminal_ids = tokenizer(terminal_search,
                             add_special_tokens=False)["input_ids"]

    followup = conversation.append_followup(
        terminal_search,
        parse_action(
            terminal_search,
            QWEN35_NATIVE,
            qwen35_reasoning_mode=QWEN35_REASONING_CONTINUATION,
        ),
        "Paris is the capital of France.",
        max_obs_length=500,
        response_token_ids=terminal_ids,
        terminal_answer_only=True,
    )

    assert conversation.prompt_token_ids == (
        before_terminal + terminal_ids + list(followup.token_ids))
    assert [message["role"] for message in conversation.messages[-3:]] == [
        "assistant", "tool", "user"
    ]
    assert conversation.messages[-3] == {
        "role": "assistant",
        "content": "<think>\n" + terminal_search,
        "reasoning_content": "",
    }
    assert QWEN35_TERMINAL_PROMPT in tokenizer.decode(followup.token_ids)
    assert render_qwen35_prompt(tokenizer, conversation.messages) == (
        tokenizer.decode(conversation.prompt_token_ids))


@pytest.mark.parametrize("sampled_eos", [True, False])
def test_native_conversation_preserves_trimmed_noncanonical_sample(
        sampled_eos):
    tokenizer = _TrimNoncanonicalTokenizer()
    messages = qwen35_messages("Capital of France?")
    initial = render_qwen35_prompt(tokenizer, messages)
    initial_ids = tokenizer(initial, add_special_tokens=False)["input_ids"]
    conversation = Qwen35Conversation(tokenizer, messages, initial_ids)
    response = (
        "</think>\n  <tool_call><function=search>"
        "<parameter=query>capital France"
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
        parse_action(
            response,
            QWEN35_NATIVE,
            qwen35_reasoning_mode=QWEN35_REASONING_CONTINUATION,
        ),
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


def test_native_conversation_rejects_inconsistent_sampled_token_ids():
    tokenizer = _TrimNoncanonicalTokenizer()
    messages = qwen35_messages("Capital of France?")
    initial = render_qwen35_prompt(tokenizer, messages)
    initial_ids = tokenizer(initial, add_special_tokens=False)["input_ids"]
    conversation = Qwen35Conversation(tokenizer, messages, initial_ids)
    response = (
        "</think>\n<tool_call><function=search>"
        "<parameter=query>capital France"
        "</parameter></function></tool_call>")
    response_ids = tokenizer(response,
                             add_special_tokens=False)["input_ids"]
    action = parse_action(
        response,
        QWEN35_NATIVE,
        qwen35_reasoning_mode=QWEN35_REASONING_CONTINUATION,
    )

    with pytest.raises(ProtocolError, match="decode exactly"):
        conversation.append_followup(
            response,
            action,
            "Paris",
            400,
            tokenizer(response.replace("France", "Lyon"),
                      add_special_tokens=False)["input_ids"],
        )
    with pytest.raises(ProtocolError, match="internal EOS"):
        conversation.append_followup(
            response,
            action,
            "Paris",
            400,
            response_ids[:2] + [tokenizer.eos_token_id] + response_ids[2:],
        )
    with pytest.raises(ProtocolError, match="padding token"):
        conversation.append_followup(
            response,
            action,
            "Paris",
            400,
            response_ids + [tokenizer.pad_token_id],
        )
    with pytest.raises(ProtocolError, match="internal EOS"):
        conversation.append_followup(
            response,
            action,
            "Paris",
            400,
            response_ids + [tokenizer.eos_token_id, tokenizer.eos_token_id],
        )
    assert conversation.prompt_token_ids == initial_ids


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
