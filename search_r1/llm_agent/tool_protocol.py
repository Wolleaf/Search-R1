"""Model-facing tool protocol adapters for Search-R1 rollouts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping, Optional, Sequence


LEGACY_XML = "legacy_xml"
QWEN35_NATIVE = "qwen35_native"
SUPPORTED_TOOL_PROTOCOLS = (LEGACY_XML, QWEN35_NATIVE)

QWEN35_REASONING_CONTINUATION = "continuation"
QWEN35_REASONING_FULL = "full"
QWEN35_REASONING_MODES = (
    QWEN35_REASONING_CONTINUATION,
    QWEN35_REASONING_FULL,
)

QWEN35_PROMPT_VERSION = "qwen35-native-search-v3-original-aligned"
QWEN35_MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
QWEN35_CHAT_TEMPLATE_SHA256 = (
    "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80"
)
QWEN35_RETRY_PROMPT = "My action is not correct. Let me rethink."

_QWEN35_USER_PROMPT_PREFIX = (
    "Answer the given question. You must conduct reasoning inside <think> and "
    "</think> first every time you get new information. After reasoning, if "
    "you find you lack some knowledge, you can call the available search tool "
    "with a query, and it will return the top searched results in a tool "
    "response. You can search as many times as you want. If you find no "
    "further external knowledge needed, you can directly provide the answer "
    "inside <answer> and </answer>, without detailed illustrations. For "
    "example, <answer> Beijing </answer>. Question: "
)

_QWEN35_TOOLS = ({
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search an external knowledge base for relevant passages.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The text query to search for.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}, )

_PROTOCOL_MARKERS = (
    "<tool_call",
    "</tool_call>",
    "<function=",
    "</function>",
    "<parameter=",
    "</parameter>",
    "<tool_response",
    "</tool_response>",
    "<search",
    "</search>",
    "<answer",
    "</answer>",
    "<information",
    "</information>",
    "<think",
    "</think>",
    "<|im_start|>",
    "<|im_end|>",
)
_NATIVE_SEARCH_CALL = re.compile(
    r"\A(?P<prefix>.*?)"
    r"<tool_call>\s*"
    r"<function=(?P<function>[A-Za-z_][A-Za-z0-9_.-]*)>\s*"
    r"<parameter=(?P<parameter>[A-Za-z_][A-Za-z0-9_.-]*)>\s*"
    r"(?P<content>.*?)\s*"
    r"</parameter>\s*</function>\s*</tool_call>\s*\Z",
    flags=re.DOTALL,
)
_NATIVE_ANSWER = re.compile(
    r"\A(?P<prefix>.*?)<answer>(?P<content>.*?)</answer>\s*\Z",
    flags=re.DOTALL,
)
_LEGACY_ACTION = re.compile(r"<(search|answer)>(.*?)</\1>", re.DOTALL)


class ProtocolError(ValueError):
    """Raised when a configured protocol or rendered conversation is invalid."""


@dataclass(frozen=True)
class ParsedAction:
    action: Optional[str]
    content: str
    error: Optional[str] = None
    prefix: str = ""

    @property
    def valid(self) -> bool:
        return self.action in ("search", "answer") and self.error is None


@dataclass(frozen=True)
class Qwen35ActionBoundary:
    """Reasoning split and first closing delimiter in one native response."""

    reasoning: str
    action_start: Optional[int]
    boundary: Optional[str]
    delimiter_start: Optional[int]
    logical_end: Optional[int]
    error: Optional[str] = None


@dataclass(frozen=True)
class ProtocolFollowup:
    token_ids: tuple[int, ...]
    visible_observation: str


def normalize_tool_protocol(value: Any) -> str:
    protocol = str(value)
    if protocol not in SUPPORTED_TOOL_PROTOCOLS:
        raise ProtocolError(
            f"tool_protocol must be one of {SUPPORTED_TOOL_PROTOCOLS}, got "
            f"{protocol!r}")
    return protocol


def normalize_qwen35_reasoning_mode(value: Any) -> str:
    mode = str(value)
    if mode not in QWEN35_REASONING_MODES:
        raise ProtocolError(
            f"qwen35 reasoning mode must be one of {QWEN35_REASONING_MODES}, "
            f"got {mode!r}")
    return mode


def locate_qwen35_action_boundary(
        text: str, reasoning_mode: str) -> Qwen35ActionBoundary:
    """Locate the first native close marker after the reasoning channel."""
    if not isinstance(text, str):
        raise TypeError("model response must be a string")
    mode = normalize_qwen35_reasoning_mode(reasoning_mode)

    opening_start: Optional[int] = None
    if mode == QWEN35_REASONING_CONTINUATION:
        close_start = text.find("</think>")
        reasoning_start = 0
    else:
        opening_start = text.find("<think>")
        if opening_start < 0 or text[:opening_start].strip():
            return Qwen35ActionBoundary("", None, None, None, None,
                                        "invalid_thinking_prefix")
        close_start = text.find("</think>", opening_start + len("<think>"))
        reasoning_start = opening_start + len("<think>")
    if close_start < 0:
        return Qwen35ActionBoundary("", None, None, None, None,
                                    "invalid_thinking_prefix")

    action_start = close_start + len("</think>")
    completed = []
    for marker, boundary in (("</tool_call>", "tool_call"),
                             ("</answer>", "answer")):
        position = text.find(marker, action_start)
        if position >= 0:
            completed.append((position, boundary, len(marker)))
    delimiter_start = None
    boundary = None
    logical_end = None
    if completed:
        delimiter_start, boundary, marker_length = min(
            completed, key=lambda item: item[0])
        logical_end = delimiter_start + marker_length

    # Raw tail after the first complete action is audit-only and cannot make
    # the already selected reasoning/action prefix invalid.
    validation_end = logical_end if logical_end is not None else len(text)
    validated_prefix = text[:validation_end]
    if mode == QWEN35_REASONING_CONTINUATION:
        thinking_valid = (validated_prefix.count("</think>") == 1
                          and "<think" not in validated_prefix)
    else:
        thinking_valid = (
            validated_prefix.count("<think>") == 1
            and validated_prefix.count("<think") == 1
            and validated_prefix.count("</think>") == 1
            and opening_start is not None)
    if not thinking_valid:
        return Qwen35ActionBoundary("", None, None, None, None,
                                    "invalid_thinking_prefix")

    reasoning = text[reasoning_start:close_start].strip()
    if logical_end is None:
        return Qwen35ActionBoundary(reasoning, action_start, None, None, None)
    return Qwen35ActionBoundary(
        reasoning=reasoning,
        action_start=action_start,
        boundary=boundary,
        delimiter_start=delimiter_start,
        logical_end=logical_end,
    )


def qwen35_tools() -> list[dict[str, Any]]:
    """Return an isolated schema copy suitable for ``apply_chat_template``."""
    return deepcopy(list(_QWEN35_TOOLS))


def qwen35_tool_schema_sha256() -> str:
    payload = json.dumps(qwen35_tools(),
                         ensure_ascii=True,
                         sort_keys=True,
                         separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def qwen35_user_prompt(question: str) -> str:
    question = re.sub(r"\s+", " ", str(question)).strip()
    if not question:
        raise ValueError("question must not be empty")
    if any(marker in question for marker in _PROTOCOL_MARKERS):
        raise ProtocolError("native question contains a reserved protocol marker")
    if not question.endswith("?"):
        question += "?"
    return f"{_QWEN35_USER_PROMPT_PREFIX}{question}\n"


def qwen35_messages(question: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": qwen35_user_prompt(question)}]


def validate_qwen35_messages(
        messages: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Validate the materialized single-user native prompt contract."""
    if len(messages) != 1:
        raise ProtocolError("native prompt must contain exactly one user message")
    user = messages[0]
    if set(user) != {"role", "content"} or user.get("role") != "user":
        raise ProtocolError("native prompt must contain exactly one user message")
    content = user.get("content")
    if (not isinstance(content, str)
            or not content.startswith(_QWEN35_USER_PROMPT_PREFIX)
            or not content.endswith("\n")):
        raise ProtocolError("native user message does not match the original prompt")
    question = content[len(_QWEN35_USER_PROMPT_PREFIX):-1]
    try:
        expected = qwen35_user_prompt(question)
    except (ProtocolError, ValueError) as error:
        raise ProtocolError(
            "native user message does not contain a valid question") from error
    if content != expected:
        raise ProtocolError("native user message is not canonical")
    return deepcopy([dict(user)])


def render_qwen35_prompt(tokenizer: Any,
                         messages: Sequence[Mapping[str, Any]]) -> str:
    if not messages:
        raise ProtocolError("native Qwen prompt requires at least one message")
    return tokenizer.apply_chat_template(
        list(messages),
        tools=qwen35_tools(),
        enable_thinking=True,
        add_generation_prompt=True,
        tokenize=False,
    )


def parse_action(text: str,
                 tool_protocol: str,
                 *,
                 qwen35_reasoning_mode: Optional[str] = None) -> ParsedAction:
    protocol = normalize_tool_protocol(tool_protocol)
    if not isinstance(text, str):
        raise TypeError("model response must be a string")
    if protocol == LEGACY_XML:
        if qwen35_reasoning_mode is not None:
            raise ProtocolError(
                "qwen35 reasoning mode is invalid for the legacy protocol")
        match = _LEGACY_ACTION.search(text)
        if match is None:
            return ParsedAction(None, "", "missing_legacy_action")
        return ParsedAction(match.group(1), match.group(2).strip())
    if qwen35_reasoning_mode is None:
        raise ProtocolError("qwen35 reasoning mode is required")
    return _parse_qwen35_action(text, qwen35_reasoning_mode)


def _parse_qwen35_action(text: str, reasoning_mode: str) -> ParsedAction:
    if not text.strip():
        return ParsedAction(None, "", "empty_response")
    located = locate_qwen35_action_boundary(text, reasoning_mode)
    if located.error is not None:
        return ParsedAction(None, "", located.error)
    if located.action_start is None:
        raise AssertionError("valid native reasoning has no action start")

    logical_end = located.logical_end if located.logical_end is not None else len(
        text)
    candidate = text[located.action_start:logical_end].strip()
    if not candidate:
        return ParsedAction(None, "", "missing_native_action")

    if located.boundary == "answer":
        if (candidate.count("<answer>") != 1
                or candidate.count("</answer>") != 1):
            return ParsedAction(None, "", "multiple_or_unbalanced_answers")
        match = _NATIVE_ANSWER.fullmatch(candidate)
        if match is None:
            return ParsedAction(None, "", "malformed_answer")
        prefix, prefix_error = _normalize_qwen35_action_prefix(
            match.group("prefix"))
        if prefix_error is not None:
            return ParsedAction(None, "", prefix_error)
        answer = match.group("content").strip()
        if not answer:
            return ParsedAction(None, "", "empty_answer")
        if any(marker in answer for marker in _PROTOCOL_MARKERS):
            return ParsedAction(None, "", "nested_protocol_marker")
        return ParsedAction(
            "answer",
            answer,
            prefix=_join_reasoning_prefix(located.reasoning, prefix),
        )

    if located.boundary is None:
        has_answer_marker = ("<answer" in candidate
                             or "</answer>" in candidate)
        if has_answer_marker:
            return ParsedAction(None, "", "multiple_or_unbalanced_answers")
        has_marker = any(marker in candidate for marker in _PROTOCOL_MARKERS)
        if not has_marker:
            if _looks_like_json_tool_call(candidate):
                return ParsedAction(None, "", "json_tool_call_not_supported")
            return ParsedAction(None, "", "missing_native_action")
        return ParsedAction(None, "", "multiple_or_unbalanced_tool_calls")

    if (candidate.count("<tool_call>") != 1
            or candidate.count("</tool_call>") != 1
            or candidate.count("<function=") != 1
            or candidate.count("</function>") != 1
            or candidate.count("<parameter=") != 1
            or candidate.count("</parameter>") != 1):
        return ParsedAction(None, "", "multiple_or_unbalanced_tool_calls")
    match = _NATIVE_SEARCH_CALL.fullmatch(candidate)
    if match is None:
        return ParsedAction(None, "", "malformed_tool_call")
    if match.group("function") != "search":
        return ParsedAction(None, "", "unknown_tool")
    if match.group("parameter") != "query":
        return ParsedAction(None, "", "invalid_search_parameter")
    query = match.group("content").strip()
    if not query:
        return ParsedAction(None, "", "empty_search_query")
    if any(marker in query for marker in _PROTOCOL_MARKERS):
        return ParsedAction(None, "", "nested_protocol_marker")
    prefix, prefix_error = _normalize_qwen35_action_prefix(
        match.group("prefix"))
    if prefix_error is not None:
        return ParsedAction(None, "", prefix_error)
    return ParsedAction("search",
                        query,
                        prefix=_join_reasoning_prefix(located.reasoning,
                                                      prefix))


def _normalize_qwen35_action_prefix(
        prefix: str) -> tuple[str, Optional[str]]:
    """Validate marker-free prose between thinking and the native action."""
    candidate = prefix.strip()
    if any(marker in candidate for marker in _PROTOCOL_MARKERS):
        return "", "invalid_action_prefix"
    return candidate, None


def _join_reasoning_prefix(reasoning: str, action_prefix: str) -> str:
    return "\n".join(part for part in (reasoning, action_prefix) if part)


def _looks_like_json_tool_call(candidate: str) -> bool:
    if not candidate.startswith("{"):
        return False
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        return bool(
            re.search(
                r'"(?:name|function|tool|tool_calls|arguments|query)"\s*:',
                candidate,
            ))
    if not isinstance(value, dict):
        return False
    tool_keys = {"name", "function", "tool", "tool_calls", "arguments", "query"}
    return bool(tool_keys.intersection(value))


def canonical_response_text(text: str, tool_protocol: str) -> str:
    protocol = normalize_tool_protocol(tool_protocol)
    if protocol == QWEN35_NATIVE:
        return text
    if "</search>" in text:
        return text.split("</search>", 1)[0] + "</search>"
    if "</answer>" in text:
        return text.split("</answer>", 1)[0] + "</answer>"
    return text


def _tokenize_text(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    token_ids = encoded["input_ids"]
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        if len(token_ids) != 1:
            raise ProtocolError("tokenizer returned an unexpected batch")
        token_ids = token_ids[0]
    if not isinstance(token_ids, list) or not all(
            isinstance(token_id, int) for token_id in token_ids):
        raise ProtocolError("tokenizer returned invalid input_ids")
    return token_ids


def _decode_token_ids(tokenizer: Any, token_ids: Sequence[int]) -> str:
    try:
        return tokenizer.decode(
            list(token_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(list(token_ids), skip_special_tokens=False)


def _validate_response_token_ids(tokenizer: Any, response_text: str,
                                 response_ids: Sequence[int]) -> None:
    """Validate sampled IDs by decoding only; never canonicalize the sample."""
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    text_ids = list(response_ids)
    for index, token_id in enumerate(text_ids):
        if eos_token_id is not None and token_id == eos_token_id:
            if index != len(text_ids) - 1:
                raise ProtocolError(
                    "response_token_ids contain an internal EOS token")
            text_ids = text_ids[:-1]
            break
        if pad_token_id is not None and token_id == pad_token_id:
            raise ProtocolError("response_token_ids contain a padding token")
    decoded = _decode_token_ids(tokenizer, text_ids)
    if decoded != response_text:
        raise ProtocolError(
            "response_token_ids do not decode exactly to response_text")


class Qwen35Conversation:
    """Render exact native multi-turn suffixes while retaining model tokens."""

    def __init__(self, tokenizer: Any, messages: Sequence[Mapping[str, Any]],
                 prompt_token_ids: Sequence[int]) -> None:
        self.tokenizer = tokenizer
        self.messages = deepcopy(list(messages))
        self.prompt_token_ids = [int(token_id) for token_id in prompt_token_ids]
        rendered = render_qwen35_prompt(self.tokenizer, self.messages)
        expected = _tokenize_text(self.tokenizer, rendered)
        if expected != self.prompt_token_ids:
            raise ProtocolError(
                "native dataset prompt tokens differ from apply_chat_template")

    def _render_candidate(
        self,
        response_text: str,
        action: ParsedAction,
        observation: str,
        observation_token_limit: Optional[int] = None,
    ) -> tuple[list[dict[str, Any]], list[int], str]:
        messages = deepcopy(self.messages)
        if not action.valid:
            # Qwen suppresses ``reasoning_content`` for every assistant turn
            # before the latest ordinary user message. Materialize those
            # wrappers before adding the user-role retry so rerendering cannot
            # rewrite any already sampled token.
            for message in reversed(messages):
                if message.get("role") == "user":
                    break
                if (message.get("role") == "assistant"
                        and "reasoning_content" in message):
                    reasoning = str(message["reasoning_content"]).strip()
                    content = str(message.get("content", "")).strip()
                    message["content"] = (
                        f"<think>\n{reasoning}\n</think>\n\n{content}")
                    message["reasoning_content"] = ""
        assistant_index = len(messages)
        if action.valid:
            marker = "<tool_call>" if action.action == "search" else "<answer>"
            candidate = response_text
            located = locate_qwen35_action_boundary(
                candidate, QWEN35_REASONING_CONTINUATION)
            expected_boundary = (
                "tool_call" if action.action == "search" else "answer")
            if (located.error is not None
                    or located.boundary != expected_boundary
                    or located.action_start is None):
                raise ProtocolError(
                    "parsed native action boundary does not match response")
            marker_index = candidate.find(marker, located.action_start)
            if marker_index < 0:
                raise ProtocolError("parsed native action marker is missing")
            messages.append({
                "role": "assistant",
                "content": candidate[marker_index:].strip(),
                "reasoning_content": action.prefix,
            })
        else:
            # A later user retry makes Qwen stop adding the historical think
            # wrapper, so retain the opening tag that was already in the
            # sampled generation prefix as assistant content.
            messages.append({
                "role": "assistant",
                "content": "<think>\n" + response_text,
                "reasoning_content": "",
            })
        visible_observation = ""
        if action.action == "search":
            observation_ids = _tokenize_text(self.tokenizer,
                                             observation.strip())
            if observation_token_limit is not None:
                observation_ids = observation_ids[:observation_token_limit]
            visible_observation = self.tokenizer.decode(
                observation_ids, skip_special_tokens=True).strip()
            messages.append({"role": "tool", "content": visible_observation})
        elif not action.valid:
            messages.append({"role": "user", "content": QWEN35_RETRY_PROMPT})
        else:
            raise ProtocolError("answer actions do not have a follow-up prompt")

        rendered = render_qwen35_prompt(self.tokenizer, messages)
        sentinel_index = 0
        while True:
            sentinel = f"SEARCHR1ASSISTANTBOUNDARY{sentinel_index}END"
            if sentinel not in rendered:
                break
            sentinel_index += 1

        sentinel_messages = deepcopy(messages)
        sentinel_messages[assistant_index]["content"] = sentinel
        sentinel_rendered = render_qwen35_prompt(self.tokenizer,
                                                 sentinel_messages)
        if sentinel_rendered.count(sentinel) != 1:
            raise ProtocolError(
                "native chat template did not preserve the assistant sentinel")
        _, _, suffix_text = sentinel_rendered.partition(sentinel)
        if not suffix_text or not rendered.endswith(suffix_text):
            raise ProtocolError(
                "native chat template changed the post-assistant suffix")

        suffix_ids = _tokenize_text(self.tokenizer, suffix_text)
        rendered_ids = _tokenize_text(self.tokenizer, rendered)
        sentinel_ids = _tokenize_text(self.tokenizer, sentinel_rendered)
        if (not suffix_ids or rendered_ids[-len(suffix_ids):] != suffix_ids
                or sentinel_ids[-len(suffix_ids):] != suffix_ids):
            raise ProtocolError(
                "native chat template suffix is not an independent token tail")
        return messages, suffix_ids, visible_observation

    def append_followup(self, response_text: str, action: ParsedAction,
                        observation: str,
                        max_obs_length: int,
                        response_token_ids: Optional[Sequence[int]] = None,
                        ) -> ProtocolFollowup:
        if (isinstance(max_obs_length, bool)
                or not isinstance(max_obs_length, int)
                or max_obs_length <= 0):
            raise ValueError("max_obs_length must be a positive integer")
        if response_token_ids is None:
            response_ids = _tokenize_text(self.tokenizer, response_text)
        else:
            response_ids = [int(token_id) for token_id in response_token_ids]
            _validate_response_token_ids(self.tokenizer, response_text,
                                         response_ids)

        def render(limit: Optional[int]):
            messages, template_suffix, visible = self._render_candidate(
                response_text, action, observation, limit)
            suffix = list(template_suffix)
            eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
            if (response_ids and eos_token_id is not None
                    and response_ids[-1] == eos_token_id):
                if not suffix or suffix[0] != eos_token_id:
                    raise ProtocolError(
                        "native chat template suffix does not start with EOS")
                suffix = suffix[1:]
            prompt_ids = self.prompt_token_ids + response_ids + suffix
            return messages, prompt_ids, visible, suffix

        messages, prompt_ids, visible, suffix = render(None)
        if len(suffix) > max_obs_length and action.action == "search":
            observation_ids = _tokenize_text(self.tokenizer,
                                             observation.strip())
            low, high = 0, len(observation_ids)
            best = render(0)
            if len(best[3]) > max_obs_length:
                raise ProtocolError(
                    "max_obs_length cannot fit the native tool-response wrapper")
            while low <= high:
                middle = (low + high) // 2
                candidate = render(middle)
                if len(candidate[3]) <= max_obs_length:
                    best = candidate
                    low = middle + 1
                else:
                    high = middle - 1
            messages, prompt_ids, visible, suffix = best
        if len(suffix) > max_obs_length:
            raise ProtocolError(
                "max_obs_length cannot fit the native retry wrapper")

        self.messages = messages
        self.prompt_token_ids = prompt_ids
        return ProtocolFollowup(tuple(suffix), visible)


def validate_presence_penalty(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("presence_penalty must be a finite number")
    try:
        penalty = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("presence_penalty must be a finite number") from error
    if not math.isfinite(penalty) or not 0.0 <= penalty <= 2.0:
        raise ValueError("presence_penalty must be between 0 and 2")
    return penalty
