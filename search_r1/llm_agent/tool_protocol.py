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

QWEN35_PROMPT_VERSION = "qwen35-native-search-v1"
QWEN35_MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
QWEN35_CHAT_TEMPLATE_SHA256 = (
    "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80"
)
QWEN35_FORCE_SEARCH_INSTRUCTION = "Call search at least once before answering. "
QWEN35_RETRY_PROMPT = (
    "The previous response was not executable. Either call the available "
    "search function once with specific search terms, or return only the "
    "short final answer."
)

_QWEN35_TOOLS = ({
    "type": "function",
    "function": {
        "name": "search",
        "description": (
            "Search the external knowledge base for evidence needed to answer "
            "the question. Call exactly one search per assistant turn."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A specific factual search query.",
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


def qwen35_tools() -> list[dict[str, Any]]:
    """Return an isolated schema copy suitable for ``apply_chat_template``."""
    return deepcopy(list(_QWEN35_TOOLS))


def qwen35_tool_schema_sha256() -> str:
    payload = json.dumps(qwen35_tools(),
                         ensure_ascii=True,
                         sort_keys=True,
                         separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def qwen35_system_prompt() -> str:
    return (
        "Call at most one tool per assistant turn. Use search when external "
        "evidence is needed. After each search result, decide whether another "
        "search is needed. Use at most four searches. "
        "When you have enough evidence, answer with only the short final "
        "answer, without a tool call or explanation."
    )


def qwen35_user_prompt(question: str, force_search: bool = False) -> str:
    question = str(question).strip()
    if not question:
        raise ValueError("question must not be empty")
    force_instruction = QWEN35_FORCE_SEARCH_INSTRUCTION if force_search else ""
    return f"{force_instruction}Question: {question}\n"


def qwen35_messages(question: str,
                    force_search: bool = False) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": qwen35_system_prompt(),
        },
        {
            "role": "user",
            "content": qwen35_user_prompt(question,
                                           force_search=force_search),
        },
    ]


def validate_qwen35_messages(
        messages: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Validate the materialized two-message native prompt contract."""
    if len(messages) != 2:
        raise ProtocolError("native prompt must contain system and user messages")
    system, user = messages
    if set(system) != {"role", "content"} or system.get("role") != "system":
        raise ProtocolError("native prompt must start with the canonical system message")
    if system.get("content") != qwen35_system_prompt():
        raise ProtocolError("native prompt system contract does not match")
    if set(user) != {"role", "content"} or user.get("role") != "user":
        raise ProtocolError("native prompt must end with one user question")
    content = user.get("content")
    if isinstance(content, str) and content.startswith(
            QWEN35_FORCE_SEARCH_INSTRUCTION):
        content = content[len(QWEN35_FORCE_SEARCH_INSTRUCTION):]
    if (not isinstance(content, str) or not content.startswith("Question: ")
            or not content.endswith("\n")
            or not content[len("Question: "):-1].strip()):
        raise ProtocolError("native user message must be 'Question: ...\\n'")
    if any(marker in content for marker in _PROTOCOL_MARKERS):
        raise ProtocolError("native question contains a reserved protocol marker")
    return deepcopy([dict(system), dict(user)])


def render_qwen35_prompt(tokenizer: Any,
                         messages: Sequence[Mapping[str, Any]]) -> str:
    if not messages:
        raise ProtocolError("native Qwen prompt requires at least one message")
    return tokenizer.apply_chat_template(
        list(messages),
        tools=qwen35_tools(),
        enable_thinking=False,
        add_generation_prompt=True,
        tokenize=False,
    )


def parse_action(text: str, tool_protocol: str) -> ParsedAction:
    protocol = normalize_tool_protocol(tool_protocol)
    if not isinstance(text, str):
        raise TypeError("model response must be a string")
    if protocol == LEGACY_XML:
        match = _LEGACY_ACTION.search(text)
        if match is None:
            return ParsedAction(None, "", "missing_legacy_action")
        return ParsedAction(match.group(1), match.group(2).strip())
    return _parse_qwen35_action(text)


def _parse_qwen35_action(text: str) -> ParsedAction:
    candidate = text.strip()
    if not candidate:
        return ParsedAction(None, "", "empty_response")

    has_marker = any(marker in candidate for marker in _PROTOCOL_MARKERS)
    if not has_marker:
        if _looks_like_json_tool_call(candidate):
            return ParsedAction(None, "", "json_tool_call_not_supported")
        return ParsedAction("answer", candidate)

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
    if query.casefold() in {"query", "and"}:
        return ParsedAction(None, "", "placeholder_search_query")
    prefix = match.group("prefix").strip()
    if any(marker in prefix for marker in _PROTOCOL_MARKERS):
        return ParsedAction(None, "", "invalid_search_prefix")
    return ParsedAction("search",
                        query,
                        prefix=prefix)


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
        assistant_index = len(messages)
        messages.append({
            "role": "assistant",
            "content": response_text,
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
            # A plain user retry makes Qwen's template rewrite the preceding
            # assistant prefix. Tool role preserves the sampled token prefix.
            messages.append({"role": "tool", "content": QWEN35_RETRY_PROMPT})
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
