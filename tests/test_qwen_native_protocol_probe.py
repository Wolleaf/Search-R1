from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = (Path(__file__).resolve().parents[1] / "scripts" / "autodl" /
               "qwen_native_protocol_probe.py")
SPEC = importlib.util.spec_from_file_location(
    "autodl_qwen_native_protocol_probe", MODULE_PATH)
assert SPEC and SPEC.loader
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def _prompt_contract() -> dict[str, object]:
    return {
        "schema_version": 5,
        "prompt_contract": {
            "tool_protocol": "qwen35_native",
            "prompt_version": PROBE.PROMPT_VERSION,
            "terminal_answer_only": True,
            "terminal_prompt_version": PROBE.TERMINAL_PROMPT_VERSION,
            "terminal_prompt_sha256": PROBE.TERMINAL_PROMPT_SHA256,
        },
    }


def _resolved(checkpoint_digest: str) -> dict[str, object]:
    return {
        "schema": PROBE.SCHEMA,
        "schema_version": PROBE.SCHEMA_VERSION,
        "checkpoint_digest": checkpoint_digest,
        "prompt_version": PROBE.PROMPT_VERSION,
        "terminal_answer_only": True,
        "terminal_prompt_version": PROBE.TERMINAL_PROMPT_VERSION,
        "terminal_prompt_sha256": PROBE.TERMINAL_PROMPT_SHA256,
        "max_action_budget": 4,
        "max_obs_length": 500,
        "sampling": PROBE.SAMPLING,
    }


def test_active_v4_contract_versions_are_distinct() -> None:
    checkpoint_digest = "a" * 64

    PROBE.validate_active_prompt_contract(_prompt_contract())
    PROBE.validate_resolved_contract(_resolved(checkpoint_digest), checkpoint_digest)

    assert PROBE.SCHEMA_VERSION == 4
    assert PROBE.ACTIVE_DATA_SCHEMA_VERSION == 5
    assert PROBE.PROMPT_VERSION == "qwen35-native-search-v4-terminal-answer-only"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 4),
        ("prompt_version", "qwen35-native-search-v3-original-aligned"),
        ("terminal_answer_only", False),
        ("terminal_prompt_version", "old-terminal"),
        ("terminal_prompt_sha256", "0" * 64),
    ],
)
def test_old_or_incomplete_data_contract_cannot_authorize_v4(
        field: str, value: object) -> None:
    payload = _prompt_contract()
    target = payload if field == "schema_version" else payload["prompt_contract"]
    assert isinstance(target, dict)
    target[field] = value

    with pytest.raises(ValueError, match="active v4 data prompt contract mismatch"):
        PROBE.validate_active_prompt_contract(payload)


def test_historical_resolved_probe_cannot_be_republished_as_v4() -> None:
    checkpoint_digest = "b" * 64
    resolved = _resolved(checkpoint_digest)
    resolved.update({
        "schema_version": 3,
        "prompt_version": "qwen35-native-search-v3-original-aligned",
    })

    with pytest.raises(ValueError, match="active v4 resolved protocol contract mismatch"):
        PROBE.validate_resolved_contract(resolved, checkpoint_digest)
