from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


MODULE_PATH = (Path(__file__).resolve().parents[1] / "scripts" / "autodl" /
               "qwen_native_gate_analysis.py")
SPEC = importlib.util.spec_from_file_location("autodl_qwen_native_gate_analysis",
                                              MODULE_PATH)
assert SPEC and SPEC.loader
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)


def _record(slot: int, answer: str | None, em: int) -> dict[str, object]:
    return {
        "sample_id": "hotpotqa:train:0",
        "group_slot": slot,
        "question": "What is the capital city?",
        "gold_answers": ["Paris"],
        "extracted_answer": answer,
        "em": em,
        "turns": [{
            "action": "answer",
            "valid_action": True,
            "search_query": None,
            "retrieval_executed": False,
            "observation": None,
            "retrieved_docs": [],
        }],
        "executed_search_count": 0,
        "invalid_action_count": 0,
        "response_clipped": False,
    }


def _active_record(slot: int, action: str = "answer") -> dict[str, object]:
    record = _record(slot, "Paris" if action == "answer" else None,
                     1 if action == "answer" else 0)
    action_text = ("Reasoning</think><answer>Paris</answer>" if action == "answer"
                   else "Reasoning</think><tool_call>x</tool_call>")
    event = {
        "turn": 0,
        "text": action_text,
        "raw_text": action_text + "ignored tail",
        "raw_token_ids": [1, 2, 3],
        "action_token_ids": [1, 2],
        "tail_dropped": True,
        "generation_context": "initial_question",
        "action": action,
    }
    record.update({
        "schema_version": 3,
        "generation_events": [event],
        "retrieval_events": [],
        "max_action_budget": 4,
        "action_count": 1,
        "policy_token_count": 2,
        "observation_token_count": 0,
        "observation_policy_token_count": 0,
        "info_mask_consistent": True,
    })
    return record


def _terminal_record(requested_action: str = "answer") -> dict[str, object]:
    record = _active_record(0)
    normal = []
    for turn in range(4):
        event = dict(record["generation_events"][0])
        event.update({
            "turn": turn,
            "terminal_generation": False,
            "executed_search": False,
        })
        normal.append(event)
    if requested_action == "answer":
        result = {
            "requested_action": "answer",
            "action": "answer",
            "parse_error": None,
            "terminal_rejection_reason": None,
            "valid_action": True,
        }
    elif requested_action == "search":
        result = {
            "requested_action": "search",
            "action": None,
            "parse_error": "search_disallowed_after_budget",
            "terminal_rejection_reason": "search_disallowed_after_budget",
            "valid_action": False,
        }
    else:
        result = {
            "requested_action": None,
            "action": None,
            "parse_error": "missing_qwen35_action",
            "terminal_rejection_reason": "missing_qwen35_action",
            "valid_action": False,
        }
    terminal = dict(normal[-1])
    terminal.update({
        "turn": 4,
        "terminal_generation": True,
        "terminal_instruction_applied": True,
        "terminal_prompt_version": ANALYSIS.TERMINAL_PROMPT_VERSION,
        "terminal_prompt_sha256": ANALYSIS.TERMINAL_PROMPT_SHA256,
        "terminal_prompt_text": ANALYSIS.TERMINAL_PROMPT_TEXT,
        "terminal_prompt_policy_token_count": 0,
        "terminal_followup_token_count": 8,
        "generation_context": "terminal_answer",
        "done": True,
        **result,
    })
    record.update({
        "generation_events": [*normal, terminal],
        "action_count": 5,
    })
    return record


def _g3_args(tmp_path: Path) -> SimpleNamespace:
    trace = tmp_path / "g3.jsonl"
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text('{"golden_answers":["Paris"]}\n', encoding="utf-8")
    return SimpleNamespace(
        stage="g3",
        trace=trace,
        catalog=catalog,
        expected_checkpoint_digest="a" * 64,
        output_dir=tmp_path / "output",
    )


def _data_contract(tmp_path: Path, schema_version: int,
                   prompt_version: str) -> tuple[Path, Path]:
    sample_ids = [f"hotpotqa:train:{index}" for index in range(16)]
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text("".join(
        json.dumps({
            "sample_id": sample_id,
            "question": "What is the capital city?",
            "golden_answers": ["Paris"],
        }) + "\n" for sample_id in sample_ids), encoding="utf-8")
    manifest = {
        "schema_version": schema_version,
        "prompt_contract": {
            "tool_protocol": "qwen35_native",
            "prompt_version": prompt_version,
            **({
                "terminal_answer_only": True,
                "terminal_prompt_version": ANALYSIS.TERMINAL_PROMPT_VERSION,
                "terminal_prompt_sha256": ANALYSIS.TERMINAL_PROMPT_SHA256,
            } if prompt_version == ANALYSIS.ACTIVE_PROMPT_VERSION else {}),
        },
        "artifacts": {
            "catalog": {
                "file": "catalog.jsonl",
                "sha256": ANALYSIS.sha256_file(catalog),
            },
            "probe_g0": {"rows": 8, "sample_ids": sample_ids[:8]},
            "probe_autonomous": {"rows": 16, "sample_ids": sample_ids},
            "probe_forced": {"rows": 16, "sample_ids": sample_ids},
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, catalog


def _write_fake_g3_outputs(args: SimpleNamespace, record: dict[str, object]) -> None:
    args.output_dir.mkdir()
    trace_digest = ANALYSIS.sha256_file(args.trace)
    catalog_digest = ANALYSIS.sha256_file(args.catalog)
    summary = {
        "decision": "NO-GO",
        "input": {
            "stage": "qwen_native_g3",
            "trace_sha256": trace_digest,
            "catalog_sha256": catalog_digest,
            "checkpoint_digest": args.expected_checkpoint_digest,
        },
        "overall": {
            "correct_count": record["em"],
        },
    }
    decision = {
        "decision": "NO-GO",
        "trace_sha256": trace_digest,
        "catalog_sha256": catalog_digest,
        "checkpoint_digest": args.expected_checkpoint_digest,
    }
    report = {
        "sample_id": record["sample_id"],
        "group_slot": record["group_slot"],
        "em": record["em"],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary),
                                                   encoding="utf-8")
    (args.output_dir / "go_no_go.json").write_text(json.dumps(decision),
                                                    encoding="utf-8")
    (args.output_dir / "per_trajectory.jsonl").write_text(
        json.dumps(report) + "\n", encoding="utf-8")


def test_strict_replay_uses_catalog_gold_and_qa_em_normalization() -> None:
    record = _record(0, "The Paris", 1)

    replay = ANALYSIS.replay_strict_exact_match(
        [record], {"hotpotqa:train:0": ["Paris"]})

    assert replay[("hotpotqa:train:0", 0)]["strict_em"] == 1
    assert replay[("hotpotqa:train:0", 0)]["subem"] == 1
    assert replay[("hotpotqa:train:0", 0)]["answer_field"] == "extracted_answer"


def test_strict_replay_accepts_aligned_final_answer_field() -> None:
    record = _record(0, "Paris", 1)
    record["final_answer"] = record.pop("extracted_answer")

    replay = ANALYSIS.replay_strict_exact_match(
        [record], {"hotpotqa:train:0": ["Paris"]})

    assert replay[("hotpotqa:train:0", 0)]["answer_field"] == "final_answer"
    assert replay[("hotpotqa:train:0", 0)]["strict_em"] == 1


@pytest.mark.parametrize("answer", [None, "", "  \t\n"])
def test_strict_replay_treats_missing_or_blank_answer_as_wrong(
        answer: str | None) -> None:
    record = _record(0, answer, 0)
    record["gold_answers"] = ["The"]

    replay = ANALYSIS.replay_strict_exact_match(
        [record], {"hotpotqa:train:0": ["The"]})

    assert replay[("hotpotqa:train:0", 0)]["strict_em"] == 0
    assert replay[("hotpotqa:train:0", 0)]["subem"] == 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda record: record.update(gold_answers=["London"]),
         "gold answers do not match"),
        (lambda record: record.update(em=1), "EM does not match"),
        (lambda record: record.update(final_answer="London"),
         "extracted and final answers disagree"),
    ],
)
def test_strict_replay_rejects_trace_drift(mutation, message: str) -> None:
    record = _record(0, "The answer is Paris.", 0)
    mutation(record)

    with pytest.raises(ValueError, match=message):
        ANALYSIS.replay_strict_exact_match(
            [record], {"hotpotqa:train:0": ["Paris"]})


def test_g3_replays_strict_em_before_accepting_registered_output(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = _g3_args(tmp_path)
    record = _record(0, "Paris", 1)
    record["checkpoint_digest"] = args.expected_checkpoint_digest
    args.trace.write_text(json.dumps(record) + "\n", encoding="utf-8")
    monkeypatch.setitem(ANALYSIS.STAGE_SHAPES, "g3", (1, 1))

    def fake_registered_gate(_args: SimpleNamespace) -> int:
        _write_fake_g3_outputs(_args, record)
        return 0

    monkeypatch.setattr(ANALYSIS, "analyze_g3_with_registered_gate",
                        fake_registered_gate)
    decision = ANALYSIS.write_outputs(
        args, ["hotpotqa:train:0"],
        {"hotpotqa:train:0": "What is the capital city?"},
        {"hotpotqa:train:0": ["Paris"]}, [])

    summary = json.loads((args.output_dir / "summary.json").read_text())
    report = json.loads(
        (args.output_dir / "per_trajectory.jsonl").read_text())
    assert decision["strict_em_replay"]["verified"] is True
    assert summary["strict_em_replay"]["strict_em_positive_count"] == 1
    assert report["strict_em_replay"]["strict_em"] == 1
    assert report["strict_em_replay"]["answer"] == "Paris"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda record: record.update(gold_answers=["London"]),
         "gold answers do not match"),
        (lambda record: record.update(final_answer="London"),
         "extracted and final answers disagree"),
        (lambda record: record.update(em=0), "EM does not match"),
    ],
)
def test_g3_rejects_input_reward_drift_before_registered_analysis(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation,
        message: str) -> None:
    args = _g3_args(tmp_path)
    record = _record(0, "Paris", 1)
    record["checkpoint_digest"] = args.expected_checkpoint_digest
    mutation(record)
    args.trace.write_text(json.dumps(record) + "\n", encoding="utf-8")
    monkeypatch.setitem(ANALYSIS.STAGE_SHAPES, "g3", (1, 1))

    def unexpected_registered_gate(_args: SimpleNamespace) -> int:
        raise AssertionError("registered analyzer ran before strict EM replay")

    monkeypatch.setattr(ANALYSIS, "analyze_g3_with_registered_gate",
                        unexpected_registered_gate)
    with pytest.raises(ValueError, match=message):
        ANALYSIS.write_outputs(
            args, ["hotpotqa:train:0"],
            {"hotpotqa:train:0": "What is the capital city?"},
            {"hotpotqa:train:0": ["Paris"]}, [])
    assert not args.output_dir.exists()


def test_g3_rejects_registered_report_em_drift(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = _g3_args(tmp_path)
    record = _record(0, "Paris", 1)
    record["checkpoint_digest"] = args.expected_checkpoint_digest
    args.trace.write_text(json.dumps(record) + "\n", encoding="utf-8")
    monkeypatch.setitem(ANALYSIS.STAGE_SHAPES, "g3", (1, 1))

    def drifting_registered_gate(_args: SimpleNamespace) -> int:
        _write_fake_g3_outputs(_args, record)
        report_path = _args.output_dir / "per_trajectory.jsonl"
        report = json.loads(report_path.read_text())
        report["em"] = 0
        report_path.write_text(json.dumps(report) + "\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(ANALYSIS, "analyze_g3_with_registered_gate",
                        drifting_registered_gate)
    with pytest.raises(ValueError, match="report EM disagrees"):
        ANALYSIS.write_outputs(
            args, ["hotpotqa:train:0"],
            {"hotpotqa:train:0": "What is the capital city?"},
            {"hotpotqa:train:0": ["Paris"]}, [])
    assert not args.output_dir.exists()


def test_g2_readiness_requires_strict_positive_and_mixed_group() -> None:
    records = [
        _record(0, "Paris", 1),
        _record(1, "London", 0),
        _record(2, "The answer is Paris.", 0),
    ]
    replay = ANALYSIS.replay_strict_exact_match(
        records, {"hotpotqa:train:0": ["Paris"]})

    overall, _, questions = ANALYSIS.analyze_trace_stage(
        "g2", records, replay)

    assert overall["criteria"]["strict_em_positive_count"]["passed"] is True
    assert overall["criteria"]["strict_em_mixed_group_count"]["passed"] is True
    assert overall["subem_positive_count"] == 2
    assert "subem_positive_count" not in overall["criteria"]
    assert questions[0]["strict_em_mixed"] is True


def test_subem_only_signal_does_not_unlock_g2() -> None:
    records = [
        _record(slot, "The answer is Paris.", 0) for slot in range(3)
    ]
    replay = ANALYSIS.replay_strict_exact_match(
        records, {"hotpotqa:train:0": ["Paris"]})

    overall, _, _ = ANALYSIS.analyze_trace_stage("g2", records, replay)

    assert overall["subem_positive_count"] == 3
    assert overall["criteria"]["strict_em_positive_count"]["passed"] is False
    assert overall["criteria"]["strict_em_mixed_group_count"]["passed"] is False
    assert "subem_positive_count" not in overall["criteria"]


def test_v4_g1_structure_does_not_gate_search_em_or_thinking() -> None:
    records = [_active_record(0), _active_record(1)]

    overall, _, _ = ANALYSIS.analyze_trace_stage(
        "g0_g1", records, active_v4=True)

    assert overall["search_turn_count"] == 0
    assert overall["em_count"] == 2
    assert overall["thinking"]["diagnostic_only"] is True
    assert overall["thinking"]["nonempty_reasoning_count"] == 2
    assert all(item["passed"] for item in overall["criteria"].values())
    assert "legal_first_action_count" not in overall["criteria"]


def test_data_contract_distinguishes_active_schema5_from_legacy_schema3(
        tmp_path: Path) -> None:
    active_dir = tmp_path / "active"
    active_dir.mkdir()
    manifest, catalog = _data_contract(
        active_dir, 5, ANALYSIS.ACTIVE_PROMPT_VERSION)
    expected_ids, _, _, _, active = ANALYSIS.load_data_contract(
        manifest, catalog, "g0_g1")
    assert active is True
    assert len(expected_ids) == 16

    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    manifest, catalog = _data_contract(
        legacy_dir, 3, "qwen35-native-search-v2")
    _, _, _, _, active = ANALYSIS.load_data_contract(
        manifest, catalog, "g0_g1")
    assert active is False


def test_active_prompt_cannot_reuse_legacy_data_schema(tmp_path: Path) -> None:
    manifest, catalog = _data_contract(
        tmp_path, 3, ANALYSIS.ACTIVE_PROMPT_VERSION)

    with pytest.raises(ValueError, match="schema/prompt version mismatch"):
        ANALYSIS.load_data_contract(manifest, catalog, "g0_g1")


def test_historical_v3_schema4_cannot_authorize_active_gate(
        tmp_path: Path) -> None:
    manifest, catalog = _data_contract(
        tmp_path, 4, "qwen35-native-search-v3-original-aligned")

    with pytest.raises(ValueError, match="manifest contract mismatch"):
        ANALYSIS.load_data_contract(manifest, catalog, "g0_g1")


def test_v4_terminal_answer_contract_and_metrics(
        monkeypatch: pytest.MonkeyPatch) -> None:
    record = _terminal_record("answer")
    record["checkpoint_digest"] = "a" * 64
    monkeypatch.setitem(ANALYSIS.STAGE_SHAPES, "g0_g1", (1, 1))

    ANALYSIS.validate_traces(
        [record], "g0_g1", "a" * 64, ["hotpotqa:train:0"],
        {"hotpotqa:train:0": "What is the capital city?"},
        active_v4=True)
    overall, _, _ = ANALYSIS.analyze_trace_stage(
        "g0_g1", [record], active_v4=True)

    assert overall["terminal_generation_count"] == 1
    assert overall["terminal_instruction_applied_count"] == 1
    assert overall["terminal_answer_rate"] == 1.0
    assert overall["terminal_requested_search_rate"] == 0.0
    assert "terminal_answer_rate" not in overall["criteria"]
    assert "terminal_requested_search_rate" not in overall["criteria"]
    assert all(item["passed"] for item in overall["criteria"].values())


def test_v4_terminal_search_is_rejected_but_behavior_rates_are_diagnostic(
        monkeypatch: pytest.MonkeyPatch) -> None:
    record = _terminal_record("search")
    record["checkpoint_digest"] = "a" * 64
    monkeypatch.setitem(ANALYSIS.STAGE_SHAPES, "g0_g1", (1, 1))

    ANALYSIS.validate_traces(
        [record], "g0_g1", "a" * 64, ["hotpotqa:train:0"],
        {"hotpotqa:train:0": "What is the capital city?"},
        active_v4=True)
    overall, _, _ = ANALYSIS.analyze_trace_stage(
        "g0_g1", [record], active_v4=True)

    assert overall["terminal_search_request_count"] == 1
    assert overall["terminal_accepted_search_count"] == 0
    assert overall["terminal_executed_search_count"] == 0
    assert overall["terminal_answer_rate"] == 0.0
    assert overall["terminal_requested_search_rate"] == 1.0
    assert "terminal_answer_rate" not in overall["criteria"]
    assert "terminal_requested_search_rate" not in overall["criteria"]
    assert overall["criteria"]["terminal_accepted_search_count"]["passed"] is True
    assert overall["criteria"]["terminal_executed_search_count"]["passed"] is True
    assert all(item["passed"] for item in overall["criteria"].values())


def test_v4_invalid_terminal_answer_is_audited_as_diagnostic_behavior(
        monkeypatch: pytest.MonkeyPatch) -> None:
    record = _terminal_record("answer")
    record["checkpoint_digest"] = "a" * 64
    record["generation_events"][-1].update({
        "action": None,
        "parse_error": "invalid_terminal_answer_format",
        "terminal_rejection_reason": "invalid_terminal_answer_format",
        "valid_action": False,
    })
    monkeypatch.setitem(ANALYSIS.STAGE_SHAPES, "g0_g1", (1, 1))

    ANALYSIS.validate_traces(
        [record], "g0_g1", "a" * 64, ["hotpotqa:train:0"],
        {"hotpotqa:train:0": "What is the capital city?"},
        active_v4=True)
    overall, _, _ = ANALYSIS.analyze_trace_stage(
        "g0_g1", [record], active_v4=True)

    assert overall["terminal_answer_count"] == 0
    assert overall["terminal_invalid_count"] == 1
    assert overall["terminal_answer_rate"] == 0.0
    assert "terminal_answer_rate" not in overall["criteria"]
    assert "terminal_requested_search_rate" not in overall["criteria"]
    assert all(item["passed"] for item in overall["criteria"].values())


def test_v5_gate_reports_terminal_behavior_without_blocking_admission(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    record = _terminal_record("search")
    record["checkpoint_digest"] = "a" * 64
    trace = tmp_path / "trace.jsonl"
    trace.write_text(json.dumps(record) + "\n", encoding="utf-8")
    args = SimpleNamespace(
        stage="g0_g1",
        trace=trace,
        catalog=tmp_path / "catalog.jsonl",
        data_manifest=tmp_path / "manifest.json",
        expected_checkpoint_digest="a" * 64,
        protocol_probe_dir=tmp_path / "protocol",
        output_dir=tmp_path / "output",
    )
    monkeypatch.setitem(ANALYSIS.STAGE_SHAPES, "g0_g1", (1, 1))
    monkeypatch.setattr(
        ANALYSIS, "analyze_protocol_probe",
        lambda *_args, **_kwargs: ({
            "criteria": {
                "protocol_integrity_count": ANALYSIS.criterion(1, "==", 1),
            },
        }, []),
    )

    decision = ANALYSIS.write_outputs(
        args, ["hotpotqa:train:0"],
        {"hotpotqa:train:0": "What is the capital city?"},
        {"hotpotqa:train:0": ["Paris"]}, [], active_v4=True)

    summary = json.loads((args.output_dir / "summary.json").read_text())
    markdown = (args.output_dir / "summary.md").read_text(encoding="utf-8")
    assert decision["schema_version"] == 5
    assert summary["schema_version"] == 5
    assert decision["decision"] == "GO"
    assert decision["failed_criteria"] == []
    assert summary["overall"]["terminal_answer_rate"] == 0.0
    assert summary["overall"]["terminal_requested_search_rate"] == 1.0
    assert "g1_terminal_answer_rate" not in decision["criteria"]
    assert "g1_terminal_requested_search_rate" not in decision["criteria"]
    assert decision["criteria"]["g1_terminal_accepted_search_count"]["passed"] is True
    assert decision["criteria"]["g1_terminal_executed_search_count"]["passed"] is True
    assert "## Diagnostics" in markdown
    assert "terminal_answer_rate: 0.000 (diagnostic only; does not affect GO/NO-GO)" in markdown
    assert "terminal_requested_search_rate: 1.000 (diagnostic only; does not affect GO/NO-GO)" in markdown


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("action", "answer"),
        ("terminal_rejection_reason", None),
        ("valid_action", True),
    ],
)
def test_v4_invalid_terminal_answer_still_fails_closed_when_inconsistent(
        monkeypatch: pytest.MonkeyPatch, field: str, value: object) -> None:
    record = _terminal_record("answer")
    record["checkpoint_digest"] = "a" * 64
    terminal = record["generation_events"][-1]
    terminal.update({
        "action": None,
        "parse_error": "invalid_terminal_answer_format",
        "terminal_rejection_reason": "invalid_terminal_answer_format",
        "valid_action": False,
    })
    terminal[field] = value
    monkeypatch.setitem(ANALYSIS.STAGE_SHAPES, "g0_g1", (1, 1))

    with pytest.raises(ValueError, match="terminal allowlist result mismatch"):
        ANALYSIS.validate_traces(
            [record], "g0_g1", "a" * 64, ["hotpotqa:train:0"],
            {"hotpotqa:train:0": "What is the capital city?"},
            active_v4=True)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("action", "search"),
        ("parse_error", None),
        ("terminal_rejection_reason", None),
        ("valid_action", True),
    ],
)
def test_v5_diagnostic_search_rate_does_not_relax_terminal_field_validation(
        monkeypatch: pytest.MonkeyPatch, field: str, value: object) -> None:
    record = _terminal_record("search")
    record["checkpoint_digest"] = "a" * 64
    record["generation_events"][-1][field] = value
    monkeypatch.setitem(ANALYSIS.STAGE_SHAPES, "g0_g1", (1, 1))

    with pytest.raises(ValueError, match="terminal allowlist result mismatch"):
        ANALYSIS.validate_traces(
            [record], "g0_g1", "a" * 64, ["hotpotqa:train:0"],
            {"hotpotqa:train:0": "What is the capital city?"},
            active_v4=True)


def test_v4_terminal_generation_requires_answer_only_instruction(
        monkeypatch: pytest.MonkeyPatch) -> None:
    record = _terminal_record("answer")
    record["checkpoint_digest"] = "a" * 64
    record["generation_events"][-1].pop("terminal_instruction_applied")
    monkeypatch.setitem(ANALYSIS.STAGE_SHAPES, "g0_g1", (1, 1))

    with pytest.raises(ValueError, match="terminal answer-only contract"):
        ANALYSIS.validate_traces(
            [record], "g0_g1", "a" * 64, ["hotpotqa:train:0"],
            {"hotpotqa:train:0": "What is the capital city?"},
            active_v4=True)


@pytest.mark.parametrize(
    "field",
    ["terminal_prompt_text", "done", "terminal_followup_token_count"],
)
def test_v5_terminal_generation_requires_complete_prompt_evidence(
        monkeypatch: pytest.MonkeyPatch, field: str) -> None:
    record = _terminal_record("answer")
    record["checkpoint_digest"] = "a" * 64
    record["generation_events"][-1].pop(field)
    monkeypatch.setitem(ANALYSIS.STAGE_SHAPES, "g0_g1", (1, 1))

    with pytest.raises(ValueError, match="terminal answer-only contract"):
        ANALYSIS.validate_traces(
            [record], "g0_g1", "a" * 64, ["hotpotqa:train:0"],
            {"hotpotqa:train:0": "What is the capital city?"},
            active_v4=True)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("terminal_prompt_text", "tampered terminal prompt"),
        ("generation_context", "tool_response"),
        ("done", False),
        ("executed_search", 0),
        ("executed_search", True),
        ("terminal_prompt_policy_token_count", False),
        ("terminal_prompt_policy_token_count", True),
        ("terminal_prompt_policy_token_count", 1),
        ("terminal_followup_token_count", True),
        ("terminal_followup_token_count", 0),
    ],
)
def test_v5_terminal_generation_rejects_invalid_prompt_evidence(
        monkeypatch: pytest.MonkeyPatch, field: str, value: object) -> None:
    record = _terminal_record("answer")
    record["checkpoint_digest"] = "a" * 64
    record["generation_events"][-1][field] = value
    monkeypatch.setitem(ANALYSIS.STAGE_SHAPES, "g0_g1", (1, 1))

    with pytest.raises(ValueError, match="terminal answer-only contract"):
        ANALYSIS.validate_traces(
            [record], "g0_g1", "a" * 64, ["hotpotqa:train:0"],
            {"hotpotqa:train:0": "What is the capital city?"},
            active_v4=True)


def test_v5_terminal_prompt_digest_is_replayed_from_recorded_text(
        monkeypatch: pytest.MonkeyPatch) -> None:
    record = _terminal_record("answer")
    record["checkpoint_digest"] = "a" * 64
    forged_digest = "0" * 64
    record["generation_events"][-1]["terminal_prompt_sha256"] = forged_digest
    monkeypatch.setattr(ANALYSIS, "TERMINAL_PROMPT_SHA256", forged_digest)
    monkeypatch.setitem(ANALYSIS.STAGE_SHAPES, "g0_g1", (1, 1))

    with pytest.raises(ValueError, match="terminal prompt digest mismatch"):
        ANALYSIS.validate_traces(
            [record], "g0_g1", "a" * 64, ["hotpotqa:train:0"],
            {"hotpotqa:train:0": "What is the capital city?"},
            active_v4=True)


def test_v4_g1_separates_requested_executed_retrieval_and_tool_response() -> None:
    record = _active_record(0, action="search")
    record["executed_search_count"] = 0

    overall, decorated, _ = ANALYSIS.analyze_trace_stage(
        "g0_g1", [record], active_v4=True)

    diagnostics = decorated[0]["diagnostics"]
    assert diagnostics["requested_search_count"] == 1
    assert diagnostics["executed_search_count"] == 0
    assert diagnostics["retrieval_event_count"] == 0
    assert diagnostics["nonempty_tool_response_count"] == 0
    assert overall["criteria"]["retrieval_alignment_error_count"]["passed"] is False


def test_v4_validation_rejects_terminal_search_compatibility_field(
        monkeypatch: pytest.MonkeyPatch) -> None:
    record = _active_record(0)
    record.update({
        "sample_id": "hotpotqa:train:0",
        "checkpoint_digest": "a" * 64,
        "terminal_search_request": True,
    })
    monkeypatch.setitem(ANALYSIS.STAGE_SHAPES, "g0_g1", (1, 1))

    with pytest.raises(ValueError, match="action-budget contract"):
        ANALYSIS.validate_traces(
            [record], "g0_g1", "a" * 64, ["hotpotqa:train:0"],
            {"hotpotqa:train:0": "What is the capital city?"},
            active_v4=True)
