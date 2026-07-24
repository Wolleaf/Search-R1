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
