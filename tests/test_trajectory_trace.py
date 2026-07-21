import hashlib
import json
from pathlib import Path

import pytest

from search_r1.trajectory_trace import (TRACE_SCHEMA, TRACE_SCHEMA_VERSION,
                                        TraceJsonlWriter, inspect_trace_jsonl,
                                        main, parse_search_r1_transcript,
                                        stable_sample_id,
                                        verify_trace_manifest)


def _eval_record(sample_index=7, raw_trajectory=None):
    if raw_trajectory is None:
        raw_trajectory = ("<think>I can answer directly.</think>"
                          "<answer>Paris</answer>")
    turns = parse_search_r1_transcript(raw_trajectory)
    return {
        "sample_id": stable_sample_id("nq", "test", sample_index),
        "source_index": sample_index,
        "question": "What is the capital of France?",
        "gold_answers": ["Paris"],
        "raw_trajectory": raw_trajectory,
        "turns": turns,
        "extracted_answer": "Paris",
        "em": 1,
        "executed_search_count": 0,
        "posthoc_utility": 1.0,
        "response_tokens": 12,
        "response_clipped": False,
        "turns_used": len(turns),
        "invalid_action_count":
        sum(not turn["valid_action"] for turn in turns),
        "checkpoint_digest": "a" * 64,
    }


def _train_record(sample_index=7):
    record = _eval_record(sample_index)
    record.pop("checkpoint_digest")
    record.update({
        "step": 1,
        "group_uid": str(sample_index),
        "group_slot": 0,
        "reward_em_only": 1.0,
        "train_reward": 1.0,
        "group_correct_count": 1,
        "group_reward_mean": 0.2,
        "group_reward_std": 0.4,
        "sequence_advantage": 2.0,
    })
    return record


def test_stable_sample_id_matches_dataset_manifest_identity():
    assert stable_sample_id("nq", "train", 123) == "nq:train:123"
    assert stable_sample_id("nq", "train",
                            123) == stable_sample_id("nq", "train", "123")

    with pytest.raises(ValueError, match="source_split"):
        stable_sample_id("nq", "train split", 123)
    with pytest.raises(ValueError, match="boolean"):
        stable_sample_id("nq", "train", True)


def test_parser_ignores_prompt_examples_and_preserves_multiturn_search():
    transcript = (
        "<|im_start|>user\nUse <answer> Beijing </answer> as an example."
        "<|im_end|>\n<|im_start|>assistant\n"
        "<think>I need evidence.</think><search>France capital</search>\n"
        "<information>Doc 1(Title: France) Paris is the capital.</information>"
        "<think>The document answers it.</think><answer>Paris</answer>"
        "<|im_end|>")

    assert parse_search_r1_transcript(transcript) == [
        {
            "turn": 0,
            "think": "I need evidence.",
            "action": "search",
            "search_query": "France capital",
            "answer": None,
            "observation": "Doc 1(Title: France) Paris is the capital.",
            "invalid_text": [],
            "valid_action": True,
        },
        {
            "turn": 1,
            "think": "The document answers it.",
            "action": "answer",
            "search_query": None,
            "answer": "Paris",
            "observation": None,
            "invalid_text": [],
            "valid_action": True,
        },
    ]


def test_parser_keeps_invalid_output_and_unmatched_tags():
    turns = parse_search_r1_transcript(
        "plain failed action <search>unterminated\n"
        "<think>retry</think><answer>ok</answer>")

    assert turns[0]["action"] == "invalid"
    assert turns[0]["invalid_text"] == [
        "plain failed action <search>unterminated"
    ]
    assert turns[1]["think"] == "retry"
    assert turns[1]["action"] == "answer"
    assert turns[1]["answer"] == "ok"


def test_parser_keeps_extra_text_on_a_recognized_environment_action():
    turns = parse_search_r1_transcript(
        "unstructured preface <answer>Paris</answer>")

    assert len(turns) == 1
    assert turns[0]["action"] == "answer"
    assert turns[0]["valid_action"] is True
    assert turns[0]["invalid_text"] == ["unstructured preface"]


def test_writer_finalizes_exact_rows_with_manifest_and_sha256(tmp_path):
    trace_path = tmp_path / "eval_predictions.jsonl"
    writer = TraceJsonlWriter(
        trace_path,
        record_type="eval",
        expected_rows=2,
        run_id="eval-b",
        stage="control",
    )
    first = writer.append(_eval_record(7))
    writer.append(_eval_record(8))
    manifest = writer.finalize()

    assert first["schema"] == TRACE_SCHEMA
    assert first["schema_version"] == TRACE_SCHEMA_VERSION
    assert writer.rows == 2
    assert trace_path.is_file()
    assert not writer.partial_path.exists()
    assert manifest["artifact"]["rows"] == 2
    assert manifest["artifact"]["expected_rows"] == 2
    assert manifest["artifact"]["sha256"] == hashlib.sha256(
        trace_path.read_bytes()).hexdigest()
    assert inspect_trace_jsonl(trace_path, 2, "eval", "eval-b",
                               "control")["rows"] == 2
    assert verify_trace_manifest(writer.manifest_path,
                                 expected_rows=2) == manifest

    records = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len({record["record_id"] for record in records}) == 2


def test_writer_validates_training_specific_fields(tmp_path):
    writer = TraceJsonlWriter(
        tmp_path / "train_trajectories.jsonl",
        record_type="train",
        expected_rows=1,
        run_id="gated20",
        stage="cost-aware-gated",
    )
    prepared = writer.append(_train_record())
    manifest = writer.finalize()

    assert prepared["record_type"] == "train"
    assert prepared["step"] == 1
    assert manifest["artifact"]["record_type"] == "train"


def test_finalize_rejects_wrong_row_count_without_publishing(tmp_path):
    writer = TraceJsonlWriter(
        tmp_path / "train_trajectories.jsonl",
        record_type="eval",
        expected_rows=2,
        run_id="incomplete",
        stage="cost-aware-gated",
    )
    writer.append(_eval_record())

    with pytest.raises(ValueError, match="expected exactly 2"):
        writer.finalize()

    assert writer.partial_path.is_file()
    assert not writer.path.exists()
    assert not writer.manifest_path.exists()


def test_append_rejects_duplicate_and_invalid_records(tmp_path):
    writer = TraceJsonlWriter(
        tmp_path / "eval_predictions.jsonl",
        record_type="eval",
        expected_rows=2,
        run_id="duplicates",
        stage="control",
    )
    record = _eval_record()
    writer.append(record)
    with pytest.raises(ValueError, match="duplicate"):
        writer.append(record)

    invalid = _eval_record(8)
    invalid["executed_search_count"] = -1
    with pytest.raises(ValueError, match="executed_search_count"):
        writer.append(invalid)
    writer.close()


def test_manifest_verification_detects_trace_tampering(tmp_path):
    writer = TraceJsonlWriter(
        tmp_path / "eval_predictions.jsonl",
        record_type="eval",
        expected_rows=1,
        run_id="tamper-test",
        stage="control",
    )
    writer.append(_eval_record())
    writer.finalize()
    writer.path.write_bytes(writer.path.read_bytes() + b"\n")

    with pytest.raises(ValueError):
        verify_trace_manifest(writer.manifest_path)


def test_verify_cli_checks_exact_rows(tmp_path, capsys):
    writer = TraceJsonlWriter(
        tmp_path / "eval_predictions.jsonl",
        record_type="eval",
        expected_rows=1,
        run_id="cli-test",
        stage="control",
    )
    writer.append(_eval_record())
    writer.finalize()

    assert main([
        "verify", "--manifest",
        str(writer.manifest_path), "--expected-rows", "1"
    ]) == 0
    assert json.loads(capsys.readouterr().out)["artifact"]["rows"] == 1
