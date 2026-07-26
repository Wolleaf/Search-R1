import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from search_r1.trajectory_trace import (TRACE_SCHEMA, TRACE_SCHEMA_VERSION,
                                        TraceJsonlWriter, inspect_trace_jsonl,
                                        main, parse_search_r1_transcript,
                                        prepare_trace_record,
                                        stable_sample_id,
                                        verify_trace_manifest)


def _eval_record(sample_index=7, raw_trajectory=None):
    if raw_trajectory is None:
        raw_trajectory = ("<think>I can answer directly.</think>"
                          "<answer>Paris</answer>")
    turns = parse_search_r1_transcript(raw_trajectory)
    action_text = "I can answer directly.</think><answer>Paris</answer>"
    action_ids = [101, 102]
    return {
        "sample_id": stable_sample_id("nq", "test", sample_index),
        "source_index": sample_index,
        "question": "What is the capital of France?",
        "gold_answers": ["Paris"],
        "raw_trajectory": raw_trajectory,
        "raw_generations": [{
            "turn": 0,
            "raw_text": action_text,
            "raw_token_ids": action_ids,
            "raw_token_count": len(action_ids),
            "action_text": action_text,
            "action_token_ids": action_ids,
            "action_token_count": len(action_ids),
            "boundary": "answer",
            "tail_dropped": False,
            "raw_clipped": False,
        }],
        "turns": turns,
        "extracted_answer": "Paris",
        "em": 1,
        "executed_search_count": 0,
        "max_action_budget": 4,
        "action_count": 1,
        "policy_token_count": len(action_ids),
        "observation_token_count": 0,
        "observation_policy_token_count": 0,
        "info_mask_consistent": True,
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


def _v1_eval_record(sample_index=7):
    record = _eval_record(sample_index)
    for name in (
            "raw_generations", "max_action_budget", "action_count",
            "policy_token_count", "observation_token_count",
            "observation_policy_token_count", "info_mask_consistent"):
        record.pop(name)
    record["max_searches"] = 4
    return record


def _replace_raw_generation(record, action_text, boundary, *, raw_clipped=False):
    token_ids = [101, 102]
    record["raw_generations"] = [{
        "turn": 0,
        "raw_text": action_text,
        "raw_token_ids": token_ids,
        "raw_token_count": len(token_ids),
        "action_text": action_text,
        "action_token_ids": token_ids,
        "action_token_count": len(token_ids),
        "boundary": boundary,
        "tail_dropped": False,
        "raw_clipped": raw_clipped,
    }]
    record["policy_token_count"] = len(token_ids)


def _invalid_eval_record(action_text, boundary, *, raw_clipped=False):
    record = _eval_record()
    record.update({
        "raw_trajectory": f"<think>{action_text}",
        "turns": [{
            "turn": 0,
            "think": "",
            "action": "invalid",
            "search_query": None,
            "answer": None,
            "observation": None,
            "invalid_text": [action_text],
            "valid_action": False,
        }],
        "extracted_answer": None,
        "em": 0,
        "posthoc_utility": 0.0,
        "turns_used": 1,
        "invalid_action_count": 1,
    })
    _replace_raw_generation(record,
                            action_text,
                            boundary,
                            raw_clipped=raw_clipped)
    return record


def _terminal_eval_record():
    record = _eval_record()
    template = record["raw_generations"][0]
    record["raw_generations"] = []
    record["generation_events"] = []
    for turn in range(5):
        terminal = turn == 4
        generation = deepcopy(template)
        generation.update({
            "turn": turn,
            "generation_context": "tool_response",
            "terminal_generation": terminal,
        })
        record["raw_generations"].append(generation)
        record["generation_events"].append({
            "turn": turn,
            "terminal_generation": terminal,
            "executed_search": False,
        })
    record["action_count"] = 5
    record["policy_token_count"] = sum(
        generation["action_token_count"]
        for generation in record["raw_generations"])
    return record


def test_v3_accepts_one_marked_terminal_generation_after_action_budget():
    record = _terminal_eval_record()

    prepared = prepare_trace_record(record, "eval", "unit", "terminal")

    assert prepared["max_action_budget"] == 4
    assert prepared["action_count"] == 5
    assert prepared["raw_generations"][-1]["terminal_generation"] is True


def test_v3_rejects_unmarked_or_misplaced_terminal_generation():
    unmarked = _terminal_eval_record()
    unmarked["raw_generations"][-1]["terminal_generation"] = False
    unmarked["generation_events"][-1]["terminal_generation"] = False
    with pytest.raises(ValueError, match="final terminal generation"):
        prepare_trace_record(unmarked, "eval", "unit", "terminal")

    misplaced = _terminal_eval_record()
    misplaced["raw_generations"][-1]["terminal_generation"] = False
    misplaced["generation_events"][-1]["terminal_generation"] = False
    misplaced["raw_generations"][-2]["terminal_generation"] = True
    misplaced["generation_events"][-2]["terminal_generation"] = True
    with pytest.raises(ValueError, match="final terminal generation"):
        prepare_trace_record(misplaced, "eval", "unit", "terminal")


def test_v3_rejects_retrieval_or_a_second_generation_beyond_terminal():
    retrieval = _terminal_eval_record()
    retrieval["generation_events"][-1]["executed_search"] = True
    with pytest.raises(ValueError, match="executed retrieval"):
        prepare_trace_record(retrieval, "eval", "unit", "terminal")

    over_budget = _terminal_eval_record()
    over_budget["action_count"] = 6
    with pytest.raises(ValueError, match="plus terminal generation"):
        prepare_trace_record(over_budget, "eval", "unit", "terminal")


@pytest.mark.parametrize(
    ("action_text", "boundary", "wrong_boundary"),
    [
        (
            "Mention </tool_call> while reasoning.</think>\n"
            "<answer>Paris</answer>",
            "answer",
            "tool_call",
        ),
        (
            "Mention </answer> while reasoning.</think>\n"
            "<tool_call>broken</tool_call>",
            "tool_call",
            "answer",
        ),
    ],
)
def test_v3_boundary_matches_first_post_think_close(action_text, boundary,
                                                     wrong_boundary):
    record = _eval_record()
    _replace_raw_generation(record, action_text, boundary)

    prepared = prepare_trace_record(record, "eval", "unit", "boundary")
    assert prepared["raw_generations"][0]["boundary"] == boundary

    bad_record = _eval_record()
    _replace_raw_generation(bad_record, action_text, wrong_boundary)
    with pytest.raises(ValueError, match="boundary"):
        prepare_trace_record(bad_record, "eval", "unit", "boundary")


@pytest.mark.parametrize(
    ("action_text", "boundary"),
    [
        (
            "A literal </answer> belongs to reasoning.</think>\nNo action.",
            "answer",
        ),
        (
            "A literal </tool_call> belongs to reasoning.</think>\nNo action.",
            "tool_call",
        ),
    ],
)
def test_v3_reasoning_only_marker_does_not_prove_action_boundary(
        action_text, boundary):
    record = _eval_record()
    _replace_raw_generation(record, action_text, boundary)

    with pytest.raises(ValueError, match="boundary"):
        prepare_trace_record(record, "eval", "unit", "reasoning-marker")


@pytest.mark.parametrize(
    ("action_text", "boundary"),
    [
        ("Ready.</think><answer>Paris</answer>.", "answer"),
        ("Search.</think><tool_call>broken</tool_call>X", "tool_call"),
    ],
)
def test_v3_accepts_decoded_token_overshoot_after_action_close(
        action_text, boundary):
    record = _eval_record()
    _replace_raw_generation(record, action_text, boundary)

    prepared = prepare_trace_record(record, "eval", "unit", "overshoot")
    generation = prepared["raw_generations"][0]
    assert generation["action_text"] == action_text
    assert generation["action_token_ids"] == [101, 102]


@pytest.mark.parametrize(
    ("action_text", "boundary"),
    [
        ("Ready.</think></answer>", "answer"),
        ("Ready.</think></tool_call>", "tool_call"),
    ],
)
def test_v3_keeps_close_only_malformed_generation_traceable(
        action_text, boundary):
    record = _invalid_eval_record(action_text, boundary)

    prepared = prepare_trace_record(record, "eval", "unit", "close-only")
    assert prepared["raw_generations"][0]["action_text"] == action_text
    assert prepared["turns"][0]["action"] == "invalid"
    assert prepared["invalid_action_count"] == 1


@pytest.mark.parametrize("boundary", ["eos", "length"])
def test_v3_keeps_eos_and_length_with_invalid_thinking_traceable(boundary):
    action_text = "Unclosed reasoning contains </answer> but no think close."
    record = _invalid_eval_record(action_text,
                                  boundary,
                                  raw_clipped=boundary == "length")

    prepared = prepare_trace_record(record, "eval", "unit", "unfinished")
    generation = prepared["raw_generations"][0]
    assert generation["boundary"] == boundary
    assert generation["raw_clipped"] is (boundary == "length")


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


def test_reader_accepts_historical_v1_without_rewriting_it(tmp_path):
    record = _v1_eval_record()
    record.update({
        "schema": TRACE_SCHEMA,
        "schema_version": 1,
        "record_type": "eval",
        "record_id": "trace:historical-v1",
        "run_id": "historical",
        "stage": "control",
    })
    path = tmp_path / "historical.jsonl"
    path.write_bytes((json.dumps(record,
                                 ensure_ascii=False,
                                 sort_keys=True,
                                 separators=(",", ":")) + "\n").encode("utf-8"))

    inspected = inspect_trace_jsonl(path, 1, "eval", "historical",
                                    "control", schema_version=1)

    assert inspected["rows"] == 1
    assert json.loads(path.read_text(encoding="utf-8"))["max_searches"] == 4


def test_writer_explicitly_publishes_v1_record_and_manifest(tmp_path):
    trace_path = tmp_path / "legacy_eval_predictions.jsonl"
    writer = TraceJsonlWriter(
        trace_path,
        record_type="eval",
        expected_rows=1,
        run_id="legacy-eval",
        stage="legacy-control",
        schema_version=1,
    )
    writer.append(_v1_eval_record())

    manifest = writer.finalize()
    record = json.loads(trace_path.read_text(encoding="utf-8"))

    assert record["schema_version"] == 1
    assert record["max_searches"] == 4
    assert "raw_generations" not in record
    assert manifest["artifact"]["record_schema_version"] == 1
    assert verify_trace_manifest(writer.manifest_path) == manifest


def test_writer_rejects_unpublishable_or_mixed_schema_versions(tmp_path):
    with pytest.raises(ValueError, match="explicitly publishable"):
        TraceJsonlWriter(
            tmp_path / "v2.jsonl",
            record_type="eval",
            expected_rows=1,
            run_id="v2",
            stage="legacy",
            schema_version=2,
        )

    writer = TraceJsonlWriter(
        tmp_path / "mixed.jsonl",
        record_type="eval",
        expected_rows=1,
        run_id="mixed",
        stage="legacy",
        schema_version=1,
    )
    with pytest.raises(ValueError, match="v3-only audit fields"):
        writer.append(_eval_record())
    writer.close()


def test_v3_rejects_legacy_max_searches_field(tmp_path):
    writer = TraceJsonlWriter(
        tmp_path / "eval_predictions.jsonl",
        record_type="eval",
        expected_rows=1,
        run_id="v3-only",
        stage="control",
    )
    record = _eval_record()
    record["max_searches"] = 4

    with pytest.raises(ValueError, match="must not contain max_searches"):
        writer.append(record)
    writer.close()


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
