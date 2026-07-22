from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE_PATH = Path(
    __file__).resolve().parents[1] / "scripts" / "autodl" / "probe_analysis.py"
SPEC = importlib.util.spec_from_file_location("autodl_probe_analysis",
                                              MODULE_PATH)
assert SPEC and SPEC.loader
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)

DIGEST = "a" * 64


def _documents(sample_index: int, suffix: str,
               title: str) -> list[dict[str, object]]:
    return [{
        "document_id": f"{sample_index}-{suffix}",
        "score": 2.0,
        "document": {
            "title": title,
            "contents": f"{title}\nEvidence for sample {sample_index}.",
        },
    }]


def _turn(
        index: int,
        action: str,
        *,
        query: str | None = None,
        answer: str | None = None,
        observation: str | None = None,
        documents: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {
        "turn": index,
        "think": "Use the available evidence.",
        "action": action,
        "search_query": query,
        "answer": answer,
        "observation": observation,
        "invalid_text": [],
        "valid_action": action != "invalid",
        "generation_turn": index,
        "environment_action": action in ("search", "answer"),
        "synthetic_environment_action": False,
        "retrieved_docs": documents or [],
        "retrieval_executed": action == "search",
    }


def _record(sample_index: int,
            slot: int,
            *,
            em: int,
            searches: int,
            group_uid: bool = True) -> dict[str, object]:
    sample_id = f"hotpotqa:train:{sample_index}"
    question = f"Question for sample {sample_index}?"
    answer = f"Gold {sample_index}"
    titles = (f"Alpha {sample_index}", f"Beta {sample_index}")
    events: list[dict[str, object]] = []
    turns: list[dict[str, object]] = []
    raw_parts: list[str] = []
    for search_index in range(searches):
        if search_index == 0:
            query = f"initial query topic {sample_index}"
            documents = _documents(sample_index, "a", titles[0])
            observation = f"{titles[0]} contains a bridge clue."
        elif search_index == 1:
            query = f"bridge followup evidence {sample_index}"
            documents = _documents(sample_index, "b", titles[1])
            observation = f"{titles[1]} states that the answer is {answer}."
        else:
            query = f"additional distinct lookup {search_index} {sample_index}"
            documents = _documents(sample_index, f"x{search_index}",
                                   f"Extra {sample_index} {search_index}")
            observation = f"Extra observation {search_index}."
        events.append({
            "turn": search_index,
            "query": query,
            "documents": documents,
            "observation": observation,
            "visible_observation": observation,
        })
        turns.append(
            _turn(search_index,
                  "search",
                  query=query,
                  observation=observation,
                  documents=documents))
        raw_parts.append(f"<think>search</think><search>{query}</search>"
                         f"<information>{observation}</information>")
    extracted = answer if em else f"Wrong {sample_index}"
    turns.append(_turn(len(turns), "answer", answer=extracted))
    raw_parts.append(f"<think>answer</think><answer>{extracted}</answer>")
    record: dict[str, object] = {
        "schema": "search-r1.trajectory",
        "schema_version": 1,
        "record_type": "eval",
        "record_id": f"trace-{sample_index}-{slot}",
        "run_id": "base-probe",
        "stage": "grouped-probe-base",
        "sample_id": sample_id,
        "source_index": sample_index,
        "source_split": "train",
        "data_source": "hotpotqa",
        "question": question,
        "gold_answers": [answer],
        "raw_trajectory": "".join(raw_parts),
        "turns": turns,
        "extracted_answer": extracted,
        "em": em,
        "executed_search_count": searches,
        "posthoc_utility": em - 0.10 * searches / 4,
        "response_tokens": 80 + searches,
        "response_clipped": False,
        "turns_used": len(turns),
        "invalid_action_count": 0,
        "checkpoint_digest": DIGEST,
        "group_slot": slot,
        "group_size": 5,
        "max_searches": 4,
        "retrieval_events": events,
    }
    if group_uid:
        record["group_uid"] = f"probe:{sample_index}"
    return record


def _catalog() -> list[dict[str, object]]:
    records = []
    for sample_index in range(64):
        category = "comparison" if sample_index < 16 else "bridge"
        records.append({
            "sample_id":
            f"hotpotqa:train:{sample_index}",
            "source_id":
            f"hotpotqa:train:{sample_index}",
            "data_source":
            "hotpotqa",
            "source_split":
            "train",
            "source_index":
            sample_index,
            "output_split":
            "val",
            "category":
            category,
            "question":
            f"Question for sample {sample_index}?",
            "golden_answers": [f"Gold {sample_index}"],
            "supporting_titles":
            [f"Alpha {sample_index}", f"Beta {sample_index}"],
        })
    return records


def _passing_records(*, group_uid: bool = True) -> list[dict[str, object]]:
    records = []
    for sample_index in range(64):
        for slot in range(5):
            qualifies = sample_index < 8 and slot < 2
            records.append(
                _record(
                    sample_index,
                    slot,
                    em=int(qualifies),
                    searches=2 if qualifies else 1,
                    group_uid=group_uid,
                ))
    return records


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("".join(
        json.dumps(
            record, ensure_ascii=False, sort_keys=True, separators=(",",
                                                                    ":")) +
        "\n" for record in records),
                    encoding="utf-8")


def _fixture(tmp_path: Path, records: list[dict[str,
                                                object]]) -> tuple[Path, Path]:
    trace = tmp_path / "eval_predictions.jsonl"
    catalog = tmp_path / "catalog.jsonl"
    _write_jsonl(trace, records)
    _write_jsonl(catalog, _catalog())
    return trace, catalog


def _args(trace: Path,
          catalog: Path,
          output: Path,
          digest: str = DIGEST) -> SimpleNamespace:
    return SimpleNamespace(
        trace=trace,
        catalog=catalog,
        expected_checkpoint_digest=digest,
        output_dir=output,
        fail_on_no_go=False,
    )


def _find(records: list[dict[str, object]], sample_index: int,
          slot: int) -> dict[str, object]:
    return next(record for record in records
                if record["source_index"] == sample_index
                and record["group_slot"] == slot)


def _add_invalid_turn(record: dict[str, object]) -> None:
    turns = record["turns"]
    assert isinstance(turns, list)
    turns.append(_turn(len(turns), "invalid"))
    record["turns_used"] = len(turns)
    record["invalid_action_count"] = 1


def test_go_outputs_strict_metrics_and_traceable_artifacts(
        tmp_path: Path) -> None:
    trace, catalog = _fixture(tmp_path, _passing_records())
    output = tmp_path / "analysis"

    summary = PROBE.analyze(_args(trace, catalog, output))

    assert summary["decision"] == "GO"
    assert summary["input"]["group_key_mode"] == "group_uid"
    assert summary["overall"]["valid_correct_multi_search_count"] == 16
    assert summary["overall"]["covered_question_count"] == 8
    assert summary["overall"]["learnable_group_count"] == 8
    assert summary["overall"]["clipped_ratio"] == 0
    assert summary["overall"]["invalid_action_ratio"] == 0
    assert summary["overall"]["qualifying_evidence_branches"] == {
        "answer": 0,
        "supporting_title": 0,
        "answer_and_supporting_title": 16,
    }
    assert summary["by_category"]["comparison"]["question_count"] == 16
    assert summary["by_category"]["bridge"]["question_count"] == 48
    assert {path.name for path in output.iterdir()} == PROBE.OUTPUT_FILES
    assert len((output / "per_trajectory.jsonl").read_text(
        encoding="utf-8").splitlines()) == 320
    question_rows = [
        json.loads(line) for line in (output / "per_question.jsonl").read_text(
            encoding="utf-8").splitlines()
    ]
    assert len(question_rows) == 64
    assert question_rows[0]["slots"] == [0, 1, 2, 3, 4]
    assert "query-token Jaccard below 0.8" in (output /
                                               "summary.md").read_text(
                                                   encoding="utf-8")


def test_sample_identity_fallback_groups_without_group_uid(
        tmp_path: Path) -> None:
    trace, catalog = _fixture(tmp_path, _passing_records(group_uid=False))

    summary = PROBE.analyze(_args(trace, catalog, tmp_path / "analysis"))

    assert summary["decision"] == "GO"
    assert summary["input"]["group_key_mode"] == "sample_id"


def test_no_go_is_exit_zero_unless_explicitly_requested(
        tmp_path: Path) -> None:
    records = [
        _record(sample_index, slot, em=0, searches=1)
        for sample_index in range(64) for slot in range(5)
    ]
    trace, catalog = _fixture(tmp_path, records)
    common = [
        "--trace",
        str(trace),
        "--catalog",
        str(catalog),
        "--expected-checkpoint-digest",
        DIGEST,
    ]

    assert PROBE.main([*common, "--output-dir", str(tmp_path / "normal")]) == 0
    assert json.loads(
        (tmp_path / "normal" /
         "go_no_go.json").read_text(encoding="utf-8"))["decision"] == "NO-GO"
    assert PROBE.main([
        *common,
        "--output-dir",
        str(tmp_path / "strict"),
        "--fail-on-no-go",
    ]) == 2


def test_near_miss_diagnostic_preserves_strict_no_go(tmp_path: Path) -> None:
    records = []
    for sample_index in range(64):
        for slot in range(5):
            is_near_miss = sample_index < 8 and slot < 2
            record = _record(sample_index,
                             slot,
                             em=0,
                             searches=2 if is_near_miss else 1)
            if is_near_miss:
                expanded = f"Gold {sample_index} expanded"
                record["extracted_answer"] = expanded
                record["turns"][-1]["answer"] = expanded
                record["raw_trajectory"] = record["raw_trajectory"].replace(
                    f"<answer>Wrong {sample_index}</answer>",
                    f"<answer>{expanded}</answer>")
            records.append(record)
    trace, catalog = _fixture(tmp_path, records)
    output = tmp_path / "analysis"

    summary = PROBE.analyze(_args(trace, catalog, output))

    assert summary["decision"] == "NO-GO"
    assert summary["overall"]["valid_correct_multi_search_count"] == 0
    assert summary["overall"]["near_miss_count"] == 16
    assert summary["overall"]["near_miss_covered_question_count"] == 8
    assert summary["overall"]["near_miss_cover_em_count"] == 16
    diagnostic = summary["near_miss_diagnostic"]
    assert diagnostic["threshold_met"] is True
    assert diagnostic["affects_go_no_go"] is False
    assert diagnostic[
        "diagnosis"] == "retrieval_chain_present_review_answer_extraction"
    assert diagnostic["by_category"] == {
        "comparison": 16,
        "bridge": 0,
    }
    assert len(diagnostic["examples"]) == 5
    assert diagnostic["examples"][0]["raw_trajectory"]
    decision = json.loads(
        (output / "go_no_go.json").read_text(encoding="utf-8"))
    assert decision["decision"] == "NO-GO"
    assert decision["near_miss_count"] == 16
    assert decision["near_miss_threshold_met"] is True
    rows = [
        json.loads(line)
        for line in (output / "per_trajectory.jsonl").read_text(
            encoding="utf-8").splitlines()
    ]
    assert sum(row["near_miss"] for row in rows) == 16
    assert "Near-Miss Examples" in (output /
                                    "summary.md").read_text(encoding="utf-8")


def test_learnable_requires_correct_multisearch_and_clean_wrong_contrast(
        tmp_path: Path) -> None:
    records = _passing_records()
    for sample_index in range(8):
        for slot in range(2, 5):
            record = _find(records, sample_index, slot)
            record["em"] = 1
            record["extracted_answer"] = f"Gold {sample_index}"
            record["posthoc_utility"] = 0.975
    trace, catalog = _fixture(tmp_path, records)

    summary = PROBE.analyze(_args(trace, catalog, tmp_path / "analysis"))

    assert summary["overall"]["valid_correct_multi_search_count"] == 16
    assert summary["overall"]["covered_question_count"] == 8
    assert summary["overall"]["learnable_group_count"] == 0
    assert summary["overall"]["cost_contrast_group_count"] == 8
    assert summary["decision"] == "NO-GO"
    assert summary["go_no_go"]["failed_criteria"] == ["learnable_group_count"]


def test_near_duplicate_new_document_and_new_evidence_are_all_required(
        tmp_path: Path) -> None:
    records = _passing_records()

    near_duplicate = _find(records, 0, 0)
    near_duplicate["retrieval_events"][1][
        "query"] = "initial query topic 0 results"
    near_duplicate["turns"][1][
        "search_query"] = "initial query topic 0 results"

    no_new_document = _find(records, 0, 1)
    first_document = no_new_document["retrieval_events"][0]["documents"]
    no_new_document["retrieval_events"][1]["documents"] = first_document
    no_new_document["turns"][1]["retrieved_docs"] = first_document

    no_new_evidence = _find(records, 1, 0)
    unrelated = _documents(1, "unrelated", "Unrelated title")
    no_new_evidence["retrieval_events"][1]["documents"] = unrelated
    no_new_evidence["retrieval_events"][1][
        "observation"] = "Unrelated title has no useful fact."
    no_new_evidence["retrieval_events"][1][
        "visible_observation"] = "Unrelated title has no useful fact."
    no_new_evidence["turns"][1]["retrieved_docs"] = unrelated
    no_new_evidence["turns"][1][
        "observation"] = "Unrelated title has no useful fact."

    trace, catalog = _fixture(tmp_path, records)
    summary = PROBE.analyze(_args(trace, catalog, tmp_path / "analysis"))

    failures = summary["overall"]["qualification_failure_counts"]
    assert failures["near_duplicate_or_empty_query"] == 1
    assert failures["no_new_document"] == 1
    assert failures["no_new_supporting_or_answer_evidence"] == 1
    assert summary["overall"]["valid_correct_multi_search_count"] == 13
    assert summary["overall"]["covered_question_count"] == 7
    assert summary["decision"] == "NO-GO"


def test_evidence_after_visible_384_token_window_does_not_qualify(
        tmp_path: Path) -> None:
    records = _passing_records()
    record = _find(records, 0, 0)
    visible = " ".join(f"filler-{index}" for index in range(384))
    hidden = "Beta 0 states that the answer is Gold 0."
    record["retrieval_events"][1]["observation"] = f"{visible} {hidden}"
    record["retrieval_events"][1]["visible_observation"] = visible
    record["turns"][1]["observation"] = visible
    trace, catalog = _fixture(tmp_path, records)

    summary = PROBE.analyze(_args(trace, catalog, tmp_path / "analysis"))

    failures = summary["overall"]["qualification_failure_counts"]
    assert failures["no_new_document"] == 1
    assert failures["no_new_supporting_or_answer_evidence"] == 1
    assert summary["overall"]["valid_correct_multi_search_count"] == 15
    assert summary["decision"] == "NO-GO"


def test_punctuation_only_query_is_scientific_failure_not_analysis_error(
        tmp_path: Path) -> None:
    records = _passing_records()
    empty_query = _find(records, 0, 0)
    empty_query["retrieval_events"][1]["query"] = "... !!!"
    empty_query["turns"][1]["search_query"] = "... !!!"
    trace, catalog = _fixture(tmp_path, records)

    summary = PROBE.analyze(_args(trace, catalog, tmp_path / "analysis"))

    assert summary["overall"]["qualification_failure_counts"][
        "near_duplicate_or_empty_query"] == 1
    rows = [
        json.loads(line)
        for line in (tmp_path / "analysis" / "per_trajectory.jsonl").read_text(
            encoding="utf-8").splitlines()
    ]
    row = next(
        item for item in rows
        if item["sample_id"] == "hotpotqa:train:0" and item["group_slot"] == 0)
    assert row["query_token_jaccard"] == 1.0
    assert row["valid_correct_multi_search"] is False


def test_clipped_and_invalid_rates_use_trajectory_denominator(
        tmp_path: Path) -> None:
    records = _passing_records()
    for record in records[:17]:
        record["response_clipped"] = True
    for record in records[17:34]:
        _add_invalid_turn(record)
    trace, catalog = _fixture(tmp_path, records)

    summary = PROBE.analyze(_args(trace, catalog, tmp_path / "analysis"))

    assert summary["overall"]["clipped_ratio"] == pytest.approx(17 / 320)
    assert summary["overall"]["invalid_action_ratio"] == pytest.approx(17 /
                                                                       320)
    assert not summary["go_no_go"]["criteria"]["clipped_ratio"]["passed"]
    assert not summary["go_no_go"]["criteria"]["invalid_action_ratio"]["passed"]


@pytest.mark.parametrize("mutation", [
    "missing_document_id",
    "missing_visible_observation",
    "duplicate_slot",
    "mixed_group_uid",
    "wrong_checkpoint",
    "missing_row",
    "catalog_question_mismatch",
])
def test_structural_errors_are_infrastructure_failures(tmp_path: Path,
                                                       mutation: str) -> None:
    records = _passing_records()
    catalog_records = _catalog()
    if mutation == "missing_document_id":
        records[0]["retrieval_events"][0]["documents"][0].pop("document_id")
    elif mutation == "missing_visible_observation":
        records[0]["retrieval_events"][0].pop("visible_observation")
    elif mutation == "duplicate_slot":
        _find(records, 0, 1)["group_slot"] = 0
    elif mutation == "mixed_group_uid":
        records[0].pop("group_uid")
    elif mutation == "wrong_checkpoint":
        records[0]["checkpoint_digest"] = "b" * 64
    elif mutation == "missing_row":
        records.pop()
    elif mutation == "catalog_question_mismatch":
        catalog_records[0]["question"] = "A different question?"

    trace = tmp_path / "trace.jsonl"
    catalog = tmp_path / "catalog.jsonl"
    _write_jsonl(trace, records)
    _write_jsonl(catalog, catalog_records)

    assert PROBE.main([
        "--trace",
        str(trace),
        "--catalog",
        str(catalog),
        "--expected-checkpoint-digest",
        DIGEST,
        "--output-dir",
        str(tmp_path / "analysis"),
    ]) == 1
    assert not (tmp_path / "analysis").exists()
