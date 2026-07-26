import json
from collections import Counter
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
from types import ModuleType

import pytest

from scripts.data_process import search_mix
from search_r1.llm_agent.tool_protocol import (QWEN35_REASONING_CONTINUATION,
                                               parse_action)

PARQUET_FIELDS = {
    "data_source",
    "prompt",
    "ability",
    "reward_model",
    "extra_info",
}
SMALL_QUOTAS = {
    "single": {
        "train": 2,
        "val": 1
    },
    "comparison": {
        "train": 2,
        "val": 2
    },
    "bridge": {
        "train": 1,
        "val": 1
    },
}
SMALL_RETRIEVAL_TARGETS = {
    "comparison": 4,
    "bridge": 2,
}


class CharacterTokenizer:
    """Keep the visible-observation tests independent of transformers."""

    def __call__(self,
                 text,
                 add_special_tokens=False,
                 return_offsets_mapping=False):
        del add_special_tokens
        result = {"input_ids": [ord(character) for character in text]}
        if return_offsets_mapping:
            result["offset_mapping"] = [(index, index + 1)
                                        for index in range(len(text))]
        return result

    def apply_chat_template(self,
                            messages,
                            tools=None,
                            enable_thinking=False,
                            add_generation_prompt=True,
                            tokenize=False,
                            return_dict=False):
        del tools, enable_thinking, add_generation_prompt, return_dict
        rendered = "chat|" + "|".join(
            f"{message['role']}:{message['content']}" for message in messages)
        if tokenize:
            return [ord(character) for character in rendered]
        return rendered

    def decode(self, input_ids, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(chr(token) for token in input_ids)


def _hotpot_row(category, index=0):
    first_title = f"Alpha {index}"
    second_title = f"Beta {index}"
    first_fact = f"{first_title} points readers to {second_title}."
    answer = f"Answer {index}"
    second_fact = f"{second_title} identifies {answer}."
    if category == "comparison":
        question = f"How do {first_title} and {second_title} compare"
    else:
        question = f"Who is identified through {first_title}"
    return {
        "question": question,
        "golden_answers": [answer],
        "metadata": {
            "type": category,
            "level": "hard",
            "supporting_facts": {
                "title": [first_title, second_title],
                "sent_id": [0, 0],
            },
            "context": {
                "title": [first_title, second_title],
                "sentences": [[first_fact], [second_fact]],
            },
        },
    }


def _hit(document_id, title, body, rank=1):
    return {
        "rank": rank,
        "document_id": str(document_id),
        "title": title,
        "score": float(10 - rank),
        "contents": f'"{title}"\n{body}',
    }


def _single_evidence(index=0, include_answer=True):
    answer = f"Northville {index}"
    body = (f"The archive lists {answer} as the location."
            if include_answer else "The archive omits the location.")
    return {
        "source_id": f"nq:train:{index}",
        "data_source": "nq",
        "source_split": "train",
        "source_index": index,
        "question": f"Where was subject {index} born",
        "golden_answers": [answer],
        "category": "single",
        "first_query": f"Where was subject {index} born",
        "first_results": [_hit(f"nq-{index}", f"Subject {index}", body)],
        "second_query": None,
        "second_results": [],
    }


def _multihop_evidence(category, index=0):
    candidate = search_mix._candidate("hotpotqa", index,
                                      _hotpot_row(category, index))
    first_title, second_title = candidate["supporting_titles"]
    first_fact = candidate["supporting_facts"][first_title][0]
    second_fact = candidate["supporting_facts"][second_title][0]
    candidate.update({
        "first_query":
        candidate["question"],
        "first_results": [_hit(f"first-{index}", first_title, first_fact)],
        "first_supporting_title":
        first_title,
        "second_query":
        second_title,
        "second_query_derivable_from":
        ("question" if category == "comparison" else "first_observation"),
        "second_results": [_hit(f"second-{index}", second_title, second_fact)],
        "new_document_count":
        1,
    })
    return candidate


def _small_evidence():
    evidence = [_single_evidence(index) for index in range(3)]
    evidence.extend(
        _multihop_evidence("comparison", 100 + index) for index in range(4))
    evidence.extend(
        _multihop_evidence("bridge", 200 + index) for index in range(2))
    return evidence


def test_registered_mix_uses_hotpot_majority_with_bridge_weighting():
    assert search_mix.SCHEMA_VERSION == 2
    assert search_mix.QUOTAS == {
        "single": {
            "train": 192,
            "val": 64
        },
        "comparison": {
            "train": 56,
            "val": 16
        },
        "bridge": {
            "train": 264,
            "val": 48
        },
    }
    assert search_mix.RETRIEVAL_TARGETS == {
        "comparison": 240,
        "bridge": 144,
    }
    assert sum(category["train"]
               for category in search_mix.QUOTAS.values()) == 512
    assert sum(category["val"]
               for category in search_mix.QUOTAS.values()) == 128

    native = search_mix.parse_args([
        "materialize-native", "--source-manifest", "source/manifest.json",
        "--output-dir", "native", "--model-dir", "model",
        "--no-reselection"
    ])
    assert native.source_manifest == Path("source/manifest.json")
    assert native.output_dir == Path("native")
    assert native.no_reselection is True
    verify = search_mix.parse_args([
        "verify", "--manifest", "native/manifest.json", "--model-dir",
        "model", "--expected-tool-protocol", "qwen35_native",
        "--source-manifest", "source/manifest.json"
    ])
    assert verify.expected_tool_protocol == search_mix.QWEN35_NATIVE
    assert verify.source_manifest == Path("source/manifest.json")
    verify_no_reselection = search_mix.parse_args([
        "verify-no-reselection", "--manifest", "native/manifest.json",
        "--model-dir", "model", "--expected-tool-protocol",
        "qwen35_native", "--source-manifest", "source/manifest.json"
    ])
    assert verify_no_reselection.command == "verify-no-reselection"
    assert verify_no_reselection.source_manifest == Path(
        "source/manifest.json")
    visibility = search_mix.parse_args([
        "validate-native-evidence", "--manifest", "native/manifest.json",
        "--source-manifest", "source/manifest.json", "--model-dir", "model"
    ])
    assert visibility.command == "validate-native-evidence"
    assert visibility.source_manifest == Path("source/manifest.json")
    assert search_mix.NATIVE_EVAL_FILES == {
        "nq_test_eval": "nq_test_128_native_v3.parquet",
        "multihop_eval": "multihop_eval_256_native_v3.parquet",
    }


def test_native_prompt_contract_registers_v3_original_alignment():
    contract = search_mix.prompt_contract(search_mix.QWEN35_NATIVE)
    messages = search_mix.make_prompt("Who wrote Hamlet?",
                                      search_mix.QWEN35_NATIVE)

    assert contract[
        "prompt_version"] == "qwen35-native-search-v3-original-aligned"
    assert contract["tool_protocol"] == search_mix.QWEN35_NATIVE
    assert messages == search_mix.qwen35_messages("Who wrote Hamlet?")
    assert [message["role"] for message in messages] == ["user"]
    prompt = messages[0]["content"]
    assert "<answer> Beijing </answer>" in prompt
    assert "at least once" not in prompt
    assert "at most four" not in prompt


def test_native_eval_materializations_bind_sealed_sources(tmp_path,
                                                          monkeypatch):
    from scripts.data_process import multihop_search_gate, nq_small

    nq_dir = tmp_path / "nq"
    multihop_dir = tmp_path / "multihop"
    nq_dir.mkdir()
    multihop_dir.mkdir()
    nq_manifest = nq_dir / "manifest.json"
    nq_file = nq_dir / "test_128.parquet"
    multihop_manifest = multihop_dir / "manifest.json"
    multihop_catalog = multihop_dir / "catalog.jsonl"
    multihop_file = multihop_dir / "eval_256.parquet"
    for path in (nq_manifest, nq_file, multihop_manifest, multihop_catalog,
                 multihop_file):
        path.write_text(path.name, encoding="utf-8")
    nq_rows = [{"prompt": search_mix.qwen35_messages("NQ?")}]
    multihop_rows = [{"prompt": search_mix.qwen35_messages("Multi?")}]
    monkeypatch.setattr(
        nq_small, "load_native_test_records",
        lambda manifest, parquet: (nq_rows, ["nq:test:1"], Path(parquet)))
    monkeypatch.setattr(
        multihop_search_gate, "load_native_eval_records",
        lambda manifest, catalog: (multihop_rows, ["hotpotqa:test:2"],
                                   multihop_file))

    loaded = search_mix._native_eval_materializations([multihop_catalog],
                                                       [nq_file])

    assert loaded["nq_test_eval"]["sample_ids"] == ["nq:test:1"]
    assert loaded["multihop_eval"]["sample_ids"] == ["hotpotqa:test:2"]
    contract = search_mix._native_eval_source_contract(loaded)
    assert contract["nq_test_eval"]["source_manifest_sha256"] == (
        search_mix.sha256_file(nq_manifest))
    assert contract["multihop_eval"]["source_artifact_file"] == (
        "eval_256.parquet")


def test_no_reselection_cli_dispatches_explicit_mode(tmp_path, monkeypatch):
    calls = []

    def record_verify(*args, reselect_catalog=True, **kwargs):
        del args, kwargs
        calls.append(("verify", reselect_catalog))

    def record_materialize(source_manifest,
                           output_dir,
                           model_dir,
                           eval_catalogs=(),
                           eval_parquets=(),
                           reselect_catalog=True):
        del source_manifest, model_dir, eval_catalogs, eval_parquets
        calls.append(("materialize-native", reselect_catalog))
        return output_dir / search_mix.MANIFEST_FILE

    monkeypatch.setattr(search_mix, "verify_manifest", record_verify)
    monkeypatch.setattr(search_mix, "materialize_native", record_materialize)
    assert search_mix.main([
        "verify-no-reselection", "--manifest",
        str(tmp_path / "native" / "manifest.json"), "--model-dir",
        str(tmp_path / "model")
    ]) == 0
    assert search_mix.main([
        "materialize-native", "--source-manifest",
        str(tmp_path / "source" / "manifest.json"), "--output-dir",
        str(tmp_path / "native"), "--model-dir",
        str(tmp_path / "model"), "--no-reselection"
    ]) == 0
    assert calls == [("verify", False), ("materialize-native", False)]


def test_candidate_accepts_nq_comparison_and_bridge():
    nq = search_mix._candidate(
        "nq", 7, {
            "question": "Where was the seventh subject born",
            "golden_answers": ["Northville"],
        })
    comparison = search_mix._candidate("hotpotqa", 8,
                                       _hotpot_row("comparison", 8))
    bridge = search_mix._candidate("hotpotqa", 9, _hotpot_row("bridge", 9))

    assert nq["category"] == "single"
    assert comparison["category"] == "comparison"
    assert bridge["category"] == "bridge"
    assert comparison["supporting_titles"] == ["Alpha 8", "Beta 8"]
    assert bridge["supporting_titles"] == ["Alpha 9", "Beta 9"]


def test_retrieve_evidence_scans_all_capped_candidates(tmp_path, monkeypatch):
    comparison = [
        search_mix._candidate("hotpotqa", index,
                              _hotpot_row("comparison", index))
        for index in (10, 11)
    ]
    bridge = [search_mix._candidate("hotpotqa", 20, _hotpot_row("bridge", 20))]

    def fake_collect(_rows, source, _limits):
        if source == "nq":
            return {
                "single": []
            }, Counter(), {
                "source_rows": 0,
                "candidates_after_prescreen": {
                    "single": 0
                },
                "candidates_after_cap": {
                    "single": 0
                },
            }
        return {
            "comparison": comparison,
            "bridge": bridge
        }, Counter(), {
            "source_rows": len(comparison) + len(bridge),
            "candidates_after_prescreen": {
                "comparison": len(comparison),
                "bridge": len(bridge),
            },
            "candidates_after_cap": {
                "comparison": len(comparison),
                "bridge": len(bridge),
            },
        }

    results = {}
    for candidate in comparison + bridge:
        first_title, second_title = candidate["supporting_titles"]
        first_fact = candidate["supporting_facts"][first_title][0]
        second_fact = candidate["supporting_facts"][second_title][0]
        results[candidate["question"]] = _hit(
            f"first-{candidate['source_index']}", first_title, first_fact)
        results[second_title] = _hit(f"second-{candidate['source_index']}",
                                     second_title, second_fact)

    calls = []

    class FakeRetriever:

        def __init__(self, *_args, **_kwargs):
            pass

        def search(self, query, topk, return_scores):
            calls.append((query, topk, return_scores))
            hit = results[query]
            return [{
                "document": {
                    "title": hit["title"],
                    "contents": hit["contents"],
                },
                "document_id": hit["document_id"],
                "score": hit["score"],
            }]

    from search_r1.search import bm25_server

    monkeypatch.setattr(search_mix, "JsonlRows", lambda _path: object())
    monkeypatch.setattr(search_mix, "verify_source",
                        lambda _local_dir, source: tmp_path / source)
    monkeypatch.setattr(search_mix, "collect_candidates", fake_collect)
    monkeypatch.setattr(search_mix, "RETRIEVAL_TARGETS", {
        "comparison": 1,
        "bridge": 1,
    })
    monkeypatch.setattr(bm25_server, "BM25Retriever", FakeRetriever)

    search_mix.retrieve_evidence(tmp_path, tmp_path / "index",
                                 tmp_path / "corpus", tmp_path / "offsets")

    evidence = search_mix.read_canonical_jsonl(tmp_path /
                                               search_mix.EVIDENCE_FILE)
    assert {record["source_id"]
            for record in evidence
            } == {candidate["source_id"]
                  for candidate in comparison + bridge}
    assert {candidate["question"]
            for candidate in comparison}.issubset({call[0]
                                                   for call in calls})


def test_retrieval_quota_failure_writes_structured_funnel(
        tmp_path, monkeypatch):

    def fake_collect(_rows, source, _limits):
        categories = ["single"] if source == "nq" else ["comparison", "bridge"]
        return {
            category: []
            for category in categories
        }, Counter(), {
            "source_rows": 0,
            "candidates_after_prescreen": {
                category: 0
                for category in categories
            },
            "candidates_after_cap": {
                category: 0
                for category in categories
            },
        }

    class FakeRetriever:

        def __init__(self, *_args, **_kwargs):
            pass

        def search(self, *_args, **_kwargs):
            raise AssertionError("No candidates should trigger retrieval")

    from search_r1.search import bm25_server

    monkeypatch.setattr(search_mix, "JsonlRows", lambda _path: object())
    monkeypatch.setattr(search_mix, "verify_source",
                        lambda _local_dir, source: tmp_path / source)
    monkeypatch.setattr(search_mix, "collect_candidates", fake_collect)
    monkeypatch.setattr(search_mix, "RETRIEVAL_TARGETS", {
        "comparison": 1,
        "bridge": 1,
    })
    monkeypatch.setattr(bm25_server, "BM25Retriever", FakeRetriever)

    with pytest.raises(ValueError, match="Retrieval quota shortfall"):
        search_mix.retrieve_evidence(tmp_path, tmp_path / "index",
                                     tmp_path / "corpus", tmp_path / "offsets")

    funnel_path = tmp_path / search_mix.SELECTION_FUNNEL_FILE
    funnel = json.loads(funnel_path.read_bytes())
    assert funnel["status"] == "failed"
    assert funnel["stage"] == "retrieve"
    assert funnel["failure"] == {
        "stage": "retrieve",
        "shortfalls": {
            "bridge": {
                "available": 0,
                "required": 1,
                "missing": 1,
            },
            "comparison": {
                "available": 0,
                "required": 1,
                "missing": 1,
            },
        },
    }
    assert funnel["retrieval"]["candidate_items_queried"] == 0
    assert not (tmp_path / search_mix.EVIDENCE_FILE).exists()
    search_mix._verify_sidecar(funnel_path)


@pytest.mark.parametrize(
    ("source", "row", "message"),
    [
        ("nq", {
            "question": "Is the answer Northville",
            "golden_answers": ["Northville"],
        }, "already visible"),
        ("nq", {
            "question": "Is the statement correct",
            "golden_answers": ["yes"],
        }, "ambiguous short or boolean"),
    ],
)
def test_candidate_rejects_bad_nq(source, row, message):
    with pytest.raises(ValueError, match=message):
        search_mix._candidate(source, 0, row)


def test_candidate_rejects_wrong_hotpot_title_exposure():
    comparison = _hotpot_row("comparison", 1)
    comparison["question"] = "How does Alpha 1 compare"
    bridge = _hotpot_row("bridge", 2)
    bridge["question"] = "How are Alpha 2 and Beta 2 connected"

    with pytest.raises(ValueError, match="Comparison titles must both"):
        search_mix._candidate("hotpotqa", 1, comparison)
    with pytest.raises(ValueError,
                       match="Bridge question must expose exactly one"):
        search_mix._candidate("hotpotqa", 2, bridge)


@pytest.mark.parametrize("category", ["comparison", "bridge"])
def test_candidate_rejects_hotpot_answer_visible_in_question(category):
    row = _hotpot_row(category, 3)
    row["question"] += " Answer 3"

    with pytest.raises(ValueError, match="HotpotQA answer is already visible"):
        search_mix._candidate("hotpotqa", 3, row)


@pytest.mark.parametrize("category", ["single", "comparison", "bridge"])
def test_evaluate_evidence_accepts_all_categories(category):
    record = (_single_evidence()
              if category == "single" else _multihop_evidence(category))

    accepted, reason, audit = search_mix.evaluate_evidence(
        record, CharacterTokenizer())

    assert accepted is True
    assert reason == "accepted"
    if category == "single":
        assert audit["first_gold_seen"] is True
    else:
        assert audit["new_document_count"] == 1
        assert len(audit["first_supporting_titles"]) == 1
        assert len(audit["second_supporting_titles"]) == 1


def test_evaluate_evidence_rejects_nq_without_visible_gold():
    accepted, reason, audit = search_mix.evaluate_evidence(
        _single_evidence(include_answer=False), CharacterTokenizer())

    assert (accepted, reason, audit) == (False, "single_gold_not_visible", {})


def test_evaluate_evidence_rejects_second_search_without_new_document():
    record = _multihop_evidence("comparison")
    record["second_results"][0]["document_id"] = record["first_results"][0][
        "document_id"]

    accepted, reason, _ = search_mix.evaluate_evidence(record,
                                                       CharacterTokenizer())

    assert accepted is False
    assert reason == "no_visible_new_document"


def test_evaluate_evidence_rejects_bridge_query_not_derivable():
    record = _multihop_evidence("bridge")
    first_title = record["supporting_titles"][0]
    replacement = f"{first_title} has an archived record."
    record["supporting_facts"][first_title] = [replacement]
    record["first_results"][0]["contents"] = f'"{first_title}"\n{replacement}'

    accepted, reason, _ = search_mix.evaluate_evidence(record,
                                                       CharacterTokenizer())

    assert accepted is False
    assert reason == "bridge_query_not_visible"


def test_evaluate_evidence_rejects_first_search_with_both_supports():
    record = _multihop_evidence("comparison")
    second_title = record["supporting_titles"][1]
    second_fact = record["supporting_facts"][second_title][0]
    record["first_results"].append(
        _hit("unexpected-second", second_title, second_fact, rank=2))

    accepted, reason, _ = search_mix.evaluate_evidence(record,
                                                       CharacterTokenizer())

    assert accepted is False
    assert reason == "visible_first_support_count_2"


def test_evaluate_evidence_requires_every_first_supporting_fact_visible():
    record = _multihop_evidence("comparison")
    first_title = record["supporting_titles"][0]
    record["supporting_facts"][first_title].append(
        "A second annotated fact is not present in the observation.")

    accepted, reason, _ = search_mix.evaluate_evidence(record,
                                                       CharacterTokenizer())

    assert accepted is False
    assert reason == "first_supporting_fact_not_visible"


def test_evaluate_evidence_rejects_second_fact_leaked_in_first_search():
    record = _multihop_evidence("comparison")
    second_title = record["supporting_titles"][1]
    leaked_fact = record["supporting_facts"][second_title][0]
    record["first_results"][0]["contents"] += f" {leaked_fact}"

    accepted, reason, _ = search_mix.evaluate_evidence(record,
                                                       CharacterTokenizer())

    assert accepted is False
    assert reason == "second_supporting_fact_visible_in_first"


def test_evaluate_evidence_rejects_partial_second_fact_leak():
    record = _multihop_evidence("comparison")
    second_title = record["supporting_titles"][1]
    leaked_fact = "A separate annotated second-side fact leaks early."
    record["supporting_facts"][second_title].append(leaked_fact)
    record["first_results"][0]["contents"] += f" {leaked_fact}"
    record["second_results"][0]["contents"] += f" {leaked_fact}"

    accepted, reason, _ = search_mix.evaluate_evidence(record,
                                                       CharacterTokenizer())

    assert accepted is False
    assert reason == "second_supporting_fact_visible_in_first"


def test_evaluate_evidence_binds_visible_evidence_to_new_supporting_doc():
    record = _multihop_evidence("comparison", 31)
    first_title, second_title = record["supporting_titles"]
    first_fact = record["supporting_facts"][first_title][0]
    second_fact = record["supporting_facts"][second_title][0]
    repeated = _hit("repeated",
                    "Archive",
                    second_fact + " filler" * 80,
                    rank=2)
    record["first_results"] = [
        _hit("first", first_title, first_fact + " padding" * 80),
        repeated,
    ]
    repeated_first = deepcopy(repeated)
    repeated_first["rank"] = 1
    record["second_results"] = [
        repeated_first,
        _hit("new-support", second_title, second_fact, rank=2),
    ]

    accepted, reason, _ = search_mix.evaluate_evidence(record,
                                                       CharacterTokenizer())

    assert accepted is False
    assert reason == "new_supporting_evidence_not_visible"


def test_quota_split_builds_fixed_multihop_probe(monkeypatch):
    monkeypatch.setattr(search_mix, "QUOTAS", deepcopy(SMALL_QUOTAS))
    catalog, rejected, stats = search_mix.select_catalog(
        _small_evidence(), CharacterTokenizer(), excluded_questions=set())

    counts = Counter(
        (record["category"], record["output_split"]) for record in catalog)
    expected = Counter({
        (category, split): count
        for category, quotas in SMALL_QUOTAS.items()
        for split, count in quotas.items()
    })
    probe = search_mix._ordered_records(catalog, "probe")

    assert counts == expected
    assert rejected == Counter()
    assert stats["selected_by_category"] == {
        category: sum(quotas.values())
        for category, quotas in SMALL_QUOTAS.items()
    }
    assert len(probe) == 3
    assert all(record["extra_info"]["split"] == "train" for record in probe)
    expected_probe_indices = {
        record["source_index"]
        for record in catalog
        if record["output_split"] == "val" and record["category"] in (
            "comparison", "bridge")
    }
    assert {record["extra_info"]["index"]
            for record in probe} == expected_probe_indices


def test_native_prompt_changes_only_model_facing_messages(monkeypatch):
    monkeypatch.setattr(search_mix, "QUOTAS", deepcopy(SMALL_QUOTAS))
    catalog, _, _ = search_mix.select_catalog(_small_evidence(),
                                              CharacterTokenizer(),
                                              excluded_questions=set())

    legacy = search_mix._ordered_records(catalog, "probe")
    native = search_mix._ordered_records(catalog, "probe",
                                         search_mix.QWEN35_NATIVE)

    assert len(legacy) == len(native)
    for old, new in zip(legacy, native):
        assert {key: value for key, value in old.items() if key != "prompt"} == {
            key: value for key, value in new.items() if key != "prompt"
        }
        assert [message["role"] for message in new["prompt"]] == ["user"]
        source_index = new["extra_info"]["index"]
        selected = next(item for item in catalog
                        if item["source_index"] == source_index)
        assert new["prompt"] == search_mix.qwen35_messages(
            selected["question"])
        assert "at least once" not in new["prompt"][0]["content"]


def test_native_probe_subsets_use_same_autonomous_probe_prefix(monkeypatch):
    monkeypatch.setattr(search_mix, "QUOTAS", deepcopy(SMALL_QUOTAS))
    catalog, _, _ = search_mix.select_catalog(_small_evidence(),
                                              CharacterTokenizer(),
                                              excluded_questions=set())
    probe = search_mix._ordered_records(catalog, "probe",
                                        search_mix.QWEN35_NATIVE)
    autonomous = search_mix._ordered_records(catalog,
                                             "probe",
                                             search_mix.QWEN35_NATIVE,
                                             limit=2)

    identity = lambda row: (row["data_source"], row["extra_info"]["split"],
                            row["extra_info"]["index"])
    assert [identity(row) for row in autonomous] == [
        identity(row) for row in probe[:2]
    ]
    assert search_mix.NATIVE_PROBE_FILES["probe_autonomous"] == (
        "probe_autonomous_16.parquet", 16)
    assert not any("forced" in label or "forced" in filename
                   for label, (filename, _) in
                   search_mix.NATIVE_PROBE_FILES.items())


def test_output_records_expose_only_five_trainer_fields(monkeypatch):
    monkeypatch.setattr(search_mix, "QUOTAS", deepcopy(SMALL_QUOTAS))
    catalog, _, _ = search_mix.select_catalog(_small_evidence(),
                                              CharacterTokenizer(),
                                              excluded_questions=set())
    private_fields = {
        "category",
        "supporting_titles",
        "supporting_facts",
        "first_documents",
        "second_documents",
        "second_oracle_query",
    }

    for split in ("train", "val", "probe"):
        for record in search_mix._ordered_records(catalog, split):
            assert set(record) == PARQUET_FIELDS
            assert private_fields.isdisjoint(record)
            assert private_fields.isdisjoint(record["extra_info"])


def test_validate_evidence_sources_rejects_pinned_source_field_tampering(
        tmp_path, monkeypatch):
    nq_source = tmp_path / "nq.jsonl"
    nq_source.write_bytes(
        search_mix.canonical_json_bytes({
            "question": "Where was subject 0 born",
            "golden_answers": ["Northville 0"],
        }))
    hotpot_source = tmp_path / "hotpot.jsonl"
    hotpot_source.write_bytes(search_mix.canonical_json_bytes({}))
    sources = {"nq": nq_source, "hotpotqa": hotpot_source}
    monkeypatch.setattr(search_mix, "verify_source",
                        lambda _local_dir, source: sources[source])
    evidence = [_single_evidence(0)]

    search_mix.validate_evidence_sources(tmp_path, evidence)
    tampered = deepcopy(evidence)
    tampered[0]["golden_answers"] = ["A forged answer"]

    with pytest.raises(ValueError, match="differs from pinned source"):
        search_mix.validate_evidence_sources(tmp_path, tampered)


def test_validate_evidence_sources_rejects_pinned_hotpot_fact_tampering(
        tmp_path, monkeypatch):
    nq_source = tmp_path / "nq.jsonl"
    nq_source.write_bytes(search_mix.canonical_json_bytes({}))
    hotpot_source = tmp_path / "hotpot.jsonl"
    hotpot_source.write_bytes(
        search_mix.canonical_json_bytes(_hotpot_row("bridge", 0)))
    sources = {"nq": nq_source, "hotpotqa": hotpot_source}
    monkeypatch.setattr(search_mix, "verify_source",
                        lambda _local_dir, source: sources[source])
    evidence = [_multihop_evidence("bridge", 0)]

    search_mix.validate_evidence_sources(tmp_path, evidence)
    tampered = deepcopy(evidence)
    first_title = tampered[0]["supporting_titles"][0]
    tampered[0]["supporting_facts"][first_title][0] = "A forged fact."

    with pytest.raises(ValueError, match="differs from pinned source"):
        search_mix.validate_evidence_sources(tmp_path, tampered)


def _write_json_rows(path, rows):
    path.write_text(json.dumps(rows,
                               ensure_ascii=True,
                               sort_keys=True,
                               separators=(",", ":")),
                    encoding="utf-8")


def _build_manifest_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(search_mix, "QUOTAS", deepcopy(SMALL_QUOTAS))
    monkeypatch.setattr(search_mix, "RETRIEVAL_TARGETS",
                        deepcopy(SMALL_RETRIEVAL_TARGETS))
    monkeypatch.setattr(search_mix, "verify_source",
                        lambda local_dir, source: Path(local_dir) / source)
    monkeypatch.setattr(search_mix, "validate_evidence_sources",
                        lambda local_dir, evidence: None)

    transformers = ModuleType("transformers")

    class FakeAutoTokenizer:

        @staticmethod
        def from_pretrained(model_dir, local_files_only=True):
            del model_dir, local_files_only
            return CharacterTokenizer()

    transformers.AutoTokenizer = FakeAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    evidence = _small_evidence()
    catalog, rejection_counts, selection_stats = search_mix.select_catalog(
        evidence, CharacterTokenizer(), excluded_questions=set())
    catalog.sort(
        key=lambda item: search_mix.stable_key("catalog", item["source_id"]))

    evidence_path = tmp_path / search_mix.EVIDENCE_FILE
    evidence_path.write_bytes(b"".join(
        search_mix.canonical_json_bytes(record) for record in evidence))
    search_mix.write_digest_sidecar(evidence_path)

    ledger = {
        "schema_version": search_mix.SCHEMA_VERSION,
        "selection_policy": search_mix.SELECTION_POLICY,
        "seed": search_mix.SEED,
        "topk": search_mix.TOPK,
        "candidate_limits": search_mix.CANDIDATE_LIMITS,
        "retrieval_targets": search_mix.RETRIEVAL_TARGETS,
        "candidate_order_sha256": "0" * 64,
        "candidate_queries": len(evidence),
        "evidence_rows": len(evidence),
        "evidence_sha256": search_mix.sha256_file(evidence_path),
        "rejection_counts": {},
        "bm25_revision": search_mix.BM25_REVISION,
        "corpus_revision": search_mix.CORPUS_REVISION,
        "corpus_sha256": search_mix.CORPUS_SHA256,
    }
    ledger_path = tmp_path / search_mix.RETRIEVAL_LEDGER_FILE
    ledger_path.write_bytes(search_mix.canonical_json_bytes(ledger))
    search_mix.write_digest_sidecar(ledger_path)

    retrieval_summary = {
        "source_rows": {
            "nq": 3,
            "hotpotqa": 6,
        },
        "candidates_after_prescreen": {
            "single": 3,
            "comparison": 4,
            "bridge": 2,
        },
        "candidates_after_cap": {
            "single": 3,
            "comparison": 4,
            "bridge": 2,
        },
        "candidates_truncated_by_cap": {
            "single": 0,
            "comparison": 0,
            "bridge": 0,
        },
        "candidate_items_queried": len(evidence),
        "bm25_query_calls": 15,
        "structurally_valid": {
            "single": 3,
            "comparison": 4,
            "bridge": 2,
        },
        "evidence_rows": len(evidence),
        "candidate_order_sha256": "0" * 64,
        "evidence_sha256": search_mix.sha256_file(evidence_path),
        "rejection_counts": {},
    }

    catalog_path = tmp_path / search_mix.CATALOG_FILE
    catalog_path.write_bytes(b"".join(
        search_mix.canonical_json_bytes(record) for record in catalog))
    exclusions_path = tmp_path / search_mix.EXCLUSIONS_FILE
    exclusions_path.write_bytes(
        search_mix.canonical_json_bytes({
            "sources": [],
            "normalized_questions": [],
        }))

    artifacts = {}
    for split, filename in search_mix.OUTPUT_FILES.items():
        records = search_mix._ordered_records(catalog, split)
        path = tmp_path / filename
        _write_json_rows(path, records)
        artifacts[split] = {
            "file":
            filename,
            "rows":
            len(records),
            "sha256":
            search_mix.sha256_file(path),
            "sample_ids": [
                f"{record['data_source']}:{record['extra_info']['split']}:{record['extra_info']['index']}"
                for record in records
            ],
        }
    for label, path in (
        ("catalog", catalog_path),
        ("exclusions", exclusions_path),
        ("retrieval_evidence", evidence_path),
        ("retrieval_ledger", ledger_path),
    ):
        artifacts[label] = {
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": search_mix.sha256_file(path),
        }

    funnel_path = search_mix._write_selection_funnel(
        tmp_path,
        search_mix._selection_funnel_payload(
            "complete",
            "materialize",
            retrieval_summary,
            materialize={
                **selection_stats,
                "evidence_rows": len(evidence),
                "excluded_question_count": 0,
                "rejection_counts": dict(sorted(rejection_counts.items())),
            }))
    artifacts["selection_funnel"] = {
        "file": funnel_path.name,
        "bytes": funnel_path.stat().st_size,
        "sha256": search_mix.sha256_file(funnel_path),
    }

    manifest = {
        "schema_version": search_mix.SCHEMA_VERSION,
        "selection_policy": search_mix.SELECTION_POLICY,
        "seed": search_mix.SEED,
        "sources": {
            "dataset": search_mix.DATASET_NAME,
            "revision": search_mix.DATASET_REVISION,
            "files": search_mix.SOURCE_SPECS,
        },
        "retrieval": {
            "topk": search_mix.TOPK,
            "bm25_revision": search_mix.BM25_REVISION,
            "corpus_revision": search_mix.CORPUS_REVISION,
            "corpus_sha256": search_mix.CORPUS_SHA256,
            "ledger_sha256": search_mix.sha256_file(ledger_path),
        },
        "tokenizer": {
            "revision": search_mix.MODEL_REVISION,
            "max_obs_length": search_mix.MAX_OBS_LENGTH,
        },
        "quotas": deepcopy(SMALL_QUOTAS),
        "materialize_rejection_counts": dict(sorted(rejection_counts.items())),
        "overlap_checks": {
            "passed": True,
            "train_val_questions": 0,
            "train_existing_eval_questions": 0,
            "val_existing_eval_questions": 0,
            "probe_is_hotpot_val_subset": True,
        },
        "artifacts": artifacts,
    }
    manifest_path = tmp_path / search_mix.MANIFEST_FILE
    manifest_path.write_bytes(search_mix.canonical_json_bytes(manifest))
    search_mix.write_digest_sidecar(manifest_path)

    monkeypatch.setattr(
        search_mix,
        "_read_parquet",
        lambda path: json.loads(Path(path).read_text(encoding="utf-8")),
    )
    search_mix.verify_manifest(manifest_path, model_dir)
    return {
        "manifest": manifest_path,
        "catalog": catalog_path,
        "evidence": evidence_path,
        "ledger": ledger_path,
        "funnel": funnel_path,
        "model_dir": model_dir,
    }


def _rewrite_manifest_artifact(manifest_path, label, artifact_path):
    manifest = json.loads(manifest_path.read_bytes())
    manifest["artifacts"][label]["sha256"] = search_mix.sha256_file(
        artifact_path)
    if "bytes" in manifest["artifacts"][label]:
        manifest["artifacts"][label]["bytes"] = artifact_path.stat().st_size
    manifest_path.write_bytes(search_mix.canonical_json_bytes(manifest))
    search_mix.write_digest_sidecar(manifest_path)


def _regular_file_snapshot(root):
    files = (path for path in Path(root).rglob("*")
             if path.is_file() and not path.is_symlink())
    return {
        path.relative_to(root).as_posix(): search_mix.sha256_file(path)
        for path in sorted(files, key=lambda value: value.as_posix())
    }


def _prepare_native_fixture_source(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    paths = _build_manifest_fixture(source_dir, monkeypatch)
    source_manifest_before = paths["manifest"].read_bytes()

    for source in search_mix.SOURCE_SPECS:
        path = search_mix.source_path(source_dir, source)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")

    def verify_fixture_source(local_dir, source):
        path = search_mix.source_path(Path(local_dir), source)
        if not path.is_file() or path.is_symlink():
            raise ValueError("fixture source is missing")
        return path

    monkeypatch.setattr(search_mix, "verify_source", verify_fixture_source)
    monkeypatch.setattr(search_mix, "_atomic_write_parquet",
                        lambda records, path: _write_json_rows(path, records))
    monkeypatch.setattr(
        search_mix, "NATIVE_PROBE_FILES", {
            "probe_g0": ("probe_g0_8.parquet", 1),
            "probe_autonomous": ("probe_autonomous_16.parquet", 2),
        })
    source_snapshot_before = _regular_file_snapshot(source_dir)

    def forbidden_upstream_call(*args, **kwargs):
        del args, kwargs
        raise AssertionError("prompt-only materialization called an upstream stage")

    monkeypatch.setattr(search_mix, "download_sources", forbidden_upstream_call)
    monkeypatch.setattr(search_mix, "retrieve_evidence",
                        forbidden_upstream_call)

    _, replay_steps = _selected_replay_steps(paths)
    _install_fake_retriever(monkeypatch, replay_steps)
    search_mix.replay_selected_retrieval(paths["manifest"],
                                         tmp_path / "index",
                                         tmp_path / "corpus.jsonl",
                                         tmp_path / "corpus.offsets")
    source_snapshot_before = _regular_file_snapshot(source_dir)
    return paths, source_snapshot_before, source_manifest_before


def _build_native_fixture(tmp_path, monkeypatch):
    paths, source_snapshot_before, source_manifest_before = (
        _prepare_native_fixture_source(tmp_path, monkeypatch))

    output_dir = tmp_path / "native"
    manifest_path = search_mix.materialize_native(paths["manifest"],
                                                   output_dir,
                                                   paths["model_dir"])
    assert (_regular_file_snapshot(paths["manifest"].parent) ==
            source_snapshot_before)
    return paths, output_dir, manifest_path, source_manifest_before


def test_materialize_native_refuses_existing_output_directory(tmp_path):
    output_dir = tmp_path / "native"
    output_dir.mkdir()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        search_mix.materialize_native(tmp_path / "source" / "manifest.json",
                                      output_dir, tmp_path / "model")


def test_materialize_native_refuses_output_inside_source(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()

    with pytest.raises(ValueError, match="must not be inside source data"):
        search_mix.materialize_native(source_dir / "manifest.json",
                                      source_dir / "native",
                                      tmp_path / "model")


def test_materialize_native_reselects_source_once_without_generic_materialize(
        tmp_path, monkeypatch):
    paths, source_snapshot, _ = _prepare_native_fixture_source(
        tmp_path, monkeypatch)
    original_select = search_mix.select_catalog
    calls = []

    def recording_select(*args, **kwargs):
        calls.append(1)
        return original_select(*args, **kwargs)

    def forbidden_materialize(*args, **kwargs):
        del args, kwargs
        raise AssertionError("native builder called generic materialize")

    monkeypatch.setattr(search_mix, "select_catalog", recording_select)
    monkeypatch.setattr(search_mix, "materialize", forbidden_materialize)
    output_dir = tmp_path / "native"
    manifest_path = search_mix.materialize_native(paths["manifest"],
                                                   output_dir,
                                                   paths["model_dir"])

    assert manifest_path == output_dir / search_mix.MANIFEST_FILE
    assert len(calls) == 1
    assert (_regular_file_snapshot(paths["manifest"].parent) ==
            source_snapshot)


def test_materialize_native_no_reselection_skips_catalog_selection(
        tmp_path, monkeypatch):
    paths, source_snapshot, _ = _prepare_native_fixture_source(
        tmp_path, monkeypatch)

    def forbidden_select(*args, **kwargs):
        del args, kwargs
        raise AssertionError("no-reselection materialization selected a catalog")

    monkeypatch.setattr(search_mix, "select_catalog", forbidden_select)
    output_dir = tmp_path / "native"
    manifest_path = search_mix.materialize_native(
        paths["manifest"],
        output_dir,
        paths["model_dir"],
        reselect_catalog=False)

    assert manifest_path == output_dir / search_mix.MANIFEST_FILE
    assert (_regular_file_snapshot(paths["manifest"].parent) ==
            source_snapshot)


def test_materialize_native_failure_does_not_publish_output(
        tmp_path, monkeypatch):
    paths, source_snapshot, _ = _prepare_native_fixture_source(
        tmp_path, monkeypatch)
    write_parquet = search_mix._atomic_write_parquet
    calls = []

    def fail_second_write(records, path):
        calls.append(Path(path).name)
        if len(calls) == 2:
            raise RuntimeError("injected native parquet failure")
        write_parquet(records, path)

    monkeypatch.setattr(search_mix, "_atomic_write_parquet",
                        fail_second_write)
    output_dir = tmp_path / "native"
    with pytest.raises(RuntimeError, match="injected native parquet failure"):
        search_mix.materialize_native(paths["manifest"], output_dir,
                                      paths["model_dir"])

    assert not output_dir.exists()
    assert not list(tmp_path.glob(".native.*"))
    assert (_regular_file_snapshot(paths["manifest"].parent) ==
            source_snapshot)


def test_materialize_native_validation_failure_does_not_publish_output(
        tmp_path, monkeypatch):
    paths, source_snapshot, _ = _prepare_native_fixture_source(
        tmp_path, monkeypatch)

    def fail_validation(*args, **kwargs):
        del args, kwargs
        raise ValueError("injected final native validation failure")

    monkeypatch.setattr(search_mix, "verify_manifest", fail_validation)
    output_dir = tmp_path / "native"
    with pytest.raises(ValueError,
                       match="injected final native validation failure"):
        search_mix.materialize_native(paths["manifest"], output_dir,
                                      paths["model_dir"])

    assert not output_dir.exists()
    assert not list(tmp_path.glob(".native.*"))
    assert (_regular_file_snapshot(paths["manifest"].parent) ==
            source_snapshot)


def test_materialize_native_is_prompt_only_and_preserves_source_and_selection(
        tmp_path, monkeypatch):
    paths, output_dir, manifest_path, source_manifest_before = (
        _build_native_fixture(tmp_path, monkeypatch))
    native = search_mix.verify_manifest(
        manifest_path,
        paths["model_dir"],
        expected_tool_protocol=search_mix.QWEN35_NATIVE,
        source_manifest=paths["manifest"])
    legacy = json.loads(paths["manifest"].read_bytes())

    assert paths["manifest"].read_bytes() == source_manifest_before
    assert native["schema_version"] == search_mix.MATERIALIZED_SCHEMA_VERSION
    assert native["prompt_contract"] == search_mix.prompt_contract(
        search_mix.QWEN35_NATIVE)
    assert native["tokenizer"] == {
        "revision": search_mix.MODEL_REVISION,
        "selection_observation_length": 384,
        "rollout_observation_length": 500,
    }
    assert native["evaluation_sources"] == {}
    assert native["derived_from"] == {
        "source_manifest_sha256": search_mix.sha256_file(paths["manifest"]),
        "source_catalog_sha256": search_mix.sha256_file(paths["catalog"]),
    }
    assert (output_dir / search_mix.CATALOG_FILE).read_bytes() == paths[
        "catalog"].read_bytes()
    copied_artifacts = {
        "catalog": search_mix.CATALOG_FILE,
        "exclusions": search_mix.EXCLUSIONS_FILE,
        "retrieval_evidence": search_mix.EVIDENCE_FILE,
        "retrieval_ledger": search_mix.RETRIEVAL_LEDGER_FILE,
        "selection_funnel": search_mix.SELECTION_FUNNEL_FILE,
    }
    for label, filename in copied_artifacts.items():
        assert (output_dir / filename).read_bytes() == (
            paths["manifest"].parent / filename).read_bytes()
        assert native["artifacts"][label] == legacy["artifacts"][label]
    for filename in (
            search_mix.EVIDENCE_FILE,
            search_mix.RETRIEVAL_LEDGER_FILE,
            search_mix.SELECTION_FUNNEL_FILE,
    ):
        sidecar = filename + ".sha256"
        assert (output_dir / sidecar).read_bytes() == (
            paths["manifest"].parent / sidecar).read_bytes()
    for split in search_mix.OUTPUT_FILES:
        assert native["artifacts"][split]["sample_ids"] == legacy["artifacts"][
            split]["sample_ids"]
        source_rows = search_mix._read_parquet(
            paths["manifest"].parent / search_mix.OUTPUT_FILES[split])
        native_rows = search_mix._read_parquet(
            output_dir / search_mix.OUTPUT_FILES[split])
        assert [{
            key: value
            for key, value in row.items() if key != "prompt"
        } for row in native_rows] == [{
            key: value
            for key, value in row.items() if key != "prompt"
        } for row in source_rows]
    probe_ids = native["artifacts"]["probe"]["sample_ids"]
    assert native["artifacts"]["probe_g0"]["sample_ids"] == probe_ids[:1]
    assert native["artifacts"]["probe_autonomous"][
        "sample_ids"] == probe_ids[:2]
    native_probe_rows = search_mix._read_parquet(
        output_dir / search_mix.OUTPUT_FILES["probe"])
    for label, (filename, rows) in search_mix.NATIVE_PROBE_FILES.items():
        probe_rows = search_mix._read_parquet(output_dir / filename)
        assert [{
            key: value
            for key, value in row.items() if key != "prompt"
        } for row in probe_rows] == [{
            key: value
            for key, value in row.items() if key != "prompt"
        } for row in native_probe_rows[:rows]]
    with pytest.raises(ValueError, match="tool protocol mismatch"):
        search_mix.verify_manifest(
            manifest_path,
            paths["model_dir"],
            expected_tool_protocol=search_mix.LEGACY_XML)
    current_manifest = manifest_path.read_bytes()
    stale = json.loads(current_manifest)
    stale["prompt_contract"]["prompt_version"] = "qwen35-native-search-v1"
    manifest_path.write_bytes(search_mix.canonical_json_bytes(stale))
    search_mix.write_digest_sidecar(manifest_path)
    with pytest.raises(ValueError, match="prompt contract mismatch"):
        search_mix.verify_manifest(manifest_path, paths["model_dir"])

    manifest_path.write_bytes(current_manifest)
    search_mix.write_digest_sidecar(manifest_path)
    tampered = json.loads(manifest_path.read_bytes())
    tampered["prompt_contract"]["tool_schema_sha256"] = "0" * 64
    manifest_path.write_bytes(search_mix.canonical_json_bytes(tampered))
    search_mix.write_digest_sidecar(manifest_path)
    with pytest.raises(ValueError, match="prompt contract mismatch"):
        search_mix.verify_manifest(manifest_path, paths["model_dir"])


def test_native_prompt_validation_rejects_reserved_markers_and_token_drift():
    tokenizer = CharacterTokenizer()
    messages = search_mix.qwen35_messages("Where is Paris?")
    assert search_mix._validate_native_prompt(tokenizer, messages) > 0

    poisoned = deepcopy(messages)
    poisoned[0]["content"] = "Question: Where is <search>Paris</search>?\n"
    with pytest.raises(ValueError, match="original prompt"):
        search_mix._validate_native_prompt(tokenizer, poisoned)

    class DriftTokenizer(CharacterTokenizer):

        def apply_chat_template(self, *args, **kwargs):
            value = super().apply_chat_template(*args, **kwargs)
            if kwargs.get("tokenize"):
                return value + [1]
            return value

    with pytest.raises(ValueError, match="prompt tokens differ"):
        search_mix._validate_native_prompt(DriftTokenizer(), messages)


def test_native_search_response_matches_thinking_continuation():
    text, expected = search_mix._native_search_response("Hamlet author")

    assert text.startswith("Inspect the retrieved evidence.\n</think>\n\n")
    assert not text.startswith("<think>")
    assert parse_action(
        text,
        search_mix.QWEN35_NATIVE,
        qwen35_reasoning_mode=QWEN35_REASONING_CONTINUATION,
    ) == expected


def test_native_prompt_validation_enforces_exact_start_limit():
    tokenizer = CharacterTokenizer()
    base = search_mix.qwen35_messages("x")
    base_length = search_mix._validate_native_prompt(tokenizer, base, 10_000)
    exact = search_mix.qwen35_messages("x" * (1024 - base_length + 1))
    assert search_mix._validate_native_prompt(tokenizer, exact) == 1024
    too_long = search_mix.qwen35_messages("x" * (1024 - base_length + 2))
    with pytest.raises(ValueError, match="1025 tokens; maximum is 1024"):
        search_mix._validate_native_prompt(tokenizer, too_long)


def test_verify_native_scans_every_materialized_prompt(tmp_path, monkeypatch):
    paths, _, manifest_path, _ = _build_native_fixture(tmp_path, monkeypatch)
    original = search_mix._validate_native_prompt
    calls = []

    def recording_validator(tokenizer, messages, max_start_length=1024):
        calls.append(messages)
        return original(tokenizer, messages, max_start_length)

    monkeypatch.setattr(search_mix, "_validate_native_prompt",
                        recording_validator)
    payload = search_mix.verify_manifest(
        manifest_path,
        paths["model_dir"],
        expected_tool_protocol=search_mix.QWEN35_NATIVE,
        source_manifest=paths["manifest"])
    expected = sum(payload["artifacts"][label]["rows"] for label in (
        *search_mix.OUTPUT_FILES, *search_mix.NATIVE_PROBE_FILES))
    assert len(calls) == expected


def test_verify_native_without_reselection_preserves_integrity_checks(
        tmp_path, monkeypatch):
    paths, output_dir, manifest_path, _ = _build_native_fixture(
        tmp_path, monkeypatch)
    tokenizer_calls = []
    prompt_calls = []

    class LocalOnlyAutoTokenizer:

        @staticmethod
        def from_pretrained(model_dir, local_files_only=False):
            tokenizer_calls.append((Path(model_dir), local_files_only))
            if local_files_only is not True:
                raise AssertionError("tokenizer verification was not local-only")
            return CharacterTokenizer()

    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = LocalOnlyAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    def forbidden_select(*args, **kwargs):
        del args, kwargs
        raise AssertionError("no-reselection verification selected a catalog")

    def forbidden_retrieval_load(*args, **kwargs):
        del args, kwargs
        raise AssertionError("no-reselection verification loaded the evidence pool")

    original_prompt_validator = search_mix._validate_native_prompt

    def record_prompt_validation(tokenizer,
                                 messages,
                                 max_start_length=1024):
        prompt_calls.append(messages)
        return original_prompt_validator(tokenizer, messages,
                                         max_start_length)

    monkeypatch.setattr(search_mix, "select_catalog", forbidden_select)
    monkeypatch.setattr(search_mix, "_load_retrieval_contract",
                        forbidden_retrieval_load)
    monkeypatch.setattr(search_mix, "_validate_native_prompt",
                        record_prompt_validation)
    payload = search_mix.verify_manifest(
        manifest_path,
        paths["model_dir"],
        expected_tool_protocol=search_mix.QWEN35_NATIVE,
        source_manifest=paths["manifest"],
        reselect_catalog=False)

    assert payload["schema_version"] == search_mix.MATERIALIZED_SCHEMA_VERSION
    assert tokenizer_calls == [(paths["model_dir"], True)] * 2
    expected_prompts = sum(payload["artifacts"][label]["rows"] for label in (
        *search_mix.OUTPUT_FILES, *search_mix.NATIVE_PROBE_FILES))
    assert len(prompt_calls) == expected_prompts

    replay = paths["manifest"].parent / search_mix.REPLAY_FILE
    replay_bytes = replay.read_bytes()
    replay.write_bytes(replay_bytes + b"\n")
    with pytest.raises(ValueError, match="Digest sidecar mismatch"):
        search_mix.verify_manifest(
            manifest_path,
            paths["model_dir"],
            expected_tool_protocol=search_mix.QWEN35_NATIVE,
            source_manifest=paths["manifest"],
            reselect_catalog=False)
    replay.write_bytes(replay_bytes)

    native_catalog = output_dir / search_mix.CATALOG_FILE
    native_catalog.write_bytes(native_catalog.read_bytes() + b"\n")
    _rewrite_manifest_artifact(manifest_path, "catalog", native_catalog)
    with pytest.raises(ValueError, match="Native catalog does not match source"):
        search_mix.verify_manifest(
            manifest_path,
            paths["model_dir"],
            expected_tool_protocol=search_mix.QWEN35_NATIVE,
            source_manifest=paths["manifest"],
            reselect_catalog=False)


def test_verify_native_requires_source_manifest(tmp_path, monkeypatch):
    paths, _, manifest_path, _ = _build_native_fixture(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="requires --source-manifest"):
        search_mix.verify_manifest(manifest_path, paths["model_dir"])


def test_verify_native_rejects_modified_source_manifest(tmp_path, monkeypatch):
    paths, _, manifest_path, _ = _build_native_fixture(tmp_path, monkeypatch)
    source = json.loads(paths["manifest"].read_bytes())
    source["seed"] += 1
    paths["manifest"].write_bytes(search_mix.canonical_json_bytes(source))
    search_mix.write_digest_sidecar(paths["manifest"])

    with pytest.raises(ValueError, match="Manifest contract mismatch"):
        search_mix.verify_manifest(manifest_path,
                                   paths["model_dir"],
                                   source_manifest=paths["manifest"])


def test_verify_native_rejects_source_catalog_drift(tmp_path, monkeypatch):
    paths, _, manifest_path, _ = _build_native_fixture(tmp_path, monkeypatch)
    paths["catalog"].write_bytes(paths["catalog"].read_bytes() + b"\n")

    with pytest.raises(ValueError, match="Artifact identity mismatch"):
        search_mix.verify_manifest(manifest_path,
                                   paths["model_dir"],
                                   source_manifest=paths["manifest"])


def test_verify_native_rejects_coordinated_retrieval_ledger_drift(
        tmp_path, monkeypatch):
    paths, output_dir, manifest_path, _ = _build_native_fixture(
        tmp_path, monkeypatch)
    ledger_path = output_dir / search_mix.RETRIEVAL_LEDGER_FILE
    ledger = json.loads(ledger_path.read_bytes())
    ledger["candidate_order_sha256"] = "1" * 64
    ledger_path.write_bytes(search_mix.canonical_json_bytes(ledger))
    search_mix.write_digest_sidecar(ledger_path)

    funnel_path = output_dir / search_mix.SELECTION_FUNNEL_FILE
    funnel = json.loads(funnel_path.read_bytes())
    funnel["retrieval"]["candidate_order_sha256"] = "1" * 64
    funnel_path.write_bytes(search_mix.canonical_json_bytes(funnel))
    search_mix.write_digest_sidecar(funnel_path)

    manifest = json.loads(manifest_path.read_bytes())
    for label, artifact_path in (("retrieval_ledger", ledger_path),
                                 ("selection_funnel", funnel_path)):
        manifest["artifacts"][label]["sha256"] = search_mix.sha256_file(
            artifact_path)
        manifest["artifacts"][label]["bytes"] = artifact_path.stat().st_size
    manifest["retrieval"]["ledger_sha256"] = search_mix.sha256_file(
        ledger_path)
    manifest_path.write_bytes(search_mix.canonical_json_bytes(manifest))
    search_mix.write_digest_sidecar(manifest_path)

    with pytest.raises(
            ValueError,
            match="Native artifact identity does not match source: retrieval_ledger"
    ):
        search_mix.verify_manifest(manifest_path,
                                   paths["model_dir"],
                                   source_manifest=paths["manifest"])


def test_verify_native_rejects_coordinated_selection_funnel_drift(
        tmp_path, monkeypatch):
    paths, output_dir, manifest_path, _ = _build_native_fixture(
        tmp_path, monkeypatch)
    funnel_path = output_dir / search_mix.SELECTION_FUNNEL_FILE
    funnel = json.loads(funnel_path.read_bytes())
    funnel["retrieval"]["source_rows"]["nq"] += 1
    funnel_path.write_bytes(search_mix.canonical_json_bytes(funnel))
    search_mix.write_digest_sidecar(funnel_path)
    _rewrite_manifest_artifact(manifest_path, "selection_funnel", funnel_path)

    with pytest.raises(
            ValueError,
            match="Native artifact identity does not match source: selection_funnel"
    ):
        search_mix.verify_manifest(manifest_path,
                                   paths["model_dir"],
                                   source_manifest=paths["manifest"])


def test_verify_native_rejects_tampered_lineage(tmp_path, monkeypatch):
    paths, _, manifest_path, _ = _build_native_fixture(tmp_path, monkeypatch)
    manifest = json.loads(manifest_path.read_bytes())
    manifest["derived_from"]["source_manifest_sha256"] = "0" * 64
    manifest_path.write_bytes(search_mix.canonical_json_bytes(manifest))
    search_mix.write_digest_sidecar(manifest_path)

    with pytest.raises(ValueError, match="lineage does not match source"):
        search_mix.verify_manifest(manifest_path,
                                   paths["model_dir"],
                                   source_manifest=paths["manifest"])


def test_generic_materialize_cli_rejects_tool_protocol(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        search_mix.parse_args([
            "materialize", "--local-dir",
            str(tmp_path), "--model-dir",
            str(tmp_path / "model"), "--tool-protocol",
            search_mix.QWEN35_NATIVE
        ])
    assert error.value.code == 2
    assert "unrecognized arguments: --tool-protocol" in capsys.readouterr().err


def test_materialize_quota_failure_replaces_funnel_with_receipt(
        tmp_path, monkeypatch):
    paths = _build_manifest_fixture(tmp_path, monkeypatch)
    complete_funnel = json.loads(paths["funnel"].read_bytes())
    impossible_quotas = deepcopy(SMALL_QUOTAS)
    impossible_quotas["single"]["train"] = 3
    monkeypatch.setattr(search_mix, "QUOTAS", impossible_quotas)
    search_mix._write_selection_funnel(
        tmp_path,
        search_mix._selection_funnel_payload("retrieval_complete", "retrieve",
                                             complete_funnel["retrieval"]))

    with pytest.raises(search_mix.SelectionQuotaError,
                       match="Materialization quota shortfall"):
        search_mix.materialize(tmp_path, paths["model_dir"])

    failed = json.loads(paths["funnel"].read_bytes())
    assert failed["status"] == "failed"
    assert failed["stage"] == "materialize"
    assert failed["failure"] == {
        "stage": "materialize",
        "shortfalls": {
            "single": {
                "available": 3,
                "required": 4,
                "missing": 1,
            }
        },
    }
    assert failed["materialize"]["valid_after_visibility"] == {
        "single": 3,
        "comparison": 4,
        "bridge": 2,
    }
    assert failed["materialize"]["selected_by_category"]["single"] == 3
    search_mix._verify_sidecar(paths["funnel"])


def _selected_replay_steps(paths):
    catalog = search_mix.read_canonical_jsonl(paths["catalog"])
    evidence = {
        record["source_id"]: record
        for record in search_mix.read_canonical_jsonl(paths["evidence"])
    }
    steps = []
    for selected in catalog:
        record = evidence[selected["source_id"]]
        steps.append({
            "phase": "first",
            "query": record["first_query"],
            "results": record["first_results"],
        })
        if record["category"] != "single":
            steps.append({
                "phase": "second",
                "query": record["second_query"],
                "results": record["second_results"],
            })
    return catalog, steps


def _raw_retrieval_results(results):
    return [{
        "document": {
            "title": result["title"],
            "contents": result["contents"],
        },
        "document_id": result["document_id"],
        "score": result["score"],
    } for result in results]


def _install_fake_retriever(monkeypatch, steps):
    calls = []
    initializations = []

    class FakeBM25Retriever:

        def __init__(self, index_path, topk, corpus_path, offsets_path):
            initializations.append({
                "index_path": index_path,
                "topk": topk,
                "corpus_path": corpus_path,
                "offsets_path": offsets_path,
            })

        def search(self, query, topk, return_scores):
            call_index = len(calls)
            if call_index >= len(steps):
                raise AssertionError(f"unexpected replay query: {query}")
            expected = steps[call_index]
            assert query == expected["query"]
            calls.append((query, topk, return_scores))
            return deepcopy(_raw_retrieval_results(expected["results"]))

    module = ModuleType("search_r1.search.bm25_server")
    module.BM25Retriever = FakeBM25Retriever
    monkeypatch.setitem(sys.modules, "search_r1.search.bm25_server", module)
    return calls, initializations


def test_replay_selected_retrieval_replays_queries_and_seals_receipt(
        tmp_path, monkeypatch):
    paths = _build_manifest_fixture(tmp_path, monkeypatch)
    catalog, steps = _selected_replay_steps(paths)
    calls, initializations = _install_fake_retriever(monkeypatch, steps)
    index_path = tmp_path / "index"
    corpus_path = tmp_path / "corpus.jsonl"
    offsets_path = tmp_path / "corpus.offsets"

    receipt_path = search_mix.replay_selected_retrieval(
        paths["manifest"], index_path, corpus_path, offsets_path)

    receipt = json.loads(receipt_path.read_bytes())
    selected_ids = [record["source_id"] for record in catalog]
    expected_order_digest = hashlib.sha256("".join(
        f"{source_id}\n"
        for source_id in selected_ids).encode("utf-8")).hexdigest()
    assert calls == [(step["query"], search_mix.TOPK, True) for step in steps]
    assert initializations == [{
        "index_path": str(index_path),
        "topk": search_mix.TOPK,
        "corpus_path": str(corpus_path),
        "offsets_path": str(offsets_path),
    }]
    assert receipt["selected_rows"] == len(catalog)
    assert receipt["query_count"] == len(steps)
    assert receipt["selected_order_sha256"] == expected_order_digest
    assert receipt["manifest_sha256"] == search_mix.sha256_file(
        paths["manifest"])
    assert receipt["catalog_sha256"] == search_mix.sha256_file(
        paths["catalog"])
    assert receipt["evidence_sha256"] == search_mix.sha256_file(
        paths["evidence"])
    search_mix._verify_sidecar(receipt_path)
    assert search_mix.verify_replay_receipt(paths["manifest"]) == receipt


@pytest.mark.parametrize("field,value", [
    ("manifest_sha256", "0" * 64),
    ("query_count", 0),
    ("selected_order_sha256", "0" * 64),
])
def test_verify_replay_receipt_rejects_resigned_semantic_drift(
        tmp_path, monkeypatch, field, value):
    paths = _build_manifest_fixture(tmp_path, monkeypatch)
    _, steps = _selected_replay_steps(paths)
    _install_fake_retriever(monkeypatch, steps)
    receipt_path = search_mix.replay_selected_retrieval(
        paths["manifest"], tmp_path / "index", tmp_path / "corpus.jsonl",
        tmp_path / "corpus.offsets")
    receipt = json.loads(receipt_path.read_bytes())
    receipt[field] = value
    receipt_path.write_bytes(search_mix.canonical_json_bytes(receipt))
    search_mix.write_digest_sidecar(receipt_path)

    with pytest.raises(ValueError, match="Replay receipt contract mismatch"):
        search_mix.verify_replay_receipt(paths["manifest"])


def test_verify_replay_receipt_rejects_missing_sidecar_and_symlink(
        tmp_path, monkeypatch):
    paths = _build_manifest_fixture(tmp_path, monkeypatch)
    _, steps = _selected_replay_steps(paths)
    _install_fake_retriever(monkeypatch, steps)
    receipt_path = search_mix.replay_selected_retrieval(
        paths["manifest"], tmp_path / "index", tmp_path / "corpus.jsonl",
        tmp_path / "corpus.offsets")
    sidecar = receipt_path.with_suffix(receipt_path.suffix + ".sha256")
    sidecar.unlink()
    with pytest.raises(ValueError, match="Digest sidecar mismatch"):
        search_mix.verify_replay_receipt(paths["manifest"])

    receipt_path.rename(tmp_path / "real-receipt.json")
    receipt_path.symlink_to(tmp_path / "real-receipt.json")
    with pytest.raises(ValueError, match="missing or symlinked"):
        search_mix.verify_replay_receipt(paths["manifest"])


@pytest.mark.parametrize("phase", ["first", "second"])
@pytest.mark.parametrize("field", ["document_id", "score"])
def test_replay_selected_retrieval_rejects_document_or_score_drift(
        tmp_path, monkeypatch, phase, field):
    paths = _build_manifest_fixture(tmp_path, monkeypatch)
    _, steps = _selected_replay_steps(paths)
    target = next(step for step in steps if step["phase"] == phase)
    target["results"] = deepcopy(target["results"])
    if field == "document_id":
        target["results"][0][field] += "-drift"
    else:
        target["results"][0][field] += 0.25
    _install_fake_retriever(monkeypatch, steps)

    with pytest.raises(ValueError,
                       match=f"{phase.title()} retrieval replay mismatch"):
        search_mix.replay_selected_retrieval(paths["manifest"],
                                             tmp_path / "index",
                                             tmp_path / "corpus.jsonl",
                                             tmp_path / "corpus.offsets")


def test_verify_manifest_rejects_manifest_tampering(tmp_path, monkeypatch):
    paths = _build_manifest_fixture(tmp_path, monkeypatch)
    manifest = json.loads(paths["manifest"].read_bytes())
    manifest["seed"] += 1
    paths["manifest"].write_bytes(search_mix.canonical_json_bytes(manifest))
    search_mix.write_digest_sidecar(paths["manifest"])

    with pytest.raises(ValueError, match="Manifest contract mismatch"):
        search_mix.verify_manifest(paths["manifest"], paths["model_dir"])


def test_verify_manifest_rejects_catalog_tampering(tmp_path, monkeypatch):
    paths = _build_manifest_fixture(tmp_path, monkeypatch)
    records = search_mix.read_canonical_jsonl(paths["catalog"])
    records[0] = dict(records[0])
    records[0]["question"] += " tampered"
    paths["catalog"].write_bytes(b"".join(
        search_mix.canonical_json_bytes(record) for record in records))
    _rewrite_manifest_artifact(paths["manifest"], "catalog", paths["catalog"])

    with pytest.raises(ValueError, match="deterministic reselection"):
        search_mix.verify_manifest(paths["manifest"], paths["model_dir"])


def test_verify_manifest_rejects_evidence_tampering(tmp_path, monkeypatch):
    paths = _build_manifest_fixture(tmp_path, monkeypatch)
    records = search_mix.read_canonical_jsonl(paths["evidence"])
    records[0] = dict(records[0])
    records[0]["question"] += " tampered"
    paths["evidence"].write_bytes(b"".join(
        search_mix.canonical_json_bytes(record) for record in records))
    search_mix.write_digest_sidecar(paths["evidence"])
    _rewrite_manifest_artifact(paths["manifest"], "retrieval_evidence",
                               paths["evidence"])

    with pytest.raises(ValueError,
                       match="Retrieval ledger values are invalid"):
        search_mix.verify_manifest(paths["manifest"], paths["model_dir"])


def test_coordinated_evidence_hash_updates_still_fail_reselection(
        tmp_path, monkeypatch):
    paths = _build_manifest_fixture(tmp_path, monkeypatch)
    evidence = search_mix.read_canonical_jsonl(paths["evidence"])
    evidence[0] = deepcopy(evidence[0])
    evidence[0]["first_results"][0]["score"] += 0.5
    paths["evidence"].write_bytes(b"".join(
        search_mix.canonical_json_bytes(record) for record in evidence))
    search_mix.write_digest_sidecar(paths["evidence"])

    ledger = json.loads(paths["ledger"].read_bytes())
    ledger["evidence_sha256"] = search_mix.sha256_file(paths["evidence"])
    paths["ledger"].write_bytes(search_mix.canonical_json_bytes(ledger))
    search_mix.write_digest_sidecar(paths["ledger"])

    funnel = json.loads(paths["funnel"].read_bytes())
    funnel["retrieval"]["evidence_sha256"] = search_mix.sha256_file(
        paths["evidence"])
    paths["funnel"].write_bytes(search_mix.canonical_json_bytes(funnel))
    search_mix.write_digest_sidecar(paths["funnel"])

    manifest = json.loads(paths["manifest"].read_bytes())
    for label, path in (("retrieval_evidence", paths["evidence"]),
                        ("retrieval_ledger", paths["ledger"]),
                        ("selection_funnel", paths["funnel"])):
        manifest["artifacts"][label]["sha256"] = search_mix.sha256_file(path)
        manifest["artifacts"][label]["bytes"] = path.stat().st_size
    manifest["retrieval"]["ledger_sha256"] = search_mix.sha256_file(
        paths["ledger"])
    paths["manifest"].write_bytes(search_mix.canonical_json_bytes(manifest))
    search_mix.write_digest_sidecar(paths["manifest"])

    with pytest.raises(ValueError, match="deterministic reselection"):
        search_mix.verify_manifest(paths["manifest"], paths["model_dir"])
