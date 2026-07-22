import json
from collections import Counter
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
from types import ModuleType

import pytest

from scripts.data_process import search_mix

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
            return {"single": []}, Counter()
        return {"comparison": comparison, "bridge": bridge}, Counter()

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
    catalog, rejected = search_mix.select_catalog(_small_evidence(),
                                                  CharacterTokenizer(),
                                                  excluded_questions=set())

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


def test_output_records_expose_only_five_trainer_fields(monkeypatch):
    monkeypatch.setattr(search_mix, "QUOTAS", deepcopy(SMALL_QUOTAS))
    catalog, _ = search_mix.select_catalog(_small_evidence(),
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
    catalog, rejection_counts = search_mix.select_catalog(
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

    manifest = json.loads(paths["manifest"].read_bytes())
    for label, path in (("retrieval_evidence", paths["evidence"]),
                        ("retrieval_ledger", paths["ledger"])):
        manifest["artifacts"][label]["sha256"] = search_mix.sha256_file(path)
        manifest["artifacts"][label]["bytes"] = path.stat().st_size
    manifest["retrieval"]["ledger_sha256"] = search_mix.sha256_file(
        paths["ledger"])
    paths["manifest"].write_bytes(search_mix.canonical_json_bytes(manifest))
    search_mix.write_digest_sidecar(paths["manifest"])

    with pytest.raises(ValueError, match="deterministic reselection"):
        search_mix.verify_manifest(paths["manifest"], paths["model_dir"])
