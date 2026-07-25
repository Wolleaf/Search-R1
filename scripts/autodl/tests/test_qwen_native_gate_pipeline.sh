#!/usr/bin/env bash
set -Eeuo pipefail

AUTODL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
CHECKOUT="$(cd -- "$AUTODL_DIR/../.." && pwd -P)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ROOT="$(mktemp -d)"
trap 'rm -rf -- "$ROOT"' EXIT

DATA_DIR="$ROOT/data/search_mix_qwen35_native_v3"
LEGACY_DIR="$ROOT/data/search_mix"
mkdir -p "$DATA_DIR" "$LEGACY_DIR" "$ROOT/data/nq_small" \
    "$ROOT/data/search_opportunity_gate" "$ROOT/envs/train/bin" \
    "$ROOT/models/Qwen3.5-2B" "$ROOT/output"
ln -s "$CHECKOUT" "$ROOT/checkout"
cat >"$ROOT/envs/train/bin/python" <<'SH'
#!/usr/bin/env bash
if [[ -n "${CAPTURE_LOG:-}" ]]; then
    printf '%s\n' "$@" >"$CAPTURE_LOG"
    exit 0
fi
exec "${PYTHON_BIN:-python3}" "$@"
SH
chmod +x "$ROOT/envs/train/bin/python"
for file in train_512.parquet val_128.parquet probe_multi_64.parquet \
    probe_g0_8.parquet probe_autonomous_16.parquet; do
    printf 'fixture\n' >"$DATA_DIR/$file"
done
"$PYTHON_BIN" - "$DATA_DIR" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
sample_ids = [f"hotpotqa:train:{index}" for index in range(64)]
catalog = b"".join(
    (json.dumps({"sample_id": sample_id,
                 "question": "What is the capital city?",
                 "golden_answers": ["Paris"]},
                sort_keys=True, separators=(",", ":")) + "\n").encode()
    for sample_id in sample_ids
)
(root / "catalog.jsonl").write_bytes(catalog)
manifest = {
    "schema_version": 4,
    "prompt_contract": {
        "tool_protocol": "qwen35_native",
        "prompt_version": "qwen35-native-search-v3-original-aligned",
    },
    "artifacts": {
        "catalog": {
            "file": "catalog.jsonl",
            "sha256": hashlib.sha256(catalog).hexdigest(),
        },
        "probe": {"rows": 64, "sample_ids": sample_ids},
        "probe_g0": {"rows": 8, "sample_ids": sample_ids[:8]},
        "probe_autonomous": {"rows": 16, "sample_ids": sample_ids[:16]},
    },
}
(root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
PY
(
    cd "$DATA_DIR"
    sha256sum manifest.json >manifest.json.sha256
)
printf 'fixture\n' >"$LEGACY_DIR/probe_multi_64.parquet"
printf 'fixture\n' >"$LEGACY_DIR/manifest.json"
printf 'fixture\n' >"$LEGACY_DIR/catalog.jsonl"
printf 'fixture\n' >"$LEGACY_DIR/retrieval_replay.json"
printf 'fixture\n' >"$LEGACY_DIR/retrieval_replay.json.sha256"
printf 'fixture\n' >"$ROOT/data/nq_small/test_128.parquet"
printf 'fixture\n' >"$ROOT/data/search_opportunity_gate/catalog.jsonl"

AUTODL_ROOT="$ROOT"
GPU_COUNT=2
AUTODL_PRICE_PER_HOUR=5.76
QWEN_NATIVE_GATE_STAGE=g0_g1
source "$AUTODL_DIR/08_gpu_qwen_native_gate.sh"
require_native_gate
[[ "$AUTODL_GPU_PIPELINE" == qwen_native_gate && "$TOOL_PROTOCOL" == qwen35_native ]]
[[ "$DATA_DIR" == "$ROOT/data/search_mix_qwen35_native_v3" ]]
[[ "$EVAL_EXPECTED_ROWS" == 16 && "$EVAL_GROUP_SIZE" == 2 ]]

# GPU helpers must import the sealed checkout without relying on an editable install.
(
    unset PYTHONPATH
    native_deadline_check() { return 0; }
    require_native_gate() { return 0; }
    verify_native_data_contract() {
        [[ "$PYTHONPATH" == "$CHECKOUT_DIR" ]]
        (
            cd "$ROOT"
            "$TRAIN_ENV/bin/python" -S \
                "$CHECKOUT_DIR/scripts/data_process/search_mix.py" --help \
                >/dev/null
        )
    }
    file_sha256() { printf 'd%.0s' {1..64}; }
    native_gate_input_digest() { printf 'e%.0s' {1..64}; }
    verify_native_predecessor() { return 0; }
    qwen_native_gate_preflight "$(printf 'c%.0s' {1..40})" \
        "$(printf 'a%.0s' {1..64})" "$(printf 'b%.0s' {1..64})"
)

reseal_native_evidence() {
    local evidence="$1" marker="$2" outer="$3"
    local line relative digest tmp="${evidence}.resealed.$$"
    : >"$tmp"
    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ "$line" =~ ^[0-9a-f]{64}\ \ (.+)$ ]] || return 1
        relative="${BASH_REMATCH[1]}"
        digest="$(sha256sum "$ROOT/$relative" | cut -d' ' -f1)" || return 1
        printf '%s  %s\n' "$digest" "$relative" >>"$tmp"
    done <"$evidence"
    mv "$tmp" "$evidence"
    digest="$(sha256sum "$evidence" | cut -d' ' -f1)" || return 1
    printf '%s\n' "$digest" >"$marker"
    printf '%s\n' "$digest" >"$outer/evidence-digest"
}

# The outer stage deadline also bounds work outside the nested generation commands.
set +e
run_with_native_deadline 1 bash -c \
    ': >"$1"; sleep 5; : >"$2"' _ \
    "$ROOT/preflight-started" "$ROOT/evidence-published"
DEADLINE_RC=$?
set -e
[[ "$DEADLINE_RC" == 124 && -f "$ROOT/preflight-started" &&
    ! -e "$ROOT/evidence-published" ]]

# Nested generation leaves a fixed tail for run records, analysis, evidence,
# and bounded retriever cleanup before the outer deadline can issue KILL.
QWEN_NATIVE_DEADLINE_EPOCH="$(( $(date +%s) + NATIVE_FINALIZE_GRACE_SECONDS + 20 ))"
AVAILABLE_SECONDS="$(native_work_timeout_seconds test-reserve)"
((AVAILABLE_SECONDS >= 19 && AVAILABLE_SECONDS <= 20))
mkdir -p "$ROOT/runs/eval/qwen_native_g0/attempts"
BEFORE_ATTEMPTS="$(find "$ROOT/runs/eval/qwen_native_g0/attempts" \
    -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)"
QWEN_NATIVE_DEADLINE_EPOCH="$(( $(date +%s) + NATIVE_FINALIZE_GRACE_SECONDS ))"
set +e
run_protocol_probe "$ROOT/models/Qwen3.5-2B" "$(printf 'a%.0s' {1..64})"
EXPIRED_PROBE_RC=$?
set -e
AFTER_ATTEMPTS="$(find "$ROOT/runs/eval/qwen_native_g0/attempts" \
    -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)"
[[ "$EXPIRED_PROBE_RC" == 124 && "$BEFORE_ATTEMPTS" == "$AFTER_ATTEMPTS" ]]
unset QWEN_NATIVE_DEADLINE_EPOCH

# A TERM-ignoring retriever is escalated to KILL within a bounded cleanup.
setsid bash -c 'trap "" TERM; : >"$1"; while :; do sleep 1; done' \
    _ "$ROOT/retriever-ready" &
STUB_RETRIEVER_PID=$!
for _ in {1..50}; do [[ -f "$ROOT/retriever-ready" ]] && break; sleep 0.02; done
[[ -f "$ROOT/retriever-ready" ]]
stop_retriever_group "$STUB_RETRIEVER_PID"
! kill -0 -- "-$STUB_RETRIEVER_PID" 2>/dev/null
grep -Fq '9>&- >"$retriever_log"' "$AUTODL_DIR/03_gpu_run.sh"

DIGEST="$(printf 'a%.0s' {1..64})"
CAPTURE="$ROOT/native-config-args.txt"
AUTODL_CONFIG_ONLY=1 AUTODL_ROOT="$ROOT" GPU_COUNT=2 \
    DATA_DIR="$DATA_DIR" OUTPUT_DIR="$ROOT/output" CAPTURE_LOG="$CAPTURE" \
    TOOL_PROTOCOL=qwen35_native MAX_RESPONSE_LENGTH=500 \
    EVAL_DATA_FILE="$DATA_DIR/probe_autonomous_16.parquet" EVAL_GROUP_SIZE=2 \
    TRACE_OUTPUT_DIR="$ROOT/traces" TRACE_STAGE=qwen_native_g1 \
    TRACE_RUN_ID=test TRACE_CHECKPOINT_DIGEST="$DIGEST" \
    bash "$AUTODL_DIR/train_small_grpo.sh" eval qwen_native_g1 \
        "$ROOT/models/Qwen3.5-2B"
grep -Fxq "data.train_files=$DATA_DIR/train_512.parquet" "$CAPTURE"
grep -Fxq 'data.return_raw_chat=true' "$CAPTURE"
grep -Fxq 'data.eval_group_size=2' "$CAPTURE"
grep -Fxq 'actor_rollout_ref.rollout.top_k=0' "$CAPTURE"
grep -Fxq 'actor_rollout_ref.rollout.presence_penalty=0.0' "$CAPTURE"
grep -Fxq '++tool_protocol=qwen35_native' "$CAPTURE"

FIXTURE="$ROOT/protocol-fixture.json"
TRACE="$ROOT/g1-trace.jsonl"
"$PYTHON_BIN" - "$FIXTURE" "$TRACE" "$DIGEST" <<'PY'
import json
from pathlib import Path
import sys

fixture_path, trace_path = map(Path, sys.argv[1:3])
digest = sys.argv[3]
sampling = {
    "temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0,
    "presence_penalty": 0.0, "repetition_penalty": 1.0,
}
records = []
for mode in ("direct", "native_manager"):
    for question in range(8):
        for slot in range(2):
            records.append({
                "sample_id": f"hotpotqa:train:{question}",
                "group_slot": slot,
                "mode": mode,
                "prompt_token_sha256": "b" * 64,
                "raw_text": "reasoning</think><tool_call><function=search><parameter=query>capital city</parameter></function></tool_call> tail",
                "raw_token_ids": [1, 2, 3],
                "action_text": "reasoning</think><tool_call><function=search><parameter=query>capital city</parameter></function></tool_call>",
                "action_token_ids": [1, 2],
                "action_boundary": "tool_call",
                "tail_dropped": True,
                "parsed_action": {
                    "action": "search", "content": "capital city", "error": None,
                    "prefix": "reasoning", "valid": True,
                },
                "thinking": {
                    "context": "initial_question", "template_opening_provided": True,
                    "nonempty_reasoning": True, "closing_before_action": True,
                },
                "generation_events": [], "retrieval_events": [], "final_answer": None,
            })
fixture = {
    "resolved_config": {
        "checkpoint_digest": digest, "sampling": sampling,
        "prompt_version": "qwen35-native-search-v3-original-aligned",
        "max_action_budget": 4, "max_obs_length": 500,
    },
    "records": records,
    "environment_replay": {
        "sample_id": "hotpotqa:train:0", "query": "Barack Obama",
        "action_text": "fixed", "requested_search_count": 1,
        "executed_search_count": 1, "retrieval_event_count": 1,
        "nonempty_tool_response_count": 1, "retrieved_document_count": 3,
        "visible_tool_response": "Doc 1 evidence", "tool_role_rendered": True,
        "policy_token_count": 2, "tool_response_token_count": 4,
        "tool_response_policy_token_count": 0, "info_mask_consistent": True,
        "scientific_metric": False,
    },
}
fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
with trace_path.open("w", encoding="utf-8", newline="\n") as handle:
    for question in range(16):
        for slot in range(2):
            trace = {
                "sample_id": f"hotpotqa:train:{question}", "group_slot": slot,
                "schema_version": 3,
                "checkpoint_digest": digest, "question": "What is the capital city?",
                "turns": [{
                    "action": "search", "valid_action": True,
                    "search_query": f"capital city {question}", "retrieval_executed": True,
                    "observation": "Doc 1 evidence", "retrieved_docs": [{"title": "Doc"}],
                }],
                "executed_search_count": 1,
                "generation_events": [{
                    "turn": 0, "text": "reasoning</think><tool_call>x</tool_call>",
                    "raw_text": "reasoning</think><tool_call>x</tool_call> tail",
                    "raw_token_ids": [1, 2, 3], "action_token_ids": [1, 2],
                    "tail_dropped": True, "generation_context": "initial_question",
                    "action": "search", "clipped": False,
                }],
                "retrieval_events": [{"turn": 0, "query": f"capital city {question}"}],
                "max_action_budget": 4, "action_count": 1,
                "policy_token_count": 2, "observation_token_count": 1,
                "observation_policy_token_count": 0, "info_mask_consistent": True,
                "invalid_action_count": 0, "response_clipped": False, "em": 0,
            }
            handle.write(json.dumps(trace, sort_keys=True) + "\n")
PY

PROBE_OUTPUT="$ROOT/protocol-output"
"$PYTHON_BIN" "$AUTODL_DIR/qwen_native_protocol_probe.py" \
    --checkpoint-digest "$DIGEST" --seed 42 --fixture-input "$FIXTURE" \
    --output-dir "$PROBE_OUTPUT"
[[ "$(wc -l <"$PROBE_OUTPUT/records.jsonl")" == 32 ]]

ANALYSIS="$ROOT/analysis"
"$PYTHON_BIN" "$AUTODL_DIR/qwen_native_gate_analysis.py" \
    --stage g0_g1 --trace "$TRACE" --catalog "$DATA_DIR/catalog.jsonl" \
    --data-manifest "$DATA_DIR/manifest.json" \
    --expected-checkpoint-digest "$DIGEST" \
    --protocol-probe-dir "$PROBE_OUTPUT" --output-dir "$ANALYSIS"
grep -Fq '"decision":"GO"' "$ANALYSIS/go_no_go.json"
[[ "$(wc -l <"$ANALYSIS/per_trajectory.jsonl")" == 32 ]]

RAW_MISMATCH_FIXTURE="$ROOT/protocol-raw-mismatch-fixture.json"
DIRECT_INVALID_FIXTURE="$ROOT/protocol-direct-invalid-fixture.json"
"$PYTHON_BIN" - "$FIXTURE" "$RAW_MISMATCH_FIXTURE" \
    "$DIRECT_INVALID_FIXTURE" <<'PY'
import copy
import json
from pathlib import Path
import sys

source, raw_target, invalid_target = map(Path, sys.argv[1:])
fixture = json.loads(source.read_text(encoding="utf-8"))

raw_mismatch = copy.deepcopy(fixture)
record = next(item for item in raw_mismatch["records"]
              if item["mode"] == "native_manager"
              and item["sample_id"] == "hotpotqa:train:0"
              and item["group_slot"] == 0)
record["raw_token_ids"][1] = 9
record["action_token_ids"][1] = 9
raw_target.write_text(json.dumps(raw_mismatch), encoding="utf-8")

direct_invalid = copy.deepcopy(fixture)
changed = 0
for record in direct_invalid["records"]:
    if record["mode"] != "direct" or changed == 2:
        continue
    record["parsed_action"] = {
        "action": "invalid", "content": None, "error": "fixture parse error",
        "prefix": record["raw_text"], "valid": False,
    }
    changed += 1
assert changed == 2
invalid_target.write_text(json.dumps(direct_invalid), encoding="utf-8")
PY

# Complete structural NO-GO results return zero and retain their reports.
RAW_MISMATCH_PROBE="$ROOT/protocol-raw-mismatch-output"
RAW_MISMATCH_ANALYSIS="$ROOT/raw-mismatch-analysis"
"$PYTHON_BIN" "$AUTODL_DIR/qwen_native_protocol_probe.py" \
    --checkpoint-digest "$DIGEST" --seed 42 \
    --fixture-input "$RAW_MISMATCH_FIXTURE" --output-dir "$RAW_MISMATCH_PROBE"
"$PYTHON_BIN" "$AUTODL_DIR/qwen_native_gate_analysis.py" \
    --stage g0_g1 --trace "$TRACE" --catalog "$DATA_DIR/catalog.jsonl" \
    --data-manifest "$DATA_DIR/manifest.json" \
    --expected-checkpoint-digest "$DIGEST" \
    --protocol-probe-dir "$RAW_MISMATCH_PROBE" \
    --output-dir "$RAW_MISMATCH_ANALYSIS"

DIRECT_INVALID_PROBE="$ROOT/protocol-direct-invalid-output"
DIRECT_INVALID_ANALYSIS="$ROOT/direct-invalid-analysis"
"$PYTHON_BIN" "$AUTODL_DIR/qwen_native_protocol_probe.py" \
    --checkpoint-digest "$DIGEST" --seed 42 \
    --fixture-input "$DIRECT_INVALID_FIXTURE" --output-dir "$DIRECT_INVALID_PROBE"
"$PYTHON_BIN" "$AUTODL_DIR/qwen_native_gate_analysis.py" \
    --stage g0_g1 --trace "$TRACE" --catalog "$DATA_DIR/catalog.jsonl" \
    --data-manifest "$DATA_DIR/manifest.json" \
    --expected-checkpoint-digest "$DIGEST" \
    --protocol-probe-dir "$DIRECT_INVALID_PROBE" \
    --output-dir "$DIRECT_INVALID_ANALYSIS"

"$PYTHON_BIN" - "$RAW_MISMATCH_ANALYSIS/go_no_go.json" \
    "$DIRECT_INVALID_ANALYSIS/go_no_go.json" <<'PY'
import json
from pathlib import Path
import sys

raw_result, invalid_result = (
    json.loads(Path(path).read_text(encoding="utf-8")) for path in sys.argv[1:])
assert raw_result["decision"] == "NO-GO"
assert raw_result["failed_criteria"] == ["g0_first_action_token_match_count"]
assert raw_result["criteria"]["g0_first_action_token_match_count"]["observed"] == 15
assert invalid_result["decision"] == "GO"
assert invalid_result["failed_criteria"] == []
PY

WRONG_TRACE="$ROOT/g1-wrong-sample-trace.jsonl"
"$PYTHON_BIN" - "$TRACE" "$WRONG_TRACE" <<'PY'
import json
from pathlib import Path
import sys

source, target = map(Path, sys.argv[1:])
rows = [json.loads(line) for line in source.read_text().splitlines()]
rows[0]["sample_id"] = "hotpotqa:train:999"
target.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
PY
if "$PYTHON_BIN" "$AUTODL_DIR/qwen_native_gate_analysis.py" \
        --stage g0_g1 --trace "$WRONG_TRACE" --catalog "$DATA_DIR/catalog.jsonl" \
        --data-manifest "$DATA_DIR/manifest.json" \
        --expected-checkpoint-digest "$DIGEST" \
        --protocol-probe-dir "$PROBE_OUTPUT" --output-dir "$ROOT/wrong-analysis" \
        >/dev/null 2>&1; then
    printf 'A shape-correct trace from the wrong fixed sample set was accepted.\n' >&2
    exit 1
fi

# LEGACY_V2_READ_ONLY_BEGIN: active v3 has no pre-training G2/G3 shell stages.
# The historical fixtures below document old bindings and are not executed.
printf 'qwen native v3 E0/G0/G1 pipeline tests passed\n'
exit 0

# A complete scientific NO-GO is still exit 0 and retains its report.
G2_TRACE="$ROOT/g2-trace.jsonl"
"$PYTHON_BIN" - "$G2_TRACE" "$DIGEST" <<'PY'
import json
from pathlib import Path
import sys

path, digest = Path(sys.argv[1]), sys.argv[2]
with path.open("w", encoding="utf-8", newline="\n") as handle:
    for question in range(32):
        for slot in range(3):
            invalid = question < 2
            if invalid:
                answer = None
            elif question == 2 and slot == 0:
                answer = "Paris"
            elif question == 3 and slot == 0:
                answer = "The answer is Paris."
            else:
                answer = "London"
            trace = {
                "sample_id": f"hotpotqa:train:{question}", "group_slot": slot,
                "checkpoint_digest": digest, "question": "What is the capital city?",
                "gold_answers": ["Paris"], "extracted_answer": answer,
                "turns": [{
                    "action": "invalid" if invalid else "answer",
                    "valid_action": not invalid, "search_query": None,
                    "retrieval_executed": False, "observation": None,
                    "retrieved_docs": [],
                }],
                "executed_search_count": 0, "generation_events": [{"clipped": False}],
                "invalid_action_count": int(invalid), "response_clipped": False,
                "em": int(answer == "Paris"),
            }
            handle.write(json.dumps(trace, sort_keys=True) + "\n")
PY
"$PYTHON_BIN" "$AUTODL_DIR/qwen_native_gate_analysis.py" \
    --stage g2 --trace "$G2_TRACE" --catalog "$DATA_DIR/catalog.jsonl" \
    --data-manifest "$DATA_DIR/manifest.json" \
    --expected-checkpoint-digest "$DIGEST" --output-dir "$ROOT/g2-analysis"
grep -Fq '"decision":"NO-GO"' "$ROOT/g2-analysis/go_no_go.json"
"$PYTHON_BIN" - "$ROOT/g2-analysis/summary.json" <<'PY'
import json
from pathlib import Path
import sys

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert summary["failed_criteria"] == ["invalid_trajectory_count"]
assert summary["overall"]["strict_em_positive_count"] == 1
assert summary["overall"]["strict_em_mixed_group_count"] == 1
assert summary["overall"]["subem_positive_count"] == 2
assert "subem_positive_count" not in summary["criteria"]
PY

# G2 readiness is independently replayed from the catalog, not trusted from
# trace EM or unlocked by diagnostic substring matches.
G2_READY_TRACE="$ROOT/g2-ready-trace.jsonl"
G2_SUBEM_TRACE="$ROOT/g2-subem-trace.jsonl"
G2_BAD_EM_TRACE="$ROOT/g2-bad-em-trace.jsonl"
G2_BAD_GOLD_TRACE="$ROOT/g2-bad-gold-trace.jsonl"
"$PYTHON_BIN" - "$G2_TRACE" "$G2_READY_TRACE" "$G2_SUBEM_TRACE" \
    "$G2_BAD_EM_TRACE" "$G2_BAD_GOLD_TRACE" <<'PY'
import copy
import json
from pathlib import Path
import sys

source, ready_path, subem_path, bad_em_path, bad_gold_path = map(Path, sys.argv[1:])
ready = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
for row in ready:
    row["turns"][0]["action"] = "answer"
    row["turns"][0]["valid_action"] = True
    row["invalid_action_count"] = 0
write = lambda path, rows: path.write_text(
    "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    encoding="utf-8", newline="\n")
write(ready_path, ready)

subem = copy.deepcopy(ready)
for row in subem:
    row["extracted_answer"] = "The answer is Paris."
    row["em"] = 0
write(subem_path, subem)

bad_em = copy.deepcopy(ready)
bad_em[6]["em"] = 0
write(bad_em_path, bad_em)

bad_gold = copy.deepcopy(ready)
bad_gold[0]["gold_answers"] = ["London"]
write(bad_gold_path, bad_gold)
PY

"$PYTHON_BIN" "$AUTODL_DIR/qwen_native_gate_analysis.py" \
    --stage g2 --trace "$G2_READY_TRACE" --catalog "$DATA_DIR/catalog.jsonl" \
    --data-manifest "$DATA_DIR/manifest.json" \
    --expected-checkpoint-digest "$DIGEST" --output-dir "$ROOT/g2-ready-analysis"
grep -Fq '"decision":"GO"' "$ROOT/g2-ready-analysis/go_no_go.json"

"$PYTHON_BIN" "$AUTODL_DIR/qwen_native_gate_analysis.py" \
    --stage g2 --trace "$G2_SUBEM_TRACE" --catalog "$DATA_DIR/catalog.jsonl" \
    --data-manifest "$DATA_DIR/manifest.json" \
    --expected-checkpoint-digest "$DIGEST" --output-dir "$ROOT/g2-subem-analysis"
"$PYTHON_BIN" - "$ROOT/g2-subem-analysis/summary.json" <<'PY'
import json
from pathlib import Path
import sys

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert summary["decision"] == "NO-GO"
assert summary["overall"]["strict_em_positive_count"] == 0
assert summary["overall"]["subem_positive_count"] == 96
assert set(summary["failed_criteria"]) == {
    "strict_em_positive_count", "strict_em_mixed_group_count",
}
PY

for pair in "$G2_BAD_EM_TRACE:$ROOT/g2-bad-em-analysis" \
    "$G2_BAD_GOLD_TRACE:$ROOT/g2-bad-gold-analysis"; do
    trace="${pair%%:*}"
    output="${pair#*:}"
    if "$PYTHON_BIN" "$AUTODL_DIR/qwen_native_gate_analysis.py" \
            --stage g2 --trace "$trace" --catalog "$DATA_DIR/catalog.jsonl" \
            --data-manifest "$DATA_DIR/manifest.json" \
            --expected-checkpoint-digest "$DIGEST" --output-dir "$output" \
            >/dev/null 2>&1; then
        printf 'G2 accepted trace EM or gold drift: %s\n' "$trace" >&2
        exit 1
    fi
done

PYTHONPATH="$AUTODL_DIR${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - \
    "$ROOT" "$DATA_DIR/catalog.jsonl" <<'PY'
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import types

import qwen_native_gate_analysis as analysis

root, catalog = map(Path, sys.argv[1:])
digest = "a" * 64
trace_path = root / "g3.jsonl"
records = []
for question in range(64):
    for slot in range(5):
        answer = "Paris" if question == 0 and slot == 0 else "London"
        records.append({
            "sample_id": f"hotpotqa:train:{question}",
            "group_slot": slot,
            "checkpoint_digest": digest,
            "question": "What is the capital city?",
            "gold_answers": ["Paris"],
            "extracted_answer": answer,
            "turns": [{
                "action": "answer", "valid_action": True,
                "search_query": None, "retrieval_executed": False,
                "observation": None, "retrieved_docs": [],
            }],
            "executed_search_count": 0,
            "invalid_action_count": 0,
            "response_clipped": False,
            "em": int(answer == "Paris"),
        })

def write(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8", newline="\n")

write(trace_path, records)
for name, mutation in (
    ("g3-bad-em.jsonl", lambda row: row.update(em=0)),
    ("g3-bad-gold.jsonl", lambda row: row.update(gold_answers=["London"])),
    ("g3-bad-answer.jsonl", lambda row: row.update(final_answer="London")),
):
    bad = copy.deepcopy(records)
    mutation(bad[0])
    write(root / name, bad)

captured = {}
fake = types.ModuleType("probe_analysis")
def delegated(argv):
    captured["argv"] = argv
    trace = Path(argv[argv.index("--trace") + 1])
    catalog_path = Path(argv[argv.index("--catalog") + 1])
    checkpoint = argv[argv.index("--expected-checkpoint-digest") + 1]
    output = Path(argv[argv.index("--output-dir") + 1])
    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    trace_digest = hashlib.sha256(trace.read_bytes()).hexdigest()
    catalog_digest = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps({
        "decision": "NO-GO",
        "input": {
            "stage": "qwen_native_g3", "trace_sha256": trace_digest,
            "catalog_sha256": catalog_digest, "checkpoint_digest": checkpoint,
        },
        "overall": {"correct_count": sum(row["em"] for row in rows)},
    }), encoding="utf-8")
    (output / "go_no_go.json").write_text(json.dumps({
        "decision": "NO-GO", "trace_sha256": trace_digest,
        "catalog_sha256": catalog_digest, "checkpoint_digest": checkpoint,
    }), encoding="utf-8")
    (output / "per_trajectory.jsonl").write_text("".join(
        json.dumps({
            "sample_id": row["sample_id"], "group_slot": row["group_slot"],
            "em": row["em"],
        }, sort_keys=True) + "\n" for row in rows
    ), encoding="utf-8")
    return 0
fake.main = delegated
sys.modules["probe_analysis"] = fake
args = argparse.Namespace(
    stage="g3",
    trace=trace_path, catalog=catalog,
    expected_checkpoint_digest=digest, output_dir=root / "g3-output")
expected_ids = [f"hotpotqa:train:{question}" for question in range(64)]
questions = {sample_id: "What is the capital city?" for sample_id in expected_ids}
gold = {sample_id: ["Paris"] for sample_id in expected_ids}
decision = analysis.write_outputs(args, expected_ids, questions, gold, [])
assert captured["argv"][:6] == [
    "--trace", str(args.trace), "--catalog", str(args.catalog),
    "--expected-checkpoint-digest", args.expected_checkpoint_digest,
]
assert captured["argv"][6] == "--output-dir"
staged_output = Path(captured["argv"][7])
assert staged_output != args.output_dir
assert staged_output.parent.parent == args.output_dir.parent
assert decision["stage"] == "g3"
assert decision["strict_em_replay"]["strict_em_positive_count"] == 1
summary = json.loads((args.output_dir / "summary.json").read_bytes())
assert summary["strict_em_replay"]["trajectory_count"] == 320
reports = [json.loads(line) for line in
           (args.output_dir / "per_trajectory.jsonl").read_text().splitlines()]
assert all(report["em"] == report["strict_em_replay"]["strict_em"]
           for report in reports)
PY

for name in em gold answer; do
    output="$ROOT/g3-bad-$name-analysis"
    if "$PYTHON_BIN" "$AUTODL_DIR/qwen_native_gate_analysis.py" \
            --stage g3 --trace "$ROOT/g3-bad-$name.jsonl" \
            --catalog "$DATA_DIR/catalog.jsonl" \
            --data-manifest "$DATA_DIR/manifest.json" \
            --expected-checkpoint-digest "$DIGEST" --output-dir "$output" \
            >/dev/null 2>&1; then
        printf 'G3 accepted answer, gold, or trace EM drift: %s\n' "$name" >&2
        exit 1
    fi
    [[ ! -e "$output" ]]
done

TRACE_CONTRACT="$ROOT/trace-contract"
TRACE_CONTRACT_RESULTS="$ROOT/trace-contract-results"
mkdir -p "$TRACE_CONTRACT_RESULTS"
PYTHONPATH="$CHECKOUT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" - \
    "$TRACE_CONTRACT" "$TRACE_CONTRACT_RESULTS" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

from search_r1.trajectory_trace import TraceJsonlWriter, parse_search_r1_transcript

trace_dir, results_dir = map(Path, sys.argv[1:])
trace_dir.mkdir()
raw = "<think>ready</think><answer>Paris</answer>"
writer = TraceJsonlWriter(
    trace_dir / "eval_predictions.jsonl", record_type="eval",
    expected_rows=1, run_id="native-trace-contract", stage="qwen_native_g1")
writer.append({
    "sample_id": "nq:test:7", "source_index": 7,
    "question": "What is the capital of France?", "gold_answers": ["Paris"],
    "raw_trajectory": raw, "turns": parse_search_r1_transcript(raw),
    "extracted_answer": "Paris", "em": 1, "executed_search_count": 0,
    "posthoc_utility": 1.0, "response_tokens": 12,
    "response_clipped": False, "turns_used": 1, "invalid_action_count": 0,
    "checkpoint_digest": "a" * 64,
})
manifest = writer.finalize()
digest = manifest["artifact"]["sha256"]
(results_dir / "summary.json").write_text(
    json.dumps({"trace_sha256": digest}), encoding="utf-8")
(results_dir / "go_no_go.json").write_text(
    json.dumps({"trace_sha256": digest}), encoding="utf-8")
PY
verify_native_trace_evidence \
    "$TRACE_CONTRACT/eval_predictions.manifest.json" \
    "$TRACE_CONTRACT_RESULTS" 1 >/dev/null
printf '{"trace_sha256":"%s"}\n' "$(printf '0%.0s' {1..64})" \
    >"$TRACE_CONTRACT_RESULTS/go_no_go.json"
if verify_native_trace_evidence \
        "$TRACE_CONTRACT/eval_predictions.manifest.json" \
        "$TRACE_CONTRACT_RESULTS" 1 >/dev/null 2>&1; then
    printf 'Analysis evidence with a tampered trace digest was accepted.\n' >&2
    exit 1
fi

OUTER_NAME=20260723T120000Z-123-456
OUTER="$ROOT/state/attempts/gpu/$OUTER_NAME"
RESULTS="$ROOT/runs/qwen-native-gate/attempts/$OUTER_NAME"
EVAL_RUN="$ROOT/runs/eval/qwen_native_g1/attempts/eval"
CURRENT_COMMIT="$(printf 'c%.0s' {1..40})"
mkdir -p "$ROOT/manifests"
printf '{"fixture":true}\n' >"$ROOT/manifests/cpu_handoff.json"
CURRENT_HANDOFF_DIGEST="$(sha256sum "$ROOT/manifests/cpu_handoff.json" | cut -d' ' -f1)"
printf '%s  cpu_handoff.json\n' "$CURRENT_HANDOFF_DIGEST" \
    >"$ROOT/manifests/cpu_handoff.json.sha256"
printf '%s\n' "$CURRENT_HANDOFF_DIGEST" >"$ROOT/manifests/cpu.ok"
CURRENT_DATA_DIGEST="$(sha256sum "$DATA_DIR/manifest.json" | cut -d' ' -f1)"
mkdir -p "$OUTER" "$RESULTS" "$EVAL_RUN/traces"
cp "$ANALYSIS"/* "$RESULTS/"
printf 'g0_g1\n' >"$RESULTS/stage.txt"
printf '%s\n%s\n' \
    $'stage\tcheckpoint\tcheckpoint_digest\tevaluation_checkout_commit\tcpu_handoff_digest\tdata_manifest_sha256\tresolved_config_sha256\ttrace_sha256\ttrace_manifest_sha256\tpredecessor_evidence_sha256' \
    "g0_g1"$'\t'"$ROOT/models/Qwen3.5-2B"$'\t'"$DIGEST"$'\t'"$CURRENT_COMMIT"$'\t'"$CURRENT_HANDOFF_DIGEST"$'\t'"$CURRENT_DATA_DIGEST"$'\t'"$(printf 'e%.0s' {1..64})"$'\t'"$(printf 'f%.0s' {1..64})"$'\t'"$(printf '1%.0s' {1..64})"$'\t-' \
    >"$RESULTS/lineage.tsv"
printf 'index\n' >"$RESULTS/run-index.tsv"
printf 'fixture\n' >"$EVAL_RUN/train.log"
"$PYTHON_BIN" - "$EVAL_RUN/resolved-config.yaml" "$DATA_DIR" <<'PY'
import json
from pathlib import Path
import sys

path, data_dir = map(Path, sys.argv[1:])
config = {
    "tool_protocol": "qwen35_native",
    "max_turns": 4,
    "retriever": {"topk": 3},
    "data": {
        "train_files": str(data_dir / "train_512.parquet"),
        "val_files": str(data_dir / "probe_forced_16.parquet"),
        "train_batch_size": 8, "val_batch_size": 8, "eval_group_size": 2,
        "return_raw_chat": True, "max_prompt_length": 4096,
        "max_response_length": 500, "max_start_length": 1024,
        "max_obs_length": 384,
    },
    "actor_rollout_ref": {"rollout": {
        "temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0,
        "presence_penalty": 0.0, "repetition_penalty": 1.0,
    }},
}
path.write_text(json.dumps(config), encoding="utf-8")
PY
printf '%s\n' \
    "data_dir=$DATA_DIR" \
    "eval_data_file=$DATA_DIR/probe_forced_16.parquet" \
    'eval_group_size=2' \
    'max_response_length=500' \
    'tool_protocol=qwen35_native' \
    'rollout_top_k=0' \
    'rollout_min_p=0.0' \
    'rollout_presence_penalty=0.0' \
    'rollout_repetition_penalty=1.0' \
    'input_model=/sealed/model' \
    >"$EVAL_RUN/run.env"
native_sampling_from_resolved_config "$EVAL_RUN/resolved-config.yaml" \
    "$EVAL_RUN/run.env" >"$RESULTS/sampling.json"
cp "$EVAL_RUN/resolved-config.yaml" "$ROOT/resolved-config.saved"
sed -i 's/"presence_penalty": 0.0/"presence_penalty": 1.0/' \
    "$EVAL_RUN/resolved-config.yaml"
if native_sampling_from_resolved_config "$EVAL_RUN/resolved-config.yaml" \
        "$EVAL_RUN/run.env" >/dev/null 2>&1; then
    printf 'A mismatched native resolved config was accepted.\n' >&2
    exit 1
fi
mv "$ROOT/resolved-config.saved" "$EVAL_RUN/resolved-config.yaml"
for file in eval_predictions.jsonl eval_predictions.manifest.json \
    eval_predictions.manifest.json.sha256; do
    printf 'fixture\n' >"$EVAL_RUN/traces/$file"
done
printf 'success\n' >"$OUTER/terminal"
printf '0\n' >"$OUTER/exit-code"
: >"$OUTER/.success"

# Evidence input failures must not escape command substitutions or seal an attempt.
MISSING_MARKER="$ROOT/manifests/qwen-native-gate/$OUTER_NAME.ok"
mv "$EVAL_RUN/train.log" "$ROOT/train.log.saved"
if publish_native_gate_evidence "$OUTER" "$RESULTS" "$EVAL_RUN" \
        >/dev/null 2>&1; then
    printf 'Missing Qwen native evidence was accepted.\n' >&2
    exit 1
fi
[[ ! -e "$MISSING_MARKER" && ! -e "$OUTER/result-contract" &&
    ! -e "$OUTER/result-root" && ! -e "$OUTER/evidence-marker" &&
    ! -e "$OUTER/evidence-digest" ]]
ln -s "$ROOT/train.log.saved" "$EVAL_RUN/train.log"
if publish_native_gate_evidence "$OUTER" "$RESULTS" "$EVAL_RUN" \
        >/dev/null 2>&1; then
    printf 'Symlinked Qwen native evidence was accepted.\n' >&2
    exit 1
fi
[[ ! -e "$MISSING_MARKER" && ! -e "$OUTER/result-contract" ]]
rm "$EVAL_RUN/train.log"
mv "$ROOT/train.log.saved" "$EVAL_RUN/train.log"

# Simulate a file changing after its checksum is collected but before sealing.
ORIGINAL_FILE_SHA256="$(declare -f file_sha256)"
HASH_RACE_TARGET="$EVAL_RUN/train.log"
file_sha256() {
    local path="$1" digest
    [[ -f "$path" && ! -L "$path" ]] || return 1
    digest="$(sha256sum -- "$path" | cut -d' ' -f1)" || return 1
    printf '%s\n' "$digest"
    if [[ "$path" == "$HASH_RACE_TARGET" ]]; then
        printf 'post-hash drift\n' >>"$path"
    fi
}
cp "$EVAL_RUN/train.log" "$ROOT/train.log.clean"
if publish_native_gate_evidence "$OUTER" "$RESULTS" "$EVAL_RUN" \
        >/dev/null 2>&1; then
    printf 'Hash-invalid Qwen native evidence was sealed.\n' >&2
    exit 1
fi
[[ ! -e "$MISSING_MARKER" && ! -e "$OUTER/result-contract" &&
    ! -e "$OUTER/evidence-marker" && ! -e "$OUTER/evidence-digest" ]]
mv "$ROOT/train.log.clean" "$EVAL_RUN/train.log"
eval "$ORIGINAL_FILE_SHA256"

publish_native_gate_evidence "$OUTER" "$RESULTS" "$EVAL_RUN"
MARKER="$ROOT/manifests/qwen-native-gate/$OUTER_NAME.ok"
[[ -s "$RESULTS/evidence.sha256" && -s "$MARKER" ]]
if publish_native_gate_evidence "$OUTER" "$RESULTS" "$EVAL_RUN" >/dev/null 2>&1; then
    printf 'Qwen native evidence publisher overwrote an immutable marker.\n' >&2
    exit 1
fi

# Build a second, internally valid G0 envelope from an older checkout lineage.
# G3 must not accept it when a G2 envelope splices that marker into its chain.
ALT_G0_NAME=20260723T120500Z-123-999
ALT_G0_OUTER="$ROOT/state/attempts/gpu/$ALT_G0_NAME"
ALT_G0_RESULTS="$ROOT/runs/qwen-native-gate/attempts/$ALT_G0_NAME"
ALT_G0_COMMIT="$(printf 'b%.0s' {1..40})"
mkdir -p "$ALT_G0_OUTER" "$ALT_G0_RESULTS"
for file in summary.json summary.md go_no_go.json per_trajectory.jsonl \
    per_question.jsonl lineage.tsv run-index.tsv stage.txt sampling.json; do
    cp "$RESULTS/$file" "$ALT_G0_RESULTS/$file"
done
sed -i "s/$CURRENT_COMMIT/$ALT_G0_COMMIT/" "$ALT_G0_RESULTS/lineage.tsv"
printf 'success\n' >"$ALT_G0_OUTER/terminal"
printf '0\n' >"$ALT_G0_OUTER/exit-code"
: >"$ALT_G0_OUTER/.success"
publish_native_gate_evidence "$ALT_G0_OUTER" "$ALT_G0_RESULTS" "$EVAL_RUN"
ALT_G0_MARKER="$ROOT/manifests/qwen-native-gate/$ALT_G0_NAME.ok"
ALT_G0_EVIDENCE_DIGEST="$(sha256sum "$ALT_G0_RESULTS/evidence.sha256" | cut -d' ' -f1)"

# The final seal revalidation catches data, evaluation, and model drift.
(
    verify_native_data_contract() { return 0; }
    NATIVE_STAGE=g0_g1
    NATIVE_PREDECESSOR_EVIDENCE=''
    EVAL_DATA_FILE="$DATA_DIR/probe_forced_16.parquet"
    PARENT_EVIDENCE_DIGEST='-'
    SEAL_MODEL="$ROOT/models/Qwen3.5-2B"
    SEAL_MODEL_DIGEST="$(tree_sha256 "$SEAL_MODEL")"
    NATIVE_PREFLIGHT_COMMIT="$CURRENT_COMMIT"
    NATIVE_PREFLIGHT_HANDOFF_DIGEST="$CURRENT_HANDOFF_DIGEST"
    NATIVE_PREFLIGHT_CHECKPOINT_DIGEST="$SEAL_MODEL_DIGEST"
    NATIVE_PREFLIGHT_DATA_DIGEST="$CURRENT_DATA_DIGEST"
    NATIVE_PREFLIGHT_INPUT_DIGEST="$(native_gate_input_digest)"
    NATIVE_PREFLIGHT_PARENT_EVIDENCE_DIGEST='-'
    revalidate_native_gate_seal_inputs "$CURRENT_COMMIT" \
        "$CURRENT_HANDOFF_DIGEST" "$SEAL_MODEL" "$SEAL_MODEL_DIGEST" \
        "$CURRENT_DATA_DIGEST"

    cp "$HANDOFF" "$ROOT/handoff.clean"
    printf 'drift\n' >>"$HANDOFF"
    if revalidate_native_gate_seal_inputs "$CURRENT_COMMIT" \
            "$CURRENT_HANDOFF_DIGEST" "$SEAL_MODEL" "$SEAL_MODEL_DIGEST" \
            "$CURRENT_DATA_DIGEST" >/dev/null 2>&1; then
        printf 'CPU handoff drift passed final seal validation.\n' >&2
        exit 1
    fi
    mv "$ROOT/handoff.clean" "$HANDOFF"

    cp "$DATA_DIR/catalog.jsonl" "$ROOT/catalog.clean"
    printf '{"drift":true}\n' >>"$DATA_DIR/catalog.jsonl"
    if revalidate_native_gate_seal_inputs "$CURRENT_COMMIT" \
            "$CURRENT_HANDOFF_DIGEST" "$SEAL_MODEL" "$SEAL_MODEL_DIGEST" \
            "$CURRENT_DATA_DIGEST" >/dev/null 2>&1; then
        printf 'Native catalog drift passed final seal validation.\n' >&2
        exit 1
    fi
    mv "$ROOT/catalog.clean" "$DATA_DIR/catalog.jsonl"

    cp "$LEGACY_REPLAY_RECEIPT" "$ROOT/replay.clean"
    printf 'drift\n' >>"$LEGACY_REPLAY_RECEIPT"
    if revalidate_native_gate_seal_inputs "$CURRENT_COMMIT" \
            "$CURRENT_HANDOFF_DIGEST" "$SEAL_MODEL" "$SEAL_MODEL_DIGEST" \
            "$CURRENT_DATA_DIGEST" >/dev/null 2>&1; then
        printf 'Legacy replay receipt drift passed final seal validation.\n' >&2
        exit 1
    fi
    mv "$ROOT/replay.clean" "$LEGACY_REPLAY_RECEIPT"

    cp "$EVAL_DATA_FILE" "$ROOT/eval.clean"
    printf 'drift\n' >>"$EVAL_DATA_FILE"
    if revalidate_native_gate_seal_inputs "$CURRENT_COMMIT" \
            "$CURRENT_HANDOFF_DIGEST" "$SEAL_MODEL" "$SEAL_MODEL_DIGEST" \
            "$CURRENT_DATA_DIGEST" >/dev/null 2>&1; then
        printf 'Native evaluation drift passed final seal validation.\n' >&2
        exit 1
    fi
    mv "$ROOT/eval.clean" "$EVAL_DATA_FILE"

    printf 'drift\n' >"$SEAL_MODEL/model-drift.bin"
    if revalidate_native_gate_seal_inputs "$CURRENT_COMMIT" \
            "$CURRENT_HANDOFF_DIGEST" "$SEAL_MODEL" "$SEAL_MODEL_DIGEST" \
            "$CURRENT_DATA_DIGEST" >/dev/null 2>&1; then
        printf 'Native checkpoint drift passed final seal validation.\n' >&2
        exit 1
    fi
    rm "$SEAL_MODEL/model-drift.bin"
    [[ ! -e "$ROOT/manifests/qwen-native-gate/preseal-drift.ok" ]]
)

SEAL_SAMPLING="$(native_sampling_from_resolved_config \
    "$EVAL_RUN/resolved-config.yaml" "$EVAL_RUN/run.env")"
SEAL_CONFIG_DIGEST="$(sha256sum "$EVAL_RUN/resolved-config.yaml" | cut -d' ' -f1)"
SEAL_RUN_ENV_DIGEST="$(sha256sum "$EVAL_RUN/run.env" | cut -d' ' -f1)"
revalidate_native_eval_seal_inputs "$EVAL_RUN" "$SEAL_SAMPLING" \
    "$SEAL_CONFIG_DIGEST" "$SEAL_RUN_ENV_DIGEST" "$RESULTS/sampling.json"
cp "$EVAL_RUN/resolved-config.yaml" "$ROOT/resolved-config.clean"
sed -i 's/"presence_penalty": 0.0/"presence_penalty": 1.0/' \
    "$EVAL_RUN/resolved-config.yaml"
if revalidate_native_eval_seal_inputs "$EVAL_RUN" "$SEAL_SAMPLING" \
        "$SEAL_CONFIG_DIGEST" "$SEAL_RUN_ENV_DIGEST" \
        "$RESULTS/sampling.json" >/dev/null 2>&1; then
    printf 'Resolved-config drift passed final seal validation.\n' >&2
    exit 1
fi
mv "$ROOT/resolved-config.clean" "$EVAL_RUN/resolved-config.yaml"
cp "$EVAL_RUN/run.env" "$ROOT/run-env.clean"
sed -i 's/rollout_top_k=0/rollout_top_k=1/' "$EVAL_RUN/run.env"
if revalidate_native_eval_seal_inputs "$EVAL_RUN" "$SEAL_SAMPLING" \
        "$SEAL_CONFIG_DIGEST" "$SEAL_RUN_ENV_DIGEST" \
        "$RESULTS/sampling.json" >/dev/null 2>&1; then
    printf 'run.env drift passed final seal validation.\n' >&2
    exit 1
fi
mv "$ROOT/run-env.clean" "$EVAL_RUN/run.env"
cp "$EVAL_RUN/run.env" "$ROOT/run-env.clean"
sed -i 's|input_model=/sealed/model|input_model=/drifted/model|' \
    "$EVAL_RUN/run.env"
if revalidate_native_eval_seal_inputs "$EVAL_RUN" "$SEAL_SAMPLING" \
        "$SEAL_CONFIG_DIGEST" "$SEAL_RUN_ENV_DIGEST" \
        "$RESULTS/sampling.json" >/dev/null 2>&1; then
    printf 'Unparsed run.env drift passed final seal validation.\n' >&2
    exit 1
fi
mv "$ROOT/run-env.clean" "$EVAL_RUN/run.env"
cp "$RESULTS/sampling.json" "$ROOT/sampling.clean"
printf '{}\n' >"$RESULTS/sampling.json"
if revalidate_native_eval_seal_inputs "$EVAL_RUN" "$SEAL_SAMPLING" \
        "$SEAL_CONFIG_DIGEST" "$SEAL_RUN_ENV_DIGEST" \
        "$RESULTS/sampling.json" >/dev/null 2>&1; then
    printf 'Sampling evidence drift passed final seal validation.\n' >&2
    exit 1
fi
mv "$ROOT/sampling.clean" "$RESULTS/sampling.json"

# Build a complete G2 envelope that includes the exact G0 marker, then ensure
# G3 follows that nested chain instead of trusting only a resealed G2 marker.
G0_EVIDENCE_DIGEST="$(sha256sum "$RESULTS/evidence.sha256" | cut -d' ' -f1)"
G2_OUTER_NAME=20260723T121000Z-124-457
G2_OUTER="$ROOT/state/attempts/gpu/$G2_OUTER_NAME"
G2_RESULTS="$ROOT/runs/qwen-native-gate/attempts/$G2_OUTER_NAME"
mkdir -p "$G2_OUTER" "$G2_RESULTS"
for file in summary.json summary.md go_no_go.json per_trajectory.jsonl \
    per_question.jsonl sampling.json; do
    cp "$RESULTS/$file" "$G2_RESULTS/$file"
done
"$PYTHON_BIN" - "$G2_RESULTS/go_no_go.json" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
value = json.loads(path.read_text(encoding="utf-8"))
value["stage"] = "g2"
value["decision"] = "GO"
value["failed_criteria"] = []
path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
PY
printf 'g2\n' >"$G2_RESULTS/stage.txt"
printf 'index\n' >"$G2_RESULTS/run-index.tsv"
printf '%s\n%s\n' \
    $'stage\tcheckpoint\tcheckpoint_digest\tevaluation_checkout_commit\tcpu_handoff_digest\tdata_manifest_sha256\tresolved_config_sha256\ttrace_sha256\ttrace_manifest_sha256\tpredecessor_evidence_sha256' \
    "g2"$'\t'"$ROOT/models/Qwen3.5-2B"$'\t'"$DIGEST"$'\t'"$CURRENT_COMMIT"$'\t'"$CURRENT_HANDOFF_DIGEST"$'\t'"$CURRENT_DATA_DIGEST"$'\t'"$(printf 'e%.0s' {1..64})"$'\t'"$(printf 'f%.0s' {1..64})"$'\t'"$(printf '1%.0s' {1..64})"$'\t'"$G0_EVIDENCE_DIGEST" \
    >"$G2_RESULTS/lineage.tsv"
printf 'success\n' >"$G2_OUTER/terminal"
printf '0\n' >"$G2_OUTER/exit-code"
: >"$G2_OUTER/.success"
NATIVE_STAGE=g2
NATIVE_PREDECESSOR_EVIDENCE="$MARKER"
EVAL_DATA_FILE="$DATA_DIR/probe_autonomous_32.parquet"
publish_native_gate_evidence "$G2_OUTER" "$G2_RESULTS" "$EVAL_RUN"
G2_MARKER="$ROOT/manifests/qwen-native-gate/$G2_OUTER_NAME.ok"

NATIVE_STAGE=g3
NATIVE_PREDECESSOR_EVIDENCE="$G2_MARKER"
verify_native_predecessor "$CURRENT_COMMIT" "$CURRENT_HANDOFF_DIGEST" \
    "$DIGEST" "$CURRENT_DATA_DIGEST"
sed -i "s/$G0_EVIDENCE_DIGEST/$ALT_G0_EVIDENCE_DIGEST/" \
    "$G2_RESULTS/lineage.tsv"
sed -i "s|${MARKER#"$ROOT/"}|${ALT_G0_MARKER#"$ROOT/"}|" \
    "$G2_RESULTS/evidence.sha256"
reseal_native_evidence "$G2_RESULTS/evidence.sha256" "$G2_MARKER" "$G2_OUTER"
if verify_native_predecessor "$CURRENT_COMMIT" "$CURRENT_HANDOFF_DIGEST" \
        "$DIGEST" "$CURRENT_DATA_DIGEST" >/dev/null 2>&1; then
    printf 'A G2 envelope spliced in an old but valid G0 lineage and unlocked G3.\n' >&2
    exit 1
fi

NATIVE_STAGE=g2
NATIVE_PREDECESSOR_EVIDENCE="$MARKER"
verify_native_predecessor "$CURRENT_COMMIT" "$CURRENT_HANDOFF_DIGEST" \
    "$DIGEST" "$CURRENT_DATA_DIGEST"
[[ "$PARENT_EVIDENCE_DIGEST" == "$(sha256sum "$RESULTS/evidence.sha256" | cut -d' ' -f1)" ]]

# Predecessor terminal evidence must be real files, and even a dangling
# running-marker symlink means the attempt is not safely terminal.
for file in terminal exit-code .success; do
    mv "$OUTER/$file" "$ROOT/${file#.}.clean"
    ln -s "$ROOT/${file#.}.clean" "$OUTER/$file"
    if verify_native_predecessor >/dev/null 2>&1; then
        printf 'A symlinked predecessor %s file was accepted.\n' "$file" >&2
        exit 1
    fi
    rm "$OUTER/$file"
    mv "$ROOT/${file#.}.clean" "$OUTER/$file"
done
ln -s "$ROOT/does-not-exist" "$OUTER/.running"
if verify_native_predecessor >/dev/null 2>&1; then
    printf 'A dangling predecessor running marker was accepted.\n' >&2
    exit 1
fi
rm "$OUTER/.running"
: >"$OUTER/.failed"
if verify_native_predecessor >/dev/null 2>&1; then
    printf 'Conflicting predecessor success and failed markers were accepted.\n' >&2
    exit 1
fi
rm "$OUTER/.failed"
ln -s "$ROOT/does-not-exist" "$OUTER/.starting"
if verify_native_predecessor >/dev/null 2>&1; then
    printf 'A dangling predecessor starting marker was accepted.\n' >&2
    exit 1
fi
rm "$OUTER/.starting"

cp "$RESULTS/summary.md" "$ROOT/summary.saved"
printf 'tampered\n' >>"$RESULTS/summary.md"
if verify_native_predecessor >/dev/null 2>&1; then
    printf 'Tampered predecessor evidence was accepted.\n' >&2
    exit 1
fi
mv "$ROOT/summary.saved" "$RESULTS/summary.md"

# Even a self-consistent old marker must not unlock a different checkout/handoff/data chain.
BAD_COMMIT="$(printf 'f%.0s' {1..40})"
sed -i "s/$CURRENT_COMMIT/$BAD_COMMIT/" "$RESULTS/lineage.tsv"
reseal_native_evidence "$RESULTS/evidence.sha256" "$MARKER" "$OUTER"
verify_native_predecessor
if verify_native_predecessor "$CURRENT_COMMIT" "$CURRENT_HANDOFF_DIGEST" \
        "$DIGEST" "$CURRENT_DATA_DIGEST" >/dev/null 2>&1; then
    printf 'A predecessor from a different checkout lineage was accepted.\n' >&2
    exit 1
fi

mv "$ROOT/manifests/qwen-native-gate" \
    "$ROOT/manifests/qwen-native-gate.saved"
ln -s "$ROOT/manifests/qwen-native-gate.saved" \
    "$ROOT/manifests/qwen-native-gate"
NATIVE_PREDECESSOR_EVIDENCE="$ROOT/manifests/qwen-native-gate.saved/$OUTER_NAME.ok"
if verify_native_predecessor >/dev/null 2>&1; then
    printf 'A predecessor marker reached through a symlinked marker directory was accepted.\n' >&2
    exit 1
fi

grep -Fq 'AUTODL_QWEN_NATIVE_INCREMENTAL' "$AUTODL_DIR/02_cpu_prepare.sh"
grep -Fq 'QWEN_NATIVE_DATA_DIR="$DATA_ROOT/search_mix_qwen35_native_v2"' \
    "$AUTODL_DIR/02_cpu_prepare.sh"
grep -Fq 'search_mix.py" materialize-native' "$AUTODL_DIR/02_cpu_prepare.sh"
grep -Fq -- '--no-reselection' "$AUTODL_DIR/02_cpu_prepare.sh"
grep -Fq 'verify-no-reselection' "$AUTODL_DIR/02_cpu_prepare.sh"
grep -Fq -- '--expected-tool-protocol qwen35_native' "$AUTODL_DIR/02_cpu_prepare.sh"
grep -Fq -- '--source-manifest "$SEARCH_MIX_DATA_DIR/manifest.json"' \
    "$AUTODL_DIR/02_cpu_prepare.sh"
grep -Fq 'if [[ "$build_qwen_native" != 1 ]]' \
    "$AUTODL_DIR/02_cpu_prepare.sh"
grep -Fq 'New output was fully verified before its atomic publication.' \
    "$AUTODL_DIR/02_cpu_prepare.sh"
grep -Fq 'QWEN35_CHAT_TEMPLATE_SHA256' "$AUTODL_DIR/02_cpu_prepare.sh"
grep -Fq 'add_generation_prompt=True, tokenize=True, return_dict=False' \
    "$AUTODL_DIR/02_cpu_prepare.sh"
[[ "$(grep -Fc 'if [[ "${AUTODL_QWEN_NATIVE_INCREMENTAL:-0}" != 1 ]]' \
    "$AUTODL_DIR/02_cpu_prepare.sh")" == 4 ]]
[[ "$(grep -Ec '^[[:space:]]*build_qwen_native=1$' \
    "$AUTODL_DIR/02_cpu_prepare.sh")" == 2 ]]
grep -Fq 'elif [[ -f "$QWEN_NATIVE_DATA_DIR/manifest.json" &&' \
    "$AUTODL_DIR/02_cpu_prepare.sh"
! grep -Fq 'elif [[ "$seal_qwen_native" == 0 &&' \
    "$AUTODL_DIR/02_cpu_prepare.sh"
[[ "$(grep -Fc 'rm -f -- "$MANIFEST_DIR/cpu.ok"' \
    "$AUTODL_DIR/02_cpu_prepare.sh")" == 1 ]]
for variant in smoke reproduce control cost_aware_gated; do
    grep -Fq "config-2gpu-qwen-native-$variant-train.yaml" \
        "$AUTODL_DIR/02_cpu_prepare.sh"
    grep -Fq -- \
        "--extra-file \"\$config_manifest_dir/config-2gpu-qwen-native-$variant-train.yaml\"" \
        "$AUTODL_DIR/02_cpu_prepare.sh"
done
for variant in qwen_native_b qwen_native_c; do
    grep -Fq "config-2gpu-$variant-eval.yaml" \
        "$AUTODL_DIR/02_cpu_prepare.sh"
    grep -Fq -- \
        "--extra-file \"\$config_manifest_dir/config-2gpu-$variant-eval.yaml\"" \
        "$AUTODL_DIR/02_cpu_prepare.sh"
done
PREFLIGHT_LINE="$(grep -n 'qwen_native_gate_preflight "$commit"' \
    "$AUTODL_DIR/03_gpu_run.sh" | cut -d: -f1)"
RETRIEVER_LINE="$(grep -n 'setsid "$RETRIEVER_ENV/bin/python"' \
    "$AUTODL_DIR/03_gpu_run.sh" | cut -d: -f1)"
[[ "$PREFLIGHT_LINE" -lt "$RETRIEVER_LINE" ]]

printf 'Qwen native gate pipeline tests passed.\n'
