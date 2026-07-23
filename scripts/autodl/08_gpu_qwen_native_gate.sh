#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
NATIVE_PROJECT_ROOT="${AUTODL_ROOT:-/root/autodl-tmp/search-r1}"
NATIVE_STAGE="${QWEN_NATIVE_GATE_STAGE:-}"
NATIVE_PREDECESSOR_EVIDENCE="${QWEN_NATIVE_PREDECESSOR_EVIDENCE:-}"
NATIVE_DATA_DIR="$NATIVE_PROJECT_ROOT/data/search_mix_qwen35_native"
export DATA_DIR="$NATIVE_DATA_DIR"

export AUTODL_RUN_BUDGET_PROFILE=gated_followup
export TOOL_PROTOCOL=qwen35_native
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-500}"
case "$NATIVE_STAGE" in
    g0_g1)
        export EVAL_DATA_FILE="$NATIVE_DATA_DIR/probe_forced_16.parquet"
        export EVAL_EXPECTED_ROWS=16
        export EVAL_GROUP_SIZE=2
        ;;
    g2)
        export EVAL_DATA_FILE="$NATIVE_DATA_DIR/probe_autonomous_32.parquet"
        export EVAL_EXPECTED_ROWS=32
        export EVAL_GROUP_SIZE=3
        ;;
    g3)
        export EVAL_DATA_FILE="$NATIVE_DATA_DIR/probe_multi_64.parquet"
        export EVAL_EXPECTED_ROWS=64
        export EVAL_GROUP_SIZE=5
        ;;
    *)
        export EVAL_DATA_FILE=''
        export EVAL_EXPECTED_ROWS=1
        export EVAL_GROUP_SIZE=1
        ;;
esac
AUTODL_GPU_PIPELINE=qwen_native_gate
# shellcheck source=03_gpu_run.sh
source "$SCRIPT_DIR/03_gpu_run.sh"

readonly NATIVE_RESULTS_ROOT="$RUNS_ROOT/qwen-native-gate"
readonly NATIVE_MANIFEST="$NATIVE_DATA_DIR/manifest.json"
readonly NATIVE_CATALOG="$NATIVE_DATA_DIR/catalog.jsonl"
readonly LEGACY_DATA_DIR="$PROJECT_ROOT/data/search_mix"
readonly LEGACY_REPLAY_RECEIPT="$LEGACY_DATA_DIR/retrieval_replay.json"
readonly PROTOCOL_PROBE_CLI="$CHECKOUT_DIR/scripts/autodl/qwen_native_protocol_probe.py"
readonly GATE_ANALYSIS_CLI="$CHECKOUT_DIR/scripts/autodl/qwen_native_gate_analysis.py"
readonly NATIVE_FINALIZE_GRACE_SECONDS=180

PARENT_EVIDENCE_DIGEST='-'
NATIVE_PREFLIGHT_COMMIT=''
NATIVE_PREFLIGHT_HANDOFF_DIGEST=''
NATIVE_PREFLIGHT_CHECKPOINT_DIGEST=''
NATIVE_PREFLIGHT_DATA_DIGEST=''
NATIVE_PREFLIGHT_INPUT_DIGEST=''
NATIVE_PREFLIGHT_PARENT_EVIDENCE_DIGEST=''

require_native_gate() {
    validate_gpu_inputs
    [[ "$GPU_COUNT" == 2 && "$TRAIN_BATCH_SIZE" == 8 &&
        "$MAX_RESPONSE_LENGTH" == 500 && "$TOOL_PROTOCOL" == qwen35_native ]] || {
        printf 'Qwen native gates require two GPUs, batch 8, response 500, and qwen35_native.\n' >&2
        return 64
    }
    [[ -f "$PROTOCOL_PROBE_CLI" && ! -L "$PROTOCOL_PROBE_CLI" &&
        -f "$GATE_ANALYSIS_CLI" && ! -L "$GATE_ANALYSIS_CLI" ]] || {
        printf 'Qwen native gate CLIs are missing or symlinked.\n' >&2
        return 1
    }
    case "$NATIVE_STAGE" in
        g0_g1)
            [[ "$EVAL_DATA_FILE" == "$NATIVE_DATA_DIR/probe_forced_16.parquet" &&
                "$EVAL_EXPECTED_ROWS" == 16 && "$EVAL_GROUP_SIZE" == 2 &&
                -z "$NATIVE_PREDECESSOR_EVIDENCE" ]] || {
                printf 'G0+G1 is fixed to 16x2 and must not name a predecessor.\n' >&2
                return 64
            }
            ;;
        g2)
            [[ "$EVAL_DATA_FILE" == "$NATIVE_DATA_DIR/probe_autonomous_32.parquet" &&
                "$EVAL_EXPECTED_ROWS" == 32 && "$EVAL_GROUP_SIZE" == 3 &&
                -n "$NATIVE_PREDECESSOR_EVIDENCE" ]] || {
                printf 'G2 is fixed to 32x3 and requires exact G0+G1 evidence.\n' >&2
                return 64
            }
            ;;
        g3)
            [[ "$EVAL_DATA_FILE" == "$NATIVE_DATA_DIR/probe_multi_64.parquet" &&
                "$EVAL_EXPECTED_ROWS" == 64 && "$EVAL_GROUP_SIZE" == 5 &&
                -n "$NATIVE_PREDECESSOR_EVIDENCE" ]] || {
                printf 'G3 is fixed to 64x5 and requires exact G2 evidence.\n' >&2
                return 64
            }
            ;;
        *)
            printf 'Set QWEN_NATIVE_GATE_STAGE to g0_g1, g2, or g3.\n' >&2
            return 64
            ;;
    esac
}

native_stage_budget_rmb() {
    case "$NATIVE_STAGE" in
        g0_g1) printf '2\n' ;;
        g2) printf '3\n' ;;
        g3) printf '10\n' ;;
        *) return 64 ;;
    esac
}

native_stage_timeout_seconds() {
    local budget
    budget="$(native_stage_budget_rmb)"
    awk -v budget="$budget" -v price="$PRICE_PER_HOUR" \
        'BEGIN { printf "%d", budget / price * 3600 }'
}

native_remaining_seconds() {
    local now
    [[ "${QWEN_NATIVE_DEADLINE_EPOCH:-}" =~ ^[1-9][0-9]*$ ]] || return 64
    now="$(date +%s)"
    printf '%s\n' "$((QWEN_NATIVE_DEADLINE_EPOCH - now))"
}

native_deadline_check() {
    local label="$1" remaining
    remaining="$(native_remaining_seconds)" || {
        printf 'Qwen native deadline is missing before %s.\n' "$label" >&2
        return 64
    }
    ((remaining > 0)) || {
        printf 'Qwen native stage deadline expired before %s.\n' "$label" >&2
        return 124
    }
}

native_work_timeout_seconds() {
    local label="$1" remaining available
    remaining="$(native_remaining_seconds)" || return $?
    available=$((remaining - NATIVE_FINALIZE_GRACE_SECONDS))
    ((available > 0)) || {
        printf 'Qwen native deadline has no finalize reserve before %s.\n' \
            "$label" >&2
        return 124
    }
    printf '%s\n' "$available"
}

run_with_native_deadline() {
    local timeout_seconds="$1"
    shift
    [[ "$timeout_seconds" =~ ^[1-9][0-9]*$ ]] || return 64
    timeout --signal=TERM --kill-after=120s "${timeout_seconds}s" "$@"
}

verify_native_data_contract() {
    local path
    local -a required=(
        "$NATIVE_MANIFEST"
        "$NATIVE_MANIFEST.sha256"
        "$NATIVE_CATALOG"
        "$NATIVE_DATA_DIR/probe_g0_8.parquet"
        "$NATIVE_DATA_DIR/probe_forced_16.parquet"
        "$NATIVE_DATA_DIR/probe_autonomous_32.parquet"
        "$NATIVE_DATA_DIR/probe_multi_64.parquet"
        "$LEGACY_REPLAY_RECEIPT"
        "$LEGACY_REPLAY_RECEIPT.sha256"
    )
    for path in "${required[@]}"; do
        [[ -f "$path" && ! -L "$path" ]] || {
            printf 'Qwen native data evidence is missing or symlinked: %s\n' "$path" >&2
            return 1
        }
    done
    "$TRAIN_ENV/bin/python" "$CHECKOUT_DIR/scripts/data_process/search_mix.py" verify \
        --manifest "$NATIVE_MANIFEST" \
        --source-manifest "$LEGACY_DATA_DIR/manifest.json" \
        --model-dir "$MODEL_DIR" \
        --eval-catalog "$PROJECT_ROOT/data/search_opportunity_gate/catalog.jsonl" \
        --eval-parquet "$PROJECT_ROOT/data/nq_small/test_128.parquet" \
        --expected-tool-protocol qwen35_native
}

native_project_relative_file() {
    local path="$1" canonical root
    [[ -f "$path" && ! -L "$path" ]] || return 1
    canonical="$(readlink -f -- "$path")" || return 1
    root="$(readlink -f -- "$PROJECT_ROOT")" || return 1
    [[ "$canonical" == "$root/"* ]] || return 1
    printf '%s\n' "${canonical#"$root/"}"
}

verify_native_checksum_manifest() {
    local manifest="$1" manifest_relative line relative path actual_relative
    local checksum_pattern='^[0-9a-f]{64}  (.+)$'
    local count=0
    local -A seen=()
    [[ -f "$manifest" && ! -L "$manifest" ]] || {
        printf 'Qwen native checksum manifest is missing or symlinked: %s\n' \
            "$manifest" >&2
        return 1
    }
    manifest_relative="$(native_project_relative_file "$manifest")" || return 1
    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ "$line" =~ $checksum_pattern ]] || {
            printf 'Malformed Qwen native checksum entry in %s\n' "$manifest" >&2
            return 1
        }
        relative="${BASH_REMATCH[1]}"
        [[ "$relative" != /* && -z "${seen[$relative]+present}" ]] || {
            printf 'Unsafe or duplicate Qwen native evidence path: %s\n' \
                "$relative" >&2
            return 1
        }
        seen["$relative"]=1
        path="$PROJECT_ROOT/$relative"
        actual_relative="$(native_project_relative_file "$path")" || {
            printf 'Qwen native evidence is missing, symlinked, or escaped: %s\n' \
                "$relative" >&2
            return 1
        }
        [[ "$actual_relative" == "$relative" ]] || {
            printf 'Qwen native evidence path drifted: %s\n' "$relative" >&2
            return 1
        }
        ((count += 1))
    done <"$manifest"
    ((count > 0)) || {
        printf 'Qwen native checksum manifest is empty: %s\n' "$manifest" >&2
        return 1
    }
    if ! (
        cd "$PROJECT_ROOT"
        sha256sum --strict --check "$manifest_relative" >/dev/null
    ); then
        printf 'Qwen native evidence checksum verification failed: %s\n' \
            "$manifest" >&2
        return 1
    fi
}

verify_native_attempt_binding() {
    local outer="$1" results="$2" marker="$3" evidence_digest="$4"
    local contract recorded_results recorded_marker recorded_digest path
    for path in result-contract result-root evidence-marker evidence-digest; do
        [[ -f "$outer/$path" && ! -L "$outer/$path" ]] || {
            printf 'Qwen native predecessor attempt binding is missing: %s\n' \
                "$outer/$path" >&2
            return 1
        }
    done
    contract="$(tr -d '\r\n' <"$outer/result-contract")" || return 1
    recorded_results="$(tr -d '\r\n' <"$outer/result-root")" || return 1
    recorded_marker="$(tr -d '\r\n' <"$outer/evidence-marker")" || return 1
    recorded_digest="$(tr -d '\r\n' <"$outer/evidence-digest")" || return 1
    [[ "$contract" == qwen-native-gate-v1 &&
        "$recorded_results" == "$results" &&
        "$recorded_marker" == "$marker" &&
        "$recorded_digest" == "$evidence_digest" ]] || {
        printf 'Qwen native predecessor attempt binding does not match its marker and result.\n' >&2
        return 1
    }
}

verify_native_g3_g0_chain() {
    local g2_evidence="$1" lineage_parent_digest="$2"
    local expected_checkpoint="$3" expected_commit="$4"
    local expected_handoff="$5" expected_data="$6"
    local line relative g0_marker_relative='' g0_marker='' g0_attempt=''
    local g0_results='' g0_outer='' g0_evidence='' g0_marker_digest=''
    local g0_evidence_digest='' metadata stage decision lineage_metadata
    local g0_checkpoint g0_commit g0_handoff g0_data g0_parent_digest
    local checksum_pattern='^[0-9a-f]{64}  (.+)$'
    local marker_pattern='^manifests/qwen-native-gate/([0-9]{8}T[0-9]{6}Z-[0-9]+-[0-9]+)[.]ok$'
    local marker_count=0
    [[ "$lineage_parent_digest" =~ ^[0-9a-f]{64}$ ]] || {
        printf 'G2 lineage has no valid G0 predecessor evidence digest.\n' >&2
        return 1
    }
    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ "$line" =~ $checksum_pattern ]] || return 1
        relative="${BASH_REMATCH[1]}"
        if [[ "$relative" =~ $marker_pattern ]]; then
            ((marker_count += 1))
            g0_marker_relative="$relative"
            g0_attempt="${BASH_REMATCH[1]}"
        fi
    done <"$g2_evidence"
    ((marker_count == 1)) || {
        printf 'G2 evidence must contain exactly one G0 predecessor marker.\n' >&2
        return 1
    }
    g0_marker="$PROJECT_ROOT/$g0_marker_relative"
    relative="$(native_project_relative_file "$g0_marker")" || return 1
    [[ "$relative" == "$g0_marker_relative" ]] || return 1
    g0_marker_digest="$(tr -d '\r\n' <"$g0_marker")" || return 1
    [[ "$g0_marker_digest" == "$lineage_parent_digest" ]] || {
        printf 'G2 lineage does not name the exact G0 marker content.\n' >&2
        return 1
    }

    g0_results="$NATIVE_RESULTS_ROOT/attempts/$g0_attempt"
    g0_outer="$ATTEMPTS_ROOT/gpu/$g0_attempt"
    g0_evidence="$g0_results/evidence.sha256"
    [[ -d "$g0_results" && ! -L "$g0_results" &&
        -d "$g0_outer" && ! -L "$g0_outer" &&
        -f "$g0_evidence" && ! -L "$g0_evidence" &&
        -f "$g0_outer/terminal" && ! -L "$g0_outer/terminal" &&
        -f "$g0_outer/exit-code" && ! -L "$g0_outer/exit-code" &&
        "$(tr -d '\r\n' <"$g0_outer/terminal")" == success &&
        "$(tr -d '\r\n' <"$g0_outer/exit-code")" == 0 &&
        -f "$g0_outer/.success" && ! -L "$g0_outer/.success" &&
        ! -e "$g0_outer/.starting" && ! -L "$g0_outer/.starting" &&
        ! -e "$g0_outer/.running" && ! -L "$g0_outer/.running" &&
        ! -e "$g0_outer/.failed" && ! -L "$g0_outer/.failed" ]] || {
        printf 'G0 predecessor attempt has no complete success evidence.\n' >&2
        return 1
    }
    g0_evidence_digest="$(file_sha256 "$g0_evidence")" || return 1
    [[ "$g0_evidence_digest" == "$g0_marker_digest" ]] || {
        printf 'G0 evidence digest does not match the marker included by G2.\n' >&2
        return 1
    }
    verify_native_checksum_manifest "$g0_evidence" || return 1
    verify_native_attempt_binding "$g0_outer" "$g0_results" "$g0_marker" \
        "$g0_evidence_digest" || return 1
    metadata="$("$TRAIN_ENV/bin/python" - "$g0_results/go_no_go.json" <<'PY'
import json
from pathlib import Path
import sys

value = json.loads(Path(sys.argv[1]).read_bytes())
print(value.get("stage", ""), value.get("decision", ""), sep="\t")
PY
    )" || return 1
    IFS=$'\t' read -r stage decision <<<"$metadata"
    [[ "$stage" == g0_g1 && "$decision" == GO ]] || {
        printf 'G2 evidence does not resolve to a G0+G1 GO result.\n' >&2
        return 1
    }
    lineage_metadata="$("$TRAIN_ENV/bin/python" - "$g0_results/lineage.tsv" <<'PY'
import csv
from pathlib import Path
import sys

path = Path(sys.argv[1])
if not path.is_file() or path.is_symlink():
    raise SystemExit("G0 lineage is missing or symlinked")
expected = {
    "stage", "checkpoint", "checkpoint_digest",
    "evaluation_checkout_commit", "cpu_handoff_digest",
    "data_manifest_sha256", "resolved_config_sha256", "trace_sha256",
    "trace_manifest_sha256", "predecessor_evidence_sha256",
}
with path.open(encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    if reader.fieldnames is None or set(reader.fieldnames) != expected:
        raise SystemExit("G0 lineage schema mismatch")
    rows = list(reader)
if len(rows) != 1 or rows[0]["stage"] != "g0_g1":
    raise SystemExit("G0 lineage must contain one G0+G1 row")
row = rows[0]
print(row["checkpoint_digest"], row["evaluation_checkout_commit"],
      row["cpu_handoff_digest"], row["data_manifest_sha256"],
      row["predecessor_evidence_sha256"], sep="\t")
PY
    )" || return 1
    IFS=$'\t' read -r g0_checkpoint g0_commit g0_handoff g0_data \
        g0_parent_digest <<<"$lineage_metadata"
    [[ "$g0_checkpoint" == "$expected_checkpoint" &&
        "$g0_commit" == "$expected_commit" &&
        "$g0_handoff" == "$expected_handoff" &&
        "$g0_data" == "$expected_data" && "$g0_parent_digest" == - ]] || {
        printf 'G0 lineage does not match the G2 checkpoint, checkout, handoff, and data.\n' >&2
        return 1
    }
}

verify_native_predecessor() {
    local current_commit="${1:-}" current_handoff_digest="${2:-}"
    local current_checkpoint_digest="${3:-}" current_data_digest="${4:-}"
    local expected_stage marker marker_parent attempt_name results evidence outer
    local marker_digest evidence_digest decision stage go_metadata lineage_metadata
    local predecessor_checkpoint predecessor_commit predecessor_handoff predecessor_data
    local predecessor_parent_digest
    [[ $# == 0 || $# == 4 ]] || {
        printf 'Predecessor verification accepts either zero or four current lineage values.\n' >&2
        return 64
    }
    [[ "$NATIVE_STAGE" == g2 || "$NATIVE_STAGE" == g3 ]] || {
        PARENT_EVIDENCE_DIGEST='-'
        return 0
    }
    expected_stage=g0_g1
    [[ "$NATIVE_STAGE" == g3 ]] && expected_stage=g2
    marker="$(readlink -f -- "$NATIVE_PREDECESSOR_EVIDENCE")" || return 1
    [[ "$marker" == "$NATIVE_PREDECESSOR_EVIDENCE" && -f "$marker" && ! -L "$marker" ]] || {
        printf 'Pass the canonical regular predecessor evidence marker.\n' >&2
        return 1
    }
    [[ -d "$MANIFEST_DIR/qwen-native-gate" &&
        ! -L "$MANIFEST_DIR/qwen-native-gate" ]] || {
        printf 'Predecessor marker directory is missing or symlinked.\n' >&2
        return 1
    }
    marker_parent="$(readlink -f -- "$MANIFEST_DIR/qwen-native-gate")" || return 1
    [[ "$marker_parent" == "$(readlink -f -- "$MANIFEST_DIR")/qwen-native-gate" ]] || {
        printf 'Predecessor marker directory escaped the manifest root.\n' >&2
        return 1
    }
    [[ "$(readlink -f -- "$(dirname -- "$marker")")" == "$marker_parent" &&
        "$(basename -- "$marker")" =~ ^[0-9]{8}T[0-9]{6}Z-[0-9]+-[0-9]+[.]ok$ ]] || {
        printf 'Predecessor marker is outside the registered marker directory.\n' >&2
        return 1
    }
    attempt_name="$(basename -- "$marker" .ok)"
    results="$NATIVE_RESULTS_ROOT/attempts/$attempt_name"
    outer="$ATTEMPTS_ROOT/gpu/$attempt_name"
    evidence="$results/evidence.sha256"
    [[ -d "$results" && ! -L "$results" && -d "$outer" && ! -L "$outer" &&
        -f "$evidence" && ! -L "$evidence" &&
        -f "$outer/terminal" && ! -L "$outer/terminal" &&
        -f "$outer/exit-code" && ! -L "$outer/exit-code" &&
        "$(tr -d '\r\n' <"$outer/terminal")" == success &&
        "$(tr -d '\r\n' <"$outer/exit-code")" == 0 &&
        -f "$outer/.success" && ! -L "$outer/.success" &&
        ! -e "$outer/.starting" && ! -L "$outer/.starting" &&
        ! -e "$outer/.running" && ! -L "$outer/.running" &&
        ! -e "$outer/.failed" && ! -L "$outer/.failed" ]] || {
        printf 'Predecessor attempt has no complete success evidence.\n' >&2
        return 1
    }
    evidence_digest="$(file_sha256 "$evidence")" || return 1
    marker_digest="$(tr -d '\r\n' <"$marker")" || return 1
    [[ "$marker_digest" == "$evidence_digest" ]] || {
        printf 'Predecessor marker digest mismatch.\n' >&2
        return 1
    }
    verify_native_attempt_binding "$outer" "$results" "$marker" \
        "$evidence_digest" || return 1
    verify_native_checksum_manifest "$evidence" || return 1
    go_metadata="$("$TRAIN_ENV/bin/python" - "$results/go_no_go.json" <<'PY'
import json
from pathlib import Path
import sys

value = json.loads(Path(sys.argv[1]).read_bytes())
print(value.get("stage", ""), value.get("decision", ""), sep="\t")
PY
    )" || return 1
    IFS=$'\t' read -r stage decision <<<"$go_metadata"
    [[ "$stage" == "$expected_stage" && "$decision" == GO ]] || {
        printf 'Predecessor is not the required %s GO result.\n' "$expected_stage" >&2
        return 1
    }
    lineage_metadata="$("$TRAIN_ENV/bin/python" - "$results/lineage.tsv" <<'PY'
import csv
from pathlib import Path
import sys

path = Path(sys.argv[1])
if not path.is_file() or path.is_symlink():
    raise SystemExit("predecessor lineage is missing or symlinked")
with path.open(encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    required = {
        "checkpoint_digest", "evaluation_checkout_commit",
        "cpu_handoff_digest", "data_manifest_sha256",
        "predecessor_evidence_sha256",
    }
    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
        raise SystemExit("predecessor lineage schema mismatch")
    rows = list(reader)
if len(rows) != 1:
    raise SystemExit("predecessor lineage must contain exactly one row")
row = rows[0]
print(row["checkpoint_digest"], row["evaluation_checkout_commit"],
      row["cpu_handoff_digest"], row["data_manifest_sha256"],
      row["predecessor_evidence_sha256"], sep="\t")
PY
    )" || return 1
    IFS=$'\t' read -r predecessor_checkpoint predecessor_commit \
        predecessor_handoff predecessor_data predecessor_parent_digest \
        <<<"$lineage_metadata"
    if [[ $# == 4 ]]; then
        [[ "$predecessor_checkpoint" == "$current_checkpoint_digest" &&
            "$predecessor_commit" == "$current_commit" &&
            "$predecessor_handoff" == "$current_handoff_digest" &&
            "$predecessor_data" == "$current_data_digest" ]] || {
            printf 'Predecessor lineage does not match the current checkpoint, checkout, handoff, and data.\n' >&2
            return 1
        }
    fi
    if [[ "$NATIVE_STAGE" == g3 ]]; then
        verify_native_g3_g0_chain "$evidence" "$predecessor_parent_digest" \
            "$predecessor_checkpoint" "$predecessor_commit" \
            "$predecessor_handoff" "$predecessor_data" || return 1
    fi
    PARENT_EVIDENCE_DIGEST="$evidence_digest"
}

native_gate_input_digest() {
    local path digest input_lines=''
    for path in \
        "$NATIVE_MANIFEST" \
        "$NATIVE_CATALOG" \
        "$EVAL_DATA_FILE" \
        "$LEGACY_DATA_DIR/manifest.json" \
        "$LEGACY_DATA_DIR/catalog.jsonl" \
        "$LEGACY_REPLAY_RECEIPT" \
        "$LEGACY_REPLAY_RECEIPT.sha256"; do
        digest="$(file_sha256 "$path")" || return 1
        input_lines+="$digest  $path"$'\n'
    done
    printf '%s' "$input_lines" | sha256sum | cut -d' ' -f1
}

verify_native_handoff_seal() {
    local expected_digest="$1" actual_digest sidecar_value marker_value
    [[ "$expected_digest" =~ ^[0-9a-f]{64}$ &&
        -f "$HANDOFF" && ! -L "$HANDOFF" &&
        -f "$HANDOFF.sha256" && ! -L "$HANDOFF.sha256" &&
        -f "$MANIFEST_DIR/cpu.ok" && ! -L "$MANIFEST_DIR/cpu.ok" ]] || {
        printf 'CPU handoff seal is missing, symlinked, or malformed.\n' >&2
        return 1
    }
    actual_digest="$(file_sha256 "$HANDOFF")" || return 1
    sidecar_value="$(tr -d '\r\n' <"$HANDOFF.sha256")" || return 1
    marker_value="$(tr -d '\r\n' <"$MANIFEST_DIR/cpu.ok")" || return 1
    [[ "$actual_digest" == "$expected_digest" &&
        "$sidecar_value" == "$expected_digest  $(basename -- "$HANDOFF")" &&
        "$marker_value" == "$expected_digest" ]] || {
        printf 'CPU handoff seal changed after GPU admission.\n' >&2
        return 1
    }
}

qwen_native_gate_preflight() {
    local commit="$1" handoff_digest="$2" checkpoint_digest="$3"
    local data_digest input_digest
    export PYTHONPATH="$CHECKOUT_DIR${PYTHONPATH:+:$PYTHONPATH}"
    native_deadline_check preflight
    require_native_gate
    verify_native_data_contract
    data_digest="$(file_sha256 "$NATIVE_MANIFEST")" || return 1
    input_digest="$(native_gate_input_digest)" || return 1
    verify_native_predecessor "$commit" "$handoff_digest" \
        "$checkpoint_digest" "$data_digest"
    NATIVE_PREFLIGHT_COMMIT="$commit"
    NATIVE_PREFLIGHT_HANDOFF_DIGEST="$handoff_digest"
    NATIVE_PREFLIGHT_CHECKPOINT_DIGEST="$checkpoint_digest"
    NATIVE_PREFLIGHT_DATA_DIGEST="$data_digest"
    NATIVE_PREFLIGHT_INPUT_DIGEST="$input_digest"
    NATIVE_PREFLIGHT_PARENT_EVIDENCE_DIGEST="$PARENT_EVIDENCE_DIGEST"
}

revalidate_native_gate_seal_inputs() {
    local commit="$1" handoff_digest="$2" base_model="$3" base_digest="$4"
    local initial_data_digest="$5" current_data_digest current_input_digest
    local current_model_digest
    verify_native_handoff_seal "$handoff_digest" || return 1
    verify_native_data_contract || return 1
    current_data_digest="$(file_sha256 "$NATIVE_MANIFEST")" || return 1
    current_input_digest="$(native_gate_input_digest)" || return 1
    [[ "$NATIVE_PREFLIGHT_COMMIT" == "$commit" &&
        "$NATIVE_PREFLIGHT_HANDOFF_DIGEST" == "$handoff_digest" &&
        "$NATIVE_PREFLIGHT_CHECKPOINT_DIGEST" == "$base_digest" &&
        "$NATIVE_PREFLIGHT_DATA_DIGEST" == "$initial_data_digest" &&
        "$current_data_digest" == "$NATIVE_PREFLIGHT_DATA_DIGEST" &&
        "$current_input_digest" == "$NATIVE_PREFLIGHT_INPUT_DIGEST" &&
        "$PARENT_EVIDENCE_DIGEST" == \
            "$NATIVE_PREFLIGHT_PARENT_EVIDENCE_DIGEST" ]] || {
        printf 'Qwen native inputs or recorded lineage drifted before evidence sealing.\n' >&2
        return 1
    }
    current_model_digest="$(tree_sha256 "$base_model")" || return 1
    [[ "$current_model_digest" == "$base_digest" ]] || {
        printf 'Qwen native checkpoint changed before evidence sealing.\n' >&2
        return 1
    }
    verify_native_predecessor "$commit" "$handoff_digest" \
        "$base_digest" "$current_data_digest" || return 1
    [[ "$PARENT_EVIDENCE_DIGEST" == \
        "$NATIVE_PREFLIGHT_PARENT_EVIDENCE_DIGEST" ]] || {
        printf 'Qwen native predecessor evidence changed before sealing.\n' >&2
        return 1
    }
}

native_sampling_from_resolved_config() {
    local resolved_config="$1" run_env="$2"
    "$TRAIN_ENV/bin/python" - "$resolved_config" "$run_env" \
        "$NATIVE_DATA_DIR/train_512.parquet" "$EVAL_DATA_FILE" \
        "$EVAL_GROUP_SIZE" <<'PY'
import json
from pathlib import Path
import sys

config_path, run_env_path, train_file, eval_file = map(Path, sys.argv[1:5])
expected_group = int(sys.argv[5])
for path in (config_path, run_env_path):
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"native run evidence is missing or symlinked: {path}")
raw = config_path.read_text(encoding="utf-8")
try:
    config = json.loads(raw)
except json.JSONDecodeError:
    from omegaconf import OmegaConf
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
if not isinstance(config, dict):
    raise SystemExit("resolved config must be a mapping")

def value(*keys):
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            raise SystemExit(f"resolved config is missing {'.'.join(keys)}")
        current = current[key]
    return current

sampling = {
    "temperature": float(value("actor_rollout_ref", "rollout", "temperature")),
    "top_p": float(value("actor_rollout_ref", "rollout", "top_p")),
    "top_k": int(value("actor_rollout_ref", "rollout", "top_k")),
    "min_p": float(value("actor_rollout_ref", "rollout", "min_p")),
    "presence_penalty": float(value("actor_rollout_ref", "rollout", "presence_penalty")),
    "repetition_penalty": float(value("actor_rollout_ref", "rollout", "repetition_penalty")),
    "tool_protocol": value("tool_protocol"),
}
expected_sampling = {
    "temperature": 1.0, "top_p": 1.0, "top_k": 20, "min_p": 0.0,
    "presence_penalty": 2.0, "repetition_penalty": 1.0,
    "tool_protocol": "qwen35_native",
}
checks = {
    "sampling": sampling == expected_sampling,
    "train_file": Path(value("data", "train_files")) == train_file,
    "eval_file": Path(value("data", "val_files")) == eval_file,
    "train_batch_size": value("data", "train_batch_size") == 8,
    "val_batch_size": value("data", "val_batch_size") == 8,
    "eval_group_size": value("data", "eval_group_size") == expected_group,
    "return_raw_chat": value("data", "return_raw_chat") is True,
    "max_prompt_length": value("data", "max_prompt_length") == 4096,
    "max_response_length": value("data", "max_response_length") == 500,
    "max_start_length": value("data", "max_start_length") == 1024,
    "max_obs_length": value("data", "max_obs_length") == 384,
    "max_turns": value("max_turns") == 4,
    "retriever_topk": value("retriever", "topk") == 3,
}
failed = sorted(name for name, passed in checks.items() if not passed)
if failed:
    raise SystemExit("native resolved config mismatch: " + ", ".join(failed))

run_env = {}
for line in run_env_path.read_text(encoding="utf-8").splitlines():
    if not line or "=" not in line:
        continue
    key, item = line.split("=", 1)
    if key in run_env:
        raise SystemExit(f"duplicate run.env key: {key}")
    run_env[key] = item
expected_env = {
    "data_dir": str(train_file.parent),
    "eval_data_file": str(eval_file),
    "eval_group_size": str(expected_group),
    "max_response_length": "500",
    "tool_protocol": "qwen35_native",
    "rollout_top_k": "20",
    "rollout_min_p": "0.0",
    "rollout_presence_penalty": "2.0",
    "rollout_repetition_penalty": "1.0",
}
if any(run_env.get(key) != expected for key, expected in expected_env.items()):
    raise SystemExit("run.env does not match the resolved native config")
print(json.dumps(sampling, sort_keys=True, separators=(",", ":")))
PY
}

revalidate_native_eval_seal_inputs() {
    local eval_run="$1" expected_sampling="$2" expected_config_digest="$3"
    local expected_run_env_digest="$4" sampling_file="$5"
    local current_sampling current_config_digest current_run_env_digest
    local recorded_sampling
    current_sampling="$(native_sampling_from_resolved_config \
        "$eval_run/resolved-config.yaml" "$eval_run/run.env")" || return 1
    current_config_digest="$(file_sha256 \
        "$eval_run/resolved-config.yaml")" || return 1
    current_run_env_digest="$(file_sha256 "$eval_run/run.env")" || return 1
    [[ -f "$sampling_file" && ! -L "$sampling_file" ]] || return 1
    recorded_sampling="$(tr -d '\r\n' <"$sampling_file")" || return 1
    [[ "$current_sampling" == "$expected_sampling" &&
        "$current_config_digest" == "$expected_config_digest" &&
        "$current_run_env_digest" == "$expected_run_env_digest" &&
        "$recorded_sampling" == "$expected_sampling" ]] || {
        printf 'Qwen native config, run.env, or sampling evidence changed before sealing.\n' >&2
        return 1
    }
}

verify_native_trace_evidence() {
    local trace_manifest="$1" results_dir="$2" expected_rows="$3"
    PYTHONPATH="$CHECKOUT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
        "$TRAIN_ENV/bin/python" - "$trace_manifest" "$results_dir" \
        "$expected_rows" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

from search_r1.trajectory_trace import verify_trace_manifest

manifest_path, results_dir = map(Path, sys.argv[1:3])
manifest = verify_trace_manifest(manifest_path, int(sys.argv[3]))
trace_sha256 = manifest["artifact"]["sha256"]
summary = json.loads((results_dir / "summary.json").read_bytes())
decision = json.loads((results_dir / "go_no_go.json").read_bytes())
summary_trace = summary.get("trace_sha256")
if summary_trace is None and isinstance(summary.get("input"), dict):
    summary_trace = summary["input"].get("trace_sha256")
if summary_trace != trace_sha256 or decision.get("trace_sha256") != trace_sha256:
    raise SystemExit("analysis trace digest does not match the verified trace manifest")
manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
print(trace_sha256, manifest_sha256, sep="\t")
PY
}

finish_protocol_probe_run() {
    local run_dir="$1" rc="$2" started_epoch="$3" timeout_seconds="$4"
    local state marker elapsed
    elapsed=$(( $(date +%s) - started_epoch ))
    if ((rc == 0)); then state=success; marker=.success; else state=failed; marker=.failed; fi
    atomic_write "$run_dir/exit-code" "$rc"$'\n'
    atomic_write "$run_dir/run.env" \
        "finished_at=$(utc_now)"$'\n'\
"elapsed_seconds=$elapsed"$'\n'\
"job_mode=protocol_probe"$'\n'\
"variant=qwen_native_g0"$'\n'\
"seed=42"$'\n'\
"tool_protocol=qwen35_native"$'\n'\
"timeout_seconds=$timeout_seconds"$'\n'\
"price_per_hour=$PRICE_PER_HOUR"$'\n'\
"budget_rmb=1"$'\n'
    atomic_write "$run_dir/terminal" "$state"$'\n'
    : >"$run_dir/$marker"
    sync_path "$run_dir"
    rm -f -- "$run_dir/.running"
    sync_path "$run_dir"
}

run_protocol_probe() {
    local base_model="$1" base_digest="$2"
    local parent run_dir timeout_seconds available_seconds started_epoch rc
    timeout_seconds="$(awk -v budget=1 -v price="$PRICE_PER_HOUR" \
        'BEGIN { printf "%d", budget / price * 3600 }')"
    if [[ -n "${QWEN_NATIVE_DEADLINE_EPOCH:-}" ]]; then
        available_seconds="$(native_work_timeout_seconds protocol-probe)" || return $?
        ((timeout_seconds <= available_seconds)) || timeout_seconds="$available_seconds"
    fi
    ((timeout_seconds > 0)) || return 64
    parent="$RUNS_ROOT/eval/qwen_native_g0"
    mkdir -p "$parent/attempts"
    run_dir="$parent/attempts/$(date -u +'%Y%m%dT%H%M%SZ')-$$-$RANDOM"
    mkdir "$run_dir"
    : >"$run_dir/.running"
    atomic_write "$parent/latest" "$run_dir"$'\n'
    started_epoch="$(date +%s)"
    : >"$run_dir/probe.log"
    set +e
    timeout --signal=TERM --kill-after=120s "${timeout_seconds}s" \
        "$TRAIN_ENV/bin/python" "$PROTOCOL_PROBE_CLI" \
        --model-dir "$base_model" \
        --native-data "$NATIVE_DATA_DIR/probe_g0_8.parquet" \
        --native-manifest "$NATIVE_MANIFEST" \
        --legacy-data "$LEGACY_DATA_DIR/probe_multi_64.parquet" \
        --legacy-manifest "$LEGACY_DATA_DIR/manifest.json" \
        --retriever-url http://127.0.0.1:8000/retrieve \
        --checkpoint-digest "$base_digest" \
        --seed 42 \
        --output-dir "$run_dir/output" \
        >>"$run_dir/probe.log" 2>&1
    rc=$?
    set -e
    if ((rc == 0)); then
        [[ -s "$run_dir/output/records.jsonl" &&
            -s "$run_dir/output/resolved-config.json" &&
            -s "$run_dir/output/manifest.json" ]] || rc=1
    fi
    finish_protocol_probe_run "$run_dir" "$rc" "$started_epoch" "$timeout_seconds"
    LAST_G0_RUN_DIR="$run_dir"
    if ((rc != 0)); then
        printf 'Qwen native G0 failed with exit code %s; inspect %s/probe.log\n' \
            "$rc" "$run_dir" >&2
        return "$rc"
    fi
}

publish_native_gate_evidence() {
    local outer_attempt="$1" results_dir="$2" eval_run="$3" g0_run="${4:-}"
    local marker_dir marker evidence_digest relative path digest checksum_lines=''
    local -a required_outputs=(
        summary.json summary.md go_no_go.json per_trajectory.jsonl
        per_question.jsonl lineage.tsv run-index.tsv stage.txt sampling.json
    )
    local -a evidence_files=() relative_files=() sorted_relative_files=()
    for path in "${required_outputs[@]}"; do
        [[ -s "$results_dir/$path" && ! -L "$results_dir/$path" ]] || {
            printf 'Required Qwen native gate output is missing: %s\n' "$results_dir/$path" >&2
            return 1
        }
        evidence_files+=("$results_dir/$path")
    done
    evidence_files+=(
        "$NATIVE_MANIFEST" "$NATIVE_MANIFEST.sha256" "$NATIVE_CATALOG"
        "$LEGACY_REPLAY_RECEIPT" "$LEGACY_REPLAY_RECEIPT.sha256"
        "$EVAL_DATA_FILE" "$eval_run/train.log" "$eval_run/resolved-config.yaml"
        "$eval_run/run.env" "$eval_run/traces/eval_predictions.jsonl"
        "$eval_run/traces/eval_predictions.manifest.json"
        "$eval_run/traces/eval_predictions.manifest.json.sha256"
    )
    if [[ -n "$g0_run" ]]; then
        evidence_files+=(
            "$g0_run/probe.log" "$g0_run/run.env" "$g0_run/exit-code"
            "$g0_run/terminal" "$g0_run/output/records.jsonl"
            "$g0_run/output/resolved-config.json" "$g0_run/output/manifest.json"
            "$results_dir/protocol_records.jsonl"
        )
    fi
    if [[ -n "$NATIVE_PREDECESSOR_EVIDENCE" ]]; then
        evidence_files+=("$NATIVE_PREDECESSOR_EVIDENCE")
    fi
    for path in "${evidence_files[@]}"; do
        relative="$(native_project_relative_file "$path")" || {
            printf 'Cannot register Qwen native evidence path: %s\n' "$path" >&2
            return 1
        }
        relative_files+=("$relative")
    done
    mapfile -t sorted_relative_files < <(
        printf '%s\n' "${relative_files[@]}" | LC_ALL=C sort -u
    )
    for relative in "${sorted_relative_files[@]}"; do
        path="$PROJECT_ROOT/$relative"
        digest="$(file_sha256 "$path")" || return 1
        checksum_lines+="$digest  $relative"$'\n'
        sync_path "$path"
    done
    atomic_write "$results_dir/evidence.sha256" "$checksum_lines"
    sync_path "$results_dir"
    verify_native_checksum_manifest "$results_dir/evidence.sha256" || return 1
    evidence_digest="$(file_sha256 "$results_dir/evidence.sha256")" || return 1

    marker_dir="$MANIFEST_DIR/qwen-native-gate"
    if [[ ! -e "$MANIFEST_DIR" && ! -L "$MANIFEST_DIR" ]]; then
        mkdir -p "$MANIFEST_DIR"
    fi
    [[ -d "$MANIFEST_DIR" && ! -L "$MANIFEST_DIR" ]] || {
        printf 'Manifest root is missing or symlinked.\n' >&2
        return 1
    }
    if [[ -e "$marker_dir" || -L "$marker_dir" ]]; then
        [[ -d "$marker_dir" && ! -L "$marker_dir" ]] || {
            printf 'Qwen native marker directory exists but is not a real directory.\n' >&2
            return 1
        }
    else
        mkdir "$marker_dir"
    fi
    [[ "$(readlink -f -- "$marker_dir")" == \
        "$(readlink -f -- "$MANIFEST_DIR")/qwen-native-gate" ]] || {
        printf 'Qwen native marker directory escaped the manifest root.\n' >&2
        return 1
    }
    marker="$marker_dir/$(basename -- "$outer_attempt").ok"
    [[ ! -e "$marker" && ! -L "$marker" ]] || {
        printf 'Refusing to overwrite Qwen native gate marker: %s\n' "$marker" >&2
        return 1
    }
    atomic_write "$marker" "$evidence_digest"$'\n'
    sync_path "$marker_dir"
    atomic_write "$outer_attempt/result-contract" 'qwen-native-gate-v1'$'\n'
    atomic_write "$outer_attempt/result-root" "$results_dir"$'\n'
    atomic_write "$outer_attempt/evidence-marker" "$marker"$'\n'
    atomic_write "$outer_attempt/evidence-digest" "$evidence_digest"$'\n'
    sync_path "$outer_attempt"
    atomic_write "$NATIVE_RESULTS_ROOT/latest" "$results_dir"$'\n'
    sync_path "$NATIVE_RESULTS_ROOT"
}

qwen_native_gate_pipeline() {
    local outer_attempt="$1" commit="$2" handoff_digest="$3"
    local base_model="$4" base_digest="$5"
    local variant eval_run g0_run='' results_dir config_digest run_env_digest
    local data_digest input_digest
    local final_model_digest sampling_json
    local trace_identity final_trace_identity trace_digest trace_manifest_digest
    local -a analysis_args
    require_native_gate
    data_digest="$(file_sha256 "$NATIVE_MANIFEST")" || return 1
    input_digest="$(native_gate_input_digest)" || return 1
    [[ "$NATIVE_PREFLIGHT_COMMIT" == "$commit" &&
        "$NATIVE_PREFLIGHT_HANDOFF_DIGEST" == "$handoff_digest" &&
        "$NATIVE_PREFLIGHT_CHECKPOINT_DIGEST" == "$base_digest" &&
        "$NATIVE_PREFLIGHT_DATA_DIGEST" == "$data_digest" &&
        "$NATIVE_PREFLIGHT_INPUT_DIGEST" == "$input_digest" &&
        "$NATIVE_PREFLIGHT_PARENT_EVIDENCE_DIGEST" == \
            "$PARENT_EVIDENCE_DIGEST" ]] || {
        printf 'Qwen native paid-stage inputs drifted after GPU preflight.\n' >&2
        return 1
    }
    verify_native_predecessor "$commit" "$handoff_digest" \
        "$base_digest" "$data_digest" || return 1
    [[ "$PARENT_EVIDENCE_DIGEST" == \
        "$NATIVE_PREFLIGHT_PARENT_EVIDENCE_DIGEST" ]] || {
        printf 'Qwen native predecessor changed after GPU preflight.\n' >&2
        return 1
    }
    case "$NATIVE_STAGE" in
        g0_g1)
            run_protocol_probe "$base_model" "$base_digest"
            g0_run="$LAST_G0_RUN_DIR"
            variant=qwen_native_g1
            ;;
        g2) variant=qwen_native_g2 ;;
        g3) variant=qwen_native_g3 ;;
    esac
    run_job eval "$variant" "$base_model" '' "$base_digest"
    eval_run="$LAST_RUN_DIR"
    sampling_json="$(native_sampling_from_resolved_config \
        "$eval_run/resolved-config.yaml" "$eval_run/run.env")"
    run_env_digest="$(file_sha256 "$eval_run/run.env")" || return 1
    results_dir="$NATIVE_RESULTS_ROOT/attempts/$(basename -- "$outer_attempt")"
    [[ ! -e "$results_dir" && ! -L "$results_dir" ]] || {
        printf 'Refusing to overwrite Qwen native gate results: %s\n' "$results_dir" >&2
        return 1
    }
    analysis_args=(
        --stage "$NATIVE_STAGE"
        --trace "$eval_run/traces/eval_predictions.jsonl"
        --catalog "$NATIVE_CATALOG"
        --data-manifest "$NATIVE_MANIFEST"
        --expected-checkpoint-digest "$base_digest"
        --output-dir "$results_dir"
    )
    if [[ "$NATIVE_STAGE" == g0_g1 ]]; then
        analysis_args+=(--protocol-probe-dir "$g0_run/output")
    fi
    native_deadline_check analysis
    "$TRAIN_ENV/bin/python" "$GATE_ANALYSIS_CLI" "${analysis_args[@]}"
    trace_identity="$(verify_native_trace_evidence \
        "$eval_run/traces/eval_predictions.manifest.json" "$results_dir" \
        "$((EVAL_EXPECTED_ROWS * EVAL_GROUP_SIZE))")"
    IFS=$'\t' read -r trace_digest trace_manifest_digest <<<"$trace_identity"
    atomic_write "$results_dir/stage.txt" "$NATIVE_STAGE"$'\n'
    atomic_write "$results_dir/sampling.json" "$sampling_json"$'\n'
    config_digest="$(file_sha256 "$eval_run/resolved-config.yaml")" || return 1
    atomic_write "$results_dir/lineage.tsv" \
        $'stage\tcheckpoint\tcheckpoint_digest\tevaluation_checkout_commit\tcpu_handoff_digest\tdata_manifest_sha256\tresolved_config_sha256\ttrace_sha256\ttrace_manifest_sha256\tpredecessor_evidence_sha256\n'\
"$NATIVE_STAGE"$'\t'"$base_model"$'\t'"$base_digest"$'\t'"$commit"$'\t'"$handoff_digest"$'\t'"$data_digest"$'\t'"$config_digest"$'\t'"$trace_digest"$'\t'"$trace_manifest_digest"$'\t'"$PARENT_EVIDENCE_DIGEST"$'\n'
    atomic_write "$results_dir/run-index.tsv" \
        $'stage\tmode\trun_dir\ttrace\n'\
"$NATIVE_STAGE"$'\t'"eval"$'\t'"$eval_run"$'\t'"$eval_run/traces/eval_predictions.jsonl"$'\n'
    if [[ -n "$g0_run" ]]; then
        printf 'g0\tprotocol_probe\t%s\t%s\n' "$g0_run" \
            "$g0_run/output/records.jsonl" >>"$results_dir/run-index.tsv"
    fi
    sync_path "$results_dir"
    native_deadline_check evidence-publication
    revalidate_native_gate_seal_inputs "$commit" "$handoff_digest" \
        "$base_model" "$base_digest" "$data_digest" || return 1
    revalidate_native_eval_seal_inputs "$eval_run" "$sampling_json" \
        "$config_digest" "$run_env_digest" "$results_dir/sampling.json" || return 1
    final_trace_identity="$(verify_native_trace_evidence \
        "$eval_run/traces/eval_predictions.manifest.json" "$results_dir" \
        "$((EVAL_EXPECTED_ROWS * EVAL_GROUP_SIZE))")" || return 1
    [[ "$final_trace_identity" == "$trace_identity" ]] || {
        printf 'Qwen native trace identity changed before evidence publication.\n' >&2
        return 1
    }
    final_model_digest="$(tree_sha256 "$base_model")" || return 1
    [[ "$final_model_digest" == "$base_digest" ]] || {
        printf 'Qwen native checkpoint changed before evidence publication.\n' >&2
        return 1
    }
    verify_checkout "$commit" || return 1
    publish_native_gate_evidence "$outer_attempt" "$results_dir" "$eval_run" "$g0_run"
    printf 'Completed Qwen native %s gate: %s/summary.md\n' "$NATIVE_STAGE" "$results_dir"
    printf 'GO or NO-GO is a scientific result; both complete successfully.\n'
    printf 'Guest shutdown is only a dispatch; confirm stopped billing in AutoDL.\n'
}

qwen_native_main() {
    local timeout_seconds
    case "${1:-}" in
        --worker)
            phase_worker gpu "${2:?missing attempt directory}" "$0"
            ;;
        --action)
            require_native_gate
            timeout_seconds="$(native_stage_timeout_seconds)"
            ((timeout_seconds > 0)) || {
                printf 'Computed Qwen native stage timeout is not positive.\n' >&2
                return 64
            }
            export QWEN_NATIVE_STAGE_TIMEOUT_SECONDS="$timeout_seconds"
            export QWEN_NATIVE_DEADLINE_EPOCH="$(( $(date +%s) + timeout_seconds ))"
            run_with_native_deadline "$timeout_seconds" \
                bash "$0" --action-inner "${2:?missing attempt directory}"
            ;;
        --action-inner)
            require_native_gate
            [[ "${QWEN_NATIVE_STAGE_TIMEOUT_SECONDS:-}" =~ ^[1-9][0-9]*$ ]]
            native_deadline_check action-start
            gpu_action "${2:?missing attempt directory}"
            ;;
        '')
            require_native_gate
            if [[ "$NATIVE_STAGE" == g2 || "$NATIVE_STAGE" == g3 ]]; then
                verify_native_predecessor
            fi
            phase_launch gpu "$0"
            ;;
        *)
            printf 'Usage: QWEN_NATIVE_GATE_STAGE={g0_g1|g2|g3} GPU_COUNT=2 AUTODL_PRICE_PER_HOUR=<price> bash %s\n' "$0" >&2
            exit 64
            ;;
    esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    qwen_native_main "$@"
fi
