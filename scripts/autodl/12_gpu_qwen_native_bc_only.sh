#!/usr/bin/env bash
set -Eeuo pipefail

# Post-hoc B/C comparison from one exact, already sealed R60 checkpoint.
BC_PROJECT_ROOT="${AUTODL_ROOT:-/root/autodl-tmp/search-r1}"
BC_SOURCE_ENTRY="$BC_PROJECT_ROOT/checkout/scripts/autodl/09_gpu_qwen_native_train.sh"
BC_RUNNER_PATH="$(readlink -f -- "${BASH_SOURCE[0]}")"
export QWEN_NATIVE_TRAIN_STAGE=main

[[ -f "$BC_SOURCE_ENTRY" && ! -L "$BC_SOURCE_ENTRY" ]] || {
    printf 'Missing sealed Qwen native training entrypoint: %s\n' \
        "$BC_SOURCE_ENTRY" >&2
    exit 1
}

# shellcheck source=09_gpu_qwen_native_train.sh
source "$BC_SOURCE_ENTRY"

readonly BC_ONLY_CONTRACT=qwen-native-training-bc-only-v1
readonly BC_ONLY_NAMESPACE=qwen-native-training-bc-only
readonly BC_CPU_CONTRACT=qwen-native-bc-only-cpu-v1
readonly BC_ONLY_TERMINAL_CODE=202
readonly BC_DEPLOY_PATH="$BC_PROJECT_ROOT/operator/12_gpu_qwen_native_bc_only.sh"
readonly BC_CPU_ROOT="$MANIFEST_DIR/qwen-native-bc-only-cpu"

readonly BC_EXPECTED_CHECKOUT=f8c1cd7e87078d07385f74ca8710add5d5f79c06
readonly BC_EXPECTED_HANDOFF=9ffaa89f88990887b368750ccfdc559d93afdb9a1f5719411cf3e3c1147da90b
readonly BC_EXPECTED_BASE=bc67be20efb353ba14d9c1b291a64410afec94f2310e94b59b6b76047b164e78
readonly BC_EXPECTED_DATA=db6cacf865365ee977fc280d5195556e9903907b28b228ba8f3d72133d674167
readonly BC_EXPECTED_GATE=168bd35d6c72d2f00aa03c6e9490f8cc5a601d005729369982214ea0ab1839a5
readonly BC_EXPECTED_SMOKE=dd6f8c496a1ee415f382a68d64aef6311aec8ca5617e375ce164e2ef5e8c3014

readonly BC_R60_ATTEMPT=20260728T092026Z-2906-16060
readonly BC_R60_RUN_ID=20260728T092332Z-2946-8453
readonly BC_EXPECTED_R60_MARKER="$MANIFEST_DIR/qwen-native-training-r60-only/$BC_R60_ATTEMPT.ok"
readonly BC_EXPECTED_R60_EVIDENCE=2366ce2da28b530f12af30a22d3dc3e2bd33fedbfcb3a2bed5acdd9d0537ffed
readonly BC_R60_RESULTS="$NATIVE_TRAIN_RESULTS_ROOT/attempts/$BC_R60_ATTEMPT"
readonly BC_R60_OUTER="$ATTEMPTS_ROOT/gpu/$BC_R60_ATTEMPT"
readonly BC_R60_RUN="$RUNS_ROOT/reproduce/attempts/$BC_R60_RUN_ID"
readonly BC_R60_CHECKPOINT="$BC_R60_RUN/checkpoints/actor/global_step_60"
readonly BC_EXPECTED_R60_CHECKPOINT=583771b131b6e2aa663ee2ef13ea6421524cdd7fed839f35fd6f9eb246a2c231

readonly BC_G3_OUTER_ID=20260729T073211Z-3516-18146
readonly BC_G3_INNER_ID=20260729T073654Z-3558-15376
readonly BC_G3_OUTER="$ATTEMPTS_ROOT/gpu/$BC_G3_OUTER_ID"
readonly BC_G3_RUN="$RUNS_ROOT/eval/qwen_native_g3/attempts/$BC_G3_INNER_ID"
readonly BC_G3_TRACE="$BC_G3_RUN/traces/eval_predictions.jsonl"
readonly BC_G3_TRACE_MANIFEST="$BC_G3_RUN/traces/eval_predictions.manifest.json"
readonly BC_G3_TRACE_SIDECAR="$BC_G3_TRACE_MANIFEST.sha256"
readonly BC_EXPECTED_G3_CONFIG=34225820ca841a35b2078573f8d92440c1a7fcba7134673cdf4434464f897f9c
readonly BC_EXPECTED_G3_TRACE=0020eb4b2f35fc16ea1115b1f5fde5ad01f4451e3a0a25a59776d9cd55051075
readonly BC_EXPECTED_G3_TRACE_MANIFEST=3304968a28e1fd88f8d44fd73158e16c1d4c56059d138f2a5cd41ff192441fce
readonly BC_EXPECTED_G3_TRACE_SIDECAR=8ea08a56a2514c9c7185c851cc69016d6bb1c440f5a91a377894f204be91aa9d
readonly BC_G3_ANALYSIS="$NATIVE_TRAIN_RESULTS_ROOT/analysis-recoveries/$BC_G3_OUTER_ID/registered-probe"
readonly BC_EXPECTED_G3_SUMMARY=a70ef012d07d3bd7c1da6235d8fd7bc85bd409db94eabfc59c006dbfba7dc376
readonly BC_EXPECTED_G3_SUMMARY_MD=ce5c060306cc0b94911cd600ec9b7c944c401f9092b63cf906517e4429401671
readonly BC_EXPECTED_G3_DECISION=b4ee78d70804ec993a3b734f39c277177967535ed69ab115ea763029b7e79c9d
readonly BC_EXPECTED_G3_PER_QUESTION=c12e9472990d632d99cb5ccb90538ad49deccc2b5eb39cdfb8a7a7365095fdba
readonly BC_EXPECTED_G3_PER_TRAJECTORY=4ea2cee2523ce0c16dd4b7e35e59ea61b983ecb93666a8ae1d3f35b797620229

readonly BC_CONTROL_CONFIG="$MANIFEST_DIR/config-2gpu-qwen-native-control-train.yaml"
readonly BC_COST_CONFIG="$MANIFEST_DIR/config-2gpu-qwen-native-cost_aware_gated-train.yaml"
readonly BC_EXPECTED_CONTROL_CONFIG=5079fcde2dc7fab35b91f6387ae6cd6be347b9007845b68ced41094d893522cf
readonly BC_EXPECTED_COST_CONFIG=44fcb75011fb1a8a4b7e59c24bf3633fe8d62b3a2cdad974321fdc216e7a92b3

BC_CPU_RECEIPT=''
BC_CPU_RECEIPT_DIGEST=''
BC_PREFLIGHT_COMMIT=''
BC_PREFLIGHT_HANDOFF=''
BC_PREFLIGHT_BASE_DIGEST=''
BC_PREFLIGHT_DATA_DIGEST=''
BC_PREFLIGHT_RUNNER_DIGEST=''

bc_value() {
    local path="$1"
    [[ -f "$path" && ! -L "$path" ]] || return 1
    tr -d '\r\n' <"$path"
}

bc_require_digest() {
    local path="$1" expected="$2"
    [[ -f "$path" && ! -L "$path" &&
        "$(file_sha256 "$path")" == "$expected" ]] || {
        printf 'Exact B/C predecessor file changed: %s\n' "$path" >&2
        return 1
    }
}

verify_bc_r60() {
    local marker="$1" evidence="$BC_R60_RESULTS/evidence.sha256"
    [[ "$marker" == "$BC_EXPECTED_R60_MARKER" &&
        -f "$marker" && ! -L "$marker" &&
        "$(bc_value "$marker")" == "$BC_EXPECTED_R60_EVIDENCE" ]] || {
        printf 'B/C requires the exact sealed R60 marker: %s\n' \
            "$BC_EXPECTED_R60_MARKER" >&2
        return 1
    }
    [[ -d "$BC_R60_RESULTS" && ! -L "$BC_R60_RESULTS" &&
        -d "$BC_R60_OUTER" && ! -L "$BC_R60_OUTER" &&
        -f "$BC_R60_OUTER/.failed" && ! -L "$BC_R60_OUTER/.failed" &&
        ! -e "$BC_R60_OUTER/.success" &&
        "$(bc_value "$BC_R60_OUTER/terminal")" == failed &&
        "$(bc_value "$BC_R60_OUTER/exit-code")" == 200 &&
        "$(bc_value "$BC_R60_OUTER/result-contract")" == qwen-native-training-r60-only-v1 &&
        "$(bc_value "$BC_R60_OUTER/result-root")" == "$BC_R60_RESULTS" &&
        "$(bc_value "$BC_R60_OUTER/evidence-marker")" == "$marker" &&
        "$(bc_value "$BC_R60_OUTER/evidence-digest")" == "$BC_EXPECTED_R60_EVIDENCE" &&
        "$(file_sha256 "$evidence")" == "$BC_EXPECTED_R60_EVIDENCE" ]] || {
        printf 'Exact R60 outer/evidence binding is incomplete.\n' >&2
        return 1
    }
    verify_native_training_checksum_manifest "$evidence" || return $?
    grep -Fxq 'schema=qwen-native-training-r60-only-v1' \
        "$BC_R60_RESULTS/contract.env" || return 1
    grep -Fxq 'stage_order=R60' "$BC_R60_RESULTS/contract.env" || return 1
    grep -Fq "$BC_R60_CHECKPOINT"$'\t'"$BC_EXPECTED_R60_CHECKPOINT" \
        "$BC_R60_RESULTS/lineage.tsv" || return 1
    verify_native_checkpoint_digest "$BC_R60_CHECKPOINT" \
        "$BC_EXPECTED_R60_CHECKPOINT"
}

verify_bc_g3() {
    local trace_identity trace_digest manifest_digest
    [[ -d "$BC_G3_OUTER" && ! -L "$BC_G3_OUTER" &&
        -f "$BC_G3_OUTER/.failed" && ! -L "$BC_G3_OUTER/.failed" &&
        ! -e "$BC_G3_OUTER/.success" &&
        "$(bc_value "$BC_G3_OUTER/terminal")" == failed &&
        "$(bc_value "$BC_G3_OUTER/exit-code")" == 1 &&
        -d "$BC_G3_RUN" && ! -L "$BC_G3_RUN" &&
        -f "$BC_G3_RUN/.success" && ! -L "$BC_G3_RUN/.success" &&
        "$(bc_value "$BC_G3_RUN/terminal")" == success &&
        "$(bc_value "$BC_G3_RUN/exit-code")" == 0 ]] || {
        printf 'Exact historical G3 outer/inner terminal state changed.\n' >&2
        return 1
    }
    bc_require_digest "$BC_G3_RUN/resolved-config.yaml" \
        "$BC_EXPECTED_G3_CONFIG" || return $?
    bc_require_digest "$BC_G3_TRACE" "$BC_EXPECTED_G3_TRACE" || return $?
    bc_require_digest "$BC_G3_TRACE_MANIFEST" \
        "$BC_EXPECTED_G3_TRACE_MANIFEST" || return $?
    bc_require_digest "$BC_G3_TRACE_SIDECAR" \
        "$BC_EXPECTED_G3_TRACE_SIDECAR" || return $?
    trace_identity="$(verify_native_trace_identity "$BC_G3_TRACE_MANIFEST" 320)" ||
        return 1
    IFS=$'\t' read -r trace_digest manifest_digest <<<"$trace_identity"
    [[ "$trace_digest" == "$BC_EXPECTED_G3_TRACE" &&
        "$manifest_digest" == "$BC_EXPECTED_G3_TRACE_MANIFEST" ]] || return 1
    bc_require_digest "$BC_G3_ANALYSIS/summary.json" \
        "$BC_EXPECTED_G3_SUMMARY" || return $?
    bc_require_digest "$BC_G3_ANALYSIS/summary.md" \
        "$BC_EXPECTED_G3_SUMMARY_MD" || return $?
    bc_require_digest "$BC_G3_ANALYSIS/go_no_go.json" \
        "$BC_EXPECTED_G3_DECISION" || return $?
    bc_require_digest "$BC_G3_ANALYSIS/per_question.jsonl" \
        "$BC_EXPECTED_G3_PER_QUESTION" || return $?
    bc_require_digest "$BC_G3_ANALYSIS/per_trajectory.jsonl" \
        "$BC_EXPECTED_G3_PER_TRAJECTORY" || return $?
    grep -Eq '"decision"[[:space:]]*:[[:space:]]*"NO-GO"' \
        "$BC_G3_ANALYSIS/go_no_go.json" || return 1
    grep -Eq '"cost_contrast_group_count"[[:space:]]*:[[:space:]]*13' \
        "$BC_G3_ANALYSIS/summary.json" || return 1
}

verify_bc_foundation() {
    local commit="$1" handoff_digest="$2" base_model="$3" base_digest="$4"
    local data_digest
    export PYTHONDONTWRITEBYTECODE=1
    export PYTHONPATH="$CHECKOUT_DIR${PYTHONPATH:+:$PYTHONPATH}"
    [[ "$commit" == "$BC_EXPECTED_CHECKOUT" &&
        "$handoff_digest" == "$BC_EXPECTED_HANDOFF" &&
        "$base_digest" == "$BC_EXPECTED_BASE" &&
        "$BC_RUNNER_PATH" == "$BC_DEPLOY_PATH" ]] || {
        printf 'Frozen checkout, handoff, base model, or operator path changed.\n' >&2
        return 1
    }
    [[ "${QWEN_NATIVE_PROTOCOL_GATE_EVIDENCE:-}" == \
            "$MANIFEST_DIR/qwen-native-gate/20260728T044634Z-2051-4045.ok" &&
        "${QWEN_NATIVE_SMOKE_EVIDENCE:-}" == \
            "$MANIFEST_DIR/qwen-native-training-smoke/20260728T061246Z-1316-15067.ok" &&
        "${QWEN_NATIVE_R60_EVIDENCE:-}" == "$BC_EXPECTED_R60_MARKER" ]] || {
        printf 'Pass the three exact predecessor markers.\n' >&2
        return 64
    }
    [[ ! -e "$MANIFEST_DIR/cpu-seal.pending.json" ]] || return 1
    verify_checkout "$commit" || return $?
    [[ "$(bc_value "$MANIFEST_DIR/cpu.ok")" == "$handoff_digest" &&
        "$(file_sha256 "$HANDOFF")" == "$handoff_digest" ]] || return 1
    verify_qwen_native_train_data || return $?
    data_digest="$(file_sha256 "$NATIVE_TRAIN_MANIFEST")" || return 1
    [[ "$data_digest" == "$BC_EXPECTED_DATA" ]] || return 1
    bc_require_digest "$BC_CONTROL_CONFIG" "$BC_EXPECTED_CONTROL_CONFIG" ||
        return $?
    bc_require_digest "$BC_COST_CONFIG" "$BC_EXPECTED_COST_CONFIG" || return $?
    verify_protocol_gate_evidence "$QWEN_NATIVE_PROTOCOL_GATE_EVIDENCE" \
        "$base_model" "$base_digest" "$commit" "$handoff_digest" \
        "$data_digest" || return $?
    [[ "$NATIVE_TRAIN_PROTOCOL_GATE_DIGEST" == "$BC_EXPECTED_GATE" ]] || return 1
    verify_smoke_evidence "$QWEN_NATIVE_SMOKE_EVIDENCE" "$base_model" \
        "$base_digest" "$commit" "$handoff_digest" "$data_digest" \
        "$BC_EXPECTED_GATE" || return $?
    [[ "$NATIVE_TRAIN_SMOKE_DIGEST" == "$BC_EXPECTED_SMOKE" ]] || return 1
    verify_bc_r60 "$QWEN_NATIVE_R60_EVIDENCE" || return $?
    verify_bc_g3 || return $?
}

bc_receipt_content() {
    local runner_digest="$1"
    printf '%s\n' \
        "schema=$BC_CPU_CONTRACT" \
        "experiment_class=post_hoc_exploratory" \
        "operator_override=true" \
        "checkout_commit=$BC_EXPECTED_CHECKOUT" \
        "cpu_handoff_sha256=$BC_EXPECTED_HANDOFF" \
        "base_model_sha256=$BC_EXPECTED_BASE" \
        "data_manifest_sha256=$BC_EXPECTED_DATA" \
        "protocol_gate_sha256=$BC_EXPECTED_GATE" \
        "smoke_sha256=$BC_EXPECTED_SMOKE" \
        "r60_evidence_sha256=$BC_EXPECTED_R60_EVIDENCE" \
        "r60_checkpoint=$BC_R60_CHECKPOINT" \
        "r60_checkpoint_sha256=$BC_EXPECTED_R60_CHECKPOINT" \
        "original_g3_decision=NO-GO" \
        "original_g3_cost_contrast_group_count=13" \
        "g3_trace_sha256=$BC_EXPECTED_G3_TRACE" \
        "g3_analysis_sha256=$BC_EXPECTED_G3_SUMMARY" \
        "runner=$BC_DEPLOY_PATH" \
        "runner_sha256=$runner_digest" \
        "gpu_training_authorized=false" \
        "manual_gpu_start_required=true"
}

verify_bc_cpu_receipt() {
    local runner_digest="$1" directory receipt expected digest sidecar
    directory="$BC_CPU_ROOT/$runner_digest"
    receipt="$directory/receipt.env"
    expected="$(bc_receipt_content "$runner_digest")"$'\n'
    [[ -d "$directory" && ! -L "$directory" &&
        -f "$receipt" && ! -L "$receipt" &&
        -f "$directory/.success" && ! -L "$directory/.success" &&
        ! -e "$directory/.failed" &&
        "$(bc_value "$directory/terminal")" == success &&
        "$(bc_value "$directory/exit-code")" == 0 ]] || return 1
    cmp -s <(printf '%s' "$expected") "$receipt" || return 1
    digest="$(file_sha256 "$receipt")" || return 1
    sidecar="$(bc_value "$receipt.sha256")" || return 1
    [[ "$sidecar" == "$digest  ${receipt#"$BC_PROJECT_ROOT/"}" ]] || return 1
    BC_CPU_RECEIPT="$receipt"
    BC_CPU_RECEIPT_DIGEST="$digest"
}

verify_bc_publish_identity() {
    local commit="$1" data_digest="$2" runner_digest="$3"
    verify_checkout "$commit" || return $?
    [[ "$(file_sha256 "$NATIVE_TRAIN_MANIFEST")" == "$data_digest" ]] || {
        printf 'B/C data manifest drifted before evidence publication.\n' >&2
        return 1
    }
    [[ "$(file_sha256 "$BC_RUNNER_PATH")" == "$runner_digest" ]] || {
        printf 'B/C operator runner drifted before evidence publication.\n' >&2
        return 1
    }
    verify_bc_cpu_receipt "$runner_digest" || {
        printf 'B/C CPU receipt drifted before evidence publication.\n' >&2
        return 1
    }
}

prepare_bc_cpu_receipt() {
    local commit handoff_digest base_model base_digest runner_digest
    local directory receipt content digest sidecar
    [[ "$BC_RUNNER_PATH" == "$BC_DEPLOY_PATH" &&
        -f "$BC_RUNNER_PATH" && ! -L "$BC_RUNNER_PATH" ]] || {
        printf 'Deploy this exact runner first: %s\n' "$BC_DEPLOY_PATH" >&2
        return 1
    }
    export CUDA_VISIBLE_DEVICES=''
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
    export WANDB_MODE=offline PIP_NO_INDEX=1 PYTHONDONTWRITEBYTECODE=1
    commit="$(expected_commit)" || return 1
    handoff_digest="$(bc_value "$MANIFEST_DIR/cpu.ok")" || return 1
    base_model="$(readlink -f -- "$MODEL_DIR")" || return 1
    base_digest="$(tree_sha256 "$base_model")" || return 1
    verify_bc_foundation "$commit" "$handoff_digest" "$base_model" \
        "$base_digest" || return $?
    runner_digest="$(file_sha256 "$BC_RUNNER_PATH")" || return 1
    directory="$BC_CPU_ROOT/$runner_digest"
    receipt="$directory/receipt.env"
    content="$(bc_receipt_content "$runner_digest")"$'\n'
    mkdir -p "$BC_CPU_ROOT"
    if [[ -e "$directory" || -L "$directory" ]]; then
        verify_bc_cpu_receipt "$runner_digest" || {
            printf 'Refusing a differing or incomplete B/C CPU receipt.\n' >&2
            return 1
        }
    else
        mkdir "$directory"
        atomic_write "$receipt" "$content"
        digest="$(file_sha256 "$receipt")" || return 1
        sidecar="$digest  ${receipt#"$BC_PROJECT_ROOT/"}"$'\n'
        atomic_write "$receipt.sha256" "$sidecar"
        atomic_write "$directory/exit-code" $'0\n'
        atomic_write "$directory/terminal" $'success\n'
        atomic_write "$directory/.success" ''
        sync_path "$directory"
    fi
    verify_bc_cpu_receipt "$runner_digest" || return $?
    printf 'B/C CPU preparation complete: %s\n' "$BC_CPU_RECEIPT"
    printf 'GPU training was not started.\n'
}

# Override main-v5 admission: this post-hoc slice consumes exact R60 evidence.
require_qwen_native_train() {
    validate_gpu_inputs || return $?
    [[ "$GPU_COUNT" == 2 && "$TRAIN_BATCH_SIZE" == 8 &&
        "$MAX_RESPONSE_LENGTH" == 500 && "$TOOL_PROTOCOL" == qwen35_native &&
        "$DATA_DIR" == "$NATIVE_TRAIN_DATA_DIR" &&
        "$EVAL_GROUP_SIZE" == 1 && -z "$EVAL_DATA_FILE" &&
        "$RUN_BUDGET_PROFILE" == gated_followup &&
        "$BC_RUNNER_PATH" == "$BC_DEPLOY_PATH" &&
        "${QWEN_NATIVE_PROTOCOL_GATE_EVIDENCE:-}" == \
            "$MANIFEST_DIR/qwen-native-gate/20260728T044634Z-2051-4045.ok" &&
        "${QWEN_NATIVE_SMOKE_EVIDENCE:-}" == \
            "$MANIFEST_DIR/qwen-native-training-smoke/20260728T061246Z-1316-15067.ok" &&
        "${QWEN_NATIVE_R60_EVIDENCE:-}" == "$BC_EXPECTED_R60_MARKER" ]] || {
        printf 'B/C-only requires exact markers, two GPUs, batch 8, response 500, and native-v4 settings.\n' >&2
        return 64
    }
}

qwen_native_train_preflight() {
    local commit="$1" handoff_digest="$2" base_digest="$3"
    local base_model runner_digest data_digest
    base_model="$(readlink -f -- "$MODEL_DIR")" || return 1
    verify_bc_foundation "$commit" "$handoff_digest" "$base_model" \
        "$base_digest" || return $?
    runner_digest="$(file_sha256 "$BC_RUNNER_PATH")" || return 1
    verify_bc_cpu_receipt "$runner_digest" || {
        printf 'Run --cpu-prepare with this exact operator runner first.\n' >&2
        return 1
    }
    data_digest="$(file_sha256 "$NATIVE_TRAIN_MANIFEST")" || return 1
    BC_PREFLIGHT_COMMIT="$commit"
    BC_PREFLIGHT_HANDOFF="$handoff_digest"
    BC_PREFLIGHT_BASE_DIGEST="$base_digest"
    BC_PREFLIGHT_DATA_DIGEST="$data_digest"
    BC_PREFLIGHT_RUNNER_DIGEST="$runner_digest"

    # qwen_native_train_pipeline() is inherited from 09 and revalidates this
    # exact state before dispatching qwen_native_main_pipeline(). Keep both the
    # B/C-specific identity and the inherited native-v4 identity in lockstep.
    NATIVE_TRAIN_PREFLIGHT_STAGE="$NATIVE_TRAIN_STAGE"
    NATIVE_TRAIN_PREFLIGHT_COMMIT="$commit"
    NATIVE_TRAIN_PREFLIGHT_HANDOFF="$handoff_digest"
    NATIVE_TRAIN_PREFLIGHT_BASE_DIGEST="$base_digest"
    NATIVE_TRAIN_PREFLIGHT_DATA_DIGEST="$data_digest"
    NATIVE_TRAIN_PREFLIGHT_PROTOCOL_GATE_DIGEST="$NATIVE_TRAIN_PROTOCOL_GATE_DIGEST"
    NATIVE_TRAIN_PREFLIGHT_SMOKE_DIGEST="$NATIVE_TRAIN_SMOKE_DIGEST"
}

verify_bc_results_semantics() {
    local results="$1" control_checkpoint="$2" control_digest="$3"
    local cost_checkpoint="$4" cost_digest="$5"
    "$TRAIN_ENV/bin/python" - "$results" "$BC_ONLY_CONTRACT" \
        "$BC_R60_CHECKPOINT" "$BC_EXPECTED_R60_CHECKPOINT" \
        "$control_checkpoint" "$control_digest" "$cost_checkpoint" "$cost_digest" <<'PY'
import csv
import json
from pathlib import Path
import sys

(root_raw, contract, r60_raw, r60_digest, control_raw, control_digest,
 cost_raw, cost_digest) = sys.argv[1:]
root = Path(root_raw)
r60 = Path(r60_raw)
control = Path(control_raw)
cost = Path(cost_raw)

def env(path):
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split("=", 1)
        if key in result:
            raise SystemExit(f"duplicate env key: {key}")
        result[key] = value
    return result

contract_env = env(root / "contract.env")
expected_contract = {
    "schema": contract,
    "stage": "bc_only",
    "stage_order": (
        "B20,C20,B-VAL-EVAL,C-VAL-EVAL,B-NQ-TEST-EVAL,"
        "C-NQ-TEST-EVAL,B-MULTIHOP-EVAL,C-MULTIHOP-EVAL"
    ),
    "experiment_class": "post_hoc_exploratory",
    "operator_override": "true",
    "preregistered_branch_authorized": "false",
    "original_g3_decision": "NO-GO",
    "original_g3_cost_contrast_group_count": "13",
    "controlled_outer_exit_code": "202",
}
for key, value in expected_contract.items():
    if contract_env.get(key) != value:
        raise SystemExit(f"B/C contract mismatch: {key}")

with (root / "lineage.tsv").open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
expected = [
    ("B", "control"), ("C", "cost_aware_gated"),
    ("B-VAL-EVAL", "control_val_eval"),
    ("C-VAL-EVAL", "cost_aware_gated_val_eval"),
    ("B-NQ-TEST-EVAL", "control_nq_test_eval"),
    ("C-NQ-TEST-EVAL", "cost_aware_gated_nq_test_eval"),
    ("B-MULTIHOP-EVAL", "control_multihop_eval"),
    ("C-MULTIHOP-EVAL", "cost_aware_gated_multihop_eval"),
]
if [(row["stage"], row["role"]) for row in rows] != expected:
    raise SystemExit("B/C lineage stage order mismatch")
for row in rows[:2]:
    if row["parent_checkpoint"] != str(r60) or \
            row["parent_checkpoint_digest"] != r60_digest:
        raise SystemExit("B and C must independently inherit exact R60")
if rows[0]["checkpoint"] != str(control) or rows[0]["checkpoint_digest"] != control_digest:
    raise SystemExit("B lineage checkpoint mismatch")
if rows[1]["checkpoint"] != str(cost) or rows[1]["checkpoint_digest"] != cost_digest:
    raise SystemExit("C lineage checkpoint mismatch")
for row in rows[2:]:
    is_control = row["stage"].startswith("B-")
    checkpoint = control if is_control else cost
    digest = control_digest if is_control else cost_digest
    if row["checkpoint"] != str(checkpoint) or row["checkpoint_digest"] != digest:
        raise SystemExit("endpoint lineage checkpoint mismatch")

for row, variant, role, lam, mode in (
    (rows[0], "control", "control", 0.0, "linear"),
    (rows[1], "cost_aware_gated", "cost_aware_gated", 0.1, "correct_only"),
):
    payload = json.loads(
        (Path(row["run_dir"]) / "native-training-contract.json").read_bytes()
    )
    if payload.get("variant") != variant or payload.get("role") != role or \
            payload.get("steps") != 20 or payload.get("parent") != str(r60) or \
            payload.get("parent_digest") != r60_digest or \
            float(payload.get("cost_lambda")) != lam or \
            payload.get("cost_reward_mode") != mode:
        raise SystemExit(f"invalid {role} run contract")

with (root / "run-index.tsv").open(newline="", encoding="utf-8") as handle:
    index = list(csv.DictReader(handle, delimiter="\t"))
if len(index) != len(rows):
    raise SystemExit("B/C run index length mismatch")
for row, item in zip(rows, index):
    if (item["stage"], item["role"], item["run_dir"]) != (
            row["stage"], row["role"], row["run_dir"]):
        raise SystemExit("B/C run index mismatch")

for name in ("val", "nq_test", "multihop"):
    paired = root / f"paired-{name}"
    for filename in (
        "summary.json", "summary.md", "paired_results.csv",
        "correct_questions.csv", "wrong_questions.csv", "search_transition.csv",
    ):
        path = paired / filename
        if not path.is_file() or path.is_symlink():
            raise SystemExit(f"missing paired evidence: {path}")
PY
}

qwen_native_main_pipeline() {
    local outer_attempt="$1" commit="$2" handoff_digest="$3"
    local _base_model="$4" base_digest="$5" data_digest="$6"
    local control_run cost_run control_checkpoint cost_checkpoint
    local control_digest cost_digest control_trace cost_trace
    local control_trace_digest control_trace_manifest_digest
    local cost_trace_digest cost_trace_manifest_digest
    local results_dir runner_digest index_content branch_rows=''
    local eval_key eval_rows eval_stage eval_artifact trace_identity
    local eval_trace_digest eval_trace_manifest_digest run config_digest
    local control_config cost_config control_contract cost_contract
    local wandb_run wandb_file wandb_steps wandb_count
    local marker evidence_digest
    local -a eval_keys=(val nq_test multihop)
    local -A eval_expected_rows=([val]=128 [nq_test]=128 [multihop]=256)
    local -A eval_artifacts=([val]=val [nq_test]=nq_test_eval [multihop]=multihop_eval)
    local -A control_eval_runs=() cost_eval_runs=()
    local -A control_eval_trace_digests=() control_eval_trace_manifest_digests=()
    local -A cost_eval_trace_digests=() cost_eval_trace_manifest_digests=()
    local -A paired_dirs=()
    local -a evidence_files=() wandb_runs=()

    runner_digest="$(file_sha256 "$BC_RUNNER_PATH")" || return 1
    [[ "$BC_PREFLIGHT_COMMIT" == "$commit" &&
        "$BC_PREFLIGHT_HANDOFF" == "$handoff_digest" &&
        "$BC_PREFLIGHT_BASE_DIGEST" == "$base_digest" &&
        "$BC_PREFLIGHT_DATA_DIGEST" == "$data_digest" &&
        "$BC_PREFLIGHT_RUNNER_DIGEST" == "$runner_digest" ]] || {
        printf 'B/C inputs drifted after GPU preflight.\n' >&2
        return 1
    }
    verify_bc_cpu_receipt "$runner_digest" || return $?
    verify_bc_publish_identity "$commit" "$data_digest" "$runner_digest" ||
        return $?
    verify_native_checkpoint_digest "$BC_R60_CHECKPOINT" \
        "$BC_EXPECTED_R60_CHECKPOINT" || return $?
    results_dir="$(create_native_training_results_dir "$outer_attempt")" || return 1

    export EVAL_EXPECTED_ROWS=128 EVAL_GROUP_SIZE=1 EVAL_DATA_FILE=''
    run_job train control "$BRANCH_STEPS" "$BC_R60_CHECKPOINT" \
        "$BC_EXPECTED_R60_CHECKPOINT" || return $?
    control_run="$LAST_RUN_DIR"
    append_native_train_index "$outer_attempt" B control "$control_run" || return $?
    control_checkpoint="$(fixed_checkpoint "$control_run" "$BRANCH_STEPS")" || return 1
    control_digest="$(tree_sha256 "$control_checkpoint")" || return 1
    record_lineage "$control_run" control "$control_checkpoint" "$control_digest" \
        "$BC_R60_CHECKPOINT" "$BC_EXPECTED_R60_CHECKPOINT" "$commit" \
        "$handoff_digest" 0 linear || return $?
    seal_native_training_run "$control_run" control "$control_checkpoint" \
        "$control_digest" "$BC_R60_CHECKPOINT" "$BC_EXPECTED_R60_CHECKPOINT" \
        control "$BRANCH_STEPS" 0 linear "$commit" "$handoff_digest" || return $?
    control_trace="$(verify_native_trace_identity \
        "$control_run/traces/train_trajectories.manifest.json" \
        "$((BRANCH_STEPS * TRAIN_BATCH_SIZE * 5))")" || return 1
    IFS=$'\t' read -r control_trace_digest control_trace_manifest_digest \
        <<<"$control_trace"

    # C starts from the same R60 weights, never from B.
    verify_native_checkpoint_digest "$BC_R60_CHECKPOINT" \
        "$BC_EXPECTED_R60_CHECKPOINT" || return $?
    run_job train cost_aware_gated "$BRANCH_STEPS" "$BC_R60_CHECKPOINT" \
        "$BC_EXPECTED_R60_CHECKPOINT" || return $?
    cost_run="$LAST_RUN_DIR"
    append_native_train_index "$outer_attempt" C cost_aware_gated "$cost_run" ||
        return $?
    cost_checkpoint="$(fixed_checkpoint "$cost_run" "$BRANCH_STEPS")" || return 1
    cost_digest="$(tree_sha256 "$cost_checkpoint")" || return 1
    record_lineage "$cost_run" cost_aware_gated "$cost_checkpoint" "$cost_digest" \
        "$BC_R60_CHECKPOINT" "$BC_EXPECTED_R60_CHECKPOINT" "$commit" \
        "$handoff_digest" 0.10 correct_only || return $?
    seal_native_training_run "$cost_run" cost_aware_gated "$cost_checkpoint" \
        "$cost_digest" "$BC_R60_CHECKPOINT" "$BC_EXPECTED_R60_CHECKPOINT" \
        cost_aware_gated "$BRANCH_STEPS" 0.10 correct_only "$commit" \
        "$handoff_digest" || return $?
    cost_trace="$(verify_native_trace_identity \
        "$cost_run/traces/train_trajectories.manifest.json" \
        "$((BRANCH_STEPS * TRAIN_BATCH_SIZE * 5))")" || return 1
    IFS=$'\t' read -r cost_trace_digest cost_trace_manifest_digest \
        <<<"$cost_trace"

    for eval_key in "${eval_keys[@]}"; do
        eval_rows="${eval_expected_rows[$eval_key]}"
        eval_artifact="${eval_artifacts[$eval_key]}"
        export EVAL_EXPECTED_ROWS="$eval_rows"
        run_job eval "qwen_native_b_$eval_key" "$control_checkpoint" '' \
            "$control_digest" || return $?
        run="$LAST_RUN_DIR"
        control_eval_runs[$eval_key]="$run"
        eval_stage="${eval_key^^}"
        eval_stage="B-${eval_stage//_/-}-EVAL"
        append_native_train_index "$outer_attempt" "$eval_stage" \
            "control_${eval_key}_eval" "$run" || return $?
        trace_identity="$(verify_native_trace_identity \
            "$run/traces/eval_predictions.manifest.json" "$eval_rows")" || return 1
        IFS=$'\t' read -r eval_trace_digest eval_trace_manifest_digest \
            <<<"$trace_identity"
        control_eval_trace_digests[$eval_key]="$eval_trace_digest"
        control_eval_trace_manifest_digests[$eval_key]="$eval_trace_manifest_digest"

        run_job eval "qwen_native_c_$eval_key" "$cost_checkpoint" '' \
            "$cost_digest" || return $?
        run="$LAST_RUN_DIR"
        cost_eval_runs[$eval_key]="$run"
        eval_stage="${eval_key^^}"
        eval_stage="C-${eval_stage//_/-}-EVAL"
        append_native_train_index "$outer_attempt" "$eval_stage" \
            "cost_aware_gated_${eval_key}_eval" "$run" || return $?
        trace_identity="$(verify_native_trace_identity \
            "$run/traces/eval_predictions.manifest.json" "$eval_rows")" || return 1
        IFS=$'\t' read -r eval_trace_digest eval_trace_manifest_digest \
            <<<"$trace_identity"
        cost_eval_trace_digests[$eval_key]="$eval_trace_digest"
        cost_eval_trace_manifest_digests[$eval_key]="$eval_trace_manifest_digest"

        paired_dirs[$eval_key]="$results_dir/paired-$eval_key"
        "$TRAIN_ENV/bin/python" "$NATIVE_TRAIN_PAIRED_CLI" \
            --control "${control_eval_runs[$eval_key]}/traces/eval_predictions.jsonl" \
            --cost-aware-gated "${cost_eval_runs[$eval_key]}/traces/eval_predictions.jsonl" \
            --data-manifest "$NATIVE_TRAIN_MANIFEST" \
            --eval-artifact "$eval_artifact" \
            --expected-control-checkpoint-digest "$control_digest" \
            --expected-cost-aware-gated-checkpoint-digest "$cost_digest" \
            --output-dir "${paired_dirs[$eval_key]}" \
            --expected-rows "$eval_rows" || return $?
    done

    control_config="$(file_sha256 "$control_run/resolved-config.yaml")" || return 1
    cost_config="$(file_sha256 "$cost_run/resolved-config.yaml")" || return 1
    control_contract="$(file_sha256 "$control_run/native-training-contract.json")" ||
        return 1
    cost_contract="$(file_sha256 "$cost_run/native-training-contract.json")" ||
        return 1
    atomic_write "$results_dir/contract.env" \
"schema=$BC_ONLY_CONTRACT
stage=bc_only
stage_order=B20,C20,B-VAL-EVAL,C-VAL-EVAL,B-NQ-TEST-EVAL,C-NQ-TEST-EVAL,B-MULTIHOP-EVAL,C-MULTIHOP-EVAL
experiment_class=post_hoc_exploratory
operator_override=true
preregistered_branch_authorized=false
original_g3_decision=NO-GO
original_g3_cost_contrast_group_count=13
controlled_outer_exit_code=$BC_ONLY_TERMINAL_CODE
protocol_gate_evidence=$QWEN_NATIVE_PROTOCOL_GATE_EVIDENCE
protocol_gate_evidence_sha256=$BC_EXPECTED_GATE
smoke_evidence=$QWEN_NATIVE_SMOKE_EVIDENCE
smoke_evidence_sha256=$BC_EXPECTED_SMOKE
r60_evidence=$QWEN_NATIVE_R60_EVIDENCE
r60_evidence_sha256=$BC_EXPECTED_R60_EVIDENCE
r60_checkpoint=$BC_R60_CHECKPOINT
r60_checkpoint_sha256=$BC_EXPECTED_R60_CHECKPOINT
g3_outer_attempt=$BC_G3_OUTER_ID
g3_inner_run=$BC_G3_INNER_ID
g3_trace_sha256=$BC_EXPECTED_G3_TRACE
cpu_receipt=$BC_CPU_RECEIPT
cpu_receipt_sha256=$BC_CPU_RECEIPT_DIGEST
runner=$BC_DEPLOY_PATH
runner_sha256=$runner_digest
"
    atomic_write "$results_dir/lineage.tsv" \
        $'stage\trole\trun_dir\tcheckpoint\tcheckpoint_digest\tparent_checkpoint\tparent_checkpoint_digest\tcheckout_commit\tcpu_handoff_digest\tdata_manifest_sha256\tresolved_config_sha256\ttrace_sha256\ttrace_manifest_sha256\trun_contract_sha256\tpredecessor_evidence_sha256\n'
    branch_rows="B"$'\t'"control"$'\t'"$control_run"$'\t'"$control_checkpoint"$'\t'"$control_digest"$'\t'"$BC_R60_CHECKPOINT"$'\t'"$BC_EXPECTED_R60_CHECKPOINT"$'\t'"$commit"$'\t'"$handoff_digest"$'\t'"$data_digest"$'\t'"$control_config"$'\t'"$control_trace_digest"$'\t'"$control_trace_manifest_digest"$'\t'"$control_contract"$'\t'"$BC_EXPECTED_R60_EVIDENCE"$'\n'
    branch_rows+="C"$'\t'"cost_aware_gated"$'\t'"$cost_run"$'\t'"$cost_checkpoint"$'\t'"$cost_digest"$'\t'"$BC_R60_CHECKPOINT"$'\t'"$BC_EXPECTED_R60_CHECKPOINT"$'\t'"$commit"$'\t'"$handoff_digest"$'\t'"$data_digest"$'\t'"$cost_config"$'\t'"$cost_trace_digest"$'\t'"$cost_trace_manifest_digest"$'\t'"$cost_contract"$'\t'"$BC_EXPECTED_R60_EVIDENCE"$'\n'
    for eval_key in "${eval_keys[@]}"; do
        eval_stage="${eval_key^^}"
        eval_stage="${eval_stage//_/-}"
        run="${control_eval_runs[$eval_key]}"
        config_digest="$(file_sha256 "$run/resolved-config.yaml")" || return 1
        branch_rows+="B-$eval_stage-EVAL"$'\t'"control_${eval_key}_eval"$'\t'"$run"$'\t'"$control_checkpoint"$'\t'"$control_digest"$'\t'"$control_checkpoint"$'\t'"$control_digest"$'\t'"$commit"$'\t'"$handoff_digest"$'\t'"$data_digest"$'\t'"$config_digest"$'\t'"${control_eval_trace_digests[$eval_key]}"$'\t'"${control_eval_trace_manifest_digests[$eval_key]}"$'\t-\t'"$BC_EXPECTED_R60_EVIDENCE"$'\n'
        run="${cost_eval_runs[$eval_key]}"
        config_digest="$(file_sha256 "$run/resolved-config.yaml")" || return 1
        branch_rows+="C-$eval_stage-EVAL"$'\t'"cost_aware_gated_${eval_key}_eval"$'\t'"$run"$'\t'"$cost_checkpoint"$'\t'"$cost_digest"$'\t'"$cost_checkpoint"$'\t'"$cost_digest"$'\t'"$commit"$'\t'"$handoff_digest"$'\t'"$data_digest"$'\t'"$config_digest"$'\t'"${cost_eval_trace_digests[$eval_key]}"$'\t'"${cost_eval_trace_manifest_digests[$eval_key]}"$'\t-\t'"$BC_EXPECTED_R60_EVIDENCE"$'\n'
    done
    printf '%s' "$branch_rows" >>"$results_dir/lineage.tsv"
    index_content="$(cat "$outer_attempt/native-training-runs.tsv")"
    atomic_write "$results_dir/run-index.tsv" "$index_content"$'\n'
    atomic_write "$results_dir/branch-checkpoints.env" \
        "control_checkpoint=$control_checkpoint"$'\n'\
"control_checkpoint_sha256=$control_digest"$'\n'\
"cost_checkpoint=$cost_checkpoint"$'\n'\
"cost_checkpoint_sha256=$cost_digest"$'\n'
    atomic_write "$outer_attempt/bc-only-complete.env" \
        "schema=$BC_ONLY_CONTRACT"$'\n'\
"result_root=$results_dir"$'\n'\
"control_checkpoint=$control_checkpoint"$'\n'\
"control_checkpoint_sha256=$control_digest"$'\n'\
"cost_checkpoint=$cost_checkpoint"$'\n'\
"cost_checkpoint_sha256=$cost_digest"$'\n'\
"runner_sha256=$runner_digest"$'\n'

    wandb_runs=("$control_run" "$cost_run")
    for eval_key in "${eval_keys[@]}"; do
        wandb_runs+=("${control_eval_runs[$eval_key]}" "${cost_eval_runs[$eval_key]}")
    done
    for wandb_run in "${wandb_runs[@]}"; do
        [[ -d "$wandb_run/wandb" && ! -L "$wandb_run/wandb" ]] || return 1
        if [[ "$wandb_run" == "$control_run" || "$wandb_run" == "$cost_run" ]]; then
            wandb_steps="$BRANCH_STEPS"
        else
            wandb_steps=0
        fi
        write_wandb_receipt "$wandb_run" "$wandb_steps" || return $?
        verify_wandb_receipt "$wandb_run" "$wandb_steps" || return $?
        evidence_files+=("$wandb_run/wandb-receipt.json")
        wandb_count=0
        while IFS= read -r -d '' wandb_file; do
            evidence_files+=("$wandb_file")
            ((wandb_count += 1))
        done < <(find "$wandb_run/wandb" -type f -size +0c -print0 | LC_ALL=C sort -z)
        ((wandb_count > 0)) || return 1
    done

    evidence_files+=(
        "$results_dir/contract.env" "$results_dir/lineage.tsv"
        "$results_dir/run-index.tsv" "$results_dir/branch-checkpoints.env"
        "$outer_attempt/bc-only-complete.env" "$BC_RUNNER_PATH"
        "$BC_CPU_RECEIPT" "$BC_CPU_RECEIPT.sha256"
        "$QWEN_NATIVE_PROTOCOL_GATE_EVIDENCE" "$NATIVE_TRAIN_PROTOCOL_GATE_MANIFEST"
        "$QWEN_NATIVE_SMOKE_EVIDENCE" "$NATIVE_TRAIN_SMOKE_MANIFEST"
        "$QWEN_NATIVE_R60_EVIDENCE" "$BC_R60_RESULTS/evidence.sha256"
        "$BC_G3_RUN/resolved-config.yaml" "$BC_G3_RUN/run.env"
        "$BC_G3_RUN/train.log" "$BC_G3_TRACE" "$BC_G3_TRACE_MANIFEST"
        "$BC_G3_TRACE_SIDECAR" "$BC_G3_ANALYSIS/summary.json"
        "$BC_G3_ANALYSIS/summary.md" "$BC_G3_ANALYSIS/go_no_go.json"
        "$BC_G3_ANALYSIS/per_question.jsonl"
        "$BC_G3_ANALYSIS/per_trajectory.jsonl"
        "$control_run/train.log" "$control_run/resolved-config.yaml"
        "$control_run/run.env" "$control_run/lineage.tsv"
        "$control_run/native-training-contract.json"
        "$control_run/traces/train_trajectories.jsonl"
        "$control_run/traces/train_trajectories.manifest.json"
        "$control_run/traces/train_trajectories.manifest.json.sha256"
        "$cost_run/train.log" "$cost_run/resolved-config.yaml"
        "$cost_run/run.env" "$cost_run/lineage.tsv"
        "$cost_run/native-training-contract.json"
        "$cost_run/traces/train_trajectories.jsonl"
        "$cost_run/traces/train_trajectories.manifest.json"
        "$cost_run/traces/train_trajectories.manifest.json.sha256"
    )
    for eval_key in "${eval_keys[@]}"; do
        for run in "${control_eval_runs[$eval_key]}" "${cost_eval_runs[$eval_key]}"; do
            evidence_files+=(
                "$run/train.log" "$run/resolved-config.yaml" "$run/run.env"
                "$run/traces/eval_predictions.jsonl"
                "$run/traces/eval_predictions.manifest.json"
                "$run/traces/eval_predictions.manifest.json.sha256"
            )
        done
        evidence_files+=(
            "${paired_dirs[$eval_key]}/summary.json"
            "${paired_dirs[$eval_key]}/summary.md"
            "${paired_dirs[$eval_key]}/paired_results.csv"
            "${paired_dirs[$eval_key]}/correct_questions.csv"
            "${paired_dirs[$eval_key]}/wrong_questions.csv"
            "${paired_dirs[$eval_key]}/search_transition.csv"
        )
    done

    verify_bc_publish_identity "$commit" "$data_digest" "$runner_digest" ||
        return $?
    verify_native_checkpoint_digest "$BC_R60_CHECKPOINT" \
        "$BC_EXPECTED_R60_CHECKPOINT" || return $?
    verify_native_checkpoint_digest "$control_checkpoint" "$control_digest" || return $?
    verify_native_checkpoint_digest "$cost_checkpoint" "$cost_digest" || return $?
    verify_bc_results_semantics "$results_dir" "$control_checkpoint" \
        "$control_digest" "$cost_checkpoint" "$cost_digest" || return $?
    sync_path "$results_dir"
    sync_path "$outer_attempt"

    NATIVE_TRAIN_STAGE=bc-only
    publish_native_training_evidence "$outer_attempt" "$results_dir" \
        "$BC_ONLY_NAMESPACE" "$BC_ONLY_CONTRACT" \
        "${evidence_files[@]}" || return $?
    marker="$MANIFEST_DIR/$BC_ONLY_NAMESPACE/$(basename -- "$outer_attempt").ok"
    evidence_digest="$(file_sha256 "$results_dir/evidence.sha256")" || return 1
    [[ "$(bc_value "$marker")" == "$evidence_digest" &&
        "$(bc_value "$outer_attempt/result-contract")" == "$BC_ONLY_CONTRACT" &&
        "$(bc_value "$outer_attempt/result-root")" == "$results_dir" &&
        "$(bc_value "$outer_attempt/evidence-marker")" == "$marker" &&
        "$(bc_value "$outer_attempt/evidence-digest")" == "$evidence_digest" ]] ||
        return 1
    verify_native_training_checksum_manifest "$results_dir/evidence.sha256" ||
        return $?
    NATIVE_TRAIN_STAGE=main

    printf 'Completed and sealed post-hoc B20/C20 comparison: %s\n' "$results_dir"
    printf 'Returning controlled outer status %s for the pinned shutdown watchdog.\n' \
        "$BC_ONLY_TERMINAL_CODE"
    return "$BC_ONLY_TERMINAL_CODE"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    case "${1:-}" in
        --cpu-prepare)
            [[ "$#" == 1 ]] || exit 64
            prepare_bc_cpu_receipt
            ;;
        *)
            qwen_native_train_main "$@"
            ;;
    esac
fi
