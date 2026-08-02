#!/usr/bin/env bash
set -Eeuo pipefail

# Complete the canonical A/R endpoint matrix without performing another update.
# This operator deliberately inherits the deployed B/C-only runner: that runner
# already pins the canonical checkout, native-v4 handoff, post-trained parent,
# exact R60 checkpoint, G3 evidence, and all predecessor receipts used by B/C.
AR_PROJECT_ROOT="${AUTODL_ROOT:-/root/autodl-tmp/search-r1}"
AR_SOURCE_ENTRY="$AR_PROJECT_ROOT/operator/12_gpu_qwen_native_bc_only.sh"
AR_RUNNER_PATH="$(readlink -f -- "${BASH_SOURCE[0]}")"

# Set the exact predecessor defaults before sourcing 12 -> 09.  The inherited
# native runner snapshots these values while it is sourced.
export QWEN_NATIVE_PROTOCOL_GATE_EVIDENCE="${QWEN_NATIVE_PROTOCOL_GATE_EVIDENCE:-$AR_PROJECT_ROOT/manifests/qwen-native-gate/20260728T044634Z-2051-4045.ok}"
export QWEN_NATIVE_SMOKE_EVIDENCE="${QWEN_NATIVE_SMOKE_EVIDENCE:-$AR_PROJECT_ROOT/manifests/qwen-native-training-smoke/20260728T061246Z-1316-15067.ok}"
export QWEN_NATIVE_R60_EVIDENCE="${QWEN_NATIVE_R60_EVIDENCE:-$AR_PROJECT_ROOT/manifests/qwen-native-training-r60-only/20260728T092026Z-2906-16060.ok}"

[[ -f "$AR_SOURCE_ENTRY" && ! -L "$AR_SOURCE_ENTRY" ]] || {
    printf 'Missing deployed B/C foundation runner: %s\n' "$AR_SOURCE_ENTRY" >&2
    exit 1
}

# shellcheck source=12_gpu_qwen_native_bc_only.sh
source "$AR_SOURCE_ENTRY"

readonly AR_CONTRACT=qwen-native-training-ar-eval-only-v1
readonly AR_NAMESPACE=qwen-native-training-ar-eval-only
readonly AR_CPU_CONTRACT=qwen-native-ar-eval-only-cpu-v1
readonly AR_DEPLOY_PATH="$AR_PROJECT_ROOT/operator/14_gpu_qwen_native_ar_eval_only.sh"
readonly AR_CPU_ROOT="$MANIFEST_DIR/qwen-native-ar-eval-only-cpu"
readonly AR_EXPECTED_SOURCE_SHA256=d696fcd752e301e81e7b555df961613837605e2edd25add550ad1c867e6f1a9f
readonly AR_EXPECTED_VAL_SHA256=4e600c66a52dd98e24f5f2d9b750e0c86784173fb107f5aa8ce1073c83bf1bfe
readonly AR_EXPECTED_NQ_SHA256=6dc2bd4bbc80ad37618d38e3ec30590e25ae65f6c92024eb354ff6259289dfbf
readonly AR_EXPECTED_MULTIHOP_SHA256=ae0b08cb7b6cb69255ce9af17455fa5b6ba8e3e58373866ca94aa26f3cac87d6
readonly AR_EXPECTED_PAIRED_SHA256=0070b03b8ec37887f9b20695431c3c7c7edd6fa9c768e9ebb493217f509ec95c
readonly AR_EXPECTED_CONFIG_A_VAL_SHA256=06d1820a298652662f565142a485e4d0af62c60ed468f2fae1ff2bc72a3959b2
readonly AR_EXPECTED_CONFIG_R_VAL_SHA256=9213c24bf28bc582c182fb64c581b1d8873944c47e19ceaa556ebb4db8351ce3
readonly AR_EXPECTED_CONFIG_A_NQ_SHA256=c33ee5379fe9a1837c4c5cc4eecb7dcb346a96902305ac1a942fdc180f4c3da1
readonly AR_EXPECTED_CONFIG_R_NQ_SHA256=bf5f9d54f8fd950024d99f21a1ade459edc11902336ab5b1855e163ff3bcac85
readonly AR_EXPECTED_CONFIG_A_MULTIHOP_SHA256=20c88b24423fdd43cf2c3aa5c726dff4279e43da0ea62ad286fb0a7cb064f4f8
readonly AR_EXPECTED_CONFIG_R_MULTIHOP_SHA256=d9da8aca8fad778c730acd4fb311cb7067a9e49283db479ad1b419285e37d04b

readonly AR_G3_RUNNER_SHA256=3b1985a8321ad6b19d5c7559fb55a5731d4e85f2926547233c933754c4ef4d3e
readonly AR_G3_RECEIPT="$MANIFEST_DIR/qwen-native-g3-only-cpu/$BC_R60_ATTEMPT-$AR_G3_RUNNER_SHA256.env"
readonly AR_EXPECTED_G3_RECEIPT_SHA256=b499b53cfa7561bd57aeb4cb4cde25bbd8488907fac610fc9e75e3a1c6f7c4c1

# Bind this completion run to the final B/C recovery result that motivated it.
readonly AR_BC_ATTEMPT=20260801T041947Z-7015-12411
readonly AR_BC_CONTRACT=qwen-native-training-bc-recovery-v1
readonly AR_BC_NAMESPACE=qwen-native-training-bc-recovery
readonly AR_BC_RESULTS="$NATIVE_TRAIN_RESULTS_ROOT/attempts/$AR_BC_ATTEMPT"
readonly AR_BC_OUTER="$ATTEMPTS_ROOT/gpu/$AR_BC_ATTEMPT"
readonly AR_BC_MARKER="$MANIFEST_DIR/$AR_BC_NAMESPACE/$AR_BC_ATTEMPT.ok"
readonly AR_BC_EVIDENCE="$AR_BC_RESULTS/evidence.sha256"
readonly AR_EXPECTED_BC_EVIDENCE=8ec72644d618c194a248e82965fcf4b7473e69acad0a5c5cd519f9df2c03827c

readonly AR_STAGE_ORDER=A-VAL-EVAL,R-VAL-EVAL,A-NQ-TEST-EVAL,R-NQ-TEST-EVAL,A-MULTIHOP-EVAL,R-MULTIHOP-EVAL

export QWEN_NATIVE_TRAIN_STAGE=main

AR_CPU_RECEIPT=''
AR_CPU_RECEIPT_DIGEST=''
AR_SOURCE_DIGEST=''
AR_RUNNER_DIGEST=''
AR_PAIRED_DIGEST=''
AR_DATA_MANIFEST_DIGEST=''
AR_DATA_VAL_DIGEST=''
AR_DATA_NQ_DIGEST=''
AR_DATA_MULTIHOP_DIGEST=''
AR_CONFIG_A_VAL_DIGEST=''
AR_CONFIG_R_VAL_DIGEST=''
AR_CONFIG_A_NQ_DIGEST=''
AR_CONFIG_R_NQ_DIGEST=''
AR_CONFIG_A_MULTIHOP_DIGEST=''
AR_CONFIG_R_MULTIHOP_DIGEST=''
AR_PREFLIGHT_COMMIT=''
AR_PREFLIGHT_HANDOFF=''
AR_PREFLIGHT_BASE_DIGEST=''
AR_PREFLIGHT_DATA_DIGEST=''
AR_PREFLIGHT_RUNNER_DIGEST=''

ar_value() {
    local path="$1"
    [[ -f "$path" && ! -L "$path" ]] || return 1
    tr -d '\r\n' <"$path"
}

require_ar_cpu_root() {
    local create="${1:-false}" project_root manifest_root cpu_root
    [[ -d "$MANIFEST_DIR" && ! -L "$MANIFEST_DIR" ]] || {
        printf 'Manifest root is not a regular directory: %s\n' \
            "$MANIFEST_DIR" >&2
        return 1
    }
    if [[ -e "$AR_CPU_ROOT" || -L "$AR_CPU_ROOT" ]]; then
        [[ -d "$AR_CPU_ROOT" && ! -L "$AR_CPU_ROOT" ]] || {
            printf 'A/R CPU receipt root is not a regular directory: %s\n' \
                "$AR_CPU_ROOT" >&2
            return 1
        }
    elif [[ "$create" == true ]]; then
        mkdir "$AR_CPU_ROOT" || return $?
    else
        return 1
    fi
    project_root="$(readlink -f -- "$AR_PROJECT_ROOT")" || return 1
    manifest_root="$(readlink -f -- "$MANIFEST_DIR")" || return 1
    cpu_root="$(readlink -f -- "$AR_CPU_ROOT")" || return 1
    [[ "$manifest_root" == "$project_root/manifests" ]] || {
        printf 'Manifest root escaped the project directory.\n' >&2
        return 1
    }
    [[ "$cpu_root" == "$manifest_root/qwen-native-ar-eval-only-cpu" ]] || {
        printf 'A/R CPU receipt root escaped the manifest directory.\n' >&2
        return 1
    }
}

ar_config_file() {
    local variant="$1"
    case "$variant" in
        qwen_native_a_val|qwen_native_r_val|qwen_native_a_nq_test|qwen_native_r_nq_test|qwen_native_a_multihop|qwen_native_r_multihop)
            printf '%s/config-2gpu-%s-eval.yaml\n' "$MANIFEST_DIR" "$variant"
            ;;
        *)
            printf 'Unregistered A/R config variant: %s\n' "$variant" >&2
            return 64
            ;;
    esac
}

ar_eval_spec() {
    local variant="$1"
    case "$variant" in
        qwen_native_a_val|qwen_native_r_val)
            printf 'val\t128\tval\t5\n'
            ;;
        qwen_native_a_nq_test|qwen_native_r_nq_test)
            printf 'nq_test\t128\tnq_test_eval\t5\n'
            ;;
        qwen_native_a_multihop|qwen_native_r_multihop)
            # The completed B/C multihop runs needed about 3,141-3,203 seconds.
            # Five RMB at 5.76 RMB/hour is only 3,125 seconds, so use the
            # recovery runner's conservative ten-RMB envelope here.
            printf 'multihop\t256\tmultihop_eval\t10\n'
            ;;
        *)
            printf 'Unregistered A/R evaluation variant: %s\n' "$variant" >&2
            return 64
            ;;
    esac
}

verify_ar_bc_recovery() {
    local marker_digest evidence_digest
    [[ -d "$AR_BC_RESULTS" && ! -L "$AR_BC_RESULTS" &&
        -d "$AR_BC_OUTER" && ! -L "$AR_BC_OUTER" &&
        -f "$AR_BC_MARKER" && ! -L "$AR_BC_MARKER" &&
        -f "$AR_BC_EVIDENCE" && ! -L "$AR_BC_EVIDENCE" &&
        -f "$AR_BC_OUTER/.failed" && ! -L "$AR_BC_OUTER/.failed" &&
        ! -e "$AR_BC_OUTER/.success" &&
        ! -e "$AR_BC_OUTER/.starting" && ! -e "$AR_BC_OUTER/.running" &&
        "$(ar_value "$AR_BC_OUTER/terminal")" == failed &&
        "$(ar_value "$AR_BC_OUTER/exit-code")" == 203 &&
        "$(ar_value "$AR_BC_OUTER/result-contract")" == "$AR_BC_CONTRACT" &&
        "$(ar_value "$AR_BC_OUTER/result-root")" == "$AR_BC_RESULTS" &&
        "$(ar_value "$AR_BC_OUTER/evidence-marker")" == "$AR_BC_MARKER" ]] || {
        printf 'Final B/C recovery terminal binding changed.\n' >&2
        return 1
    }
    marker_digest="$(ar_value "$AR_BC_MARKER")" || return 1
    evidence_digest="$(file_sha256 "$AR_BC_EVIDENCE")" || return 1
    [[ "$marker_digest" == "$AR_EXPECTED_BC_EVIDENCE" &&
        "$evidence_digest" == "$AR_EXPECTED_BC_EVIDENCE" &&
        "$(ar_value "$AR_BC_OUTER/evidence-digest")" == "$AR_EXPECTED_BC_EVIDENCE" ]] || {
        printf 'Final B/C recovery evidence digest changed.\n' >&2
        return 1
    }
    verify_native_training_checksum_manifest "$AR_BC_EVIDENCE" || return $?
    grep -Fxq "schema=$AR_BC_CONTRACT" "$AR_BC_RESULTS/contract.env" || return 1
    grep -Fxq 'stage=bc_recovery' "$AR_BC_RESULTS/contract.env" || return 1
    grep -Fxq "c_parent_checkpoint=$BC_R60_CHECKPOINT" \
        "$AR_BC_RESULTS/contract.env" || return 1
    grep -Fxq "c_parent_checkpoint_sha256=$BC_EXPECTED_R60_CHECKPOINT" \
        "$AR_BC_RESULTS/contract.env" || return 1
}

load_ar_bound_hashes() {
    local path
    AR_SOURCE_DIGEST="$(file_sha256 "$AR_SOURCE_ENTRY")" || return 1
    [[ "$AR_SOURCE_DIGEST" == "$AR_EXPECTED_SOURCE_SHA256" ]] || {
        printf 'Deployed B/C foundation runner changed.\n' >&2
        return 1
    }
    AR_PAIRED_DIGEST="$(file_sha256 "$NATIVE_TRAIN_PAIRED_CLI")" || return 1
    [[ "$AR_PAIRED_DIGEST" == "$AR_EXPECTED_PAIRED_SHA256" ]] || {
        printf 'The sealed paired A/R analyzer changed.\n' >&2
        return 1
    }
    AR_DATA_MANIFEST_DIGEST="$(file_sha256 "$NATIVE_TRAIN_MANIFEST")" || return 1
    AR_DATA_VAL_DIGEST="$(file_sha256 "$NATIVE_TRAIN_DATA_DIR/val_128.parquet")" || return 1
    AR_DATA_NQ_DIGEST="$(file_sha256 "$NATIVE_TRAIN_DATA_DIR/nq_test_128_native_v4.parquet")" || return 1
    AR_DATA_MULTIHOP_DIGEST="$(file_sha256 "$NATIVE_TRAIN_DATA_DIR/multihop_eval_256_native_v4.parquet")" || return 1
    [[ "$AR_DATA_VAL_DIGEST" == "$AR_EXPECTED_VAL_SHA256" &&
        "$AR_DATA_NQ_DIGEST" == "$AR_EXPECTED_NQ_SHA256" &&
        "$AR_DATA_MULTIHOP_DIGEST" == "$AR_EXPECTED_MULTIHOP_SHA256" ]] || {
        printf 'One or more sealed A/R endpoint datasets changed.\n' >&2
        return 1
    }

    path="$(ar_config_file qwen_native_a_val)" || return $?
    [[ -f "$path" && ! -L "$path" ]] || return 1
    AR_CONFIG_A_VAL_DIGEST="$(file_sha256 "$path")" || return 1
    path="$(ar_config_file qwen_native_r_val)" || return $?
    [[ -f "$path" && ! -L "$path" ]] || return 1
    AR_CONFIG_R_VAL_DIGEST="$(file_sha256 "$path")" || return 1
    path="$(ar_config_file qwen_native_a_nq_test)" || return $?
    [[ -f "$path" && ! -L "$path" ]] || return 1
    AR_CONFIG_A_NQ_DIGEST="$(file_sha256 "$path")" || return 1
    path="$(ar_config_file qwen_native_r_nq_test)" || return $?
    [[ -f "$path" && ! -L "$path" ]] || return 1
    AR_CONFIG_R_NQ_DIGEST="$(file_sha256 "$path")" || return 1
    path="$(ar_config_file qwen_native_a_multihop)" || return $?
    [[ -f "$path" && ! -L "$path" ]] || return 1
    AR_CONFIG_A_MULTIHOP_DIGEST="$(file_sha256 "$path")" || return 1
    path="$(ar_config_file qwen_native_r_multihop)" || return $?
    [[ -f "$path" && ! -L "$path" ]] || return 1
    AR_CONFIG_R_MULTIHOP_DIGEST="$(file_sha256 "$path")" || return 1
    [[ "$AR_CONFIG_A_VAL_DIGEST" == "$AR_EXPECTED_CONFIG_A_VAL_SHA256" &&
        "$AR_CONFIG_R_VAL_DIGEST" == "$AR_EXPECTED_CONFIG_R_VAL_SHA256" &&
        "$AR_CONFIG_A_NQ_DIGEST" == "$AR_EXPECTED_CONFIG_A_NQ_SHA256" &&
        "$AR_CONFIG_R_NQ_DIGEST" == "$AR_EXPECTED_CONFIG_R_NQ_SHA256" &&
        "$AR_CONFIG_A_MULTIHOP_DIGEST" == "$AR_EXPECTED_CONFIG_A_MULTIHOP_SHA256" &&
        "$AR_CONFIG_R_MULTIHOP_DIGEST" == "$AR_EXPECTED_CONFIG_R_MULTIHOP_SHA256" ]] || {
        printf 'One or more sealed A/R endpoint configs changed.\n' >&2
        return 1
    }
}

verify_ar_foundation() {
    local commit="$1" handoff_digest="$2" base_model="$3" base_digest="$4"
    [[ "$AR_RUNNER_PATH" == "$AR_DEPLOY_PATH" &&
        "$AR_SOURCE_ENTRY" == "$BC_DEPLOY_PATH" &&
        -f "$AR_RUNNER_PATH" && ! -L "$AR_RUNNER_PATH" ]] || {
        printf 'Deploy the exact A/R runner and source the deployed B/C runner.\n' >&2
        return 1
    }
    verify_bc_foundation "$commit" "$handoff_digest" "$base_model" \
        "$base_digest" || return $?
    load_ar_bound_hashes || return $?
    [[ "$AR_DATA_MANIFEST_DIGEST" == "$BC_EXPECTED_DATA" ]] || return 1
    verify_bc_cpu_receipt "$AR_SOURCE_DIGEST" || {
        printf 'The exact G3/B/C foundation CPU receipt is unavailable.\n' >&2
        return 1
    }
    [[ -f "$AR_G3_RECEIPT" && ! -L "$AR_G3_RECEIPT" &&
        -f "$AR_G3_RECEIPT.sha256" && ! -L "$AR_G3_RECEIPT.sha256" &&
        "$(file_sha256 "$AR_G3_RECEIPT")" == "$AR_EXPECTED_G3_RECEIPT_SHA256" &&
        "$(ar_value "$AR_G3_RECEIPT.sha256")" == \
            "$AR_EXPECTED_G3_RECEIPT_SHA256  ${AR_G3_RECEIPT#"$AR_PROJECT_ROOT/"}" ]] || {
        printf 'The exact G3-only CPU receipt changed or is missing.\n' >&2
        return 1
    }
    verify_ar_bc_recovery || return $?
}

ar_receipt_content() {
    local runner_digest="$1"
    printf '%s\n' \
        "schema=$AR_CPU_CONTRACT" \
        'experiment_class=post_hoc_endpoint_completion' \
        'cpu_prepare_only=true' \
        'gpu_started=false' \
        'training_executed=false' \
        "checkout_commit=$BC_EXPECTED_CHECKOUT" \
        "cpu_handoff_sha256=$BC_EXPECTED_HANDOFF" \
        "base_model=$MODEL_DIR" \
        "base_model_sha256=$BC_EXPECTED_BASE" \
        "data_manifest=$NATIVE_TRAIN_MANIFEST" \
        "data_manifest_sha256=$AR_DATA_MANIFEST_DIGEST" \
        "val_data=$NATIVE_TRAIN_DATA_DIR/val_128.parquet" \
        "val_data_sha256=$AR_DATA_VAL_DIGEST" \
        "nq_test_data=$NATIVE_TRAIN_DATA_DIR/nq_test_128_native_v4.parquet" \
        "nq_test_data_sha256=$AR_DATA_NQ_DIGEST" \
        "multihop_data=$NATIVE_TRAIN_DATA_DIR/multihop_eval_256_native_v4.parquet" \
        "multihop_data_sha256=$AR_DATA_MULTIHOP_DIGEST" \
        "config_a_val=$(ar_config_file qwen_native_a_val)" \
        "config_a_val_sha256=$AR_CONFIG_A_VAL_DIGEST" \
        "config_r_val=$(ar_config_file qwen_native_r_val)" \
        "config_r_val_sha256=$AR_CONFIG_R_VAL_DIGEST" \
        "config_a_nq_test=$(ar_config_file qwen_native_a_nq_test)" \
        "config_a_nq_test_sha256=$AR_CONFIG_A_NQ_DIGEST" \
        "config_r_nq_test=$(ar_config_file qwen_native_r_nq_test)" \
        "config_r_nq_test_sha256=$AR_CONFIG_R_NQ_DIGEST" \
        "config_a_multihop=$(ar_config_file qwen_native_a_multihop)" \
        "config_a_multihop_sha256=$AR_CONFIG_A_MULTIHOP_DIGEST" \
        "config_r_multihop=$(ar_config_file qwen_native_r_multihop)" \
        "config_r_multihop_sha256=$AR_CONFIG_R_MULTIHOP_DIGEST" \
        "source_runner=$AR_SOURCE_ENTRY" \
        "source_runner_sha256=$AR_SOURCE_DIGEST" \
        "runner=$AR_DEPLOY_PATH" \
        "runner_sha256=$runner_digest" \
        "paired_eval=$NATIVE_TRAIN_PAIRED_CLI" \
        "paired_eval_sha256=$AR_PAIRED_DIGEST" \
        "r60_evidence=$BC_EXPECTED_R60_MARKER" \
        "r60_evidence_sha256=$BC_EXPECTED_R60_EVIDENCE" \
        "r60_checkpoint=$BC_R60_CHECKPOINT" \
        "r60_checkpoint_sha256=$BC_EXPECTED_R60_CHECKPOINT" \
        "source_cpu_receipt=$BC_CPU_RECEIPT" \
        "source_cpu_receipt_sha256=$BC_CPU_RECEIPT_DIGEST" \
        "g3_receipt=$AR_G3_RECEIPT" \
        "g3_receipt_sha256=$AR_EXPECTED_G3_RECEIPT_SHA256" \
        "g3_trace_sha256=$BC_EXPECTED_G3_TRACE" \
        "g3_analysis_sha256=$BC_EXPECTED_G3_SUMMARY" \
        "g3_decision=NO-GO" \
        "bc_recovery_marker=$AR_BC_MARKER" \
        "bc_recovery_evidence=$AR_BC_EVIDENCE" \
        "bc_recovery_evidence_sha256=$AR_EXPECTED_BC_EVIDENCE" \
        "prompt_version=$NATIVE_TRAIN_PROMPT_VERSION" \
        'eval_group_size=1' \
        'decoding=greedy' \
        'seed=42' \
        'val_rows=128' \
        'nq_test_rows=128' \
        'multihop_rows=256' \
        'gpu_training_authorized=false' \
        'gpu_evaluation_authorized=true' \
        'manual_gpu_start_required=true'
}

verify_ar_cpu_receipt() {
    local runner_digest="$1" directory receipt expected digest sidecar
    local canonical_root canonical_directory listing expected_listing
    require_ar_cpu_root false || return $?
    directory="$AR_CPU_ROOT/$runner_digest"
    receipt="$directory/receipt.env"
    expected="$(ar_receipt_content "$runner_digest")"$'\n'
    [[ -d "$directory" && ! -L "$directory" &&
        -f "$receipt" && ! -L "$receipt" &&
        -f "$receipt.sha256" && ! -L "$receipt.sha256" &&
        -f "$directory/.success" && ! -L "$directory/.success" &&
        ! -e "$directory/.failed" && ! -e "$directory/.running" &&
        ! -e "$directory/.starting" &&
        "$(ar_value "$directory/terminal")" == success &&
        "$(ar_value "$directory/exit-code")" == 0 ]] || return 1
    canonical_root="$(readlink -f -- "$AR_CPU_ROOT")" || return 1
    canonical_directory="$(readlink -f -- "$directory")" || return 1
    [[ "$canonical_directory" == "$canonical_root/$runner_digest" ]] || return 1
    listing="$(find "$directory" -mindepth 1 -maxdepth 1 -printf '%f\n' | LC_ALL=C sort)" ||
        return 1
    expected_listing="$(printf '%s\n' .success exit-code receipt.env receipt.env.sha256 terminal | LC_ALL=C sort)"
    [[ "$listing" == "$expected_listing" ]] || return 1
    cmp -s <(printf '%s' "$expected") "$receipt" || return 1
    digest="$(file_sha256 "$receipt")" || return 1
    sidecar="$(ar_value "$receipt.sha256")" || return 1
    [[ "$sidecar" == "$digest  ${receipt#"$AR_PROJECT_ROOT/"}" ]] || return 1
    AR_CPU_RECEIPT="$receipt"
    AR_CPU_RECEIPT_DIGEST="$digest"
}

prepare_ar_cpu_receipt() {
    local commit handoff_digest base_model base_digest runner_digest
    local directory staging receipt content digest sidecar
    [[ "$AR_RUNNER_PATH" == "$AR_DEPLOY_PATH" &&
        -f "$AR_RUNNER_PATH" && ! -L "$AR_RUNNER_PATH" ]] || {
        printf 'Deploy this exact runner first: %s\n' "$AR_DEPLOY_PATH" >&2
        return 1
    }
    export CUDA_VISIBLE_DEVICES=''
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
    export WANDB_MODE=offline PIP_NO_INDEX=1 PYTHONDONTWRITEBYTECODE=1
    commit="$(expected_commit)" || return 1
    handoff_digest="$(ar_value "$MANIFEST_DIR/cpu.ok")" || return 1
    base_model="$(readlink -f -- "$MODEL_DIR")" || return 1
    base_digest="$(tree_sha256 "$base_model")" || return 1
    verify_ar_foundation "$commit" "$handoff_digest" "$base_model" \
        "$base_digest" || return $?
    runner_digest="$(file_sha256 "$AR_RUNNER_PATH")" || return 1
    AR_RUNNER_DIGEST="$runner_digest"
    directory="$AR_CPU_ROOT/$runner_digest"
    receipt="$directory/receipt.env"
    require_ar_cpu_root true || return $?
    if [[ -e "$directory" || -L "$directory" ]]; then
        verify_ar_cpu_receipt "$runner_digest" || {
            printf 'Refusing a differing or incomplete A/R CPU receipt.\n' >&2
            return 1
        }
    else
        staging="$AR_CPU_ROOT/.pending-$runner_digest-$$-$RANDOM"
        [[ ! -e "$staging" && ! -L "$staging" ]] || return 1
        mkdir "$staging"
        content="$(ar_receipt_content "$runner_digest")"$'\n'
        atomic_write "$staging/receipt.env" "$content"
        digest="$(file_sha256 "$staging/receipt.env")" || return 1
        sidecar="$digest  ${receipt#"$AR_PROJECT_ROOT/"}"$'\n'
        atomic_write "$staging/receipt.env.sha256" "$sidecar"
        atomic_write "$staging/exit-code" $'0\n'
        atomic_write "$staging/terminal" $'success\n'
        atomic_write "$staging/.success" ''
        sync_path "$staging"
        mv -- "$staging" "$directory"
        sync_path "$AR_CPU_ROOT"
        verify_ar_cpu_receipt "$runner_digest" || return $?
    fi
    printf 'A/R evaluation CPU preparation complete: %s\n' "$AR_CPU_RECEIPT"
    printf 'GPU evaluation and training were not started.\n'
}

# Validate an actual run config against the sealed CPU-composed variant while
# allowing only run-specific output, trace, checkpoint path, and trace digest.
verify_ar_eval_config() {
    local actual="$1" reference="$2" variant="$3" model="$4"
    local model_digest="$5" output_dir="$6" trace_dir="$7" run_id="$8"
    "$TRAIN_ENV/bin/python" - "$actual" "$reference" "$variant" "$model" \
        "$model_digest" "$output_dir" "$trace_dir" "$run_id" <<'PY'
from copy import deepcopy
from pathlib import Path
import sys

from omegaconf import OmegaConf

(actual_raw, reference_raw, variant, model_raw, model_digest, output_raw,
 trace_raw, run_id) = sys.argv[1:]
actual_path = Path(actual_raw)
reference_path = Path(reference_raw)
for path in (actual_path, reference_path):
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"evaluation config is missing or symlinked: {path}")

actual = OmegaConf.to_container(OmegaConf.load(actual_path), resolve=True)
reference = OmegaConf.to_container(OmegaConf.load(reference_path), resolve=True)

def get(root, *keys):
    current = root
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            raise SystemExit(f"missing config field: {'.'.join(keys)}")
        current = current[key]
    return current

checks = {
    "model": Path(get(actual, "actor_rollout_ref", "model", "path")).resolve()
        == Path(model_raw).resolve(),
    "critic_tokenizer": Path(get(actual, "critic", "model", "tokenizer_path")).resolve()
        == Path(model_raw).resolve(),
    "reward_input_tokenizer": Path(
        get(actual, "reward_model", "model", "input_tokenizer")
    ).resolve() == Path(model_raw).resolve(),
    "output": Path(get(actual, "trainer", "default_local_dir"))
        == Path(output_raw),
    "trace_output": Path(get(actual, "trainer", "trace_output_dir"))
        == Path(trace_raw),
    "trace_stage": get(actual, "trainer", "trace_stage") == variant,
    "trace_run": get(actual, "trainer", "trace_run_id") == run_id,
    "trace_digest": get(actual, "trainer", "trace_checkpoint_digest")
        == model_digest,
    "protocol": get(actual, "tool_protocol") == "qwen35_native",
    "group": get(actual, "data", "eval_group_size") == 1,
    "greedy": get(actual, "actor_rollout_ref", "rollout", "do_sample") is False,
    "seed": get(actual, "trainer", "seed") == 42,
    "val_only": get(actual, "trainer", "val_only") is True,
    "val_before": get(actual, "trainer", "val_before_train") is True,
    "steps": get(actual, "trainer", "total_training_steps") == 1,
    "save": get(actual, "trainer", "save_freq") == -1,
    "test": get(actual, "trainer", "test_freq") == -1,
    "gpus": get(actual, "trainer", "n_gpus_per_node") == 2,
    "response": get(actual, "data", "max_response_length") == 500,
    "observation": get(actual, "data", "max_obs_length") == 500,
    "turns": get(actual, "max_turns") == 4,
    "topk": get(actual, "retriever", "topk") == 3,
}
failed = sorted(name for name, passed in checks.items() if not passed)
if failed:
    raise SystemExit("A/R evaluation config mismatch: " + ", ".join(failed))

dynamic = (
    ("actor_rollout_ref", "model", "path"),
    ("critic", "model", "tokenizer_path"),
    ("reward_model", "model", "input_tokenizer"),
    ("trainer", "default_local_dir"),
    ("trainer", "trace_output_dir"),
    ("trainer", "trace_run_id"),
    ("trainer", "trace_checkpoint_digest"),
)
for root in (actual, reference):
    for keys in dynamic:
        current = root
        for key in keys[:-1]:
            current = current[key]
        current[keys[-1]] = "<BOUND-DYNAMIC>"
if actual != reference:
    raise SystemExit("actual A/R config differs from its sealed CPU config")
PY
}

# Eval-only analogue of the recovery job wrapper.  It has no train case, and
# rejects every variant outside the six registered A/R endpoints.
run_ar_eval_job() {
    local variant="$1" model_path="$2" input_model_digest="$3"
    local endpoint expected_rows eval_artifact budget_rmb timeout_seconds
    local parent run_dir started_epoch started_at rc trace_manifest reference
    local trace_output_dir run_id
    IFS=$'\t' read -r endpoint expected_rows eval_artifact budget_rmb \
        <<<"$(ar_eval_spec "$variant")" || return $?
    [[ "$input_model_digest" =~ ^[0-9a-f]{64}$ &&
        "$EVAL_GROUP_SIZE" == 1 && -z "$EVAL_DATA_FILE" &&
        "$EVAL_EXPECTED_ROWS" == "$expected_rows" ]] || return 64
    case "$variant" in
        qwen_native_a_*)
            [[ "$(readlink -f -- "$model_path")" == "$(readlink -f -- "$MODEL_DIR")" &&
                "$input_model_digest" == "$BC_EXPECTED_BASE" ]] || return 64
            ;;
        qwen_native_r_*)
            [[ "$(readlink -f -- "$model_path")" == "$(readlink -f -- "$BC_R60_CHECKPOINT")" &&
                "$input_model_digest" == "$BC_EXPECTED_R60_CHECKPOINT" ]] || return 64
            ;;
        *) return 64 ;;
    esac
    reference="$(ar_config_file "$variant")" || return $?
    timeout_seconds="$(awk -v budget="$budget_rmb" -v price="$PRICE_PER_HOUR" \
        'BEGIN { printf "%d", budget / price * 3600 }')"
    ((timeout_seconds > 0)) || return 64

    parent="$RUNS_ROOT/eval/$variant"
    mkdir -p "$parent/attempts"
    run_dir="$parent/attempts/$(date -u +'%Y%m%dT%H%M%SZ')-$$-$RANDOM"
    mkdir "$run_dir"
    : >"$run_dir/.running"
    atomic_write "$parent/latest" "$run_dir"$'\n'
    started_epoch="$(date +%s)"
    started_at="$(utc_now)"
    trace_output_dir="$run_dir/traces"
    run_id="$(basename -- "$run_dir")"
    : >"$run_dir/train.log"
    printf 'Starting A/R eval %s (endpoint %s, budget %s RMB, timeout %ss); log: %s/train.log\n' \
        "$variant" "$endpoint" "$budget_rmb" "$timeout_seconds" "$run_dir"

    set +e
    AUTODL_CONFIG_ONLY=1 \
        OUTPUT_DIR="$run_dir/checkpoints" GPU_COUNT="$GPU_COUNT" \
        TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
        MAX_RESPONSE_LENGTH="$MAX_RESPONSE_LENGTH" \
        EVAL_DATA_FILE='' EVAL_GROUP_SIZE=1 \
        TRACE_OUTPUT_DIR="$trace_output_dir" TRACE_STAGE="$variant" \
        TRACE_RUN_ID="$run_id" TRACE_CHECKPOINT_DIGEST="$input_model_digest" \
        TOOL_PROTOCOL="$TOOL_PROTOCOL" AUTODL_ROOT="$PROJECT_ROOT" \
        bash "$CHECKOUT_DIR/scripts/autodl/train_small_grpo.sh" \
        eval "$variant" "$model_path" >"$run_dir/resolved-config.yaml" \
        2>>"$run_dir/train.log"
    rc=$?
    if ((rc == 0)); then
        verify_ar_eval_config "$run_dir/resolved-config.yaml" "$reference" \
            "$variant" "$model_path" "$input_model_digest" \
            "$run_dir/checkpoints" "$trace_output_dir" "$run_id" \
            >>"$run_dir/train.log" 2>&1
        rc=$?
    fi
    if ((rc == 0)); then
        mkdir -p "$run_dir/wandb"
        OUTPUT_DIR="$run_dir/checkpoints" GPU_COUNT="$GPU_COUNT" \
            TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
            MAX_RESPONSE_LENGTH="$MAX_RESPONSE_LENGTH" \
            EVAL_DATA_FILE='' EVAL_GROUP_SIZE=1 \
            TRACE_OUTPUT_DIR="$trace_output_dir" TRACE_STAGE="$variant" \
            TRACE_RUN_ID="$run_id" TRACE_CHECKPOINT_DIGEST="$input_model_digest" \
            WANDB_DIR="$run_dir/wandb" TOOL_PROTOCOL="$TOOL_PROTOCOL" \
            AUTODL_ROOT="$PROJECT_ROOT" \
            timeout --signal=TERM --kill-after=120s "${timeout_seconds}s" \
            bash "$CHECKOUT_DIR/scripts/autodl/train_small_grpo.sh" \
            eval "$variant" "$model_path" >>"$run_dir/train.log" 2>&1
        rc=$?
    fi
    if ((rc == 0)); then
        trace_manifest="$run_dir/traces/eval_predictions.manifest.json"
        PYTHONPATH="$CHECKOUT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
            "$TRAIN_ENV/bin/python" -m search_r1.trajectory_trace verify \
                --manifest "$trace_manifest" --expected-rows "$expected_rows" \
                >>"$run_dir/train.log" 2>&1
        rc=$?
    fi
    set -e
    finish_run_record "$run_dir" "$rc" "$started_epoch" "$started_at" \
        "$budget_rmb" "$timeout_seconds" eval "$variant" 0 "$model_path" \
        "$trace_output_dir"
    LAST_RUN_DIR="$run_dir"
    if ((rc != 0)); then
        printf 'A/R evaluation %s failed with exit code %s; inspect %s/train.log\n' \
            "$variant" "$rc" "$run_dir" >&2
        return "$rc"
    fi
}

# Override the training admission inherited from 09/12.  This contract permits
# only evaluation and requires the exact CPU receipt before any paid work.
require_qwen_native_train() {
    validate_gpu_inputs || return $?
    [[ "$GPU_COUNT" == 2 && "$TRAIN_BATCH_SIZE" == 8 &&
        "$MAX_RESPONSE_LENGTH" == 500 && "$TOOL_PROTOCOL" == qwen35_native &&
        "$DATA_DIR" == "$NATIVE_TRAIN_DATA_DIR" &&
        "$EVAL_GROUP_SIZE" == 1 && -z "$EVAL_DATA_FILE" &&
        "$RUN_BUDGET_PROFILE" == gated_followup &&
        "$NATIVE_TRAIN_STAGE" == main &&
        "$AR_RUNNER_PATH" == "$AR_DEPLOY_PATH" &&
        "$AR_SOURCE_ENTRY" == "$BC_DEPLOY_PATH" &&
        "$QWEN_NATIVE_PROTOCOL_GATE_EVIDENCE" == "$MANIFEST_DIR/qwen-native-gate/20260728T044634Z-2051-4045.ok" &&
        "$QWEN_NATIVE_SMOKE_EVIDENCE" == "$MANIFEST_DIR/qwen-native-training-smoke/20260728T061246Z-1316-15067.ok" &&
        "$QWEN_NATIVE_R60_EVIDENCE" == "$BC_EXPECTED_R60_MARKER" ]] || {
        printf 'A/R eval-only requires exact native-v4 markers, two GPUs, group 1, and response 500.\n' >&2
        return 64
    }
}

qwen_native_train_preflight() {
    local commit="$1" handoff_digest="$2" base_digest="$3"
    local base_model runner_digest
    base_model="$(readlink -f -- "$MODEL_DIR")" || return 1
    verify_ar_foundation "$commit" "$handoff_digest" "$base_model" \
        "$base_digest" || return $?
    runner_digest="$(file_sha256 "$AR_RUNNER_PATH")" || return 1
    AR_RUNNER_DIGEST="$runner_digest"
    verify_ar_cpu_receipt "$runner_digest" || {
        printf 'Run --cpu-prepare with this exact A/R runner first.\n' >&2
        return 1
    }
    AR_PREFLIGHT_COMMIT="$commit"
    AR_PREFLIGHT_HANDOFF="$handoff_digest"
    AR_PREFLIGHT_BASE_DIGEST="$base_digest"
    AR_PREFLIGHT_DATA_DIGEST="$AR_DATA_MANIFEST_DIGEST"
    AR_PREFLIGHT_RUNNER_DIGEST="$runner_digest"

    # Keep the inherited native-v4 preflight state in lockstep; the inherited
    # dispatcher independently revalidates it before calling our pipeline.
    NATIVE_TRAIN_PREFLIGHT_STAGE=main
    NATIVE_TRAIN_PREFLIGHT_COMMIT="$commit"
    NATIVE_TRAIN_PREFLIGHT_HANDOFF="$handoff_digest"
    NATIVE_TRAIN_PREFLIGHT_BASE_DIGEST="$base_digest"
    NATIVE_TRAIN_PREFLIGHT_DATA_DIGEST="$AR_DATA_MANIFEST_DIGEST"
    NATIVE_TRAIN_PREFLIGHT_PROTOCOL_GATE_DIGEST="$NATIVE_TRAIN_PROTOCOL_GATE_DIGEST"
    NATIVE_TRAIN_PREFLIGHT_SMOKE_DIGEST="$NATIVE_TRAIN_SMOKE_DIGEST"
}

verify_ar_results_semantics() {
    local results="$1"
    "$TRAIN_ENV/bin/python" - "$results" "$AR_CONTRACT" "$AR_STAGE_ORDER" \
        "$MODEL_DIR" "$BC_EXPECTED_BASE" "$BC_R60_CHECKPOINT" \
        "$BC_EXPECTED_R60_CHECKPOINT" "$PROJECT_ROOT" <<'PY'
import csv
import hashlib
from pathlib import Path
import sys

(root_raw, contract, stage_order, base_raw, base_digest, r60_raw,
 r60_digest, project_raw) = sys.argv[1:]
root = Path(root_raw)
base = Path(base_raw)
r60 = Path(r60_raw)
project = Path(project_raw)

def env(path):
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in values:
            raise SystemExit(f"duplicate env key: {path}:{key}")
        values[key] = value
    return values

contract_env = env(root / "contract.env")
expected_contract = {
    "schema": contract,
    "stage": "ar_eval_only",
    "stage_order": stage_order,
    "training_executed": "false",
}
for key, value in expected_contract.items():
    if contract_env.get(key) != value:
        raise SystemExit(f"A/R result contract mismatch: {key}")

with (root / "lineage.tsv").open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
with (root / "run-index.tsv").open(encoding="utf-8", newline="") as handle:
    index = list(csv.DictReader(handle, delimiter="\t"))
expected = [
    ("A-VAL-EVAL", "parent_val_eval", base, base_digest,
     "qwen_native_a_val", 128, 5),
    ("R-VAL-EVAL", "reproduced_val_eval", r60, r60_digest,
     "qwen_native_r_val", 128, 5),
    ("A-NQ-TEST-EVAL", "parent_nq_test_eval", base, base_digest,
     "qwen_native_a_nq_test", 128, 5),
    ("R-NQ-TEST-EVAL", "reproduced_nq_test_eval", r60, r60_digest,
     "qwen_native_r_nq_test", 128, 5),
    ("A-MULTIHOP-EVAL", "parent_multihop_eval", base, base_digest,
     "qwen_native_a_multihop", 256, 10),
    ("R-MULTIHOP-EVAL", "reproduced_multihop_eval", r60, r60_digest,
     "qwen_native_r_multihop", 256, 10),
]
if len(rows) != len(expected) or len(index) != len(expected):
    raise SystemExit("A/R result row count mismatch")
for row, item, index_row in zip(rows, expected, index):
    stage, role, checkpoint, digest, variant, expected_rows, budget = item
    if (row["stage"], row["role"]) != (stage, role):
        raise SystemExit("A/R lineage order mismatch")
    if (Path(row["checkpoint"]) != checkpoint or
            row["checkpoint_digest"] != digest or
            Path(row["parent_checkpoint"]) != checkpoint or
            row["parent_checkpoint_digest"] != digest):
        raise SystemExit("A/R checkpoint lineage mismatch")
    run = Path(row["run_dir"])
    expected_parent = project / "runs" / "eval" / variant / "attempts"
    if (not run.is_dir() or run.is_symlink() or
            run.resolve(strict=True).parent != expected_parent.resolve(strict=True)):
        raise SystemExit("A/R run directory escaped its registered variant")
    if index_row != {"stage": stage, "role": role, "run_dir": str(run)}:
        raise SystemExit("A/R run index mismatch")
    run_env = env(run / "run.env")
    if (run_env.get("job_mode") != "eval" or
            run_env.get("variant") != variant or
            run_env.get("train_steps") != "0" or
            run_env.get("input_model") != str(checkpoint) or
            run_env.get("eval_expected_rows") != str(expected_rows) or
            run_env.get("eval_group_size") != "1" or
            run_env.get("tool_protocol") != "qwen35_native" or
            run_env.get("budget_rmb") != str(budget) or
            run_env.get("timed_out") != "false"):
        raise SystemExit("non-evaluation work appeared in A/R results")
    if ((run / "terminal").read_text(encoding="utf-8").strip() != "success" or
            (run / "exit-code").read_text(encoding="utf-8").strip() != "0"):
        raise SystemExit("A/R inner evaluation is not a complete success")
    def sha256(path):
        digest_obj = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest_obj.update(chunk)
        return digest_obj.hexdigest()
    if (row["resolved_config_sha256"] != sha256(run / "resolved-config.yaml") or
            row["trace_sha256"] != sha256(
                run / "traces" / "eval_predictions.jsonl") or
            row["trace_manifest_sha256"] != sha256(
                run / "traces" / "eval_predictions.manifest.json")):
        raise SystemExit("A/R lineage digest mismatch")
for key in ("val", "nq_test", "multihop"):
    paired = root / f"paired-ar-{key}"
    for name in ("summary.json", "summary.md", "paired_results.csv",
                 "correct_questions.csv", "wrong_questions.csv",
                 "search_transition.csv"):
        path = paired / name
        if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
            raise SystemExit(f"missing paired A/R evidence: {path}")
PY
}

# Publish the A/R package with its marker as the final commit point.  Unlike the
# historical generic publisher, this never exposes a canonical marker before
# every result pointer and checksum has been durably written and re-read.
publish_ar_training_evidence() {
    local outer_attempt="$1" results_dir="$2"
    shift 2
    local marker_dir marker evidence_digest path relative checksum_lines=''
    local actual_result_parent actual_marker_parent
    local -a evidence_files=(
        "$NATIVE_TRAIN_MANIFEST" "$NATIVE_TRAIN_MANIFEST.sha256"
        "$NATIVE_TRAIN_CATALOG"
        "$NATIVE_TRAIN_DATA_DIR/search_mix_answer_quality_exclusions.v1.json"
        "$NATIVE_TRAIN_DATA_DIR/train_512.parquet"
        "$NATIVE_TRAIN_DATA_DIR/val_128.parquet"
        "$NATIVE_TRAIN_DATA_DIR/nq_test_128_native_v4.parquet"
        "$NATIVE_TRAIN_DATA_DIR/multihop_eval_256_native_v4.parquet"
        "$NATIVE_TRAIN_G3_DATA" "$HANDOFF" "$HANDOFF.sha256"
        "$MANIFEST_DIR/cpu.ok" "$MANIFEST_DIR/git.ok"
        "$MANIFEST_DIR/checkout-tree.sha256" "$@"
    )
    local -a relative_files=()

    actual_result_parent="$(readlink -f -- "$(dirname -- "$results_dir")")" ||
        return 1
    [[ "$actual_result_parent" == "$(readlink -f -- "$NATIVE_TRAIN_RESULTS_ROOT/attempts")" &&
        "$(basename -- "$results_dir")" == "$(basename -- "$outer_attempt")" ]] ||
        return 1
    for path in "${evidence_files[@]}"; do
        relative_files+=("$(native_train_project_relative_file "$path")") ||
            return 1
    done
    while IFS= read -r relative; do
        path="$PROJECT_ROOT/$relative"
        checksum_lines+="$(file_sha256 "$path")  $relative"$'\n'
        sync_path "$path" || return $?
    done < <(printf '%s\n' "${relative_files[@]}" | LC_ALL=C sort -u)
    atomic_write "$results_dir/evidence.sha256" "$checksum_lines"
    verify_native_training_checksum_manifest "$results_dir/evidence.sha256" ||
        return $?
    evidence_digest="$(file_sha256 "$results_dir/evidence.sha256")" || return 1

    marker_dir="$MANIFEST_DIR/$AR_NAMESPACE"
    if [[ -e "$marker_dir" || -L "$marker_dir" ]]; then
        [[ -d "$marker_dir" && ! -L "$marker_dir" ]] || return 1
    else
        mkdir "$marker_dir" || return $?
    fi
    actual_marker_parent="$(readlink -f -- "$marker_dir")" || return 1
    [[ "$actual_marker_parent" == "$(readlink -f -- "$MANIFEST_DIR")/$AR_NAMESPACE" ]] ||
        return 1
    marker="$marker_dir/$(basename -- "$outer_attempt").ok"
    [[ ! -e "$marker" && ! -L "$marker" ]] || {
        printf 'Refusing to overwrite A/R evidence marker: %s\n' "$marker" >&2
        return 1
    }

    # Pointer writes are recoverable metadata until the final marker exists.
    atomic_write "$outer_attempt/result-contract" "$AR_CONTRACT"$'\n'
    atomic_write "$outer_attempt/result-root" "$results_dir"$'\n'
    atomic_write "$outer_attempt/evidence-marker" "$marker"$'\n'
    atomic_write "$outer_attempt/evidence-digest" "$evidence_digest"$'\n'
    [[ "$(ar_value "$outer_attempt/result-contract")" == "$AR_CONTRACT" &&
        "$(ar_value "$outer_attempt/result-root")" == "$results_dir" &&
        "$(ar_value "$outer_attempt/evidence-marker")" == "$marker" &&
        "$(ar_value "$outer_attempt/evidence-digest")" == "$evidence_digest" ]] ||
        return 1
    sync_path "$results_dir" || return $?
    sync_path "$outer_attempt" || return $?
    sync_path "$marker_dir" || return $?

    # This is the only scientific-completion commit point.
    atomic_write "$marker" "$evidence_digest"$'\n'
    sync_path "$marker_dir" || return $?
}

# This override is evaluation-only.  It does not call the inherited main
# pipeline, R60 training, G3, or either B/C branch.
qwen_native_main_pipeline() {
    local outer_attempt="$1" commit="$2" handoff_digest="$3"
    local base_model="$4" base_digest="$5" data_digest="$6"
    local results_dir runner_digest eval_key eval_stage rows artifact variant run rc
    local trace_identity trace_digest trace_manifest_digest config_digest
    local receipt_digest
    local lineage_rows='' index_content wandb_file wandb_count
    local -a eval_keys=(val nq_test multihop)
    local -a evidence_files=() wandb_runs=()
    local -A expected_rows=([val]=128 [nq_test]=128 [multihop]=256)
    local -A eval_artifacts=([val]=val [nq_test]=nq_test_eval [multihop]=multihop_eval)
    local -A a_runs=() r_runs=() a_trace=() r_trace=()
    local -A a_manifest=() r_manifest=() paired_dirs=()

    runner_digest="$(file_sha256 "$AR_RUNNER_PATH")" || return 1
    [[ "$AR_PREFLIGHT_COMMIT" == "$commit" &&
        "$AR_PREFLIGHT_HANDOFF" == "$handoff_digest" &&
        "$AR_PREFLIGHT_BASE_DIGEST" == "$base_digest" &&
        "$AR_PREFLIGHT_DATA_DIGEST" == "$data_digest" &&
        "$AR_PREFLIGHT_RUNNER_DIGEST" == "$runner_digest" ]] || {
        printf 'A/R CPU-to-GPU preflight identity drifted.\n' >&2
        return 1
    }
    verify_ar_cpu_receipt "$runner_digest" || return $?
    receipt_digest="$AR_CPU_RECEIPT_DIGEST"
    verify_native_checkpoint_digest "$base_model" "$base_digest" || return $?
    verify_native_checkpoint_digest "$BC_R60_CHECKPOINT" \
        "$BC_EXPECTED_R60_CHECKPOINT" || return $?
    results_dir="$(create_native_training_results_dir "$outer_attempt")" || return $?

    export EVAL_DATA_FILE='' EVAL_GROUP_SIZE=1
    for eval_key in "${eval_keys[@]}"; do
        rows="${expected_rows[$eval_key]}"
        artifact="${eval_artifacts[$eval_key]}"
        export EVAL_EXPECTED_ROWS="$rows"

        variant="qwen_native_a_$eval_key"
        run_ar_eval_job "$variant" "$base_model" "$base_digest" || {
            rc=$?
            return "$rc"
        }
        run="$LAST_RUN_DIR"
        a_runs[$eval_key]="$run"
        eval_stage="${eval_key^^}"
        eval_stage="${eval_stage//_/-}"
        append_native_train_index "$outer_attempt" \
            "A-$eval_stage-EVAL" "parent_${eval_key}_eval" "$run" || return $?
        trace_identity="$(verify_native_trace_identity \
            "$run/traces/eval_predictions.manifest.json" "$rows")" || return 1
        IFS=$'\t' read -r trace_digest trace_manifest_digest <<<"$trace_identity"
        a_trace[$eval_key]="$trace_digest"
        a_manifest[$eval_key]="$trace_manifest_digest"

        variant="qwen_native_r_$eval_key"
        run_ar_eval_job "$variant" "$BC_R60_CHECKPOINT" \
            "$BC_EXPECTED_R60_CHECKPOINT" || {
            rc=$?
            return "$rc"
        }
        run="$LAST_RUN_DIR"
        r_runs[$eval_key]="$run"
        append_native_train_index "$outer_attempt" \
            "R-$eval_stage-EVAL" "reproduced_${eval_key}_eval" "$run" || return $?
        trace_identity="$(verify_native_trace_identity \
            "$run/traces/eval_predictions.manifest.json" "$rows")" || return 1
        IFS=$'\t' read -r trace_digest trace_manifest_digest <<<"$trace_identity"
        r_trace[$eval_key]="$trace_digest"
        r_manifest[$eval_key]="$trace_manifest_digest"

        paired_dirs[$eval_key]="$results_dir/paired-ar-$eval_key"
        "$TRAIN_ENV/bin/python" "$NATIVE_TRAIN_PAIRED_CLI" \
            --parent "${a_runs[$eval_key]}/traces/eval_predictions.jsonl" \
            --reproduced "${r_runs[$eval_key]}/traces/eval_predictions.jsonl" \
            --data-manifest "$NATIVE_TRAIN_MANIFEST" \
            --eval-artifact "$artifact" \
            --expected-parent-checkpoint-digest "$base_digest" \
            --expected-reproduced-checkpoint-digest "$BC_EXPECTED_R60_CHECKPOINT" \
            --output-dir "${paired_dirs[$eval_key]}" \
            --expected-rows "$rows" || return $?
    done

    atomic_write "$results_dir/contract.env" \
        "schema=$AR_CONTRACT"$'\n'\
"stage=ar_eval_only"$'\n'\
"stage_order=$AR_STAGE_ORDER"$'\n'\
"training_executed=false"$'\n'\
"checkout_commit=$commit"$'\n'\
"cpu_handoff_sha256=$handoff_digest"$'\n'\
"base_model=$base_model"$'\n'\
"base_model_sha256=$base_digest"$'\n'\
"r60_evidence=$BC_EXPECTED_R60_MARKER"$'\n'\
"r60_evidence_sha256=$BC_EXPECTED_R60_EVIDENCE"$'\n'\
"r60_checkpoint=$BC_R60_CHECKPOINT"$'\n'\
"r60_checkpoint_sha256=$BC_EXPECTED_R60_CHECKPOINT"$'\n'\
"bc_recovery_marker=$AR_BC_MARKER"$'\n'\
"bc_recovery_evidence_sha256=$AR_EXPECTED_BC_EVIDENCE"$'\n'\
"cpu_receipt=$AR_CPU_RECEIPT"$'\n'\
"cpu_receipt_sha256=$receipt_digest"$'\n'\
"runner_sha256=$runner_digest"$'\n'\
"eval_group_size=1"$'\n'\
"decoding=greedy"$'\n'\
"seed=42"$'\n'

    atomic_write "$results_dir/lineage.tsv" \
        $'stage\trole\trun_dir\tcheckpoint\tcheckpoint_digest\tparent_checkpoint\tparent_checkpoint_digest\tcheckout_commit\tcpu_handoff_digest\tdata_manifest_sha256\tresolved_config_sha256\ttrace_sha256\ttrace_manifest_sha256\trun_contract_sha256\tpredecessor_evidence_sha256\n'
    for eval_key in "${eval_keys[@]}"; do
        rows="${expected_rows[$eval_key]}"
        eval_stage="${eval_key^^}"
        eval_stage="${eval_stage//_/-}"
        run="${a_runs[$eval_key]}"
        config_digest="$(file_sha256 "$run/resolved-config.yaml")" || return 1
        lineage_rows+="A-$eval_stage-EVAL"$'\t'"parent_${eval_key}_eval"$'\t'"$run"$'\t'"$base_model"$'\t'"$base_digest"$'\t'"$base_model"$'\t'"$base_digest"$'\t'"$commit"$'\t'"$handoff_digest"$'\t'"$data_digest"$'\t'"$config_digest"$'\t'"${a_trace[$eval_key]}"$'\t'"${a_manifest[$eval_key]}"$'\t-\t'"$receipt_digest"$'\n'
        run="${r_runs[$eval_key]}"
        config_digest="$(file_sha256 "$run/resolved-config.yaml")" || return 1
        lineage_rows+="R-$eval_stage-EVAL"$'\t'"reproduced_${eval_key}_eval"$'\t'"$run"$'\t'"$BC_R60_CHECKPOINT"$'\t'"$BC_EXPECTED_R60_CHECKPOINT"$'\t'"$BC_R60_CHECKPOINT"$'\t'"$BC_EXPECTED_R60_CHECKPOINT"$'\t'"$commit"$'\t'"$handoff_digest"$'\t'"$data_digest"$'\t'"$config_digest"$'\t'"${r_trace[$eval_key]}"$'\t'"${r_manifest[$eval_key]}"$'\t-\t'"$BC_EXPECTED_R60_EVIDENCE"$'\n'
    done
    printf '%s' "$lineage_rows" >>"$results_dir/lineage.tsv"
    index_content="$(cat "$outer_attempt/native-training-runs.tsv")"
    atomic_write "$results_dir/run-index.tsv" "$index_content"$'\n'

    for eval_key in "${eval_keys[@]}"; do
        wandb_runs+=("${a_runs[$eval_key]}" "${r_runs[$eval_key]}")
        for run in "${a_runs[$eval_key]}" "${r_runs[$eval_key]}"; do
            write_wandb_receipt "$run" 0 || return $?
            verify_wandb_receipt "$run" 0 || return $?
            evidence_files+=(
                "$run/train.log" "$run/resolved-config.yaml" "$run/run.env"
                "$run/terminal" "$run/exit-code" "$run/.success"
                "$run/traces/eval_predictions.jsonl"
                "$run/traces/eval_predictions.manifest.json"
                "$run/traces/eval_predictions.manifest.json.sha256"
                "$run/wandb-receipt.json"
            )
            wandb_count=0
            while IFS= read -r -d '' wandb_file; do
                evidence_files+=("$wandb_file")
                ((wandb_count += 1))
            done < <(find "$run/wandb" -type f -size +0c -print0 | LC_ALL=C sort -z)
            ((wandb_count > 0)) || {
                printf 'Run-specific WandB history is empty: %s\n' "$run" >&2
                return 1
            }
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

    evidence_files+=(
        "$results_dir/contract.env" "$results_dir/lineage.tsv"
        "$results_dir/run-index.tsv"
        "$AR_CPU_RECEIPT" "$AR_CPU_RECEIPT.sha256"
        "$AR_RUNNER_PATH" "$AR_SOURCE_ENTRY" "$NATIVE_TRAIN_PAIRED_CLI"
        "$BC_EXPECTED_R60_MARKER" "$BC_R60_RESULTS/evidence.sha256"
        "$BC_CPU_RECEIPT" "$BC_CPU_RECEIPT.sha256"
        "$AR_G3_RECEIPT" "$AR_G3_RECEIPT.sha256"
        "$BC_G3_TRACE" "$BC_G3_TRACE_MANIFEST" "$BC_G3_TRACE_SIDECAR"
        "$BC_G3_ANALYSIS/summary.json" "$BC_G3_ANALYSIS/go_no_go.json"
        "$AR_BC_MARKER" "$AR_BC_EVIDENCE"
        "$(ar_config_file qwen_native_a_val)"
        "$(ar_config_file qwen_native_r_val)"
        "$(ar_config_file qwen_native_a_nq_test)"
        "$(ar_config_file qwen_native_r_nq_test)"
        "$(ar_config_file qwen_native_a_multihop)"
        "$(ar_config_file qwen_native_r_multihop)"
    )
    sync_path "$results_dir"

    verify_ar_results_semantics "$results_dir" || return $?
    verify_checkout "$commit" || return $?
    verify_ar_foundation "$commit" "$handoff_digest" "$base_model" \
        "$base_digest" || return $?
    [[ "$(file_sha256 "$AR_RUNNER_PATH")" == "$runner_digest" ]] || return 1
    verify_ar_cpu_receipt "$runner_digest" || return $?
    verify_native_checkpoint_digest "$base_model" "$base_digest" || return $?
    verify_native_checkpoint_digest "$BC_R60_CHECKPOINT" \
        "$BC_EXPECTED_R60_CHECKPOINT" || return $?

    publish_ar_training_evidence "$outer_attempt" "$results_dir" \
        "${evidence_files[@]}" || return $?
    printf 'Completed and sealed A/R evaluation-only comparison: %s\n' "$results_dir"
    return 0
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    case "${1:-}" in
        --cpu-prepare)
            [[ "$#" == 1 ]] || exit 64
            prepare_ar_cpu_receipt
            ;;
        *)
            qwen_native_train_main "$@"
            ;;
    esac
fi
