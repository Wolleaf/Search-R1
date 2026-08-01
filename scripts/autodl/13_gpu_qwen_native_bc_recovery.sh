#!/usr/bin/env bash
set -Eeuo pipefail

# Recover the interrupted post-hoc B/C experiment without retraining the
# already sealed B20 branch.  This operator deliberately sources the deployed
# B/C runner so all historical R60/G3/foundation checks remain anchored to the
# frozen canonical checkout.
RC_PROJECT_ROOT="${AUTODL_ROOT:-/root/autodl-tmp/search-r1}"
RC_SOURCE_ENTRY="${AUTODL_BC_SOURCE_ENTRY:-$RC_PROJECT_ROOT/operator/12_gpu_qwen_native_bc_only.sh}"
RC_RUNNER_PATH="$(readlink -f -- "${BASH_SOURCE[0]}")"
export QWEN_NATIVE_TRAIN_STAGE=main

[[ -f "$RC_SOURCE_ENTRY" && ! -L "$RC_SOURCE_ENTRY" ]] || {
    printf 'Missing deployed B/C foundation runner: %s\n' "$RC_SOURCE_ENTRY" >&2
    exit 1
}

# shellcheck source=12_gpu_qwen_native_bc_only.sh
source "$RC_SOURCE_ENTRY"

readonly RC_CONTRACT=qwen-native-training-bc-recovery-v1
readonly RC_NAMESPACE=qwen-native-training-bc-recovery
readonly RC_CPU_CONTRACT=qwen-native-bc-recovery-cpu-v1
readonly RC_TERMINAL_CODE=203
readonly RC_DEPLOY_PATH="$RC_PROJECT_ROOT/operator/13_gpu_qwen_native_bc_recovery.sh"
readonly RC_CPU_ROOT="$MANIFEST_DIR/qwen-native-bc-recovery-cpu"
readonly RC_RECOVERY_COMMIT="${AUTODL_RECOVERY_COMMIT:-}"
readonly RC_CHECKOUT="$RC_PROJECT_ROOT/recovery-checkouts/$RC_RECOVERY_COMMIT"
readonly RC_RECOVERY_BASE_COMMIT=9fc2e3a4245d9b9a09ae43b41a9fa268b4b3d68e

readonly RC_SOURCE_OUTER_ID=20260731T060828Z-1554-11705
readonly RC_SOURCE_OUTER="$ATTEMPTS_ROOT/gpu/$RC_SOURCE_OUTER_ID"
readonly RC_SOURCE_PHASE_SHA256=3a6161c1128cc9dcbcb2a371c89750db08bbe67a164862e9f7f337ad6e4eee32
readonly RC_SOURCE_INDEX_SHA256=a4951f051dca7cb6837c3d54214ba44ad653ff98bf362aa8c2d8a275faaa4449

readonly RC_B_RUN_ID=20260731T061302Z-1592-7937
readonly RC_B_RUN="$RUNS_ROOT/control/attempts/$RC_B_RUN_ID"
readonly RC_B_CHECKPOINT="$RC_B_RUN/checkpoints/actor/global_step_20"
readonly RC_B_CHECKPOINT_SHA256=8df3d6ab13e154b4a2360b1535f763731f955baebba21b75f9f6e017000ad6d2
readonly RC_B_CONFIG_SHA256=ea52724698d2b058c30523b16292aec88b8307550372fb277234b98d0ab366ea
readonly RC_B_CONTRACT_SHA256=a3720c240e9f17848f0bd97cb056a3e1f684a3242094a6f8096dbb1e4accd835
readonly RC_B_LINEAGE_SHA256=fb5e264d4e0e48d34d5627b39306c1de438470d0bc6f8aaabf4286792396c467
readonly RC_B_RUN_ENV_SHA256=42ee948513b894509928ddca75aeee843e9efedfe97e9d10536c1057869ac744
readonly RC_B_LOG_SHA256=6769b26dacd335c910c441d62a791babf94a3c05063034341bfcf7dda72e2491
readonly RC_B_TRACE_SHA256=b1907e579ecece4fe4c102d229cc3103b8ee71348bbd8c162bfe320a04a51cf5
readonly RC_B_TRACE_MANIFEST_SHA256=ce72545ead4e49da9eb044467950be293e4d51635f303f3b9445efa4ccb29837
readonly RC_B_TRACE_SIDECAR_SHA256=2a71d1b498922560a4144a84bb33763bdd32638afda1c6cee1000888d65b3406
readonly RC_B_WANDB_TREE_SHA256=ed46a0e8abafdf87d8343c74eeaa572031295aa867aa850857d033a0be0588ec

readonly RC_OLD_C_RUN_ID=20260731T110656Z-1592-183
readonly RC_OLD_C_RUN="$RUNS_ROOT/cost_aware_gated/attempts/$RC_OLD_C_RUN_ID"
readonly RC_OLD_C_CONFIG_SHA256=d911886d50253dacb925b9726ffda938aaef1cc9ae477f9a06e13e63b3ff5785
readonly RC_OLD_C_RUN_ENV_SHA256=d85cb88745a85c5ed998234dd3b10264d54e98a2b23331c60e336355b68ba02c
readonly RC_OLD_C_LOG_SHA256=3d80fdf1fe4ba56e44b4e2c0d772f9cf944b5556d2b570e63288f18309e4b5a1
readonly RC_OLD_C_TRACE="$RC_OLD_C_RUN/traces/train_trajectories.jsonl.partial"
readonly RC_OLD_C_TRACE_SHA256=02b2dd53874b24a59ce252a670363af7865368ab5340bbadb05f0ed9727bb4ae

readonly RC_C_BUDGET_RMB=40
readonly RC_MIN_C_TIMEOUT_SECONDS=21948

RC_CPU_RECEIPT=''
RC_CPU_RECEIPT_DIGEST=''
RC_CPU_B_WANDB_RECEIPT=''
RC_CPU_B_WANDB_RECEIPT_DIGEST=''
RC_CODE_TRAIN_SHA256=''
RC_CODE_TRAINER_SHA256=''
RC_CODE_RUNNER_SHA256=''
RC_MOUNT_IDENTITY_SHA256=''
RC_PREFLIGHT_COMMIT=''
RC_PREFLIGHT_HANDOFF=''
RC_PREFLIGHT_BASE_DIGEST=''
RC_PREFLIGHT_DATA_DIGEST=''
RC_PREFLIGHT_RUNNER_DIGEST=''

rc_value() {
    local path="$1"
    [[ -f "$path" && ! -L "$path" ]] || return 1
    tr -d '\r\n' <"$path"
}

rc_require_digest() {
    local path="$1" expected="$2"
    [[ -f "$path" && ! -L "$path" &&
        "$(file_sha256 "$path")" == "$expected" ]] || {
        printf 'Recovery predecessor file changed: %s\n' "$path" >&2
        return 1
    }
}

rc_mount_identity_sha256() {
    {
        printf 'host=%s\n' "$(hostname)"
        findmnt -n -o SOURCE,FSTYPE,TARGET -T "$RC_PROJECT_ROOT"
    } | sha256sum | cut -d' ' -f1
}

verify_recovery_checkout() {
    local actual status changed expected
    [[ "$RC_RECOVERY_COMMIT" =~ ^[0-9a-f]{40}$ &&
        "$RC_RUNNER_PATH" == "$RC_DEPLOY_PATH" &&
        -d "$RC_CHECKOUT" && ! -L "$RC_CHECKOUT" ]] || {
        printf 'Set AUTODL_RECOVERY_COMMIT to the exact deployed recovery commit.\n' >&2
        return 64
    }
    [[ "$(readlink -f -- "$RC_CHECKOUT")" == "$RC_CHECKOUT" ]] || return 1
    actual="$(git -C "$RC_CHECKOUT" rev-parse HEAD)" || return 1
    [[ "$actual" == "$RC_RECOVERY_COMMIT" ]] || return 1
    if git -C "$RC_CHECKOUT" symbolic-ref -q HEAD >/dev/null 2>&1; then
        printf 'Recovery checkout must be detached.\n' >&2
        return 1
    fi
    status="$(git -C "$RC_CHECKOUT" status --porcelain=v1 --untracked-files=all)" || return 1
    [[ -z "$status" ]] || {
        printf 'Recovery checkout is dirty.\n' >&2
        return 1
    }
    [[ "$(git -C "$RC_CHECKOUT" rev-parse "$RC_RECOVERY_COMMIT^")" == \
        "$RC_RECOVERY_BASE_COMMIT" ]] || {
        printf 'Recovery commit is not the registered one-commit patch.\n' >&2
        return 1
    }
    changed="$(git -C "$RC_CHECKOUT" diff --name-only \
        "$RC_RECOVERY_BASE_COMMIT..$RC_RECOVERY_COMMIT" | LC_ALL=C sort)" || return 1
    expected="$(printf '%s\n' \
        scripts/autodl/03_gpu_run.sh \
        scripts/autodl/13_gpu_qwen_native_bc_recovery.sh \
        scripts/autodl/tests/test_gated_config.sh \
        scripts/autodl/train_small_grpo.sh \
        verl/trainer/ppo/ray_trainer.py | LC_ALL=C sort)"
    [[ "$changed" == "$expected" ]] || {
        printf 'Recovery commit contains files outside the registered allowlist.\n' >&2
        printf '%s\n' "$changed" >&2
        return 1
    }
    cmp -s "$RC_RUNNER_PATH" \
        "$RC_CHECKOUT/scripts/autodl/13_gpu_qwen_native_bc_recovery.sh" || {
        printf 'Deployed recovery runner differs from its pinned checkout.\n' >&2
        return 1
    }
    RC_CODE_TRAIN_SHA256="$(file_sha256 \
        "$RC_CHECKOUT/scripts/autodl/train_small_grpo.sh")" || return 1
    RC_CODE_TRAINER_SHA256="$(file_sha256 \
        "$RC_CHECKOUT/verl/trainer/ppo/ray_trainer.py")" || return 1
    RC_CODE_RUNNER_SHA256="$(file_sha256 "$RC_RUNNER_PATH")" || return 1
    RC_MOUNT_IDENTITY_SHA256="$(rc_mount_identity_sha256)" || return 1
    (
        cd "$RC_CHECKOUT"
        PYTHONPATH="$RC_CHECKOUT${PYTHONPATH:+:$PYTHONPATH}" \
            "$TRAIN_ENV/bin/python" - "$RC_CHECKOUT" <<'PY'
from importlib.util import find_spec
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
spec = find_spec("verl.trainer.ppo.ray_trainer")
if spec is None or spec.origin is None:
    raise SystemExit("cannot resolve recovery ray_trainer")
origin = Path(spec.origin).resolve()
expected = root / "verl" / "trainer" / "ppo" / "ray_trainer.py"
if origin != expected:
    raise SystemExit(f"recovery Python import escaped checkout: {origin}")
PY
    ) || return $?
}

verify_recovery_source_failure() {
    [[ -d "$RC_SOURCE_OUTER" && ! -L "$RC_SOURCE_OUTER" &&
        -f "$RC_SOURCE_OUTER/.failed" && ! -L "$RC_SOURCE_OUTER/.failed" &&
        ! -e "$RC_SOURCE_OUTER/.success" &&
        ! -e "$RC_SOURCE_OUTER/.running" &&
        ! -e "$RC_SOURCE_OUTER/.starting" &&
        "$(rc_value "$RC_SOURCE_OUTER/terminal")" == failed &&
        "$(rc_value "$RC_SOURCE_OUTER/exit-code")" == 124 ]] || {
        printf 'Historical B/C outer state changed.\n' >&2
        return 1
    }
    rc_require_digest "$RC_SOURCE_OUTER/phase.log" \
        "$RC_SOURCE_PHASE_SHA256" || return $?
    rc_require_digest "$RC_SOURCE_OUTER/native-training-runs.tsv" \
        "$RC_SOURCE_INDEX_SHA256" || return $?
    [[ -d "$RC_OLD_C_RUN" && ! -L "$RC_OLD_C_RUN" &&
        -f "$RC_OLD_C_RUN/.failed" && ! -L "$RC_OLD_C_RUN/.failed" &&
        ! -e "$RC_OLD_C_RUN/.success" && ! -e "$RC_OLD_C_RUN/.running" &&
        "$(rc_value "$RC_OLD_C_RUN/terminal")" == failed &&
        "$(rc_value "$RC_OLD_C_RUN/exit-code")" == 124 &&
        ! -d "$RC_OLD_C_RUN/checkpoints/actor/global_step_20" ]] || {
        printf 'Historical incomplete C run changed or became adoptable.\n' >&2
        return 1
    }
    rc_require_digest "$RC_OLD_C_RUN/resolved-config.yaml" \
        "$RC_OLD_C_CONFIG_SHA256" || return $?
    rc_require_digest "$RC_OLD_C_RUN/run.env" "$RC_OLD_C_RUN_ENV_SHA256" || return $?
    rc_require_digest "$RC_OLD_C_RUN/train.log" "$RC_OLD_C_LOG_SHA256" || return $?
    rc_require_digest "$RC_OLD_C_TRACE" "$RC_OLD_C_TRACE_SHA256" || return $?
    [[ ! -e "$RC_OLD_C_RUN/traces/train_trajectories.manifest.json" ]] || return 1
}

verify_reused_b() {
    local identity trace_digest manifest_digest
    [[ -d "$RC_B_RUN" && ! -L "$RC_B_RUN" &&
        -f "$RC_B_RUN/.success" && ! -L "$RC_B_RUN/.success" &&
        ! -e "$RC_B_RUN/.failed" && ! -e "$RC_B_RUN/.running" &&
        "$(rc_value "$RC_B_RUN/terminal")" == success &&
        "$(rc_value "$RC_B_RUN/exit-code")" == 0 ]] || {
        printf 'Reusable B20 is no longer a sealed successful inner run.\n' >&2
        return 1
    }
    rc_require_digest "$RC_B_RUN/resolved-config.yaml" "$RC_B_CONFIG_SHA256" || return $?
    rc_require_digest "$RC_B_RUN/native-training-contract.json" \
        "$RC_B_CONTRACT_SHA256" || return $?
    rc_require_digest "$RC_B_RUN/lineage.tsv" "$RC_B_LINEAGE_SHA256" || return $?
    rc_require_digest "$RC_B_RUN/run.env" "$RC_B_RUN_ENV_SHA256" || return $?
    rc_require_digest "$RC_B_RUN/train.log" "$RC_B_LOG_SHA256" || return $?
    rc_require_digest "$RC_B_RUN/traces/train_trajectories.jsonl" \
        "$RC_B_TRACE_SHA256" || return $?
    rc_require_digest "$RC_B_RUN/traces/train_trajectories.manifest.json" \
        "$RC_B_TRACE_MANIFEST_SHA256" || return $?
    rc_require_digest "$RC_B_RUN/traces/train_trajectories.manifest.json.sha256" \
        "$RC_B_TRACE_SIDECAR_SHA256" || return $?
    verify_native_checkpoint_digest "$RC_B_CHECKPOINT" \
        "$RC_B_CHECKPOINT_SHA256" || return $?
    identity="$(verify_native_trace_identity \
        "$RC_B_RUN/traces/train_trajectories.manifest.json" 800)" || return 1
    IFS=$'\t' read -r trace_digest manifest_digest <<<"$identity"
    [[ "$trace_digest" == "$RC_B_TRACE_SHA256" &&
        "$manifest_digest" == "$RC_B_TRACE_MANIFEST_SHA256" ]] || return 1
    if ! "$TRAIN_ENV/bin/python" - "$NATIVE_TRAIN_WANDB_CLI" \
        "$RC_B_RUN/wandb" "$RC_B_RUN/train.log" \
        "$RC_B_WANDB_TREE_SHA256" <<'PY'
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

cli, wandb_dir, log, expected_tree = sys.argv[1:]
spec = spec_from_file_location("search_r1_wandb_history", cli)
if spec is None or spec.loader is None:
    raise SystemExit("cannot load WandB history verifier")
module = module_from_spec(spec)
spec.loader.exec_module(module)
receipt = module.build_receipt(
    Path(wandb_dir), log=Path(log), expected_steps=range(1, 21)
)
if receipt.get("decision") != "GO":
    raise SystemExit("reusable B20 WandB receipt is not GO")
if receipt.get("inputs", {}).get("wandb_tree_sha256") != expected_tree:
    raise SystemExit("reusable B20 WandB tree changed")
PY
    then
        return 1
    fi
    grep -Fq "$BC_R60_CHECKPOINT"$'\t'"$BC_EXPECTED_R60_CHECKPOINT" \
        "$RC_B_RUN/lineage.tsv" || return 1
}

rc_build_b_wandb_receipt() {
    local output="$1" decision step
    local -a args=(
        --wandb-dir "$RC_B_RUN/wandb"
        --log "$RC_B_RUN/train.log"
        --output "$output"
    )
    for ((step = 1; step <= BRANCH_STEPS; step += 1)); do
        args+=(--expected-step "$step")
    done
    decision="$("$TRAIN_ENV/bin/python" "$NATIVE_TRAIN_WANDB_CLI" \
        "${args[@]}")" || return $?
    [[ "$decision" == GO ]] || return 1
    args=(
        --wandb-dir "$RC_B_RUN/wandb"
        --log "$RC_B_RUN/train.log"
        --verify-receipt "$output"
    )
    for ((step = 1; step <= BRANCH_STEPS; step += 1)); do
        args+=(--expected-step "$step")
    done
    [[ "$("$TRAIN_ENV/bin/python" "$NATIVE_TRAIN_WANDB_CLI" \
        "${args[@]}")" == GO ]]
}

rc_receipt_content() {
    local runner_digest="$1" b_wandb_digest="$2"
    printf '%s\n' \
        "schema=$RC_CPU_CONTRACT" \
        'experiment_class=post_hoc_exploratory_recovery' \
        'cloned_volume_revalidated=true' \
        "mount_identity_sha256=$RC_MOUNT_IDENTITY_SHA256" \
        "canonical_checkout_commit=$BC_EXPECTED_CHECKOUT" \
        "recovery_base_commit=$RC_RECOVERY_BASE_COMMIT" \
        "recovery_checkout_commit=$RC_RECOVERY_COMMIT" \
        "cpu_handoff_sha256=$BC_EXPECTED_HANDOFF" \
        "base_model_sha256=$BC_EXPECTED_BASE" \
        "data_manifest_sha256=$BC_EXPECTED_DATA" \
        "r60_checkpoint=$BC_R60_CHECKPOINT" \
        "r60_checkpoint_sha256=$BC_EXPECTED_R60_CHECKPOINT" \
        "source_outer=$RC_SOURCE_OUTER" \
        'source_outer_terminal=failed' \
        'source_outer_exit_code=124' \
        'source_outer_adopted=false' \
        "reused_b_run=$RC_B_RUN" \
        "reused_b_checkpoint_sha256=$RC_B_CHECKPOINT_SHA256" \
        "reused_b_trace_sha256=$RC_B_TRACE_SHA256" \
        "reused_b_wandb_receipt_sha256=$b_wandb_digest" \
        "ignored_failed_c_run=$RC_OLD_C_RUN" \
        "ignored_failed_c_trace_sha256=$RC_OLD_C_TRACE_SHA256" \
        'old_c_adopted=false' \
        "recovery_train_script_sha256=$RC_CODE_TRAIN_SHA256" \
        "recovery_trainer_sha256=$RC_CODE_TRAINER_SHA256" \
        "runner=$RC_DEPLOY_PATH" \
        "runner_sha256=$runner_digest" \
        'gpu_training_authorized=false' \
        'manual_gpu_start_required=true'
}

verify_recovery_cpu_receipt() {
    local runner_digest="$1" directory receipt b_receipt expected digest sidecar b_digest
    verify_recovery_checkout || return $?
    directory="$RC_CPU_ROOT/$runner_digest"
    receipt="$directory/receipt.env"
    b_receipt="$directory/b20-reused-wandb-receipt.json"
    [[ -d "$directory" && ! -L "$directory" &&
        -f "$receipt" && ! -L "$receipt" &&
        -f "$b_receipt" && ! -L "$b_receipt" &&
        -f "$directory/.success" && ! -L "$directory/.success" &&
        ! -e "$directory/.failed" &&
        "$(rc_value "$directory/terminal")" == success &&
        "$(rc_value "$directory/exit-code")" == 0 ]] || return 1
    b_digest="$(file_sha256 "$b_receipt")" || return 1
    expected="$(rc_receipt_content "$runner_digest" "$b_digest")"$'\n'
    cmp -s <(printf '%s' "$expected") "$receipt" || return 1
    digest="$(file_sha256 "$receipt")" || return 1
    sidecar="$(rc_value "$receipt.sha256")" || return 1
    [[ "$sidecar" == "$digest  ${receipt#"$RC_PROJECT_ROOT/"}" ]] || return 1
    rc_build_b_wandb_receipt_verify "$b_receipt" || return $?
    RC_CPU_RECEIPT="$receipt"
    RC_CPU_RECEIPT_DIGEST="$digest"
    RC_CPU_B_WANDB_RECEIPT="$b_receipt"
    RC_CPU_B_WANDB_RECEIPT_DIGEST="$b_digest"
}

rc_build_b_wandb_receipt_verify() {
    local receipt="$1" decision step
    local -a args=(
        --wandb-dir "$RC_B_RUN/wandb"
        --log "$RC_B_RUN/train.log"
        --verify-receipt "$receipt"
    )
    for ((step = 1; step <= BRANCH_STEPS; step += 1)); do
        args+=(--expected-step "$step")
    done
    decision="$("$TRAIN_ENV/bin/python" "$NATIVE_TRAIN_WANDB_CLI" \
        "${args[@]}")" || return $?
    [[ "$decision" == GO ]]
}

prepare_recovery_cpu_receipt() {
    local commit handoff_digest base_model base_digest old_runner_digest runner_digest
    local directory receipt b_receipt content digest
    [[ "$RC_RUNNER_PATH" == "$RC_DEPLOY_PATH" ]] || return 1
    export CUDA_VISIBLE_DEVICES=''
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
    export WANDB_MODE=offline PIP_NO_INDEX=1 PYTHONDONTWRITEBYTECODE=1
    commit="$(expected_commit)" || return 1
    handoff_digest="$(rc_value "$MANIFEST_DIR/cpu.ok")" || return 1
    base_model="$(readlink -f -- "$MODEL_DIR")" || return 1
    base_digest="$(tree_sha256 "$base_model")" || return 1
    verify_bc_foundation "$commit" "$handoff_digest" "$base_model" \
        "$base_digest" || return $?
    old_runner_digest="$(file_sha256 "$BC_RUNNER_PATH")" || return 1
    verify_bc_cpu_receipt "$old_runner_digest" || return $?
    verify_recovery_source_failure || return $?
    verify_reused_b || return $?
    verify_recovery_checkout || return $?
    runner_digest="$RC_CODE_RUNNER_SHA256"
    directory="$RC_CPU_ROOT/$runner_digest"
    receipt="$directory/receipt.env"
    b_receipt="$directory/b20-reused-wandb-receipt.json"
    mkdir -p "$RC_CPU_ROOT"
    if [[ -e "$directory" || -L "$directory" ]]; then
        verify_recovery_cpu_receipt "$runner_digest" || {
            printf 'Refusing a differing or incomplete recovery CPU receipt.\n' >&2
            return 1
        }
    else
        mkdir "$directory"
        rc_build_b_wandb_receipt "$b_receipt" || return $?
        content="$(rc_receipt_content "$runner_digest" \
            "$(file_sha256 "$b_receipt")")"$'\n'
        atomic_write "$receipt" "$content"
        digest="$(file_sha256 "$receipt")" || return 1
        atomic_write "$receipt.sha256" \
            "$digest  ${receipt#"$RC_PROJECT_ROOT/"}"$'\n'
        atomic_write "$directory/exit-code" $'0\n'
        atomic_write "$directory/terminal" $'success\n'
        atomic_write "$directory/.success" ''
        sync_path "$directory"
    fi
    verify_recovery_cpu_receipt "$runner_digest" || return $?
    printf 'B/C recovery CPU preparation complete: %s\n' "$RC_CPU_RECEIPT"
    printf 'GPU training was not started.\n'
}

verify_recovery_publish_identity() {
    local commit="$1" data_digest="$2" runner_digest="$3" old_runner_digest
    verify_checkout "$commit" || return $?
    [[ "$commit" == "$BC_EXPECTED_CHECKOUT" &&
        "$(file_sha256 "$NATIVE_TRAIN_MANIFEST")" == "$data_digest" ]] || return 1
    old_runner_digest="$(file_sha256 "$BC_RUNNER_PATH")" || return 1
    verify_bc_cpu_receipt "$old_runner_digest" || return $?
    verify_recovery_source_failure || return $?
    verify_reused_b || return $?
    verify_recovery_cpu_receipt "$runner_digest" || return $?
}

# Override the B/C-only admission.  The outer launcher and retriever stay in the
# sealed canonical checkout; only the new C20 process imports recovery code.
require_qwen_native_train() {
    validate_gpu_inputs || return $?
    [[ "$GPU_COUNT" == 2 && "$TRAIN_BATCH_SIZE" == 8 &&
        "$MAX_RESPONSE_LENGTH" == 500 && "$TOOL_PROTOCOL" == qwen35_native &&
        "$DATA_DIR" == "$NATIVE_TRAIN_DATA_DIR" &&
        "$EVAL_GROUP_SIZE" == 1 && -z "$EVAL_DATA_FILE" &&
        "$RUN_BUDGET_PROFILE" == gated_followup &&
        "$RC_RUNNER_PATH" == "$RC_DEPLOY_PATH" &&
        "$RC_RECOVERY_COMMIT" =~ ^[0-9a-f]{40}$ &&
        "${QWEN_NATIVE_PROTOCOL_GATE_EVIDENCE:-}" == \
            "$MANIFEST_DIR/qwen-native-gate/20260728T044634Z-2051-4045.ok" &&
        "${QWEN_NATIVE_SMOKE_EVIDENCE:-}" == \
            "$MANIFEST_DIR/qwen-native-training-smoke/20260728T061246Z-1316-15067.ok" &&
        "${QWEN_NATIVE_R60_EVIDENCE:-}" == "$BC_EXPECTED_R60_MARKER" ]] || {
        printf 'Recovery requires the exact native-v4 markers and pinned recovery commit.\n' >&2
        return 64
    }
}

qwen_native_train_preflight() {
    local commit="$1" handoff_digest="$2" base_digest="$3"
    local base_model data_digest runner_digest old_runner_digest
    base_model="$(readlink -f -- "$MODEL_DIR")" || return 1
    verify_bc_foundation "$commit" "$handoff_digest" "$base_model" \
        "$base_digest" || return $?
    old_runner_digest="$(file_sha256 "$BC_RUNNER_PATH")" || return 1
    verify_bc_cpu_receipt "$old_runner_digest" || return $?
    verify_recovery_source_failure || return $?
    verify_reused_b || return $?
    verify_recovery_checkout || return $?
    runner_digest="$RC_CODE_RUNNER_SHA256"
    verify_recovery_cpu_receipt "$runner_digest" || {
        printf 'Run --cpu-prepare with this exact recovery runner first.\n' >&2
        return 1
    }
    data_digest="$(file_sha256 "$NATIVE_TRAIN_MANIFEST")" || return 1
    RC_PREFLIGHT_COMMIT="$commit"
    RC_PREFLIGHT_HANDOFF="$handoff_digest"
    RC_PREFLIGHT_BASE_DIGEST="$base_digest"
    RC_PREFLIGHT_DATA_DIGEST="$data_digest"
    RC_PREFLIGHT_RUNNER_DIGEST="$runner_digest"
    NATIVE_TRAIN_PREFLIGHT_STAGE="$NATIVE_TRAIN_STAGE"
    NATIVE_TRAIN_PREFLIGHT_COMMIT="$commit"
    NATIVE_TRAIN_PREFLIGHT_HANDOFF="$handoff_digest"
    NATIVE_TRAIN_PREFLIGHT_BASE_DIGEST="$base_digest"
    NATIVE_TRAIN_PREFLIGHT_DATA_DIGEST="$data_digest"
    NATIVE_TRAIN_PREFLIGHT_PROTOCOL_GATE_DIGEST="$NATIVE_TRAIN_PROTOCOL_GATE_DIGEST"
    NATIVE_TRAIN_PREFLIGHT_SMOKE_DIGEST="$NATIVE_TRAIN_SMOKE_DIGEST"
}

verify_recovery_config_delta() {
    local baseline="$1" recovery="$2"
    "$TRAIN_ENV/bin/python" - "$baseline" "$recovery" <<'PY'
from pathlib import Path
import sys
from omegaconf import OmegaConf

before = OmegaConf.to_container(OmegaConf.load(Path(sys.argv[1])), resolve=True)
after = OmegaConf.to_container(OmegaConf.load(Path(sys.argv[2])), resolve=True)

def flatten(value, prefix=()):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            out.update(flatten(item, prefix + (str(key),)))
        return out
    return {".".join(prefix): value}

a, b = flatten(before), flatten(after)
changed = {key for key in set(a) | set(b) if a.get(key) != b.get(key)}
allowed = {
    "trainer.test_freq",
    "trainer.val_after_train",
    "trainer.recovery_c20",
}
if changed != allowed:
    raise SystemExit(f"unregistered recovery config delta: {sorted(changed)}")
if a.get("trainer.test_freq") != 20 or b.get("trainer.test_freq") != -1:
    raise SystemExit("recovery must change only terminal test frequency 20 -> -1")
if b.get("trainer.val_after_train") is not False:
    raise SystemExit("recovery must disable in-process final validation")
if b.get("trainer.recovery_c20") is not True:
    raise SystemExit("recovery config lacks its explicit identity")
PY
}

run_recovery_job() {
    local mode="$1" variant="$2" argument="$3" model_path="${4:-}"
    local input_model_digest="${5:-}" budget_rmb timeout_seconds
    local parent run_dir started_epoch started_at rc requested_steps=0 input_model
    local trace_manifest expected_trace_rows trace_output_dir trace_stage
    local trace_checkpoint_digest='' trace_parent_checkpoint_digest=''
    local entry code_checkout recovery_flag=0 available_seconds
    local -a job_args
    case "$mode:$variant" in
        train:cost_aware_gated)
            [[ "$argument" == "$BRANCH_STEPS" &&
                "$model_path" == "$BC_R60_CHECKPOINT" &&
                "$input_model_digest" == "$BC_EXPECTED_R60_CHECKPOINT" ]] || {
                printf 'Recovery C20 must start from exact R60.\n' >&2
                return 64
            }
            budget_rmb="$RC_C_BUDGET_RMB"
            requested_steps="$argument"
            input_model="$model_path"
            parent="$RUNS_ROOT/cost_aware_gated"
            entry="$RC_CHECKOUT/scripts/autodl/train_small_grpo.sh"
            code_checkout="$RC_CHECKOUT"
            recovery_flag=1
            trace_parent_checkpoint_digest="$input_model_digest"
            ;;
        eval:qwen_native_b_val|eval:qwen_native_c_val|eval:qwen_native_b_nq_test|eval:qwen_native_c_nq_test)
            budget_rmb=5
            input_model="$argument"
            parent="$RUNS_ROOT/eval/$variant"
            entry="$CHECKOUT_DIR/scripts/autodl/train_small_grpo.sh"
            code_checkout="$CHECKOUT_DIR"
            trace_checkpoint_digest="$input_model_digest"
            ;;
        eval:qwen_native_b_multihop|eval:qwen_native_c_multihop)
            budget_rmb=10
            input_model="$argument"
            parent="$RUNS_ROOT/eval/$variant"
            entry="$CHECKOUT_DIR/scripts/autodl/train_small_grpo.sh"
            code_checkout="$CHECKOUT_DIR"
            trace_checkpoint_digest="$input_model_digest"
            ;;
        *)
            printf 'Unregistered recovery job: %s:%s\n' "$mode" "$variant" >&2
            return 64
            ;;
    esac
    [[ "$input_model_digest" =~ ^[0-9a-f]{64}$ ]] || return 64
    timeout_seconds="$(awk -v budget="$budget_rmb" -v price="$PRICE_PER_HOUR" \
        'BEGIN { printf "%d", budget / price * 3600 }')"
    ((timeout_seconds > 0)) || return 64
    if [[ "$mode" == train && "$timeout_seconds" -lt "$RC_MIN_C_TIMEOUT_SECONDS" ]]; then
        printf 'Registered C20 timeout %ss is below the recovery floor %ss.\n' \
            "$timeout_seconds" "$RC_MIN_C_TIMEOUT_SECONDS" >&2
        return 64
    fi
    mkdir -p "$parent/attempts"
    run_dir="$parent/attempts/$(date -u +'%Y%m%dT%H%M%SZ')-$$-$RANDOM"
    mkdir "$run_dir"
    : >"$run_dir/.running"
    atomic_write "$parent/latest" "$run_dir"$'\n'
    started_epoch="$(date +%s)"
    started_at="$(utc_now)"
    trace_output_dir="$run_dir/traces"
    trace_stage="$variant"
    printf 'Starting recovery %s %s (budget %s RMB, timeout %ss); log: %s/train.log\n' \
        "$mode" "$variant" "$budget_rmb" "$timeout_seconds" "$run_dir"
    job_args=("$mode" "$variant" "$argument")
    [[ -z "$model_path" ]] || job_args+=("$model_path")
    : >"$run_dir/train.log"
    set +e
    if [[ "$mode" == train ]]; then
        AUTODL_CONFIG_ONLY=1 \
            OUTPUT_DIR="$run_dir/checkpoints" GPU_COUNT="$GPU_COUNT" \
            TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
            MAX_RESPONSE_LENGTH="$MAX_RESPONSE_LENGTH" \
            EVAL_DATA_FILE="$EVAL_DATA_FILE" EVAL_GROUP_SIZE="$EVAL_GROUP_SIZE" \
            TRACE_OUTPUT_DIR="$trace_output_dir" TRACE_STAGE="$trace_stage" \
            TRACE_RUN_ID="$(basename -- "$run_dir")" \
            TRACE_PARENT_CHECKPOINT_DIGEST="$trace_parent_checkpoint_digest" \
            TOOL_PROTOCOL="$TOOL_PROTOCOL" AUTODL_ROOT="$PROJECT_ROOT" \
            bash "$CHECKOUT_DIR/scripts/autodl/train_small_grpo.sh" \
            "${job_args[@]}" >"$run_dir/baseline-resolved-config.yaml" \
            2>>"$run_dir/train.log"
        rc=$?
    else
        rc=0
    fi
    if ((rc == 0)); then
        AUTODL_CONFIG_ONLY=1 AUTODL_QWEN_NATIVE_RECOVERY_C20="$recovery_flag" \
            AUTODL_CODE_CHECKOUT="$code_checkout" \
            PYTHONPATH="$code_checkout${PYTHONPATH:+:$PYTHONPATH}" \
            OUTPUT_DIR="$run_dir/checkpoints" GPU_COUNT="$GPU_COUNT" \
            TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
            MAX_RESPONSE_LENGTH="$MAX_RESPONSE_LENGTH" \
            EVAL_DATA_FILE="$EVAL_DATA_FILE" EVAL_GROUP_SIZE="$EVAL_GROUP_SIZE" \
            TRACE_OUTPUT_DIR="$trace_output_dir" TRACE_STAGE="$trace_stage" \
            TRACE_RUN_ID="$(basename -- "$run_dir")" \
            TRACE_CHECKPOINT_DIGEST="$trace_checkpoint_digest" \
            TRACE_PARENT_CHECKPOINT_DIGEST="$trace_parent_checkpoint_digest" \
            TOOL_PROTOCOL="$TOOL_PROTOCOL" AUTODL_ROOT="$PROJECT_ROOT" \
            bash "$entry" "${job_args[@]}" >"$run_dir/resolved-config.yaml" \
            2>>"$run_dir/train.log"
        rc=$?
    fi
    if ((rc == 0)) && [[ "$mode" == train ]]; then
        verify_recovery_config_delta "$run_dir/baseline-resolved-config.yaml" \
            "$run_dir/resolved-config.yaml" >>"$run_dir/train.log" 2>&1
        rc=$?
    fi
    if ((rc == 0)); then
        mkdir -p "$run_dir/wandb"
        AUTODL_QWEN_NATIVE_RECOVERY_C20="$recovery_flag" \
            AUTODL_CODE_CHECKOUT="$code_checkout" \
            PYTHONPATH="$code_checkout${PYTHONPATH:+:$PYTHONPATH}" \
            OUTPUT_DIR="$run_dir/checkpoints" GPU_COUNT="$GPU_COUNT" \
            TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
            MAX_RESPONSE_LENGTH="$MAX_RESPONSE_LENGTH" \
            EVAL_DATA_FILE="$EVAL_DATA_FILE" EVAL_GROUP_SIZE="$EVAL_GROUP_SIZE" \
            TRACE_OUTPUT_DIR="$trace_output_dir" TRACE_STAGE="$trace_stage" \
            TRACE_RUN_ID="$(basename -- "$run_dir")" \
            TRACE_CHECKPOINT_DIGEST="$trace_checkpoint_digest" \
            TRACE_PARENT_CHECKPOINT_DIGEST="$trace_parent_checkpoint_digest" \
            WANDB_DIR="$run_dir/wandb" TOOL_PROTOCOL="$TOOL_PROTOCOL" \
            AUTODL_ROOT="$PROJECT_ROOT" \
            timeout --signal=TERM --kill-after=120s "${timeout_seconds}s" \
            bash "$entry" "${job_args[@]}" >>"$run_dir/train.log" 2>&1
        rc=$?
    fi
    if ((rc == 0)); then
        if [[ "$mode" == train ]]; then
            trace_manifest="$run_dir/traces/train_trajectories.manifest.json"
            expected_trace_rows=$((requested_steps * TRAIN_BATCH_SIZE * 5))
        else
            trace_manifest="$run_dir/traces/eval_predictions.manifest.json"
            expected_trace_rows="$EVAL_EXPECTED_ROWS"
        fi
        PYTHONPATH="$code_checkout${PYTHONPATH:+:$PYTHONPATH}" \
            "$TRAIN_ENV/bin/python" -m search_r1.trajectory_trace verify \
                --manifest "$trace_manifest" --expected-rows "$expected_trace_rows" \
                >>"$run_dir/train.log" 2>&1
        rc=$?
    fi
    set -e
    finish_run_record "$run_dir" "$rc" "$started_epoch" "$started_at" \
        "$budget_rmb" "$timeout_seconds" "$mode" "$variant" \
        "$requested_steps" "$input_model" "$trace_output_dir"
    LAST_RUN_DIR="$run_dir"
    if ((rc != 0)); then
        printf 'Recovery %s %s failed with exit code %s; inspect %s/train.log\n' \
            "$mode" "$variant" "$rc" "$run_dir" >&2
        return "$rc"
    fi
}

seal_recovery_c_run() {
    local run_dir="$1" checkpoint="$2" checkpoint_digest="$3"
    local commit="$4" handoff="$5" contract
    contract="$("$TRAIN_ENV/bin/python" - "$run_dir" "$checkpoint" \
        "$checkpoint_digest" "$BC_R60_CHECKPOINT" \
        "$BC_EXPECTED_R60_CHECKPOINT" "$commit" "$handoff" \
        "$NATIVE_TRAIN_DATA_DIR" "$NATIVE_TRAIN_PROMPT_VERSION" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
from omegaconf import OmegaConf

(run_raw, checkpoint_raw, checkpoint_digest, parent_raw, parent_digest,
 commit, handoff, data_raw, prompt_version) = sys.argv[1:]
run = Path(run_raw)
checkpoint = Path(checkpoint_raw)
parent = Path(parent_raw)
data = Path(data_raw)
config_path = run / "resolved-config.yaml"
baseline_path = run / "baseline-resolved-config.yaml"
env_path = run / "run.env"
lineage_path = run / "lineage.tsv"
for path in (config_path, baseline_path, env_path, lineage_path):
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"missing recovery evidence: {path}")
config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)

def value(*keys):
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            raise SystemExit(f"missing config key: {'.'.join(keys)}")
        current = current[key]
    return current

checks = {
    "protocol": value("tool_protocol") == "qwen35_native",
    "prompt": value("qwen35_prompt_version") == prompt_version,
    "variant": value("trainer", "native_training_variant") == "cost_aware_gated",
    "model": Path(value("actor_rollout_ref", "model", "path")).resolve() == parent.resolve(),
    "train": Path(value("data", "train_files")) == data / "train_512.parquet",
    "val": Path(value("data", "val_files")) == data / "val_128.parquet",
    "raw_chat": value("data", "return_raw_chat") is True,
    "batch": value("data", "train_batch_size") == 8,
    "val_batch": value("data", "val_batch_size") == 8,
    "group": value("actor_rollout_ref", "rollout", "n_agent") == 5,
    "mini_batch": value("actor_rollout_ref", "actor", "ppo_mini_batch_size") == 40,
    "micro_batch": value("actor_rollout_ref", "actor", "ppo_micro_batch_size") == 2,
    "steps": value("trainer", "total_training_steps") == 20,
    "save": value("trainer", "save_freq") == 20,
    "test": value("trainer", "test_freq") == -1,
    "no_final_val": value("trainer", "val_after_train") is False,
    "recovery": value("trainer", "recovery_c20") is True,
    "seed": value("trainer", "seed") == 42,
    "cost_lambda": float(value("algorithm", "cost_lambda")) == 0.10,
    "cost_mode": value("algorithm", "cost_reward_mode") == "correct_only",
    "trace_parent": value("trainer", "trace_parent_checkpoint_digest") == parent_digest,
    "response": value("data", "max_response_length") == 500,
    "observation": value("data", "max_obs_length") == 500,
    "turns": value("max_turns") == 4,
    "topk": value("retriever", "topk") == 3,
}
failed = sorted(key for key, passed in checks.items() if not passed)
if failed:
    raise SystemExit("recovery C config mismatch: " + ", ".join(failed))
run_env = {}
for line in env_path.read_text(encoding="utf-8").splitlines():
    if line and "=" in line:
        key, item = line.split("=", 1)
        if key in run_env:
            raise SystemExit(f"duplicate run.env key: {key}")
        run_env[key] = item
config_digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
baseline_digest = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
expected_env = {
    "job_mode": "train", "variant": "cost_aware_gated",
    "gpu_count": "2", "train_steps": "20", "train_batch_size": "8",
    "input_model": str(parent), "resolved_config_sha256": config_digest,
    "role": "cost_aware_gated", "checkpoint": str(checkpoint),
    "checkpoint_digest": checkpoint_digest, "parent_checkpoint": str(parent),
    "parent_checkpoint_digest": parent_digest, "checkout_commit": commit,
    "cpu_handoff_digest": handoff, "seed": "42", "cost_lambda": "0.10",
    "cost_reward_mode": "correct_only", "budget_rmb": "40",
}
bad = sorted(key for key, expected in expected_env.items()
             if run_env.get(key) != expected)
if bad:
    raise SystemExit("recovery C run.env mismatch: " + ", ".join(bad))
lines = lineage_path.read_text(encoding="utf-8").splitlines()
header = ("role\tcheckpoint\tcheckpoint_digest\tparent_checkpoint\t"
          "parent_checkpoint_digest\tcheckout_commit\tcpu_handoff_digest\t"
          "resolved_config_sha256")
row = "\t".join([
    "cost_aware_gated", str(checkpoint), checkpoint_digest, str(parent),
    parent_digest, commit, handoff, config_digest,
])
if lines != [header, row]:
    raise SystemExit("recovery C lineage mismatch")
payload = {
    "baseline_config_sha256": baseline_digest,
    "checkpoint": str(checkpoint), "checkpoint_digest": checkpoint_digest,
    "config_sha256": config_digest, "cost_lambda": 0.10,
    "cost_reward_mode": "correct_only", "execution_checkout_commit": commit,
    "external_val_required": True, "group_size": 5,
    "inline_final_validation": False, "parent": str(parent),
    "parent_digest": parent_digest, "prompt_version": prompt_version,
    "role": "cost_aware_gated", "save_freq": 20,
    "schema": "qwen-native-recovery-c20-v1", "steps": 20,
    "test_freq": -1, "train_batch_size": 8, "variant": "cost_aware_gated",
}
print(json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
PY
)" || return 1
    atomic_write "$run_dir/native-training-contract.json" "$contract"$'\n'
    sync_path "$run_dir"
}

verify_recovery_results_semantics() {
    local results="$1" cost_run="$2" cost_checkpoint="$3" cost_digest="$4"
    "$TRAIN_ENV/bin/python" - "$results" "$RC_CONTRACT" "$RC_B_RUN" \
        "$RC_B_CHECKPOINT" "$RC_B_CHECKPOINT_SHA256" "$cost_run" \
        "$cost_checkpoint" "$cost_digest" "$BC_R60_CHECKPOINT" \
        "$BC_EXPECTED_R60_CHECKPOINT" "$RC_OLD_C_RUN" <<'PY'
import csv
import json
from pathlib import Path
import sys

(root_raw, contract, b_run_raw, b_checkpoint_raw, b_digest, c_run_raw,
 c_checkpoint_raw, c_digest, r60_raw, r60_digest, old_c_raw) = sys.argv[1:]
root = Path(root_raw)
b_run, c_run = Path(b_run_raw), Path(c_run_raw)
b_checkpoint, c_checkpoint = Path(b_checkpoint_raw), Path(c_checkpoint_raw)
r60, old_c = Path(r60_raw), Path(old_c_raw)

def env(path):
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split("=", 1)
        if key in out:
            raise SystemExit(f"duplicate env key: {key}")
        out[key] = value
    return out

contract_env = env(root / "contract.env")
expected_contract = {
    "schema": contract,
    "stage": "bc_recovery",
    "stage_order": (
        "B20-REUSED,C20-RECOVERY,B-VAL-EVAL,C-VAL-EVAL,"
        "B-NQ-TEST-EVAL,C-NQ-TEST-EVAL,B-MULTIHOP-EVAL,C-MULTIHOP-EVAL"
    ),
    "controlled_outer_exit_code": "203",
    "source_failed_outer_adopted": "false",
    "reused_b": "true",
    "old_c_adopted": "false",
    "c_inline_validation_deferred": "true",
    "external_val_required": "true",
}
for key, value in expected_contract.items():
    if contract_env.get(key) != value:
        raise SystemExit(f"recovery contract mismatch: {key}")
with (root / "lineage.tsv").open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
expected = [
    ("B20-REUSED", "control"), ("C20-RECOVERY", "cost_aware_gated"),
    ("B-VAL-EVAL", "control_val_eval"),
    ("C-VAL-EVAL", "cost_aware_gated_val_eval"),
    ("B-NQ-TEST-EVAL", "control_nq_test_eval"),
    ("C-NQ-TEST-EVAL", "cost_aware_gated_nq_test_eval"),
    ("B-MULTIHOP-EVAL", "control_multihop_eval"),
    ("C-MULTIHOP-EVAL", "cost_aware_gated_multihop_eval"),
]
if [(row["stage"], row["role"]) for row in rows] != expected:
    raise SystemExit("recovery lineage stage order mismatch")
if Path(rows[0]["run_dir"]) != b_run or Path(rows[0]["checkpoint"]) != b_checkpoint or rows[0]["checkpoint_digest"] != b_digest:
    raise SystemExit("reused B identity mismatch")
if Path(rows[1]["run_dir"]) != c_run or Path(rows[1]["checkpoint"]) != c_checkpoint or rows[1]["checkpoint_digest"] != c_digest:
    raise SystemExit("new C identity mismatch")
for row in rows[:2]:
    if Path(row["parent_checkpoint"]) != r60 or row["parent_checkpoint_digest"] != r60_digest:
        raise SystemExit("B and C must independently inherit exact R60")
if any(Path(row["run_dir"]) == old_c for row in rows):
    raise SystemExit("failed historical C was adopted")
payload = json.loads((c_run / "native-training-contract.json").read_bytes())
if (payload.get("schema") != "qwen-native-recovery-c20-v1" or
        payload.get("inline_final_validation") is not False or
        payload.get("external_val_required") is not True or
        payload.get("parent") != str(r60) or payload.get("parent_digest") != r60_digest):
    raise SystemExit("new C recovery contract mismatch")
with (root / "run-index.tsv").open(newline="", encoding="utf-8") as handle:
    index = list(csv.DictReader(handle, delimiter="\t"))
if len(index) != 8:
    raise SystemExit("recovery run index must have exactly eight rows")
for row, item in zip(rows, index):
    if (row["stage"], row["role"], row["run_dir"]) != (
            item["stage"], item["role"], item["run_dir"]):
        raise SystemExit("recovery run index disagrees with lineage")
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
    local control_run="$RC_B_RUN" control_checkpoint="$RC_B_CHECKPOINT"
    local control_digest="$RC_B_CHECKPOINT_SHA256" cost_run cost_checkpoint cost_digest
    local control_trace_digest="$RC_B_TRACE_SHA256"
    local control_trace_manifest_digest="$RC_B_TRACE_MANIFEST_SHA256"
    local cost_trace cost_trace_digest cost_trace_manifest_digest
    local results_dir runner_digest index_content branch_rows='' b_receipt
    local eval_key eval_rows eval_stage eval_artifact trace_identity run config_digest
    local eval_trace_digest eval_trace_manifest_digest cost_config cost_contract
    local wandb_run wandb_file wandb_steps wandb_count marker evidence_digest
    local -a eval_keys=(val nq_test multihop)
    local -A eval_expected_rows=([val]=128 [nq_test]=128 [multihop]=256)
    local -A eval_artifacts=([val]=val [nq_test]=nq_test_eval [multihop]=multihop_eval)
    local -A control_eval_runs=() cost_eval_runs=()
    local -A control_eval_trace_digests=() control_eval_trace_manifest_digests=()
    local -A cost_eval_trace_digests=() cost_eval_trace_manifest_digests=()
    local -A paired_dirs=()
    local -a evidence_files=() wandb_runs=()

    verify_recovery_checkout || return $?
    runner_digest="$RC_CODE_RUNNER_SHA256"
    [[ "$RC_PREFLIGHT_COMMIT" == "$commit" &&
        "$RC_PREFLIGHT_HANDOFF" == "$handoff_digest" &&
        "$RC_PREFLIGHT_BASE_DIGEST" == "$base_digest" &&
        "$RC_PREFLIGHT_DATA_DIGEST" == "$data_digest" &&
        "$RC_PREFLIGHT_RUNNER_DIGEST" == "$runner_digest" ]] || {
        printf 'Recovery inputs drifted after GPU preflight.\n' >&2
        return 1
    }
    verify_recovery_publish_identity "$commit" "$data_digest" "$runner_digest" || return $?
    verify_native_checkpoint_digest "$BC_R60_CHECKPOINT" \
        "$BC_EXPECTED_R60_CHECKPOINT" || return $?
    results_dir="$(create_native_training_results_dir "$outer_attempt")" || return 1

    append_native_train_index "$outer_attempt" B20-REUSED control "$control_run" || return $?
    mkdir "$results_dir/wandb-receipts"
    b_receipt="$results_dir/wandb-receipts/B20-reused.json"
    rc_build_b_wandb_receipt "$b_receipt" || return $?
    atomic_write "$results_dir/source-b.env" \
        "schema=qwen-native-reused-b20-v1"$'\n'\
"reuse_kind=verified_inner_success"$'\n'\
"source_outer=$RC_SOURCE_OUTER"$'\n'\
"source_failed_outer_adopted=false"$'\n'\
"run=$control_run"$'\n'\
"checkpoint=$control_checkpoint"$'\n'\
"checkpoint_sha256=$control_digest"$'\n'\
"trace_sha256=$control_trace_digest"$'\n'\
"wandb_receipt=$b_receipt"$'\n'

    verify_native_checkpoint_digest "$BC_R60_CHECKPOINT" \
        "$BC_EXPECTED_R60_CHECKPOINT" || return $?
    export EVAL_EXPECTED_ROWS=128 EVAL_GROUP_SIZE=1 EVAL_DATA_FILE=''
    run_recovery_job train cost_aware_gated "$BRANCH_STEPS" \
        "$BC_R60_CHECKPOINT" "$BC_EXPECTED_R60_CHECKPOINT" || return $?
    cost_run="$LAST_RUN_DIR"
    append_native_train_index "$outer_attempt" C20-RECOVERY cost_aware_gated \
        "$cost_run" || return $?
    cost_checkpoint="$(fixed_checkpoint "$cost_run" "$BRANCH_STEPS")" || return 1
    cost_digest="$(tree_sha256 "$cost_checkpoint")" || return 1
    record_lineage "$cost_run" cost_aware_gated "$cost_checkpoint" "$cost_digest" \
        "$BC_R60_CHECKPOINT" "$BC_EXPECTED_R60_CHECKPOINT" \
        "$RC_RECOVERY_COMMIT" "$handoff_digest" 0.10 correct_only || return $?
    seal_recovery_c_run "$cost_run" "$cost_checkpoint" "$cost_digest" \
        "$RC_RECOVERY_COMMIT" "$handoff_digest" || return $?
    cost_trace="$(verify_native_trace_identity \
        "$cost_run/traces/train_trajectories.manifest.json" 800)" || return 1
    IFS=$'\t' read -r cost_trace_digest cost_trace_manifest_digest <<<"$cost_trace"

    for eval_key in "${eval_keys[@]}"; do
        eval_rows="${eval_expected_rows[$eval_key]}"
        eval_artifact="${eval_artifacts[$eval_key]}"
        export EVAL_EXPECTED_ROWS="$eval_rows"
        verify_native_checkpoint_digest "$control_checkpoint" "$control_digest" || return $?
        run_recovery_job eval "qwen_native_b_$eval_key" "$control_checkpoint" '' \
            "$control_digest" || return $?
        run="$LAST_RUN_DIR"
        control_eval_runs[$eval_key]="$run"
        eval_stage="${eval_key^^}"
        eval_stage="B-${eval_stage//_/-}-EVAL"
        append_native_train_index "$outer_attempt" "$eval_stage" \
            "control_${eval_key}_eval" "$run" || return $?
        trace_identity="$(verify_native_trace_identity \
            "$run/traces/eval_predictions.manifest.json" "$eval_rows")" || return 1
        IFS=$'\t' read -r eval_trace_digest eval_trace_manifest_digest <<<"$trace_identity"
        control_eval_trace_digests[$eval_key]="$eval_trace_digest"
        control_eval_trace_manifest_digests[$eval_key]="$eval_trace_manifest_digest"

        verify_native_checkpoint_digest "$cost_checkpoint" "$cost_digest" || return $?
        run_recovery_job eval "qwen_native_c_$eval_key" "$cost_checkpoint" '' \
            "$cost_digest" || return $?
        run="$LAST_RUN_DIR"
        cost_eval_runs[$eval_key]="$run"
        eval_stage="${eval_key^^}"
        eval_stage="C-${eval_stage//_/-}-EVAL"
        append_native_train_index "$outer_attempt" "$eval_stage" \
            "cost_aware_gated_${eval_key}_eval" "$run" || return $?
        trace_identity="$(verify_native_trace_identity \
            "$run/traces/eval_predictions.manifest.json" "$eval_rows")" || return 1
        IFS=$'\t' read -r eval_trace_digest eval_trace_manifest_digest <<<"$trace_identity"
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

    cost_config="$(file_sha256 "$cost_run/resolved-config.yaml")" || return 1
    cost_contract="$(file_sha256 "$cost_run/native-training-contract.json")" || return 1
    atomic_write "$results_dir/contract.env" \
"schema=$RC_CONTRACT
stage=bc_recovery
stage_order=B20-REUSED,C20-RECOVERY,B-VAL-EVAL,C-VAL-EVAL,B-NQ-TEST-EVAL,C-NQ-TEST-EVAL,B-MULTIHOP-EVAL,C-MULTIHOP-EVAL
experiment_class=post_hoc_exploratory_recovery
controlled_outer_exit_code=$RC_TERMINAL_CODE
source_failed_outer=$RC_SOURCE_OUTER
source_failed_outer_terminal=failed
source_failed_outer_exit_code=124
source_failed_outer_adopted=false
reused_b=true
reused_b_kind=verified_inner_success
reused_b_run=$control_run
reused_b_checkpoint_sha256=$control_digest
ignored_failed_c_run=$RC_OLD_C_RUN
ignored_failed_c_trace_sha256=$RC_OLD_C_TRACE_SHA256
old_c_adopted=false
c_parent_checkpoint=$BC_R60_CHECKPOINT
c_parent_checkpoint_sha256=$BC_EXPECTED_R60_CHECKPOINT
c_budget_rmb=$RC_C_BUDGET_RMB
c_save_freq=20
c_test_freq=-1
c_inline_validation_deferred=true
external_val_required=true
canonical_execution_checkout=$BC_EXPECTED_CHECKOUT
recovery_execution_checkout=$RC_RECOVERY_COMMIT
cpu_receipt=$RC_CPU_RECEIPT
cpu_receipt_sha256=$RC_CPU_RECEIPT_DIGEST
runner=$RC_DEPLOY_PATH
runner_sha256=$runner_digest
"
    atomic_write "$results_dir/lineage.tsv" \
        $'stage\trole\trun_dir\tcheckpoint\tcheckpoint_digest\tparent_checkpoint\tparent_checkpoint_digest\tcheckout_commit\tcpu_handoff_digest\tdata_manifest_sha256\tresolved_config_sha256\ttrace_sha256\ttrace_manifest_sha256\trun_contract_sha256\tpredecessor_evidence_sha256\n'
    branch_rows="B20-REUSED"$'\t'"control"$'\t'"$control_run"$'\t'"$control_checkpoint"$'\t'"$control_digest"$'\t'"$BC_R60_CHECKPOINT"$'\t'"$BC_EXPECTED_R60_CHECKPOINT"$'\t'"$BC_EXPECTED_CHECKOUT"$'\t'"$handoff_digest"$'\t'"$data_digest"$'\t'"$RC_B_CONFIG_SHA256"$'\t'"$control_trace_digest"$'\t'"$control_trace_manifest_digest"$'\t'"$RC_B_CONTRACT_SHA256"$'\t'"$BC_EXPECTED_R60_EVIDENCE"$'\n'
    branch_rows+="C20-RECOVERY"$'\t'"cost_aware_gated"$'\t'"$cost_run"$'\t'"$cost_checkpoint"$'\t'"$cost_digest"$'\t'"$BC_R60_CHECKPOINT"$'\t'"$BC_EXPECTED_R60_CHECKPOINT"$'\t'"$RC_RECOVERY_COMMIT"$'\t'"$handoff_digest"$'\t'"$data_digest"$'\t'"$cost_config"$'\t'"$cost_trace_digest"$'\t'"$cost_trace_manifest_digest"$'\t'"$cost_contract"$'\t'"$BC_EXPECTED_R60_EVIDENCE"$'\n'
    for eval_key in "${eval_keys[@]}"; do
        eval_stage="${eval_key^^}"
        eval_stage="${eval_stage//_/-}"
        run="${control_eval_runs[$eval_key]}"
        config_digest="$(file_sha256 "$run/resolved-config.yaml")" || return 1
        branch_rows+="B-$eval_stage-EVAL"$'\t'"control_${eval_key}_eval"$'\t'"$run"$'\t'"$control_checkpoint"$'\t'"$control_digest"$'\t'"$control_checkpoint"$'\t'"$control_digest"$'\t'"$BC_EXPECTED_CHECKOUT"$'\t'"$handoff_digest"$'\t'"$data_digest"$'\t'"$config_digest"$'\t'"${control_eval_trace_digests[$eval_key]}"$'\t'"${control_eval_trace_manifest_digests[$eval_key]}"$'\t-\t'"$BC_EXPECTED_R60_EVIDENCE"$'\n'
        run="${cost_eval_runs[$eval_key]}"
        config_digest="$(file_sha256 "$run/resolved-config.yaml")" || return 1
        branch_rows+="C-$eval_stage-EVAL"$'\t'"cost_aware_gated_${eval_key}_eval"$'\t'"$run"$'\t'"$cost_checkpoint"$'\t'"$cost_digest"$'\t'"$cost_checkpoint"$'\t'"$cost_digest"$'\t'"$BC_EXPECTED_CHECKOUT"$'\t'"$handoff_digest"$'\t'"$data_digest"$'\t'"$config_digest"$'\t'"${cost_eval_trace_digests[$eval_key]}"$'\t'"${cost_eval_trace_manifest_digests[$eval_key]}"$'\t-\t'"$BC_EXPECTED_R60_EVIDENCE"$'\n'
    done
    printf '%s' "$branch_rows" >>"$results_dir/lineage.tsv"
    index_content="$(cat "$outer_attempt/native-training-runs.tsv")"
    atomic_write "$results_dir/run-index.tsv" "$index_content"$'\n'
    atomic_write "$results_dir/branch-checkpoints.env" \
        "control_checkpoint=$control_checkpoint"$'\n'\
"control_checkpoint_sha256=$control_digest"$'\n'\
"cost_checkpoint=$cost_checkpoint"$'\n'\
"cost_checkpoint_sha256=$cost_digest"$'\n'
    atomic_write "$outer_attempt/bc-recovery-complete.env" \
        "schema=$RC_CONTRACT"$'\n'\
"result_root=$results_dir"$'\n'\
"control_checkpoint=$control_checkpoint"$'\n'\
"control_checkpoint_sha256=$control_digest"$'\n'\
"cost_checkpoint=$cost_checkpoint"$'\n'\
"cost_checkpoint_sha256=$cost_digest"$'\n'\
"runner_sha256=$runner_digest"$'\n'

    wandb_runs=("$cost_run")
    for eval_key in "${eval_keys[@]}"; do
        wandb_runs+=("${control_eval_runs[$eval_key]}" "${cost_eval_runs[$eval_key]}")
    done
    for wandb_run in "${wandb_runs[@]}"; do
        if [[ "$wandb_run" == "$cost_run" ]]; then
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
    while IFS= read -r -d '' wandb_file; do
        evidence_files+=("$wandb_file")
    done < <(find "$RC_B_RUN/wandb" -type f -size +0c -print0 | LC_ALL=C sort -z)

    evidence_files+=(
        "$results_dir/contract.env" "$results_dir/lineage.tsv"
        "$results_dir/run-index.tsv" "$results_dir/branch-checkpoints.env"
        "$results_dir/source-b.env" "$b_receipt"
        "$outer_attempt/bc-recovery-complete.env" "$RC_RUNNER_PATH"
        "$RC_CPU_RECEIPT" "$RC_CPU_RECEIPT.sha256" "$RC_CPU_B_WANDB_RECEIPT"
        "$RC_CHECKOUT/scripts/autodl/train_small_grpo.sh"
        "$RC_CHECKOUT/verl/trainer/ppo/ray_trainer.py"
        "$QWEN_NATIVE_PROTOCOL_GATE_EVIDENCE" "$NATIVE_TRAIN_PROTOCOL_GATE_MANIFEST"
        "$QWEN_NATIVE_SMOKE_EVIDENCE" "$NATIVE_TRAIN_SMOKE_MANIFEST"
        "$QWEN_NATIVE_R60_EVIDENCE" "$BC_R60_RESULTS/evidence.sha256"
        "$BC_G3_RUN/resolved-config.yaml" "$BC_G3_RUN/run.env"
        "$BC_G3_RUN/train.log" "$BC_G3_TRACE" "$BC_G3_TRACE_MANIFEST"
        "$BC_G3_TRACE_SIDECAR" "$BC_G3_ANALYSIS/summary.json"
        "$BC_G3_ANALYSIS/summary.md" "$BC_G3_ANALYSIS/go_no_go.json"
        "$BC_G3_ANALYSIS/per_question.jsonl"
        "$BC_G3_ANALYSIS/per_trajectory.jsonl"
        "$RC_SOURCE_OUTER/phase.log" "$RC_SOURCE_OUTER/native-training-runs.tsv"
        "$RC_SOURCE_OUTER/terminal" "$RC_SOURCE_OUTER/exit-code"
        "$RC_OLD_C_RUN/train.log" "$RC_OLD_C_RUN/resolved-config.yaml"
        "$RC_OLD_C_RUN/run.env" "$RC_OLD_C_RUN/terminal"
        "$RC_OLD_C_RUN/exit-code" "$RC_OLD_C_TRACE"
        "$control_run/train.log" "$control_run/resolved-config.yaml"
        "$control_run/run.env" "$control_run/lineage.tsv"
        "$control_run/terminal" "$control_run/exit-code"
        "$control_run/native-training-contract.json"
        "$control_run/traces/train_trajectories.jsonl"
        "$control_run/traces/train_trajectories.manifest.json"
        "$control_run/traces/train_trajectories.manifest.json.sha256"
        "$cost_run/train.log" "$cost_run/resolved-config.yaml"
        "$cost_run/baseline-resolved-config.yaml" "$cost_run/run.env"
        "$cost_run/lineage.tsv" "$cost_run/native-training-contract.json"
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

    verify_recovery_publish_identity "$commit" "$data_digest" "$runner_digest" || return $?
    verify_native_checkpoint_digest "$BC_R60_CHECKPOINT" \
        "$BC_EXPECTED_R60_CHECKPOINT" || return $?
    verify_native_checkpoint_digest "$control_checkpoint" "$control_digest" || return $?
    verify_native_checkpoint_digest "$cost_checkpoint" "$cost_digest" || return $?
    verify_recovery_results_semantics "$results_dir" "$cost_run" \
        "$cost_checkpoint" "$cost_digest" || return $?
    sync_path "$results_dir"
    sync_path "$outer_attempt"
    NATIVE_TRAIN_STAGE=bc-recovery
    publish_native_training_evidence "$outer_attempt" "$results_dir" \
        "$RC_NAMESPACE" "$RC_CONTRACT" "${evidence_files[@]}" || return $?
    marker="$MANIFEST_DIR/$RC_NAMESPACE/$(basename -- "$outer_attempt").ok"
    evidence_digest="$(file_sha256 "$results_dir/evidence.sha256")" || return 1
    [[ "$(rc_value "$marker")" == "$evidence_digest" &&
        "$(rc_value "$outer_attempt/result-contract")" == "$RC_CONTRACT" &&
        "$(rc_value "$outer_attempt/result-root")" == "$results_dir" &&
        "$(rc_value "$outer_attempt/evidence-marker")" == "$marker" &&
        "$(rc_value "$outer_attempt/evidence-digest")" == "$evidence_digest" ]] || return 1
    verify_native_training_checksum_manifest "$results_dir/evidence.sha256" || return $?
    NATIVE_TRAIN_STAGE=main
    printf 'Completed and sealed B/C recovery comparison: %s\n' "$results_dir"
    printf 'Returning controlled outer status %s for the pinned shutdown watchdog.\n' \
        "$RC_TERMINAL_CODE"
    return "$RC_TERMINAL_CODE"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    case "${1:-}" in
        --cpu-prepare)
            [[ "$#" == 1 ]] || exit 64
            prepare_recovery_cpu_receipt
            ;;
        *)
            qwen_native_train_main "$@"
            ;;
    esac
fi
