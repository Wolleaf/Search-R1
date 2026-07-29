#!/usr/bin/env bash
set -Eeuo pipefail

# This operator entrypoint keeps the sealed checkout untouched while limiting the
# approved main workflow to its R60 stage.
R60_PROJECT_ROOT="${AUTODL_ROOT:-/root/autodl-tmp/search-r1}"
R60_SOURCE_ENTRY="$R60_PROJECT_ROOT/checkout/scripts/autodl/09_gpu_qwen_native_train.sh"
R60_RUNNER_PATH="$(readlink -f -- "$0")"
export QWEN_NATIVE_TRAIN_STAGE=main

[[ -f "$R60_SOURCE_ENTRY" && ! -L "$R60_SOURCE_ENTRY" ]] || {
    printf 'Missing sealed Qwen native training entrypoint: %s\n' "$R60_SOURCE_ENTRY" >&2
    exit 1
}

# shellcheck source=09_gpu_qwen_native_train.sh
source "$R60_SOURCE_ENTRY"

readonly R60_ONLY_CONTRACT=qwen-native-training-r60-only-v1
readonly R60_ONLY_NAMESPACE=qwen-native-training-r60-only
# The checkout watchdog does not yet understand the R60-only success contract.
# A distinct nonzero outer status lets its fail-closed failure path shut down the
# paid instance after the independently sealed R run has completed successfully.
readonly R60_ONLY_TERMINAL_CODE=200

qwen_native_main_pipeline() {
    local outer_attempt="$1" commit="$2" handoff_digest="$3"
    local base_model="$4" base_digest="$5" data_digest="$6"
    local reproduce_run reproduce_checkpoint reproduce_digest
    local reproduce_config reproduce_contract reproduce_trace
    local reproduce_trace_digest reproduce_trace_manifest_digest
    local results_dir index_content runner_digest wandb_file wandb_count=0
    local -a evidence_files wandb_files=()

    verify_native_checkpoint_digest "$base_model" "$base_digest" || return $?
    run_job train reproduce "$REPRODUCE_STEPS" '' "$base_digest" || return $?
    reproduce_run="$LAST_RUN_DIR"
    append_native_train_index "$outer_attempt" R reproduced "$reproduce_run" || return $?
    reproduce_checkpoint="$(fixed_checkpoint "$reproduce_run" "$REPRODUCE_STEPS")" || return 1
    reproduce_digest="$(tree_sha256 "$reproduce_checkpoint")" || return 1
    record_lineage "$reproduce_run" reproduced "$reproduce_checkpoint" \
        "$reproduce_digest" "$base_model" "$base_digest" "$commit" \
        "$handoff_digest" 0 linear || return $?
    seal_native_training_run "$reproduce_run" reproduced "$reproduce_checkpoint" \
        "$reproduce_digest" "$base_model" "$base_digest" reproduce \
        "$REPRODUCE_STEPS" 0 linear "$commit" "$handoff_digest" || return $?
    reproduce_trace="$(verify_native_trace_identity \
        "$reproduce_run/traces/train_trajectories.manifest.json" \
        "$((REPRODUCE_STEPS * TRAIN_BATCH_SIZE * 5))")" || return 1
    IFS=$'\t' read -r reproduce_trace_digest reproduce_trace_manifest_digest \
        <<<"$reproduce_trace"

    [[ -d "$reproduce_run/wandb" && ! -L "$reproduce_run/wandb" ]] || {
        printf 'Run-specific WandB directory is missing: %s\n' "$reproduce_run" >&2
        return 1
    }
    write_wandb_receipt "$reproduce_run" "$REPRODUCE_STEPS" || return $?
    verify_wandb_receipt "$reproduce_run" "$REPRODUCE_STEPS" || return $?
    while IFS= read -r -d '' wandb_file; do
        wandb_files+=("$wandb_file")
        ((wandb_count += 1))
    done < <(find "$reproduce_run/wandb" -type f -size +0c -print0 | LC_ALL=C sort -z)
    ((wandb_count > 0)) || {
        printf 'Run-specific WandB history is empty: %s\n' "$reproduce_run" >&2
        return 1
    }

    results_dir="$(create_native_training_results_dir "$outer_attempt")" || return 1
    reproduce_config="$(file_sha256 "$reproduce_run/resolved-config.yaml")" || return 1
    reproduce_contract="$(file_sha256 "$reproduce_run/native-training-contract.json")" || return 1
    runner_digest="$(file_sha256 "$R60_RUNNER_PATH")" || return 1
    atomic_write "$results_dir/contract.env" \
        "schema=$R60_ONLY_CONTRACT"$'\n'\
"stage=r60_only"$'\n'\
"stage_order=R60"$'\n'\
"decision=COMPLETE"$'\n'\
"branch_training_authorized=false"$'\n'\
"controlled_outer_exit_code=$R60_ONLY_TERMINAL_CODE"$'\n'\
"protocol_gate_evidence=$NATIVE_PROTOCOL_GATE_EVIDENCE"$'\n'\
"protocol_gate_evidence_sha256=$NATIVE_TRAIN_PROTOCOL_GATE_DIGEST"$'\n'\
"smoke_evidence=$NATIVE_SMOKE_EVIDENCE"$'\n'\
"smoke_evidence_sha256=$NATIVE_TRAIN_SMOKE_DIGEST"$'\n'
    atomic_write "$results_dir/lineage.tsv" \
        $'stage\trole\trun_dir\tcheckpoint\tcheckpoint_digest\tparent_checkpoint\tparent_checkpoint_digest\tcheckout_commit\tcpu_handoff_digest\tdata_manifest_sha256\tresolved_config_sha256\ttrace_sha256\ttrace_manifest_sha256\trun_contract_sha256\tpredecessor_evidence_sha256\n'\
"R"$'\t'"reproduced"$'\t'"$reproduce_run"$'\t'"$reproduce_checkpoint"$'\t'"$reproduce_digest"$'\t'"$base_model"$'\t'"$base_digest"$'\t'"$commit"$'\t'"$handoff_digest"$'\t'"$data_digest"$'\t'"$reproduce_config"$'\t'"$reproduce_trace_digest"$'\t'"$reproduce_trace_manifest_digest"$'\t'"$reproduce_contract"$'\t'"$NATIVE_TRAIN_SMOKE_DIGEST"$'\n'
    index_content="$(cat "$outer_attempt/native-training-runs.tsv")"
    atomic_write "$results_dir/run-index.tsv" "$index_content"$'\n'
    atomic_write "$results_dir/checkpoint-tree.env" \
        "checkpoint=$reproduce_checkpoint"$'\n'\
"checkpoint_tree_sha256=$reproduce_digest"$'\n'
    record_smoke_storage "$results_dir" "$reproduce_checkpoint" || return $?
    atomic_write "$outer_attempt/r60-only-complete.env" \
        "schema=$R60_ONLY_CONTRACT"$'\n'\
"run_dir=$reproduce_run"$'\n'\
"checkpoint=$reproduce_checkpoint"$'\n'\
"checkpoint_tree_sha256=$reproduce_digest"$'\n'\
"trace_sha256=$reproduce_trace_digest"$'\n'\
"runner=$R60_RUNNER_PATH"$'\n'\
"runner_sha256=$runner_digest"$'\n'\
"next_stage_requires_manual_approval=true"$'\n'
    sync_path "$results_dir"
    sync_path "$outer_attempt"

    verify_checkout "$commit" || return $?
    [[ "$(file_sha256 "$NATIVE_TRAIN_MANIFEST")" == "$data_digest" ]] || return 1
    verify_protocol_gate_evidence "$NATIVE_PROTOCOL_GATE_EVIDENCE" "$base_model" \
        "$base_digest" "$commit" "$handoff_digest" "$data_digest" || return $?
    verify_smoke_evidence "$NATIVE_SMOKE_EVIDENCE" "$base_model" "$base_digest" \
        "$commit" "$handoff_digest" "$data_digest" \
        "$NATIVE_TRAIN_PROTOCOL_GATE_DIGEST" || return $?
    verify_native_checkpoint_digest "$base_model" "$base_digest" || return $?
    verify_native_checkpoint_digest "$reproduce_checkpoint" "$reproduce_digest" || return $?

    evidence_files=(
        "$results_dir/contract.env" "$results_dir/lineage.tsv"
        "$results_dir/run-index.tsv" "$results_dir/storage.env"
        "$results_dir/checkpoint-tree.env" "$outer_attempt/r60-only-complete.env"
        "$R60_RUNNER_PATH" "$NATIVE_PROTOCOL_GATE_EVIDENCE"
        "$NATIVE_TRAIN_PROTOCOL_GATE_MANIFEST" "$NATIVE_SMOKE_EVIDENCE"
        "$NATIVE_TRAIN_SMOKE_MANIFEST" "$NATIVE_TRAIN_SMOKE_RESULTS/contract.env"
        "$NATIVE_TRAIN_SMOKE_RESULTS/lineage.tsv"
        "$reproduce_run/train.log" "$reproduce_run/resolved-config.yaml"
        "$reproduce_run/run.env" "$reproduce_run/lineage.tsv"
        "$reproduce_run/native-training-contract.json"
        "$reproduce_run/wandb-receipt.json"
        "$reproduce_run/traces/train_trajectories.jsonl"
        "$reproduce_run/traces/train_trajectories.manifest.json"
        "$reproduce_run/traces/train_trajectories.manifest.json.sha256"
        "${wandb_files[@]}"
    )
    # Do not repoint latest-main: this slice deliberately stops before G3/B/C.
    NATIVE_TRAIN_STAGE=r60-only
    publish_native_training_evidence "$outer_attempt" "$results_dir" \
        "$R60_ONLY_NAMESPACE" "$R60_ONLY_CONTRACT" \
        "${evidence_files[@]}" || return $?

    printf 'Completed and sealed R60-only training: %s\n' "$results_dir"
    printf 'G3, endpoint evaluation, B20, and C20 were not started.\n'
    printf 'Returning controlled outer status %s for the pinned shutdown watchdog.\n' \
        "$R60_ONLY_TERMINAL_CODE"
    return "$R60_ONLY_TERMINAL_CODE"
}

qwen_native_train_main "$@"
