#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export AUTODL_RUN_BUDGET_PROFILE=gated_followup
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-256}"
AUTODL_GPU_PIPELINE=cost_aware_gated
# This archived follow-up remains pinned to the historical 256-token comparison.
# Reuse the already-tested GPU admission, BM25 lifecycle, run records, and checkpoint validators.
# shellcheck source=03_gpu_run.sh
source "$SCRIPT_DIR/03_gpu_run.sh"

readonly FOLLOWUP_RESULTS_ROOT="$RUNS_ROOT/cost-aware-gated"
readonly LEGACY_RESULTS_DIR="$RUNS_ROOT/comparison"

require_two_gpu_followup() {
    validate_gpu_inputs
    [[ "$GPU_COUNT" == 2 ]] || {
        printf 'The registered C-gated follow-up requires GPU_COUNT=2.\n' >&2
        return 64
    }
    [[ "$TRAIN_BATCH_SIZE" == 8 && "$MAX_RESPONSE_LENGTH" == 256 ]] || {
        printf 'C-gated must keep TRAIN_BATCH_SIZE=8 and MAX_RESPONSE_LENGTH=256.\n' >&2
        return 64
    }
}

load_historical_lineage() {
    local header stage role checkpoint digest parent parent_digest commit handoff config extra
    local rows=0 legacy_digest
    [[ -d "$LEGACY_RESULTS_DIR" && ! -L "$LEGACY_RESULTS_DIR" ]] || {
        printf 'Historical comparison directory is missing: %s\n' "$LEGACY_RESULTS_DIR" >&2
        return 1
    }
    (
        cd "$LEGACY_RESULTS_DIR"
        sha256sum --strict --check comparison.sha256 >/dev/null
    )
    [[ -f "$MANIFEST_DIR/gpu.ok" && ! -L "$MANIFEST_DIR/gpu.ok" ]] || {
        printf 'Historical gpu.ok marker is missing.\n' >&2
        return 1
    }
    legacy_digest="$(file_sha256 "$LEGACY_RESULTS_DIR/comparison.sha256")"
    [[ "$(tr -d '\r\n' <"$MANIFEST_DIR/gpu.ok")" == "$legacy_digest" ]] || {
        printf 'Historical comparison no longer matches gpu.ok.\n' >&2
        return 1
    }
    header="$(head -n 1 "$LEGACY_RESULTS_DIR/lineage.tsv")"
    [[ "$header" == $'stage\trole\tcheckpoint\tcheckpoint_digest\tparent_checkpoint\tparent_checkpoint_digest\tcheckout_commit\tcpu_handoff_digest\tresolved_config_sha256' ]] || {
        printf 'Historical lineage header is invalid.\n' >&2
        return 1
    }
    while IFS=$'\t' read -r stage role checkpoint digest parent parent_digest commit handoff config extra; do
        [[ -z "${extra:-}" && "$digest" =~ ^[0-9a-f]{64}$ &&
            "$commit" =~ ^[0-9a-f]{40}$ && "$handoff" =~ ^[0-9a-f]{64}$ ]] || {
            printf 'Malformed historical lineage row for stage %s.\n' "$stage" >&2
            return 1
        }
        case "$stage:$role" in
            A:base)
                HIST_BASE_CHECKPOINT="$checkpoint"
                HIST_BASE_DIGEST="$digest"
                ;;
            R:reproduced)
                HIST_R_CHECKPOINT="$checkpoint"
                HIST_R_DIGEST="$digest"
                HIST_R_PARENT="$parent"
                HIST_R_PARENT_DIGEST="$parent_digest"
                HIST_R_COMMIT="$commit"
                HIST_R_HANDOFF="$handoff"
                HIST_R_CONFIG="$config"
                ;;
            B:control)
                HIST_B_CHECKPOINT="$checkpoint"
                HIST_B_DIGEST="$digest"
                HIST_B_PARENT="$parent"
                HIST_B_PARENT_DIGEST="$parent_digest"
                HIST_B_COMMIT="$commit"
                HIST_B_HANDOFF="$handoff"
                HIST_B_CONFIG="$config"
                ;;
            C:cost_aware)
                HIST_C_CHECKPOINT="$checkpoint"
                HIST_C_DIGEST="$digest"
                HIST_C_PARENT="$parent"
                HIST_C_PARENT_DIGEST="$parent_digest"
                HIST_C_COMMIT="$commit"
                HIST_C_HANDOFF="$handoff"
                HIST_C_CONFIG="$config"
                ;;
            *)
                printf 'Unexpected historical lineage stage: %s/%s\n' "$stage" "$role" >&2
                return 1
                ;;
        esac
        rows=$((rows + 1))
    done < <(tail -n +2 "$LEGACY_RESULTS_DIR/lineage.tsv")
    [[ "$rows" == 4 && -n "${HIST_R_CHECKPOINT:-}" &&
        -n "${HIST_B_CHECKPOINT:-}" && -n "${HIST_C_CHECKPOINT:-}" ]] || {
        printf 'Historical A/R/B/C lineage is incomplete.\n' >&2
        return 1
    }
    [[ "$HIST_R_PARENT" == "$HIST_BASE_CHECKPOINT" &&
        "$HIST_R_PARENT_DIGEST" == "$HIST_BASE_DIGEST" &&
        "$HIST_B_PARENT" == "$HIST_R_CHECKPOINT" &&
        "$HIST_C_PARENT" == "$HIST_R_CHECKPOINT" &&
        "$HIST_B_PARENT_DIGEST" == "$HIST_R_DIGEST" &&
        "$HIST_C_PARENT_DIGEST" == "$HIST_R_DIGEST" &&
        "$HIST_R_COMMIT" == "$HIST_B_COMMIT" && "$HIST_R_COMMIT" == "$HIST_C_COMMIT" &&
        "$HIST_R_HANDOFF" == "$HIST_B_HANDOFF" && "$HIST_R_HANDOFF" == "$HIST_C_HANDOFF" ]] || {
        printf 'Historical branch parent or identity mismatch.\n' >&2
        return 1
    }
}

verify_historical_checkpoint() {
    local checkpoint="$1" digest="$2" role="$3" run_variant="$4" step="$5"
    local suffix="/checkpoints/actor/global_step_$step" run_dir resolved actual_digest
    [[ "$checkpoint" == "$RUNS_ROOT/$run_variant/attempts/"*"$suffix" ]] || {
        printf 'Historical %s checkpoint has an unexpected path: %s\n' "$role" "$checkpoint" >&2
        return 1
    }
    run_dir="${checkpoint%$suffix}"
    resolved="$(fixed_checkpoint "$run_dir" "$step")"
    [[ "$resolved" == "$checkpoint" ]] || {
        printf 'Historical %s checkpoint resolved to a different path.\n' "$role" >&2
        return 1
    }
    grep -Fxq "role=$role" "$run_dir/run.env" \
        && grep -Fxq "checkpoint=$checkpoint" "$run_dir/run.env" \
        && grep -Fxq "checkpoint_digest=$digest" "$run_dir/run.env" || {
        printf 'Historical %s run metadata is inconsistent.\n' "$role" >&2
        return 1
    }
    actual_digest="$(tree_sha256 "$checkpoint")"
    [[ "$actual_digest" == "$digest" ]] || {
        printf 'Historical %s checkpoint digest mismatch.\n' "$role" >&2
        return 1
    }
}

project_relative_file() {
    local path="$1" canonical root
    [[ -f "$path" && ! -L "$path" ]] || {
        printf 'Evidence file is missing or symlinked: %s\n' "$path" >&2
        return 1
    }
    canonical="$(readlink -f -- "$path")"
    root="$(readlink -f -- "$PROJECT_ROOT")"
    [[ "$canonical" == "$root/"* ]] || {
        printf 'Evidence file escapes the project root: %s\n' "$path" >&2
        return 1
    }
    printf '%s\n' "${canonical#"$root/"}"
}

publish_followup_evidence() {
    local outer_attempt="$1" results_dir="$2" gated_run="$3" gate_run="$4"
    local control_eval="$5" cost_eval="$6" gated_eval="$7"
    local marker_dir marker evidence_digest relative path checksum_lines=''
    local -a evidence_files relative_files required_outputs
    required_outputs=(
        paired_results.csv correct_questions.csv wrong_questions.csv
        search_transition.csv summary.json summary.md lineage.tsv run-index.tsv
        gated_training_metrics.csv gated_training_curves.svg
    )
    for path in "${required_outputs[@]}"; do
        [[ -s "$results_dir/$path" && ! -L "$results_dir/$path" ]] || {
            printf 'Required follow-up output is missing: %s\n' "$results_dir/$path" >&2
            return 1
        }
        evidence_files+=("$results_dir/$path")
    done
    evidence_files+=(
        "$LEGACY_RESULTS_DIR/comparison.sha256"
        "$LEGACY_RESULTS_DIR/lineage.tsv"
    )
    for path in "$gate_run" "$gated_run" "$control_eval" "$cost_eval" "$gated_eval"; do
        [[ -d "$path/traces" && ! -L "$path/traces" ]] || {
            printf 'Trace directory is missing or symlinked: %s\n' "$path/traces" >&2
            return 1
        }
        evidence_files+=("$path/train.log" "$path/resolved-config.yaml" "$path/run.env")
        while IFS= read -r -d '' relative; do evidence_files+=("$relative"); done \
            < <(find "$path/traces" -maxdepth 1 -type f -print0)
    done
    for path in "${evidence_files[@]}"; do
        relative_files+=("$(project_relative_file "$path")")
    done
    while IFS= read -r relative; do
        path="$PROJECT_ROOT/$relative"
        checksum_lines+="$(file_sha256 "$path")  $relative"$'\n'
        sync_path "$path"
    done < <(
        printf '%s\n' "${relative_files[@]}" | LC_ALL=C sort -u
    )
    atomic_write "$results_dir/evidence.sha256" "$checksum_lines"
    sync_path "$results_dir"
    evidence_digest="$(file_sha256 "$results_dir/evidence.sha256")"
    marker_dir="$MANIFEST_DIR/cost-aware-gated"
    mkdir -p "$marker_dir"
    marker="$marker_dir/$(basename -- "$outer_attempt").ok"
    [[ ! -e "$marker" && ! -L "$marker" ]] || {
        printf 'Refusing to overwrite follow-up evidence marker: %s\n' "$marker" >&2
        return 1
    }
    atomic_write "$marker" "$evidence_digest"$'\n'
    sync_path "$marker_dir"
    atomic_write "$outer_attempt/result-contract" 'cost-aware-gated-v1'$'\n'
    atomic_write "$outer_attempt/result-root" "$results_dir"$'\n'
    atomic_write "$outer_attempt/evidence-marker" "$marker"$'\n'
    atomic_write "$outer_attempt/evidence-digest" "$evidence_digest"$'\n'
    sync_path "$outer_attempt"
    atomic_write "$FOLLOWUP_RESULTS_ROOT/latest" "$results_dir"$'\n'
    sync_path "$FOLLOWUP_RESULTS_ROOT"
}

cost_aware_gated_pipeline() {
    local outer_attempt="$1" commit="$2" handoff_digest="$3"
    local _base_model="$4" _base_digest="$5"
    local gate_run gate_checkpoint gate_digest gated_run gated_checkpoint gated_digest gated_config
    local control_eval cost_eval gated_eval results_dir result_parent
    load_historical_lineage
    verify_historical_checkpoint "$HIST_R_CHECKPOINT" "$HIST_R_DIGEST" reproduced reproduce 60
    verify_historical_checkpoint "$HIST_B_CHECKPOINT" "$HIST_B_DIGEST" control control 20
    verify_historical_checkpoint "$HIST_C_CHECKPOINT" "$HIST_C_DIGEST" cost_aware cost_aware 20

    run_job train cost_aware_gated "$BASE_GATE_STEPS" "$HIST_R_CHECKPOINT" "$HIST_R_DIGEST"
    gate_run="$LAST_RUN_DIR"
    gate_checkpoint="$(fixed_checkpoint "$gate_run" "$BASE_GATE_STEPS")"
    gate_digest="$(tree_sha256 "$gate_checkpoint")"
    record_lineage "$gate_run" cost_aware_gated_gate "$gate_checkpoint" "$gate_digest" \
        "$HIST_R_CHECKPOINT" "$HIST_R_DIGEST" "$commit" "$handoff_digest" 0.10 correct_only
    delete_gate_checkpoint "$gate_run" "$gate_checkpoint" cost_aware_gated "$BASE_GATE_STEPS"

    run_job train cost_aware_gated "$BRANCH_STEPS" "$HIST_R_CHECKPOINT" "$HIST_R_DIGEST"
    gated_run="$LAST_RUN_DIR"
    gated_checkpoint="$(fixed_checkpoint "$gated_run" "$BRANCH_STEPS")"
    gated_digest="$(tree_sha256 "$gated_checkpoint")"
    gated_config="$(file_sha256 "$gated_run/resolved-config.yaml")"
    record_lineage "$gated_run" cost_aware_gated "$gated_checkpoint" "$gated_digest" \
        "$HIST_R_CHECKPOINT" "$HIST_R_DIGEST" "$commit" "$handoff_digest" 0.10 correct_only

    run_job eval control "$HIST_B_CHECKPOINT" '' "$HIST_B_DIGEST"
    control_eval="$LAST_RUN_DIR"
    run_job eval cost_aware "$HIST_C_CHECKPOINT" '' "$HIST_C_DIGEST"
    cost_eval="$LAST_RUN_DIR"
    run_job eval cost_aware_gated "$gated_checkpoint" '' "$gated_digest"
    gated_eval="$LAST_RUN_DIR"

    result_parent="$FOLLOWUP_RESULTS_ROOT/attempts"
    mkdir -p "$result_parent"
    results_dir="$result_parent/$(basename -- "$outer_attempt")"
    [[ ! -e "$results_dir" && ! -L "$results_dir" ]] || {
        printf 'Refusing to overwrite follow-up results: %s\n' "$results_dir" >&2
        return 1
    }
    "$TRAIN_ENV/bin/python" "$CHECKOUT_DIR/scripts/autodl/paired_eval.py" \
        --control "$control_eval/traces/eval_predictions.jsonl" \
        --cost-aware-old "$cost_eval/traces/eval_predictions.jsonl" \
        --cost-aware-gated "$gated_eval/traces/eval_predictions.jsonl" \
        --output-dir "$results_dir"
    "$TRAIN_ENV/bin/python" "$CHECKOUT_DIR/scripts/autodl/export_gated_training.py" \
        --log "$gated_run/train.log" \
        --output-dir "$results_dir"
    atomic_write "$results_dir/lineage.tsv" \
        $'stage\trole\tcheckpoint\tcheckpoint_digest\tparent_checkpoint\tparent_checkpoint_digest\tcheckout_commit\tcpu_handoff_digest\tresolved_config_sha256\n'\
"R"$'\t'"reproduced"$'\t'"$HIST_R_CHECKPOINT"$'\t'"$HIST_R_DIGEST"$'\t'"$HIST_R_PARENT"$'\t'"$HIST_R_PARENT_DIGEST"$'\t'"$HIST_R_COMMIT"$'\t'"$HIST_R_HANDOFF"$'\t'"$HIST_R_CONFIG"$'\n'\
"B"$'\t'"control"$'\t'"$HIST_B_CHECKPOINT"$'\t'"$HIST_B_DIGEST"$'\t'"$HIST_B_PARENT"$'\t'"$HIST_B_PARENT_DIGEST"$'\t'"$HIST_B_COMMIT"$'\t'"$HIST_B_HANDOFF"$'\t'"$HIST_B_CONFIG"$'\n'\
"C-old"$'\t'"cost_aware"$'\t'"$HIST_C_CHECKPOINT"$'\t'"$HIST_C_DIGEST"$'\t'"$HIST_C_PARENT"$'\t'"$HIST_C_PARENT_DIGEST"$'\t'"$HIST_C_COMMIT"$'\t'"$HIST_C_HANDOFF"$'\t'"$HIST_C_CONFIG"$'\n'\
"C-gated"$'\t'"cost_aware_gated"$'\t'"$gated_checkpoint"$'\t'"$gated_digest"$'\t'"$HIST_R_CHECKPOINT"$'\t'"$HIST_R_DIGEST"$'\t'"$commit"$'\t'"$handoff_digest"$'\t'"$gated_config"$'\n'
    atomic_write "$results_dir/run-index.tsv" \
        $'stage\tmode\trun_dir\ttrace\n'\
"C-gated-gate"$'\ttrain\t'"$gate_run"$'\t'"$gate_run/traces/train_trajectories.jsonl"$'\n'\
"C-gated"$'\ttrain\t'"$gated_run"$'\t'"$gated_run/traces/train_trajectories.jsonl"$'\n'\
"B"$'\teval\t'"$control_eval"$'\t'"$control_eval/traces/eval_predictions.jsonl"$'\n'\
"C-old"$'\teval\t'"$cost_eval"$'\t'"$cost_eval/traces/eval_predictions.jsonl"$'\n'\
"C-gated"$'\teval\t'"$gated_eval"$'\t'"$gated_eval/traces/eval_predictions.jsonl"$'\n'
    verify_checkout "$commit"
    publish_followup_evidence "$outer_attempt" "$results_dir" "$gated_run" "$gate_run" \
        "$control_eval" "$cost_eval" "$gated_eval"
    printf 'Completed C-gated follow-up: %s/summary.md\n' "$results_dir"
    printf 'The script does not prove provider shutdown; confirm stopped billing in AutoDL.\n'
}

followup_main() {
    case "${1:-}" in
        --worker)
            phase_worker gpu "${2:?missing attempt directory}" "$0"
            ;;
        --action)
            require_two_gpu_followup
            gpu_action "${2:?missing attempt directory}"
            ;;
        '')
            require_two_gpu_followup
            phase_launch gpu "$0"
            ;;
        *)
            printf 'Usage: GPU_COUNT=2 AUTODL_PRICE_PER_HOUR=<price> bash %s\n' "$0" >&2
            exit 64
            ;;
    esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    followup_main "$@"
fi
