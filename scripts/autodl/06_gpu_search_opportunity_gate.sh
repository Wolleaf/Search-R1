#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SEARCH_GATE_PROJECT_ROOT="${AUTODL_ROOT:-/root/autodl-tmp/search-r1}"
export AUTODL_RUN_BUDGET_PROFILE=gated_followup
export EVAL_DATA_FILE="$SEARCH_GATE_PROJECT_ROOT/data/search_opportunity_gate/eval_256.parquet"
export EVAL_EXPECTED_ROWS=256
AUTODL_GPU_PIPELINE=search_opportunity_gate
# Reuse the validated GPU admission, detached worker, BM25 lifecycle, and run records.
# shellcheck source=03_gpu_run.sh
source "$SCRIPT_DIR/03_gpu_run.sh"

readonly SEARCH_GATE_RESULTS_ROOT="$RUNS_ROOT/search-opportunity-gate"
readonly LEGACY_RESULTS_DIR="$RUNS_ROOT/comparison"
readonly SEARCH_GATE_DATA_DIR="$PROJECT_ROOT/data/search_opportunity_gate"

require_search_gate() {
    validate_gpu_inputs
    [[ "$TRAIN_BATCH_SIZE" == 8 && "$MAX_RESPONSE_LENGTH" == 256 ]] || {
        printf 'Search-opportunity evaluation keeps TRAIN_BATCH_SIZE=8 and MAX_RESPONSE_LENGTH=256.\n' >&2
        return 64
    }
    [[ "$EVAL_DATA_FILE" == "$SEARCH_GATE_DATA_DIR/eval_256.parquet" &&
        "$EVAL_EXPECTED_ROWS" == 256 ]] || {
        printf 'Search-opportunity data identity is fixed to the sealed 256-row set.\n' >&2
        return 64
    }
}

load_control_checkpoint() {
    local header stage role checkpoint digest parent parent_digest commit handoff config extra
    local rows=0 legacy_digest run_dir suffix resolved actual_digest
    [[ -d "$LEGACY_RESULTS_DIR" && ! -L "$LEGACY_RESULTS_DIR" ]] || {
        printf 'Historical comparison is missing: %s\n' "$LEGACY_RESULTS_DIR" >&2
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
        rows=$((rows + 1))
        if [[ "$stage:$role" == B:control ]]; then
            [[ -z "${extra:-}" && "$digest" =~ ^[0-9a-f]{64}$ &&
                "$parent_digest" =~ ^[0-9a-f]{64}$ && "$commit" =~ ^[0-9a-f]{40}$ &&
                "$handoff" =~ ^[0-9a-f]{64}$ && "$config" =~ ^[0-9a-f]{64}$ ]] || {
                printf 'Historical B lineage row is malformed.\n' >&2
                return 1
            }
            GATE_B_CHECKPOINT="$checkpoint"
            GATE_B_DIGEST="$digest"
            GATE_B_PARENT="$parent"
            GATE_B_PARENT_DIGEST="$parent_digest"
            GATE_B_COMMIT="$commit"
            GATE_B_HANDOFF="$handoff"
            GATE_B_CONFIG="$config"
        fi
    done < <(tail -n +2 "$LEGACY_RESULTS_DIR/lineage.tsv")
    [[ "$rows" == 4 && -n "${GATE_B_CHECKPOINT:-}" ]] || {
        printf 'Historical comparison must contain exactly one complete A/R/B/C lineage.\n' >&2
        return 1
    }

    suffix='/checkpoints/actor/global_step_20'
    [[ "$GATE_B_CHECKPOINT" == "$RUNS_ROOT/control/attempts/"*"$suffix" ]] || {
        printf 'Historical B checkpoint has an unexpected path: %s\n' "$GATE_B_CHECKPOINT" >&2
        return 1
    }
    run_dir="${GATE_B_CHECKPOINT%$suffix}"
    resolved="$(fixed_checkpoint "$run_dir" 20)"
    [[ "$resolved" == "$GATE_B_CHECKPOINT" ]] || {
        printf 'Historical B checkpoint resolved to a different path.\n' >&2
        return 1
    }
    grep -Fxq 'role=control' "$run_dir/run.env" \
        && grep -Fxq "checkpoint=$GATE_B_CHECKPOINT" "$run_dir/run.env" \
        && grep -Fxq "checkpoint_digest=$GATE_B_DIGEST" "$run_dir/run.env" || {
        printf 'Historical B run metadata is inconsistent.\n' >&2
        return 1
    }
    actual_digest="$(tree_sha256 "$GATE_B_CHECKPOINT")"
    [[ "$actual_digest" == "$GATE_B_DIGEST" ]] || {
        printf 'Historical B checkpoint digest mismatch.\n' >&2
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

publish_search_gate_evidence() {
    local outer_attempt="$1" results_dir="$2" eval_run="$3"
    local marker_dir marker evidence_digest relative path checksum_lines=''
    local -a evidence_files relative_files required_outputs
    required_outputs=(
        summary.json summary.md go_no_go.json per_question.jsonl
        correct_questions.csv wrong_questions.csv two_plus_search.csv
        redundant_search_candidates.csv strata.csv lineage.tsv run-index.tsv
    )
    for path in "${required_outputs[@]}"; do
        [[ -s "$results_dir/$path" && ! -L "$results_dir/$path" ]] || {
            printf 'Required search-gate output is missing: %s\n' "$results_dir/$path" >&2
            return 1
        }
        evidence_files+=("$results_dir/$path")
    done
    evidence_files+=(
        "$SEARCH_GATE_DATA_DIR/eval_256.parquet"
        "$SEARCH_GATE_DATA_DIR/catalog.jsonl"
        "$SEARCH_GATE_DATA_DIR/manifest.json"
        "$SEARCH_GATE_DATA_DIR/manifest.json.sha256"
        "$SEARCH_GATE_DATA_DIR/sources/hotpotqa/dev.jsonl"
        "$SEARCH_GATE_DATA_DIR/sources/2wikimultihopqa/dev.jsonl"
        "$LEGACY_RESULTS_DIR/comparison.sha256"
        "$LEGACY_RESULTS_DIR/lineage.tsv"
        "$eval_run/train.log"
        "$eval_run/resolved-config.yaml"
        "$eval_run/run.env"
        "$eval_run/traces/eval_predictions.jsonl"
        "$eval_run/traces/eval_predictions.manifest.json"
        "$eval_run/traces/eval_predictions.manifest.json.sha256"
    )
    for path in "${evidence_files[@]}"; do
        relative_files+=("$(project_relative_file "$path")")
    done
    while IFS= read -r relative; do
        path="$PROJECT_ROOT/$relative"
        checksum_lines+="$(file_sha256 "$path")  $relative"$'\n'
        sync_path "$path"
    done < <(printf '%s\n' "${relative_files[@]}" | LC_ALL=C sort -u)
    atomic_write "$results_dir/evidence.sha256" "$checksum_lines"
    sync_path "$results_dir"
    evidence_digest="$(file_sha256 "$results_dir/evidence.sha256")"

    marker_dir="$MANIFEST_DIR/search-opportunity-gate"
    mkdir -p "$marker_dir"
    marker="$marker_dir/$(basename -- "$outer_attempt").ok"
    [[ ! -e "$marker" && ! -L "$marker" ]] || {
        printf 'Refusing to overwrite search-gate evidence marker: %s\n' "$marker" >&2
        return 1
    }
    atomic_write "$marker" "$evidence_digest"$'\n'
    sync_path "$marker_dir"
    atomic_write "$outer_attempt/result-contract" 'search-opportunity-gate-v1'$'\n'
    atomic_write "$outer_attempt/result-root" "$results_dir"$'\n'
    atomic_write "$outer_attempt/evidence-marker" "$marker"$'\n'
    atomic_write "$outer_attempt/evidence-digest" "$evidence_digest"$'\n'
    sync_path "$outer_attempt"
    atomic_write "$SEARCH_GATE_RESULTS_ROOT/latest" "$results_dir"$'\n'
    sync_path "$SEARCH_GATE_RESULTS_ROOT"
}

search_opportunity_gate_pipeline() {
    local outer_attempt="$1" commit="$2" handoff_digest="$3"
    local _base_model="$4" _base_digest="$5"
    local eval_run results_parent results_dir data_manifest_digest

    "$TRAIN_ENV/bin/python" "$CHECKOUT_DIR/scripts/data_process/multihop_search_gate.py" verify \
        --manifest "$SEARCH_GATE_DATA_DIR/manifest.json"
    data_manifest_digest="$(file_sha256 "$SEARCH_GATE_DATA_DIR/manifest.json")"
    load_control_checkpoint

    run_job eval search_opportunity "$GATE_B_CHECKPOINT" '' "$GATE_B_DIGEST"
    eval_run="$LAST_RUN_DIR"
    results_parent="$SEARCH_GATE_RESULTS_ROOT/attempts"
    mkdir -p "$results_parent"
    results_dir="$results_parent/$(basename -- "$outer_attempt")"
    [[ ! -e "$results_dir" && ! -L "$results_dir" ]] || {
        printf 'Refusing to overwrite search-gate results: %s\n' "$results_dir" >&2
        return 1
    }

    "$TRAIN_ENV/bin/python" "$CHECKOUT_DIR/scripts/autodl/search_opportunity_gate.py" \
        --trace "$eval_run/traces/eval_predictions.jsonl" \
        --catalog "$SEARCH_GATE_DATA_DIR/catalog.jsonl" \
        --data-manifest "$SEARCH_GATE_DATA_DIR/manifest.json" \
        --expected-checkpoint-digest "$GATE_B_DIGEST" \
        --output-dir "$results_dir"
    atomic_write "$results_dir/lineage.tsv" \
        $'role\tcheckpoint\tcheckpoint_digest\tparent_checkpoint\tparent_checkpoint_digest\ttraining_checkout_commit\ttraining_handoff_digest\ttraining_config_sha256\tevaluation_checkout_commit\tevaluation_handoff_digest\tdata_manifest_sha256\n'\
"B"$'\t'"$GATE_B_CHECKPOINT"$'\t'"$GATE_B_DIGEST"$'\t'"$GATE_B_PARENT"$'\t'"$GATE_B_PARENT_DIGEST"$'\t'"$GATE_B_COMMIT"$'\t'"$GATE_B_HANDOFF"$'\t'"$GATE_B_CONFIG"$'\t'"$commit"$'\t'"$handoff_digest"$'\t'"$data_manifest_digest"$'\n'
    atomic_write "$results_dir/run-index.tsv" \
        $'stage\tmode\trun_dir\ttrace\n'\
"B-multihop"$'\t'"eval"$'\t'"$eval_run"$'\t'"$eval_run/traces/eval_predictions.jsonl"$'\n'
    verify_checkout "$commit"
    publish_search_gate_evidence "$outer_attempt" "$results_dir" "$eval_run"
    printf 'Completed search-opportunity gate: %s/summary.md\n' "$results_dir"
    printf 'GO or NO-GO is a scientific result; both complete successfully and retain evidence.\n'
    printf 'The script does not prove provider shutdown; confirm stopped billing in AutoDL.\n'
}

search_gate_main() {
    case "${1:-}" in
        --worker)
            phase_worker gpu "${2:?missing attempt directory}" "$0"
            ;;
        --action)
            require_search_gate
            gpu_action "${2:?missing attempt directory}"
            ;;
        '')
            require_search_gate
            phase_launch gpu "$0"
            ;;
        *)
            printf 'Usage: GPU_COUNT={1|2} AUTODL_PRICE_PER_HOUR=<price> bash %s\n' "$0" >&2
            exit 64
            ;;
    esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    search_gate_main "$@"
fi
