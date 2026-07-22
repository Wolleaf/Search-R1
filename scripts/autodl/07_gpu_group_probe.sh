#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
GROUP_PROBE_PROJECT_ROOT="${AUTODL_ROOT:-/root/autodl-tmp/search-r1}"
export AUTODL_RUN_BUDGET_PROFILE=gated_followup
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-500}"
export EVAL_GROUP_SIZE="${EVAL_GROUP_SIZE:-5}"
export EVAL_DATA_FILE="$GROUP_PROBE_PROJECT_ROOT/data/search_mix/probe_multi_64.parquet"
export EVAL_EXPECTED_ROWS=64
AUTODL_GPU_PIPELINE=group_probe
# Reuse GPU admission, the detached worker, BM25 lifecycle, and immutable run records.
# shellcheck source=03_gpu_run.sh
source "$SCRIPT_DIR/03_gpu_run.sh"

readonly GROUP_PROBE_RESULTS_ROOT="$RUNS_ROOT/group-probe"
readonly GROUP_PROBE_DATA_DIR="$PROJECT_ROOT/data/search_mix"
readonly GROUP_PROBE_MANIFEST="$GROUP_PROBE_DATA_DIR/manifest.json"

require_group_probe() {
    validate_gpu_inputs
    [[ "$GPU_COUNT" == 2 ]] || {
        printf 'The grouped Base probe requires GPU_COUNT=2.\n' >&2
        return 64
    }
    [[ "$TRAIN_BATCH_SIZE" == 8 && "$MAX_RESPONSE_LENGTH" == 500 &&
        "$EVAL_GROUP_SIZE" == 5 ]] || {
        printf 'Grouped Base probe is fixed to batch=8, response=500, and eval_group_size=5.\n' >&2
        return 64
    }
    [[ "$EVAL_DATA_FILE" == "$GROUP_PROBE_DATA_DIR/probe_multi_64.parquet" &&
        "$EVAL_EXPECTED_ROWS" == 64 ]] || {
        printf 'Grouped Base probe is fixed to the sealed 64-question search_mix probe.\n' >&2
        return 64
    }
}

verify_probe_data_contract() {
    local path
    local -a required=(
        "$GROUP_PROBE_MANIFEST"
        "$GROUP_PROBE_MANIFEST.sha256"
        "$GROUP_PROBE_DATA_DIR/probe_multi_64.parquet"
        "$GROUP_PROBE_DATA_DIR/catalog.jsonl"
        "$GROUP_PROBE_DATA_DIR/retrieval_replay.json"
        "$GROUP_PROBE_DATA_DIR/retrieval_replay.json.sha256"
    )
    for path in "${required[@]}"; do
        [[ -f "$path" && ! -L "$path" ]] || {
            printf 'Grouped-probe data evidence is missing or symlinked: %s\n' "$path" >&2
            return 1
        }
    done
    (
        cd "$GROUP_PROBE_DATA_DIR"
        sha256sum --strict --check manifest.json.sha256 >/dev/null
        sha256sum --strict --check retrieval_replay.json.sha256 >/dev/null
    )
    "$TRAIN_ENV/bin/python" - "$GROUP_PROBE_MANIFEST" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

manifest_path = Path(sys.argv[1]).resolve()
data_dir = manifest_path.parent
manifest = json.loads(manifest_path.read_bytes())
probe = manifest.get("artifacts", {}).get("probe", {})
catalog = manifest.get("artifacts", {}).get("catalog", {})

def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()

probe_path = data_dir / "probe_multi_64.parquet"
catalog_path = data_dir / "catalog.jsonl"
if (manifest.get("selection_policy") != "retrieval-verified-search-mix-v1"
        or probe.get("file") != probe_path.name
        or probe.get("rows") != 64
        or probe.get("sha256") != digest(probe_path)
        or catalog.get("file") != catalog_path.name
        or catalog.get("sha256") != digest(catalog_path)
        or manifest.get("overlap_checks", {}).get("probe_is_hotpot_val_subset") is not True):
    raise SystemExit("Grouped-probe manifest contract mismatch")

replay = json.loads((data_dir / "retrieval_replay.json").read_bytes())
if (replay.get("selection_policy") != manifest["selection_policy"]
        or replay.get("manifest_sha256") != digest(manifest_path)
        or replay.get("catalog_sha256") != digest(catalog_path)
        or replay.get("selected_rows") != 640):
    raise SystemExit("Grouped-probe retrieval replay contract mismatch")
PY
}

group_probe_project_relative_file() {
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

publish_group_probe_evidence() {
    local outer_attempt="$1" results_dir="$2" eval_run="$3"
    local marker_dir marker evidence_digest relative path checksum_lines=''
    local -a evidence_files relative_files required_outputs
    marker_dir="$MANIFEST_DIR/group-probe"
    marker="$marker_dir/$(basename -- "$outer_attempt").ok"
    [[ ! -e "$marker" && ! -L "$marker" ]] || {
        printf 'Refusing to overwrite grouped-probe evidence marker: %s\n' "$marker" >&2
        return 1
    }
    required_outputs=(
        summary.json summary.md go_no_go.json
        per_trajectory.jsonl per_question.jsonl lineage.tsv run-index.tsv
    )
    for path in "${required_outputs[@]}"; do
        [[ -s "$results_dir/$path" && ! -L "$results_dir/$path" ]] || {
            printf 'Required grouped-probe output is missing: %s\n' "$results_dir/$path" >&2
            return 1
        }
        evidence_files+=("$results_dir/$path")
    done
    evidence_files+=(
        "$GROUP_PROBE_MANIFEST"
        "$GROUP_PROBE_MANIFEST.sha256"
        "$GROUP_PROBE_DATA_DIR/probe_multi_64.parquet"
        "$GROUP_PROBE_DATA_DIR/catalog.jsonl"
        "$GROUP_PROBE_DATA_DIR/retrieval_replay.json"
        "$GROUP_PROBE_DATA_DIR/retrieval_replay.json.sha256"
        "$eval_run/train.log"
        "$eval_run/resolved-config.yaml"
        "$eval_run/run.env"
        "$eval_run/traces/eval_predictions.jsonl"
        "$eval_run/traces/eval_predictions.manifest.json"
        "$eval_run/traces/eval_predictions.manifest.json.sha256"
    )
    for path in "${evidence_files[@]}"; do
        relative_files+=("$(group_probe_project_relative_file "$path")")
    done
    while IFS= read -r relative; do
        path="$PROJECT_ROOT/$relative"
        checksum_lines+="$(file_sha256 "$path")  $relative"$'\n'
        sync_path "$path"
    done < <(printf '%s\n' "${relative_files[@]}" | LC_ALL=C sort -u)
    atomic_write "$results_dir/evidence.sha256" "$checksum_lines"
    sync_path "$results_dir"
    evidence_digest="$(file_sha256 "$results_dir/evidence.sha256")"

    mkdir -p "$marker_dir"
    atomic_write "$marker" "$evidence_digest"$'\n'
    sync_path "$marker_dir"
    atomic_write "$outer_attempt/result-contract" 'group-probe-v1'$'\n'
    atomic_write "$outer_attempt/result-root" "$results_dir"$'\n'
    atomic_write "$outer_attempt/evidence-marker" "$marker"$'\n'
    atomic_write "$outer_attempt/evidence-digest" "$evidence_digest"$'\n'
    sync_path "$outer_attempt"
    atomic_write "$GROUP_PROBE_RESULTS_ROOT/latest" "$results_dir"$'\n'
    sync_path "$GROUP_PROBE_RESULTS_ROOT"
}

group_probe_pipeline() {
    local outer_attempt="$1" commit="$2" handoff_digest="$3"
    local base_model="$4" base_digest="$5"
    local eval_run results_parent results_dir data_manifest_digest config_digest

    verify_probe_data_contract
    run_job eval group_probe "$base_model" '' "$base_digest"
    eval_run="$LAST_RUN_DIR"
    results_parent="$GROUP_PROBE_RESULTS_ROOT/attempts"
    mkdir -p "$results_parent"
    results_dir="$results_parent/$(basename -- "$outer_attempt")"
    [[ ! -e "$results_dir" && ! -L "$results_dir" ]] || {
        printf 'Refusing to overwrite grouped-probe results: %s\n' "$results_dir" >&2
        return 1
    }

    "$TRAIN_ENV/bin/python" "$CHECKOUT_DIR/scripts/autodl/probe_analysis.py" \
        --trace "$eval_run/traces/eval_predictions.jsonl" \
        --catalog "$GROUP_PROBE_DATA_DIR/catalog.jsonl" \
        --expected-checkpoint-digest "$base_digest" \
        --output-dir "$results_dir"
    data_manifest_digest="$(file_sha256 "$GROUP_PROBE_MANIFEST")"
    config_digest="$(file_sha256 "$eval_run/resolved-config.yaml")"
    atomic_write "$results_dir/lineage.tsv" \
        $'stage\trole\tcheckpoint\tcheckpoint_digest\tevaluation_checkout_commit\tcpu_handoff_digest\tdata_manifest_sha256\tresolved_config_sha256\n'\
"P"$'\t'"base_group_probe"$'\t'"$base_model"$'\t'"$base_digest"$'\t'"$commit"$'\t'"$handoff_digest"$'\t'"$data_manifest_digest"$'\t'"$config_digest"$'\n'
    atomic_write "$results_dir/run-index.tsv" \
        $'stage\tmode\trun_dir\ttrace\n'\
"P"$'\t'"eval"$'\t'"$eval_run"$'\t'"$eval_run/traces/eval_predictions.jsonl"$'\n'
    verify_checkout "$commit"
    publish_group_probe_evidence "$outer_attempt" "$results_dir" "$eval_run"
    printf 'Completed grouped Base probe: %s/summary.md\n' "$results_dir"
    printf 'GO or NO-GO is a scientific result; both complete successfully and retain evidence.\n'
    printf 'Guest shutdown is only a dispatch; confirm stopped billing in AutoDL.\n'
}

group_probe_main() {
    case "${1:-}" in
        --worker)
            phase_worker gpu "${2:?missing attempt directory}" "$0"
            ;;
        --action)
            require_group_probe
            gpu_action "${2:?missing attempt directory}"
            ;;
        '')
            require_group_probe
            phase_launch gpu "$0"
            ;;
        *)
            printf 'Usage: GPU_COUNT=2 AUTODL_PRICE_PER_HOUR=<price> bash %s\n' "$0" >&2
            exit 64
            ;;
    esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    group_probe_main "$@"
fi
