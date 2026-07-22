#!/usr/bin/env bash
set -Eeuo pipefail

AUTODL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
ROOT="$(mktemp -d)"
trap 'rm -rf -- "$ROOT"' EXIT
mkdir -p "$ROOT/runs/comparison" "$ROOT/manifests" \
    "$ROOT/data/search_opportunity_gate"

COMMIT="$(printf 'a%.0s' {1..40})"
DIGEST="$(printf 'b%.0s' {1..64})"
HANDOFF="$(printf 'c%.0s' {1..64})"
BASE="$ROOT/models/base"
R="$ROOT/runs/reproduce/attempts/r/checkpoints/actor/global_step_60"
B="$ROOT/runs/control/attempts/b/checkpoints/actor/global_step_20"
C="$ROOT/runs/cost_aware/attempts/c/checkpoints/actor/global_step_20"
mkdir -p "$B"
printf 'placeholder\n' >"$ROOT/runs/comparison/results.csv"
printf 'placeholder\n' >"$ROOT/runs/comparison/results.md"
printf '%s\n' \
    $'stage\trole\tcheckpoint\tcheckpoint_digest\tparent_checkpoint\tparent_checkpoint_digest\tcheckout_commit\tcpu_handoff_digest\tresolved_config_sha256' \
    "A"$'\t'"base"$'\t'"$BASE"$'\t'"$DIGEST"$'\t-\t-\t'"$COMMIT"$'\t'"$HANDOFF"$'\t-' \
    "R"$'\t'"reproduced"$'\t'"$R"$'\t'"$DIGEST"$'\t'"$BASE"$'\t'"$DIGEST"$'\t'"$COMMIT"$'\t'"$HANDOFF"$'\t'"$DIGEST" \
    "B"$'\t'"control"$'\t'"$B"$'\t'"$DIGEST"$'\t'"$R"$'\t'"$DIGEST"$'\t'"$COMMIT"$'\t'"$HANDOFF"$'\t'"$DIGEST" \
    "C"$'\t'"cost_aware"$'\t'"$C"$'\t'"$DIGEST"$'\t'"$R"$'\t'"$DIGEST"$'\t'"$COMMIT"$'\t'"$HANDOFF"$'\t'"$DIGEST" \
    >"$ROOT/runs/comparison/lineage.tsv"
(
    cd "$ROOT/runs/comparison"
    sha256sum results.csv results.md lineage.tsv >comparison.sha256
)
sha256sum "$ROOT/runs/comparison/comparison.sha256" | cut -d' ' -f1 \
    >"$ROOT/manifests/gpu.ok"
run_dir="${B%/checkpoints/actor/global_step_20}"
mkdir -p "$run_dir"
printf '%s\n' 'role=control' "checkpoint=$B" "checkpoint_digest=$DIGEST" \
    >"$run_dir/run.env"

AUTODL_ROOT="$ROOT"
GPU_COUNT=2
AUTODL_PRICE_PER_HOUR=5.76
source "$AUTODL_DIR/06_gpu_search_opportunity_gate.sh"
fixed_checkpoint() { printf '%s\n' "$B"; }
tree_sha256() { printf '%s\n' "$DIGEST"; }

require_search_gate
load_control_checkpoint
[[ "$GATE_B_CHECKPOINT" == "$B" && "$GATE_B_DIGEST" == "$DIGEST" &&
    "$GATE_B_PARENT" == "$R" ]]

printf 'tampered\n' >>"$ROOT/runs/comparison/lineage.tsv"
if load_control_checkpoint >/dev/null 2>&1; then
    printf 'Tampered historical lineage was accepted by the search gate.\n' >&2
    exit 1
fi

printf 'Search-opportunity pipeline tests passed.\n'
