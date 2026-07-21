#!/usr/bin/env bash
set -Eeuo pipefail

AUTODL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
ROOT="$(mktemp -d)"
trap 'rm -rf -- "$ROOT"' EXIT
mkdir -p "$ROOT/runs/comparison"

COMMIT="$(printf 'a%.0s' {1..40})"
DIGEST="$(printf 'b%.0s' {1..64})"
HANDOFF="$(printf 'c%.0s' {1..64})"
BASE="$ROOT/models/base"
R="$ROOT/runs/reproduce/attempts/r/checkpoints/actor/global_step_60"
B="$ROOT/runs/control/attempts/b/checkpoints/actor/global_step_20"
C="$ROOT/runs/cost_aware/attempts/c/checkpoints/actor/global_step_20"
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
mkdir -p "$ROOT/manifests"
sha256sum "$ROOT/runs/comparison/comparison.sha256" | cut -d' ' -f1 >"$ROOT/manifests/gpu.ok"

AUTODL_ROOT="$ROOT" GPU_COUNT=2 AUTODL_PRICE_PER_HOUR=5.76 \
    source "$AUTODL_DIR/05_gpu_cost_aware_gated.sh"
load_historical_lineage
[[ "$HIST_R_CHECKPOINT" == "$R" && "$HIST_B_CHECKPOINT" == "$B" &&
    "$HIST_C_CHECKPOINT" == "$C" ]]

printf 'tampered\n' >>"$ROOT/runs/comparison/lineage.tsv"
if load_historical_lineage >/dev/null 2>&1; then
    printf 'Tampered historical lineage was accepted.\n' >&2
    exit 1
fi

printf 'C-gated follow-up lineage tests passed.\n'
