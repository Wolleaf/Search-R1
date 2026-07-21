#!/usr/bin/env bash
set -Eeuo pipefail

AUTODL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
CHECKOUT="$(cd -- "$AUTODL_DIR/../.." && pwd -P)"
ROOT="$(mktemp -d)"
trap 'rm -rf -- "$ROOT"' EXIT

mkdir -p "$ROOT/envs/train/bin" "$ROOT/models/Qwen3.5-2B" "$ROOT/data/nq_small" \
    "$ROOT/parent-checkpoint" "$ROOT/output" "$ROOT/traces"
ln -s "$CHECKOUT" "$ROOT/checkout"
cat >"$ROOT/envs/train/bin/python" <<'SH'
#!/usr/bin/env bash
printf '%s\n' "$@" >"$CAPTURE_LOG"
SH
chmod 0755 "$ROOT/envs/train/bin/python"

DIGEST="$(printf 'a%.0s' {1..64})"
CAPTURE="$ROOT/args.txt"
COMMON=(
    AUTODL_CONFIG_ONLY=1 AUTODL_ROOT="$ROOT" GPU_COUNT=2
    OUTPUT_DIR="$ROOT/output" CAPTURE_LOG="$CAPTURE"
)

env "${COMMON[@]}" \
    TRACE_OUTPUT_DIR="$ROOT/traces" \
    TRACE_STAGE=cost_aware_gated TRACE_RUN_ID=test-run \
    TRACE_PARENT_CHECKPOINT_DIGEST="$DIGEST" \
    bash "$AUTODL_DIR/train_small_grpo.sh" \
    train cost_aware_gated 20 "$ROOT/parent-checkpoint"
grep -Fxq -- 'algorithm.cost_lambda=0.10' "$CAPTURE"
grep -Fxq -- '++algorithm.cost_reward_mode=correct_only' "$CAPTURE"
grep -Fxq -- 'actor_rollout_ref.rollout.n_agent=5' "$CAPTURE"
grep -Fxq -- 'data.train_batch_size=8' "$CAPTURE"
grep -Fxq -- "++trainer.trace_parent_checkpoint_digest=$DIGEST" "$CAPTURE"

env "${COMMON[@]}" \
    TRACE_OUTPUT_DIR="$ROOT/traces" \
    TRACE_STAGE=control TRACE_RUN_ID=test-eval \
    TRACE_CHECKPOINT_DIGEST="$DIGEST" \
    bash "$AUTODL_DIR/train_small_grpo.sh" \
    eval control "$ROOT/parent-checkpoint"
grep -Fxq -- '++algorithm.cost_reward_mode=linear' "$CAPTURE"
grep -Fxq -- "++trainer.trace_checkpoint_digest=$DIGEST" "$CAPTURE"

if env "${COMMON[@]}" \
    TRACE_OUTPUT_DIR="$ROOT/traces" TRACE_STAGE=control TRACE_RUN_ID=missing-digest \
    bash "$AUTODL_DIR/train_small_grpo.sh" \
    eval control "$ROOT/parent-checkpoint" >/dev/null 2>&1; then
    printf 'Trace-only evaluation accepted a missing checkpoint digest.\n' >&2
    exit 1
fi

env "${COMMON[@]}" \
    bash "$AUTODL_DIR/train_small_grpo.sh" \
    train control 20 "$ROOT/parent-checkpoint"
grep -Fxq -- '++algorithm.cost_reward_mode=linear' "$CAPTURE"
! grep -Fq -- 'trainer.trace_output_dir' "$CAPTURE"

printf 'C-gated config tests passed.\n'
