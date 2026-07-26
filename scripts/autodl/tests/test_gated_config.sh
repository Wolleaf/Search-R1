#!/usr/bin/env bash
set -Eeuo pipefail

AUTODL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
CHECKOUT="$(cd -- "$AUTODL_DIR/../.." && pwd -P)"
ROOT="$(mktemp -d)"
trap 'rm -rf -- "$ROOT"' EXIT

mkdir -p "$ROOT/envs/train/bin" "$ROOT/models/Qwen3.5-2B" "$ROOT/data/nq_small" \
    "$ROOT/data/search_mix" "$ROOT/data/search_mix_qwen35_native_v3" \
    "$ROOT/parent-checkpoint" "$ROOT/output" "$ROOT/traces"
printf 'parquet-placeholder\n' >"$ROOT/data/search-opportunity.parquet"
printf 'parquet-placeholder\n' >"$ROOT/data/group-probe.parquet"
for file in probe_autonomous_16.parquet val_128.parquet \
    nq_test_128_native_v3.parquet multihop_eval_256_native_v3.parquet; do
    printf 'parquet-placeholder\n' >"$ROOT/data/search_mix_qwen35_native_v3/$file"
done
ln -s "$CHECKOUT" "$ROOT/checkout"
cat >"$ROOT/envs/train/bin/python" <<'SH'
#!/usr/bin/env bash
printf '%s\n' "$@" >"$CAPTURE_LOG"
SH
chmod 0755 "$ROOT/envs/train/bin/python"

DIGEST="$(printf 'a%.0s' {1..64})"
CAPTURE="$ROOT/args.txt"
unset DATA_DIR
COMMON=(
    AUTODL_CONFIG_ONLY=1 AUTODL_ROOT="$ROOT" GPU_COUNT=2
    OUTPUT_DIR="$ROOT/output" CAPTURE_LOG="$CAPTURE"
)

# The archived 03 workflow stays on its original response budget; 07 overrides it explicitly.
(
    unset MAX_RESPONSE_LENGTH
    AUTODL_ROOT="$ROOT"
    GPU_COUNT=2
    AUTODL_PRICE_PER_HOUR=5.76
    source "$AUTODL_DIR/03_gpu_run.sh"
    [[ "$MAX_RESPONSE_LENGTH" == 256 ]]
    MAX_RESPONSE_LENGTH=500
    if validate_gpu_inputs >/dev/null 2>&1; then
        printf 'Legacy 03 accepted the grouped-probe response length.\n' >&2
        exit 1
    fi
    MAX_RESPONSE_LENGTH=192
    validate_gpu_inputs
    mkdir -p "$MANIFEST_DIR"
    : >"$MANIFEST_DIR/cpu-seal.pending.json"
    if gpu_action "$ROOT/pending-attempt" >/dev/null 2>&1; then
        printf 'GPU admission accepted an unresolved CPU seal transaction.\n' >&2
        exit 1
    fi
    rm -f -- "$MANIFEST_DIR/cpu-seal.pending.json"
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
grep -Fxq -- 'actor_rollout_ref.rollout.top_k=0' "$CAPTURE"
grep -Fxq -- 'actor_rollout_ref.rollout.presence_penalty=0.0' "$CAPTURE"
grep -Fxq -- 'data.return_raw_chat=false' "$CAPTURE"
grep -Fxq -- '++tool_protocol=legacy_xml' "$CAPTURE"
grep -Fxq -- 'data.train_batch_size=8' "$CAPTURE"
grep -Fxq -- 'data.max_response_length=500' "$CAPTURE"
grep -Fxq -- 'data.max_prompt_length=4096' "$CAPTURE"
grep -Fxq -- "++trainer.trace_parent_checkpoint_digest=$DIGEST" "$CAPTURE"
grep -Fxq -- "data.train_files=$ROOT/data/nq_small/train_512.parquet" "$CAPTURE"

env "${COMMON[@]}" MAX_RESPONSE_LENGTH=384 \
    bash "$AUTODL_DIR/train_small_grpo.sh" \
    train control 20 "$ROOT/parent-checkpoint"
grep -Fxq -- 'data.max_response_length=384' "$CAPTURE"
grep -Fxq -- 'data.max_prompt_length=4096' "$CAPTURE"

env "${COMMON[@]}" MAX_RESPONSE_LENGTH=256 \
    bash "$AUTODL_DIR/train_small_grpo.sh" \
    train control 20 "$ROOT/parent-checkpoint"
grep -Fxq -- 'data.max_response_length=256' "$CAPTURE"
grep -Fxq -- 'data.max_prompt_length=3584' "$CAPTURE"

if env "${COMMON[@]}" MAX_RESPONSE_LENGTH=499 \
    bash "$AUTODL_DIR/train_small_grpo.sh" \
    train control 20 "$ROOT/parent-checkpoint" >/dev/null 2>&1; then
    printf 'Training accepted an unsupported response length.\n' >&2
    exit 1
fi

env "${COMMON[@]}" \
    TRACE_OUTPUT_DIR="$ROOT/traces" \
    TRACE_STAGE=control TRACE_RUN_ID=test-eval \
    TRACE_CHECKPOINT_DIGEST="$DIGEST" \
    bash "$AUTODL_DIR/train_small_grpo.sh" \
    eval control "$ROOT/parent-checkpoint"
grep -Fxq -- '++algorithm.cost_reward_mode=linear' "$CAPTURE"
grep -Fxq -- "++trainer.trace_checkpoint_digest=$DIGEST" "$CAPTURE"
grep -Fxq -- 'data.eval_group_size=1' "$CAPTURE"
grep -Fxq -- 'data.val_batch_size=64' "$CAPTURE"

env "${COMMON[@]}" \
    EVAL_DATA_FILE="$ROOT/data/search-opportunity.parquet" \
    TRACE_OUTPUT_DIR="$ROOT/traces" \
    TRACE_STAGE=search_opportunity TRACE_RUN_ID=search-gate-eval \
    TRACE_CHECKPOINT_DIGEST="$DIGEST" \
    bash "$AUTODL_DIR/train_small_grpo.sh" \
    eval search_opportunity "$ROOT/parent-checkpoint"
grep -Fxq -- "data.val_files=$ROOT/data/search-opportunity.parquet" "$CAPTURE"
grep -Fxq -- 'max_turns=4' "$CAPTURE"

env "${COMMON[@]}" \
    DATA_DIR="$ROOT/data/search_mix" \
    EVAL_DATA_FILE="$ROOT/data/group-probe.parquet" \
    EVAL_GROUP_SIZE=5 \
    TRACE_OUTPUT_DIR="$ROOT/traces" \
    TRACE_STAGE=group_probe TRACE_RUN_ID=group-probe-eval \
    TRACE_CHECKPOINT_DIGEST="$DIGEST" \
    bash "$AUTODL_DIR/train_small_grpo.sh" \
    eval group_probe "$ROOT/models/Qwen3.5-2B"
grep -Fxq -- "data.val_files=$ROOT/data/group-probe.parquet" "$CAPTURE"
grep -Fxq -- 'data.eval_group_size=5' "$CAPTURE"
grep -Fxq -- 'data.val_batch_size=8' "$CAPTURE"
grep -Fxq -- 'data.max_response_length=500' "$CAPTURE"
grep -Fxq -- 'data.max_prompt_length=4096' "$CAPTURE"
grep -Fxq -- "data.train_files=$ROOT/data/search_mix/train_512.parquet" "$CAPTURE"

env "${COMMON[@]}" \
    DATA_DIR="$ROOT/data/search_mix_qwen35_native_v3" \
    TOOL_PROTOCOL=qwen35_native \
    EVAL_DATA_FILE="$ROOT/data/search_mix_qwen35_native_v3/probe_autonomous_16.parquet" \
    EVAL_GROUP_SIZE=2 \
    bash "$AUTODL_DIR/train_small_grpo.sh" \
    eval qwen_native_g1 "$ROOT/models/Qwen3.5-2B"
grep -Fxq -- "data.train_files=$ROOT/data/search_mix_qwen35_native_v3/train_512.parquet" "$CAPTURE"
grep -Fxq -- 'actor_rollout_ref.rollout.top_k=0' "$CAPTURE"
grep -Fxq -- 'actor_rollout_ref.rollout.presence_penalty=0.0' "$CAPTURE"
grep -Fxq -- '++tool_protocol=qwen35_native' "$CAPTURE"
grep -Fxq -- 'data.max_prompt_length=4500' "$CAPTURE"
grep -Fxq -- 'actor_rollout_ref.rollout.do_sample=true' "$CAPTURE"

for spec in \
    'qwen_native_a_val|val_128.parquet' \
    'qwen_native_r_val|val_128.parquet' \
    'qwen_native_b_val|val_128.parquet' \
    'qwen_native_c_val|val_128.parquet' \
    'qwen_native_a_nq_test|nq_test_128_native_v3.parquet' \
    'qwen_native_r_nq_test|nq_test_128_native_v3.parquet' \
    'qwen_native_b_nq_test|nq_test_128_native_v3.parquet' \
    'qwen_native_c_nq_test|nq_test_128_native_v3.parquet' \
    'qwen_native_a_multihop|multihop_eval_256_native_v3.parquet' \
    'qwen_native_r_multihop|multihop_eval_256_native_v3.parquet' \
    'qwen_native_b_multihop|multihop_eval_256_native_v3.parquet' \
    'qwen_native_c_multihop|multihop_eval_256_native_v3.parquet'; do
    IFS='|' read -r endpoint_variant endpoint_file <<<"$spec"
    env "${COMMON[@]}" \
        DATA_DIR="$ROOT/data/search_mix_qwen35_native_v3" \
        TOOL_PROTOCOL=qwen35_native \
        bash "$AUTODL_DIR/train_small_grpo.sh" \
        eval "$endpoint_variant" "$ROOT/parent-checkpoint"
    grep -Fxq -- \
        "data.val_files=$ROOT/data/search_mix_qwen35_native_v3/$endpoint_file" \
        "$CAPTURE"
    grep -Fxq -- 'data.eval_group_size=1' "$CAPTURE"
    grep -Fxq -- 'actor_rollout_ref.rollout.do_sample=false' "$CAPTURE"
done

env "${COMMON[@]}" \
    DATA_DIR="$ROOT/data/search_mix_qwen35_native_v3" \
    TOOL_PROTOCOL=qwen35_native \
    bash "$AUTODL_DIR/train_small_grpo.sh" train smoke 2
grep -Fxq -- "data.train_files=$ROOT/data/search_mix_qwen35_native_v3/train_512.parquet" "$CAPTURE"
grep -Fxq -- "data.val_files=$ROOT/data/search_mix_qwen35_native_v3/val_128.parquet" "$CAPTURE"
grep -Fxq -- 'data.train_batch_size=8' "$CAPTURE"
grep -Fxq -- 'data.val_batch_size=8' "$CAPTURE"
grep -Fxq -- 'data.max_response_length=500' "$CAPTURE"
grep -Fxq -- 'data.max_obs_length=500' "$CAPTURE"
grep -Fxq -- 'data.max_prompt_length=4500' "$CAPTURE"
grep -Fxq -- 'actor_rollout_ref.rollout.n_agent=5' "$CAPTURE"
grep -Fxq -- 'actor_rollout_ref.actor.ppo_mini_batch_size=40' "$CAPTURE"
grep -Fxq -- 'actor_rollout_ref.actor.ppo_micro_batch_size=2' "$CAPTURE"
grep -Fxq -- 'actor_rollout_ref.rollout.top_k=0' "$CAPTURE"
grep -Fxq -- 'actor_rollout_ref.rollout.min_p=0.0' "$CAPTURE"
grep -Fxq -- 'actor_rollout_ref.rollout.presence_penalty=0.0' "$CAPTURE"
grep -Fxq -- 'actor_rollout_ref.rollout.repetition_penalty=1.0' "$CAPTURE"
grep -Fxq -- 'actor_rollout_ref.rollout.do_sample=true' "$CAPTURE"
grep -Fxq -- 'data.return_raw_chat=true' "$CAPTURE"
grep -Fxq -- '++qwen35_prompt_version=qwen35-native-search-v3-original-aligned' "$CAPTURE"
grep -Fxq -- '++trainer.native_training_variant=smoke' "$CAPTURE"

env "${COMMON[@]}" \
    DATA_DIR="$ROOT/data/search_mix_qwen35_native_v3" \
    TOOL_PROTOCOL=qwen35_native \
    bash "$AUTODL_DIR/train_small_grpo.sh" train reproduce 60
grep -Fxq -- 'trainer.total_training_steps=60' "$CAPTURE"
grep -Fxq -- "actor_rollout_ref.model.path=$ROOT/models/Qwen3.5-2B" "$CAPTURE"
grep -Fxq -- '++trainer.native_training_variant=reproduce' "$CAPTURE"

env "${COMMON[@]}" \
    DATA_DIR="$ROOT/data/search_mix_qwen35_native_v3" \
    TOOL_PROTOCOL=qwen35_native \
    bash "$AUTODL_DIR/train_small_grpo.sh" \
    train control 20 "$ROOT/parent-checkpoint"
grep -Fxq -- 'algorithm.cost_lambda=0.0' "$CAPTURE"
grep -Fxq -- '++algorithm.cost_reward_mode=linear' "$CAPTURE"
grep -Fxq -- "actor_rollout_ref.model.path=$ROOT/parent-checkpoint" "$CAPTURE"
grep -Fxq -- '++trainer.native_training_variant=control' "$CAPTURE"

env "${COMMON[@]}" \
    DATA_DIR="$ROOT/data/search_mix_qwen35_native_v3" \
    TOOL_PROTOCOL=qwen35_native \
    bash "$AUTODL_DIR/train_small_grpo.sh" \
    train cost_aware_gated 20 "$ROOT/parent-checkpoint"
grep -Fxq -- 'algorithm.cost_lambda=0.10' "$CAPTURE"
grep -Fxq -- '++algorithm.cost_reward_mode=correct_only' "$CAPTURE"
grep -Fxq -- '++trainer.native_training_variant=cost_aware_gated' "$CAPTURE"

expect_exit_64() {
    set +e
    "$@" >/dev/null 2>&1
    local rc=$?
    set -e
    if ((rc != 64)); then
        printf 'Expected exit 64, got %s: %s\n' "$rc" "$*" >&2
        exit 1
    fi
}

expect_exit_64 env "${COMMON[@]}" \
    DATA_DIR="$ROOT/data/search_mix" \
    bash "$AUTODL_DIR/train_small_grpo.sh" \
    train control 20 "$ROOT/parent-checkpoint"
expect_exit_64 env "${COMMON[@]}" \
    DATA_DIR="$ROOT/data/nq_small" \
    EVAL_DATA_FILE="$ROOT/data/group-probe.parquet" EVAL_GROUP_SIZE=5 \
    bash "$AUTODL_DIR/train_small_grpo.sh" \
    eval group_probe "$ROOT/models/Qwen3.5-2B"
expect_exit_64 env "${COMMON[@]}" \
    DATA_DIR="$ROOT/data/search_mix" TOOL_PROTOCOL=qwen35_native \
    EVAL_DATA_FILE="$ROOT/data/search_mix_qwen35_native_v3/probe_autonomous_16.parquet" \
    EVAL_GROUP_SIZE=2 \
    bash "$AUTODL_DIR/train_small_grpo.sh" \
    eval qwen_native_g1 "$ROOT/models/Qwen3.5-2B"
expect_exit_64 env "${COMMON[@]}" \
    DATA_DIR="$ROOT/data/search_mix_qwen35_native_v3" TOOL_PROTOCOL=qwen35_native \
    bash "$AUTODL_DIR/train_small_grpo.sh" \
    train reproduce 2
expect_exit_64 env "${COMMON[@]}" \
    DATA_DIR="$ROOT/data/search_mix_qwen35_native_v3" TOOL_PROTOCOL=qwen35_native \
    bash "$AUTODL_DIR/train_small_grpo.sh" \
    train cost_aware 20 "$ROOT/parent-checkpoint"
expect_exit_64 env "${COMMON[@]}" GPU_COUNT=1 \
    DATA_DIR="$ROOT/data/search_mix_qwen35_native_v3" TOOL_PROTOCOL=qwen35_native \
    bash "$AUTODL_DIR/train_small_grpo.sh" train smoke 2
expect_exit_64 env "${COMMON[@]}" TRAIN_BATCH_SIZE=4 \
    DATA_DIR="$ROOT/data/search_mix_qwen35_native_v3" TOOL_PROTOCOL=qwen35_native \
    bash "$AUTODL_DIR/train_small_grpo.sh" train smoke 2
expect_exit_64 env "${COMMON[@]}" MAX_RESPONSE_LENGTH=384 \
    DATA_DIR="$ROOT/data/search_mix_qwen35_native_v3" TOOL_PROTOCOL=qwen35_native \
    bash "$AUTODL_DIR/train_small_grpo.sh" train smoke 2

if env "${COMMON[@]}" \
    EVAL_DATA_FILE="$ROOT/data/group-probe.parquet" \
    bash "$AUTODL_DIR/train_small_grpo.sh" \
    eval group_probe "$ROOT/models/Qwen3.5-2B" >/dev/null 2>&1; then
    printf 'Group probe accepted EVAL_GROUP_SIZE other than 5.\n' >&2
    exit 1
fi

if env "${COMMON[@]}" \
    EVAL_DATA_FILE="$ROOT/data/search-opportunity.parquet" \
    bash "$AUTODL_DIR/train_small_grpo.sh" \
    eval control "$ROOT/parent-checkpoint" >/dev/null 2>&1; then
    printf 'Legacy evaluation accepted the search-opportunity data override.\n' >&2
    exit 1
fi

if env "${COMMON[@]}" \
    bash "$AUTODL_DIR/train_small_grpo.sh" \
    eval search_opportunity "$ROOT/parent-checkpoint" >/dev/null 2>&1; then
    printf 'Search-opportunity evaluation accepted a missing data file.\n' >&2
    exit 1
fi

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

printf 'Gated and search-opportunity config tests passed.\n'
