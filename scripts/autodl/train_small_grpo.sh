#!/usr/bin/env bash
set -Eeuo pipefail

MODE="${1:-}"
VARIANT="${2:-}"
PROJECT_ROOT="${AUTODL_ROOT:-/root/autodl-tmp/search-r1}"
CHECKOUT_DIR="$PROJECT_ROOT/checkout"
TRAIN_PYTHON="$PROJECT_ROOT/envs/train/bin/python"
MODEL_DIR="$PROJECT_ROOT/models/Qwen3.5-2B"
DATA_DIR="$PROJECT_ROOT/data/nq_small"
GPU_COUNT="${GPU_COUNT:?Set GPU_COUNT explicitly to 1 or 2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-256}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR to the run checkpoint directory}"
GRPO_GROUP_SIZE=8
PPO_MINI_BATCH_SIZE=$((TRAIN_BATCH_SIZE * GRPO_GROUP_SIZE))

[[ "$GPU_COUNT" == 1 || "$GPU_COUNT" == 2 ]] || { printf 'GPU_COUNT must be 1 or 2.\n' >&2; exit 64; }
[[ "$TRAIN_BATCH_SIZE" == 4 || "$TRAIN_BATCH_SIZE" == 2 ]] || {
    printf 'TRAIN_BATCH_SIZE must be 4 (default) or the documented OOM fallback 2.\n' >&2
    exit 64
}
[[ "$MAX_RESPONSE_LENGTH" == 256 || "$MAX_RESPONSE_LENGTH" == 192 ]] || {
    printf 'MAX_RESPONSE_LENGTH must be 256 (default) or the documented OOM fallback 192.\n' >&2
    exit 64
}
[[ -x "$TRAIN_PYTHON" && -d "$MODEL_DIR" && -d "$DATA_DIR" ]] || {
    printf 'CPU preparation is incomplete under %s.\n' "$PROJECT_ROOT" >&2
    exit 1
}

case "$MODE:$VARIANT" in
    train:smoke)
        COST_LAMBDA=0.0
        TOTAL_STEPS="${3:-1}"
        SAVE_FREQ=1
        TEST_FREQ=1
        MODEL_PATH="$MODEL_DIR"
        VAL_FILE="$DATA_DIR/val_64.parquet"
        VAL_ONLY=false
        VAL_BEFORE_TRAIN=false
        USE_KL_LOSS=true
        ;;
    train:baseline)
        COST_LAMBDA=0.0
        TOTAL_STEPS="${3:-60}"
        SAVE_FREQ=20
        TEST_FREQ=20
        MODEL_PATH="$MODEL_DIR"
        VAL_FILE="$DATA_DIR/val_64.parquet"
        VAL_ONLY=false
        VAL_BEFORE_TRAIN=false
        USE_KL_LOSS=true
        ;;
    train:cost_aware)
        COST_LAMBDA=0.10
        TOTAL_STEPS="${3:-60}"
        SAVE_FREQ=20
        TEST_FREQ=20
        MODEL_PATH="$MODEL_DIR"
        VAL_FILE="$DATA_DIR/val_64.parquet"
        VAL_ONLY=false
        VAL_BEFORE_TRAIN=false
        USE_KL_LOSS=true
        ;;
    eval:baseline|eval:cost_aware)
        COST_LAMBDA=0.10
        TOTAL_STEPS=1
        SAVE_FREQ=-1
        TEST_FREQ=-1
        MODEL_PATH="${3:?Pass the selected checkpoint for evaluation}"
        VAL_FILE="$DATA_DIR/test_128.parquet"
        VAL_ONLY=true
        VAL_BEFORE_TRAIN=true
        USE_KL_LOSS=false
        ;;
    *)
        printf 'Usage: %s train {smoke|baseline|cost_aware} [steps]\n' "$0" >&2
        printf '   or: %s eval {baseline|cost_aware} CHECKPOINT\n' "$0" >&2
        exit 64
        ;;
esac

[[ "$TOTAL_STEPS" =~ ^[1-9][0-9]*$ ]] || { printf 'steps must be a positive integer.\n' >&2; exit 64; }
[[ -d "$MODEL_PATH" ]] || { printf 'Model/checkpoint path is missing: %s\n' "$MODEL_PATH" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR" "$PROJECT_ROOT/cache/ray" "$PROJECT_ROOT/cache/wandb"

if [[ "$GPU_COUNT" == 1 ]]; then
    export CUDA_VISIBLE_DEVICES=0
else
    export CUDA_VISIBLE_DEVICES=0,1
fi
export PYTHONPATH="$CHECKOUT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=42
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export WANDB_DIR="$PROJECT_ROOT/cache/wandb"
export RAY_TMPDIR="$PROJECT_ROOT/cache/ray"

HYDRA_ARGS=(
    "data.train_files=$DATA_DIR/train_512.parquet"
    "data.val_files=$VAL_FILE"
    data.train_data_num=null
    data.val_data_num=null
    "data.train_batch_size=$TRAIN_BATCH_SIZE"
    data.val_batch_size=64
    data.max_prompt_length=2048
    "data.max_response_length=$MAX_RESPONSE_LENGTH"
    data.max_start_length=1024
    data.max_obs_length=384
    data.shuffle_train_dataloader=true
    algorithm.adv_estimator=grpo
    "algorithm.cost_lambda=$COST_LAMBDA"
    algorithm.no_think_rl=false
    "actor_rollout_ref.model.path=$MODEL_PATH"
    actor_rollout_ref.model.enable_gradient_checkpointing=true
    actor_rollout_ref.model.use_remove_padding=false
    ++actor_rollout_ref.model.attn_implementation=sdpa
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.10
    "actor_rollout_ref.actor.use_kl_loss=$USE_KL_LOSS"
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    "actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE"
    "actor_rollout_ref.actor.ppo_micro_batch_size=$GPU_COUNT"
    actor_rollout_ref.actor.fsdp_config.param_offload=true
    actor_rollout_ref.actor.fsdp_config.grad_offload=true
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true
    "actor_rollout_ref.rollout.log_prob_micro_batch_size=$GPU_COUNT"
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.name=hf
    actor_rollout_ref.rollout.n=1
    "actor_rollout_ref.rollout.n_agent=$GRPO_GROUP_SIZE"
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=0.95
    actor_rollout_ref.rollout.top_k=0
    ++actor_rollout_ref.rollout.micro_batch_size=1
    "actor_rollout_ref.ref.log_prob_micro_batch_size=$GPU_COUNT"
    actor_rollout_ref.ref.fsdp_config.param_offload=true
    actor_rollout_ref.actor.state_masking=true
    "trainer.logger=['console','wandb']"
    "++trainer.val_only=$VAL_ONLY"
    "++trainer.val_before_train=$VAL_BEFORE_TRAIN"
    trainer.default_hdfs_dir=null
    "trainer.n_gpus_per_node=$GPU_COUNT"
    trainer.nnodes=1
    "trainer.save_freq=$SAVE_FREQ"
    "trainer.test_freq=$TEST_FREQ"
    trainer.project_name=search-r1-small
    "trainer.experiment_name=${VARIANT}-${MODE}"
    trainer.seed=42
    trainer.total_epochs=1
    "trainer.total_training_steps=$TOTAL_STEPS"
    "trainer.default_local_dir=$OUTPUT_DIR"
    "hydra.run.dir=$OUTPUT_DIR/hydra"
    hydra.output_subdir=null
    hydra.job.chdir=false
    max_turns=2
    retriever.url=http://127.0.0.1:8000/retrieve
    retriever.topk=3
)

cd "$CHECKOUT_DIR"
if [[ "${AUTODL_CONFIG_ONLY:-0}" == 1 ]]; then
    exec "$TRAIN_PYTHON" -m verl.trainer.main_ppo --cfg job --resolve "${HYDRA_ARGS[@]}"
fi
exec "$TRAIN_PYTHON" -m verl.trainer.main_ppo "${HYDRA_ARGS[@]}"
