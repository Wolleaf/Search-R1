#!/usr/bin/env bash
set -Eeuo pipefail

MODE="${1:-}"
VARIANT="${2:-}"
PROJECT_ROOT="${AUTODL_ROOT:-/root/autodl-tmp/search-r1}"
CHECKOUT_DIR="${AUTODL_CODE_CHECKOUT:-$PROJECT_ROOT/checkout}"
TRAIN_PYTHON="$PROJECT_ROOT/envs/train/bin/python"
MODEL_DIR="$PROJECT_ROOT/models/Qwen3.5-2B"
DATA_DIR_WAS_SET="${DATA_DIR+x}"
CALLER_DATA_DIR="${DATA_DIR-}"
EVAL_DATA_FILE="${EVAL_DATA_FILE:-}"
EVAL_GROUP_SIZE="${EVAL_GROUP_SIZE:-1}"
GPU_COUNT="${GPU_COUNT:?Set GPU_COUNT explicitly to 1 or 2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-500}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR to the run checkpoint directory}"
TRACE_OUTPUT_DIR="${TRACE_OUTPUT_DIR:-}"
TRACE_STAGE="${TRACE_STAGE:-}"
TRACE_RUN_ID="${TRACE_RUN_ID:-}"
TRACE_CHECKPOINT_DIGEST="${TRACE_CHECKPOINT_DIGEST:-}"
TRACE_PARENT_CHECKPOINT_DIGEST="${TRACE_PARENT_CHECKPOINT_DIGEST:-}"
TOOL_PROTOCOL="${TOOL_PROTOCOL:-legacy_xml}"
QWEN_NATIVE_RECOVERY_C20="${AUTODL_QWEN_NATIVE_RECOVERY_C20:-0}"
GRPO_GROUP_SIZE=5
PPO_MINI_BATCH_SIZE=$((TRAIN_BATCH_SIZE * GRPO_GROUP_SIZE))
readonly MAX_TURNS=4
readonly MAX_START_LENGTH=1024
if [[ "$TOOL_PROTOCOL" == qwen35_native ]]; then
    readonly MAX_OBS_LENGTH=500
else
    readonly MAX_OBS_LENGTH=384
fi
readonly RETRIEVER_TOPK=3
readonly SMOKE_STEPS=2
readonly QWEN35_PROMPT_VERSION=qwen35-native-search-v4-terminal-answer-only

[[ "$GPU_COUNT" == 1 || "$GPU_COUNT" == 2 ]] || { printf 'GPU_COUNT must be 1 or 2.\n' >&2; exit 64; }
[[ "$TOOL_PROTOCOL" == legacy_xml || "$TOOL_PROTOCOL" == qwen35_native ]] || {
    printf 'TOOL_PROTOCOL must be legacy_xml or qwen35_native.\n' >&2
    exit 64
}
[[ "$QWEN_NATIVE_RECOVERY_C20" == 0 || "$QWEN_NATIVE_RECOVERY_C20" == 1 ]] || {
    printf 'AUTODL_QWEN_NATIVE_RECOVERY_C20 must be 0 or 1.\n' >&2
    exit 64
}
if [[ -n "${AUTODL_CODE_CHECKOUT+x}" ]]; then
    [[ -d "$CHECKOUT_DIR" && ! -L "$CHECKOUT_DIR" ]] || {
        printf 'Explicit code checkout is missing or symlinked: %s\n' "$CHECKOUT_DIR" >&2
        exit 1
    }
    project_canonical="$(readlink -f -- "$PROJECT_ROOT")"
    checkout_canonical="$(readlink -f -- "$CHECKOUT_DIR")"
    canonical_checkout="$(readlink -f -- "$PROJECT_ROOT/checkout")"
    if [[ "$QWEN_NATIVE_RECOVERY_C20" == 1 ]]; then
        recovery_root="$project_canonical/recovery-checkouts"
        [[ "$CHECKOUT_DIR" == "$checkout_canonical" &&
            "$(dirname -- "$checkout_canonical")" == "$recovery_root" &&
            "$(basename -- "$checkout_canonical")" =~ ^[0-9a-f]{40}$ ]] || {
            printf 'Recovery C20 requires one exact commit checkout under %s.\n' \
                "$recovery_root" >&2
            exit 64
        }
    elif [[ "$checkout_canonical" != "$canonical_checkout" ]]; then
        printf 'A non-recovery job may use only the canonical checkout.\n' >&2
        exit 64
    fi
    CHECKOUT_DIR="$checkout_canonical"
fi
[[ "$TRAIN_BATCH_SIZE" == 8 || "$TRAIN_BATCH_SIZE" == 4 ]] || {
    printf 'TRAIN_BATCH_SIZE must be 8 (default) or the documented OOM fallback 4.\n' >&2
    exit 64
}
[[ "$EVAL_GROUP_SIZE" == 1 || "$EVAL_GROUP_SIZE" == 2 ||
    "$EVAL_GROUP_SIZE" == 3 || "$EVAL_GROUP_SIZE" == 5 ]] || {
    printf 'EVAL_GROUP_SIZE must be 1, 2, 3, or 5.\n' >&2
    exit 64
}
if [[ "$TOOL_PROTOCOL" == legacy_xml && "$EVAL_GROUP_SIZE" != 1 &&
      "$EVAL_GROUP_SIZE" != 5 ]]; then
    printf 'Legacy XML evaluation supports group size 1 or 5 only.\n' >&2
    exit 64
fi
case "$MAX_RESPONSE_LENGTH" in
    500|384)
        if [[ "$TOOL_PROTOCOL" == qwen35_native ]]; then
            readonly MAX_PROMPT_LENGTH=4500
        else
            readonly MAX_PROMPT_LENGTH=4096
        fi
        ;;
    256|192)
        # Preserve the archived experiment entrypoints without changing their configs.
        readonly MAX_PROMPT_LENGTH=$((MAX_START_LENGTH + MAX_TURNS * (MAX_RESPONSE_LENGTH + MAX_OBS_LENGTH)))
        ;;
    *)
        printf 'MAX_RESPONSE_LENGTH must be 500 (default), 384 (OOM fallback), or a legacy value 256/192.\n' >&2
        exit 64
        ;;
esac

case "$TOOL_PROTOCOL:$MODE:$VARIANT" in
    qwen35_native:eval:qwen_native_g1|qwen35_native:eval:qwen_native_g3|qwen35_native:eval:qwen_native_a_val|qwen35_native:eval:qwen_native_r_val|qwen35_native:eval:qwen_native_b_val|qwen35_native:eval:qwen_native_c_val|qwen35_native:eval:qwen_native_a_nq_test|qwen35_native:eval:qwen_native_r_nq_test|qwen35_native:eval:qwen_native_b_nq_test|qwen35_native:eval:qwen_native_c_nq_test|qwen35_native:eval:qwen_native_a_multihop|qwen35_native:eval:qwen_native_r_multihop|qwen35_native:eval:qwen_native_b_multihop|qwen35_native:eval:qwen_native_c_multihop)
        EXPECTED_DATA_DIR="$PROJECT_ROOT/data/search_mix_qwen35_native_v4"
        ;;
    qwen35_native:train:smoke|qwen35_native:train:reproduce|qwen35_native:train:control|qwen35_native:train:cost_aware_gated)
        EXPECTED_DATA_DIR="$PROJECT_ROOT/data/search_mix_qwen35_native_v4"
        ;;
    legacy_xml:eval:group_probe)
        EXPECTED_DATA_DIR="$PROJECT_ROOT/data/search_mix"
        ;;
    legacy_xml:train:smoke|legacy_xml:train:reproduce|legacy_xml:train:control|legacy_xml:train:cost_aware|legacy_xml:train:cost_aware_gated)
        EXPECTED_DATA_DIR="$PROJECT_ROOT/data/nq_small"
        ;;
    legacy_xml:eval:base|legacy_xml:eval:reproduced|legacy_xml:eval:control|legacy_xml:eval:cost_aware|legacy_xml:eval:cost_aware_gated|legacy_xml:eval:search_opportunity)
        EXPECTED_DATA_DIR="$PROJECT_ROOT/data/nq_small"
        ;;
    *)
        printf 'TOOL_PROTOCOL=%s is not registered for %s:%s.\n' \
            "$TOOL_PROTOCOL" "$MODE" "$VARIANT" >&2
        exit 64
        ;;
esac
if [[ "$DATA_DIR_WAS_SET" == x && "$CALLER_DATA_DIR" != "$EXPECTED_DATA_DIR" ]]; then
    printf 'DATA_DIR for %s:%s with TOOL_PROTOCOL=%s must be exactly %s.\n' \
        "$MODE" "$VARIANT" "$TOOL_PROTOCOL" "$EXPECTED_DATA_DIR" >&2
    exit 64
fi
readonly DATA_DIR="$EXPECTED_DATA_DIR"

if [[ "$TOOL_PROTOCOL" == qwen35_native ]]; then
    [[ "$GPU_COUNT" == 2 && "$TRAIN_BATCH_SIZE" == 8 &&
        "$MAX_RESPONSE_LENGTH" == 500 && "$GRPO_GROUP_SIZE" == 5 &&
        "$MAX_OBS_LENGTH" == 500 &&
        $((MAX_TURNS * (MAX_RESPONSE_LENGTH + MAX_OBS_LENGTH) + MAX_RESPONSE_LENGTH)) -le $MAX_PROMPT_LENGTH ]] || {
        printf 'Qwen native v4 requires two GPUs, batch 8, group 5, response/observation 500, four searchable turns, and one terminal answer-only generation within 4500 tokens.\n' >&2
        exit 64
    }
fi

[[ -x "$TRAIN_PYTHON" && -d "$MODEL_DIR" && -d "$DATA_DIR" ]] || {
    printf 'CPU preparation is incomplete under %s.\n' "$PROJECT_ROOT" >&2
    exit 1
}

case "$MODE:$VARIANT" in
    train:smoke)
        COST_LAMBDA=0.0
        COST_REWARD_MODE=linear
        TOTAL_STEPS="${3:-$SMOKE_STEPS}"
        [[ "$TOTAL_STEPS" == "$SMOKE_STEPS" && $# -le 3 ]] || {
            printf 'Smoke training is fixed at %s steps.\n' "$SMOKE_STEPS" >&2
            exit 64
        }
        SAVE_FREQ="$TOTAL_STEPS"
        TEST_FREQ="$TOTAL_STEPS"
        MODEL_PATH="$MODEL_DIR"
        if [[ "$TOOL_PROTOCOL" == qwen35_native ]]; then
            VAL_FILE="$DATA_DIR/val_128.parquet"
        else
            VAL_FILE="$DATA_DIR/val_64.parquet"
        fi
        VAL_ONLY=false
        VAL_BEFORE_TRAIN=false
        USE_KL_LOSS=true
        ;;
    train:reproduce)
        COST_LAMBDA=0.0
        COST_REWARD_MODE=linear
        TOTAL_STEPS="${3:-60}"
        [[ $# -le 3 ]] || {
            printf 'Reproduction training always starts from the prepared base model.\n' >&2
            exit 64
        }
        SAVE_FREQ="$TOTAL_STEPS"
        TEST_FREQ="$TOTAL_STEPS"
        MODEL_PATH="$MODEL_DIR"
        if [[ "$TOOL_PROTOCOL" == qwen35_native ]]; then
            [[ "$TOTAL_STEPS" == 60 ]] || {
                printf 'Qwen native reproduction is fixed at 60 steps.\n' >&2
                exit 64
            }
            VAL_FILE="$DATA_DIR/val_128.parquet"
        else
            VAL_FILE="$DATA_DIR/val_64.parquet"
        fi
        VAL_ONLY=false
        VAL_BEFORE_TRAIN=false
        USE_KL_LOSS=true
        ;;
    train:control|train:cost_aware|train:cost_aware_gated)
        if [[ $# == 3 && ! "$3" =~ ^[1-9][0-9]*$ ]]; then
            TOTAL_STEPS=20
            MODEL_PATH="$3"
        elif [[ $# == 4 ]]; then
            TOTAL_STEPS="$3"
            MODEL_PATH="$4"
        else
            printf 'Pass the reproduced checkpoint shared by both second-stage branches.\n' >&2
            exit 64
        fi
        if [[ "$VARIANT" == cost_aware_gated ]]; then
            COST_LAMBDA=0.10
            COST_REWARD_MODE=correct_only
        elif [[ "$VARIANT" == cost_aware ]]; then
            COST_LAMBDA=0.10
            COST_REWARD_MODE=linear
        else
            COST_LAMBDA=0.0
            COST_REWARD_MODE=linear
        fi
        if [[ "$TOOL_PROTOCOL" == qwen35_native ]]; then
            [[ "$VARIANT" != cost_aware && "$TOTAL_STEPS" == 20 ]] || {
                printf 'Qwen native v4 only registers 20-step control and cost_aware_gated branches.\n' >&2
                exit 64
            }
        fi
        SAVE_FREQ="$TOTAL_STEPS"
        TEST_FREQ="$TOTAL_STEPS"
        if [[ "$QWEN_NATIVE_RECOVERY_C20" == 1 ]]; then
            [[ "$TOOL_PROTOCOL" == qwen35_native &&
                "$VARIANT" == cost_aware_gated && "$TOTAL_STEPS" == 20 ]] || {
                printf 'Recovery mode is restricted to native C20.\n' >&2
                exit 64
            }
            # The recovery workflow performs a separately sealed val-128 run.
            # Save and finalize C20 without a second in-process validation so a
            # slow endpoint cannot destroy an already completed checkpoint.
            TEST_FREQ=-1
        fi
        if [[ "$TOOL_PROTOCOL" == qwen35_native ]]; then
            VAL_FILE="$DATA_DIR/val_128.parquet"
        else
            VAL_FILE="$DATA_DIR/val_64.parquet"
        fi
        VAL_ONLY=false
        VAL_BEFORE_TRAIN=false
        USE_KL_LOSS=true
        ;;
    eval:base|eval:reproduced|eval:control|eval:cost_aware|eval:cost_aware_gated|eval:search_opportunity|eval:group_probe|eval:qwen_native_g1|eval:qwen_native_g3|eval:qwen_native_a_val|eval:qwen_native_r_val|eval:qwen_native_b_val|eval:qwen_native_c_val|eval:qwen_native_a_nq_test|eval:qwen_native_r_nq_test|eval:qwen_native_b_nq_test|eval:qwen_native_c_nq_test|eval:qwen_native_a_multihop|eval:qwen_native_r_multihop|eval:qwen_native_b_multihop|eval:qwen_native_c_multihop)
        [[ $# == 3 ]] || {
            printf 'Pass exactly one model/checkpoint path for evaluation.\n' >&2
            exit 64
        }
        COST_LAMBDA=0.10
        COST_REWARD_MODE=linear
        TOTAL_STEPS=1
        SAVE_FREQ=-1
        TEST_FREQ=-1
        MODEL_PATH="${3:?Pass the selected checkpoint for evaluation}"
        if [[ "$VARIANT" == search_opportunity || "$VARIANT" == group_probe ||
              "$VARIANT" == qwen_native_g1 || "$VARIANT" == qwen_native_g3 ]]; then
            [[ -n "$EVAL_DATA_FILE" ]] || {
                printf 'EVAL_DATA_FILE is required for %s evaluation.\n' "$VARIANT" >&2
                exit 64
            }
            eval_data_canonical="$(readlink -f -- "$EVAL_DATA_FILE")" || {
                printf 'Cannot resolve EVAL_DATA_FILE: %s\n' "$EVAL_DATA_FILE" >&2
                exit 1
            }
            project_canonical="$(readlink -f -- "$PROJECT_ROOT")"
            [[ -f "$eval_data_canonical" && ! -L "$EVAL_DATA_FILE" &&
                "$eval_data_canonical" == "$project_canonical/"* ]] || {
                printf 'EVAL_DATA_FILE must be a regular file under %s.\n' "$PROJECT_ROOT" >&2
                exit 64
            }
            VAL_FILE="$eval_data_canonical"
        elif [[ "$VARIANT" == qwen_native_a_val ||
                "$VARIANT" == qwen_native_r_val ||
                "$VARIANT" == qwen_native_b_val ||
                "$VARIANT" == qwen_native_c_val ]]; then
            [[ -z "$EVAL_DATA_FILE" ]] || {
                printf 'Native curated endpoint evaluation uses sealed val_128.parquet.\n' >&2
                exit 64
            }
            VAL_FILE="$DATA_DIR/val_128.parquet"
        elif [[ "$VARIANT" == qwen_native_a_nq_test ||
                "$VARIANT" == qwen_native_r_nq_test ||
                "$VARIANT" == qwen_native_b_nq_test ||
                "$VARIANT" == qwen_native_c_nq_test ]]; then
            [[ -z "$EVAL_DATA_FILE" ]] || {
                printf 'Native NQ endpoint evaluation uses its sealed v4 artifact.\n' >&2
                exit 64
            }
            VAL_FILE="$DATA_DIR/nq_test_128_native_v4.parquet"
        elif [[ "$VARIANT" == qwen_native_a_multihop ||
                "$VARIANT" == qwen_native_r_multihop ||
                "$VARIANT" == qwen_native_b_multihop ||
                "$VARIANT" == qwen_native_c_multihop ]]; then
            [[ -z "$EVAL_DATA_FILE" ]] || {
                printf 'Native multihop endpoint evaluation uses its sealed v4 artifact.\n' >&2
                exit 64
            }
            VAL_FILE="$DATA_DIR/multihop_eval_256_native_v4.parquet"
        else
            [[ -z "$EVAL_DATA_FILE" ]] || {
                printf 'EVAL_DATA_FILE is only valid for a registered custom evaluation.\n' >&2
                exit 64
            }
            VAL_FILE="$DATA_DIR/test_128.parquet"
        fi
        VAL_ONLY=true
        VAL_BEFORE_TRAIN=true
        USE_KL_LOSS=false
        ;;
    *)
        printf 'Usage: %s train smoke [2]\n' "$0" >&2
        printf '   or: %s train reproduce [STEPS]\n' "$0" >&2
        printf '   or: %s train {control|cost_aware|cost_aware_gated} [STEPS] REPRODUCED_CHECKPOINT\n' "$0" >&2
        printf '   or: %s eval {base|reproduced|control|cost_aware|cost_aware_gated} MODEL_PATH\n' "$0" >&2
        printf '   or: EVAL_DATA_FILE=<parquet> %s eval search_opportunity MODEL_PATH\n' "$0" >&2
        printf '   or: EVAL_DATA_FILE=<parquet> EVAL_GROUP_SIZE=5 %s eval group_probe MODEL_PATH\n' "$0" >&2
        printf '   or: TOOL_PROTOCOL=qwen35_native EVAL_DATA_FILE=<parquet> %s eval qwen_native_g{1,3} MODEL_PATH\n' "$0" >&2
        printf '   or: TOOL_PROTOCOL=qwen35_native %s eval qwen_native_{a,r,b,c}_{val,nq_test,multihop} MODEL_PATH\n' "$0" >&2
        exit 64
        ;;
esac

if [[ "$QWEN_NATIVE_RECOVERY_C20" == 1 &&
      "$MODE:$VARIANT" != train:cost_aware_gated ]]; then
    printf 'Recovery mode is valid only for train:cost_aware_gated.\n' >&2
    exit 64
fi

if [[ "$MODE:$VARIANT" != eval:search_opportunity &&
      "$MODE:$VARIANT" != eval:group_probe &&
       "$MODE:$VARIANT" != eval:qwen_native_g1 &&
       "$MODE:$VARIANT" != eval:qwen_native_g3 && -n "$EVAL_DATA_FILE" ]]; then
    printf 'EVAL_DATA_FILE is only valid for a registered custom evaluation.\n' >&2
    exit 64
fi
case "$MODE:$VARIANT" in
eval:group_probe)
    [[ "$TOOL_PROTOCOL" == legacy_xml && "$EVAL_GROUP_SIZE" == 5 ]] || {
        printf 'Legacy group probe requires TOOL_PROTOCOL=legacy_xml and EVAL_GROUP_SIZE=5.\n' >&2
        exit 64
    }
    VAL_BATCH_SIZE=8
    ;;
eval:qwen_native_g1)
    [[ "$TOOL_PROTOCOL" == qwen35_native && "$EVAL_GROUP_SIZE" == 2 ]] || {
        printf 'Qwen native G1 requires TOOL_PROTOCOL=qwen35_native and EVAL_GROUP_SIZE=2.\n' >&2
        exit 64
    }
    VAL_BATCH_SIZE=8
    ;;
eval:qwen_native_g3)
    [[ "$TOOL_PROTOCOL" == qwen35_native && "$EVAL_GROUP_SIZE" == 5 ]] || {
        printf 'Qwen native G3 requires TOOL_PROTOCOL=qwen35_native and EVAL_GROUP_SIZE=5.\n' >&2
        exit 64
    }
    VAL_BATCH_SIZE=8
    ;;
*)
    [[ "$EVAL_GROUP_SIZE" == 1 ]] || {
        printf 'Only registered grouped evaluations may use EVAL_GROUP_SIZE greater than 1.\n' >&2
        exit 64
    }
    if [[ "$TOOL_PROTOCOL" == qwen35_native ]]; then
        VAL_BATCH_SIZE=8
    else
        VAL_BATCH_SIZE=64
    fi
    ;;
esac

ROLLOUT_TOP_K=0
ROLLOUT_MIN_P=0.0
ROLLOUT_PRESENCE_PENALTY=0.0
ROLLOUT_REPETITION_PENALTY=1.0
if [[ "$TOOL_PROTOCOL" == qwen35_native ]]; then
    RETURN_RAW_CHAT=true
else
    RETURN_RAW_CHAT=false
fi
if [[ "$MODE" == eval && "$EVAL_GROUP_SIZE" == 1 ]]; then
    ROLLOUT_DO_SAMPLE=false
else
    ROLLOUT_DO_SAMPLE=true
fi

[[ "$TOTAL_STEPS" =~ ^[1-9][0-9]*$ ]] || { printf 'steps must be a positive integer.\n' >&2; exit 64; }
[[ -d "$MODEL_PATH" ]] || { printf 'Model/checkpoint path is missing: %s\n' "$MODEL_PATH" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR" "$PROJECT_ROOT/cache/ray" "$PROJECT_ROOT/cache/wandb"
TRACE_HYDRA_ARGS=()
NATIVE_TRAIN_HYDRA_ARGS=()
if [[ "$TOOL_PROTOCOL:$MODE" == qwen35_native:train ]]; then
    NATIVE_TRAIN_HYDRA_ARGS+=(
        "++qwen35_prompt_version=$QWEN35_PROMPT_VERSION"
        "++trainer.native_training_variant=$VARIANT"
    )
fi
if [[ -n "$TRACE_OUTPUT_DIR" ]]; then
    [[ "$TRACE_OUTPUT_DIR" == "$PROJECT_ROOT/"* && "$TRACE_OUTPUT_DIR" != *'/../'* ]] || {
        printf 'TRACE_OUTPUT_DIR must stay under %s.\n' "$PROJECT_ROOT" >&2
        exit 64
    }
    [[ -n "$TRACE_STAGE" && -n "$TRACE_RUN_ID" ]] || {
        printf 'TRACE_STAGE and TRACE_RUN_ID are required when tracing is enabled.\n' >&2
        exit 64
    }
    if [[ "$MODE" == eval && ! "$TRACE_CHECKPOINT_DIGEST" =~ ^[0-9a-f]{64}$ ]]; then
        printf 'Evaluation tracing requires a 64-hex TRACE_CHECKPOINT_DIGEST.\n' >&2
        exit 64
    fi
    if [[ -n "$TRACE_PARENT_CHECKPOINT_DIGEST" &&
          ! "$TRACE_PARENT_CHECKPOINT_DIGEST" =~ ^[0-9a-f]{64}$ ]]; then
        printf 'TRACE_PARENT_CHECKPOINT_DIGEST must be a 64-hex digest.\n' >&2
        exit 64
    fi
    mkdir -p "$TRACE_OUTPUT_DIR"
    TRACE_HYDRA_ARGS+=(
        "++trainer.trace_output_dir=$TRACE_OUTPUT_DIR"
        "++trainer.trace_stage=$TRACE_STAGE"
        "++trainer.trace_run_id=$TRACE_RUN_ID"
        "++trainer.trace_checkpoint_digest=$TRACE_CHECKPOINT_DIGEST"
        "++trainer.trace_parent_checkpoint_digest=$TRACE_PARENT_CHECKPOINT_DIGEST"
    )
fi

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
export WANDB_DIR="${WANDB_DIR:-$PROJECT_ROOT/cache/wandb}"
export RAY_TMPDIR="$PROJECT_ROOT/cache/ray"

HYDRA_ARGS=(
    "data.train_files=$DATA_DIR/train_512.parquet"
    "data.val_files=$VAL_FILE"
    data.train_data_num=null
    data.val_data_num=null
    "data.train_batch_size=$TRAIN_BATCH_SIZE"
    "data.val_batch_size=$VAL_BATCH_SIZE"
    "data.eval_group_size=$EVAL_GROUP_SIZE"
    "data.return_raw_chat=$RETURN_RAW_CHAT"
    "data.max_prompt_length=$MAX_PROMPT_LENGTH"
    "data.max_response_length=$MAX_RESPONSE_LENGTH"
    "data.max_start_length=$MAX_START_LENGTH"
    "data.max_obs_length=$MAX_OBS_LENGTH"
    data.shuffle_train_dataloader=true
    algorithm.adv_estimator=grpo
    "algorithm.cost_lambda=$COST_LAMBDA"
    "++algorithm.cost_reward_mode=$COST_REWARD_MODE"
    algorithm.no_think_rl=false
    "actor_rollout_ref.model.path=$MODEL_PATH"
    actor_rollout_ref.model.enable_gradient_checkpointing=true
    actor_rollout_ref.model.use_remove_padding=false
    ++actor_rollout_ref.model.attn_implementation=sdpa
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285
    "actor_rollout_ref.actor.use_kl_loss=$USE_KL_LOSS"
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    "actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE"
    "actor_rollout_ref.actor.ppo_micro_batch_size=$GPU_COUNT"
    "++actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[Qwen3_5DecoderLayer]"
    actor_rollout_ref.actor.fsdp_config.param_offload=true
    actor_rollout_ref.actor.fsdp_config.grad_offload=true
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true
    "actor_rollout_ref.rollout.log_prob_micro_batch_size=$GPU_COUNT"
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.name=hf
    actor_rollout_ref.rollout.n=1
    "actor_rollout_ref.rollout.n_agent=$GRPO_GROUP_SIZE"
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    "actor_rollout_ref.rollout.top_k=$ROLLOUT_TOP_K"
    "actor_rollout_ref.rollout.min_p=$ROLLOUT_MIN_P"
    "actor_rollout_ref.rollout.presence_penalty=$ROLLOUT_PRESENCE_PENALTY"
    "actor_rollout_ref.rollout.repetition_penalty=$ROLLOUT_REPETITION_PENALTY"
    "actor_rollout_ref.rollout.do_sample=$ROLLOUT_DO_SAMPLE"
    ++actor_rollout_ref.rollout.micro_batch_size=1
    "actor_rollout_ref.ref.log_prob_micro_batch_size=$GPU_COUNT"
    "++actor_rollout_ref.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[Qwen3_5DecoderLayer]"
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
    "max_turns=$MAX_TURNS"
    "++tool_protocol=$TOOL_PROTOCOL"
    retriever.url=http://127.0.0.1:8000/retrieve
    "retriever.topk=$RETRIEVER_TOPK"
    "${NATIVE_TRAIN_HYDRA_ARGS[@]}"
    "${TRACE_HYDRA_ARGS[@]}"
)

if [[ "$QWEN_NATIVE_RECOVERY_C20" == 1 ]]; then
    HYDRA_ARGS+=(
        ++trainer.val_after_train=false
        ++trainer.recovery_c20=true
    )
fi

cd "$CHECKOUT_DIR"
if [[ "${AUTODL_CONFIG_ONLY:-0}" == 1 ]]; then
    exec "$TRAIN_PYTHON" -m verl.trainer.main_ppo --cfg job --resolve "${HYDRA_ARGS[@]}"
fi
exec "$TRAIN_PYTHON" -m verl.trainer.main_ppo "${HYDRA_ARGS[@]}"
