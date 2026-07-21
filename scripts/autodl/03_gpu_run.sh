#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib/runtime.sh
source "$SCRIPT_DIR/lib/runtime.sh"

TRAIN_ENV="$PROJECT_ROOT/envs/train"
RETRIEVER_ENV="$PROJECT_ROOT/envs/retriever"
MODEL_DIR="$PROJECT_ROOT/models/Qwen3.5-2B"
BM25_INDEX="$PROJECT_ROOT/data/wiki-18-bm25-index/bm25"
CORPUS_ROOT="$PROJECT_ROOT/data/wiki-18-corpus"
CORPUS_JSONL="$CORPUS_ROOT/wiki-18.jsonl"
CORPUS_OFFSETS="$CORPUS_ROOT/wiki-18.offsets.u64"
HANDOFF="$MANIFEST_DIR/cpu_handoff.json"
RESULTS_DIR="$RUNS_ROOT/comparison"
GPU_COUNT="${GPU_COUNT:-}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-256}"
PRICE_PER_HOUR="${AUTODL_PRICE_PER_HOUR:-}"
ALLOCATOR_CONFIG="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
RUN_BUDGET_PROFILE="${AUTODL_RUN_BUDGET_PROFILE:-legacy}"
readonly REPRODUCE_STEPS=60
readonly BRANCH_STEPS=20
# Step 2 exercises backward after Adam has initialized its optimizer state.
readonly BASE_GATE_STEPS=2
readonly CHECKPOINT_LOAD_GATE_STEPS=1

validate_gpu_inputs() {
    [[ "$GPU_COUNT" == 1 || "$GPU_COUNT" == 2 ]] || {
        printf 'Set GPU_COUNT explicitly to 1 or 2. No GPU auto-detection is performed.\n' >&2
        return 64
    }
    [[ "$TRAIN_BATCH_SIZE" == 8 || "$TRAIN_BATCH_SIZE" == 4 ]] || {
        printf 'TRAIN_BATCH_SIZE must be 8 or the documented OOM fallback 4.\n' >&2
        return 64
    }
    [[ "$MAX_RESPONSE_LENGTH" == 256 || "$MAX_RESPONSE_LENGTH" == 192 ]] || {
        printf 'MAX_RESPONSE_LENGTH must be 256 or the documented OOM fallback 192.\n' >&2
        return 64
    }
    [[ "$ALLOCATOR_CONFIG" == expandable_segments:True ]] || {
        printf 'PYTORCH_CUDA_ALLOC_CONF must be expandable_segments:True.\n' >&2
        return 64
    }
    if [[ ! "$PRICE_PER_HOUR" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
            ! awk -v price="$PRICE_PER_HOUR" 'BEGIN { exit !(price > 0) }'; then
        printf 'Set AUTODL_PRICE_PER_HOUR to the positive total instance price.\n' >&2
        return 64
    fi
    command -v timeout >/dev/null 2>&1 || {
        printf 'GNU timeout is required for the GPU budget cap.\n' >&2
        return 1
    }
    [[ "$RUN_BUDGET_PROFILE" == legacy || "$RUN_BUDGET_PROFILE" == gated_followup ]] || {
        printf 'AUTODL_RUN_BUDGET_PROFILE must be legacy or gated_followup.\n' >&2
        return 64
    }
}

file_sha256() {
    local path="$1"
    [[ -f "$path" && ! -L "$path" ]] || {
        printf 'Cannot hash missing or symlinked file: %s\n' "$path" >&2
        return 1
    }
    sha256sum -- "$path" | cut -d' ' -f1
}

tree_sha256() {
    local root="$1"
    [[ -d "$root" && ! -L "$root" ]] || {
        printf 'Cannot hash missing or symlinked directory: %s\n' "$root" >&2
        return 1
    }
    (
        cd "$root"
        [[ -z "$(find . -path './.cache' -prune -o -type l -print -quit)" ]] || {
            printf 'Artifact tree contains a symlink: %s\n' "$root" >&2
            return 1
        }
        find . -path './.cache' -prune -o -type f -print0 \
            | LC_ALL=C sort -z \
            | xargs -0 -r sha256sum \
            | sha256sum \
            | cut -d' ' -f1
    )
}

finish_run_record() {
    local run_dir="$1" rc="$2" started_epoch="$3" started_at="$4"
    local budget_rmb="$5" timeout_seconds="$6"
    local job_mode="$7" variant="$8" requested_steps="$9" input_model="${10}"
    local trace_output_dir="${11:-}"
    local finished_epoch elapsed state marker timed_out=false
    local resolved_config_sha256=unavailable
    finished_epoch="$(date +%s)"
    elapsed=$((finished_epoch - started_epoch))
    if [[ -f "$run_dir/resolved-config.yaml" && ! -L "$run_dir/resolved-config.yaml" ]]; then
        resolved_config_sha256="$(file_sha256 "$run_dir/resolved-config.yaml")"
    fi
    if ((rc == 0)); then
        state=success
        marker=.success
    else
        state=failed
        marker=.failed
    fi
    if ((rc == 124)); then
        timed_out=true
    fi
    atomic_write "$run_dir/exit-code" "$rc"$'\n'
    atomic_write "$run_dir/run.env" \
        "started_at=$started_at"$'\n'\
"finished_at=$(utc_now)"$'\n'\
"elapsed_seconds=$elapsed"$'\n'\
"job_mode=$job_mode"$'\n'\
"variant=$variant"$'\n'\
"gpu_count=$GPU_COUNT"$'\n'\
"train_steps=$requested_steps"$'\n'\
"train_batch_size=$TRAIN_BATCH_SIZE"$'\n'\
"max_response_length=$MAX_RESPONSE_LENGTH"$'\n'\
"input_model=$input_model"$'\n'\
"trace_output_dir=$trace_output_dir"$'\n'\
"resolved_config_sha256=$resolved_config_sha256"$'\n'\
"pytorch_cuda_alloc_conf=$ALLOCATOR_CONFIG"$'\n'\
"price_per_hour=$PRICE_PER_HOUR"$'\n'\
"budget_rmb=$budget_rmb"$'\n'\
"timeout_seconds=$timeout_seconds"$'\n'\
"timed_out=$timed_out"$'\n'
    atomic_write "$run_dir/terminal" "$state"$'\n'
    : >"$run_dir/$marker"
    sync_path "$run_dir"
    rm -f -- "$run_dir/.running"
    sync_path "$run_dir"
}

run_job() {
    local mode="$1" variant="$2" argument="${3:-}" model_path="${4:-}"
    local input_model_digest="${5:-}"
    local parent run_dir started_epoch started_at rc budget_rmb timeout_seconds
    local requested_steps=0 input_model trace_manifest expected_trace_rows trace_stage
    local trace_checkpoint_digest='' trace_parent_checkpoint_digest=''
    local trace_output_dir=''
    local -a job_args
    case "$mode:$variant" in
        train:smoke)
            [[ "$argument" == "$BASE_GATE_STEPS" ]] || {
                printf 'Base smoke gate is fixed at %s steps.\n' "$BASE_GATE_STEPS" >&2
                return 64
            }
            budget_rmb=15
            ;;
        train:reproduce)
            [[ "$argument" == "$REPRODUCE_STEPS" ]] || {
                printf 'Reproduction endpoint is fixed at step %s.\n' "$REPRODUCE_STEPS" >&2
                return 64
            }
            budget_rmb=100
            ;;
        train:control)
            if [[ "$argument" == "$CHECKPOINT_LOAD_GATE_STEPS" ]]; then
                budget_rmb=15
            elif [[ "$argument" == "$BRANCH_STEPS" ]]; then
                budget_rmb=40
            else
                printf 'Control is allowed only for the %s-step checkpoint-load gate or step %s endpoint.\n' \
                    "$CHECKPOINT_LOAD_GATE_STEPS" "$BRANCH_STEPS" >&2
                return 64
            fi
            ;;
        train:cost_aware)
            [[ "$argument" == "$BRANCH_STEPS" ]] || {
                printf 'Cost-aware endpoint is fixed at step %s.\n' "$BRANCH_STEPS" >&2
                return 64
            }
            budget_rmb=40
            ;;
        train:cost_aware_gated)
            if [[ "$RUN_BUDGET_PROFILE" != gated_followup ]]; then
                printf 'C-gated is available only in the gated follow-up workflow.\n' >&2
                return 64
            elif [[ "$argument" == "$BASE_GATE_STEPS" ]]; then
                budget_rmb=8
            elif [[ "$argument" == "$BRANCH_STEPS" ]]; then
                budget_rmb=25
            else
                printf 'C-gated is allowed only for the %s-step gate or step %s endpoint.\n' \
                    "$BASE_GATE_STEPS" "$BRANCH_STEPS" >&2
                return 64
            fi
            ;;
        eval:base|eval:reproduced|eval:control|eval:cost_aware|eval:cost_aware_gated)
            if [[ "$RUN_BUDGET_PROFILE" == gated_followup ]]; then
                budget_rmb=5
            else
                budget_rmb=10
            fi
            ;;
        *) printf 'No budget is defined for %s:%s.\n' "$mode" "$variant" >&2; return 64 ;;
    esac
    if [[ "$mode" == train ]]; then
        requested_steps="$argument"
        if [[ "$variant" == control || "$variant" == cost_aware ||
              "$variant" == cost_aware_gated ]]; then
            input_model="$model_path"
        else
            input_model="$MODEL_DIR"
        fi
    else
        input_model="$argument"
    fi
    if [[ -n "$input_model_digest" && ! "$input_model_digest" =~ ^[0-9a-f]{64}$ ]]; then
        printf 'Input model digest is invalid for %s:%s.\n' "$mode" "$variant" >&2
        return 64
    fi
    if [[ "$RUN_BUDGET_PROFILE" == gated_followup && "$mode" == eval &&
          -z "$input_model_digest" ]]; then
        printf 'Evaluation requires the exact input checkpoint digest for trace provenance.\n' >&2
        return 64
    fi
    if [[ "$mode" == eval ]]; then
        trace_checkpoint_digest="$input_model_digest"
    else
        trace_parent_checkpoint_digest="$input_model_digest"
    fi
    timeout_seconds="$(awk -v budget="$budget_rmb" -v price="$PRICE_PER_HOUR" \
        'BEGIN { printf "%d", budget / price * 3600 }')"
    ((timeout_seconds > 0)) || {
        printf 'Computed timeout is not positive for %s:%s.\n' "$mode" "$variant" >&2
        return 64
    }
    if [[ "$mode" == eval ]]; then
        parent="$RUNS_ROOT/eval/$variant"
    else
        parent="$RUNS_ROOT/$variant"
    fi
    mkdir -p "$parent/attempts"
    run_dir="$parent/attempts/$(date -u +'%Y%m%dT%H%M%SZ')-$$-$RANDOM"
    mkdir "$run_dir"
    : >"$run_dir/.running"
    atomic_write "$parent/latest" "$run_dir"$'\n'
    started_epoch="$(date +%s)"
    started_at="$(utc_now)"
    trace_stage="$variant"
    if [[ "$mode:$variant:$argument" == "train:cost_aware_gated:$BASE_GATE_STEPS" ]]; then
        trace_stage=cost_aware_gated_gate
    fi
    if [[ "$RUN_BUDGET_PROFILE" == gated_followup ]]; then
        trace_output_dir="$run_dir/traces"
    fi
    printf 'Starting %s %s (budget %s RMB, timeout %ss); log: %s/train.log\n' \
        "$mode" "$variant" "$budget_rmb" "$timeout_seconds" "$run_dir"

    set +e
    : >"$run_dir/train.log"
    job_args=("$mode" "$variant" "$argument")
    if [[ -n "$model_path" ]]; then
        job_args+=("$model_path")
    fi
    AUTODL_CONFIG_ONLY=1 \
        OUTPUT_DIR="$run_dir/checkpoints" \
        GPU_COUNT="$GPU_COUNT" \
        TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
        MAX_RESPONSE_LENGTH="$MAX_RESPONSE_LENGTH" \
        TRACE_OUTPUT_DIR="$trace_output_dir" \
        TRACE_STAGE="$trace_stage" \
        TRACE_RUN_ID="$(basename -- "$run_dir")" \
        TRACE_CHECKPOINT_DIGEST="$trace_checkpoint_digest" \
        TRACE_PARENT_CHECKPOINT_DIGEST="$trace_parent_checkpoint_digest" \
        AUTODL_ROOT="$PROJECT_ROOT" \
        bash "$CHECKOUT_DIR/scripts/autodl/train_small_grpo.sh" \
        "${job_args[@]}" \
        >"$run_dir/resolved-config.yaml" 2>>"$run_dir/train.log"
    rc=$?
    if ((rc == 0)); then
        OUTPUT_DIR="$run_dir/checkpoints" \
        GPU_COUNT="$GPU_COUNT" \
        TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
        MAX_RESPONSE_LENGTH="$MAX_RESPONSE_LENGTH" \
        TRACE_OUTPUT_DIR="$trace_output_dir" \
        TRACE_STAGE="$trace_stage" \
        TRACE_RUN_ID="$(basename -- "$run_dir")" \
        TRACE_CHECKPOINT_DIGEST="$trace_checkpoint_digest" \
        TRACE_PARENT_CHECKPOINT_DIGEST="$trace_parent_checkpoint_digest" \
        AUTODL_ROOT="$PROJECT_ROOT" \
        timeout --signal=TERM --kill-after=120s "${timeout_seconds}s" \
            bash "$CHECKOUT_DIR/scripts/autodl/train_small_grpo.sh" \
            "${job_args[@]}" >>"$run_dir/train.log" 2>&1
        rc=$?
    fi
    if ((rc == 0)) && [[ -n "$trace_output_dir" ]]; then
        if [[ "$mode" == train ]]; then
            trace_manifest="$run_dir/traces/train_trajectories.manifest.json"
            expected_trace_rows=$((requested_steps * TRAIN_BATCH_SIZE * 5))
        else
            trace_manifest="$run_dir/traces/eval_predictions.manifest.json"
            expected_trace_rows=128
        fi
        PYTHONPATH="$CHECKOUT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
        "$TRAIN_ENV/bin/python" -m search_r1.trajectory_trace verify \
            --manifest "$trace_manifest" \
            --expected-rows "$expected_trace_rows" >>"$run_dir/train.log" 2>&1
        rc=$?
    fi
    set -e
    finish_run_record "$run_dir" "$rc" "$started_epoch" "$started_at" \
        "$budget_rmb" "$timeout_seconds" "$mode" "$variant" "$requested_steps" "$input_model" \
        "$trace_output_dir"
    LAST_RUN_DIR="$run_dir"
    if ((rc != 0)); then
        printf '%s %s failed with exit code %s; inspect %s/train.log\n' "$mode" "$variant" "$rc" "$run_dir" >&2
        return "$rc"
    fi
}

fixed_checkpoint() {
    local run_dir="$1" step="$2" checkpoint canonical_run canonical_checkpoint canonical_runs_root
    [[ "$(tr -d '\r\n' <"$run_dir/terminal")" == success ]] || {
        printf 'Run is not successful: %s\n' "$run_dir" >&2
        return 1
    }
    [[ "$(tr -d '\r\n' <"$run_dir/exit-code")" == 0 && -f "$run_dir/.success" ]] || {
        printf 'Run has no valid success evidence: %s\n' "$run_dir" >&2
        return 1
    }
    [[ ! -e "$run_dir/.running" ]] || {
        printf 'Run is still active: %s\n' "$run_dir" >&2
        return 1
    }
    checkpoint="$run_dir/checkpoints/actor/global_step_$step"
    [[ -d "$checkpoint" && ! -L "$checkpoint" && -f "$checkpoint/config.json" ]] || {
        printf 'Fixed endpoint checkpoint is incomplete: %s\n' "$checkpoint" >&2
        return 1
    }
    compgen -G "$checkpoint/model*.safetensors" >/dev/null || {
        printf 'Fixed endpoint checkpoint has no model weights: %s\n' "$checkpoint" >&2
        return 1
    }
    canonical_run="$(readlink -f -- "$run_dir")"
    canonical_checkpoint="$(readlink -f -- "$checkpoint")"
    canonical_runs_root="$(readlink -f -- "$RUNS_ROOT")"
    [[ "$canonical_run" == "$canonical_runs_root/"* &&
        "$canonical_checkpoint" == "$canonical_run/checkpoints/actor/global_step_$step" ]] || {
        printf 'Fixed endpoint checkpoint escapes its run directory: %s\n' "$checkpoint" >&2
        return 1
    }
    printf '%s\n' "$canonical_checkpoint"
}

record_lineage() {
    local run_dir="$1" role="$2" checkpoint="$3" checkpoint_digest="$4"
    local parent_checkpoint="$5" parent_checkpoint_digest="$6"
    local checkout_commit="$7" cpu_handoff_digest="$8" cost_lambda="$9"
    local cost_reward_mode="${10:-linear}"
    local metadata resolved_config_sha256 value
    resolved_config_sha256="$(file_sha256 "$run_dir/resolved-config.yaml")"
    for value in "$role" "$checkpoint" "$parent_checkpoint"; do
        [[ -n "$value" && "$value" != *$'\n'* && "$value" != *$'\t'* ]] || {
            printf 'Invalid lineage value for %s.\n' "$run_dir" >&2
            return 1
        }
    done
    [[ "$checkpoint_digest" =~ ^[0-9a-f]{64}$ &&
        "$parent_checkpoint_digest" =~ ^[0-9a-f]{64}$ &&
        "$cpu_handoff_digest" =~ ^[0-9a-f]{64}$ &&
        "$resolved_config_sha256" =~ ^[0-9a-f]{64}$ &&
        "$checkout_commit" =~ ^[0-9a-f]{40}$ ]] || {
        printf 'Invalid digest in lineage for %s.\n' "$run_dir" >&2
        return 1
    }
    grep -Fxq "resolved_config_sha256=$resolved_config_sha256" "$run_dir/run.env" || {
        printf 'Resolved config digest changed for %s.\n' "$run_dir" >&2
        return 1
    }
    metadata="$(cat "$run_dir/run.env")"
    atomic_write "$run_dir/run.env" \
        "$metadata"$'\n'\
"role=$role"$'\n'\
"checkpoint=$checkpoint"$'\n'\
"checkpoint_digest=$checkpoint_digest"$'\n'\
"parent_checkpoint=$parent_checkpoint"$'\n'\
"parent_checkpoint_digest=$parent_checkpoint_digest"$'\n'\
"checkout_commit=$checkout_commit"$'\n'\
"cpu_handoff_digest=$cpu_handoff_digest"$'\n'\
"seed=42"$'\n'\
"cost_lambda=$cost_lambda"$'\n'\
"cost_reward_mode=$cost_reward_mode"$'\n'
    atomic_write "$run_dir/lineage.tsv" \
        $'role\tcheckpoint\tcheckpoint_digest\tparent_checkpoint\tparent_checkpoint_digest\tcheckout_commit\tcpu_handoff_digest\tresolved_config_sha256\n'\
"$role"$'\t'"$checkpoint"$'\t'"$checkpoint_digest"$'\t'"$parent_checkpoint"$'\t'"$parent_checkpoint_digest"$'\t'"$checkout_commit"$'\t'"$cpu_handoff_digest"$'\t'"$resolved_config_sha256"$'\n'
    sync_path "$run_dir"
}

delete_gate_checkpoint() {
    local run_dir="$1" checkpoint="$2" variant="$3" gate_steps="$4"
    local canonical_run canonical_runs_root expected_checkpoint bytes metadata
    [[ "$variant:$gate_steps" == "smoke:$BASE_GATE_STEPS" ||
        "$variant:$gate_steps" == "control:$CHECKPOINT_LOAD_GATE_STEPS" ||
        "$variant:$gate_steps" == "cost_aware_gated:$BASE_GATE_STEPS" ]] || {
        printf 'Refusing cleanup for unknown gate endpoint: %s step %s\n' "$variant" "$gate_steps" >&2
        return 1
    }
    canonical_run="$(readlink -f -- "$run_dir")"
    canonical_runs_root="$(readlink -f -- "$RUNS_ROOT")"
    expected_checkpoint="$canonical_run/checkpoints/actor/global_step_$gate_steps"
    [[ "$canonical_run" == "$canonical_runs_root/"* ]] || {
        printf 'Refusing gate cleanup outside the runs root: %s\n' "$run_dir" >&2
        return 1
    }
    [[ "$checkpoint" == "$expected_checkpoint" && -d "$checkpoint" && ! -L "$checkpoint" ]] || {
        printf 'Refusing unsafe gate checkpoint cleanup: %s\n' "$checkpoint" >&2
        return 1
    }
    [[ "$(tr -d '\r\n' <"$run_dir/terminal")" == success &&
        "$(tr -d '\r\n' <"$run_dir/exit-code")" == 0 &&
        -f "$run_dir/.success" && ! -e "$run_dir/.running" ]] || {
        printf 'Refusing cleanup without successful gate evidence: %s\n' "$run_dir" >&2
        return 1
    }
    grep -Fxq 'job_mode=train' "$run_dir/run.env" \
        && grep -Fxq "variant=$variant" "$run_dir/run.env" \
        && grep -Fxq "train_steps=$gate_steps" "$run_dir/run.env" || {
        printf 'Refusing cleanup of a non-gate run: %s\n' "$run_dir" >&2
        return 1
    }
    bytes="$(du -sb "$checkpoint" | cut -f1)"
    atomic_write "$run_dir/checkpoint-cleanup.env" \
        "requested_at=$(utc_now)"$'\n'\
"target=$checkpoint"$'\n'\
"train_steps=$gate_steps"$'\n'\
"bytes=$bytes"$'\n'\
"reason=successful-gate-is-not-a-scientific-checkpoint"$'\n'
    sync_path "$run_dir"
    rm -rf -- "$checkpoint"
    [[ ! -e "$checkpoint" && ! -L "$checkpoint" ]] || return 1
    metadata="$(cat "$run_dir/checkpoint-cleanup.env")"
    atomic_write "$run_dir/checkpoint-cleanup.env" \
        "$metadata"$'\n'"completed_at=$(utc_now)"$'\n'
    sync_path "$run_dir"
}

write_lineage_manifest() {
    local base_checkpoint="$1" base_digest="$2"
    local reproduced_checkpoint="$3" reproduced_digest="$4" reproduced_config="$5"
    local control_checkpoint="$6" control_digest="$7" control_config="$8"
    local cost_checkpoint="$9" cost_digest="${10}" cost_config="${11}"
    local checkout_commit="${12}" cpu_handoff_digest="${13}" value
    for value in "$base_checkpoint" "$reproduced_checkpoint" "$control_checkpoint" "$cost_checkpoint"; do
        [[ -n "$value" && "$value" != *$'\n'* && "$value" != *$'\t'* ]] || {
            printf 'Invalid checkpoint path in comparison lineage.\n' >&2
            return 1
        }
    done
    for value in "$base_digest" "$reproduced_digest" "$reproduced_config" \
        "$control_digest" "$control_config" "$cost_digest" "$cost_config" \
        "$cpu_handoff_digest"; do
        [[ "$value" =~ ^[0-9a-f]{64}$ ]] || {
            printf 'Invalid digest in comparison lineage.\n' >&2
            return 1
        }
    done
    [[ "$checkout_commit" =~ ^[0-9a-f]{40}$ ]] || {
        printf 'Invalid checkout commit in comparison lineage.\n' >&2
        return 1
    }
    atomic_write "$RESULTS_DIR/lineage.tsv" \
        $'stage\trole\tcheckpoint\tcheckpoint_digest\tparent_checkpoint\tparent_checkpoint_digest\tcheckout_commit\tcpu_handoff_digest\tresolved_config_sha256\n'\
"A"$'\t'"base"$'\t'"$base_checkpoint"$'\t'"$base_digest"$'\t-\t-\t'"$checkout_commit"$'\t'"$cpu_handoff_digest"$'\t-\n'\
"R"$'\t'"reproduced"$'\t'"$reproduced_checkpoint"$'\t'"$reproduced_digest"$'\t'"$base_checkpoint"$'\t'"$base_digest"$'\t'"$checkout_commit"$'\t'"$cpu_handoff_digest"$'\t'"$reproduced_config"$'\n'\
"B"$'\t'"control"$'\t'"$control_checkpoint"$'\t'"$control_digest"$'\t'"$reproduced_checkpoint"$'\t'"$reproduced_digest"$'\t'"$checkout_commit"$'\t'"$cpu_handoff_digest"$'\t'"$control_config"$'\n'\
"C"$'\t'"cost_aware"$'\t'"$cost_checkpoint"$'\t'"$cost_digest"$'\t'"$reproduced_checkpoint"$'\t'"$reproduced_digest"$'\t'"$checkout_commit"$'\t'"$cpu_handoff_digest"$'\t'"$cost_config"$'\n'
    sync_path "$RESULTS_DIR"
}

gpu_action() {
    local _attempt="$1"
    local commit python_version torch_version recorded_digest actual_digest
    local retriever_pid='' retriever_log ready=false base_model base_model_digest
    local base_gate_run branch_gate_run reproduce_run control_run cost_run
    local base_eval reproduced_eval control_eval cost_eval
    local base_gate_checkpoint branch_gate_checkpoint base_gate_digest
    local reproduce_checkpoint control_checkpoint cost_checkpoint
    local reproduce_digest control_digest cost_digest
    local reproduce_config_digest control_config_digest cost_config_digest
    local comparison_checksums comparison_digest
    validate_gpu_inputs
    commit="$(expected_commit)"
    verify_checkout "$commit"
    [[ -f "$MANIFEST_DIR/cpu.ok" && -f "$HANDOFF" && -f "$HANDOFF.sha256" ]] || {
        printf 'CPU phase is not sealed; refusing paid GPU work.\n' >&2
        return 1
    }
    if [[ "${AUTODL_GPU_PIPELINE:-legacy}" == legacy ]]; then
        rm -f -- "$MANIFEST_DIR/gpu.ok"
        sync_path "$MANIFEST_DIR"
    fi

    # Everything after this point is deliberately offline except localhost retrieval.
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    export WANDB_MODE=offline
    export PIP_NO_INDEX=1
    export HF_HOME="$PROJECT_ROOT/cache/huggingface"
    export HF_HUB_CACHE="$HF_HOME/hub"
    export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
    export TOKENIZERS_PARALLELISM=false
    export PYTHONDONTWRITEBYTECODE=1
    export PYTORCH_CUDA_ALLOC_CONF="$ALLOCATOR_CONFIG"
    export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
    export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
    export no_proxy="$NO_PROXY"

    java -version >"$_attempt/java-version.txt" 2>&1
    grep -Eq 'version "21([.]|")' "$_attempt/java-version.txt" || {
        printf 'OpenJDK 21 from the CPU phase is unavailable; refusing GPU work.\n' >&2
        return 1
    }

    "$TRAIN_ENV/bin/python" - <<'PY'
import sys
import torch
if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"Python 3.12 is required, found {sys.version.split()[0]}")
if torch.__version__.split("+")[0] != "2.8.0" or torch.version.cuda != "12.8":
    raise SystemExit(f"Expected PyTorch 2.8.0 CUDA 12.8, found {torch.__version__} CUDA {torch.version.cuda}")
PY
    "$TRAIN_ENV/bin/python" -m pip freeze --all >"$_attempt/train-freeze.current.txt"
    "$RETRIEVER_ENV/bin/python" -m pip freeze --all >"$_attempt/retriever-freeze.current.txt"
    cmp -s "$MANIFEST_DIR/train-freeze.txt" "$_attempt/train-freeze.current.txt" || {
        printf 'Train environment changed after the CPU handoff.\n' >&2
        return 1
    }
    cmp -s "$MANIFEST_DIR/retriever-freeze.txt" "$_attempt/retriever-freeze.current.txt" || {
        printf 'Retriever environment changed after the CPU handoff.\n' >&2
        return 1
    }
    python_version="$($TRAIN_ENV/bin/python -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
    torch_version="$($TRAIN_ENV/bin/python -c 'import torch; print(torch.__version__)')"
    "$TRAIN_ENV/bin/python" "$CHECKOUT_DIR/scripts/autodl/handoff.py" verify \
        --root "$PROJECT_ROOT" \
        --commit "$commit" \
        --python-version "$python_version" \
        --torch-version "$torch_version" \
        --manifest "$HANDOFF"
    recorded_digest="$(tr -d '\r\n' <"$MANIFEST_DIR/cpu.ok")"
    actual_digest="$(cut -d' ' -f1 "$HANDOFF.sha256")"
    [[ "$recorded_digest" == "$actual_digest" ]] || {
        printf 'cpu.ok does not match the sealed handoff.\n' >&2
        return 1
    }
    verify_checkout "$commit"

    base_model="$(readlink -f -- "$MODEL_DIR")"
    base_model_digest="$(tree_sha256 "$base_model")"

    mkdir -p "$RESULTS_DIR"
    retriever_log="$LOG_DIR/bm25-$(date -u +'%Y%m%dT%H%M%SZ').log"
    setsid "$RETRIEVER_ENV/bin/python" "$CHECKOUT_DIR/search_r1/search/bm25_server.py" \
        --index-path "$BM25_INDEX" \
        --corpus-path "$CORPUS_JSONL" \
        --offsets-path "$CORPUS_OFFSETS" \
        --topk 3 --host 127.0.0.1 --port 8000 \
        >"$retriever_log" 2>&1 < /dev/null &
    retriever_pid=$!
    cleanup_retriever() {
        if [[ -n "$retriever_pid" ]] && kill -0 "$retriever_pid" 2>/dev/null; then
            kill -TERM -- "-$retriever_pid" 2>/dev/null || true
            wait "$retriever_pid" 2>/dev/null || true
        fi
    }
    trap cleanup_retriever EXIT

    for _ in {1..60}; do
        if ! kill -0 "$retriever_pid" 2>/dev/null; then
            printf 'BM25 service exited early; inspect %s\n' "$retriever_log" >&2
            return 1
        fi
        if curl --noproxy '*' --fail --silent --show-error --max-time 5 \
            http://127.0.0.1:8000/health >/dev/null; then
            ready=true
            break
        fi
        sleep 2
    done
    [[ "$ready" == true ]] || {
        printf 'BM25 service did not become ready; inspect %s\n' "$retriever_log" >&2
        return 1
    }

    if [[ "${AUTODL_GPU_PIPELINE:-legacy}" == cost_aware_gated ]]; then
        declare -F cost_aware_gated_pipeline >/dev/null || {
            printf 'The C-gated pipeline callback is unavailable.\n' >&2
            return 1
        }
        cost_aware_gated_pipeline "$_attempt" "$commit" "$recorded_digest" \
            "$base_model" "$base_model_digest"
        cleanup_retriever
        trap - EXIT
        return 0
    elif [[ "${AUTODL_GPU_PIPELINE:-legacy}" != legacy ]]; then
        printf 'Unknown GPU pipeline: %s\n' "$AUTODL_GPU_PIPELINE" >&2
        return 64
    fi

    run_job train smoke "$BASE_GATE_STEPS" '' "$base_model_digest"
    base_gate_run="$LAST_RUN_DIR"
    base_gate_checkpoint="$(fixed_checkpoint "$base_gate_run" "$BASE_GATE_STEPS")"
    base_gate_digest="$(tree_sha256 "$base_gate_checkpoint")"
    run_job train control "$CHECKPOINT_LOAD_GATE_STEPS" "$base_gate_checkpoint" "$base_gate_digest"
    branch_gate_run="$LAST_RUN_DIR"
    branch_gate_checkpoint="$(fixed_checkpoint "$branch_gate_run" "$CHECKPOINT_LOAD_GATE_STEPS")"
    delete_gate_checkpoint "$branch_gate_run" "$branch_gate_checkpoint" control "$CHECKPOINT_LOAD_GATE_STEPS"
    delete_gate_checkpoint "$base_gate_run" "$base_gate_checkpoint" smoke "$BASE_GATE_STEPS"

    run_job train reproduce "$REPRODUCE_STEPS" '' "$base_model_digest"
    reproduce_run="$LAST_RUN_DIR"
    reproduce_checkpoint="$(fixed_checkpoint "$reproduce_run" "$REPRODUCE_STEPS")"
    reproduce_digest="$(tree_sha256 "$reproduce_checkpoint")"
    reproduce_config_digest="$(file_sha256 "$reproduce_run/resolved-config.yaml")"
    record_lineage "$reproduce_run" reproduced "$reproduce_checkpoint" "$reproduce_digest" \
        "$base_model" "$base_model_digest" "$commit" "$recorded_digest" 0

    run_job train control "$BRANCH_STEPS" "$reproduce_checkpoint" "$reproduce_digest"
    control_run="$LAST_RUN_DIR"
    control_checkpoint="$(fixed_checkpoint "$control_run" "$BRANCH_STEPS")"
    control_digest="$(tree_sha256 "$control_checkpoint")"
    control_config_digest="$(file_sha256 "$control_run/resolved-config.yaml")"
    record_lineage "$control_run" control "$control_checkpoint" "$control_digest" \
        "$reproduce_checkpoint" "$reproduce_digest" "$commit" "$recorded_digest" 0

    run_job train cost_aware "$BRANCH_STEPS" "$reproduce_checkpoint" "$reproduce_digest"
    cost_run="$LAST_RUN_DIR"
    cost_checkpoint="$(fixed_checkpoint "$cost_run" "$BRANCH_STEPS")"
    cost_digest="$(tree_sha256 "$cost_checkpoint")"
    cost_config_digest="$(file_sha256 "$cost_run/resolved-config.yaml")"
    record_lineage "$cost_run" cost_aware "$cost_checkpoint" "$cost_digest" \
        "$reproduce_checkpoint" "$reproduce_digest" "$commit" "$recorded_digest" 0.10

    run_job eval base "$base_model" '' "$base_model_digest"
    base_eval="$LAST_RUN_DIR"
    run_job eval reproduced "$reproduce_checkpoint" '' "$reproduce_digest"
    reproduced_eval="$LAST_RUN_DIR"
    run_job eval control "$control_checkpoint" '' "$control_digest"
    control_eval="$LAST_RUN_DIR"
    run_job eval cost_aware "$cost_checkpoint" '' "$cost_digest"
    cost_eval="$LAST_RUN_DIR"

    write_lineage_manifest \
        "$base_model" "$base_model_digest" \
        "$reproduce_checkpoint" "$reproduce_digest" "$reproduce_config_digest" \
        "$control_checkpoint" "$control_digest" "$control_config_digest" \
        "$cost_checkpoint" "$cost_digest" "$cost_config_digest" \
        "$commit" "$recorded_digest"
    "$TRAIN_ENV/bin/python" "$CHECKOUT_DIR/scripts/autodl/results.py" summarize \
        --base-log "$base_eval/train.log" \
        --reproduced-log "$reproduced_eval/train.log" \
        --control-log "$control_eval/train.log" \
        --cost-aware-log "$cost_eval/train.log" \
        --base-checkpoint "$base_model" \
        --base-checkpoint-digest "$base_model_digest" \
        --reproduced-run-metadata "$reproduce_run/run.env" \
        --control-run-metadata "$control_run/run.env" \
        --cost-aware-run-metadata "$cost_run/run.env" \
        --output-dir "$RESULTS_DIR"
    verify_checkout "$commit"
    [[ -s "$RESULTS_DIR/results.csv" && -s "$RESULTS_DIR/results.md" &&
        -s "$RESULTS_DIR/lineage.tsv" &&
        "$(wc -l <"$RESULTS_DIR/results.csv")" == 5 &&
        "$(wc -l <"$RESULTS_DIR/lineage.tsv")" == 5 ]] || {
        printf 'Comparison results or four-stage lineage are incomplete.\n' >&2
        return 1
    }
    comparison_checksums="$(
        cd "$RESULTS_DIR"
        sha256sum results.csv results.md lineage.tsv
    )"
    atomic_write "$RESULTS_DIR/comparison.sha256" "$comparison_checksums"$'\n'
    sync_path "$RESULTS_DIR"
    comparison_digest="$(file_sha256 "$RESULTS_DIR/comparison.sha256")"
    atomic_write "$MANIFEST_DIR/gpu.ok" "$comparison_digest"$'\n'
    sync_path "$MANIFEST_DIR"
    atomic_write "$_attempt/comparison-digest" "$comparison_digest"$'\n'
    sync_path "$_attempt"
    cleanup_retriever
    trap - EXIT
    printf 'Completed comparison: %s/results.md\n' "$RESULTS_DIR"
    printf 'The script does not shut down AutoDL. Confirm stopped state and billing in the console.\n'
}

main() {
    case "${1:-}" in
        --worker)
            phase_worker gpu "${2:?missing attempt directory}" "$0"
            ;;
        --action)
            gpu_action "${2:?missing attempt directory}"
            ;;
        '')
            validate_gpu_inputs
            phase_launch gpu "$0"
            ;;
        *)
            printf 'Usage: GPU_COUNT={1|2} AUTODL_PRICE_PER_HOUR=<price> bash %s\n' "$0" >&2
            exit 64
            ;;
    esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
