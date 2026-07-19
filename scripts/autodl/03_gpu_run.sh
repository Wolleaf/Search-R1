#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib/runtime.sh
source "$SCRIPT_DIR/lib/runtime.sh"

TRAIN_ENV="$PROJECT_ROOT/envs/train"
RETRIEVER_ENV="$PROJECT_ROOT/envs/retriever"
BM25_INDEX="$PROJECT_ROOT/data/wiki-18-bm25-index/bm25"
CORPUS_ROOT="$PROJECT_ROOT/data/wiki-18-corpus"
CORPUS_JSONL="$CORPUS_ROOT/wiki-18.jsonl"
CORPUS_OFFSETS="$CORPUS_ROOT/wiki-18.offsets.u64"
HANDOFF="$MANIFEST_DIR/cpu_handoff.json"
RESULTS_DIR="$RUNS_ROOT/comparison"
GPU_COUNT="${GPU_COUNT:-}"
TRAIN_STEPS="${TRAIN_STEPS:-60}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-256}"
PRICE_PER_HOUR="${AUTODL_PRICE_PER_HOUR:-}"

validate_gpu_inputs() {
    [[ "$GPU_COUNT" == 1 || "$GPU_COUNT" == 2 ]] || {
        printf 'Set GPU_COUNT explicitly to 1 or 2. No GPU auto-detection is performed.\n' >&2
        return 64
    }
    [[ "$TRAIN_STEPS" =~ ^[1-9][0-9]*$ ]] || {
        printf 'TRAIN_STEPS must be a positive integer.\n' >&2
        return 64
    }
    [[ "$TRAIN_BATCH_SIZE" == 4 || "$TRAIN_BATCH_SIZE" == 2 ]] || {
        printf 'TRAIN_BATCH_SIZE must be 4 or the documented OOM fallback 2.\n' >&2
        return 64
    }
    [[ "$MAX_RESPONSE_LENGTH" == 256 || "$MAX_RESPONSE_LENGTH" == 192 ]] || {
        printf 'MAX_RESPONSE_LENGTH must be 256 or the documented OOM fallback 192.\n' >&2
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
}

finish_run_record() {
    local run_dir="$1" rc="$2" started_epoch="$3" started_at="$4"
    local budget_rmb="$5" timeout_seconds="$6"
    local finished_epoch elapsed state marker timed_out=false
    finished_epoch="$(date +%s)"
    elapsed=$((finished_epoch - started_epoch))
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
"gpu_count=$GPU_COUNT"$'\n'\
"train_steps=$TRAIN_STEPS"$'\n'\
"train_batch_size=$TRAIN_BATCH_SIZE"$'\n'\
"max_response_length=$MAX_RESPONSE_LENGTH"$'\n'\
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
    local mode="$1" variant="$2" argument="$3"
    local parent run_dir started_epoch started_at rc budget_rmb timeout_seconds
    case "$mode:$variant" in
        train:smoke) budget_rmb=30 ;;
        train:baseline|train:cost_aware) budget_rmb=100 ;;
        eval:baseline|eval:cost_aware) budget_rmb=15 ;;
        *) printf 'No budget is defined for %s:%s.\n' "$mode" "$variant" >&2; return 64 ;;
    esac
    timeout_seconds="$(awk -v budget="$budget_rmb" -v price="$PRICE_PER_HOUR" \
        'BEGIN { printf "%d", budget / price * 3600 }')"
    ((timeout_seconds > 0)) || {
        printf 'Computed timeout is not positive for %s:%s.\n' "$mode" "$variant" >&2
        return 64
    }
    parent="$RUNS_ROOT/$variant"
    mkdir -p "$parent/attempts"
    run_dir="$parent/attempts/$(date -u +'%Y%m%dT%H%M%SZ')-$$-$RANDOM"
    mkdir "$run_dir"
    : >"$run_dir/.running"
    atomic_write "$parent/latest" "$run_dir"$'\n'
    started_epoch="$(date +%s)"
    started_at="$(utc_now)"
    printf 'Starting %s %s (budget %s RMB, timeout %ss); log: %s/train.log\n' \
        "$mode" "$variant" "$budget_rmb" "$timeout_seconds" "$run_dir"

    set +e
    : >"$run_dir/train.log"
    AUTODL_CONFIG_ONLY=1 \
        OUTPUT_DIR="$run_dir/checkpoints" \
        GPU_COUNT="$GPU_COUNT" \
        TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
        MAX_RESPONSE_LENGTH="$MAX_RESPONSE_LENGTH" \
        AUTODL_ROOT="$PROJECT_ROOT" \
        bash "$CHECKOUT_DIR/scripts/autodl/train_small_grpo.sh" \
        "$mode" "$variant" "$argument" \
        >"$run_dir/resolved-config.yaml" 2>>"$run_dir/train.log"
    rc=$?
    if ((rc == 0)); then
        OUTPUT_DIR="$run_dir/checkpoints" \
        GPU_COUNT="$GPU_COUNT" \
        TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE" \
        MAX_RESPONSE_LENGTH="$MAX_RESPONSE_LENGTH" \
        AUTODL_ROOT="$PROJECT_ROOT" \
        timeout --signal=TERM --kill-after=120s "${timeout_seconds}s" \
        bash "$CHECKOUT_DIR/scripts/autodl/train_small_grpo.sh" \
        "$mode" "$variant" "$argument" >>"$run_dir/train.log" 2>&1
        rc=$?
    fi
    set -e
    finish_run_record "$run_dir" "$rc" "$started_epoch" "$started_at" \
        "$budget_rmb" "$timeout_seconds"
    LAST_RUN_DIR="$run_dir"
    if ((rc != 0)); then
        printf '%s %s failed with exit code %s; inspect %s/train.log\n' "$mode" "$variant" "$rc" "$run_dir" >&2
        return "$rc"
    fi
}

selected_checkpoint() {
    local selection_file="$1"
    "$TRAIN_ENV/bin/python" - "$selection_file" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))["checkpoint"])
PY
}

gpu_action() {
    local _attempt="$1"
    local commit python_version torch_version recorded_digest actual_digest
    local retriever_pid='' retriever_log ready=false
    local baseline_run cost_run baseline_eval cost_eval baseline_checkpoint cost_checkpoint
    validate_gpu_inputs
    commit="$(expected_commit)"
    verify_checkout "$commit"
    [[ -f "$MANIFEST_DIR/cpu.ok" && -f "$HANDOFF" && -f "$HANDOFF.sha256" ]] || {
        printf 'CPU phase is not sealed; refusing paid GPU work.\n' >&2
        return 1
    }

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

    run_job train smoke 1
    run_job train baseline "$TRAIN_STEPS"
    baseline_run="$LAST_RUN_DIR"
    "$TRAIN_ENV/bin/python" "$CHECKOUT_DIR/scripts/autodl/results.py" select \
        --variant baseline \
        --log "$baseline_run/train.log" \
        --checkpoint-root "$baseline_run/checkpoints" \
        --output "$baseline_run/selected_checkpoint.json"
    baseline_checkpoint="$(selected_checkpoint "$baseline_run/selected_checkpoint.json")"

    run_job train cost_aware "$TRAIN_STEPS"
    cost_run="$LAST_RUN_DIR"
    "$TRAIN_ENV/bin/python" "$CHECKOUT_DIR/scripts/autodl/results.py" select \
        --variant cost_aware \
        --log "$cost_run/train.log" \
        --checkpoint-root "$cost_run/checkpoints" \
        --output "$cost_run/selected_checkpoint.json"
    cost_checkpoint="$(selected_checkpoint "$cost_run/selected_checkpoint.json")"

    run_job eval baseline "$baseline_checkpoint"
    baseline_eval="$LAST_RUN_DIR"
    run_job eval cost_aware "$cost_checkpoint"
    cost_eval="$LAST_RUN_DIR"

    "$TRAIN_ENV/bin/python" "$CHECKOUT_DIR/scripts/autodl/results.py" summarize \
        --baseline-log "$baseline_eval/train.log" \
        --cost-log "$cost_eval/train.log" \
        --baseline-run-metadata "$baseline_run/run.env" \
        --cost-run-metadata "$cost_run/run.env" \
        --output-dir "$RESULTS_DIR"
    verify_checkout "$commit"
    atomic_write "$MANIFEST_DIR/gpu.ok" "$(sha256sum "$RESULTS_DIR/results.csv" | cut -d' ' -f1)"$'\n'
    sync_path "$MANIFEST_DIR"
    cleanup_retriever
    trap - EXIT
    printf 'Completed comparison: %s/results.md\n' "$RESULTS_DIR"
    printf 'The script does not shut down AutoDL. Confirm stopped state and billing in the console.\n'
}

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
