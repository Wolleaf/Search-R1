#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib/runtime.sh
source "$SCRIPT_DIR/lib/runtime.sh"

readonly MODEL_REVISION='15852e8c16360a2fea060d615a32b45270f8a8fc'
readonly DATA_REVISION='bcafb8dd07d453be3cbeeeb3f78be1841bddf92c'
readonly BM25_REVISION='2c7554f25f425038c4bcb155735a0f831851fd78'
readonly CORPUS_REVISION='69c1c00ffe7c5554c68d8548355cb22e46aabc51'
readonly CORPUS_SHA256='7abd929223399cd63c52b499f289bf4f9039be1e9f8c43e1cb3938305b2317db'
readonly CORPUS_BYTES=5123307260
readonly CORPUS_MEMBER_BYTES=14393573105
TRAIN_ENV="$PROJECT_ROOT/envs/train"
RETRIEVER_ENV="$PROJECT_ROOT/envs/retriever"
CACHE_ROOT="$PROJECT_ROOT/cache"
DATA_ROOT="$PROJECT_ROOT/data"
SMALL_DATA_DIR="$DATA_ROOT/nq_small"
SEARCH_GATE_DATA_DIR="$DATA_ROOT/search_opportunity_gate"
SEARCH_MIX_DATA_DIR="$DATA_ROOT/search_mix"
BM25_ROOT="$DATA_ROOT/wiki-18-bm25-index"
CORPUS_SOURCE_ROOT="$DATA_ROOT/wiki-18-corpus-source"
CORPUS_ROOT="$DATA_ROOT/wiki-18-corpus"
CORPUS_GZIP="$CORPUS_SOURCE_ROOT/wiki-18.jsonl.gz"
CORPUS_JSONL="$CORPUS_ROOT/wiki-18.jsonl"
CORPUS_OFFSETS="$CORPUS_ROOT/wiki-18.offsets.u64"
MODEL_DIR="$PROJECT_ROOT/models/Qwen3.5-2B"
HANDOFF="$MANIFEST_DIR/cpu_handoff.json"

export CUDA_VISIBLE_DEVICES=''
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$CHECKOUT_DIR${PYTHONPATH:+:$PYTHONPATH}"

check_runtime() {
    local python_bin="$1"
    "$python_bin" - <<'PY'
import sys
if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"Python 3.12 is required, found {sys.version.split()[0]}")
import torch
if torch.__version__.split("+")[0] != "2.8.0":
    raise SystemExit(f"PyTorch 2.8.0 is required, found {torch.__version__}")
if torch.version.cuda != "12.8":
    raise SystemExit(f"CUDA 12.8 PyTorch build is required, found {torch.version.cuda}")
PY
}

resolve_llmdevelop_python() {
    local conda_bin output
    if [[ -n "${AUTODL_PYTHON:-}" ]]; then
        printf '%s\n' "$AUTODL_PYTHON"
        return 0
    fi
    if command -v conda >/dev/null 2>&1; then
        conda_bin="$(command -v conda)"
    elif [[ -x /root/miniconda3/bin/conda ]]; then
        conda_bin=/root/miniconda3/bin/conda
    else
        printf 'Cannot find conda; set AUTODL_PYTHON to llmdevelop/bin/python.\n' >&2
        return 1
    fi
    output="$("$conda_bin" run -n llmdevelop python -c 'import sys; print(sys.executable)')"
    printf '%s\n' "$output" | awk 'NF {value=$0} END {print value}'
}

cpu_action() {
    local _attempt="$1"
    local commit base_python train_python retriever_python python_version torch_version handoff_digest gpu_count
    local previous_commit previous_python previous_torch build_search_mix=0 seal_search_mix=0
    local spec mode variant steps model_path config_output_dir parent_placeholder trace_placeholder
    local config_response_length eval_group_size
    local trace_digest_placeholder trace_checkpoint_digest trace_parent_digest trace_output trace_stage
    local -a command_args search_mix_handoff_args
    commit="$(expected_commit)"
    verify_checkout "$commit"
    [[ -f "$CHECKOUT_DIR/requirements-autodl.lock" ]] || {
        printf 'Missing requirements-autodl.lock in the pinned checkout.\n' >&2
        return 1
    }
    rm -f -- "$MANIFEST_DIR/cpu.ok"
    sync_path "$MANIFEST_DIR"

    [[ "${AUTODL_RESEAL_ONLY:-0}" == 0 || "${AUTODL_RESEAL_ONLY:-0}" == 1 ]] || {
        printf 'AUTODL_RESEAL_ONLY must be 0 or 1.\n' >&2
        return 64
    }
    [[ "${AUTODL_SEARCH_GATE_INCREMENTAL:-0}" == 0 ||
        "${AUTODL_SEARCH_GATE_INCREMENTAL:-0}" == 1 ]] || {
        printf 'AUTODL_SEARCH_GATE_INCREMENTAL must be 0 or 1.\n' >&2
        return 64
    }
    [[ "${AUTODL_SEARCH_MIX_INCREMENTAL:-0}" == 0 ||
        "${AUTODL_SEARCH_MIX_INCREMENTAL:-0}" == 1 ]] || {
        printf 'AUTODL_SEARCH_MIX_INCREMENTAL must be 0 or 1.\n' >&2
        return 64
    }
    if (( ${AUTODL_RESEAL_ONLY:-0} + ${AUTODL_SEARCH_GATE_INCREMENTAL:-0} +
          ${AUTODL_SEARCH_MIX_INCREMENTAL:-0} > 1 )); then
        printf 'Choose only one incremental or offline reseal mode.\n' >&2
        return 64
    fi

    if [[ "${AUTODL_RESEAL_ONLY:-0}" == 1 ||
          "${AUTODL_SEARCH_GATE_INCREMENTAL:-0}" == 1 ||
          "${AUTODL_SEARCH_MIX_INCREMENTAL:-0}" == 1 ]]; then
        export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PIP_NO_INDEX=1
        train_python="$TRAIN_ENV/bin/python"
        retriever_python="$RETRIEVER_ENV/bin/python"
        [[ -x "$train_python" && -x "$retriever_python" &&
            -f "$HANDOFF" && -f "$HANDOFF.sha256" ]] || {
            printf 'Incremental reseal requires the existing environments and CPU handoff.\n' >&2
            return 1
        }
        check_runtime "$train_python"
        "$train_python" -m pip check
        "$retriever_python" -c 'import sys; assert sys.version_info[:2] == (3, 12)'
        IFS=$'\t' read -r previous_commit previous_python previous_torch < <(
            "$train_python" - "$HANDOFF" <<'PY'
import json
from pathlib import Path
import sys

payload = json.loads(Path(sys.argv[1]).read_text())
print(payload["checkout_commit"], payload["python_version"], payload["torch_version"], sep="\t")
PY
        )
        "$train_python" "$CHECKOUT_DIR/scripts/autodl/handoff.py" verify \
            --root "$PROJECT_ROOT" \
            --commit "$previous_commit" \
            --python-version "$previous_python" \
            --torch-version "$previous_torch" \
            --manifest "$HANDOFF"
        "$train_python" -m pip freeze --all >"$_attempt/train-freeze.current.txt"
        "$retriever_python" -m pip freeze --all >"$_attempt/retriever-freeze.current.txt"
        cmp -s "$MANIFEST_DIR/train-freeze.txt" "$_attempt/train-freeze.current.txt" || {
            printf 'Train environment changed since the previous handoff.\n' >&2
            return 1
        }
        cmp -s "$MANIFEST_DIR/retriever-freeze.txt" "$_attempt/retriever-freeze.current.txt" || {
            printf 'Retriever environment changed since the previous handoff.\n' >&2
            return 1
        }
        java -version >"$MANIFEST_DIR/java-version.txt" 2>&1
        grep -Eq 'version "21([.]|")' "$MANIFEST_DIR/java-version.txt" || {
            printf 'OpenJDK 21 is required by Pyserini 1.1.\n' >&2
            return 1
        }
        if [[ "${AUTODL_SEARCH_GATE_INCREMENTAL:-0}" == 1 ]]; then
            unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE
            "$train_python" "$CHECKOUT_DIR/scripts/data_process/multihop_search_gate.py" build \
                --local-dir "$SEARCH_GATE_DATA_DIR"
            export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
        fi
        if [[ "${AUTODL_SEARCH_MIX_INCREMENTAL:-0}" == 1 ]]; then
            unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE PIP_NO_INDEX
            "$train_python" "$CHECKOUT_DIR/scripts/data_process/search_mix.py" download \
                --local-dir "$SEARCH_MIX_DATA_DIR"
            export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PIP_NO_INDEX=1
            build_search_mix=1
        fi
    elif [[ "${AUTODL_RESEAL_ONLY:-0}" == 0 ]]; then
        base_python="$(resolve_llmdevelop_python)"
        [[ -x "$base_python" ]] || {
            printf 'llmdevelop Python is not executable: %s\n' "$base_python" >&2
            return 1
        }
        check_runtime "$base_python"

        mkdir -p "$PROJECT_ROOT/envs" "$CACHE_ROOT" "$DATA_ROOT" "$PROJECT_ROOT/models" "$MANIFEST_DIR"
        if [[ ! -x "$TRAIN_ENV/bin/python" ]]; then
            "$base_python" -m venv --system-site-packages "$TRAIN_ENV"
        fi
        train_python="$TRAIN_ENV/bin/python"
        check_runtime "$train_python"
        "$train_python" -m pip install --requirement "$CHECKOUT_DIR/requirements-autodl.lock"
        "$train_python" -m pip check
        check_runtime "$train_python"

        # Pyserini 1.1 is kept isolated so it cannot replace the image's PyTorch.
        apt-get update
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends openjdk-21-jre-headless
        java -version >"$MANIFEST_DIR/java-version.txt" 2>&1
        grep -Eq 'version "21([.]|")' "$MANIFEST_DIR/java-version.txt" || {
            printf 'OpenJDK 21 is required by Pyserini 1.1.\n' >&2
            return 1
        }
        if [[ ! -x "$RETRIEVER_ENV/bin/python" ]]; then
            "$base_python" -m venv "$RETRIEVER_ENV"
        fi
        retriever_python="$RETRIEVER_ENV/bin/python"
        "$retriever_python" -m pip install \
            'pyserini==1.1.0' --no-deps
        "$retriever_python" -m pip install \
            'pyjnius>=1.6,<2' \
            'fastapi==0.139.2' \
            'uvicorn==0.51.0' \
            'pydantic>=2.10,<3'
        "$retriever_python" -c 'import sys; assert sys.version_info[:2] == (3, 12)'

        "$train_python" - "$MODEL_DIR" "$BM25_ROOT" "$CORPUS_SOURCE_ROOT" <<PY
from huggingface_hub import snapshot_download
import sys

model_dir, bm25_dir, corpus_source_dir = sys.argv[1:]
# Keep snapshot downloads within the memory limit of AutoDL's CPU-only mode.
snapshot_download(
    repo_id="Qwen/Qwen3.5-2B",
    revision="$MODEL_REVISION",
    local_dir=model_dir,
    max_workers=1,
)
snapshot_download(
    repo_id="PeterJinGo/wiki-18-bm25-index",
    repo_type="dataset",
    revision="$BM25_REVISION",
    local_dir=bm25_dir,
    max_workers=1,
)
snapshot_download(
    repo_id="PeterJinGo/wiki-18-corpus",
    repo_type="dataset",
    revision="$CORPUS_REVISION",
    local_dir=corpus_source_dir,
    allow_patterns=["wiki-18.jsonl.gz"],
    max_workers=1,
)
PY

        "$train_python" "$CHECKOUT_DIR/scripts/autodl/build_corpus_offsets.py" prepare \
        --source "$CORPUS_GZIP" \
        --output-dir "$CORPUS_ROOT" \
        --revision "$CORPUS_REVISION" \
        --source-sha256 "$CORPUS_SHA256" \
        --source-bytes "$CORPUS_BYTES" \
        --member-bytes "$CORPUS_MEMBER_BYTES"

        "$train_python" "$CHECKOUT_DIR/scripts/data_process/nq_small.py" \
        --local-dir "$SMALL_DATA_DIR" \
        --revision "$DATA_REVISION" \
        --seed 42 \
        --train-size 512 \
        --val-size 64 \
        --test-size 128
        "$train_python" "$CHECKOUT_DIR/scripts/data_process/multihop_search_gate.py" build \
            --local-dir "$SEARCH_GATE_DATA_DIR"
        "$train_python" "$CHECKOUT_DIR/scripts/data_process/search_mix.py" download \
            --local-dir "$SEARCH_MIX_DATA_DIR"
        export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PIP_NO_INDEX=1
        build_search_mix=1
    fi

    if [[ "$build_search_mix" == 1 ]]; then
        PYTHONPATH="$CHECKOUT_DIR" "$retriever_python" \
            "$CHECKOUT_DIR/scripts/data_process/search_mix.py" retrieve \
            --local-dir "$SEARCH_MIX_DATA_DIR" \
            --index-path "$BM25_ROOT/bm25" \
            --corpus-path "$CORPUS_JSONL" \
            --offsets-path "$CORPUS_OFFSETS"
        "$train_python" "$CHECKOUT_DIR/scripts/data_process/search_mix.py" materialize \
            --local-dir "$SEARCH_MIX_DATA_DIR" \
            --model-dir "$MODEL_DIR" \
            --eval-catalog "$SEARCH_GATE_DATA_DIR/catalog.jsonl" \
            --eval-parquet "$SMALL_DATA_DIR/test_128.parquet"
        seal_search_mix=1
    elif [[ -f "$SEARCH_MIX_DATA_DIR/manifest.json" &&
            ! -L "$SEARCH_MIX_DATA_DIR/manifest.json" ]]; then
        # Legacy reseals may include an already-complete mix, but do not require one.
        seal_search_mix=1
    elif [[ -e "$SEARCH_MIX_DATA_DIR/manifest.json" ||
            -L "$SEARCH_MIX_DATA_DIR/manifest.json" ]]; then
        printf 'Search-mix manifest exists but is not a regular non-symlink file.\n' >&2
        return 1
    fi
    if [[ "$seal_search_mix" == 1 ]]; then
        # Use a fresh process so the large parsed evidence pool is not held twice.
        "$train_python" "$CHECKOUT_DIR/scripts/data_process/search_mix.py" verify \
            --manifest "$SEARCH_MIX_DATA_DIR/manifest.json" \
            --model-dir "$MODEL_DIR" \
            --eval-catalog "$SEARCH_GATE_DATA_DIR/catalog.jsonl" \
            --eval-parquet "$SMALL_DATA_DIR/test_128.parquet"
        PYTHONPATH="$CHECKOUT_DIR" "$retriever_python" \
            "$CHECKOUT_DIR/scripts/data_process/search_mix.py" replay \
            --manifest "$SEARCH_MIX_DATA_DIR/manifest.json" \
            --index-path "$BM25_ROOT/bm25" \
            --corpus-path "$CORPUS_JSONL" \
            --offsets-path "$CORPUS_OFFSETS"
    fi

    "$train_python" - "$MODEL_DIR" "$SMALL_DATA_DIR" <<'PY'
import json
from pathlib import Path
import sys

import pandas as pd
from transformers import AutoTokenizer

model_dir, data_dir = map(Path, sys.argv[1:])
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
if not tokenizer("Who wrote Hamlet?")["input_ids"]:
    raise SystemExit("tokenizer smoke test returned no tokens")
expected = {"train_512.parquet": 512, "val_64.parquet": 64, "test_128.parquet": 128}
for name, rows in expected.items():
    actual = len(pd.read_parquet(data_dir / name))
    if actual != rows:
        raise SystemExit(f"{name}: expected {rows} rows, found {actual}")
manifest = json.loads((data_dir / "manifest.json").read_text())
if not manifest["overlap_checks"]["passed"]:
    raise SystemExit("NQ split overlap validation failed")
PY
    "$train_python" "$CHECKOUT_DIR/scripts/data_process/multihop_search_gate.py" verify \
        --manifest "$SEARCH_GATE_DATA_DIR/manifest.json"

    PYTHONPATH="$CHECKOUT_DIR" "$retriever_python" - \
        "$BM25_ROOT/bm25" "$CORPUS_JSONL" "$CORPUS_OFFSETS" <<'PY'
from search_r1.search.bm25_server import BM25Retriever
import sys

retriever = BM25Retriever(
    sys.argv[1],
    topk=1,
    corpus_path=sys.argv[2],
    offsets_path=sys.argv[3],
)
hits = retriever.search("Who wrote Hamlet?", topk=1, return_scores=True)
if not hits or not hits[0]["document"]["contents"]:
    raise SystemExit("BM25 smoke query returned no document")
PY

    "$train_python" -m pytest -q -p no:cacheprovider "$CHECKOUT_DIR/tests"
    PYTHON_BIN="$train_python" bash "$CHECKOUT_DIR/scripts/autodl/tests/test_runtime.sh"
    bash "$CHECKOUT_DIR/scripts/autodl/tests/test_gated_config.sh"
    bash "$CHECKOUT_DIR/scripts/autodl/tests/test_gated_followup.sh"
    bash "$CHECKOUT_DIR/scripts/autodl/tests/test_search_opportunity_pipeline.sh"
    PYTHON_BIN="$train_python" bash "$CHECKOUT_DIR/scripts/autodl/tests/test_group_probe_pipeline.sh"
    bash "$CHECKOUT_DIR/scripts/autodl/tests/test_shutdown_watchdog.sh"
    "$train_python" "$CHECKOUT_DIR/scripts/autodl/tests/test_results.py"
    "$train_python" "$CHECKOUT_DIR/scripts/autodl/tests/test_paired_eval.py"
    "$train_python" "$CHECKOUT_DIR/scripts/autodl/tests/test_search_opportunity_gate.py"
    "$train_python" "$CHECKOUT_DIR/scripts/autodl/tests/test_export_gated_training.py"
    "$train_python" - "$CHECKOUT_DIR" <<'PY'
from pathlib import Path
import sys

checkout = Path(sys.argv[1])
for relative_root in ("search_r1", "verl", "scripts"):
    for path in (checkout / relative_root).rglob("*.py"):
        compile(path.read_bytes(), str(path), "exec")
PY

    config_output_dir="$PROJECT_ROOT/cache/config-compose/resolved-output"
    parent_placeholder="$PROJECT_ROOT/cache/config-compose/reproduced-checkpoint-placeholder"
    trace_placeholder="$PROJECT_ROOT/cache/config-compose/trace-output-placeholder"
    trace_digest_placeholder="$(printf 'a%.0s' {1..64})"
    mkdir -p "$parent_placeholder" "$trace_placeholder"
    for gpu_count in 1 2; do
        for spec in \
            'train|smoke|2|' \
            'train|reproduce|60|' \
            "train|control|20|$parent_placeholder" \
            "train|cost_aware|20|$parent_placeholder" \
            "train|cost_aware_gated|20|$parent_placeholder" \
            "eval|base||$MODEL_DIR" \
            "eval|reproduced||$parent_placeholder" \
            "eval|control||$parent_placeholder" \
            "eval|cost_aware||$parent_placeholder" \
            "eval|cost_aware_gated||$parent_placeholder" \
            "eval|search_opportunity||$parent_placeholder" \
            "eval|group_probe||$MODEL_DIR"; do
            IFS='|' read -r mode variant steps model_path <<<"$spec"
            if [[ "$variant" == group_probe && "$seal_search_mix" != 1 ]]; then
                continue
            fi
            command_args=("$mode" "$variant")
            case "$mode:$variant" in
                train:smoke|train:reproduce)
                    command_args+=("$steps")
                    ;;
                train:control|train:cost_aware|train:cost_aware_gated)
                    command_args+=("$steps" "$model_path")
                    ;;
                eval:*)
                    command_args+=("$model_path")
                    ;;
            esac
            config_response_length=256
            trace_output=''
            trace_stage=''
            trace_checkpoint_digest=''
            trace_parent_digest=''
            eval_data_file=''
            eval_group_size=1
            if [[ "$variant" == group_probe ]]; then
                config_response_length=500
            fi
            if [[ "$variant" == cost_aware_gated || "$variant" == search_opportunity ||
                  "$variant" == group_probe ]]; then
                trace_output="$trace_placeholder"
                trace_stage="$variant"
                if [[ "$mode" == eval ]]; then
                    trace_checkpoint_digest="$trace_digest_placeholder"
                else
                    trace_parent_digest="$trace_digest_placeholder"
                fi
            fi
            if [[ "$variant" == search_opportunity ]]; then
                eval_data_file="$SEARCH_GATE_DATA_DIR/eval_256.parquet"
            elif [[ "$variant" == group_probe ]]; then
                eval_data_file="$SEARCH_MIX_DATA_DIR/probe_multi_64.parquet"
                eval_group_size=5
            fi
            AUTODL_CONFIG_ONLY=1 \
                AUTODL_ROOT="$PROJECT_ROOT" \
                GPU_COUNT="$gpu_count" \
                MAX_RESPONSE_LENGTH="$config_response_length" \
                OUTPUT_DIR="$config_output_dir" \
                EVAL_DATA_FILE="$eval_data_file" \
                EVAL_GROUP_SIZE="$eval_group_size" \
                TRACE_OUTPUT_DIR="$trace_output" \
                TRACE_STAGE="$trace_stage" \
                TRACE_RUN_ID=config-compose \
                TRACE_CHECKPOINT_DIGEST="$trace_checkpoint_digest" \
                TRACE_PARENT_CHECKPOINT_DIGEST="$trace_parent_digest" \
                bash "$CHECKOUT_DIR/scripts/autodl/train_small_grpo.sh" \
                "${command_args[@]}" \
                >"$MANIFEST_DIR/config-${gpu_count}gpu-$variant-$mode.yaml"
        done
        AUTODL_CONFIG_ONLY=1 \
            AUTODL_ROOT="$PROJECT_ROOT" \
            GPU_COUNT="$gpu_count" \
            MAX_RESPONSE_LENGTH=256 \
            OUTPUT_DIR="$config_output_dir" \
            TRACE_OUTPUT_DIR="$trace_placeholder" \
            TRACE_STAGE=cost_aware_gated_gate \
            TRACE_RUN_ID=config-compose \
            TRACE_PARENT_CHECKPOINT_DIGEST="$trace_digest_placeholder" \
            bash "$CHECKOUT_DIR/scripts/autodl/train_small_grpo.sh" \
            train cost_aware_gated 2 "$parent_placeholder" \
            >"$MANIFEST_DIR/config-${gpu_count}gpu-cost_aware_gated-gate-train.yaml"
        for variant in control cost_aware; do
            AUTODL_CONFIG_ONLY=1 \
                AUTODL_ROOT="$PROJECT_ROOT" \
                GPU_COUNT="$gpu_count" \
                MAX_RESPONSE_LENGTH=256 \
                OUTPUT_DIR="$config_output_dir" \
                TRACE_OUTPUT_DIR="$trace_placeholder" \
                TRACE_STAGE="$variant" \
                TRACE_RUN_ID=config-compose \
                TRACE_CHECKPOINT_DIGEST="$trace_digest_placeholder" \
                bash "$CHECKOUT_DIR/scripts/autodl/train_small_grpo.sh" \
                eval "$variant" "$parent_placeholder" \
                >"$MANIFEST_DIR/config-${gpu_count}gpu-$variant-trace-eval.yaml"
        done
    done

    "$train_python" - \
        "$MANIFEST_DIR" \
        "$MODEL_DIR" \
        "$parent_placeholder" \
        "$trace_placeholder" \
        "$SEARCH_GATE_DATA_DIR/eval_256.parquet" \
        "$SEARCH_MIX_DATA_DIR/probe_multi_64.parquet" \
        "$seal_search_mix" <<'PY'
from copy import deepcopy
from pathlib import Path
import sys

from omegaconf import OmegaConf

manifest_dir, model_dir, parent_placeholder, trace_placeholder, search_gate_data, group_probe_data = map(
    Path, sys.argv[1:7]
)
seal_search_mix = sys.argv[7] == "1"
train_variants = ("smoke", "reproduce", "control", "cost_aware", "cost_aware_gated")
eval_variants = (
    "base", "reproduced", "control", "cost_aware", "cost_aware_gated",
    "search_opportunity",
) + (("group_probe",) if seal_search_mix else ())

for gpu_count in (1, 2):
    configs = {}
    for mode, variants in (("train", train_variants), ("eval", eval_variants)):
        for variant in variants:
            path = manifest_dir / f"config-{gpu_count}gpu-{variant}-{mode}.yaml"
            config = OmegaConf.load(path)
            configs[(mode, variant)] = config
            trace_output = config.trainer.get("trace_output_dir", None)
            if variant in ("cost_aware_gated", "search_opportunity", "group_probe"):
                if Path(trace_output) != trace_placeholder:
                    raise SystemExit(f"trace output placeholder mismatch in {path}")
            elif trace_output:
                raise SystemExit(f"legacy config unexpectedly enables traces in {path}")
            group_size = config.actor_rollout_ref.rollout.n_agent
            mini_batch_size = config.actor_rollout_ref.actor.ppo_mini_batch_size
            expected_mini_batch_size = config.data.train_batch_size * group_size
            if group_size != 5 or mini_batch_size != expected_mini_batch_size:
                raise SystemExit(f"GRPO group or actor mini-batch mismatch in {path}")
            if config.data.train_batch_size != 8:
                raise SystemExit(f"default train batch size must be 8 in {path}")
            expected_eval_group_size = 5 if variant == "group_probe" else 1
            expected_val_batch_size = 8 if variant == "group_probe" else 64
            if (config.data.eval_group_size != expected_eval_group_size
                    or config.data.val_batch_size != expected_val_batch_size):
                raise SystemExit(f"evaluation grouping mismatch in {path}")
            if config.actor_rollout_ref.actor.optim.lr_warmup_steps_ratio != 0.285:
                raise SystemExit(f"actor warmup ratio must be 0.285 in {path}")
            if config.actor_rollout_ref.rollout.top_p != 1.0:
                raise SystemExit(f"rollout top-p must be 1.0 in {path}")
            if config.trainer.n_gpus_per_node != gpu_count:
                raise SystemExit(f"GPU count mismatch in {path}")
            if config.max_turns != 4:
                raise SystemExit(f"max_turns must be 4 in {path}")
            if config.retriever.topk != 3:
                raise SystemExit(f"retriever top-k must be 3 in {path}")
            if config.data.max_start_length != 1024:
                raise SystemExit(f"max_start_length must be 1024 in {path}")
            expected_response_length = 500 if variant == "group_probe" else 256
            expected_prompt_length = 4096 if variant == "group_probe" else 3584
            if config.data.max_response_length != expected_response_length:
                raise SystemExit(f"max_response_length mismatch in {path}")
            if config.data.max_obs_length != 384:
                raise SystemExit(f"max_obs_length must be 384 in {path}")
            if config.data.max_prompt_length != expected_prompt_length:
                raise SystemExit(f"max_prompt_length mismatch in {path}")
            expected_wrap_classes = ["Qwen3_5DecoderLayer"]
            actor_wrap_classes = list(
                config.actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap
            )
            ref_wrap_classes = list(
                config.actor_rollout_ref.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap
            )
            if actor_wrap_classes != expected_wrap_classes or ref_wrap_classes != expected_wrap_classes:
                raise SystemExit(f"Qwen3.5 FSDP wrap policy mismatch in {path}")

    smoke = configs[("train", "smoke")]
    reproduced = configs[("train", "reproduce")]
    control = configs[("train", "control")]
    cost_aware = configs[("train", "cost_aware")]
    cost_aware_gated = configs[("train", "cost_aware_gated")]
    gated_gate_path = manifest_dir / f"config-{gpu_count}gpu-cost_aware_gated-gate-train.yaml"
    gated_gate = OmegaConf.load(gated_gate_path)
    if Path(gated_gate.trainer.trace_output_dir) != trace_placeholder:
        raise SystemExit(f"trace output placeholder mismatch in {gated_gate_path}")
    if (gated_gate.data.max_response_length != 256
            or gated_gate.data.max_prompt_length != 3584):
        raise SystemExit(f"historical response lengths mismatch in {gated_gate_path}")
    trace_eval_configs = {}
    for variant in ("control", "cost_aware"):
        path = manifest_dir / f"config-{gpu_count}gpu-{variant}-trace-eval.yaml"
        trace_config = OmegaConf.load(path)
        trace_eval_configs[variant] = trace_config
        if (Path(trace_config.trainer.trace_output_dir) != trace_placeholder
                or trace_config.trainer.trace_stage != variant
                or trace_config.trainer.trace_checkpoint_digest != "a" * 64
                or not trace_config.trainer.val_only
                or trace_config.data.max_response_length != 256
                or trace_config.data.max_prompt_length != 3584):
            raise SystemExit(f"trace-only evaluation contract mismatch in {path}")
    expected_steps = (
        (smoke, 2), (reproduced, 60), (control, 20), (cost_aware, 20),
        (cost_aware_gated, 20), (gated_gate, 2),
    )
    for config, steps in expected_steps:
        if config.trainer.total_training_steps != steps:
            raise SystemExit(f"training step mismatch for {config.trainer.experiment_name}")
        if config.trainer.save_freq != steps or config.trainer.test_freq != steps:
            raise SystemExit(f"checkpoint/validation is not fixed to the final step for {config.trainer.experiment_name}")

    if Path(reproduced.actor_rollout_ref.model.path) != model_dir:
        raise SystemExit("reproduction must start from the prepared base model")
    for config in (control, cost_aware, cost_aware_gated, gated_gate):
        if Path(config.actor_rollout_ref.model.path) != parent_placeholder:
            raise SystemExit("second-stage branch does not use the reproduced-checkpoint placeholder")
    train_lambdas = {
        variant: configs[("train", variant)].algorithm.cost_lambda for variant in train_variants
    }
    if train_lambdas != {
        "smoke": 0.0, "reproduce": 0.0, "control": 0.0,
        "cost_aware": 0.10, "cost_aware_gated": 0.10,
    }:
        raise SystemExit(f"unexpected training cost lambdas: {train_lambdas}")
    train_reward_modes = {
        variant: configs[("train", variant)].algorithm.cost_reward_mode
        for variant in train_variants
    }
    if train_reward_modes != {
        "smoke": "linear", "reproduce": "linear", "control": "linear",
        "cost_aware": "linear", "cost_aware_gated": "correct_only",
    } or gated_gate.algorithm.cost_reward_mode != "correct_only":
        raise SystemExit(f"unexpected training reward modes: {train_reward_modes}")

    normalized_control = deepcopy(OmegaConf.to_container(control, resolve=True))
    normalized_cost = deepcopy(OmegaConf.to_container(cost_aware, resolve=True))
    for normalized in (normalized_control, normalized_cost):
        normalized["algorithm"]["cost_lambda"] = None
        normalized["algorithm"]["cost_reward_mode"] = None
        normalized["trainer"]["experiment_name"] = None
        normalized["trainer"]["trace_output_dir"] = None
        normalized["trainer"]["trace_stage"] = None
        normalized["trainer"]["trace_run_id"] = None
        normalized["trainer"]["trace_checkpoint_digest"] = None
        normalized["trainer"]["trace_parent_checkpoint_digest"] = None
    if normalized_control != normalized_cost:
        raise SystemExit("current stage-2 configs differ beyond reward settings and variant")

    for variant, trace_config in trace_eval_configs.items():
        plain = deepcopy(OmegaConf.to_container(configs[("eval", variant)], resolve=True))
        traced = deepcopy(OmegaConf.to_container(trace_config, resolve=True))
        for normalized in (plain, traced):
            for key in (
                "trace_output_dir", "trace_stage", "trace_run_id",
                "trace_checkpoint_digest", "trace_parent_checkpoint_digest",
            ):
                normalized["trainer"][key] = None
        for key in ("max_response_length", "max_prompt_length"):
            plain["data"][key] = None
            traced["data"][key] = None
        if plain != traced:
            raise SystemExit(f"trace-only evaluation changes scientific config for {variant}")

    for variant in eval_variants:
        config = configs[("eval", variant)]
        expected_path = model_dir if variant in ("base", "group_probe") else parent_placeholder
        if Path(config.actor_rollout_ref.model.path) != expected_path:
            raise SystemExit(f"evaluation model placeholder mismatch for {variant}")
        if (config.algorithm.cost_lambda != 0.10
                or config.algorithm.cost_reward_mode != "linear"
                or not config.trainer.val_only):
            raise SystemExit(f"evaluation contract mismatch for {variant}")
        if variant == "search_opportunity":
            expected_val = search_gate_data
        elif variant == "group_probe":
            expected_val = group_probe_data
        else:
            expected_val = Path(configs[("eval", "control")].data.val_files)
        if Path(config.data.val_files) != expected_val:
            raise SystemExit(f"evaluation data path mismatch for {variant}")
PY

    "$train_python" -m pip freeze --all >"$MANIFEST_DIR/train-freeze.txt"
    "$retriever_python" -m pip freeze --all >"$MANIFEST_DIR/retriever-freeze.txt"
    python_version="$($train_python -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
    torch_version="$($train_python -c 'import torch; print(torch.__version__)')"

    search_mix_handoff_args=()
    if [[ "$seal_search_mix" == 1 ]]; then
        search_mix_handoff_args+=(
            --data "$SEARCH_MIX_DATA_DIR"
            --extra-file "$MANIFEST_DIR/config-1gpu-group_probe-eval.yaml"
            --extra-file "$MANIFEST_DIR/config-2gpu-group_probe-eval.yaml"
        )
    fi

    "$train_python" "$CHECKOUT_DIR/scripts/autodl/handoff.py" create \
        --root "$PROJECT_ROOT" \
        --commit "$commit" \
        --model "$MODEL_DIR" \
        --bm25 "$BM25_ROOT/bm25" \
        --corpus "$CORPUS_ROOT" \
        --data "$SMALL_DATA_DIR" \
        "${search_mix_handoff_args[@]}" \
        --requirements "$CHECKOUT_DIR/requirements-autodl.lock" \
        --extra-file "$CORPUS_GZIP" \
        --extra-file "$MANIFEST_DIR/train-freeze.txt" \
        --extra-file "$MANIFEST_DIR/retriever-freeze.txt" \
        --extra-file "$MANIFEST_DIR/java-version.txt" \
        --extra-file "$MANIFEST_DIR/config-1gpu-smoke-train.yaml" \
        --extra-file "$MANIFEST_DIR/config-1gpu-reproduce-train.yaml" \
        --extra-file "$MANIFEST_DIR/config-1gpu-control-train.yaml" \
        --extra-file "$MANIFEST_DIR/config-1gpu-cost_aware-train.yaml" \
        --extra-file "$MANIFEST_DIR/config-1gpu-cost_aware_gated-train.yaml" \
        --extra-file "$MANIFEST_DIR/config-1gpu-cost_aware_gated-gate-train.yaml" \
        --extra-file "$MANIFEST_DIR/config-1gpu-base-eval.yaml" \
        --extra-file "$MANIFEST_DIR/config-1gpu-reproduced-eval.yaml" \
        --extra-file "$MANIFEST_DIR/config-1gpu-control-eval.yaml" \
        --extra-file "$MANIFEST_DIR/config-1gpu-cost_aware-eval.yaml" \
        --extra-file "$MANIFEST_DIR/config-1gpu-cost_aware_gated-eval.yaml" \
        --extra-file "$MANIFEST_DIR/config-1gpu-control-trace-eval.yaml" \
        --extra-file "$MANIFEST_DIR/config-1gpu-cost_aware-trace-eval.yaml" \
        --extra-file "$MANIFEST_DIR/config-2gpu-smoke-train.yaml" \
        --extra-file "$MANIFEST_DIR/config-2gpu-reproduce-train.yaml" \
        --extra-file "$MANIFEST_DIR/config-2gpu-control-train.yaml" \
        --extra-file "$MANIFEST_DIR/config-2gpu-cost_aware-train.yaml" \
        --extra-file "$MANIFEST_DIR/config-2gpu-cost_aware_gated-train.yaml" \
        --extra-file "$MANIFEST_DIR/config-2gpu-cost_aware_gated-gate-train.yaml" \
        --extra-file "$MANIFEST_DIR/config-2gpu-base-eval.yaml" \
        --extra-file "$MANIFEST_DIR/config-2gpu-reproduced-eval.yaml" \
        --extra-file "$MANIFEST_DIR/config-2gpu-control-eval.yaml" \
        --extra-file "$MANIFEST_DIR/config-2gpu-cost_aware-eval.yaml" \
        --extra-file "$MANIFEST_DIR/config-2gpu-cost_aware_gated-eval.yaml" \
        --extra-file "$MANIFEST_DIR/config-2gpu-control-trace-eval.yaml" \
        --extra-file "$MANIFEST_DIR/config-2gpu-cost_aware-trace-eval.yaml" \
        --extra-file "$SEARCH_GATE_DATA_DIR/eval_256.parquet" \
        --extra-file "$SEARCH_GATE_DATA_DIR/catalog.jsonl" \
        --extra-file "$SEARCH_GATE_DATA_DIR/manifest.json" \
        --extra-file "$SEARCH_GATE_DATA_DIR/manifest.json.sha256" \
        --extra-file "$SEARCH_GATE_DATA_DIR/sources/hotpotqa/dev.jsonl" \
        --extra-file "$SEARCH_GATE_DATA_DIR/sources/2wikimultihopqa/dev.jsonl" \
        --extra-file "$MANIFEST_DIR/config-1gpu-search_opportunity-eval.yaml" \
        --extra-file "$MANIFEST_DIR/config-2gpu-search_opportunity-eval.yaml" \
        --python-version "$python_version" \
        --torch-version "$torch_version" \
        --output "$HANDOFF"
    "$train_python" "$CHECKOUT_DIR/scripts/autodl/handoff.py" verify \
        --root "$PROJECT_ROOT" \
        --commit "$commit" \
        --python-version "$python_version" \
        --torch-version "$torch_version" \
        --manifest "$HANDOFF"
    verify_checkout "$commit"

    handoff_digest="$(cut -d' ' -f1 "$HANDOFF.sha256")"
    atomic_write "$MANIFEST_DIR/cpu.ok" "$handoff_digest"$'\n'
    sync_path "$MANIFEST_DIR"
}

case "${1:-}" in
    --worker)
        phase_worker cpu "${2:?missing attempt directory}" "$0"
        ;;
    --action)
        cpu_action "${2:?missing attempt directory}"
        ;;
    '')
        phase_launch cpu "$0"
        ;;
    *)
        printf 'Usage: bash %s\n' "$0" >&2
        exit 64
        ;;
esac
