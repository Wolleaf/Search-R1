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
    commit="$(expected_commit)"
    verify_checkout "$commit"
    [[ -f "$CHECKOUT_DIR/requirements-autodl.lock" ]] || {
        printf 'Missing requirements-autodl.lock in the pinned checkout.\n' >&2
        return 1
    }

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
    "$train_python" "$CHECKOUT_DIR/scripts/autodl/tests/test_results.py"
    "$train_python" - "$CHECKOUT_DIR" <<'PY'
from pathlib import Path
import sys

checkout = Path(sys.argv[1])
for relative_root in ("search_r1", "verl", "scripts"):
    for path in (checkout / relative_root).rglob("*.py"):
        compile(path.read_bytes(), str(path), "exec")
PY

    for gpu_count in 1 2; do
        for spec in 'train smoke 1' 'train baseline 60' 'train cost_aware 60' "eval baseline $MODEL_DIR" "eval cost_aware $MODEL_DIR"; do
            read -r mode variant argument <<<"$spec"
            AUTODL_CONFIG_ONLY=1 \
                AUTODL_ROOT="$PROJECT_ROOT" \
                GPU_COUNT="$gpu_count" \
                OUTPUT_DIR="$PROJECT_ROOT/cache/config-compose/${gpu_count}gpu-$variant-$mode" \
                bash "$CHECKOUT_DIR/scripts/autodl/train_small_grpo.sh" \
                "$mode" "$variant" "$argument" \
                >"$MANIFEST_DIR/config-${gpu_count}gpu-$variant-$mode.yaml"
        done
    done

    "$train_python" - \
        "$MANIFEST_DIR/config-1gpu-baseline-train.yaml" \
        "$MANIFEST_DIR/config-2gpu-baseline-train.yaml" <<'PY'
from pathlib import Path
import sys

from omegaconf import OmegaConf

for path, expected_gpu_count in zip(map(Path, sys.argv[1:]), (1, 2)):
    config = OmegaConf.load(path)
    group_size = config.actor_rollout_ref.rollout.n_agent
    mini_batch_size = config.actor_rollout_ref.actor.ppo_mini_batch_size
    expected_mini_batch_size = config.data.train_batch_size * group_size
    if group_size != 8 or mini_batch_size != expected_mini_batch_size:
        raise SystemExit("GRPO group and actor mini-batch configuration are inconsistent")
    if config.trainer.n_gpus_per_node != expected_gpu_count:
        raise SystemExit(f"GPU count mismatch in {path}")
PY

    "$train_python" -m pip freeze --all >"$MANIFEST_DIR/train-freeze.txt"
    "$retriever_python" -m pip freeze --all >"$MANIFEST_DIR/retriever-freeze.txt"
    python_version="$($train_python -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
    torch_version="$($train_python -c 'import torch; print(torch.__version__)')"

    "$train_python" "$CHECKOUT_DIR/scripts/autodl/handoff.py" create \
        --root "$PROJECT_ROOT" \
        --commit "$commit" \
        --model "$MODEL_DIR" \
        --bm25 "$BM25_ROOT/bm25" \
        --corpus "$CORPUS_ROOT" \
        --data "$SMALL_DATA_DIR" \
        --requirements "$CHECKOUT_DIR/requirements-autodl.lock" \
        --extra-file "$CORPUS_GZIP" \
        --extra-file "$MANIFEST_DIR/train-freeze.txt" \
        --extra-file "$MANIFEST_DIR/retriever-freeze.txt" \
        --extra-file "$MANIFEST_DIR/java-version.txt" \
        --extra-file "$MANIFEST_DIR/config-1gpu-smoke-train.yaml" \
        --extra-file "$MANIFEST_DIR/config-1gpu-baseline-train.yaml" \
        --extra-file "$MANIFEST_DIR/config-1gpu-cost_aware-train.yaml" \
        --extra-file "$MANIFEST_DIR/config-1gpu-baseline-eval.yaml" \
        --extra-file "$MANIFEST_DIR/config-1gpu-cost_aware-eval.yaml" \
        --extra-file "$MANIFEST_DIR/config-2gpu-smoke-train.yaml" \
        --extra-file "$MANIFEST_DIR/config-2gpu-baseline-train.yaml" \
        --extra-file "$MANIFEST_DIR/config-2gpu-cost_aware-train.yaml" \
        --extra-file "$MANIFEST_DIR/config-2gpu-baseline-eval.yaml" \
        --extra-file "$MANIFEST_DIR/config-2gpu-cost_aware-eval.yaml" \
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
