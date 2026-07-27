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
QWEN_NATIVE_DATA_DIR="$DATA_ROOT/search_mix_qwen35_native_v4"
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

run_pytest_file_shards() {
    local python_bin="$1" test_root="$2" test_file
    local -a test_files=()
    while IFS= read -r -d '' test_file; do
        test_files+=("$test_file")
    done < <(find "$test_root" -type f -name 'test_*.py' -print0 |
        LC_ALL=C sort -z)
    ((${#test_files[@]} > 0)) || {
        printf 'No pytest files found under %s.\n' "$test_root" >&2
        return 1
    }
    for test_file in "${test_files[@]}"; do
        printf 'Running pytest shard: %s\n' "${test_file#"$CHECKOUT_DIR/"}"
        "$python_bin" -m pytest -q -p no:cacheprovider "$test_file"
    done
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

cpu_seal_transaction() {
    local action="$1"
    local seal_dir="${2:-}"
    local python_bin="${PYTHON_BIN:-python3}"
    "$python_bin" - "$action" "$PROJECT_ROOT" "$MANIFEST_DIR" "$seal_dir" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile


action, root_raw, manifest_raw, seal_raw = sys.argv[1:]
root = Path(root_raw).resolve()
manifest_dir = Path(manifest_raw).resolve()
pending = manifest_dir / "cpu-seal.pending.json"
canonical = (
    manifest_dir / "cpu_handoff.json",
    manifest_dir / "cpu_handoff.json.sha256",
    manifest_dir / "java-version.txt",
    manifest_dir / "train-freeze.txt",
    manifest_dir / "retriever-freeze.txt",
    manifest_dir / "cpu.ok",
)


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("ascii")


def sync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.",
                                             dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        sync_directory(path.parent)
    finally:
        temporary_path.unlink(missing_ok=True)


def regular(path: Path) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"CPU seal file is missing or symlinked: {path}")


def verify_bundle(paths: tuple[Path, ...]) -> tuple[str, str]:
    handoff, sidecar, java_version, train_freeze, retriever_freeze, marker = paths
    for path in paths:
        regular(path)
    raw = handoff.read_bytes()
    handoff_digest = hashlib.sha256(raw).hexdigest()
    expected_sidecar = f"{handoff_digest}  cpu_handoff.json\n".encode("ascii")
    if sidecar.read_bytes() != expected_sidecar:
        raise ValueError("CPU handoff sidecar does not match its manifest")
    if marker.read_bytes() != f"{handoff_digest}\n".encode("ascii"):
        raise ValueError("cpu.ok does not match its handoff")
    if not java_version.read_bytes():
        raise ValueError("CPU Java version evidence is empty")
    if not train_freeze.read_bytes() or not retriever_freeze.read_bytes():
        raise ValueError("CPU environment freeze evidence is empty")
    bundle = hashlib.sha256()
    for path in paths:
        content_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        bundle.update(f"{path.name}\0{content_digest}\n".encode("ascii"))
    return handoff_digest, bundle.hexdigest()


def resolve_seal(relative: str) -> Path:
    relative_path = Path(relative)
    if (relative_path.is_absolute() or ".." in relative_path.parts
            or not relative_path.parts):
        raise ValueError("CPU seal transaction path is unsafe")
    seal_root = manifest_dir / "cpu-seals"
    if seal_root.is_symlink() or not seal_root.is_dir():
        raise ValueError("CPU seal transaction root is invalid")
    unresolved = root / relative_path
    try:
        relative_to_seals = unresolved.relative_to(seal_root)
    except ValueError as error:
        raise ValueError("CPU seal transaction path escapes cpu-seals") from error
    cursor = seal_root
    for part in relative_to_seals.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError("CPU seal transaction path contains a symlink")
    seal = unresolved.resolve()
    seal.relative_to(seal_root.resolve())
    if not seal.is_dir():
        raise ValueError("CPU seal transaction directory is invalid")
    return seal


def resolve_attempt(relative: str) -> Path:
    relative_path = Path(relative)
    if (relative_path.is_absolute() or ".." in relative_path.parts
            or not relative_path.parts):
        raise ValueError("CPU seal attempt path is unsafe")
    attempts_root = root / "state" / "attempts" / "cpu"
    if attempts_root.is_symlink() or not attempts_root.is_dir():
        raise ValueError("CPU attempt root is invalid")
    unresolved = root / relative_path
    try:
        relative_to_attempts = unresolved.relative_to(attempts_root)
    except ValueError as error:
        raise ValueError("CPU seal attempt escapes the attempt root") from error
    cursor = attempts_root
    for part in relative_to_attempts.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError("CPU seal attempt path contains a symlink")
    attempt = unresolved.resolve()
    attempt.relative_to(attempts_root.resolve())
    if not attempt.is_dir():
        raise ValueError("CPU seal attempt directory is invalid")
    return attempt


def read_pending() -> tuple[dict[str, object], Path, Path]:
    regular(pending)
    raw = pending.read_bytes()
    payload = json.loads(raw)
    if canonical_bytes(payload) != raw or set(payload) != {
            "attempt_dir",
            "new_bundle_digest", "new_handoff_digest",
            "old_bundle_digest", "old_handoff_digest", "schema",
            "seal_dir"}:
        raise ValueError("CPU seal pending marker is malformed")
    if payload["schema"] != 2:
        raise ValueError("CPU seal pending marker schema mismatch")
    for key in ("new_bundle_digest", "new_handoff_digest",
                "old_bundle_digest", "old_handoff_digest"):
        if not isinstance(payload[key], str) or re.fullmatch(
                r"[0-9a-f]{64}", payload[key]) is None:
            raise ValueError("CPU seal pending digest is malformed")
    if (not isinstance(payload["seal_dir"], str)
            or not isinstance(payload["attempt_dir"], str)):
        raise ValueError("CPU seal pending path is malformed")
    seal = resolve_seal(payload["seal_dir"])
    attempt = resolve_attempt(payload["attempt_dir"])
    if seal.name != attempt.name:
        raise ValueError("CPU seal and attempt identities differ")
    return payload, seal, attempt


def remove_pending() -> None:
    pending.unlink()
    sync_directory(manifest_dir)


def attempt_succeeded(attempt: Path) -> bool:
    terminal = attempt / "terminal"
    exit_code = attempt / "exit-code"
    success = attempt / ".success"
    for path in (terminal, exit_code, success):
        if not path.is_file() or path.is_symlink():
            return False
    if success.stat().st_size != 0:
        return False
    if terminal.read_bytes() != b"success\n" or exit_code.read_bytes() != b"0\n":
        return False
    for name in (".starting", ".running", ".failed"):
        path = attempt / name
        if path.exists() or path.is_symlink():
            return False
    return True


def recover() -> None:
    if not pending.exists() and not pending.is_symlink():
        return
    payload, seal, attempt = read_pending()
    candidate = (seal / "cpu_handoff.json",
                 seal / "cpu_handoff.json.sha256",
                 seal / "java-version.txt", seal / "train-freeze.txt",
                 seal / "retriever-freeze.txt", seal / "cpu.ok")
    previous = (seal / "previous" / "cpu_handoff.json",
                seal / "previous" / "cpu_handoff.json.sha256",
                seal / "previous" / "java-version.txt",
                seal / "previous" / "train-freeze.txt",
                seal / "previous" / "retriever-freeze.txt",
                seal / "previous" / "cpu.ok")
    try:
        current_identity = verify_bundle(canonical)
    except (OSError, ValueError):
        current_identity = None
    new_identity = (payload["new_handoff_digest"],
                    payload["new_bundle_digest"])
    old_identity = (payload["old_handoff_digest"],
                    payload["old_bundle_digest"])
    if current_identity == new_identity:
        if verify_bundle(candidate) != current_identity:
            raise ValueError("Published CPU seal differs from its candidate")
        if attempt_succeeded(attempt):
            remove_pending()
            return
    elif attempt_succeeded(attempt):
        raise ValueError("Successful CPU attempt is not bound to its candidate seal")
    if verify_bundle(previous) != old_identity:
        raise ValueError("Previous CPU seal backup is invalid")
    for source, target in zip(previous, canonical):
        atomic_write(target, source.read_bytes())
    if verify_bundle(canonical) != old_identity:
        raise ValueError("Previous CPU seal restoration failed")
    remove_pending()


def promote(seal: Path) -> None:
    if pending.exists() or pending.is_symlink():
        raise ValueError("Recover the pending CPU seal before promotion")
    if not seal.is_absolute():
        seal = Path.cwd() / seal
    try:
        relative = seal.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("CPU seal candidate escapes the persistent root") from error
    seal = resolve_seal(relative)
    attempt_relative = (Path("state") / "attempts" / "cpu" /
                        seal.name).as_posix()
    resolve_attempt(attempt_relative)
    candidate = (seal / "cpu_handoff.json",
                 seal / "cpu_handoff.json.sha256",
                 seal / "java-version.txt", seal / "train-freeze.txt",
                 seal / "retriever-freeze.txt", seal / "cpu.ok")
    previous = (seal / "previous" / "cpu_handoff.json",
                seal / "previous" / "cpu_handoff.json.sha256",
                seal / "previous" / "java-version.txt",
                seal / "previous" / "train-freeze.txt",
                seal / "previous" / "retriever-freeze.txt",
                seal / "previous" / "cpu.ok")
    old_identity = verify_bundle(canonical)
    new_identity = verify_bundle(candidate)
    if old_identity == new_identity:
        raise ValueError("Candidate CPU seal must differ from the previous seal")
    previous_dir = previous[0].parent
    if previous_dir.is_symlink():
        raise ValueError("Previous CPU seal backup directory is symlinked")
    if previous_dir.exists():
        expected_names = {path.name for path in previous}
        if ({path.name for path in previous_dir.iterdir()} != expected_names
                or verify_bundle(previous) != old_identity):
            raise ValueError("Existing previous CPU seal backup is invalid")
    else:
        previous_dir.mkdir(parents=True, exist_ok=False)
        for source, target in zip(canonical, previous):
            atomic_write(target, source.read_bytes())
        sync_directory(previous_dir)
        sync_directory(seal)
    if verify_bundle(previous) != old_identity:
        raise ValueError("Previous CPU seal backup verification failed")
    payload = {
        "attempt_dir": attempt_relative,
        "new_bundle_digest": new_identity[1],
        "new_handoff_digest": new_identity[0],
        "old_bundle_digest": old_identity[1],
        "old_handoff_digest": old_identity[0],
        "schema": 2,
        "seal_dir": relative,
    }
    atomic_write(pending, canonical_bytes(payload))
    for index, (source, target) in enumerate(zip(candidate, canonical)):
        atomic_write(target, source.read_bytes())
        if (os.environ.get("AUTODL_TEST_MODE") == "1"
                and os.environ.get("AUTODL_TEST_CPU_SEAL_CRASH_AFTER")
                == str(index)):
            os._exit(91)
    if verify_bundle(canonical) != new_identity:
        raise ValueError("Published CPU seal verification failed")


if action == "recover":
    recover()
elif action == "promote":
    if not seal_raw:
        raise SystemExit("promote requires a CPU seal directory")
    promote(Path(seal_raw))
else:
    raise SystemExit(f"unknown CPU seal transaction action: {action}")
PY
}

phase_terminal_hook() {
    local phase="$1" python_bin
    [[ "$phase" == cpu ]] || return 0
    [[ -e "$MANIFEST_DIR/cpu-seal.pending.json" ||
       -L "$MANIFEST_DIR/cpu-seal.pending.json" ]] || return 0
    if [[ -x "$TRAIN_ENV/bin/python" ]]; then
        python_bin="$TRAIN_ENV/bin/python"
    else
        python_bin="$(command -v python3)"
    fi
    PYTHON_BIN="$python_bin" cpu_seal_transaction recover
}

locked_cpu_seal_transaction() {
    local action="$1" seal_dir="${2:-}"
    [[ -d "$(dirname -- "$LOCK_FILE")" && ! -L "$LOCK_FILE" ]] || {
        printf 'CPU seal transaction lock path is invalid.\n' >&2
        return 1
    }
    exec 8>"$LOCK_FILE"
    flock -n 8 || {
        printf 'Another AutoDL phase owns %s\n' "$LOCK_FILE" >&2
        return 75
    }
    cpu_seal_transaction "$action" "$seal_dir"
}

cpu_action() {
    local _attempt="$1"
    local commit base_python train_python retriever_python python_version torch_version handoff_digest gpu_count
    local previous_commit previous_python previous_torch build_search_mix=0 seal_search_mix=0
    local build_qwen_native=0 seal_qwen_native=0
    local spec mode variant steps model_path config_output_dir parent_placeholder trace_placeholder
    local config_manifest_dir candidate_handoff seal_dir
    local train_freeze_file retriever_freeze_file java_version_file
    local config_response_length eval_group_size
    local trace_digest_placeholder trace_checkpoint_digest trace_parent_digest trace_output trace_stage
    local -a command_args previous_handoff_args search_mix_handoff_args qwen_native_handoff_args
    local -a native_handoff_contract_args native_handoff_verify_args
    commit="$(expected_commit)"
    verify_checkout "$commit"
    config_manifest_dir="$MANIFEST_DIR"
    candidate_handoff="$HANDOFF"
    seal_dir=''
    train_freeze_file="$MANIFEST_DIR/train-freeze.txt"
    retriever_freeze_file="$MANIFEST_DIR/retriever-freeze.txt"
    java_version_file="$MANIFEST_DIR/java-version.txt"
    [[ -f "$CHECKOUT_DIR/requirements-autodl.lock" ]] || {
        printf 'Missing requirements-autodl.lock in the pinned checkout.\n' >&2
        return 1
    }
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
    [[ "${AUTODL_QWEN_NATIVE_INCREMENTAL:-0}" == 0 ||
        "${AUTODL_QWEN_NATIVE_INCREMENTAL:-0}" == 1 ]] || {
        printf 'AUTODL_QWEN_NATIVE_INCREMENTAL must be 0 or 1.\n' >&2
        return 64
    }
    if (( ${AUTODL_RESEAL_ONLY:-0} + ${AUTODL_SEARCH_GATE_INCREMENTAL:-0} +
          ${AUTODL_SEARCH_MIX_INCREMENTAL:-0} +
          ${AUTODL_QWEN_NATIVE_INCREMENTAL:-0} > 1 )); then
        printf 'Choose only one incremental or offline reseal mode.\n' >&2
        return 64
    fi

    if [[ "${AUTODL_RESEAL_ONLY:-0}" == 1 ||
          "${AUTODL_SEARCH_GATE_INCREMENTAL:-0}" == 1 ||
          "${AUTODL_SEARCH_MIX_INCREMENTAL:-0}" == 1 ||
          "${AUTODL_QWEN_NATIVE_INCREMENTAL:-0}" == 1 ]]; then
        export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PIP_NO_INDEX=1
        train_python="$TRAIN_ENV/bin/python"
        retriever_python="$RETRIEVER_ENV/bin/python"
        [[ -x "$train_python" && -x "$retriever_python" ]] || {
            printf 'Incremental reseal requires the existing environments.\n' >&2
            return 1
        }
        PYTHON_BIN="$train_python" cpu_seal_transaction recover
        [[ -x "$train_python" && -x "$retriever_python" &&
            -f "$HANDOFF" && ! -L "$HANDOFF" &&
            -f "$HANDOFF.sha256" && ! -L "$HANDOFF.sha256" &&
            -f "$MANIFEST_DIR/cpu.ok" && ! -L "$MANIFEST_DIR/cpu.ok" ]] || {
            printf 'Incremental reseal requires the existing CPU handoff.\n' >&2
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
        previous_handoff_args=()
        if [[ "${AUTODL_QWEN_NATIVE_INCREMENTAL:-0}" == 1 ]]; then
            previous_handoff_args+=(
                --require-artifact data/search_mix/retrieval_replay.json
                --require-artifact data/search_mix/retrieval_replay.json.sha256
            )
        fi
        "$train_python" "$CHECKOUT_DIR/scripts/autodl/handoff.py" verify \
            --root "$PROJECT_ROOT" \
            --commit "$previous_commit" \
            --python-version "$previous_python" \
            --torch-version "$previous_torch" \
            --manifest "$HANDOFF" \
            "${previous_handoff_args[@]}"
        handoff_digest="$(sha256sum -- "$HANDOFF" | cut -d' ' -f1)"
        [[ "$(tr -d '\r\n' <"$MANIFEST_DIR/cpu.ok")" == "$handoff_digest" ]] || {
            printf 'cpu.ok does not match the previous sealed handoff.\n' >&2
            return 1
        }
        # Keep the previous marker until the replacement handoff is verified.
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
        if [[ "${AUTODL_QWEN_NATIVE_INCREMENTAL:-0}" == 1 ]]; then
            java_version_file="$_attempt/java-version.current.txt"
        fi
        java -version >"$java_version_file" 2>&1
        grep -Eq 'version "21([.]|")' "$java_version_file" || {
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
        if [[ "${AUTODL_QWEN_NATIVE_INCREMENTAL:-0}" == 1 ]]; then
            [[ -f "$SEARCH_MIX_DATA_DIR/manifest.json" &&
                ! -L "$SEARCH_MIX_DATA_DIR/manifest.json" ]] || {
                printf 'Qwen native materialization requires the sealed search_mix manifest.\n' >&2
                return 1
            }
            if [[ -f "$QWEN_NATIVE_DATA_DIR/manifest.json" &&
                  ! -L "$QWEN_NATIVE_DATA_DIR/manifest.json" ]]; then
                seal_qwen_native=1
            elif [[ -e "$QWEN_NATIVE_DATA_DIR" || -L "$QWEN_NATIVE_DATA_DIR" ]]; then
                printf 'Qwen native data target exists without a regular manifest: %s\n' \
                    "$QWEN_NATIVE_DATA_DIR" >&2
                return 1
            else
                build_qwen_native=1
            fi
        fi
    elif [[ "${AUTODL_RESEAL_ONLY:-0}" == 0 ]]; then
        rm -f -- "$MANIFEST_DIR/cpu.ok"
        sync_path "$MANIFEST_DIR"
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
        if [[ -f "$QWEN_NATIVE_DATA_DIR/manifest.json" &&
              ! -L "$QWEN_NATIVE_DATA_DIR/manifest.json" ]]; then
            # A failed post-materialization retry may reuse only source-bound data.
            seal_qwen_native=1
        elif [[ -e "$QWEN_NATIVE_DATA_DIR" || -L "$QWEN_NATIVE_DATA_DIR" ]]; then
            printf 'Qwen native data target exists without a regular manifest: %s\n' \
                "$QWEN_NATIVE_DATA_DIR" >&2
            return 1
        else
            build_qwen_native=1
        fi
    fi

    if [[ "${AUTODL_QWEN_NATIVE_INCREMENTAL:-0}" == 1 ]]; then
        seal_dir="$MANIFEST_DIR/cpu-seals/$(basename -- "$_attempt")"
        [[ ! -e "$seal_dir" && ! -L "$seal_dir" ]] || {
            printf 'CPU seal candidate already exists: %s\n' "$seal_dir" >&2
            return 1
        }
        mkdir -p "$seal_dir/configs"
        config_manifest_dir="$seal_dir/configs"
        candidate_handoff="$seal_dir/cpu_handoff.json"
        train_freeze_file="$seal_dir/train-freeze.txt"
        retriever_freeze_file="$seal_dir/retriever-freeze.txt"
        java_version_file="$seal_dir/java-version.txt"
        cp -- "$_attempt/train-freeze.current.txt" "$train_freeze_file"
        cp -- "$_attempt/retriever-freeze.current.txt" "$retriever_freeze_file"
        cp -- "$_attempt/java-version.current.txt" "$java_version_file"
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
        # Native verification recursively verifies this source without
        # recomputing selection. Other modes retain deterministic reselection.
        if [[ "${AUTODL_QWEN_NATIVE_INCREMENTAL:-0}" != 1 ]]; then
            "$train_python" "$CHECKOUT_DIR/scripts/data_process/search_mix.py" verify \
                --manifest "$SEARCH_MIX_DATA_DIR/manifest.json" \
                --model-dir "$MODEL_DIR" \
                --eval-catalog "$SEARCH_GATE_DATA_DIR/catalog.jsonl" \
                --eval-parquet "$SMALL_DATA_DIR/test_128.parquet"
        fi
        if [[ "${AUTODL_QWEN_NATIVE_INCREMENTAL:-0}" != 1 ]]; then
            PYTHONPATH="$CHECKOUT_DIR" "$retriever_python" \
                "$CHECKOUT_DIR/scripts/data_process/search_mix.py" replay \
                --manifest "$SEARCH_MIX_DATA_DIR/manifest.json" \
                --index-path "$BM25_ROOT/bm25" \
                --corpus-path "$CORPUS_JSONL" \
                --offsets-path "$CORPUS_OFFSETS"
        fi
        if [[ "${AUTODL_QWEN_NATIVE_INCREMENTAL:-0}" != 1 ]]; then
            "$train_python" "$CHECKOUT_DIR/scripts/data_process/search_mix.py" verify-replay \
                --manifest "$SEARCH_MIX_DATA_DIR/manifest.json"
        fi
    fi
    if [[ "$build_qwen_native" == 1 ]]; then
        "$train_python" "$CHECKOUT_DIR/scripts/data_process/search_mix.py" materialize-native \
            --source-manifest "$SEARCH_MIX_DATA_DIR/manifest.json" \
            --output-dir "$QWEN_NATIVE_DATA_DIR" \
            --model-dir "$MODEL_DIR" \
            --eval-catalog "$SEARCH_GATE_DATA_DIR/catalog.jsonl" \
            --eval-parquet "$SMALL_DATA_DIR/test_128.parquet"
        seal_qwen_native=1
    elif [[ -f "$QWEN_NATIVE_DATA_DIR/manifest.json" &&
            ! -L "$QWEN_NATIVE_DATA_DIR/manifest.json" ]]; then
        seal_qwen_native=1
    elif [[ -e "$QWEN_NATIVE_DATA_DIR/manifest.json" ||
            -L "$QWEN_NATIVE_DATA_DIR/manifest.json" ]]; then
        printf 'Qwen native manifest exists but is not a regular non-symlink file.\n' >&2
        return 1
    fi
    if [[ "$seal_qwen_native" == 1 ]]; then
        "$train_python" - "$MODEL_DIR/chat_template.jinja" <<'PY'
import hashlib
from pathlib import Path
import sys

from transformers import AutoTokenizer

from search_r1.llm_agent.tool_protocol import (
    QWEN35_CHAT_TEMPLATE_SHA256,
    QWEN35_NATIVE,
    QWEN35_REASONING_CONTINUATION,
    QWEN35_RETRY_PROMPT,
    QWEN35_TERMINAL_PROMPT,
    QWEN35_TERMINAL_PROMPT_SHA256,
    QWEN35_TERMINAL_PROMPT_VERSION,
    Qwen35Conversation,
    parse_action,
    qwen35_messages,
    qwen35_tools,
    render_qwen35_prompt,
)

path = Path(sys.argv[1])
if not path.is_file() or path.is_symlink():
    raise SystemExit(f"Qwen chat template is missing or symlinked: {path}")
actual = hashlib.sha256(path.read_bytes()).hexdigest()
if actual != QWEN35_CHAT_TEMPLATE_SHA256:
    raise SystemExit(
        f"Qwen chat template digest mismatch: expected "
        f"{QWEN35_CHAT_TEMPLATE_SHA256}, found {actual}")

tokenizer = AutoTokenizer.from_pretrained(path.parent, local_files_only=True)
messages = qwen35_messages("Who wrote Hamlet?")
prompt_ids = tokenizer(render_qwen35_prompt(tokenizer, messages),
                       add_special_tokens=False)["input_ids"]
direct_prompt_ids = tokenizer.apply_chat_template(
    messages, tools=qwen35_tools(), enable_thinking=True,
    add_generation_prompt=True, tokenize=True, return_dict=False)
if direct_prompt_ids != prompt_ids:
    raise SystemExit("Qwen native rendered-string and direct template tokens differ")
conversation = Qwen35Conversation(tokenizer, messages, prompt_ids)

def native_search(reasoning, query):
    return (
        f"{reasoning}\n</think>\n\n"
        "<tool_call>\n<function=search>\n<parameter=query>\n"
        f"{query}\n</parameter>\n</function>\n</tool_call>"
    )

def assert_roundtrip(value, expected_prefix, label):
    if value.prompt_token_ids[:len(expected_prefix)] != expected_prefix:
        raise SystemExit(f"{label} did not preserve cumulative sampled tokens")
    rerendered = tokenizer(
        render_qwen35_prompt(tokenizer, value.messages),
        add_special_tokens=False)["input_ids"]
    if rerendered != value.prompt_token_ids:
        raise SystemExit(f"{label} cumulative token roundtrip differs")

search_text = native_search("I should verify the author.", "Hamlet author")
search_ids = tokenizer(search_text, add_special_tokens=False)["input_ids"]
search_prefix = list(conversation.prompt_token_ids) + list(search_ids)
search = conversation.append_followup(
    search_text, parse_action(
        search_text, QWEN35_NATIVE,
        qwen35_reasoning_mode=QWEN35_REASONING_CONTINUATION),
    "  Hamlet was written by William Shakespeare.  ", 500,
    response_token_ids=search_ids)
if not search.token_ids or not search.visible_observation:
    raise SystemExit("Qwen native search wrapper did not preserve sampled tokens")
assert_roundtrip(conversation, search_prefix, "first search/tool response")

second_text = native_search("I need a second source.", "Shakespeare Hamlet")
second_ids = tokenizer(second_text, add_special_tokens=False)["input_ids"]
second_prefix = list(conversation.prompt_token_ids) + list(second_ids)
second = conversation.append_followup(
    second_text, parse_action(
        second_text, QWEN35_NATIVE,
        qwen35_reasoning_mode=QWEN35_REASONING_CONTINUATION),
    "  Shakespeare is credited as the author of Hamlet.  ", 500,
    response_token_ids=second_ids)
if not second.token_ids or not second.visible_observation:
    raise SystemExit("Qwen native second search wrapper is empty")
assert_roundtrip(conversation, second_prefix, "second search/tool response")

retry_conversation = Qwen35Conversation(tokenizer, messages, prompt_ids)
retry_search_ids = tokenizer(search_text,
                             add_special_tokens=False)["input_ids"]
retry_search_prefix = (list(retry_conversation.prompt_token_ids)
                       + list(retry_search_ids))
retry_conversation.append_followup(
    search_text, parse_action(
        search_text, QWEN35_NATIVE,
        qwen35_reasoning_mode=QWEN35_REASONING_CONTINUATION),
    "  Hamlet was written by William Shakespeare.  ", 500,
    response_token_ids=retry_search_ids)
assert_roundtrip(retry_conversation, retry_search_prefix,
                 "search before invalid retry")

invalid_text = (
    "I used the wrong action syntax.\n</think>\n\n"
    '{"name":"search","arguments":{"query":"Hamlet author"}}')
invalid_ids = tokenizer(invalid_text, add_special_tokens=False)["input_ids"]
invalid_prefix = (list(retry_conversation.prompt_token_ids)
                  + list(invalid_ids))
retry = retry_conversation.append_followup(
    invalid_text, parse_action(
        invalid_text, QWEN35_NATIVE,
        qwen35_reasoning_mode=QWEN35_REASONING_CONTINUATION), "", 500,
    response_token_ids=invalid_ids)
if (not retry.token_ids
        or retry_conversation.messages[-1] != {
            "role": "user", "content": QWEN35_RETRY_PROMPT}):
    raise SystemExit("Qwen native retry wrapper did not preserve sampled tokens")
assert_roundtrip(retry_conversation, invalid_prefix,
                 "search/invalid user retry")

if (QWEN35_TERMINAL_PROMPT_VERSION != "qwen35-terminal-answer-v1"
        or hashlib.sha256(QWEN35_TERMINAL_PROMPT.encode("utf-8")).hexdigest()
        != QWEN35_TERMINAL_PROMPT_SHA256):
    raise SystemExit("Qwen native terminal prompt identity is inconsistent")

terminal_search_conversation = Qwen35Conversation(tokenizer, messages, prompt_ids)
terminal_search_prefix = (list(terminal_search_conversation.prompt_token_ids)
                          + list(search_ids))
terminal_search = terminal_search_conversation.append_followup(
    search_text, parse_action(
        search_text, QWEN35_NATIVE,
        qwen35_reasoning_mode=QWEN35_REASONING_CONTINUATION),
    "  Hamlet was written by William Shakespeare.  ", 500,
    response_token_ids=search_ids, terminal_answer_only=True)
if (not terminal_search.terminal_instruction_applied
        or [message["role"] for message in terminal_search_conversation.messages[-3:]]
        != ["assistant", "tool", "user"]
        or terminal_search_conversation.messages[-1] != {
            "role": "user", "content": QWEN35_TERMINAL_PROMPT}
        or any(message.get("content") == QWEN35_RETRY_PROMPT
               for message in terminal_search_conversation.messages)):
    raise SystemExit("Qwen native terminal search wrapper is invalid")
assert_roundtrip(terminal_search_conversation, terminal_search_prefix,
                 "search/tool/terminal user")
rejected_search = parse_action(
    search_text, QWEN35_NATIVE,
    qwen35_reasoning_mode=QWEN35_REASONING_CONTINUATION,
    qwen35_answer_only=True)
if (rejected_search.valid or rejected_search.action != "search"
        or rejected_search.error != "search_disallowed_after_budget"):
    raise SystemExit("Qwen native terminal search was not rejected")

terminal_invalid_conversation = Qwen35Conversation(tokenizer, messages, prompt_ids)
terminal_invalid_prefix = (list(terminal_invalid_conversation.prompt_token_ids)
                           + list(invalid_ids))
terminal_invalid = terminal_invalid_conversation.append_followup(
    invalid_text, parse_action(
        invalid_text, QWEN35_NATIVE,
        qwen35_reasoning_mode=QWEN35_REASONING_CONTINUATION), "", 500,
    response_token_ids=invalid_ids, terminal_answer_only=True)
if (not terminal_invalid.terminal_instruction_applied
        or [message["role"] for message in terminal_invalid_conversation.messages[-2:]]
        != ["assistant", "user"]
        or terminal_invalid_conversation.messages[-1] != {
            "role": "user", "content": QWEN35_TERMINAL_PROMPT}
        or any(message.get("content") == QWEN35_RETRY_PROMPT
               for message in terminal_invalid_conversation.messages)):
    raise SystemExit("Qwen native terminal invalid wrapper is invalid")
assert_roundtrip(terminal_invalid_conversation, terminal_invalid_prefix,
                 "invalid/terminal user")
PY
        "$train_python" "$CHECKOUT_DIR/scripts/data_process/search_mix.py" \
            validate-native-evidence \
            --manifest "$QWEN_NATIVE_DATA_DIR/manifest.json" \
            --source-manifest "$SEARCH_MIX_DATA_DIR/manifest.json" \
            --model-dir "$MODEL_DIR" \
            --eval-catalog "$SEARCH_GATE_DATA_DIR/catalog.jsonl" \
            --eval-parquet "$SMALL_DATA_DIR/test_128.parquet"
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

    # Native-only resealing changes prompts, so the prior handoff already covers BM25.
    if [[ "${AUTODL_QWEN_NATIVE_INCREMENTAL:-0}" != 1 ]]; then
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
    fi

    run_pytest_file_shards "$train_python" "$CHECKOUT_DIR/tests"
    PYTHON_BIN="$train_python" bash "$CHECKOUT_DIR/scripts/autodl/tests/test_runtime.sh"
    bash "$CHECKOUT_DIR/scripts/autodl/tests/test_gated_config.sh"
    bash "$CHECKOUT_DIR/scripts/autodl/tests/test_gated_followup.sh"
    bash "$CHECKOUT_DIR/scripts/autodl/tests/test_search_opportunity_pipeline.sh"
    PYTHON_BIN="$train_python" bash "$CHECKOUT_DIR/scripts/autodl/tests/test_group_probe_pipeline.sh"
    PYTHON_BIN="$train_python" bash "$CHECKOUT_DIR/scripts/autodl/tests/test_qwen_native_gate_pipeline.sh"
    bash -n "$CHECKOUT_DIR/scripts/autodl/09_gpu_qwen_native_train.sh"
    PYTHON_BIN="$train_python" bash \
        "$CHECKOUT_DIR/scripts/autodl/tests/test_qwen_native_training_pipeline.sh"
    PYTHON_BIN="$train_python" bash \
        "$CHECKOUT_DIR/scripts/autodl/tests/test_cpu_native_reseal_transaction.sh"
    PYTHON_BIN="$train_python" bash \
        "$CHECKOUT_DIR/scripts/autodl/tests/test_shutdown_watchdog.sh"
    "$train_python" "$CHECKOUT_DIR/scripts/autodl/tests/test_results.py"
    "$train_python" "$CHECKOUT_DIR/scripts/autodl/tests/test_paired_eval.py"
    "$train_python" "$CHECKOUT_DIR/scripts/autodl/tests/test_search_opportunity_gate.py"
    "$train_python" "$CHECKOUT_DIR/scripts/autodl/tests/test_export_gated_training.py"
    PYTHONPYCACHEPREFIX="$CACHE_ROOT/pycache" "$train_python" -m py_compile \
        "$CHECKOUT_DIR/scripts/autodl/qwen_native_smoke_analysis.py"
    "$train_python" \
        "$CHECKOUT_DIR/scripts/autodl/tests/test_qwen_native_smoke_analysis.py"
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
                >"$config_manifest_dir/config-${gpu_count}gpu-$variant-$mode.yaml"
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
            >"$config_manifest_dir/config-${gpu_count}gpu-cost_aware_gated-gate-train.yaml"
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
                >"$config_manifest_dir/config-${gpu_count}gpu-$variant-trace-eval.yaml"
        done
    done

    if [[ "$seal_qwen_native" == 1 ]]; then
        for spec in \
            "qwen_native_g1|$QWEN_NATIVE_DATA_DIR/probe_autonomous_16.parquet|2" \
            "qwen_native_g3|$QWEN_NATIVE_DATA_DIR/probe_multi_64.parquet|5"; do
            IFS='|' read -r variant eval_data_file eval_group_size <<<"$spec"
            AUTODL_CONFIG_ONLY=1 \
                AUTODL_ROOT="$PROJECT_ROOT" \
                GPU_COUNT=2 \
                MAX_RESPONSE_LENGTH=500 \
                DATA_DIR="$QWEN_NATIVE_DATA_DIR" \
                OUTPUT_DIR="$config_output_dir" \
                EVAL_DATA_FILE="$eval_data_file" \
                EVAL_GROUP_SIZE="$eval_group_size" \
                TRACE_OUTPUT_DIR="$trace_placeholder" \
                TRACE_STAGE="$variant" \
                TRACE_RUN_ID=config-compose \
                TRACE_CHECKPOINT_DIGEST="$trace_digest_placeholder" \
                TOOL_PROTOCOL=qwen35_native \
                bash "$CHECKOUT_DIR/scripts/autodl/train_small_grpo.sh" \
                eval "$variant" "$MODEL_DIR" \
                >"$config_manifest_dir/config-2gpu-$variant-eval.yaml"
        done
        for variant in \
            qwen_native_a_val qwen_native_r_val \
            qwen_native_b_val qwen_native_c_val \
            qwen_native_a_nq_test qwen_native_r_nq_test \
            qwen_native_b_nq_test qwen_native_c_nq_test \
            qwen_native_a_multihop qwen_native_r_multihop \
            qwen_native_b_multihop qwen_native_c_multihop; do
            endpoint_model="$parent_placeholder"
            if [[ "$variant" == qwen_native_a_* ]]; then
                endpoint_model="$MODEL_DIR"
            fi
            AUTODL_CONFIG_ONLY=1 \
                AUTODL_ROOT="$PROJECT_ROOT" \
                GPU_COUNT=2 \
                TRAIN_BATCH_SIZE=8 \
                MAX_RESPONSE_LENGTH=500 \
                DATA_DIR="$QWEN_NATIVE_DATA_DIR" \
                OUTPUT_DIR="$config_output_dir" \
                EVAL_DATA_FILE='' \
                EVAL_GROUP_SIZE=1 \
                TRACE_OUTPUT_DIR="$trace_placeholder" \
                TRACE_STAGE="$variant" \
                TRACE_RUN_ID=config-compose \
                TRACE_CHECKPOINT_DIGEST="$trace_digest_placeholder" \
                TOOL_PROTOCOL=qwen35_native \
                bash "$CHECKOUT_DIR/scripts/autodl/train_small_grpo.sh" \
                eval "$variant" "$endpoint_model" \
                >"$config_manifest_dir/config-2gpu-$variant-eval.yaml"
        done
        for spec in \
            "smoke|2|" \
            "reproduce|60|" \
            "control|20|$parent_placeholder" \
            "cost_aware_gated|20|$parent_placeholder"; do
            IFS='|' read -r variant steps model_path <<<"$spec"
            command_args=(train "$variant" "$steps")
            if [[ -n "$model_path" ]]; then
                command_args+=("$model_path")
            fi
            AUTODL_CONFIG_ONLY=1 \
                AUTODL_ROOT="$PROJECT_ROOT" \
                GPU_COUNT=2 \
                TRAIN_BATCH_SIZE=8 \
                MAX_RESPONSE_LENGTH=500 \
                DATA_DIR="$QWEN_NATIVE_DATA_DIR" \
                OUTPUT_DIR="$config_output_dir" \
                TRACE_OUTPUT_DIR="$trace_placeholder" \
                TRACE_STAGE="$variant" \
                TRACE_RUN_ID=config-compose \
                TRACE_PARENT_CHECKPOINT_DIGEST="$trace_digest_placeholder" \
                TOOL_PROTOCOL=qwen35_native \
                bash "$CHECKOUT_DIR/scripts/autodl/train_small_grpo.sh" \
                "${command_args[@]}" \
                >"$config_manifest_dir/config-2gpu-qwen-native-$variant-train.yaml"
        done
        "$train_python" - \
            "$config_manifest_dir" \
            "$QWEN_NATIVE_DATA_DIR" \
            "$MODEL_DIR" \
            "$parent_placeholder" \
            "$trace_placeholder" <<'PY'
from copy import deepcopy
from pathlib import Path
import sys

from omegaconf import OmegaConf

manifest_dir, data_dir, model_dir, parent_placeholder, trace_placeholder = map(
    Path, sys.argv[1:]
)
specs = {
    "qwen_native_g1": ("probe_autonomous_16.parquet", 2, model_dir),
    "qwen_native_g3": ("probe_multi_64.parquet", 5, model_dir),
    "qwen_native_a_val": ("val_128.parquet", 1, model_dir),
    "qwen_native_r_val": ("val_128.parquet", 1, parent_placeholder),
    "qwen_native_b_val": ("val_128.parquet", 1, parent_placeholder),
    "qwen_native_c_val": ("val_128.parquet", 1, parent_placeholder),
    "qwen_native_a_nq_test": ("nq_test_128_native_v4.parquet", 1, model_dir),
    "qwen_native_r_nq_test": ("nq_test_128_native_v4.parquet", 1, parent_placeholder),
    "qwen_native_b_nq_test": ("nq_test_128_native_v4.parquet", 1, parent_placeholder),
    "qwen_native_c_nq_test": ("nq_test_128_native_v4.parquet", 1, parent_placeholder),
    "qwen_native_a_multihop": ("multihop_eval_256_native_v4.parquet", 1, model_dir),
    "qwen_native_r_multihop": ("multihop_eval_256_native_v4.parquet", 1, parent_placeholder),
    "qwen_native_b_multihop": ("multihop_eval_256_native_v4.parquet", 1, parent_placeholder),
    "qwen_native_c_multihop": ("multihop_eval_256_native_v4.parquet", 1, parent_placeholder),
}
for variant, (filename, group_size, checkpoint) in specs.items():
    path = manifest_dir / f"config-2gpu-{variant}-eval.yaml"
    config = OmegaConf.load(path)
    rollout = config.actor_rollout_ref.rollout
    trace_output = config.trainer.get("trace_output_dir", None)
    trace_stage = config.trainer.get("trace_stage", None)
    trace_run_id = config.trainer.get("trace_run_id", None)
    trace_checkpoint = config.trainer.get("trace_checkpoint_digest", None)
    if (config.tool_protocol != "qwen35_native"
            or config.data.return_raw_chat is not True
            or Path(config.data.val_files) != data_dir / filename
            or config.data.eval_group_size != group_size
            or config.data.val_batch_size != 8
            or config.data.max_response_length != 500
            or config.data.max_obs_length != 500
            or config.data.max_prompt_length != 4500
            or Path(config.actor_rollout_ref.model.path) != checkpoint
            or float(rollout.temperature) != 1.0
            or float(rollout.top_p) != 1.0
            or rollout.top_k != 0
            or float(rollout.min_p) != 0.0
            or float(rollout.presence_penalty) != 0.0
            or float(rollout.repetition_penalty) != 1.0
            or rollout.do_sample != (group_size > 1)
            or config.trainer.n_gpus_per_node != 2
            or config.trainer.val_only is not True
            or Path(trace_output) != trace_placeholder
            or trace_stage != variant
            or trace_run_id != "config-compose"
            or trace_checkpoint != "a" * 64):
        raise SystemExit(f"Qwen native resolved config mismatch: {path}")

train_specs = {
    "smoke": (2, model_dir, 0.0, "linear"),
    "reproduce": (60, model_dir, 0.0, "linear"),
    "control": (20, parent_placeholder, 0.0, "linear"),
    "cost_aware_gated": (20, parent_placeholder, 0.10, "correct_only"),
}
train_configs = {}
for variant, (steps, model_path, cost_lambda, reward_mode) in train_specs.items():
    path = manifest_dir / f"config-2gpu-qwen-native-{variant}-train.yaml"
    config = OmegaConf.load(path)
    train_configs[variant] = config
    rollout = config.actor_rollout_ref.rollout
    trace_output = config.trainer.get("trace_output_dir", None)
    trace_stage = config.trainer.get("trace_stage", None)
    trace_run_id = config.trainer.get("trace_run_id", None)
    trace_checkpoint = config.trainer.get("trace_checkpoint_digest", None)
    trace_parent = config.trainer.get("trace_parent_checkpoint_digest", None)
    if (config.tool_protocol != "qwen35_native"
            or config.qwen35_prompt_version != "qwen35-native-search-v4-terminal-answer-only"
            or config.trainer.native_training_variant != variant
            or Path(config.data.train_files) != data_dir / "train_512.parquet"
            or Path(config.data.val_files) != data_dir / "val_128.parquet"
            or config.data.train_batch_size != 8
            or config.data.val_batch_size != 8
            or config.data.eval_group_size != 1
            or config.data.return_raw_chat is not True
            or config.data.max_response_length != 500
            or config.data.max_prompt_length != 4500
            or config.data.max_start_length != 1024
            or config.data.max_obs_length != 500
            or rollout.n_agent != 5
            or float(rollout.temperature) != 1.0
            or float(rollout.top_p) != 1.0
            or rollout.top_k != 0
            or float(rollout.min_p) != 0.0
            or float(rollout.presence_penalty) != 0.0
            or float(rollout.repetition_penalty) != 1.0
            or rollout.do_sample is not True
            or config.actor_rollout_ref.actor.ppo_mini_batch_size != 40
            or config.actor_rollout_ref.actor.ppo_micro_batch_size != 2
            or config.actor_rollout_ref.rollout.log_prob_micro_batch_size != 2
            or config.actor_rollout_ref.ref.log_prob_micro_batch_size != 2
            or float(config.actor_rollout_ref.actor.optim.lr) != 1e-6
            or float(config.actor_rollout_ref.actor.optim.lr_warmup_steps_ratio) != 0.285
            or config.actor_rollout_ref.actor.use_kl_loss is not True
            or float(config.actor_rollout_ref.actor.kl_loss_coef) != 0.001
            or config.actor_rollout_ref.actor.state_masking is not True
            or config.trainer.n_gpus_per_node != 2
            or config.max_turns != 4
            or (config.max_turns * (config.data.max_response_length
                                    + config.data.max_obs_length)
                + config.data.max_response_length) > 4500
            or config.retriever.topk != 3
            or config.trainer.total_training_steps != steps
            or config.trainer.save_freq != steps
            or config.trainer.test_freq != steps
            or Path(config.actor_rollout_ref.model.path) != model_path
            or float(config.algorithm.cost_lambda) != cost_lambda
            or config.algorithm.cost_reward_mode != reward_mode):
        raise SystemExit(f"Qwen native training config mismatch: {path}")
    if (Path(trace_output) != trace_placeholder
            or trace_stage != variant
            or trace_run_id != "config-compose"
            or trace_checkpoint
            or trace_parent != "a" * 64):
        raise SystemExit(f"Qwen native trace contract mismatch: {path}")

control = deepcopy(OmegaConf.to_container(train_configs["control"], resolve=True))
cost = deepcopy(OmegaConf.to_container(train_configs["cost_aware_gated"], resolve=True))
for normalized in (control, cost):
    normalized["algorithm"]["cost_lambda"] = None
    normalized["algorithm"]["cost_reward_mode"] = None
    normalized["trainer"]["experiment_name"] = None
    normalized["trainer"]["native_training_variant"] = None
    for key in (
            "trace_output_dir", "trace_stage", "trace_run_id",
            "trace_checkpoint_digest", "trace_parent_checkpoint_digest"):
        normalized["trainer"][key] = None
if control != cost:
    raise SystemExit("Qwen native B/C configs differ beyond reward and trace identity")
PY
    fi

    "$train_python" - \
        "$config_manifest_dir" \
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

    "$train_python" -m pip freeze --all >"$train_freeze_file"
    "$retriever_python" -m pip freeze --all >"$retriever_freeze_file"
    python_version="$($train_python -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
    torch_version="$($train_python -c 'import torch; print(torch.__version__)')"

    search_mix_handoff_args=()
    if [[ "$seal_search_mix" == 1 ]]; then
        search_mix_handoff_args+=(
            --data "$SEARCH_MIX_DATA_DIR"
            --extra-file "$config_manifest_dir/config-1gpu-group_probe-eval.yaml"
            --extra-file "$config_manifest_dir/config-2gpu-group_probe-eval.yaml"
        )
    fi
    qwen_native_handoff_args=()
    native_handoff_contract_args=()
    native_handoff_verify_args=()
    if [[ "$seal_qwen_native" == 1 ]]; then
        qwen_native_handoff_args+=(
            --data "$QWEN_NATIVE_DATA_DIR"
            --extra-file "$config_manifest_dir/config-2gpu-qwen_native_g1-eval.yaml"
            --extra-file "$config_manifest_dir/config-2gpu-qwen_native_g3-eval.yaml"
            --extra-file "$config_manifest_dir/config-2gpu-qwen_native_a_val-eval.yaml"
            --extra-file "$config_manifest_dir/config-2gpu-qwen_native_r_val-eval.yaml"
            --extra-file "$config_manifest_dir/config-2gpu-qwen_native_b_val-eval.yaml"
            --extra-file "$config_manifest_dir/config-2gpu-qwen_native_c_val-eval.yaml"
            --extra-file "$config_manifest_dir/config-2gpu-qwen_native_a_nq_test-eval.yaml"
            --extra-file "$config_manifest_dir/config-2gpu-qwen_native_r_nq_test-eval.yaml"
            --extra-file "$config_manifest_dir/config-2gpu-qwen_native_b_nq_test-eval.yaml"
            --extra-file "$config_manifest_dir/config-2gpu-qwen_native_c_nq_test-eval.yaml"
            --extra-file "$config_manifest_dir/config-2gpu-qwen_native_a_multihop-eval.yaml"
            --extra-file "$config_manifest_dir/config-2gpu-qwen_native_r_multihop-eval.yaml"
            --extra-file "$config_manifest_dir/config-2gpu-qwen_native_b_multihop-eval.yaml"
            --extra-file "$config_manifest_dir/config-2gpu-qwen_native_c_multihop-eval.yaml"
            --extra-file "$config_manifest_dir/config-2gpu-qwen-native-smoke-train.yaml"
            --extra-file "$config_manifest_dir/config-2gpu-qwen-native-reproduce-train.yaml"
            --extra-file "$config_manifest_dir/config-2gpu-qwen-native-control-train.yaml"
            --extra-file "$config_manifest_dir/config-2gpu-qwen-native-cost_aware_gated-train.yaml"
        )
        native_handoff_contract_args+=(
            --native-prompt-version qwen35-native-search-v4-terminal-answer-only
            --native-thinking-enabled
            --max-action-budget 4
            --selection-observation-length 384
            --rollout-observation-length 500
        )
        native_handoff_verify_args+=(
            --expect-native-prompt-version qwen35-native-search-v4-terminal-answer-only
            --expect-native-thinking-enabled
            --expect-max-action-budget 4
            --expect-selection-observation-length 384
            --expect-rollout-observation-length 500
            --require-artifact data/search_mix_qwen35_native_v4/manifest.json
            --require-artifact data/search_mix_qwen35_native_v4/catalog.jsonl
            --require-artifact data/search_mix_qwen35_native_v4/search_mix_answer_quality_exclusions.v1.json
            --require-artifact data/search_mix_qwen35_native_v4/train_512.parquet
            --require-artifact data/search_mix_qwen35_native_v4/val_128.parquet
            --require-artifact data/search_mix_qwen35_native_v4/probe_g0_8.parquet
            --require-artifact data/search_mix_qwen35_native_v4/probe_autonomous_16.parquet
            --require-artifact data/search_mix_qwen35_native_v4/probe_multi_64.parquet
            --require-artifact data/search_mix_qwen35_native_v4/nq_test_128_native_v4.parquet
            --require-artifact data/search_mix_qwen35_native_v4/multihop_eval_256_native_v4.parquet
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
        "${qwen_native_handoff_args[@]}" \
        --requirements "$CHECKOUT_DIR/requirements-autodl.lock" \
        --extra-file "$CORPUS_GZIP" \
        --extra-file "$train_freeze_file" \
        --extra-file "$retriever_freeze_file" \
        --extra-file "$java_version_file" \
        --extra-file "$config_manifest_dir/config-1gpu-smoke-train.yaml" \
        --extra-file "$config_manifest_dir/config-1gpu-reproduce-train.yaml" \
        --extra-file "$config_manifest_dir/config-1gpu-control-train.yaml" \
        --extra-file "$config_manifest_dir/config-1gpu-cost_aware-train.yaml" \
        --extra-file "$config_manifest_dir/config-1gpu-cost_aware_gated-train.yaml" \
        --extra-file "$config_manifest_dir/config-1gpu-cost_aware_gated-gate-train.yaml" \
        --extra-file "$config_manifest_dir/config-1gpu-base-eval.yaml" \
        --extra-file "$config_manifest_dir/config-1gpu-reproduced-eval.yaml" \
        --extra-file "$config_manifest_dir/config-1gpu-control-eval.yaml" \
        --extra-file "$config_manifest_dir/config-1gpu-cost_aware-eval.yaml" \
        --extra-file "$config_manifest_dir/config-1gpu-cost_aware_gated-eval.yaml" \
        --extra-file "$config_manifest_dir/config-1gpu-control-trace-eval.yaml" \
        --extra-file "$config_manifest_dir/config-1gpu-cost_aware-trace-eval.yaml" \
        --extra-file "$config_manifest_dir/config-2gpu-smoke-train.yaml" \
        --extra-file "$config_manifest_dir/config-2gpu-reproduce-train.yaml" \
        --extra-file "$config_manifest_dir/config-2gpu-control-train.yaml" \
        --extra-file "$config_manifest_dir/config-2gpu-cost_aware-train.yaml" \
        --extra-file "$config_manifest_dir/config-2gpu-cost_aware_gated-train.yaml" \
        --extra-file "$config_manifest_dir/config-2gpu-cost_aware_gated-gate-train.yaml" \
        --extra-file "$config_manifest_dir/config-2gpu-base-eval.yaml" \
        --extra-file "$config_manifest_dir/config-2gpu-reproduced-eval.yaml" \
        --extra-file "$config_manifest_dir/config-2gpu-control-eval.yaml" \
        --extra-file "$config_manifest_dir/config-2gpu-cost_aware-eval.yaml" \
        --extra-file "$config_manifest_dir/config-2gpu-cost_aware_gated-eval.yaml" \
        --extra-file "$config_manifest_dir/config-2gpu-control-trace-eval.yaml" \
        --extra-file "$config_manifest_dir/config-2gpu-cost_aware-trace-eval.yaml" \
        --extra-file "$SEARCH_GATE_DATA_DIR/eval_256.parquet" \
        --extra-file "$SEARCH_GATE_DATA_DIR/catalog.jsonl" \
        --extra-file "$SEARCH_GATE_DATA_DIR/manifest.json" \
        --extra-file "$SEARCH_GATE_DATA_DIR/manifest.json.sha256" \
        --extra-file "$SEARCH_GATE_DATA_DIR/sources/hotpotqa/dev.jsonl" \
        --extra-file "$SEARCH_GATE_DATA_DIR/sources/2wikimultihopqa/dev.jsonl" \
        --extra-file "$config_manifest_dir/config-1gpu-search_opportunity-eval.yaml" \
        --extra-file "$config_manifest_dir/config-2gpu-search_opportunity-eval.yaml" \
        --python-version "$python_version" \
        --torch-version "$torch_version" \
        "${native_handoff_contract_args[@]}" \
        --output "$candidate_handoff"
    "$train_python" "$CHECKOUT_DIR/scripts/autodl/handoff.py" verify \
        --root "$PROJECT_ROOT" \
        --commit "$commit" \
        --python-version "$python_version" \
        --torch-version "$torch_version" \
        --manifest "$candidate_handoff" \
        "${native_handoff_verify_args[@]}"
    verify_checkout "$commit"

    handoff_digest="$(cut -d' ' -f1 "$candidate_handoff.sha256")"
    if [[ "${AUTODL_QWEN_NATIVE_INCREMENTAL:-0}" == 1 ]]; then
        atomic_write "$seal_dir/cpu.ok" "$handoff_digest"$'\n'
        sync_path "$seal_dir"
        PYTHON_BIN="$train_python" cpu_seal_transaction promote "$seal_dir"
    else
        atomic_write "$MANIFEST_DIR/cpu.ok" "$handoff_digest"$'\n'
    fi
    "$train_python" "$CHECKOUT_DIR/scripts/autodl/handoff.py" verify \
        --root "$PROJECT_ROOT" \
        --commit "$commit" \
        --python-version "$python_version" \
        --torch-version "$torch_version" \
        --manifest "$HANDOFF" \
        "${native_handoff_verify_args[@]}"
    handoff_digest="$(sha256sum -- "$HANDOFF" | cut -d' ' -f1)"
    [[ "$(tr -d '\r\n' <"$MANIFEST_DIR/cpu.ok")" == "$handoff_digest" ]] || {
        printf 'Published cpu.ok does not match the canonical handoff.\n' >&2
        return 1
    }
    verify_checkout "$commit"
    sync_path "$MANIFEST_DIR"
}

case "${1:-}" in
    --worker)
        phase_worker cpu "${2:?missing attempt directory}" "$0"
        ;;
    --action)
        cpu_action "${2:?missing attempt directory}"
        ;;
    --recover-cpu-seal)
        [[ $# == 1 ]] || { printf 'Usage: bash %s --recover-cpu-seal\n' "$0" >&2; exit 64; }
        locked_cpu_seal_transaction recover
        ;;
    --promote-cpu-seal)
        [[ $# == 2 ]] || { printf 'Usage: bash %s --promote-cpu-seal SEAL_DIR\n' "$0" >&2; exit 64; }
        locked_cpu_seal_transaction promote "$2"
        ;;
    '')
        phase_launch cpu "$0"
        ;;
    *)
        printf 'Usage: bash %s\n' "$0" >&2
        exit 64
        ;;
esac
