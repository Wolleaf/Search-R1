#!/usr/bin/env python3
"""Create and verify the immutable CPU-to-GPU artifact handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Iterable

SCHEMA = 2
MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
DATA_REVISION = "bcafb8dd07d453be3cbeeeb3f78be1841bddf92c"
BM25_REVISION = "2c7554f25f425038c4bcb155735a0f831851fd78"
CORPUS_REVISION = "69c1c00ffe7c5554c68d8548355cb22e46aabc51"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def iter_artifacts(directories: Iterable[Path], files: Iterable[Path]) -> list[Path]:
    found: set[Path] = set()
    for directory in directories:
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError(f"artifact directory is missing or a symlink: {directory}")
        for path in directory.rglob("*"):
            relative = path.relative_to(directory)
            if ".cache" in relative.parts:
                continue
            if path.is_symlink():
                raise ValueError(f"artifact symlink is not allowed: {path}")
            if path.is_file():
                found.add(path.resolve())
    for path in files:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"artifact file is missing or a symlink: {path}")
        found.add(path.resolve())
    return sorted(found)


def contained_relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"artifact escapes persistent root: {path}") from error


def create(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    artifacts = []
    paths = iter_artifacts([args.model, args.bm25, args.corpus, *args.data],
                           [args.requirements, *args.extra_file])
    for path in paths:
        artifacts.append({
            "bytes": path.stat().st_size,
            "path": contained_relative(root, path),
            "sha256": sha256_file(path),
        })
    artifacts.sort(key=lambda item: item["path"])

    payload = {
        "artifacts": artifacts,
        "bm25_revision": BM25_REVISION,
        "checkout_commit": args.commit.lower(),
        "corpus_revision": CORPUS_REVISION,
        "data_revision": DATA_REVISION,
        "model_revision": MODEL_REVISION,
        "persistent_root": str(root),
        "python_version": args.python_version,
        "schema": SCHEMA,
        "torch_version": args.torch_version,
    }
    raw = canonical_bytes(payload)
    atomic_write(args.output, raw)
    digest = hashlib.sha256(raw).hexdigest()
    atomic_write(args.output.with_suffix(args.output.suffix + ".sha256"), f"{digest}  {args.output.name}\n".encode())


def has_symlink_component(root: Path, relative: Path) -> bool:
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def verify(args: argparse.Namespace) -> None:
    manifest = args.manifest.resolve()
    sidecar = manifest.with_suffix(manifest.suffix + ".sha256")
    raw = manifest.read_bytes()
    match = re.fullmatch(rb"([0-9a-f]{64})  ([^\r\n]+)\n", sidecar.read_bytes())
    if not match or match.group(2).decode() != manifest.name:
        raise ValueError("malformed handoff checksum sidecar")
    if hashlib.sha256(raw).hexdigest() != match.group(1).decode():
        raise ValueError("handoff manifest checksum mismatch")
    payload = json.loads(raw)
    if canonical_bytes(payload) != raw:
        raise ValueError("handoff manifest is not canonical JSON")
    expected_keys = {
        "artifacts", "bm25_revision", "checkout_commit", "corpus_revision",
        "data_revision", "model_revision", "persistent_root", "python_version",
        "schema", "torch_version",
    }
    if set(payload) != expected_keys:
        raise ValueError("handoff manifest has missing or unknown fields")
    expected = {
        "schema": SCHEMA,
        "persistent_root": str(args.root.resolve()),
        "checkout_commit": args.commit.lower(),
        "model_revision": MODEL_REVISION,
        "data_revision": DATA_REVISION,
        "bm25_revision": BM25_REVISION,
        "corpus_revision": CORPUS_REVISION,
        "python_version": args.python_version,
        "torch_version": args.torch_version,
    }
    for key, value in expected.items():
        if payload[key] != value:
            raise ValueError(f"handoff {key} mismatch: expected {value!r}, got {payload[key]!r}")

    root = args.root.resolve()
    previous = ""
    artifact_paths: set[str] = set()
    for item in payload["artifacts"]:
        if set(item) != {"bytes", "path", "sha256"}:
            raise ValueError("artifact entry has missing or unknown fields")
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() <= previous:
            raise ValueError("artifact paths must be unique, sorted, and relative")
        previous = relative.as_posix()
        artifact_paths.add(previous)
        if has_symlink_component(root, relative):
            raise ValueError(f"artifact path contains a symlink: {relative}")
        path = root / relative
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            raise ValueError(f"artifact size mismatch: {relative}")
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"artifact checksum mismatch: {relative}")

    for required in args.require_artifact:
        relative = Path(required)
        normalized = relative.as_posix()
        if (relative.is_absolute() or not normalized or ".." in relative.parts
                or normalized not in artifact_paths):
            raise ValueError(f"required artifact is not sealed: {required}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    subparsers = result.add_subparsers(dest="command", required=True)
    create_parser = subparsers.add_parser("create")
    create_parser.add_argument("--root", type=Path, required=True)
    create_parser.add_argument("--commit", required=True)
    create_parser.add_argument("--model", type=Path, required=True)
    create_parser.add_argument("--bm25", type=Path, required=True)
    create_parser.add_argument("--corpus", type=Path, required=True)
    create_parser.add_argument("--data", type=Path, action="append", required=True)
    create_parser.add_argument("--requirements", type=Path, required=True)
    create_parser.add_argument("--extra-file", type=Path, action="append", default=[])
    create_parser.add_argument("--python-version", required=True)
    create_parser.add_argument("--torch-version", required=True)
    create_parser.add_argument("--output", type=Path, required=True)
    create_parser.set_defaults(handler=create)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--root", type=Path, required=True)
    verify_parser.add_argument("--commit", required=True)
    verify_parser.add_argument("--python-version", required=True)
    verify_parser.add_argument("--torch-version", required=True)
    verify_parser.add_argument("--manifest", type=Path, required=True)
    verify_parser.add_argument("--require-artifact", action="append", default=[])
    verify_parser.set_defaults(handler=verify)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        args.handler(args)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"handoff error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
