#!/usr/bin/env python3
"""Extract the pinned wiki corpus and build a low-memory row offset index."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import tarfile
import tempfile
from typing import BinaryIO

SCHEMA = 1
MEMBER_NAME = "data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/wiki_dump.jsonl"
JSONL_NAME = "wiki-18.jsonl"
OFFSETS_NAME = "wiki-18.offsets.u64"
MANIFEST_NAME = "manifest.json"


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True,
                       separators=(",", ":")) + "\n").encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _artifact(path: Path) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"corpus artifact is missing or a symlink: {path}")
    return {
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def verify(output_dir: Path, revision: str, source_sha256: str,
           source_bytes: int, member_bytes: int) -> dict[str, object]:
    manifest_path = output_dir / MANIFEST_NAME
    raw = manifest_path.read_bytes()
    payload = json.loads(raw)
    if canonical_bytes(payload) != raw:
        raise ValueError("corpus manifest is not canonical JSON")
    if set(payload) != {"artifacts", "member", "rows", "schema", "source"}:
        raise ValueError("corpus manifest has missing or unknown fields")
    if payload["schema"] != SCHEMA:
        raise ValueError("corpus manifest schema mismatch")
    if payload["source"] != {
            "bytes": source_bytes,
            "revision": revision,
            "sha256": source_sha256,
    }:
        raise ValueError("corpus source provenance mismatch")
    if payload["member"] != {"bytes": member_bytes, "name": MEMBER_NAME}:
        raise ValueError("corpus tar member mismatch")
    if set(payload["artifacts"]) != {JSONL_NAME, OFFSETS_NAME}:
        raise ValueError("corpus artifact list mismatch")

    jsonl_path = output_dir / JSONL_NAME
    offsets_path = output_dir / OFFSETS_NAME
    for path in (jsonl_path, offsets_path):
        if payload["artifacts"][path.name] != _artifact(path):
            raise ValueError(f"corpus artifact digest mismatch: {path.name}")

    rows = payload["rows"]
    if not isinstance(rows, int) or rows <= 0:
        raise ValueError("corpus row count is invalid")
    if offsets_path.stat().st_size != (rows + 1) * 8:
        raise ValueError("corpus offset count does not match the row count")
    with offsets_path.open("rb") as handle:
        first = struct.unpack("<Q", handle.read(8))[0]
        handle.seek(-8, os.SEEK_END)
        last = struct.unpack("<Q", handle.read(8))[0]
    if first != 0 or last != jsonl_path.stat().st_size:
        raise ValueError("corpus offset boundaries do not match the JSONL file")
    return payload


def _write_row(row: int, line: bytes, jsonl: BinaryIO, offsets: BinaryIO,
               jsonl_digest: hashlib._Hash, offsets_digest: hashlib._Hash,
               position: int) -> int:
    payload = json.loads(line)
    if not isinstance(payload, dict):
        raise ValueError(f"corpus row {row} is not a JSON object")
    if str(payload.get("id")) != str(row):
        raise ValueError(f"corpus row {row} has mismatched id {payload.get('id')!r}")
    contents = payload.get("contents")
    if not isinstance(contents, str) or not contents.strip():
        raise ValueError(f"corpus row {row} has no non-empty contents")

    jsonl.write(line)
    jsonl_digest.update(line)
    position += len(line)
    encoded_offset = struct.pack("<Q", position)
    offsets.write(encoded_offset)
    offsets_digest.update(encoded_offset)
    return position


def prepare(source: Path, output_dir: Path, revision: str, source_sha256: str,
            source_bytes: int, member_bytes: int) -> dict[str, object]:
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"corpus source is missing or a symlink: {source}")
    if source.stat().st_size != source_bytes:
        raise ValueError("corpus source size mismatch")
    if sha256_file(source) != source_sha256:
        raise ValueError("corpus source SHA-256 mismatch")

    if output_dir.is_symlink():
        raise ValueError("corpus output directory must not be a symlink")
    if output_dir.exists():
        manifest_path = output_dir / MANIFEST_NAME
        if not manifest_path.is_file():
            raise ValueError("unsealed corpus output exists; preserve and inspect it")
        payload = verify(output_dir, revision, source_sha256, source_bytes,
                         member_bytes)
        print(f"Reused verified corpus with {payload['rows']} rows", flush=True)
        return payload

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir.parent / f".{output_dir.name}.building"
    if staging_dir.exists() or staging_dir.is_symlink():
        raise ValueError(
            f"interrupted corpus build exists; preserve and inspect {staging_dir}")
    staging_dir.mkdir()
    jsonl_path = staging_dir / JSONL_NAME
    offsets_path = staging_dir / OFFSETS_NAME
    rows = 0
    position = 0
    jsonl_digest = hashlib.sha256()
    offsets_digest = hashlib.sha256()
    initial_offset = struct.pack("<Q", 0)
    with jsonl_path.open("xb") as jsonl, offsets_path.open("xb") as offsets:
        offsets.write(initial_offset)
        offsets_digest.update(initial_offset)
        regular_members = 0
        with tarfile.open(source, mode="r|gz") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                regular_members += 1
                if member.name != MEMBER_NAME or member.size != member_bytes:
                    raise ValueError(
                        f"unexpected corpus tar member: {member.name} ({member.size} bytes)")
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError("cannot read corpus tar member")
                for line in extracted:
                    position = _write_row(rows, line, jsonl, offsets,
                                          jsonl_digest, offsets_digest,
                                          position)
                    rows += 1
                    if rows % 1_000_000 == 0:
                        print(f"Indexed {rows} corpus rows", flush=True)
        if regular_members != 1:
            raise ValueError("corpus archive must contain exactly one regular file")
        if position != member_bytes:
            raise ValueError("extracted corpus size does not match the tar header")
        jsonl.flush()
        offsets.flush()
        os.fsync(jsonl.fileno())
        os.fsync(offsets.fileno())

    payload = {
        "artifacts": {
            JSONL_NAME: {
                "bytes": position,
                "sha256": jsonl_digest.hexdigest(),
            },
            OFFSETS_NAME: {
                "bytes": (rows + 1) * 8,
                "sha256": offsets_digest.hexdigest(),
            },
        },
        "member": {"bytes": member_bytes, "name": MEMBER_NAME},
        "rows": rows,
        "schema": SCHEMA,
        "source": {
            "bytes": source_bytes,
            "revision": revision,
            "sha256": source_sha256,
        },
    }
    atomic_write(staging_dir / MANIFEST_NAME, canonical_bytes(payload))
    os.replace(staging_dir, output_dir)
    if os.name != "nt":
        directory_fd = os.open(output_dir.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    print(f"Prepared {rows} corpus rows in {output_dir}", flush=True)
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    for command in ("prepare", "verify"):
        subparser = subparsers.add_parser(command)
        if command == "prepare":
            subparser.add_argument("--source", type=Path, required=True)
        subparser.add_argument("--output-dir", type=Path, required=True)
        subparser.add_argument("--revision", required=True)
        subparser.add_argument("--source-sha256", required=True)
        subparser.add_argument("--source-bytes", type=int, required=True)
        subparser.add_argument("--member-bytes", type=int, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == "prepare":
            prepare(args.source, args.output_dir, args.revision,
                    args.source_sha256, args.source_bytes, args.member_bytes)
        else:
            payload = verify(args.output_dir, args.revision,
                             args.source_sha256, args.source_bytes,
                             args.member_bytes)
            print(f"Verified corpus with {payload['rows']} rows")
    except (OSError, ValueError, json.JSONDecodeError, tarfile.TarError) as error:
        print(f"corpus preparation error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
