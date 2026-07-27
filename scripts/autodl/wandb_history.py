#!/usr/bin/env python3
"""Read a completed offline WandB run as fail-closed training evidence."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from typing import Any, Iterable


SUPPORTED_WANDB_VERSION = "0.21.1"
RECEIPT_SCHEMA = "search-r1.wandb-offline-receipt"
RECEIPT_SCHEMA_VERSION = 1
CORE_ACTOR_METRICS = (
    "actor/pg_loss",
    "actor/kl_loss",
    "actor/entropy_loss",
    "actor/grad_norm",
    "actor/ppo_kl",
)
METRIC_REL_TOL = 1e-6
METRIC_ABS_TOL = 1e-6
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class WandbHistoryError(ValueError):
    """Raised when an offline run cannot be treated as immutable evidence."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_files(root: Path) -> tuple[list[Path], list[Path]]:
    files: list[Path] = []
    run_files: list[Path] = []

    def walk_error(error: OSError) -> None:
        raise WandbHistoryError(
            f"cannot enumerate offline WandB directory: {root}"
        ) from error

    for current_raw, directories, names in os.walk(
        root, topdown=True, onerror=walk_error, followlinks=False
    ):
        current = Path(current_raw)
        kept_directories: list[str] = []
        for name in sorted(directories):
            candidate = current / name
            info = candidate.lstat()
            if stat.S_ISLNK(info.st_mode):
                continue
            if stat.S_ISDIR(info.st_mode):
                kept_directories.append(name)
        directories[:] = kept_directories
        for name in sorted(names):
            candidate = current / name
            info = candidate.lstat()
            is_run_file = name.startswith("run-") and name.endswith(".wandb")
            if stat.S_ISLNK(info.st_mode):
                if is_run_file:
                    raise WandbHistoryError(
                        f"offline WandB run file must not be a symlink: {candidate}"
                    )
                continue
            if not stat.S_ISREG(info.st_mode):
                if is_run_file:
                    raise WandbHistoryError(
                        f"offline WandB run file is not regular: {candidate}"
                    )
                continue
            if is_run_file:
                run_files.append(candidate)
            if info.st_size > 0:
                files.append(candidate)
    files.sort(key=lambda path: path.relative_to(root).as_posix())
    run_files.sort(key=lambda path: path.relative_to(root).as_posix())
    return files, run_files


def _stable_identity(path: Path) -> tuple[int, int, int, int]:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise WandbHistoryError(f"offline WandB run file is unsafe: {path}")
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _item_key(item: Any) -> str:
    parts = list(item.nested_key) if item.nested_key else [item.key]
    if not parts or any(not isinstance(part, str) or not part for part in parts):
        raise WandbHistoryError("WandB history contains an empty metric key")
    return ".".join(parts)


def _history_values(history: Any) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for item in history.item:
        key = _item_key(item)
        if key in values:
            raise WandbHistoryError(f"duplicate WandB history metric: {key}")
        try:
            values[key] = json.loads(item.value_json)
        except (TypeError, json.JSONDecodeError) as error:
            raise WandbHistoryError(
                f"invalid JSON for WandB history metric {key}"
            ) from error
    return values


def _summary_update(summary: Any, values: dict[str, Any]) -> None:
    for item in summary.update:
        key = _item_key(item)
        try:
            values[key] = json.loads(item.value_json)
        except (TypeError, json.JSONDecodeError) as error:
            raise WandbHistoryError(
                f"invalid JSON for WandB summary metric {key}"
            ) from error
    for item in summary.remove:
        values.pop(_item_key(item), None)


def _history_step(values: dict[str, Any]) -> int:
    value = values.get("_step")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or int(value) != value
        or int(value) < 0
    ):
        raise WandbHistoryError("WandB history record has no valid integer _step")
    return int(value)


def _scan_records(
    path: Path, metric_keys: frozenset[str]
) -> tuple[
    dict[str, int], list[dict[str, Any]], dict[str, Any], list[int], str | None
]:
    try:
        import wandb
        from wandb.proto import wandb_internal_pb2
        from wandb.sdk.internal.datastore import (
            LEVELDBLOG_BLOCK_LEN,
            LEVELDBLOG_HEADER_LEN,
            DataStore,
        )
    except (ImportError, AttributeError) as error:
        raise WandbHistoryError("the pinned WandB reader is unavailable") from error

    if wandb.__version__ != SUPPORTED_WANDB_VERSION:
        raise WandbHistoryError(
            "unsupported WandB reader version: "
            f"expected {SUPPORTED_WANDB_VERSION}, found {wandb.__version__}"
        )

    size = path.stat().st_size
    record_counts: dict[str, int] = {}
    histories: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    exit_codes: list[int] = []
    last_record_type: str | None = None
    store = DataStore()
    try:
        store.open_for_scan(str(path))
        while True:
            before = store._index
            payload = store.scan_data()
            if payload is None:
                if store._index != before:
                    space_left = LEVELDBLOG_BLOCK_LEN - (
                        before % LEVELDBLOG_BLOCK_LEN
                    )
                    trailing_padding = (
                        space_left < LEVELDBLOG_HEADER_LEN
                        and before + space_left == size
                        and store._index == size
                    )
                    if not trailing_padding:
                        raise WandbHistoryError(
                            "offline WandB run ends inside a fragmented record"
                        )
                break
            record = wandb_internal_pb2.Record()
            consumed = record.ParseFromString(payload)
            if consumed != len(payload):
                raise WandbHistoryError("WandB protobuf record was not fully parsed")
            kind = record.WhichOneof("record_type")
            if kind is None:
                raise WandbHistoryError("WandB record has no record type")
            last_record_type = kind
            record_counts[kind] = record_counts.get(kind, 0) + 1
            if kind == "history":
                values = _history_values(record.history)
                step = _history_step(values)
                if record.history.HasField("step") and record.history.step.num != step:
                    raise WandbHistoryError(
                        "WandB history protobuf step disagrees with its _step metric"
                    )
                selected = {
                    key: values[key]
                    for key in sorted(metric_keys)
                    if key in values
                }
                histories.append({"metrics": selected, "step": step})
            elif kind == "summary":
                _summary_update(record.summary, summary)
            elif kind == "exit":
                exit_codes.append(int(record.exit.exit_code))
        if store._index != size or store._fp.tell() != size:
            raise WandbHistoryError("offline WandB run was not scanned to exact EOF")
    except WandbHistoryError:
        raise
    except Exception as error:
        raise WandbHistoryError(f"cannot parse offline WandB run: {path}") from error
    finally:
        try:
            store.close()
        except Exception as error:
            raise WandbHistoryError(f"cannot close offline WandB run: {path}") from error
    return record_counts, histories, summary, exit_codes, last_record_type


def scan_offline_run(
    root: Path, *, metric_keys: Iterable[str] = ()
) -> dict[str, Any]:
    """Scan the unique completed ``run-*.wandb`` below ``root``.

    The returned dictionary is JSON-compatible unless a selected metric itself
    contains a non-finite JSON number. Callers must validate selected metric
    semantics before serializing the result.
    """

    root = Path(root)
    try:
        root_info = root.lstat()
    except OSError as error:
        raise WandbHistoryError(f"offline WandB directory is missing: {root}") from error
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise WandbHistoryError(f"offline WandB path is not a regular directory: {root}")

    _, run_files = _tree_files(root)
    if len(run_files) != 1:
        raise WandbHistoryError(
            "expected exactly one regular non-symlink run-*.wandb file, "
            f"found {len(run_files)}"
        )
    run_file = run_files[0]
    identity = _stable_identity(run_file)
    (
        record_counts,
        histories,
        summary,
        exit_codes,
        last_record_type,
    ) = _scan_records(run_file, frozenset(metric_keys))
    if _stable_identity(run_file) != identity:
        raise WandbHistoryError("offline WandB run changed while it was scanned")

    files, run_files_after = _tree_files(root)
    if run_files_after != [run_file]:
        raise WandbHistoryError("offline WandB run set changed while it was scanned")
    identities = {path: _stable_identity(path) for path in files}
    if identities.get(run_file) != identity:
        raise WandbHistoryError("offline WandB run changed before it was hashed")
    entries: list[tuple[str, str]] = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        if "\n" in relative or "\r" in relative:
            raise WandbHistoryError(f"unsafe WandB evidence path: {path}")
        entries.append((relative, _sha256_file(path)))
    final_files, final_run_files = _tree_files(root)
    if final_run_files != [run_file] or final_files != files:
        raise WandbHistoryError("offline WandB file set changed while it was hashed")
    if any(_stable_identity(path) != expected for path, expected in identities.items()):
        raise WandbHistoryError("offline WandB file changed while it was hashed")
    tree_payload = "".join(
        f"{digest}  {name}\n" for name, digest in entries
    ).encode("utf-8")
    tree_digest = hashlib.sha256(tree_payload).hexdigest() if entries else ""
    run_relative = run_file.relative_to(root).as_posix()
    run_digest = dict(entries).get(run_relative)
    if not run_digest:
        raise WandbHistoryError("offline WandB run file is empty")

    return {
        "exit_codes": exit_codes,
        "file_count": len(entries),
        "history": histories,
        "history_count": record_counts.get("history", 0),
        "history_steps": [record["step"] for record in histories],
        "last_record_type": last_record_type,
        "record_counts": dict(sorted(record_counts.items())),
        "run_file": run_relative,
        "run_file_sha256": run_digest,
        "summary": dict(sorted(summary.items())),
        "summary_count": record_counts.get("summary", 0),
        "tree_sha256": tree_digest,
        "wandb_version": SUPPORTED_WANDB_VERSION,
    }


def _metric_lines(path: Path) -> dict[int, dict[str, list[float]]]:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise WandbHistoryError(f"training log is not a regular file: {path}")
    metrics: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for raw_line in path.read_text(encoding="utf-8", errors="strict").splitlines():
        line = ANSI_ESCAPE.sub("", raw_line)
        marker = line.find("step:")
        if marker < 0:
            continue
        fields = line[marker:].split(" - ")
        if not fields or not fields[0].startswith("step:"):
            continue
        try:
            step = int(fields[0].split(":", 1)[1])
        except ValueError:
            continue
        for field in fields[1:]:
            if ":" not in field:
                continue
            key, raw_value = field.rsplit(":", 1)
            try:
                value = float(raw_value)
            except ValueError:
                continue
            metrics[step][key].append(value)
    return metrics


def _selected_metrics(
    scan: dict[str, Any], steps: Iterable[int]
) -> dict[int, dict[str, list[Any]]]:
    selected = {
        step: {key: [] for key in CORE_ACTOR_METRICS} for step in steps
    }
    for history in scan["history"]:
        step = history["step"]
        if step not in selected:
            continue
        for key in CORE_ACTOR_METRICS:
            if key in history["metrics"]:
                selected[step][key].append(history["metrics"][key])
    return selected


def _metrics_match(
    history: dict[int, dict[str, list[Any]]],
    log: dict[int, dict[str, list[float]]],
    steps: Iterable[int],
) -> bool:
    steps = tuple(steps)
    logged_actor_steps = sorted(
        step
        for step, metrics in log.items()
        if any(metrics.get(key) for key in CORE_ACTOR_METRICS)
    )
    if logged_actor_steps != list(steps):
        return False
    for step in steps:
        for key in CORE_ACTOR_METRICS:
            history_values = history[step][key]
            log_values = log.get(step, {}).get(key, [])
            if not history_values or len(history_values) != len(log_values):
                return False
            for history_value, log_value in zip(history_values, log_values):
                if (
                    isinstance(history_value, bool)
                    or not isinstance(history_value, (int, float))
                    or not math.isfinite(float(history_value))
                    or not math.isfinite(log_value)
                    or not math.isclose(
                        float(history_value),
                        log_value,
                        rel_tol=METRIC_REL_TOL,
                        abs_tol=METRIC_ABS_TOL,
                    )
                ):
                    return False
    return True


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def build_receipt(
    wandb_dir: Path,
    *,
    log: Path | None = None,
    expected_steps: Iterable[int] = (),
) -> dict[str, Any]:
    """Build a canonicalizable receipt for a training or evaluation run."""

    steps = tuple(expected_steps)
    if len(set(steps)) != len(steps) or any(
        isinstance(step, bool) or not isinstance(step, int) or step < 0
        for step in steps
    ):
        raise WandbHistoryError("expected WandB steps must be unique integers")
    steps = tuple(sorted(steps))
    if (log is None) != (not steps):
        raise WandbHistoryError(
            "training receipts require both --log and --expected-step; "
            "evaluation receipts require neither"
        )

    metric_keys = CORE_ACTOR_METRICS if log is not None else ()
    scan = scan_offline_run(wandb_dir, metric_keys=metric_keys)
    observed_steps = scan["history_steps"]
    history_metrics = _selected_metrics(scan, steps)
    log_digest = _sha256_file(log) if log is not None else None
    log_metrics = _metric_lines(log) if log is not None else {}
    if log is not None and _sha256_file(log) != log_digest:
        raise WandbHistoryError("training log changed while it was scanned")

    checks: dict[str, dict[str, Any]] = {}

    def add(name: str, passed: bool, observed: Any) -> None:
        checks[name] = {"observed": _json_safe(observed), "passed": bool(passed)}

    add("history_present", scan["history_count"] > 0, scan["history_count"])
    add(
        "expected_steps",
        log is None or observed_steps == list(steps),
        observed_steps,
    )
    add(
        "actor_metrics_match_log",
        log is None or _metrics_match(history_metrics, log_metrics, steps),
        (
            "not-required"
            if log is None
            else {
                str(step): history_metrics[step]
                for step in steps
            }
        ),
    )
    add("summary_present", scan["summary_count"] > 0, scan["summary_count"])
    add(
        "exit_zero",
        scan["exit_codes"] == [0] and scan["last_record_type"] == "exit",
        {
            "codes": scan["exit_codes"],
            "last_record_type": scan["last_record_type"],
        },
    )

    return {
        "checks": checks,
        "decision": (
            "GO" if all(check["passed"] for check in checks.values()) else "NO-GO"
        ),
        "inputs": {
            "expected_steps": list(steps),
            "log_sha256": log_digest,
            "run_file_sha256": scan["run_file_sha256"],
            "wandb_tree_sha256": scan["tree_sha256"],
        },
        "metrics": {
            "exit_codes": scan["exit_codes"],
            "files": scan["file_count"],
            "history_records": scan["history_count"],
            "history_steps": scan["history_steps"],
            "last_record_type": scan["last_record_type"],
            "record_counts": scan["record_counts"],
            "run_file": scan["run_file"],
            "summary": _json_safe(scan["summary"]),
            "summary_records": scan["summary_count"],
            "wandb_version": scan["wandb_version"],
        },
        "mode": "train" if log is not None else "eval",
        "schema": RECEIPT_SCHEMA,
        "schema_version": RECEIPT_SCHEMA_VERSION,
    }


def canonical_receipt_bytes(receipt: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            receipt,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _atomic_write(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, path)
        except FileExistsError as error:
            raise WandbHistoryError(f"WandB receipt already exists: {path}") from error
        os.unlink(temporary_name)
        temporary_name = ""
        if os.name == "posix":
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            directory_fd = os.open(path.parent, flags)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _canonical_output_path(
    output: Path, wandb_dir: Path, log: Path | None
) -> Path:
    output = Path(output)
    try:
        output.lstat()
    except FileNotFoundError:
        pass
    else:
        raise WandbHistoryError(f"WandB receipt output already exists: {output}")
    try:
        parent_info = output.parent.lstat()
        canonical_parent = output.parent.resolve(strict=True)
    except OSError as error:
        raise WandbHistoryError(
            f"WandB receipt parent is unavailable: {output.parent}"
        ) from error
    if stat.S_ISLNK(parent_info.st_mode) or not stat.S_ISDIR(parent_info.st_mode):
        raise WandbHistoryError(
            f"WandB receipt parent is not a regular directory: {output.parent}"
        )
    if canonical_parent != output.parent.absolute():
        raise WandbHistoryError("WandB receipt output path is not canonical")
    candidate = canonical_parent / output.name
    wandb_root = Path(wandb_dir).resolve(strict=True)
    try:
        candidate.relative_to(wandb_root)
    except ValueError:
        pass
    else:
        raise WandbHistoryError("WandB receipt output must be outside the WandB tree")
    if log is not None and candidate == Path(log).resolve(strict=True):
        raise WandbHistoryError("WandB receipt output aliases the training log")
    return candidate


def _validate_receipt_path(path: Path, wandb_dir: Path) -> Path:
    path = Path(path)
    canonical = path.resolve(strict=True)
    wandb_root = Path(wandb_dir).resolve(strict=True)
    try:
        canonical.relative_to(wandb_root)
    except ValueError:
        return path
    raise WandbHistoryError("WandB receipt must be outside the WandB tree")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wandb-dir", type=Path, required=True)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path)
    destination.add_argument("--verify-receipt", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--expected-step", type=int, action="append", default=[])
    return parser


def _regular_receipt_bytes(path: Path) -> bytes:
    try:
        info = path.lstat()
    except OSError as error:
        raise WandbHistoryError(f"WandB receipt is missing: {path}") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise WandbHistoryError(f"WandB receipt is not a regular file: {path}")
    return path.read_bytes()


def main() -> int:
    args = _build_parser().parse_args()
    try:
        if args.output is not None:
            args.output = _canonical_output_path(
                args.output, args.wandb_dir, args.log
            )
        else:
            args.verify_receipt = _validate_receipt_path(
                args.verify_receipt, args.wandb_dir
            )
        receipt = build_receipt(
            args.wandb_dir,
            log=args.log,
            expected_steps=args.expected_step,
        )
        payload = canonical_receipt_bytes(receipt)
        if args.output is not None:
            _atomic_write(args.output, payload)
        elif _regular_receipt_bytes(args.verify_receipt) != payload:
            raise WandbHistoryError(
                "WandB receipt does not match the current binary evidence and mode"
            )
    except (OSError, WandbHistoryError) as error:
        print(f"WandB history scan failed: {error}", file=sys.stderr)
        return 1
    print(receipt["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
