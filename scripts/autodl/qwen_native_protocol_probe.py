#!/usr/bin/env python3
"""Run the small direct-HF/native-manager protocol comparison used by G0."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping, Sequence


SCHEMA = "search-r1.qwen-native-protocol-probe"
SCHEMA_VERSION = 1
EXPECTED_QUESTIONS = 8
GROUP_SIZE = 2
MODES = ("direct", "native_manager", "legacy_manager")
SAMPLING = {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
}


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(path for path in root.rglob("*") if path.is_file()
                   and ".cache" not in path.parts)
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"model tree contains a symlink: {path}")
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative + b"\0" + sha256_file(path).encode("ascii") + b"\n")
    return digest.hexdigest()


def parsed_record(parsed: Any) -> dict[str, Any]:
    return {
        "action": parsed.action if parsed.valid else "invalid",
        "content": parsed.content,
        "error": parsed.error,
        "prefix": parsed.prefix,
        "valid": bool(parsed.valid),
    }


def validate_record(record: Mapping[str, Any]) -> None:
    expected = {
        "sample_id", "group_slot", "mode", "prompt_token_sha256", "raw_text",
        "parsed_action", "generation_events", "retrieval_events", "final_answer",
    }
    if set(record) != expected:
        raise ValueError("protocol probe record has missing or unknown fields")
    if not isinstance(record["sample_id"], str) or not record["sample_id"]:
        raise ValueError("protocol probe sample_id is invalid")
    if record["group_slot"] not in (0, 1) or record["mode"] not in MODES:
        raise ValueError("protocol probe identity is invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", str(record["prompt_token_sha256"])):
        raise ValueError("protocol probe prompt digest is invalid")
    if not isinstance(record["raw_text"], str):
        raise ValueError("protocol probe raw_text must be a string")
    action = record["parsed_action"]
    if not isinstance(action, Mapping) or set(action) != {
            "action", "content", "error", "prefix", "valid"}:
        raise ValueError("protocol probe parsed action is invalid")
    if not isinstance(record["generation_events"], list) \
            or not isinstance(record["retrieval_events"], list):
        raise ValueError("protocol probe events must be lists")


def publish(output_dir: Path, resolved: Mapping[str, Any],
            records: list[dict[str, Any]], checkpoint_digest: str) -> None:
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError(f"refusing to overwrite protocol probe output: {output_dir}")
    if len(records) != EXPECTED_QUESTIONS * GROUP_SIZE * len(MODES):
        raise ValueError("protocol probe must contain exactly 48 records")
    identities = set()
    mode_counts = {mode: 0 for mode in MODES}
    for record in records:
        validate_record(record)
        identity = (record["mode"], record["sample_id"], record["group_slot"])
        if identity in identities:
            raise ValueError(f"duplicate protocol probe record: {identity}")
        identities.add(identity)
        mode_counts[record["mode"]] += 1
    if set(mode_counts.values()) != {EXPECTED_QUESTIONS * GROUP_SIZE}:
        raise ValueError(f"protocol probe mode counts are invalid: {mode_counts}")

    output_dir.mkdir(parents=True)
    records_bytes = b"".join(canonical_bytes(record) for record in sorted(
        records, key=lambda item: (item["mode"], item["sample_id"], item["group_slot"])))
    atomic_write(output_dir / "records.jsonl", records_bytes)
    atomic_write(output_dir / "resolved-config.json", canonical_bytes(dict(resolved)))
    manifest = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "checkpoint_digest": checkpoint_digest,
        "mode_counts": mode_counts,
        "records_sha256": sha256_bytes(records_bytes),
        "resolved_config_sha256": sha256_file(output_dir / "resolved-config.json"),
    }
    atomic_write(output_dir / "manifest.json", canonical_bytes(manifest))


def load_fixture(path: Path, checkpoint_digest: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_bytes())
    if not isinstance(payload, Mapping) or set(payload) != {"resolved_config", "records"}:
        raise ValueError("fixture must contain only resolved_config and records")
    resolved = payload["resolved_config"]
    records = payload["records"]
    if not isinstance(resolved, dict) or not isinstance(records, list):
        raise ValueError("fixture resolved_config/records types are invalid")
    if resolved.get("sampling") != SAMPLING:
        raise ValueError("fixture sampling does not match the registered G0 config")
    if resolved.get("checkpoint_digest") != checkpoint_digest:
        raise ValueError("fixture checkpoint digest mismatch")
    return resolved, records


def normalize_messages(value: Any) -> list[dict[str, Any]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list) or not value:
        raise ValueError("Parquet prompt must be a non-empty message list")
    result = []
    for message in value:
        if hasattr(message, "item"):
            message = message.item()
        if not isinstance(message, Mapping):
            raise ValueError("Parquet prompt message must be an object")
        role, content = message.get("role"), message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError("Parquet prompt role/content must be strings")
        result.append({"role": role, "content": content})
    return result


def load_probe_rows(data_path: Path, manifest_path: Path,
                    artifact_name: str,
                    expected_rows: int) -> list[tuple[str, list[dict[str, Any]]]]:
    import pandas as pd

    manifest = json.loads(manifest_path.read_bytes())
    artifact = manifest.get("artifacts", {}).get(artifact_name, {})
    sample_ids = artifact.get("sample_ids")
    if artifact.get("file") != data_path.name or artifact.get("rows") != expected_rows \
            or not isinstance(sample_ids, list) or len(sample_ids) != expected_rows \
            or artifact.get("sha256") != sha256_file(data_path):
        raise ValueError(f"probe artifact contract mismatch: {artifact_name}")
    frame = pd.read_parquet(data_path)
    if len(frame) != expected_rows:
        raise ValueError(f"{data_path} must contain {expected_rows} rows")
    return [(str(sample_id), normalize_messages(frame.iloc[index]["prompt"]))
            for index, sample_id in enumerate(sample_ids)]


def prompt_text(tokenizer: Any, messages: list[dict[str, Any]], native: bool) -> str:
    if native:
        from search_r1.llm_agent.tool_protocol import render_qwen35_prompt
        return render_qwen35_prompt(tokenizer, messages)
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                             tokenize=False)
    return messages[0]["content"]


def make_batch(tokenizer: Any, rows: list[tuple[str, list[dict[str, Any]]]],
               native: bool, device: Any) -> tuple[Any, list[str], list[int], list[Any]]:
    import numpy as np
    import torch
    from tensordict import TensorDict
    from verl import DataProto

    expanded_ids, expanded_messages, texts = [], [], []
    for sample_id, messages in rows:
        rendered = prompt_text(tokenizer, messages, native)
        for _ in range(GROUP_SIZE):
            expanded_ids.append(sample_id)
            expanded_messages.append(messages)
            texts.append(rendered)
    tokenizer.padding_side = "left"
    encoded = tokenizer(texts, add_special_tokens=False, padding=True,
                        return_tensors="pt")
    if encoded["input_ids"].shape[1] > 1024:
        raise ValueError("G0 prompt exceeds max_start_length=1024")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    batch = TensorDict({
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }, batch_size=input_ids.shape[0])
    raw = np.empty(len(expanded_messages), dtype=object)
    raw[:] = expanded_messages
    proto = DataProto(batch=batch, non_tensor_batch={"raw_prompt": raw})
    proto.meta_info.update({
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "do_sample": True,
        "response_length": 500,
        **SAMPLING,
    })
    slots = [slot for _ in rows for slot in range(GROUP_SIZE)]
    return proto, expanded_ids, slots, expanded_messages


def decode_responses(tokenizer: Any, responses: Any) -> list[str]:
    output = []
    for response in responses:
        ids = []
        for token_id in response.tolist():
            token_id = int(token_id)
            if token_id in {tokenizer.eos_token_id, tokenizer.pad_token_id}:
                break
            ids.append(token_id)
        output.append(tokenizer.decode(ids, skip_special_tokens=False,
                                       clean_up_tokenization_spaces=False))
    return output


def prompt_digest(input_ids: Any, attention_mask: Any) -> str:
    ids = input_ids[attention_mask.bool()].detach().cpu().tolist()
    return sha256_bytes(json.dumps(ids, separators=(",", ":")).encode("ascii"))


def set_seed(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def direct_records(actor: Any, batch: Any, tokenizer: Any, sample_ids: list[str],
                   slots: list[int], seed: int) -> list[dict[str, Any]]:
    from search_r1.llm_agent.tool_protocol import QWEN35_NATIVE, parse_action

    set_seed(seed)
    output = actor.generate_sequences(batch)
    texts = decode_responses(tokenizer, output.batch["responses"])
    records = []
    for index, text in enumerate(texts):
        parsed = parse_action(text, QWEN35_NATIVE)
        records.append({
            "sample_id": sample_ids[index],
            "group_slot": slots[index],
            "mode": "direct",
            "prompt_token_sha256": prompt_digest(
                batch.batch["input_ids"][index], batch.batch["attention_mask"][index]),
            "raw_text": text,
            "parsed_action": parsed_record(parsed),
            "generation_events": [{"turn": 0, "text": text}],
            "retrieval_events": [],
            "final_answer": parsed.content if parsed.action == "answer" else None,
        })
    return records


def manager_records(actor: Any, batch: Any, tokenizer: Any, sample_ids: list[str],
                    slots: list[int], raw_messages: list[Any], seed: int,
                    protocol: str, retriever_url: str) -> list[dict[str, Any]]:
    from search_r1.llm_agent.generation import (GenerationConfig,
                                                LLMGenerationManager)
    from search_r1.llm_agent.tool_protocol import parse_action

    config = GenerationConfig(
        max_turns=4,
        max_start_length=1024,
        max_prompt_length=4096,
        max_response_length=500,
        max_obs_length=384,
        num_gpus=1,
        no_think_rl=False,
        search_url=retriever_url,
        topk=3,
        tool_protocol=protocol,
    )
    manager = LLMGenerationManager(tokenizer, actor, config, is_validation=True)
    prompt_hashes = [prompt_digest(batch.batch["input_ids"][index],
                                   batch.batch["attention_mask"][index])
                     for index in range(len(sample_ids))]
    set_seed(seed)
    output = manager.run_llm_loop(
        batch, batch.batch["input_ids"],
        raw_messages=raw_messages if protocol == "qwen35_native" else None)
    generations = output.non_tensor_batch["generation_events"].tolist()
    retrievals = output.non_tensor_batch["retrieval_events"].tolist()
    answers = (output.non_tensor_batch["final_answer"].tolist()
               if "final_answer" in output.non_tensor_batch else [None] * len(sample_ids))
    records = []
    for index, events in enumerate(generations):
        first_text = events[0]["text"] if events else ""
        parsed = parse_action(first_text, protocol)
        records.append({
            "sample_id": sample_ids[index],
            "group_slot": slots[index],
            "mode": "native_manager" if protocol == "qwen35_native" else "legacy_manager",
            "prompt_token_sha256": prompt_hashes[index],
            "raw_text": first_text,
            "parsed_action": parsed_record(parsed),
            "generation_events": events,
            "retrieval_events": retrievals[index],
            "final_answer": answers[index],
        })
    return records


def run_real(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch
    from omegaconf import OmegaConf
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from verl.workers.rollout.hf_rollout import HFRollout

    if not torch.cuda.is_available():
        raise ValueError("the real G0 probe requires CUDA")
    native_rows = load_probe_rows(args.native_data, args.native_manifest,
                                  "probe_g0", EXPECTED_QUESTIONS)
    legacy_rows = load_probe_rows(args.legacy_data, args.legacy_manifest,
                                  "probe", 64)[:EXPECTED_QUESTIONS]
    if [item[0] for item in native_rows] != [item[0] for item in legacy_rows]:
        raise ValueError("native and legacy G0 sample IDs differ")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    device = torch.device("cuda:0")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device)
    rollout_config = OmegaConf.create({
        "do_sample": True,
        "response_length": 500,
        "micro_batch_size": 1,
        **SAMPLING,
    })
    actor = HFRollout(model, rollout_config)
    native_batch, sample_ids, slots, native_messages = make_batch(
        tokenizer, native_rows, True, device)
    records = direct_records(actor, native_batch, tokenizer, sample_ids, slots, args.seed)
    native_batch, _, _, native_messages = make_batch(tokenizer, native_rows, True, device)
    records.extend(manager_records(actor, native_batch, tokenizer, sample_ids, slots,
                                   native_messages, args.seed, "qwen35_native",
                                   args.retriever_url))
    legacy_batch, legacy_ids, legacy_slots, legacy_messages = make_batch(
        tokenizer, legacy_rows, False, device)
    records.extend(manager_records(actor, legacy_batch, tokenizer, legacy_ids,
                                   legacy_slots, legacy_messages, args.seed,
                                   "legacy_xml", args.retriever_url))
    resolved = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "checkpoint_digest": args.checkpoint_digest,
        "model_tree_sha256": sha256_tree(args.model_dir),
        "seed": args.seed,
        "questions": EXPECTED_QUESTIONS,
        "group_size": GROUP_SIZE,
        "max_turns": 4,
        "max_start_length": 1024,
        "max_prompt_length": 4096,
        "max_response_length": 500,
        "max_obs_length": 384,
        "retriever_topk": 3,
        "sampling": SAMPLING,
    }
    return resolved, records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--native-data", type=Path)
    parser.add_argument("--native-manifest", type=Path)
    parser.add_argument("--legacy-data", type=Path)
    parser.add_argument("--legacy-manifest", type=Path)
    parser.add_argument("--retriever-url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--checkpoint-digest", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fixture-input", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if not re.fullmatch(r"[0-9a-f]{64}", args.checkpoint_digest):
            raise ValueError("checkpoint digest must be 64 lowercase hex")
        if args.seed != 42:
            raise ValueError("G0 seed is fixed at 42")
        if args.fixture_input is not None:
            resolved, records = load_fixture(args.fixture_input, args.checkpoint_digest)
        else:
            required = (args.model_dir, args.native_data, args.native_manifest,
                        args.legacy_data, args.legacy_manifest)
            if any(path is None for path in required):
                raise ValueError("real G0 requires model and native/legacy data manifests")
            for path in required:
                if not path.exists() or path.is_symlink():
                    raise ValueError(f"G0 input is missing or symlinked: {path}")
            resolved, records = run_real(args)
        publish(args.output_dir, resolved, records, args.checkpoint_digest)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Qwen native protocol probe error: {error}", file=sys.stderr)
        return 1
    print(f"Wrote Qwen native protocol probe: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
