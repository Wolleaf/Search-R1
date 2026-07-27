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
SCHEMA_VERSION = 4
ACTIVE_DATA_SCHEMA_VERSION = 5
PROMPT_VERSION = "qwen35-native-search-v4-terminal-answer-only"
TERMINAL_PROMPT_VERSION = "qwen35-terminal-answer-v1"
TERMINAL_PROMPT_SHA256 = (
    "afc18b79afaafccece6927aec5ccd7898ef2ae17766bce7ffda244eef388d7f2"
)
EXPECTED_QUESTIONS = 8
GROUP_SIZE = 2
MODES = ("direct", "native_manager")
SAMPLING = {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
}


def validate_active_prompt_contract(payload: Mapping[str, Any]) -> None:
    prompt_contract = payload.get("prompt_contract")
    if (payload.get("schema_version") != ACTIVE_DATA_SCHEMA_VERSION
            or not isinstance(prompt_contract, Mapping)
            or prompt_contract.get("tool_protocol") != "qwen35_native"
            or prompt_contract.get("prompt_version") != PROMPT_VERSION
            or prompt_contract.get("terminal_answer_only") is not True
            or prompt_contract.get("terminal_prompt_version") !=
            TERMINAL_PROMPT_VERSION
            or prompt_contract.get("terminal_prompt_sha256") !=
            TERMINAL_PROMPT_SHA256):
        raise ValueError("active v4 data prompt contract mismatch")


def validate_resolved_contract(resolved: Mapping[str, Any],
                               checkpoint_digest: str) -> None:
    expected = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "checkpoint_digest": checkpoint_digest,
        "prompt_version": PROMPT_VERSION,
        "terminal_answer_only": True,
        "terminal_prompt_version": TERMINAL_PROMPT_VERSION,
        "terminal_prompt_sha256": TERMINAL_PROMPT_SHA256,
        "max_action_budget": 4,
        "max_obs_length": 500,
        "sampling": SAMPLING,
    }
    if any(resolved.get(name) != value for name, value in expected.items()):
        raise ValueError("active v4 resolved protocol contract mismatch")


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


def _valid_token_ids(value: Any) -> bool:
    return (isinstance(value, list)
            and all(isinstance(item, int) and not isinstance(item, bool)
                    and item >= 0 for item in value))


def thinking_record(text: str, context: str = "initial_question") -> dict[str, Any]:
    from search_r1.llm_agent.tool_protocol import (
        QWEN35_REASONING_CONTINUATION,
        locate_qwen35_action_boundary,
    )

    located = locate_qwen35_action_boundary(
        text, QWEN35_REASONING_CONTINUATION)
    closing_before_action = (located.error is None
                             and located.action_start is not None)
    reasoning = located.reasoning if closing_before_action else ""
    return {
        "context": context,
        "template_opening_provided": True,
        "nonempty_reasoning": bool(reasoning.strip()),
        "closing_before_action": closing_before_action,
    }


def validate_record(record: Mapping[str, Any]) -> None:
    expected = {
        "sample_id", "group_slot", "mode", "prompt_token_sha256", "raw_text",
        "raw_token_ids", "action_text", "action_token_ids", "action_boundary",
        "tail_dropped", "parsed_action", "thinking", "generation_events",
        "retrieval_events", "final_answer",
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
    if not isinstance(record["action_text"], str) \
            or not _valid_token_ids(record["raw_token_ids"]) \
            or not _valid_token_ids(record["action_token_ids"]):
        raise ValueError("protocol probe raw/action token evidence is invalid")
    if record["raw_token_ids"][:len(record["action_token_ids"])] \
            != record["action_token_ids"]:
        raise ValueError("protocol probe action tokens are not a raw-token prefix")
    if record["tail_dropped"] != (
            len(record["action_token_ids"]) < len(record["raw_token_ids"])):
        raise ValueError("protocol probe tail flag is not aligned with token IDs")
    if not isinstance(record["action_boundary"], str) \
            or not isinstance(record["thinking"], Mapping):
        raise ValueError("protocol probe action/thinking evidence is invalid")
    action = record["parsed_action"]
    if not isinstance(action, Mapping) or set(action) != {
            "action", "content", "error", "prefix", "valid"}:
        raise ValueError("protocol probe parsed action is invalid")
    if not isinstance(record["generation_events"], list) \
            or not isinstance(record["retrieval_events"], list):
        raise ValueError("protocol probe events must be lists")


def validate_environment_replay(replay: Mapping[str, Any]) -> None:
    required = {
        "sample_id", "query", "action_text", "requested_search_count",
        "executed_search_count", "retrieval_event_count",
        "nonempty_tool_response_count", "retrieved_document_count",
        "visible_tool_response", "tool_role_rendered", "policy_token_count",
        "tool_response_token_count", "tool_response_policy_token_count",
        "info_mask_consistent", "scientific_metric",
    }
    if set(replay) != required:
        raise ValueError("E0 environment replay has missing or unknown fields")
    count_fields = required & {
        "requested_search_count", "executed_search_count",
        "retrieval_event_count", "nonempty_tool_response_count",
        "retrieved_document_count", "policy_token_count",
        "tool_response_token_count", "tool_response_policy_token_count",
    }
    if any(isinstance(replay[name], bool) or not isinstance(replay[name], int)
           or replay[name] < 0 for name in count_fields):
        raise ValueError("E0 environment replay counts are invalid")
    if not all(isinstance(replay[name], str)
               for name in ("sample_id", "query", "action_text",
                            "visible_tool_response")):
        raise ValueError("E0 environment replay text fields are invalid")
    if not isinstance(replay["tool_role_rendered"], bool) \
            or not isinstance(replay["info_mask_consistent"], bool):
        raise ValueError("E0 environment replay flags are invalid")
    if replay["scientific_metric"] is not False:
        raise ValueError("E0 must be marked as non-scientific")


def publish(output_dir: Path, resolved: Mapping[str, Any],
            records: list[dict[str, Any]], environment_replay: Mapping[str, Any],
            checkpoint_digest: str) -> None:
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError(f"refusing to overwrite protocol probe output: {output_dir}")
    validate_resolved_contract(resolved, checkpoint_digest)
    if len(records) != EXPECTED_QUESTIONS * GROUP_SIZE * len(MODES):
        raise ValueError("protocol probe must contain exactly 32 records")
    validate_environment_replay(environment_replay)
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
    atomic_write(output_dir / "environment-replay.json",
                 canonical_bytes(dict(environment_replay)))
    manifest = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "checkpoint_digest": checkpoint_digest,
        "mode_counts": mode_counts,
        "records_sha256": sha256_bytes(records_bytes),
        "resolved_config_sha256": sha256_file(output_dir / "resolved-config.json"),
        "environment_replay_sha256": sha256_file(
            output_dir / "environment-replay.json"),
    }
    atomic_write(output_dir / "manifest.json", canonical_bytes(manifest))


def load_fixture(path: Path, checkpoint_digest: str) -> tuple[
        dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_bytes())
    if not isinstance(payload, Mapping) or set(payload) != {
            "resolved_config", "records", "environment_replay"}:
        raise ValueError("fixture must contain resolved_config, records, and E0")
    resolved = payload["resolved_config"]
    records = payload["records"]
    if not isinstance(resolved, dict) or not isinstance(records, list):
        raise ValueError("fixture resolved_config/records types are invalid")
    validate_resolved_contract(resolved, checkpoint_digest)
    replay = payload["environment_replay"]
    if not isinstance(replay, dict):
        raise ValueError("fixture E0 environment replay is invalid")
    return resolved, records, replay


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
    if not isinstance(manifest, Mapping):
        raise ValueError("native data manifest must be an object")
    validate_active_prompt_contract(manifest)
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
        if native and not rendered.endswith("<think>\n"):
            raise ValueError("native generation prefix does not open thinking")
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
    from search_r1.llm_agent.generation import slice_first_complete_native_action
    from search_r1.llm_agent.tool_protocol import (
        QWEN35_NATIVE,
        QWEN35_REASONING_CONTINUATION,
        parse_action,
    )

    set_seed(seed)
    output = actor.generate_sequences(batch)
    records = []
    for index, response in enumerate(output.batch["responses"]):
        sliced = slice_first_complete_native_action(tokenizer, response)
        parsed = parse_action(
            sliced.action_text,
            QWEN35_NATIVE,
            qwen35_reasoning_mode=QWEN35_REASONING_CONTINUATION,
        )
        records.append({
            "sample_id": sample_ids[index],
            "group_slot": slots[index],
            "mode": "direct",
            "prompt_token_sha256": prompt_digest(
                batch.batch["input_ids"][index], batch.batch["attention_mask"][index]),
            "raw_text": sliced.raw_text,
            "raw_token_ids": list(sliced.raw_token_ids),
            "action_text": sliced.action_text,
            "action_token_ids": list(sliced.action_token_ids),
            "action_boundary": sliced.boundary,
            "tail_dropped": sliced.tail_dropped,
            "parsed_action": parsed_record(parsed),
            "thinking": thinking_record(sliced.action_text),
            "generation_events": [{
                "turn": 0,
                "text": sliced.action_text,
                "raw_text": sliced.raw_text,
                "raw_token_ids": list(sliced.raw_token_ids),
                "action_token_ids": list(sliced.action_token_ids),
                "tail_dropped": sliced.tail_dropped,
                "generation_context": "initial_question",
            }],
            "retrieval_events": [],
            "final_answer": parsed.content if parsed.action == "answer" else None,
        })
    return records


def manager_records(actor: Any, batch: Any, tokenizer: Any, sample_ids: list[str],
                    slots: list[int], raw_messages: list[Any], seed: int,
                    protocol: str, retriever_url: str) -> list[dict[str, Any]]:
    from search_r1.llm_agent.generation import (GenerationConfig,
                                                LLMGenerationManager)
    from search_r1.llm_agent.tool_protocol import (
        QWEN35_NATIVE,
        QWEN35_REASONING_CONTINUATION,
        parse_action,
    )

    config = GenerationConfig(
        max_turns=4,
        max_start_length=1024,
        max_prompt_length=4500,
        max_response_length=500,
        max_obs_length=500,
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
        if not events:
            raise ValueError("native manager produced no generation event")
        first = events[0]
        first_text = first["text"]
        raw_text = first.get("raw_text", first_text)
        raw_ids = first.get("raw_token_ids")
        action_ids = first.get("action_token_ids")
        if not _valid_token_ids(raw_ids) or not _valid_token_ids(action_ids):
            raise ValueError("native manager omitted raw/action token evidence")
        reasoning_mode = (QWEN35_REASONING_CONTINUATION
                          if protocol == QWEN35_NATIVE else None)
        parsed = parse_action(
            first_text,
            protocol,
            qwen35_reasoning_mode=reasoning_mode,
        )
        records.append({
            "sample_id": sample_ids[index],
            "group_slot": slots[index],
            "mode": "native_manager" if protocol == "qwen35_native" else "legacy_manager",
            "prompt_token_sha256": prompt_hashes[index],
            "raw_text": raw_text,
            "raw_token_ids": list(raw_ids),
            "action_text": first_text,
            "action_token_ids": list(action_ids),
            "action_boundary": str(first.get("boundary", "unknown")),
            "tail_dropped": bool(first.get("tail_dropped", False)),
            "parsed_action": parsed_record(parsed),
            "thinking": thinking_record(first_text),
            "generation_events": events,
            "retrieval_events": retrievals[index],
            "final_answer": answers[index],
        })
    return records


def run_environment_replay(tokenizer: Any, batch: Any, raw_messages: list[Any],
                           sample_id: str, retriever_url: str) -> dict[str, Any]:
    """Inject one fixed native action through the real manager environment."""
    import torch
    from verl import DataProto

    from search_r1.llm_agent.generation import (GenerationConfig,
                                                LLMGenerationManager)
    from search_r1.llm_agent.tool_protocol import (
        QWEN35_NATIVE,
        QWEN35_REASONING_CONTINUATION,
        parse_action,
    )

    config = GenerationConfig(
        max_turns=4,
        max_start_length=1024,
        max_prompt_length=4500,
        max_response_length=500,
        max_obs_length=500,
        num_gpus=1,
        no_think_rl=False,
        search_url=retriever_url,
        topk=3,
        tool_protocol=QWEN35_NATIVE,
    )
    manager = LLMGenerationManager(tokenizer, None, config, is_validation=True)
    one = DataProto(
        batch=batch.batch[:1],
        non_tensor_batch={name: values[:1]
                          for name, values in batch.non_tensor_batch.items()},
        meta_info=batch.meta_info.copy(),
    )
    conversations = manager._prepare_native_conversations(one, raw_messages[:1])
    action_text = (
        "Use the external evidence.</think>\n"
        "<tool_call>\n<function=search>\n<parameter=query>\n"
        "Barack Obama\n</parameter>\n</function>\n</tool_call>"
    )
    parsed = parse_action(
        action_text,
        QWEN35_NATIVE,
        qwen35_reasoning_mode=QWEN35_REASONING_CONTINUATION,
    )
    if not parsed.valid or parsed.action != "search":
        raise ValueError("E0 fixed native search action is not parseable")
    observations, dones, valid, executed = manager.execute_predictions(
        [action_text], tokenizer.pad_token, active_mask=[True], do_search=True)
    retrievals = manager._last_execution_retrieval_events
    action_ids = tokenizer(action_text, add_special_tokens=False,
                           return_tensors="pt")["input_ids"].to(
                               batch.batch["input_ids"].device)
    suffix_ids, visible = manager._process_native_followups(
        conversations, action_ids, [action_text], [parsed], observations,
        torch.tensor([True], device=action_ids.device), device=action_ids.device)
    empty = action_ids[:, :0]
    right = manager._update_right_side(
        {"responses": empty, "responses_with_info_mask": empty},
        action_ids, suffix_ids)
    attention = manager.tensor_fn.create_attention_mask(right["responses"])
    policy = manager.tensor_fn.create_attention_mask(
        right["responses_with_info_mask"])
    policy_count = int(policy.sum().item())
    total_count = int(attention.sum().item())
    docs = retrievals[0]["documents"] if retrievals and retrievals[0] else []
    return {
        "sample_id": sample_id,
        "query": parsed.content,
        "action_text": action_text,
        "requested_search_count": 1,
        "executed_search_count": int(executed[0]),
        "retrieval_event_count": len([item for item in retrievals if item]),
        "nonempty_tool_response_count": int(bool(visible[0].strip())),
        "retrieved_document_count": len(docs),
        "visible_tool_response": visible[0],
        "tool_role_rendered": bool(conversations[0].messages
                                   and conversations[0].messages[-1].get("role") == "tool"),
        "policy_token_count": policy_count,
        "tool_response_token_count": total_count - len(action_ids[0]),
        "tool_response_policy_token_count": max(0, policy_count - len(action_ids[0])),
        "info_mask_consistent": bool(
            valid == [1] and dones == [0]
            and policy_count == len(action_ids[0])
            and total_count > policy_count),
        "scientific_metric": False,
    }


def run_real(args: argparse.Namespace) -> tuple[
        dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    import torch
    from omegaconf import OmegaConf
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from verl.workers.rollout.hf_rollout import HFRollout

    if not torch.cuda.is_available():
        raise ValueError("the real G0 probe requires CUDA")
    native_rows = load_probe_rows(args.native_data, args.native_manifest,
                                  "probe_g0", EXPECTED_QUESTIONS)
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
    replay_batch, _, _, replay_messages = make_batch(
        tokenizer, native_rows, True, device)
    environment_replay = run_environment_replay(
        tokenizer, replay_batch, replay_messages, native_rows[0][0],
        args.retriever_url)
    resolved = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "checkpoint_digest": args.checkpoint_digest,
        "prompt_version": PROMPT_VERSION,
        "terminal_answer_only": True,
        "terminal_prompt_version": TERMINAL_PROMPT_VERSION,
        "terminal_prompt_sha256": TERMINAL_PROMPT_SHA256,
        "model_tree_sha256": sha256_tree(args.model_dir),
        "seed": args.seed,
        "questions": EXPECTED_QUESTIONS,
        "group_size": GROUP_SIZE,
        "max_action_budget": 4,
        "max_start_length": 1024,
        "max_prompt_length": 4500,
        "max_response_length": 500,
        "max_obs_length": 500,
        "retriever_topk": 3,
        "sampling": SAMPLING,
    }
    return resolved, records, environment_replay


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--native-data", type=Path)
    parser.add_argument("--native-manifest", type=Path)
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
            resolved, records, environment_replay = load_fixture(
                args.fixture_input, args.checkpoint_digest)
        else:
            required = (args.model_dir, args.native_data, args.native_manifest)
            if any(path is None for path in required):
                raise ValueError("real G0 requires model and native data manifest")
            for path in required:
                if not path.exists() or path.is_symlink():
                    raise ValueError(f"G0 input is missing or symlinked: {path}")
            resolved, records, environment_replay = run_real(args)
        publish(args.output_dir, resolved, records, environment_replay,
                args.checkpoint_digest)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Qwen native protocol probe error: {error}", file=sys.stderr)
        return 1
    print(f"Wrote Qwen native protocol probe: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
