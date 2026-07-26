#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
NATIVE_TRAIN_PROJECT_ROOT="${AUTODL_ROOT:-/root/autodl-tmp/search-r1}"
NATIVE_TRAIN_DATA_DIR="$NATIVE_TRAIN_PROJECT_ROOT/data/search_mix_qwen35_native_v3"
NATIVE_TRAIN_STAGE="${QWEN_NATIVE_TRAIN_STAGE:-}"
NATIVE_PROTOCOL_GATE_EVIDENCE="${QWEN_NATIVE_PROTOCOL_GATE_EVIDENCE:-}"
NATIVE_SMOKE_EVIDENCE="${QWEN_NATIVE_SMOKE_EVIDENCE:-}"

export AUTODL_RUN_BUDGET_PROFILE=gated_followup
export TOOL_PROTOCOL=qwen35_native
export DATA_DIR="$NATIVE_TRAIN_DATA_DIR"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-500}"
export EVAL_DATA_FILE=''
export EVAL_EXPECTED_ROWS=128
export EVAL_GROUP_SIZE=1
AUTODL_GPU_PIPELINE=qwen_native_train
# Reuse sealed GPU admission, BM25 lifecycle, immutable run attempts, and lineage helpers.
# shellcheck source=03_gpu_run.sh
source "$SCRIPT_DIR/03_gpu_run.sh"

readonly NATIVE_TRAIN_RESULTS_ROOT="$RUNS_ROOT/qwen-native-training"
readonly NATIVE_TRAIN_MANIFEST="$NATIVE_TRAIN_DATA_DIR/manifest.json"
readonly NATIVE_TRAIN_CATALOG="$NATIVE_TRAIN_DATA_DIR/catalog.jsonl"
readonly NATIVE_TRAIN_G3_DATA="$NATIVE_TRAIN_DATA_DIR/probe_multi_64.parquet"
readonly NATIVE_TRAIN_SOURCE_DIR="$PROJECT_ROOT/data/search_mix"
readonly NATIVE_TRAIN_PROMPT_VERSION=qwen35-native-search-v3-original-aligned
readonly NATIVE_TRAIN_SMOKE_CONTRACT=qwen-native-training-smoke-v3
readonly NATIVE_TRAIN_MAIN_CONTRACT=qwen-native-training-main-v3
readonly NATIVE_TRAIN_GATE_CONTRACT=qwen-native-gate-v3
readonly NATIVE_TRAIN_GATE_CLI="$CHECKOUT_DIR/scripts/autodl/qwen_native_gate_analysis.py"
readonly NATIVE_TRAIN_PAIRED_CLI="$CHECKOUT_DIR/scripts/autodl/paired_eval.py"
readonly NATIVE_TRAIN_SMOKE_CLI="$CHECKOUT_DIR/scripts/autodl/qwen_native_smoke_analysis.py"

NATIVE_TRAIN_PREFLIGHT_STAGE=''
NATIVE_TRAIN_PREFLIGHT_COMMIT=''
NATIVE_TRAIN_PREFLIGHT_HANDOFF=''
NATIVE_TRAIN_PREFLIGHT_BASE_DIGEST=''
NATIVE_TRAIN_PREFLIGHT_DATA_DIGEST=''
NATIVE_TRAIN_PREFLIGHT_PROTOCOL_GATE_DIGEST=''
NATIVE_TRAIN_PREFLIGHT_SMOKE_DIGEST='-'
NATIVE_TRAIN_PROTOCOL_GATE_DIGEST=''
NATIVE_TRAIN_PROTOCOL_GATE_RESULTS=''
NATIVE_TRAIN_PROTOCOL_GATE_OUTER=''
NATIVE_TRAIN_PROTOCOL_GATE_MANIFEST=''
NATIVE_TRAIN_SMOKE_DIGEST=''
NATIVE_TRAIN_SMOKE_RESULTS=''
NATIVE_TRAIN_SMOKE_OUTER=''
NATIVE_TRAIN_SMOKE_MANIFEST=''
NATIVE_TRAIN_SMOKE_CHECKPOINT=''
NATIVE_TRAIN_SMOKE_CHECKPOINT_DIGEST=''
VERIFIED_ATTEMPT_NAME=''
VERIFIED_ATTEMPT_RESULTS=''
VERIFIED_ATTEMPT_OUTER=''
VERIFIED_ATTEMPT_MANIFEST=''
VERIFIED_ATTEMPT_DIGEST=''

require_qwen_native_train() {
    validate_gpu_inputs
    [[ "$NATIVE_TRAIN_STAGE" == smoke || "$NATIVE_TRAIN_STAGE" == main ]] || {
        printf 'Set QWEN_NATIVE_TRAIN_STAGE to smoke or main.\n' >&2
        return 64
    }
    [[ "$GPU_COUNT" == 2 && "$TRAIN_BATCH_SIZE" == 8 &&
        "$MAX_RESPONSE_LENGTH" == 500 && "$TOOL_PROTOCOL" == qwen35_native &&
        "$DATA_DIR" == "$NATIVE_TRAIN_DATA_DIR" &&
        "$EVAL_GROUP_SIZE" == 1 && -z "$EVAL_DATA_FILE" &&
        "$RUN_BUDGET_PROFILE" == gated_followup &&
        -n "$NATIVE_PROTOCOL_GATE_EVIDENCE" ]] || {
        printf 'Qwen native v3 training requires the exact structural G0/G1 gate, two GPUs, group 5, batch 8, response/observation 500, and total action budget 4.\n' >&2
        return 64
    }
    if [[ "$NATIVE_TRAIN_STAGE" == main && -z "$NATIVE_SMOKE_EVIDENCE" ]]; then
        printf 'Main training requires QWEN_NATIVE_SMOKE_EVIDENCE.\n' >&2
        return 64
    fi
}

native_train_project_relative_file() {
    local path="$1" canonical root
    [[ -f "$path" && ! -L "$path" ]] || {
        printf 'Native training evidence file is missing or symlinked: %s\n' "$path" >&2
        return 1
    }
    canonical="$(readlink -f -- "$path")" || return 1
    root="$(readlink -f -- "$PROJECT_ROOT")" || return 1
    [[ "$canonical" == "$root/"* ]] || {
        printf 'Native training evidence escapes the project root: %s\n' "$path" >&2
        return 1
    }
    printf '%s\n' "${canonical#"$root/"}"
}

verify_native_training_checksum_manifest() {
    local manifest="$1" line relative path actual_relative count=0
    local pattern='^[0-9a-f]{64}  (.+)$'
    local -A seen=()
    [[ -f "$manifest" && ! -L "$manifest" ]] || return 1
    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ "$line" =~ $pattern ]] || return 1
        relative="${BASH_REMATCH[1]}"
        [[ "$relative" != /* && "$relative" != *'/../'* &&
            -z "${seen[$relative]+present}" ]] || return 1
        seen["$relative"]=1
        path="$PROJECT_ROOT/$relative"
        actual_relative="$(native_train_project_relative_file "$path")" || return 1
        [[ "$actual_relative" == "$relative" ]] || return 1
        ((count += 1))
    done <"$manifest"
    ((count > 0)) || return 1
    (
        cd "$PROJECT_ROOT"
        sha256sum --strict --check "${manifest#"$PROJECT_ROOT/"}" >/dev/null
    )
}

verify_complete_training_attempt() {
    local marker="$1" namespace="$2" results_root="$3" contract="$4"
    local marker_dir attempt_name canonical_marker results outer evidence
    local marker_digest path
    marker_dir="$MANIFEST_DIR/$namespace"
    attempt_name="$(basename -- "$marker" .ok)"
    [[ "$(dirname -- "$marker")" == "$marker_dir" &&
        "$(basename -- "$marker")" == "$attempt_name.ok" &&
        "$attempt_name" =~ ^[0-9]{8}T[0-9]{6}Z-[0-9]+-[0-9]+$ ]] || {
        printf 'Evidence must name one exact %s marker.\n' "$namespace" >&2
        return 1
    }
    [[ -d "$marker_dir" && ! -L "$marker_dir" &&
        -f "$marker" && ! -L "$marker" ]] || {
        printf 'Evidence marker is missing or symlinked: %s\n' "$marker" >&2
        return 1
    }
    canonical_marker="$(readlink -f -- "$marker")" || return 1
    [[ "$canonical_marker" == "$(readlink -f -- "$marker_dir")/$attempt_name.ok" ]] || return 1
    results="$results_root/$attempt_name"
    outer="$ATTEMPTS_ROOT/gpu/$attempt_name"
    evidence="$results/evidence.sha256"
    [[ -d "$results" && ! -L "$results" && -d "$outer" && ! -L "$outer" &&
        -f "$evidence" && ! -L "$evidence" &&
        -f "$outer/terminal" && ! -L "$outer/terminal" &&
        -f "$outer/exit-code" && ! -L "$outer/exit-code" &&
        -f "$outer/.success" && ! -L "$outer/.success" &&
        "$(tr -d '\r\n' <"$outer/terminal")" == success &&
        "$(tr -d '\r\n' <"$outer/exit-code")" == 0 &&
        ! -e "$outer/.failed" && ! -L "$outer/.failed" &&
        ! -e "$outer/.starting" && ! -L "$outer/.starting" &&
        ! -e "$outer/.running" && ! -L "$outer/.running" ]] || {
        printf 'Evidence outer attempt is not a complete success: %s\n' "$outer" >&2
        return 1
    }
    marker_digest="$(tr -d '\r\n' <"$marker")"
    [[ "$marker_digest" =~ ^[0-9a-f]{64}$ &&
        "$(file_sha256 "$evidence")" == "$marker_digest" ]] || {
        printf 'Evidence marker does not bind its checksum manifest: %s\n' "$marker" >&2
        return 1
    }
    verify_native_training_checksum_manifest "$evidence" || {
        printf 'Evidence checksum validation failed: %s\n' "$evidence" >&2
        return 1
    }
    for path in result-contract result-root evidence-marker evidence-digest; do
        [[ -f "$outer/$path" && ! -L "$outer/$path" ]] || return 1
    done
    [[ "$(tr -d '\r\n' <"$outer/result-contract")" == "$contract" &&
        "$(tr -d '\r\n' <"$outer/result-root")" == "$results" &&
        "$(tr -d '\r\n' <"$outer/evidence-marker")" == "$marker" &&
        "$(tr -d '\r\n' <"$outer/evidence-digest")" == "$marker_digest" ]] || {
        printf 'Evidence attempt binding is inconsistent: %s\n' "$outer" >&2
        return 1
    }
    VERIFIED_ATTEMPT_NAME="$attempt_name"
    VERIFIED_ATTEMPT_RESULTS="$results"
    VERIFIED_ATTEMPT_OUTER="$outer"
    VERIFIED_ATTEMPT_MANIFEST="$evidence"
    VERIFIED_ATTEMPT_DIGEST="$marker_digest"
}

verify_protocol_gate_evidence() {
    local marker="$1" expected_checkpoint="$2" expected_digest="$3"
    local expected_commit="$4" expected_handoff="$5" expected_data="$6"
    verify_complete_training_attempt "$marker" qwen-native-gate \
        "$RUNS_ROOT/qwen-native-gate/attempts" "$NATIVE_TRAIN_GATE_CONTRACT" || return $?
    "$TRAIN_ENV/bin/python" - \
        "$VERIFIED_ATTEMPT_RESULTS" "$expected_checkpoint" "$expected_digest" \
        "$expected_commit" "$expected_handoff" "$expected_data" <<'PY'
import csv
import json
from pathlib import Path
import sys

results = Path(sys.argv[1])
expected = sys.argv[2:]
for name in ("stage.txt", "go_no_go.json", "summary.json", "lineage.tsv"):
    path = results / name
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"structural protocol-gate evidence is missing or symlinked: {path}")
if (results / "stage.txt").read_text(encoding="utf-8").strip() != "g0_g1":
    raise SystemExit("training predecessor is not the structural G0/G1 gate")
decision = json.loads((results / "go_no_go.json").read_bytes())
summary = json.loads((results / "summary.json").read_bytes())
if (decision.get("schema") != "search-r1.qwen-native-gate" or
        decision.get("schema_version") != 3 or
        decision.get("stage") != "g0_g1" or
        decision.get("decision") != "GO"):
    raise SystemExit("structural G0/G1 decision is not a v3 GO")
criteria = decision.get("criteria")
required_criteria = {
    "g0_prompt_token_match_count",
    "g0_first_action_token_match_count",
    "g0_direct_action_prefix_integrity_count",
    "g0_manager_action_prefix_integrity_count",
    "g0_e0_search_roundtrip_count",
    "g0_e0_retrieved_document_count",
    "g0_e0_tool_role_count",
    "g0_e0_mask_leak_count",
    "g1_action_token_prefix_integrity_count",
    "g1_action_tail_leak_count",
    "g1_info_mask_consistent_count",
    "g1_observation_policy_token_count",
    "g1_retrieval_alignment_error_count",
}
if (not isinstance(criteria, dict) or set(criteria) != required_criteria or
        any(not isinstance(item, dict) or item.get("passed") is not True
            for item in criteria.values())):
    raise SystemExit("structural G0/G1 criteria are incomplete or failed")
if (summary.get("schema") != "search-r1.qwen-native-gate" or
        summary.get("schema_version") != 3 or
        summary.get("stage") != "g0_g1" or summary.get("decision") != "GO"):
    raise SystemExit("structural G0/G1 summary is not a v3 GO")
with (results / "lineage.tsv").open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
if len(rows) != 1:
    raise SystemExit("structural G0/G1 lineage must contain exactly one row")
row = rows[0]
checks = {
    "stage": row.get("stage") == "g0_g1",
    "checkpoint": row.get("checkpoint") == expected[0],
    "checkpoint_digest": row.get("checkpoint_digest") == expected[1],
    "commit": row.get("evaluation_checkout_commit") == expected[2],
    "handoff": row.get("cpu_handoff_digest") == expected[3],
    "data": row.get("data_manifest_sha256") == expected[4],
}
failed = sorted(key for key, passed in checks.items() if not passed)
if failed:
    raise SystemExit("structural G0/G1 lineage mismatch: " + ", ".join(failed))
PY
    NATIVE_TRAIN_PROTOCOL_GATE_DIGEST="$VERIFIED_ATTEMPT_DIGEST"
    NATIVE_TRAIN_PROTOCOL_GATE_RESULTS="$VERIFIED_ATTEMPT_RESULTS"
    NATIVE_TRAIN_PROTOCOL_GATE_OUTER="$VERIFIED_ATTEMPT_OUTER"
    NATIVE_TRAIN_PROTOCOL_GATE_MANIFEST="$VERIFIED_ATTEMPT_MANIFEST"
}

verify_smoke_evidence() {
    local marker="$1" expected_base="$2" expected_base_digest="$3"
    local expected_commit="$4" expected_handoff="$5" expected_data="$6"
    local expected_gate_digest="$7" metadata checkpoint checkpoint_digest
    local base base_digest commit handoff data gate_marker gate_digest
    verify_complete_training_attempt "$marker" qwen-native-training-smoke \
        "$NATIVE_TRAIN_RESULTS_ROOT/attempts" "$NATIVE_TRAIN_SMOKE_CONTRACT" || return $?
    metadata="$("$TRAIN_ENV/bin/python" - \
        "$VERIFIED_ATTEMPT_RESULTS" "$NATIVE_TRAIN_CATALOG" <<'PY'
import csv
import hashlib
import json
from pathlib import Path
import sys

results = Path(sys.argv[1])
catalog = Path(sys.argv[2])
for name in ("contract.env", "lineage.tsv", "run-index.tsv", "storage.env",
             "checkpoint-tree.env", "smoke-decision.json"):
    path = results / name
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"smoke evidence is missing or symlinked: {path}")

def read_env(path):
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or "=" not in line:
            raise SystemExit(f"malformed env evidence: {path}")
        key, value = line.split("=", 1)
        if key in values:
            raise SystemExit(f"duplicate env evidence key: {key}")
        values[key] = value
    return values

contract = read_env(results / "contract.env")
storage = read_env(results / "storage.env")
checkpoint_tree = read_env(results / "checkpoint-tree.env")
if contract.get("schema") != "qwen-native-training-smoke-v3" or contract.get("stage") != "smoke":
    raise SystemExit("smoke contract identity mismatch")
decision = json.loads((results / "smoke-decision.json").read_bytes())
if (contract.get("decision") != "GO" or
        decision.get("schema") != "search-r1.qwen-native-smoke-decision" or
        decision.get("schema_version") != 1 or decision.get("decision") != "GO"):
    raise SystemExit("smoke decision is not GO")
for key in ("checkpoint_bytes", "filesystem_available_bytes"):
    try:
        value = int(storage[key])
    except (KeyError, ValueError):
        raise SystemExit(f"invalid smoke storage value: {key}")
    if value < 0:
        raise SystemExit(f"negative smoke storage value: {key}")
with (results / "lineage.tsv").open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    rows = list(reader)
if len(rows) != 1 or rows[0].get("stage") != "S" or rows[0].get("role") != "smoke":
    raise SystemExit("smoke lineage must contain exactly one S row")
row = rows[0]
if not row.get("run_dir"):
    raise SystemExit("smoke lineage has no run directory")
run_dir = Path(row["run_dir"])
keys = (
    "checkpoint", "checkpoint_digest", "parent_checkpoint",
    "parent_checkpoint_digest", "checkout_commit", "cpu_handoff_digest",
    "data_manifest_sha256", "protocol_gate_evidence",
    "protocol_gate_evidence_sha256",
)
if any(not row.get(key) for key in keys):
    raise SystemExit("smoke lineage is incomplete")
if (checkpoint_tree.get("checkpoint") != row["checkpoint"] or
        checkpoint_tree.get("checkpoint_tree_sha256") != row["checkpoint_digest"]):
    raise SystemExit("smoke checkpoint tree evidence disagrees with lineage")

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

inputs = decision.get("inputs", {})
if (inputs.get("trace_sha256") != sha256_file(
        run_dir / "traces" / "train_trajectories.jsonl") or
        inputs.get("log_sha256") != sha256_file(run_dir / "train.log") or
        inputs.get("catalog_sha256") != sha256_file(catalog) or
        not isinstance(inputs.get("wandb_tree_sha256"), str) or
        len(inputs["wandb_tree_sha256"]) != 64):
    raise SystemExit("smoke decision input digests are inconsistent")
print(*(row[key] for key in keys), sep="\t")
PY
)" || return 1
    IFS=$'\t' read -r checkpoint checkpoint_digest base base_digest commit handoff \
        data gate_marker gate_digest <<<"$metadata"
    [[ "$base" == "$expected_base" && "$base_digest" == "$expected_base_digest" &&
        "$commit" == "$expected_commit" && "$handoff" == "$expected_handoff" &&
        "$data" == "$expected_data" &&
        "$gate_marker" == "$NATIVE_PROTOCOL_GATE_EVIDENCE" &&
        "$gate_digest" == "$expected_gate_digest" ]] || {
        printf 'Smoke evidence does not bind the selected base, commit, handoff, data, and structural protocol gate.\n' >&2
        return 1
    }
    [[ "$checkpoint_digest" =~ ^[0-9a-f]{64}$ &&
        -d "$checkpoint" && ! -L "$checkpoint" &&
        "$(readlink -f -- "$checkpoint")" == \
            "$(readlink -f -- "$RUNS_ROOT")/smoke/attempts/"*'/checkpoints/actor/global_step_2' ]] || {
        printf 'Smoke checkpoint path is not the fixed two-step endpoint.\n' >&2
        return 1
    }
    verify_native_checkpoint_digest "$checkpoint" "$checkpoint_digest" || return $?
    NATIVE_TRAIN_SMOKE_DIGEST="$VERIFIED_ATTEMPT_DIGEST"
    NATIVE_TRAIN_SMOKE_RESULTS="$VERIFIED_ATTEMPT_RESULTS"
    NATIVE_TRAIN_SMOKE_OUTER="$VERIFIED_ATTEMPT_OUTER"
    NATIVE_TRAIN_SMOKE_MANIFEST="$VERIFIED_ATTEMPT_MANIFEST"
    NATIVE_TRAIN_SMOKE_CHECKPOINT="$checkpoint"
    NATIVE_TRAIN_SMOKE_CHECKPOINT_DIGEST="$checkpoint_digest"
}

verify_qwen_native_train_data() {
    local path
    local -a required=(
        "$NATIVE_TRAIN_MANIFEST"
        "$NATIVE_TRAIN_MANIFEST.sha256"
        "$NATIVE_TRAIN_CATALOG"
        "$NATIVE_TRAIN_DATA_DIR/train_512.parquet"
        "$NATIVE_TRAIN_DATA_DIR/val_128.parquet"
        "$NATIVE_TRAIN_DATA_DIR/nq_test_128_native_v3.parquet"
        "$NATIVE_TRAIN_DATA_DIR/multihop_eval_256_native_v3.parquet"
        "$NATIVE_TRAIN_G3_DATA"
        "$NATIVE_TRAIN_SOURCE_DIR/manifest.json"
        "$NATIVE_TRAIN_SOURCE_DIR/retrieval_replay.json"
        "$NATIVE_TRAIN_SOURCE_DIR/retrieval_replay.json.sha256"
        "$NATIVE_TRAIN_GATE_CLI"
        "$NATIVE_TRAIN_PAIRED_CLI"
        "$NATIVE_TRAIN_SMOKE_CLI"
    )
    for path in "${required[@]}"; do
        [[ -f "$path" && ! -L "$path" ]] || {
            printf 'Qwen native training data evidence is missing or symlinked: %s\n' "$path" >&2
            return 1
        }
    done
    "$TRAIN_ENV/bin/python" "$CHECKOUT_DIR/scripts/data_process/search_mix.py" verify \
        --manifest "$NATIVE_TRAIN_MANIFEST" \
        --source-manifest "$NATIVE_TRAIN_SOURCE_DIR/manifest.json" \
        --model-dir "$MODEL_DIR" \
        --eval-catalog "$PROJECT_ROOT/data/search_opportunity_gate/catalog.jsonl" \
        --eval-parquet "$PROJECT_ROOT/data/nq_small/test_128.parquet" \
        --expected-tool-protocol qwen35_native
    "$TRAIN_ENV/bin/python" - "$HANDOFF" "$NATIVE_TRAIN_MANIFEST" <<'PY'
import json
from pathlib import Path
import sys

handoff = json.loads(Path(sys.argv[1]).read_bytes())
manifest = json.loads(Path(sys.argv[2]).read_bytes())
expected_handoff = {
    "schema": 3,
    "native_prompt_version": "qwen35-native-search-v3-original-aligned",
    "native_thinking_enabled": True,
    "max_action_budget": 4,
    "selection_observation_length": 384,
    "rollout_observation_length": 500,
}
failed = sorted(key for key, value in expected_handoff.items()
                if handoff.get(key) != value)
if failed:
    raise SystemExit("native v3 CPU handoff mismatch: " + ", ".join(failed))
tokenizer = manifest.get("tokenizer", {})
prompt_contract = manifest.get("prompt_contract", {})
if (manifest.get("schema_version") != 4 or
        prompt_contract.get("prompt_version") != expected_handoff["native_prompt_version"] or
        tokenizer.get("selection_observation_length") != 384 or
        tokenizer.get("rollout_observation_length") != 500):
    raise SystemExit("native v3 data manifest contract mismatch")
PY
}

qwen_native_train_preflight() {
    local commit="$1" handoff_digest="$2" base_digest="$3" base_model data_digest
    export PYTHONPATH="$CHECKOUT_DIR${PYTHONPATH:+:$PYTHONPATH}"
    require_qwen_native_train
    verify_qwen_native_train_data
    base_model="$(readlink -f -- "$MODEL_DIR")"
    data_digest="$(file_sha256 "$NATIVE_TRAIN_MANIFEST")" || return 1
    verify_protocol_gate_evidence "$NATIVE_PROTOCOL_GATE_EVIDENCE" "$base_model" \
        "$base_digest" "$commit" "$handoff_digest" "$data_digest" || return $?
    if [[ "$NATIVE_TRAIN_STAGE" == main ]]; then
        verify_smoke_evidence "$NATIVE_SMOKE_EVIDENCE" "$base_model" "$base_digest" \
            "$commit" "$handoff_digest" "$data_digest" \
            "$NATIVE_TRAIN_PROTOCOL_GATE_DIGEST" || return $?
    fi
    NATIVE_TRAIN_PREFLIGHT_STAGE="$NATIVE_TRAIN_STAGE"
    NATIVE_TRAIN_PREFLIGHT_COMMIT="$commit"
    NATIVE_TRAIN_PREFLIGHT_HANDOFF="$handoff_digest"
    NATIVE_TRAIN_PREFLIGHT_BASE_DIGEST="$base_digest"
    NATIVE_TRAIN_PREFLIGHT_DATA_DIGEST="$data_digest"
    NATIVE_TRAIN_PREFLIGHT_PROTOCOL_GATE_DIGEST="$NATIVE_TRAIN_PROTOCOL_GATE_DIGEST"
    NATIVE_TRAIN_PREFLIGHT_SMOKE_DIGEST="${NATIVE_TRAIN_SMOKE_DIGEST:--}"
}

append_native_train_index() {
    local outer_attempt="$1" stage="$2" role="$3" run_dir="$4"
    local index="$outer_attempt/native-training-runs.tsv" current=''
    if [[ -f "$index" && ! -L "$index" ]]; then
        current="$(cat "$index")"$'\n'
    elif [[ -e "$index" || -L "$index" ]]; then
        printf 'Native training run index is not a regular file: %s\n' "$index" >&2
        return 1
    else
        current=$'stage\trole\trun_dir\n'
    fi
    atomic_write "$index" "$current$stage"$'\t'"$role"$'\t'"$run_dir"$'\n'
    sync_path "$outer_attempt"
}

verify_native_checkpoint_digest() {
    local checkpoint="$1" expected="$2" actual
    actual="$(tree_sha256 "$checkpoint")" || return 1
    [[ "$actual" == "$expected" ]] || {
        printf 'Checkpoint changed after its lineage was recorded: %s\n' "$checkpoint" >&2
        return 1
    }
}

seal_native_training_run() {
    local run_dir="$1" role="$2" checkpoint="$3" checkpoint_digest="$4"
    local parent="$5" parent_digest="$6" variant="$7" steps="$8"
    local cost_lambda="$9" cost_reward_mode="${10}" commit="${11}" handoff="${12}"
    local contract
    contract="$("$TRAIN_ENV/bin/python" - \
        "$run_dir" "$role" "$checkpoint" "$checkpoint_digest" \
        "$parent" "$parent_digest" "$variant" "$steps" \
        "$cost_lambda" "$cost_reward_mode" "$commit" "$handoff" \
        "$NATIVE_TRAIN_DATA_DIR" "$NATIVE_TRAIN_PROMPT_VERSION" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

from omegaconf import OmegaConf

(run_dir, role, checkpoint, checkpoint_digest, parent, parent_digest,
 variant, steps, cost_lambda_raw, cost_reward_mode, commit, handoff, data_dir,
 prompt_version) = sys.argv[1:]
run_dir = Path(run_dir)
checkpoint = Path(checkpoint)
parent = Path(parent)
data_dir = Path(data_dir)
steps = int(steps)
cost_lambda = float(cost_lambda_raw)
config_path = run_dir / "resolved-config.yaml"
env_path = run_dir / "run.env"
lineage_path = run_dir / "lineage.tsv"
for path in (config_path, env_path, lineage_path):
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"native training evidence is missing or symlinked: {path}")
config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)

def value(*keys):
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            raise SystemExit(f"resolved config is missing {'.'.join(keys)}")
        current = current[key]
    return current

checks = {
    "tool_protocol": value("tool_protocol") == "qwen35_native",
    "prompt_version": value("qwen35_prompt_version") == prompt_version,
    "native_variant": value("trainer", "native_training_variant") == variant,
    "model": Path(value("actor_rollout_ref", "model", "path")).resolve() == parent.resolve(),
    "train_file": Path(value("data", "train_files")) == data_dir / "train_512.parquet",
    "val_file": Path(value("data", "val_files")) == data_dir / "val_128.parquet",
    "raw_chat": value("data", "return_raw_chat") is True,
    "batch": value("data", "train_batch_size") == 8,
    "val_batch": value("data", "val_batch_size") == 8,
    "eval_group": value("data", "eval_group_size") == 1,
    "response": value("data", "max_response_length") == 500,
    "observation": value("data", "max_obs_length") == 500,
    "prompt": value("data", "max_prompt_length") == 4500,
    "start": value("data", "max_start_length") == 1024,
    "turns": value("max_turns") == 4,
    "capacity": (value("max_turns") * (
        value("data", "max_response_length") + value("data", "max_obs_length"))
        + value("data", "max_response_length")) <= 4500,
    "retriever": value("retriever", "topk") == 3,
    "gpus": value("trainer", "n_gpus_per_node") == 2,
    "steps": value("trainer", "total_training_steps") == steps,
    "save": value("trainer", "save_freq") == steps,
    "test": value("trainer", "test_freq") == steps,
    "seed": value("trainer", "seed") == 42,
    "group": value("actor_rollout_ref", "rollout", "n_agent") == 5,
    "sampling": value("actor_rollout_ref", "rollout", "do_sample") is True,
    "mini_batch": value("actor_rollout_ref", "actor", "ppo_mini_batch_size") == 40,
    "micro_batch": value("actor_rollout_ref", "actor", "ppo_micro_batch_size") == 2,
    "temperature": float(value("actor_rollout_ref", "rollout", "temperature")) == 1.0,
    "top_p": float(value("actor_rollout_ref", "rollout", "top_p")) == 1.0,
    "top_k": int(value("actor_rollout_ref", "rollout", "top_k")) == 0,
    "min_p": float(value("actor_rollout_ref", "rollout", "min_p")) == 0.0,
    "presence": float(value("actor_rollout_ref", "rollout", "presence_penalty")) == 0.0,
    "repetition": float(value("actor_rollout_ref", "rollout", "repetition_penalty")) == 1.0,
    "cost_lambda": float(value("algorithm", "cost_lambda")) == cost_lambda,
    "cost_mode": value("algorithm", "cost_reward_mode") == cost_reward_mode,
    "trace_parent": value("trainer", "trace_parent_checkpoint_digest") == parent_digest,
}
failed = sorted(name for name, passed in checks.items() if not passed)
if failed:
    raise SystemExit("native training resolved config mismatch: " + ", ".join(failed))
run_env = {}
for line in env_path.read_text(encoding="utf-8").splitlines():
    if line and "=" in line:
        key, item = line.split("=", 1)
        if key in run_env:
            raise SystemExit(f"duplicate run.env key: {key}")
        run_env[key] = item
config_digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
expected_env = {
    "job_mode": "train", "variant": variant, "gpu_count": "2",
    "train_steps": str(steps), "train_batch_size": "8",
    "max_response_length": "500", "input_model": str(parent),
    "data_dir": str(data_dir), "eval_data_file": "", "eval_group_size": "1",
    "tool_protocol": "qwen35_native", "rollout_top_k": "0",
    "rollout_min_p": "0.0", "rollout_presence_penalty": "0.0",
    "rollout_repetition_penalty": "1.0", "trace_output_dir": str(run_dir / "traces"),
    "resolved_config_sha256": config_digest, "role": role,
    "checkpoint": str(checkpoint), "checkpoint_digest": checkpoint_digest,
    "parent_checkpoint": str(parent), "parent_checkpoint_digest": parent_digest,
    "checkout_commit": commit, "cpu_handoff_digest": handoff, "seed": "42",
    "cost_lambda": cost_lambda_raw, "cost_reward_mode": cost_reward_mode,
}
bad_env = sorted(key for key, expected in expected_env.items()
                 if run_env.get(key) != expected)
if bad_env:
    raise SystemExit("native training run.env mismatch: " + ", ".join(bad_env))
lines = lineage_path.read_text(encoding="utf-8").splitlines()
header = ("role\tcheckpoint\tcheckpoint_digest\tparent_checkpoint\t"
          "parent_checkpoint_digest\tcheckout_commit\tcpu_handoff_digest\t"
          "resolved_config_sha256")
expected_row = "\t".join([
    role, str(checkpoint), checkpoint_digest, str(parent), parent_digest,
    commit, handoff, config_digest,
])
if lines != [header, expected_row]:
    raise SystemExit("native training lineage.tsv mismatch")
payload = {
    "checkpoint": str(checkpoint), "checkpoint_digest": checkpoint_digest,
    "config_sha256": config_digest, "cost_lambda": cost_lambda,
    "cost_reward_mode": cost_reward_mode, "group_size": 5,
    "max_action_budget": 4, "max_obs_length": 500, "max_response_length": 500,
    "parent": str(parent), "parent_digest": parent_digest,
    "prompt_version": prompt_version, "role": role, "schema": 3,
    "steps": steps, "train_batch_size": 8, "variant": variant,
}
print(json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
PY
)" || return 1
    atomic_write "$run_dir/native-training-contract.json" "$contract"$'\n'
    sync_path "$run_dir"
}

verify_native_trace_identity() {
    local manifest="$1" expected_rows="$2"
    PYTHONPATH="$CHECKOUT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
        "$TRAIN_ENV/bin/python" - "$manifest" "$expected_rows" <<'PY'
import hashlib
from pathlib import Path
import sys
from search_r1.trajectory_trace import verify_trace_manifest

path = Path(sys.argv[1])
manifest = verify_trace_manifest(path, int(sys.argv[2]))
print(manifest["artifact"]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest(), sep="\t")
PY
}

create_native_training_results_dir() {
    local outer_attempt="$1" results_dir
    results_dir="$NATIVE_TRAIN_RESULTS_ROOT/attempts/$(basename -- "$outer_attempt")"
    mkdir -p "$NATIVE_TRAIN_RESULTS_ROOT/attempts"
    [[ ! -e "$results_dir" && ! -L "$results_dir" ]] || {
        printf 'Refusing to overwrite native training results: %s\n' "$results_dir" >&2
        return 1
    }
    mkdir "$results_dir"
    printf '%s\n' "$results_dir"
}

record_smoke_storage() {
    local results_dir="$1" checkpoint="$2" checkpoint_bytes available_bytes
    checkpoint_bytes="$(du -sb -- "$checkpoint" | awk '{print $1}')" || return 1
    available_bytes="$(df -B1 --output=avail "$PROJECT_ROOT" | tail -n 1 | tr -d '[:space:]')" || return 1
    [[ "$checkpoint_bytes" =~ ^[0-9]+$ && "$available_bytes" =~ ^[0-9]+$ ]] || return 1
    atomic_write "$results_dir/storage.env" \
        "checkpoint_bytes=$checkpoint_bytes"$'\n'\
"filesystem_available_bytes=$available_bytes"$'\n'\
"recorded_at=$(utc_now)"$'\n'
}

publish_native_training_evidence() {
    local outer_attempt="$1" results_dir="$2" namespace="$3" contract="$4"
    shift 4
    local marker_dir marker evidence_digest path relative checksum_lines=''
    local -a evidence_files=(
        "$NATIVE_TRAIN_MANIFEST" "$NATIVE_TRAIN_MANIFEST.sha256"
        "$NATIVE_TRAIN_CATALOG" "$NATIVE_TRAIN_DATA_DIR/train_512.parquet"
        "$NATIVE_TRAIN_DATA_DIR/val_128.parquet" "$NATIVE_TRAIN_G3_DATA"
        "$NATIVE_TRAIN_DATA_DIR/nq_test_128_native_v3.parquet"
        "$NATIVE_TRAIN_DATA_DIR/multihop_eval_256_native_v3.parquet"
        "$HANDOFF" "$HANDOFF.sha256" "$MANIFEST_DIR/cpu.ok"
        "$MANIFEST_DIR/git.ok" "$MANIFEST_DIR/checkout-tree.sha256" "$@"
    )
    local -a relative_files=()
    for path in "${evidence_files[@]}"; do
        relative_files+=("$(native_train_project_relative_file "$path")") || return 1
    done
    while IFS= read -r relative; do
        path="$PROJECT_ROOT/$relative"
        checksum_lines+="$(file_sha256 "$path")  $relative"$'\n'
        sync_path "$path"
    done < <(printf '%s\n' "${relative_files[@]}" | LC_ALL=C sort -u)
    atomic_write "$results_dir/evidence.sha256" "$checksum_lines"
    sync_path "$results_dir"
    verify_native_training_checksum_manifest "$results_dir/evidence.sha256" || return 1
    evidence_digest="$(file_sha256 "$results_dir/evidence.sha256")" || return 1
    marker_dir="$MANIFEST_DIR/$namespace"
    if [[ -e "$marker_dir" || -L "$marker_dir" ]]; then
        [[ -d "$marker_dir" && ! -L "$marker_dir" ]] || return 1
    else
        mkdir "$marker_dir"
    fi
    marker="$marker_dir/$(basename -- "$outer_attempt").ok"
    [[ ! -e "$marker" && ! -L "$marker" ]] || {
        printf 'Refusing to overwrite native training evidence marker: %s\n' "$marker" >&2
        return 1
    }
    atomic_write "$marker" "$evidence_digest"$'\n'
    sync_path "$marker_dir"
    atomic_write "$outer_attempt/result-contract" "$contract"$'\n'
    atomic_write "$outer_attempt/result-root" "$results_dir"$'\n'
    atomic_write "$outer_attempt/evidence-marker" "$marker"$'\n'
    atomic_write "$outer_attempt/evidence-digest" "$evidence_digest"$'\n'
    sync_path "$outer_attempt"
    atomic_write "$NATIVE_TRAIN_RESULTS_ROOT/latest-${NATIVE_TRAIN_STAGE}" "$results_dir"$'\n'
    sync_path "$NATIVE_TRAIN_RESULTS_ROOT"
}

validate_r_g3_analysis() {
    local results_dir="$1" checkpoint_digest="$2" trace_digest="$3"
    "$TRAIN_ENV/bin/python" - "$results_dir" "$checkpoint_digest" "$trace_digest" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
checkpoint_digest, trace_digest = sys.argv[2:]
decision = json.loads((root / "go_no_go.json").read_bytes())
summary = json.loads((root / "summary.json").read_bytes())
count = summary.get("overall", {}).get("cost_contrast_group_count")
if isinstance(count, bool) or not isinstance(count, int) or count < 0:
    raise SystemExit("R-G3 cost_contrast_group_count is invalid")
checks = {
    "schema": (decision.get("schema") == "search-r1.grouped-probe-analysis" and
               decision.get("schema_version") == 3 and
               summary.get("schema") == "search-r1.grouped-probe-analysis" and
               summary.get("schema_version") == 3),
    "decision": decision.get("decision") in {"GO", "NO-GO"},
    "summary_decision": summary.get("decision") == decision.get("decision"),
    "checkpoint": summary.get("input", {}).get("checkpoint_digest") == checkpoint_digest,
    "trace": summary.get("input", {}).get("trace_sha256") == trace_digest,
    "decision_trace": decision.get("trace_sha256") == trace_digest,
    "trace_stage": summary.get("input", {}).get("stage") == "qwen_native_g3",
}
failed = sorted(name for name, passed in checks.items() if not passed)
if failed:
    raise SystemExit("R-G3 analysis mismatch: " + ", ".join(failed))
print(decision["decision"], count, sep="\t")
PY
}

write_branch_decision() {
    local output="$1" analysis_decision="$2" contrast_count="$3"
    "$TRAIN_ENV/bin/python" - "$output" "$analysis_decision" "$contrast_count" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
analysis_decision = sys.argv[2]
count = int(sys.argv[3])
authorized = analysis_decision == "GO" and count >= 8
payload = {
    "analysis_decision": analysis_decision,
    "branches_authorized": authorized,
    "cost_contrast_group_count": count,
    "cost_contrast_group_minimum": 8,
    "decision": "GO" if authorized else "NO-GO",
    "schema": "qwen-native-post-r-gate-v3",
}
print(json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
PY
}

verify_native_training_preflight_state() {
    local commit="$1" handoff_digest="$2" base_digest="$3" data_digest="$4"
    [[ "$NATIVE_TRAIN_PREFLIGHT_STAGE" == "$NATIVE_TRAIN_STAGE" &&
        "$NATIVE_TRAIN_PREFLIGHT_COMMIT" == "$commit" &&
        "$NATIVE_TRAIN_PREFLIGHT_HANDOFF" == "$handoff_digest" &&
        "$NATIVE_TRAIN_PREFLIGHT_BASE_DIGEST" == "$base_digest" &&
        "$NATIVE_TRAIN_PREFLIGHT_DATA_DIGEST" == "$data_digest" &&
        "$NATIVE_TRAIN_PREFLIGHT_PROTOCOL_GATE_DIGEST" == "$NATIVE_TRAIN_PROTOCOL_GATE_DIGEST" ]] || {
        printf 'Qwen native training inputs drifted after GPU preflight.\n' >&2
        return 1
    }
    if [[ "$NATIVE_TRAIN_STAGE" == main &&
          "$NATIVE_TRAIN_PREFLIGHT_SMOKE_DIGEST" != "$NATIVE_TRAIN_SMOKE_DIGEST" ]]; then
        printf 'Qwen native smoke evidence drifted after GPU preflight.\n' >&2
        return 1
    fi
}

qwen_native_smoke_pipeline() {
    local outer_attempt="$1" commit="$2" handoff_digest="$3"
    local base_model="$4" base_digest="$5" data_digest="$6"
    local run checkpoint checkpoint_digest config_digest contract_digest
    local trace_identity trace_digest trace_manifest_digest results_dir index_content
    local smoke_decision
    local -a smoke_wandb_files=()
    verify_native_checkpoint_digest "$base_model" "$base_digest" || return $?
    run_job train smoke "$BASE_GATE_STEPS" '' "$base_digest" || return $?
    run="$LAST_RUN_DIR"
    append_native_train_index "$outer_attempt" S smoke "$run" || return $?
    checkpoint="$(fixed_checkpoint "$run" "$BASE_GATE_STEPS")" || return 1
    checkpoint_digest="$(tree_sha256 "$checkpoint")" || return 1
    record_lineage "$run" smoke "$checkpoint" "$checkpoint_digest" \
        "$base_model" "$base_digest" "$commit" "$handoff_digest" 0 linear || return $?
    seal_native_training_run "$run" smoke "$checkpoint" "$checkpoint_digest" \
        "$base_model" "$base_digest" smoke "$BASE_GATE_STEPS" 0 linear \
        "$commit" "$handoff_digest" || return $?
    trace_identity="$(verify_native_trace_identity \
        "$run/traces/train_trajectories.manifest.json" \
        "$((BASE_GATE_STEPS * TRAIN_BATCH_SIZE * 5))")" || return 1
    IFS=$'\t' read -r trace_digest trace_manifest_digest <<<"$trace_identity"
    results_dir="$(create_native_training_results_dir "$outer_attempt")" || return 1
    smoke_decision="$("$TRAIN_ENV/bin/python" "$NATIVE_TRAIN_SMOKE_CLI" \
        --trace "$run/traces/train_trajectories.jsonl" \
        --log "$run/train.log" \
        --catalog "$NATIVE_TRAIN_CATALOG" \
        --wandb-dir "$run/wandb" \
        --output "$results_dir/smoke-decision.json")" || return $?
    [[ "$smoke_decision" == GO || "$smoke_decision" == NO-GO ]] || {
        printf 'Unexpected smoke decision: %s\n' "$smoke_decision" >&2
        return 1
    }
    record_smoke_storage "$results_dir" "$checkpoint" || return $?
    config_digest="$(file_sha256 "$run/resolved-config.yaml")" || return 1
    contract_digest="$(file_sha256 "$run/native-training-contract.json")" || return 1
    atomic_write "$results_dir/contract.env" \
"schema=$NATIVE_TRAIN_SMOKE_CONTRACT"$'\n'\
"stage=smoke"$'\n'\
"stage_order=S2"$'\n'\
"decision=$smoke_decision"$'\n'\
"manual_review_required=true"$'\n'\
"protocol_gate_evidence=$NATIVE_PROTOCOL_GATE_EVIDENCE"$'\n'\
"protocol_gate_evidence_sha256=$NATIVE_TRAIN_PROTOCOL_GATE_DIGEST"$'\n'
    atomic_write "$results_dir/lineage.tsv" \
        $'stage\trole\trun_dir\tcheckpoint\tcheckpoint_digest\tparent_checkpoint\tparent_checkpoint_digest\tcheckout_commit\tcpu_handoff_digest\tdata_manifest_sha256\tresolved_config_sha256\ttrace_sha256\ttrace_manifest_sha256\trun_contract_sha256\tprotocol_gate_evidence\tprotocol_gate_evidence_sha256\n'\
"S"$'\t'"smoke"$'\t'"$run"$'\t'"$checkpoint"$'\t'"$checkpoint_digest"$'\t'"$base_model"$'\t'"$base_digest"$'\t'"$commit"$'\t'"$handoff_digest"$'\t'"$data_digest"$'\t'"$config_digest"$'\t'"$trace_digest"$'\t'"$trace_manifest_digest"$'\t'"$contract_digest"$'\t'"$NATIVE_PROTOCOL_GATE_EVIDENCE"$'\t'"$NATIVE_TRAIN_PROTOCOL_GATE_DIGEST"$'\n'
    index_content="$(cat "$outer_attempt/native-training-runs.tsv")"
    atomic_write "$results_dir/run-index.tsv" "$index_content"$'\n'
    atomic_write "$results_dir/checkpoint-tree.env" \
        "checkpoint=$checkpoint"$'\n'\
"checkpoint_tree_sha256=$checkpoint_digest"$'\n'
    sync_path "$results_dir"
    if [[ -d "$run/wandb" && ! -L "$run/wandb" ]]; then
        mapfile -d '' -t smoke_wandb_files < <(
            find "$run/wandb" -type f -size +0c -print0 | LC_ALL=C sort -z
        )
    fi
    verify_checkout "$commit" || return $?
    [[ "$(file_sha256 "$NATIVE_TRAIN_MANIFEST")" == "$data_digest" ]] || return 1
    verify_protocol_gate_evidence "$NATIVE_PROTOCOL_GATE_EVIDENCE" "$base_model" \
        "$base_digest" "$commit" "$handoff_digest" "$data_digest" || return $?
    verify_native_checkpoint_digest "$base_model" "$base_digest" || return $?
    verify_native_checkpoint_digest "$checkpoint" "$checkpoint_digest" || return $?
    publish_native_training_evidence "$outer_attempt" "$results_dir" \
        qwen-native-training-smoke "$NATIVE_TRAIN_SMOKE_CONTRACT" \
        "$results_dir/contract.env" "$results_dir/lineage.tsv" \
        "$results_dir/run-index.tsv" "$results_dir/storage.env" \
        "$results_dir/checkpoint-tree.env" "$NATIVE_PROTOCOL_GATE_EVIDENCE" \
        "$NATIVE_TRAIN_PROTOCOL_GATE_MANIFEST" "$run/train.log" \
        "$run/resolved-config.yaml" "$run/run.env" "$run/lineage.tsv" \
        "$run/native-training-contract.json" \
        "$results_dir/smoke-decision.json" \
        "$run/traces/train_trajectories.jsonl" \
        "$run/traces/train_trajectories.manifest.json" \
        "$run/traces/train_trajectories.manifest.json.sha256" \
        "${smoke_wandb_files[@]}" || return $?
    printf 'Completed Qwen native two-step smoke (%s): %s\n' \
        "$smoke_decision" "$results_dir"
    if [[ "$smoke_decision" == GO ]]; then
        printf 'Review storage.env; passing the exact marker to main records explicit approval.\n'
    else
        printf 'Smoke is NO-GO; main training remains blocked.\n'
    fi
}

qwen_native_main_pipeline() {
    local outer_attempt="$1" commit="$2" handoff_digest="$3"
    local base_model="$4" base_digest="$5" data_digest="$6"
    local reproduce_run eval_run control_run='' cost_run=''
    local reproduce_checkpoint control_checkpoint='' cost_checkpoint=''
    local reproduce_digest control_digest='' cost_digest=''
    local reproduce_config control_config='' cost_config='' eval_config
    local reproduce_contract control_contract='' cost_contract=''
    local reproduce_trace control_trace='' cost_trace='' eval_trace
    local reproduce_trace_digest reproduce_trace_manifest_digest
    local control_trace_digest='' control_trace_manifest_digest=''
    local cost_trace_digest='' cost_trace_manifest_digest=''
    local eval_trace_digest eval_trace_manifest_digest
    local results_dir analysis_dir decision_metadata analysis_decision contrast_count
    local branch_authorized=false branch_rows='' ar_rows='' branch_json stage_order index_content
    local wandb_run wandb_file wandb_count wandb_history_count
    local eval_key eval_suffix eval_artifact eval_rows eval_stage control_eval_run cost_eval_run
    local trace_identity eval_trace_digest_tmp eval_trace_manifest_digest_tmp
    local -a evidence_files wandb_runs
    local -a eval_keys=(val nq_test multihop)
    local -A control_eval_runs cost_eval_runs control_eval_trace_digests
    local -A control_eval_trace_manifest_digests cost_eval_trace_digests
    local -A cost_eval_trace_manifest_digests paired_dirs eval_expected_rows eval_artifacts
    local -A parent_eval_runs reproduced_eval_runs parent_eval_trace_digests
    local -A parent_eval_trace_manifest_digests reproduced_eval_trace_digests
    local -A reproduced_eval_trace_manifest_digests ar_dirs
    eval_expected_rows[val]=128
    eval_expected_rows[nq_test]=128
    eval_expected_rows[multihop]=256
    eval_artifacts[val]=val
    eval_artifacts[nq_test]=nq_test_eval
    eval_artifacts[multihop]=multihop_eval

    verify_native_checkpoint_digest "$base_model" "$base_digest" || return $?
    # R is always a fresh process from the sealed base. Smoke weights are never an input.
    run_job train reproduce "$REPRODUCE_STEPS" '' "$base_digest" || return $?
    reproduce_run="$LAST_RUN_DIR"
    append_native_train_index "$outer_attempt" R reproduced "$reproduce_run" || return $?
    reproduce_checkpoint="$(fixed_checkpoint "$reproduce_run" "$REPRODUCE_STEPS")" || return 1
    reproduce_digest="$(tree_sha256 "$reproduce_checkpoint")" || return 1
    record_lineage "$reproduce_run" reproduced "$reproduce_checkpoint" "$reproduce_digest" \
        "$base_model" "$base_digest" "$commit" "$handoff_digest" 0 linear || return $?
    seal_native_training_run "$reproduce_run" reproduced "$reproduce_checkpoint" \
        "$reproduce_digest" "$base_model" "$base_digest" reproduce "$REPRODUCE_STEPS" \
        0 linear "$commit" "$handoff_digest" || return $?
    reproduce_trace="$(verify_native_trace_identity \
        "$reproduce_run/traces/train_trajectories.manifest.json" \
        "$((REPRODUCE_STEPS * TRAIN_BATCH_SIZE * 5))")" || return 1
    IFS=$'\t' read -r reproduce_trace_digest reproduce_trace_manifest_digest \
        <<<"$reproduce_trace"

    results_dir="$(create_native_training_results_dir "$outer_attempt")" || return 1
    analysis_dir="$results_dir/r-g3-analysis"
    export EVAL_DATA_FILE="$NATIVE_TRAIN_G3_DATA"
    export EVAL_EXPECTED_ROWS=64
    export EVAL_GROUP_SIZE=5
    verify_native_checkpoint_digest "$reproduce_checkpoint" "$reproduce_digest" || return $?
    run_job eval qwen_native_g3 "$reproduce_checkpoint" '' "$reproduce_digest" || return $?
    eval_run="$LAST_RUN_DIR"
    append_native_train_index "$outer_attempt" G3 capability_gate "$eval_run" || return $?
    "$TRAIN_ENV/bin/python" "$NATIVE_TRAIN_GATE_CLI" \
        --stage g3 \
        --trace "$eval_run/traces/eval_predictions.jsonl" \
        --catalog "$NATIVE_TRAIN_CATALOG" \
        --data-manifest "$NATIVE_TRAIN_MANIFEST" \
        --expected-checkpoint-digest "$reproduce_digest" \
        --output-dir "$analysis_dir" || return $?
    eval_trace="$(verify_native_trace_identity \
        "$eval_run/traces/eval_predictions.manifest.json" 320)" || return 1
    IFS=$'\t' read -r eval_trace_digest eval_trace_manifest_digest <<<"$eval_trace"
    decision_metadata="$(validate_r_g3_analysis "$analysis_dir" \
        "$reproduce_digest" "$eval_trace_digest")" || return 1
    IFS=$'\t' read -r analysis_decision contrast_count <<<"$decision_metadata"
    if [[ "$analysis_decision" == GO && "$contrast_count" -ge 8 ]]; then
        branch_authorized=true
    fi
    branch_json="$(write_branch_decision "$results_dir/branch-decision.json" \
        "$analysis_decision" "$contrast_count")" || return 1
    atomic_write "$results_dir/branch-decision.json" "$branch_json"$'\n'

    # B/C use the sealed native-v3 train/eval contract, not the temporary 64x5 probe.
    export EVAL_DATA_FILE=''
    export EVAL_EXPECTED_ROWS=128
    export EVAL_GROUP_SIZE=1

    for eval_key in "${eval_keys[@]}"; do
        eval_rows="${eval_expected_rows[$eval_key]}"
        export EVAL_EXPECTED_ROWS="$eval_rows"

        run_job eval "qwen_native_a_$eval_key" "$base_model" '' "$base_digest" || return $?
        parent_eval_runs[$eval_key]="$LAST_RUN_DIR"
        eval_stage="${eval_key^^}"
        eval_stage="A-${eval_stage//_/-}-EVAL"
        append_native_train_index "$outer_attempt" "$eval_stage" \
            "parent_${eval_key}_eval" "${parent_eval_runs[$eval_key]}" || return $?
        trace_identity="$(verify_native_trace_identity \
            "${parent_eval_runs[$eval_key]}/traces/eval_predictions.manifest.json" \
            "$eval_rows")" || return 1
        IFS=$'\t' read -r eval_trace_digest_tmp eval_trace_manifest_digest_tmp \
            <<<"$trace_identity"
        parent_eval_trace_digests[$eval_key]="$eval_trace_digest_tmp"
        parent_eval_trace_manifest_digests[$eval_key]="$eval_trace_manifest_digest_tmp"

        run_job eval "qwen_native_r_$eval_key" "$reproduce_checkpoint" '' \
            "$reproduce_digest" || return $?
        reproduced_eval_runs[$eval_key]="$LAST_RUN_DIR"
        eval_stage="${eval_key^^}"
        eval_stage="R-${eval_stage//_/-}-EVAL"
        append_native_train_index "$outer_attempt" "$eval_stage" \
            "reproduced_${eval_key}_eval" "${reproduced_eval_runs[$eval_key]}" || return $?
        trace_identity="$(verify_native_trace_identity \
            "${reproduced_eval_runs[$eval_key]}/traces/eval_predictions.manifest.json" \
            "$eval_rows")" || return 1
        IFS=$'\t' read -r eval_trace_digest_tmp eval_trace_manifest_digest_tmp \
            <<<"$trace_identity"
        reproduced_eval_trace_digests[$eval_key]="$eval_trace_digest_tmp"
        reproduced_eval_trace_manifest_digests[$eval_key]="$eval_trace_manifest_digest_tmp"

        ar_dirs[$eval_key]="$results_dir/paired-ar-$eval_key"
        "$TRAIN_ENV/bin/python" "$NATIVE_TRAIN_PAIRED_CLI" \
            --parent "${parent_eval_runs[$eval_key]}/traces/eval_predictions.jsonl" \
            --reproduced "${reproduced_eval_runs[$eval_key]}/traces/eval_predictions.jsonl" \
            --data-manifest "$NATIVE_TRAIN_MANIFEST" \
            --eval-artifact "${eval_artifacts[$eval_key]}" \
            --expected-parent-checkpoint-digest "$base_digest" \
            --expected-reproduced-checkpoint-digest "$reproduce_digest" \
            --output-dir "${ar_dirs[$eval_key]}" \
            --expected-rows "$eval_rows" || return $?
    done

    if [[ "$branch_authorized" == true ]]; then
        # Endpoint loops end on multihop-256; restore the neutral training metadata.
        export EVAL_EXPECTED_ROWS=128
        verify_native_checkpoint_digest "$reproduce_checkpoint" "$reproduce_digest" || return $?
        run_job train control "$BRANCH_STEPS" "$reproduce_checkpoint" \
            "$reproduce_digest" || return $?
        control_run="$LAST_RUN_DIR"
        append_native_train_index "$outer_attempt" B control "$control_run" || return $?
        control_checkpoint="$(fixed_checkpoint "$control_run" "$BRANCH_STEPS")" || return 1
        control_digest="$(tree_sha256 "$control_checkpoint")" || return 1
        record_lineage "$control_run" control "$control_checkpoint" "$control_digest" \
            "$reproduce_checkpoint" "$reproduce_digest" "$commit" "$handoff_digest" \
            0 linear || return $?
        seal_native_training_run "$control_run" control "$control_checkpoint" "$control_digest" \
            "$reproduce_checkpoint" "$reproduce_digest" control "$BRANCH_STEPS" 0 linear \
            "$commit" "$handoff_digest" || return $?
        control_trace="$(verify_native_trace_identity \
            "$control_run/traces/train_trajectories.manifest.json" \
            "$((BRANCH_STEPS * TRAIN_BATCH_SIZE * 5))")" || return 1
        IFS=$'\t' read -r control_trace_digest control_trace_manifest_digest \
            <<<"$control_trace"

        verify_native_checkpoint_digest "$reproduce_checkpoint" "$reproduce_digest" || return $?
        run_job train cost_aware_gated "$BRANCH_STEPS" "$reproduce_checkpoint" \
            "$reproduce_digest" || return $?
        cost_run="$LAST_RUN_DIR"
        append_native_train_index "$outer_attempt" C cost_aware_gated "$cost_run" || return $?
        cost_checkpoint="$(fixed_checkpoint "$cost_run" "$BRANCH_STEPS")" || return 1
        cost_digest="$(tree_sha256 "$cost_checkpoint")" || return 1
        record_lineage "$cost_run" cost_aware_gated "$cost_checkpoint" "$cost_digest" \
            "$reproduce_checkpoint" "$reproduce_digest" "$commit" "$handoff_digest" \
            0.10 correct_only || return $?
        seal_native_training_run "$cost_run" cost_aware_gated "$cost_checkpoint" "$cost_digest" \
            "$reproduce_checkpoint" "$reproduce_digest" cost_aware_gated "$BRANCH_STEPS" \
            0.10 correct_only "$commit" "$handoff_digest" || return $?
        cost_trace="$(verify_native_trace_identity \
            "$cost_run/traces/train_trajectories.manifest.json" \
            "$((BRANCH_STEPS * TRAIN_BATCH_SIZE * 5))")" || return 1
        IFS=$'\t' read -r cost_trace_digest cost_trace_manifest_digest <<<"$cost_trace"

        # Evaluate both fixed endpoints greedily on curated and unfiltered v3 sets.
        for eval_key in "${eval_keys[@]}"; do
            eval_suffix="$eval_key"
            eval_artifact="${eval_artifacts[$eval_key]}"
            eval_rows="${eval_expected_rows[$eval_key]}"
            export EVAL_EXPECTED_ROWS="$eval_rows"

            run_job eval "qwen_native_b_$eval_suffix" "$control_checkpoint" '' \
                "$control_digest" || return $?
            control_eval_run="$LAST_RUN_DIR"
            control_eval_runs[$eval_key]="$control_eval_run"
            eval_stage="${eval_key^^}"
            eval_stage="B-${eval_stage//_/-}-EVAL"
            append_native_train_index "$outer_attempt" "$eval_stage" \
                "control_${eval_key}_eval" "$control_eval_run" || return $?
            trace_identity="$(verify_native_trace_identity \
                "$control_eval_run/traces/eval_predictions.manifest.json" \
                "$eval_rows")" || return 1
            IFS=$'\t' read -r eval_trace_digest_tmp \
                eval_trace_manifest_digest_tmp <<<"$trace_identity"
            control_eval_trace_digests[$eval_key]="$eval_trace_digest_tmp"
            control_eval_trace_manifest_digests[$eval_key]="$eval_trace_manifest_digest_tmp"

            run_job eval "qwen_native_c_$eval_suffix" "$cost_checkpoint" '' \
                "$cost_digest" || return $?
            cost_eval_run="$LAST_RUN_DIR"
            cost_eval_runs[$eval_key]="$cost_eval_run"
            eval_stage="${eval_key^^}"
            eval_stage="C-${eval_stage//_/-}-EVAL"
            append_native_train_index "$outer_attempt" "$eval_stage" \
                "cost_aware_gated_${eval_key}_eval" "$cost_eval_run" || return $?
            trace_identity="$(verify_native_trace_identity \
                "$cost_eval_run/traces/eval_predictions.manifest.json" \
                "$eval_rows")" || return 1
            IFS=$'\t' read -r eval_trace_digest_tmp \
                eval_trace_manifest_digest_tmp <<<"$trace_identity"
            cost_eval_trace_digests[$eval_key]="$eval_trace_digest_tmp"
            cost_eval_trace_manifest_digests[$eval_key]="$eval_trace_manifest_digest_tmp"

            paired_dirs[$eval_key]="$results_dir/paired-$eval_key"
            "$TRAIN_ENV/bin/python" "$NATIVE_TRAIN_PAIRED_CLI" \
                --control "$control_eval_run/traces/eval_predictions.jsonl" \
                --cost-aware-gated "$cost_eval_run/traces/eval_predictions.jsonl" \
                --data-manifest "$NATIVE_TRAIN_MANIFEST" \
                --eval-artifact "$eval_artifact" \
                --expected-control-checkpoint-digest "$control_digest" \
                --expected-cost-aware-gated-checkpoint-digest "$cost_digest" \
                --output-dir "${paired_dirs[$eval_key]}" \
                --expected-rows "$eval_rows" || return $?
        done
    fi

    stage_order=R60,G3,A-VAL-EVAL,R-VAL-EVAL,A-NQ-TEST-EVAL,R-NQ-TEST-EVAL,A-MULTIHOP-EVAL,R-MULTIHOP-EVAL
    if [[ "$branch_authorized" == true ]]; then
        stage_order+=,B20,C20,B-VAL-EVAL,C-VAL-EVAL,B-NQ-TEST-EVAL,C-NQ-TEST-EVAL,B-MULTIHOP-EVAL,C-MULTIHOP-EVAL
    fi
    reproduce_config="$(file_sha256 "$reproduce_run/resolved-config.yaml")" || return 1
    reproduce_contract="$(file_sha256 "$reproduce_run/native-training-contract.json")" || return 1
    eval_config="$(file_sha256 "$eval_run/resolved-config.yaml")" || return 1
    atomic_write "$results_dir/contract.env" \
        "schema=$NATIVE_TRAIN_MAIN_CONTRACT"$'\n'\
"stage=main"$'\n'\
"stage_order=$stage_order"$'\n'\
"analysis_decision=$analysis_decision"$'\n'\
"cost_contrast_group_count=$contrast_count"$'\n'\
"branch_authorized=$branch_authorized"$'\n'\
"protocol_gate_evidence=$NATIVE_PROTOCOL_GATE_EVIDENCE"$'\n'\
"protocol_gate_evidence_sha256=$NATIVE_TRAIN_PROTOCOL_GATE_DIGEST"$'\n'\
"smoke_evidence=$NATIVE_SMOKE_EVIDENCE"$'\n'\
"smoke_evidence_sha256=$NATIVE_TRAIN_SMOKE_DIGEST"$'\n'
    atomic_write "$results_dir/lineage.tsv" \
        $'stage\trole\trun_dir\tcheckpoint\tcheckpoint_digest\tparent_checkpoint\tparent_checkpoint_digest\tcheckout_commit\tcpu_handoff_digest\tdata_manifest_sha256\tresolved_config_sha256\ttrace_sha256\ttrace_manifest_sha256\trun_contract_sha256\tpredecessor_evidence_sha256\n'\
"R"$'\t'"reproduced"$'\t'"$reproduce_run"$'\t'"$reproduce_checkpoint"$'\t'"$reproduce_digest"$'\t'"$base_model"$'\t'"$base_digest"$'\t'"$commit"$'\t'"$handoff_digest"$'\t'"$data_digest"$'\t'"$reproduce_config"$'\t'"$reproduce_trace_digest"$'\t'"$reproduce_trace_manifest_digest"$'\t'"$reproduce_contract"$'\t'"$NATIVE_TRAIN_SMOKE_DIGEST"$'\n'\
"G3"$'\t'"capability_gate"$'\t'"$eval_run"$'\t'"$reproduce_checkpoint"$'\t'"$reproduce_digest"$'\t'"$reproduce_checkpoint"$'\t'"$reproduce_digest"$'\t'"$commit"$'\t'"$handoff_digest"$'\t'"$data_digest"$'\t'"$eval_config"$'\t'"$eval_trace_digest"$'\t'"$eval_trace_manifest_digest"$'\t-'$'\t'"$NATIVE_TRAIN_SMOKE_DIGEST"$'\n'
    for eval_key in "${eval_keys[@]}"; do
        eval_stage="${eval_key^^}"
        eval_stage="${eval_stage//_/-}"
        ar_rows+="A-$eval_stage-EVAL"$'\t'"parent_${eval_key}_eval"$'\t'"${parent_eval_runs[$eval_key]}"$'\t'"$base_model"$'\t'"$base_digest"$'\t'"$base_model"$'\t'"$base_digest"$'\t'"$commit"$'\t'"$handoff_digest"$'\t'"$data_digest"$'\t'"$(file_sha256 "${parent_eval_runs[$eval_key]}/resolved-config.yaml")"$'\t'"${parent_eval_trace_digests[$eval_key]}"$'\t'"${parent_eval_trace_manifest_digests[$eval_key]}"$'\t-'$'\t'"$NATIVE_TRAIN_SMOKE_DIGEST"$'\n'
        ar_rows+="R-$eval_stage-EVAL"$'\t'"reproduced_${eval_key}_eval"$'\t'"${reproduced_eval_runs[$eval_key]}"$'\t'"$reproduce_checkpoint"$'\t'"$reproduce_digest"$'\t'"$reproduce_checkpoint"$'\t'"$reproduce_digest"$'\t'"$commit"$'\t'"$handoff_digest"$'\t'"$data_digest"$'\t'"$(file_sha256 "${reproduced_eval_runs[$eval_key]}/resolved-config.yaml")"$'\t'"${reproduced_eval_trace_digests[$eval_key]}"$'\t'"${reproduced_eval_trace_manifest_digests[$eval_key]}"$'\t-'$'\t'"$NATIVE_TRAIN_SMOKE_DIGEST"$'\n'
    done
    printf '%s' "$ar_rows" >>"$results_dir/lineage.tsv"
    evidence_files=(
        "$results_dir/contract.env" "$results_dir/lineage.tsv"
        "$results_dir/branch-decision.json" "$NATIVE_PROTOCOL_GATE_EVIDENCE"
        "$NATIVE_TRAIN_PROTOCOL_GATE_MANIFEST" "$NATIVE_SMOKE_EVIDENCE"
        "$NATIVE_TRAIN_SMOKE_MANIFEST" "$NATIVE_TRAIN_SMOKE_RESULTS/contract.env"
        "$NATIVE_TRAIN_SMOKE_RESULTS/lineage.tsv" "$NATIVE_TRAIN_SMOKE_RESULTS/storage.env"
        "$reproduce_run/train.log" "$reproduce_run/resolved-config.yaml"
        "$reproduce_run/run.env" "$reproduce_run/lineage.tsv"
        "$reproduce_run/native-training-contract.json"
        "$reproduce_run/traces/train_trajectories.jsonl"
        "$reproduce_run/traces/train_trajectories.manifest.json"
        "$reproduce_run/traces/train_trajectories.manifest.json.sha256"
        "$eval_run/train.log" "$eval_run/resolved-config.yaml" "$eval_run/run.env"
        "$eval_run/traces/eval_predictions.jsonl"
        "$eval_run/traces/eval_predictions.manifest.json"
        "$eval_run/traces/eval_predictions.manifest.json.sha256"
        "$analysis_dir/summary.json" "$analysis_dir/summary.md"
        "$analysis_dir/go_no_go.json" "$analysis_dir/per_trajectory.jsonl"
        "$analysis_dir/per_question.jsonl"
    )
    for eval_key in "${eval_keys[@]}"; do
        for wandb_run in "${parent_eval_runs[$eval_key]}" \
            "${reproduced_eval_runs[$eval_key]}"; do
            evidence_files+=(
                "$wandb_run/train.log" "$wandb_run/resolved-config.yaml"
                "$wandb_run/run.env" "$wandb_run/traces/eval_predictions.jsonl"
                "$wandb_run/traces/eval_predictions.manifest.json"
                "$wandb_run/traces/eval_predictions.manifest.json.sha256"
            )
        done
        evidence_files+=(
            "${ar_dirs[$eval_key]}/summary.json"
            "${ar_dirs[$eval_key]}/summary.md"
            "${ar_dirs[$eval_key]}/paired_results.csv"
            "${ar_dirs[$eval_key]}/correct_questions.csv"
            "${ar_dirs[$eval_key]}/wrong_questions.csv"
            "${ar_dirs[$eval_key]}/search_transition.csv"
        )
    done
    if [[ "$branch_authorized" == true ]]; then
        control_config="$(file_sha256 "$control_run/resolved-config.yaml")" || return 1
        cost_config="$(file_sha256 "$cost_run/resolved-config.yaml")" || return 1
        control_contract="$(file_sha256 "$control_run/native-training-contract.json")" || return 1
        cost_contract="$(file_sha256 "$cost_run/native-training-contract.json")" || return 1
        branch_rows="B"$'\t'"control"$'\t'"$control_run"$'\t'"$control_checkpoint"$'\t'"$control_digest"$'\t'"$reproduce_checkpoint"$'\t'"$reproduce_digest"$'\t'"$commit"$'\t'"$handoff_digest"$'\t'"$data_digest"$'\t'"$control_config"$'\t'"$control_trace_digest"$'\t'"$control_trace_manifest_digest"$'\t'"$control_contract"$'\t'"$NATIVE_TRAIN_SMOKE_DIGEST"$'\n'\
"C"$'\t'"cost_aware_gated"$'\t'"$cost_run"$'\t'"$cost_checkpoint"$'\t'"$cost_digest"$'\t'"$reproduce_checkpoint"$'\t'"$reproduce_digest"$'\t'"$commit"$'\t'"$handoff_digest"$'\t'"$data_digest"$'\t'"$cost_config"$'\t'"$cost_trace_digest"$'\t'"$cost_trace_manifest_digest"$'\t'"$cost_contract"$'\t'"$NATIVE_TRAIN_SMOKE_DIGEST"$'\n'
        for eval_key in "${eval_keys[@]}"; do
            eval_stage="${eval_key^^}"
            branch_rows+="B-${eval_stage//_/-}-EVAL"$'\t'"control_${eval_key}_eval"$'\t'"${control_eval_runs[$eval_key]}"$'\t'"$control_checkpoint"$'\t'"$control_digest"$'\t'"$control_checkpoint"$'\t'"$control_digest"$'\t'"$commit"$'\t'"$handoff_digest"$'\t'"$data_digest"$'\t'"$(file_sha256 "${control_eval_runs[$eval_key]}/resolved-config.yaml")"$'\t'"${control_eval_trace_digests[$eval_key]}"$'\t'"${control_eval_trace_manifest_digests[$eval_key]}"$'\t-'$'\t'"$NATIVE_TRAIN_SMOKE_DIGEST"$'\n'
            branch_rows+="C-${eval_stage//_/-}-EVAL"$'\t'"cost_aware_gated_${eval_key}_eval"$'\t'"${cost_eval_runs[$eval_key]}"$'\t'"$cost_checkpoint"$'\t'"$cost_digest"$'\t'"$cost_checkpoint"$'\t'"$cost_digest"$'\t'"$commit"$'\t'"$handoff_digest"$'\t'"$data_digest"$'\t'"$(file_sha256 "${cost_eval_runs[$eval_key]}/resolved-config.yaml")"$'\t'"${cost_eval_trace_digests[$eval_key]}"$'\t'"${cost_eval_trace_manifest_digests[$eval_key]}"$'\t-'$'\t'"$NATIVE_TRAIN_SMOKE_DIGEST"$'\n'
        done
        printf '%s' "$branch_rows" >>"$results_dir/lineage.tsv"
        evidence_files+=(
            "$control_run/train.log" "$control_run/resolved-config.yaml"
            "$control_run/run.env" "$control_run/lineage.tsv"
            "$control_run/native-training-contract.json"
            "$control_run/traces/train_trajectories.jsonl"
            "$control_run/traces/train_trajectories.manifest.json"
            "$control_run/traces/train_trajectories.manifest.json.sha256"
            "$cost_run/train.log" "$cost_run/resolved-config.yaml" "$cost_run/run.env"
            "$cost_run/lineage.tsv" "$cost_run/native-training-contract.json"
            "$cost_run/traces/train_trajectories.jsonl"
            "$cost_run/traces/train_trajectories.manifest.json"
            "$cost_run/traces/train_trajectories.manifest.json.sha256"
        )
        for eval_key in "${eval_keys[@]}"; do
            control_eval_run="${control_eval_runs[$eval_key]}"
            cost_eval_run="${cost_eval_runs[$eval_key]}"
            evidence_files+=(
                "$control_eval_run/train.log" "$control_eval_run/resolved-config.yaml"
                "$control_eval_run/run.env"
                "$control_eval_run/traces/eval_predictions.jsonl"
                "$control_eval_run/traces/eval_predictions.manifest.json"
                "$control_eval_run/traces/eval_predictions.manifest.json.sha256"
                "$cost_eval_run/train.log" "$cost_eval_run/resolved-config.yaml"
                "$cost_eval_run/run.env"
                "$cost_eval_run/traces/eval_predictions.jsonl"
                "$cost_eval_run/traces/eval_predictions.manifest.json"
                "$cost_eval_run/traces/eval_predictions.manifest.json.sha256"
                "${paired_dirs[$eval_key]}/summary.json"
                "${paired_dirs[$eval_key]}/summary.md"
                "${paired_dirs[$eval_key]}/paired_results.csv"
                "${paired_dirs[$eval_key]}/correct_questions.csv"
                "${paired_dirs[$eval_key]}/wrong_questions.csv"
                "${paired_dirs[$eval_key]}/search_transition.csv"
            )
        done
    fi
    wandb_runs=("$reproduce_run" "$eval_run")
    for eval_key in "${eval_keys[@]}"; do
        wandb_runs+=("${parent_eval_runs[$eval_key]}" "${reproduced_eval_runs[$eval_key]}")
    done
    if [[ "$branch_authorized" == true ]]; then
        wandb_runs+=("$control_run" "$cost_run")
        for eval_key in "${eval_keys[@]}"; do
            wandb_runs+=("${control_eval_runs[$eval_key]}" "${cost_eval_runs[$eval_key]}")
        done
    fi
    for wandb_run in "${wandb_runs[@]}"; do
        [[ -d "$wandb_run/wandb" && ! -L "$wandb_run/wandb" ]] || {
            printf 'Run-specific WandB directory is missing: %s\n' "$wandb_run" >&2
            return 1
        }
        wandb_count=0
        wandb_history_count=0
        while IFS= read -r -d '' wandb_file; do
            evidence_files+=("$wandb_file")
            ((wandb_count += 1))
            if [[ "$wandb_file" == *.wandb ]]; then
                ((wandb_history_count += 1))
            fi
        done < <(find "$wandb_run/wandb" -type f -size +0c -print0 | LC_ALL=C sort -z)
        ((wandb_count > 0)) || {
            printf 'Run-specific WandB history is empty: %s\n' "$wandb_run" >&2
            return 1
        }
        ((wandb_history_count > 0)) || {
            printf 'Run-specific WandB has no nonempty *.wandb history: %s\n' \
                "$wandb_run" >&2
            return 1
        }
    done
    index_content="$(cat "$outer_attempt/native-training-runs.tsv")"
    atomic_write "$results_dir/run-index.tsv" "$index_content"$'\n'
    evidence_files+=("$results_dir/run-index.tsv")
    sync_path "$results_dir"

    verify_checkout "$commit" || return $?
    [[ "$(file_sha256 "$NATIVE_TRAIN_MANIFEST")" == "$data_digest" ]] || return 1
    verify_protocol_gate_evidence "$NATIVE_PROTOCOL_GATE_EVIDENCE" "$base_model" \
        "$base_digest" "$commit" "$handoff_digest" "$data_digest" || return $?
    verify_smoke_evidence "$NATIVE_SMOKE_EVIDENCE" "$base_model" "$base_digest" \
        "$commit" "$handoff_digest" "$data_digest" \
        "$NATIVE_TRAIN_PROTOCOL_GATE_DIGEST" || return $?
    verify_native_checkpoint_digest "$base_model" "$base_digest" || return $?
    verify_native_checkpoint_digest "$reproduce_checkpoint" "$reproduce_digest" || return $?
    if [[ "$branch_authorized" == true ]]; then
        verify_native_checkpoint_digest "$control_checkpoint" "$control_digest" || return $?
        verify_native_checkpoint_digest "$cost_checkpoint" "$cost_digest" || return $?
    fi
    publish_native_training_evidence "$outer_attempt" "$results_dir" \
        qwen-native-training-main "$NATIVE_TRAIN_MAIN_CONTRACT" \
        "${evidence_files[@]}" || return $?
    if [[ "$branch_authorized" == true ]]; then
        printf 'Completed Qwen native R60/G3/B20/C20 main training: %s\n' "$results_dir"
    else
        printf 'R-G3 did not authorize B/C; sealed the scientific result without branch training: %s\n' \
            "$results_dir"
    fi
}

qwen_native_train_pipeline() {
    local outer_attempt="$1" commit="$2" handoff_digest="$3"
    local base_model="$4" base_digest="$5" data_digest
    require_qwen_native_train || return $?
    data_digest="$(file_sha256 "$NATIVE_TRAIN_MANIFEST")" || return 1
    verify_protocol_gate_evidence "$NATIVE_PROTOCOL_GATE_EVIDENCE" "$base_model" \
        "$base_digest" "$commit" "$handoff_digest" "$data_digest" || return $?
    if [[ "$NATIVE_TRAIN_STAGE" == main ]]; then
        verify_smoke_evidence "$NATIVE_SMOKE_EVIDENCE" "$base_model" "$base_digest" \
            "$commit" "$handoff_digest" "$data_digest" \
            "$NATIVE_TRAIN_PROTOCOL_GATE_DIGEST" || return $?
    fi
    verify_native_training_preflight_state "$commit" "$handoff_digest" \
        "$base_digest" "$data_digest" || return $?
    case "$NATIVE_TRAIN_STAGE" in
        smoke)
            qwen_native_smoke_pipeline "$outer_attempt" "$commit" "$handoff_digest" \
                "$base_model" "$base_digest" "$data_digest"
            ;;
        main)
            qwen_native_main_pipeline "$outer_attempt" "$commit" "$handoff_digest" \
                "$base_model" "$base_digest" "$data_digest"
            ;;
    esac
}

qwen_native_train_main() {
    case "${1:-}" in
        --worker)
            phase_worker gpu "${2:?missing attempt directory}" "$0"
            ;;
        --action)
            require_qwen_native_train
            gpu_action "${2:?missing attempt directory}"
            ;;
        '')
            require_qwen_native_train
            phase_launch gpu "$0"
            ;;
        *)
            printf 'Usage: QWEN_NATIVE_TRAIN_STAGE={smoke|main} QWEN_NATIVE_PROTOCOL_GATE_EVIDENCE=<g0_g1-marker> [QWEN_NATIVE_SMOKE_EVIDENCE=<marker>] GPU_COUNT=2 AUTODL_PRICE_PER_HOUR=<price> bash %s\n' "$0" >&2
            exit 64
            ;;
    esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    qwen_native_train_main "$@"
fi
