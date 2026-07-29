#!/usr/bin/env bash
set -Eeuo pipefail

# Continue one already sealed R60 run without changing its checkout or CPU handoff.
G3_PROJECT_ROOT="${AUTODL_ROOT:-/root/autodl-tmp/search-r1}"
G3_SOURCE_ENTRY="$G3_PROJECT_ROOT/checkout/scripts/autodl/09_gpu_qwen_native_train.sh"
G3_RUNNER_PATH="$(readlink -f -- "${BASH_SOURCE[0]}")"
export QWEN_NATIVE_TRAIN_STAGE=main

[[ -f "$G3_SOURCE_ENTRY" && ! -L "$G3_SOURCE_ENTRY" ]] || {
    printf 'Missing sealed Qwen native training entrypoint: %s\n' "$G3_SOURCE_ENTRY" >&2
    exit 1
}

# shellcheck source=09_gpu_qwen_native_train.sh
source "$G3_SOURCE_ENTRY"

readonly G3_ONLY_CONTRACT=qwen-native-training-g3-only-v1
readonly G3_ONLY_NAMESPACE=qwen-native-training-g3-only
readonly G3_CPU_CONTRACT=qwen-native-g3-only-cpu-v1
readonly G3_ONLY_TERMINAL_CODE=201
readonly G3_R60_ATTEMPT=20260728T092026Z-2906-16060
readonly G3_EXPECTED_R60_MARKER="$MANIFEST_DIR/qwen-native-training-r60-only/$G3_R60_ATTEMPT.ok"
readonly G3_EXPECTED_R60_DIGEST=2366ce2da28b530f12af30a22d3dc3e2bd33fedbfcb3a2bed5acdd9d0537ffed
readonly G3_EXPECTED_CHECKOUT=f8c1cd7e87078d07385f74ca8710add5d5f79c06
readonly G3_EXPECTED_HANDOFF=9ffaa89f88990887b368750ccfdc559d93afdb9a1f5719411cf3e3c1147da90b
readonly G3_EXPECTED_BASE=bc67be20efb353ba14d9c1b291a64410afec94f2310e94b59b6b76047b164e78
readonly G3_EXPECTED_DATA=db6cacf865365ee977fc280d5195556e9903907b28b228ba8f3d72133d674167
readonly G3_EXPECTED_CHECKPOINT=583771b131b6e2aa663ee2ef13ea6421524cdd7fed839f35fd6f9eb246a2c231
readonly G3_EXPECTED_TRAIN_TRACE=cfcb118cb16dead6bb77e4f072b1dbc0009bfaf7257bf463c7a04a9c7f85bde4
readonly G3_EXPECTED_TRAIN_TRACE_MANIFEST=932e5e6d72823d55029517f9adf62990fdee8609d3783f3e61477f76cb3c5e24
readonly G3_EXPECTED_GATE=168bd35d6c72d2f00aa03c6e9490f8cc5a601d005729369982214ea0ab1839a5
readonly G3_EXPECTED_SMOKE=dd6f8c496a1ee415f382a68d64aef6311aec8ca5617e375ce164e2ef5e8c3014
readonly G3_EXPECTED_G3_DATA=b8de9f20ba2eb41184578f543fd6be988866617702811300be079ece74ae47d1
readonly G3_CONFIG="$MANIFEST_DIR/config-2gpu-qwen_native_g3-eval.yaml"
readonly G3_EXPECTED_CONFIG=a730954407dbe730a8b3253079986c33ae203ac19081c34535676636f13e041e
readonly G3_DEPLOY_PATH="$G3_PROJECT_ROOT/operator/11_gpu_qwen_native_g3_only.sh"
readonly G3_CPU_RECEIPT_ROOT="$MANIFEST_DIR/qwen-native-g3-only-cpu"

G3_R60_RESULTS=''
G3_R60_OUTER=''
G3_R60_MANIFEST=''
G3_R60_DIGEST=''
G3_R60_RUN=''
G3_R60_CHECKPOINT=''
G3_R60_CHECKPOINT_DIGEST=''
G3_R60_TRACE_DIGEST=''
G3_R60_TRACE_MANIFEST_DIGEST=''
G3_R60_GATE_MARKER=''
G3_R60_GATE_DIGEST=''
G3_R60_SMOKE_MARKER=''
G3_R60_SMOKE_DIGEST=''
G3_CPU_RECEIPT=''
G3_CPU_RECEIPT_DIGEST=''
G3_PREFLIGHT_COMMIT=''
G3_PREFLIGHT_HANDOFF=''
G3_PREFLIGHT_BASE_DIGEST=''
G3_PREFLIGHT_DATA_DIGEST=''
G3_PREFLIGHT_RUNNER_DIGEST=''

g3_regular_value() {
    local path="$1"
    [[ -f "$path" && ! -L "$path" ]] || {
        printf 'Missing or symlinked evidence file: %s\n' "$path" >&2
        return 1
    }
    tr -d '\r\n' <"$path"
}

verify_r60_only_evidence() {
    local marker="$1" expected_commit="$2" expected_handoff="$3"
    local expected_base="$4" expected_base_digest="$5" expected_data="$6"
    local marker_digest results outer evidence metadata fixed trace_identity
    local checkpoint_digest trace_digest trace_manifest_digest
    local gate_marker gate_digest smoke_marker smoke_digest

    [[ "$marker" == "$G3_EXPECTED_R60_MARKER" &&
        "$(dirname -- "$marker")" == "$MANIFEST_DIR/qwen-native-training-r60-only" &&
        "$(basename -- "$marker" .ok)" == "$G3_R60_ATTEMPT" &&
        -f "$marker" && ! -L "$marker" ]] || {
        printf 'G3 requires the exact sealed R60 marker: %s\n' \
            "$G3_EXPECTED_R60_MARKER" >&2
        return 1
    }
    results="$NATIVE_TRAIN_RESULTS_ROOT/attempts/$G3_R60_ATTEMPT"
    outer="$ATTEMPTS_ROOT/gpu/$G3_R60_ATTEMPT"
    evidence="$results/evidence.sha256"
    [[ -d "$results" && ! -L "$results" && -d "$outer" && ! -L "$outer" &&
        -f "$evidence" && ! -L "$evidence" &&
        -f "$outer/.failed" && ! -L "$outer/.failed" &&
        ! -e "$outer/.success" && ! -L "$outer/.success" &&
        ! -e "$outer/.starting" && ! -L "$outer/.starting" &&
        ! -e "$outer/.running" && ! -L "$outer/.running" &&
        "$(g3_regular_value "$outer/terminal")" == failed &&
        "$(g3_regular_value "$outer/exit-code")" == 200 ]] || {
        printf 'R60 outer attempt is not the controlled failed/200 terminal: %s\n' \
            "$outer" >&2
        return 1
    }
    marker_digest="$(g3_regular_value "$marker")" || return 1
    [[ "$marker_digest" == "$G3_EXPECTED_R60_DIGEST" &&
        "$(file_sha256 "$evidence")" == "$marker_digest" ]] || {
        printf 'R60 marker does not bind the expected evidence manifest.\n' >&2
        return 1
    }
    verify_native_training_checksum_manifest "$evidence" || {
        printf 'R60 evidence checksum replay failed: %s\n' "$evidence" >&2
        return 1
    }
    [[ "$(g3_regular_value "$outer/result-contract")" == qwen-native-training-r60-only-v1 &&
        "$(g3_regular_value "$outer/result-root")" == "$results" &&
        "$(g3_regular_value "$outer/evidence-marker")" == "$marker" &&
        "$(g3_regular_value "$outer/evidence-digest")" == "$marker_digest" ]] || {
        printf 'R60 outer attempt bindings are inconsistent.\n' >&2
        return 1
    }

    metadata="$("$TRAIN_ENV/bin/python" - \
        "$results" "$outer" "$G3_PROJECT_ROOT" "$expected_commit" \
        "$expected_handoff" "$expected_base" "$expected_base_digest" \
        "$expected_data" "$G3_EXPECTED_CHECKPOINT" \
        "$G3_EXPECTED_TRAIN_TRACE" "$G3_EXPECTED_TRAIN_TRACE_MANIFEST" \
        "$G3_EXPECTED_GATE" "$G3_EXPECTED_SMOKE" <<'PY'
import csv
import hashlib
import json
from pathlib import Path
import sys

(results_raw, outer_raw, root_raw, expected_commit, expected_handoff,
 expected_base_raw, expected_base_digest, expected_data, expected_checkpoint,
 expected_trace, expected_trace_manifest, expected_gate, expected_smoke) = sys.argv[1:]
results = Path(results_raw)
outer = Path(outer_raw)
root = Path(root_raw)
expected_base = Path(expected_base_raw)

def regular(path):
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"missing or symlinked R60 evidence: {path}")
    return path

def digest(path):
    return hashlib.sha256(regular(path).read_bytes()).hexdigest()

def env(path):
    values = {}
    for line in regular(path).read_text(encoding="utf-8").splitlines():
        if not line or "=" not in line:
            raise SystemExit(f"malformed env evidence: {path}")
        key, value = line.split("=", 1)
        if key in values:
            raise SystemExit(f"duplicate env key: {key}")
        values[key] = value
    return values

contract = env(results / "contract.env")
expected_contract = {
    "schema": "qwen-native-training-r60-only-v1",
    "stage": "r60_only",
    "stage_order": "R60",
    "decision": "COMPLETE",
    "branch_training_authorized": "false",
    "controlled_outer_exit_code": "200",
    "protocol_gate_evidence": str(root / "manifests/qwen-native-gate/20260728T044634Z-2051-4045.ok"),
    "protocol_gate_evidence_sha256": expected_gate,
    "smoke_evidence": str(root / "manifests/qwen-native-training-smoke/20260728T061246Z-1316-15067.ok"),
    "smoke_evidence_sha256": expected_smoke,
}
if contract != expected_contract:
    raise SystemExit("R60-only contract is not exact")

with regular(results / "lineage.tsv").open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    rows = list(reader)
expected_fields = [
    "stage", "role", "run_dir", "checkpoint", "checkpoint_digest",
    "parent_checkpoint", "parent_checkpoint_digest", "checkout_commit",
    "cpu_handoff_digest", "data_manifest_sha256", "resolved_config_sha256",
    "trace_sha256", "trace_manifest_sha256", "run_contract_sha256",
    "predecessor_evidence_sha256",
]
if reader.fieldnames != expected_fields or len(rows) != 1:
    raise SystemExit("R60 lineage must contain exactly one R row")
row = rows[0]
run = root / "runs/reproduce/attempts/20260728T092332Z-2946-8453"
checkpoint = run / "checkpoints/actor/global_step_60"
checks = {
    "stage": row["stage"] == "R",
    "role": row["role"] == "reproduced",
    "run": row["run_dir"] == str(run),
    "checkpoint": row["checkpoint"] == str(checkpoint),
    "checkpoint_digest": row["checkpoint_digest"] == expected_checkpoint,
    "parent": row["parent_checkpoint"] == str(expected_base),
    "parent_digest": row["parent_checkpoint_digest"] == expected_base_digest,
    "commit": row["checkout_commit"] == expected_commit,
    "handoff": row["cpu_handoff_digest"] == expected_handoff,
    "data": row["data_manifest_sha256"] == expected_data,
    "config": row["resolved_config_sha256"] == digest(run / "resolved-config.yaml"),
    "trace": row["trace_sha256"] == expected_trace,
    "trace_manifest": row["trace_manifest_sha256"] == expected_trace_manifest,
    "run_contract": row["run_contract_sha256"] == digest(run / "native-training-contract.json"),
    "predecessor": row["predecessor_evidence_sha256"] == expected_smoke,
}
failed = sorted(key for key, passed in checks.items() if not passed)
if failed:
    raise SystemExit("R60 lineage mismatch: " + ", ".join(failed))

with regular(results / "run-index.tsv").open(newline="", encoding="utf-8") as handle:
    index_reader = csv.DictReader(handle, delimiter="\t")
    index = list(index_reader)
if (index_reader.fieldnames != ["stage", "role", "run_dir"] or len(index) != 1 or
        index[0] != {"stage": "R", "role": "reproduced", "run_dir": str(run)}):
    raise SystemExit("R60 run index is not the one-row R contract")

complete = env(outer / "r60-only-complete.env")
tree = env(results / "checkpoint-tree.env")
expected_complete = {
    "schema": "qwen-native-training-r60-only-v1",
    "run_dir": str(run),
    "checkpoint": str(checkpoint),
    "checkpoint_tree_sha256": expected_checkpoint,
    "trace_sha256": expected_trace,
    "runner": str(root / "operator/10_gpu_qwen_native_r60_only.sh"),
    "runner_sha256": "d9465a64710af031a066016f286d4ff396a7eeda16b185b463dab717030f07b3",
    "next_stage_requires_manual_approval": "true",
}
if complete != expected_complete:
    raise SystemExit("R60 complete receipt is not exact")
if tree != {"checkpoint": str(checkpoint), "checkpoint_tree_sha256": expected_checkpoint}:
    raise SystemExit("R60 checkpoint-tree receipt disagrees with lineage")

for name, wanted in (("terminal", "success"), ("exit-code", "0")):
    if regular(run / name).read_text(encoding="utf-8").strip() != wanted:
        raise SystemExit("inner R60 run is not success/0")
if not (run / ".success").is_file() or (run / ".success").is_symlink():
    raise SystemExit("inner R60 .success is missing")
if any((run / name).exists() or (run / name).is_symlink()
       for name in (".failed", ".running")):
    raise SystemExit("inner R60 has a conflicting terminal marker")

native = json.loads(regular(run / "native-training-contract.json").read_bytes())
native_checks = {
    "schema": native.get("schema") == 4,
    "role": native.get("role") == "reproduced",
    "variant": native.get("variant") == "reproduce",
    "steps": native.get("steps") == 60,
    "batch": native.get("train_batch_size") == 8,
    "group": native.get("group_size") == 5,
    "turns": native.get("max_action_budget") == 4,
    "response": native.get("max_response_length") == 500,
    "observation": native.get("max_obs_length") == 500,
    "checkpoint": native.get("checkpoint") == str(checkpoint),
    "checkpoint_digest": native.get("checkpoint_digest") == expected_checkpoint,
    "parent": native.get("parent") == str(expected_base),
    "parent_digest": native.get("parent_digest") == expected_base_digest,
    "config": native.get("config_sha256") == row["resolved_config_sha256"],
    "cost_lambda": native.get("cost_lambda") == 0.0,
    "cost_mode": native.get("cost_reward_mode") == "linear",
}
failed = sorted(key for key, passed in native_checks.items() if not passed)
if failed:
    raise SystemExit("R60 run contract mismatch: " + ", ".join(failed))

gate_marker = Path(contract["protocol_gate_evidence"])
smoke_marker = Path(contract["smoke_evidence"])
if regular(gate_marker).read_text(encoding="utf-8").strip() != expected_gate:
    raise SystemExit("R60 protocol-gate marker digest mismatch")
if regular(smoke_marker).read_text(encoding="utf-8").strip() != expected_smoke:
    raise SystemExit("R60 smoke marker digest mismatch")
print(run, checkpoint, expected_checkpoint, expected_trace, expected_trace_manifest,
      gate_marker, expected_gate, smoke_marker, expected_smoke, sep="\t")
PY
    )" || return 1
    IFS=$'\t' read -r G3_R60_RUN G3_R60_CHECKPOINT checkpoint_digest \
        trace_digest trace_manifest_digest gate_marker gate_digest \
        smoke_marker smoke_digest <<<"$metadata"
    [[ -n "$G3_R60_RUN" && -n "$G3_R60_CHECKPOINT" ]] || return 1

    fixed="$(fixed_checkpoint "$G3_R60_RUN" "$REPRODUCE_STEPS")" || return 1
    [[ "$fixed" == "$G3_R60_CHECKPOINT" ]] || return 1
    verify_native_checkpoint_digest "$G3_R60_CHECKPOINT" "$checkpoint_digest" || return $?
    trace_identity="$(verify_native_trace_identity \
        "$G3_R60_RUN/traces/train_trajectories.manifest.json" 2400)" || return 1
    IFS=$'\t' read -r G3_R60_TRACE_DIGEST G3_R60_TRACE_MANIFEST_DIGEST \
        <<<"$trace_identity"
    [[ "$G3_R60_TRACE_DIGEST" == "$trace_digest" &&
        "$G3_R60_TRACE_MANIFEST_DIGEST" == "$trace_manifest_digest" ]] || {
        printf 'R60 2,400-row trace identity drifted.\n' >&2
        return 1
    }
    verify_wandb_receipt "$G3_R60_RUN" "$REPRODUCE_STEPS" || return $?
    verify_protocol_gate_evidence "$gate_marker" "$expected_base" \
        "$expected_base_digest" "$expected_commit" "$expected_handoff" \
        "$expected_data" || return $?
    [[ "$NATIVE_TRAIN_PROTOCOL_GATE_DIGEST" == "$gate_digest" ]] || return 1
    verify_smoke_evidence "$smoke_marker" "$expected_base" \
        "$expected_base_digest" "$expected_commit" "$expected_handoff" \
        "$expected_data" "$gate_digest" || return $?
    [[ "$NATIVE_TRAIN_SMOKE_DIGEST" == "$smoke_digest" ]] || return 1

    G3_R60_RESULTS="$results"
    G3_R60_OUTER="$outer"
    G3_R60_MANIFEST="$evidence"
    G3_R60_DIGEST="$marker_digest"
    G3_R60_CHECKPOINT_DIGEST="$checkpoint_digest"
    G3_R60_GATE_MARKER="$gate_marker"
    G3_R60_GATE_DIGEST="$gate_digest"
    G3_R60_SMOKE_MARKER="$smoke_marker"
    G3_R60_SMOKE_DIGEST="$smoke_digest"
}

g3_cpu_receipt_content() {
    local runner_digest="$1" g3_data_digest="$2" config_digest="$3"
    printf '%s' \
"schema=$G3_CPU_CONTRACT
r60_evidence=$G3_EXPECTED_R60_MARKER
r60_evidence_sha256=$G3_R60_DIGEST
r60_attempt=$G3_R60_ATTEMPT
r60_checkpoint=$G3_R60_CHECKPOINT
r60_checkpoint_tree_sha256=$G3_R60_CHECKPOINT_DIGEST
r60_trace_sha256=$G3_R60_TRACE_DIGEST
r60_trace_manifest_sha256=$G3_R60_TRACE_MANIFEST_DIGEST
checkout_commit=$G3_EXPECTED_CHECKOUT
cpu_handoff_digest=$G3_EXPECTED_HANDOFF
base_model_sha256=$G3_EXPECTED_BASE
data_manifest_sha256=$G3_EXPECTED_DATA
g3_data=$NATIVE_TRAIN_G3_DATA
g3_data_sha256=$g3_data_digest
g3_resolved_config=$G3_CONFIG
g3_resolved_config_sha256=$config_digest
g3_rows=64
g3_group_size=5
g3_trajectory_count=320
runner=$G3_DEPLOY_PATH
runner_sha256=$runner_digest
gpu_evaluation_authorized=false
manual_gpu_start_required=true
"
}

verify_g3_foundation() {
    local commit="$1" handoff_digest="$2" base_digest="$3"
    local base_model="$4" recorded sidecar data_digest g3_data_digest config_digest
    export PYTHONPATH="$CHECKOUT_DIR${PYTHONPATH:+:$PYTHONPATH}"
    [[ ! -e "$MANIFEST_DIR/cpu-seal.pending.json" &&
       ! -L "$MANIFEST_DIR/cpu-seal.pending.json" ]] || {
        printf 'CPU seal transaction is unresolved.\n' >&2
        return 1
    }
    [[ "$commit" == "$G3_EXPECTED_CHECKOUT" &&
        "$handoff_digest" == "$G3_EXPECTED_HANDOFF" &&
        "$base_digest" == "$G3_EXPECTED_BASE" ]] || {
        printf 'Frozen checkout, handoff, or base-model identity changed.\n' >&2
        return 1
    }
    verify_checkout "$commit" || return $?
    [[ -f "$MANIFEST_DIR/cpu.ok" && ! -L "$MANIFEST_DIR/cpu.ok" &&
        -f "$HANDOFF" && ! -L "$HANDOFF" &&
        -f "$HANDOFF.sha256" && ! -L "$HANDOFF.sha256" ]] || return 1
    recorded="$(g3_regular_value "$MANIFEST_DIR/cpu.ok")" || return 1
    sidecar="$(awk 'NR == 1 {print $1} NR > 1 {exit 1}' "$HANDOFF.sha256")" || return 1
    [[ "$recorded" == "$handoff_digest" && "$sidecar" == "$handoff_digest" &&
        "$(file_sha256 "$HANDOFF")" == "$handoff_digest" ]] || {
        printf 'Frozen CPU handoff digest is inconsistent.\n' >&2
        return 1
    }
    verify_qwen_native_train_data || return $?
    data_digest="$(file_sha256 "$NATIVE_TRAIN_MANIFEST")" || return 1
    g3_data_digest="$(file_sha256 "$NATIVE_TRAIN_G3_DATA")" || return 1
    config_digest="$(file_sha256 "$G3_CONFIG")" || return 1
    [[ "$data_digest" == "$G3_EXPECTED_DATA" &&
        "$g3_data_digest" == "$G3_EXPECTED_G3_DATA" &&
        "$config_digest" == "$G3_EXPECTED_CONFIG" ]] || {
        printf 'Frozen G3 data or resolved config changed.\n' >&2
        return 1
    }
    verify_native_checkpoint_digest "$base_model" "$base_digest" || return $?
    verify_r60_only_evidence "$QWEN_NATIVE_R60_EVIDENCE" "$commit" \
        "$handoff_digest" "$base_model" "$base_digest" "$data_digest" || return $?
}

expected_g3_cpu_receipt() {
    local runner_digest="$1"
    printf '%s/%s-%s.env\n' "$G3_CPU_RECEIPT_ROOT" "$G3_R60_ATTEMPT" "$runner_digest"
}

require_g3_cpu_receipt_root() {
    local create="${1:-false}" canonical_manifest canonical_root
    if [[ -e "$G3_CPU_RECEIPT_ROOT" || -L "$G3_CPU_RECEIPT_ROOT" ]]; then
        [[ -d "$G3_CPU_RECEIPT_ROOT" && ! -L "$G3_CPU_RECEIPT_ROOT" ]] || return 1
    elif [[ "$create" == true ]]; then
        mkdir "$G3_CPU_RECEIPT_ROOT"
    else
        return 1
    fi
    canonical_manifest="$(readlink -f -- "$MANIFEST_DIR")" || return 1
    canonical_root="$(readlink -f -- "$G3_CPU_RECEIPT_ROOT")" || return 1
    [[ "$canonical_root" == "$canonical_manifest/qwen-native-g3-only-cpu" ]]
}

verify_g3_cpu_receipt() {
    local runner_digest="$1" receipt content digest sidecar expected_sidecar
    local g3_data_digest config_digest
    require_g3_cpu_receipt_root false || return 1
    g3_data_digest="$(file_sha256 "$NATIVE_TRAIN_G3_DATA")" || return 1
    config_digest="$(file_sha256 "$G3_CONFIG")" || return 1
    receipt="$(expected_g3_cpu_receipt "$runner_digest")" || return 1
    content="$(g3_cpu_receipt_content "$runner_digest" "$g3_data_digest" \
        "$config_digest")"$'\n'
    [[ -f "$receipt" && ! -L "$receipt" &&
        -f "$receipt.sha256" && ! -L "$receipt.sha256" ]] || {
        printf 'Exact G3 CPU receipt is missing: %s\n' "$receipt" >&2
        return 1
    }
    cmp -s <(printf '%s' "$content") "$receipt" || {
        printf 'G3 CPU receipt content does not match current frozen inputs.\n' >&2
        return 1
    }
    digest="$(file_sha256 "$receipt")" || return 1
    expected_sidecar="$digest  ${receipt#"$G3_PROJECT_ROOT/"}"
    sidecar="$(g3_regular_value "$receipt.sha256")" || return 1
    [[ "$sidecar" == "$expected_sidecar" ]] || {
        printf 'G3 CPU receipt checksum sidecar is inconsistent.\n' >&2
        return 1
    }
    G3_CPU_RECEIPT="$receipt"
    G3_CPU_RECEIPT_DIGEST="$digest"
}

prepare_g3_cpu_receipt() {
    local commit handoff_digest base_model base_digest runner_digest
    local receipt content digest sidecar_content g3_data_digest config_digest
    [[ "$G3_RUNNER_PATH" == "$G3_DEPLOY_PATH" &&
        -f "$G3_RUNNER_PATH" && ! -L "$G3_RUNNER_PATH" ]] || {
        printf 'Deploy this exact runner outside checkout before CPU preparation: %s\n' \
            "$G3_DEPLOY_PATH" >&2
        return 1
    }
    [[ -n "${QWEN_NATIVE_R60_EVIDENCE:-}" ]] || {
        printf 'Set QWEN_NATIVE_R60_EVIDENCE to the exact R60 marker.\n' >&2
        return 64
    }
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    export WANDB_MODE=offline
    export PIP_NO_INDEX=1
    export PYTHONDONTWRITEBYTECODE=1
    export PYTHONPATH="$CHECKOUT_DIR${PYTHONPATH:+:$PYTHONPATH}"
    commit="$(expected_commit)" || return 1
    handoff_digest="$(g3_regular_value "$MANIFEST_DIR/cpu.ok")" || return 1
    base_model="$(readlink -f -- "$MODEL_DIR")" || return 1
    base_digest="$(tree_sha256 "$base_model")" || return 1
    verify_g3_foundation "$commit" "$handoff_digest" "$base_digest" \
        "$base_model" || return $?
    runner_digest="$(file_sha256 "$G3_RUNNER_PATH")" || return 1
    g3_data_digest="$(file_sha256 "$NATIVE_TRAIN_G3_DATA")" || return 1
    config_digest="$(file_sha256 "$G3_CONFIG")" || return 1
    receipt="$(expected_g3_cpu_receipt "$runner_digest")" || return 1
    content="$(g3_cpu_receipt_content "$runner_digest" "$g3_data_digest" \
        "$config_digest")"$'\n'
    verify_checkout "$commit" || return $?
    [[ "$(file_sha256 "$G3_RUNNER_PATH")" == "$runner_digest" ]] || return 1
    require_g3_cpu_receipt_root true || return 1
    if [[ -e "$receipt" || -L "$receipt" ]]; then
        [[ -f "$receipt" && ! -L "$receipt" ]] || return 1
        cmp -s <(printf '%s' "$content") "$receipt" || {
            printf 'Refusing to overwrite a differing G3 CPU receipt: %s\n' \
                "$receipt" >&2
            return 1
        }
    else
        atomic_write "$receipt" "$content"
    fi
    digest="$(file_sha256 "$receipt")" || return 1
    sidecar_content="$digest  ${receipt#"$G3_PROJECT_ROOT/"}"$'\n'
    if [[ -e "$receipt.sha256" || -L "$receipt.sha256" ]]; then
        [[ -f "$receipt.sha256" && ! -L "$receipt.sha256" ]] || return 1
        cmp -s <(printf '%s' "$sidecar_content") "$receipt.sha256" || {
            printf 'Refusing to overwrite a differing G3 CPU receipt checksum.\n' >&2
            return 1
        }
    else
        atomic_write "$receipt.sha256" "$sidecar_content"
    fi
    sync_path "$G3_CPU_RECEIPT_ROOT"
    verify_g3_cpu_receipt "$runner_digest" || return $?
    printf 'G3 CPU preparation complete: %s\n' "$G3_CPU_RECEIPT"
    printf 'Receipt SHA-256: %s\n' "$G3_CPU_RECEIPT_DIGEST"
    printf 'GPU evaluation was not started.\n'
}

# Override the main-v5 admission callback: G3 consumes R60 evidence, not smoke.
require_qwen_native_train() {
    validate_gpu_inputs || return $?
    [[ "$GPU_COUNT" == 2 && "$TRAIN_BATCH_SIZE" == 8 &&
        "$MAX_RESPONSE_LENGTH" == 500 && "$TOOL_PROTOCOL" == qwen35_native &&
        "$DATA_DIR" == "$NATIVE_TRAIN_DATA_DIR" &&
        "$EVAL_GROUP_SIZE" == 1 && -z "$EVAL_DATA_FILE" &&
        "$RUN_BUDGET_PROFILE" == gated_followup &&
        "${QWEN_NATIVE_R60_EVIDENCE:-}" == "$G3_EXPECTED_R60_MARKER" ]] || {
        printf 'G3-only requires the exact R60 marker, two GPUs, batch 8, response 500, and native-v4 settings.\n' >&2
        return 64
    }
    [[ "$G3_RUNNER_PATH" == "$G3_DEPLOY_PATH" &&
        -f "$G3_RUNNER_PATH" && ! -L "$G3_RUNNER_PATH" ]] || return 1
}

qwen_native_train_preflight() {
    local commit="$1" handoff_digest="$2" base_digest="$3"
    local base_model runner_digest data_digest
    base_model="$(readlink -f -- "$MODEL_DIR")" || return 1
    verify_g3_foundation "$commit" "$handoff_digest" "$base_digest" \
        "$base_model" || return $?
    runner_digest="$(file_sha256 "$G3_RUNNER_PATH")" || return 1
    verify_g3_cpu_receipt "$runner_digest" || return $?
    data_digest="$(file_sha256 "$NATIVE_TRAIN_MANIFEST")" || return 1
    G3_PREFLIGHT_COMMIT="$commit"
    G3_PREFLIGHT_HANDOFF="$handoff_digest"
    G3_PREFLIGHT_BASE_DIGEST="$base_digest"
    G3_PREFLIGHT_DATA_DIGEST="$data_digest"
    G3_PREFLIGHT_RUNNER_DIGEST="$runner_digest"
}

write_g3_only_branch_decision() {
    local analysis_decision="$1" contrast_count="$2"
    "$TRAIN_ENV/bin/python" - "$analysis_decision" "$contrast_count" <<'PY'
import json
import sys

decision = sys.argv[1]
count = int(sys.argv[2])
candidate = decision == "GO" and count >= 8
payload = {
    "analysis_decision": decision,
    "branch_candidate_passed": candidate,
    "branch_training_authorized": False,
    "cost_contrast_group_count": count,
    "cost_contrast_group_minimum": 8,
    "decision": "CANDIDATE-GO" if candidate else "NO-GO",
    "manual_review_required": True,
    "schema": "qwen-native-g3-only-branch-decision-v1",
}
print(json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
PY
}

verify_g3_runtime_config() {
    local runtime_config="$1" eval_run="$2"
    "$TRAIN_ENV/bin/python" - "$G3_CONFIG" "$runtime_config" \
        "$G3_R60_CHECKPOINT" "$G3_R60_CHECKPOINT_DIGEST" "$eval_run" <<'PY'
from copy import deepcopy
from pathlib import Path
import sys

from omegaconf import OmegaConf

template_path, runtime_path, checkpoint_raw, checkpoint_digest, run_raw = sys.argv[1:]
template_path = Path(template_path)
runtime_path = Path(runtime_path)
checkpoint = Path(checkpoint_raw)
run = Path(run_raw)
for path in (template_path, runtime_path):
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"missing or symlinked G3 config: {path}")
template = OmegaConf.to_container(OmegaConf.load(template_path), resolve=True)
actual = OmegaConf.to_container(OmegaConf.load(runtime_path), resolve=True)
expected = deepcopy(template)
expected["actor_rollout_ref"]["model"]["path"] = str(checkpoint)
expected["trainer"]["default_local_dir"] = str(run / "checkpoints")
expected["trainer"]["trace_output_dir"] = str(run / "traces")
expected["trainer"]["trace_stage"] = "qwen_native_g3"
expected["trainer"]["trace_run_id"] = run.name
expected["trainer"]["trace_checkpoint_digest"] = checkpoint_digest
expected["trainer"]["trace_parent_checkpoint_digest"] = ""
if actual != expected:
    raise SystemExit("runtime G3 config differs from the sealed CPU template")
PY
}

# Override the training dispatcher. This function contains exactly one eval job.
qwen_native_train_pipeline() {
    local outer_attempt="$1" commit="$2" handoff_digest="$3"
    local _base_model="$4" base_digest="$5" data_digest runner_digest
    local results_dir analysis_dir eval_run trace_identity eval_trace_digest
    local eval_trace_manifest_digest decision_metadata analysis_decision contrast_count
    local branch_json eval_config index_content wandb_file wandb_count=0
    local -a evidence_files wandb_files=()

    data_digest="$(file_sha256 "$NATIVE_TRAIN_MANIFEST")" || return 1
    runner_digest="$(file_sha256 "$G3_RUNNER_PATH")" || return 1
    [[ "$G3_PREFLIGHT_COMMIT" == "$commit" &&
        "$G3_PREFLIGHT_HANDOFF" == "$handoff_digest" &&
        "$G3_PREFLIGHT_BASE_DIGEST" == "$base_digest" &&
        "$G3_PREFLIGHT_DATA_DIGEST" == "$data_digest" &&
        "$G3_PREFLIGHT_RUNNER_DIGEST" == "$runner_digest" ]] || {
        printf 'G3-only inputs drifted after GPU preflight.\n' >&2
        return 1
    }
    verify_g3_cpu_receipt "$runner_digest" || return $?
    verify_native_checkpoint_digest "$G3_R60_CHECKPOINT" \
        "$G3_R60_CHECKPOINT_DIGEST" || return $?

    results_dir="$(create_native_training_results_dir "$outer_attempt")" || return 1
    analysis_dir="$results_dir/r-g3-analysis"
    export EVAL_DATA_FILE="$NATIVE_TRAIN_G3_DATA"
    export EVAL_EXPECTED_ROWS=64
    export EVAL_GROUP_SIZE=5
    run_job eval qwen_native_g3 "$G3_R60_CHECKPOINT" '' \
        "$G3_R60_CHECKPOINT_DIGEST" || return $?
    eval_run="$LAST_RUN_DIR"
    verify_g3_runtime_config "$eval_run/resolved-config.yaml" "$eval_run" || return $?
    append_native_train_index "$outer_attempt" G3 capability_gate "$eval_run" || return $?
    "$TRAIN_ENV/bin/python" "$NATIVE_TRAIN_GATE_CLI" \
        --stage g3 \
        --trace "$eval_run/traces/eval_predictions.jsonl" \
        --catalog "$NATIVE_TRAIN_CATALOG" \
        --data-manifest "$NATIVE_TRAIN_MANIFEST" \
        --expected-checkpoint-digest "$G3_R60_CHECKPOINT_DIGEST" \
        --output-dir "$analysis_dir" || return $?
    trace_identity="$(verify_native_trace_identity \
        "$eval_run/traces/eval_predictions.manifest.json" 320)" || return 1
    IFS=$'\t' read -r eval_trace_digest eval_trace_manifest_digest \
        <<<"$trace_identity"
    decision_metadata="$(validate_r_g3_analysis "$analysis_dir" \
        "$G3_R60_CHECKPOINT_DIGEST" "$eval_trace_digest")" || return 1
    IFS=$'\t' read -r analysis_decision contrast_count <<<"$decision_metadata"
    branch_json="$(write_g3_only_branch_decision "$analysis_decision" \
        "$contrast_count")" || return 1
    atomic_write "$results_dir/branch-decision.json" "$branch_json"$'\n'
    write_wandb_receipt "$eval_run" 0 || return $?
    verify_wandb_receipt "$eval_run" 0 || return $?

    eval_config="$(file_sha256 "$eval_run/resolved-config.yaml")" || return 1
    atomic_write "$results_dir/contract.env" \
"schema=$G3_ONLY_CONTRACT
stage=g3_only
stage_order=G3
analysis_decision=$analysis_decision
cost_contrast_group_count=$contrast_count
branch_candidate_passed=$([[ "$analysis_decision" == GO && "$contrast_count" -ge 8 ]] && printf true || printf false)
branch_training_authorized=false
manual_review_required=true
controlled_outer_exit_code=$G3_ONLY_TERMINAL_CODE
r60_evidence=$G3_EXPECTED_R60_MARKER
r60_evidence_sha256=$G3_R60_DIGEST
cpu_receipt=$G3_CPU_RECEIPT
cpu_receipt_sha256=$G3_CPU_RECEIPT_DIGEST
"
    atomic_write "$results_dir/lineage.tsv" \
        $'stage\trole\trun_dir\tcheckpoint\tcheckpoint_digest\tparent_checkpoint\tparent_checkpoint_digest\tcheckout_commit\tcpu_handoff_digest\tdata_manifest_sha256\tresolved_config_sha256\ttrace_sha256\ttrace_manifest_sha256\trun_contract_sha256\tpredecessor_evidence_sha256\n'\
"G3"$'\t'"capability_gate"$'\t'"$eval_run"$'\t'"$G3_R60_CHECKPOINT"$'\t'"$G3_R60_CHECKPOINT_DIGEST"$'\t'"$G3_R60_CHECKPOINT"$'\t'"$G3_R60_CHECKPOINT_DIGEST"$'\t'"$commit"$'\t'"$handoff_digest"$'\t'"$data_digest"$'\t'"$eval_config"$'\t'"$eval_trace_digest"$'\t'"$eval_trace_manifest_digest"$'\t-\t'"$G3_R60_DIGEST"$'\n'
    index_content="$(cat "$outer_attempt/native-training-runs.tsv")"
    atomic_write "$results_dir/run-index.tsv" "$index_content"$'\n'
    atomic_write "$outer_attempt/g3-only-complete.env" \
"schema=$G3_ONLY_CONTRACT
eval_run=$eval_run
checkpoint=$G3_R60_CHECKPOINT
checkpoint_tree_sha256=$G3_R60_CHECKPOINT_DIGEST
trace_sha256=$eval_trace_digest
analysis_decision=$analysis_decision
branch_training_authorized=false
manual_review_required=true
runner=$G3_RUNNER_PATH
runner_sha256=$runner_digest
"
    [[ -d "$eval_run/wandb" && ! -L "$eval_run/wandb" ]] || return 1
    while IFS= read -r -d '' wandb_file; do
        wandb_files+=("$wandb_file")
        ((wandb_count += 1))
    done < <(find "$eval_run/wandb" -type f -size +0c -print0 | LC_ALL=C sort -z)
    ((wandb_count > 0)) || {
        printf 'G3 WandB evidence is empty: %s\n' "$eval_run" >&2
        return 1
    }
    evidence_files=(
        "$results_dir/contract.env" "$results_dir/lineage.tsv"
        "$results_dir/run-index.tsv" "$results_dir/branch-decision.json"
        "$outer_attempt/g3-only-complete.env" "$G3_RUNNER_PATH"
        "$G3_CPU_RECEIPT" "$G3_CPU_RECEIPT.sha256"
        "$G3_EXPECTED_R60_MARKER" "$G3_R60_MANIFEST"
        "$G3_R60_RESULTS/contract.env" "$G3_R60_RESULTS/lineage.tsv"
        "$G3_R60_RESULTS/run-index.tsv" "$G3_R60_RESULTS/checkpoint-tree.env"
        "$G3_R60_OUTER/r60-only-complete.env" "$G3_CONFIG"
        "$eval_run/train.log" "$eval_run/resolved-config.yaml" "$eval_run/run.env"
        "$eval_run/wandb-receipt.json"
        "$eval_run/traces/eval_predictions.jsonl"
        "$eval_run/traces/eval_predictions.manifest.json"
        "$eval_run/traces/eval_predictions.manifest.json.sha256"
        "$analysis_dir/summary.json" "$analysis_dir/summary.md"
        "$analysis_dir/go_no_go.json" "$analysis_dir/per_trajectory.jsonl"
        "$analysis_dir/per_question.jsonl" "${wandb_files[@]}"
    )
    sync_path "$results_dir"
    verify_checkout "$commit" || return $?
    [[ "$(file_sha256 "$NATIVE_TRAIN_MANIFEST")" == "$data_digest" &&
        "$(file_sha256 "$G3_RUNNER_PATH")" == "$runner_digest" ]] || return 1
    verify_g3_cpu_receipt "$runner_digest" || return $?
    verify_g3_runtime_config "$eval_run/resolved-config.yaml" "$eval_run" || return $?
    verify_native_checkpoint_digest "$G3_R60_CHECKPOINT" \
        "$G3_R60_CHECKPOINT_DIGEST" || return $?
    NATIVE_TRAIN_STAGE=g3-only
    publish_native_training_evidence "$outer_attempt" "$results_dir" \
        "$G3_ONLY_NAMESPACE" "$G3_ONLY_CONTRACT" \
        "${evidence_files[@]}" || return $?
    printf 'Completed and sealed G3-only evaluation (%s): %s\n' \
        "$analysis_decision" "$results_dir"
    printf 'R60 was not retrained; A/R endpoints and B/C were not started.\n'
    printf 'Returning controlled outer status %s for the pinned shutdown watchdog.\n' \
        "$G3_ONLY_TERMINAL_CODE"
    return "$G3_ONLY_TERMINAL_CODE"
}

g3_only_main() {
    case "${1:-}" in
        --cpu-prepare)
            prepare_g3_cpu_receipt
            ;;
        --worker|--action|'')
            qwen_native_train_main "$@"
            ;;
        *)
            printf 'Usage: QWEN_NATIVE_R60_EVIDENCE=%s [GPU_COUNT=2 AUTODL_PRICE_PER_HOUR=<price>] bash %s [--cpu-prepare]\n' \
                "$G3_EXPECTED_R60_MARKER" "$0" >&2
            return 64
            ;;
    esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    g3_only_main "$@"
fi
