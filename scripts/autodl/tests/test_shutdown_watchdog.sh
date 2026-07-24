#!/usr/bin/env bash
set -Eeuo pipefail

AUTODL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
SOURCE_WATCHDOG="$AUTODL_DIR/04_watch_and_shutdown.sh"
TEST_ROOT="$(mktemp -d)"
LOCK_HOLDER_PID=''
trap '[[ -z "$LOCK_HOLDER_PID" ]] || kill "$LOCK_HOLDER_PID" 2>/dev/null || true; rm -rf -- "$TEST_ROOT"' EXIT

case_number=0
CASE_ROOT=''
PROJECT_ROOT=''
ATTEMPT=''
WATCHDOG=''
EVENT_LOG=''
FAKE_SHUTDOWN=''
NATIVE_RESULT_DIR=''
NATIVE_MARKER=''
NATIVE_PRETRAIN_MARKER=''
NATIVE_PRETRAIN_DIGEST=''
NATIVE_PRETRAIN_OUTER=''
NATIVE_SMOKE_MARKER=''
NATIVE_SMOKE_DIGEST=''
NATIVE_SMOKE_OUTER=''
NATIVE_HANDOFF_DIGEST=''
NATIVE_DATA_DIGEST=''
declare -a NATIVE_EVIDENCE_PATHS=()

tree_sha256() {
    (
        cd "$1"
        find . -path './.git' -prune -o -type f -print0 \
            | LC_ALL=C sort -z \
            | xargs -0 -r sha256sum \
            | sha256sum \
            | cut -d' ' -f1
    )
}

new_case() {
    local name="$1" rc="$2" state="$3" marker commit digest
    case_number=$((case_number + 1))
    CASE_ROOT="$TEST_ROOT/$name"
    PROJECT_ROOT="$CASE_ROOT/project"
    ATTEMPT="$PROJECT_ROOT/state/attempts/gpu/20260720T120000Z-$case_number-1"
    WATCHDOG="$PROJECT_ROOT/checkout/scripts/autodl/04_watch_and_shutdown.sh"
    EVENT_LOG="$CASE_ROOT/backend-events.log"
    FAKE_SHUTDOWN="$CASE_ROOT/shutdown"
    mkdir -p "$(dirname -- "$WATCHDOG")" "$PROJECT_ROOT/manifests" \
        "$PROJECT_ROOT/state/latest" "$ATTEMPT"
    cp "$SOURCE_WATCHDOG" "$WATCHDOG"
    chmod 0755 "$WATCHDOG"
    git -C "$PROJECT_ROOT/checkout" init -q
    git -C "$PROJECT_ROOT/checkout" config user.name 'Watchdog Test'
    git -C "$PROJECT_ROOT/checkout" config user.email 'watchdog@example.invalid'
    git -C "$PROJECT_ROOT/checkout" add scripts/autodl/04_watch_and_shutdown.sh
    git -C "$PROJECT_ROOT/checkout" commit -q -m watchdog
    git -C "$PROJECT_ROOT/checkout" checkout --detach -q
    commit="$(git -C "$PROJECT_ROOT/checkout" rev-parse HEAD)"
    digest="$(tree_sha256 "$PROJECT_ROOT/checkout")"
    printf '%s\n' "$commit" >"$PROJECT_ROOT/manifests/git.ok"
    printf '%s\n' "$digest" >"$PROJECT_ROOT/manifests/checkout-tree.sha256"
    : >"$PROJECT_ROOT/state/phase.lock"
    printf '%s\n' "$ATTEMPT" >"$PROJECT_ROOT/state/latest/gpu"
    printf '%s\n' "AUTODL_PHASE_TERMINAL state=$state exit_code=$rc" >"$ATTEMPT/phase.log"
    printf '%s\n' "$rc" >"$ATTEMPT/exit-code"
    printf '%s\n' "$state" >"$ATTEMPT/terminal"
    if [[ "$state" == success ]]; then marker=.success; else marker=.failed; fi
    : >"$ATTEMPT/$marker"
    if [[ "$state" == success ]]; then
        mkdir -p "$PROJECT_ROOT/runs/comparison"
        printf 'variant,metric\nbase,0\nreproduced,1\ncontrol,1\ncost_aware,1\n' \
            >"$PROJECT_ROOT/runs/comparison/results.csv"
        printf '# Test results\n' >"$PROJECT_ROOT/runs/comparison/results.md"
        printf 'stage\trole\nA\tbase\nR\treproduced\nB\tcontrol\nC\tcost_aware\n' \
            >"$PROJECT_ROOT/runs/comparison/lineage.tsv"
        (
            cd "$PROJECT_ROOT/runs/comparison"
            sha256sum results.csv results.md lineage.tsv >comparison.sha256
        )
        sha256sum "$PROJECT_ROOT/runs/comparison/comparison.sha256" | cut -d' ' -f1 \
            >"$PROJECT_ROOT/manifests/gpu.ok"
        cp "$PROJECT_ROOT/manifests/gpu.ok" "$ATTEMPT/comparison-digest"
    fi
    printf 'terminal-published\n' >"$EVENT_LOG"
    printf '%s\n' \
        '#!/usr/bin/env bash' \
        "printf 'REAL_BACKEND_EXECUTED\\n' >>'$EVENT_LOG'" \
        'exit 99' >"$FAKE_SHUTDOWN"
    chmod 0700 "$FAKE_SHUTDOWN"
}

replace_with_followup_results() {
    local result_dir marker relative digest
    local -a files=(
        paired_results.csv correct_questions.csv wrong_questions.csv search_transition.csv
        summary.json summary.md lineage.tsv run-index.tsv
        gated_training_metrics.csv gated_training_curves.svg
    )
    result_dir="$PROJECT_ROOT/runs/cost-aware-gated/attempts/$(basename -- "$ATTEMPT")"
    marker="$PROJECT_ROOT/manifests/cost-aware-gated/$(basename -- "$ATTEMPT").ok"
    mkdir -p "$result_dir" "$(dirname -- "$marker")"
    for relative in "${files[@]}"; do printf 'evidence:%s\n' "$relative" >"$result_dir/$relative"; done
    (
        cd "$PROJECT_ROOT"
        for relative in "${files[@]}"; do
            printf 'runs/cost-aware-gated/attempts/%s/%s\n' "$(basename -- "$ATTEMPT")" "$relative"
        done | LC_ALL=C sort | while IFS= read -r relative; do sha256sum "$relative"; done
    ) >"$result_dir/evidence.sha256"
    digest="$(sha256sum "$result_dir/evidence.sha256" | cut -d' ' -f1)"
    printf '%s\n' "$digest" >"$marker"
    printf 'cost-aware-gated-v1\n' >"$ATTEMPT/result-contract"
    printf '%s\n' "$result_dir" >"$ATTEMPT/result-root"
    printf '%s\n' "$marker" >"$ATTEMPT/evidence-marker"
    printf '%s\n' "$digest" >"$ATTEMPT/evidence-digest"
}

replace_with_search_gate_results() {
    local result_dir marker relative digest
    local -a files=(
        summary.json summary.md go_no_go.json per_question.jsonl
        correct_questions.csv wrong_questions.csv two_plus_search.csv
        redundant_search_candidates.csv strata.csv lineage.tsv run-index.tsv
    )
    result_dir="$PROJECT_ROOT/runs/search-opportunity-gate/attempts/$(basename -- "$ATTEMPT")"
    marker="$PROJECT_ROOT/manifests/search-opportunity-gate/$(basename -- "$ATTEMPT").ok"
    mkdir -p "$result_dir" "$(dirname -- "$marker")"
    for relative in "${files[@]}"; do printf 'evidence:%s\n' "$relative" >"$result_dir/$relative"; done
    printf '{"decision":"NO-GO"}\n' >"$result_dir/go_no_go.json"
    (
        cd "$PROJECT_ROOT"
        for relative in "${files[@]}"; do
            printf 'runs/search-opportunity-gate/attempts/%s/%s\n' \
                "$(basename -- "$ATTEMPT")" "$relative"
        done | LC_ALL=C sort | while IFS= read -r relative; do sha256sum "$relative"; done
    ) >"$result_dir/evidence.sha256"
    digest="$(sha256sum "$result_dir/evidence.sha256" | cut -d' ' -f1)"
    printf '%s\n' "$digest" >"$marker"
    printf 'search-opportunity-gate-v1\n' >"$ATTEMPT/result-contract"
    printf '%s\n' "$result_dir" >"$ATTEMPT/result-root"
    printf '%s\n' "$marker" >"$ATTEMPT/evidence-marker"
    printf '%s\n' "$digest" >"$ATTEMPT/evidence-digest"
}

replace_with_group_probe_results() {
    local result_dir marker relative digest
    local -a files=(
        summary.json summary.md go_no_go.json per_trajectory.jsonl
        per_question.jsonl lineage.tsv run-index.tsv raw-trace.jsonl
    )
    result_dir="$PROJECT_ROOT/runs/group-probe/attempts/$(basename -- "$ATTEMPT")"
    marker="$PROJECT_ROOT/manifests/group-probe/$(basename -- "$ATTEMPT").ok"
    mkdir -p "$result_dir" "$(dirname -- "$marker")"
    for relative in "${files[@]}"; do printf 'evidence:%s\n' "$relative" >"$result_dir/$relative"; done
    printf '{"decision":"NO-GO"}\n' >"$result_dir/go_no_go.json"
    (
        cd "$PROJECT_ROOT"
        for relative in "${files[@]}"; do
            printf 'runs/group-probe/attempts/%s/%s\n' \
                "$(basename -- "$ATTEMPT")" "$relative"
        done | LC_ALL=C sort | while IFS= read -r relative; do sha256sum "$relative"; done
    ) >"$result_dir/evidence.sha256"
    digest="$(sha256sum "$result_dir/evidence.sha256" | cut -d' ' -f1)"
    printf '%s\n' "$digest" >"$marker"
    printf 'group-probe-v1\n' >"$ATTEMPT/result-contract"
    printf '%s\n' "$result_dir" >"$ATTEMPT/result-root"
    printf '%s\n' "$marker" >"$ATTEMPT/evidence-marker"
    printf '%s\n' "$digest" >"$ATTEMPT/evidence-digest"
}

replace_with_qwen_native_gate_results() {
    local result_dir marker relative digest
    local -a files=(
        summary.json summary.md go_no_go.json per_trajectory.jsonl
        per_question.jsonl lineage.tsv run-index.tsv stage.txt sampling.json
        raw-trace.jsonl
    )
    result_dir="$PROJECT_ROOT/runs/qwen-native-gate/attempts/$(basename -- "$ATTEMPT")"
    marker="$PROJECT_ROOT/manifests/qwen-native-gate/$(basename -- "$ATTEMPT").ok"
    mkdir -p "$result_dir" "$(dirname -- "$marker")"
    for relative in "${files[@]}"; do printf 'evidence:%s\n' "$relative" >"$result_dir/$relative"; done
    printf '{"decision":"NO-GO","stage":"g2"}\n' >"$result_dir/go_no_go.json"
    printf 'g2\n' >"$result_dir/stage.txt"
    (
        cd "$PROJECT_ROOT"
        for relative in "${files[@]}"; do
            printf 'runs/qwen-native-gate/attempts/%s/%s\n' \
                "$(basename -- "$ATTEMPT")" "$relative"
        done | LC_ALL=C sort | while IFS= read -r relative; do sha256sum "$relative"; done
    ) >"$result_dir/evidence.sha256"
    digest="$(sha256sum "$result_dir/evidence.sha256" | cut -d' ' -f1)"
    printf '%s\n' "$digest" >"$marker"
    printf 'qwen-native-gate-v1\n' >"$ATTEMPT/result-contract"
    printf '%s\n' "$result_dir" >"$ATTEMPT/result-root"
    printf '%s\n' "$marker" >"$ATTEMPT/evidence-marker"
    printf '%s\n' "$digest" >"$ATTEMPT/evidence-digest"
}

add_native_evidence_path() {
    NATIVE_EVIDENCE_PATHS+=("$1")
}

write_native_manifest() {
    local output="$1"
    shift
    local path relative
    (
        cd "$PROJECT_ROOT"
        for path in "$@"; do
            relative="${path#"$PROJECT_ROOT/"}"
            printf '%s\n' "$relative"
        done | LC_ALL=C sort -u | while IFS= read -r relative; do
            sha256sum "$relative"
        done
    ) >"$output"
}

create_native_outer_attempt() {
    local attempt_name="$1" contract="$2" result_dir="$3" marker="$4" digest="$5"
    local outer="$PROJECT_ROOT/state/attempts/gpu/$attempt_name"
    mkdir -p "$outer"
    : >"$outer/.success"
    printf 'success\n' >"$outer/terminal"
    printf '0\n' >"$outer/exit-code"
    printf '%s\n' "$contract" >"$outer/result-contract"
    printf '%s\n' "$result_dir" >"$outer/result-root"
    printf '%s\n' "$marker" >"$outer/evidence-marker"
    printf '%s\n' "$digest" >"$outer/evidence-digest"
    printf '%s\n' "$outer"
}

create_native_sealed_inputs() {
    local data_dir="$PROJECT_ROOT/data/search_mix_qwen35_native_v2"
    local handoff="$PROJECT_ROOT/manifests/cpu_handoff.json"
    mkdir -p "$data_dir"
    python3 - "$data_dir" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
sample_ids = [f"sample-{index:03d}" for index in range(128)]
catalog = root / "catalog.jsonl"
catalog.write_text("".join(
    json.dumps({
        "sample_id": sample_id,
        "question": f"Question {sample_id}?",
        "golden_answers": [sample_id],
    }, ensure_ascii=True, separators=(",", ":")) + "\n"
    for sample_id in sample_ids
), encoding="utf-8")
catalog_digest = hashlib.sha256(catalog.read_bytes()).hexdigest()
manifest = {
    "schema_version": 3,
    "prompt_contract": {"tool_protocol": "qwen35_native"},
    "artifacts": {
        "catalog": {"file": "catalog.jsonl", "sha256": catalog_digest},
        "val": {
            "file": "val_128.parquet",
            "rows": 128,
            "sha256": "a" * 64,
            "sample_ids": sample_ids,
        },
    },
}
(root / "manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=True, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
    printf '{"schema":"test-cpu-handoff"}\n' >"$handoff"
    NATIVE_DATA_DIGEST="$(sha256sum "$data_dir/manifest.json" | cut -d' ' -f1)"
    NATIVE_HANDOFF_DIGEST="$(sha256sum "$handoff" | cut -d' ' -f1)"
    printf '%s\n' "$NATIVE_HANDOFF_DIGEST" >"$PROJECT_ROOT/manifests/cpu.ok"
    add_native_evidence_path "$PROJECT_ROOT/manifests/git.ok"
    add_native_evidence_path "$PROJECT_ROOT/manifests/cpu.ok"
    add_native_evidence_path "$handoff"
    add_native_evidence_path "$data_dir/manifest.json"
    add_native_evidence_path "$data_dir/catalog.jsonl"
}

write_smoke_decision() {
    local output="$1" decision="$2" catalog_digest="$3" log_digest="$4"
    local trace_digest="$5" wandb_digest="$6"
    python3 - "$output" "$decision" "$catalog_digest" "$log_digest" \
        "$trace_digest" "$wandb_digest" <<'PY'
import json
from pathlib import Path
import sys

output, decision, catalog, log, trace, wandb = sys.argv[1:]
check_names = (
    "strict_em_positive",
    "mixed_reward_group",
    "nonzero_trace_advantage",
    "finite_actor_pg_loss",
    "finite_actor_kl_loss",
    "finite_actor_entropy_loss",
    "finite_actor_grad_norm",
    "finite_actor_ppo_kl",
    "native_batch_contract_valid",
    "native_batch_info_loss_mask_match",
    "native_batch_policy_mask_subset",
    "native_batch_old_log_prob_finite_ratio",
    "native_batch_advantage_finite_ratio",
    "native_batch_reward_finite_ratio",
    "native_batch_policy_tokens",
    "native_batch_policy_tokens_min_per_trajectory",
    "native_batch_policy_coverage",
    "native_batch_nonzero_advantage_tokens",
    "native_batch_advantage_abs_max",
    "wandb_offline_history",
)
metrics = {
    "groups": 16,
    "mixed_groups": 1,
    "nonzero_trace_advantages": 1,
    "strict_em_positive_count": 1 if decision == "GO" else 0,
    "trajectories": 80,
    "wandb_files": 1,
}
observed = {
    "strict_em_positive": metrics["strict_em_positive_count"],
    "mixed_reward_group": metrics["mixed_groups"],
    "nonzero_trace_advantage": metrics["nonzero_trace_advantages"],
    "wandb_offline_history": metrics["wandb_files"],
}
checks = {
    name: {
        "observed": observed.get(name, 1),
        "passed": not (decision == "NO-GO" and name == "strict_em_positive"),
    }
    for name in check_names
}
payload = {
    "checks": checks,
    "decision": decision,
    "inputs": {
        "catalog_sha256": catalog,
        "log_sha256": log,
        "trace_sha256": trace,
        "wandb_tree_sha256": wandb,
    },
    "metrics": metrics,
    "schema": "search-r1.qwen-native-smoke-decision",
    "schema_version": 1,
}
Path(output).write_text(
    json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    + "\n",
    encoding="utf-8",
)
PY
}

create_native_pretrain_evidence() {
    local attempt_name result_dir manifest digest
    attempt_name="20260724T010000Z-$case_number-10"
    result_dir="$PROJECT_ROOT/runs/qwen-native-gate/attempts/$attempt_name"
    NATIVE_PRETRAIN_MARKER="$PROJECT_ROOT/manifests/qwen-native-gate/$attempt_name.ok"
    mkdir -p "$result_dir" "$(dirname -- "$NATIVE_PRETRAIN_MARKER")"
    printf 'g3\n' >"$result_dir/stage.txt"
    printf '{"decision":"GO","stage":"g3"}\n' >"$result_dir/go_no_go.json"
    write_native_manifest "$result_dir/evidence.sha256" \
        "$result_dir/go_no_go.json" "$result_dir/stage.txt"
    NATIVE_PRETRAIN_DIGEST="$(sha256sum "$result_dir/evidence.sha256" | cut -d' ' -f1)"
    printf '%s\n' "$NATIVE_PRETRAIN_DIGEST" >"$NATIVE_PRETRAIN_MARKER"
    NATIVE_PRETRAIN_OUTER="$(create_native_outer_attempt "$attempt_name" \
        qwen-native-gate-v1 "$result_dir" "$NATIVE_PRETRAIN_MARKER" \
        "$NATIVE_PRETRAIN_DIGEST")"
    add_native_evidence_path "$NATIVE_PRETRAIN_MARKER"
    add_native_evidence_path "$result_dir/evidence.sha256"
}

create_native_smoke_predecessor() {
    local attempt_name result_dir digest
    attempt_name="20260724T020000Z-$case_number-20"
    result_dir="$PROJECT_ROOT/runs/qwen-native-training/attempts/$attempt_name"
    NATIVE_SMOKE_MARKER="$PROJECT_ROOT/manifests/qwen-native-training-smoke/$attempt_name.ok"
    mkdir -p "$result_dir" "$(dirname -- "$NATIVE_SMOKE_MARKER")"
    printf '%s\n' \
        'schema=qwen-native-training-smoke-v1' \
        'stage=smoke' \
        'stage_order=S2' \
        'decision=GO' \
        'manual_review_required=true' \
        "pretrain_g3_evidence=$NATIVE_PRETRAIN_MARKER" \
        "pretrain_g3_evidence_sha256=$NATIVE_PRETRAIN_DIGEST" \
        >"$result_dir/contract.env"
    write_smoke_decision "$result_dir/smoke-decision.json" GO \
        "$(printf '1%.0s' {1..64})" "$(printf '2%.0s' {1..64})" \
        "$(printf '3%.0s' {1..64})" "$(printf '4%.0s' {1..64})"
    printf 'stage\trole\nS\tsmoke\n' >"$result_dir/lineage.tsv"
    printf 'checkpoint_bytes=1\nfilesystem_available_bytes=1\nrecorded_at=2026-07-24T02:00:00Z\n' \
        >"$result_dir/storage.env"
    write_native_manifest "$result_dir/evidence.sha256" \
        "$result_dir/contract.env" "$result_dir/lineage.tsv" "$result_dir/storage.env" \
        "$result_dir/smoke-decision.json"
    digest="$(sha256sum "$result_dir/evidence.sha256" | cut -d' ' -f1)"
    NATIVE_SMOKE_DIGEST="$digest"
    printf '%s\n' "$digest" >"$NATIVE_SMOKE_MARKER"
    NATIVE_SMOKE_OUTER="$(create_native_outer_attempt "$attempt_name" \
        qwen-native-training-smoke-v1 "$result_dir" "$NATIVE_SMOKE_MARKER" \
        "$digest")"
    add_native_evidence_path "$NATIVE_SMOKE_MARKER"
    add_native_evidence_path "$result_dir/evidence.sha256"
    add_native_evidence_path "$result_dir/contract.env"
}

create_native_run_evidence() {
    local run_dir="$1" kind="$2" file
    mkdir -p "$run_dir/traces"
    for file in train.log resolved-config.yaml run.env; do
        printf 'native evidence: %s\n' "$file" >"$run_dir/$file"
        add_native_evidence_path "$run_dir/$file"
    done
    if [[ "$kind" == train ]]; then
        for file in lineage.tsv native-training-contract.json; do
            printf 'native evidence: %s\n' "$file" >"$run_dir/$file"
            add_native_evidence_path "$run_dir/$file"
        done
        for file in train_trajectories.jsonl train_trajectories.manifest.json \
            train_trajectories.manifest.json.sha256; do
            printf 'native evidence: %s\n' "$file" >"$run_dir/traces/$file"
            add_native_evidence_path "$run_dir/traces/$file"
        done
    else
        for file in eval_predictions.jsonl eval_predictions.manifest.json \
            eval_predictions.manifest.json.sha256; do
            printf 'native evidence: %s\n' "$file" >"$run_dir/traces/$file"
            add_native_evidence_path "$run_dir/traces/$file"
        done
    fi
}

append_native_main_lineage() {
    local stage="$1" role="$2" run_dir="$3" checkpoint="$4" parent="$5"
    local kind="$6" predecessor_digest="$7"
    local checkpoint_digest parent_digest config_digest trace_digest
    local trace_manifest_digest run_contract_digest commit
    local trace trace_manifest
    local -a values
    checkpoint_digest="$(tree_sha256 "$checkpoint")"
    parent_digest="$(tree_sha256 "$parent")"
    config_digest="$(sha256sum "$run_dir/resolved-config.yaml" | cut -d' ' -f1)"
    if [[ "$kind" == train ]]; then
        trace="$run_dir/traces/train_trajectories.jsonl"
        trace_manifest="$run_dir/traces/train_trajectories.manifest.json"
        run_contract_digest="$(sha256sum "$run_dir/native-training-contract.json" | cut -d' ' -f1)"
    else
        trace="$run_dir/traces/eval_predictions.jsonl"
        trace_manifest="$run_dir/traces/eval_predictions.manifest.json"
        run_contract_digest=-
    fi
    trace_digest="$(sha256sum "$trace" | cut -d' ' -f1)"
    trace_manifest_digest="$(sha256sum "$trace_manifest" | cut -d' ' -f1)"
    commit="$(git -C "$PROJECT_ROOT/checkout" rev-parse HEAD)"
    values=(
        "$stage" "$role" "$run_dir" "$checkpoint" "$checkpoint_digest"
        "$parent" "$parent_digest" "$commit" "$NATIVE_HANDOFF_DIGEST"
        "$NATIVE_DATA_DIGEST" "$config_digest" "$trace_digest"
        "$trace_manifest_digest" "$run_contract_digest" "$predecessor_digest"
    )
    (
        IFS=$'\t'
        printf '%s\n' "${values[*]}"
    ) >>"$NATIVE_RESULT_DIR/lineage.tsv"
}

write_native_g3_analysis() {
    local output_dir="$1" decision="$2" count="$3" trace="$4" catalog="$5"
    local checkpoint_digest="$6"
    python3 - "$output_dir" "$decision" "$count" "$trace" "$catalog" \
        "$checkpoint_digest" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

output_dir = Path(sys.argv[1])
decision = sys.argv[2]
count = int(sys.argv[3])
trace = Path(sys.argv[4])
catalog = Path(sys.argv[5])
checkpoint_digest = sys.argv[6]
trace_digest = hashlib.sha256(trace.read_bytes()).hexdigest()
catalog_digest = hashlib.sha256(catalog.read_bytes()).hexdigest()
replay = {
    "schema": "search-r1.strict-em-replay",
    "schema_version": 1,
    "source": "catalog.golden_answers+trace.extracted_answer/final_answer",
    "verified": True,
    "trajectory_count": 320,
    "strict_em_positive_count": 1,
    "subem_positive_count": 1,
    "trace_sha256": trace_digest,
    "catalog_sha256": catalog_digest,
}
summary = {
    "schema": "search-r1.grouped-probe-analysis",
    "schema_version": 2,
    "decision": decision,
    "input": {
        "trace_path": str(trace),
        "trace_sha256": trace_digest,
        "catalog_path": str(catalog),
        "catalog_sha256": catalog_digest,
        "checkpoint_digest": checkpoint_digest,
        "stage": "qwen_native_g3",
    },
    "contract": {
        "expected_questions": 64,
        "trajectories_per_question": 5,
        "expected_trajectories": 320,
        "max_searches": 4,
    },
    "overall": {"cost_contrast_group_count": count},
    "go_no_go": {"decision": decision},
    "strict_em_replay": replay,
}
gate = {
    "schema": "search-r1.grouped-probe-analysis",
    "schema_version": 2,
    "stage": "g3",
    "decision": decision,
    "trace_sha256": trace_digest,
    "catalog_sha256": catalog_digest,
    "checkpoint_digest": checkpoint_digest,
    "strict_em_replay": replay,
}
for name, payload in (("summary.json", summary), ("go_no_go.json", gate)):
    (output_dir / name).write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
PY
}

write_native_paired_summary() {
    local output="$1" data_manifest="$2" catalog="$3"
    local control_trace="$4" cost_trace="$5" control_digest="$6" cost_digest="$7"
    python3 - "$output" "$data_manifest" "$catalog" "$control_trace" "$cost_trace" \
        "$control_digest" "$cost_digest" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

output = Path(sys.argv[1])
data_manifest = Path(sys.argv[2])
catalog = Path(sys.argv[3])
control_trace = Path(sys.argv[4])
cost_trace = Path(sys.argv[5])
control_digest = sys.argv[6]
cost_digest = sys.argv[7]
manifest = json.loads(data_manifest.read_bytes())
sample_ids = manifest["artifacts"]["val"]["sample_ids"]
sample_ids_digest = hashlib.sha256(json.dumps(
    sample_ids, ensure_ascii=False, separators=(",", ":")
).encode("utf-8")).hexdigest()
catalog_digest = hashlib.sha256(catalog.read_bytes()).hexdigest()
payload = {
    "schema_version": 1,
    "expected_rows": 128,
    "cost_lambda": 0.10,
    "max_searches": 4,
    "inputs": {
        "control": {
            "path": str(control_trace),
            "sha256": hashlib.sha256(control_trace.read_bytes()).hexdigest(),
        },
        "cost_aware_gated": {
            "path": str(cost_trace),
            "sha256": hashlib.sha256(cost_trace.read_bytes()).hexdigest(),
        },
    },
    "stages": {
        "control": {"stage": "qwen_native_b"},
        "cost_aware_gated": {"stage": "qwen_native_c"},
    },
    "comparisons": {"cost_aware_gated": {}},
    "catalog": {
        "path": str(catalog),
        "sha256": catalog_digest,
        "row_count": len(catalog.read_text(encoding="utf-8").splitlines()),
        "matched_rows": 128,
        "replay_status": "passed",
        "strict_em_scorer": "qa_em.em_check",
    },
    "formal_contract": {
        "mode": "qwen35_native_v2_b_c",
        "data_manifest": {
            "path": str(data_manifest),
            "sha256": hashlib.sha256(data_manifest.read_bytes()).hexdigest(),
            "schema_version": 3,
        },
        "catalog": {
            "path": str(catalog),
            "sha256": catalog_digest,
        },
        "val": {
            "file": "val_128.parquet",
            "rows": 128,
            "sample_ids_sha256": sample_ids_digest,
            "sample_set_status": "exact",
        },
        "endpoints": {
            "control": {
                "stage": "qwen_native_b",
                "checkpoint_digest": control_digest,
            },
            "cost_aware_gated": {
                "stage": "qwen_native_c",
                "checkpoint_digest": cost_digest,
            },
        },
    },
}
output.write_text(
    json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    + "\n",
    encoding="utf-8",
)
PY
}

seal_native_training_results() {
    local contract="$1" namespace="$2" digest
    write_native_manifest "$NATIVE_RESULT_DIR/evidence.sha256" "${NATIVE_EVIDENCE_PATHS[@]}"
    digest="$(sha256sum "$NATIVE_RESULT_DIR/evidence.sha256" | cut -d' ' -f1)"
    mkdir -p "$(dirname -- "$NATIVE_MARKER")"
    printf '%s\n' "$digest" >"$NATIVE_MARKER"
    printf '%s\n' "$contract" >"$ATTEMPT/result-contract"
    printf '%s\n' "$NATIVE_RESULT_DIR" >"$ATTEMPT/result-root"
    printf '%s\n' "$NATIVE_MARKER" >"$ATTEMPT/evidence-marker"
    printf '%s\n' "$digest" >"$ATTEMPT/evidence-digest"
}

replace_with_native_training_smoke_results() {
    local decision="${1:-GO}" attempt_name run_dir checkpoint base_model catalog wandb_digest
    local base_digest checkpoint_digest handoff_digest data_digest config_digest
    local trace_digest trace_manifest_digest run_contract_digest commit
    local wandb_file
    attempt_name="$(basename -- "$ATTEMPT")"
    NATIVE_EVIDENCE_PATHS=()
    create_native_pretrain_evidence
    create_native_sealed_inputs
    NATIVE_RESULT_DIR="$PROJECT_ROOT/runs/qwen-native-training/attempts/$attempt_name"
    NATIVE_MARKER="$PROJECT_ROOT/manifests/qwen-native-training-smoke/$attempt_name.ok"
    run_dir="$PROJECT_ROOT/runs/smoke/attempts/smoke-$case_number"
    checkpoint="$run_dir/checkpoints/actor/global_step_2"
    base_model="$PROJECT_ROOT/models/Qwen3.5-2B"
    catalog="$PROJECT_ROOT/data/search_mix_qwen35_native_v2/catalog.jsonl"
    mkdir -p "$NATIVE_RESULT_DIR" "$checkpoint" "$base_model" "$(dirname -- "$catalog")"
    add_native_evidence_path "$catalog"
    create_native_run_evidence "$run_dir" train
    printf 'base model\n' >"$base_model/model.bin"
    printf 'smoke checkpoint\n' >"$checkpoint/model.bin"
    wandb_file="$run_dir/wandb/offline-run/run-test.wandb"
    mkdir -p "$(dirname -- "$wandb_file")"
    printf 'offline history\n' >"$wandb_file"
    add_native_evidence_path "$wandb_file"
    base_digest="$(tree_sha256 "$base_model")"
    checkpoint_digest="$(tree_sha256 "$checkpoint")"
    handoff_digest="$NATIVE_HANDOFF_DIGEST"
    data_digest="$NATIVE_DATA_DIGEST"
    config_digest="$(sha256sum "$run_dir/resolved-config.yaml" | cut -d' ' -f1)"
    trace_digest="$(sha256sum "$run_dir/traces/train_trajectories.jsonl" | cut -d' ' -f1)"
    trace_manifest_digest="$(sha256sum \
        "$run_dir/traces/train_trajectories.manifest.json" | cut -d' ' -f1)"
    run_contract_digest="$(sha256sum "$run_dir/native-training-contract.json" \
        | cut -d' ' -f1)"
    commit="$(git -C "$PROJECT_ROOT/checkout" rev-parse HEAD)"
    printf '%s\n' \
        'schema=qwen-native-training-smoke-v1' \
        'stage=smoke' \
        'stage_order=S2' \
        "decision=$decision" \
        'manual_review_required=true' \
        "pretrain_g3_evidence=$NATIVE_PRETRAIN_MARKER" \
        "pretrain_g3_evidence_sha256=$NATIVE_PRETRAIN_DIGEST" \
        >"$NATIVE_RESULT_DIR/contract.env"
    wandb_digest="$(printf '%s  offline-run/run-test.wandb\n' \
        "$(sha256sum "$wandb_file" | cut -d' ' -f1)" | sha256sum | cut -d' ' -f1)"
    write_smoke_decision "$NATIVE_RESULT_DIR/smoke-decision.json" "$decision" \
        "$(sha256sum "$catalog" | cut -d' ' -f1)" \
        "$(sha256sum "$run_dir/train.log" | cut -d' ' -f1)" \
        "$(sha256sum "$run_dir/traces/train_trajectories.jsonl" | cut -d' ' -f1)" \
        "$wandb_digest"
    printf '%s\n' \
        $'stage\trole\trun_dir\tcheckpoint\tcheckpoint_digest\tparent_checkpoint\tparent_checkpoint_digest\tcheckout_commit\tcpu_handoff_digest\tdata_manifest_sha256\tresolved_config_sha256\ttrace_sha256\ttrace_manifest_sha256\trun_contract_sha256\tpretrain_g3_evidence\tpretrain_g3_evidence_sha256' \
        "S"$'\t'"smoke"$'\t'"$run_dir"$'\t'"$checkpoint"$'\t'"$checkpoint_digest"$'\t'"$base_model"$'\t'"$base_digest"$'\t'"$commit"$'\t'"$handoff_digest"$'\t'"$data_digest"$'\t'"$config_digest"$'\t'"$trace_digest"$'\t'"$trace_manifest_digest"$'\t'"$run_contract_digest"$'\t'"$NATIVE_PRETRAIN_MARKER"$'\t'"$NATIVE_PRETRAIN_DIGEST" \
        >"$NATIVE_RESULT_DIR/lineage.tsv"
    printf 'stage\trole\trun_dir\nS\tsmoke\t%s\n' "$run_dir" \
        >"$NATIVE_RESULT_DIR/run-index.tsv"
    printf 'checkpoint_bytes=1\nfilesystem_available_bytes=2\nrecorded_at=2026-07-24T03:00:00Z\n' \
        >"$NATIVE_RESULT_DIR/storage.env"
    printf 'checkpoint=%s\ncheckpoint_tree_sha256=%s\n' "$checkpoint" "$checkpoint_digest" \
        >"$NATIVE_RESULT_DIR/checkpoint-tree.env"
    add_native_evidence_path "$NATIVE_RESULT_DIR/contract.env"
    add_native_evidence_path "$NATIVE_RESULT_DIR/lineage.tsv"
    add_native_evidence_path "$NATIVE_RESULT_DIR/run-index.tsv"
    add_native_evidence_path "$NATIVE_RESULT_DIR/storage.env"
    add_native_evidence_path "$NATIVE_RESULT_DIR/checkpoint-tree.env"
    add_native_evidence_path "$NATIVE_RESULT_DIR/smoke-decision.json"
    seal_native_training_results qwen-native-training-smoke-v1 qwen-native-training-smoke
}

replace_with_native_training_main_results() {
    local authorized="$1" attempt_name decision count stage_order
    local base_model r_checkpoint b_checkpoint c_checkpoint
    local r_run g3_run b_run c_run b_eval_run c_eval_run
    local r_digest b_digest c_digest smoke_digest paired file
    local catalog data_manifest
    attempt_name="$(basename -- "$ATTEMPT")"
    NATIVE_EVIDENCE_PATHS=()
    create_native_pretrain_evidence
    create_native_smoke_predecessor
    create_native_sealed_inputs
    NATIVE_RESULT_DIR="$PROJECT_ROOT/runs/qwen-native-training/attempts/$attempt_name"
    NATIVE_MARKER="$PROJECT_ROOT/manifests/qwen-native-training-main/$attempt_name.ok"
    mkdir -p "$NATIVE_RESULT_DIR/r-g3-analysis"
    base_model="$PROJECT_ROOT/models/Qwen3.5-2B"
    r_run="$PROJECT_ROOT/runs/reproduce/attempts/r-$case_number"
    g3_run="$PROJECT_ROOT/runs/eval/qwen_native_g3/attempts/g3-$case_number"
    r_checkpoint="$r_run/checkpoints/actor/global_step_60"
    mkdir -p "$base_model" "$r_checkpoint"
    printf 'base model\n' >"$base_model/model.bin"
    printf 'R checkpoint\n' >"$r_checkpoint/model.bin"
    add_native_evidence_path "$base_model/model.bin"
    add_native_evidence_path "$r_checkpoint/model.bin"
    create_native_run_evidence "$r_run" train
    create_native_run_evidence "$g3_run" eval
    r_digest="$(tree_sha256 "$r_checkpoint")"
    smoke_digest="$NATIVE_SMOKE_DIGEST"
    catalog="$PROJECT_ROOT/data/search_mix_qwen35_native_v2/catalog.jsonl"
    data_manifest="$PROJECT_ROOT/data/search_mix_qwen35_native_v2/manifest.json"
    if [[ "$authorized" == true ]]; then
        decision=GO
        count=9
        stage_order=R60,G3,B20,C20,B-EVAL,C-EVAL
    else
        decision=NO-GO
        count=2
        stage_order=R60,G3
    fi
    printf '%s\n' \
        'schema=qwen-native-training-main-v1' \
        'stage=main' \
        "stage_order=$stage_order" \
        "analysis_decision=$decision" \
        "cost_contrast_group_count=$count" \
        "branch_authorized=$authorized" \
        "pretrain_g3_evidence=$NATIVE_PRETRAIN_MARKER" \
        "pretrain_g3_evidence_sha256=$NATIVE_PRETRAIN_DIGEST" \
        "smoke_evidence=$NATIVE_SMOKE_MARKER" \
        "smoke_evidence_sha256=$smoke_digest" \
        >"$NATIVE_RESULT_DIR/contract.env"
    printf '{"analysis_decision":"%s","branches_authorized":%s,"cost_contrast_group_count":%s,"cost_contrast_group_minimum":8,"decision":"%s","schema":"qwen-native-post-r-gate-v1"}\n' \
        "$decision" "$authorized" "$count" "$([[ "$authorized" == true ]] && printf GO || printf NO-GO)" \
        >"$NATIVE_RESULT_DIR/branch-decision.json"
    write_native_g3_analysis "$NATIVE_RESULT_DIR/r-g3-analysis" "$decision" "$count" \
        "$g3_run/traces/eval_predictions.jsonl" "$catalog" "$r_digest"
    printf '# R-G3 analysis\n' >"$NATIVE_RESULT_DIR/r-g3-analysis/summary.md"
    printf '{}\n' >"$NATIVE_RESULT_DIR/r-g3-analysis/per_trajectory.jsonl"
    printf '{}\n' >"$NATIVE_RESULT_DIR/r-g3-analysis/per_question.jsonl"
    printf '%s\n' \
        $'stage\trole\trun_dir\tcheckpoint\tcheckpoint_digest\tparent_checkpoint\tparent_checkpoint_digest\tcheckout_commit\tcpu_handoff_digest\tdata_manifest_sha256\tresolved_config_sha256\ttrace_sha256\ttrace_manifest_sha256\trun_contract_sha256\tpredecessor_evidence_sha256' \
        >"$NATIVE_RESULT_DIR/lineage.tsv"
    printf 'stage\trole\trun_dir\n' >"$NATIVE_RESULT_DIR/run-index.tsv"
    append_native_main_lineage R reproduced "$r_run" "$r_checkpoint" "$base_model" \
        train "$smoke_digest"
    append_native_main_lineage G3 capability_gate "$g3_run" "$r_checkpoint" \
        "$r_checkpoint" eval "$smoke_digest"
    printf 'R\treproduced\t%s\nG3\tcapability_gate\t%s\n' "$r_run" "$g3_run" \
        >>"$NATIVE_RESULT_DIR/run-index.tsv"
    if [[ "$authorized" == true ]]; then
        b_run="$PROJECT_ROOT/runs/control/attempts/b-$case_number"
        c_run="$PROJECT_ROOT/runs/cost_aware_gated/attempts/c-$case_number"
        b_eval_run="$PROJECT_ROOT/runs/eval/control/attempts/b-eval-$case_number"
        c_eval_run="$PROJECT_ROOT/runs/eval/cost_aware_gated/attempts/c-eval-$case_number"
        b_checkpoint="$b_run/checkpoints/actor/global_step_20"
        c_checkpoint="$c_run/checkpoints/actor/global_step_20"
        mkdir -p "$b_checkpoint" "$c_checkpoint"
        printf 'B checkpoint\n' >"$b_checkpoint/model.bin"
        printf 'C checkpoint\n' >"$c_checkpoint/model.bin"
        add_native_evidence_path "$b_checkpoint/model.bin"
        add_native_evidence_path "$c_checkpoint/model.bin"
        create_native_run_evidence "$b_run" train
        create_native_run_evidence "$c_run" train
        create_native_run_evidence "$b_eval_run" eval
        create_native_run_evidence "$c_eval_run" eval
        b_digest="$(tree_sha256 "$b_checkpoint")"
        c_digest="$(tree_sha256 "$c_checkpoint")"
        append_native_main_lineage B control "$b_run" "$b_checkpoint" "$r_checkpoint" \
            train "$smoke_digest"
        append_native_main_lineage C cost_aware_gated "$c_run" "$c_checkpoint" \
            "$r_checkpoint" train "$smoke_digest"
        append_native_main_lineage B-EVAL control_eval "$b_eval_run" "$b_checkpoint" \
            "$b_checkpoint" eval "$smoke_digest"
        append_native_main_lineage C-EVAL cost_aware_gated_eval "$c_eval_run" \
            "$c_checkpoint" "$c_checkpoint" eval "$smoke_digest"
        printf 'B\tcontrol\t%s\nC\tcost_aware_gated\t%s\nB-EVAL\tcontrol_eval\t%s\nC-EVAL\tcost_aware_gated_eval\t%s\n' \
            "$b_run" "$c_run" "$b_eval_run" "$c_eval_run" \
            >>"$NATIVE_RESULT_DIR/run-index.tsv"
        paired="$NATIVE_RESULT_DIR/paired"
        mkdir -p "$paired"
        for file in summary.json summary.md paired_results.csv correct_questions.csv \
            wrong_questions.csv search_transition.csv; do
            printf 'paired evidence: %s\n' "$file" >"$paired/$file"
            add_native_evidence_path "$paired/$file"
        done
        write_native_paired_summary "$paired/summary.json" "$data_manifest" "$catalog" \
            "$b_eval_run/traces/eval_predictions.jsonl" \
            "$c_eval_run/traces/eval_predictions.jsonl" "$b_digest" "$c_digest"
    fi
    add_native_evidence_path "$NATIVE_RESULT_DIR/contract.env"
    add_native_evidence_path "$NATIVE_RESULT_DIR/lineage.tsv"
    add_native_evidence_path "$NATIVE_RESULT_DIR/run-index.tsv"
    add_native_evidence_path "$NATIVE_RESULT_DIR/branch-decision.json"
    for file in summary.json summary.md go_no_go.json per_trajectory.jsonl per_question.jsonl; do
        add_native_evidence_path "$NATIVE_RESULT_DIR/r-g3-analysis/$file"
    done
    seal_native_training_results qwen-native-training-main-v1 qwen-native-training-main
}

run_watchdog() {
    local backend_rc="$1" mode="$2"
    SEARCH_R1_AUTODL_TEST_MODE=1 \
    SEARCH_R1_AUTODL_TEST_ROOT="$CASE_ROOT" \
    SEARCH_R1_AUTODL_TEST_SHUTDOWN_LOG="$EVENT_LOG" \
    SEARCH_R1_AUTODL_TEST_SHUTDOWN_BINARY="$FAKE_SHUTDOWN" \
    SEARCH_R1_AUTODL_TEST_BACKEND_RC="$backend_rc" \
    SEARCH_R1_AUTODL_WATCH_TIMEOUT_SECONDS=1 \
    SEARCH_R1_AUTODL_LOCK_WAIT_SECONDS=1 \
    AUTODL_ROOT="$PROJECT_ROOT" \
    AUTODL_PERSISTENT_ROOT="$CASE_ROOT" \
        bash "$WATCHDOG" "$mode" "$ATTEMPT"
}

run_worker() {
    SEARCH_R1_AUTODL_TEST_MODE=1 \
    SEARCH_R1_AUTODL_TEST_ROOT="$CASE_ROOT" \
    SEARCH_R1_AUTODL_TEST_SHUTDOWN_LOG="$EVENT_LOG" \
    SEARCH_R1_AUTODL_TEST_SHUTDOWN_BINARY="$FAKE_SHUTDOWN" \
    SEARCH_R1_AUTODL_TEST_BACKEND_RC=0 \
        bash "$WATCHDOG" --worker "$ATTEMPT/shutdown-capability.tsv"
}

run_detached_watchdog() {
    SEARCH_R1_AUTODL_TEST_MODE=1 \
    SEARCH_R1_AUTODL_TEST_ROOT="$CASE_ROOT" \
    SEARCH_R1_AUTODL_TEST_SHUTDOWN_LOG="$EVENT_LOG" \
    SEARCH_R1_AUTODL_TEST_SHUTDOWN_BINARY="$FAKE_SHUTDOWN" \
    SEARCH_R1_AUTODL_TEST_BACKEND_RC=0 \
    SEARCH_R1_AUTODL_WATCH_TIMEOUT_SECONDS=5 \
    AUTODL_ROOT="$PROJECT_ROOT" \
    AUTODL_PERSISTENT_ROOT="$CASE_ROOT" \
        bash "$WATCHDOG" "$ATTEMPT"
}

run_poisoned_detached_watchdog() {
    local poison_path="$1"
    PATH="$poison_path" \
    SEARCH_R1_AUTODL_TEST_MODE=1 \
    SEARCH_R1_AUTODL_TEST_ROOT="$CASE_ROOT" \
    SEARCH_R1_AUTODL_TEST_SHUTDOWN_LOG="$EVENT_LOG" \
    SEARCH_R1_AUTODL_TEST_SHUTDOWN_BINARY="$FAKE_SHUTDOWN" \
    SEARCH_R1_AUTODL_TEST_BACKEND_RC=0 \
    SEARCH_R1_AUTODL_WATCH_TIMEOUT_SECONDS=5 \
    AUTODL_ROOT="$PROJECT_ROOT" \
    AUTODL_PERSISTENT_ROOT="$CASE_ROOT" \
        /usr/bin/bash "$WATCHDOG" "$ATTEMPT"
}

wait_for_watchdog_state() {
    local state="$1"
    for _ in {1..100}; do
        if [[ -e "$ATTEMPT/$state" ]] && grep -Fxq "state:$state" "$EVENT_LOG"; then
            return 0
        fi
        sleep 0.05
    done
    return 1
}

assert_no_real_backend() {
    ! grep -Fq 'REAL_BACKEND_EXECUTED' "$EVENT_LOG"
}

assert_no_backend_event() {
    ! grep -Fq 'backend:' "$EVENT_LOG"
    assert_no_real_backend
}

assert_order() {
    local expected_backend="backend:$FAKE_SHUTDOWN argc=0"
    mapfile -t events <"$EVENT_LOG"
    [[ "${events[*]}" == "terminal-published state:shutdown-safe state:shutdown-requested $expected_backend state:shutdown-dispatched" ]]
    assert_no_real_backend
}

# Successful work authorizes a simulated no-argument dispatch.
new_case success 0 success
run_watchdog 0 --test-foreground
[[ -f "$ATTEMPT/shutdown-safe" && -f "$ATTEMPT/shutdown-requested" && -f "$ATTEMPT/shutdown-dispatched" ]]
[[ ! -e "$ATTEMPT/shutdown-skipped" && ! -e "$ATTEMPT/shutdown-failed" ]]
[[ "$(stat -c '%a' "$ATTEMPT/shutdown-capability.tsv")" == 600 ]]
assert_order

# The follow-up contract validates its own immutable result root without replacing legacy results.
new_case followup-success 0 success
replace_with_followup_results
run_watchdog 0 --test-foreground
[[ -f "$ATTEMPT/shutdown-safe" && -f "$ATTEMPT/shutdown-dispatched" ]]
assert_order

new_case followup-tampered 0 success
replace_with_followup_results
printf 'tampered\n' >>"$(cat "$ATTEMPT/result-root")/paired_results.csv"
run_watchdog 0 --test-foreground
[[ -f "$ATTEMPT/shutdown-skipped" && ! -e "$ATTEMPT/shutdown-safe" ]]
grep -Fq 'reason=authorization-revalidation-failed' "$ATTEMPT/shutdown-skipped"
assert_no_backend_event

# A complete scientific NO-GO is durable evidence and may safely stop billing.
new_case search-gate-no-go 0 success
replace_with_search_gate_results
run_watchdog 0 --test-foreground
[[ -f "$ATTEMPT/shutdown-safe" && -f "$ATTEMPT/shutdown-dispatched" ]]
assert_order

new_case search-gate-tampered 0 success
replace_with_search_gate_results
printf 'tampered\n' >>"$(cat "$ATTEMPT/result-root")/summary.json"
run_watchdog 0 --test-foreground
[[ -f "$ATTEMPT/shutdown-skipped" && ! -e "$ATTEMPT/shutdown-safe" ]]
grep -Fq 'reason=authorization-revalidation-failed' "$ATTEMPT/shutdown-skipped"
assert_no_backend_event

# A grouped-probe NO-GO is complete scientific evidence under its own contract.
new_case group-probe-no-go 0 success
replace_with_group_probe_results
run_watchdog 0 --test-foreground
[[ -f "$ATTEMPT/shutdown-safe" && -f "$ATTEMPT/shutdown-dispatched" ]]
assert_order

new_case group-probe-tampered 0 success
replace_with_group_probe_results
printf 'tampered\n' >>"$(cat "$ATTEMPT/result-root")/per_trajectory.jsonl"
run_watchdog 0 --test-foreground
[[ -f "$ATTEMPT/shutdown-skipped" && ! -e "$ATTEMPT/shutdown-safe" ]]
grep -Fq 'reason=authorization-revalidation-failed' "$ATTEMPT/shutdown-skipped"
assert_no_backend_event

# A native-gate NO-GO is also complete scientific evidence.
new_case qwen-native-no-go 0 success
replace_with_qwen_native_gate_results
run_watchdog 0 --test-foreground
[[ -f "$ATTEMPT/shutdown-safe" && -f "$ATTEMPT/shutdown-dispatched" ]]
assert_order

new_case qwen-native-tampered 0 success
replace_with_qwen_native_gate_results
printf 'tampered\n' >>"$(cat "$ATTEMPT/result-root")/sampling.json"
run_watchdog 0 --test-foreground
[[ -f "$ATTEMPT/shutdown-skipped" && ! -e "$ATTEMPT/shutdown-safe" ]]
grep -Fq 'reason=authorization-revalidation-failed' "$ATTEMPT/shutdown-skipped"
assert_no_backend_event

# A sealed two-step native smoke has its own exact result and predecessor contract.
new_case qwen-native-training-smoke 0 success
replace_with_native_training_smoke_results
run_watchdog 0 --test-foreground
[[ -f "$ATTEMPT/shutdown-safe" && -f "$ATTEMPT/shutdown-dispatched" ]]
assert_order

# A smoke scientific NO-GO is still a complete terminal result and may shut down.
new_case qwen-native-training-smoke-no-go 0 success
replace_with_native_training_smoke_results NO-GO
run_watchdog 0 --test-foreground
[[ -f "$ATTEMPT/shutdown-safe" && -f "$ATTEMPT/shutdown-dispatched" ]]
assert_order

new_case qwen-native-training-smoke-tampered 0 success
replace_with_native_training_smoke_results
printf 'checkpoint_tree_sha256=%s\n' "$(printf '9%.0s' {1..64})" \
    >>"$NATIVE_RESULT_DIR/checkpoint-tree.env"
run_watchdog 0 --test-foreground
[[ -f "$ATTEMPT/shutdown-skipped" && ! -e "$ATTEMPT/shutdown-safe" ]]
grep -Fq 'reason=authorization-revalidation-failed' "$ATTEMPT/shutdown-skipped"
assert_no_backend_event

# Re-sealing cannot hide a missing registered smoke check.
new_case qwen-native-training-smoke-check-missing 0 success
replace_with_native_training_smoke_results
python3 - "$NATIVE_RESULT_DIR/smoke-decision.json" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
payload = json.loads(path.read_bytes())
del payload["checks"]["wandb_offline_history"]
path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
PY
seal_native_training_results qwen-native-training-smoke-v1 qwen-native-training-smoke
run_watchdog 0 --test-foreground
[[ -f "$ATTEMPT/shutdown-skipped" && ! -e "$ATTEMPT/shutdown-safe" ]]
grep -Fq 'reason=authorization-revalidation-failed' "$ATTEMPT/shutdown-skipped"
assert_no_backend_event

# A post-R scientific NO-GO is complete evidence and omits B/C and endpoint eval rows.
new_case qwen-native-training-main-no-go 0 success
replace_with_native_training_main_results false
run_watchdog 0 --test-foreground
[[ -f "$ATTEMPT/shutdown-safe" && -f "$ATTEMPT/shutdown-dispatched" ]]
assert_order

# A predecessor marker cannot borrow a same-named result from a mismatched outer attempt.
new_case qwen-native-training-main-predecessor-outer-tampered 0 success
replace_with_native_training_main_results false
printf 'qwen-native-training-main-v1\n' >"$NATIVE_SMOKE_OUTER/result-contract"
run_watchdog 0 --test-foreground
[[ -f "$ATTEMPT/shutdown-skipped" && ! -e "$ATTEMPT/shutdown-safe" ]]
grep -Fq 'reason=authorization-revalidation-failed' "$ATTEMPT/shutdown-skipped"
assert_no_backend_event

# A re-sealed G3 summary still has to bind the exact evaluated trace.
new_case qwen-native-training-main-g3-input-tampered 0 success
replace_with_native_training_main_results false
python3 - "$NATIVE_RESULT_DIR/r-g3-analysis/summary.json" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
payload = json.loads(path.read_bytes())
payload["input"]["trace_sha256"] = "0" * 64
path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
PY
seal_native_training_results qwen-native-training-main-v1 qwen-native-training-main
run_watchdog 0 --test-foreground
[[ -f "$ATTEMPT/shutdown-skipped" && ! -e "$ATTEMPT/shutdown-safe" ]]
grep -Fq 'reason=authorization-revalidation-failed' "$ATTEMPT/shutdown-skipped"
assert_no_backend_event

# A main GO must include both branches, both endpoint evals, and paired outputs.
new_case qwen-native-training-main-go 0 success
replace_with_native_training_main_results true
run_watchdog 0 --test-foreground
[[ -f "$ATTEMPT/shutdown-safe" && -f "$ATTEMPT/shutdown-dispatched" ]]
assert_order

# A re-sealed paired report cannot substitute another control endpoint.
new_case qwen-native-training-main-paired-endpoint-tampered 0 success
replace_with_native_training_main_results true
python3 - "$NATIVE_RESULT_DIR/paired/summary.json" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
payload = json.loads(path.read_bytes())
payload["formal_contract"]["endpoints"]["control"]["checkpoint_digest"] = "0" * 64
path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
PY
seal_native_training_results qwen-native-training-main-v1 qwen-native-training-main
run_watchdog 0 --test-foreground
[[ -f "$ATTEMPT/shutdown-skipped" && ! -e "$ATTEMPT/shutdown-safe" ]]
grep -Fq 'reason=authorization-revalidation-failed' "$ATTEMPT/shutdown-skipped"
assert_no_backend_event

# Rehashing a forged branch decision cannot bypass semantic contract checks.
new_case qwen-native-training-main-forged 0 success
replace_with_native_training_main_results true
printf '%s\n' \
    '{"analysis_decision":"GO","branches_authorized":false,"cost_contrast_group_count":9,"cost_contrast_group_minimum":8,"decision":"NO-GO","schema":"qwen-native-post-r-gate-v1"}' \
    >"$NATIVE_RESULT_DIR/branch-decision.json"
seal_native_training_results qwen-native-training-main-v1 qwen-native-training-main
run_watchdog 0 --test-foreground
[[ -f "$ATTEMPT/shutdown-skipped" && ! -e "$ATTEMPT/shutdown-safe" ]]
grep -Fq 'reason=authorization-revalidation-failed' "$ATTEMPT/shutdown-skipped"
assert_no_backend_event

# The real detached path cannot run until its PID and admission nonce are durable.
new_case detached-success 0 success
run_detached_watchdog
wait_for_watchdog_state shutdown-dispatched
[[ -f "$ATTEMPT/shutdown-watchdog-pid" && -f "$ATTEMPT/shutdown-watchdog-admitted" ]]
[[ -f "$ATTEMPT/shutdown-safe" && -f "$ATTEMPT/shutdown-requested" ]]
assert_order

# A caller-controlled PATH cannot shadow authorization or detached-launch tools.
new_case poisoned-path 0 success
poison_path="$CASE_ROOT/poison"
poison_event="$CASE_ROOT/poison-used"
mkdir "$poison_path"
for tool in readlink sha256sum git stat nohup setsid; do
    printf '%s\n' '#!/usr/bin/env bash' "printf used >>'$poison_event'" 'exit 99' \
        >"$poison_path/$tool"
    chmod 0755 "$poison_path/$tool"
done
run_poisoned_detached_watchdog "$poison_path"
wait_for_watchdog_state shutdown-dispatched
[[ ! -e "$poison_event" ]]
assert_order

# Successful work also requires a complete, digest-bound comparison package.
new_case results-missing 0 success
rm -f -- "$PROJECT_ROOT/runs/comparison/results.md"
run_watchdog 0 --test-foreground
[[ -f "$ATTEMPT/shutdown-skipped" && ! -e "$ATTEMPT/shutdown-safe" ]]
grep -Fq 'reason=authorization-revalidation-failed' "$ATTEMPT/shutdown-skipped"
assert_no_backend_event

new_case results-tampered 0 success
printf 'tampered\n' >>"$PROJECT_ROOT/runs/comparison/results.csv"
run_watchdog 0 --test-foreground
[[ -f "$ATTEMPT/shutdown-skipped" && ! -e "$ATTEMPT/shutdown-safe" ]]
grep -Fq 'reason=authorization-revalidation-failed' "$ATTEMPT/shutdown-skipped"
assert_no_backend_event

new_case results-substituted 0 success
printf 'valid replacement\n' >>"$PROJECT_ROOT/runs/comparison/results.md"
(
    cd "$PROJECT_ROOT/runs/comparison"
    sha256sum results.csv results.md lineage.tsv >comparison.sha256
)
sha256sum "$PROJECT_ROOT/runs/comparison/comparison.sha256" | cut -d' ' -f1 \
    >"$PROJECT_ROOT/manifests/gpu.ok"
run_watchdog 0 --test-foreground
[[ -f "$ATTEMPT/shutdown-skipped" && ! -e "$ATTEMPT/shutdown-safe" ]]
grep -Fq 'reason=authorization-revalidation-failed' "$ATTEMPT/shutdown-skipped"
assert_no_backend_event

new_case attempt-digest-missing 0 success
rm -f -- "$ATTEMPT/comparison-digest"
run_watchdog 0 --test-foreground
[[ -f "$ATTEMPT/shutdown-skipped" && ! -e "$ATTEMPT/shutdown-safe" ]]
grep -Fq 'reason=authorization-revalidation-failed' "$ATTEMPT/shutdown-skipped"
assert_no_backend_event

# Failed work is still terminal and may request shutdown without changing its exit code.
new_case work-failure 7 failed
run_watchdog 0 --test-foreground
[[ -f "$ATTEMPT/shutdown-safe" && -f "$ATTEMPT/shutdown-requested" && -f "$ATTEMPT/shutdown-dispatched" ]]
[[ "$(cat "$ATTEMPT/exit-code")" == 7 ]]
assert_order

# Exit 75 means this attempt never acquired the global phase lock. Even after
# the competing worker releases it, this terminal attempt must not shut down.
new_case released-lock-conflict 75 failed
run_watchdog 0 --test-foreground
[[ -f "$ATTEMPT/shutdown-skipped" && ! -e "$ATTEMPT/shutdown-safe" ]]
grep -Fq 'reason=lock-conflict' "$ATTEMPT/shutdown-skipped"
assert_no_backend_event

# Partial terminal publication fails closed.
new_case incomplete 0 success
rm -f -- "$ATTEMPT/terminal" "$ATTEMPT/.success"
printf 'incomplete-created\n' >"$EVENT_LOG"
run_watchdog 0 --test-foreground
[[ -f "$ATTEMPT/shutdown-skipped" ]]
grep -Fq 'reason=terminal-evidence-incomplete' "$ATTEMPT/shutdown-skipped"
assert_no_backend_event

# A mismatched log sentinel is not durable terminal evidence.
new_case bad-sentinel 0 success
printf 'AUTODL_PHASE_TERMINAL state=failed exit_code=1\n' >"$ATTEMPT/phase.log"
run_watchdog 0 --test-foreground
[[ -f "$ATTEMPT/shutdown-skipped" && ! -e "$ATTEMPT/shutdown-safe" ]]
grep -Fq 'reason=terminal-evidence-incomplete' "$ATTEMPT/shutdown-skipped"
assert_no_backend_event

# A competing phase lock always keeps the guest running.
new_case lock-conflict 0 success
ready="$CASE_ROOT/lock-ready"
flock "$PROJECT_ROOT/state/phase.lock" bash -c "printf ready >'$ready'; sleep 10" &
LOCK_HOLDER_PID=$!
for _ in {1..50}; do [[ -f "$ready" ]] && break; sleep 0.02; done
[[ -f "$ready" ]]
run_watchdog 0 --test-foreground
[[ -f "$ATTEMPT/shutdown-skipped" ]]
grep -Fq 'reason=phase-lock-unavailable' "$ATTEMPT/shutdown-skipped"
assert_no_backend_event
kill "$LOCK_HOLDER_PID" 2>/dev/null || true
wait "$LOCK_HOLDER_PID" 2>/dev/null || true
LOCK_HOLDER_PID=''

# A changed mode-0600 capability cannot be reused by a worker.
new_case capability-tamper 0 success
run_watchdog 0 --test-foreground-dry-run
printf 'tampered\ttrue\n' >>"$ATTEMPT/shutdown-capability.tsv"
printf 'capability-tampered\n' >"$EVENT_LOG"
set +e
run_worker
worker_rc=$?
set -e
[[ "$worker_rc" == 1 ]]
assert_no_backend_event

# A dirty detached checkout is rejected before arming.
new_case checkout-tamper 0 success
printf '\n' >>"$WATCHDOG"
set +e
run_watchdog 0 --test-foreground
checkout_rc=$?
set -e
[[ "$checkout_rc" == 64 && ! -e "$ATTEMPT/shutdown-capability.tsv" ]]
assert_no_backend_event

# An older exact path cannot borrow a newer GPU attempt's global results.
new_case stale-attempt 0 success
printf '%s\n' "$PROJECT_ROOT/state/attempts/gpu/20260720T130000Z-999-1" \
    >"$PROJECT_ROOT/state/latest/gpu"
set +e
run_watchdog 0 --test-foreground
stale_rc=$?
set -e
[[ "$stale_rc" == 64 && ! -e "$ATTEMPT/shutdown-capability.tsv" ]]
assert_no_backend_event

# Dry-run validates terminal state and the lock but never publishes a request.
new_case dry-run 0 success
run_watchdog 0 --test-foreground-dry-run
[[ -f "$ATTEMPT/shutdown-skipped" && ! -e "$ATTEMPT/shutdown-safe" ]]
grep -Fq 'reason=dry-run' "$ATTEMPT/shutdown-skipped"
assert_no_backend_event

# Backend failure is distinct from authorization skip and preserves work state.
new_case backend-failure 0 success
set +e
run_watchdog 9 --test-foreground
watchdog_rc=$?
set -e
[[ "$watchdog_rc" == 1 ]]
[[ -f "$ATTEMPT/shutdown-safe" && -f "$ATTEMPT/shutdown-requested" && -f "$ATTEMPT/shutdown-failed" ]]
[[ ! -e "$ATTEMPT/shutdown-skipped" && "$(cat "$ATTEMPT/exit-code")" == 0 ]]
grep -Fq 'backend_exit_code=9' "$ATTEMPT/shutdown-failed"
mapfile -t events <"$EVENT_LOG"
[[ "${events[0]}" == terminal-published ]]
[[ "${events[1]}" == state:shutdown-safe ]]
[[ "${events[2]}" == state:shutdown-requested ]]
[[ "${events[3]}" == "backend:$FAKE_SHUTDOWN argc=0" ]]
[[ "${events[4]}" == state:shutdown-failed ]]
assert_no_real_backend

printf 'AutoDL shutdown watchdog tests passed.\n'
