#!/usr/bin/env bash
set -Eeuo pipefail

AUTODL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
CHECKOUT="$(cd -- "$AUTODL_DIR/../.." && pwd -P)"
ENTRYPOINT="$AUTODL_DIR/11_gpu_qwen_native_g3_only.sh"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TEST_ROOT="$(mktemp -d)"
trap 'rm -rf -- "$TEST_ROOT"' EXIT

[[ -f "$ENTRYPOINT" && ! -L "$ENTRYPOINT" ]]
bash -n "$ENTRYPOINT"

PROJECT_ROOT="$TEST_ROOT/project"
export AUTODL_ROOT="$PROJECT_ROOT"
export GPU_COUNT=2
export AUTODL_PRICE_PER_HOUR=5.76
export TRAIN_BATCH_SIZE=8
export MAX_RESPONSE_LENGTH=500
mkdir -p "$PROJECT_ROOT/envs/train/bin" "$PROJECT_ROOT/models/Qwen3.5-2B" \
    "$PROJECT_ROOT/operator"
ln -s "$CHECKOUT" "$PROJECT_ROOT/checkout"

# The sealed entrypoint is sourced from the canonical checkout. Tests deploy a
# separate regular operator copy, matching the production CPU/GPU handoff.
source "$ENTRYPOINT"
cp "$ENTRYPOINT" "$G3_DEPLOY_PATH"
chmod 0755 "$G3_DEPLOY_PATH"
G3_RUNNER_PATH="$G3_DEPLOY_PATH"
export QWEN_NATIVE_R60_EVIDENCE="$G3_EXPECTED_R60_MARKER"

fail() {
    printf '%s\n' "$1" >&2
    exit 1
}

# Missing and content-tampered exact markers fail before recursive parsing.
if verify_r60_only_evidence "$G3_EXPECTED_R60_MARKER" \
        "$G3_EXPECTED_CHECKOUT" "$G3_EXPECTED_HANDOFF" "$MODEL_DIR" \
        "$G3_EXPECTED_BASE" "$G3_EXPECTED_DATA" >/dev/null 2>&1; then
    fail 'G3-only accepted a missing R60 marker.'
fi
mkdir -p "$(dirname -- "$G3_EXPECTED_R60_MARKER")" \
    "$NATIVE_TRAIN_RESULTS_ROOT/attempts/$G3_R60_ATTEMPT" \
    "$ATTEMPTS_ROOT/gpu/$G3_R60_ATTEMPT"
printf '%064d\n' 0 >"$G3_EXPECTED_R60_MARKER"
printf 'tampered evidence\n' \
    >"$NATIVE_TRAIN_RESULTS_ROOT/attempts/$G3_R60_ATTEMPT/evidence.sha256"
R60_OUTER_FIXTURE="$ATTEMPTS_ROOT/gpu/$G3_R60_ATTEMPT"
printf 'failed\n' >"$R60_OUTER_FIXTURE/terminal"
printf '200\n' >"$R60_OUTER_FIXTURE/exit-code"
: >"$R60_OUTER_FIXTURE/.failed"
if verify_r60_only_evidence "$G3_EXPECTED_R60_MARKER" \
        "$G3_EXPECTED_CHECKOUT" "$G3_EXPECTED_HANDOFF" "$MODEL_DIR" \
        "$G3_EXPECTED_BASE" "$G3_EXPECTED_DATA" >/dev/null 2>&1; then
    fail 'G3-only accepted a tampered R60 marker digest.'
fi

CALLS="$TEST_ROOT/calls.log"
export CALLS
: >"$CALLS"

# Use the real receipt writer/checker while replacing only the expensive,
# already-covered recursive R60 and model/data verification.
expected_commit() { printf '%s\n' "$G3_EXPECTED_CHECKOUT"; }
verify_checkout() { return 0; }
verify_g3_foundation() {
    G3_R60_DIGEST="$G3_EXPECTED_R60_DIGEST"
    G3_R60_CHECKPOINT="$RUNS_ROOT/reproduce/attempts/source/checkpoints/actor/global_step_60"
    G3_R60_CHECKPOINT_DIGEST="$(tree_sha256 "$G3_R60_CHECKPOINT")"
    G3_R60_TRACE_DIGEST="$G3_EXPECTED_TRAIN_TRACE"
    G3_R60_TRACE_MANIFEST_DIGEST="$G3_EXPECTED_TRAIN_TRACE_MANIFEST"
}
sync_path() { return 0; }

G3_R60_CHECKPOINT="$RUNS_ROOT/reproduce/attempts/source/checkpoints/actor/global_step_60"
mkdir -p "$G3_R60_CHECKPOINT" "$(dirname -- "$G3_CONFIG")" \
    "$(dirname -- "$NATIVE_TRAIN_G3_DATA")"
printf 'checkpoint\n' >"$G3_R60_CHECKPOINT/model.bin"
printf 'g3 data\n' >"$NATIVE_TRAIN_G3_DATA"
printf 'g3 config\n' >"$G3_CONFIG"
printf '%s\n' "$G3_EXPECTED_HANDOFF" >"$MANIFEST_DIR/cpu.ok"
printf 'base model\n' >"$MODEL_DIR/model.bin"

prepare_g3_cpu_receipt
[[ -f "$G3_CPU_RECEIPT" && ! -L "$G3_CPU_RECEIPT" &&
    -f "$G3_CPU_RECEIPT.sha256" && ! -L "$G3_CPU_RECEIPT.sha256" ]]
grep -Fxq 'gpu_evaluation_authorized=false' "$G3_CPU_RECEIPT"
grep -Fxq 'manual_gpu_start_required=true' "$G3_CPU_RECEIPT"
[[ ! -s "$CALLS" ]] || fail 'CPU preparation started a GPU job.'

# The rest of the fixture exercises the real one-eval dispatcher and evidence
# publisher, with deterministic fake evaluation/analyzer outputs.
mkdir -p "$(dirname -- "$NATIVE_TRAIN_MANIFEST")" "$TRAIN_ENV/bin"
for path in \
    "$NATIVE_TRAIN_MANIFEST" "$NATIVE_TRAIN_MANIFEST.sha256" \
    "$NATIVE_TRAIN_CATALOG" \
    "$NATIVE_TRAIN_DATA_DIR/search_mix_answer_quality_exclusions.v1.json" \
    "$NATIVE_TRAIN_DATA_DIR/train_512.parquet" \
    "$NATIVE_TRAIN_DATA_DIR/val_128.parquet" \
    "$NATIVE_TRAIN_DATA_DIR/nq_test_128_native_v4.parquet" \
    "$NATIVE_TRAIN_DATA_DIR/multihop_eval_256_native_v4.parquet" \
    "$HANDOFF" "$HANDOFF.sha256" "$MANIFEST_DIR/git.ok" \
    "$MANIFEST_DIR/checkout-tree.sha256"; do
    mkdir -p "$(dirname -- "$path")"
    printf 'fixture:%s\n' "${path#"$PROJECT_ROOT/"}" >"$path"
done

SYSTEM_PYTHON="$PYTHON_BIN"
export SYSTEM_PYTHON
cat >"$TRAIN_ENV/bin/python" <<'SH'
#!/usr/bin/env bash
set -Eeuo pipefail
tool="${1##*/}"
if [[ "$tool" == qwen_native_gate_analysis.py ]]; then
    output=''
    while (($#)); do
        if [[ "$1" == --output-dir ]]; then
            output="$2"
            break
        fi
        shift
    done
    [[ -n "$output" ]]
    mkdir -p "$output"
    printf '{"decision":"%s"}\n' "${ANALYSIS_OUTCOME:-GO}" >"$output/summary.json"
    printf '# G3 analysis\n' >"$output/summary.md"
    printf '{"decision":"%s"}\n' "${ANALYSIS_OUTCOME:-GO}" >"$output/go_no_go.json"
    printf '{}\n' >"$output/per_trajectory.jsonl"
    printf '{}\n' >"$output/per_question.jsonl"
    exit 0
fi
exec "$SYSTEM_PYTHON" "$@"
SH
chmod 0755 "$TRAIN_ENV/bin/python"

RUN_SUFFIX=''
FAIL_EVAL=0
DRIFT_CHECKPOINT_AFTER_EVAL=0
run_job() {
    local mode="$1" variant="$2" argument="${3:-}" parent="${4:-}" digest="${5:-}"
    printf 'run_job|%s|%s|%s|%s|%s|eval=%s|rows=%s|group=%s\n' \
        "$mode" "$variant" "$argument" "$parent" "$digest" \
        "$EVAL_DATA_FILE" "$EVAL_EXPECTED_ROWS" "$EVAL_GROUP_SIZE" >>"$CALLS"
    ((FAIL_EVAL == 0)) || return "$FAIL_EVAL"
    LAST_RUN_DIR="$RUNS_ROOT/eval/$variant/attempts/fake-$RUN_SUFFIX"
    mkdir -p "$LAST_RUN_DIR/traces" "$LAST_RUN_DIR/wandb/offline-run-test"
    printf 'eval log\n' >"$LAST_RUN_DIR/train.log"
    printf 'resolved config\n' >"$LAST_RUN_DIR/resolved-config.yaml"
    printf 'run env\n' >"$LAST_RUN_DIR/run.env"
    printf '{}\n' >"$LAST_RUN_DIR/traces/eval_predictions.jsonl"
    printf '{}\n' >"$LAST_RUN_DIR/traces/eval_predictions.manifest.json"
    printf 'manifest sidecar\n' \
        >"$LAST_RUN_DIR/traces/eval_predictions.manifest.json.sha256"
    printf 'wandb history\n' \
        >"$LAST_RUN_DIR/wandb/offline-run-test/run-test.wandb"
    if ((DRIFT_CHECKPOINT_AFTER_EVAL == 1)); then
        printf 'drift\n' >"$G3_R60_CHECKPOINT/post-eval-drift.bin"
    fi
}
verify_native_trace_identity() {
    printf '%s\t%s\n' "$(printf '5%.0s' {1..64})" "$(printf '6%.0s' {1..64})"
}
validate_r_g3_analysis() {
    printf '%s\t%s\n' "${ANALYSIS_OUTCOME:-GO}" "${CONTRAST_COUNT:-8}"
}
write_wandb_receipt() {
    printf '{"decision":"GO"}\n' >"$1/wandb-receipt.json"
}
verify_wandb_receipt() { return 0; }
verify_checkout() { return 0; }
verify_g3_runtime_config() { return 0; }

G3_R60_RESULTS="$NATIVE_TRAIN_RESULTS_ROOT/attempts/$G3_R60_ATTEMPT"
G3_R60_OUTER="$R60_OUTER_FIXTURE"
G3_R60_MANIFEST="$G3_R60_RESULTS/evidence.sha256"
for path in "$G3_R60_RESULTS/contract.env" "$G3_R60_RESULTS/lineage.tsv" \
    "$G3_R60_RESULTS/run-index.tsv" "$G3_R60_RESULTS/checkpoint-tree.env" \
    "$G3_R60_OUTER/r60-only-complete.env"; do
    printf 'R60 fixture\n' >"$path"
done
# Restore the exact marker text used by the G3-only contract. The dispatcher
# consumes the already-verified globals; recursive verification is tested above.
printf '%s\n' "$G3_EXPECTED_R60_DIGEST" >"$G3_EXPECTED_R60_MARKER"

RUNNER_DIGEST="$(file_sha256 "$G3_RUNNER_PATH")"
DATA_DIGEST="$(file_sha256 "$NATIVE_TRAIN_MANIFEST")"
BASE_DIGEST="$(tree_sha256 "$MODEL_DIR")"
G3_PREFLIGHT_COMMIT="$G3_EXPECTED_CHECKOUT"
G3_PREFLIGHT_HANDOFF="$G3_EXPECTED_HANDOFF"
G3_PREFLIGHT_BASE_DIGEST="$BASE_DIGEST"
G3_PREFLIGHT_DATA_DIGEST="$DATA_DIGEST"
G3_PREFLIGHT_RUNNER_DIGEST="$RUNNER_DIGEST"

assert_one_g3_eval_only() {
    local -a jobs
    mapfile -t jobs < <(grep '^run_job|' "$CALLS")
    [[ "${#jobs[@]}" == 1 ]] || fail 'G3-only did not execute exactly one job.'
    [[ "${jobs[0]}" == "run_job|eval|qwen_native_g3|$G3_R60_CHECKPOINT||$G3_R60_CHECKPOINT_DIGEST|eval=$NATIVE_TRAIN_G3_DATA|rows=64|group=5" ]] ||
        fail 'G3-only executed an unexpected job contract.'
    ! grep -Eq '^run_job\|train\||^run_job\|eval\|qwen_native_[arbc]_' "$CALLS" ||
        fail 'G3-only started training or a downstream endpoint.'
}

# Drift before evaluation is rejected without starting a paid job.
: >"$CALLS"
printf 'pre-eval drift\n' >"$G3_R60_CHECKPOINT/pre-eval-drift.bin"
set +e
qwen_native_train_pipeline \
    "$ATTEMPTS_ROOT/gpu/20260729T010000Z-1-1" \
    "$G3_EXPECTED_CHECKOUT" "$G3_EXPECTED_HANDOFF" "$MODEL_DIR" "$BASE_DIGEST" \
    >/dev/null 2>&1
rc=$?
set -e
[[ "$rc" != 0 && ! -s "$CALLS" ]] ||
    fail 'Pre-evaluation checkpoint drift was not rejected before the job.'
rm "$G3_R60_CHECKPOINT/pre-eval-drift.bin"

# Drift introduced by the eval job is caught by final seal revalidation.
: >"$CALLS"
RUN_SUFFIX=post-drift
DRIFT_CHECKPOINT_AFTER_EVAL=1
POST_DRIFT_ATTEMPT="$ATTEMPTS_ROOT/gpu/20260729T010100Z-1-2"
mkdir -p "$POST_DRIFT_ATTEMPT"
set +e
qwen_native_train_pipeline "$POST_DRIFT_ATTEMPT" \
    "$G3_EXPECTED_CHECKOUT" "$G3_EXPECTED_HANDOFF" "$MODEL_DIR" "$BASE_DIGEST" \
    >/dev/null 2>&1
rc=$?
set -e
[[ "$rc" != "$G3_ONLY_TERMINAL_CODE" ]]
assert_one_g3_eval_only
[[ ! -e "$MANIFEST_DIR/$G3_ONLY_NAMESPACE/$(basename -- "$POST_DRIFT_ATTEMPT").ok" ]]
rm "$G3_R60_CHECKPOINT/post-eval-drift.bin"
DRIFT_CHECKPOINT_AFTER_EVAL=0

# An engineering eval failure preserves its own code and cannot seal evidence.
: >"$CALLS"
RUN_SUFFIX=engineering-failure
FAIL_EVAL=23
FAIL_ATTEMPT="$ATTEMPTS_ROOT/gpu/20260729T010200Z-1-3"
mkdir -p "$FAIL_ATTEMPT"
set +e
qwen_native_train_pipeline "$FAIL_ATTEMPT" \
    "$G3_EXPECTED_CHECKOUT" "$G3_EXPECTED_HANDOFF" "$MODEL_DIR" "$BASE_DIGEST" \
    >/dev/null 2>&1
rc=$?
set -e
[[ "$rc" == 23 ]]
assert_one_g3_eval_only
[[ ! -e "$MANIFEST_DIR/$G3_ONLY_NAMESPACE/$(basename -- "$FAIL_ATTEMPT").ok" ]]
FAIL_EVAL=0

printf 'do-not-change\n' >"$NATIVE_TRAIN_RESULTS_ROOT/latest-main"

run_scientific_case() {
    local outcome="$1" contrast="$2" attempt_name="$3" candidate="$4"
    local outer results marker rc
    : >"$CALLS"
    export ANALYSIS_OUTCOME="$outcome" CONTRAST_COUNT="$contrast"
    RUN_SUFFIX="${outcome,,}-$contrast"
    outer="$ATTEMPTS_ROOT/gpu/$attempt_name"
    mkdir -p "$outer"
    set +e
    qwen_native_train_pipeline "$outer" "$G3_EXPECTED_CHECKOUT" \
        "$G3_EXPECTED_HANDOFF" "$MODEL_DIR" "$BASE_DIGEST"
    rc=$?
    set -e
    [[ "$rc" == "$G3_ONLY_TERMINAL_CODE" ]] ||
        fail "Scientific $outcome did not return controlled outer status 201."
    assert_one_g3_eval_only
    results="$NATIVE_TRAIN_RESULTS_ROOT/attempts/$attempt_name"
    marker="$MANIFEST_DIR/$G3_ONLY_NAMESPACE/$attempt_name.ok"
    [[ -s "$marker" && -s "$results/evidence.sha256" ]]
    [[ "$(file_sha256 "$results/evidence.sha256")" == "$(tr -d '\r\n' <"$marker")" ]]
    grep -Fxq 'schema=qwen-native-training-g3-only-v1' "$results/contract.env"
    grep -Fxq 'stage_order=G3' "$results/contract.env"
    grep -Fxq "analysis_decision=$outcome" "$results/contract.env"
    grep -Fxq "branch_candidate_passed=$candidate" "$results/contract.env"
    grep -Fxq 'branch_training_authorized=false' "$results/contract.env"
    grep -Fxq 'manual_review_required=true' "$results/contract.env"
    grep -Fxq 'controlled_outer_exit_code=201' "$results/contract.env"
    [[ "$(wc -l <"$results/lineage.tsv")" == 2 ]]
    [[ "$(wc -l <"$results/run-index.tsv")" == 2 ]]
    grep -Fq '"branch_training_authorized":false' "$results/branch-decision.json"
    grep -Fq '"manual_review_required":true' "$results/branch-decision.json"
    grep -Fxq 'do-not-change' "$NATIVE_TRAIN_RESULTS_ROOT/latest-main"
    [[ "$(tr -d '\r\n' <"$NATIVE_TRAIN_RESULTS_ROOT/latest-g3-only")" == "$results" ]]
    grep -Fxq 'branch_training_authorized=false' "$outer/g3-only-complete.env"
}

run_scientific_case GO 8 20260729T010300Z-1-4 true
run_scientific_case NO-GO 12 20260729T010400Z-1-5 false

unset ANALYSIS_OUTCOME CONTRAST_COUNT
printf 'Qwen native G3-only pipeline tests passed.\n'
