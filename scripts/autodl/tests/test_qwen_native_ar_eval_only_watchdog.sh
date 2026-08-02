#!/usr/bin/env bash
set -Eeuo pipefail

AUTODL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
SOURCE_WATCHDOG="$AUTODL_DIR/15_watch_qwen_native_ar_eval_only.sh"
TEST_ROOT="$(mktemp -d)"
LOCK_HOLDER_PID=''
trap '[[ -z "$LOCK_HOLDER_PID" ]] || kill "$LOCK_HOLDER_PID" 2>/dev/null || true; rm -rf -- "$TEST_ROOT"' EXIT

readonly HANDOFF_DIGEST="$(printf 'c%.0s' {1..64})"
readonly BASE_DIGEST="$(printf 'd%.0s' {1..64})"
readonly R60_EVIDENCE_DIGEST="$(printf 'e%.0s' {1..64})"
readonly R60_DIGEST="$(printf 'f%.0s' {1..64})"
readonly DATA_DIGEST="$(printf '1%.0s' {1..64})"

case_number=0
GIT_COMMIT=''
TREE_DIGEST=''
CASE_ROOT=''
PROJECT_ROOT=''
ATTEMPT=''
WATCHDOG=''
RUNNER=''
RUNNER_DIGEST=''
RECEIPT=''
RECEIPT_DIGEST=''
EVENT_LOG=''
FAKE_SHUTDOWN=''
BASE_MODEL=''
R60_CHECKPOINT=''
DATA_MANIFEST=''
R60_MARKER=''
RESULTS=''
RESULT_MARKER=''
declare -a EVIDENCE_PATHS=()

fail() {
    printf 'A/R watchdog test failed: %s\n' "$*" >&2
    exit 1
}

write_terminal() {
    local rc="$1" state="$2" marker
    rm -f -- "$ATTEMPT/.success" "$ATTEMPT/.failed" "$ATTEMPT/.running" "$ATTEMPT/.starting"
    if [[ "$state" == success ]]; then marker=.success; else marker=.failed; fi
    printf 'outer work log\nAUTODL_PHASE_TERMINAL state=%s exit_code=%s\n' "$state" "$rc" \
        >"$ATTEMPT/phase.log"
    printf '%s\n' "$rc" >"$ATTEMPT/exit-code"
    printf '%s\n' "$state" >"$ATTEMPT/terminal"
    : >"$ATTEMPT/$marker"
}

make_case() {
    local name="$1" rc="$2" state="$3" receipt_dir receipt_relative sidecar
    case_number=$((case_number + 1))
    CASE_ROOT="$TEST_ROOT/$name"
    PROJECT_ROOT="$CASE_ROOT/project"
    ATTEMPT="$PROJECT_ROOT/state/attempts/gpu/20260801T120000Z-$case_number-1"
    WATCHDOG="$PROJECT_ROOT/operator/15_watch_qwen_native_ar_eval_only.sh"
    RUNNER="$PROJECT_ROOT/operator/14_gpu_qwen_native_ar_eval_only.sh"
    EVENT_LOG="$CASE_ROOT/events.log"
    FAKE_SHUTDOWN="$CASE_ROOT/fake-shutdown"
    BASE_MODEL="$PROJECT_ROOT/models/Qwen3.5-2B"
    R60_CHECKPOINT="$PROJECT_ROOT/runs/reproduce/attempts/r60/checkpoints/actor/global_step_60"
    DATA_MANIFEST="$PROJECT_ROOT/data/search_mix_qwen35_native_v4/manifest.json"
    R60_MARKER="$PROJECT_ROOT/manifests/qwen-native-training-r60-only/r60.ok"
    mkdir -p "$PROJECT_ROOT/operator" "$PROJECT_ROOT/state/attempts/gpu" \
        "$PROJECT_ROOT/state/latest" "$PROJECT_ROOT/manifests/qwen-native-training-r60-only" \
        "$BASE_MODEL" "$R60_CHECKPOINT" "$(dirname -- "$DATA_MANIFEST")" "$ATTEMPT" \
        "$PROJECT_ROOT/checkout"
    git -C "$PROJECT_ROOT/checkout" init -q
    git -C "$PROJECT_ROOT/checkout" config user.name 'A/R Watchdog Test'
    git -C "$PROJECT_ROOT/checkout" config user.email 'watchdog@example.invalid'
    printf 'sealed checkout\n' >"$PROJECT_ROOT/checkout/README.test"
    git -C "$PROJECT_ROOT/checkout" add README.test
    git -C "$PROJECT_ROOT/checkout" commit -q -m sealed
    git -C "$PROJECT_ROOT/checkout" checkout --detach -q
    GIT_COMMIT="$(git -C "$PROJECT_ROOT/checkout" rev-parse HEAD)"
    TREE_DIGEST="$(
        cd "$PROJECT_ROOT/checkout"
        find . -path './.git' -prune -o -type f -print0 \
            | LC_ALL=C sort -z | xargs -0 -r sha256sum | sha256sum | cut -d' ' -f1
    )"
    cp "$SOURCE_WATCHDOG" "$WATCHDOG"
    chmod 0755 "$WATCHDOG"
    printf '#!/usr/bin/env bash\nexit 0\n' >"$RUNNER"
    chmod 0755 "$RUNNER"
    RUNNER_DIGEST="$(sha256sum "$RUNNER" | cut -d' ' -f1)"
    printf '%s\n' "$RUNNER" >"$ATTEMPT/entrypoint"
    printf 'base\n' >"$BASE_MODEL/model.bin"
    printf 'r60\n' >"$R60_CHECKPOINT/model.bin"
    printf '{"schema_version":5}\n' >"$DATA_MANIFEST"
    printf '%s\n' "$R60_EVIDENCE_DIGEST" >"$R60_MARKER"
    printf '%s\n' "$GIT_COMMIT" >"$PROJECT_ROOT/manifests/git.ok"
    printf '%s\n' "$TREE_DIGEST" >"$PROJECT_ROOT/manifests/checkout-tree.sha256"
    printf '%s\n' "$HANDOFF_DIGEST" >"$PROJECT_ROOT/manifests/cpu.ok"
    : >"$PROJECT_ROOT/state/phase.lock"
    printf '%s\n' "$ATTEMPT" >"$PROJECT_ROOT/state/latest/gpu"
    write_terminal "$rc" "$state"
    receipt_dir="$PROJECT_ROOT/manifests/qwen-native-ar-eval-only-cpu/$RUNNER_DIGEST"
    RECEIPT="$receipt_dir/receipt.env"
    mkdir -p "$receipt_dir"
    printf '%s\n' \
        'schema=qwen-native-ar-eval-only-cpu-v1' \
        'cpu_prepare_only=true' \
        'gpu_started=false' \
        'training_executed=false' \
        'gpu_evaluation_authorized=true' \
        "checkout_commit=$GIT_COMMIT" \
        "cpu_handoff_sha256=$HANDOFF_DIGEST" \
        "runner=$RUNNER" \
        "runner_sha256=$RUNNER_DIGEST" \
        "base_model=$BASE_MODEL" \
        "base_model_sha256=$BASE_DIGEST" \
        "r60_evidence=$R60_MARKER" \
        "r60_evidence_sha256=$R60_EVIDENCE_DIGEST" \
        "r60_checkpoint=$R60_CHECKPOINT" \
        "r60_checkpoint_sha256=$R60_DIGEST" \
        "data_manifest=$DATA_MANIFEST" \
        "data_manifest_sha256=$DATA_DIGEST" \
        >"$RECEIPT"
    RECEIPT_DIGEST="$(sha256sum "$RECEIPT" | cut -d' ' -f1)"
    receipt_relative="${RECEIPT#"$PROJECT_ROOT/"}"
    printf '%s  %s\n' "$RECEIPT_DIGEST" "$receipt_relative" >"$RECEIPT.sha256"
    printf '0\n' >"$receipt_dir/exit-code"
    printf 'success\n' >"$receipt_dir/terminal"
    : >"$receipt_dir/.success"
    printf 'terminal-published\n' >"$EVENT_LOG"
    printf '%s\n' \
        '#!/usr/bin/env bash' \
        "printf 'REAL_BACKEND_EXECUTED\\n' >>'$EVENT_LOG'" \
        'exit 99' >"$FAKE_SHUTDOWN"
    chmod 0700 "$FAKE_SHUTDOWN"
    RESULTS="$PROJECT_ROOT/runs/qwen-native-training/attempts/$(basename -- "$ATTEMPT")"
    RESULT_MARKER="$PROJECT_ROOT/manifests/qwen-native-training-ar-eval-only/$(basename -- "$ATTEMPT").ok"
}

add_evidence() {
    EVIDENCE_PATHS+=("${1#"$PROJECT_ROOT/"}")
}

create_eval_run() {
    local variant="$1" checkpoint="$2" rows="$3" run="$PROJECT_ROOT/runs/eval/$variant/attempts/run-$case_number"
    mkdir -p "$run/traces"
    printf '%s\n' \
        'job_mode=eval' \
        "variant=$variant" \
        'train_steps=0' \
        "input_model=$checkpoint" \
        "eval_expected_rows=$rows" \
        'eval_group_size=1' \
        'timed_out=false' \
        >"$run/run.env"
    printf 'resolved: %s\n' "$variant" >"$run/resolved-config.yaml"
    printf '{"variant":"%s"}\n' "$variant" >"$run/traces/eval_predictions.jsonl"
    printf '{"rows":%s}\n' "$rows" >"$run/traces/eval_predictions.manifest.json"
    printf '0\n' >"$run/exit-code"
    printf 'success\n' >"$run/terminal"
    : >"$run/.success"
    add_evidence "$run/run.env"
    add_evidence "$run/resolved-config.yaml"
    add_evidence "$run/traces/eval_predictions.jsonl"
    add_evidence "$run/traces/eval_predictions.manifest.json"
    add_evidence "$run/exit-code"
    add_evidence "$run/terminal"
    add_evidence "$run/.success"
    printf '%s\n' "$run"
}

seal_success_results() {
    local stage role variant rows checkpoint checkpoint_digest predecessor run
    local config_digest trace_digest trace_manifest_digest evidence_digest file relative
    local -a stages=(A-VAL-EVAL R-VAL-EVAL A-NQ-TEST-EVAL R-NQ-TEST-EVAL A-MULTIHOP-EVAL R-MULTIHOP-EVAL)
    local -a roles=(parent_val_eval reproduced_val_eval parent_nq_test_eval reproduced_nq_test_eval parent_multihop_eval reproduced_multihop_eval)
    local -a variants=(qwen_native_a_val qwen_native_r_val qwen_native_a_nq_test qwen_native_r_nq_test qwen_native_a_multihop qwen_native_r_multihop)
    local -a row_counts=(128 128 128 128 256 256) runs=()
    EVIDENCE_PATHS=()
    mkdir -p "$RESULTS" "$(dirname -- "$RESULT_MARKER")"
    printf '%s\n' \
        'schema=qwen-native-training-ar-eval-only-v1' \
        'stage=ar_eval_only' \
        'stage_order=A-VAL-EVAL,R-VAL-EVAL,A-NQ-TEST-EVAL,R-NQ-TEST-EVAL,A-MULTIHOP-EVAL,R-MULTIHOP-EVAL' \
        'training_executed=false' \
        "checkout_commit=$GIT_COMMIT" \
        "cpu_handoff_sha256=$HANDOFF_DIGEST" \
        "base_model=$BASE_MODEL" \
        "base_model_sha256=$BASE_DIGEST" \
        "r60_checkpoint=$R60_CHECKPOINT" \
        "r60_checkpoint_sha256=$R60_DIGEST" \
        "cpu_receipt=$RECEIPT" \
        "cpu_receipt_sha256=$RECEIPT_DIGEST" \
        "runner_sha256=$RUNNER_DIGEST" \
        'eval_group_size=1' \
        'decoding=greedy' \
        'seed=42' \
        >"$RESULTS/contract.env"
    printf '%s\n' \
        $'stage\trole\trun_dir\tcheckpoint\tcheckpoint_digest\tparent_checkpoint\tparent_checkpoint_digest\tcheckout_commit\tcpu_handoff_digest\tdata_manifest_sha256\tresolved_config_sha256\ttrace_sha256\ttrace_manifest_sha256\trun_contract_sha256\tpredecessor_evidence_sha256' \
        >"$RESULTS/lineage.tsv"
    printf 'stage\trole\trun_dir\n' >"$RESULTS/run-index.tsv"
    for position in "${!stages[@]}"; do
        stage="${stages[$position]}"
        role="${roles[$position]}"
        variant="${variants[$position]}"
        rows="${row_counts[$position]}"
        if [[ "$stage" == A-* ]]; then
            checkpoint="$BASE_MODEL"
            checkpoint_digest="$BASE_DIGEST"
            predecessor="$RECEIPT_DIGEST"
        else
            checkpoint="$R60_CHECKPOINT"
            checkpoint_digest="$R60_DIGEST"
            predecessor="$R60_EVIDENCE_DIGEST"
        fi
        run="$(create_eval_run "$variant" "$checkpoint" "$rows")"
        runs+=("$run")
        add_evidence "$run/run.env"
        add_evidence "$run/resolved-config.yaml"
        add_evidence "$run/traces/eval_predictions.jsonl"
        add_evidence "$run/traces/eval_predictions.manifest.json"
        add_evidence "$run/exit-code"
        add_evidence "$run/terminal"
        add_evidence "$run/.success"
        config_digest="$(sha256sum "$run/resolved-config.yaml" | cut -d' ' -f1)"
        trace_digest="$(sha256sum "$run/traces/eval_predictions.jsonl" | cut -d' ' -f1)"
        trace_manifest_digest="$(sha256sum "$run/traces/eval_predictions.manifest.json" | cut -d' ' -f1)"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t-\t%s\n' \
            "$stage" "$role" "$run" "$checkpoint" "$checkpoint_digest" \
            "$checkpoint" "$checkpoint_digest" "$GIT_COMMIT" "$HANDOFF_DIGEST" \
            "$DATA_DIGEST" "$config_digest" "$trace_digest" "$trace_manifest_digest" \
            "$predecessor" >>"$RESULTS/lineage.tsv"
        printf '%s\t%s\t%s\n' "$stage" "$role" "$run" >>"$RESULTS/run-index.tsv"
    done
    for key in val nq_test multihop; do
        mkdir -p "$RESULTS/paired-ar-$key"
        for file in summary.json summary.md paired_results.csv correct_questions.csv \
            wrong_questions.csv search_transition.csv; do
            printf 'paired:%s:%s\n' "$key" "$file" >"$RESULTS/paired-ar-$key/$file"
            add_evidence "$RESULTS/paired-ar-$key/$file"
        done
    done
    add_evidence "$RESULTS/contract.env"
    add_evidence "$RESULTS/lineage.tsv"
    add_evidence "$RESULTS/run-index.tsv"
    add_evidence "$RUNNER"
    add_evidence "$RECEIPT"
    add_evidence "$RECEIPT.sha256"
    (
        cd "$PROJECT_ROOT"
        printf '%s\n' "${EVIDENCE_PATHS[@]}" | LC_ALL=C sort -u | while IFS= read -r relative; do
            sha256sum -- "$relative"
        done
    ) >"$RESULTS/evidence.sha256"
    evidence_digest="$(sha256sum "$RESULTS/evidence.sha256" | cut -d' ' -f1)"
    printf '%s\n' "$evidence_digest" >"$RESULT_MARKER"
    printf 'qwen-native-training-ar-eval-only-v1\n' >"$ATTEMPT/result-contract"
    printf '%s\n' "$RESULTS" >"$ATTEMPT/result-root"
    printf '%s\n' "$RESULT_MARKER" >"$ATTEMPT/evidence-marker"
    printf '%s\n' "$evidence_digest" >"$ATTEMPT/evidence-digest"
}

run_watchdog() {
    local backend_rc="$1" drift="${2:-}"
    SEARCH_R1_AR_WATCHDOG_TEST_MODE=1 \
    SEARCH_R1_AR_WATCHDOG_TEST_ROOT="$CASE_ROOT" \
    SEARCH_R1_AR_WATCHDOG_TEST_EVENT_LOG="$EVENT_LOG" \
    SEARCH_R1_AR_WATCHDOG_TEST_SHUTDOWN_BINARY="$FAKE_SHUTDOWN" \
    SEARCH_R1_AR_WATCHDOG_TEST_BACKEND_RC="$backend_rc" \
    SEARCH_R1_AR_WATCHDOG_TEST_DRIFT_AFTER_ARM="$drift" \
    SEARCH_R1_AR_WATCHDOG_TIMEOUT_SECONDS=1 \
    SEARCH_R1_AR_WATCHDOG_LOCK_WAIT_SECONDS="${LOCK_WAIT_SECONDS:-1}" \
    AUTODL_ROOT="$PROJECT_ROOT" AUTODL_PERSISTENT_ROOT="$CASE_ROOT" \
        bash "$WATCHDOG" --test-foreground "$ATTEMPT"
}

assert_no_real_backend() {
    ! grep -Fq REAL_BACKEND_EXECUTED "$EVENT_LOG"
}

assert_dispatched_order() {
    local -a events
    mapfile -t events <"$EVENT_LOG"
    [[ "${events[0]}" == terminal-published ]]
    [[ "${events[1]}" == state:shutdown-safe ]]
    [[ "${events[2]}" == state:shutdown-requested ]]
    [[ "${events[3]}" == "backend:$FAKE_SHUTDOWN argc=0" ]]
    [[ "${events[4]}" == state:shutdown-dispatched ]]
    assert_no_real_backend
}

# Complete success: exact result marker and every checksum are revalidated twice.
make_case success 0 success
seal_success_results
run_watchdog 0
[[ -f "$ATTEMPT/shutdown-safe" && -f "$ATTEMPT/shutdown-requested" &&
    -f "$ATTEMPT/shutdown-dispatched" ]]
grep -Fxq "results_digest=$(cat "$ATTEMPT/evidence-digest")" "$ATTEMPT/shutdown-safe"
assert_dispatched_order

# A real evaluation failure has no result pointer/marker, but its durable outer
# terminal is sufficient to stop paid compute while preserving the true status.
make_case real-failure 7 failed
run_watchdog 0
[[ -f "$ATTEMPT/shutdown-dispatched" && "$(cat "$ATTEMPT/exit-code")" == 7 ]]
grep -Fxq 'work_exit_code=7' "$ATTEMPT/shutdown-safe"
assert_dispatched_order

# A sealed result followed by a nonzero outer status is ambiguous, not a real
# failure.  This regression prevents repeating the historical controlled-203 gap.
make_case controlled-nonzero 0 success
seal_success_results
write_terminal 203 failed
run_watchdog 0
[[ -f "$ATTEMPT/shutdown-skipped" && ! -e "$ATTEMPT/shutdown-safe" ]]
grep -Fxq 'reason=authorization-revalidation-failed' "$ATTEMPT/shutdown-skipped"
assert_no_real_backend

# Incomplete outer terminal evidence always keeps the guest running.
make_case incomplete-terminal 0 success
rm -f -- "$ATTEMPT/.success"
run_watchdog 0
[[ -f "$ATTEMPT/shutdown-skipped" && ! -e "$ATTEMPT/shutdown-safe" ]]
grep -Fxq 'reason=terminal-evidence-incomplete' "$ATTEMPT/shutdown-skipped"
assert_no_real_backend

# The terminal sentinel must be unique and the final phase-log line.
make_case trailing-log 7 failed
printf 'late output\n' >>"$ATTEMPT/phase.log"
run_watchdog 0
[[ -f "$ATTEMPT/shutdown-skipped" && ! -e "$ATTEMPT/shutdown-safe" ]]
grep -Fxq 'reason=terminal-evidence-incomplete' "$ATTEMPT/shutdown-skipped"
assert_no_real_backend

# Rehashing is mandatory; a changed result cannot authorize shutdown.
make_case result-drift 0 success
seal_success_results
printf 'tampered\n' >>"$RESULTS/paired-ar-val/summary.json"
run_watchdog 0
[[ -f "$ATTEMPT/shutdown-skipped" && ! -e "$ATTEMPT/shutdown-safe" ]]
grep -Fxq 'reason=authorization-revalidation-failed' "$ATTEMPT/shutdown-skipped"
assert_no_real_backend

# Capability-bound runner identity is checked after arming as well as before it.
make_case runner-drift 7 failed
run_watchdog 0 runner
[[ -f "$ATTEMPT/shutdown-skipped" && ! -e "$ATTEMPT/shutdown-safe" ]]
grep -Fxq 'reason=authorization-revalidation-failed' "$ATTEMPT/shutdown-skipped"
assert_no_real_backend

# The outer attempt must remain bound to the exact evaluation-only entrypoint.
make_case entrypoint-drift 7 failed
run_watchdog 0 entrypoint
[[ -f "$ATTEMPT/shutdown-skipped" && ! -e "$ATTEMPT/shutdown-safe" ]]
grep -Fxq 'reason=authorization-revalidation-failed' "$ATTEMPT/shutdown-skipped"
assert_no_real_backend

# A short handoff race between terminal publication and lock release is tolerated.
make_case transient-lock 7 failed
LOCK_WAIT_SECONDS=4
ready="$CASE_ROOT/lock-ready"
flock "$PROJECT_ROOT/state/phase.lock" bash -c "printf ready >'$ready'; sleep 1" &
LOCK_HOLDER_PID=$!
for _ in {1..100}; do [[ -f "$ready" ]] && break; sleep 0.01; done
[[ -f "$ready" ]]
run_watchdog 0
wait "$LOCK_HOLDER_PID"
LOCK_HOLDER_PID=''
unset LOCK_WAIT_SECONDS
[[ -f "$ATTEMPT/shutdown-dispatched" ]]
assert_dispatched_order

# A phase that remains active beyond the bounded wait is never powered off.
make_case persistent-lock 7 failed
ready="$CASE_ROOT/lock-ready"
flock "$PROJECT_ROOT/state/phase.lock" bash -c "printf ready >'$ready'; sleep 10" &
LOCK_HOLDER_PID=$!
for _ in {1..100}; do [[ -f "$ready" ]] && break; sleep 0.01; done
[[ -f "$ready" ]]
run_watchdog 0
[[ -f "$ATTEMPT/shutdown-skipped" && ! -e "$ATTEMPT/shutdown-safe" ]]
grep -Fxq 'reason=phase-lock-unavailable' "$ATTEMPT/shutdown-skipped"
kill "$LOCK_HOLDER_PID" 2>/dev/null || true
wait "$LOCK_HOLDER_PID" 2>/dev/null || true
LOCK_HOLDER_PID=''
assert_no_real_backend

# Backend failure remains distinct from work status and never overwrites exit 0.
make_case backend-failure 0 success
seal_success_results
set +e
run_watchdog 9
watchdog_rc=$?
set -e
[[ "$watchdog_rc" == 1 && -f "$ATTEMPT/shutdown-safe" &&
    -f "$ATTEMPT/shutdown-requested" && -f "$ATTEMPT/shutdown-failed" ]]
[[ "$(cat "$ATTEMPT/exit-code")" == 0 ]]
grep -Fxq 'backend_exit_code=9' "$ATTEMPT/shutdown-failed"
assert_no_real_backend

# The production path is detached by the watchdog itself and dispatches the
# bound backend with an empty argument vector.
grep -Fq '/usr/bin/nohup /usr/bin/setsid' "$SOURCE_WATCHDOG"
grep -Fq '"${CAP[shutdown_binary]}"' "$SOURCE_WATCHDOG"
! grep -Eq '(^|[[:space:]])(poweroff|systemctl[[:space:]]+poweroff)' "$SOURCE_WATCHDOG"

printf 'Qwen native A/R evaluation-only watchdog tests passed.\n'
