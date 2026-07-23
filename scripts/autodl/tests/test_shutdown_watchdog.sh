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
