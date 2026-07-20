#!/usr/bin/env bash
set -Eeuo pipefail

AUTODL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
TEST_ROOT="$(mktemp -d)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
trap 'rm -rf -- "$TEST_ROOT"' EXIT

wait_for_terminal() {
    local attempt="$1"
    for _ in {1..100}; do
        if [[ -f "$attempt/exit-code" && -f "$attempt/terminal" &&
              ! -e "$attempt/.starting" && ! -e "$attempt/.running" ]] &&
            { [[ -f "$attempt/.success" ]] || [[ -f "$attempt/.failed" ]]; }; then
            return 0
        fi
        sleep 0.05
    done
    printf 'Timed out waiting for %s\n' "$attempt" >&2
    return 1
}

FAKE="$TEST_ROOT/fake_phase.sh"
printf '%s\n' \
    '#!/usr/bin/env bash' \
    'set -Eeuo pipefail' \
    "source '$AUTODL_DIR/lib/runtime.sh'" \
    'if [[ -n "${SYNC_EVENT_LOG:-}" ]]; then' \
    '  sync_path() { printf "%s\n" "$1" >>"$SYNC_EVENT_LOG"; }' \
    'fi' \
    'case "${1:-}" in' \
    '  --worker) phase_worker test "$2" "$0" ;;' \
    '  --action) exit "${ACTION_RC:-0}" ;;' \
    '  "") phase_launch test "$0" ;;' \
    'esac' >"$FAKE"

AUTODL_ROOT="$TEST_ROOT/project" ACTION_RC=0 bash "$FAKE"
attempt="$(tr -d '\r\n' <"$TEST_ROOT/project/state/latest/test")"
wait_for_terminal "$attempt"
[[ "$(cat "$attempt/exit-code")" == 0 && -f "$attempt/.success" && ! -f "$attempt/.running" ]]
[[ "$(grep -Fxc 'AUTODL_PHASE_TERMINAL state=success exit_code=0' "$attempt/phase.log")" == 1 ]]

AUTODL_ROOT="$TEST_ROOT/project" ACTION_RC=7 bash "$FAKE"
attempt="$(tr -d '\r\n' <"$TEST_ROOT/project/state/latest/test")"
wait_for_terminal "$attempt"
[[ "$(cat "$attempt/exit-code")" == 7 && -f "$attempt/.failed" ]]
[[ "$(grep -Fxc 'AUTODL_PHASE_TERMINAL state=failed exit_code=7' "$attempt/phase.log")" == 1 ]]

exec 8>"$TEST_ROOT/project/state/phase.lock"
flock -n 8
AUTODL_ROOT="$TEST_ROOT/project" ACTION_RC=0 bash "$FAKE"
attempt="$(tr -d '\r\n' <"$TEST_ROOT/project/state/latest/test")"
wait_for_terminal "$attempt"
[[ "$(cat "$attempt/exit-code")" == 75 && -f "$attempt/.failed" ]]
[[ "$(grep -Fxc 'AUTODL_PHASE_TERMINAL state=failed exit_code=75' "$attempt/phase.log")" == 1 ]]
exec 8>&-

sync_events="$TEST_ROOT/sync-events.log"
AUTODL_ROOT="$TEST_ROOT/project" ACTION_RC=0 SYNC_EVENT_LOG="$sync_events" bash "$FAKE"
attempt="$(tr -d '\r\n' <"$TEST_ROOT/project/state/latest/test")"
wait_for_terminal "$attempt"
mapfile -t terminal_syncs < <(tail -n 3 "$sync_events")
[[ "${terminal_syncs[0]}" == "$attempt/phase.log" ]]
[[ "${terminal_syncs[1]}" == "$attempt" && "${terminal_syncs[2]}" == "$attempt" ]]

source_repo="$TEST_ROOT/source-repo"
mkdir "$source_repo"
git -C "$source_repo" init -q
git -C "$source_repo" config user.name 'AutoDL Test'
git -C "$source_repo" config user.email 'autodl-test@example.invalid'
printf 'pinned\n' >"$source_repo/README.md"
git -C "$source_repo" add README.md
git -C "$source_repo" commit -q -m initial
source_commit="$(git -C "$source_repo" rev-parse HEAD)"
AUTODL_ROOT="$TEST_ROOT/git-project" \
    REPO_URL="$source_repo" \
    COMMIT_SHA="$source_commit" \
    bash "$AUTODL_DIR/01_git.sh"
attempt="$(tr -d '\r\n' <"$TEST_ROOT/git-project/state/latest/git")"
wait_for_terminal "$attempt"
[[ "$(cat "$attempt/exit-code")" == 0 && -f "$attempt/.success" ]]
[[ "$(git -C "$TEST_ROOT/git-project/checkout" rev-parse HEAD)" == "$source_commit" ]]
[[ -s "$TEST_ROOT/git-project/manifests/checkout-tree.sha256" ]]

mkdir -p "$TEST_ROOT/handoff/model" "$TEST_ROOT/handoff/bm25" \
    "$TEST_ROOT/handoff/corpus" "$TEST_ROOT/handoff/data" \
    "$TEST_ROOT/handoff/manifests"
printf 'model\n' >"$TEST_ROOT/handoff/model/config.json"
printf 'index\n' >"$TEST_ROOT/handoff/bm25/segments_1"
printf 'corpus\n' >"$TEST_ROOT/handoff/corpus/wiki-18.jsonl"
printf 'source\n' >"$TEST_ROOT/handoff/corpus-source.gz"
printf 'data\n' >"$TEST_ROOT/handoff/data/train.parquet"
printf 'lock\n' >"$TEST_ROOT/handoff/requirements.lock"
"$PYTHON_BIN" "$AUTODL_DIR/handoff.py" create \
    --root "$TEST_ROOT/handoff" \
    --commit aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
    --model "$TEST_ROOT/handoff/model" \
    --bm25 "$TEST_ROOT/handoff/bm25" \
    --corpus "$TEST_ROOT/handoff/corpus" \
    --data "$TEST_ROOT/handoff/data" \
    --requirements "$TEST_ROOT/handoff/requirements.lock" \
    --extra-file "$TEST_ROOT/handoff/corpus-source.gz" \
    --python-version 3.12.0 \
    --torch-version 2.8.0+cu128 \
    --output "$TEST_ROOT/handoff/manifests/cpu_handoff.json"
"$PYTHON_BIN" "$AUTODL_DIR/handoff.py" verify \
    --root "$TEST_ROOT/handoff" \
    --commit aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
    --python-version 3.12.0 \
    --torch-version 2.8.0+cu128 \
    --manifest "$TEST_ROOT/handoff/manifests/cpu_handoff.json"
printf 'tampered\n' >"$TEST_ROOT/handoff/model/config.json"
if "$PYTHON_BIN" "$AUTODL_DIR/handoff.py" verify \
    --root "$TEST_ROOT/handoff" \
    --commit aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
    --python-version 3.12.0 \
    --torch-version 2.8.0+cu128 \
    --manifest "$TEST_ROOT/handoff/manifests/cpu_handoff.json" >/dev/null 2>&1; then
    printf 'Tampered handoff was accepted.\n' >&2
    exit 1
fi

printf 'AutoDL runtime tests passed.\n'
