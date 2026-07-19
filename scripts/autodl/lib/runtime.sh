#!/usr/bin/env bash

# Shared launcher state for the three AutoDL phases. This file is sourced.
if [[ -n "${SEARCH_R1_AUTODL_RUNTIME_LOADED:-}" ]]; then
    return 0
fi
readonly SEARCH_R1_AUTODL_RUNTIME_LOADED=1

PROJECT_ROOT="${AUTODL_ROOT:-/root/autodl-tmp/search-r1}"
STATE_ROOT="$PROJECT_ROOT/state"
ATTEMPTS_ROOT="$STATE_ROOT/attempts"
LATEST_ROOT="$STATE_ROOT/latest"
LOCK_FILE="$STATE_ROOT/phase.lock"
CHECKOUT_DIR="$PROJECT_ROOT/checkout"
MANIFEST_DIR="$PROJECT_ROOT/manifests"
LOG_DIR="$PROJECT_ROOT/logs"
RUNS_ROOT="$PROJECT_ROOT/runs"

utc_now() {
    date -u +'%Y-%m-%dT%H:%M:%SZ'
}

atomic_write() {
    local path="$1"
    local value="$2"
    local tmp
    mkdir -p "$(dirname "$path")"
    tmp="${path}.tmp.$$.$RANDOM"
    printf '%s' "$value" >"$tmp"
    mv -f -- "$tmp" "$path"
}

sync_path() {
    sync -f "$1" 2>/dev/null || sync
}

checkout_tree_digest() {
    (
        cd "$CHECKOUT_DIR"
        find . -path './.git' -prune -o -type f -print0 \
            | LC_ALL=C sort -z \
            | xargs -0 -r sha256sum \
            | sha256sum \
            | cut -d' ' -f1
    )
}

new_attempt() {
    local phase="$1"
    local stamp attempt
    stamp="$(date -u +'%Y%m%dT%H%M%SZ')-$$-$RANDOM"
    attempt="$ATTEMPTS_ROOT/$phase/$stamp"
    mkdir -p "$ATTEMPTS_ROOT/$phase" "$LATEST_ROOT" "$LOG_DIR" "$MANIFEST_DIR" "$RUNS_ROOT"
    (umask 077 && mkdir "$attempt")
    : >"$attempt/.starting"
    atomic_write "$attempt/requested-at" "$(utc_now)"$'\n'
    atomic_write "$LATEST_ROOT/$phase" "$attempt"$'\n'
    printf '%s\n' "$attempt"
}

publish_terminal() {
    local attempt="$1"
    local rc="$2"
    local state marker
    if ((rc == 0)); then
        state='success'
        marker='.success'
    else
        state='failed'
        marker='.failed'
    fi

    atomic_write "$attempt/exit-code" "$rc"$'\n'
    atomic_write "$attempt/finished-at" "$(utc_now)"$'\n'
    atomic_write "$attempt/terminal" "$state"$'\n'
    : >"$attempt/$marker"
    sync_path "$attempt"
    rm -f -- "$attempt/.starting" "$attempt/.running"
    sync_path "$attempt"
}

phase_worker() {
    local phase="$1"
    local attempt="$2"
    local entrypoint="$3"
    local child_pid='' rc

    case "$attempt" in
        "$ATTEMPTS_ROOT/$phase/"*) ;;
        *) printf 'Refusing attempt path outside %s\n' "$ATTEMPTS_ROOT/$phase" >&2; exit 64 ;;
    esac

    exec 9>"$LOCK_FILE"
    if ! flock -n 9; then
        printf 'Another AutoDL phase owns %s\n' "$LOCK_FILE" >&2
        publish_terminal "$attempt" 75
        exit 75
    fi

    rm -f -- "$attempt/.starting"
    : >"$attempt/.running"
    atomic_write "$attempt/worker-pid" "$$"$'\n'
    sync_path "$attempt"

    on_signal() {
        local signal_rc="$1"
        trap - INT TERM HUP
        if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
            kill -TERM -- "-$child_pid" 2>/dev/null || true
            wait "$child_pid" 2>/dev/null || true
        fi
        publish_terminal "$attempt" "$signal_rc"
        exit "$signal_rc"
    }
    trap 'on_signal 130' INT
    trap 'on_signal 143' TERM
    trap 'on_signal 129' HUP

    setsid bash "$entrypoint" --action "$attempt" &
    child_pid=$!
    atomic_write "$attempt/action-pid" "$child_pid"$'\n'
    set +e
    wait "$child_pid"
    rc=$?
    set -e
    trap - INT TERM HUP
    publish_terminal "$attempt" "$rc"
    exit "$rc"
}

phase_launch() {
    local phase="$1"
    local entrypoint="$2"
    local attempt worker_pid
    attempt="$(new_attempt "$phase")"
    atomic_write "$attempt/entrypoint" "$entrypoint"$'\n'

    nohup setsid bash "$entrypoint" --worker "$attempt" \
        >>"$attempt/phase.log" 2>&1 </dev/null &
    worker_pid=$!
    atomic_write "$attempt/launcher-pid" "$worker_pid"$'\n'

    # Wait only for admission; the phase itself remains detached.
    for _ in {1..50}; do
        [[ -e "$attempt/.running" || -e "$attempt/terminal" ]] && break
        sleep 0.1
    done

    printf 'Started %s phase: %s\n' "$phase" "$attempt"
    printf 'Log: %s/phase.log\n' "$attempt"
    printf 'Completion: exit-code=0 and .success in that directory\n'
}

expected_commit() {
    local marker="$MANIFEST_DIR/git.ok"
    [[ -f "$marker" ]] || { printf 'Missing phase-1 marker: %s\n' "$marker" >&2; return 1; }
    tr -d '\r\n' <"$marker"
}

verify_checkout() {
    local expected="$1"
    local actual status
    [[ -d "$CHECKOUT_DIR/.git" ]] || { printf 'Missing checkout: %s\n' "$CHECKOUT_DIR" >&2; return 1; }
    actual="$(git -C "$CHECKOUT_DIR" rev-parse HEAD)"
    [[ "$actual" == "$expected" ]] || {
        printf 'Checkout commit mismatch: expected %s, got %s\n' "$expected" "$actual" >&2
        return 1
    }
    status="$(git -C "$CHECKOUT_DIR" status --porcelain --untracked-files=all)"
    [[ -z "$status" ]] || {
        printf 'Checkout is dirty; refusing to use it:\n%s\n' "$status" >&2
        return 1
    }
    if [[ -f "$MANIFEST_DIR/checkout-tree.sha256" ]]; then
        local expected_tree actual_tree
        expected_tree="$(tr -d '\r\n' <"$MANIFEST_DIR/checkout-tree.sha256")"
        actual_tree="$(checkout_tree_digest)"
        [[ "$actual_tree" == "$expected_tree" ]] || {
            printf 'Checkout files changed since phase 1 (including ignored files).\n' >&2
            return 1
        }
    fi
}
