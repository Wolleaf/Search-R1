#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=lib/runtime.sh
source "$SCRIPT_DIR/lib/runtime.sh"

validate_inputs() {
    : "${REPO_URL:?Set REPO_URL to the public Git repository URL}"
    : "${COMMIT_SHA:?Set COMMIT_SHA to the trusted 40-character commit}"
    [[ "$COMMIT_SHA" =~ ^[0-9a-fA-F]{40}$ ]] || {
        printf 'COMMIT_SHA must be a full 40-character hexadecimal commit.\n' >&2
        return 64
    }
    if [[ "$REPO_URL" =~ ^https?://[^/]+@ ]]; then
        printf 'Do not put credentials in REPO_URL; use a temporary Git credential helper.\n' >&2
        return 64
    fi
}

git_action() {
    local attempt="$1"
    local expected clone_tmp actual_remote
    validate_inputs
    expected="${COMMIT_SHA,,}"
    mkdir -p "$PROJECT_ROOT" "$LOG_DIR" "$MANIFEST_DIR"

    if [[ -e "$CHECKOUT_DIR" ]]; then
        [[ -d "$CHECKOUT_DIR/.git" ]] || {
            printf 'Existing checkout path is not a Git repository: %s\n' "$CHECKOUT_DIR" >&2
            return 1
        }
        [[ -f "$MANIFEST_DIR/checkout-tree.sha256" ]] || {
            printf 'Existing checkout has no phase-1 tree seal; preserve and inspect it manually.\n' >&2
            return 1
        }
        actual_remote="$(git -C "$CHECKOUT_DIR" remote get-url origin)"
        [[ "$actual_remote" == "$REPO_URL" ]] || {
            printf 'Existing origin mismatch: %s\n' "$actual_remote" >&2
            return 1
        }
        verify_checkout "$expected"
    else
        clone_tmp="$PROJECT_ROOT/checkout.clone.$(basename "$attempt")"
        [[ ! -e "$clone_tmp" ]] || {
            printf 'Clone staging path already exists; preserve and inspect it: %s\n' "$clone_tmp" >&2
            return 1
        }
        git clone --no-checkout -- "$REPO_URL" "$clone_tmp"
        git -C "$clone_tmp" fetch --no-tags origin "$expected"
        git -C "$clone_tmp" checkout --detach "$expected"
        [[ "$(git -C "$clone_tmp" rev-parse HEAD)" == "$expected" ]]
        [[ -z "$(git -C "$clone_tmp" status --porcelain --untracked-files=all)" ]]
        mv -- "$clone_tmp" "$CHECKOUT_DIR"
        verify_checkout "$expected"
    fi

    {
        printf 'requested_commit=%s\n' "$expected"
        git -C "$CHECKOUT_DIR" remote -v
        git -C "$CHECKOUT_DIR" rev-parse HEAD
        git -C "$CHECKOUT_DIR" status --short
    } | tee "$LOG_DIR/git.log"

    atomic_write "$MANIFEST_DIR/checkout-tree.sha256" "$(checkout_tree_digest)"$'\n'
    atomic_write "$MANIFEST_DIR/git.ok" "$expected"$'\n'
    atomic_write "$MANIFEST_DIR/repository.tsv" $'repository_url\t'"$REPO_URL"$'\ncommit\t'"$expected"$'\n'
    sync_path "$MANIFEST_DIR"
}

case "${1:-}" in
    --worker)
        phase_worker git "${2:?missing attempt directory}" "$0"
        ;;
    --action)
        git_action "${2:?missing attempt directory}"
        ;;
    '')
        validate_inputs
        phase_launch git "$0"
        ;;
    *)
        printf 'Usage: REPO_URL=... COMMIT_SHA=<40-char-sha> bash %s\n' "$0" >&2
        exit 64
        ;;
esac
