#!/usr/bin/env bash
set -Eeuo pipefail

# Authorization and launch tools must not come from an activated training environment.
PATH='/usr/sbin:/usr/bin:/sbin:/bin'
readonly PATH
export PATH

readonly TERMINAL_SENTINEL_PREFIX='AUTODL_PHASE_TERMINAL'
readonly DEFAULT_PROJECT_ROOT='/root/autodl-tmp/search-r1'
readonly DEFAULT_PERSISTENT_ROOT='/root/autodl-tmp'
readonly PRODUCTION_SHUTDOWN_BINARY='/usr/bin/shutdown'
readonly CAPABILITY_NAME='shutdown-capability.tsv'
readonly WATCHDOG_LOG_NAME='shutdown-watchdog.log'

SCRIPT_PATH="$(readlink -f -- "${BASH_SOURCE[0]}")"

utc_now() {
    date -u +'%Y-%m-%dT%H:%M:%SZ'
}

die() {
    printf '%s\n' "$*" >&2
    exit 64
}

validate_test_mode_setting() {
    [[ -z "${SEARCH_R1_AUTODL_TEST_MODE:-}" || "${SEARCH_R1_AUTODL_TEST_MODE}" == 1 ]]
}

is_regular_file() {
    [[ -f "$1" && ! -L "$1" ]]
}

is_contained() {
    local parent="$1" child="$2"
    if [[ "$parent" == / ]]; then
        [[ "$child" == /* && "$child" != / ]]
    else
        [[ "$child" == "$parent/"* ]]
    fi
}

file_sha256() {
    is_regular_file "$1" || return 1
    sha256sum -- "$1" | cut -d' ' -f1
}

tree_sha256() {
    local root="$1"
    [[ -d "$root" && ! -L "$root" ]] || return 1
    (
        cd "$root"
        find . -path './.git' -prune -o -type f -print0 \
            | LC_ALL=C sort -z \
            | xargs -0 -r sha256sum \
            | sha256sum \
            | cut -d' ' -f1
    )
}

sync_required() {
    sync -f "$1"
}

atomic_publish() {
    local path="$1" value="$2" tmp
    [[ ! -e "$path" && ! -L "$path" ]] || {
        printf 'Refusing to overwrite state: %s\n' "$path" >&2
        return 1
    }
    tmp="${path}.tmp.$$.$RANDOM"
    (umask 077 && printf '%s' "$value" >"$tmp")
    chmod 0600 "$tmp"
    mv -T -- "$tmp" "$path"
    sync_required "$path"
    sync_required "$(dirname -- "$path")"
}

canonical_existing() {
    local path="$1" canonical
    canonical="$(readlink -f -- "$path")" || return 1
    [[ -n "$canonical" ]] || return 1
    printf '%s\n' "$canonical"
}

validate_owner_and_mode() {
    local path="$1" expected_uid="$2" expected_mode="$3"
    is_regular_file "$path" || return 1
    [[ "$(stat -c '%u' -- "$path")" == "$expected_uid" ]] || return 1
    [[ "$(stat -c '%a' -- "$path")" == "$expected_mode" ]] || return 1
}

validate_protected_regular() {
    local path="$1" expected_uid="$2" mode
    is_regular_file "$path" || return 1
    [[ "$(stat -c '%u' -- "$path")" == "$expected_uid" ]] || return 1
    mode="$(stat -c '%a' -- "$path")"
    (( (8#$mode & 0022) == 0 ))
}

validate_protected_executable() {
    validate_protected_regular "$1" "$2" && [[ -x "$1" ]]
}

validate_mount() {
    local persistent_root="$1" mount_line root_line target fs source majmin root_majmin
    command -v findmnt >/dev/null 2>&1 || return 1
    mount_line="$(findmnt -rn -T "$persistent_root" -o TARGET,FSTYPE,SOURCE,MAJ:MIN)" || return 1
    root_line="$(findmnt -rn -T / -o TARGET,FSTYPE,SOURCE,MAJ:MIN)" || return 1
    read -r target fs source majmin <<<"$mount_line"
    read -r _ _ _ root_majmin <<<"$root_line"
    [[ "$target" == "$persistent_root" && -n "$source" && "$majmin" != "$root_majmin" ]] || return 1
    [[ ! "$fs" =~ ^(overlay|tmpfs|ramfs|squashfs)$ ]] || return 1
}

validate_host_and_roots() {
    local mode="$1" persistent_root="$2" project_root="$3" test_root="$4"
    [[ "$(uname -s)" == Linux ]] || return 1
    is_contained "$persistent_root" "$project_root" || return 1
    if [[ "$mode" == production ]]; then
        [[ "$(id -u)" == 0 ]] || return 1
        ! grep -Eqi '(microsoft|wsl)' /proc/sys/kernel/osrelease 2>/dev/null || return 1
        [[ "$persistent_root" == "$DEFAULT_PERSISTENT_ROOT" ]] || return 1
        validate_mount "$persistent_root"
    else
        [[ -n "$test_root" && "$test_root" != "$DEFAULT_PERSISTENT_ROOT" ]] || return 1
        is_contained "$test_root" "$persistent_root" || [[ "$test_root" == "$persistent_root" ]] || return 1
        ! is_contained "$test_root" "$DEFAULT_PERSISTENT_ROOT" || return 1
        ! is_contained "$DEFAULT_PERSISTENT_ROOT" "$test_root" || return 1
    fi
}

validate_attempt_path() {
    local project_root="$1" attempt="$2" attempts_root parent
    attempts_root="$(canonical_existing "$project_root/state/attempts/gpu")" || return 1
    parent="$(canonical_existing "$(dirname -- "$attempt")")" || return 1
    [[ "$parent" == "$attempts_root" && -d "$attempt" && ! -L "$attempt" ]] || return 1
    [[ "$(basename -- "$attempt")" =~ ^[0-9]{8}T[0-9]{6}Z-[0-9]+-[0-9]+$ ]] || return 1
}

validate_checkout() {
    local project_root="$1" expected_commit="$2" expected_tree="$3"
    local checkout="$project_root/checkout" manifest_dir="$project_root/manifests"
    local actual_commit actual_tree status recorded_commit recorded_tree
    [[ -d "$checkout/.git" && ! -L "$checkout" ]] || return 1
    is_regular_file "$manifest_dir/git.ok" || return 1
    is_regular_file "$manifest_dir/checkout-tree.sha256" || return 1
    recorded_commit="$(tr -d '\r\n' <"$manifest_dir/git.ok")"
    recorded_tree="$(tr -d '\r\n' <"$manifest_dir/checkout-tree.sha256")"
    [[ "$recorded_commit" == "$expected_commit" && "$recorded_tree" == "$expected_tree" ]] || return 1
    actual_commit="$(git -C "$checkout" rev-parse --verify HEAD^{commit})" || return 1
    [[ "$actual_commit" == "$expected_commit" ]] || return 1
    if git -C "$checkout" symbolic-ref -q HEAD >/dev/null 2>&1; then
        return 1
    fi
    status="$(git -C "$checkout" status --porcelain --untracked-files=all)" || return 1
    [[ -z "$status" ]] || return 1
    actual_tree="$(tree_sha256 "$checkout")" || return 1
    [[ "$actual_tree" == "$expected_tree" ]]
}

declare -A CAP=()
CAPABILITY_PATH=''
CAPABILITY_SHA256=''
WORK_RC=''
WORK_STATE=''
RESULTS_DIGEST='not-required'
LOCK_FD=''

readonly -a CAPABILITY_KEYS=(
    schema_version mode authorized capability_path project_root persistent_root
    attempt commit checkout_tree_sha256 lock_file latest_gpu phase_log shutdown_binary
    shutdown_binary_sha256 watchdog_script watchdog_script_sha256 dry_run
    foreground launch_nonce wait_timeout_seconds test_root test_event_log test_backend_rc
)

load_capability() {
    local path="$1" key value extra expected_uid
    CAP=()
    CAPABILITY_PATH="$(canonical_existing "$path")" || return 1
    while IFS=$'\t' read -r key value extra; do
        [[ -n "$key" && -n "$value" && -z "${extra:-}" ]] || return 1
        [[ "$key" =~ ^[a-z0-9_]+$ && ! ${CAP[$key]+present} ]] || return 1
        CAP["$key"]="$value"
    done <"$CAPABILITY_PATH"
    for key in "${CAPABILITY_KEYS[@]}"; do
        [[ ${CAP[$key]+present} ]] || return 1
    done
    [[ "${#CAP[@]}" == "${#CAPABILITY_KEYS[@]}" ]] || return 1
    [[ "${CAP[schema_version]}" == 1 && "${CAP[authorized]}" == yes ]] || return 1
    [[ "${CAP[capability_path]}" == "$CAPABILITY_PATH" ]] || return 1
    [[ "${CAP[commit]}" =~ ^[0-9a-f]{40}$ && "${CAP[checkout_tree_sha256]}" =~ ^[0-9a-f]{64}$ ]] || return 1
    [[ "${CAP[shutdown_binary_sha256]}" =~ ^[0-9a-f]{64}$ && "${CAP[watchdog_script_sha256]}" =~ ^[0-9a-f]{64}$ ]] || return 1
    [[ "${CAP[launch_nonce]}" =~ ^[0-9a-f]{64}$ ]] || return 1
    [[ "${CAP[dry_run]}" == true || "${CAP[dry_run]}" == false ]] || return 1
    [[ "${CAP[foreground]}" == true || "${CAP[foreground]}" == false ]] || return 1
    [[ "${CAP[wait_timeout_seconds]}" =~ ^[1-9][0-9]*$ && "${CAP[wait_timeout_seconds]}" -le 604800 ]] || return 1
    if [[ "${CAP[mode]}" == production ]]; then
        expected_uid=0
        [[ "${CAP[foreground]}" == false ]] || return 1
    elif [[ "${CAP[mode]}" == test ]]; then
        expected_uid="$(id -u)"
    else
        return 1
    fi
    validate_owner_and_mode "$CAPABILITY_PATH" "$expected_uid" 600 || return 1
    CAPABILITY_SHA256="$(file_sha256 "$CAPABILITY_PATH")" || return 1
}

revalidate_capability() {
    local expected_uid
    if [[ "${CAP[mode]}" == production ]]; then expected_uid=0; else expected_uid="$(id -u)"; fi
    validate_owner_and_mode "$CAPABILITY_PATH" "$expected_uid" 600 || return 1
    [[ "$(file_sha256 "$CAPABILITY_PATH")" == "$CAPABILITY_SHA256" ]]
}

validate_test_contract() {
    local test_root event_log shutdown_binary
    if [[ "${CAP[mode]}" == production ]]; then
        [[ -z "${SEARCH_R1_AUTODL_TEST_MODE:-}" ]] || return 1
        [[ "${CAP[test_root]}" == - && "${CAP[test_event_log]}" == - && "${CAP[test_backend_rc]}" == - ]] || return 1
        return 0
    fi
    [[ "${SEARCH_R1_AUTODL_TEST_MODE:-}" == 1 ]] || return 1
    test_root="$(canonical_existing "${SEARCH_R1_AUTODL_TEST_ROOT:?}")" || return 1
    event_log="$(canonical_existing "${SEARCH_R1_AUTODL_TEST_SHUTDOWN_LOG:?}")" || return 1
    shutdown_binary="$(canonical_existing "${SEARCH_R1_AUTODL_TEST_SHUTDOWN_BINARY:?}")" || return 1
    [[ "$test_root" == "${CAP[test_root]}" && "$event_log" == "${CAP[test_event_log]}" ]] || return 1
    [[ "$shutdown_binary" == "${CAP[shutdown_binary]}" ]] || return 1
    is_contained "$test_root" "$event_log" && is_contained "$test_root" "$shutdown_binary" || return 1
    is_regular_file "$event_log" || return 1
    [[ "${SEARCH_R1_AUTODL_TEST_BACKEND_RC:-0}" == "${CAP[test_backend_rc]}" ]]
}

validate_worker_admission() {
    local attempt="${CAP[attempt]}" pid_file admitted_file expected_uid pid nonce
    pid_file="$attempt/shutdown-watchdog-pid"
    admitted_file="$attempt/shutdown-watchdog-admitted"
    if [[ "${CAP[mode]}" == production ]]; then expected_uid=0; else expected_uid="$(id -u)"; fi
    validate_owner_and_mode "$pid_file" "$expected_uid" 600 || return 1
    validate_owner_and_mode "$admitted_file" "$expected_uid" 600 || return 1
    (( $(wc -l <"$pid_file") == 1 && $(wc -l <"$admitted_file") == 1 )) || return 1
    pid="$(tr -d '\r\n' <"$pid_file")"
    nonce="$(tr -d '\r\n' <"$admitted_file")"
    [[ "$pid" == "$$" && "$nonce" == "${CAP[launch_nonce]}" ]]
}

wait_for_worker_admission() {
    local deadline=$(( $(date +%s) + 30 ))
    while (( $(date +%s) <= deadline )); do
        if [[ -e "${CAP[attempt]}/shutdown-watchdog-pid" ||
              -e "${CAP[attempt]}/shutdown-watchdog-admitted" ]]; then
            if [[ -e "${CAP[attempt]}/shutdown-watchdog-pid" &&
                  -e "${CAP[attempt]}/shutdown-watchdog-admitted" ]]; then
                validate_worker_admission
                return
            fi
        fi
        sleep 0.1
    done
    return 1
}

append_test_event() {
    local event="$1"
    [[ "${CAP[mode]}" == test ]] || return 0
    validate_test_contract || return 1
    printf '%s\n' "$event" >>"${CAP[test_event_log]}"
    sync_required "${CAP[test_event_log]}"
}

publish_state() {
    local name="$1" value="$2" path
    [[ "$name" =~ ^shutdown-(safe|requested|dispatched|failed|skipped)$ ]] || return 1
    validate_attempt_path "${CAP[project_root]}" "${CAP[attempt]}" || return 1
    path="${CAP[attempt]}/$name"
    atomic_publish "$path" "$value"
    append_test_event "state:$name"
}

publish_skipped() {
    local reason="$1"
    printf 'Shutdown skipped: %s\n' "$reason" >&2
    if [[ ! -e "${CAP[attempt]}/shutdown-skipped" && ! -L "${CAP[attempt]}/shutdown-skipped" ]]; then
        publish_state shutdown-skipped \
            "at=$(utc_now)"$'\n'"reason=$reason"$'\n' || true
    fi
}

validate_terminal_evidence() {
    local attempt="${CAP[attempt]}" exit_file terminal_file success failed line_count marker_count path
    local expected_uid
    exit_file="$attempt/exit-code"
    terminal_file="$attempt/terminal"
    success="$attempt/.success"
    failed="$attempt/.failed"
    is_regular_file "${CAP[phase_log]}" || return 1
    if [[ "${CAP[mode]}" == production ]]; then expected_uid=0; else expected_uid="$(id -u)"; fi
    for path in "$exit_file" "$terminal_file"; do
        validate_protected_regular "$path" "$expected_uid" || return 1
        (( $(wc -l <"$path") == 1 )) || return 1
        (( $(stat -c '%s' -- "$path") <= 64 )) || return 1
    done
    [[ ! -e "$attempt/.starting" && ! -L "$attempt/.starting" ]] || return 1
    [[ ! -e "$attempt/.running" && ! -L "$attempt/.running" ]] || return 1
    marker_count=0
    if validate_protected_regular "$success" "$expected_uid"; then ((marker_count += 1)); fi
    if validate_protected_regular "$failed" "$expected_uid"; then ((marker_count += 1)); fi
    [[ "$marker_count" == 1 ]] || return 1
    WORK_RC="$(tr -d '\r\n' <"$exit_file")"
    WORK_STATE="$(tr -d '\r\n' <"$terminal_file")"
    [[ "$WORK_RC" =~ ^(0|[1-9][0-9]*)$ && "$WORK_RC" -le 255 ]] || return 1
    if [[ "$WORK_RC" == 0 ]]; then
        [[ "$WORK_STATE" == success && -f "$success" && ! -L "$success" && ! -e "$failed" && ! -L "$failed" ]] || return 1
    else
        [[ "$WORK_STATE" == failed && -f "$failed" && ! -L "$failed" && ! -e "$success" && ! -L "$success" ]] || return 1
    fi
    line_count="$(grep -Fxc -- "$TERMINAL_SENTINEL_PREFIX state=$WORK_STATE exit_code=$WORK_RC" "${CAP[phase_log]}" || true)"
    marker_count="$(grep -Ec '^AUTODL_PHASE_TERMINAL state=(success|failed) exit_code=(0|[1-9][0-9]*)$' "${CAP[phase_log]}" || true)"
    [[ "$line_count" == 1 && "$marker_count" == 1 ]]
}

validate_legacy_success_artifacts() {
    local project="${CAP[project_root]}" results expected_results gpu_ok attempt_digest_file
    local expected_digest recorded_digest attempt_digest expected_uid index file actual_line
    local -a files=(results.csv results.md lineage.tsv) checksum_lines
    RESULTS_DIGEST='not-required'
    [[ "$WORK_STATE" == success ]] || return 0
    if [[ "${CAP[mode]}" == production ]]; then expected_uid=0; else expected_uid="$(id -u)"; fi
    expected_results="$project/runs/comparison"
    results="$(canonical_existing "$expected_results")" || return 1
    [[ "$results" == "$expected_results" && -d "$results" && ! -L "$results" ]] || return 1
    for file in "${files[@]}" comparison.sha256; do
        validate_protected_regular "$results/$file" "$expected_uid" && [[ -s "$results/$file" ]] || return 1
    done
    [[ "$(wc -l <"$results/results.csv")" == 5 ]] || return 1
    [[ "$(wc -l <"$results/lineage.tsv")" == 5 ]] || return 1
    mapfile -t checksum_lines <"$results/comparison.sha256"
    [[ "${#checksum_lines[@]}" == "${#files[@]}" ]] || return 1
    for index in "${!files[@]}"; do
        file="${files[$index]}"
        expected_digest="$(file_sha256 "$results/$file")" || return 1
        actual_line="${checksum_lines[$index]}"
        [[ "$actual_line" == "$expected_digest  $file" ]] || return 1
    done
    RESULTS_DIGEST="$(file_sha256 "$results/comparison.sha256")" || return 1
    gpu_ok="$project/manifests/gpu.ok"
    validate_protected_regular "$gpu_ok" "$expected_uid" || return 1
    (( $(wc -l <"$gpu_ok") == 1 )) || return 1
    (( $(stat -c '%s' -- "$gpu_ok") <= 128 )) || return 1
    recorded_digest="$(tr -d '\r\n' <"$gpu_ok")"
    [[ "$recorded_digest" == "$RESULTS_DIGEST" ]] || return 1
    attempt_digest_file="${CAP[attempt]}/comparison-digest"
    validate_protected_regular "$attempt_digest_file" "$expected_uid" || return 1
    (( $(wc -l <"$attempt_digest_file") == 1 )) || return 1
    (( $(stat -c '%s' -- "$attempt_digest_file") <= 128 )) || return 1
    attempt_digest="$(tr -d '\r\n' <"$attempt_digest_file")"
    [[ "$attempt_digest" == "$RESULTS_DIGEST" ]] || return 1
    for file in "${files[@]}" comparison.sha256; do sync_required "$results/$file" || return 1; done
    sync_required "$results" || return 1
    sync_required "$gpu_ok" || return 1
    sync_required "$attempt_digest_file" || return 1
    sync_required "$(dirname -- "$gpu_ok")" || return 1
    sync_required "${CAP[attempt]}"
}

validate_qwen_native_training_evidence() {
    local contract="$1" project="$2" results="$3" attempt_name="$4" expected_uid="$5"
    command -v python3 >/dev/null 2>&1 || return 1
    python3 - "$contract" "$project" "$results" "$attempt_name" "$expected_uid" <<'PY'
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys

contract, project_raw, results_raw, attempt_name, expected_uid_raw = sys.argv[1:]
project = Path(project_raw).resolve(strict=True)
results = Path(results_raw).resolve(strict=True)
expected_uid = int(expected_uid_raw)
digest_re = re.compile(r"[0-9a-f]{64}")
attempt_re = re.compile(r"[0-9]{8}T[0-9]{6}Z-[0-9]+-[0-9]+")
manifest_re = re.compile(r"([0-9a-f]{64})  ([A-Za-z0-9._/-]+)")
smoke_check_names = frozenset({
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
})
smoke_metric_names = frozenset({
    "groups",
    "mixed_groups",
    "nonzero_trace_advantages",
    "strict_em_positive_count",
    "trajectories",
    "wandb_files",
})


def fail(message):
    raise SystemExit(message)


def protected_file(path):
    path = Path(path)
    try:
        info = path.lstat()
    except OSError as exc:
        fail(f"missing protected evidence: {path}: {exc}")
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        fail(f"evidence is not a regular non-symlink file: {path}")
    if info.st_uid != expected_uid or info.st_mode & 0o022:
        fail(f"unsafe evidence ownership or mode: {path}")
    return path


def sha256(path):
    hasher = hashlib.sha256()
    with protected_file(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_checksum_manifest(path, verify_files=True):
    entries = {}
    previous = None
    for line in protected_file(path).read_text(encoding="utf-8").splitlines():
        match = manifest_re.fullmatch(line)
        if match is None:
            fail(f"malformed checksum entry: {path}")
        digest, relative = match.groups()
        if relative.startswith("/") or "//" in relative or ".." in Path(relative).parts:
            fail(f"unsafe checksum path: {relative}")
        if previous is not None and relative <= previous:
            fail(f"checksum entries are not strictly sorted: {path}")
        previous = relative
        if relative in entries:
            fail(f"duplicate checksum path: {relative}")
        candidate = project / relative
        try:
            canonical = candidate.resolve(strict=True)
            canonical.relative_to(project)
        except (OSError, ValueError):
            fail(f"checksum path escapes project: {relative}")
        if canonical != candidate:
            fail(f"checksum path is not canonical: {relative}")
        protected_file(candidate)
        if verify_files and sha256(candidate) != digest:
            fail(f"checksum mismatch: {relative}")
        entries[relative] = digest
    if not entries:
        fail(f"empty checksum manifest: {path}")
    return entries


# The shell caller has just hash-checked every top-level entry. Reparse its
# structure here without paying for a second full trace read.
entries = parse_checksum_manifest(results / "evidence.sha256", verify_files=False)
if results.name != attempt_name:
    fail("native training result directory does not match the outer attempt")


def relative(path):
    path = Path(path)
    try:
        canonical = path.resolve(strict=True)
        value = canonical.relative_to(project).as_posix()
    except (OSError, ValueError):
        fail(f"evidence path escapes project: {path}")
    if canonical != path:
        fail(f"evidence path is not canonical: {path}")
    return value


def require_evidence(path):
    item = relative(path)
    if item not in entries:
        fail(f"required file is absent from evidence.sha256: {item}")
    protected_file(path)
    return Path(path)


def load_json(path, label):
    path = require_evidence(path)
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"invalid {label}: {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"{label} must be a JSON object: {path}")
    return value


def load_env(path, expected_keys):
    values = {}
    lines = require_evidence(path).read_text(encoding="utf-8").splitlines()
    for line in lines:
        if not line or "=" not in line:
            fail(f"malformed env evidence: {path}")
        key, value = line.split("=", 1)
        if key in values:
            fail(f"duplicate env evidence key: {key}")
        values[key] = value
    if list(values) != list(expected_keys):
        fail(f"unexpected env evidence keys or ordering: {path}")
    return values


def load_tsv(path, expected_fields):
    with require_evidence(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != list(expected_fields):
            fail(f"unexpected TSV header: {path}")
        rows = list(reader)
    if any(None in row or any(value in (None, "") for value in row.values()) for row in rows):
        fail(f"empty or extra TSV field: {path}")
    return rows


def require_digest(value, label):
    if digest_re.fullmatch(value) is None:
        fail(f"invalid digest for {label}")


def require_directory(path, label):
    path = Path(path)
    try:
        info = path.lstat()
        canonical = path.resolve(strict=True)
        canonical.relative_to(project)
    except (OSError, ValueError):
        fail(f"invalid {label} directory: {path}")
    if not stat.S_ISDIR(info.st_mode) or path.is_symlink() or canonical != path:
        fail(f"unsafe {label} directory: {path}")
    return path


tree_digest_cache = {}


def tree_sha256(path):
    root = require_directory(path, "artifact tree")
    cache_key = str(root)
    if cache_key in tree_digest_cache:
        return tree_digest_cache[cache_key]
    files = []
    for current_raw, directories, names in os.walk(root, topdown=True, followlinks=False):
        current = Path(current_raw)
        kept = []
        for name in sorted(directories):
            candidate = current / name
            rel = candidate.relative_to(root)
            if rel.parts == (".cache",):
                continue
            info = candidate.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                fail(f"artifact tree contains an unsafe directory: {candidate}")
            kept.append(name)
        directories[:] = kept
        for name in sorted(names):
            candidate = current / name
            protected_file(candidate)
            rel = candidate.relative_to(root).as_posix()
            if "\n" in rel or "\r" in rel:
                fail(f"artifact tree contains an unsafe path: {candidate}")
            files.append((rel, sha256(candidate)))
    payload = "".join(
        f"{digest}  ./{rel}\n" for rel, digest in sorted(files)
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    tree_digest_cache[cache_key] = digest
    return digest


def wandb_tree(path):
    path = Path(path)
    if not path.is_dir() or path.is_symlink():
        return 0, ""
    root = path.resolve()
    files = []
    for item in sorted(path.rglob("*"), key=lambda value: value.as_posix()):
        if item.is_symlink():
            try:
                target = item.resolve(strict=True)
            except OSError as exc:
                fail(f"broken WandB symlink: {item}: {exc}")
            if target != root and root not in target.parents:
                fail(f"WandB symlink escapes its run directory: {item}")
            continue
        if item.is_file():
            protected_file(item)
            files.append((item.relative_to(path).as_posix(), sha256(item)))
    if not files or not any(name.endswith(".wandb") for name, _ in files):
        return len(files), ""
    payload = "".join(
        f"{digest}  {name}\n" for name, digest in files
    ).encode("utf-8")
    return len(files), hashlib.sha256(payload).hexdigest()


def validate_smoke_decision_shape(decision):
    expected_keys = {
        "checks", "decision", "inputs", "metrics", "schema", "schema_version",
    }
    if set(decision) != expected_keys or \
            decision.get("schema") != "search-r1.qwen-native-smoke-decision" or \
            decision.get("schema_version") != 1 or \
            decision.get("decision") not in {"GO", "NO-GO"}:
        fail("smoke-decision identity is invalid")
    checks = decision.get("checks")
    if not isinstance(checks, dict) or set(checks) != smoke_check_names:
        fail("smoke-decision does not contain the fixed check set")
    for name, item in checks.items():
        if not isinstance(item, dict) or set(item) != {"observed", "passed"} or \
                not isinstance(item["passed"], bool):
            fail(f"malformed smoke check: {name}")
    if all(item["passed"] for item in checks.values()) != \
            (decision["decision"] == "GO"):
        fail("smoke decision disagrees with its checks")
    inputs = decision.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {
            "catalog_sha256", "log_sha256", "trace_sha256",
            "wandb_tree_sha256"}:
        fail("smoke-decision input schema is invalid")
    metrics = decision.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != smoke_metric_names:
        fail("smoke-decision metrics schema is invalid")
    if any(isinstance(metrics[name], bool) or not isinstance(metrics[name], int) or
           metrics[name] < 0 for name in smoke_metric_names):
        fail("smoke-decision metrics must be non-negative integers")
    expected_metrics = {
        "groups": 16,
        "trajectories": 80,
    }
    if any(metrics[name] != value for name, value in expected_metrics.items()):
        fail("smoke-decision registered shape is invalid")
    observed_pairs = {
        "strict_em_positive": "strict_em_positive_count",
        "mixed_reward_group": "mixed_groups",
        "nonzero_trace_advantage": "nonzero_trace_advantages",
        "wandb_offline_history": "wandb_files",
    }
    if any(checks[check]["observed"] != metrics[metric]
           for check, metric in observed_pairs.items()):
        fail("smoke check observations disagree with metrics")
    return checks, metrics


def validate_bound_predecessor(
        marker_raw, namespace, result_parent, digest, expected_contract):
    require_digest(digest, f"{namespace} predecessor")
    marker = Path(marker_raw)
    marker_id = marker.name.removesuffix(".ok")
    expected_marker = project / "manifests" / namespace / f"{marker_id}.ok"
    if attempt_re.fullmatch(marker_id) is None or marker != expected_marker:
        fail(f"invalid {namespace} predecessor marker")
    require_evidence(marker)
    source_manifest = project / result_parent / marker_id / "evidence.sha256"
    require_evidence(source_manifest)
    if marker.read_text(encoding="utf-8").strip() != digest:
        fail(f"{namespace} marker digest mismatch")
    if sha256(source_manifest) != digest:
        fail(f"{namespace} evidence digest mismatch")
    source_entries = parse_checksum_manifest(source_manifest)
    source_root = source_manifest.parent
    outer = require_directory(
        project / "state" / "attempts" / "gpu" / marker_id,
        f"{namespace} outer attempt",
    )
    success = protected_file(outer / ".success")
    if success.stat().st_size != 0:
        fail(f"{namespace} predecessor success marker is not empty")
    for name in (".failed", ".starting", ".running"):
        path = outer / name
        if path.exists() or path.is_symlink():
            fail(f"{namespace} predecessor has a nonterminal marker: {name}")
    bindings = {
        "terminal": "success",
        "exit-code": "0",
        "result-contract": expected_contract,
        "result-root": str(source_root),
        "evidence-marker": str(marker),
        "evidence-digest": digest,
    }
    for name, expected in bindings.items():
        value = protected_file(outer / name).read_text(encoding="utf-8").strip()
        if value != expected:
            fail(f"{namespace} predecessor outer binding mismatch: {name}")
    return source_manifest.parent, source_entries


def require_run_evidence(run_dir_raw, kind):
    run_dir = require_directory(run_dir_raw, "run")
    common = ("train.log", "resolved-config.yaml", "run.env")
    if kind == "train":
        required = common + (
            "lineage.tsv",
            "native-training-contract.json",
            "traces/train_trajectories.jsonl",
            "traces/train_trajectories.manifest.json",
            "traces/train_trajectories.manifest.json.sha256",
        )
    elif kind == "eval":
        required = common + (
            "traces/eval_predictions.jsonl",
            "traces/eval_predictions.manifest.json",
            "traces/eval_predictions.manifest.json.sha256",
        )
    else:
        fail(f"unknown native run evidence kind: {kind}")
    for item in required:
        require_evidence(run_dir / item)
    return run_dir


smoke_lineage_fields = (
    "stage", "role", "run_dir", "checkpoint", "checkpoint_digest",
    "parent_checkpoint", "parent_checkpoint_digest", "checkout_commit",
    "cpu_handoff_digest", "data_manifest_sha256", "resolved_config_sha256",
    "trace_sha256", "trace_manifest_sha256", "run_contract_sha256",
    "pretrain_g3_evidence", "pretrain_g3_evidence_sha256",
)
main_lineage_fields = smoke_lineage_fields[:-2] + ("predecessor_evidence_sha256",)
index_fields = ("stage", "role", "run_dir")
checkout_commit = require_evidence(project / "manifests/git.ok").read_text(
    encoding="utf-8"
).strip()
handoff_digest = require_evidence(project / "manifests/cpu.ok").read_text(
    encoding="utf-8"
).strip()
data_manifest = require_evidence(
    project / "data/search_mix_qwen35_native_v2/manifest.json"
)
data_manifest_digest = sha256(data_manifest)
if re.fullmatch(r"[0-9a-f]{40}", checkout_commit) is None:
    fail("invalid sealed checkout commit")
require_digest(handoff_digest, "sealed CPU handoff")


def validate_common_lineage(row, stage, role, run_kind, predecessor_digest):
    if row["stage"] != stage or row["role"] != role:
        fail(f"unexpected lineage stage/role: {row['stage']}/{row['role']}")
    run_dir = require_run_evidence(row["run_dir"], run_kind)
    checkpoint = require_directory(row["checkpoint"], "checkpoint")
    parent = require_directory(row["parent_checkpoint"], "parent checkpoint")
    for key in (
        "checkpoint_digest", "parent_checkpoint_digest", "cpu_handoff_digest",
        "data_manifest_sha256", "resolved_config_sha256", "trace_sha256",
        "trace_manifest_sha256",
    ):
        require_digest(row[key], f"{stage}.{key}")
    if re.fullmatch(r"[0-9a-f]{40}", row["checkout_commit"]) is None:
        fail(f"invalid checkout commit for {stage}")
    run_contract = row["run_contract_sha256"]
    if run_contract != "-":
        require_digest(run_contract, f"{stage}.run_contract_sha256")
    if row.get("predecessor_evidence_sha256") != predecessor_digest:
        fail(f"wrong predecessor digest for {stage}")
    if row["checkout_commit"] != checkout_commit or \
            row["cpu_handoff_digest"] != handoff_digest or \
            row["data_manifest_sha256"] != data_manifest_digest:
        fail(f"sealed identity mismatch for {stage}")
    if row["checkpoint_digest"] != tree_sha256(checkpoint) or \
            row["parent_checkpoint_digest"] != tree_sha256(parent):
        fail(f"checkpoint tree mismatch for {stage}")
    if row["resolved_config_sha256"] != sha256(run_dir / "resolved-config.yaml"):
        fail(f"resolved config digest mismatch for {stage}")
    if run_kind == "train":
        trace = run_dir / "traces/train_trajectories.jsonl"
        trace_manifest = run_dir / "traces/train_trajectories.manifest.json"
        if run_contract != sha256(run_dir / "native-training-contract.json"):
            fail(f"run contract digest mismatch for {stage}")
    else:
        trace = run_dir / "traces/eval_predictions.jsonl"
        trace_manifest = run_dir / "traces/eval_predictions.manifest.json"
        if run_contract != "-":
            fail(f"eval run unexpectedly declares a run contract for {stage}")
    if row["trace_sha256"] != sha256(trace) or \
            row["trace_manifest_sha256"] != sha256(trace_manifest):
        fail(f"trace digest mismatch for {stage}")


if contract == "qwen-native-training-smoke-v1":
    env = load_env(
        results / "contract.env",
        (
            "schema", "stage", "stage_order", "decision", "manual_review_required",
            "pretrain_g3_evidence", "pretrain_g3_evidence_sha256",
        ),
    )
    expected = {
        "schema": contract,
        "stage": "smoke",
        "stage_order": "S2",
        "manual_review_required": "true",
    }
    if any(env[key] != value for key, value in expected.items()):
        fail("smoke contract.env identity mismatch")
    if env["decision"] not in {"GO", "NO-GO"}:
        fail("invalid smoke contract decision")
    pretrain_root, pretrain_entries = validate_bound_predecessor(
        env["pretrain_g3_evidence"],
        "qwen-native-gate",
        "runs/qwen-native-gate/attempts",
        env["pretrain_g3_evidence_sha256"],
        "qwen-native-gate-v1",
    )
    if any(relative(pretrain_root / name) not in pretrain_entries
           for name in ("stage.txt", "go_no_go.json")):
        fail("smoke predecessor omits its G3 decision evidence")
    if (pretrain_root / "stage.txt").read_text(encoding="utf-8").strip() != "g3":
        fail("smoke predecessor is not G3")
    if json.loads((pretrain_root / "go_no_go.json").read_bytes()).get("decision") != "GO":
        fail("smoke predecessor is not a G3 GO")
    rows = load_tsv(results / "lineage.tsv", smoke_lineage_fields)
    index = load_tsv(results / "run-index.tsv", index_fields)
    if len(rows) != 1 or len(index) != 1:
        fail("smoke evidence must contain exactly one S run")
    row = rows[0]
    if row["stage"] != "S" or row["role"] != "smoke":
        fail("smoke lineage stage/role mismatch")
    if index[0] != {"stage": "S", "role": "smoke", "run_dir": row["run_dir"]}:
        fail("smoke run-index does not match lineage")
    run_dir = require_run_evidence(row["run_dir"], "train")
    checkpoint = require_directory(row["checkpoint"], "smoke checkpoint")
    parent = require_directory(row["parent_checkpoint"], "smoke parent checkpoint")
    for key in (
        "checkpoint_digest", "parent_checkpoint_digest", "cpu_handoff_digest",
        "data_manifest_sha256", "resolved_config_sha256", "trace_sha256",
        "trace_manifest_sha256", "run_contract_sha256",
    ):
        require_digest(row[key], f"smoke.{key}")
    if re.fullmatch(r"[0-9a-f]{40}", row["checkout_commit"]) is None:
        fail("invalid smoke checkout commit")
    if row["checkout_commit"] != checkout_commit or \
            row["cpu_handoff_digest"] != handoff_digest or \
            row["data_manifest_sha256"] != data_manifest_digest:
        fail("smoke sealed identity mismatch")
    if row["checkpoint_digest"] != tree_sha256(checkpoint) or \
            row["parent_checkpoint_digest"] != tree_sha256(parent):
        fail("smoke checkpoint tree mismatch")
    smoke_artifact_digests = {
        "resolved_config_sha256": sha256(run_dir / "resolved-config.yaml"),
        "trace_sha256": sha256(run_dir / "traces/train_trajectories.jsonl"),
        "trace_manifest_sha256": sha256(
            run_dir / "traces/train_trajectories.manifest.json"
        ),
        "run_contract_sha256": sha256(run_dir / "native-training-contract.json"),
    }
    if any(row[key] != value for key, value in smoke_artifact_digests.items()):
        fail("smoke run artifact digest mismatch")
    if row["pretrain_g3_evidence"] != env["pretrain_g3_evidence"] or \
            row["pretrain_g3_evidence_sha256"] != env["pretrain_g3_evidence_sha256"]:
        fail("smoke lineage predecessor mismatch")
    checkpoint_env = load_env(
        results / "checkpoint-tree.env", ("checkpoint", "checkpoint_tree_sha256")
    )
    if checkpoint_env["checkpoint"] != row["checkpoint"] or \
            checkpoint_env["checkpoint_tree_sha256"] != row["checkpoint_digest"]:
        fail("smoke checkpoint-tree evidence mismatch")
    smoke_decision = load_json(results / "smoke-decision.json", "smoke decision")
    checks, metrics = validate_smoke_decision_shape(smoke_decision)
    if smoke_decision["decision"] != env["decision"]:
        fail("smoke-decision identity does not match contract.env")
    inputs = smoke_decision.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {
            "catalog_sha256", "log_sha256", "trace_sha256", "wandb_tree_sha256"}:
        fail("malformed smoke decision inputs")
    catalog = require_evidence(project / "data/search_mix_qwen35_native_v2/catalog.jsonl")
    expected_inputs = {
        "catalog_sha256": sha256(catalog),
        "log_sha256": sha256(run_dir / "train.log"),
        "trace_sha256": sha256(run_dir / "traces/train_trajectories.jsonl"),
    }
    wandb_files, wandb_digest = wandb_tree(run_dir / "wandb")
    if any(inputs.get(key) != value for key, value in expected_inputs.items()) or \
            inputs.get("wandb_tree_sha256") != wandb_digest or \
            metrics["wandb_files"] != wandb_files:
        fail("smoke decision input digest mismatch")
    storage = load_env(
        results / "storage.env",
        ("checkpoint_bytes", "filesystem_available_bytes", "recorded_at"),
    )
    for key in ("checkpoint_bytes", "filesystem_available_bytes"):
        if re.fullmatch(r"[0-9]+", storage[key]) is None:
            fail(f"invalid smoke storage value: {key}")
    if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", storage["recorded_at"]) is None:
        fail("invalid smoke storage timestamp")

elif contract == "qwen-native-training-main-v1":
    env = load_env(
        results / "contract.env",
        (
            "schema", "stage", "stage_order", "analysis_decision",
            "cost_contrast_group_count", "branch_authorized",
            "pretrain_g3_evidence", "pretrain_g3_evidence_sha256",
            "smoke_evidence", "smoke_evidence_sha256",
        ),
    )
    if env["schema"] != contract or env["stage"] != "main":
        fail("main contract.env identity mismatch")
    if env["analysis_decision"] not in {"GO", "NO-GO"}:
        fail("invalid main analysis decision")
    if re.fullmatch(r"[0-9]+", env["cost_contrast_group_count"]) is None:
        fail("invalid cost contrast group count")
    contrast_count = int(env["cost_contrast_group_count"])
    if env["branch_authorized"] not in {"true", "false"}:
        fail("invalid branch authorization value")
    authorized = env["branch_authorized"] == "true"
    expected_authorized = env["analysis_decision"] == "GO" and contrast_count >= 8
    if authorized != expected_authorized:
        fail("branch authorization is inconsistent with the post-R gate")
    expected_stage_order = "R60,G3,B20,C20,B-EVAL,C-EVAL" if authorized else "R60,G3"
    if env["stage_order"] != expected_stage_order:
        fail("main stage order is inconsistent with branch authorization")
    pretrain_root, pretrain_entries = validate_bound_predecessor(
        env["pretrain_g3_evidence"],
        "qwen-native-gate",
        "runs/qwen-native-gate/attempts",
        env["pretrain_g3_evidence_sha256"],
        "qwen-native-gate-v1",
    )
    if any(relative(pretrain_root / name) not in pretrain_entries
           for name in ("stage.txt", "go_no_go.json")):
        fail("main predecessor omits its G3 decision evidence")
    if (pretrain_root / "stage.txt").read_text(encoding="utf-8").strip() != "g3" or \
            json.loads((pretrain_root / "go_no_go.json").read_bytes()).get("decision") != "GO":
        fail("main pretraining predecessor is not a G3 GO")
    smoke_root, smoke_entries = validate_bound_predecessor(
        env["smoke_evidence"],
        "qwen-native-training-smoke",
        "runs/qwen-native-training/attempts",
        env["smoke_evidence_sha256"],
        "qwen-native-training-smoke-v1",
    )
    if any(relative(smoke_root / name) not in smoke_entries
           for name in ("contract.env", "smoke-decision.json")):
        fail("main predecessor omits its smoke decision evidence")
    smoke_env = load_env(
        smoke_root / "contract.env",
        (
            "schema", "stage", "stage_order", "decision", "manual_review_required",
            "pretrain_g3_evidence", "pretrain_g3_evidence_sha256",
        ),
    )
    if (smoke_env["schema"] != "qwen-native-training-smoke-v1" or
            smoke_env["stage"] != "smoke" or smoke_env["stage_order"] != "S2" or
            smoke_env["decision"] != "GO" or smoke_env["manual_review_required"] != "true"):
        fail("main smoke predecessor contract mismatch")
    try:
        smoke_decision = json.loads(
            (smoke_root / "smoke-decision.json").read_bytes()
        )
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"invalid main smoke predecessor decision: {exc}")
    if not isinstance(smoke_decision, dict):
        fail("main smoke predecessor decision must be an object")
    validate_smoke_decision_shape(smoke_decision)
    if smoke_decision.get("decision") != "GO":
        fail("main smoke predecessor decision is not GO")
    if smoke_env["pretrain_g3_evidence"] != env["pretrain_g3_evidence"] or \
            smoke_env["pretrain_g3_evidence_sha256"] != env["pretrain_g3_evidence_sha256"]:
        fail("main predecessors do not share the same G3 evidence")
    decision_path = require_evidence(results / "branch-decision.json")
    decision = json.loads(decision_path.read_bytes())
    expected_decision_keys = {
        "analysis_decision", "branches_authorized", "cost_contrast_group_count",
        "cost_contrast_group_minimum", "decision", "schema",
    }
    if set(decision) != expected_decision_keys:
        fail("unexpected branch-decision schema")
    expected_decision = {
        "analysis_decision": env["analysis_decision"],
        "branches_authorized": authorized,
        "cost_contrast_group_count": contrast_count,
        "cost_contrast_group_minimum": 8,
        "decision": "GO" if authorized else "NO-GO",
        "schema": "qwen-native-post-r-gate-v1",
    }
    if decision != expected_decision:
        fail("branch-decision.json is inconsistent with contract.env")
    analysis_root = results / "r-g3-analysis"
    for name in (
        "summary.json", "summary.md", "go_no_go.json", "per_trajectory.jsonl",
        "per_question.jsonl",
    ):
        require_evidence(analysis_root / name)
    analysis_decision = load_json(
        analysis_root / "go_no_go.json", "R-G3 decision"
    )
    analysis_summary = load_json(
        analysis_root / "summary.json", "R-G3 summary"
    )
    if analysis_decision.get("stage") != "g3" or \
            analysis_decision.get("decision") != env["analysis_decision"]:
        fail("R-G3 decision is inconsistent with main contract")
    if analysis_summary.get("decision") != env["analysis_decision"] or \
            analysis_summary.get("overall", {}).get("cost_contrast_group_count") != contrast_count:
        fail("R-G3 summary is inconsistent with main contract")
    rows = load_tsv(results / "lineage.tsv", main_lineage_fields)
    index = load_tsv(results / "run-index.tsv", index_fields)
    expected = [
        ("R", "reproduced", "train"),
        ("G3", "capability_gate", "eval"),
    ]
    if authorized:
        expected.extend((
            ("B", "control", "train"),
            ("C", "cost_aware_gated", "train"),
            ("B-EVAL", "control_eval", "eval"),
            ("C-EVAL", "cost_aware_gated_eval", "eval"),
        ))
    if len(rows) != len(expected) or len(index) != len(expected):
        fail("main lineage/run-index row count does not match branch decision")
    by_stage = {}
    for position, ((stage, role, kind), row, index_row) in enumerate(zip(expected, rows, index)):
        validate_common_lineage(row, stage, role, kind, env["smoke_evidence_sha256"])
        expected_index = {"stage": stage, "role": role, "run_dir": row["run_dir"]}
        if index_row != expected_index:
            fail(f"run-index does not match lineage at position {position}")
        by_stage[stage] = row
    r_row = by_stage["R"]
    g3_row = by_stage["G3"]
    if g3_row["checkpoint"] != r_row["checkpoint"] or \
            g3_row["checkpoint_digest"] != r_row["checkpoint_digest"]:
        fail("R-G3 does not evaluate the sealed R checkpoint")
    if g3_row["parent_checkpoint"] != r_row["checkpoint"] or \
            g3_row["parent_checkpoint_digest"] != r_row["checkpoint_digest"]:
        fail("R-G3 evaluation lineage does not bind its input checkpoint")
    g3_run = Path(g3_row["run_dir"])
    g3_trace = g3_run / "traces/eval_predictions.jsonl"
    g3_trace_digest = sha256(g3_trace)
    catalog = require_evidence(
        project / "data/search_mix_qwen35_native_v2/catalog.jsonl"
    )
    catalog_digest = sha256(catalog)
    summary_input = analysis_summary.get("input")
    summary_contract = analysis_summary.get("contract")
    summary_gate = analysis_summary.get("go_no_go")
    replay = analysis_summary.get("strict_em_replay")
    if (analysis_summary.get("schema") != "search-r1.grouped-probe-analysis" or
            analysis_summary.get("schema_version") != 2 or
            not isinstance(summary_input, dict) or
            not isinstance(summary_contract, dict) or
            not isinstance(summary_gate, dict) or
            not isinstance(replay, dict)):
        fail("R-G3 summary schema is incomplete")
    expected_summary_input = {
        "trace_path": str(g3_trace),
        "trace_sha256": g3_trace_digest,
        "catalog_path": str(catalog),
        "catalog_sha256": catalog_digest,
        "checkpoint_digest": r_row["checkpoint_digest"],
        "stage": "qwen_native_g3",
    }
    if any(summary_input.get(key) != value
           for key, value in expected_summary_input.items()):
        fail("R-G3 summary input is not bound to the G3 lineage")
    expected_probe_contract = {
        "expected_questions": 64,
        "trajectories_per_question": 5,
        "expected_trajectories": 320,
        "max_searches": 4,
    }
    if any(summary_contract.get(key) != value
           for key, value in expected_probe_contract.items()):
        fail("R-G3 registered probe contract is invalid")
    expected_replay = {
        "schema": "search-r1.strict-em-replay",
        "schema_version": 1,
        "source": "catalog.golden_answers+trace.extracted_answer/final_answer",
        "verified": True,
        "trajectory_count": 320,
        "trace_sha256": g3_trace_digest,
        "catalog_sha256": catalog_digest,
    }
    if any(replay.get(key) != value for key, value in expected_replay.items()):
        fail("R-G3 strict-EM replay binding is invalid")
    for key in ("strict_em_positive_count", "subem_positive_count"):
        value = replay.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 320:
            fail(f"R-G3 strict-EM replay count is invalid: {key}")
    expected_decision_binding = {
        "schema": "search-r1.grouped-probe-analysis",
        "schema_version": 2,
        "stage": "g3",
        "decision": env["analysis_decision"],
        "trace_sha256": g3_trace_digest,
        "catalog_sha256": catalog_digest,
        "checkpoint_digest": r_row["checkpoint_digest"],
    }
    if any(analysis_decision.get(key) != value
           for key, value in expected_decision_binding.items()) or \
            analysis_decision.get("strict_em_replay") != replay or \
            summary_gate.get("decision") != env["analysis_decision"]:
        fail("R-G3 decision is not bound to its strict-EM summary")
    if authorized:
        for stage in ("B", "C"):
            row = by_stage[stage]
            if row["parent_checkpoint"] != r_row["checkpoint"] or \
                    row["parent_checkpoint_digest"] != r_row["checkpoint_digest"]:
                fail(f"{stage} does not descend from the sealed R checkpoint")
        for eval_stage, train_stage in (("B-EVAL", "B"), ("C-EVAL", "C")):
            eval_row = by_stage[eval_stage]
            train_row = by_stage[train_stage]
            if eval_row["checkpoint"] != train_row["checkpoint"] or \
                    eval_row["checkpoint_digest"] != train_row["checkpoint_digest"]:
                fail(f"{eval_stage} does not evaluate the matching branch checkpoint")
            if eval_row["parent_checkpoint"] != train_row["checkpoint"] or \
                    eval_row["parent_checkpoint_digest"] != train_row["checkpoint_digest"]:
                fail(f"{eval_stage} lineage does not bind its input checkpoint")
        paired = results / "paired"
        for name in (
            "summary.json", "summary.md", "paired_results.csv", "correct_questions.csv",
            "wrong_questions.csv", "search_transition.csv",
        ):
            require_evidence(paired / name)
        paired_summary = load_json(paired / "summary.json", "paired summary")
        expected_paired_keys = {
            "schema_version", "expected_rows", "cost_lambda", "max_searches",
            "inputs", "stages", "comparisons", "catalog", "formal_contract",
        }
        if set(paired_summary) != expected_paired_keys or \
                paired_summary["schema_version"] != 1 or \
                paired_summary["expected_rows"] != 128 or \
                paired_summary["cost_lambda"] != 0.10 or \
                paired_summary["max_searches"] != 4:
            fail("paired summary schema or fixed evaluation contract is invalid")
        paired_inputs = paired_summary.get("inputs")
        paired_stages = paired_summary.get("stages")
        paired_comparisons = paired_summary.get("comparisons")
        if not isinstance(paired_inputs, dict) or set(paired_inputs) != {
                "control", "cost_aware_gated"} or \
                not isinstance(paired_stages, dict) or set(paired_stages) != {
                    "control", "cost_aware_gated"} or \
                not isinstance(paired_comparisons, dict) or \
                set(paired_comparisons) != {"cost_aware_gated"}:
            fail("paired summary roles are invalid")
        paired_runs = {
            "control": (
                Path(by_stage["B-EVAL"]["run_dir"]),
                "qwen_native_b",
            ),
            "cost_aware_gated": (
                Path(by_stage["C-EVAL"]["run_dir"]),
                "qwen_native_c",
            ),
        }
        for role, (run_dir, expected_stage) in paired_runs.items():
            trace = run_dir / "traces/eval_predictions.jsonl"
            expected_input = {
                "path": str(trace),
                "sha256": sha256(trace),
            }
            if paired_inputs[role] != expected_input or \
                    not isinstance(paired_stages[role], dict) or \
                    paired_stages[role].get("stage") != expected_stage:
                fail(f"paired summary input mismatch for {role}")
        paired_catalog = paired_summary.get("catalog")
        catalog_rows = catalog.read_text(encoding="utf-8").splitlines()
        if not isinstance(paired_catalog, dict) or paired_catalog != {
                "path": str(catalog),
                "sha256": catalog_digest,
                "row_count": len(catalog_rows),
                "matched_rows": 128,
                "replay_status": "passed",
                "strict_em_scorer": "qa_em.em_check",
        }:
            fail("paired summary catalog replay binding is invalid")
        try:
            data_contract = json.loads(data_manifest.read_bytes())
        except json.JSONDecodeError as exc:
            fail(f"invalid formal data manifest: {exc}")
        artifacts = data_contract.get("artifacts", {}) \
            if isinstance(data_contract, dict) else {}
        val_artifact = artifacts.get("val", {}) \
            if isinstance(artifacts, dict) else {}
        sample_ids = val_artifact.get("sample_ids") \
            if isinstance(val_artifact, dict) else None
        if not isinstance(sample_ids, list) or len(sample_ids) != 128:
            fail("formal data manifest does not bind val_128 sample IDs")
        sample_ids_digest = hashlib.sha256(json.dumps(
            sample_ids, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        expected_formal_contract = {
            "mode": "qwen35_native_v2_b_c",
            "data_manifest": {
                "path": str(data_manifest),
                "sha256": data_manifest_digest,
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
                    "checkpoint_digest": by_stage["B"]["checkpoint_digest"],
                },
                "cost_aware_gated": {
                    "stage": "qwen_native_c",
                    "checkpoint_digest": by_stage["C"]["checkpoint_digest"],
                },
            },
        }
        if paired_summary.get("formal_contract") != expected_formal_contract:
            fail("paired summary formal contract is invalid")
else:
    fail(f"unsupported native training contract: {contract}")
PY
}

validate_followup_success_artifacts() {
    local project="${CAP[project_root]}" attempt="${CAP[attempt]}" expected_uid
    local contract_file result_root_file marker_file digest_file results expected_results_parent
    local marker expected_marker evidence expected_digest recorded_digest line relative path previous=''
    local contract results_relative_parent marker_relative_parent
    local entry_count=0 file
    local -a required=()
    declare -A seen=()
    if [[ "${CAP[mode]}" == production ]]; then expected_uid=0; else expected_uid="$(id -u)"; fi
    contract_file="$attempt/result-contract"
    result_root_file="$attempt/result-root"
    marker_file="$attempt/evidence-marker"
    digest_file="$attempt/evidence-digest"
    for file in "$contract_file" "$result_root_file" "$marker_file" "$digest_file"; do
        validate_protected_regular "$file" "$expected_uid" || return 1
        (( $(wc -l <"$file") == 1 )) || return 1
        (( $(stat -c '%s' -- "$file") <= 4096 )) || return 1
    done
    contract="$(tr -d '\r\n' <"$contract_file")"
    case "$contract" in
        cost-aware-gated-v1)
            results_relative_parent='runs/cost-aware-gated/attempts'
            marker_relative_parent='manifests/cost-aware-gated'
            required=(
                paired_results.csv correct_questions.csv wrong_questions.csv search_transition.csv
                summary.json summary.md lineage.tsv run-index.tsv
                gated_training_metrics.csv gated_training_curves.svg
            )
            ;;
        search-opportunity-gate-v1)
            results_relative_parent='runs/search-opportunity-gate/attempts'
            marker_relative_parent='manifests/search-opportunity-gate'
            required=(
                summary.json summary.md go_no_go.json per_question.jsonl
                correct_questions.csv wrong_questions.csv two_plus_search.csv
                redundant_search_candidates.csv strata.csv lineage.tsv run-index.tsv
            )
            ;;
        group-probe-v1)
            results_relative_parent='runs/group-probe/attempts'
            marker_relative_parent='manifests/group-probe'
            required=(
                summary.json summary.md go_no_go.json per_trajectory.jsonl
                per_question.jsonl lineage.tsv run-index.tsv
            )
            ;;
        qwen-native-gate-v1)
            results_relative_parent='runs/qwen-native-gate/attempts'
            marker_relative_parent='manifests/qwen-native-gate'
            required=(
                summary.json summary.md go_no_go.json per_trajectory.jsonl
                per_question.jsonl lineage.tsv run-index.tsv stage.txt sampling.json
            )
            ;;
        qwen-native-training-smoke-v1)
            results_relative_parent='runs/qwen-native-training/attempts'
            marker_relative_parent='manifests/qwen-native-training-smoke'
            required=(
                contract.env lineage.tsv run-index.tsv storage.env checkpoint-tree.env
                smoke-decision.json
            )
            ;;
        qwen-native-training-main-v1)
            results_relative_parent='runs/qwen-native-training/attempts'
            marker_relative_parent='manifests/qwen-native-training-main'
            required=(
                contract.env lineage.tsv run-index.tsv branch-decision.json
                r-g3-analysis/summary.json r-g3-analysis/summary.md
                r-g3-analysis/go_no_go.json r-g3-analysis/per_trajectory.jsonl
                r-g3-analysis/per_question.jsonl
            )
            ;;
        *) return 1 ;;
    esac

    results="$(canonical_existing "$(tr -d '\r\n' <"$result_root_file")")" || return 1
    expected_results_parent="$project/$results_relative_parent"
    [[ -d "$expected_results_parent" && ! -L "$expected_results_parent" ]] || return 1
    [[ "$(canonical_existing "$(dirname -- "$results")")" == "$expected_results_parent" ]] || return 1
    [[ "$(basename -- "$results")" == "$(basename -- "$attempt")" &&
        -d "$results" && ! -L "$results" ]] || return 1

    marker="$(canonical_existing "$(tr -d '\r\n' <"$marker_file")")" || return 1
    expected_marker="$project/$marker_relative_parent/$(basename -- "$attempt").ok"
    [[ "$marker" == "$expected_marker" ]] || return 1
    validate_protected_regular "$marker" "$expected_uid" || return 1
    evidence="$results/evidence.sha256"
    validate_protected_regular "$evidence" "$expected_uid" && [[ -s "$evidence" ]] || return 1

    while IFS= read -r line; do
        [[ "$line" =~ ^([0-9a-f]{64})\ \ ([A-Za-z0-9._/-]+)$ ]] || return 1
        expected_digest="${BASH_REMATCH[1]}"
        relative="${BASH_REMATCH[2]}"
        [[ "$relative" != /* && "$relative" != *'//'*
            && "/$relative/" != *'/../'* && "$relative" > "$previous" ]] || return 1
        previous="$relative"
        path="$project/$relative"
        [[ "$(canonical_existing "$path")" == "$path" ]] || return 1
        validate_protected_regular "$path" "$expected_uid" || return 1
        [[ "$(file_sha256 "$path")" == "$expected_digest" ]] || return 1
        seen["$relative"]=1
        entry_count=$((entry_count + 1))
        sync_required "$path" || return 1
    done <"$evidence"
    ((entry_count >= 8)) || return 1
    for file in "${required[@]}"; do
        [[ ${seen["$results_relative_parent/$(basename -- "$attempt")/$file"]+present} ]] || return 1
    done
    case "$contract" in
        qwen-native-training-smoke-v1|qwen-native-training-main-v1)
            validate_qwen_native_training_evidence "$contract" "$project" "$results" \
                "$(basename -- "$attempt")" "$expected_uid" || return 1
            ;;
    esac

    RESULTS_DIGEST="$(file_sha256 "$evidence")" || return 1
    recorded_digest="$(tr -d '\r\n' <"$marker")"
    [[ "$recorded_digest" == "$RESULTS_DIGEST" ]] || return 1
    [[ "$(tr -d '\r\n' <"$digest_file")" == "$RESULTS_DIGEST" ]] || return 1
    sync_required "$evidence" || return 1
    sync_required "$results" || return 1
    sync_required "$marker" || return 1
    sync_required "$(dirname -- "$marker")" || return 1
    sync_required "$attempt"
}

validate_success_artifacts() {
    local contract="${CAP[attempt]}/result-contract"
    RESULTS_DIGEST='not-required'
    [[ "$WORK_STATE" == success ]] || return 0
    if [[ -e "$contract" || -L "$contract" ]]; then
        validate_followup_success_artifacts
    else
        validate_legacy_success_artifacts
    fi
}

wait_for_terminal_evidence() {
    local deadline=$(( $(date +%s) + CAP[wait_timeout_seconds] )) attempt="${CAP[attempt]}"
    while (( $(date +%s) <= deadline )); do
        if [[ -e "$attempt/.starting" || -L "$attempt/.starting" || -e "$attempt/.running" || -L "$attempt/.running" ]]; then
            sleep 1
            continue
        fi
        if [[ -e "$attempt/exit-code" || -L "$attempt/exit-code" ||
              -e "$attempt/terminal" || -L "$attempt/terminal" ||
              -e "$attempt/.success" || -L "$attempt/.success" ||
              -e "$attempt/.failed" || -L "$attempt/.failed" ]]; then
            validate_terminal_evidence
            return
        fi
        sleep 1
    done
    return 1
}

validate_paths_and_files() {
    local expected_uid project attempt lock latest_gpu phase_log shutdown_binary script latest_attempt
    if [[ "${CAP[mode]}" == production ]]; then expected_uid=0; else expected_uid="$(id -u)"; fi
    project="$(canonical_existing "${CAP[project_root]}")" || return 1
    attempt="$(canonical_existing "${CAP[attempt]}")" || return 1
    lock="$(canonical_existing "${CAP[lock_file]}")" || return 1
    latest_gpu="$(canonical_existing "${CAP[latest_gpu]}")" || return 1
    phase_log="$(canonical_existing "${CAP[phase_log]}")" || return 1
    shutdown_binary="$(canonical_existing "${CAP[shutdown_binary]}")" || return 1
    script="$(canonical_existing "${CAP[watchdog_script]}")" || return 1
    [[ "$project" == "${CAP[project_root]}" && "$attempt" == "${CAP[attempt]}" ]] || return 1
    [[ "$lock" == "${CAP[lock_file]}" && "$latest_gpu" == "${CAP[latest_gpu]}" ]] || return 1
    [[ "$phase_log" == "${CAP[phase_log]}" ]] || return 1
    [[ "$shutdown_binary" == "${CAP[shutdown_binary]}" && "$script" == "${CAP[watchdog_script]}" ]] || return 1
    [[ "$SCRIPT_PATH" == "$script" ]] || return 1
    validate_attempt_path "$project" "$attempt" || return 1
    [[ "$lock" == "$project/state/phase.lock" && "$latest_gpu" == "$project/state/latest/gpu" ]] || return 1
    [[ "$phase_log" == "$attempt/phase.log" ]] || return 1
    [[ "$script" == "$project/checkout/scripts/autodl/04_watch_and_shutdown.sh" ]] || return 1
    validate_protected_regular "$lock" "$expected_uid" || return 1
    validate_protected_regular "$latest_gpu" "$expected_uid" || return 1
    validate_protected_regular "$phase_log" "$expected_uid" || return 1
    (( $(wc -l <"$latest_gpu") == 1 )) || return 1
    latest_attempt="$(tr -d '\r\n' <"$latest_gpu")"
    [[ "$latest_attempt" == "$attempt" ]] || return 1
    [[ "$(file_sha256 "$script")" == "${CAP[watchdog_script_sha256]}" ]] || return 1
    validate_protected_executable "$shutdown_binary" "$expected_uid" || return 1
    [[ "$(file_sha256 "$shutdown_binary")" == "${CAP[shutdown_binary_sha256]}" ]] || return 1
    if [[ "${CAP[mode]}" == production ]]; then
        [[ "$shutdown_binary" == "$PRODUCTION_SHUTDOWN_BINARY" ]] || return 1
    fi
}

lock_is_held_by_watchdog() {
    local fd_path fd_identity path_identity
    [[ -n "$LOCK_FD" ]] || return 1
    fd_path="/proc/$$/fd/$LOCK_FD"
    [[ "$(readlink -f -- "$fd_path")" == "${CAP[lock_file]}" ]] || return 1
    fd_identity="$(stat -Lc '%d:%i' -- "$fd_path")" || return 1
    path_identity="$(stat -c '%d:%i' -- "${CAP[lock_file]}")" || return 1
    [[ "$fd_identity" == "$path_identity" ]]
}

acquire_phase_lock() {
    is_regular_file "${CAP[lock_file]}" || return 1
    exec {LOCK_FD}<>"${CAP[lock_file]}"
    flock -n "$LOCK_FD" || return 1
    lock_is_held_by_watchdog
}

verify_authorization() {
    revalidate_capability || return 1
    validate_test_contract || return 1
    validate_worker_admission || return 1
    validate_host_and_roots "${CAP[mode]}" "${CAP[persistent_root]}" "${CAP[project_root]}" "${CAP[test_root]}" || return 1
    validate_paths_and_files || return 1
    validate_checkout "${CAP[project_root]}" "${CAP[commit]}" "${CAP[checkout_tree_sha256]}" || return 1
    lock_is_held_by_watchdog || return 1
    sync_required "${CAP[phase_log]}" || return 1
    sync_required "${CAP[attempt]}" || return 1
    validate_terminal_evidence || return 1
    validate_success_artifacts
}

dispatch_backend() {
    local rc
    local -a shutdown_argv=("${CAP[shutdown_binary]}")
    if [[ "${CAP[mode]}" == test ]]; then
        append_test_event "backend:${shutdown_argv[0]} argc=$((${#shutdown_argv[@]} - 1))"
        return "${CAP[test_backend_rc]}"
    fi
    set +e
    /usr/bin/env -i PATH=/usr/sbin:/usr/bin:/sbin:/bin \
        /usr/bin/timeout --signal=TERM --kill-after=5s 30s \
        "${shutdown_argv[@]}"
    rc=$?
    set -e
    return "$rc"
}

watchdog_worker() {
    local capability="$1" backend_rc safe_rc safe_state safe_results
    validate_test_mode_setting || {
        printf 'Invalid test-mode setting; keeping the instance running.\n' >&2
        return 1
    }
    load_capability "$capability" || {
        printf 'Invalid shutdown capability; keeping the instance running.\n' >&2
        return 1
    }
    validate_test_contract || {
        printf 'Test/production mode mismatch; keeping the instance running.\n' >&2
        return 1
    }
    wait_for_worker_admission || {
        printf 'Watchdog admission is not durable; keeping the instance running.\n' >&2
        return 1
    }
    if ! wait_for_terminal_evidence; then
        publish_skipped terminal-evidence-incomplete
        return 0
    fi
    if [[ "$WORK_RC" == 75 ]]; then
        publish_skipped lock-conflict
        return 0
    fi
    if ! acquire_phase_lock; then
        publish_skipped phase-lock-unavailable
        return 0
    fi
    if ! verify_authorization; then
        publish_skipped authorization-revalidation-failed
        return 0
    fi
    safe_rc="$WORK_RC"
    safe_state="$WORK_STATE"
    safe_results="$RESULTS_DIGEST"
    if [[ "${CAP[dry_run]}" == true ]]; then
        publish_skipped dry-run
        return 0
    fi
    publish_state shutdown-safe \
        "at=$(utc_now)"$'\n'"attempt=${CAP[attempt]}"$'\n'"work_state=$safe_state"$'\n'"work_exit_code=$safe_rc"$'\n'"results_digest=$safe_results"$'\n' || return 1
    if ! verify_authorization; then
        publish_skipped authorization-changed-after-shutdown-safe
        return 0
    fi
    if [[ "$WORK_RC" != "$safe_rc" || "$WORK_STATE" != "$safe_state" ||
          "$RESULTS_DIGEST" != "$safe_results" ]]; then
        publish_skipped authorization-changed-after-shutdown-safe
        return 0
    fi
    publish_state shutdown-requested \
        "at=$(utc_now)"$'\n'"backend=${CAP[shutdown_binary]}"$'\n'"backend_kind=$([[ ${CAP[mode]} == test ]] && printf test || printf autodl-guest)"$'\n'"work_exit_code=$safe_rc"$'\n'"provider_control_plane_confirmed=false"$'\n' || return 1
    if dispatch_backend; then
        publish_state shutdown-dispatched \
            "at=$(utc_now)"$'\n'"backend=${CAP[shutdown_binary]}"$'\n'"backend_exit_code=0"$'\n'"provider_control_plane_confirmed=false"$'\n' || return 1
        printf 'Shutdown backend dispatched; confirm stopped state and billing in the AutoDL console.\n'
        return 0
    else
        backend_rc=$?
    fi
    publish_state shutdown-failed \
        "at=$(utc_now)"$'\n'"backend=${CAP[shutdown_binary]}"$'\n'"backend_exit_code=$backend_rc"$'\n'"work_exit_code=$WORK_RC"$'\n' || return 1
    printf 'Shutdown backend failed with exit code %s; the instance remains running.\n' "$backend_rc" >&2
    return 1
}

create_capability_and_launch() {
    local attempt_input="$1" dry_run="$2" foreground="$3"
    local mode project_root persistent_root attempt attempts_root capability phase_log lock_file
    local latest_gpu checkout commit tree tree_manifest shutdown_binary shutdown_digest script_digest
    local expected_uid test_root=- test_event_log=- test_backend_rc=- wait_timeout launch_nonce
    local watchdog_log capability_value worker_pid pid_file admitted_file tool

    validate_test_mode_setting || die 'SEARCH_R1_AUTODL_TEST_MODE must be unset or exactly 1.'
    for tool in /usr/bin/nohup /usr/bin/setsid /usr/bin/env /usr/bin/bash /usr/bin/timeout; do
        validate_protected_executable "$tool" 0 || die "Unsafe system launch tool: $tool"
    done
    project_root="$(canonical_existing "${AUTODL_ROOT:-$DEFAULT_PROJECT_ROOT}")" || die 'Project root does not exist.'
    if [[ "${SEARCH_R1_AUTODL_TEST_MODE:-}" == 1 ]]; then
        mode=test
        test_root="$(canonical_existing "${SEARCH_R1_AUTODL_TEST_ROOT:?Set SEARCH_R1_AUTODL_TEST_ROOT in test mode.}")" || die 'Invalid test root.'
        persistent_root="$(canonical_existing "${AUTODL_PERSISTENT_ROOT:-$test_root}")" || die 'Invalid test persistent root.'
        test_event_log="$(canonical_existing "${SEARCH_R1_AUTODL_TEST_SHUTDOWN_LOG:?Set SEARCH_R1_AUTODL_TEST_SHUTDOWN_LOG.}")" || die 'Invalid test event log.'
        shutdown_binary="$(canonical_existing "${SEARCH_R1_AUTODL_TEST_SHUTDOWN_BINARY:?Set SEARCH_R1_AUTODL_TEST_SHUTDOWN_BINARY.}")" || die 'Invalid test shutdown binary.'
        test_backend_rc="${SEARCH_R1_AUTODL_TEST_BACKEND_RC:-0}"
        [[ "$test_backend_rc" =~ ^(0|[1-9][0-9]*)$ && "$test_backend_rc" -le 255 ]] || die 'Invalid test backend exit code.'
        expected_uid="$(id -u)"
    else
        mode=production
        persistent_root="$(canonical_existing "${AUTODL_PERSISTENT_ROOT:-$DEFAULT_PERSISTENT_ROOT}")" || die 'Persistent root does not exist.'
        shutdown_binary="$(canonical_existing "$PRODUCTION_SHUTDOWN_BINARY")" || die 'AutoDL shutdown binary is unavailable.'
        [[ "$shutdown_binary" == "$PRODUCTION_SHUTDOWN_BINARY" ]] || die '/usr/bin/shutdown must be a regular non-symlink path.'
        expected_uid=0
    fi
    validate_host_and_roots "$mode" "$persistent_root" "$project_root" "$test_root" || die 'Host or persistent-root authorization failed.'
    attempt="$(canonical_existing "$attempt_input")" || die 'The exact GPU attempt path does not exist.'
    [[ "$attempt_input" == "$attempt" ]] || die 'Pass the canonical exact GPU attempt path.'
    validate_attempt_path "$project_root" "$attempt" || die 'Attempt is not an exact GPU phase attempt.'
    phase_log="$attempt/phase.log"
    lock_file="$project_root/state/phase.lock"
    latest_gpu="$project_root/state/latest/gpu"
    is_regular_file "$phase_log" || die 'phase.log must be a regular non-symlink file.'
    is_regular_file "$lock_file" || die 'phase.lock must be a regular non-symlink file.'
    is_regular_file "$latest_gpu" || die 'The exact GPU attempt must have a regular latest pointer.'
    [[ "$(tr -d '\r\n' <"$latest_gpu")" == "$attempt" ]] || die 'The exact GPU attempt is no longer current.'
    validate_protected_executable "$shutdown_binary" "$expected_uid" || die 'Shutdown backend ownership or permissions are unsafe.'
    shutdown_digest="$(file_sha256 "$shutdown_binary")" || die 'Cannot hash the shutdown backend.'
    checkout="$project_root/checkout"
    [[ "$SCRIPT_PATH" == "$checkout/scripts/autodl/04_watch_and_shutdown.sh" ]] || die 'Run the watchdog from the pinned checkout.'
    commit="$(tr -d '\r\n' <"$project_root/manifests/git.ok")"
    tree_manifest="$(tr -d '\r\n' <"$project_root/manifests/checkout-tree.sha256")"
    [[ "$commit" =~ ^[0-9a-f]{40}$ && "$tree_manifest" =~ ^[0-9a-f]{64}$ ]] || die 'Invalid checkout identity manifest.'
    validate_checkout "$project_root" "$commit" "$tree_manifest" || die 'Checkout identity validation failed.'
    tree="$tree_manifest"
    script_digest="$(file_sha256 "$SCRIPT_PATH")" || die 'Cannot hash watchdog script.'
    wait_timeout="${SEARCH_R1_AUTODL_WATCH_TIMEOUT_SECONDS:-604800}"
    [[ "$wait_timeout" =~ ^[1-9][0-9]*$ && "$wait_timeout" -le 604800 ]] || die 'Watch timeout must be 1..604800 seconds.'
    launch_nonce="$(printf '%s\n' "$(utc_now)-$$-$RANDOM-$RANDOM" | sha256sum | cut -d' ' -f1)"
    capability="$attempt/$CAPABILITY_NAME"
    watchdog_log="$attempt/$WATCHDOG_LOG_NAME"
    pid_file="$attempt/shutdown-watchdog-pid"
    admitted_file="$attempt/shutdown-watchdog-admitted"
    for path in "$capability" "$watchdog_log" "$attempt/shutdown-safe" "$attempt/shutdown-requested" \
        "$attempt/shutdown-dispatched" "$attempt/shutdown-failed" "$attempt/shutdown-skipped" \
        "$pid_file" "$admitted_file"; do
        [[ ! -e "$path" && ! -L "$path" ]] || die "Refusing an already armed attempt: $path"
    done
    (umask 077 && : >"$watchdog_log")
    chmod 0600 "$watchdog_log"
    sync_required "$watchdog_log" || die 'Cannot sync watchdog log.'
    capability_value=\
"schema_version"$'\t'"1"$'\n'\
"mode"$'\t'"$mode"$'\n'\
"authorized"$'\t'"yes"$'\n'\
"capability_path"$'\t'"$capability"$'\n'\
"project_root"$'\t'"$project_root"$'\n'\
"persistent_root"$'\t'"$persistent_root"$'\n'\
"attempt"$'\t'"$attempt"$'\n'\
"commit"$'\t'"$commit"$'\n'\
"checkout_tree_sha256"$'\t'"$tree"$'\n'\
"lock_file"$'\t'"$lock_file"$'\n'\
"latest_gpu"$'\t'"$latest_gpu"$'\n'\
"phase_log"$'\t'"$phase_log"$'\n'\
"shutdown_binary"$'\t'"$shutdown_binary"$'\n'\
"shutdown_binary_sha256"$'\t'"$shutdown_digest"$'\n'\
"watchdog_script"$'\t'"$SCRIPT_PATH"$'\n'\
"watchdog_script_sha256"$'\t'"$script_digest"$'\n'\
"dry_run"$'\t'"$dry_run"$'\n'\
"foreground"$'\t'"$foreground"$'\n'\
"launch_nonce"$'\t'"$launch_nonce"$'\n'\
"wait_timeout_seconds"$'\t'"$wait_timeout"$'\n'\
"test_root"$'\t'"$test_root"$'\n'\
"test_event_log"$'\t'"$test_event_log"$'\n'\
"test_backend_rc"$'\t'"$test_backend_rc"$'\n'
    atomic_publish "$capability" "$capability_value" || die 'Cannot publish shutdown capability.'
    validate_owner_and_mode "$capability" "$expected_uid" 600 || die 'Capability is not owner-only.'

    if [[ "$foreground" == true ]]; then
        [[ "$mode" == test ]] || die '--test-foreground is test-only.'
        atomic_publish "$pid_file" "$$"$'\n' || die 'Cannot publish foreground watchdog PID.'
        atomic_publish "$admitted_file" "$launch_nonce"$'\n' || die 'Cannot admit foreground watchdog.'
        watchdog_worker "$capability"
        return
    fi
    if [[ "$mode" == test ]]; then
        /usr/bin/nohup /usr/bin/setsid /usr/bin/env \
            SEARCH_R1_AUTODL_TEST_MODE=1 \
            SEARCH_R1_AUTODL_TEST_ROOT="$test_root" \
            SEARCH_R1_AUTODL_TEST_SHUTDOWN_LOG="$test_event_log" \
            SEARCH_R1_AUTODL_TEST_SHUTDOWN_BINARY="$shutdown_binary" \
            SEARCH_R1_AUTODL_TEST_BACKEND_RC="$test_backend_rc" \
            /usr/bin/bash "$SCRIPT_PATH" --worker "$capability" >>"$watchdog_log" 2>&1 </dev/null &
    else
        /usr/bin/nohup /usr/bin/setsid /usr/bin/env -i PATH=/usr/sbin:/usr/bin:/sbin:/bin \
            /usr/bin/bash "$SCRIPT_PATH" --worker "$capability" >>"$watchdog_log" 2>&1 </dev/null &
    fi
    worker_pid=$!
    if ! atomic_publish "$pid_file" "$worker_pid"$'\n'; then
        kill -TERM "$worker_pid" 2>/dev/null || true
        wait "$worker_pid" 2>/dev/null || true
        die 'Cannot publish watchdog PID.'
    fi
    if ! atomic_publish "$admitted_file" "$launch_nonce"$'\n'; then
        kill -TERM "$worker_pid" 2>/dev/null || true
        wait "$worker_pid" 2>/dev/null || true
        die 'Cannot publish watchdog admission.'
    fi
    printf 'Armed shutdown watchdog for exact attempt: %s\n' "$attempt"
    printf 'Watchdog log: %s\n' "$watchdog_log"
}

usage() {
    printf 'Usage: bash %s [--dry-run] EXACT_GPU_ATTEMPT\n' "$0" >&2
}

case "${1:-}" in
    --worker)
        [[ "$#" == 2 ]] || die 'Worker requires one capability path.'
        watchdog_worker "$2"
        ;;
    --dry-run)
        [[ "$#" == 2 ]] || { usage; exit 64; }
        create_capability_and_launch "$2" true false
        ;;
    --test-foreground)
        [[ "$#" == 2 ]] || { usage; exit 64; }
        create_capability_and_launch "$2" false true
        ;;
    --test-foreground-dry-run)
        [[ "$#" == 2 ]] || { usage; exit 64; }
        create_capability_and_launch "$2" true true
        ;;
    '')
        usage
        exit 64
        ;;
    *)
        [[ "$#" == 1 ]] || { usage; exit 64; }
        create_capability_and_launch "$1" false false
        ;;
esac
