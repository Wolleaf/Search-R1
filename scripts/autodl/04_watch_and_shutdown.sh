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
