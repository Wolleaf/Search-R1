#!/usr/bin/env bash
set -Eeuo pipefail

PATH='/usr/sbin:/usr/bin:/sbin:/bin'
readonly PATH
export PATH
export LC_ALL=C

readonly AR_WATCHDOG_SCHEMA=search-r1.qwen-native-ar-eval-watchdog-v1
readonly AR_RESULT_CONTRACT=qwen-native-training-ar-eval-only-v1
readonly AR_RESULT_NAMESPACE=qwen-native-training-ar-eval-only
readonly AR_CPU_CONTRACT=qwen-native-ar-eval-only-cpu-v1
readonly AR_RUNNER_RELATIVE=operator/14_gpu_qwen_native_ar_eval_only.sh
readonly AR_WATCHDOG_RELATIVE=operator/15_watch_qwen_native_ar_eval_only.sh
readonly DEFAULT_PROJECT_ROOT=/root/autodl-tmp/search-r1
readonly DEFAULT_PERSISTENT_ROOT=/root/autodl-tmp
readonly PRODUCTION_SHUTDOWN_BINARY=/usr/bin/shutdown
readonly TERMINAL_SENTINEL_PREFIX=AUTODL_PHASE_TERMINAL
readonly CAPABILITY_NAME=ar-shutdown-capability.tsv
readonly WATCHDOG_LOG_NAME=ar-shutdown-watchdog.log
readonly WATCHDOG_PID_NAME=ar-shutdown-watchdog-pid
readonly WATCHDOG_ADMITTED_NAME=ar-shutdown-watchdog-admitted
readonly SCRIPT_PATH="$(readlink -f -- "${BASH_SOURCE[0]}")"

declare -A CAP=()
declare -A EVIDENCE_SEEN=()
CAPABILITY_DIGEST=''
CAPABILITY_PATH=''
LOCK_FD=''
WORK_RC=''
WORK_STATE=''
RESULTS_DIGEST='not-required'

utc_now() {
    date -u +'%Y-%m-%dT%H:%M:%SZ'
}

die() {
    printf 'A/R watchdog: %s\n' "$*" >&2
    exit 64
}

is_regular_file() {
    [[ -f "$1" && ! -L "$1" ]]
}

canonical_existing() {
    local value
    value="$(readlink -f -- "$1")" || return 1
    [[ -e "$value" || -L "$value" ]] || return 1
    printf '%s\n' "$value"
}

is_contained() {
    local parent="$1" child="$2"
    [[ "$child" == "$parent" || "$child" == "$parent/"* ]]
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
    sync -f -- "$1" 2>/dev/null
}

validate_protected_regular() {
    local path="$1" expected_uid="$2" mode
    is_regular_file "$path" || return 1
    [[ "$(stat -c '%u' -- "$path")" == "$expected_uid" ]] || return 1
    mode="$(stat -c '%a' -- "$path")" || return 1
    (( (8#$mode & 0022) == 0 ))
}

validate_protected_executable() {
    validate_protected_regular "$1" "$2" && [[ -x "$1" ]]
}

atomic_publish() {
    local path="$1" content="$2" directory temporary
    directory="$(dirname -- "$path")"
    temporary="$directory/.${path##*/}.tmp.$$.$RANDOM"
    [[ -d "$directory" && ! -L "$directory" && ! -e "$path" && ! -L "$path" &&
        ! -e "$temporary" && ! -L "$temporary" ]] || return 1
    (umask 077 && printf '%s' "$content" >"$temporary") || return 1
    chmod 0600 "$temporary" || return 1
    sync_required "$temporary" || return 1
    mv -- "$temporary" "$path" || return 1
    sync_required "$directory"
}

env_single_value() {
    local path="$1" key="$2"
    awk -F= -v key="$key" '
        $1 == key { count += 1; value = substr($0, length(key) + 2) }
        END { if (count != 1) exit 1; print value }
    ' "$path"
}

validate_small_marker() {
    local path="$1" expected_uid="$2" value
    validate_protected_regular "$path" "$expected_uid" || return 1
    (( $(wc -l <"$path") == 1 && $(stat -c '%s' -- "$path") <= 4096 )) || return 1
    value="$(tr -d '\r\n' <"$path")"
    [[ -n "$value" ]]
}

validate_mount() {
    local persistent_root="$1" mount_line root_line target fs source majmin root_majmin
    command -v findmnt >/dev/null 2>&1 || return 1
    mount_line="$(findmnt -rn -T "$persistent_root" -o TARGET,FSTYPE,SOURCE,MAJ:MIN)" || return 1
    root_line="$(findmnt -rn -T / -o TARGET,FSTYPE,SOURCE,MAJ:MIN)" || return 1
    read -r target fs source majmin <<<"$mount_line"
    read -r _ _ _ root_majmin <<<"$root_line"
    [[ "$target" == "$persistent_root" && -n "$source" && "$majmin" != "$root_majmin" ]] || return 1
    [[ ! "$fs" =~ ^(overlay|tmpfs|ramfs|squashfs)$ ]]
}

validate_host_and_roots() {
    local mode="$1" persistent_root="$2" project_root="$3" test_root="$4"
    [[ "$(uname -s)" == Linux ]] || return 1
    is_contained "$persistent_root" "$project_root" || return 1
    if [[ "$mode" == production ]]; then
        [[ "$(id -u)" == 0 && "$persistent_root" == "$DEFAULT_PERSISTENT_ROOT" &&
            "$project_root" == "$DEFAULT_PROJECT_ROOT" ]] || return 1
        ! grep -Eqi '(microsoft|wsl)' /proc/sys/kernel/osrelease 2>/dev/null || return 1
        validate_mount "$persistent_root"
    else
        [[ -n "$test_root" && "$test_root" != "$DEFAULT_PERSISTENT_ROOT" ]] || return 1
        is_contained "$test_root" "$persistent_root" || [[ "$test_root" == "$persistent_root" ]] || return 1
        is_contained "$test_root" "$project_root" || return 1
        ! is_contained "$test_root" "$DEFAULT_PERSISTENT_ROOT" || return 1
        ! is_contained "$DEFAULT_PERSISTENT_ROOT" "$test_root" || return 1
    fi
}

validate_attempt_path() {
    local project="$1" attempt="$2" parent expected_parent
    expected_parent="$(canonical_existing "$project/state/attempts/gpu")" || return 1
    parent="$(canonical_existing "$(dirname -- "$attempt")")" || return 1
    [[ "$parent" == "$expected_parent" && -d "$attempt" && ! -L "$attempt" ]] || return 1
    [[ "$(basename -- "$attempt")" =~ ^[0-9]{8}T[0-9]{6}Z-[0-9]+-[0-9]+$ ]]
}

validate_checkout() {
    local project="$1" expected_commit="$2" expected_tree="$3"
    local checkout="$project/checkout" actual_commit actual_tree status
    [[ -d "$checkout/.git" && ! -L "$checkout" ]] || return 1
    actual_commit="$(git -C "$checkout" rev-parse HEAD)" || return 1
    [[ "$actual_commit" == "$expected_commit" ]] || return 1
    if git -C "$checkout" symbolic-ref -q HEAD >/dev/null 2>&1; then return 1; fi
    status="$(git -C "$checkout" status --porcelain=v1 --untracked-files=all)" || return 1
    [[ -z "$status" ]] || return 1
    actual_tree="$(tree_sha256 "$checkout")" || return 1
    [[ "$actual_tree" == "$expected_tree" ]]
}

validate_cpu_receipt() {
    local project="$1" runner="$2" runner_digest="$3" receipt="$4"
    local receipt_digest="$5" expected_uid="$6" git_commit="$7" cpu_handoff="$8"
    local directory expected_directory sidecar listing expected_listing value path digest
    expected_directory="$project/manifests/qwen-native-ar-eval-only-cpu/$runner_digest"
    directory="$(canonical_existing "$(dirname -- "$receipt")")" || return 1
    [[ "$receipt" == "$expected_directory/receipt.env" && "$directory" == "$expected_directory" &&
        -d "$directory" && ! -L "$directory" ]] || return 1
    for path in "$receipt" "$receipt.sha256" "$directory/exit-code" \
        "$directory/terminal" "$directory/.success"; do
        validate_protected_regular "$path" "$expected_uid" || return 1
    done
    [[ ! -e "$directory/.failed" && ! -L "$directory/.failed" &&
        ! -e "$directory/.running" && ! -L "$directory/.running" &&
        ! -e "$directory/.starting" && ! -L "$directory/.starting" &&
        "$(tr -d '\r\n' <"$directory/exit-code")" == 0 &&
        "$(tr -d '\r\n' <"$directory/terminal")" == success ]] || return 1
    listing="$(find "$directory" -mindepth 1 -maxdepth 1 -printf '%f\n' | LC_ALL=C sort)" || return 1
    expected_listing="$(printf '%s\n' .success exit-code receipt.env receipt.env.sha256 terminal | LC_ALL=C sort)"
    [[ "$listing" == "$expected_listing" ]] || return 1
    digest="$(file_sha256 "$receipt")" || return 1
    [[ "$digest" == "$receipt_digest" ]] || return 1
    sidecar="$(tr -d '\r\n' <"$receipt.sha256")"
    [[ "$sidecar" == "$receipt_digest  ${receipt#"$project/"}" ]] || return 1
    [[ "$(env_single_value "$receipt" schema)" == "$AR_CPU_CONTRACT" &&
        "$(env_single_value "$receipt" cpu_prepare_only)" == true &&
        "$(env_single_value "$receipt" gpu_started)" == false &&
        "$(env_single_value "$receipt" training_executed)" == false &&
        "$(env_single_value "$receipt" gpu_evaluation_authorized)" == true &&
        "$(env_single_value "$receipt" runner)" == "$runner" &&
        "$(env_single_value "$receipt" runner_sha256)" == "$runner_digest" &&
        "$(env_single_value "$receipt" checkout_commit)" == "$git_commit" &&
        "$(env_single_value "$receipt" cpu_handoff_sha256)" == "$cpu_handoff" ]] || return 1
    for value in base_model_sha256 r60_evidence_sha256 r60_checkpoint_sha256 \
        data_manifest_sha256; do
        [[ "$(env_single_value "$receipt" "$value")" =~ ^[0-9a-f]{64}$ ]] || return 1
    done
    for value in base_model r60_checkpoint data_manifest r60_evidence; do
        path="$(env_single_value "$receipt" "$value")" || return 1
        is_contained "$project" "$path" || return 1
        [[ "$(canonical_existing "$path")" == "$path" ]] || return 1
    done
    path="$(env_single_value "$receipt" r60_evidence)" || return 1
    validate_small_marker "$path" "$expected_uid" || return 1
    value="$(tr -d '\r\n' <"$path")"
    [[ "$value" == "$(env_single_value "$receipt" r60_evidence_sha256)" ]]
}

validate_test_mode_setting() {
    [[ -z "${SEARCH_R1_AR_WATCHDOG_TEST_MODE:-}" ||
        "${SEARCH_R1_AR_WATCHDOG_TEST_MODE:-}" == 1 ]]
}

load_capability() {
    local capability="$1" expected_uid key value extra
    local -a required=(
        schema mode authorized capability_path project_root persistent_root test_root
        attempt lock_file latest_gpu phase_log entrypoint runner_path runner_sha256 cpu_receipt
        cpu_receipt_sha256 git_marker git_commit tree_marker tree_digest cpu_marker
        cpu_handoff watchdog_script watchdog_script_sha256 shutdown_binary
        shutdown_binary_sha256 launch_nonce wait_timeout_seconds test_event_log
        test_backend_rc lock_wait_seconds dry_run
    )
    [[ -z "$CAPABILITY_PATH" || "$capability" == "$CAPABILITY_PATH" ]] || return 1
    if [[ "${SEARCH_R1_AR_WATCHDOG_TEST_MODE:-}" == 1 ]]; then expected_uid="$(id -u)"; else expected_uid=0; fi
    validate_protected_regular "$capability" "$expected_uid" || return 1
    [[ "$(stat -c '%a' -- "$capability")" == 600 ]] || return 1
    if [[ -n "$CAPABILITY_DIGEST" && "$(file_sha256 "$capability")" != "$CAPABILITY_DIGEST" ]]; then
        return 1
    fi
    CAP=()
    while IFS=$'\t' read -r key value extra; do
        [[ -n "$key" && -n "$value" && -z "$extra" && ! ${CAP[$key]+present} ]] || return 1
        CAP[$key]="$value"
    done <"$capability"
    ((${#CAP[@]} == ${#required[@]})) || return 1
    for key in "${required[@]}"; do [[ ${CAP[$key]+present} ]] || return 1; done
    [[ "${CAP[schema]}" == "$AR_WATCHDOG_SCHEMA" && "${CAP[authorized]}" == yes &&
        "${CAP[capability_path]}" == "$capability" &&
        "${CAP[runner_sha256]}" =~ ^[0-9a-f]{64}$ &&
        "${CAP[cpu_receipt_sha256]}" =~ ^[0-9a-f]{64}$ &&
        "${CAP[watchdog_script_sha256]}" =~ ^[0-9a-f]{64}$ &&
        "${CAP[shutdown_binary_sha256]}" =~ ^[0-9a-f]{64}$ &&
        "${CAP[git_commit]}" =~ ^[0-9a-f]{40}$ &&
        "${CAP[tree_digest]}" =~ ^[0-9a-f]{64}$ &&
        "${CAP[cpu_handoff]}" =~ ^[0-9a-f]{64}$ &&
        "${CAP[wait_timeout_seconds]}" =~ ^[1-9][0-9]*$ &&
        "${CAP[wait_timeout_seconds]}" -le 604800 &&
        "${CAP[lock_wait_seconds]}" =~ ^[1-9][0-9]*$ &&
        "${CAP[lock_wait_seconds]}" -le 600 &&
        "${CAP[dry_run]}" =~ ^(true|false)$ ]] || return 1
    CAPABILITY_PATH="$capability"
    if [[ -z "$CAPABILITY_DIGEST" ]]; then CAPABILITY_DIGEST="$(file_sha256 "$capability")" || return 1; fi
}

validate_mode_contract() {
    if [[ "${CAP[mode]}" == test ]]; then
        [[ "${SEARCH_R1_AR_WATCHDOG_TEST_MODE:-}" == 1 &&
            -n "${SEARCH_R1_AR_WATCHDOG_TEST_ROOT:-}" &&
            "$(canonical_existing "$SEARCH_R1_AR_WATCHDOG_TEST_ROOT")" == "${CAP[test_root]}" &&
            "$(canonical_existing "$SEARCH_R1_AR_WATCHDOG_TEST_EVENT_LOG")" == "${CAP[test_event_log]}" &&
            "$(canonical_existing "$SEARCH_R1_AR_WATCHDOG_TEST_SHUTDOWN_BINARY")" == "${CAP[shutdown_binary]}" ]] || return 1
    else
        [[ "${CAP[mode]}" == production && -z "${SEARCH_R1_AR_WATCHDOG_TEST_MODE:-}" &&
            "${CAP[test_root]}" == - && "${CAP[test_event_log]}" == - &&
            "${CAP[test_backend_rc]}" == - ]]
    fi
}

append_test_event() {
    [[ "${CAP[mode]}" == test ]] || return 0
    printf '%s\n' "$1" >>"${CAP[test_event_log]}"
}

publish_state() {
    local name="$1" content="$2" path
    path="${CAP[attempt]}/$name"
    atomic_publish "$path" "$content" || return 1
    append_test_event "state:$name"
}

publish_skipped() {
    local reason="$1"
    printf 'A/R shutdown skipped: %s\n' "$reason" >&2
    if [[ ! -e "${CAP[attempt]}/shutdown-skipped" && ! -L "${CAP[attempt]}/shutdown-skipped" ]]; then
        publish_state shutdown-skipped "at=$(utc_now)"$'\n'"reason=$reason"$'\n' || true
    fi
}

validate_worker_admission() {
    local expected_uid pid admitted
    if [[ "${CAP[mode]}" == production ]]; then expected_uid=0; else expected_uid="$(id -u)"; fi
    validate_small_marker "${CAP[attempt]}/$WATCHDOG_PID_NAME" "$expected_uid" || return 1
    validate_small_marker "${CAP[attempt]}/$WATCHDOG_ADMITTED_NAME" "$expected_uid" || return 1
    pid="$(tr -d '\r\n' <"${CAP[attempt]}/$WATCHDOG_PID_NAME")"
    admitted="$(tr -d '\r\n' <"${CAP[attempt]}/$WATCHDOG_ADMITTED_NAME")"
    [[ "$pid" == "$$" && "$admitted" == "${CAP[launch_nonce]}" ]]
}

wait_for_worker_admission() {
    local deadline=$(( $(date +%s) + 30 ))
    while (( $(date +%s) <= deadline )); do
        validate_worker_admission && return 0
        sleep 1
    done
    return 1
}

validate_terminal_evidence() {
    local attempt="${CAP[attempt]}" expected_uid marker_count=0 line_count sentinel_count
    local exit_file="$attempt/exit-code" terminal_file="$attempt/terminal"
    if [[ "${CAP[mode]}" == production ]]; then expected_uid=0; else expected_uid="$(id -u)"; fi
    validate_protected_regular "${CAP[phase_log]}" "$expected_uid" || return 1
    for path in "$exit_file" "$terminal_file"; do
        validate_small_marker "$path" "$expected_uid" || return 1
    done
    [[ ! -e "$attempt/.starting" && ! -L "$attempt/.starting" &&
        ! -e "$attempt/.running" && ! -L "$attempt/.running" ]] || return 1
    validate_protected_regular "$attempt/.success" "$expected_uid" && marker_count=$((marker_count + 1))
    validate_protected_regular "$attempt/.failed" "$expected_uid" && marker_count=$((marker_count + 1))
    [[ "$marker_count" == 1 ]] || return 1
    WORK_RC="$(tr -d '\r\n' <"$exit_file")"
    WORK_STATE="$(tr -d '\r\n' <"$terminal_file")"
    [[ "$WORK_RC" =~ ^(0|[1-9][0-9]*)$ && "$WORK_RC" -le 255 ]] || return 1
    if [[ "$WORK_RC" == 0 ]]; then
        [[ "$WORK_STATE" == success && -f "$attempt/.success" && ! -L "$attempt/.success" &&
            ! -e "$attempt/.failed" && ! -L "$attempt/.failed" ]] || return 1
    else
        [[ "$WORK_STATE" == failed && -f "$attempt/.failed" && ! -L "$attempt/.failed" &&
            ! -e "$attempt/.success" && ! -L "$attempt/.success" ]] || return 1
    fi
    line_count="$(grep -Fxc -- "$TERMINAL_SENTINEL_PREFIX state=$WORK_STATE exit_code=$WORK_RC" "${CAP[phase_log]}" || true)"
    sentinel_count="$(grep -Ec '^AUTODL_PHASE_TERMINAL state=(success|failed) exit_code=(0|[1-9][0-9]*)$' "${CAP[phase_log]}" || true)"
    [[ "$line_count" == 1 && "$sentinel_count" == 1 &&
        "$(tail -n 1 -- "${CAP[phase_log]}")" == \
            "$TERMINAL_SENTINEL_PREFIX state=$WORK_STATE exit_code=$WORK_RC" ]]
}

wait_for_terminal_evidence() {
    local deadline=$(( $(date +%s) + CAP[wait_timeout_seconds] )) attempt="${CAP[attempt]}"
    while (( $(date +%s) <= deadline )); do
        if [[ -e "$attempt/.starting" || -L "$attempt/.starting" ||
              -e "$attempt/.running" || -L "$attempt/.running" ]]; then
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

validate_static_identity() {
    local expected_uid project attempt latest runner receipt script shutdown path
    local git_value tree_value cpu_value
    [[ "$(file_sha256 "$CAPABILITY_PATH")" == "$CAPABILITY_DIGEST" ]] || return 1
    load_capability "$CAPABILITY_PATH" || return 1
    validate_mode_contract || return 1
    if [[ "${CAP[mode]}" == production ]]; then expected_uid=0; else expected_uid="$(id -u)"; fi
    project="$(canonical_existing "${CAP[project_root]}")" || return 1
    attempt="$(canonical_existing "${CAP[attempt]}")" || return 1
    latest="$(canonical_existing "${CAP[latest_gpu]}")" || return 1
    runner="$(canonical_existing "${CAP[runner_path]}")" || return 1
    receipt="$(canonical_existing "${CAP[cpu_receipt]}")" || return 1
    script="$(canonical_existing "${CAP[watchdog_script]}")" || return 1
    shutdown="$(canonical_existing "${CAP[shutdown_binary]}")" || return 1
    [[ "$project" == "${CAP[project_root]}" && "$attempt" == "${CAP[attempt]}" &&
        "$runner" == "$project/$AR_RUNNER_RELATIVE" &&
        "$script" == "$project/$AR_WATCHDOG_RELATIVE" && "$SCRIPT_PATH" == "$script" &&
        "$latest" == "$project/state/latest/gpu" &&
        "${CAP[lock_file]}" == "$project/state/phase.lock" &&
        "${CAP[phase_log]}" == "$attempt/phase.log" &&
        "${CAP[entrypoint]}" == "$attempt/entrypoint" ]] || return 1
    validate_host_and_roots "${CAP[mode]}" "${CAP[persistent_root]}" "$project" "${CAP[test_root]}" || return 1
    validate_attempt_path "$project" "$attempt" || return 1
    for path in "${CAP[lock_file]}" "$latest" "${CAP[phase_log]}" \
        "${CAP[entrypoint]}" "${CAP[git_marker]}" "${CAP[tree_marker]}" \
        "${CAP[cpu_marker]}"; do
        validate_protected_regular "$path" "$expected_uid" || return 1
    done
    [[ "$(tr -d '\r\n' <"$latest")" == "$attempt" ]] || return 1
    validate_small_marker "${CAP[entrypoint]}" "$expected_uid" || return 1
    [[ "$(tr -d '\r\n' <"${CAP[entrypoint]}")" == "$runner" ]] || return 1
    validate_protected_executable "$runner" "$expected_uid" || return 1
    validate_protected_executable "$script" "$expected_uid" || return 1
    validate_protected_executable "$shutdown" "$expected_uid" || return 1
    [[ "$(file_sha256 "$runner")" == "${CAP[runner_sha256]}" &&
        "$(file_sha256 "$script")" == "${CAP[watchdog_script_sha256]}" &&
        "$(file_sha256 "$shutdown")" == "${CAP[shutdown_binary_sha256]}" ]] || return 1
    [[ "${CAP[git_marker]}" == "$project/manifests/git.ok" &&
        "${CAP[tree_marker]}" == "$project/manifests/checkout-tree.sha256" &&
        "${CAP[cpu_marker]}" == "$project/manifests/cpu.ok" ]] || return 1
    git_value="$(tr -d '\r\n' <"${CAP[git_marker]}")"
    tree_value="$(tr -d '\r\n' <"${CAP[tree_marker]}")"
    cpu_value="$(tr -d '\r\n' <"${CAP[cpu_marker]}")"
    [[ "$git_value" == "${CAP[git_commit]}" && "$tree_value" == "${CAP[tree_digest]}" &&
        "$cpu_value" == "${CAP[cpu_handoff]}" ]] || return 1
    validate_checkout "$project" "$git_value" "$tree_value" || return 1
    validate_cpu_receipt "$project" "$runner" "${CAP[runner_sha256]}" "$receipt" \
        "${CAP[cpu_receipt_sha256]}" "$expected_uid" "$git_value" "$cpu_value"
}

require_evidence_relative() {
    [[ ${EVIDENCE_SEEN[$1]+present} ]]
}

validate_result_semantics() {
    local results="$1" contract="$results/contract.env" lineage="$results/lineage.tsv"
    local index="$results/run-index.tsv" receipt="${CAP[cpu_receipt]}" project="${CAP[project_root]}"
    local base_model base_digest r60 r60_digest data_digest expected_checkpoint expected_digest
    local expected_predecessor stage role variant rows run run_env relative
    local -a expected_stages=(A-VAL-EVAL R-VAL-EVAL A-NQ-TEST-EVAL R-NQ-TEST-EVAL A-MULTIHOP-EVAL R-MULTIHOP-EVAL)
    local -a expected_roles=(parent_val_eval reproduced_val_eval parent_nq_test_eval reproduced_nq_test_eval parent_multihop_eval reproduced_multihop_eval)
    local -a expected_variants=(qwen_native_a_val qwen_native_r_val qwen_native_a_nq_test qwen_native_r_nq_test qwen_native_a_multihop qwen_native_r_multihop)
    local -a expected_rows=(128 128 128 128 256 256) lineage_lines index_lines
    [[ "$(env_single_value "$contract" schema)" == "$AR_RESULT_CONTRACT" &&
        "$(env_single_value "$contract" stage)" == ar_eval_only &&
        "$(env_single_value "$contract" stage_order)" == A-VAL-EVAL,R-VAL-EVAL,A-NQ-TEST-EVAL,R-NQ-TEST-EVAL,A-MULTIHOP-EVAL,R-MULTIHOP-EVAL &&
        "$(env_single_value "$contract" training_executed)" == false &&
        "$(env_single_value "$contract" checkout_commit)" == "${CAP[git_commit]}" &&
        "$(env_single_value "$contract" cpu_handoff_sha256)" == "${CAP[cpu_handoff]}" &&
        "$(env_single_value "$contract" runner_sha256)" == "${CAP[runner_sha256]}" &&
        "$(env_single_value "$contract" cpu_receipt)" == "${CAP[cpu_receipt]}" &&
        "$(env_single_value "$contract" cpu_receipt_sha256)" == "${CAP[cpu_receipt_sha256]}" &&
        "$(env_single_value "$contract" eval_group_size)" == 1 &&
        "$(env_single_value "$contract" decoding)" == greedy &&
        "$(env_single_value "$contract" seed)" == 42 ]] || return 1
    base_model="$(env_single_value "$receipt" base_model)" || return 1
    base_digest="$(env_single_value "$receipt" base_model_sha256)" || return 1
    r60="$(env_single_value "$receipt" r60_checkpoint)" || return 1
    r60_digest="$(env_single_value "$receipt" r60_checkpoint_sha256)" || return 1
    data_digest="$(env_single_value "$receipt" data_manifest_sha256)" || return 1
    [[ "$(env_single_value "$contract" base_model)" == "$base_model" &&
        "$(env_single_value "$contract" base_model_sha256)" == "$base_digest" &&
        "$(env_single_value "$contract" r60_checkpoint)" == "$r60" &&
        "$(env_single_value "$contract" r60_checkpoint_sha256)" == "$r60_digest" ]] || return 1
    [[ "$(head -n 1 -- "$lineage")" == $'stage\trole\trun_dir\tcheckpoint\tcheckpoint_digest\tparent_checkpoint\tparent_checkpoint_digest\tcheckout_commit\tcpu_handoff_digest\tdata_manifest_sha256\tresolved_config_sha256\ttrace_sha256\ttrace_manifest_sha256\trun_contract_sha256\tpredecessor_evidence_sha256' ]] || return 1
    [[ "$(head -n 1 -- "$index")" == $'stage\trole\trun_dir' ]] || return 1
    mapfile -t lineage_lines < <(tail -n +2 -- "$lineage")
    mapfile -t index_lines < <(tail -n +2 -- "$index")
    ((${#lineage_lines[@]} == 6 && ${#index_lines[@]} == 6)) || return 1
    for position in "${!expected_stages[@]}"; do
        local checkpoint checkpoint_digest parent_checkpoint parent_digest checkout handoff data config_digest trace_digest trace_manifest run_contract predecessor extra
        IFS=$'\t' read -r stage role run checkpoint checkpoint_digest parent_checkpoint parent_digest checkout handoff data config_digest trace_digest trace_manifest run_contract predecessor extra <<<"${lineage_lines[$position]}"
        [[ -z "$extra" && "$stage" == "${expected_stages[$position]}" &&
            "$role" == "${expected_roles[$position]}" &&
            "$checkout" == "${CAP[git_commit]}" && "$handoff" == "${CAP[cpu_handoff]}" &&
            "$data" == "$data_digest" && "$run_contract" == - &&
            "$config_digest" =~ ^[0-9a-f]{64}$ && "$trace_digest" =~ ^[0-9a-f]{64}$ &&
            "$trace_manifest" =~ ^[0-9a-f]{64}$ ]] || return 1
        variant="${expected_variants[$position]}"
        rows="${expected_rows[$position]}"
        if [[ "$stage" == A-* ]]; then
            expected_checkpoint="$base_model"
            expected_digest="$base_digest"
            expected_predecessor="${CAP[cpu_receipt_sha256]}"
        else
            expected_checkpoint="$r60"
            expected_digest="$r60_digest"
            expected_predecessor="$(env_single_value "$receipt" r60_evidence_sha256)" || return 1
        fi
        [[ "$checkpoint" == "$expected_checkpoint" && "$parent_checkpoint" == "$expected_checkpoint" &&
            "$checkpoint_digest" == "$expected_digest" && "$parent_digest" == "$expected_digest" &&
            "$predecessor" == "$expected_predecessor" ]] || return 1
        [[ "$(canonical_existing "$run")" == "$run" &&
            "$(canonical_existing "$(dirname -- "$run")")" == "$project/runs/eval/$variant/attempts" ]] || return 1
        IFS=$'\t' read -r index_stage index_role index_run index_extra <<<"${index_lines[$position]}"
        [[ -z "$index_extra" && "$index_stage" == "$stage" && "$index_role" == "$role" &&
            "$index_run" == "$run" ]] || return 1
        run_env="$run/run.env"
        [[ "$(env_single_value "$run_env" job_mode)" == eval &&
            "$(env_single_value "$run_env" variant)" == "$variant" &&
            "$(env_single_value "$run_env" train_steps)" == 0 &&
            "$(env_single_value "$run_env" input_model)" == "$expected_checkpoint" &&
            "$(env_single_value "$run_env" eval_expected_rows)" == "$rows" &&
            "$(env_single_value "$run_env" eval_group_size)" == 1 &&
            "$(env_single_value "$run_env" timed_out)" == false &&
            "$(tr -d '\r\n' <"$run/terminal")" == success &&
            "$(tr -d '\r\n' <"$run/exit-code")" == 0 && -f "$run/.success" && ! -L "$run/.success" ]] || return 1
        [[ "$(file_sha256 "$run/resolved-config.yaml")" == "$config_digest" &&
            "$(file_sha256 "$run/traces/eval_predictions.jsonl")" == "$trace_digest" &&
            "$(file_sha256 "$run/traces/eval_predictions.manifest.json")" == "$trace_manifest" ]] || return 1
        for path in "$run/run.env" "$run/terminal" "$run/exit-code" "$run/.success" \
            "$run/resolved-config.yaml" "$run/traces/eval_predictions.jsonl" \
            "$run/traces/eval_predictions.manifest.json"; do
            relative="${path#"$project/"}"
            require_evidence_relative "$relative" || return 1
        done
    done
}

validate_success_artifacts() {
    local project="${CAP[project_root]}" attempt="${CAP[attempt]}" expected_uid
    local contract_file="$attempt/result-contract" root_file="$attempt/result-root"
    local marker_file="$attempt/evidence-marker" digest_file="$attempt/evidence-digest"
    local results marker evidence evidence_digest marker_digest expected_digest relative path line
    local previous='' count=0 required
    if [[ "${CAP[mode]}" == production ]]; then expected_uid=0; else expected_uid="$(id -u)"; fi
    for path in "$contract_file" "$root_file" "$marker_file" "$digest_file"; do
        validate_small_marker "$path" "$expected_uid" || return 1
    done
    [[ "$(tr -d '\r\n' <"$contract_file")" == "$AR_RESULT_CONTRACT" ]] || return 1
    results="$(canonical_existing "$(tr -d '\r\n' <"$root_file")")" || return 1
    marker="$(canonical_existing "$(tr -d '\r\n' <"$marker_file")")" || return 1
    [[ "$results" == "$project/runs/qwen-native-training/attempts/$(basename -- "$attempt")" &&
        -d "$results" && ! -L "$results" &&
        "$marker" == "$project/manifests/$AR_RESULT_NAMESPACE/$(basename -- "$attempt").ok" ]] || return 1
    validate_small_marker "$marker" "$expected_uid" || return 1
    evidence="$results/evidence.sha256"
    validate_protected_regular "$evidence" "$expected_uid" || return 1
    EVIDENCE_SEEN=()
    while IFS= read -r line; do
        [[ "$line" =~ ^([0-9a-f]{64})\ \ ([A-Za-z0-9._/-]+)$ ]] || return 1
        expected_digest="${BASH_REMATCH[1]}"
        relative="${BASH_REMATCH[2]}"
        [[ "$relative" != /* && "$relative" != *'//'*
            && "/$relative/" != *'/../'* && "$relative" > "$previous" &&
            ! ${EVIDENCE_SEEN[$relative]+present} ]] || return 1
        previous="$relative"
        path="$project/$relative"
        [[ "$(canonical_existing "$path")" == "$path" ]] || return 1
        validate_protected_regular "$path" "$expected_uid" || return 1
        [[ "$(file_sha256 "$path")" == "$expected_digest" ]] || return 1
        EVIDENCE_SEEN[$relative]=1
        count=$((count + 1))
        sync_required "$path" || return 1
    done <"$evidence"
    ((count >= 40)) || return 1
    local -a required_files=(
        contract.env lineage.tsv run-index.tsv
        paired-ar-val/summary.json paired-ar-val/summary.md paired-ar-val/paired_results.csv
        paired-ar-val/correct_questions.csv paired-ar-val/wrong_questions.csv paired-ar-val/search_transition.csv
        paired-ar-nq_test/summary.json paired-ar-nq_test/summary.md paired-ar-nq_test/paired_results.csv
        paired-ar-nq_test/correct_questions.csv paired-ar-nq_test/wrong_questions.csv paired-ar-nq_test/search_transition.csv
        paired-ar-multihop/summary.json paired-ar-multihop/summary.md paired-ar-multihop/paired_results.csv
        paired-ar-multihop/correct_questions.csv paired-ar-multihop/wrong_questions.csv paired-ar-multihop/search_transition.csv
    )
    for required in "${required_files[@]}"; do
        relative="runs/qwen-native-training/attempts/$(basename -- "$attempt")/$required"
        require_evidence_relative "$relative" || return 1
        [[ -s "$project/$relative" ]] || return 1
    done
    require_evidence_relative "$AR_RUNNER_RELATIVE" || return 1
    require_evidence_relative "${CAP[cpu_receipt]#"$project/"}" || return 1
    require_evidence_relative "${CAP[cpu_receipt]#"$project/"}.sha256" || return 1
    validate_result_semantics "$results" || return 1
    evidence_digest="$(file_sha256 "$evidence")" || return 1
    marker_digest="$(tr -d '\r\n' <"$marker")"
    [[ "$marker_digest" == "$evidence_digest" &&
        "$(tr -d '\r\n' <"$digest_file")" == "$evidence_digest" ]] || return 1
    sync_required "$evidence" || return 1
    sync_required "$results" || return 1
    sync_required "$marker" || return 1
    sync_required "$(dirname -- "$marker")" || return 1
    sync_required "$attempt" || return 1
    RESULTS_DIGEST="$evidence_digest"
}

validate_failure_artifacts() {
    local attempt="${CAP[attempt]}" expected_marker
    expected_marker="${CAP[project_root]}/manifests/$AR_RESULT_NAMESPACE/$(basename -- "$attempt").ok"
    for path in "$attempt/result-contract" "$attempt/result-root" "$attempt/evidence-marker" \
        "$attempt/evidence-digest" "$expected_marker"; do
        [[ ! -e "$path" && ! -L "$path" ]] || return 1
    done
    RESULTS_DIGEST=not-required
}

validate_artifacts() {
    if [[ "$WORK_STATE" == success ]]; then
        validate_success_artifacts
    else
        validate_failure_artifacts
    fi
}

lock_is_held() {
    local fd_path fd_identity path_identity
    [[ -n "$LOCK_FD" ]] || return 1
    fd_path="/proc/$$/fd/$LOCK_FD"
    [[ "$(readlink -f -- "$fd_path")" == "${CAP[lock_file]}" ]] || return 1
    fd_identity="$(stat -Lc '%d:%i' -- "$fd_path")" || return 1
    path_identity="$(stat -c '%d:%i' -- "${CAP[lock_file]}")" || return 1
    [[ "$fd_identity" == "$path_identity" ]]
}

acquire_phase_lock() {
    local deadline=$(( $(date +%s) + CAP[lock_wait_seconds] ))
    is_regular_file "${CAP[lock_file]}" || return 1
    exec {LOCK_FD}<>"${CAP[lock_file]}"
    while (( $(date +%s) <= deadline )); do
        if flock -n "$LOCK_FD"; then
            lock_is_held
            return
        fi
        sleep 1
    done
    return 1
}

verify_authorization() {
    validate_static_identity || return 1
    validate_worker_admission || return 1
    lock_is_held || return 1
    validate_terminal_evidence || return 1
    sync_required "${CAP[phase_log]}" || return 1
    sync_required "${CAP[attempt]}" || return 1
    validate_artifacts
}

dispatch_backend() {
    local rc
    if [[ "${CAP[mode]}" == test ]]; then
        append_test_event "backend:${CAP[shutdown_binary]} argc=0"
        return "${CAP[test_backend_rc]}"
    fi
    set +e
    /usr/bin/env -i PATH=/usr/sbin:/usr/bin:/sbin:/bin \
        /usr/bin/timeout --signal=TERM --kill-after=5s 30s \
        "${CAP[shutdown_binary]}"
    rc=$?
    set -e
    return "$rc"
}

watchdog_worker() {
    local capability="$1" safe_rc safe_state safe_results backend_rc
    validate_test_mode_setting || return 1
    load_capability "$capability" || return 1
    validate_mode_contract || return 1
    wait_for_worker_admission || {
        printf 'A/R watchdog admission is not durable; keeping the instance running.\n' >&2
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
    if ! verify_authorization || [[ "$WORK_RC" != "$safe_rc" || "$WORK_STATE" != "$safe_state" ||
        "$RESULTS_DIGEST" != "$safe_results" ]]; then
        publish_skipped authorization-changed-after-shutdown-safe
        return 0
    fi
    publish_state shutdown-requested \
        "at=$(utc_now)"$'\n'"backend=${CAP[shutdown_binary]}"$'\n'"backend_kind=$([[ ${CAP[mode]} == test ]] && printf test || printf autodl-guest)"$'\n'"work_exit_code=$safe_rc"$'\n'"provider_control_plane_confirmed=false"$'\n' || return 1
    if dispatch_backend; then
        publish_state shutdown-dispatched \
            "at=$(utc_now)"$'\n'"backend=${CAP[shutdown_binary]}"$'\n'"backend_exit_code=0"$'\n'"provider_control_plane_confirmed=false"$'\n' || return 1
        printf 'A/R shutdown backend dispatched; confirm stopped state and billing in AutoDL.\n'
        return 0
    else
        backend_rc=$?
    fi
    publish_state shutdown-failed \
        "at=$(utc_now)"$'\n'"backend=${CAP[shutdown_binary]}"$'\n'"backend_exit_code=$backend_rc"$'\n'"work_exit_code=$safe_rc"$'\n' || return 1
    printf 'A/R shutdown backend failed with exit code %s; instance remains running.\n' "$backend_rc" >&2
    return 1
}

create_capability_and_launch() {
    local attempt_input="$1" dry_run="$2" foreground="$3"
    local mode project persistent test_root=- expected_uid attempt runner runner_digest receipt receipt_digest
    local lock latest phase_log entrypoint shutdown shutdown_digest script_digest capability log pid_file admitted_file
    local git_marker git_commit tree_marker tree_digest cpu_marker cpu_handoff nonce wait_timeout worker_pid
    local lock_wait test_event=- test_backend=- drift_setting path capability_content
    validate_test_mode_setting || die 'invalid SEARCH_R1_AR_WATCHDOG_TEST_MODE'
    if [[ "${SEARCH_R1_AR_WATCHDOG_TEST_MODE:-}" == 1 ]]; then
        mode=test
        test_root="$(canonical_existing "${SEARCH_R1_AR_WATCHDOG_TEST_ROOT:?set test root}")" || die 'invalid test root'
        persistent="$(canonical_existing "${AUTODL_PERSISTENT_ROOT:-$test_root}")" || die 'invalid test persistent root'
        project="$(canonical_existing "${AUTODL_ROOT:?set AUTODL_ROOT in test mode}")" || die 'invalid test project'
        test_event="$(canonical_existing "${SEARCH_R1_AR_WATCHDOG_TEST_EVENT_LOG:?set test event log}")" || die 'invalid test event log'
        shutdown="$(canonical_existing "${SEARCH_R1_AR_WATCHDOG_TEST_SHUTDOWN_BINARY:?set fake shutdown}")" || die 'invalid fake shutdown'
        test_backend="${SEARCH_R1_AR_WATCHDOG_TEST_BACKEND_RC:-0}"
        [[ "$test_backend" =~ ^(0|[1-9][0-9]*)$ && "$test_backend" -le 255 ]] || die 'invalid test backend rc'
        expected_uid="$(id -u)"
    else
        mode=production
        persistent="$(canonical_existing "${AUTODL_PERSISTENT_ROOT:-$DEFAULT_PERSISTENT_ROOT}")" || die 'persistent root unavailable'
        project="$(canonical_existing "${AUTODL_ROOT:-$DEFAULT_PROJECT_ROOT}")" || die 'project root unavailable'
        shutdown="$(canonical_existing "$PRODUCTION_SHUTDOWN_BINARY")" || die 'AutoDL shutdown unavailable'
        expected_uid=0
    fi
    validate_host_and_roots "$mode" "$persistent" "$project" "$test_root" || die 'host or root authorization failed'
    [[ "$SCRIPT_PATH" == "$project/$AR_WATCHDOG_RELATIVE" ]] || die 'run the deployed A/R watchdog'
    validate_protected_executable "$SCRIPT_PATH" "$expected_uid" || die 'unsafe watchdog file'
    attempt="$(canonical_existing "$attempt_input")" || die 'exact GPU attempt unavailable'
    [[ "$attempt" == "$attempt_input" ]] || die 'pass the canonical attempt path'
    validate_attempt_path "$project" "$attempt" || die 'invalid GPU attempt path'
    runner="$(canonical_existing "$project/$AR_RUNNER_RELATIVE")" || die 'A/R runner unavailable'
    [[ "$runner" == "$project/$AR_RUNNER_RELATIVE" ]] || die 'A/R runner path escaped'
    validate_protected_executable "$runner" "$expected_uid" || die 'unsafe A/R runner'
    runner_digest="$(file_sha256 "$runner")" || die 'cannot hash A/R runner'
    receipt="$project/manifests/qwen-native-ar-eval-only-cpu/$runner_digest/receipt.env"
    receipt="$(canonical_existing "$receipt")" || die 'exact A/R CPU receipt unavailable'
    receipt_digest="$(file_sha256 "$receipt")" || die 'cannot hash CPU receipt'
    lock="$project/state/phase.lock"
    latest="$project/state/latest/gpu"
    phase_log="$attempt/phase.log"
    entrypoint="$attempt/entrypoint"
    for path in "$lock" "$latest" "$phase_log" "$entrypoint"; do
        validate_protected_regular "$path" "$expected_uid" || die "unsafe required state: $path"
    done
    [[ "$(tr -d '\r\n' <"$latest")" == "$attempt" ]] || die 'attempt is no longer latest GPU attempt'
    validate_small_marker "$entrypoint" "$expected_uid" || die 'invalid outer entrypoint marker'
    [[ "$(tr -d '\r\n' <"$entrypoint")" == "$runner" ]] || die 'outer attempt did not launch the exact A/R runner'
    git_marker="$project/manifests/git.ok"
    tree_marker="$project/manifests/checkout-tree.sha256"
    cpu_marker="$project/manifests/cpu.ok"
    for path in "$git_marker" "$tree_marker" "$cpu_marker"; do
        validate_small_marker "$path" "$expected_uid" || die "invalid identity marker: $path"
    done
    git_commit="$(tr -d '\r\n' <"$git_marker")"
    tree_digest="$(tr -d '\r\n' <"$tree_marker")"
    cpu_handoff="$(tr -d '\r\n' <"$cpu_marker")"
    [[ "$git_commit" =~ ^[0-9a-f]{40}$ && "$tree_digest" =~ ^[0-9a-f]{64}$ &&
        "$cpu_handoff" =~ ^[0-9a-f]{64}$ ]] || die 'invalid project identity values'
    validate_checkout "$project" "$git_commit" "$tree_digest" || die 'checkout identity drifted'
    validate_cpu_receipt "$project" "$runner" "$runner_digest" "$receipt" "$receipt_digest" \
        "$expected_uid" "$git_commit" "$cpu_handoff" || die 'CPU receipt does not bind this runner and project'
    validate_protected_executable "$shutdown" "$expected_uid" || die 'unsafe shutdown backend'
    if [[ "$mode" == production && "$shutdown" != "$PRODUCTION_SHUTDOWN_BINARY" ]]; then
        die 'production backend must be /usr/bin/shutdown'
    fi
    shutdown_digest="$(file_sha256 "$shutdown")" || die 'cannot hash shutdown backend'
    script_digest="$(file_sha256 "$SCRIPT_PATH")" || die 'cannot hash watchdog'
    wait_timeout="${SEARCH_R1_AR_WATCHDOG_TIMEOUT_SECONDS:-604800}"
    [[ "$wait_timeout" =~ ^[1-9][0-9]*$ && "$wait_timeout" -le 604800 ]] || die 'watch timeout must be 1..604800 seconds'
    lock_wait="${SEARCH_R1_AR_WATCHDOG_LOCK_WAIT_SECONDS:-120}"
    [[ "$lock_wait" =~ ^[1-9][0-9]*$ && "$lock_wait" -le 600 ]] || die 'lock wait must be 1..600 seconds'
    nonce="$(printf '%s\n' "$(utc_now)-$$-$RANDOM-$RANDOM" | sha256sum | cut -d' ' -f1)"
    capability="$attempt/$CAPABILITY_NAME"
    log="$attempt/$WATCHDOG_LOG_NAME"
    pid_file="$attempt/$WATCHDOG_PID_NAME"
    admitted_file="$attempt/$WATCHDOG_ADMITTED_NAME"
    for path in "$capability" "$log" "$pid_file" "$admitted_file" \
        "$attempt/shutdown-capability.tsv" "$attempt/shutdown-watchdog-pid" \
        "$attempt/shutdown-watchdog-admitted" "$attempt/shutdown-safe" \
        "$attempt/shutdown-requested" "$attempt/shutdown-dispatched" \
        "$attempt/shutdown-failed" "$attempt/shutdown-skipped"; do
        [[ ! -e "$path" && ! -L "$path" ]] || die "attempt is already armed: $path"
    done
    (umask 077 && : >"$log")
    chmod 0600 "$log"
    sync_required "$log" || die 'cannot sync watchdog log'
    capability_content=\
"schema"$'\t'"$AR_WATCHDOG_SCHEMA"$'\n'\
"mode"$'\t'"$mode"$'\n'\
"authorized"$'\t'"yes"$'\n'\
"capability_path"$'\t'"$capability"$'\n'\
"project_root"$'\t'"$project"$'\n'\
"persistent_root"$'\t'"$persistent"$'\n'\
"test_root"$'\t'"$test_root"$'\n'\
"attempt"$'\t'"$attempt"$'\n'\
"lock_file"$'\t'"$lock"$'\n'\
"latest_gpu"$'\t'"$latest"$'\n'\
"phase_log"$'\t'"$phase_log"$'\n'\
"entrypoint"$'\t'"$entrypoint"$'\n'\
"runner_path"$'\t'"$runner"$'\n'\
"runner_sha256"$'\t'"$runner_digest"$'\n'\
"cpu_receipt"$'\t'"$receipt"$'\n'\
"cpu_receipt_sha256"$'\t'"$receipt_digest"$'\n'\
"git_marker"$'\t'"$git_marker"$'\n'\
"git_commit"$'\t'"$git_commit"$'\n'\
"tree_marker"$'\t'"$tree_marker"$'\n'\
"tree_digest"$'\t'"$tree_digest"$'\n'\
"cpu_marker"$'\t'"$cpu_marker"$'\n'\
"cpu_handoff"$'\t'"$cpu_handoff"$'\n'\
"watchdog_script"$'\t'"$SCRIPT_PATH"$'\n'\
"watchdog_script_sha256"$'\t'"$script_digest"$'\n'\
"shutdown_binary"$'\t'"$shutdown"$'\n'\
"shutdown_binary_sha256"$'\t'"$shutdown_digest"$'\n'\
"launch_nonce"$'\t'"$nonce"$'\n'\
"wait_timeout_seconds"$'\t'"$wait_timeout"$'\n'\
"test_event_log"$'\t'"$test_event"$'\n'\
"test_backend_rc"$'\t'"$test_backend"$'\n'\
"lock_wait_seconds"$'\t'"$lock_wait"$'\n'\
"dry_run"$'\t'"$dry_run"$'\n'
    atomic_publish "$capability" "$capability_content" || die 'cannot publish A/R shutdown capability'
    drift_setting="${SEARCH_R1_AR_WATCHDOG_TEST_DRIFT_AFTER_ARM:-}"
    if [[ -n "$drift_setting" ]]; then
        [[ "$mode" == test ]] || die 'test drift hook is test-only'
        case "$drift_setting" in
            runner) printf 'test-drift\n' >>"$runner" ;;
            receipt) printf 'test-drift\n' >>"$receipt" ;;
            entrypoint) printf 'test-drift\n' >>"$entrypoint" ;;
            *) die 'unknown test drift hook' ;;
        esac
    fi
    if [[ "$foreground" == true ]]; then
        [[ "$mode" == test ]] || die '--test-foreground is test-only'
        atomic_publish "$pid_file" "$$"$'\n' || die 'cannot publish watchdog PID'
        atomic_publish "$admitted_file" "$nonce"$'\n' || die 'cannot admit watchdog'
        watchdog_worker "$capability"
        return
    fi
    if [[ "$mode" == test ]]; then
        /usr/bin/nohup /usr/bin/setsid /usr/bin/env \
            SEARCH_R1_AR_WATCHDOG_TEST_MODE=1 \
            SEARCH_R1_AR_WATCHDOG_TEST_ROOT="$test_root" \
            SEARCH_R1_AR_WATCHDOG_TEST_EVENT_LOG="$test_event" \
            SEARCH_R1_AR_WATCHDOG_TEST_SHUTDOWN_BINARY="$shutdown" \
            SEARCH_R1_AR_WATCHDOG_TEST_BACKEND_RC="$test_backend" \
            /usr/bin/bash "$SCRIPT_PATH" --worker "$capability" >>"$log" 2>&1 </dev/null &
    else
        /usr/bin/nohup /usr/bin/setsid /usr/bin/env -i PATH=/usr/sbin:/usr/bin:/sbin:/bin \
            /usr/bin/bash "$SCRIPT_PATH" --worker "$capability" >>"$log" 2>&1 </dev/null &
    fi
    worker_pid=$!
    if ! atomic_publish "$pid_file" "$worker_pid"$'\n'; then
        kill -TERM "$worker_pid" 2>/dev/null || true
        wait "$worker_pid" 2>/dev/null || true
        die 'cannot publish watchdog PID'
    fi
    if ! atomic_publish "$admitted_file" "$nonce"$'\n'; then
        kill -TERM "$worker_pid" 2>/dev/null || true
        wait "$worker_pid" 2>/dev/null || true
        die 'cannot admit watchdog'
    fi
    printf 'Armed A/R shutdown watchdog for exact attempt: %s\n' "$attempt"
    printf 'Watchdog log: %s\n' "$log"
}

usage() {
    printf 'Usage: bash %s [--dry-run] EXACT_GPU_ATTEMPT\n' "$0" >&2
    printf 'Production launch (detached automatically):\n' >&2
    printf '  AUTODL_ROOT=%s bash %s EXACT_GPU_ATTEMPT\n' \
        "$DEFAULT_PROJECT_ROOT" "$DEFAULT_PROJECT_ROOT/$AR_WATCHDOG_RELATIVE" >&2
}

case "${1:-}" in
    --worker)
        [[ "$#" == 2 ]] || die 'worker requires one capability path'
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
