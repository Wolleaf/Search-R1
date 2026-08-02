#!/usr/bin/env bash
set -Eeuo pipefail

AUTODL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
CHECKOUT="$(cd -- "$AUTODL_DIR/../.." && pwd -P)"
ENTRYPOINT="$AUTODL_DIR/14_gpu_qwen_native_ar_eval_only.sh"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TEST_ROOT="$(mktemp -d)"
trap 'rm -rf -- "$TEST_ROOT"' EXIT

fail() {
    printf '%s\n' "$1" >&2
    exit 1
}

assert_eq() {
    local expected="$1" actual="$2" label="$3"
    [[ "$actual" == "$expected" ]] ||
        fail "$label mismatch: expected $expected, got $actual"
}

[[ -f "$ENTRYPOINT" && ! -L "$ENTRYPOINT" ]] ||
    fail "Missing A/R eval-only runner: $ENTRYPOINT"
bash -n "$ENTRYPOINT"

PROJECT_ROOT="$TEST_ROOT/project"
export AUTODL_ROOT="$PROJECT_ROOT"
export GPU_COUNT=2
export AUTODL_PRICE_PER_HOUR=5.76
export TRAIN_BATCH_SIZE=8
export MAX_RESPONSE_LENGTH=500
mkdir -p "$PROJECT_ROOT/envs/train/bin" \
    "$PROJECT_ROOT/models/Qwen3.5-2B" "$PROJECT_ROOT/operator"
ln -s "$CHECKOUT" "$PROJECT_ROOT/checkout"

# Source the same regular operator copies used in production. Merely sourcing
# the new runner must define its overrides without preparing CPU state or
# starting a paid job.
cp "$AUTODL_DIR/12_gpu_qwen_native_bc_only.sh" \
    "$PROJECT_ROOT/operator/12_gpu_qwen_native_bc_only.sh"
cp "$ENTRYPOINT" \
    "$PROJECT_ROOT/operator/14_gpu_qwen_native_ar_eval_only.sh"
chmod 0755 "$PROJECT_ROOT/operator/12_gpu_qwen_native_bc_only.sh" \
    "$PROJECT_ROOT/operator/14_gpu_qwen_native_ar_eval_only.sh"

CALLS="$TEST_ROOT/calls.log"
export CALLS
: >"$CALLS"
source "$PROJECT_ROOT/operator/14_gpu_qwen_native_ar_eval_only.sh"
[[ ! -s "$CALLS" ]] || fail 'Sourcing the A/R runner started work.'
assert_eq "$QWEN_NATIVE_PROTOCOL_GATE_EVIDENCE" \
    "$NATIVE_PROTOCOL_GATE_EVIDENCE" 'inherited protocol marker snapshot'
assert_eq "$QWEN_NATIVE_SMOKE_EVIDENCE" \
    "$NATIVE_SMOKE_EVIDENCE" 'inherited smoke marker snapshot'

for function_name in prepare_ar_cpu_receipt verify_ar_cpu_receipt \
        qwen_native_train_preflight qwen_native_main_pipeline run_ar_eval_job; do
    declare -F "$function_name" >/dev/null ||
        fail "Missing A/R runner function: $function_name"
done
assert_eq qwen-native-training-ar-eval-only-v1 "$AR_CONTRACT" \
    'A/R result contract'
assert_eq qwen-native-training-ar-eval-only "$AR_NAMESPACE" \
    'A/R marker namespace'
assert_eq qwen-native-ar-eval-only-cpu-v1 "$AR_CPU_CONTRACT" \
    'A/R CPU contract'

# The dedicated helper is an eval-only boundary. The end-to-end call log below
# is authoritative; these source checks make an accidental train dispatcher
# visible even if a future mock happens to return early.
ar_eval_job_source="$(declare -f run_ar_eval_job)"
grep -Fq 'train_small_grpo.sh' <<<"$ar_eval_job_source" ||
    fail 'run_ar_eval_job no longer dispatches through the eval entrypoint.'
grep -Fq 'eval "$variant" "$model_path"' <<<"$ar_eval_job_source" ||
    fail 'run_ar_eval_job no longer passes the registered eval variant.'
! grep -Eq 'run_job[[:space:]]+train|train_small_grpo\.sh[^\n]*[[:space:]]train([[:space:]]|$)' \
        <<<"$ar_eval_job_source" ||
    fail 'run_ar_eval_job contains a train dispatch.'

COMMIT="$BC_EXPECTED_CHECKOUT"
HANDOFF_DIGEST="$BC_EXPECTED_HANDOFF"
mkdir -p "$MANIFEST_DIR" "$BC_R60_CHECKPOINT" \
    "$(dirname -- "$NATIVE_TRAIN_MANIFEST")"
printf '%s\n' "$HANDOFF_DIGEST" >"$MANIFEST_DIR/cpu.ok"
printf 'base model fixture\n' >"$MODEL_DIR/model.bin"
printf 'R60 checkpoint fixture\n' >"$BC_R60_CHECKPOINT/model.bin"

# The CPU receipt binds the data manifest and all six resolved endpoint
# configurations. Their scientific contents are covered elsewhere; this test
# exercises the receipt transaction and GPU admission bridge.
printf 'native manifest fixture\n' >"$NATIVE_TRAIN_MANIFEST"
printf 'val data fixture\n' >"$NATIVE_TRAIN_DATA_DIR/val_128.parquet"
printf 'NQ test data fixture\n' \
    >"$NATIVE_TRAIN_DATA_DIR/nq_test_128_native_v4.parquet"
printf 'multihop data fixture\n' \
    >"$NATIVE_TRAIN_DATA_DIR/multihop_eval_256_native_v4.parquet"
for variant in \
        qwen_native_a_val qwen_native_r_val \
        qwen_native_a_nq_test qwen_native_r_nq_test \
        qwen_native_a_multihop qwen_native_r_multihop; do
    printf 'resolved config fixture: %s\n' "$variant" \
        >"$MANIFEST_DIR/config-2gpu-$variant-eval.yaml"
done

BASE_MODEL="$(readlink -f -- "$MODEL_DIR")"
BASE_DIGEST="$(tree_sha256 "$BASE_MODEL")"
DATA_DIGEST="$(file_sha256 "$NATIVE_TRAIN_MANIFEST")"

# Replace only recursive historical evidence verification. Receipt creation,
# checksum sidecars, immutable reuse, and the A/R preflight remain production
# code. Any attempted CPU-side run_job is recorded and fails the test.
expected_commit() { printf '%s\n' "$COMMIT"; }
verify_checkout() { [[ "$1" == "$COMMIT" ]]; }
verify_bc_foundation() {
    assert_eq "$COMMIT" "$1" 'foundation commit'
    assert_eq "$HANDOFF_DIGEST" "$2" 'foundation handoff'
    assert_eq "$BASE_MODEL" "$3" 'foundation base model'
    assert_eq "$BASE_DIGEST" "$4" 'foundation base digest'
    printf 'foundation\n' >>"$CALLS"
}
verify_bc_cpu_receipt() { return 0; }
verify_bc_r60() { return 0; }
verify_native_checkpoint_digest() { return 0; }
sync_path() { return 0; }
BC_CPU_RECEIPT="$TEST_ROOT/g3-foundation-receipt.env"
printf 'sealed G3 foundation receipt\n' >"$BC_CPU_RECEIPT"
printf 'sidecar\n' >"$BC_CPU_RECEIPT.sha256"
BC_CPU_RECEIPT_DIGEST="$(file_sha256 "$BC_CPU_RECEIPT")"

# Populate the exact hashes consumed by ar_receipt_content while replacing the
# large historical B/C recovery tree with this bounded fixture.
verify_ar_foundation() {
    local config
    assert_eq "$COMMIT" "$1" 'A/R foundation commit'
    assert_eq "$HANDOFF_DIGEST" "$2" 'A/R foundation handoff'
    assert_eq "$BASE_MODEL" "$3" 'A/R foundation base model'
    assert_eq "$BASE_DIGEST" "$4" 'A/R foundation base digest'
    AR_SOURCE_DIGEST="$(file_sha256 "$AR_SOURCE_ENTRY")"
    AR_PAIRED_DIGEST="$(file_sha256 "$NATIVE_TRAIN_PAIRED_CLI")"
    AR_DATA_MANIFEST_DIGEST="$(file_sha256 "$NATIVE_TRAIN_MANIFEST")"
    AR_DATA_VAL_DIGEST="$(file_sha256 \
        "$NATIVE_TRAIN_DATA_DIR/val_128.parquet")"
    AR_DATA_NQ_DIGEST="$(file_sha256 \
        "$NATIVE_TRAIN_DATA_DIR/nq_test_128_native_v4.parquet")"
    AR_DATA_MULTIHOP_DIGEST="$(file_sha256 \
        "$NATIVE_TRAIN_DATA_DIR/multihop_eval_256_native_v4.parquet")"
    config="$(ar_config_file qwen_native_a_val)"
    AR_CONFIG_A_VAL_DIGEST="$(file_sha256 "$config")"
    config="$(ar_config_file qwen_native_r_val)"
    AR_CONFIG_R_VAL_DIGEST="$(file_sha256 "$config")"
    config="$(ar_config_file qwen_native_a_nq_test)"
    AR_CONFIG_A_NQ_DIGEST="$(file_sha256 "$config")"
    config="$(ar_config_file qwen_native_r_nq_test)"
    AR_CONFIG_R_NQ_DIGEST="$(file_sha256 "$config")"
    config="$(ar_config_file qwen_native_a_multihop)"
    AR_CONFIG_A_MULTIHOP_DIGEST="$(file_sha256 "$config")"
    config="$(ar_config_file qwen_native_r_multihop)"
    AR_CONFIG_R_MULTIHOP_DIGEST="$(file_sha256 "$config")"
    printf 'ar-foundation\n' >>"$CALLS"
}
run_job() {
    printf 'run_job|%s\n' "$*" >>"$CALLS"
    fail 'CPU preparation attempted to start a model job.'
}

prepare_ar_cpu_receipt
[[ -n "${AR_CPU_RECEIPT:-}" && -f "$AR_CPU_RECEIPT" &&
    ! -L "$AR_CPU_RECEIPT" && -f "$AR_CPU_RECEIPT.sha256" &&
    ! -L "$AR_CPU_RECEIPT.sha256" ]] ||
    fail 'A/R CPU preparation did not publish a regular receipt and sidecar.'
grep -Fxq 'gpu_training_authorized=false' "$AR_CPU_RECEIPT"
grep -Fxq 'gpu_evaluation_authorized=true' "$AR_CPU_RECEIPT"
grep -Fxq 'manual_gpu_start_required=true' "$AR_CPU_RECEIPT"
! grep -q '^run_job|' "$CALLS" ||
    fail 'CPU preparation started a GPU job.'

receipt_directory="$(dirname -- "$AR_CPU_RECEIPT")"
receipt_fingerprint_before="$(
    sha256sum -- "$AR_CPU_RECEIPT" "$AR_CPU_RECEIPT.sha256" \
        "$receipt_directory/terminal" "$receipt_directory/exit-code" \
        "$receipt_directory/.success" | sha256sum | cut -d' ' -f1
)"
prepare_ar_cpu_receipt
receipt_fingerprint_after="$(
    sha256sum -- "$AR_CPU_RECEIPT" "$AR_CPU_RECEIPT.sha256" \
        "$receipt_directory/terminal" "$receipt_directory/exit-code" \
        "$receipt_directory/.success" | sha256sum | cut -d' ' -f1
)"
assert_eq "$receipt_fingerprint_before" "$receipt_fingerprint_after" \
    'idempotent CPU receipt'
! grep -q '^run_job|' "$CALLS" ||
    fail 'Idempotent CPU preparation started a GPU job.'

# A differing existing receipt must never be overwritten or silently adopted.
cp "$AR_CPU_RECEIPT" "$TEST_ROOT/receipt.good"
printf 'drift=true\n' >>"$AR_CPU_RECEIPT"
if prepare_ar_cpu_receipt >/dev/null 2>&1; then
    fail 'CPU preparation accepted a drifted existing receipt.'
fi
grep -Fxq 'drift=true' "$AR_CPU_RECEIPT" ||
    fail 'CPU preparation overwrote a drifted receipt.'
cp "$TEST_ROOT/receipt.good" "$AR_CPU_RECEIPT"
verify_ar_cpu_receipt "$(file_sha256 "$AR_RUNNER_PATH")" ||
    fail 'Restored A/R CPU receipt did not verify.'

# The receipt namespace itself must not be redirectable through a symlink.
mv -- "$AR_CPU_ROOT" "$TEST_ROOT/cpu-root.saved"
mkdir "$TEST_ROOT/cpu-root-escape"
ln -s "$TEST_ROOT/cpu-root-escape" "$AR_CPU_ROOT"
if prepare_ar_cpu_receipt >/dev/null 2>&1; then
    fail 'CPU preparation followed a symlinked receipt root.'
fi
rm -- "$AR_CPU_ROOT"
mv -- "$TEST_ROOT/cpu-root.saved" "$AR_CPU_ROOT"
verify_ar_cpu_receipt "$(file_sha256 "$AR_RUNNER_PATH")" ||
    fail 'A/R CPU receipt failed after symlink-root rejection.'

# Its manifest parent must also remain a regular directory inside the project.
mv -- "$MANIFEST_DIR" "$TEST_ROOT/manifests.saved"
mkdir "$TEST_ROOT/manifests-escape"
ln -s "$TEST_ROOT/manifests-escape" "$MANIFEST_DIR"
if prepare_ar_cpu_receipt >/dev/null 2>&1; then
    fail 'CPU preparation followed a symlinked manifest root.'
fi
rm -- "$MANIFEST_DIR"
mv -- "$TEST_ROOT/manifests.saved" "$MANIFEST_DIR"
verify_ar_cpu_receipt "$(file_sha256 "$AR_RUNNER_PATH")" ||
    fail 'A/R CPU receipt failed after manifest-root rejection.'

# Exercise the real A/R preflight before replacing runtime-only helpers.
qwen_native_train_preflight "$COMMIT" "$HANDOFF_DIGEST" "$BASE_DIGEST"

SYSTEM_PYTHON="$PYTHON_BIN"
export SYSTEM_PYTHON
cat >"$TRAIN_ENV/bin/python" <<'SH'
#!/usr/bin/env bash
set -Eeuo pipefail
tool="${1##*/}"
if [[ "$tool" == paired_eval.py ]]; then
    output=''
    while (($#)); do
        if [[ "$1" == --output-dir ]]; then
            output="$2"
            break
        fi
        shift
    done
    [[ -n "$output" ]]
    printf 'python|paired_eval.py|%s\n' "$output" >>"$CALLS"
    mkdir -p "$output"
    printf '{}\n' >"$output/summary.json"
    printf '# paired A/R fixture\n' >"$output/summary.md"
    printf 'sample_id\n' >"$output/paired_results.csv"
    printf 'sample_id\n' >"$output/correct_questions.csv"
    printf 'sample_id\n' >"$output/wrong_questions.csv"
    printf 'from,to,count\n' >"$output/search_transition.csv"
    exit 0
fi
exec "$SYSTEM_PYTHON" "$@"
SH
chmod 0755 "$TRAIN_ENV/bin/python"

FAIL_VARIANT=''
RUN_SUFFIX=success
run_ar_eval_job() {
    local variant="$1" model="$2" digest="$3"
    local expected_model expected_digest
    case "$variant" in
        qwen_native_a_*)
            expected_model="$BASE_MODEL"
            expected_digest="$BASE_DIGEST"
            ;;
        qwen_native_r_*)
            expected_model="$BC_R60_CHECKPOINT"
            expected_digest="$BC_EXPECTED_R60_CHECKPOINT"
            ;;
        *) return 64 ;;
    esac
    assert_eq "$expected_model" "$model" "$variant model"
    assert_eq "$expected_digest" "$digest" "$variant model digest"
    [[ "$EVAL_GROUP_SIZE" == 1 && -z "$EVAL_DATA_FILE" ]] || return 64
    printf 'run_job|%s|%s|%s|%s|%s|eval=%s|rows=%s|group=%s\n' \
        eval "$variant" "$model" '' "$digest" \
        "$EVAL_DATA_FILE" "$EVAL_EXPECTED_ROWS" "$EVAL_GROUP_SIZE" \
        >>"$CALLS"
    [[ "$variant" != "$FAIL_VARIANT" ]] || return 23
    LAST_RUN_DIR="$RUNS_ROOT/eval/$variant/attempts/fake-$RUN_SUFFIX"
    mkdir -p "$LAST_RUN_DIR/traces" "$LAST_RUN_DIR/wandb/offline-run-test"
    printf 'eval log\n' >"$LAST_RUN_DIR/train.log"
    printf 'resolved config\n' >"$LAST_RUN_DIR/resolved-config.yaml"
    printf 'run env\n' >"$LAST_RUN_DIR/run.env"
    printf '{}\n' >"$LAST_RUN_DIR/traces/eval_predictions.jsonl"
    printf '{}\n' >"$LAST_RUN_DIR/traces/eval_predictions.manifest.json"
    printf 'sidecar\n' \
        >"$LAST_RUN_DIR/traces/eval_predictions.manifest.json.sha256"
    printf 'success\n' >"$LAST_RUN_DIR/terminal"
    printf '0\n' >"$LAST_RUN_DIR/exit-code"
    : >"$LAST_RUN_DIR/.success"
    printf 'wandb fixture\n' \
        >"$LAST_RUN_DIR/wandb/offline-run-test/run-test.wandb"
}

verify_native_trace_identity() {
    printf '%s\t%s\n' "$(printf '5%.0s' {1..64})" \
        "$(printf '6%.0s' {1..64})"
}
verify_ar_eval_runtime_config() { return 0; }
verify_ar_runtime_config() { return 0; }
verify_ar_endpoint_config() { return 0; }
verify_ar_results_semantics() { return 0; }
write_wandb_receipt() {
    printf '{"decision":"GO"}\n' >"$1/wandb-receipt.json"
}
verify_wandb_receipt() { return 0; }
verify_native_training_checksum_manifest() { return 0; }

# Keep publication observable while preserving its marker/result-pointer
# ordering contract. This isolates the A/R dispatcher from the already-covered
# generic evidence manifest implementation.
publish_ar_training_evidence() {
    local outer="$1" results="$2"
    local marker="$MANIFEST_DIR/$AR_NAMESPACE/$(basename -- "$outer").ok"
    printf 'publish|%s|%s|%s|%s\n' "$outer" "$results" "$AR_NAMESPACE" \
        "$AR_CONTRACT" >>"$CALLS"
    mkdir -p "$(dirname -- "$marker")"
    printf 'fixture evidence\n' >"$results/evidence.sha256"
    printf '%s\n' "$(file_sha256 "$results/evidence.sha256")" >"$marker"
    printf '%s\n' "$AR_CONTRACT" >"$outer/result-contract"
    printf '%s\n' "$results" >"$outer/result-root"
    printf '%s\n' "$marker" >"$outer/evidence-marker"
    printf '%s\n' "$(file_sha256 "$results/evidence.sha256")" \
        >"$outer/evidence-digest"
}

: >"$CALLS"
SUCCESS_OUTER="$ATTEMPTS_ROOT/gpu/20260801T200000Z-1-1"
mkdir -p "$SUCCESS_OUTER"
qwen_native_main_pipeline "$SUCCESS_OUTER" "$COMMIT" "$HANDOFF_DIGEST" \
    "$BASE_MODEL" "$BASE_DIGEST" "$DATA_DIGEST"

mapfile -t jobs < <(grep '^run_job|' "$CALLS")
assert_eq 6 "${#jobs[@]}" 'A/R eval job count'
expected_variants=(
    qwen_native_a_val qwen_native_r_val
    qwen_native_a_nq_test qwen_native_r_nq_test
    qwen_native_a_multihop qwen_native_r_multihop
)
expected_rows=(128 128 128 128 256 256)
for index in "${!expected_variants[@]}"; do
    [[ "${jobs[$index]}" == "run_job|eval|${expected_variants[$index]}|"* ]] ||
        fail "Unexpected A/R eval order at index $index: ${jobs[$index]}"
    [[ "${jobs[$index]}" == *"|rows=${expected_rows[$index]}|group=1" ]] ||
        fail "Unexpected row/group contract: ${jobs[$index]}"
done
! grep -Eq '^run_job\|train\|' "$CALLS" ||
    fail 'A/R eval-only pipeline started training.'

assert_eq "$BASE_MODEL" "$(cut -d'|' -f4 <<<"${jobs[0]}")" \
    'A val checkpoint'
assert_eq "$BASE_MODEL" "$(cut -d'|' -f4 <<<"${jobs[2]}")" \
    'A NQ checkpoint'
assert_eq "$BASE_MODEL" "$(cut -d'|' -f4 <<<"${jobs[4]}")" \
    'A multihop checkpoint'
assert_eq "$BC_R60_CHECKPOINT" "$(cut -d'|' -f4 <<<"${jobs[1]}")" \
    'R val checkpoint'
assert_eq "$BC_R60_CHECKPOINT" "$(cut -d'|' -f4 <<<"${jobs[3]}")" \
    'R NQ checkpoint'
assert_eq "$BC_R60_CHECKPOINT" "$(cut -d'|' -f4 <<<"${jobs[5]}")" \
    'R multihop checkpoint'

mapfile -t paired_calls < <(grep '^python|paired_eval.py|' "$CALLS")
assert_eq 3 "${#paired_calls[@]}" 'paired A/R analysis count'
SUCCESS_RESULTS="$(tr -d '\r\n' <"$SUCCESS_OUTER/result-root")"
for eval_key in val nq_test multihop; do
    paired="$SUCCESS_RESULTS/paired-ar-$eval_key"
    [[ -s "$paired/summary.json" && -s "$paired/summary.md" &&
        -s "$paired/paired_results.csv" &&
        -s "$paired/correct_questions.csv" &&
        -s "$paired/wrong_questions.csv" &&
        -s "$paired/search_transition.csv" ]] ||
        fail "Missing paired A/R evidence: $paired"
done
assert_eq 1 "$(grep -c '^publish|' "$CALLS")" \
    'successful evidence publication count'
assert_eq "$AR_CONTRACT" "$(tr -d '\r\n' <"$SUCCESS_OUTER/result-contract")" \
    'published A/R contract'
actual_stage_order="$(tail -n +2 "$SUCCESS_RESULTS/run-index.tsv" |
    cut -f1 | paste -sd, -)"
assert_eq "$AR_STAGE_ORDER" "$actual_stage_order" \
    'published A/R stage order'
lineage_stage_order="$(tail -n +2 "$SUCCESS_RESULTS/lineage.tsv" |
    cut -f1 | paste -sd, -)"
assert_eq "$AR_STAGE_ORDER" "$lineage_stage_order" \
    'published A/R lineage stage order'

# An engineering eval failure preserves the original code, stops subsequent
# endpoints, and cannot publish a marker or any outer result pointer.
: >"$CALLS"
RUN_SUFFIX=failure
FAIL_VARIANT=qwen_native_r_nq_test
FAILED_OUTER="$ATTEMPTS_ROOT/gpu/20260801T200100Z-1-2"
mkdir -p "$FAILED_OUTER"
set +e
qwen_native_main_pipeline "$FAILED_OUTER" "$COMMIT" "$HANDOFF_DIGEST" \
    "$BASE_MODEL" "$BASE_DIGEST" "$DATA_DIGEST"
failure_rc=$?
set -e
assert_eq 23 "$failure_rc" 'failed eval original status'
! grep -q '^publish|' "$CALLS" ||
    fail 'Failed A/R evaluation published evidence.'
[[ ! -e "$FAILED_OUTER/result-contract" &&
    ! -e "$FAILED_OUTER/evidence-marker" ]] ||
    fail 'Failed A/R evaluation published outer result pointers.'
failed_marker="$MANIFEST_DIR/$AR_NAMESPACE/$(basename -- "$FAILED_OUTER").ok"
[[ ! -e "$failed_marker" && ! -L "$failed_marker" ]] ||
    fail 'Failed A/R evaluation published a success marker.'
! grep -Eq '^run_job\|train\|' "$CALLS" ||
    fail 'Failed A/R evaluation attempted training.'

printf 'Qwen native A/R eval-only pipeline tests passed.\n'
