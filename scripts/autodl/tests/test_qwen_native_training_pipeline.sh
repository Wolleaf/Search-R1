#!/usr/bin/env bash
set -Eeuo pipefail

AUTODL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
ENTRYPOINT="$AUTODL_DIR/09_gpu_qwen_native_train.sh"
GPU_RUN="$AUTODL_DIR/03_gpu_run.sh"
TEST_ROOT="$(mktemp -d)"
trap 'rm -rf -- "$TEST_ROOT"' EXIT

PROTOCOL_GATE_MARKER="$TEST_ROOT/protocol-gate.ok"
SMOKE_MARKER="$TEST_ROOT/smoke.ok"
printf 'marker\n' >"$PROTOCOL_GATE_MARKER"
printf 'marker\n' >"$SMOKE_MARKER"

if AUTODL_ROOT="$TEST_ROOT/reject-gpu" GPU_COUNT=1 \
        AUTODL_PRICE_PER_HOUR=5.76 TRAIN_BATCH_SIZE=8 MAX_RESPONSE_LENGTH=500 \
        QWEN_NATIVE_TRAIN_STAGE=smoke \
        QWEN_NATIVE_PROTOCOL_GATE_EVIDENCE="$PROTOCOL_GATE_MARKER" \
        bash -c 'source "$1"; require_qwen_native_train' _ "$ENTRYPOINT" \
        >/dev/null 2>&1; then
    printf 'Native training accepted a one-GPU configuration.\n' >&2
    exit 1
fi
if AUTODL_ROOT="$TEST_ROOT/reject-response" GPU_COUNT=2 \
        AUTODL_PRICE_PER_HOUR=5.76 TRAIN_BATCH_SIZE=8 MAX_RESPONSE_LENGTH=384 \
        QWEN_NATIVE_TRAIN_STAGE=smoke \
        QWEN_NATIVE_PROTOCOL_GATE_EVIDENCE="$PROTOCOL_GATE_MARKER" \
        bash -c 'source "$1"; require_qwen_native_train' _ "$ENTRYPOINT" \
        >/dev/null 2>&1; then
    printf 'Native training accepted the response-length fallback.\n' >&2
    exit 1
fi
if AUTODL_ROOT="$TEST_ROOT/reject-stage" GPU_COUNT=2 \
        AUTODL_PRICE_PER_HOUR=5.76 TRAIN_BATCH_SIZE=8 MAX_RESPONSE_LENGTH=500 \
        QWEN_NATIVE_TRAIN_STAGE=other \
        QWEN_NATIVE_PROTOCOL_GATE_EVIDENCE="$PROTOCOL_GATE_MARKER" \
        bash -c 'source "$1"; require_qwen_native_train' _ "$ENTRYPOINT" \
        >/dev/null 2>&1; then
    printf 'Native training accepted an unknown paid stage.\n' >&2
    exit 1
fi
if AUTODL_ROOT="$TEST_ROOT/reject-main" GPU_COUNT=2 \
        AUTODL_PRICE_PER_HOUR=5.76 TRAIN_BATCH_SIZE=8 MAX_RESPONSE_LENGTH=500 \
        QWEN_NATIVE_TRAIN_STAGE=main \
        QWEN_NATIVE_PROTOCOL_GATE_EVIDENCE="$PROTOCOL_GATE_MARKER" \
        bash -c 'source "$1"; require_qwen_native_train' _ "$ENTRYPOINT" \
        >/dev/null 2>&1; then
    printf 'Main training accepted a missing smoke marker.\n' >&2
    exit 1
fi

AUTODL_ROOT="$TEST_ROOT/project"
GPU_COUNT=2
AUTODL_PRICE_PER_HOUR=5.76
TRAIN_BATCH_SIZE=8
MAX_RESPONSE_LENGTH=500
QWEN_NATIVE_TRAIN_STAGE=smoke
QWEN_NATIVE_PROTOCOL_GATE_EVIDENCE="$PROTOCOL_GATE_MARKER"
QWEN_NATIVE_SMOKE_EVIDENCE=''
source "$ENTRYPOINT"

# The shared run record must describe the neutral native-v3 sampling contract.
record_dir="$TEST_ROOT/project/runs/record"
mkdir -p "$record_dir"
: >"$record_dir/.running"
printf 'config\n' >"$record_dir/resolved-config.yaml"
finish_run_record "$record_dir" 0 "$(date +%s)" "$(utc_now)" \
    1 60 train smoke 2 "$MODEL_DIR" "$record_dir/traces"
grep -Fxq 'rollout_top_k=0' "$record_dir/run.env"
grep -Fxq 'rollout_min_p=0.0' "$record_dir/run.env"
grep -Fxq 'rollout_presence_penalty=0.0' "$record_dir/run.env"
grep -Fxq 'rollout_repetition_penalty=1.0' "$record_dir/run.env"

# Missing and content-tampered exact predecessor markers fail before metadata parsing.
MISSING="$TEST_ROOT/project/manifests/qwen-native-training-smoke/20260724T010101Z-1-1.ok"
if verify_complete_training_attempt "$MISSING" qwen-native-training-smoke \
        "$NATIVE_TRAIN_RESULTS_ROOT/attempts" "$NATIVE_TRAIN_SMOKE_CONTRACT" \
        >/dev/null 2>&1; then
    printf 'Missing smoke predecessor was accepted.\n' >&2
    exit 1
fi
TAMPER_NAME=20260724T010102Z-1-2
TAMPER_DIR="$TEST_ROOT/project/manifests/qwen-native-training-smoke"
TAMPER_RESULTS="$NATIVE_TRAIN_RESULTS_ROOT/attempts/$TAMPER_NAME"
TAMPER_OUTER="$TEST_ROOT/project/state/attempts/gpu/$TAMPER_NAME"
mkdir -p "$TAMPER_DIR" "$TAMPER_RESULTS" "$TAMPER_OUTER"
printf '%064d\n' 0 >"$TAMPER_DIR/$TAMPER_NAME.ok"
printf 'tampered evidence\n' >"$TAMPER_RESULTS/evidence.sha256"
printf 'success\n' >"$TAMPER_OUTER/terminal"
printf '0\n' >"$TAMPER_OUTER/exit-code"
: >"$TAMPER_OUTER/.success"
if verify_complete_training_attempt "$TAMPER_DIR/$TAMPER_NAME.ok" \
        qwen-native-training-smoke "$NATIVE_TRAIN_RESULTS_ROOT/attempts" \
        "$NATIVE_TRAIN_SMOKE_CONTRACT" >/dev/null 2>&1; then
    printf 'Tampered smoke predecessor was accepted.\n' >&2
    exit 1
fi

CALLS="$TEST_ROOT/calls.log"
export CALLS
BASE_DIGEST="$(printf 'a%.0s' {1..64})"
SMOKE_DIGEST="$(printf 'b%.0s' {1..64})"
R_DIGEST="$(printf 'c%.0s' {1..64})"
B_DIGEST="$(printf 'd%.0s' {1..64})"
C_DIGEST="$(printf 'e%.0s' {1..64})"
DATA_DIGEST="$(printf 'f%.0s' {1..64})"
HANDOFF_DIGEST="$(printf '1%.0s' {1..64})"
COMMIT="$(printf '2%.0s' {1..40})"
PROTOCOL_GATE_DIGEST="$(printf '3%.0s' {1..64})"
SMOKE_EVIDENCE_DIGEST="$(printf '4%.0s' {1..64})"
TRACE_DIGEST="$(printf '5%.0s' {1..64})"
TRACE_MANIFEST_DIGEST="$(printf '6%.0s' {1..64})"
BASE_MODEL="$TEST_ROOT/project/models/Qwen3.5-2B"
mkdir -p "$BASE_MODEL"

FAKE_ENV="$TEST_ROOT/fake-env"
mkdir -p "$FAKE_ENV/bin"
cat >"$FAKE_ENV/bin/python" <<'SH'
#!/usr/bin/env bash
set -Eeuo pipefail
printf 'python|%s\n' "$*" >>"$CALLS"
tool="${1##*/}"
output=''
while (($#)); do
    if [[ "$1" == --output-dir || "$1" == --output ]]; then
        output="$2"
        break
    fi
    shift
done
[[ -n "$output" ]]
if [[ "$tool" == qwen_native_smoke_analysis.py ]]; then
    mkdir -p "$(dirname -- "$output")"
    printf '{"decision":"GO","schema":"search-r1.qwen-native-smoke-decision","schema_version":1}\n' >"$output"
    printf 'GO\n'
elif [[ "$tool" == paired_eval.py ]]; then
    mkdir -p "$output"
    for name in summary.json paired_results.csv correct_questions.csv wrong_questions.csv search_transition.csv; do
        printf 'paired\n' >"$output/$name"
    done
    printf '# paired\n' >"$output/summary.md"
else
    mkdir -p "$output"
    for name in summary.json go_no_go.json; do printf '{}\n' >"$output/$name"; done
    printf '# analysis\n' >"$output/summary.md"
    printf '{}\n' >"$output/per_trajectory.jsonl"
    printf '{}\n' >"$output/per_question.jsonl"
fi
SH
chmod +x "$FAKE_ENV/bin/python"
TRAIN_ENV="$FAKE_ENV"

require_qwen_native_train() { return 0; }
verify_checkout() { return 0; }
file_sha256() { printf '%s\n' "$DATA_DIGEST"; }
verify_native_checkpoint_digest() {
    printf 'verify_checkpoint|%s|%s\n' "$1" "$2" >>"$CALLS"
}
verify_protocol_gate_evidence() {
    printf 'verify_protocol_gate|%s|%s|%s|%s|%s|%s\n' "$@" >>"$CALLS"
    NATIVE_TRAIN_PROTOCOL_GATE_DIGEST="$PROTOCOL_GATE_DIGEST"
    NATIVE_TRAIN_PROTOCOL_GATE_RESULTS="$TEST_ROOT/protocol-gate-results"
    NATIVE_TRAIN_PROTOCOL_GATE_OUTER="$TEST_ROOT/protocol-gate-outer"
    NATIVE_TRAIN_PROTOCOL_GATE_MANIFEST="$TEST_ROOT/protocol-gate-evidence.sha256"
}
verify_smoke_evidence() {
    printf 'verify_smoke|%s|%s|%s|%s|%s|%s|%s\n' "$@" >>"$CALLS"
    NATIVE_TRAIN_SMOKE_DIGEST="$SMOKE_EVIDENCE_DIGEST"
    NATIVE_TRAIN_SMOKE_RESULTS="$TEST_ROOT/smoke-results"
    NATIVE_TRAIN_SMOKE_OUTER="$TEST_ROOT/smoke-outer"
    NATIVE_TRAIN_SMOKE_MANIFEST="$TEST_ROOT/smoke-evidence.sha256"
    NATIVE_TRAIN_SMOKE_CHECKPOINT="$TEST_ROOT/smoke-checkpoint"
    NATIVE_TRAIN_SMOKE_CHECKPOINT_DIGEST="$SMOKE_DIGEST"
}
publish_native_training_evidence() {
    printf 'publish|%s|%s|%s|%s\n' "$1" "$2" "$3" "$4" >>"$CALLS"
}
run_job() {
    local mode="$1" variant="$2"
    printf 'run_job|%s|%s|%s|%s|%s|eval=%s|rows=%s|group=%s\n' \
        "$1" "$2" "${3:-}" "${4:-}" "${5:-}" \
        "$EVAL_DATA_FILE" "$EVAL_EXPECTED_ROWS" "$EVAL_GROUP_SIZE" >>"$CALLS"
    if [[ "${FAIL_VARIANT:-}" == "$variant" ]]; then
        return 23
    fi
    LAST_RUN_DIR="$TEST_ROOT/project/runs/$variant/attempts/fake-${RUN_SUFFIX:-ok}"
    mkdir -p "$LAST_RUN_DIR/checkpoints/actor/global_step_2" \
        "$LAST_RUN_DIR/checkpoints/actor/global_step_20" \
        "$LAST_RUN_DIR/checkpoints/actor/global_step_60" "$LAST_RUN_DIR/traces" \
        "$LAST_RUN_DIR/wandb/offline-run-test"
    if [[ "${WANDB_DEBUG_ONLY_VARIANT:-}" == "$variant" ]]; then
        printf 'debug-only\n' >"$LAST_RUN_DIR/wandb/offline-run-test/debug.log"
    else
        printf 'offline-history\n' \
            >"$LAST_RUN_DIR/wandb/offline-run-test/run-test.wandb"
    fi
}
fixed_checkpoint() {
    printf '%s/checkpoints/actor/global_step_%s\n' "$1" "$2"
}
tree_sha256() {
    case "$1" in
        "$BASE_MODEL") printf '%s\n' "$BASE_DIGEST" ;;
        *'/smoke/'*) printf '%s\n' "$SMOKE_DIGEST" ;;
        *'/reproduce/'*) printf '%s\n' "$R_DIGEST" ;;
        *'/control/'*) printf '%s\n' "$B_DIGEST" ;;
        *'/cost_aware_gated/'*) printf '%s\n' "$C_DIGEST" ;;
        *) printf 'unexpected checkpoint: %s\n' "$1" >&2; return 1 ;;
    esac
}
record_lineage() {
    printf 'lineage|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\n' "$@" >>"$CALLS"
}
seal_native_training_run() {
    printf 'seal|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\n' "$@" >>"$CALLS"
}
verify_native_trace_identity() {
    printf '%s\t%s\n' "$TRACE_DIGEST" "$TRACE_MANIFEST_DIGEST"
}
record_smoke_storage() {
    atomic_write "$1/storage.env" \
        'checkpoint_bytes=1'$'\n''filesystem_available_bytes=2'$'\n''recorded_at=test'$'\n'
}
validate_r_g3_analysis() {
    printf '%s\t%s\n' "${ANALYSIS_OUTCOME:-GO}" "${CONTRAST_COUNT:-8}"
}
write_branch_decision() {
    printf '{"analysis_decision":"%s","cost_contrast_group_count":%s}\n' "$2" "$3"
}

prepare_preflight() {
    local stage="$1"
    NATIVE_TRAIN_STAGE="$stage"
    NATIVE_TRAIN_PREFLIGHT_STAGE="$stage"
    NATIVE_TRAIN_PREFLIGHT_COMMIT="$COMMIT"
    NATIVE_TRAIN_PREFLIGHT_HANDOFF="$HANDOFF_DIGEST"
    NATIVE_TRAIN_PREFLIGHT_BASE_DIGEST="$BASE_DIGEST"
    NATIVE_TRAIN_PREFLIGHT_DATA_DIGEST="$DATA_DIGEST"
    NATIVE_TRAIN_PREFLIGHT_PROTOCOL_GATE_DIGEST="$PROTOCOL_GATE_DIGEST"
    if [[ "$stage" == main ]]; then
        NATIVE_TRAIN_PREFLIGHT_SMOKE_DIGEST="$SMOKE_EVIDENCE_DIGEST"
        NATIVE_SMOKE_EVIDENCE="$SMOKE_MARKER"
    else
        NATIVE_TRAIN_PREFLIGHT_SMOKE_DIGEST=-
        NATIVE_SMOKE_EVIDENCE=''
    fi
}

# Smoke is an independent outer attempt and stops after exactly two steps.
: >"$CALLS"
prepare_preflight smoke
smoke_outer="$TEST_ROOT/project/state/attempts/gpu/20260724T020000Z-10-1"
mkdir -p "$smoke_outer"
qwen_native_train_pipeline "$smoke_outer" "$COMMIT" "$HANDOFF_DIGEST" \
    "$BASE_MODEL" "$BASE_DIGEST"
mapfile -t smoke_jobs < <(grep '^run_job|' "$CALLS")
[[ "${#smoke_jobs[@]}" == 1 ]]
[[ "${smoke_jobs[0]}" == "run_job|train|smoke|2||$BASE_DIGEST|eval=|rows=128|group=1" ]]
! grep -Eq 'run_job\|train\|(reproduce|control|cost_aware_gated)' "$CALLS"
grep -q '^publish|.*|qwen-native-training-smoke|qwen-native-training-smoke-v3$' "$CALLS"
smoke_results="$NATIVE_TRAIN_RESULTS_ROOT/attempts/$(basename -- "$smoke_outer")"
grep -Fxq 'stage_order=S2' "$smoke_results/contract.env"
grep -Fxq 'decision=GO' "$smoke_results/contract.env"
grep -Fxq 'manual_review_required=true' "$smoke_results/contract.env"
grep -Fxq "protocol_gate_evidence=$PROTOCOL_GATE_MARKER" "$smoke_results/contract.env"
grep -Fxq "protocol_gate_evidence_sha256=$PROTOCOL_GATE_DIGEST" \
    "$smoke_results/contract.env"
grep -Fxq 'checkpoint_bytes=1' "$smoke_results/storage.env"
[[ "$(wc -l <"$smoke_results/lineage.tsv")" == 2 ]]

# Main starts R from base, evaluates exact 64x5 G3, then branches only on both gates.
: >"$CALLS"
prepare_preflight main
ANALYSIS_OUTCOME=GO
CONTRAST_COUNT=8
main_outer="$TEST_ROOT/project/state/attempts/gpu/20260724T020100Z-10-2"
mkdir -p "$main_outer"
qwen_native_train_pipeline "$main_outer" "$COMMIT" "$HANDOFF_DIGEST" \
    "$BASE_MODEL" "$BASE_DIGEST"
mapfile -t main_jobs < <(grep '^run_job|' "$CALLS")
[[ "${#main_jobs[@]}" == 16 ]]
[[ "${main_jobs[0]}" == "run_job|train|reproduce|60||$BASE_DIGEST|eval=|rows=128|group=1" ]]
R_CHECKPOINT="$TEST_ROOT/project/runs/reproduce/attempts/fake-ok/checkpoints/actor/global_step_60"
[[ "${main_jobs[1]}" == "run_job|eval|qwen_native_g3|$R_CHECKPOINT||$R_DIGEST|eval=$NATIVE_TRAIN_G3_DATA|rows=64|group=5" ]]
[[ "${main_jobs[2]}" == "run_job|eval|qwen_native_a_val|$BASE_MODEL||$BASE_DIGEST|eval=|rows=128|group=1" ]]
[[ "${main_jobs[3]}" == "run_job|eval|qwen_native_r_val|$R_CHECKPOINT||$R_DIGEST|eval=|rows=128|group=1" ]]
[[ "${main_jobs[4]}" == "run_job|eval|qwen_native_a_nq_test|$BASE_MODEL||$BASE_DIGEST|eval=|rows=128|group=1" ]]
[[ "${main_jobs[5]}" == "run_job|eval|qwen_native_r_nq_test|$R_CHECKPOINT||$R_DIGEST|eval=|rows=128|group=1" ]]
[[ "${main_jobs[6]}" == "run_job|eval|qwen_native_a_multihop|$BASE_MODEL||$BASE_DIGEST|eval=|rows=256|group=1" ]]
[[ "${main_jobs[7]}" == "run_job|eval|qwen_native_r_multihop|$R_CHECKPOINT||$R_DIGEST|eval=|rows=256|group=1" ]]
B_CHECKPOINT="$TEST_ROOT/project/runs/control/attempts/fake-ok/checkpoints/actor/global_step_20"
C_CHECKPOINT="$TEST_ROOT/project/runs/cost_aware_gated/attempts/fake-ok/checkpoints/actor/global_step_20"
[[ "${main_jobs[8]}" == "run_job|train|control|20|$R_CHECKPOINT|$R_DIGEST|eval=|rows=128|group=1" ]]
[[ "${main_jobs[9]}" == "run_job|train|cost_aware_gated|20|$R_CHECKPOINT|$R_DIGEST|eval=|rows=128|group=1" ]]
[[ "${main_jobs[10]}" == "run_job|eval|qwen_native_b_val|$B_CHECKPOINT||$B_DIGEST|eval=|rows=128|group=1" ]]
[[ "${main_jobs[11]}" == "run_job|eval|qwen_native_c_val|$C_CHECKPOINT||$C_DIGEST|eval=|rows=128|group=1" ]]
[[ "${main_jobs[12]}" == "run_job|eval|qwen_native_b_nq_test|$B_CHECKPOINT||$B_DIGEST|eval=|rows=128|group=1" ]]
[[ "${main_jobs[13]}" == "run_job|eval|qwen_native_c_nq_test|$C_CHECKPOINT||$C_DIGEST|eval=|rows=128|group=1" ]]
[[ "${main_jobs[14]}" == "run_job|eval|qwen_native_b_multihop|$B_CHECKPOINT||$B_DIGEST|eval=|rows=256|group=1" ]]
[[ "${main_jobs[15]}" == "run_job|eval|qwen_native_c_multihop|$C_CHECKPOINT||$C_DIGEST|eval=|rows=256|group=1" ]]
! grep -q 'run_job|train|smoke' "$CALLS"
main_results="$NATIVE_TRAIN_RESULTS_ROOT/attempts/$(basename -- "$main_outer")"
grep -Fxq 'stage_order=R60,G3,A-VAL-EVAL,R-VAL-EVAL,A-NQ-TEST-EVAL,R-NQ-TEST-EVAL,A-MULTIHOP-EVAL,R-MULTIHOP-EVAL,B20,C20,B-VAL-EVAL,C-VAL-EVAL,B-NQ-TEST-EVAL,C-NQ-TEST-EVAL,B-MULTIHOP-EVAL,C-MULTIHOP-EVAL' \
    "$main_results/contract.env"
grep -Fxq 'branch_authorized=true' "$main_results/contract.env"
[[ "$(wc -l <"$main_results/lineage.tsv")" == 17 ]]
for eval_key in val nq_test multihop; do
    [[ -s "$main_results/paired-ar-$eval_key/summary.json" ]]
    [[ -s "$main_results/paired-ar-$eval_key/paired_results.csv" ]]
    [[ -s "$main_results/paired-$eval_key/summary.json" ]]
    [[ -s "$main_results/paired-$eval_key/paired_results.csv" ]]
done
mapfile -t paired_calls < <(grep '^python|.*paired_eval.py ' "$CALLS")
[[ "${#paired_calls[@]}" == 6 ]]
eval_keys=(val nq_test multihop)
eval_artifacts=(val nq_test_eval multihop_eval)
eval_row_counts=(128 128 256)
for index in 0 1 2; do
    call="${paired_calls[$index]}"
    [[ "$call" == *"--parent "* ]]
    [[ "$call" == *"--reproduced "* ]]
    [[ "$call" == *"--data-manifest $NATIVE_TRAIN_MANIFEST"* ]]
    [[ "$call" == *"--eval-artifact ${eval_artifacts[$index]}"* ]]
    [[ "$call" == *"--expected-parent-checkpoint-digest $BASE_DIGEST"* ]]
    [[ "$call" == *"--expected-reproduced-checkpoint-digest $R_DIGEST"* ]]
    [[ "$call" == *"--output-dir $main_results/paired-ar-${eval_keys[$index]}"* ]]
    [[ "$call" == *"--expected-rows ${eval_row_counts[$index]}"* ]]
done
for index in 0 1 2; do
    call="${paired_calls[$((index + 3))]}"
    [[ "$call" == *"--control "* ]]
    [[ "$call" == *"--cost-aware-gated "* ]]
    [[ "$call" == *"--data-manifest $NATIVE_TRAIN_MANIFEST"* ]]
    [[ "$call" == *"--eval-artifact ${eval_artifacts[$index]}"* ]]
    [[ "$call" == *"--expected-control-checkpoint-digest $B_DIGEST"* ]]
    [[ "$call" == *"--expected-cost-aware-gated-checkpoint-digest $C_DIGEST"* ]]
    [[ "$call" == *"--output-dir $main_results/paired-${eval_keys[$index]}"* ]]
    [[ "$call" == *"--expected-rows ${eval_row_counts[$index]}"* ]]
done
awk -F '\t' -v checkpoint="$R_CHECKPOINT" -v digest="$R_DIGEST" '
    $1 == "B" || $1 == "C" {
        if ($6 != checkpoint || $7 != digest) exit 1
        seen += 1
    }
    END { exit !(seen == 2) }
' "$main_results/lineage.tsv"
awk -F '\t' '
    $1 == "G3" || $1 ~ /^[ABCR]-(VAL|NQ-TEST|MULTIHOP)-EVAL$/ {
        if ($4 != $6 || $5 != $7) exit 1
        seen += 1
    }
    END { exit !(seen == 13) }
' "$main_results/lineage.tsv"
awk -F '\t' -v base_checkpoint="$BASE_MODEL" -v base_digest="$BASE_DIGEST" \
        -v reproduced_checkpoint="$R_CHECKPOINT" -v reproduced_digest="$R_DIGEST" '
    $1 ~ /^A-(VAL|NQ-TEST|MULTIHOP)-EVAL$/ {
        if ($4 != base_checkpoint || $5 != base_digest) exit 1
        parent_seen += 1
    }
    $1 ~ /^R-(VAL|NQ-TEST|MULTIHOP)-EVAL$/ {
        if ($4 != reproduced_checkpoint || $5 != reproduced_digest) exit 1
        reproduced_seen += 1
    }
    END { exit !(parent_seen == 3 && reproduced_seen == 3) }
' "$main_results/lineage.tsv"

# Non-history WandB files remain evidence but cannot satisfy the main-run history gate.
: >"$CALLS"
prepare_preflight main
ANALYSIS_OUTCOME=GO
CONTRAST_COUNT=8
RUN_SUFFIX=wandb-debug-only
WANDB_DEBUG_ONLY_VARIANT=reproduce
debug_only_outer="$TEST_ROOT/project/state/attempts/gpu/20260724T020150Z-10-9"
mkdir -p "$debug_only_outer"
set +e
qwen_native_train_pipeline "$debug_only_outer" "$COMMIT" "$HANDOFF_DIGEST" \
    "$BASE_MODEL" "$BASE_DIGEST"
rc=$?
set -e
[[ "$rc" != 0 ]]
! grep -q '^publish|' "$CALLS"
unset WANDB_DEBUG_ONLY_VARIANT

# A scientific NO-GO still seals all A/R endpoints without creating B or C.
: >"$CALLS"
prepare_preflight main
ANALYSIS_OUTCOME=NO-GO
CONTRAST_COUNT=12
RUN_SUFFIX=no-go
no_go_outer="$TEST_ROOT/project/state/attempts/gpu/20260724T020200Z-10-3"
mkdir -p "$no_go_outer"
qwen_native_train_pipeline "$no_go_outer" "$COMMIT" "$HANDOFF_DIGEST" \
    "$BASE_MODEL" "$BASE_DIGEST"
mapfile -t no_go_jobs < <(grep '^run_job|' "$CALLS")
[[ "${#no_go_jobs[@]}" == 8 ]]
! grep -Eq 'run_job\|train\|(control|cost_aware_gated)' "$CALLS"
no_go_results="$NATIVE_TRAIN_RESULTS_ROOT/attempts/$(basename -- "$no_go_outer")"
grep -Fxq 'stage_order=R60,G3,A-VAL-EVAL,R-VAL-EVAL,A-NQ-TEST-EVAL,R-NQ-TEST-EVAL,A-MULTIHOP-EVAL,R-MULTIHOP-EVAL' \
    "$no_go_results/contract.env"
grep -Fxq 'branch_authorized=false' "$no_go_results/contract.env"
[[ "$(wc -l <"$no_go_results/lineage.tsv")" == 9 ]]
for eval_key in val nq_test multihop; do
    [[ -s "$no_go_results/paired-ar-$eval_key/summary.json" ]]
    [[ ! -e "$no_go_results/paired-$eval_key" ]]
done
grep -q '^publish|.*|qwen-native-training-main|qwen-native-training-main-v3$' "$CALLS"

# GO alone is insufficient when fewer than eight groups provide cost contrast.
: >"$CALLS"
prepare_preflight main
ANALYSIS_OUTCOME=GO
CONTRAST_COUNT=7
RUN_SUFFIX=low-contrast
low_outer="$TEST_ROOT/project/state/attempts/gpu/20260724T020300Z-10-4"
mkdir -p "$low_outer"
qwen_native_train_pipeline "$low_outer" "$COMMIT" "$HANDOFF_DIGEST" \
    "$BASE_MODEL" "$BASE_DIGEST"
[[ "$(grep -c '^run_job|' "$CALLS")" == 8 ]]
! grep -Eq 'run_job\|train\|(control|cost_aware_gated)' "$CALLS"
grep -Fxq 'branch_authorized=false' \
    "$NATIVE_TRAIN_RESULTS_ROOT/attempts/$(basename -- "$low_outer")/contract.env"

# Engineering failure preserves the original nonzero code and never publishes or branches.
: >"$CALLS"
prepare_preflight main
ANALYSIS_OUTCOME=GO
CONTRAST_COUNT=8
RUN_SUFFIX=failed
FAIL_VARIANT=qwen_native_g3
failed_outer="$TEST_ROOT/project/state/attempts/gpu/20260724T020400Z-10-5"
mkdir -p "$failed_outer"
set +e
qwen_native_train_pipeline "$failed_outer" "$COMMIT" "$HANDOFF_DIGEST" \
    "$BASE_MODEL" "$BASE_DIGEST"
rc=$?
set -e
[[ "$rc" == 23 ]]
[[ "$(grep -c '^run_job|' "$CALLS")" == 2 ]]
! grep -Eq 'run_job\|train\|(control|cost_aware_gated)' "$CALLS"
! grep -q '^publish|' "$CALLS"

grep -Fq 'qwen_native_train:qwen35_native' "$GPU_RUN"
grep -Fq 'qwen_native_train_preflight "$commit"' "$GPU_RUN"
grep -Fq 'qwen_native_train_pipeline "$_attempt"' "$GPU_RUN"
grep -Fq 'QWEN_NATIVE_TRAIN_STAGE={smoke|main}' "$ENTRYPOINT"
grep -Fq 'QWEN_NATIVE_PROTOCOL_GATE_EVIDENCE=<g0_g1-marker>' "$ENTRYPOINT"
! grep -Fq 'QWEN_NATIVE_PRETRAIN_G3_EVIDENCE' "$ENTRYPOINT"
! grep -Fq 'delete_gate_checkpoint' "$ENTRYPOINT"
! grep -Eq '(^|[[:space:]])(rm -rf|shutdown|poweroff|systemctl)[[:space:]]' "$ENTRYPOINT"

printf 'Qwen native training pipeline tests passed.\n'
