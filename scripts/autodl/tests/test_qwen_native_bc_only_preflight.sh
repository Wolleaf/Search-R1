#!/usr/bin/env bash
set -Eeuo pipefail

AUTODL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
CHECKOUT="$(cd -- "$AUTODL_DIR/../.." && pwd -P)"
ENTRYPOINT="$AUTODL_DIR/12_gpu_qwen_native_bc_only.sh"
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

[[ -f "$ENTRYPOINT" && ! -L "$ENTRYPOINT" ]]
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

# Sourcing the operator runner must define the B/C overrides without starting
# CPU preparation or a paid GPU pipeline.
source "$ENTRYPOINT"

COMMIT="$(printf 'c%.0s' {1..40})"
HANDOFF_DIGEST="$(printf 'a%.0s' {1..64})"
BASE_DIGEST="$(printf 'b%.0s' {1..64})"
DATA_DIGEST="$(printf 'd%.0s' {1..64})"
RUNNER_DIGEST="$(printf 'e%.0s' {1..64})"
GATE_DIGEST="$(printf 'f%.0s' {1..64})"
SMOKE_DIGEST="$(printf '1%.0s' {1..64})"

# Replace only recursive filesystem/evidence verification. The preflight
# snapshot and inherited drift checker below remain the production functions.
verify_bc_foundation() {
    local commit="$1" handoff="$2" base_model="$3" base_digest="$4"
    assert_eq "$COMMIT" "$commit" 'foundation commit'
    assert_eq "$HANDOFF_DIGEST" "$handoff" 'foundation handoff'
    assert_eq "$(readlink -f -- "$MODEL_DIR")" "$base_model" \
        'foundation base model'
    assert_eq "$BASE_DIGEST" "$base_digest" 'foundation base digest'
}

verify_bc_cpu_receipt() {
    assert_eq "$RUNNER_DIGEST" "$1" 'CPU receipt runner digest'
}

file_sha256() {
    case "$1" in
        "$BC_RUNNER_PATH")
            printf '%s\n' "$RUNNER_DIGEST"
            ;;
        "$NATIVE_TRAIN_MANIFEST")
            printf '%s\n' "$DATA_DIGEST"
            ;;
        *)
            fail "unexpected file_sha256 input: $1"
            ;;
    esac
}

NATIVE_TRAIN_PROTOCOL_GATE_DIGEST="$GATE_DIGEST"
NATIVE_TRAIN_SMOKE_DIGEST="$SMOKE_DIGEST"
qwen_native_train_preflight "$COMMIT" "$HANDOFF_DIGEST" "$BASE_DIGEST"

assert_eq main "$NATIVE_TRAIN_STAGE" 'native training stage'
assert_eq "$COMMIT" "$BC_PREFLIGHT_COMMIT" 'B/C commit'
assert_eq "$HANDOFF_DIGEST" "$BC_PREFLIGHT_HANDOFF" 'B/C handoff'
assert_eq "$BASE_DIGEST" "$BC_PREFLIGHT_BASE_DIGEST" 'B/C base digest'
assert_eq "$DATA_DIGEST" "$BC_PREFLIGHT_DATA_DIGEST" 'B/C data digest'
assert_eq "$RUNNER_DIGEST" "$BC_PREFLIGHT_RUNNER_DIGEST" \
    'B/C runner digest'

assert_eq "$NATIVE_TRAIN_STAGE" "$NATIVE_TRAIN_PREFLIGHT_STAGE" \
    'inherited stage'
assert_eq "$BC_PREFLIGHT_COMMIT" "$NATIVE_TRAIN_PREFLIGHT_COMMIT" \
    'inherited commit'
assert_eq "$BC_PREFLIGHT_HANDOFF" "$NATIVE_TRAIN_PREFLIGHT_HANDOFF" \
    'inherited handoff'
assert_eq "$BC_PREFLIGHT_BASE_DIGEST" \
    "$NATIVE_TRAIN_PREFLIGHT_BASE_DIGEST" 'inherited base digest'
assert_eq "$BC_PREFLIGHT_DATA_DIGEST" \
    "$NATIVE_TRAIN_PREFLIGHT_DATA_DIGEST" 'inherited data digest'
assert_eq "$GATE_DIGEST" "$NATIVE_TRAIN_PREFLIGHT_PROTOCOL_GATE_DIGEST" \
    'inherited protocol gate digest'
assert_eq "$SMOKE_DIGEST" "$NATIVE_TRAIN_PREFLIGHT_SMOKE_DIGEST" \
    'inherited smoke digest'

verify_native_training_preflight_state "$COMMIT" "$HANDOFF_DIGEST" \
    "$BASE_DIGEST" "$DATA_DIGEST"

# Every inherited snapshot field must independently fail closed on drift.
for field in \
    NATIVE_TRAIN_PREFLIGHT_STAGE \
    NATIVE_TRAIN_PREFLIGHT_COMMIT \
    NATIVE_TRAIN_PREFLIGHT_HANDOFF \
    NATIVE_TRAIN_PREFLIGHT_BASE_DIGEST \
    NATIVE_TRAIN_PREFLIGHT_DATA_DIGEST \
    NATIVE_TRAIN_PREFLIGHT_PROTOCOL_GATE_DIGEST \
    NATIVE_TRAIN_PREFLIGHT_SMOKE_DIGEST; do
    qwen_native_train_preflight "$COMMIT" "$HANDOFF_DIGEST" "$BASE_DIGEST"
    printf -v "$field" '%s' drifted
    if verify_native_training_preflight_state "$COMMIT" "$HANDOFF_DIGEST" \
            "$BASE_DIGEST" "$DATA_DIGEST" >/dev/null 2>&1; then
        fail "inherited preflight accepted drift in $field"
    fi
done

# Exercise the inherited dispatcher without entering gpu_action or any model
# code. It must run the main admission chain, append the data digest as the
# sixth main-pipeline argument, and preserve the scientific pipeline status.
OUTER_ATTEMPT="$PROJECT_ROOT/state/attempts/gpu/test-bc-dispatch"
BASE_MODEL="$(readlink -f -- "$MODEL_DIR")"
NATIVE_PROTOCOL_GATE_EVIDENCE=protocol-marker
NATIVE_SMOKE_EVIDENCE=smoke-marker
NATIVE_TRAIN_STAGE=main
DISPATCH_CALLS=()
MAIN_ARGS=()

require_qwen_native_train() {
    (($# == 0)) || fail 'dispatcher passed arguments to require'
    DISPATCH_CALLS+=(require)
}

verify_protocol_gate_evidence() {
    (($# == 6)) || fail 'dispatcher protocol verifier argument count changed'
    assert_eq "$NATIVE_PROTOCOL_GATE_EVIDENCE" "$1" \
        'dispatcher protocol marker'
    assert_eq "$BASE_MODEL" "$2" 'dispatcher protocol base model'
    assert_eq "$BASE_DIGEST" "$3" 'dispatcher protocol base digest'
    assert_eq "$COMMIT" "$4" 'dispatcher protocol commit'
    assert_eq "$HANDOFF_DIGEST" "$5" 'dispatcher protocol handoff'
    assert_eq "$DATA_DIGEST" "$6" 'dispatcher protocol data digest'
    DISPATCH_CALLS+=(protocol)
}

verify_smoke_evidence() {
    (($# == 7)) || fail 'dispatcher smoke verifier argument count changed'
    assert_eq "$NATIVE_SMOKE_EVIDENCE" "$1" 'dispatcher smoke marker'
    assert_eq "$BASE_MODEL" "$2" 'dispatcher smoke base model'
    assert_eq "$BASE_DIGEST" "$3" 'dispatcher smoke base digest'
    assert_eq "$COMMIT" "$4" 'dispatcher smoke commit'
    assert_eq "$HANDOFF_DIGEST" "$5" 'dispatcher smoke handoff'
    assert_eq "$DATA_DIGEST" "$6" 'dispatcher smoke data digest'
    assert_eq "$GATE_DIGEST" "$7" 'dispatcher smoke gate digest'
    DISPATCH_CALLS+=(smoke)
}

verify_native_training_preflight_state() {
    (($# == 4)) || fail 'dispatcher state verifier argument count changed'
    assert_eq "$COMMIT" "$1" 'dispatcher state commit'
    assert_eq "$HANDOFF_DIGEST" "$2" 'dispatcher state handoff'
    assert_eq "$BASE_DIGEST" "$3" 'dispatcher state base digest'
    assert_eq "$DATA_DIGEST" "$4" 'dispatcher state data digest'
    DISPATCH_CALLS+=(state)
}

qwen_native_smoke_pipeline() {
    fail 'main dispatcher selected the smoke pipeline'
}

qwen_native_main_pipeline() {
    (($# == 6)) || fail 'dispatcher main pipeline argument count changed'
    MAIN_ARGS=("$@")
    DISPATCH_CALLS+=(main)
    return 23
}

set +e
qwen_native_train_pipeline "$OUTER_ATTEMPT" "$COMMIT" "$HANDOFF_DIGEST" \
    "$BASE_MODEL" "$BASE_DIGEST"
dispatcher_rc=$?
set -e

assert_eq 23 "$dispatcher_rc" 'dispatcher main sentinel status'
assert_eq 'require protocol smoke state main' "${DISPATCH_CALLS[*]}" \
    'dispatcher call order'
assert_eq 6 "${#MAIN_ARGS[@]}" 'dispatcher forwarded argument count'
assert_eq "$OUTER_ATTEMPT" "${MAIN_ARGS[0]}" 'dispatcher outer attempt'
assert_eq "$COMMIT" "${MAIN_ARGS[1]}" 'dispatcher main commit'
assert_eq "$HANDOFF_DIGEST" "${MAIN_ARGS[2]}" 'dispatcher main handoff'
assert_eq "$BASE_MODEL" "${MAIN_ARGS[3]}" 'dispatcher main base model'
assert_eq "$BASE_DIGEST" "${MAIN_ARGS[4]}" 'dispatcher main base digest'
assert_eq "$DATA_DIGEST" "${MAIN_ARGS[5]}" 'dispatcher main data digest'

# Publication must fail closed if the long-running B/C job sees checkout, data,
# runner, or its CPU receipt drift before sealing evidence.
PUBLISH_CALLS=()
verify_checkout() {
    assert_eq "$COMMIT" "$1" 'publish checkout commit'
    PUBLISH_CALLS+=(checkout)
}

verify_bc_cpu_receipt() {
    assert_eq "$RUNNER_DIGEST" "$1" 'publish CPU receipt runner digest'
    PUBLISH_CALLS+=(receipt)
}

file_sha256() {
    case "$1" in
        "$NATIVE_TRAIN_MANIFEST")
            printf '%s\n' "$DATA_DIGEST"
            ;;
        "$BC_RUNNER_PATH")
            printf '%s\n' "$RUNNER_DIGEST"
            ;;
        *)
            fail "unexpected publish file_sha256 input: $1"
            ;;
    esac
}

verify_bc_publish_identity "$COMMIT" "$DATA_DIGEST" "$RUNNER_DIGEST"
assert_eq 'checkout receipt' "${PUBLISH_CALLS[*]}" \
    'publication identity call order'

for drift_target in checkout data runner receipt; do
    PUBLISH_CALLS=()
    verify_checkout() {
        PUBLISH_CALLS+=(checkout)
        [[ "$drift_target" != checkout ]]
    }
    verify_bc_cpu_receipt() {
        PUBLISH_CALLS+=(receipt)
        [[ "$drift_target" != receipt ]]
    }
    file_sha256() {
        case "$1" in
            "$NATIVE_TRAIN_MANIFEST")
                [[ "$drift_target" == data ]] &&
                    printf '%s\n' drifted ||
                    printf '%s\n' "$DATA_DIGEST"
                ;;
            "$BC_RUNNER_PATH")
                [[ "$drift_target" == runner ]] &&
                    printf '%s\n' drifted ||
                    printf '%s\n' "$RUNNER_DIGEST"
                ;;
            *)
                fail "unexpected drift file_sha256 input: $1"
                ;;
        esac
    }
    if verify_bc_publish_identity "$COMMIT" "$DATA_DIGEST" \
            "$RUNNER_DIGEST" >/dev/null 2>&1; then
        fail "publication identity accepted $drift_target drift"
    fi
done

printf 'Qwen native B/C-only preflight bridge tests passed.\n'
