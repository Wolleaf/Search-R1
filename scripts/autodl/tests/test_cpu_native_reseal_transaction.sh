#!/usr/bin/env bash
set -Eeuo pipefail

AUTODL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ROOT="$(mktemp -d)"
trap 'rm -rf -- "$ROOT"' EXIT

"$PYTHON_BIN" - "$AUTODL_DIR/02_cpu_prepare.sh" <<'PY'
from pathlib import Path
import sys

text = Path(sys.argv[1]).read_text(encoding="utf-8")
required = (
    'search_mix_qwen35_native_v4',
    'validate-native-evidence',
    '--native-prompt-version qwen35-native-search-v4-terminal-answer-only',
    '--native-thinking-enabled',
    '--max-action-budget 4',
    '--selection-observation-length 384',
    '--rollout-observation-length 500',
    'run_pytest_file_shards "$train_python" "$CHECKOUT_DIR/tests"',
)
missing = [value for value in required if value not in text]
if missing:
    raise SystemExit(f"CPU native v4 contract is incomplete: {missing}")
validation = text.index('validate-native-evidence')
publication = text.index('"$train_python" "$CHECKOUT_DIR/scripts/autodl/handoff.py" create')
if validation >= publication:
    raise SystemExit("native evidence validation must precede handoff publication")
endpoint_start = text.index("        for variant in \\\n            qwen_native_a_val")
endpoint_end = text.index("        for spec in \\\n            \"smoke|2|\"", endpoint_start)
endpoint_block = text[endpoint_start:endpoint_end]
endpoint_contract = (
    'endpoint_model="$parent_placeholder"',
    'if [[ "$variant" == qwen_native_a_* ]]; then',
    'endpoint_model="$MODEL_DIR"',
    'eval "$variant" "$endpoint_model"',
)
missing_endpoint = [value for value in endpoint_contract
                    if value not in endpoint_block]
if missing_endpoint:
    raise SystemExit(
        f"CPU native endpoint model binding is incomplete: {missing_endpoint}")
if 'eval "$variant" "$parent_placeholder"' in endpoint_block:
    raise SystemExit("native A endpoint configs cannot use the parent placeholder")
PY

MANIFEST_DIR="$ROOT/manifests"
SEAL_DIR="$MANIFEST_DIR/cpu-seals/attempt-1"
ATTEMPT_DIR="$ROOT/state/attempts/cpu/attempt-1"
LOCK_FILE="$ROOT/state/phase.lock"
mkdir -p "$MANIFEST_DIR" "$SEAL_DIR" "$ATTEMPT_DIR"

readonly -a BUNDLE_FILES=(
    cpu_handoff.json
    cpu_handoff.json.sha256
    java-version.txt
    train-freeze.txt
    retriever-freeze.txt
    cpu.ok
)

write_bundle() {
    local directory="$1"
    local payload="$2"
    local version="$3"
    local digest
    printf '%s\n' "$payload" >"$directory/cpu_handoff.json"
    digest="$(sha256sum -- "$directory/cpu_handoff.json" | cut -d' ' -f1)"
    printf '%s  cpu_handoff.json\n' "$digest" \
        >"$directory/cpu_handoff.json.sha256"
    printf 'openjdk version "21.0.%s"\n' "$version" >"$directory/java-version.txt"
    printf 'train-package==%s\n' "$version" >"$directory/train-freeze.txt"
    printf 'retriever-package==%s\n' "$version" >"$directory/retriever-freeze.txt"
    printf '%s\n' "$digest" >"$directory/cpu.ok"
}

verify_bundle() {
    local directory="$1"
    local expected="$2"
    local actual
    actual="$(sha256sum -- "$directory/cpu_handoff.json" | cut -d' ' -f1)"
    [[ "$actual" == "$expected" ]]
    [[ "$(cat "$directory/cpu_handoff.json.sha256")" == \
        "$expected  cpu_handoff.json" ]]
    [[ "$(cat "$directory/cpu.ok")" == "$expected" ]]
    [[ -s "$directory/java-version.txt" ]]
    [[ -s "$directory/train-freeze.txt" ]]
    [[ -s "$directory/retriever-freeze.txt" ]]
}

bundle_fingerprint() {
    local directory="$1"
    local name
    local -a paths=()
    for name in "${BUNDLE_FILES[@]}"; do
        paths+=("$directory/$name")
    done
    sha256sum -- "${paths[@]}" | sha256sum | cut -d' ' -f1
}

write_terminal() {
    local directory="$1" state="$2" rc="$3" marker
    marker=".$state"
    printf '%s\n' "$state" >"$directory/terminal"
    printf '%s\n' "$rc" >"$directory/exit-code"
    : >"$directory/$marker"
    rm -f -- "$directory/.starting" "$directory/.running" \
        "$directory/$([[ "$state" == success ]] && printf .failed || printf .success)"
}

write_bundle "$MANIFEST_DIR" '{"seal":"old"}' old
write_bundle "$SEAL_DIR" '{"seal":"new"}' new
OLD_DIGEST="$(cat "$MANIFEST_DIR/cpu.ok")"
NEW_DIGEST="$(cat "$SEAL_DIR/cpu.ok")"
OLD_FINGERPRINT="$(bundle_fingerprint "$MANIFEST_DIR")"
[[ "$OLD_DIGEST" != "$NEW_DIGEST" ]]

# Invalid candidate metadata must fail before any canonical file is changed.
: >"$SEAL_DIR/train-freeze.txt"
if AUTODL_ROOT="$ROOT" PYTHON_BIN="$PYTHON_BIN" \
        bash "$AUTODL_DIR/02_cpu_prepare.sh" \
        --promote-cpu-seal "$SEAL_DIR" >/dev/null 2>&1; then
    printf 'An invalid CPU seal candidate was promoted.\n' >&2
    exit 1
fi
[[ "$(bundle_fingerprint "$MANIFEST_DIR")" == "$OLD_FINGERPRINT" ]]
[[ ! -e "$MANIFEST_DIR/cpu-seal.pending.json" ]]
[[ ! -e "$SEAL_DIR/previous" ]]
write_bundle "$SEAL_DIR" '{"seal":"new"}' new

# A symlink alias must not bypass the cpu-seals path contract.
ln -s "$SEAL_DIR" "$MANIFEST_DIR/cpu-seals/alias"
if AUTODL_ROOT="$ROOT" PYTHON_BIN="$PYTHON_BIN" \
        bash "$AUTODL_DIR/02_cpu_prepare.sh" \
        --promote-cpu-seal "$MANIFEST_DIR/cpu-seals/alias" >/dev/null 2>&1; then
    printf 'A symlinked CPU seal candidate was promoted.\n' >&2
    exit 1
fi
[[ "$(bundle_fingerprint "$MANIFEST_DIR")" == "$OLD_FINGERPRINT" ]]

# Direct recovery and promotion must honor the same global phase lock.
exec 7>"$LOCK_FILE"
flock -n 7
set +e
AUTODL_ROOT="$ROOT" PYTHON_BIN="$PYTHON_BIN" \
    bash "$AUTODL_DIR/02_cpu_prepare.sh" --recover-cpu-seal \
    >/dev/null 2>&1
LOCK_RC=$?
set -e
[[ "$LOCK_RC" == 75 ]]
flock -u 7

# The cpu-seals root itself must not be accepted through a symlink.
mv "$MANIFEST_DIR/cpu-seals" "$MANIFEST_DIR/cpu-seals-real"
ln -s "$MANIFEST_DIR/cpu-seals-real" "$MANIFEST_DIR/cpu-seals"
if AUTODL_ROOT="$ROOT" PYTHON_BIN="$PYTHON_BIN" \
        bash "$AUTODL_DIR/02_cpu_prepare.sh" \
        --promote-cpu-seal "$SEAL_DIR" >/dev/null 2>&1; then
    printf 'A symlinked CPU seal root was accepted.\n' >&2
    exit 1
fi
unlink "$MANIFEST_DIR/cpu-seals"
mv "$MANIFEST_DIR/cpu-seals-real" "$MANIFEST_DIR/cpu-seals"

# Simulate death after publishing Java metadata but before both freezes and cpu.ok.
set +e
AUTODL_ROOT="$ROOT" PYTHON_BIN="$PYTHON_BIN" AUTODL_TEST_MODE=1 \
    AUTODL_TEST_CPU_SEAL_CRASH_AFTER=2 \
    bash "$AUTODL_DIR/02_cpu_prepare.sh" \
    --promote-cpu-seal "$SEAL_DIR" >/dev/null 2>&1
CRASH_RC=$?
set -e
[[ "$CRASH_RC" == 91 ]]
[[ -f "$MANIFEST_DIR/cpu-seal.pending.json" ]]

# Startup recovery rolls an incomplete publication back to the verified old seal.
AUTODL_ROOT="$ROOT" PYTHON_BIN="$PYTHON_BIN" \
    bash "$AUTODL_DIR/02_cpu_prepare.sh" --recover-cpu-seal
verify_bundle "$MANIFEST_DIR" "$OLD_DIGEST"
[[ "$(bundle_fingerprint "$MANIFEST_DIR")" == "$OLD_FINGERPRINT" ]]
[[ ! -e "$MANIFEST_DIR/cpu-seal.pending.json" ]]

# Retrying the same candidate leaves a journal until the outer attempt succeeds.
AUTODL_ROOT="$ROOT" PYTHON_BIN="$PYTHON_BIN" \
    bash "$AUTODL_DIR/02_cpu_prepare.sh" --promote-cpu-seal "$SEAL_DIR"
[[ -f "$MANIFEST_DIR/cpu-seal.pending.json" ]]
verify_bundle "$MANIFEST_DIR" "$NEW_DIGEST"
write_terminal "$ATTEMPT_DIR" success 0
AUTODL_ROOT="$ROOT" PYTHON_BIN="$PYTHON_BIN" \
    bash "$AUTODL_DIR/02_cpu_prepare.sh" --recover-cpu-seal
for name in "${BUNDLE_FILES[@]}"; do
    cmp -s "$MANIFEST_DIR/$name" "$SEAL_DIR/$name"
done
[[ ! -e "$MANIFEST_DIR/cpu-seal.pending.json" ]]

# A failure after all canonical writes still rolls back to the last committed seal.
SECOND_SEAL="$MANIFEST_DIR/cpu-seals/attempt-2"
SECOND_ATTEMPT="$ROOT/state/attempts/cpu/attempt-2"
mkdir -p "$SECOND_SEAL" "$SECOND_ATTEMPT"
write_bundle "$SECOND_SEAL" '{"seal":"uncommitted"}' uncommitted
AUTODL_ROOT="$ROOT" PYTHON_BIN="$PYTHON_BIN" \
    bash "$AUTODL_DIR/02_cpu_prepare.sh" --promote-cpu-seal "$SECOND_SEAL"
[[ -f "$MANIFEST_DIR/cpu-seal.pending.json" ]]
write_terminal "$SECOND_ATTEMPT" failed 23
AUTODL_ROOT="$ROOT" PYTHON_BIN="$PYTHON_BIN" \
    bash "$AUTODL_DIR/02_cpu_prepare.sh" --recover-cpu-seal
verify_bundle "$MANIFEST_DIR" "$NEW_DIGEST"

# Recovery is idempotent once no transaction is pending.
AUTODL_ROOT="$ROOT" PYTHON_BIN="$PYTHON_BIN" \
    bash "$AUTODL_DIR/02_cpu_prepare.sh" --recover-cpu-seal
verify_bundle "$MANIFEST_DIR" "$NEW_DIGEST"

printf 'CPU native reseal transaction tests passed.\n'
