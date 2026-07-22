#!/usr/bin/env bash
set -Eeuo pipefail

AUTODL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
ROOT="$(mktemp -d)"
trap 'rm -rf -- "$ROOT"' EXIT
DATA_DIR="$ROOT/data/search_mix"
CHECKOUT="$(cd -- "$AUTODL_DIR/../.." && pwd -P)"
mkdir -p "$DATA_DIR" "$ROOT/envs/train/bin" "$ROOT/models/Qwen3.5-2B" \
    "$ROOT/data/nq_small" "$ROOT/output"
ln -s "$CHECKOUT" "$ROOT/checkout"

cat >"$ROOT/envs/train/bin/python" <<'SH'
#!/usr/bin/env bash
if [[ -n "${CAPTURE_LOG:-}" ]]; then
    printf '%s\n' "$@" >"$CAPTURE_LOG"
    exit 0
fi
exec /usr/bin/python3 "$@"
SH
chmod +x "$ROOT/envs/train/bin/python"

printf 'probe-fixture\n' >"$DATA_DIR/probe_multi_64.parquet"
printf '{"source_id":"hotpotqa:train:1"}\n' >"$DATA_DIR/catalog.jsonl"
PROBE_DIGEST="$(sha256sum "$DATA_DIR/probe_multi_64.parquet" | cut -d' ' -f1)"
CATALOG_DIGEST="$(sha256sum "$DATA_DIR/catalog.jsonl" | cut -d' ' -f1)"
printf '%s\n' \
    "{\"selection_policy\":\"retrieval-verified-search-mix-v1\",\"overlap_checks\":{\"probe_is_hotpot_val_subset\":true},\"artifacts\":{\"probe\":{\"file\":\"probe_multi_64.parquet\",\"rows\":64,\"sha256\":\"$PROBE_DIGEST\"},\"catalog\":{\"file\":\"catalog.jsonl\",\"sha256\":\"$CATALOG_DIGEST\"}}}" \
    >"$DATA_DIR/manifest.json"
MANIFEST_DIGEST="$(sha256sum "$DATA_DIR/manifest.json" | cut -d' ' -f1)"
printf '%s\n' \
    "{\"selection_policy\":\"retrieval-verified-search-mix-v1\",\"manifest_sha256\":\"$MANIFEST_DIGEST\",\"catalog_sha256\":\"$CATALOG_DIGEST\",\"selected_rows\":640}" \
    >"$DATA_DIR/retrieval_replay.json"
(
    cd "$DATA_DIR"
    sha256sum manifest.json >manifest.json.sha256
    sha256sum retrieval_replay.json >retrieval_replay.json.sha256
)

AUTODL_ROOT="$ROOT"
GPU_COUNT=2
AUTODL_PRICE_PER_HOUR=5.76
source "$AUTODL_DIR/07_gpu_group_probe.sh"

require_group_probe
verify_probe_data_contract
[[ "$AUTODL_GPU_PIPELINE" == group_probe && "$EVAL_EXPECTED_ROWS" == 64 &&
    "$EVAL_GROUP_SIZE" == 5 && "$MAX_RESPONSE_LENGTH" == 500 &&
    "$TRAIN_BATCH_SIZE" == 8 ]]

CAPTURE="$ROOT/group-probe-args.txt"
AUTODL_CONFIG_ONLY=1 AUTODL_ROOT="$ROOT" GPU_COUNT=2 \
    OUTPUT_DIR="$ROOT/output" CAPTURE_LOG="$CAPTURE" \
    EVAL_DATA_FILE="$DATA_DIR/probe_multi_64.parquet" EVAL_GROUP_SIZE=5 \
    MAX_RESPONSE_LENGTH=500 \
    bash "$AUTODL_DIR/train_small_grpo.sh" eval group_probe \
        "$ROOT/models/Qwen3.5-2B" >/dev/null
grep -Fxq "data.val_files=$DATA_DIR/probe_multi_64.parquet" "$CAPTURE"
grep -Fxq 'data.val_batch_size=8' "$CAPTURE"
grep -Fxq 'data.eval_group_size=5' "$CAPTURE"
grep -Fxq 'data.max_prompt_length=4096' "$CAPTURE"
grep -Fxq 'data.max_response_length=500' "$CAPTURE"

EVAL_GROUP_SIZE=4
if require_group_probe >/dev/null 2>&1; then
    printf 'Grouped probe accepted an eval group size other than 5.\n' >&2
    exit 1
fi
EVAL_GROUP_SIZE=5

printf 'tampered\n' >>"$DATA_DIR/probe_multi_64.parquet"
if verify_probe_data_contract >/dev/null 2>&1; then
    printf 'Grouped probe accepted a probe file that disagrees with its manifest.\n' >&2
    exit 1
fi
printf 'probe-fixture\n' >"$DATA_DIR/probe_multi_64.parquet"

OUTER_ATTEMPT="$ROOT/state/attempts/gpu/probe-attempt"
RESULTS_DIR="$ROOT/runs/group-probe/attempts/probe-attempt"
EVAL_RUN="$ROOT/runs/eval/group_probe/attempts/eval-attempt"
mkdir -p "$OUTER_ATTEMPT" "$RESULTS_DIR" "$EVAL_RUN/traces"
for file in summary.json summary.md go_no_go.json per_trajectory.jsonl \
    per_question.jsonl lineage.tsv run-index.tsv; do
    printf 'fixture\n' >"$RESULTS_DIR/$file"
done
for file in train.log resolved-config.yaml run.env; do
    printf 'fixture\n' >"$EVAL_RUN/$file"
done
for file in eval_predictions.jsonl eval_predictions.manifest.json \
    eval_predictions.manifest.json.sha256; do
    printf 'fixture\n' >"$EVAL_RUN/traces/$file"
done

publish_group_probe_evidence "$OUTER_ATTEMPT" "$RESULTS_DIR" "$EVAL_RUN"
[[ "$(tr -d '\r\n' <"$OUTER_ATTEMPT/result-contract")" == group-probe-v1 ]]
[[ "$(tr -d '\r\n' <"$ROOT/runs/group-probe/latest")" == "$RESULTS_DIR" ]]
[[ -s "$RESULTS_DIR/evidence.sha256" ]]
EVIDENCE_DIGEST="$(sha256sum "$RESULTS_DIR/evidence.sha256" | cut -d' ' -f1)"
(
    cd "$ROOT"
    sha256sum --strict --check "${RESULTS_DIR#"$ROOT/"}/evidence.sha256" >/dev/null
)
if publish_group_probe_evidence "$OUTER_ATTEMPT" "$RESULTS_DIR" "$EVAL_RUN" \
        >/dev/null 2>&1; then
    printf 'Grouped probe overwrote an immutable evidence marker.\n' >&2
    exit 1
fi
[[ "$(sha256sum "$RESULTS_DIR/evidence.sha256" | cut -d' ' -f1)" == "$EVIDENCE_DIGEST" ]]

grep -Fq 'run_job eval group_probe "$base_model"' "$AUTODL_DIR/07_gpu_group_probe.sh"
grep -Fq -- '--expected-checkpoint-digest "$base_digest"' \
    "$AUTODL_DIR/07_gpu_group_probe.sh"
grep -Fq 'group_probe_pipeline "$_attempt"' "$AUTODL_DIR/03_gpu_run.sh"
grep -Fq 'expected_trace_rows=$((EVAL_EXPECTED_ROWS * EVAL_GROUP_SIZE))' \
    "$AUTODL_DIR/03_gpu_run.sh"
if grep -Fq -- '--fail-on-no-go' "$AUTODL_DIR/07_gpu_group_probe.sh"; then
    printf 'Scientific NO-GO must not become a launcher failure.\n' >&2
    exit 1
fi

printf 'Grouped-probe pipeline tests passed.\n'
