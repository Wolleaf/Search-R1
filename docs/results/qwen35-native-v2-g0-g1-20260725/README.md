# Qwen3.5 Native-v2 G0/G1 Evidence Archive

This directory preserves the exact evidence for the fresh native-v2 G0/G1 gate
run completed on 2026-07-25. The process completed successfully and sealed its
outputs, but the scientific decision is **NO-GO**. No G2, G3, smoke training, or
GRPO update was run.

## Experiment Identity

| Item | Value |
| --- | --- |
| Branch | `experiment/hotpot-search-gate` |
| Checkout | `eae57c3ade2a3cad3549f70db1c5dd4e601ec949` |
| Model | `Qwen/Qwen3.5-2B` revision `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| Outer attempt | `20260725T034333Z-1636-28791` |
| G0 attempt | `20260725T034515Z-1685-14070` |
| G1 attempt | `20260725T035137Z-1685-10976` |
| CPU handoff digest | `0490f9bddffa8474de917ac66c507e25688d33e156874356bc663e374804189b` |
| Checkpoint digest | `bc67be20efb353ba14d9c1b291a64410afec94f2310e94b59b6b76047b164e78` |
| G1 trace digest | `f9f78496a94e06829ac9219b2251f04d7cbd320157c145526386727e0c40bd79` |
| Evidence digest | `536e4e49daa4435a80ce73cfe977d068563ece3ff8b7504d47bcd0094b139b1c` |

The outer attempt, G0, and G1 each contain `.success`, `exit-code=0`, and
`terminal=success`. This means the workflow executed correctly; it does not
override the `NO-GO` result in `go_no_go.json`.

## Contents

- `raw/runs/qwen-native-gate/` contains the gate decision, summary, exact
  lineage, sampling contract, per-question analysis, and all 32 embedded G1
  trajectories.
- `raw/runs/eval/` contains the 48-record G0 protocol probe and the original
  32-row G1 trace, resolved configuration, logs, and trace manifest.
- `raw/data/` contains the fixed native-v2 catalog, the exact G0/G1 Parquet
  inputs, and retrieval replay metadata required by the sealed evidence list.
- `raw/state/` and `raw/manifests/` preserve the outer launcher result,
  evidence marker, and shutdown-watchdog records.
- `archive.sha256` covers this README and every regular file under `raw/`.

## Verification Status

All 29 paths registered by the exact `evidence.sha256` are present, and all
29 hashes match. The evidence-list digest equals both the outer
`evidence-digest` and the exact `.ok` marker. Additional checks also confirmed:

- the G1 trace has exactly 32 records and matches its byte count, row count,
  trace manifest, and sidecar digest;
- the 32 traces embedded in `per_trajectory.jsonl` match the original trace in
  order and content;
- independent strict-EM replay returns `2/32`, matching the trace and summary;
- all 16 questions contain exactly group slots `[0, 1]`, with no duplicate
  sample/slot or record IDs;
- the 640-row catalog and fixed 16-question/8-question inputs match the data
  manifest, including question, gold answer, source, order, and file digests.

Run `sha256sum -c archive.sha256` from this directory on Linux or Git Bash to
verify the repository archive itself.

## Evidence Boundary

The W&B offline run is supplementary execution evidence, not part of the
29-file sealed gate manifest. It contains step-0 evaluation only because
`train_steps=0`; there is no optimizer update, training loss curve, or new
checkpoint in this attempt. Empty or duplicate W&B log files are retained only
to preserve the downloaded run layout.

This archive does not include model weights, the BM25 corpus/index, complete
train/validation datasets, or a full environment image. The shutdown record
proves that `/usr/bin/shutdown` was dispatched successfully inside the guest;
it does not independently prove that the AutoDL control plane stopped billing.
