# Qwen3.5 Native-v3 G0/G1 Evidence

This directory preserves the exact evidence used by
[`qwen35_native_v3_g0_g1_trajectory_analysis.md`](../../qwen35_native_v3_g0_g1_trajectory_analysis.md).
The run evaluated the frozen Qwen3.5-2B parent only: `train_steps=0`, with no
optimizer update and no new checkpoint.

## Run Identity

| Item | Value |
| --- | --- |
| Date | 2026-07-26 |
| Git checkout | `experiment/hotpot-search-gate@6b1623191e6d2929fb9975bfa68acb342d8cf6de` |
| Model revision | `Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc` |
| Outer attempt | `20260726T072318Z-1616-11480` |
| G0/E0 attempt | `20260726T072457Z-1665-26451` |
| G1 attempt | `20260726T072849Z-1665-11820` |
| Gate decision | `GO` |
| Evidence-list digest | `8ce979e0bcc6c4b9bf75ef8a4bca11d9808c7752b9d9fc8b22e37b7f1171ceb2` |

## Contents

- `raw/runs/eval/qwen_native_g0/` contains the direct/native protocol probe and E0 environment replay.
- `raw/runs/eval/qwen_native_g1/` contains the 32 original autonomous trajectories, resolved configuration, and logs.
- `raw/runs/qwen-native-gate/` contains the structural decision and per-trajectory/per-question analysis generated on the instance.
- `raw/state/` and `raw/manifests/` preserve the exact outer terminal, lineage, evidence marker, and shutdown records.
- `raw/data/` preserves the registered probe inputs and dataset manifests needed to identify the evaluated questions.
- `archive.sha256` authenticates every archived file under `raw/` plus this README. Paths are relative to this directory.

## Verified Invariants

- Outer, G0, and G1 all ended with `terminal=success` and `exit-code=0`.
- The remote `evidence.sha256` list verifies all 30 registered files.
- G1 contains 32 rows covering 16 sample IDs and exactly slots `0` and `1` per sample.
- The trace manifest and sidecar match the raw trace SHA-256
  `c2c6701e075d21a74e973be53d7ab492a51932231d689a4ff694a2eaa2361f76`.
- The 32 traces embedded in `per_trajectory.jsonl` match the raw trace records, and strict EM independently recomputes to `6/32`.
- The exact marker and outer `evidence-digest` both match the SHA-256 of the original evidence list.

The structural `GO` confirms protocol, retrieval, token-prefix, and mask
integrity only. It does not certify answer quality or reproduce a training
improvement.
