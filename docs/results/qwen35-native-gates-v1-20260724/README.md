# Qwen3.5 Native v1 Gate Evidence

This directory preserves the immutable evidence from the first Qwen3.5 native-tool
experiment, before the final-answer contract was changed to native v2. The remote
checkout was `b0563c287b3b8a1203879276bf05af4e7d06fc5b`; both outer GPU attempts ended
with `terminal=success` and `exit-code=0`.

## Result

- G0+G1: **GO**. All 32 first actions were legal and non-degenerate; 45/45
  executed searches had aligned tool responses.
- G2: **NO-GO**. It saved 96 trajectories for 32 questions at group size 3.
  Seven trajectories were invalid, exceeding the registered limit of five;
  one trajectory was clipped.
- Strict EM was 0/96 under the v1 marker-free answer contract. This negative
  result remains historical evidence and is not reinterpreted as a v2 result.

## Evidence Layout

- `raw/runs/qwen-native-gate/`: gate summaries, decisions, normalized records,
  lineage, sampling config, and the original evidence manifests.
- `raw/runs/eval/`: G0 protocol output plus G1/G2 resolved configs, run metadata,
  full trajectories, trace manifests, and logs.
- `raw/state/attempts/gpu/`: exact outer-attempt terminal and shutdown evidence.
- `raw/data/` and `raw/manifests/`: the v1 data contract, fixed probe inputs,
  replay receipt, and predecessor markers required to verify the evidence chain.

The G2 trace SHA-256 is
`6da6f333b3c97a92d34b8b377bde4228ccf26827cfd513785c4d6f210b8f4755`.
Each gate attempt contains its original `evidence.sha256`; `archive.sha256`
checks every byte-preserved file below `raw/`.

Verify the repository archive from this directory with:

```bash
sha256sum -c archive.sha256
```

The archive preserves the gate artifacts and their recorded lineage digests. It
does not contain the full original model tree or every byte of the CPU handoff;
those external objects therefore cannot be reconstructed from this directory.
