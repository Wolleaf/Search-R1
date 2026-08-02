# Control vs Cost-aware Efficiency Report

All 2 active stages contain the same 128 test samples. Utility is `EM - 0.10 * searches / 4`.

## Stage Totals

| Role | Stage | N | K | S | EM | Mean searches | E[S|correct] | E[S|wrong] | Utility |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control | qwen_native_b_val | 128 | 79 | 247 | 0.6172 | 1.9297 | 1.6709 | 2.3469 | 0.5689 |
| cost_aware_gated | qwen_native_c_val | 128 | 83 | 215 | 0.6484 | 1.6797 | 1.4940 | 2.0222 | 0.6064 |

## Deterministic Endpoint Contract

Group size `1`, `do_sample=false` (greedy), seed `42`, group slot `0`. B/C preserve the manifest sample order and pair by `typed_sample_id`.
One rollout per question; temperature/top-p `1.0/1.0`, top-k `0`, min-p `0.0`, presence penalty `0.0`, repetition penalty `1.0`.

| Role | Mean actions | Mean trajectory tokens | Mean invalid actions | Invalid trajectory ratio | Clipping ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 3.0391 | 1337.4453 | 0.2344 | 0.1328 | 0.8203 |
| cost_aware_gated | 2.7891 | 1171.4297 | 0.1875 | 0.1562 | 0.8281 |

## B / Control vs cost_aware_gated

| Category | Questions | B / Control searches | Candidate searches | Search delta | Utility delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| Both correct | 74 | 119 | 112 | -7 | +0.1750 |
| B / Control correct, candidate wrong | 5 | 13 | 13 | +0 | -5.0000 |
| B / Control wrong, candidate correct | 9 | 20 | 12 | -8 | +9.2000 |
| Both wrong | 40 | 95 | 78 | -17 | +0.4250 |

Overall: correct-count delta `+4`, search delta `-32`, mean Utility delta `+0.0375`.

Paired bootstrap: `10000` resamples, seed `42`, 95% percentile CI. All estimates are candidate minus B / Control.

| Metric | Estimate | CI lower | CI upper |
| --- | ---: | ---: | ---: |
| em | 0.0312 | -0.0234 | 0.0859 |
| executed_searches | -0.2500 | -0.4062 | -0.1016 |
| correct_only_searches | -0.1769 | -0.3293 | -0.0363 |
| action_count | -0.2500 | -0.4375 | -0.0703 |
| trajectory_tokens | -166.0156 | -299.6119 | -40.6797 |
| invalid_actions | -0.0469 | -0.1953 | 0.0859 |
| clipping_rate | 0.0078 | -0.0703 | 0.0859 |

Search reductions in `both_correct` are lossless efficiency gains. Reductions in `baseline_correct_candidate_wrong` are harmful savings; reductions in `both_wrong` only remove unsuccessful calls.
