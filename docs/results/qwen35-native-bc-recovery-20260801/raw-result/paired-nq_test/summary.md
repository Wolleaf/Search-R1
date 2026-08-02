# Control vs Cost-aware Efficiency Report

All 2 active stages contain the same 128 test samples. Utility is `EM - 0.10 * searches / 4`.

## Stage Totals

| Role | Stage | N | K | S | EM | Mean searches | E[S|correct] | E[S|wrong] | Utility |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control | qwen_native_b_nq_test | 128 | 30 | 286 | 0.2344 | 2.2344 | 1.6333 | 2.4184 | 0.1785 |
| cost_aware_gated | qwen_native_c_nq_test | 128 | 31 | 267 | 0.2422 | 2.0859 | 1.6452 | 2.2268 | 0.1900 |

## Deterministic Endpoint Contract

Group size `1`, `do_sample=false` (greedy), seed `42`, group slot `0`. B/C preserve the manifest sample order and pair by `typed_sample_id`.
One rollout per question; temperature/top-p `1.0/1.0`, top-k `0`, min-p `0.0`, presence penalty `0.0`, repetition penalty `1.0`.

| Role | Mean actions | Mean trajectory tokens | Mean invalid actions | Invalid trajectory ratio | Clipping ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 3.3516 | 1543.1562 | 0.4062 | 0.2891 | 0.7109 |
| cost_aware_gated | 3.2656 | 1518.0156 | 0.4531 | 0.3438 | 0.7578 |

## B / Control vs cost_aware_gated

| Category | Questions | B / Control searches | Candidate searches | Search delta | Utility delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| Both correct | 20 | 31 | 25 | -6 | +0.1500 |
| B / Control correct, candidate wrong | 10 | 18 | 20 | +2 | -10.0500 |
| B / Control wrong, candidate correct | 11 | 23 | 26 | +3 | +10.9250 |
| Both wrong | 87 | 214 | 196 | -18 | +0.4500 |

Overall: correct-count delta `+1`, search delta `-19`, mean Utility delta `+0.0115`.

Paired bootstrap: `10000` resamples, seed `42`, 95% percentile CI. All estimates are candidate minus B / Control.

| Metric | Estimate | CI lower | CI upper |
| --- | ---: | ---: | ---: |
| em | 0.0078 | -0.0625 | 0.0781 |
| executed_searches | -0.1484 | -0.3750 | 0.0859 |
| correct_only_searches | 0.0118 | -0.4467 | 0.4903 |
| action_count | -0.0859 | -0.3281 | 0.1484 |
| trajectory_tokens | -25.1406 | -173.2008 | 123.4637 |
| invalid_actions | 0.0469 | -0.1328 | 0.2266 |
| clipping_rate | 0.0469 | -0.0547 | 0.1484 |

Search reductions in `both_correct` are lossless efficiency gains. Reductions in `baseline_correct_candidate_wrong` are harmful savings; reductions in `both_wrong` only remove unsuccessful calls.
