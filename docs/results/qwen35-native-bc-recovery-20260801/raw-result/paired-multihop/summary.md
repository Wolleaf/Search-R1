# Control vs Cost-aware Efficiency Report

All 2 active stages contain the same 256 test samples. Utility is `EM - 0.10 * searches / 4`.

## Stage Totals

| Role | Stage | N | K | S | EM | Mean searches | E[S|correct] | E[S|wrong] | Utility |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control | qwen_native_b_multihop | 256 | 77 | 787 | 0.3008 | 3.0742 | 2.2987 | 3.4078 | 0.2239 |
| cost_aware_gated | qwen_native_c_multihop | 256 | 76 | 730 | 0.2969 | 2.8516 | 2.1447 | 3.1500 | 0.2256 |

## Deterministic Endpoint Contract

Group size `1`, `do_sample=false` (greedy), seed `42`, group slot `0`. B/C preserve the manifest sample order and pair by `typed_sample_id`.
One rollout per question; temperature/top-p `1.0/1.0`, top-k `0`, min-p `0.0`, presence penalty `0.0`, repetition penalty `1.0`.

| Role | Mean actions | Mean trajectory tokens | Mean invalid actions | Invalid trajectory ratio | Clipping ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 4.1289 | 2068.7188 | 0.5312 | 0.4805 | 0.6523 |
| cost_aware_gated | 3.9961 | 2009.4219 | 0.6055 | 0.5078 | 0.6406 |

## B / Control vs cost_aware_gated

| Category | Questions | B / Control searches | Candidate searches | Search delta | Utility delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| Both correct | 56 | 116 | 117 | +1 | -0.0250 |
| B / Control correct, candidate wrong | 21 | 61 | 66 | +5 | -21.1250 |
| B / Control wrong, candidate correct | 20 | 63 | 46 | -17 | +20.4250 |
| Both wrong | 159 | 547 | 501 | -46 | +1.1500 |

Overall: correct-count delta `-1`, search delta `-57`, mean Utility delta `+0.0017`.

Paired bootstrap: `10000` resamples, seed `42`, 95% percentile CI. All estimates are candidate minus B / Control.

| Metric | Estimate | CI lower | CI upper |
| --- | ---: | ---: | ---: |
| em | -0.0039 | -0.0547 | 0.0430 |
| executed_searches | -0.2227 | -0.3398 | -0.1055 |
| correct_only_searches | -0.1540 | -0.3699 | 0.0539 |
| action_count | -0.1328 | -0.2500 | -0.0156 |
| trajectory_tokens | -59.2969 | -141.3870 | 23.1187 |
| invalid_actions | 0.0742 | -0.0195 | 0.1680 |
| clipping_rate | -0.0117 | -0.0898 | 0.0664 |

Search reductions in `both_correct` are lossless efficiency gains. Reductions in `baseline_correct_candidate_wrong` are harmful savings; reductions in `both_wrong` only remove unsuccessful calls.
