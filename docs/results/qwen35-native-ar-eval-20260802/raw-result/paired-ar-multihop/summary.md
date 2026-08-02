# Parent vs Reproduced Capability Report

All 2 active stages contain the same 256 test samples. Utility is `EM - 0.10 * searches / 4`.

## Stage Totals

| Role | Stage | N | K | S | EM | Mean searches | E[S|correct] | E[S|wrong] | Utility |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| parent | qwen_native_a_multihop | 256 | 56 | 927 | 0.2188 | 3.6211 | 3.2143 | 3.7350 | 0.1282 |
| reproduced | qwen_native_r_multihop | 256 | 95 | 743 | 0.3711 | 2.9023 | 2.4105 | 3.1925 | 0.2985 |

## Deterministic Endpoint Contract

Group size `1`, `do_sample=false` (greedy), seed `42`, group slot `0`. A/R preserve the manifest sample order and pair by `typed_sample_id`.
One rollout per question; temperature/top-p `1.0/1.0`, top-k `0`, min-p `0.0`, presence penalty `0.0`, repetition penalty `1.0`.

| Role | Mean actions | Mean trajectory tokens | Mean invalid actions | Invalid trajectory ratio | Clipping ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| parent | 4.6406 | 2436.0703 | 0.6211 | 0.6016 | 0.2500 |
| reproduced | 4.1680 | 2276.0273 | 0.6914 | 0.4453 | 0.3672 |

## A / Parent (post-trained) vs reproduced

| Category | Questions | A / Parent (post-trained) searches | Candidate searches | Search delta | Utility delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| Both correct | 45 | 145 | 108 | -37 | +0.9250 |
| A / Parent (post-trained) correct, candidate wrong | 11 | 35 | 24 | -11 | -10.7250 |
| A / Parent (post-trained) wrong, candidate correct | 50 | 175 | 121 | -54 | +51.3500 |
| Both wrong | 150 | 572 | 490 | -82 | +2.0500 |

Overall: correct-count delta `+39`, search delta `-184`, mean Utility delta `+0.1703`.

Paired bootstrap: `10000` resamples, seed `42`, 95% percentile CI. All estimates are candidate minus A / Parent (post-trained).

| Metric | Estimate | CI lower | CI upper |
| --- | ---: | ---: | ---: |
| em | 0.1523 | 0.0977 | 0.2109 |
| executed_searches | -0.7188 | -0.8750 | -0.5664 |
| correct_only_searches | -0.8038 | -1.0975 | -0.5040 |
| action_count | -0.4727 | -0.6133 | -0.3320 |
| trajectory_tokens | -160.0430 | -252.0434 | -68.5105 |
| invalid_actions | 0.0703 | -0.0625 | 0.2109 |
| clipping_rate | 0.1172 | 0.0508 | 0.1836 |

A/R deltas measure Search-R1 capability change on identical deterministic endpoint samples; they are separate from the B/C cost-efficiency comparison.
