# Parent vs Reproduced Capability Report

All 2 active stages contain the same 128 test samples. Utility is `EM - 0.10 * searches / 4`.

## Stage Totals

| Role | Stage | N | K | S | EM | Mean searches | E[S|correct] | E[S|wrong] | Utility |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| parent | qwen_native_a_val | 128 | 43 | 414 | 0.3359 | 3.2344 | 3.0233 | 3.3412 | 0.2551 |
| reproduced | qwen_native_r_val | 128 | 75 | 255 | 0.5859 | 1.9922 | 1.7600 | 2.3208 | 0.5361 |

## Deterministic Endpoint Contract

Group size `1`, `do_sample=false` (greedy), seed `42`, group slot `0`. A/R preserve the manifest sample order and pair by `typed_sample_id`.
One rollout per question; temperature/top-p `1.0/1.0`, top-k `0`, min-p `0.0`, presence penalty `0.0`, repetition penalty `1.0`.

| Role | Mean actions | Mean trajectory tokens | Mean invalid actions | Invalid trajectory ratio | Clipping ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| parent | 4.3281 | 2173.4844 | 0.5312 | 0.4453 | 0.0625 |
| reproduced | 3.1562 | 1543.2578 | 0.2812 | 0.1797 | 0.3828 |

## A / Parent (post-trained) vs reproduced

| Category | Questions | A / Parent (post-trained) searches | Candidate searches | Search delta | Utility delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| Both correct | 36 | 113 | 65 | -48 | +1.2000 |
| A / Parent (post-trained) correct, candidate wrong | 7 | 17 | 20 | +3 | -7.0750 |
| A / Parent (post-trained) wrong, candidate correct | 39 | 132 | 67 | -65 | +40.6250 |
| Both wrong | 46 | 152 | 103 | -49 | +1.2250 |

Overall: correct-count delta `+32`, search delta `-159`, mean Utility delta `+0.2811`.

Paired bootstrap: `10000` resamples, seed `42`, 95% percentile CI. All estimates are candidate minus A / Parent (post-trained).

| Metric | Estimate | CI lower | CI upper |
| --- | ---: | ---: | ---: |
| em | 0.2500 | 0.1562 | 0.3438 |
| executed_searches | -1.2422 | -1.4922 | -0.9766 |
| correct_only_searches | -1.2633 | -1.6246 | -0.8712 |
| action_count | -1.1719 | -1.4375 | -0.8906 |
| trajectory_tokens | -630.2266 | -795.8609 | -451.3000 |
| invalid_actions | -0.2500 | -0.4453 | -0.0469 |
| clipping_rate | 0.3203 | 0.2266 | 0.4141 |

A/R deltas measure Search-R1 capability change on identical deterministic endpoint samples; they are separate from the B/C cost-efficiency comparison.
