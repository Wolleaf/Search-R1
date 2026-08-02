# Parent vs Reproduced Capability Report

All 2 active stages contain the same 128 test samples. Utility is `EM - 0.10 * searches / 4`.

## Stage Totals

| Role | Stage | N | K | S | EM | Mean searches | E[S|correct] | E[S|wrong] | Utility |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| parent | qwen_native_a_nq_test | 128 | 10 | 349 | 0.0781 | 2.7266 | 1.5000 | 2.8305 | 0.0100 |
| reproduced | qwen_native_r_nq_test | 128 | 32 | 323 | 0.2500 | 2.5234 | 1.7188 | 2.7917 | 0.1869 |

## Deterministic Endpoint Contract

Group size `1`, `do_sample=false` (greedy), seed `42`, group slot `0`. A/R preserve the manifest sample order and pair by `typed_sample_id`.
One rollout per question; temperature/top-p `1.0/1.0`, top-k `0`, min-p `0.0`, presence penalty `0.0`, repetition penalty `1.0`.

| Role | Mean actions | Mean trajectory tokens | Mean invalid actions | Invalid trajectory ratio | Clipping ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| parent | 3.9219 | 1975.3125 | 0.6875 | 0.5000 | 0.1562 |
| reproduced | 3.7500 | 1940.6875 | 0.5312 | 0.3672 | 0.4062 |

## A / Parent (post-trained) vs reproduced

| Category | Questions | A / Parent (post-trained) searches | Candidate searches | Search delta | Utility delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| Both correct | 7 | 8 | 10 | +2 | -0.0500 |
| A / Parent (post-trained) correct, candidate wrong | 3 | 7 | 8 | +1 | -3.0250 |
| A / Parent (post-trained) wrong, candidate correct | 25 | 74 | 45 | -29 | +25.7250 |
| Both wrong | 93 | 260 | 260 | +0 | -0.0000 |

Overall: correct-count delta `+22`, search delta `-26`, mean Utility delta `+0.1770`.

Paired bootstrap: `10000` resamples, seed `42`, 95% percentile CI. All estimates are candidate minus A / Parent (post-trained).

| Metric | Estimate | CI lower | CI upper |
| --- | ---: | ---: | ---: |
| em | 0.1719 | 0.1016 | 0.2500 |
| executed_searches | -0.2031 | -0.4924 | 0.1016 |
| correct_only_searches | 0.2188 | -0.6700 | 1.0631 |
| action_count | -0.1719 | -0.4766 | 0.1406 |
| trajectory_tokens | -34.6250 | -222.9930 | 155.6984 |
| invalid_actions | -0.1562 | -0.3984 | 0.0859 |
| clipping_rate | 0.2500 | 0.1484 | 0.3516 |

A/R deltas measure Search-R1 capability change on identical deterministic endpoint samples; they are separate from the B/C cost-efficiency comparison.
