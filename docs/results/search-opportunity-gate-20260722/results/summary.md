# Search Opportunity Gate

**Decision: NO-GO**

This is a baseline-only audit of 128 HotpotQA and 128 2WikiMultiHopQA questions. It does not compare checkpoints or prove that any search was redundant.

## Baseline Metrics

| Questions | Correct | EM | Mean searches | P(S>=2) | P(S>=3) | Clipped | Invalid-action questions |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 61 | 0.2383 | 1.0312 | 0.0312 | 0.0000 | 0.1172 | 0.1172 |

## Gate Criteria

| Criterion | Observed | Rule | Pass |
| --- | ---: | ---: | :---: |
| total_correct | 61 | >= 20 | yes |
| correct_with_two_plus_searches_count | 0 | >= 8 | no |
| correct_with_two_plus_searches_ratio | 0 | >= 0.2 | no |
| all_with_three_plus_searches_count | 0 | >= 10 | no |
| redundancy_candidate_count | 0 | >= 5 | no |
| candidate_ratio_among_three_plus | n/a | >= 0.3 | no |
| clipped_ratio | 0.117188 | <= 0.05 | no |
| invalid_action_ratio | 0.117188 | <= 0.05 | no |

## Evidence Notes

- `per_question.jsonl` preserves every question, final answer, full raw trajectory, structured turns, retrieval events, and heuristic redundancy evidence.
- `hop_proxy` is only the count of distinct annotated supporting-fact titles; it is not a semantic or oracle hop count.
- Strong/medium labels inspect only observable third-search query similarity, document novelty, and whether answer text was already present before that search.
