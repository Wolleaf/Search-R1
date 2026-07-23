# Grouped Probe Analysis

Decision: **NO-GO**

The fixed probe contains 64 questions and 320 sampled trajectories. It found 2 valid correct multi-search trajectories across 2 questions.

## Gate Criteria

| Criterion | Observed | Requirement | Result |
| --- | ---: | ---: | --- |
| `valid_correct_multi_search_count` | 2 | >= 16 | FAIL |
| `covered_question_count` | 2 | >= 8 | FAIL |
| `learnable_group_count` | 2 | >= 8 | FAIL |
| `clipped_ratio` | 0.3187 | <= 0.0500 | FAIL |
| `invalid_action_ratio` | 0.5781 | <= 0.0500 | FAIL |

## Diagnostics

- Learnable groups: 2; cost-contrast groups: 0.
- Clipped trajectories: 102 (31.87%); invalid-action trajectories: 185 (57.81%).
- Near misses: 0 trajectories across 0 questions; cover-EM 0; diagnosis: `correct_multisearch_exploration_absent_or_rare`.
- Category metrics are available in `summary.json`; question- and trajectory-level evidence is in the companion JSONL files.

## Definitions

A valid correct multi-search trajectory has EM=1, at least two searches, no clipping or invalid action, query-token Jaccard below 0.8, a new second-round document, and new supporting-title or first-visible answer evidence.
A learnable group contains at least one such trajectory and at least one clean wrong trajectory, so correctness-gated group reward has non-zero contrast.
A near miss has EM=0 but passes every other valid multi-search condition. The >=16 diagnostic threshold never changes the strict GO/NO-GO decision.
