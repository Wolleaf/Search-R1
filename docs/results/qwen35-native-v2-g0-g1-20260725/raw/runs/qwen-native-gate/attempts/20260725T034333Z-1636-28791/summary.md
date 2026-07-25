# Qwen Native Gate g0_g1

Decision: **NO-GO**

- Trajectories: 32
- EM: 2/32 (0.062)
- Searches: 101
- Repeated queries: 2
- Non-ASCII queries: 7

## Criteria

- legal_first_action_count: 30 >= 31 (fail)
- non_degenerate_first_search_count: 30 >= 29 (pass)
- degenerate_first_search_count: 0 <= 1 (pass)
- first_turn_clipped_count: 0 <= 1 (pass)
- aligned_tool_response_count: 88 >= 101 (fail)
- g0_prompt_token_match_count: 16 >= 16 (pass)
- g0_raw_text_match_count: 16 >= 16 (pass)
- g0_direct_parseable_count: 14 >= 15 (fail)
- g0_native_parseable_count: 14 >= 15 (fail)
- g0_native_degenerate_query_count: 0 <= 0 (pass)
