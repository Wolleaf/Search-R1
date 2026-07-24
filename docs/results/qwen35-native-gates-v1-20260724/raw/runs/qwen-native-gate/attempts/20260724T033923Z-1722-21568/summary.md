# Qwen Native Gate g0_g1

Decision: **GO**

- Trajectories: 32
- EM: 0/32 (0.000)
- Searches: 45
- Repeated queries: 0
- Non-ASCII queries: 3

## Criteria

- legal_first_action_count: 32 >= 31 (pass)
- non_degenerate_first_search_count: 32 >= 29 (pass)
- degenerate_first_search_count: 0 <= 1 (pass)
- first_turn_clipped_count: 0 <= 1 (pass)
- aligned_tool_response_count: 45 >= 45 (pass)
- g0_prompt_token_match_count: 16 >= 16 (pass)
- g0_raw_text_match_count: 16 >= 16 (pass)
- g0_direct_parseable_count: 16 >= 15 (pass)
- g0_native_parseable_count: 16 >= 15 (pass)
- g0_native_degenerate_query_count: 0 <= 0 (pass)
