# Qwen Native Gate g0_g1

Decision: **GO**

- Trajectories: 32
- EM: 6/32 (0.188)
- Searches: 91
- Repeated queries: 0
- Non-ASCII queries: 5

## Criteria

- g1_action_token_prefix_integrity_count: 113 == 113 (pass)
- g1_action_tail_leak_count: 0 == 0 (pass)
- g1_info_mask_consistent_count: 32 == 32 (pass)
- g1_observation_policy_token_count: 0 == 0 (pass)
- g1_retrieval_alignment_error_count: 0 == 0 (pass)
- g0_prompt_token_match_count: 16 == 16 (pass)
- g0_first_action_token_match_count: 16 == 16 (pass)
- g0_direct_action_prefix_integrity_count: 16 == 16 (pass)
- g0_manager_action_prefix_integrity_count: 16 == 16 (pass)
- g0_e0_search_roundtrip_count: 1 == 1 (pass)
- g0_e0_retrieved_document_count: 3 == 3 (pass)
- g0_e0_tool_role_count: 1 == 1 (pass)
- g0_e0_mask_leak_count: 0 == 0 (pass)
