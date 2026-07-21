# Test-128 Results

| Model | NQ EM | Avg searches | No-search ratio | Utility (lambda=0.10) | Train seconds | GPU hours | RMB | Checkpoint | Parent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| A / Base | 0.039 | 1.281 | 0.227 | 0.007 | 0 | 0.00 | 0.00 | /root/autodl-tmp/search-r1/models/Qwen3.5-2B | - |
| R / Reproduced | 0.164 | 1.305 | 0.000 | 0.131 | 22510 | 12.51 | 36.02 | /root/autodl-tmp/search-r1/runs/reproduce/attempts/20260720T065037Z-1472-27656/checkpoints/actor/global_step_60 | /root/autodl-tmp/search-r1/models/Qwen3.5-2B |
| B / Control | 0.180 | 1.023 | 0.000 | 0.154 | 7487 | 4.16 | 11.98 | /root/autodl-tmp/search-r1/runs/control/attempts/20260720T130625Z-1472-2335/checkpoints/actor/global_step_20 | /root/autodl-tmp/search-r1/runs/reproduce/attempts/20260720T065037Z-1472-27656/checkpoints/actor/global_step_60 |
| C / Cost-aware | 0.070 | 0.031 | 0.984 | 0.070 | 5189 | 2.88 | 8.30 | /root/autodl-tmp/search-r1/runs/cost_aware/attempts/20260720T151148Z-1472-18456/checkpoints/actor/global_step_20 | /root/autodl-tmp/search-r1/runs/reproduce/attempts/20260720T065037Z-1472-27656/checkpoints/actor/global_step_60 |
