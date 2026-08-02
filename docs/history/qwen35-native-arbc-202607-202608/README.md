# Qwen3.5 A/R/B/C 实验归档

- 状态：**已完成，历史只读**
- 时间范围：2026-07 至 2026-08
- 后续实验：[B/C Agent 并发实验方案](../../qwen35_native_bc_concurrency_experiment_plan.md)

本目录保存从原始训练失败、工具协议适配、R60 训练与门禁，到 B/C 分叉恢复和 A/R/B/C 补齐评测的完整叙事。历史文档中的云端路径、实例命令和执行入口用于复盘，不应直接当作新实验的启动合同。

## 推荐阅读顺序

1. [完整实验交接](final/qwen35_native_complete_experiment_handoff.md)：先了解实验身份、时间线、产物和最终结论。
2. [A/R/B/C 最终结果分析](final/qwen35_native_arbc_final_results_analysis.md)：查看四模型统一口径下的结果。
3. [B/C 恢复实验完整分析](final/qwen35_native_bc_recovery_complete_analysis_report.md)：查看 B/C 重训、恢复和评测细节。
4. [B/C 后处理执行交接](plans/qwen35_native_bc_posthoc_execution_handoff.md)：需要复核 CPU/GPU 后处理链路时再读。

原始评测和运行证据保存在 [docs/results](../../results/)；归档文档只负责解释实验，不重复保存大文件。

## 阶段与证据

| 阶段 | 归档结论 | 证据目录 |
| --- | --- | --- |
| Search-R1-small | 完成早期 RL 复现，并观察到旧 C 的 no-search collapse | [`search-r1-small-20260720`](../../results/search-r1-small-20260720/) |
| 搜索机会门 | `NO-GO` | [`search-opportunity-gate-20260722`](../../results/search-opportunity-gate-20260722/) |
| grouped probe | `NO-GO`，暴露工具调用协议问题 | [`grouped-probe-20260723`](../../results/grouped-probe-20260723/) |
| native v1 | G0/G1 `GO`，G2 `NO-GO` | [`qwen35-native-gates-v1-20260724`](../../results/qwen35-native-gates-v1-20260724/) |
| native v2 | `NO-GO` | [`qwen35-native-v2-g0-g1-20260725`](../../results/qwen35-native-v2-g0-g1-20260725/) |
| native v3 | 结构门 `GO` | [`qwen35-native-v3-g0-g1-20260726`](../../results/qwen35-native-v3-g0-g1-20260726/) |
| 首次 B/C | B 完成、C 超时，不能形成最终 B/C 结论 | [`qwen35-native-bc-partial-audit-20260731`](../../results/qwen35-native-bc-partial-audit-20260731/) |
| B/C recovery | B/C 恢复完成，并完成三端点评测 | [`qwen35-native-bc-recovery-20260801`](../../results/qwen35-native-bc-recovery-20260801/) |
| A/R 补评 | 补齐 A/R，形成 A/R/B/C 统一结果 | [`qwen35-native-ar-eval-20260802`](../../results/qwen35-native-ar-eval-20260802/) |

部分 `docs/results/` 文件被各自的 `archive.sha256` 覆盖，是不可改写的证据快照。它们可能仍保留归档前的反向文档路径；不要为修导航而重写旧文件或重封旧校验清单，应以本索引中的当前路径为准。

## 最终结论与交接

- [qwen35_native_complete_experiment_handoff.md](final/qwen35_native_complete_experiment_handoff.md)
- [qwen35_native_arbc_final_results_analysis.md](final/qwen35_native_arbc_final_results_analysis.md)
- [qwen35_native_bc_recovery_complete_analysis_report.md](final/qwen35_native_bc_recovery_complete_analysis_report.md)

## 阶段分析

- [工具适配实施报告](stages/qwen35_native_tool_adaptation_implementation_report.md)
- [V2 G0/G1 轨迹分析](stages/qwen35_native_v2_g0_g1_trajectory_analysis.md)
- [V3 G0/G1 轨迹分析](stages/qwen35_native_v3_g0_g1_trajectory_analysis.md)
- [V3 terminal rollout 分析](stages/qwen35_native_v3_terminal_rollout_g0_g1_analysis.md)
- [V4 terminal gate / V5 G0/G1 分析](stages/qwen35_native_v4_terminal_gate_v5_g0_g1_trajectory_analysis.md)
- [V4 two-step smoke 分析](stages/qwen35_native_v4_two_step_smoke_analysis.md)
- [R60 训练与轨迹分析](stages/qwen35_native_r60_training_and_trajectory_analysis.md)
- [R60 G3 评测与轨迹分析](stages/qwen35_native_r60_g3_evaluation_and_trajectory_analysis.md)

## 历史方案与执行记录

- [AutoDL 复现实验方案](plans/autodl_search_r1_reproduction_plan.md)
- [成本感知坍缩分析与改进建议](plans/成本感知坍缩分析与改进建议.md)
- [多跳搜索机会门评测方案](plans/多跳搜索机会门评测方案.md)
- [原始 Search-R1 对齐修复方案](plans/qwen35_search_r1_original_alignment_remediation_plan.md)
- [工具适配方案](plans/qwen35_native_tool_adaptation_plan.md)
- [G1 边界失败修复方案](plans/qwen35_native_g1_boundary_failure_remediation_plan.md)
- [G2 问题与训练解阻方案](plans/qwen35_native_g2_issues_and_training_unblock_plan.md)
- [terminal gold / WandB 修复方案](plans/qwen35_native_terminal_gold_wandb_remediation_plan.md)
- [SFT + RL + B/C 后续方案](plans/qwen35_native_sft_rl_bc_followup_plan.md)
- [B/C 后处理执行交接](plans/qwen35_native_bc_posthoc_execution_handoff.md)
