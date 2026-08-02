# 文档导航

`docs/` 根目录只保留当前实验入口。已完成实验的结论、过程记录和旧方案统一归档，避免新旧实验身份混淆。

## 当前实验

- [B/C Agent 并发实验方案](qwen35_native_bc_concurrency_experiment_plan.md)：下一阶段的冻结实验设计与实施边界。

## 已完成的 Qwen3.5 A/R/B/C 实验

- [归档索引](history/qwen35-native-arbc-202607-202608/README.md)：完整阅读顺序、最终结论、阶段分析和历史方案。
- [最终结果分析](history/qwen35-native-arbc-202607-202608/final/qwen35_native_arbc_final_results_analysis.md)：A、R、B、C 的统一对比结果。
- [完整实验交接](history/qwen35-native-arbc-202607-202608/final/qwen35_native_complete_experiment_handoff.md)：实验身份、执行过程、证据位置和恢复说明。
- [原始结果证据](results/)：评测表、日志、环境快照与可复核产物。

## 通用参考

- [Search-R1 论文](reference/2503.09516v5.pdf)
- [Retriever 说明](reference/retriever.md)
- [多机训练说明](reference/multinode.md)
- [上游实验日志](reference/experiment_log.md)

## 归档约定

- 新实验的活动方案可以放在 `docs/` 根目录。
- 实验完成后，将最终报告、阶段分析和已失效方案移入 `docs/history/<experiment-id>/`。
- 大体量但需要复盘的机器可读证据保留在 `docs/results/<experiment-id>/`，不复制到叙事文档目录。
- 历史文档保持原意，只修复因归档产生的路径和导航问题；其中的旧命令不等于当前推荐入口。
