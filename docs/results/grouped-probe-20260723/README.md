# RL Parent Grouped Probe 结果归档

## 结论

本轮结论为 **NO-GO**，不启动 `R-mix60`，也不实现或训练后续 `B-mix20/C-gated-mix20`。固定的 64 道 held-out HotpotQA 题按每题 5 条轨迹采样，共得到 320 条完整轨迹；仅有 2 条“答对且有效多搜”轨迹，覆盖 2 道题，低于预注册的 16 条/8 题门槛。near-miss 为 0，说明问题不只是答案抽取格式，而是当前 parent 与自定义工具协议尚未形成稳定的正确多跳搜索探索。

这里的 parent 是官方 post-trained `Qwen/Qwen3.5-2B`，不是原始预训练 `Qwen/Qwen3.5-2B-Base`。历史 stage 名 `A/Base` 只表示“未经过本项目 Search-R1 RL”。补充分析还定位到提示词占位符复制，以及恢复文案被宽松正则解析为 `and` 搜索的确定性反馈环；详见 [`analysis_zh.md`](analysis_zh.md)。

完整归因、两条正例、论文规模对照和面试口径见 [`analysis_zh.md`](analysis_zh.md)；Qwen 原生 `<tool_call>` 与当前 `<search>` 协议、采样配置和最小验证方案的专项审计见 [`qwen35_tool_protocol_audit_zh.md`](qwen35_tool_protocol_audit_zh.md)。

## 运行与恢复

- 评测 commit：`5a3bfb82a3c0ded8b4b91d22d6b8a026a855e067`
- 两张 RTX 5090，`response_length=500`、`max_searches=4`、`group_size=5`
- GPU 评测耗时 5,073 秒，按 5.76 元/小时折算约 8.12 元，不含实例启停空闲时间
- 评测 attempt 成功，320 条 trace 的 SHA-256 为 `39c8332e429ccdf438b2edf5d77dfedabf960845efdb4102baf5cb0b71d06382`
- 原外层 attempt 在评测完成后因第 28 条轨迹包含模型真实生成的空 `<search>` query 而失败；原失败状态被完整保留
- 分析器 commit `975e40298c67107c47fdf7d37176f23c7e64d39e` 只把字符串型空 query 从“文件损坏”改为 `near_duplicate_or_empty_query` 科学失败；本地 17 项测试通过
- 新的离线分析 attempt 独立成功，未重跑 GPU 评测，也未改训练、数据、检索或采样参数

## 核心指标

| 指标 | 结果 | 预注册要求 |
| --- | ---: | ---: |
| EM | 7/320（2.19%） | 诊断项 |
| 有效正确多搜轨迹 | 2/320（0.63%） | >= 16 |
| 覆盖题数 | 2/64 | >= 8 |
| 可学习 group | 2/64 | >= 8 |
| cost-contrast group | 0/64 | 后续成本实验需要 |
| 截断轨迹 | 102/320（31.88%） | <= 5% |
| 非法动作轨迹 | 185/320（57.81%） | <= 5% |
| near-miss | 0/320 | 诊断项 |

实际搜索次数分布为 `0/1/2/3/4 = 140/90/53/32/5`，均值为 0.975。7 条正确轨迹中，搜索次数分布为 `0/1/2 = 3/2/2`；没有正确的三搜或四搜轨迹。因此当前 parent 既没有足够的正确多搜奖励信号，也没有同题答对轨迹之间可用于成本优化的搜索次数差异。

## 证据索引

- `eval/`：成功评测的 resolved config、日志、终态和 320 条原始轨迹；轨迹含模型生成的 `<think>` 文本、query、检索结果、答案、截断与非法动作
- `results/summary.json` 与 `results/summary.md`：聚合指标和门禁结论
- `results/per_question.jsonl`：64 个 group 的逐题结果
- `results/per_trajectory.jsonl`：320 条规范化逐轨迹诊断
- `failed-phase/`：原分析器严格校验导致的真实失败现场
- `analysis/` 与 `results/recovery-lineage.tsv`：离线恢复 attempt 及评测/分析 commit 血缘

该 NO-GO 只说明当前 parent、当前协议和预注册的 60-step 低成本方案没有达到启动条件，不证明 Qwen3.5 不会工具调用，也不证明论文的长程 Search-R1 或成本感知方法普遍无效。若后续继续，应建立新的明确假设和 commit，而不是事后降低本次门槛或重复采样直到通过。
