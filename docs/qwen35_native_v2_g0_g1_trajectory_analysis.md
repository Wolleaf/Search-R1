# Qwen3.5 Native-v2 G0/G1 轨迹分析与训练决策

> 分析日期：2026-07-25
>
> 对照文档：[Qwen3.5 Native G2 问题清单与训练解阻计划](qwen35_native_g2_issues_and_training_unblock_plan.md)
>
> 精确证据：[qwen35-native-v2-g0-g1-20260725](results/qwen35-native-v2-g0-g1-20260725/README.md)
>
> 当前结论：**工程执行成功，科学门禁 NO-GO；不得进入 G2 或训练。**

## 1. 结论先行

这次不是 OOM、CUDA、双卡、BM25 或脚本崩溃。外层任务、G0 和 G1 都是 `exit-code=0`、`terminal=success`，证据也已完整封存。真正的结果是：native-v2 首次产生了可用于原版 strict EM 的非零正奖励，但格式遵循和停止策略仍不稳定，因此预注册门禁正确地阻止了后续训练。

最重要的四个结论是：

1. **冻结 parent 在 forced-search G1 中能执行真实多搜。** 32 条轨迹执行了 88 次检索，平均 2.75 次；28/32 至少检索两次，14 条形成完整二搜链。当前尚未运行 autonomous v2 G2，不能外推为自主能力门已通过。
2. **检索链路没有丢响应。** 88 次实际检索都有 retrieval event、非空 observation 和 3 篇文档。`88/101` 中缺少的 13 次不是 BM25 故障，而是额外收尾生成回合仍请求 search，但环境按设计不再执行。
3. **完整 v2 合同证明 tagged short-answer 抽取和非零 strict signal 可行。** v1 的 strict EM 为 0/32；v2 首次得到 2/32。可是仅 8/32 生成可抽取答案，24/32 没有答案，21/32 出现至少一次非法动作；由于多项接口和采样同时变化，不能把这 2 个正例归因于 `<answer>` 单一变量。
4. **现在不能直接训练。** G0 direct/native 可解析均为 14/16，G1 合法首动作 30/32，且模型大量不能及时终止。此时启动 GRPO 会把预算消耗在稀疏、噪声较大的组内信号上。

所以更准确的项目判断不是“Qwen3.5-2B 不会搜索”，也不是“已经可以训练”，而是：

> 搜索、工具回填和 strict reward 链路均已打通；当前阻塞从“没有奖励信号”收敛为“单动作格式、短答案标签和终止收敛不稳定”。

## 2. 本次到底运行了什么

| 项目 | 精确值 |
| --- | --- |
| 分支 / checkout | `experiment/hotpot-search-gate@eae57c3ade2a3cad3549f70db1c5dd4e601ec949` |
| 模型 | `Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc` |
| 外层 attempt | `20260725T034333Z-1636-28791` |
| G0 attempt | `20260725T034515Z-1685-14070` |
| G1 attempt | `20260725T035137Z-1685-10976` |
| G0 规模 | 8 题 × 2 槽 × direct/native/legacy = 48 条协议记录 |
| G1 规模 | 固定 16 题 × 2 槽 = 32 条 forced-search 轨迹 |
| 采样 | temperature 1.0、top-p 1.0、top-k 0、presence penalty 0 |
| 轨迹容量 | response 500、observation 384、`max_turns=4`，必要时再生成一个 terminal action |
| 实际更新 | `train_steps=0`，没有训练、loss 曲线或新 checkpoint |
| G1 耗时 | 193 秒；G0 382 秒；外层从请求到封存约 12 分 39 秒 |

外层 evidence manifest 的 SHA-256 为：

```text
536e4e49daa4435a80ce73cfe977d068563ece3ff8b7504d47bcd0094b139b1c
```

它与 outer `evidence-digest` 和 exact `.ok` marker 完全一致。归档最初漏下载的两个 retrieval replay 文件已从无卡实例补齐；现在封存清单为 29/29 文件存在、29/29 哈希匹配。

## 3. 相对旧计划实际改了什么

旧文档的主要目标是恢复论文的最终答案合同，并消除 Qwen native 采样与 PPO log-prob 的明显不一致。本次实际落地如下。

| 项目 | native-v1 | 本次 native-v2 |
| --- | --- | --- |
| prompt 版本 | `qwen35-native-search-v1` | `qwen35-native-search-v2-answer-tag` |
| 最终答案 | 普通 assistant 文本整段作为答案 | 唯一且闭合的 `<answer>短答案</answer>` |
| 无标签文本 | 直接作为合法 answer | `missing_native_action`，进入 retry |
| thinking 前缀 | empty-think 边界不稳定 | 只容忍一个空 `<think></think>`，随后允许无协议 marker 的 reasoning prefix |
| 非法动作 retry | search 或 marker-free short answer | search 或一个严格 `<answer>` |
| 采样 | top-k 20、presence penalty 2 | top-k 0、presence penalty 0 |
| response | 500 | 保持 500 |
| 数据 | native-v1 prompt | 同一批样本只重物化 prompt，不重新选题 |
| 正确性检查 | 主要信任 trace 中 EM | 新增 G2/G3 自动 replay；本次 G1 在归档审计时独立复算，SubEM 只写入本报告 |
| 流程 | G0/G1/G2 | fresh G0+G1 失败即停止；未复用旧 G2 |

当前 forced-search system prompt 的核心文本是：

```text
Call at most one tool per assistant turn. Use search when external
evidence is needed. After each search result, decide whether another
search is needed. Use at most four searches. When you have enough
evidence, output the opening tag <answer>, then only the short final
answer text, then the closing tag </answer>, and end the response.
Do not combine a tool call with a final answer in the same assistant
response, and do not output any text after the closing tag.
```

G1 的 user 消息还会在问题前加：

```text
Call search at least once before answering.
```

搜索仍使用 Qwen3.5 原生 `<tool_call>`，检索结果仍以 native tool role 回填，`enable_thinking=False` 保持不变。也就是说，这次没有改 BM25、top-3、问题选择、reward 公式、GRPO 算法或最大搜索轮数；主要改的是 Qwen 接口合同及与训练一致的采样。

旧文档中“共 760 个样本”的说法需要纠正：唯一 catalog 是 640 条，即 train-512 + val-128；G0/G1/G2/G3 probe 都是其中的固定子集，不能重复加到唯一题数。

## 4. native-v1 与 native-v2 的直接对照

两次 G1 使用相同 16 道题、相同 2 个采样槽、相同 checkpoint、response 500、原生工具 schema 和 forced-search 条件，因此可以做端到端行为对照。

| 指标 | native-v1 G1 | native-v2 G1 | 解释 |
| --- | ---: | ---: | --- |
| G0 direct/native 可解析 | 16/16、16/16 | 14/16、14/16 | v2 同一道双实体题的两个槽都生成多 tool call |
| 合法首动作 | 32/32 | 30/32 | v2 有两个并行多调用首动作 |
| 实际执行搜索 | 45 | 88 | 增加 43 次 |
| 每轨迹平均实际搜索 | 1.406 | 2.75 | 接近翻倍 |
| 可抽取最终答案 | 32/32 | 8/32 | v1 的 32 条是宽松 marker-free 合同，不代表都是短答案 |
| 含非法动作的轨迹 | 0/32 | 21/32 | v2 输出分布和 fail-closed 合同共同作用下的记录值 |
| terminal generation 请求 search | 0 | 13 | v2 的主要停止问题 |
| strict EM | 0/32 | 2/32 | 首次得到非零严格奖励 |
| 生成 token 中位数 | 93 | 159.5 | 更多重试和搜索伴随轨迹变长 |

这张表不能被解读为单变量消融。v2 同时改变了答案标签、parser、retry、top-k 和 presence penalty；尤其 marker-free 文本在 v1 会直接终止，在 v2 会被拒绝并继续 rollout。因此：

- `45 -> 88` 既包含真实的持续检索，也包含普通答案被拒后的重试搜索，以及最终不会停；
- `0 -> 2` 证明完整 v2 合同能产生 strict reward，但不能断言全部增益只来自 `<answer>`；
- checkpoint 是冻结的，本轮没有学习，不能写成“模型训练后学会了多搜”。

旧 native-v1 G2 是 32 题 × 3 槽的 96 条 autonomous 轨迹，与本轮 G1 不是同一阶段。旧 G2 只能作为背景证据，不能与当前 32 条 forced-search 直接拼接或比较百分比。

## 5. G0：manager 没有改坏输出，失败来自模型动作形态

G0 的 direct HF 与 native manager 在 16 个槽位上：

- prompt token hash 16/16 相同；
- raw output 16/16 逐字相同；
- parsed action 16/16 相同；
- 两条路径都只有 `hotpotqa:train:19652` 的两个槽不可解析。

这道题询问 Lisa Cholodenko 和 Pierre Morel 的共同职业。slot 0 在单个 assistant turn 连续输出了 3 个完整 search 调用：

```text
Lisa Cholodenko occupation
Pierre Morel occupation
Article about Lisa Cholodenko and Pierre Morel shared experience
```

slot 1 连续输出了 2 个完整调用。错误均为 `multiple_or_unbalanced_tool_calls`；这里是“多个闭合调用”，不是缺标签。native manager 每个槽都用满 5 个生成动作（首轮加 4 次 retry），但模型始终倾向批量调用，最终没有执行检索。

因此可以排除“manager 重写了 prompt 或 raw output”这一假设。更可信的解释是双实体问题触发了 Qwen 的并行工具调用倾向，而 Search-R1 环境要求严格的“每轮一次 search -> 回填 observation -> 再决策”。parser 拒绝多调用是正确的，不能为了过门禁直接把多个 query 静默合并。

值得注意的是，v1 在同一道题的两个槽都只生成一次合法调用。由于 v2 同时改了 prompt 和采样，现有证据无法判断回归究竟由哪一个变量单独造成。

## 6. G1 全量统计

### 6.1 搜索与终止

| 指标 | 数值 |
| --- | ---: |
| 轨迹 | 32 |
| 实际搜索次数分布 0/1/2/3/4 | 1 / 3 / 6 / 15 / 7 |
| 实际搜索总数 | 88 |
| 每轨迹平均搜索 | 2.75 |
| 至少二搜 | 28/32 |
| 完整二搜链 | 14（分析器还要求全部 parsed search 均有真实回填且 query 不重复） |
| parsed search action | 101 |
| terminal generation 的未执行 search | 13 |
| 用满 5 个 generation action | 26/32 |

`101` 是所有被 parser 识别为 search 的生成动作数；`88` 才是真正交给 BM25 的请求数。逐条对齐显示：

- 88/88 实际搜索都有 retrieval event；
- 88/88 都有非空 observation；
- 88/88 都返回 3 篇文档；
- query、turn、event identity 全部一致。

另外 13 个动作全部位于零基 `turn=4=max_turns` 的额外 terminal generation。这个回合调用环境时固定 `do_search=False`，所以模型虽然输出了合法 search action，检索器不会再执行。不能笼统称它们为“第 5 次搜索”：此前实际搜索数分布是 2 次 × 1 条、3 次 × 7 条、4 次 × 5 条；准确说法是“第 5 个生成动作仍请求 search”。

13 条中有 5 条此前完全没有 parser invalid，8 条此前有 invalid。这说明终止失败不只是 retry 的副作用；即便前四轮语法全合法，模型也可能拿到答案证据后继续搜。

预注册 `aligned_tool_response_count >= search_turn_count` 因而仍然正式失败，不能事后改判为 GO。但该指标把两件事混在了一起：

1. 检索基础设施对真实请求是否完整回填：本轮为 88/88，健康；
2. terminal 时模型是否停止请求工具：本轮 13 条失败。

下一版分析器应分别命名和报告这两个指标，但不能用新名字追溯修改本次 NO-GO。

### 6.2 答案、EM 与非法动作

| 指标 | 数值 |
| --- | ---: |
| 可抽取 `<answer>` | 8/32 |
| 无最终答案 | 24/32 |
| strict EM | 2/32 = 6.25% |
| 诊断 SubEM | 3/32 |
| 含至少一个非法动作 | 21/32 |
| 非法事件总数 | 38 |
| response clipping | 0 |
| generated tokens 均值 / 中位数 / 最大值 | 169.47 / 159.5 / 444 |

38 个非法事件的分布为：

| 错误 | 次数 | 含义 |
| --- | ---: | --- |
| `missing_native_action` | 23 | 普通文本既不是 tool call，也没有 `<answer>` |
| `multiple_or_unbalanced_tool_calls` | 9 | 同回合多个调用或标签不平衡 |
| `unknown_tool` | 4 | 模型发明 `question_tags` 或 `check` |
| `multiple_or_unbalanced_answers` | 2 | answer 标签数量不对或未闭合 |

24 条无答案轨迹中，19 条曾有 invalid，另有 5 条从头到尾动作语法都合法，却一直 search 到 terminal。8 条有答案轨迹中，6 条全程合法，2 条经过 retry 后才输出答案。

本轮没有任何轨迹触及 500-token response 上限，最大值为 444。因此可以排除“本轮主要因为 response=500 太短而截断”；不能由此泛化为所有后续训练都永远不存在上下文容量问题。

strict EM 会小写化、去 ASCII 标点和英文冠词并压缩空白，然后要求完整字符串相等。SubEM 只检查归一化 gold 是否包含在 prediction 中，是离线诊断，不是 reward、语义 judge 或正式门禁。当前 2 个 strict mixed group 是 `86102` 和 `88825`，但两组错误槽都含 invalid，所以 `per_question.jsonl` 中仍为 `learnable=false`。最多只能说“出现了两个组内 strict reward 差异的小样本”，不能写成“已经有两个合格训练组”。

另一个有价值的诊断是：13 个 `missing_native_action` 事件横跨 8 条轨迹，其归一化普通文本满足与 SubEM 相同的 gold containment；字面不区分大小写包含则是 12 个事件、7 条轨迹。另有 11 次真实搜索发生在一个 marker-free 答案被拒之后。这证明 v2 在严格合同下暴露了边界问题，同时也解释了部分额外搜索从何而来。

## 7. 真实轨迹案例

以下案例都来自本次封存的 `per_trajectory.jsonl` 和原始 32 行 trace。表格省略完整 BM25 文档正文，但保留每个生成动作、实际 query、关键证据、parser 结果和终局。

### 7.1 同题一条成功、一条失败：Ryan Neates

问题：Ryan Neates 当前所属俱乐部的官方颜色是什么？gold 为 `navy blue and gold`。

成功轨迹 `trace:146d5f886055a18a2254fcfd`，slot 1：

| Turn | 模型动作 | 环境结果 |
| ---: | --- | --- |
| 0 | search `Ryan Neates football club current team` | top doc 明确其当前效力 Claremont |
| 1 | search `Claremont Football Club official colours` | top doc 原文为 `Its official colours are navy blue and gold.` |
| 2 | search `West Coast Eagles official colours` | 成功检索，但对当前俱乐部已属冗余 |
| 3 | `<answer>Navy blue and gold</answer>` | 合法终止，strict EM=1 |

这条轨迹执行 3 次搜索、0 invalid。它证明 native-v2 能形成“实体定位 -> 属性查询 -> 严格短答案”的真实正奖励；同时第三搜是清晰的成本优化空间。

失败轨迹 `trace:63b1f074bb97a71d33f1b090`，slot 0：

| Turn | 模型动作 | 环境结果 |
| ---: | --- | --- |
| 0 | search `Ryan Neates football club` | 找到 Claremont |
| 1 | search `Claremont Football Club official colours` | 已直接拿到 `navy blue and gold` |
| 2 | 普通文本 `The cliff Newport ... colors are gold and blue.` | `missing_native_action`，不作为答案 |
| 3 | search `Ryan Neates Claremont WAFL offical colours` | 第三次真实检索 |
| 4 | search `Claremont Football Club WAFL official colours` | terminal action，合法解析但不执行 |

这条同样拿到了决定性证据，却因无标签文本、重试后继续搜索和 terminal 不收尾而无答案、EM=0。它是“检索已成功，终止合同失败”的最直接对照。

### 7.2 第二个 strict 正例及其失败槽：Jung Joon-young

成功轨迹 `trace:54733b5a92cf8b16dd942efb`：

```text
search: Jung Joon-young big screen debut film
search: Love Forecast 2015 director
search: Park Jin-pyo film director
prefix: Jung Joon-young made his big screen debut ... directed by Park Jin-pyo.
answer: <answer>Park Jin-pyo</answer>
```

parser 保留 marker-free reasoning prefix，但只抽取标签内的 `Park Jin-pyo`，所以 strict EM=1。这个案例证明 safe-prefix 设计有效。

失败轨迹 `trace:e5b8ccd9cfe99886b2c49652` 在两次搜索后已经从文档得到 Park Jin-pyo，并输出：

```text
Jung Joon-young made his big screen debut in the film Love Forecast
(2015). The film was directed by Park Jin-pyo.
```

由于没有 `<answer>`，该 turn 被判 `missing_native_action`；后两轮又错误调用不存在的 `question_tags`，最终无答案。这说明 strict 标签确实防止了“整段长文被当答案”，但模型尚未稳定遵循标签。

### 7.3 完全无 parser 错误仍不会停：机场问题

轨迹 `trace:2c43660ffd99fcc473433b56`，gold 为 `Honolulu International Airport`：

| Turn | Query | 关键结果 |
| ---: | --- | --- |
| 0 | `17th Operational Weather Squadron airport base` | 找到基地 Hickam AFB |
| 1 | `Hickam Air Force Base neighbor airport` | 文档逐字写明基地邻近 Honolulu International Airport |
| 2 | `Honolulu International Airport airport in Hawaii` | 再次确认现名 Daniel K. Inouye International Airport |
| 3 | `17th ... Hickam ... neighbor airport` | 冗余确认 |
| 4 | `neighbor airport of Hickam Air Force Base` | terminal 再请求 search，不执行 |

这条有 4 次真实搜索、0 invalid、0 clipping，却没有 `<answer>`。因此 over-search 不能全部归咎于 parser retry；模型在已有充分证据时也缺少稳定的停止行为。

### 7.4 单回合并行多调用：Gran DT

轨迹 `trace:ac4fafea281e936f975d7db4` 的第一个回复一次生成 4 个完整调用：

```text
search: Gran DT newspaper founder
search: Gran DT competition newspaper founded by
search: Gran DT competition organized by
search: The Witness newspaper gran dt founder
```

之后四个 generation turn 也各生成 3 个调用。五轮均为 `multiple_or_unbalanced_tool_calls`，实际搜索 0 次、invalid 5 次、无答案。严格 parser 的行为是一致的；问题是模型把“并行规划多个 query”编码成了同一 assistant turn 的多个调用，与 Search-R1 串行状态机冲突。

这里还有一个实现层差异值得登记：legacy 路径会在后处理时截到第一个 `</search>` 或 `</answer>`，而当前 native 路径为了保留真实采样 token 与 log-prob，只在 EOS 处停止并保留完整回复。若下一版要恢复“一个 action 到闭标签即结束”的论文语义，应在采样阶段实现真正的 native stop，而不是事后静默裁掉多余 token；否则会再次引入 proposal/log-prob 不一致。

### 7.5 empty-think 修复成功，但短答案仍失败

轨迹 `trace:0ecb9cc05f57d4261b64e8f9` 的第三次搜索前带有：

```text
<think>

</think>
<tool_call>...</tool_call>
```

该动作被正确接受，说明 safe-prefix 修复生效。检索已得到 `21 August 1765`，随后模型先输出无标签正确长句，被判 `missing_native_action`；terminal retry 最终生成：

```text
<answer>
William IV was born in 1765 (specifically on August 21, 1765).
</answer>
```

答案边界成功抽取，但 gold 是 `21 August 1765`，整段规范化字符串不相等，因此 strict EM=0。这不是 parser bug，而是模型没有遵守“标签内只放短答案”。

### 7.6 标签合法、内容正确但仍过长：Herington

轨迹 `trace:901ea06a0ad6759e119c2b69` 先找到 Louis Durant 在 Herington 长大，再检索到 2010 年人口 `2,526`，两跳链完全正确。最终输出：

```text
<answer>Herington, Kansas, where racecar driver Louis Durant grew up,
has a population of 2,526. This figure is based on the 2010 census.
</answer>
```

parser 合法，SubEM=1，但 strict EM=0，因为 gold 仅为 `2,526`。这说明 `<answer>` 解决的是“答案字段边界”，不会自动解决字段内部的冗余表达。

### 7.7 非法首动作后能恢复检索，但仍无法收尾：Urby Emanuelson

轨迹 `trace:f2d568b74e8bf90dd42b73c2`：

| Turn | 动作 | 结果 |
| ---: | --- | --- |
| 0 | 同回合 3 个 search | invalid |
| 1 | 同回合 3 个 search | invalid |
| 2 | search `Urby Emanuelson born year` | 找到其 Ajax Youth Academy / manager 线索 |
| 3 | search `Martin Jol born year` | top doc 明确 `born 16 January 1956` |
| 4 | 输出未注册且缺参数标签的 `claim` 调用 | 按 `multiple_or_unbalanced_tool_calls` 归为 invalid，无答案 |

这条说明 retry 可以把模型从非法多调用拉回合法搜索，而且检索已经命中 gold `1956`；最终失败仍发生在动作约束和收尾，不应描述为“模型完全不会用搜索工具”。

## 8. 根因分层

### 8.1 已由证据确认

1. **不是检索服务丢包。** 88 次真实请求全部完整回填。
2. **不是 500-token 截断。** 32/32 无 clipping，最大生成 444 token。
3. **不是 native manager 改写。** G0 direct/native 的 prompt 和 raw output 16/16 相同。
4. **完整 v2 合同已能产生 tagged short-answer 和非零 strict signal。** parser 确实能抽取带 reasoning prefix 的短答案；从 0 到 2 的端到端变化不能识别 `<answer>` 单变量的因果效应。
5. **主要失败在终止与格式遵循。** 仅 8 条有答案、24 条无答案；23 个普通文本缺标签事件，13 个 terminal search 请求。
6. **terminal 的模型界面与执行层不对称。** 代码在额外收尾回合仍保留 search 工具，也没有新增“预算耗尽”提示，只在环境执行时传 `do_search=False`。这是可直接核对的实现事实，但是否为 13 个 terminal search 的单一原因尚未验证。
7. **native 动作边界与 legacy 不同。** native 保留到 EOS 的全部采样 token，能观察并拒绝单回合多调用；这保证审计真实性，但也暴露了原先后处理截断掩盖的动作分段问题。

### 8.2 目前仍只是候选解释

- Qwen chat template 自带“没有函数调用时正常回答”的通用指令，与 Search-R1 要求 `<answer>` 的合同存在软冲突；当前 system 虽然后置强调标签，但可能仍不足。
- terminal interface 没有撤下工具或刷新预算提示，可能使收尾 conditioning 不够强；但原 system 已明确 `Use at most four searches`，因此不能断言模型“不知道”预算，也没有单变量因果证据。
- `top_k=20 -> 0` 和 `presence_penalty=2 -> 0` 可能提高了某些并行多调用模式的概率，但没有单变量实验，不能下因果结论。
- 2B 后训练模型倾向完整句和解释，可能导致答案字段冗长；这与轨迹一致，但不能把所有 EM=0 都归为 verbosity，因为还存在事实错误、实体漂移和召回歧义。
- 多搜数量上升不等于推理能力等比例上升。部分搜索是有效多跳，部分是 retry、重复确认或错误锚定。

## 9. 对旧解阻计划逐项验收

| 旧计划项目 | 本次状态 | 结论 |
| --- | --- | --- |
| P0-1：恢复 `<answer>` | 已实现 | 边界抽取有效、strict EM 非零；标签遵循率和字段简洁度仍不够 |
| P0-2：修 empty-think/parser 边界 | 已实现 | empty-think 案例通过；真正的多调用、未知工具仍正确 fail-closed |
| P0-3：训练兼容中性采样 | 已实现 | 配置与 run.env 均为 top-k 0 / presence 0；但当前行为门未通过 |
| P1-1：不得事后重判旧 G2 | 已遵守 | 使用 fresh exact attempt；旧 v1 G2 没有被当作 v2 前驱 |
| P1-2：形成自动 strict reward | 部分完成 | forced-G1 的两个 group-2 pair 有 strict reward 差异；未达到 G2 readiness，更不是 group-5 训练组 |
| P1-3：G2/G3 能力门 | 未执行 | 被前置 G0/G1 NO-GO 按流程阻断，没有启动 G2、G3 或训练 |
| P2：完整证据归档 | 已完成 | 本地 sealed evidence 29/29 哈希通过，并另有 archive checksum |

旧计划的核心判断“方向值得继续，但不能直接训练”仍成立；需要更新的是阻塞位置。v1 的自由文本合同让一批语义正确但冗长的答案也得 0，叠加真实内容错误、无有效终局和 alias/日期顺序差异后，正式 strict EM 为 0；本次 forced-search G1 的奖励已不再恒为 0，但暴露出更具体的 single-action、标签遵循和 terminal 收敛问题。

## 10. 是否继续训练与下一步建议

### 10.1 当前决策

**不能从本次结果直接进入 G2，更不能启动 GRPO。** 也不建议原参数原代码重复跑一次碰运气，更不能降低阈值、把 SubEM 当 reward、接受 marker-free 长句，或把 13 个 terminal search 从结果中删掉后改判 GO。

本轮应作为一份有价值的负结果保留：

- 它证明基础设施和真实检索健康；
- 它给出了 2 条严格正轨迹和清晰的冗余搜索案例；
- 它把失败定位到模型—协议边界，而不是继续盲目调整 batch、response 或显存参数；
- 它解释了为什么“搜索次数变多”仍不足以启动成本感知训练。

### 10.2 推荐的最小后续顺序

如果决定继续，应另立 native-v3 预注册，绝不改写本次 evidence。优先顺序如下：

1. **CPU 先修指标命名，不改判定。** 分开记录 requested search、executed search、retrieval response 和 terminal search request；保留本次正式 NO-GO。
2. **只处理必要的 Qwen 状态机适配。** 重点审计两个边界：在采样层按第一个完整 `</tool_call>` / `</answer>` 真正停止一个 action，以及在 terminal generation 明确告知搜索预算耗尽、只允许 `<answer>`。这是两项独立接口变化：若为节省 GPU 合成一个预注册 adapter bundle，就只能评价 bundle，不能分别归因；若优先因果识别，则先用 G0 隔离 action stop，再用 G1 检验 terminal instruction。两种方案都不能用 post-hoc 删除 token 伪造合法动作。
3. **保持科学变量冻结。** checkpoint、固定 16×2 问题、response 500、temperature/top-p 1、top-k 0、presence 0、strict parser、strict EM、`max_turns=4` 和原门槛不变。
4. **先只重跑廉价 G0+G1。** 必须重新达到 direct/native 可解析和合法首动作门槛；基础设施条件显式设为 executed search = retrieval event，terminal search request 则保持与原门等价的阈值 0。同时报告 `<answer>` 率、无答案率及错误分布。
5. **通过后再顺序进入 G2 -> G3 -> 2-step smoke。** 任何一级 NO-GO 都封存并停止，不越级启动 R/B/C。
6. **若最小协议修复后仍缺乏短答案正例，再另立 format-only SFT 假设。** 这比放宽 reward 更可解释，但属于新的实验分支，不能偷偷并入当前复现。

为了保持最小实现原则，不建议下一轮同时做模型 sweep、LoRA、dense retriever 切换、prompt 大改、多 seed、宽松语义 judge 或正式成本奖励训练。先让同一 parent 在严格 Search-R1 合同下稳定产生“单 search 动作 -> 多轮证据 -> 短 `<answer>` -> 及时停止”，才有资格讨论成本感知是否真的减少了不必要搜索。

## 11. 可用于简历和面试的准确表述

可以表述为：

> 我将 Search-R1 适配到 Qwen3.5 原生工具协议，并恢复论文的
> `<answer>` + strict EM 奖励合同。在同 checkpoint 的 forced-search
> v1/v2 端到端协议 probe 中，我观察到平均真实检索从 1.41 增至
> 2.75 次，并首次产生 2 条 strict-EM 正轨迹；同时定位到 13 个
> terminal search、23 个 `missing_native_action` 非法文本事件和
> 单回合多工具调用问题。进一步
> 对齐证明 88 次真实 BM25 请求全部成功回填，因此没有把终止失败
> 误判成检索故障，并在能力门失败后停止训练以避免继续烧卡。

不能表述为“v2 已经训练成功”“准确率提升到可用”“成本感知已降低搜索次数”或“13 次检索接口失败”。本次是 0-update 协议门禁，它的价值在于建立了可复算的正奖励、负结果和下一步因果假设，而不是产出最终模型。
