# RL Parent Grouped Probe 详细分析

## 1. 执行结论

本轮得到的是按预注册合同执行的有效 **NO-GO**，不是训练或评测程序失败。实际 parent 是官方 **post-trained** `Qwen/Qwen3.5-2B`，不是原始预训练 `Qwen/Qwen3.5-2B-Base`；此前的 `A/Base` 只是“尚未经过本项目 Search-R1 RL”的实验角色名，容易造成误解，本文改称 `A/parent`。

该 parent 在固定 64 道 held-out HotpotQA、每题 5 条采样轨迹上，只产生 2 条满足全部条件的正确多搜轨迹，覆盖 2 道题；预注册门槛分别是 16 条和 8 道题。64 个 group 中只有 2 个具备严格可学习信号，没有任何 cost-contrast group。因此没有启动原计划仅 60 steps 的 `R-mix60`，也没有实现或训练后续 `B-mix20/C-gated-mix20`。

独立检查完整轨迹后，失败原因按影响排序是：

1. **提示词占位符被直接复制。** 312 次实际检索中有 139 次 query 规范化后只是字面量 `query`；其中 64 次发生在第一轮生成，和用户提示中的 `<search> query </search>` 直接对应。
2. **恢复文案与解析器形成确定性误调用。** 70 次 `and` 全部来自生成文本中的 `<search> and </search>`；涉及的 56 条轨迹在第一次 `and` 前全部已有非法动作。解析正则把模型复述“between `<search>` and `</search>`”时两标签之间的英文连词 `and` 当成 query。
3. **动作格式和长轨迹稳定性不足。** 185/320 条轨迹包含非法动作，102/320 至少一次生成被截断；全部截断轨迹也包含非法动作且全部答错。
4. **正确奖励过于稀疏，且没有成本对照。** 只有 7/320 条轨迹答对，58/64 个 group 五条轨迹全错；仅有的两个有效二搜 group 都只有一条正确轨迹。

这说明把 response 从 256 提高到 500 确实让模型有机会产生更多多搜轨迹，但没有解决自定义 XML 协议、恢复反馈、query 质量和正确率问题。它不说明“Qwen3.5 连搜索工具都不会用”，也不说明 Search-R1 无法训练出多搜；论文的多搜证据来自大规模 RL **之后**，而本轮测的是训练前 parent。继续单独增加长度、group size 或成本惩罚仍得不到支持，但修复协议后重新设计能力训练是一个不同且合理的新实验。

## 2. 本轮到底验证什么

前一轮 B/control20 多跳门禁中，256 题有 248 题只搜索一次，61 条正确轨迹也全部只搜索一次。为避免继续在“几乎没有正确多搜行为”的 parent 上优化搜索成本，本轮先构建由同一个 BM25 检索器验证的二搜数据，再用真实下一阶段 parent 做训练前 grouped probe。

本轮回答四个问题：

1. 当前 RL parent 是否会在未经本项目训练前自然探索两次以上搜索；
2. 多搜轨迹是否能改变 query、获得新文档和新证据并答对；
3. group 内是否存在足够的正确/错误对照供 GRPO 学习；
4. 是否存在两条以上都正确但搜索次数不同的轨迹，供成本奖励学习效率差异。

它不是 B/C 效果对比，也不是 HotpotQA benchmark 排名。`NO-GO` 的严格含义是：当前 parent、当前协议和仅 60 steps 的低成本计划没有达到事先约定的启动条件。它不表示 post-trained Qwen3.5 没有 Agent 能力，也不能外推为 500-step Search-R1 无法从更弱的 Base 模型学会搜索。

## 3. 固定实验合同

| 项目 | 固定值 |
| --- | --- |
| Parent | post-trained `Qwen/Qwen3.5-2B`，HF revision `15852e8c16360a2fea060d615a32b45270f8a8fc`，本地 digest `bc67be20efb353ba14d9c1b291a64410afec94f2310e94b59b6b76047b164e78` |
| 数据 | HotpotQA train 来源的 held-out 64 题：48 bridge + 16 comparison |
| 采样 | 每题 5 条，seed 42，temperature 1.0，top-p 1.0，共 320 条 |
| 长度 | prompt 4096，单轮 response 500，observation 384，轨迹容量 4096 |
| 搜索 | 本地 Wiki-18 BM25，top-3，最多 4 次真实检索 |
| 批次 | val batch 8；每批实际生成 8 x 5 = 40 条轨迹 |
| 硬件 | 2 x RTX 5090；只评测，不反向传播，不保存 checkpoint |
| 时间/成本 | 5,073 秒；按 5.76 元/小时折算约 8.12 元 |
| 评测 commit | `5a3bfb82a3c0ded8b4b91d22d6b8a026a855e067` |
| resolved config digest | `5f3efbf50bdcded9bf1f780b32eea423004eabd06cd2510fa4f22f6e330102d3` |
| eval data digest | `38ae778537ec04978c603612b8771c1c27f049c6224091b2b474c2f23ed0df9b` |

Parquet 中不包含 oracle query、supporting metadata 或 benchmark context；这些信息只存在于审计 catalog。配置中的 train 字段在 `val_only` 评测中不参与优化。

### 3.1 模型身份：不是 raw Base，也没有漏用 Instruct

CPU 下载脚本固定的是 `Qwen/Qwen3.5-2B@15852e8...`。该 revision 的官方 model card 明确写着 “post-trained model”，元数据把 `Qwen/Qwen3.5-2B-Base` 列为它的 base model；真正的 Base 是另一个仓库。因此：

- 本轮没有误用未后训练的 Base。Qwen3.5 的官方命名是主模型 `Qwen3.5-2B` 与 `Qwen3.5-2B-Base`，不像 Qwen2.5 一定把后训练版写成 `-Instruct`。
- 为保持历史血缘，结果文件里的 stage 仍可能写 `base`，但它只表示 `A/parent`，即没有做本项目 Search-R1 RL；不能把它解释成模型训练阶段。
- 官方主模型具备 instruction following、reasoning 和 Agent/tool-use 后训练，但这些能力针对 Qwen 原生工具模板。当前实验没有向 chat template 传 `tools` schema，而是把 `<search>...</search>` 当普通文本协议，所以不能把“官方支持工具调用”等同于“零样本稳定遵守 Search-R1 私有标签”。

## 4. 证据完整性与分析恢复

GPU 评测 attempt `20260723T022333Z-2790-20987` 本身成功退出，manifest 固定为 320 行、7,141,636 字节，trace SHA-256 为：

```text
39c8332e429ccdf438b2edf5d77dfedabf960845efdb4102baf5cb0b71d06382
```

原外层 attempt 随后以 exit code 1 失败，原因是 trace 第 28 行包含模型真实生成的空 `<search>\n</search>`。该记录是 `hotpotqa:train:18485` 的 slot 2：执行 3 次搜索、非法动作数为 3、EM=0；不是 logger 丢字段，也不是 trace 损坏。旧分析器却要求每个 query 非空，因而在聚合前中止。

修复 commit `975e40298c67107c47fdf7d37176f23c7e64d39e` 只做一件事：仍要求 query 必须是字符串，但允许空字符串进入既有 `near_duplicate_or_empty_query` 科学失败逻辑。非字符串、事件数、document ID、turn 对齐、schema、digest 和行数校验均保持严格；17 项测试通过。之后使用原 trace 做独立离线分析，未重新运行 GPU，也没有修改模型、数据、检索或采样参数。原失败 attempt 保持不变，恢复血缘单独落盘。

## 5. 独立复算总览

以下数字由 `per_trajectory.jsonl`、`per_question.jsonl` 和原始 trace 交叉复算，不只引用聚合 summary。

| 指标 | 结果 |
| --- | ---: |
| 题目 / 轨迹 | 64 / 320 |
| 答对轨迹 | 7/320（2.19%） |
| 至少一条答对轨迹的题 | 6/64（9.38%） |
| 总检索调用 / 平均每轨迹 | 312 / 0.975 |
| 二搜及以上 | 90/320（28.13%） |
| clean 轨迹 | 135/320（42.19%） |
| 截断轨迹 | 102/320（31.88%） |
| 非法动作轨迹 | 185/320（57.81%） |
| 有效正确多搜轨迹 | 2/320（0.63%） |
| near-miss | 0/320 |

### 搜索次数与正确性

| 实际搜索数 `S` | 轨迹数 | 占比 | 答对 | 该层 EM | clean | 截断 | 非法动作 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 140 | 43.75% | 3 | 2.14% | 88 | 22 | 52 |
| 1 | 90 | 28.13% | 2 | 2.22% | 39 | 24 | 51 |
| 2 | 53 | 16.56% | 2 | 3.77% | 4 | 38 | 49 |
| 3 | 32 | 10.00% | 0 | 0 | 0 | 18 | 32 |
| 4 | 5 | 1.56% | 0 | 0 | 4 | 0 | 1 |

7 条正确轨迹的搜索数分布为 `S=0/1/2 -> 3/2/2`。没有任何正确三搜或四搜轨迹，搜索更多没有表现出更高正确率；但这不能解释为“多搜有害”，因为高搜索数与格式失控、重复 query 和截断高度耦合。

## 6. 预注册门禁结果

| 硬条件 | 观测值 | 要求 | 结果 |
| --- | ---: | ---: | --- |
| 有效正确多搜轨迹 | 2 | >= 16 | FAIL |
| 覆盖题数 | 2 | >= 8 | FAIL |
| 可学习 group | 2 | >= 8 | FAIL |
| 截断率 | 31.88% | <= 5% | FAIL |
| 非法动作轨迹率 | 57.81% | <= 5% | FAIL |

五项同时失败，且不是贴近阈值的边界结果。有效轨迹数只达到要求的 12.5%；截断率和非法动作率分别是容许上限的 6.38 倍和 11.56 倍。因此无需通过调小阈值来“解释性放行”。

## 7. 从多搜到有效多搜的损耗

90 条 `S>=2` 轨迹看似已经产生多搜，但严格累计筛选显示，大部分调用不是可训练的迭代检索：

| 累计条件 | 剩余轨迹 |
| --- | ---: |
| `S>=2` | 90 |
| 再要求无截断、无非法动作 | 8 |
| 再要求前两次 query token Jaccard < 0.8 | 4 |
| 再要求第二搜增加可见新文档 | 4 |
| 再要求新增 supporting title 或答案首次可见 | 2 |
| 再要求 EM=1 | 2 |

从非累计诊断看，90 条多搜轨迹中只有 34 条 query 足够不同、23 条增加新文档、10 条增加预注册证据。失败原因可以重叠：56 条是近重复/空 query，67 条没有新文档，80 条没有新 supporting/answer evidence。

8 条 clean 多搜中，4 条是两搜且 query 有变化；另外 4 条虽然语法合法，却连续四次搜索字面量 `query`，没有答案。由此可见 `clean` 只代表结构上没有截断或非法 tag，不代表工具调用在语义上有效。

## 8. Query 质量的事后诊断

下面的统计没有加入预注册 GO/NO-GO，只用于解释失败机制。对所有实际执行 query 做 NFKD、大小写、标点和空白规范化后：

| Query 类型 | 调用数 | 无返回文档 |
| --- | ---: | ---: |
| 精确为 `query` | 139 | 0 |
| 精确为 `and` | 70 | 70 |
| 空字符串 | 1 | 1 |
| 三者合计 | 210/312（67.31%） | 71 |

这些调用涉及 126 条轨迹，其中 72 条是多搜轨迹，没有一条答对。`query` 会从 BM25 取回“Query understanding”等泛化文档，看似有 observation，实质与问题无关；`and` 和空 query 则没有返回文档。90 条多搜轨迹中，43 条所有 query 完全相同，53 条前两次 query 完全相同；预注册的 Jaccard 规则合计判定 56 条近重复或空 query。

5 条四搜轨迹全部四次重复字面量 `query`，并且全部答错。这直接说明 `max_searches=4` 不是当前瓶颈：模型已经能触发四次工具调用，但尚不能稳定生成有效的检索内容。优先增加到更多搜索次数只会放大无效调用。

进一步回看每轮原始生成后，可以把 210 次退化调用拆成三种不同机制，而不是笼统归因于“2B 模型太弱”：

1. **照抄占位符。** `query` 共出现 139 次、涉及 82 条轨迹；64 次发生在 turn 0，33 次整轮输出就是一个独立的 `<search> query </search>` action。`hotpotqa:train:83873` 的 slot 3 在没有截断、没有非法动作的情况下连续四次搜索 `query`，检索器随即返回 Query Language 类文档，模型又沿着无关 observation 重复同一调用，最终没有答案。这是“语法 clean、语义无效”的最清楚样例。
2. **解释性文字被正则误判。** 解析器使用第一个匹配 `r'<(search|answer)>(.*?)</\1>'` 的标签对，不要求 action 独立出现，也不检查 query 语义。当前非法动作恢复文案同时展示 `<search>` 与 `</search>`；模型复述 “use `<search>` and `</search>`” 时，正则恰好抽取两标签中间的 `and`。70/70 次 `and` 调用的生成文本都包含这个标签对，0 次发生在第一轮；涉及的 56/56 条轨迹此前都出现过非法动作，其中 42 条的第一次 `and` 紧跟非法动作。这里有直接的代码和轨迹因果链，不应归因于检索器或模型知识。
3. **空内容也被执行。** `hotpotqa:train:18485` 的 slot 2 首轮生成 `<search> </search>`，解析器 `strip()` 后得到空字符串，却仍标记为合法 search 并调用检索器；之后又进入两次 `query` 循环。分析器现已正确把它计为科学失败，但生成环境本身仍没有语义护栏。

### 8.1 提示词确实有问题，但不是唯一原因

当前提示与论文 Table 1 基本一致，论文原文也使用 `<search> query </search>` 这个抽象占位符，并刻意不提供具体解题策略。因此不能说我们“抄错了论文提示”；本地唯一明显文字偏差是 `as your want`，论文 v5 为 `as you want`，这个语病不是 67.31% 退化的充分解释。

但是，对当前 2B post-trained 模型而言，这个最简模板确实不够稳健：它只给了一个最终答案示例，没有给出“问题实体 -> 有意义 query”的正例，也没有说明不得原样输出 `query`、`and` 或空字符串。第一轮 64 次 `query` 与四搜复制轨迹说明，字面占位符已经进入模型行为，不能只当作措辞瑕疵。

还存在两项 Qwen3.5 适配错位：

- 官方 Qwen3.5 工具能力通过 `tools` schema 和 `<tool_call><function=...>` 模板激活；本项目只调用 `apply_chat_template(chat, add_generation_prompt=True)`，没有注册工具。对模型来说，`<search>` 是用户临时定义的 XML，而不是它原生后训练过的 function call。
- Qwen3.5-2B 默认 non-thinking。其 chat template 在未传 `enable_thinking=True` 时自动插入空的 `<think>\n\n</think>` 后才开始生成；归档 raw trajectory 可直接看到这一前缀。用户提示却要求“每次得到新信息都先在 `<think>` 内推理”，实际模板已经提前关闭该区域，模型大量在标签外推理。这一冲突会增加格式不稳定，但不能单独解释全部退化 query。

因此更准确的判断是：**论文式最简 prompt 是可学习协议，不是当前 Qwen3.5 的原生工具协议；占位符复制是提示问题，`and` 循环则是恢复文案与解析器共同制造的实现问题。** `hotpotqa:train:86102` 的成功二跳轨迹又证明模型偶尔能正确执行“人物 -> 俱乐部 -> 颜色”的自定义协议，所以问题是稳定性和训练信号，而不是绝对不会调用工具。

## 9. Response 500 起了什么作用

`max_response_length=500` 是每轮生成上限，不是完整交互轨迹总长度。多轮生成与 observation 拼接后，原始记录中的累计 `response_tokens` 中位数为 655、均值为 889、最大值为 3,085；全部轨迹容量固定为 4,096。因此出现总长度超过 500 是预期行为。

仍有 102 条轨迹至少一轮生成触顶：其中 63 条触顶一次，39 条触顶两次以上。102 条截断轨迹全部同时存在非法动作，也全部答错；另外还有 83 条未截断但含非法动作，说明格式失败不能全部归因于长度。

与上一轮 response 256 的 B/control20 门禁相比，本轮 `S>=2` 从 8/256 增至 90/320，证明更长窗口至少让多轮行为更容易被观察到。但两轮实验同时改变了 parent、数据、每题采样数和 prompt 条件，不能把增量因果归于 response 500。本轮 67.31% 的退化 query、57.81% 的非法动作率以及三搜/四搜零正确，也表明继续从 500 单独加长不能解决主要问题，只会增加显存、耗时和截断后无效输出成本。

所以更准确的结论是：**500 对诊断多搜是必要改进，但对形成可靠多跳策略仍不充分。**

## 10. 题型与 Group 信号

| 分层 | 题数/轨迹 | 答对 | `S>=2` | 有效正确多搜 | clean | 截断 | 非法动作 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bridge | 48/240 | 5 | 69 | 2 | 107 | 74 | 133 |
| comparison | 16/80 | 2 | 21 | 0 | 28 | 28 | 52 |

两个有效正例都来自 bridge；comparison 的两条正确轨迹来自同一道题且都没有搜索。comparison 没有表现出成本对照，bridge 也只有 2 个严格可学习 group，因而不能简单通过重新调整两类占比解决。

Group 层面的分布更直接：

- 58/64 个 group 五条轨迹全部答错；5 个 group 只有一条答对；1 个 group 有两条答对。
- 28/64 个 group 的 clean 轨迹出现不同搜索次数，但只有 4/64 同时出现 clean 的正确/错误结果差异。
- 只有 2/64（3.13%）含“有效正确多搜 + clean 错误”对照。
- 唯一有两条正确轨迹的 group，两条都 `S=0`，其中一条还有非法动作，所以 cost-contrast 仍为 0。

按有效正确多搜的轨迹率 2/320 做仅供直觉的独立采样近似，group 5 至少出现一条的概率约 3.1%，group 8 也只有约 4.9%。后者还会把 rollout 成本提高 60%。真实同题轨迹并不独立，但这个数量级足以说明：把 group 从 5 调到 8 不会把覆盖题数从 2 提升到预注册的 8，也不会创造缺失的 query 能力。

## 11. 七条正确轨迹和两个有效案例

| Sample | 类别 | 正确 slot | 搜索数 | 质量判断 |
| --- | --- | ---: | ---: | --- |
| `18485` | bridge | 3 | 1 | clean，但少于二搜 |
| `21756` | bridge | 2 | 0 | clean，依赖参数知识/猜测 |
| `64082` | comparison | 2、3 | 0、0 | 一条 clean，一条非法；无成本差异 |
| `80364` | bridge | 4 | 2 | 有效正确多搜，supporting-title 分支 |
| `85865` | bridge | 0 | 1 | 答对但含非法动作 |
| `86102` | bridge | 4 | 2 | 有效正确多搜，answer-evidence 分支 |

### 案例 A：`hotpotqa:train:80364`

问题询问 2006 年与 JL Racing 合作的制造商国籍，gold 为 `Swedish`。第一搜把 `JL` 写成 `JT`，返回无关赛车资料；第二搜改成 `manufacturer partnered with JL racing 2006 nationality`，取回 `JL Racing` 文档，其中明确提到与 Saab Canada 合作。模型最终回答 `Swedish`。

该轨迹通过预注册的 supporting-title 分支：第二搜首次增加 catalog supporting title `JL Racing`。它没有直接取回写有 `Swedish` 的 Saab 页面，因此比“答案直接可见”的链路弱，最终国籍仍部分依赖模型参数知识；但按事先固定规则，它是合法正例，不能在看到结果后删除。

### 案例 B：`hotpotqa:train:86102`

问题询问 Ryan Neates 所属俱乐部的官方颜色。第一搜围绕人物，文档指出他效力于 Claremont Football Club；第二搜将中间实体改写为 `Claremont Football Club official colours`，检索结果直接给出 `navy blue and gold`，模型随后正确作答。

这是本轮最清晰的迭代检索样例：人物 -> 俱乐部 -> 属性，query 随新证据变化，第二轮答案首次可见，且无截断或非法动作。

另外两条 clean、query 有变化且获得新文档的二搜轨迹分别回答 `47,707`（gold `4,334`）和 `1985`（gold `2009`），但第二搜没有新增 supporting title 或答案证据，所以不构成 near-miss。全部错误轨迹中只有 3 条通过宽松 cover-EM 发现别名包含关系，而且三条都是 `S=0`；即使放宽答案匹配，也不会增加有效多搜数量。near-miss 为 0 因而不是 strict EM 把大量好链路误判掉。

## 12. 为什么现在没有成本奖励的学习空间

后续预注册奖励为：

```text
B-mix:       r = EM
C-gated-mix: r = EM * (1 - 0.10 * n_search / 4)
```

`correct_only` 已修复旧线性成本奖励的关键问题：答错轨迹无论搜索几次都为 0，不会直接得到“答错时不搜索更好”的负梯度。但它不能凭空创造正确的多搜策略，也不能解决 group 全错导致的零 advantage。

当前 58/64 个 probe group 全错，只有 6 个存在任何 EM 对比；按更严格的多搜定义只有 2 个可学习。两个有效 group 中，唯一正确轨迹都是必要的 `S=2`，其 C 奖励为 `0.95`，其余轨迹为 0；group 内没有正确的 `S=0/1` 证明搜索可以安全减少。此时加入成本项只会略微降低唯一正确链路的奖励，而无法监督“更便宜且同样正确”的动作。

这正是 cost-contrast 门槛要求同题至少两条正确、搜索数不同的原因。观测值为 0/64，所以先训练成本分支没有可辩护的因果对照。

## 13. 主要归因与边界

### 可以从证据支持的归因

1. **首要问题是协议链，不是“Base 模型完全不会工具”。** 占位符复制由 prompt 暴露，`and` 误调用由恢复文案和非锚定正则确定性放大；当前实际使用的还是 post-trained Qwen3.5。
2. **稀疏 EM 不足以支撑原计划的短程 Hotpot GRPO。** 90.63% 的 group 全错，严格可学习 group 仅 3.13%；60 steps 很难同时完成格式适配、两跳 query 学习和正确率提升。
3. **长度问题存在但不是唯一病因。** 截断严重；同时 83 条未截断轨迹仍有非法动作，210 次退化 query 也不会由单纯加长自动修复。
4. **检索基础设施并未整体失效。** 312 次调用被实际执行，两个正例展示了可用的二搜链，另外 23 条多搜轨迹也确实增加了新文档；问题在于策略无法稳定利用它。
5. **搜索上限不是瓶颈。** 达到四搜的 5 条轨迹全部只是重复 `query`，全部答错。

### 不能从本轮证明的结论

- 不能证明 HotpotQA 通常不需要多次搜索；本轮只测当前 parent 的行为。
- 不能证明 response 500 比 256 更差；checkpoint、数据和采样合同不同。
- 不能证明 Qwen3.5 系列不适合 Agent；这里只测 2B post-trained parent 和非原生 XML 工具协议。
- 不能证明成本感知奖励无效；这里只证明应用它所需的正确成本对照尚未形成。
- 不能把两个正例外推为稳定能力；它们只占 0.63% 轨迹、3.13% 题目。

## 14. 为什么论文能够训练出多次搜索

本轮结果与论文并不矛盾，关键差别是 **checkpoint 阶段、规模和决策目标**。

| 项目 | Search-R1 论文 v5 | 本项目本轮/原计划 |
| --- | --- | --- |
| 模型 | Qwen2.5-3B/7B，Base 与 Instruct 都训练 | post-trained Qwen3.5-2B |
| 数据 | 完整 NQ + HotpotQA train 合并 | 精选混合 train-512；本轮只测 held-out 64 |
| RL | 默认 PPO，另报告 GRPO | 计划 GRPO |
| 训练 | 500 steps，total batch 512 | 计划 60 steps，prompt batch 8 |
| 每 prompt rollout | GRPO 为 5 | 5 |
| 硬件 | 8 x H100 | 2 x RTX 5090 |
| 当前观察点 | 论文图表和案例主要是 RL 后模型 | `R-mix60` 之前的 parent probe |

按配置语义，论文 GRPO 的 500 x 512 约为 256,000 个 prompt 实例、1,280,000 条采样 response；本项目计划只有 60 x 8 = 480 个 prompt 实例、2,400 条 response，规模相差约 533 倍。这个数量级差异比 group 5 改 8 更关键。

论文第 5.3 节还直接给出学习阶段：Qwen2.5-7B-Base 在前 100 steps 主要学习删掉冗余文本并适应任务格式；**100 steps 之后** response、reward 和有效搜索次数才明显上升。换言之，论文并没有要求原始 Base 在训练前已稳定多搜，搜索行为本来就是 outcome reward 在长程 RL 中学出来的。我们把“训练前必须已有 16 条正确多搜”作为省钱门禁，对原定 60-step 成本实验是合理止损，但它会拒绝论文所展示的“先学格式、后学搜索”路径，不能当成 Search-R1 可训练性的通用判据。

论文也没有只依赖 Instruct：推理式零样本 baseline 因 Base 难以服从指令而只用 Instruct，但 RL 实验同时训练 Base 和 Instruct。第 5.2 节显示 Instruct 初始性能更高、收敛更快，长程训练后两者接近。附录第 J 节的多搜案例明确来自 **训练后的 Qwen2.5-7B-Base + PPO**，例如 Chris Jericho/Gary Barlow 案例连续四次改写 query；它不能证明训练前 Base 会自然产生同样轨迹。

本项目已有的 NQ 结果也支持这一点：同一个 Qwen3.5 parent 经 60-step Search-R1 训练后，EM 从 3.9% 升到 16.4%，平均搜索约 1.305，说明它能够通过 RL 学会当前 `<search>` 协议和稳定单搜。尚未建立的是更难的“观察第一跳 -> 提取中间实体 -> 改写第二 query”，不是所有工具调用能力都缺失。

## 15. 与上一轮门禁的关系

| 项目 | 2026-07-22 B 门禁 | 本轮 parent grouped probe |
| --- | --- | --- |
| Parent | B/control20 | post-trained Qwen3.5-2B（未做本轮 RL） |
| 数据 | HotpotQA + 2Wiki dev，共 256 题 | 检索验证的 held-out HotpotQA 64 题 |
| 每题采样 | 1 | 5 |
| Response | 256 | 500 |
| `S>=2` | 8/256（3.13%） | 90/320（28.13%） |
| 正确多搜 | 0 | 2 |
| 截断 | 30/256（11.72%） | 102/320（31.88%） |
| 结论 | 无成本优化空间 | 有极少数二搜能力，但仍不足以训练 |

上一轮提出“256 可能压住多搜”的担忧是合理的，本轮也确实观察到更多多搜；但 500 并没有把这些行为转化为稳定正确链路。更准确的复盘是：**256 是局部限制，核心问题仍是 parent 没学会可靠的工具格式、query 改写和证据利用。**

## 16. 决策与重新开启实验的条件

按本轮预注册合同，停止决定保持不变：不事后降低阈值，不把同一 probe 重采到通过，也不把这批结果伪装成成功训练。补充诊断改变的是下一实验的假设，而不是本轮结论。若以后新开分支，最小优先级应是：

1. **先修协议放大器。** 保留 Search-R1 的 `<search>` 机制，但恢复文案不再同时出现可被正则匹配的成对标签；parser 拒绝空、`query`、`and`，并避免把解释性提及当 action。这比改 group 或 search 上限直接。
2. **做 prompt/模板的预注册小对照。** 控制组保留论文 Table 1；适配组给一个具体有效 query 示例并消除 `<think>` 与 Qwen3.5 默认 non-thinking 的冲突。原生 Qwen `<tool_call>` 会扩大架构改动，不作为最小首选。
3. **修改能力门禁的时点。** 训练前 probe 只要求动作格式和有效 query 达到可接受水平，不再要求已经稳定正确多搜；能力训练至少覆盖论文所揭示的约 100-step 格式学习阶段，再在 R checkpoint 上用当前严格规则判断多搜和 cost-contrast。
4. **成本实验仍后置。** 只有 R 上出现足够的正确多搜和同题正确成本对照，才从同一 checkpoint 对称训练 B/C；否则 cost reward 仍没有明确优化对象。

优先增加 group 到 8、把 response 再拉长或提高最大搜索次数都不应作为第一修复，因为它们增加成本，却没有处理 67.31% 退化 query。是否把能力训练从 60 延长到 120-150 steps，应在协议修复后的 probe 与新预算合同中单独决定，不能从本轮数据直接保证成功。

## 17. 面试口径

可以概括为：

> 我没有在第一次成本奖励失败后继续调 lambda，而是先验证 parent 是否具备可学习的正确多搜行为。我对 64 道 HotpotQA 每题采样 5 条并保存完整工具轨迹。320 条轨迹中只有 2 条是无截断、query 有变化、获得新证据且答对的二搜；67.3% 的调用退化为 `query`、`and` 或空字符串。逐 turn 审计后发现，这不是简单的“2B 不会工具”：实际 checkpoint 是 post-trained Qwen3.5，而不是 raw Base；`query` 来自提示占位符复制，70 次 `and` 则全部是宽松正则把模型复述的 `<search> and </search>` 当作调用。论文多搜案例来自 500-step、batch-512 的 RL 后 7B 模型，并明确在约 100 steps 后搜索次数才上升；我们的计划只有 60 x 8 个 prompt。因此我保留本轮 NO-GO 作为低成本合同的正确决策，同时把下一假设修正为“先修协议并完成能力学习，再做成本优化”，而不是误判 Qwen3.5 没有 Agent 能力。

不要表述成“训练失败”或“成本奖励被证明无效”。更准确的是：工程链路、数据构建、检索、轨迹记录和预注册决策都成功；科学结果否定了当前 parent 直接进入下一阶段的可行性。

## 18. 证据定位

- `results/summary.json`：权威聚合指标与门禁条件
- `results/per_question.jsonl`：64 个 group 的正确性、搜索多样性和可学习性
- `results/per_trajectory.jsonl`：320 条规范化轨迹、失败原因和证据分支
- `eval/traces/eval_predictions.jsonl`：评测器直接输出的完整模型生成、turn、query 和检索文档
- `eval/resolved-config.yaml` 与 `eval/run.env`：实际配置、时间和预算
- `failed-phase/`：原分析器失败现场
- `analysis/` 与 `results/recovery-lineage.tsv`：离线恢复终态和 commit 血缘
- `scripts/data_process/multihop_search_gate.py:174`：实际 prompt，包括字面占位符与 `as your want`
- `search_r1/llm_agent/generation.py:494`、`:523`：非法动作恢复文案和非锚定 action 正则
- `verl/utils/dataset/rl_dataset.py:128`：只传 chat，没有 `tools` schema 或 `enable_thinking`
- `scripts/autodl/02_cpu_prepare.sh:215`：实际模型 repo 与固定 revision
- [`../../2503.09516v5.pdf`](../../2503.09516v5.pdf) 第 5、7、9、16、22 页：论文 prompt、模型设置、学习阶段、训练规模和 RL 后多搜案例
- [Qwen3.5-2B 固定 model card](https://huggingface.co/Qwen/Qwen3.5-2B/tree/15852e8c16360a2fea060d615a32b45270f8a8fc)：post-trained 身份、Base 血缘、默认 non-thinking 与原生工具模板
- `archive.sha256`：Git 归档内全部证据的 SHA-256 清单

在 Linux 或 Git Bash 中可从本目录执行：

```bash
sha256sum -c archive.sha256
```
