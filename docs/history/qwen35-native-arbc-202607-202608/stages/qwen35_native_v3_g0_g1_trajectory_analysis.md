# Qwen3.5 Native-v3 G0/G1 全量轨迹分析与训练决策

> 分析日期：2026-07-26
> checkout：`experiment/hotpot-search-gate@6b1623191e6d2929fb9975bfa68acb342d8cf6de`
> 模型：`Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc`
> 精确证据：[qwen35-native-v3-g0-g1-20260726](../../../results/qwen35-native-v3-g0-g1-20260726/README.md)
> 当前结论：**协议与轨迹工程门为 GO；parent 在固定 16 题探针上已能稳定产生多次搜索动作，但主要科学问题已经从“不搜索”转为“证据充分后仍继续搜索、耗尽动作预算”。建议保持配置不变，下一步先跑 2-step 数值 smoke，成功后再启动 R60。**

## 1. 结论先行

本轮是 `train_steps=0` 的冻结 parent 评测，没有 optimizer update、训练 loss 或新 checkpoint。因此不能写成“训练后提升”，也不能把结构门的 `GO` 当作模型效果门。

本文所称“全量轨迹”是正式 G1 的 32 条自主评测轨迹；G0 的 32 条 direct/native protocol record 用于接口一致性与边界回归，另在第 4 节汇总，不与 G1 效果统计混算。

最重要的结果如下：

1. **Qwen3.5-2B 已经能自主多搜。** G1 共 32 条轨迹、91 次真实检索，平均 `2.84375` 次，中位数 3 次；27/32 至少完成两次非退化搜索。仅 1 条零搜索轨迹还是连续非法输出，并非主动选择不搜索。此前“模型不会调用工具或只搜一次”的判断不再成立。
2. **主要瓶颈是不会及时停止。** 13/32 条轨迹连续执行 4 次搜索，在严格总动作预算 `B=4` 下没有剩余 action 输出答案，13 条全部 EM=0。18 条无合法答案轨迹中，人工逐条审计确认 13 条已经在模型实际可见的 observation 中获得足够证据，其中 10 条甚至在 reasoning 中明确说出了正确答案。
3. **正式 strict EM 为 6/32=18.75%。** 14/32 输出合法 `<answer>`；其中 6 条 strict EM 正确，另有 4 条是人工判断明确正确或近正确但不满足完整字符串匹配，1 条部分正确但范围过宽，3 条明确答错。人工诊断不能替代 reward，但说明回答内容本身比 18.75% 更好；主要损失来自未提交合法答案，其次是答案精确性及 strict-EM 全串不匹配。
4. **边界修复达到了目的。** 上一轮会误判的 6 个 reasoning-marker 实例全部正确解析，0 次 parser-induced retry；真实出现的 same-token `</answer>**` overshoot 也被正确执行并完整落盘。G1 正式封存 32/32 条，不再因 trace validator 崩溃。
5. **仍有真实模型格式问题，但已不构成工程阻塞。** 113 个 generation event 中有 8 个 invalid，分布在 5 条轨迹：7 个 `invalid_thinking_prefix`、1 个 `missing_native_action`。只有 1 条轨迹发生 clipping，共 3 个 500-token generation；31/32 没有截断，因此当前证据不支持再次提高 response 长度。
6. **已有可训练信号和成本空间。** 6 个 strict mixed question 中，正确槽的搜索次数都少于同题错误槽，平均少 `1.5` 次；其中 5 个问题满足分析器的 clean `learnable=true`。这足以放行能力训练 R60，但当前没有“两条都正确、仅搜索成本不同”的配对，尚不能宣称成本分支一定有效。

## 2. 本次运行合同

| 项目 | 精确值 |
| --- | --- |
| CPU handoff digest | `0c43255f9046fe0e6906a0a99385f4cfb2c82f62af1fb6a27d162530a86a082f` |
| checkpoint digest | `bc67be20efb353ba14d9c1b291a64410afec94f2310e94b59b6b76047b164e78` |
| outer attempt | `20260726T072318Z-1616-11480` |
| G0 attempt | `20260726T072457Z-1665-26451` |
| G1 attempt | `20260726T072849Z-1665-11820` |
| prompt contract | `qwen35-native-search-v3-original-aligned` |
| G1 数据 | 固定 HotpotQA 16 题 × 2 slots = 32 条，自主搜索 |
| thinking | Qwen 原生 `enable_thinking=True`；每个 assistant generation prefix 打开 thinking |
| 动作预算 | 严格总 `B=4`；search、answer、invalid 都消耗 action；没有免费第 5 次回答 |
| 长度 | prompt 4096、单次 response 500、tool observation 500 |
| 检索 | 本地 BM25，top-3 |
| 采样 | temperature 1.0、top-p 1.0、top-k 0、min-p 0、presence penalty 0、repetition penalty 1 |
| 更新 | `val_only=true`、`train_steps=0` |
| G1 时间 | 384 秒 |
| outer 时间 / 估算 GPU 费用 | 802 秒；按 5.76 元/小时约 1.28 元 |

G1 实际使用的用户提示词是原项目语义提示，不含“至少搜索一次”、停止提醒或额外 few-shot：

```text
Answer the given question. You must conduct reasoning inside <think> and
</think> first every time you get new information. After reasoning, if you
find you lack some knowledge, you can call the available search tool with a
query, and it will return the top searched results in a tool response. You
can search as many times as you want. If you find no further external
knowledge needed, you can directly provide the answer inside <answer> and
</answer>, without detailed illustrations. For example,
<answer> Beijing </answer>. Question: ...
```

Qwen chat template 另行注入原生 tool schema、tool role 和函数调用格式。这是把 Search-R1 动作映射到 Qwen3.5 接口所需的适配，不是新增搜索策略提示。文字里的 “as many times as you want” 沿用论文与原项目“允许重复搜索”的语义，并按论文修正原仓库 `as many times as your want` 的 typo；运行时仍由预注册的严格 `B=4` 控制成本。

## 3. 终态与证据完整性

- outer、G0、G1 均为 `.success`、`terminal=success`、`exit-code=0`。
- `evidence.sha256` 注册的 30 个文件已从持久化根目录逐个复核，30/30 可读且哈希匹配。
- evidence-list digest 为 `8ce979e0bcc6c4b9bf75ef8a4bca11d9808c7752b9d9fc8b22e37b7f1171ceb2`，与 outer `evidence-digest` 和 exact marker 一致。
- 原始 G1 trace 为 32 行、16 个 sample、每题严格覆盖 slots `[0, 1]`；trace SHA-256 为 `c2c6701e075d21a74e973be53d7ab492a51932231d689a4ff694a2eaa2361f76`。
- `per_trajectory.jsonl` 内嵌的 32 条 trace 与原始 trace 一致；strict EM 已用项目 `qa_em.em_check` 规则独立复算。
- watchdog 在 durable terminal 之后写入 `shutdown-safe`、`shutdown-requested` 和 `shutdown-dispatched`；`/usr/bin/shutdown` 返回 0。记录仍正确标注 `provider_control_plane_confirmed=false`，即 guest dispatch 本身不是云平台停止计费证明。

## 4. G0/E0 与边界回归

### 4.1 G0 direct/native 一致性

G0 共 32 条 protocol record：direct 16、native manager 16，覆盖 8 题 × 2 slots。16 对记录在 prompt SHA、raw/action text、raw/action token IDs、boundary、parsed action 和 `tail_dropped` 上全部一致。

- 32/32 首动作都是合法 `tool_call -> search`，query 非空。
- 32/32 有 template thinking opening、非空 reasoning，并在真实 action 前关闭 thinking。
- native manager 的完整运行共 55 turns：47 search、7 answer、1 invalid。唯一 invalid 是 `19652/slot1/turn2` 在 500 tokens 处仍未闭合 thinking，属于真实长度/格式失败，之后只触发 1 次 retry。

这排除了“native manager 改写 prompt 或 token，导致模型不会搜索”的假设。

### 4.2 E0 环境闭环

固定 `Barack Obama` action 的请求、执行、retrieval event 和非空 tool response 均为 1；返回 3 篇文档，visible observation 为 500 tokens，`tool_role_rendered=true`，tool-response policy token 为 0，`info_mask_consistent=true`。因此 search -> BM25 -> native tool role -> 下一轮上下文 -> loss mask 的闭环完整。

### 4.3 上一轮边界故障是否真的修好

本轮实际出现 6 个“reasoning 中提到 action 标签、`</think>` 后再给真实 action”的候选：5 个真实 search、1 个真实 answer。6/6 均 `parse_error=null`，没有 marker-induced invalid 或 retry，覆盖了上一修复文档的三类案例：

- reasoning 内完整 `<answer>...</answer>`，真实 action 是 search；
- reasoning 内只提 opening `<answer>`，真实 action 是 search；
- reasoning 内提 `<answer>`，真实 action 本身也是 answer。

G1 还实际出现了 1 次同 token overshoot：`69168/slot0` 最后一个 policy token 同时解码出 `</answer>**`。系统正确执行 logical answer `Bremen`，保留原 sampled token，没有 invalid/retry，trace 正常落盘。其 strict EM=0 是因为 gold 为 `Bremen, Germany`，与 parser 无关。

所以，相对 [上一轮边界故障修复计划](../plans/qwen35_native_g1_boundary_failure_remediation_plan.md)，这次首先证明的是：**logical action、policy-token prefix 和 trace validator 已经一致；工程修复没有通过修改 prompt、reward 或采样参数来掩盖模型行为。**

## 5. G1 全量统计

### 5.1 总览

| 指标 | 数值 |
| --- | ---: |
| 轨迹 / 问题 | 32 / 16 |
| 实际搜索总数 | 91 |
| 搜索均值 / 中位数 | 2.84375 / 3 |
| 搜索分布 `0/1/2/3/4` | `1 / 4 / 7 / 7 / 13` |
| 至少二搜 / 完整二搜链诊断 | 27/32 / 27/32 |
| 合法 answer | 14/32 = 43.75% |
| strict EM | 6/32 = 18.75% |
| 至少一槽 strict 正确的问题 | 6/16 |
| 人工明确正确或近正确 | 10/32；另有 1 条部分正确 |
| 无合法 answer | 18/32 = 56.25% |
| invalid trajectory / event | 5/32 / 8/113 |
| clipped trajectory / event | 1/32 / 3/113 |
| clean（无 invalid 且无 clipping） | 27/32 |
| 平均 post-hoc utility | 0.11640625 |
| 可训练 clean mixed group | 5/16 |

strict EM 的正式规则是：小写化、去 ASCII 标点和英文冠词、压缩空白，然后要求 prediction 与任一 gold 完整相等。SubEM 仍为 6/32，没有额外覆盖人工近正确项；所有人工语义判断都只是诊断，不参与 reward 或 `GO`。

### 5.2 搜索次数与结果的交叉关系

| 实际搜索数 | 轨迹 | 输出 answer | strict EM | invalid 轨迹 | 主要现象 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 1 | 0 | 0 | 1 | 四轮全 invalid，并非主动免搜 |
| 1 | 4 | 4 | 0 | 0 | 全部过早回答；1 条语义正确但冗余，3 条答错 |
| 2 | 7 | 7 | 4 | 0 | **7/7 都成功提交答案**；另 3 条人工近正确 |
| 3 | 7 | 3 | 2 | 4 | 3 条提交答案，4 条被 invalid/预算阻断 |
| 4 | 13 | 0 | 0 | 0 | **13/13 搜满预算，无一能提交答案** |

这个表给出了本轮最重要的训练假设：当前 parent 的最佳区域明显在两次搜索附近，而不是一次或四次。两搜组正式 EM 为 4/7，另外三条分别是实体拼写错误、带额外修饰的 `public state university`、以及只写城市 `Bremen`；人工判断七条都已回答到目标语义。相反，四搜组没有 parser 错误，却因为严格总预算没有 answer action。

正确轨迹平均搜索 `2.333` 次，错误轨迹 `2.962` 次；能提交答案的轨迹平均 `1.929` 次，无答案轨迹平均 `3.556` 次。相关性不能证明因果，但“少搜且答对、过搜后无答案”的方向在这批样本上非常一致。

### 5.3 动作序列与终止原因

约定 `S=search`、`A=answer`、`I=invalid`：

| 动作序列 | 数量 | 终局 |
| --- | ---: | --- |
| `S-S-S-S` | 13 | 最后一次仍 search，预算耗尽 |
| `S-S-A` | 7 | 合法 answer |
| `S-A` | 4 | 合法 answer |
| `S-S-S-A` | 3 | 合法 answer |
| `I-S-S-S` | 2 | 首轮 invalid 后恢复搜索，最终预算耗尽 |
| `S-S-S-I` | 2 | 已搜索三次，最后格式非法 |
| `I-I-I-I` | 1 | 四轮全部非法 |

按可靠派生的 stop reason 汇总：14 条 `answered`，15 条 `budget_exhausted_after_search`，3 条 `budget_exhausted_after_invalid`。`unfinished_generation_count` 不能当作“未完成轨迹数”；它只是 event 的 `done=false` 计数。

### 5.4 模型看到了证据，为什么仍然失败

证据判断只扫描 `retrieval_events[].visible_observation`，即模型在 500-token 截断后实际看到的文本；没有把完整 document 尾部冒充为已见证据。

- 人工逐条核对确认 24/32 已看到足以回答的证据。
- 这 24 条中，10 条给出明确正确/近正确答案，1 条给出范围过宽的部分正确答案，13 条没有合法 answer。
- 18 条无答案轨迹中，13 条已经有足够证据，10 条的 reasoning 甚至明确写出了 gold，但实际 action 仍是 search 或 invalid。
- 机械“visible observation 是否包含 normalized gold 字符串”诊断为 25/32；其中 16 条在首次命中后仍继续搜索，累计多做 25 次检索。该机械指标可能有同词误命中，所以报告采用更保守的人工 24/32 作为主结论。
- 91 次真实搜索全部有 retrieval event、非空 visible observation 和恰好 3 篇文档；requested/executed/retrieval/tool-response 一一对应，无基础设施丢失。

因此当前失败不能再主要归因于 BM25 服务或 response=500。更准确的分层是：

1. **动作选择/停止失败：** 证据充分后继续 search，占无答案轨迹的主要部分。
2. **query 漂移或证据不足：** 8/32 没形成足够可见证据；其中 3 条仍过早给出错误答案，5 条无答案。
3. **答案精确性：** 模型提交了语义正确但不满足 strict EM 的翻译、修饰语、地点粒度或拼写。
4. **真实格式失败：** 8 个 invalid event 与 1 条长 reasoning clipping；它们与已修复的 parser 误判不同。

### 5.5 回答为什么看起来仍然长

合法 `<answer>` 槽实际并不普遍冗长：14 个答案的空白分词中位数为 2.5，均值 3.43，最大 12。真正长的是 `<think>`：113 个 generation event 的 reasoning 平均约 73 个空白词、中位数 56；answer turn 的 reasoning 中位数为 116.5。

14 个合法答案可分为：

- 6 个 strict EM 正确；
- 4 个人工明确正确/近正确但 strict EM=0：`Robtoro Noble`（应为 Roberto）、两条 California State University 的完整句，以及 `Bremen` 对 gold `Bremen, Germany`；
- 1 个部分正确但范围过宽：`Filmmakers/Screenwriters/Directors` 对 gold `film director`；
- 3 个明确错误：把发行方答成艺人、把发行方答成专辑、把出生日期答成人名。

这说明“回复看起来长”需要拆成 reasoning 长度与 answer 槽精度两个问题。原版 prompt 要求每次新信息后 thinking，不能为了让日志短而关闭；后续 R60 应由 strict reward 学会更及时、精确地提交答案，而不是再改提示词。

### 5.6 invalid、retry 与 clipping

| parse error | 次数 | 解释 |
| --- | ---: | --- |
| `invalid_thinking_prefix` | 7 | 模型在 reasoning 中回显/嵌套 thinking 标签，或长分析未形成合法 action |
| `missing_native_action` | 1 | `</think>` 后输出普通 `Answer: ...`，没有 `<answer>` 标签 |

8 个 invalid 中有 5 个后续 user-retry generation：2 次恢复成 search，3 次再次 invalid；没有 retry 轨迹最终答对。3 个 invalid 位于最后 action，已经没有重试预算。

唯一 clipped 轨迹是 `41225/slot0`：前三个 generation 都输出长编号分析并达到 500 tokens，第四次仍为非法 thinking prefix；全轨迹没有真实搜索。其他 31/32 轨迹、110/113 generation 都未截断。因而 response=500 已基本足够，本轮不应根据单个病理样本再改长度或 batch。

thinking 诊断为：113/113 template opening 已提供，104/113 有非空 reasoning 且在 action 前正确闭合；初始问题上下文为 30/32，tool-response 上下文为 72/76，user-retry 上下文只有 2/5。retry 后格式遵循明显更差，但它只占 5 个 turn。

## 6. 32 条轨迹逐条台账

`S1/S2/...` 表示第几次真实检索后首次出现足以回答的可见证据；`--` 表示没有形成足够闭环。人工语义列不改变正式 EM。

| sample/slot | 问题简写 | 动作 | 证据 | 合法 answer | EM | 逐条结论 |
| --- | --- | --- | ---: | --- | ---: | --- |
| `86102/0` | Ryan 所属俱乐部颜色 | `S-S-S-S` | S2 | -- | 0 | S2 已得 navy blue and gold，reasoning 两次写出答案却继续搜满 |
| `86102/1` | Ryan 所属俱乐部颜色 | `S-S-A` | S2 | `Navy blue and gold` | 1 | 标准两跳成功 |
| `41000/0` | 气象中队基地邻近机场 | `S-S-S-S` | S2 | -- | 0 | 已得 Honolulu Airport，随后 query 漂到 Glenn Farm/Travis |
| `41000/1` | 气象中队基地邻近机场 | `S-S-S-A` | S2 | `Honolulu International Airport` | 1 | 成功，但第三搜是冗余复核 |
| `88825/0` | Jung 银幕首作导演 | `S-S-S-S` | S3 | -- | 0 | 第三搜得 Park Jin-pyo，第四搜仍确认 |
| `88825/1` | Jung 银幕首作导演 | `S-S-S-S` | S4 | -- | 0 | gold 到达最后一次搜索，已无 answer action |
| `83873/0` | Gran DT 报纸创始人 | `S-S-A` | S2 | `Robtoro Noble` | 0 | 明显拼写错误；语义近正确 |
| `83873/1` | Gran DT 报纸创始人 | `I-S-S-S` | S2 | -- | 0 | 首轮非法消耗预算，得出 Roberto Noble 后仍搜索 |
| `19652/0` | Cholodenko/Morel 共同职业 | `S-S-S-A` | S2 | `Filmmakers/Screenwriters/Directors` | 0 | 含正确项但混入非共同职业，范围过宽 |
| `19652/1` | Cholodenko/Morel 共同职业 | `S-S-S-S` | S2 | -- | 0 | 未整合跨轮导演证据，持续查询 |
| `18485/0` | William Burnes 园艺训练国家 | `S-S-S-A` | S2 | `Scotland` | 1 | 成功，但第三搜冗余 |
| `18485/1` | William Burnes 园艺训练国家 | `S-S-S-S` | -- | -- | 0 | 可见片段缺关键国家，随后同名/拼写 query 漂移 |
| `63783/0` | 五项 Grammy 专辑发行方 | `S-A` | -- | `Robert Plant & Alison Krauss` | 0 | 把发行方误答成艺人，证据不足却过早停止 |
| `63783/1` | 五项 Grammy 专辑发行方 | `S-A` | -- | `Raising Sand` | 0 | 把发行方误答成专辑名，缺 record-label 第二跳 |
| `9765/0` | Blondie Purcell 出生县 | `S-S-A` | S2 | `Passaic County` | 1 | 标准两跳成功 |
| `9765/1` | Blondie Purcell 出生县 | `S-S-S-S` | S2 | -- | 0 | reasoning 已正确，仍连续两次确认 |
| `21763/0` | CSUB 是什么性质大学 | `S-A` | S1 | 中文完整句 | 0 | 语义正确，翻译和附加实体导致 strict EM=0 |
| `21763/1` | CSUB 是什么性质大学 | `S-S-A` | S2 | `...public state university...` | 0 | 语义正确，额外修饰导致 strict EM=0 |
| `82320/0` | Flores 并入城市年份 | `S-S-S-S` | S2 | -- | 0 | reasoning 已说 1888，仍再搜两次 |
| `82320/1` | Flores 并入城市年份 | `S-S-S-S` | -- | -- | 0 | 从 Flores 错误漂移到 Venezuela/Caracas |
| `25811/0` | Lady Augusta 父亲出生日期 | `S-S-S-S` | S3 | -- | 0 | 纠正父亲并得 21 Aug 1765 后仍搜索 |
| `25811/1` | Lady Augusta 父亲出生日期 | `S-A` | -- | `William IV of the United Kingdom` | 0 | 回答“谁”而非“何时” |
| `37266/0` | Louis Durant 城市人口 | `S-S-S-S` | S2 | -- | 0 | 已得 2,526，继续两次搜索 |
| `37266/1` | Louis Durant 城市人口 | `S-S-S-I` | S2 | -- | 0 | raw 中有正确 `<answer>2,526</answer>`，但嵌套第二个 think，合法拒绝 |
| `71629/0` | 首届 Summadayze 场馆开业日 | `S-S-A` | S2 | `12 February 1959` | 1 | 检索纠正先验，标准两跳成功 |
| `71629/1` | 首届 Summadayze 场馆开业日 | `S-S-S-S` | -- | -- | 0 | 持续拼错 Summadayze，检索偏离 |
| `60969/0` | Emanuelson 青训经理出生年 | `S-S-A` | S2 | `1956` | 1 | 答案正确；reasoning 对人物关系仍有误述 |
| `60969/1` | Emanuelson 青训经理出生年 | `I-S-S-S` | -- | -- | 0 | 首轮非法，后续未推进到 Martin Jol birth 第二跳 |
| `41225/0` | Ted/Kyle 共同职业 | `I-I-I-I` | -- | -- | 0 | 三次 500-token 截断，持续长分析和人物幻觉，无检索 |
| `41225/1` | Ted/Kyle 共同职业 | `S-S-S-S` | S2 | -- | 0 | 已推得 film director，仍继续两次搜索 |
| `69168/0` | St. Pauli Girl 啤酒城市 | `S-S-A` | S2 | `Bremen` | 0 | 城市语义正确；gold 多了 Germany，且 overshoot 修复成功 |
| `69168/1` | St. Pauli Girl 啤酒城市 | `S-S-S-I` | S2 | -- | 0 | 被 St. Pauli 名称误导到 Hamburg，最终普通文本无标签 |

## 7. 代表性完整过程

### 7.1 同题成功与过搜：Ryan Neates

问题 gold 为 `navy blue and gold`。

成功槽 `trace:146d5f886055a18a2254fcfd`：

```text
S1: Ryan Neates current football club official colours
    -> Ryan 当前效力 Claremont Football Club
S2: Claremont Football Club official colours
    -> visible doc: Its official colours are navy blue and gold.
A3: <answer> Navy blue and gold </answer>
```

失败槽 `trace:63b1f074bb97a71d33f1b090`：

```text
S1: Ryan Neates football club official colours
S2: Claremont Football Club official colours WAFL
    -> 已直接得到 navy blue and gold
S3: Claremont Football Club official colours
    -> reasoning 中已写 <answer>navy blue and gold</answer>，真实 action 仍是 search
S4: Claremont Football Club colours
    -> reasoning 再次确认答案，真实 action 仍是 search；预算结束
```

两条使用相同问题和 parent。差异不是检索器能否返回答案，而是模型何时把“我已经知道”转成 answer action。

### 7.2 标准两跳成功：Summadayze

`trace:d0a758ce6f0205b661fd75fe` 的初始先验把活动联想到错误地点，但真实检索完成了纠正：

```text
S1: first Summadayze festival arena opened
    -> 首届活动位于 Sidney Myer Music Bowl
S2: Sidney Myer Music Bowl opened year venue
    -> officially opened ... 12 February 1959
A3: <answer>12 February 1959</answer>
```

这是一条清晰的“检索纠错 -> 桥接第二跳 -> 精确短答案”正例，证明 native tool role 和多轮上下文真的被模型使用。

### 7.3 证据不足却过早回答：Grammy 发行方

`trace:8a24192fbd7930eed71c3589` 只搜了一次。visible observation 只说明 Plant & Krauss 的专辑 `Raising Sand` 获五奖，没有出现发行方 `Rounder Records`。模型没有继续搜索 record label，而是输出：

```text
<answer>Robert Plant & Alison Krauss</answer>
```

同题另一槽回答 `Raising Sand`。这两条不是 parser 或停止过慢，而是问题语义角色理解错误加第二跳缺失；后续 R60 不能只学“更早回答”，还必须保留必要搜索。

### 7.4 内容正确但格式应当拒绝：Louis Durant

`trace:901ea06a0ad6759e119c2b69` 的第二次搜索已看到 Herington 人口 `2,526`。第三次搜索后，模型生成：

```text
... I have all the information I need ...
</think>
<think>
... I can now provide the answer directly.
</think>
<answer> 2,526 </answer>
```

因为 continuation 中在关闭 thinking 后又开启第二个 `<think>`，整个 action 被判 `invalid_thinking_prefix`。内容虽然正确，严格 adapter 仍应拒绝；否则训练时 action token、状态机和 log-prob 语义会重新不一致。

### 7.5 唯一系统性截断个案：Ted Post / Kyle Schickner

`trace:0e599d392544382c9e2045c4` 连续生成长编号分析，前三轮都在 500 tokens 处截断，第四轮仍回显 thinking/action 指令并非法结束。它还把两位电影导演幻觉成参议员，0 次真实检索、0 答案。这是本轮唯一 clipped trajectory；应作为失败样例保留，但不足以推翻 response=500 对其余 31 条的可用性。

### 7.6 same-token overshoot 与 strict EM 的边界

`trace:c366c36d34c00d4b71bf6b59` 在两次搜索后看到 `Bremen, Germany`，最后生成的 token 同时覆盖 `</answer>` 后的 `**`：

```text
**<answer> Bremen </answer>**
```

logical parser 正确执行 `Bremen`，policy token 原样留档，轨迹没有 invalid 或 retry。这证明边界修复有效。它的 EM=0 仅因为 strict gold 是 `Bremen, Germany`；对“什么城市”而言人工判断答案正确，但正式 reward 不变。

## 8. 组内信号与成本训练含义

16 个问题的 paired outcome 为：0 个双对、6 个一对一错、10 个双错。6 个 mixed question 中，正确槽的搜索数分别为 `2/3/3/2/2/2`，错误槽为 `4/4/4/4/4/3`；正确槽在 6/6 组都更省搜索，平均少 1.5 次。

分析器认定 6 个问题 `covered=true`，其中 5 个同时满足 clean `learnable=true`；`60969` 的错误槽含真实 invalid，因此不计 clean learnable。训练实际 group size 为 5，而门禁只有 2 个 slots，更多采样通常会提高组内出现正例的概率，但这仍需 R60 轨迹验证。

当前 post-hoc utility 为：

```text
utility = EM - 0.10 * executed_search_count / 4
```

总体为 `0.1875 - 0.10 * 2.84375 / 4 = 0.11640625`。两搜 strict 正例为 0.95，三搜 strict 正例为 0.925，四搜错误为 -0.10。它说明存在节省搜索的空间，但本轮是 eval-only，utility 没有更新模型。

需要保留两个边界：

1. 当前没有同题两个槽都答对、但搜索数不同的纯成本对照，因而尚不能把准确率与成本因果拆开。
2. 全错组若直接使用线性成本惩罚，会偏好少搜甚至不搜；预注册流程先用原奖励做 R60，再由 R 后 G3 决定是否开启 `B20` 与 `C-gated20`，且 C 使用 `correct_only` 成本模式，正是为了避免这类坍缩。

## 9. 与历史结果的对照

| 指标 | native-v2 G1 | 本次 native-v3 G1 |
| --- | ---: | ---: |
| 条件 | forced-search、自写 prompt、thinking off、额外 terminal | autonomous、原版语义 prompt、thinking on、严格总 B=4 |
| observation | 384 | 500 |
| G0 direct/native 可解析 | 14/16、14/16 | 16/16、16/16 |
| 实际搜索 | 88 | 91 |
| 合法 answer | 8/32 | 14/32 |
| strict EM | 2/32 | 6/32 |
| invalid trajectory | 21/32 | 5/32 |
| invalid event | 38 | 8 |
| clipping | 0 | 1 条轨迹 / 3 events |
| 结构决策 | NO-GO | GO |

这些数字只能描述，不能做单变量归因，因为 prompt、thinking、预算语义、observation 和门禁标准都不同。可以可靠下结论的只有：当前 v3 比历史 v2 更接近原项目合同；当前合同下观测到的 invalid trajectory 由 `21/32` 降至 `5/32`，只作描述性比较；并且本轮首次在自主条件下完整封存了 32 条可分析轨迹。

与紧邻的上一轮 v3 工程失败相比，差别更直接：上一轮只留下 1 条 `.partial`，本轮 32/32 正式落盘；6/6 reasoning-marker 候选和 1 个真实 same-token overshoot 都被正确处理。因此“轨迹仍不好看”现在主要是可训练的模型行为，不再是 parser/validator 混入的假失败。

## 10. 是否继续 2-step smoke / R60

建议结论是：**可以继续，但顺序仍按预注册流程，不直接跳过数值 smoke，也不修改科学参数。**

1. 先运行 2-step smoke，验证双卡全参数反向、loss/gradient、checkpoint、trace 和原始 exit code；smoke 权重不作为 R60 parent。
2. smoke `GO` 后，R60 从同一个 sealed Qwen3.5-2B 全新启动，使用原始 EM reward（`cost_lambda=0`），不继承 smoke 权重。
3. R60 后运行固定 64×5 的 G3，重点看 answer rate、strict EM、4-search budget exhaustion、invalid/clipping，以及 correct multi-search / cost-contrast group 数。
4. 只有 G3 达到既定门槛，才从同一个 R checkpoint 对称训练 B20（原奖励）和 C-gated20（`lambda=0.10, correct_only`）；否则封存 R 的负结果并停止，不为得到成本结果而反复调参。

本轮不建议改 prompt、`B=4`、response/observation 500、group 5、top-p、top-k、presence penalty、reward 或数据配比。原因是工程合同刚刚稳定，现有轨迹已经给出足够清晰的训练目标；此时同时调参会再次失去归因。

## 11. 面试可用表述与限制

可以准确表述为：

> 我把 Search-R1 适配到 Qwen3.5 原生 reasoning/tool-call 协议，并把 logical action 与不可拆分的 policy token prefix 分离。结构门在 32 条自主 HotpotQA 轨迹上验证了 113/113 action-token prefix、91/91 检索回填和 32/32 loss mask；冻结 parent 平均搜索 2.84 次，strict EM 18.75%。逐轨迹分析发现主要失败不是不搜索，而是 40.6% 轨迹把四个动作全部用于搜索，很多在证据充分后仍不提交答案。这为后续“先学会正确停止，再做 correct-only 成本对照”提供了可检验假设。

不能夸大为：

- `GO` 证明模型效果达到论文水平；
- 6/32 是训练提升；
- 人工 10/32 可以替代 strict EM；
- 这 16 道筛选题代表论文七个标准 benchmark；
- 当前已经证明成本奖励优于原奖励。

## 12. 证据索引

本地归档保留完整原始数据，不在本文复制 2.5 MB trace：

- G1 原始 32 条：[eval_predictions.jsonl](../../../results/qwen35-native-v3-g0-g1-20260726/raw/runs/eval/qwen_native_g1/attempts/20260726T072849Z-1665-11820/traces/eval_predictions.jsonl)
- 逐轨迹诊断：[per_trajectory.jsonl](../../../results/qwen35-native-v3-g0-g1-20260726/raw/runs/qwen-native-gate/attempts/20260726T072318Z-1616-11480/per_trajectory.jsonl)
- 逐题配对：[per_question.jsonl](../../../results/qwen35-native-v3-g0-g1-20260726/raw/runs/qwen-native-gate/attempts/20260726T072318Z-1616-11480/per_question.jsonl)
- G0 记录：[records.jsonl](../../../results/qwen35-native-v3-g0-g1-20260726/raw/runs/eval/qwen_native_g0/attempts/20260726T072457Z-1665-26451/output/records.jsonl)
- 结构决策：[go_no_go.json](../../../results/qwen35-native-v3-g0-g1-20260726/raw/runs/qwen-native-gate/attempts/20260726T072318Z-1616-11480/go_no_go.json)
- 运行配置：[resolved-config.yaml](../../../results/qwen35-native-v3-g0-g1-20260726/raw/runs/eval/qwen_native_g1/attempts/20260726T072849Z-1665-11820/resolved-config.yaml)
- 完整证据清单：[evidence.sha256](../../../results/qwen35-native-v3-g0-g1-20260726/raw/runs/qwen-native-gate/attempts/20260726T072318Z-1616-11480/evidence.sha256)

本文的科学结论以原始 trace、模型实际可见的 observation 和正式 strict EM 为准；完整 retrieved document 只用于检索器审计，不能冒充模型上下文。
