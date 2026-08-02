# Qwen3.5 Native-v4 Terminal Gate-v5 G0/G1 轨迹分析

> 分析日期：2026-07-28
>
> 模型：`Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc`（post-trained，不是 `Qwen3.5-2B-Base`）
>
> 运行提交：`f8c1cd7e87078d07385f74ca8710add5d5f79c06`
>
> 当前结论：**Gate-v5 工程门禁 GO；只批准进入预注册的 2-step smoke，不批准直接启动 R60。**

## 1. 结论先行

本轮不是训练，也没有产生 loss、梯度或新 checkpoint；它是冻结 parent 上的 G0 协议探针和 G1 自主 rollout 评测，`train_steps=0`。结果可以概括为四点：

**这里的 GO 仅表示该 exact 配置通过 Gate-v5 协议/工程合同并允许 2-step smoke，不表示模型能力、训练收益或成本优化已经通过。**

1. **工程链路已经通过。** G0 prompt/token/action 边界、E0 真实检索回填、G1 token prefix、observation token 的 policy mask、retrieval lineage 和 terminal 边界共 17 项硬条件全部通过，正式决策为 `GO`。
2. **parent 已经能做真实多搜。** 32 条轨迹执行 88 次 BM25 检索，平均 2.75 次；26/32 形成至少两次、query 不重复且 observation 全对齐的搜索链，strict EM 为 `12/32 = 37.5%`。这与早期“基本只搜一次、没有多搜信号”的状态不同。
3. **terminal 安全边界有效，但软提醒服从仍弱。** 17 条未结束轨迹全部收到提醒；7 条回答、9 条仍请求 search、1 条调用未知工具。9 次 search 请求均被记录但未接受、未执行。`GO` 证明环境不会越过预算继续检索，不代表模型已经学会停止。
4. **下一步值得做极小反向验证。** 当前已有多搜、正确答案和错误答案，但尚未验证两卡全参数 backward、组内 advantage、数值稳定性和 checkpoint/WandB 闭环。因此只应运行 2-step smoke；看完 smoke 轨迹和 loss 后再决定 R60。

## 2. 精确运行身份与完整性

| 项目 | 精确值 |
| --- | --- |
| 外层 attempt | `20260728T044634Z-2051-4045` |
| G0 attempt | `20260728T044826Z-2100-21776` |
| G1 attempt | `20260728T045208Z-2100-14954` |
| Gate result contract | `qwen-native-gate-v5` |
| Gate JSON schema | `search-r1.qwen-native-gate@5` |
| checkpoint digest | `bc67be20efb353ba14d9c1b291a64410afec94f2310e94b59b6b76047b164e78` |
| CPU handoff digest | `9ffaa89f88990887b368750ccfdc559d93afdb9a1f5719411cf3e3c1147da90b` |
| data manifest digest | `db6cacf865365ee977fc280d5195556e9903907b28b228ba8f3d72133d674167` |
| G1 resolved config digest | `d7d0b3a7e59b02107c75b53f5b112a5ef37c4c2cd96c403789e4eec013fa7100` |
| raw trace digest | `48880a8f03ade831c98afc01cb22343d49e66f47e9d9bd7da0212f27c14f0d02` |
| trace manifest digest | `37c509f5f55d4cc0bc37885575e018c3c003311d89d7da1aea8fc800812e69a1` |
| evidence/marker digest | `168bd35d6c72d2f00aa03c6e9490f8cc5a601d005729369982214ea0ab1839a5` |
| 外层终态 | `terminal=success`、`exit-code=0` |

远端从项目持久化根目录执行 `sha256sum -c`，evidence manifest 的 30/30 项全部通过。外层从 `04:46:34Z` 到 `04:59:39Z`，共 785 秒；按两卡 `5.76 元/小时` 做 wall-clock 乘法，约为 `1.26 元`。这只是该 attempt 的实验成本估算，不包含启动前准备、shutdown 后控制面延迟或平台计费粒度。G0 用时 221 秒，G1 用时 348 秒，其余时间用于检索器准备、分析、证据落盘和关机流程。

自动关机证据按 `shutdown-safe -> shutdown-requested -> shutdown-dispatched` 顺序存在，backend 返回 0。它只证明 guest shutdown 已派发，不能单独证明 AutoDL 控制面已经停止计费；用户随后确认关机并以无卡模式重启，这是独立的 operator observation，不属于 30/30 哈希证据。

## 3. 本轮实际合同

### 3.1 初始 prompt 与 Qwen 原生协议

G1 使用 `qwen35-native-search-v4-terminal-answer-only`。初始数据仍只有一条 user message，没有“至少搜索一次”的额外指令；核心文本为：

```text
Answer the given question. You must conduct reasoning inside <think> and
</think> first every time you get new information. After reasoning, if
you find you lack some knowledge, you can call the available search tool
with a query, and it will return the top searched results in a tool
response. You can search as many times as you want. If you find no
further external knowledge needed, you can directly provide the answer
inside <answer> and </answer>, without detailed illustrations. For
example, <answer> Beijing </answer>. Question: {question}
```

“as many times as you want”是保留的上游提示词语义；真实环境仍以四个常规 action turn 为硬预算。工具通过 Qwen3.5 原生 tool schema 传入，`enable_thinking=True`，检索结果以原生 `tool` role 回填。代码合同见 [tool_protocol.py](../../../../search_r1/llm_agent/tool_protocol.py)。

采样固定为：

| 参数 | 值 |
| --- | ---: |
| temperature / top-p | `1.0 / 1.0` |
| top-k / min-p | `0 / 0.0` |
| presence / repetition penalty | `0.0 / 1.0` |
| response / observation | `500 / 500` |
| max regular action turns | `4` |
| eval group size | `2` |

### 3.2 Terminal reminder 的真实语义

四个常规 action 后仍 active 的轨迹，会追加一条固定 user message：

```text
The search budget is exhausted. You must not call the search tool again.
Using only the question and information already available, give your best
answer even if uncertain. After reasoning, output exactly one concise
final answer inside <answer> and </answer>, with no text after </answer>.
```

这是**软提醒 + 硬环境边界**：

- reminder 作为上下文进入下一次生成，但 reminder/template token 的 policy mask 为 0；
- terminal assistant 自己生成的 token 仍是 policy token，后续训练时可以被 reward 学习；
- terminal `<answer>` 可以合法结束；
- terminal search 只记录请求，强制 `accepted=false`、`executed=false`；
- 其他非法格式 fail closed，不从 reasoning 文本中猜造答案。

预算是四个常规“动作回合”，不是承诺每条轨迹一定执行四次检索。非法动作也消耗回合，因此 17 条 terminal 轨迹中有 4 条只执行三搜、1 条只执行一搜，这不是搜索计数错误。

## 4. 为什么 Gate-v4 是 NO-GO，而本轮 Gate-v5 是 GO

上一轮独立 lineage attempt `20260727T143620Z-1528-6819` 运行于 `98cac287...`，合同为 Gate-v4；本轮运行于 `f8c1cd7...`，合同为 Gate-v5。两轮固定相同的模型、数据、prompt、采样设置和 seed，但 checkout、完整 config digest 与 lineage 身份不同。

| 指标 | Gate-v4 | Gate-v5 | 是否改变 |
| --- | ---: | ---: | --- |
| strict EM | 12/32 | 12/32 | 否 |
| terminal answer | 7/17 | 7/17 | 否 |
| terminal requested search | 9/17 | 9/17 | 否 |
| terminal other invalid | 1/17 | 1/17 | 否 |
| terminal accepted/executed search | 0/0 | 0/0 | 否 |
| 决策 | `NO-GO` | `GO` | **是** |

两次 raw trace 的 SHA 不同，但逐字段比较显示 32 个原始 JSON 对象唯一的差异路径是 `/run_id`；`record_id` 及其他字段均相同。去除 `run_id` 后，32/32 条 JSON 逐字段一致。这是一项等价性核对：它只证明该 32 条切片上的生成行为未变，改变的是准入合同；两个不同合同的 `NO-GO/GO` 标签不能当成同口径性能提升。

Gate-v4 把以下模型行为目标误设成硬门：

```text
terminal_answer_rate >= 90%
terminal_requested_search_rate <= 5%
```

这两个比率正是计划交给 RL 验证的候选行为目标，不适合作为本阶段的工程准入条件。Gate-v5 保留比率和完整轨迹作为诊断，但只让工程能够保证的事实决定 GO/NO-GO：提醒必须完整注入、提醒 token 不得进入 policy loss、terminal search 不得被接受或执行，以及原有 prompt/token/mask/retrieval/lineage 合同必须通过。

`f8c1cd7` 同时提升了 gate/smoke contract 版本，并加强 terminal prompt 文本、digest、follow-up token、run-index、原始 trace 和 analyzer replay 的可验证性；它没有修改 prompt 文案、parser、模型、数据、BM25、采样、reward 或生成参数。Gate-v5 是看到 Gate-v4 失败语义后进行的 post-hoc 工程合同修订，不是预注册的 confirmatory success；旧 Gate-v4 `NO-GO` 证据保持不变，也没有被追溯覆盖。具体语义见 [主复现方案第 10 节](../plans/autodl_search_r1_reproduction_plan.md#10-terminal-门禁语义修正)。

## 5. G0/E0：协议与真实工具闭环

G0 manifest 记录 16 条 direct 和 16 条 native-manager 输出。硬检查全部通过：

| 检查 | 结果 |
| --- | ---: |
| dataset prompt token 与 direct HF prompt 一致 | 16/16 |
| direct 与 manager 首动作 token 一致 | 16/16 |
| direct action prefix 完整 | 16/16 |
| manager action prefix 完整 | 16/16 |
| E0 search roundtrip | 1/1 |
| E0 返回文档 | 3/3 |
| E0 以 tool role 回填 | 1/1 |
| E0 observation mask leak | 0 |

E0 用 `Barack Obama` 做一次真实检索，返回 3 篇文档；tool response 有 500 个可见 token、0 个 policy token。阶段日志在检索器启动前有一次 `127.0.0.1:8000 connection refused` 的预检查，随后服务正常启动；G1 的 88/88 次真实请求均成功，故它不是运行期检索故障。

## 6. G1 总体结果

### 6.1 核心统计

| 指标 | 数值 |
| --- | ---: |
| 问题 / 轨迹 | 16 / 32 |
| generation turns | 128 |
| strict EM | 12/32 = 37.5% |
| 有合法最终 answer | 22/32 |
| 无最终 answer | 10/32 |
| 含至少一个 invalid 的轨迹 | 12/32 |
| 合法首动作 | 27/32 |
| 首动作 search / answer / invalid | 26 / 1 / 5 |
| 常规 accepted / executed search | 88 / 88 |
| retrieval events / 非空 tool response | 88 / 88 |
| 重复 query / 非 ASCII query | 0 / 5 |
| response clipped | 1/32 |

`128/128` 个 action token prefix 完整，action tail leak 为 0；`32/32` 的 info mask 一致，observation policy token 为 0；retrieval alignment error 为 0。这说明 answer/search 行为可以归因给 policy，而不是 prompt 重渲染、检索丢包或 observation 进入 loss。

“search 请求”还有三个必须分开的口径：88 是常规回合中被接受并执行的 search；加上 9 个 terminal 被拒请求后，模型共有 97 个可解析的 `requested_action=search`；另有 6 段 raw output 包含 `<function=search>`，但因 `invalid_thinking_prefix` 没有形成可解析请求，所以 raw search tool-call 文本共 103 段。后文的成本代理只计 88 次真实执行。

### 6.2 搜索次数与正确率

| 实际搜索数 | 轨迹数 | strict EM | 组内 EM |
| ---: | ---: | ---: | ---: |
| 0 | 2 | 0 | 0% |
| 1 | 4 | 0 | 0% |
| 2 | 6 | 3 | 50.0% |
| 3 | 8 | 4 | 50.0% |
| 4 | 12 | 5 | 41.7% |
| **合计** | **32** | **12** | **37.5%** |

所有 12 条严格正确轨迹都至少执行两次搜索。26 条完整二搜链中有 12 条正确；其余 6 条不构成完整二搜链且全部错误。这说明该切片存在正确的多次检索轨迹，而不是仅观察到零搜或一搜正例；但没有逐条证据归因或 fewer-search 反事实，不能据此断言所有正确答案都因果依赖跨文档多跳检索。

正确轨迹共执行 38 搜，平均 `3.167`；错误轨迹共执行 50 搜，平均 `2.5`。小样本不能证明“多搜导致正确”，也不能只凭平均数证明可无损减少搜索；不过 Ryan 等轨迹在 reasoning 已确认答案后仍继续搜索，提示可能存在可压缩调用。真正的成本改善必须由相同正确性下的 B/C 配对或搜索截断消融证明。当前 gate 没有改 `lambda`，也不应把这组前向数据写成成本分支已经有效。

本项目当前的成本代理仅为 `executed_search_count`，不是 AutoDL 人民币、端到端时延、生成 token 或 GPU 总算力。terminal search 没有执行，因此不增加该代理值和 BM25 调用，但额外 terminal generation 仍消耗推理算力。

### 6.3 问题级信号

16 题中：8 题两条都错、4 题一对一错、4 题两条都对；至少一条正确的题为 8/16。分析器的保守定义下：

- `covered=7/16`：至少有一条 strict 正确、至少二搜、无 invalid、无 clipping 的轨迹；
- `clean-pair learnable diagnostic=0/16`：同题还必须同时有一条无 invalid、无 clipping 的错误轨迹；
- 有效 clean correct multi-search 共 11 条。

四个 mixed-EM 问题的错误 partner 都带 terminal search rejection；`60969` 的正确轨迹又有 clipping/invalid，因此分析器特定的 `clean-pair learnable diagnostic` 为 0。它不等于“GRPO 没有 reward 差异”：这是 group=2 的前向启发式诊断，并且当前实现把预期的 terminal 安全拒绝也计入轨迹 `invalid_action_count`。若只在这项离线诊断中排除 9 个预期拒绝，`25811`、`86102`、`88825` 三题会成为 clean mixed pair，诊断值为 3/16；正式代码没有追溯改写该结果。正式训练 group=5。按照 Gate-v5 工程合同，它足以放行 smoke，但不是能力成功证据，更不足以跳过 smoke 直接声称 R60 会成功。

## 7. Terminal 专项台账

| Terminal 结果 | 条数 | strict EM | 此前实际搜索 | 环境处理 |
| --- | ---: | ---: | --- | --- |
| 合法 `<answer>` | 7 | 6/7 | 3搜×1，4搜×6 | 正常终止 |
| 再次请求 search | 9 | 0/9 | 1搜×1，3搜×3，4搜×5 | 9/9 拒绝，0 次执行 |
| unknown tool | 1 | 0/1 | 4搜×1 | fail closed |

结构结果为：

- reminder applied `17/17`；
- terminal prompt policy token `0`；
- terminal accepted search `0`；
- terminal executed search `0`；
- terminal answer rate `7/17 = 41.18%`；
- terminal requested-search rate `9/17 = 52.94%`。

前四回合提前回答的 15 条轨迹中有 6 条正确；进入 terminal 后，条件在“确实输出 answer”上的正确率是 6/7。真正的瓶颈不是 terminal answer 一旦生成就普遍答错，而是 10/17 根本没有生成合法 answer。全部 10 条无答案轨迹正好由 9 条 terminal search 和 1 条 unknown tool 构成。

因此最准确的判断是：

> 环境边界已经安全，terminal policy 尚未服从。后者是希望 outcome reward 优化、但仍待训练验证的目标，而不是训练前必须由脚本伪造为 100% 的条件。

## 8. Invalid、冗长与截断

32 条轨迹共有 18 个 parse-error event，分布如下：

| parse error | event | 涉及轨迹 | terminal event |
| --- | ---: | ---: | ---: |
| `search_disallowed_after_budget` | 9 | 9 | 9 |
| `invalid_thinking_prefix` | 6 | 4 | 0 |
| `missing_native_action` | 1 | 1 | 0 |
| `multiple_or_unbalanced_answers` | 1 | 1 | 0 |
| `unknown_tool` | 1 | 1 | 1 |

`summary.json` 的 `terminal_invalid_count=1` 专指 terminal 的非-search非法动作；9 个被拒 search 单独计入 `terminal_search_request_count`。而轨迹级 `invalid_action_count` 会包含被环境拒绝的 terminal search，所以总共有 12 条 invalid trajectory。这两个数字口径不同但不矛盾。

18 个 parse error 中，9 个是环境按设计产生的 terminal 安全拒绝；排除它们后，真正的格式/协议错误为 9 个 event、涉及 7 条轨迹。因此 `invalid_trajectory_count=12` 不能直接解读成“12 条都存在模型格式错误”，也解释了上一节 `clean-pair learnable diagnostic` 为什么格外保守。

本轮只有 `60969/slot1` 一条 response clipped，而且不是首轮；`first_turn_clipped=0`。因此 500 token 不是本轮 terminal 服从差的主要原因，再加长 response 既没有当前证据支持，也会扩大反向显存。

冗长答案仍会伤害 strict EM。两条 CSUB 轨迹分别抽取 33 词和 26 词的 answer，虽然都包含“public university”，却不等于 gold `public university`。`<answer>` 解决了字段边界，不会自动保证字段内部简短；这应由原始 EM reward 学习，而不是靠宽松 parser 从长句中挑答案。

## 9. 代表性完整轨迹

以下均来自本轮封存的 raw trace；省略每次返回的三篇文档全文，但保留全部动作顺序、关键可见证据和终局。

### 9.1 已知道答案仍请求搜索：Ryan Neates

`trace:63b1f074bb97a71d33f1b090`，gold `navy blue and gold`：

```text
S1 Ryan Neates football club official colours
   -> 定位 Claremont Football Club
S2 Claremont Football Club official colours WAFL
   -> 文档明确 official colours are navy blue and gold
S3 Claremont Football Club official colours
S4 Claremont Football Club colours
T  Claremont Football Club Excel Colour
   -> requested=search, parse_error=search_disallowed_after_budget,
      accepted=false, executed=false, EM=0
```

terminal reasoning 原文已经写出：

```text
I has sufficient information to answer the question without needing to
call the search tool again, so I can provide the final answer directly...
```

随后真实 action 却仍是 search。这不是检索失败，也不是 reminder 没注入，而是 reasoning 与最终 action 不一致；是最典型的 RL 停止策略目标。

### 9.2 Terminal 正常收尾：机场问题

`trace:2c43660ffd99fcc473433b56`，gold `Honolulu International Airport`：

```text
S1 17th Operational Weather Squadron base airport
S2 Hickam Air Force Base neighbor airport
   -> 文档明确 Hickam neighbors Honolulu International Airport
S3 Glenn Farm Airport 17th Weather Squadron base
S4 Travis Glenn Farm Airport 17 Weather Squadron
T  <answer> Honolulu International Airport </answer>
   -> valid=true, EM=1
```

它正向证明四个常规动作后，reminder、native thinking、严格 `<answer>` parser 和 EM 可以完整闭环。第二搜已经返回直接答案，第三、四搜因而提示可能存在可压缩调用；是否能在保持正确性的同时省掉它们，仍需配对对照验证。

### 9.3 reasoning 写了 answer，但真实动作是未知工具

`trace:0114a4febe76404670cc1910`，问题为 Lisa Cholodenko 与 Pierre Morel 的共同职业，gold `film director`：

```text
S1 Lisa Cholodenko Pierre Morel occupation work
S2 Pierre Morel filmmaker writer screenwriter director
S3 Lisa Cholodenko Pierre Morel mutual occupation ...
S4 Pierre Morel filmmaker occupation screenwriter
T  reasoning: Final answer: <answer>screenwriter and director</answer>
   actual action: <function=feed>...</function>
   -> parse_error=unknown_tool, no extracted answer, EM=0
```

严格 parser 的处理是正确的：reasoning 中出现 `<answer>` 字符串，不等于模型最终选择了 answer action。若事后从思考文本捞答案，会掩盖真实策略错误并污染 policy/reward 对齐。

### 9.4 原版一次 retry 生效，但长答案 strict EM 为 0

`trace:8e17baf88c0f39cf645f992d`，gold `public university`：

```text
Turn 0: 同一回复出现不平衡/嵌套 answer，multiple_or_unbalanced_answers
User retry: My action is not correct. Let me rethink.
Turn 1: 合法 <answer>，但内容是 26 词解释句
Result: extracted answer 存在，strict EM=0
```

这证明免费 retry 路径已经恢复且能把非法动作转成合法 answer；失败发生在答案冗长，而不是“没有最终回答机会”。

### 9.5 同题展示检索规划与 strict EM 的两种失败

Summadayze 问题 gold 为 `12 February 1959,`：

- `trace:d0a758ce6f0205b661fd75fe` 错把场馆定位为 Kirjurinluoto Arena，四搜后 terminal 回答 `2001`，属于检索链/实体消歧失败；
- `trace:1ffe69002c37b3e426aed0db` 两搜正确定位 Sidney Myer Music Bowl，回答 `February 12, 1959`，语义正确但 strict EM 仍为 0。

项目 EM 会小写、去标点/英文冠词并压缩空白，但不会重排 token；因此：

```text
gold -> 12 february 1959
pred -> february 12 1959
```

两者不完全相等。该例是 metric 表达敏感性的透明记录，不应据此临时放宽 reward；正式实现见 [qa_em.py](../../../../verl/utils/reward_score/qa_em.py)。

## 10. 当前能证明什么、不能证明什么

### 10.1 已经证明

1. Qwen3.5 原生 chat template、thinking、单工具 action 边界和 Search-R1 token replay 对齐。
2. BM25 top-3 的真实 HTTP roundtrip、tool-role 回填和 observation mask 健康。
3. parent 能在自主 prompt 下执行多次不同 query，并产生 12 条 strict 正例。
4. terminal reminder 覆盖完整且不进入 policy loss；预算耗尽后的搜索不会产生额外检索成本。
5. Gate-v5 可以把工程安全条件与待学习的模型行为目标分开。

### 10.2 尚未证明

1. 没有 backward，因此没有 finite loss、KL、entropy、grad norm、显存稳定性或 checkpoint 证据。
2. 没有训练前后对照，不能说 terminal compliance、EM 或搜索成本已经改善。
3. 16 题 × 2、单 seed 只能做门禁，不足以估计总体 benchmark 效果。
4. 记录值 `clean-pair learnable diagnostic=0/16` 受 terminal 安全拒绝计入 invalid 的口径影响；排除该预期拒绝时为 3/16。两种口径都不能保证 group=5 的每个训练 batch 有非零 advantage，必须由 smoke 实测。
5. 本轮使用官方支持的 BM25，而论文主实验是 E5 dense retriever；内部 B/C 可比性可以成立，绝对指标不能冒充论文数值复刻。

## 11. 决策与下一步

本轮决策为：**允许执行 2-step 原始 EM reward smoke，除此之外不变。**

下一步不再调整 prompt、response/observation 500、四轮 action budget、batch 8、group 5、采样参数、BM25、数据配比或 reward。smoke 必须从同一 sealed parent 启动，并至少验证：

- 两步 actor policy-gradient loss、KL、entropy 和 grad norm 均存在且有限；
- 至少一个 group 有 mixed reward，policy advantage 非零且 mask/log-prob 对齐；
- terminal 四项硬合同继续通过，answer/search/invalid 比率只作诊断；
- step-2 checkpoint、完整轨迹、WandB offline history、exit code 和 storage evidence 全部封存；
- 自动关机派发后仍由 AutoDL 控制台确认停止计费。

smoke checkpoint 不作为 R60 parent。只有 smoke `GO` 且人工看过实际轨迹后，才从同一个原始 sealed parent 新启 R60；这符合当前“先看门禁与轨迹，不急着长训”的最小实现原则。

2-step 只验证 backward、数值和证据闭环，没有判断 EM、terminal compliance 或成本改善的统计效力。后续是否启动 R60，只能依据工程无阻断和确有非零训练信号来决定，不能期待两步本身产生可解释的行为提升。

## 12. 面试可用表述

可以准确表述为：

> 我把 Qwen3.5 原生 tool calling 接入 Search-R1，并把四个常规 action 后的最终回答机会实现为模型可见、reminder token 的 policy mask 为 0（不计入 policy loss）的 user reminder。Gate-v5 在 32 条冻结 parent 轨迹上验证了 128/128 action-token prefix、88/88 真实 BM25 回填、32/32 info mask 和 17/17 terminal 提醒；预算耗尽后的 9 次 search 请求全部被拒绝且 0 次执行。模型本身只有 7/17 次按提醒回答，所以我把它保留为待 RL 验证的优化目标，而不再用 90% 服从率阻止工程 smoke。当前 strict EM 为 12/32，下一步只放行 2-step backward smoke。

还应主动说明：Gate-v4 与 Gate-v5 的 32 条生成行为完全一致，`NO-GO -> GO` 来自 post-hoc 门禁语义修正而不是挑 seed，也不是性能提升；旧负结果没有删除。这个失败—审计—修正链比单独展示一个 GO 更能说明实验归因和工程严谨性。

## 13. 证据索引

远端持久化路径：

```text
/root/autodl-tmp/search-r1/state/attempts/gpu/20260728T044634Z-2051-4045
/root/autodl-tmp/search-r1/runs/qwen-native-gate/attempts/20260728T044634Z-2051-4045
/root/autodl-tmp/search-r1/runs/eval/qwen_native_g0/attempts/20260728T044826Z-2100-21776
/root/autodl-tmp/search-r1/runs/eval/qwen_native_g1/attempts/20260728T045208Z-2100-14954
/root/autodl-tmp/search-r1/runs/eval/qwen_native_g1/attempts/20260728T045208Z-2100-14954/traces/eval_predictions.jsonl
/root/autodl-tmp/search-r1/manifests/qwen-native-gate/20260728T044634Z-2051-4045.ok
```

本报告的汇总数字来自 `summary.json`，问题级结论来自 `per_question.jsonl`，逐轨迹结论来自 `per_trajectory.jsonl` 与原始 `eval_predictions.jsonl`；四者已独立交叉重算，未发现不一致。完整流程合同见 [AutoDL 操作说明](../../../../scripts/autodl/README.md) 和 [terminal/gold/WandB 修复方案](../plans/qwen35_native_terminal_gold_wandb_remediation_plan.md)。
