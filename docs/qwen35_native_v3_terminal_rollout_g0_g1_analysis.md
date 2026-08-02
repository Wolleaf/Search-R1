# Qwen3.5 Native-v3 上游 Terminal Rollout G0/G1 完整执行分析

> 分析日期：2026-07-26
> checkout：`experiment/hotpot-search-gate@846ce35996e7a7691f79c403724f672c82cac994`
> 模型：`Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc`
> 对照报告：[上一轮严格四动作 G0/G1 分析](qwen35_native_v3_g0_g1_trajectory_analysis.md)
> 最终结论：**工程与协议门为 GO。上游 Search-R1 的“4 个常规动作 + 未完成轨迹再生成 1 次、但不再执行检索”语义已真实恢复；然而 19 次 terminal generation 中只有 1 次形成正确答案，15 次仍请求搜索，3 次格式非法。主要科学问题仍是 Base 模型在证据充分后不会稳定停止，而不是检索链路或 terminal 实现失效。建议不再改 prompt、采样或预算，下一步只放行 2-step 数值 smoke；smoke 通过后再决定启动原始 EM 奖励的 R60。**

## 1. 本轮回答了什么

本轮只运行冻结 parent 的 G0/G1 门禁，`val_only=true`、`train_steps=0`，没有 optimizer update、训练 loss 或新 checkpoint。因此本文分析的是：

1. terminal rollout 是否与原项目语义一致；
2. Qwen 原生 tool-call、检索回填、policy token 和 loss mask 是否仍正确；
3. 额外收尾 generation 是否真的让模型回答；
4. 相比上一轮“严格总动作 `B=4`、没有收尾 generation”，哪些差异能做因果解释；
5. 当前证据是否足以放行 2-step smoke / R60。

结构门的 `GO` 只证明实现和证据合同通过，**不等于模型效果通过，更不等于已经训练成功**。

## 2. 恢复前后语义

| 项目 | 上一轮严格四动作 | 本轮恢复上游语义 |
| --- | --- | --- |
| 常规动作预算 | 4 | 4 |
| 常规 search | 执行检索并计成本 | 执行检索并计成本 |
| 四轮后仍 unfinished | 直接结束 | 再生成 1 次 terminal action |
| terminal 请求 search | 不存在 | 可解析、保留 policy token，但不执行检索、不计成本 |
| 最大 generation event | 4 | 5 |
| 最大真实检索 | 4 | 4 |
| 右侧容量 | `4*(500+500)=4000` | `4*(500+500)+500=4500` |

这里的额外 generation 是“最后一次作答机会”，不是第五次可检索动作。若模型仍输出 search，轨迹保持 unfinished；系统不会伪造 observation，也不会把它计入 `executed_search_count`。terminal token 仍由 policy 采样并进入 `responses`、`info_mask` 和后续 PPO loss，这一点不能隐藏，否则训练语义会与实际动作不一致。

本轮没有新增“至少搜索一次”“最后必须回答”或停止提示。用户 prompt 仍为原项目语义版本：允许缺知识时搜索，知识足够时在 `<answer>...</answer>` 内直接回答；Qwen chat template 仅注入原生 thinking/tool schema。`enable_thinking=True`，不是关闭 think 后靠手写格式提示驱动。

## 3. 精确运行合同

| 项目 | 精确值 |
| --- | --- |
| outer attempt | `20260726T105027Z-1597-2709` |
| G0 attempt | `20260726T105205Z-1646-15875` |
| G1 attempt | `20260726T105627Z-1646-13281` |
| result contract | `qwen-native-gate-v3` |
| CPU handoff digest | `24f9ab439cbc999dac2f2ca65b41735ca51441dc5cfdfc22b06ff545c07323ca` |
| checkpoint digest | `bc67be20efb353ba14d9c1b291a64410afec94f2310e94b59b6b76047b164e78` |
| data manifest SHA-256 | `d27b8026314aec3d09faf5daa9435d4c1f69516fc42d0b951a339d6313947185` |
| G1 resolved config SHA-256 | `e7e811884b14e17e1e5cafc4d084a48c07a6bb06b10bc644feb55810d75334b1` |
| prompt contract | `qwen35-native-search-v3-original-aligned` |
| G1 数据 | 固定 HotpotQA 16 题 × 2 slots = 32 条 |
| budget | `max_turns=4`；unfinished 可有第 5 个 terminal event |
| 长度 | start 1024、response 500、observation 500、right-side 4500 |
| 检索 | 本地 BM25，top-3 |
| 采样 | temperature 1.0、top-p 1.0、top-k 0、min-p 0、presence penalty 0、repetition penalty 1 |
| G1 实际 group | 2（门禁采样）；配置中的 `n_agent=5` 是后续训练 group |
| 更新 | `val_only=true`，无参数更新 |

本次 capacity 的唯一配套变化是 `4000 -> 4500`，用于保留 terminal 最多 500 个 policy token。按最坏长度 `(1024+4500)*micro_batch_2=11048`，仍低于 `ppo_max_token_len_per_gpu=16384`；本轮评测也没有 OOM。它不能替代反向传播 smoke，但当前没有证据要求降低 batch、group 或 response。

## 4. 终态、完整性与费用

- outer、G0、G1 均有 `.success`、`terminal=success`、`exit-code=0`，无 `.failed`、超时、OOM、Traceback 或 NaN。
- `evidence.sha256` 注册 30 个文件；从持久化项目根执行 `sha256sum -c` 为 30/30 `OK`。
- evidence-list digest 为 `761fa06b995780a59099fbbc3f09cba23154e8d0e2f9a660da5a9fc72698548c`，与 outer `evidence-digest` 和 exact marker 一致。
- 原始 G1 trace 为 32 行、16 个 sample、每题严格 slots `[0,1]`，SHA-256 为 `c5eb2ef32c89a71fce8ddd72c3af3465fec45bda956ef39bf536386458eb658b`。
- trace manifest SHA-256 为 `784d18b8c112fca9378fa6e5b87fbc28b9611a35eaf9aa1c0ce0e68d31ead3ce`；记录的 `rows=expected_rows=32`、字节数 `2,619,914` 均匹配。
- 聚合 `per_trajectory.jsonl` 内嵌的 32 条 trace 与原始 trace 按顺序逐对象完全相等；strict EM 已独立复算 32/32 一致。

| 阶段 | 耗时 | 按 5.76 元/小时估算 |
| --- | ---: | ---: |
| G0 | 262 秒 | 约 0.42 元 |
| G1 | 540 秒 | 约 0.86 元 |
| outer 编排与收尾 | 184 秒 | 约 0.29 元 |
| 合计 | 986 秒（16 分 26 秒） | **约 1.58 元** |

`finished-at=11:06:53Z` 后依次写入 `shutdown-safe=11:06:54Z`、`shutdown-requested=11:06:55Z`、`shutdown-dispatched=11:06:55Z`；`/usr/bin/shutdown` 返回 0。证据仍诚实记录 `provider_control_plane_confirmed=false`，即 guest dispatch 本身不能证明云平台已经停止计费。

日志中有一次启动前 `127.0.0.1:8000 connection refused`，随后检索服务正常启动并完成 92/92 回填，所以它是瞬时 readiness 探测，不是实验故障。NCCL device-id warning、Torch 慢路径和 `top_k=0` 被忽略提示也没有造成 hang 或结果缺失。

## 5. G0/E0 工程门

G0 共 32 条 protocol record：direct 16、native manager 16，形成 16 个严格配对。

- prompt token match：16/16；
- 首 action token match：16/16；
- direct / manager action-prefix integrity：16/16、16/16；
- 两路首动作均为合法 search：16/16、16/16；
- E0 固定搜索的 requested / executed / retrieval / nonempty tool response：`1/1/1/1`；
- E0 返回 3 篇文档，tool role 正确，observation policy-token leakage 为 0，mask 一致。

因此 Qwen 原生 renderer、logical action、不可拆分 token prefix、BM25 环境和 tool-role 回填仍闭环。terminal 修复没有破坏上一轮已经通过的协议层。

## 6. G1 全量统计

### 6.1 总览

| 指标 | 本轮结果 |
| --- | ---: |
| 轨迹 / 问题 | 32 / 16 |
| generation event | 133 |
| 常规真实检索 | 92 |
| 搜索均值 | 2.875 |
| 搜索分布 `0/1/2/3/4` | `3 / 3 / 5 / 5 / 16` |
| 合法 answer | 14/32 = 43.75% |
| strict EM | **5/32 = 15.625%** |
| 至少一槽正确的问题 | 5/16 |
| invalid trajectory / event | 7/32 / 12/133 |
| clipped trajectory / event | 2/32 / 3/133 |
| terminal generation | 19 |
| terminal search / answer / invalid | `15 / 1 / 3` |
| 重复 query | 0 |
| 非 ASCII query | 7 |

92 次常规 search 的 requested、executed、retrieval event 和非空 tool response 全部一一对应。133/133 action token 是 raw sampled token 的完整前缀，32/32 info mask 一致，observation policy token 为 0，action-tail leakage 为 0。

thinking 诊断为：133/133 都由 Qwen template 提供 opening；113 个 event 形成非空 reasoning 并在 action 前正常闭合。初始问题、tool response、retry 三类 generation 数分别为 `32/92/9`。非法输出应作为真实 Base 行为保留，不能用 parser 放宽去制造答案。

### 6.2 搜索次数与结果

| 真实搜索 | 轨迹 | 合法 answer | strict EM | invalid 轨迹 | clipped 轨迹 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3 | 2 | 0 | 2 | 1 |
| 1 | 3 | 3 | 0 | 1 | 0 |
| 2 | 5 | 5 | 2 | 0 | 0 |
| 3 | 5 | 3 | 2 | 2 | 0 |
| 4 | 16 | 1 | 1 | 2 | 1 |

正确轨迹的搜索数为 `[2,3,3,2,4]`；错误轨迹平均搜索 `2.889` 次。能提交答案的轨迹平均搜索 `1.857` 次，无答案轨迹平均 `3.667` 次。相关性不能证明“少搜导致正确”，但再次说明失败集中在证据充分后不提交答案。

14 个合法 `<answer>` 的空白分词中位数为 2、均值 2.86、最大 11。也就是说，**真正冗长的是 reasoning 或非法 generation，不是合法 answer 槽普遍过长**。

### 6.3 动作序列

约定 `S=执行 search`、`A=answer`、`I=invalid`，`*` 表示 terminal generation：

| 动作序列 | 数量 | 终局 |
| --- | ---: | --- |
| `S-S-S-S-S*` | 13 | terminal 再次请求 search，但不执行 |
| `S-S-A` | 5 | 常规两搜后回答 |
| `S-S-S-A` | 3 | 常规三搜后回答 |
| `S-S-S-S-I*` | 2 | 四搜后 terminal 非法 |
| `I-S-S-S-S*` | 2 | normal invalid 后三搜，terminal 再请求 search |
| `S-A` | 2 | 一搜后回答 |
| `A` | 1 | 不检索直接回答 |
| `I-S-A` | 1 | invalid 后一搜回答 |
| `S-S-S-S-A*` | 1 | **四搜后 terminal 正确回答** |
| `I-I-I-I-I*` | 1 | normal 与 terminal 全部非法 |
| `I-I-A` | 1 | 两次非法后回答 |

### 6.4 组内信号与 post-hoc utility

本轮 5 个问题出现“一槽 strict 正确、一槽错误”，没有双槽都正确。正确槽与错误槽的搜索数分别为：

```text
18485: 3 vs 4
37266: 4 vs 4
41000: 3 vs 4
86102: 2 vs 4
9765: 2 vs 4
```

其中 4/5 个问题的正确槽更省搜索，正确槽均值 2.8、错误槽均值 4；`37266` 则由 terminal answer 在同样四次检索下区分成败。分析器认定 5 个问题 `covered=true`，其中 4 个同时为 clean `learnable=true`。这里 clean learnable 的口径是：同题组内至少有一条“strict 正确、至少二搜、无 invalid/clipping”和一条“strict 错误、无 invalid/clipping”的轨迹；`86102` 的错误槽含真实 invalid，因而不计 clean learnable。

按当前仅用于诊断的公式：

```text
utility = EM - 0.10 * executed_search_count / 4
        = 0.15625 - 0.10 * 2.875 / 4
        = 0.084375
```

terminal search 没有真实执行，因此不进入 cost 分子；但它也没有 answer，EM 仍为 0。本轮是 eval-only，这个 utility 没有更新模型。当前也没有“双槽都正确、仅搜索成本不同”的纯成本对照，所以只能放行能力训练，不能据此提前启动成本分支。

## 7. Terminal generation 专项审计

### 7.1 正确口径

terminal event 总数是 **19，不是 16**。原因是“是否进入 terminal”取决于四个常规 generation 后是否 done，而不是是否恰好执行了四次搜索：

- 16 条执行了四次搜索：terminal 为 13 search、1 answer、2 invalid；
- 2 条先有一个 normal invalid，只执行三次搜索：terminal 都是 search；
- 1 条四个 normal generation 全 invalid、零搜索：terminal 仍 invalid。

19 个 terminal context 中，18 个是上一轮真实 `tool_response`，1 个是 `user_retry`。15 个 terminal search 全部满足：

```text
valid_action=true
executed_search=false
retrieval_executed=false
observation=null
parse_error=null
```

所以正式 92 次真实检索不包含这 15 个请求。19 个 terminal event 共贡献 3,397 个 policy token，19/19 都被 info mask 保留且 action-prefix 完整；在训练阶段这些 token 是 loss-bearing policy token，但本轮 `val_only` 没有 backward 或 optimizer update。

### 7.2 19 条 terminal 台账

| sample/slot | normal 动作 | terminal | reasoning / 证据状态 | 结论 |
| --- | --- | --- | --- | --- |
| `86102/0` | `S-S-S-S` | `I*` | 多次明确 `navy blue and gold` | 知道答案，格式失败 |
| `41000/0` | `S-S-S-S` | `S*` | 已明确 Honolulu，仍想确认 | 停止失败 |
| `88825/0` | `S-S-S-S` | `I*` | 多次写出 Park Jin-pyo，500 token 截断 | 冗长 + 格式失败 |
| `88825/1` | `S-S-S-S` | `S*` | 明说答案 Park Jin-pyo 且无需再搜 | 停止失败 |
| `83873/1` | `I-S-S-S` | `S*` | 明说 Roberto Noble 且应输出 answer | 停止失败 |
| `19652/1` | `S-S-S-S` | `S*` | 已归纳共同职业为 film director | 停止失败 |
| `18485/1` | `S-S-S-S` | `S*` | 文本含 Scotland，但训练地点证据仍不清楚 | 合理地仍想检索，但 terminal 不执行 |
| `9765/1` | `S-S-S-S` | `S*` | 已明确 Passaic County | 停止失败 |
| `82320/0` | `S-S-S-S` | `S*` | 已明确 1888，并说无需再搜 | 停止失败 |
| `82320/1` | `S-S-S-S` | `S*` | 已明确 1888 | 停止失败 |
| `25811/1` | `S-S-S-S` | `S*` | 已明确 21 August 1765 | 停止失败 |
| `37266/0` | `S-S-S-S` | `A*` | 已明确人口 2,526 | **唯一 terminal 正例** |
| `37266/1` | `S-S-S-S` | `S*` | 已明确人口 2,526 | 停止失败 |
| `71629/0` | `I-S-S-S` | `S*` | 已明确 12 February 1959 | 停止失败 |
| `71629/1` | `S-S-S-S` | `S*` | 已明确 12 February 1959 | 停止失败 |
| `60969/0` | `I-I-I-I` | `I*` | 持续回显/破坏 thinking 格式 | 协议失败 |
| `60969/1` | `S-S-S-S` | `S*` | 未找到 1956，terminal query 退化为 `The` | 检索/query 失败 |
| `69168/0` | `S-S-S-S` | `S*` | 证据有 Bremen，却错误归纳 St. Paul | 推理错误后继续搜 |
| `69168/1` | `S-S-S-S` | `S*` | 已正确归纳 Bremen | 停止失败 |

15 个 terminal search 中，人工逐条核对其 raw reasoning：

- 12/15 已明确说出正确或近正确答案，动作却仍是 search；
- 1/15 已形成错误答案；
- 2/15 确实没有解决问题；
- 14/15 的历史 visible observation 含 literal gold，唯一例外是 `60969/1`。literal 命中只是诊断，`18485/1` 就是“出现 Scotland 但语义关系仍不足”的反例。

3 个 terminal invalid 均为 `invalid_thinking_prefix`；其中 2 个边界为 EOS，1 个在 500 tokens 的 length boundary 截断。唯一 terminal answer 为 `37266/0`：`<answer>2,526</answer>`，strict EM=1。

如果只截断本次已记录轨迹的四个 normal action，则为 13 个合法 answer、4 个 EM；terminal 最终增加了 **1 个 answer 和 1 个 EM**。因此机制不是完全无效，但转化率只有 `1/19=5.26%`。

## 8. 与上一轮 v3 的可比性

### 8.1 表面总量

| 指标 | 上一轮无 terminal | 本轮 restored-terminal |
| --- | ---: | ---: |
| strict EM | 6/32 | 5/32 |
| 合法 answer | 14/32 | 14/32 |
| 常规 executed search | 91 | 92 |
| generation event | 113 | 133 |
| invalid trajectory / event | `5 / 8` | `7 / 12` |
| clipped trajectory / event | `1 / 3` | `2 / 3` |
| 搜索分布 `0/1/2/3/4` | `1/4/7/7/13` | `3/3/5/5/16` |

同 `(sample_id, group_slot)` 的 EM 转移为：1 条 `0->1`、2 条 `1->0`、4 条保持正确、25 条保持错误。answer 新增 4 条、丢失 4 条；搜索减少 5 条、持平 20 条、增加 7 条。

**不能据此说 terminal 导致 EM 从 6 降到 5。** 虽然模型、问题、seed 相同，采样流并不是逐样本独立重置。

### 8.2 可控的前半批与 RNG 边界

G1 实际分成两个 16-slot batch。新旧运行的前 16 条，其全部非 terminal `raw_text`、action token、动作与检索次数逐字节一致：

- 新旧均为 EM `4/16`；
- 新旧均为 answer `8/16`；
- 新旧均为 executed search `48`；
- 其中 8 条 unfinished 获得 terminal：6 search、2 invalid、0 answer；
- 8 条中 7 条在 terminal raw reasoning 已形成正确答案，仍没有合法回答。

这是本轮最强的受控结论：**terminal 路径确实运行了，但冻结 Base policy 不会稳定把额外机会转成 answer。**

第一批新增的 8 次 terminal sampling 随后推进了全局 RNG。第二批 16 条因此从首轮起全部重新采样：旧批 EM=2、新批 EM=1；新批出现唯一 terminal 正例 `37266/0`，同时丢失旧批的 `60969/0` 和 `71629/0` 两个正例。这个 `-1` 是重采样区间的净变化，不能当作代码修复的因果效果。

若未来要估计 terminal 的平均效果，应使用多 seed 或按 batch/样本隔离 RNG；当前最小复现不需要为此再改训练代码。

## 9. 32 条轨迹逐条台账

`S1/S2/...` 表示第几次真实检索后 visible observation 首次出现 literal gold；它不自动等价于语义证据充分。`*` 为 terminal action。

| sample/slot | 问题简写 | 动作 | literal gold | answer | EM | 逐条结论 |
| --- | --- | --- | ---: | --- | ---: | --- |
| `86102/0` | Ryan 俱乐部颜色 | `S-S-S-S-I*` | S2 | -- | 0 | normal reasoning 两次写出答案，terminal 仍非法 |
| `86102/1` | Ryan 俱乐部颜色 | `S-S-A` | S2 | `Navy blue and gold` | 1 | 标准两跳成功 |
| `41000/0` | 气象中队基地邻近机场 | `S-S-S-S-S*` | S2 | -- | 0 | 已知 Honolulu，terminal 继续确认 |
| `41000/1` | 气象中队基地邻近机场 | `S-S-S-A` | S2 | `Honolulu International Airport` | 1 | 成功，第三搜略冗余 |
| `88825/0` | Jung 银幕首作导演 | `S-S-S-S-I*` | S3 | -- | 0 | terminal 多次写正确答案但长文本截断、格式非法 |
| `88825/1` | Jung 银幕首作导演 | `S-S-S-S-S*` | S4 | -- | 0 | terminal 明说无需外部知识仍 search |
| `83873/0` | Gran DT 报纸创始人 | `S-S-A` | S2 | `Robtoro Noble` | 0 | 拼写错误，语义近正确 |
| `83873/1` | Gran DT 报纸创始人 | `I-S-S-S-S*` | S2 | -- | 0 | 已知 Roberto Noble，terminal 仍 search |
| `19652/0` | 两人共同职业 | `S-S-S-A` | S2 | `Filmmakers/Screenwriters/Directors` | 0 | 含正确项但范围过宽 |
| `19652/1` | 两人共同职业 | `S-S-S-S-S*` | S2 | -- | 0 | 已归纳 director，terminal 仍 search |
| `18485/0` | Burnes 训练国家 | `S-S-S-A` | S2 | `Scotland` | 1 | 成功，第三搜冗余 |
| `18485/1` | Burnes 训练国家 | `S-S-S-S-S*` | S1† | -- | 0 | `Scotland` 为同词命中，训练地点关系仍不清楚 |
| `63783/0` | Grammy 专辑发行方 | `S-A` | -- | `Robert Plant & Alison Krauss` | 0 | 把发行方答成艺人，缺第二跳 |
| `63783/1` | Grammy 专辑发行方 | `S-A` | -- | `Raising Sand` | 0 | 把发行方答成专辑，缺第二跳 |
| `9765/0` | Blondie Purcell 出生县 | `S-S-A` | S2 | `Passaic County` | 1 | 标准两跳成功 |
| `9765/1` | Blondie Purcell 出生县 | `S-S-S-S-S*` | S2 | -- | 0 | 已明确县名，terminal 仍 search |
| `21763/0` | CSUB 大学性质 | `A` | -- | `...public research university...` | 0 | 免搜近义长答案，strict 全串不等 |
| `21763/1` | CSUB 大学性质 | `I-S-A` | S1 | `California State University, Bakersfield` | 0 | 回答实体而非大学性质 |
| `82320/0` | Flores 并入年份 | `S-S-S-S-S*` | S3 | -- | 0 | 已知 1888，terminal 仍 search |
| `82320/1` | Flores 并入年份 | `S-S-S-S-S*` | S2 | -- | 0 | 已知 1888，terminal 仍 search |
| `25811/0` | Lady Augusta 父亲生日 | `S-S-A` | S2 | `1765` | 0 | 只给年份，缺日月 |
| `25811/1` | Lady Augusta 父亲生日 | `S-S-S-S-S*` | S2 | -- | 0 | terminal reasoning 已给完整日期仍 search |
| `37266/0` | Louis Durant 城市人口 | `S-S-S-S-A*` | S2 | `2,526` | 1 | **唯一 terminal 正确回答** |
| `37266/1` | Louis Durant 城市人口 | `S-S-S-S-S*` | S2 | -- | 0 | 同题同证据，terminal 继续 search |
| `71629/0` | Summadayze 场馆开业日 | `I-S-S-S-S*` | S2 | -- | 0 | terminal 已写正确日期仍 search |
| `71629/1` | Summadayze 场馆开业日 | `S-S-S-S-S*` | S2 | -- | 0 | terminal 已写正确日期仍 search |
| `60969/0` | Emanuelson 青训经理出生年 | `I-I-I-I-I*` | -- | -- | 0 | 五次均破坏 thinking/action 格式 |
| `60969/1` | Emanuelson 青训经理出生年 | `S-S-S-S-S*` | -- | -- | 0 | 人物/query 漂移，terminal query=`The` |
| `41225/0` | Ted/Kyle 共同职业 | `S-S-A` | -- | `film professionals` | 0 | 推到共同电影领域，但答案过宽且检索未证实 Ted |
| `41225/1` | Ted/Kyle 共同职业 | `I-I-A` | -- | `acting` | 0 | 两次长格式失败后猜错职业 |
| `69168/0` | St. Pauli Girl 啤酒城市 | `S-S-S-S-S*` | S2 | -- | 0 | 看见 Bremen 却误推 St. Paul，terminal query 漂移 |
| `69168/1` | St. Pauli Girl 啤酒城市 | `S-S-S-S-S*` | S3 | -- | 0 | 正确推得 Bremen，terminal 仍 search |

## 10. 代表性完整过程

### 10.1 Terminal 机制成功但策略分叉：Louis Durant

问题 gold 为 `2,526`。同题两个 slot 都在第二搜拿到直接证据，前四个动作都为 search。

`37266/0`（`trace:12b434a55d107a7ca93647e0`）：

```text
S1: Louis Durant racecar driver grew up Kansas city
    -> Louis Durant grew up in Herington, Kansas
S2: Herington Kansas city population
    -> As of the 2010 census, the city population was 2,526
S3: Herington Kansas 2010 census population
S4: Herington Kansas population 2020 census 2030 estimate
A5 terminal: <answer>2,526</answer>
```

`37266/1`（`trace:901ea06a0ad6759e119c2b69`）：

```text
S1: Louis Durant Kansas city grew up personal history birthplace
S2: Herington Kansas population
S3: Herington Kansas population 2010 census
S4: Louis Durant name change Herington Kansas longest name racecar driver
S5 terminal request: Herington Kansas Herington city total population
    -> executed_search=false；没有第五个 observation
```

第二条 terminal reasoning 已逐字写出 “I have all the information I need” 和 “population was 2,526”，却仍生成 `<tool_call>`。这证明 terminal plumbing 正常，差异出在 policy 的 answer-vs-search 动作选择。

### 10.2 受控前半批的停止失败：Park Jin-pyo

`88825/1`（`trace:54733b5a92cf8b16dd942efb`）的新旧四个 normal generation 完全相同：

```text
S1: Jung Joon-young big screen debut film director
S2: Love Forecast 2015 director Jung Joon-young film
S3: Love Forecast 2015 film director Jung Joon-young movie
S4: Lee Seung-gi film debut Love Forecast director 2015
    -> visible observation: Love Forecast ... directed by Park Jin-pyo
terminal reasoning:
    The answer is: Park Jin-pyo.
    ... no need for further external knowledge.
S5 terminal request: Park Jin-pyo director Love Forecast 2015 film
```

terminal 请求被正确记录但未执行。这个案例没有第二批 RNG 混杂，是“额外 generation 本身不能保证回答”的干净证据。

### 10.3 知道答案却协议失败：Ryan Neates

`86102/0`（`trace:63b1f074bb97a71d33f1b090`）：

```text
S1: Ryan Neates football club official colours
S2: Claremont Football Club official colours WAFL
    -> Its official colours are navy blue and gold.
S3: Claremont Football Club official colours
    -> reasoning 写出 <answer>navy blue and gold</answer>，真实 action 仍 search
S4: Claremont Football Club colours
    -> reasoning 再次写出同一答案，真实 action 仍 search
I5 terminal:
    raw text 明确答案并说可以回答，最后只输出普通 `Answers`
    -> invalid_thinking_prefix
```

parser 拒绝它是正确的：内容中“提到答案”不等于 policy 生成了一个合法 answer action。训练正需要对这种轨迹给低回报，而不是靠宽松解析把它伪装成成功。

### 10.4 冗长、截断和格式混乱：Jung Joon-young slot 0

`88825/0`（`trace:e5b8ccd9cfe99886b2c49652`）四搜后已经看到 Park Jin-pyo。terminal raw text 多次写出正确答案，却混入多个 `<answer>`、第二段 thinking、`Jean-Luc Godard - Park Jin-pyo` 和说明性长文，最终在 500 tokens 截断，判为 `invalid_thinking_prefix`。

它是本轮“回答冗余”的真正问题样例：不是 `<answer>` 槽太长，而是模型未稳定生成单一合法 action。全局只有 2 条轨迹、3 个 event clipping，所以不能据此再次提高 response；提高长度只会让这一病理样例继续展开，并增加反向显存。

### 10.5 真正的检索失败：Emanuelson

`60969/1`（`trace:ca3400b12092805a46f8a92f`）：

```text
S1: Urby Emanuelson Ajax Youth Academy manager
S2: Marischal Emanuelson birth year
S3: Ajax Youth Academy manager born
S4: Marischal Emanuelson born June 2016
S5 terminal request: The
```

gold `1956` 从未出现在 visible observation，query 还把人物关系漂移到 Marischal Emanuelson。它与前述“证据已足够仍搜索”不同，是少数真正的检索规划失败；后续训练不能简单奖励一律早停。

## 11. 当前科学判断

### 11.1 为什么仍有训练希望

1. **工程信号干净。** 92/92 检索闭环、133/133 policy prefix、32/32 mask 通过，失败可以归因到模型行为，而不是工具或日志损坏。
2. **parent 已有正例。** 5/16 问题在两个 eval slots 中至少一条 strict 正确；其中 4 个问题为分析器认可的 clean learnable group，另 1 个问题的错误槽含 invalid。
3. **失败目标清晰。** 大量轨迹已获取答案，甚至在 reasoning 中说出答案，却没有切换到 `<answer>`。这正是 outcome reward 可区分的行为，而不是模型完全没有知识或不会使用工具。
4. **terminal token 可训练。** 额外 generation 的 3,397 个 policy token 已进入 loss mask；terminal search 不执行检索但最终 EM=0，正确 terminal answer 得 EM=1，信用方向与上游一致。
5. **训练 group 更大。** G1 只是每题 2 条；R 阶段 group=5，提高同题组内出现正确/错误对照的概率，但不能保证每组都有正例。

### 11.2 主要风险

- 19 个 terminal 机会只有 1 个答案，Base 的动作控制偏置很强；
- 12 个 invalid event、2 条 clipped trajectory 说明格式仍有学习空间；
- strict EM 对 `1765`、`Robtoro Noble`、额外修饰和地点粒度非常严格，语义近正确仍为 0；
- 当前只有 16 题 × 2 slots、单 seed，不能用 15.625% 推断总体能力；
- G1 是纯前向，尚未验证 4500 capacity 下双卡全参数反向的显存和数值稳定性。

综合判断：**这不是“模型没希望”的 NO-GO，而是“实现已对齐、行为问题明确、值得用极小训练成本验证”的 GO。**

## 12. 后续执行建议

严格遵循最小实现原则，下一步不修改科学参数：

1. 先跑预注册的 **2-step 原始 EM reward smoke**，验证双卡全参数 backward、loss、gradient、checkpoint、trace 和退出码；smoke 权重不作为 R60 parent。
2. 只有 smoke 全部通过，才从同一个 sealed Qwen3.5-2B parent 全新启动 R60；不继承 smoke 权重。
3. R60 后跑固定 G3，重点比较：strict EM、合法 answer rate、四搜后 unfinished、terminal search rate、invalid/clipping、正确多搜轨迹和组内 reward 方差。
4. 只有 R/G3 证明能力阶段没有坍缩且形成成本对照，才从同一个 R checkpoint 对称启动 B20 与 `C-gated20(correct_only, lambda=0.10)`；本轮不提前调 lambda。

当前**不要**改：prompt、thinking、`max_turns=4`、terminal 语义、response/observation 500、train batch 8、group 5、top-p、top-k、presence penalty、数据配比或奖励函数。否则会再次把“上游语义恢复”和“额外调参”混在一起，失去归因。

## 13. 面试可用表述与限制

可以准确表述为：

> 我对照 Search-R1 上游实现，恢复了四轮可检索 rollout 后的一次 terminal policy generation；terminal 再请求搜索时只记录动作、不执行检索，但 token 仍参与 PPO loss。结构门在 32 条 Qwen3.5-2B 自主轨迹上验证了 133/133 policy-token prefix、92/92 检索回填和 32/32 loss mask。19 个 unfinished 轨迹获得 terminal generation，只有 1 个正确回答，15 个继续请求搜索、3 个格式非法；其中 12 个 terminal search 的 reasoning 已明确给出正确或近正确答案。由此把主要瓶颈定位为“证据充分后的停止和合法 answer 动作”，而不是模型不会搜索或检索服务失效。

还可以解释实验严谨性：

> 新旧 run 的前 16 条 normal generation 逐字节一致，新增 terminal 没有改变其原结果；第二批因 terminal sampling 推进 RNG 而重新采样，所以总 EM 6/32 到 5/32 不能做单变量因果结论。我只把结构一致性和受控前半批作为确定性证据，把效果结论留给训练后多样本评测。

不能宣称：

- “terminal 让 EM 提升”——全局没有受控提升；
- “模型已经学会停止”——15/19 terminal 仍请求 search；
- “R60 一定成功”——当前尚未跑 backward smoke；
- “关机脚本证明停止计费”——证据只证明 guest shutdown dispatch 返回 0。

## 14. 证据索引

远端持久化证据：

```text
/root/autodl-tmp/search-r1/state/attempts/gpu/20260726T105027Z-1597-2709
/root/autodl-tmp/search-r1/runs/qwen-native-gate/attempts/20260726T105027Z-1597-2709
/root/autodl-tmp/search-r1/runs/eval/qwen_native_g0/attempts/20260726T105205Z-1646-15875
/root/autodl-tmp/search-r1/runs/eval/qwen_native_g1/attempts/20260726T105627Z-1646-13281
/root/autodl-tmp/search-r1/runs/eval/qwen_native_g1/attempts/20260726T105627Z-1646-13281/traces/eval_predictions.jsonl
```

本报告的数字以原始 trace、模型实际可见的 `visible_observation` 和项目 strict EM 为准。完整 retrieved document 只用于检索器审计，不能冒充模型已见上下文；reasoning 中出现正确字符串也不能冒充合法 `<answer>` action。
