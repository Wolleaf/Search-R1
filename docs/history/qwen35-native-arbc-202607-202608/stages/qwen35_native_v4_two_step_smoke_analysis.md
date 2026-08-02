# Qwen3.5 Native-v4 2-step Smoke 训练与轨迹分析

> 分析日期：2026-07-28
>
> 模型：`Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc`（post-trained，不是 Base）
>
> 训练提交：`f8c1cd7e87078d07385f74ca8710add5d5f79c06`
>
> 当前结论：**训练流水线 operational GO；只允许保持 exact 合同启动 R60，不代表两步已经证明能力或成本优化有效。**

## 1. 结论先行

这次 smoke 完成了两次真实的全参数 GRPO 更新，不再只是前向 Gate。最终退出码为 0，`global_step_2` checkpoint、WandB、完整训练轨迹及其哈希均已封存。结论分为六点：

1. **双卡全参数训练链路已经跑通。** 两步 rollout、reference log-prob、GRPO advantage、backward、FSDP update、验证和 checkpoint 保存全部完成；未出现 OOM、NaN、Inf、Traceback 或 NCCL 运行错误。
2. **数值信号存在且有限。** actor policy-gradient loss、reference KL、entropy 和 gradient norm 两步都有有限值；16 个 prompt group 中有 7 个 mixed-reward group，35/80 条轨迹和 18,581 个 policy token 得到非零 advantage。
3. **当前策略仍有明显问题。** 80 条轨迹 strict EM 为 `15/80 = 18.75%`，仅 40 条形成合法答案；其余 40 条没有答案。共执行 231 次 BM25 检索，平均 `2.8875` 次。
4. **成本优化空间真实存在。** strict 正确轨迹平均搜索 `2.333` 次，无答案轨迹平均搜索 `3.5` 次；26/40 条无答案轨迹用满四次搜索。该相关性提示“找到证据后没有及时回答”是主要浪费来源，但不构成减少搜索仍保持正确的反事实证明。
5. **terminal 安全成立，软提醒服从仍弱。** 49 条耗尽四个 action 槽的轨迹全部收到 reminder，31 次越界 search 均被拒绝且 0 次执行；只有 9/49 合法回答，另有 9/49 产生其他非法动作。
6. **可以进入 R60，但不能改实验合同后沿用本结论。** R60 必须从原始 sealed Qwen3.5-2B parent 重新启动，不能从 smoke checkpoint 接着训；保持 group 5、batch 8、500/500、四轮预算、采样参数、prompt/reminder、strict EM 和 `lambda=0`。启动前先解决自动关机的 WandB reader 问题。

这里的 `GO` 只说明 exact 配置具备继续训练所需的工程与数值条件。它没有证明 R60 后 EM 会提升、搜索次数会下降、terminal reminder 会被学会，也没有验证成本感知奖励；本次 `cost_lambda=0`。

## 2. 运行身份、血缘与证据

| 项目 | 精确值 |
| --- | --- |
| 外层 attempt | `20260728T061246Z-1316-15067` |
| smoke run | `20260728T061454Z-1352-31865` |
| result contract | `qwen-native-training-smoke-v5` |
| smoke decision | `search-r1.qwen-native-smoke-decision@3` / `GO` |
| parent digest | `bc67be20efb353ba14d9c1b291a64410afec94f2310e94b59b6b76047b164e78` |
| CPU handoff digest | `9ffaa89f88990887b368750ccfdc559d93afdb9a1f5719411cf3e3c1147da90b` |
| resolved config digest | `41a642bdfa105fec0d1d17348b2decf708656fbd1de0f3f087362701c09ee7d5` |
| trace digest | `7f771c356ad24cb086e49b78bcce1085d856c593f7de1f387ff7b1d37533a3e1` |
| train log digest | `df5d1d8a9fee6862d07f9b28bfd1e08c194124ceb2c8092513d9aed4e17ae0d4` |
| WandB tree digest | `52a05cf67a4f6b75e7575a154b5ad6336cbeeb31d0f8a1a7e25de6ce3de8f8a5` |
| step-2 checkpoint digest | `229b22e7c8e3ac5d64961cc4a7c5bc1676db9e4ab19d0c9839eaba4b939911de` |
| 外层 evidence digest | `dd6f8c496a1ee415f382a68d64aef6311aec8ca5617e375ce164e2ef5e8c3014` |
| 外层终态 | `terminal=success`、`exit-code=0` |

原始 trace 为 `search-r1.trajectory@3`，共 80 行、6,677,597 bytes；manifest 声明的行数、文件大小和 SHA-256 与独立复算完全一致。16 个 group 均恰好包含 `group_slot=0..4`，没有缺行、重复或跨组污染。

外层 attempt 从 `06:12:46Z` 到 `06:54:31Z`，共 2,505 秒，即 41 分 45 秒。按两卡整机 `5.76 元/小时` 做 wall-clock 乘法，实验 attempt 约 `4.01 元`；这不包含平台启动、计费粒度以及实验结束后控制面真正停止的延迟。

WandB runtime 为 2,275.16 秒，即 37.92 分钟。smoke 目录占约 9.0 GB，几乎全部来自全参数 checkpoint。当前数据盘为 150 GB，已用 104 GB、可用约 47 GB；单个 R60 末端 checkpoint 加轨迹可以容纳，但后续同时保留多个 R/B/C checkpoint 会变得紧张，届时需按实验血缘有选择地保留。

### 2.1 为什么训练完成后没有自动关机

训练和证据封存均成功，失败仅发生在 watchdog 的关机前二次授权：

```text
WandB history scan failed: the pinned WandB reader is unavailable
Shutdown skipped: authorization-revalidation-failed
```

watchdog 按 fail-closed 设计保留了机器，没有把关机失败覆盖为训练失败；随后已人工补发 guest shutdown。下一次付费 GPU 运行前必须修复或重新验证 pinned WandB reader，否则长训结束后仍可能空挂机。guest 收到 shutdown 命令也不等于 AutoDL 控制面已经停止计费，仍应以平台状态为准。

## 3. 本轮 exact 训练合同

| 参数 | 值 |
| --- | --- |
| 训练方式 | Qwen3.5-2B 全参数 FSDP，非 LoRA |
| GPU | 2 卡 |
| train batch / group | `8 / 5`，每步 40 条 rollout |
| steps / trajectories | `2 / 80` |
| max regular actions | `4`，未结束时再进行 1 次 terminal generation |
| response / observation | 每次 generation `500` / 每次检索回填 `500` token |
| temperature / top-p | `1.0 / 1.0` |
| top-k / min-p | `0 / 0.0` |
| presence / repetition penalty | `0.0 / 1.0` |
| learning rate / warmup ratio | `1e-6 / 0.285` |
| PPO epochs / mini batch | `1 / 40` |
| actor micro batch | `2` |
| grad clip | `1.0` |
| KL / entropy coefficient | `0.001 / 0.001` |
| reward | strict EM，`cost_lambda=0`，linear mode |
| retriever | 官方支持的 BM25，top-3 |
| prompt | `qwen35-native-search-v4-terminal-answer-only` |

actor、reference 和 optimizer 使用 CPU offload，actor 开启 gradient checkpointing，rollout 为 bfloat16。日志中的 `gpu_memory_utilization=0.5` 是配置值而不是实测显存；本轮没有记录 peak allocated/reserved，因此只能证明已观测两批可运行，不能量化剩余显存或保证所有 R60 长轨迹绝不 OOM。

### 3.1 数据组成

训练集共 512 题：320 条 HotpotQA（264 bridge、56 comparison）和 192 条 NQ single，比例为 `62.5% / 37.5%`。本次两个 shuffle batch 恰好各抽到 5 条 HotpotQA 和 3 条 NQ，因此 16 个 prompt 为：

| 来源 | prompt | rollout |
| --- | ---: | ---: |
| HotpotQA | 10 | 50 |
| NQ | 6 | 30 |
| 合计 | 16 | 80 |

验证集为固定 128 题，NQ 和 HotpotQA 各 64。train/val question overlap 已由数据合同验证为 0。

### 3.2 两步的 policy 时点不同

本次 80 条 trace 不是同一个冻结模型产生的：

```text
sealed parent -> step-1 rollout -> update 1
              -> step-2 rollout -> update 2 -> val-128 -> global_step_2
```

step 1 和 step 2 又使用完全不同的 8 道题。因此，不能把 step 1 的 EM 27.5% 与 step 2 的 10% 写成“一次更新导致模型退化”，也不能把第二批 10% 当成最终 checkpoint 的 EM。最终 checkpoint 只由 update 2 后的 val-128 指标覆盖。

## 4. Loss、梯度与训练数值

| 指标 | Step 1 | Step 2 | 解释 |
| --- | ---: | ---: | --- |
| actor pg loss | `0.022361` | `-0.005948` | 有限；不同 on-policy batch 下变号不代表好坏 |
| actor entropy | `0.554647` | `0.576223` | 有限，未见两步内熵坍缩 |
| reference KL loss | `0.000307` | `0.000546` | 绝对值仍约 `1e-4`，未见爆炸 |
| actor PPO KL | `0` | `0` | 单 PPO epoch 下 old/current ratio 的预期表现之一 |
| PPO clip fraction | `0` | `0` | 本 smoke 未覆盖多 epoch ratio clipping |
| grad norm | `3.004049` | `1.747180` | 均有限；日志是 clip 前 norm，两步都触发 `grad_clip=1` |
| train EM/reward | `11/40 = 0.275` | `4/40 = 0.10` | 不同题目，不能当趋势 |
| mixed groups | `4/8` | `3/8` | 共 7/16 |
| nonzero-adv trajectories | `20/40` | `15/40` | 共 35/80 |
| policy tokens | `19,384` | `29,812` | observation/reminder token 不在其中 |
| nonzero-adv policy tokens | `8,630` | `9,951` | 合计 18,581/49,196 = 37.77% |
| true clipped trajectories | `1/40` | `5/40` | 来自 raw trace，不是日志动态宽度指标 |

`pg_loss` 不是监督学习中应单调下降的交叉熵 loss。它由当前 batch 的组内 advantage、old/new log-prob 和 PPO ratio 共同决定；只有两个点时，最合理的判断是“存在有限梯度且 update 能完成”，不能画出收敛趋势。

`actor/ppo_kl=0` 也不表示没有更新：两步的 pg loss 和 clip 前 grad norm 均非零，checkpoint digest 也与 parent 不同。当前 `ppo_epochs=1`，rollout old policy 与该批开始更新前的 current policy 相同，因此近似 PPO KL 和 clip fraction 为 0 并不反常。

### 4.1 最终验证指标

| 数据集 | strict EM | 平均实际搜索 | no-search ratio |
| --- | ---: | ---: | ---: |
| NQ，64 题 | `0.25` | `2.90625` | `0.046875` |
| HotpotQA，64 题 | `0.50` | `3.328125` | `0.03125` |
| 算术合计 | `48/128 = 0.375` | - | - |

本轮设置 `val_before_train=false`，没有同一 val-128 上的 sealed-parent 基线，因此 37.5% 只能作为最终 checkpoint 的 endpoint，不能写成训练提升。验证也没有封存逐条 trace，所以不能从该聚合指标推断最终 checkpoint 的 terminal compliance 或 invalid 分布。

### 4.2 耗时与 R60 粗估

| 项目 | Step 1 | Step 2 |
| --- | ---: | ---: |
| generation | 302.17 s | 491.08 s |
| reference | 8.31 s | 9.20 s |
| actor update | 68.34 s | 74.69 s |
| validation | - | 1,271.74 s |
| checkpoint | - | 12.63 s |
| step wall time | 386.62 s | 1,868.10 s |

扣除最终验证与 checkpoint，两个训练 step 分别约 6.44 和 9.73 分钟。若长度分布和吞吐量保持相近，R60 可先按约 7-10 小时、40-60 元做宽松预算；这是两点外推，不是工期承诺。R60 只在末端验证和保存，真实时长仍会受生成长度、检索服务和平台波动影响。

## 5. 80 条训练轨迹的总体结果

| 指标 | 总计 | Step 1 | Step 2 |
| --- | ---: | ---: | ---: |
| prompt / trajectories | 16 / 80 | 8 / 40 | 8 / 40 |
| strict EM | 15/80 = 18.75% | 11/40 = 27.5% | 4/40 = 10% |
| 合法答案 | 40 | 23 | 17 |
| 无答案 | 40 | 17 | 23 |
| 实际搜索 | 231 | 109 | 122 |
| 平均搜索 | 2.8875 | 2.725 | 3.05 |
| 进入 terminal | 49 | 20 | 29 |
| any-invalid trajectory | 45 | 20 | 25 |
| clipped trajectory/event | 6/10 | 1/1 | 5/9 |

按数据源分层：

| 来源 | 轨迹 | strict EM | 合法答案 | 平均搜索 | terminal |
| --- | ---: | ---: | ---: | ---: | ---: |
| NQ | 30 | 4/30 = 13.33% | 16 | 2.633 | 18 |
| HotpotQA | 50 | 11/50 = 22% | 24 | 3.04 | 31 |

### 5.1 搜索次数与结果

| 实际搜索数 | 轨迹 | strict 正确 | 正确率 |
| ---: | ---: | ---: | ---: |
| 0 | 1 | 1 | 100% |
| 1 | 12 | 3 | 25% |
| 2 | 16 | 5 | 31.25% |
| 3 | 17 | 2 | 11.76% |
| 4 | 34 | 4 | 11.76% |
| **合计** | **80** | **15** | **18.75%** |

更有解释力的是按终局拆分：

| 终局 | 轨迹 | 搜索总数 | 平均 | 中位数 |
| --- | ---: | ---: | ---: | ---: |
| strict 正确答案 | 15 | 35 | 2.333 | 2 |
| 有答案但 strict 错 | 25 | 56 | 2.24 | 2 |
| 无答案 | 40 | 140 | 3.5 | 4 |
| 所有有答案 | 40 | 91 | 2.275 | 2 |

正确和“提交了错误答案”的搜索均值很接近，主要成本浪费集中在“搜索很多但没有提交答案”。40 条无答案轨迹中，26 条执行了四次搜索，另有 8 条执行三次、6 条执行两次。

这些数字说明当前任务具备成本优化空间，但不能直接证明“少搜导致正确”。题目难度、检索质量、输出格式和是否回答共同混杂。真正的成本收益仍需在相同题目、相同 parent、相同采样合同下比较 B/C，并同时约束正确率。

### 5.2 Group reward 与 advantage

16 个 group 的正确数分布为：

| group 内正确数 | group 数 | trajectory 数 | advantage |
| ---: | ---: | ---: | --- |
| 0/5 | 8 | 40 | 全部 0 |
| 1/5 | 4 | 20 | 正确 `+1.78885`，错误 `-0.447213` |
| 2/5 | 3 | 15 | 正确 `+1.095443`，错误 `-0.730295` |
| 5/5 | 1 | 5 | 全部 0 |

所以只有 7 个 mixed group、35 条轨迹产生相对训练信号。15 条 strict 正确中有 10 条为正 advantage；全对 group 的 5 条正确轨迹没有组内偏好。40 条无答案中有 18 条为负 advantage、22 条为 0；如果同一道题的五条 rollout 全部不回答，GRPO 无法仅靠组内标准化区分哪条更接近正确。

这也是 group 5 合理但并非万能的原因：它已经成功制造了 7 个 mixed group，但 reward 稀疏和 strict EM 噪声仍决定有效信号密度。

## 6. Terminal reminder 的实际训练影响

四个常规 action 槽耗尽后，当前实现注入：

```text
The search budget is exhausted. You must not call the search tool again.
Using only the question and information already available, give your best
answer even if uncertain. After reasoning, output exactly one concise
final answer inside <answer> and </answer>, with no text after </answer>.
```

49 条进入 terminal 的轨迹结果如下：

| Terminal 行为 | 轨迹 | strict EM | 环境处理 |
| --- | ---: | ---: | --- |
| 合法 `<answer>` | 9/49 = 18.37% | 5/9 | 正常结束 |
| 再次请求 search | 31/49 = 63.27% | 0/31 | 31/31 拒绝，0 接受、0 执行 |
| 其他非法动作 | 9/49 = 18.37% | 0/9 | fail closed |

其他非法动作由 `invalid_thinking_prefix` 6 条、`invalid_terminal_answer_format` 1 条、`unknown_tool` 1 条和 `missing_native_action` 1 条组成。原始 `requested_action=answer` 有 10 条，其中 1 条没有满足 terminal 的“恰好一个 answer 且之后无文本”合同，所以正式合法 answer 数为 9。

### 6.1 `policy mask=0` 不等于没有训练影响

- reminder 49/49 完整注入；它及 chat-template token 的 policy mask 为 0，因此这些 user token 本身不计入 policy loss。
- reminder 后的 terminal assistant 共生成 10,823 个 policy token，占本次全部 49,196 个 policy token 的 `22.00%`。
- 这些 assistant token 的概率分布以 reminder 为条件，并根据整条轨迹的 sequence advantage 更新，所以 reminder 对训练上下文有实质影响。

终局行为实际收到的 GRPO 信号为：

| Terminal 行为 | 正 advantage | 负 advantage | 0 advantage |
| --- | ---: | ---: | ---: |
| 合法 answer | 3 | 0 | 6 |
| 被拒 search | 0 | 14 | 17 |
| 其他非法 | 0 | 4 | 5 |

这说明 reminder 确实提供了可学习的正负样本：3 条 terminal 正确回答得到正向强化，18 条失败终局得到负向更新。但仍有 28 条 terminal 轨迹位于全对或全错 group，advantage 为 0，不会产生组内策略偏好。

### 6.2 应不应该保留

本次有 5 条 strict 正确答案来自 terminal continuation；若没有最后一次 generation，它们会停留在“尚未回答”状态。但没有复用同一前四轮状态的 no-reminder continuation，因此不能把这 5 条因果归功于 reminder，而不是原版已有的免费 terminal generation。

当前 R reward 只看 EM，`cost_lambda=0`。一条四搜后靠 reminder 答对的轨迹可以得到正 advantage，从而同时强化此前四次搜索和最后回答；reminder 自身不能教会模型提前停止。后续成本感知 C 分支才负责在正确条件下区分不同真实搜索成本。

对当前实验线，建议 **保留 reminder**：Gate、smoke、后续 train/eval 都已使用同一环境合同，且它解决了原 prompt 写“可以任意搜索”而环境实际四轮封锁的状态不可见问题。现在删除会改变 observation/MDP 并使现有血缘不能直接沿用。报告和简历必须披露：上游原版只有免费 terminal generation，没有这条 user reminder；这是本项目新增的有限预算状态适配。

若后续要证明 reminder 本身有效，最严谨的低成本实验仍是冻结完全相同的前四轮状态，只对 terminal continuation 做 paired 有/无 reminder A/B，而不是训练两个完整模型。

## 7. Invalid 与 500-token clipping

337 个 generation/action event 可以精确对账为 231 次真实搜索、40 次合法答案和 66 次非法 action：

| Parse error | 常规轮 | Terminal | 合计 |
| --- | ---: | ---: | ---: |
| `search_disallowed_after_budget` | 0 | 31 | 31 |
| `invalid_thinking_prefix` | 20 | 6 | 26 |
| `multiple_or_unbalanced_tool_calls` | 2 | 0 | 2 |
| `missing_native_action` | 1 | 1 | 2 |
| `unknown_tool` | 2 | 1 | 3 |
| `multiple_or_unbalanced_answers` | 1 | 0 | 1 |
| `invalid_terminal_answer_format` | 0 | 1 | 1 |

31 个 terminal search rejection 是模型不服从 reminder，但也是环境按设计执行的安全拒绝。排除它们后，真正的格式/协议错误为 35 个 event、涉及 22/80 条轨迹；19 条轨迹在常规轮就至少发生一次错误。

10 个 generation 恰好达到单轮 500 token 并被截断，集中在 6/80 条轨迹；5 个发生于常规/免费 retry，5 个发生于 terminal。10 个全部最终无答案，其中 9 个形成 `invalid_thinking_prefix`，1 个形成不平衡 tool call。step 2 占 9/10，但两步题目不同，不能归因于一次 update。

日志的 `response_length/clip_ratio=0.025` 只是 assembled response tensor 中达到该批动态宽度的比例，不是单轮 500-token 真实截断率。判断截断必须使用 trace：本轮真实受影响轨迹为 7.5%。500 已经让绝大多数轨迹不截断，因此当前不应再次加长并放大显存；R60 继续监控 `response_clipped` 和逐 generation `clipped` 即可。

## 8. Strict EM 与答案表达噪声

项目 strict EM 会小写、去英文标点和冠词、合并空白，但不会转换数字/单词、交换日期顺序或接受包含额外内容的长答案。trace 支持 `gold_answers` 列表并对任一 alias 取最大值，但本次 80 条轨迹对应的 16 题都只有一个 gold answer。

15/80 是正确的官方复算结果，80/80 与 trace 一致；同时存在清晰的语义近似漏计：

| Gold | 预测 | strict EM | 原因 |
| --- | --- | ---: | --- |
| `five` | `5` | 0 | 数字形式不转换 |
| `27 April` | `April 27` | 0 | token 顺序不同 |
| `German` | `Germany` | 0 | 国籍与国家名不同 |
| `Bergen County` | `Bergen County, New Jersey` | 0 | 预测含额外内容 |
| `Alexander "Sandy" Mayer` | `Sandy Mayer` | 0 | gold alias 不完整 |
| `1,236` | `Sullivan, Maine had a population of 1,236...` | 0 | 冗长答案 |

25 条“有答案但 strict 错”的轨迹中，有 7 条能通过现有 substring-EM 启发式；数字和日期顺序等其他语义等价项甚至不会被 substring-EM 捕获。这说明 reward 中确实混有表达噪声，但不应在看到 smoke 后临时换 reward、放宽 parser 或追溯改写结果。

R 基线继续保持论文/项目 strict EM。可以在最终评测旁路增加 alias/sub-EM/人工审计作为诊断，但不能用它替代预注册主指标；如果未来要扩充 gold alias，应形成独立数据版本和对照，而不是静默修改当前 lineage。

## 9. 与 Gate-v5 parent 的正确比较方式

[Gate-v5 报告](qwen35_native_v4_terminal_gate_v5_g0_g1_trajectory_analysis.md)和本次 smoke 共享同一 sealed parent、native prompt/tool、BM25、四轮预算、500/500、采样参数和 terminal reminder。但两者不是配对前后评测：

- Gate-v5：16 个 held-out HotpotQA val prompt × group 2，32 条冻结 parent 前向轨迹，无训练。
- Smoke：16 个不重叠 train prompt × group 5，80 条轨迹；step 1 来自 parent，step 2 来自 update 1 后的 policy。

以下只能作为两个阶段的观测台账，不能计算训练 delta：

| 指标 | Gate-v5 | Smoke |
| --- | ---: | ---: |
| 轨迹 | 32 | 80 |
| strict EM | 12/32 = 37.5% | 15/80 = 18.75% |
| 合法答案 | 22/32 = 68.75% | 40/80 = 50% |
| 平均实际搜索 | 2.75 | 2.8875 |
| any-invalid | 12/32 | 45/80 |
| clipped trajectory | 1/32 | 6/80 |
| terminal 合法回答 | 7/17 = 41.18% | 9/49 = 18.37% |
| terminal 再请求 search | 9/17 = 52.94% | 31/49 = 63.27% |
| terminal accepted/executed search | 0/0 | 0/0 |

不能据此写“训练使 EM 下降 18.75 个百分点”或“reminder 服从退化 22.81 个百分点”。样本、group、policy 时点和随机 rollout 均不匹配。唯一可靠的跨阶段结论是：terminal reminder/mask/越界搜索硬边界在真实训练路径中仍保持成立，同时 backward 和非零 advantage 已被 smoke 新验证。

## 10. 代表性轨迹

以下均来自封存 raw trace。检索回填的三篇文档正文从展示中省略，但动作顺序、query、终局和 reward 均完整保留。

### 10.1 一搜后正确回答：理想短轨迹

`trace:09b16974b1c77cf765c3b03a`，`hotpotqa:train:48335/slot4`：

```text
Question: A Life Too Short was a biography about the goalkeeper of what nationality?
Gold: German
R0 search: A Life Too Short biography goalkeeper nationality author
R1 answer: <answer> German </answer>
Result: searches=1, EM=1, no invalid, no clipping
```

同题另外两条轨迹回答 `Germany` 而 strict EM 为 0，展示了正确检索、简短表达和 gold 形式三者必须同时满足。

### 10.2 Terminal 答对，但已经多搜两次

`trace:0ac92a47879ca2a74d3b0649`，`hotpotqa:train:14702/slot2`：

```text
Question: What is the middle name of the inventor of The Chromatron?
Gold: Orlando
R0 search: The Chromatron inventor
R1 search: Ernest Lawrence middle name
R2 search: Ernest Orlando Lawrence middle name
   -> reasoning 已确定 Orlando
R3 search: Ernest Orlando Lawrence name
T4 answer: <answer> Orlando </answer>
Result: searches=4, EM=1, positive advantage=1.78885
```

它证明 terminal reminder 可以参与正确闭环，也展示了至少两次看似可压缩的搜索。由于 `lambda=0`，这条完整四搜路径仍得到正 advantage；只有后续成本对照才能证明是否能无损提前回答。

### 10.3 已在 reasoning 写出金标，仍继续 search

`trace:d1a3b59b7aa9b2076fb12da3`，`hotpotqa:train:83462/slot0`：

```text
Question: Mindaugas Sabutis was born in a city that had this population as of 2011?
Gold: 26,444
R0 search: Mindaugas Sabutis born city
R1 search: Taurage population 2011
R2 search: Taurage population 2011 number LB
R3 search: Taurage 2011 population
   -> R2/R3 reasoning 已明确 26,444
T4 search: Taurage population 2011
   -> search_disallowed_after_budget, accepted=false, executed=false
Result: no answer, EM=0
```

这个 prompt 的五条 rollout 全部四搜后无答案，group 全错，因此五条 advantage 全为 0。它准确展示了当前瓶颈：证据已经出现，但 policy 没有选择 answer；同时也展示纯组内 GRPO 在 all-wrong group 上没有相对信号。

### 10.4 500-token reasoning 循环

`trace:59fceb49c4719baf1b355e59`，`hotpotqa:train:14386/slot4`：

```text
Question: Who is older, Sandy Mayer or Stefan Edberg?
Gold: Alexander "Sandy" Mayer
R0 search: Sandy Mayer age biography
R1 search: Stefan Edberg age biography birthdate
   -> 已得到 1952 与 1966
R2: 500-token reasoning loop, no valid action
R3 retry: 500-token reasoning loop, no valid action
T4 reminder: 500-token reasoning loop, no valid action
Result: searches=2, clipped events=3, no answer, EM=0
```

这是少数真正受 500 限制影响的失败，而不是所有无答案轨迹的通因：40 条无答案中只有 6 条涉及 clipping。

### 10.5 Strict EM 语义等价漏计

`trace:240f7efc4789e38756cc8b9b`，`hotpotqa:train:49513/slot1`：

```text
Question: Oranjegekte takes place during the holiday celebrated on what day?
Gold: 27 April
R0 search: Oranjegekte holiday celebrated on what day
R1 search: Koningsdag day December
R2 search: Oranjegekte orange color Dutch royal family birthday
R3 answer: <answer> April 27 </answer>
Result: searches=3, semantic answer correct, strict EM=0
```

该轨迹应作为 metric sensitivity 案例保留，而不是事后改成正 reward。面试时可以说明：主指标遵循原版 strict EM，同时对 alias 和表达噪声做旁路审计。

### 10.6 检索证据冲突后的错误综合

`trace:f2a5c5d6c26cd7fcc2b1410b`，`nq:train:33921/slot0`：

```text
Question: how many scrabble tiles are in a game?
Gold: 102 tiles
R0 search: how many Scrabble tiles are in a game
   -> 回填同时出现“100 original tiles + 2 blanks”和“standard version 100 tiles”
R1 answer: <answer> 100 </answer>
Result: searches=1, EM=0
```

这属于证据消歧/综合失败，不是 terminal 或格式问题。它说明增加搜索次数不一定解决问题，成本策略仍必须与正确性共同评测。

## 11. 当前已经证明与尚未证明

### 11.1 已经证明

1. Qwen3.5 native thinking/tool protocol 可以完成真实多轮训练 rollout。
2. BM25 top-3 回填、observation mask 和 policy token 边界在 backward 路径中成立。
3. Group 5 能在该混合数据上产生 mixed reward 和非零 GRPO advantage。
4. 两卡全参数 FSDP 可以完成两次 update、一次 val-128 和 checkpoint 保存。
5. loss、KL、entropy、grad norm、log-prob、reward 和 advantage 均有限。
6. terminal reminder 49/49 注入且不直接进入 policy loss，越界搜索不会被真实执行。

### 11.2 尚未证明

1. 两步不足以证明 loss 收敛、EM 提升、搜索减少或策略泛化。
2. 没有相同 val-128 的 parent baseline，不能量化最终 checkpoint 增益。
3. 没有 reminder/no-reminder paired continuation，不能证明 reminder 的因果收益。
4. `lambda=0`，因此没有验证成本感知 reward、C-gated 或 B/C 差异。
5. 没有 peak VRAM 证据，不能把两步无 OOM 外推为所有最坏轨迹无 OOM。
6. strict EM 中存在 alias/表达噪声，当前有效 reward 密度不等于语义正确密度。

## 12. 决策与下一步

本次结论为：**R60 conditional GO。** 条件和顺序如下：

1. 在无卡阶段修复并验证 watchdog 的 pinned WandB reader；若修改 tracked checkout 或 handoff，按工作流重新 reseal，不能静默复用旧 marker。
2. R60 从同一原始 sealed `Qwen3.5-2B` parent 启动，不使用 `global_step_2` smoke checkpoint。
3. 不因两个 on-policy batch 的 EM 差异而调 group、batch、500/500、四轮预算、sampling、prompt/reminder、strict EM、warmup 或 learning rate。
4. R60 继续完整记录 train trajectory、WandB history、loss/KL/grad、真实 clipping、terminal 分类和 checkpoint digest。
5. 重点监控 mixed-group ratio、nonzero-adv policy-token ratio、terminal answer/search/invalid、correct/wrong/no-answer 的搜索数；单个 batch 波动不触发科学早停，NaN/OOM/证据合同破坏才属于工程阻断。
6. R60 完成后先做同一 frozen evaluation 的 A/R 对照，再决定 B20/C20；不要用本次 smoke 的 80 条训练轨迹替代正式评测。

本次 AutoDL 训练工作流的 fail-closed 关机策略保护了证据，但缺失 reader 导致额外空挂机。该经验应保留在项目复盘中：训练成功、证据成功和关机成功是三个独立状态，不能只凭 GPU 利用率或进程退出推断平台已经停止计费。

## 13. 面试可用表述

可以准确表述为：

> 我先用冻结 parent 的 Gate 验证 Qwen3.5 原生 tool calling、BM25 回填和 terminal 安全边界，再做两卡全参数 2-step GRPO smoke。Smoke 每步是 8 个 prompt、每题 5 条 rollout，共 80 条轨迹；7/16 个 group 有 mixed reward，35 条轨迹和约 37.8% 的 policy token 获得非零 advantage，两个 step 的 policy loss、KL、entropy 和 grad norm 均有限，最终 checkpoint 和 WandB/trace 证据闭环成功。行为上，正确轨迹平均 2.33 搜，无答案轨迹平均 3.5 搜，说明主要浪费来自搜到信息后没有及时回答。49 条预算耗尽轨迹全部收到 policy-mask 为 0 的 reminder，但只有 9 条合法回答，31 条仍请求 search 且全部被环境拒绝、0 次执行。我把这定义为“工程可训练、策略仍待长训”，而没有把两步 GO 冒充模型效果提升。

还应主动说明：reminder 后的 assistant token 仍参与 policy loss，所以它是新增环境 observation，不是“对训练无影响”；上游原版没有该 user reminder。当前保留它是为了显式暴露有限搜索预算，并在所有比较分支中保持相同合同。

## 14. 证据索引

远端持久化路径：

```text
/root/autodl-tmp/search-r1/state/attempts/gpu/20260728T061246Z-1316-15067
/root/autodl-tmp/search-r1/runs/qwen-native-training/attempts/20260728T061246Z-1316-15067
/root/autodl-tmp/search-r1/runs/smoke/attempts/20260728T061454Z-1352-31865
/root/autodl-tmp/search-r1/runs/smoke/attempts/20260728T061454Z-1352-31865/traces/train_trajectories.jsonl
/root/autodl-tmp/search-r1/runs/smoke/attempts/20260728T061454Z-1352-31865/wandb
```

本报告数字来自 `smoke-decision.json`、`train.log`、`wandb-receipt.json`、`resolved-config.yaml`、`native-training-contract.json` 和原始 80 条 trace，并完成三次独立交叉复算。实现语义可查阅 [tool protocol](../../../../search_r1/llm_agent/tool_protocol.py)、[trajectory logging](../../../../verl/trainer/ppo/ray_trainer.py)、[strict EM](../../../../verl/utils/reward_score/qa_em.py)、[AutoDL 操作说明](../../../../scripts/autodl/README.md)及[主复现方案](../plans/autodl_search_r1_reproduction_plan.md)。
