# Search-R1 实验全流程复盘与面试讲稿

> 用途：简历只保留最终三条高密度描述；本文用于面试时按对方兴趣展开完整的“失败 - 诊断 - 修复 - 门禁 - 训练 - 评测”故事。
>
> 最后核对：2026-08-03。事实与数字以仓库内封存报告和逐题结果为准。

## 1. 项目一句话定位

我在预算受控的条件下，基于 Qwen3.5-2B、Wiki-2018 BM25 和 GRPO 搭建了一个 Search-R1 风格的多轮检索 Agent：模型可以自主决定何时搜索、如何改写 query、如何使用检索结果继续推理，以及何时提交最终答案。项目不是只追求一个最终分数，而是完整经历了不搜索策略坍缩、奖励函数修复、数据分布重构、工具协议适配、分层门禁、长训练、分叉对照和证据封存。

这段经历最适合体现三类能力：

- **Agent 系统能力**：tool calling、多轮状态机、检索结果回填、token mask、终止边界与失败恢复；
- **算法与实验能力**：GRPO 信用分配、奖励设计、数据配比、受控分叉、配对评测和负结果诊断；
- **工程能力**：CPU/GPU 分阶段准备、两卡训练、门禁、日志与轨迹、checkpoint 血缘、哈希封存、超时恢复和自动关机。

## 2. 先统一名称，避免面试时把不同实验讲混

### 2.1 最终 A/R/B/C 身份

```text
A：封存的 post-trained Qwen3.5-2B parent
└── R：从 A 独立训练 60 步 direct outcome-RL
    ├── B：从 exact R60 权重独立启动，继续训练 20 步，reward=EM
    └── C：从同一 exact R60 权重独立启动，继续训练 20 步，加入 correct-only 搜索成本
```

- A 是“未经本项目 RL 的 parent baseline”，不是官方 `Qwen3.5-2B-Base`。
- C 不是接着 B 训练；B 和 C 是从同一个 R 权重启动的平行分支。
- R60 只保存了 Hugging Face 模型权重，没有 optimizer、scheduler 和 trainer state。因此 B/C 是从 R 权重建立新 optimizer 的新 run，不应说成严格断点续训。

### 2.2 项目里实际出现过三个不同的 C

| 名称 | 阶段 | 含义 | 结果定位 |
|---|---|---|---|
| C-old | 早期 NQ-only | 对所有轨迹线性扣搜索成本 | 发生 no-search collapse |
| C-gated | 早期 NQ-only follow-up | 只对答对轨迹计搜索成本 | 修复坍缩，但未取得成本收益 |
| 最终 C20 | native 混合数据实验 | 从最终 R60 平行训练的 correct-only 成本分支 | 相对最终 B20 降低搜索，EM 基本持平 |

面试中如果只说“C”，必须先说明指的是哪一轮。早期 NQ-only 数字不能与最终 native 混合数据数字拼成同一组实验。

### 2.3 第一次“崩”是什么

第一次不是 OOM、NaN、NCCL 或程序退出，而是 **no-search collapse（不搜索策略坍缩）**：训练和评测工程上都正常完成，但策略学会了“为了不被扣搜索成本，干脆不调用工具”。这是科学负结果，不是工程崩溃。

## 3. 最终系统与实验合同

### 3.1 Agent 闭环

```text
问题
  → Qwen3.5-2B 生成 reasoning/action
  → 选择 search 或 answer
  → search 时调用 Wiki-2018 BM25 top-3
  → 将模型实际可见的 observation 回填上下文
  → 继续生成下一轮 action
  → 最多执行 4 次真实搜索
  → 仍未结束时进入 answer-only terminal generation
  → 用 strict EM 评价最终短答案
```

最终实现使用 Qwen3.5 native tool calling，不再强迫模型遵守早期 legacy XML 文本协议。terminal reminder 属于环境上下文，不计入 policy loss；模型在 reminder 后生成的 assistant token 仍属于策略输出。terminal 阶段若继续请求搜索，系统会记录并拒绝，保证不会越过搜索预算。

### 3.2 训练配置

| 项目 | 最终合同 |
|---|---|
| 模型 | Qwen3.5-2B，全参数 FSDP，bf16 |
| GPU | 2 张 RTX 5090 级 GPU |
| 算法 | GRPO，group size 5 |
| R | 60 steps，batch 8，共 2,400 条训练轨迹 |
| B/C | 各 20 steps、各 800 条训练轨迹 |
| 检索 | Wiki-2018 BM25，top-3，最多 4 次真实搜索 |
| 单轮长度 | response 500 token，observation 500 token |
| 主指标 | strict exact match（EM） |
| 最终解码 | greedy，`do_sample=false`，每题 1 条 rollout，seed 42 |

### 3.3 奖励函数

早期 C-old 使用：

```text
r_old = EM - 0.10 × n_search / 4
```

修复后的 correct-only 奖励使用：

```text
r_gated = EM × (1 - 0.10 × n_search / 4)
```

等价地，最终 C 可写为：

```text
r_C = EM - 0.025 × n_search × EM
```

统一报告的效用指标另行计算：

```text
utility_report = EM - 0.025 × n_search
```

训练奖励和报告 utility 的区别是：错误轨迹的搜索成本不进入 C 的策略梯度，但真实搜索浪费仍进入最终评价，避免在报告中隐藏失败调用。

### 3.4 最终数据

训练集由 NQ 与 HotpotQA 组成：

- NQ 192 题；
- HotpotQA 320 题，其中 comparison 56、bridge 264；
- 合计 train-512；R60 无放回实际消费 480 个 prompt group。

最终评测为：

| 端点 | 题量 | 构成 |
|---|---:|---|
| val | 128 | NQ 64 + HotpotQA 64 |
| NQ-test | 128 | held-out NQ 128 |
| multihop | 256 | HotpotQA 128 + 2WikiMultiHopQA 128 |

四个模型在三个端点上共进行 12 个模型端点评测，即 2,048 条模型-问题轨迹。

## 4. 完整实验时间线

### 阶段 0：先让云端训练链路“值得相信”

项目最初优先解决的不是模型效果，而是如何确认一次训练真的完整、可恢复、可审计：

- 固定代码 commit、模型 revision、数据、语料和 BM25 索引；
- 将 Git、CPU 数据准备和 GPU 训练分开，使用持久盘传递 handoff；
- 修复 Qwen3.5 FSDP wrap、optimizer state 装卸和 checkpoint 保存；
- 为日志、原始 exit code、WandB、轨迹、manifest 和 SHA-256 建立统一证据路径；
- 训练或科学门禁结束后先落盘、校验、同步，再允许 watchdog 请求关机。

一个关键工程教训是：单步 smoke 不够。Adam 的部分状态在第一次 `optimizer.step` 后才创建，很多路径要到第二次更新才真正触发。因此所有长训练之前，至少完成两次真实 forward/backward/update，并验证 loss、KL、entropy、grad norm、checkpoint 和日志都有限且完整。

面试可强调：我没有用“脚本没报错”作为训练成功标准，而是把进程终态、训练数值、模型产物和证据封存分成四层检查。

### 阶段 1：NQ-only 先证明 direct RL 可学，再观察到首次不搜索坍缩

早期缩小实验使用 NQ train-512 / val-64 / test-128。它先验证了 Search-R1 主闭环能工作：

| 模型 | Test-128 EM | 总搜索 | no-search ratio | 结论 |
|---|---:|---:|---:|---|
| A | 3.91% | 164 | 22.66% | 训练前基线 |
| R60 | 16.41% | 167 | 0% | direct RL 明确学到能力 |
| B20 | 17.97% | 131 | 0% | 继续使用 EM reward 的控制组 |
| C-old20 | 7.03% | 4 | 98.44% | no-search collapse |

C-old 相比 B 把搜索减少了 96.95%，却少答对 14 题，EM 从 17.97% 降到 7.03%。训练前 5 步还有 205 次搜索和 23/200 条正确轨迹；最后 5 步只剩 4 次搜索和 6/200 条正确轨迹，第 16、20 步都出现 40/40 不搜索且 0/40 正确。

这个结果说明：成本确实下降了，但模型学到的是 reward hacking，不是高效检索。

### 阶段 2：从 GRPO 组内信用分配定位奖励漏洞

GRPO 对同一个问题采样 5 条轨迹，在 group 内做标准化：

```text
advantage_i = (reward_i - group_mean) / (group_std + 1e-6)
```

如果 5 条轨迹全部答错，EM 全部为 0，旧奖励只剩搜索成本。例如搜索次数是：

```text
[0, 1, 1, 2, 4]
```

那么“不搜索但答错”的轨迹奖励最高，会得到相对正 advantage；搜索更多但同样答错的轨迹则得到负 advantage。模型于是被明确推动去学习“答错也不要搜”。

仅把 `lambda` 从 0.10 调成更小值不能结构性解决问题，因为 group 内 mean/std 标准化会在很大程度上消掉统一缩放。真正需要改变的是排序关系，而不是只缩小惩罚数值。

因此将奖励改为：

```text
r_gated = EM × (1 - 0.10 × n_search / 4)
```

修改后：

- 全错 group 中所有 reward 都是 0，不再产生“错误但不搜索”的正梯度；
- 任意正确轨迹始终优于任意错误轨迹；
- 只有同组出现多个正确方案时，成本项才偏好搜索更少的正确方案。

这就是简历里“仅对答对轨迹计入搜索成本”的技术来源。

### 阶段 3：correct-only 修复了坍缩，但没有立刻获得成本收益

第二次 NQ-only C-gated follow-up 完成 20/20 步和 800/800 条训练轨迹：

| 模型 | EM | 总搜索 | no-search ratio | Utility |
|---|---:|---:|---:|---:|
| B | 17.97% | 131 | 0% | 0.1541 |
| C-old | 7.03% | 4 | 98.44% | 0.0695 |
| C-gated | 14.06% | 137 | 0% | 0.1139 |

C-gated 相对 C-old 将 EM 从 7.03% 拉回 14.06%，no-search ratio 从 98.44% 恢复到 0%，证明结构性反搜索激励已经移除。但它仍比 B 少答对 5 题，并多搜索 6 次，所以这轮只能定义为“结构修复成功，成本优化未成功”。

逐轨迹分析进一步解释了为什么：

- 160 个 GRPO group 中有 108 个全错，占 67.5%；
- 109 个 group 的 reward 方差为 0；
- 800 条轨迹中有 545 条 sequence advantage 为 0；
- B 在 128 题中已有 125 题只搜索一次，继续压缩的空间本来就很小。

这一步得到的关键判断是：奖励公式安全只是必要条件。要让成本奖励真正可学，还需要更多“同题都答对、但搜索次数不同”的轨迹，以及本身具有真实多跳搜索机会的数据。

### 阶段 4：先做多跳搜索机会门禁，不再盲目训练

为了判断旧策略是否真的存在可压缩的多搜行为，项目没有马上启动第三轮成本训练，而是固定 B/control20，在 HotpotQA 和 2WikiMultiHopQA dev 各抽取 128 题，共 256 题进行只评测门禁。

结果为：

- 61/256 答对，EM 23.83%；
- 248/256 只搜索一次；
- 8/256 搜索两次，但 8 条全部答错、全部截断且包含非法动作；
- 7/8 的第二次 query 与第一次完全相同；
- 所有答对题都只搜索一次；
- 没有任何三搜或四搜轨迹。

因此门禁判定 NO-GO。它说明旧 B 没有暴露出“从三搜压到二搜，同时保持答案正确”的学习空间；最容易被成本奖励学到的仍然是“一搜变零搜”。

这个门禁的价值是节省预算并避免无效调参：先确认训练信号是否存在，再决定是否烧 GPU，而不是调到结果好看为止。

### 阶段 5：从 NQ-only 转向检索可验证的 NQ + HotpotQA 混合数据

数据调整不是单纯“增加多跳题”这么简单。训练样本必须在当前 Wiki-2018 BM25、top-3 和 observation 截断条件下，真正把所需证据暴露给模型，否则 benchmark 标签是多跳，模型实际看到的 observation 却不具备可学习链路。

项目因此使用 `visible_observation` 作为筛选真值：先执行检索，再按 tokenizer 和长度上限截断，最后检查模型真正可见的文本，而不是只看检索器返回的原始全文。

数据设计经历了三个变化：

1. 从早期 NQ-only 转向 NQ + HotpotQA；
2. 从偏均衡的 NQ/Hotpot 方案调整为 NQ 37.5%、Hotpot 62.5%，增加多跳训练机会；
3. Hotpot 内部最终重登记为 comparison 56、bridge 264。

comparison 候选中只有 74 题严格满足 BM25 与 384-token 可见证据条件，而 bridge 有 766 题可用。项目没有为了凑齐原配比放宽标准，而是显式调整配额，由更适合学习“从第一轮证据提取第二个 query”的 bridge 承担主要多跳信号，同时保留 192 道 NQ 维持稳定的一搜作答梯度。

最终 train-512 为：

```text
NQ 192
HotpotQA comparison 56
HotpotQA bridge 264
```

NQ test、HotpotQA/2Wiki dev 等正式评测数据被排除在训练选择之外，并通过 manifest、catalog、ledger 和 digest 固定。

### 阶段 6：旧 XML grouped probe 暴露的不是“模型不会搜”，而是协议错配

在训练前 grouped probe 中，固定 64 道 HotpotQA、每题 5 条，共生成 320 条轨迹：

- strict EM 只有 7/320；
- invalid 185/320；
- clipped 102/320；
- 312 次搜索中有 210 次 query 是字面量 `query`、`and` 或空字符串；
- 有效正确多搜只有 2 条。

逐轨迹检查发现两个确定性问题：

1. prompt 示例中的 `<search> query </search>` 被模型直接复制，导致搜索字面量 `query`；
2. parser 使用非锚定正则，模型复述“between `<search>` and `</search>`”时，标签之间的英文 `and` 被误当成真实 query。

这说明 Qwen3.5 已经过原生 tool-use 后训练，但实验却强迫它遵循 legacy XML 文本协议，prompt、parser 和 action boundary 之间存在系统性错配。此时继续增加 response 长度或训练步数，只会扩大错误轨迹成本。

因此后续方向从“继续调奖励”切换为“先修 Agent 协议”。

### 阶段 7：适配 Qwen3.5 native tool calling，并建立分层门禁

协议适配不是一次完成的，而是按失败证据逐层迭代：

- native-v1 已出现可学习内容，但整段自由文本被当作答案，正式 strict EM 为 0；
- v2 恢复唯一 `<answer>` 短答案边界，并统一训练与评测采样合同；
- 首次 v3 暴露 reasoning marker 与单 token overshoot 的 answer boundary 工程故障；
- 修复后 v3 能健康检索，主要问题收敛为“已经看到答案仍继续搜索”；
- native-v4 加入 answer-only terminal reminder、Gold 样本审计和更严格 evidence/W&B gate；
- 51 条有缺陷的 gold 记录被剔除，并按原分布确定性补位。

项目建立了以下门禁框架：

| 门禁 | 主要回答的问题 | 失败时的处理 |
|---|---|---|
| G0 | prompt、token、action region、direct/native replay 是否一致 | 停止，不进入真实工具闭环 |
| E0 | BM25 是否能真实启动、回填 top-3，observation mask 是否正确 | 停止，不解释为模型问题 |
| G1 | 冻结 parent 是否能自主产生合法 search/answer action | 修 prompt/parser/state machine |
| G2/readiness | invalid、clipping、退化 query、短答案和 on-policy 合同是否健康 | 停止，不进入长训练 |
| 2-step smoke | 两次真实 backward/update 是否数值稳定且有非零学习信号 | 停止，不运行 R60 |
| G3 | R60 后是否存在可学习的正确多搜与成本对比，同时协议质量达标 | 作为科学 GO/NO-GO，不自动进入 B/C |

这些门禁不是一次性直线通关。早期 G1/G2 类型失败推动了 native-v1 到 v4 的协议修复；最终 Gate-v5 的 G0/G1 工程 GO 只允许进入两步 smoke，不等于模型能力已经通过。

Gate-v5 验证了 token prefix、真实 BM25 回填、tool-response mask 和 terminal 安全边界。模型在耗尽预算后仍有 9 次请求 search，但环境 9 次全部拒绝、0 次执行，说明硬预算生效；模型是否学会主动停止则留给 RL。

### 阶段 8：两步 smoke 通过后，才启动 R60

两步 smoke 完成了两次真实全参数 GRPO update：

- 7/16 group 存在 mixed reward；
- reward、advantage、loss、KL、entropy、grad norm 均为有限值；
- checkpoint、W&B history 和 80 条训练轨迹完整；
- 无 OOM、CUDA、NCCL、Ray 或检索服务错误。

正式 R60 不继承 smoke 权重，而是从同一个 sealed parent 重新启动，因为 2-step 和 60-step 的 scheduler horizon 不同。

R60 最终完成：

- 60/60 次全参数更新；
- 480 个 prompt group；
- 2,400 条训练轨迹；
- 5,697 次真实 BM25 搜索；
- 无 OOM、NaN、Inf、NCCL 或运行时错误；
- 保存 `global_step_60` 和完整 W&B/trace/evidence。

训练期固定 val-128 上，从独立 smoke step-2 endpoint 到 R60 endpoint：

| 指标 | Smoke step 2 | R60 | 变化 |
|---|---:|---:|---:|
| strict EM | 37.50% | 58.59% | +21.09pp |
| 平均搜索 | 3.117 | 1.992 | -36.1% |
| no-search | 5/128 | 1/128 | 未发生不搜索坍缩 |

训练轨迹也显示二搜正确率最高：二搜轨迹 457/790 正确，正确率 57.85%；零搜索只有 43/2,400，说明模型学到的是更有效地搜索和作答，而不是通过不搜索来缩短轨迹。

但 R60 不是“全面变好”：

- 增益主要来自 NQ，固定 val 中 NQ 多答对 25 题，HotpotQA 只多 2 题；
- step 52 后 KL、生成长度、clipping 和 `invalid_thinking_prefix` 同步上升；
- 最后 10 步更新前轨迹 clipping 达到 50.5%，协议异常达到 46.5%。

因此没有直接启动成本分支，而是对冻结 R60 运行 held-out G3。

### 阶段 9：G3 证明“能力存在”，也证明“稳定性没有过门”

G3 使用 held-out HotpotQA 64 题、每题 5 条，共 320 条轨迹。结果为：

- strict EM：151/320 = 47.19%；
- 总搜索：724 次；
- 至少二搜：280/320 = 87.5%；
- 严格有效的正确多搜轨迹：98/320；
- 覆盖题目：34/64；
- clean cost-contrast group：13/64。

这证明 R60 确实学到了多轮检索和成本对比机会。它与早期 grouped probe 的 7/320 EM、2 条有效正确多搜形成明显对照。

但预注册能力门要求五项同时通过：

| 指标 | 实际值 | 门槛 | 判定 |
|---|---:|---:|---|
| 有效正确多搜轨迹 | 98/320 | 至少 16 | PASS |
| 覆盖题目 | 34/64 | 至少 8 | PASS |
| clean learnable group | 5/64 | 至少 8 | FAIL |
| clipping | 46.88% | 不高于 5% | FAIL |
| invalid | 43.75% | 不高于 5% | FAIL |

clean 轨迹 EM 为 71.15%，dirty 轨迹只有 24.39%。因此瓶颈已经从“不会搜索”转变为“长输出时丢失 native action 与答案边界”。G3 的正式结论是科学 NO-GO，不能说已经通过。

后续 B/C 不是 G3 自动放行的确认性实验，而是在 G3 已有 13/64 cost-contrast 信号后，另立合同开展的 post-hoc exploratory 分叉。这一点面试时要主动讲清楚。

### 阶段 10：第一次 B/C 的失败是调度顺序，不是训练数值崩溃

第一次最终 B/C 运行中：

- B20 完整成功并保存 checkpoint；
- 旧 C 也生成了 800 条训练轨迹；
- 但第 20 步先进入 inline validation，在最终 checkpoint 保存前触发超时；
- 六个外部评测尚未执行；
- outer 终态为 `failed/124`。

根因是预算余量不足，加上“先验证、后保存”的调度顺序。它不是 OOM、磁盘写满、NCCL、NaN 或 retriever 失败。由于旧 C 没有完整最终 checkpoint，所以不能采用，也不能根据训练轨迹补造最终结果。

恢复方案没有重算 B：

1. 校验并复用已经完整的 B20；
2. 从 exact R60 权重重新启动 C20；
3. 关闭 inline validation；
4. 先保存 `global_step_20`；
5. 再顺序执行 B/C 的 val、NQ-test 和 multihop 六项评测；
6. 最后生成 lineage、paired result、evidence manifest 和终态 marker。

恢复后八个科学 inner run 全部 `success/0`。outer 的 `failed/203` 是 seal 完成后用于冻结 watchdog failure path 的受控状态，不代表训练失败。这个案例说明为什么必须区分 launcher 退出码、内层科学结果和证据封存状态。

### 阶段 11：A/R 补评后完成统一 A/R/B/C 闭环

最后又将 A 和 exact R60 放到与 B/C 完全一致的三套 endpoint 上评测，确保四模型使用相同问题、相同行序、相同 gold answer、相同 greedy 解码和相同检索协议。

最终结果如下，单元格为“strict EM / 平均搜索”：

| 端点 | A | R | B | C |
|---|---:|---:|---:|---:|
| val-128 | 33.59% / 3.234 | 58.59% / 1.992 | 61.72% / 1.930 | **64.84% / 1.680** |
| NQ-test-128 | 7.81% / 2.727 | **25.00% / 2.523** | 23.44% / 2.234 | 24.22% / 2.086 |
| multihop-256 | 21.88% / 3.621 | **37.11% / 2.902** | 30.08% / 3.074 | 29.69% / 2.852 |
| 512 题描述性汇总 | 21.29% / 3.301 | **39.45% / 2.580** | 36.33% / 2.578 | 37.11% / **2.367** |

### A→R：最可靠的主结果

R 相对 A：

- val EM +25.00pp，95% CI `[15.63, 34.38]`；
- NQ-test EM +17.19pp，95% CI `[10.16, 25.00]`；
- multihop EM +15.23pp，95% CI `[9.77, 21.09]`；
- 三个置信区间均不跨 0；
- 512 题中多答对 93 题，同时少搜索 369 次。

这证明 direct outcome-RL 的能力提升不是靠更多检索堆出来的。

### B→C：主要成果是搜索效率

C 相对 B：

- 512 题多答对 4 题，EM 36.33%→37.11%；
- 总搜索 1320→1212，减少 108 次，即 -8.18%；
- 4-search 饱和率 38.28%→30.27%；
- 两者都只有 1/512 零搜索轨迹，没有再次发生 no-search collapse；
- 三个端点的 C-B EM 置信区间都跨 0，不能说准确率显著提升；
- 108 次搜索净节省中有 81 次来自两者都答错的题，主要机制是减少失败轨迹的无效深搜。

最终模型定位不是单一排名，而是两个 Pareto 端点：

- R 偏综合能力和 multihop 稳健性；
- C 偏低搜索成本和较浅搜索路径；
- B 是必要的同源控制组，不是默认部署模型；
- A 只用于量化本轮 RL 的净增益。

## 5. 这段项目真正形成的技术闭环

整个项目不是“调一个 lambda 得到更好的数字”，而是以下闭环：

1. **先证明主链可学**：NQ-only A→R 明确提升。
2. **观察有效负结果**：无条件成本奖励导致 98.44% 不搜索。
3. **从算法定位根因**：GRPO 全错 group 中成本成为唯一排序信号。
4. **修复结构性激励**：改为只对正确轨迹计成本，消除反搜索梯度。
5. **承认第一次修复不等于收益**：C-gated 不再坍缩，但没有超过 B。
6. **检查训练信号是否存在**：多跳搜索机会门禁给出 NO-GO，停止盲目烧卡。
7. **重构数据**：从 NQ-only 改为检索可验证的 NQ/Hotpot 混合，并提高 bridge 比例。
8. **修 Agent 协议**：从 legacy XML 迁移到 Qwen native tool calling，修 action/answer/token 边界。
9. **建立分层门禁**：协议、检索、模型行为、真实 backward、held-out 能力逐层放行。
10. **完成长训练并主动检查退化**：R60 能力提高，但后期 clipping/invalid 上升。
11. **区分科学 NO-GO 与工程失败**：G3 能力信号强，但稳定性不过门；B/C 首次失败来自调度。
12. **恢复并统一评测**：最终得到可配对、可复算、可审计的 A/R/B/C 结果。

## 6. 面试讲法

### 6.1 30 秒版本

> 我基于 Qwen3.5-2B 做了一个 Search-R1 风格的检索 Agent，用 GRPO 学习何时搜索和何时作答。早期直接对搜索次数扣分，导致 98.4% 的题不搜索；我从 GRPO 全错 group 的组内标准化定位到反搜索信用分配，把奖励改成只对正确轨迹计成本。之后又通过多跳数据重构、Qwen 原生工具协议适配和分层门禁完成 60 步训练。最终 R 相对训练前模型在 512 题上 EM 从 21.29% 提升到 39.45%，同时少搜索 369 次；成本分支 C 相对同源 B 再减少 8.18% 搜索，并且没有再次不搜索坍缩。

### 6.2 2 分钟版本

> 项目目标是让一个 2B 模型在多轮问答中自主调用 BM25 检索，并用强化学习同时优化答案正确率和工具成本。最开始我先在 NQ-small 上跑通 Search-R1 主链，A 到 R 的 EM 从 3.9% 提高到 16.4%。但从同一 R 分叉后，朴素成本奖励让 C 的不搜索比例达到 98.4%，EM 反而降到 7.0%。
>
> 我分析 GRPO 的组内 advantage 后发现，大量 group 五条轨迹全错，此时 EM 没有区分度，搜索成本成为唯一排序信号，导致“错误但不搜索”获得正 advantage。简单调小 lambda 不能根治，因为 mean/std 标准化会抵消统一缩放。因此我把奖励改为 `EM × (1-cost)`，只在答对轨迹之间比较搜索成本。这个改动把不搜索坍缩修掉了，但第一轮 gated 实验仍没有超过控制组，因为 67.5% group 全错，而且 NQ 控制组本身已接近一次搜索下限。
>
> 接着我没有继续盲调，而是先做 HotpotQA+2Wiki 的多跳机会门禁。结果 256 题中 248 题只搜一次，仅有的 8 条二搜都错误且截断，所以我停止训练，改造数据和协议：训练集改为 37.5% NQ、62.5% Hotpot，并按模型实际可见的检索证据提高 bridge 比例；同时把 legacy XML 工具协议迁移为 Qwen native tool calling，加入 G0/G1、两步 backward smoke 和 held-out G3。
>
> 最终 R60 完成 2,400 条训练轨迹。统一评测中 R 相对 A 在 val、NQ-test、multihop 三个端点分别提升 25.00、17.19 和 15.23 个百分点，置信区间都不跨 0。成本分支 C 相对 B 的 EM 基本持平，但总搜索减少 108 次、下降 8.18%，且没有再次出现不搜索坍缩。最终我把 R 定位为能力端点，C 定位为效率端点。

### 6.3 5-8 分钟深挖版本的讲述顺序

如果面试官说“你展开讲讲”，按以下顺序讲，不要一开始就堆所有数字：

1. 用 20 秒讲系统：Qwen3.5-2B + native tool calling + BM25 top-3 + GRPO。
2. 讲第一次失败：C-old 98.44% 不搜索，说明成本下降不等于策略更好。
3. 在白板写两条奖励公式，解释全错 group 中的 advantage 排序。
4. 讲 correct-only 为什么修复结构问题，以及为什么第一次仍没收益。
5. 讲多跳机会门禁如何阻止继续烧 GPU，并引出数据重构。
6. 讲 `visible_observation`、37.5%/62.5% 配比和 bridge 264 的理由。
7. 讲 XML/parser 失败如何推动 native tool calling 和 token/action boundary 修复。
8. 讲 G0/E0/G1/smoke/G3 各自隔离什么失败路径。
9. 讲 R60 主结果和 step 52 后的协议漂移。
10. 讲最终 B/C 对照、C 的 8.18% 搜索下降和 R/C Pareto 定位。
11. 最后主动说限制：单 seed、B/C clipping 高、不是论文完整榜单复现。

## 7. 高频追问与建议回答

### Q1：为什么旧成本奖励会让模型完全不搜索？

因为 GRPO 在同题 group 内做相对标准化。全组都答错时，EM 都是 0，搜索成本成为唯一排序信号；不搜索的错误轨迹比搜索的错误轨迹 reward 高，从而得到正 advantage。模型学到的是避开工具，而不是更有效地使用工具。

### Q2：把 lambda 调小不就行了吗？

不能根治。只要全错 group 中搜索次数不同，排序方向仍然是“少搜的错误轨迹更优”；mean/std 标准化还会抵消统一缩放。正确修复是改变奖励结构，让所有错误轨迹同为 0，只在正确方案之间比较成本。

### Q3：correct-only 会不会忽略错误轨迹的真实调用成本？

训练 surrogate 确实不对错误轨迹施加成本梯度，这是为了避免反搜索信用分配。但评价阶段仍使用 `EM - 0.025×searches` 对所有真实调用计成本，所以不会在报告中隐藏错误轨迹的浪费。

### Q4：为什么奖励修好后第一次 C-gated 还是没超过 B？

当时 67.5% 的 GRPO group 全错，没有相对学习信号；正确且搜索次数不同的轨迹太少。同时 B 在 NQ test-128 中已有 125 题只搜索一次，可压缩空间接近下限。公式修复解决了错误方向，但数据和探索密度仍不足。

### Q5：为什么要改数据配比？

成本优化需要“正确且多搜”的轨迹。NQ 主要是一搜后作答，旧 B 在多跳门禁中也几乎只搜一次。增加 Hotpot bridge 是为了提供“第一轮找到中间实体，第二轮围绕中间实体继续检索”的可学习链路；同时保留 37.5% NQ，避免训练初期全部是高难多跳导致大量全错 group。

### Q6：为什么不能直接按 HotpotQA 的多跳标签选数据？

benchmark 标注为多跳，不代表当前 BM25 top-3 和 observation 截断后，模型真的能看到两跳证据。项目用 `visible_observation` 检查经过 tokenizer 截断后送入模型的实际文本，确保训练题在当前检索器和上下文预算下具有可执行证据链。

### Q7：各种门禁分别解决了什么？

G0/E0 隔离 prompt、token、parser、真实检索和 mask 的工程错误；G1/G2 类型检查模型 action、query、clipping 和短答案边界；两步 smoke 验证真实 backward、optimizer 和分布式训练；G3 才检查训练后是否有足够的正确多搜、learnable group 和协议稳定性。这样可以明确失败属于工程、协议、训练数值还是科学假设。

### Q8：G3 是 NO-GO，为什么后来还有 B/C？

G3 的能力门没有正式放行，主要因为 clipping 和 invalid 过高；但它同时观察到 13/64 个 clean cost-contrast group。后续 B/C 因此被重新登记为 post-hoc exploratory 实验，而不是原预注册主流程的确认性分支。面试时要主动说明这层身份区别。

### Q9：B 比 R 多训练 20 步，为什么综合结果反而更低？

强化学习的 held-out 指标不保证随步数单调上升。B 在 val 上从 R 的 58.59% 提高到 61.72%，但 multihop 从 37.11% 降到 30.08%，表现为验证分布特化，而不是全面提升。B/C 的 clipping 又升到约 71%，说明后续训练仍有协议稳定性问题。因此 B 的作用是同源能力控制组，不能机械地把最后 checkpoint 当成最佳模型。

### Q10：那 C 的实际价值是什么？

C 不是全面能力升级，而是搜索深度调节器。相对 B，它在 512 题上 EM 从 36.33% 变为 37.11%，总搜索减少 108 次、下降 8.18%，4-search 饱和率从 38.28% 降到 30.27%，且没有退化成零搜索。它适合作为低成本候选；R 仍是 multihop 和综合能力端点。

### Q11：为什么用 BM25，不用论文里的 dense retriever？

这是预算缩小复现。BM25 是仓库支持的检索后端，不占 GPU 显存，索引和语料可以固定并做 digest，也便于 B/C 使用完全相同的环境。代价是不能声称复现论文的绝对榜单，只能研究同一冻结检索器下的相对训练和成本行为。

### Q12：如果再做一轮，最优先修什么？

不是直接增加训练步数，而是先降低长输出 clipping、修复 `invalid_thinking_prefix` 和 terminal answer compliance。之后用统一 checkout、多训练 seed、预注册 primary endpoint 和非劣界，再从同一新 parent 平行训练 B/C。SFT 格式热身可以作为独立新假设，但不能与本轮 direct-RL 结果混成一个实验。

### Q13：你个人做了什么？

可按实际职责删减后回答：

> 我的工作覆盖了四层：系统层完成 Qwen native tool calling、BM25 回填、action/answer 边界和 token mask；算法层定位 GRPO 全错 group 的反搜索信用分配并设计 correct-only 奖励；实验层设计 NQ/Hotpot 数据配比、G0-G3 门禁和同 parent B/C 对照；工程层实现 AutoDL CPU/GPU handoff、两步 smoke、checkpoint 血缘、逐轨迹证据、超时 recovery 和安全关机。最终不只是拿到分数，也能解释每次失败属于哪一层。

## 8. 数字速查表

### 8.1 早期 NQ-only

| 模型 | EM | 总搜索 | no-search |
|---|---:|---:|---:|
| A | 3.91% | 164 | 22.66% |
| R | 16.41% | 167 | 0% |
| B | 17.97% | 131 | 0% |
| C-old | 7.03% | 4 | 98.44% |
| C-gated follow-up | 14.06% | 137 | 0% |

### 8.2 最终 native 混合数据

| 模型 | 512 题 EM | 总搜索 | 平均搜索 |
|---|---:|---:|---:|
| A | 21.29% | 1690 | 3.301 |
| R | 39.45% | 1321 | 2.580 |
| B | 36.33% | 1320 | 2.578 |
| C | 37.11% | 1212 | 2.367 |

### 8.3 三个最值得记住的结果

1. A→R：512 题多答对 93 题、少搜索 369 次，三个端点 EM 增益的 95% CI 都不跨 0。
2. C-old：98.44% 不搜索、EM 7.03%，证明朴素成本惩罚会诱发策略坍缩。
3. B→最终 C：EM 36.33%→37.11%，总搜索 1320→1212，下降 8.18%，没有再次发生不搜索坍缩。

## 9. 面试中最容易讲错的地方

- 不要把 no-search collapse 说成程序崩溃；它是工程成功、科学失败。
- 不要把 A 说成官方 Base；它是 post-trained Qwen3.5-2B parent。
- 不要把 C 说成接着 B 训练；B/C 从同一个 R 权重平行启动。
- 不要把早期 NQ-only C-old、NQ-only C-gated 和最终 native C20 混为一谈。
- 不要说 G3 通过；G3 有真实能力信号，但正式结论是 NO-GO。
- 不要说最终 C 显著提升准确率；它可靠的成果是搜索效率改善。
- 不要把 R60 训练期最后十步的 50.5% clipping 当成冻结模型最终指标；冻结 R60 的 G3 clipping 是 46.88%。
- 不要用训练期 smoke→R60 的 37.50%→58.59% 替代最终封存 A→R 比较；最终统一 val 是 33.59%→58.59%。
- 不要说完整复现了 Search-R1 论文榜单；应说预算缩小的 Search-R1 风格复现与改进实验。
- 不要回避单 seed、greedy 单 rollout和 B/C 高 clipping；主动说明限制反而更能体现实验判断力。

## 10. 证据入口

- [早期 NQ-only A/R/B/C-old 结果](../docs/results/search-r1-small-20260720/README.md)
- [成本感知坍缩根因与 correct-only 奖励](../docs/成本感知坍缩分析与改进建议.md)
- [NQ-only C-gated 二次实验](../docs/history/成本感知二次实验结果分析.md)
- [HotpotQA/2Wiki 搜索机会门禁](../docs/results/search-opportunity-gate-20260722/analysis_zh.md)
- [混合数据与复现方案](../docs/autodl_search_r1_reproduction_plan.md)
- [旧 XML grouped probe 分析](../docs/results/grouped-probe-20260723/analysis_zh.md)
- [Qwen3.5 native 工具协议适配](../docs/qwen35_native_tool_adaptation_implementation_report.md)
- [native-v4 两步 smoke](../docs/qwen35_native_v4_two_step_smoke_analysis.md)
- [R60 训练与全量轨迹分析](../docs/qwen35_native_r60_training_and_trajectory_analysis.md)
- [R60 G3 评测与门禁分析](../docs/qwen35_native_r60_g3_evaluation_and_trajectory_analysis.md)
- [B/C recovery 完整分析](../docs/qwen35_native_bc_recovery_complete_analysis_report.md)
- [最终 A/R/B/C 统一评测](../docs/qwen35_native_arbc_final_results_analysis.md)
- [完整实验交接文档](../docs/qwen35_native_complete_experiment_handoff.md)

## 11. 最终收束句

> 这个项目最有价值的不是我把一个指标调高了，而是我把 Agent 强化学习中“奖励漏洞、数据不可学、工具协议错配、训练数值问题和科学假设失败”拆成了不同层次，用门禁和逐轨迹证据逐个定位。最终既得到 A→R 的明确能力提升，也得到 C 相对 B 的搜索效率改善，并且能够解释为什么早期会不搜索、为什么数据要改、为什么继续训练不保证单调变好，以及下一轮应该优先修什么。
