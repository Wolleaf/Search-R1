# Search-R1 面试知识体系与学习路线

> 用途：为 Agent 开发、LLM 应用开发及相关后端岗位准备 Search-R1 项目的技术追问。
>
> 核对日期：2026-08-03。项目事实以仓库代码、封存报告和《Search-R1 实验全流程复盘与面试讲稿》为准；本文重点回答“要学什么、学到多深、如何映射到项目”。

## 1. 先给结论：GRPO、PPO、DPO 分别学到什么程度

需要了解 PPO 和 DPO，但三者的学习深度不一样：

| 方法 | 建议深度 | 面试标准 |
|---|---:|---|
| **GRPO** | 100% | 能写核心公式、逐项解释、结合项目分析 no-search collapse、零方差 group、KL、clip 和 loss mask |
| **REINFORCE / Policy Gradient** | 80% | 能从策略梯度解释 GRPO 为什么能更新模型，以及 baseline 为什么能降方差 |
| **PPO** | 70% | 能解释 actor、critic、GAE、importance ratio、clipped objective，并与 GRPO 做完整比较 |
| **SFT** | 60% | 能解释监督轨迹、teacher forcing、格式冷启动，以及为什么 SFT 和 RL 解决的问题不同 |
| **DPO** | 50% | 能解释 chosen/rejected 偏好对、reference model 和 loss 的直觉，并回答为什么本项目没有直接用 DPO |
| **RLOO / Actor-Critic / TRPO** | 20%—40% | 知道它们在方法谱系中的位置，能做一句到两句比较，不要求完整推导 |
| **DAPO / Dr.GRPO 等 GRPO 改进** | 20% | 知道它们主要在解决零梯度 group、clip、长度偏置和训练稳定性问题 |

对当前投递方向而言，最重要的不是把自己准备成纯 RL 研究员，而是证明自己能把 **Agent 交互、强化学习、检索、评测和训练工程** 串成一个可解释、可运行、可诊断的系统。

## 2. 面试知识依赖图

```text
概率、期望、log probability
        ↓
策略梯度 / REINFORCE ──→ baseline 与方差降低
        ↓                         ↓
       PPO ── critic、GAE、clip、KL
        ↓
      GRPO ── 同题多采样、组内相对 advantage、无 critic
        ↓
项目奖励设计 ── EM、搜索成本、全错 group、策略坍缩

SFT ── 示范轨迹、格式冷启动 ─────────────┐
DPO ── 离线偏好对、reference policy ─────┤→ 与在线 Agent RL 做方法选择
RLOO / DAPO ── 相邻与改进方法 ───────────┘

Agent 状态机 ── tool calling ── observation mask ── terminal boundary
       ↓
BM25 / RAG ── 单跳与多跳数据 ── 检索可学性
       ↓
EM / 搜索次数 / 配对评测 / bootstrap CI
       ↓
FSDP / vLLM / Ray / checkpoint / 可复现工程
```

## 3. 掌握层级

为了避免“看过很多名词，但一追问就断”，本文把知识分成三级：

- **P0：必须掌握。** 能脱稿解释、画图或写公式，并立即映射到本项目。
- **P1：必须会比较。** 能说清输入数据、优化目标、优缺点和为什么选或不选。
- **P2：知道位置。** 听到名词不陌生，能用一两句话说明它解决什么问题。

建议把有限时间按以下比例分配：

- 35%：GRPO、奖励函数与策略坍缩；
- 25%：Agent tool calling、检索和 token mask；
- 20%：实验设计、指标与负结果诊断；
- 10%：PPO、DPO、SFT 等方法比较；
- 10%：FSDP、vLLM、Ray、checkpoint 和复现工程。

---

## 4. P0：必须掌握的强化学习基础

### 4.1 把语言模型表述成强化学习问题

至少能说清以下映射：

| RL 概念 | 在 Search-R1 中的含义 |
|---|---|
| Agent / policy | Qwen3.5-2B 当前策略模型 `πθ` |
| State | 问题、已生成 reasoning、历史工具调用及检索 observation |
| Action | 从 token 视角是下一个 token；从 Agent 视角是继续推理、调用 search 或提交 answer |
| Environment | 工具协议解析器、Wiki-2018 BM25 服务和终止规则 |
| Observation | 搜索工具返回并真正进入模型上下文的文本 |
| Trajectory | 从问题开始，到若干 reasoning/search/observation，再到最终答案的完整序列 |
| Reward | 最终答案 EM，以及可选的搜索成本项 |
| Return | 本项目主要是 outcome reward，奖励集中在轨迹末端 |

一句话版本：**模型不是只学习“答案是什么”，而是在环境反馈下学习整条生成轨迹中哪些 token 和工具决策更可能获得最终奖励。**

### 4.2 四个必须会的概念

1. **On-policy 与 off-policy**
   - on-policy：用当前或接近当前的策略采样，再更新这个策略；PPO、GRPO 的标准训练属于这一类。
   - off-policy：可以反复利用其他策略或历史策略产生的数据，但需要处理分布偏移。
   - 本项目的关键是模型必须真实执行搜索并生成 rollout，因此在线采样成本是训练成本的重要部分。

2. **Credit assignment**
   - 最终只有一个 EM，模型却生成了很多 token、可能还调用多次工具；怎样把结果归因给前面的动作，就是信用分配。
   - 本项目的 no-search collapse 本质上就是奖励和组内归一化共同造成了错误的信用分配。

3. **Exploration 与 exploitation**
   - 训练 rollout 需要采样差异，才能在同题 group 内形成好坏对比。
   - 最终评测使用 greedy 单 rollout，是为了固定口径，但它不能展示策略分布的全部方差。

4. **Reward hacking / specification gaming**
   - 模型优化的是给定奖励，不是设计者脑中的真实目标。
   - “不搜索就不扣成本”是一个可被策略利用的捷径，即使最终准确率下降，训练信号仍可能鼓励它。

### 4.3 策略梯度与 REINFORCE

必须理解这条核心关系：

```text
∇θ J(θ) = E[ A_t · ∇θ log πθ(a_t | s_t) ]
```

- `A_t > 0`：提高该动作或 token 的概率；
- `A_t < 0`：降低该动作或 token 的概率；
- `A_t = 0`：这一样本基本不提供策略更新信号；
- `log πθ(a_t|s_t)`：模型对自己实际采样 token 的对数概率。

REINFORCE 直接用完整轨迹回报做 Monte Carlo 策略梯度。减去一个与当前动作无关的 baseline，不改变期望梯度方向，却能降低方差：

```text
A = R - b
```

这是理解 PPO 的 critic baseline 和 GRPO 的 group baseline 的共同起点。面试官如果问 GRPO，不应直接从 GRPO 术语开始背；先讲“策略梯度需要 advantage，GRPO 用同题组内相对奖励替代学习出来的 value baseline”。

---

## 5. P0：GRPO 必须掌握到可以白板推导

### 5.1 GRPO 的核心流程

对于同一个问题 `q`：

1. 当前策略采样 `G` 条完整回答或 Agent 轨迹；
2. 环境分别执行搜索、解析最终答案并计算 reward；
3. 在同一问题的 `G` 条轨迹内计算均值和标准差；
4. 得到每条轨迹的 group-relative advantage；
5. 使用 PPO 风格的 probability ratio 和 clip 更新策略；
6. 可加入相对 reference policy 的 KL 约束，防止策略漂移过快。

项目中 `G=5`。核心 advantage 可以写成：

```text
             r_i - mean(r_1, ..., r_G)
A_i =  ------------------------------------
       std(r_1, ..., r_G) + ε
```

项目的 `compute_grpo_outcome_advantage` 先把轨迹级 `A_i` 广播到 response 位置；native 路径再通过 `info_mask/loss_mask` 排除 observation 等环境 token，最终只让 policy 生成 token 贡献策略损失。

### 5.2 GRPO 与 PPO 共享的 clipped objective

先定义新旧策略概率比：

```text
ratio_t(θ) = πθ(a_t|s_t) / πold(a_t|s_t)
           = exp(logπθ - logπold)
```

PPO/本仓库 GRPO 路径使用的核心直觉是：

```text
Lclip = E[min(ratio_t · A_t,
              clip(ratio_t, 1-ε, 1+ε) · A_t)]
```

clip 的作用不是保证模型一定变好，而是限制单次更新中概率比变化过大。`clip fraction` 很高说明大量 token 的更新已经撞到裁剪边界，通常意味着更新过激、策略与 rollout policy 偏离较大，或数据/长度/协议造成了不稳定；它是诊断信号，不是单独的成败指标。

### 5.3 为什么 GRPO 不需要单独的 critic

PPO 通常训练 value/critic 来估计 `V(s)`，再计算 advantage；GRPO 用同一问题多条回答的相对奖励作为 baseline，因此不需要额外的 learned critic。

收益：

- 少维护一个与策略模型同规模或较大规模的 value model，节省显存和系统复杂度；
- 对答案可验证、能一次采多条回答的任务很自然；
- 同题比较能削弱“不同问题天生难度不同”造成的 reward 尺度差异。

代价：

- 每个问题必须采多条 rollout，生成和工具执行成本上升；
- 信号强依赖 group 内存在 reward 差异；
- 对纯 0/1 outcome reward，全对或全错 group 会零方差；更一般地，任何 reward 完全相同的 group 都没有相对 advantage；
- group size 越小，均值和方差估计越噪；越大则 rollout 成本越高。

### 5.4 本项目最重要的白板题：为什么会 no-search collapse

早期成本奖励：

```text
r_old = EM - λ · n_search / N_max
```

如果同一问题的五条轨迹全部答错，则 `EM=0`：

```text
r_i = -λ · n_i / N_max
```

此时 reward 唯一能区分轨迹的因素就是搜索次数。搜索越少，reward 越高；不搜索的错误轨迹会得到正的组内相对 advantage，于是策略被训练成“即使不会，也不要搜索”。

为什么仅把 `λ` 调小不能根治：

```text
A_i = (λx_i - mean(λx)) / std(λx)
    ≈ (x_i - mean(x)) / std(x)       当 λ > 0 且忽略 ε 时
```

组内标准化会消掉统一缩放，reward 排序也没有变化。因此小 `λ` 可能改变数值细节，但不消除全错 group 中的反搜索方向。

修复后的 correct-only 奖励：

```text
r_gated = EM · (1 - λ · n_search / N_max)
```

- 全错 group：所有 reward 都是 0，不再产生“少搜索更好”的梯度；
- 有正确轨迹的 group：答对仍是第一目标，在正确轨迹之间再偏好更低搜索成本；
- 局限：全错 group 同时也没有能力学习信号，所以还必须解决数据难度、协议正确性、探索和模型能力问题。

这是整个项目最应该讲透的算法闭环：**先从实际坍缩观察出发，再用 GRPO 的组内 advantage 解释根因，修改奖励结构，最后承认奖励修复只消除了错误激励，并不会自动创造正确轨迹。**

### 5.5 GRPO 高频追问

必须能回答：

- 为什么 group 必须来自同一个 prompt？
- group size 变大或变小分别有什么影响？
- group 全错、全对、标准差为零时会怎样？
- outcome reward 为什么信用分配粗糙？是否需要 process reward？
- KL 是与哪个策略比较，为什么需要 reference policy？
- clip 和 KL 都限制策略变化，它们的作用有什么区别？
- retrieved observation 为什么不能当成模型生成 token 一起训练？
- reward normalization 会不会改变绝对 reward 大小的意义？
- 为什么继续多训练 20 步不保证指标单调上升？

---

## 6. P1：PPO 要会完整比较，但不必达到复现论文的深度

### 6.1 PPO 的五个核心词

1. **Actor**：要优化的策略模型。
2. **Critic / value model**：估计当前状态的期望回报 `V(s)`。
3. **GAE**：用 value 估计构造 advantage，在偏差和方差之间折中。
4. **Importance ratio**：比较新旧策略对同一动作的概率。
5. **Clipping**：限制一次策略更新过大。

GAE 至少知道这两个式子在做什么：

```text
δ_t = r_t + γV(s_{t+1}) - V(s_t)
A_t^GAE = δ_t + γλδ_{t+1} + (γλ)^2δ_{t+2} + ...
```

不要求手算长序列，但要知道：`γ` 控制未来奖励折扣，GAE 的 `λ` 控制 bias-variance tradeoff。注意不要把 **GAE λ** 与项目搜索成本的 **cost λ** 混为一谈。

### 6.2 PPO 与 GRPO 对比标准答案

| 维度 | PPO | GRPO |
|---|---|---|
| Advantage 来源 | critic/value + GAE | 同一 prompt 多条 rollout 的组内相对 reward |
| 是否需要 critic | 通常需要 | 不需要 learned critic |
| 主要显存成本 | actor、critic、reference、optimizer 等 | 省 critic，但需同题多采样 |
| 适合信号 | 一般序列决策、可做 token/state 级 value | 可验证 outcome reward、同题能采多条答案 |
| 主要风险 | critic 不准、训练系统复杂 | 全同 reward group 无信号、组统计噪声、rollout 贵 |
| 共同点 | 都是策略梯度；常用旧策略 ratio、clip 和 KL 约束 | 同左 |

推荐口述：

> GRPO 可以看作面向 LLM 可验证任务的一种 PPO 变体。PPO 用 critic 估计 advantage，GRPO 不训练 critic，而是对同一道题采多条回答，用组内均值和标准差构造相对 advantage。它省掉了 critic 的显存和训练复杂度，但把代价转移到了多 rollout 上，而且非常依赖 group 内奖励有差异。本项目正因为全错 group 中只剩搜索成本差异，才出现了 no-search collapse。

---

## 7. P1：DPO、SFT 以及为什么本项目仍需要在线 RL

### 7.1 DPO 要掌握什么

标准 DPO 使用同一 prompt 下的偏好对：

```text
(x, y_chosen, y_rejected)
```

它比较 policy 与 reference model 对 chosen/rejected 的相对 log probability，通过一个分类式目标直接提高 chosen 相对 rejected 的概率。需要知道以下边界：

- 标准 DPO 不显式训练 reward model；
- 标准 DPO 微调时不需要像 PPO/GRPO 那样持续在线采样 rollout；
- 它需要已有的 chosen/rejected 偏好数据；
- reference policy 用于表达“相对原策略改变了多少”，并形成隐式 KL 约束。

不要求背完整 DPO 推导，但最好能认出以下结构：

```text
log σ(β[(logπθ(yw|x)-logπref(yw|x))
      -(logπθ(yl|x)-logπref(yl|x))])
```

其中 `yw` 是 preferred/chosen，`yl` 是 rejected。

### 7.2 为什么不直接用 DPO 做本项目

标准回答不能只说“DPO 不好”，而要说任务匹配：

> DPO 适合已经有静态偏好对的离线对齐。本项目的核心变量是策略在多轮环境中何时搜索、搜什么，以及搜索返回的新 observation 如何改变后续动作；我需要让当前策略真实与检索环境交互，并用最终 EM 和执行成本评价完整 rollout。GRPO 更直接地优化这个在线 outcome。DPO 也不是完全不能用，但需要先记录成对 Agent 轨迹并定义可靠偏好，且静态数据难以覆盖策略更新后出现的新 query 和失败路径，因此不能直接替代在线环境训练。

可以进一步提出组合方案：

- 先从成功轨迹构造 SFT 数据，学习合法工具协议；
- 从同题轨迹中构造 chosen/rejected，做 DPO 式离线预热；
- 最后用 GRPO 在真实检索环境中继续优化任务成功率和搜索成本。

这属于合理的后续方案，不要说成项目已经完成。

### 7.3 SFT 要掌握什么

SFT 用高质量输入—输出或完整 Agent 示范轨迹做 token-level cross-entropy：

- 优点：训练稳定，能快速教会格式、工具 schema 和基本行为；
- 缺点：只能模仿数据分布，不直接优化最终 EM/成本；错误示范会被照学；推理时会遇到训练数据没有覆盖的状态；
- 与 RL 的关系：SFT 提供行为先验和冷启动，RL 用环境结果继续优化。

项目事实边界：**最终 A→R→B/C 路线是 direct outcome-RL，没有实际执行 SFT40；“先 SFT 再 RL”只可作为下一轮改进方案。** 面试中绝不能把提议说成已完成实验。

### 7.4 RLHF、RLAIF、rule-based reward 的区别

- **经典 RLHF**：SFT → 人类偏好数据 → reward model → PPO 等 RL 优化。
- **DPO**：从偏好对直接优化 policy，省掉显式 reward model 和标准在线 PPO 环节。
- **RLAIF**：偏好或反馈主要由 AI 产生。
- **本项目**：答案可以用 strict EM 程序化验证，搜索次数也可直接记录，因此使用 rule-based outcome reward，不需要训练人类偏好 reward model。

---

## 8. P2：相邻方法知道它们解决什么即可

| 方法 | 一句话定位 | 与本项目的关系 |
|---|---|---|
| REINFORCE | 最基础的 Monte Carlo 策略梯度 | 是理解所有后续方法的根 |
| Actor-Critic | actor 决策，critic 估值 | PPO 的基础框架 |
| TRPO | 用 trust region 约束策略更新 | PPO 的历史前身；PPO 用更易实现的 surrogate/clip 近似目标 |
| RLOO | 用同题其他样本的平均 reward 做 leave-one-out baseline | 与无 critic、多采样的 GRPO 很接近，值得知道比较点 |
| DAPO | 在大规模 LLM RL 中改造 clip、动态采样、token loss 和过长序列处理 | 能解释为什么零方差 group、长度和 clip 是现代 GRPO 训练重点 |
| Process Reward Model | 对中间推理步骤给反馈 | 可缓解只有最终 EM 时信用分配过粗，但标注和可靠性成本更高 |
| DeepSeek-R1-Zero 路线 | 无 SFT 冷启动直接做大规模 RL | 与本项目 direct RL 的思路相近，但规模、任务和结论不可类比 |
| DeepSeek-R1 多阶段路线 | cold-start/SFT、RL、再对齐等多阶段训练 | 用于解释“direct RL 能学，但稳定格式和可读性常需要额外阶段” |

如果时间很少，RLOO、DAPO 和 Dr.GRPO 只需要能回答“它主要修什么”，不要为了追新方法挤占 GRPO、Agent 系统和项目证据的复习时间。

---

## 9. P0：Agent 与 tool calling 知识

### 9.1 ReAct、RAG 与 Search-R1 的区别

- **普通 RAG**：通常先用固定 query 检索一次，再把文档交给模型生成。
- **ReAct**：让模型交替产生 reasoning 和 action，在环境 observation 后继续推理。
- **Search-R1 风格 Agent**：不仅在推理时调用搜索，还通过 RL 学习什么时候搜索、query 怎么写、什么时候停止。

推荐口述：

> 这个系统不是简单把 top-k 文档拼到 prompt。模型每轮都可以在 search 和 answer 之间决策，搜索结果会作为新的环境 observation 回填，后续 query 取决于前面看到的证据，所以它是一个多轮闭环策略。

### 9.2 必须掌握的工程边界

1. **工具 schema**：名称、参数类型、必填字段，以及模型看到的 tool definition。
2. **解析边界**：如何确定一段生成是合法 search、最终 answer 还是 invalid action。
3. **执行边界**：只有解析成功且预算允许的 search 才真正访问检索服务。
4. **状态回填**：tool response 进入下一轮上下文，但不是模型自己生成的动作。
5. **终止边界**：最多四次真实搜索；预算耗尽后进入 answer-only terminal generation。
6. **审计字段**：action count、executed search count、invalid action、clipping、最终答案和可见 observation 都要能落盘。

### 9.3 为什么 observation 必须 mask 掉 policy loss

检索结果由环境产生，不是模型采样的 token。如果把 observation 也纳入 policy gradient：

- 会把环境文本错误归因给策略；
- log probability 与真实采样过程不一致；
- 模型可能被训练去“生成检索结果”，而不是学习调用工具和利用结果；
- rollout 与训练 replay 的概率口径会失真。

正确做法是：observation 仍保留在 attention context 中，供后续 token 条件化；但在 `info_mask/loss_mask` 中置零，不参与 policy loss。terminal reminder 同理属于环境输入，而 reminder 后模型生成的 assistant token 仍参与训练。

### 9.4 常见 Agent 失败模式

- 工具 JSON/schema 不合法；
- 生成协议与 parser 期待不一致；
- 把“请求搜索”误计为“真实执行搜索”；
- observation 截断后，分析脚本仍按原始全文判断证据可见；
- 达到预算后仍反复 search；
- response 被长度裁剪，最终答案标签缺失；
- 训练 chat template 与推理 template 不同；
- rollout engine 和训练 actor 权重或上下文没有同步。

这些问题都可能表现为“模型不会搜索”，但根因分别属于协议、环境、数据、长度或训练系统，不能只靠调学习率解决。

---

## 10. P0/P1：检索、RAG 与数据构建

### 10.1 BM25 至少会解释公式直觉

BM25 对 query 中每个词累积分数，核心形式是：

```text
score(D,Q) = Σ IDF(q) ·
             f(q,D)(k1+1)
             ---------------------------------------
             f(q,D) + k1(1-b+b·|D|/avgdl)
```

- `IDF`：稀有词更有区分度；
- `f(q,D)`：词频越高通常越相关，但收益会饱和；
- `b`：文档长度归一化强度；
- `k1`：词频饱和速度；
- `top-k`：返回得分最高的若干文档，本项目为 top-3。

### 10.2 为什么本项目用 BM25

可讲的取舍：

- 成本低、CPU 友好、结果可解释、索引和语料容易固定；
- 对实体名和百科问答中的精确词匹配通常有效；
- 适合预算受控实验，把主要变量留给 Agent 策略和 RL；
- 局限是语义同义改写、隐式关系和 query 表达差异可能召回不足；dense retriever 可能改善语义召回，但会增加 encoder、FAISS、显存、版本和复现变量。

不要说“BM25 一定比 dense 好”；应该说它是本项目预算、可解释性和变量控制下的工程选择。

### 10.3 单跳、comparison 与 bridge

- **NQ single**：多为一个核心事实，常能一次检索解决。
- **HotpotQA comparison**：通常需要找到两个实体的信息再比较。
- **HotpotQA bridge**：先找到中间实体，再用中间实体发起下一次搜索，更能提供 query reformulation 和多轮搜索机会。

项目最终训练配比不是随意混合，而是先检查检索证据在 **模型实际可见的截断 observation** 中是否存在，再按 single/comparison/bridge 固定配额。面试时要理解：数据“标成多跳”不等于当前 retriever、top-k、语料版本和截断配置下真的可学。

### 10.4 数据集与数据工程要掌握的词

- train/validation/test 隔离与数据泄漏；
- 固定 source revision、hash 和样本 ID；
- 类别配额与分层采样；
- answer normalization 与坏答案排除；
- retrieval feasibility / evidence visibility；
- prompt/template/tokenizer 长度门禁；
- 数据难度分布与 curriculum；
- 同一道题多 rollout 的 group 构造。

---

## 11. P0：指标、统计与实验设计

### 11.1 指标必须能说出口径

| 指标 | 说明 | 不能单独证明什么 |
|---|---|---|
| strict EM | 归一化后的最终短答案与 gold 是否精确匹配 | 不反映语义接近、解释质量或证据忠实度 |
| executed search count | 实际执行的搜索次数 | 少不一定好，可能是 no-search collapse |
| no-search rate | 完全没执行搜索的题目比例 | 高低都需结合正确率和题型解释 |
| correct-only searches | 只在答对样本中统计平均搜索次数 | 会受到“哪些题答对了”这个样本集合变化影响 |
| post-hoc utility | 用固定成本系数组合 EM 与搜索成本 | 权重是人为选择，不能替代分别报告能力和效率 |
| invalid/clipping rate | 协议和长度稳定性指标 | 不等于任务正确率，但能解释训练退化 |

必须区分：训练 C 使用的 `EM × (1-cost)` 与报告中的 `EM - cost` post-hoc utility 不是同一个量，不能混写。

### 11.2 为什么要配对评测

A/R/B/C 应在同一批题、同一检索服务、同一解码和同一评测脚本上比较。配对评测关注同一道题从错到对、从对到错，以及搜索次数如何变化，比只比较两个总体平均数更能定位差异。

项目使用 paired percentile bootstrap：对“题目对”进行有放回重采样，反复计算 candidate-baseline 差值，得到置信区间。要能解释：

- CI 不跨 0：在当前样本和抽样假设下，差值方向较稳定；
- CI 跨 0：不能声称有可靠提升，不等于两个模型绝对相同；
- 置信区间不能修复数据泄漏、协议不一致或单 seed 的外部有效性问题。

可额外知道 McNemar 检验用于成对二分类结果，但项目当前主报告使用 bootstrap；不要把“知道的方法”说成“项目已经使用的方法”。

### 11.3 为什么继续训练不保证更好

“R 再训 20 步得到 B，所以 B 理应全面优于 R”是错误前提。可能原因包括：

- stochastic optimization 和单 seed 波动；
- 训练分布与三个评测端点分布不同；
- 后期过拟合或能力 specialization；
- policy 更新过大、clip fraction 上升；
- response 变长后发生截断，合法 answer 反而减少；
- optimizer 在 B/C 新 run 中重新初始化，并非完整 trainer-state 断点续训；
- EM 是离散指标，小参数变化可能让若干边界样本翻转。

正确说法是：**多训练代表进行了更多优化步，不构成 held-out 指标单调性的保证；因此需要固定评测、轨迹诊断和早停/选点。**

### 11.4 负结果如何讲得专业

一个完整负结果要包含：

```text
观察 → 排除工程故障 → 提出机制假设 → 设计可证伪检查
    → 得到证据 → 修改方案 → 复测 → 说明仍然存在的边界
```

项目中的典型例子：

- no-search collapse：奖励结构问题；
- G3 NO-GO：能力存在，但 clipping/invalid 稳定性不过门；
- 第一次 B/C 调度失败：属于训练编排问题，不应包装成数值崩溃；
- correct-only 修复后未立即超过 B：说明消除错误激励不等于数据中已有足够正样本。

---

## 12. P1：LLM 训练与分布式系统

### 12.1 LLM 训练基础

至少会解释：

- causal language modeling 和 next-token probability；
- token、logit、softmax、log probability；
- temperature、top-p、top-k 与 greedy decoding；
- chat template 为什么会改变 token 序列和工具格式；
- full-parameter fine-tuning 与 LoRA 的显存、容量和部署取舍；
- bf16 为什么比 fp32 节省显存，以及它和 fp16 的数值范围差异；
- gradient accumulation、micro-batch、global batch 的关系。

### 12.2 FSDP、vLLM、Ray、verl 各自做什么

| 组件 | 项目中的角色 | 面试需要掌握的深度 |
|---|---|---|
| verl | RL 训练框架，组织 actor、rollout、reference、reward 与训练循环 | 能画出数据如何在这些角色间流转 |
| Ray | 多进程/多 GPU worker 编排和远程调用 | 知道它解决调度与角色分布，不需要背 API |
| FSDP | 分片参数、梯度和 optimizer state，支持两卡全参数训练 | 能与 DDP 的每卡完整副本做比较 |
| vLLM | 高吞吐 rollout 生成 | 知道训练 actor 与推理引擎权重同步、KV cache 和显存竞争是关键问题 |

### 12.3 Checkpoint 必须讲准确

- 只有 Hugging Face 模型权重：可以作为新 run 的初始化模型；
- 完整训练 checkpoint：还应包含 optimizer、scheduler、global step、随机状态和 trainer state；
- 本项目 R60 只保存了模型权重，因此 B/C 从 exact R 权重启动，但新建 optimizer，不能说成严格 resume；
- B/C 从同一个 R 权重平行分叉，不是 R→B→C 串行训练。

### 12.4 可复现与云训练工程

需要知道为什么项目记录：

- Git commit、配置、模型与数据 hash；
- CPU 数据准备与 GPU 训练分阶段；
- smoke test 和分层 gate；
- 原始日志、退出码、轨迹、manifest 和 checkpoint lineage；
- 超时恢复、失败阶段定位和付费 GPU 自动关机。

Agent RL 的成本主要不只在一次 backward，还在同题多 rollout、真实工具执行、长上下文和反复评测。门禁的价值是先用便宜证据排除协议、数据和环境错误，再允许昂贵训练。

---

## 13. 方法总对比表

| 方法 | 训练数据 | 是否需在线环境 rollout | critic / baseline | 主要目标 | 对本项目的适配 |
|---|---|---:|---|---|---|
| SFT | 专家答案或 Agent 示范轨迹 | 否 | 无 | token-level imitation | 适合工具格式和冷启动，但不直接优化 EM/搜索成本 |
| REINFORCE | 当前策略 rollout + scalar reward | 是 | 可用简单 baseline | 最大化期望 reward | 理论基础，方差可能较高 |
| PPO | rollout + reward | 是 | learned critic + GAE | clipped policy optimization | 可做，但 critic 显存和系统复杂度更高 |
| GRPO | 同题多条 rollout + reward | 是 | group mean/std，无 learned critic | group-relative clipped policy optimization | 本项目实际主方法；适合可验证答案和多采样 |
| RLOO | 同题多条 rollout + reward | 是 | 其他样本的 leave-one-out mean | REINFORCE-style optimization | 相邻替代方案，适合无 critic RLHF |
| DPO | chosen/rejected 偏好对 | 标准版否 | reference policy 的相对 log-ratio | 偏好分类式优化 | 可做离线预热，但不能直接替代持续工具交互 |
| DAPO | 在线 group rollout + rule reward | 是 | GRPO 家族 | 改善 clip、采样、token 聚合和长度稳定性 | 后续稳定性改进的参考，不是本项目已完成项 |

---

## 14. 高频面试问题与答题骨架

### Q1：用一分钟介绍项目架构

必须覆盖：Qwen3.5-2B、native tool calling、Wiki-2018 BM25 top-3、多轮 search/answer、最多四次真实搜索、observation 回填、GRPO、strict EM、两卡全参数训练。

### Q2：为什么选 GRPO，不选 PPO？

先承认 PPO 可行，再讲 critic 显存/复杂度、同题可验证 reward、group relative baseline；最后主动讲 GRPO 的全同 reward group 和多 rollout 成本。

### Q3：为什么不是 DPO？

讲清静态偏好对与在线环境交互的区别；说明 DPO 可作为轨迹偏好预热，但本项目需要当前策略真实搜索并用执行结果计 reward。

### Q4：GRPO 的 advantage 怎么算？

写 `(r_i-mean)/std`，解释同题采样、正负 advantage、零方差 group、group size trade-off。

### Q5：为什么成本奖励导致完全不搜索？

写全错 group 的 `r=-λn/N`，说明少搜索错误轨迹得到正相对 advantage；再说明缩小 λ 会被标准化抵消。

### Q6：correct-only 奖励有什么缺点？

它消除全错 group 的反搜索信号，但全错 group 也变成零学习信号；还可能只优化成功轨迹内部的成本，无法直接教模型如何从错误变正确。

### Q7：为什么 retrieval observation 不参与 loss？

它是环境 token，不是 policy action；可以作为 attention context，但必须从 policy-gradient mask 排除。

### Q8：为什么用 BM25，不用 dense retrieval？

讲预算、CPU、可解释、固定索引和实体匹配；同时承认语义召回局限，并提出后续受控对比 dense/hybrid 的方案。

### Q9：为什么要提高 bridge 数据比例？

单跳题一次搜索即可完成，无法提供二次 query 学习信号；bridge 必须利用中间实体，更能产生可学习的多轮检索轨迹。但必须先验证证据在实际可见 observation 中存在。

### Q10：训练多 20 步为什么可能变差？

讲非单调优化、分布偏移、过拟合、clip/invalid/截断、单 seed，以及 B/C 新 optimizer 而非完整 resume。

### Q11：如何证明提升不是偶然？

讲同题配对、固定环境和解码、逐题 transition、paired bootstrap CI；主动说明单 seed、greedy 单 rollout 的限制。

### Q12：如果再做一轮，先改什么？

优先级建议：

1. 降低 clipping/invalid 和过长轨迹；
2. 增加能产生组内正确/错误差异的可学数据；
3. 比较 SFT/DPO 冷启动后再 GRPO；
4. 尝试 DAPO 式 dynamic sampling 或更稳的 advantage/token 聚合；
5. 做多 seed 和 dense/hybrid retriever 消融。

### Q13：这个项目最能证明你的什么能力？

不要只答“会 GRPO”。应答：能把 Agent 协议、检索环境、奖励机制、分布式训练和评测证据分层诊断；既能拿到能力提升，也能解释失败为何发生、何时应该停止烧卡。

---

## 15. 按项目代码阅读，而不是只看博客

建议阅读顺序：

1. **项目全貌与事实边界**
   - [`Search-R1实验全流程复盘与面试讲稿.md`](./Search-R1实验全流程复盘与面试讲稿.md)
   - [`Search-R1项目经历-量化稿.md`](./Search-R1项目经历-量化稿.md)

2. **GRPO、GAE、PPO clip 与 KL**
   - [`verl/trainer/ppo/core_algos.py`](../verl/trainer/ppo/core_algos.py)
   - 重点：`compute_gae_advantage_return`、`compute_grpo_outcome_advantage`、`compute_policy_loss`、`kl_penalty`。

3. **EM、搜索成本与 correct-only reward**
   - [`verl/trainer/main_ppo.py`](../verl/trainer/main_ppo.py)
   - 重点：`search_costs`、`posthoc_utilities`、`cost_reward_mode == 'correct_only'`。

4. **Agent 多轮生成和 observation mask**
   - [`search_r1/llm_agent/generation.py`](../search_r1/llm_agent/generation.py)
   - [`search_r1/llm_agent/tool_protocol.py`](../search_r1/llm_agent/tool_protocol.py)
   - 重点：native follow-up、`responses_with_info_mask`、terminal answer-only、真实搜索计数。

5. **轨迹审计与协议失败**
   - [`search_r1/trajectory_trace.py`](../search_r1/trajectory_trace.py)
   - 重点：turn、invalid action、executed search、clipping、observation policy token count。

6. **检索系统**
   - [`search_r1/search/retrieval.py`](../search_r1/search/retrieval.py)
   - [`search_r1/search/index_builder.py`](../search_r1/search/index_builder.py)
   - [`search_r1/search/bm25_server.py`](../search_r1/search/bm25_server.py)

7. **数据配比和检索可见性**
   - [`scripts/data_process/search_mix.py`](../scripts/data_process/search_mix.py)
   - 重点：single/comparison/bridge quotas、pinned source、retrieval evidence、manifest。

8. **配对评测和置信区间**
   - [`scripts/autodl/paired_eval.py`](../scripts/autodl/paired_eval.py)
   - 重点：paired rows、transition、`_paired_bootstrap`、EM/search/clipping 差值。

9. **分布式与 rollout 系统**
   - [`verl/workers/fsdp_workers.py`](../verl/workers/fsdp_workers.py)
   - [`verl/workers/sharding_manager/fsdp_vllm.py`](../verl/workers/sharding_manager/fsdp_vllm.py)

阅读每个文件时只回答四个问题：输入是什么、输出是什么、关键不变量是什么、失败会被什么指标看见。不要一开始逐行背框架代码。

---

## 16. 复习计划

### 16.1 今晚四小时最低可用版

| 时间 | 内容 | 完成标准 |
|---|---|---|
| 0:00—0:30 | 项目架构与 A/R/B/C 身份 | 能画系统图，绝不混淆 C、G3 和 checkpoint 身份 |
| 0:30—1:20 | Policy Gradient → PPO → GRPO | 能写策略梯度、PPO clip、GRPO advantage 三个式子 |
| 1:20—1:50 | no-search collapse | 能在白板上证明为什么调小 λ 不根治 |
| 1:50—2:20 | SFT、DPO、PPO、GRPO 对比 | 能回答为什么本项目选 GRPO，以及组合方案 |
| 2:20—3:00 | tool calling、mask、BM25、多跳数据 | 能解释 observation 为什么只进 context 不进 loss |
| 3:00—3:30 | 指标、bootstrap、负结果 | 能解释 CI、配对评测和“多训不保证更好” |
| 3:30—4:00 | 口述演练 | 连续完成 30 秒、2 分钟、5 分钟三个版本 |

### 16.2 三天面试版

**Day 1：算法主线**

- Policy Gradient、REINFORCE、baseline；
- PPO actor/critic/GAE/clip/KL；
- GRPO 公式、group size、全同 reward、no-search collapse；
- DPO、SFT、RLOO、DAPO 方法地图。

**Day 2：Agent 与实验主线**

- native tool calling 状态机；
- response/observation/loss mask；
- BM25、single/comparison/bridge 和 evidence visibility；
- EM、search、utility、paired bootstrap；
- 阅读关键代码和三份核心报告。

**Day 3：系统与表达**

- FSDP、vLLM、Ray、checkpoint；
- 逐题复盘 no-search、G3 NO-GO、B/C 恢复；
- 做两轮模拟面试：一轮 Agent 开发，一轮 RL 深挖；
- 把答不完整的问题回填到本文旁边，而不是继续漫无目的看新论文。

### 16.3 一周进阶版

- 手推 PPO、GRPO、DPO 目标各一次；
- 用一个五轨迹 toy example 手算 group advantage；
- 读完关键代码路径并画数据流图；
- 阅读原论文的 algorithm 和 ablation 部分；
- 准备 dense/hybrid retrieval、SFT→GRPO、DAPO dynamic sampling 三个后续实验设计；
- 进行至少三轮有追问的模拟面试并录音复盘。

---

## 17. 自测清单

如果以下问题中有三项答不出来，就还不能把项目放到简历后直接参加深挖面试：

- [ ] 我能在 30 秒内说清项目问题、方案和结果。
- [ ] 我能写出策略梯度、PPO clip 和 GRPO advantage。
- [ ] 我能解释 baseline 为什么降低方差。
- [ ] 我能比较 PPO 的 critic 与 GRPO 的 group baseline。
- [ ] 我能解释全错 group 为什么导致旧奖励反向鼓励不搜索。
- [ ] 我能证明简单缩小 cost λ 为什么不能根治。
- [ ] 我能说出 correct-only reward 的收益和局限。
- [ ] 我能解释 DPO 的数据形式，以及为什么不直接替代在线 Agent RL。
- [ ] 我不会把 SFT 提议说成已完成实验。
- [ ] 我能解释 observation 进入 attention 但不进入 policy loss。
- [ ] 我能区分请求搜索、合法搜索和真实执行搜索。
- [ ] 我能说出 BM25 的 IDF、词频饱和和长度归一化直觉。
- [ ] 我能解释 bridge 数据为什么更适合学习多轮 query。
- [ ] 我能说清 strict EM、搜索次数和 utility 各自的边界。
- [ ] 我能解释 paired bootstrap CI，而不把它说成绝对真理。
- [ ] 我能解释为什么 B 多训练 20 步仍可能弱于 R。
- [ ] 我能解释 clipping、invalid action 和 response truncation 的区别。
- [ ] 我能区分模型权重初始化与完整 optimizer-state resume。
- [ ] 我能解释 FSDP、vLLM、Ray 在系统中分别负责什么。
- [ ] 我能主动说出单 seed、greedy 单 rollout 和高 clipping 等限制。

---

## 18. 原始资料阅读顺序

优先读论文摘要、方法图、核心公式和 ablation，不必第一遍逐页精读：

1. [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516)
2. [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models（GRPO 来源）](https://arxiv.org/abs/2402.03300)
3. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
4. [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
5. [Training Language Models to Follow Instructions with Human Feedback（InstructGPT / RLHF）](https://arxiv.org/abs/2203.02155)
6. [Back to Basics: Revisiting REINFORCE Style Optimization for RLHF（RLOO）](https://arxiv.org/abs/2402.14740)
7. [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
8. [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
9. [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)
10. [Understanding R1-Zero-Like Training: A Critical Perspective（Dr. GRPO）](https://arxiv.org/abs/2503.20783)

阅读时始终带着三个问题：它需要什么数据、它如何得到 advantage/偏好信号、它能否直接处理真实的多轮工具环境。

## 19. 最小必背十点

时间只够最后冲刺时，至少记住：

1. GRPO = 同题多 rollout + 组内相对 advantage + PPO 风格 clip，省 critic 但依赖组内差异。
2. 策略梯度中正 advantage 提高动作概率，负 advantage 降低动作概率。
3. 全错 group 下旧奖励只剩搜索成本，因此错误但少搜索的轨迹被强化。
4. 组内标准化会抵消统一缩放，所以仅调小成本 λ 不能根治方向错误。
5. correct-only 消除反搜索梯度，但全错 group 仍然没有能力学习信号。
6. PPO 用 critic/GAE，GRPO 用 group baseline；DPO 用离线偏好对，SFT 用示范轨迹。
7. observation 是环境 token：进入上下文，不进入 policy loss。
8. BM25 是本项目预算和可复现性选择；bridge 数据提供更真实的二次 query 机会。
9. 能力看 EM，效率看真实搜索，稳定性看 clipping/invalid；三者不能互相替代。
10. 更多训练步不保证 held-out 指标单调；必须靠固定配对评测、轨迹证据和 checkpoint 血缘判断。

最后的面试定位不是“我背过 GRPO”，而是：

> 我理解 GRPO 为什么适合可验证的工具 Agent，也亲自处理过它在稀疏奖励、组内归一化、工具协议、检索可见性和长轨迹训练中的真实失败路径；我能从算法、系统和实验三层定位问题，并用配对证据验证修复是否真的有效。
