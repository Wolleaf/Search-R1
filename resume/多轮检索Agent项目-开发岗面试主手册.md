# 多轮检索 Agent 项目：开发岗面试主手册

> 目标岗位：Agent 开发、LLM 应用开发、AI 后端及相关工程岗位。
>
> 使用方式：面试前以本文为主；需要补算法时查《Search-R1 面试知识体系与学习路线》，需要核对完整实验时间线与证据时查《Search-R1 实验全流程复盘与面试讲稿》。
>
> 核对日期：2026-08-03。本文中的“我”用于模拟口述；正式回答时应使用自然表达，不要逐字背诵。

## 1. 整体面试策略

### 1.1 对外项目名

简历和默认口述使用：

> **基于强化学习的多轮检索 Agent**

不要在开场先抛出论文名。开发面试官首先需要理解的是：

- 这是一个什么 Agent；
- 它如何调用工具和管理上下文；
- 你解决了什么工程问题；
- 结果如何，哪里还不完善；
- 如果生产化，你会怎样继续设计。

### 1.2 一句话定位

> 我做了一个面向知识密集型问答的多轮检索 Agent，让 2B 模型能够在推理过程中自主决定何时搜索、搜索什么、如何利用结果继续推理，以及何时停止并给出答案，并通过强化学习同时优化答案正确率和检索开销。

### 1.3 关于 Search-R1 和开源基础的表述原则

不主动用论文名开场，不等于把公开方法说成个人原创。最稳妥也最能经受追问的边界是：

- 可以说这是你主导完成的个人项目；
- 可以说你实现和改造了 Agent、训练、数据、评测与故障诊断闭环；
- 不要说 GRPO 是你提出的；
- 不要说整个 RL 框架从零手写；
- 被问到技术来源、代码基础或是否参考论文时，直接说明参考了 Search-R1 的研究范式、开源仓库和 verl，再重点讲你的实质改造。

如果面试官问“这是不是 Search-R1 复现”，推荐回答：

> 研究范式和初始工程底座确实参考了 Search-R1 及其开源实现，我没有把论文方法当成自己的原创。我的项目也不是原样跑论文：模型换成了 Qwen3.5-2B，检索环境固定为 Wiki-2018 BM25，并针对原生 Tool Calling 重写了协议适配、上下文续接和环境 Token Mask；之后又做了奖励函数改造、检索可见性数据筛选、分层训练门禁、轨迹审计和统一配对评测。所以我把它定位为一个预算受控的多轮检索 Agent 训练与优化项目，而不是论文榜单的完整复现。

这个回答不会削弱项目，反而能把“会运行开源项目”与“真正做了工程和实验改造”区分开。

### 1.4 默认回答顺序

开发岗面试优先按照以下顺序讲：

```text
业务问题
  → Agent 架构
  → 最难的工程边界
  → 一次关键失败与诊断
  → 可靠结果
  → 当前限制和未来工作
```

不要一上来先推 GRPO 公式。只有面试官追问算法时，再逐层进入 advantage、PPO clip 和 reward normalization。

---

## 2. 面试开场：四个可直接使用的版本

### 2.1 十秒版本

> 我做的是一个能自主调用搜索工具的多轮检索 Agent。它不是固定先检索一次，而是在推理过程中动态决定是否继续搜索；我主要完成了 Tool Calling 闭环、强化学习奖励、训练门禁和逐轨迹评测。

### 2.2 三十秒版本

> 我基于 Qwen3.5-2B 做了一个多轮检索 Agent。模型每一轮可以选择调用 Wiki BM25 搜索，或者直接提交答案；搜索结果会作为环境观测回填上下文，最多执行四次真实检索。我适配了模型原生 Tool Calling、环境 Token Loss Mask 和终局强制作答，并用 GRPO 训练搜索与作答策略。最终在 512 题统一评测中，strict EM 从 21.29% 提升到 39.45%，总检索同时下降 21.8%。

### 2.3 九十秒主版本

这是最推荐背熟的版本：

> 这个项目的目标，是让一个小参数模型在知识密集型问答里，不只是被动接收一次 RAG 结果，而是能够自己决定什么时候搜索、query 怎么写、看到结果后要不要继续搜，以及什么时候结束。
>
> 系统上我用 Qwen3.5-2B 作为策略模型，接了一个固定 Wiki-2018 语料的 BM25 Top-3 检索服务。模型每轮生成 reasoning 后，通过原生 Tool Calling 选择 search 或 answer；search 的结果作为新的 observation 回填，最多执行四次真实搜索，预算耗尽后进入 answer-only 终局。我重点处理了工具协议、parser、上下文续接，以及一个很关键的 Token Mask：检索结果可以进入注意力上下文，但它是环境产生的，不能当成模型动作参与策略梯度。
>
> 训练时我遇到过一次很典型的奖励漏洞。最开始直接对每次搜索扣成本，模型最后有 98.44% 的题完全不搜索。我从 GRPO 同题五条轨迹的组内 advantage 定位到：当五条都答错时，答案奖励没有差异，搜索成本反而成为唯一排序信号，所以模型被鼓励成“不会也别搜”。我把奖励改成只在答对轨迹之间比较搜索成本，先消除了这个反向激励。
>
> 最终主训练模型在 512 题上把 strict EM 从 21.29% 提升到 39.45%，总搜索从 1690 次降到 1321 次。成本分支相对同源控制组又少了 8.18% 的搜索，准确率点估计基本持平。不过后期长输出截断和工具协议漂移仍然明显，所以我的结论不是“全面优化完成”，而是能力提升成立、成本控制出现了正向信号，下一阶段应先解决输出长度与终止稳定性。

说完后停下来，等待面试官选择追问方向。

### 2.4 三分钟展开版

> 普通 RAG 经常是固定拿用户原问题检索一次，再把文档拼到 Prompt。这个方案对于多跳问题不够灵活，因为第一轮检索可能只拿到中间实体，模型需要根据新证据改写下一轮 query。因此我做了一个多轮检索 Agent，把“推理、搜索、接收 observation、继续推理、最终回答”做成状态机。
>
> Agent 侧我主要做了三个边界。第一是结构化工具协议：模型只能生成合法 search 或 answer，parser 区分请求动作、合法动作和真实执行动作。第二是预算与终止：最多四次真实搜索，预算耗尽后追加一个环境侧 answer-only reminder；如果仍请求 search，系统记录并拒绝，不会越过预算。第三是训练 mask：assistant 生成的 reasoning、tool call 和 answer 参与策略更新，tool response、role marker 和 terminal reminder 只作为上下文，不参与 loss。
>
> 算法侧采用 GRPO。对同一道题采五条 rollout，用组内相对 reward 形成 advantage，不再训练单独 critic。最开始的 reward 是 `EM-搜索成本`，它导致了策略坍缩：在全错 group 中，少搜索的错误答案反而 reward 更高。我把它改成 `EM×(1-搜索成本)`，让全错 group 不再产生反搜索梯度，只在正确轨迹之间比较效率。
>
> 但奖励修复以后，第一轮效果仍不理想。进一步审计发现两个工程混杂：一是早期数据大多一次搜索就能完成，没有足够二次 query 的学习机会；二是 Qwen3.5 的原生 Tool Calling 被旧 XML 协议约束，模型能力、Prompt 和 parser 错配。我后来重做了 NQ/Hotpot 的数据配比和可见证据筛选，把工具链迁移为模型原生协议，并建立从离线模板、真实检索、模型行为、两步 backward 到 held-out 轨迹的分层门禁。
>
> 最终主模型在 val、NQ-test、multihop 三个端点都提升 strict EM，512 题描述性汇总从 21.29% 到 39.45%，同时减少 369 次搜索。成本分支又把同源控制组的总搜索从 1320 降到 1212，四次搜索跑满率从 38.28% 降到 30.27%，但三个端点的准确率差异区间都跨 0，所以我只把它定义为效率改善，不说能力提升。
>
> 当前最大的未解决问题是长输出 clipping 和 native action 稳定性。主模型虽然更准，但 clipping 也上升；后续 20 步在 val 上继续涨、multihop 却下降，表现出分布特化。下一轮我会先做短格式 SFT smoke、长度与终止约束，再用完整 trainer-state checkpoint、多 seed 和预注册非劣界重做成本实验，而不是继续盲目加训练步数。

---

## 3. 面试官追问路线图

```text
“系统怎么做的？”
  → 第 4、5、6 节：架构、状态机、协议与 Mask

“你个人做了什么？”
  → 第 7 节：上游基础与个人贡献边界

“为什么要强化学习？”
  → 第 8、9 节：训练目标、GRPO 与奖励坍缩

“结果为什么不完美？”
  → 第 11、12 节：结果边界、截断、分布特化、未来工作

“怎么工程化/上线？”
  → 第 14 节：服务架构、延迟、缓存、容错、安全和监控

“是不是论文复现？”
  → 第 15 节：项目来源与原创性问答
```

开发面试官如果没有主动深入算法，重点放在状态机、可靠执行、可观测、检索服务、训练数据流和故障隔离上。

---

## 4. 系统架构

### 4.1 白板图

```text
                         ┌──────────────────────────┐
用户问题 ──────────────→│ Qwen3.5-2B Policy Model │
                         └─────────────┬────────────┘
                                       │ reasoning + structured action
                                       ▼
                         ┌──────────────────────────┐
                         │ Protocol Adapter / Parser│
                         └─────────┬─────────┬──────┘
                                   │ search  │ answer
                                   ▼         ▼
                         ┌──────────────┐   最终短答案
                         │ Budget Guard │
                         └──────┬───────┘
                                │ 合法且预算允许
                                ▼
                         ┌──────────────┐
                         │ Wiki BM25    │──→ Top-3 documents
                         └──────┬───────┘
                                │ tool observation
                                ▼
                  回填对话上下文，继续下一轮生成
                  observation 进 attention，不进 policy loss

每一步同时写入 Trace：
action / query / executed_search / observation / invalid / clipping / answer
                                │
                                ▼
                    EM Reward + Search Cost
                                │
                                ▼
                    GRPO / FSDP / Ray / vLLM
```

### 4.2 八个核心模块

| 模块 | 职责 | 面试重点 |
|---|---|---|
| Policy Model | 生成 reasoning、tool call 和 answer | 模型既是语言模型，也是策略 |
| Protocol Adapter | 渲染原生工具 schema、续接多轮对话 | 训练和推理必须使用同一模板与 token 边界 |
| Action Parser | 判定 search、answer 或 invalid | 不能把自由文本或请求动作直接当作成功执行 |
| Budget Guard | 限制最多四次真实搜索 | 预算是环境硬约束，不只靠 Prompt 提醒 |
| Retriever | Wiki-2018 BM25 Top-3 | 固定语料、索引与配置，控制实验变量 |
| Context/Mask | 回填 observation，构造 attention 与 loss mask | 环境 Token 不参与策略梯度 |
| Reward/Trainer | 计算 EM、搜索成本并进行 GRPO 更新 | 奖励方向和组内归一化共同决定策略行为 |
| Trace/Evaluator | 保存逐步事件并做配对评测 | 总体分数之外，还能定位协议、截断和检索失败 |

### 4.3 一条请求的完整生命周期

1. 输入问题和工具 schema，初始化对话状态。
2. 模型生成一段 assistant 输出。
3. parser 在合法 action region 内提取动作。
4. 如果是 answer，保存短答案并结束。
5. 如果是 search，先检查格式和剩余预算。
6. 合法 search 才调用 BM25，并增加 `executed_search_count`。
7. Top-3 文档经过 token 上限裁剪，形成模型真正可见的 observation。
8. observation 以 tool response 身份写回上下文。
9. 继续下一轮，直到 answer 或搜索预算耗尽。
10. 预算耗尽后进入 answer-only terminal generation；继续 search 会被拒绝并记录。
11. 轨迹结束后计算 strict EM、搜索成本、invalid、clipping 等指标。
12. 训练阶段把同题五条轨迹组成 GRPO group，计算相对 advantage 并更新模型。

### 4.4 四个系统不变量

面试时能主动说出这四条，会很像真正做过系统：

- **执行不变量：**真实搜索数不超过预算；请求搜索不等于执行搜索。
- **Token 不变量：**环境 observation 能影响后续生成，但不能贡献 policy loss。
- **协议不变量：**训练 rollout、log-prob replay 和最终评测使用同一 chat template、tool schema 与 action boundary。
- **证据不变量：**每个分数都能回到 sample ID、模型身份、逐步 trace 和固定评测配置。

---

## 5. 为什么它不是普通 RAG

| 维度 | 固定式 RAG | 本项目多轮检索 Agent |
|---|---|---|
| 检索时机 | 通常生成前固定一次 | 模型在推理中动态选择 search 或 answer |
| Query | 常直接使用原问题 | 可依据中间证据重写后续 query |
| 轮数 | 一次或固定流程 | 0—4 次真实搜索，取决于策略 |
| 状态 | 问题 + 一次检索结果 | 问题 + reasoning + 多轮工具历史 |
| 优化目标 | 常优化召回或最终生成 | 直接用答案结果和工具成本训练完整轨迹 |
| 主要难点 | 文档切分、召回、Prompt | 额外包含动作协议、信用分配、终止和环境 Token Mask |

标准回答：

> 普通 RAG 解决“给模型哪些外部知识”，这个项目还解决“模型何时需要知识、下一步搜什么、什么时候应该停”。因此 retriever 只是一个工具，真正被训练的是围绕工具使用的序列策略。

---

## 6. Agent 开发最值得讲的工程细节

### 6.1 为什么迁移到 Qwen3.5 原生 Tool Calling

早期路径强迫模型输出旧 XML 标签，而 Qwen3.5 本身有原生工具 schema 和 `<tool_call>/<tool_response>` 对话语义。错配会造成：

- Prompt 教一种格式，模型预训练偏好另一种格式；
- parser 与模型真实 action boundary 不一致；
- tool response 的 role 和 token 边界错误；
- 自由文本、reasoning、action 和 answer 混在一起；
- 最终把协议失败误判为模型没有检索能力。

迁移后做的不是简单替换字符串，而是统一：

- tool schema；
- chat template；
- assistant suffix 的 token 续接；
- structured action parser；
- tool response 回填；
- answer 与 terminal boundary；
- rollout 和 log-prob replay 的上下文。

### 6.2 为什么不能 decode 全文再 encode

多轮生成中，如果把已有 token 全部 decode 成文本再重新 encode：

- BPE 切分不保证往返完全一致；
- chat template 可能 trim 或重写 assistant 内容；
- 新旧 token 序列不同，rollout 时记录的 old log-prob 与训练 replay 不再对应；
- PPO/GRPO 的 ratio 失去严格语义。

项目使用真实模板提取本轮 assistant 后缀，并以 token 级方式拼接后续上下文，避免静默改变已生成动作。

### 6.3 Observation Mask

推荐回答：

> 搜索结果是环境给模型的 observation，不是 policy 采样的 action。它需要进入 attention，让模型能够基于证据继续推理；但如果它参与策略梯度，就相当于把环境产生的文本也当成模型自己生成的 token，会破坏 log-prob 和信用分配。因此我维护 attention mask 与 info/loss mask 两套语义：前者表示上下文可见，后者只选择 assistant 真正生成的 token。

### 6.4 请求搜索与真实搜索为什么要分开

模型可能：

- 生成非法 JSON；
- 缺失 query；
- 在预算耗尽后继续请求 search；
- 输出一个看起来像工具调用但 parser 不接受的片段。

这些只能计为 requested/invalid action，不能增加真实工具成本。成本奖励使用 `executed_search_count`，从环境执行层记录，避免模型文本与真实资源消耗不一致。

### 6.5 为什么需要 answer-only terminal

只设置“最多四轮”但直接截断，会产生没有最终答案的轨迹；允许第五次 search 又破坏预算。解决方式是：

- 四次真实搜索后，环境追加固定 answer-only 指令；
- 该指令是环境 token，不参与 policy loss；
- 模型获得额外一次只允许回答的生成机会；
- 如果仍请求 search，parser 明确拒绝并记录；
- 这样搜索预算和最终可评分答案可以同时成立。

---

## 7. “你个人做了什么”

### 7.1 二十秒版本

> 我不是从零发明 GRPO 或训练框架。上游提供了基础 RL 和检索框架，我主要负责把它改造成适配 Qwen3.5 的可训练多轮 Agent：重做原生工具协议、上下文和 Token Mask，设计成本奖励，重构多跳数据，建立训练门禁与云端恢复链路，再用逐轨迹和配对评测定位策略坍缩、截断和协议漂移。

### 7.2 上游基础与个人工作边界

| 上游已有基础 | 项目中的个人核心工作 |
|---|---|
| Search-R1 的检索强化学习研究范式 | 把问题重新落到 Qwen3.5-2B、BM25 和预算受控硬件环境 |
| verl 的 PPO/GRPO、Ray、FSDP 基础设施 | 接通 native Agent rollout、loss mask、训练合同和门禁 |
| 检索抽象和基础 BM25 能力 | 固定 Wiki 语料/索引、验证实际可见证据、构建多跳配额 |
| 基础生成循环 | 实现原生 Tool Calling、严格 parser、终局回答和事件记录 |
| outcome reward 基础 | 增加真实搜索成本、correct-only 模式及对应诊断指标 |
| 常规模型评测 | 建立同题 A/R/B/C 配对评测、bootstrap CI 和逐轨迹 transition |
| 普通训练脚本 | 建立 CPU/GPU 分阶段、immutable attempt、日志/退出码、恢复与自动关机 |

### 7.3 可以点名的代码入口

- `search_r1/llm_agent/tool_protocol.py`：原生工具 schema、conversation adapter 和 parser；
- `search_r1/llm_agent/generation.py`：多轮 Agent loop、observation 回填、terminal 和 mask；
- `search_r1/trajectory_trace.py`：可审计轨迹 schema；
- `verl/trainer/main_ppo.py`：EM、真实搜索成本和 correct-only reward；
- `scripts/data_process/search_mix.py`：数据配额、检索证据与 manifest；
- `scripts/autodl/paired_eval.py`：配对结果、transition 和 bootstrap；
- `scripts/autodl/`：分层门禁、训练、watchdog、恢复和证据封存。

面试官如果要求打开代码，优先从 `generation.py` 的状态更新与 mask、`main_ppo.py` 的 reward、`paired_eval.py` 的统计三个入口讲。

---

## 8. 为什么使用强化学习，而不是只做 Prompt 或 SFT

### 8.1 开发面试官版本

> Prompt 可以告诉模型“必要时搜索”，SFT 可以模仿已有轨迹，但它们都不能直接表达我最终关心的组合目标：答案要正确，同时不要做冗余搜索。强化学习可以让模型在真实执行工具后，根据整条轨迹的结果更新策略，因此更适合学习何时搜索、搜几次和何时停止。

### 8.2 进一步追问

- Prompt 的行为依赖模型原有能力，遇到未覆盖状态时不稳定；
- SFT 需要高质量专家轨迹，并会照抄轨迹中的冗余搜索；
- DPO 需要 chosen/rejected 偏好对，标准版训练时不持续与检索环境交互；
- 在线 GRPO 可以让当前策略真实执行 query，再用最终 EM 和成本评价 rollout。

### 8.3 为什么未来又考虑 SFT

SFT 与 RL 不是二选一。当前主要瓶颈已经从“完全不会搜索”转为“长输出和工具格式不稳定”，短格式 SFT 可以先教稳定的工具与终止协议，再由 RL 优化任务结果。

必须说明：本轮实际路线是 direct outcome-RL；SFT 格式热身是未来工作，没有在当前结果中执行。

---

## 9. GRPO 与奖励坍缩

### 9.1 非算法面试官版本

> GRPO 可以理解为：同一道题让模型生成五种完整解法，再在这五条里比较谁更好。比组内平均更好的轨迹被提高概率，更差的轨迹被压低概率。它不需要另外训练一个 critic，但要求同题的多条轨迹之间确实存在好坏差异。

### 9.2 公式版本

```text
             r_i - mean(r_1, ..., r_G)
A_i =  ------------------------------------
       std(r_1, ..., r_G) + ε
```

项目中 `G=5`。更新仍使用 PPO 风格的 probability ratio 和 clip。

### 9.3 旧奖励为什么会坍缩

旧奖励：

```text
r_old = EM - 0.10 × n_search / 4
```

如果五条轨迹都错：

```text
r_i = -0.10 × n_i / 4
```

不搜索的错误轨迹 reward 最高，会得到正相对 advantage；搜索更多但同样错误的轨迹得到负 advantage。模型最终学成“答错也不要搜索”。

### 9.4 为什么把 lambda 调小不能根治

组内标准化近似满足：

```text
(λx - mean(λx)) / std(λx)
≈ (x - mean(x)) / std(x)
```

统一缩小成本不会改变组内排序，标准化还会抵消尺度。因此问题是奖励方向，而不只是系数太大。

### 9.5 Correct-only reward

```text
r_gated = EM × (1 - 0.10 × n_search / 4)
```

- 全错 group：全部为 0，不再产生反搜索梯度；
- 有正确轨迹：先区分对错，再在正确轨迹中偏好更低成本；
- 局限：全错 group 也没有学习信号，数据、探索和格式仍然必须能产生正确轨迹。

### 9.6 如果问“这个奖励完美吗”

> 不完美。它解决的是错误轨迹上的反搜索激励，但没有解决 all-wrong group 信号稀疏，也不能直接判断某次搜索是否真正贡献了答案。当前搜索节省又有较大部分来自共同失败轨迹，所以未来需要引入更细的过程信号，例如合法动作、证据利用、终止质量或可验证的中间子目标，同时要防止过程奖励被模型再次钻空子。

---

## 10. 数据与检索为什么也是训练问题

### 10.1 早期数据的问题

如果大部分题一次搜索就能回答，那么成本优化最容易学到的是“一次降到零”，而不是“从四次冗余搜索降到两次必要搜索”。这会让成本实验缺少真正的决策空间。

### 10.2 为什么增加 Hotpot bridge

- NQ 更偏单事实；
- Hotpot comparison 通常需要比较两个实体；
- Hotpot bridge 需要先找到中间实体，再发起下一轮 query；
- bridge 更能制造“看完第一轮结果后，下一步搜什么”的学习机会。

### 10.3 为什么标签是多跳还不够

数据集标成多跳，不代表当前 Wiki 版本、BM25、Top-3 和 observation 截断下真的能看到证据。项目用 `visible_observation` 作为真值：只有模型截断后实际可见的文本才能证明样本可学。

### 10.4 为什么用 BM25

> 我选择 BM25 不是认为它一定优于 dense retriever，而是为了在预算受控实验中固定检索环境。它 CPU 友好、结果可解释、索引与语料容易做 hash，实体问答也有较强词面匹配。代价是同义改写和隐式关系召回较弱；如果生产化，我会做 BM25+dense 的 hybrid recall 和 rerank，但保持训练对比中 retriever 版本固定。

---

## 11. 结果应该怎样讲

### 11.1 三层结论

**第一层：能力主结果，最可靠。**

- 512 题描述性汇总 strict EM：`109/512 → 202/512`，即 `21.29% → 39.45%`；
- 总搜索：`1690 → 1321`，减少 369 次，即 `-21.83%`；
- val、NQ-test、multihop 三端点的 EM 配对 bootstrap 95% CI 均不跨 0；
- 因而可以说答案能力明确提升，而且不是靠更多搜索换来的。

**第二层：成本分支，效率结论强于能力结论。**

- strict EM：`36.33% → 37.11%`，只增加 0.78 个百分点；
- 三端点的准确率差异 CI 全部跨 0，不能说显著提升；
- 总搜索：`1320 → 1212`，减少 108 次，即 `-8.18%`；
- 四次搜索饱和率：`38.28% → 30.27%`；
- val 和 multihop 的搜索下降有配对区间支持；
- 零搜索轨迹仍只有 `1/512`，没有再次发生 no-search collapse。

**第三层：稳定性仍未解决。**

- 主模型相对基线更准，但 clipping 从 `17.97%` 上升到 `38.09%`；
- 后续 B/C 评测 clipping 约为 `70.90% / 71.68%`；
- B/C 的 invalid trajectory 约为 `34.57% / 37.89%`；
- 所以后续训练不是全面改进，协议与长度问题仍会吞掉最终答案。

### 11.2 一句话结果

> 主训练证明模型学会了更有效的搜索与作答，能力提升明确；成本奖励进一步降低了搜索深度但没有确认准确率提升；长输出截断和协议稳定性仍是下一阶段瓶颈。

### 11.3 为什么结果不完美反而可以讲

因为它提供了三种不同层面的判断：

- **成功项：**Agent 能力和部分效率真正提高；
- **trade-off：**成本分支少搜索，但不能证明更准；
- **失效项：**继续训练后出现 clipping、invalid 和 multihop 回退。

这比“所有指标都涨”更能展示你会做系统诊断，而不是只会挑一个最好数字。

---

## 12. 截断、准确率下降与未来工作

### 12.1 为什么后续多训练 20 步反而下降

推荐回答：

> 强化学习的 held-out 指标不保证随训练步数单调上升。这次后续 20 步在 val 上继续提高，但 multihop 从 37.11% 降到 30.08%，更像向训练分布特化。与此同时 B/C 的 clipping 升到约 71%，很多轨迹在最终答案前就撞到单轮长度上限，或者发生 native action 格式漂移。另外 B/C 是从 R 的模型权重启动新 optimizer，不是带 optimizer 和 scheduler 的严格断点续训。因此“多 20 步”不等于在同一优化轨迹上稳定续跑，也不能推出效果应当单调更好。

### 12.2 Clipping 为什么会降低准确率

可能路径：

```text
reasoning / query 越来越长
  → 单轮生成达到 500-token 上限
  → 合法 action 或最终 answer 没有生成完整
  → parser 记为 invalid、missing answer 或 forced terminal
  → 即使中间已经看到正确证据，strict EM 仍为 0
```

还要注意：clipping 与错误高度相关，不等于已经证明 clipping 单独造成所有回退；二者可能共同来自策略漂移或长度偏好。

### 12.3 为什么不直接把 500 token 调大

> 直接加长度会提高显存、KV cache、rollout 时间和工具循环成本，还可能纵容模型继续冗长推理。我的优先方案是先让协议和答案更短、更稳定，例如短格式 SFT、长度分桶、明确的终止训练信号和基于轨迹洁净度的门禁；只有确认正确轨迹确实被硬长度截断后，才受控增加上限做消融。

### 12.4 下一轮未来工作优先级

1. **先修格式稳定性。** 用少量高质量短轨迹做 SFT smoke，降低 thinking-prefix、missing action 和 terminal answer 错误。
2. **改进长度与终止。** 对 response length、重复 query、达到证据后仍继续搜索建立可审计指标；避免简单按长度扣分再次产生 reward hacking。
3. **恢复完整 checkpoint。** 保存 optimizer、scheduler、trainer state 和 RNG，支持真正 resume、早停和最佳 step 选择。
4. **提高有效 group 比例。** 通过难度课程、dynamic sampling 或 DAPO 式过滤减少全对/全错零信号 group。
5. **做确认性成本实验。** 从同一稳定 parent 平行训练控制与成本分支，使用多 seed、预注册 primary endpoint 和 EM 非劣界。
6. **改进检索。** 在固定协议下比较 BM25、dense 和 hybrid，单独衡量召回问题与策略问题。
7. **扩展评测。** 增加语义答案指标、工具成功率、证据利用、延迟和真实资源成本，但保留 strict EM 作为既有主指标。

### 12.5 如果让你只选一个未来工作

> 我会先修 clipping 和 terminal compliance，而不是先换更大的模型或更多训练步。因为 clean 轨迹的正确率明显更高，当前瓶颈已经不只是知识能力，而是模型经常无法把已有能力稳定地通过合法 action 和最终答案表达出来。

---

## 13. 高频问答：开发、算法与实验

### Q1：为什么要做这个项目？

> 普通 RAG 通常固定检索一次，对需要分解问题和多轮补证据的任务不够灵活。我想研究能否让小模型把搜索当成真实工具，自主学习 query 改写、停止和成本控制，同时把整个过程做成可观测、可复算的工程系统。

### Q2：最难的部分是什么？

> 最难的不是把 BM25 API 接起来，而是保证 rollout、工具环境和训练 replay 的语义一致。特别是 native Tool Calling 的 token 边界、environment observation mask、真实执行搜索计数和 terminal action，都直接决定策略梯度是否在优化正确的动作。

### Q3：为什么不用 LangChain 或现成 Agent 框架？

> 这类框架适合应用编排，但训练场景需要精确控制每个 token 的来源、old log-prob、loss mask 和 action boundary。高层框架可能隐藏 decode/encode、消息重写和重试逻辑，破坏 on-policy 合同。因此核心训练 loop 使用可审计的自定义实现；生产应用层仍可以在外部使用成熟编排框架。

### Q4：Tool Calling 如何保证可靠？

> 使用模型原生工具 schema、严格 parser 和环境硬约束。生成文本只有在合法 action region 内解析成 search，并且参数、预算都通过检查后才执行。非法动作、预算后 search、缺失答案都会作为不同失败类型记录，不会静默修成成功。

### Q5：模型生成了 search 就一定会访问检索服务吗？

> 不一定。请求、解析成功和真实执行是三个层级。成本按环境侧 `executed_search_count` 计算，避免非法调用或预算后调用被误算为真实成本。

### Q6：为什么最多四次搜索？

> 它是在多跳能力、上下文长度和训练成本之间的实验预算。四次足以覆盖多数两到三跳链路，又能限制无限循环。是否最优没有被证明，生产化会根据任务分布、延迟 SLO 和边际收益重新标定。

### Q7：为什么 Top-3？

> Top-3 控制 observation 长度和噪声，同时保留少量候选证据。Top-k 越大不一定越好，会增加上下文、干扰和生成成本。当前把 Top-3 固定是为了隔离策略训练变量，后续可对 Top-k 做召回—延迟消融。

### Q8：检索结果太长怎么办？

> 先按 token 上限裁剪，并把裁剪后真正进入模型的 `visible_observation` 落盘。数据可学性和轨迹分析都以可见文本为准，不能拿原始检索全文证明模型已经看到证据。

### Q9：如何判断失败来自检索还是模型？

> 逐轨迹检查四层：gold 证据是否存在于固定语料；BM25 Top-3 是否召回；裁剪后 observation 是否仍可见；模型是否基于可见证据生成合法后续 action 和答案。这样能把 corpus、retriever、context 和 policy 失败分开。

### Q10：为什么使用 strict EM？

> 它简单、可程序化、可直接作为 outcome reward，适合短答案问答。但它对 alias、日期格式和冗长表达敏感，所以我保留它作为既定主指标，同时用逐题分析解释语义接近但 EM 失败的样本；未来会增加语义或判分模型指标，但不能事后替换主指标美化结果。

### Q11：为什么使用 GRPO？

> 任务答案可自动验证，同一道题也能采多条轨迹。GRPO 用组内相对 reward 替代 critic，减少 value model 的显存和训练复杂度，适合预算受控的小模型实验。代价是同题需要多 rollout，而且全对或全错 group 没有相对信号。

### Q12：GRPO 与 PPO 的区别？

> PPO 通常训练 critic，再通过 value 和 GAE 估计 advantage；GRPO 对同一 Prompt 采多条回答，用组内均值和标准差构造相对 advantage，不需要 learned critic。两者都可使用旧策略 ratio、clip 和 KL 来限制更新幅度。

### Q13：为什么不用 DPO？

> 标准 DPO 需要已有 chosen/rejected 偏好对，训练时不持续与检索环境交互。本项目需要当前策略真实生成 query、执行搜索，再根据 observation 产生后续动作，因此在线 outcome RL 更直接。DPO 可以作为成功/失败轨迹的离线预热，但不能无条件替代真实工具环境训练。

### Q14：为什么不用纯 SFT？

> SFT 适合学习工具格式和模仿专家轨迹，但不直接优化最终 EM 与搜索成本，也会继承示范中的冗余调用。当前未来方案是用短 SFT 修协议，再用 RL 优化结果，而不是把两者当作互斥方案。

### Q15：奖励函数为什么出问题？

> 因为奖励定义的局部最优与真实目标不一致。在全错 GRPO group 中，线性成本让不搜索的错误轨迹成为相对最好，模型精确地优化了错误目标。这是 reward hacking，不是训练代码没有运行。

### Q16：Correct-only 会不会忽略错误轨迹的真实成本？

> 会。它是为消除反搜索梯度做的结构修复，不是最终完美效用函数。报告时我仍按所有轨迹计算 post-hoc 成本；未来要在不重新鼓励零搜索的前提下，引入更好的分层目标或约束优化。

### Q17：为什么修完奖励，第一次效果仍没超过控制组？

> 修复只消除了错误方向，没有凭空创造正确轨迹。当时大量 group 全错，正确且搜索次数不同的轨迹太少；同时 NQ 题很多一次搜索即可完成，可压缩空间接近下限，所以数据分布和探索信号仍不足。

### Q18：什么是 clipping？

> 这里有两个相关概念。PPO clip 是限制新旧策略概率比变化；generation clipping 是生成撞到 token 上限。本项目主要的工程风险是后者：动作或答案没生成完整。高 PPO clip fraction 也能提示更新过大，但不能与 response clipped trajectory 混为一个指标。

### Q19：为什么继续训练会协议漂移？

> outcome reward 只看最终 EM，未直接约束 reasoning 长度、工具格式和 terminal compliance。模型可能找到能提高局部 reward 但输出更长、更不稳定的策略；当更新幅度、数据分布和稀疏奖励叠加时，格式能力会被侵蚀。

### Q20：如何防止训练把基础能力训坏？

> 使用 reference KL、PPO clip、较小更新、固定 held-out gate、协议指标和早停；不能只看训练 reward。还应保存多个完整 checkpoint，按预注册主指标与稳定性共同选点，而不是默认最后一步最好。

### Q21：分层门禁分别做什么？

> 离线层检查模板、token、parser 和数据证据；环境层检查真实检索与预算；模型行为层检查合法 action、query 和答案；两步 smoke 验证 backward、optimizer 和 FSDP；held-out gate 才检查能力、可学习 group、invalid 和 clipping。这样昂贵训练前可以先隔离便宜错误。

### Q22：为什么两步 smoke 还不够？

> smoke 只能证明训练链路能执行、梯度和权重会变化，不能证明长训练科学上有效。策略坍缩、分布特化和协议漂移常在更多更新后出现，所以仍需要独立 held-out gate 和轨迹分析。

### Q23：FSDP、Ray、vLLM 分别做什么？

> FSDP 分片模型参数、梯度和 optimizer state，支持两卡全参数训练；Ray 编排 actor、rollout、reference 等 worker；vLLM 提供高吞吐 rollout。关键工程问题是 actor 与 rollout 权重同步，以及训练、生成和 KV cache 对显存的竞争。

### Q24：为什么 B/C 不是严格断点续训？

> R60 只保存了 Hugging Face 模型权重，没有 optimizer、scheduler、trainer state 和 RNG。B/C 使用同一 R 权重作为初始化，但新建 optimizer，所以是同源新 run，不是无缝 resume。

### Q25：如何保证实验可复现？

> 固定 commit、模型、数据 manifest、语料与索引 hash、Prompt/tool schema、seed 和解码；每次运行使用 immutable attempt，持久化原始日志、退出码、轨迹和 checkpoint lineage；评测在同题同协议下配对执行。

### Q26：为什么需要配对评测？

> A/R/B/C 在同一题上的错→对、对→错和搜索变化，比两个总体平均值更能解释行为。paired bootstrap 也是对题目对重采样，保留配对结构，适合估计 candidate-baseline 差值区间。

### Q27：可以说结果显著吗？

> A→R 的三个端点 EM 95% CI 都不跨 0，可以说提升方向稳定。B→C 的准确率区间全部跨 0，只能说点估计基本持平；搜索下降在 val 和 multihop 有配对证据。必须按指标分别说。

### Q28：为什么 512 题汇总只是描述性？

> 三个端点大小不同，multihop 256 题会获得两倍权重，合并值不是自然统一 benchmark。正式推断按端点报告，512 行只用于快速描述总体观测。

### Q29：这个项目最大的限制是什么？

> 单训练 seed、每题 greedy 单 rollout、端点规模较小，以及高 clipping/invalid。它能证明这一次受控实验中的能力与效率方向，但不能证明对所有 seed、模型和线上流量都稳定成立。

### Q30：如果准确率下降，项目还有价值吗？

> 有。主训练的能力提升明确；下降发生在后续探索分支和特定 multihop 分布。更重要的是，轨迹证据把原因收敛到长度、协议稳定性、分布特化和成本—深推理 trade-off，为下一轮提供了可验证假设。负结果不是价值本身，能定位并设计下一步才是价值。

### Q31：为什么不用更大的模型？

> 选择 2B 是为了在两卡预算内完成多 rollout 全参数训练和多轮评测，并把工程变量控制住。更大模型可能有更强基础能力，但会显著提高 rollout 和 optimizer 成本，不能替代对协议、奖励和数据问题的诊断。

### Q32：如果检索服务挂了怎么办？

> 当前实验使用固定本地服务并通过门禁检查健康性，失败时应 fail closed 并记录环境错误，不能把空 observation 当正常轨迹继续训练。生产化还需要超时、有限重试、熔断、降级和请求级 trace ID。

### Q33：如何控制云训练费用？

> CPU 完成数据、hash、模板和静态检查；GPU 先跑小 gate 和两步 smoke，再启动长训练；任务 detached 运行，持久化原始退出码和日志；成功或失败后触发关机，同时由控制台人工确认计费停止。

### Q34：为什么不删除失败轨迹再算结果？

> 主结果必须包含所有符合预定评测合同的轨迹。clean/dirty 切片只能用于诊断，不能用 clean 子集替代总体结果，否则会产生选择偏差。

### Q35：这个项目对 Agent 开发岗位有什么直接价值？

> 它覆盖了 Agent 开发最容易出问题的边界：结构化工具调用、多轮状态、上下文来源、预算与终止、真实工具成本、可观测、失败恢复和离线评测。我不仅接了一个工具，还把 Agent 的行为变成可训练、可审计和可诊断的系统。

### Q36：你怎样测试这个 Agent？

> 我把测试分成四层：单元层测试 schema、parser、reward 和 trace 校验；合同层验证 chat template、token 续接与 loss mask；集成层跑真实 BM25、预算和 terminal 闭环；训练层先做两步 backward smoke，再做 held-out gate 和逐题配对评测。这样失败时能知道是函数错误、协议错误、环境错误还是策略问题。

### Q37：模型重复搜索或陷入循环怎么办？

> 当前环境用四次真实搜索预算提供硬上限，trace 记录每轮 query，门禁会统计重复或退化 query，因此循环不会无限执行，但可能浪费预算。生产化会再对规范化 query 做历史去重、连续重复检测和边际证据检查，在触发时要求改写 query 或进入回答阶段。

### Q38：多轮上下文超过容量怎么办？

> 当前 native 路径为 prompt、response 和 observation 设置显式容量，并拒绝静默左裁有效历史；每次 observation 截断后的可见文本也会记录。生产化可以对旧 observation 做带引用的证据压缩或分层记忆，但压缩前后必须保留来源和可追溯性，不能让摘要悄悄改变训练或评测语义。

### Q39：为什么 parser 不自动把非法动作修好？

> 评测和训练中自动修复会把模型没有生成的合法动作伪造成成功，还会隐藏策略退化，因此核心路径选择 fail closed，并把错误分类落盘。线上产品可以允许一次有边界的格式重试，但重试提示属于环境事件，原始动作、重试次数和最终执行结果都必须保留。

---

## 14. 如果面试官让你做生产化设计

> 本节是“如果继续上线，我会怎样设计”，不能说成实验版已经全部实现。

### 14.1 生产架构

```text
API Gateway / Auth / Rate Limit
              │
              ▼
      Agent Orchestrator
       ├─ Session State
       ├─ Policy / Model Gateway
       ├─ Tool Registry + Schema Validation
       ├─ Budget / Timeout Controller
       └─ Trace / Metrics
              │
       ┌──────┴─────────┐
       ▼                ▼
 Search Service     Other Tools
 BM25/Dense/Hybrid  allowlisted APIs
       │
       ▼
 Rerank / Evidence Compression / Cache
```

### 14.2 延迟和成本

总延迟近似：

```text
总延迟 = Σ 每轮模型生成 + Σ 工具调用 + 排队与网络开销
```

优化顺序：

- 限制最大工具轮数和单轮 token；
- 流式生成与工具调用并行准备；
- 对规范化 query 做 Top-k 结果缓存；
- BM25 与 dense 并行召回，按预算决定是否 rerank；
- 小模型负责 query/stop，必要时路由大模型回答；
- 记录每题 token、检索、延迟和成功率，按业务效用调预算。

### 14.3 并发与状态

- 每个请求使用独立 session/trajectory ID；
- Agent state 显式存储，不依赖进程内隐式变量；
- search 是只读工具，可按 query+index revision 做幂等缓存；
- 模型生成本身可能随机，重试必须保留 attempt ID，不能覆盖第一次结果；
- 限制单租户并发和累计工具预算，防止失控循环。

### 14.4 容错

- 工具调用设置超时和有限次数重试；
- 参数错误不重试，网络瞬时错误才重试；
- 检索服务熔断时可降级为直接回答，但必须在结果中标记无检索；
- terminal 和总请求 deadline 是硬边界；
- 每次降级、拒绝和重试写入统一 trace。

### 14.5 安全

- 工具 allowlist 和参数 schema 校验；
- 将检索内容视为不可信 observation，不能覆盖 system policy；
- 对 URL、文件、数据库等高风险工具增加权限与审批层；
- 防止 Prompt Injection、数据外传和跨租户状态污染；
- 日志脱敏，不记录密钥和不必要的用户原文。

### 14.6 线上指标

至少监控：

- 任务成功率/人工满意度；
- 平均及 P95 工具次数；
- 零工具、预算耗尽和重复 query 比例；
- tool success、timeout、invalid action；
- 首 Token、总延迟、token 成本；
- answer-after-evidence、evidence citation 和 fallback；
- 按任务类型分层的成功率，防止总体平均掩盖多跳回退。

---

## 15. 项目来源、开源与原创性问答

### Q1：这是不是 Search-R1？

> 是参考了 Search-R1 的研究范式和开源工程底座，但不是原配置照跑，也没有复现论文榜单。我的工作重点是 Qwen3.5 native Tool Calling 适配、环境 Token Mask、奖励漏洞修复、可见证据数据重构、训练门禁、云端恢复和配对评测。

### Q2：为什么简历没有写“复现 Search-R1”？

> 一页简历更希望先让开发面试官看到问题、架构和我的工程贡献，所以用了“多轮检索 Agent”这个任务型标题，没有用论文名占据第一信息位。如果讨论技术 lineage，我会明确说明参考来源，不把公开算法和上游代码说成原创。

### Q3：有多少代码是你自己写的？

> 不是从空仓库开始。上游已有 verl 的训练框架和检索抽象；我主要改动的是模型侧 native protocol、多轮 token 续接、observation/loss mask、reward manager、数据物化与证据筛选、轨迹 schema、分层 gate、云端训练恢复和 paired evaluator。面试时我可以从这些文件逐个讲输入、输出和不变量。

### Q4：你提出了什么新算法？

> 我没有声称提出新的通用 RL 算法。我的贡献是发现线性成本在 GRPO 全错 group 中形成反搜索激励，把它改成 correctness-gated 目标，并在真实多轮工具环境中完成诊断和受控评测。这是任务级奖励与工程改造，不是 GRPO 本身的原创。

### Q5：是否完整复现了论文结果？

> 没有。模型、retriever、硬件、数据规模和 benchmark 都不同，所以只能称为预算缩小的研究范式复现与改进，结论限于同一冻结环境中的相对比较。

### Q6：既然用了开源代码，项目价值在哪里？

> 工程项目的价值不只在从零写框架，而在于能否理解系统不变量、做出有效改造并用证据证明。这个项目中，如果不修工具协议、mask、reward 和数据可见性，训练虽然能跑完，却会得到错误科学结论。我的主要价值是把这些混杂拆开并建立可审计闭环。

---

## 16. 面试措辞：可以说与不要说

| 推荐说法 | 不要说 |
|---|---|
| 基于强化学习的多轮检索 Agent | 我原创了 Search-R1 |
| 参考公开研究范式与开源底座并完成针对性改造 | 整套框架完全从零手写 |
| 主模型在三个端点的 EM 提升区间均不跨 0 | 所有指标都显著提升 |
| 成本分支的可信结果是搜索效率改善 | C 的能力一定比 B 更强 |
| 后续训练出现分布特化和协议稳定性问题 | 多训 20 步理论上一定更好 |
| B/C 从同一模型权重启动新 optimizer | B/C 是严格断点续训 |
| G3 有能力信号，但正式门禁是 NO-GO | G3 已经通过 |
| SFT 是下一轮格式热身方案 | 本轮已经先 SFT 再 RL |
| clipping 与错误强相关，是重要风险 | 已证明所有错误都是截断造成 |
| 512 题是描述性汇总，正式结论按端点 | 在统一 512 benchmark 显著提升 |

### 16.1 不熟悉算法的开发面试官

用“同题五条方案做组内比较”解释 GRPO；不要主动写一屏公式。

### 16.2 熟悉论文的算法面试官

主动交代上游来源、direct outcome-RL、group size、reward normalization、PPO clip、KL、zero-variance group 和实验限制。

### 16.3 对结果提出质疑的面试官

先承认边界，再拿配对证据和逐轨迹诊断回答；不要为了维护项目而把不显著说成显著。

---

## 17. 十二分钟模拟面试

### 面试官：介绍一下这个项目。

使用第 2.3 节九十秒版本。

### 面试官：这和普通 RAG 有什么不同？

> 普通 RAG 通常固定检索一次；这里模型在每一轮都要在 search 和 answer 间做决策，新的 query 依赖上一轮 observation，并且整条工具轨迹参与结果优化。

### 面试官：你最核心的工作是什么？

> 一是把 Qwen3.5 原生 Tool Calling 接成可训练状态机，保证工具响应只进上下文不进 loss；二是修复搜索成本导致的不搜索坍缩；三是建立逐轨迹 trace、分层门禁和同题配对评测，把协议、检索、模型和训练失败分开。

### 面试官：为什么会不搜索？

> 当同题五条都答错时，EM 都是零，线性搜索成本成为唯一差异。不搜索的错误轨迹相对 reward 最高，GRPO 就会提高它的概率。缩小系数不能根治，因为组内标准化抵消了统一尺度。

### 面试官：结果如何？

> 主模型在 512 题上 EM 从 21.29% 到 39.45%，总搜索减少 21.8%，三个端点的 EM 区间都不跨零。成本分支相对同源控制少 8.18% 搜索，但准确率差异未确认，所以我把它定义为效率改善。

### 面试官：还有什么问题？

> 最大问题是后期 response clipping 和 native action 漂移。后续 20 步在 val 上变好、multihop 变差，说明训练分布特化，而且 B/C clipping 约 71%。下一轮先做格式稳定和完整 checkpoint，再做多 seed 成本实验。

### 面试官：是不是开源项目改的？

> 是，研究范式和底座参考了 Search-R1 与 verl；我没有把它们说成原创。我的改造集中在 native protocol、多轮 token/mask、reward、数据证据、门禁、恢复和 paired evaluation，这些都有对应代码和报告可以展开。

### 面试官：如果上线怎么做？

> 实验版是本地模型和冻结检索器。生产化我会把 Agent orchestrator、model gateway、tool registry 和 search service 拆开，给每次工具调用设置 schema、timeout、budget 和 trace；对 query 做版本化缓存，按任务难度路由模型，并同时监控成功率、P95 延迟、工具次数、invalid 和预算耗尽率。

---

## 18. 数字速查卡

### 18.1 系统配置

| 项目 | 数字 |
|---|---:|
| 模型 | Qwen3.5-2B |
| 训练 | 全参数 FSDP、bf16、两张 5090 级 GPU |
| 方法 | GRPO，group size 5 |
| 检索 | Wiki-2018 BM25 Top-3 |
| 最大真实搜索 | 4 次 |
| 单轮 response / observation | 500 / 500 token |
| 主训练 | 60 steps、2,400 条 rollout |
| B/C | 各 20 steps、各 800 条 rollout |
| 最终评测 | val-128、NQ-test-128、multihop-256 |
| 解码 | greedy、单 rollout、seed 42 |

### 18.2 必背结果

| 对比 | EM | 搜索 | 正确结论 |
|---|---:|---:|---|
| 主模型 vs 基线 | 21.29% → 39.45% | 1690 → 1321，-21.83% | 能力提升明确，部分效率同时改善 |
| 成本 vs 同源控制 | 36.33% → 37.11% | 1320 → 1212，-8.18% | 准确率基本持平，可信成果是少搜索 |
| 四搜饱和 B→C | — | 38.28% → 30.27% | C 主要调浅搜索深度，不是彻底不搜 |
| 早期线性成本 | 17.97% → 7.03% | no-search 98.44% | 典型 reward hacking |

### 18.3 必背限制

- 单训练 seed；
- 每题 greedy 单 rollout；
- 512 行汇总不是自然统一 benchmark；
- R clipping 38.09%，B/C 约 71%；
- B/C 是模型权重同源新 run，不是完整 resume；
- SFT、DAPO、多 seed、hybrid retrieval 都是未来工作。

---

## 19. 面试前最后检查

- [ ] 能在 90 秒内完整介绍项目，并在结果后主动停下。
- [ ] 不主动把项目讲成论文复现，但被问来源时能坦然说明。
- [ ] 能画出 Policy → Parser → Budget → Retriever → Observation → Policy 的状态机。
- [ ] 能解释 observation 为什么进 attention、不进 policy loss。
- [ ] 能区分 requested search、valid action 和 executed search。
- [ ] 能手写旧 reward、correct-only reward 和 GRPO advantage。
- [ ] 能解释为什么调小 cost lambda 不能根治。
- [ ] 能说清主能力结果与成本分支结果的证据强度不同。
- [ ] 能解释 B 多训练 20 步为什么不保证优于 R。
- [ ] 能区分 PPO clip 与 generation clipping。
- [ ] 能主动说明单 seed、greedy、clipping 和 checkpoint 限制。
- [ ] 能用一分钟说明 FSDP、Ray、vLLM 各自角色。
- [ ] 能回答“用了多少开源代码”和“你个人做了什么”。
- [ ] 能把未来工作按“先稳定协议，再确认成本，再扩展检索”排序。
- [ ] 能完成一轮生产化架构追问，不把未来设计说成已实现。

## 20. 最终收束句

如果面试官问“这个项目你最大的收获是什么”，用这段结束：

> 我最大的收获不是把某个指标调高，而是理解了 Agent 训练里模型、工具环境和奖励函数必须共享同一套可审计语义。一次看起来像“模型不会搜索”的失败，可能来自奖励方向、Prompt 协议、parser、检索不可见、Token Mask 或生成截断。这个项目让我建立了一套从状态机、轨迹证据到训练门禁的诊断方法，也让我知道结果不完美时，应该怎样把问题收敛成下一轮可以验证的工程假设。

## 21. 关联材料

- [项目经历量化稿](./Search-R1项目经历-量化稿.md)
- [完整实验流程与事实边界](./Search-R1实验全流程复盘与面试讲稿.md)
- [面试知识体系与学习路线](./Search-R1面试知识体系与学习路线.md)
- [最终 A/R/B/C 统一结果](../docs/qwen35_native_arbc_final_results_analysis.md)
- [B/C 训练恢复、限制与后续建议](../docs/qwen35_native_bc_recovery_complete_analysis_report.md)
- [Qwen3.5 原生工具协议适配](../docs/qwen35_native_tool_adaptation_implementation_report.md)
