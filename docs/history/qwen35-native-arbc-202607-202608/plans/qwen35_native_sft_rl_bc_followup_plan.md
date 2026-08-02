# Qwen3.5 Native SFT → RL → B/C 后续实验方案

## 0. 文档状态与执行顺序

本文只定义未来可能实施的 `task-SFT warm start → outcome-RL → B/C`
实验，不授权现在启动训练，也不改变已经封存的 direct-RL A/R/B/C 结果。

固定顺序为：

```text
direct-RL B/C recovery 与 A/R 三端点评测已经完成并封存
  -> 基于最终结果人工决定是否值得实施本文
  -> 若决定实施，再新增代码、数据合同和独立 AutoDL workflow
```

已完成的 B/C 从已封存的 exact R60 checkpoint 平行启动，最终状态见
[`qwen35_native_complete_experiment_handoff.md`](../final/qwen35_native_complete_experiment_handoff.md)。原执行前合同保留在
[`qwen35_native_bc_posthoc_execution_handoff.md`](qwen35_native_bc_posthoc_execution_handoff.md)。
本文不得用来：

- 替换或回写已完成的 direct-RL B/C；
- 修改旧 R60、G3、B/C attempt 或 evidence；
- 把 G3 capability NO-GO 改写成 GO；
- 把已看过的 held-out 评测轨迹采纳为 SFT 训练数据；
- 在没有新 commit、新 handoff 和新实验身份时直接接续旧 checkout。

本文当前状态是：

```text
design_only=true
execution_authorized=false
implementation_started=false
current_priority=analysis_and_manual_go_no_go
```

## 1. 研究动机与结论边界

Search-R1 原论文的研究重点是：不依赖大规模人工标注的中间搜索轨迹，
只使用最终答案的 outcome reward，让模型在真实检索环境中自主学习何时搜索、
如何改写 query、如何利用 observation 和何时回答。论文同时报告了 SFT 与
rejection-sampling baseline，但没有直接回答“任务专项 SFT 后再做 RL”是否优于
纯 RL。

本项目使用的起点不是纯 Base，而是固定 revision 的、已经经过通用后训练的：

```text
Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc
```

现有 R60 和 G3 已表明：

- R60 确实学到了回答与多轮搜索能力；
- G3 strict EM 为 `151/320 = 47.19%`；
- valid correct multi-search 为 `98/320`；
- clean cost-contrast group 为 `13/64`；
- 但 frozen R60 的 clipping 为 `150/320 = 46.88%`；
- raw invalid-action 为 `140/320 = 43.75%`；
- clean learnable group 只有 `5/64`，能力门因此 NO-GO。

所以本文的 SFT 不是“给模型补事实知识”，而是检验任务专项行为监督能否：

1. 稳定 Qwen3.5 native tool-call/action 边界；
2. 缩短无效 continuation，并提高合法终止率；
3. 让后续短程 RL 更快获得有效 group advantage；
4. 在保留搜索探索和 cost-contrast 的同时降低 clipping/invalid；
5. 为成本奖励 B/C 提供一个更干净的共同 parent。

本实验属于新的 `post_hoc_method_exploration`。即使结果正向，也不能写成原论文
预注册复现的一部分，更不能用它覆盖当前 R60/B/C 的负向或混合结果。

## 2. 研究问题与假设

### 2.1 主要研究问题

`RQ1`：在额外使用 teacher/oracle 数据和 40 次 SFT update 后，`SR20`
相对同样只做 20 次 RL update 的 `R20-control`，是否获得更好的 endpoint 表现与
协议稳定性？

`RQ2`：SFT warm start 是否在降低 clipping/invalid 的同时保留正确多搜能力、
mixed-reward group 和 cost-contrast group？

`RQ3`：在 SFT→RL parent 上，correctness-gated cost reward 是否能相对纯 EM
control 降低搜索成本而不明显损失 strict EM？

### 2.2 预注册假设

- `H1`：在 G3-64×5 上，`SR20` 相比 `R20-control` 的 clipping 和 non-safe
  invalid 各至少下降 10 个百分点，且按 question 聚类 bootstrap 的 95% 区间上界
  小于 0。
- `H2`：在主准确率 endpoint `multihop-256` 上，`SR20 - R20-control` 的
  strict EM question-level paired bootstrap 95% 区间下界不低于 `-3 pp`。
- `H3`：`SR20` 仍满足当前 G3 的正确多搜、learnable-group 和 cost-contrast 门槛。
- `H4`：在主成本 endpoint `multihop-256` 上，`SC20` 相比 `SB20` 的全题
  mean searches 下降、全题 mean utility 提升，且 strict EM 非劣界为 `-3 pp`；
  三项均使用 question-level paired bootstrap 95% 区间判定。
- `H5`：SFT 的收益主要表现为格式、长度、终止和早期 RL 信号改善，而不一定表现为
  SFT checkpoint 本身的最高 EM。

任何假设失败都是完整科学结果，不自动换 seed、延长步数、改变数据或调高
`cost_lambda`。

`H1/H2/H4` 的 bootstrap 只描述固定题集、固定 seed 下的题目抽样不确定性，
不能解释为跨 seed、跨模型或总体训练随机性的显著性。

## 3. 实验树与命名

### 3.1 核心实验

```text
P0 / exact sealed Qwen3.5-2B post-trained parent
├── R20-control / pure-EM GRPO, 20 steps
└── S40 / task-specific SFT, 40 optimizer steps
    └── SR20 / pure-EM GRPO, 20 steps
        ├── SB20 / pure-EM control, 20 steps
        └── SC20 / correct-only cost reward, 20 steps
```

角色定义：

| 名称 | Parent | 训练 | 用途 |
| --- | --- | --- | --- |
| `P0` | 固定 Qwen3.5 revision | 无 | 所有新实验的共同原点 |
| `S40` | `P0` | task-SFT 40 steps | 协议与行为 warm start |
| `R20-control` | `P0` | 纯 EM GRPO 20 steps | 判断 SFT 是否改善短程 RL 的最低对照 |
| `SR20` | `S40` | 纯 EM GRPO 20 steps | 未来 B/C 的候选共同 parent |
| `SB20` | exact `SR20` | 纯 EM GRPO 20 steps | SFT parent 下的奖励 control |
| `SC20` | exact `SR20` | correct-only cost GRPO 20 steps | SFT parent 下的成本分支 |

`SR20` 的 policy、reference、optimizer 和 scheduler 必须按以下方式创建：

- policy 从 exact `S40` HF 权重加载；
- reference 也从 exact `S40` 初始化；
- optimizer/scheduler 全新创建，不能继承 SFT；
- scheduler horizon 固定为 20 steps；
- 不把 reference 错设为 `P0`，否则 KL 会把 policy 拉回 SFT 前状态。

`SB20` 与 `SC20`：

- 必须从同一个 exact `SR20` tree digest 独立启动；
- 各自创建新 optimizer/scheduler；
- 不能让 C 继承 B；
- 除 reward 外共享数据顺序、seed、batch、group、采样、长度和检索环境。

### 3.2 可选的完整 2×2

若希望回答“SFT 是否增强了成本奖励的相对效果”，仅比较 `SB20/SC20`
还不够。高保证版本需增加：

```text
R20-control
├── RB20 / pure EM
└── RC20 / correct-only cost
```

然后比较：

```text
(SC20 - SB20) - (RC20 - RB20)
```

结论边界：

| 已完成组合 | 可以回答 | 不可以回答 |
| --- | --- | --- |
| `S40 + SR20` | SFT 后能否完成短程 RL | SFT 是否优于纯 RL |
| `R20 + S40 + SR20` | SFT 是否改善前 20 步 RL | SFT 是否增强 cost reward |
| `SR20 + SB20 + SC20` | SFT parent 上 C 是否优于 B | SFT 是否改变 C-B 效应 |
| 加 `RB20 + RC20` | SFT 是否改变 cost-reward treatment effect | 多 seed 总体显著性 |

现有 `R60 → B/C` 只能作为历史参考，因为它有 60-step parent、不同训练历程，
且 R60 没有 step-20 checkpoint；它不能替代 `R20-control`。

`R20-control` 与 `SR20` 的比较也不是总数据/总算力匹配实验：`SR20` 额外消费
teacher/oracle 数据与 40 次 SFT update，而且 teacher 的 R60 计算是方法的必要前置。
该比较只能估计“在固定 20-step RL 预算下，增加这套 SFT warm-start package 的效果”。
不得据此声称总体样本效率、总算力效率或从零成本优于纯 RL。若未来要回答这些问题，
必须另设 teacher compute 与训练 token/算力匹配对照；不属于本文核心范围。

## 4. 已完成 B/C 后的决策门

direct-RL B/C recovery 与 A/R 补评已经完成，以下决策前提均已满足：

1. exact B/C parent、checkpoint 和 tree digest；
2. B/C 完整 train trace、resolved config、WandB history 和 lineage；
3. val-128、NQ-test-128、multihop-256 三套逐题结果；
4. 三套 paired summary；
5. clipping、invalid、answer rate、搜索分布和 utility；
6. 原始终态、exit code、evidence marker 和 watchdog receipt。

决策矩阵：

| 已完成 B/C 结果 | 对本文的建议 |
| --- | --- |
| C 少搜、EM 基本不降、协议稳定 | 本文降为可选；优先总结已完成闭环 |
| C 少搜但 EM 明显下降 | 本文有价值，重点检验更干净 parent 是否缓解 trade-off |
| B/C 都受 clipping/invalid 主导 | 本文优先级高，SFT 直接针对当前瓶颈 |
| B/C 协议干净但 cost-contrast 不足 | SFT 可能进一步压缩探索；优先重新审视数据和奖励机会 |
| B/C 是工程/证据失败 | 先修基础设施；不得用 SFT 绕过失败 |
| 磁盘或预算不足 | 只保留本文，不开始实现或训练 |

本轮实际属于“C 少搜、EM 未确认下降，但 B/C 都受 clipping/invalid 主导”。因此 SFT 协议稳定假设有研究价值，但仍需用户单独批准；该结果只能决定“是否执行本文”，不能改变本文 SFT 数据的筛选规则，
也不能把 B/C、G3 或 endpoint 轨迹加入 SFT。

## 5. 冻结项与允许变量

### 5.1 新实验继续冻结

- base model 与 exact revision；
- Qwen3.5 native tool protocol 和 chat template；
- Wiki-18 corpus、BM25 index、top-k 3；
- train-512、val-128、G3-64、NQ-test-128、multihop-256；
- 最多 4 次 executed search 与 answer-only terminal generation；
- response/observation 上限 500；
- RL batch 8、group 5；
- RL temperature/top-p/top-k/min-p/presence/repetition；
- RL learning rate、KL、clip、数据顺序和 seed；
- B/C 的 `cost_lambda=0.10` 与 `correct_only` 定义；
- strict EM 主指标；
- 三套 endpoint 的 group 1、greedy、seed 42。

### 5.2 新增变量

只有以下新因素进入核心实验：

1. 是否先进行 task-SFT；
2. SFT 数据与 assistant-only loss mask；
3. SFT optimizer/scheduler；
4. SFT 后 reference 改为 exact `S40`。

不得同时更换 retriever、prompt、parser、terminal reminder、RL 算法、模型 revision、
训练集配比或成本系数，否则无法把结果归因于 SFT warm start。

## 6. SFT 数据构建合同

### 6.1 可用原料

优先级如下：

1. exact R60 的 train-only 完整轨迹；
2. native-v4 train-512 catalog；
3. 与 catalog 绑定的真实 BM25 retrieval evidence 和 replay；
4. 固定 tokenizer、chat template、tool schema 与 terminal contract。

禁止使用：

- native-v4 val-128；
- G3 probe-64 及其 320 条轨迹；
- NQ-test-128；
- multihop-256；
- 当前或未来 B/C endpoint/train trace；
- 旧 v1/v2/v3 probe 轨迹；
- 人工查看结果后挑出的“看起来漂亮”的评测样本；
- 未绑定 source revision 和 retrieval digest 的外部轨迹。

如果使用 R60 轨迹，实验必须标记为：

```text
data_origin=r60_train_trace_distillation
teacher_compute_included=false_in_training_bill_but_required_for_method
experiment_class=post_hoc_method_exploration
```

这意味着不能宣称该方法从零比 RL60 更省总算力。它是利用已完成 R60 teacher
轨迹进行自蒸馏，再研究 warm start 与后续 RL。

### 6.2 两阶段构建

CPU 构建分为：

```text
audit-only census
  -> 冻结数量、分层和 source-type receipt
  -> deterministic materialize
  -> independent verify/replay
  -> atomic manifest publication
```

`audit-only` 必须先报告：

- 每个 train prompt 的 R60 group 成员；
- strict-correct、clean-correct、clip、invalid、unfinished 数量；
- NQ/comparison/bridge 分层；
- 1/2/3/4 搜分布；
- 可重建 native conversation 的数量；
- observation/retrieval lineage 可复核数量；
- teacher-clean 不足时需要 oracle-action 补齐的数量。

审计完成后才冻结 materialization policy，不能边构建边降低标准。

### 6.3 默认数据规模

建议固定：

| split | NQ | Hotpot comparison | Hotpot bridge | 合计 |
| --- | ---: | ---: | ---: | ---: |
| SFT train | 120 | 35 | 165 | 320 |
| SFT validation | 24 | 7 | 33 | 64 |

这保持当前 train-512 的 `37.5% NQ / 62.5% Hotpot` 总比例。SFT validation
只用于 loss、格式和过拟合诊断，不属于最终科学 endpoint；最终 val-128 仍完全不可见。

若 384 条目标在严格规则下无法构建：

- 先发布完整失败 census/receipt；
- 不静默缩小数据集；
- 不改用评测数据补齐；
- 不自动放宽 clean、evidence 或 token 规则；
- 由人工决定新 contract。

### 6.4 Teacher-clean 过滤

R60 轨迹进入候选必须同时满足：

- source 属于 exact native-v4 train-512；
- strict EM `=1`；
- 有且只有一个合法 final answer；
- `response_clipped=false`，所有 generation event 均未截断；
- raw、environment 和 non-safe invalid 均为 0；
- unfinished generation 为 0；
- 每个 tool call 均合法、query 非空；
- accepted search 与 executed search 一致；
- executed search 数在 `[1,4]`；
- terminal search accepted/executed 均为 0；
- observation policy-token count 为 0；
- `info_mask_consistent=true`；
- query、retrieved docs、visible observation 与封存 evidence 一致；
- NQ gold alias 或 Hotpot supporting evidence 在模型实际可见 observation 中成立；
- 总 token 不超过未来 SFT resolved config；
- tokenizer/chat-template 重放得到相同 token digest。

重复 query、无新增文档、答案已出现后继续搜索和异常长 reasoning 作为独立质量字段。
默认不按“搜索最少”排序选样，而是在 source/category/search-count 内做确定性分层，
避免在 B/C 之前偷偷注入成本偏好。

### 6.5 Oracle-action 补位

teacher-clean 覆盖不足时，可从 train-512 catalog 与真实 retrieval evidence 构建
`oracle-action`，但必须满足：

- NQ query 和 Hotpot 第二跳 query 来自既有、封存的可执行证据合同；
- observation 必须由 exact BM25 evidence 重放，不能手写；
- final answer 使用封存 gold；
- supporting metadata 只用于构建和验证 target，不进入初始 user prompt；
- 不人工编造 chain-of-thought；
- 若 native wrapper 需要占位文本，该文本的 token loss 必须为 0；
- 只监督可验证的 assistant action、tool call 和 final answer span；
- materializer 报告 teacher-clean 与 oracle-action 的数量和比例。

如果 oracle-action 成为多数，最终名称应写成
`oracle-assisted task SFT`，不能仍称为纯 R60 self-distillation。

### 6.6 建议 schema

每条 canonical record 至少包含：

```text
schema
schema_version
record_id
sample_id
data_source
source_split
source_index
category
sft_split
target_origin
messages
assistant_loss_spans
assistant_policy_token_count
masked_environment_token_count
executed_search_count
final_answer
gold_answers
strict_em
retrieval_event_digests
visible_observation_digests
tokenizer_revision
chat_template_sha256
tool_schema_sha256
rendered_input_sha256
loss_mask_sha256
teacher_trace_sha256
source_manifest_sha256
```

同时输出 human-auditable JSONL 和训练用 Parquet。Parquet 不得只保存自由文本
`prompt/response`，必须能复验 native 多轮消息与 token-level mask。

### 6.7 Loss mask

`loss=1`：

- 模型生成的合法 reasoning token；
- 模型生成的 native tool-call wrapper 与 query；
- 模型生成的合法 final answer。

`loss=0`：

- system prompt；
- user question；
- tool schema；
- tool response / retrieved observation；
- terminal reminder；
- synthetic environment action；
- oracle-action 中为了满足模板而插入、但并非 teacher 生成的桥接文本；
- padding。

mask verifier 必须逐条证明：

- tool/observation token 的 policy count 为 0；
- assistant target 至少有一个 token；
- label shift 后 mask 与 token 对齐；
- 无 prompt、gold metadata 或 padding 泄漏；
- token hash 与固定 tokenizer/template 一致。

### 6.8 去重与泄漏

使用规范化 `source_id + question` 双重去重，并对以下集合做全量交叉检查：

```text
SFT train
SFT validation
native-v4 val-128
G3 probe-64
NQ-test-128
multihop-256
current B/C endpoints
```

任何重叠、缺失 source、额外字段、未知 category 或未绑定 evidence 的样本都使构建失败。

## 7. SFT trainer 实现与训练合同

### 7.1 现有代码缺口

当前 bundled SFT 路径不能直接用于生产实验：

- `fsdp_sft_trainer.py` 导入 `SFTDataset`；
- 当前 `verl/utils/dataset/__init__.py` 只导出 RL/RM dataset；
- 默认配置面向普通 `question/answer`；
- 默认 hard-code FlashAttention2，不符合已验证的 Qwen3.5 SDPA 路径；
- `total_training_steps` 的 early-exit 与 scheduler horizon 需要统一；
- 没有 native 多轮 tool-aware mask、lineage 和 evidence 合同；
- checkpoint/evidence 不满足现有 AutoDL 严格封存要求。

实现时应新增专用 dataset/adapter，而不是把 RL Parquet 强行伪装成普通 SFT。

### 7.2 暂定固定配置

| 参数 | 值 |
| --- | ---: |
| model | exact `P0` |
| training | full-parameter FSDP，非 LoRA |
| GPU | 2 |
| precision | bf16 |
| attention | 已验证的 SDPA 路径 |
| gradient checkpointing | on |
| effective batch | 8 |
| micro-batch | 每卡 1，使用 accumulation 达到 effective 8 |
| optimizer | AdamW |
| learning rate | `1e-6` |
| betas | `(0.9, 0.95)` |
| weight decay | `0.01` |
| warmup ratio | `0.10` |
| gradient clip | `1.0` |
| total steps | 40 |
| seed | 42 |
| truncation | error |
| packing | off |
| fixed endpoint | step 40 |

这些值是未来实现的首个固定合同，不是已经验证的最优超参数。任何修改都必须新建
contract/version，不能在一次付费运行中自动 sweep。

`40 steps × effective batch 8 = 320 example exposures`，与默认 SFT train-320
对应一轮。报告同时保存 assistant target token 总数，避免把 SFT step 与
`8 prompts × group 5` 的 RL step 当作同一算力单位。

### 7.3 Smoke 与正式 S40

先运行独立的 2-step SFT smoke，验证：

- 两卡 FSDP/SDPA 路径可运行；
- loss、grad norm、learning rate 全部有限；
- assistant policy token 数大于 0；
- environment/prompt mask 泄漏为 0；
- step-2 checkpoint 相对 P0 内容 digest 已变化；
- 保存/加载后 token logits 和 mask 复验通过；
- 原始 exit code、log、resolved config、WandB offline history 和 checkpoint 完整。

正式 `S40`：

- 必须从 P0 重新启动；
- 不继承 smoke 权重或 optimizer；
- scheduler horizon 从一开始就是 40；
- 可保存 step-20 供诊断，但 fixed parent 始终为 step-40；
- 不因 step-20 看起来更好而事后替换；
- interrupted S40 不自动采用 partial checkpoint。

## 8. R20-control 与 SR20 合同

两条 20-step RL 必须完全对称：

| 项目 | R20-control | SR20 |
| --- | --- | --- |
| policy parent | P0 | S40 |
| reference parent | P0 | S40 |
| optimizer | new | new |
| scheduler | new, horizon 20 | new, horizon 20 |
| reward | strict EM | strict EM |
| train data | same native-v4 train-512 | same |
| consumed order | same 160 prompts | same 160 prompts |
| batch/group | 8/5 | 8/5 |
| rollout seed | same | same |
| sampling | same | same |
| search environment | same | same |
| trace/evidence | same schema | same schema |

`20 steps × 8 prompts × group 5 = 800` 条 train rollout。两条 run 都必须保留：

- 20-step HF checkpoint；
- 完整 800 条 trace；
- per-step loss/KL/entropy/grad norm；
- EM、搜索次数、answer/clip/invalid；
- mixed/all-wrong/all-correct group；
- non-zero advantage trajectory/token ratio；
- parent/reference/checkpoint digests；
- resolved config、WandB history 和 original exit code。

## 9. 门禁

### 9.1 S40 → RL admission

S40 先通过工程门，不要求它已经达到最终 RL 能力：

- checkpoint/tree/tokenizer/template digest 完整；
- SFT loss 与 grad norm 全程有限；
- assistant-only mask replay 100% 一致；
- G0 模板和 direct/manager action boundary 一致；
- G1 tool-call、tool-response、terminal hard boundary 全部通过；
- 无 observation token 进入 loss；
- 固定 behavioral probe 满足下面的数值门槛。

S40 admission probe 固定为 CPU handoff 中封存的 exact
`probe_autonomous_16.parquet`：

```text
questions=16
group=1
greedy=true
seed=42
max_executed_searches=4
response_tokens=500
observation_tokens=500
```

同一 experiment contract 必须先用 P0 跑出 exact baseline receipt，再用 S40 跑相同
manifest/config；P0 probe 可放在 Stage 1 的 SFT smoke 后或 Stage 2 core 的付费
prelude，S40 probe 则在 S40 完成后运行，两者最终一起进入 core evidence。不允许借用
不同合同的旧 G1 聚合值。S40 必须同时满足：

| 指标 | 门槛 |
| --- | ---: |
| legal answer count | `>= max(8, P0_answer_count - 2)` |
| at-least-one-search count | `>= max(8, P0_search_count - 2)` |
| zero-search count | `<= min(8, P0_zero_search_count + 2)` |
| clipped count | `<= P0_clipped_count + 1` |
| non-safe invalid count | `<= P0_non_safe_invalid_count + 1` |
| terminal accepted/executed search | `0/0` |

probe manifest、P0 receipt、S40 receipt、生成参数和逐题结果都进入 SFT-smoke evidence。
任一门槛失败即不启动 R20/SR20；不得人工解释“看起来没有坍缩”后放行。

工程门失败不启动 R20/SR20。

### 9.2 R20/SR20 比较

R20 和 SR20 完成后，使用相同 held-out G3-64、每题 group 5，共 320 条，
分别报告：

- strict EM；
- valid correct multi-search；
- coverage；
- clean learnable group；
- clean cost-contrast group；
- clipping；
- raw invalid 与 non-safe invalid；
- answer/terminal rate；
- 0/1/2/3/4 搜分布；
- `E[searches | correct]`；
- mixed/all-wrong/all-correct group；
- response/trajectory length。

### 9.3 SR20 → B/C authorization

只有 SR20 同时满足以下条件才允许 SB20/SC20：

| 条件 | 门槛 |
| --- | ---: |
| valid correct multi-search | `>=16/320` |
| correct multi-search coverage | `>=8/64` |
| clean learnable group | `>=8/64` |
| clean cost-contrast group | `>=8/64` |
| clipped ratio | `<=5%` |
| raw invalid-action ratio | `<=5%` |
| val-128 EM 相对 R20 | 不下降超过 3 pp |
| G3 zero-search ratio | `<=10%`，即 `<=32/320` |

任何一项失败：

- 以 `exit-code=0` 封存完整科学 NO-GO；
- 不自动启动 B/C；
- 不换 seed、不选择 S20、不增加 RL steps；
- 不放宽 strict EM、clip/invalid 或 clean 定义。

## 10. SB20/SC20 奖励合同

设：

```text
c = executed_searches / 4
```

奖励固定为：

```text
SB20: r = EM
SC20: r = EM * (1 - 0.10 * c)
```

统一评测 utility：

```text
u = EM - 0.10 * c
```

SC20 答错轨迹仍为 0，不允许恢复旧 linear-all-trajectory 成本惩罚，也不提高
`lambda`。B/C 除奖励外必须完全对称并串行执行；每条分支独占两张 GPU。

## 11. 评测与统计

### 11.1 固定 endpoint

至少评测：

```text
curated val-128
NQ-test-128
multihop-256
```

endpoint 全部使用：

```text
group=1
greedy=true
seed=42
```

核心阶段至少评测 `S40`、`R20`、`SR20`；分支阶段评测 `SB20`、`SC20`。
P0 与现有 R60/B/C 只在 exact 数据/协议/digest 可证明一致时作为历史列展示，
不能混入新配对统计。

主 endpoint 与判定单位固定为：

| 问题 | 主 endpoint | 聚类/配对单位 |
| --- | --- | --- |
| H1 协议稳定性 | G3-64×5 | question，共 64 个 cluster |
| H2 warm-start accuracy | multihop-256 | question，共 256 对 |
| H4 cost-reward effect | multihop-256 | question，共 256 对 |

val-128 与 NQ-test-128 是预注册 secondary endpoint；必须完整报告，但不替代主
endpoint。G3 的 320 条 trajectory 不能被当作 320 个独立样本。

### 11.2 必报指标

每个 endpoint 报告：

| 指标 | 含义 |
| --- | --- |
| strict EM | 主正确率 |
| mean executed searches | 总体成本 |
| `E[searches | correct]` | 正确集合内成本，仅作描述 |
| 0/1/2/3/4 搜分布 | 策略形态 |
| no-search ratio | 是否坍缩 |
| clipped ratio | 长度稳定性 |
| raw / non-safe invalid ratio | 协议稳定性 |
| answer rate | 合法终止 |
| mean response/trajectory tokens | 冗长程度 |
| utility | `EM - 0.10 * searches/4` |

训练阶段另外报告：

- train EM/reward 分离；
- PG loss、reference-KL surrogate、entropy、grad norm；
- mixed/all-wrong/all-correct group；
- non-zero advantage trajectory/token ratio；
- NQ/Hotpot 分层；
- terminal reminder 与越界 search；
- SFT assistant-token NLL 和 mask counts。

### 11.3 配对分析

保存逐题：

- 两模型答案与 strict EM；
- executed search；
-完整 trajectory；
- B 对/C 错、B 错/C 对、共同正确、共同错误；
- search transition；
- “少搜且同样正确”的代表性轨迹；
- “少搜但失去正确性”的 trade-off 轨迹。

使用 paired bootstrap 给出 EM、搜索次数和 utility 差值的 95% 区间；
小样本、单 seed 结果仍以效应量和逐题证据为主，不把区间包装成多 seed 总体结论。

bootstrap 固定按 question 重采样 10,000 次并使用固定分析 seed；同一 question 的
全部 group member 必须一起重采样。H4 的主要判据为：

```text
upper_CI(delta_all_question_mean_searches) < 0
lower_CI(delta_all_question_mean_utility) > 0
lower_CI(delta_strict_EM) >= -0.03
```

其中 `delta = SC20 - SB20`。H2 的主要判据为：

```text
lower_CI(SR20_EM - R20_EM) >= -0.03
```

`E[searches | correct]` 条件化于训练后正确集合，会受到组成变化影响，因此不能作为
主要提效判据。报告必须另外拆出：

- 全题 paired mean-search 差；
- 共同正确题上的 paired search 差；
- `B correct → C wrong` 与 `B wrong → C correct`；
- 因正确集合变化产生的 composition effect；
- 全题 utility 差。

## 12. 结果判读

### 12.1 SFT warm start

| 结果 | 解释 |
| --- | --- |
| SR20 EM 保持、clip/invalid 下降 | warm start 正向 |
| SR20 格式改善但 EM/多搜下降 | 行为克隆过强或探索受损 |
| SR20 与 R20 接近 | SFT 主要没有增加短程 RL 价值 |
| S40 好、SR20 变差 | RL 配置或 reference/KL 需诊断，但不自动调参 |
| S40/SR20 都失败 | 数据/格式监督不充分或实现错误，先区分工程与科学失败 |

### 12.2 SFT parent 上的 B/C

| 结果 | 解释 |
| --- | --- |
| C 少搜且 EM 基本不降 | 成本奖励正向 |
| C 少搜但 EM 明显下降 | cost-quality trade-off |
| C 搜索/EM 都无改善 | 探索性负结果 |
| C no-search collapse | 奖励或 group 信号失败 |
| 两者协议错误重新升高 | parent 稳定性没有延续到 branch RL |

只有完成可选 `RB20/RC20` 后，才可讨论 SFT 是否改变了 C-B treatment effect。

## 13. 预期实现切片

文件名是预案，真正实施前检查编号是否已被占用：

```text
scripts/data_process/qwen_native_sft.py
verl/utils/dataset/qwen_native_sft_dataset.py
verl/trainer/qwen_native_sft_trainer.py
verl/trainer/config/qwen_native_sft_trainer.yaml
scripts/autodl/NN_gpu_qwen_native_sft_rl.sh
scripts/autodl/04_watch_and_shutdown.sh
scripts/autodl/README.md
scripts/autodl/tests/test_qwen_native_sft_pipeline.sh
scripts/autodl/tests/test_qwen_native_sft_data.py
scripts/autodl/tests/test_qwen_native_sft_trainer.py
scripts/autodl/tests/test_shutdown_watchdog.sh
scripts/autodl/cloud-audit-policy.json
```

编号与复用逻辑：

- `12_gpu_qwen_native_bc_only.sh`、`13_gpu_qwen_native_bc_recovery.sh`、`14_gpu_qwen_native_ar_eval_only.sh` 与 `15_watch_qwen_native_ar_eval_only.sh` 都属于已完成 direct-RL 实验；
- 原草案预留的 `13_gpu_qwen_native_sft_rl.sh` 编号已经被 recovery 占用，真正实施时必须重新分配脚本编号、namespace 和 contract，不能覆盖 `13/14/15`；
- 不大改 `09`；
- 复用现有 run job、trace、paired eval、evidence 和 watchdog 组件；
- 但现有 watchdog 不认识未来 namespace，必须在新 checkout 中为每个新 contract
  增加 exact-attempt/evidence verifier，不能假设旧 failure path 可以安全代替；
- 新合同使用独立 namespace，不能更新旧 `latest-main` 或 B/C marker。

## 14. Git 与 CPU 阶段

### 14.1 Git

真正实施时：

1. 保持当前用户未跟踪文件不被删除或顺手提交；
2. 在新 commit 实现 data/trainer/runner/tests/docs；
3. 运行完整本地测试；
4. 提交并 push 当前实验分支；
5. 使用 40 位 commit 在新的 experiment root 创建 clean detached checkout；
6. 新 commit 必然使旧 CPU handoff 对本实验失效。

新旧目录合同固定为：

```text
legacy evidence root (read-only):
  /root/autodl-tmp/search-r1

new SFT/RL experiment root:
  /root/autodl-tmp/search-r1-sft-rl

new clean checkout:
  /root/autodl-tmp/search-r1-sft-rl/checkout
```

不得对旧 `/root/autodl-tmp/search-r1/checkout` 运行现有 `01_git.sh`、pull、
checkout、rename 或零散覆盖；现有脚本遇到不同 commit 本来也应拒绝。未来实现应让
bootstrap/init 显式接受并绑定新的 project root，或为新 root 提供等价的安全初始化，
而不是原子切换旧 canonical checkout。旧 root 只作为 exact R60、数据、模型和
retrieval evidence 的只读 provenance source；任何复用都必须记录绝对路径和 digest。

### 14.2 CPU acquire 与 offline finalize

现有 Qwen native incremental 是离线合同，不能把“必要时联网取得 R60 trace”
混入同一个模式。未来 CPU 阶段必须拆成两个有独立 receipt 的子阶段：

```text
cpu-acquire (network allowed)
  -> acquisition seal
  -> network disabled
  -> cpu-finalize (offline, CUDA hidden)
  -> complete CPU handoff
```

如果 exact R60 trace、native-v4 evidence 和全部依赖已在同一持久盘且 digest
通过，`cpu-acquire` 只验证并封存引用，不发生下载。若确实缺失：

- 只能使用预先登记的 URI/source marker、revision、bytes 和 SHA-256；
- 下载到临时文件，传输成功后校验 digest/schema，再原子发布 acquisition seal；
- 失败文件不得占据 canonical source path；
- 不允许从 docs 摘要、`latest` 或未固定远程目录重建；
- credential 不进入命令参数、日志、env receipt 或结果归档。

离线 finalize 建议使用显式模式：

```bash
AUTODL_QWEN_NATIVE_SFT_INCREMENTAL=1 \
bash /root/autodl-tmp/search-r1-sft-rl/checkout/scripts/autodl/02_cpu_prepare.sh
```

该模式只消费 acquisition seal，不联网、不安装、不下载。CPU finalize 应：

1. 校验 P0、R60 train trace、native-v4 data、catalog、retrieval evidence 和 tokenizer；
2. 运行 audit-only census；
3. 物化 SFT train/validation；
4. 独立重放消息、token、mask 和 retrieval digest；
5. 运行数据泄漏检查；
6. 组合 SFT smoke/S40/R20/SR20/SB20/SC20 resolved config；
7. 运行 Python/shell tests、compileall 和配置校验；
8. 发布新的 self-hashed CPU handoff。

SFT 数据必须先写入 attempt-scoped staging，验证成功后发布到新的不可变目录，例如：

```text
/root/autodl-tmp/search-r1-sft-rl/data/qwen35_native_sft_v1/<dataset-digest>/
```

不得原地修改旧 `search_mix_qwen35_native_v4`、R60 trace 或 canonical evidence。

CPU handoff 至少绑定：

- `schema=4` 与 exact experiment contract version；
- full Git commit；
- runner SHA；
- dependency freeze；
- P0 revision/tree digest；
- R60 evidence marker 与 train trace digest；
- native-v4 manifest；
- catalog/evidence/replay digests；
- SFT census、curation ledger、Parquet/JSONL digests；
- tokenizer/chat template/tool schema；
- 全部 resolved config；
- 测试清单与结果。

schema 4 verifier 必须严格拒绝 legacy schema、缺字段、未知字段、非 canonical manifest、
symlink/path escape 和语义校验未通过的数据；仅有文件 SHA-256 不足以证明 SFT
schema、mask 和 provenance 正确。

CPU acquisition seal、final handoff 和 shutdown capability 还必须记录由操作者从
AutoDL 控制台确认的 provider volume ID。guest 脚本不能自行证明 provider volume
identity，因此 CPU 完成后必须：

1. 持久化 volume ID 与 mount evidence；
2. 停止/卸载 CPU 实例；
3. 明确确认 CPU 不再写盘；
4. 才允许把同一 volume 挂到 GPU 实例。

禁止 CPU/GPU 两个主机并发挂载写入；单机 `flock` 不能防止双主机同时写盘。

## 15. GPU 阶段

### 15.1 阶段拆分

建议按付费风险分三次人工批准：

```text
Stage 1: sft_smoke
Stage 2: core = S40 + R20-control + SR20 + comparison gate
Stage 3: branches = SB20 + SC20 + endpoints + paired summaries
Optional Stage 4: RB20 + RC20 2×2
```

每一阶段都消费 exact predecessor marker，不能用 `latest` 猜测。

未来接口草案：

```bash
QWEN_NATIVE_SFT_RL_STAGE=sft_smoke \
GPU_COUNT=2 AUTODL_PRICE_PER_HOUR=<price> \
bash /root/autodl-tmp/search-r1-sft-rl/checkout/scripts/autodl/NN_gpu_qwen_native_sft_rl.sh
```

```bash
QWEN_NATIVE_SFT_RL_STAGE=core \
QWEN_NATIVE_SFT_SMOKE_EVIDENCE=<exact-marker> \
GPU_COUNT=2 AUTODL_PRICE_PER_HOUR=<price> \
bash /root/autodl-tmp/search-r1-sft-rl/checkout/scripts/autodl/NN_gpu_qwen_native_sft_rl.sh
```

```bash
QWEN_NATIVE_SFT_RL_STAGE=branches \
QWEN_NATIVE_SFT_RL_CORE_EVIDENCE=<exact-marker> \
GPU_COUNT=2 AUTODL_PRICE_PER_HOUR=<price> \
bash /root/autodl-tmp/search-r1-sft-rl/checkout/scripts/autodl/NN_gpu_qwen_native_sft_rl.sh
```

以上命令是设计草案，当前不存在、不得执行。

### 15.2 Admission

GPU 在任何模型加载或付费编译前必须验证：

- handoff/capability 中记录的 provider volume ID 与操作者本次确认完全一致；
- CPU instance 已停止/卸载且没有第二主机写入同一 volume；
- same persistent mount source、filesystem identity 与 project root；
- clean detached exact commit；
- complete CPU handoff；
- offline mode；
- 两张预期 GPU；
- 磁盘余量；
- P0/R60/data/SFT/config digest；
- predecessor marker；
- global lock；
- runner 与 watchdog contract。

GPU 不能 pip/apt、下载模型/数据、重建 SFT 或 reseal CPU handoff。

### 15.3 Detached execution

入口只表示后台 worker 已被提交，不表示阶段成功。实现需继续使用：

```text
nohup + setsid
exact attempt directory
global flock
persistent log
original exit code
atomic terminal marker
```

启动后立即记录 exact attempt，并将该路径绑定 watchdog；不得从 `latest`、
PID 或时间戳猜测。

### 15.4 Watchdog admission

未来 `04_watch_and_shutdown.sh` 必须显式支持：

```text
qwen-native-sft-smoke-v1
qwen-native-sft-rl-core-v1
qwen-native-sft-rl-bc-v1
qwen-native-sft-rl-2x2-v1
```

每个 contract 都要有独立 exact-attempt verifier，验证对应 marker、evidence、
lineage、checkpoint、trace、paired output、original exit code 和终态。runner 在
watchdog 尚未支持该 contract 时必须拒绝启动付费 stage；不能再依赖“不认识成功合同，
故意走 failure path”作为常规方案。

测试必须覆盖 success、scientific NO-GO、engineering failure、tampered evidence、
lock conflict、sync failure、backend failure 和 test hook。README 必须给出 exact
attempt 的绑定命令。静态 audit policy 也要登记新入口、watchdog owner、shutdown
backend 和 test-hook token。

## 16. Evidence、lineage 与 marker

建议 namespace：

```text
qwen-native-sft-smoke
qwen-native-sft40
qwen-native-rl20-control
qwen-native-sft-rl20
qwen-native-sft-rl-core
qwen-native-sft-rl-bc
qwen-native-sft-rl-2x2
```

每个训练角色的 lineage 至少包含：

```text
stage
role
run_dir
parent_checkpoint
parent_checkpoint_digest
reference_checkpoint
reference_checkpoint_digest
checkpoint
checkpoint_digest
optimizer_origin
scheduler_horizon
checkout_commit
cpu_handoff_digest
data_manifest_sha256
sft_manifest_sha256
resolved_config_sha256
trace_sha256
trace_manifest_sha256
run_contract_sha256
predecessor_evidence_sha256
```

SFT 还需保存：

- `sft-census.json`；
- `curation-ledger.jsonl`；
- train/validation JSONL 与 Parquet；
- token/mask replay receipt；
- per-step training metrics；
- step-20 diagnostic 与 fixed step-40 checkpoint；
- teacher/oracle source breakdown；
- exact rejected counts/reasons。

evidence marker 只有在全部期望文件、基数、schema、digest 和 lineage 验证成功后原子发布。
旧 attempt 始终只读；失败 attempt 不能改写成成功。每个 marker schema 名称必须与
watchdog verifier 使用的 contract 完全一致，不能只靠 namespace 前缀匹配。

## 17. Retry、恢复与科学失败

- 不自动 retry OOM、NaN、schema mismatch、hash mismatch 或科学 NO-GO；
- retry 必须由用户检查原因后显式授权；
- 每次 retry 创建新 attempt，不覆盖旧目录；
- 复用成功 predecessor 前重新校验全部 identity；
- interrupted stage 默认从其固定 parent 重新开始；
- partial checkpoint 不自动采用；
- 只有 final checkpoint 已完整原子发布、后续封存失败时，才允许独立 verifier
  生成新的 synthetic adoption attempt；
- adoption 不得用于科学 gate 失败或不完整 SFT 数据。

完整负结果使用 `exit-code=0 + scientific_decision=NO-GO`。工程失败保留原始非零
exit code，不用换 seed 或调参伪装成新科学尝试。

## 18. Watchdog 与关机

每个 GPU phase 的 success 或 failure 都只能在以下顺序后请求关机：

```text
训练/评测终态
  -> 原始 exit code
  -> evidence/terminal marker
  -> durable sync
  -> shutdown-safe
  -> 重新验证 mount/path/commit/lock/state
  -> shutdown-requested
  -> AutoDL /usr/bin/shutdown
```

以下情况必须保持实例运行：

- lock conflict；
- terminal/evidence 无法持久化；
- sync 失败；
- handoff、commit、mount、path 或 digest 不一致；
- `--keep-running`、dry-run 或 test mode；
- watchdog 无法确认 exact attempt；
- watchdog 不认识 result contract 或缺少对应 evidence verifier；
- shutdown authorization 失败。

自动化测试永远不能调用真实 shutdown backend。`shutdown-dispatched` 或 SSH 断开
也不等于控制面已停止；用户仍需在 AutoDL 控制台确认实例和计费终止。

## 19. 测试矩阵

### 19.1 数据与 trainer

- canonical message 重建；
- Qwen native tool-call/tool-response round trip；
- assistant-only loss mask；
- label shift 对齐；
- observation/prompt/terminal token 零 loss；
- 空 assistant span 拒绝；
- clip/invalid/unfinished teacher trace 拒绝；
- accepted/executed search 不一致拒绝；
- retrieval/evidence digest tamper 拒绝；
- overlength/truncation 拒绝；
- split overlap 与 normalized-question overlap 拒绝；
- unknown field/category/source 拒绝；
- census 与 materialization 数量一致；
- 40-step scheduler horizon 与 early exit 一致；
- checkpoint save/load 与 tree digest；
- SFT reference 正确传给 SR20。

### 19.2 Shell/AutoDL

- `bash -n` 所有 touched shell；
- success、nonzero、INT、TERM；
- lock conflict；
- missing/tampered predecessor；
- wrong commit/handoff/model/data/config；
- CPU/GPU phase 边界；
- GPU offline；
- exact attempt 与 marker；
- evidence 原子发布失败；
- retry/adoption；
- keep-running、dry-run、test mode；
- watchdog success/failure；
- unknown watchdog contract 拒绝；
- 每个新 contract 的 exact evidence verifier；
- backend failure 不覆盖原始 work exit code；
- 测试 hook 不触及生产 `/root/autodl-tmp`。

### 19.3 最终本地验证

```text
focused pytest
relevant full pytest
compileall
bash -n
production config composition
git diff --check
executable mode check
AutoDL static audit helper
```

只有本地测试通过、commit 已 push、远端 40 位 SHA 可解析后才进入 CPU。

## 20. 时间、费用与磁盘

### 20.1 时间与费用

根据 R60 60 步约 `9:22:07`，单个 RL20 粗估约 3.1 小时。核心加分支至少包含：

```text
R20-control
SR20
SB20
SC20
```

仅四个 RL20 就约 12.4 小时；另有 SFT40、G3、多个 endpoint、证据校验和启动开销。
因此：

- SFT40 时间必须先由 2-step smoke 实测，不预先假装与 RL step 等价；
- core + branches 的初步预算建议按 `80-120 元` 规划；
- 可选 `RB20/RC20` 2×2 另预留约 `40-60 元`；
- 每阶段应设置独立硬上限，但硬上限不能写成预计账单；
- 价格以启动时 AutoDL 控制台整机价为准。

### 20.2 磁盘

单个 full-parameter checkpoint 约 9.56 GB。核心与分支至少新增：

```text
S40
R20-control
SR20
SB20
SC20
```

仅五个 checkpoint 约 47.8 GB，尚未包含：

- optional step-20 diagnostic；
- optional RB20/RC20；
- train/eval traces；
- WandB offline history；
- paired results；
- attempt 和临时目录。

`65 GB / 85 GB` 只能作为早期规划下限，不能直接作为 GPU admission。真正门禁必须
由 SFT smoke 输出实测：

```text
checkpoint_bytes
checkpoint_atomic_publish_peak_bytes
trace_bytes_per_step
wandb_and_evidence_bytes
failed_attempt_retention_bytes
terminal_reserve_bytes
```

每个后续 stage 在模型加载前动态计算：

```text
required_bytes =
  remaining_role_checkpoint_bytes
  + current_stage_atomic_publish_peak
  + projected_trace_and_evidence_bytes
  + failed_attempt_retention_allowance
  + non-consumable_terminal_reserve
```

并使用 byte-level filesystem free-space 检查，而不是只解析人类可读 `df -h`。
建议：

- core + branches 的初步容量规划仍按至少 65 GB；
- 完整 2×2 的初步容量规划仍按至少 85 GB；
- 但实测动态门禁优先，任何估算下限都不能覆盖它；
- 启动任何新实验前必须重新运行 `df -h /root/autodl-tmp` 并做 byte-level 容量门禁；
- 不自动删除 R60、已完成 B/C、smoke、G3 或失败 attempt；
- 空间不足时报告精确目录与大小，由用户选择扩盘或校验后外部归档。

最后一次 B/C 复核时 150 GB 持久盘只剩约 19 GB，之后又完成了 A/R 评测；因此本文不能默认在同一布局上直接执行，必须重新审计并由用户决定归档或扩盘。

## 21. 实施顺序

只有用户基于已经封存的 B/C 与 A/R 结果明确批准，才按以下顺序推进：

1. 复核已完成 B/C 三套 endpoint、A/R 对照和失败模式；
2. 形成“实施/不实施本文”的书面 decision；
3. audit R60 train trace 与 native-v4 evidence；
4. 冻结 SFT 数据 contract/version；
5. 实现 materializer、dataset、trainer 和测试；
6. 实现新的 AutoDL CPU/GPU contract；
7. 本地完整验证；
8. commit/push；
9. CPU incremental prepare 与 handoff；
10. SFT 2-step smoke；
11. 人工批准 core；
12. S40、R20、SR20 和 gate；
13. 只有 gate GO 才人工批准 SB20/SC20；
14. 三套 endpoint、paired summary 和 evidence seal；
15. 可选决定是否补 RB20/RC20；
16. watchdog 安全关机并在控制台确认停止计费；
17. 归档完整正向或负向结果。

## 22. 最终成功标准

工程成功要求：

- 数据、mask、trainer、checkpoint、trace、lineage 和 evidence 全部可复算；
- CPU/GPU 边界、exact commit 和 persistent volume 可验证；
- 原始 exit code 与终态完整；
- 自动关机顺序安全；
- 当前 R60/B/C 与新实验互不污染。

科学正向结果的最低标准：

1. `SR20` 相比 `R20-control` 达到 H1 的 question-clustered 协议稳定性判据；
2. `SR20` 保持 strict EM 和正确多搜能力；
3. `SR20` 仍有足够 learnable/cost-contrast group；
4. `SC20` 相比 `SB20` 降低全题 mean searches，并提升全题 mean utility；
5. `SC20` strict EM 满足预注册的 `-3 pp` 非劣界；
6. 三套 endpoint 方向一致，而不是只挑最好的一套。

如果只改善格式而不改善 RL/B/C，结论应写成：

> task-SFT 提高了 native tool protocol 的稳定性，但没有证明其能改善短程
> outcome-RL 或成本奖励效果。

如果 SFT 降低探索或 cost-contrast，结论应写成：

> 行为克隆缩小了策略分布与组内差异，使 GRPO 和成本奖励的可学习信号变弱。

无论结果如何，保留完整数据构建 ledger、训练轨迹、checkpoint、指标和 lineage，
不得把探索性负结果包装成显著提升。

## 23. 关联资料

- [`qwen35_native_bc_posthoc_execution_handoff.md`](qwen35_native_bc_posthoc_execution_handoff.md)
- [`qwen35_native_r60_training_and_trajectory_analysis.md`](../stages/qwen35_native_r60_training_and_trajectory_analysis.md)
- [`qwen35_native_r60_g3_evaluation_and_trajectory_analysis.md`](../stages/qwen35_native_r60_g3_evaluation_and_trajectory_analysis.md)
- [`autodl_search_r1_reproduction_plan.md`](autodl_search_r1_reproduction_plan.md)
- [`../scripts/autodl/README.md`](../../../../scripts/autodl/README.md)
- [Search-R1 paper](https://arxiv.org/abs/2503.09516)
- [Search-R1 empirical study](https://arxiv.org/abs/2505.15117)
