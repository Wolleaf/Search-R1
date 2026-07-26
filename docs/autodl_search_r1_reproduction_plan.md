# Search-R1 小规模成本感知复现完整方案（AutoDL）

> 本文件是后续实现和云端运行的唯一规划依据。旧实验的配置、日志和负结果继续原样保留在 `docs/results/` 及结果分析文档中，不用新参数覆盖或改写。

## 1. 目标、现状与边界

目标是用 Qwen3.5-2B 复现 Search-R1 的核心闭环：模型生成 `<search>`，读取 BM25 检索结果后继续推理，再通过 GRPO 和答案 EM 学习搜索策略；在模型已具备正确多搜能力后，比较原奖励与成本感知奖励。

此前 NQ-only、response 256 实验已经完整跑通工程链路，但多跳机会门禁得到 61/256 EM、248 题只搜一次；仅有的 8 条二搜全部截断且答错，多数重复 query。因此旧结果作为失败经验保留，不能继续从“几乎只搜一次”的策略直接优化成本。

本轮只解决两个已观测问题：

1. 将单次生成上限从 256 调为论文使用的 500，消除 11.72% 轨迹发生截断的干扰。
2. 用实际 Wiki-18 BM25 top-3 结果筛选一搜充分题和可执行的二搜证据链，再以训练前探针决定是否值得重训。

本项目仍是面向 Agent 算法岗位的缩小复现，不宣称复刻论文数值。固定两张 32 GiB 5090 级 GPU、单 seed、CPU BM25 和全参数微调；不做 PPO、dense retriever、模型/超参数 sweep、多 seed 或通用实验平台。

## 2. 固定搜索与长度配置

论文最大 action budget 为 4，默认返回 top-3 passages。本项目保持：

- 最多 4 个允许 `<search>` 的交互轮次；第 4 轮后仍未结束时，按上游源码再生成一次关闭检索执行的收尾动作。
- `n_search` 只统计真正发给检索服务的请求，范围为 `[0, 4]`。
- `max_response_length=500` 是每次生成的上限，不是整条轨迹的总长度。
- Qwen adapter 的右侧轨迹容量固定为 `max_prompt_length=4500`，完整保留上游收尾生成及其 policy token。

四次 response、四段 observation 和最终回答的右侧理论上限为：

```text
4 * (500 + 500) + 500 = 4500
```

其中前四轮允许检索，最后 500 token 仅供仍 active 的轨迹收尾；该轮若再次生成 search，也不执行检索、不增加成本。`max_action_budget` 仍为 4，完整耗尽预算的轨迹在日志中记为 `action_count=5` 和 `terminal_generation=true`。当前固定参数为：

| 参数 | 值 |
| --- | ---: |
| `max_start_length` | 1024 |
| `max_response_length` | 500 |
| `max_obs_length` | 500 |
| `max_prompt_length` | 4500 |
| `max_turns` / retriever top-k | 4 / 3 |
| train batch / GRPO group | 8 / 5 |
| temperature / top-p | 1.0 / 1.0 |
| learning rate / warmup | `1e-6` / 0.285 |
| KL coefficient | 0.001 |

保持 HF rollout、SDPA、bf16、retrieved-token loss masking、gradient checkpointing，以及 parameter/gradient/optimizer CPU offload。两卡下 actor、rollout 和 reference 的实际每卡 micro-batch 已为 1，不预先降低 batch 或 group。

两步 smoke 固定验证 `trajectory/response/observation=4500/500/500`。若第二次 backward 明确 OOM，保留失败 attempt 并停止；任何 response、batch 或 group 降配都必须另立实验身份，不能在当前配置下静默回退。历史 `05/06` 入口继续显式锁定 256，仅用于解释已归档实验。

## 3. 检索验证混合数据

源数据固定为 `RUC-NLPIR/FlashRAG_datasets@bcafb8dd07d453be3cbeeeb3f78be1841bddf92c`：

| 文件 | bytes | SHA-256 |
| --- | ---: | --- |
| `nq/train.jsonl` | 9,960,189 | `572685d3f384d9c37479b1fb14232984c85ff13b58923c0c9442232bd5d5647b` |
| `hotpotqa/train.jsonl` | 569,520,788 | `a81274abafa899ec0ee073102edbe6bb694a8a1174201b4e43e2bc6c98964d1a` |

固定组成：

| split | NQ 一搜充分 | Hotpot comparison | Hotpot bridge | 总数 |
| --- | ---: | ---: | ---: | ---: |
| train | 192 | 56 | 264 | 512 |
| val | 64 | 16 | 48 | 128 |

val 中 64 道 Hotpot 题同时作为训练前及 R-mix 后的 held-out probe，不进入训练。现有 HotpotQA/2Wiki dev-256 继续作为独立最终多跳评测，不参与筛选或训练。

训练集固定为 37.5% NQ 与 62.5% Hotpot，而不是原方案的 1:1。NQ 仍提供稳定的“一搜后作答”格式奖励，但降到 192 题，避免容易答对的单跳样本再次主导 GRPO；320 道 Hotpot 则明确增加二搜策略的学习机会。首轮完整 CPU 漏斗中，comparison 的 4,159 个 prescreen 候选已全部查询，2,012 个通过结构检查，但在实际 384-token observation 中同时满足两跳 supporting fact 和“第一跳无答案、第二跳有答案”的只有 74 题；bridge 在同样严格规则下有 766 题。因为 comparison 没有被 candidate cap 截断，提高 cap 无效；本轮不放宽证据可见性，而是在成功构建前显式重登记为 train `56/264`、val `16/48`，使用 72/74 个严格合格 comparison，并由更能训练“从第一轮 observation 提取第二 query”的 bridge 承担缺口。按总比例预期 60 steps 会看到约 180 个 NQ prompt（900 条轨迹）与 300 个 Hotpot prompt（1500 条轨迹），足以兼顾动作格式与多搜信号；不进一步降到 25% NQ，以免训练早期出现过多全错 Hotpot group。

训练 DataLoader 使用固定 seed、全局 shuffle、无放回并 `drop_last=True`。`batch 8 × 60 steps = 480`，即实际消费 512 题中前 480 题（93.75%），留下 32 题未见；不为凑满一轮把训练改成 64 steps，也不做配比 sweep。每个 batch 只在期望上约为 3 个 NQ + 5 个 Hotpot，不强行做分层采样。

### 3.1 一搜充分标准

NQ 候选用原问题查询相同 Wiki-18 BM25 top-3。模型实际可见的 384-token observation 中必须出现至少一个规范化 gold alias；优先选择 top-1 即命中的样本。排除答案已在问题中、空答案、`yes/no`、过短歧义答案以及跨 split 重复问题。

### 3.2 多搜证据链标准

HotpotQA 候选必须恰好有两个不同 supporting titles，并满足：

1. 原问题的第一次 top-3 只覆盖其中一个 supporting title，不能已经覆盖两个。
2. 第二个 oracle query 是缺失 title；它必须返回新文档并命中第二个 supporting title。
3. comparison 的两个实体应能从问题中获得；bridge 的缺失 title 必须出现在第一次实际可见的 observation 中。
4. 每个 title 的全部 annotated supporting facts 必须在对应 observation 中可见；第二个 title 的事实不得在第一轮泄漏。
5. gold 不得出现在问题或第一轮 observation 中，且必须在第二轮 observation 中首次可见，排除“题目里二选一即可猜中”的奖励捷径。
6. 排除重复文档、无新增证据、非法或空字段、`yes/no` 以及 source/question 重复。

oracle query 和 supporting metadata 只进入审计 catalog，绝不写入 prompt。Parquet 每行只保留 `data_source`、`prompt`、`ability`、`reward_model` 和 `extra_info`，防止把 benchmark context 泄漏给模型。

构建器必须输出 `train_512.parquet`、`val_128.parquet`、`probe_multi_64.parquet`、`catalog.jsonl`、筛选 ledger、`selection_funnel.json`、`manifest.json` 及 SHA-256 sidecar。漏斗按 category 固定记录 prescreen 后候选数、cap 后实际查询数、结构检索通过数、tokenizer 可见性通过数、去重后可用数、最终配额及全部拒绝原因；即使 retrieval 或 materialize 配额不足，也要先原子写出带 `status=failed`、失败阶段和差额的 receipt，再返回非零。成功漏斗作为 manifest artifact 固定，不能只留在终端日志。

本轮不允许 comparison/bridge 静默互补，也不放宽单条标准。若 CPU 漏斗显示配额不足，先保留完整失败证据并停止；第一选择是在新 commit 下提高相应 candidate cap 后重建，只有 cap 仍不足时才显式登记新的配比或筛选 policy。manifest 继续绑定源文件、BM25/corpus revision、tokenizer revision、top-k、长度配置及全部产物摘要；verify 从 pinned JSONL 重建 evidence 对应的 source record，并在固定 evidence 上重跑确定性选样，但不宣称离线重放全部未入选查询。最终入选 640 题须重放 1024 次 BM25 查询（256 道 NQ 各一次，384 道 Hotpot 各两次）。现有 NQ test-128 与 HotpotQA/2Wiki dev-256 都作为 exclusion 输入，不能进入新 train/val。

## 4. 训练前行为探针

探针必须使用下一阶段真实 parent。本轮 `R-mix60` 从原始 Qwen3.5-2B 开始，因此 probe 也使用该基座，不使用已经偏向单搜的 B/control20。

对 held-out Hotpot-64 每题按训练配置随机采样 5 条轨迹，共 320 条；`val_batch_size=8`，每批 40 条 rollout，不反向传播、不保存 checkpoint。有效正确多搜轨迹要求：

- `EM=1`、`n_search>=2`、无截断、无非法动作；
- 第二 query 与第一 query 不是近重复：规范化 token set Jaccard `<0.8` 才通过，`>=0.8` 或空 query 均拒绝；
- 第二轮带来新文档，并新增 supporting title 或让答案证据首次可见。

只有同时满足以下条件才进入训练：

| 门槛 | 要求 |
| --- | ---: |
| 有效正确多搜轨迹 | 至少 16/320 |
| 覆盖问题 | 至少 8/64 |
| 可学习 group | 至少 8 个 group 同时含正确多搜与错误轨迹 |
| 全局 clipped rate | 不高于 5% |
| 全局 invalid-action rate | 不高于 5% |

comparison/bridge 分层报告；至少 2 道 bridge 通过作为诊断目标，但首轮不设为硬门槛。probe NO-GO 是有效科学结果：停止长训练，不靠增加步数掩盖缺少探索轨迹的问题。

额外预注册 near-miss，但不改变上述 strict EM 门槛：`EM=0`，其余正确多搜条件全部满足。报告 near-miss 轨迹数、覆盖题数、comparison/bridge 分层、确定性的 cover-EM 及示例轨迹；cover-EM 定义为规范化后 prediction 与任一 gold alias 在 token 边界上双向包含，只用于发现过长或过短答案（如 `writ` 与 `writ of certiorari`），不参与奖励或 GO。若 strict gate 为 NO-GO 且 near-miss `>=16/320`，结论标为“存在检索链，优先诊断答案抽取/格式”；否则标为“正确多搜探索仍缺失”。

实现上只有 `data.eval_group_size=5` 的 probe 会复制样本并设置 `do_sample=True`；默认值 1 继续使用原有贪心评测。每条 probe trace 都带 `group_uid/group_slot/group_size`，并保存完整思考、query、检索文档、答案、截断及非法动作。分析产物固定包含机器可读 summary、逐题与逐轨迹 JSONL；科学 NO-GO 返回正常完成状态，只有 schema、digest 或基数错误才是工程失败。

group size 保持 5。改成 8 会把每 step rollout 从 40 条增至 64 条、成本提高 60%，却不能创造基座原本不存在的二搜能力；当前能力门槛要求至少 8/64 个 learnable group（12.5%），已经高于评审建议的 10% 边界。只有后续实测明确显示“有二搜正确轨迹但组内对比过稀”时，才把 group 8 作为独立实验，而不是本轮默认参数。

## 5. 能力训练与成本分叉

```text
A / Base Qwen3.5-2B
└── R-mix / 原始 EM 奖励，混合 train-512，60 steps
    ├── B-mix / 原始 EM 奖励，20 steps
    └── C-gated-mix / correct-only 成本奖励，20 steps
```

先只训练 `R-mix60`。完成后在同一 held-out probe-64 上重新采样，除第 4 节能力门槛外，还要至少有 8/64 个 group 出现“两条以上都正确但搜索次数不同”的 cost-contrast 信号。只有此时才从完全相同的 R-mix60 checkpoint 对称训练 B 与 C。

设 `c=n_search/4`，训练奖励为：

```text
R-mix / B-mix:       r = EM
C-gated-mix:         r = EM * (1 - 0.10 * c)
统一评测 utility:    u = EM - 0.10 * c
```

`correct_only` 使答错轨迹始终为 0，不再出现“搜索后答错比不搜索答错更差”的直接梯度。B/C 除奖励模式外必须共享 parent digest、数据顺序、seed、batch、group、长度、检索器和训练步数。

成本对比的预期来源也固定：NQ 主要学习把多余的 2/3/4 搜降到必要的 1 搜，Hotpot 主要把 3/4 搜降到必要的 2 搜，而不是把必须二搜的 Hotpot 降成一搜。R-mix 后必须记录正确轨迹的搜索次数对构成（`1 vs 2`、`2 vs 3`、`2 vs 4` 等）；若 8/64 cost-contrast 门槛未过，应报告“当前 parent 没有可学习成本信号”，不归因于工程失败，也不启动 B/C。

R-mix 入口实现时必须按 `data_source` 每 step 记录 EM、平均 `n_search`、no-search ratio 和全错 group ratio，并额外输出 10-step rolling 值。Hotpot rolling `n_search<1.5` 只落盘为单搜坍缩预警，不自动提前停止，也不采用“连续 10 步严格单调下降”这种对噪声过敏的规则；完整 60 steps 结束后再结合 Hotpot EM 与全错 group ratio 归因。

成功标准预注册为：C 相对 B 在共同答对题中的平均搜索量下降至少 10%，总体 EM 下降不超过 3 个百分点，统一 utility 提升，且多搜题不发生 no-search collapse。未达到也作为完整负结果保留。

## 6. 三阶段执行

### 阶段一：Git

本地完成实现、测试、commit 和 push。云端只接受固定 40 位 commit、detached HEAD 和干净 checkout；GitHub 失败时使用绑定 SHA-256 的 Git bundle，不使用未固定源码。

### 阶段二：CPU 无卡准备

复用现有模型、环境、Wiki corpus 和 BM25 索引，只新增固定的 NQ/Hotpot train 源文件、检索 evidence、混合 Parquet 和 manifest。CPU 阶段不运行 `nvidia-smi`，也不依据无卡实例规格调参。

新 commit 和新数据都会使旧 `cpu_handoff.json` 失效。CPU 阶段必须：

1. 验证源文件、旧资产和 checkout identity。
2. 用 retriever venv 生成 BM25 evidence，并先检查 `selection_funnel.json` 的 retrieval 产量；再用 train venv 和 Qwen tokenizer 做 384-token 可见性筛选及 Parquet materialize，最后用 retriever venv 重放所有入选查询。
3. 验证固定配额、完整筛选漏斗、train/val/既有测试题零重叠、Parquet 无 metadata/context 泄漏、manifest 可重算。
4. 组合 1/2 GPU resolved configs，精确断言 response 500、trajectory 4500、turns 4、top-k 3 和预算外一次非检索收尾生成。
5. 运行测试后发布新的自校验 handoff。

### 阶段三：两卡 GPU

当前实现切片只运行训练前 probe：

1. 启动 CPU BM25 服务并 health check。
2. 运行 Base × held-out Hotpot-64 × group-5 行为探针；NO-GO 则归档并停止。
3. 只有 GO 后才进入下一实现切片：接通混合训练入口，运行 `4500/500` 的 2-step smoke，再训练 R-mix60 并复测 probe。
4. 只有 R-mix 同时通过能力及 cost-contrast 门槛，才实现并训练对称的 B-mix20/C-gated-mix20。

当前 GPU 命令唯一为：

```bash
GPU_COUNT=2 AUTODL_PRICE_PER_HOUR=5.76 \
bash /root/autodl-tmp/search-r1/checkout/scripts/autodl/07_gpu_group_probe.sh
```

历史 `03/05/06` 不用于本轮 probe。这样若 Base 本身没有足够的正确二搜探索，最多只支付一次评测成本，不会先写或启动没有学习信号的长训练。

任一工程 gate 失败都停止，不自动改参或采用不完整 checkpoint。科学 NO-GO 不自动重试。终态、原始 exit code、日志和摘要全部落盘后，才允许已绑定 exact attempt 的 watchdog 请求关机；用户仍需在 AutoDL 控制台确认停止计费。

## 7. 证据、预算与面试口径

每个 attempt 保存 resolved config、run.env、完整 train/eval log、WandB offline history、逐轨迹 JSONL、trace manifest、checkpoint/parent digest、terminal 和 exit code。最终导出 loss/KL/entropy/grad norm、EM、搜索次数、截断率、非法动作率、query 变化、新文档覆盖、按 comparison/bridge 分层的成功轨迹及典型完整思考过程。

两卡单价按 5.76 元/小时记录。当前 probe 的硬上限为 10 元；response 500 只在输出实际变长时增加耗时。后续训练仍以 300 元为总硬上限，但只有 probe GO 后才启用训练预算。100 GB 数据盘足够：新增 Hotpot train 原文件约 0.57 GB，检索 ledger 和混合 Parquet 远小于 checkpoint。

面试中应如实表述：256 是局部截断干扰，但不是缺少多搜的唯一原因；真正的改进是把“多跳数据集标签”转化为由相同检索器验证的可执行二搜证据链，并用 group-level 探针在训练前检查稀疏奖励是否存在可学习信号。

## 8. 2026-07-23 实际探针结果

Base grouped probe 已按预注册配置完成，结论为 **NO-GO**。64 题、每题 5 条轨迹共 320 条中，EM 为 7/320；搜索次数 `0/1/2/3/4` 分布为 `140/90/53/32/5`。只有 2 条有效正确多搜轨迹，覆盖 2 题并形成 2 个可学习 group；cost-contrast group 和 near-miss 均为 0。截断率为 31.88%，非法动作轨迹率为 57.81%，五项硬门槛全部失败。

因此本轮在训练前停止，不启动 `R-mix60`，也不实现或训练 `B-mix20/C-gated-mix20`。一次空 `<search>` query 曾使原外层 analysis 错误退出；评测本身完整成功，最小分析器修复仅将该行为计为科学失败，并在独立离线 attempt 中生成正式 NO-GO。原失败 attempt、成功 eval、恢复血缘、完整轨迹和逐题/逐轨迹报告均归档于 `docs/results/grouped-probe-20260723/`。

## 9. Qwen3.5 协议适配后的后续入口

专项审计确认当前 NO-GO 混入了 Qwen3.5 原生工具协议未启用、提示词占位符复制和宽松 parser 误触发等因素。后续不直接启动 `R-mix60`，统一按 [`qwen35_native_tool_adaptation_plan.md`](qwen35_native_tool_adaptation_plan.md) 先完成原生协议适配和 G0-G3 分层门禁；只有新 grouped gate 通过，才恢复本文件第 5 节的能力训练与成本分叉。
