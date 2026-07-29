# Qwen3.5 Native-v4 R60 训练与全量轨迹分析

> 分析日期：2026-07-29
>
> 模型：`Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc`（post-trained，不是 Base）
>
> Sealed checkout：`f8c1cd7e87078d07385f74ca8710add5d5f79c06`
>
> 当前结论：**R60 工程与训练信号 GO；只放行一次冻结 R60 checkpoint 的 G3-only 评测，B/C 分支仍为 NO-GO。**

## 1. 结论先行

本轮 R60 已真实完成 60 次全参数 GRPO 更新，并保存 `global_step_60`、WandB 60 步历史和 2,400 条完整训练轨迹。没有 OOM、NaN、Inf、NCCL、Traceback 或运行时错误。相较从同一 sealed parent 独立启动的 2-step smoke endpoint，R60 在同一固定 val-128 上的 strict EM 为 `75/128 = 58.59%`，比 `48/128 = 37.50%` 高 21.09 个百分点；平均搜索由 `3.117` 降到 `1.992`，no-search 由 5 题降到 1 题。这是单 seed 随机 endpoint 的观测性对照，但足以排除“表面省搜索完全来自不检索”的解释。

但 R60 不能直接批准成本分支。主要原因有三点：

1. 本轮 `cost_lambda=0`，只在固定 endpoint 观测到纯 EM 基线有更高正确率和更少 executed search，尚未检验成本奖励。
2. EM 增益主要来自 NQ：NQ 增加 25 题，HotpotQA 只增加 2 题。成本实验真正关心的 held-out 多跳能力仍需单独验证。
3. 第 52 步后出现明显的长输出漂移。末 10 步真实截断达到 `202/400 = 50.5%`，含非安全类协议异常的轨迹达到 `186/400 = 46.5%`；同时 KL 后段升高。训练没有数值爆炸，但最终 checkpoint 的格式与多搜能力不能只靠训练 rollout 判定。

因此最小下一步不是运行旧的完整 `main`，也不是直接联跑 B/C，而是只对 sealed R60 checkpoint 跑一次预注册的 G3。当前已增加最小 G3-only operator：它保留 R60 当时的 checkout/handoff/数据身份，只消费 exact R60 marker 和 `global_step_60`，封存 G3 后立即停止。G3 完成后先人工分析，不自动进入 endpoint 或 B/C。

## 2. 终态、血缘与证据完整性

### 2.1 为什么外层显示 `failed / 200`

本轮存在两个不同层次的终态：

| 层次 | attempt / run | 终态 | 含义 |
| --- | --- | --- | --- |
| 内层训练 | `20260728T092332Z-2946-8453` | `success`、`exit-code=0` | 60 步训练、验证、checkpoint 和 trace job 成功 |
| 外层 R60-only | `20260728T092026Z-2906-16060` | `failed`、`exit-code=200` | 包装器按设计返回受控非零码，让旧 watchdog 进入可靠关机路径 |

所以外层的 200 **不是训练失败**。内层结束后，外层 runner 又完成 lineage、WandB receipt、checkpoint/trace 复验以及 R60-only marker/evidence 的发布。R60-only 合同明确记录：

```text
stage_order=R60
branch_training_authorized=false
next_stage_requires_manual_approval=true
controlled_outer_exit_code=200
```

本轮没有运行 G3、A/R endpoint 详细评测、B20 或 C20，也没有更新会让旧入口误判阶段的 `latest-main`。后续若直接运行旧 `main`，可能从 parent 重训 R60 并继续后续阶段，不能作为 G3-only 的启动方式。旧 watchdog 在 outer `failed/200` 下只负责走失败态关机路径，并不理解或验证 R60-only 的科学成功合同；训练成功必须依据独立的 inner `success/0`、marker 和 evidence，不能从 `shutdown-dispatched` 反推。

### 2.2 可复现身份

| 证据 | 精确值 |
| --- | --- |
| R60-only `evidence.sha256` manifest digest | `2366ce2da28b530f12af30a22d3dc3e2bd33fedbfcb3a2bed5acdd9d0537ffed` |
| R60 checkpoint tree digest | `583771b131b6e2aa663ee2ef13ea6421524cdd7fed839f35fd6f9eb246a2c231` |
| 2,400 行 trace digest | `cfcb118cb16dead6bb77e4f072b1dbc0009bfaf7257bf463c7a04a9c7f85bde4` |
| resolved config digest | `9d339e69b98693c5d7aa6a1a863d62945d8f5cbc271d28f271f34289c64448c2` |
| parent checkpoint digest | `bc67be20efb353ba14d9c1b291a64410afec94f2310e94b59b6b76047b164e78` |
| CPU handoff digest | `9ffaa89f88990887b368750ccfdc559d93afdb9a1f5719411cf3e3c1147da90b` |
| R60-only runner SHA-256 | `d9465a64710af031a066016f286d4ff396a7eeda16b185b463dab717030f07b3` |

远端关键位置：

```text
/root/autodl-tmp/search-r1/state/attempts/gpu/20260728T092026Z-2906-16060
/root/autodl-tmp/search-r1/runs/reproduce/attempts/20260728T092332Z-2946-8453
/root/autodl-tmp/search-r1/manifests/qwen-native-training-r60-only/20260728T092026Z-2906-16060.ok
```

trace manifest 声明 2,400 行，独立解析得到恰好 `60 steps × 8 prompts × group 5`。480 个 group 都包含 `group_slot=0..4`，480 个 `sample_id` 全部不同，没有缺槽、重复记录或 prompt 重放。checkpoint 树约 9.6 GB，checkpoint、trace、config、WandB 60 步和 evidence manifest 均已重新哈希并通过。

`f8c1cd7` 是实验时的 sealed checkout；R60-only wrapper 当时位于 checkout 外，以独立 runner SHA-256 绑定。将该 wrapper 随本报告提交属于事后可复现归档，不应把新的归档 commit 误写成训练时 checkout。

### 2.3 耗时、费用与关机

| 口径 | 时间 | 按 5.76 元/小时估算 |
| --- | ---: | ---: |
| 内层 run wall-clock | `33,727 s = 9:22:07` | 约 `53.96 元` |
| 外层完整证据窗口 | `9:27:06` | 约 `54.44 元` |

外层在北京时间约 2026-07-28 17:20 启动，内层训练约 17:23 开始，2026-07-29 02:45 完成，约 02:47 发出关机请求。WandB `_runtime` 为约 33,700 秒，和 inner wall-clock 的小差异来自启动/收尾口径。watchdog 已依次发布 `shutdown-safe`、`shutdown-requested` 和 `shutdown-dispatched`，backend 返回 0；用户随后实际观察到实例关机。持久化状态仍为 `provider_control_plane_confirmed=false`，因此严格说只能证明 guest 关机请求成功并由用户从平台侧观察到停机，而不是程序拿到了 AutoDL 控制面回执。费用只是证据时间窗乘单价，不含 attempt 前开机和 shutdown dispatch 后平台实际停止的延迟，不能替代 AutoDL 账单。

## 3. 本轮 exact 训练合同

| 参数 | 值 |
| --- | --- |
| 训练方式 | Qwen3.5-2B 全参数 FSDP，非 LoRA |
| GPU | 2 卡 |
| steps | `60` |
| batch / group | `8 / 5`，每步 40 条 rollout |
| prompt groups / trajectories | `480 / 2,400` |
| regular action budget | `4`；未结束时额外做一次 terminal generation |
| response / observation | 每次 generation `500` / 每次检索回填 `500` token |
| temperature / top-p / top-k | `1.0 / 1.0 / 0` |
| learning rate / warmup ratio | `1e-6 / 0.285` |
| PPO epochs | `1` |
| reward | strict EM，linear，`cost_lambda=0` |
| retriever | 项目官方支持的 BM25，top-3 |
| prompt | `qwen35-native-search-v4-terminal-answer-only` |

训练集合同为 512 题：320 条 HotpotQA 和 192 条 NQ。本轮无放回 shuffle 实际覆盖其中 480 题，轨迹分布为 NQ 182 题/910 条、HotpotQA 298 题/1,490 条。验证集为固定 val-128，NQ 和 HotpotQA 各 64 题。

Transformers worker 曾提示 `top_k` generation flag 可能被忽略；当前配置本来就是 `top_k=0`，所以这不是运行错误，也不能写成“启用了 top-k 截断采样”。

## 4. Loss 与数值稳定性

WandB receipt 为 `decision=GO`，有完整的 60/60 history。reward、advantage、old log-prob 全部有限，训练合同、policy mask 和 loss mask 检查始终为 1。

| 指标 | 1-20 步 | 21-40 步 | 41-60 步 | 全程解释 |
| --- | ---: | ---: | ---: | --- |
| train EM / reward | `27.63%` | `44.25%` | `46.50%` | 总计 `947/2400 = 39.46%` |
| actor PG loss | `-0.0168` | `+0.0350` | `-0.0399` | 全程均值 `-0.00723`，范围 `[-0.26381, 0.33910]` |
| `actor/kl_loss` | `0.00183` | `0.02521` | `0.07076` | `low_var_kl` reference-KL surrogate，后段上升，峰值 `0.182869@59` |
| entropy | `0.6058` | `0.5148` | `0.5600` | 有限且非单调，没有熵坍缩 |
| grad norm | `2.5666` | `1.9349` | `1.6783` | 全部有限，最大 `5.176@6` |
| valid-action ratio | `87.84%` | `93.07%` | `85.99%` | 先算每轨迹合法比例再宏平均；中段改善，后段回落 |
| 真实 clipped trajectory | `12.75%` | `12.00%` | `33.50%` | 来自原始 trace，不用日志动态宽度代理 |

GRPO 的 policy-gradient loss 不应像监督学习交叉熵一样单调下降。每一步使用不同问题和 on-policy rollout，组内标准化 advantage 又会让正负项抵消；本轮合理结论是“loss 有限、更新真实发生且 reward/endpoint 改善”，而不是“PG loss 越小越好”。

`ppo_kl` 和 `pg_clipfrac` 全程为 0。当前只有一个 policy update epoch，old policy 与该 batch 开始更新时的 current policy 一致，所以不能据此说模型没更新，也不能声称 PPO clipping 在本轮发挥了作用。独立 reference-KL surrogate、非零梯度以及 R60 checkpoint 相对 parent 不同的内容 digest 均证明参数发生了更新。

需要警惕的是 KL 随 step 持续升高，step 与 KL 的相关系数约 `0.828`。它没有伴随 NaN、梯度爆炸或熵坍缩，因此不是数值崩溃；但它与第 52 步后的长输出和格式恶化同时出现，必须由冻结 checkpoint 的 G3 轨迹判断是否已经影响泛化。

## 5. 2,400 条训练轨迹总体结果

| 指标 | 结果 |
| --- | ---: |
| strict EM | `947/2400 = 39.46%` |
| 提交合法答案 | `1874/2400 = 78.08%` |
| 无答案 | `526/2400 = 21.92%` |
| 实际 BM25 搜索 | `5697` |
| 平均搜索 | `2.3738` |
| 正确轨迹平均搜索 | `1969/947 = 2.0792` |
| 错误轨迹平均搜索 | `3728/1453 = 2.5657` |
| 完全不搜索 | `43/2400 = 1.79%` |
| 进入 terminal reminder | `974` |
| 真实 clipped trajectory | `466/2400 = 19.42%` |

### 5.1 搜索次数与正确率

| 实际搜索数 | 轨迹 | strict 正确 | 正确率 |
| ---: | ---: | ---: | ---: |
| 0 | 43 | 2 | 4.65% |
| 1 | 580 | 249 | 42.93% |
| 2 | 790 | 457 | 57.85% |
| 3 | 411 | 150 | 36.50% |
| 4 | 576 | 89 | 15.45% |

这组分布说明当前模型没有“干脆不搜索”的坍缩：只有 43 条零搜索轨迹，而且其中仅 2 条正确。两次搜索的观测正确率最高，四次搜索的正确率最低；正确轨迹也比错误轨迹平均少 `0.4865` 次搜索。它支持进一步检查“找到证据后是否及时回答”，但不能解释成少搜本身导致答对，因为困难问题更可能同时需要更多搜索且更容易答错。

完全重复 query 并不常见：只有 6/2,400 条轨迹出现归一化后相同的重复查询。代表性抽样中可见“证据已经出现仍继续改写查询”、证据抽取失败和长输出后违反协议；目前没有对全量 evidence-hit 后续搜索做量化，不能断言其中某一种是全局主要原因。

### 5.2 数据源分层

| 来源 | 轨迹 | strict EM | 平均搜索 |
| --- | ---: | ---: | ---: |
| NQ | 910 | `379/910 = 41.65%` | `2.136` |
| HotpotQA | 1,490 | `568/1490 = 38.12%` | `2.519` |

从前 10 步到后 10 步，NQ 的训练 rollout EM 从 `45/160 = 28.13%` 升至 `89/175 = 50.86%`，平均搜索从 `2.400` 降至 `2.034`；HotpotQA 从 `43/240 = 17.92%` 升至 `106/225 = 47.11%`，搜索从 `3.038` 降至 `2.298`。两类数据都出现能力与效率信号，但每一步是不同题目，且前后来源占比不同，所以这是按来源控制后的观察性趋势，不是同题配对的因果估计。

当前 trace 只有 `data_source`，没有 HotpotQA 的 `bridge/comparison/type/hop_proxy` 字段；若要做可靠子类型分层，必须按 `sample_id/source_index` join sealed catalog，不能从问题文本人工猜类别。

### 5.3 Group、advantage 与可学习信号

| 每组 5 条中的正确数 | group 数 |
| ---: | ---: |
| 0 | 182 |
| 1 | 50 |
| 2 | 55 |
| 3 | 55 |
| 4 | 68 |
| 5 | 70 |

480 个 group 中有 `228 = 47.5%` 为 mixed-reward group，它们恰好产生 1,140 条非零 advantage 轨迹；其余全错或全对组的组内 advantage 为 0。总 policy token 为 1,657,513，其中 `779,085 = 47.0%` 位于非零 advantage 轨迹。说明 group 5 已提供相当密度的相对学习信号，没有因 reward 全同而失去训练作用。

若忽略格式与截断，训练轨迹中有 171/480 个 raw cost-contrast group；按“至少两条正确、正确成员无 invalid/clip、且正确成员搜索次数不同”的更严格 clean 口径有 123/480 个。这证明训练分布内存在正确答案之间的成本差异，但这些题已参与更新，不能替代 held-out G3，也不能证明加成本惩罚后不会损失正确率。

## 6. 早期到后期：进步与退化同时存在

| 指标 | Step 1-10 | Step 51-60 | 变化 |
| --- | ---: | ---: | ---: |
| strict EM | `88/400 = 22.00%` | `195/400 = 48.75%` | `+26.75 pp` |
| 合法答案 | `233/400` | `291/400` | `+58` |
| 平均搜索 | `2.7825` | `2.1825` | `-21.6%` |
| 四次搜索 | `156/400 = 39.0%` | `57/400 = 14.25%` | `-24.75 pp` |
| terminal 进入数 | `231` | `187` | `-44` |
| terminal 合法回答 | `64/231 = 27.7%` | `78/187 = 41.7%` | `+14.0 pp` |
| terminal 越界搜索被拒 | `113` | `13` | `-100` |
| 平均 generated tokens | 约 `677` | 约 `1068` | `+57.7%` |
| clipped trajectory | `48/400 = 12.0%` | `202/400 = 50.5%` | `+38.5 pp` |
| 含非安全协议异常的轨迹 | `125/400 = 31.25%` | `186/400 = 46.5%` | `+15.25 pp` |

积极变化很明确：EM 提高、搜索减少、四搜比例下降，terminal 后继续请求搜索也显著减少。后 10 步已经出现任务形态相关的模式：NQ 最常使用一次搜索，HotpotQA 最常使用两次搜索。这正是成本分支需要的基础行为。

问题同样明确：从 step 52 起，生成长度、截断和 `invalid_thinking_prefix` 同时跳升。`invalid_thinking_prefix` 事件从前 10 步的 128 增到后 10 步的 313。后 10 步的 clipped 轨迹 strict EM 约 30.2%，未截断轨迹约 67.7%；clipped 轨迹的 raw any-invalid 约 85.6%，未截断轨迹约 11.1%。这只是关联，不能证明截断是唯一原因，但足以把最终 checkpoint 的长度/格式行为列为 G3 硬门。

还要注意时点：每一步训练 trace 都是该步更新前的 on-policy rollout，且 480 题不重复；最后十步不等于冻结 `global_step_60` 的 held-out 表现。不能把 `50.5% clipping` 直接写成最终 checkpoint 的确定指标，也不能因末段异常就宣告 R60 已失败。

## 7. Terminal reminder 与非法动作

974 条轨迹耗尽常规 action 后进入 terminal generation。提醒文本为环境消息，不计入 policy loss（`policy_token_count=0`）；模型在提醒后的 assistant 输出仍按正常策略生成并参与轨迹结果/reward。它是软提示，不是约束解码。

| Terminal 结果 | 数量 | 比例 |
| --- | ---: | ---: |
| 合法 `<answer>` | 448 | 46.0% |
| 再次请求 search，被环境安全拒绝 | 190 | 19.5% |
| 其他格式/协议失败 | 336 | 34.5% |

所有 190 次越界搜索都执行了 0 次，说明预算硬边界有效。448 个 terminal 合法答案中有 175 个 strict 正确，EM 为 39.1%。全部 526 条无答案轨迹都来自 terminal 最终失败，说明“额外一次回答机会”明显有用但无法保证回答。

全程原始 any-invalid 为 817/2,400 条轨迹、1,277 个事件。若把 190 次“模型越界请求但环境按设计安全拒绝”单独列出，真正的解析/协议异常为 1,087 个事件，涉及 `673/2400 = 28.04%` 轨迹。

| 错误类型 | 事件数 |
| --- | ---: |
| `invalid_thinking_prefix` | 762 |
| `search_disallowed_after_budget` | 190 |
| `missing_native_action` | 168 |
| multiple/unbalanced tool calls | 66 |
| multiple/unbalanced answers | 43 |
| `invalid_terminal_answer_format` | 32 |
| unknown tool | 13 |
| 其他 | 3 |

因此，当前首要风险不是检索器反复返回错误，而是长生成后的 native action 边界和最终答案格式。G3 应同时报告 raw invalid、排除安全拒绝后的 protocol invalid、真实 clip 和 terminal outcome，不能只看 EM。

## 8. 固定 val-128：Smoke step 2 与 R60 step 60

两次 endpoint 使用同一 val-128 与同一 strict EM/检索合同。R60 的前两步核心训练 rollout 聚合指纹与 smoke 两步一致，支持二者从同一 parent 和配置重新起跑；R60 不是从 smoke checkpoint 接训。由于 endpoint 使用随机生成、只有一个 seed 且没有逐题配对 trace，表中差值是受控程度较高的观测性对照，不是严格因果估计。

| Endpoint | NQ EM | HotpotQA EM | 总 EM | 平均搜索 | no-search |
| --- | ---: | ---: | ---: | ---: | ---: |
| Smoke step 2 | `16/64 = 25.00%` | `32/64 = 50.00%` | `48/128 = 37.50%` | `399/128 = 3.117` | `5/128` |
| R60 step 60 | `41/64 = 64.06%` | `34/64 = 53.13%` | `75/128 = 58.59%` | `255/128 = 1.992` | `1/128` |
| 变化 | `+25 / +39.06 pp` | `+2 / +3.13 pp` | `+27 / +21.09 pp` | `-144 / -36.1%` | `-4` |

这是本轮最强的 endpoint 证据：更深训练后，固定验证集同时出现正确率提升和搜索下降。但它仍有边界：

- 没有同一 val-128 上的 sealed-parent（0-step）结果，所以不能报告“相对原始 Qwen parent 提升多少”。
- 最终验证日志只有聚合指标，没有逐题 trace；无法从 val-128 计算最终 checkpoint 的 clipping、invalid、terminal compliance 或 cost-contrast group。
- 只有一个 seed，没有置信区间。
- 总增益中 25/27 道来自 NQ，HotpotQA 只增加 2 道；不能据此跳过多跳门禁。

## 9. 代表性 action 轨迹

以下只展示问题、搜索 query、短检索证据、动作结果和答案，不复制隐藏思考或完整 raw generation。

### 9.1 标准两跳成功：学校到城市再到县

`hotpotqa:train:70907`，step 1，`group_slot=3`：

```text
Question: The Elisabeth Morrow School is located in what county?
Search 1: Elisabeth Morrow School location county
Evidence: school is located in Englewood, New Jersey
Search 2: Englewood New Jersey county
Evidence: Englewood is located in Bergen County
Answer: Bergen County
Result: EM=1, 2 searches, no invalid, no clipping
```

这说明早期 parent 已能完成标准实体链，但同组另有 rollout 用满四搜仍不回答，group 内存在有效偏好信号。

### 9.2 后期高效 NQ：一次搜索直接回答

`nq:train:59612`，step 51，`group_slot=1`：

```text
Question: who wrote the poem if we must die?
Search: If we must die poem author
Evidence: "If We Must Die" is a 1919 poem by Claude McKay
Answer: Claude McKay
Result: EM=1, 1 search, no invalid, no clipping
```

### 9.3 后期标准 Hotpot 两跳

`hotpotqa:train:27237`，step 51，`group_slot=0`：

```text
Question: The person who composed music for the movie Sivappu is of what nationality?
Search 1: Sivappu movie composer
Evidence: music composed by N. R. Raghunanthan
Search 2: N R Raghunanthan nationality
Evidence: N. R. Raghunanthan is an Indian film score and soundtrack composer
Answer: Indian
Result: EM=1, 2 searches, no invalid, no clipping
```

### 9.4 明确的成本优化空间：答案已出现仍继续搜

`hotpotqa:train:40313`，step 51，`group_slot=4`：

```text
Question: "Silly Love Songs" is a song on an album released on which date?
Search 1: "Silly Love Songs" song album release date
Search 2: "Wings at the Speed of Sound" release date 1976
Evidence: the album was released on 25 March 1976
Search 3: "Wings at the Speed of Sound" released date 25 March 1976
Search 4: "Silly Love Songs" sung Wings at the Speed of Sound release date 25 March 1976
Terminal reminder: applied after the four-search budget was exhausted
Answer: 25 March 1976
Result: EM=1, 4 searches, no invalid, no clipping
```

同一 group 的另外三条正确 rollout 只用了两次搜索。这是训练分布内最直观的 cost-contrast：准确率相同但成本不同，适合未来 B/C 奖励进行组内选择。

### 9.5 检索成功但最终无答案

`nq:train:62300`，step 51，`group_slot=0`：

```text
Question: what is gecko's name on pj masks?
Search 1 evidence: BM25 hits the PJ Masks page; the truncated chunk contains "and Greg ..."
Subsequent searches: continued to query character/product wording
Terminal: invalid_thinking_prefix, no legal answer
Result: EM=0, 4 searches, clipped
```

这里不是 BM25 完全偏离主题，但 passage 起始被截断，没有明确给出 `Gekko/Gecko -> Greg` 的实体映射。检索片段不完整、模型未利用弱证据、后续搜索漂移和 terminal 格式失败共同导致无答案；不能把失败全部归因于模型抽取。

### 9.6 Strict EM 假阴性

`hotpotqa:train:14574`，step 52，`group_slot=1`：

```text
Question: Which filmmaker is younger, Géza von Cziffra or John Sayles?
Prediction: John Sayles
Gold list: John Thomas Sayles
Terminal reminder: applied after the four-search budget was exhausted
Recorded reward: EM=0
```

语义上预测指向正确人物，但当前金标只包含全名，strict EM 将其计为 0。该噪声会影响 group advantage；因此报告应保留 strict EM 作为论文主口径，同时把 alias/sub-EM 仅作为诊断，不能偷偷替换训练 reward 后仍声称完全复现。

## 10. 下一步决策：只跑 G3

### 10.1 当前分层结论

| 决策层 | 结论 | 理由 |
| --- | --- | --- |
| 工程链路 | GO | 60/60、exit 0、checkpoint/trace/WandB/evidence 完整 |
| R60 训练信号 | GO | 固定 val EM 提升 21.09 pp，搜索下降 36.1%，无 no-search 坍缩 |
| 最终多跳/格式能力 | 待 G3 | 末段 clipping、invalid、KL 上升，final val 无逐题 trace |
| 直接训练 B/C | NO-GO | `lambda=0` 基线尚未通过 held-out cost-contrast 与协议硬门 |

### 10.2 G3-only 预注册门

G3 使用 sealed R60 `global_step_60`，固定 held-out HotpotQA 64 题，每题 group 5，共 320 条；不换 seed、不挑训练中间点、不因首次结果不好而重采样。五项 capability gate 必须同时满足：

其中 qualifying correct multi-search 要求 `EM=1`、`n_search>=2`、无 clip/invalid；第二 query 与第一 query 的规范化 token Jaccard `<0.8`，并带来新文档以及新 supporting title 或首次可见的答案证据。`clean wrong` 指 `EM=0` 且无 clip/invalid 的同组轨迹。

1. 有效正确多搜轨迹 `>=16/320`；
2. 正确多搜覆盖 `>=8/64` 题；
3. 可学习 group `>=8/64`：同组同时含 qualifying clean correct multi-search 与 clean wrong 轨迹；
4. 真实 clipped ratio `<=5%`；
5. 全局 raw any-invalid-action ratio `<=5%`，按 `invalid_action_count>0` 统计并包含被安全拒绝的 terminal 越界搜索。

排除安全拒绝后的 protocol-invalid 继续作为解释性诊断，但不得事后替换第 5 项注册门。末 10 步邻近 policy 的 clip/invalid 远高于 5%，所以 G3 有较高 NO-GO 风险；运行它的目的正是取得冻结 checkpoint 的 endpoint 证据，而不是默认放行 B/C。

五项中任一失败，应记录 `G3 capability=NO-GO` 并作为完整科学结果停止。若五项全部通过，则记录 `G3 capability=GO`，再单独检查 branch authorization：clean cost-contrast group 必须 `>=8/64`。cost-contrast 不足时，结论是“capability GO、branches NO-GO”，不能反过来改写 G3 能力结论。

即使 capability 和 cost-contrast 都通过，也只进入人工复核；现有完整 `09...main` 还包含 A/R endpoint 和潜在 B/C 自动续跑，不能直接复用为下一条命令。`10_gpu_qwen_native_r60_only.sh` 也会从 base 重训 R60，同样不能运行。后续应按既定血缘决定是否执行 A/R endpoint，以及是否另行批准同 parent、同数据、同采样和同 20 步预算的 B20/C20；两个训练分支只允许改变预注册奖励项。

### 10.3 G3-only 实现边界与命令

[`11_gpu_qwen_native_g3_only.sh`](../scripts/autodl/11_gpu_qwen_native_g3_only.sh) 不成为 R60 实验 checkout 的一部分。R60 绑定的 checkout 继续固定为 `f8c1cd7e87078d07385f74ca8710add5d5f79c06`；新 runner 通过受控 SFTP 放到 `/root/autodl-tmp/search-r1/operator/`，并把自身 SHA-256 写入 CPU receipt 和最终 evidence。这样既可以增加“只跑 G3”的操作边界，又不会用新代码身份伪装成 R60 当时的训练代码。

CPU 无卡阶段只执行前驱校验：

```bash
QWEN_NATIVE_PROTOCOL_GATE_EVIDENCE=/root/autodl-tmp/search-r1/manifests/qwen-native-gate/20260728T044634Z-2051-4045.ok \
QWEN_NATIVE_SMOKE_EVIDENCE=/root/autodl-tmp/search-r1/manifests/qwen-native-training-smoke/20260728T061246Z-1316-15067.ok \
QWEN_NATIVE_R60_EVIDENCE=/root/autodl-tmp/search-r1/manifests/qwen-native-training-r60-only/20260728T092026Z-2906-16060.ok \
bash /root/autodl-tmp/search-r1/operator/11_gpu_qwen_native_g3_only.sh --cpu-prepare
```

该步验证冻结 checkout、CPU handoff、native-v4 数据、G0/G1、smoke、R60 marker、R60 trace 和 checkpoint digest，不重封 handoff、不更新 checkout、不重建数据。因此不得运行 `AUTODL_RESEAL_ONLY=1` 或 `02_cpu_prepare.sh`。CPU verifier 通过后，挂载两卡只执行：

```bash
QWEN_NATIVE_PROTOCOL_GATE_EVIDENCE=/root/autodl-tmp/search-r1/manifests/qwen-native-gate/20260728T044634Z-2051-4045.ok \
QWEN_NATIVE_SMOKE_EVIDENCE=/root/autodl-tmp/search-r1/manifests/qwen-native-training-smoke/20260728T061246Z-1316-15067.ok \
QWEN_NATIVE_R60_EVIDENCE=/root/autodl-tmp/search-r1/manifests/qwen-native-training-r60-only/20260728T092026Z-2906-16060.ok \
GPU_COUNT=2 AUTODL_PRICE_PER_HOUR=5.76 \
bash /root/autodl-tmp/search-r1/operator/11_gpu_qwen_native_g3_only.sh
```

入口只在 held-out HotpotQA 64 题上每题采样 5 条，不更新权重、不产生 checkpoint、不运行 endpoint 或 B/C。无论科学结论为 GO 还是 NO-GO，都会封存 `qwen-native-training-g3-only-v1` marker 和 evidence。旧 watchdog 不识别该新成功合同，所以 runner 在封存完成后按设计返回受控外层状态 `201`，用于走旧 failure-path 关机。`201` 不是评测失败；必须用 G3-only marker、evidence manifest 和内层 eval `success/0` 判定完成。

当前数据盘为 150 GB，已用约 113 GB、剩余约 38 GB。一次 G3 评测不产生新的 9 GB 全参数 checkpoint，空间足够；若继续保留两个约 9 GB 的 B/C checkpoint、轨迹和临时目录，38 GB 会偏紧，启动分支前应先重新做保留策略和空间预算，而不是现在付费扩容。

## 11. 结论边界与面试表述

本轮可以可靠表述为：

> 我在两张 5090 级 GPU 上对 Qwen3.5-2B 做了 60 步全参数 GRPO Search-R1 缩小复现，共采样 2,400 条可审计轨迹。相较从同一 parent 独立启动的两步 smoke endpoint，固定 val-128 的 strict EM 高 21.1 个百分点，平均检索从 3.12 次降到 1.99 次，并且零检索比例没有上升。与此同时，我通过轨迹发现 52 步后 reference-KL surrogate、生成长度、截断和协议错误共同上升，因此没有直接启动成本奖励分支，而是预注册 held-out 多跳 G3，分别检查能力门和成本分支授权门。

不能表述为：

- “成本感知奖励已经有效”：本轮 `cost_lambda=0`；
- “相对原始 parent 提升 21.09 pp”：比较对象是 smoke step 2，不是 0-step parent；
- “最终模型有 50.5% 截断”：50.5% 来自最后十步更新前训练 rollout，不是冻结 checkpoint 评测；
- “搜索越少所以越正确”：当前只有相关关系，题目难度是混杂因素；
- “R60 已通过多跳能力门”：Hotpot endpoint 增益很小，G3 尚未运行。

WandB 已保留全部 60 步 PG loss、reference-KL surrogate、entropy、grad norm、reward、搜索和时长指标；原始 trace 保留逐题 query、检索文档、action、答案、reward、invalid 与 clipping 字段。后续绘图与面试追问都有可复算的原始资料，不依赖本报告中的手工摘要。
