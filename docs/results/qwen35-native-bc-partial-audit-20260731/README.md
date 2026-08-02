# Qwen3.5 native B/C 训练终态、数据完整性与可分析性报告

审计日期：2026-08-01（Asia/Shanghai）
实验身份：direct-RL R60 的 post-hoc exploratory B/C
审计结论：**整套 B/C 实验未完成；B 完整，C 在第 20 步的内置验证阶段超时；现有数据只能用于部分分析，不能用于正式 B20/C20 端点结论。**

## 1. 一句话结论

- 训练是否完毕：**没有按实验合同完毕**。outer 是 `failed/124`，不是合同要求的受控 `failed/202`。
- 数据是否完全：**不完全**。B 的训练、checkpoint 和 trace 完整；C 有 800 条合法但未封存的 `.partial` 训练轨迹，没有 checkpoint；6 个独立 endpoint eval 和 3 套 paired 结果全部未生成。
- 能否分析：**可以做 B 完整训练分析、C 超时复盘、B/C 训练过程的探索性配对分析；不能做正式 B20 vs C20 泛化优劣、NQ/多跳对比或最终论文结论。**

## 2. 精确实验身份

| 项目 | 值 |
|---|---|
| outer attempt | `20260731T060828Z-1554-11705` |
| outer 路径 | `/root/autodl-tmp/search-r1/state/attempts/gpu/20260731T060828Z-1554-11705` |
| 预定 result root | `/root/autodl-tmp/search-r1/runs/qwen-native-training/attempts/20260731T060828Z-1554-11705` |
| 预定 marker | `/root/autodl-tmp/search-r1/manifests/qwen-native-training-bc-only/20260731T060828Z-1554-11705.ok` |
| runner contract | `qwen-native-training-bc-only-v1` |
| runner commit | `9fc2e3a4245d9b9a09ae43b41a9fa268b4b3d68e` |
| runner SHA-256 | `d696fcd752e301e81e7b555df961613837605e2edd25add550ad1c867e6f1a9f` |
| frozen train checkout | `f8c1cd7e87078d07385f74ca8710add5d5f79c06` |
| 共同 R60 parent digest | `583771b131b6e2aa663ee2ef13ea6421524cdd7fed839f35fd6f9eb246a2c231` |

该合同的科学成功终态被有意设计为 outer `failed/202`，同时必须有同 attempt marker、outer bindings、`evidence.sha256` strict check、两个 checkpoint tree digest 和全部 inner run 证据。当前的 `failed/124` 是真实超时失败，不能被解释成受控成功。

## 3. 终态证据

### 3.1 outer

| 证据 | 实际值 | 判断 |
|---|---:|---|
| `terminal` | `failed` | 失败终态 |
| `exit-code` | `124` | timeout，不是受控 `202` |
| terminal sentinel | 只有 `.failed` | 状态写入完整 |
| requested | `2026-07-31T06:08:28Z` |  |
| finished | `2026-07-31T15:27:30Z` |  |
| `native-training-runs.tsv` | 只有 B 一行 | C 未登记为成功 run |
| final result root | 空目录 | 未进入最终封存 |
| scientific marker | 缺失 | 科学成功不成立 |
| evidence/bindings | 全缺失 | 无法做最终 checksum 审计 |

`phase.log` 的最后结论是：

```text
train cost_aware_gated failed with exit code 124
AUTODL_PHASE_TERMINAL state=failed exit_code=124
```

关机状态机本身完整执行：`shutdown-safe → shutdown-requested → shutdown-dispatched`。这证明 guest 内已发出关机命令，不证明实验成功，也不等同于云厂商控制面的停机确认。

### 3.2 B / control

| 项目 | 实际值 |
|---|---:|
| run | `20260731T061302Z-1592-7937` |
| terminal / exit | `success / 0` |
| elapsed | 17,558 秒（4:52:38） |
| optimizer metric steps | 20/20 |
| train trace | 800/800 行，20 步 × 40，全部合法 JSON |
| trace manifest | 存在，sidecar 与 strict SHA-256 检查通过 |
| checkpoint | `checkpoints/actor/global_step_20` |
| checkpoint 大小 | `model.safetensors` 9,561,579,440 bytes |
| checkpoint tree digest | `8df3d6ab13e154b4a2360b1535f763731f955baebba21b75f9f6e017000ad6d2`，重算匹配 |
| trace SHA-256 | `b1907e579ecece4fe4c102d229cc3103b8ee71348bbd8c162bfe320a04a51cf5` |
| W&B | 非空 offline binary；没有最终 workflow receipt |

B 是可恢复、可独立评测的真实 B20 endpoint。

### 3.3 C / cost-aware gated

| 项目 | 实际值 |
|---|---:|
| run | `20260731T110656Z-1592-183` |
| terminal / exit | `failed / 124` |
| elapsed / timeout | 15,633 / 15,625 秒 |
| 日志中完整 metric steps | 1–19 |
| partial train trace | 800 行，20 步 × 40，全部合法 JSON |
| partial trace SHA-256 | `02b2dd53874b24a59ce252a670363af7865368ab5340bbadb05f0ed9727bb4ae` |
| trace manifest / sidecar | 缺失 |
| checkpoint | 缺失 |
| lineage / native contract | 缺失 |
| W&B | 非空 offline binary，但未正常收尾、无 receipt |

C 的 `save_freq=20`、`test_freq=20`。冻结代码在第 20 步的顺序是：

```text
update_actor
→ append_train_traces
→ validate 128 rows
→ save_checkpoint
→ log step metrics
```

对应代码顺序见 `verl/trainer/ppo/ray_trainer.py:1680-1700`。因此，20 步各 40 条 trace 已存在，强烈说明第 20 次 actor update 已经返回，随后进程在内置 val-128 中被 timeout SIGTERM；但它在保存 checkpoint、记录 step-20 metrics 和 finalize trace 之前死亡。这是一条代码路径推断，不是已封存模型证据。**C20 权重已经随进程退出而丢失，不能把 `.partial` trace 冒充为 C20 endpoint。**

## 4. 完整性矩阵

| 合同产物 | 预期 | 实际 | 可用性 |
|---|---:|---:|---|
| 成功训练 run | 2 | 1（B） | B 可用，C 不可用 |
| `global_step_20` checkpoint | 2 | 1（B） | 不能做 checkpoint 对照 |
| 封存 train trace | 2 × 800 | B 800；C 800 partial | 过程分析可用，正式证据仅 B |
| 独立 endpoint eval | 6 | 0 | 不可做正式泛化比较 |
| val eval rows | 2 × 128 | 0 | B 只有训练内置 val，未形成配对包 |
| NQ-test rows | 2 × 128 | 0 | 缺失 |
| multihop rows | 2 × 256 | 0 | 缺失 |
| paired 结果目录 | 3 | 0 | 无正式 paired CI/转移矩阵 |
| lineage / run-index 数据行 | 各 8 | 0 | 最终 lineage 缺失 |
| W&B receipt | 8 | 0 | 只有两份训练 raw offline data |
| final `evidence.sha256` | 1 | 0 | 无最终证据封存 |
| scientific marker | 1 | 0 | 实验未完成 |

合同预期新增 trace 共 2,624 行；目前有 800 行封存 B train trace 与 800 行未封存 C partial train trace，独立 eval 的 1,024 行全部缺失。

## 5. 现有数据能回答什么

### 可以回答

1. B 在 20 步训练中的 reward、EM、搜索次数、KL、entropy、grad norm、invalid、clipping 和 group reward 稀疏性如何变化。
2. B20 checkpoint 是否存在且内容完整。
3. C 为什么失败、失败发生在流水线哪一段、哪些计算已经完成、哪些 artifact 没有持久化。
4. 在同一 R60 parent、同一 seed、同一训练问题顺序下，B/C 的 on-policy 训练行为如何分叉。
5. 当前 cost-aware gated 信号有没有再次诱发旧实验中的 no-search collapse。

### 不能回答

1. B20 与 C20 谁在固定 held-out endpoint 上更好。
2. C 是否在 NQ-test 或 multihop 上以不降 EM 的方式减少搜索。
3. B/C 的正式 paired bootstrap CI、逐题正确性转移和搜索转移矩阵。
4. C20 checkpoint 的泛化、权重差异或稳定性。
5. 可以写进论文摘要的最终 B/C 因果结论。

## 6. 探索性训练轨迹分析

以下统计使用两边各 800 条训练 rollout。它们是不断变化的 on-policy 训练样本，不是固定 endpoint；C 又是未封存 partial run，所以只能用于提出假设。

### 6.1 全程聚合

| 指标 | B | C partial | C − B |
|---|---:|---:|---:|
| train-rollout EM | 42.75% | 44.00% | +1.25 pp |
| mean searches | 2.1925 | 2.1825 | −0.0100 |
| 共同口径 utility `EM - 0.1*S/4` | 0.37269 | 0.38544 | +0.01275 |
| no-search ratio | 2.875% | 3.000% | +0.125 pp |
| invalid-action trajectory ratio | 39.125% | 27.500% | −11.625 pp |
| response-clipped ratio | 62.000% | 48.250% | −13.750 pp |
| mean unfinished generations | 2.7488 | 2.4838 | −0.2650 |
| nonzero-advantage trajectory ratio | 39.375% | 57.500% | +18.125 pp |
| all-wrong group ratio | 38.750% | 31.875% | −6.875 pp |
| mixed group ratio | 39.375% | 47.500% | +8.125 pp |

两边 800/800 个 `(step, group_uid, group_slot)` 可严格配对，数据源计数均为 HotpotQA 515、NQ 285。逐轨迹正确性转移为：

| 状态 | 条数 |
|---|---:|
| 两边都错 | 377 |
| B 对、C 错 | 71 |
| B 错、C 对 | 81 |
| 两边都对 | 271 |

以 160 个 step-question group 为 cluster 的 10,000 次 paired bootstrap，95% CI 为：

- EM 差：`[-1.75, +4.25] pp`
- mean-search 差：`[-0.1200, +0.1025]`
- 共同口径 utility 差：`[-0.01756, +0.04272]`

三个区间都跨 0，因此全程训练 rollout 不能证明 C 明确优于 B。

### 6.2 后五步信号

在 step 16–20 的 200 条 rollout 上，C 相对 B 表现为：EM `+5.0 pp`、mean searches `−0.475`、共同口径 utility `+0.061875`。这是一个值得继续验证的信号，而且 no-search 约 3%，没有重演旧线性成本奖励的 98.44% no-search collapse。

但该窗口是事后选取、只有 40 条/步、数据随 step 变化、C 无 checkpoint 且没有 held-out eval；所以正确表述只能是：**cost-aware gated 在训练后段出现“少搜而不明显损失正确率”的候选趋势，值得补跑正式 endpoint，尚不能下结论。**

### 6.3 B 的独立诊断值

B 的训练内置 val-128（64 NQ + 64 HotpotQA）在 step 20 完成：

| split | EM | mean searches | no-search |
|---|---:|---:|---:|
| NQ | 65.625% | 1.65625 | 0% |
| HotpotQA | 57.8125% | 2.203125 | 0% |
| 合并 | 61.71875% | 1.9296875 | 0% |

这是有效的 B-only 诊断，但因为 C 的对应验证未完成、也没有独立 run/evidence seal，不能替代预注册的 B/C paired val。

## 7. 失败根因与代码风险

直接根因不是磁盘不足、数据损坏或新实例克隆，而是 **C 的 wall-clock timeout 预算过短**：

- B 的实际完整耗时是 17,558 秒；
- C 的固定预算只有 15,625 秒，比 B 的实际耗时短 1,933 秒；
- C 正好在最末端 val-128 中收到 SIGTERM；
- 通用预算映射把 `train:control` 的 20 步预算设为 40 RMB，把 `train:cost_aware_gated` 的 20 步预算设为 25 RMB，见 `scripts/autodl/03_gpu_run.sh:238-267`。

这里还有一个可复现的 artifact 风险：终点 step 的长验证排在 checkpoint 保存之前。即使 20 次 actor update 已在显存中完成，验证超时仍会让整个 C 权重不可恢复。只把 timeout 调大能缓解本次问题，但不能消除同类风险。

## 8. 全流程叙事位置

当前结果应放进下面这条完整故事，而不是孤立成一次“C 挂了”：

1. **早期工程阻塞与首轮成功运行（7 月 19–20 日）**：修复 Qwen3.5/FSDP/offload/GPU gate/关机状态机。首轮 A→R60→B/C 的进程最终成功，真正的科学失败是策略坍缩，不是单纯 OOM。
2. **旧 NQ-only 线性成本奖励坍缩（7 月 20 日）**：C-old 把 no-search 推到 98.44%，EM 从 B 的 17.97% 降到 7.03%。全错 GRPO group 中，线性成本项错误地奖励了“不搜索但答错”。详见 [首轮结果](../search-r1-small-20260720/README.md) 和 [坍缩分析](../../成本感知坍缩分析与改进建议.md)。
3. **correct-only gated 修复（7 月 21 日）**：no-search 回到 0%，EM 恢复到 14.06%，但仍低于旧 B 的 17.97%，也没有节省搜索。详见 [二次实验分析](../../history/成本感知二次实验结果分析.md)。
4. **搜索机会门（7 月 22 日）**：旧 B 的多跳集里几乎全是一搜；仅有的二搜样本又错误、重复且截断，所以继续压搜索没有可利用空间。详见 [search-opportunity gate](../search-opportunity-gate-20260722/analysis_zh.md)。
5. **数据配比调整（7 月 22 日）**：训练数据从 NQ/Hotpot 1:1 改为 37.5%/62.5%，Hotpot comparison/bridge 从 200/120 改为 56/264，以增加可学习二跳链，同时保留 NQ 稳定梯度。
6. **mixed-data grouped probe（7 月 23 日）**：320 条里只有 7 条正确，非法与截断严重；大量 query 是 `query`、`and` 或空串，问题被定位到私有 XML 协议和解析器，而不是只缺奖励。详见 [grouped probe](../grouped-probe-20260723/analysis_zh.md)。
7. **Qwen3.5 native tool adaptation（7 月 23–28 日）**：迁移到 native tool schema，逐步修复 token/mask、答案边界、停止行为、W&B 与 evidence seal。详见 [实现报告](../../qwen35_native_tool_adaptation_implementation_report.md) 和 [两步 smoke](../../qwen35_native_v4_two_step_smoke_analysis.md)。
8. **新 native mixed-data R60（7 月 28–29 日）**：60/60、2,400 条轨迹完成；固定 val EM 从 37.50% 升到 58.59%，平均搜索从 3.117 降到 1.992，且没有 no-search collapse；但后段 clipping/invalid/KL 同步上升。详见 [R60 分析](../../qwen35_native_r60_training_and_trajectory_analysis.md)。
9. **G3 NO-GO（7 月 29 日）**：320 条推理完整，但 clipping 46.88%、invalid 43.75%、clean learnable group 仅 5/64；预注册门禁失败。cost contrast 13/64 只支持另立 post-hoc exploratory B/C，不能改写 G3。详见 [G3 分析](../../qwen35_native_r60_g3_evaluation_and_trajectory_analysis.md)。
10. **本次 native B/C（7 月 31 日）**：B20 完整；C 在最终内置验证中超时，训练证据包未封存。它提供了候选趋势和一个清晰的工程故障点，但没有完成正式 B/C 问题。实验边界见 [执行 handoff](../../qwen35_native_bc_posthoc_execution_handoff.md)。

未来的 SFT40→RL20→B/C 必须继续作为独立路线，不替换、也不回写当前 direct-RL 结果，见 [SFT+RL 后续方案](../../qwen35_native_sft_rl_bc_followup_plan.md)。

## 9. 推荐恢复方案

当前不应重算 R60，也没有必要重训已经校验完整的 B。推荐下一次有卡时采用新的 recovery contract：

1. 冻结并引用现有 B20 checkpoint/digest。
2. 让 C 从 exact R60 parent 独立重跑；不允许从 B 继续训练。
3. C 的训练 timeout 至少按 B 的实测时间加 25%–35% 余量，不再使用 15,625 秒。
4. 把终点 checkpoint 放到长 validation 之前，或至少增加 step 10/15 的可恢复 checkpoint；验证失败不能抹掉已完成的训练权重。
5. C 成功后，再分别运行 B/C × val、NQ-test、multihop 六个独立 eval。
6. 生成三套 paired summary、8 行 lineage/run-index、8 个 W&B receipt、最终 `evidence.sha256` 和 scientific marker。
7. 只有新合同全部 strict check 通过，才撰写正式 B/C 结论。

磁盘当前约 150 GB 中已用 122 GB，剩余约 29 GB。它足以做本次 CPU 分析，并大概率容纳一个新的约 9 GB C checkpoint和紧凑 eval 证据，但余量不宽；恢复训练前应再次做只读空间清单，只删已经证明无引用价值的废弃 checkpoint。

## 10. 最终判定

```text
训练合同：FAILED / INCOMPLETE
B20：COMPLETE AND ANALYZABLE
C20 checkpoint：MISSING / NOT RECOVERABLE
C partial train trace：VALID JSON, UNSEALED, EXPLORATORY ONLY
6 endpoint eval：NOT RUN
3 paired analyses：NOT RUN
正式 B/C 结论：NOT AUTHORIZED BY EVIDENCE
```

本报告没有重跑训练、没有修改远端实验状态、没有删除 checkpoint。新实例仅用于只读终态、hash、目录和日志核验。
