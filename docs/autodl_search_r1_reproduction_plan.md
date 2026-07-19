# Search-R1 小规模成本感知复现完整方案（AutoDL）

> 本文件合并并取代《实验流程改进》和《最大搜索次数》中的执行决策，是后续实现与云端运行的唯一规划依据。两份来源笔记暂时保留，不作为配置真相。用户已批准按本方案完成最小实现；云端重训仍须从新 commit 的 Git 与 CPU 阶段重新开始。

## 1. 项目目标与结论边界

目标是复现 Search-R1 的核心闭环：Qwen3.5 在推理中自主生成 `<search>`，读取 BM25 检索结果后继续推理，再通过 GRPO 和最终答案 EM 奖励学习搜索策略。在公共复现 checkpoint 上，只改变奖励函数是否包含检索成本，形成公平的 B/C 对照。

本项目定位为面向 Agent 算法岗位的 **Search-R1-small 方法复现**，不是论文数值复刻。论文使用 8 张 H100、500 steps、NQ + HotpotQA、dense retriever 和更大 batch；本方案固定为两张 5090 级 GPU、Qwen3.5-2B、NQ 小数据、CPU BM25 和单 seed。可以展示 Agent loop、检索调用、RL 奖励设计、受控实验和成本权衡，但不能宣称完整复现论文多跳能力或统计显著性。

明确不做：PPO、7 个数据集全量评测、dense E5 大索引、模型或 lambda sweep、多 seed 自动化、按 test 选 checkpoint、优化器精确续训、自动硬件探测，以及通用实验 DAG。HotpotQA 小规模混合可作为未来工作，不进入本轮最小实现。

## 2. 最大搜索预算固定为 4

论文 v5 第 16 页明确写道：`The maximum action budget B is set to 4, and we retrieve the top 3 passages by default.` 仓库原始多机 recipe 也使用 `max_turns=4`。此前的 `max_turns=2` 是为 NQ-only 和低成本运行做的人为缩小，并非论文配置。它能运行，但缩窄了 Agent 的查询修正空间，也使成本感知的可比较区间过小，因此本轮废弃该选择。

为避免与 `B / Control` 的模型名称混淆，本文用 `T_max=4` 表示论文中的 action budget：

- 最多执行 4 个允许 `<search>` 的交互轮次，每次返回 top-3 passages。
- 若第 4 轮后轨迹仍未结束，代码再执行 1 次 `do_search=False` 的最终回答生成。
- 因而每条轨迹的真实检索次数 `n_search` 属于 `[0, 4]`，但最坏情况下共有 5 次模型生成。
- `T_max` 是防止无限调用的硬约束；成本奖励是在合法范围内鼓励少而有效搜索的软目标，两者不重复。

不能只把 turn 数改成 4 而保留旧的 2048 rolling prompt 上限。按实际张量拼接，进入最终回答前的最坏上下文为：

```text
max_prompt_length = 1024 + 4 * (256 + 384) = 3584
```

其中 1024 是初始 prompt 上限，256 是每次生成上限，384 是每次 observation 上限。默认固定为 3584；若人工启用 `MAX_RESPONSE_LENGTH=192` 的 OOM 回退，则联动计算为 3328。这样不会把名义上的四轮实现成会静默截断第四轮历史的两轮配置。四个搜索轮次后的最终回答会再使用一次独立生成预算：默认最多 256 tokens，OOM 回退时为 192 tokens。

## 3. 固定实验设计

```text
A / Base：固定 Qwen3.5-2B
└── R / Reproduced：原始 EM 奖励训练 60 steps
    ├── B / Control：从同一 R60，以原始 EM 奖励训练 20 steps
    └── C / Cost-aware：从同一 R60，以成本感知奖励训练 20 steps
```

| 项目 | 固定选择 |
| --- | --- |
| 模型 | `Qwen/Qwen3.5-2B` post-trained，revision `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| 算法 | GRPO，group size 5 |
| 数据 | NQ train-512、val-64、test-128，seed 42 |
| 检索 | Wikipedia 2018 CPU BM25，top-k=3，`T_max=4` |
| 硬件 | `GPU_COUNT=2`，两张 5090 级 GPU；不自动检测或改参 |
| 正式终点 | R=`global_step_60`，B/C=`global_step_20` |
| 统一评测 | A/R/B/C 均使用 test-128、`T_max=4`、评测 `lambda=0.10` |
| 总硬预算 | 300 元，其中 GPU 分项上限合计 250 元 |

每个训练 step 使用 8 个 prompts，每个 prompt 采样 5 条轨迹，即 40 条 agent trajectories 和 1 次 actor update。R/B/C 合计 100 次更新、4000 条训练轨迹。相较旧的 batch 4、group 8 配置，每步轨迹数从 32 增至 40，理论工作量增加 25%，因此不能宣称新配置训练更快。R、B、C 只在固定终点保存正式权重；val-64 只作终点记录，不用于选择 checkpoint。

B/C 必须读取同一个 R60 路径及 digest，使用相同数据、shuffle、seed、batch、group size、学习率、warmup、KL、搜索预算、检索器、token 上限和硬件。唯一科学变量是训练奖励：B 的 `cost_lambda=0`，C 的 `cost_lambda=0.10`。两条分支都从 R 的模型权重创建新的 Adam、warmup 和数据加载状态，因此应称为 **stage-2 受控分叉/二阶段微调**，不是 optimizer 精确断点续训。

## 4. 奖励函数与公平评测

设最终答案精确匹配为 `r_em in {0, 1}`，真实执行检索次数为 `n_search`：

```text
R/B 原奖励：       r = r_em
C 成本感知奖励：   r = r_em - 0.10 * n_search / 4
统一评测 utility： u = EM - 0.10 * avg_searches / 4
```

| 实际检索次数 | C 中检索成本 | 答对时奖励 |
| ---: | ---: | ---: |
| 0 | 0.000 | 1.000 |
| 1 | 0.025 | 0.975 |
| 2 | 0.050 | 0.950 |
| 3 | 0.075 | 0.925 |
| 4 | 0.100 | 0.900 |

搜索后答对的最低奖励仍为 0.90，显著高于不搜索但答错的 0，因此目标不是禁止搜索，而是减少无收益调用。代码只统计真正发给检索服务的请求，并用随 batch 重排的 `executed_search_count` tensor 计算成本；无效格式、未执行请求和最后一次禁止搜索的生成都不增加计数。

训练期 R/B 的 reward 只反映 EM，C 的 reward 已包含成本。为了公平，最终 A/R/B/C 一律重新按同一个 `/4` 公式报告 EM、平均检索次数、no-search ratio 和 utility，不能直接横向比较不同定义的训练 reward。

## 5. Qwen3.5 与两卡最小配置

仓库原依赖不能直接支持 Qwen3.5 与 RTX 5090，保留现有最小兼容策略：

1. 使用 AutoDL PyTorch 2.8.0 / Python 3.12 / CUDA 12.8 镜像，Conda 环境名沿用 `llmdevelop`，在持久盘创建隔离 venv。
2. 使用 HF rollout 而不是旧版 vLLM 适配层；PyTorch SDPA、bf16、temperature 1.0、top-p 1.0。
3. 关闭 `flash_attention_2` 和 `use_remove_padding`，保留 retrieved-token loss masking、FSDP、gradient checkpointing 和 CPU offload。
4. 默认 batch=8、group=5、actor mini-batch=40；actor/log-prob micro-batch 在两卡时为 2，rollout micro-batch=1。

| 参数 | 默认值 |
| --- | ---: |
| `max_start_length` | 1024 |
| 每次 `max_response_length` | 256 |
| 每次 `max_obs_length` | 384 |
| `max_turns` | 4 |
| `max_prompt_length` | 3584（由上述参数计算） |
| learning rate / warmup | `1e-6` / 28.5% |
| KL coefficient | `0.001` |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` |

不先试单卡，不在运行中自动改参数。只有新的两卡 gate 明确 OOM，才由用户决定是否先让 R/B/C 与全部评测共同回退到 batch=4；仍 OOM 时再将 response 改为 192，并把 prompt 派生为 3328。`T_max=4`、group size=5 和固定步数不降级。

## 6. 三阶段执行与旧实验迁移

所有环境、资产、状态和产物放在持久盘 `/root/autodl-tmp/search-r1/`，严格按 **Git -> CPU -> GPU** 执行。

### 阶段一：Git 固定新实现

用户批准本方案后才修改代码、运行测试、提交和 push。云端 checkout 必须固定到新的 40 位 commit、detached HEAD 且工作区干净。GitHub 直连失败时继续使用完整 Git bundle，不用不可信的临时源码副本。

旧的两轮 GPU attempt `20260719T135124Z-1441-19817` 已按用户要求 TERM，外层终态为 `failed/143`。它的 1-step gate 日志和 R 的已完成 step 只保留为 B=2 工程诊断证据；没有 `global_step_60`，不得作为新实验的 R，也不得进入最终 A/R/B/C 表。

### 阶段二：CPU 无卡准备

新 commit 会使旧 `cpu_handoff.json` 失效，必须重新封存。模型、语料、索引、环境和 NQ 数据可在 hash 校验后复用，不重复下载或重建大文件。CPU 阶段不运行 `nvidia-smi`，不根据无卡时的机器规格调整训练参数。

CPU 阶段需要重新完成：

1. 校验 commit、依赖锁、Qwen 模型、BM25 索引、wiki corpus 和 NQ 数据。
2. 组合 1/2 GPU 下 smoke、R、B、C 与 A/R/B/C eval 的 16 份 Hydra 配置。
3. 显式断言所有配置均为 `max_turns=4`、top-k=3，并满足 prompt-length 派生公式。
4. 去除 lambda 与实验名后，全量比较 B/C resolved config，确保没有第二个科学变量。
5. 运行 tokenizer、真实 BM25、奖励、结果汇总和运行时测试，再发布自校验 handoff。

### 阶段三：两卡 GPU 离线运行

GPU phase 只接受与新 commit 匹配的 handoff，然后严格串行执行：

1. 启动本机 CPU BM25 服务并通过 health check。
2. Gate 1：A -> smoke 1 step，验证四轮 rollout、检索、奖励、反向、终点 val 和保存。
3. Gate 2：从 Gate 1 checkpoint -> control 1 step，验证 actor/ref 子 checkpoint 重载。
4. 两个 gate 成功后精确删除两个 `global_step_1`，保留日志、终态和清理证据。
5. A -> R60；验证并只保留 `global_step_60`。
6. 同一 R60 -> B20 与 C20；分别只保留 `global_step_20`。
7. 依次评测 A、R、B、C，并生成结果、lineage 和 checksum。

任一 gate、正式训练或评测失败都停止流水线，不自动重试、不复用部分 checkpoint、不缩短步数、不切换模型，也不因科学负结果重跑。

## 7. 用户批准后的最小实现清单

只修改与四轮一致性和本次实测可靠性直接相关的文件：

- `scripts/autodl/train_small_grpo.sh`：固定 `MAX_TURNS=4`，由 start/response/observation 派生 prompt 上限，并让两个 gate、R/B/C 和四路 eval 共用。
- `scripts/autodl/results.py`：把统一 utility 的旧 `/2` 改为具名 `MAX_SEARCHES=4`，继续严格拒绝不一致的评测记录。
- `scripts/autodl/02_cpu_prepare.sh`：新增 resolved-config 的 turn、top-k 和 prompt-length 合同检查。
- `tests/test_cost_aware_reward.py`：覆盖 0/1/4 次搜索在 `T_max=4` 下的成本及 batch reorder。
- `scripts/autodl/tests/test_results.py`：把合法 fixture 全部更新为 `/4`，保留故意不一致的负测试。
- `scripts/autodl/README.md`：同步四轮参数、三阶段命令、失败处理和人工确认计费说明。

核心 `RewardManager` 已从 `config.max_turns` 取得归一化分母，agent loop 也已支持任意正整数 turns，因此不修改奖励、generation 或运行时状态机。根目录通用 `train_ppo.sh`/`train_grpo.sh` 不属于本 AutoDL 入口，也不改。为避免过度设计，不新增统一配置服务、恢复 DAG 或自动调参器；重复常量由 CPU resolved-config gate 防漂移。本次人工停止暴露的嵌套进程组信号问题单独记录，不混入改变科学配置的 commit；下一次云端运行仍使用绑定 exact attempt 的 watchdog 保证终态后关机。

## 8. 验证与重新准入

实现后先本地运行：

```bash
python -m pytest -q
python -m unittest -v scripts.autodl.tests.test_results
bash scripts/autodl/tests/test_runtime.sh
bash -n scripts/autodl/train_small_grpo.sh scripts/autodl/02_cpu_prepare.sh scripts/autodl/03_gpu_run.sh
yapf --diff scripts/autodl/results.py tests/test_cost_aware_reward.py scripts/autodl/tests/test_results.py
```

还要静态检查 canonical 文档和生产入口中不存在旧的 `max_turns=2`、utility `/2`、每次扣 0.05 等残留。新 commit push 后，云端必须依次重跑 Git seal 和 CPU handoff；旧 B=2 的 gate 成功不能替代新 B=4 gate。

GPU 准入以两级 gate 为准：必须同时证明四轮配置成功组合、无 OOM/Traceback、真实检索可用、一步反向完成、checkpoint 能保存并重载。gate 只验证工程可运行，不作为科学结果。

## 9. 训练资料与面试证据

每个 attempt 必须在持久盘保留：

- `train.log`：每个完成 step 的聚合指标。
- WandB offline run：原精度 history 和 config，不依赖联网账号。
- `resolved-config.yaml`、`run.env`、terminal、exit code、commit 与 handoff digest。
- R/B/C checkpoint digest 与父子 lineage。
- 最终 `results.csv`、`results.md`、`lineage.tsv` 和 `comparison.sha256`。

可恢复的训练曲线至少包括 `actor/pg_loss`、`actor/kl_loss`、entropy、grad norm、learning rate、reward/EM、平均检索次数、no-search ratio、派生 utility、step time、GPU-hours 和人民币。GRPO 的 policy loss 来自新的 on-policy 小批次与组内标准化优势，可能围绕 0 正负波动，不应在面试中声称它必须单调下降；主科学证据应是 B/C 的 EM、search count 和统一 utility，以及 A/R 的 sanity check。

GPU 完成并关机后，再在不挂 GPU 的 CPU 阶段从原始日志导出 CSV 和曲线图，不让绘图延长付费 GPU 时间。当前保存的是 per-step 聚合指标，不是每条 trajectory 的完整原始 token；报告中应如实说明这一证据粒度。

## 10. 评测与成功标准

最终表固定为四行：

| Model | 作用 | Test EM | Avg Searches | No-search Ratio | Utility (`lambda=0.10`, `/4`) |
| --- | --- | ---: | ---: | ---: | ---: |
| A / Base | 原始模型 | ... | ... | ... | ... |
| R / Reproduced | Search-R1-small 复现点 | ... | ... | ... | ... |
| B / Control | 原奖励二阶段控制组 | ... | ... | ... | ... |
| C / Cost-aware | 成本奖励实验组 | ... | ... | ... | ... |

工程成功要求：R60、B20、C20 三个固定终点存在且 digest/lineage 正确；四路 test-128 完成；比较文件 hash 全部通过。科学解释分两层：A vs R 仅检查缩小版 RL 是否改变搜索/回答行为，主结论来自 B vs C 是否在尽量保留 EM 的同时降低搜索并提高统一 utility。

若 R 没有优于 A、B 本身几乎不搜索、C 降低搜索但同时明显损害 EM，或 C 的 utility 不提升，都作为有效负结果如实报告；不追加单边训练、lambda sweep 或 checkpoint 挑选来美化结论。

## 11. 预算、存储与安全停机

| 付费项 | 硬上限（元） |
| --- | ---: |
| Base 1-step gate | 15 |
| checkpoint-load 1-step gate | 15 |
| R：60 steps | 100 |
| B：20 steps | 40 |
| C：20 steps | 40 |
| A/R/B/C 四次评测 | 10 x 4 = 40 |
| **GPU 分项合计** | **250** |
| CPU、100 GB 存储和阶段开销预留 | 50 |
| **项目总硬上限** | **300** |

四轮最坏生成次数从 3 次增至 5 次，旧 B=2 的 ETA 不能直接复用。默认 batch 8、group 5 每步 40 条轨迹，比旧 batch 4、group 8 的 32 条多 25% 工作量，不能据此承诺更快。按两卡 5.76 元/小时粗略规划为 15-25 小时、约 86-144 元；最终只以新 gate 和正式 step 的实测为准。现有分项 timeout 已留有远大于该区间的硬余量，因此暂不扩到 350 元。预算是 fail-closed 上限：超时则失败，不自动降 turns、改 batch 或加钱。

`T_max=4` 不改变模型参数量或 checkpoint 大小，只增加计算与少量日志。100 GB 盘仍预计使用 55-70 GB：固定环境、模型、语料和索引约 30-35 GB，R/B/C 权重约 12-18 GB，其余留给日志、Ray/WandB 和状态；无需再次扩容。

仓库训练脚本不做无条件关机。每次 GPU attempt 单独安装绑定 exact attempt、commit、持久盘、锁和 `/usr/bin/shutdown` hash 的 watchdog。它必须等待完整 terminal、原始 exit code、日志 sentinel 和 durable sync；成功时还要执行 `sha256sum -c comparison.sha256` 并验证 `gpu.ok == sha256(comparison.sha256)`，失败或人工 TERM 时也只有在终态完整后才请求 guest shutdown。锁冲突、状态不完整或校验失败一律保持开机。guest shutdown 只代表已派发请求，最终仍由用户在 AutoDL 控制台确认实例停止且不再计费。

## 12. 批准后的执行顺序

1. 用户审阅并批准本文件，明确接受 `T_max=4`、batch 8、group size 5、默认 prompt=3584、NQ-only 边界和 300 元硬上限。
2. 实现第 7 节的最小改动与测试，不触碰无关上游代码。
3. 提交并 push 新 commit，记录完整 SHA；不提交本地论文 PDF 和来源笔记，除非用户另行要求。
4. 用户以无 GPU 模式开机后，通过 SSH 更新固定 checkout 并重跑 CPU handoff；复用已校验的大资产。
5. CPU 成功且关机后，用户挂载两张 GPU 再开机；运行两级 B=4 gate 和完整 R/B/C/评测流程。
6. 终态持久化后由 watchdog 请求关机；用户在控制台确认停止计费。
7. 以后续 CPU 会话导出训练曲线、整理实验表和简历/面试说明。
