# Search-R1 小规模成本感知复现方案（AutoDL）

## 1. 目标与结论边界

目标是复现 `2503.09516v5` 的核心方法：模型在推理中自主生成 `<search>`，读取检索结果后继续推理，并通过 GRPO 与最终答案 EM 奖励学习搜索策略。在此基础上只修改奖励函数，比较原奖励与成本感知奖励。

本项目是面向求职展示的 **Search-R1-small 方法复现**，不是论文数值复刻。论文使用 8 张 H100、500 步、E5 dense retriever 和更大 batch；本方案固定为两张 5090 级 GPU、BM25、小数据与单个 seed。`A -> R` 只验证缩小版 Search-R1 能否学到搜索行为，主实验结论仅来自 `B vs C`。单 seed、test-128 只能报告观察到的差异，不能宣称统计显著性。

明确不做：PPO、7 个数据集全量评测、dense E5 大索引、超参或 lambda sweep、多 seed 自动化、按验证集挑 checkpoint、优化器精确续训、自动硬件探测和通用实验 DAG。

## 2. 固定实验设计

```text
A / Base：固定 Qwen3.5-2B
└── R / reproduced：原奖励训练 60 steps
    ├── B / control：从 R60、原奖励训练 20 steps
    └── C / cost_aware：从同一 R60、成本感知奖励训练 20 steps
```

| 项目 | 固定选择 |
| --- | --- |
| 模型 | `Qwen/Qwen3.5-2B` post-trained，revision `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| 算法 | GRPO，group size 8 |
| 数据 | NQ train-512、val-64、test-128，seed 42 |
| 检索 | CPU BM25，Wikipedia 2018，top-k=3，最多执行 2 次搜索 |
| 硬件 | 默认 `GPU_COUNT=2`，两张 5090 级 GPU；不自动检测 |
| 正式终点 | R=`global_step_60`，B/C=`global_step_20` |
| 总预算 | 300 元硬上限，其中 GPU 分项硬上限合计 250 元 |

R、B、C 都只在固定终点保存正式权重；val-64 只在终点留下运行记录，不用于选择 checkpoint。B/C 必须读取同一个 R60 路径及其 SHA-256，且除 `cost_lambda` 与实验名外配置相同。

R checkpoint 只包含模型权重。启动 B/C 时会分别从 R 重建 actor 与 KL reference，并重新创建 Adam、学习率 warmup、数据加载与 shuffle 状态。因此 B/C 是从同一权重出发的 **stage-2 受控分叉**，不是 R 的精确连续续训；两条分支都以 seed 42 独立重置上述状态。

## 3. Qwen3.5 最小兼容策略

仓库原依赖不能直接支持 Qwen3.5 与 RTX 5090。实现保持 agent loop 不变，只做以下兼容处理：

1. 使用 AutoDL 的 PyTorch 2.8.0 / Python 3.12 / CUDA 12.8 镜像和独立锁定依赖；仓库用 `pip install -e . --no-deps` 安装。
2. 设置 `actor_rollout_ref.rollout.name=hf`，绕过旧版 vLLM 适配层。
3. 使用 PyTorch SDPA，关闭 `flash_attention_2` 和 `use_remove_padding`。
4. 保留 retrieved-token loss masking、GRPO、搜索环境、FSDP/offload 与奖励入口。

HF rollout 比 vLLM 慢，但对本次 2B 小规模实验减少了兼容改造。任何 gate 失败都先停卡修复，不静默更换模型、算法或数据。

## 4. 奖励与公平性

设最终答案精确匹配奖励为 `r_em in {0, 1}`，轨迹实际执行搜索次数为 `n_search`，上限 `B=2`：

```text
原奖励：      r = r_em
成本感知奖励：r = r_em - 0.10 * (n_search / B)
```

每次实际搜索扣 0.05，最多扣 0.10。代码只统计真正发给检索服务的调用，并以随 batch 重排的 `executed_search_count` tensor 传入奖励，避免把未执行请求或错位样本计入成本。

B 与 C 使用相同的 R60、训练数据、seed、20 steps、检索器、采样和显存参数，唯一科学变量是训练奖励中的 `cost_lambda`：B 为 0，C 为 0.10。最终对 A/R/B/C 全部在同一 test-128 上评测，并统一以 `lambda=0.10` 报告 utility，同时分别报告纯 EM 与搜索次数。

## 5. 三阶段执行流程

所有不可变资产、环境、状态与产物位于持久盘 `/root/autodl-tmp/search-r1/`。正常操作仍严格按 **Git -> CPU -> GPU** 进行。

### 阶段一：Git 成功

`scripts/autodl/01_git.sh` 在首次 clone 时将仓库固定到一个已推送的 40 位 commit，使用 detached HEAD，保存远端、HEAD 与工作区状态；只有 commit 正确且工作区干净才发布成功标记。已有 checkout 指向其他 commit 时脚本会失败关闭，不在原地偷偷切换；当前升级或 GitHub 直连失败时，由受控 SSH 操作用完整 Git bundle 导入同一可信 commit 并重新封存 checkout。

本次实验流程与代码形成新 commit 后，旧 `cpu_handoff.json` 必然失效。必须先让云端 checkout 指向新 commit，再重新执行 CPU 阶段；不能让旧 handoff 放行新代码。

### 阶段二：CPU 无 GPU 准备

`scripts/autodl/02_cpu_prepare.sh` 在联网 CPU 实例中：

1. 复用 Conda `llmdevelop` 的镜像 Python，在持久盘创建 train/retriever venv。
2. 下载并固定 Qwen3.5-2B、BM25 索引与 wiki-18 corpus，流式生成约 14.4 GB JSONL 和 `uint64` 偏移表。
3. 生成互不重叠的 NQ 512/64/128 子集并运行 tokenizer、真实 BM25、数据和奖励测试。
4. 组合并检查 1/2 GPU 下的 `smoke`、`reproduce`、`control`、`cost_aware` 训练配置及四种评测配置。
5. 将 commit、依赖、资产、数据和 resolved config 封存到自校验的 `manifests/cpu_handoff.json`。

新 commit 要重新封存 handoff，但已经校验的模型、语料、索引、数据和环境可以从持久盘复用，无需为了改脚本重复下载。CPU 阶段不运行 `nvidia-smi`，也不根据临时 CPU 规格改训练参数。

### 阶段三：GPU 离线训练

`scripts/autodl/03_gpu_run.sh` 先离线校验 handoff，再依次执行：

1. 启动本机 CPU BM25 服务。
2. Gate 1：从 A 运行 `train smoke 1`，验证 rollout、搜索、反向更新和 step-1 checkpoint。
3. Gate 2：以 Gate 1 权重运行 `train control 1 <checkpoint>`，验证 actor/ref 能从子 checkpoint 加载并完成更新。
4. 两个 gate 都成功后，按精确路径安全删除其权重目录，但保留日志、终态和清理证据。
5. 从 A 运行 `train reproduce 60`，固定取得 R60。
6. 分别从同一 R60 运行 `train control 20` 与 `train cost_aware 20`，固定取得 B20/C20。
7. 运行 `eval base`、`eval reproduced`、`eval control`、`eval cost_aware`，汇总 test-128 四路结果与 lineage/digest。

正式只保留 R60、B20、C20 三个 checkpoint；A 使用 CPU 阶段固定的原始模型，不再复制。任何训练阶段失败都不会自动重试或改选较早 checkpoint；保留失败 attempt 后人工处理。

## 6. 缩小训练配置

| 参数 | 值 |
| --- | --- |
| `data.train_batch_size` | 4 prompts |
| `actor_rollout_ref.rollout.n_agent` | 8，即每步 32 条轨迹 |
| actor mini batch | 32 |
| actor/log-prob micro batch | `GPU_COUNT`（两卡时为 2） |
| HF rollout micro batch | 1 |
| `max_start_length` / `max_response_length` | 1024 / 256 |
| `max_obs_length` / `max_prompt_length` | 384 / 2048 |
| `max_turns` / retriever | 2 / BM25 top-k=3 |
| learning rate / warmup | `1e-6` / 10% |
| KL coefficient | `0.001` |
| rollout | HF, bf16, temperature 1.0, top-p 0.95 |
| memory | gradient checkpointing + parameter/gradient/optimizer offload |
| 正式步数 | R 60；B 20；C 20 |

已验证配置为两卡、batch 4、response 256、group size 8，并固定 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。不再为本次运行先试单卡。只有两卡 gate OOM 时才人工按顺序将 `TRAIN_BATCH_SIZE=2`、`MAX_RESPONSE_LENGTH=192`，且 R/B/C 必须共同使用同一组回退参数。

## 7. 评测与成功标准

最终表包含 A/R/B/C 的 NQ EM、平均搜索次数、不搜索比例、统一 `lambda=0.10` 的 utility，以及训练耗时、GPU-hours、人民币和 checkpoint lineage。

工程成功标准是 R/B/C 固定终点存在且 digest/父子关系通过校验，四路 test-128 均完成。`A vs R` 仅作为 Search-R1-small sanity check；主结果是 B vs C 是否在尽量保留 EM 的同时减少实际搜索。若 B 本身几乎不搜索，或 C 没有改善 utility，该负结果也应如实报告，不追加 lambda sweep 美化结果。

## 8. 预算、存储与停机

| 付费项 | 硬上限（元） |
| --- | ---: |
| Base 1-step gate | 15 |
| checkpoint-load 1-step gate | 15 |
| R：60 steps | 100 |
| B：20 steps | 40 |
| C：20 steps | 40 |
| A/R/B/C 四次评测 | 10 x 4 = 40 |
| **GPU 分项合计** | **250** |
| CPU、100 GB 存储和阶段间开销预留 | 50 |
| **项目总上限** | **300** |

脚本按 `AUTODL_PRICE_PER_HOUR` 将每项人民币上限换算为 timeout；当前两卡整机 5.76 元/小时，启动时仍以控制台实时整机价为准。以上是防失控的硬上限，不是预计实际支出，也不保证一定跑满目标步数。

100 GB 持久盘预计最终使用约 55-70 GB：固定环境、模型、BM25 与语料约 30-35 GB，R/B/C 三份权重约 12-18 GB，其余留给日志、Ray/WandB 缓存和 attempt 元数据。两个 smoke 权重会在 gate 成功后删除，因此当前无需继续扩容；若异常 attempt 留下大文件，应先核对路径与终态再人工清理，不能盲删。

三个入口均通过 `nohup + setsid` 后台运行，持久化原始 exit code、终态和日志，并用非阻塞锁避免并发写盘。脚本本身不执行 guest shutdown。无论成功、失败或人工停止，都应先通过 SSH 确认终态，再在 AutoDL 控制台确认实例已停止且不再计费；guest 内关机不等于云端计费已经停止。

## 9. 操作范围

远端操作统一使用已配置 SSH 别名与 `ssh-skill`，不直接运行裸 `ssh/scp`。用户的正常入口只有 `01_git.sh`、`02_cpu_prepare.sh`、`03_gpu_run.sh`；训练内部 mode 固定为 `smoke`、`reproduce`、`control`、`cost_aware`，评测 mode 固定为 `base`、`reproduced`、`control`、`cost_aware`。

不新增 Docker/Kubernetes、多机调度、硬件扫描器、optimizer resume、自动 checkpoint 选择、复杂恢复框架或超参 sweep。实现与执行顺序保持最小：固定代码与资产，完成两次 gate，再运行一个复现阶段和两个受控分支。
