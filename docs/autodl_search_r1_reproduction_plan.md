# Search-R1 小规模成本感知复现方案（AutoDL）

## 1. 目标与边界

目标是复现 `2503.09516v5` 的核心方法：模型在推理中自主产生 `<search>`，读取检索结果后继续推理，并通过 GRPO 和最终答案 EM 奖励学习搜索策略。在此基础上只增加一项成本感知奖励，比较“原始奖励”和“成本感知奖励”训练出的两个模型。

这是**方法复现**，不是论文数值复刻。论文使用 8 张 H100、500 步、E5 dense retriever 和更大 batch；本方案受 1-2 张 5090 及 300 元预算约束，不追求论文绝对 EM。最终交付包括两个选定 checkpoint、固定测试集结果、训练日志和成本-效果对比表。

明确不做：PPO、7 个数据集全量评测、dense E5 大索引、多机训练、超参搜索、自动硬件探测、复杂状态机和自动重试。

## 2. 固定实验选择

| 项目 | 选择 | 原因 |
| --- | --- | --- |
| 模型 | `Qwen/Qwen3.5-2B`（post-trained） | 比 Base 更容易在少量步骤内学会格式和工具调用 |
| 算法 | GRPO，group size 8 | 稀疏 EM 下组内估计更稳定，且不需要 critic |
| 数据 | NQ：训练 512、验证 64、测试 128 | 足够展示趋势，且控制 rollout 费用 |
| 检索 | CPU BM25，Wikipedia 2018，top-k=3 | 官方索引约 2.3GB，不占训练显存；避免 60-132GB 的 E5 资产 |
| 硬件 | 默认 1x5090 32GB | 只有单卡一步训练仍 OOM 时才人工切到 2 卡 |
| 随机种子 | 42 | 两个实验使用完全相同的数据顺序和采样设置 |
| 总预算 | 300 元硬上限 | 包含 CPU、GPU smoke、两次训练和最终评测 |

模型固定到 Hugging Face revision `15852e8c16360a2fea060d615a32b45270f8a8fc`。若实现时更换 revision，两个实验必须一起重做。

## 3. Qwen3.5 兼容策略

当前仓库不能仅修改 `BASE_MODEL`：`requirements.txt` 限制 `transformers<4.48`、`vllm<=0.6.3`，而 Qwen3.5 需要新版 Transformers/vLLM；RTX 5090 也需要 CUDA 12.8 时代的 PyTorch。

最小改法是不重写 agent loop，而是：

1. 新增独立的 `requirements-autodl.lock`，固定可支持 5090 的 PyTorch cu128、Qwen3.5 所需 Transformers commit，以及与旧 veRL 代码兼容的依赖；安装本仓库时使用 `pip install -e . --no-deps`，不安装旧 vLLM。
2. 设置 `actor_rollout_ref.rollout.name=hf`，用现有 `HFRollout` 绕开仓库内只支持 vLLM 0.3-0.6 的适配层。
3. 使用 PyTorch SDPA，关闭 `flash_attention_2` 和 `use_remove_padding`；后者当前只登记了 Qwen2。
4. 保留 retrieved-token loss masking、GRPO、搜索环境和奖励入口不变。

HF rollout 会比 vLLM 慢，但对 2B/小数据更稳妥，也显著减少兼容改造。GPU 阶段的第 1 个训练 step 就是付费兼容门：若在 30 元 smoke 预算内仍不能完成，不继续烧卡，先修兼容问题；不静默换模型或算法。

## 4. 双实验与成本奖励

设最终答案精确匹配奖励为 `r_em in {0, 1}`，一次轨迹实际执行的搜索次数为 `n_search`，最大实际搜索次数 `B=2`。

```text
baseline:   r = r_em
cost-aware: r = r_em - 0.10 * (n_search / B)
```

因此每次搜索成本为 0.05，最多扣 0.10，不会压过正确答案的主要信号。由于 top-k 和 observation 长度固定，搜索次数同时近似表示检索延迟与额外上下文 token 成本。本次不做 lambda sweep。

实现时要修正一个细节：`generation.py` 当前会把最终 `do_search=False` 轮次里的搜索请求也计入 `valid_search_stats`。应只统计真正发给检索服务的调用，并把 `executed_search_count` 写成随 batch 重排的 tensor；不能用不会随 `_balance_batch` 重排的 `meta_info` 做逐样本奖励。`RewardManager` 从该 tensor 计算成本项，验证阶段同时记录纯 EM、平均搜索次数和 utility。

公平性约束：两个实验都从同一原始 Qwen3.5 revision 开始，不能让 cost-aware 接着 baseline 训练；除 `cost_lambda` 和实验名外，数据 ID、seed、步数、检索器和所有训练参数完全相同。

## 5. 三阶段执行流程

所有内容放在同一持久盘：

```text
/root/autodl-tmp/search-r1/
  checkout/        # 固定 Git commit 的代码
  envs/            # train 与 retriever 两个环境
  cache/           # Hugging Face、pip 缓存
  data/            # BM25 索引及固定 NQ 子集
  runs/            # smoke、baseline、cost_aware
  logs/            # 每阶段日志和 exit code
  manifests/       # commit、模型 revision、数据 ID、依赖版本
```

### 阶段一：Git 成功

`scripts/autodl/01_git.sh` 只完成以下动作：

1. 在 `/root/autodl-tmp/search-r1/checkout` clone 用户 fork。
2. checkout 一个明确的 40 位 commit SHA，不跟随变化中的 branch tip。
3. 保存 `git remote -v`、`git rev-parse HEAD` 和 `git status --short` 到 `logs/git.log`。
4. HEAD 正确且工作区干净时写 `manifests/git.ok`，否则直接退出。

这一阶段不装环境、不下载模型、不检查 CPU/GPU 规格。若仓库为私有仓库，使用临时凭据，不把 token 写入 URL、脚本或日志。

### 阶段二：CPU 无 GPU 准备

`scripts/autodl/02_cpu_prepare.sh` 在持久盘完成所有联网和 CPU 工作：

1. 创建训练环境和独立 BM25 retriever 环境；训练环境安装锁定依赖，retriever 环境安装 Pyserini、FastAPI 和 Java 运行依赖。
2. 下载并固定 Qwen3.5-2B 到 `cache/huggingface`，不在 GPU 阶段重新下载。
3. 下载固定 revision 的 `PeterJinGo/wiki-18-bm25-index` 和 `PeterJinGo/wiki-18-corpus`；不下载 E5 flat/HNSW 索引。
4. BM25 索引只存文档 ID。脚本从 corpus 的单文件 TAR 中流式写出约 14.4 GB JSONL 和 `uint64` 行偏移表，逐行验证 `id == 0-based row`，运行时按命中 ID 随机读取，不全量载入内存。
5. 运行 NQ 预处理，以 seed 42 输出互不重叠的 `train_512.parquet`、`val_64.parquet` 和 `test_128.parquet`，并保存样本 ID 清单。
6. 仅做 CPU 可完成的最小验收：依赖可导入、模型 tokenizer 能处理一条 prompt、三份数据行数正确、真实 BM25 加外部 corpus 能返回一条结果、奖励函数的手工样例通过。
7. 将 corpus revision、源 SHA-256、模型、索引、解包语料、偏移表、依赖 freeze 和数据 manifest 一并封存，成功后写 `manifests/cpu.ok`。

这里不调用 `nvidia-smi`、不探测显卡数量、不根据机器规格自动改参数。完成后关闭无 GPU 实例，并在 AutoDL 控制台确认 GPU 实例继续挂载同一数据盘。

### 阶段三：GPU 直接训练

`scripts/autodl/03_gpu_run.sh` 负责付费阶段。显卡数量由 `GPU_COUNT=1` 明确指定，不自动检测。

1. 用 retriever 环境在 CPU 后台启动 BM25 服务，日志写入持久盘。
2. 运行与正式配置完全相同的 1-step smoke；必须完成一次 rollout、一次反向更新和一次 checkpoint 保存。
3. 从原始模型独立运行 `baseline` 60 steps。
4. 再次从原始模型独立运行 `cost_aware` 60 steps。
5. 每 20 steps 保存并在固定 val-64 上验证。baseline 按最高 val EM 选 checkpoint；cost-aware 按最高 `EM - 0.10 * n_search / 2` 选 checkpoint，平局取搜索更少者。
6. 在从未参与选择的 test-128 上统一评测两个 checkpoint，并生成一张结果表。

长任务使用 `nohup + setsid`，stdin 指向 `/dev/null`；日志和原始 exit code 写入 `runs/<variant>/`。不设计自动重试，也不保存优化器状态；中断后从原始模型人工重跑该实验。

## 6. 缩小训练配置

| 参数 | 值 |
| --- | --- |
| `algorithm.adv_estimator` | `grpo` |
| `data.train_batch_size` | 4 个 prompt |
| `actor_rollout_ref.rollout.n_agent` | 8，即每步 32 条轨迹 |
| actor mini/micro batch | 32 / 1 |
| log-prob micro batch | 1 |
| `max_start_length` | 1024 |
| 每轮 `max_response_length` | 256 |
| `max_obs_length` | 384 |
| `max_prompt_length` | 2048 |
| `max_turns` | 2 次可执行搜索，之后强制最终回答轮 |
| retriever | BM25, top-k=3 |
| learning rate | `1e-6`，10% warmup |
| KL coefficient | `0.001` |
| rollout | HF, bf16, temperature 1.0, top-p 0.95 |
| memory | gradient checkpointing + parameter/gradient/optimizer offload |
| save / validation | 每 20 steps |
| total steps | 每个实验 60 |

group size 从 4 增至 8 后，每步生成量接近翻倍；HF rollout、log-prob 和 actor 更新仍按 micro-batch 1 串行处理，因此主要增加耗时，峰值显存只小幅增加。单卡是否可行仍以 1-step smoke 为准。

若单卡 smoke OOM，按顺序只调整 `train_batch_size: 4 -> 2`、`max_response_length: 256 -> 192`。仍失败才在控制台改为 2 卡；两个正式实验必须使用相同卡数和配置。

## 7. 评测与成功标准

最终表至少包含：

| 模型 | NQ EM | 平均搜索次数 | 不搜索比例 | utility | GPU 小时 | 实际人民币 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |

工程成功标准是三阶段可重复完成且两个 checkpoint 可评测；科学目标是 cost-aware 相比 baseline 明显减少搜索，并尽量将 EM 降幅控制在 3 个百分点以内。若没有达到，该负结果也应如实报告，不通过额外调 lambda 美化结果。

如果 baseline 的平均实际搜索次数接近 0，则成本实验没有可识别性；此时先排查输出格式或 rollout，而不是解释为成本奖励有效。

## 8. 预算与停机规则

| 阶段 | 上限（元） |
| --- | ---: |
| Git + CPU 准备 + 存储 | 20 |
| GPU 兼容与 1-step smoke | 30 |
| baseline | 100 |
| cost-aware | 100 |
| checkpoint 评测与整理 | 30 |
| 预留 | 20 |
| **总计** | **300** |

按 AutoDL 页面实时单价 `p` 计算每段最长可运行时间 `预算 / p`；双卡时仍遵守同一人民币上限，而不是把上限翻倍。group size 8 的训练量接近原配置两倍，单实验 100 元是预算硬上限，不代表一定能跑满 60 steps。达到阶段上限时保留已有 checkpoint 并停止，修复后从原始模型重跑失败实验，不生成不完整对比。

脚本只记录开始/结束时间、step、GPU-hours 和 exit code，不查询机器规格或云价格。训练结束后通过 SSH 查看终态，再在 AutoDL 控制台关机；仅在 guest 内执行 `shutdown` 不能作为“停止计费”的证据。

## 9. SSH 与后续脚本范围

后续执行全部通过已配置的 SSH 别名（建议 `autodl-searchr1`）和 `ssh-skill` 完成，不直接使用裸 `ssh/scp`。第一阶段脚本先上传到持久盘再执行；第二、三阶段直接执行固定 checkout 中的脚本。用户正常只需触发三次：Git、CPU prepare、GPU run，其余命令只用于查看日志或人工重跑失败实验。

最小代码/脚本清单：

- `scripts/autodl/01_git.sh`
- `scripts/autodl/02_cpu_prepare.sh`
- `scripts/autodl/03_gpu_run.sh`
- `scripts/autodl/train_small_grpo.sh`
- `scripts/data_process/nq_small.py`
- `requirements-autodl.lock`
- `search_r1/llm_agent/generation.py`：输出实际搜索次数 tensor
- `verl/trainer/main_ppo.py`：接入 `cost_lambda`
- `verl/trainer/ppo/ray_trainer.py`：记录 EM、搜索次数和 utility

不新增 Docker/Kubernetes、多机调度、硬件扫描器、复杂恢复框架或超参 sweep。实现顺序严格按上述三个阶段推进。
