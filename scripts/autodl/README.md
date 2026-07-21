# AutoDL 三阶段操作说明

本目录是 Search-R1-small 云端复现的唯一入口。固定镜像为 **PyTorch 2.8.0 / Python 3.12 / Ubuntu 22.04 / CUDA 12.8**，持久目录为 `/root/autodl-tmp/search-r1`。正常流程始终是 **Git -> CPU -> GPU**；脚本不扫描机器规格、不自动改配置、不自动重试。GPU phase 本身不关机；需要时显式绑定本次 attempt 启动独立 watchdog。

正式实验固定为：A（原始 Qwen3.5-2B）经原奖励 60 steps 得到 R；B（原奖励）和 C（`cost_lambda=0.10`）再从同一个 R 分别训练 20 steps。R/B/C 使用固定最终 checkpoint，不按 val 指标选择。主对比是 B vs C；A vs R 仅为 Search-R1-small sanity check。

## 最短正常路径

### 1. 固定 Git 版本

先通过受控 SFTP 将整个 `scripts/autodl/` 放到 `/root/autodl-tmp/autodl-bootstrap/`，然后在无 GPU 实例执行：

```bash
REPO_URL=https://github.com/<owner>/Search-R1.git \
COMMIT_SHA=<已推送的40位commit> \
bash /root/autodl-tmp/autodl-bootstrap/01_git.sh
```

仓库以 detached HEAD 固定到 `/root/autodl-tmp/search-r1/checkout`。成功标准是最新 `git` attempt 的 `exit-code` 为 `0` 且存在 `.success`。私有仓库使用临时 credential helper，不把 token 放进 URL 或日志。`01_git.sh` 面向首次 clone；若持久盘已有旧 commit，它会拒绝偷偷改写。此时由受控 SSH 操作使用完整 Git bundle 更新到同一可信 commit 并重新生成 checkout seal；bundle 不是该脚本的隐藏参数，也不能用零散文件覆盖 checkout。

### 2. CPU 联网准备

继续使用未挂载 GPU 的实例：

```bash
bash /root/autodl-tmp/search-r1/checkout/scripts/autodl/02_cpu_prepare.sh
```

CPU 阶段复用 Conda `llmdevelop` 的镜像 Python，在持久盘创建 train/retriever venv；下载并固定 Qwen3.5-2B、BM25 索引和 wiki-18 corpus；生成 NQ 512/64/128；完成 tokenizer、真实 BM25、数据与奖励测试；组合并校验全部生产配置；最后发布自校验的 `manifests/cpu_handoff.json`。

handoff 与完整 Git commit、依赖、资产、数据和 resolved config 绑定。**本次新 commit 会使旧 handoff 失效，必须重新运行本阶段。** 已存在且通过校验的模型、语料、索引、数据和环境可以复用，不需要重复下载。CPU 成功后在 AutoDL 控制台停止 CPU 实例，并确认下一实例挂载同一个 100 GB 数据盘。

已有完整 CPU handoff、环境和资产时，本次 C-gated follow-up 使用离线增量 reseal，不执行 pip/apt、snapshot download、语料展开或数据重建：

```bash
AUTODL_RESEAL_ONLY=1 \
bash /root/autodl-tmp/search-r1/checkout/scripts/autodl/02_cpu_prepare.sh
```

该路径先校验旧 handoff 的全部 digest、两个 Python 环境 freeze、Java、模型/数据/索引/语料，再运行新增测试与配置组合，最后按新 commit 发布 handoff。任何旧证据或资产不一致都会失败关闭；不要退回联网重建来掩盖不一致。

### 3. GPU 离线训练

当前两卡实例的正常命令为：

```bash
GPU_COUNT=2 AUTODL_PRICE_PER_HOUR=5.76 \
bash /root/autodl-tmp/search-r1/checkout/scripts/autodl/03_gpu_run.sh
```

若控制台整机价格发生变化，替换 `5.76`。入口先离线验证 handoff，再启动 CPU BM25 服务。随后运行两个工程 gate：基础 gate 连续训练 2 steps，确保第 2 次 backward 覆盖已经初始化的 Adam 状态；再从其 `global_step_2` 执行 1-step control gate，验证 actor/ref 子 checkpoint 加载。两者成功后只删除 gate 权重，日志和终态保留。

仍保持全参数 FSDP 微调。`optimizer_offload=true` 时，Adam 状态在 forward/backward 期间留在 CPU，只在每次 `optimizer.step()` 前回载到 GPU，并在 step 后立即卸载。该改动不改变损失、梯度、优化器或科学配置，只缩短 optimizer state 与长序列激活同时驻留显存的时间。

正式阶段依次运行 `train reproduce 60`、`train control 20 <R60>` 和 `train cost_aware 20 <R60>`，仅保留 R60、B20、C20 三个正式 checkpoint。最后以 `eval base|reproduced|control|cost_aware` 在同一 test-128 上评测 A/R/B/C，四路都用 `lambda=0.10` 计算 utility，并分别报告 EM 与实际搜索次数。结果写入 `runs/comparison/results.md`。

B/C 只从 R 的模型权重启动；Adam、warmup、数据 shuffle 状态和 KL reference 都会分别重新初始化。这是 stage-2 受控分叉，不是精确续训。单 seed 与 test-128 只能展示趋势，不能宣称统计显著性。

### C-gated 独立增量实验

已有 R60、B20 和 C-old20 后，不再运行上面的完整 `03_gpu_run.sh`。两卡实例只运行：

```bash
GPU_COUNT=2 AUTODL_PRICE_PER_HOUR=5.76 \
bash /root/autodl-tmp/search-r1/checkout/scripts/autodl/05_gpu_cost_aware_gated.sh
```

该入口校验旧 `runs/comparison/lineage.tsv` 及 R/B/C-old checkpoint digest；从同一 R60 运行 2-step C-gated 日志 gate，成功后删除 gate 权重；再从 R60 全新训练固定 C-gated20。B 和 C-old 只做 trace-only test-128 推理，不更新权重；最后评测 C-gated，严格校验 `80/800/128` 轨迹行数和 manifest，并生成三路逐题配对、答对/答错清单、搜索转移、独立训练 CSV/SVG 曲线。旧 `runs/comparison`、`manifests/gpu.ok` 和历史 C-old 证据不会被覆盖。

follow-up 的分项硬上限是 gate 8 元、正式训练 25 元、三路评测各 5 元，合计 48 元。它只限制失控运行，不代表预计支出；不做 val EM 科学早停，只有工程错误或证据校验失败才停止。

## 固定配置与回退

默认配置为两张 GPU、train batch 8、GRPO group size 5、最多 4 次搜索、retriever top-k 3、start 1024、observation 384、response 256、warmup ratio 0.285、temperature/top-p 1.0，每步 40 条轨迹，并固定 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。`max_prompt_length` 按真实循环固定为 `1024 + 4 * (response + 384)`，默认是 3584；搜索轮数不能通过环境变量覆盖。不再先试单卡。相较旧的 batch 4、group 8 配置，每步轨迹从 32 增至 40，理论工作量增加 25%，因此不能宣称新配置训练更快。

只有新的两卡 2-step gate 仍明确 OOM 时才人工重跑整个 GPU 阶段：先保持 batch 8、group 5，仅设置 `MAX_RESPONSE_LENGTH=192`，此时 `max_prompt_length` 自动派生为 3328；仍失败才再设置 `TRAIN_BATCH_SIZE=4`。R/B/C 必须共享同一卡数和同一组回退参数。失败不会自动重试、不会覆盖旧 attempt，也不会采用早于固定终点的 checkpoint。

## 预算与存储

两个 gate 各 15 元、R 100 元、B/C 各 40 元、A/R/B/C 评测各 10 元，GPU 分项硬上限合计 250 元；另留 50 元给 CPU、100 GB 存储和阶段间开销，总上限 300 元。脚本根据 `AUTODL_PRICE_PER_HOUR` 将每项上限换算成 timeout。硬上限用于防止失控，不是预计实际花费，也不保证能跑满目标步数。

100 GB 盘预计最终使用 55-70 GB：环境、模型、BM25 与语料约 30-35 GB，R/B/C 权重约 12-18 GB，其余用于日志、Ray/WandB 缓存和 attempt 元数据。两个 gate checkpoint 会在成功后安全删除，当前无需继续扩容。异常 attempt 的大文件只能在确认其终态和精确路径后人工处理。

## 状态、日志与停机

三个入口都用 `nohup + setsid` 后台运行，并由持久盘上的非阻塞 `flock` 串行化。入口返回只表示已提交，不代表阶段成功。查看 GPU 阶段：

```bash
phase=gpu
attempt="$(cat /root/autodl-tmp/search-r1/state/latest/$phase)"
tail -f "$attempt/phase.log"
cat "$attempt/exit-code"
```

将 `phase` 改为 `git` 或 `cpu` 可查看对应阶段。完成必须同时满足原始 `exit-code=0`、终态 `success` 和 `.success`，不能只看日志末行。每个 attempt 均保留独立日志、原始 exit code、时间和成功/失败标记；GPU 阶段启用 Hugging Face、Datasets、Transformers、pip 与 WandB 离线模式，缺失资产会直接失败。

GPU phase 启动后，用入口打印出的**精确绝对路径**显式安装 watchdog：

```bash
bash /root/autodl-tmp/search-r1/checkout/scripts/autodl/04_watch_and_shutdown.sh "$attempt"
tail -f "$attempt/shutdown-watchdog.log"
```

watchdog 同时支持成功和失败终态，但只有在 commit/checkout、持久盘、exact attempt、phase lock、原始 exit code、唯一终态 marker 和日志 sentinel 全部重新验证后才会调用 AutoDL 的 `/usr/bin/shutdown`（无参数）。旧流程成功时校验 `comparison.sha256`、results、`gpu.ok` 和 attempt digest；C-gated follow-up 则校验独立 result root 的全部 `evidence.sha256` 条目、新 marker 和 attempt digest，不借用或改写旧 `gpu.ok`。锁冲突、状态不完整、校验失败或 dry-run 会保持开机并记录 `shutdown-skipped`；test mode 只记录模拟状态，绝不调用真实 backend。`shutdown-requested` 表示即将调用 backend，`shutdown-dispatched` 只表示 backend 已返回 0，两者都不能证明 AutoDL 控制平面已停止。无论 watchdog 结果如何，仍须在 AutoDL 控制台确认实例已停止且不再计费；SSH 断开本身不能证明停止计费。

## CPU 后处理

GPU 成功并关机后，不需要重新运行 CPU prepare、训练或评测。先校验 `runs/comparison/comparison.sha256`，再在无 GPU 实例或本地环境从 R/B/C 的原始日志导出逐 step CSV 和曲线：

```bash
python scripts/autodl/export_training_curves.py \
  --reproduced-log <R60-attempt>/train.log \
  --control-log <B20-attempt>/train.log \
  --cost-aware-log <C20-attempt>/train.log \
  --results-csv <project-root>/runs/comparison/results.csv \
  --output-dir <export-dir>
```

输出固定为 `training_metrics.csv`、`training_curves.png` 和 `final_comparison.png`。CSV 中的训练 EM 使用独立的 `env/em/mean`，不会把 C 日志里已经扣除成本的 reward 误当作准确率；派生 utility 与最终评测统一使用 `lambda=0.10` 和 `/4`。曲线是未平滑的 per-step 聚合值，不代表逐 trajectory 样本。
