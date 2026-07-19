# AutoDL 三阶段操作说明

本目录是 Search-R1-small 云端复现的唯一入口。固定镜像为 **PyTorch 2.8.0 / Python 3.12 / Ubuntu 22.04 / CUDA 12.8**，持久目录为 `/root/autodl-tmp/search-r1`。正常流程始终是 **Git -> CPU -> GPU**；脚本不扫描机器规格、不自动改配置、不自动重试，也不执行关机。

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

CPU 阶段复用 Conda `llmdevelop` 的镜像 Python，在持久盘创建 train/retriever venv；下载并固定 Qwen3.5-2B、BM25 索引和 wiki-18 corpus；生成 NQ 512/64/128；完成 tokenizer、真实 BM25、数据与奖励测试；组合 1/2 GPU 的四种训练 mode 和四种评测 mode；最后发布自校验的 `manifests/cpu_handoff.json`。

handoff 与完整 Git commit、依赖、资产、数据和 resolved config 绑定。**本次新 commit 会使旧 handoff 失效，必须重新运行本阶段。** 已存在且通过校验的模型、语料、索引、数据和环境可以复用，不需要重复下载。CPU 成功后在 AutoDL 控制台停止 CPU 实例，并确认下一实例挂载同一个 100 GB 数据盘。

### 3. GPU 离线训练

当前两卡实例的正常命令为：

```bash
GPU_COUNT=2 AUTODL_PRICE_PER_HOUR=5.76 \
bash /root/autodl-tmp/search-r1/checkout/scripts/autodl/03_gpu_run.sh
```

若控制台整机价格发生变化，替换 `5.76`。入口先离线验证 handoff，再启动 CPU BM25 服务。随后运行两个 1-step gate：先以 A 完成 rollout、搜索、反向与 checkpoint，再用该 checkpoint 执行 `train control 1`，验证 actor/ref 的子 checkpoint 加载。两者成功后只删除 gate 权重，日志和终态保留。

正式阶段依次运行 `train reproduce 60`、`train control 20 <R60>` 和 `train cost_aware 20 <R60>`，仅保留 R60、B20、C20 三个正式 checkpoint。最后以 `eval base|reproduced|control|cost_aware` 在同一 test-128 上评测 A/R/B/C，四路都用 `lambda=0.10` 计算 utility，并分别报告 EM 与实际搜索次数。结果写入 `runs/comparison/results.md`。

B/C 只从 R 的模型权重启动；Adam、warmup、数据 shuffle 状态和 KL reference 都会分别重新初始化。这是 stage-2 受控分叉，不是精确续训。单 seed 与 test-128 只能展示趋势，不能宣称统计显著性。

## 固定配置与回退

默认配置为两张 GPU、train batch 4、GRPO group size 8、最多 4 次搜索、retriever top-k 3、start 1024、observation 384、response 256，每步 32 条轨迹，并固定 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。`max_prompt_length` 按真实循环固定为 `1024 + 4 * (response + 384)`，默认是 3584；搜索轮数不能通过环境变量覆盖。不再先试单卡。

只有两卡 gate OOM 时才人工重跑整个 GPU 阶段：先设置 `TRAIN_BATCH_SIZE=2`，仍失败再加 `MAX_RESPONSE_LENGTH=192`，此时 `max_prompt_length` 自动派生为 3328。R/B/C 必须共享同一卡数和同一组回退参数。失败不会自动重试、不会覆盖旧 attempt，也不会采用早于固定终点的 checkpoint。

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

脚本**不会执行 guest shutdown**。无论成功、失败还是人工停止，都先通过 SSH 确认终态，再到 AutoDL 控制台确认实例已经停止且不再计费。guest 内关机或 SSH 断开本身都不能证明云端停止计费。
