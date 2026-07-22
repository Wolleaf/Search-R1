# AutoDL 三阶段操作说明

本目录是 Search-R1-small 云端复现的唯一入口。固定镜像为 **PyTorch 2.8.0 / Python 3.12 / Ubuntu 22.04 / CUDA 12.8**，持久目录为 `/root/autodl-tmp/search-r1`。正常流程始终是 **Git -> CPU -> GPU**；脚本不扫描机器规格、不自动改配置、不自动重试。GPU phase 本身不关机；需要时显式绑定本次 attempt 启动独立 watchdog。

`03/05/06` 保留此前 NQ 与搜索机会门实验的可执行证据，不作为下一轮混合数据训练入口。`07_gpu_group_probe.sh` 是当前唯一 GPU 入口：先用 Base 在检索验证的 held-out 64 题上做 group-5 探针，再决定是否训练。

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

CPU 阶段复用 Conda `llmdevelop` 的镜像 Python，在持久盘创建 train/retriever venv；下载并固定 Qwen3.5-2B、BM25 索引和 wiki-18 corpus；生成旧 NQ 集及新的 NQ/Hotpot train-512、val-128、probe-64；完成 tokenizer、真实 BM25、数据与奖励测试；组合并校验生产配置；最后发布自校验的 `manifests/cpu_handoff.json`。

handoff 与完整 Git commit、依赖、资产、数据和 resolved config 绑定。**本次新 commit 会使旧 handoff 失效，必须重新运行本阶段。** 已存在且通过校验的模型、语料、索引、数据和环境可以复用，不需要重复下载。CPU 成功后在 AutoDL 控制台停止 CPU 实例，并确认下一实例挂载同一个 100 GB 数据盘。

已有完整 CPU handoff、环境和资产时，本次 C-gated follow-up 使用离线增量 reseal，不执行 pip/apt、snapshot download、语料展开或数据重建：

```bash
AUTODL_RESEAL_ONLY=1 \
bash /root/autodl-tmp/search-r1/checkout/scripts/autodl/02_cpu_prepare.sh
```

该路径先校验旧 handoff 的全部 digest、两个 Python 环境 freeze、Java、模型/数据/索引/语料，再运行新增测试与配置组合，最后按新 commit 发布 handoff。任何旧证据或资产不一致都会失败关闭；不要退回联网重建来掩盖不一致。

搜索机会门首次加入 HotpotQA/2Wiki 数据时，不需要重装环境或重下模型、语料和索引。使用联网的无卡实例运行专用增量模式：

```bash
AUTODL_SEARCH_GATE_INCREMENTAL=1 \
bash /root/autodl-tmp/search-r1/checkout/scripts/autodl/02_cpu_prepare.sh
```

它先验证旧 handoff，然后只下载固定 revision 的两个 dev JSONL（不下载 train），逐文件校验固定字节数与 SHA-256，各抽取 128 题并生成一个 256 行评测文件；verify 会从原始 JSONL 按固定 seed 重算选择并逐题核对 catalog。下载完成后立即恢复离线模式，运行测试、组合配置并按新 commit 重封 handoff。`supporting title` 数量只作为多跳证据代理，不代表题目理论上需要几次搜索。

上述两个历史入口在没有 `search_mix/manifest.json` 时仍可独立重封旧实验所需的 handoff，并跳过 grouped-probe 配置；若 manifest 已存在，则会验证、重放并纳入 handoff。要运行 `07_gpu_group_probe.sh`，必须先完成下面的 search-mix 增量准备。

已有环境、模型、Wiki-18 corpus、BM25 索引和旧 handoff 时，新增混合训练集使用联网无卡实例运行：

```bash
AUTODL_SEARCH_MIX_INCREMENTAL=1 \
bash /root/autodl-tmp/search-r1/checkout/scripts/autodl/02_cpu_prepare.sh
```

该模式只新增固定 revision 的 NQ/HotpotQA train JSONL，不重装环境或重下大资产。train venv 下载并校验源文件，retriever venv 生成真实 top-3 evidence，train venv 按 Qwen tokenizer 的实际 384-token observation 筛选并生成 Parquet；train 固定为 NQ 192、comparison 56、bridge 264，val 为 64/16/48。该配比来自首轮完整漏斗：comparison 严格可用 74 题、bridge 766 题，且 comparison 候选未被 cap 截断；因此保持 NQ/Hotpot 总比例和证据规则不变，只显式重分配 Hotpot 类别。最终 retriever venv 对 640 题重放 1024 次查询。`selection_funnel.json` 记录各层候选、拒绝原因和差额；即使配额不足也先保存失败 receipt 再退出。NQ test-128 与现有 HotpotQA/2Wiki dev-256 都被排除；任一配额不足、source/evidence 不一致或重放漂移都会失败关闭。

### 3. GPU Base grouped probe

CPU 混合数据 handoff 完成后，两卡实例只运行：

```bash
GPU_COUNT=2 AUTODL_PRICE_PER_HOUR=5.76 \
bash /root/autodl-tmp/search-r1/checkout/scripts/autodl/07_gpu_group_probe.sh
```

若控制台整机价格变化，替换 `5.76`。该入口不训练、不保存 checkpoint；它在 `probe_multi_64.parquet` 上每题随机采样 5 条轨迹，共 320 条，固定 response 500、prompt 4096、validation batch 8。只有有效正确多搜不少于 16 条、覆盖不少于 8 题、可学习 group 不少于 8 个，且 clipped/invalid 各不高于 5%，结果才为 GO。

分析器逐条保留完整思考、query、检索文档、答案、截断和非法动作，并按 comparison/bridge 分层。它还报告“检索链合格但 strict EM=0”的 near-miss、cover-EM 和示例轨迹；`16/320` 只用于区分搜索策略缺失与答案抽取问题，不改变 GO/NO-GO。GO 与 NO-GO 都是本实现切片的终态：结果会完整封存并允许 watchdog 关机，不会自动训练或重试；只有人工确认 GO 后才另行实现下一训练切片。单次硬上限为 10 元。

## 历史 GPU 工作流（本轮不要运行）

`03_gpu_run.sh` 是旧 NQ A/R/B/C 完整流程；`05_gpu_cost_aware_gated.sh` 和 `06_gpu_search_opportunity_gate.sh` 是其后续实验。它们保留复现实证，但不读取本轮 `search_mix` 训练集，也不能代替上面的 Base probe。

### 旧 NQ A/R/B/C 流程

```bash
GPU_COUNT=2 AUTODL_PRICE_PER_HOUR=5.76 \
bash /root/autodl-tmp/search-r1/checkout/scripts/autodl/03_gpu_run.sh
```

该入口依次运行旧 NQ smoke、R60、B20、C20 和 test-128 评测。B/C 只从 R 模型权重启动，优化器等状态分别重新初始化；这是一项已完成的单 seed 缩小实验，不是本轮推荐命令。

### C-gated 独立增量实验

已有 R60、B20 和 C-old20 后，不再运行上面的完整 `03_gpu_run.sh`。两卡实例只运行：

```bash
GPU_COUNT=2 AUTODL_PRICE_PER_HOUR=5.76 \
bash /root/autodl-tmp/search-r1/checkout/scripts/autodl/05_gpu_cost_aware_gated.sh
```

该入口校验旧 `runs/comparison/lineage.tsv` 及 R/B/C-old checkpoint digest；从同一 R60 运行 2-step C-gated 日志 gate，成功后删除 gate 权重；再从 R60 全新训练固定 C-gated20。B 和 C-old 只做 trace-only test-128 推理，不更新权重；最后评测 C-gated，严格校验 `80/800/128` 轨迹行数和 manifest，并生成三路逐题配对、答对/答错清单、搜索转移、独立训练 CSV/SVG 曲线。旧 `runs/comparison`、`manifests/gpu.ok` 和历史 C-old 证据不会被覆盖。

follow-up 的分项硬上限是 gate 8 元、正式训练 25 元、三路评测各 5 元，合计 48 元。它只限制失控运行，不代表预计支出；不做 val EM 科学早停，只有工程错误或证据校验失败才停止。

### 多跳搜索机会门（只评测）

在决定是否再训练新的成本分支前，先只评测已保存的 B/control20：

```bash
GPU_COUNT=2 AUTODL_PRICE_PER_HOUR=5.76 \
bash /root/autodl-tmp/search-r1/checkout/scripts/autodl/06_gpu_search_opportunity_gate.sh
```

该入口不训练、不改 checkpoint，也不重复评测 R/C。它只加载一次 B，在 HotpotQA dev-128 与 2WikiMultiHopQA dev-128 上连续推理，保留 256 条完整轨迹，并按数据集、题型、难度和 supporting-title-count 分层统计 EM、搜索 0/1/2/3/4 次分布、`E[S|correct]`、截断和非法动作。离线分析还会列出重复 query、无新增文档或答案早已出现的“冗余候选”；这些是可观察代理，不是删除某次搜索后的反事实证明。

默认只有同时满足以下条件才建议继续花钱训练：B 至少答对 20 题；答对题中至少 8 题且至少 20% 使用两次以上搜索；全体至少 10 题使用三次以上搜索；至少 5 个 strong/medium 冗余候选且占三搜题 30%；截断率和非法动作率均不超过 5%。结果写入 `runs/search-opportunity-gate/attempts/<gpu-attempt>/`。GO/NO-GO 都是成功完成的科学结果，不会因 NO-GO 自动重试。单次评测硬上限为 10 元；按 5.76 元/小时计算约 104 分钟，只是防失控上限，不是预计耗时或支出。

GO 只表示值得进入下一阶段，不能直接在 NQ 上只重训一个 C。后续若继续，必须从未见过的 HotpotQA/2Wiki train split 固定小训练集，并从同一 parent 对称训练原奖励 `B-multihop` 与成本奖励 `C-multihop`，最后仍在本次固定 dev-256 上比较，以隔离奖励函数而不是数据暴露差异。

## 固定配置与回退

当前 `07` grouped probe 固定为两张 GPU、validation batch 8、每题 group 5、最多 4 次搜索、retriever top-k 3、start 1024、observation 384、response 500、prompt cap 4096、temperature/top-p 1.0，并固定 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。历史 `03/05/06` 继续锁定 response 256、prompt 3584；搜索轮数不能通过环境变量覆盖，且新实验不得复用历史入口。未来只有 probe GO 后另行实现的对称训练入口才会采用 train batch 8、group 5 和 `500/4096`。

只有新的两卡 2-step gate 在第二次 backward 明确 OOM 时才人工重跑：先保持 batch 8、group 5，将 `MAX_RESPONSE_LENGTH=384`，prompt cap 仍为 4096。当前各 GPU micro-batch 已为 1，降低总 batch 不能可靠解决单条长序列 OOM；只有证据表明问题来自批次级驻留时才考虑 batch 4。B/C 必须共享同一卡数和同一组回退参数。失败不会自动重试、不会覆盖旧 attempt，也不会采用早于固定终点的 checkpoint。

## 预算与存储

两个 gate 各 15 元、R 100 元、B/C 各 40 元、A/R/B/C 评测各 10 元，GPU 分项硬上限合计 250 元；另留 50 元给 CPU、100 GB 存储和阶段间开销，总上限 300 元。脚本根据 `AUTODL_PRICE_PER_HOUR` 将每项上限换算成 timeout。硬上限用于防止失控，不是预计实际花费，也不保证能跑满目标步数。

100 GB 盘预计最终使用 55-70 GB：环境、模型、BM25 与语料约 30-35 GB，R/B/C 权重约 12-18 GB，其余用于日志、Ray/WandB 缓存和 attempt 元数据。两个 gate checkpoint 会在成功后安全删除，当前无需继续扩容。异常 attempt 的大文件只能在确认其终态和精确路径后人工处理。

## 状态、日志与停机

所有阶段入口都用 `nohup + setsid` 后台运行，并由持久盘上的非阻塞 `flock` 串行化。入口返回只表示已提交，不代表阶段成功。查看 GPU 阶段：

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

watchdog 同时支持成功和失败终态，但只有在 commit/checkout、持久盘、exact attempt、phase lock、原始 exit code、唯一终态 marker 和日志 sentinel 全部重新验证后才会调用 AutoDL 的 `/usr/bin/shutdown`（无参数）。旧流程成功时校验 `comparison.sha256`、results、`gpu.ok` 和 attempt digest；C-gated、搜索机会门与 grouped probe 分别校验自己的 result root、全部 `evidence.sha256` 条目、新 marker 和 attempt digest，不借用或改写旧 `gpu.ok`。锁冲突、状态不完整、校验失败或 dry-run 会保持开机并记录 `shutdown-skipped`；test mode 只记录模拟状态，绝不调用真实 backend。`shutdown-requested` 表示即将调用 backend，`shutdown-dispatched` 只表示 backend 已返回 0，两者都不能证明 AutoDL 控制平面已停止。无论 watchdog 结果如何，仍须在 AutoDL 控制台确认实例已停止且不再计费；SSH 断开本身不能证明停止计费。

## 历史 CPU 后处理（仅旧 03 comparison）

以下步骤只适用于历史 `03_gpu_run.sh` 已生成的 A/R/B/C comparison，不属于当前 `07` grouped-probe 切片。`07` 完成后直接保留其 `runs/group-probe/attempts/<gpu-attempt>/` 证据，不存在新的 R/B/C loss 曲线。历史结果可先校验 `runs/comparison/comparison.sha256`，再在无 GPU 实例或本地环境从 R/B/C 原始日志导出逐 step CSV 和曲线：

```bash
python scripts/autodl/export_training_curves.py \
  --reproduced-log <R60-attempt>/train.log \
  --control-log <B20-attempt>/train.log \
  --cost-aware-log <C20-attempt>/train.log \
  --results-csv <project-root>/runs/comparison/results.csv \
  --output-dir <export-dir>
```

输出固定为 `training_metrics.csv`、`training_curves.png` 和 `final_comparison.png`。CSV 中的训练 EM 使用独立的 `env/em/mean`，不会把 C 日志里已经扣除成本的 reward 误当作准确率；派生 utility 与最终评测统一使用 `lambda=0.10` 和 `/4`。曲线是未平滑的 per-step 聚合值，不代表逐 trajectory 样本。
