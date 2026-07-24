# AutoDL 三阶段操作说明

本目录是 Search-R1-small 云端复现的唯一入口。固定镜像为 **PyTorch 2.8.0 / Python 3.12 / Ubuntu 22.04 / CUDA 12.8**，持久目录为 `/root/autodl-tmp/search-r1`。正常流程始终是 **Git -> CPU -> GPU**；脚本不扫描机器规格、不自动改配置、不自动重试。GPU phase 本身不关机；需要时显式绑定本次 attempt 启动独立 watchdog。

`03/05/06/07` 保留此前 NQ、搜索机会门和 XML grouped probe 的可执行证据，不作为 Qwen3.5 原生协议入口。当前流程先重封 native-v2 CPU handoff，再由 `08_gpu_qwen_native_gate.sh` 依次执行 G0+G1、G2 和 G3；G3 GO 后，`09_gpu_qwen_native_train.sh` 将两步 smoke 与正式 R/B/C 训练拆成两个独立 attempt。任一门禁未通过都不进入下一付费阶段。

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

已有完整 `search_mix` 和 CPU handoff 后，Qwen3.5 原生协议数据使用离线增量模式：

```bash
AUTODL_QWEN_NATIVE_INCREMENTAL=1 \
bash /root/autodl-tmp/search-r1/checkout/scripts/autodl/02_cpu_prepare.sh
```

该模式不联网、不重装环境、不下载资产，也不重放整套 BM25 查询。它要求上一次成功 handoff 已封存语义校验通过的 `retrieval_replay.json` 及 sidecar，只从该 source、evidence、selection funnel 和固定 tokenizer 物化独立的 `data/search_mix_qwen35_native_v2/`，保持相同 train/val/probe 样本 ID，并新增固定 G0-8、强制搜索 G1-16、自主搜索 G2-32 和 G3-64 Parquet。CPU 会用真实 tokenizer 扫描六个 Parquet 共 760 条 prompt，逐条检查严格 `<answer>` 合同、canonical messages、保留标记、模板 token 一致性和 1024-token 初始上限。旧 `data/search_mix_qwen35_native/` 和原 `data/search_mix/` 都不会被覆盖。随后运行测试，组合 G1-G3、smoke/R/B/C-gated 训练及 B/C endpoint 评测的两卡 resolved config，并把协议、tool schema、chat template、tokenizer revision、replay receipt、数据和配置 digest 纳入新 handoff。候选配置与 handoff 全部验证成功后才原子替换旧 seal；中途失败仍可安全重跑同一增量命令。

### 3. GPU Qwen native 分层门禁

每个门禁都是一个独立 GPU attempt，完成后都应把入口打印的 exact attempt 路径交给 `04_watch_and_shutdown.sh`。先运行 G0+G1：

```bash
QWEN_NATIVE_GATE_STAGE=g0_g1 GPU_COUNT=2 AUTODL_PRICE_PER_HOUR=5.76 \
bash /root/autodl-tmp/search-r1/checkout/scripts/autodl/08_gpu_qwen_native_gate.sh
```

G0 使用同一模型进程比较 direct HF、native manager 和 legacy manager；G1 在固定 16 题上做每题两条的强制搜索闭环。结果位于 `runs/qwen-native-gate/attempts/<gpu-attempt>/`。只有 `go_no_go.json` 为 `GO` 才能继续。G2 和 G3 必须显式传入上一阶段打印并封存的 exact marker，不能按 `latest` 或时间猜前驱：

```bash
QWEN_NATIVE_GATE_STAGE=g2 \
QWEN_NATIVE_PREDECESSOR_EVIDENCE=/root/autodl-tmp/search-r1/manifests/qwen-native-gate/<g0_g1-attempt>.ok \
GPU_COUNT=2 AUTODL_PRICE_PER_HOUR=5.76 \
bash /root/autodl-tmp/search-r1/checkout/scripts/autodl/08_gpu_qwen_native_gate.sh

QWEN_NATIVE_GATE_STAGE=g3 \
QWEN_NATIVE_PREDECESSOR_EVIDENCE=/root/autodl-tmp/search-r1/manifests/qwen-native-gate/<g2-attempt>.ok \
GPU_COUNT=2 AUTODL_PRICE_PER_HOUR=5.76 \
bash /root/autodl-tmp/search-r1/checkout/scripts/autodl/08_gpu_qwen_native_gate.sh
```

G2 是 32 题 × 3 条自主搜索，G3 是 held-out 64 题 × group 5。三段 GPU action 的工作 timeout 额度分别为 2、3、10 元，外层 deadline 同时覆盖 handoff/数据校验、BM25、模型生成、分析和封存；内部生成 timeout 取注册额度与“剩余时间减 180 秒收尾预留”的较小值。BM25 不继承 phase lock，并由有界 TERM→KILL 清理。进程若不响应 TERM，GNU `timeout` 最多再保留 120 秒强杀宽限（按 5.76 元/小时最坏约 0.19 元），所以额度不是含该宽限的绝对账单上限。超时作为工程失败封存并交给 watchdog 请求关机，但 guest shutdown 不等于平台已停止计费，仍须在 AutoDL 控制台确认。协议、采样、数据、checkpoint 和前驱 digest 均进入证据。科学 NO-GO 会完整封存并正常结束，不会自动换 seed、重试或进入训练；其他工程/schema 错误保留原始非零 exit code。

### 4. GPU Qwen native 训练与配对评测

G3 的 exact marker 为唯一训练前驱。先只运行两步 smoke：

```bash
QWEN_NATIVE_TRAIN_STAGE=smoke \
QWEN_NATIVE_PRETRAIN_G3_EVIDENCE=/root/autodl-tmp/search-r1/manifests/qwen-native-gate/<g3-attempt>.ok \
GPU_COUNT=2 AUTODL_PRICE_PER_HOUR=5.76 \
bash /root/autodl-tmp/search-r1/checkout/scripts/autodl/09_gpu_qwen_native_train.sh
```

smoke 自动复算 strict EM，并要求 mixed reward group、非零 policy advantage、mask/log-prob 对齐，以及两个 step 的 loss、KL、entropy、grad norm 全部存在且有限；同时封存 step-2 checkpoint、完整 trace、run-specific WandB offline history 和 `storage.env`。GO/NO-GO 都会作为完整科学终态封存并可交给 watchdog。只有 `contract.env` 与 `smoke-decision.json` 同为 `GO` 时才可继续；先人工查看 `storage.env`，确认剩余空间能容纳 R/B/C，再用传入 exact smoke marker 的动作明确批准 main：

```bash
QWEN_NATIVE_TRAIN_STAGE=main \
QWEN_NATIVE_PRETRAIN_G3_EVIDENCE=/root/autodl-tmp/search-r1/manifests/qwen-native-gate/<g3-attempt>.ok \
QWEN_NATIVE_SMOKE_EVIDENCE=/root/autodl-tmp/search-r1/manifests/qwen-native-training-smoke/<smoke-attempt>.ok \
GPU_COUNT=2 AUTODL_PRICE_PER_HOUR=5.76 \
bash /root/autodl-tmp/search-r1/checkout/scripts/autodl/09_gpu_qwen_native_train.sh
```

main 的 R60 始终从 sealed Qwen3.5-2B 全新启动，不继承 smoke。R 完成后先在固定 64×5 集合重跑 G3；只有 G3 GO 且 `cost_contrast_group_count >= 8` 才从同一个 R digest 对称启动 B20 与 C-gated20。若能力门 NO-GO，脚本以 `exit-code=0` 封存 R 负结果而不训练 B/C。若放行，B/C 完成后在同一 sealed val-128 上各评测一次，并生成 `paired/summary.json`、逐题结果、答对/答错清单、搜索转移和完整轨迹。脚本不自动重试、降配、删除 checkpoint 或关机。

### 历史 GPU Base grouped probe

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

当前 native-v2 gate 与训练固定为两张 GPU、batch 8、最多 4 次搜索、retriever top-k 3、`start/observation/response/prompt=1024/384/500/4096`，并固定 `temperature/top-p/top-k/min-p/presence/repetition=1.0/1.0/0/0.0/0.0/1.0`。G1/G2/G3 的评测 group 分别为 2/3/5，训练 group 固定为 5。历史 `07` 继续保持 XML group-5 配置，`03/05/06` 保持 response 256、prompt 3584；新实验不得复用历史入口。

native-v2 exact attempt 不接受 batch 4 或 response 384 的历史 OOM fallback。若两卡 2-step smoke 仍 OOM，保留失败 attempt 并停止；任何降配都必须另立配置版本、重新执行 readiness gate，而不是在同一实验身份下静默重跑。失败不会自动重试、不会覆盖旧 attempt，也不会采用早于固定终点的 checkpoint。

## 预算与存储

native-v2 三段 readiness gate 的分项硬上限为 2/3/10 元，两步 smoke 为 15 元。main 中 R60/G3/B20/C20/B-eval/C-eval 分别为 100/10/40/25/5/5 元：R 后科学 NO-GO 最多进入前 110 元，完整 GO 路径分项合计 185 元；从 G0 到最终配对评测的全部注册上限为 215 元。脚本根据 `AUTODL_PRICE_PER_HOUR` 将每项上限换算成 timeout。硬上限用于防止失控，不是预计实际花费；GNU timeout 的 120 秒强杀宽限和 CPU/存储费用另计。

100 GB 盘不预设“必然够用”。两步 smoke 会记录实际 `checkpoint_bytes` 和 `filesystem_available_bytes`；启动 main 前必须据此确认还能同时保留 R/B/C endpoint、日志、trace 和 WandB history。smoke checkpoint 在 main evidence 完整封存前不得删除，脚本也不会自动清理历史 checkpoint、扩容或覆盖证据。空间不足时保持停机并由人工决定精确清理对象或扩容，不能让脚本猜测路径。

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

watchdog 同时支持成功和失败终态，但只有在 commit/checkout、持久盘、exact attempt、phase lock、原始 exit code、唯一终态 marker 和日志 sentinel 全部重新验证后才会调用 AutoDL 的 `/usr/bin/shutdown`（无参数）。旧流程成功时校验 `comparison.sha256`、results、`gpu.ok` 和 attempt digest；native gate、native smoke/main、C-gated、搜索机会门与 grouped probe 分别校验自己的 result root、全部 `evidence.sha256` 条目、新 marker 和 attempt digest，不借用或改写旧 `gpu.ok`。native smoke 的 GO/NO-GO、main 的 R-G3 科学 NO-GO 和完整六阶段 GO 都有独立结构复验。锁冲突（包括 admission `exit 75` 后旧锁已经释放）、状态不完整、校验失败或 dry-run 都会保持开机并记录 `shutdown-skipped`；test mode 只记录模拟状态，绝不调用真实 backend。`shutdown-requested` 表示即将调用 backend，`shutdown-dispatched` 只表示 backend 已返回 0，两者都不能证明 AutoDL 控制平面已停止。无论 watchdog 结果如何，仍须在 AutoDL 控制台确认实例已停止且不再计费；SSH 断开本身不能证明停止计费。

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
