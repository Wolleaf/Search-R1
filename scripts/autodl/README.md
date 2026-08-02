# AutoDL 三阶段操作说明

本目录是 Search-R1-small 云端复现的唯一入口。固定镜像为 **PyTorch 2.8.0 / Python 3.12 / Ubuntu 22.04 / CUDA 12.8**，持久目录为 `/root/autodl-tmp/search-r1`。正常流程始终是 **Git -> CPU -> GPU**；脚本不扫描机器规格、不自动改配置、不自动重试。GPU phase 本身不关机；需要时显式绑定本次 attempt 启动独立 watchdog。

`03/05/06/07` 保留此前 NQ、搜索机会门和 XML grouped probe 的可执行证据，不作为 Qwen3.5 原生协议入口。从零复现时，`08_gpu_qwen_native_gate.sh` 运行结构 G0/G1 门，`09_gpu_qwen_native_train.sh` 提供 smoke 和完整 main。本轮 R60、G3、B20/C20 recovery 与 A/R 三端点评测已经全部完成并封存，**当前没有待启动的 GPU 阶段**。总交接见 [`docs/qwen35_native_complete_experiment_handoff.md`](../../docs/qwen35_native_complete_experiment_handoff.md)，最终结果见 [`docs/qwen35_native_arbc_final_results_analysis.md`](../../docs/qwen35_native_arbc_final_results_analysis.md)。

## 最终状态：实验已完成（不要重跑）

以下 A/R-only 命令保留为 2026-08-01 至 2026-08-02 实际执行记录，不是当前待办。最终 outer attempt 为 `/root/autodl-tmp/search-r1/state/attempts/gpu/20260801T123025Z-1367-15987`，终态 `success/0`；结果合同为 `qwen-native-training-ar-eval-only-v1`，143/143 项远端 evidence manifest 已通过复核，digest 为 `0058974e9cdc0eae9d0313a52ebebd60051c5b483beceab8f83bdcaf7b74a63e`。只有在另行注册全新复现实验时才可重新执行本节。

A 是 sealed post-trained Qwen3.5-2B，R 是 exact native-v4 `global_step_60`。二者绑定实验时的 clean detached checkout `f8c1cd7e87078d07385f74ca8710add5d5f79c06`、原 CPU handoff、native-v4 数据和 exact R60 evidence。为了不改写科学身份，新 runner 以受控 SFTP 部署到 `/root/autodl-tmp/search-r1/operator/`，其字节哈希单独记录；它复用已部署的 `12_gpu_qwen_native_bc_only.sh` 做前驱验证，但不会进入其中的 B/C pipeline。

CPU 无卡阶段只校验 runner、A、R60、六份 endpoint config、三份数据、G3 前驱 receipt 和最终 B/C recovery archive；不安装依赖、不下载或重建数据、不发布新 CPU handoff，也不启动 retriever 或任何评测：

```bash
QWEN_NATIVE_PROTOCOL_GATE_EVIDENCE=/root/autodl-tmp/search-r1/manifests/qwen-native-gate/20260728T044634Z-2051-4045.ok \
QWEN_NATIVE_SMOKE_EVIDENCE=/root/autodl-tmp/search-r1/manifests/qwen-native-training-smoke/20260728T061246Z-1316-15067.ok \
QWEN_NATIVE_R60_EVIDENCE=/root/autodl-tmp/search-r1/manifests/qwen-native-training-r60-only/20260728T092026Z-2906-16060.ok \
bash /root/autodl-tmp/search-r1/operator/14_gpu_qwen_native_ar_eval_only.sh --cpu-prepare
```

CPU verifier 必须重新哈希 R60 marker、`global_step_60`、原 runner、G0/G1、smoke、checkout、handoff、三套 endpoint 数据与配置，并确认 A tree digest 为 `bc67be20efb353ba14d9c1b291a64410afec94f2310e94b59b6b76047b164e78`、R checkpoint digest 为 `583771b131b6e2aa663ee2ef13ea6421524cdd7fed839f35fd6f9eb246a2c231`。任一身份不符就失败关闭；不得改用 `AUTODL_RESEAL_ONLY=1`、`02_cpu_prepare.sh` 或网络重建来规避失配。

后续挂载两卡时只执行：

```bash
QWEN_NATIVE_PROTOCOL_GATE_EVIDENCE=/root/autodl-tmp/search-r1/manifests/qwen-native-gate/20260728T044634Z-2051-4045.ok \
QWEN_NATIVE_SMOKE_EVIDENCE=/root/autodl-tmp/search-r1/manifests/qwen-native-training-smoke/20260728T061246Z-1316-15067.ok \
QWEN_NATIVE_R60_EVIDENCE=/root/autodl-tmp/search-r1/manifests/qwen-native-training-r60-only/20260728T092026Z-2906-16060.ok \
GPU_COUNT=2 AUTODL_PRICE_PER_HOUR=5.76 \
bash /root/autodl-tmp/search-r1/operator/14_gpu_qwen_native_ar_eval_only.sh
```

保存 launcher 输出的 exact attempt 路径，再把同一路径显式绑定给 A/R 专用 watchdog；不要在后续从可能已经变化的 `state/latest/gpu` 猜测：

```bash
AUTODL_ROOT=/root/autodl-tmp/search-r1 \
bash /root/autodl-tmp/search-r1/operator/15_watch_qwen_native_ar_eval_only.sh \
  /root/autodl-tmp/search-r1/state/attempts/gpu/<exact-attempt-id>
```

入口返回只表示已提交后台 worker，不表示六项评测已经完成。GPU 阶段严格按 `A-val/R-val -> A-NQ/R-NQ -> A-multihop/R-multihop` 执行，固定 group 1、greedy、seed 42；val 和 NQ 各 128 题，multihop 256 题。multihop 单项使用 10 元硬上限，避免历史 5 元超时边界；其余单项保持 5 元。结果生成 `paired-ar-{val,nq_test,multihop}`，并封存六个 inner run 的原始 `terminal/exit-code`。该入口成功时正常返回 `0`，不会再用“科学成功但 outer failed”的受控非零码伪装关机。

冻结 checkout 中的旧 `04_watch_and_shutdown.sh` 尚不识别新的 A/R-only success contract，不能直接绑定本入口；否则它会 fail closed 并保持开机。`15_watch_qwen_native_ar_eval_only.sh` 只接受本轮 exact runner、CPU receipt、outer entrypoint、A/R contract、六项 lineage 和 marker-last evidence；终态后取得全局锁并完成二次授权才会以无参数 `/usr/bin/shutdown` 请求关机。guest dispatch 不等于控制台已经停止计费，仍须在 AutoDL 控制台确认实例已关机。

> **本轮禁止运行** `QWEN_NATIVE_TRAIN_STAGE=main bash .../09_gpu_qwen_native_train.sh`。它会从 sealed base 重新训练 R60，然后运行 G3 和 A/R endpoint，并可能自动启动 B/C。`10_gpu_qwen_native_r60_only.sh`也会重训 R60，同样不得运行。不得更新、reseal 或用新 commit 重建当前 checkout。

## 从零复现的完整路径（归档；当前不运行）

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

CPU 阶段复用 Conda `llmdevelop` 的镜像 Python，在持久盘创建 train/retriever venv；下载并固定 Qwen3.5-2B、BM25 索引和 wiki-18 corpus；生成 NQ/Hotpot train-512、val-128、结构 probe、NQ test-128 与 multihop-256；完成 tokenizer、真实 BM25、数据与奖励测试；组合并校验生产配置；最后发布自校验的 `manifests/cpu_handoff.json`。

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

该模式不联网、不重装环境、不下载资产，也不重放整套 BM25 查询。它要求上一次成功 handoff 已封存语义校验通过的 `retrieval_replay.json` 及 sidecar，只从该 source、既有 retrieval evidence、selection funnel 和固定 tokenizer 物化独立的 `data/search_mix_qwen35_native_v4/`；旧 v2/v3 与原 `data/search_mix/` 均不覆盖。v4 保留论文原版单 user prompt，只把 `<search>/<information>` 映射为 Qwen3.5 原生 tool call/tool response，并开启 native thinking。版本化清单逐条封存 52 个 multi-gold 样本：仅保留 2 个已核实 alias，剔除其余 50 个以及 bit/nibble 错标，并在同 source/category/split 确定性补齐 51 个单 gold；任何未经审计的 multi-gold 都不能入选或作为补位。四个常规 action 耗尽后，仍 active 的轨迹收到固定 terminal answer-only user 提示。该提示是软约束：它作为上下文参与后续生成，但其环境 token 的 policy mask 必须为 0；模型随后生成的 terminal assistant token 仍参与 policy loss。terminal search 只记录，不被接受也不执行。

CPU 使用固定 tokenizer 验证全部 640 条 train/val evidence 在 500-token observation 下仍可见，并验证 G0-8、autonomous G1-16、R 后 G3-64、NQ test-128 和 multihop-256。随后组合 smoke/R/B/C 以及 A/R/B/C 三个 endpoint 的两卡 resolved config。handoff schema 3 显式绑定 prompt version、thinking、总 action budget 4、选样 observation 384 和 rollout observation 500；候选数据、配置与 handoff 全部通过后才原子发布，失败可安全重跑同一命令。

### 3. GPU Qwen native 结构门禁

先运行唯一的训练前结构门；完成后把入口打印的 exact attempt 路径交给 `04_watch_and_shutdown.sh`：

```bash
QWEN_NATIVE_GATE_STAGE=g0_g1 GPU_COUNT=2 AUTODL_PRICE_PER_HOUR=5.76 \
bash /root/autodl-tmp/search-r1/checkout/scripts/autodl/08_gpu_qwen_native_gate.sh
```

G0 使用同一模型进程核对 direct HF、native manager 和 legacy manager 的模板/action 边界；G1 在固定 16 题上验证自主 action、真实工具回填、原始 token 和轨迹结构。这里只阻断协议、模板、数值或证据错误，不要求未训练 parent 已经具备多搜索能力。terminal 硬门要求提醒覆盖全部待收尾轨迹、提醒 token 的 policy mask 为 0、terminal search accepted/executed 均为 0，并保留既有 token、mask、prompt、trace 与 retrieval lineage 一致性检查；terminal answer、requested-search 和 invalid count/rate 只记录为行为诊断，不决定 GO/NO-GO。结果位于 `runs/qwen-native-gate/attempts/<gpu-attempt>/`，只有 `go_no_go.json=GO` 才能进入 smoke。旧合同下的 `NO-GO` marker 和轨迹不追溯改写；合同更新后必须在新 commit 上重跑一次 G0/G1，生成新的 exact marker。该 attempt 硬上限为 2 元；超时或 schema 错误保留非零 exit code，脚本不换 seed 或自动重试。

### 4. GPU Qwen native 完整训练与配对评测（当前不要运行）

> 本节只用于从零复现完整 native-v4 流程。本轮 R60、G3、B/C 与 A/R 已封存；不得把本节 `main` 命令当作恢复或续训入口。

结构门的 exact marker 为唯一训练前驱。先只运行两步 smoke：

```bash
QWEN_NATIVE_TRAIN_STAGE=smoke \
QWEN_NATIVE_PROTOCOL_GATE_EVIDENCE=/root/autodl-tmp/search-r1/manifests/qwen-native-gate/<g0_g1-attempt>.ok \
GPU_COUNT=2 AUTODL_PRICE_PER_HOUR=5.76 \
bash /root/autodl-tmp/search-r1/checkout/scripts/autodl/09_gpu_qwen_native_train.sh
```

smoke 自动复算 strict EM，并要求 mixed reward group、非零 policy advantage、mask/log-prob 对齐，以及两个 step 的 loss、KL、entropy、grad norm 全部存在且有限；terminal 继续使用与 G0/G1 相同的硬门和诊断指标语义。同时封存 step-2 checkpoint、完整 trace、run-specific WandB offline history 和 `storage.env`。GO/NO-GO 都会作为完整科学终态封存并可交给 watchdog。只有 `contract.env` 与 `smoke-decision.json` 同为 `GO` 时才可继续；先人工查看 `storage.env`，确认剩余空间能容纳 R/B/C，再用传入 exact smoke marker 的动作明确批准 main：

```bash
QWEN_NATIVE_TRAIN_STAGE=main \
QWEN_NATIVE_PROTOCOL_GATE_EVIDENCE=/root/autodl-tmp/search-r1/manifests/qwen-native-gate/<g0_g1-attempt>.ok \
QWEN_NATIVE_SMOKE_EVIDENCE=/root/autodl-tmp/search-r1/manifests/qwen-native-training-smoke/<smoke-attempt>.ok \
GPU_COUNT=2 AUTODL_PRICE_PER_HOUR=5.76 \
bash /root/autodl-tmp/search-r1/checkout/scripts/autodl/09_gpu_qwen_native_train.sh
```

main 的 R60 始终从 sealed Qwen3.5-2B 全新启动，不继承 smoke。R 完成后在固定 64×5 集合运行能力 G3；无论结果如何，A 与 R 都会用 group 1、greedy、seed 42 在 curated val-128、未筛选 NQ test-128 和 multihop-256 上评测，并分别生成 `paired-ar-{val,nq_test,multihop}/`。只有 G3 GO 且 `cost_contrast_group_count >= 8` 才从同一个 R digest 对称启动 B20 与 C-gated20；未放行时以 `exit-code=0` 封存完整负结果。放行后 B/C 在同三套 endpoint 上评测，并生成三个 `paired-*` 目录、逐题结果、答对/答错清单、搜索转移和完整轨迹。脚本不自动重试、降配、删除 checkpoint 或关机。

### 历史 GPU Base grouped probe

CPU 混合数据 handoff 完成后，两卡实例只运行：

```bash
GPU_COUNT=2 AUTODL_PRICE_PER_HOUR=5.76 \
bash /root/autodl-tmp/search-r1/checkout/scripts/autodl/07_gpu_group_probe.sh
```

若控制台整机价格变化，替换 `5.76`。该入口不训练、不保存 checkpoint；它在 `probe_multi_64.parquet` 上每题随机采样 5 条轨迹，共 320 条，固定 response 500、prompt 4096、validation batch 8。只有有效正确多搜不少于 16 条、覆盖不少于 8 题、可学习 group 不少于 8 个，且 clipped/invalid 各不高于 5%，结果才为 GO。

分析器逐条保留完整思考、query、检索文档、答案、截断和非法动作，并按 comparison/bridge 分层。它还报告“检索链合格但 strict EM=0”的 near-miss、cover-EM 和示例轨迹；`16/320` 只用于区分搜索策略缺失与答案抽取问题，不改变 GO/NO-GO。GO 与 NO-GO 都是本实现切片的终态：结果会完整封存并允许 watchdog 关机，不会自动训练或重试；只有人工确认 GO 后才另行实现下一训练切片。单次硬上限为 10 元。

## 历史 GPU 工作流（本轮不要运行）

`03_gpu_run.sh` 是旧 NQ A/R/B/C 完整流程；`05_gpu_cost_aware_gated.sh` 和 `06_gpu_search_opportunity_gate.sh` 是其后续实验。它们保留复现实证，但不读取本轮 `search_mix` 训练集，也不能代替当前 native-v4 结构门与 Parent/R 评测。

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

当前 native-v4 gate 与训练固定为两张 GPU、batch 8、训练 group 5、四个可检索 action 加一次 answer-only 收尾生成、retriever top-k 3、`start/observation/response/trajectory=1024/500/500/4500`，并固定 `temperature/top-p/top-k/min-p/presence/repetition=1.0/1.0/0/0.0/0.0/1.0`。其中 `4500 = 4 * (500 + 500) + 500`，最后 500 token 只用于尚未结束轨迹的 terminal generation，不执行也不计搜索成本。选样阶段保留 384-token provenance，但 rollout 使用 500；endpoint 固定 group 1、greedy、seed 42。历史 `07` 保持 XML group-5 配置，`03/05/06` 保持 response 256、prompt 3584；新实验不得复用历史入口。

native-v4 exact attempt 不接受 batch 4、response 384 或关闭 thinking 的历史 fallback。两卡 2-step smoke 还必须从唯一 `run-*.wandb` 二进制重放 step 1/2 的五项 actor 指标、summary 与 `exit_code=0`，并与 `train.log` 对齐。当前门禁修正只区分 terminal 行为诊断与工程硬门，不修改 prompt、parser、reward、数据、BM25、采样或训练参数。若 smoke 失败，保留失败 attempt 并停止；任何降配都必须另立配置版本并重新执行结构门，不能在同一实验身份下静默重跑。失败不会自动重试、覆盖旧 attempt 或采用早于固定终点的 checkpoint。

## 预算与存储

历史 G3-only 注册 10 元硬上限；按 5.76 元/小时换算约为 104 分钟工作时限。它只用于防止失控，不是预计耗时，也不包含 GNU timeout 的 120 秒强杀宽限。G3-only 不生成新全参 checkpoint；该阶段已经完成并封存，不应再次运行。

从零复现时，native-v4 结构门与两步 smoke 的硬上限分别为 2 元和 15 元。完整 main 中 R60/G3/B20/C20 分别为 100/10/40/25 元；A/R 六个 endpoint 始终注册，每个 5 元，B/C 六个 endpoint 仅在分支放行后注册。完整 GO 路径的分项硬上限合计 252 元。脚本按 `AUTODL_PRICE_PER_HOUR` 换算 timeout；这是防失控上限，不是预计账单，CPU/存储费用另计。

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

watchdog 同时支持成功和失败终态，但只有在 commit/checkout、持久盘、exact attempt、phase lock、原始 exit code、唯一终态 marker 和日志 sentinel 全部重新验证后才会调用 AutoDL 的 `/usr/bin/shutdown`（无参数）。旧流程成功时校验 `comparison.sha256`、results、`gpu.ok` 和 attempt digest；native gate、native smoke/main、C-gated、搜索机会门与 grouped probe 分别校验自己的 result root、全部 `evidence.sha256` 条目、新 marker 和 attempt digest，不借用或改写旧 `gpu.ok`。native main 还逐行核对 R/G3、始终存在的六个 A/R endpoint，以及授权后才存在的 B/C 与六个 B/C endpoint，并验证三套 paired summary。锁冲突（包括 admission `exit 75` 后旧锁已经释放）、状态不完整、校验失败或 dry-run 都会保持开机并记录 `shutdown-skipped`；test mode 只记录模拟状态，绝不调用真实 backend。

历史 `11_gpu_qwen_native_g3_only.sh` 的设计合同原本是在完整封存后返回受控 `201`，以便不认识新成功合同的旧 watchdog 走 failure path 关机；但实际 G3 attempt 的严格 replay verifier 提前返回 `failed/1`，没有发布正式 G3-only marker，不能追溯写成 `201`。科学结论来自已完成的 inner eval、exact trace 与 registered analysis，并正式判为 `NO-GO`。watchdog 关机只证明外层持久状态可用，不代替 evidence 校验，也不决定科学 GO/NO-GO。`shutdown-requested` 表示即将调用 backend，`shutdown-dispatched` 只表示 backend 已返回 0，两者都不能证明 AutoDL 控制平面已停止。无论 watchdog 结果如何，仍须在 AutoDL 控制台确认实例已停止且不再计费；SSH 断开本身不能证明停止计费。

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
