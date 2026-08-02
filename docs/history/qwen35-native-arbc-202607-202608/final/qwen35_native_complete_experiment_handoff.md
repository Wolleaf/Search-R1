# Qwen3.5 Native Direct-RL 完整实验交接文档

> 最终状态：**实验执行、恢复、同协议评测与证据封存均已完成**
>
> 最后复核：2026-08-02
>
> 适用分支：`experiment/hotpot-search-gate`
>
> 当前操作：不再启动 R60、G3、B/C 或 A/R 评测；后续只做追加分析，或另立合同启动全新的 SFT/RL 实验

## 1. 交接结论

本轮可以按“工程完成、科学结论冻结、后续分析可追加”正式收口。

- R60 direct outcome-RL 确认带来了显著能力提升，是当前综合能力和 multihop 的默认模型。
- G3 是完整而有效的科学 `NO-GO`：模型已经表现出多次检索能力，但 clipping、invalid native action 和 clean learnable group 不满足正式放行门槛。
- B/C 是 G3 `NO-GO` 后另立合同的探索性分叉。C 相比 B 更省检索，但没有确认性证据证明准确率更高。
- A/R 同协议补评已经完成，四模型 A/R/B/C 的 val、NQ-test、multihop 对比闭环。
- 所有采用结果都已有本地紧凑证据包；远端 checkpoint、raw trace 和完整 manifest 保留在持久盘。
- 旧失败、科学 `NO-GO` 和受控非零退出均保留原样，不应追溯改写成“成功码”。

这里的“完成”不表示所有研究问题已经解决，也不表示复现了原 Search-R1 论文榜单。它表示本轮缩小版 direct-RL 实验已经有稳定身份、完整结果、可复核证据和明确结论，不再需要继续付费运行旧任务。

## 2. 一页实验图

```text
A：sealed post-trained Qwen3.5-2B parent
└── R：direct outcome-RL 60 steps
    ├── G3：held-out 64×5 capability/cost gate
    │   └── 科学结论：NO-GO，不产生新模型
    ├── B：从 R 权重独立启动，control +20 steps
    └── C：从同一 R 权重独立启动，correct-only cost +20 steps

A / R / B / C
└── 同数据、同协议的 val-128 / NQ-test-128 / multihop-256 最终评测

未来 SFT40 → RL20 → B/C
└── 独立新实验；不并入本轮 direct-RL 身份
```

C 不是接着 B 训练。B 和 C 都从 exact R60 的 Hugging Face 权重建立新的 optimizer，属于平行分支。R60 没有保存 optimizer、scheduler 或 trainer state，因此 B/C 也不是严格意义上的 R60 断点续训。

## 3. 实验范围与论文关系

本轮目标是做一个预算受控、证据可审计的 Search-R1 风格缩小实验，研究 Qwen3.5 原生工具协议下：

1. direct outcome-RL 能否学习搜索和短答案能力；
2. 模型是否出现可学习的多次搜索与成本对比机会；
3. correctness-gated 搜索成本项能否降低检索开销而避免早期实验中的 no-search collapse；
4. 工程失败、模型协议失败和科学门禁失败能否被清楚区分。

本轮使用 post-trained Qwen3.5-2B、Wiki-2018 BM25 top-3、两张 5090 级 GPU、小数据和短训练。原论文主实验的模型、dense retriever、数据规模和 benchmark 配置并未被逐项复制。因此允许表述是“Search-R1 风格的预算缩小复现与改进实验”，不允许表述为“完整复现论文结果”。

## 4. 不可变模型身份

| 名称 | 科学身份 | Checkpoint digest | 当前定位 |
|---|---|---|---|
| A | sealed post-trained Qwen3.5-2B parent | `bc67be20efb353ba14d9c1b291a64410afec94f2310e94b59b6b76047b164e78` | 训练前 parent baseline；不是官方 Base 变体 |
| R | A 经 60 步 direct outcome-RL | `583771b131b6e2aa663ee2ef13ea6421524cdd7fed839f35fd6f9eb246a2c231` | 综合能力与 multihop 默认端点 |
| B | 从 R 权重独立启动的 control +20 | `8df3d6ab13e154b4a2360b1535f763731f955baebba21b75f9f6e017000ad6d2` | B/C 内部能力对照 |
| C | 从同一 R 权重独立启动的 cost +20 | `106b628945cdb46ea1053fa042e7e75fb26608e15e83ffce65dd799b96fa4b2e` | 低检索成本候选 |

C 的训练奖励为：

```text
reward_C = EM - 0.025 × executed_searches × EM
```

成本只在正确轨迹上生效，以避免 all-wrong group 奖励“错误但不搜索”。最终报告为了统一比较，另行使用：

```text
utility_report = EM - 0.025 × executed_searches
```

两者对错误轨迹的处理不同，不能把报告 utility 当成 C 实际优化目标。

## 5. 最终数据与协议合同

### 5.1 训练与门禁数据

- R60 train-512：NQ 192、HotpotQA 320；HotpotQA 内 comparison 56、bridge 264。
- R60：60 steps、batch 8、group 5，共 2,400 条训练轨迹；无放回实际覆盖 480 题。
- G3：held-out HotpotQA 64 题，每题 5 条，共 320 条轨迹。
- B/C：各 20 steps、160 个 question group、800 条轨迹；seed、问题顺序和 `(step,sample,slot)` 对齐。

数据重配不是看完最终结果后的补丁。旧搜索机会审计显示几乎没有健康的多搜索轨迹，因此在正式 native-v4 训练前，把 NQ/Hotpot 从 1:1 调为 37.5%/62.5%，并把 Hotpot comparison/bridge 从 200/120 调为 56/264。选择依据是 BM25 检索结果经过 tokenizer 截断后真正可见的 `visible_observation`，配额、漏斗、ledger 和 digest 都被封存。

### 5.2 最终评测

| 端点 | 题量 | 构成 |
|---|---:|---|
| val | 128 | NQ 64 + HotpotQA 64 |
| NQ-test | 128 | held-out NQ 128 |
| multihop | 256 | HotpotQA 128 + 2WikiMultiHopQA 128 |

四模型最终评测统一使用 greedy、`do_sample=false`、group size 1、每题一个 rollout、seed 42、最多四次已执行搜索。A/R 与 B/C 的 Parquet、sample ID、问题文本、gold answers、行序和 checkpoint digest 已交叉核对；不存在重复 ID。

### 5.3 工具与 rollout 协议

- Qwen3.5 native tool calling，不再强迫走旧 XML 协议；
- Wiki-2018 BM25，top-k 3；
- 最多四个正常 action，再给仍未结束轨迹一次 answer-only terminal generation；
- rollout 单轮 observation/response 上限均为 500 token；
- terminal reminder 属于环境上下文，policy mask 为 0；模型生成的 terminal assistant token 仍参与 policy loss；
- terminal 阶段请求 search 会被记录但不接受、不执行；
- strict EM 是既定主指标，不能事后用宽松 alias 规则替代。

## 6. 从最初崩溃到最终收口的完整过程

### 阶段 0：先把云端链路修到“可相信”

7 月 19–20 日首先处理的是工程阻塞，而不是模型结论：固定 external BM25 corpus、CPU→GPU handoff 路径、Qwen3.5 FSDP wrap、optimizer state 装卸、GPU gate、证据封存和安全关机。关键修复链为：

```text
3e54c4c → c979400 → 1779a9c → 345ad0d
```

一次 update 的 smoke 无法覆盖 Adam state 在第一次 `optimizer.step` 后才出现、第二次 update 才触发的路径，因此 smoke 被提升为至少两步。只有两次真实更新、指标有限且完整后，才允许把后续失败解释为模型或实验问题。

### 阶段 1：早期 direct RL 可学习，朴素成本奖励却坍缩

旧 NQ-only 缩小实验先证明 direct RL 能提升结果。但无条件线性成本分支把 no-search 推到 98.44%，EM 从 B 的 17.97% 降到 7.03%。all-wrong group 也会偏好“错误但不搜索”，所以奖励方向本身有漏洞。

随后引入 `correct_only` gate。它消除了 no-search collapse，但在旧 NQ-only 试验里仍未显示部署优势。这一步只建立了当前 C 奖励的动机，没有预先证明 C 一定更好。

### 阶段 2：搜索机会与数据分布重构

HotpotQA+2Wiki 的旧 B 搜索机会审计中，256 题里有 248 题只搜索一次；仅有的 8 条二搜轨迹全部错误、截断或非法，其中 7 条重复 query。旧数据上最容易学到的依然是“一搜降零搜”，而不是“冗余多搜降为必要多搜”，所以门禁给出 `NO-GO`。

项目据此提高 Hotpot bridge 比例，同时保留 NQ 能力。配比使用实际可见检索证据登记，没有为了凑配额放宽标准。

### 阶段 3：旧 XML grouped probe 暴露协议错配

7 月 23 日 grouped probe 完成 320 条轨迹，只有 7 条 strict EM；invalid 185/320、clipped 102/320。312 次搜索中有 210 次 query 是字面量 `query`、`and` 或空值。

这不是“模型完全不会检索”的干净结论，而是 Qwen3.5 native tool calling 与 legacy XML prompt/parser/action boundary 错配。原 outer 因空 query 使分析器退出 1，但 GPU eval 数据已完整；后续只把空 query 纳入失败分类，没有伪造或重跑旧结果。

### 阶段 4：native 协议逐层修复

项目依次建立 native schema、raw-token replay、tool-response mask、严格短答案、on-policy sampling、action region 和 policy prefix 边界：

- native-v1 出现可学习内容，但整段自由文本被当作答案，正式 EM 为 0；
- v2 恢复唯一短答案边界并统一采样合同；
- 首次 v3 因 reasoning marker 与单-token overshoot 触发 answer-boundary 工程失败；
- 修复后 v3 能完成健康检索，主要错误收敛为“看到答案后仍继续搜索”；
- upstream terminal rollout 证明 terminal 路径本身可工作，但模型仍不服从停止机会。

native-v4 又加入 answer-only terminal reminder、Gold 样本审计与更严格的 W&B/evidence gate。51 条有缺陷 gold 记录被剔除并按原分布确定性补位。Gate-v5 的 `GO` 表示工程合同允许进入 RL，不表示 parent 模型已经具备良好停止行为。

### 阶段 5：两步 smoke 与 R60

两步 smoke 完成两次真实全参数 GRPO update，7/16 group 有 mixed reward，loss、KL、entropy、grad norm 和 W&B history 均完整有限，因此训练链路 operational。

R60 随后完成 60 次全参数更新和 2,400 条训练轨迹。在固定 val-128 上，strict EM 从独立 smoke endpoint 的 37.50% 升到 58.59%，平均搜索从 3.117 降到 1.992，证明 direct outcome-RL 学到了能力且没有坍缩成不搜索。

负面信号同样被保留：增益主要来自 NQ，Hotpot 提升较小；step 52 后 KL、生成长度、clipping 和 invalid 同步上升。R60 最终只保存 `global_step_60` HF 权重，没有 optimizer/trainer state。

### 阶段 6：G3 是科学 NO-GO

G3 内层评测完成 320/320 轨迹、724 次搜索，strict EM 为 151/320=47.19%。正确多搜轨迹有 98 条，覆盖 34/64 题；13/64 题出现成本对比，说明模型确有多搜索和成本学习机会。

正式门禁仍失败：

| 指标 | 实际值 | 门槛 | 判定 |
|---|---:|---:|---|
| clean learnable group | 5/64 | 至少 8 | NO-GO |
| clipping | 46.88% | 不高于 5% | NO-GO |
| invalid | 43.75% | 不高于 5% | NO-GO |

clean 轨迹 EM 为 71.15%，dirty 轨迹仅 24.39%。瓶颈不是完全缺少知识或搜索能力，而是长输出和 native action 边界漂移。

G3 的 inner eval 为 `success/0`，但 outer 因严格 replay verifier 缺陷实际为 `failed/1`，没有正式 G3-only marker；它不是脚本预设的受控 `201`。后续 B/C recovery 通过 exact inner trace 与 registered analysis 绑定该科学结论。交接时必须保留真实状态，不能追溯写成 `201` 或 `GO`。

### 阶段 7：第一次 B/C 的真实失败

2026-07-31 的第一次 B/C outer 为 `failed/124`。B20 已完整成功并保存 checkpoint；旧 C 也生成了 800 条训练轨迹，但第 20 步先进入 inline validation，在最终 checkpoint 保存前超时，六个外部评测尚未执行。

根因是预算余量不足叠加“先验证、后保存”的调度顺序，不是 OOM、磁盘写满、NCCL、数值崩溃或 retriever 失败。旧 C partial 没有 checkpoint，最终明确 `old_c_adopted=false`。

### 阶段 8：B/C recovery

恢复没有重算 B。runner 先验证并复用 B，然后从 exact R60 权重独立重训 C；关闭 inline validation，保证先保存 `global_step_20`，再顺序执行 B/C 的 val、NQ-test 和 multihop 六项评测。

恢复 C 使用的 checkout 比 B 多了终点 validation 调度修复。审计确认最后一次参数更新之前的 rollout、advantage 和 actor update 语义未变，但两个 checkout 并非 bit-identical，因此本轮仍是探索性证据，下一次确认性实验必须统一 checkout。

最终八个科学 inner run 全部 `success/0`。outer 的 `failed/203` 是结果 seal 完成后，为冻结 watchdog failure path 设计的受控状态，不表示训练失败。

### 阶段 9：A/R 补评与四模型闭环

2026-08-01 至 08-02，A 和 exact R60 在与 B/C 完全相同的三套 endpoint 上完成六项评测。A/R outer 和六个 inner run 均为 `success/0`，143/143 项远端 evidence manifest 复核通过。

至此，训练前 parent、R60、control B20 和 cost C20 都有同题、同协议、逐题配对结果，本轮实验正式收口。

## 7. 实际 AutoDL 执行架构

标准设计始终分三层：

```text
可信 Git commit
→ CPU 联网准备 / 离线增量 reseal
→ CPU handoff（代码、环境、模型、数据、配置、digest）
→ GPU 结构门 / smoke / 训练 / 评测
→ result contract + lineage + evidence manifest + marker
→ exact-attempt watchdog
→ guest shutdown dispatch
→ 人工确认 provider 控制台已停机
```

固定持久根为 `/root/autodl-tmp/search-r1`，关键布局如下：

| 内容 | 路径或身份 |
|---|---|
| canonical sealed checkout | `/root/autodl-tmp/search-r1/checkout`，commit `f8c1cd7e87078d07385f74ca8710add5d5f79c06` |
| checkout tree digest | `056b40fc6c3ccd979c5e8a3f22b55cdcc8d1be2d17deedf2b2ad8b4c187cc924` |
| recovery checkout | `/root/autodl-tmp/search-r1/recovery-checkouts/ffda96014d67f37751565203931940856a5c5bd8` |
| checkout 外 runner | `/root/autodl-tmp/search-r1/operator/` |
| attempts | `/root/autodl-tmp/search-r1/state/attempts/{git,cpu,gpu}/<attempt>` |
| 全局锁 | `/root/autodl-tmp/search-r1/state/phase.lock` |
| manifests | `/root/autodl-tmp/search-r1/manifests/` |
| model/data | `models/Qwen3.5-2B`、`data/search_mix_qwen35_native_v4` |
| CPU handoff digest | `9ffaa89f88990887b368750ccfdc559d93afdb9a1f5719411cf3e3c1147da90b` |
| native-v4 data manifest digest | `db6cacf865365ee977fc280d5195556e9903907b28b228ba8f3d72133d674167` |

同一路径字符串不证明新实例挂载了同一块盘；接手时仍需核对 provider volume 和 sealed digest。

### 7.1 Launcher、终态和科学成功是三件事

`nohup + setsid` launcher 返回只表示后台 worker 已接纳。普通 phase 完成至少要求：

```text
terminal=success + exit-code=0 + 唯一 .success
```

科学结果还必须验证 result contract、lineage、run index、paired outputs、`evidence.sha256` 和最终 marker。受控 `200/203` 必须按合同解释；真实 `1/124` 必须保留为失败。不能只看日志最后一行，也不能从可能变化的 `state/latest/gpu` 猜 exact attempt。

### 7.2 Scientific seal 顺序

```text
inner terminal / checkpoint / trace
→ contract.env + lineage.tsv + run-index.tsv + paired outputs
→ 重验 checkout / data / checkpoint / semantic contract
→ evidence.sha256 与 digest
→ manifests/<namespace>/<outer>.ok
→ outer result pointers
→ outer terminal sentinel / exit-code / 唯一终态 marker
```

marker 必须最后发布；否则 watchdog 可能把半成品误认成完整结果。

### 7.3 Watchdog 与关机语义

watchdog 必须显式绑定 launcher 当时打印的 exact attempt，等待持久终态后取得全局锁，再复验 host、mount、checkout、attempt、terminal 和 result，`sync` 后才请求无参数 `/usr/bin/shutdown`。

`shutdown-dispatched` 只表示 guest 内 shutdown backend 返回 0；它不证明 AutoDL 控制平面已经停止实例或停止计费。控制台确认始终是独立的人工终态。A/R 专用 `15_watch_qwen_native_ar_eval_only.sh` 已经一次性执行，不应对完成 attempt 再次 arm。

## 8. 实际运行身份与状态

| 阶段 | Outer / inner | 最终状态 | 证据语义 |
|---|---|---|---|
| G0/G1 | outer `20260728T044634Z-2051-4045` | `success/0`，GO | 正式 gate marker |
| smoke | outer `20260728T061246Z-1316-15067`；inner `20260728T061454Z-1352-31865` | `success/0`，GO | 两次真实 update |
| R60 | outer `20260728T092026Z-2906-16060`；inner `20260728T092332Z-2946-8453` | inner `success/0`；outer `failed/200` | 受控封存，不是训练失败 |
| G3 | outer `20260729T073211Z-3516-18146`；inner `20260729T073654Z-3558-15376` | inner `success/0`；outer `failed/1` | verifier 工程失败；科学 NO-GO |
| 首次 B/C | outer `20260731T060828Z-1554-11705` | `failed/124` | 真实超时；仅 B 可采用 |
| B/C recovery | outer/result `20260801T041947Z-7015-12411` | 8 inner `success/0`；outer `failed/203` | 受控封存 |
| A/R final eval | outer/result `20260801T123025Z-1367-15987` | outer 与 6 inner 均 `success/0` | 正常成功 |

关键 result contract 与 digest：

| 结果 | Contract | Evidence digest |
|---|---|---|
| B/C recovery | `qwen-native-training-bc-recovery-v1` | `8ec72644d618c194a248e82965fcf4b7473e69acad0a5c5cd519f9df2c03827c` |
| A/R final eval | `qwen-native-training-ar-eval-only-v1` | `0058974e9cdc0eae9d0313a52ebebd60051c5b483beceab8f83bdcaf7b74a63e` |

## 9. 最终四模型结果

### 9.1 Strict EM 与平均搜索

| 端点 | A | R | B | C |
|---|---:|---:|---:|---:|
| val-128 | 33.59% / 3.234 | 58.59% / 1.992 | 61.72% / 1.930 | **64.84% / 1.680** |
| NQ-test-128 | 7.81% / 2.727 | **25.00% / 2.523** | 23.44% / 2.234 | 24.22% / 2.086 |
| multihop-256 | 21.88% / 3.621 | **37.11% / 2.902** | 30.08% / 3.074 | 29.69% / 2.852 |
| 512 行描述性汇总 | 21.29% / 3.301 | **39.45% / 2.580** | 36.33% / 2.578 | 37.11% / **2.367** |

单元格为“strict EM / 每题平均已执行搜索”。512 行汇总让 256 题 multihop 权重是两个 128 题端点的两倍，因此只作描述，不是自然存在的统一 benchmark。

### 9.2 A→R：本轮最可靠的主结果

R 相比 A 的逐题 paired bootstrap：

| 端点 | EM 差 | 95% CI |
|---|---:|---:|
| val | +25.00pp | [15.63, 34.38] |
| NQ-test | +17.19pp | [10.16, 25.00] |
| multihop | +15.23pp | [9.77, 21.09] |

三个区间都不跨 0。合计 512 题中，R 比 A 多答对 93 题、少执行 369 次搜索。R 的能力提升不是靠更多检索换来的。

必须同时报告负面结果：clipped trajectory 从 A 的 17.97% 上升到 R 的 38.09%。所以准确表述是“能力明确提升，val/multihop 搜索效率改善，同时 clipping 恶化”，不是“R 全面改善”。

### 9.3 B→C：效率改善比准确率改善更可靠

C 相比 B：

- 512 题多答对 4 题；
- 总搜索从 1,320 降到 1,212，少 108 次，即 -8.18%；
- 4-search 饱和率从 38.28% 降到 30.27%；
- 三个端点 C−B EM 的 95% CI 全部跨 0；
- val 与 multihop 的平均搜索下降得到 paired bootstrap 支持，NQ-test 区间跨 0；
- 108 次净节省中有 81 次来自 B、C 都答错的题，主要机制是减少失败轨迹的无效深搜。

C 没有再次坍缩为不搜索：B/C 都只有 1/512 零搜索轨迹。它更像“搜索深度调节器”，而不是“是否检索”的 gate。

### 9.4 R、B、C 的最终定位

- R 在 NQ-test 和 multihop 排名第一；multihop 比 C 多答对 19 题，因此仍是当前综合能力默认 checkpoint。
- C 在 val 第一，三个端点都最省搜索；它是效率候选，不是无条件最佳模型。
- B 的 512 行搜索量几乎与 R 相同，却少答对 16 题；当前没有相对 R 或 C 的部署优势，但必须作为 B/C control 保留。
- A 只作为训练前 parent baseline。

## 10. 协议质量与科学限制

| 模型 | clipped trajectory | 含 invalid 轨迹 | clean 轨迹 |
|---|---:|---:|---:|
| A | 17.97% | 53.71% | 46.29% |
| R | 38.09% | 35.94% | 53.91% |
| B | 70.90% | 34.57% | 16.99% |
| C | 71.68% | 37.89% | 11.52% |

主要限制：

1. B/C 是 G3 `NO-GO` 后的 post-hoc exploratory 实验，不是预注册主流程正式放行。
2. 每个分支只有一个训练 seed；同 seed 提高配对可比性，但不能估计训练随机性。
3. 最终评测每题只有一个 greedy rollout，端点规模为 128/128/256。
4. B 与恢复 C 的 checkout 不完全相同。
5. B/C clipping 极高；更短 token 不能全部解释成主动简洁化。
6. utility 不惩罚 token、invalid 或 clipping，不能当整体可靠性分数。
7. strict EM 对 alias、日期和冗长表达敏感，但它是冻结主指标。
8. 本地紧凑包足以复算统计，不包含完整 checkpoint 和全部 raw trace，不能完全离线重演生成。
9. B/C 166 项 manifest 全部通过，但新 C20 和六个 eval 的 14 个 `terminal/exit-code` 小文件未进入原 seal；六个 eval 的 `run.env` 也没有直接记录 data hash。resolved config、parquet、trace 和 lineage 已交叉补证，但不能事后改写原 manifest。
10. guest shutdown 成功不等于云厂商计费控制面回执。

## 11. 证据地图

### 11.1 结果与分析

- [四模型最终结果分析](qwen35_native_arbc_final_results_analysis.md)：当前 A/R/B/C 数值与模型选择的权威入口。
- [B/C 训练、恢复与完整分析](qwen35_native_bc_recovery_complete_analysis_report.md)：从最初崩溃、数据重配到 recovery 的完整叙事。
- [R60 训练与轨迹分析](../stages/qwen35_native_r60_training_and_trajectory_analysis.md)：R60 训练动态、checkpoint 和 late-stage drift。
- [R60 G3 评测与轨迹分析](../stages/qwen35_native_r60_g3_evaluation_and_trajectory_analysis.md)：G3 科学 `NO-GO` 的权威报告。
- [native-v3 G0/G1 轨迹分析](../stages/qwen35_native_v3_g0_g1_trajectory_analysis.md)与 [terminal rollout 分析](../stages/qwen35_native_v3_terminal_rollout_g0_g1_analysis.md)：native 协议修复前后的轨迹证据。
- [SFT→RL→B/C 后续方案](../plans/qwen35_native_sft_rl_bc_followup_plan.md)：尚未执行的独立新实验提案。

### 11.2 本地紧凑证据包

- [A/R final evidence](../../../results/qwen35-native-ar-eval-20260802/README.md)：21 个本地 payload 复算匹配，远端 143/143 manifest 通过。
- [B/C recovery evidence](../../../results/qwen35-native-bc-recovery-20260801/README.md)：24/24 本地 payload、12/12 关键训练文件匹配，远端 166/166 manifest 通过。
- [第一次 B/C partial audit](../../../results/qwen35-native-bc-partial-audit-20260731/README.md)：解释 B 可采用、旧 C 不可采用和真实 timeout 根因。
- [native-v3 G0/G1 evidence](../../../results/qwen35-native-v3-g0-g1-20260726/README.md)：协议适配阶段的原始证据。

这些目录应长期保留。`docs/results/` 是 Git 内 canonical 紧凑归档；本地 `tmp/` 只是下载、解包、损坏或 partial 文件的工作区，不属于科学交付物，已从版本控制排除。

### 11.3 关键远端路径

```text
R60 inner:
/root/autodl-tmp/search-r1/runs/reproduce/attempts/20260728T092332Z-2946-8453

G3 inner:
/root/autodl-tmp/search-r1/runs/eval/qwen_native_g3/attempts/20260729T073654Z-3558-15376

B20:
/root/autodl-tmp/search-r1/runs/control/attempts/20260731T061302Z-1592-7937

old C partial, not adopted:
/root/autodl-tmp/search-r1/runs/cost_aware_gated/attempts/20260731T110656Z-1592-183

C20 recovery:
/root/autodl-tmp/search-r1/runs/cost_aware_gated/attempts/20260801T042516Z-7061-17197

B/C final result:
/root/autodl-tmp/search-r1/runs/qwen-native-training/attempts/20260801T041947Z-7015-12411

A/R final result:
/root/autodl-tmp/search-r1/runs/qwen-native-training/attempts/20260801T123025Z-1367-15987
```

## 12. 存储与保留策略

必须保留：

- exact A、R60、B20、C20 checkpoint identity；
- G3 完整 trace 和 registered analysis；
- 首次 B/C `failed/124`、旧 C partial 与“不采用”证据；
- B/C recovery 与 A/R 的 outer、inner、contract、lineage、run index、paired results、manifest 和 marker；
- CPU handoff、data manifest、canonical/recovery checkout seals；
- 本地 `docs/results/` 紧凑包和本交接所列分析文档。

不得仅因为磁盘紧张删除 R、B、C、G3 或最终 evidence。若未来必须腾远端空间，应先生成“路径—用途—被谁引用—digest—备份位置”清单，再只删除明确不可采用的重复下载、缓存或 partial 副本。旧失败 attempt 的小型终态和 lineage 应保留，即使大体积 partial trace 最终被另行备份后清理。

## 13. 当前禁止事项

1. 不运行 `09_gpu_qwen_native_train.sh main` 或 `10_gpu_qwen_native_r60_only.sh`；它们会重新训练 R60。
2. 不重跑 G3、B/C recovery 或 A/R final eval；新运行必须另立 contract、namespace 和 attempt。
3. 不把旧执行手册中的“当前唯一操作”当成现在的任务。
4. 不通过 reseal、联网重建或重新执行 `02_cpu_prepare.sh` 掩盖 identity mismatch。
5. 不改写 G3 `failed/1`、首次 B/C `failed/124`、R60 `failed/200` 或 recovery `failed/203` 的历史终态。
6. 不采用旧 C partial；它没有 checkpoint。
7. 不让 C 从 B 继续训练；B/C 的共同科学 parent 必须是 exact R60。
8. 不从 `state/latest/gpu` 猜 attempt，不重复绑定完成过的 watchdog。
9. 不把 launcher admission、日志末行或 `shutdown-dispatched` 当作科学完成证明。
10. 不在文档、Git、日志或命令历史中保存 SSH 密码、token 或其他凭据。

## 14. 后续工作边界

### 14.1 可以继续做

- 基于已封存逐题 CSV 做更深入的错误类型、数据来源和搜索路径分析；
- 把本轮工程失败、协议修复、科学 `NO-GO`、恢复和最终模型选择整理成论文/简历叙事；
- 在不改写原始 seal 的前提下，新增派生统计和分析文档；
- 设计新的多 seed、统一 checkout、预注册 primary endpoint 的确认性实验。

### 14.2 SFT40→RL20→B/C 是独立新实验

SFT warm start 是合理的新假设，因为本轮证明 direct RL 能增强能力，也会积累 clipping 和协议漂移。但它不能被描述为本轮的“继续训练”。新实验至少需要：

- 从 clean 轨迹、规则修订或 teacher 样本构建 SFT 数据；
- 排除 G3、val、NQ-test、multihop 及近重复，封存 source 与去重 digest；
- 同时保留继续搜索和提交答案的正例，避免机械早停；
- 先做格式/terminal/clipping smoke，再决定是否付费训练；
- SFT parent 上的 B/C 必须使用完全相同 checkout、独立 optimizer 和多个 seed；
- 在运行前登记 primary endpoint、非劣界、成本指标和停止规则。

完整方案见 [SFT → RL → B/C 后续实验方案](../plans/qwen35_native_sft_rl_bc_followup_plan.md)。

## 15. 对外表述边界

可以说：

> 在预算缩小的 Qwen3.5 native Search-R1 风格实验中，60 步 direct outcome-RL 在三个同协议端点上都显著提高 strict EM。G3 显示模型具备多次搜索能力，但 clipping 与 invalid action 使正式门禁失败。随后从同一 R60 权重启动的单 seed B/C 探索表明，correctness-gated 成本项可减少检索深度且未发生 no-search collapse；C 相比 B 少 108 次检索，但准确率差异未被确认。最终 R 是综合能力端点，C 是低成本候选。

不能说：

- “C 显著提升了准确率”或“C 全面优于 R/B”；
- “G3 已通过”或“协议漂移已经解决”；
- “首次 B/C 只是卡住但结果完整”；
- “outer failed 都是训练失败”或“outer failed 都是受控成功”；
- “已经验证 SFT+RL 优于 direct RL”；
- “完整复现了 Search-R1 论文榜单”。

## 16. 接手检查清单

- [x] R60、G3、B20/C20 和 A/R 三端点评测均有明确终态。
- [x] A/R/B/C checkpoint digest 已冻结。
- [x] B/C recovery 与 A/R evidence manifest 已复核。
- [x] 四模型同题、同协议结果已完成。
- [x] 旧 C partial 明确不采用，B 未重复训练。
- [x] guest shutdown 与 provider 控制台终态被分开描述。
- [x] 本地 canonical 结果包位于 `docs/results/`。
- [x] 旧 B/C 启动手册已标为历史文档。
- [x] 当前没有待执行 GPU 命令。
- [ ] 后续如需继续研究，先注册新的 SFT/RL 或多 seed 合同，不复用本轮身份。

本交接完成后，本轮实验的默认状态是：**不再运行，保留证据，按需追加分析。**
