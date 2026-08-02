# Qwen3.5 Native B/C Agent Serving 并发补充实验方案

> 文档状态：实施前预注册方案，尚未实现或执行
> 方案分支：`experiment/qwen35-bc-serving-concurrency`
> 分支基线：`main@c0d4cd3cd207fa62e5c54be790f0120cd76f3d41`
> 计划实验名：`qwen35-native-bc-serving-concurrency-v1`
> 创建日期：2026-08-02
> 实验性质：已完成 B20/C20 后的独立、post-hoc serving 补充实验

## 0. 文档状态与当前执行边界

本文件只冻结后续实现与执行合同。当前步骤不安装 serving 依赖、不连接远端、
不加载模型、不运行 GPU、不创建结果数据，也不授权提交、推送或自动关机。

本实验必须与已经封存的 B/C 训练实验分开：

- 不重新训练 B 或 C；
- 不修改 B/C checkpoint、reward、trainer 或原评测结果；
- 不把新的 serving 结果写回旧 B/C evidence namespace；
- 不把本实验解释为对 cost-aware RL 的多 seed 因果确认；
- 不因 serving 结果改变已完成实验中“C 搜索效率更好、能力提升未确认”的结论。

计划中的正常执行链为：

```text
实现并验证代码
  -> 提交并推送一个精确 40 位 commit
  -> CPU 联网准备并发布新实验 handoff
  -> GPU 离线 admission 与 serving 兼容门
  -> 固定 token 模型层微基准
  -> BM25 独立容量基准
  -> 完整 Agent 闭环并发粗扫
  -> 容量边界细扫与完整 512 题确认
  -> 开环 Poisson 流量确认
  -> 配对分析、结果 seal 与证据归档
  -> exact-attempt watchdog 请求 guest shutdown
  -> 人工确认 AutoDL 控制台已停止计费
```

只有前一阶段的身份、终态和 evidence 全部通过，后一阶段才可 admission。任何阶段的
入口返回只表示 worker 已提交，不表示该阶段已经成功。

## 1. 执行摘要

这个补充实验值得做，但问题必须定义成“C 是否提高完整 Agent 服务的有效容量”，
而不是“C 的权重是否突然能在显存里塞下更多并发”。B 和 C 都是同一套
Qwen3.5-2B 架构；固定输入、输出 token 后，两者的 FLOPs、KV cache 结构和静态显存
需求原则上应接近。

C 可能获得的 serving 优势来自单位 Agent 请求工作量下降：历史 512 题上，C 相对 B
少执行 108 次检索、观测到的生成 token 少 39,648 个。若这种行为在在线服务中仍然
成立，它可能降低模型轮次、BM25 压力和请求驻留时间，进而提高 SLO 内完成的任务数。

推荐实现的是一套中等规模的在线服务实验：独立的现代 serving 环境、异步 Agent
gateway、固定 workload、三层基准、完整遥测、质量感知 goodput、不可覆盖的 attempts
和安全关机。预计工程量为 3–5 人日、10–15 个新增文件、2–4 个小范围修改文件；
预计使用双卡实例 8–14 小时，执行硬上限为 16 个实例小时。

本实验最理想的结果不是“模型层 C 神奇地更快”，而是：

> 固定 token 时 B/C serving 内核性能等价；在相同硬件、共同 SLO 和质量非劣条件下，
> C 因搜索更少、轨迹更短而提高完整 Agent 的 SLO-goodput 或最大可持续 RPS。

如果没有提升，只要证据完整并能定位 GPU、队列或 BM25 瓶颈，也属于有效且可对外
说明的工程结果。

## 2. 历史证据与实验动机

### 2.1 B/C 已完成结果

下表来自已封存的 512 行描述性合并结果。三个 endpoint 的正式结论仍须分别解释，
不能把 512 行人为加权汇总当作自然统一 benchmark。

| 指标 | B20 | C20 | C−B |
| --- | ---: | ---: | ---: |
| strict EM | 186/512，36.33% | 190/512，37.11% | +4 题，+0.78pp，仅描述 |
| executed searches | 1320，2.578/题 | 1212，2.367/题 | -108，-8.18% |
| observed tokens | 898,309，1754.5/题 | 858,661，1677.1/题 | -39,648，-77.4/题，-4.41% |
| 含 invalid 的轨迹 | 177/512，34.57% | 194/512，37.89% | +17，+3.32pp |
| clipped 轨迹 | 363/512，70.90% | 367/512，71.68% | +4，+0.78pp |
| 无 clip 且无 invalid | 87/512，16.99% | 59/512，11.52% | -28，-5.47pp |

端点级历史结果为：

| Endpoint | B EM | C EM | B 搜索 | C 搜索 | 搜索变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| val-128 | 79/128，61.72% | 83/128，64.84% | 247 | 215 | -12.95% |
| NQ-test-128 | 30/128，23.44% | 31/128，24.22% | 286 | 267 | -6.64% |
| multihop-256 | 77/256，30.08% | 76/256，29.69% | 787 | 730 | -7.24% |

三个 endpoint 的 C−B EM 区间都跨 0，所以历史结果没有确认 C 提升了答案能力。
多跳来源中还存在一个必须预注册的风险：C 在 HotpotQA 上相对 B 多答对 6 题，
却在 2WikiMultiHopQA 上少答对 7 题。

### 2.2 历史耗时只能作为先验

六个顺序 HF 外部评测阶段的墙钟时间为：

| Endpoint | B | C |
| --- | ---: | ---: |
| val | 1445s | 1286s |
| NQ-test | 1393s | 1456s |
| multihop | 3203s | 3141s |
| 合计 | 6041s | 5883s |

C 合计少 158 秒，即墙钟时间约 `-2.62%`；机械换算的串行 throughput 趋势约为
`+2.69%`。这些阶段使用 HF 离线 rollout、固定 batch 和顺序调度，不是 HTTP serving
或并发压测，因此只能支持“值得进一步测量”的先验，不能被写成并发提升证据。

### 2.3 为什么不能只比较 QPS

B/C 的真实 generation clipping 都在 71% 左右。C 输出更短可能来自更早正确停止、
更少搜索，也可能部分来自截断、invalid 或错误快速结束。因此主实验必须同时报告：

- 任务吞吐和延迟；
- strict EM；
- invalid、clipping 和 terminal 状态；
- 搜索次数、模型调用次数和 token；
- 在共同质量与延迟门槛内完成的 goodput。

只要质量门没有通过，就只能表述为 cost-quality trade-off，不能表述为 C 的部署能力
全面优于 B。

## 3. 不可变实验身份

### 3.1 Checkpoint 身份

本实验只允许使用以下两个完整 Hugging Face checkpoint：

| 模型 | 远端封存路径 | Checkpoint tree SHA-256 |
| --- | --- | --- |
| B20 / control | `/root/autodl-tmp/search-r1/runs/control/attempts/20260731T061302Z-1592-7937/checkpoints/actor/global_step_20` | `8df3d6ab13e154b4a2360b1535f763731f955baebba21b75f9f6e017000ad6d2` |
| C20 / cost-aware-gated | `/root/autodl-tmp/search-r1/runs/cost_aware_gated/attempts/20260801T042516Z-7061-17197/checkpoints/actor/global_step_20` | `106b628945cdb46ea1053fa042e7e75fb26608e15e83ffce65dd799b96fa4b2e` |

这两个路径来自 2026-08-01 的封存合同；当前远端实例或持久盘是否仍可访问，必须在
后续 CPU 阶段 live 验证。路径存在不等于 checkpoint 有效，必须重新计算完整 tree
digest，并验证至少包括模型权重、`config.json`、generation config（若存在）、tokenizer
和 chat template。

若 tokenizer 没有完整保存在 checkpoint 内，必须使用原训练时固定 tokenizer 目录，
记录它的文件清单和 digest；不得自动回退到 Hub 上的当前分支。

任何 checkpoint 字节、架构字段、tokenizer、special token ID 或 chat template 变化，
都使旧 serving attempt 失效，必须创建新实验版本，不能沿用本合同结果。

### 3.2 Workload 身份

主 workload 固定为历史三组 paired 结果对应的 512 个唯一问题：

| 来源 | 数量 | 占比 |
| --- | ---: | ---: |
| NQ | 192 | 37.5% |
| HotpotQA | 192 | 37.5% |
| 2WikiMultiHopQA | 128 | 25.0% |
| 合计 | 512 | 100% |

原始 endpoint 为：

- `val_128.parquet`：NQ 64 + HotpotQA 64；
- `nq_test_128_native_v4.parquet`：NQ 128；
- `multihop_eval_256_native_v4.parquet`：HotpotQA 128 + 2WikiMultiHopQA 128。

新 workload builder 必须从已封存的 paired CSV 与 exact parquet 交叉生成一个 canonical
manifest，至少绑定：

- `typed_sample_id`；
- source 和 endpoint；
- question；
- 全部 gold answers；
- 原始行号；
- canonical JSONL SHA-256；
- 每个 source 的数量与顺序 hash；
- 四个预生成正式 run 排列的 seed 与 order hash；
- 每个开环 block 的 arrival seed、完整 inter-arrival 时间表和 SHA-256。

不得删除困难题、选择 C 更有利的题目、重复样本扩大质量样本量，或根据粗扫结果改变
来源配比。512 行是人工注册的部署流量混合，不是自然统一 benchmark；主报告必须同时
给出 NQ、HotpotQA、2Wiki 三个来源切片。

### 3.3 Agent 协议身份

以下协议在 B/C 之间完全冻结：

- tool protocol：`qwen35_native`；
- prompt version：`qwen35-native-search-v4-terminal-answer-only`；
- 最多 4 次已执行搜索；
- BM25 top-k 3；
- 每轮 observation 上限 500 token；
- 每轮 generation 上限 500 token；
- trajectory/prompt 上限沿用历史合同；
- 超过搜索预算后只能进入 answer-only terminal generation；
- 正式质量基线使用 greedy、`do_sample=false`、group size 1、seed 42；
- search action、invalid action、terminal answer 和 clipping 语义沿用现有 parser。

正式 Agent 服务必须复用仓库的 prompt 渲染、`Qwen35Conversation`、`parse_action` 和
terminal 规则。不得用 serving 引擎自带的新 tool parser 替换它，否则测到的是
“新 parser + 模型”，不再是原 B/C 协议。

模型 API 也属于冻结协议：gateway 使用项目 tokenizer 渲染后的 raw prompt 调用
`/v1/completions`，不能再经 `/v1/chat/completions` 二次套用 chat template。正式 Agent
路径统一使用 streaming 以记录 TTFT/ITL，但必须继续消费完整 generation，不能看到
第一个 action delimiter 就提前取消。

服务必须返回真实 sampled token IDs；gateway 将它们直接交给
`slice_first_complete_native_action()`，再把最短 action-prefix IDs 传给
`Qwen35Conversation.append_followup()`。禁止只把返回文本重新 tokenize。若固定的 vLLM
接口无法可靠返回 token IDs，应在 compatibility gate 失败并改用 token-aware adapter
或 HF fallback，不能静默降低语义合同。

历史 HF 语义是“先采样到 EOS/500 token，再保留第一个完整 native action 的最短 token
前缀”。主实验继续采用该语义：delimiter 后的 raw tail 仍消耗模型算力并计入 raw
generated tokens，但不进入下一轮 conversation。必须同时保存 raw sampled IDs、action
prefix IDs、raw tail IDs、boundary 类型和 boundary-token overshoot。流式检测 delimiter
后立即取消属于未来独立 serving 优化实验，不得混入 B/C 主比较。

## 4. 概念、估计量与术语

### 4.1 模型层并发与 Agent 并发

本实验区分三个系统：

1. **模型服务层**：只执行固定输入/输出 token 的生成，不调用工具；
2. **检索层**：只压测 BM25 `/retrieve`；
3. **完整 Agent 层**：模型生成 search action，调用 BM25，拼接 observation，再继续生成。

模型层用于检验 serving 公平性，不能直接回答 C 的 Agent 策略是否更省。完整 Agent 层
才承担主要部署结论。

### 4.2 最大可持续并发 `N*`

`N*` 定义为：在已经实际测试的并发档位中，所有正式重复均通过共同 SLO、服务和质量
门的最高在途请求数。

它不是理论最大值，也不是 vLLM 的 `max_num_seqs`，更不是“没有 OOM 的最高数字”。
如果更高并发只增加排队、没有提高 tasks/s 或 goodput，只能说服务容纳了更多在途
请求，不能说吞吐或容量提高。

### 4.3 最大可持续开环 RPS `R*`

`R*` 是离散容量指标：在 Poisson 到达、共同绝对 offered load 下，所有正式重复均通过
共同 SLO、队列稳定和质量门的最高已测任务到达率。它只报告点值和已测 pass/fail
区间，不伪造连续置信区间。

闭环 `N*` 更直观，开环 `R*` 更接近后端容量规划；连续主推断使用第 15.3 节的
`stress_goodput_ratio`。最终报告必须同时给出三者。

### 4.4 Goodput

定义以下速率：

- `task_throughput`：完成的 Agent 任务数/秒；
- `SLO_goodput`：在端到端延迟 SLO 内成功完成的任务数/秒；
- `correct_goodput`：在 SLO 内完成且 strict EM 正确的任务数/秒；
- `clean_correct_goodput`：在 SLO 内完成、strict EM 正确、无 invalid 且无 clipping 的任务数/秒。

vLLM 自带 goodput 只理解 TTFT、TPOT 和 E2E 延迟；strict EM、协议状态和搜索轨迹必须
由本项目分析器计算。

## 5. 研究问题与预注册假设

### 5.1 研究问题

`RQ0`：B/C 在新的在线 serving 栈中是否仍保持原 Qwen3.5 native 协议语义和可接受的
质量一致性？

`RQ1`：固定输入与输出 token 后，B/C 的纯模型吞吐、TTFT、TPOT 和 KV cache 行为是否
工程等价？

`RQ2`：在真实 Agent 请求中，C 是否继续减少搜索、模型调用、生成 token 或请求驻留
时间？

`RQ3`：在共同 SLO 和质量非劣前提下，C 的 `R*`、`N*`、SLO-goodput 或
correct-goodput 是否高于 B？

`RQ4`：若 C 没有提高容量，瓶颈位于 GPU 解码、KV cache、gateway 排队、BM25、CPU
还是协议失败？

### 5.2 预注册假设

`H-engine`：在预注册主点 `2048->500, concurrency=16`，B/C output-token-throughput
ratio 的 90% 等价区间应落入 `[0.95,1.05]`。若不通过，优先审计 serving 配置、
checkpoint 格式、GPU、热状态或测量混杂，不得直接归因于 RL 权重。

`H-work`：C 在完整 Agent 层继续表现出较低的 searches/task 或 tokens/task；这是机制
假设，不单独构成容量提升。

`H-capacity`：在 `R_quality` 上 B/C 都通过共同 SLO；在 `R_stress` 上 C 仍通过而 B
允许因容量不足失败，C/B SLO-goodput ratio 点估计至少为 1.05、paired block 95% CI
下界大于 1.0；离散辅助指标 `R*_C/R*_B` 也至少为 1.05，并同时通过模型间质量非劣门
与并发质量漂移门。

判定区间为：

- **正向**：`stress_goodput_ratio>=1.05` 且 95% CI 下界 `>1.0`，离散容量比
  `R*_C/R*_B>=1.05`，全部质量门通过；
- **工程等价**：等价性区间完全落入 `[-5%, +5%]`；
- **不确定**：点估计有方向但区间跨 0 或 ABBA 方向不一致；
- **负向**：容量降低、质量门失败，或收益只来自错误/截断更快结束。

## 6. 结论边界与主要威胁

本实验有以下不可消除的限制：

1. B/C 都只有一个训练 seed；serving 重复不能替代训练 seed 方差；
2. B/C 历史训练 checkout 不是 bit-identical，不能把 serving 差异写成 reward 的普遍因果效应；
3. 本实验在看过 B/C 科学结果后设计，属于 post-hoc 补充实验；
4. B/C clipping 约 71%，长度和提前结束存在删失混杂；
5. 512 题来源混合是人工部署权重，汇总结果不代表所有真实流量；
6. 2Wiki 已有质量回退风险，不能被总体均值掩盖；
7. 历史评测使用 HF rollout，新实验计划使用现代在线 serving，存在引擎迁移风险；
8. 单机双卡和本地 BM25 不是生产集群，结果只描述固定硬件与合同下的容量。

所以即使结果正向，允许的结论也是“该固定 B/C checkpoint 对在本实验部署合同下的
有效容量差异”，而不是“cost-aware RL 普遍提高大模型并发”。

## 7. Serving 架构与冻结配置

### 7.1 独立 serving 环境

根项目的训练依赖约束属于历史训练引擎，不能为了 serving 直接升级。新实验必须建立
独立、完全锁定的 serving 环境或容器，并把全部可执行文件和依赖保存在持久盘。

Qwen3.5-2B 的官方模型卡和 vLLM 支持矩阵已经提供 serving 路径，但官方基础模型受
支持不等于本项目 RL checkpoint、RTX 5090 和原生 Agent 协议已经兼容。因此正式 sweep
前必须通过第 11 节的 compatibility gate。

版本合同要求：

- 禁止只记录 `latest`、`main`、`nightly` 或浮动 `pip install vllm`；
- 优先固定正式版本、wheel SHA-256 或 Docker image digest；
- 若只能使用 nightly，必须固定到 vLLM commit 级 wheel 或镜像 digest；
- 记录 vLLM commit/version、PyTorch、CUDA、Triton、FlashInfer、driver 和 `collect-env`；
- GPU 阶段不得联网安装、下载或自动换版本；
- 不默认启用 `trust_remote_code`；只有兼容证据要求时才固定对应 code revision。

### 7.2 正式模型服务配置

B/C 使用相同命令，只允许模型路径和结果标签不同：

- 一张 GPU 一个 2B replica；
- `tensor_parallel_size=1`；
- BF16，不量化；
- text-only / `language-model-only`；
- 相同 `max_model_len`；
- 相同 `gpu_memory_utilization`；
- 相同 `max_num_batched_tokens`；
- 相同 `max_num_seqs`；
- 显式 tokenizer 路径和 digest；
- 使用引擎默认配置隔离模式，并在每个请求显式传入生成参数；
- 显式使用 `--generation-config vllm`，禁止 checkpoint 内默认值静默覆盖请求；
- 主实验关闭 MTP/speculative decoding；
- 主实验关闭 prefix caching；
- 主实验不使用 FP8 KV cache、不同 attention backend 或自动调参；
- compatibility 调试可统一使用 eager，正式性能模式是否关闭 eager 由门禁一次冻结；
- B/C 不得使用不同的 CUDA graph、cache 或 generation config。

历史 Hydra 中的 `max_num_seqs=1024` 只是旧 rollout 配置，不代表业务可以承受 1024
并发，也不得直接复制成 serving 结论。

`max_model_len` 必须根据本项目真实 prompt/trajectory 合同和 golden 样本最大 token 数
设置，不能无条件使用 Qwen 原生超长上下文；过大的上下文会挤压 KV cache。CPU 阶段先
统计 P50、P90、P99 和 max，再把一个双方相同、能覆盖正式 workload 的值写入 handoff。

### 7.3 Agent Gateway

完整 Agent 层使用项目自有异步 gateway：

```text
load generator
  -> bounded Agent request queue
  -> per-request Qwen35Conversation state
  -> vLLM OpenAI-compatible model endpoint
  -> existing native action parser
  -> async BM25 client
  -> terminal answer and trace recorder
```

Gateway 必须具备：

- 有界队列和明确 backpressure；
- request ID、sample ID、model ID 和 run ID 全链路传播；
- 单请求总 timeout 和每轮 model/retrieval timeout；
- 客户端断开与 cancellation 记录；
- 每一轮模型、检索、排队和解析计时；
- 正式实验关闭自动 retry；
- 任何 retry 仅用于独立故障诊断，并计入原始负载；
- HTTP 成功与协议成功分别记录；
- 原始 finish reason、token 使用量和 terminal 原因落盘。

正式模型请求必须采用 raw completion + streaming + token-ID response 合同，并完整消费到
EOS 或 500-token 上限；gateway 只在 generation 完成后切出首个完整 action。任何提前
cancel、server-side tool parser 或文本再 tokenize 都使 attempt 无效。

### 7.4 BM25

BM25 沿用相同 wiki-18 corpus、索引、offset、top-k 3 和 `/retrieve` 契约。当前 server
内部对 batch query 逐项搜索，endpoint 也是同步函数；共享 Pyserini searcher 的并发
安全和容量尚未验证。因此 BM25 必须先单独压测，不能把它的饱和点归因于 B/C 模型。

如果 BM25 需要优化，优化必须：

1. 作为独立工程阶段；
2. 对 B/C 完全相同；
3. 产生新 BM25 config/index/server digest；
4. 使此前完整 Agent 测试失效并全部重跑。

## 8. 硬件、公平性与运行顺序

### 8.1 主实验硬件

计划使用同一台两张 RTX 5090 级 GPU 的 AutoDL 实例。GPU admission 必须记录：

- GPU 型号、UUID、显存和驱动；
- CUDA runtime/toolkit；
- 功耗上限、时钟、温度和 ECC/错误状态；
- CPU 型号、核数、内存和 NUMA；
- 持久盘 mount、文件系统和可用空间；
- 是否存在其他 GPU/CPU 密集进程。

2B 模型优先采用单卡 TP=1。主比较不使用两卡 tensor parallel，因为通信开销会污染
B/C 权重比较。

### 8.2 对照顺序

完整 Agent 主实验顺序执行，避免 B/C 同时争抢 BM25、CPU、内存和磁盘。正式确认采用
四个配对 block，并交换 GPU：

```text
block 1: GPU0 上 B -> GPU0 上 C
block 2: GPU1 上 C -> GPU1 上 B
block 3: GPU0 上 C -> GPU0 上 B
block 4: GPU1 上 B -> GPU1 上 C
```

每次切换模型都使用同一重启、健康检查和 warmup 流程。不得只让 B 占 GPU0、C 占
GPU1 同时跑一次就作结论。

固定 token 模型层可以在两个 GPU 上并行取得诊断数据，但正式结论仍必须包含上述
swap 后的配对结果。

### 8.3 热状态和资源污染

每个正式 run 前：

- 停止未登记的模型进程；
- 确认 GPU memory 回到基线；
- 记录 60 秒 idle 状态；
- 用相同 warmup 请求预热；
- warmup 不计入结果；
- 正式窗口中持续记录温度、功耗、utilization 和 throttling reason。

若出现热降频、其他进程抢占、GPU reset 或服务重启，该 run 作废但必须保留，不能
覆盖后假装没有发生。

## 9. 共同 SLO 与质量门

### 9.1 项目工程 SLO

以下是本项目预注册工程 SLO，不是行业通用标准：

- HTTP 完成率 `>=99%`；
- p95 Agent 端到端延迟 `<=30s`；
- p99 Agent 端到端延迟 `<=60s`；
- 超过 `120s` 统一记为 timeout；
- 无 OOM、CUDA fatal、服务进程重启；
- 全部 offered 请求中，在各自 scheduled arrival 后 30 秒内成功完成的比例 `>=98%`；
- 按第 15.2 节固定窗口计算的稳态 queue slope 和最终 backlog 通过；
- 全部 arrival 结束后 120 秒内完成 drain。

另报告 15 秒和 60 秒 E2E goodput 敏感性结果，但不得用它们替换 30 秒主 SLO。

若任一模型在并发 1 的完整 512 题基线上已经无法满足 30/60 秒 SLO，正式 sweep
NO-GO。可以基于该事实另行注册 v2 SLO，但不能在看到 B/C 对比结果后修改 v1。

固定 token 模型层另报：

- p99 TTFT；
- p99 TPOT；
- p99 ITL；
- p99 E2E；
- request/token throughput；
- 失败率和 KV cache usage。

TTFT 3 秒、TPOT 100ms 可作为诊断参考线，但不将其冒充行业 SLA；模型层的主任务是
确认双方配置公平和性能近似，而非选取有利阈值。

### 9.2 HTTP 门与协议质量门分离

历史 B/C invalid 和 clipping 很高，所以不能要求“协议合法率 99%”作为 HTTP 服务
健康门。必须分别报告：

- transport success；
- JSON/API contract success；
- Agent terminal success；
- invalid action；
- generation clipping；
- strict EM。

在线 clipping 定义为任一真实 generation 因 `max_tokens`/length 达到上限；gateway
排队 timeout、客户端取消和 HTTP timeout 不是 clipping，必须单独分类。

质量分母固定为全部 offered 题目：timeout、HTTP 失败、无 terminal answer 和取消均按
该题 strict EM=0 计入，同时保留独立失败类别。不得只在成功完成的请求上计算 EM，
否则高负载下会产生幸存者偏差。

### 9.3 模型间质量非劣门

在新 serving 栈、并发 1、完整 512 题上定义：

```text
delta_em = EM_C - EM_B
```

按来源分层的题目级 paired bootstrap 计算单侧 95% CI。非劣界预注册为 `-3pp`：

```text
one-sided 95% CI lower bound > -3pp
```

不通过时仍可报告吞吐，但只能写成 cost-quality trade-off，不能说 C 是更优部署模型。

2Wiki 128 题作为强制风险切片单列。由于切片样本较小，不单独设置确认性非劣硬门；
但任一正式质量基线或 stress point 上 `EM_C-EM_B <= -5pp` 时，正向结论必须限定为
NQ/Hotpot 流量，并明确披露 2Wiki 回退。

### 9.4 并发引入的质量漂移门

每个模型的高负载结果相对自身并发 1 基线必须满足：

- strict EM 差的单侧 95% CI 下界 `>-2pp`；
- invalid 和 clipping 点估计增幅各 `<=2pp`；
- invalid 和 clipping 增幅的单侧 95% CI 上界各 `<=5pp`；
- final answer 一致率、搜索/action 轨迹变化完整报告。

此外，在每个正式高负载主点直接检验同题 `EM_C-EM_B` 的单侧 95% CI 下界是否
`>-3pp`，不能只依赖并发 1 的模型间门和各自相对基线门。

四个正式 block 中，每题先得到四个 `0/1` EM，再对每题取跨 block 均值；随后以 512
个题为分析单元做来源分层 paired bootstrap。并发 1 也采用同样四个 block 聚合方式。
重复运行不会把质量有效样本量扩大为 2048；重复只用于估计系统运行方差和检测并发
导致的非确定性。

## 10. Workload 与流量生成

### 10.1 正式题目顺序

预生成四个确定性排列，每个排列保持 512 题来源总量不变，并绑定 seed 和 order hash。
同一个 block 内 B/C 使用完全相同的排列。不能让 B/C 使用不同题序，也不能在运行时
根据模型完成速度动态补入不同题目。

粗扫用 128 题分层子集：

- NQ 48；
- HotpotQA 48；
- 2Wiki 32。

粗扫子集和顺序在 CPU handoff 时一次冻结；不得按历史正确率或粗扫结果重新挑选。

### 10.2 超时、失败和重试

正式客户端关闭自动 retry。一个 request ID 必须对应一个业务任务和一条终态记录。

- HTTP 429/5xx、连接失败、timeout、取消都计入 offered load；
- 诊断重试必须使用新 attempt，不得覆盖原 request；
- 超时请求即使稍后服务端完成，也同时记录 late completion，但主结果按 timeout 失败；
- 不能删除 warmup 以外的“异常慢请求”。

### 10.3 客户端容量

Load generator 应与服务分进程，最好使用独立 CPU affinity；若同机运行，必须证明
客户端 CPU、tokenizer 和 socket 没有先饱和。开环测试的客户端最大并发必须高于服务
容量，不能让客户端 semaphore 形成假上限。

## 11. Phase G0：GPU Serving 兼容门

兼容门在全部正式 sweep 之前执行，硬上限建议 1 个实例小时。

### 11.1 静态身份检查

B/C 必须分别通过：

- checkpoint tree digest 与第 3.1 节完全一致；
- `config.json` architecture、层数、hidden size、vocab、dtype 一致；
- 权重文件清单完整且无临时/partial 文件；
- tokenizer 文件、special token ID、chat template digest 一致；
- prompt renderer 在两模型上产生完全相同的 token IDs；
- serving image/wheel 与环境 freeze 匹配 CPU handoff；
- GPU 全程 offline，缺失依赖或模型时 fail closed。

### 11.2 服务启动与基础观测

分别启动 B/C，保存：

- 完整启动命令与日志；
- `/v1/models`；
- `/metrics` 初始快照；
- vLLM collect-env；
- `nvidia-smi -q` 与持续采样；
- GPU KV cache 容量和实际服务参数；
- 单请求与 streaming smoke。

启动日志中的理论 maximum concurrency 只作为 KV cache 诊断，不能替代本实验 `N*`。

### 11.3 语义对齐

使用 32–64 条固定 golden prompts 和现有 HF 路径作为 oracle：

- prompt token IDs 必须 100% 一致；
- canned action/observation/terminal 状态机测试必须 100% 一致；
- search budget、terminal search 拒绝、invalid 分类必须一致；
- 记录 greedy 输出逐 token 差异和 parser 级差异；
- live 输出允许因内核数值产生少量 token 分叉，但 strict EM、平均搜索、clipping 和
  invalid 不得越过预注册 compatibility tolerance；
- 相对 HF oracle 新出现、无法由现有分类器解释的空输出/乱码，以及任一 NaN、无限循环、
  预算越界或 system prompt 丢失立即 NO-GO；若 HF oracle 在同题产生相同的既有
  invalid/空输出，则进入质量统计，不把模型旧失败误判为 serving 迁移失败。

Compatibility tolerance 在实现时固定为机器可读 contract，至少包括：

- 64 题 strict EM 相对 HF oracle 的绝对变化不超过 2 题；
- 平均 executed search 绝对变化不超过 0.25/题；
- clipping 与 invalid 绝对变化各不超过 5pp；
- 0 个未被现有分类器解释的新协议错误类型。

若 tolerance 失败，不能静默放宽。应回退到固定版本的 HF 动态批处理服务，或修复
serving 语义后创建新 attempt；回退方案预计增加 1–2 人日。

### 11.4 小并发稳定性

兼容门最后在并发 `2/8/16` 各运行一组短测试，确认：

- 无 OOM、CUDA graph error、worker crash；
- 队列能回落；
- request/response 数量完全对账；
- gateway 不丢 request ID；
- BM25 和 model timeout 分类正确。

## 12. Phase A：固定 token 模型层微基准

### 12.1 目的

隔离 EOS、搜索轮数和策略行为，回答相同工作量下 B/C serving 是否近似等价。

### 12.2 长度矩阵

至少覆盖：

| 场景 | Input tokens | Output tokens |
| --- | ---: | ---: |
| 常规短请求 | 1024 | 128 |
| Agent 长轨迹 | 2048 | 500 |
| 上下文压力 | 4096 | 500 |

如果 CPU 阶段从历史轨迹计算出的 P50/P90 与上述差异较大，再增加对应 P50/P90 档位，
但不得删除这三档。所有请求使用固定 token 长度、temperature 0、固定 seed 和
`ignore_eos`。

### 12.3 并发与重复

闭环粗扫：

```text
1, 2, 4, 8, 16, 32, 64
```

只有 64 仍通过且 GPU/KV 尚有明显余量时才允许追加 128。每档：

- warmup 至少 20 个请求；
- 三个长度的全阶梯粗扫各使用 128 个请求、单次运行；
- `2048->500, concurrency=16` 是预注册主等价点，使用 512 个请求、四个 ABBA/swap block；
- `2048->500` 的共同 knee 与相邻点使用 512 个请求、独立重复 3 次；
- 另外两个长度只在 concurrency 1 和共同 knee 使用 256 个请求、重复 2 次；
- 保存逐请求结果，不只保存汇总；
- 使用相同 prompt token 数据和顺序。

模型层使用官方 `vllm bench serve`；它支持 `max-concurrency`、request rate、
`ignore-eos`、TTFT/TPOT/ITL/E2E 百分位和详细结果保存。

### 12.4 判读

主等价指标是 `2048->500, concurrency=16` 的 output-token-throughput ratio。使用四个
paired block 在 log ratio 上计算 90% 等价区间；只有区间完整落入 `[0.95,1.05]` 才
通过 `H-engine`。TTFT、TPOT、ITL、KV cache 和其他长度作为支持指标，不从中事后
挑选更有利的主结论。

- 主等价区间落入 `[0.95,1.05]`：通过 serving 公平性门；
- 稳定超出等价带：先审计 GPU、服务参数、tokenizer、cache、热状态和 checkpoint；
- 只有差异在 swap 与 ABBA 后仍存在，才可作为“观察到权重相关执行差异”的诊断；
- 即使 C 模型层更快，也不能直接说 RL 改变了 FLOPs 或显存容量。

## 13. Phase B：BM25 独立容量基准

### 13.1 Query workload

从历史 B/C 封存 trace 中构建固定 query 集：

- 保留原 query 文本和来源 sample ID；
- 去重版和保序原始版分别生成 manifest；
- 主基准使用保序原始版，反映真实重复 query；
- 记录 query 长度分布、空 query、invalid query 和 top-k；
- B/C 使用同一 query 集，不能分别用各自 query 造成工作量混杂。

### 13.2 负载矩阵

闭环并发：

```text
1, 2, 4, 8, 16, 32, 64, 128
```

每档至少 1000 个 query 或持续 2 分钟，独立重复 3 次。记录：

- QPS；
- p50/p95/p99 latency；
- timeout/error；
- CPU、RAM、threadpool 和 open files；
- 相同 query 在并发下的 document ID/score 一致性；
- Pyserini shared searcher 是否出现竞态或串行化。

### 13.3 Admission 判定

Phase B 只先产出完整 BM25 容量曲线。此时正式 Agent target RPS 尚未知，因此只用历史
搜索率和保守预估做预检，不能提前把未知 target 写成 NO-GO。

完成 Agent 粗扫并得到确认后的 target RPS 后，再机械计算：

```text
required_retrieval_qps = target_agent_rps * searches_per_task
```

其中 `searches_per_task` 使用同一负载点 B/C 较大的实测值。BM25 稳定容量必须满足：

```text
bm25_stable_qps >= 1.25 * target_agent_rps
                           * max(B_searches_per_task, C_searches_per_task)
```

若最终不满足，停止正式容量归因：

- 暂停完整 Agent 容量归因；
- 报告“检索后端上限”；
- 决定是否进入独立、双方共用的 BM25 优化阶段；
- 优化后重新运行全部完整 Agent 主比较。

## 14. Phase C：完整 Agent 闭环并发

### 14.1 并发 1 质量基线

先对 B/C 各运行一轮完整 512 题，建立新 serving 栈的 preliminary 质量与单位工作量
基线。该单 block 只用于 screening，必须通过：

- 第 9.1 节 SLO；
- checkpoint、workload 和 protocol identity；
- request、model call、retrieval call 和终态数量对账；
- 不出现超出 G0 compatibility tolerance 的新语义错误。

单 block 不执行正式 `-3pp` 非劣判定，也不能因单次点估计不利而 NO-GO。只有身份、
对账、SLO 或 compatibility screening 失败才停止。进入粗扫前或最迟进入任何正式
容量确认前，必须按 ABBA/GPU swap 补齐到每模型四个正式并发 1 block；第 9.3、9.4
节只使用四 block 的逐题均值。

### 14.2 粗扫

固定 128 题分层子集，闭环并发：

```text
1, 2, 4, 8, 16, 32, 64
```

每档单次粗扫，统一 warmup。64 通过时才允许继续。连续两个更高档均失败后停止升压，
防止为了得到更大数字继续烧卡。

### 14.3 边界细扫

粗扫得到第一个 pass/fail 区间后，在区间内按约 10%–12.5% 粒度或二分细扫。例如
16 通过、32 失败时，可以测试 `20/24/28`，再对最后边界补一个整数邻点。

仅使用 2 倍阶梯无法检测预期 0–10% 的小幅差异，所以细扫不可省略。

### 14.4 正式容量确认

对每个模型的候选 `N*`、其下一档和对方候选点，运行：

- 完整 512 题；
- 四个 ABBA 配对 block；
- GPU swap；
- 相同预生成题序；
- 每次独立重启与 warmup；
- 完整逐请求和系统遥测。

每个正式高负载主点同时执行第 9.4 节的直接 `C-B` paired 非劣检验；不能只分别比较
B/C 与自身并发 1。所有 timeout、429、5xx、取消和无终态请求均按 intent-to-serve
口径记该题 EM=0。

最终只报告“本次已测试的最大稳定档位”，不得声称数学意义上的绝对最大并发。

## 15. Phase D：开环 Poisson 流量确认

### 15.1 流量矩阵

闭环粗扫只用于找区间，不能用单次 128 题最大值锚定开环流量。完成第 14.4 节四个
确认 block 后，以双方各自在候选容量点的 block throughput 中位数定义：

```text
R_ref = max(median(B_confirmed_candidate_throughput),
            median(C_confirmed_candidate_throughput))
```

B/C 使用完全相同的绝对流量：

```text
0.50, 0.70, 0.85, 1.00, 1.10, 1.20 * R_ref
```

采用 Poisson 到达，burstiness 1。CPU handoff 为每个 block 预生成 arrival seed、完整
inter-arrival 时间表和 SHA-256；同一 block 内 B/C 复用完全相同的 scheduled arrival，
而不只是使用相同分布。发现 pass/fail 边界后，再细化到约 5% 精度。

### 15.2 运行长度

普通容量曲线点使用 5 分钟 arrival window；若预生成 offered count 少于 128，则延长到
最多 10 分钟。10 分钟仍不足 128 个 offered request 时，该点只作诊断，标记
`underpowered`，不得用于 `R*` pass/fail。

关键质量点使用完整 512 个唯一问题的固定 cohort，Poisson 只控制这 512 个 arrival 的
时间间隔；全部 arrival 发出后允许最多 120 秒 drain。若该点的 arrival schedule 本身
超过 15 分钟，GPU 预算门停止并要求注册新的运行计划，不能缩小质量 cohort 后继续。

所有关键容量/质量点运行四个 ABBA 配对 block。warmup、arrival window 和 drain 分开
标记；请求 latency 从各自 scheduled arrival 计算，不能把 drain 时间混入 offered rate。

机器可判定的开环指标为：

```text
offered_rate = offered_count / arrival_window_seconds
SLO_attainment = completed_within_30s_of_own_arrival / offered_count
SLO_goodput = completed_within_30s_of_own_arrival / arrival_window_seconds
```

Queue depth 每秒采样。以 arrival window 的中间 60% 为稳态窗口，对 queue depth 计算
Theil-Sen slope 及 block bootstrap 95% 上界；要求该上界不超过
`max(0.01, 0.01 * offered_rate)` requests/s，最终 backlog 不超过 offered count 的 5%，
且在 120 秒内清空。吞吐比较只使用相同 scheduled-arrival 窗口。

普通曲线得到的最高 pass 只能称 `provisional R*`。Sweep 结束后，必须把每个模型的
最高 provisional pass 和紧邻更高 fail 点提升为关键容量点，补齐完整 512 题、四个
ABBA/GPU-swap block、自身并发质量漂移门和全部系统 SLO。只有全部确认后才能称
`R*_B` 或 `R*_C`；预算不足未确认时只能报告 provisional 区间。

每个 `R*_m` 只依赖该模型自己的 transport/SLO、queue stability 和相对并发 1 的质量
漂移。高负载直接 `C-B` 非劣是宣布“C 优于 B”的额外门，不反向决定 `R*_B` 是否有效。

### 15.3 主确认点

注册两个不同职责的主点：

```text
R_quality = 0.85 * R_ref
R_stress  = 1.00 * R_ref
```

`R_quality` 使用完整 512 题，要求 B/C 都通过 98% attainment 和其余共同 SLO，承担
高负载下 C−B 直接质量非劣检验；不要求在该点出现 5% goodput 差。

`R_stress` 同样使用完整 512 题，承担连续主容量估计。正向容量场景要求 C 通过共同
SLO，但允许 B 在该压力点因容量不足而失败；否则双方都要求 98% attainment 时，
goodput ratio 数学上不可能达到 1.05。定义：

```text
stress_goodput_ratio = SLO_goodput_C / SLO_goodput_B
```

只有 C 在 `R_stress` 仍 pass、ratio 点估计 `>=1.05` 且 paired block 95% CI 下界
`>1.0`，才支持 5% 以上的连续容量提升。B 在该点的 failure 是容量曲线的一部分，
但 failure 请求仍按 intent-to-serve 进入延迟、goodput 和质量分母。

`R*` 仍是所有正式重复都通过的最高离散已测档位，只报告点值和测试区间，不附加
没有定义的 95% CI。`R*_C/R*_B` 是辅助容量比；连续置信区间只用于上述
`stress_goodput_ratio` 及预注册支持指标。

## 16. 可选 Phase E：两卡双副本扩展

只有单副本主实验完成并且证据显示存在部署价值时，才考虑双副本扩展：

- GPU0/GPU1 各一个相同模型 replica；
- 轻量 router 使用固定轮询或最短队列策略；
- B 和 C 分别独占整个双卡窗口，不同时运行；
- 不使用 tensor parallel；
- 重复相同开环 workload；
- 额外记录 router queue、负载均衡和副本间 skew。

该阶段回答 scale-out 行为，不属于 B/C 主因果比较。没有用户另行授权时不执行。

## 17. 指标与逐请求证据 Schema

### 17.1 每个 Agent 请求

逐请求 JSONL 至少包含：

- schema version；
- experiment/run/block/request ID；
- model/checkpoint digest；
- sample ID、source、endpoint 和 workload order；
- offered、admitted、started、first-token、completed 时间；
- gateway queue、model、retrieval、parser、total latency；
- 每轮 prompt/completion/total token；
- 模型调用次数和 executed search 次数；
- query、返回文档 ID 和 retrieval latency；
- action type、invalid type、terminal reason；
- clipping、timeout、HTTP status、finish reason；
- final answer、gold answers、strict EM；
- retry/cancel/late-completion 标记；
- raw trace 或其内容 digest。

### 17.2 每个模型调用

至少保存：

- parent request ID 和 turn；
- prompt token count 与 prompt digest；
- TTFT、TPOT、ITL、E2E；
- generation config；
- finish reason；
- raw sampled token IDs/count、action-prefix IDs/count、raw-tail IDs/count；
- boundary 类型、overshoot 和 response digest；
- parsed action 与 parser diagnostics。

### 17.3 系统遥测

vLLM `/metrics` 与系统采样至少包括：

- running/waiting requests；
- queue、prefill、decode latency；
- KV cache usage；
- prompt/generation token totals；
- TTFT、ITL、E2E histograms；
- request success 和 finish reason；
- GPU utilization、memory、temperature、power；
- CPU、RAM、load、thread 和 file descriptor；
- BM25 QPS、latency、error 和 queue。

所有指标使用单调时钟计算 duration，并保留 wall-clock UTC 便于跨进程对齐。

## 18. 统计分析合同

### 18.1 质量、搜索和 token

- 题目级 paired bootstrap；
- 按 NQ、HotpotQA、2Wiki 分层抽样；
- 10,000 次；
- 固定 analysis seed；
- 使用 intent-to-serve 分母，timeout、429、5xx、取消和缺失终态均为 EM=0；
- completed-only EM 只作诊断，不进入任何非劣门；
- 报告点估计、绝对差、相对差和 95% CI；
- 512 题重复运行不扩大质量样本量；
- source slice 必须完整报告，不只展示总体有利切片。

### 18.2 延迟与吞吐

请求共享队列并非独立样本，不能把数千请求直接当作独立重复。分析使用：

- run/block 作为实验单元；
- 30 秒时间块 moving/block bootstrap；
- ABBA block 与 GPU swap 分层；
- 报告 C/B ratio、绝对差、95% CI 和每个 run 的离散度；
- 检查时间趋势、热状态和 block order interaction。

固定 token 模型层的工程等价以 `2048->500, concurrency=16` output-token-throughput
ratio 为唯一主指标，使用四个 paired block 的 90% 等价区间和 `[0.95,1.05]` 带。
只有区间完整落带才写“工程等价”；点估计落带但区间越界时写“证据不确定”。

### 18.3 多负载点

主确认性结论来自第 15.3 节机械确定的 `R_quality`、`R_stress` 和离散 `R*`。其余
并发/RPS 点用于曲线和机制解释。如果对多个点都计算 p 值，使用 Holm 校正；不能拿
某一个偶然有利点替代主点。

### 18.4 质量调整后的解释顺序

最终按以下顺序判断：

1. serving compatibility 是否通过；
2. 模型层是否工程等价；
3. BM25 是否先饱和；
4. 模型间质量非劣是否通过；
5. 并发质量漂移是否通过；
6. `R*`、`N*` 和 SLO-goodput 是否提升；
7. searches/token/model calls 是否解释容量变化。

只要第 4 或第 5 步失败，就不能跳到第 6 步宣布 C 更优。

## 19. 结果判读矩阵

| 观察结果 | 允许解释 |
| --- | --- |
| 模型层 B≈C，完整 Agent 中 C 的 RPS/goodput 提高，质量门通过 | 最强正向结果：行为节省转化为端到端容量 |
| 模型层 C 明显更快 | 先排查 serving 栈、GPU、格式和配置，暂不归因 RL |
| C 延迟下降但饱和 throughput 不变 | 单请求工作量下降，容量未确认提高 |
| 并发数增加但 tasks/s 不增加 | 只是允许更多排队，不是吞吐提升 |
| raw QPS 提高但 correct/clean goodput 不提高 | 速度可能来自错误、截断或质量下降 |
| BM25 先饱和 | 当前测到检索后端上限，不是模型容量 |
| C 只在 NQ/Hotpot 提高，2Wiki 退化 | 只能作为特定流量分布下的部署候选 |
| B/C 差异落入 ±5% 等价带 | 搜索节省没有转化为实质系统容量 |
| CI 较宽或 ABBA 方向相反 | 证据不确定，需要排查漂移或增加 run |

完整工程执行即使得到等价、负向或 BM25 瓶颈，也属于成功完成的实验；科学负结果
不是基础设施错误，不应自动重试到出现正向结果。

## 20. 停止、作废与预算规则

### 20.1 当前 run 立即停止

出现以下任一条件，立即停止当前 run 并保留证据：

- OOM、CUDA fatal、GPU reset 或服务进程重启；
- checkpoint、tokenizer、index、workload 或 config digest 不符；
- 前 100 个请求服务错误超过 5%；
- 至少已有 100 个 terminal/timeout 观测后，p99 仍超过 120 秒；
- 按第 15.2 节 queue-slope 规则连续两个 60 秒检查窗不通过，且最终 backlog 无法回落；
- GPU 热降频或存在未登记资源污染；
- request ID、逐请求结果或关键遥测缺失，无法对账；
- 磁盘不足以原子发布终态和结果；
- 全局 lock 丢失或持久盘身份变化。

### 20.2 停止继续升压

- 连续两个更高负载点未通过共同 SLO；
- compatibility gate 无法与 HF 语义对齐；
- BM25 已先饱和且未被公平隔离；
- 预算或运行时硬上限到达；
- 结果持久化、sync 或 evidence seal 不可靠。

达到预算上限表示实验未完成，不是 B 或 C 的科学负结果。禁止因为已经看到有利结果
而提前结束，也禁止因结果不利而临时增加更多有利 workload。

### 20.3 GPU 预算

预计双卡实例使用：

| 阶段 | 预计实例时间 |
| --- | ---: |
| compatibility gate | 0.5–1h |
| 固定 token 模型层 | 1–2h |
| BM25 与 gateway smoke | 0.5–1h |
| Agent 粗扫和细扫 | 2–4h |
| 512 题 ABBA 确认 | 3–5h |
| 开环主点与 seal | 1–2h |
| 合计 | 8–14h |

该估算已经按缩减后的模型层粗扫计算，但仍取决于新 serving 的实测 tokens/s。G0 结束后
必须用 64 条 smoke 的实测吞吐、正式矩阵 token 上限和历史 searches/task 生成剩余时间
P50/P90 预算。若 P90 超过剩余硬上限，在进入正式 sweep 前 NO-GO 并请求用户选择一个
新的、预注册的缩减矩阵；不能运行到一半再删掉不利档位。

执行硬上限：`16` 个双卡实例小时，或用户批准的金额上限，先到者为准。

费用计算必须使用实例当时实际价格：

```text
estimated_bill = AUTODL_PRICE_PER_HOUR * elapsed_instance_hours
```

历史脚本使用的 `5.76 元/小时` 只能用于规划；若价格仍相同，8–14 小时约 46–81 元，
16 小时硬上限约 92 元。启动前必须重新确认，不能把历史价格写成账单证明。

## 21. 磁盘与保留合同

B/C checkpoint 合计约 18GB。独立 serving 环境、cache、日志、逐请求 trace 和封存包预计
再需要 5–15GB。CPU admission 建议要求至少 30GB 可用空间，并额外保留若干 MiB 的
terminal reserve，防止磁盘写满时无法发布失败终态。

禁止自动删除：

- B20、C20、R60 checkpoint，以及 G3 trace、registered analysis 和 evidence；
- 原 B/C 和 A/R 结果 evidence；
- workload parquet、BM25 corpus/index 和 tokenizer；
- 本实验任何 failed attempt 的 terminal、exit code 和关键日志。

若空间不足，先生成“路径、大小、引用关系、digest、备份位置、是否可再生”的候选清单，
只清理明确未被采用的 partial、重复 cache 或可重新下载对象。不得在 GPU runner 内做
隐式清理。

在释放旧实例前，必须确认 B/C checkpoint 已备份或仍位于将挂载到新实例的同一持久盘；
本地 Git evidence 包不包含两个约 9GB 的模型权重。

## 22. 计划代码改动

### 22.1 新增文件

推荐实现范围：

```text
search_r1/serving/__init__.py
search_r1/serving/qwen35_agent_api.py
search_r1/serving/qwen35_agent_backend.py
scripts/benchmark/build_bc_workload.py
scripts/benchmark/benchmark_retriever.py
scripts/benchmark/run_bc_concurrency.py
scripts/benchmark/analyze_bc_concurrency.py
scripts/benchmark/requirements-serving.lock 或固定容器 manifest
scripts/autodl/bootstrap_qwen_native_bc_concurrency.sh
scripts/autodl/16_gpu_qwen_native_bc_concurrency.sh
scripts/autodl/17_watch_qwen_native_bc_concurrency.sh
tests/test_qwen35_agent_api.py
tests/test_bc_workload.py
tests/test_retriever_concurrency.py
tests/test_bc_concurrency_analysis.py
docs/results/qwen35-native-bc-concurrency-<date>/
```

实际文件名允许在实现前小幅调整，但 contract、runner、watchdog、分析器和结果 namespace
必须保持职责分离。

### 22.2 预计修改文件

- `scripts/autodl/README.md`：增加新实验 CPU/GPU 正常操作入口；
- `scripts/autodl/cloud-audit-policy.json`：登记新 runner、watchdog、文档字面量和 test hook；
- `.gitignore`：仅在新本地临时结果需要时增加精确规则；
- `search_r1/search/bm25_server.py`：只有 BM25 基准证明存在瓶颈或竞态时才另阶段修改。

### 22.3 明确不修改

- `verl/` trainer 与 reward；
- B/C checkpoint；
- 已封存的 `12_*`、`13_*`、`14_*`、`15_*` 行为身份；
- `generation.py` 的历史训练/评测语义；
- 根训练依赖锁；
- 旧 evidence 文件；
- `infer.py`，它保留为原项目旧 demo，不承担正式 serving。

### 22.4 复用边界

直接复用：

- `search_r1/llm_agent/tool_protocol.py` 的 native prompt/parser/conversation；
- `search_r1/search/bm25_server.py` 的检索协议与索引；
- `scripts/autodl/paired_eval.py` 的配对身份和 bootstrap 范式；
- `13_gpu_qwen_native_bc_recovery.sh` 的 checkpoint/前驱校验范式；
- `14_gpu_qwen_native_ar_eval_only.sh` 的独立 CPU receipt 与 evidence seal 范式；
- `15_watch_qwen_native_ar_eval_only.sh` 的 marker-last 和 watchdog 范式。

不把 `LLMGenerationManager` 整体直接放进 FastAPI event loop：它绑定 actor rollout，且
当前 BM25 client 使用阻塞 `requests.post`。在线 gateway 只复用纯协议组件和经过测试的
状态语义。

## 23. Git、CPU 与 GPU 阶段

### 23.1 Git 身份

当前分支只冻结方案。实现完成后必须：

1. 本地测试通过；
2. 创建聚焦 commit；
3. 推送远端；
4. 从可信远端解析一次完整 40 位 implementation commit；
5. CPU 与 GPU 全链只使用该 commit 的 clean detached checkout；
6. checkout 运行期间禁止 pull、switch、自动编辑 config 或使用 branch tip 漂移。

当前 `c0d4cd3...` 只是本方案分支的 main 基线，不是未来执行 commit。

新实验不得覆盖或切换旧 canonical
`/root/autodl-tmp/search-r1/checkout`，它已经绑定历史实验 commit。新 checkout 固定为：

```text
/root/autodl-tmp/search-r1/serving-checkouts/<40-char-implementation-commit>
```

CPU bootstrap 必须先把脚本完整下载到临时文件再执行，解析可信远端的完整 commit，
clone/fetch 到 staging 目录，materialize detached checkout，验证 origin、HEAD、cleanliness
和无 Git lock 后再原子发布 canonical 路径。不得 `curl | bash`、不得把 branch tip 当执行
身份，也不得自动 reset 旧 checkout。

### 23.2 CPU 联网准备

CPU 阶段完成所有网络、安装和下载工作：

- 验证同一持久卷与 canonical path；
- 校验 B/C checkpoint、tokenizer、数据、corpus 和 index digest；
- 创建并固定 serving 环境；
- 固定 vLLM wheel/image 与全部依赖；
- 构建 canonical 512/128 workload 与 order manifests；
- 生成 fixed-token benchmark 数据；
- 从历史 trace 构建 BM25 query workload；
- 生成 HF golden prompt/token/action oracle；
- 运行 Python、shell、config、schema 和 static audit；
- 统计磁盘与原子发布峰值；
- 在资源足够时发布自哈希、原子写入的 complete concurrency CPU handoff。

新实验 handoff 必须使用独立 schema/namespace，不能覆盖旧 native-v4 CPU handoff。
可以复用已有不可变资产，但必须重新验证并在新 handoff 中引用其 digest。

CPU 状态只允许二选一：

1. CPU/RAM 达到记录下限时，CPU 直接发布 `cpu-handoff-complete`；GPU 只验证；
2. CPU/RAM 不足时，只发布不同 schema 的 preliminary `cpu-prepare-seal`。GPU 命令先在
   offline、CUDA-hidden 环境执行 finalize，成功后发布 `cpu-handoff-complete`。

两个 marker 不能共用名称或 schema。GPU admission 只接受 complete handoff，绝不把
preliminary seal 当成可加载模型的授权。

### 23.3 GPU admission

若只存在 preliminary seal，GPU 命令首先执行 offline、CUDA-hidden CPU finalize；若
CPU 已发布 complete handoff，则跳过 finalize，只做重验。在 complete handoff 发布并
通过前不得导入 CUDA 或加载权重。GPU admission 随后验证：

- commit 和 clean detached checkout；
- 同一持久盘与无并发 host writer；
- CPU handoff 自哈希和全部输入身份；
- serving 环境完整且不能联网；
- GPU/driver/CUDA 符合合同；
- 磁盘、terminal reserve 和预算；
- 项目级全局 lock `/root/autodl-tmp/search-r1/state/phase.lock`；
- B/C checkpoint 可读且 digest 正确。

缺失包、模型、tokenizer 或数据时 fail closed，不在付费 GPU 阶段下载或安装。

### 23.4 计划操作接口

后续实现应保持一个完整 CPU bootstrap block 和一个 GPU 命令。CPU 正常入口必须先把
bootstrap 完整下载到临时文件、验证 SHA-256，再执行；不能 `curl | bash`。计划接口为：

```bash
IMPLEMENTATION_COMMIT=<40-char-implementation-commit>
REPO_URL=<trusted-git-origin>
BOOTSTRAP_URL=<raw-url-bound-to-that-commit>
BOOTSTRAP_SHA256=<published-bootstrap-sha256>
BOOTSTRAP_TMP=/tmp/search-r1-serving-bootstrap.sh

curl --fail --location --retry 3 --connect-timeout 20 \
  --output "$BOOTSTRAP_TMP.part" "$BOOTSTRAP_URL"
mv "$BOOTSTRAP_TMP.part" "$BOOTSTRAP_TMP"
echo "$BOOTSTRAP_SHA256  $BOOTSTRAP_TMP" | sha256sum --check --strict
bash "$BOOTSTRAP_TMP" \
  --repo "$REPO_URL" \
  --commit "$IMPLEMENTATION_COMMIT" \
  --checkout-root /root/autodl-tmp/search-r1/serving-checkouts \
  --start-cpu-prepare
```

挂载同一持久卷并确认 CPU 实例已经停止后，GPU 只运行：

```bash
IMPLEMENTATION_COMMIT=<same-40-char-implementation-commit>
SERVING_CHECKOUT="/root/autodl-tmp/search-r1/serving-checkouts/$IMPLEMENTATION_COMMIT"
GPU_COUNT=2 AUTODL_PRICE_PER_HOUR=<actual-price> \
bash "$SERVING_CHECKOUT/scripts/autodl/16_gpu_qwen_native_bc_concurrency.sh"
```

入口打印 exact attempt 后，再显式绑定 watchdog：

```bash
IMPLEMENTATION_COMMIT=<same-40-char-implementation-commit>
SERVING_CHECKOUT="/root/autodl-tmp/search-r1/serving-checkouts/$IMPLEMENTATION_COMMIT"
bash "$SERVING_CHECKOUT/scripts/autodl/17_watch_qwen_native_bc_concurrency.sh" \
  /root/autodl-tmp/search-r1/runs/serving-concurrency/attempts/gpu/<exact-attempt>
```

这些命令只是计划接口，当前文件创建时脚本尚不存在，不能立即执行。

## 24. Attempt、Evidence 与结果 Seal

### 24.1 持久状态布局

建议新 namespace：

```text
/root/autodl-tmp/search-r1/runs/serving-concurrency/
  launchers/<launcher-id>/
  attempts/cpu/<attempt-id>/
  attempts/gpu/<attempt-id>/
  stages/<stage>/<attempt-id>/
  results/<result-id>/
  manifests/
  locks/  # 仅本实验子资源锁
```

Checkout 保持只读；mutable env、cache、logs、attempts 和 results 位于持久盘 checkout
之外。整个 worker 生命周期还必须持有项目级
`/root/autodl-tmp/search-r1/state/phase.lock`；本 namespace 的子锁不能替代它，否则旧
训练/评测 runner 可能与新 serving runner 同时写同一持久盘。

### 24.2 阶段

GPU pipeline 至少拆分为：

```text
gpu-admission
serving-compatibility
model-fixed-token-benchmark
retriever-benchmark
agent-concurrency-one-baseline
agent-closed-loop-coarse
agent-closed-loop-refine
agent-capacity-confirmation
agent-open-loop-confirmation
analysis-and-seal
```

每个阶段创建唯一、不可覆盖的 attempt，并验证前驱 identity。科学 NO-GO 也必须产生
完整结果和终态，不能被 pipeline 当作基础设施异常自动重试。

### 24.3 Attempt 终态

每个 inner stage attempt 至少保存：

- request/contract；
- started/finished 时间；
- `.running`；
- 原始 stdout/stderr log；
- 原始 exit code；
- terminal 与 terminal JSON；
- 恰好一个 `.success` 或 `.failed`；
- artifacts manifest；
- predecessor identities；
- result root 与 digest。

只有 outer launcher/watchdog 额外拥有：

- launcher request、PID 和 terminal sentinel；
- pipeline result 与原始 exit code；
- shutdown-safe；
- 恰好一个 shutdown-skipped / requested / failed 结果；
- backend dispatch 记录和 provider-confirmation 状态。

Inner stage 不发布关机 marker，也不争夺 shutdown 所有权。

顺序必须为：验证产物，写终态，durable sync，再删除 `.running`。不能先发布 success
再验证结果，也不能让 `tee` 的退出码覆盖真实命令退出码。

### 24.4 最终 evidence

最终结果目录至少包含：

```text
experiment-contract.json
lineage.tsv
implementation-commit
checkpoint-manifest.json
serving-environment.txt
serving-configs/
workload-manifest.json
workload-orders/
arrival-traces/
golden-parity/
model-benchmark/raw/
retriever-benchmark/raw/
agent-benchmark/requests/
agent-benchmark/traces/
metrics/vllm/
metrics/system/
metrics/retriever/
run-index.tsv
summary.json
summary.md
figures/
evidence.sha256
evidence-marker.ok
```

`evidence-marker.ok` 必须最后原子发布，并绑定 `evidence.sha256` 自身以外的完整清单。
结果 archive 还需保存 SHA-256，下载到本地后重新验证。

## 25. Retry、恢复与失败分类

### 25.1 不自动重试

以下情况不得自动重试：

- OOM；
- NaN/CUDA fatal；
- schema、checkpoint、tokenizer、index 或 hash mismatch；
- compatibility NO-GO；
- 质量门失败；
- BM25 科学容量不足；
- 已完成的负向或等价结果。

下载失败只允许在 CPU 阶段有界重试，并且最终文件必须通过 digest。

### 25.2 显式 retry

用户检查原因后，使用显式 retry flag 创建新 attempt，从第一个没有有效 success 的阶段
继续。复用每一个成功前驱前都重新验证 commit、config、环境、模型、数据和 artifacts。

失败 attempt 永不改写。只有阶段主产物已经完整、原子发布，而后续 terminal/evidence
封存失败时，才允许独立 verifier 创建一个指向原 attempt 的 synthetic adoption；
compatibility、partial benchmark 和科学失败不能 adoption。

### 25.3 基础设施失败与科学结果

- 进程 crash、网络遗漏、磁盘写失败、lock/identity 错误：基础设施失败；
- compatibility 语义漂移：接口 NO-GO；
- BM25 先饱和：系统瓶颈结果；
- B/C 工程等价、C 更慢或质量不非劣：完整科学/工程结果；
- 到达预算上限且关键阶段未完成：不完整实验，不能给模型结论。

## 26. Watchdog 与安全关机

GPU runner 本身不直接无条件关机。新 watchdog 只接受本实验 exact attempt、exact runner、
contract 和 marker-last evidence。

成功或失败后的关机顺序固定为：

```text
stage/pipeline terminal + original exit code
  -> launcher terminal sentinel
  -> durable sync
  -> shutdown-safe
  -> 重新验证 host/mount/path/commit/capability/lock/state
  -> durable sync
  -> shutdown-requested
  -> 调用已授权的 AutoDL guest shutdown backend
```

以下情况保持实例运行并写明原因：

- lock conflict；
- terminal、日志或 evidence 不完整；
- sync 失败或磁盘不足；
- mount、path、capability、commit、cleanliness 或 backend 身份失配；
- `--keep-running`、`--dry-run` 或 test mode；
- watchdog 不认识本实验 terminal contract。

Watchdog 同时识别两类 marker-last 合同：durable success terminal + success evidence seal，
以及 durable recognized failure/NO-GO terminal + failure evidence seal。后一类保留原始
非零 exit code，也允许在证据完整后请求关机；只有无法完整封存和验证的失败才保持
实例运行。

在已验证 AutoDL host 上，计划沿用 provider 文档语义，以无参数 `/usr/bin/shutdown`
请求 guest shutdown。原工作 exit code 与 shutdown dispatch 结果分别保存；backend 失败
不能覆盖实验 exit code。

Guest dispatch 返回 0 也不等于 AutoDL 控制面已经停止，更不等于计费结束。用户仍须
在控制台确认 exact instance 已关机并停止计费。

## 27. 测试矩阵

### 27.1 Python 与协议

- workload 构建数量、来源、顺序和 SHA；
- duplicate/missing/extra sample 拒绝；
- tokenizer、prompt token ID 和 template parity；
- action/search/observation/terminal 状态机；
- max-turn 和 terminal-search 拒绝；
- invalid、clipping、timeout 分类；
- gateway bounded queue、backpressure、cancel；
- no-retry 正式模式；
- BM25 response identity under concurrency；
- analyzer 的 EM、goodput、CI 和 source stratification；
- 512 重复不扩大质量样本量；
- 固定 golden fixture 的 Markdown/JSON 输出。

### 27.2 Serving 集成

- fake OpenAI-compatible server；
- streamed/non-streamed response；
- empty/invalid/slow/timeout/429/5xx；
- model process restart 与 health/ready；
- request ID 全链路对账；
- metrics scrape 缺失时 fail closed；
- B/C config 除模型路径与标签外 exact diff；
- fixed-token client 与 Agent client 隔离。

### 27.3 Shell 与云端状态机

- 所有 shell 通过 `bash -n`；
- success、nonzero、INT、TERM、早期 worker failure；
- `tee` 保留原始 exit code；
- lock conflict 返回稳定码且不关机；
- keep-running、dry-run、test mode 不关机；
- mount、commit、capability、symlink/path escape 拒绝；
- CPU handoff tamper、extra/missing field 拒绝；
- GPU offline，缺失包时不联网；
- immutable attempt 和显式 retry；
- terminal -> sync -> shutdown-safe -> shutdown-requested 事件顺序；
- 测试 hook 只写临时目录，绝不调用真实 shutdown；
- provider dispatch failure 不覆盖实验 exit code。

### 27.4 最终本地验证

实施阶段至少运行：

```text
python -m pytest -q <focused tests>
python -m compileall search_r1/serving scripts/benchmark
bash -n -- <each new shell>
python <skill>/scripts/audit_cloud_bundle.py --repo . --policy scripts/autodl/cloud-audit-policy.json
git diff --check
git status --short
```

Windows 本地若没有 Bash，必须用 WSL 或 Git Bash 实际执行 `bash -n`，不能因为 PowerShell
无法解析 shell 而把所有脚本误判失败。

## 28. 实施切片与规模

### 28.1 推荐实施顺序

1. 冻结本方案和机器可读 contract schema；
2. 实现 workload builder、manifest 和 analyzer golden tests；
3. 实现异步 gateway 与 fake-model 集成测试；
4. 实现 BM25 benchmark 和并发一致性测试；
5. 固定独立 serving 环境与 compatibility verifier；
6. 实现 AutoDL runner 的 CPU vertical slice；
7. 实现 GPU admission 和 compatibility gate；
8. 实现 fixed-token/BM25/Agent stage；
9. 实现 immutable results、lineage 和 evidence seal；
10. 实现专用 watchdog 和负面测试；
11. 完整本地验证、commit、push 和远端 bootstrap 验证；
12. 用户批准后再启动 CPU/GPU 实验。

### 28.2 改动规模

| 版本 | 工程量 | 文件/代码量 | GPU/实例时间 | 结论能力 |
| --- | ---: | ---: | ---: | --- |
| 最小 HF 批处理筛查 | 1–2 人日 | 4–6 文件，约 400–800 行 | 4–6h | 只能称批处理吞吐筛查 |
| 本文推荐在线版 | 3–5 人日 | 10–15 新增、2–4 修改，约 1500–3000 行 | 8–14h | 可形成可信 Agent 后端并发实验 |
| 扩展生产版 | 6–9 人日 | 15–25 文件，约 3000–5000 行 | 18–30h | 增加 soak、故障注入和双副本扩缩容 |

本文只批准“推荐在线版”的方案设计，不自动批准实现、GPU 运行或扩展生产版。

## 29. 最终交付物

工程完成必须交付：

- 固定版本的 Agent serving gateway；
- 可复现 workload 和 load generator；
- 模型层、BM25、Agent 层三层原始结果；
- 新 serving 与原 HF 协议 parity 证据；
- 逐请求 JSONL、系统 metrics、run index 和完整 lineage；
- 容量/延迟/goodput/质量图表；
- B/C 主结果和 source slice；
- failed attempt 与 retry 记录；
- evidence SHA-256、archive digest 和本地紧凑结果包；
- 中文完整分析报告；
- README 中的 CPU/GPU 正常操作、状态检查和恢复命令；
- guest shutdown dispatch 与控制台人工确认记录。

推荐图表：

1. 固定 token throughput/TTFT/TPOT vs concurrency；
2. BM25 QPS 和 p95/p99 vs concurrency；
3. Agent tasks/s、SLO-goodput、correct-goodput vs concurrency/RPS；
4. p50/p95/p99 E2E 和 queue latency；
5. GPU、KV cache、BM25 latency 分解；
6. searches/model calls/tokens per task；
7. strict EM、invalid、clipping、clean-correct 按来源切片；
8. ABBA block 与 GPU swap 的重复离散度。

## 30. 对外表述合同

### 30.1 正向结果模板

只有全部质量门通过后才可写：

> 构建了异步 Search-R1 Agent 服务与可复现并发压测流水线。在固定硬件、共同延迟
> SLO 和 strict-EM 非劣约束下，成本感知 C20 相对 B20 将检索调用降低 X%，
> SLO-goodput 提升 Y%，p95 端到端延迟降低 Z%，最大可持续开环负载由 R1 提升至 R2。

### 30.2 等价或负向结果模板

> 通过模型、检索和 Agent 三层压测确认 B/C 固定 token serving 性能等价；C 的搜索
> 节省没有稳定转化为系统容量提升。分层遥测将瓶颈定位于 GPU 解码、KV cache、BM25
> 或队列调度，并保留了质量感知 goodput 与失败证据。

### 30.3 流量限定模板

> C 在 NQ/Hotpot 流量下提高了有效吞吐，但 2Wiki 切片存在质量回退，因此仅作为该
> 流量组成下的成本优化候选，不作为通用能力基线。

### 30.4 禁止表述

不得写：

- “C 改变了模型显存容量，所以天然支持更多并发”；
- “不 OOM 就是可持续并发”；
- “同时挂 64 个请求，所以吞吐是 64”；
- “离线耗时少 2.62%，所以并发提升 2.62%”；
- “C 输出更短，所以推理一定更简洁”；
- “QPS 更高，所以部署价值一定更高”；
- “BM25 或客户端先触顶就是模型容量”；
- “一次 GPU0/GPU1 对照证明 C 更快”；
- “四次 512 题等于 2048 个独立质量样本”；
- “总体胜出代表所有来源都胜出”；
- “固定 checkpoint 对比证明 cost-aware RL 普遍提高并发”；
- “C 比 B 并发高，所以 C 比 R 或所有模型更好”；
- “单机 AutoDL 实验是生产级集群容量证明”。

## 31. 最终成功标准

### 31.1 工程实验完成

无论结果正负，以下全部满足才算实验完成：

- B/C、commit、serving 环境、tokenizer、数据、BM25 和 workload 身份可复验；
- compatibility、模型层、BM25、Agent 闭环和开环阶段均有明确终态；
- 原始逐请求、系统遥测、退出码、失败和 retry 未被覆盖；
- 所有主指标、source slice 和质量风险完整报告；
- evidence seal 与 archive digest 通过；
- 自动关机只在 durable terminal 后请求；
- AutoDL 控制台停止计费由人工确认。

### 31.2 C 并发提升成立

必须同时满足：

1. compatibility gate 通过；
2. 固定 token 模型层没有未解释的 >5% 混杂；
3. BM25 不是未隔离的先行瓶颈；
4. C−B strict EM 通过 `-3pp` 非劣门；
5. 高负载相对并发 1 的质量漂移门通过；
6. B/C 在 `R_quality` 都 pass，C 在 `R_stress` 仍 pass；
7. `stress_goodput_ratio>=1.05`、其 paired block 95% CI 下界 `>1.0`，且离散
   `R*_C/R*_B>=1.05`；
8. ABBA 与 GPU swap 方向一致；
9. 2Wiki 风险被单独披露并限定适用流量。

不满足这些条件时，应报告等价、不确定、瓶颈或 cost-quality trade-off，不能通过换 SLO、
删样本、挑并发点或忽略 clipping 强行生成正向结论。

## 32. 关联资料

仓库内证据：

- [`qwen35_native_complete_experiment_handoff.md`](history/qwen35-native-arbc-202607-202608/final/qwen35_native_complete_experiment_handoff.md)
- [`qwen35_native_arbc_final_results_analysis.md`](history/qwen35-native-arbc-202607-202608/final/qwen35_native_arbc_final_results_analysis.md)
- [`qwen35_native_bc_recovery_complete_analysis_report.md`](history/qwen35-native-arbc-202607-202608/final/qwen35_native_bc_recovery_complete_analysis_report.md)
- [`results/qwen35-native-bc-recovery-20260801/`](results/qwen35-native-bc-recovery-20260801/)
- [`qwen35_native_sft_rl_bc_followup_plan.md`](history/qwen35-native-arbc-202607-202608/plans/qwen35_native_sft_rl_bc_followup_plan.md)
- [`../scripts/autodl/README.md`](../scripts/autodl/README.md)

外部官方资料：

- [Qwen3.5-2B model card](https://huggingface.co/Qwen/Qwen3.5-2B)
- [vLLM supported models](https://docs.vllm.ai/en/stable/models/supported_models/)
- [vLLM model resolution](https://docs.vllm.ai/en/latest/configuration/model_resolution/)
- [vLLM Qwen3.5 recipe](https://github.com/vllm-project/recipes/blob/main/Qwen/Qwen3.5.md)
- [vLLM bench serve](https://docs.vllm.ai/en/stable/cli/bench/serve/)
- [vLLM engine arguments](https://docs.vllm.ai/en/stable/configuration/engine_args/)
- [vLLM OpenAI-compatible server](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/)
- [vLLM production metrics](https://docs.vllm.ai/en/stable/usage/metrics/)
- [vLLM nightly builds](https://docs.vllm.ai/en/stable/contributing/ci/nightly_builds/)
