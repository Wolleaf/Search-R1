# Qwen3.5 Native G2 问题清单与训练解阻计划

> 记录日期：2026-07-24
>
> 审计基线：`experiment/hotpot-search-gate@b0563c287b3b8a1203879276bf05af4e7d06fc5b`
>
> 当前结论：方向仍然值得继续，但当前提交不能直接启动 native GRPO 训练。

## 1. 结论

本轮不是 OOM、CUDA 或检索服务故障。G0/G1/G2 均完成了工程执行，Qwen3.5-2B 已能稳定使用原生 `search` 工具，并且 G2 的 96 条轨迹中有 51 条实际执行了至少两次搜索。当前阻塞已经收敛到三个局部接口：最终短答案合同、少量 parser 边界，以及训练采样与 PPO log-prob 的一致性。

因此，“模型不会搜索”这一假设已基本排除；但“已有可直接用于 strict-EM GRPO 的自动奖励信号”尚未成立。现在没有 native 训练 checkpoint，下一步是解阻后启动新训练，而不是续跑一个已开始的训练。

## 2. 已完成证据

| 阶段 | 工程结果 | 科学结果 |
| --- | --- | --- |
| G0+G1 | 外层成功；G1 保存 32/32 条轨迹，32/32 首动作合法且非退化，45/45 tool response 对齐 | 原生模板、tool call、tool response 和多轮回填可用 |
| G2 | 外层与 eval 均为 `exit-code=0`、`.success`，96/96 轨迹和哈希完整；无 OOM、CUDA、NCCL 或超时 | 正式 `NO-GO`；唯一失败门槛是 invalid trajectory `7 > 5`，clipped `1 <= 5`、degenerate query `0 <= 3` 均通过 |

G2 实际执行搜索分布为：

```text
n_search=0/1/2/3/4: 5 / 40 / 36 / 13 / 2
总实际检索请求: 159
至少二搜: 51/96
```

结果中的 `sum(executed_search_count)=159`，对应真正送入检索器并产生 retrieval event 的请求；分析器的 `search_turn_count=161` 还包含两条达到四次执行上限后，在 terminal generation 再次发出但没有执行的第五次 search action。成本与实际检索行为分析必须使用前者。

对最终答案进行独立离线审计后，96 条轨迹可分为：

| 人工审计类别 | 数量 | 含义 |
| --- | ---: | --- |
| 内容明确正确 | 34 | 多数因完整句子、别名、日期顺序或词形差异被 strict EM 判 0 |
| 有证据但部分/接近正确 | 6 | 不能直接作为正确奖励 |
| 内容错误 | 53 | 检索或答案推理确实失败 |
| 无有效终局答案 | 3 | 没有可评分答案 |

按人工明确正确标签，13/32 个 group 同时含正确和错误轨迹，说明潜在 GRPO 组内差异已经存在。对全部 96 条轨迹进行、不按 valid 过滤的 `subem_check` 机械复算会标记 26/96 条并产生 15/32 个 mixed group，但其中包含 1 条明确矛盾和 3 条部分/近似答案，不能冒充正式准确率。当前人工分类尚未物化成仓库内逐轨迹 ledger；在 CPU 归档生成 rubric、标签、source trace digest 和 SHA-256 前，这组数字只视为审计记录，不能作为可复算实验产物。

上述数字只用于诊断潜力，不改变 G2 的正式 `NO-GO`，也不能替代新的 GPU 门禁。

## 3. 当前问题清单

### P0-1：自由文本终局与 strict EM 的合同不匹配

[`tool_protocol.py`](../search_r1/llm_agent/tool_protocol.py) 当前把所有不含协议 marker 的非空回复整体作为 `final_answer`；[`main_ppo.py`](../verl/trainer/main_ppo.py) 再用 `qa_em.em_check` 对整段文本做规范化后的全串相等比较。

例如 gold 为 `navy blue and gold`，模型回答 `The official colours ... are navy blue and gold.`，内容正确但正式 EM 为 0。类似问题还包括：

- `2,526` 或 `1980` 被包在解释句中；
- `Piers Sellers` 对 `Piers John Sellers`；
- `August 21, 1765` 对 `21 August 1765`；
- Markdown 加粗、句号和额外说明。

这不是 EM 实现随机出错，而是 native 自由文本接口没有像原 Search-R1 的 `<answer>...</answer>` 一样提供明确短答案边界。若强行训练，当前 96 条轨迹的正确性奖励和 GRPO 组内 advantage 都为 0；correctness-gated 成本奖励此时也仍为 0。只有历史 ungated linear cost 才会额外产生错误的少搜压力。

固定短答案边界主要解决完整句、Markdown 和额外解释，不能自动解决 `Piers Sellers` 对 `Piers John Sellers` 或日期顺序等语义等价问题；这些情况仍按 strict EM 记 0，并在诊断报告中单独展示。

### P0-2：parser 把部分合法答案误判为工具格式错误

当前 `_PROTOCOL_MARKERS` 包含 `<think>` 与 `</think>`。只要回复出现这些 marker，就会进入“必须是一个完整 tool call”的分支。因此至少两条带 canonical empty-think 前缀、随后给出正常答案的轨迹被记为 `multiple_or_unbalanced_tool_calls`。

G2 的 7 条 invalid trajectory 共包含 10 个 invalid event：8 个 multiple/unbalanced 类事件和 2 个空 query。只有 thinking 前缀误判应修复；双 tool call、残缺闭合、空 query 和未知函数仍应保持 invalid，不能为了过门禁全面放宽 parser。

### P0-3：native 训练采样与 PPO log-prob 不一致

G0-G2 使用：

```text
temperature=1.0, top_p=1.0, top_k=20, min_p=0.0,
presence_penalty=2.0, repetition_penalty=1.0
```

HF rollout 从经过 top-k 和 presence penalty 处理的 proposal 分布采样，但 actor old/current log-prob 仍按 temperature 后的完整词表分布重算。两者不能直接构成严格一致的 PPO ratio。为防止误训，[`ray_trainer.py`](../verl/trainer/ppo/ray_trainer.py) 已显式拒绝 `qwen35_native && !trainer.val_only`。

直接删除该 guard 不可接受。按最小实现原则，不实现 processed-proposal log-prob 全链路，而是在训练及其 readiness gate 中关闭 top-k/presence penalty。

### P1-1：旧 G2 不能被事后重判为新前驱

G2 的正式 `NO-GO` 只由 invalid `7 > 5` 触发，EM 本来就是报告项。离线修 parser 或换答案指标可以解释原因，但不能把既有证据改写成 `GO`。prompt、parser、reward 或采样合同变化后，必须提升协议/实验版本并生成新的 exact attempt。

### P1-2：潜在正确轨迹尚未转化为自动、可防投机的奖励

人工审计得到 34 条明确正确和 13 个 mixed group，证明方向有希望；但人工标签不能直接进入在线训练。简单改用 substring reward 虽能恢复部分信号，却会奖励矛盾句、部分答案，且存在让模型堆砌候选答案的 reward hacking 风险。

本轮默认方案应先建立确定性的短答案边界，再继续使用 strict EM。`subem_check` 只作为诊断指标；若短答案方案仍无法产生足够信号，substring bootstrap 必须作为新的、单独预注册实验，而不是本轮隐式兜底。

### P1-3：G2 只证明协议健康，不证明能力门已通过

G2 的注册目标是动作格式、query 和闭环健康，因此 `EM=0/96` 不参与其正式判定。真正检查正确多搜轨迹、覆盖题数和 learnable group 的 G3 尚未运行。即使修复 G2，也必须通过新的 G3 才能启动长训练。

### P2：证据归档与磁盘余量

当前 G0-G2 的 immutable evidence 仍只在 AutoDL 持久盘，尚未归档到本仓库 `docs/results/`。数据盘为 100 GB，最近检查约使用 75 GB、剩余 26 GB：足够进行 CPU 增量重封和门禁，但尚不能证明足够同时保存 R/B/C 的全参数 optimizer checkpoint。2-step smoke 后必须先记录实际 checkpoint 大小并注册顺序保留方案；R 至少保留到 B/C 均完成，先完成的分支应在校验 inference 权重、指标和 lineage 后再考虑释放 optimizer 状态。任何清理都需另行确认；若测得余量不足，则在长训前停止，不自动扩容或删除历史证据。

## 4. 最小修复方案

### 4.1 建立 Qwen native v2 最终答案合同

搜索仍使用 Qwen3.5 原生 `search` tool call，不增加 `finish` 工具，不改 Agent 搜索状态机。只把普通 assistant 终局固定为一行：

```text
FINAL_ANSWER: <short answer>
```

对应改动仅包括：

1. 将 prompt version 提升为 v2，并明确要求唯一一行短答案，不增加 few-shot 或复杂模板。
2. parser 先剥离至多一个完整、平衡且内容只含空白的 leading `<think></think>`；非空、重复、嵌套或不平衡 thinking 仍拒绝。
3. 剩余内容只能是一个完整 tool call，或唯一 `FINAL_ANSWER:` 行；只把冒号后的短文本送入 strict EM。
4. 原始回复、thinking prefix、抽取答案和 parse error 全部继续写入 trace。
5. v1 parser 与旧 evidence 保持可读，不修改历史结果。

### 4.2 保留 strict EM，补齐诊断指标

训练正确性 reward 默认仍为抽取后 strict EM。每条 trace 同时保存：

- `raw_final_response`；
- `extracted_answer`；
- strict EM；
- 仅报告用的 normalized containment/subEM；
- 代表性 alias、日期顺序和矛盾答案轨迹；
- 实际搜索次数、post-hoc utility 和完整工具轨迹。

分析器必须从 `extracted_answer + gold_answers` 独立复算 strict EM，并核对 trace、Parquet 和 catalog 的 gold 一致性。这样既避免再次把格式问题误判为能力问题，也不通过宽松 reward 美化正式结果。

### 4.3 注册最小 on-policy 训练采样合同

所有新的 readiness gate、2-step smoke 和 native 训练统一使用：

```text
temperature=1.0, top_p=1.0, top_k=0, min_p=0.0,
presence_penalty=0.0, repetition_penalty=1.0
```

`top_k=0` 是当前 HF rollout 的禁用值。核心 guard 不直接删除，而是仅在上述精确配置、native v2、完整策略 mask 和训练 variant 均匹配时允许训练；任何 top-k、min-p、presence/repetition penalty 漂移继续 fail closed。

这会牺牲上一轮为提高生成多样性加入的 top-k/presence 参数，但避免实现大范围 processed-log-prob 张量链路，符合本项目的最小复现原则。由于采样分布改变，旧 G2 的良好工具行为不能直接外推，必须用相同训练分布重跑低成本门禁。

### 4.4 只做增量 CPU 重封

新 prompt version、parser 和训练配置会使旧 handoff 失效，但不需要重新下载或构建全部资产。CPU 无卡阶段只执行：

1. 将现有 G0-G2 evidence 归档并校验哈希；同时把人工答案审计物化为逐轨迹 ledger，绑定 rubric、source trace digest 和 SHA-256。两者都不改变正式结论或门禁。
2. 用相同 760 条样本、sample ID、顺序和配额重新物化 native v2 prompt。
3. 运行 parser/reward golden tests、完整 pytest、shell 语法和 Hydra 配置展开。
4. 重新校验模型、语料、BM25、数据 manifest 和依赖 freeze。
5. 发布绑定新 commit 与 prompt version 的 `cpu_handoff.json` 和 `cpu.ok`。

不重新下载模型，不重建 Wiki/BM25，不重新筛选 NQ/Hotpot，也不改变数据配比。

## 5. 验证与训练顺序

```text
本地实现与测试
  -> CPU 增量重封
  -> native-v2 G0/G1
  -> native-v2 G2 readiness
  -> native-v2 G3 capability gate
  -> 2-step training smoke
  -> R-mix60-native-v2
  -> R 后能力与 cost-contrast gate
  -> B/C-native-v2 对称分叉
```

### 5.1 G0/G1 与 G2 readiness

新 G0/G1/G2 必须使用和训练完全相同的 `top_k=0/presence_penalty=0` 采样。G2 保留原有门槛：

- invalid trajectory `<=5/96`；
- clipped trajectory `<=5/96`；
- degenerate search 不超过全部 search action 的 2%。

另外增加训练 readiness 事实检查，而不事后修改旧 G2：

- 至少出现 1 条 strict-EM 正样本；
- 至少出现 1 个 strict-EM mixed group；
- `FINAL_ANSWER:` 抽取、trace 对齐和 reward 重放完全一致。

任何一项为零都停止，不进入 G3 或训练。

### 5.2 G3 capability gate

在新的 exact attempt 中复用原 G3 固定的 held-out Hotpot-64、group-5、320 条规模和保守阈值。native-v2 答案合同与训练兼容采样是本文新注册的实验修订，不追溯声称为旧 G3 的事前设置：

- 有效正确多搜轨迹 `>=16/320`；
- 覆盖题目 `>=8/64`；
- learnable group `>=8/64`；
- clipped 与 invalid rate 均 `<=5%`。

若 G3 `NO-GO`，保留完整证据并停止。只有在确认检索链可用、但仍缺乏正确探索时，才另立小规模 oracle trajectory SFT warm-start 假设；不能把它混入当前 GRPO 复现。

### 5.3 两步训练 smoke

G3 `GO` 后只运行 2 个更新 step。必须同时满足：

- reward、advantage、loss、KL、entropy、grad norm 全部有限；
- 至少一个 group 有正确/错误奖励差，策略 token 上存在非零 advantage；
- old/current log-prob、response mask 与 tool observation mask 对齐；
- 无 OOM、CUDA、NCCL、Ray 或 checkpoint 错误；
- 保存 step-2 checkpoint、resolved config、完整 train log、WandB offline history 和逐轨迹 trace。

smoke 不通过则停止，不通过降门槛、盲降 batch 或直接跑 60 steps 掩盖问题。

### 5.4 R/B/C 分支关系与奖励公式不变

2-step smoke 通过后才启动全参数 `R-mix60-native-v2`。R 完成后重新通过能力门，并要求至少 8/64 个 cost-contrast group；只有此时才从同一个 R checkpoint 对称训练：

```text
B-mix20-native-v2:       r = EM
C-gated-mix20-native-v2: r = EM * (1 - 0.10 * n_search / 4)
```

B/C 继续共享 parent digest、数据顺序、seed、batch、group、长度、协议、检索器和步数。最终比较 strict EM、搜索次数、统一 utility、共同答对题搜索差，以及代表性完整轨迹。

## 6. 明确不改的内容

为避免再次扩大实验空间，本轮固定：

- 已后训练的 Qwen3.5-2B，全参数微调，不改 LoRA；
- 两张 5090 级 GPU；
- response 500、最多搜索 4 次；
- train batch 8、GRPO group 5；
- 固定 train-512/val-128 样本集合、NQ/Hotpot `37.5%/62.5%` 配比、既定 Hotpot comparison/bridge 配额和 seed；
- Wiki-18 BM25 top-3、现有 corpus/index；
- 学习率 `1e-6`、warmup 0.285 和已有显存配置；
- G3、R 后能力门和 B/C 成功标准；
- 每个 paid attempt 的 exact evidence、原始 exit code 与 watchdog 关机合同。

不做模型 sweep、参数 sweep、多 seed、LLM judge、dense retriever、processed-proposal PPO 或通用训练平台。

## 7. GO/NO-GO 与停止条件

| 检查点 | GO | NO-GO / 工程失败后的动作 |
| --- | --- | --- |
| CPU handoff | 新 commit、数据、配置、测试和 handoff 全部一致 | 不开 GPU，修复后重新增量重封 |
| G2 readiness | 格式门通过且 strict reward/mixed group 非零 | 归档并停止，不靠 substring 或调阈值强行放行 |
| G3 | 三项能力门和质量门全部通过 | 归档科学负结果，不启动 GRPO 长训 |
| 2-step smoke | 数值、mask、reward、advantage、checkpoint 全部正常 | 保留失败 checkpoint/log，不启动 R-mix60-native-v2 |
| R 后复测 | 能力门和 cost-contrast 门均通过 | 只报告 R，不启动 B/C |
| B/C 完成 | 同一 parent 下完成配对评测 | 无论正负结果均归档，不通过继续调参制造成功 |

## 8. 为什么仍然有希望

本轮最重要的正面证据不是“门槛只差两条”，而是问题已经从模糊的“模型不搜索”收敛为可验证接口：

1. 原生 tool call、tool response 和多轮上下文已在真实模型上跑通。
2. 51/96 条轨迹至少二搜，36 条恰好二搜；搜索能力不再是零。
3. 多条轨迹已经检索到答案证据并生成内容正确的答案，只是短答案接口没有把它们转成 strict reward。
4. 人工明确正确标签下已有 13/32 个 mixed group，说明不是所有 group 都全错。
5. G2 没有显存、分布式或运行时错误，工程链路本身稳定。

剩余不确定性是：关闭 top-k/presence 后工具行为是否仍稳定、canonical 短答案遵循率是否足够，以及 2B 模型能否在预注册 G3 门槛下形成充足的正确多搜探索。因此当前判断应表述为“值得按最小修复继续验证”，而不是“已经保证训练成功”。

## 9. 证据位置与文档关系

当前 AutoDL immutable evidence：

```text
G0+G1 outer:
/root/autodl-tmp/search-r1/state/attempts/gpu/20260724T033923Z-1722-21568

G2 outer:
/root/autodl-tmp/search-r1/state/attempts/gpu/20260724T041132Z-1559-23367

G2 eval:
/root/autodl-tmp/search-r1/runs/eval/qwen_native_g2/attempts/20260724T041309Z-1728-2268

G2 analysis:
/root/autodl-tmp/search-r1/runs/qwen-native-gate/attempts/20260724T041132Z-1559-23367

G2 trace SHA-256:
6da6f333b3c97a92d34b8b377bde4228ccf26827cfd513785c4d6f210b8f4755
```

文档职责：

- [`qwen35_native_tool_adaptation_plan.md`](qwen35_native_tool_adaptation_plan.md)：运行前的协议适配与 G0-G3 预注册计划；
- [`qwen35_native_tool_adaptation_implementation_report.md`](qwen35_native_tool_adaptation_implementation_report.md)：运行前实现快照与 fail-closed 原因；
- 本文：G0-G2 运行后的问题清单、新实验假设与训练解阻顺序；
- [`autodl_search_r1_reproduction_plan.md`](autodl_search_r1_reproduction_plan.md)：R/B/C 总体实验不变量与最终比较标准；
- [`成本感知坍缩分析与改进建议.md`](成本感知坍缩分析与改进建议.md)：提供历史坍缩证据与 correctness-gated 奖励设计依据；其中旧 parent、数据和三路回填安排不直接套用于 native-v2，当前执行顺序以本文和总体复现计划为准。

旧 legacy grouped probe 继续保存在 [`results/grouped-probe-20260723/`](results/grouped-probe-20260723/)，不得与本次 96 条 native G2 轨迹混算。
