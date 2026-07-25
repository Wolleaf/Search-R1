# Qwen3.5 Search-R1 原版对齐审计与一次性修复方案

> 日期：2026-07-25
>
> 状态：修复实现已完成并通过本地回归；CPU 增量封存由同一提交的
> `cpu_handoff.json` 记录；尚未重新运行 GPU
>
> 原版代码基线：`origin/main@598e61bd1d36895726d28a8d06b3a15bed19f5d3`
>
> 当前审计基线：`experiment/hotpot-search-gate@eae57c3ade2a3cad3549f70db1c5dd4e601ec949`
>
> 最新证据：[`qwen35-native-v2-g0-g1-20260725`](results/qwen35-native-v2-g0-g1-20260725/README.md)

本文是后续 Qwen3.5 协议补正的统一依据。它取代此前文档中与“原版语义 + 必要 Qwen3.5 适配”冲突的建议，但不覆盖旧实验、旧轨迹或旧 NO-GO。旧结果继续作为失败证据原样保留。

实现提交在交付前通过 369 项 pytest、16 项子测试、9 个 AutoDL shell
回归、11 个变更 shell 文件语法检查、Python `compileall` 和云工作流静态
审计（0 error / 0 warning）。这些是本地工程验证；真实 tokenizer、Parquet
和 640 条 evidence 的最终验收仍以无卡实例生成的 handoff 为准。

## 1. 结论与修复边界

当前问题不是单独多了一句“至少搜索一次”，而是 Qwen3.5 native 适配同时改变了提示词、thinking、action 边界、非法动作恢复、parser 和门禁含义。最新运行证明检索基础设施正常，但没有证明原版提示词下的自主搜索能力，也没有进入 RL 训练。

下一次不得继续局部打补丁。应先在同一个 CPU 修复周期内完成以下内容，再只跑一次 GPU 门禁：

1. 恢复原版完整提示语义和单 user message，只翻译原生工具调用与结果回填格式。
2. 使用 Qwen3.5 原生 thinking 模式，恢复每次获得新信息后显式推理。
3. 恢复“首个完整 action 即结束本轮”的 Search-R1 边界。
4. 恢复中性 retry，不再提示必须搜索或优先搜索。
5. 删除科学流程中的 forced-search 数据和依赖 forced-search 的门槛。
6. 把 action budget 修为论文的总 `B=4`，并把 observation 从临时的 384 恢复为 500。
7. 取消“未训练 parent 必须先会多搜”的 G3 硬门槛；能力指标只诊断，协议和数值正确性才阻断训练。
8. 将筛选后的 `val_128` 降级为机制集，最终增加无检索可见性筛选的标准子集评测。
9. 保留 strict EM、retrieved-token loss mask、原生 tool role、完整轨迹和成本日志。

本轮不同时修改数据配比、模型、batch、group、response、采样、奖励、检索器或训练步数。`B=4` 与 observation 500 是一组互相约束的原版状态机恢复，不是为了调效果；除此之外冻结所有训练变量，避免再次成为多变量实验。

## 2. 依据与优先级

发生冲突时按以下顺序判断：

1. 论文 [`2503.09516v5.pdf`](2503.09516v5.pdf) 第 5 页 Table 1、第 6 页 Algorithm 1/奖励、第 7 页实验设置和第 16 页 Appendix B.2 决定科学语义。
2. `origin/main` 决定论文未展开的官方实现方式。
3. Qwen3.5 固定 revision 的 chat template 决定不可避免的接口序列化。
4. 用户明确要求的 Qwen3.5-2B、两张 5090、数百元预算和缩小数据规模作为已披露的复现边界。
5. 日志、哈希、断点、关机和门禁只能增加可复核性，不能改变模型看到的策略指令或 reward。

审计不需要用户另下载一份原项目。当前 Git 对象库已经包含官方基线 commit `598e61bd1d36895726d28a8d06b3a15bed19f5d3`，后续统一用 `git show 598e61b:<path>` 和 `git diff 598e61b..HEAD` 做只读逐文件比对；单独复制一份源码反而容易比较错版本。

## 3. 原版 Search-R1 合同

### 3.1 原版提示词

论文 Table 1 的提示词为：

```text
Answer the given question. You must conduct reasoning inside <think> and </think>
first every time you get new information. After reasoning, if you find you lack some
knowledge, you can call a search engine by <search> query </search>, and it will
return the top searched results between <information> and </information>. You
can search as many times as you want. If you find no further external knowledge
needed, you can directly provide the answer inside <answer> and </answer> without
detailed illustrations. For example, <answer> xxx </answer>. Question: question.
```

上游仓库在 [`nq_search.py`](../scripts/data_process/nq_search.py) 和 [`qa_search_train_merge.py`](../scripts/data_process/qa_search_train_merge.py) 中保存同一语义的单条 user message，只把答案示例换成 `<answer> Beijing </answer>`，并带有 `as your want` 的拼写错误。关键语义是：

- 每次得到问题或新检索信息后先在 `<think>` 中推理；
- 只有缺少知识时才可以搜索，允许零搜索直接回答；
- 是否继续搜索由模型自主决定；
- 最终答案位于 `<answer>` 中且应简短；
- prompt 不要求至少一次搜索，也不暴露最大搜索次数。

仓库真实案例 [`example/case.txt`](../example/case.txt) 在没有强制首搜的情况下自主搜索两次，再输出 `<answer>`。

论文和 Qwen2.5 原实现没有 Qwen3.5 的 `enable_thinking` 模板开关；它们通过 prompt 和生成文本中的 `<think>...</think>` 表达 thinking。当前适配设置 `enable_thinking=False` 后，固定 Qwen3.5 template 会预填空 thinking，而 parser 又拒绝非空 `<think>`，实际效果恰好是取消论文要求。因而在 Qwen3.5 上改为 `enable_thinking=True` 是协议映射，不是新增思维链技巧。

### 3.2 原版状态机与奖励

论文 Algorithm 1 的每个 action 生成到第一个 `</search>`、`</answer>` 或 EOS 即停止。search 触发检索并追加 `<information>`，answer 终止，非法动作追加中性提醒：

```text
My action is not correct. Let me rethink.
```

论文奖励只有抽取最终答案后的 EM：

```text
r(x, y) = EM(a_pred, a_gold)
```

没有 format reward、搜索奖励或神经 reward model。上游 `qa_em.py` 从完整 transcript 取最后一个 `<answer>`；之所以要求至少两个 match，是因为 prompt 本身带有一个答案示例。native 路径把当前 assistant action 中抽取的答案保存为 batch-aligned `final_answer`，再调用同一个 `em_check()`，属于必要接口适配，数学奖励没有变化。

严格按 Algorithm 1 时，如果第 4 个 action 仍是 search 或 invalid，环境在预算耗尽后直接返回已有 rollout，不偷偷追加一次回答机会；该轨迹没有合法 `<answer>`，EM 就是 0。这一点虽然会降低部分完成率，却是恢复论文状态机，不能靠 terminal 提示或免费第 5 action 修饰结果。

### 3.3 论文、上游 recipe 与当前实现的差异

论文 GRPO 配置包括 group 5、学习率 `1e-6`、warmup `0.285`、temperature/top-p `1.0/1.0`、response 500、序列 4096、retrieved content 500、top-3、500 steps、总 batch 512 和 8 张 H100。论文 `B=4` 表示 search、answer、invalid 共用的总 action budget。

论文与官方仓库的 runnable recipe 并非逐项一致，不能把“继承上游”自动等同于“遵循论文”：

| 项目 | 论文 | `origin/main` | 当前结论 |
| --- | --- | --- | --- |
| action budget | `B=4` 个总 action | 先跑 `max_turns`，未结束再额外生成 terminal action | 以论文为准，删除额外第 5 action |
| observation | retrieved content 500 | `max_obs_length=500` | 当前 384 应随严格 `B=4` 恢复为 500 |
| GRPO steps | 500 | recipe 写 1005 | 缩小实验继续用已批准的 R60/B20/C20 |
| top-p | 1.0 | recipe 未覆盖、会继承框架默认值 | 当前显式 1.0 更贴论文 |
| 起始 prompt 上限 | 只规定总序列 4096 | recipe 写 2048 | 当前 1024 可保留，但必须证明所有 prompt 零截断 |

当前容量校验按“4 × (response 500 + observation 384) + 额外 response 500 = 4036”设计，所以 384 与第 5 action 是同一个派生偏差。修成总 `B=4` 后，最坏四个 search action 为 `4 × (500 + 500) = 4000`，可在 4096 的右侧轨迹预算内恢复论文 observation 500。两项必须一起改、一起测，但应以独立提交与 prompt/Qwen 适配区分，便于审计归因。

## 4. 全量差异审计

### 4.1 会改变模型策略的非必要偏差

| 环节 | 论文 / 上游 | 当前实现 | 处理 |
| --- | --- | --- | --- |
| 初始消息 | 完整模板放在一条 user message | 手写 system + 简化 user | 删除手写 system，恢复完整 user |
| 问题规范化 | 压缩空白，缺少时补 `?` | native 仅 `strip()` | 与上游使用同一清洗/补问号规则 |
| 首次搜索 | 自主选择 | G1 加 `Call search at least once...` | 从科学流程删除 |
| thinking | 每次新信息后 `<think>...</think>` | `enable_thinking=False`，非空 think 被拒 | 改为原生 thinking，并保留非空推理 |
| 搜索上限 | prompt 写可搜索任意次，环境控制预算 | system 写 `Use at most four searches` | 从 prompt 删除 |
| 工具策略 | 只定义搜索功能 | schema 写 `Call exactly one search per assistant turn` 和 specific query | schema 只描述输入输出 |
| action 结束 | 第一个完整 search/answer action | native 生成到 EOS 后整段 fullmatch | 按首个完整 native action 截断 |
| 多调用 | 第一处闭标签后已结束本轮 | 同回合多个完整调用全部 invalid | 执行首 action；完整 raw 另存审计 |
| retry | 中性 rethink | `Call search once... or output...` | 恢复中性语义 |
| retry role | 原版继续 rollout；无真实工具事件 | native 伪装成 tool response | 使用合法的非工具重试回合 |
| query 质量 | parser 只判断动作语法 | `query`、`and` 黑名单混入 parser | 语法与质量分离；退化 query 只记指标 |
| action budget | 论文 `B=4` 个总 action | 4 个可搜索 action 后再生成第 5 个 terminal | 严格恢复总 `B=4` |
| observation | 论文与上游均为 500 | 为容纳第 5 action 压成 384 | 随 `B=4` 恢复 500 |
| terminal | 达到 `B` 即返回，无“只准回答”提示 | 额外 terminal 且旧计划建议加终止指令 | 删除额外 action，不采用新提示 |
| G1 门槛 | 无强制首搜门 | 要求至少 29/32 非退化首搜 | 删除该门槛 |
| 训练前能力门 | RL 用于学习搜索行为 | parent G3 必须 GO 才允许 R 训练 | G2/G3 只诊断，不阻断 R |
| 检索对齐 | 真实请求应有真实回填 | 把 terminal 未执行 search 计作丢回填 | requested/executed/response 分开统计 |
| 最终评测 | 七个标准 test/dev split | 主要使用检索可见性筛选后的 `val_128` | 机制集与无筛选泛化集分开 |

这些偏差集中在 [`tool_protocol.py`](../search_r1/llm_agent/tool_protocol.py)、[`generation.py`](../search_r1/llm_agent/generation.py)、[`search_mix.py`](../scripts/data_process/search_mix.py)、[`qwen_native_gate_analysis.py`](../scripts/autodl/qwen_native_gate_analysis.py) 和 [`08_gpu_qwen_native_gate.sh`](../scripts/autodl/08_gpu_qwen_native_gate.sh)。

范围必须说准确：`Call search at least once...` 只写入当前 `probe_forced_16`，正式 `train_512` 和 `val_128` 的 `force_search=False`，因此还没有“强制首搜训练污染”。但自写 system、关闭 thinking、EOS-only action 边界、parser 与 retry 是正式 native 训练候选路径共用的，若不修就会真实进入 RL。

### 4.2 必须保留的 Qwen3.5 接口适配

以下变化不可避免，但不能借此改写搜索策略：

| 原版表示 | Qwen3.5 必要映射 |
| --- | --- |
| `<search>query</search>` | 固定 schema 的原生 `search(query)` tool call |
| `<information>...</information>` | chat template 生成的 tool-response role |
| 新信息后继续同一 XML rollout | 模板生成新的 assistant thinking 回合 |
| `<answer>short answer</answer>` | 继续保留，不改成 marker-free 普通文本 |
| 从 transcript 抽最终答案 | 环境保存 batch-aligned `final_answer` 后执行同一 strict EM |
| retrieved token 不参与 loss/KL | tool response 和角色 wrapper 保持 observation mask |

Qwen chat template 自动注入的默认系统内容、`# Tools`、`<IMPORTANT>`、工具调用语法和 generation prefix 属于第三类 vendor 必要序列化；数据本身仍只有原版单 user message。该层必须绑定固定 revision/template hash，并验证 direct render 与 manager token 完全一致。本项目不应删除 vendor 协议，也不应再叠加一份自写的行为 system prompt。

### 4.3 已明确接受的缩小复现差异

这些差异影响论文绝对数值，但来自模型、硬件和预算约束，不在本次协议修复中更换：

| 项目 | 论文 | 当前缩小复现 | 定位 |
| --- | --- | --- | --- |
| 模型 | Qwen2.5-3B/7B Base 与 Instruct | post-trained `Qwen/Qwen3.5-2B` 固定 revision | 用户指定的小模型路线；不是 Base |
| GPU | 8×H100 | 2×5090 级 32 GiB | 预算约束 |
| 算法 | 默认 PPO，同时报告 GRPO | GRPO | 论文支持，项目简化 |
| 训练规模 | 500 steps、batch 512 | R 60 steps、batch 8 | 缩小复现 |
| 数据 | NQ + HotpotQA 完整训练集 | 512 train / 128 val，37.5% NQ + 62.5% Hotpot | 用户批准的检索验证子集 |
| 样本选择 | 未披露检索可见性筛选 | 按 BM25 可见证据筛选 | 提高小样本信号，不能声称同分布 |
| 检索器 | Wiki-2018 + E5 dense | 同类 Wiki-2018 语料 + BM25 | 存储与费用约束 |
| rollout | vLLM | HF + SDPA | Qwen3.5/5090 工程选择 |

`Qwen/Qwen3.5-2B` 是带 chat template 和 thinking 能力的后训练模型，不是 `Qwen3.5-2B-Base`。论文同时研究 Base 与 Instruct，因此选后训练路线本身合理，但简历和报告必须明确为“Instruct/post-trained 小模型复现”，不能写成只把 Qwen2.5 Base 换代。当前不偷偷切 Base；那会同时改变模板、先验能力、历史轨迹和训练成本，应另立实验才有可比性。

新 v3 报告和图表应把现有 `A / Base` 改名为 `A / Parent (post-trained)`；旧结果中的标签不回写，只在历史说明中更正。这里的 A 是“尚未经过本项目 RL 的 parent”，不是预训练意义上的 Base。

因此最终项目应表述为“复现 Search-R1 核心训练闭环并做成本扩展”，不能表述为复刻论文 benchmark 数值。论文七个评测集是 NQ、TriviaQA、PopQA、HotpotQA、2WikiMultiHopQA、Musique 和 Bamboogle；当前 val 是按检索机会筛选的 held-out 机制集，不等于这些标准测试分布。最终至少补上已经定义好的 NQ test-128，以及 HotpotQA/2WikiMultiHopQA 各 128 条组成的无筛选 `eval_256`；其余四个数据集作为预算未覆盖项披露。

### 4.4 不改变科学语义的工程增强

以下内容可以保留：固定 revision 和哈希、CPU/GPU handoff、离线 W&B、完整轨迹、配置快照、checkpoint lineage、退出码、自动关机、下载重试、双卡 padding 修复、FSDP offload、日志精度和 fail-closed 文件校验。HF + SDPA、top-k 0、min-p 0、presence penalty 0、repetition penalty 1 也是两卡后端或中性关闭值；它们保留论文的 temperature/top-p 分布目标，但不能宣称与 vLLM bitwise 等价。检索 observation 继续使用上游的 `Doc n(Title: ...)` payload，只把外层 `<information>` 换成 native tool response，不顺手改文档拼接格式。

已经核实无需回退的核心项包括：全参数训练而非 LoRA、GRPO group 5、学习率 `1e-6`、warmup `0.285`、temperature/top-p `1/1`、KL `0.001`、clip `0.2`、top-3、strict normalized EM，以及检索/tool token 的 loss/KL mask。成本分支的 correct-only reward 不是论文内容，但它只在 C 分支启用；R/B 保持 `lambda=0`，因此应继续作为用户明确的独立扩展隔离。

## 5. 这次失败应如何重新解释

### 5.1 精确运行边界

最新运行只有 G0 和 forced-G1：G0 为 8 题 × 2 槽的 direct/native/legacy 对照；G1 为 16 题 × 2 槽，共 32 条轨迹；`train_steps=0`，没有 optimizer update、新 checkpoint 或 loss 曲线。外层、G0、G1 均 `exit-code=0`，科学结果为 NO-GO。

G1 的 32/32 条轨迹都看到“至少搜索一次”，所以以下数字不能证明自主搜索：平均实际搜索 2.75、28/32 至少二搜、14 条完整二搜链、native-v1 到 v2 的搜索数增长。后续 retry 和 terminal 还会进一步抬高搜索计数。

### 5.2 仍然成立的证据

- 88/88 次真正发给 BM25 的请求都有 retrieval event、非空 observation 和 3 篇文档；检索服务没有丢包。
- 32 条轨迹中没有 response clipping，最大生成 444 tokens；本轮失败不是 response 500 太短。
- G0 direct 与 manager 的 prompt/raw output 逐槽相同；manager 没有偷偷改写当前输入。
- 当前 parser 能从带前缀的 `<answer>` 抽取短答案，strict EM 为 2/32；奖励链路不是恒为零。
- 只有 8/32 有可抽取答案，24/32 无答案，21/32 含非法动作；当前格式和终止不能支持直接训练。

### 5.3 轨迹对修复的直接指向

1. G0 的 Lisa Cholodenko / Pierre Morel 问题在同一回复生成 2-3 个完整 tool call。原版会在第一个完整 action 结束，当前却把整段判 invalid。这直接支持修 action 边界。
2. Ryan Neates 成功槽通过“球队 -> 官方颜色”得到答案，第三搜已经冗余；失败槽第二搜已拿到答案，却因无标签文本和 retry 继续搜索。这证明检索与成本空间存在，但不能从 forced 条件外推搜索策略。
3. 机场轨迹第二搜已得到 Honolulu International Airport，之后仍搜到 terminal；它没有 parser invalid，说明终止问题不全是 retry 造成的。
4. Jung Joon-young 失败槽已经回答 Park Jin-pyo，却没有 `<answer>`，随后发明新工具；同题另一槽正确产生 `<answer>Park Jin-pyo</answer>`。这支持保留原版答案边界，而不是放宽为整段文本。

已经确认的是接口和状态机存在偏差；尚未通过单变量实验确认的是 thinking、custom system、采样或 2B 模型先验各自贡献多少。修复前不能把 NO-GO 归因于“Qwen3.5 不会工具”或“模型太小”。

## 6. 修复后的唯一 prompt 合同

数据仍保存一条 user message。先按上游规则压缩问题空白并在缺失时补 `?`，再以仓库 prompt 为底稿，只把 XML 搜索和 observation 语法翻译为 Qwen 原生接口：

```text
Answer the given question. You must conduct reasoning inside <think> and </think>
first every time you get new information. After reasoning, if you find you lack some
knowledge, you can call the available search tool with a query, and it will return
the top searched results in a tool response. You can search as many times as you
want. If you find no further external knowledge needed, you can directly provide
the answer inside <answer> and </answer>, without detailed illustrations. For
example, <answer> Beijing </answer>. Question: {question}
```

项目自写的 user 行为语义相对上游只允许以下三项白名单差异：

- `<search> query </search>` -> `search(query)`；
- `<information>...</information>` -> native tool response；
- 把上游 typo `as many times as your want` 恢复成论文 Table 1 的 `as many times as you want`。

前两项是 Qwen 接口映射，第三项是论文优先级下的非语义拼写纠正。答案示例继续采用上游的 `<answer> Beijing </answer>`。测试按这份显式白名单逐字比较，不能笼统写成“只改两处”后再让实现猜测。

固定 Qwen vendor template 额外注入的 tools 说明和控制 token 是必要的第三类序列化差异，不计作项目自写 prompt；其完整 token/hash 必须冻结，不能手工摘抄或再改写。

不得加入“至少搜索一次”“最多四次”“每轮必须一个工具”“必须使用具体查询”“预算耗尽只能回答”、few-shot 搜索策略或反思技巧。工具 schema 描述仅说明“使用一个字符串 query 搜索外部知识库并返回相关段落”。

`enable_thinking=True` 时，固定 Qwen template 会打开原生 thinking。实际 sampled continuation 可能是“reasoning tokens + `</think>` + action”，因为 opening `<think>` 已在 generation prefix 中；parser 和 conversation 重建必须按真实 token 模板处理，不能要求 sampled text 再重复 opening tag。

## 7. 文件级一次性修复计划

### P0-1：先用测试锁定合同

先改测试、后改实现。测试必须明确禁止 `at least once`、`at most four`、`exactly one search` 和带搜索倾向的 retry，并覆盖：

- 原版完整单 user prompt；
- 上游空白规范化与末尾补 `?`，按“两处接口映射 + 一处论文 typo 修正”白名单逐字比较 user text；
- 真实 tokenizer 的 `enable_thinking=True + tools`；
- 零搜索 `<answer>`、一次 search、多次 search；
- 非空 thinking 后的 search/answer；
- 同一 raw generation 有多个调用时只执行第一个完整 action；
- invalid 后的中性 rethink；
- tool response mask、raw token 保真和 strict EM。

主要文件：`tests/test_qwen35_tool_protocol.py`、`tests/test_generation_qwen35_native.py`、`tests/test_rl_dataset_tool_protocol.py`、`tests/test_search_mix.py`。

### P0-2：修 prompt、thinking、schema 和 parser

在 [`tool_protocol.py`](../search_r1/llm_agent/tool_protocol.py) 中：

1. 删除手写策略 system、`force_search` 和搜索倾向 retry。
2. 复用上游问题清洗和补问号规则，物化一条完整 user prompt；只让 chat template 注入 native tools。
3. 工具 schema 移除一次调用、搜索上限和 query 质量策略。
4. 所有 renderer 改为 `enable_thinking=True`。
5. parser 接受 Qwen 原生 thinking continuation，并保存真实 reasoning；不再拒绝非空 think。
6. parser 只判动作语法；`query`、`and`、重复或低相关 query 进入 trace 指标，不决定语法合法性。
7. 普通 marker-free 最终文本仍不是合法答案；原版 `<answer>` 合同保持。

论文没有 format reward，所以 `<think>` 合规率应记录，但不能额外加格式分或格式罚分。

### P0-3：修 action 边界和多轮重建

在 [`generation.py`](../search_r1/llm_agent/generation.py) 中恢复上游首 action 语义。为保持实现最小且不 decode/re-tokenize：

1. 保留 HF 返回的完整 raw token/text 用于审计。
2. 在原 sampled token IDs 上按最早的完整 `</tool_call>`、`</answer>` 或 EOS action 边界切片；没有完整 action 时保留到 EOS 作为 invalid。
3. 只把该精确 token prefix 放入策略轨迹、环境执行和 PPO log-prob；不得重新编码字符串。
4. parser 对切片后的单 action 工作，不再因 raw tail 中第二个调用把第一调用作废。
5. thinking 内容正确写入 Qwen message 的 `reasoning_content`，action 写入 `content`；tool result 后由模板开启下一次 thinking。
6. invalid action 的 sampled reasoning/action 先按真实 assistant message 重建，再追加一条 **user-role** 的论文原文 `My action is not correct. Let me rethink.`，然后由固定 template 开启下一次 assistant thinking。只允许这一种 role/文案；从重渲染结果提取 suffix 时必须断言此前累计 sampled token 逐 token 不变，做不到就判 adapter 失败，不能退回伪 tool role 或自造固定 suffix。

这与上游“生成后截到首个 action”一致，同时比上游 decode-truncate-reencode 更严格地保存 Qwen 实际 token。完整 raw 仍能展示模型是否有并行多调用倾向，但 raw tail 不进入训练轨迹。

### P0-4：定义 prompt-only 数据重物化合同

在 [`search_mix.py`](../scripts/data_process/search_mix.py) 中提升 prompt/schema version。v3 不再生成或引用 `probe_forced_16.parquet`，而是用相同 16 个 sample ID 生成自主 `probe_autonomous_16.parquet`；历史 v2 forced 文件仍原样归档。P0 只实现和测试 builder，等 P1/P2 的最终运行合同确定后再从现有 sealed source **一次** materialize 和 seal 新目录：

- catalog、问题、gold、split、顺序、配额和检索证据逐项相同；
- Parquet 行只改变 prompt；manifest/handoff 还会因 prompt/schema version、chat-template digest、总 `B=4` 和 observation 500 合同而更新，所有派生 digest 重新封存；
- provenance 必须保留 `selection_observation_length=384`：现有 catalog 的检索可见性筛选确实按 384 完成；新运行另记 `rollout_observation_length=500`，不能覆写历史字段或谎称按 500 选题。500 包含原 384-token 前缀，因此已有“证据在 384 内可见”的样本仍成立，无需重选；
- 不重新下载数据、不重新运行 BM25、不重新选题；
- v1/v2 目录和 evidence 保持只读。

### P0-5：修 gate，而不是换一句 prompt 后沿用旧门槛

更新 `qwen_native_protocol_probe.py`、`qwen_native_gate_analysis.py` 和 `08_gpu_qwen_native_gate.sh`：

- G1 使用自主 16×2，不要求任何最低首次搜索率或平均搜索次数；
- direct/manager 比较 canonical first-action token prefix，同时保存完整 raw；
- `requested_search`、`executed_search`、`retrieval_event` 和 `nonempty_tool_response` 分开；旧 v2 的 `terminal_search_request` 仅为历史兼容字段；
- 真实执行搜索必须一一得到回填，未执行请求不得再命名为检索丢包；
- strict EM、答案率、thinking 合规、非法动作、clipping 和搜索分布全部报告，但不事后修改旧阈值或旧结果。

### P0-6：机械更新全部 active 绑定

协议实现通过后，一次性更新 `02_cpu_prepare.sh`、`train_small_grpo.sh`、`08_gpu_qwen_native_gate.sh`、`09_gpu_qwen_native_train.sh`、`04_watch_and_shutdown.sh`、trainer/trace/result contract 及对应 Python/shell tests。重点清除 active path 中硬编码的 `enable_thinking=False`、native-v2 prompt version、forced artifact、rollout observation 384、额外 terminal 证据和 pretrain-G3 前置依赖；watchdog 的 stage order、lineage 与关机条件也必须同步，否则主脚本放行后仍会被 watcher 拒绝。新产物统一使用 v3 目录/名称，绝不覆盖或复用 v1/v2 handoff。最后用全局 `rg` 反查旧常量；legacy/history/test fixture 中有意保留的命中必须逐条标注来源，而不是盲目全局替换。

### P1：独立提交恢复论文 `B=4` 和 observation 500

P0 完成后，用独立提交把当前“4 个可搜索 turn + 1 terminal”改成论文的“总 action budget 4”：search、answer、invalid 都消耗一次 action，达到 4 后直接返回当前 rollout，不新增第 5 个 terminal，也不增加“只准回答”的 prompt。同时把 `max_obs_length` 恢复为 500，并把容量合同改为 `B × (response + observation) <= 4096` 的最坏 search 路径。

这会改变上游 Agent 循环，而不是 Qwen 接口映射，因此与 P0 分开 commit、测试和记录；但 P0/P1 应在同一 CPU 开机阶段全部完成，再统一跑一次 GPU gate，不能先按旧状态机训练后再返工。

同时提升 trace schema：active v3 记录使用 `max_action_budget=4`，并分别保存 `action_count` 与 `executed_search_count`。当前把 `config.max_turns` 写成 `max_searches` 的字段不得继续出现在新结果中；旧 v1/v2 reader 只做兼容映射，不能把历史字段改名后重写 archive。同步修改 `ray_trainer.py`、`trajectory_trace.py`、gate/result analyzer 及 schema tests。

### P2：取消 parent 能力硬门，保留工程门禁

修改 `09_gpu_qwen_native_train.sh`、`04_watch_and_shutdown.sh` 及其 handoff/evidence/result 合同：

- G0/G1 只判断 prompt/token/action/retrieval/mask 协议是否正确；
- 未训练 parent 的 G2/G3 搜索率、EM、组内 reward 方差只作为诊断，不再要求 `G3=GO` 才能训练；
- 2-step smoke 仍必须验证 loss、梯度、checkpoint、双卡退出码和轨迹落盘；
- R 训练后的 G3 才用于判断是否已有足够“答对且多搜”的样本值得开启 B/C 成本分支；
- 不得用能力不足触发 forced prompt、format reward 或临时 SFT。

R 后 G3 继续使用预注册的 64 题 × group 5 = 320 轨迹，门槛本轮冻结，不在看到结果后调整：

| 条件 | 固定门槛 |
| --- | --- |
| valid correct multi-search | `>=16/320` |
| 被 qualifying 轨迹覆盖的问题 | `>=8/64` |
| learnable groups | `>=8/64` |
| response clipping ratio | `<=5%` |
| 含 invalid action 的轨迹比例 | `<=5%` |
| B/C cost-contrast groups | G3 GO 后还需 `>=8/64` |

qualifying 的定义保持现有预注册合同：strict EM=1、executed searches >=2、未 clipping、invalid count=0、前两次 query token Jaccard <0.8，且第二次检索增加新 document ID，并首次增加 supporting title 或让 gold alias 可见。learnable group 需要至少一条 qualifying 与一条干净 EM=0；cost-contrast group 需要至少两条干净 EM=1 且正确轨迹搜索数不同。near-miss 只诊断，不参与 GO。

### P3：机制评测与泛化评测分开

保留筛选后的 `val_128` 作为训练期机制监控；把现有 `data/nq_small/test_128.parquet` 和 `multihop_search_gate.py` 生成的 HotpotQA/2WikiMultiHopQA `eval_256.parquet` 接入 native 最终评测。现有文件仍是 legacy XML prompt，不能直接喂给 native 路径；应从各自 sealed catalog 按相同 IDs/gold/order 只重物化 v3 prompt，发布新的 native eval artifact 和 digest，不重新抽样。所有 checkpoint 使用同一 prompt version、BM25、采样与 sample IDs：

1. A（未训练 parent）与 R（Search-R1）比较能力增益；
2. 同一 R parent 出发的 B（原 EM reward）与 C（成本 reward）做逐题配对比较；
3. 同时报告 EM、executed searches、correct-only search、轨迹长度、invalid、clipping 和置信区间；
4. `val_128` 结果只能说明富集机制集，共 384 题的无筛选子集才承担最小泛化证据；不声称覆盖论文七数据集。

同步更新新结果生成器和曲线标签为 `A / Parent (post-trained)`，避免继续把该 checkpoint 错称 Base；历史 archive 不改名。

终点评测固定为每题 group 1、一个 rollout、`do_sample=false`（greedy），temperature/top-p 配置仍记录为 `1/1`，top-k 0、min-p 0、presence 0、repetition 1、seed 42，并保持完全相同的数据顺序和 slot。B/C 以 sample ID 做逐题配对，CI 使用固定 seed 的 paired bootstrap；本轮不在看到结果后开启 sampling、增加 samples 或挑选有利 seed。训练和 grouped probe 仍按 group>1 使用 `do_sample=true`，两种用途不得混写。

### P4：协议通过后恢复训练和成本分支

最终顺序为：P0/P1 CPU 一次性补正 -> G0/G1 结构门 -> 2-step 数值 smoke -> R60 -> R 后能力 G3 -> 同一 R parent 的 B20/C20 对称分支 -> curated 与无筛选子集配对评测。不得在 P0-P3 期间改成本 reward 或根据本次结果调 lambda。

## 8. CPU 与 GPU 验收

### 8.1 CPU 阶段

CPU 阶段只做增量修复，无需重建环境、模型、语料或 BM25：

1. 用固定真实 tokenizer 渲染全部 v3 prompt；直接 tokenize 与字符串 tokenize 必须逐 token 相同。
2. 验证每个 generation prefix 确实开启 thinking，native tool schema 存在，自写策略 system 不存在。
3. 用合成 token 覆盖零搜回答、合法 search、二次 search、非空 think、双调用 raw、invalid retry 和 answer。
4. 验证总 action 计数严格不超过 4、没有额外 terminal、rollout observation 上限为 500，最坏容量不超过 4096；新 trace 只写 `max_action_budget`。
5. 验证每轮 reconstructed tokens 与 manager tokens 完全一致，observation/wrapper 全部被 info mask 排除。
6. 验证 v3 与 source 的非 prompt 字段逐行相同，manifest 同时保留 selection obs=384 与 rollout obs=500；用真实 native template 对 train 512 + val 128 共 640 条证据渲染 500-token follow-up，逐条确认原 catalog 记录的 gold/supporting evidence 仍可见。任一失败只封存报告并停止，不自动重选题或重跑 BM25；forced probe 不在 v3 科学入口。
7. 验证主训练、watchdog、lineage 都不再依赖 parent G3 GO，最终评测入口同时接受 curated 与无筛选 sealed 数据。
8. 运行聚焦 pytest、`compileall`、相关 shell tests、`bash -n` 和三路 Hydra config compose。
9. 生成新的 handoff 和 prompt/data digest；旧 handoff 不复用。

### 8.2 GPU 阶段

只运行 0-update 原版 prompt 门禁：

**E0：确定性环境 replay（非科学指标）**

- 不采样模型、不修改 prompt；向 manager 注入一条预写且合法的 native `search(query)` action；
- 在付费实例的真实 BM25 服务上要求 exactly one request/event、3 篇非空 `Doc n(Title: ...)`、一个 native tool response 和正确 info mask；
- 该结果只证明检索到 tool-role 回填闭环非真空，不计入自主搜索率、EM 或 G1 样本。这样即使 parent 在 G1 零搜索，也不会用 forced prompt 才能验基础设施。

**G0：8 题 × 2 槽**

- prompt token match 16/16；
- direct 与 manager 的 canonical first-action token prefix 16/16 相同；
- manager 不改写 thinking/action；
- raw 多调用可以记录，但策略 action 只含第一完整调用。

**G1：自主 16 题 × 2 槽**

- legal first action、thinking、answer/search 分布和 strict EM 全量报告；
- 每个真实执行 search 必须有且只有一个 retrieval event 和非空 tool response；
- 不要求“至少搜索一次”，直接 answer 不是接口失败；
- 不用平均搜索次数决定 adapter 是否正确；
- strict EM、组内 reward 方差和自主搜索率全部记录为 parent 能力基线，但不作为 adapter 的结构性 GO 条件。

thinking 指标以“每个实际 assistant generation turn”为分母，分别报告：template 是否提供 opening `<think>`、sampled reasoning 是否非空、是否在首 action 前出现合法 `</think>`，并按初始问题后/真实 tool response 后/user retry 后分层。它只诊断模型遵循度，不进入 reward，也不因 reasoning 为空单独判 adapter 失败。

E0/G0/G1 的结构性 GO 只看零 token 改写、零 action-tail 泄漏、零 observation mask 泄漏、执行搜索与 retrieval/tool-response 一一对应、无异常退出；legal rate、EM、搜索率和 thinking 遵循率不设事后能力阈值。

若 G0/G1 只是因为原版 prompt 下模型自主少搜，应记录为 parent 能力现状，而不是重新加 forced prompt。只有 prompt/token/action/retrieval/mask 协议错误才回 CPU 修代码。

G0/G1 结构通过后直接运行 2-step smoke；smoke 数值和落盘合同失败才阻断 R。未训练 parent 的大样本 G2/G3 可省略，若运行也只作一次诊断。R 后 G3 不通过时封存 R 结果并停止 B/C，避免为没有成本优化空间的 parent 继续烧钱。

## 9. 一次性决策与冻结变量

本轮的决策已经收敛，不在 GPU 运行途中再临时选择：

- **必须修复**：原版 prompt、thinking、首 action 边界、中性 retry、自主 gate、严格总 `B=4`、observation 500、取消 parent G3 硬门和补充无筛选评测。
- **保留但披露**：post-trained Qwen3.5-2B、BM25、富集 512/128 数据、GRPO、HF + SDPA、两卡小 batch 和缩短 steps；这些是模型/预算/研究范围差异，不伪装成必要工具适配。
- **保持不变**：strict EM、loss mask、全参数训练、group 5、response 500、采样/优化参数和 correct-only 成本公式。

E5 dense 是论文级检索器，BM25 不是等价替代；但现在切 E5 会重建大索引、重选富集数据并使全部历史轨迹失去同一检索环境，超出数百元最小复现范围。因此本方案明确保留并冻结 BM25，用相同 retriever 做 A/R/B/C 内部公平比较；若未来追求论文 benchmark，再把 E5 作为独立大规模复现实验，不能混进本轮补正。

| 变量 | 固定值 / 状态 |
| --- | --- |
| checkpoint | post-trained `Qwen/Qwen3.5-2B@15852e8c...` |
| 微调 | 全参数、双卡 FSDP/offload |
| train batch / GRPO group | 8 / 5 |
| 状态机 / 长度 | 总 `B=4`；start 1024、response 500、observation 500、trajectory 4096 |
| 搜索 | Wiki-18 BM25、top-3、同一 sealed index |
| 数据 | train 512、val 128、现有 37.5/62.5 配比和 sample IDs |
| 最终评测 | NQ test-128 + HotpotQA/2Wiki 无筛选 eval-256，同一 IDs |
| 采样 | train/grouped `do_sample=true`，endpoint group-1 `do_sample=false`；其余为 temperature/top-p 1/1、top-k 0、min-p 0、presence 0、repetition 1 |
| 优化 | LR `1e-6`、warmup `0.285`、KL `0.001`、现有 steps |
| baseline reward | strict EM、format score 0、cost lambda 0 |
| 成本分支 | 现有 correct-only 公式与 lambda 预注册值不修改；仅 R 后门禁通过才训练 |

不得同时加入 terminal 指令、few-shot、format SFT、LoRA、dense retriever、模型 sweep、多 seed、宽松 judge、SubEM reward 或新的数据配比。若 thinking 使生成更长，先记录 clipping；不要在同一轮立刻关闭 thinking、缩 response 或改 batch，从而再次失去归因。

## 10. 历史结果如何保留和纠正

最新 v2 archive 的文件、哈希、NO-GO 和轨迹保持不变。后续报告必须加上以下限定：

- G1 是 forced-search adapter stress test，不是原版 prompt 复现；
- 2.75 次平均搜索不能证明自主多搜，也不能证明模型已经学习；
- 88/88 真实检索回填健康、500 无截断和真实格式/终止故障仍有效；
- G0 manager 等价只证明 manager 没改写“当时的自定义 prompt”，不证明该 prompt 与论文一致；
- 本轮没有训练，不能提供 loss 曲线或新模型效果。

此前文档中“继续保持 `enable_thinking=False`”“不放原版答案示例”“native 必须生成到 EOS”“terminal 增加只准回答提示”以及 forced-G1 能力门的建议，均由本文取代。

## 11. 完成定义

只有同时满足以下条件，才算“原版语义的 Qwen3.5 必要适配”完成：

1. 项目自写的 user 行为语义与上游相比只有 search/tool-response 两处映射及论文 `your` -> `you` 的 typo 修正；额外模型可见内容只能来自固定 hash 的 Qwen vendor template。
2. Qwen thinking 在每个 assistant generation prefix 中真实开启，非空 reasoning 可保留、可审计。
3. 每轮只执行首个完整 action；raw tail 保留但不污染策略轨迹或 reward。
4. 搜索自主，任何训练/验证数据都没有 forced-search 或搜索次数提示。
5. retry 不诱导 search；总 action 数不超过 4，不存在额外 terminal action 或终止提示。
6. observation 恢复 500，最坏轨迹容量、prompt 截断和 info mask 均有自动测试。
7. parent 能力不足不会被误判为 adapter 故障，也不再由 G3 阻止 R 学习。
8. strict EM、loss mask、真实搜索计数和成本公式保持不变。
9. v3 数据行除 prompt 外与 sealed source 完全相同；manifest 明确区分按 384 形成的选样 provenance 与 500 的 rollout 合同。
10. curated 机制集与无筛选泛化集分开报告，A/R/B/C 使用同一评测合同。
11. 旧失败证据不改判，新 GPU 结果使用新 prompt version、commit、handoff 和 digest。

在此之前不得启动 R、B 或 C 正式训练。
