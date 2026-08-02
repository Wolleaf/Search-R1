# Qwen3.5 Native G1 边界故障与冗余回答修复计划

> 日期：2026-07-26
>
> 状态：修复实施版；生产代码与 CPU 回归已完成，尚未重跑 GPU
>
> 原版代码基线：`origin/main@598e61bd1d36895726d28a8d06b3a15bed19f5d3`
>
> 本次运行代码：`experiment/hotpot-search-gate@68de93040cf8a1887967d95ba92d2cf30b7d26ea`
>
> 失败 attempt：`20260725T142621Z-1663-1919`

本文补充
[`qwen35_search_r1_original_alignment_remediation_plan.md`](qwen35_search_r1_original_alignment_remediation_plan.md)，
只处理本次 G1 新暴露的 native reasoning/action 边界错误和 trace 写入错误，并解释
原版 prompt 下仍出现冗余回答的原因。不修改数据、奖励、模型、prompt、采样或训练
流程。完成新门禁并人工检查轨迹前，不启动 2-step smoke 或 R60。

## 1. 本次结果和结论

先给出直接答案：**当前项目自写的 user prompt 已与上游 Search-R1 保持行为语义
一致；冗余没有因此自动消失，是因为现在评测的是尚未经过本项目 RL 的
Qwen3.5 post-trained parent，而原版 prompt 本身只是软约束。** 本次 G1 中途停止的
直接原因也不是回答太长，而是 native action 边界的两个实现假设有误。

E0/G0 原注册门槛通过：direct HF 与 native manager 的 prompt digest、首 action
token 和 token-prefix integrity 均为 `16/16`；确定性环境 replay 完成一次真实
search、一次 retrieval、一个非空 tool response、3 篇文档，且 tool response 的
policy token 数为 0。完整轨迹复核又发现 3 个 reasoning/action 隔离漏测案例，因此
不能把“原门槛通过”扩大解释为 native adapter 已完全正确。

G1 不是科学 `NO-GO`，而是工程失败。第一批 `8 prompts x group 2 = 16`
条 rollout 已完成生成，但 trace writer 写入第二条记录时抛出：

```text
ValueError: raw_generations[2] answer boundary is inconsistent
```

因此只留下 1 条完整 `.partial` 记录，未生成正式 32 条 trace、manifest、
`summary.json` 或 `go_no_go.json`。该错误与 OOM、NCCL、BM25、双卡参数或
response=500 无关。外层运行 8 分 53 秒后以 exit code 1 结束，watchdog 已成功
下发关机；按 5.76 元/小时计算，本次约花费 0.85 元。

## 2. 当前 prompt 是否已经与原版一致

准确说法是：**项目自写的 user prompt 已恢复原版行为语义，但完整 rendered
prompt 不可能与上游 Qwen2.5 + legacy XML 输入逐字或逐 token 相同。** Qwen3.5 原生工具模板
必须加入 vendor system/tool schema、`<tool_call>`、tool role 和 generation
prefix；这些是模型接口序列化，不是新增搜索策略。

原版 user prompt 位于
[`scripts/data_process/nq_search.py`](../../../../scripts/data_process/nq_search.py)，当前合同位于
[`search_r1/llm_agent/tool_protocol.py`](../../../../search_r1/llm_agent/tool_protocol.py)。对当前
sealed 样本的实际 user 文本，相对原版只保留以下三项白名单差异：

| 原版 | 当前 Qwen3.5 | 性质 |
| --- | --- | --- |
| `<search> query </search>` | 原生 `search(query)` tool call | 必要接口映射 |
| `<information>...</information>` | 原生 tool response role | 必要接口映射 |
| `as many times as your want` | `as many times as you want` | 恢复论文中的正确拼写 |

函数级输入合同还有两项防御性处理：当前 adapter 会压缩问题内部空白，并拒绝问题中
直接出现保留协议 marker；原版只做首尾 `strip()`。当前 sealed 问题没有触发这两项，
因此不会形成模型可见差异，但文档不把它们伪装成逐函数完全一致。原版和当前都会为
缺少问号的问题补 `?`。

其余关键语义均保留：每次获得新信息后先思考；只有缺知识时才搜索；允许零次或多次
搜索；最终答案放入 `<answer>...</answer>`；`without detailed illustrations` 和
`<answer> Beijing </answer>` 示例仍在。当前也没有 `must search at least once`、
`at most four searches`、terminal 强制回答或 few-shot 搜索策略。

当前项目实际拥有的 user 文本模板如下，问题会接在最后：

```text
Answer the given question. You must conduct reasoning inside <think> and
</think> first every time you get new information. After reasoning, if you
find you lack some knowledge, you can call the available search tool with a
query, and it will return the top searched results in a tool response. You can
search as many times as you want. If you find no further external knowledge
needed, you can directly provide the answer inside <answer> and </answer>,
without detailed illustrations. For example, <answer> Beijing </answer>.
Question: {question}
```

此外，当前通过 `apply_chat_template(..., tools=..., enable_thinking=True)` 使用固定
Qwen3.5 原生模板。原项目没有 `enable_thinking` 这个 vendor 开关，但原版 prompt
明确要求 `<think>...</think>`；因此打开 Qwen3.5 thinking 是语义映射。数据仍只有
一条普通 user message，没有叠加项目自写的行为 system prompt。固定 vendor system
块允许工具调用前出现自然语言 reasoning，并在不调用工具时正常回答；它可能强化
Instruct 模型的解释/复述倾向，但属于不可省略的原生接口，不强制搜索。

所以“已经和原版一样”必须限定为：**行为指令等价，接口表示不同**。若比较模型最终
看到的完整 token 序列，答案必然是否定的；Qwen chat template 自动加入的 system、
tool schema、role wrapper 和 generation prefix 都会造成差异。

## 3. 为什么原版对齐后回答仍然冗余

这里必须把四件事分开，否则很容易误修：

| 现象 | 本次证据 | 当前判断 |
| --- | --- | --- |
| reasoning 较长 | 多轮 `<think>` 会解释证据和下一步 | 原版要求思考，且 parent 尚未用 RL 学会压缩 |
| 最终答案复述问题 | G1 唯一落盘答案为 8 词，gold 为 4 词 | 真实模型误差，strict EM 正确记 0 |
| 证据足够后仍搜索 | Ryan Neates 在第二搜已有答案，仍做第三搜 | 未训练策略的停止决策较弱 |
| action/trace 边界错误 | G0 有 3 个误解析，G1 写 trace 崩溃 | 独立工程问题，不能算模型能力结论 |

### 3.1 先区分 reasoning 长和最终答案冗余

唯一落盘的 G1 轨迹问题是：

```text
Ryan Neates is currently listed with a football club whose offical colours are what?
Gold: navy blue and gold
```

它先搜索人物并找到 Claremont，再搜索俱乐部并找到 `navy blue and gold`。第二搜已
完成必要二跳；模型随后又做了一次冗余确认，第四个 action 才输出：

```text
<answer> Are the official colours navy blue and gold </answer>
```

这里有两种不同的“长”：

1. `<think>` 内 reasoning 较长，是原版 prompt 明确要求的行为，不能简单删掉。
2. `<answer>` 内不是长篇解释，但把问题谓语一起复述了。它只有 8 个英文词，却比
   4 词 gold 多一层句式，因此 strict normalized EM 仍为 0。

但不能据这一条就断言当前 prompt 系统性地产生长 final answer。G0 的 16 条
native-manager 轨迹中共有 6 条合法 answer，其中 5 条只有 1-2 个词，只有 1 条是
10 词的问题复述。direct 与 manager 的首 action 又是 `16/16` token 完全一致，
manager 不是长回答的来源。现有证据更准确地说明：**答案槽位偶发不够精确，完整
轨迹则因 thinking、多轮 search 和检索结果而显得很长。**

还有一层容易被肉眼混淆：这 6 条合法 answer 中有 5 条在 `</think>` 与 `<answer>`
之间仍输出了解释性 prose，但其中 4 条的 answer 槽只有 1-2 个词。上游正则和当前
strict parser 都只抽取标签内文本，前置 prose 不参与 EM；因此“整段回复很长”和
“最终答案槽很长”不是同一个指标。前置 prose 暂不通过新 prompt 强行消除，后续门禁
分别统计 reasoning、pre-answer prose 和 answer 槽的 token 数。

### 3.2 prompt 是软约束，parent 尚未接受本项目 RL

当前 checkpoint 是固定 revision 的 post-trained `Qwen/Qwen3.5-2B`，不是
`Qwen3.5-2B-Base`，也不是已经完成 Search-R1 RL 的模型。本次 G1 是
`val_only=true` 的 0-update parent 评测，没有 optimizer step。后训练模型通常倾向
用完整自然语言复述问题；一句 `without detailed illustrations` 和一个 Beijing 示例
只能提供概率性引导，不能保证每次都抽取成 gold 的最短槽位。该句禁止详细解释，
却没有形式化规定只能输出与 gold 等长的名词短语，所以完整问句或完整句仍有非零
采样概率。

论文 `2503.09516v5.pdf` 第 9 页 Section 5.3 也观察到：训练前 100 steps 中 response
length 先明显下降，模型在这一阶段学习删除 `excessive filler words`。这说明原论文
同样没有假设“使用原版 prompt 后，初始模型天然就会简洁”。论文没有为 filler 单独
增加长度奖励，但训练动力学仍学会了压缩；标签内冗余还会通过 strict EM 直接失分。
论文 Section 5.2 也说明 Instruct 模型起点更好、收敛更快，但依然需要 RL。

### 3.3 当前采样和状态机有意保留探索

门禁使用论文侧的 `temperature=1.0`、`top_p=1.0` 和随机采样，而不是贪心解码。
总 action budget `B=4` 只提供上限，不要求模型取得证据后立即停止。当前 prompt
还明确允许模型“search as many times as you want”。因此出现“reasoning 已说信息
足够，但动作仍选择 search”的不一致，是未训练策略的真实基线，不是 prompt 漂移。

G0 的 16 条 native manager 轨迹进一步印证这一点：搜索次数分布为
`1:3, 2:2, 3:5, 4:6`，13/16 至少搜索两次；但只有 6/16 在 `B=4` 内给出合法
answer，10/16 用完预算仍未合法终止。现在的主要模型问题已经不是“不搜索”，而是
**过度搜索、停止不及时和答案槽位偶发不够简洁**。其中一部分 invalid 和后续 retry
由 adapter 误读 reasoning 中的字面标签造成，不能算到模型头上；第 4 节单独处理。

### 3.4 strict EM 正在按设计暴露问题

当前 parser 只抽取 `<answer>` 内文本，EM 不使用语义 judge，也不因为答案包含 gold
子串就放宽。本例的 post-hoc utility 为：

```text
当前：EM 0 - 0.10 * 3 / 4 = -0.075
若三搜后给出精确短答案：1 - 0.10 * 3 / 4 = 0.925
若二搜后立即精确回答：1 - 0.10 * 2 / 4 = 0.950
```

所以 `lambda=0.10` 下每次搜索只影响 0.025，正确性仍占主导。现在看到 EM=0
不是评分器太严格的偶然噪声，而是后续 RL 可以区分“语义接近但格式冗余”和“精确、
及时回答”的必要信号。

## 4. 两个边界根因

### 4.1 reasoning 和 action 没有真正隔离

Qwen 原生 template 已把 assistant 输出分成 reasoning 与后续 action：当前 sampled
continuation 通常先生成 reasoning，再出现 `</think>`，之后才是 tool call 或
`<answer>`。但 [`slice_first_complete_native_action()`](../../../../search_r1/llm_agent/generation.py)
和 [`_parse_qwen35_action()`](../../../../search_r1/llm_agent/tool_protocol.py) 仍在整段 decoded
文本中全局查找/计数 action marker。于是 reasoning 中只是谈论输出格式的文字，也会
被当成真正 action。

G0 的 56 个 assistant turn 中有 3 次 reasoning 提到 `<answer>`，3 次全部被误判：

| sample / turn | reasoning 中的文字 | `</think>` 后的真实 action | 当前结果 |
| --- | --- | --- | --- |
| `hotpotqa:train:83873`, slot 1, turn 3 | `in <answer> and </answer>.` | 完整 search call | 在 reasoning 内提前截断，`malformed_answer` |
| `hotpotqa:train:86102`, slot 1, turn 2 | `in <answer> format` | 完整 search call | 全局计数后报 `multiple_or_unbalanced_answers` |
| `hotpotqa:train:9765`, slot 0, turn 2 | `within <answer>` | `<answer>Passaic County</answer>` | 两个 opening 被当成同一区域，正确答案遭拒 |

三条都被误判为 invalid。Ryan 与 Blondie 两条在下一 action 进入 user-retry context，
且随后都执行了 search；Gran DT 已处于第 4 个也是最后一个 action，没有第 5 次
generation，但本应执行的 search 被抑制并污染了 invalid/终止统计。所以旧 G0 的
invalid、预算耗尽和搜索次数混入了 adapter 制造的成分，不能全部解释为模型“冗余
搜索”。reasoning 内引用标签本身不是环境 action，也不应受 action grammar 惩罚。

上游 prompt 要求先在 `<think>...</think>` 中推理，但论文 Algorithm 1 和 runnable
源码的全局正则/字符串 split 并不验证 thinking 已关闭。这里必须诚实披露：仅在
post-thinking action region 识别动作，是针对 Qwen vendor reasoning channel 新增的
必要隔离规则，不是 origin 全局 parser 的逐字复制。它不改变 reward、搜索策略或
action 集，但会改变 malformed 行为：若模型未关闭 thinking 就输出 call，origin
可能执行，新的 native adapter 会将其记为 invalid。这样才能避免把 reasoning 中的
格式讨论误执行为环境动作。

### 4.2 token 边界不等于字符边界

真实 sampled token prefix 必须原样保留，不能把 token 拆成字符，也不能 decode 后
重新 tokenize。某个 token 可能同时解码出 closing marker 和后续标点，例如：

```text
</answer>.
```

因此“包含首个完整 closing marker 的最短 token prefix”不一定在字符层面以 marker
结尾。当前 [`_validate_raw_generation()`](../../../../search_r1/trajectory_trace.py) 使用
`rstrip().endswith("</answer>")`，把字符末尾当成 token 边界的二次校验，最终让本应
落盘的 generation event 摧毁整轮 G1。

origin 会先在字符层截到 `</answer>`/`</search>`，再重新 tokenize，所以
`<answer>x</answer>.` 会按 `<answer>x</answer>` 执行。为同时保留这个逻辑和 Qwen
真实 sampled token，本轮把两个概念分开：

- **logical action**：action region 中从起点到最早 closing delimiter（含 delimiter）
  的字符片段，用于严格语法解析和环境执行；
- **policy action prefix**：首次让该 delimiter 完整出现的最短原 sampled token
  prefix，用于轨迹、mask、log-prob 和审计，不拆 token、不重新 tokenize。

因此一个合法 action 的最后一个 token 若额外解码出 `.`/`X`，逻辑 action 仍按 origin
接受，额外字符保留在 policy token/text 证据中但不参与执行；如果 delimiter 前本来
就是 close-only、缺 opening、嵌套或其他 malformed 内容，parser 仍判 invalid，trace
也必须照常落盘。这比“把所有 overshoot 一律判 invalid”更接近原版，不会仅因
tokenizer 分词方式改变 reward。

G1 失败行没有成功写入，无法恢复其原文，所以不能武断地说它一定是句点 overshoot
或一定是 reasoning marker；异常只能证明 `boundary="answer"` 与 `endswith` 假设
冲突。G0 的 Gran DT 例同时实证了 reasoning marker 和 `</answer>.` token overshoot，
足以为两类情况补回归。

这两个边界问题都与答案冗余独立。修好后不会自动让 parent 更简洁，只会保证模型
真正选择的 logical action 被正确执行、malformed action 被正确记录、门禁能够完整
结束。

## 5. 最小生产代码修复

本轮只引入一个概念：**reasoning-aware action region**。用一个共享的纯解析规则让
generation slicer、strict parser 和 trace validator 对同一边界达成一致，而不是在
三个位置各自猜测。

1. **显式分区模式**：生产路径固定传 `continuation`，因为 vendor template 已预填
   opening `<think>`；sampled 文本只允许恰好一个结束 reasoning 的 `</think>`，再次
   输出 `<think>` 视为嵌套并 invalid。独立 parser 测试若需要完整
   `<think>...</think>`，必须显式传 `full`，只接受单个平衡 pair。两种模式都拒绝
   未闭合、重复或嵌套 thinking marker，禁止 helper 靠启发式自动猜模式。
2. **delimiter 切片**：只在 action region 中寻找位置最早的完整 `</tool_call>` 或
   `</answer>` 字符串，不要求它已经构成语法合法 action；boundary 表示首个 closing
   delimiter 类型。policy action IDs 必须是让该 delimiter 首次完整出现的最短 raw
   sampled-token prefix，raw tail 继续单独审计。
3. **logical 解析**：parser 只解析 action region 到该 delimiter 为止的 logical
   action，reasoning 可以原样提到 `<answer>`，boundary token 解码出的后缀字符不参与
   grammar。delimiter 前仍使用严格规则，marker-free answer、缺 opening、嵌套或重复
   marker、错误工具名仍拒绝；首 delimiter 后的 raw tail 不执行。为保持现有/origin
   行为，`</think>` 后、opening action marker 前的 marker-free prose 继续允许并记入
   reasoning prefix。
4. **落盘校验**：validator 复用同一显式分区和首 delimiter 规则，确认声明的
   boundary 与 action region 中最早 close 类型一致；允许 boundary token 同时解码出
   marker 后字符，但不允许仅靠 reasoning 内 marker 通过。close-only/unbalanced
   action 仍能以对应 boundary 落盘，语法有效性由 parser 单独给出。

该最小修复会涉及
[`search_r1/llm_agent/generation.py`](../../../../search_r1/llm_agent/generation.py)、
[`search_r1/llm_agent/tool_protocol.py`](../../../../search_r1/llm_agent/tool_protocol.py) 和
[`search_r1/trajectory_trace.py`](../../../../search_r1/trajectory_trace.py)，并同步更新只读协议探针
的同一解析调用。最小原则按责任和行为变量判断，不以“只能改一个文件”判断；若只把
`endswith` 改成全局 `contains`，G1 虽可能落盘，三条已知误解析仍会污染后续训练。

以下内容冻结不改：

- 不从 policy token/text 中裁掉句点、不重新 tokenize、不修改 sampled IDs；
- 只在 logical action 层按 origin 截至首 delimiter；同 token overshoot 不改变有效性，
  delimiter 前的 malformed 内容仍不能获得 EM；
- 不修改 conversation 状态机、trace schema 或 observation mask；历史动作渲染只复用
  同一个 post-thinking 边界，避免再次从 reasoning 中误找 opening marker；
- 不修改 prompt、answer 示例、thinking、retry、reward、数据或训练参数；
- 不增加“只输出最短答案”“获得证据后立即停止”等额外提示。

## 6. 测试与 CPU 验收

### 6.1 聚焦回归

1. `tests/test_generation_qwen35_native.py`
   - 将三条真实 G0 文本固化为回归：reasoning 内完整 answer 标签 + search、单个
     opening answer 标签 + search、单个 opening 标签 + 真正 answer；结果分别必须是
     search/search/answer，且不触发 retry；
   - 增加 reasoning 内 `</tool_call>` 字面量的对称用例；
   - reasoning 内有 closing action marker、但合法 `</think>` 后没有完整 action 时，
     采样以 EOS 结束必须是 `boundary=eos`，触顶必须是 `boundary=length`；
   - action region 内真正的双 action 仍只执行首个完整 action，raw tail 保留；
   - 构造单 token 解码为 `</answer>.` 和 `</tool_call>X` 的 tokenizer，验证原 sampled
     token prefix 不被改写、logical action 截至 delimiter，合法 action 仍按 origin
     执行；delimiter 前 malformed 的对照组仍 invalid。
2. `tests/test_qwen35_tool_protocol.py`
   - production `continuation` 与 standalone `full` 必须显式选择，不能自动猜测；
   - production 中第二个 `<think>`、thinking 未关闭/重复/嵌套，以及 action region
     marker 缺失/嵌套/重复时仍 fail-closed；
   - `</think>` 后 marker-free prose + 合法 answer/search 继续接受并保留 prefix；
   - close-only/unbalanced marker 得到相应 delimiter boundary，但语法解析仍 invalid。
3. `tests/test_trajectory_trace.py`
   - answer/tool-call token overshoot 可以被 trace 忠实保存；
   - marker 只存在于 reasoning 时不能证明 action boundary；
   - 声明的 boundary 与 action region 最早 closing delimiter 类型不一致时拒绝；
   - close-only/unbalanced action 可以留档为 invalid，不因 validator 再次崩溃。
4. `tests/test_trainer_trajectory_logging.py`
   - 合法 logical action 的 overshoot event 可以落盘并保持原版执行语义；
   - malformed delimiter event 可以落盘且该 turn 仍为 invalid；
   - 三条 reasoning-marker 回归不再产生 parser-induced retry；
   - action count、retrieval、mask、extracted answer 和 EM 保持语义一致。

先运行：

```bash
python -m pytest -q \
  tests/test_generation_qwen35_native.py \
  tests/test_qwen35_tool_protocol.py \
  tests/test_trajectory_trace.py \
  tests/test_trainer_trajectory_logging.py
```

随后运行全量 pytest、相关 AutoDL shell 回归、Python compile 和 shell syntax 检查。

### 6.2 无卡实例增量封存

无需重新下载模型、语料或索引，也不重新选数据。代码提交后在无卡实例执行：

```bash
AUTODL_QWEN_NATIVE_INCREMENTAL=1 \
bash scripts/autodl/02_cpu_prepare.sh
```

新 CPU attempt 必须 `terminal=success`、`exit-code=0`，checkout commit/tree、handoff
和 `cpu.ok` 相互一致。prompt、Qwen template、模型、数据和训练配置 digest 应保持不变；
只有代码 commit/tree 与由其派生的 handoff digest 更新。无需重新下载模型、语料、
索引，也不重新选题或重跑 BM25。

## 7. GPU 只重跑门禁

action 解释已变化，不能把旧 G0 与新 G1 拼成同一证据链。下一次仍重跑短
`E0/G0/G1`，但不运行 2-step smoke 或 R60：

1. G0 必须生成 direct 16 + native manager 16，所有结构条件通过。
2. G1 必须生成正式 `eval_predictions.jsonl` 共 32 行，覆盖 16 个 sample、每题
   slots `0/1`。
3. 必须发布 trace manifest、sidecar、`summary.json`、`per_trajectory.jsonl`、
   `per_question.jsonl` 和 `go_no_go.json`，不得只剩 `.partial`。
4. reasoning 中引用 action 标签、随后给出合法 action 时，必须执行后者且不触发
   parser-induced retry。
5. action region 的 token overshoot 若再现，policy token 必须原样落盘；有效性只由
   delimiter 前的 logical action 决定，不能因 trace validator 再次终止作业。
6. 门禁后自动关机，再在无卡模式分析全部轨迹。

门禁结构条件仍只判断 token 不改写、action prefix、retrieval/tool-response 对齐和
observation mask。EM、合法答案率、搜索次数、冗余搜索和 thinking 长度全部报告，
但不为得到“好看结果”事后改门槛。

## 8. R60 前的人工决策材料

完整 G1 结束后至少汇总：

- 32 条轨迹的 `0/1/2/3/4` 搜索次数分布；
- 合法 answer、strict EM、invalid、clipping 和预算耗尽率；
- 必要二跳、冗余第三/第四搜、重复 query 和新文档覆盖；
- 语义正确但答案槽位冗余的案例；
- reasoning 中引用 `<answer>`、action region 格式错误和真实 retrieval 失败的分层；
- parser-induced invalid/retry 数、retry-context 后实际搜索数、clean trajectory 搜索
  分布，以及固定样本修复前后的描述性比较；不把旧总数简单相减成反事实结果；
- 代表性的正确、错误、单搜、有效多搜、冗余多搜完整轨迹。

只有用户看完这些材料后再决定是否启动 2-step smoke/R60。修复边界后仍可能得到
较低 parent EM 或较高 invalid 率；那是应被保留的模型基线，不再与工程崩溃混为一谈。

## 9. 最终决策

本轮采用以下最小单概念方案：

> 为 Qwen vendor reasoning channel 明确定义 `</think>` 后的 native action region，
> 让 slicer、parser 和 trace validator 共用同一最早 closing delimiter；logical action
> 按 origin 截至 delimiter 执行，policy action 保留不可拆分 token overshoot。保持
> 原版语义 prompt、delimiter 前 strict grammar、strict EM、B=4、
> response/observation=500 和全部采样参数不变。重跑完整门禁并人工分析 32 条轨迹，
> 在此之前不启动 R60。

这样既不会用新增提示词隐藏 parent 的冗余行为，也不会把 reasoning 中的普通文字
误执行成 action，更不会让一个本应被记录的 invalid action 再次摧毁整轮证据。
