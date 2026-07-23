# Qwen3.5-2B 工具协议与实验配置审计

## 1. 结论

当前异常的首要解释不是“Qwen3.5-2B 太笨”，而是 **模型原生工具协议、Search-R1 论文协议和本项目解析实现三者没有对齐**。证据强度分为三层：

1. **已由轨迹直接证明：提示词和解析器制造了大部分退化调用。** 312 次检索中，139 次 query 是字面量 `query`，70 次是 `and`，1 次为空，共 210/312（67.31%）。`query` 对应提示中的占位符；70/70 次 `and` 都来自模型复述的 `<search> and </search>`，被宽松正则误当成调用。
2. **结构上高度可疑：Qwen 的原生工具能力完全没有被激活。** 官方模板需要 system 中的 JSON tool schema、`<tool_call><function=...>` action、`<tool_response>` 回填和新的 chat role；当前代码只把 `<search>` 写在普通 user 文本里，并把 `<information>` 继续拼在同一个 assistant 流中。
3. **配置进一步放大不稳定性。** `temperature=1.0/top_p=1.0` 对 Qwen3.5 non-thinking 是正确的；但当前 `top_k=0` 等于关闭 top-k，HF rollout 也没有官方建议的 `presence_penalty=2.0`。这会增加低概率输出和重复循环，但不是占位符与 `and` 误调用的根因。

所以，现有结果只能说明“post-trained Qwen3.5-2B 在未经适配的 Search-R1 私有协议下不稳定”，不能说明它不会工具调用。下一步应先做一个很小的协议隔离实验，不应直接换模型、增加 group 或重跑完整训练。

## 2. 审计对象与证据

- 实际模型：`Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc`，官方元数据的 base model 是独立的 `Qwen/Qwen3.5-2B-Base`。
- 固定 revision 的 `chat_template.jinja` SHA-256：`273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80`。
- 固定 revision 的 `tokenizer_config.json` SHA-256：`49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c`。
- 实验配置：`eval/resolved-config.yaml`；完整证据为 320 条 `eval/traces/eval_predictions.jsonl`。
- 论文对照：`docs/2503.09516v5.pdf` 第 5、6、9、16 页。

统计均从归档原始 trace 重新计算，没有修改 GO/NO-GO 规则，也没有运行新模型推理。

## 3. Qwen3.5 原生工具调用到底是什么格式

固定 tokenizer 模板只有在 `apply_chat_template(..., tools=[...])` 收到 tool schema 时，才会在 system 消息中注入类似内容：

```text
<|im_start|>system
# Tools

You have access to the following functions:

<tools>
{"type":"function","function":{"name":"search",...,"parameters":...}}
</tools>

If you choose to call a function ONLY reply ...
<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
</function>
</tool_call>
<|im_end|>
```

对于本项目的搜索函数，模型预期生成的 action 是：

```text
<tool_call>
<function=search>
<parameter=query>
Ryan Neates football club
</parameter>
</function>
</tool_call>
```

工具结果不是直接接在 assistant 文本后，而是序列化为新的 user/tool-response 回合：

```text
<|im_end|>
<|im_start|>user
<tool_response>
Ryan Neates is listed with Claremont Football Club.
</tool_response><|im_end|>
<|im_start|>assistant
<think>

</think>
```

官方 model card 对 vLLM/SGLang 要求 `--tool-call-parser qwen3_coder`。这个 parser 主要负责把上述原始标签转换为结构化 API tool call；直接使用 HF `generate()` 也可以支持原生协议，但必须手工传入 `tools`、解析 `<tool_call>`，并按模板回填 tool response。单纯把 rollout 从 HF 换成 vLLM 并不会自动修复当前 prompt。

Qwen3.5-2B 默认是 **non-thinking**。未传 `enable_thinking=True` 时，chat template 会预先写入空的 `<think>\n\n</think>`，然后模型才开始输出正文。当前 raw trajectory 中正是这个序列。

## 4. 当前 Search-R1 实际喂给模型的格式

本项目的真实输入近似如下：

```text
<|im_start|>user
... call a search engine by <search> query </search> ...
<|im_end|>
<|im_start|>assistant
<think>

</think>
模型正文 <search>Query: "Ryan Neates football club"</search>
<information>检索文档...</information>
模型继续正文 <search>Query: "Claremont Football Club official colours"</search>
<information>检索文档...</information>
模型继续正文 <answer>Navy blue and gold</answer>
```

成功样例 `hotpotqa:train:86102` 的整条 7,320 字符轨迹只有 1 个 user role、1 个 assistant role 和 1 个 `<|im_end|>`；两次检索结果都在同一个 assistant 续写流里，没有 `<tool_response>` 或新的 assistant generation prompt。这忠实于 Search-R1 论文“把 `<information>` append 到 ongoing rollout”的设计，却不是 Qwen3.5 工具后训练时使用的多回合模板。

| 环节 | Qwen3.5 原生协议 | 当前实现 |
| --- | --- | --- |
| 工具声明 | system 中 JSON schema | user 文本一句说明 |
| 调用标签 | `<tool_call><function=search><parameter=query>` | `<search>...</search>` |
| 结果回填 | 新 user 回合中的 `<tool_response>` | 同一 assistant 流中的 `<information>` |
| 继续生成 | 新 `<|im_start|>assistant` | 在旧 assistant 序列后续写 |
| 原生 parser | `qwen3_coder` 或等价手写 parser | 正则寻找 `<search|answer>` |
| thinking | 默认模板先生成空 think block | prompt 又要求必须在 think 内推理 |

`verl/utils/dataset/rl_dataset.py:129` 调用 `apply_chat_template(chat, add_generation_prompt=True)` 时没有传 `tools` 或 `enable_thinking`。因此模型虽然是经过 Agent/tool-use 后训练的 checkpoint，本轮却没有给它熟悉的工具 schema 和调用语法。

## 5. 轨迹说明模型到底会不会搜索

### 第一轮动作

| 第一轮结果 | 轨迹数 | 占 320 条 |
| --- | ---: | ---: |
| 非退化 search query（不保证相关） | 61 | 19.06% |
| 字面量 `query` | 64 | 20.00% |
| 空 search query | 1 | 0.31% |
| 直接 answer | 105 | 32.81% |
| 非法动作 | 89 | 27.81% |

模型第一轮实际触发了 126 次搜索；其中 61/126（48.41%）至少不是占位符或空 query，64/126（50.79%）复制了 prompt 占位符。这里的“非退化”只表示有具体文本，不保证与问题相关。这仍然不是“完全不会调用”，而是可解析行为和错误模板模仿几乎各占一半。prompt 又明确允许“认为不需要知识时直接回答”，所以 105 条直接 answer 也不能全部算工具失败；训练前强制要求自然多搜，本身与这个可选搜索提示存在张力。

### 全部检索调用

| Query 类型 | 调用数 | 占 312 次 |
| --- | ---: | ---: |
| 字面量 `query` | 139 | 44.55% |
| 字面量 `and` | 70 | 22.44% |
| 空字符串 | 1 | 0.32% |
| 其他有内容 query | 102 | 32.69% |

- `query` 涉及 82 条轨迹，其中 64 次发生在第一轮。`hotpotqa:train:83873` 在语法完全合法的情况下连续四搜 `query`，BM25 返回 Query Language 文档后形成无关反馈循环。
- `and` 涉及 56 条轨迹，0 次发生在第一轮。56/56 条在第一次 `and` 前都有非法动作；70/70 个生成文本都包含 `<search> and </search>`。`generation.py:523` 的非锚定正则把说明句中两个标签之间的连词抽成 query。
- 当前恢复文案比论文 Algorithm 1 更危险。论文只追加 “My action is not correct. Let me rethink.”；本实现重复展示完整 opening/closing 标签，正好给宽松 parser 制造可执行假 action。
- 2 条轨迹、3 个 generation event 还出现了 `<tool_call>`，但没有 `<function=...>` 或 `<tool_response>`。这更像 Qwen 原生工具习惯在无 schema 情况下的泄漏；当前 parser 无法消费它们。
- 897 个 generation event 中有 21 个同时包含完整 search 和 answer，22 个包含多个 action pair。`_postprocess_responses()` 优先检查任意 `</search>`，而 `postprocess_predictions()` 又取第一个 action，混合输出的处理语义并不稳固。

同时，`hotpotqa:train:86102` 已经成功完成“Ryan Neates -> Claremont Football Club -> official colours”两跳链。这条证据足以否定“模型根本理解不了搜索”的解释，但不能证明当前协议已稳定。

## 6. 哪些实验配置真的有影响

| 配置 | 本轮实际值 | Qwen3.5 / 论文参照 | 判断 |
| --- | --- | --- | --- |
| checkpoint | post-trained Qwen3.5-2B | 正确，不是 raw Base | 保留 |
| 工具 schema | 未传 `tools` | Qwen 原生工具必须传 schema | **高影响** |
| tool result role | 同一 assistant 中 `<information>` | Qwen 原生为 user `<tool_response>`；论文为 ongoing rollout | **高影响、协议取舍** |
| thinking | 默认 non-thinking，但 user 要求 `<think>` 推理 | 2B 默认 non-thinking；thinking 模式易循环 | **明显冲突** |
| temperature | 1.0 | Qwen non-thinking=1.0；论文=1.0 | 正确 |
| top-p | 1.0 | Qwen non-thinking=1.0；论文=1.0 | 正确 |
| top-k | 0，即不截断词表 | Qwen non-thinking 建议 20 | **可能放大噪声** |
| presence penalty | 未实现，等价于无 | Qwen non-thinking 建议 2.0 | **可能放大重复** |
| response/turn | 500 | 论文=500 | 足够生成 action；继续加长不是首修 |
| observation | 384 | 论文 retrieved content=500 | 可能损伤第二跳，不解释首轮占位符 |
| prompt cap | 4096 | 论文=4096 | 正确 |
| max turns / top-k docs | 4 / 3 | 论文=4 / 3 | 正确 |
| group size | 5 | 论文 GRPO=5 | 不影响单条格式能力 |
| warmup 0.285 | probe 不训练 | 只影响训练优化器 | 与本轮工具失败无关 |
| 两卡、batch 8、全参数 | 推理阶段 | 主要影响吞吐/训练显存 | 与首轮调用格式无关 |

`top_p=1.0` 不是错误，也无需因为本次失败改回 0.95：官方对 Qwen3.5 non-thinking text 正是推荐 `temperature=1.0/top_p=1.0`。真正的采样偏差是 `top_k=0` 与没有 presence penalty。当前 HF rollout 只构造 temperature、top-p、top-k 三项 `GenerationConfig`，并不支持 presence penalty。

也没有证据表明 checkpoint、tokenizer、bf16、SDPA 或双卡 padding 把模型“加载坏了”：下载 revision 与官方当前 SHA 一致，raw prompt 中能看到该 revision 的空 think 模板，模型能生成连贯答案、具体 query 和两条有效二搜。若底层权重或 tokenizer 错配，通常不会稳定地只退化成与提示占位符完全一致的 `query`，也不会出现由恢复文本精确触发的 `and`。

102 条轨迹发生截断，其中 38 条第一轮就触顶。对一个正常工具 action 来说 500 token 已经很多；触顶说明模型没有及时结束为合法 action，而不是搜索 query 需要更长。盲目把 response 再加长可能只会延长错误续写。

## 7. 因果优先级

1. **P0 - 已证实的实现问题：** 占位符、恢复文案、非锚定 parser、空 query 仍执行。这些问题不需要新 GPU 实验即可确认，且直接覆盖 67.31% 的检索调用。
2. **P0 - 待 A/B 定量的协议错配：** 未注册原生工具、不同 action tag、不同 tool-result role。结构证据很强，但当前没有“同模型同题只换协议”的对照，不能声称它单独解释了多少 EM。
3. **P1 - 模型模式与采样：** non-thinking 模板和 `<think>` 指令矛盾；top-k 与 presence penalty 未按 Qwen 建议。它们更可能放大 rambling、重复和非法动作。
4. **P2 - 多跳容量：** observation 384、2B 参数规模、仅计划 60 steps 会影响第二跳和训练收敛，但无法解释第一轮复制 `query` 或把 `and` 当工具调用。

因此，“换更大模型”不应是第一动作。协议问题若不修，4B/8B 也可能更流畅地重复错误格式；反过来，如果协议隔离实验显著改善，2B 仍然适合这个低成本简历项目。

## 8. 最小验证方案

### 第一步：只修 Search-R1 XML，不重写为原生工具

建立新 commit，保持 `<search>/<information>/<answer>` 主结构，只做以下最小改动：

1. prompt 删除 `query` 字面占位符和 `as your want`，给出一个具体、无关题目的有效 search 示例；默认 non-thinking 时不再要求模型在 `<think>` 内输出推理。
2. 恢复文案不出现成对 action 标签，只要求“按初始 action 格式重试”。
3. parser 只接受独立行、位于输出结尾的一个 action；拒绝空、`query`、`and`，混合 search/answer 时明确判 invalid，而不是取第一个。
4. 保持 `temperature=1.0/top_p=1.0`，把已支持的 `top_k` 从 0 改为 20。暂不为 presence penalty 扩大 HF rollout 改动。

先跑“机械调用 probe”：固定 16 题、每题 2 条，明确要求至少调用一次搜索，只测格式和 query，不测自主搜索决策。按本轮吞吐线性估算约 8-10 分钟、约 0.8 元。预注册建议为：合法首 action >=95%、非退化 query >=90%、退化 query <=2%、首轮截断 <=5%；query 相关性另做人工抽样，不能仅靠非空判定。

机械调用通过后，再用原来的可选搜索提示跑 32 题 x 3 条自主 probe，约 25 分钟、约 2.4 元；重点比较首轮非退化 search、人工 query 相关性、非法动作、截断、重复 query 和 EM，不在训练前继续要求已经稳定正确多搜。

### 第二步：只有 XML 适配仍失败，才试 Qwen 原生协议

原生分支可把 Agent action 映射为两个函数：

```text
search(query: string)
finish(answer: string)
```

它需要同步修改 tool schema 注入、`<tool_call>` parser、`<tool_response>` role 回填、answer 抽取、tool-response loss mask、轨迹 logger 和测试，改动明显大于第一步。如果原生协议显著优于适配后的 XML，应把实验命名为“Qwen3.5-adapted Search-R1”，而不是声称 action format 与论文完全一致。

### 决策规则

- XML 适配已达到格式门槛：保留论文协议，进入能力训练，避免过度设计。
- XML 仍差、原生协议明显改善：采用 Qwen 原生 action adapter。
- 两者都差：再考虑 Qwen3.5 更大尺寸或小规模格式 SFT；此时才有证据把问题归到模型容量。

## 9. 最终判断

用户的直觉基本正确：**Qwen3.5-2B 不应因为这次 probe 被判定为“不会搜索”**。当前实验没有使用它训练过的工具 schema/role/template，又用论文占位符和本地恢复正则制造了强烈错误反馈。现有两条有效二搜、61 条首轮非退化搜索和历史 NQ RL 提升都说明模型具有可利用的 Agent 起点。

但也不能反向断言“换成原生格式必然训练成功”。最严谨的结论是：协议错配是当前最高优先级假设，且已有大量直接证据；用不超过几元的小型配对 probe 即可把它与模型容量、数据和 RL 规模分离，再决定是否值得启动下一轮训练。

## 10. 证据定位

- `scripts/autodl/02_cpu_prepare.sh:215`：模型 repo 和固定 revision
- `verl/utils/dataset/rl_dataset.py:128`：chat template 调用未传 tools/thinking kwargs
- `search_r1/llm_agent/generation.py:55`：action 后处理顺序
- `search_r1/llm_agent/generation.py:99`：同一 rolling sequence 拼接 response 与 observation
- `search_r1/llm_agent/generation.py:494`：含成对标签的恢复文案
- `search_r1/llm_agent/generation.py:523`：非锚定 action 正则
- `verl/workers/rollout/hf_rollout.py:67`：实际只支持 temperature/top-p/top-k
- `eval/resolved-config.yaml`：本轮最终生效配置
- `eval/traces/eval_predictions.jsonl`：320 条原始轨迹
- [`../../2503.09516v5.pdf`](../../2503.09516v5.pdf)：论文 Search-R1 协议与训练设置
- [固定 Qwen3.5-2B chat template](https://huggingface.co/Qwen/Qwen3.5-2B/blob/15852e8c16360a2fea060d615a32b45270f8a8fc/chat_template.jinja)
- [固定 Qwen3.5-2B model card](https://huggingface.co/Qwen/Qwen3.5-2B/blob/15852e8c16360a2fea060d615a32b45270f8a8fc/README.md)
