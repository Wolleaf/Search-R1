# Qwen3.5 Native G2 问题清单与训练解阻计划

> 记录日期：2026-07-24
>
> 审计基线：`experiment/hotpot-search-gate@b0563c287b3b8a1203879276bf05af4e7d06fc5b`
>
> 当前结论：方向仍然值得继续，但当前提交不能直接启动 native GRPO 训练。

## 1. 结论

本轮不是 OOM、CUDA 或检索服务故障。G0/G1/G2 均完成了工程执行，Qwen3.5-2B 已能稳定使用原生 `search` 工具，并且 G2 的 96 条轨迹中有 51 条实际执行了至少两次搜索。当前阻塞已经收敛到三个局部接口：最终短答案合同、少量 parser 边界，以及训练采样与 PPO log-prob 的一致性。

因此，“模型不会搜索”这一假设已基本排除；但“已有可直接用于 strict-EM GRPO 的自动奖励信号”尚未成立。现在没有 native 训练 checkpoint，下一步是解阻后启动新训练，而不是续跑一个已开始的训练。

### 1.1 复现原则：论文合同优先

后续修改采用“论文明确内容优先、官方仓库补实现细节、所有例外单独登记”的原则。论文明确给出的科学语义和超参数直接保留；论文没有展开的代码细节才以官方仓库为准。若论文、官方 recipe 和当前预算三者冲突，不再用“原版”一词合并，而是在下表逐项说明取舍。只有 Qwen3.5 接口、两卡/100 GB/付费预算和成本感知研究问题要求的差异才允许修改；已存在但不准备修补的上游差异也必须明示。

| 项目 | 论文 | 官方仓库 | 本项目 | 结论 |
| --- | --- | --- | --- | --- |
| 搜索/回填 | `<search>` / `<information>` | 同论文 | Qwen 原生 `<tool_call>` / `<tool_response>` | Qwen3.5 必要适配 |
| 最终答案 | `<answer>answer text</answer>` | 同论文 | native v2 恢复同一标签 | 保留论文合同 |
| R/B reward | 抽取答案后 EM；无 format reward/neural RM | 同论文 | 原样保留 | 不改目标函数 |
| C reward | 无 | 无 | correctness-gated 搜索成本 | 唯一算法研究变量 |
| RL 与采样 | 默认 PPO，也报告 GRPO；GRPO group 5、temperature/top-p 1.0 | 提供 GRPO recipe，个别默认值与论文有差异 | 采用论文 GRPO 值；额外采样处理关闭 | 低成本论文变体 |
| 检索器 | Wiki-2018 + E5 dense、top-3 | 默认 E5/FAISS、top-3 | 同语料 BM25、top-3 | 100 GB/预算偏差，影响高 |
| action budget | `B=4` 个总 action | `max_turns` 个可检索 turn 后另有 terminal；不同 recipe 出现过 2/4 | 保留上游循环，固定 `max_turns=4` | 已登记语义差异 |
| 长度 | response/retrieved 500，total sequence 4096 | 以 `max_turns` 组合容量 | response 500、observation 384 | 当前循环的容量适配 |
| rollout 后端 | vLLM | vLLM/FSDP 路径 | HF + SDPA/FSDP | 冻结已验证工程链路 |
| 模型与规模 | Qwen2.5-3B/7B，8×H100、500 steps、batch 512 | recipe 的步数/turn 值并非全部与论文一致 | Qwen3.5-2B、2×5090、R60/B20/C20、batch 8 | 模型与预算缩放 |
| 数据 | 完整 NQ + HotpotQA train | 提供合并处理脚本 | train-512/val-128 与固定配比 | 预算及成本对照缩放 |

`enable_thinking=False` 作为已经通过 G0-G2 的固定模板选择保留，不在修答案边界时同时改变 thinking 模式；它不是模型或显卡的硬限制。实际 prompt 不照抄论文/仓库答案示例中的 `xxx` 或 `Beijing`，因为既有轨迹已证明该模型会复制字面占位符；这项删除只消除已观测的接口干扰，不改变 `<answer>` 合同。

384-token observation 是官方代码“4 个可检索 turn + terminal generation”路径下的显式容量偏差：policy right side 上界为 `4*(500+384)+500=4036`；若 observation 也取 500，则为 4500。native rolling state 还可保留最多 1024-token initial left side，因此这里的 4036 不能冒充论文的 4096-token total-sequence 口径。除上表和第 9.2 节登记的项目外，不新增答案语法、宽松 reward、额外工具、模型 sweep、参数 sweep 或新的训练阶段。任何后续偏离必须先单独说明必要性和影响，不能在排错时顺手加入。

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

## 3. 当前 prompt、答案抽取与 EM 合同

### 3.1 G2 实际测的是什么

G2 使用固定 revision `Qwen/Qwen3.5-2B@15852e8c16360a2fea060d615a32b45270f8a8fc`。这是已经后训练的 Qwen3.5-2B；纯 base 是另一个模型 ID `Qwen/Qwen3.5-2B-Base`。本轮是 `probe_autonomous_32`：32 道题、每题采样 3 条，共 96 条轨迹，`force_search=False`，且没有进行任何 RL 参数更新。因此，G2 只能观察初始模型在当前接口下的行为，不能把它描述成“训练后的模型”。

### 3.2 当前自定义 prompt 与工具 schema

当前代码在 [`tool_protocol.py`](../../../../search_r1/llm_agent/tool_protocol.py) 中构造两条消息。G2 的精确自定义内容如下；system 的视觉换行仅用于文档排版，代码中是拼接后的一行字符串：

```text
system:
Call at most one tool per assistant turn. Use search when external
evidence is needed. After each search result, decide whether another
search is needed. Use at most four searches. When you have enough
evidence, answer with only the short final answer, without a tool call
or explanation.

user:
Question: <question>
```

实际 user 字符串在问题后保留一个尾随 `\n`。只有 G1 强制搜索探针才会在 user 问题前增加 `Call search at least once before answering.`；G2 没有这句话。注册给模型的唯一工具为：

```json
[
  {
    "type": "function",
    "function": {
      "name": "search",
      "description": "Search the external knowledge base for evidence needed to answer the question. Call exactly one search per assistant turn.",
      "parameters": {
        "type": "object",
        "properties": {
          "query": {
            "type": "string",
            "description": "A specific factual search query."
          }
        },
        "required": ["query"],
        "additionalProperties": false
      }
    }
  }
]
```

数据加载和 rollout 都通过同一个固定模板渲染：

```python
tokenizer.apply_chat_template(
    messages,
    tools=qwen35_tools(),
    enable_thinking=False,
    add_generation_prompt=True,
    tokenize=False,
)
```

固定 `chat_template.jinja` 的 SHA-256 为 `273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80`。模型真正看到的结构近似如下；`# Tools`、调用格式和 `<IMPORTANT>` 段由 Qwen 模板注入，不是本项目手写 prompt：

```text
<|im_start|>system
# Tools

You have access to the following functions:

<tools>
{"type":"function","function":{"name":"search", ...}}
</tools>

If you choose to call a function ONLY reply in the following format
with NO suffix:
<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
</function>
</tool_call>

<IMPORTANT>
...
- If there is no function call available, answer the question like
  normal with your current knowledge and do not tell the user about
  function calls
</IMPORTANT>

Call at most one tool per assistant turn. ...
<|im_end|>
<|im_start|>user
Question: <question>
<|im_end|>
<|im_start|>assistant
<think>

</think>

```

`enable_thinking=False` 只让模板预填一个空 thinking 块，不会限制后续正文长度。搜索动作必须使用原生格式：

```text
<tool_call>
<function=search>
<parameter=query>
Ryan Neates football club
</parameter>
</function>
</tool_call>
```

代码内部把检索结果追加为 `role="tool"`，固定 Qwen 模板再把它渲染为新的 user `<tool_response>...</tool_response>` 回合，随后打开新的 assistant 回合。最终答案则不是工具调用，也没有边界标签，只是普通 assistant 文本。

### 3.3 为什么模型会给长回答

当前 system 已经写了 `only the short final answer`，所以根因不是“忘了要求短答案”，而是软指令、模型先验和解析合同叠加：

1. `short final answer` 只是自然语言要求，没有像 `<answer>...</answer>` 一样可机器校验的答案边界。
2. Qwen 模板同时要求在不调用工具时 `answer the question like normal`；后训练 chat 模型通常倾向完整句、Markdown 和解释，本轮轨迹也呈现这一特征。
3. 工具 schema 只约束 `search(query)`，不约束普通 assistant 终局的字段、行数或 token 数。
4. `enable_thinking=False` 关闭的是显式 thinking 模式，不是 verbose 模式；`max_response_length=500` 只是安全上限，也不是“生成 500 token”的目标。
5. 除疑似 JSON tool call 外，native parser 会把普通的不含协议 marker 的非空回复整段接收为答案；final generation 又只在 EOS 或长度上限停止，模型即使多写一句，接口也不会拒绝。
6. G2 是 0-update 门禁，模型尚未收到 strict EM 的训练反馈。论文第 5.3 节在 Qwen2.5-7B-Base 上观察到前 100 steps 先删除 `excessive filler words`，之后随着有效搜索增多，总轨迹长度再上升；这是与当前解释相容的训练动态证据，不是对 Qwen3.5-2B 长回答成因的直接证明。

因此应区分“总轨迹长”和“最终答案长”：论文允许 reasoning、搜索结果和多轮轨迹很长，但只对一个很窄的最终答案字段评分。`top_k=20` 和 `presence_penalty=2.0` 可能影响措辞与多样性，但没有证据表明它们是长回答的主因。

### 3.4 当前 native 的解析与奖励链

当前链路是：

```text
普通 assistant 终局
  -> _parse_qwen35_action(): 整段文本作为 ParsedAction("answer", ...)
  -> generation.py: 整段写入 final_answer
  -> main_ppo.py: qa_em.em_check(final_answer, gold)
  -> reward 写到最后一个 response token
```

[`qa_em.py`](../../../../verl/utils/reward_score/qa_em.py) 会转小写、去标点、去英语冠词并压缩空白，但最后仍要求两个完整规范化字符串相等。以已逐字确认的 G2 终局为例：

```text
sample_id:  hotpotqa:train:86102
question:   Ryan Neates is currently listed with a football club whose
            offical colours are what?
gold:       navy blue and gold
prediction: The official colours of the Claremont Football Club are
            **navy blue and gold**.

normalized prediction:
official colours of claremont football club are navy blue and gold

normalized gold:
navy blue and gold

strict EM: 0
```

Markdown 星号和句号会被去掉，但额外句子不会，所以内容正确仍判 0。G2 中 `2,526`、`1980` 被包在说明句里的样本属于同类；`Piers Sellers`/`Piers John Sellers` 和日期顺序差异则是另一类 alias 问题，单靠答案边界不能解决。

## 4. 论文原版 prompt、循环与 EM

### 4.1 论文的精确 prompt

论文 [`2503.09516v5.pdf`](../../../reference/2503.09516v5.pdf) Table 1（PDF 第 5 页）给出的是一个完整模板，没有公开 system/user role 或 chat-template 拆分：

```text
Answer the given question. You must conduct reasoning inside <think> and </think>
first every time you get new information. After reasoning, if you find you lack some
knowledge, you can call a search engine by <search> query </search>, and it will
return the top searched results between <information> and </information>. You
can search as many times as you want. If you find no further external knowledge
needed, you can directly provide the answer inside <answer> and </answer> without
detailed illustrations. For example, <answer> xxx </answer>. Question: question.
```

这里的关键不是它比当前 prompt 更强调“短”，而是它同时给出了机器可解析的 `<answer>` 边界和短答案示例。文案虽说可搜索任意次数，Algorithm 1 实际受最大 action budget `B` 约束；Appendix B.2（PDF 第 15-16 页）明确使用 Wiki-2018 + E5 dense retriever、top-3 passages、`B=4`、最大 response 500 tokens、retrieved content 500 tokens 和总序列 4096 tokens。当前 BM25 不能再被写成论文原值。

仓库原版 prompt 可直接在 [`qa_search_train_merge.py`](../../../../scripts/data_process/qa_search_train_merge.py) 和 [`search_mix.py`](../../../../scripts/data_process/search_mix.py) 的 `make_prefix()` 中核对。legacy 模板与论文合同相同，但不是逐字副本：它把示例改成 `<answer> Beijing </answer>`，并保留了 `as your want` 等小拼写差异。本文因此分别引用论文原文和仓库真实轨迹，不把两者混称为同一字符串；此前“本地缺失”的对象仅指本轮 native G2 原始 JSONL，不是原版 prompt。

### 4.2 原版 rollout 和奖励怎样工作

论文 Algorithm 1（PDF 第 6 页）的循环为：

```text
<think>...</think><search>query</search>
  -> 环境检索
  -> 追加 <information>...</information>
  -> 模型继续生成下一 action
  -> <answer>short answer</answer> 后终止
```

每个 action 在生成到 `</search>`、`</answer>` 或 EOS 时停止；非法 action 会追加 `My action is not correct. Let me rethink.`。训练只有最终结果规则奖励：

```text
r_phi(x, y) = EM(a_pred, a_gold)
```

其中 `a_pred` 是从完整 response `y` 中抽出的最终答案，不是整段 reasoning、search 和 information。论文明确不加 format reward，也不训练 neural reward model。论文只声明 exact string matching/EM，没有给出大小写、冠词、标点等归一化细则；这些细节来自仓库代码，不能冒充论文原文。

论文把 `a_pred` 定义为从 response 中抽取的最终答案。仓库 legacy 路径中的 [`qa_em.extract_solution()`](../../../../verl/utils/reward_score/qa_em.py) 会找到全部 `<answer>...</answer>` 并取最后一个；之所以要求至少两个 match，是因为完整 prompt 自带一个 `<answer> Beijing </answer>` 示例，生成答案是第二个。抽取完成后才调用同一个 `em_check()`，所以该实现没有对“整段自然语言回复”做 EM。

需要区分论文伪代码、官方控制流与本项目配置。Algorithm 1 在 `while b < B` 内生成 action，search、answer 和 invalid 都占一次预算，`B=4` 后没有额外生成；官方初始提交的 [`generation.py`](../../../../search_r1/llm_agent/generation.py) 则先运行 `max_turns` 个可检索 turn，对仍未终止的样本再给一个 `do_search=False` 的 terminal generation。官方顶层 recipe 使用过 `max_turns=2`，版本化 recipe 也有 `max_turns=4`；本项目明确注册的是 4。因此当前语义是“最多 4 次真实检索，必要时再生成第 5 个收尾 action”，不是论文 `B=4` 的字面同义词。本项目保留上游控制流和已注册值，不把预算语义差异误算成 Qwen3.5 改动；terminal 中再次生成 search 不执行检索、不计成本，且没有合法终局答案时 EM 为 0。

论文还同时研究 Qwen2.5-3B/7B 的 Base 与 Instruct：Instruct 初始表现更高、收敛更快，但两者都接受 RL 训练，最终表现接近。Appendix B.2 的 GRPO 参考配置是每 prompt 采样 5 条、500 steps、总 batch 512、8 张 H100、`temperature=1.0`、`top_p=1.0`。这与本项目的预算缩小版不能直接横比，也说明论文里的稳定多搜轨迹主要是训练结果，而不是要求未训练 parent 天然具备。

### 4.3 三种合同的直接对照

| 环节 | 论文/legacy XML | 当前 Qwen native v1 | 计划 native v2 |
| --- | --- | --- | --- |
| 搜索动作 | `<search>query</search>` | Qwen `<tool_call><function=search>...` | 保持 Qwen 原生 tool call |
| 检索回填 | `<information>...</information>` | `<tool_response>...</tool_response>` 新回合 | 保持 native 回填 |
| 最终答案 | `<answer>answer text</answer>`，示例/案例通常很短 | 无标记自由文本 | 恢复严格 `<answer>answer text</answer>` |
| 答案抽取 | reasoning 后取最终 `<answer>` 内容 | 合法 marker-free 终局的整段文本 | 允许安全 reasoning prefix；只取唯一且位于末尾的 `<answer>` 内容 |
| 正确性奖励 | 抽取后 strict EM | 该整段文本 strict EM；invalid 为 0 | 抽取后 strict EM |
| rollout budget | 论文 `B=4` 总 action；官方代码为 `max_turns` 个可检索 turn + terminal | 沿用官方代码和本项目已注册的 `max_turns=4` | v2 不再改预算状态机 |
| 对搜索/RL 核心的影响 | 原论文/仓库基线 | 已完成 Qwen 适配 | 恢复论文终局抽取合同，并加 current-turn fail-closed 边界 |

因此 native v2 采用混合但边界清晰的最小适配：搜索交互使用 Qwen 原生协议，最终答案继续使用 Search-R1 原版 `<answer>` 合同。它不是放宽 EM，也不是为结果“改评分”，更不引入新的答案语法。

## 5. 可复核的完整轨迹举例

### 5.1 本轮 native G2 的证据边界

本轮 native G2 原始文件仍在 AutoDL 持久盘：

```text
/root/autodl-tmp/search-r1/runs/eval/qwen_native_g2/attempts/
20260724T041309Z-1728-2268/traces/eval_predictions.jsonl

SHA-256:
6da6f333b3c97a92d34b8b377bde4228ccf26827cfd513785c4d6f210b8f4755
```

实现阶段已从实例校验两个原始 `evidence.sha256`、outer attempt 的 `success/0` 终态及上述 trace 哈希，并将完整 G0-G2 证据归档到 [`results/qwen35-native-gates-v1-20260724/`](../../../results/qwen35-native-gates-v1-20260724/)。该归档保留 v1 的原始数据、配置、完整轨迹、分析结果和血缘，不用 v2 parser 追溯重写旧结论。逐轨迹人工 ledger 仍可作为并行分析产物生成，但不阻塞 CPU handoff。

已经完成审计、可用于下次从原件快速定位的样本包括：

| Sample ID | Gold | 已核实的诊断用途 |
| --- | --- | --- |
| `hotpotqa:train:86102` | `navy blue and gold` | 逐字终局见 3.4；内容正确但整句 strict EM 为 0 |
| `hotpotqa:train:37266` | `2,526` | 三条终局均包含正确数字，仍因终局边界/格式失分 |
| `hotpotqa:train:23105` | `1980` | slot 1、2 包含正确年份，可检查 mixed group |
| `hotpotqa:train:47507` | `Piers John Sellers` | `Piers Sellers` 与全名的 alias 差异 |
| `hotpotqa:train:25811` | `21 August 1765` | substantive answer 前回显 empty-think，触发 parser 误判 |

这些定位信息只帮助找原始记录，不能替代原始 `generation_events`、`turns` 和 `retrieval_events`。

### 5.2 本仓库真实完整二搜轨迹：legacy XML 成功例

下面不是 native G2，而是本仓库已经归档、可逐字复算的历史 legacy probe。来源是 [`eval_predictions.jsonl`](../../../results/grouped-probe-20260723/eval/traces/eval_predictions.jsonl)，`record_id=trace:dc9e45fc219270e6f172848f`、`sample_id=hotpotqa:train:80364`、slot 4、`executed_search_count=2`、`invalid_action_count=0`、`response_clipped=false`、`EM=1`。它展示模型如何在首搜写错 `JL/JT` 后改写第二个 query，并用 `<answer>` 保持短终局。

<details>
<summary>展开完整 raw trajectory</summary>

```text
<|im_start|>user
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: What nationality is the manufacterer that partnered with JL racing in 2006?<|im_end|>
<|im_start|>assistant
<think>

</think>

<reasoning>
The user is asking for the nationality of the manufacterer that partnered with JL racing in 2006. I need to search for information regarding this specific partnership.
</reasoning>

<search>
Query: "JT racing c1 manufacterer 2006 partnership nationality"
</search>

<information>Doc 1(Title: "Cunningham Motorsports") Scherer signed on with Dodge to be a driver development program for the manufacterer in the ARCA Racing Series from 2001 to 2007. The team then announced a partnership with Team Penske (then Penske Racing) in June 2008, which would be effective for competition in the ARCA Racing Series (then ARCA ReMax Series), USAC Midgets, and the NASCAR Camping World Truck Series (then Craftsman Truck Series). Seven years later, Cunningham announced an ARCA-specific driver development program with Roush Fenway Racing to put Kyle Weatherman in a car. The team maintained both relationships until its closure, with Penske's Austin Cindric and
Doc 2(Title: "JT McNamara") many others open to professionals after forming a hugely successful partnership with the Bolger yard. In his native Ireland, McNamara was ever-present at the National Hunt Festival at Punchestown for the duration of his career. His haul included five La Touche Cups (2010 L'Ami, 2007, 2004 Spot Thedifference, 2005 Good Step, 2002 Risk Of Thunder), three Ladies Cups (2011, Outlaw Pete, 2006 Abrams Bridge, 2003 Buailtes and Fadas), three Bishopscourt Cups (2004, 2003 Ever Onward, 1998 Master Julian), two Champion Hunters' Chases (2012, 2010 On The Fringe) and the Grade 1 Champion NH Flat Race (2001, Like-A-Butterfly, run at Leopardstown).
</information>

After a brief analysis, it seems that the specific detail about a manufacterer partnering with "JL racing" (which might refer to Johnline Racing or a typo for "A.L. Racing" as seen in Doc 1) in 2006 could require precise sports database knowledge. The provided snippets discuss "Cunningham Motorsports" (Doc 1) and "J.T. McNamara" (Doc 2), but neither explicitly matches the "manufacterer that partnered with JL racing in 2006" query. Given the potential ambiguity and the specific constraint to search, I will perform a dedicated search on "manufacterer partnered with JL racing 2006".
</think>

<search>
Query: "manufacterer partnered with JL racing 2006 nationality"
</search>

<information>Doc 1(Title: "JL Racing") 2007 the team entered the Castrol Canadian Touring Car Championship. The month of September saw a win at the British Empire Motor Club (BEMC) Indian Summer War Bonnet three-hour endurance race. Another record was set as it was the first time a Saab 9-3 had won an endurance race in Canada. The final weekend of racing saw the team claim fifth position for the championship. JL Racing JL Racing is one of Canada's premier sport sedan racing teams. In 2006, JL Racing formed a partnership with Saab Canada to race and showcase Canada's only Saab 9-3 race car. The team
Doc 2(Title: "JL Racing") JL Racing JL Racing is one of Canada's premier sport sedan racing teams. In 2006, JL Racing formed a partnership with Saab Canada to race and showcase Canada's only Saab 9-3 race car. The team made headlines several times with a first-race, first-place finish and subsequent podium finishes in the Saab 9-3. The team currently competes in the Castrol Canadian Touring Car Championship. In 2008 they will enter two Saab 9-3 MY 08 cars in the Super Touring class and will run North America's first Saab 9-3 BioPower race car in the Touring Class. The team's current full-time drivers are
Doc 3(Title: "Cunningham Motorsports") Scherer signed on with Dodge to be a driver development program for the manufacterer in the ARCA Racing Series from 2001 to 2007. The team then announced a partnership with Team Penske (then Penske Racing) in June 2008, which would be effective for competition in the ARCA Racing Series (then ARCA ReMax Series), USAC Midgets, and the NASCAR Camping World Truck Series (then Craftsman Truck Series).
</information>

Based on the search results, the details about a manufacterer partnering specifically with "JL Racing" in 2006 refer to "JL Racing" itself. When searching for the details of the "manufacterer that partnered with JL Racing", the provided text indicates that "JL Racing" is a team of Canada's premier sport sedan racing teams. The context of the partnership involves "Saab 9-3". The Saab brand is Swedish. Therefore, the nationality of the manufacterer (Saab) is Swedish.

<answer> Swedish </answer>
```

</details>

这条轨迹也说明：模型可以在一条长 reasoning/search 轨迹末尾给出短答案。EM 成功并不是因为整段输出短，而是 scorer 只抽取 `Swedish`。

### 5.3 论文完整三搜轨迹：第三搜用于自验证

论文 Appendix I Table 9（PDF 第 20 页）给出 Qwen2.5-7B-Base 训练后的案例。以下省略号属于论文表格中的检索片段截断，不是本报告删除了 action：

<details>
<summary>展开论文 Search-R1 完整 action sequence</summary>

```text
Question: Curious is a women's fragrance by a singer born in what city and state?
Ground Truth: McComb, Mississippi

<think> I need to find out which city and state a singer, by whom the women's fragrance
"Curious" was created, was born in. </think>
<search> Curious fragrance information </search>
<information> Doc 1(Title: "Curious (fragrance)") Curious (fragrance) Curious is a women's
fragrance by Britney Spears for Elizabeth Arden, and is the first perfume to be endorsed by
Britney Spears, preceding "Fantasy." Spears reportedly earned $52 million from the endorsement.
"Curious" was released in September 2004, and was very successful internationally. It was the
number one fragrance of 2004 in department stores, and in 2005, "Curious" was honored by the
Fragrance Foundation as Best Women's Fragrance. The fragrance is a white floral scent. Its notes
are Louisiana magnolia, golden Anjou pear, lotus flower, tuberose, star jasmine, pink cyclamen,
vanilla-infused musk, sandalwood and blonde woods. Elizabeth Arden ... </information>

<think> I need to find out which city and state Britney Spears, the singer behind
"Curious" fragrance, was born in. </think>
<search> Britney Spears birthplace </search>
<information> Doc 1(Title: "Britney Spears") the best-selling female albums artist of the 2000s. In
2004, she launched a perfume brand with Elizabeth Arden, Inc., from which sales exceeded US$1.5
billion. Spears was born in McComb, Mississippi, the second child of Lynne Irene Bridges and
James Parnell Spears. Her maternal grandmother, Lillian Portell, was English (born in London),
and one of Spears's maternal great-great-grandfathers was Maltese. Her siblings are Bryan James
and Jamie Lynn. Britney was born in the Bible Belt, where socially conservative evangelical
Protestantism is a particularly strong religious influence. Spears was baptized into the Southern
Baptist Convention, but later in life studied ... </information>

<think> Since Britney Spears was born in McComb, Mississippi, I need to find out which
city and state that is. </think>
<search> McComb, Mississippi location </search>
<information> Doc 1(Title: "McComb, Mississippi") McComb, Mississippi McComb is a city in
Pike County, Mississippi, United States, approximately south of Jackson. As of the 2010 census,
the city had a total population of 12,790. It is the principal city of the McComb, Mississippi
Micropolitan Statistical Area. McComb was founded in 1872 after Henry Simpson McComb of the
New Orleans, Jackson and Great Northern Railroad, a predecessor of the Illinois Central Railroad
(now part of the Canadian National Railway), decided to move the railroad's maintenance shops
away from New Orleans, Louisiana, to avoid the attractions of that city's saloons. The railroad
purchased land in Pike County, ... </information>

<think> I have found that Britney Spears was born in McComb, Mississippi. Since the
question asks for the city and state, I just need to confirm the state, which I already
know from the previous search. </think>
<answer> McComb, Mississippi </answer>
```

</details>

第二搜其实已经给出答案，第三搜用于 self-verification。这正是成本实验要研究的空间：B 可能保留第三搜，而 C 在不损失 EM 的前提下学会省掉它。无论是否省搜，最终 scorer 都只比较 `McComb, Mississippi`，不会把 preceding reasoning 算进答案。

## 6. 当前问题清单

### P0-1：自由文本终局与 strict EM 的合同不匹配

[`tool_protocol.py`](../../../../search_r1/llm_agent/tool_protocol.py) 当前把所有不含协议 marker 的非空回复整体作为 `final_answer`；[`main_ppo.py`](../../../../verl/trainer/main_ppo.py) 再用 `qa_em.em_check` 对整段文本做规范化后的全串相等比较。

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

HF rollout 从经过 top-k 和 presence penalty 处理的 proposal 分布采样，但 actor old/current log-prob 仍按 temperature 后的完整词表分布重算。两者不能直接构成严格一致的 PPO ratio。旧实现因此拒绝全部 `qwen35_native && !trainer.val_only`；native-v2 实现保留 fail-closed guard，只放行本方案精确注册的 smoke/R/B/C 四种训练合同和中性采样，其他 native 训练仍拒绝。

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

## 7. 最小修复方案

### 7.1 恢复论文原版 `<answer>` 最终答案合同

搜索仍使用 Qwen3.5 原生 `search` tool call，不增加 `finish` 工具，不改官方仓库的 Agent 搜索状态机。普通 assistant 终局恢复为论文和原仓库已有的格式：

```text
<answer>...</answer>
```

上面的省略号仅是文档元语法，不会作为字面占位符写入模型 prompt。上一轮已有模型照抄 `query` 占位符的直接证据，因此 v2 不在答案标签内部放 `short answer`、`xxx` 或 `Beijing` 示例。

为把新 G2 的 prompt 变量限制在最终答案边界，native v2 原样保留 v1 已验证的四句搜索文案，只替换最后一句 marker-free 终局要求。这里的 `Use at most four searches` 对应本项目已注册的 `max_turns=4`，是 Qwen adapter 的现状，不冒充论文 Table 1 原句；本轮不再同时试验另一套搜索决策 prompt：

```text
Call at most one tool per assistant turn. Use search when external evidence is
needed. After each search result, decide whether another search is needed. Use at
most four searches. When you have enough evidence, output the opening tag
<answer>, then only the short final answer text, then the closing tag </answer>,
and end the response. Do not combine a tool call with a final answer in the same
assistant response, and do not output any text after the closing tag.
```

user message 仍只是 `Question: <question>\n`；tool schema 与固定 Qwen chat template 不变。Qwen3.5-2B 的已验证链路继续冻结在 non-thinking 模板，允许模型在 action 前保留普通 marker-free reasoning，但不再强迫它生成论文的 XML thinking 协议。该选择是为避免同时更换 thinking 模式，不是声称 Qwen3.5 或 5090 无法运行 thinking。

对应改动严格限制为：

1. 将 prompt version 提升为 `qwen35-native-search-v2-answer-tag`，不增加 few-shot、反思模板或新的搜索策略提示。
2. answer 和 search 先共用同一个 safe-prefix 规范化：允许 marker-free reasoning，并可在最前面容忍至多一个内容仅含空白的完整 `<think>...</think>` echo；非空、重复、嵌套或不平衡 thinking、其他协议 marker，以及现有 `_looks_like_json_tool_call()` 可识别的 JSON-style tool call，全部拒绝。该窄规则只修复 G2 已观测的 empty-think 误判，不为 non-thinking 模式再引入第二套隐藏协议。
3. 在 tool-call 判断之前识别唯一、非空且位于当前 assistant 回复末尾的 `<answer>...</answer>`；只把标签内部写入 `ParsedAction("answer", content)`，并像 search 分支一样单独保留前置 reasoning 供 trace 审计。
4. answer content 内不得出现任何保留 marker 或可识别的 JSON-style tool call。空答案、多个或嵌套 answer、answer/tool-call 混合、`</answer>` 后非空正文以及旧 marker-free 自由文本终局全部判 invalid，不为通过门禁保留宽松 fallback。answer 不要求从回复第一个字符开始，避免把允许的 marker-free reasoning 错判为非法。
5. 搜索分支继续使用现有 Qwen 原生 tool-call fullmatch，但改用第 2 条同一 prefix 规则；不重新启用 legacy `<search>`。
6. native reward 继续读取现有 batch-aligned `final_answer` 并调用 `qa_em.em_check()`；不复用 legacy `qa_em.extract_solution()`，因为后者针对“含 prompt 示例的完整轨迹”并要求至少两个 answer match。
7. native 路径不把 `</answer>` 注册成 stop string，也不复制 legacy 的 decode-truncate-reencode。HF generation 仍由 EOS 或 500-token 上限终止；parser 只能验证 decoded response 在 `</answer>` 后没有非空文本，不能反推模型一定生成了 EOS。尾随正文判 invalid，`response_clipped` 独立记录并纳入门禁，从而不改写 sampled token 与 PPO log-prob。
8. 原始回复、thinking prefix、抽取答案和 parse error 全部继续保留在现有 trace 事件中，不修改 reward placement 或 trace schema。
9. 不维护双 parser；旧 commit、prompt version 和原始 evidence 足以复算 v1 历史，新主路径只保留 v2 严格 parser。

### 7.2 保留 strict EM，补齐诊断指标

训练正确性 reward 默认仍为抽取后 strict EM。每条 trace 同时保存：

- 从现有终局 `generation_events` 派生的 `raw_final_response`，不新增顶层 trace 字段；
- `extracted_answer`；
- strict EM；
- 仅报告用的 normalized containment/subEM；
- 代表性 alias、日期顺序和矛盾答案轨迹；
- 实际搜索次数、post-hoc utility 和完整工具轨迹。

分析器必须从 `extracted_answer + gold_answers` 独立复算 strict EM，并核对 trace、Parquet 和 catalog 的 gold 一致性。这样既避免再次把格式问题误判为能力问题，也不通过宽松 reward 美化正式结果。

### 7.3 注册最小 on-policy 训练采样合同

所有新的 readiness gate、2-step smoke 和 native 训练统一使用：

```text
temperature=1.0, top_p=1.0, top_k=0, min_p=0.0,
presence_penalty=0.0, repetition_penalty=1.0
```

`top_k=0` 是当前 HF rollout 的禁用值。核心 guard 不直接删除，而是仅在上述精确配置、native v2、完整策略 mask 和训练 variant 均匹配时允许训练；任何 top-k、min-p、presence/repetition penalty 漂移继续 fail closed。

这会牺牲上一轮为提高生成多样性加入的 top-k/presence 参数，但避免实现大范围 processed-log-prob 张量链路，符合本项目的最小复现原则。由于采样分布改变，旧 G2 的良好工具行为不能直接外推，必须用相同训练分布重跑低成本门禁。

native-v2 exact attempt 还必须锁定 group 5、train batch 8、response 500 和 observation 384。现有通用脚本中的 batch 4 或 response 384 OOM fallback 不得在同一实验身份下静默启用；若 2-step smoke 仍 OOM，则归档并停止，任何降配都要另立配置版本后重新走 readiness gate。

### 7.4 只做增量 CPU 重封

新 prompt version、parser 和训练配置会使旧 handoff 失效，但不需要重新下载或构建全部资产。CPU 无卡阶段只执行：

1. 将现有 G0-G2 原始 evidence 归档并校验哈希；人工答案审计并行物化为逐轨迹 ledger，绑定 rubric、source trace digest 和 SHA-256，但 ledger 未完成不阻塞 handoff。两者都不改变正式结论或门禁。
2. 用相同 760 条样本、sample ID、顺序和配额，在独立的 `data/search_mix_qwen35_native_v2/` 中重新物化 native v2 prompt；旧 v1 目录不覆盖、不删除。
3. 运行 parser/reward golden tests、完整 pytest、shell 语法、云端事务测试和 Hydra 配置展开；golden tests 必须覆盖 JSON tool call 出现在 answer prefix/content 的两个拒绝分支。
4. 重新校验模型、语料、BM25、数据 manifest 和依赖 freeze。
5. 发布绑定新 commit 与 prompt version 的 `cpu_handoff.json` 和 `cpu.ok`。

不重新下载模型，不重建 Wiki/BM25，不重新筛选 NQ/Hotpot，也不改变数据配比。

## 8. 验证与训练顺序

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

### 8.1 G0/G1 与 G2 readiness

新 G0/G1/G2 必须使用和训练完全相同的 `top_k=0/presence_penalty=0` 采样。G2 保留原有门槛：

- invalid trajectory `<=5/96`；
- clipped trajectory `<=5/96`；
- degenerate search 不超过全部 search action 的 2%。

另外增加训练 readiness 事实检查，而不事后修改旧 G2：

- 至少出现 1 条 strict-EM 正样本；
- 至少出现 1 个 strict-EM mixed group；
- 唯一 terminal `<answer>...</answer>` 在带/不带安全 reasoning prefix 时均能严格抽取，trace 对齐和 reward 重放完全一致；
- answer/search 对同一 empty-think echo 和 marker-free prefix 给出一致判定；
- 不出现 marker-free 终局被静默当成合法答案的兼容路径。

任何一项为零都停止，不进入 G3 或训练。

### 8.2 G3 capability gate

在新的 exact attempt 中复用原 G3 固定的 held-out Hotpot-64、group-5、320 条规模和保守阈值。native-v2 答案合同与训练兼容采样是本文新注册的实验修订，不追溯声称为旧 G3 的事前设置：

- 有效正确多搜轨迹 `>=16/320`；
- 覆盖题目 `>=8/64`；
- learnable group `>=8/64`；
- clipped 与 invalid rate 均 `<=5%`。

若 G3 `NO-GO`，保留完整证据并停止。只有在确认检索链可用、但仍缺乏正确探索时，才另立小规模 oracle trajectory SFT warm-start 假设；不能把它混入当前 GRPO 复现。

### 8.3 两步训练 smoke

G3 `GO` 后只运行 2 个更新 step。必须同时满足：

- reward、advantage、loss、KL、entropy、grad norm 全部有限；
- 至少一个 group 有正确/错误奖励差，策略 token 上存在非零 advantage；
- old/current log-prob、response mask 与 tool observation mask 对齐；
- 无 OOM、CUDA、NCCL、Ray 或 checkpoint 错误；
- 保存 step-2 checkpoint、resolved config、完整 train log、WandB offline history 和逐轨迹 trace。

smoke 分析器从 catalog 与 trajectory 独立重算 strict EM，并从两个 step 的 console/WandB 证据核验上述数值和 mask 条件；`smoke-decision.json` 为 NO-GO 时仍封存完整科学终态，但 main 拒绝该 marker。smoke 不通过则停止，不通过降门槛、盲降 batch 或直接跑 60 steps 掩盖问题。

### 8.4 R/B/C 分支关系与奖励公式不变

2-step smoke 通过后才启动全参数 `R-mix60-native-v2`。smoke 只保留验证证据；正式 R 必须从封存的 Qwen3.5-2B revision 全新启动，不继承 smoke 的模型权重、optimizer 或 scheduler，因为 2-step 与 60-step 的 scheduler horizon 不同。R 完成后重新通过能力门，并要求至少 8/64 个 cost-contrast group；只有此时才从同一个 R checkpoint 对称训练：

```text
B-mix20-native-v2:       r = EM
C-gated-mix20-native-v2: r = EM * (1 - 0.10 * n_search / 4)
```

B/C 继续共享 parent digest、数据顺序、seed、batch、group、长度、协议、检索器和步数。两个固定终点会在同一 sealed val-128 上分别运行 group-1 endpoint eval，再生成逐题配对的 strict EM、搜索次数、统一 utility、共同答对题搜索差，以及代表性完整轨迹；分析器独立重算 strict EM 并拒绝 catalog/trace 漂移。

## 9. 论文不变量与必要适配

### 9.1 保持不变的论文合同与官方实现

- 最终答案严格使用 `<answer>...</answer>`，只对抽取内容计算 strict EM。
- R 和 B 使用纯 outcome reward `r=EM`，不增加 format reward、substring reward 或 neural reward model。
- 采用论文正式报告的 GRPO variant，而不是自创 RL 算法：每 prompt 采样 5 条，`temperature=1.0`、`top_p=1.0`、学习率 `1e-6`、warmup 0.285。
- 每次检索返回 top-3，单轮 response 上限为 500 tokens；论文的 E5/BM25 差异和 retrieved-content 长度单独登记，不能混写成不变量。
- 保留 retrieved-token loss mask、全参数微调、`KL=0.001`、clip ratio `0.2`、strict EM 规范化实现和官方仓库 Agent 主循环。

### 9.2 必要适配与显式保留的差异

- **模型与工具接口**：用户指定已后训练的 Qwen3.5-2B；搜索和检索回填使用其原生 tool-call/chat-template。`enable_thinking=False` 不是模型或显卡硬限制，而是冻结已经通过 G0-G2 的 non-thinking 链路，避免在修答案边界时再改变 thinking 模式；普通 marker-free reasoning prefix 仍由 parser 和 trace 保留。
- **答案示例**：保留论文 `<answer>` 语法，但实际 prompt 不放 `xxx`、`Beijing` 或 `short answer` 字面示例。历史轨迹已出现大量 `query`/`and` 占位符复制，删除示例是有直接证据的最小防干扰适配，不是新答案格式。
- **native answer parser**：论文只规定从 response 抽取 final answer；官方 legacy `qa_em.extract_solution()` 才是在含 prompt 示例的完整轨迹中取最后一个 answer。native 当前 assistant 回合不再含该示例，因此只接受唯一、非空、位于末尾且 content 无保留 marker/JSON tool call 的 answer，并让 search/answer 共用同一 safe-prefix 规则。这是角色化回合所需的 fail-closed 适配，比 legacy last-match regex 更严格，不能称为逐字复用原 extractor。
- **检索器**：论文和官方 [`retrieval_launch.sh`](../../../../retrieval_launch.sh) 默认入口使用 Wiki-2018 + E5 dense；本项目因 100 GB 持久盘、既有资产和付费预算固定使用同语料 BM25 top-3。该差异可能影响绝对 EM、证据召回和搜索次数，是必须披露的高影响混杂；B/C 使用同一封存索引，只保证内部成本对照公平，不宣称复现论文榜单。
- **保留的上游实现差异**：论文 Algorithm 1 的 `B=4` 是 search/answer/invalid 共用的总 action budget；官方主循环在当前 `max_turns=4` 配置下是 4 个允许真实检索的 turn，再给未终止样本 1 个 `do_search=False` terminal turn。本项目为避免改 Agent 核心循环而显式保留该差异：真实检索 `0..4` 次、generation action 最多 5 次；terminal search 不执行、不计成本。这不是 Qwen/硬件必需，也不是论文伪代码的同义改写。
- **终止语义**：论文 XML rollout 检测闭标签即停止；native HF 路径不新增 stop string，也不做 decode-truncate-reencode。generation 由 EOS 或 500-token 上限结束；parser 要求 `</answer>` 后无非空文本，`response_clipped` 另行检查，不能声称 parser 验证了 EOS。
- **上下文容量**：论文 retrieved content 和 total sequence 上限分别为 500/4096 tokens；本项目 observation 固定 384，使当前上游四搜加 terminal 路径的 policy right side 为 `4036`，initial left side 另按最多 1024 保存。它是当前代码路径的容量适配，不是论文 4096 total-sequence 的等价实现。
- **rollout 后端与中性采样**：论文使用 vLLM；HF + SDPA 也不是 Qwen3.5/5090 硬限制，本项目只为复用已验证链路、避免新增后端变量而冻结它。`top_k=0`、`min_p=0.0`、`presence_penalty=0.0`、`repetition_penalty=1.0` 关闭论文未登记、且当前 HF 路径无法与 actor log-prob 一致重放的额外处理。GRPO loss、policy mask、KL 和 clip 公式不改。
- **资源缩放**：论文为 8 张 H100、500 steps、总 batch 512、mini/micro batch 256/64；本项目锁定两张 5090 级 GPU、group 5、train batch 8、PPO mini/micro batch 40/2、R 60 steps、B/C 各 20 steps，不宣称等规模复现。CPU handoff 会封存 smoke/R/B/C 四份 native-v2 训练配置和 B/C 两份 endpoint eval 配置；只有门禁和 smoke 证据通过后，付费入口才按本节顺序放行。
- **数据缩放**：固定 train-512/val-128、NQ/Hotpot `37.5%/62.5%`、既定 Hotpot comparison/bridge 配额和 seed；这是预算下提高多跳与成本对照密度的预注册子集，B/C 完全共享。
- **研究变量**：C 分支仅把 B 的 `r=EM` 改为 `r=EM*(1-0.10*n_search/4)`；B/C 从同一个 R checkpoint 对称分叉，其余配置和样本顺序完全一致。
- **付费运行安全**：G2/G3、2-step smoke、exact evidence、原始 exit code 和 watchdog 关机只用于止损与复算，不改变模型目标函数。

除以上必要适配外，不做 LoRA、模型 sweep、参数 sweep、多 seed、LLM judge、临时切换 dense retriever、宽松答案匹配、processed-proposal PPO、额外 SFT 或通用训练平台。若新 G3 不通过，只归档结果；任何新训练阶段必须另立文档，不隐式并入本轮。

## 10. GO/NO-GO 与停止条件

| 检查点 | GO | NO-GO / 工程失败后的动作 |
| --- | --- | --- |
| CPU handoff | 新 commit、数据、配置、测试和 handoff 全部一致 | 不开 GPU，修复后重新增量重封 |
| G2 readiness | 格式门通过且 strict reward/mixed group 非零 | 归档并停止，不靠 substring 或调阈值强行放行 |
| G3 | 三项能力门和质量门全部通过 | 归档科学负结果，不启动 GRPO 长训 |
| 2-step smoke | 数值、mask、reward、advantage、checkpoint 全部正常 | 保留失败 checkpoint/log，不启动 R-mix60-native-v2 |
| R 后复测 | 能力门和 cost-contrast 门均通过 | 只报告 R，不启动 B/C |
| B/C 完成 | 同一 parent 下完成配对评测 | 无论正负结果均归档，不通过继续调参制造成功 |

## 11. 为什么仍然有希望

本轮最重要的正面证据不是“门槛只差两条”，而是问题已经从模糊的“模型不搜索”收敛为可验证接口：

1. 原生 tool call、tool response 和多轮上下文已在真实模型上跑通。
2. 51/96 条轨迹至少二搜，36 条恰好二搜；搜索能力不再是零。
3. 多条轨迹已经检索到答案证据并生成内容正确的答案，只是短答案接口没有把它们转成 strict reward。
4. 人工明确正确标签下已有 13/32 个 mixed group，说明不是所有 group 都全错。
5. G2 没有显存、分布式或运行时错误，工程链路本身稳定。

剩余不确定性是：关闭 top-k/presence 后工具行为是否仍稳定、canonical 短答案遵循率是否足够，以及 2B 模型能否在预注册 G3 门槛下形成充足的正确多搜探索。因此当前判断应表述为“值得按最小修复继续验证”，而不是“已经保证训练成功”。

## 12. 证据位置与文档关系

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

- [`qwen35_native_tool_adaptation_plan.md`](qwen35_native_tool_adaptation_plan.md)：运行前的协议适配与 G0-G3 预注册计划；其中 marker-free 最终答案是 v1 历史设计，已由本文的严格 `<answer>` v2 合同取代；
- [`qwen35_native_tool_adaptation_implementation_report.md`](../stages/qwen35_native_tool_adaptation_implementation_report.md)：运行前 v1 实现快照与 fail-closed 原因，不作为 v2 最终答案设计；
- 本文：G0-G2 运行后的问题清单、新实验假设与训练解阻顺序；
- [`autodl_search_r1_reproduction_plan.md`](autodl_search_r1_reproduction_plan.md)：R/B/C 总体实验不变量与最终比较标准；
- [`成本感知坍缩分析与改进建议.md`](成本感知坍缩分析与改进建议.md)：提供历史坍缩证据与 correctness-gated 奖励设计依据；其中旧 parent、数据和三路回填安排不直接套用于 native-v2，当前执行顺序以本文和总体复现计划为准。

旧 legacy grouped probe 继续保存在 [`results/grouped-probe-20260723/`](../../../results/grouped-probe-20260723/)，不得与本次 96 条 native G2 轨迹混算。
