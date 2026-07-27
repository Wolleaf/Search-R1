# Qwen3.5 Terminal Answer、Gold 数据与 WandB 修复方案

> 日期：2026-07-27
>
> 状态：实现完成并通过本地验收；待 CPU 增量重物化与 handoff
>
> 基线：`experiment/hotpot-search-gate@1e31808`
>
> 正式 smoke：outer `20260727T072737Z-11373-5756`，inner `20260727T072933Z-11409-19517`

本文依据正式 2-step smoke 的 80 条轨迹，确定下一次实现的最小边界。它不回写历史结果，也不把新增行为冒充论文原版。相对原版 Search-R1，唯一策略改进是预算耗尽后的 terminal answer-only 提示；strict EM、前四轮 Agent 循环、检索、GRPO 和成本奖励公式保持不变。

就后续实验的 terminal 行为而言，本文取代 `qwen35_search_r1_original_alignment_remediation_plan.md` 中“保留上游 terminal generation 但不增加终止提示”的旧决定；旧文档和旧轨迹继续保留，用于说明从原版对齐到本次新增改进的演进过程。

## 1. 决策与非目标

1. **非法动作保持原样。** 当前 parser 未发现误杀合法 Qwen 调用的证据；不放宽格式、不加 format reward、不做约束解码。
2. **新增 terminal answer-only 改进。** 四个常规 action 耗尽后，仅对仍 active 的轨迹注入 user message，并规定 terminal 轮唯一可接受动作是 `<answer>...</answer>`。
3. **保留论文 strict EM。** 只剔除明确错标或 gold 语义含混样本；不全局改成 substring、LLM judge 或顺序无关奖励。
4. **修复 WandB 日志与门禁。** 每步必须真正写入 offline history，正常或异常结束必须显式 finish；文件存在不再等价于 history 存在。
5. 在新证据通过前不启动 R60，不改变 group 5、batch 8、response/observation 500、action budget 4、采样参数、学习率或奖励系数。

本次 smoke 仍作为不可变历史保留：checkpoint、轨迹、`train.log` 和训练终态有效；其 `.wandb` 只有 stats/output，没有 metric history，因此旧 marker 不能授权 R60。

## 2. Terminal Answer-Only 改进

### 2.1 触发条件与提示词

只在 Qwen3.5 native 路径、完成第 4 个常规 action 后仍未回答时注入。提前输出 answer 的轨迹不受影响。使用以下固定英文 user message，并绑定文本 SHA-256：

```text
The search budget is exhausted. You must not call the search tool again.
Using only the question and information already available, give your best
answer even if uncertain. After reasoning, output exactly one concise final
answer inside <answer> and </answer>, with no text after </answer>.
```

这里使用“search budget is exhausted”，不声称一定执行过四次检索：invalid 同样消耗常规 action。提示允许 Qwen 保留 thinking，但要求 action 必须是短 `<answer>`。

### 2.2 正确注入方式

不能在现有 assistant generation header 后直接拼字符串。terminal user message 必须与第 4 轮 follow-up 在同一次 Qwen chat-template 渲染中生成：

```text
第 4 轮为 search:
assistant search -> tool response -> terminal user -> assistant generation header

第 4 轮为 invalid:
assistant invalid -> terminal user -> assistant generation header
```

invalid 路径不再先追加普通 `My action is not correct...` retry。实现应在 `Qwen35Conversation.append_followup()` 增加显式 terminal 模式，复用现有历史 reasoning 物化逻辑，并断言完整 prompt token IDs 等于原 prompt、原样 sampled response IDs 与新 suffix IDs 的拼接。禁止 decode 后重新 tokenize 策略输出。

tool response、terminal user 和模板 wrapper 都作为环境状态追加到 `next_obs_ids`，其 `info_mask/loss_mask` 必须为 0；terminal assistant 的 sampled tokens 保持 policy mask 1。若 suffix 超过 `max_obs_length`，只能进一步截短第 4 次检索 observation，不能截断 terminal instruction；固定 wrapper 与 instruction 自身放不下时 fail closed。容量合同必须计入该 suffix，但不调整 GPU 参数。

### 2.3 “强制”的精确定义

terminal 轮采用 answer-only allowlist：

- 合法 `<answer>...</answer>`：写入 `final_answer` 并结束；
- 再次生成 search：记为 `search_disallowed_after_budget`，不执行检索、结束轨迹、奖励 0；
- 其他非法格式：保留原始输出和错误原因、结束轨迹、奖励 0；
- 不从 thinking、search query 或自由文本中事后拼造答案。

这能保证 terminal search 永远不会被接受或执行，但文本提示不能数学保证生成模型 100% 输出合法答案。若要硬性保证标签，只能引入 constrained decoding 或无限重试；前者会使 rollout 采样分布与当前 actor/ref log-prob 不一致，后者改变 action budget，均超出最小实现。因此新门禁必须如实报告 terminal answer、search violation 和 invalid rate。

不在末轮临时移除 search schema。工具定义已经出现在初始 token context 中，使用另一套 `tools=[]` 重渲染可能改写历史前缀并破坏 PPO token/log-prob 对齐。

### 2.4 代码与证据范围

实际修改边界：

- `search_r1/llm_agent/tool_protocol.py`：固定 terminal prompt、模板 suffix 与 prompt hash；
- `search_r1/llm_agent/generation.py`：第 4 轮 follow-up 注入、mask 和 answer-only 执行；
- `search_r1/trajectory_trace.py`：记录 terminal instruction、请求动作、拒绝原因和对齐事实；
- `verl/utils/tracking.py`、`verl/trainer/main_ppo.py`：管理具体 WandB Run 的成功/异常终态；
- `scripts/autodl/qwen_native_*`、watchdog 与测试：提升协议/证据 contract，复核新字段。

初始问题 prompt、前四轮工具协议、检索结果格式和 strict EM 均不改。新版本明确命名为 `qwen35-native-search-v4-terminal-answer-only`，报告中注明这是本项目对原版 terminal rollout 的新增改进。

## 3. Gold Answer 语义与数据修复

### 3.1 当前规则

`golden_answers` 当前是 **OR 列表**。prediction 归一化后，只要完整等于其中任意一条即为 1：

```text
prediction = "Lee Hays"
gold = ["Pete Seeger", "Lee Hays"]
结果 = 1

prediction = "Pete Seeger and Lee Hays"
gold = ["Pete Seeger", "Lee Hays"]
结果 = 0
```

名称或实体顺序不会自动忽略；`Winger or Striker` 与 `Striker or Winger` 仍不相等。当前 640 条 catalog 中有 588 条单 gold、52 条多 gold；52 条全部来自 NQ，其中 train 38/512、val 14/128。多 gold 同时混有真正 alias 和“多个实体都必须回答”的 component list，不能把整个数组全局改成无序集合，否则会产生新的假阳性。

### 3.2 最小处理

不直接编辑 pinned source、旧 catalog 或旧 Parquet。新增版本化人工审计清单 `scripts/data_process/search_mix_answer_quality_exclusions.v1.json`，每条必须包含：

- pinned source revision 与 `source_id`；
- expected question 和 expected `golden_answers`；
- `disposition=retain_alias` 或显式的 `exclude_*`；
- 可复核原因与依据。

构建时先严格核对 expected 内容；源数据有任何漂移立即失败。52 条 multi-gold 已逐条人工审计：2 条为真实 alias 并保留，33 条要求同时回答多个实体，16 条语义含混、1 条 gold 不完整并剔除；另剔除已确认的 bit/nibble 错标。合计封存 53 条审计、排除 51 条。任何新入选 multi-gold 若没有显式 `retain_alias` 审计都会失败关闭，避免补位再次引入未经审计的 OR gold。

排除后从同 source/category 的既有合格候选中按原 seed 和稳定排序确定性补位，并保持原 output split；数据 manifest 封存人工清单 digest、审计/排除逐原因数量和完整替换 lineage。现有 NQ single 合格候选充足，因此无需重新下载数据或重建 BM25，只需 CPU 重物化为独立的 `data/search_mix_qwen35_native_v4/`，旧 v3 保持不可变。

处理顺序：

1. 明确错标（已知 `nq:train:32855` bit/nibble）直接剔除，不根据模型输出把 gold 改成 `0.25`；
2. 对 52 条多 gold 做一次静态人工分类：真实 alias 保留，multi-required 或语义不明确题剔除；
3. 不自动 split/join 人名，不自动生成排列组合，不依据 smoke 答案补 gold；
4. 额外输出 strict EM、subEM、gold count 和人工 audit 标签作为诊断，训练 reward 仍只用 strict EM。

如果以后要支持名字顺序无关，必须新增显式 `answer_mode=unordered_entity_set` 和结构化实体 gold；不能复用当前混合语义的字符串数组。本轮不引入该奖励变化。

## 4. WandB 修复

### 4.1 写入与生命周期

`verl/utils/tracking.py` 改为保存 `wandb.init()` 返回的具体 Run，而不是 wandb 模块：

- 每次使用 `Run.log(data, step=step)`，由 step 推进和显式 finish 提交 history；固定 WandB 0.21.1 下强制 `commit=True` 会使同 step 的后续验证指标被丢弃，因此不采用；
- 增加幂等 `finish(exit_code)`；正常完成调用 `Run.finish(0)`；
- Ray owner 中 `init_workers()+fit()` 成功后必须 finish(0) 才能返回；
- 任意训练异常时 best-effort finish(1)，随后重新抛出原异常；finish 的次生异常不得覆盖原训练错误；
- 成功训练但 finish 失败属于工程失败，不得发布 success evidence。

该修改只涉及日志，不改变模型、reward、optimizer 或训练顺序。

### 4.2 真实二进制门禁

`wandb_history.py` 与 `qwen_native_smoke_analysis.py` 使用当前固定 WandB 版本的 `DataStore` 扫描唯一的普通非软链 `run-*.wandb`；解析失败、截断、伪字节、多 run、扫描中输入漂移或 exit 后仍有 record 均 fail closed。2-step smoke 的 GO 必须同时满足：

- history 精确覆盖 `_step=[1,2]`，每步恰有一条完整 actor metric record；
- 每个 step 同时包含有限的 PG loss、KL loss、entropy、grad norm 和 PPO KL；
- history 数值与 `train.log` 在固定容差内一致；
- 至少一个 summary record；
- 恰好一个 exit record，且 `exit_code=0`。

scanner 为每个 run 写不可覆盖的 canonical receipt，绑定模式、预期 steps、训练日志 digest、唯一 run 和完整 WandB 文件树；写入前后复核输入稳定性并 fsync 文件和父目录。`wandb_offline_history.observed` 改为 history record 数，不再是文件数。WandB 文件树 SHA 继续封存，但不能替代内容解析。watchdog 和 main evidence publisher 必须重新验证 receipt，不能只信 decision JSON 或 `*.wandb` 文件存在。

旧 smoke marker 不删除、不改写，也不从 `train.log` 伪造 history。新 contract 自然拒绝旧 marker。

## 5. 测试与验收

### 5.1 本地与 CPU 阶段

必须覆盖：

- search、invalid、提前 answer 和 terminal answer 四类 Qwen 对话角色顺序；
- terminal prompt 完整、历史 token 未重写、环境 token mask 0、answer token mask 1；
- terminal search 被拒绝且零 retrieval，原始请求仍可审计；
- observation 可截断但 terminal prompt 不可截断；
- legacy XML 路径完全不变；
- gold OR 命中、组合答案失败、倒序失败；exclusion 防漂移、确定性补位和 manifest digest；
- fake WandB Run 的默认 commit 行为与 finish(0/1)；owner 正常/异常生命周期；
- 用临时目录真实生成 offline 2-step `.wandb`，验证完整正例，以及缺 history、缺 step、NaN、finish(1)、损坏文件和软链负例；
- smoke analyzer、training pipeline、watchdog、CPU reseal transaction、完整 pytest 与 shell 语法。

CPU 阶段复用现有模型、Wiki、BM25 和检索 evidence，只重物化受 exclusion 与 prompt contract 影响的数据，并发布绑定新 commit 的 handoff。

### 5.2 GPU 阶段

新 commit 不复用旧 gate 或 smoke marker：

```text
新 G0/G1 结构门禁
  -> 2-step smoke（从同一封存 parent 全新启动）
  -> 人工审计 terminal 轨迹与 WandB binary
  -> 再决定是否启动 R60
```

除原有 finite loss、mixed group、advantage、checkpoint 和 trace 条件外，新增：

- terminal instruction applied count 与第 4 轮后 active count 完全一致；
- terminal prompt policy-token count 为 0；
- terminal accepted/executed search 均为 0；
- terminal `final_answer` 只来自严格 parser；
- G0/G1 与 smoke 单独报告 terminal answer/requested-search/invalid rate；相较当前 `8/55` answer、`43/55` requested search，R60 前目标为 terminal answer rate 至少 90%、requested-search rate 至多 5%；
- WandB binary 满足第 4.2 节全部条件；
- evidence 持久化后再由 watchdog 关机。

任何结构、数值、日志或行为门禁不通过都归档并停止，不自动降 batch、缩 response、放宽 EM 或启动 R60。

## 6. 实验解释与后续分支

后续 R、B、C 若获准，必须统一使用同一个 terminal-answer-v1 环境与同一份修正后数据，因此 B/C 成本比较仍是对称的；C 相对 B 仍只改变成本奖励。最终报告分开表述：

- **Search-R1 原版保持项**：前四轮 Agent 循环、strict EM、GRPO、检索与奖励；
- **Qwen3.5 必要适配**：native search/tool response、thinking 与 token mask；
- **本项目新增改进**：预算耗尽后的 terminal user instruction + answer-only allowlist；
- **工程修复**：gold quality exclusions 与真实 WandB offline evidence。

这样既保留论文复现边界，又能把“已找到答案但继续搜索”的失败模式转化为一个可解释、可复核的 Agent 架构改进点。
