# Qwen3.5 Native R60 G3 评测与轨迹分析

## 1. 结论

本轮 G3 的 **320 条模型推理完整成功**，但按预注册五项能力门判定为 **NO-GO**。R60 已显著学会正确多搜：strict EM 为 `151/320 = 47.19%`，其中 `98` 条轨迹满足“正确、至少二搜、无截断/非法动作且第二次检索带来新证据”的严格条件，覆盖 `34/64` 题；独立的 clean cost-contrast group 也达到 `13/64`。因此，本轮失败不是“模型不会搜索”或“检索器不可用”。

真正的阻断项是冻结 checkpoint 的输出稳定性：`150/320 = 46.88%` 轨迹发生真实 clipping，`140/320 = 43.75%` 含 invalid action，clean learnable group 只有 `5/64`，未达到 `8/64`。这与 R60 第 52 步后已经观察到的长输出、`invalid_thinking_prefix` 和 KL 同步上升一致。按事前规则，G3 capability 为 NO-GO，即使 cost-contrast 单项通过，也不得把 B/C 写成正式获批。

此外，外层 attempt 在评测结束后的证据校验中误报失败，实际返回 `1`，而非封存后的受控 `201`。这是两个可在 CPU 阶段修复的 verifier 问题，不影响已经生成的模型轨迹，也不需要重跑 GPU。

## 2. 运行与证据身份

| 项目 | 固定值 |
| --- | --- |
| 外层 GPU attempt | `20260729T073211Z-3516-18146` |
| 内层 G3 eval | `20260729T073654Z-3558-15376` |
| sealed checkout | `f8c1cd7e87078d07385f74ca8710add5d5f79c06` |
| G3-only runner SHA-256 | `3b1985a8321ad6b19d5c7559fb55a5731d4e85f2926547233c933754c4ef4d3e` |
| G3 CPU receipt SHA-256 | `b499b53cfa7561bd57aeb4cb4cde25bbd8488907fac610fc9e75e3a1c6f7c4c1` |
| R60 evidence SHA-256 | `2366ce2da28b530f12af30a22d3dc3e2bd33fedbfcb3a2bed5acdd9d0537ffed` |
| R60 checkpoint | `runs/reproduce/attempts/20260728T092332Z-2946-8453/checkpoints/actor/global_step_60` |
| checkpoint tree SHA-256 | `583771b131b6e2aa663ee2ef13ea6421524cdd7fed839f35fd6f9eb246a2c231` |
| probe | held-out HotpotQA `64 x 5 = 320` |
| probe Parquet SHA-256 | `b8de9f20ba2eb41184578f543fd6be988866617702811300be079ece74ae47d1` |
| trace SHA-256 | `0020eb4b2f35fc16ea1115b1f5fde5ad01f4451e3a0a25a59776d9cd55051075` |
| trace 大小 | `28,525,156` bytes，严格 `320` 行 |
| 内层终态 | `.success`、`terminal=success`、`exit-code=0` |
| 外层终态 | `.failed`、`terminal=failed`、`exit-code=1` |

内层评测从 `07:36:54Z` 到 `09:11:05Z`，耗时 `5651` 秒，即 94 分 11 秒；外层从启动到终态共 98 分 55 秒。按两卡整机 `5.76 元/小时` 估算，本次约 `9.50 元`。耗时合理：320 条轨迹共执行 724 次 BM25 检索，每条轨迹平均生成约 992 token，P95 为 2034，最大达到五轮各 500 token 的 2500。

watchdog 在外层终态持久化后成功 dispatch `/usr/bin/shutdown`。这只证明 guest 关机请求已发出；实际停止计费仍以 AutoDL 控制台为准。

## 3. 预注册门禁结果

| 指标 | 观测值 | 要求 | 结果 |
| --- | ---: | ---: | --- |
| valid correct multi-search | `98/320` | `>=16` | PASS |
| covered questions | `34/64` | `>=8` | PASS |
| clean learnable groups | `5/64` | `>=8` | **FAIL** |
| clipped ratio | `150/320 = 46.88%` | `<=5%` | **FAIL** |
| raw invalid-action ratio | `140/320 = 43.75%` | `<=5%` | **FAIL** |

能力门要求五项同时通过，因此正式结论为 NO-GO。分支授权还要求 capability GO 且 clean cost-contrast group `>=8/64`。本轮 cost-contrast 为 `13/64`，说明同题答对轨迹之间确实存在搜索成本差异，但它不能覆盖 capability 的三个失败项，`branch_training_authorized` 仍必须为 `false`。

`64` 个 group 的 raw EM 构成为：23 组全错、19 组全对、22 组对错混合。只有 6 组同时包含 clean 正确与 clean 错误成员，其中 5 组还包含严格 qualifying 的正确多搜轨迹，所以 learnable group 最终为 5，而不是 raw mixed group 的 22。

## 4. 搜索、回答与正确性

| 实际搜索数 | 轨迹 | strict EM | 正确率 |
| ---: | ---: | ---: | ---: |
| 0 | 9 | 0 | 0% |
| 1 | 31 | 7 | 22.58% |
| 2 | 181 | 111 | 61.33% |
| 3 | 65 | 22 | 33.85% |
| 4 | 34 | 11 | 32.35% |
| **合计** | **320** | **151** | **47.19%** |

`280/320 = 87.5%` 的轨迹至少执行两次搜索，平均搜索 `2.2625`。二搜轨迹正确率最高；三搜和四搜正确率下降，说明存在成本优化空间，但不能据此推断“多搜导致错误”，因为题目难度和失败后继续搜索都是混杂因素。

`190` 条轨迹在四个常规 action 内直接回答，其中 `127` 条正确；另有 `130` 条进入 answer-only terminal generation：

- 75 条请求 answer，其中 71 条格式合法；
- 7 条再次请求 search，全部按安全边界拒绝且没有执行；
- 48 条没有形成可解析动作，其中 45 条 terminal generation 被截断；
- terminal 轨迹最终 strict EM 为 `24/130`。

最终共有 261 条合法 answer、59 条无答案。终局越界 search 已从早期常见行为降到 7 条，说明 terminal reminder 的停止提示有效但不是硬约束；当前主要失败已经转为长生成被截断和动作边界丢失。

## 5. Clipping 与非法动作根因

150 条 clipped 轨迹可进一步拆分为：105 条只在常规 action clipping，12 条只在 terminal clipping，33 条两处都 clipping。共有 206 个常规 clipped event 和 45 个 terminal clipped event。

非法事件分布如下：

| 类型 | event 数 |
| --- | ---: |
| `invalid_thinking_prefix` | 211 |
| `missing_native_action` | 21 |
| multiple/unbalanced tool calls | 12 |
| `search_disallowed_after_budget` | 7 |
| `invalid_terminal_answer_format` | 4 |
| multiple/unbalanced answers | 1 |

raw invalid 涉及 140 条轨迹。只有 4 条轨迹的 invalid 完全由 terminal 越界 search 的预期安全拒绝造成；排除它们后仍有 `136/320 = 42.5%` 轨迹存在真实格式/协议异常。因此，NO-GO 不能归因于 terminal reminder 把正常行为错误计成 invalid。

`invalid_thinking_prefix` 是绝对主因。典型输出在拿到检索文档后直接以 `The search results show...` 等自然语言开始，没有先进入 `<think>`；部分输出已经在正文中推导出正确答案，却在 500 token 上限前未形成合法 `<answer>` 或下一次 tool call，随后被环境判 invalid 并要求重试。这既增加推理长度，也消耗 action budget。

能力和稳定性可以由 clean/dirty 分层清楚区分：

| 轨迹状态 | 轨迹 | strict EM | 正确率 |
| --- | ---: | ---: | ---: |
| 无 clip 且无 invalid | 156 | 111 | 71.15% |
| 至少含 clip 或 invalid | 164 | 40 | 24.39% |

模型在协议保持正常时已经具备较强能力，当前瓶颈不是知识或 BM25 完全失效，而是 policy 的长输出和 native action 格式不稳定。

## 6. 数据类型分层

| 指标 | comparison | bridge |
| --- | ---: | ---: |
| 问题 / 轨迹 | `16 / 80` | `48 / 240` |
| strict EM | `9/80 = 11.25%` | `142/240 = 59.17%` |
| valid correct multi-search | 5 | 93 |
| covered questions | 3 | 31 |
| learnable groups | 2 | 3 |
| cost-contrast groups | 1 | 12 |
| clipped | `51/80 = 63.75%` | `99/240 = 41.25%` |
| invalid | `47/80 = 58.75%` | `93/240 = 38.75%` |

R60 的多搜收益主要来自 bridge；comparison 的正确率、稳定性和成本对比信号都明显不足。即使后续把 B/C 作为探索实验，也不能把 bridge 上的成本结果外推成两类 HotpotQA 问题都得到改善。

## 7. 代表性轨迹

### 7.1 明确的成本对比：Louis Durant

问题 `hotpotqa:train:37266` 询问 Louis Durant 成长城市在 2010 年的人口。五个 slot 全部 clean 且 strict EM=1，答案均为 `2,526`，搜索次数分别为 `2/3/2/4/2`。同一问题、相同正确答案下，二搜已经足够，而三搜、四搜没有增加正确性。这是最直接的 cost-contrast 示例。

### 7.2 可学习 group：VTES 3rd Edition

问题 `hotpotqa:train:24453` 中，slot 0 和 2 分别用 2、3 次搜索正确回答 `Wizard Entertainment`；slot 3 同样 clean、执行 2 次搜索，但错误回答 `White Wolf`。该 group 同时提供 qualifying correct 与 clean wrong，能产生 correctness advantage；正确成员之间又存在 2/3 搜差异，也能提供成本方向信号。

### 7.3 正确知识被格式与截断污染：Hickam Air Force Base

`hotpotqa:train:41000` 的轨迹已经从文档确认目标为 `Honolulu International Airport`，生成正文也反复写出该答案，但后续 continuation 没有合法 `<think>` 前缀并两次触及 500 token 上限。该轨迹最终 strict EM=1，却被计为 dirty，说明 reward 可以在最终答对时保留正确性，却没有直接惩罚中间的冗长和协议错误。

### 7.4 Strict EM 近失配：James Ellroy

near-miss `trace:348f1a70066080ae8898f912` 用两次搜索正确比较出生日期，抽取 `James Ellroy`，而 catalog gold 为 `Lee Earle "James" Ellroy`。它满足所有非 EM 的正确多搜条件，cover-EM 为 true，但 strict EM 仍为 0。本轮共有 28 条 near-miss、覆盖 15 题，其中 12 条 cover-EM 为 true。它们只能用于诊断答案抽取与别名问题，不能事后替换训练 reward 或 GO/NO-GO 口径。

## 8. 与历史结果的关系

早期 base grouped probe 同样为 64x5，但使用的是旧工具适配与不同实验合同，只能作方向性背景：当时 EM 为 `7/320`，搜索分布为 `140/90/53/32/5`，有效正确多搜仅 2 条、cost-contrast 为 0。当前 R60 G3 的 EM 达到 151、至少二搜达到 280、有效正确多搜达到 98、cost-contrast 达到 13，足以说明 R60 确实学到了搜索与回答能力；不能把提升全部解释成协议改造，但也不能将两个合同写成严格同题因果对照。

更接近当前协议的 native-v4 G0/G1 parent 在 32 条轨迹上为 `12/32` EM、`1/32` clipping、`12/32` raw invalid；排除预期的 terminal 安全拒绝后，真正协议异常涉及 `7/32` 轨迹。R60 后 G3 的能力更强，但 clipping 上升到 46.88%，真实非安全 invalid 上升到 42.5%。R60 训练末十步曾观测到 `50.5%` clipping 和 `46.5%` 非安全协议异常；冻结 step-60 的 held-out G3 与该趋势高度一致，验证了“后期长输出漂移”不是训练日志偶然噪声。

这也说明单纯把 response 从 500 再调大不是可靠修复：500 是当前论文对齐配置，提高上限可能减少表面截断，却允许 policy 生成更长文本、增加耗时，而且不会自动恢复 `<think>`/tool/answer 边界。

## 9. 两个外层工程假失败

### 9.1 runtime config verifier 漏掉解析联动字段

实际 `resolved-config.yaml` 与 CPU 模板只有预期的运行身份差异：actor model 切换到 sealed R60 checkpoint、输出目录和 trace identity 切换到本次 eval。verifier 已替换 `actor_rollout_ref.model.path`，但 CPU 模板是 OmegaConf resolve 后的静态值，因此它还需要同步替换由 actor path 插值得到的：

- `critic.model.tokenizer_path`；
- `reward_model.model.input_tokenizer`。

运行时将这两个 tokenizer path 指向同一个 R60 checkpoint 是正常解析结果；遗漏它们只导致评测完成后的 Python 对象比较失败，不改变已运行模型、数据或采样参数。

### 9.2 catalog 与 prompt 的问号规范不一致

`hotpotqa:train:14150` 的 catalog 原文末尾没有 `?`，而 `search_mix.py` 和 Qwen3.5 native prompt adapter 按既有合同自动补问号，因此其 5 个 slot 的 trace 问题多了一个尾随 `?`。其余 315 行完全一致；该题经标点规范化后也完全一致。进一步核对结果为：

- gold answer 列表不一致：`0/320`；
- strict EM 独立 replay 不一致：`0/320`；
- source ID/index/split、group slot 和 checkpoint identity 均通过 registered analyzer。

所以这是 verifier 没有复用 prompt 的规范化规则，不是题目错位或数据污染。

## 10. 当前证据状态与恢复方案

原外层 attempt 必须保持 `.failed/1`，不能追溯改写成成功。内层 eval 及 320 条 trace 也必须保持不可变。当前 diagnostic registered analysis 位于：

```text
/root/autodl-tmp/search-r1/runs/qwen-native-training/analysis-recoveries/
  20260729T073211Z-3516-18146/registered-probe
```

它已经验证 checkpoint、catalog、source identity、320 行结构、检索/action 对齐并生成正式门禁算法的 NO-GO；但由于 Qwen wrapper 的严格 replay 在尾随问号处提前退出，目前还没有 `qwen-native-training-g3-only` 官方 marker。

下一步应只做一个最小 CPU recovery：

1. 修正 runtime config verifier 的两个解析联动字段；
2. 对 active-v4 question 使用与 prompt adapter 相同的“缺少时补一个尾随 `?`”精确规范，而不是放宽成任意模糊匹配；
3. 新建不可变 recovery attempt，严格消费本次 exact outer 和 inner eval，复核 `.success/0`、trace manifest、checkpoint、数据、配置和原 runner 身份；
4. 重新运行 registered analyzer并发布独立 recovery evidence，保留源 outer `.failed/1`，不得伪造它曾返回受控 `201`；
5. 不启动 GPU、不重新采样、不自动进入 B/C。

按预注册实验，最终决策仍是 `G3 capability=NO-GO`、`branch_training_authorized=false`。如果为了简历展示继续成本奖励对比，必须另立明确的 post-hoc exploratory 合同，并如实说明它是在 capability NO-GO 后利用 `13/64` cost-contrast 信号做的探索，而不能包装成原计划确认性实验。

## 11. 面试表述边界

可以如实表述：

> 我对 Qwen3.5-2B 的 60-step 全参数 GRPO checkpoint 做了 held-out HotpotQA 64x5 评测。模型在 320 条轨迹上达到 47.2% strict EM，87.5% 轨迹执行至少两次搜索，并产生 13 个同题正确但搜索成本不同的 clean group，证明多搜和成本优化信号真实存在。同时，预注册能力门发现 46.9% clipping 和 43.8% invalid，根因是后期 policy 变得冗长并丢失 native action 边界，因此我没有自动启动成本分支，而是保留完整负结果和轨迹，区分模型能力、协议稳定性与工程封存错误。

不能表述为：

- “G3 运行失败，没有结果”：内层评测完整 `success/0`，失败发生在事后 verifier；
- “G3 已通过”：五项能力门有三项失败；
- “成本奖励已经有效”：本轮只评测原始 EM 奖励的 R60；
- “terminal reminder 导致 invalid 过高”：仅 4 条轨迹的 invalid 完全由安全拒绝造成；
- “把长度改大即可解决”：当前主要问题还包括 `<think>`、tool call 和 answer 边界漂移。
