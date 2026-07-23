# Qwen3.5-2B 原生工具协议适配与续跑计划

## 1. 决策与目标

2026-07-23 grouped probe 的 NO-GO 继续作为真实失败证据保留，但不能据此认定 Qwen3.5-2B 不会搜索。审计已经直接证明，占位符复制、含标签的恢复文案和宽松正则制造了大量退化调用；同时，当前输入没有启用 Qwen3.5 训练时使用的 tool schema、`<tool_call>` 和 `<tool_response>` 角色结构。

下一轮不再只修补旧提示词后直接长训，而是优先完成 **Qwen3.5 原生协议适配与分层验证**。只有协议和 Agent 循环通过低成本门禁，才恢复 `R-mix60 -> B-mix20/C-gated-mix20`。详细证据见 [`results/grouped-probe-20260723/qwen35_tool_protocol_audit_zh.md`](results/grouped-probe-20260723/qwen35_tool_protocol_audit_zh.md)。

## 2. 不变项与允许改动

以下属于 Search-R1 核心实验语义，保持不变：

- 内部动作仍只有 `search(query)` 与 `answer(text)`；搜索后继续生成，答案动作结束轨迹。
- Wiki-18 BM25、top-3 文档、最多 4 次搜索、`4096/500/384` 长度、Qwen3.5-2B 固定 revision、全参数训练、batch 8、group 5 均不改变。
- GRPO、EM 奖励、正确性门控成本奖励、数据题目及 train/val 划分不因协议适配而改变。
- 历史 XML checkpoint、日志和结果只作历史证据，不覆盖，也不与新协议 checkpoint 混用。

允许改动仅位于模型边界：提示词渲染、工具 schema、动作解析、工具结果回填、生成停止条件、采样兼容项以及原始/规范化轨迹日志。

为避免再次把多个问题混成“模型不会搜索”，验证按因素拆开：

| 待排除因素 | 控制方法 |
| --- | --- |
| 权重或 tokenizer 错配 | 固定 model/tokenizer revision 与文件 digest；直接 HF 参考和 manager 记录同一组权重及 tokenizer 身份 |
| chat template 未真正生效 | 比较直接 `apply_chat_template` 与 adapter 首轮 token，要求逐 token 相同 |
| 协议而非采样造成退化 | G0 在相同问题、seed、`top_k=20` 和 `presence_penalty=2.0` 下保留 legacy/native 小型配对；legacy 只作诊断，不改历史结论 |
| parser 或回填错误 | CPU golden tests 加 G1 强制调用，先不以答案正确率评价 |
| 检索质量或多跳能力不足 | G0 不依赖检索答案，G1 只验证闭环；到 G2 才评价 query、证据链和 EM |
| RL 或奖励函数影响 | G0-G3 全部不更新权重；协议门禁通过前不接触能力/成本训练 |
| 随机碰巧通过 | 固定样本、seed、group slot 和阈值，不因结果重采样 |

## 3. 最小实现设计

### 3.1 可切换的协议边界

新增配置 `tool_protocol=legacy_xml|qwen35_native`。默认保留 `legacy_xml`，保证上游行为和历史测试不被静默改变；本项目后续 Qwen3.5 配置显式使用 `qwen35_native`。不建设通用插件框架，只实现一个小型 Qwen3.5 adapter，并向现有 generation manager 返回相同的内部 `(action, content)`。

原生模式只注册一个函数：

```text
search(query: string)  # 检索外部证据
```

Qwen 官方模板的终局是普通 assistant 文本，不是自定义 `finish` 工具：合法 `search` call 映射为内部 search；完全不含协议 marker 的非空文本映射为内部 answer。初始输入固定为 canonical system + `Question: ...` user 两条消息，再由固定 revision tokenizer 执行 `apply_chat_template(..., tools=TOOLS, enable_thinking=False, add_generation_prompt=True)`；不在 user 文本中展示 `query` 占位符。搜索结果按模板作为新的 `<tool_response>` 回合回填，再开启新的 assistant 回合。

多轮时不能把采样文本 decode 后再 encode，因为官方模板会 `trim` assistant content，且非规范 BPE 分段不保证可逆。实现用唯一 ASCII sentinel 从真实模板中只提取当前 assistant 之后的 suffix，再严格拼接 `exact prompt IDs + exact sampled response IDs + suffix IDs`；若采样结果已含 EOS，则去掉 suffix 开头的重复 EOS。这样既保留真实采样 token，又继续由官方模板定义 tool response 和下一轮 assistant wrapper。

### 3.2 严格解析与训练语义

- 只接受输出末尾唯一、结构完整的官方 `<tool_call>`；函数名、参数名、嵌套或数量错误均判 invalid。
- 拒绝空 query 和字面量 `query`/`and`，恢复提示不包含任何成对 action 标签。
- marker-free 的普通最终文本只在环境元数据中映射为 `answer`；它不被伪造成 `<answer>` token，EM 直接读取 batch-aligned `final_answer`，成本奖励公式不变。
- assistant 实际采样的 tool call、普通答案和 EOS token 参与策略训练；tool response、恢复消息、role 边界和下一轮 generation prefix 全部进入 observation mask，不计入策略 loss。
- 长度合同按 policy right side 计算：`4 * (500 + 384) + 500 = 4036 <= 4096`。initial left prompt 不计入这 4036 个 token；native rolling cap 为 `1024 + 4096 = 5120`，因此最坏 `1024 + 4036 = 5060` token 的采样上下文与 PPO 重算上下文保持一致。有效容量内不左裁历史，policy right side 超限也直接报错，不能静默截断训练 token。
- 日志同时保存模型原始生成、规范化 action、parse error reason、query、原始检索文档、可见 observation、最终答案、截断和每个回合的 role，使失败可以逐条复算。

### 3.3 数据与采样

数据构建器增加协议参数，使用相同 sample ID、配额、seed 和检索证据重新物化一份 native Parquet；只替换说明文本，不重新挑题。manifest 必须记录协议、tool schema digest、tokenizer revision 和源数据 digest。

采样完整采用固定 Qwen3.5 model card 对 non-thinking 文本任务的建议：

```text
temperature=1.0, top_p=1.0, top_k=20, min_p=0.0,
presence_penalty=2.0, repetition_penalty=1.0
```

`top_k=20` 直接进入 HF `GenerationConfig`。当前 Transformers 的 `GenerationConfig` 没有 `presence_penalty`，因此在 HF rollout 内增加一个小型、可配置的 logits processor：只对当前 assistant 回合已经生成过的 token 减去一次固定 penalty，不按出现次数累加，不惩罚 prompt/tool response，并在新的 assistant 回合开始时重置。这与 presence penalty 区别于 frequency/repetition penalty 的定义一致。legacy 默认仍为 `top_k=0/presence_penalty=0`，避免历史入口静默变化；Qwen native 入口显式设为 `20/2.0`。

实现必须同时支持 `presence_penalty=0` 的严格 no-op。有效采样值由 resolved config 与 `run.env` 交叉校验，阶段另写 `sampling.json`；这些文件与 trace manifest/digest 一起进入 evidence seal，而不是假定每个 trace manifest 自带全部采样字段。单测覆盖首次 token 不受罚、已生成 token 只减一次、重复多次不额外累加、batch 独立、回合重置以及 0 值等价。官方提示高 penalty 偶尔会引发语言混杂，因此 G0/G1 同时统计重复片段和异常语言 query；只有出现可复现的明显退化，才将 `2.0 -> 1.5` 注册为一次独立采样对照，不能事后按 EM 挑值。不迁移 vLLM，不更换模型尺寸。

## 4. CPU 无卡阶段

1. 实现 adapter、配置开关、native 数据 prompt、answer 映射和轨迹字段；旧 XML 代码路径不删除。
2. 增加 parser 正反例、空/多 action、模板 token 前缀、tool-response role、loss mask、reward 等价、batch reorder、presence penalty 精确语义和 legacy 回归测试。
3. 用固定 tokenizer 做真实模板集成测试：schema 确实进入 system，`enable_thinking=False` 生效，第二轮上下文与直接 `apply_chat_template` 的 token 完全一致。
4. 从已有固定源文件和检索 ledger 重新物化 native probe/train/val Parquet；不重下 Wiki、不重建 BM25、不运行整套历史 CPU 流程。
5. 生成 resolved config、manifest、SHA-256 与新的 CPU handoff。任一 token 前缀、mask、样本集合或 digest 不一致均停止，不启动 GPU。

最低验证命令为 `python -m pytest -q`、native 数据构建器的校验模式以及 `AUTODL_CONFIG_ONLY=1` 配置解析。CPU 增量入口只复用已封存的检索 ledger 和选题结果，不重新下载语料或运行 BM25。

## 5. GPU 分层验证

所有阶段使用同一 checkpoint、检索器、题目、生成长度和采样配置。每一阶段先完整写出 trace、summary、resolved config 和 hash，再由绑定 exact attempt 的 watchdog 关机；不得为得到 GO 自动换 seed 或降低门槛。

| 阶段 | 规模与目的 | 通过条件 |
| --- | --- | --- |
| G0 原生参考 smoke | 8 个固定问题，每题 2 条；比较直接 HF/native manager，并以同一 Qwen 采样参数的 legacy manager 作诊断对照 | direct/native 首轮 prompt token `16/16` 相同，首轮 `raw_text` 逐条 `16/16` 相同；direct 与 native 各自可解析 `>=15/16`；resolved config 为 `top_k=20/presence_penalty=2.0`，且 native 退化 query 为 0 |
| G1 强制搜索 probe | 16 题 × 2 条；明确要求先搜索，只测 schema、parser 和回填闭环 | 合法首 action `>=31/32`，非退化 query `>=29/32`，退化 query `<=1/32`，首轮截断 `<=1/32`，所有检索均有对齐的 tool response，重复/异常语言 query 单独列出 |
| G2 自主搜索 probe | 固定 held-out Hotpot 32 题 × 3 条；允许自主决定搜索和结束 | 非法轨迹、截断轨迹各 `<=5/96`；退化调用不超过全部搜索的 2%；逐题报告 EM、搜索数、query 相关性及完整二搜链 |
| G3 正式 grouped gate | 现有 held-out Hotpot-64 × group 5 | 沿用原预注册门槛：有效正确多搜 `>=16/320`、覆盖题 `>=8/64`、learnable group `>=8/64`，并满足原非法动作和截断门槛 |

G0+G1、G2、G3 分别建立独立 exact attempt，整个 paid stage 的外层 TERM deadline 依次由 2、3、10 元和当次整机时价换算。deadline 覆盖 handoff/数据复核、BM25、模型生成、分析与 evidence seal；内部生成 timeout 取注册额度与“剩余时间减 180 秒收尾预留”的较小值。BM25 不继承 phase lock，清理使用有界 TERM→KILL，保证异常时 watchdog 仍能接管。进程若不响应 TERM，GNU `timeout` 最多再保留 120 秒强杀宽限（按 5.76 元/小时最坏约 0.19 元），因此该额度是工作时限，不是含强杀宽限的绝对账单上限。G2 必须绑定 G0+G1 的 evidence digest，G3 必须绑定 G2 的 evidence digest；科学 NO-GO 正常归档且不自动重试，超时或其他工程错误返回非零并交给已绑定 exact attempt 的 watchdog。三个注册额度合计 15 元，任一阶段失败都不自动进入下一阶段或训练。

## 6. 归因与停止规则

- **直接 HF 参考也失败**：优先核对 checkpoint revision、tokenizer/template、`top_k/presence_penalty` 实际值和官方调用示例；此时不能归因于 Search-R1，也不训练。
- **参考通过但 G0/G1 失败**：确定是 adapter、batch padding、停止或 parser 实现问题；回 CPU 修复，不靠增大模型或 response 掩盖。
- **G1 通过但 G2 格式失败**：检查多轮 role 回填、context 裁剪和 observation mask；不改数据配比或奖励。
- **G2 格式稳定但 G3 能力 NO-GO**：协议因素已经基本排除，剩余假设才是 2B 容量、检索证据质量或缺少格式/多跳 SFT；保留结果后另立实验，不启动稀疏奖励长训。
- **G3 GO**：说明新 parent 中存在足够正确多搜探索，才进入能力训练。

## 7. 通过后的训练顺序

新协议不能沿用旧 XML 权重做因果对照。G3 通过后，从同一个官方 Qwen3.5-2B revision 训练 `R-mix60-native`；完成后重新运行 grouped probe，并要求原能力门槛及 `cost-contrast group >=8/64`。只有再次通过，才从同一个 R checkpoint 对称训练：

```text
R-mix60-native
├── B-mix20-native         # EM reward
└── C-gated-mix20-native   # EM * (1 - 0.10 * n_search / 4)
```

B/C 除奖励模式外共享 parent digest、数据顺序、seed、batch、group、长度、协议和检索器。最终在同一 test 集配对报告 EM、搜索成本、Utility、共同答对题的搜索差及完整轨迹。旧 NO-GO 和 C-old 坍缩仍作为项目中“发现协议问题与奖励问题并逐层修复”的失败经验，不改写为成功结果。

当前脚本和核心 trainer 都只允许 native 的 `val_only` 门禁，任何 native train 组合都会 fail closed。G3 通过后再开放 `R/B/C-native` 训练入口时，必须先预注册采样分布口径：HF rollout 的 `top_k=20/presence_penalty=2.0` 是 proposal sampling，而现有 PPO actor 按 temperature 后的完整词表重算 log-prob。二选一：训练显式关闭 `top_k/presence_penalty`、继续使用完整词表 log-prob；或保存并在 old/current policy 中一致复算逐 token processed proposal log-prob。仅让 B/C 使用相同 proposal 参数可以保证对比口径一致，但不等于严格的 on-policy PPO，不能据此解除训练门禁。

## 8. 本轮交付边界

当前实现已完成 adapter、严格 parser、native role 回填、采样兼容、plain-answer reward、完整轨迹、独立数据物化与 G0-G3 云端入口；旧 XML 路径和历史结果保持不变。下一步只在无卡实例执行增量 CPU 物化与 handoff，验证通过后再按 `G0/G1 -> G2 -> G3 -> 训练` 逐段开 GPU；任何前置 gate 未通过，后续高成本阶段均不启动。
