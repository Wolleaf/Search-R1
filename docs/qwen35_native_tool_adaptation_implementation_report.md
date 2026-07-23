# Qwen3.5 原生工具协议适配实现报告

> 记录日期：2026-07-23
> 实现基线：`experiment/hotpot-search-gate` 分支，基线提交 `d80dec9fb09730769faa35f7784ef7453761a2c9`
> 报告范围：本轮代码、数据合同、AutoDL 门禁和本地验证；不把尚未运行的 GPU 门禁或训练写成实验结果。

## 1. 本轮解决了什么问题

上一轮 grouped probe 的 NO-GO 是有效的失败证据，但不能直接推出“Qwen3.5-2B 不会搜索”。审计发现，旧实验仍用 Search-R1 的 XML 动作格式，并存在占位符复制、带动作标签的恢复提示和宽松解析等干扰；Qwen3.5 实际训练时使用的 tool schema、`<tool_call>` 与 `<tool_response>` 上下文没有被正确接入。因此，模型能力、提示协议、解析器和采样因素混在了一起。

失败轨迹提供了直接证据：312 次检索中，字面量 `query` 有 139 次、`and` 有 70 次、空 query 有 1 次，合计 210/312（67.31%）属于退化调用。实际 checkpoint 是已后训练的 `Qwen3.5-2B`，不是未经后训练的 `Qwen3.5-2B-Base`；如此高的占位符比例更应先排查适配边界，而不是先归因于模型智力。原始审计保存在 [`results/grouped-probe-20260723/qwen35_tool_protocol_audit_zh.md`](results/grouped-probe-20260723/qwen35_tool_protocol_audit_zh.md)。

本轮没有直接重训，而是先完成一个可切换、可验证、可回退的 Qwen3.5 原生协议边界，并把验证拆为低成本 G0-G3 门禁。核心结论是：

- legacy XML 路径继续作为默认值，历史实验与 checkpoint 不被覆盖。
- native 数据沿用相同题目、样本 ID、配额、seed 和检索证据，只改变模型边界的 prompt 表达。
- G0-G3 只做生成评测，不更新权重；协议未通过前禁止烧钱训练。
- native PPO 在脚本层和核心 trainer 层均 fail closed，避免采样分布合同未定义时误训。

## 2. 保持不变的实验语义

为了让后续对比仍能回答“成本奖励是否减少无效搜索”，以下设置没有因协议适配而改变：

| 项目 | 固定值或规则 |
| --- | --- |
| 模型 | 固定 revision 的已后训练 Qwen3.5-2B 本地资产，不使用 Base checkpoint |
| 检索 | Wiki-18 BM25，返回 top-3 文档 |
| Agent | 最多搜索 4 次，内部动作仍是 `search(query)` 与 `answer(text)` |
| 长度 | 初始 prompt 1024、单轮 response 500、observation 384、policy right side 4096 |
| 正式 grouped 配置 | batch size 8、group size 5、两张 GPU |
| 能力训练 | 后续仍是全参数 GRPO；本轮没有改成 LoRA |
| 奖励 | EM 基线和正确性门控成本奖励公式保持不变 |
| 因果对比 | B/C 必须共享 parent、数据顺序、seed、长度、协议和检索器，仅奖励不同 |

## 3. 新流程总览

```text
已封存 search_mix + retrieval replay
                |
                v
CPU 离线物化 native 数据 -> 真实 tokenizer 全量扫描 -> 新 CPU handoff
                |
                v
GPU G0+G1 -> exact evidence -> G2 -> exact evidence -> G3
                |                                  |
              NO-GO                              GO
                |                                  |
      保存完整失败证据并停止             才允许设计 native R/B/C 训练
```

每个箭头都绑定文件 digest，而不是按 `latest`、目录时间或文件是否存在来猜测前驱。科学 NO-GO 是正常结果，不会自动换 seed 或重试到通过；schema、超时、哈希漂移等工程错误保留非零退出码。

## 4. 原生工具协议实现

### 4.1 Canonical prompt 与工具 schema

新增 [`search_r1/llm_agent/tool_protocol.py`](../search_r1/llm_agent/tool_protocol.py)，集中管理：

- `legacy_xml|qwen35_native` 协议枚举与规范化；
- 固定 Qwen3.5 model revision、chat template digest 和 prompt version；
- canonical system/user messages；
- 唯一工具 `search(query: string)` 的 JSON schema；
- `apply_chat_template(..., tools=..., enable_thinking=False, add_generation_prompt=True)` 渲染；
- native 动作严格解析、重试消息和 schema digest。

这里只注册 `search`，没有人为增加 `finish` 工具。Qwen 官方模板的终局是普通 assistant 文本：完整合法的官方 tool call 映射为内部 `search`，不含任何协议 marker 的非空文本映射为内部 `answer`。

解析器只接受输出末尾唯一、结构完整的官方 `<tool_call>`。空 query、字面量 `query`/`and`、多 tool call、错误函数名、错误参数、嵌套错误、残缺 marker 和 JSON 风格伪调用都明确判为 invalid，并记录具体 `parse_error`，不再用宽松正则猜测模型意图。

### 4.2 多轮 token 保真与工具回填

[`search_r1/llm_agent/generation.py`](../search_r1/llm_agent/generation.py) 增加 native conversation 路径。初始 batch 必须携带 canonical raw messages；搜索结果由官方模板作为 tool response 回填，再开始下一轮 assistant。

多轮续接不能简单执行“decode 全文再 encode”，因为 chat template 会 trim assistant content，非规范 BPE 分段也不保证往返一致。实现使用唯一 ASCII sentinel 从真实模板中提取本轮 assistant 后缀，然后拼接：

```text
上一轮精确 prompt token
+ 模型实际采样 token
+ 模板产生的 tool-response/下一轮 assistant suffix token
```

若采样已经包含 EOS，会去掉 suffix 开头的重复 EOS。这样模型采样 token 保持逐 token 原样，角色边界仍由官方模板定义。

策略 mask 也按来源拆分：assistant 实际生成的 tool call、普通答案和 EOS 参与 loss；tool response、重试消息、role marker 与下一轮 generation prefix 只进入 observation，不参与策略更新。native 路径不会静默左裁有效历史；容量合同为：

```text
4 * (500 response + 384 observation) + 500 final response = 4036 <= 4096
1024 initial prompt + 4036 right side = 5060 <= 5120 rolling capacity
```

一旦超过已经验证的容量，代码直接报错，而不是截断训练 token 后继续制造不可复算轨迹。

### 4.3 Dataset 与 reward 对齐

[`verl/utils/dataset/rl_dataset.py`](../verl/utils/dataset/rl_dataset.py) 在 native 模式下验证 canonical messages、使用同一 renderer，并把 raw chat 原样交给 Agent loop。legacy 数据加载行为不变。

native 最终答案只保存在 batch-aligned `final_answer` 环境字段中，不伪造 `<answer>` token。[`verl/trainer/main_ppo.py`](../verl/trainer/main_ppo.py) 直接用该字段计算 EM，并继续使用真实 `executed_search_count` 计算成本奖励。缺失 final answer、搜索计数形状错误、负数或超过 4 次都会失败关闭。

## 5. 数据物化与 CPU 增量阶段

### 5.1 不重新选题，只重新表达 prompt

[`scripts/data_process/search_mix.py`](../scripts/data_process/search_mix.py) 新增 `materialize-native` 与离线校验。它从已封存的 source manifest、catalog、selection funnel 和 retrieval evidence 重新物化独立目录 `data/search_mix_qwen35_native/`，并逐项证明：

- train/val/probe 的 sample ID、顺序、seed 和类别配额不变；
- 除 `prompt` 外的字段逐行不变；
- 原 `data/search_mix/` 不被覆盖；
- native manifest 记录 prompt version、tool schema、chat template、tokenizer revision 和来源 digest。

需要扫描的 6 个 Parquet 共 760 条：

| 文件 | 行数 | 用途 |
| --- | ---: | --- |
| `train_512.parquet` | 512 | 后续能力/成本训练候选 |
| `val_128.parquet` | 128 | 训练期验证 |
| `probe_multi_64.parquet` | 64 | G3 grouped gate |
| `probe_g0_8.parquet` | 8 | G0 direct/manager 对照 |
| `probe_forced_16.parquet` | 16 | G1 强制搜索闭环 |
| `probe_autonomous_32.parquet` | 32 | G2 自主搜索 |

CPU 使用固定真实 tokenizer 对每一行执行四类检查：canonical messages 完全一致；问题中不存在协议保留 marker；“渲染字符串再 tokenize”与“chat template 直接 tokenize”逐 token 相同；初始 prompt 非空且不超过 1024 token。检查直接集成在现有 `verify_manifest` 循环，没有另建扫描服务；错误信息包含文件、行或 sample ID，不能把问题推迟到付费 GPU 阶段。

### 5.2 Retrieval replay receipt 闭环

增量模式不重新执行 BM25，但也不能盲信一个同名 JSON。`retrieval_replay.json` 会重新校验 manifest/catalog/evidence 哈希、corpus 与 BM25 revision、640 条入选样本、1024 次查询、选择顺序哈希、canonical JSON 和 `.sha256` sidecar。

[`scripts/autodl/handoff.py`](../scripts/autodl/handoff.py) 的 `verify --require-artifact` 让 CPU 增量入口和 GPU admission 都必须证明 replay receipt 与 sidecar 属于旧 handoff。增量入口先要求 `cpu.ok` 等于旧 handoff digest，再删除旧 `cpu.ok` 并开始重封，关闭“handoff 已更新但 cpu.ok 尚未更新”的 crash window；只有新 handoff 完整成功才重新发布 `cpu.ok`。新 receipt 随后进入 GPU input digest 和最终 evidence。

无卡实例只需运行：

```bash
AUTODL_QWEN_NATIVE_INCREMENTAL=1 \
bash /root/autodl-tmp/search-r1/checkout/scripts/autodl/02_cpu_prepare.sh
```

该入口不联网、不重装环境、不下载模型、不重建 Wiki/BM25，也不训练；它只做 native 物化、真实 tokenizer 验证、测试、三路 Hydra 配置展开和 handoff 重封。

## 6. 采样适配与 PPO 安全边界

### 6.1 G0-G3 的评测采样

native 门禁显式固定：

```text
temperature=1.0, top_p=1.0, top_k=20, min_p=0.0,
presence_penalty=2.0, repetition_penalty=1.0
```

[`verl/workers/rollout/hf_rollout.py`](../verl/workers/rollout/hf_rollout.py) 把 `top_k` 传入 HF `GenerationConfig`。由于当前 Transformers generation config 没有同语义的 `presence_penalty`，新增一个很小的 logits processor：只对当前 assistant 回合已经生成过的 token 减一次固定 penalty，不按出现次数累加，不惩罚 prompt/tool response，并在新 assistant 回合重置。`presence_penalty=0` 是严格 no-op。

legacy 默认仍是 `top_k=0/presence_penalty=0`。native 的有效值会同时从 resolved config 与 `run.env` 交叉核对，并写入 `sampling.json` 后进入 evidence seal，避免“代码默认值看起来正确、实际运行参数不同”。

### 6.2 为什么当前禁止 native PPO

HF rollout 在生成时使用经过 top-k/presence 处理的 proposal 分布，但现有 actor 在 PPO old/current log-prob 重算时只使用 temperature 后的完整词表分布。两者比较口径一致不代表数学上是严格 on-policy PPO。

因此 [`verl/trainer/ppo/ray_trainer.py`](../verl/trainer/ppo/ray_trainer.py) 在核心层拒绝 `qwen35_native && !trainer.val_only`，[`scripts/autodl/train_small_grpo.sh`](../scripts/autodl/train_small_grpo.sh) 也只注册 native 的 G1/G2/G3 eval 组合。G0-G3 不做 actor update，所以当前采样设置没有 PPO ratio 风险。

G3 GO 后若开放训练，必须先二选一：

1. 最小方案：native 训练关闭 `top_k/presence_penalty`，继续按完整词表重算 log-prob；评测门禁仍保留 `20/2.0`。
2. 严格方案：保存并在 old/current policy 中一致复算逐 token processed proposal log-prob，同时处理 top-k 支持集变化和 reference KL。

仅让 B/C 使用相同 proposal 参数只能保证实验对比口径相同，不能据此声称严格 on-policy PPO。本轮选择 fail closed，而不是扩大范围改造 actor/reference/worker 张量接口。

## 7. 完整轨迹与面试证据

[`verl/trainer/ppo/ray_trainer.py`](../verl/trainer/ppo/ray_trainer.py) 和 native Agent loop 会保存 batch 对齐的原始与规范化事件。单条轨迹可回答：

- 哪道题、哪个 `sample_id`、group/slot、checkpoint 和 stage 产生了结果；
- 每轮模型原始文本、token 数、是否 clipped、规范化 action 和 parse error；
- 实际执行的 query、原始 top-3 文档、模型可见的截断后 observation；
- 搜索是否真正执行、总搜索次数、最终普通文本答案和 EM；
- 训练场景中的 reward、post-hoc utility、advantage 与策略 token 覆盖率；
- trace、config、数据、模型、前驱 evidence 的 SHA-256 lineage。

事件对齐会检查“执行的 search action 数 = retrieval event 数 = `executed_search_count`”，final answer 也必须与终局 generation event 一致。任何错位直接报错，避免生成日志看似完整但无法复算。

[`scripts/autodl/qwen_native_gate_analysis.py`](../scripts/autodl/qwen_native_gate_analysis.py) 输出 `per_trajectory.jsonl`、`per_question.jsonl`、`summary.json`、`summary.md` 和 `go_no_go.json`。因此后续不仅能报告总体 EM，还能逐题比较“需要几次搜索、实际用了几次、哪一步 query 或证据链失败”。

## 8. GPU 分层门禁与成本控制

新增入口 [`scripts/autodl/08_gpu_qwen_native_gate.sh`](../scripts/autodl/08_gpu_qwen_native_gate.sh)，每次只运行一个 exact stage：

| 阶段 | 固定规模 | 主要排除因素 | GO 条件摘要 | 工作预算 |
| --- | --- | --- | --- | ---: |
| G0+G1 | G0 8×2×3 路（48 条）；G1 16×2 | 模板/token、manager、parser、tool-response 闭环 | G0 direct/native token 与 raw text 对齐；两路可解析至少 15/16；G1 合法首调用至少 31/32、非退化 query 至少 29/32 | 2 元 |
| G2 | 32×3 | 自主搜索格式、query 质量、二跳闭环 | 96 条中 invalid 与 clipped 各不超过 5；退化 query 不超过全部搜索 2% | 3 元 |
| G3 | 64×5 | 是否存在足够可学习的正确多搜探索 | 正确多搜至少 16/320，覆盖至少 8/64 题，learnable group 至少 8/64 | 10 元 |

G0 同一模型进程内让 direct HF、native manager、legacy manager 各生成 16 条，共 48 条 record；prompt token、raw text 和可解析率的硬门槛只比较 direct/native 的 16 对，legacy 仅作诊断，不用它覆盖历史结论。G2 必须绑定 G0+G1 的 exact evidence，G3 必须绑定 G2 evidence，并继续验证 G0 链路；不能用 `latest` 偷换前驱。

G2 刻意不设置 EM 最低门槛：该阶段只判断自主调用格式和 query 是否健康，EM、query relevance 与完整二搜链作为报告字段；真正决定是否存在足够可学习能力探索的是 G3。

三段外层 deadline 覆盖 handoff/data 复核、BM25、模型生成、分析和 evidence 发布。内部 generation timeout 会再减去固定 180 秒，为结果文件、哈希和清理留尾。BM25 子进程关闭继承的 phase-lock FD 9，并使用有界 TERM→KILL 清理。GNU `timeout` 的额外 KILL 宽限最多 120 秒，因此 2/3/10 元是工作额度，不是包含强杀宽限的绝对账单上限。

GPU 顺序为：

```bash
QWEN_NATIVE_GATE_STAGE=g0_g1 GPU_COUNT=2 AUTODL_PRICE_PER_HOUR=5.76 \
bash /root/autodl-tmp/search-r1/checkout/scripts/autodl/08_gpu_qwen_native_gate.sh

QWEN_NATIVE_GATE_STAGE=g2 \
QWEN_NATIVE_PREDECESSOR_EVIDENCE=<g0_g1-exact-marker> \
GPU_COUNT=2 AUTODL_PRICE_PER_HOUR=5.76 \
bash /root/autodl-tmp/search-r1/checkout/scripts/autodl/08_gpu_qwen_native_gate.sh

QWEN_NATIVE_GATE_STAGE=g3 \
QWEN_NATIVE_PREDECESSOR_EVIDENCE=<g2-exact-marker> \
GPU_COUNT=2 AUTODL_PRICE_PER_HOUR=5.76 \
bash /root/autodl-tmp/search-r1/checkout/scripts/autodl/08_gpu_qwen_native_gate.sh
```

## 9. AutoDL 状态与关机安全加固

本轮沿用既有 detached launcher、immutable attempt、persistent log、original exit code 和 watchdog，只增加 native pipeline 接入及必要加固：

- G0/G1、G2、G3 分别形成独立 attempt 和 evidence，不覆盖历史失败。
- 只有 terminal state、日志 sentinel、exit code 与 durable sync 完整后才允许请求关机。
- exit code 75 表示未取得全局 phase lock；即使旧锁后来释放，watchdog 仍明确跳过关机，避免关掉其他正在运行的任务。
- test、dry-run、keep-running、锁冲突、授权变化、证据不完整、sync 失败都保持实例运行。
- AutoDL guest 只调用经过 capability 绑定和重新校验的 `/usr/bin/shutdown`，且不附加参数。
- 关机 dispatch 成功不等于平台控制面已停止，仍必须在 AutoDL 控制台确认实例状态和计费。

自动化关机测试只调用临时假后端，从未在本地测试中执行真实 `shutdown`、`poweroff` 或 `systemctl poweroff`。

## 10. 验证结果

截至本报告生成时，本地验证结果为：

| 检查 | 结果 |
| --- | --- |
| 全量 pytest | `237 passed`，仅 1 条既有 Ray deprecation warning |
| AutoDL shell 测试 | `7/7` 通过 |
| shell 语法 | `39/39` 通过 |
| Python 无写入编译 | 17 个变更/新增 Python 文件通过 |
| Hydra 展开 | G1/G2/G3 三路 `--cfg job --resolve` 全部通过，无残留插值 |
| 关键配置 | response 500、max turns 4、group 2/3/5、双卡、HF rollout、SDPA、`top_k=20`、`presence_penalty=2.0` 均确认 |
| 静态云审计 | 临时索引下 `errors=0 warnings=0 files_checked=33` |
| Git 模式 | 两个新增 shell 文件均验证为 `100755` |
| 空白检查 | `git diff --check` 通过 |

这些验证没有联网、没有加载模型、没有调用 GPU、没有连接 SSH，也没有执行真实关机。它们证明接口、配置和安全状态机在本地闭合，但不替代 AutoDL 上的真实 tokenizer 物化、GPU 显存与运行时间测量。

## 11. 主要文件职责

| 文件 | 本轮职责 |
| --- | --- |
| `search_r1/llm_agent/tool_protocol.py` | Qwen3.5 schema、prompt、parser、conversation adapter |
| `search_r1/llm_agent/generation.py` | native 多轮 Agent loop、精确 token 续接、事件和 mask |
| `verl/utils/dataset/rl_dataset.py` | canonical raw chat 加载与渲染 |
| `verl/workers/rollout/hf_rollout.py` | top-k 与 turn-local presence penalty |
| `verl/trainer/main_ppo.py` | plain final answer EM 与成本奖励对齐 |
| `verl/trainer/ppo/ray_trainer.py` | 完整轨迹、loss mask、native PPO 核心门禁 |
| `scripts/data_process/search_mix.py` | native 数据物化、760 条 prompt 扫描、replay receipt 复验 |
| `scripts/autodl/02_cpu_prepare.sh` | 无卡增量准备、Hydra 展开、handoff 重封 |
| `scripts/autodl/08_gpu_qwen_native_gate.sh` | G0-G3 exact-stage 入口、deadline、lineage 与 evidence |
| `scripts/autodl/qwen_native_protocol_probe.py` | G0 direct/native/legacy 配对探针 |
| `scripts/autodl/qwen_native_gate_analysis.py` | 逐轨迹、逐题、汇总与 GO/NO-GO 分析 |
| `scripts/autodl/04_watch_and_shutdown.sh` | terminal evidence 后的授权关机与 exit-75 保护 |

## 12. 尚未完成与已知边界

1. 尚未在新 AutoDL checkout 执行 native CPU 增量重封，因此还没有新的 handoff digest。
2. 尚未执行 G0-G3，所以不能声称 Qwen3.5 原生协议已经在真实模型上通过，也不能给出新的 EM 或多搜比例。
3. native `R/B/C` 训练入口刻意未开放；必须先获得 G3 GO 并确定 PPO 采样/log-prob 合同。
4. BM25 TERM→KILL 已有测试，但极端情况下 KILL 后没有额外的“进程组已消失”硬断言；root 环境下风险较低，当前按最小实现原则保留为非阻断项。
5. guest shutdown 只能证明关机请求已派发，无法证明 AutoDL 控制面已经停止计费。

## 13. 后续执行顺序

1. 将本轮代码与报告提交并推送，确认远端 full commit。
2. 在无卡实例运行 `AUTODL_QWEN_NATIVE_INCREMENTAL=1`，只接受 `exit-code=0 + .success + cpu.ok/handoff 复验一致`。
3. 两卡实例依次运行 G0+G1、G2、G3，每段结束后先下载或检查完整 evidence，再决定是否继续。
4. 若任一阶段 NO-GO，归档轨迹并停止；不要通过重采样或事后改阈值制造 GO。
5. 只有 G3 GO 后，才选择 native 训练采样方案，重新注册 `R-mix60-native -> B/C-mix20-native`，并在同一 test 集比较 EM、搜索数、Utility 和共同答对题的搜索差。

面试时应把本轮概括为：先从失败轨迹定位协议错配，再用可切换 adapter 和分层门禁隔离模型能力、parser、检索与奖励因素；同时把数据、配置、轨迹、前驱和关机状态都做成可复算证据。不要把“工程门禁已实现”表述成“训练实验已经成功”。
