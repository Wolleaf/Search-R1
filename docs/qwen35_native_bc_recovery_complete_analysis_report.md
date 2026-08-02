# Qwen3.5 Native R60 后续 B20/C20 训练、恢复与完整结果分析

> 报告日期：2026-08-01
>
> 实验性质：单 seed、post-hoc exploratory comparison
>
> 当前状态：B20、C20 与六个外部评测均已完成；完整证据已在持久盘封存，小型结果包与关键训练证据已下载到本地
>
> 本地证据入口：[qwen35-native-bc-recovery-20260801](results/qwen35-native-bc-recovery-20260801/README.md)

## 1. 执行摘要

这轮实验已经完整结束，数据足以做配对分析。最终结论不是“C 全面胜过 B”，也不是“训练失败”，而是：

> **C20 是一个搜索效率明确改善、EM 点估计接近但能力差异仍不确定、综合效用呈弱正向趋势、协议稳定性仍未解决的探索性结果。**

最核心的事实如下。

| 维度 | 结论 | 证据 |
|---|---|---|
| 执行完整性 | 通过 | B/C 各 20 个更新、各 800 条训练轨迹；6 个外部评测均 `success/0`，共 1,024 条评测轨迹 |
| 分支可比性 | 基本通过 | B、C 都从同一个 exact R60 checkpoint 独立启动；seed、训练问题和 slot 完全一致 |
| 独立评测能力 | 未确认改善 | 512 个描述性合并样本的 EM 为 `36.33%→37.11%`；三个端点的 EM 区间均跨 0 |
| 搜索效率 | 明确正向 | 总搜索 `1320→1212`，减少 108 次、`-8.18%`；val 与 multihop 的下降有配对统计支持 |
| 后验效用 | 弱正向、未确认 | 三端点点估计都提高，但各自 95% CI 均跨 0 |
| 多跳风险 | 方向性风险，未确认 | 2Wiki 子集正确数 `35/128→28/128`，同时搜索 `435→408`；事后区间跨 0 |
| 协议质量 | 仍是主要瓶颈 | 外部评测 clipping 约 72%，非安全协议异常约 23%；C 没有修复这一问题 |
| 工程终态 | 受控成功 | 外层 `failed/203` 是成功封存后故意返回给关机 watchdog 的控制码，不是训练或评测失败 |
| 总体判定 | 保留 B/C，暂不晋升 C 为新能力基线 | C 可称为有希望的成本正则分支，仍需协议修复、多 seed 和预注册复验 |

这轮最值得保留的科学故事是：R60 先学到了搜索与回答能力，但伴随长输出和 native action 漂移；带有 correctness-gated 成本项的 C 分支随后表现出更低的搜索深度，尤其没有出现 B 后半程同等幅度的搜索膨胀，却没有可靠提高独立测试正确率，也没有修复格式与截断问题。

## 2. 实验身份与边界

### 2.1 B 和 C 分别是什么

- **共同 parent**：R60 的 `global_step_60`，SHA-256 为 `583771b131b6e2aa663ee2ef13ea6421524cdd7fed839f35fd6f9eb246a2c231`。
- **B20 / control**：从 R60 独立再训练 20 步，`cost_lambda=0`，线性/不带搜索成本的对照分支。
- **C20 / cost-aware gated**：从同一 R60 独立再训练 20 步，`cost_lambda=0.10`，`correct_only` gate。
- **不是串联训练**：C 不是接着 B 的 checkpoint 继续训练；两个分支都是 R60 的平行子分支。
- **不是 SFT+RL**：本轮仍属于 direct-RL 路线。此前讨论的 `SFT40→RL20→B/C` 是未来的独立实验，不能倒过来解释本轮结果。

C 的逐轨迹训练奖励为：

```text
train_reward = EM - (0.10 / 4) * searches * EM
```

也就是说，只在答对时扣除搜索成本。正式后验比较则统一使用：

```text
posthoc_utility = EM - (0.10 / 4) * searches
                = EM - 0.025 * searches
```

这个区别很重要：C 的 `correct_only` 设计避免了旧线性成本实验中“全错组也偏爱不搜索”的坍缩，但对全部答错的 group，所有 reward 仍为 0，因此它不能直接给“答错且搜索很多”的轨迹提供成本梯度。

### 2.2 训练与评测合同

| 项目 | B20 | C20 |
|---|---:|---:|
| 更新步数 | 20 | 20 |
| 每步轨迹数 | 40 | 40 |
| 训练轨迹总数 | 800 | 800 |
| 问题 group 数 | 160 | 160 |
| seed | 42 | 42 |
| parent | exact R60 | exact R60 |
| checkpoint | `global_step_20` | `global_step_20` |
| checkpoint SHA-256 | `8df3d6a…ad6d2` | `106b628…b2e` |

800 个 `(step, sample, slot)` 键完全一致。step 1–2 的 80 条 raw trajectory 在两个分支中完全相同，step 3 起才因前两次参数更新产生策略分叉。这既符合 RL 更新时序，也为“两个分支确实从同一 parent 启动”提供了行为侧证据。

正式外部评测采用：

- val：128 题；
- NQ-test：128 题；
- multihop：256 题，其中 HotpotQA 128、2WikiMultihopQA 128；
- greedy decoding、`group_size=1`、每题 1 个 rollout；
- B/C 按 `typed_sample_id` 严格一一配对且顺序一致；
- 最大执行搜索次数 4；
- sealed formal contract 预先记录 paired percentile bootstrap 10,000 次、seed 42、95% CI；
- 本报告另行追加 exact McNemar 与 exact sign test，作为 post-hoc sensitivity check，不属于 sealed formal contract。

这是一套确定性、单 rollout 的小规模比较，能回答“这两个 checkpoint 在这一组题上的配对差异”，不能估计不同训练 seed 或生成采样的全部方差。

## 3. 独立评测主结果

### 3.1 正确率：没有足够证据证明 C 提升能力

| 端点 | B | C | C−B 与 95% bootstrap CI | C 独对 / B 独对 | McNemar p |
|---|---:|---:|---:|---:|---:|
| val-128 | 79/128，61.72% | 83/128，64.84% | `+3.125pp [-2.344,+8.594]` | 9 / 5 | 0.424 |
| NQ-test-128 | 30/128，23.44% | 31/128，24.22% | `+0.781pp [-6.250,+7.813]` | 11 / 10 | 1.000 |
| multihop-256 | 77/256，30.08% | 76/256，29.69% | `-0.391pp [-5.469,+4.297]` | 20 / 21 | 1.000 |

三个端点的置信区间都跨 0，追加的 McNemar 检验也都未检出正确率差异。因此：

- 可以说 val 上 C 多答对 4 题，NQ-test 多 1 题，multihop 少 1 题；
- 不可以说 C 已经显著提高任务能力；
- 同样也没有证据表明 C 显著损害整体能力。

表中的 McNemar p 值是报告阶段追加的稳健性检查，不能解释为预注册检验。

### 3.2 搜索成本：本轮最可靠的正向结果

| 端点 | B→C 平均搜索/题 | C−B 与 95% CI | 相对下降 | 少搜 / 多搜 / 不变 | sign-test p |
|---|---:|---:|---:|---:|---:|
| val | `1.930→1.680` | `-0.250 [-0.406,-0.102]` | -12.95% | 24 / 8 / 96 | 0.0070 |
| NQ-test | `2.234→2.086` | `-0.148 [-0.375,+0.086]` | -6.64% | 27 / 18 / 83 | 0.233 |
| multihop | `3.074→2.852` | `-0.223 [-0.340,-0.105]` | -7.24% | 59 / 28 / 169 | 0.00117 |

val 与 multihop 的 sealed bootstrap CI 不跨 0，是本轮搜索下降的主要统计证据。作为事后敏感性检查，精确符号检验也支持下降；若对这三个追加检验机械使用 Bonferroni 阈值 `0.05/3=0.0167`，val 与 multihop 的 p 值仍低于阈值，但这不能把它们追认为预注册确认性检验。NQ-test 方向一致，样本证据不足。

总搜索次数分别是：

- val：`247→215`，少 32 次；
- NQ-test：`286→267`，少 19 次；
- multihop：`787→730`，少 57 次；
- 描述性合计：`1320→1212`，少 108 次，下降 8.18%。

### 3.3 后验效用：三个点估计均正，但尚未确认

统一使用 `U=EM-0.025*searches`。下表 utility 点估计来自 sealed summary，CI 则是在报告阶段从同一 `paired_results.csv` 按相同 seed 和 10,000 次 paired bootstrap 追加复算，属于 post-hoc uncertainty analysis：

| 端点 | B utility | C utility | C−B 与 95% CI |
|---|---:|---:|---:|
| val | 0.56895 | 0.60645 | `+0.03750 [-0.01895,+0.09688]` |
| NQ-test | 0.17852 | 0.19004 | `+0.01152 [-0.05899,+0.08242]` |
| multihop | 0.22393 | 0.22559 | `+0.00166 [-0.04941,+0.05049]` |

三端点的点估计都偏向 C，但 CI 全部跨 0。multihop 最能体现 trade-off：EM 下降 0.00391，少搜索贡献 0.00557，最终效用只净增 0.00166。

因此应写“效用有一致的弱正向趋势”，不应写“综合效用已显著提高”。

### 3.4 512 题合并只作描述，不作为正式主检验

| 指标 | B | C | C−B |
|---|---:|---:|---:|
| strict EM | 186/512，36.33% | 190/512，37.11% | +0.78pp |
| 搜索/题 | 2.578 | 2.367 | -0.211，-8.18% |
| utility | 0.29883 | 0.31191 | +0.01309 |
| action/题 | 3.662 | 3.512 | -0.150，-4.11% |
| token/题 | 1754.5 | 1677.1 | -77.4，-4.41% |
| 含 invalid 的轨迹 | 34.57% | 37.89% | +3.32pp |
| clipping | 70.90% | 71.68% | +0.78pp |

由于 val、NQ-test、HotpotQA 与 2Wiki 难度和任务结构不同，本节合并数字只描述这 512 行观测，不对合并总体赋予正式推断含义；正式结论仍必须按端点分别报告，不能用合并数字选择性放大结果。

## 4. C 到底改变了什么行为

### 4.1 它是“搜索深度调节器”，不是“是否搜索的 gate”

512 个样本的搜索次数分布：

| 执行搜索数 | B | C |
|---:|---:|---:|
| 0 | 1 | 1 |
| 1 | 141 | 173 |
| 2 | 127 | 130 |
| 3 | 47 | 53 |
| 4 | 196 | 155 |

无搜索轨迹仍只有 `1/512`，所以 C 没有学会大规模直接回答。它主要把一部分跑满 4 次预算的轨迹压缩到了 1–3 次：4-search 饱和从 38.28% 降到 30.27%。

配对看，C 更少、相同、更多搜索的样本分别是 `110/348/54`。少搜样本共节省 195 次，但多搜样本又增加 87 次，最终净省 108 次。C 不是对每题机械减一次搜索，而是在一部分问题上重分配深度。

### 4.2 大部分节省发生在失败轨迹上

| 配对结果 | 样本数 | C 更少 / 相同 / 更多 | C−B 净搜索变化 |
|---|---:|---:|---:|
| 两者都正确 | 150 | 20 / 119 / 11 | -12 |
| B 错、C 对 | 40 | 15 / 20 / 5 | -22 |
| B 对、C 错 | 36 | 8 / 18 / 10 | +7 |
| 两者都错 | 286 | 67 / 191 / 28 | -81 |

存在真正的好案例：

- 20 个“两者都正确”的样本中，C 少搜 26 次；扣除其他 both-correct 样本多出的搜索后，净省 12 次；
- 15 个“B 错、C 对”的样本同时少搜，共省 30 次。

但 67 个“两者都错”的少搜样本共省 128 次，占 gross savings 的 65.6%；both-wrong 类别贡献净节省 81/108。更准确的机制解释是：

> C 学会更早结束一部分无效的深层搜索，同时也产生了少量无损节省和少搜纠错；现有证据不能证明它已经稳定做到“用更少搜索找到更多正确证据”。

### 4.3 数据来源揭示了多跳关系链风险

| 端点 × 来源 | N | B→C 正确数 | B→C 搜索总数 |
|---|---:|---:|---:|
| val HotpotQA | 64 | `37→37` | `141→137` |
| val NQ | 64 | `42→46` | `106→78` |
| NQ-test | 128 | `30→31` | `286→267` |
| multihop HotpotQA | 128 | `42→48` | `352→322` |
| multihop 2Wiki | 128 | `35→28` | `435→408` |

C 在 NQ 和 HotpotQA 上均呈现“少搜、正确数方向性上升”，但 2Wiki 少搜 27 次的同时少答对 7 题。C−B EM 为 -5.47pp，报告阶段 10,000 次 paired bootstrap 的 95% CI 约为 `[-13.28,+2.34]pp`；该切片中 C 独对 9 题、B 独对 16 题，事后 exact McNemar `p=0.2295`（未做多重校正），没有确认回退。2Wiki 中大量问题要求完整走完父亲、祖父、创作者、所属实体等关系链，因此当前结果只提出“成本正则可能诱发过早停在中间实体”的风险假设，必须复验。

这个分层是事后探索，没有独立预注册显著性检验，所以不能作为确认性结论；但它是下一轮必须预先设置的风险切片。

### 4.4 代表性轨迹

| 样本 | B | C | 含义 |
|---|---|---|---|
| `nq:test:1187` | 3 搜，Trent Cotchin，正确 | 1 搜，同答案，正确 | 首次证据已足够，典型无损节省 |
| `nq:test:1571` | 3 搜，Anatomy，正确 | 1 搜，同答案，正确 | 单跳定义题及时停止 |
| `2wikimultihopqa:test:1514` | 3 搜，Ceawlin，正确 | 1 搜，Cerdic，错误 | 未走完祖父关系链的有害早停 |
| `nq:test:1749` | 4 搜，isocitrate dehydrogenase，正确 | 2 搜，citrate synthase，错误 | 在竞争候选上过早收敛 |
| `hotpotqa:test:1119` | 4 搜后仍请求搜索、被拒绝、无答案 | 1 搜，Saoirse Ronan，正确 | C 改善停止策略 |
| `hotpotqa:test:2980` | 4 搜后越界搜索、无答案 | 1 搜，Aristotle，正确 | 少搜并纠错 |
| `hotpotqa:test:2907` | 1 搜，June 22 1953，正确 | 4 搜后 invalid、无答案 | C 并非总是少搜，协议漂移仍会吞掉答案 |
| `nq:test:825` | 1 搜，Canberra，错误 | 4 搜，Melbourne，正确 | 多搜也可能纠正错误先验 |

这里的“无损”“纠错”和“改善停止”只按 strict EM 与执行搜索次数定义，不代表轨迹协议完全 clean；例如部分表中成功案例仍有 `response_clipped=true`。这些案例用于解释搜索机制，不能抵消第 6 节的协议风险结论。

## 5. 训练阶段动态

训练 rollout 不是独立测试集，但能解释策略怎样产生差异。

### 5.1 全 20 步汇总

| 指标 | B20 | C20 | C−B |
|---|---:|---:|---:|
| on-policy strict EM | 342/800，0.4275 | 368/800，0.4600 | +3.25pp |
| 平均搜索 | 2.1925 | 2.0325 | -0.160，-7.3% |
| 固定成本 utility | 0.37269 | 0.40919 | +0.03650 |
| 平均 response length | 1927 | 1781 | -146，-7.6% |
| all-wrong groups | 62/160 | 59/160 | -3 |
| 平均 ref KL | 0.03290 | 0.03084 | -0.00206 |
| 平均 entropy | 0.478 | 0.522 | +0.044 |

报告阶段按 160 个问题 group 聚类追加计算的描述性 95% 重采样区间为：EM 差 `[+0.0016,+0.0634]`、搜索差 `[-0.254,-0.066]`、固定成本 utility 差 `[+0.0050,+0.0680]`。这些区间条件于本次单 seed、自适应更新的训练路径；group 既不覆盖训练 seed 方差，也不是独立同分布的时间样本，不能当作通常意义上的训练效应 CI。它只说明本次 on-policy 序列方向一致，不能替代 held-out 泛化结论。

### 5.2 四个 5-step 窗口

| 分支 | steps | EM | 训练 reward | 固定成本 utility | 搜索 | all-wrong | ref KL | entropy | response tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B | 1–5 | .365 | .365 | .3151 | 1.995 | .525 | .0024 | .561 | 1747 |
| B | 6–10 | .465 | .465 | .4204 | 1.785 | .275 | .0236 | .550 | 1903 |
| B | 11–15 | .445 | .445 | .3859 | 2.365 | .325 | .0506 | .437 | 2046 |
| B | 16–20 | .435 | .435 | .3694 | 2.625 | .425 | .0549 | .365 | 2012 |
| C | 1–5 | .390 | .3718 | .3405 | 1.980 | .475 | .0076 | .546 | 1759 |
| C | 6–10 | .500 | .4790 | .4525 | 1.900 | .275 | .0128 | .585 | 1857 |
| C | 11–15 | .455 | .4338 | .4010 | 2.160 | .375 | .0463 | .545 | 1867 |
| C | 16–20 | .495 | .4729 | .4428 | 2.090 | .350 | .0567 | .413 | 1640 |

B 日志里的 utility 等于 EM，是因为它配置了 `cost_lambda=0`；上表为公平比较额外按固定 `lambda=0.10` 反事实复算。

C 自己的搜索不是单调下降的：`1.98→1.90→2.16→2.09`。观察到的差别是 B 后半程膨胀到 `2.37→2.63`，而 C 保持在约 2.1。最后五步，C 比 B 少 0.535 次搜索、短约 372 tokens、EM 高 6pp。因此本次单 seed 训练路径与“成本项像抑制后期过度搜索的正则”这一解释一致，而不像持续把搜索压到零的硬 gate；这仍不是多 seed 因果确认。

### 5.3 correct-only 奖励扩大了信号，但覆盖仍有限

- 非零 reward-std group：B `63/160`，C `75/160`；
- 非零 sequence-advantage 轨迹：B `315/800`，C `375/800`；
- C 中 10 个 all-correct group 可因搜索数不同获得新的成本 advantage；
- C 中仍有 `59/160` 个 all-wrong group，所有 reward 都为 0，成本项无法区分它们。

与此并存的是：C 的净搜索节省主要出现在最终失败轨迹上。由于 all-wrong group 恰好没有直接成本梯度，这种 both-wrong 节省只能是从有信号 group 学到的停止倾向向外泛化、一般策略漂移或两者共同作用，不能从奖励公式直接做单一因果归因。

### 5.4 数值稳定性

- 两条分支所有 reward、log-prob、advantage、loss、KL、entropy 和 grad 指标均为有限值；
- native batch、policy/environment mask、info loss mask 合同每步均通过；
- 最终 ref KL：B `0.07219`、C `0.08041`，同一量级且没有失控；
- `grad_norm>1` 的步数为 B 12/20、C 18/20；日志是裁剪前范数，配置 `grad_clip=1`，说明梯度裁剪在工作，不等于梯度爆炸；
- `ppo_kl=0`、`pg_clipfrac=0` 在 `ppo_epochs=1`、整批一次更新结构下是机械结果：计算 loss 时当前 policy 尚未更新。不能据此说“没有训练”；不同 checkpoint digest、非零梯度、ref KL 和 step 3 起的轨迹分叉都证明参数已更新。

### 5.5 训练时长不能直接当作算法加速

- B 总时长 17,558 秒，即 4:52:38；其中 step 20 inline validation 占 1,744 秒；扣除后训练约 4:23:34。
- C 总时长 11,988 秒，即 3:19:48；恢复版关闭 inline validation，保存 checkpoint 后统一做外部评测。
- 可比训练时长约降低 24%，C 的生成 token 总量少约 7.6%。

剩余差异可能来自轨迹长度、系统负载、缓存和运行时环境，不能把全部时间下降归因于成本奖励。

## 6. 协议与轨迹质量

### 6.1 工程安全合同是健康的

六份评测 trace 均满足：

- `search-r1.trajectory` schema v3；
- `info_mask_consistent=true`；
- 实际执行搜索不超过 4；
- terminal reminder 使用同一固定 hash；
- terminal prompt 的 policy token count 为 0；
- terminal 阶段请求的搜索均被拒绝，实际 terminal search 执行数为 0；
- B/C 数据、顺序、checkpoint 和 paired replay 合同一致。

所以 parser 安全边界、mask、search budget、retrieval 回填和配对评测基础设施没有出现新的漂移。

### 6.2 模型协议行为仍然很差

外部评测 512 题：

| 指标 | B | C |
|---|---:|---:|
| 含任意 invalid 的轨迹 | 177/512，34.57% | 194/512，37.89% |
| 排除 terminal 越界搜索后的非安全异常 | 115/512，22.46% | 116/512，22.66% |
| `invalid_thinking_prefix` 轨迹 | 107 | 81 |
| `invalid_thinking_prefix` 事件 | 148 | 116 |
| clipping 轨迹 | 363/512，70.90% | 367/512，71.68% |
| 无 clip 且无 invalid 的 clean 轨迹 | 87/512，16.99% | 59/512，11.52% |

C 的 thinking-prefix 错误减少，但 terminal answer 格式错误从 8 增到 28，并出现 11 个 `missing_native_action`；错误形态发生迁移，非安全异常总率几乎不变。C 的 terminal generation 更少，进入 terminal 后的合法 answer 机会条件率为 B `42/217=19.4%`、C `18/181=9.9%`。两个分母由各策略是否进入 terminal 决定，存在 selection/composition shift，因此这里只能作描述，不能单凭该比例推断 C 导致 terminal compliance 退化。

训练轨迹的方向略好，但仍不可称为已修复：

| 指标 | B | C |
|---|---:|---:|
| 含真实 generation clipping 的轨迹 | 496/800，62.0% | 378/800，47.25% |
| invalid action 事件 | 624 | 518 |
| forced-terminal 轨迹 | 311 | 253 |
| `invalid_thinking_prefix` 事件 | 545 | 397 |
| forced-terminal 中合法 answer | 132/311，42.4% | 96/253，37.9% |
| `invalid_terminal_answer_format` | 17 | 38 |

训练阶段 forced-terminal 中合法 answer 的机会条件率 B 为 `132/311=42.4%`、C 为 `96/253=37.9%`，同样受到“是否进入 forced terminal”的策略选择影响。结合全量 `invalid_terminal_answer_format` 从 17 增到 38，可以说终局错误形态存在风险，但不能把这两个条件率单独解释成因果退化。

因此正确表述是：C 降低了训练阶段的总体 clipping、invalid 和 thinking-prefix 失败，但终局答案格式存在局部风险；在独立评测上，整体协议洁净度没有改善。

这里的 `response_clipped` 指至少一次真实 generation 达到单轮 500-token 上限。日志里恒定的 `response_length/clip_ratio=.025` 只是批内动态 tensor 宽度指标，不能代替 trace 级真实截断率。

## 7. 为什么这次运行接近七小时

恢复外层 attempt：

```text
/root/autodl-tmp/search-r1/state/attempts/gpu/20260801T041947Z-7015-12411
```

从 `2026-08-01T04:19:47Z` 到 `11:09:33Z`，总计 24,586 秒，即 **6:49:46**。耗时主体不是单一训练卡死，而是顺序完成 C20 和六个评测：

| 阶段 | 耗时 |
|---|---:|
| C20 recovery 训练 | 11,988 秒，3:19:48 |
| B-val | 1,445 秒 |
| C-val | 1,286 秒 |
| B-NQ-test | 1,393 秒 |
| C-NQ-test | 1,456 秒 |
| B-multihop | 3,203 秒 |
| C-multihop | 3,141 秒 |
| 六个评测合计 | 11,924 秒，3:18:44 |

C 训练与评测合计已经是 6:38:32；其余约 11 分钟用于启动/停止 retriever、校验、汇总、封存和阶段切换。因此“接近七小时”是当前逐分支、逐端点串行执行的合理结果，不是训练在最后一步挂死。

若按脚本预算采用的 5.76 元/小时估算：

- 本次 recovery 外层约 39.34 元；
- 7 月 31 日首次失败外层 9:19:02，约 53.67 元；
- 两次合计约 93.00 元。

这些只是按时长和配置费率计算的工程估算，不是云厂商账单或控制台结算证明。

## 8. 首次 B/C 失败与本次恢复

### 8.1 第一次 attempt 实际发生了什么

7 月 31 日首次 B/C 外层 attempt：

```text
/root/autodl-tmp/search-r1/state/attempts/gpu/20260731T060828Z-1554-11705
terminal=failed
exit_code=124
```

这次不能笼统叫“B/C 都失败”：

- B20 已完整成功，产生 800 条轨迹、20 步训练历史和有效 `global_step_20` checkpoint；
- 旧 C 也跑到了 800 条训练轨迹，但第 20 步先进入 inline validation，checkpoint 尚未保存；
- 旧 C20 单个训练 job 的预算/超时上限约 15,625 秒，短于 B 的实际总耗时 17,558 秒，本身就缺乏安全余量；
- 因 timeout 发生在“最后一步训练完成、验证进行中、保存之前”，旧 C 没有可采用的最终 checkpoint；
- 六个正式外部评测也都没有完成。

根因是超时预算与“先验证、后保存”的调度组合，不是磁盘写满、OOM、NCCL、模型数值崩溃或检索器科学失败。

### 8.2 恢复方案做了什么

恢复 runner 没有重算已验证成功的 B，而是：

1. 校验并复用 B 的完整内层成功 run 与 checkpoint；
2. 明确把第一次失败的 outer attempt 标为 `source_failed_outer_adopted=false`；
3. 明确忽略无 checkpoint 的旧 C，`old_c_adopted=false`；
4. 从 exact R60 重新独立训练 C，而不是从旧 C partial 或 B 接续；
5. 把 C 预算提高到 40 元 / 25,000 秒级别；
6. 设置 `save_freq=20`、`test_freq=-1`、`val_after_train=false`，确保先保存最终 checkpoint；
7. checkpoint 成功后再分别执行 B/C 的 val、NQ-test、multihop；
8. 所有阶段成功后生成 lineage、paired summary 和 evidence manifest，再返回受控状态给关机 watchdog。

B 使用 checkout `f8c1cd7…`，恢复 C 使用 `ffda960…`。代码差异审计表明，核心 trainer 的相关变化是把训练末尾 validation 变成可关闭的调度开关，发生在最后一次参数更新之后，不改变 rollout、advantage 或 actor update 语义。不过两个 checkout 不是 bit-identical，仍应在限制中保留这一事实；下一次确认性实验应让 B/C 使用完全相同 checkout。

### 8.3 为什么最终显示 `failed/203` 反而是受控成功

最终 phase log 明确记录：

```text
Completed and sealed B/C recovery comparison
Returning controlled outer status 203 for the pinned shutdown watchdog
AUTODL_PHASE_TERMINAL state=failed exit_code=203
```

`203` 是脚本合同中的 `controlled_outer_exit_code`：工作完成并封存后，故意交给既有 fail-safe shutdown watchdog。它不能被解释为训练失败。真正的科学执行状态来自 8 个内层 run，全部为 `success/0` 且 `timed_out=false`。

关机证据显示 guest 内 `/usr/bin/shutdown` 已 `backend_exit_code=0`。同时 `provider_control_plane_confirmed=false`，所以我们能证明 guest 关机请求已成功派发，不能仅凭该文件声称云厂商计费控制面已确认停止。用户实际观察实例关机并以无卡模式重新启动，补足了操作层终态。

## 9. 从最初失败到本轮结果的完整叙事

本项目的关键不是“一次训练调参”，而是逐层把工程混杂和模型科学问题分开。

### 阶段 0：先把云端训练链路从“能启动”修到“能可信完成”

7 月 19–20 日的最初问题属于 cloud/runtime 与证据链阻塞，还不能拿来判断模型方法：external BM25 corpus 与 CPU→GPU handoff 路径需要固定；Qwen3.5 的 FSDP wrap、optimizer state offload/load 顺序、GPU gate 和关机 watchdog 也需要补齐。对应修复链记录为 `3e54c4c→c979400→1779a9c→345ad0d`。

尤其是 1-step smoke 覆盖不到 Adam 状态在第一次 `optimizer.step` 后初始化、第二次 update 才暴露的路径，因此工程门禁被升级为至少 2-step，并把 optimizer state 的加载与卸载包围在真正的 update 周围。当前紧凑归档没有保留最初异常全文，所以本报告不杜撰具体 OOM 或错误码；可验证的结论只是：初始训练链路曾崩溃或门禁不足，完成上述修复和两步验证后，才有资格进入模型科学比较。

### 阶段 A：旧 direct RL 证明可学习，成本奖励却暴露无搜索坍缩

7 月 20 日的 NQ-only 缩小实验首先给出一条正面基线：A→R60 的 EM 从 3.91% 提升到 16.41%，说明预算缩小版 direct outcome-RL 确实能学习搜索与作答，而不是训练链路只会产出随机权重。

早期 NQ-only 线性成本 C 相对 B 把搜索量降低 96.95%，却把 no-search 推到 98.44%，EM 从 17.97% 降到 7.03%，utility 从 0.1541 降到 0.0695。原因是普通线性成本会让 all-wrong GRPO group 也偏好“错误但不搜索”的轨迹。随后改成 correctness-gated 后，no-search 回到 0%，EM 恢复到 14.06%，但仍低于 B 的 17.97%；搜索总数 137 反而高于 B 的 131，utility 0.1139 也低于 0.1541。

这一步建立了当前 `correct_only` 设计的动机：成本不能无条件奖励错误轨迹。

### 阶段 B：数据分布和旧 XML 协议造成伪科学信号

search-opportunity 审计在 HotpotQA+2Wiki 256 题上发现，旧 B 有 248 题只搜索一次；仅 8 条二搜轨迹全部错误、截断或非法，其中 7/8 重复 query，正确题平均恰好一次搜索。也就是说，当时成本项最容易学到的仍是从一次搜索降到零，而不是从冗余多搜降到必要多搜，因此门禁给出 NO-GO。

数据随后从 NQ/Hotpot 1:1 调整为 37.5%/62.5%，Hotpot 内 comparison/bridge 从 200/120 改为 56/264，增加需要多跳 bridge 的学习机会，同时保留 NQ。这个配比是在正式训练前依据 BM25+384-token `visible_observation` 漏斗重登记：comparison 候选只有 74 条严格合格，而 bridge 有 766 条；标准没有为凑数而放宽，manifest、ledger 与 digest 均封存。

但 7 月 23 日 grouped probe 只有 7/320 正确，58/64 group 全错，invalid 为 185/320、clipped 为 102/320；312 次搜索中有 210 次 query 是字面量 `query`、`and` 或空值。原 outer 因空 query 触发分析器 exit 1，但 320 条 eval 已完整；CPU 修复只是把空 query 纳入科学失败分类，没有重跑 GPU。根因不是简单的“模型不会搜索”，而是 Qwen3.5 原生 tool calling 被强迫走 legacy XML，prompt、parser、答案边界和 sampling 合同互相错配。

### 阶段 C：native tool adaptation 逐步清除工程混杂

项目随后建立 Qwen native schema、原始 token replay、tool-response mask、严格短答案和 on-policy sampling。

- native-v1 G2 已出现 34 条人工内容正确和 13/32 个潜在 mixed group，但正式 EM 为 0，因为整段自由文本被当作答案；
- v2 恢复唯一 `<answer>短答案</answer>`，统一 `top_k=0`，forced-search stress 得到 2/32 strict EM、88 次健康检索，但 21/32 含非法动作；
- 恢复原版自主 thinking 语义后的首次 v3，在第 2 条 trace 因 `answer boundary is inconsistent` 崩溃。根因是 reasoning 中的 marker 被误当 action，以及单 token 解码跨越逻辑答案边界；这是纯工程失败；
- 修复 action region 与 policy token prefix 后，v3 32 条自主轨迹达到 6/32 strict EM、91 次健康检索，真正问题收敛为“已经看到答案却继续搜索”；
- 恢复 upstream terminal generation 后，19 个 terminal opportunity 仅 1 个回答，15 个仍请求搜索；terminal 路径工作正常，模型不服从停止机会。

### 阶段 D：v4 修复 terminal、Gold 和证据门禁

v4 加入 answer-only terminal reminder，并保证 reminder token 不进 policy loss；同时对 multi-gold 和错标样本做确定性审计，剔除 51 条有缺陷记录再按原分布补位；WandB 门禁也从“文件存在”升级为必须有完整 step history、有限指标、summary 与 exit record。

Gate-v4 到 Gate-v5 的 32 条生成完全一致，NO-GO 变 GO 不是模型突然变好，而是门禁不再把“需要 RL 学会的 terminal compliance”错误当作训练前工程硬门。

两步 smoke 随后完成两次真实全参数 GRPO update，所有数值有限，7/16 group 有 mixed reward，证明训练链路 operational；它只证明“可以训练”，不证明两步已经改善模型。

### 阶段 E：R60 学到了能力，也产生了协议漂移

R60 完成 60 次全参数更新、2,400 条轨迹。在同一固定 val-128 上，相对从同一 parent 独立启动的 smoke step-2 endpoint：

- strict EM `37.50%→58.59%`；
- 平均搜索 `3.117→1.992`；
- no-search `5→1`。

这证明 direct outcome-RL 确实学到了搜索与作答，并没有坍缩成不搜索。但增益主要来自 NQ，Hotpot 只增加 2 题；且 step 52 后 KL、生成长度、clipping 和 invalid 同步上升。R60 只有 `global_step_60` 模型权重，没有 optimizer/trainer state，不能做严格意义的断点续训；B/C 因而是从该权重启动的新注册 run。

### 阶段 F：G3 给出真实科学 NO-GO

R60 G3 完成 320/320 轨迹、724 次搜索，strict EM 为 151/320=47.19%。其中正确多搜轨迹 98 条，覆盖 34/64 问题，13/64 有成本对比，说明模型确有多搜能力和成本学习机会。

但正式门禁失败：clean learnable group `5/64<8`、clipping `46.88%>5%`、invalid `43.75%>5%`。clean 轨迹 EM 为 71.15%，dirty 仅 24.39%，表明瓶颈不是完全没有知识或检索能力，而是长输出与协议稳定性。外层 verifier 的 tokenizer 派生字段和问号规范错误是另一个事后工程问题，不改变 G3 的科学 NO-GO。

### 阶段 G：B/C 是 G3 NO-GO 后的探索，不是预注册主实验

因此 B/C 的身份是：在已知 G3 协议风险下，探索“correct-only 成本项能否降低搜索而不再次 no-search collapse”。第一次 B/C 因调度超时留下完整 B 和不可采用的 C partial；本次恢复重用 B、从 R60 重训 C、完成六个外部评测并封存证据。

最终结果回答了这个探索问题：

- no-search 没有坍缩；
- 搜索深度确实下降；
- 正确率没有得到显著改善或显著损害的证据；
- 2Wiki 出现未确认的方向性有害早停风险；
- 协议漂移仍然存在。

这是一条完整、可辩护的研究叙事：先识别错误奖励，再修数据与 native 协议，再证明训练可用，再观察 R60 的能力—稳定性张力，最后用 B/C 证明成本正则能改变预算分配，但尚不足以解决总体可靠性。

## 10. 证据、完整性与存储状态

### 10.1 本地封存

本地证据目录为：

```text
docs/results/qwen35-native-bc-recovery-20260801/
```

其中 `raw-result/` 是远端最终结果根的完整小型证据包，共 25 个文件、56,640,731 bytes（约 54.02 MiB）；还下载了 outer terminal/shutdown 文件以及 B/C 的关键训练日志、resolved config、run contract、lineage 和 trace manifest。

本地校验结果：

- `raw-result/evidence.sha256` 自身 SHA-256 为 `8ec72644d618c194a248e82965fcf4b7473e69acad0a5c5cd519f9df2c03827c`；
- 与远端 outer `evidence-digest` 及单独下载的 `evidence-marker.ok` 完全一致；
- `raw-result/` 中 manifest 覆盖的 24/24 payload 在本地复算匹配；
- 另行下载的 B/C 12 个关键训练证据文件在完整 manifest 中 12/12 匹配。

2026-08-01T11:29:10Z 又在远端当前持久盘做了独立的全量字节级复验：

- R60 checkpoint tree digest `583771b…c231`：通过；
- B20 checkpoint tree digest `8df3d6a…ad6d2`：通过；
- C20 checkpoint tree digest `106b628…b2e`：通过；
- 完整 `evidence.sha256` 的 166 项执行 `sha256sum --quiet -c`：返回码 0，无失败项。

因此当前持久盘上的三个关键 checkpoint 与全部封存证据均无可见静默损坏。

有一个轻微但必须披露的封存覆盖缺口：166 项 manifest 包含复用 B 和历史失败 C 的 `terminal/exit-code`，但没有纳入新 C20 与六个新 eval 的 7 组 `terminal/exit-code` 小文件。这七个阶段的文件在现场均为 `success/0`，并且各自的 `run.env`（含 `finished_at`、`timed_out=false`）、resolved config、日志、完整 trace/manifest 和 W&B receipt 已被封存，可独立支持“运行确实完成”；只是这 14 个终态小文件本身不受最终 manifest 的字节级防篡改保护。为了保持原始 seal 身份，本轮不事后改写 manifest，后续 runner 应补齐这一项。

还有一个纯元数据缺口：六个 eval 的 `run.env` 将 `eval_data_file/hash` 记为 `unavailable`。不过各自 sealed resolved config 明确绑定 `val_128.parquet`、`nq_test_128_native_v4.parquet` 或 `multihop_eval_256_native_v4.parquet`，这三份 parquet 本身均在 166 项清单中通过哈希；六份 trace 的 checkpoint digest 也逐条与 B20/C20 复算 digest 相同，B/C 样本身份与顺序完全一致且无重复。因此这是 `run.env` 字段覆盖不足，不是评测数据来源不明或配对身份失效。

完整文件说明见[证据包 README](results/qwen35-native-bc-recovery-20260801/README.md)，原始 paired summaries 分别见 [val](results/qwen35-native-bc-recovery-20260801/raw-result/paired-val/summary.md)、[NQ-test](results/qwen35-native-bc-recovery-20260801/raw-result/paired-nq_test/summary.md) 与 [multihop](results/qwen35-native-bc-recovery-20260801/raw-result/paired-multihop/summary.md)。

### 10.2 远端磁盘

审计时远端持久盘约 150G，总使用 132G，剩余 19G，使用率 88%。B/C 两个 run 目录各约 9.0G，最终结果汇总仅约 55M。

目前不应删除 B、C、R60、G3 或本次结果证据：它们分别承担对照 checkpoint、成本分支 checkpoint、共同 parent、NO-GO 门禁和最终配对分析的可复核角色。若后续需要腾空间，应先生成“可删候选—引用关系—digest—备份位置”清单，再仅清理明确未被采用的 partial/重复缓存；本次分析没有执行删除。

## 11. 限制与威胁

1. **post-hoc exploratory**：B/C 是在 G3 NO-GO 后提出的探索，不是原预注册确认性实验。
2. **单训练 seed**：两个分支都只有 seed 42；同 seed 增强配对可比性，但不能估计训练随机性。
3. **单 rollout、greedy**：每题只有一个确定性生成，无法估计 checkpoint 内生成方差。
4. **端点规模小**：128/128/256 导致 EM 翻转区间较宽。
5. **多指标、多切片**：来源分层和代表案例用于机制分析，不能事后升格为主显著性结论。
6. **checkout 非完全一致**：恢复 C 的 checkout 比 B 多了终点 validation 调度修复；训练更新语义经审计未变，但不是 bit-identical 可执行环境。
7. **训练目标与报告效用不完全一致**：C 只对正确轨迹收成本，报告效用对所有轨迹收成本。
8. **协议混杂仍高**：64%–83% 的外部端点轨迹被真实 generation clipping 影响，invalid 也高，搜索差异可能部分由格式失败或终止失败驱动。
9. **严格 EM 局限**：它是训练和主评测的既定指标，必须保留；同时会对 alias、日期和冗长表达敏感。
10. **预算缩小复现**：当前使用 post-trained Qwen3.5-2B、BM25、两卡和小数据/短训练，不是 Search-R1 论文的 Base 模型、dense retriever 和完整 benchmark 复刻。
11. **没有 provider 账单证明**：关机只证明 guest shutdown 成功派发；成本是估算。
12. **封存元数据小缺口**：新 C20 和六个新 eval 的 `terminal/exit-code` 未收入 166 项 seal；六个 eval 的 `run.env` 也未直接记录数据文件/hash。其他 sealed config、trace、parquet、receipt 和 lineage 足以交叉证明完成与配对身份，但下一版 runner 应显式补齐。

## 12. 后续建议

### 12.1 现在应如何处理 B/C

- 保留 R60、B20、C20 与 G3 checkpoint/trace/evidence；
- 将 B 继续作为能力对照基线；
- 将 C 标记为 `efficiency-positive / capability-inconclusive` 的候选，而不是直接替换 B；
- 暂不盲目增加 RL 步数或仅提高 token 上限，因为当前主要风险是协议/长度稳定性，而不是训练没跑够。

### 12.2 先做协议稳定，再做确认性成本实验

优先目标应是：

1. 降低单轮 500-token clipping；
2. 提高 terminal answer-only 服从率；
3. 对 `invalid_thinking_prefix`、missing action、terminal answer format 建立明确且可审计的训练信号；
4. 保持 terminal search 的硬拒绝与 environment-token mask；
5. 不通过放宽 strict EM、删掉失败轨迹或增大搜索上限来掩盖问题。

### 12.3 SFT40→RL20→B/C 可以做，但必须是独立新实验

先做少量格式/停止行为 SFT，再做 RL，是合理的下一条假设，因为 direct RL 已经证明会增强能力，也证明会累积协议漂移。SFT 数据可以由现有轨迹构建，但必须：

- 只选协议干净、答案边界正确、检索证据充分的样本；
- 为“继续搜索”和“提交答案”同时保留正例，避免把模型 SFT 成机械早停；
- 排除正式 held-out 评测题及其近重复，特别是 G3、val、NQ-test、multihop；
- 保留 source、question id、teacher/修订来源和去重 digest；
- 先做小规模格式 smoke，确认 clipping、terminal compliance、zero-search 与 multi-search 都在合理范围；
- 再注册 `SFT40→RL20`，从这个新 parent 平行训练 B/C。

现有[后续方案](qwen35_native_sft_rl_bc_followup_plan.md)应作为这一新实验的起点，但不能与本轮 direct-RL B/C 合并成同一个实验身份。

### 12.4 下一轮确认性设计

- 在运行前指定 primary endpoint、主指标与停止规则；
- 用至少多个训练 seed，并固定完全相同 checkout；
- 对 EM 设置非劣界，同时把平均搜索或 utility 设为共同主指标；
- 预先把 2Wiki/关系链深度设为风险切片；
- 同时报 clean-only 与全量结果，但不能用 clean-only 替换主结果；
- 若目标真的包含“无需检索”，必须加入可验证的 0-search 数据与决策目标；当前 C 只学到减深度；
- 对 correct-only 的盲区，可探索只在正确轨迹内部做成对成本排序，并单独加入格式/terminal 奖励，避免再次奖励错误不搜索。

## 13. 可以如何对外表述

建议论文/项目复盘使用：

> 在相同 R60 parent、相同训练问题和单 seed 设置下，C20 在 512 个确定性配对样本上减少 108 次检索，并将 4-search 饱和率从 38.3% 降到 30.3%。val 与 multihop 的平均搜索下降得到配对统计支持；strict EM 从 36.3% 变为 37.1%，各端点均未检出显著 EM 差异。效用点估计在三个端点上均为正，但区间跨零。搜索节省主要来自共同失败轨迹，且 2Wiki 子集提出了未确认的正确率回退风险。协议安全层保持健康，但 clipping 与 native action 异常仍高。因此本轮提供了“correctness-gated 成本奖励与预算下降一致”的单 seed 探索性证据，尚未证明总体能力、非劣性或可靠性得到稳定提升。

不应使用以下表述：

- “C 显著提升了准确率”；
- “C 已全面优于 B”；
- “协议漂移已经解决”；
- “复现了原 Search-R1 论文结果”；
- “七小时是在异常卡死”；
- “外层 failed/203 代表训练失败”；
- “本轮已经验证 SFT+RL 优于 direct RL”。

## 14. 最终判定

本轮 B/C recovery **工程完成、证据完整、可以进入正式复盘**。科学上：

- C 的搜索效率改善成立；
- C 的能力提升没有被确认；
- 效用只有正向趋势；
- no-search collapse 没有复发；
- 多跳关系链出现了未确认的方向性早停风险；
- 长输出、clipping、terminal compliance 和 native action 格式仍是首要瓶颈。

因此最合理的决策是：**保留 B 作为能力基线、保留 C 作为成本正则候选，冻结本轮结论；下一阶段先做小规模协议型 SFT 验证，再决定是否启动独立注册的 SFT40→RL20→B/C。**
