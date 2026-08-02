# Qwen3.5 Native B/C 探索实验交接手册

> **历史文档，执行任务已完成。** 本文保留 B/C 启动前的决策与操作边界，不再代表当前“下一步”。B/C recovery、A/R 补评和四模型封存均已完成；当前总交接见 [Qwen3.5 Native 完整实验交接文档](../final/qwen35_native_complete_experiment_handoff.md)，最终结果见 [Qwen3.5 Native A/R/B/C 最终结果分析](../final/qwen35_native_arbc_final_results_analysis.md)。除非注册全新的复现实验，不要照本文重新启动 GPU。

## 1. 交接目标与结论边界

下一会话的唯一目标是：**不重训 R60，直接从同一个已封存 R60 checkpoint 分别训练 B20 与 C-gated20，再完成配对评测、结果归档和自动关机。** B 是纯 EM control，C 是 correctness-gated 成本奖励；两者除奖励外必须完全对称。

本轮属于 `post-hoc exploratory`，不是原预注册流程的确认性分支。原因是 G3 的 cost-contrast 已达到 `13/64`，但 capability 因 clipping、invalid 和 clean learnable group 未通过而为 NO-GO。简历可以突出“完成端到端 Agent RL、轨迹审计和成本奖励对照”，但报告、代码和面试回答不得把 NO-GO 或负向 B/C 结果写成正向显著提升。

## 2. 当前不可变事实

| 项目 | 值 |
| --- | --- |
| 本地分支 | `experiment/hotpot-search-gate` |
| 当前已推送 HEAD | `824dfb504dc22678080b68691ade80770c61bf20` |
| R60 训练 checkout | `f8c1cd7e87078d07385f74ca8710add5d5f79c06` |
| R60 run | `/root/autodl-tmp/search-r1/runs/reproduce/attempts/20260728T092332Z-2946-8453` |
| R60 checkpoint | 上述目录的 `checkpoints/actor/global_step_60` |
| R60 tree SHA-256 | `583771b131b6e2aa663ee2ef13ea6421524cdd7fed839f35fd6f9eb246a2c231` |
| R60 marker | `/root/autodl-tmp/search-r1/manifests/qwen-native-training-r60-only/20260728T092026Z-2906-16060.ok` |
| G0/G1 marker | `/root/autodl-tmp/search-r1/manifests/qwen-native-gate/20260728T044634Z-2051-4045.ok` |
| smoke marker | `/root/autodl-tmp/search-r1/manifests/qwen-native-training-smoke/20260728T061246Z-1316-15067.ok` |
| G3 outer / inner | `20260729T073211Z-3516-18146` / `20260729T073654Z-3558-15376` |
| G3 trace SHA-256 | `0020eb4b2f35fc16ea1115b1f5fde5ad01f4451e3a0a25a59776d9cd55051075` |
| 固定训练参数 | 2 GPU、batch 8、group 5、20 steps、response/observation 500、search budget 4 |
| B reward | `cost_lambda=0`, `cost_reward_mode=linear` |
| C reward | `cost_lambda=0.10`, `cost_reward_mode=correct_only` |

R60 只有 `global_step_60`，没有中间 checkpoint，也没有 optimizer/scheduler state。B/C 都应把 R60 HF 权重作为独立初始模型，各自创建新 optimizer；不得让 C 继承 B。

本地存在以下用户未跟踪内容，接手时不得删除、覆盖或顺手提交：

```text
docs/history/qwen35-native-arbc-202607-202608/stages/qwen35_native_v3_g0_g1_trajectory_analysis.md
docs/history/qwen35-native-arbc-202607-202608/stages/qwen35_native_v3_terminal_rollout_g0_g1_analysis.md
docs/results/qwen35-native-v3-g0-g1-20260726/
tmp/
```

## 3. 为什么不能直接运行现有 main

不要运行：

```bash
QWEN_NATIVE_TRAIN_STAGE=main bash scripts/autodl/09_gpu_qwen_native_train.sh
```

该入口会从 base **重新训练 R60**、重复 G3 和 A/R endpoint，并仍要求 G3 capability GO 才启动 B/C。它既浪费约 9.5 小时，也不会满足“无论 G3 是否 GO 都训练 B/C”的新实验合同。

下一会话应新增独立入口 `scripts/autodl/12_gpu_qwen_native_bc_only.sh`，复用 `09` 中已经验证的 `run_job`、checkpoint、trace、WandB、lineage、paired-eval 和 evidence 逻辑，但只执行：

```text
校验现有证据
  -> B20 from R60
  -> C-gated20 from the same R60
  -> B/C val-128
  -> B/C NQ-test-128
  -> B/C multihop-256
  -> 三套 paired summary
  -> evidence seal
  -> watchdog 安全关机
```

不要通过伪造 `branch_training_authorized=true`、修改旧 G3 JSON 或回写旧 outer attempt 来绕过门禁。新 runner 应显式记录：`experiment_class=post_hoc_exploratory`、`operator_override=true`、原 G3 decision、R60/G3 digest 和本次独立合同版本。

## 4. 最小实现范围

### 4.1 必须实现

1. 新增 B/C-only runner 和对应 shell 测试；不要大改 `09`。
2. `--cpu-prepare` 只读校验旧 checkout、CPU handoff、G0/G1、smoke、R60、native-v4 数据和现有 G3 trace，不安装、不下载、不重建数据。
3. 运行时将既有 control/C resolved config 的 parent 路径替换为 exact R60 checkpoint，并同步验证所有 OmegaConf 插值字段。
4. B/C 必须串行执行。每个全参训练都会同时使用两张 GPU，不能并行各占一张卡。
5. 每个分支保留 `train.log`、20-step checkpoint、完整 train trace、resolved config、lineage、WandB offline history、原始 exit code和 SHA-256。
6. 三套 endpoint 均使用 group 1、greedy、seed 42；保存逐题 EM、搜索次数、轨迹、答对/答错清单和 B/C paired summary。
7. 新 evidence marker 只能在所有期望文件校验成功后原子发布；旧 R/G3 attempt 始终只读。
8. watchdog 必须识别新的 B/C-only 合同，或由新 runner 提供同等严格的 exact-attempt verifier。成功或失败都应先持久化原始终态，再请求 AutoDL `/usr/bin/shutdown`。

### 4.2 不要顺手修改

- 不改 prompt、Qwen native tool-call、parser、terminal reminder 或 BM25。
- 不改 batch/group/response/search budget、temperature/top-p、学习率、KL 或训练数据。
- 不提高 `lambda`，本轮固定 `0.10`，否则无法与此前设计对齐。
- 不重跑 G0/G1、smoke、R60 或 G3，不删除旧 checkpoint。
- 不因负结果自动换 seed、加步数、放宽 EM 或重试科学实验。

## 5. CPU 阶段操作

接手后先在本地执行：

```powershell
git status --short --branch
git log -5 --oneline --decorate
git diff --check
```

完成 runner、测试和本文引用更新后，运行相关 shell/Python 测试，提交并 push 当前实验分支。不要把未跟踪的用户文件包含进 commit。随后通过 SSH skill 将新 runner 部署到 checkout 外的 `/root/autodl-tmp/search-r1/operator/`；因为旧科学 checkout 固定为 `f8c1cd7...`，不得 pull、切分支或零散覆盖它。

所有远程操作必须使用 `ssh-skill`，禁止直接调用 `ssh/scp`。最近使用的别名是：

```text
searchr1-autodl-bjb2-21585
```

本次交接时该别名无法连接，视为实例已停机或端口已失效。新会话应让用户提供当前实例连接信息，再用 SSH skill 创建/更新别名。密码只保存在本机 SSH 配置，不得写入仓库、日志或本文。

典型调用形式（Windows 本机）：

```powershell
python C:/Users/Admin/.codex/skills/ssh-skill/scripts/ssh_execute.py <alias> "<read-only command>" --timeout 60
python C:/Users/Admin/.codex/skills/ssh-skill/scripts/ssh_upload.py <alias> "<local-file>" "/root/autodl-tmp/search-r1/operator/<file>"
```

先只读确认 R60 checkpoint/digest、旧 marker、磁盘空间和 checkout 状态，再上传。CPU 命令的目标接口应为：

```bash
QWEN_NATIVE_PROTOCOL_GATE_EVIDENCE=/root/autodl-tmp/search-r1/manifests/qwen-native-gate/20260728T044634Z-2051-4045.ok \
QWEN_NATIVE_SMOKE_EVIDENCE=/root/autodl-tmp/search-r1/manifests/qwen-native-training-smoke/20260728T061246Z-1316-15067.ok \
QWEN_NATIVE_R60_EVIDENCE=/root/autodl-tmp/search-r1/manifests/qwen-native-training-r60-only/20260728T092026Z-2906-16060.ok \
bash /root/autodl-tmp/search-r1/operator/12_gpu_qwen_native_bc_only.sh --cpu-prepare
```

CPU 成功标准不是“脚本没报错”，而是新 receipt/manifest、`exit-code=0`、唯一 `.success` 和全部 digest 校验通过。CPU receipt 必须绑定 operator runner SHA，而不是假装旧 checkout 包含新代码。

## 6. GPU 阶段操作

用户挂载两张 5090 级 GPU 后，只启动一次 B/C-only：

```bash
QWEN_NATIVE_PROTOCOL_GATE_EVIDENCE=/root/autodl-tmp/search-r1/manifests/qwen-native-gate/20260728T044634Z-2051-4045.ok \
QWEN_NATIVE_SMOKE_EVIDENCE=/root/autodl-tmp/search-r1/manifests/qwen-native-training-smoke/20260728T061246Z-1316-15067.ok \
QWEN_NATIVE_R60_EVIDENCE=/root/autodl-tmp/search-r1/manifests/qwen-native-training-r60-only/20260728T092026Z-2906-16060.ok \
GPU_COUNT=2 AUTODL_PRICE_PER_HOUR=<控制台整机价格> \
bash /root/autodl-tmp/search-r1/operator/12_gpu_qwen_native_bc_only.sh
```

入口返回只表示后台 worker 已提交。立刻记录入口打印的 exact attempt 路径并绑定 watchdog，绝不能用 `latest` 猜测：

```bash
attempt=/root/autodl-tmp/search-r1/state/attempts/gpu/<exact-attempt>
bash /root/autodl-tmp/search-r1/operator/04_watch_and_shutdown.sh "$attempt"
tail -f "$attempt/shutdown-watchdog.log"
```

确认训练启动后无需持续占用会话监控。只需把 exact attempt、日志路径和预计时间告诉用户；watchdog 在完整成功或完整失败终态后关机。仍要提醒用户在 AutoDL 控制台确认实例真正停止计费，`shutdown-dispatched` 或 SSH 断开不等于控制平面已停机。

## 7. 时间、预算与磁盘

R60 的 60 步训练耗时 `9:22:07`，所以同配置下单个 20-step 分支粗估约 3 小时，B+C 训练约 6.2 小时。按 5.76 元/小时仅训练约 36 元；再加六个 endpoint、证据校验和启动开销，建议为整个 B/C-only 预留 **50-70 元**。硬超时可以略高，但不得把硬上限描述成预计账单。

每个全参 checkpoint 约 9.56 GB，B+C 至少新增约 19.2 GB。G3 后文档记录约 38 GB 可用，理论上足够，但余量不宽。GPU 启动前必须只读检查 `df -h /root/autodl-tmp`；建议至少保留 25 GB 可用。不要自动删除 smoke、R60、G3、旧 trace 或失败 attempt；不足时把精确目录和大小报告给用户决定。

## 8. 结果判读

至少生成下表，三套 endpoint 分别报告，不只挑最好的一套：

| 指标 | B | C | C-B |
| --- | ---: | ---: | ---: |
| strict EM |  |  |  |
| mean executed searches |  |  |  |
| `E[searches | correct]` |  |  |  |
| 0/1/2/3/4 搜分布 |  |  |  |
| clipped rate |  |  |  |
| non-safe invalid rate |  |  |  |
| answer rate |  |  |  |
| utility `EM - 0.10 * searches/4` |  |  |  |

正向结果的最低解释是：C 的搜索成本下降，同时 strict EM 没有明显下降；最好再有逐题的“B 多搜、C 少搜但同样答对”轨迹。若 C 搜索更少但 EM 明显下降，只能叫 cost-quality trade-off；若两者都没有改善，就归档为探索性负结果。无论结果如何，保留完整轨迹、loss/KL/entropy/grad norm、checkpoint 和 lineage，面试时可重点讲实验设计、失败诊断与工程闭环。

## 9. 新会话建议首条任务

可以把下面内容直接交给下一会话：

> 阅读 `docs/history/qwen35-native-arbc-202607-202608/plans/qwen35_native_bc_posthoc_execution_handoff.md`、`docs/history/qwen35-native-arbc-202607-202608/stages/qwen35_native_r60_g3_evaluation_and_trajectory_analysis.md`、`scripts/autodl/README.md` 和 `scripts/autodl/09_gpu_qwen_native_train.sh`。按最小改动新增 `12_gpu_qwen_native_bc_only.sh`：不重训 R/G3，B20 与 C-gated20 都从 exact R60 digest 独立启动，完成三套 endpoint 配对评测、证据封存和安全自动关机。先实现并跑完本地测试/CPU prepare；没有用户明确告知 GPU 已挂载前，不得启动付费训练。

## 10. 关联资料

- `docs/history/qwen35-native-arbc-202607-202608/stages/qwen35_native_r60_training_and_trajectory_analysis.md`：R60 训练、loss、轨迹和成本。
- `docs/history/qwen35-native-arbc-202607-202608/stages/qwen35_native_r60_g3_evaluation_and_trajectory_analysis.md`：G3 数值、NO-GO 根因和证据路径。
- `docs/history/qwen35-native-arbc-202607-202608/plans/autodl_search_r1_reproduction_plan.md`：总体实验设计与 B/C 奖励定义。
- `scripts/autodl/README.md`：现有三阶段入口、状态、watchdog 和存储合同。
- `scripts/autodl/09_gpu_qwen_native_train.sh`：已经实现但被门禁包围的 B/C 核心代码。
- `scripts/autodl/11_gpu_qwen_native_g3_only.sh`：checkout 外受控 runner、CPU receipt 和固定 R60 校验范例。
