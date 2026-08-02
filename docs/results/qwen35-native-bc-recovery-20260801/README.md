# Qwen3.5 Native B/C Recovery Evidence Archive

本目录是 2026-08-01 B20/C20 recovery 与配对评测的本地紧凑证据包。完整中文分析见：

- [Qwen3.5 Native R60 后续 B20/C20 训练、恢复与完整结果分析](../../qwen35_native_bc_recovery_complete_analysis_report.md)

## 1. 证据来源

- recovery outer attempt：`/root/autodl-tmp/search-r1/state/attempts/gpu/20260801T041947Z-7015-12411`
- final result root：`/root/autodl-tmp/search-r1/runs/qwen-native-training/attempts/20260801T041947Z-7015-12411`
- B20 run：`/root/autodl-tmp/search-r1/runs/control/attempts/20260731T061302Z-1592-7937`
- C20 run：`/root/autodl-tmp/search-r1/runs/cost_aware_gated/attempts/20260801T042516Z-7061-17197`
- result contract：`qwen-native-training-bc-recovery-v1`

所有文件均以只读方式从无卡实例下载。本目录没有模型 checkpoint；checkpoint 仍保留在远端持久盘。

## 2. 目录说明

### `raw-result/`

远端 final result root 的完整文件内容：

- `contract.env`：恢复实验合同；
- `run-index.tsv`：B/C 与六个评测 run 的索引；
- `lineage.tsv`：checkpoint、checkout、data、config、trace、contract digest 血缘；
- `branch-checkpoints.env`：B/C 最终 checkpoint 与 digest；
- `source-b.env`：复用 B 的来源和验证身份；
- `evidence.sha256`：完整远端 evidence manifest；
- `wandb-receipts/`：B20 复用前的 W&B 收据；
- `paired-val/`、`paired-nq_test/`、`paired-multihop/`：配对 summary、逐题结果、正确/错误题和搜索转移。

`raw-result/` 共 25 个文件、56,640,731 bytes（约 54.02 MiB）。

### `raw-outer/`

外层 phase 的终态与关机证据，包括：

- `terminal`、`exit-code`、`.failed`；
- `phase.log`；
- `result-root`、`result-contract`、`evidence-digest`、`evidence-marker`；
- `shutdown-safe`、`shutdown-requested`、`shutdown-dispatched`；
- freeze、PID、时间戳和 runner 索引。

外层 `failed/203` 是成功封存后交给关机 watchdog 的受控状态。科学执行状态应看 `run-index.tsv` 中八个内层 run，它们均为 `success/0`。

### B20/C20 关键训练文件

根目录另行保留两条分支的：

- `*-train.log`；
- `*-run.env`；
- `*-resolved-config.yaml`；
- `*-native-training-contract.json`；
- `*-lineage.tsv`；
- `*-trace-manifest.json`。

这些文件足以审计每步指标、奖励定义、训练配置、lineage 和 trace 完整性；800×2 条 raw training trace 与两个约 9G checkpoint 没有重复下载到 Git 工作区。

## 3. 完整性校验

本地复算结果：

```text
evidence.sha256 SHA-256
8ec72644d618c194a248e82965fcf4b7473e69acad0a5c5cd519f9df2c03827c

raw-result payloads matched: 24/24
selected B/C training evidence matched: 12/12
```

该 manifest digest 与 outer `evidence-digest`、outer `evidence-marker` 和单独下载的 `evidence-marker.ok` 一致。

2026-08-01T11:29:10Z 的远端独立全量复验还确认：

```text
R60 checkpoint tree: OK
B20 checkpoint tree: OK
C20 checkpoint tree: OK
evidence.sha256: 166/166 OK, rc=0
```

所以当前持久盘上的三个关键 checkpoint 与完整封存证据都通过了字节级复验。

封存覆盖有一个已知轻微缺口：新 C20 与六个新 eval 的 7 组 `terminal/exit-code` 没有列入 166 项 manifest。它们现场均为 `success/0`，且相应 `run.env`、config、日志、trace/manifest 和 W&B receipt 已封存，因此不影响结果可用性；但这 14 个小文件本身没有被 manifest 防篡改保护。为保留原始 seal 身份，本目录不事后修改该 manifest。

六个 eval 的 `run.env` 还将 `eval_data_file/hash` 记录为 `unavailable`。sealed resolved config 已绑定三个确切 parquet，parquet 自身在 manifest 中通过哈希，B/C trace 的样本身份、顺序和 checkpoint digest 也已逐条交叉验证；因此这是元数据字段缺口，不是数据 lineage 缺失。

若再次手工验证，注意 `evidence.sha256` 使用远端 final result root 的相对路径；本地未下载的超大 checkpoint/raw-trace 项应在远端按 manifest 校验，不能因为本地紧凑包未包含它们而判定丢失。

## 4. 快速结果入口

- [val summary](raw-result/paired-val/summary.md)
- [NQ-test summary](raw-result/paired-nq_test/summary.md)
- [multihop summary](raw-result/paired-multihop/summary.md)
- [recovery contract](raw-result/contract.env)
- [lineage](raw-result/lineage.tsv)
- [run index](raw-result/run-index.tsv)
- [outer phase log](raw-outer/phase.log)

## 5. 保留策略

这个目录是复盘与结果复算所需的小型证据包，应长期保留。远端的 exact R60、B20、C20、G3 checkpoint 和对应 raw trace 仍有科学用途；在建立引用关系与备份清单之前，不应仅因为磁盘占用而删除。
