# 多跳搜索机会门禁归档（2026-07-22）

## 结论

本轮结论为 **NO-GO**：不建议直接基于当前 B/control20 checkpoint 开启新的成本感知训练分支。256 道多跳题中，248 道只搜索一次，8 道搜索两次，没有题目搜索三次或四次；61 道答对题全部只搜索一次。观察到的第二次搜索又全部伴随截断和非法动作，更像生成格式恢复，而不是有效的多跳规划。因此，当前策略没有暴露出“保留正确率并减少冗余搜索”的可用空间。

详细解释见 [analysis_zh.md](analysis_zh.md)，机器生成的权威结论见 [results/go_no_go.json](results/go_no_go.json)。

## 实验范围

| 项目 | 固定值 |
| --- | --- |
| 被评 checkpoint | B/control20，Qwen3.5-2B 全参数训练产物 |
| 数据 | HotpotQA dev 128 + 2WikiMultiHopQA dev 128 |
| 检索 | 本地 BM25，top-3，最多 4 次搜索 |
| 推理 | 2 GPU，seed 42，单次采样，最大 response 256 |
| 运行方式 | 仅评测，不训练，不改 checkpoint |
| 代码 | evaluation commit `ac2c1dd7ae8150cc3870a3cc2bc817801853f2dc` |

`resolved-config.yaml` 中的 `cost_lambda=0.1` 只用于记录统一的事后 utility；本次是 `val_only=true`，不会更新参数。被评的 B checkpoint 本身按原奖励（训练时 `lambda=0`）得到。

## 核心结果

| 数据 | 答对 | EM | 1 次搜索 | 2 次搜索 | 3/4 次搜索 |
| --- | ---: | ---: | ---: | ---: | ---: |
| HotpotQA | 28/128 | 21.88% | 124 | 4 | 0 |
| 2WikiMultiHopQA | 33/128 | 25.78% | 124 | 4 | 0 |
| 合计 | 61/256 | 23.83% | 248 | 8 | 0 |

30/256（11.72%）轨迹被截断，恰好也是出现非法动作的 30 题。8 条两次搜索轨迹全部答错且被截断；其中 7 条原样重复第一次 query，7 条没有提取出最终答案。所有 226 条未截断轨迹都只搜索一次。

## 文件说明

- `results/per_question.jsonl`：256 题的答案、完整思考/动作轨迹、结构化 turns、检索文档和诊断字段，是逐题分析的首选文件。
- `results/{correct,wrong,two_plus_search,redundant_search_candidates}.csv`：按结果切分的人工检查视图；CSV 中保留完整轨迹。
- `results/summary.json`、`results/strata.csv`：总体、数据集、题型和 supporting-title 数量分层统计。
- `results/go_no_go.json`：预注册的 8 项门禁条件及唯一科学决策。
- `eval/traces/eval_predictions.jsonl`：评测器直接写出的 256 条原始 trace；manifest 记录行数、字节数和 SHA-256。
- `eval/resolved-config.yaml`、`eval/run.env`、`results/lineage.tsv`：运行配置、时间/预算及 checkpoint 血缘。
- `phase/`：outer phase 终态、证据 digest 和自动关机分派记录；日志文件按仓库 `.gitignore` 仅保留在本地副本中。
- `archive.sha256`：本次 Git 归档内全部文件（不含自身）的本地校验清单。

## 来源与校验

远端结果、评测和 outer attempt 分别为：

```text
/root/autodl-tmp/search-r1/runs/search-opportunity-gate/attempts/20260722T031309Z-1678-8425
/root/autodl-tmp/search-r1/runs/eval/search_opportunity/attempts/20260722T031434Z-1714-18467
/root/autodl-tmp/search-r1/state/attempts/gpu/20260722T031309Z-1678-8425
```

outer phase 从 `2026-07-22T03:13:09Z` 到 `03:48:04Z`，成功退出（exit code 0）；按 5.76 元/小时估算约 3.35 元。原始远端完整证据契约是 `results/evidence.sha256`，其 digest 为 `621c0eb715a010462dfe764d4441658cb9f3ce5b596119fcb67397d64884c64b`。该契约还引用未纳入 Git 的源数据、Parquet、模型比较清单和日志；它不等同于本地 `archive.sha256`。

在 Linux 或 Git Bash 中，从本目录运行：

```bash
sha256sum -c archive.sha256
```
