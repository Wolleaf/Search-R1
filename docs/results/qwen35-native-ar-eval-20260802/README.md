# Qwen3.5 Native A/R Final Evaluation Evidence

本目录保存 2026-08-01 至 2026-08-02 完成的 A/R 三端点最终评测紧凑证据。完整结果分析见：

- [Qwen3.5 Native A/R/B/C 最终结果分析](../../qwen35_native_arbc_final_results_analysis.md)

## 结果身份

- A：sealed post-trained Qwen3.5-2B parent baseline（不是官方 Base 变体），checkpoint digest `bc67be20efb353ba14d9c1b291a64410afec94f2310e94b59b6b76047b164e78`
- R：direct-RL R60，checkpoint digest `583771b131b6e2aa663ee2ef13ea6421524cdd7fed839f35fd6f9eb246a2c231`
- result contract：`qwen-native-training-ar-eval-only-v1`
- remote result attempt：`20260801T123025Z-1367-15987`
- evidence digest：`0058974e9cdc0eae9d0313a52ebebd60051c5b483beceab8f83bdcaf7b74a63e`

评测只执行 A/R 的 val、NQ-test、multihop 六个 endpoint run，没有执行训练。合同为 greedy、group size 1、每题单 rollout、seed 42，并按 exact sample ID 配对。

## 目录说明

- `raw-result/contract.env`：结果合同与模型身份；
- `raw-result/run-index.tsv`：六个 endpoint run 索引；
- `raw-result/lineage.tsv`：数据、checkpoint、trace、配置和结果血缘；
- `raw-result/paired-ar-*/summary.json` 与 `summary.md`：封存聚合结果和 paired bootstrap；
- `raw-result/paired-ar-*/paired_results.csv`：逐题 A/R 对齐结果，包含答案、轨迹和行为指标；
- `raw-result/paired-ar-*/correct_questions.csv`、`wrong_questions.csv`、`search_transition.csv`：错误、正确与搜索转移切片；
- `raw-result/evidence.sha256`：远端 143 项完整 evidence manifest；
- `raw-outer/`：outer `success/0`、phase log、result marker 与安全关机证据；
- `evidence-marker.ok`：正式 manifest digest marker。

## 完整性复核

2026-08-02 在无卡实例上从持久盘执行只读复核：

```text
outer terminal: success
outer exit-code: 0
remote evidence manifest: 143/143 OK
manifest SHA-256: 0058974e9cdc0eae9d0313a52ebebd60051c5b483beceab8f83bdcaf7b74a63e
local raw-result payloads: 21/21 matched, 0 mismatch
local evidence marker: matched
```

本目录没有 checkpoint，也没有重复下载 manifest 中的全部原始 eval trace、W&B 文件、数据文件和配置文件。当前紧凑包足够逐题复算最终结果，但不是完整的离线生成重演包；完整远端 evidence set 由 sealed manifest 索引并校验，本地只保存其中的紧凑子集。

## 快速入口

- [val summary](raw-result/paired-ar-val/summary.md)
- [NQ-test summary](raw-result/paired-ar-nq_test/summary.md)
- [multihop summary](raw-result/paired-ar-multihop/summary.md)
- [contract](raw-result/contract.env)
- [run index](raw-result/run-index.tsv)
- [lineage](raw-result/lineage.tsv)
