# AutoDL 三阶段操作说明

本目录是 Search-R1 小规模复现的唯一云端入口。固定环境为 AutoDL 的 **PyTorch 2.8.0 / Python 3.12 / Ubuntu 22.04 / CUDA 12.8** 镜像；持久目录固定为 `/root/autodl-tmp/search-r1`。脚本不会扫描机器规格、自动换配置、自动重试或关机。

## 阶段一：固定 Git 版本

先通过 SFTP 将整个 `scripts/autodl/` 上传到 `/root/autodl-tmp/autodl-bootstrap/`，然后在无 GPU 实例执行：

```bash
REPO_URL=https://github.com/<owner>/Search-R1.git \
COMMIT_SHA=<已推送的40位commit> \
bash /root/autodl-tmp/autodl-bootstrap/01_git.sh
```

仓库会以 detached HEAD 克隆到 `/root/autodl-tmp/search-r1/checkout`。私有仓库使用临时 Git credential helper，不要把 token 放进 URL。成功标准是最新 `git` attempt 中 `exit-code` 为 `0` 且存在 `.success`。

## 阶段二：CPU 联网准备

继续使用无 GPU 实例：

```bash
bash /root/autodl-tmp/search-r1/checkout/scripts/autodl/02_cpu_prepare.sh
```

此阶段通过 Conda 的 `llmdevelop` 环境解析镜像 Python，并在持久盘创建带 `--system-site-packages` 的 train venv，以复用镜像自带 PyTorch 2.8.0+cu128，不修改 `llmdevelop`，也不重新安装 PyTorch。随后创建隔离的 retriever venv，安装依赖和 Java，下载固定 revision 的 Qwen3.5-2B 与 BM25 索引，生成 NQ 512/64/128 数据，并执行 tokenizer、数据、BM25 和成本奖励测试。最后生成自校验的 `manifests/cpu_handoff.json`；它是 GPU 阶段唯一认可的交接。完成后在 AutoDL 控制台确认 CPU 实例已停止，并确认 GPU 实例挂载的是同一个数据盘。

## 阶段三：GPU 离线训练

挂载 GPU 后只执行：

```bash
GPU_COUNT=1 AUTODL_PRICE_PER_HOUR=<实例每小时价格> \
bash /root/autodl-tmp/search-r1/checkout/scripts/autodl/03_gpu_run.sh
```

`GPU_COUNT` 必须明确为 `1` 或 `2`，`AUTODL_PRICE_PER_HOUR` 必须填写控制台显示的整个实例时价。脚本先离线校验 CPU handoff，再启动 CPU BM25 服务，依次完成 1-step smoke、从相同原始权重独立训练 baseline 和 cost-aware（各 60 steps）、按 val-64 选择 checkpoint，并在 test-128 上评测。训练固定 GRPO group size 8，每步 4 个 prompt、32 条轨迹；生成和训练 micro-batch 保持为 1，以控制单卡显存。最终表为公平对比，统一用 `lambda=0.10` 计算两个 checkpoint 的 test utility；EM 和搜索次数仍分别报告。结果写到 `runs/comparison/results.md`。

单次阶段三按 smoke 30 元、baseline 100 元、cost-aware 100 元、两次评测各 15 元设置硬超时，合计最多 260 元；其余 40 元留给 CPU、存储和阶段间开销。group size 8 的训练量接近原配置两倍，100 元是预算硬上限，不是跑满 60 steps 的保证。达到单项上限时 `timeout` 返回非零状态，已有 checkpoint 保留，后续实验不会继续启动。

单卡 smoke 若 OOM，只允许按顺序重跑整个 GPU 阶段：先设置 `TRAIN_BATCH_SIZE=2`，仍失败再加 `MAX_RESPONSE_LENGTH=192`；再失败才人工换两张卡。两个正式实验始终共享同一组设置。失败不会自动重试，也不会覆盖旧 attempt；本最小版本不保存优化器状态，因此中断后从原始模型重跑该实验，不把权重 checkpoint 描述为精确续训。

## 状态、日志与费用

三个入口均通过 `nohup + setsid` 后台运行，并由持久盘上的非阻塞 `flock` 串行化。入口返回仅表示已提交，不代表成功。查看最新阶段：

```bash
attempt=$(cat /root/autodl-tmp/search-r1/state/latest/cpu)
tail -f "$attempt/phase.log"
cat "$attempt/exit-code"
```

每次阶段和训练均保留独立目录、日志、原始 exit code、开始/结束时间及成功或失败标记。GPU 阶段设置 Hugging Face、Datasets、Transformers、pip 和 WandB 离线模式；缺失资产会直接失败，不会临时下载。

脚本**不会执行 guest shutdown**。无论成功或失败，都应先通过 SSH 检查终态，再到 AutoDL 控制台关机并确认停止计费；guest 内关机本身也不能证明云端已停止计费。
