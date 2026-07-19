# Repository Guidelines

## Project Structure & Module Organization

Core Search-R1 code lives in `search_r1/`: `llm_agent/` implements generation helpers, while `search/` contains corpus indexing and local or online retrieval services. `verl/` is the bundled RL engine; trainer entry points are under `verl/trainer/`, with Hydra configuration in `verl/trainer/config/`. Data preparation and experiment recipes live in `scripts/`; runnable retriever and multinode variants are in `example/`. Use `docs/` for technical notes and `public/` for README images. Root scripts (`infer.py`, `retrieval_launch.sh`, and `train_*.sh`) are primary entry points.

## Build, Test, and Development Commands

The main workflows assume Linux/Bash, NVIDIA GPUs, and CUDA-compatible dependencies.

- `conda create -n searchr1 python=3.9 && conda activate searchr1` creates the documented environment.
- `pip install -e ".[test]"` installs the package editably with pytest and YAPF; follow `README.md` for pinned Torch, vLLM, and FlashAttention versions.
- `python scripts/data_process/nq_search.py` prepares NQ Parquet files under `data/nq_search/`.
- Configure corpus/index placeholders, then run `bash retrieval_launch.sh`; use `python infer.py` against the running service.
- `bash train_ppo.sh` launches the default eight-GPU PPO recipe. `train_grpo.sh` also requires `TRAIN_DATA_DIR` and `TEST_DATA_DIR` to be set.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` for modules/functions/variables, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants and environment variables. Add type hints to new public interfaces. YAPF is the declared formatter, but no repository-specific style file exists; check touched files with `yapf --diff path/to/file.py` and avoid unrelated reformatting.

## Testing Guidelines

Focused pytest tests live under `tests/`; no coverage threshold is enforced. Name files `test_<module>.py` and functions `test_<behavior>`, then run `python -m pytest -q`. For GPU, distributed, or retrieval changes, document the CUDA/model configuration and the smoke-test command used.

## Commit & Pull Request Guidelines

History favors short, lowercase, verb-first subjects such as `add reranker`, `fix proto bug`, and `update train script`; Conventional Commit prefixes are not used. Keep commits focused. PRs should explain the change and motivation, link related issues, and list validation commands/results. Include relevant logs or screenshots for training behavior, retrieval responses, or documentation visuals.

## Configuration & Security

Review placeholder paths, model IDs, and `CUDA_VISIBLE_DEVICES` before running scripts. Never commit API keys in `example/retriever/`. Keep corpora, checkpoints, logs, and WandB outputs out of Git as specified by `.gitignore`.
