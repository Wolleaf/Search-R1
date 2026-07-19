import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from verl.utils.random_utils import derive_rank_seed, seed_everything


def test_seed_everything_replays_process_rng_streams():
    seed_everything(42)
    first = (random.random(), np.random.random(), torch.rand(3))

    seed_everything(42)
    second = (random.random(), np.random.random(), torch.rand(3))

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


def test_seed_everything_seeds_python_numpy_torch_and_cuda(monkeypatch):
    calls = []
    monkeypatch.setattr(random, "seed", lambda value: calls.append(("python", value)))
    monkeypatch.setattr(np.random, "seed", lambda value: calls.append(("numpy", value)))
    monkeypatch.setattr(torch, "manual_seed", lambda value: calls.append(("torch", value)))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda value: calls.append(("cuda", value)))

    seed_everything(42)

    assert calls == [("python", 42), ("numpy", 42), ("torch", 42), ("cuda", 42)]


def test_rank_seed_mapping_is_stable_and_distinct():
    assert [derive_rank_seed(42, rank) for rank in range(2)] == [42, 43]
    assert derive_rank_seed(42, 1) == derive_rank_seed(42, 1)


def test_ppo_config_propagates_trainer_seed_to_workers():
    config_path = Path(__file__).parents[1] / "verl" / "trainer" / "config" / "ppo_trainer.yaml"
    config = OmegaConf.load(config_path)

    assert config.trainer.seed == 42
    assert config.actor_rollout_ref.seed == 42
