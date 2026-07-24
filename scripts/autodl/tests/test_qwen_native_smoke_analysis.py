#!/usr/bin/env python3
"""Pure-Python tests for the two-step native training smoke decision."""

from __future__ import annotations

from argparse import Namespace
import importlib.util
import json
from pathlib import Path
import statistics
import tempfile
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "qwen_native_smoke_analysis.py"
SPEC = importlib.util.spec_from_file_location("qwen_native_smoke_analysis", MODULE_PATH)
assert SPEC and SPEC.loader
SMOKE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SMOKE)


class QwenNativeSmokeAnalysisTest(unittest.TestCase):

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
        path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )

    def _fixture(self, root: Path) -> Namespace:
        catalog = []
        for sample in range(8):
            catalog.append({
                "sample_id": f"nq:train:{sample}",
                "question": f"Question {sample}?",
                "golden_answers": [f"gold-{sample}"],
            })
        catalog_path = root / "catalog.jsonl"
        self._write_jsonl(catalog_path, catalog)

        traces = []
        for step in (1, 2):
            for sample in range(8):
                rewards = [1.0, 0.0, 0.0, 0.0, 0.0]
                reward_std = statistics.stdev(rewards)
                for slot, reward in enumerate(rewards):
                    sample_id = f"nq:train:{sample}"
                    traces.append({
                        "record_type": "train",
                        "stage": "smoke",
                        "step": step,
                        "sample_id": sample_id,
                        "question": f"Question {sample}?",
                        "gold_answers": [f"gold-{sample}"],
                        "extracted_answer": f"gold-{sample}" if reward else "wrong",
                        "em": int(reward),
                        "reward_em_only": reward,
                        "train_reward": reward,
                        "sequence_advantage": 2.0 if reward else -0.5,
                        "group_uid": f"{step}:{sample_id}",
                        "group_slot": slot,
                        "group_correct_count": 1,
                        "group_reward_std": reward_std,
                    })
        trace_path = root / "train.jsonl"
        self._write_jsonl(trace_path, traces)

        values = {
            "actor/pg_loss": 0.1,
            "actor/kl_loss": 0.01,
            "actor/entropy_loss": 2.0,
            "actor/grad_norm": 1.0,
            "actor/ppo_kl": 0.01,
            **{key: 1.0 for key in SMOKE.NATIVE_EXACT_ONE},
            "native_batch/policy_tokens": 40.0,
            "native_batch/policy_tokens_min_per_trajectory": 1.0,
            "native_batch/policy_coverage": 0.5,
            "native_batch/nonzero_advantage_tokens": 20.0,
            "native_batch/advantage_abs_max": 2.0,
        }
        lines = []
        for step in (1, 2):
            fields = " - ".join(f"{key}:{value:.6f}" for key, value in values.items())
            lines.append(f"step:{step} - {fields}\n")
        log_path = root / "train.log"
        log_path.write_text("".join(lines), encoding="utf-8")

        wandb_dir = root / "wandb" / "offline-run-test"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        (wandb_dir / "run-test.wandb").write_bytes(b"offline-history")
        return Namespace(
            trace=trace_path,
            log=log_path,
            catalog=catalog_path,
            wandb_dir=root / "wandb",
            output=root / "decision.json",
            expected_steps=2,
            batch_size=8,
            group_size=5,
        )

    def test_go_requires_mixed_strict_reward_and_complete_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = SMOKE.analyze(self._fixture(Path(temporary)))
        self.assertEqual(result["decision"], "GO")
        self.assertEqual(result["metrics"]["trajectories"], 80)
        self.assertEqual(result["metrics"]["mixed_groups"], 16)
        self.assertTrue(all(check["passed"] for check in result["checks"].values()))

    def test_one_zero_advantage_step_is_analyzed_across_the_full_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self._fixture(Path(temporary))
            lines = args.log.read_text(encoding="utf-8").splitlines()
            lines[1] = lines[1].replace(
                "native_batch/nonzero_advantage_tokens:20.000000",
                "native_batch/nonzero_advantage_tokens:0.000000",
            ).replace(
                "native_batch/advantage_abs_max:2.000000",
                "native_batch/advantage_abs_max:0.000000",
            )
            args.log.write_text("\n".join(lines) + "\n", encoding="utf-8")
            result = SMOKE.analyze(args)

        self.assertEqual(result["decision"], "GO")
        self.assertTrue(
            result["checks"]["native_batch_nonzero_advantage_tokens"]["passed"]
        )

    def test_no_go_for_nonfinite_or_missing_actor_metric(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self._fixture(Path(temporary))
            text = args.log.read_text(encoding="utf-8")
            args.log.write_text(text.replace("actor/grad_norm:1.000000", "actor/grad_norm:nan", 1), encoding="utf-8")
            result = SMOKE.analyze(args)
            self.assertEqual(result["decision"], "NO-GO")
            self.assertFalse(result["checks"]["finite_actor_grad_norm"]["passed"])

            args = self._fixture(Path(temporary))
            text = args.log.read_text(encoding="utf-8")
            args.log.write_text(text.replace("native_batch/policy_mask_subset:1.000000 - ", ""), encoding="utf-8")
            result = SMOKE.analyze(args)
            self.assertEqual(result["decision"], "NO-GO")

    def test_no_go_without_wandb_history(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self._fixture(Path(temporary))
            (args.wandb_dir / "offline-run-test" / "run-test.wandb").unlink()
            result = SMOKE.analyze(args)
        self.assertEqual(result["decision"], "NO-GO")
        self.assertFalse(result["checks"]["wandb_offline_history"]["passed"])

    def test_rejects_strict_em_or_catalog_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self._fixture(Path(temporary))
            rows = [json.loads(line) for line in args.trace.read_text().splitlines()]
            rows[0]["em"] = 0
            self._write_jsonl(args.trace, rows)
            with self.assertRaisesRegex(ValueError, "strict EM does not replay"):
                SMOKE.analyze(args)

            args = self._fixture(Path(temporary))
            catalog = [json.loads(line) for line in args.catalog.read_text().splitlines()]
            catalog[0]["golden_answers"] = ["forged"]
            self._write_jsonl(args.catalog, catalog)
            with self.assertRaisesRegex(ValueError, "gold answers differ"):
                SMOKE.analyze(args)


if __name__ == "__main__":
    unittest.main()
