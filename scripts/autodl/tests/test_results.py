#!/usr/bin/env python3
"""Pure-stdlib tests for AutoDL result selection and summarization."""

from __future__ import annotations

import csv
import importlib.util
import json
from argparse import Namespace
from pathlib import Path
import tempfile
import unittest

MODULE_PATH = Path(__file__).resolve().parents[1] / "results.py"
SPEC = importlib.util.spec_from_file_location("autodl_results", MODULE_PATH)
assert SPEC and SPEC.loader
RESULTS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RESULTS)


class ResultsTest(unittest.TestCase):

    @staticmethod
    def write_eval(path: Path, em: float, searches: float, no_search: float, utility: float) -> None:
        path.write_text(
            f"step:0 - val/em/nq:{em} - val/search_count/nq:{searches} - "
            f"val/no_search_ratio/nq:{no_search} - val/utility/nq:{utility}\n"
        )

    @staticmethod
    def write_metadata(
        path: Path,
        role: str,
        checkpoint: str,
        parent_checkpoint: str,
        elapsed: int,
        checkpoint_digest: str,
        parent_checkpoint_digest: str,
        checkout_commit: str = "commit-a",
        cpu_handoff_digest: str = "handoff-a",
    ) -> None:
        cost_lambda = "0.10" if role == "cost_aware" else "0"
        path.write_text(
            f"role={role}\n"
            f"checkpoint={checkpoint}\n"
            f"checkpoint_digest={checkpoint_digest}\n"
            f"parent_checkpoint={parent_checkpoint}\n"
            f"parent_checkpoint_digest={parent_checkpoint_digest}\n"
            f"checkout_commit={checkout_commit}\n"
            f"cpu_handoff_digest={cpu_handoff_digest}\n"
            f"resolved_config_sha256=config-{role}\n"
            "seed=42\n"
            f"cost_lambda={cost_lambda}\n"
            f"elapsed_seconds={elapsed}\n"
            "gpu_count=2\n"
            "price_per_hour=5.76\n"
        )

    def test_selects_best_saved_checkpoint_and_cost_tie_break(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint_root = root / "checkpoints"
            for step in (20, 40, 60):
                (checkpoint_root / "actor" / f"global_step_{step}").mkdir(parents=True)
            log = root / "train.log"
            log.write_text(
                "step:20 - val/em/nq:0.500 - val/search_count/nq:1.500 - val/utility/nq:0.425\n"
                "step:40 - val/em/nq:0.600 - val/search_count/nq:1.500 - val/utility/nq:0.525\n"
                "step:60 - val/em/nq:0.600 - val/search_count/nq:0.500 - val/utility/nq:0.525\n"
            )
            output = root / "selected.json"
            RESULTS.select(Namespace(
                variant="cost_aware", log=log, checkpoint_root=checkpoint_root, output=output,
            ))
            self.assertEqual(json.loads(output.read_text())["step"], 60)

            RESULTS.select(Namespace(
                variant="baseline", log=log, checkpoint_root=checkpoint_root, output=output,
            ))
            self.assertEqual(json.loads(output.read_text())["step"], 40)

    def test_zero_utility_is_not_replaced_by_em(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log = root / "eval.log"
            log.write_text(
                "step:0 - val/em/nq:0.100 - val/search_count/nq:2.000 - "
                "val/no_search_ratio/nq:0.000 - val/utility/nq:0.000\n"
            )
            self.assertEqual(RESULTS.final_metrics(log)["utility"], 0.0)

            rounded = root / "rounded.log"
            self.write_eval(rounded, 0.333333, 0.666667, 0, 0.300000)
            self.assertEqual(RESULTS.final_metrics(rounded)["utility"], 0.3)

    def test_summarizes_four_fixed_endpoints_and_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logs = {}
            for index, model in enumerate(("base", "reproduced", "control", "cost_aware")):
                logs[model] = root / f"{model}.log"
                em = 0.1 + index / 10
                searches = float(index)
                self.write_eval(
                    logs[model], em, searches, index / 10, em - 0.10 * searches / 2,
                )

            metadata = {}
            specs = (
                ("reproduced", "/checkpoints/r", "/models/base", 3600),
                ("control", "/checkpoints/b", "/checkpoints/r", 1800),
                ("cost_aware", "/checkpoints/c", "/checkpoints/r", 900),
            )
            for role, checkpoint, parent, elapsed in specs:
                metadata[role] = root / f"{role}.env"
                digest = f"digest-{role}"
                parent_digest = "digest-base" if role == "reproduced" else "digest-reproduced"
                self.write_metadata(
                    metadata[role], role, checkpoint, parent, elapsed, digest, parent_digest,
                )

            output_dir = root / "results"
            RESULTS.summarize(Namespace(
                base_checkpoint="/models/base",
                base_checkpoint_digest="digest-base",
                base_log=logs["base"],
                reproduced_log=logs["reproduced"],
                control_log=logs["control"],
                cost_aware_log=logs["cost_aware"],
                reproduced_run_metadata=metadata["reproduced"],
                control_run_metadata=metadata["control"],
                cost_aware_run_metadata=metadata["cost_aware"],
                output_dir=output_dir,
            ))

            with (output_dir / "results.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["model"] for row in rows], [
                "base", "reproduced", "control", "cost_aware",
            ])
            self.assertEqual(rows[0]["checkpoint"], "/models/base")
            self.assertEqual(rows[0]["checkpoint_digest"], "digest-base")
            self.assertEqual(rows[0]["elapsed_seconds"], "0")
            self.assertEqual(rows[0]["gpu_hours"], "0.0")
            self.assertEqual(rows[0]["actual_rmb"], "0.0")
            self.assertEqual(rows[1]["gpu_hours"], "2.0")
            self.assertEqual(rows[1]["actual_rmb"], "5.76")
            self.assertEqual(rows[2]["parent_checkpoint"], "/checkpoints/r")
            self.assertEqual(rows[3]["parent_checkpoint"], "/checkpoints/r")
            self.assertEqual(rows[2]["parent_checkpoint_digest"], "digest-reproduced")
            self.assertEqual(rows[3]["parent_checkpoint_digest"], "digest-reproduced")
            self.assertEqual(rows[1]["checkout_commit"], "commit-a")
            self.assertEqual(rows[3]["cost_lambda"], "0.1")
            markdown = (output_dir / "results.md").read_text()
            for label in ("A / Base", "R / Reproduced", "B / Control", "C / Cost-aware"):
                self.assertIn(f"| {label} |", markdown)

    def test_rejects_missing_duplicate_and_multiple_eval_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = root / "missing.log"
            missing.write_text(
                "step:0 - val/em/nq:0.1 - val/search_count/nq:1 - val/utility/nq:0.0\n"
            )
            with self.assertRaisesRegex(ValueError, "missing: val/no_search_ratio"):
                RESULTS.final_metrics(missing)

            duplicate = root / "duplicate.log"
            duplicate.write_text(
                "step:0 - val/em/nq:0.1 - val/em/nq:0.2 - val/search_count/nq:1 - "
                "val/no_search_ratio/nq:0 - val/utility/nq:0\n"
            )
            with self.assertRaisesRegex(ValueError, "duplicate metric"):
                RESULTS.final_metrics(duplicate)

            multiple = root / "multiple.log"
            self.write_eval(multiple, 0.1, 1, 0, 0)
            with multiple.open("a") as handle:
                handle.write(
                    "step:1 - val/em/nq:0.2 - val/search_count/nq:1 - "
                    "val/no_search_ratio/nq:0 - val/utility/nq:0.1\n"
                )
            with self.assertRaisesRegex(ValueError, "found 2"):
                RESULTS.final_metrics(multiple)

            inconsistent = root / "inconsistent.log"
            self.write_eval(inconsistent, 0.5, 2, 0, 0.5)
            with self.assertRaisesRegex(ValueError, "utility.*inconsistent"):
                RESULTS.final_metrics(inconsistent)

    def test_rejects_branch_with_wrong_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logs = {}
            for model in ("base", "reproduced", "control", "cost_aware"):
                logs[model] = root / f"{model}.log"
                self.write_eval(logs[model], 0.1, 1, 0, 0.05)
            metadata = {}
            for role, checkpoint, parent in (
                ("reproduced", "/checkpoints/r", "/models/base"),
                ("control", "/checkpoints/b", "/checkpoints/not-r"),
                ("cost_aware", "/checkpoints/c", "/checkpoints/r"),
            ):
                metadata[role] = root / f"{role}.env"
                digest = f"digest-{role}"
                parent_digest = "digest-base" if role == "reproduced" else "digest-reproduced"
                self.write_metadata(
                    metadata[role], role, checkpoint, parent, 1, digest, parent_digest,
                )
            args = Namespace(
                base_checkpoint="/models/base",
                base_checkpoint_digest="digest-base",
                base_log=logs["base"],
                reproduced_log=logs["reproduced"],
                control_log=logs["control"],
                cost_aware_log=logs["cost_aware"],
                reproduced_run_metadata=metadata["reproduced"],
                control_run_metadata=metadata["control"],
                cost_aware_run_metadata=metadata["cost_aware"],
                output_dir=root / "results",
            )
            with self.assertRaisesRegex(ValueError, "control parent checkpoint"):
                RESULTS.summarize(args)

            self.write_metadata(
                metadata["control"], "control", "/checkpoints/b", "/checkpoints/r", 1,
                "digest-control", "digest-reproduced",
            )
            self.write_metadata(
                metadata["reproduced"], "reproduced", "/checkpoints/r", "/models/not-base", 1,
                "digest-reproduced", "digest-base",
            )
            with self.assertRaisesRegex(ValueError, "reproduced parent checkpoint"):
                RESULTS.summarize(args)

            self.write_metadata(
                metadata["reproduced"], "reproduced", "/checkpoints/r", "/models/base", 1,
                "digest-reproduced", "digest-not-base",
            )
            with self.assertRaisesRegex(ValueError, "reproduced parent digest"):
                RESULTS.summarize(args)

            self.write_metadata(
                metadata["reproduced"], "reproduced", "/checkpoints/r", "/models/base", 1,
                "digest-reproduced", "digest-base",
            )
            self.write_metadata(
                metadata["control"], "control", "/checkpoints/b", "/checkpoints/r", 1,
                "digest-control", "digest-not-reproduced",
            )
            with self.assertRaisesRegex(ValueError, "control parent digest"):
                RESULTS.summarize(args)

            self.write_metadata(
                metadata["control"], "control", "/checkpoints/b", "/checkpoints/r", 1,
                "digest-control", "digest-reproduced", checkout_commit="commit-b",
            )
            with self.assertRaisesRegex(ValueError, "checkout_commit"):
                RESULTS.summarize(args)

            self.write_metadata(
                metadata["control"], "control", "/checkpoints/b", "/checkpoints/r", 1,
                "digest-control", "digest-reproduced",
            )
            self.write_metadata(
                metadata["cost_aware"], "cost_aware", "/checkpoints/c", "/checkpoints/r", 1,
                "digest-cost_aware", "digest-reproduced", cpu_handoff_digest="handoff-b",
            )
            with self.assertRaisesRegex(ValueError, "cpu_handoff_digest"):
                RESULTS.summarize(args)


if __name__ == "__main__":
    unittest.main()
