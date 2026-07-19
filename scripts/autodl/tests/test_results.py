#!/usr/bin/env python3
"""Pure-stdlib tests for AutoDL metric parsing and checkpoint selection."""

from __future__ import annotations

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
            log = Path(temporary) / "eval.log"
            log.write_text(
                "step:0 - val/em/nq:0.100 - val/search_count/nq:2.000 - "
                "val/no_search_ratio/nq:0.000 - val/utility/nq:0.000\n"
            )
            self.assertEqual(RESULTS.final_metrics(log)["utility"], 0.0)


if __name__ == "__main__":
    unittest.main()
