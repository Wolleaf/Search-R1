import importlib.util
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "export_training_curves.py"
SPEC = importlib.util.spec_from_file_location("export_training_curves",
                                              MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ExportTrainingCurvesTest(unittest.TestCase):

    def test_parses_ansi_training_record_and_derives_utility(self):
        metrics = (
            "\x1b[36m(worker)\x1b[0m step:1 - actor/pg_loss:-0.125 - "
            "actor/kl_loss:0.25 - actor/entropy_loss:1.5 - actor/grad_norm:2.0 - "
            "actor/lr:0.000001 - env/em/mean:0.5 - critic/rewards/mean:0.475 - "
            "env/executed_search_count/mean:1.0 - env/no_search_ratio:0.25 - "
            "env/search_cost/mean:0.025 - response_length/mean:128 - "
            "response_length/clip_ratio:0.0 - timing_s/step:12.5\n")
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "train.log"
            log.write_text(metrics, encoding="utf-8")
            rows = MODULE.parse_training_log(log, "C", "Cost-aware", 0.10, 4)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["step"], 1)
        self.assertEqual(rows[0]["actor_pg_loss"], -0.125)
        self.assertEqual(rows[0]["posthoc_utility"], 0.475)
        self.assertEqual(rows[0]["val_em"], "")

    def test_rejects_duplicate_completed_steps(self):
        record = (
            "step:1 - actor/pg_loss:0 - actor/kl_loss:0 - actor/entropy_loss:1 - "
            "actor/grad_norm:1 - actor/lr:0 - env/em/mean:0 - "
            "critic/rewards/mean:0 - env/executed_search_count/mean:0 - "
            "env/no_search_ratio:1 - env/search_cost/mean:0 - "
            "response_length/mean:1 - response_length/clip_ratio:0 - timing_s/step:1\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "train.log"
            log.write_text(record * 2, encoding="utf-8")
            with self.assertRaisesRegex(ValueError,
                                        "duplicate completed step"):
                MODULE.parse_training_log(log, "R", "Reproduced", 0.10, 4)

    def test_rejects_truncated_stage(self):
        record = (
            "step:1 - actor/pg_loss:0 - actor/kl_loss:0 - actor/entropy_loss:1 - "
            "actor/grad_norm:1 - actor/lr:0 - env/em/mean:0 - "
            "critic/rewards/mean:0 - env/executed_search_count/mean:0 - "
            "env/no_search_ratio:1 - env/search_cost/mean:0 - "
            "response_length/mean:1 - response_length/clip_ratio:0 - timing_s/step:1\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "train.log"
            log.write_text(record, encoding="utf-8")
            with self.assertRaisesRegex(ValueError,
                                        "expected completed steps 1..2"):
                MODULE.parse_training_log(log, "R", "Reproduced", 0.10, 4, 2)


if __name__ == "__main__":
    unittest.main()
