#!/usr/bin/env python3
"""Pure-Python tests for the two-step native training smoke decision."""

from __future__ import annotations

from argparse import Namespace
import importlib.util
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "qwen_native_smoke_analysis.py"
SPEC = importlib.util.spec_from_file_location("qwen_native_smoke_analysis", MODULE_PATH)
assert SPEC and SPEC.loader
SMOKE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SMOKE)
import wandb_history as WANDB_HISTORY  # noqa: E402


class QwenNativeSmokeAnalysisTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        import wandb

        cls._template_root = tempfile.TemporaryDirectory()
        root = Path(cls._template_root.name)
        variants = {
            "valid": ((1, 2), 0, None),
            "stats_only": ((), 0, None),
            "missing_step": ((1,), 0, None),
            "nan": ((1, 2), 0, (2, "actor/grad_norm")),
            "exit_one": ((1, 2), 1, None),
        }
        with mock.patch.dict(
            os.environ,
            {
                "WANDB_DISABLE_CODE": "true",
                "WANDB_SILENT": "true",
            },
            clear=False,
        ):
            for name, (steps, exit_code, nonfinite) in variants.items():
                directory = root / name
                directory.mkdir()
                run = wandb.init(
                    project="search-r1-smoke-analysis-test",
                    dir=str(directory),
                    id=f"smoke-{name}",
                    mode="offline",
                    settings=wandb.Settings(console="off", disable_git=True),
                )
                for step in steps:
                    values = cls._actor_values()
                    if nonfinite and nonfinite[0] == step:
                        values[nonfinite[1]] = float("nan")
                    run.log(values, step=step, commit=True)
                run.summary["fixture_complete"] = True
                run.finish(exit_code=exit_code)
                wandb.teardown()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._template_root.cleanup()

    @staticmethod
    def _actor_values() -> dict[str, float]:
        return {
            "actor/pg_loss": 0.1,
            "actor/kl_loss": 0.01,
            "actor/entropy_loss": 2.0,
            "actor/grad_norm": 1.0,
            "actor/ppo_kl": 0.01,
        }

    @staticmethod
    def _terminal_event() -> dict[str, object]:
        return {
            "terminal_generation": True,
            "terminal_instruction_applied": True,
            "terminal_prompt_version": SMOKE.QWEN35_TERMINAL_PROMPT_VERSION,
            "terminal_prompt_sha256": SMOKE.QWEN35_TERMINAL_PROMPT_SHA256,
            "terminal_prompt_text": SMOKE.QWEN35_TERMINAL_PROMPT,
            "terminal_prompt_policy_token_count": 0,
            "terminal_followup_token_count": 8,
            "generation_context": "terminal_answer",
            "requested_action": "answer",
            "action": "answer",
            "parse_error": None,
            "terminal_rejection_reason": None,
            "valid_action": True,
            "done": True,
            "executed_search": False,
        }

    def _write_synthetic_wandb(
        self,
        args: Namespace,
        *,
        history_rows: tuple[tuple[int, dict[str, float]], ...] | None = None,
        include_summary: bool = True,
        exit_codes: tuple[int, ...] = (0,),
        summary_after_exit: bool = False,
    ) -> Path:
        from wandb.proto import wandb_internal_pb2
        from wandb.sdk.internal.datastore import DataStore

        shutil.rmtree(args.wandb_dir)
        run_dir = args.wandb_dir / "offline-run-synthetic"
        run_dir.mkdir(parents=True)
        history = run_dir / "run-synthetic.wandb"
        store = DataStore()
        store.open_for_write(str(history))
        rows = history_rows or (
            (1, self._actor_values()),
            (2, self._actor_values()),
        )
        for step, metrics in rows:
            record = wandb_internal_pb2.Record()
            record.history.step.num = step
            for key, value in (("_step", step), *metrics.items()):
                item = record.history.item.add()
                item.nested_key.append(key)
                item.value_json = json.dumps(value)
            store.write(record)
        if include_summary:
            summary = wandb_internal_pb2.Record()
            item = summary.summary.update.add()
            item.nested_key.append("fixture_complete")
            item.value_json = "true"
            store.write(summary)
        for exit_code in exit_codes:
            exit_record = wandb_internal_pb2.Record()
            exit_record.exit.exit_code = exit_code
            store.write(exit_record)
        if summary_after_exit:
            summary = wandb_internal_pb2.Record()
            item = summary.summary.update.add()
            item.nested_key.append("after_exit")
            item.value_json = "true"
            store.write(summary)
        store.close()
        return history

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
        path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )

    def _fixture(self, root: Path, wandb_template: str = "valid") -> Namespace:
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
                        "max_action_budget": 4,
                        "action_count": 5,
                        "generation_events": [
                            {"terminal_generation": False} for _ in range(4)
                        ] + [self._terminal_event()],
                    })
        trace_path = root / "train.jsonl"
        self._write_jsonl(trace_path, traces)

        values = {
            **self._actor_values(),
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

        wandb_root = root / "wandb"
        if wandb_root.exists():
            shutil.rmtree(wandb_root)
        shutil.copytree(
            Path(self._template_root.name) / wandb_template / "wandb",
            wandb_root,
            symlinks=True,
        )
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
        self.assertEqual(result["schema_version"], 2)
        self.assertEqual(result["metrics"]["trajectories"], 80)
        self.assertEqual(result["metrics"]["mixed_groups"], 16)
        self.assertEqual(result["metrics"]["terminal_generation_count"], 80)
        self.assertEqual(result["metrics"]["terminal_answer_rate"], 1.0)
        self.assertEqual(result["metrics"]["terminal_requested_search_rate"], 0.0)
        self.assertEqual(result["metrics"]["wandb"]["history_records"], 2)
        self.assertEqual(result["metrics"]["wandb"]["history_steps"], [1, 2])
        self.assertEqual(
            result["checks"]["wandb_offline_history"]["observed"], 2
        )
        self.assertTrue(all(check["passed"] for check in result["checks"].values()))

    def test_no_go_when_terminal_search_request_rate_exceeds_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self._fixture(Path(temporary))
            rows = [json.loads(line) for line in args.trace.read_text().splitlines()]
            for row in rows[:5]:
                terminal = row["generation_events"][-1]
                terminal.update({
                    "requested_action": "search",
                    "action": None,
                    "parse_error": "search_disallowed_after_budget",
                    "terminal_rejection_reason": "search_disallowed_after_budget",
                    "valid_action": False,
                })
            self._write_jsonl(args.trace, rows)

            result = SMOKE.analyze(args)

        self.assertEqual(result["decision"], "NO-GO")
        self.assertEqual(result["metrics"]["terminal_requested_search_count"], 5)
        self.assertFalse(result["checks"]["terminal_requested_search_rate"]["passed"])

    def test_no_go_when_terminal_instruction_or_policy_mask_is_wrong(self) -> None:
        cases = (
            ("terminal_instruction_applied", False, "terminal_instruction_applied_count"),
            ("terminal_prompt_policy_token_count", 1, "terminal_prompt_policy_token_count"),
        )
        for field, value, check_name in cases:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                args = self._fixture(Path(temporary))
                rows = [
                    json.loads(line) for line in args.trace.read_text().splitlines()
                ]
                rows[0]["generation_events"][-1][field] = value
                self._write_jsonl(args.trace, rows)

                result = SMOKE.analyze(args)

                self.assertEqual(result["decision"], "NO-GO")
                self.assertFalse(result["checks"][check_name]["passed"])

    def test_no_go_when_terminal_search_is_accepted_or_executed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self._fixture(Path(temporary))
            rows = [json.loads(line) for line in args.trace.read_text().splitlines()]
            terminal = rows[0]["generation_events"][-1]
            terminal.update({
                "requested_action": "search",
                "action": "search",
                "parse_error": None,
                "terminal_rejection_reason": None,
                "valid_action": True,
                "executed_search": True,
            })
            self._write_jsonl(args.trace, rows)

            result = SMOKE.analyze(args)

        self.assertEqual(result["decision"], "NO-GO")
        self.assertFalse(result["checks"]["terminal_accepted_search_count"]["passed"])
        self.assertFalse(result["checks"]["terminal_executed_search_count"]["passed"])

    def test_valid_early_answer_does_not_require_terminal_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self._fixture(Path(temporary))
            rows = [json.loads(line) for line in args.trace.read_text().splitlines()]
            rows[0]["action_count"] = 1
            rows[0]["generation_events"] = [{
                "terminal_generation": False,
                "requested_action": "answer",
                "action": "answer",
                "parse_error": None,
                "valid_action": True,
                "done": True,
                "executed_search": False,
            }]
            self._write_jsonl(args.trace, rows)

            result = SMOKE.analyze(args)

        self.assertEqual(result["decision"], "GO")
        self.assertEqual(result["metrics"]["terminal_generation_count"], 79)

    def test_rejects_trajectory_without_early_or_terminal_answer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self._fixture(Path(temporary))
            rows = [json.loads(line) for line in args.trace.read_text().splitlines()]
            rows[0]["action_count"] = 4
            rows[0]["generation_events"] = rows[0]["generation_events"][:4]
            self._write_jsonl(args.trace, rows)

            with self.assertRaisesRegex(
                ValueError, "does not end with a valid early answer"
            ):
                SMOKE.analyze(args)

    def test_shared_scanner_builds_canonical_train_and_eval_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self._fixture(root)
            train_receipt = WANDB_HISTORY.build_receipt(
                args.wandb_dir,
                log=args.log,
                expected_steps=(1, 2),
            )
            eval_receipt = WANDB_HISTORY.build_receipt(args.wandb_dir)
            payload = WANDB_HISTORY.canonical_receipt_bytes(train_receipt)

            self.assertEqual(train_receipt["decision"], "GO")
            self.assertEqual(train_receipt["mode"], "train")
            self.assertEqual(eval_receipt["decision"], "GO")
            self.assertEqual(eval_receipt["mode"], "eval")
            self.assertEqual(train_receipt["metrics"]["history_records"], 2)
            self.assertEqual(
                json.loads(payload),
                train_receipt,
            )
            self.assertRegex(
                train_receipt["inputs"]["run_file_sha256"], r"^[0-9a-f]{64}$"
            )
            self.assertRegex(
                train_receipt["inputs"]["wandb_tree_sha256"], r"^[0-9a-f]{64}$"
            )

            output = root / "receipt.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(WANDB_HISTORY.__file__)),
                    "--wandb-dir",
                    str(args.wandb_dir),
                    "--log",
                    str(args.log),
                    "--expected-step",
                    "1",
                    "--expected-step",
                    "2",
                    "--output",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stdout.strip(), "GO")
            self.assertEqual(json.loads(output.read_bytes()), train_receipt)

            weak_verify = subprocess.run(
                [
                    sys.executable,
                    str(Path(WANDB_HISTORY.__file__)),
                    "--wandb-dir",
                    str(args.wandb_dir),
                    "--verify-receipt",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(weak_verify.returncode, 0)
            self.assertNotEqual(weak_verify.stdout.strip(), "GO")

            exact_verify = subprocess.run(
                [
                    sys.executable,
                    str(Path(WANDB_HISTORY.__file__)),
                    "--wandb-dir",
                    str(args.wandb_dir),
                    "--log",
                    str(args.log),
                    "--expected-step",
                    "1",
                    "--expected-step",
                    "2",
                    "--verify-receipt",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(exact_verify.returncode, 0, exact_verify.stderr)
            self.assertEqual(exact_verify.stdout.strip(), "GO")

            original_receipt = output.read_bytes()
            output.write_bytes(original_receipt + b" ")
            noncanonical_verify = subprocess.run(
                [
                    sys.executable,
                    str(Path(WANDB_HISTORY.__file__)),
                    "--wandb-dir",
                    str(args.wandb_dir),
                    "--log",
                    str(args.log),
                    "--expected-step",
                    "1",
                    "--expected-step",
                    "2",
                    "--verify-receipt",
                    str(output),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(noncanonical_verify.returncode, 0)
            self.assertNotEqual(noncanonical_verify.stdout.strip(), "GO")

    def test_receipt_output_cannot_overwrite_or_enter_wandb_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self._fixture(root)
            scanner = str(Path(WANDB_HISTORY.__file__))
            history = next(args.wandb_dir.rglob("run-*.wandb"))
            original_history = history.read_bytes()
            original_log = args.log.read_bytes()

            destinations = [
                history,
                args.log,
                args.wandb_dir / "receipt.json",
            ]
            log_alias = root / "train-log-hardlink"
            os.link(args.log, log_alias)
            destinations.append(log_alias)
            for destination in destinations:
                with self.subTest(destination=destination):
                    completed = subprocess.run(
                        [
                            sys.executable,
                            scanner,
                            "--wandb-dir",
                            str(args.wandb_dir),
                            "--log",
                            str(args.log),
                            "--expected-step",
                            "1",
                            "--expected-step",
                            "2",
                            "--output",
                            str(destination),
                        ],
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    self.assertNotEqual(completed.returncode, 0)
                    self.assertNotEqual(completed.stdout.strip(), "GO")

            self.assertEqual(history.read_bytes(), original_history)
            self.assertEqual(args.log.read_bytes(), original_log)

            output = root / "immutable-receipt.json"
            command = [
                sys.executable,
                scanner,
                "--wandb-dir",
                str(args.wandb_dir),
                "--log",
                str(args.log),
                "--expected-step",
                "1",
                "--expected-step",
                "2",
                "--output",
                str(output),
            ]
            first_write = subprocess.run(
                command, check=False, capture_output=True, text=True
            )
            self.assertEqual(first_write.returncode, 0, first_write.stderr)
            original_receipt = output.read_bytes()
            second_write = subprocess.run(
                command, check=False, capture_output=True, text=True
            )
            self.assertNotEqual(second_write.returncode, 0)
            self.assertEqual(output.read_bytes(), original_receipt)

    def test_catalog_question_replays_native_prompt_materialization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self._fixture(Path(temporary))
            catalog = [
                json.loads(line) for line in args.catalog.read_text().splitlines()
            ]
            catalog[0]["question"] = "Question 0"
            self._write_jsonl(args.catalog, catalog)

            result = SMOKE.analyze(args)
            self.assertEqual(result["decision"], "GO")

            traces = [
                json.loads(line) for line in args.trace.read_text().splitlines()
            ]
            traces[0]["question"] = "Different question?"
            self._write_jsonl(args.trace, traces)
            with self.assertRaisesRegex(ValueError, "question differs"):
                SMOKE.analyze(args)

    def test_rejects_smoke_input_changed_after_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self._fixture(Path(temporary))
            original_metric_lines = SMOKE._metric_lines

            def mutate_log_after_parse(path: Path):
                metrics = original_metric_lines(path)
                with path.open("a", encoding="utf-8") as handle:
                    handle.write("step:99 - actor/grad_norm:nan\n")
                return metrics

            with mock.patch.object(
                SMOKE, "_metric_lines", side_effect=mutate_log_after_parse
            ):
                with self.assertRaisesRegex(ValueError, "input changed"):
                    SMOKE.analyze(args)

            args = self._fixture(Path(temporary))
            original_catalog = SMOKE._catalog

            def replace_catalog_after_parse(path: Path):
                catalog = original_catalog(path)
                replacement = path.with_suffix(".replacement")
                shutil.copy2(path, replacement)
                os.replace(replacement, path)
                return catalog

            with mock.patch.object(
                SMOKE, "_catalog", side_effect=replace_catalog_after_parse
            ):
                with self.assertRaisesRegex(ValueError, "input changed"):
                    SMOKE.analyze(args)

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

            args = self._fixture(Path(temporary))
            with args.log.open("a", encoding="utf-8") as handle:
                handle.write(
                    "step:3 - actor/pg_loss:0.1 - actor/kl_loss:0.01 - "
                    "actor/entropy_loss:2 - actor/grad_norm:1 - actor/ppo_kl:0.01\n"
                )
            result = SMOKE.analyze(args)
            self.assertEqual(result["decision"], "NO-GO")
            self.assertFalse(
                result["checks"]["wandb_actor_metrics_match_log"]["passed"]
            )
            receipt = WANDB_HISTORY.build_receipt(
                args.wandb_dir, log=args.log, expected_steps=(1, 2)
            )
            self.assertEqual(receipt["decision"], "NO-GO")
            self.assertFalse(
                receipt["checks"]["actor_metrics_match_log"]["passed"]
            )

    def test_no_go_for_stats_only_wandb_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self._fixture(root, "stats_only")
            result = SMOKE.analyze(args)
            receipt = root / "stats-only-receipt.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(WANDB_HISTORY.__file__)),
                    "--wandb-dir",
                    str(args.wandb_dir),
                    "--output",
                    str(receipt),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            receipt_decision = json.loads(receipt.read_bytes())["decision"]
        self.assertEqual(result["decision"], "NO-GO")
        self.assertFalse(result["checks"]["wandb_offline_history"]["passed"])
        self.assertEqual(result["checks"]["wandb_offline_history"]["observed"], 0)
        self.assertTrue(result["checks"]["wandb_summary_present"]["passed"])
        self.assertTrue(result["checks"]["wandb_exit_zero"]["passed"])
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "NO-GO")
        self.assertEqual(receipt_decision, "NO-GO")

    def test_no_go_for_missing_step_nonfinite_or_log_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self._fixture(Path(temporary), "missing_step")
            result = SMOKE.analyze(args)
            self.assertEqual(result["decision"], "NO-GO")
            self.assertFalse(result["checks"]["wandb_history_steps"]["passed"])
            self.assertFalse(
                result["checks"]["wandb_actor_metrics_match_log"]["passed"]
            )

            args = self._fixture(Path(temporary), "nan")
            result = SMOKE.analyze(args)
            self.assertEqual(result["decision"], "NO-GO")
            self.assertFalse(
                result["checks"]["wandb_actor_metrics_match_log"]["passed"]
            )
            self.assertIn(
                "NaN",
                result["checks"]["wandb_actor_metrics_match_log"]["observed"][
                    "2"
                ]["actor/grad_norm"],
            )

            args = self._fixture(Path(temporary))
            text = args.log.read_text(encoding="utf-8")
            args.log.write_text(
                text.replace("actor/pg_loss:0.100000", "actor/pg_loss:0.200000", 1),
                encoding="utf-8",
            )
            result = SMOKE.analyze(args)
            self.assertEqual(result["decision"], "NO-GO")
            self.assertTrue(result["checks"]["finite_actor_pg_loss"]["passed"])
            self.assertFalse(
                result["checks"]["wandb_actor_metrics_match_log"]["passed"]
            )

    def test_no_go_for_nonzero_wandb_exit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = SMOKE.analyze(self._fixture(Path(temporary), "exit_one"))
        self.assertEqual(result["decision"], "NO-GO")
        self.assertEqual(
            result["checks"]["wandb_exit_zero"]["observed"]["codes"], [1]
        )
        self.assertFalse(result["checks"]["wandb_exit_zero"]["passed"])

    def test_no_go_for_missing_summary_duplicate_or_nonterminal_exit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self._fixture(root)
            self._write_synthetic_wandb(args, include_summary=False)

            result = SMOKE.analyze(args)
            self.assertEqual(result["decision"], "NO-GO")
            self.assertFalse(result["checks"]["wandb_summary_present"]["passed"])
            self.assertTrue(result["checks"]["wandb_exit_zero"]["passed"])

            args = self._fixture(root)
            self._write_synthetic_wandb(args, exit_codes=(0, 0))
            result = SMOKE.analyze(args)
            self.assertEqual(result["decision"], "NO-GO")
            self.assertEqual(
                result["checks"]["wandb_exit_zero"]["observed"]["codes"],
                [0, 0],
            )

            args = self._fixture(root)
            self._write_synthetic_wandb(args, summary_after_exit=True)
            result = SMOKE.analyze(args)
            self.assertEqual(result["decision"], "NO-GO")
            self.assertEqual(
                result["checks"]["wandb_exit_zero"]["observed"][
                    "last_record_type"
                ],
                "summary",
            )

    def test_rejects_wandb_file_set_changed_during_hashing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self._fixture(Path(temporary))
            original_sha256 = WANDB_HISTORY._sha256_file
            injected = False

            def inject_duplicate(path: Path) -> str:
                nonlocal injected
                digest = original_sha256(path)
                if path.name.startswith("run-") and path.suffix == ".wandb" and not injected:
                    injected = True
                    duplicate = args.wandb_dir / "raced" / "run-raced.wandb"
                    duplicate.parent.mkdir()
                    shutil.copy2(path, duplicate)
                return digest

            with mock.patch.object(
                WANDB_HISTORY, "_sha256_file", side_effect=inject_duplicate
            ):
                with self.assertRaisesRegex(ValueError, "file set changed"):
                    WANDB_HISTORY.scan_offline_run(
                        args.wandb_dir,
                        metric_keys=WANDB_HISTORY.CORE_ACTOR_METRICS,
                    )

            def unreadable_walk(root, *, topdown, onerror, followlinks):
                onerror(PermissionError("denied"))
                return iter(())

            with mock.patch.object(
                WANDB_HISTORY.os, "walk", side_effect=unreadable_walk
            ):
                with self.assertRaisesRegex(ValueError, "cannot enumerate"):
                    WANDB_HISTORY.scan_offline_run(args.wandb_dir)

    def test_no_go_for_split_or_duplicate_history_step(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self._fixture(root)
            actor_items = list(self._actor_values().items())
            rows = (
                (1, dict(actor_items[:2])),
                (1, dict(actor_items[2:])),
                (2, dict(actor_items)),
            )
            self._write_synthetic_wandb(args, history_rows=rows)

            result = SMOKE.analyze(args)
            self.assertEqual(result["decision"], "NO-GO")
            self.assertEqual(
                result["checks"]["wandb_history_steps"]["observed"], [1, 1, 2]
            )
            self.assertFalse(result["checks"]["wandb_history_steps"]["passed"])

    def test_ignores_wandb_metadata_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self._fixture(root)
            external_log = root / "core-debug.log"
            external_log.write_text("wandb core log", encoding="utf-8")
            logs = next(args.wandb_dir.rglob("run-*.wandb")).parent / "logs"
            logs.mkdir(exist_ok=True)
            (logs / "empty.log").touch()
            link = logs / "scanner-metadata-link.log"
            try:
                link.symlink_to(external_log)
            except OSError as error:
                self.skipTest(f"symlinks are unavailable: {error}")

            result = SMOKE.analyze(args)

        self.assertEqual(result["decision"], "GO")
        self.assertGreater(result["metrics"]["wandb"]["files"], 1)

    def test_symlink_only_wandb_history_is_not_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self._fixture(root)
            history = next(args.wandb_dir.rglob("run-*.wandb"))
            external_history = root / "external.wandb"
            shutil.copy2(history, external_history)
            history.unlink()
            try:
                history.symlink_to(external_history)
            except OSError as error:
                self.skipTest(f"symlinks are unavailable: {error}")

            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                SMOKE.analyze(args)

    def test_rejects_pseudo_truncated_or_multiple_wandb_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self._fixture(Path(temporary))
            history = next(args.wandb_dir.rglob("run-*.wandb"))
            history.write_bytes(b"offline-history")
            with self.assertRaisesRegex(ValueError, "cannot parse offline WandB run"):
                SMOKE.analyze(args)

            args = self._fixture(Path(temporary))
            history = next(args.wandb_dir.rglob("run-*.wandb"))
            history.write_bytes(history.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, "cannot parse offline WandB run"):
                SMOKE.analyze(args)

            args = self._fixture(Path(temporary))
            history = next(args.wandb_dir.rglob("run-*.wandb"))
            duplicate = args.wandb_dir / "duplicate" / "run-duplicate.wandb"
            duplicate.parent.mkdir()
            shutil.copy2(history, duplicate)
            with self.assertRaisesRegex(ValueError, "expected exactly one"):
                SMOKE.analyze(args)

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
