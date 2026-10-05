"""Focused launch/cap controls; no synthetic Git commits or science launches."""
from copy import deepcopy
from fractions import Fraction as F
import importlib.util
import json
from pathlib import Path
import resource
import signal
import tempfile
import unittest
from unittest.mock import patch

from trottertracks.algorithm_codesign.synthesis_placement.wrapper_launch import (
    validate_binding, verify_source, consume_marker, check_limits, serialize_result, BudgetGuard,
)

ROOT = Path(__file__).resolve().parents[3]
source = importlib.util.spec_from_file_location("sp1_runner_under_test",
    ROOT/"scripts/tracks/algorithm_codesign/run_sp1_wrapper_pilot.py")
runner = importlib.util.module_from_spec(source)
source.loader.exec_module(runner)


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.s, self.a, self.digest = "1"*40, "2"*40, "3"*64
        self.allowed = {"authorization.json", "receipt.md"}
        self.auth = {"status": "APPROVED_FOR_ONE_SP1_RUN", "science_execution_authorized": True,
            "source_commit": self.s, "contract_sha256": self.digest, "runs": 1, "retries": 0,
            "mandatory_STOP": True, "explicit_execution_instruction": "fixed source SP-1を一回だけ実行し必ず停止"}

    def verify(self, **overrides):
        args = dict(auth=self.auth, contract_hash=self.digest, head=self.a, parents=[self.s],
                    changed=["authorization.json"], dirty=False, allowed=self.allowed)
        args.update(overrides)
        return validate_binding(**args)

    def test_clean_direct_authorization_child_is_accepted(self):
        self.verify(changed=["authorization.json", "receipt.md"])

    def test_science_source_or_review_grandchild_head_is_refused(self):
        with self.assertRaises(PermissionError):
            self.verify(head=self.s)
        with self.assertRaises(PermissionError):
            self.verify(parents=[self.a])
        with self.assertRaises(PermissionError):
            self.verify(parents=[self.s, self.a])

    def test_source_change_in_auth_commit_is_refused(self):
        with self.assertRaises(PermissionError):
            self.verify(changed=["authorization.json", "wrapper_adapter.py"])
        with self.assertRaises(PermissionError):
            self.verify(dirty=True)

    def test_pending_SP05_or_retry_authorization_is_refused(self):
        for field, value in (("source_commit", None), ("status", "APPROVED_FOR_ONE_SP05_RUN"),
            ("runs", 2), ("runs", True), ("retries", 1), ("mandatory_STOP", False),
            ("science_execution_authorized", False), ("explicit_execution_instruction", None)):
            auth = dict(self.auth, **{field: value})
            with self.assertRaises(PermissionError):
                self.verify(auth=auth)

    def test_contract_mismatch_is_refused(self):
        with self.assertRaises(PermissionError):
            self.verify(contract_hash="4"*64)

    def test_real_pending_runner_refuses_before_marker_or_pilot(self):
        # Isolated pending fixture: safe even if tests are later run on child A.
        # Never read a real execution authorization or consume its marker.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract_path = root/"contract.json"
            contract_path.write_text(json.dumps({"authorization_path": "authorization.json"}))
            (root/"authorization.json").write_text(json.dumps({"source_commit": None,
                "science_execution_authorized": False}))
            with patch.object(runner, "ROOT", root), patch.object(runner, "CONTRACT", contract_path), \
                 patch.object(runner, "consume_marker") as marker, \
                 patch.object(runner, "perform_pilot") as science, \
                 patch.object(runner, "verify_runtime") as runtime:
                with self.assertRaises(PermissionError):
                    runner.run()
                marker.assert_not_called()
                science.assert_not_called()
                runtime.assert_not_called()

    def test_one_shot_marker_is_exclusive_and_not_removed(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)/"output"
            p = consume_marker(directory, {"runs": 1, "retries": 0})
            with self.assertRaises(FileExistsError):
                consume_marker(directory, {"runs": 1})
            self.assertEqual(json.loads(p.read_text())["runs"], 1)

    def test_existing_output_is_not_overwritten_without_a_marker(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory/"result.json").write_text("existing evidence")
            with self.assertRaises(FileExistsError):
                consume_marker(directory, {"runs": 1})
            self.assertEqual((directory/"result.json").read_text(), "existing evidence")

    def test_incomplete_focused_source_preparation_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/"manifest.json").write_text(json.dumps({"focused_tests_passed": False}))
            with self.assertRaises(PermissionError):
                verify_source(root, {"source_manifest_path": "manifest.json"})

    def test_caps_cover_wall_cpu_and_memory(self):
        caps = {"wall_seconds": 2, "cpu_seconds": 1, "RSS_MiB": 1}
        check_limits(caps, 1, 0.5, 1024)
        for elapsed, cpu, rss, exception in ((2, 0, 0, TimeoutError),
                                           (0, 1, 0, TimeoutError), (0, 0, 1025, MemoryError)):
            with self.assertRaises(exception):
                check_limits(caps, elapsed, cpu, rss)

    def test_guard_checks_are_usable_without_installing_a_timer(self):
        guard = BudgetGuard({"wall_seconds": 2, "cpu_seconds": 1, "RSS_MiB": 1})
        with patch.object(guard, "usage", return_value={"wall_seconds": 2, "cpu_seconds": 0, "peak_RSS_KiB": 0}):
            with self.assertRaises(TimeoutError):
                guard.check()
        with self.assertRaises(TimeoutError):
            guard.cpu_signal()

    def test_failed_guard_install_restores_handlers_limits_and_timer(self):
        guard = BudgetGuard({"wall_seconds": 2, "cpu_seconds": 1, "RSS_MiB": 1,
                             "virtual_address_MiB": 2, "watchdog_period_seconds": 0.1})
        # Mock all OS mutation: this semantic test installs no real cap/timer.
        with patch("signal.getsignal", return_value=signal.SIG_DFL), \
             patch("signal.getitimer", return_value=(0, 0)), \
             patch("resource.getrlimit", return_value=(resource.RLIM_INFINITY, resource.RLIM_INFINITY)), \
             patch("signal.signal") as handlers, patch("resource.setrlimit") as limits, \
             patch("signal.setitimer", side_effect=[RuntimeError("timer unavailable"), None]) as timer:
            with self.assertRaises(RuntimeError):
                guard.__enter__()
            self.assertEqual(timer.call_args_list[-1].args, (signal.ITIMER_REAL, 0))
            self.assertEqual(handlers.call_args_list[-1].args, (signal.SIGXCPU, signal.SIG_DFL))
            self.assertEqual(limits.call_args_list[-1].args,
                             (resource.RLIMIT_AS, (resource.RLIM_INFINITY, resource.RLIM_INFINITY)))

    def test_serialization_rejects_unsafe_numbers_and_output_cap(self):
        for result in ({"value": float("nan")}, {"value": F(1, 3)}, {"value": "x"*100}):
            with self.assertRaises((ValueError, TypeError, RuntimeError)):
                serialize_result(result, 50)
        self.assertEqual(json.loads(serialize_result({"shots": 1000000001}, 100)), {"shots": 1000000001})

    def test_plan_source_requires_full_sha(self):
        with self.assertRaises(PermissionError):
            runner.static_plan(ROOT, {}, "HEAD")


if __name__ == "__main__":
    unittest.main()
