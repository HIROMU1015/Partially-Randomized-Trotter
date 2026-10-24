"""Fixed outer launcher/scope tests: opaque labels and stdlib mocks only."""
from dataclasses import replace
from fractions import Fraction as F
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(Path(__file__).resolve().parent)]
from test_b2d_evaluator import Barrier, FORBIDDEN
from test_b2d_stage1_launcher import mock_loader, mock_save, MockBackend as OldMockBackend
guard = Barrier(); sys.meta_path.insert(0, guard)
try:
    from trottertracks.b2d_adapter.adapter import ScalarTerm
    from trottertracks.b2d_adapter._published import symmetric_s2_sequence
    from trottertracks.b2d_evaluator.outer import INNER
    from trottertracks.b2d_evaluator.outer_backend import OuterDenseBackend
    from trottertracks.b2d_evaluator.outer_execution import (
        candidate_graph, fixed_cases, registry, validate_registry, resolved_graph,
        OuterScopeLedger, OuterRunContext, verify_execution,
    )
    from trottertracks.b2d_evaluator.scope import Stage1Failure
    from trottertracks.b2d_evaluator.launcher import _run, launch_outer, Evidence
finally:
    sys.meta_path.remove(guard)

SOURCE = "a" * 40


class MockBackend(OldMockBackend):
    def __init__(self, scope, fail=None, failure_at=1, code="MOCK_FAILURE"):
        super().__init__(scope)
        self.fail_kind, self.failure_at, self.failure_code = fail, failure_at, code
        self.kind_calls = []

    def perform(self, kind, args, params):
        self.kind_calls.append(kind)
        if kind == self.fail_kind and self.kind_calls.count(kind) == self.failure_at:
            raise Stage1Failure(self.failure_code)
        if kind == "exponential":
            self.exponential(*params)
        if kind == "outer_d_check":
            stage, = params
            blocks = stage.context.block_ids
            for weight in symmetric_s2_sequence(INNER[stage.inner_arm]()):
                for block in blocks[:-1] + blocks[::-1]:
                    self.exponential(block, stage.tau * F(weight) / (1 if block == blocks[-1] else 2))
        if kind == "outer_order_check":
            for label, duration in params[0]:
                self.exponential(label, duration)
        if kind == "outer_rte_moments":
            for boundary in ("rte.config", "rte.moments"):
                self.scope.call(boundary, lambda: "OPAQUE_MOMENT", identity=str(params[0]))
        if kind == "outer_signals":
            for _ in range(5):
                self.scope.call("signal.expectation", lambda: "OPAQUE_SIGNAL")
        return dict(mock_kind=kind)


def run_context(output):
    return OuterRunContext(ROOT, Path(output), SOURCE, registry(candidate_graph(SOURCE)),
                           dict(mode="MOCK_ENVIRONMENT"))


def run_mock(output, **backend_options):
    run = run_context(output)
    backends = []
    def factory(settings):
        backend = MockBackend(run.scope, **backend_options)
        backends.append(backend)
        return backend
    with patch("trottertracks.b2d_evaluator.preflight._check_source"):
        status = _run(run, outer=True, mock_only=True, input_loader=mock_loader,
                      input_saver=mock_save, backend_factory=factory)
    return status, run, backends


class OuterExecutionTests(unittest.TestCase):
    def setUp(self):
        guard = Barrier(); sys.meta_path.insert(0, guard)
        self.addCleanup(sys.meta_path.remove, guard)

    def cli(self, args):
        with patch.object(sys, "argv", ["run_b2d_outer_stage1.py", *args]), patch(
                "sys.stdout", new_callable=io.StringIO) as out:
            with self.assertRaises(SystemExit) as code:
                runpy.run_path(str(ROOT / "scripts/run_b2d_outer_stage1.py"), run_name="__main__")
        return code.exception.code, out.getvalue()

    def test_fixed_16_rows_and_science_scope_are_derived_from_new_graph(self):
        g = candidate_graph(SOURCE); s = registry(g)
        self.assertEqual(len(fixed_cases()), 16)
        self.assertEqual({c.total_time for c in fixed_cases()}, {F(9,50), F(-9,50)})
        self.assertEqual({c.inner_arm for c in fixed_cases()}, {"A2", "A3"})
        self.assertTrue(all(c.macro_steps == 2 and c.n_d == 1 for c in fixed_cases()))
        self.assertTrue(all(c.refinement == 2 for c in fixed_cases() if c.method == "refined-S2"))
        self.assertEqual((s["dispatch_total"], len(s["exponential_keys"])), (906, 490))
        self.assertEqual([s["boundaries"][k] for k in ("rte.config", "rte.moments", "signal.expectation")], [24,24,80])
        self.assertEqual(s["dispatch_counts"]["outer_d_check"], 40)
        self.assertEqual(s["dispatch_counts"]["outer_order_check"], 8)
        self.assertEqual(s["dispatch_counts"]["outer_rescale_check"], 16)
        validate_registry(json.loads(json.dumps(s)), g)

    def test_resolved_input_context_and_scalars_preserve_scope_layout(self):
        g = candidate_graph(SOURCE)
        context = replace(g.settings.context, preparation_id="MOCK_PREP", tail_id="MOCK_TAIL",
            block_ids=("renamed-one", "renamed-df-a", "renamed-df-b"),
            scalar_terms=(ScalarTerm("constant", F(3,7)),))
        actual = resolved_graph(g, context, NS(hamiltonian_id="MOCK_H", sector_id="MOCK_N1",
            state_id="MOCK_STATE"), dict(input_data_sha256="MOCK_INPUT"))
        self.assertEqual(registry(actual), registry(g))
        self.assertEqual(actual.settings.context, context)

    def test_graph_grid_scope_and_stage_mutations_are_rejected(self):
        g = candidate_graph(SOURCE)
        for bad in (replace(g, rows=g.rows[:-1]), replace(g, nodes=g.nodes[::-1]),
                    replace(g, settings=replace(g.settings, r_rte=3)),
                    replace(g, settings=replace(g.settings, q_values=(2,3)))):
            with self.subTest(bad=type(bad)), self.assertRaises((ValueError, Stage1Failure)):
                registry(bad)
        bad = registry(g); bad["boundaries"]["backend.expm"] -= 1
        with self.assertRaises(Stage1Failure): validate_registry(bad, g)

    def test_stage_identity_duplicate_and_comparison_before_gates_are_rejected(self):
        g = candidate_graph(SOURCE); ledger = OuterScopeLedger(registry(g), g)
        for identity in ("9/100", "abs_time_instead_of_signed_spec"):
            with self.assertRaises(Stage1Failure):
                ledger.call("rte.config", lambda: self.fail("Unregistered work"), identity=identity)
        identity = ledger.scope["rte_stage_specs"][0]
        ledger.call("rte.config", lambda: "OPAQUE_CONFIG", identity=identity)
        with self.assertRaises(Stage1Failure):
            ledger.call("rte.config", lambda: self.fail("Duplicate work"), identity=identity)
        index = next(i for i,n in enumerate(g.nodes) if n.kind == "outer_signals")
        with self.assertRaises(Stage1Failure):
            ledger.dispatch(index, g.nodes[index], lambda: self.fail("Early comparison"))

    def test_full_mock_pipeline_matches_every_registered_call_and_saves_16_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            status, run, backends = run_mock(directory)
            self.assertEqual(status["status"], "COMPLETED_DESCRIPTIVE_MANDATORY_STOP", status)
            self.assertEqual(status["calls"]["completed"], run.call_scope["boundaries"])
            result = json.loads((Path(directory) / "result.json").read_text())
            self.assertEqual(result["schema"], "b2d_outer_splitting_v1")
            self.assertEqual(len(result["rows"]), 16)
            self.assertEqual(len(backends), 1)
            self.assertEqual(result["mode"], "MOCK_ONLY")
            self.assertEqual(len(json.loads((Path(directory)/"partial_results.json").read_text())["rows"]), 16)
            with self.assertRaises(ValueError):
                with patch("trottertracks.b2d_evaluator.preflight._check_source"):
                    run.begin_evaluation(run.scope.graph)
            self.assertFalse(json.loads((Path(directory)/"STOP.json").read_text())["retry_allowed"])

    def test_all_correctness_failures_stop_before_any_comparison(self):
        for kind in ("validate_input", "outer_d_check", "outer_order_check", "outer_rescale_check"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                status, run, backends = run_mock(directory, fail=kind)
                self.assertEqual(status["status"], "FAILED_MANDATORY_STOP")
                self.assertNotIn("operator_error", backends[0].kind_calls)
                self.assertNotIn("outer_signals", backends[0].kind_calls)
                self.assertEqual(backends[0].kind_calls.count(kind), 1)
                self.assertEqual(json.loads((Path(directory)/"partial_results.json").read_text())["rows"], [])
                self.assertTrue((Path(directory)/"STOP.json").is_file())

    def test_partial_comparison_failure_preserves_rows_and_does_not_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            status, run, backends = run_mock(directory, fail="outer_signals", failure_at=2)
            rows = json.loads((Path(directory)/"partial_results.json").read_text())["rows"]
            self.assertEqual(status["status"], "FAILED_MANDATORY_STOP")
            self.assertTrue(0 < len(rows) < 16)
            self.assertEqual(rows[0]["outputs"]["signals"]["mock_kind"], "outer_signals")
            self.assertEqual(rows[0]["outputs"]["normalization"]["mock_kind"], "outer_normalization")
            self.assertEqual(backends[0].kind_calls.count("outer_signals"), 2)

    def test_numerical_inconclusive_and_save_failure_both_leave_STOP(self):
        with tempfile.TemporaryDirectory() as directory:
            status, _, _ = run_mock(directory, fail="outer_order_check", code="INCONCLUSIVE_NUMERICAL_PRECISION")
            self.assertEqual(status["status"], "INCONCLUSIVE_MANDATORY_STOP")
        original = Evidence.write
        def write(writer, name, value):
            if name == "result.json": raise OSError("MOCK_SAVE_FAILURE")
            return original(writer, name, value)
        with tempfile.TemporaryDirectory() as directory, patch.object(Evidence, "write", write):
            status, _, _ = run_mock(directory)
            self.assertEqual(status["status"], "EVIDENCE_SAVE_FAILED_MANDATORY_STOP")
            self.assertTrue((Path(directory)/"STOP.json").is_file())

    def test_fresh_output_collision_preserves_existing_STOP_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory)/"STOP.json"; marker.write_bytes(b"EXISTING_MOCK_STOP")
            with patch("trottertracks.b2d_evaluator.outer_execution.verify_execution",return_value=run_context(directory)), patch(
                    "trottertracks.b2d_evaluator.launcher._run") as work:
                status = launch_outer(ROOT, directory, SOURCE)
            self.assertEqual(status["status"], "FRESH_OUTPUT_FAILED_MANDATORY_STOP")
            work.assert_not_called()
            self.assertEqual(marker.read_bytes(), b"EXISTING_MOCK_STOP")

    def test_cli_missing_execute_and_inspect_never_launch_science(self):
        with patch("trottertracks.b2d_evaluator.launcher.launch_outer") as work:
            for args in ([], ["--execute"], ["--output", "MOCK"]):
                with patch("sys.stderr"):
                    code, _ = self.cli(args)
                self.assertEqual(code, 2)
            code, output = self.cli(["--inspect"])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(output)["scope"]["dispatch_total"], 906)
            work.assert_not_called()

    def test_execute_CLI_needs_no_approval_packet_and_is_stubbed(self):
        with patch("trottertracks.b2d_evaluator.launcher.launch_outer",return_value=dict(
                status="COMPLETED_DESCRIPTIVE_MANDATORY_STOP")) as work:
            code, _ = self.cli(["--execute", "--source-commit", SOURCE, "--output", "MOCK_NOT_CREATED"])
            self.assertEqual(code, 0)
            work.assert_called_once_with(ROOT, "MOCK_NOT_CREATED", SOURCE)

    def test_preflight_is_read_only_and_binds_source_conditions_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); output = root/"artifacts/research_b_outer_splitting/MOCK"
            with patch("trottertracks.b2d_evaluator.outer_execution._check_source"), patch(
                    "trottertracks.b2d_evaluator.outer_execution.read_fixture_definition"), patch(
                    "trottertracks.b2d_evaluator.outer_execution.inspect_environment",return_value={"MOCK":True}):
                run = verify_execution(root, output, SOURCE)
                self.assertIsInstance(run, OuterRunContext)
                self.assertFalse(output.parent.exists())
                bad = registry(candidate_graph(SOURCE)); bad["rows"] = 17
                with self.assertRaises(Stage1Failure): verify_execution(root, output, SOURCE, call_scope=bad)
                with self.assertRaises(Stage1Failure):
                    verify_execution(root, output, SOURCE, graph=candidate_graph("b"*40))
                with self.assertRaises(ValueError):
                    verify_execution(root, root/"artifacts/research_b_b2d_stage1/MOCK", SOURCE)

    def test_actual_source_rejection_never_reaches_science_or_output(self):
        with patch("trottertracks.b2d_evaluator.launcher._run") as work:
            status = launch_outer(ROOT, ROOT/"artifacts/research_b_outer_splitting/MOCK_NOT_CREATED", "0"*40)
        work.assert_not_called()
        self.assertEqual(status["status"], "REJECTED_BEFORE_SCIENCE")

    def test_five_target_signal_glue_uses_only_opaque_stub_inputs(self):
        backend = OuterDenseBackend.__new__(OuterDenseBackend)
        # Artificial scalar callback values; NO operator/state/signal evaluator.
        record = dict(X=0.0, Y=0.0, radius=0.0, principal_phase=None, phase_status="NOT_REQUESTED")
        backend._signals = lambda values: {name:dict(record) for name in ("physical","ideal_tail","corrected","raw")}
        calls = []
        backend.signal = lambda matrix, state: calls.append((matrix,state)) or complex(3,4)
        backend.i = NS(state="OPAQUE_STATE_LABEL")
        backend.execution = NS(science_call=lambda name, fn: fn())
        value = backend._outer_signals("PHYSICAL", "OUTER", "IDEAL", "CORRECTED", "RAW")
        self.assertEqual(value["exact_outer_physical_signal_error"], 5)
        self.assertEqual(calls, [("OUTER", "OPAQUE_STATE_LABEL")])
        self.assertIn("raw_physical_signal_error", value)
        self.assertFalse(FORBIDDEN.intersection(n.split(".")[0] for n in sys.modules))


if __name__ == "__main__":
    unittest.main()
