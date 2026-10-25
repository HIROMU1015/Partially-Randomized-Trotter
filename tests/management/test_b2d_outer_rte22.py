"""22-row metadata and opaque pipeline tests; science imports are blocked."""
from dataclasses import dataclass, replace
from fractions import Fraction as F
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace as NS, ModuleType
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"src"), str(Path(__file__).resolve().parent)]
from test_b2d_evaluator import Barrier, FORBIDDEN
from test_b2d_outer_execution import MockBackend as PreviousMockBackend
from test_b2d_stage1_launcher import mock_loader, mock_save
guard = Barrier(); sys.meta_path.insert(0, guard)
try:
    from trottertracks.b2d_evaluator.outer import RTEStage, normalization_product
    from trottertracks.b2d_evaluator.outer_backend import OuterDenseBackend
    from trottertracks.b2d_evaluator.outer_rte_execution import (
        CANDIDATES, OUTPUT_PARENT, candidate_graph, registry, validate_registry,
        resolved_graph, verify_execution, budget_metadata, bias_qualification,
        OuterRTE22ScopeLedger, OuterRTE22RunContext,
    )
    from trottertracks.b2d_evaluator.launcher import _run, launch_outer_rte22, Evidence
    from trottertracks.b2d_evaluator.scope import Stage1Failure
finally:
    sys.meta_path.remove(guard)
SOURCE = "a"*40


class MockBackend(PreviousMockBackend):
    def perform(self, kind, args, params):
        result = super().perform(kind, args, params)
        if kind == "outer_taylor_bound":
            self.scope.call("rte.truncation", lambda: "OPAQUE_BOUND", identity=str(params[0]))
        if kind == "outer_taylor_total":
            self.scope.call("bound.compose", lambda: "OPAQUE_TOTAL_BOUND")
        return result


def run_context(output):
    return OuterRTE22RunContext(ROOT, Path(output), SOURCE, registry(candidate_graph(SOURCE)), {"mock":True})


def run_mock(output, **options):
    run, backends = run_context(output), []
    def factory(settings):
        backend = MockBackend(run.scope, **options); backends.append(backend)
        return backend
    with patch("trottertracks.b2d_evaluator.preflight._check_source"):
        status = _run(run, outer="rte22", mock_only=True, input_loader=mock_loader,
                      input_saver=mock_save, backend_factory=factory)
    return status, run, backends


def fake_signals(bias):
    records = {k:dict(X=0., Y=0., radius=0.)
               for k in ("physical", "exact_outer", "ideal_tail", "corrected", "raw")}
    records.update({k+"_physical_signal_error":bias
                    for k in ("exact_outer", "ideal_tail", "corrected", "raw")})
    return records


class OuterRTE22Tests(unittest.TestCase):
    def setUp(self):
        guard = Barrier(); sys.meta_path.insert(0, guard)
        self.addCleanup(sys.meta_path.remove, guard)

    def cli(self, args):
        with patch.object(sys, "argv", ["run_b2d_outer_rte22.py", *args]), patch(
                "sys.stdout", new_callable=io.StringIO) as out:
            with self.assertRaises(SystemExit) as code:
                runpy.run_path(str(ROOT/"scripts/run_b2d_outer_rte22.py"), run_name="__main__")
        return code.exception.code, out.getvalue()

    def test_exact_grid_and_per_candidate_law_in_every_occurrence(self):
        graph = candidate_graph(SOURCE)
        self.assertEqual(len(graph.rows), 22)
        self.assertEqual({r["case"]["total_time"] for r in graph.rows}, {F(9,50),F(-9,50)})
        for row in graph.rows:
            cid, method, arm, q, f, r, k = next(c for c in CANDIDATES if c[0] == row["candidate_id"])
            case = row["case"]
            self.assertEqual((case["method"],case["inner_arm"],case["macro_steps"],case["refinement"],case["n_d"]),
                             (method,arm,q,f,1))
            stages = [s for s in row["plan"]["body"] if "payload" in s]
            count = q*(f or 1)*(3 if method == "outer-Yoshida4" else 2 if method == "SABA2" else 1)
            self.assertEqual(len(stages), count)
            self.assertTrue(all((s["payload"]["rte_steps"],s["payload"]["cutoff"]) == (r,k) for s in stages))
            if method == "outer-Yoshida4":
                self.assertTrue(any(s["evolution_time"] * case["total_time"] < 0 for s in stages))
            if cid == "F1": self.assertEqual((q*f,len(stages)), (16,16))
            if cid == "S1": self.assertEqual((q,len(stages)), (4,8))
            self.assertEqual(len({s["occurrence_id"] for s in stages}), count)

    def test_cache_identity_separates_signed_time_r_K_and_bound_inputs(self):
        g = candidate_graph(SOURCE)
        nodes = [n for n in g.nodes if n.kind == "outer_rte_moments"]
        self.assertEqual(len(nodes), 96)
        specs = {n.parameters[0] for n in nodes}
        self.assertEqual({(s.rte_steps,s.cutoff) for s in specs}, {(2,2),(4,2),(2,4)})
        self.assertTrue(all(s.duration != 0 for s in specs))
        for n in nodes:
            self.assertEqual(n.parameters[1], (g.settings.context.tail_id,g.settings.hamiltonian_id,
                g.settings.input_data_sha256,"complex128","analytic_moments_no_sampling"))
        # A2/A3 reuse identical R moments, retaining each row's full list.
        for rows in (g.rows[:11],g.rows[11:]):
            a,b = rows[2],rows[4]
            self.assertEqual(g.nodes[a["outputs"]["normalization"]].dependencies,
                             g.nodes[b["outputs"]["normalization"]].dependencies)
            self.assertEqual(a["cost"]["R_occurrences"],6)

    def test_new_scope_is_exact_and_old_scopes_are_unchanged(self):
        from trottertracks.b2d_evaluator import outer_execution, scope
        graph = candidate_graph(SOURCE); s = registry(graph)
        self.assertEqual((s["rows"],s["dispatch_total"],len(s["exponential_keys"])), (22,1239,274))
        self.assertEqual([s["boundaries"][k] for k in
            ("rte.config","rte.moments","rte.truncation","bound.compose","signal.expectation")], [96,96,96,14,110])
        self.assertEqual(sum(v for k,v in s["boundaries"].items() if k.startswith("input.")),10)
        self.assertEqual(outer_execution.registry(outer_execution.candidate_graph(SOURCE))["dispatch_total"],906)
        self.assertEqual(scope.registry(scope.candidate_graph(SOURCE))["dispatch_total"],941)
        validate_registry(json.loads(json.dumps(s)), graph)

    def test_input_binding_preserves_scope_but_binds_cache_to_actual_inputs(self):
        g = candidate_graph(SOURCE)
        context = replace(g.settings.context, preparation_id="MOCK_PREP",tail_id="MOCK_TAIL",
                          block_ids=("mock-one","mock-two","mock-three"))
        actual = resolved_graph(g,context,NS(hamiltonian_id="MOCK_H",sector_id="MOCK_N1",state_id="MOCK_STATE"),
                                dict(input_data_sha256="MOCK_ARRAY_ID"))
        self.assertEqual(registry(actual),registry(g))
        self.assertTrue(all(n.parameters[1][:3] == ("MOCK_TAIL","MOCK_H","MOCK_ARRAY_ID")
                           for n in actual.nodes if n.kind == "outer_rte_moments"))

    def test_graph_scope_and_law_mutations_are_rejected(self):
        g = candidate_graph(SOURCE)
        index = next(i for i,n in enumerate(g.nodes) if n.kind == "outer_rte_moments")
        node = g.nodes[index]
        changed = replace(node, parameters=(replace(node.parameters[0],cutoff=4),node.parameters[1]))
        for bad in (replace(g,rows=g.rows[:-1]), replace(g,nodes=g.nodes+(node,)),
                    replace(g,nodes=g.nodes[:index]+(changed,)+g.nodes[index+1:]),
                    replace(g,settings=replace(g.settings,r_rte=4)),
                    replace(g,settings=replace(g.settings,n_d_values=(1,2)))):
            with self.subTest(mutation=type(bad)), self.assertRaises((ValueError,Stage1Failure)):
                registry(bad)
        bad = registry(g); bad["boundaries"]["rte.moments"] -= 1
        with self.assertRaises(Stage1Failure): validate_registry(bad,g)

    def test_mock_pipeline_matches_every_registered_boundary_and_saves_all_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            status,run,backends = run_mock(directory)
            self.assertEqual(status["status"],"COMPLETED_DESCRIPTIVE_MANDATORY_STOP",status)
            self.assertEqual(status["calls"]["completed"],run.call_scope["boundaries"])
            result = json.loads((Path(directory)/"result.json").read_text())
            self.assertEqual(len(result["rows"]),22)
            self.assertEqual(result["mode"],"MOCK_ONLY")
            self.assertEqual(len(backends),1)
            self.assertEqual(len(json.loads((Path(directory)/"partial_results.json").read_text())["rows"]),22)
            self.assertFalse(json.loads((Path(directory)/"STOP.json").read_text())["retry_allowed"])
            with patch("trottertracks.b2d_evaluator.preflight._check_source"), self.assertRaises(ValueError):
                run.begin_evaluation(run.scope.graph)

    def test_all_gate_failures_prevent_comparison_and_preserve_STOP(self):
        for kind in ("validate_input","outer_tail_bound_check","outer_d_check","outer_order_check",
                     "outer_rte_check","outer_rescale_check"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                status,run,backends = run_mock(directory,fail=kind)
                self.assertEqual(status["status"],"FAILED_MANDATORY_STOP")
                self.assertNotIn("operator_error",backends[0].kind_calls)
                self.assertNotIn("outer_signals",backends[0].kind_calls)
                self.assertTrue((Path(directory)/"STOP.json").is_file())
        g = candidate_graph(SOURCE); ledger = OuterRTE22ScopeLedger(registry(g),g)
        idx = next(i for i,n in enumerate(g.nodes) if n.kind == "outer_signals")
        with self.assertRaises(Stage1Failure): ledger.dispatch(idx,g.nodes[idx],lambda:self.fail("Early comparison"))

    def test_partial_failure_inconclusive_and_save_failure_stop_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            status,run,backends = run_mock(directory,fail="outer_signals",failure_at=2)
            self.assertEqual(status["status"],"FAILED_MANDATORY_STOP")
            rows = json.loads((Path(directory)/"partial_results.json").read_text())["rows"]
            self.assertTrue(0 < len(rows) < 22)
            self.assertEqual(backends[0].kind_calls.count("outer_signals"),2)
        with tempfile.TemporaryDirectory() as directory:
            status,_,_ = run_mock(directory,fail="outer_rte_check",code="INCONCLUSIVE_NUMERICAL_PRECISION")
            self.assertEqual(status["status"],"INCONCLUSIVE_MANDATORY_STOP")
        original = Evidence.write
        def write(writer,name,value):
            if name == "result.json": raise OSError("MOCK_SAVE_FAILURE")
            return original(writer,name,value)
        with tempfile.TemporaryDirectory() as directory, patch.object(Evidence,"write",write):
            status,_,_ = run_mock(directory)
            self.assertEqual(status["status"],"EVIDENCE_SAVE_FAILED_MANDATORY_STOP")
            self.assertTrue((Path(directory)/"STOP.json").is_file())

    def test_fresh_collision_preserves_old_STOP_and_other_namespace_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory)/"STOP.json"; marker.write_bytes(b"OLD_MOCK_STOP")
            with patch("trottertracks.b2d_evaluator.outer_rte_execution.verify_execution",return_value=run_context(directory)), patch(
                    "trottertracks.b2d_evaluator.launcher._run") as work:
                status = launch_outer_rte22(ROOT,directory,SOURCE)
            self.assertEqual(status["status"],"FRESH_OUTPUT_FAILED_MANDATORY_STOP")
            self.assertEqual(marker.read_bytes(),b"OLD_MOCK_STOP"); work.assert_not_called()
            with patch("trottertracks.b2d_evaluator.outer_execution._check_source"), patch(
                    "trottertracks.b2d_evaluator.outer_execution.read_fixture_definition"), patch(
                    "trottertracks.b2d_evaluator.outer_execution.inspect_environment",return_value={}):
                root = Path(directory)
                run = verify_execution(root,root/OUTPUT_PARENT/"FRESH",SOURCE)
                self.assertFalse(run.output.parent.exists())
                with self.assertRaises(ValueError):
                    verify_execution(root,root/"artifacts/research_b_outer_splitting/OLD",SOURCE)

    def test_cli_is_metadata_only_without_explicit_execute_and_needs_no_packet(self):
        with patch("trottertracks.b2d_evaluator.launcher.launch_outer_rte22",return_value={
                "status":"COMPLETED_DESCRIPTIVE_MANDATORY_STOP"}) as work:
            for args in ([],["--execute"],["--output","MOCK"]):
                with patch("sys.stderr"):
                    code,_ = self.cli(args)
                self.assertEqual(code,2)
            code,text = self.cli(["--inspect"])
            self.assertEqual((code,json.loads(text)["scope"]["rows"]),(0,22)); work.assert_not_called()
            code,_ = self.cli(["--execute","--source-commit",SOURCE,"--output","MOCK_NOT_CREATED"])
            self.assertEqual(code,0); work.assert_called_once_with(ROOT,"MOCK_NOT_CREATED",SOURCE)
        self.assertFalse(FORBIDDEN.intersection(n.split(".")[0] for n in sys.modules))

    def test_budget_does_not_invent_u_num_or_shots_even_for_zero_fake_bias(self):
        for bias in (0.,5e-9,5e-10,1.):
            result = bias_qualification(fake_signals(bias))
            self.assertIsNone(result["u_num"])
            self.assertTrue(all(t["deterministic_qualified"] is None for t in result["targets"]))
            self.assertEqual([t["observed_bias_within_budget"] for t in result["targets"]],
                             [bias <= 5e-9,bias <= 5e-10])
        budget = budget_metadata()
        self.assertEqual((budget["X_alpha"]+budget["Y_alpha"],budget["alpha"]),(.05,.05))
        self.assertIsNone(budget["shots"]); self.assertIsNone(budget["implementation_gamma"])
        with self.assertRaises(Stage1Failure): bias_qualification(fake_signals(float("nan")))

    def test_normalization_keeps_multiplicity_and_config_glue_is_stubbed(self):
        # Arbitrary dummy N values, unrelated to the fixture or any RTE law.
        self.assertEqual(normalization_product([NS(normalization_product=2),NS(normalization_product=3),
                                               NS(normalization_product=2)])["normalization_product"],12)
        calls = []
        for r,k in ((2,2),(4,2),(2,4)):
            spec = RTEStage(F(-1,7),r,k,1.,41)
            spec.make_config("OPAQUE_TAIL",lambda tail,**kw:calls.append((tail,kw)))
        self.assertEqual([(c[1]["rte_steps"],c[1]["finite_taylor_order"]) for c in calls],[(2,2),(4,2),(2,4)])
        self.assertTrue(all(c[1]["evolution_time"] < 0 for c in calls))

    def test_bound_glue_reuses_existing_functions_with_stub_modules_only(self):
        # Never import real trotterlib, or call the real bound implementation.
        @dataclass
        class Record:
            step_truncation_residual_bound: float = 0.125
            rte_steps: int = 2
        calls = []
        module = ModuleType("trotterlib.rte")
        module.rte_occurrence_truncation_from_config = lambda label,config: calls.append((label,config)) or Record()
        module.compose_truncation_residual_bounds = lambda triples: calls.append(triples) or 0.5
        backend = OuterDenseBackend.__new__(OuterDenseBackend); backend.validated = True
        backend.execution = NS(science_call=lambda name,fn,**kw:fn())
        spec = RTEStage(F(-1,7),4,2,1.,41); params = (spec,("MOCK_TAIL","MOCK_H","MOCK_ARRAYS","complex128","analytic_moments_no_sampling"))
        backend.outer_configs = {params:("OPAQUE_CONFIG","OPAQUE_DISTRIBUTION")}
        with patch.dict(sys.modules,{"trotterlib":ModuleType("trotterlib"),"trotterlib.rte":module}):
            bound = backend.perform("outer_taylor_bound",("OPAQUE_GATE",),params)
            total = backend.perform("outer_taylor_total",(bound,bound),())
        self.assertEqual(calls[0],(str(spec),"OPAQUE_CONFIG"))
        self.assertEqual(calls[1],[(.125,2,1),(.125,2,1)])
        self.assertEqual(total["occurrence_count"],2)
        self.assertIsNone(total["floating_point_uncertainty"])
        self.assertFalse(FORBIDDEN.intersection(n.split(".")[0] for n in sys.modules))

    def test_signed_config_and_normalization_gate_glue_with_opaque_operators(self):
        @dataclass
        class Config:
            tail_id: str = "TAIL_LABEL"
            tail_hash: str = "MOCK_TAIL"
            lambda_r: float = 1.
            evolution_time: float = -1/7
            rte_steps: int = 4
            finite_taylor_order: int = 2
            truncation_tolerance: float = 1.
            seed: int = 41
            taylor_distribution: str = "rte_even_taylor_paired"
            distribution_normalization: float = 1.
        @dataclass
        class Distribution:
            orders: tuple = (0,2)
        backend = OuterDenseBackend.__new__(OuterDenseBackend); backend.validated = True
        backend.s = NS(context=NS(tail_id="MOCK_TAIL"),hamiltonian_id="MOCK_H",input_data_sha256="MOCK_ARRAYS",
                       precision="complex128",finite_bit_policy="analytic_moments_no_sampling",numerical_atol=1e-12)
        backend.i = NS(symbolic_tail=NS(tail_id="TAIL_LABEL",tail_hash="MOCK_TAIL",lambda_r=1.))
        spec = RTEStage(F(-1,7),4,2,1.,41)
        binding = ("MOCK_TAIL","MOCK_H","MOCK_ARRAYS","complex128","analytic_moments_no_sampling")
        params = (spec,binding); config = Config()
        backend.outer_configs = {params:(config,Distribution())}
        moments = NS(normalization_product=1.,attenuation_factor=1.,corrected_operator="OPAQUE",attenuated_event_mean_operator="OPAQUE")
        result = backend.perform("outer_rte_check",(moments,),params)
        self.assertEqual(result["config"]["evolution_time"],-1/7)
        for change in (dict(evolution_time=1/7),dict(rte_steps=2),dict(finite_taylor_order=4)):
            with self.subTest(change=change), self.assertRaises(Stage1Failure):
                backend._check_rte_config(spec,binding,replace(config,**change))
        with self.assertRaises(Stage1Failure): backend._check_rte_config(spec,("WRONG",*binding[1:]),config)
        with self.assertRaises(Stage1Failure):
            backend.perform("outer_rte_check",(NS(normalization_product=2.,attenuation_factor=1.),),params)
        calls = []
        def make_config(tail, **kw):
            calls.append(("config",tail,kw))
            return replace(config,evolution_time=kw["evolution_time"],rte_steps=kw["rte_steps"],
                           finite_taylor_order=kw["finite_taylor_order"],seed=kw["seed"]),Distribution()
        backend.make_rte_config = make_config
        backend.i.normalized_tail = "OPAQUE_NORMALIZED_TAIL"
        backend.finite_rte_operator_moments = lambda tail,c: calls.append(("moments",tail,c)) or moments
        backend.execution = NS(science_call=lambda name,fn,**kw:fn())
        for duration,r,k in ((F(-1,7),2,2),(F(1,7),4,2),(F(-1,7),2,4)):
            current = RTEStage(duration,r,k,1.,41)
            self.assertIs(backend.perform("outer_rte_moments",(),(current,binding)),moments)
            self.assertEqual(calls[-1][2].evolution_time,float(duration))
            self.assertEqual((calls[-1][2].rte_steps,calls[-1][2].finite_taylor_order),(r,k))


if __name__ == "__main__":
    unittest.main()
