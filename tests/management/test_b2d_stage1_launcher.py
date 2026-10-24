"""Artificial scientific pipeline tests; no arrays, circuits or science imports."""
from dataclasses import replace
from fractions import Fraction as F
import importlib.abc
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"src"),str(Path(__file__).resolve().parent)]
from test_b2d_evaluator import Barrier, FORBIDDEN
barrier = Barrier()
sys.meta_path.insert(0,barrier)
try:
    from trottertracks.b2d_evaluator.scope import (
        ScopeLedger, Stage1Failure, STOP, candidate_graph, registry, exp_key,
        validate_registry, resolved_graph,
    )
    from trottertracks.b2d_evaluator.preflight import RunContext, validate_scope
    from trottertracks.b2d_evaluator.launcher import launch, _run, Evidence
    from trottertracks.b2d_adapter.adapter import published_math_sequences
finally:
    sys.meta_path.remove(barrier)


def preparation_receipt():
    # Arbitrary legal mock permutation; NEVER represents resolved fixture ranking.
    return dict(selected=[2,0],residual=[1],D_blocks=[
        dict(id="D0",kind="one_body",factor_index=None),
        dict(id="D1",kind="df_fragment",factor_index=2),
        dict(id="D2",kind="df_fragment",factor_index=0)],
        tail_bases=[dict(basis_id="MOCK_BASIS",basis_hash="MOCK_HASH",operations=2)],
        tail_components=[dict(fragment="df-fragment-1",basis_id="MOCK_BASIS",support=[0])])


class MockRun:
    def __init__(self,output):
        self.source_commit="MOCK_SOURCE"
        self.call_scope=registry(candidate_graph(self.source_commit))
        self.environment={"mode":"MOCK_METADATA_ONLY"}
        self.source_root=ROOT
        self.output=Path(output)
        self.evaluations=0
    def attach_scope(self,scope): self.scope=scope
    def begin_input(self,root,recipe): pass
    def begin_evaluation(self,graph):
        self.check_settings(graph.settings)
        self.evaluations+=1
        if self.evaluations!=1: raise ValueError("No retry or resume")
    def science_call(self,name,function,**kwargs): return self.scope.call(name,function,**kwargs)
    def record_resolution(self,name,value): self.scope.record_resolution(name,value)
    def bind_input(self,context,inputs,provenance):
        self.binding=(context,inputs.hamiltonian_id,inputs.sector_id,inputs.state_id,provenance["input_data_sha256"])
    def check_settings(self,s):
        if self.binding != (s.context,s.hamiltonian_id,s.sector_id,s.state_id,s.input_data_sha256):
            raise Stage1Failure("INPUT_IDENTITY_ORDER_MISMATCH")


def mock_loader(root,*,execution):
    from trottertracks.b2d_evaluator.input_package import InputRecipe
    execution.begin_input(root,InputRecipe())
    for name in ("fixture","split","preparation","sector","full"):
        execution.science_call("input."+name,lambda: "MOCK_LABEL_NO_SCIENCE")
    execution.record_resolution("preparation",preparation_receipt())
    execution.record_resolution("sector",dict(n_modes=2,N=1,basis_indices=[1,2],occupied_modes=[0]))
    for position in range(3):
        execution.science_call("input.block",lambda: "MOCK_BLOCK_LABEL",identity=position)
    for name in ("tail","state"):
        execution.science_call("input."+name,lambda: "MOCK_LABEL_NO_ARRAY")
    context=replace(candidate_graph("MOCK").settings.context,
                    preparation_id="MOCK_PREPARATION",tail_id="MOCK_TAIL")
    inputs=NS(hamiltonian_id="MOCK_H",sector_id="MOCK_N1",state_id="MOCK_OCCUPATION")
    provenance=dict(input_data_sha256="MOCK_NOT_ARRAY_HASH",mode="MOCK_ONLY")
    execution.record_resolution("input",provenance)
    execution.bind_input(context,inputs,provenance)
    return context,inputs,provenance


class MockBackend:
    is_mock=True
    def __init__(self,scope,fail=None): self.scope,self.fail,self.cache=scope,fail,set()
    def exponential(self,label,time):
        key=exp_key(self.scope.graph,label,time)
        if key not in self.cache:
            self.scope.call("backend.expm",lambda: "MOCK_EXP_DESCRIPTOR",identity=key)
            self.cache.add(key)
    def perform(self,kind,args,params):
        if kind==self.fail:
            raise Stage1Failure("A4_ORDER_OR_INVERSE_CORRECTNESS_FAILURE" if kind=="a4_reference_check" else "HAMILTONIAN_IDENTITY_FAILURE")
        if kind=="exponential": self.exponential(*params)
        if kind=="a4_reference_check":
            h,n=params
            kernel,processor=published_math_sequences()
            for weight in kernel+processor:
                for block in self.scope.graph.settings.context.block_ids:
                    self.exponential(block,h*weight/(2*n))
            self.scope.call("published.convention",lambda: "MOCK_REFERENCE",identity=(str(h),n))
        if kind=="rte_moments":
            for name in ("rte.config","rte.moments"):
                self.scope.call(name,lambda: "MOCK_MOMENT_LABEL",identity=str(params[0]))
        if kind=="signals":
            for _ in range(4): self.scope.call("signal.expectation",lambda: "MOCK_SIGNAL_LABEL")
        return dict(mock_kind=kind)


def mock_save(inputs,output):
    # Explicit non-numeric placeholder, never an array archive/science artifact.
    (Path(output)/"inputs.npz").write_bytes(b"MOCK_ONLY_NO_ARRAY_DATA\n")


class LauncherTests(unittest.TestCase):
    def setUp(self):
        guard=Barrier();sys.meta_path.insert(0,guard)
        self.addCleanup(sys.meta_path.remove,guard)

    def test_registry_covers_external_and_nested_science_without_claiming_total(self):
        g=candidate_graph("MOCK")
        scope=registry(g)
        self.assertEqual(scope["dispatch_total"],941)
        self.assertEqual(scope["boundaries"]["backend.expm"],448)
        self.assertEqual(scope["boundaries"]["signal.expectation"],144)
        self.assertIn("Gaussian",scope["library_expansion"]["preparation"])
        self.assertEqual(scope["runtime_domain"]["retained_components"],[1,3])
        self.assertNotIn("total_scientific_calls",scope)
        validate_scope(json.loads(json.dumps(scope)),g)

    def test_scope_layout_and_grid_mutations_rejected(self):
        g=candidate_graph("MOCK")
        for field in ("boundaries","exponential_keys","runtime_domain","outputs"):
            bad=registry(g);bad[field]={}
            with self.subTest(field=field),self.assertRaises(Stage1Failure): validate_registry(bad,g)
        with self.assertRaises(PermissionError): registry(replace(g,settings=replace(g.settings,q_values=(2,))))

    def test_unregistered_duplicate_and_excess_calls_never_invoke_function(self):
        g=candidate_graph("MOCK");ledger=ScopeLedger(registry(g),g)
        ledger.call("input.fixture",lambda: None)
        for name,kwargs in (("input.fixture",{}),("unknown",{}),
                            ("input.block",dict(identity=3)),
                            ("backend.expm",dict(identity=("unknown","9/100")))):
            with self.subTest(name=name),self.assertRaises(Stage1Failure):
                ledger.call(name,lambda:self.fail("Forbidden work reached"),**kwargs)

    def test_runtime_input_expansion_legal_domain_and_bad_sector_rejected(self):
        g=candidate_graph("MOCK")
        for change in (lambda r:r.update(selected=[0,0]),lambda r:r.update(tail_bases=[]),
                       lambda r:r["tail_components"][0].update(support=[])):
            ledger=ScopeLedger(registry(g),g);receipt=preparation_receipt();change(receipt)
            with self.assertRaises(Stage1Failure): ledger.record_resolution("preparation",receipt)
            self.assertIn("preparation",ledger.snapshot()["resolution_candidates"])
        with self.assertRaises(Stage1Failure):
            ScopeLedger(registry(g),g).record_resolution("sector",dict(N=2))

    def test_launch_has_only_scientific_arguments_and_rejects_wrong_source(self):
        from inspect import signature
        self.assertEqual(tuple(signature(launch).parameters),("source_root","output","source_commit"))
        with patch("trottertracks.b2d_evaluator.launcher._run") as run:
            result=launch(ROOT,"not created","0"*40)
        run.assert_not_called()
        self.assertEqual(result["status"],"REJECTED_BEFORE_SCIENCE")

    def test_exclusive_fresh_output_rejects_a_racing_creation_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/"MOCK_OUTPUT";output.mkdir()
            (output/"STOP.json").write_text("MOCK_EXISTING_STOP")
            context=MockRun(output)
            with patch("trottertracks.b2d_evaluator.launcher.verify_execution",return_value=context),patch(
                    "trottertracks.b2d_evaluator.launcher._run") as run:
                result=launch(ROOT,output,"0"*40)
            run.assert_not_called()
            self.assertEqual(result["status"],"FRESH_OUTPUT_FAILED_MANDATORY_STOP")
            self.assertEqual((output/"STOP.json").read_text(),"MOCK_EXISTING_STOP")

    def test_mock_full_launcher_records_scope_and_36_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            run=MockRun(directory)
            status=_run(run,mock_only=True,input_loader=mock_loader,
                backend_factory=lambda s:MockBackend(run.scope),input_saver=mock_save)
            self.assertEqual(status["status"],"COMPLETED_DESCRIPTIVE_MANDATORY_STOP")
            self.assertEqual(status["mode"],"MOCK_ONLY")
            result=json.loads((Path(directory)/"result.json").read_text())
            self.assertEqual(len(result["rows"]),36)
            self.assertEqual(status["calls"]["completed"]["backend.expm"],448)
            self.assertEqual(len(json.loads((Path(directory)/"partial_results.json").read_text())["rows"]),36)
            self.assertEqual(json.loads((Path(directory)/"STOP.json").read_text())["stop"],STOP)

    def test_a4_failure_preserves_evidence_and_forbids_all_comparisons(self):
        with tempfile.TemporaryDirectory() as directory:
            run=MockRun(directory)
            status=_run(run,mock_only=True,input_loader=mock_loader,
                backend_factory=lambda s:MockBackend(run.scope,fail="a4_reference_check"),input_saver=mock_save)
            self.assertEqual(status["reason"],"A4_ORDER_OR_INVERSE_CORRECTNESS_FAILURE")
            self.assertNotIn("dag.operator_error",status["calls"]["attempted"])
            self.assertNotIn("dag.signals",status["calls"]["attempted"])
            self.assertTrue((Path(directory)/"inputs.json").is_file())
            self.assertTrue((Path(directory)/"STOP.json").is_file())
            self.assertEqual(json.loads((Path(directory)/"partial_results.json").read_text())["rows"],[])

    def test_input_correctness_failure_and_resolution_mismatch_stop(self):
        for wrong in (False,True):
            with self.subTest(wrong=wrong),tempfile.TemporaryDirectory() as directory:
                run=MockRun(directory)
                def loader(root,**kwargs):
                    c,i,p=mock_loader(root,**kwargs)
                    return (replace(c,tail_id="WRONG"),i,p) if wrong else (c,i,p)
                status=_run(run,mock_only=True,input_loader=loader,
                    backend_factory=lambda s:MockBackend(run.scope,fail="validate_input"),input_saver=mock_save)
                self.assertIn(status["reason"],("INPUT_IDENTITY_ORDER_MISMATCH","HAMILTONIAN_IDENTITY_FAILURE"))
                self.assertNotIn("dag.operator_error",status["calls"]["attempted"])

    def test_mid_comparison_exception_retains_partial_rows_and_no_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            run=MockRun(directory)
            class FailLater(MockBackend):
                def perform(self,kind,args,params):
                    if kind=="signals" and self.scope.attempted["dag.signals"]==3:
                        raise RuntimeError("SECRET_MUST_NOT_BE_SAVED")
                    return super().perform(kind,args,params)
            status=_run(run,mock_only=True,input_loader=mock_loader,
                backend_factory=lambda s:FailLater(run.scope),input_saver=mock_save)
            self.assertEqual(status["status"],"FAILED_MANDATORY_STOP")
            rows=json.loads((Path(directory)/"partial_results.json").read_text())["rows"]
            self.assertEqual(len(rows),2)
            for path in Path(directory).glob("*.json*"):
                self.assertNotIn("SECRET_MUST_NOT_BE_SAVED",path.read_text())
            with self.assertRaises(FileExistsError):
                _run(MockRun(directory),mock_only=True,input_loader=mock_loader,
                    backend_factory=lambda s:None,input_saver=mock_save)

    def test_inconclusive_precision_failure_never_extends_conditions(self):
        with tempfile.TemporaryDirectory() as directory:
            run=MockRun(directory)
            class Inconclusive(MockBackend):
                def perform(self,*args): raise Stage1Failure("INCONCLUSIVE_NUMERICAL_PRECISION")
            status=_run(run,mock_only=True,input_loader=mock_loader,
                backend_factory=lambda s:Inconclusive(run.scope),input_saver=mock_save)
            self.assertEqual(status["status"],"INCONCLUSIVE_MANDATORY_STOP")
            self.assertEqual(sum(status["calls"]["attempted"].get("dag."+k,0) for k in ("operator_error","signals")),0)

    def test_missing_mock_dependencies_reject_before_any_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            status=_run(MockRun(directory),mock_only=True)
            self.assertEqual(status["reason"],"MOCK_DEPENDENCIES_REQUIRED")

    def test_rejection_and_ledger_use_non_sensitive_failure_codes(self):
        from trottertracks.b2d_evaluator.launcher import rejection
        self.assertNotIn("SECRET",json.dumps(rejection(RuntimeError("SECRET"))))
        g=candidate_graph("MOCK");ledger=ScopeLedger(registry(g),g)
        with self.assertRaises(Stage1Failure): ledger.finish()
        node=next((i,n) for i,n in enumerate(g.nodes) if n.kind=="signals")
        with self.assertRaises(Stage1Failure):
            ledger.dispatch(*node,lambda:self.fail("Comparison before correctness"))

    def test_science_imports_remain_absent(self):
        self.assertFalse(FORBIDDEN.intersection(n.split(".")[0] for n in sys.modules))

    def test_save_failure_still_attempts_stop_and_does_not_report_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            g=candidate_graph("MOCK");ledger=ScopeLedger(registry(g),g)
            evidence=Evidence(directory,mock_only=True)
            original=evidence.write
            def failing(name,value):
                if name=="partial_results.json": raise OSError("SECRET_IO_DETAIL")
                return original(name,value)
            evidence.write=failing
            status=evidence.finish(dict(status="COMPLETED_DESCRIPTIVE_MANDATORY_STOP"),ledger)
            self.assertEqual(status["status"],"EVIDENCE_SAVE_FAILED_MANDATORY_STOP")
            self.assertTrue((Path(directory)/"STOP.json").is_file())
            self.assertNotIn("SECRET",json.dumps(status))


if __name__ == "__main__": unittest.main()
