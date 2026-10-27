"""P1 API/wiring tests; no molecular, matrix, RNG or Qiskit execution."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import ast
import inspect
import json
import os
import subprocess
import sys

import pytest

from trottertracks.resource_applicability.pipeline import (
    CompilerProfile, EvidenceLocator, PipelineError, ScientificExecutionDisabled,
    STAGES, StageResult, compose,
    write_new_result_directory,
)
from trottertracks.resource_applicability.h4_geometry_pipeline import H4GeometryAdapter
from trottertracks.resource_applicability.pipeline_fixture import synthetic_context, synthetic_fixture
from trottertracks.resource_applicability import df_core_stages as stages

ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_h4_adapter_to_result_uses_existing_api_stage_wrappers():
    c = synthetic_context()
    payload, providers = synthetic_fixture(c)
    out = compose(c, H4GeometryAdapter(), mode="synthetic", payload=payload, providers=providers)
    assert out.qualification == "SYNTHETIC_END_TO_END_PASS"
    assert tuple(r.stage for r in out.stages) == STAGES
    assert not out.science_executed and not out.numerical_equivalence_verified
    core = providers.core
    converted = core.hamiltonian.call_args.kwargs
    assert converted["one_body"] is payload.arrays["one"]
    assert converted["constant"] is payload.arrays["nuclear"]
    assert converted["lambdas"] is payload.arrays["lambdas"]
    assert core.prepare_s2.call_args.args[1] is payload.fixed_partition
    assert core.prepare_s2.call_args.kwargs["partition_policy"] == "explicit_ordered_partition"
    assert core.configure_rte.call_args.kwargs == dict(
        evolution_time=c.evolution_time, rte_steps=c.rte_steps,
        truncation_tolerance=1e-10, finite_taylor_order=c.finite_taylor_order, seed=c.seed)
    assert core.event_sequence.call_args.kwargs["events"] == ("ARTIFICIAL_EVENT",)
    assert core.step_request.call_args.kwargs["seed"] == c.seed
    assert core.s2_builder.return_value.build_step.call_count == 1
    assert core.compiler_settings.call_args.kwargs["qiskit_version"] == "MOCK_NOT_QISKIT"
    assert core.signal_shots.call_count == 2
    core.rpe_accounting.assert_not_called()
    last = out.stages[-1]
    assert last.values["rz_count"] is last.values["N"] is last.values["G"] is None
    assert last.values["scientific_qualification"] == "NOT_ACQUIRED"
    encoded = json.dumps(out.to_dict(), allow_nan=False)
    assert "one_body" not in encoded  # Opaque arrays never serialized.
    assert converted["one_body"].shape == (8, 8)
    assert converted["lambdas"].shape == (12,)
    assert not converted["one_body"].flags.writeable
    assert core.signal_shots.call_args.args == (1.0, 0.0, 0.01)


def test_plan_invokes_no_input_or_provider():
    c = synthetic_context()
    adapter = H4GeometryAdapter()
    adapter.input = Mock(side_effect=AssertionError("input must remain unopened"))
    result = compose(c, adapter)
    assert result.qualification == "PLAN_ONLY"
    assert all(s.status == "NOT_EXECUTED" for s in result.stages)
    adapter.input.assert_not_called()


@pytest.mark.parametrize("mode", ["scientific", "real", "resume", "execute"])
def test_real_mode_rejected_before_adapter_input_and_imports(mode):
    adapter = Mock()
    with pytest.raises(ScientificExecutionDisabled):
        compose(synthetic_context(), adapter, mode=mode)
    assert adapter.mock_calls == []


def test_synthetic_mode_rejects_real_input_and_scientific_providers():
    c = synthetic_context()
    data, p = synthetic_fixture(c)
    for ctx, providers in [(replace(c, input_kind="frozen_scientific"), p),
                           (c, replace(p, kind="scientific")), (c, None)]:
        with pytest.raises(ScientificExecutionDisabled):
            compose(ctx, H4GeometryAdapter(), mode="synthetic", payload=data, providers=providers)
    p.core.hamiltonian.assert_not_called()
    with pytest.raises(ScientificExecutionDisabled):
        stages.scientific_providers()


@pytest.mark.parametrize("changes", [
    {"source": None}, {"compiler": None}, {"representation": "Pauli"},
    {"metric": "T_count"}, {"estimand": "phase_rpe"}, {"cost_scope": "time_evolution_only"},
    {"rank_policy": "automatic_rank_selection"}, {"source_stop_status": "CLEARED"},
    {"q": 2}, {"rte_law": "different_law"}, {"formula": "4th"},
    {"normalization_policy": "apply_twice"},
    {"stage_order": ("finite_rte", "deterministic_half", "reverse_deterministic_half")},
    {"geometry": "linear-H6:0.90Angstrom"},
    {"evolution_time": True}, {"ld": True}, {"deterministic_indices": (False, 1, 2)},
])
def test_h4_rejects_incompatible_or_missing_context(changes):
    with pytest.raises(PipelineError):
        c = replace(synthetic_context(), **changes)
        compose(c, H4GeometryAdapter())


def test_no_default_compiler_and_fixed_evidence_path():
    with pytest.raises(TypeError):
        CompilerProfile(basis_gates=("rz",))
    for path in ["/private/input", "../input", "C:/input"]:
        with pytest.raises(PipelineError):
            EvidenceLocator("a" * 40, path, "SOURCE")
    with pytest.raises(PipelineError):
        EvidenceLocator("short", "input", "SOURCE")


@pytest.mark.parametrize("changes", [{"cohort_id": "OTHER"}, {"state_id": "OTHER"},
                                       {"precision": "OTHER"}, {"input_id": "OTHER"},
                                       {"result_version": "OTHER"}])
def test_cross_cohort_or_scope_output_rejected(changes):
    c = synthetic_context()
    data, p = synthetic_fixture(c)
    other = replace(c, **changes)
    bad = lambda ctx, prev: StageResult("pf_pr", other.binding, "SYNTHETIC")
    result = compose(c, H4GeometryAdapter(), mode="synthetic", payload=data,
                     providers=replace(p, prepare=bad))
    assert result.qualification == "FAILED"
    assert "different SOURCE/input/cohort" in result.stages[-1].detail
    p.core.configure_rte.assert_not_called()


@pytest.mark.parametrize("status", ["STOP", "FAILED", "UNQUALIFIED", "MISSING"])
def test_stage_qualification_and_missing_values_preserved_without_retry(status):
    c = synthetic_context()
    data, p = synthetic_fixture(c)
    failed = Mock(return_value=StageResult("pf_pr", c.binding, status, {"N": None, "G": None}))
    result = compose(c, H4GeometryAdapter(), mode="synthetic", payload=data,
                     providers=replace(p, prepare=failed))
    assert result.qualification == status and len(result.stages) == 2
    assert result.stages[-1].values == {"N": None, "G": None}
    assert failed.call_count == 1
    p.core.configure_rte.assert_not_called()


def test_stage_cannot_change_metric_or_claim_real_scientific_pass():
    c = synthetic_context()
    for status, values in [("SYNTHETIC", {"metric": "cx_count"}), ("PASS", {})]:
        data, p = synthetic_fixture(c)
        result = compose(c, H4GeometryAdapter(), mode="synthetic", payload=data,
                         providers=replace(p, prepare=lambda *a: StageResult(
                             "pf_pr", c.binding, status, values)))
        assert result.qualification == "FAILED"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True, "unknown"])
def test_missing_cost_is_none_not_invalid_or_nonfinite_value(value):
    c = synthetic_context()
    data, p = synthetic_fixture(c)
    result = compose(c, H4GeometryAdapter(), mode="synthetic", payload=data,
                     providers=replace(p, prepare=lambda *a: StageResult(
                         "pf_pr", c.binding, "SYNTHETIC", {"G": value})))
    assert result.qualification == "FAILED"
    p.core.configure_rte.assert_not_called()


def test_input_identity_rank_partition_and_missing_arrays_rejected():
    c = synthetic_context()
    data, p = synthetic_fixture(c)
    bad = [replace(data, input_id="OTHER"), replace(data, state_id="OTHER"),
           replace(data, arrays={}), replace(data, binding=replace(c, cohort_id="other").binding),
           replace(data, fixed_partition=SimpleNamespace(ld=4,
               deterministic_block_indices=c.deterministic_indices,
               randomized_block_indices=c.randomized_indices))]
    for payload in bad:
        result = compose(c, H4GeometryAdapter(), mode="synthetic", payload=payload, providers=p)
        assert result.qualification == "FAILED" and len(result.stages) == 1
    p.core.hamiltonian.assert_not_called()


def test_no_sampling_fallback_or_shot_imputation():
    c = synthetic_context()
    data, p = synthetic_fixture(c)
    core = p.core
    with pytest.raises(PipelineError):
        stages.event_request(core, c, None, None, None, events=None)
    result = stages.account_fixed_signal(core, c, normalization=None,
                                         bias_by_axis={}, epsilon=None)
    assert result == {"N_cosine": None, "N_sine": None, "G": None, "status": "MISSING"}
    core.signal_shots.assert_not_called()
    with pytest.raises(PipelineError):
        stages.account_phase_rpe(core, c)
    core.rpe_accounting.assert_not_called()


def _signature_from_source(module, name):
    """Inspect real source signatures without importing numerical/optional modules."""
    tree = ast.parse((ROOT / "src" / "trotterlib" / (module + ".py")).read_text())
    fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == name)
    args = fn.args
    positional = args.posonlyargs + args.args
    params = [inspect.Parameter(a.arg, inspect.Parameter.POSITIONAL_OR_KEYWORD,
                               default=None if i >= len(positional)-len(args.defaults) else inspect.Parameter.empty)
              for i, a in enumerate(positional)]
    params += [inspect.Parameter(a.arg, inspect.Parameter.KEYWORD_ONLY,
                                 default=None if d is not None else inspect.Parameter.empty)
               for a, d in zip(args.kwonlyargs, args.kw_defaults)]
    return inspect.Signature(params)


def test_real_core_signature_compatibility_of_injected_calls():
    c = synthetic_context()
    data, p = synthetic_fixture(c)
    compose(c, H4GeometryAdapter(), mode="synthetic", payload=data, providers=p)
    for module, name, mock in [
        ("df_partial_s2", "prepare_df_partial_s2", p.core.prepare_s2),
        ("rte", "make_rte_config", p.core.configure_rte),
        ("rte_compiled_cost", "transpile_and_measure_cost", p.core.compile_cost),
    ]:
        call = mock.call_args
        _signature_from_source(module, name).bind(*call.args, **call.kwargs)
    # Builder method and dataclass fields are the real existing names.
    s = ast.parse((ROOT / "src/trotterlib/df_partial_s2.py").read_text())
    cls = next(x for x in s.body if isinstance(x, ast.ClassDef) and x.name == "QiskitDFPartialS2CircuitBuilder")
    assert any(isinstance(x, ast.FunctionDef) and x.name == "build_step" for x in cls.body)
    request = next(x for x in s.body if isinstance(x, ast.ClassDef) and x.name == "DFPartialS2StepRequest")
    fields = {x.target.id for x in request.body if isinstance(x, ast.AnnAssign)}
    assert set(p.core.step_request.call_args.kwargs) <= fields


def test_optional_dependency_failure_does_not_start_a_stage(monkeypatch):
    load = Mock(side_effect=ModuleNotFoundError("not installed", name="optional_mock"))
    monkeypatch.setattr(stages, "import_module", load)
    with pytest.raises(PipelineError, match="optional core dependency"):
        stages.bind_existing_core()
    assert load.call_count == 1


def test_cli_plan_synthetic_and_scientific_rejection(tmp_path):
    script = ROOT / "scripts/run_df_research_pipeline.py"
    for args, status in [([], "PLAN_ONLY"), (["--mode", "synthetic"], "SYNTHETIC_END_TO_END_PASS")]:
        r = subprocess.run([sys.executable, "-B", str(script), *args], cwd=tmp_path,
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        result = json.loads(r.stdout)
        assert result["qualification"] == status and not result["science_executed"]
    r = subprocess.run([sys.executable, "-B", str(script), "--mode", "scientific"],
                       cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 2 and "disabled" in r.stderr and not r.stdout
    assert list(tmp_path.iterdir()) == []


def test_imports_without_site_packages_or_optional_science_dependencies(tmp_path):
    code = """
import sys
sys.path.insert(0, sys.argv[1])
import trotterlib
import trottertracks.resource_applicability.pipeline
import trottertracks.resource_applicability.h4_geometry_pipeline
import trottertracks.resource_applicability.df_core_stages
assert not any(x in sys.modules for x in ('numpy', 'qiskit', 'scipy', 'openfermion', 'pyscf'))
assert len(trotterlib.__all__) == 413
assert set(trotterlib.__all__) == set(trotterlib._EXPORTS)
try:
    trotterlib.nonexistent_p1_attribute
except AttributeError:
    pass
else:
    raise AssertionError('unexpected export')
"""
    result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", code, str(ROOT / "src")],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert not list(tmp_path.iterdir())


def test_existing_public_export_resolves_same_attribute():
    import trotterlib
    from trotterlib.rte import CompilerSettings
    assert trotterlib.CompilerSettings is CompilerSettings
    assert "CompilerSettings" in dir(trotterlib)


def test_existing_executor_atomic_publication_preserves_old_output(tmp_path):
    result = compose(synthetic_context(), H4GeometryAdapter())
    output = tmp_path / "new-pipeline-output"
    target = write_new_result_directory(result, output)
    assert json.loads(target.read_text())["qualification"] == "PLAN_ONLY"
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        write_new_result_directory(result, output)
    assert target.read_bytes() == before
    assert list(output.iterdir()) == [target]


def test_cli_refuses_science_before_creating_output(tmp_path):
    output = tmp_path / "no-science-output"
    r = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/run_df_research_pipeline.py"),
                        "--mode", "scientific", "--output-dir", str(output)],
                       cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 2 and not output.exists()
