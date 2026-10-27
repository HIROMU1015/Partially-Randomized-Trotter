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
    {"cohort_id": ""}, {"input_id": ""},
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
                                       {"result_version": "OTHER"},
                                       {"source": EvidenceLocator("b" * 40, "other/source.json", "SOURCE")},
                                       {"compiler": replace(synthetic_context().compiler, transpiler_seed=18)},
                                       {"metric": "cx_count"}])
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
    bad = [None, replace(data, input_id="OTHER"), replace(data, state_id="OTHER"),
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
        stages.bind_existing_core(signal_shots=Mock(), signal_wrapper=Mock())
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
import trottertracks.resource_applicability.pipeline
import trottertracks.resource_applicability.h4_geometry_pipeline
import trottertracks.resource_applicability.df_core_stages
assert not any(x in sys.modules for x in ('numpy', 'qiskit', 'scipy', 'openfermion', 'pyscf'))
assert 'trotterlib' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", code, str(ROOT / "src")],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert not list(tmp_path.iterdir())


def test_existing_eager_initializer_is_byte_unchanged():
    baseline = subprocess.run(["git", "show",
        "f29eafbf4412193e8e8660d9f051889c2cba684b:src/trotterlib/__init__.py"],
        cwd=ROOT, capture_output=True, check=True,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0", "GIT_NO_LAZY_FETCH": "1"})
    assert (ROOT / "src/trotterlib/__init__.py").read_bytes() == baseline.stdout
    tree = ast.parse(baseline.stdout)
    names = next(ast.literal_eval(n.value) for n in tree.body
                 if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name)
                 and t.id == "__all__" for t in n.targets))
    assert len(names) == len(set(names)) == 413
    assert not any(isinstance(n, ast.FunctionDef) and n.name == "__getattr__" for n in tree.body)


def test_existing_executor_atomic_publication_preserves_old_output(tmp_path):
    result = compose(synthetic_context(), H4GeometryAdapter())
    output = tmp_path / "new-pipeline-output"
    try:
        target = write_new_result_directory(result, output)
    except PipelineError as exc:
        if isinstance(exc.__cause__, ModuleNotFoundError):
            assert not output.exists()
            pytest.skip(f"legacy atomic writer dependency unavailable: {exc.__cause__.name}")
        raise
    assert json.loads(target.read_text())["qualification"] == "PLAN_ONLY"
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        write_new_result_directory(result, output)
    assert target.read_bytes() == before
    assert list(output.iterdir()) == [target]


@pytest.mark.parametrize("mode", ["scientific", "real", "resume"])
def test_cli_refuses_science_before_creating_output(tmp_path, mode):
    output = tmp_path / "no-science-output"
    r = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/run_df_research_pipeline.py"),
                        "--mode", mode, "--output-dir", str(output)],
                       cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 2 and not output.exists()


def _fake_core_modules():
    """Interface-only mocks, not an alternate loader for real scientific code."""
    return {
        "trotterlib.df_hamiltonian": SimpleNamespace(DFHamiltonian=Mock()),
        "trotterlib.df_partial_s2": SimpleNamespace(prepare_df_partial_s2=Mock(),
            DFPartialS2StepRequest=Mock(), QiskitDFPartialS2CircuitBuilder=Mock()),
        "trotterlib.rte": SimpleNamespace(make_rte_config=Mock(), CompilerSettings=Mock()),
        "trotterlib.df_rte_circuit": SimpleNamespace(DFRTEEventSequenceCircuitRequest=Mock()),
        "trotterlib.rte_compiled_cost": SimpleNamespace(transpile_and_measure_cost=Mock()),
        "trotterlib.rpe_resource_accounting": SimpleNamespace(evaluate_rpe_round_candidate=Mock()),
    }


def test_h4_adapter_selects_original_callbacks_and_fresh_explicit_policy(monkeypatch):
    from trottertracks.resource_applicability.h4_geometry_pipeline import shots, wrapper
    modules = _fake_core_modules()
    load = Mock(side_effect=modules.__getitem__)
    monkeypatch.setattr(stages, "import_module", load)
    adapter = H4GeometryAdapter()
    core = adapter.bind_core(synthetic_context())
    assert core.signal_shots is shots and core.signal_wrapper is wrapper
    assert core.rpe_accounting is None and core.estimand == "fixed_complex_signal"
    assert [c.args[0] for c in load.call_args_list] == list(modules)[:-1]
    expected = {"identity_policy": "extract_identity_phase", "coefficient_atol": 0.0,
        "diagonal_sort": "descending_abs", "basis_construction_policy": "fermionic_gaussian_jw",
        "partition_policy": "explicit_ordered_partition"}
    assert adapter.preparation_policy == expected
    modified = adapter.preparation_policy
    modified["identity_policy"] = "CHANGED"
    assert adapter.preparation_policy == expected
    # Test propagation with a mock; no real preparation or selected H4 callable runs.
    core.prepare_s2.return_value = object()
    value = stages.prepare(core, "ARTIFICIAL_H", "ARTIFICIAL_PARTITION", policy=adapter.preparation_policy)
    core.prepare_s2.assert_called_once_with("ARTIFICIAL_H", "ARTIFICIAL_PARTITION", **expected)
    assert value is core.prepare_s2.return_value


@pytest.mark.parametrize("missing", ["signal_shots", "signal_wrapper"])
def test_missing_study_callback_rejected_before_core_import(monkeypatch, missing):
    load = Mock(side_effect=AssertionError("must not import"))
    monkeypatch.setattr(stages, "import_module", load)
    hooks = {"signal_shots": Mock(), "signal_wrapper": Mock()}
    hooks[missing] = None
    with pytest.raises(PipelineError, match=missing):
        stages.bind_existing_core(**hooks)
    load.assert_not_called()


def test_phase_callback_resolved_only_for_explicit_estimand(monkeypatch):
    modules = _fake_core_modules()
    load = Mock(side_effect=modules.__getitem__)
    monkeypatch.setattr(stages, "import_module", load)
    core = stages.bind_existing_core(signal_wrapper=Mock(), estimand="phase_rpe")
    assert core.signal_shots is None
    assert load.call_args_list[-1].args == ("trotterlib.rpe_resource_accounting",)
    context = replace(synthetic_context(), estimand="phase_rpe")
    token = object()
    stages.account_phase_rpe(core, context, token, scope="ARTIFICIAL")
    core.rpe_accounting.assert_called_once_with(token, scope="ARTIFICIAL")
    with pytest.raises(PipelineError, match="mixed"):
        stages.account_fixed_signal(core, synthetic_context(), normalization=None, bias_by_axis={}, epsilon=None)
    with pytest.raises(PipelineError, match="disagree"):
        stages.compile_axis(core, synthetic_context(), None, axis="cosine")
    core.s2_builder.assert_not_called()


def test_fixed_binding_cannot_silently_become_phase_accounting():
    _, providers = synthetic_fixture()
    core = replace(providers.core, rpe_accounting=None)
    context = replace(synthetic_context(), estimand="phase_rpe")
    with pytest.raises(PipelineError):
        stages.account_phase_rpe(core, context)
    core.signal_shots.assert_not_called()


@pytest.mark.parametrize("missing", [None, "identity_policy", "coefficient_atol", "diagonal_sort",
                                    "basis_construction_policy", "partition_policy"])
def test_missing_preparation_policy_rejected_before_core_call(missing):
    core = SimpleNamespace(prepare_s2=Mock())
    policy = H4GeometryAdapter().preparation_policy
    if missing is None:
        policy = None
    else:
        policy.pop(missing)
    with pytest.raises(PipelineError, match="policy"):
        stages.prepare(core, None, None, policy=policy)
    core.prepare_s2.assert_not_called()


@pytest.mark.parametrize("bad", [{"identity_policy": ""}, {"coefficient_atol": True},
                                 {"coefficient_atol": float("nan")}, {"unexpected": "policy"}])
def test_invalid_preparation_policy_is_not_defaulted(bad):
    core = SimpleNamespace(prepare_s2=Mock())
    policy = {**H4GeometryAdapter().preparation_policy, **bad}
    with pytest.raises(PipelineError, match="policy"):
        stages.prepare(core, None, None, policy=policy)
    core.prepare_s2.assert_not_called()


@pytest.mark.parametrize("name", ["prepare", "finite_rte_event", "accounting", "compiled_cost"])
def test_missing_stage_provider_rejected_before_input(name):
    c = synthetic_context()
    data, providers = synthetic_fixture(c)
    with pytest.raises(PipelineError, match=name):
        compose(c, H4GeometryAdapter(), mode="synthetic", payload=data,
                providers=replace(providers, **{name: None}))
    providers.core.hamiltonian.assert_not_called()


def test_output_dependency_error_preserves_cause_and_creates_no_directory(monkeypatch, tmp_path):
    import builtins
    original = builtins.__import__
    missing = ModuleNotFoundError("artificial missing dependency", name="ARTIFICIAL_OPTIONAL")
    def reject_writer(name, *args, **kwargs):
        if name == "trotterlib.parallel_validation_executor":
            raise missing
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", reject_writer)
    output = tmp_path / "not_created"
    with pytest.raises(PipelineError, match="ARTIFICIAL_OPTIONAL") as error:
        write_new_result_directory(compose(synthetic_context(), H4GeometryAdapter()), output)
    assert error.value.__cause__ is missing
    assert not output.exists()


def test_new_directory_and_collision_boundary_with_artificial_writer(monkeypatch, tmp_path):
    import builtins
    original = builtins.__import__
    writer = Mock(side_effect=lambda path, payload: path.write_text(json.dumps(payload)))
    def artificial_writer_import(name, *args, **kwargs):
        if name == "trotterlib.parallel_validation_executor":
            return SimpleNamespace(atomic_write_json=writer)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", artificial_writer_import)
    result = compose(synthetic_context(), H4GeometryAdapter())
    output = tmp_path / "artificial-output"
    path = write_new_result_directory(result, output)
    before = path.read_bytes()
    assert json.loads(before)["qualification"] == "PLAN_ONLY"
    with pytest.raises(FileExistsError):
        write_new_result_directory(result, output)
    assert writer.call_count == 1 and path.read_bytes() == before
    assert list(output.iterdir()) == [path]


def test_cli_missing_legacy_output_dependencies_are_explicit(tmp_path):
    output = tmp_path / "not_created"
    # No site packages guarantees an unavailable legacy dependency, without fakes.
    r = subprocess.run([sys.executable, "-I", "-S", "-B", str(ROOT / "scripts/run_df_research_pipeline.py"),
                        "--output-dir", str(output)], cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 2 and "legacy core dependencies" in r.stderr
    assert "no output directory was created" in r.stderr and "Traceback" not in r.stderr
    assert not output.exists() and not r.stdout


def test_cli_synthetic_missing_numpy_is_explicit_and_creates_nothing(tmp_path):
    output = tmp_path / "not_created"
    r = subprocess.run([sys.executable, "-I", "-S", "-B", str(ROOT / "scripts/run_df_research_pipeline.py"),
                        "--mode", "synthetic", "--output-dir", str(output)],
                       cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 2 and "fixture dependency unavailable: numpy" in r.stderr
    assert not output.exists() and "Traceback" not in r.stderr
