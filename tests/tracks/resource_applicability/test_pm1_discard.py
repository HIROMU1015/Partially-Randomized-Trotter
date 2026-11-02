"""PM-1 pure synthetic/mock tests. Never load/stat/hash a molecular NPZ."""
import ast
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys

import pytest

from trottertracks.resource_applicability import pm1_discard_contract as c
from trottertracks.resource_applicability import pm1_discard_execution as e


def fixture_plan(commit=None):
    refs = [{"candidate_id": name, "work": {m: 100.0 for m in c.METRICS}, "total_shots": 100}
            for name in c.COMPARATORS]
    return c.make_plan(refs, {"real": 1.0, "imag": 0.0}, {"synthetic.py": "a" * 64}, {}, source_commit=commit)


def axes_for(candidate):
    return {axis: {"axis": axis, "status": "complete", "evaluation_method": "exact",
                   "state_preparation_included": False, "measurement_included": True,
                   "additional_control_applied": False, "transpile_configuration": copy.deepcopy(c.COMPILER),
                   "quantum_shots_executed": 0, "backend_execution_included": False,
                   "q_m": candidate["q"], "repetition_count": candidate["q"], "delta_time": candidate["delta"],
                   "t_m": .8, "r_m": 0, "K_m": 0, **{m: 2.0 for m in c.METRICS}}
            for axis in c.AXES}


class FakeBackend:
    def __init__(self, *, ineligible=False, fail_compile=None):
        self.events, self.ineligible, self.fail_compile = [], ineligible, fail_compile

    def evaluate(self, candidate, target):
        self.events.append(("signal", candidate["candidate_id"]))
        return .5 + 0j if self.ineligible else target

    def compile(self, candidate):
        self.events.append(("compile", candidate["candidate_id"]))
        if sum(event[0] == "compile" for event in self.events) == self.fail_compile:
            raise RuntimeError("synthetic compiler failure")
        return axes_for(candidate)


def test_exact_eight_candidates_and_time_are_frozen():
    rows = c.candidates()
    assert [(r["rank"], r["q"]) for r in rows] == [(rank, q) for rank in (4, 5) for q in (1, 2, 4, 8)]
    assert all(r["T"] == .8 and r["delta"] * r["q"] == .8 and r["r"] == r["K"] == 0 for r in rows)
    assert len({r["candidate_fingerprint"] for r in rows}) == 8


def test_wrapper_identity_axis_separation_source_binding_and_no_seed():
    plan = fixture_plan()
    keys = [c.wrapper_key(plan, row, axis) for row in plan["candidates"] for axis in c.AXES]
    assert len(set(keys)) == 16
    assert c.wrapper_key(fixture_plan("b" * 40), plan["candidates"][0], "cosine") != keys[0]
    with pytest.raises(ValueError):
        c.wrapper_key(plan, plan["candidates"][0], "other")


@pytest.mark.parametrize("path", ["x.npz", "x.npy", "../x.json", "/x.py", ".runtime/x.json", "x.pickle"])
def test_text_boundary_rejects_before_any_file_access(monkeypatch, path):
    def forbid(*args, **kwargs):
        raise AssertionError("unexpected filesystem access")
    monkeypatch.setattr(Path, "read_bytes", forbid)
    monkeypatch.setattr(Path, "stat", forbid)
    monkeypatch.setattr(Path, "resolve", forbid)
    with pytest.raises(ValueError):
        c.read_text_bytes(Path("synthetic"), path)


@pytest.mark.parametrize("field,value", [("resource_caps", {"full_wrappers": 17}), ("execution_authorized", True),
                                       ("next_stage_authorized", True), ("research_decision", "CONTINUE"),
                                       ("compile_ineligible_for_completeness", False), ("comparison_scope", "all_methods")])
def test_resigning_modified_plan_does_not_bypass_contract(field, value):
    plan = fixture_plan()
    plan[field] = value
    plan["plan_fingerprint"] = c.fingerprint({k: v for k, v in plan.items() if k != "plan_fingerprint"})
    with pytest.raises(ValueError, match="frozen contract"):
        c.validate_plan(plan)


def test_boolean_candidate_rejected_even_if_json_resigned():
    plan = fixture_plan()
    plan["candidates"][0]["q"] = True
    plan["plan_fingerprint"] = c.fingerprint({k: v for k, v in plan.items() if k != "plan_fingerprint"})
    with pytest.raises(ValueError):
        c.validate_plan(plan)


def test_unsealed_source_stops_before_import_or_private_access(monkeypatch):
    monkeypatch.setattr(c, "git", lambda *args: pytest.fail("git must not be called before early gate"))
    monkeypatch.setattr(e, "_DevelopmentBackend", lambda *args: pytest.fail("private boundary crossed"))
    with pytest.raises(ValueError, match="source commit not frozen"):
        e.validate_execution_gate(Path("synthetic"), fixture_plan(), {}, b"{}", "auth.json")


def test_old_m2_authorization_cannot_be_reused():
    with pytest.raises(ValueError, match="separate PM1"):
        e.validate_execution_gate(Path("synthetic"), fixture_plan("b" * 40),
                                  {"status": "M2_EXECUTION_AUTHORIZED_ONCE"}, b"{}", "auth.json")


def mocked_authorization_gate(monkeypatch, root):
    plan = fixture_plan("b" * 40)
    data = (c.canonical(plan) + "\n").encode()
    auth = {"schema_version": c.AUTH_VERSION, "status": "PM1_EXECUTION_AUTHORIZED_ONCE",
            "source_commit": plan["source_commit"], "plan_fingerprint": plan["plan_fingerprint"],
            "plan_sha256": c.hashlib.sha256(data).hexdigest(), "resource_caps": copy.deepcopy(c.CAPS),
            "permissions": copy.deepcopy(c.PERMISSIONS), "final_review_approved": True,
            "fixed_project_root": str(root.resolve()),
            "output_relative": "artifacts/resource_applicability/pr2_pm1_discard_execution/synthetic_not_production"}
    committed = c.canonical(auth).encode()
    monkeypatch.setattr(c, "git", lambda root, *args: b"c" * 40 if args[:1] == ("rev-parse",) else committed if args[:1] == ("show",) else b"")
    monkeypatch.setattr(c, "read_text_bytes", lambda *args: committed)
    monkeypatch.setattr(c, "source_inventory", lambda *args: plan["source_sha256"])
    monkeypatch.setattr(c, "load_saved_inputs", lambda *args: ({}, {}))
    monkeypatch.setattr(c, "frozen_references", lambda *args: (plan["saved_comparators"], plan["saved_full_H_target"]))
    monkeypatch.setattr(e, "environment_identity", lambda: c.ENVIRONMENT)
    monkeypatch.setattr(e, "_DevelopmentBackend", lambda *args: pytest.fail("private data boundary crossed"))
    for key, value in {**c.THREAD_ENVIRONMENT, "PYTHONPATH": "src"}.items():
        monkeypatch.setenv(key, value)
    return plan, auth, data


def test_positive_authorization_gate_is_mock_only_and_creates_no_output(monkeypatch, tmp_path):
    plan, auth, data = mocked_authorization_gate(monkeypatch, tmp_path)
    out = e.validate_execution_gate(tmp_path, plan, auth, data, "synthetic_auth.json")
    assert out.as_posix() == auth["output_relative"]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("field,value", [("status", "M2_EXECUTION_AUTHORIZED_ONCE"),
                                        ("source_commit", "d" * 40), ("plan_sha256", "0" * 64),
                                        ("resource_caps", {"full_wrappers": 17}), ("permissions", {}),
                                        ("final_review_approved", False), ("output_relative", "../elsewhere")])
def test_authorization_mutations_stop_before_private_boundary(monkeypatch, tmp_path, field, value):
    plan, auth, data = mocked_authorization_gate(monkeypatch, tmp_path)
    auth[field] = value
    with pytest.raises(ValueError):
        e.validate_execution_gate(tmp_path, plan, auth, data, "synthetic_auth.json")
    assert list(tmp_path.iterdir()) == []


def test_development_constructor_uses_mock_loader_only(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from trotterlib import pr2_matched_accuracy_m1_execution as m1
    selected = []
    ham = SimpleNamespace(n_qubits=8, n_blocks=12,
                          select_blocks=lambda indices: selected.append(indices) or "synthetic_prefix")
    def loader(root, counters):
        counters["development_raw_hash_checks"] += 1
        counters["development_npz_loads"] += 1
        return ham, "synthetic_state", {"synthetic": True}
    monkeypatch.setattr(m1, "_load_development_only", loader)
    monkeypatch.setattr(m1, "_dense_block_operators", lambda ham: ("synthetic_one_body", tuple(range(5)), {}))
    audit = e.new_audit()
    backend = e._DevelopmentBackend(tmp_path, audit)
    assert backend.state == "synthetic_state" and selected == [tuple(range(5))]
    assert audit["development_loads_completed"] == audit["development_load_calls"] == 1


def test_corrected_shot_formula_and_bias_labels():
    row = e.signal_record(c.candidates()[0], .9 + 0j, .9 + 0j)
    expected = math.ceil(2.0 / (.05 / math.sqrt(2.0))**2 * math.log(2.0 / .025))
    assert row["axis_shots"] == {"real": expected, "imag": expected}
    assert row["total_shots"] == 2 * expected
    assert row["pure_discard_bias_abs"] is row["pure_pf_bias_abs"] is None
    assert row["normalization_multiplier"] == 1.0


def test_zero_allowance_is_scientific_ineligible_not_failure():
    row = e.signal_record(c.candidates()[0], 0j, complex(.05 / math.sqrt(2.0), 0))
    assert row["accuracy_eligible"] is False and row["total_shots"] is None
    assert row["axis_shots"]["real"] is None


@pytest.mark.parametrize("mean", [complex(float("nan"), 0), complex(float("inf"), 0), 1.01 + 0j])
def test_signal_numerical_failure(mean):
    with pytest.raises(ValueError):
        e.signal_record(c.candidates()[0], mean, 1 + 0j)


@pytest.mark.parametrize("ineligible", [False, True])
def test_all_eight_signals_before_exact_sixteen_wrappers_and_stop(ineligible):
    plan, backend, audit = fixture_plan(), FakeBackend(ineligible=ineligible), e.new_audit()
    rows = e.evaluate_eight(plan, backend, audit)
    assert [event[0] for event in backend.events] == ["signal"] * 8 + ["compile"] * 8
    assert audit["signal_completed"] == 8 and audit["wrappers_completed"] == 16
    result = e.result_payload(plan, rows, audit)
    assert result["status"] == c.COMPLETE_STATUS and result["next_stage_authorized"] is False
    assert result["research_decision"] is None and result["mandatory_stop_reached"] is True
    assert all(r["work"] is None for r in rows) if ineligible else all(r["work"]["rz_count"] > 0 for r in rows)
    assert len(result["point_comparisons"]) == (0 if ineligible else 40)


def test_signal_and_compile_caps_prevent_extra_calls():
    for key, amount in (("signal_attempts", 8), ("wrapper_reservations", 16)):
        backend, audit = FakeBackend(), e.new_audit()
        audit[key] = amount
        with pytest.raises(ValueError, match="budget"):
            e.evaluate_eight(fixture_plan(), backend, audit)
        assert sum(event[0] == "signal" for event in backend.events) <= 8
        assert sum(event[0] == "compile" for event in backend.events) == 0


def test_failure_keeps_partial_null_ledger_and_does_not_retry():
    plan, backend, audit, rows = fixture_plan(), FakeBackend(fail_compile=3), e.new_audit(), []
    with pytest.raises(RuntimeError, match="synthetic"):
        e.evaluate_eight(plan, backend, audit, rows)
    assert len(rows) == 8 and sum(r["axes"] is not None for r in rows) == 2
    assert audit["wrapper_reservations"] == 6 and audit["wrappers_completed"] == 4
    assert audit["unresolved_wrapper_reservations"] == 2
    result = e.result_payload(plan, rows, audit, "synthetic compiler failure")
    assert result["status"] == c.FAILURE_STATUS and result["actual_wrapper_count_if_failure"] is None
    assert result["research_decision"] is None and result["next_stage_authorized"] is False
    with pytest.raises(ValueError, match="partial ledger"):
        e.evaluate_eight(plan, backend, audit, rows)


@pytest.mark.parametrize("field,value", [("q_m", 16), ("delta_time", .01), ("measurement_included", False),
                                        ("additional_control_applied", True), ("rz_count", -1), ("rz_count", float("nan")),
                                        ("quantum_shots_executed", 1)])
def test_bad_cost_semantics_or_metric_fails(field, value):
    candidate = c.candidates()[0]
    axes = axes_for(candidate)
    axes["cosine"][field] = value
    with pytest.raises(ValueError):
        e._check_axes(axes, candidate)


def test_stdlib_preparation_and_top_level_execution_imports_are_science_free():
    for module in (c, e):
        tree = ast.parse(Path(module.__file__).read_text())
        imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        names = [a.name for node in imports if isinstance(node, ast.Import) for a in node.names]
        names += [node.module or "" for node in imports if isinstance(node, ast.ImportFrom)]
        assert not any(name.startswith(("numpy", "qiskit", "scipy", "cupy", "trotterlib")) for name in names)


def test_private_compile_adapter_with_tiny_synthetic_hamiltonian_only():
    # 2-qubit synthetic diagonal DF model, not either H4 dataset. Two synthetic
    # wrappers are explicitly distinguished from zero production H4 wrappers.
    import numpy as np
    from trotterlib.df_hamiltonian import DFHamiltonian
    from trotterlib.pr2_matched_accuracy_m1_execution import _prepare_discard
    ham = DFHamiltonian(constant=.13, one_body=np.diag([.2, -.1]).astype(complex),
                        lambdas=np.asarray([.2]), g_matrices=(np.diag([.4, 1.0]).astype(complex),),
                        metadata={"name": "pm1_synthetic_not_H4"})
    backend = e._DevelopmentBackend.__new__(e._DevelopmentBackend)
    backend.preparations = {4: (_prepare_discard(ham, 1), [])}
    candidate = c.candidates()[0]
    axes = backend.compile(candidate)
    e._check_axes(axes, candidate)
    assert all(a["enumerated_trajectory_count"] == 1 for a in axes.values())
    assert all(a["quantum_shots_executed"] == 0 for a in axes.values())


@pytest.mark.parametrize("q", [1, 2, 4, 8])
def test_synthetic_discard_action_preserves_existing_signal_path(q):
    import numpy as np
    from trotterlib.df_hamiltonian import DFHamiltonian
    from trotterlib import pr2_matched_accuracy_m1_execution as m1
    ham = DFHamiltonian(constant=.13, one_body=np.asarray([[.2, .04], [.04, -.1]], complex),
                        lambdas=np.asarray([.2, -.1, .15, .03, -.02]),
                        g_matrices=tuple(np.asarray([[.4+i*.02, .03], [.03, 1.0]], complex) for i in range(5)),
                        metadata={"name": "pm1_synthetic_action_not_H4"})
    backend = e._DevelopmentBackend.__new__(e._DevelopmentBackend)
    backend.m1, backend.hamiltonian, backend.state = m1, ham, np.asarray([1, 0, 0, 0], complex)
    backend.one_body, backend.fragments, _ = m1._dense_block_operators(ham)
    backend.block_cache, backend.preparations = {}, {}
    candidate = next(row for row in c.candidates() if row["rank"] == 4 and row["q"] == q)
    mean = backend.evaluate(candidate, 1 + 0j)
    preparation, deterministic = backend.preparations[4]
    legacy = m1._deterministic_signal_record(candidate, preparation, deterministic, backend.state, 1 + 0j)
    assert mean == complex(legacy["corrected_mean"]["real"], legacy["corrected_mean"]["imag"])
    row = e.signal_record(candidate, mean, 1 + 0j)
    assert row["axis_shots"] == legacy["axis_shots"]


def test_preparation_runner_saved_json_only_guard(monkeypatch):
    root = Path(__file__).parents[3]
    path = root / "scripts/resource_applicability/run_pr2_pm1_discard_contract.py"
    spec = importlib.util.spec_from_file_location("pm1_planner_for_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_read, original_stat, original_resolve = Path.read_bytes, Path.stat, Path.resolve

    def guarded(function):
        def wrapped(path, *args, **kwargs):
            assert path.suffix not in {".npz", ".npy", ".pickle", ".pkl"}, "protected scientific data accessed"
            assert ".runtime" not in path.parts, "runtime accessed"
            return function(path, *args, **kwargs)
        return wrapped
    monkeypatch.setattr(Path, "read_bytes", guarded(original_read))
    monkeypatch.setattr(Path, "stat", guarded(original_stat))
    monkeypatch.setattr(Path, "resolve", guarded(original_resolve))
    bundle = module.build_bundle(root)
    plan = bundle["zero_science_plan_v1.json"]
    assert plan["execution_authorized"] is False and plan["source_commit"] is None
    assert len(bundle["wrapper_identity_plan_v1.json"]["wrapper_keys"]) == 16
    assert set(plan["source_sha256"]) >= set(c.NEW_SOURCE_PATHS)
    assert plan["saved_comparators"][0]["work"]["rz_count"] == 130774896.65625
    assert bundle["preparation_access_audit_v1.json"]["prior_attempt_incident"]["npz_stat_hash_accesses"] == 4
