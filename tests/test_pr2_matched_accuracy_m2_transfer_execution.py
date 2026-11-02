from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from trotterlib import pr2_matched_accuracy_m2_transfer_contract as contract
from trotterlib import pr2_matched_accuracy_m2_transfer_execution as execution
from trotterlib.df_hamiltonian import DFHamiltonian

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def forbid_molecular_snapshot_access(monkeypatch):
    def no_npz(*_args, **_kwargs):
        raise AssertionError("implementation tests must not load any NPZ")
    monkeypatch.setattr(np, "load", no_npz)
    for name in ("open", "stat", "resolve"):
        original = getattr(Path, name)
        def guarded(path, *args, _original=original, **kwargs):
            if "held_out" in str(path):
                raise AssertionError("held-out path access in implementation tests")
            return _original(path, *args, **kwargs)
        monkeypatch.setattr(Path, name, guarded)


def contract_plan():
    return contract.load_json(ROOT / execution.CONTRACT_PLAN_PATH)


def plan():
    return execution.build_execution_plan(
        contract_plan(), source_commit="a" * 40,
        source_hashes={p: "b" * 64 for p in execution.PRIMARY_SOURCE_PATHS},
        environment=execution.environment_identity(),
    )


def signal(cell, real=1, imag=1):
    return {
        "candidate_id": cell["candidate_id"],
        "candidate_fingerprint": cell["execution_candidate_fingerprint"],
        "accuracy_eligible": real is not None and imag is not None,
        "axis_shots": {"real": real, "imag": imag},
    }


def fake_compiled(cell, cosine=20, sine=20):
    rows = []
    for index, seed in enumerate(cell["future_trajectory_seeds"] or [None]):
        axes = {}
        for axis, value in (("cosine", cosine), ("sine", sine)):
            cost = value(index) if callable(value) else value
            axes[axis] = {
                "wrapper_key": cell["wrapper_cache_keys"][axis][index],
                "metrics": {m: cost for m in contract.METRICS},
            }
        rows.append({"trajectory_index": index, "trajectory_seed": seed, "axes": axes})
    return {"candidate_id": cell["candidate_id"], "task_fingerprint": cell["task_fingerprint"],
            "paired_trajectory_rows": rows}


def test_zero_compute_plan_matches_frozen_5_configurations_and_196_keys(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("held-out/science access before authorization")
    monkeypatch.setattr(execution, "load_held_out", forbidden)
    monkeypatch.setattr(execution, "evaluate_signals", forbidden)
    monkeypatch.setattr(execution, "compile_cell", forbidden)
    monkeypatch.setattr(np, "load", forbidden)
    value = plan()
    execution.validate_execution_plan(value, contract_plan())
    keys = [k for cell in value["frozen_candidates"] for v in cell["wrapper_cache_keys"].values() for k in v]
    assert len(keys) == len(set(keys)) == 196
    assert value["execution_authorized"] is False
    assert value["held_out_identity"]["file_sha256"] == (
        "ad7e3e7165c55dbaa395eef7a1dd74db89e1f7ab29a69ac64333f4aebf8b3e37"
    )
    assert all(v == 0 for v in value["zero_science_counters"].values())


@pytest.mark.parametrize("field", ["source_commit", "wrapper_key", "seed", "configuration", "budget"])
def test_rehashed_plan_mutations_fail_without_held_out(field):
    value = plan()
    if field == "source_commit":
        value["source_commit"] = "c" * 40
    elif field == "wrapper_key":
        value["frozen_candidates"][0]["wrapper_cache_keys"]["cosine"][0] = "d" * 64
    elif field == "seed":
        value["frozen_candidates"][0]["future_trajectory_seeds"][0] += 1
    elif field == "configuration":
        value["frozen_candidates"][0]["transfer_configuration"]["q"] = 8
    else:
        value["resource_caps"]["total_full_wrappers"] = 200
    value["plan_fingerprint"] = contract.fingerprint(
        {k: v for k, v in value.items() if k != "plan_fingerprint"})
    with pytest.raises(ValueError, match="differs"):
        execution.validate_execution_plan(value, contract_plan())


def test_source_commit_changes_wrapper_identity_but_not_random_draws():
    first = plan()
    second = execution.build_execution_plan(
        contract_plan(), source_commit="c" * 40, source_hashes=first["source_hashes"],
        environment=first["environment_identity"])
    for left, right in zip(first["frozen_candidates"], second["frozen_candidates"], strict=True):
        assert left["future_trajectory_seeds"] == right["future_trajectory_seeds"]
        assert left["wrapper_cache_keys"] != right["wrapper_cache_keys"]


def test_signal_and_cost_cannot_use_different_candidate_fingerprints():
    cell = plan()["frozen_candidates"][0]
    corrupted = signal(cell)
    corrupted["candidate_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="signal/cost candidate"):
        execution.candidate_summary(cell, corrupted, fake_compiled(cell))


@pytest.mark.parametrize("workers", [0, 6, True])
def test_worker_cap_gate_precedes_held_out(workers):
    with pytest.raises(ValueError, match="workers"):
        execution.validate_authorization(
            ROOT, {}, plan(), plan_sha256="d" * 64, authorization_path="auth.json",
            authorization_sha256="e" * 64, output_relative="artifacts/m2-test", workers=workers)


def test_authorization_missing_rejected_before_data_boundary(tmp_path, monkeypatch):
    value = plan()
    def forbidden(*_args, **_kwargs):
        raise AssertionError("held-out access")
    monkeypatch.setattr(execution, "load_held_out", forbidden)
    monkeypatch.setattr(np, "load", forbidden)
    execution._atomic_json(tmp_path / "plan.json", value)
    execution._atomic_json(tmp_path / "auth.json", {})
    execution._atomic_json(tmp_path / execution.CONTRACT_PLAN_PATH, contract_plan())
    # The raw SHA is exactly reproduced by canonical write_json formatting.
    frozen = ROOT / execution.CONTRACT_PLAN_PATH
    (tmp_path / execution.CONTRACT_PLAN_PATH).write_bytes(frozen.read_bytes())
    with pytest.raises(ValueError, match="not authorized"):
        execution.run_transfer(tmp_path, plan_path=tmp_path / "plan.json",
                               authorization_path=tmp_path / "auth.json",
                               output_relative="artifacts/m2-test")
    assert not (tmp_path / "artifacts/m2-test").exists()


def test_plan_argument_cannot_open_held_out_before_gate(tmp_path, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("np.load is not allowed")
    monkeypatch.setattr(np, "load", forbidden)
    with pytest.raises(ValueError, match="public non-held-out"):
        execution.run_transfer(tmp_path, plan_path=tmp_path / contract.HELD_OUT_RELATIVE_LITERAL,
                               authorization_path=tmp_path / "auth.json",
                               output_relative="artifacts/m2-test")


def authorization(value):
    return {
        "schema_version": execution.AUTH_SCHEMA, "status": "M2_EXECUTION_AUTHORIZED_ONCE",
        "source_commit": value["source_commit"], "source_hashes": value["source_hashes"],
        "execution_plan_sha256": "d" * 64, "execution_plan_fingerprint": value["plan_fingerprint"],
        "contract_plan_sha256": execution.CONTRACT_PLAN_SHA256,
        "held_out_identity": execution.HELD_OUT_IDENTITY,
        "permissions": dict(execution.PERMISSIONS), "resource_caps": value["resource_caps"],
        "execution_run_limit": 1, "fixed_output_relative": "artifacts/m2-test",
        "result_terminal_statuses": list(contract.TRANSFER_STATUSES),
    }


@pytest.mark.parametrize("change", ["permissions", "caps", "source", "output", "run_limit"])
def test_authorization_binding_rejects_mutations(change):
    value = plan()
    auth = authorization(value)
    if change == "permissions":
        auth["permissions"]["candidate_search_authorized"] = True
    elif change == "caps":
        auth["resource_caps"] = {**auth["resource_caps"], "additional_trajectories": 96}
    elif change == "source":
        auth["source_commit"] = "c" * 40
    elif change == "output":
        auth["fixed_output_relative"] = "artifacts/another"
    else:
        auth["execution_run_limit"] = 2
    with pytest.raises(ValueError, match="identity/permissions/budget"):
        execution.validate_authorization(
            ROOT, auth, value, plan_sha256="d" * 64, authorization_path="auth.json",
            authorization_sha256="e" * 64, output_relative="artifacts/m2-test", workers=5)


def test_complete_authorization_gate_checks_committed_blobs_and_environment(monkeypatch):
    body = b"synthetic execution source\n"
    digest = hashlib.sha256(body).hexdigest()
    value = execution.build_execution_plan(
        contract_plan(), source_commit="a" * 40,
        source_hashes={p: digest for p in execution.PRIMARY_SOURCE_PATHS},
        environment=execution.environment_identity())
    auth = authorization(value)
    auth_bytes = contract.canonical_json(auth)
    auth_sha = hashlib.sha256(auth_bytes).hexdigest()
    def fake_git(_root, *args):
        if args[0] == "rev-parse":
            return ("f" * 40 + "\n").encode()
        if args[0] == "merge-base":
            return (args[1] + "\n").encode()
        return auth_bytes if args[1] == "HEAD:auth.json" else body
    monkeypatch.setattr(execution, "git", fake_git)
    monkeypatch.setattr(execution, "source_paths", lambda *_args: execution.PRIMARY_SOURCE_PATHS)
    monkeypatch.setattr(contract, "file_sha256", lambda _path: digest)
    for name, setting in execution.ENVIRONMENT.items():
        monkeypatch.setenv(name, setting)
    execution.validate_authorization(
        ROOT, auth, value, plan_sha256="d" * 64, authorization_path="auth.json",
        authorization_sha256=auth_sha, output_relative="artifacts/m2-test", workers=5)
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "2")
    with pytest.raises(ValueError, match="BLAS/process"):
        execution.validate_authorization(
            ROOT, auth, value, plan_sha256="d" * 64, authorization_path="auth.json",
            authorization_sha256=auth_sha, output_relative="artifacts/m2-test", workers=5)


def test_pair_covariance_is_retained_in_primary_standard_error():
    cell = plan()["frozen_candidates"][0]
    cost = fake_compiled(cell, cosine=lambda i: 100 + i, sine=lambda i: 200 - i)
    summary = execution.candidate_summary(cell, signal(cell), cost)
    assert summary["primary_work"] == 300
    assert summary["primary_standard_error"] == 0.0
    assert np.std([100 + i for i in range(32)], ddof=1) > 0


def test_strict_ten_percent_underestimate_boundary():
    cell = copy.deepcopy(plan()["frozen_candidates"][0])
    cell["development_prediction"]["axis_one_shot_compiled_cost"] = {
        axis: {m: 100.0 for m in contract.METRICS} for axis in contract.AXES}
    boundary = execution.candidate_summary(cell, signal(cell), fake_compiled(cell, 110, 110))
    above = execution.candidate_summary(cell, signal(cell), fake_compiled(cell, 111, 111))
    assert boundary["major_cost_underestimate"] is False
    assert boundary["transfer_support_usable"] is True
    assert above["major_cost_underestimate"] is True
    assert above["transfer_support_usable"] is False


def test_usable_b2_rule_controls_both_pareto_and_ratio_and_schema():
    value = plan()
    signals = [signal(cell) for cell in value["frozen_candidates"]]
    costs = [fake_compiled(cell, 1 if cell["method"] == "B2" else 100,
                           1 if cell["method"] == "B2" else 100) for cell in value["frozen_candidates"]]
    result = execution.assemble_result(value, signals, costs)
    schema = contract.load_json(ROOT / execution.RESULT_SCHEMA_PATH)
    execution.validate_result(result, schema)
    assert result["status"] == "TRANSFER_SUPPORTED"
    assert result["next_stage_authorized"] is False
    assert result["mandatory_stop_reached"] is True
    # Only the pre-prediction is synthetic here, to force 20% underestimate.
    for cell in value["frozen_candidates"]:
        if cell["method"] == "B2":
            cell["development_prediction"]["axis_one_shot_compiled_cost"] = {
                axis: {m: 1 / 1.2 for m in contract.METRICS} for axis in contract.AXES}
    result = execution.assemble_result(value, signals, costs)
    assert result["status"] == "TRANSFER_NOT_SUPPORTED"
    assert all(not r["transfer_support_usable"] for r in result["candidate_results"])
    execution.validate_result(result, schema)


@pytest.mark.parametrize("eligible_b2", [False, True])
def test_empty_endpoint_precedence(eligible_b2):
    value = plan()
    signals = [signal(cell) if eligible_b2 and cell["method"] == "B2"
               else signal(cell, real=None) for cell in value["frozen_candidates"]]
    costs = [fake_compiled(cell, 1, 1) for cell in value["frozen_candidates"]]
    result = execution.assemble_result(value, signals, costs)
    assert result["status"] == ("TRANSFER_INCONCLUSIVE" if eligible_b2 else "TRANSFER_NOT_SUPPORTED")
    assert len(result["candidate_results"]) == 5


def synthetic_hamiltonian():
    return DFHamiltonian(
        constant=0.11, one_body=np.diag([0.2, 0.3]).astype(complex),
        lambdas=np.asarray([0.04 + 0.01 * i for i in range(12)]),
        g_matrices=tuple(np.diag([0.7 + 0.01 * i, 0.9 - 0.02 * i]).astype(complex) for i in range(12)),
        metadata={"name": "M2-pure-synthetic-not-H4"},
    )


def test_five_synthetic_signals_use_fixed_m1_formula_and_no_load(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("snapshot load is forbidden in source tests")
    monkeypatch.setattr(np, "load", forbidden)
    state = np.asarray([1, 0, 0, 0], dtype=complex)
    records, preparations, audit = execution.evaluate_signals(
        synthetic_hamiltonian(), state, plan()["frozen_candidates"])
    assert len(records) == len(preparations) == 5
    assert all(r["candidate_fingerprint"] == cell["execution_candidate_fingerprint"]
               for r, cell in zip(records, plan()["frozen_candidates"], strict=True))
    assert audit["ground_state_solves"] == 0
    assert audit["method_parameter_searches"] == 0
    assert all(r["normalization_multiplier"] >= 1 for r in records)


@pytest.mark.parametrize("ordinal", [0, 2, 4])
def test_synthetic_requests_keep_dynamic_delta_and_seed_independence(ordinal):
    from trotterlib.pr2_matched_accuracy_m1_execution import _prepare, _prepare_discard
    hamiltonian = synthetic_hamiltonian()
    cell = plan()["frozen_candidates"][ordinal]
    prep = _prepare_discard(hamiltonian, cell["rank"]) if cell["method"] == "B0" else _prepare(hamiltonian, cell["rank"])
    seed = (cell["future_trajectory_seeds"] or [None])[0]
    request = execution._trajectory_request(prep, cell, seed)
    assert request.repetition_count == cell["q"]
    assert request.step_time == pytest.approx(0.8 / cell["q"])
    assert request.trajectory_seed == seed
    if cell["q"] > 1:
        assert len(set(request.step_seeds)) == cell["q"]
    if seed is not None:
        second = execution._trajectory_request(prep, cell, cell["future_trajectory_seeds"][1])
        assert request.step_seeds != second.step_seeds


def test_synthetic_cell_builds_two_axes_per_seed_and_exact_checkpoint_reuse(tmp_path, monkeypatch):
    from trotterlib.pr2_matched_accuracy_m1_execution import _prepare
    import trotterlib.rte_compiled_cost as costs
    captured = []
    def synthetic_compile(circuit, compiler):
        captured.append((circuit.num_clbits, compiler.optimization_level, compiler.transpiler_seed))
        return SimpleNamespace(
            **{m: 20 for m in contract.METRICS},
            actual_circuit_fingerprint=hashlib.sha256(str(len(captured)).encode()).hexdigest())
    monkeypatch.setattr(costs, "transpile_and_measure_cost", synthetic_compile)
    cell = plan()["frozen_candidates"][0]
    prep = _prepare(synthetic_hamiltonian(), 3)
    job = execution.CompileJob(cell, prep, str(tmp_path / "cell"))
    first = execution.compile_cell(job)
    assert first["full_wrappers_computed"] == 64
    assert first["full_wrappers_reused"] == 0
    assert len(captured) == 64
    assert set(captured) == {(1, 1, 17)}
    second = execution.compile_cell(job)
    assert second["full_wrappers_computed"] == 0
    assert second["full_wrappers_reused"] == 64
    assert len(captured) == 64
    assert second["trajectory_samples"] == 0
    changed = copy.deepcopy(cell)
    changed["wrapper_cache_keys"]["cosine"][0] = "0" * 64
    with pytest.raises(ValueError, match="checkpoint wrapper identity"):
        execution.compile_cell(execution.CompileJob(changed, prep, str(tmp_path / "cell")))


def test_unresolved_or_corrupt_checkpoint_never_recompiled(tmp_path):
    path = tmp_path / "checkpoint.json"
    execution._atomic_json(path, {"wrapper_key": "a" * 64, "state": "compile_reserved"})
    with pytest.raises(ValueError, match="retry prohibited"):
        execution._read_wrapper_checkpoint(path, "a" * 64)
    execution._atomic_json(path, {"wrapper_key": "a" * 64, "state": "complete",
                                 "checkpoint_fingerprint": "b" * 64})
    with pytest.raises(ValueError, match="checksum"):
        execution._read_wrapper_checkpoint(path, "a" * 64)


def test_small_synthetic_baseline_real_qiskit_transpile(tmp_path):
    from trotterlib.pr2_matched_accuracy_m1_execution import _prepare_discard
    cell = plan()["frozen_candidates"][2]
    job = execution.CompileJob(cell, _prepare_discard(synthetic_hamiltonian(), 6), str(tmp_path / "baseline"))
    result = execution.compile_cell(job)
    assert result["full_wrappers_computed"] == 2
    assert result["trajectory_samples"] == result["occurrence_samples"] == 0
    assert all(type(v) is int and v >= 0 for axis in result["paired_trajectory_rows"][0]["axes"].values()
               for v in axis["metrics"].values())


def test_lower_envelope_is_secondary_and_preserves_crossings():
    records = [
        {"candidate_id": "a", "accuracy_eligible": True, "primary_work": 100.0,
         "axis_shots": {"real": 20, "imag": 20}},
        {"candidate_id": "b", "accuracy_eligible": True, "primary_work": 200.0,
         "axis_shots": {"real": 5, "imag": 5}},
    ]
    result = execution.secondary_envelope(records)
    assert result["secondary_only"] is True
    assert result["crossings"][0]["P"] == pytest.approx(100 / 30)
    assert result["intervals"][0]["candidate_id"] == "a"
    assert result["intervals"][-1]["candidate_id"] == "b"


def test_partial_failure_audit_marks_unresolved_attempts_not_zero(tmp_path):
    cell = plan()["frozen_candidates"][0]
    path = tmp_path / "checkpoints" / cell["task_fingerprint"] / "00_cosine.json"
    execution._atomic_json(path, {
        "wrapper_key": cell["wrapper_cache_keys"]["cosine"][0], "state": "compile_reserved"})
    audit = execution.partial_wrapper_audit(tmp_path, [cell])
    assert audit["complete_records"] == 0
    assert audit["unresolved_reservations"] == 1
    assert audit["not_a_complete_resource_count"] is True


def test_committed_sources_reject_dirty_science_code(tmp_path, monkeypatch):
    target = "execution.py"
    (tmp_path / target).write_text("clean\n", encoding="utf-8")
    commit = "a" * 40
    monkeypatch.setattr(execution, "source_paths", lambda *_args: (target,))
    monkeypatch.setattr(execution, "git", lambda _root, *args:
                        (commit + "\n").encode() if args[0] == "rev-parse" else b"clean\n")
    assert execution.committed_source_hashes(tmp_path, commit)[target] == hashlib.sha256(b"clean\n").hexdigest()
    (tmp_path / target).write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="uncommitted"):
        execution.committed_source_hashes(tmp_path, commit)


def test_mocked_one_shot_pipeline_stops_and_cannot_switch_outputs(tmp_path, monkeypatch):
    # Exercise orchestration only: no molecular input, sampling, build or compiler.
    value = plan()
    frozen_path = tmp_path / execution.CONTRACT_PLAN_PATH
    frozen_path.parent.mkdir(parents=True)
    frozen_path.write_bytes((ROOT / execution.CONTRACT_PLAN_PATH).read_bytes())
    schema_path = tmp_path / execution.RESULT_SCHEMA_PATH
    schema_path.write_bytes((ROOT / execution.RESULT_SCHEMA_PATH).read_bytes())
    execution._atomic_json(tmp_path / "execution_plan.json", value)
    execution._atomic_json(tmp_path / "synthetic_auth.json", {"test_fixture_only": True})
    monkeypatch.setattr(execution, "validate_authorization", lambda *_args, **_kwargs: None)
    opened = []
    def synthetic_load(_root, counters):
        opened.append("synthetic model; no held-out path")
        return object(), np.asarray([1]), {}
    monkeypatch.setattr(execution, "load_held_out", synthetic_load)
    monkeypatch.setattr(execution, "evaluate_signals", lambda _ham, _state, cells:
                        ([signal(cell) for cell in cells], [object()] * 5, {"synthetic": True}))
    def synthetic_cell(job):
        cell = job.cell
        result = fake_compiled(cell)
        result.update(full_wrappers_computed=2 * len(result["paired_trajectory_rows"]),
                      full_wrappers_reused=0, peak_rss_kib=0,
                      trajectory_samples=len(cell["future_trajectory_seeds"]),
                      occurrence_samples=len(cell["future_trajectory_seeds"]) * cell["q"] * cell["r"])
        return result
    monkeypatch.setattr(execution, "compile_cell", synthetic_cell)
    result = execution.run_transfer(
        tmp_path, plan_path=tmp_path / "execution_plan.json",
        authorization_path=tmp_path / "synthetic_auth.json", output_relative="artifacts/m2-test", workers=1)
    assert result["status"] == "TRANSFER_SUPPORTED"
    assert result["execution"]["unique_wrapper_evaluations"] == 196
    assert result["next_stage_authorized"] is False
    assert (tmp_path / "artifacts/m2-test/M2_COMPLETE.json").exists()
    assert len(opened) == 1
    with pytest.raises(ValueError, match="already used"):
        execution.run_transfer(
            tmp_path, plan_path=tmp_path / "execution_plan.json",
            authorization_path=tmp_path / "synthetic_auth.json",
            output_relative="artifacts/another-test", workers=1)
    assert len(opened) == 1
