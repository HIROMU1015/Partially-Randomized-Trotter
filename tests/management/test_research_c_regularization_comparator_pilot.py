"""Artificial Stage A orchestration and byte-only contract preflight."""
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import scipy.optimize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"src"))
from trottertracks import c_rcdf_comparator_pilot as m


def reference(mass):
    return dict(optimizer=dict(success=True),
                diagnostic=dict(exact_tail_mass_Ha=mass),
                sector=dict(sector_error=1e-8, numerical_guard=1e-9))


@pytest.mark.parametrize("fault", [None, "initial_budget", "final_budget",
                                  "representation", "optimizer", "tied_mass"])
def test_artificial_17_call_end_to_end_and_STOP(tmp_path, monkeypatch, fault):
    c = dict(fresh_output="run01", entry="runner.py", source_manifest="SOURCE.json",
             comparators={"baseline": {"optimizer": "base_optimizer.json",
                           "diagnostic": "base_diagnostic.json", "sector": "base_sector.json"},
                          "proposed": {"optimizer": "pr_optimizer.json",
                           "diagnostic": "pr_diagnostic.json", "sector": "pr_sector.json"}})
    m.io.save(tmp_path/"SOURCE.json", dict(files={}))
    for role, name in (("baseline", "base"), ("proposed", "pr")):
        for key, value in reference(3. if role == "baseline" else 2.).items():
            m.io.save(tmp_path/(name+"_"+key+".json"), value)
    monkeypatch.setattr(m, "preflight", lambda *a, **kw: c)
    p = dict(h=np.eye(4), g=np.zeros((4,)*4), constant=np.array(.5))
    a = dict(J=np.ones((8, 4, 4)), U=np.array([np.eye(4)]*8))
    monkeypatch.setattr(m, "load", lambda *args: (p, a, {}))
    monkeypatch.setattr(m.old, "tensor_metrics", lambda *args: dict(frobenius=0.))
    monkeypatch.setattr(m.old, "resource_plan", lambda **kw: dict(workers=1))
    monkeypatch.setattr(m.molecule, "SectorCAR",
                        lambda n: SimpleNamespace(physical=lambda *args: np.zeros((2, 2))))

    def assessment(ctx, p, J, U, physical, rank):
        initial = J[0, 0, 0] == 1.
        error = .001 if ((fault == "initial_budget" and initial)
                        or (fault == "final_budget" and not initial)) else 1e-6
        return dict(sector_error=error, numerical_guard=1e-9), [np.zeros((2, 2))]
    monkeypatch.setattr(m.molecule, "rank_assessment", assessment)
    def representation(*args):
        if fault == "representation":
            raise ValueError("ARTIFICIAL_REFERENCE_MISMATCH")
        return dict(status="PASS")
    monkeypatch.setattr(m.molecule, "representation_gate", representation)
    out = tmp_path/"run01"
    def child(args, **kwargs):
        worker = out/"fit"
        worker.mkdir()
        m.existing.arrays(worker/"final.npz", J=2*a["J"], U=a["U"])
        m.io.save(worker/"optimizer.json", dict(success=fault != "optimizer"))
        m.io.save(worker/"final_diagnostic.json",
                  dict(exact_tail_mass_Ha=2. if fault == "tied_mass" else 1.))
        m.io.save(worker/"tensor.json", dict(frobenius=0.))
        m.io.save(worker/"STOP.json", dict(failed=False, call_counts=m.FIT))
    monkeypatch.setattr(m.subprocess, "run", child)
    monkeypatch.setattr(m.subprocess, "check_output", lambda *a, **k: "synthetic-SOURCE\n")
    if fault in ("initial_budget", "representation"):
        with pytest.raises(ValueError):
            m.run(tmp_path, tmp_path/"contract.json", out, "fit-python")
        assert not (out/"fit").exists()
        assert m.io.read(out/"STOP.json")["failed"]
    else:
        m.run(tmp_path, tmp_path/"contract.json", out, "fit-python")
        stop = m.io.read(out/"STOP.json")
        result = m.io.read(out/"result.json")
        assert stop["scientific_units"] == 17 and stop["call_counts"] == m.LIMITS
        assert result["optimizer_success"] == (fault != "optimizer")
        assert result["DF_budget"] == ("NOT_MET" if fault == "final_budget" else "PASS")
        assert result["status"] == (
            "STAGE_A_NOT_ELIGIBLE" if fault in ("optimizer", "final_budget")
            else "STAGE_A_ELIGIBLE_FOR_RESEARCH_DECISION")
        assert result["tail_vs_saved_proposed"] == (
            "UNRESOLVED_WITHIN_ENGINEERING_ALLOWANCE" if fault == "tied_mass"
            else "LOWER_NUMERICAL_POINT")
        assert not result["stages_B_C_authorized"]
        assert all(result[k] is None for k in (
            "shots", "native_cost", "total_cost", "CA_achieved", "PF_error", "RTE_error"))
        before = (out/"STOP.json").read_bytes()
        with pytest.raises(ValueError, match="FRESH"):
            m.run(tmp_path, tmp_path/"contract.json", out, "fit-python")
        assert (out/"STOP.json").read_bytes() == before


def test_real_fit_worker_on_artificial_rank8_zero_iterations(tmp_path):
    rng = np.random.default_rng(2003)
    a = rng.normal(size=(8, 4, 4))*.02
    J = a+a.transpose(0, 2, 1)
    U = np.array([np.eye(4)]*8)
    U[0, :, 0] *= -1
    U[1, :2, :2] *= -1
    g = m.old.reconstruct(J, U)+1e-4*np.einsum("pq,rs->pqrs", np.eye(4), np.eye(4))
    J, U, x, info = m.objective.pr.prepare(g, J, U, ld=2)
    # The worker keeps the fixed Stage A scale. Scale the artificial residual
    # to that value; never load a molecular snapshot.
    g = m.old.reconstruct(J, U)+np.sqrt(
        m.CONDITIONS["loss_scale_Ha_squared"]/(.5*16))*np.einsum(
            "pq,rs->pqrs", np.eye(4), np.eye(4))
    info["spec"]["loss_scale"] = m.CONDITIONS["loss_scale_Ha_squared"]
    m.existing.arrays(tmp_path/"input.npz", h=np.eye(4), g=g, constant=np.array(.5))
    m.existing.arrays(tmp_path/"initial.npz", J=J, U=U, x0=x)
    m.io.save(tmp_path/"scales.json", info)
    m.io.save(tmp_path/"initial_gate.json", dict(status="PASS"))
    c = dict(input_snapshot="input.npz", initial_snapshot="initial.npz",
             initial_scales="scales.json")
    def minimize(fun, x0, **kwargs):
        np.testing.assert_array_equal(x0, x)
        v, dg = fun(x0)
        return scipy.optimize.OptimizeResult(
            x=x0.copy(), jac=dg, fun=v, success=True, status=0,
            message="MOCK_ZERO_ITERATIONS", nit=0, nfev=1, njev=1)
    with patch.object(scipy.optimize, "minimize", side_effect=minimize) as opt:
        m.fit_worker(tmp_path, tmp_path/"fit", c)
    opt.assert_called_once()
    assert m.io.read(tmp_path/"fit/optimizer.json")["nit"] == 0
    assert m.io.read(tmp_path/"fit/STOP.json")["scientific_units"] == 6
    assert m.io.read(tmp_path/"fit/initial_diagnostic.json")["combined_objective"] == pytest.approx(1.1, abs=1e-8)


def test_byte_preflight_binds_opaque_input_and_source(tmp_path):
    for name in ("source.py", "input.npz", "initial.npz", "scales.json"):
        (tmp_path/name).write_bytes(b"invented opaque bytes; not molecular data")
    m.io.save(tmp_path/"SOURCE.json",
              dict(files={"source.py": m.io.sha(tmp_path/"source.py")}))
    c = dict(scope=m.SCOPE, status="UNAPPROVED_PROPOSAL",
        conditions=m.CONDITIONS, call_limits=m.LIMITS,
        maximum_logical_units=17, objective=m.objective.VERSION,
        stages_B_C_authorized=False, fresh_output=m.FAMILY+"/run01",
        initialization=m.objective.pr.INITIALIZATION,
        source_manifest="SOURCE.json",
        source_manifest_sha256=m.io.sha(tmp_path/"SOURCE.json"),
        saved_inputs={name: m.io.sha(tmp_path/name) for name in
                      ("input.npz", "initial.npz", "scales.json")},
        input_snapshot="input.npz", initial_snapshot="initial.npz",
        initial_scales="scales.json", comparators={}, evaluation_packages={})
    m.io.save(tmp_path/"contract.json", c)
    assert m.preflight(tmp_path, tmp_path/"contract.json") == c
    (tmp_path/"initial.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SOURCE_OR_INPUT"):
        m.preflight(tmp_path, tmp_path/"contract.json")
    assert not (tmp_path/"run01").exists()
