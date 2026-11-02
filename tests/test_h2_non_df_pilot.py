"""Artificial contract/runner checks. No H2 SCF or molecular evolution."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pytest

from trotterlib.df_hamiltonian import DFHamiltonian
from trotterlib.h2_non_df_pilot import (
    canonical_ground, case_config, check, contract, evaluate_case,
    pauli_matrix, precision_accounting, reverse_bits, seed_for,
)
from trotterlib.h2_non_df_pilot_validation import validate_case, validate_df_primitives
from trotterlib.fixed_signal_evaluation import prepare_signal_adapter
from trotterlib.pauli_signal_adapter import PauliHamiltonian

ROOT = Path(__file__).resolve().parents[1]


def runner():
    spec = importlib.util.spec_from_file_location("_h2_pilot_runner", ROOT/"scripts/run_h2_non_df_pilot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixed_contract_one_shot_scope_and_existing_signal_definition():
    plan = contract()
    assert plan["system"]["molecule"] == "H2"
    assert plan["system"]["geometry_Angstrom"][1][1][2] == 1.
    assert plan["evaluation"]["T"] == .8
    assert plan["evaluation"]["q"] == 4
    assert plan["evaluation"]["alpha_axis"] == .025
    assert plan["execution"]["fixed_evaluations"] == 4
    assert not plan["execution"]["automatic_retry"]
    assert not plan["execution"]["candidate_search"]
    assert not plan["execution"]["H4_or_larger"]
    assert plan["DF"]["split_order"] != plan["Pauli"]["split_order"]
    json.dumps(plan, allow_nan=False)


def test_contract_returns_independent_values():
    first = contract()
    first["evaluation"]["q"] = 128
    assert contract()["evaluation"]["q"] == 4


def test_seed_distinguishes_representation_and_method():
    seeds = [seed_for(rep, method) for rep in ["pauli_jw", "df_native"] for method in ["PF", "PR"]]
    assert len(set(seeds)) == 4
    assert seeds[0] == seed_for("pauli_jw", "PF")


def test_bit_order_and_sector_ground():
    assert reverse_bits(2).tolist() == [0, 2, 1, 3]
    energy, state, gap = canonical_ground(np.diag([10., 2., 1., 9.]), np.array([1, 2]))
    assert energy == 1. and gap == 1.
    assert_allclose(state, [0., 0., 1., 0.])
    assert_allclose(pauli_matrix("YX"), np.kron([[0., -1j], [1j, 0.]], [[0., 1.], [1., 0.]]))


@pytest.mark.parametrize("error", [1e-6, float("inf"), float("nan")])
def test_gate_rejects_errors_without_changing_threshold(error):
    with pytest.raises(ValueError):
        check(error, 1e-9, "synthetic_gate")


def test_save_is_exclusive(tmp_path):
    r = runner()
    r.save(tmp_path/"marker.json", dict(state="CONSUMED"))
    with pytest.raises(FileExistsError):
        r.save(tmp_path/"marker.json", dict(state="REFUNDED"))
    assert r.read(tmp_path/"marker.json")["state"] == "CONSUMED"


def test_source_gate_detects_contract_mutation_before_hashing(monkeypatch):
    r = runner()
    plan = contract()
    source = dict(contract_digest=r.digest(plan))
    plan["evaluation"]["q"] = 8
    with pytest.raises(ValueError, match="CONTRACT_CHANGED"):
        r.verify_binding(plan, source)


def test_source_gate_detects_hash_mutation(monkeypatch, tmp_path):
    r = runner()
    p = tmp_path/"source.py"
    p.write_text("changed")
    monkeypatch.setattr(r, "ROOT", tmp_path)
    monkeypatch.setattr(r, "git", lambda *args: "fixed_commit")
    plan = contract()
    source = dict(contract_digest=r.digest(plan), commit="fixed_commit", files={"source.py":"0"*64})
    with pytest.raises(ValueError, match="SOURCE_BYTES_CHANGED"):
        r.verify_binding(plan, source)


def test_consumed_run_cannot_enter_science_again(monkeypatch, tmp_path):
    r = runner()
    (tmp_path/"run01").mkdir()
    monkeypatch.setattr(r, "ROOT", tmp_path)
    monkeypatch.setattr(r, "BASE", tmp_path)
    plan = contract()
    plan["execution"]["run_namespace"] = "run01"
    source = dict(commit="fixed")
    grant = dict(plan_digest=r.digest(plan), source_manifest_sha256="mock",
                 authorization_sha256="mock", SOURCE="fixed")
    monkeypatch.setattr(r, "read", lambda p: plan if p.name=="execution_contract.json" else source if p.name=="SOURCE.json" else grant)
    monkeypatch.setattr(r, "sha", lambda p: "mock")
    monkeypatch.setattr(r, "verify_binding", lambda *args: None)

    def forbidden(*args):
        raise AssertionError("science entered on a consumed run")

    monkeypatch.setattr(r, "build_h2_inputs", forbidden)
    with pytest.raises(FileExistsError):
        r.execute()


@pytest.mark.parametrize("representation", ["pauli_jw", "df_native"])
@pytest.mark.parametrize("method", ["PF", "PR"])
def test_synthetic_case_covers_new_orchestration_without_molecules(representation, method):
    h = (PauliHamiltonian.from_terms([("II", .1), ("ZI", .4), ("XZ", -.2), ("YY", .1)])
         if representation == "pauli_jw" else
         DFHamiltonian(.1, np.array([[.2, .03], [.03, -.1]]), np.array([.4, -.2]),
             (np.array([[.5, .1], [.1, .2]]), np.diag([.1, .6])), {"basis":"artificial"}))
    plan = contract()
    plan["evaluation"].update(T=.2, q=2, PR_cost_samples=2)
    psi = np.array([0., 1., 1j, 0.])/np.sqrt(2)
    c, a, evidence, result, rows = evaluate_case(h, psi, representation, method, plan, dict(SOURCE="synthetic"))
    assert evidence["certified"] is False
    assert result["shots"]["N_total"] is not None
    assert len(rows) == 5 and rows[0]["total_RZ"] is not None
    validation = validate_case(a, psi, c, result, plan)
    assert validation["status"] == "PASS"
    assert not validation["all_trajectories_certified"]
    if representation == "df_native" and method == "PF":
        assert len(validate_df_primitives(a, plan)) == 6


def test_allowance_boundary_guard_keeps_N_G_undefined():
    result = dict(signal=dict(signed_bias=dict(real=0., imag=0.)),
        shots=dict(axes={a:dict(allowance=.001) for a in ["cosine", "sine"]}, evidence_ref="synthetic"),
        normalization=dict(log_B_upper=0.), circuit_cost={})
    row = precision_accounting(result, .1, contract())
    assert row["shots"]["N_total"] is None and row["total_RZ"] is None
