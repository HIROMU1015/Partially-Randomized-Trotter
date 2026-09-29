from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from trotterlib.pr2_matched_accuracy_m1_contract import (
    enumerate_base_candidates,
    fingerprint,
)
from trotterlib.pr2_matched_accuracy_m1_execution import (
    MAXIMUM_SIGNAL_CANDIDATES,
    _apply_exponential,
    _apply_spectral_values,
    _axis_record,
    _eigendecomposition,
    _explicit_cutoff_tolerance,
    validate_m1_a_result,
)
from trotterlib.rte import (
    InvolutoryTailTerm,
    make_rte_config,
    normalize_involutory_tail,
    step_taylor_truncation_residual_bound,
)


ROOT = Path(__file__).resolve().parents[1]


def _minimal_result(*, limited: bool) -> dict:
    candidate = enumerate_base_candidates()[0]
    ledger = [copy.deepcopy(candidate) for _ in range(208)]
    for index, item in enumerate(ledger):
        item["candidate_id"] = f"synthetic-{index}"
    reasons = ["unselected_proxy_nondominated_candidates"] if limited else []
    status = "SELECTION_LIMITED" if limited else "M1_A_COMPLETE_M1_B_ELIGIBLE"
    payload = {
        "schema_version": "pr2_matched_accuracy_m1_a_result_v2",
        "series_id": "pr2-rebaseline-de7a5492-v1",
        "status": status,
        "execution_authorization_sha256": "a" * 64,
        "development_snapshot": {},
        "candidate_ledger": ledger,
        "signal_records": [{} for _ in ledger],
        "compile_selection": {
            "selection_limited": limited,
            "selection_limited_reasons": reasons,
        },
        "precompile_barrier": {
            "status": status,
            "compile_jobs_materialized_at_barrier": 0,
        },
        "compile_records": [],
        "counters": {
            "development_npz_loads": 1,
            "held_out_path_stats": 0,
            "held_out_raw_hash_checks": 0,
            "held_out_npz_loads": 0,
            "molecular_calculations": 0,
            "signal_evaluations": 208,
            "circuits_built": 0,
            "circuit_compilations": 0,
            "random_trajectories_sampled": 0,
            "random_trajectories_compiled": 0,
            "full_wrappers_compiled": 0,
            "quantum_shots_executed": 0,
        },
        "decision": {},
        "held_out": {},
        "S3_authorized": False,
        "automatic_next_stage": None,
    }
    payload["result_fingerprint"] = fingerprint(payload)
    return payload


def test_dense_spectral_actions_match_direct_eigendecomposition() -> None:
    matrix = np.asarray([[0.4, 0.2j], [-0.2j, -0.7]], dtype=np.complex128)
    vector = np.asarray([0.6 + 0.1j, -0.3 + 0.7j], dtype=np.complex128)
    eigensystem = _eigendecomposition(matrix)
    values, vectors = eigensystem
    time_value = 0.31
    direct = vectors @ (
        np.exp(-1j * time_value * values) * (vectors.conj().T @ vector)
    )
    assert np.allclose(
        _apply_exponential(eigensystem, vector, time_value), direct, atol=1e-13
    )
    polynomial = 1.0 - 0.2j * values - 0.02 * values**2
    assert np.allclose(
        _apply_spectral_values(eigensystem, vector, polynomial),
        vectors @ (polynomial * (vectors.conj().T @ vector)),
        atol=1e-13,
    )


def test_axis_shots_enforce_normalization_and_bias_allowance() -> None:
    biases, shots, total = _axis_record(0.5 + 0.1j, 0.5 + 0.1j, 1.25)
    assert biases == {"real": 0.0, "imag": 0.0}
    assert shots["real"] == shots["imag"]
    assert total == shots["real"] + shots["imag"]
    _biases, failed_shots, failed_total = _axis_record(
        0.6 + 0.1j, 0.5 + 0.1j, 1.0
    )
    assert failed_shots["real"] is None
    assert failed_total is None


@pytest.mark.parametrize("limited", [False, True])
def test_result_validator_accepts_only_consistent_barrier(limited: bool) -> None:
    payload = _minimal_result(limited=limited)
    validate_m1_a_result(payload)
    broken = copy.deepcopy(payload)
    broken["precompile_barrier"]["compile_jobs_materialized_at_barrier"] = 1
    broken["result_fingerprint"] = fingerprint(
        {key: value for key, value in broken.items() if key != "result_fingerprint"}
    )
    with pytest.raises(ValueError, match="compile jobs"):
        validate_m1_a_result(broken)


def test_signal_candidate_cap_and_base_count_are_frozen() -> None:
    assert len(enumerate_base_candidates()) == 208
    assert MAXIMUM_SIGNAL_CANDIDATES == 212


@pytest.mark.parametrize(("tau", "cutoff"), [(2.0, 2), (4.0, 4)])
def test_explicit_cutoff_tolerance_accepts_frozen_candidate(
    tau: float,
    cutoff: int,
) -> None:
    tail = normalize_involutory_tail(
        "synthetic-z",
        (
            InvolutoryTailTerm(
                "z",
                1.0,
                np.diag([1.0, -1.0]).astype(np.complex128),
            ),
        ),
    )
    tolerance = _explicit_cutoff_tolerance(tau, cutoff)
    config, _distribution = make_rte_config(
        tail,
        evolution_time=tau,
        rte_steps=1,
        truncation_tolerance=tolerance,
        finite_taylor_order=cutoff,
    )
    residual = step_taylor_truncation_residual_bound(tau, cutoff)
    assert config.step_truncation_residual_bound == residual
    assert tolerance >= residual


def test_execution_source_has_no_heldout_path_or_circuit_builder_calls() -> None:
    path = ROOT / "src/trotterlib/pr2_matched_accuracy_m1_execution.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert "HELD_OUT_RELATIVE_PATH" not in source
    assert "QuantumCircuit" not in source
    assert "transpile(" not in source
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "qiskit" not in imported_modules


def test_execution_authorization_hashes_and_prohibitions_are_fixed() -> None:
    path = (
        ROOT
        / "artifacts/pr2_matched_accuracy_m1_execution/2026-09-30/"
        "pr2_matched_accuracy_m1_execution_authorization_v1_1.json"
    )
    authorization = json.loads(path.read_text(encoding="utf-8"))
    for relative, expected in authorization["required_source_hashes"].items():
        observed = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert observed == expected
    permissions = authorization["permissions"]
    assert permissions["m1_a_scientific_execution_authorized"] is True
    assert permissions["development_npz_load_authorized"] is True
    assert permissions["held_out_access_authorized"] is False
    assert permissions["m1_b_direct_compile_under_this_authorization"] is False
    assert permissions["trajectory_sampling_authorized"] is False
    assert permissions["circuit_build_authorized"] is False
    assert permissions["direct_compile_authorized"] is False
    assert permissions["quantum_shots_authorized"] is False
