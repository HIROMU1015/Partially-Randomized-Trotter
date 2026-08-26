from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import qiskit

from trotterlib.df_hamiltonian import DFHamiltonian
from trotterlib.hierarchical_cost_validation import (
    validate_controlled_repetition_extension,
    validate_hierarchical_cost_payload,
    validate_rte_cluster_extension,
    write_hierarchical_cost_validation,
)
from trotterlib.rte import CompilerSettings


def _compiler() -> CompilerSettings:
    return CompilerSettings(
        basis_gates=("rz", "sx", "x", "cx"),
        backend_name=None,
        coupling_map=None,
        optimization_level=1,
        layout_method=None,
        routing_method=None,
        transpiler_seed=17,
        qiskit_version=qiskit.__version__,
    )


def _hamiltonian() -> DFHamiltonian:
    return DFHamiltonian(
        constant=0.11,
        one_body=np.asarray([[0.2]], dtype=np.complex128),
        lambdas=np.asarray([0.7]),
        g_matrices=(np.asarray([[1.0]], dtype=np.complex128),),
        metadata={"name": "hierarchical-cost-validation-toy"},
    )


def test_rte_cluster_extension_uses_independent_holdout_and_fingerprints(
    tmp_path,
) -> None:
    payload = validate_rte_cluster_extension(
        _hamiltonian(),
        ld=0,
        reference_delta_time=0.08,
        reference_rte_steps=4,
        finite_taylor_order=0,
        compiler=_compiler(),
        calibration_sample_count=4,
        holdout_sequence_lengths=(4,),
        holdout_sample_count=4,
        seed=101,
        maximum_exact_events=100,
        maximum_samples=10,
        provenance={"test": True},
    )

    assert payload["final_cost_evaluation_performed"] is False
    assert payload["configuration"]["finite_taylor_order"] == 0
    assert payload["c1"]["estimate_kind"].startswith("exact")
    metric = payload["holdout_predictions"][0]["metrics"]["rz_count"]
    c1 = payload["c1"]["expected_cost"]["rz_count"]
    c2 = payload["calibration"]["2"]["full_expected_cost"]["rz_count"]
    c3 = payload["calibration"]["3"]["full_expected_cost"]["rz_count"]
    assert metric["naive"]["prediction"] == pytest.approx(4 * c1)
    assert metric["pair"]["prediction"] == pytest.approx(-2 * c1 + 3 * c2)
    assert metric["triple"]["prediction"] == pytest.approx(-c2 + 2 * c3)
    calibration_seeds = {
        item["seed"] for item in payload["calibration"].values()
    }
    holdout_seeds = {item["seed"] for item in payload["holdout"].values()}
    assert calibration_seeds.isdisjoint(holdout_seeds)
    validate_hierarchical_cost_payload(payload)

    output = tmp_path / "cluster.json"
    write_hierarchical_cost_validation(payload, output)
    assert output.exists()

    tampered = deepcopy(payload)
    tampered["configuration"]["finite_taylor_order"] = 2
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_hierarchical_cost_payload(tampered)


def test_rte_cluster_extension_supports_k2_monte_carlo_c1() -> None:
    payload = validate_rte_cluster_extension(
        _hamiltonian(),
        ld=0,
        reference_delta_time=0.08,
        reference_rte_steps=4,
        finite_taylor_order=2,
        compiler=_compiler(),
        calibration_sample_count=4,
        holdout_sequence_lengths=(4,),
        holdout_sample_count=4,
        seed=211,
        maximum_exact_events=1,
        maximum_samples=10,
    )

    assert payload["configuration"]["finite_taylor_order"] == 2
    assert payload["configuration"]["event_space_size"] > 1
    assert payload["c1"]["estimate_kind"].startswith("monte_carlo")
    assert payload["c1"]["seed"] == 212
    validate_hierarchical_cost_payload(payload)


def test_controlled_q4_is_held_out_from_q1_q2_affine_calibration() -> None:
    payload = validate_controlled_repetition_extension(
        _hamiltonian(),
        ld=0,
        reference_delta_time=0.08,
        reference_rte_steps=1,
        finite_taylor_order=0,
        compiler=_compiler(),
        calibration_sample_count=4,
        holdout_sample_count=4,
        seed=307,
        maximum_samples=10,
    )

    assert payload["configuration"]["controlled"] is True
    assert payload["configuration"]["evaluation_mode"] == "selected_only"
    assert payload["configuration"]["calibration_repetition_counts"] == [1, 2]
    assert payload["configuration"]["holdout_repetition_counts"] == [4]
    c1 = payload["estimates"]["1"]["expected_cost"]["rz_count"]
    c2 = payload["estimates"]["2"]["expected_cost"]["rz_count"]
    result = payload["q4_affine_holdout_prediction"]["metrics"]["rz_count"]
    assert result["prediction"] == pytest.approx(-2 * c1 + 3 * c2)
    calibration_seeds = {
        payload["estimates"]["1"]["seed"],
        payload["estimates"]["2"]["seed"],
    }
    assert payload["estimates"]["4"]["seed"] not in calibration_seeds
    validate_hierarchical_cost_payload(payload)


def test_controlled_q8_is_held_out_from_q1_q2_affine_calibration() -> None:
    payload = validate_controlled_repetition_extension(
        _hamiltonian(),
        ld=0,
        reference_delta_time=0.08,
        reference_rte_steps=1,
        finite_taylor_order=0,
        compiler=_compiler(),
        calibration_sample_count=4,
        holdout_sample_count=4,
        holdout_repetition_count=8,
        seed=401,
        maximum_samples=10,
    )

    c1 = payload["estimates"]["1"]["expected_cost"]["rz_count"]
    c2 = payload["estimates"]["2"]["expected_cost"]["rz_count"]
    result = payload["q8_affine_holdout_prediction"]["metrics"]["rz_count"]
    assert result["prediction"] == pytest.approx(-6 * c1 + 7 * c2)
    assert payload["configuration"]["holdout_repetition_counts"] == [8]
    validate_hierarchical_cost_payload(payload)
