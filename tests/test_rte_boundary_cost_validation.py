from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import qiskit

from trotterlib.df_hamiltonian import DFHamiltonian
from trotterlib.rte import CompilerSettings
from trotterlib.rte_boundary_cost_validation import (
    validate_rte_boundary_cost_model,
    validate_rte_boundary_cost_payload,
    write_rte_boundary_cost_validation,
)


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
        metadata={"name": "rte-boundary-cost-validation-toy"},
    )


def test_boundary_cluster_validation_is_held_out_and_tamper_evident(
    tmp_path,
) -> None:
    payload = validate_rte_boundary_cost_model(
        _hamiltonian(),
        ld=0,
        reference_delta_time=0.08,
        reference_rte_steps=4,
        finite_taylor_order=0,
        compiler=_compiler(),
        calibration_sample_count=4,
        holdout_sequence_lengths=(4,),
        holdout_sample_count=4,
        maximum_exact_events=100,
        maximum_samples=10,
        provenance={"test": True},
    )

    assert payload["final_cost_evaluation_performed"] is False
    assert payload["acceptance_threshold_decided"] is False
    assert payload["configuration"]["calibration_sequence_lengths"] == [1, 2, 3]
    assert payload["configuration"]["holdout_sequence_lengths"] == [4]
    assert payload["scope"]["fixed_short_step_distribution"] is True
    metric_result = payload["holdout_predictions"][0]["metrics"]["rz_count"]
    assert set(metric_result) == {
        "naive_additive",
        "pair_corrected",
        "triple_corrected",
    }
    coefficients = payload["cluster_coefficients"]["metrics"]["rz_count"]
    mu1 = coefficients["single_event_mu1"]
    mu2 = coefficients["pair_boundary_mu2"]
    mu3 = coefficients["triple_residual_mu3"]
    assert metric_result["naive_additive"]["prediction"] == pytest.approx(4 * mu1)
    assert metric_result["pair_corrected"]["prediction"] == pytest.approx(
        4 * mu1 + 3 * mu2
    )
    assert metric_result["triple_corrected"]["prediction"] == pytest.approx(
        4 * mu1 + 3 * mu2 + 2 * mu3
    )
    calibration_seeds = {
        item["seed"]
        for item in payload["calibration_sequence_estimates"].values()
    }
    holdout_seeds = {item["seed"] for item in payload["holdout_predictions"]}
    assert calibration_seeds.isdisjoint(holdout_seeds)
    validate_rte_boundary_cost_payload(payload)

    output = tmp_path / "validation.json"
    write_rte_boundary_cost_validation(payload, output)
    assert output.exists()

    tampered = deepcopy(payload)
    tampered["fragment_distribution"][
        "same_fragment_boundary_probability"
    ] = 0.0
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_rte_boundary_cost_payload(tampered)


def test_boundary_validation_v1_rejects_nonzero_taylor_order() -> None:
    with pytest.raises(ValueError, match="requires K=0"):
        validate_rte_boundary_cost_model(
            _hamiltonian(),
            ld=0,
            reference_delta_time=0.08,
            reference_rte_steps=4,
            finite_taylor_order=2,
            compiler=_compiler(),
            calibration_sample_count=2,
            holdout_sequence_lengths=(4,),
            holdout_sample_count=2,
            maximum_exact_events=100,
            maximum_samples=2,
        )
