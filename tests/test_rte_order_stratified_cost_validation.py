from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import qiskit

from trotterlib.df_hamiltonian import DFHamiltonian
from trotterlib.rte import CompilerSettings
from trotterlib.rte_order_stratified_cost_validation import (
    _cluster_prediction_forms,
    _connected_window_prediction,
    validate_paired_cluster_payload,
    validate_paired_order_stratified_k2_cluster_model,
    validate_order_stratified_cost_payload,
    validate_order_stratified_k2_cost_model,
    write_order_stratified_cost_validation,
    write_paired_cluster_validation,
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
        metadata={"name": "order-stratified-cost-validation-toy"},
    )


def test_cluster_forms_reduce_to_the_standard_triple_formula() -> None:
    forms = _cluster_prediction_forms()

    assert forms[4]["0,0,0,0"] == {
        "2:0,0": -1.0,
        "3:0,0,0": 2.0,
    }
    assert forms[6]["0,0,0,0,0,0"] == {
        "2:0,0": -3.0,
        "3:0,0,0": 4.0,
    }


def test_connected_window_prediction_includes_four_event_coefficients() -> None:
    sequence_length = 6
    windows = {}
    for width in range(1, 5):
        for start in range(sequence_length - width + 1):
            value = width + 0.25 * start + 0.5 * width**2
            windows[(start, width)] = np.full(6, value)

    prediction = _connected_window_prediction(
        windows,
        sequence_length=sequence_length,
        maximum_cluster_length=4,
    )
    expected = sum(windows[(start, 1)] for start in range(sequence_length))
    for width in range(2, 5):
        for start in range(sequence_length - width + 1):
            expected += (
                windows[(start, width)]
                - windows[(start, width - 1)]
                - windows[(start + 1, width - 1)]
            )
            if width > 2:
                expected += windows[(start + 1, width - 2)]
    assert np.allclose(prediction, expected)


def test_order_stratified_validation_forces_all_patterns_and_fingerprints(
    tmp_path,
) -> None:
    payload = validate_order_stratified_k2_cost_model(
        _hamiltonian(),
        ld=0,
        reference_delta_time=0.08,
        reference_rte_steps=4,
        compiler=_compiler(),
        common_sample_count=2,
        single_rare_sample_count=2,
        multi_rare_sample_count=2,
        seed=101,
        maximum_workers=1,
        provenance={"test": True},
    )

    assert payload["final_cost_evaluation_performed"] is False
    distribution = payload["configuration"]["distribution"]
    assert distribution["orders"] == [0, 2]
    assert sum(distribution["order_probabilities"]) == pytest.approx(1.0)
    assert len(payload["calibration"]["3"]["strata"]) == 8
    assert len(payload["holdout"]["6"]["strata"]) == 64
    rare = payload["holdout"]["4"]["strata"]["2,0,0,0"]
    assert rare["sample_count"] == 2
    assert rare["forced_order_counts"] == {"0": 6, "2": 2}
    seeds = {
        stratum["seed"]
        for role in ("calibration", "holdout")
        for length in payload[role].values()
        for stratum in length["strata"].values()
    }
    assert len(seeds) == 2 + 4 + 8 + 16 + 64
    for length in ("4", "6"):
        aggregates = payload["holdout_predictions"][length]
        assert aggregates["full_distribution"]["unnormalized_probability"] == pytest.approx(1.0)
        assert aggregates["exactly_one_order2_condition"][
            "unnormalized_probability"
        ] > 0.0
        assert aggregates["two_or_more_order2_condition"][
            "unnormalized_probability"
        ] > 0.0
        assert set(
            aggregates["two_or_more_order2_condition"][
                "normalized_pattern_weights"
            ]
        ) == {
            key
            for key, value in payload["holdout"][length]["strata"].items()
            if value["rare_order_count"] >= 2
        }

    validate_order_stratified_cost_payload(payload)
    output = tmp_path / "order-stratified.json"
    write_order_stratified_cost_validation(payload, output)
    assert output.exists()

    tampered = deepcopy(payload)
    tampered["holdout"]["4"]["strata"]["2,0,0,0"]["sample_count"] = 3
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_order_stratified_cost_payload(tampered)


def test_paired_validation_uses_identical_local_and_full_trajectories(
    tmp_path,
) -> None:
    payload = validate_paired_order_stratified_k2_cluster_model(
        _hamiltonian(),
        ld=0,
        reference_delta_time=0.08,
        reference_rte_steps=4,
        compiler=_compiler(),
        common_sample_count=2,
        single_rare_sample_count=2,
        seed=303,
        maximum_workers=1,
        provenance={"test": True},
    )

    assert set(payload["holdout"]) == {"4", "6"}
    assert len(payload["holdout"]["4"]["strata"]) == 5
    assert len(payload["holdout"]["6"]["strata"]) == 7
    rare = payload["holdout"]["6"]["strata"]["0,0,2,0,0,0"]
    assert rare["forced_order_counts"] == {"0": 10, "2": 2}
    metric = payload["paired_residual_results"]["4"][
        "exactly_one_order2_condition"
    ]["metrics"]["rz_count"]
    assert metric["prediction_minus_actual"] == pytest.approx(
        metric["prediction"] - metric["actual"]
    )
    validate_paired_cluster_payload(payload)
    output = tmp_path / "paired.json"
    write_paired_cluster_validation(payload, output)
    assert output.exists()

    tampered = deepcopy(payload)
    tampered["configuration"]["common_sample_count"] = 3
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_paired_cluster_payload(tampered)


def test_paired_validation_can_force_multiple_order2_patterns(tmp_path) -> None:
    payload = validate_paired_order_stratified_k2_cluster_model(
        _hamiltonian(),
        ld=0,
        reference_delta_time=0.08,
        reference_rte_steps=4,
        compiler=_compiler(),
        common_sample_count=2,
        single_rare_sample_count=2,
        multi_rare_sample_count=2,
        seed=404,
        maximum_workers=1,
        provenance={"test": True},
    )

    assert len(payload["holdout"]["4"]["strata"]) == 16
    assert len(payload["holdout"]["6"]["strata"]) == 64
    for length in ("4", "6"):
        result = payload["paired_residual_results"][length][
            "two_or_more_order2_condition"
        ]
        assert set(result["patterns"]) == {
            key
            for key, value in payload["holdout"][length]["strata"].items()
            if value["rare_order_count"] >= 2
        }
        assert sum(result["conditional_pattern_weights"].values()) == pytest.approx(
            1.0
        )
    validate_paired_cluster_payload(payload)
    output = tmp_path / "paired-multi.json"
    write_paired_cluster_validation(payload, output)
    assert output.exists()
