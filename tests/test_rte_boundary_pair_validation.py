from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import qiskit

from trotterlib.df_hamiltonian import DFHamiltonian
from trotterlib.rte import CompilerSettings
from trotterlib.rte_boundary_pair_validation import (
    validate_rte_boundary_pair_payload,
    validate_rte_boundary_pairs,
    write_rte_boundary_pair_validation,
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
        constant=0.0,
        one_body=np.zeros((2, 2), dtype=np.complex128),
        lambdas=np.asarray([0.7, 0.7]),
        g_matrices=(
            np.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128),
            np.asarray([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128),
        ),
        metadata={"name": "rte-boundary-pair-validation-toy"},
    )


def test_pair_stratification_uses_disjoint_holdout_and_is_tamper_evident(
    tmp_path,
) -> None:
    payload = validate_rte_boundary_pairs(
        _hamiltonian(),
        ld=0,
        reference_delta_time=0.08,
        reference_rte_steps=4,
        finite_taylor_order=0,
        compiler=_compiler(),
        calibration_sample_count=20,
        holdout_sample_count=20,
        calibration_seed=101,
        holdout_seed=202,
        maximum_samples=20,
        provenance={"test": True},
    )

    assert payload["final_cost_evaluation_performed"] is False
    assert payload["configuration"]["calibration_seed"] == 101
    assert payload["configuration"]["holdout_seed"] == 202
    assert payload["calibration"]["same_fragment"]["sample_count"] > 1
    assert payload["calibration"]["different_fragment"]["sample_count"] > 1
    assert payload["calibration"]["directed_fragment_pair_rows"]
    assert set(payload["holdout_model_comparison"]["metrics"]["rz_count"]) == {
        "same_fragment_only",
        "same_vs_different",
    }
    validate_rte_boundary_pair_payload(payload)

    output = tmp_path / "pair-validation.json"
    write_rte_boundary_pair_validation(payload, output)
    assert output.exists()

    tampered = deepcopy(payload)
    tampered["fragment_distribution"]["same_fragment_probability"] = 0.0
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_rte_boundary_pair_payload(tampered)


def test_pair_stratification_rejects_nonzero_taylor_order() -> None:
    with pytest.raises(ValueError, match="requires K=0"):
        validate_rte_boundary_pairs(
            _hamiltonian(),
            ld=0,
            reference_delta_time=0.08,
            reference_rte_steps=4,
            finite_taylor_order=2,
            compiler=_compiler(),
            calibration_sample_count=2,
            holdout_sample_count=2,
            calibration_seed=1,
            holdout_seed=2,
            maximum_samples=2,
        )
