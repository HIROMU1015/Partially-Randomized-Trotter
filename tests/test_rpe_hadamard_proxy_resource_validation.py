from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import qiskit

from trotterlib.df_hamiltonian import DFHamiltonian, PhysicalSector
from trotterlib.rpe_hadamard_proxy_resource_validation import (
    validate_rpe_hadamard_proxy_resource_connection,
    validate_rpe_hadamard_proxy_resource_connection_payload,
)
from trotterlib.rte import CompilerSettings


def _hamiltonian() -> DFHamiltonian:
    return DFHamiltonian(
        constant=0.0,
        one_body=np.asarray(
            [[0.2, 0.0], [0.0, -0.1]],
            dtype=np.complex128,
        ),
        lambdas=np.asarray([0.5]),
        g_matrices=(
            np.asarray([[1.0, 0.0], [0.0, 0.4]], dtype=np.complex128),
        ),
        metadata={"name": "proxy-resource-toy"},
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


def test_proxy_resource_validation_connects_only_passed_holdout() -> None:
    bundle = validate_rpe_hadamard_proxy_resource_connection(
        _hamiltonian(),
        PhysicalSector.number_sector(n_qubits=2, n_electrons=1),
        _compiler(),
        ld=0,
        delta_time=0.05,
        calibration_q_values=(1, 2),
        holdout_q_values=(8,),
        rte_steps_per_occurrence=1,
        finite_taylor_order=0,
        sample_count=2,
        pf_coefficient=0.0,
        relative_error_tolerance=1.0,
        rz_relative_standard_error_tolerance=0.5,
        model_label="toy",
        basis_label="toy",
        provenance={"test": True},
    )
    payload = bundle.connection
    assert payload["summary"]["overall_pass"]
    assert payload["summary"]["validated_q_m_values"] == [8]
    assert payload["unvalidated_q_rejection"]["rejected"]
    assert payload["scope"]["final_total_cost_evaluation_performed"] is False
    candidate = payload["resource_candidates"][0]
    assert candidate["q_m"] == 8
    assert candidate["circuit_cost_scope"] == (
        "single_hadamard_interrogation_without_state_preparation"
    )
    validate_rpe_hadamard_proxy_resource_connection_payload(payload)

    tampered = deepcopy(payload)
    tampered["summary"]["checks"]["benchmark_complete"] = False
    with pytest.raises(ValueError, match="content_fingerprint"):
        validate_rpe_hadamard_proxy_resource_connection_payload(tampered)
