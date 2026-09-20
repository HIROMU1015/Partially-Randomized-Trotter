from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import qiskit

from trotterlib.df_hamiltonian import DFHamiltonian, PhysicalSector
from trotterlib.rpe_round_cost_connection_validation import (
    validate_rpe_round_cost_connection,
    validate_rpe_round_cost_connection_payload,
    write_rpe_round_cost_connection_validation,
)
from trotterlib.rte import CompilerSettings


def _hamiltonian() -> DFHamiltonian:
    return DFHamiltonian(
        constant=0.13,
        one_body=np.asarray(
            [[0.2, 0.03], [0.03, -0.1]],
            dtype=np.complex128,
        ),
        lambdas=np.asarray([0.2, -0.3]),
        g_matrices=(
            np.asarray([[1.0, 0.0], [0.0, 0.4]], dtype=np.complex128),
            np.asarray([[0.2, 0.0], [0.0, 2.0]], dtype=np.complex128),
        ),
        metadata={"name": "round-cost-connection-toy"},
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


def test_short_round_connection_passes_and_is_tamper_evident(tmp_path) -> None:
    payload = validate_rpe_round_cost_connection(
        _hamiltonian(),
        PhysicalSector.number_sector(n_qubits=2, n_electrons=1),
        _compiler(),
        ld=1,
        delta_time=0.05,
        q_values=(1,),
        rte_steps_per_occurrence=1,
        finite_taylor_order=0,
        compiled_cost_sample_count=2,
        pf_coefficient=0.0,
        provenance={"test": True},
    )

    assert payload["summary"]["overall_pass"]
    assert payload["summary"]["round_count"] == 1
    assert payload["final_cost_evaluation_performed"] is False
    round_result = payload["rounds"][0]
    assert round_result["checks"]["round_cost_identity_pass"]
    assert round_result["checks"]["cost_scopes_are_distinct_and_correct"]
    assert round_result["time_evolution_candidate"]["circuit_cost_scope"] == (
        "compiled_time_evolution_subcircuit"
    )
    assert round_result["hadamard_interrogation_candidate"][
        "circuit_cost_scope"
    ] == "single_hadamard_interrogation_without_state_preparation"

    target = tmp_path / "connection.json"
    write_rpe_round_cost_connection_validation(payload, target)
    assert target.exists()

    tampered = deepcopy(payload)
    tampered["rounds"][0]["checks"]["round_cost_identity_pass"] = False
    with pytest.raises(ValueError, match="check summary mismatch"):
        validate_rpe_round_cost_connection_payload(tampered)
