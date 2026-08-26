from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import qiskit

from trotterlib.df_hamiltonian import DFHamiltonian
from trotterlib.random_circuit_cost_validation import (
    validate_random_circuit_cost_model,
    validate_random_circuit_cost_payload,
    write_random_circuit_cost_validation,
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
        metadata={"name": "random-circuit-cost-validation-toy"},
    )


def test_paired_cost_validation_is_serializable_and_tamper_evident(tmp_path) -> None:
    payload = validate_random_circuit_cost_model(
        _hamiltonian(),
        ld=0,
        delta_time=0.05,
        rte_steps=2,
        finite_taylor_order=0,
        monte_carlo_sample_counts=(2, 4),
        compiler=_compiler(),
        maximum_exact_event_sequences=100,
        maximum_samples=10,
        provenance={"test": True},
    )

    assert payload["final_cost_evaluation_performed"] is False
    assert payload["acceptance_threshold_decided"] is False
    assert payload["partial_s2"]["exact"] is not None
    assert len(payload["partial_s2"]["monte_carlo"]) == 2
    assert len(payload["rte_occurrence"]["monte_carlo"]) == 2
    assert payload["scope"]["paired_sampling"] is True
    validate_random_circuit_cost_payload(payload)

    output = tmp_path / "validation.json"
    write_random_circuit_cost_validation(payload, output)
    assert output.exists()

    tampered = deepcopy(payload)
    tampered["summary"][
        "largest_monte_carlo_sample_count"
    ] = 999
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_random_circuit_cost_payload(tampered)


def test_exact_enumeration_is_explicitly_skipped_above_limit() -> None:
    payload = validate_random_circuit_cost_model(
        _hamiltonian(),
        ld=0,
        delta_time=0.05,
        rte_steps=2,
        finite_taylor_order=0,
        monte_carlo_sample_counts=(1,),
        compiler=_compiler(),
        maximum_exact_event_sequences=0,
        maximum_samples=1,
    )

    assert payload["partial_s2"]["exact"] is None
    assert "exceeds" in payload["partial_s2"]["exact_skip_reason"]


def test_occurrence_only_scope_skips_the_more_expensive_partial_s2_path() -> None:
    payload = validate_random_circuit_cost_model(
        _hamiltonian(),
        ld=0,
        delta_time=0.05,
        rte_steps=2,
        finite_taylor_order=0,
        monte_carlo_sample_counts=(2,),
        compiler=_compiler(),
        evaluation_scopes=("rte_occurrence",),
        maximum_exact_event_sequences=100,
        maximum_samples=2,
    )

    assert payload["partial_s2"]["status"] == "not_requested"
    assert payload["partial_s2"]["exact"] is None
    assert payload["partial_s2"]["monte_carlo"] == []
    assert payload["rte_occurrence"]["status"] == "executed"
    assert len(payload["rte_occurrence"]["monte_carlo"]) == 1
