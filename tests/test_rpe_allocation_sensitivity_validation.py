from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import qiskit

from trotterlib.df_hamiltonian import DFHamiltonian, PhysicalSector
from trotterlib.rpe_allocation_sensitivity_validation import (
    validate_rpe_allocation_sensitivity,
    validate_rpe_allocation_sensitivity_payload,
)
from trotterlib.rpe_hadamard_proxy_resource_validation import (
    validate_rpe_hadamard_proxy_resource_connection,
)
from trotterlib.rpe_round_cost_connection_validation import (
    validate_rpe_round_cost_connection,
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
        metadata={"name": "allocation-sensitivity-toy"},
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


def test_allocation_sensitivity_uses_fixed_costs_and_is_tamper_evident() -> None:
    hamiltonian = _hamiltonian()
    compiler = _compiler()
    sector = PhysicalSector.number_sector(n_qubits=2, n_electrons=1)
    direct = validate_rpe_round_cost_connection(
        hamiltonian,
        sector,
        compiler,
        ld=0,
        delta_time=0.05,
        q_values=(1,),
        rte_steps_per_occurrence=1,
        finite_taylor_order=2,
        compiled_cost_sample_count=2,
        pf_coefficient=0.01,
        provenance={"test": "direct"},
    )
    proxy_bundle = validate_rpe_hadamard_proxy_resource_connection(
        hamiltonian,
        sector,
        compiler,
        ld=0,
        delta_time=0.05,
        calibration_q_values=(2, 4),
        holdout_q_values=(8,),
        rte_steps_per_occurrence=1,
        finite_taylor_order=2,
        sample_count=2,
        pf_coefficient=0.01,
        relative_error_tolerance=1.0,
        rz_relative_standard_error_tolerance=1.0,
        model_label="toy",
        basis_label="toy",
        provenance={"test": "proxy"},
    )
    payload = validate_rpe_allocation_sensitivity(
        hamiltonian,
        compiler,
        direct,
        proxy_bundle.proxy_validation,
        ld=0,
        delta_time=0.05,
        q_values=(1, 8),
        rte_steps_per_occurrence=1,
        finite_taylor_order=2,
        beta_profiles=(
            ("current_provisional", 0.08, 0.08, 0.24),
            ("guarded_provisional", 0.02, 0.02, 0.36),
        ),
        pf_coefficient=0.01,
        minimum_provisional_headroom_factor=2.0,
        provenance={"test": True},
    )

    assert payload["summary"]["overall_pass"]
    assert payload["summary"]["scenario_count"] == 4
    assert payload["scope"]["one_shot_costs_recompiled_during_sweep"] is False
    assert payload["scope"]["final_total_cost_evaluation_performed"] is False
    assert payload["selection"]["selected_alpha_policy"] == (
        "cost_sensitivity_weighted"
    )
    validate_rpe_allocation_sensitivity_payload(payload)

    tampered = deepcopy(payload)
    tampered["selection"]["selected_total_shots"] += 1
    with pytest.raises(ValueError, match="content_fingerprint"):
        validate_rpe_allocation_sensitivity_payload(tampered)
