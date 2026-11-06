"""Explicit scientific check harness for the single permitted Track D fixture.

Called once by run_track_d_minimal_reference.py, not by ordinary pytest collection.
"""
from __future__ import annotations

import numpy as np
from qiskit.quantum_info import Operator


def run_checks(fixture, implementation, matrix_gate):
    tolerance = implementation.TOLERANCE
    records = []

    def same(actual, expected, label):
        records.append(matrix_gate(actual, expected, tolerance, label=label))

    v = fixture.physical_reference
    same(Operator(fixture.physical).data, v, "physical_vs_independent_occupation_reference")
    same(v.conj().T @ v, np.eye(4), "physical_unitarity")
    expected_vacuum = np.zeros((4, 1), complex)
    expected_vacuum[0, 0] = np.exp(1j * fixture.phase)
    same(v[:, :1], expected_vacuum, "known_vacuum_phase_no_energy_oracle")
    for i, (actual, expected) in enumerate(fixture.basis_checks):
        same(actual, expected, "existing_gaussian_Fock_lift_" + str(i))
    a, b = Operator(fixture.later_half).data, Operator(fixture.earlier_half).data
    same(a @ b, v, "AB_is_the_original_full_path")
    same(Operator(fixture.later_half.inverse()).data, a.conj().T, "actual_adjoint_including_scalar")

    expected_controlled = np.zeros((8, 8), complex)
    expected_controlled[:4, :4] = np.eye(4)
    expected_controlled[4:, 4:] = v
    for arm in implementation.ARMS[:2]:
        same(Operator(fixture.native_controls[arm]).data, expected_controlled, arm + ":existing_native")
        same(Operator(fixture.cores[arm]).data, expected_controlled, arm + ":shared_fused")

    gadget = Operator(fixture.cores["reference"]).data
    expected_gadget = np.zeros((32, 32), complex)
    expected_gadget[:16, :16] = np.kron(b, a.conj().T)
    expected_gadget[16:, 16:] = np.exp(1j * fixture.phase) * np.kron(a.conj().T, b)
    same(gadget, expected_gadget, "SE_two_cswap_layers_and_branch_order")

    observables = {}
    for (arm, axis), wrapper in fixture.wrappers.items():
        actual = implementation.induced_observable(wrapper)
        same(actual, implementation.expected_observable(v, axis), arm + ":" + axis + ":all_input_states")
        observables[arm + ":" + axis] = implementation.complex_array(actual)
    negative = {}
    for axis, wrapper in fixture.negative_wrappers.items():
        actual = implementation.induced_observable(wrapper)
        difference = float(np.max(np.abs(actual - implementation.expected_observable(v, axis))))
        if not np.isfinite(difference) or difference <= tolerance:
            raise ValueError("Omitted reference phase did not fail: " + axis)
        negative[axis] = {"omitted_phase_detected": True, "maximum_entry_difference": difference}
    return dict(status="PASS", tolerance=tolerance, checks=records,
                induced_observables=observables, omitted_phase_negative_control=negative)
