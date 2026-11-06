"""One authorized real-word check sequence; not automatic test collection."""
from __future__ import annotations

import math
import numpy as np


def run_checks(bundle, implementation, output, records, threads):
    from trottertracks.c1_stage3_molecular import component_check
    from threadpoolctl import threadpool_limits
    m = implementation
    arrays = {}

    def same(a, b, label, tolerance=m.ATOL):
        row = component_check(a, b, tolerance, label=label)
        records.append(row)
        return row

    def observable(actual, expected, label):
        row = same(actual, expected, label)
        # Numerical Frobenius diagnostic also controls every normalized-input
        # expectation in exact arithmetic; this is not a rounding enclosure.
        if row["frobenius"] > m.SIGNAL_ATOL:
            raise ValueError(label + ":observable_Frobenius_above_signal_allowance")
        arrays[label.replace(":", "_")] = actual

    v = bundle.reference
    d = len(v); eye = np.eye(d, dtype=complex)
    target = np.zeros((2*d, 2*d), complex)
    target[:d, :d] = eye; target[d:, d:] = v
    same(v.conj().T @ v, eye, "saved_CAR_reference_unitarity")
    a, b, adj = [m.small_operator(q) for q in (bundle.later, bundle.earlier, bundle.later_adjoint)]
    same(a @ b, v, "AB_saved_ordered_path_scalar_once")
    same(adj, a.conj().T, "actual_later_adjoint_including_scalar")
    same(adj @ a, eye, "actual_later_adjoint_returns_identity")
    for name, matrix in (("A", a), ("B", b)):
        same(matrix.conj().T @ matrix, eye, name + ":unitarity")
        expected = np.zeros((d, 1), complex); expected[0, 0] = matrix[0, 0]
        same(matrix[:, :1], expected, name + ":vacuum_ray")
    vacuum = np.zeros((d, 1), complex); vacuum[0, 0] = np.exp(1j*bundle.phase)
    same(v[:, :1], vacuum, "analytic_metadata_chi_no_ground_or_phase_fit")
    same(a[:, :1] * b[0, 0], vacuum, "split_vacuum_phase")
    saved = m.small_operator(bundle.saved_raw)
    same(saved, target, "saved_raw_vs_saved_CAR")
    for arm in m.ARMS[:2]:
        value = m.small_operator(bundle.cores[arm])
        same(value, target, arm + ":both_control_branches")
        same(value, saved, arm + ":unchanged_original_operator_phase")
        same(value.conj().T @ value, np.eye(2*d), arm + ":unitarity")
        arrays["raw_core_" + arm] = value
        for axis in m.AXES:
            wrapper = bundle.wrappers[arm, axis].remove_final_measurements(inplace=False)
            columns = m.small_operator(wrapper)[:, :d].T
            actual = m.observable_from_columns(columns, d)
            observable(actual, m.artificial.expected_observable(v, axis), "raw_" + arm + "_" + axis)
    # Complete 256-dimensional input subspace, all 131072 output amplitudes.
    columns = m.raw_reference_columns(bundle, threads)
    u0 = b[0, 0] * adj / math.sqrt(2)
    u1 = np.exp(1j*bundle.phase) * adj[0, 0] * b / math.sqrt(2)
    leakage = m.reference_column_check(columns, u0, u1, same, "raw_reference_isometry")
    half = columns.shape[1] // 2
    with threadpool_limits(limits=threads):
        same(columns.conj() @ columns.T, eye, "raw_reference_input_isometry")
        coherence = 2 * (columns[:, :half].conj() @ columns[:, half:].T)
    same(coherence, v, "raw_reference_full_complex_coherence")
    for axis in m.AXES:
        observable(m.artificial.expected_observable(coherence, axis),
            m.artificial.expected_observable(v, axis), "raw_reference_" + axis)
    negative = {}
    # P(chi) is the final core operation: removing it multiplies the measured
    # coherence by exp(-i chi). This contraction uses the actual raw isometry.
    uncorrected = coherence * np.exp(-1j*bundle.phase)
    for axis in m.AXES:
        error = float(np.max(abs(m.artificial.expected_observable(uncorrected, axis) - m.artificial.expected_observable(v, axis))))
        if not np.isfinite(error) or error <= m.ATOL:
            raise ValueError("OMITTED_CHI_NOT_DETECTED:" + axis)
        negative[axis] = dict(omitted_phase_detected=True, max_entry_difference=error)
    arrays.update(V=v, A=a, B=b, A_adjoint=adj, raw_branch0=u0, raw_branch1=u1)
    np.savez_compressed(output / "references.npz", **arrays)
    return dict(status="PASS", checks=records, reference_leakage=leakage,
        omitted_chi_negative_control=negative, input_basis_columns=256,
        reference_output_amplitudes_per_column=131072,
        full_reference_Operator_constructed=False, rigorous_enclosure=None)


def check_compiled(compiled, arm, axis, arrays, m, threads):
    from trottertracks.c1_stage3_molecular import component_check
    from trottertracks.c1_h6_state_action import StateActions, measured_tail
    records = []
    def same(a, b, label, tolerance=m.ATOL):
        row = component_check(a, b, tolerance, label=label); records.append(row); return row
    clean = measured_tail(compiled)
    d = 1 << m.N
    phase_factor = 1 if axis == "X" else -1j
    if arm == "reference":
        inputs = np.zeros((d, 1 << (2*m.N+1)), complex)
        inputs[np.arange(d), np.arange(d)] = 1
        states = StateActions(threads).apply(clean, inputs)
        u0, u1 = arrays["raw_branch0"], arrays["raw_branch1"]
        expected0 = (u0 + phase_factor*u1) / math.sqrt(2)
        expected1 = (u0 - phase_factor*u1) / math.sqrt(2)
        leakage = m.reference_column_check(states, expected0, expected1, same, "compiled_reference_vs_raw_isometry")
        same(states.conj() @ states.T, np.eye(d), "compiled_reference_input_isometry")
        obs = m.observable_from_columns(states, states.shape[1]//2)
    else:
        matrix = m.small_operator(clean)
        shell = np.kron(np.array([[1., 1.], [1., -1.]]) / math.sqrt(2), np.eye(d))
        tail = shell if axis == "X" else shell @ np.diag(np.r_[np.ones(d), -1j*np.ones(d)])
        raw = tail @ arrays["raw_core_" + arm] @ shell
        same(matrix, raw, "compiled_vs_raw_full_wrapper_global_phase")
        same(matrix.conj().T @ matrix, np.eye(2*d), "compiled_full_wrapper_unitarity")
        obs = m.observable_from_columns(matrix[:, :d].T, d)
        leakage = None
    for expected, label in [(m.artificial.expected_observable(arrays["V"], axis), "compiled_vs_CAR_observable"),
                            (arrays["raw_" + arm + "_" + axis], "compiled_vs_raw_observable")]:
        row = same(obs, expected, label)
        if row["frobenius"] > m.SIGNAL_ATOL:
            raise ValueError(label + ":observable_signal_allowance")
    return obs, dict(status="PASS", checks=records, reference_leakage=leakage,
        input_state="all 256 system basis columns; reference/ancilla initially zero",
        rigorous_enclosure=None, global_phase_fit=False)
