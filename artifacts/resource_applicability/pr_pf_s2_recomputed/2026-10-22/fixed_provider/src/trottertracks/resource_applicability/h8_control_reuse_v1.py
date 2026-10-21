"""Reuse E|psi,0> and E|psi,1> for both Hadamard axes without rescaling.

Linearity preserves arbitrary system/ancilla mixing and circuit global phase.
The two expensive actual evolution simulations remain independent branch
probes. X/Y suffix gates are evaluated with explicit vector formulas.
"""
import math
import numpy as np


def hadamard_states_from_branches(low_evolved, high_evolved, n):
    if type(n) is not int or not 1 <= n <= 16:
        raise ValueError('HADAMARD_REUSE_SYSTEM_QUBITS')
    a = np.asarray(low_evolved, dtype=np.complex128)
    b = np.asarray(high_evolved, dtype=np.complex128)
    dim = 1 << n
    if a.shape != (2*dim,) or b.shape != a.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('HADAMARD_REUSE_BRANCH_LAYOUT_OR_NONFINITE')
    combined = (a+b)/math.sqrt(2.)
    low, high = combined[:dim], combined[dim:]
    cosine = np.concatenate(((low+high)/math.sqrt(2.), (low-high)/math.sqrt(2.)))
    sine = np.concatenate(((low-1j*high)/math.sqrt(2.), (low+1j*high)/math.sqrt(2.)))
    if not np.isfinite(cosine).all() or not np.isfinite(sine).all():
        raise ValueError('HADAMARD_REUSE_RESULT_NONFINITE')
    return dict(cosine=cosine, sine=sine)


def validate_control(evolution, qstate, *, calls, simulate, expectation, expected_signal=None,
                     previous_high=None):
    n = evolution.num_system_qubits
    psi = np.asarray(qstate, dtype=complex)
    if psi.shape != (1 << n,) or not np.isfinite(psi).all() or abs(np.linalg.norm(psi)-1.) > 1e-12:
        raise ValueError('CONTROL_REUSE_SAVED_STATE')
    low = np.concatenate((psi, np.zeros_like(psi)))
    high = np.concatenate((np.zeros_like(psi), psi))
    calls.take('control_probe', 2)
    a = simulate(evolution.circuit, low)
    b = simulate(evolution.circuit, high)
    errors = [float(np.linalg.norm(a-low)), float(np.linalg.norm(b[:len(psi)]))]
    z = complex(np.vdot(psi, b[len(psi):]))
    if expected_signal is not None:
        errors.append(abs(z-complex(expected_signal['real'], expected_signal['imag'])))
    if previous_high is not None:
        errors.append(float(np.linalg.norm(b-previous_high)))
    for axis, v in hadamard_states_from_branches(a, b, n).items():
        # Logical probe budget is unchanged; record the actual algorithm too.
        calls.take('control_probe')
        expected = z.real if axis == 'cosine' else z.imag
        errors.append(abs(expectation(v, n)-expected))
    if not all(math.isfinite(e) for e in errors):
        raise ValueError('CONTROL_REUSE_NONFINITE_ERROR')
    return dict(max_error=max(errors), signal=dict(real=float(z.real), imag=float(z.imag)),
        probe='actual saved-state evolution branches plus linear Hadamard X/Y suffix',
        full_evolution_simulations=2, expensive_full_wrapper_simulations=0,
        reused_axis_states=2, logical_control_probes=4,
        normalization_policy='no renormalization', random_signal_is_ensemble=False,
        validation_method_changed=True, certified=False), b
