"""Artificial numerical diagnostics only; no saved molecular-result analysis.

This file does not repair/requalify the P1 runner. It separates construction
error from matrix-power arithmetic on tiny, explicitly artificial examples.
Run with --output to save scalar technical evidence. No private input option.
"""
import argparse
from contextlib import ExitStack
from fractions import Fraction
import hashlib
import inspect
import json
from pathlib import Path
import platform
import sys
from types import SimpleNamespace
from unittest.mock import patch

import mpmath
import numpy as np
from scipy.linalg import schur

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from trotterlib import b2_p1_strong_pf_validation as existing
from trotterlib.df_trotter.ops import DiagonalEvolutionPrimitives

Q = 2**17  # Artificial repetition test; no real-H4 signal is evaluated.


def mp_matrix(ctx, matrix):
    # mpf(float) preserves the exact binary64 value, not a decimal surrogate.
    return ctx.matrix([[ctx.mpc(float(complex(x).real), float(complex(x).imag))
                        for x in row] for row in np.asarray(matrix)])


def maximum(ctx, matrix):
    return max((abs(x) for x in matrix), default=ctx.mpf(0))


def error(ctx, left, right):
    return maximum(ctx, left-right)


def exact_unitary_control():
    u = np.array([[0., -1.], [1., 0.]], complex)
    value = np.linalg.matrix_power(u, Q)
    assert np.array_equal(value, np.eye(2))
    return dict(status='PASS', q=Q, max_error=0.,
                meaning='Exactly representable unitary powers need not drift')


def radial_input_defect():
    ctx = mpmath.mp.clone(); ctx.dps = 80
    eta = 2.**-46
    u = np.diag([1., -1.]).astype(complex)
    a = (1+eta)*u
    rows = []
    for q in (1, 2**8, 2**15, Q):
        calculated = mp_matrix(ctx, np.linalg.matrix_power(a, q))
        exact_stored_power = mp_matrix(ctx, a)**q
        exact_unitary = mp_matrix(ctx, np.linalg.matrix_power(u, q))
        predicted = ctx.expm1(q*ctx.log1p(ctx.mpf(eta)))
        observed = error(ctx, calculated, exact_unitary)
        arithmetic = error(ctx, calculated, exact_stored_power)
        assert abs(observed-predicted) < ctx.mpf('1e-14')
        rows.append(dict(q=q, measured_difference=float(observed),
            predicted_input_amplification=float(predicted), power_arithmetic_error=float(arithmetic)))
    assert rows[-1]['measured_difference'] > existing.p.MATCH
    return dict(status='PASS', eta=eta,
        base_unitarity_max_abs=float(np.max(abs(a.conj().T@a-np.eye(2)))), rounds=rows,
        meaning='Known radial input defect is amplified even by an exact power')


def construction_vs_power_precision():
    ctx = mpmath.mp.clone(); ctx.dps = 80
    theta = ctx.mpf(float(.3))
    c, s = ctx.cos(theta), ctx.sin(theta)
    exact_step = ctx.matrix([[c, -s], [s, c]])
    stored_step = np.asarray(exact_step.tolist(), complex)
    exact_stored_power = mp_matrix(ctx, stored_step)**Q
    direct = ctx.matrix([[ctx.cos(Q*theta), -ctx.sin(Q*theta)],
                         [ctx.sin(Q*theta), ctx.cos(Q*theta)]])
    computed = mp_matrix(ctx, np.linalg.matrix_power(stored_step, Q))
    input_error = error(ctx, exact_stored_power, direct)
    multiplication_error = error(ctx, computed, exact_stored_power)
    full_precision_error = error(ctx, exact_step**Q, direct)
    assert full_precision_error < ctx.mpf('1e-65')
    assert input_error > full_precision_error and multiplication_error > full_precision_error
    return dict(status='PASS', q=Q, decimal_precision=ctx.dps,
        exact_binary64_angle=float(theta),
        high_precision_power_of_stored_input_error=float(input_error),
        binary64_power_arithmetic_error=float(multiplication_error),
        binary64_total_error=float(error(ctx, computed, direct)),
        high_precision_construction_and_power_error=ctx.nstr(full_precision_error, 20),
        meaning='Raising precision after storing a step removes neither its input error nor its nonunitarity')


def schur_counterexamples():
    """Negative controls only: never clip eigenvalues or alter real results."""
    eta = 2.**-46
    rows = []
    for name, a in (
        ('radial', np.diag([1+eta, 1.])),
        ('nonnormal', np.array([[1., eta], [0., 1.]]))):
        t, z = schur(a, output='complex')
        faithful = z@np.linalg.matrix_power(t, Q)@z.conj().T
        direct = np.linalg.matrix_power(a.astype(complex), Q)
        # Deliberately invalid alternative, demonstrating what gets discarded.
        phase_only = (z*np.exp(1j*Q*np.angle(np.diag(t))))@z.conj().T
        residual = float(np.max(abs(faithful-direct)))
        discarded_effect = float(np.max(abs(phase_only-direct)))
        assert residual < 1e-14 and discarded_effect > existing.p.MATCH
        rows.append(dict(example=name, faithful_Schur_power_error=residual,
            phase_only_wrong_operator_error=discarded_effect))
    return dict(status='PASS', q=Q, examples=rows,
        meaning='Full Schur T preserves the input defect; phase-only replacement can hide it by changing the operator')


def zero_time_gaussian_control():
    ref = existing.p.car_reference()  # Existing helper, no main()/input load.
    diagonal = DiagonalEvolutionPrimitives(0., (), ())
    schedule = ((0, Fraction(0)), (1, Fraction(0)), (0, Fraction(0)))
    eta = 2.**-46
    values = []
    for scale in (1., 1+eta):
        blocks = tuple(existing.strong.native.NativeBlock(str(i),
            existing.strong.native._rotation(rotation), diagonal)
            for i, rotation in enumerate((np.eye(2), scale*np.eye(2))))
        model = existing.strong.native.NativeModel('ARTIFICIAL', blocks, 0., 'toy_SOURCE', 'toy_input')
        step, leak = existing.native_mean_step(model, schedule, ref, np.arange(4), scalar_time=0.)
        value = np.linalg.matrix_power(step, Q)
        delta = float(np.max(abs(value-np.eye(4))))
        assert leak == 0
        if scale == 1:
            assert delta == 0
        else:
            # The two relative transitions each lift scale*I; the doubly
            # occupied component scales as scale**4 per step, with no phase.
            ctx = mpmath.mp.clone(); ctx.dps = 80
            prediction = float(ctx.expm1(4*Q*ctx.log1p(ctx.mpf(eta))))
            assert abs(delta-prediction) < 1e-13 and delta > existing.p.MATCH
        values.append(dict(orbital_scale=scale, base_identity_error=float(np.max(abs(step-np.eye(4)))),
            base_unitarity_error=float(np.max(abs(step.conj().T@step-np.eye(4)))),
            powered_identity_error=delta))
    return dict(status='PASS', q=Q, examples=values,
        meaning='Exterior lifts of near-unitary matrices need not be unitary, even for a zero-time diagnostic')


def wrong_order_control():
    ctx = mpmath.mp.clone(); ctx.dps = 80
    identity = np.eye(2)
    # Artificial Pauli generators, with a signed term and explicit identity.
    generators = (.37*identity+np.diag([-.8, .8]), np.array([[0., .6], [.6, 0.]]))
    schedule = ((0, Fraction(1, 8)), (1, Fraction(-1, 16)))
    def exponential(t, h):
        rational = Fraction(t)
        time = ctx.mpf(rational.numerator)/rational.denominator
        return np.asarray(ctx.expm(-ctx.j*time*mp_matrix(ctx, h)).tolist(), dtype=object)
    arithmetic = SimpleNamespace(eye=lambda n: np.asarray(ctx.eye(n).tolist(), object),
                                 exponential=exponential)
    right_mp = existing._ordered_exponentials(generators, schedule, arithmetic=arithmetic)
    wrong_mp = existing._ordered_exponentials(generators, tuple(reversed(schedule)), arithmetic=arithmetic)
    right64 = existing._ordered_exponentials(generators, schedule)
    wrong_difference = maximum(ctx, ctx.matrix((right_mp-wrong_mp).tolist()))
    precision_difference = error(ctx, mp_matrix(ctx, right64), ctx.matrix(right_mp.tolist()))
    assert wrong_difference > ctx.mpf('1e-4') and precision_difference < ctx.mpf('1e-14')
    return dict(status='PASS', high_precision_wrong_order_difference=float(wrong_difference),
        same_order_binary64_vs_high_precision=float(precision_difference),
        meaning='A genuine order mismatch survives high precision; increasing precision is not a semantic repair')


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    def forbidden(*args, **kwargs):
        raise AssertionError('This diagnostic may not load molecular data, run H4, build circuits or calculate shots')
    # Reuse the artificial-test boundaries of test_b2_p1_strong_pf_validation.
    from qiskit import QuantumCircuit
    cases = (exact_unitary_control, radial_input_defect, construction_vs_power_precision,
             schur_counterexamples, zero_time_gaussian_control, wrong_order_control)
    records = {}
    with ExitStack() as stack:
        for target, name in ((np, 'load'), (existing.p, 'load_private_inputs'),
            (existing.saved, 'fixed_json'), (existing.p, 'public_json'),
            (existing, 'run'), (QuantumCircuit, '__init__'),
            (existing.strong.StrongBuilder, 'sequence'),
            (existing.p.connection.accounting, '_hoeffding_shots')):
            stack.enter_context(patch.object(target, name, forbidden))
        for case in cases:
            records[case.__name__] = case()
            print(case.__name__+': PASS', flush=True)
    result = dict(status='SYNTHETIC_TECHNICAL_DIAGNOSTICS_PASS', synthetic_only=True,
        real_H4_evaluated=False, saved_molecular_numerics_reanalyzed=False,
        production_algorithm_modified=False, historical_status_changed=False,
        script_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        numpy_matrix_power_source_SHA256=hashlib.sha256(inspect.getsource(np.linalg.matrix_power).encode()).hexdigest(),
        fixed_scientific_SOURCE='a7940b58c6cb939d77e9f8e44d214b3787b7e101',
        fixed_result='2583ce4957ed44e914ce6571e3e676830439dfa7',
        unchanged_P1_independent_threshold=existing.p.MATCH,
        environment=dict(host=platform.node(), python=platform.python_version(),
            numpy=np.__version__, mpmath=mpmath.__version__),
        cases=records, mandatory_stop=True)
    (output/'synthetic_results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.output)
