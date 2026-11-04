"""Numerical bridge primitives. Inert on import; no real-data loader or RNG.

Kernel coefficients are scientific inputs with a derivation, not fitted toy
tolerances. The installed BLAS contract is deliberately not invented here.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
import math
from pathlib import Path
import sys

import numpy as np
from flint import arb, acb, acb_mat

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'docs/research/rpe_m17_rigorous_mean_enclosure_preparation_20261103'))
import ball_adapter as b
import mean_adapter as ideal


def nonnegative(value):
    value = value if isinstance(value, arb) else b.real(value)
    if not value.is_finite() or not value >= 0:
        raise ValueError('Nonnegative finite bound required')
    return b.upper(value)


def interval(saved):
    lo, hi = b.rational(saved['lower']), b.rational(saved['upper'])
    if lo > hi:
        raise ValueError('Reversed interval')
    # Midpoint/radius construction includes rational conversion uncertainty.
    return b.real((lo+hi)/2) + arb(0, b.upper(b.real((hi-lo)/2)))


def exact_matrix(matrix):
    matrix = np.asarray(matrix)
    if matrix.dtype != np.complex128 or matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError('Finite complex128 matrix required')
    return acb_mat([[acb(b.real(float(z.real)), b.real(float(z.imag))) for z in row] for row in matrix])


def matrix_residual(numeric, target):
    """||numeric-target||_2 bound, retaining the ideal global radius."""
    numeric = exact_matrix(numeric)
    if (numeric.nrows(), numeric.ncols()) != (target.center.nrows(), target.center.ncols()):
        raise ValueError('Matrix shape mismatch')
    return b.upper(b.frobenius(numeric-target.center)+target.radius)


def component_signs(item, ordered, n_qubits):
    support = item['support']
    if (item['sign'] not in (-1, 1) or len(set(support)) != len(support)
            or any(type(w) is not int or not 0 <= w < n_qubits for w in support)):
        raise ValueError('Invalid component support/sign')
    return [item['sign']*(-1)**sum((state >> q) & 1 for q in support) for state in ordered]


def numeric_basis(spec, ordered, n_qubits, n_electrons):
    """Legacy full-space Operator composition, using saved local matrices only.

    Creating an Operator from an array is numerical matrix composition, not
    regeneration/build/compile of a controlled or molecular circuit.
    """
    from qiskit.quantum_info import Operator
    ideal.validate_sector(ordered, n_qubits, n_electrons)
    op = Operator(np.eye(1 << n_qubits, dtype=np.complex128))
    for operation in spec['operations']:
        wires = operation['qubits']
        if (len(set(wires)) != len(wires)
                or any(type(w) is not int or not 0 <= w < n_qubits for w in wires)):
            raise ValueError('Invalid basis wires')
        pairs = operation['local_binary64_matrix']
        size = 1 << len(wires)
        if len(pairs) != size or any(len(row) != size for row in pairs):
            raise ValueError('Local matrix shape mismatch')
        raw = np.asarray([[complex(float.fromhex(z[0]), float.fromhex(z[1])) for z in row]
                          for row in pairs], dtype=np.complex128)
        if not np.isfinite(raw).all():
            raise ValueError('Nonfinite local matrix')
        op = op.compose(Operator(raw), qargs=wires)
    data = np.asarray(op.data, dtype=np.complex128)
    return np.ascontiguousarray(data[np.ix_(ordered, ordered)])


def numeric_components(export):
    """Once-only future numerical construction; never called on H4 by tests."""
    ordered, n, electrons = export['ordered_sector'], export['n_qubits'], export['n_electrons']
    ideal.validate_sector(ordered, n, electrons)
    bases = {key: numeric_basis(spec, ordered, n, electrons) for key, spec in export['bases'].items()}
    result = {}
    for item in export['components']:
        if item['id'] in result or item['basis_id'] not in bases:
            raise ValueError('Invalid/duplicate component ID')
        signs = np.asarray(component_signs(item, ordered, n), dtype=np.float64)
        basis = bases[item['basis_id']]
        result[item['id']] = np.ascontiguousarray((basis*signs[np.newaxis, :]) @ basis.conj().T)
    return result


def component_bounds(export, numeric):
    """Pure numerical residual pass; basis cache; no q/r or trajectory work."""
    ordered, n, electrons = export['ordered_sector'], export['n_qubits'], export['n_electrons']
    if set(numeric) != {item['id'] for item in export['components']}:
        raise ValueError('Numeric component inventory mismatch')
    bases = {key: ideal.basis_matrix(spec, ordered, n, electrons) for key, spec in export['bases'].items()}
    out = {}
    for item in export['components']:
        basis = bases[item['basis_id']]
        signs = component_signs(item, ordered, n)
        target = b.product(b.product(basis, ideal.diagonal(signs, 1)), b.adjoint(basis))
        delta = matrix_residual(numeric[item['id']], target)
        out[item['id']] = {'residual': delta, 'frobenius': b.frobenius(exact_matrix(numeric[item['id']]))}
    return out


def decode_matrix_ball(saved):
    rows, cols = saved['shape']
    if type(rows) is not int or type(cols) is not int or min(rows, cols) < 1 or len(saved['center']) != rows*cols:
        raise ValueError('Invalid saved MatrixBall')
    matrix = acb_mat([[b.zball(saved['center'][i*cols+j]) for j in range(cols)] for i in range(rows)])
    # Converting exact saved centers at a lower precision can add uncertainty.
    return b.recenter(matrix, nonnegative(saved['spectral_radius_upper']),
                      nonnegative(saved['target_norm_upper']))


@dataclass(frozen=True)
class KernelBounds:
    """Declared, derived bounds: mv error <= mv_rel*||A||F*||v||+mv_abs.

    scale error <= scale_rel*|scalar|*||v||+scale_abs;
    subtraction error <= sub_rel*(||x||+||y||)+sub_abs;
    dot error <= dot_rel*||psi||*||v||+dot_abs.
    Coefficients include underflow if applicable. No default real BLAS values.
    """
    mv_rel: object
    mv_abs: object
    scale_rel: object
    scale_abs: object
    sub_rel: object
    sub_abs: object
    dot_rel: object
    dot_abs: object

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            nonnegative(getattr(self, name))


@dataclass(frozen=True)
class Growth:
    """Unitary action error <= (gain-1)||v|| + additive; norm <= gain||v||+additive."""
    gain: object
    additive: object = 0

    def __post_init__(self):
        if not nonnegative(self.gain) >= 1:
            raise ValueError('Gain must be >= 1')
        nonnegative(self.additive)


def compose(after, before):
    return Growth(b.upper(nonnegative(after.gain)*nonnegative(before.gain)),
                  b.upper(nonnegative(after.gain)*nonnegative(before.additive)+nonnegative(after.additive)))


def power(value, count):
    if type(count) is not int or count < 0:
        raise ValueError('Nonnegative integer count required')
    result = Growth(1)
    while count:
        if count & 1:
            result = compose(value, result)
        count >>= 1
        if count:
            value = compose(value, value)
    return result


def action_growth(residual, frobenius, kernel):
    return Growth(b.upper(1+nonnegative(residual)+nonnegative(kernel.mv_rel)*nonnegative(frobenius)), kernel.mv_abs)


def rotation_growth(theta, cosine64, sine64, residual, frobenius, kernel):
    """P-product already acted. Bound c*v - i*s*(P64@v) against ideal rotation.

    Finite +/-1 complex phase and -1j*s64 are exact sign/swap operations.
    No renormalization, clipping, mean evaluation or RNG occurs here.
    """
    if any(type(x) is not float or not math.isfinite(x) for x in (cosine64, sine64)):
        raise ValueError('Finite binary64 trig outputs required')
    t = b.real(theta)
    c, s = b.real(cosine64), b.real(sine64)
    cabs, sabs = b.upper(abs(c)), b.upper(abs(s))
    delta = nonnegative(residual)
    mv = nonnegative(kernel.mv_rel)*nonnegative(frobenius)
    ma, kr, sa = map(nonnegative, (kernel.mv_abs, kernel.scale_rel, kernel.scale_abs))
    sr, aa = map(nonnegative, (kernel.sub_rel, kernel.sub_abs))
    pnorm = 1+delta
    # A=c*v and B=-i*s*fl(Pv), each with a complex scalar multiplication.
    norm_a, abs_a = (1+kr)*cabs, sa
    norm_b, abs_b = (1+kr)*sabs*(pnorm+mv), (1+kr)*sabs*ma+sa
    ideal_difference = abs(c-t.cos())+abs(s-t.sin())+sabs*delta
    relative = ideal_difference + kr*cabs + sabs*mv + kr*sabs*(pnorm+mv) + sr*(norm_a+norm_b)
    additive = abs_a+abs_b+sr*(abs_a+abs_b)+aa
    return Growth(b.upper(1+relative), b.upper(additive))


def trajectory_growth(forward, reverse, events, masses, orders, *, r, q):
    if (type(r) is not int or type(q) is not int or min(r, q) < 1
            or not events or len(events) != len(masses) or len(events) != len(orders)
            or any(type(n) is not int or n < 0 or n % 2 for n in orders)):
        raise ValueError('Invalid order/trajectory scope')
    masses = [b.rational(p) for p in masses]
    if any(p < 0 for p in masses) or sum(masses) != 1:
        raise ValueError('Exact normalized grid law required')
    active = [(event, p) for event, p in zip(events, masses, strict=True) if p > 0]
    averaged = Growth(b.upper(sum((b.real(p)*nonnegative(e.gain) for e, p in active), arb(0))),
                      b.upper(sum((b.real(p)*nonnegative(e.additive) for e, p in active), arb(0))))
    worst = Growth(max(nonnegative(e.gain) for e, _ in active), max(nonnegative(e.additive) for e, _ in active))
    def chain(event):
        return power(compose(reverse, compose(power(event, r), forward)), q)
    return chain(averaged), chain(worst)


def signal_error(norm_squared, growth, kernel):
    n2 = interval(norm_squared)
    if not n2 > 0:
        raise ValueError('Saved state norm not positive')
    psi = b.upper(n2.sqrt())
    gain, additive = nonnegative(growth.gain), nonnegative(growth.additive)
    state_error = (gain-1)*psi+additive
    norm_final = gain*psi+additive
    dot_error = nonnegative(kernel.dot_rel)*psi*norm_final+nonnegative(kernel.dot_abs)
    error = b.upper(abs(n2-1)+psi*state_error+dot_error)
    magnitude = b.upper(psi*norm_final+dot_error)
    return error, magnitude


def guard_completion(magnitude, *, abs_evaluation_error=None, finite_intermediates=False):
    """Sufficient all-support test. A failed bound is not an abort witness."""
    if abs_evaluation_error is None or finite_intermediates is not True:
        return False
    magnitude = nonnegative(magnitude)
    # SOURCE computes these sums in binary64, not exact decimal arithmetic.
    coordinate_limit = b.real(1.+1e-12)
    magnitude_limit = b.real(1.+1e-10)
    return bool(magnitude <= coordinate_limit and
                b.upper(magnitude+nonnegative(abs_evaluation_error)) <= magnitude_limit)


def classify_axes(logical, saved_means, saved_shots, saved_epsilon, error, *, completion, arithmetic_resolved):
    outputs = {}
    for axis, coord in [('cosine', 'real'), ('sine', 'imag')]:
        out = {'U_virtual_IID_upper': None, 'U_exec_upper': None, 'decision': 'UNDETERMINED',
               'fixed_PCG64_IID_certified': False}
        if error is not None and completion is True and arithmetic_resolved is True:
            logical_upper = b.upper(abs(interval(logical[coord])-b.real(saved_means[coord])))
            total = b.upper(logical_upper+nonnegative(error)+b.real(Q(3, 1 << 53)))
            out.update(U_virtual_IID_upper=str(b.dyadic(total)),
                       decision='SUFFICIENT_CONDITION_CONFIRMED' if b.upper(total+interval(saved_shots[axis])) < b.real(saved_epsilon[axis]).lower()
                       else 'NOT_CERTIFIED_BY_THIS_BOUND')
            # U_exec remains null for a concrete fixed-PCG execution.
        outputs[axis] = out
    return outputs


def installed_kernel_contract():
    """No coefficient is inferred from a version string or a toy-test PASS."""
    return {'status': 'UNRESOLVED', 'bounds': None, 'abs_evaluation_error': None,
            'finite_intermediates_proved': False,
            'reason': 'Installed CBLAS GEMV/DOT arithmetic-depth and underflow contract is not established'}


def bound_context(export, numeric, forward, reverse, saved_halves, angles, trig64,
                  masses, orders, *, r, q, norm_squared, kernel):
    """One residual/majorant pass; no trajectory or new logical mean evaluation."""
    if len(angles) != len(orders) or len(trig64) != len(orders):
        raise ValueError('Angle/order inventory mismatch')
    components = component_bounds(export, numeric)
    if not components:
        raise ValueError('Empty component inventory')
    residual = max(value['residual'] for value in components.values())
    frobenius = max(value['frobenius'] for value in components.values())
    product = action_growth(residual, frobenius, kernel)
    events = [compose(rotation_growth(angle, trig[0], trig[1], residual, frobenius, kernel),
                      power(product, order)) for angle, trig, order in zip(angles, trig64, orders, strict=True)]
    halves = []
    for value, name in [(forward, 'forward'), (reverse, 'reverse')]:
        delta = matrix_residual(value, decode_matrix_ball(saved_halves[name]))
        halves.append(action_growth(delta, b.frobenius(exact_matrix(value)), kernel))
    averaged, worst = trajectory_growth(*halves, events, masses, orders, r=r, q=q)
    average_error, _ = signal_error(norm_squared, averaged, kernel)
    uniform_error, uniform_magnitude = signal_error(norm_squared, worst, kernel)
    return {'component_bounds': components, 'average_error': average_error,
            'uniform_error': uniform_error, 'uniform_magnitude': uniform_magnitude,
            'average_growth': averaged, 'uniform_growth': worst}
