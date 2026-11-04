"""Cached ideal categorical enclosures. No real-data loader or execution at import."""
from __future__ import annotations

from fractions import Fraction as Q
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'docs/research/rpe_m17_rigorous_mean_enclosure_preparation_20261103'))
import ball_adapter as b
import mean_adapter as mean
from flint import acb, acb_mat, arb


def decode(saved):
    """Same recentering as virtual_bridge_adapter.decode_matrix_ball; no bridge import."""
    rows, cols = saved['shape']
    if (type(rows) is not int or type(cols) is not int or min(rows, cols) < 1
            or rows != cols or len(saved['center']) != rows*cols):
        raise ValueError('UNKNOWN/STOP: invalid saved MatrixBall')
    matrix = acb_mat([[b.zball(saved['center'][i*cols+j]) for j in range(cols)] for i in range(rows)])
    # Retain saved radius AND uncertainty introduced when decoding at this precision.
    return b.recenter(matrix, b.real(saved['spectral_radius_upper']), b.real(saved['target_norm_upper']))


def interval(saved):
    lo, hi = Q(saved['lower']), Q(saved['upper'])
    if lo > hi:
        raise ValueError('UNKNOWN/STOP: inverted saved interval')
    return arb(b.real((lo+hi)/2), b.upper(b.real((hi-lo)/2)))


def expectation(operator, state_pairs, saved_norm2):
    """Reuse the certified norm² of these exact bytes; never recompute psi†psi."""
    if operator.center.nrows() != len(state_pairs) or operator.center.ncols() != len(state_pairs):
        raise ValueError('UNKNOWN/STOP: state dimension')
    norm2 = interval(saved_norm2)
    if not norm2 > 0:
        raise ValueError('UNKNOWN/STOP: norm² not positive')
    psi = acb_mat([[b.zball(pair)] for pair in state_pairs])
    value = (b.dagger(psi)*operator.center*psi)[0, 0]/norm2
    # True normalized vectors give transfer <= spectral radius, once only.
    value += acb(arb(0, operator.radius), arb(0, operator.radius))
    if not value.is_finite():
        raise ValueError('UNKNOWN/STOP: nonfinite expectation')
    return value


def replay_order_law(tau64, cutoff):
    """New SOURCE replay, not recovery of historical probabilities/CDF/theta."""
    if type(tau64) is not float or not math.isfinite(tau64):
        raise ValueError('UNKNOWN/STOP: finite binary64 tau required')
    sys.path.insert(0, str(ROOT/'src/trotterlib'))
    import rte
    if Path(rte.__file__).resolve() != ROOT/'src/trotterlib/rte.py':
        raise ValueError('UNKNOWN/STOP: wrong finite distribution SOURCE')
    distribution = rte.finite_rte_distribution(tau64, cutoff)
    orders = list(distribution.orders)
    angles = [math.atan(tau64/(n+1)) for n in orders]  # existing B-1 replay convention
    if any(not math.isfinite(2*x) or Q.from_float(2*x) != 2*Q.from_float(x) for x in angles):
        raise ValueError('UNKNOWN/STOP: rotation doubling not exact')
    law = mean.categorical(list(distribution.order_probabilities))
    return orders, angles, law


def make_step(a, forward, reverse, *, r, orders, angles, law):
    if type(r) is not int or r < 1:
        raise ValueError('UNKNOWN/STOP: positive integer r required')
    event = mean.event_average(a, orders, [Q(p) for p in law['grid_masses']], angles)
    # Column-vector action is F, r events, R. No extra B or attenuation division.
    return b.product(b.product(reverse, b.power(event, r)), forward)


def evaluate_round(step, state, norm2, row):
    if (row['ld'] != 3 or row['K'] != 2 or row['m'] == 17
            or not 0 <= row['m'] < 17 or row['q'] != 1 << row['m']):
        raise ValueError('UNKNOWN/STOP: fixed Partial K2 m0..16 only; m17 is reused')
    powered = b.power(step, row['q'])
    value = expectation(powered, state, norm2)
    logical = {'real': b.endpoints(value.real), 'imag': b.endpoints(value.imag)}
    phase = mean.builder_phase_bound(row['q'], row['r'])  # justified only for K2
    value += acb(arb(0, phase), arb(0, phase))
    axes = {}
    for axis, coordinate in [('cosine', 'real'), ('sine', 'imag')]:
        fixed = row['axes'][axis]
        axes[axis] = b.classify_interval(getattr(value, coordinate), float(row['mu0'][coordinate]),
            int(fixed['N']), fixed['alpha'], fixed['epsilon'], bridge=None)
    return {'m': row['m'], 'q': row['q'], 'r': row['r'], 'K': row['K'], 'axes': axes,
            'logical_signed_mean_interval': logical, 'builder_phase_transfer_upper': str(b.dyadic(phase)),
            'powered_radius_upper': str(b.dyadic(powered.radius)), 'norm2_source': 'saved m17 96-bit interval'}


def aggregate(rows):
    if len(rows) != 18 or sorted(x['m'] for x in rows) != list(range(18)):
        raise ValueError('UNKNOWN/STOP: all 18 unique rounds required')
    decisions = [row['axes'][axis]['categorical_decision'] for row in rows for axis in ('cosine', 'sine')]
    return ('CONDITIONAL_IDEAL_CATEGORICAL_LOCAL_CONDITIONS_CONFIRMED'
            if all(x == 'SUFFICIENT_CONDITION_CONFIRMED' for x in decisions) else 'UNDETERMINED')
