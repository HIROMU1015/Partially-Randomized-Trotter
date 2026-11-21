"""MSE bounds from finite adjacent-support checks of the fixed Z1+T1 decoder."""

from fractions import Fraction
from itertools import product
import math

import numpy as np


SCHEMA = 'RPE_FINITE_SUPPORT_MSE_V1'
MAX_SHOTS_PER_AXIS = 2048
MAX_ROUNDS = 14
MAX_AXIS_INTEGER_BITS = 1 << 20
MAX_ROUND_CELLS = 16384
MAX_ROUND_PAIRS = 1 << 22
TAIL_PER_AXIS = Fraction(1, 1 << 48)
REFINEMENT_MASS_BITS = 80


def _encode(value):
    return dict(numerator=value.numerator, denominator=value.denominator)


def _decode(value):
    if (not isinstance(value, dict) or set(value) != {'numerator', 'denominator'}
            or any(type(v) is not int for v in value.values()) or value['denominator'] <= 0):
        raise ValueError('Expected an exact rational')
    return Fraction(value['numerator'], value['denominator'])


def supports(shots):
    return (1 <= len(shots) <= MAX_ROUNDS
            and all(type(row[axis]) is int and 1 <= row[axis] <= MAX_SHOTS_PER_AXIS
                    for row in shots for axis in ('N_X', 'N_Y')))


def _window(axis):
    # Order by exact binomial mass, breaking equal weights by count.
    order = sorted(range(len(axis['counts'])),
                   key=lambda i: (-axis['numerators'][i], axis['counts'][i]))
    numerator = 0
    for length, i in enumerate(order, 1):
        numerator += axis['numerators'][i]
        if (axis['denominator']-numerator)*TAIL_PER_AXIS.denominator <= axis['denominator']:
            return order[:length]
    return order


def _count_support(row, risk, probability_rule, maximum_cells):
    axes = []
    for name, component in (('N_X', 'real'), ('N_Y', 'imag')):
        raw, shots = row['raw'][component], row[name]
        p = risk._axis_probability(raw, probability_rule)
        if (p.denominator.bit_length()-1)*shots > MAX_AXIS_INTEGER_BITS:
            raise ValueError('Binomial axis exceeds the declared integer-size scope')
        axes.append(risk.binomial_axis(raw, shots, probability_rule=probability_rule))
    xs, ys = map(_window, axes)
    xaxis, yaxis = axes
    while len(xs)*len(ys) > maximum_cells:
        xmass = xaxis['numerators'][xs[-1]] * yaxis['denominator']
        ymass = yaxis['numerators'][ys[-1]] * xaxis['denominator']
        if len(ys) == 1 or (len(xs) > 1 and xmass <= ymass):
            xs.pop()
        else:
            ys.pop()
    xx = [risk.S.count_mean(xaxis['counts'][i], row['N_X'])['mean'] for i in xs]
    yy = [risk.S.count_mean(yaxis['counts'][i], row['N_Y'])['mean'] for i in ys]
    # Match the frozen atan2 and wrap operation order, including signed zeros.
    xy = list(product(xx, yy))
    phases = risk.F._wrap(np.fromiter((math.atan2(y, x) for x, y in xy), float, len(xy)))
    zeros = np.fromiter((x == 0 and y == 0 for x, y in xy), bool, len(xy))
    xw = [xaxis['numerators'][i] for i in xs]
    yw = [yaxis['numerators'][i] for i in ys]
    denominator = xaxis['denominator']*yaxis['denominator']
    return phases, zeros, xw, yw, denominator


def _closure(decoder, phases, zeros, previous, q):
    if q == 1:
        return decoder.initial(phases, zeros), np.ones(len(phases), dtype=bool)
    result, _ = decoder.step(phases, np.full(len(phases), previous[0]), q, zeros)
    keep = np.ones(len(phases), dtype=bool)
    # Every retained current cell must have one output across the whole
    # retained previous support. The vector length changes memory, not scope.
    block = max(1, min(len(phases), 32768//len(previous)))
    for begin in range(0, len(phases), block):
        s = slice(begin, begin+block)
        xy = np.broadcast_to(phases[s, None], (len(phases[s]), len(previous)))
        values, _ = decoder.step(xy, np.broadcast_to(previous, xy.shape), q, zeros[s, None])
        keep[s] = np.all(values.view(np.uint64) == result[s, None].view(np.uint64), axis=1)
    return result, keep


def summarize(certificate, threshold, *, expected_rounds):
    """Recompute the result and account for each omitted history exactly once."""
    if (certificate['schema'] != SCHEMA or not 1 <= expected_rounds <= MAX_ROUNDS
            or len(certificate['rounds']) != expected_rounds):
        raise ValueError('Unexpected finite-support evidence scope')
    cap = _decode(certificate['loss_cap'])
    terminal = _decode(certificate['terminal_loss'])
    prefix = Fraction(1)
    residual = Fraction(0)
    for m, row in enumerate(certificate['rounds']):
        if row['m'] != m:
            raise ValueError('Expected consecutive finite-support rounds')
        kept, represented = (_decode(row[k]) for k in ('retained_probability','window_probability'))
        if not 0 <= kept <= represented <= 1:
            raise ValueError('Invalid retained mass')
        residual += prefix*(1-kept)
        if m < expected_rounds-1:
            prefix *= kept
    if cap < 0 or not 0 <= terminal <= cap*kept:
        raise ValueError('Invalid finite-support terminal loss')
    lower = prefix*terminal
    upper = min(cap, lower + residual*cap)
    status = ('MODEL_BOUND_MEETS_ADOPTED_THRESHOLD' if upper <= threshold else
              'MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD' if lower > threshold else 'INCONCLUSIVE')
    return dict(status=status, MSE_lower=_encode(lower), MSE_upper=_encode(upper),
                residual_probability=_encode(residual))


def evaluate(rows, *, h, target, probability_rule, threshold, risk, total,
             maximum_round_cells=MAX_ROUND_CELLS, maximum_round_pairs=MAX_ROUND_PAIRS):
    """Use pinned risk helpers while leaving probability outside the proof in the MSE."""
    if not supports(rows) or not math.isfinite(h) or h <= 0 or not math.isfinite(target):
        raise ValueError('Invalid finite-support evaluation inputs')
    if maximum_round_cells < 1 or maximum_round_pairs < maximum_round_cells or threshold < 0:
        raise ValueError('Invalid finite-support resource scope or threshold')
    cap, _ = total.loss_range(h, target)
    if cap is None:
        raise ValueError('Fixed decoder energy range is unavailable')
    decoder = total.T
    previous = np.array([], dtype=np.float64)
    evidence = []
    terminal = Fraction(0)
    for m, row in enumerate(rows):
        if row['m'] != m:
            raise ValueError('Expected consecutive fixed input rounds')
        if m and not len(previous):
            evidence.append(dict(m=m, retained_probability=_encode(Fraction()), window_probability=_encode(Fraction()),
                                 checked_pairs=0, represented_cells=0))
            continue
        previous_count = max(1, len(previous))
        limit = min(maximum_round_cells, maximum_round_pairs//previous_count)
        phases, zeros, xw, yw, denominator = _count_support(row, risk, probability_rule, limit)
        values, keep = _closure(decoder, phases, zeros, previous, 1 << m)
        admitted = values[keep]
        if not np.all(np.isfinite(admitted)) or np.any(np.abs(admitted) > math.pi):
            raise ValueError('Fixed decoder exceeded its declared phase range')
        previous = np.unique(admitted.view(np.uint64)).view(np.float64)
        retained = 0
        moment = Fraction(0)
        for start, weight in enumerate(xw):
            left = start*len(yw)
            valid = np.flatnonzero(keep[left:left+len(yw)])
            retained += weight * sum(yw[i] for i in valid)
            if m == len(rows)-1:
                moment += weight * sum((yw[i]*(Fraction(-float(values[left+i])/h) - Fraction(target))**2
                                        for i in valid), Fraction(0))
        evidence.append(dict(m=m, retained_probability=_encode(Fraction(retained, denominator)),
                             window_probability=_encode(Fraction(sum(xw)*sum(yw), denominator)),
                             checked_pairs=len(phases)*previous_count,
                             represented_cells=len(phases)))
        terminal = moment/denominator
    certificate = dict(schema=SCHEMA, rounds=evidence, loss_cap=_encode(cap), terminal_loss=_encode(terminal))
    return dict(summarize(certificate, threshold, expected_rounds=len(rows)), certificate=certificate)


def refine(rows, *, h, target, probability_rule, threshold, risk, total,
           maximum_round_cells=MAX_ROUND_CELLS, maximum_round_pairs=MAX_ROUND_PAIRS):
    """Propagate lower masses through exact evidence from the fixed decoder.

    Each mass is rounded down to a multiple of 2**-80. Loss of mass from
    rounding, windows and the transition budget enters the same residual.
    """
    if (not supports(rows) or not math.isfinite(h) or h <= 0 or not math.isfinite(target)
            or maximum_round_cells < 1 or maximum_round_pairs < maximum_round_cells):
        raise ValueError('Invalid finite-support refinement inputs')
    cap, _ = total.loss_range(h, target)
    if cap is None:
        raise ValueError('Fixed decoder energy range is unavailable')
    decoder = total.T
    unit = 1 << REFINEMENT_MASS_BITS
    states = {0: unit}
    terminal = Fraction(0)
    evidence = []
    for m, row in enumerate(rows):
        if row['m'] != m:
            raise ValueError('Expected consecutive fixed input rounds')
        before = sum(states.values())
        if not before:
            evidence.append(dict(m=m, retained_probability=_encode(Fraction()),
                                 window_probability=_encode(Fraction()), checked_pairs=0, represented_cells=0))
            continue
        phases, zeros, xw, yw, denominator = _count_support(row, risk, probability_rule, maximum_round_cells)
        weights = [(x*y << REFINEMENT_MASS_BITS)//denominator for x, y in product(xw, yw)]
        selected = sorted(states, key=lambda state: (-states[state], state))[:maximum_round_pairs//len(phases)]
        next_states = {}
        for state in selected:
            if m == 0:
                values = decoder.initial(phases, zeros)
            else:
                previous = np.array([state], dtype=np.uint64).view(np.float64)[0]
                values, _ = decoder.step(phases, np.full(len(phases), previous), 1 << m, zeros)
            if not np.all(np.isfinite(values)) or np.any(np.abs(values) > math.pi):
                raise ValueError('Fixed decoder exceeded its declared phase range')
            mass = states[state]
            for bits, weight in zip(values.view(np.uint64).tolist(), weights):
                contribution = (mass*weight) >> REFINEMENT_MASS_BITS
                if contribution:
                    next_states[bits] = next_states.get(bits, 0) + contribution
        if m == len(rows)-1 and next_states:
            positions = list(next_states)
            estimates = np.array(positions,dtype=np.uint64).view(np.float64)
            estimate = Fraction(target)
            terminal = sum((next_states[bits]*(Fraction(-float(value)/h)-estimate)**2
                            for bits, value in zip(positions, estimates)), Fraction(0))/before
        represented = Fraction(sum(xw)*sum(yw), denominator)
        evidence.append(dict(m=m, retained_probability=_encode(Fraction(sum(next_states.values()), before)),
                             window_probability=_encode(represented),
                             checked_pairs=len(selected)*len(phases), represented_cells=len(phases)))
        states = next_states
    certificate = dict(schema=SCHEMA, rounds=evidence, loss_cap=_encode(cap), terminal_loss=_encode(terminal),
                       refinement='RETAINED_STATE_DP', mass_grid_bits=REFINEMENT_MASS_BITS)
    return dict(summarize(certificate, threshold, expected_rounds=len(rows)), certificate=certificate)


def combine(certificates, threshold, *, expected_rounds):
    reports = [summarize(certificate, threshold, expected_rounds=expected_rounds)
               for certificate in certificates]
    lower = max(_decode(row['MSE_lower']) for row in reports)
    upper = min(_decode(row['MSE_upper']) for row in reports)
    if lower > upper:
        raise ValueError('Finite-support refinement contradicts the previous bounds')
    status = ('MODEL_BOUND_MEETS_ADOPTED_THRESHOLD' if upper <= threshold else
              'MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD' if lower > threshold else 'INCONCLUSIVE')
    return dict(status=status,MSE_lower=_encode(lower), MSE_upper=_encode(upper),
                residual_probability=_encode(min(_decode(r['residual_probability']) for r in reports)))
