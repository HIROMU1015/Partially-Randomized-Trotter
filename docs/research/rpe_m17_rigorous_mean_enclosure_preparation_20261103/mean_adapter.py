"""Specified IID categorical law + exact-parameter ideal gates. No real loader."""
from __future__ import annotations

from fractions import Fraction as Q
import math

import numpy as np
from flint import arb, acb, acb_mat

import ball_adapter as b


def categorical(probabilities, *, component=False):
    """NumPy1.26.4 choice CDF, without RNG. Grid masses are exact rationals."""
    if not probabilities or any(type(p) is not float or not math.isfinite(p) or p < 0 for p in probabilities):
        raise ValueError("Finite nonnegative binary64 probabilities required")
    total = math.fsum(probabilities)
    if not math.isfinite(total) or not math.isclose(total, 1., rel_tol=0., abs_tol=1e-12):
        raise ValueError("Fixed sampler probabilities do not sum near one")
    used = [p/total for p in probabilities] if component else probabilities.copy()
    cdf = np.asarray(used, dtype=np.float64).cumsum()
    if not math.isfinite(float(cdf[-1])) or cdf[-1] <= 0:
        raise ValueError("Invalid cumulative normalization")
    cdf /= cdf[-1]
    if (not np.isfinite(cdf).all() or cdf[-1] != 1. or np.any(np.diff(cdf) < 0)
            or np.any(cdf < 0) or np.any(cdf > 1)):
        raise ValueError("Invalid monotone categorical CDF")
    previous, masses = Q(0), []
    for value in cdf:
        scaled = Q.from_float(float(value))*(1 << 53)
        ceiling = -(-scaled.numerator//scaled.denominator)
        cumulative = Q(ceiling, 1 << 53)
        masses.append(cumulative-previous)
        previous = cumulative
    if sum(masses) != 1 or any(p < 0 for p in masses):
        raise ValueError("Invalid exact categorical mass")
    return {"input_probability64_hex": [p.hex() for p in probabilities],
            "component_fsum64_hex": total.hex() if component else None,
            "choice_input64_hex": [p.hex() for p in used],
            "cdf64_hex": [float(c).hex() for c in cdf],
            "grid_masses": [str(p) for p in masses], "grid_bits": 53,
            "assumption": "independent uniform 53-bit words; not a statement about fixed PCG64 streams"}


def sector_indices(n, electrons):
    if type(n) is not int or type(electrons) is not int or n < 1 or not 0 <= electrons <= n:
        raise ValueError("Invalid particle-number sector")
    return [i for i in range(1 << n) if i.bit_count() == electrons]


def validate_sector(ordered, n, electrons):
    if len(ordered) != len(set(ordered)) or sorted(ordered) != sector_indices(n, electrons):
        raise ValueError("Complete particle-number sector required")


def gate_matrix(operation):
    """Analytic Qiskit1.3.0 RZ and XXPlusYY parameter conventions."""
    name, params = operation['name'], operation['params']
    if name == 'rz' and len(params) == 1 and len(operation['qubits']) == 1:
        t = b.real(params[0])/2
        return acb_mat([[acb(0, -t).exp(), 0], [0, acb(0, t).exp()]])
    if name == 'xx_plus_yy' and len(params) == 2 and len(operation['qubits']) == 2:
        t, beta = b.real(params[0])/2, b.real(params[1])
        cosine, sine = t.cos(), t.sin()
        return acb_mat([[1,0,0,0], [0,cosine,-acb(0,1)*sine*acb(0,-beta).exp(),0],
                        [0,-acb(0,1)*sine*acb(0,beta).exp(),cosine,0], [0,0,0,1]])
    raise ValueError("Unsupported ideal basis gate or parameter count")


def basis_matrix(spec, ordered, n, electrons):
    """Sparse row application. Supported gates preserve number exactly by formula."""
    validate_sector(ordered, n, electrons)
    positions, dimension = {s: i for i, s in enumerate(ordered)}, len(ordered)
    out = b.identity(dimension)
    for operation in spec['operations']:
        wires = operation['qubits']
        if len(set(wires)) != len(wires) or any(type(q) is not int or not 0 <= q < n for q in wires):
            raise ValueError("Invalid basis wires")
        local = gate_matrix(operation)
        if local.nrows() != 1 << len(wires):
            raise ValueError("Local gate arity mismatch")
        updated = acb_mat(dimension, dimension)
        mask = sum(1 << q for q in wires)
        for source_index, state in enumerate(ordered):
            local_source = sum(((state >> q) & 1) << k for k, q in enumerate(wires))
            for local_dest in range(1 << len(wires)):
                value = local[local_dest, local_source]
                if value.is_zero():
                    continue
                target_state = (state & ~mask) | sum(((local_dest >> k) & 1) << q for k, q in enumerate(wires))
                if target_state not in positions:
                    raise ValueError("Basis gate does not preserve the declared sector")
                target_index = positions[target_state]
                for column in range(dimension):
                    updated[target_index, column] += value*out[source_index, column]
        # All interval uncertainty stays here until one rigorous recenter per basis.
        out = updated
    return b.unit(out)


def diagonal(values, norm):
    matrix = acb_mat(len(values), len(values))
    for i, value in enumerate(values):
        matrix[i,i] = value
    return b.recenter(matrix, norm=norm)


def component_average(components, masses, bases, ordered):
    if len(components) != len(masses) or sum(masses) != 1 or any(p < 0 for p in masses):
        raise ValueError("Normalized fixed component mass required")
    groups, group_mass = {}, {}
    seen = set()
    for item, p in zip(components, masses, strict=True):
        if item['id'] in seen or item['sign'] not in (-1, 1) or item['basis_id'] not in bases:
            raise ValueError("Invalid component identity/sign/basis")
        seen.add(item['id'])
        support = item['support']
        if len(set(support)) != len(support) or any(type(q) is not int or q < 0 for q in support):
            raise ValueError("Invalid component support")
        values = groups.setdefault(item['basis_id'], [Q(0)]*len(ordered))
        group_mass[item['basis_id']] = group_mass.get(item['basis_id'], Q(0))+p
        for i, state in enumerate(ordered):
            parity = sum((state >> q) & 1 for q in support) % 2
            values[i] += p*item['sign']*(-1)**parity
    terms = []
    for key in sorted(groups):
        u = bases[key]
        d = diagonal([b.real(v) for v in groups[key]], group_mass[key])
        terms.append((1, b.product(b.product(u, d), b.adjoint(u))))
    return b.weighted_sum(terms, norm=1)


def primitive_block(block, basis, ordered):
    values = []
    p = block['primitives']
    for state in ordered:
        phase = b.real(p['global_phase'])
        z = lambda q: 1-2*((state >> q) & 1)
        for q, angle in p['rz']:
            phase -= b.real(angle)*z(q)/2
        for left, right, angle in p['rzz']:
            phase -= b.real(angle)*z(left)*z(right)/2
        values.append(acb(0, phase).exp())
    return b.product(b.product(basis, diagonal(values, 1)), b.adjoint(basis))


def half_operators(export, bases):
    dimension = len(export['ordered_sector'])
    phase = acb(1)
    # Source builder _append_step_phases applies these independently.
    for angle in export['step_phase_angles']:
        phase *= acb(0, b.real(angle)).exp()
    forward = b.unit(phase*b.identity(dimension))
    reverse = b.unit(b.identity(dimension))
    blocks = [primitive_block(block, bases[block['basis_id']], export['ordered_sector']) for block in export['deterministic_blocks']]
    for block in blocks:
        forward = b.product(block, forward)
    for block in reversed(blocks):
        reverse = b.product(block, reverse)
    return forward, reverse


def event_average(a, orders, probabilities, angles):
    if (len(orders) != len(probabilities) or len(orders) != len(angles)
            or any(type(n) is not int or n < 0 or n % 2 for n in orders)
            or len(set(orders)) != len(orders) or sum(probabilities) != 1 or any(p < 0 for p in probabilities)):
        raise ValueError("Invalid normalized finite event law")
    one = b.unit(b.identity(a.center.nrows()))
    terms = []
    for n, probability, angle in zip(orders, probabilities, angles, strict=True):
        theta = b.real(angle)
        rotation = b.weighted_sum([(theta.cos(), one), (-acb(0,1)*theta.sin(), a)], norm=1)
        terms.append((probability*(-1)**(n//2), b.product(rotation, b.power(a, n))))
    return b.weighted_sum(terms, norm=1)


def enclose(export, component_law, order_law, *, q, r, orders, angles):
    validate_sector(export['ordered_sector'], export['n_qubits'], export['n_electrons'])
    if any(any(qubit >= export['n_qubits'] for qubit in c['support']) for c in export['components']):
        raise ValueError("Component outside register")
    bases = {key: basis_matrix(spec, export['ordered_sector'], export['n_qubits'], export['n_electrons'])
             for key, spec in export['bases'].items()}
    a = component_average(export['components'], [Q(p) for p in component_law['grid_masses']], bases, export['ordered_sector'])
    event = event_average(a, orders, [Q(p) for p in order_law['grid_masses']], angles)
    tail = b.power(event, r)
    forward, reverse = half_operators(export, bases)
    step = b.product(b.product(reverse, tail), forward)
    powered = b.power(step, q)
    value, state_info = b.expectation(powered, export['state'])
    return value, {'A': a, 'event': event, 'tail': tail, 'forward': forward,
                   'reverse': reverse, 'step': step, 'powered': powered}, state_info


def builder_phase_bound(q, r):
    """K2, extracted identity: at most 3r positive pi64 additions per tail.

    The fixed builder first accumulates event phases, then negative-product
    signs. Standard binary64 additions of these nonnegative normal terms have
    relative error <= gamma_(3r); no cancellation/underflow. Exact zero adds
    do not enlarge the error. q unitary steps use telescoping contraction.
    """
    if type(q) is not int or type(r) is not int or q < 1 or r < 1:
        raise ValueError('Positive fixed q/r required')
    count, u = 3*r, b.real(Q(1, 1 << 53))
    if not count*u < 1:
        raise ValueError('Binary64 phase summation bound unavailable')
    pi64 = b.real(math.pi)
    gamma = count*u/(1-count*u)
    per_tail = b.upper(count*abs(pi64-arb.pi())+gamma*count*abs(pi64))
    return b.upper(q*per_tail)
