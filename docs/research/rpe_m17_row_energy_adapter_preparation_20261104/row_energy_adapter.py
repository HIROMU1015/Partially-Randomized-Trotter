"""Conditional finite-domain artificial proof adapter; no loader, RNG or CLI.

Enumerates every declared input state/primitive, not sampled trajectories. A
missing closure edge is UNKNOWN. No result here is an H4 certificate. Binary,
dispatch/partition determinism and all-thread FP stability are theorem assumptions supplied by the
caller; this module does not perform the live runtime observer's job.
"""
from dataclasses import dataclass
from fractions import Fraction as Q
from pathlib import Path
import math
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'docs/research/rpe_m17_guard_proof_refinement_preflight_20261104'))
sys.path.insert(0, str(ROOT/'docs/research/rpe_m17_virtual_bridge_scientific_runner_preparation_20261104'))
import energy_prototype as energy
import virtual_bridge_adapter as bridge
from runtime_checks import validate_array

k = energy.kernel
b = bridge.b
ZERO = (Q(0), Q(0))


def z(value):
    return Q.from_float(float(value.real)), Q.from_float(float(value.imag))


def add(a, c): return a[0]+c[0], a[1]+c[1]
def neg(a): return -a[0], -a[1]
def mul(a, c): return a[0]*c[0]-a[1]*c[1], a[0]*c[1]+a[1]*c[0]
def conj(a): return a[0], -a[1]
def vec(value): return [z(item) for item in value]
def norm2(value): return sum((re*re+im*im for re, im in value), Q(0))
def inner(a, c): return sum_complex(mul(conj(x), y) for x, y in zip(a, c, strict=True))
def sum_complex(items):
    result = ZERO
    for item in items: result = add(result, item)
    return result


def sqrt_upper(value):
    return b.dyadic(b.upper(b.real(value).sqrt()))


def exact_action(matrix, vector):
    return [sum_complex(mul(z(a), x) for a, x in zip(row, vector, strict=True)) for row in matrix]


def energy_record(input_exact, target, output):
    observed = vec(output)
    error = [add(y, neg(t)) for y, t in zip(observed, target, strict=True)]
    n, drift = norm2(input_exact), norm2(target)-norm2(input_exact)
    radial, square = inner(target, error)[0], norm2(error)
    final = energy.energy_step(n, drift, radial, square)
    if final != norm2(observed):
        raise ValueError('Signed energy identity failed')
    return dict(input_energy=n, drift=drift, radial=radial, error_squared=square,
                output_energy=final)


def _row_proof(coefficients, vector, observed, exact):
    sums_re = sum((abs(a[0]*x[0])+abs(a[1]*x[1]) for a, x in zip(coefficients, vector, strict=True)), Q(0))
    sums_im = sum((abs(a[0]*x[1])+abs(a[1]*x[0]) for a, x in zip(coefficients, vector, strict=True)), Q(0))
    bounds = energy.real_row_error(sums_re, len(vector)), energy.real_row_error(sums_im, len(vector))
    error = add(observed, neg(exact))
    if any(abs(e) > bound for e, bound in zip(error, bounds, strict=True)):
        raise ValueError('Observed real-row residual outside forest bound')
    nonzero = [a for a in coefficients if a != ZERO]
    structural = (not nonzero or len(nonzero) == 1 and nonzero[0] in
                  [(Q(1), Q(0)), (Q(-1), Q(0)), (Q(0), Q(1)), (Q(0), Q(-1))])
    if structural and error != ZERO:
        raise ValueError('Exact zero/sign/swap row claim failed')
    return dict(error=error, forest_bounds=bounds, absolute_product_sums=(sums_re, sums_im),
                structurally_exact=structural)


def gemv(matrix, state):
    """Exact dyadic residual and energy for ONE declared artificial input."""
    n = len(state)
    validate_array(matrix, (n, n)); validate_array(state, (n,))
    k.path_bounds(n, n)  # Inspected shape family only.
    x, target = vec(state), exact_action(matrix, vec(state))
    output = matrix @ state
    validate_array(output, (n,))
    rows = [_row_proof([z(a) for a in row], x, z(y), t)
            for row, y, t in zip(matrix, output, target, strict=True)]
    return dict(output=output, target=target, rows=rows, energy=energy_record(x, target, output),
                prefix_energy=max(norm2(x), norm2(vec(output))))


def _arithmetic_record(output, target, norm_bound):
    error = [add(y, neg(t)) for y, t in zip(vec(output), target, strict=True)]
    if norm2(error) > norm_bound**2:
        raise ValueError('Scale/subtract residual outside declared bound')
    return dict(error=error, error_squared=norm2(error), norm_bound=norm_bound)


def rotation(matrix, state, cosine64, sine64):
    """Same c*v - (i*s)*(P@v) graph, retaining every branch's roundoff."""
    if any(type(t) is not float or not math.isfinite(t) for t in [cosine64, sine64]):
        raise ValueError('Finite binary64 trig outputs required')
    proof = gemv(matrix, state)
    pv = proof['output']
    c, s = Q.from_float(cosine64), Q.from_float(sine64)
    x, px = vec(state), proof['target']
    bounds = k.path_bounds(len(state), len(state))
    first, second = cosine64*state, (1j*sine64)*pv
    output = first-second
    for item in [first, second, output]: validate_array(item, state.shape)
    first_target = [mul((c, Q(0)), value) for value in x]
    second_target = [mul((Q(0), s), value) for value in vec(pv)]
    first_bound = bounds.scale_rel*abs(c)*sqrt_upper(norm2(x))+bounds.scale_abs
    second_bound = bounds.scale_rel*abs(s)*sqrt_upper(norm2(vec(pv)))+bounds.scale_abs
    sub_target = [add(a, neg(t)) for a, t in zip(vec(first), vec(second), strict=True)]
    sub_bound = bounds.sub_rel*(sqrt_upper(norm2(vec(first)))+sqrt_upper(norm2(vec(second))))+bounds.sub_abs
    arithmetic = [_arithmetic_record(first, first_target, first_bound),
                  _arithmetic_record(second, second_target, second_bound),
                  _arithmetic_record(output, sub_target, sub_bound)]
    target = [add(mul((c, Q(0)), a), neg(mul((Q(0), s), t))) for a, t in zip(x, px, strict=True)]
    record = energy_record(x, target, output)
    skew = 2*c*s*inner(x, px)[1]
    expanded = (c*c+s*s-1)*norm2(x)+s*s*(norm2(px)-norm2(x))+skew
    if expanded != record['drift']:
        raise ValueError('Non-Hermitian rotation identity failed')
    return dict(output=output, energy=record, nonhermitian_term=skew, gemv=proof,
                arithmetic=arithmetic, prefix_energy=max(norm2(vec(a)) for a in [state, pv, first, second, output]))


@dataclass(frozen=True)
class Event:
    products: tuple
    rotation_id: str
    cosine64: float
    sine64: float
    phase: int
    mass: Q


def identity(energies):
    return [{i: value} for i, value in enumerate(energies)]


def compose(after, before):
    """Reachability plus maximum squared norm on EVERY prefix of each path."""
    result = []
    for starts in before:
        ends = {}
        for middle, peak_before in starts.items():
            for end, peak_after in after[middle].items():
                peak = max(peak_before, peak_after)
                ends[end] = max(ends.get(end, Q(0)), peak)
        result.append(ends)
    return result


def union(relations):
    result = [{} for _ in relations[0]]
    for relation in relations:
        for combined, row in zip(result, relation, strict=True):
            for end, peak in row.items(): combined[end] = max(combined.get(end, Q(0)), peak)
    return result


def power(relation, count, energies):
    if type(count) is not int or count < 0:
        raise ValueError('Nonnegative integer count required')
    result = identity(energies)
    while count:
        if count & 1: result = compose(relation, result)
        count >>= 1
        if count: relation = compose(relation, relation)
    return result


def _closed_relation(states, lookup, operation, evidence):
    relation = []
    for state in states:
        proof = operation(state)
        evidence.append(proof)
        end = lookup.get(proof['output'].tobytes())
        if end is None:
            raise ValueError('UNKNOWN_CLOSURE: output missing from declared finite domain')
        relation.append({end: proof['prefix_energy']})
    return relation


def _phase(state, phase):
    output = complex(phase)*state
    proof = energy_record(vec(state), [mul((Q(phase), Q(0)), a) for a in vec(state)], output)
    if proof['error_squared']:
        raise ValueError('Exact event phase claim failed')
    return dict(output=output, energy=proof, prefix_energy=proof['input_energy'])


def _dot(psi, state):
    target = inner(vec(psi), vec(state))
    output = complex(np.vdot(psi, state))
    if not math.isfinite(output.real) or not math.isfinite(output.imag):
        raise ValueError('Nonfinite dot')
    rows = _row_proof([conj(a) for a in vec(psi)], vec(state), z(output), target)
    return dict(output=output, rows=rows)


def analyze(states, psi, forward, reverse, components, events, *, r, q, runtime=None, action=None):
    """Complete declared artificial support, conditional on supplied FP facts.

    No implicit support truncation, tolerance merging, new input-state norm
    calculation for H4, or signed-zero equivalence. Domain membership is bytes.
    """
    result = dict(scope='ARTIFICIAL_FINITE_DOMAIN_ONLY', status='UNKNOWN_STOP', mandatory_STOP=True,
                  U_virtual_IID_upper=None, U_exec_upper=None, scientific_status='UNDETERMINED',
                  RPE_tolerance_certified=False, scientific_execution_ready=False,
                  conditional_guard_proved=False, evidence=[], runtime_live_observed=False)
    try:
        n = len(psi)
        if (not isinstance(runtime, dict) or runtime.get('stable_all_threads') is not True
                or runtime.get('deterministic_graph') is not True):
            raise ValueError('UNKNOWN_RUNTIME: stable FP and deterministic dispatch/partition conditions missing')
        contract = k.selected_path_contract(runtime.get('profile'), rows=n, columns=n, layout='C',
                                           mxcsr_values=runtime.get('mxcsr', []))
        if contract['bounds'] is None:
            raise ValueError('UNKNOWN_RUNTIME: binary/profile or FP mismatch')
        if any(type(count) is not int or count < 1 for count in [r, q]):
            raise ValueError('Positive r/q required')
        validate_array(psi, (n,))
        if not states or norm2(vec(psi)) <= 0:
            raise ValueError('Nonempty finite domain and nonzero raw state required')
        for state in states: validate_array(state, (n,))
        lookup = {state.tobytes(): i for i, state in enumerate(states)}
        if len(lookup) != len(states) or psi.tobytes() not in lookup:
            raise ValueError('Duplicate domain bytes or missing raw state')
        matrices = [forward, reverse, *components.values()]
        for matrix in matrices: validate_array(matrix, (n, n))
        masses = [energy.rational(event.mass) for event in events]
        if not events or any(mass < 0 for mass in masses) or sum(masses) != 1:
            raise ValueError('Exact normalized complete artificial event law required')
        active = [event for event, mass in zip(events, masses, strict=True) if mass > 0]
        for event in active:
            if (len(event.products) not in [0, 2] or type(event.phase) is not int
                    or event.phase != (-1)**(len(event.products)//2)
                    or event.rotation_id not in components or any(key not in components for key in event.products)
                    or any(type(t) is not float or not math.isfinite(t) for t in [event.cosine64, event.sine64])):
                raise ValueError('Invalid fixed K2 event graph')
        energies = [norm2(vec(state)) for state in states]
        matrix_norm = max(sqrt_upper(sum((norm2(vec(row)) for row in matrix), Q(0))) for matrix in matrices)
        scalars = max([Q(1)]+[abs(Q.from_float(t)) for event in active for t in [event.cosine64, event.sine64]])
        finite = k.finite_intermediate_envelope(matrix_frobenius=matrix_norm, initial_norm=sqrt_upper(norm2(vec(psi))),
                                               boundary_norm=sqrt_upper(max(energies)), scalar_modulus=scalars)
        if not finite: raise ValueError('UNKNOWN_FINITE: no-overflow envelope failed')
        result['finite_envelope_conditional'] = True
        result.update(declared_domain_size=len(states), active_event_count=len(active), q=q, r=r,
                      runtime_assumptions=runtime)
        evidence = result['evidence']
        f = _closed_relation(states, lookup, lambda x: gemv(forward, x), evidence)
        rev = _closed_relation(states, lookup, lambda x: gemv(reverse, x), evidence)
        cache = {key: _closed_relation(states, lookup, lambda x, a=a: gemv(a, x), evidence)
                 for key, a in components.items()}
        rotations, phases, alternatives = {}, {}, []
        for event in active:
            key = (event.rotation_id, event.cosine64.hex(), event.sine64.hex())
            if key not in rotations:
                rotations[key] = _closed_relation(states, lookup,
                    lambda x, e=event: rotation(components[e.rotation_id], x, e.cosine64, e.sine64), evidence)
            if event.phase not in phases:
                phases[event.phase] = _closed_relation(states, lookup, lambda x, p=event.phase: _phase(x, p), evidence)
            chain = identity(energies)
            for key_product in event.products: chain = compose(cache[key_product], chain)
            chain = compose(rotations[key], chain)
            alternatives.append(compose(phases[event.phase], chain))
        block = compose(rev, compose(power(union(alternatives), r, energies), f))
        full = power(block, q, energies)
        final_states = full[lookup[psi.tobytes()]]
        dots = [_dot(psi, states[end]) for end in final_states]
        magnitude = sqrt_upper(max(norm2([z(record['output'])]) for record in dots))
        abs_error = k.complex_abs_error(magnitude)
        complete = bridge.guard_completion(magnitude, abs_evaluation_error=abs_error, finite_intermediates=finite)
        result.update(status='CONDITIONAL_ARTIFICIAL_GUARD_PROVED' if complete else 'UNKNOWN_GUARD_STOP',
                      conditional_guard_proved=complete, endpoint_count=len(final_states),
                      prefix_energy_upper=max(final_states.values()), signal_magnitude_upper=magnitude,
                      abs_error_upper=abs_error, dot_evidence=dots)
        if action is not None:
            required = {'logical', 'means', 'shots', 'epsilon'}
            if not required.issubset(action):
                result['action_status'] = 'UNKNOWN_ACTION_INPUTS'
                result['artificial_axis_check'] = bridge.classify_axes({}, {}, {}, {}, None,
                    completion=complete, arithmetic_resolved=finite)
            else:
                result['artificial_axis_check'] = bridge.classify_axes(action['logical'], action['means'], action['shots'],
                    action['epsilon'], action.get('error'), completion=complete, arithmetic_resolved=finite)
    except ValueError as exc:
        result.update(status='UNKNOWN_STOP', reason=str(exc), conditional_guard_proved=False)
    return result
