"""Non-enumerative artificial H=I certificates; no loader, RNG or scientific CLI.

All allowed graphs in the existing forest family are enclosed. No runtime
determinism, negative radial sign or favorable rounding is inferred. Exact
rational/ball arithmetic is used only to analyze the unchanged FP program.
"""
from dataclasses import dataclass
from fractions import Fraction as Q
from pathlib import Path
import math
import struct
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'docs/research/rpe_m17_row_energy_adapter_preparation_20261104'))
import row_energy_adapter as old

e, k, b = old.energy, old.k, old.b


def qmatrix(matrix):
    n = len(matrix)
    old.validate_array(matrix, (n, n))
    k.path_bounds(n, n)
    return [[old.z(x) for x in row] for row in matrix]


def norm_squared_upper(matrix):
    """Gershgorin upper bound on lambda_max(A* A), retaining complex phases."""
    n = len(matrix)
    gram = [[old.sum_complex(old.mul(old.conj(matrix[t][i]), matrix[t][j]) for t in range(n))
             for j in range(n)] for i in range(n)]
    return max(gram[i][i][0]+sum((old.sqrt_upper(old.norm2([gram[i][j]]))
                                for j in range(n) if j != i), Q(0)) for i in range(n))


def _exact_row(row):
    nonzero = [x for x in row if x != old.ZERO]
    if not nonzero: return True
    if len(nonzero) != 1 or all(x != 0 for x in nonzero[0]): return False
    scale = max(abs(x) for x in nonzero[0])
    # Scaling UP by a power of two is exact when finite; scaling down can underflow.
    return scale.denominator == 1 and scale.numerator > 0 and scale.numerator.bit_count() == 1


def _real_rows(matrix):
    return [[a[0] for a in row]+[-a[1] for a in row] for row in matrix] + [
        [a[1] for a in row]+[a[0] for a in row] for row in matrix]


def matrix_certificate(matrix):
    a = qmatrix(matrix); n = len(a)
    bounds = k.path_bounds(n, n)
    gamma, tau = bounds.mv_rel/2, bounds.dot_abs/2
    exact = [_exact_row(row) for row in a]*2
    rows = _real_rows(a)
    error_weights = [[Q(0) if is_exact else abs(x) for x in row]
                     for row, is_exact in zip(rows, exact, strict=True)]
    row_sum = max(sum(row) for row in error_weights)
    col_sum = max(sum(row[j] for row in error_weights) for j in range(2*n))
    eta = gamma*old.sqrt_upper(row_sum*col_sum)
    additive = tau*old.sqrt_upper(Q(sum(not flag for flag in exact)))
    nu2 = norm_squared_upper(a); nu = old.sqrt_upper(nu2)
    norm_map = e.primitive(nu+eta, additive)
    return dict(matrix=a, real_rows=rows, exact_rows=exact, gamma=gamma, tau=tau,
                norm_squared_upper=nu2, norm_upper=nu, error_relative=eta, error_additive=additive,
                norm_map=norm_map, drift_coefficient_upper=nu2-1,
                frobenius_upper=old.sqrt_upper(sum((old.norm2(row) for row in a), Q(0))))


def energy_envelope(certificate, radius):
    r = e.rational(radius)
    if r < 0: raise ValueError('Nonnegative radius required')
    nu, eta, additive = (certificate[name] for name in ['norm_upper', 'error_relative', 'error_additive'])
    error = eta*r+additive
    radial = nu*r*error
    return dict(drift_coefficient_upper=certificate['drift_coefficient_upper'],
                radial_interval=(-radial, radial), error_squared_upper=error**2,
                norm_upper=(nu+eta)*r+additive, energy_upper=((nu+eta)*r+additive)**2,
                radial_status='SYMMETRIC_FOREST_FALLBACK')


def box_image(certificate, box):
    box = [(e.rational(lo), e.rational(hi)) for lo, hi in box]
    rows = certificate['real_rows']
    if len(box) != len(rows) or any(lo > hi for lo, hi in box):
        raise ValueError('Valid real/imag box required')
    result = []
    for row, exact in zip(rows, certificate['exact_rows'], strict=True):
        lo = sum(min(a*x, a*y) for a, (x, y) in zip(row, box, strict=True))
        hi = sum(max(a*x, a*y) for a, (x, y) in zip(row, box, strict=True))
        products = sum(abs(a)*max(abs(x), abs(y)) for a, (x, y) in zip(row, box, strict=True))
        error = Q(0) if exact else certificate['gamma']*products+certificate['tau']
        result.append((lo-error, hi+error))
    return result


def box_invariant(certificate, box):
    image = box_image(certificate, box)
    included = all(e.rational(lo) <= a and z <= e.rational(hi)
                   for (lo, hi), (a, z) in zip(box, image, strict=True))
    radius = old.sqrt_upper(sum((max(abs(e.rational(lo)), abs(e.rational(hi)))**2 for lo, hi in box), Q(0)))
    finite = k.finite_intermediate_envelope(matrix_frobenius=certificate['frobenius_upper'],
                                           initial_norm=radius, boundary_norm=radius, scalar_modulus=1)
    return included and finite


def single_product_cell(coefficients, box, candidate):
    """Signed residual/radial for a row with at most ONE effective real product.

    Multi-product GEMV rows are not assumed correctly rounded as a whole.
    Under the confirmed graph, one product plus exact zeros has this RNE cell
    for every allowed reduction path. Box arithmetic contains all real inputs.
    """
    if any(type(x) is not float or not math.isfinite(x) for x in coefficients):
        raise ValueError('Stored finite binary64 row coefficients required')
    coeffs = [Q.from_float(x) for x in coefficients]
    box = [(e.rational(lo), e.rational(hi)) for lo, hi in box]
    if len(coeffs) != len(box) or any(lo > hi for lo, hi in box):
        raise ValueError('Valid row box required')
    active = [i for i, c in enumerate(coeffs) if c]
    if len(active) > 1: return dict(status='UNKNOWN_MULTIPRODUCT_CELL')
    lo = hi = Q(0)
    if active:
        i = active[0]; c = coeffs[i]
        lo, hi = sorted([c*box[i][0], c*box[i][1]])
    if type(candidate) is not float or not math.isfinite(candidate):
        raise ValueError('Finite binary64 cell candidate required')
    bits = struct.unpack('>Q', struct.pack('>d', candidate))[0]
    if candidate == 0:
        prev, following = -float.fromhex('0x0.0000000000001p-1022'), float.fromhex('0x0.0000000000001p-1022')
    else:
        delta = 1 if candidate > 0 else -1
        prev = struct.unpack('>d', struct.pack('>Q', bits-delta))[0]
        following = struct.unpack('>d', struct.pack('>Q', bits+delta))[0]
    if not all(math.isfinite(x) for x in [prev, following]):
        return dict(status='UNKNOWN_CELL_RANGE')
    y = Q.from_float(candidate)
    left, right = (Q.from_float(prev)+y)/2, (y+Q.from_float(following))/2
    even = bits & 1 == 0
    inside = left <= lo and hi <= right if even else left < lo and hi < right
    if not inside: return dict(status='UNKNOWN_CELL_CROSSING')
    values = [lo*(y-lo), hi*(y-hi)]
    maximum = max(values+[y*y/4] if lo <= y/2 <= hi else values)
    return dict(status='PROVED_SINGLE_PRODUCT_CELL', target_interval=(lo, hi),
                residual_interval=(y-hi, y-lo), radial_interval=(min(values), maximum),
                input_enumerations=0)


def _maximum(maps):
    return e.NormAffine(*(max(getattr(x, field) for x in maps) for field in e.NormAffine.__dataclass_fields__))


def rotation_map(certificate, cosine64, sine64):
    if any(type(x) is not float or not math.isfinite(x) for x in [cosine64, sine64]):
        raise ValueError('Finite binary64 trig outputs required')
    c, s = Q.from_float(cosine64), Q.from_float(sine64)
    p = certificate; n = len(p['matrix']); bounds = k.path_bounds(n, n)
    target = [[old.add((c if i == j else Q(0), Q(0)), old.neg(old.mul((Q(0), s), p['matrix'][i][j])))
               for j in range(n)] for i in range(n)]
    nu_t = old.sqrt_upper(norm_squared_upper(target))
    nu, eta, additive = p['norm_upper'], p['error_relative'], p['error_additive']
    scale, sub = bounds.scale_rel, bounds.sub_rel
    first_g, second_g = (1+scale)*abs(c), (1+scale)*abs(s)*(nu+eta)
    first_a, second_a = bounds.scale_abs, (1+scale)*abs(s)*additive+bounds.scale_abs
    relative = scale*abs(c)+abs(s)*eta+scale*abs(s)*(nu+eta)+sub*(first_g+second_g)
    absolute = (1+sub)*(first_a+second_a)+bounds.sub_abs
    g, a = nu_t+relative, absolute
    if s == 0 and abs(c) in [0, 1]: g, a = abs(c), Q(0)
    elif c == 0 and abs(s) == 1: g, a = nu+eta, additive
    # P@v and both branches still execute, even when a scalar is zero.
    return e.NormAffine(g, a, max(1, nu+eta, first_g, second_g, g),
                        max(additive, first_a, second_a, a))


@dataclass(frozen=True)
class OrderFamily:
    order: int
    cosine64: float
    sine64: float
    mass: Q


def image(norm_map, radius): return norm_map.final_scale*radius+norm_map.final_offset
def prefix(norm_map, radius): return norm_map.prefix_scale*radius+norm_map.prefix_offset


def prove(forward, reverse, components, component_masses, orders, state_norm_squared, *, r, q, runtime=None, action=None):
    result = dict(scope='ARTIFICIAL_INVARIANT_REGION_ONLY', status='UNKNOWN_STOP', mandatory_STOP=True,
                  U_virtual_IID_upper=None, U_exec_upper=None, scientific_status='UNDETERMINED',
                  RPE_tolerance_certified=False, scientific_execution_ready=False,
                  conditional_guard_proved=False, state_enumerations=0, trajectory_enumerations=0,
                  runtime_live_observed=False)
    try:
        n = len(forward)
        if (not isinstance(runtime, dict) or runtime.get('stable_all_threads') is not True
                or runtime.get('forest_family_confirmed') is not True):
            raise ValueError('UNKNOWN_RUNTIME: allowed graph family or stable FP not established')
        contract = k.selected_path_contract(runtime.get('profile'), rows=n, columns=n, layout='C',
                                           mxcsr_values=runtime.get('mxcsr', []))
        if contract['bounds'] is None: raise ValueError('UNKNOWN_RUNTIME: profile/FP mismatch')
        if any(type(count) is not int or count < 1 for count in [r, q]):
            raise ValueError('Positive integer r/q required')
        lo, hi = (e.rational(state_norm_squared[name]) for name in ['lower', 'upper'])
        if not 0 < lo <= hi: raise ValueError('Positive saved norm-squared interval required')
        radius = old.sqrt_upper(hi)
        masses = [e.rational(x) for x in component_masses]
        if (not components or len(masses) != len(components) or any(x < 0 for x in masses) or sum(masses) != 1):
            raise ValueError('Complete exact component law required')
        active = [x for x, mass in zip(components, masses, strict=True) if mass > 0]
        probabilities = [e.rational(order.mass) for order in orders]
        if not orders or any(x < 0 for x in probabilities) or sum(probabilities) != 1:
            raise ValueError('Complete exact order law required')
        active_orders = [o for o, mass in zip(orders, probabilities, strict=True) if mass > 0]
        if any(type(o.order) is not int or o.order not in [0, 2] for o in active_orders):
            raise ValueError('Fixed K2 order family required')
        f, rev = matrix_certificate(forward), matrix_certificate(reverse)
        certificates = [matrix_certificate(matrix) for matrix in active]
        if any(len(c['matrix']) != n for c in certificates+[rev]): raise ValueError('Shape mismatch')
        product = _maximum([c['norm_map'] for c in certificates])
        event_maps = []
        for order in active_orders:
            rotations = _maximum([rotation_map(c, order.cosine64, order.sine64) for c in certificates])
            event_maps.append(e.compose(rotations, e.power(product, order.order)))
        event_radius = image(f['norm_map'], radius)
        result.update(initial_radius=radius, event_radius=event_radius, matrix_certificates=certificates,
                      event_maps=event_maps, r=r, q=q, radial_status='SYMMETRIC_FOREST_FALLBACK')
        if any(image(event, event_radius) > event_radius for event in event_maps):
            raise ValueError('UNKNOWN_INVARIANCE: event family not certified inside common ball')
        if image(rev['norm_map'], event_radius) > radius:
            raise ValueError('UNKNOWN_INVARIANCE: full block not certified inside initial ball')
        peak = max([radius, prefix(f['norm_map'], radius), prefix(rev['norm_map'], event_radius)]
                   +[prefix(event, event_radius) for event in event_maps])
        matrix_norm = max(c['frobenius_upper'] for c in [f, rev]+certificates)
        scalar = max([Q(1)]+[abs(Q.from_float(t)) for o in active_orders for t in [o.cosine64, o.sine64]])
        finite = k.finite_intermediate_envelope(matrix_frobenius=matrix_norm, initial_norm=radius,
                                               boundary_norm=peak, scalar_modulus=scalar)
        if not finite: raise ValueError('UNKNOWN_FINITE: prefix/intermediate envelope failed')
        bounds = k.path_bounds(n, n)
        magnitude = (1+bounds.dot_rel)*radius*radius+bounds.dot_abs
        abs_error = k.complex_abs_error(magnitude)
        complete = old.bridge.guard_completion(magnitude, abs_evaluation_error=abs_error, finite_intermediates=finite)
        result.update(status='CONDITIONAL_ARTIFICIAL_INVARIANT_GUARD_PROVED' if complete else 'UNKNOWN_GUARD_STOP',
                      conditional_guard_proved=complete, prefix_norm_upper=peak, signal_magnitude_upper=magnitude,
                      abs_error_upper=abs_error, finite_conditional=finite, allowed_graph_variation_enclosed=True)
        if action is not None:
            result['artificial_axis_check'] = old.bridge.classify_axes(action['logical'], action['means'],
                action['shots'], action['epsilon'], action.get('error'), completion=complete, arithmetic_resolved=finite)
    except (ValueError, KeyError, TypeError) as exc:
        result.update(status='UNKNOWN_STOP', reason=str(exc), conditional_guard_proved=False)
    return result
