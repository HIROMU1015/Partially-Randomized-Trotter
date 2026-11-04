"""Pure sufficient conditions. Importing this file never loads research data.

Arb bounds certify the supplied mathematical inputs, not their physical meaning.
Float64 envelopes must be supplied by a separate uniform proof; None is UNKNOWN.
"""
from __future__ import annotations

from fractions import Fraction as Q
import importlib.util
import math
from pathlib import Path
import sys

from flint import acb, arb

ROOT = Path(__file__).resolve().parents[3]
BALL = ROOT / 'docs/research/rpe_m17_rigorous_mean_enclosure_preparation_20261103'
sys.path.insert(0, str(BALL))
import ball_adapter as ball

STATUS = dict(RPE_tolerance_certified=False, scientific_status='UNDETERMINED',
              U_virtual_IID_upper=None, U_exec_upper=None, mandatory_STOP=True)


def stop(reason):
    raise ValueError('UNKNOWN/STOP: ' + reason)


def real(value):
    try:
        result = value if isinstance(value, arb) else ball.real(value)
    except (ValueError, TypeError, OverflowError):
        stop('invalid finite numeric input')
    if not result.is_finite():
        stop('nonfinite input')
    return result


def nonnegative(value):
    value = real(value)
    if not value >= 0:
        stop('nonnegative bound not established')
    return value


def positive(value):
    value = real(value)
    if not value > 0:
        stop('strict positivity not established')
    return value


def integer(value):
    if type(value) is not int or value < 1:
        stop('positive integer required')
    return value


def upper(value):
    return ball.upper(real(value))


def encode(value):
    return None if value is None else ball.endpoints(real(value))


def interval(saved):
    lo, hi = Q(saved['lower']), Q(saved['upper'])
    if lo > hi:
        stop('inverted interval')
    return arb(real((lo + hi) / 2), upper(real((hi - lo) / 2)))


def circle_distance(left, right):
    """Branch-cut-safe upper enclosure of circular distance, including +/-pi."""
    difference = real(left) - real(right)
    angle = acb(0, difference).exp().arg()
    return upper(abs(angle))


def round_geometry(mu_real, mu_imag, eps_cosine, eps_sine, phase_reference, q):
    """Saved center +/- epsilon, after the already certified local event.

    Does NOT take U_cat: that error is already inside epsilon. Does NOT assume
    unit radius, PF eigenstate preparation, or a rigorous empirical PF budget.
    The rectangle fallback is fixed in advance and only used if tighter.
    """
    q = integer(q)
    x, y = real(mu_real), real(mu_imag)
    ex, ey = positive(eps_cosine), positive(eps_sine)
    phi = real(phase_reference)
    center = acb(x, y)
    rho = abs(center)
    if not rho > 0:
        stop('saved center radius not strictly positive')
    target_conjugate = acb(0, -q * phi).exp()
    systematic = upper(abs((center * target_conjugate).arg()))
    disk_radius = (ex**2 + ey**2).sqrt()
    disk, statistical = None, None
    if disk_radius < rho:
        statistical = upper((disk_radius / rho).asin())
        disk = upper(systematic + statistical)
    corners = [acb(x + sx * ex, y + sy * ey) * target_conjugate
               for sx in (-1, 1) for sy in (-1, 1)]
    rectangle = None
    if all(z.real > 0 for z in corners):
        rectangle = max(upper(abs(z.arg())) for z in corners)
    choices = [(v, name) for v, name in ((disk, 'disk'), (rectangle, 'rectangle'))
               if v is not None]
    bound, method = min(choices, key=lambda item: ball.dyadic(item[0])) if choices else (None, None)
    return dict(radius=encode(rho), coordinate_disk_radius=encode(disk_radius),
                systematic_phase_upper=encode(systematic),
                coordinate_phase_upper=encode(statistical), disk_phase_upper=encode(disk),
                rectangle_phase_upper=encode(rectangle), phase_upper=encode(bound),
                method=method, decision='SUFFICIENT' if bound is not None else 'UNKNOWN/STOP')


def phase_upper(geometry):
    if geometry['phase_upper'] is None:
        stop('phase geometry unresolved')
    return real(geometry['phase_upper']['upper'])


def schedule(q_values):
    q = tuple(q_values)
    if not q or any(type(x) is not int for x in q) or q[0] != 1 or any(
            right != 2 * left for left, right in zip(q, q[1:])):
        stop('consecutive doubling q schedule starting at one required')
    return q


def branch_conditions(bounds, q_values, float_envelopes=None):
    """Ideal proof and conditional implementation proof; see PROOF.md section 4.

    Per round: angle = phase extraction + principal-wrap circular error;
    candidate = uniform base-phase root-generation error (m0 final wrap too);
    distance = error of each computed circular distance. All three are absolute
    bounds, uniform over the whole coordinate-success event and every candidate.
    """
    q = schedule(q_values)
    if len(bounds) != len(q):
        stop('phase bound inventory mismatch')
    b = [nonnegative(x) for x in bounds]
    margins = [arb.pi() - 2 * left - right for left, right in zip(b, b[1:])]
    ideal = all(x > 0 for x in margins) and all(x < arb.pi() for x in b)
    out = dict(ideal_decision='SUFFICIENT' if ideal else 'UNKNOWN/STOP',
               ideal_margins=[encode(x) for x in margins],
               ideal_final_phase_upper=encode(upper(b[-1] / q[-1])),
               implementation_decision='UNKNOWN/STOP', implementation_margins=None,
               implementation_final_phase_upper=None)
    if float_envelopes is None:
        return out
    if len(float_envelopes) != len(q):
        stop('Float64 envelope inventory mismatch')
    a, c, d = [], [], []
    for entry in float_envelopes:
        if entry is None or set(entry) != {'angle', 'candidate', 'distance'}:
            stop('incomplete uniform Float64 envelope')
        a.append(nonnegative(entry['angle']))
        c.append(nonnegative(entry['candidate']))
        d.append(nonnegative(entry['distance']))
    impl = [arb.pi() - (2 * (b[m-1] + a[m-1]) + b[m] + a[m]
            + q[m] * (c[m-1] + c[m] + d[m])) for m in range(1, len(q))]
    resolved = ideal and all(x > 0 for x in impl) and all(
        b[i] + a[i] + q[i] * c[i] < arb.pi() for i in range(len(q)))
    out.update(implementation_decision='CONDITIONAL_ON_SUPPLIED_UNIFORM_ENVELOPES' if resolved else 'UNKNOWN/STOP',
               implementation_margins=[encode(x) for x in impl],
               implementation_final_phase_upper=encode(upper((b[-1] + a[-1]) / q[-1] + c[-1])))
    return out


def energy_conditions(phase_reference, final_phase_error, delta, target,
                      reference_energy_error=None, conversion_error=0):
    """Bounds to the literal reference; physical energy needs an external eta.

    phase_reference = -E_ref*delta. A reference eta is added here ONCE; it is not
    also added q*delta*eta to every round in this convention.
    """
    phi, error = real(phase_reference), nonnegative(final_phase_error)
    delta, target = positive(delta), positive(target)
    conversion = nonnegative(conversion_error)
    margin = arb.pi() - abs(phi)
    alias = margin > 0
    lift = alias and margin > error
    literal_bound = upper(error / delta + conversion)
    out = dict(reference_alias_decision='SUFFICIENT' if alias else 'UNKNOWN/STOP',
               representative_lift_decision='SUFFICIENT' if lift else 'UNKNOWN/STOP',
               boundary_margin=encode(margin), literal_energy_upper=encode(literal_bound),
               literal_target_decision='SUFFICIENT' if lift and literal_bound < target else 'UNKNOWN/STOP',
               physical_energy_upper=None, physical_target_decision='UNKNOWN/STOP',
               physical_alias_decision='UNKNOWN/STOP')
    if reference_energy_error is not None:
        eta = nonnegative(reference_energy_error)
        physical = upper(literal_bound + eta)
        out.update(physical_energy_upper=encode(physical),
                   physical_alias_decision='CONDITIONAL_ON_REFERENCE_BOUND' if margin > delta * eta else 'UNKNOWN/STOP',
                   physical_target_decision='CONDITIONAL_ON_REFERENCE_BOUND' if lift and physical < target else 'UNKNOWN/STOP')
    return out


def union_bound(alphas, budget):
    """Exact rational arithmetic, with a distinct outward binary64 display.

    Strings denote exact saved decimal/rational alpha, floats exact represented
    binary64 alpha. The caller must choose and label the convention explicitly.
    """
    values = tuple(ball.rational(x) for x in alphas)
    limit = ball.rational(budget)
    if not values or any(not 0 < x < 1 for x in values) or not 0 < limit < 1:
        stop('invalid alpha or budget')
    total = sum(values, Q())
    rounded = float(total)
    if Q.from_float(rounded) < total:
        rounded = math.nextafter(rounded, math.inf)
    return dict(alpha_sum_exact=str(total), alpha_sum_float_upper=rounded.hex(),
                success_lower_exact=str(max(Q(), 1-total)), budget_exact=str(limit),
                saved_budget_decision='SUFFICIENT' if total <= limit else 'UNKNOWN/STOP')


def local_axis_condition(mean_interval, saved_mean, n, alpha, epsilon):
    """Reuse existing outward Hoeffding classification, mainly for toy tests.

    The future H4 runner reuses prior decisions and never calls this function.
    """
    integer(n)
    return ball.classify_interval(interval(mean_interval), float(saved_mean), n, alpha, epsilon)


def existing_functions():
    """Reuse V2's unchanged AST loader, never import scientific library modules.

    Reads SOURCE code only, no saved means, results, schedules, or state arrays.
    The caller's fixed-input byte check binds the code for scientific use.
    """
    path = ROOT / 'docs/research/pr_rpe_v2_branch_success_20261103/run_saved_v2.py'
    spec = importlib.util.spec_from_file_location('d_existing_v2_loader', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    functions, records = module.selected_functions(
        (ROOT / 'src/trotterlib/rpe_four_round_phase_validation.py').read_text(),
        ('_wrap_phase', '_phase_distance', 'reconstruct_rpe_phase'))
    sums, sum_records = module.selected_functions(
        (ROOT / 'src/trotterlib/rpe_resource_accounting.py').read_text(),
        ('_conservative_nonnegative_sum',))
    sums['_conservative_nonnegative_sum'].__globals__['Fraction'] = Q
    return {**functions, **sums}, {**records, **sum_records}
