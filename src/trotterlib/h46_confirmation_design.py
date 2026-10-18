"""Pure, unadopted confirmation-design calculations. No execution capability."""
from decimal import Decimal, localcontext
import heapq
import math
from statistics import NormalDist


MINIMAL = ('H6_S2_q8', 'H6_Y4_q2', 'H6_Y4_q4')
EXPANDED = MINIMAL + ('H6_N2_q1', 'H4_S2_q8', 'H4_Y4_q2', 'H4_N2_q1')
PRIMARY_PAIRS = (('H6_S2_q8', 'H6_Y4_q2'), ('H6_Y4_q4', 'H6_Y4_q2'))
EXPANDED_PAIRS = PRIMARY_PAIRS + (
    ('H6_S2_q8', 'H6_N2_q1'), ('H6_Y4_q2', 'H6_N2_q1'),
    ('H4_S2_q8', 'H4_Y4_q2'), ('H4_S2_q8', 'H4_N2_q1'),
    ('H4_Y4_q2', 'H4_N2_q1'))


def checked(value, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError('FINITE_NUMBER_REQUIRED')
    if value < 0 or (positive and value <= 0):
        raise ValueError('NUMBER_RANGE')
    return value


def paired_law_strata(coefficients, steps, q, lambda_r, T=.8):
    """Ideal arithmetic of the existing paired K4 law; no cost imputation.

    Decimal stabilizes tiny residual strata. This is not certification of
    floating-point PRNG probabilities or physical coefficient errors.
    """
    if len(coefficients) != len(steps) or sum(steps) != 32 or type(q) is not int or q < 1:
        raise ValueError('SCHEDULE_SCOPE')
    if any(type(n) is not int or n < 1 for n in steps):
        raise ValueError('STEP_COUNTS')
    if any(isinstance(w, bool) or not math.isfinite(w) for w in coefficients):
        raise ValueError('FINITE_COEFFICIENT_REQUIRED')
    checked(lambda_r, positive=True); checked(T, positive=True)
    with localcontext() as ctx:
        ctx.prec = 60
        p0 = Decimal(1); no4 = Decimal(1); exactly2_ratio = Decimal(0)
        log_b = Decimal(0)
        for weight, n in zip(coefficients, steps):
            # Preserve the existing binary64 time-expression evaluation.
            tau = Decimal.from_float(abs(lambda_r * (T * weight / q) / n))
            weights = [((Decimal(1) if order == 0 else tau ** order) / Decimal(math.factorial(order))) *
                       (1 + (tau / Decimal(order + 1)) ** 2).sqrt() for order in (0, 2, 4)]
            normalizer = sum(weights); probs = [w / normalizer for w in weights]
            count = q * n
            p0 *= probs[0] ** count
            no4 *= (probs[0] + probs[1]) ** count
            exactly2_ratio += count * probs[1] / probs[0]
            log_b += count * normalizer.ln()
        one2 = p0 * exactly2_ratio
        result = {'s0': float(p0), 's2_single': float(one2),
                  's2_multi': float(no4 - p0 - one2), 's4': float(1 - no4)}
        if any(v < -1e-40 for v in result.values()) or abs(sum(result.values()) - 1) > 1e-14:
            raise ValueError('STRATUM_PROBABILITY_CLOSURE')
        return {'probabilities': result, 'log_B_from_saved_schedule': float(log_b),
                'costs_of_unobserved_strata': None, 'population_variance_bound': None,
                'floating_PRNG_law_certified': False}


def difference_precision_n(va, vb, effect, *, family_size, halfwidth_fraction, variance_factor=1):
    for value in (va, vb): checked(value)
    checked(effect, positive=True); checked(halfwidth_fraction, positive=True)
    checked(variance_factor, positive=True)
    if type(family_size) is not int or family_size < 1: raise ValueError('FAMILY_SIZE')
    z = NormalDist().inv_cdf(1 - .05 / (2 * family_size))
    return max(2, math.ceil(z * z * variance_factor * (va + vb) /
                            (halfwidth_fraction * effect) ** 2))


def design(pilots, ids, pairs, planner, *, power=.8, variance_factor=1,
           effect_scale=1, halfwidth_fraction=None):
    if not pairs or len(set(ids)) != len(ids): raise ValueError('DESIGN_SCOPE')
    if any(a not in ids or b not in ids or a == b for a, b in pairs): raise ValueError('PAIR_SCOPE')
    if len({tuple(sorted(p)) for p in pairs}) != len(pairs): raise ValueError('DUPLICATE_PAIR')
    checked(effect_scale, positive=True); checked(variance_factor, positive=True)
    allocation = {candidate: 2 for candidate in ids}; comparisons = []
    for a, b in pairs:
        left, right = pilots[a], pilots[b]
        for p in (left, right):
            checked(p['mean'], positive=True); checked(p['variance'])
            if p['pilot_n'] < 2: raise ValueError('PILOT_VARIANCE_REQUIRED')
        effect = abs(left['mean'] - right['mean']) * effect_scale
        checked(effect, positive=True)
        if halfwidth_fraction is None:
            n = planner(left['variance'] * variance_factor, right['variance'] * variance_factor,
                        effect, alpha=.05, power=power, comparisons=len(pairs))['equal_n_plugin']
        else:
            n = difference_precision_n(left['variance'], right['variance'], effect,
                                       family_size=len(pairs), halfwidth_fraction=halfwidth_fraction,
                                       variance_factor=variance_factor)
        allocation[a] = max(allocation[a], n); allocation[b] = max(allocation[b], n)
        comparisons.append({'left': a, 'right': b, 'planning_effect': effect,
                            'equal_n_pair_plugin': n})
    equal = max(allocation.values())
    return {'family_size': len(pairs), 'family_alpha_nominal': .05, 'power_planning': power,
            'variance_factor_sensitivity': variance_factor, 'effect_scale': effect_scale,
            'halfwidth_fraction': halfwidth_fraction,
            'endpoint_allocation_plugin': allocation,
            'equal_allocation_plugin': {candidate: equal for candidate in ids},
            'comparisons': comparisons, 'candidate_independence_assumed_for_planning': True,
            'coverage_or_power_guaranteed': False, 'variance_factor_is_not_an_upper_bound': True,
            'adopted_n': None, 'new_sampling': 0, 'new_compile': 0}


def resource_projection(allocation, resources, workers):
    if type(workers) is not int or workers < 1: raise ValueError('WORKERS')
    tasks = []
    for candidate, n in allocation.items():
        if type(n) is not int or n < 2: raise ValueError('SAMPLE_COUNT')
        duration = checked(resources[candidate]['seconds_per_trajectory_proxy'], positive=True)
        tasks.extend([duration] * n)
    use = min(workers, len(tasks)); slots = [0.] * use
    for duration in sorted(tasks, reverse=True):
        end = heapq.heappop(slots) + duration
        heapq.heappush(slots, end)
    rss = max(resources[c]['peak_worker_RSS_bytes'] for c in allocation)
    return {'trajectories': len(tasks), 'cost_wrapper_axes': 2 * len(tasks),
            'workers_scenario': use, 'worker_hours_proxy': sum(tasks) / 3600,
            'LPT_wall_minutes_proxy': max(slots) / 60,
            'worker_RSS_GiB_proxy': use * rss / 2 ** 30,
            'actual_CPU_time_measured': False, 'resource_or_wall_upper_bound': False,
            'control_initialization_and_unobserved_event_overhead_included_as_bound': False}


def refuse_unapproved(plan, grant):
    """No native runner exists here, even for an apparently approved input."""
    if plan.get('status') != 'SEALED_APPROVED' or grant.get('valid') is not True:
        raise ValueError('EXPLICIT_CONFIRMATION_EXECUTION_APPROVAL_REQUIRED')
    raise ValueError('CONFIRMATION_NATIVE_ADAPTER_AND_FINAL_SOURCE_SEAL_REQUIRED')
