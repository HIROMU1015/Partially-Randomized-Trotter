"""Planning arithmetic only; no sampler, native provider or execution option.

The normal plug-in calculation is a scenario, not a power/coverage certificate.
Saved n8 costs cannot reveal an unobserved Taylor stratum's cost or variance.
"""
from fractions import Fraction
import math
from statistics import NormalDist


def require(value, reason):
    if not value:
        raise ValueError(reason)


def probability(value):
    require(type(value) in (float, int) and math.isfinite(value) and 0 <= value <= 1,
            'PROBABILITY')
    return float(value)


def normal_scenario(mean_exact, se, n_pilot, comparator_exact, *, family_alpha=.05,
                    comparisons=2, power=.8, sd_multiplier=1.):
    require(type(n_pilot) is int and n_pilot >= 2, 'PILOT_SIZE')
    require(type(comparisons) is int and comparisons > 0, 'FIXED_FAMILY')
    require(0 < probability(family_alpha) < 1 and 0 < probability(power) < 1, 'ALPHA_POWER')
    require(math.isfinite(se) and se > 0 and math.isfinite(sd_multiplier)
            and sd_multiplier > 0, 'POSITIVE_FINITE_VARIANCE_SCENARIO')
    mean, comparator = Fraction(mean_exact), Fraction(comparator_exact)
    require(mean > 0 and comparator > mean, 'POSITIVE_PILOT_MARGIN')
    margin = float(comparator - mean)
    sigma = se * math.sqrt(n_pilot) * sd_multiplier
    alpha_each = family_alpha / comparisons
    z = NormalDist().inv_cdf(1-alpha_each) + NormalDist().inv_cdf(power)
    n = math.ceil((z*sigma/margin)**2)
    return dict(normal_plugin_n=max(n, 1), sd_multiplier=sd_multiplier,
        alpha_each=alpha_each, assumed_power=power, margin_exact=str(comparator-mean),
        pilot_SD=se*math.sqrt(n_pilot), unknown_variance_t_adjustment=False,
        assumed_normal_finite_variance=True, actual_power_certified=False,
        all_strata_variance_covered=False)


def at_least_one(p, n):
    p = probability(p)
    require(type(n) is int and n >= 0, 'NONNEGATIVE_INTEGER_TRIALS')
    if n == 0 or p == 0: return 0.
    if p == 1: return 1.
    return -math.expm1(n*math.log1p(-p))


def detection_n(p, failure_probability):
    """Visibility only, never precision of a mean or completeness of support."""
    p = probability(p)
    alpha = probability(failure_probability)
    require(0 < alpha < 1, 'FAILURE_PROBABILITY')
    if p == 0: return None
    if p == 1: return 1
    return math.ceil(math.log(alpha)/math.log1p(-p))


def zero_count_probability_upper(n, alpha=.025):
    """One-sided binomial bound if n IID trajectory indicators all equal zero.

    This bounds an event probability, not the event's cost contribution.
    """
    require(type(n) is int and n > 0 and 0 < probability(alpha) < 1, 'BINOMIAL_DOMAIN')
    return -math.expm1(math.log(alpha)/n)


def visibility(p2, p4, events_per_trajectory, n):
    p2, p4 = probability(p2), probability(p4)
    require(p2+p4 <= 1 and type(events_per_trajectory) is int
            and events_per_trajectory > 0, 'FIXED_ORIGINAL_LAW')
    any_nonzero = at_least_one(p2+p4, events_per_trajectory)
    any_four = at_least_one(p4, events_per_trajectory)
    return dict(any_nonzero_trajectory_probability=any_nonzero,
        any_four_trajectory_probability=any_four,
        n_trajectory_nonzero_visibility=at_least_one(any_nonzero, n),
        n_trajectory_four_visibility=at_least_one(any_four, n),
        n_for_nonzero_visibility_97_5_percent=detection_n(any_nonzero, .025),
        n_for_four_visibility_97_5_percent=detection_n(any_four, .025),
        literal_saved_probability_rounding_not_renormalized=True,
        independent_original_law_assumed=True, cost_bound=None,
        population_mean_certified=False)


def calls(n_each):
    require(type(n_each) is int and n_each >= 2, 'REGISTERED_FIXED_SAMPLE_SIZE')
    return dict(unconditional_trajectory=2*n_each, wrapper_compile=4*n_each,
        native_evolution=2*n_each, occurrence=18*n_each, RTE_event=72*n_each,
        max_control_probe=8, max_independent_event_oracle=2,
        new_signal=0, new_SCF_DF_state=0, fit=0, quantum_measurement_shots=0)


if __name__ == '__main__':
    raise SystemExit('DESIGN FUNCTIONS ONLY; NO SCIENTIFIC EXECUTION ENTRY')
