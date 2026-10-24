"""All-law coverage rules, distinct from numerical sample correctness tests.

Whole-kernel P0 optimization does not factor over sampled event words. A small
finite support may be covered exhaustively within the registered build budget;
otherwise neither sample maxima nor a standard error supplies a support bound.
"""
import itertools
import math
from .c1_stage3_accuracy import METRICS


def support_size(component_count, orders, repetitions, *, budget):
    """Saturating count; do not enumerate a molecular exponential word space."""
    if component_count<1 or repetitions<1: return 0
    one=0
    for n in orders:
        one+=component_count**(n+1)
        if one>budget:return budget+1
    total=1
    for _ in range(repetitions):
        total*=one
        if total>budget:return budget+1
    return total


def exhaustive_trajectories(tail, distribution, repetitions, *, budget):
    from trotterlib.rte import enumerate_rte_events
    # A numerically underflowed positive ideal weight does not remove its word
    # from the ideal law. Such a support cannot be certified by this enumerator.
    if distribution.dimensionless_step_time!=0 and any(p==0 for p in distribution.order_probabilities):
        return None
    orders=[n for n,p in zip(distribution.orders,distribution.order_probabilities) if p>0]
    count=support_size(len(tail.components),orders,repetitions,budget=budget)
    if not count or count>budget:return None
    events=enumerate_rte_events(tail.components,distribution,max_events=budget)
    return tuple(itertools.product(events,repeat=repetitions))


def trajectory_key(events):
    return tuple((e.taylor_order,e.rotation_component_id,e.product_component_ids) for e in events)


def support_cost_envelope(records, *, complete_support):
    """A deterministic envelope for the expectation under ANY supported law.

    This uses all possible circuits, not the extrema of a random sample. It
    remains valid for the ideal reference law even when calibration uses a
    finite-bit PRNG. No unproved implemented-to-ideal mean transfer is needed.
    """
    if not complete_support or not records:
        return dict(status='UNRESOLVED',lower=None,upper=None,
                    simultaneous_interval_qualified=False,
                    reason='Whole-law compiler output support was not enumerated')
    return dict(status='ALL_FINITE_SUPPORT_COVERED',
        lower={k:min(r[k] for r in records) for k in METRICS},
        upper={k:max(r[k] for r in records) for k in METRICS},
        simultaneous_interval_qualified=True,
        method='deterministic_complete_support_range; not sample-Hoeffding or observed maximum')


def synthesis_envelope(*, whole_support_covered, reference_enclosures=None,
                       shell_enclosure=None, numerical_diagnostic_max=None):
    """Compose certified norm enclosures only, never promote a float residual.

    For unitary circuits the universal norm bound2 is valid but cannot certify
    the pilot phase room. It is reported as a coarse fallback, not a small
    rounding allowance. Numeric CAR/expm/Operator comparisons are diagnostics;
    absent an enclosure of those references the useful bound stays unknown.
    Exact artificial identities can supply exact analytic enclosures to test
    the full qualified path. The molecular backend never accepts such values
    from a user file or turns a sample residual into an enclosure.
    """
    enclosures=tuple(reference_enclosures) if reference_enclosures is not None else None
    if not whole_support_covered or not enclosures or shell_enclosure is None:
        return dict(status='UNRESOLVED',synthesis_error_per_kernel=None,
            synthesis_error_shell=None,universal_unitary_norm_bound=2.,
            numerical_diagnostic_max=numerical_diagnostic_max,
            reason='No complete useful compiler-plus-CAR reference error enclosure')
    if any(not math.isfinite(x) or x<0 for x in (*enclosures,shell_enclosure)):
        raise ValueError('Invalid analytic synthesis enclosure')
    return dict(status='ALL_LAW_ENCLOSURE_AVAILABLE',
        synthesis_error_per_kernel=max(enclosures),synthesis_error_shell=shell_enclosure,
        whole_support_covered=True,numerical_diagnostic_max=numerical_diagnostic_max)
