"""Artificial-only executable sketch, never a real-H8 numerical entry point.

Future real analysis needs a separately approved, source-bound implementation.
Keep error evidence empirical; never convert triangle inequalities of numerical
components into a rigorous bound on the unknown exact physical bias.
"""
import math
from h8_bias_evidence import require, finite_number, AXES


def artificial_envelopes(record):
    require(record.get('fixture_kind') == 'ARTIFICIAL', 'ARTIFICIAL_DATA_ONLY')
    parts = record.get('signed_parts')
    if parts is None:
        return dict(status='UNRESOLVED_COMPONENTS', signed=None, noncancellation=None,
                    rigorous_bound=None, certified=False)
    require(set(parts) in ({'discard', 'PF'}, {'PF'}, {'outer_PF', 'finite_RTE'}), 'COMPONENT_NAMES')
    total = record['total_signed']
    signed, envelope = {}, {}
    for axis in AXES:
        require(finite_number(total[axis]) and all(finite_number(p[axis]) for p in parts.values()), 'FINITE_COMPONENTS')
        summed = sum(p[axis] for p in parts.values())
        require(math.isfinite(summed) and math.isclose(summed, total[axis], rel_tol=1e-12, abs_tol=1e-12),
                'COMPONENT_CLOSURE')
        signed[axis] = abs(total[axis])
        envelope[axis] = sum(abs(p[axis]) for p in parts.values())
        require(math.isfinite(envelope[axis]), 'ENVELOPE_OVERFLOW')
    return dict(status='EMPIRICAL_NONCANCELLATION_DIAGNOSTIC', signed=signed,
                noncancellation=envelope, rigorous_bound=None, certified=False)


def artificial_shots(record, *, epsilon, allowance_factor):
    require(record.get('fixture_kind') == 'ARTIFICIAL', 'ARTIFICIAL_DATA_ONLY')
    result = artificial_envelopes(record)
    if result['noncancellation'] is None:
        return dict(status='UNRESOLVED_COMPONENTS', shots=None, population_guaranteed=False)
    require(epsilon in (.05, .01, .005, .001, .0005) and allowance_factor in (1., 10.), 'FIXED_PRECISION_FACTOR')
    require(record['allowance_kind'] == 'EMPIRICAL' and record['allowance_certified'] is False,
            'NO_CERTIFICATE_UPGRADE')
    axes = {}
    for axis in AXES:
        u = record['allowance'][axis]
        require(finite_number(u) and u >= 0, 'ALLOWANCE')
        headroom = epsilon/math.sqrt(2)-result['noncancellation'][axis]-allowance_factor*u
        if headroom <= 0:
            return dict(status='INELIGIBLE_OR_UNRESOLVED_BOUNDARY', shots=None, population_guaranteed=False)
        if allowance_factor*u/headroom > .01:
            return dict(status='NUMERICALLY_UNRESOLVED', shots=None, population_guaranteed=False)
        # Deterministic fixture normalization=1. Never used for real PR shots.
        n = 2/headroom**2*math.log(2/.025)
        require(math.isfinite(n) and n < 2**52, 'SHOT_INTEGER_RANGE')
        axes[axis] = max(1, math.ceil(n))
    return dict(status='ARTIFICIAL_EMPIRICAL_ELIGIBLE', shots=axes, population_guaranteed=False)


def artificial_full_domain(rows, expected_ids):
    """Missing parts prohibit a complete-domain winner, regardless of a subset."""
    require(all(r.get('fixture_kind') == 'ARTIFICIAL' for r in rows), 'ARTIFICIAL_DATA_ONLY')
    require(len(rows) == len(expected_ids) and len({r['id'] for r in rows}) == len(rows) and
            {r['id'] for r in rows} == set(expected_ids), 'FULL_REGISTERED_DOMAIN_REQUIRED')
    if any(r['status'] not in ('ELIGIBLE', 'INELIGIBLE') for r in rows):
        return dict(status='UNRESOLVED_FULL_DOMAIN', best=None, PR_reduction=None)
    valid = [r for r in rows if r['status'] == 'ELIGIBLE']
    require(all(r.get('cost') is not None and finite_number(r['cost']) and r['cost'] > 0 for r in valid),
            'ELIGIBLE_COST_MISSING')
    best = min(valid, key=lambda r: (r['cost'], r['id']))['id'] if valid else None
    return dict(status='ARTIFICIAL_SUBSET_COMPLETE', best=best, rigorous_bound=None)
