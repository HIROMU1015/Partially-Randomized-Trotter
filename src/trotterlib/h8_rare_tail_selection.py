"""Preparation primitives for rare-stratum/selection diagnostics.

Standard library only. No molecular imports, native compiler, production inputs,
job launcher or authorization issuer. The CLI is fixture-only in this SOURCE.
Numerical functions are exercised on synthetic fixtures only in this campaign.
"""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
import random
import statistics


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def in_stratum(n2, n4, stratum):
    if stratum == 's0':
        return n2 == 0 and n4 == 0
    if stratum == 's2':
        return n2 >= 1 and n4 == 0
    if stratum == 's4':
        return n4 >= 1
    if stratum == 's2_single':
        return n2 == 1 and n4 == 0
    if stratum == 's2_multi':
        return n2 >= 2 and n4 == 0
    raise ValueError('UNKNOWN_STRATUM')


def _probabilities(values):
    require(len(values) > 0 and all(type(p) in (int, float) and math.isfinite(p)
            and 0 <= p <= 1 for p in values), 'PROBABILITIES')
    require(math.isclose(math.fsum(values), 1., rel_tol=0, abs_tol=2e-15), 'NORMALIZATION')
    return tuple(values)


@dataclass(frozen=True)
class CountLaw:
    R: int
    stratum: str
    counts: tuple
    probabilities: tuple
    log_stratum_mass: float


def _xlog(n, p):
    return 0. if n == 0 else -math.inf if p == 0 else n*math.log(p)


@lru_cache(maxsize=128)
def conditional_counts(R, probabilities, stratum):
    """O(R**2) multinomial support; normalize in log space, no rare rejection.

    log_stratum_mass avoids forming 1-nearly_one or subtracting rare masses.
    Negligible normalized cells may underflow; mass > 0 is never inferred from
    cancellation. Float representability is a limitation, not exact arithmetic.
    """
    require(type(R) is int and R >= 1, 'R')
    p0, p2, p4 = _probabilities(probabilities)
    support, logs = [], []
    for n4 in range(R+1):
        for n2 in range(R-n4+1):
            if not in_stratum(n2, n4, stratum):
                continue
            n0 = R-n2-n4
            v = (math.log(math.comb(R, n4)) + math.log(math.comb(R-n4, n2))
                 + _xlog(n0, p0) + _xlog(n2, p2) + _xlog(n4, p4))
            if math.isfinite(v):
                support.append((n2, n4)); logs.append(v)
    require(bool(logs), 'EMPTY_CONDITIONAL_STRATUM')
    high = max(logs)
    weights = tuple(math.exp(v-high) for v in logs)
    total = math.fsum(weights)
    return CountLaw(R, stratum, tuple(support), tuple(w/total for w in weights),
                    high+math.log(total))


def categorical(probabilities, rng):
    _probabilities(probabilities)
    u, running, last = rng.random(), 0., None
    for i, p in enumerate(probabilities):
        if p > 0:
            last = i
        running += p
        if u < running:
            return i
    require(last is not None, 'EMPTY_CATEGORY')
    return last


def place_orders(R, n2, n4, rng):
    require(all(type(n) is int and n >= 0 for n in (R, n2, n4))
            and n2+n4 <= R, 'COUNT_POSITION_DOMAIN')
    # A uniform permutation of labelled positions gives every order word with
    # these counts equal probability; never pin an event at a chosen occurrence.
    positions = rng.sample(range(R), n2+n4)
    out = [0]*R
    for i in positions[:n2]:
        out[i] = 2
    for i in positions[n2:]:
        out[i] = 4
    return tuple(out)


def draw_orders(law, rng):
    n2, n4 = law.counts[categorical(law.probabilities, rng)]
    return place_orders(law.R, n2, n4, rng)


def validate_scope(orders, *, q, r, stratum):
    require(type(q) is int and q in (4, 8) and type(r) is int and r == 4
            and len(orders) == q*r and all(type(k) is int and k in (0, 2, 4) for k in orders),
            'WHOLE_TRAJECTORY_SCOPE')
    require(in_stratum(orders.count(2), orders.count(4), stratum), 'STRATUM_MEMBERSHIP')


def component_events(orders, components, probabilities, rngs, r, make_event):
    """Draw rotation index first, then k products; delegate ALL physics to provider.

    make_event(k, indices, components) must be the pinned provider _make_event
    closure. No component/basis sorting, importance reweighting or unit-cost sum.
    rngs are occurrence-step streams; product/component application order is
    determined by _make_event, not rewritten here. Mock callback in these tests.
    """
    _probabilities(probabilities)
    require(len(components) == len(probabilities) and type(r) is int and r > 0
            and len(orders) == len(rngs)*r, 'COMPONENT_SCOPE')
    require(all(type(k) is int and k in (0, 2, 4) for k in orders), 'ORDER_SUPPORT')
    return tuple(make_event(k, tuple(categorical(probabilities, rngs[i//r])
                                    for _ in range(k+1)), components)
                 for i, k in enumerate(orders))


def proposed_seed(stage, candidate, replica, occupied=()):
    require(stage in ('S4_PILOT', 'S2_MULTI_PILOT', 'INDEPENDENT_HOLDOUT'), 'SEED_STAGE')
    require(type(replica) is int and replica >= 0, 'REPLICA')
    # Proposed namespace, not an issued sampling seed registry.
    value = int.from_bytes(hashlib.sha256(
        f'H8_RARE_TAIL_SELECTION_DRAFT_20261018|{stage}|{candidate}|{replica}'.encode()).digest()[:8], 'big')
    require(value not in set(occupied), 'SEED_COLLISION')
    return value


def mock_wrapper_pair(events, binding, expected, compile_wrapper):
    """Full-wrapper orchestration using an injected SYNTHETIC compiler only."""
    require(binding.get('input_kind') == 'SYNTHETIC', 'SYNTHETIC_ONLY')
    require(binding == expected, 'SOURCE_INPUT_COMPILER_BINDING')
    identity = digest(events)
    rows = []
    for axis in ('cosine', 'sine'):
        wrapper = dict(scope='whole_Hadamard_wrapper', control='symmetric_directional',
                       preparation_cost_P=0, axis=axis, events=events, event_digest=identity,
                       binding=dict(binding))
        measured = compile_wrapper(wrapper)
        require(digest(events) == identity, 'PAIRED_EVENT_MUTATION')
        require(measured.get('cost_kind') == 'MEASURED_MOCK_WRAPPER', 'MEASURED_NOT_MODEL')
        require(all(type(measured.get(k)) is int and measured[k] >= 0
                    for k in ('RZ', 'CX', 'depth')), 'MOCK_COST_METRICS')
        rows.append(dict(axis=axis, event_digest=identity, **measured))
    return rows


def validate_selection_data(rows, candidate_set, *, expected_n, metadata):
    """Validate identities only; same replica/seed does not prove cross-candidate CRN."""
    require(metadata.get('input_kind') == 'SYNTHETIC', 'SYNTHETIC_ONLY')
    require(metadata.get('selected_before_confirmation') is True, 'SELECTION_TIMING')
    require(metadata.get('cross_candidate_design') in ('DISTINCT_STREAMS', 'NOT_VERIFIED'),
            'CROSS_CANDIDATE_PAIRING_UNPROVED')
    require(metadata.get('within_trajectory_axes_paired') is True, 'AXIS_PAIRING')
    require(len(set(candidate_set)) == len(candidate_set) and candidate_set, 'CANDIDATE_SET')
    seen, seed_owner, ids, by = set(), {}, set(), {c: [] for c in candidate_set}
    for row in rows:
        require(row['candidate'] in by and row['phase'] == 'confirmation'
                and row['law'] == 'unconditional', 'NO_EXPLORATION_OR_CONDITIONAL_POOLING')
        key = (row['candidate'], row['trajectory_id'])
        require(key not in seen, 'DUPLICATE_TRAJECTORY_ACROSS_ENDPOINTS')
        seen.add(key)
        require(row['sample_id'] not in ids, 'DUPLICATE_SAMPLE_ID')
        ids.add(row['sample_id'])
        for axis in ('cosine', 'sine'):
            require(type(row[axis]) in (int, float) and math.isfinite(row[axis])
                    and row[axis] >= 0, 'MISSING_NONFINITE_COST_NOT_ZERO')
        require(row['event_digest_cosine'] == row['event_digest_sine'], 'AXIS_PAIRING')
        if metadata['cross_candidate_design'] == 'DISTINCT_STREAMS':
            require(row['seed'] not in seed_owner, 'SEED_REUSE_NOT_INDEPENDENT_DESIGN')
            seed_owner[row['seed']] = key
        by[row['candidate']].append(row)
    require(all(len(v) == expected_n for v in by.values()), 'CONFIRMATION_SAMPLE_COUNT')
    exploration = set(metadata.get('exploration_sample_ids', ()))
    require(not ids & exploration, 'EXPLORATION_CONFIRMATION_OVERLAP')
    require(not set(seed_owner) & set(metadata.get('exploration_seeds', ())), 'EXPLORATION_SEED_OVERLAP')
    return by


def paired_cost(row, shots):
    require(set(shots) == {'cosine', 'sine'} and all(type(n) is int and n >= 0
            for n in shots.values()), 'SAVED_SHOT_SCHEMA')
    return shots['cosine']*row['cosine'] + shots['sine']*row['sine']


def synthetic_estimands(by, shots, fixed_candidate):
    require(fixed_candidate in by and set(by) == set(shots), 'FIXED_CANDIDATE')
    summaries = {}
    for c, rows in by.items():
        values = [paired_cost(row, shots[c]) for row in rows]
        summaries[c] = dict(n=len(values), mean=statistics.mean(values),
                            SE=statistics.stdev(values)/math.sqrt(len(values)) if len(values)>1 else None)
    selected = min(summaries, key=lambda c: (summaries[c]['mean'], c))
    return dict(fixed_before_confirmation=fixed_candidate,
                fixed_candidate_confirmation=summaries[fixed_candidate],
                post_confirmation_point_selected=selected, summaries=summaries,
                selected_population_mean_certified=False, unobserved_s4_mean=None,
                selected_mean_interval=None)


def synthetic_cluster_bootstrap(by, endpoint_shots, *, replicates, seed, dependence):
    """Exploratory fixture bootstrap. Resample trajectory once per candidate/b.

    Both axes and ALL epsilon/allowance/bias endpoints reuse those indices.
    This does not estimate missing rare support, selection-adjusted coverage or
    a rigorous population winner probability. Independent-candidate mode needs
    documented distinct streams; no row-index pairing of unrelated candidates.
    """
    require(dependence == 'DISTINCT_STREAMS', 'BOOTSTRAP_DEPENDENCE_NOT_VERIFIED')
    require(type(replicates) is int and replicates > 0 and by, 'BOOTSTRAP_REPLICATES')
    streams = {c: random.Random(int(digest([seed, c]), 16)) for c in sorted(by)}
    out = []
    for _ in range(replicates):
        indices = {c: [streams[c].randrange(len(v)) for _ in v] for c, v in by.items()}
        out.append({endpoint: {c: statistics.mean(
            paired_cost(by[c][i], shots[c]) for i in indices[c]) for c in by}
            for endpoint, shots in endpoint_shots.items()})
    return dict(kind='SYNTHETIC_EXPLORATORY_BOOTSTRAP', replicates=out,
                tail_support_recovered=False, simultaneous_coverage_certified=False)


def bounded_simultaneous_radius(*, bound, n, family_size, alpha, support_proof):
    """Hoeffding + union bound, conditional on a real all-support proof.

    The observed maximum is NOT a support proof. No actual H8 invocation here.
    The union bound needs no cross-candidate independence; within-candidate IID
    samples and prespecified family/valid bound remain necessary.
    """
    require(support_proof == 'PROVEN_ALL_SUPPORT_BOUND', 'ALL_SUPPORT_BOUND_REQUIRED')
    require(type(bound) in (int, float) and math.isfinite(bound) and bound > 0
            and type(n) is int and n > 0 and type(family_size) is int
            and family_size > 0 and 0 < alpha < 1, 'INTERVAL_SCHEMA')
    return bound*math.sqrt(math.log(2*family_size/alpha)/(2*n))


def authorization_gate(config, grant):
    require(config.get('authorization_status') == 'APPROVED'
            and grant.get('valid') is True, 'EXPLICIT_SCIENTIFIC_EXECUTION_APPROVAL_REQUIRED')
    # Defense in depth: no production implementation is connected in this SOURCE.
    raise ValueError('PRODUCTION_DISPATCH_UNAVAILABLE_IN_READINESS_SOURCE')
