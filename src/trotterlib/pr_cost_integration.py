"""Saved-evidence cost adapters. No quantum, sampling, fit or I/O at import.

The caller must pass a scoped execution permit before processing scientific
values. Artificial tests use an explicitly different data kind and permit.
Shot counts and normalization are inherited, never optimized by this module.
"""
from copy import deepcopy
from fractions import Fraction as F
import hashlib
import json
import math
import statistics

AXES = ('cosine', 'sine')
CHANNEL = dict(cosine='real', sine='imag')
EPS = (.05, .01, .005, .001, .0005)
MODES = ('USUAL_SIGNED', 'EMPIRICAL_NONCANCELLATION')
IDENTITY = ('system', 'geometry', 'basis', 'rank', 'rank_policy', 'T',
            'hamiltonian', 'state', 'snapshot', 'compiler', 'profile', 'control')


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def fraction(value):
    require(type(value) in (int, float, str), 'FINITE_NUMERIC_VALUE')
    if isinstance(value, float):
        require(math.isfinite(value), 'NONFINITE')
    result = F(value)
    return result


def same_binding(actual, expected):
    require(all(k in actual and k in expected for k in IDENTITY), 'INCOMPLETE_BINDING')
    require(all(actual[k] == expected[k] for k in IDENTITY), 'INPUT_COMPILER_CONTROL_BINDING')


def cell_key(cell):
    require(cell['method'] == 'B2' and cell['formula'] == '2nd', 'EXISTING_S2_PR_ONLY')
    require(all(type(cell[k]) is int and cell[k] > 0 for k in ('prefix', 'q', 'r', 'K', 'R')),
            'INTEGER_PR_PARAMETERS')
    require(cell['K'] in (2, 4) and cell['R'] == cell['q'] * cell['r'], 'RTE_LAW_IDENTITY')
    require(cell['id'] == f"B2_p{cell['prefix']}_q{cell['q']}_2nd_r{cell['r']}_K{cell['K']}",
            'CANDIDATE_ID')
    return tuple(cell[k] for k in ('id', 'prefix', 'q', 'r', 'K', 'R', 'formula'))


def acquisition_candidates(signals, registered):
    """Only read the saved endpoint flags; do not recompute scientific eligibility."""
    require(len(signals) == len(registered) and len({x['cell']['id'] for x in signals}) == len(signals),
            'COMPLETE_UNIQUE_A_SIGNAL')
    by = {x['id']: cell_key(x) for x in registered}
    selected, excluded = [], []
    for row in signals:
        cell = row['cell']
        require(cell['id'] in by and cell_key(cell) == by[cell['id']], 'A_SIGNAL_CELL_BINDING')
        fields = ('precision_rows', 'noncancellation_precision_rows' if
                  'noncancellation_precision_rows' in row else 'empirical_noncancellation_precision_rows')
        flags = []
        for field in fields:
            for e in EPS:
                saved = row[field][str(e)]['1.0']
                flag = saved['cost_acquisition_eligible']
                require(type(flag) is bool, 'SAVED_ELIGIBILITY_FLAG')
                flags.append(flag)
        (selected if any(flags) else excluded).append(deepcopy(cell))
    return dict(selected=selected, ineligible_at_all_registered_endpoints=excluded,
                endpoint_rule='saved five epsilons, both bias modes, allowance factor 1',
                recalculated_eligibility=False)


class Permit:
    def __init__(self, stage, *, synthetic, authorization=None):
        require(stage in ('A_COST', 'D_ACCOUNTING', 'B_ANALYSIS'), 'SCOPED_STAGE')
        require(synthetic is True or authorization is not None, 'SCOPED_AUTHORIZATION_REQUIRED')
        self.stage, self.synthetic, self.authorization = stage, synthetic, authorization

    def check(self, stage, kind):
        require(kind in ('ARTIFICIAL','SAVED_REAL_VALUES'), 'DECLARED_DATA_KIND')
        require(self.stage == stage and self.synthetic == (kind == 'ARTIFICIAL'), 'DATA_SCOPE_PERMIT')


def choose_unit_cost(evidence, binding, candidate, axis):
    require(axis in AXES, 'AXIS')
    matched = []
    for item in evidence:
        if item['cell_id'] != candidate or item['axis'] != axis:
            continue
        same_binding(item['binding'], binding)
        require(item.get('SOURCE') and item.get('result_sha256'), 'COST_SOURCE_RESULT_PROVENANCE')
        require(item.get('preparation_identity') and item.get('scalar_policy') == 'original_plus_tail_once',
                'PREPARATION_SCALAR_PROVENANCE')
        kind = item['kind']
        require(kind in ('DIRECT_MEASURED', 'FIXED_MODEL', 'CONDITIONAL'), 'COST_KIND')
        if kind == 'CONDITIONAL':
            continue
        value = fraction(item['RZ'])
        require(value > 0, 'POSITIVE_RZ')
        if kind == 'DIRECT_MEASURED':
            require(value.denominator == 1 and item.get('native_exit_code') == 0
                    and item.get('reaped') is True and item.get('final_gate') == 'PASS',
                    'DIRECT_MEASUREMENT_ACCEPTANCE')
        else:
            require(item.get('model_sha256') and item.get('model_validation') == 'ACCEPTED_FIXED_MODEL',
                    'FIXED_VALIDATED_MODEL_REQUIRED')
        matched.append(item)
    for kind in ('DIRECT_MEASURED', 'FIXED_MODEL'):
        selected = [x for x in matched if x['kind'] == kind]
        if selected:
            require(len({fraction(x['RZ']) for x in selected}) == 1, 'CONFLICTING_COST_EVIDENCE')
            return deepcopy(sorted(selected, key=lambda x: (x['SOURCE'], x['result_sha256']))[0])
    return None


def saved_shots(accounting):
    """Use the exact acceptance/shot projection already used by the H8 reaccounting."""
    # That existing module is script-loadable from src/trotterlib, not a package import.
    from h8_six_axis_reaccount import shots_from_accounting
    eligible = accounting.get('cost_acquisition_eligible')
    require(type(eligible) is bool, 'SAVED_ELIGIBILITY_FLAG')
    values = shots_from_accounting(accounting, eligible)
    return None if values is None else {a: values[CHANNEL[a]] for a in AXES}


def paired_statistics(pairs, shots, *, cohort):
    require(cohort in ('EXPLORATION_N8', 'CONFIRMATION_N32', 'CONDITIONAL_N8'), 'COST_COHORT')
    expected = 32 if cohort == 'CONFIRMATION_N32' else 8
    require(len(pairs) == expected, 'EXACT_COHORT_SIZE')
    require(set(shots) == set(AXES) and all(type(x) is int and x > 0 for x in shots.values()), 'INTEGER_SHOTS')
    replicas, seeds, totals = set(), set(), []
    identity = None
    for pair in pairs:
        require(pair['cohort'] == cohort, 'N8_N32_CONDITIONAL_MIXING_FORBIDDEN')
        require(set(pair['axes']) == set(AXES), 'PAIRED_AXES_REQUIRED')
        require(type(pair['replica']) is int and 0 <= pair['replica'] < expected
                and pair['replica'] not in replicas, 'UNIQUE_REPLICA')
        replicas.add(pair['replica'])
        total = 0
        for axis in AXES:
            item = pair['axes'][axis]
            require(item['seed'] == pair['seed'] and item['event_digest'] == pair['event_digest'],
                    'PAIRED_TRAJECTORY_IDENTITY')
            require(type(item['RZ']) is int and item['RZ'] > 0, 'MEASURED_INTEGER_RZ')
            total += shots[axis] * item['RZ']
        require(pair['seed'] not in seeds, 'UNIQUE_TRAJECTORY_SEED')
        seeds.add(pair['seed'])
        key = (pair['cell_id'], digest(pair['binding']))
        require(identity is None or identity == key, 'CANDIDATE_OR_INPUT_MIXING')
        identity = key
        totals.append(total)
    require(replicas == set(range(expected)), 'COMPLETE_REPLICA_COVERAGE')
    return dict(mean_exact=str(F(sum(totals), expected)),
                sample_SE=statistics.stdev(totals) / math.sqrt(expected), n=expected,
                cohort=cohort, finite_sample_only=True, population_mean_certified=False)


def account_rows(data, permit):
    """Cost-only update: preserve saved shots, bias, allowance and normalization."""
    permit.check('D_ACCOUNTING', data['kind'])
    before = digest(data)
    results = []
    keys = set()
    for source in data['saved_accounting']:
        key = (source['cell_id'], source['bias_mode'], source['epsilon'], source['factor'])
        require(key not in keys, 'DUPLICATE_ACCOUNTING_ENDPOINT'); keys.add(key)
        require(source['bias_mode'] in MODES and source['epsilon'] in EPS and source['factor'] in (1., 10.),
                'FIXED_BIAS_EPSILON_ALLOWANCE_DOMAIN')
        same_binding(source['binding'], data['binding'])
        row = deepcopy(source)
        row.update(G_RZ_exact=None, G_RZ_SE=None, cost_kind='MISSING', axis_evidence=None)
        accounting = source['accounting']
        if accounting is None:
            row['status']='ACCOUNTING_MISSING';results.append(row);continue
        shots = saved_shots(accounting)
        if shots is None:
            row['status'] = ('INELIGIBLE' if accounting.get('eligibility_under_declared_allowance')=='INELIGIBLE'
                             else 'NUMERICALLY_UNRESOLVED')
        elif source['method'] == 'B2':
            pairs = data.get('paired_PR', {}).get(source['cell_id'])
            if pairs is None:
                row['status'] = 'ELIGIBLE_COST_MISSING'
            else:
                require(source['cohort'] != 'CONDITIONAL_N8', 'CONDITIONAL_NOT_UNCONDITIONAL')
                for pair in pairs:
                    same_binding(pair['binding'], source['binding'])
                value = paired_statistics(pairs, shots, cohort=source['cohort'])
                row.update(status='FINITE_SAMPLE_POINT', G_RZ_exact=value['mean_exact'],
                           G_RZ_SE=value['sample_SE'], cost_kind=source['cohort'], n=value['n'])
        else:
            costs = {a: choose_unit_cost(data['cost_evidence'], source['binding'], source['cell_id'], a)
                     for a in AXES}
            if any(value is None for value in costs.values()):
                row['status'] = 'ELIGIBLE_COST_MISSING'
            else:
                total = sum(shots[a] * fraction(costs[a]['RZ']) for a in AXES)
                model = any(v['kind'] == 'FIXED_MODEL' for v in costs.values())
                row.update(status='MODEL_CONDITIONAL_POINT' if model else 'DIRECT_MEASURED_POINT',
                           G_RZ_exact=str(total), axis_evidence=costs,
                           cost_kind='MODEL_CONDITIONAL' if model else 'DIRECT_MEASURED')
        results.append(row)
    require(digest(data) == before, 'INPUT_MUTATION')
    return results


def compare(rows, deterministic_ids, pr_ids):
    """Report available-set minima; missing eligible costs prevent a full-set claim."""
    require(set(deterministic_ids).isdisjoint(pr_ids), 'SEPARATE_PR_DETERMINISTIC_SET')
    ids = deterministic_ids + pr_ids
    require(len(ids) == len(set(ids)), 'UNIQUE_COMPARISON_CANDIDATES')
    require(set(ids) == {r['cell_id'] for r in rows} and len(rows) == len(ids), 'COMPLETE_COMPARISON_ROWS')
    require(len({(r['bias_mode'], r['epsilon'], r['factor'], digest(r['binding'])) for r in rows}) == 1,
            'SAME_ENDPOINT_INPUT_COMPARISON')
    missing = [r['cell_id'] for r in rows if r['status'] in
               ('ELIGIBLE_COST_MISSING','NUMERICALLY_UNRESOLVED','ACCOUNTING_MISSING')]
    def best(candidates):
        available = [r for r in rows if r['cell_id'] in candidates and r['G_RZ_exact'] is not None]
        return min(available, key=lambda r: (fraction(r['G_RZ_exact']), r['cell_id'])) if available else None
    d, p = best(deterministic_ids), best(pr_ids)
    reduction = None if d is None or p is None or missing else str(100 * (1-fraction(p['G_RZ_exact'])/fraction(d['G_RZ_exact'])))
    return dict(best_available_deterministic=deepcopy(d), best_available_PR=deepcopy(p),
                full_registered_set_PR_reduction_percent_exact=reduction,
                missing_eligible_costs=missing, comparison_complete=not missing,
                all_candidates_direct_measured=False, population_mean_certified=False,
                strict_bias_certificate=None)


def conditional_diagnostic(data, permit):
    """A conditional mean alone cannot replace an unconditional trajectory mean."""
    permit.check('B_ANALYSIS', data['kind'])
    groups = data['groups']
    require(len({x['stratum'] for x in groups}) == len(groups), 'DISJOINT_UNIQUE_STRATA')
    for group in groups:
        same_binding(group['binding'], data['binding'])
        require(group['cell_id'] == data['cell_id'] and group['law_id'] == data['law_id'], 'SAME_CANDIDATE_ORIGINAL_LAW')
        require(group['cohort'] == 'CONDITIONAL_N8', 'CONDITIONAL_COHORT_REQUIRED')
    partition = data.get('partition_proof')
    required = set(data['required_strata'])
    found = {x['stratum'] for x in groups}
    if partition is None or required != found or any(x.get('probability') is None or
                                                   x.get('mean_RZ') is None for x in groups):
        return dict(status='UNRESOLVED_STRATIFIED_MEAN', unconditional_mean_RZ=None,
                    missing_strata=sorted(required-found), ordinary_sample_unchanged=True,
                    reason='Need complete disjoint law partition, original masses and conditional costs',
                    population_mean_certified=False)
    require(partition['law_id'] == data['law_id'] and partition['disjoint_exhaustive'] is True,
            'PARTITION_PROVENANCE')
    require(set(partition['strata']) == required and partition.get('SOURCE') and partition.get('sha256'),
            'EXHAUSTIVE_PARTITION_PROOF')
    masses = [fraction(x['probability']) for x in groups]
    require(all(0 <= x <= 1 for x in masses) and sum(masses) == 1, 'EXACT_ORIGINAL_MASS_CLOSURE')
    require(all(x.get('mass_kind') == 'ORIGINAL_RTE_LAW' for x in groups), 'NO_CONDITIONAL_RENORMALIZATION')
    costs = [fraction(x['mean_RZ']) for x in groups]
    require(all(x > 0 for x in costs), 'NO_MISSING_COST_ZERO_FILL')
    return dict(status='LAW_WEIGHTED_FINITE_SAMPLE_POINT',
                unconditional_mean_RZ_exact=str(sum(p*c for p, c in zip(masses, costs))),
                unconditional_SE=None, SE_reason='stratum covariance/independence proof not supplied',
                ordinary_sample_unchanged=True, population_mean_certified=False)
