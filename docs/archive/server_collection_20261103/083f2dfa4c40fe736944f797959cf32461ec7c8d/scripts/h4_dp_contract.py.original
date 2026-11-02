"""H4 D/P metadata contracts. This module imports no scientific provider."""
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import random
import sys

EPS = (0.05, 0.01, 0.005, 0.001, 0.0005)
FIELDS = {'USUAL_SIGNED': 'precision_rows',
          'EMPIRICAL_NONCANCELLATION': 'noncancellation_precision_rows'}
AXES = ('cosine', 'sine')
SCHEMA = 'h4_dp_phase0_v1'


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda _: require(False, 'NONFINITE_JSON'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def saved_endpoint(record, mode, epsilon):
    """Copy an existing decision. Never recompute bias, headroom or shots."""
    value = record[FIELDS[mode]][str(epsilon)]['1.0']
    require(type(value['cost_acquisition_eligible']) is bool, 'UNRESOLVED_ELIGIBILITY')
    if value['cost_acquisition_eligible']:
        require(all(type(value['axes'][a]['shots']) is int and value['axes'][a]['shots'] > 0
                    for a in ('real', 'imag')), 'MISSING_SAVED_SHOTS')
    else:
        require(value['eligibility_under_declared_allowance'] == 'INELIGIBLE',
                'UNRESOLVED_IS_NOT_AN_EXCLUSION')
    return value


def deterministic_tasks(records, measured_ids):
    ids = [r['cell']['id'] for r in records]
    require(len(ids) == len(set(ids)), 'DUPLICATE_DOMAIN_ID')
    require(set(measured_ids) <= set(ids), 'MEASURED_ID_OUTSIDE_DOMAIN')
    eligible, excluded = [], []
    for record in records:
        decisions = [saved_endpoint(record, mode, eps)['cost_acquisition_eligible']
                     for mode in FIELDS for eps in EPS]
        (eligible if any(decisions) else excluded).append(record['cell']['id'])
    return dict(registered=ids, eligible=eligible, excluded=excluded,
                compile_ids=[x for x in eligible if x not in measured_ids])


def independent_seed(namespace, source, plan_digest, candidate, replica):
    require(namespace == 'H4_PR_INDEPENDENT_CONFIRMATION_V1', 'SEED_NAMESPACE')
    return int(digest([namespace, source, plan_digest, candidate, replica])[:16], 16)


def seed_hierarchy(master, q):
    # Original direct_master_to_step_seeds_v1; repeated q>1 consumes child seeds.
    if q == 1:
        return [master]
    rng = random.Random(master)
    return [master] + [rng.randrange(0, 2**63) for _ in range(q)]


def reserve_seed(existing, master, q):
    seeds = seed_hierarchy(master, q)
    require(len(set(seeds)) == len(seeds) and set(seeds).isdisjoint(existing), 'SEED_COLLISION')
    existing.update(seeds)
    return seeds


def check_pair(pair, compiler):
    require(set(pair['axes']) == set(AXES), 'MISSING_PAIRED_AXIS')
    for axis in AXES:
        row = pair['axes'][axis]
        require(row['seed'] == pair['seed'] and row['event_digest'] == pair['event_digest'],
                'UNPAIRED_TRAJECTORY')
        require(row['compiler_hash'] == compiler and len(row['fingerprint']) == 64,
                'COMPILE_IDENTITY')
        require(type(row['RZ']) is int and row['RZ'] > 0, 'INVALID_COMPILED_RZ')


def evaluate_pairs(pairs, shots, comparator, expected_direction):
    """Future authorized accounting, or artificial data tests only in Phase 0."""
    require(len(pairs) >= 2, 'NEED_COMPLETE_FIXED_N')
    require(set(shots) == set(AXES) and all(type(x) is int and x > 0 for x in shots.values()),
            'SAVED_SHOTS_REQUIRED')
    values = [sum(shots[a] * p['axes'][a]['RZ'] for a in AXES) for p in pairs]
    mean = Fraction(sum(values), len(values))
    variance = sum((Fraction(x) - mean)**2 for x in values) / (len(values) - 1)
    delta = Fraction(comparator) - mean
    direction = 1 if delta > 0 else -1 if delta < 0 else 0
    status = ('INCONCLUSIVE' if direction == 0 or expected_direction == 0 else
              'REPRODUCED' if direction == expected_direction else 'NOT_REPRODUCED')
    return dict(G_exact=str(mean), sample_variance_exact=str(variance),
                sample_SD=math.sqrt(variance), paired_SE=math.sqrt(variance / len(values)),
                point_direction=direction, status=status, population_mean_certified=False)


def normal_n_formula(sd, difference, z_family, z_power):
    """Artificial tests now; real saved inputs require separate S_STATS approval."""
    require(all(math.isfinite(x) and x > 0 for x in (sd, difference, z_family, z_power)),
            'INVALID_PLANNING_INPUT')
    return max(2, math.ceil(((z_family + z_power) * sd / difference)**2))


def within_candidate_covariance(pairs, shot_vectors):
    """Future P accounting or synthetic fixtures; no Phase 0 real-data caller."""
    require(len(pairs) >= 2, 'COVARIANCE_NEEDS_FIXED_N')
    weighted = [[sum(w[a] * p['axes'][a]['RZ'] for a in AXES) for p in pairs] for w in shot_vectors]
    means = [Fraction(sum(v), len(v)) for v in weighted]
    return [[str(sum((Fraction(x)-means[i])*(Fraction(y)-means[j])
                     for x, y in zip(weighted[i], weighted[j])) / (len(pairs)-1))
             for j in range(len(weighted))] for i in range(len(weighted))]


def event_stratum(events):
    orders = [e['taylor_order'] for step in events for e in step]
    require(all(type(x) is int and x in (0, 2, 4) for x in orders), 'UNREGISTERED_TAYLOR_ORDER')
    if any(x == 4 for x in orders):
        return 'S4_PRESENT'
    count = orders.count(2)
    return 'S0_ONLY' if count == 0 else 'S2_SINGLE' if count == 1 else 'S2_MULTI'


def boundary_attributes(cell, cells):
    result = {}
    for coordinate, other in [('prefix', 'q'), ('q', 'prefix')]:
        values = sorted({x[coordinate] for x in cells if x['method'] == cell['method']
                         and x['formula'] == cell['formula'] and x[other] == cell[other]})
        result[coordinate] = dict(registered_slice=values, position='SINGLETON_NOT_AN_EDGE_TEST' if len(values) == 1
            else 'LOWER' if cell[coordinate] == values[0] else 'UPPER' if cell[coordinate] == values[-1] else 'INTERIOR')
    result['formula_options'] = sorted({x['formula'] for x in cells if x['method'] == cell['method']
                                       and x['prefix'] == cell['prefix'] and x['q'] == cell['q']})
    result['physical_full_rank'] = cell['prefix'] == 12
    result['physical_q1'] = cell['q'] == 1
    return result


def validate_tasks(spec):
    require(len({t['id'] for t in spec['tasks']}) == len(spec['tasks']), 'DUPLICATE_TASK')
    if spec['stage'] == 'D':
        domain = read(spec['registered_tasks'])
        require({t['cell']['id'] for t in spec['tasks']} == set(domain['compile_ids'])
                and len(spec['tasks']) == len(domain['compile_ids']), 'EXACT_D_COMPILE_SCOPE')
        require(spec['calls']['wrapper_compile'] == 2*len(spec['tasks'])
                and spec['calls']['control_probe'] == 4*len(spec['tasks']), 'D_CALL_BUDGET')
        return
    registration = read(spec['registration'])
    signals = read(spec['signal_bindings'])
    preparations = read(spec['saved_preparation_identities'])['prefixes']
    cells = {x['id']: x for x in registration['candidates']}
    require(set(spec['n_fixed']) == set(cells), 'FIXED_P_CANDIDATES')
    expected_endpoints = {(x['cell_id'], x['epsilon']) for x in registration['assignments']}
    require({(x['cell_id'], x['epsilon']) for x in spec['evaluation_endpoints']} == expected_endpoints
            and len(spec['evaluation_endpoints']) == len(expected_endpoints), 'FIXED_P_ENDPOINTS')
    fixed_shots = {(x['cell_id'], x['epsilon']): x['saved_shots'] for x in registration['saved_statistics']}
    require(sha(spec['D_result_path']) == spec['D_result_sha256'], 'ACCEPTED_D_RESULT_CHANGED')
    d_winners = {r['epsilon']: r['best'] for r in read(spec['D_result_path'])['winners']
                 if r['bias_mode'] == 'USUAL_SIGNED'}
    for ep in spec['evaluation_endpoints']:
        require(ep['saved_shots'] == fixed_shots[(ep['cell_id'], ep['epsilon'])], 'FIXED_SAVED_P_SHOTS')
        winner = d_winners[ep['epsilon']]
        require(winner is not None and ep['PF_id'] == winner['cell_id']
                and ep['PF_G_exact'] == winner['G_exact'], 'COMPLETED_D_COMPARATOR_REQUIRED')
    old = set(read(spec['seed_exclusions'])['values'])
    for cid, n in spec['n_fixed'].items():
        require(type(n) is int and n >= 2, 'FIXED_P_N')
        tasks = [t for t in spec['tasks'] if t['cell']['id'] == cid]
        require({t['replica'] for t in tasks} == set(range(n)) and len(tasks) == n, 'FIXED_P_REPLICAS')
        for task in tasks:
            require(task['cell'] == cells[cid], 'P_CELL_CHANGED')
            require(task['saved_log_B'] == signals[cid]['log_B']
                    and task['log_B_tolerance'] == signals[cid]['log_B_tolerance']
                    and task['preparation_hash'] == preparations[str(task['cell']['prefix'])]['preparation_hash'],
                    'SAVED_P_SIGNAL_PREPARATION_BINDING')
            master = independent_seed('H4_PR_INDEPENDENT_CONFIRMATION_V1', spec['SOURCE'],
                                      spec['seed_plan_digest'], cid, task['replica'])
            require(task['seed'] == master, 'SEALED_CONFIRMATION_SEED')
            reserve_seed(old, master, task['cell']['q'])
    require(all(t['cell']['id'] in cells for t in spec['tasks']), 'EXTRA_P_TASK')
    require(spec['calls']['wrapper_compile'] == 2*sum(spec['n_fixed'].values()), 'P_CALL_BUDGET')


def authorize(spec, grant, root):
    # Check approval before touching science, output, lock, cache or raw trajectories.
    require(spec.get('status') == 'SEALED_FOR_EXECUTION' and spec.get('technical_sealed') is True,
            'DRAFT_UNAUTHORIZED')
    require(grant.get('approved_by_user') is True and grant.get('runtime_authorization') is True,
            'SEPARATE_EXPLICIT_SCIENTIFIC_APPROVAL_REQUIRED')
    require(spec['stage'] in ('D', 'P') and grant['stage'] == spec['stage'], 'STAGE_BINDING')
    require(grant.get('one_shot') is True and grant.get('retry') is False
            and grant.get('next_stage_authorized') is False, 'ONE_SHOT_STOP_REQUIRED')
    require(grant['spec_digest'] == digest(spec) and grant['SOURCE'] == spec['SOURCE'], 'CONTRACT_BINDING')
    require(len(spec['SOURCE']) == 40, 'FIXED_SOURCE_REQUIRED')
    receipt = Path(grant['approval_receipt'])
    require(sha(receipt) == grant['approval_receipt_sha256'], 'APPROVAL_RECEIPT_CHANGED')
    for name, h in spec['source_hashes'].items():
        require(sha(Path(root) / name) == h, 'SOURCE_CHANGED:' + name)
    for row in spec['inputs']:
        require(sha(row['path']) == row['sha256'], 'INPUT_CHANGED:' + row['role'])
    for path, h in spec['runtime_hashes'].items():
        require(sha(path) == h, 'RUNTIME_CHANGED')
    require(spec.get('runtime_hashes') and spec.get('provider_hashes'), 'COMPLETE_RUNTIME_BINDING_REQUIRED')
    for path, h in spec['provider_hashes'].items():
        require(sha(path) == h, 'PROVIDER_CHANGED')
    require(sha(sys.executable) == spec['python_sha256'], 'PINNED_PYTHON_REQUIRED')
    require(not Path(spec['output']).exists() and not Path(spec['lock']).exists(), 'USED_ONE_SHOT_NAMESPACE')
    require(type(spec.get('workers')) is int and spec['workers'] > 0, 'FRESH_RESOURCE_ADMISSION_REQUIRED')
    if spec['stage'] == 'P':
        require(spec.get('n_fixed') and spec.get('D_result_sha256') and spec.get('D_result_path')
                and spec.get('seed_collision_check_complete')
                and spec.get('stats_approval_receipt_sha256') and spec.get('seed_plan_digest')
                and spec.get('seed_exclusions'), 'P_PENDING_DEPENDENCIES')
        require(spec.get('frozen_stage_closure'), 'P_PROVIDER_CLOSURE_REQUIRED')
        for directory, files in spec['frozen_stage_closure'].items():
            for name, h in files.items():
                require(sha(Path(directory) / name) == h, 'P_FROZEN_KERNEL_CHANGED')
