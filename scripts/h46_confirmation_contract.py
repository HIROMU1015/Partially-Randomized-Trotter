"""Authorization and identities for the proposed H6 higher-PF confirmation."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PHASE = 'H6_HIGHER_PF_C_INDEPENDENT'
GROUPS = (
    {'id': 'H6_S2_q8', 'system': 'H6', 'label': '2nd', 'q': 8, 'prefix': 5},
    {'id': 'H6_Y4_q2', 'system': 'H6', 'label': '4th', 'q': 2, 'prefix': 5},
    {'id': 'H6_Y4_q4', 'system': 'H6', 'label': '4th', 'q': 4, 'prefix': 5})
PROPOSED_N = {'H6_S2_q8': 13, 'H6_Y4_q2': 45, 'H6_Y4_q4': 45}
INPUT_SHA = '99440a59d903dfd6330e786d84a956f1dd8a687a592295a1cd771c839769b005'
PAIRS = (('H6_S2_q8', 'H6_Y4_q2'), ('H6_Y4_q4', 'H6_Y4_q2'))
PROFILE = '1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b'
KERNEL_SHA = 'ccc71379739c243e86024e897643220ae67fa0b3409d8d8be4d8193183228308'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')


def task_id(plan, group, sample):
    if group not in list(GROUPS) or type(sample) is not int or not 0 <= sample < plan['n_per_candidate'][group['id']]:
        raise ValueError('CONFIRMATION_TASK_OUTSIDE_FIXED_SCOPE')
    return group['id'] + '/sample_%03d' % sample


def scope(plan):
    if plan.get('phase') != PHASE or plan.get('study_id') != 'HIGHER_PF_PURPOSE_A_REPLICATION':
        raise ValueError('SEPARATE_HIGHER_PF_STUDY_REQUIRED')
    if plan.get('groups') != list(GROUPS) or plan.get('n_per_candidate') != PROPOSED_N:
        raise ValueError('FIXED_THREE_CANDIDATES_AND_PROPOSED_N_REQUIRED')
    if any(type(v) is not int for v in plan['n_per_candidate'].values()): raise ValueError('INTEGER_N_REQUIRED')
    if plan.get('comparison_pairs') != [list(pair) for pair in PAIRS]: raise ValueError('FIXED_COMPARISON_FAMILY')
    if plan.get('compiler_settings_hash') != PROFILE or plan.get('numeric_tolerance') != 1e-9:
        raise ValueError('FROZEN_COMPILER_OR_CONTROL')
    if plan.get('fixed_science') != {'T': .8, 'K': 4, 'RTE_steps_per_outer': 32,
                                     'epsilon': .001, 'alpha_axis': .025, 'allowance': 1e-9,
                                     'log_B_margin': 1e-12, 'P': 0, 'control': 'ordinary'}:
        raise ValueError('FROZEN_SCIENTIFIC_CONDITIONS')
    if plan.get('purpose') != 'INDEPENDENT_POINT_ORDER_REPLICATION_NOT_POPULATION_GUARANTEE':
        raise ValueError('PURPOSE_OR_GUARANTEE_SCOPE')
    if plan.get('analysis_policy') != {'primary': 'EXACT_MEAN_DIFFERENCE_SIGN',
                                       'secondary': 'NORMAL_BONFERRONI_APPROX_DIAGNOSTIC',
                                       'MC_family_alpha': .05, 'family_size': 2,
                                       'population_guarantee': False, 'pool_exploration': False}:
        raise ValueError('FIXED_ANALYSIS_POLICY')
    tasks = [task_id(plan, group, i) for group in GROUPS for i in range(PROPOSED_N[group['id']])]
    if plan.get('tasks') != tasks or len(set(tasks)) != 103: raise ValueError('FIXED_TASK_REGISTRY')
    if (plan.get('trajectories'), plan.get('sample_axes'), plan.get('extra_scientific_compiles')) != (103, 206, 0):
        raise ValueError('FIXED_COMPILE_BUDGET')
    if plan.get('seed_scope_digest') != digest(plan.get('seed_scope_payload')):
        raise ValueError('SEED_SCOPE_DIGEST')
    expected_seed_scope = {'study_id': plan['study_id'], 'stage': 'CONFIRMATION',
                           'SOURCE': plan.get('SOURCE'),
                           'groups': list(GROUPS), 'n_per_candidate': PROPOSED_N,
                           'input_sha256': plan.get('input_sha256'),
                           'selection_evidence_sha256': plan.get('selection_evidence_sha256')}
    if plan.get('seed_scope_payload') != expected_seed_scope: raise ValueError('SEED_SCOPE_IDENTITY')
    if plan.get('input_sha256') != INPUT_SHA: raise ValueError('FIXED_H6_INPUT')
    if plan.get('retry') is not False or plan.get('mandatory_STOP') is not True:
        raise ValueError('ONE_SHOT_STOP_REQUIRED')
    return tasks


def registry_audit(plan, registry, excluded):
    if registry.get('stage') != 'CONFIRMATION' or registry.get('seed_scope_digest') != plan['seed_scope_digest']:
        raise ValueError('CONFIRMATION_SEED_STAGE_SCOPE')
    if registry.get('namespace') != plan['seed_namespace'] or not registry['namespace'].startswith('H6_HIGHER_PF_C_CONFIRMATION:'):
        raise ValueError('SEPARATE_SEED_NAMESPACE')
    if set(registry.get('tasks', {})) != set(plan['tasks']): raise ValueError('SEED_TASK_REGISTRY')
    expected_namespace = 'H6_HIGHER_PF_C_CONFIRMATION:' + plan['SOURCE'] + ':' + plan['seed_scope_digest']
    if registry['namespace'] != expected_namespace: raise ValueError('SOURCE_BOUND_SEED_NAMESPACE')
    values = []
    for group in GROUPS:
        pieces = 1 if group['label'] == '2nd' else 3
        expected = {f'{outer}:{piece}' for outer in range(group['q']) for piece in range(pieces)}
        for sample in range(PROPOSED_N[group['id']]):
            seeds = registry['tasks'][task_id(plan, group, sample)]
            if set(seeds) != expected: raise ValueError('SEED_OCCURRENCE_REGISTRY')
            for occurrence, value in seeds.items():
                outer, piece = map(int, occurrence.split(':'))
                payload = {'version': 'HIGHER_PF_INDEPENDENT_V1', 'campaign': registry['namespace'],
                           'stage': 'CONFIRMATION', 'candidate': group['id'], 'replica': sample,
                           'outer': outer, 'piece': piece, 'input_sha256': INPUT_SHA}
                # Same frozen seed_for() algorithm; no event sampling occurs here.
                if value != int(digest(payload)[:16], 16): raise ValueError('DETERMINISTIC_SEED_DERIVATION')
            values.extend(seeds.values())
    if any(type(v) is not int or not 0 <= v < 2**64 for v in values): raise ValueError('UINT64_SEEDS_REQUIRED')
    if len(values) != 914 or len(set(values)) != len(values): raise ValueError('DUPLICATE_CONFIRMATION_SEEDS')
    if set(values) & set(excluded): raise ValueError('EXPLORATION_CONFIRMATION_SEED_COLLISION')
    return {'fresh_values': len(values), 'excluded_unique_values': len(set(excluded)),
            'collisions': 0, 'PRNG_independence_proved': False}


def authorize(config):
    plan, grant = config.get('plan', {}), config.get('grant', {})
    # This is deliberately before SOURCE/input/seed reads or any native import.
    if not config.get('authorized') or plan.get('status') != 'SEALED_APPROVED' or grant.get('valid') is not True:
        raise ValueError('EXPLICIT_SAMPLING_COMPILE_APPROVAL_REQUIRED')
    scope(plan)
    if grant.get('phase') != PHASE or grant.get('SOURCE') != config.get('SOURCE') or plan.get('SOURCE') != config.get('SOURCE'):
        raise ValueError('CONFIRMATION_GRANT_SOURCE_SCOPE')
    if not grant.get('approval_reference') or not grant.get('grant_id'):
        raise ValueError('EXPLICIT_APPROVAL_REFERENCE_REQUIRED')
    if plan.get('sample_size_formally_adopted') is not True: raise ValueError('SAMPLE_SIZE_ADOPTION_REQUIRED')
    if config.get('provider_SOURCE') != 'c31ef424175c02379eafebe8fc6c7548c1d9aabe':raise ValueError('FROZEN_PROVIDER_SOURCE')
    if Path(sys.executable).resolve()!=Path(config['python']).resolve() or sys.version_info[:3]!=(3,12,3):
        raise ValueError('FIXED_PYTHON_3_12_3_REQUIRED')
    from importlib.metadata import version
    if {p:version(p) for p in ('qiskit','numpy','scipy','openfermion')}!={'qiskit':'1.3.0','numpy':'1.26.4','scipy':'1.14.1','openfermion':'1.6.1'}:
        raise ValueError('FROZEN_PACKAGE_PROFILE')
    if Path(config['source_root']).resolve() != ROOT: raise ValueError('SEALED_SOURCE_ROOT')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if head != config['SOURCE']: raise ValueError('EXACT_SOURCE_CHECKOUT_REQUIRED')
    if sha(config['plan_path']) != config['plan_sha256'] or grant.get('plan_sha256') != config['plan_sha256']:
        raise ValueError('GRANT_PLAN_DIGEST')
    if json.loads(Path(config['plan_path']).read_text()) != plan: raise ValueError('INLINE_PLAN_MUST_MATCH_SEALED_FILE')
    if not config.get('sealed_files') or not config.get('excluded_seed_path'): raise ValueError('SOURCE_INPUT_SEED_SEAL_REQUIRED')
    if digest(config['sealed_files']) != plan.get('closure_digest'): raise ValueError('SEALED_CLOSURE_BINDING')
    if digest(config.get('inputs')) != plan.get('input_bindings_digest') or digest(config.get('group_binding')) != plan.get('group_binding_digest'):
        raise ValueError('INPUT_OR_SAVED_SHOT_BINDING')
    if config.get('provider_SOURCE') != plan.get('provider_SOURCE'): raise ValueError('PROVIDER_SOURCE_BINDING')
    if config['inputs']['H6_snapshot.npz']['sha256'] != INPUT_SHA: raise ValueError('SNAPSHOT_BINDING')
    required = {str(Path(v['path']).resolve()) for v in config['inputs'].values()}
    required |= {str(Path(config['seed_registry_path']).resolve()), str(Path(config['excluded_seed_path']).resolve())}
    if not required <= {str(Path(e['path']).resolve()) for e in config['sealed_files']}:
        raise ValueError('REQUIRED_INPUT_AND_SEED_FILES_NOT_SEALED')
    for key, value in [('maximum_trajectories',103), ('maximum_cost_axes',206), ('maximum_extra_compiles',0), ('retry',False)]:
        if grant.get(key) != value: raise ValueError('GRANT_BUDGET')
    for key in ('output', 'call_ledger', 'grant_marker'):
        if not config.get(key) or grant.get(key) != config[key]: raise ValueError('OWN_EXECUTION_PATH_BINDING')
    if len({str(Path(config[k]).resolve()) for k in ('output','call_ledger','grant_marker')}) != 3:
        raise ValueError('DISTINCT_OWN_EXECUTION_PATHS')
    for ent in config['sealed_files']:
        if sha(ent['path']) != ent['sha256']: raise ValueError('SOURCE_INPUT_RUNTIME_HASH')
    for item in config['inputs'].values():
        if sha(item['path'])!=item['sha256']:raise ValueError('DECLARED_INPUT_HASH')
    if sha(ROOT/'scripts/h46_acquire_native.py') != KERNEL_SHA: raise ValueError('FROZEN_NATIVE_KERNEL_CHANGED')
    if sha(config['seed_registry_path']) != config['seed_registry_sha256'] or plan.get('seed_registry_sha256') != config['seed_registry_sha256']:
        raise ValueError('SEED_REGISTRY_DIGEST')
    registry = json.loads(Path(config['seed_registry_path']).read_text())
    exclusions = json.loads(Path(config['excluded_seed_path']).read_text())
    return registry_audit(plan, registry, exclusions['values'])
