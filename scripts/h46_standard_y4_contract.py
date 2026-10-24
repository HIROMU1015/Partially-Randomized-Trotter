"""One-shot standard-Y4 confirmation bindings; imports no numerical runtime.

The frozen c028 pipeline implements the science. This module binds its inputs,
new namespace, fixed counts and a separate saved-shot evaluation to one approval.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
OLD_SOURCE = 'c028f1c2885afdba65ef8c62c075738aadd7323d'
RESULT_SOURCE = '3eca46148070ba4d4b0f5815ef45943407cc959e'
PROVIDER_SOURCE = '217c971dd537b1cc7fbc13742bc0f6c263ff7f45'
BASE = 'eebdbfcbd0d5f5608d50bfd91c9cd526c68f0d23'
STAGES = ('H4_CONFIRMATION', 'H6_CONFIRMATION')
SERIES = 'H46_STANDARD_Y4_Q4_INDEPENDENT_20261024_V1'
SOURCE_PATHS = ('scripts/h46_standard_y4_contract.py',
    'scripts/h46_standard_y4_runtime.py', 'scripts/h46_standard_y4_evaluation.py',
    'scripts/prepare_h46_standard_y4.py', 'scripts/run_h46_standard_y4.py',
    'tests/test_h46_standard_y4.py')
GROUPS = {
    'H4': dict(rank=10, prefix=3, n=45, n_s2=25,
        S2='M5_MIN_72b305020499dd68', Y4='M5_MIN_ab57ad5bd40dc583',
        N_S2=[24249726, 18445335], N_Y4=[19240638, 18807773],
        preparation='317662c0ff954eefa5808bed84fa1bc5cbcd2e803383c62503fff8492b7b3508'),
    'H6': dict(rank=19, prefix=5, n=32, n_s2=23,
        S2='M5_MIN_a08e8bfb6642eb93', Y4='M5_MIN_37ff3428bb58f5cd',
        N_S2=[22907931, 27494282], N_Y4=[21115055, 21415717],
        preparation='8f1a392f8ab2bcb6cf59cad513b96f15ff518cb6dc237233410b03bb14567938')}
COMPILE_TOLERANCE, OTHER_TOLERANCE = 1e-7, 1e-9
d = SimpleNamespace(COHORT='CONFIRMATION')
_assets = None


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def save(path, value):
    """Durable create-only metadata; used for new outputs only."""
    path = Path(path)
    partial = path.with_name(path.name + '.partial')
    with partial.open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.link(partial, path)
    partial.unlink()


def seal(path, role):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size, role=role)


def install_paths(assets, site_packages=None):
    """Private recovered modules before unrelated checkout modules; no execution."""
    global _assets, law, storage
    assets = Path(assets).resolve()
    require(_assets is None or _assets == assets, 'ONE_PRIVATE_CLOSURE_PER_PROCESS')
    _assets = assets
    paths = [str(assets / 'frozen/scripts'), str(ROOT / 'scripts'),
             str(assets / 'provider/src')]
    if site_packages:
        paths.append(str(Path(site_packages).resolve()))
    sys.path[:] = paths + [p for p in sys.path if p not in paths]
    import h46_pr_design_pilot
    import h46_pr_pilot_storage_runtime
    law, storage = h46_pr_design_pilot, h46_pr_pilot_storage_runtime


def tasks_for(system, candidate):
    require(system in GROUPS, 'H4_H6_ONLY')
    spec = GROUPS[system]
    require(candidate['id'] == spec['Y4'] and candidate['system'] == system and
        candidate['formula'] == '4th' and candidate['q'] == 4 and
        candidate['prefix'] == spec['prefix'] and candidate['K'] == 4 and
        candidate['method'] == 'PR' and candidate['r_policy'] == 'ABS_TIME_PER_OUTER'
        and candidate['r_value'] == 32, 'STANDARD_Y4_EXACT_CANDIDATE')
    return [dict(id='SY4C1__' + candidate['id'] + '__' + str(i),
        candidate=copy.deepcopy(candidate), sample=i, axes=['cosine', 'sine'],
        stage=system + '_CONFIRMATION') for i in range(spec['n'])]


def analysis_scope():
    return dict(epsilon=.001, alpha_axis=.025, mode='USUAL_SIGNED', P=0,
        allowance_multiplier=1, pooling=False, adaptive_top_up=False,
        paired_within_trajectory=True, cross_formula_independent=True,
        family_size=2, family_alpha=.05,
        interval='NORMAL_BONFERRONI_APPROXIMATE_NOT_CERTIFIED',
        rare_cost_imputation=False, new_shots=0, ranking=0, retry=False,
        conditional_on_saved_signal_bias_normalization_shots=True,
        population_guarantee=False, mandatory_STOP=True)


def context_scope():
    # Boundaries are explicit; internal matrix/diagonal work belongs to these
    # unchanged functions, not to a fictitious zero-simulation setup claim.
    return dict(bootstrap=1, setup=1, saved_snapshot_restore=1,
        saved_reference_read=1, occupation_oracle_initialization=1,
        gaussian_prepare=2, matrix_only_tail_coefficients=1, tail_spectrum=1)


def evaluation_scope():
    return dict(bootstrap=1, bundle_load=1, family_constant=1,
        paired_statistics=4, direction_comparison=2)


def task_limits(task):
    s = law.call_scope([task], 'PR')
    return {'TASK': 1, **{k: s[k] for k in ('trajectories',
        'request_sampling_calls', 'native_piece_compositions', 'native_evolution',
        'native_low_actions', 'native_high_actions', 'independent_oracle_actions',
        'uncompiled_wrapper_actions', 'compiled_wrapper_actions', 'compile_axes')}}


def manifest():
    return read(_assets / 'frozen/references/h46_m5_cost_readiness_v1.json')


def check_files(rows):
    require(len(rows) == len({r['path'] for r in rows}), 'UNIQUE_INPUT_PATHS')
    for r in rows:
        p = Path(r['path'])
        require(str(p.resolve()) == r['path'] and p.stat().st_size == r['bytes']
            and sha(p) == r['sha256'], 'SEALED_SHA:' + r['role'])


def verify_package(package_path, *, check_closure=True):
    p = read(package_path)
    require(p['schema'] == SERIES and p['base'] == BASE and
        p['old_SOURCE'] == OLD_SOURCE and p['analysis_scope'] == analysis_scope()
        and p['evaluation_scope'] == evaluation_scope(), 'EXACT_PACKAGE_POLICY')
    require(p['n'] == {s: g['n'] for s, g in GROUPS.items()} and
        p['reused_S2_n'] == {s: g['n_s2'] for s, g in GROUPS.items()}, 'FIXED_N')
    require(p['scientific_execution_authorized'] is False and p['valid_grants'] == 0,
        'IMMUTABLE_DISABLED_PACKAGE')
    require(len(p['stages']) == 2 and {r['stage'] for r in p['stages']} == set(STAGES),
        'EXACT_TWO_NEW_GROUPS')
    for r in p['stages']:
        require(sha(r['plan_path']) == r['plan_sha256'], 'STAGE_PLAN_SHA')
        body = read(r['plan_path'])
        require(body['SOURCE'] == p['SOURCE'] and body['input_manifest_sha256'] ==
            p['input_manifest_sha256'], 'PACKAGE_STAGE_BINDING')
    require(sha(p['input_manifest_path']) == p['input_manifest_sha256'], 'INPUT_MANIFEST_SHA')
    if check_closure:
        check_files(read(p['input_manifest_path'])['files'])
        for rel in SOURCE_PATHS:
            require((ROOT / rel).read_bytes() == subprocess.check_output(
                ['git', 'show', p['SOURCE'] + ':' + rel], cwd=ROOT),
                'COMMITTED_NEW_SOURCE:' + rel)
    return p


def verify(cfg):
    body = read(cfg['plan_path'])
    require(sha(cfg['plan_path']) == cfg['plan_sha256'] and cfg['plan'] == body,
        'IMMUTABLE_STAGE_PLAN')
    require(all(cfg.get(k) == v for k, v in body.items()), 'ALL_STAGE_BINDINGS')
    p = verify_package(cfg['package_path'])
    require(any(r['plan_path'] == cfg['plan_path'] and r['plan_sha256'] ==
        cfg['plan_sha256'] and r['stage'] == cfg['stage'] for r in p['stages']),
        'REGISTERED_STAGE_ONLY')
    system = cfg['stage'][:2]
    require(cfg['stage'] in STAGES and cfg['cohort'] == 'CONFIRMATION' and
        cfg['analysis_scope'] == analysis_scope(), 'CONFIRMATION_COHORT_POLICY')
    require(cfg['tasks'] == tasks_for(system, cfg['tasks'][0]['candidate']) and
        cfg['call_scope'] == law.call_scope(cfg['tasks'], 'PR') and
        cfg['context_scope'] == context_scope(), 'EXACT_CALLS_AND_TASKS')
    require(cfg['systems'][system]['binding'] == cfg['binding'] and
        cfg['binding']['rank'] == GROUPS[system]['rank'] and
        cfg['preparation_hash'] == GROUPS[system]['preparation'], 'SYSTEM_RANK_PREPARATION')
    require(cfg['compiler_settings_hash'] == law.COMPILER_HASH and
        cfg['storage'] == storage.storage_policy() and
        cfg['compile_tolerance'] == COMPILE_TOLERANCE and
        cfg['other_tolerance'] == OTHER_TOLERANCE, 'UNCHANGED_NATIVE_COMPILER')
    require(cfg['runtime']['numpy'] == '1.26.4' and cfg['runtime']['qiskit'] == '1.3.0'
        and cfg['runtime']['provider_SOURCE'] == PROVIDER_SOURCE, 'RUNTIME_BINDING')
    check_files(cfg['sealed_files'])
    excluded = read(cfg['excluded_seeds_path'])['values']
    require(read(cfg['registry_path']) == law.registry(cfg['tasks'], cfg['SOURCE'],
        cfg['campaign_id'], 'CONFIRMATION', excluded), 'REGISTRY_SOURCE_NAMESPACE')
    require(cfg['complete_historical_seed_noncollision_guaranteed'] is False,
        'NO_INVENTED_COMPLETE_SEED_COVERAGE')
    root = Path(cfg['run_root'])
    require(root == root.resolve() and root.is_relative_to(_assets) and
        not root.is_relative_to(ROOT), 'OWN_DISJOINT_OUTPUT')
    for k, name in [('output', 'output'), ('ledger', 'ledger'), ('marker', 'CONSUMED.json')]:
        require(cfg[k] == str(root / name), 'OWN_PATH:' + k)
    require(not any(Path(r['path']).is_relative_to(root) for r in
        read(p['input_manifest_path'])['files'] + cfg['sealed_files']), 'INPUT_OUTPUT_DISJOINT')
    r = cfg['resources']
    require(r['runtime_caps'] is False and r['host_PSI_kill'] is False and
        r['process_start_method'] == 'fork' and r['prewarm_readonly_contexts'] and
        r['worker_input_cache'], 'RESOURCE_POLICY')


def authorize_package(package_path, approval_path):
    p = verify_package(package_path)
    a = read(approval_path)
    require(a.get('role') == 'user' and a.get('execution_authorized') is True and
        a.get('package_sha256') == sha(package_path) and a.get('SOURCE') == p['SOURCE']
        and a.get('input_manifest_sha256') == p['input_manifest_sha256'] and
        a.get('retry') is False and a.get('next_phase_authorized') is False,
        'ONE_EXPLICIT_EXACT_PACKAGE_APPROVAL_REQUIRED')
    return p


def activate(package_path, approval_path):
    """Only after the user's affirmative Phase C response; never during preparation."""
    p = authorize_package(package_path, approval_path)
    assets = Path(p['assets'])
    issue = assets / 'GRANT_ISSUED.json'
    # Atomic create-only package marker precedes any valid stage grant.
    save(issue, dict(package_sha256=sha(package_path),
        approval_sha256=sha(approval_path), SOURCE=p['SOURCE'], retry=False))
    paths = []
    for r in p['stages']:
        cfg = read(r['draft_config_path'])
        verify(cfg)
        require(not cfg['authorized'] and not cfg['grant']['valid'] and
            not Path(cfg['run_root']).exists(), 'UNUSED_DISABLED_STAGE')
        cfg.update(authorized=True, approval_receipt_path=str(Path(approval_path).resolve()),
            approval_receipt_sha256=sha(approval_path), issuance_path=str(issue))
        cfg['grant'] = dict(valid=True, grant_id=SERIES + '_' + cfg['stage'] + '_' +
            cfg['plan_sha256'], SOURCE=cfg['SOURCE'], stage=cfg['stage'],
            plan_sha256=cfg['plan_sha256'], package_sha256=sha(package_path),
            call_scope=cfg['call_scope'], context_scope=cfg['context_scope'],
            approval_sha256=sha(approval_path), retry=False, next_phase_authorized=False)
        authorize(cfg)
        path = Path(r['draft_config_path']).with_name('config_APPROVED_PRIVATE.json')
        save(path, cfg)
        paths.append(str(path))
    return paths


def authorize(cfg):
    g = cfg.get('grant', {})
    require(cfg.get('authorized') is True and g.get('valid') is True,
        'EXPLICIT_STANDARD_Y4_SCIENCE_APPROVAL_REQUIRED')
    p = authorize_package(cfg['package_path'], cfg['approval_receipt_path'])
    require(g == dict(valid=True, grant_id=SERIES + '_' + cfg['stage'] + '_' +
        cfg['plan_sha256'], SOURCE=cfg['SOURCE'], stage=cfg['stage'],
        plan_sha256=cfg['plan_sha256'], package_sha256=sha(cfg['package_path']),
        call_scope=cfg['call_scope'], context_scope=cfg['context_scope'],
        approval_sha256=sha(cfg['approval_receipt_path']), retry=False,
        next_phase_authorized=False), 'EXACT_ONE_SHOT_STAGE_GRANT')
    require(read(cfg['issuance_path']) == dict(package_sha256=sha(cfg['package_path']),
        approval_sha256=sha(cfg['approval_receipt_path']), SOURCE=p['SOURCE'], retry=False)
        and cfg['approval_receipt_sha256'] == sha(cfg['approval_receipt_path']),
        'ONE_TIME_ISSUANCE_BINDING')
    verify(cfg)


def consumed(cfg):
    require(read(cfg['marker']) == dict(state='CONSUMED_NO_RETRY_NO_REFUND',
        grant=cfg['grant']), 'OWN_CONSUMED_MARKER')


def reserve(cfg, task, kind, index=0):
    consumed(cfg)
    require(task in cfg['tasks'] and kind in task_limits(task) and type(index) is int
        and 0 <= index < task_limits(task)[kind], 'EXACT_TASK_CALL_KIND_INDEX')
    save(Path(cfg['ledger']) / (task['id'] + '__' + kind + '__' + str(index) + '.json'),
        dict(task=task['id'], kind=kind, index=index, SOURCE=cfg['SOURCE'],
        stage=cfg['stage'], state='RESERVED_NONREFUNDABLE', actual_call_completion_separate=True))


def native_contract(cfg):
    view = SimpleNamespace(**{k: globals()[k] for k in ('sha', 'read', 'save',
        'require', 'digest', 'authorize', 'verify', 'consumed', 'reserve',
        'task_limits', 'storage', 'manifest', 'COMPILE_TOLERANCE', 'OTHER_TOLERANCE')})
    view.ROOT = _assets / 'frozen'
    view.SIGNAL_RECORDS = cfg['signal_path']
    view.design = SimpleNamespace(COHORT='CONFIRMATION', event_stratum=law.event_stratum,
        COMPILER_HASH=law.COMPILER_HASH, NATIVE_POLICY=law.NATIVE_POLICY)
    return view
