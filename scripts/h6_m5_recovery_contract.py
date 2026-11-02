"""One fixed failed point, fresh authorization and immutable origin evidence."""
from pathlib import Path
import subprocess
import h46_m5_tol_contract as base

ROOT = base.ROOT
require, read, save, sha, core = base.require, base.read, base.save, base.sha, base.core
POLICY, COMPILER_HASH = base.POLICY, base.COMPILER_HASH
ORIGIN_SOURCE = 'c7d53f04c3be131078ee91e8bef3c78ca434ed3e'
LANE = 'H6_FULL_M5_DIAGNOSTIC_RECOVERY'
TASK = next(t for t in base.scope('H6_TOL_PILOT_DET')['tasks']
            if t['group']['method'] == 'FULL' and t['group']['formula'] == '4th(m5_best)')
SOURCE_PATHS = base.SOURCE_PATHS + (
    'scripts/h6_m5_recovery_contract.py', 'scripts/h6_m5_recovery_native.py',
    'scripts/launch_h6_m5_recovery.py', 'scripts/prepare_h6_m5_recovery.py',
)


def scope():
    return dict(lane=LANE, system='H6', tasks=[TASK], compile_axes=2,
        axis_order=['cosine', 'sine_only_after_cosine_PASS'], PR_trajectories=0,
        full_H_reference_jobs=0, numeric_tolerance=1e-9, native_policy=POLICY,
        fixed_point_signal_and_correctness_recomputed=True, signal_search=False,
        quarantine_before_numeric_gate=True, phase_alignment_acceptance=False,
        shot_accounting=False, ranking=False, refit=False, retry=False,
        phase3_authorized=False, mandatory_STOP=True)


def verify_seals(cfg):
    require(cfg['lane'] == LANE and cfg['system'] == 'H6', 'FIXED_RECOVERY_LANE')
    require(cfg['origin_SOURCE'] == ORIGIN_SOURCE and cfg['native_policy'] == POLICY,
            'ORIGIN_SCIENCE_POLICY')
    require(cfg['plan']['scope'] == scope() and cfg['plan']['SOURCE'] == cfg['SOURCE'], 'FIXED_POINT_SCOPE')
    require(read(cfg['plan_path']) == cfg['plan'] and sha(cfg['plan_path']) == cfg['plan_sha256'], 'PLAN_BYTES')
    sealed = {str(Path(x['path']).resolve()): x for x in cfg['sealed_files']}
    require(len(sealed) == len(cfg['sealed_files']), 'UNIQUE_SEAL_PATHS')
    require(core.digest(cfg['sealed_files']) == cfg['plan']['closure_digest'], 'CLOSURE_DIGEST')
    for x in sealed.values():
        require(Path(x['path']).stat().st_size == x['bytes'] and sha(x['path']) == x['sha256'], 'SEALED_BYTES:' + x['role'])
    for relative in SOURCE_PATHS:
        path = ROOT/relative
        require(str(path.resolve()) in sealed, 'USED_SOURCE_SEALED')
        require(path.read_bytes() == subprocess.check_output(['git', 'show', cfg['SOURCE']+':'+relative], cwd=ROOT), 'GIT_SOURCE_BYTES:'+relative)
    for key in ('origin_config_path', 'origin_STOP_path', 'origin_completion_path', 'origin_exit_path'):
        require(str(Path(cfg[key]).resolve()) in sealed, 'ORIGIN_EVIDENCE_SEALED')
    origin = read(cfg['origin_config_path'])
    base.verify_seals(origin)
    require(origin['SOURCE'] == ORIGIN_SOURCE and origin['lane'] == 'H6_TOL_PILOT_DET', 'ORIGIN_H6_DET')
    for key in ('binding', 'inputs', 'runtime', 'provider_root', 'reference_root'):
        require(cfg[key] == origin[key], 'ORIGIN_BINDING:'+key)
        require(core.digest(cfg[key]) == cfg['plan'][key+'_digest'], 'USED_BINDING_DIGEST:'+key)
    for x in cfg['inputs'].values():
        require(str(Path(x['path']).resolve()) in sealed and sha(x['path']) == x['sha256'], 'USED_INPUT_SEALED')
    stop = read(cfg['origin_STOP_path'])
    require(stop['task'] == TASK['id'] and stop['status'] == 'STOP'
        and stop['reason'] == 'NUMERIC_DISAGREEMENT:compiled_vector_cosine', 'PRESERVED_FAILED_TASK')
    completion = read(cfg['origin_completion_path'])
    require(completion['all_reaped'] and not completion['next_phase_started']
        and completion['lanes']['H6_TOL_PILOT_DET']['status'] == 'STOP', 'ORIGIN_STOP_AND_REAP')
    exit_receipt = read(cfg['origin_exit_path'])
    require(exit_receipt['native_exit'] == 1 and exit_receipt['reaped'], 'ORIGIN_NATIVE_EXIT')
    for name in ('reference.npz', 'receipt.json'):
        require(str((Path(cfg['reference_root'])/name).resolve()) in sealed, 'SAVED_REFERENCE_SEALED')
    ref = read(Path(cfg['reference_root'])/'receipt.json')
    require(ref['status'] == 'PASS' and ref['SOURCE'] == ORIGIN_SOURCE
        and ref['reference_sha256'] == sha(Path(cfg['reference_root'])/'reference.npz')
        and ref['snapshot'] == cfg['binding']['snapshot'], 'REUSED_REFERENCE_LINEAGE')


def authorize(cfg):
    grant = cfg.get('grant', {})
    require(cfg.get('authorized') is True and grant.get('valid') is True
        and isinstance(grant.get('approval_reference'), str) and grant['approval_reference'].strip(),
        'EXPLICIT_FIXED_POINT_RECALCULATION_APPROVAL_REQUIRED')
    expected = dict(lane=LANE, SOURCE=cfg['SOURCE'], plan_sha256=cfg['plan_sha256'],
        maximum_axes=2, maximum_trajectories=0, maximum_reference_jobs=0,
        retry=False, phase3_authorized=False)
    require(all(grant.get(k) == v for k, v in expected.items()), 'GRANT_FIXED_POINT_ONLY')
    require(isinstance(grant.get('grant_id'), str) and grant['grant_id'], 'NEW_GRANT_ID')
    root = Path(cfg['run_root']).resolve()
    require(not root.is_relative_to(ROOT) and not ROOT.is_relative_to(root), 'OWN_OUTPUT_OUTSIDE_SOURCE')
    for key in ('output', 'ledger', 'marker'):
        p = Path(cfg[key]).resolve()
        require(str(p) == cfg[key] == grant.get(key) and p.is_relative_to(root), 'NEW_GRANT_PATH:'+key)
    require(len({cfg[k] for k in ('output', 'ledger', 'marker')}) == 3, 'DISTINCT_OUTPUT_PATHS')
    require(not root.is_relative_to(Path(read(cfg['origin_config_path'])['run_root']).resolve()), 'NO_OLD_RUN_WRITES')
    verify_seals(cfg)


def consumed(cfg):
    require(read(cfg['marker']) == dict(state='CONSUMED_NO_REFUND_NO_RETRY', grant=cfg['grant']), 'CONSUMED_GRANT')


def reserve(cfg, task, axis, fingerprint):
    require(task == TASK and axis in ('cosine', 'sine'), 'REGISTERED_FIXED_POINT_AXIS')
    if axis == 'sine':
        require((Path(cfg['output'])/TASK['id']/'cost_cosine.json').is_file(), 'COSINE_PASS_BEFORE_SINE')
    path = Path(cfg['ledger'])/(task['id']+'__'+axis+'.json')
    save(path, dict(state='ENTERED_NONREFUNDABLE', task=task['id'], axis=axis,
        SOURCE=cfg['SOURCE'], plan_sha256=cfg['plan_sha256'], fingerprint=fingerprint))
    return path
