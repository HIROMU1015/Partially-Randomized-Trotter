"""Stdlib-only authorization for four saved-circuit diagnostic replays.

No original grant is reused. Draft rejection precedes NumPy/Qiskit or array I/O.
"""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'artifacts/resource_applicability/track_a_h6_m5_saved_phase_plan/2026-10-19'
ORIGIN = 'f743a88225f7cb2c8d2c91d7e19813a5780f803b'
TASK = 'H6_TOL_FULL_0e289606747ea369__0'
SCOPE = 'H6_M5_SAVED_PHASE_DIAGNOSTIC_V1'
JOBS = ('WRAPPER_ORDINARY', 'COMPILED_ORDINARY', 'WRAPPER_EXTENDED', 'COMPILED_EXTENDED')
SOURCE_PATHS = (
    'scripts/h6_m5_phase_contract.py', 'scripts/h6_m5_phase_engine.py',
    'scripts/h6_m5_phase_adapter.py', 'scripts/run_h6_m5_saved_phase.py',
    'scripts/prepare_h6_m5_saved_phase.py',
    str((BASE/'plan_DRAFT.json').relative_to(ROOT)),
    str((BASE/'input_manifest_metadata.json').relative_to(ROOT)),
)


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    """Durable exclusive publication; never replace an existing receipt."""
    path = Path(path)
    tmp = path.with_name(path.name+'.partial')
    with tmp.open('x') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.link(tmp, path)
    tmp.unlink()


def scope():
    old = read(BASE/'plan_DRAFT.json')
    return dict(scope_id=SCOPE, target=old['target'], maximum_work=old['maximum_work'],
        jobs=list(JOBS), nqubits=13, ancilla=12, initial_state='SAVED_LOW_ONLY',
        saved_metric_and_ordinary_replay_atol=1e-12,
        technical_tolerance_status='DRAFT_FIXED_FOR_APPROVAL_NOT_YET_ADOPTED',
        raw_original_strictly_below=1e-9, acceptance_override=False,
        phase_alignment='DIAGNOSTIC_ONLY', mandatory_STOP=True, retry=False,
        next_phase_authorized=False, main_comparison_AD03='DEFERRED')


def seal(path, role):
    path = Path(path).resolve()
    return dict(path=str(path), role=role, bytes=path.stat().st_size, sha256=sha(path))


def verify_lineage(cfg):
    roles = cfg['inputs']
    expected = {r['role']: r for r in read(BASE/'input_manifest_metadata.json')['roles']}
    require(set(roles) == set(expected), 'EXACT_TWELVE_INPUT_ROLES')
    for role, row in roles.items():
        require(row['sha256'] == expected[role]['sha256'] and row['bytes'] == expected[role]['bytes'], 'FIXED_INPUT:'+role)
        require(Path(row['path']).resolve() == Path(row['path']), 'CANONICAL_INPUT_PATH')
    origin = read(roles['recovery_config']['path'])
    require(origin['SOURCE'] == ORIGIN and origin['system'] == 'H6', 'ORIGIN_SOURCE')
    require(cfg['runtime'] == origin['runtime'], 'SAME_RECOVERY_RUNTIME')
    stop = read(roles['saved_STOP']['path'])
    require(stop['task'] == TASK and stop['status'] == 'STOP' and
        stop['reason'] == 'NUMERIC_DISAGREEMENT:compiled_vector_cosine', 'OLD_STOP_UNCHANGED')
    completion = read(roles['saved_completion']['path'])
    require(completion['SOURCE'] == ORIGIN and completion['status'] == 'STOP' and
        completion['worker_exit'] == 0 and completion['worker_reaped'] and
        not completion['next_phase_started'] and completion['accepted_cost_axes'] == 0, 'ORIGIN_END_AND_REAP')
    witness = read(roles['saved_exit']['path'])
    require(witness['native_exit'] == 1 and witness['reaped'] and not witness['next_phase_started'], 'ORIGIN_NATIVE_EXIT')
    for receipt_role, field, data_role in (
        ('wrapper_receipt','circuit','wrapper_qpy'), ('wrapper_receipt','state','initial_npz'),
        ('compiled_receipt','circuit','compiled_qpy'), ('vectors_receipt','vectors','vectors_npz')):
        receipt = read(roles[receipt_role]['path'])
        require(receipt['status'] == 'UNVALIDATED_NOT_FOR_ACCOUNTING', 'UNVALIDATED_LABEL')
        require(all(receipt[field][k] == roles[data_role][k] for k in ('bytes','sha256')), 'RECEIPT_JOIN:'+data_role)
    wr = read(roles['wrapper_receipt']['path']); cr = read(roles['compiled_receipt']['path'])
    require(wr['SOURCE'] == ORIGIN and wr['axis'] == 'cosine' and
        wr['wrapper_fingerprint'] == cr['pretranspile_fingerprint'], 'WRAPPER_LINEAGE')
    require(cr['layout'] == 'None' and cr['compiler_settings_hash'] ==
        '1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b', 'FIXED_COMPILER')
    vr = read(roles['vectors_receipt']['path'])
    require(vr['dtype'] == 'complex128' and vr['shape'] == [8192], 'SAVED_VECTOR_SCHEMA')


def verify_seals(cfg):
    require(re.fullmatch('[0-9a-f]{40}', cfg['SOURCE']) is not None, 'COMMITTED_SOURCE_SHA')
    require(cfg['plan']['scope'] == scope() and cfg['plan']['SOURCE'] == cfg['SOURCE'], 'EXACT_DIAGNOSTIC_SCOPE')
    require(read(cfg['plan_path']) == cfg['plan'] and sha(cfg['plan_path']) == cfg['plan_sha256'], 'PLAN_BYTES')
    rows = cfg['sealed_files']; sealed = {r['path']: r for r in rows}
    require(len(rows) == len(sealed) and digest(rows) == cfg['plan']['closure_digest'], 'UNIQUE_CLOSURE')
    for row in rows:
        p = Path(row['path'])
        require(str(p.resolve()) == row['path'] and p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], 'SEALED_BYTES:'+row['role'])
    for rel in SOURCE_PATHS:
        p = ROOT/rel
        require(str(p.resolve()) in sealed, 'USED_SOURCE_SEALED')
        require(p.read_bytes() == subprocess.check_output(['git','show',cfg['SOURCE']+':'+rel],cwd=ROOT), 'GIT_SOURCE_BYTES:'+rel)
    for role, row in cfg['inputs'].items():
        require(row == sealed.get(row['path']), 'USED_INPUT_SEALED:'+role)
    require(digest(cfg['inputs']) == cfg['plan']['inputs_digest'] and
        digest(cfg['runtime']) == cfg['plan']['runtime_digest'], 'USED_BINDING_DIGEST')
    # Bind the completed recovery's existing closure without importing its science.
    origin = read(cfg['inputs']['recovery_config']['path'])
    for row in origin['sealed_files']:
        require(row['path'] in sealed and all(sealed[row['path']][k] == row[k] for k in ('bytes','sha256')), 'ORIGIN_CLOSURE_REUSED')
    verify_lineage(cfg)


def authorize(cfg):
    grant = cfg.get('grant', {})
    require(cfg.get('authorized') is True and grant.get('valid') is True and
        isinstance(grant.get('approval_reference'), str) and grant['approval_reference'].strip(),
        'EXPLICIT_SAVED_NUMERICAL_ANALYSIS_APPROVAL_REQUIRED')
    expected = dict(scope_id=SCOPE, SOURCE=cfg['SOURCE'], plan_sha256=cfg['plan_sha256'],
        maximum_replays=4, maximum_saved_initial_states=1, new_compile=0,
        new_sampling=0, new_state_generation=0, accounting=False, retry=False,
        acceptance_override=False, next_phase_authorized=False)
    require(all(grant.get(k) == v for k,v in expected.items()), 'GRANT_EXACT_SCOPE')
    require(isinstance(grant.get('grant_id'), str) and grant['grant_id'].strip(), 'FRESH_GRANT_ID')
    root = Path(cfg['run_root']).resolve()
    require(str(root) == cfg['run_root'] and not root.is_relative_to(ROOT) and not ROOT.is_relative_to(root), 'OWN_NEW_OUTPUT_OUTSIDE_SOURCE')
    require(cfg['plan']['run_root'] == str(root), 'PLAN_OUTPUT_BINDING')
    expected_supervisor={k:str(root.parent/name) for k,name in
        [('entered','SUPERVISOR_ENTERED.json'),('log','native_PRIVATE.log'),('exit','external_exit_witness_PRIVATE.json')]}
    require(cfg['supervisor']==expected_supervisor and cfg['plan']['supervisor']==expected_supervisor and
        grant.get('supervisor')==expected_supervisor,'NEW_SUPERVISOR_PATH_BINDING')
    for key in ('output','ledger','marker'):
        p = Path(cfg[key]).resolve()
        require(str(p) == cfg[key] == grant.get(key) and p.is_relative_to(root), 'NEW_GRANT_PATH:'+key)
    require(len({cfg[k] for k in ('output','ledger','marker')}) == 3, 'DISTINCT_OUTPUTS')
    for row in cfg['inputs'].values():
        p = Path(row['path']).resolve()
        require(not p.is_relative_to(root) and not root.is_relative_to(p.parent), 'NO_OLD_INPUT_AREA_WRITES')
    require(cfg['resources'] == cfg['plan']['resources'] and all(type(cfg['resources'][k]) is int and cfg['resources'][k] > 0
        for k in ('per_worker_RSS_proxy_bytes','disk_proxy_bytes')), 'RESOURCE_PLANNING_BINDING')
    verify_seals(cfg)


def consumed(cfg):
    require(read(cfg['marker']) == dict(state='CONSUMED_NO_RETRY_NO_REFUND', grant=cfg['grant']), 'CONSUMED_NEW_GRANT')


def reserve(cfg, job):
    require(job in JOBS, 'EXACT_REGISTERED_REPLAY')
    consumed(cfg)
    save(Path(cfg['ledger'])/(job+'.json'), dict(job=job, SOURCE=cfg['SOURCE'],
        plan_sha256=cfg['plan_sha256'], state='ENTERED_NONREFUNDABLE', new_compile=0))
