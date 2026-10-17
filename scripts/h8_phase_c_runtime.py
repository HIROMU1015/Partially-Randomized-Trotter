"""Future C execution gates. Preparation never issues a valid grant."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import h8_phase_c_inputs as inputs
c = inputs.c
ROOT = Path(__file__).resolve().parents[1]
PYTHON = '/home/AbeHiromu/projects/h6-environments/20261012/native-dependency-v2/venv/bin/python'
SCOPE = 'H8_PHASE_C_SAVED_FIVE_EPSILON_RZ_P0_V1'
THREAD_KEYS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS', 'QISKIT_NUM_PROCS', 'RAYON_NUM_THREADS')
RESOURCES = dict(processes=2, max_workers=0, threads=1, physical_cores=2,
    AS_bytes=2*2**30, RSS_bytes=2**30, driver_AS_bytes=2*2**30, driver_RSS_bytes=2**30,
    observer_AS_bytes=512*2**20, observer_RSS_bytes=256*2**20,
    aggregate_RSS_bytes=1280*2**20, admission_reserve_bytes=768*2**20,
    CPU_seconds=600, observer_CPU_seconds=120, aggregate_CPU_seconds=720, wall_seconds=900,
    output_bytes=512*2**20, log_bytes=16*2**20, stage_storage_bytes=2**30,
    minimum_free_inodes=40000, output_files_max=2048)
CAPS = dict(reaccounting_pass=1, saved_row=1900, model_axis_estimate=352, comparison=20,
            new_signal=0, new_shots=0, compile=0, fit=0, sampling=0)
SOURCES = ('src/trotterlib/h8_phase_c_accounting.py', 'scripts/h8_phase_c_inputs.py',
    'scripts/h8_phase_c_runtime.py', 'scripts/run_h8_phase_c.py', 'scripts/prepare_h8_phase_c.py',
    'scripts/launch_h8_strengthened_phase_a.py', 'scripts/track_a_phase_a_parallel.py',
    'scripts/h8_strengthened_runtime.py', 'scripts/h8_strengthened_contract.py',
    'references/registration_v1.json', 'references/input_contract_v1.json')
GIT = ('git', '--no-optional-locks', '-c', 'index.threads=1', '-c', 'core.preloadIndex=false',
       '-c', 'core.fsmonitor=false', '-c', 'core.untrackedCache=false')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    path = Path(path)
    c.require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 32*2**20, 'BOUNDED_JSON')
    return loads(path.read_bytes())


def loads(data):
    c.require(len(data) <= 32*2**20, 'BOUNDED_JSON')
    def pairs(rows):
        out = {}
        for k, v in rows:
            c.require(k not in out, 'DUPLICATE_JSON_KEY'); out[k] = v
        return out
    def number(text):
        value = float(text); c.require(math.isfinite(value), 'NONFINITE_JSON'); return value
    def reject(text):
        raise ValueError('NONFINITE_JSON:' + text)
    return json.loads(data, object_pairs_hook=pairs, parse_float=number, parse_constant=reject)


def plan():
    return dict(schema='H8_PHASE_C_PLAN_V1', scope=SCOPE, B1_result=c.B1_RESULT,
        model_sha256=c.MODEL_SHA, required_B2_SOURCE=c.B2_SOURCE, compiler_hash=c.COMPILER,
        profile_sha256=c.PROFILE, deterministic_candidates=184, old_candidates=32,
        added_candidates=152, confirmation_union=6, paired_samples=32,
        epsilons=list(c.EPSILONS), allowance_factors=list(c.FACTORS), P=0,
        registry_file_sha256=sha(ROOT/'references/registration_v1.json'),
        input_contract_file_sha256=sha(ROOT/'references/input_contract_v1.json'),
        call_caps=CAPS, resources=RESOURCES, selection='EXACT_RATIONAL_COST_THEN_LEXICAL_ID',
        missing_eligible_cost='STOP', unresolved_eligibility='NO_FULL_DOMAIN_MINIMUM',
        B2_partial_or_STOP='STOP_BEFORE_ANY_MODEL_ESTIMATE_OR_REACCOUNT',
        new_shot_calculation=False, compiler_invocations=0, refit=False,
        one_shot=True, retry=False, resume=False, mandatory_stop=True, next_stage_authorized=False)


def authorize(config, grant):
    source = config.get('SOURCE')
    c.require(type(source) is str and len(source) == 40 and all(x in '0123456789abcdef' for x in source) and
        config.get('execution_sealed') is True and grant.get('schema') == 'H8_PHASE_C_NEW_GRANT_V1' and
        grant.get('approved_by_user') is True and grant.get('runtime_authorization') is True and
        grant.get('approval_text') and grant.get('scope') == SCOPE and grant.get('SOURCE') == source and
        grant.get('config_sha256') == c.digest(config) and
        grant.get('plan_sha256') == config.get('plan_sha256') == c.digest(plan()) and
        grant.get('model_sha256') == c.MODEL_SHA and grant.get('B1_result_commit') == c.B1_RESULT and
        grant.get('B2_receipt_set_sha256') == config.get('B2_receipt_set_sha256') and
        type(config.get('B2_receipt_set_sha256')) is str and len(config['B2_receipt_set_sha256']) == 64 and
        grant.get('resources') == config.get('resources') == RESOURCES and grant.get('call_caps') == CAPS and
        all(grant.get(k) is v for k, v in [('one_shot', True), ('retry', False), ('resume', False),
            ('refit', False), ('next_stage_authorized', False)]), 'NEW_SEALED_C_USER_GRANT_REQUIRED')
    cpus = grant.get('allowed_cpus', [])
    c.require(cpus == [grant.get('driver_cpu'), grant.get('observer_cpu')] and len(cpus) == 2 and
        all(type(x) is int and x >= 0 for x in cpus) and len(set(cpus)) == 2 and
        set(cpus) <= os.sched_getaffinity(0), 'TWO_EXPLICIT_C_CPUS_REQUIRED')
    return cpus


def protected(config):
    for row in config['protected_files']:
        c.require(Path(row['path']).stat().st_size == row['bytes'] and sha(row['path']) == row['sha256'], 'PROTECTED_BYTES')
    for row in config['protected_worktrees']:
        c.require(subprocess.check_output([*GIT, 'rev-parse', 'HEAD'], cwd=row['path']).decode().strip() == row['HEAD'] and
            hashlib.sha256(subprocess.check_output([*GIT, 'status', '--porcelain=v1', '-z'], cwd=row['path'])).hexdigest() ==
            row['status_sha256'], 'PROTECTED_WORKTREE')


def verify(config, *, git=True, protection=True):
    c.require(config['profile_sha256'] == c.PROFILE and config['compiler_hash'] == c.COMPILER and
              config['python'] == PYTHON and sha(config['python']) == config['python_sha256'], 'PROFILE_PYTHON_BINDING')
    if git:
        c.require(subprocess.check_output([*GIT, 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip() == config['SOURCE'],
                  'IMMUTABLE_COMMITTED_C_SOURCE_REQUIRED')
    c.require(set(config['source_hashes']) == set(SOURCES), 'EXACT_C_SOURCE_CLOSURE')
    for name, h in config['source_hashes'].items():
        rel = Path(name)
        c.require(not rel.is_absolute() and '..' not in rel.parts and sha(ROOT/rel) == h,
            'C_SOURCE_CLOSURE_CHANGED')
        if git:
            c.require(hashlib.sha256(subprocess.check_output([*GIT, 'show', config['SOURCE']+':'+name], cwd=ROOT)).hexdigest() == h,
                      'C_SOURCE_GIT_CHANGED')
    specifications = read(ROOT/'references/input_contract_v1.json')['files']
    actual = c.unique(config['inputs'], lambda row: row['role'])
    c.require(set(actual) == {x['role'] for x in specifications}, 'EXACT_C_INPUT_ROLES')
    for spec in specifications:
        row = actual[spec['role']]
        if spec['bytes'] is not None:
            c.require(row['bytes'] == spec['bytes'] and row['sha256'] == spec['sha256'], 'PINNED_C_INPUT_RECEIPT')
    for row in config['inputs']:
        path = Path(row['path'])
        c.require(path.is_absolute() and path.is_relative_to('/home/AbeHiromu') and not path.is_symlink() and
            path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'C_INPUT_BYTES')
    if protection:
        protected(config)


def load_saved(config):
    rows = c.unique(config['inputs'], lambda x: x['role'])
    def get(role):
        return read(rows[role]['path'])
    proof = {k: get('B2_' + k) for k in ('terminal', 'cleanup', 'final', 'launcher_native_exit')}
    c.require(c.digest([rows['B2_'+k] for k in ('terminal', 'cleanup', 'final', 'launcher_native_exit')]) ==
              config['B2_receipt_set_sha256'], 'SEALED_B2_RECEIPTS')
    c.b2_gate(proof)
    # Real scalar inputs are not parsed until the explicit grant and B2 gate.
    import gzip
    raw = {k: get(k) for k in ('D0', 'B0I_accounting', 'model', 'B1_terminal', 'B1_anchors')}
    path = Path(rows['Phase_A']['path'])
    with gzip.open(path, 'rb') as stream:
        data = stream.read(32*2**20+1)
    c.require(len(data) <= 32*2**20, 'GZIP_DECOMPRESSED_LIMIT')
    raw['Phase_A'] = loads(data)
    raw['B2'] = proof
    return inputs.normalize(raw, get('registry')), get('registry')


def path_set(config, *, supervisor_control_exists=False):
    paths = [Path(config[k]) for k in ('output', 'control', 'lock')]
    c.require(len(set(paths)) == 3 and all(p.is_absolute() and p.resolve() == p and
        p.is_relative_to('/home/AbeHiromu') for p in paths) and len({p.parent for p in paths}) == 1,
        'EXCLUSIVE_HOME_STAGE_PATHS')
    protected_paths = [Path(r['path']).resolve() for r in config['protected_files'] + config['protected_worktrees']]
    c.require(all(not p.is_relative_to(q) and not q.is_relative_to(p) for p in paths for q in protected_paths),
              'OUTPUT_OVERLAPS_PROTECTED_WORK')
    if supervisor_control_exists:
        c.require(paths[1].is_dir() and not paths[1].is_symlink(), 'OWN_SUPERVISOR_CONTROL_REQUIRED')
    c.require(not any(p.exists() for i, p in enumerate(paths) if not (supervisor_control_exists and i == 1)),
              'ONE_SHOT_ALREADY_SPENT')
    return paths


def fresh_binding(config, grant, fresh, now=None):
    now = time.monotonic() if now is None else now
    c.require(fresh['SOURCE'] == config['SOURCE'] and fresh['config_sha256'] == c.digest(config) and
        fresh['grant_sha256'] == c.digest(grant) and fresh['allowed_cpus'] == grant['allowed_cpus'] and
        0 <= now-fresh['monotonic_seconds'] <= 30 and all(fresh[k] is True for k in
        ('source_input_verified', 'physical_cores_verified', 'resources_idle_verified',
         'quota_verified', 'protected_jobs_unchanged', 'external_monitor_ready')), 'FRESH_C_GATE_REQUIRED')
