"""Future C execution gates. Preparation never issues a valid grant.

Kept from the Phase C snapshot: sealed config, grant/plan/B2-receipt binding, source and
input closure, protected work, exclusive paths, fresh-gate staleness and one-shot lock.
Changed for the final B2 contract and the current resource rule: no agent-imposed
CPU/RAM/worker/wall caps, no idle-core admission, PSI recorded as observation only.
"""
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
REGISTRY = 'references/h8_phase_c_registration_v1.json'
INPUT_CONTRACT = 'references/h8_phase_c_input_contract_v2.json'
# Admission facts, not limits on the job: the job is one arithmetic process.
RESOURCE_POLICY = dict(agent_imposed_CPU_RAM_worker_wall_caps=None, processes=1, child_processes=0,
    estimates=dict(peak_RSS_bytes=256*2**20, output_bytes=64*2**20, wall='seconds'),
    admission=dict(minimum_memory_available_bytes=4*2**30, minimum_free_disk_bytes=2**30, minimum_free_inodes=10000),
    CPU_selection='ANY_AVAILABLE_CPUS_EXCLUDING_CPUS_OF_RUNNING_PROTECTED_JOBS',
    PSI='OBSERVATION_ONLY', own_child_signal_or_OOM_kill='STOP', supervision='LAUNCHER_WAIT4_DIRECT_CHILD')
CAPS = dict(reaccounting_pass=1, saved_row=1900, model_axis_estimate=352, comparison=20,
            new_signal=0, new_shots=0, compile=0, fit=0, sampling=0)
SOURCES = ('src/trotterlib/h8_phase_c_accounting.py', 'scripts/h8_phase_c_inputs.py',
    'scripts/h8_phase_c_runtime.py', 'scripts/run_h8_phase_c.py', 'scripts/prepare_h8_phase_c.py',
    'tests/test_h8_phase_c.py', REGISTRY, INPUT_CONTRACT)
B2_ROLES = ('B2_binding', 'B2_execution_receipts')
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


def identity(pid):
    """(pid, start ticks, parent, uid) of a live process, or None."""
    try:
        fields = (Path('/proc')/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()
        return [pid, int(fields[19]), int(fields[1]), (Path('/proc')/str(pid)).stat().st_uid]
    except (OSError, IndexError, ValueError):
        return None


def plan():
    return dict(schema='H8_PHASE_C_PLAN_V2', scope=SCOPE, B1_result=c.B1_RESULT,
        execution_contract_revision='POST_REVIEW_SOURCE_BINDING_V3',
        source_binding='SOURCE_HEAD_EXACT_CLOSURE_GIT_BYTES_CONFIG_AND_NEW_GRANT',
        protection_policy=dict(enforced='OWN_RUNTIME_DEPENDENCIES_ONLY', unrelated_worktrees='OBSERVATION_ONLY'),
        launcher_cleanup=dict(owner='UNREAPED_DIRECT_CHILD_PID_START_PPID_UID_PIDFD',
            interrupted='TERM_THEN_KILL_AND_WAIT4', errors='STOP_WITH_ERRORS_AND_REAP_STATE',
            first_stop='PRESERVED', repeated_cleanup='NO_DOUBLE_SIGNAL_OR_WAIT'),
        model_sha256=c.MODEL_SHA, compiler_hash=c.COMPILER, profile_sha256=c.PROFILE,
        B2_result=c.B2_RESULT, B2_status=c.B2_STATUS, B2_final_SOURCE=c.B2_FINAL_SOURCE,
        B2_axes=[dict(case_id=a, axis=b, SOURCE=s, frozen_prediction=p) for a, b, s, p in c.B2_AXES],
        B2_saved_axes_accepted=3, B2_new_axes_completed=1, deterministic_candidates=184, old_candidates=32,
        added_candidates=152, confirmation_union=6, paired_samples=32, exploration_n8_pooled=False,
        epsilons=list(c.EPSILONS), allowance_factors=list(c.FACTORS), P=0,
        metric='SHOT_WEIGHTED_FULL_WRAPPER_RZ_STATE_PREPARATION_EXCLUDED', bias_kind=c.USUAL_BIAS,
        noncancellation_shots_used=False, registry_file_sha256=sha(ROOT/REGISTRY),
        input_contract_file_sha256=sha(ROOT/INPUT_CONTRACT), call_caps=CAPS, resource_policy=RESOURCE_POLICY,
        unit_cost_order=['B2_REGISTERED_S4_MEASUREMENT', 'B1_REGISTERED_ANCHOR', 'LEGACY_FULL_PF_DECLARED_LIMITS',
                         'FIXED_MODEL_ESTIMATE_S2_OR_S4'], old_B0_unresolved_costs=False,
        selection='EXACT_RATIONAL_COST_THEN_LEXICAL_ID', missing_eligible_cost='STOP',
        unresolved_eligibility='NO_FULL_DOMAIN_MINIMUM', B2_partial_or_STOP='STOP_BEFORE_ANY_MODEL_ESTIMATE_OR_REACCOUNT',
        new_shot_calculation=False, compiler_invocations=0, refit=False,
        one_shot=True, retry=False, resume=False, mandatory_stop=True, next_stage_authorized=False)


def authorize(config, grant):
    source = config.get('SOURCE')
    c.require(type(source) is str and len(source) == 40 and all(x in '0123456789abcdef' for x in source) and
        config.get('execution_sealed') is True and grant.get('schema') == 'H8_PHASE_C_NEW_GRANT_V2' and
        grant.get('approved_by_user') is True and grant.get('runtime_authorization') is True and
        grant.get('approval_text') and grant.get('scope') == SCOPE and grant.get('SOURCE') == source and
        grant.get('config_sha256') == c.digest(config) and
        grant.get('plan_sha256') == config.get('plan_sha256') == c.digest(plan()) and
        grant.get('model_sha256') == c.MODEL_SHA and grant.get('B1_result_commit') == c.B1_RESULT and
        grant.get('B2_result_commit') == c.B2_RESULT and
        grant.get('B2_receipt_set_sha256') == config.get('B2_receipt_set_sha256') and
        type(config.get('B2_receipt_set_sha256')) is str and len(config['B2_receipt_set_sha256']) == 64 and
        grant.get('resource_policy') == config.get('resource_policy') == RESOURCE_POLICY and grant.get('call_caps') == CAPS and
        all(grant.get(k) is v for k, v in [('one_shot', True), ('retry', False), ('resume', False),
            ('refit', False), ('next_stage_authorized', False), ('noncancellation_shots_allowed', False)]),
        'NEW_SEALED_C_USER_GRANT_REQUIRED')
    cpus = grant.get('allowed_cpus', [])
    c.require(type(cpus) is list and cpus and all(type(x) is int and x >= 0 for x in cpus) and
        len(set(cpus)) == len(cpus) and set(cpus) <= os.sched_getaffinity(0), 'EXPLICIT_C_CPUS_REQUIRED')
    return cpus


def protected(config):
    """Enforce this run's dependencies; other work is observed, never frozen."""
    critical = {str((ROOT/name).resolve()) for name in config.get('source_hashes', {})}
    critical.update(str(Path(row['path']).resolve()) for row in config.get('inputs', []))
    if config.get('python'):
        critical.add(str(Path(config['python']).resolve()))
    own_paths = [Path(config[k]).resolve() for k in ('output', 'control', 'lock') if config.get(k)]
    report = dict(policy='OWN_RUNTIME_DEPENDENCIES_ONLY_OTHER_WORK_OBSERVATION',
                  critical_files_checked=0, unrelated_file_receipts_observed=0, worktrees=[])
    for row in config['protected_files']:
        path = Path(row['path']).resolve()
        if str(path) in critical or any(path == p or path.is_relative_to(p) for p in own_paths):
            c.require(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'PROTECTED_BYTES')
            report['critical_files_checked'] += 1
        else:
            # Do not rehash unrelated jobs' runtime/ledger trees or reject their progress.
            report['unrelated_file_receipts_observed'] += 1
    for row in config['protected_worktrees']:
        observation = dict(path=row['path'], enforced=False)
        try:
            head = subprocess.check_output([*GIT, 'rev-parse', 'HEAD'], cwd=row['path'], stderr=subprocess.PIPE, timeout=5).decode().strip()
            status = hashlib.sha256(subprocess.check_output([*GIT, 'status', '--porcelain=v1', '-z'],
                cwd=row['path'], stderr=subprocess.PIPE, timeout=5)).hexdigest()
            observation.update(HEAD=head, status_sha256=status,
                changed=head != row['HEAD'] or status != row['status_sha256'])
        except (OSError, subprocess.SubprocessError) as error:
            observation.update(observation_error=type(error).__name__, changed=None)
        report['worktrees'].append(observation)
    return report


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
    specifications = read(ROOT/INPUT_CONTRACT)['files']
    actual = c.unique(config['inputs'], lambda row: row['role'])
    c.require(set(actual) == {x['role'] for x in specifications}, 'EXACT_C_INPUT_ROLES')
    for spec in specifications:
        row = actual[spec['role']]
        c.require(row['bytes'] == spec['bytes'] and row['sha256'] == spec['sha256'], 'PINNED_C_INPUT_RECEIPT')
    for row in config['inputs']:
        path = Path(row['path'])
        c.require(path.is_absolute() and path.is_relative_to('/home/AbeHiromu') and not path.is_symlink() and
            path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'C_INPUT_BYTES')
    if protection:
        return protected(config)


def load_saved(config):
    rows = c.unique(config['inputs'], lambda x: x['role'])
    def get(role):
        return read(rows[role]['path'])
    proof = dict(binding=get('B2_binding'), receipts=get('B2_execution_receipts'))
    c.require(c.digest([rows[k] for k in B2_ROLES]) == config['B2_receipt_set_sha256'], 'SEALED_B2_RECEIPTS')
    c.b2_gate(proof)
    # Real scalar inputs are not parsed until the explicit grant and the final B2 gate.
    import gzip
    raw = {k: get(k) for k in ('D0', 'D0_terminal', 'B0I_accounting', 'B0I_file_receipts', 'model',
                               'B1_terminal', 'B1_launcher_native_exit', 'B1_anchors')}
    path = Path(rows['Phase_A']['path'])
    with gzip.open(path, 'rb') as stream:
        data = stream.read(32*2**20+1)
    c.require(len(data) <= 32*2**20, 'GZIP_DECOMPRESSED_LIMIT')
    raw['Phase_A'] = loads(data)
    raw['B2'] = proof
    inputs.lineage(raw, rows)
    registry = get('registry')
    return inputs.normalize(raw, registry), registry


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
        ('source_input_verified', 'CPU_selection_verified', 'resources_admission_verified',
         'quota_verified', 'protected_jobs_unchanged', 'launcher_wait_ready')), 'FRESH_C_GATE_REQUIRED')
