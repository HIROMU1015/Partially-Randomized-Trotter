"""Metadata validation and separate one-shot authority. Candidate always closes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import sys
import traceback
from .spec import (ART, AUTH_PREFIX, OUTPUT_PREFIX, RESOURCES,
                   SOURCE_FILES, EVIDENCE, FORECAST, template, contrast_keys)


class ExecutionClosed(PermissionError):
    pass


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(raw):
    def pairs(values):
        out = {}
        for k, v in values:
            if k in out:
                raise ValueError('duplicate JSON key')
            out[k] = v
        return out
    def invalid(value):
        raise ValueError('nonfinite JSON token: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def same(a, b):
    # JSON canonical equality keeps bool distinct from 0/1.
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)


def relative(value, prefix=None):
    if (type(value) is not str or '\\' in value or Path(value).is_absolute()
            or any(x in ['', '.', '..'] for x in value.split('/'))
            or (prefix is not None and not value.startswith(prefix))):
        raise ExecutionClosed('invalid relative path')
    return value


def sha(value, length=64):
    if type(value) is not str or re.fullmatch('[0-9a-f]{' + str(length) + '}', value) is None:
        raise ExecutionClosed('invalid hash')
    return value


def inside(root, path, prefix=None):
    p = root / relative(path, prefix)
    if p.resolve() != p.absolute():
        raise ExecutionClosed('symlink or noncanonical path')
    return p


def validate_contract(c, require_adopted=False):
    t = template()
    if not isinstance(c, dict) or set(c) != set(t):
        raise ExecutionClosed('contract schema fields')
    mutable = {'status','decision_gate_adopted','adopted','coverage_model_adopted','source_identity'}
    for k in set(t)-mutable:
        if not same(c[k],t[k]):
            raise ExecutionClosed('immutable proposed contract field: '+k)
    for k in ['decision_gate_adopted','adopted','coverage_model_adopted']:
        if type(c[k]) is not bool:
            raise ExecutionClosed('adoption flags must be boolean')
    if require_adopted and (not all(c[k] is True for k in ['decision_gate_adopted','adopted','coverage_model_adopted'])
                            or c['status'] != 'ADOPTED_FOR_SEPARATE_EXECUTION_APPROVAL'):
        raise ExecutionClosed('separate decision gate/contract/conditional coverage adoption required')
    if c['input']['bytes'] > c['resources']['input_bytes']:
        raise ExecutionClosed('saved input exceeds declared byte cap')
    source=c['source_identity']
    if source is None:
        if require_adopted:
            raise ExecutionClosed('source not frozen')
        return
    if not isinstance(source,dict) or set(source)!={'commit','files'} or set(source['files'])!=set(SOURCE_FILES):
        raise ExecutionClosed('source inventory')
    sha(source['commit'],40)
    for value in source['files'].values():
        sha(value)


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args],
                                   stderr=subprocess.PIPE,
                                   env=dict(os.environ, GIT_OPTIONAL_LOCKS='0'))


_TOKEN = object()


class _Permit:
    def __init__(self, token, config, provenance):
        if token is not _TOKEN:
            raise ExecutionClosed('private permit')
        self.token, self.config, self.provenance = token, config, provenance


def require_permit(permit):
    if not isinstance(permit, _Permit) or permit.token is not _TOKEN:
        raise ExecutionClosed('separate source-bound execution permit required')


def authorize(root, contract_path, authority_commit, authority_path):
    root = Path(root).resolve()
    cpath = inside(root, contract_path)
    contract_bytes = cpath.read_bytes()
    c = read_json(contract_bytes)
    # This must precede authority, saved scientific input reads, and engine imports.
    validate_contract(c, require_adopted=True)
    sha(authority_commit, 40)
    apath = inside(root, authority_path, AUTH_PREFIX)
    if git(root, 'rev-parse', 'HEAD').decode().strip() != authority_commit:
        raise ExecutionClosed('HEAD is not fixed execution authority commit')
    if git(root, 'status', '--porcelain', '--untracked-files=all').strip():
        raise ExecutionClosed('execution worktree must be clean')
    C = git(root, 'rev-parse', authority_commit + '^').decode().strip()
    if git(root, 'diff-tree', '--no-commit-id', '--name-only', '-r', authority_commit).decode().splitlines() != [authority_path]:
        raise ExecutionClosed('authority-only child required')
    if git(root, 'show', C + ':' + contract_path) != contract_bytes:
        raise ExecutionClosed('contract not frozen in authority parent')
    Araw = apath.read_bytes()
    if git(root, 'show', authority_commit + ':' + authority_path) != Araw:
        raise ExecutionClosed('authority bytes differ')
    A = read_json(Araw)
    expected = {'schema', 'scientific_execution_authorized', 'scope', 'contract_commit',
                'contract_path', 'contract_sha256', 'source_commit', 'one_shot', 'retry',
                'user_instruction', 'output_path', 'python_executable_sha256'}
    if set(A) != expected:
        raise ExecutionClosed('authority schema')
    required = {
        'schema': 'm1s_gamma_pi3_v1_execution_authorization',
        'scientific_execution_authorized': True, 'scope': 'SAVED_G3_G6_PI_OVER_THREE_INTERVAL_COMPARISON_ONCE',
        'contract_commit': C, 'contract_path': contract_path,
        'contract_sha256': digest(contract_bytes), 'source_commit': c['source_identity']['commit'],
        'one_shot': True, 'retry': False,
    }
    if any(not same(A[k], v) for k, v in required.items()) or not isinstance(A['user_instruction'], str) or not A['user_instruction'].strip():
        raise ExecutionClosed('explicit authority/provenance missing')
    output = inside(root, A['output_path'], OUTPUT_PREFIX)
    if output.exists():
        raise ExecutionClosed('one-shot output already exists; no retry')
    sha(A['python_executable_sha256'])
    if digest(Path(sys.executable).read_bytes()) != A['python_executable_sha256']:
        raise ExecutionClosed('future interpreter identity')
    S = c['source_identity']['commit']
    if S == C:
        raise ExecutionClosed('source must precede contract')
    git(root, 'merge-base', '--is-ancestor', S, C)
    git(root, 'merge-base', '--is-ancestor', c['base_commit'], S)
    for path, expected_sha in c['source_identity']['files'].items():
        raw = inside(root, path).read_bytes()
        if digest(raw) != expected_sha or git(root, 'show', S + ':' + path) != raw:
            raise ExecutionClosed('source identity: ' + path)
    i = c['input']
    raw = inside(root, i['path']).read_bytes()  # hash only, no numerical decoding
    if len(raw) != i['bytes'] or digest(raw) != i['sha256'] or git(root, 'show', i['commit'] + ':' + i['path']) != raw:
        raise ExecutionClosed('saved input identity')
    provenance = {'S': S, 'C': C, 'A': authority_commit,
                  'contract_sha256': digest(contract_bytes), 'authority_sha256': digest(Araw),
                  'input_sha256': digest(raw), 'output_path': A['output_path'],
                  'user_instruction': A['user_instruction'],
                  'python_executable_sha256': A['python_executable_sha256']}
    return _Permit(_TOKEN, c, provenance)


def seed_result(permit):
    c=permit.config
    return {'schema':'m1s_gamma_pi3_v1_result','status':'TECHNICAL_STOP',
            'evidence':EVIDENCE,'forecast_kind':FORECAST,'provenance':permit.provenance,
            'science':c['science'],'resources':c['resources'],
            'mandatory_STOP':True,'retry':False,'scientific_GO':None,
            'rpe_success_guaranteed':False,'resource_improvement_guaranteed':False,
            'absolute_shot_counts':None,'native_costs':None,'total_resources':None,
            'threshold':{'symbolic':'pi/3','unit':'rad','saved_pi_interval_rad':None,
                         'threshold_interval_rad':None},
            'comparisons':[dict(zip(['split','R','delta','T','comparator'],k),
                                status='NOT_EVALUATED',boundary_interval_rad=None,
                                scientific_GO=None) for k in contrast_keys()],
            'failure':None}


def run(root, permit):
    require_permit(permit)
    root = Path(root).resolve()
    c = permit.config
    output = inside(root, permit.provenance['output_path'], OUTPUT_PREFIX)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()  # exclusive consumption, BEFORE correctness/analysis
    marker = {'schema': 'm1s_gamma_pi3_v1_consumed', 'consumed': True,
              'provenance': permit.provenance, 'retry': False}
    (output / 'exclusive_launch_marker.json').write_text(json.dumps(marker, indent=2)+'\n')
    seed = (json.dumps(seed_result(permit), indent=2)+'\n').encode()
    if len(seed) > c['resources']['result_bytes']:
        raise RuntimeError('initial result bytes cap')
    (output / 'result.json').write_bytes(seed)
    from .supervisor import supervise
    parent_vm = int(Path('/proc/self/statm').read_text().split()[0])*os.sysconf('SC_PAGE_SIZE')
    child_limit = c['resources']['memory_bytes'] - parent_vm
    if child_limit <= parent_vm:
        summary = {'status': 'TECHNICAL_STOP', 'reason': 'aggregate virtual-memory reserve',
                   'mandatory_STOP': True, 'retry': False}
    else:
        summary = supervise(lambda: worker_run(root, permit, child_limit), c['resources']['wall_time_seconds'])
    summary.update(parent_virtual_bytes=parent_vm, child_address_space_cap=child_limit,
                   aggregate_memory_cap=c['resources']['memory_bytes'], provenance=permit.provenance)
    (output / 'supervisor_result.json').write_text(json.dumps(summary, indent=2)+'\n')
    if summary['status'] != 'COMPLETE_WORKER_EXIT':
        return 'TECHNICAL_STOP'
    return 'PI3_COMPARISON_COMPLETE_AWAITING_SCIENTIFIC_JUDGMENT'


def worker_run(root, permit, child_limit):
    require_permit(permit)
    root = Path(root).resolve()
    c = permit.config
    output = inside(root, permit.provenance['output_path'], OUTPUT_PREFIX)
    result = seed_result(permit)

    def checkpoint(value):
        raw = (json.dumps(value, indent=2, allow_nan=False)+'\n').encode()
        if len(raw) > c['resources']['result_bytes']:
            raise RuntimeError('result bytes cap')
        tmp = output / 'result.partial.tmp'
        with tmp.open('wb') as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, output / 'result.json')

    checkpoint(result)
    def timed_stop(signum, frame):
        raise RuntimeError('adopted CPU or wall-time cap reached')
    old_alarm = signal.signal(signal.SIGALRM, timed_stop)
    old_cpu = signal.signal(signal.SIGXCPU, timed_stop)
    try:
        # New preparation follows current no-CPU/no-wall-cap default.
        # Explicit interruption remains handled by supervisor and checkpoints.
        resource.setrlimit(resource.RLIMIT_AS, (child_limit, child_limit))
        resource.setrlimit(resource.RLIMIT_FSIZE, (c['resources']['result_bytes'], c['resources']['result_bytes']))
        for k, v in c['resources']['thread_environment'].items():
            os.environ[k] = v
        from .runtime_compatibility import configure_integer_serialization
        configure_integer_serialization()
        from .compare import evaluate
        raw_bytes = inside(root, c['input']['path']).read_bytes()
        if len(raw_bytes) != c['input']['bytes'] or digest(raw_bytes) != c['input']['sha256']:
            raise RuntimeError('input identity changed after exclusive consumption')
        raw = read_json(raw_bytes)
        evaluate(raw, c, permit, result, checkpoint)
    except BaseException as exc:
        result['status'] = 'TECHNICAL_STOP'
        result['failure'] = {'type': type(exc).__name__, 'message': str(exc)[:4096],
                             'traceback': traceback.format_exc()[-16384:]}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_alarm)
        signal.signal(signal.SIGXCPU, old_cpu)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        result['usage'] = {'maxrss_KiB_linux': usage.ru_maxrss,
                           'cpu_user_seconds': usage.ru_utime, 'cpu_system_seconds': usage.ru_stime}
        try:
            checkpoint(result)
        except BaseException as save_exc:
            # Retain the last successful atomic checkpoint even if final output
            # exceeds its cap. A bounded sidecar describes the storage STOP.
            failure = {'status': 'TECHNICAL_STOP', 'mandatory_STOP': True,
                       'retry': False, 'failure': result['failure'],
                       'checkpoint_failure': str(save_exc)[:4096]}
            (output / 'failure.json').write_text(json.dumps(failure, indent=2)+'\n')
            result['status'] = 'TECHNICAL_STOP'
    return result['status']


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['status', 'run'])
    p.add_argument('--contract', default=ART + '/execution_contract_candidate.json')
    p.add_argument('--authority-commit')
    p.add_argument('--authority-path')
    args = p.parse_args(argv)
    root = Path(__file__).resolve().parents[4]
    if args.mode == 'status':
        c = read_json(inside(root, args.contract).read_bytes())
        validate_contract(c)
        print(json.dumps({'status': c['status'], 'adopted': c['adopted'],
                          'decision_gate_adopted': c['decision_gate_adopted'], 'proposed_rules': c['science'],
                          'execution_permitted': False, 'mandatory_STOP': True}, indent=2))
        return 0
    try:
        permit = authorize(root, args.contract, args.authority_commit, args.authority_path)
    except (ExecutionClosed, ValueError, OSError, subprocess.CalledProcessError) as exc:
        print('EXECUTION_CLOSED: ' + str(exc), file=sys.stderr)
        return 2
    status = run(root, permit)
    print(status)
    return 2 if status == 'TECHNICAL_STOP' else 0
