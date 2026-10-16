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
from fractions import Fraction
from .spec import (ART, AUTH_PREFIX, OUTPUT_PREFIX, RESOURCES, SELECTIONS,
                   SOURCE_FILES, EVIDENCE, FORECAST, template, keys, contrast_keys)


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
    mutable = {'status', 'adopted', 'coverage_model_adopted', 'science', 'source_identity', 'resources'}
    for k in set(t) - mutable:
        if not same(c[k], t[k]):
            raise ExecutionClosed('immutable contract field: ' + k)
    if require_adopted and (c['adopted'] is not True or c['coverage_model_adopted'] is not True
                            or c['status'] != 'ADOPTED_FOR_SEPARATE_EXECUTION_APPROVAL'):
        raise ExecutionClosed('contract/conditional coverage not adopted')
    s = c['science']
    if not isinstance(s, dict) or set(s) != set(t['science']):
        raise ExecutionClosed('science schema')
    if require_adopted and any(v is None for v in s.values()):
        raise ExecutionClosed('unresolved scientific selections')
    for k, choices in SELECTIONS.items():
        if s[k] is not None and s[k] not in choices:
            raise ExecutionClosed('unsupported scientific policy: ' + k)
    for k in ['alpha_total']:
        # Exact integer/rational or finite decimal strings, never eval or defaults.
        if s[k] is not None and (type(s[k]) is not str or len(s[k]) > 512
                                or re.fullmatch(r'[0-9]+(?:/[1-9][0-9]*|\.[0-9]+)?', s[k]) is None):
            raise ExecutionClosed('exact rational string required: ' + k)
        if s[k] is not None and not 0 < Fraction(s[k]) < 1:
            raise ExecutionClosed('alpha must lie strictly between zero and one')
    b = s['arithmetic_bits']
    if b is not None and (type(b) is not int or not 8 <= b <= RESOURCES['max_arithmetic_bits']):
        raise ExecutionClosed('arithmetic precision outside structural implementation range')
    r = c['resources']
    if not isinstance(r, dict) or set(r) != set(RESOURCES):
        raise ExecutionClosed('resource schema')
    for k in set(RESOURCES)-{'wall_time_seconds','cpu_time_seconds'}:
        if not same(r[k], RESOURCES[k]):
            raise ExecutionClosed('immutable structural resource cap: ' + k)
    for k in ['wall_time_seconds','cpu_time_seconds']:
        if r[k] is not None and (type(r[k]) is not int or not 1 <= r[k] <= 86400):
            raise ExecutionClosed('finite positive integer time cap required: ' + k)
        if require_adopted and r[k] is None:
            raise ExecutionClosed('time caps not adopted')
    source = c['source_identity']
    if source is None:
        if require_adopted:
            raise ExecutionClosed('source not frozen')
        return
    if set(source) != {'commit', 'files'} or set(source['files']) != set(SOURCE_FILES):
        raise ExecutionClosed('source inventory')
    sha(source['commit'], 40)
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
        'schema': 'm1s_saved_signal_gamma_v1_execution_authorization',
        'scientific_execution_authorized': True, 'scope': 'V1_GAMMA_CONDITIONS_SAVED_IDEAL_SIGNAL_ONCE',
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
    c = permit.config
    return {'schema': 'm1s_saved_signal_gamma_v1_result', 'status': 'TECHNICAL_STOP',
              'evidence': EVIDENCE, 'forecast_kind': FORECAST,
              'provenance': permit.provenance, 'science': c['science'],
              'resources': c['resources'], 'mandatory_STOP': True, 'retry': False,
              'physical_shot_guarantee': False, 'scientific_GO': None,
              'native_costs': None, 'total_resources': None, 'absolute_shot_counts': None, 'gamma_grid': None,
              'rows': [dict(zip(['split', 'R', 'delta', 'T', 'method'], k),
                            status='NOT_EVALUATED') for k in keys()],
              'comparisons': [dict(zip(['split', 'R', 'delta', 'T', 'comparator'], k),
                                   status='NOT_EVALUATED',
                                   scientific_GO=None, native_costs=None,
                                   scientific_gain_claim='NOT_ASSESSED') for k in contrast_keys()],
              'failure': None}


def run(root, permit):
    require_permit(permit)
    root = Path(root).resolve()
    c = permit.config
    output = inside(root, permit.provenance['output_path'], OUTPUT_PREFIX)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()  # exclusive consumption, BEFORE correctness/analysis
    marker = {'schema': 'm1s_saved_signal_gamma_v1_consumed', 'consumed': True,
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
    return 'GAMMA_CONDITIONS_COMPLETE_AWAITING_SCIENTIFIC_JUDGMENT'


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
        resource.setrlimit(resource.RLIMIT_CPU, (c['resources']['cpu_time_seconds'], c['resources']['cpu_time_seconds']+1))
        signal.setitimer(signal.ITIMER_REAL, c['resources']['wall_time_seconds'])
        resource.setrlimit(resource.RLIMIT_AS, (child_limit, child_limit))
        resource.setrlimit(resource.RLIMIT_FSIZE, (c['resources']['result_bytes'], c['resources']['result_bytes']))
        for k, v in c['resources']['thread_environment'].items():
            os.environ[k] = v
        sys.set_int_max_str_digits(25000)  # bounded rational serialization, never a science tolerance
        from .analysis import evaluate
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
                          'science_selections': c['science'],
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
