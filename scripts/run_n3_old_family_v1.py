#!/usr/bin/env python3
"""Identity preflight or separately authorized one-shot S1 OLD-family diagnostic."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks.representation_exploration.n3_old_family_contract_v1 import CAPS, canonical
from trottertracks.representation_exploration.n3_old_family_inputs_v1 import IDS, PREFIX, digest, verify_inputs

PLAN = 'docs/research/n3_old_family_v1_execution_plan_20261012.json'
RUN = PREFIX + 'diagnostic1'
BRANCH = 'n3-old-family-v1-diagnostic-20261012'
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS', 'RAYON_NUM_THREADS')
for key in THREADS: os.environ[key] = '1'
RESOURCES = {'workers': 2, 'numeric_threads_per_worker': 1, 'worker_AS_bytes': 2 * 1024 ** 3,
             'aggregate_sampled_RSS_bytes': 4 * 1024 ** 3, 'minimum_available_bytes': 6 * 1024 ** 3,
             'output_shard_bytes': 32 * 1024 ** 2, 'total_output_bytes': 64 * 1024 ** 2,
             'log_bytes': 8 * 1024 ** 2, 'failure_bytes': 65536,
             'wall_time_cap_seconds': None, 'cpu_time_cap_seconds': None, 'caps_per_case': CAPS}


def validate_authorization(auth, plan, plan_sha):
    fields = {'scope', 'diagnostic_authorized', 'source_commit', 'source_fingerprint', 'plan_sha256',
              'input_fingerprint', 'original_payload_commit', 'ledger_sha256', 'attempts', 'retries',
              'execution_branch', 'output_path', 'next_stage_authorized', 'central_theme'}
    if set(auth) != fields or auth['scope'] != 'n3_old_family_v1_two_S1_one_diagnostic':
        raise ValueError('AUTHORIZATION_SCHEMA_NEW_SCOPE_REQUIRED')
    if auth['diagnostic_authorized'] is not True or auth['next_stage_authorized'] is not False or auth['central_theme'] is not None:
        raise ValueError('DIAGNOSTIC_NOT_EXPLICITLY_AUTHORIZED')
    for key in ('source_commit', 'source_fingerprint', 'input_fingerprint', 'original_payload_commit', 'ledger_sha256'):
        if auth[key] != plan[key]: raise ValueError('AUTHORIZATION_BINDING_' + key)
    if auth['plan_sha256'] != plan_sha or type(auth['attempts']) is not int or auth['attempts'] != 1 or type(auth['retries']) is not int or auth['retries'] != 0:
        raise ValueError('ONE_NEW_DIAGNOSTIC_ATTEMPT_ONLY')
    if auth['execution_branch'] != BRANCH or auth['output_path'] != RUN: raise ValueError('EXECUTION_SCOPE')


def preflight(root, plan):
    import hashlib
    import numpy as np
    if platform.python_version() != plan['environment']['python'] or np.__version__ != plan['environment']['numpy']:
        raise ValueError('FROZEN_ENVIRONMENT')
    if plan['resources'] != RESOURCES or plan['case_ids'] != list(IDS) or plan['target_policy'] != 'saved_OCC_positive_rational_high':
        raise ValueError('FROZEN_DIAGNOSTIC_CONTRACT')
    for path, sha in dict(plan['source_sha256'], **plan['saved_input_sha256']).items():
        if digest(root / path) != sha: raise ValueError('FROZEN_BYTES_' + path)
    if hashlib.sha256(canonical(plan['source_sha256'])).hexdigest() != plan['source_fingerprint']:
        raise ValueError('SOURCE_FINGERPRINT')
    if hashlib.sha256(canonical(plan['input_sha256'])).hexdigest() != plan['input_fingerprint']:
        raise ValueError('INPUT_FINGERPRINT')
    return verify_inputs(root)


def worker(instance):
    resource.setrlimit(resource.RLIMIT_AS, (RESOURCES['worker_AS_bytes'], RESOURCES['worker_AS_bytes']))
    from trottertracks.representation_exploration.n3_old_family_v1 import diagnose
    return diagnose(instance)


def execute(plan, auth_path):
    preflight(ROOT, plan)
    out = ROOT / RUN
    if out.exists(): raise ValueError('NO_OVERWRITE_RESUME_OR_RETRY')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT): raise ValueError('CLEAN_EXECUTION_WORKTREE')
    if subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT).decode().strip() != BRANCH:
        raise ValueError('EXECUTION_BRANCH')
    if subprocess.check_output(['git', 'remote', 'get-url', 'origin'], cwd=ROOT).decode().strip() not in (
            'git@github.com:HIROMU1015/Partially-Randomized-Trotter.git', 'https://github.com/HIROMU1015/Partially-Randomized-Trotter.git'):
        raise ValueError('REPOSITORY_SAFETY')
    subprocess.run(['git', 'merge-base', '--is-ancestor', plan['source_commit'], 'HEAD'], cwd=ROOT, check=True)
    available = int(next(r.split()[1] for r in Path('/proc/meminfo').read_text().splitlines() if r.startswith('MemAvailable:'))) * 1024
    if available < RESOURCES['minimum_available_bytes']: raise ValueError('MEMORY_AVAILABILITY')
    sys.path.insert(0, str(ROOT / 'scripts'))
    from run_n3_occupation_transfer_v1 import MemoryWatch
    from trottertracks.representation_exploration.n3_old_family_check_v1 import verify
    from trottertracks.representation_exploration.n3_old_family_inputs_v1 import audit_saved_winner
    out.mkdir(parents=True); total = 0
    resource.setrlimit(resource.RLIMIT_FSIZE, (RESOURCES['output_shard_bytes'], RESOURCES['output_shard_bytes']))

    def save(name, data):
        nonlocal total
        raw = canonical(data) + b'\n'
        if len(raw) > RESOURCES['output_shard_bytes'] or total + len(raw) > 62 * 1024 ** 2:
            raise ValueError('OUTPUT_CAP')
        (out / name).write_bytes(raw); total += len(raw)

    status, code, phase = 'TECHNICAL_STOP', 1, 'saved_input_audit'
    watch = MemoryWatch(RESOURCES['aggregate_sampled_RSS_bytes']); watch.start(); started = time.perf_counter()
    try:
        save('launch_receipt.json', {'source_commit': plan['source_commit'], 'source_fingerprint': plan['source_fingerprint'],
             'execution_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
             'plan_sha256': digest(ROOT / PLAN), 'authorization_sha256': digest(auth_path),
             'input_fingerprint': plan['input_fingerprint'], 'case_ids': list(IDS),
             'attempt': 1, 'retry': 0, 'original_results_changed': False})
        rows = [json.loads((ROOT / PREFIX / 'inputs' / (name + '.json')).read_bytes()) for name in IDS]
        for row in rows: save(row['instance']['id'] + '.saved_OCC_audit.json', audit_saved_winner(ROOT, row))
        records = []
        phase = 'diagnostic'
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = [pool.submit(worker, row['instance']) for row in rows]
            for row, future in zip(rows, futures):
                record = future.result()
                if watch.violation: raise ValueError('AGGREGATE_MEMORY_CAP')
                save(record['input_id'] + '.json', record)
                phase = 'independent_certificate_audit'
                save(record['input_id'] + '.independent_audit.json', verify(row['instance'], record))
                phase = 'diagnostic'; records.append(record)
        watch.stop()
        if watch.violation: raise ValueError('AGGREGATE_MEMORY_CAP')
        status = 'TWO_S1_OLD_FAMILY_DIAGNOSTIC_COMPLETE'
        save('result.json', {'status': status, 'denominator_inputs': 2,
             'case_statuses': {r['input_id']: r['status'] for r in records},
             'original_primary_status': 'NO_ADDITIONAL_ACCEPTANCE_OBSERVED', 'original_results_changed': False,
             'finite_failure_proves_infeasible': False, 'optimal_old_width_improvement_percent_claimed': False,
             'mandatory_stop': True, 'next_stage_authorized': False, 'central_theme': None})
        code = 0
    except Exception as ex:
        status = 'SOUNDNESS_STOP' if phase in ('saved_input_audit', 'independent_certificate_audit') else 'TECHNICAL_STOP'
        if isinstance(ex, ArithmeticError): status = 'TECHNICAL_STOP_ARITHMETIC_OR_NUMERIC_LIMIT'
        raw = canonical({'status': status, 'exception': type(ex).__name__, 'message': str(ex)[:4096],
                         'traceback': traceback.format_exc()[-32768:], 'attempt': 1, 'retry': 0}) + b'\n'
        if len(raw) > RESOURCES['failure_bytes']: raise ValueError('FAILURE_RECORD_CAP')
        (out / 'failure.json').write_bytes(raw)
    finally:
        watch.stop()
        if watch.violation: status, code = 'TECHNICAL_STOP_AGGREGATE_MEMORY', 1
        audit = {'status': status, 'wall_seconds': time.perf_counter() - started, 'aggregate_memory': watch.report(),
                 'parent_CPU_seconds': resource.getrusage(resource.RUSAGE_SELF).ru_utime + resource.getrusage(resource.RUSAGE_SELF).ru_stime,
                 'parent_peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                 'attempt': 1, 'retry': 0, 'mandatory_stop': True, 'next_stage_authorized': False, 'central_theme': None}
        for name, value in [('run_audit.json', audit), ('MANDATORY_STOP.json', {k: audit[k] for k in ('status', 'attempt', 'retry', 'mandatory_stop', 'next_stage_authorized', 'central_theme')})]:
            raw = canonical(value) + b'\n'
            if len(raw) > 1024 ** 2: raise ValueError('EMERGENCY_RECORD_CAP')
            (out / name).write_bytes(raw)
    print(status, 'Mandatory STOP')
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--diagnose', action='store_true')
    parser.add_argument('--authorization', type=Path)
    args = parser.parse_args()
    # Reject before plan/pack loading or output creation.
    if args.diagnose and (args.authorization is None or not args.authorization.is_file()):
        parser.error('separate explicit OLD-family diagnostic authorization required')
    plan = json.loads((ROOT / PLAN).read_bytes())
    if args.preflight:
        print(json.dumps(preflight(ROOT, plan), indent=2)); return 0
    validate_authorization(json.loads(args.authorization.read_bytes()), plan, digest(ROOT / PLAN))
    return execute(plan, args.authorization)


if __name__ == '__main__': raise SystemExit(main())
