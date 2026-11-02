#!/usr/bin/env python3
"""Identity-only preflight, or a separately authorized fixed saved-width evaluation."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
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
PLAN = 'docs/research/n3_saved_width_v1_execution_plan_20261012.json'
RUN = 'artifacts/n3_saved_width_v1/2026-10-12/evaluation1'
BRANCH = 'n3-saved-width-v1-evaluation-20261012'
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS', 'RAYON_NUM_THREADS')
for key in THREADS: os.environ[key] = '1'


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def authorization(auth, plan, plan_sha):
    expected = {'scope', 'evaluation_authorized', 'source_commit', 'source_fingerprint', 'plan_sha256', 'original_payload_commit',
                'ledger_sha256', 'input_sha256', 'attempts', 'retries', 'execution_branch', 'output_path', 'next_stage_authorized'}
    if set(auth) != expected: raise ValueError('AUTHORIZATION_SCHEMA')
    if auth['scope'] != 'n3_saved_width_v1_one_posthoc_evaluation' or auth['evaluation_authorized'] is not True or auth['next_stage_authorized'] is not False:
        raise ValueError('POSTHOC_NOT_AUTHORIZED')
    for key in ('source_commit', 'source_fingerprint', 'original_payload_commit', 'ledger_sha256', 'input_sha256'):
        if auth[key] != plan[key]: raise ValueError('AUTHORIZATION_BINDING_' + key)
    if auth['plan_sha256'] != plan_sha or type(auth['attempts']) is not int or auth['attempts'] != 1 or type(auth['retries']) is not int or auth['retries'] != 0:
        raise ValueError('ONE_POSTHOC_ATTEMPT_ONLY')
    if auth['execution_branch'] != BRANCH or auth['output_path'] != RUN: raise ValueError('EXECUTION_SCOPE')


def preflight(plan):
    from trottertracks.representation_exploration.n3_saved_evidence_v1 import verify
    if platform.python_version() != '3.12.3': raise ValueError('FROZEN_PYTHON_VERSION')
    for path, expected in dict(plan['source_sha256'], **plan['saved_input_sha256']).items():
        if digest(ROOT / path) != expected: raise ValueError('FIXED_BYTES_' + path)
    fingerprint = hashlib.sha256(json.dumps(plan['source_sha256'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if fingerprint != plan['source_fingerprint']: raise ValueError('SOURCE_FINGERPRINT')
    if plan['interval_bits'] != 128 or plan['workers'] != 2 or plan['width_candidates_per_case_cap'] != 512:
        raise ValueError('FIXED_COMPARISON_CONFIGURATION')
    extraction = ROOT / plan['extraction_path']
    tree = {path.relative_to(extraction).as_posix(): digest(path) for path in sorted(extraction.rglob('*')) if path.is_file()}
    tree_hash = hashlib.sha256(json.dumps(tree, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if tree_hash != plan['extraction_tree_fingerprint']: raise ValueError('FIXED_EXTRACTION_TREE')
    result = verify(ROOT, ROOT / plan['extraction_path'])
    return {'status': 'IDENTITY_LOSSLESS_PREFLIGHT_PASS_NO_WIDTHS', 'extraction_check': result,
            'new_width_evaluations': 0, 'evaluation_authorized': False, 'new_eigensolver_calls': 0}


def run(plan, auth_path):
    preflight(plan)
    out = ROOT / RUN
    if out.exists(): raise ValueError('NO_OVERWRITE_RESUME_OR_RETRY')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT): raise ValueError('CLEAN_EXECUTION_WORKTREE')
    if subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT).decode().strip() != BRANCH: raise ValueError('EXECUTION_BRANCH')
    remote = subprocess.check_output(['git', 'remote', 'get-url', 'origin'], cwd=ROOT).decode().strip()
    if remote not in ('git@github.com:HIROMU1015/Partially-Randomized-Trotter.git', 'https://github.com/HIROMU1015/Partially-Randomized-Trotter.git'):
        raise ValueError('REPOSITORY_SAFETY')
    subprocess.run(['git', 'merge-base', '--is-ancestor', plan['source_commit'], 'HEAD'], cwd=ROOT, check=True)
    available = int(next(r.split()[1] for r in Path('/proc/meminfo').read_text().splitlines() if r.startswith('MemAvailable:'))) * 1024
    if available < 10 * 1024 ** 3: raise ValueError('MEMORY_AVAILABILITY')
    # Frozen watchdog is reused as an execution utility; its scientific run() is never called.
    sys.path.insert(0, str(ROOT / 'scripts'))
    from run_n3_occupation_transfer_v1 import MemoryWatch
    from verify_n3_saved_width_v1 import verify_case
    from trottertracks.representation_exploration.n3_saved_width_analysis_v1 import evaluate_case
    out.mkdir(parents=True)
    resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024 ** 2, 32 * 1024 ** 2))
    total_bytes = 0

    def save(name, data):
        nonlocal total_bytes
        raw = json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode() + b'\n'
        if len(raw) > 32 * 1024 ** 2 or total_bytes + len(raw) > 62 * 1024 ** 2: raise ValueError('OUTPUT_CAP')
        (out / name).write_bytes(raw); total_bytes += len(raw)

    started = time.perf_counter(); watch = MemoryWatch(); watch.start(); status, code = 'TECHNICAL_STOP', 1
    phase = 'evaluation'
    try:
        save('launch_receipt.json', {'source_commit': plan['source_commit'], 'source_fingerprint': plan['source_fingerprint'],
             'execution_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
             'plan_sha256': digest(ROOT / PLAN), 'authorization_sha256': digest(auth_path),
             'saved_input_sha256': plan['saved_input_sha256'], 'attempt': 1, 'retry': 0,
             'interpretation': 'POSTHOC finite saved witnesses only; original primary result/STOP unchanged'})
        pack = json.loads((ROOT / plan['input_path']).read_bytes())
        records = json.loads((ROOT / plan['ledger_path']).read_bytes())
        secondary = json.loads((ROOT / plan['secondary_path']).read_bytes())
        saved = []
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = [pool.submit(evaluate_case, case, record, second, plan['original_source_fingerprint'], plan['ledger_sha256'])
                       for case, record, second in zip(pack['cases'], records, secondary)]
            for case, original, second, future in zip(pack['cases'], records, secondary, futures):
                r = future.result()
                if watch.violation: raise ValueError('AGGREGATE_MEMORY_CAP')
                save(case['id'] + '.json', r)
                phase = 'independent_saved_audit'
                audit = verify_case(case, original, second, r, plan['original_source_fingerprint'], plan['ledger_sha256'])
                save(case['id'] + '.saved_audit.json', audit)
                phase = 'evaluation'
                saved.append(r)
        if len(saved) != 8: raise ValueError('FIXED_EIGHT_DENOMINATOR')
        status = 'POSTHOC_SAVED_WIDTH_EVALUATION_COMPLETE'
        save('result.json', {'status': status, 'denominator_inputs': 8, 'comparison_counts':
             {label: sum(r['comparison'] == label for r in saved) for label in sorted({r['comparison'] for r in saved})},
             'original_primary_status': 'NO_ADDITIONAL_ACCEPTANCE_OBSERVED', 'original_primary_result_changed': False,
             'new_eigensolver_calls': 0, 'new_certificate_proposals': 0, 'mandatory_stop': True,
             'next_stage_authorized': False, 'central_theme': None})
        code = 0
    except Exception as ex:
        if phase == 'independent_saved_audit': status = 'SOUNDNESS_STOP'
        if 'bit cap' in str(ex).lower() or 'BIT_CAP' in str(ex): status = 'TECHNICAL_STOP_ARITHMETIC_CAP'
        failure = {'status': status, 'exception': type(ex).__name__, 'message': str(ex)[:4096],
                   'traceback': traceback.format_exc()[-32768:], 'attempt': 1, 'retry': 0}
        raw = json.dumps(failure).encode() + b'\n'
        if len(raw) > 65536: raise ValueError('FAILURE_RECORD_CAP')
        (out / 'failure.json').write_bytes(raw)
    finally:
        watch.stop()
        if watch.violation: status, code = 'TECHNICAL_STOP_AGGREGATE_MEMORY', 1
        saved_audit = {'status': status, 'wall_seconds': time.perf_counter() - started, 'aggregate_memory': watch.report(),
                       'parent_usage': {'cpu_seconds': resource.getrusage(resource.RUSAGE_SELF).ru_utime + resource.getrusage(resource.RUSAGE_SELF).ru_stime,
                                        'peak_rss_bytes_linux': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024},
                       'attempt': 1, 'retry': 0, 'mandatory_stop': True, 'next_stage_authorized': False, 'central_theme': None}
        # Emergency records are bounded independently within the reserved 2MiB.
        for name, value in (('run_audit.json', saved_audit), ('MANDATORY_STOP.json', {'status': status, 'mandatory_stop': True, 'next_stage_authorized': False, 'attempt': 1, 'retry': 0})):
            raw = json.dumps(value, sort_keys=True).encode() + b'\n'
            if len(raw) > 1024 ** 2: raise ValueError('EMERGENCY_RECORD_CAP')
            (out / name).write_bytes(raw)
    print(status, 'Mandatory STOP')
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--evaluate', action='store_true')
    parser.add_argument('--authorization', type=Path)
    args = parser.parse_args()
    # Deny before plan/pack/ledger loading and before any output directory.
    if args.evaluate and (args.authorization is None or not args.authorization.is_file()):
        parser.error('separate explicit saved-width evaluation authorization required')
    plan = json.loads((ROOT / PLAN).read_bytes())
    if args.preflight:
        print(json.dumps(preflight(plan), indent=2)); return 0
    authorization(json.loads(args.authorization.read_bytes()), plan, digest(ROOT / PLAN))
    return run(plan, args.authorization)


if __name__ == '__main__':
    raise SystemExit(main())
