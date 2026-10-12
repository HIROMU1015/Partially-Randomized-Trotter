#!/usr/bin/env python3
"""Read-only preflight, or one separately authorized N3 OCC batch. No retry."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[name] = '1'
sys.path.insert(0, str(ROOT / 'src'))
PLAN_PATH = 'docs/research/n3_occupation_transfer_v1_execution_plan_20261012.json'
RUN_PATH = 'artifacts/n3_occupation_transfer_v1/2026-10-12/run1'
INPUT_PATH = 'artifacts/n3_occupation_redesign_preparation/2026-10-12/proposed_transfer_inputs.json'
INPUT_SHA = 'c8df923eebfd999bf389ece2eb1728e7897c90e0950b51137785c708404489f4'
THREAD_KEYS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS', 'RAYON_NUM_THREADS')


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def canonical(value): return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def validate_authorization(auth, plan, plan_sha):
    fields = {'scope', 'scientific_batch_authorized', 'source_commit', 'source_fingerprint', 'plan_sha256',
              'input_sha256', 'scientific_attempts', 'scientific_retries', 'next_scientific_stage_authorized',
              'central_theme', 'execution_branch', 'output_path'}
    if set(auth) != fields or auth['scope'] != 'n3_occupation_transfer_v1_one_batch': raise ValueError('AUTHORIZATION_SCHEMA')
    if auth['scientific_batch_authorized'] is not True or auth['next_scientific_stage_authorized'] is not False or auth['central_theme'] is not None: raise ValueError('BATCH_NOT_EXPLICITLY_AUTHORIZED')
    for key in ('source_commit', 'source_fingerprint', 'input_sha256'):
        if auth[key] != plan[key]: raise ValueError('AUTHORIZATION_BINDING_' + key)
    if auth['plan_sha256'] != plan_sha or auth['scientific_attempts'] != 1 or type(auth['scientific_attempts']) is not int or auth['scientific_retries'] != 0 or type(auth['scientific_retries']) is not int: raise ValueError('AUTHORIZATION_PLAN_OR_ATTEMPT')
    if auth['execution_branch'] != 'n3-occupation-transfer-v1-execution-20261012' or auth['output_path'] != RUN_PATH: raise ValueError('AUTHORIZATION_EXECUTION_SCOPE')


def preflight(root, plan):
    from trottertracks.representation_exploration.n3_occupation_transfer_check_v1 import Checker
    import platform
    import numpy as np
    expected_resources = {'workers': 2, 'numeric_threads_per_worker': 1, 'gpu_calls': 0,
                          'worker_address_space_bytes': 4 * 1024 ** 3, 'aggregate_rss_budget_bytes': 8 * 1024 ** 3,
                          'json_shard_bytes': 32 * 1024 ** 2, 'total_output_bytes': 64 * 1024 ** 2,
                          'log_bytes': 8 * 1024 ** 2, 'rational_bit_cap': 4096,
                          'numeric_eig_calls_per_case': 850, 'producer_PSD_calls_per_case': 512,
                          'occ_spectral_pairs_per_case': 180, 'bound_spectral_pairs_per_case': 12,
                          'wall_time_cap_seconds': None, 'cpu_time_cap_seconds': None}
    if plan['resources'] != expected_resources: raise ValueError('FROZEN_RESOURCE_CONTRACT')
    if plan['input_sha256'] != INPUT_SHA or plan['input_path'] != INPUT_PATH or sha(root / INPUT_PATH) != INPUT_SHA: raise ValueError('FIXED_INPUT_IDENTITY')
    for path, expected in plan['source_sha256'].items():
        if sha(root / path) != expected: raise ValueError('FROZEN_SOURCE_IDENTITY_' + path)
    fingerprint = hashlib.sha256(canonical(plan['source_sha256'])).hexdigest()
    if fingerprint != plan['source_fingerprint']: raise ValueError('SOURCE_FINGERPRINT')
    pack = json.loads((root / INPUT_PATH).read_bytes())
    checker = Checker()
    if pack['case_count'] != 8 or len(pack['cases']) != 8 or len({c['id'] for c in pack['cases']}) != 8: raise ValueError('FIXED_INPUT_COUNT')
    for case in pack['cases']: checker.decode(case)
    if platform.python_version() != plan['environment']['python'] or np.__version__ != plan['environment']['numpy']: raise ValueError('FROZEN_ENVIRONMENT')
    if any(os.environ.get(k) != '1' for k in THREAD_KEYS): raise ValueError('THREAD_CONFIGURATION')
    if not pack['previous_12_cases_are_development'] or pack['bounds_or_acceptance_computed'] or pack['many_body_truth_computed']: raise ValueError('INPUT_PROVENANCE')
    return {'status': 'PREFLIGHT_SCHEMA_SOURCE_INPUT_PASS_NO_PREDICATES', 'source_fingerprint': fingerprint,
            'input_sha256': INPUT_SHA, 'cases': 8, 'schema_checks': checker.checks,
            'bound_evaluations': 0, 'sufficient_condition_evaluations': 0,
            'eigensolver_calls': 0, 'ground_solves': 0, 'scientific_batch_authorized': False}


class MemoryWatch:
    """Sample runner plus all descendants; not an instantaneous OS-wide RSS cap."""
    def __init__(self, limit=8 * 1024 ** 3):
        self.limit = limit; self.peak = 0; self.violation = False
        self.observed_pids = set(); self.stop_event = threading.Event(); self.thread = None
        self.samples = 0; self.last_sample = None; self.max_sample_gap = 0.

    def sample(self):
        now = time.perf_counter()
        if self.last_sample is not None: self.max_sample_gap = max(self.max_sample_gap, now - self.last_sample)
        self.last_sample = now; self.samples += 1
        pending = [os.getpid()]; seen = set(); aggregate = 0
        while pending:
            pid = pending.pop()
            if pid in seen: continue
            seen.add(pid)
            try:
                status = Path('/proc/' + str(pid) + '/status').read_text()
                rss = next((int(r.split()[1]) * 1024 for r in status.splitlines() if r.startswith('VmRSS:')), 0)
                aggregate += rss
                # Main-process and resource-tracker children; worker threads do not
                # receive independent RSS charges for the same address space.
                for children in Path('/proc/' + str(pid) + '/task').glob('*/children'):
                    try: pending.extend(map(int, children.read_text().split()))
                    except OSError: pass
            except (OSError, ValueError): pass
        self.observed_pids.update(seen); self.peak = max(self.peak, aggregate)
        if aggregate > self.limit:
            self.violation = True
            for pid in seen - {os.getpid()}:
                try: os.kill(pid, signal.SIGTERM)
                except OSError: pass
        return aggregate

    def _loop(self):
        while not self.stop_event.wait(.05): self.sample()

    def start(self):
        self.sample(); self.thread = threading.Thread(target=self._loop, daemon=True); self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None: self.thread.join()
        self.sample()

    def report(self):
        return {'aggregate_peak_sampled_rss_bytes': self.peak, 'aggregate_budget_bytes': self.limit,
                'sampling_period_seconds': .05, 'scope': 'runner parent and all live descendants including workers/resource tracker',
                'sample_count': self.samples, 'max_observed_sample_gap_seconds': self.max_sample_gap,
                'observed_pids': sorted(self.observed_pids), 'budget_violation': self.violation,
                'qualification': 'sampled watchdog; between-sample peaks are not measured'}


def worker(case, source_fingerprint, primary=None, seal=None):
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024 ** 3, 4 * 1024 ** 3))
    from trottertracks.representation_exploration.n3_occupation_transfer_v1 import generate_case, secondary_separator
    return generate_case(case, source_fingerprint) if primary is None else secondary_separator(case, primary, seal)


def primary_pool(cases, source_fingerprint, completed=None, watch=None):
    records = {}
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = {pool.submit(worker, c, source_fingerprint): c['id'] for c in cases}
        for future in as_completed(futures):
            if watch is not None and watch.violation: raise RuntimeError('TECHNICAL_STOP_AGGREGATE_MEMORY')
            r = future.result(); records[r['input_id']] = r
            if completed is not None: completed(records)
    return [records[c['id']] for c in cases]


def save(out, name, value, emergency=False):
    data = canonical(value)
    if len(data) > 32 * 1024 ** 2: raise ValueError('JSON_SHARD_CAP')
    used = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    ceiling = 64 * 1024 ** 2 if emergency else 62 * 1024 ** 2
    # Atomic temp and old target coexist and are both included in peak disk use.
    if used + len(data) > ceiling: raise ValueError('TOTAL_OUTPUT_CAP')
    temporary = out / (name + '.tmp')
    if temporary.exists(): raise ValueError('UNEXPECTED_TEMPORARY_FILE')
    temporary.write_bytes(data); os.replace(temporary, out / name)


def log_line(out, line):
    data = (line + '\n').encode()
    path = out / 'run.log'
    if (path.stat().st_size if path.exists() else 0) + len(data) > 8 * 1024 ** 2: raise ValueError('LOG_CAP')
    if sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) + len(data) > 64 * 1024 ** 2: raise ValueError('TOTAL_OUTPUT_CAP')
    with path.open('ab') as stream: stream.write(data)


def run(root, plan, auth, auth_path, out):
    from trottertracks.representation_exploration.n3_occupation_transfer_v1 import classify
    from trottertracks.representation_exploration.n3_occupation_transfer_check_v1 import check_saved
    preflight(root, plan)
    if out != root / RUN_PATH or out.exists(): raise ValueError('FIXED_NEW_RUN1_ONLY_NO_RESUME')
    remote = subprocess.check_output(['git', 'remote', 'get-url', 'origin'], cwd=root, text=True).strip()
    if remote not in ('git@github.com:HIROMU1015/Partially-Randomized-Trotter.git', 'https://github.com/HIROMU1015/Partially-Randomized-Trotter.git'): raise ValueError('REPOSITORY_SAFETY')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=root): raise ValueError('CLEAN_EXECUTION_WORKTREE_REQUIRED')
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=root, text=True).strip()
    if branch != auth['execution_branch']: raise ValueError('EXECUTION_BRANCH')
    subprocess.run(['git', 'merge-base', '--is-ancestor', plan['source_commit'], 'HEAD'], cwd=root, check=True)
    available = int(next(r.split()[1] for r in Path('/proc/meminfo').read_text().splitlines() if r.startswith('MemAvailable:'))) * 1024
    if available < 10 * 1024 ** 3: raise ValueError('MEMORY_AVAILABILITY')
    resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024 ** 2, 32 * 1024 ** 2))
    out.mkdir(parents=True); watch = MemoryWatch(); watch.start()
    start, status, code = time.perf_counter(), 'TECHNICAL_STOP', 1
    pack = json.loads((root / INPUT_PATH).read_bytes()); cases = pack['cases']
    receipt = {'source_commit': plan['source_commit'], 'execution_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
               'plan_sha256': sha(root / PLAN_PATH), 'authorization_sha256': sha(auth_path), 'input_sha256': INPUT_SHA,
               'source_fingerprint': plan['source_fingerprint'], 'source_sha256': plan['source_sha256'],
               'environment': plan['environment'], 'resources': plan['resources'], 'mem_available_before_bytes': available,
               'attempt': 1, 'retry': 0, 'ground_solves': 0, 'quantum_compiles': 0, 'gpu_calls': 0}
    phase = 'primary'
    try:
        save(out, 'launch_receipt.json', receipt)
        def completed(records):
            save(out, 'primary_checkpoint.json', records)
            log_line(out, 'primary completed count=' + str(len(records)))
            print('primary completed count=', len(records), flush=True)
        records = primary_pool(cases, plan['source_fingerprint'], completed, watch)
        save(out, 'primary_ledger.json', records)
        seal = sha(out / 'primary_ledger.json')
        save(out, 'primary_seal.json', {'sha256': seal, 'before_secondary': True, 'inputs': 8, 'raw_method_records': 96})
        # Primary file and verdicts are sealed before ANY conditional separator.
        secondary = []
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = [pool.submit(worker, case, plan['source_fingerprint'], record, seal) for case, record in zip(cases, records)]
            for future in futures:
                if sha(out / 'primary_ledger.json') != seal: raise AssertionError('SOUNDNESS_STOP_PRIMARY_CHANGED')
                secondary.append(future.result())
        save(out, 'secondary_ledger.json', secondary)
        phase = 'saved_verification'
        audit = check_saved(pack, records, secondary, plan['source_fingerprint'], seal)
        save(out, 'independent_saved_audit.json', audit)
        result = classify(records, secondary)
        if result['gain_bases'] != audit['gain_bases'] or result['separator_bases_with_norm_false'] != audit['separator_bases']: raise AssertionError('SOUNDNESS_STOP_CLASSIFICATION')
        status = result['status']; save(out, 'result.json', dict(result, source_commit=plan['source_commit'], primary_seal=seal)); code = 0
    except Exception as ex:
        status = 'SOUNDNESS_STOP' if isinstance(ex, AssertionError) or phase == 'saved_verification' else 'TECHNICAL_STOP'
        if 'checker rational bit cap' in str(ex): status = 'TECHNICAL_STOP_CHECKER_ARITHMETIC_CAP'
        if watch.violation: status = 'TECHNICAL_STOP_AGGREGATE_MEMORY'
        failure = {'status': status, 'exception': type(ex).__name__, 'message': str(ex)[:4096], 'traceback': traceback.format_exc()[-32768:], 'attempt': 1, 'retry': 0}
        save(out, 'failure.json', failure, emergency=True)
    finally:
        watch.stop()
        if watch.violation: status, code = 'TECHNICAL_STOP_AGGREGATE_MEMORY', 1
        self_usage = resource.getrusage(resource.RUSAGE_SELF); child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        save(out, 'run_audit.json', dict(receipt, status=status, wall_seconds=time.perf_counter() - start,
             parent_cpu_seconds=self_usage.ru_utime + self_usage.ru_stime, children_cpu_seconds=child_usage.ru_utime + child_usage.ru_stime,
             parent_peak_rss_bytes_linux=self_usage.ru_maxrss * 1024, largest_child_peak_rss_bytes_linux=child_usage.ru_maxrss * 1024,
             aggregate_memory=watch.report(), mandatory_stop=True, next_scientific_stage_authorized=False), emergency=True)
        save(out, 'MANDATORY_STOP.json', {'status': status, 'mandatory_stop': True, 'next_scientific_stage_authorized': False, 'central_theme': None, 'attempt': 1, 'retry': 0}, emergency=True)
        try: log_line(out, status + ' mandatory STOP')
        except ValueError: pass  # STOP evidence is already saved within its reservation.
        print(status, 'mandatory STOP', flush=True)
    return code


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True); mode.add_argument('--preflight', action='store_true'); mode.add_argument('--run', action='store_true')
    p.add_argument('--authorization', type=Path); args = p.parse_args()
    if args.run and (args.authorization is None or not args.authorization.is_file()):
        p.error('separate explicit authorization required; no scientific computation or run directory created')
    plan_file = ROOT / PLAN_PATH
    plan = json.loads(plan_file.read_bytes())
    if args.preflight:
        print(json.dumps(preflight(ROOT, plan), indent=2)); return 0
    auth = json.loads(args.authorization.read_bytes())
    validate_authorization(auth, plan, sha(plan_file))
    return run(ROOT, plan, auth, args.authorization, ROOT / RUN_PATH)


if __name__ == '__main__': raise SystemExit(main())
