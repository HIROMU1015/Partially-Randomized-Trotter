"""Bounded G6 formal technical checks, not a science/performance runner."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import resource
import signal
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'artifacts/track_b_g6_return_generator_audit/2026-10-10'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    data = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + '\n').encode()
    if len(data) > 2 ** 20:
        raise RuntimeError('per-file output cap')
    with (OUT / name).open('xb') as stream:
        stream.write(data)


def protected_check(scope):
    ledger = json.loads((OUT / 'prior_protected_hashes.json').read_text())
    failures = []
    for name, record in ledger.items():
        if name.lower().endswith('.npz'):
            raise RuntimeError('NPZ path forbidden before filesystem access')
        data = (ROOT / name).read_bytes()
        if name in scope['append_only_paths']:
            data = data[:record['bytes']]
        if hashlib.sha256(data).hexdigest() != record['sha256']:
            failures.append(name)
    return {'paths_checked': len(ledger), 'violations': failures,
            'old_artifacts_source_authorization_markers_STOP_unchanged': not failures}


def main():
    scope = json.loads((OUT / 'scope_v1.json').read_text())
    for name, digest in scope['technical_source_sha256'].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError('technical source hash mismatch: ' + name)
    if sha(Path(sys.executable).resolve()) != scope['runtime']['sha256']:
        raise RuntimeError('runtime identity mismatch')
    before = protected_check(scope)
    if before['violations']:
        raise RuntimeError('protected provenance mismatch')
    # New technical marker only. An existing marker stops this bundle at launch.
    save('technical_audit_started.json', {'kind': 'G6_OFF_DOMAIN_FORMAL_TECHNICAL_BUNDLE',
         'base_commit': scope['base_commit'], 'scope_sha256': sha(OUT / 'scope_v1.json'),
         'science_runs': 0, 'old_one_shot_retries': 0})
    resource.setrlimit(resource.RLIMIT_AS, (512 * 2 ** 20, 512 * 2 ** 20))
    resource.setrlimit(resource.RLIMIT_CPU, (60, 61))
    def timeout(signum, frame):
        raise TimeoutError('G6 technical wall cap')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(90)
    wall, cpu = time.monotonic(), time.process_time()
    log = io.StringIO()
    counts, tests_run, status, reason = {}, 0, 'G6_TECHNICAL_INCONCLUSIVE', None
    try:
        test_path = ROOT / 'tests/tracks/algorithm_codesign/test_g6_return_aggregation.py'
        spec = importlib.util.spec_from_file_location('g6_tests', test_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        outcome = unittest.TextTestRunner(stream=log, verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromModule(module))
        counts = dict(module.COUNTS)
        tests_run = outcome.testsRun
        status = 'G6_FORMAL_CHECKS_PASS' if outcome.wasSuccessful() else 'G6_FORMAL_CHECKS_FAIL'
    except Exception as error:
        reason = type(error).__name__ + ': ' + str(error)
    finally:
        signal.alarm(0)
    after = protected_check(scope)
    if after['violations']:
        status, reason = 'G6_TECHNICAL_INCONCLUSIVE', 'protected hashes changed'
    if time.monotonic() - wall > 90 or time.process_time() - cpu > 60:
        status, reason = 'G6_TECHNICAL_INCONCLUSIVE', 'resource cap exceeded'
    save('focused_tests.json', {'status': status, 'tests': tests_run, 'counts': counts,
                              'log': log.getvalue(), 'technical_reason': reason})
    save('post_execution_provenance.json', after)
    save('result_v1.json', {
        'kind': 'G6_EXPLORATORY_FORMAL_AUDIT_NOT_PERFORMANCE', 'status': status,
        'base_commit': scope['base_commit'], 'tests': tests_run, 'counts': counts,
        'technical_reason': reason, 'wall_s': time.monotonic() - wall,
        'CPU_s': time.process_time() - cpu,
        'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'technical_bundle_invocations': 1, 'science_runs': 0, 'old_one_shot_retries': 0,
        'LP_synthesis_Hamiltonian_matrix_circuit_trajectory_DF_molecule_NPZ_GPU_quantum_calls': 0,
        'prototype_random_draws': 0, 'formal_oracle_enumeration_in_tests_only': True,
        'native_implementation_validated': False, 'novelty_established': False,
        'performance_advantage_established': False, 'mandatory_STOP': True,
        'next_science_authorized': False, 'old_protected_paths_checked': after['paths_checked'],
        'scope_sha256': sha(OUT / 'scope_v1.json'),
        'technical_marker_sha256': sha(OUT / 'technical_audit_started.json')})
    save('STOP.json', {'mandatory_STOP': True, 'reason': 'G6 technical audit complete',
                      'research_decision_owner': 'GPT / user', 'next_science_authorized': False})
    if sum(p.stat().st_size for p in OUT.iterdir() if p.is_file()) > 16 * 2 ** 20:
        raise RuntimeError('G6 output cap exceeded')
    print(json.dumps({'status': status, 'tests': tests_run, 'counts': counts, 'STOP': True}))


if __name__ == '__main__':
    main()
