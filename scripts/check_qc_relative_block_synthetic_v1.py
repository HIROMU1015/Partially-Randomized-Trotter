#!/usr/bin/env python3
"""Run only the fixed registered-outside synthetic suite, never a molecular pilot."""
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/qc_relative_block_prevalidation/2026-10-13/synthetic'


def main():
    if len(sys.argv) != 1:
        raise SystemExit('No scientific inputs or execution options accepted.')
    OUT.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS', 'RAYON_NUM_THREADS'):
        env[name] = '1'
    env['PYTHONPATH'] = str(ROOT / 'src')
    child = """import resource,unittest,sys
resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
suite=unittest.defaultTestLoader.discover('tests',pattern='test_qc_relative_block_v1.py')
r=unittest.TextTestRunner(verbosity=2).run(suite)
if any(m in sys.modules for m in ('scipy','qiskit','openfermion','pyscf')):raise RuntimeError('SCIENCE_DEPENDENCY_IMPORTED')
sys.exit(0 if r.wasSuccessful() and r.testsRun==13 else 1)
"""
    start = time.perf_counter()
    run = subprocess.run([sys.executable, '-c', child], cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - start
    if len(run.stdout) > 128*1024:
        raise RuntimeError('LOG_BYTES')
    (OUT/'tests.txt').write_bytes(run.stdout)
    rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024
    files = ('src/trottertracks/representation_exploration/qc_relative_block_v1.py',
             'tests/test_qc_relative_block_v1.py', 'scripts/check_qc_relative_block_synthetic_v1.py')
    report = {'status': 'SYNTHETIC_PASS' if run.returncode == 0 and rss <= 512*1024**2 else 'SYNTHETIC_TECHNICAL_STOP',
              'command': 'python scripts/check_qc_relative_block_synthetic_v1.py', 'source_sha256': {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files},
              'tests_expected': 13, 'exit_code': run.returncode, 'wall_seconds': elapsed,
              'child_peak_RSS_bytes_linux': rss, 'workers': 1, 'threads': 1, 'AS_limit_bytes': 2*1024**3,
              'RSS_check_bytes': 512*1024**2, 'log_cap_bytes':128*1024, 'wall_or_CPU_cap':None,
              'reference': 'Fixed n<=5 hand-written signed-density full-Fock synthetic;scaling Taylor powers without eig/SVD.',
              'qualification': 'Classical scalar/matrix and reversible-label semantics;no gate-level arithmetic/control/uncompute circuit validated.',
              'new_molecular_inputs': 0, 'new_eigensolver_calls': 0, 'new_compiles': 0,
              'molecular_science_batch_authorized': False, 'mandatory_stop': True}
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if report['status']=='SYNTHETIC_PASS' else 1)


if __name__ == '__main__':
    main()
