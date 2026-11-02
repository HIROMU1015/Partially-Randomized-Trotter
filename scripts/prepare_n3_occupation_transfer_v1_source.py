#!/usr/bin/env python3
"""Archive three fixed SYNTHETIC fixtures and preparation resources only.

Never loads/evaluates the eight transfer cases or the prior twelve cases.
Does not grant scientific authorization and does not create run1.
"""
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests'), str(ROOT / 'scripts')]
SOURCE_FILES = ('src/trottertracks/__init__.py', 'src/trottertracks/representation_exploration/__init__.py',
                'src/trottertracks/representation_exploration/n3_occupation_transfer_v1.py',
                'src/trottertracks/representation_exploration/n3_occupation_transfer_check_v1.py',
                'scripts/run_n3_occupation_transfer_v1.py', 'scripts/verify_n3_occupation_transfer_v1.py',
                'scripts/prepare_n3_occupation_transfer_v1_source.py', 'tests/test_n3_occupation_transfer_v1.py')


def main():
    import numpy as np
    from test_n3_occupation_transfer_v1 import fixture
    from run_n3_occupation_transfer_v1 import MemoryWatch, primary_pool
    from trottertracks.representation_exploration.n3_occupation_transfer_v1 import canon, classify, secondary_separator
    from trottertracks.representation_exploration.n3_occupation_transfer_check_v1 import check_saved
    out = ROOT / 'artifacts/n3_occupation_transfer_v1/2026-10-12/preparation'
    out.mkdir(parents=True, exist_ok=True)
    sources = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCE_FILES}
    fingerprint = hashlib.sha256(canon(sources)).hexdigest()
    cases = [fixture(kind) for kind in ('normal', 'nu1', 'gain')]
    if not all(c['id'].startswith('synthetic_') and c['base'].startswith('synthetic_') for c in cases): raise ValueError('synthetic-only scope')
    start = time.perf_counter(); watch = MemoryWatch(); watch.start()
    try:
        records = primary_pool(cases, fingerprint, watch=watch)
        raw = canon(records); seal = hashlib.sha256(raw).hexdigest()
        (out / 'synthetic_primary_ledger.json').write_bytes(raw)
        secondary = [secondary_separator(c, r, seal) for c, r in zip(cases, records)]
        pack = {'id': 'synthetic_validation_only_not_scientific_transfer', 'cases': cases}
        audit = check_saved(pack, records, secondary, fingerprint, seal)
    finally:
        watch.stop()
    (out / 'synthetic_pack.json').write_bytes(canon(pack))
    (out / 'synthetic_secondary_ledger.json').write_bytes(canon(secondary))
    (out / 'synthetic_saved_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    usage = resource.getrusage(resource.RUSAGE_SELF); children = resource.getrusage(resource.RUSAGE_CHILDREN)
    report = {'status': 'SYNTHETIC_SOURCE_PACKET_PASS', 'source_sha256': sources, 'source_fingerprint': fingerprint,
              'synthetic_case_ids': [c['id'] for c in cases], 'synthetic_primary_sha256': seal,
              'saved_checks': audit['checks'], 'wall_seconds': time.perf_counter() - start,
              'parent_cpu_seconds': usage.ru_utime + usage.ru_stime, 'children_cpu_seconds': children.ru_utime + children.ru_stime,
              'parent_max_rss_bytes_linux': usage.ru_maxrss * 1024, 'largest_child_max_rss_bytes_linux': children.ru_maxrss * 1024,
              'sampled_aggregate_memory': watch.report(), 'python': platform.python_version(), 'numpy': np.__version__,
              'numeric_precision': 'binary64 proposal only; exact rational acceptance', 'workers': 2,
              'thread_environment': {k: os.environ[k] for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS', 'RAYON_NUM_THREADS')},
              'GPU_calls': 0, 'actual_transfer_cases_evaluated': 0, 'prior_development_cases_evaluated': 0,
              'ground_solves': 0, 'quantum_compiles': 0, 'speedup_factor_claimed': False,
              'qualification': 'small fixed synthetic validation and resource observation; no transfer or scalability result',
              'mandatory_stop': True, 'scientific_batch_authorized': False}
    (out / 'synthetic_source_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('source_sha256', 'thread_environment')}, indent=2))


if __name__ == '__main__': main()
