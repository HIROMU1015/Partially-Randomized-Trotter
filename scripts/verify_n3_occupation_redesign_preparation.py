#!/usr/bin/env python3
"""Audit only saved review witnesses. Cannot launch a transfer batch."""
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks.representation_exploration.n3_occupation_certificate_audit import check_saved


def main():
    start = time.perf_counter()
    cpu = time.process_time()
    result = check_saved(ROOT)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result['resources'] = {
        'wall_seconds': time.perf_counter() - start,
        'cpu_seconds': time.process_time() - cpu,
        'max_rss_bytes_linux': usage.ru_maxrss * 1024,
        'backend': 'stdlib exact rational; matrix-cache LDL from original entries',
        'processes': 1, 'threads': 1, 'gpu_calls': 0,
        'cpu_count': os.cpu_count(), 'cpu_affinity_count': len(os.sched_getaffinity(0)),
        'selected_configuration_reason': 'one saved m=7 witness; multiprocessing overhead unnecessary',
        'speedup_factor_claimed': False,
    }
    destination = ROOT / 'artifacts/n3_occupation_redesign_preparation/2026-10-12/independent_witness_audit.json'
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'checks', 'psd_matrices', 'max_monitored_rational_bits', 'resources')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
