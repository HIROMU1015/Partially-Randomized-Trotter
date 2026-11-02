#!/usr/bin/env python3
"""Passive proposal inspection; science requires separate user permission."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', required=True, type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--execute', action='store_true')
    group.add_argument('--evaluation-worker', action='store_true')
    args = parser.parse_args()
    from trottertracks import c_u_step_control_pilot as pilot
    if not args.execute and not args.evaluation_worker:
        pilot.preflight(ROOT, args.contract)
        print(json.dumps(dict(status='PASSIVE_PROPOSAL_VALID', scientific_calls=0,
                              maximum_logical_units=pilot.MAX_CALLS,
                              execution_requires_separate_user_permission=True)))
        return
    # Per-process settings, not shared environment or OS configuration.
    os.environ['JAX_ENABLE_X64'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    for name, leaf in (('XDG_CACHE_HOME', 'cache'), ('TMPDIR', 'tmp'),
                       ('JAX_COMPILATION_CACHE_DIR', 'jax')):
        directory = ROOT / '.cache/u-step-control' / leaf
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(directory)
    if args.evaluation_worker:
        c = pilot.preflight(ROOT, args.contract, environment=True, evaluation=True)
        pilot.evaluation_worker(ROOT, c, ROOT / pilot.FRESH)
    else:
        pilot.run(ROOT, args.contract)


if __name__ == '__main__':
    main()
