#!/usr/bin/env python3
"""Separated DF proposal; default passive SOURCE/input/environment preflight."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--fit-python', type=Path)
    p.add_argument('--check-fit-environment', action='store_true')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--fit-worker', choices=('rc2', 'pr'))
    args = p.parse_args()
    if args.check_fit_environment and (args.execute or args.fit_worker):
        p.error('Fit environment check is passive only')
    from trottertracks import c_separated_j_u_pilot as m
    c = m.preflight(ROOT, args.contract,
                    fit=bool(args.fit_worker or args.check_fit_environment))
    if not args.execute and not args.fit_worker:
        print(json.dumps(dict(status='PASSIVE_PREFLIGHT_PASS', scientific_calls=0,
                              proposed_maximum=m.MAX_CALLS)))
        return
    expected = ROOT/m.FRESH
    if args.fit_worker:
        expected = expected/args.fit_worker
    if args.output is None or args.output.resolve() != expected or expected.exists():
        p.error('Fixed fresh output required')
    os.environ['JAX_ENABLE_X64'] = '1'
    # Independent arm/JAX pools, tiny dense BLAS only: avoid oversubscription.
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    for name, leaf in (('XDG_CACHE_HOME', ''), ('MPLCONFIGDIR', 'matplotlib'),
                       ('JAX_COMPILATION_CACHE_DIR', 'jax'), ('TMPDIR', 'tmp')):
        cache = ROOT/'.cache/separated-J-U'/(args.fit_worker or 'parent')/leaf
        cache.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(cache)
    if args.fit_worker:
        m.fit_worker(ROOT, args.output, c, args.fit_worker)
    else:
        if args.fit_python is None:
            p.error('Dedicated fit Python required')
        m.run(ROOT, args.contract, args.output, args.fit_python)


if __name__ == '__main__':
    main()
