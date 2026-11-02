#!/usr/bin/env python3
"""Passive preflight; explicit opt-in only after separate chat science permission."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract',required=True,type=Path)
    group=p.add_mutually_exclusive_group()
    group.add_argument('--execute',action='store_true')
    group.add_argument('--evaluation-worker',action='store_true')
    p.add_argument('--science-permitted',action='store_true',help='deliberate execution switch after chat permission;not proof of permission')
    a=p.parse_args()
    from trottertracks import c_near_practical as m
    if (a.execute or a.evaluation_worker) and not a.science_permitted:
        raise PermissionError('SEPARATE_EXPLICIT_SCIENTIFIC_PERMISSION_REQUIRED')
    if not a.execute and not a.evaluation_worker:
        m.preflight(ROOT,a.contract)
        print(json.dumps(dict(status='PASSIVE_READY_UNAPPROVED',maximum_logical_units=m.MAX_CALLS,scientific_calls=0)))
        return
    # Own process/task-local cache only. Two independent arms, tiny fixed-U
    # contractions: one BLAS thread per arm avoids nested oversubscription.
    os.environ['JAX_ENABLE_X64']='1'
    os.environ['OPENBLAS_NUM_THREADS']='1'
    os.environ['MKL_NUM_THREADS']='1'
    for name,leaf in [('XDG_CACHE_HOME','cache'),('JAX_COMPILATION_CACHE_DIR','jax'),('TMPDIR','tmp')]:
        path=ROOT/'.cache/near-practical'/leaf;path.mkdir(parents=True,exist_ok=True);os.environ[name]=str(path)
    if a.evaluation_worker:
        c=m.preflight(ROOT,a.contract,environment=True,evaluation=True)
        m.prior.evaluation_worker(ROOT,c,ROOT/m.FRESH)
    else:m.run(ROOT,a.contract,scientific_permission=True)


if __name__=='__main__':main()
