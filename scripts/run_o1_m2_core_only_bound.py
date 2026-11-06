"""Prepared B-relative norm/energy-bound entry; molecular execution unapproved.

No defaults select molecular files. Help/import do not decode input. Execute
only under a later explicit user scope in a separate science worktree.
"""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
BASE='f891f6e3265e4f0124c598fddfb7775bc1eddd1b'


def source_provenance():
    fixed=('src/trottertracks/o1_m2_hermitian_b.py','src/trottertracks/o1_m2_residual_adapter.py')
    for p in fixed:
        if (ROOT/p).read_bytes()!=subprocess.check_output(['git','show',BASE+':'+p],cwd=ROOT):
            raise ValueError('Fixed reused SOURCE differs: '+p)
    files=('scripts/run_o1_m2_core_only_bound.py','src/trottertracks/o1_m2_core_only_bound.py',*fixed)
    return dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        reused_base_commit=BASE,files_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--hh-omission-budget',type=float)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError('Fresh result required')
    source=source_provenance()  # Fixed B/adapter byte check before input decode.
    runtime=dict(python=sys.version,host=platform.node(),
        packages={p:importlib.metadata.version(p) for p in ('numpy','scipy','openfermion','qiskit')})
    from trottertracks.o1_m2_residual_adapter import load_saved_pair,UPDATE005
    from trottertracks.o1_m2_core_only_bound import core_only_bounds
    raw=load_saved_pair(args.core,args.input)
    result=core_only_bounds(raw,hh_omission_budget=args.hh_omission_budget)
    result['source']=source
    result['runtime']=runtime
    result['saved_binding']=asdict(UPDATE005)
    result['mandatory_stop']=True
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    return 0


if __name__=='__main__':raise SystemExit(main())
