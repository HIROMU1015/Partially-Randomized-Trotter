#!/usr/bin/env python3
"""Default passive preflight; real H4 small-step trial needs separate permission."""
import argparse,json,os,sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    from trottertracks import c_pr_small_step_pilot as m
    m.preflight(ROOT,a.contract)
    if not a.execute:
        print(json.dumps(dict(status='PASSIVE_PREFLIGHT_PASS',scientific_calls=0,proposed_maximum=m.MAX_CALLS)));return
    os.environ['JAX_ENABLE_X64']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
    for key,leaf in [('XDG_CACHE_HOME','cache'),('TMPDIR','tmp'),('JAX_COMPILATION_CACHE_DIR','jax')]:
        q=ROOT/'.cache/pr-small-step'/leaf;q.mkdir(parents=True,exist_ok=True);os.environ[key]=str(q)
    m.run(ROOT,a.contract)


if __name__=='__main__':main()
