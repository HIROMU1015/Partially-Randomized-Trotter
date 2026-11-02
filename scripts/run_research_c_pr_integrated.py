#!/usr/bin/env python3
"""Fixed integrated experiment; default passive preflight, explicit --execute."""
import argparse,json,os,sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--contract',type=Path,required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--evaluation-worker',action='store_true');a=p.parse_args()
    if a.execute or a.evaluation_worker:
        os.environ['JAX_ENABLE_X64']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
    from trottertracks import c_pr_integrated_runner as m
    c=m.preflight(ROOT,a.contract,evaluation=a.evaluation_worker)
    if not a.execute and not a.evaluation_worker:
        print(json.dumps(dict(status='PASSIVE_PREFLIGHT_PASS',maximum_logical_units=m.MAX_CALLS,scientific_calls=0)));return
    os.environ['JAX_ENABLE_X64']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
    for key,leaf in [('XDG_CACHE_HOME','cache'),('TMPDIR','tmp'),('JAX_COMPILATION_CACHE_DIR','jax')]:
        q=ROOT/'.cache/pr-integrated'/leaf;q.mkdir(parents=True,exist_ok=True);os.environ[key]=str(q)
    if a.evaluation_worker:
        from trottertracks.c_pr_integrated_evaluation import worker
        worker(ROOT,c,ROOT/m.FRESH)
    else:m.run(ROOT,a.contract)


if __name__=='__main__':main()
