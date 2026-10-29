#!/usr/bin/env python3
"""Rank7 one-fit study; default passive bytes/packages preflight."""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract',type=Path,required=True)
    modes=p.add_mutually_exclusive_group()
    modes.add_argument('--execute',action='store_true')
    modes.add_argument('--worker',choices=['prepare','fit'])
    p.add_argument('--output',type=Path);p.add_argument('--fit-python',type=Path)
    a=p.parse_args()
    from trottertracks import c_cdf7_onefit as m
    if not a.execute and not a.worker:
        m.preflight(ROOT,a.contract)
        print(json.dumps(dict(status='PASSIVE_PREFLIGHT_PASS',science_calls=0,maximum_calls=m.MAX_CALLS)))
        return
    if not a.output:p.error('Fresh output required')
    for name,leaf in [('XDG_CACHE_HOME',''),('MPLCONFIGDIR','matplotlib'),('NUMBA_CACHE_DIR','numba'),
                      ('JAX_COMPILATION_CACHE_DIR','jax'),('TMPDIR','tmp')]:
        path=ROOT/'.cache'/leaf;path.mkdir(parents=True,exist_ok=True);os.environ[name]=str(path)
    output=a.output.resolve()
    c=m.preflight(ROOT,a.contract,fit=True) if a.worker else m.io.read(a.contract)
    expected=ROOT/c['fresh_output']
    if a.worker:
        if output!=expected/a.worker or output.exists() or (expected/'STOP.json').exists():
            p.error('Fresh worker within active fixed run required')
        if a.worker=='prepare':m.prepare_worker(ROOT,output,c)
        else:m.fit_worker(ROOT,output,c)
    else:
        if not a.fit_python:p.error('task fit-python required')
        if output!=expected or output.exists():p.error('Fresh fixed run01 only')
        m.run(ROOT,a.contract,output,a.fit_python)


if __name__=='__main__':main()
