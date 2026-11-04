#!/usr/bin/env python3
"""Passive by default; execute only a fixed separately permitted finite pilot."""
import argparse,json,os,sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');a=p.parse_args()
    from trottertracks import c_fixed_word_pilot as m
    if not a.execute:
        c=m.preflight(ROOT,ROOT/m.CONTRACT);print(json.dumps(dict(status='PASSIVE_FINITE_PILOT_READY',scientific_calls=0,maximum_logical_units=m.MAX_CALLS)))
    else:
        os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
        for k,leaf in [('MPLCONFIGDIR','mpl'),('XDG_CACHE_HOME','cache'),('TMPDIR','tmp')]:
            path=ROOT/'.cache/fixed-word'/leaf;path.mkdir(parents=True,exist_ok=True);os.environ[k]=str(path)
        m.run(ROOT,ROOT/m.CONTRACT,execute_science=True)
