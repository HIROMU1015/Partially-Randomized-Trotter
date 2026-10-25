#!/usr/bin/env python3
"""Fixed saved-H6 initialization diagnosis, never a fitter or PF runner."""
from pathlib import Path
import argparse,json,os,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--fit-python',type=Path)
    args=parser.parse_args()
    from trottertracks import c1_h6_initialization as m
    if not args.execute and not args.worker:
        m.preflight(ROOT,args.contract)
        print(json.dumps(dict(status='BYTE_PREFLIGHT_PASS',maximum_calls=35,new_fit=0)));return
    if not args.output:parser.error('Fresh output required')
    out=args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/m.FAMILY):parser.error('Fresh task output required')
    for name,leaf in [('XDG_CACHE_HOME',''),('MPLCONFIGDIR','matplotlib'),('NUMBA_CACHE_DIR','numba'),('TMPDIR','tmp')]:
        p=ROOT/'.cache/h6-initialization'/leaf;p.mkdir(parents=True,exist_ok=True);os.environ[name]=str(p)
    if args.worker:m.fit_diagnosis(ROOT,out,m.preflight(ROOT,args.contract,fit=True))
    else:
        if not args.fit_python:parser.error('fit-python required')
        m.run(ROOT,args.contract,out,args.fit_python)

if __name__=='__main__':main()
