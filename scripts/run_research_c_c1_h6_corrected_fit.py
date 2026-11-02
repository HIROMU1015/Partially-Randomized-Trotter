#!/usr/bin/env python3
"""H6 corrected-codec one-fit pilot; default passive preflight only."""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--contract',type=Path,required=True)
    mode=p.add_mutually_exclusive_group()
    for name in ('execute','prepare-worker','fit-worker'):mode.add_argument('--'+name,action='store_true')
    p.add_argument('--output',type=Path);p.add_argument('--fit-python',type=Path);a=p.parse_args()
    from trottertracks import c1_h6_corrected_fit as m
    if not (a.execute or a.prepare_worker or a.fit_worker):
        m.preflight(ROOT,a.contract);print(json.dumps(dict(status='BYTE_PREFLIGHT_PASS',science_calls=0,maximum_calls=m.MAX_CALLS)));return
    if not a.output:p.error('Fresh output required')
    output=a.output.resolve()
    if output.exists() or not output.is_relative_to(ROOT/m.FAMILY):p.error('Fresh task output required')
    for key,leaf in [('XDG_CACHE_HOME',''),('MPLCONFIGDIR','matplotlib'),('NUMBA_CACHE_DIR','numba'),('TMPDIR','tmp')]:
        path=ROOT/'.cache/h6-corrected-fit'/leaf;path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
    if a.prepare_worker or a.fit_worker:
        c=m.preflight(ROOT,a.contract,fit=True)
        expected=ROOT/c['fresh_output']/('prepare' if a.prepare_worker else 'fit')
        if output!=expected.resolve():p.error('Contract-fixed worker output required')
        (m.prepare_worker if a.prepare_worker else m.fit_worker)(ROOT,output,c)
    else:
        if not a.fit_python:p.error('fit-python required')
        c=m.io.read(a.contract)  # full preflight is performed once by run()
        if output!=(ROOT/c['fresh_output']).resolve():p.error('Contract-fixed fresh output required')
        m.run(ROOT,a.contract,output,a.fit_python)

if __name__=='__main__':main()
