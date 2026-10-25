#!/usr/bin/env python3
"""Saved-H6 corrected codec. Default byte preflight; science needs permission."""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract',type=Path,required=True)
    mode=p.add_mutually_exclusive_group();mode.add_argument('--execute',action='store_true');mode.add_argument('--worker',action='store_true')
    p.add_argument('--output',type=Path);p.add_argument('--fit-python',type=Path)
    a=p.parse_args()
    from trottertracks import c1_h6_codec_roundtrip as m
    if not a.execute and not a.worker:
        m.preflight(ROOT,a.contract);print(json.dumps(dict(status='BYTE_PREFLIGHT_PASS',science_calls=0,maximum_calls=20)));return
    if not a.output:p.error('Fresh output required')
    output=a.output.resolve()
    if output.exists() or not output.is_relative_to(ROOT/m.FAMILY):p.error('Fresh task output required')
    for key,leaf in [('XDG_CACHE_HOME',''),('MPLCONFIGDIR','matplotlib'),('NUMBA_CACHE_DIR','numba'),('TMPDIR','tmp')]:
        path=ROOT/'.cache/h6-codec'/leaf;path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
    if a.worker:m.worker(ROOT,output,m.preflight(ROOT,a.contract,worker=True))
    else:
        if not a.fit_python:p.error('fit-python required')
        m.run(ROOT,a.contract,output,a.fit_python)

if __name__=='__main__':main()
