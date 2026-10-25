#!/usr/bin/env python3
"""H6 native finite-probe/cost pilot; default passive preflight only."""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--contract',required=True,type=Path)
    p.add_argument('--execute',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args()
    from trottertracks import c1_h6_native as m
    if not a.execute:
        m.preflight(ROOT,a.contract);print(json.dumps(dict(status='BYTE_PREFLIGHT_PASS',science_calls=0,maximum_calls=m.MAX_CALLS)));return
    if not a.output:p.error('Fresh output required')
    c=m.io.read(a.contract);out=a.output.resolve()
    if out.exists() or out!=(ROOT/c['fresh_output']).resolve() or not out.is_relative_to(ROOT/m.FAMILY):p.error('Fresh contract-fixed output required')
    for key,leaf in [('XDG_CACHE_HOME',''),('MPLCONFIGDIR','matplotlib'),('NUMBA_CACHE_DIR','numba'),('TMPDIR','tmp')]:
        path=ROOT/'.cache/h6-native'/leaf;path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
    m.run(ROOT,a.contract,out)

if __name__=='__main__':main()
