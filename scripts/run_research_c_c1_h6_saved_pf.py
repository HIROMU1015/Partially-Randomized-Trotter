#!/usr/bin/env python3
"""Saved H6 rank14 sector PF; default passive preflight, science separately permitted."""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--contract',type=Path,required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args()
    from trottertracks import c1_h6_saved_pf as m
    if not a.execute:
        m.preflight(ROOT,a.contract);print(json.dumps(dict(status='BYTE_PREFLIGHT_PASS',science_calls=0,maximum_calls=m.MAX_CALLS)));return
    if not a.output:p.error('Fresh output required')
    c=m.io.read(a.contract);output=a.output.resolve()
    if (output.exists() or output!=(ROOT/c['fresh_output']).resolve()
            or not output.is_relative_to(ROOT/m.FAMILY)):p.error('Contract-fixed fresh task output required')
    for key,leaf in [('XDG_CACHE_HOME',''),('MPLCONFIGDIR','matplotlib'),('NUMBA_CACHE_DIR','numba'),('TMPDIR','tmp')]:
        path=ROOT/'.cache/h6-saved-pf'/leaf;path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
    m.run(ROOT,a.contract,output)

if __name__=='__main__':main()
