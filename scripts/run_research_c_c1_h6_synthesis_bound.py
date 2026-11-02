#!/usr/bin/env python3
"""Saved H6 complete-basis bound; passive preflight unless separately permitted."""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--contract',required=True,type=Path)
    p.add_argument('--execute',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args()
    from trottertracks import c1_h6_synthesis_bound as m
    c=m.preflight(ROOT,a.contract)
    if not a.execute:print(json.dumps(dict(status='BYTE_PREFLIGHT_PASS',science_calls=0,maximum_calls=m.MAX_CALLS)));return
    if not a.output or a.output.resolve()!=(ROOT/c['fresh_output']).resolve() or a.output.exists():p.error('Fixed fresh output required')
    if not a.output.resolve().is_relative_to(ROOT/m.FAMILY):p.error('Contract family required')
    for key,leaf in [('XDG_CACHE_HOME',''),('MPLCONFIGDIR','matplotlib'),('NUMBA_CACHE_DIR','numba'),('TMPDIR','tmp')]:
        path=ROOT/'.cache/h6-synthesis-bound'/leaf;path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
    m.run(ROOT,a.contract,a.output)


if __name__=='__main__':main()
