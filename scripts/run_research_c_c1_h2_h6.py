#!/usr/bin/env python3
"""H2/H6 preparation; default metadata preflight, science only after permission."""
import argparse,json,os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--molecule',type=int,choices=[2,6]);p.add_argument('--contract',type=Path)
    p.add_argument('--execute',action='store_true');p.add_argument('--output',type=Path)
    p.add_argument('--fit-python',type=Path);p.add_argument('--fit-worker',action='store_true')
    p.add_argument('--input',type=Path);p.add_argument('--rank',type=int);p.add_argument('--optimize',action='store_true')
    a=p.parse_args()
    from trottertracks import c1_molecule_extension as m
    if a.fit_worker:
        if not a.input or not a.output or not a.rank:p.error('worker input/output/rank required')
        m.factor_worker(a.input,a.output,a.rank,a.optimize);return
    if not a.molecule or not a.contract:p.error('molecule/contract required')
    if a.execute!=bool(a.output) or (a.execute and not a.fit_python):p.error('execute requires fresh output and fit-python')
    c=m.preflight(ROOT,a.contract,a.molecule)
    if not a.execute:
        print(json.dumps(dict(status='EXTENSION_BYTE_PREFLIGHT_PASS',science_calls=0,
            molecule=m.specification(a.molecule),maximum_calls=sum(m.call_limits(a.molecule).values()))));return
    out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/m.FAMILY):p.error('fresh task output required')
    for key,leaf in [('XDG_CACHE_HOME',''),('MPLCONFIGDIR','matplotlib'),('NUMBA_CACHE_DIR','numba'),('TMPDIR','tmp')]:
        path=ROOT/'.cache/h2-h6'/leaf;path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
    try:m.run(a.molecule,out,a.fit_python)
    finally:
        if out.exists():m.io.save(out/'contract.json',c)


if __name__=='__main__':main()
