"""Fixed twelve-candidate P1 only. Default is static preflight, --execute is science."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from trottertracks import pr_r_detailed_n_p1 as run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--output-dir')
    args=parser.parse_args();packet=run.preflight(ROOT)
    if not args.execute:
        print('P1_PREFLIGHT_PASS: fixed12, saved baseline scalar3, new mean/control9, new reference2; science0')
        return
    if not args.output_dir:parser.error('--execute requires --output-dir')
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if any((ROOT/path).read_bytes()!=run.blob(ROOT,source,path) for path in packet['sources']):
        raise ValueError('Scientific SOURCE must match the recorded Git commit')
    result=run.execute(packet,source,Path(args.output_dir))
    print(result['status'],result['scope_observed'],flush=True)
    if result['status']!='P1_COMPLETE_CONDITIONAL_N':sys.exit(2)


if __name__=='__main__':main()
