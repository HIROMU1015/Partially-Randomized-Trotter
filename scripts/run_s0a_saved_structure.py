"""Prepare S0a metadata, or run separately authorized eight saved-path S counts."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import s0a_saved_structure as s


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','execute'])
    parser.add_argument('--plan',required=True)
    parser.add_argument('--output')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if args.stage=='prepare':
        if args.execute:parser.error('Preparation does not execute science')
        source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        plan=s.prepare(ROOT,source)
        from trottertracks.r_price_phase1_runner import write
        write(args.plan,plan)
        print(json.dumps(dict(status=plan['status'],saved_paths=4,new_S_executed=0)));return
    if not args.execute or not args.output:parser.error('--execute and a fresh --output required; user scientific permission is separate')
    print(json.dumps(s.execute(ROOT,json.loads(Path(args.plan).read_text()),args.output,execute=True)))


if __name__=='__main__':main()
