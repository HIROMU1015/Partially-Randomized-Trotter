"""Metadata-only CLI. Never loads a Hamiltonian or invokes a scientific runner."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import pr_rpe_rmse_preflight as preflight


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory-only',action='store_true')
    parser.add_argument('--two-time-pilot',action='store_true',help='Fixed B f09 A-adapter, PR two-time metadata only')
    parser.add_argument('--configuration',type=Path)
    parser.add_argument('--schedule',type=Path)
    parser.add_argument('--inventory',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():parser.error('Fresh output required')
    if args.inventory_only:
        if args.configuration or args.schedule or args.inventory:parser.error('Inventory-only takes no supplied design')
        if args.two_time_pilot:
            from trottertracks.pr_rpe_rmse_pilot import pilot_inventory
            result=pilot_inventory(ROOT)
        else:result=preflight.saved_inventory(ROOT)
    else:
        if not all((args.configuration,args.schedule,args.inventory)):
            parser.error('Supply fixed B configuration, A schedule and metadata inventory')
        read=lambda path:json.loads(path.read_text())
        if args.two_time_pilot:
            from trottertracks.pr_rpe_rmse_pilot import prepare_pilot
            result=prepare_pilot(read(args.configuration),read(args.schedule),read(args.inventory),root=ROOT)
        else:
            result=preflight.dry_run(read(args.configuration),read(args.schedule),read(args.inventory),root=ROOT)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


if __name__=='__main__':main()
