"""Default metadata preflight. --execute is separately permitted m1 science only."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import h4_pr_m1_rmse as m1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--execute',action='store_true',help='Real saved-value science; separate user permission required')
    p.add_argument('--source',help='Fixed implementation commit for scientific provenance, not approval proof')
    p.add_argument('--full-pf-sources',action='store_true',help='Passive Full PF catalog only; no PF scientific entry')
    args=p.parse_args()
    if args.full_pf_sources and args.execute:p.error('Full PF mode is metadata only')
    if args.execute:
        if not args.source:p.error('--execute requires fixed --source')
        result=m1.execute(ROOT,args.out,args.source)
        print(json.dumps({k:result[k] for k in ('status','configuration_id','mandatory_stop')}))
    else:
        if args.out.exists():raise ValueError('Fresh preflight output required')
        result=m1.full_pf_inventory(ROOT) if args.full_pf_sources else m1.prepare(ROOT)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(status=result.get('status','FULL_PF_SOURCE_CATALOG_ONLY'),science_calls=0,mandatory_stop=True)))


if __name__=='__main__':main()
