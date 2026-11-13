"""Default remaining-ten metadata plan. Explicit S1/S2 science only after permission."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import h4_pr_rmse_completion as batch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--stage',choices=('S1','S2','S1_S2'))
    p.add_argument('--source',help='Fixed scientific implementation commit, not approval proof')
    a=p.parse_args()
    if a.execute:
        if not a.stage or not a.source:p.error('Explicit stage and fixed SOURCE required for scientific execution')
        r=batch.execute(ROOT,a.out,a.source,a.stage);print(r['status'])
    else:
        if a.out.exists():raise ValueError('Fresh plan output required')
        r=batch.plan(ROOT);a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(r,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
        print(json.dumps(dict(status=r['status'],science_calls=0,mandatory_stop=True)))


if __name__=='__main__':main()
