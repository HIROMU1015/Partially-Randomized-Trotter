"""Metadata-only PR R menu planner. This command cannot start science."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from trottertracks import pr_r_dependent_n as model


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--system',choices=('H4','H6'),required=True)
    parser.add_argument('--r-by-round',required=True,help='JSON object, for example {"12":8}; metadata only, not adopted conditions')
    parser.add_argument('--output',required=True);args=parser.parse_args();out=Path(args.output)
    if out.exists():raise FileExistsError('Preserve previous proposal')
    raw=json.loads(args.r_by_round)
    if not isinstance(raw,dict) or any(not isinstance(k,str) or str(int(k))!=k for k in raw):raise ValueError('Canonical round string keys required')
    base=model.saved_task(ROOT,args.system)
    proposal=model.with_round_R(base,{int(k):v for k,v in raw.items()},dict(module='trottertracks.pr_r_dependent_n',baseline=model.BASELINE))
    model.validate_task(proposal['task'],base)
    out.write_text(json.dumps(proposal,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='METADATA_PROPOSAL_ONLY',changed_candidates=len(proposal['changed_candidates']),science_calls=0,mandatory_stop=True)))


if __name__=='__main__':main()
