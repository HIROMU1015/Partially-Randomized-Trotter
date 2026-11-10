"""Metadata-only R candidate generator; cannot score real normalization or N."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from trottertracks import pr_r_screening as s


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--system',choices=('H4','H6'),required=True)
    parser.add_argument('--factors',required=True,help='Exact rational JSON list, e.g. ["1/2","1","2"], proposal only')
    parser.add_argument('--reference-r-by-round',default='{}',help='Optional explicit metadata reference R lists, e.g. {"12":[8,32]}')
    parser.add_argument('--output',required=True);args=parser.parse_args();out=Path(args.output)
    if out.exists():raise FileExistsError('Preserve previous proposal')
    base=s.A.saved_task(ROOT,args.system);rounds=list(range(13) if args.system=='H4' else range(4,13))
    source=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),path='scripts/plan_pr_r_screening.py')
    law=dict(model='ideal_finite_RTE_law',sampling='existing_independent_finite_RTE_event_definition',
        source=dict(commit=s.A.BASELINE,path='src/trotterlib/rte.py'))
    extra=json.loads(args.reference_r_by_round)
    if not isinstance(extra,dict) or any(str(int(k))!=k for k in extra):raise ValueError('Canonical round keys required')
    result=s.metadata_plan(ROOT,base,rounds,factors=json.loads(args.factors),sampling_law=law,source=source,reference_R={int(k):v for k,v in extra.items()})
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],PR_candidates=len(result['rows']),normalization_evaluations=0,mandatory_stop=True)))


if __name__=='__main__':main()
