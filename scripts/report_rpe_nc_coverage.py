"""Write immutable-reference N/C coverage; no science or N*C/G execution."""
import argparse
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from trottertracks.rpe_nc_coverage import build_coverage, H4_PUBLISHED_N


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',required=True)
    parser.add_argument('--h4-n-commit');parser.add_argument('--h4-n-path');parser.add_argument('--baseline-h4-n',action='store_true')
    args=parser.parse_args();out=Path(args.output_dir)
    if bool(args.h4_n_commit)!=bool(args.h4_n_path) or args.baseline_h4_n and args.h4_n_commit:
        parser.error('Supply both pinned H4 N commit/path or baseline-only, not both')
    selection=None if args.baseline_h4_n else dict(commit=args.h4_n_commit,path=args.h4_n_path) if args.h4_n_commit else H4_PUBLISHED_N
    if out.exists():raise FileExistsError('Fresh technical coverage output required')
    result=build_coverage(ROOT,h4_handoff=selection);out.mkdir(parents=True)
    (out/'COVERAGE.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    fields=['system','m','arm','T','q_RPE','q_PF','candidate_id','N_status','N_X_decimal','N_Y_decimal','direct_C_available',
        'saved_model_C_available','provider_domain_matches','unvalidated_long_q','eligible_finite_scenarios','categories']
    with (out/'ROUND_COVERAGE.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fields,lineterminator='\n');writer.writeheader()
        for system in result['systems'].values():
            for row in system['rows']:
                writer.writerow({k:row['candidate']['T'] if k=='T' else row['candidate']['id'] if k=='candidate_id' else
                    '|'.join(row[k]) if isinstance(row[k],list) else row[k] for k in fields})
    print(json.dumps(dict(status='STATIC_COVERAGE_WRITTEN',rows=sum(len(v['rows']) for v in result['systems'].values()),new_scientific_calls=0,N_times_C_calls=0)))


if __name__=='__main__':main()
