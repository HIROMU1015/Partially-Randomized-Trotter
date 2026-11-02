"""Run read-only audits and result exports after successful scientific receipts."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def save(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')


def main(common,metadata):
    common=Path(common);root=Path(__file__).resolve().parents[1]
    save(common/'FINAL_RESULT_DRIVER_ENTERED.json',dict(scientific_calls=0,retry=False))
    while not (common/'SUCCESSOR_DRIVER_COMPLETE.json').exists():
        if (common/'SUCCESSOR_DRIVER_STOP.json').exists():raise RuntimeError('SCIENTIFIC_DRIVER_STOP_NO_FINAL_RESULT_CLAIM')
        time.sleep(2)
    calls=[['scripts/audit_h6_pf_common_run.py','--private',str(common),'--output','artifacts/resource_applicability/track_a_pf_common_H6_results/2026-10-22'],
           ['scripts/finalize_h6_pf_common_results.py','--common',str(common),'--metadata',str(metadata)],
           ['scripts/check_h246_pf_common_final.py'],['scripts/document_h246_pf_common_final.py']]
    for i,args in enumerate(calls,1):
        with (common/f'final_{i}_stdout.log').open('x') as out,(common/f'final_{i}_stderr.log').open('x') as err:
            result=subprocess.run([sys.executable,'-B',*args],cwd=root,stdout=out,stderr=err)
        save(common/f'final_{i}_exit.json',dict(exit_code=result.returncode,reaped=True,args=args,scientific_calls=0))
        if result.returncode:raise RuntimeError('READ_ONLY_FINAL_STEP_FAILED:'+str(i))
    save(common/'FINAL_RESULT_DRIVER_COMPLETE.json',dict(status='PASS',scientific_calls=0,next_scientific_stage_started=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--common',required=True);parser.add_argument('--metadata',required=True);args=parser.parse_args()
    try:main(args.common,args.metadata)
    except BaseException as exc:
        p=Path(args.common)/'FINAL_RESULT_DRIVER_STOP.json'
        if not p.exists():save(p,dict(reason=str(exc),scientific_calls=0))
        raise
