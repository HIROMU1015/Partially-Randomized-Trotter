"""Read-only export/audit/reference join, then local commit and mandatory STOP."""
import argparse
from pathlib import Path
import subprocess
import sys
import pf_common_run_contract as c


def run(private):
    private=Path(private).resolve();plan=c.read(private/'plan_APPROVED.json');metadata=plan['metadata_private']
    c.require(c.read(private/'RECOVERY_SCIENTIFIC_SUCCESS.json')['status']=='PASS','SCIENCE_NOT_COMPLETE')
    for name in ('COST','ACCOUNT'):
        c.require(c.read(private/name/'completion.json')['status']=='PASS' and not (private/name/'STOP.json').exists(),'STAGE_NOT_COMPLETE:'+name)
    c.save(private/'FINAL_RESULT_DRIVER_ENTERED.json',dict(scientific_calls=0,old_grant_reuse=False))
    calls=[['scripts/audit_h6_pf_common_run.py','--private',str(private),'--output','artifacts/resource_applicability/track_a_pf_common_H6_results/2026-10-22'],
        ['scripts/finalize_h6_pf_common_results.py','--common',str(private),'--metadata',metadata],
        ['scripts/check_h246_pf_common_final.py'],['scripts/document_h246_pf_common_final.py']]
    for i,args in enumerate(calls,1):
        with (private/f'final_{i}_stdout.log').open('x') as out,(private/f'final_{i}_stderr.log').open('x') as err:
            result=subprocess.run([sys.executable,'-B',*args],cwd=c.ROOT,stdout=out,stderr=err)
        c.save(private/f'final_{i}_exit.json',dict(exit_code=result.returncode,child_reaped=True,scientific_calls=0,args=args))
        c.require(result.returncode==0,'READ_ONLY_FINAL_STEP_FAILED:'+str(i))
    c.save(private/'FINAL_RESULT_DRIVER_COMPLETE.json',dict(status='PASS',scientific_calls=0))
    subprocess.run([sys.executable,'-B','scripts/close_h246_pf_common_success.py','--common',str(private),
        '--metadata',metadata],cwd=c.ROOT,check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--private',required=True);a=p.parse_args()
    try:run(a.private)
    except BaseException as exc:
        path=Path(a.private)/'FINAL_RESULT_DRIVER_STOP.json'
        if not path.exists():c.save(path,dict(status='MANDATORY_STOP',reason=str(exc),scientific_calls=0))
        raise
