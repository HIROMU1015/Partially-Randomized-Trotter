"""Future integrated comparison CLI. No dispatch without a separate sealed grant."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from h46_acquire_native import sha,write
from h46_acquire_integration import assemble_rows
from h46_higher_pf_adapter import ROOT
from run_h46_higher_pf_comparison import compare_rows,claim_grant


def authorize(plan,grant,plan_path):
    if plan.get('status')!='SEALED_APPROVED' or grant.get('valid') is not True:
        raise ValueError('SEPARATE_INTEGRATED_COMPARISON_APPROVAL_REQUIRED')
    if plan.get('phase')!='B_INTEGRATED_COMPARE' or grant.get('phase')!=plan['phase']:
        raise ValueError('INTEGRATED_SCOPE')
    if sha(plan_path)!=grant.get('plan_sha256') or not grant.get('approval_reference'):
        raise ValueError('INTEGRATED_GRANT_PLAN')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if head!=plan.get('SOURCE') or grant.get('SOURCE')!=head:
        raise ValueError('INTEGRATED_SOURCE')
    if not plan.get('sealed_files') or not grant.get('grant_id'):
        raise ValueError('INTEGRATED_SEAL_REQUIRED')
    for item in plan['sealed_files']:
        if sha(item['path'])!=item['sha256']:raise ValueError('INTEGRATED_INPUT_SOURCE_HASH')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--plan',type=Path)
    parser.add_argument('--grant',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if not args.execute:
        print(json.dumps({'phase':'B_INTEGRATED_COMPARE','status':'UNAPPROVED_READINESS_ONLY'}));return
    if not args.plan or not args.grant:
        parser.exit(2,'SEPARATE_INTEGRATED_COMPARISON_APPROVAL_REQUIRED\n')
    try:
        plan=json.loads(args.plan.read_text());grant=json.loads(args.grant.read_text())
        authorize(plan,grant,args.plan)
        if args.output is None or args.output.exists():raise ValueError('UNUSED_OUTPUT_REQUIRED')
        claim_grant(ROOT/'.higher_pf_integrated_execution_ledger',grant)
        args.output.mkdir()
        try:
            binding=json.loads(Path(plan['binding_path']).read_text())
            rows,metadata=assemble_rows(binding,plan['acquisition_result_root'])
            result=compare_rows(rows)
            authorize(plan,grant,args.plan)
            write(args.output/'result.json',{'comparison':result,'input_metadata':metadata,'mandatory_STOP':True})
        except BaseException as exc:
            write(args.output/'STOP.json',{'reason':str(exc),'retry':False,'mandatory_STOP':True});raise
    except (ValueError,FileExistsError) as exc:parser.exit(2,str(exc)+'\n')


if __name__=='__main__':main()
