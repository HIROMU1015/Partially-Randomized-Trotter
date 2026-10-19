"""Metadata-only completion audit. No scientific arrays, costs, or statistics."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import h4_tol_main_contract as c
from prepare_h4_tol_main import PUBLIC


def audit(private):
    private=Path(private).resolve()
    source=c.read(PUBLIC/'SOURCE_provenance.json')['implementation_SOURCE']
    closure=c.source_closure()
    for name, expected in closure.items():
        blob=subprocess.check_output(['git','show',source+':'+name],cwd=c.ROOT)
        c.require(hashlib.sha256(blob).hexdigest()==expected,'SOURCE_GIT:'+name)
    rows=c.read(private/'preservation_manifest_PRIVATE.json')
    for row in rows:
        for path in {row['path'],row['copy']}:
            c.require(c.sha(path)==row['sha256'],'PRESERVATION_BYTES:'+row['role'])
    for stage in ('D','A','S_STATS','P'):
        contract=c.read(private/(stage+'_contract_DRAFT_PRIVATE.json'))
        projection=c.read(PUBLIC/(stage+'_contract_DRAFT.json'))
        grant=c.read(private/(stage+'_grant_INVALID_PRIVATE.json'))
        c.require(c.digest(contract)==projection['contract_digest']==grant['contract_digest'],'CONTRACT_DIGEST')
        c.require(contract['plan']==c.plan(stage) and contract['source_hashes']==closure,'COMPLETE_CONTRACT_SOURCE_PLAN')
        c.same_binding(contract['binding'])
        c.require(not contract['approved_by_user'] and not contract['runtime_authorization']
            and not grant['approved_by_user'] and not grant['runtime_authorization']
            and contract['actual_scientific_calls']==0 and not contract['next_stage_authorized'],'NO_SCIENTIFIC_AUTHORIZATION')
        c.require(not Path(contract['output']).exists() and not Path(contract['lock']).exists(),'NO_SCIENTIFIC_OUTPUT_MARKER')
        c.require(not Path(contract['output']).parent.exists(),'NO_NATIVE_ATTEMPT_ROOT')
        c.require(contract['worker_cpus']==[] and not contract['fresh_resources']['admitted'],'FRESH_RESOURCE_PENDING')
        for row in contract['inputs']:
            c.require(c.sha(row['path'])==row['sha256'],'INPUT_BYTES')
        if stage in ('S_STATS','P'):
            c.require(contract['SOURCE'] is None and contract['plan']['P_contract']['n'] is None,'PENDING_FINAL_CONTRACT')
        try: c.authorize(contract,grant)
        except ValueError as error:
            c.require(str(error)=='SEPARATE_EXPLICIT_SCIENTIFIC_APPROVAL_REQUIRED','EARLY_APPROVAL_REFUSAL')
        else: raise ValueError('INVALID_GRANT_ACCEPTED')
        if stage=='D':
            c.verify_provider_tree(contract)
            for group in ('provider_hashes','runtime_hashes'):
                for name,expected in contract[group].items(): c.require(c.sha(name)==expected,'ACTUAL_EXECUTION_CLOSURE')
            paths={r['path'] for r in contract['inputs']}
            c.require({contract['native_config'],contract['FP_evidence'],contract['saved_reference'],
                contract['font_seed']['path']}<=paths,'ACTUAL_METADATA_BOUND')
    tests=c.read(PUBLIC/'artificial_test_report.json')
    c.require(tests['successful'] and tests['new_tests']>=30 and tests['real_scientific_calls']==0,'ARTIFICIAL_TESTS')
    c.require(tests['test_file_sha256']==c.sha(c.ROOT/'tests/test_h4_tol_main.py'),'TEST_SOURCE')
    refusals=c.read(PUBLIC/'grant_refusal_report.json')
    c.require(refusals['cases']==8 and refusals['all_rejected_before_science'] and not refusals['science_output_or_marker'],'REAL_DRAFT_REFUSAL_CASES')
    authorization=c.read(private/'Phase0_authorization_PRIVATE.json')
    c.require(authorization['scope']=='TECHNICAL_PREPARATION_ONLY' and not authorization['scientific_execution_authorized'],'PHASE0_SCOPE')
    for path in PUBLIC.glob('*.json'):
        raw=path.read_text()
        c.require('/home/' not in raw and '.h4_tol_main_private/' not in raw,'PUBLIC_PRIVATE_PATH:'+path.name)
        obj=json.loads(raw)
        def check(value):
            if isinstance(value,dict):
                for key,item in value.items():
                    c.require(not(key in ('seed','master_seed','step_seeds','events') and item is not None),'PUBLIC_PRIVATE_SEED_OR_EVENTS')
                    check(item)
            elif isinstance(value,list):
                for item in value: check(item)
        check(obj)
    result=dict(status='PHASE0_COMPLETE_MANDATORY_STOP',SOURCE=source,scientific_execution_authorized=False,
        actual_new_scientific_calls=0,actual_new_numerical_analyses=0,new_rankings_or_cost_ratios=0,
        preservation_records=len(rows),origin_and_copy_SHA_match=True,
        own_SOURCE_files=len(closure),provider_files=len(contract['provider_hashes']),runtime_files=len(contract['runtime_hashes']),
        artificial_tests=tests['total_tests'],new_artificial_tests=tests['new_tests'],invalid_grant_CLI_cases=8,
        next_stage_authorized=False,automatic_execution=False,remote_push=False,mandatory_STOP=True,
        state_solver_policy_identical=False,P_n_SOURCE_seed_budget='PENDING_ACCEPTED_D_A_AND_SEPARATE_APPROVAL',
        author_checks_only=True,independent_reviewer_used=False)
    c.save(PUBLIC/'completion_audit.json',result)
    c.save(PUBLIC/'phase0_status.json',result)
    c.save(PUBLIC/'artifact_manifest.json',dict(schema='BYTE_SHA_ONLY',files={p.name:c.sha(p) for p in sorted(PUBLIC.glob('*.json'))
        if p.name!='artifact_manifest.json'},scientific_results_generated=False))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private',type=Path,default=c.ROOT/'.h4_tol_main_private')
    print(json.dumps(audit(parser.parse_args().private)))
