"""Prepare an INVALID corrected D contract. Never retry the stopped run."""
import argparse
from pathlib import Path
import subprocess

import h4_dp_contract as c
from run_h4_dp_verification import worker_context_spec

ROOT=Path(__file__).resolve().parents[1]
FILES=['scripts/h4_dp_contract.py','scripts/h4_dp_science.py',
       'scripts/run_h4_dp_verification.py','scripts/h4_dp_witness.py',
       'scripts/prepare_h4_dp_phase0.py','scripts/seal_h4_dp_phase0.py',
       'scripts/prepare_h4_d_binding_retry.py']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-worktree',type=Path,required=True)
    parser.add_argument('--source',required=True)
    args=parser.parse_args()
    previous=args.previous_worktree.resolve()
    old_private=previous/'.h4_dp_phase0_private'
    old_run=old_private/'D_run01_20261019'
    private=ROOT/'.h4_dp_phase0_private'
    public=ROOT/'artifacts/resource_applicability/track_a_h4_d_binding_fix/2026-10-19'
    stop=c.read(old_run/'output/STOP.json')
    c.require(stop['status']=='TECHNICAL_STOP' and stop['reason']=='ValueError:PREPARATION_BINDING',
              'EXPECTED_HISTORICAL_STOP')
    c.require(not (old_run/'output/result.json').exists(),'NO_OLD_SCIENTIFIC_RESULT')
    c.require(not list((old_run/'output').glob('tasks/*/calls_ledger.json')),'NO_OLD_NATIVE_RESERVATIONS')
    spec=c.read(old_private/'D_contract_DRAFT_PRIVATE.json')
    before=dict(spec['source_hashes'])
    for name,h in before.items():c.require(c.sha(previous/name)==h,'DO_NOT_MODIFY_OLD_SOURCE')
    spec['source_hashes']={name:c.sha(ROOT/name) for name in FILES}
    for name,h in spec['source_hashes'].items():
        raw=subprocess.check_output(['git','show',args.source+':'+name],cwd=ROOT)
        c.require(__import__('hashlib').sha256(raw).hexdigest()==h,'NEW_SOURCE_GIT_BINDING')
    for item in spec['inputs']:c.require(c.sha(item['path'])==item['sha256'],'SAVED_INPUT_CHANGED')
    run=private/'D_retry01_PENDING_AUTHORIZATION';run.mkdir(parents=True,exist_ok=True)
    c.require(not (run/'output').exists() and not (run/'ONE_SHOT.json').exists(),'NEW_RUN_UNUSED')
    spec.update(SOURCE=args.source,status='DRAFT_UNAUTHORIZED',approved_by_user=False,
        runtime_authorization=False,workers=None,output=str(run/'output'),lock=str(run/'ONE_SHOT.json'),
        requires_new_explicit_user_approval=True,prior_SOURCE=c.read(old_run/'spec_SEALED_PRIVATE.json')['SOURCE'],
        prior_STOP_sha256=c.sha(old_run/'output/STOP.json'),prior_native_main_calls=0,
        previous_one_shot_consumed=True,previous_preparation_attempt_count_not_recorded=True,
        worker_manifest_replication_removed=True)
    c.save(private/'D_corrected_contract_DRAFT_PRIVATE.json',spec)
    grant=dict(schema='INVALID_CORRECTED_D_GRANT',stage='D',SOURCE=args.source,spec_digest=c.digest(spec),
        approved_by_user=False,runtime_authorization=False,one_shot=True,retry=False,next_stage_authorized=False,
        approval_receipt=None,approval_receipt_sha256=None)
    c.save(private/'D_corrected_grant_INVALID.json',grant)
    projection={k:spec[k] for k in ['stage','SOURCE','status','approved_by_user','runtime_authorization',
        'calls','prior_SOURCE','prior_native_main_calls','requires_new_explicit_user_approval',
        'previous_one_shot_consumed','previous_preparation_attempt_count_not_recorded',
        'worker_manifest_replication_removed']}
    projection.update(private_spec_digest=c.digest(spec),source_hashes=spec['source_hashes'],
        candidate_count=len(spec['tasks']),input_hashes=[{k:v for k,v in x.items() if k!='path'} for x in spec['inputs']],
        previous_source_files_unchanged=True,worker_payload_keys=sorted(worker_context_spec(spec)))
    c.save(public/'D_corrected_contract_DRAFT.json',projection)
    c.save(public/'D_corrected_grant_INVALID.json',grant)
    c.save(public/'technical_source_binding.json',dict(SOURCE=args.source,source_hashes=spec['source_hashes'],
        old_SOURCE_preserved=spec['prior_SOURCE'],new_science_after_STOP=0,remote_push=False))
    print('CORRECTED_D_DRAFT_PREPARED; INVALID_GRANT; NO_SCIENTIFIC_RETRY; SOURCE='+args.source)


if __name__=='__main__':main()
