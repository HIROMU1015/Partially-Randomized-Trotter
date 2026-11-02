"""Prepare an invalid H8 continuation draft, or seal metadata after quiescence.

Never stops a process, creates a valid grant, restores preparations or launches
scientific work. --finalize still produces an invalid execution grant.
"""
from copy import deepcopy
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import pr_cost_batch as b
import pr_cost_candidate_parallel as p
c=b.old


def prepare(origin_binding,destination,source,*,finalize=False,handoff_receipt=None):
    destination=c.home_path(destination);destination.mkdir(parents=True,exist_ok=False)
    origin=c.read(origin_binding)
    old=c.read(origin['origin_spec']);data=c.read(old['payload']);tasks=data['new_seed_registry']['tasks']
    c.require(old['scope']=='H8_A_COST' and old['SOURCE']==origin['origin_SOURCE'],'ORIGIN_H8_SCOPE')
    origin['origin_spec_sha256']=c.sha(origin['origin_spec'])
    # Operational flags from the initial live observation are not approval.
    origin.pop('approval_to_stop_or_launch',None);origin.pop('read_only',None)
    p.verify_origin(origin,tasks,old['binding'])
    c.save(destination/'origin_binding.json',origin)
    c.save(destination/'frozen_payload.json',data)
    spec=deepcopy(old)
    spec.update(SOURCE=source,source_hashes={name:c.sha(b.ROOT/name) for name in b.FILES},
        A_scheduling=p.POLICY,continuation_origin=origin,seed_derivation_SOURCE=origin['origin_SOURCE'],
        native_kernel_sha256=c.sha(b.ROOT/'scripts/pr_cost_native.py'),
        payload=str(destination/'frozen_payload.json'),output=str(destination/'output'),
        lock=str(destination/'ONE_SHOT'),resume_manifest=None,technical_sealed=False,
        approved=False,runtime_authorization=False,allowed_cpus=[],retry=False,next_stage_authorized=False)
    remaining=list(tasks);accepted=0;manifest=None
    if finalize:
        handoff=None
        if handoff_receipt:
            raw=c.read(handoff_receipt)
            c.require(raw.get('handoff_authorized') is True and raw.get('origin_SOURCE')==origin['origin_SOURCE']
                      and raw.get('new_SOURCE')==source and raw.get('scope')=='H8_A_COST',
                      'EXPLICIT_FINAL_HANDOFF_RECEIPT')
            handoff=dict(raw,receipt=p.file_row(handoff_receipt))
        receipts=[Path(origin['origin_output'])/phase/'parallel_receipt.json'
                  for phase in ('first_control','remaining') if (Path(origin['origin_output'])/phase).exists()]
        c.require(receipts and all(path.exists() for path in receipts),'ALL_ORIGIN_MAP_NATIVE_RECEIPTS_REQUIRED')
        manifest=p.build_manifest(origin,tasks,old['binding'],receipts,handoff=handoff)
        c.save(destination/'resume_manifest.json',manifest)
        accepted=len(manifest['accepted']);remaining=[t for t in tasks if t['id'] in manifest['remaining_task_ids']]
        spec.update(resume_manifest=str(destination/'resume_manifest.json'),technical_sealed=True)
    rows=[deepcopy(row) for row in old['inputs'] if row['path']!=old['payload']]
    paths=[destination/'origin_binding.json',destination/'frozen_payload.json',Path(origin['origin_spec'])]
    if manifest:paths.append(destination/'resume_manifest.json')
    for path in paths:rows.append(dict(p.file_row(path),role='continuation_'+str(len(rows))))
    spec['inputs']=rows
    budget=dict(trajectory=len(remaining),native_evolution=len(remaining),wrapper_compile=2*len(remaining),
        occurrence=sum(t['cell']['q'] for t in remaining),RTE_event=sum(t['cell']['R'] for t in remaining),
        control_probe=4*sum(t['replica']==0 for t in remaining),
        independent_event_oracle=sum(t['replica']==0 for t in remaining),signal=0,fit=0,quantum_shots=0)
    spec['continuation_call_budget']=budget
    spec['technical_digest']=b.metadata_binding(spec)
    c.save(destination/'technical_spec.json',spec)
    grant=dict(scope='H8_A_COST',SOURCE=source,origin_SOURCE=origin['origin_SOURCE'],
        technical_digest=spec['technical_digest'],approved=False,approved_by_user=False,runtime_authorization=False,
        one_shot=True,retry=False,next_stage_authorized=False,allowed_cpus=[],approval_receipt=None,
        approval_receipt_sha256=None,candidate_parallel_handoff_authorized=False,
        absolute_launch_command=None)
    c.save(destination/'grant_INVALID.json',grant)
    c.save(destination/'preparation_status.json',dict(status='TECHNICAL_SEAL_ONLY_REQUIRES_NEW_EXECUTION_GRANT'
        if finalize else 'DRAFT_ORIGIN_LIVE_NO_RESULTS_ACCEPTED',SOURCE=source,origin_SOURCE=origin['origin_SOURCE'],
        origin_binding_sha256=c.sha(destination/'origin_binding.json'),reused_accepted_tasks=accepted,
        remaining_task_budget=budget,full_task_count=len(tasks),blind_retry=False,
        failed_charge_refunded=False,valid_grants_created=0,real_scientific_calls=0,
        stopped_processes=0,absolute_launch_command=None))
    return spec


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--origin-binding',required=True,type=Path)
    parser.add_argument('--destination',required=True,type=Path)
    parser.add_argument('--source',required=True)
    parser.add_argument('--finalize',action='store_true')
    parser.add_argument('--handoff-receipt',type=Path)
    args=parser.parse_args()
    prepare(args.origin_binding,args.destination,args.source,finalize=args.finalize,handoff_receipt=args.handoff_receipt)
    print('PREPARATION_ONLY_NO_VALID_GRANT_NO_SCIENTIFIC_LAUNCH')


if __name__=='__main__':main()
