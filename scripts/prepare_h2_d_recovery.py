"""Freeze H2 D recovery SOURCE/saved bytes/budget; no numerical computation."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import h2_d_recovery_contract as c
from prepare_h2_panels import row

MESSAGE='その方針で進めて。元もと行う予定だったところまで進めてよい'


def seal(source):
    c.STORE.mkdir(exist_ok=False);c.EVIDENCE.mkdir(exist_ok=False)
    private=c.PRIVATE;old_activation=private/'D_activation01';old_output=private/'attempts/D_run01/output'
    acceptance=dict(role='user',accepted_by_user=True,verbatim_user_message=MESSAGE,
        scope='Resume previously approved D to completion; reuse saved signals; new 28/56 costs; no A or later science',
        prior_clarification='これは私が間違って切断しただけ何度絵気にしなくてよい',
        signal_sha256=c.sha(old_output/'signals.json'),unconfirmed_compile_pairs_accepted=0,
        old_reservation_refunds=0,disconnect_remediation_requested=False)
    c.save(c.STORE/'saved_signal_acceptance.json',acceptance)
    paths=dict(base_spec=old_activation/'D_spec_EXECUTION.json',saved_signals=old_output/'signals.json',
        saved_preparations=old_output/'preparations.json',signal_receipt=old_output/'signal_workers/accepted_native_receipt.json',
        FP_result=private/'attempts/FP_run01/output/result.json',
        old_STOP=private/'D_interruption_audit01/observer_STOP.json',
        old_preservation_manifest=private/'D_interruption_audit01/original_D_preservation_manifest_PRIVATE.json',
        FP_preservation_manifest=private/'FP_activation01/FP_completed_preservation_manifest_PRIVATE.json',
        input_acceptance=c.STORE/'saved_signal_acceptance.json')
    base=c.read(paths['base_spec'])
    spec=dict(scope=c.SCOPE,stage='D',status='SEALED_PENDING_APPROVAL',SOURCE=source,plan=c.plan(),calls=c.CALLS,
        source_hashes=c.source_closure(),inputs=[row(p,k) for k,p in paths.items()],
        **{k:str(p) for k,p in paths.items()},native_config=base['native_config'],python=base['python'],
        attempt=str(private/'attempts/D_run02'),pending=['SCOPED_USER_APPROVAL_BINDING','FRESH_RESOURCE_ADMISSION'])
    c.verify_static(spec,runtime=True)
    c.save(c.STORE/'D_recovery_spec_PENDING.json',spec)
    c.save(c.EVIDENCE/'INVALID_GRANT.json',c.invalid_grant())
    c.save(c.EVIDENCE/'preregistration.json',dict(SOURCE=source,plan=c.plan(),calls=c.CALLS,
        input_rows=[{k:v for k,v in r.items() if k!='path'} for r in spec['inputs']],
        contract_sha256=c.sha(c.STORE/'D_recovery_spec_PENDING.json'),source_hashes=spec['source_hashes'],
        original_SOURCE=c.OLD_SOURCE,new_scientific_calls=0,disconnect_remediation=False,mandatory_STOP=True))
    return spec


def activate():
    pending=c.STORE/'D_recovery_spec_PENDING.json';spec=c.read(pending)
    observed=c.read(c.PRIVATE/'D_recovery_resource_observation01.json')
    import os
    cpus=[i for i in observed['candidate_worker_cpus'] if i in os.sched_getaffinity(0)]
    c.require(bool(cpus),'FREE_ALLOWED_CPUS_REQUIRED')
    activation=c.PRIVATE/'D_recovery_activation01';activation.mkdir(exist_ok=False)
    c.save(activation/'resources.json',dict(admitted=True,checked_at_UTC=datetime.now(timezone.utc).isoformat(),
        worker_cpus=cpus,observations=observed,agent_imposed_CPU_RAM_worker_wall_caps=None,
        max_concurrency='min(28 registered tasks, freshly admissible CPUs)',internal_threads=1,
        reason='outer process parallelism',shared_changes=False,effective_controller_bypass=False))
    c.save(activation/'approval_receipt.json',dict(role='user',stage='D',SOURCE=spec['SOURCE'],plan_digest=c.digest(c.plan()),
        verbatim_user_message=MESSAGE,scope='Previously planned D completion using accepted saved inputs and fixed recovery budget',
        call_budget=c.CALLS,automatic_retry=False,next_stage_authorized=False,push_authorized=False))
    spec=deepcopy(spec);spec.update(status='SEALED_FOR_EXECUTION',pending=[],pending_spec=str(pending),
        pending_spec_sha256=c.sha(pending),worker_cpus=cpus,resource_receipt=str(activation/'resources.json'),
        resource_receipt_sha256=c.sha(activation/'resources.json'))
    c.save(activation/'D_spec_EXECUTION.json',spec)
    grant=dict(scope=c.SCOPE,stage='D',SOURCE=spec['SOURCE'],spec_digest=c.digest(spec),approved_by_user=True,
        runtime_authorization=True,one_shot=True,automatic_retry=False,next_stage_authorized=False,
        approval_receipt=str(activation/'approval_receipt.json'),approval_receipt_sha256=c.sha(activation/'approval_receipt.json'))
    c.save(activation/'D_grant_APPROVED_PRIVATE.json',grant);c.authorize(spec,grant)
    c.save(activation/'preflight_identity_gate.json',dict(status='PASS',SOURCE=spec['SOURCE'],spec_digest=c.digest(spec),
        grant_digest=c.digest(grant),scientific_execution_started=False,new_attempt_authorized=True))
    c.save(activation/'activation_preservation_manifest.json',[row(p,str(p.relative_to(activation)))
        for p in sorted(activation.iterdir()) if p.is_file()])
    return dict(status='D_RECOVERY_PREFLIGHT_PASS',SOURCE=spec['SOURCE'],worker_cpus=len(cpus),new_calls=c.CALLS)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source');parser.add_argument('--activate',action='store_true')
    a=parser.parse_args()
    if a.activate:result=activate()
    else:
        c.require(a.source and len(a.source)==40,'COMMITTED_SOURCE_REQUIRED');spec=seal(a.source)
        result=dict(status='SEALED_PENDING_ACTIVATION',SOURCE=spec['SOURCE'],calls=c.CALLS)
    print(__import__('json').dumps(result,indent=2))


if __name__=='__main__':main()
