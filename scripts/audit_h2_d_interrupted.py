"""Preserve D_run01 after lost native completion; no science or retry.

Absence of a process and a saved task result do not prove exit0 or wait4.
This observer never fabricates the missing native/worker completion receipts.
"""
from datetime import datetime, timezone
from pathlib import Path
import os

import h2_panels_contract as c
from prepare_h2_panels import preservation, row


def process_observation(pid, expected_start=None):
    try:
        fields=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
    except FileNotFoundError:
        return dict(pid=pid,present=False,original_live=False)
    ticks=int(fields[19]); same=expected_start is None or expected_start==ticks
    return dict(pid=pid,present=True,state=fields[0],parent_pid=int(fields[1]),start_ticks=ticks,
        original_identity_matches=same,original_live=same and fields[0]!='Z')


def inspect():
    activation=c.PRIVATE/'D_activation01'
    spec=c.read(activation/'D_spec_EXECUTION.json');grant=c.read(activation/'D_grant_APPROVED_PRIVATE.json')
    c.require(spec['stage']=='D','D_ONLY')
    c.authorize(spec,grant,runtime=True,interpreter=False)
    for r in c.read(activation/'activation_preservation_manifest.json'):
        c.require(c.sha(r['path'])==r['sha256'],'ORIGINAL_D_ACTIVATION_CHANGED')
    for r in c.read(c.PRIVATE/'FP_activation01/FP_completed_preservation_manifest_PRIVATE.json'):
        c.require(c.sha(r['path'])==r['sha256'],'FP_BYTES_CHANGED')
    for r in c.read(c.STORE/'preparation_preservation.json'):
        c.require(c.sha(r['path'])==r['sha256'],'PREPARATION_CHANGED')
    f=preservation()
    attempt=c.owned(spec['attempt']);output=attempt/'output'
    marker=c.read(attempt/'ONE_SHOT.json');start=c.read(attempt/'native_started.json')
    c.require(marker['SOURCE']==spec['SOURCE'] and marker['spec_digest']==c.digest(spec)
        and marker['grant_digest']==c.digest(grant) and marker['parent_pid']==start['parent_pid'],'ORIGINAL_ATTEMPT_IDENTITY')
    parent=process_observation(start['parent_pid']);native=process_observation(start['pid'])
    c.require(not parent['original_live'] and not native['original_live'],'D_PROCESS_STILL_RUNNING_DO_NOT_RESTART')
    missing=['native_exit.json','completion_audit.json','output/terminal.json','output/result.json']
    c.require(all(not (attempt/p).exists() for p in missing),'DIFFERENT_COMPLETION_STATE_REQUIRES_NEW_OBSERVATION')
    reserved={k:0 for k in spec['calls']}
    for p in sorted(output.glob('reservation_*.json')):
        r=c.read(p);c.require(r['nonrefundable'],'NONREFUNDABLE_CHARGE')
        reserved[r['category']]+=r['count']
    c.require(all(reserved[k]<=v for k,v in spec['calls'].items()),'RESERVATION_BUDGET')
    c.require(reserved['endpoint_accounting_pass']==0,'UNEXPECTED_ACCOUNTING_RESERVATION')
    signals=c.read(output/'signals.json');cells=c.registrations()['D']
    c.require([r['cell'] for r in signals['signals']]==cells,'SAVED_REGISTERED_70_SIGNALS')
    byid={r['cell']['id']:r for r in signals['signals']}
    fp=c.read(spec['dependencies']['FP']['result']);old={r['cell']['id']:r for r in fp['signals']}
    reused=signals['reused_signal_ids']
    c.require(len(reused)==6 and all(byid[cid]==old[cid] for cid in reused),'UNCHANGED_SIX_FP_SIGNALS')
    signal_receipt=c.read(output/'signal_workers/accepted_native_receipt.json')
    c.require(signal_receipt['status']=='PASS' and signal_receipt['all_reaped']
        and not signal_receipt['cleanup_errors'] and len(signal_receipt['workers'])==36,'SIGNAL_WORKERS_RECEIPT')
    for worker in signal_receipt['workers']:
        p=c.owned(worker['directory']);r=c.read(p/'task_result.json')
        c.require(worker['exit_code']==0 and worker['reaped'] and c.sha(p/'task_result.json')==worker['result_sha256']
            and r==byid[r['cell']['id']],'SAVED_SIGNAL_WORKER_HASH_AND_EXIT')
    compile_dir=output/'first_control_workers'
    c.require(not (compile_dir/'accepted_native_receipt.json').exists(),'COMPILE_RECEIPT_PRESENT_REQUIRES_COMPLETION_AUDIT')
    observations=[];partials=[]
    for p in sorted(compile_dir.glob('*/started.json')):
        started=c.read(p);identity=started['identity']
        c.require(identity[0]==started['pid'] and identity[2]==start['pid'] and identity[3]==os.getuid(),'OWNED_COMPILE_IDENTITY')
        live=process_observation(identity[0],identity[1]);c.require(not live['original_live'],'COMPILE_WORKER_STILL_LIVE')
        observations.append(dict(task_id=started['task'],identity=identity,observation=live,
            exit_code=None,same_parent_reaped=None,completion_accepted=False))
        result_path=p.parent/'task_result.json'
        if result_path.exists():
            r=c.read(result_path);c.check_pair(r)
            for axis in ('cosine','sine'):
                c.require(c.read(p.parent/('measured_'+axis+'.json'))==r['axes'][axis],'PARTIAL_AXIS_COPY')
            partials.append(dict(task_id=started['task'],path=str(result_path),sha256=c.sha(result_path),
                native_completion_unconfirmed=True,eligible_for_cost_reuse=False))
    c.require(len(observations)==reserved['native_evolution']==28,'FIXED_STARTED_COMPILE_TASKS')
    c.require(reserved['wrapper_compile']==56 and reserved['control_probe']==112,'NONREFUNDED_COMPILE_RESERVATIONS')
    primitive=c.read(output/'primitive_validation.json')
    c.require(primitive['actual_time_count']*primitive['probe_count']==reserved['primitive_probe']==411,'SAVED_PRIMITIVE_SCOPE')
    c.require(not (c.PRIVATE/'attempts/A_run01').exists(),'A_NOT_AUTHORIZED')
    return dict(status='STOP_NATIVE_COMPLETION_UNCONFIRMED',stage='D',SOURCE=spec['SOURCE'],
        observed_at_UTC=datetime.now(timezone.utc).isoformat(),attempt=str(attempt),attempt_consumed=True,
        original_D_one_shot_approved=True,new_attempt_authorized=False,automatic_retry=False,
        parent_observation=parent,native_observation=native,missing_completion_receipts=missing,
        native_exit_code=None,same_parent_wait4=None,scientific_completion_PASS=False,
        cause='UNKNOWN_NO_NATIVE_EXIT_RECEIPT',chat_interruption_preceded_observation=True,
        signal_records_saved=len(byid),new_base_signal_records=36,new_B0I_overlays=28,reused_FP_signals=6,
        signal_worker_results_hash_and_native_reap_verified=36,
        compile_workers_started=len(observations),compile_worker_observations=observations,
        raw_partial_compile_task_results=len(partials),partial_compile_records=partials,
        compile_pool_receipt_missing=True,compile_worker_exit_codes_unknown=True,
        compile_control_files=sum(1 for p in compile_dir.glob('*/control.json')),
        per_axis_compile_reservation_files=sum(1 for p in compile_dir.glob('*/compile_*_reserved.json')),
        raw_measured_axis_files=sum(1 for p in compile_dir.glob('*/measured_*.json')),
        reserved_calls=reserved,call_maxima=spec['calls'],reservation_refunds=0,
        final_accounting_pass=0,final_PF_winners_obtained=False,
        primitive=primitive,F_records_preserved=f['records_checked'],FP_private_files_preserved=204,
        scientific_computation_repeated_by_observer=False,NPZ_decoded_by_observer=False,
        independent_scientific_review=False,A_S_STATS_P_executed=False,next_stage_authorized=False,push_performed=False)


def main():
    audit=inspect();folder=c.PRIVATE/'D_interruption_audit01';folder.mkdir(exist_ok=False)
    c.save(folder/'observer_STOP.json',audit)
    original=[]
    for root in [c.PRIVATE/'attempts/D_run01',c.PRIVATE/'D_activation01']:
        for p in sorted(root.rglob('*')):
            if p.is_file() and 'scratch' not in p.relative_to(root).parts:
                original.append(row(p,str(p.relative_to(c.PRIVATE))))
    c.save(folder/'original_D_preservation_manifest_PRIVATE.json',original)
    c.save(folder/'preservation_receipt.json',dict(status='PASS',original_D_files=len(original),
        manifest_sha256=c.sha(folder/'original_D_preservation_manifest_PRIVATE.json'),
        old_SOURCE_grant_ledger_marker_logs_modified=False,science_restarted=False,
        native_exit_receipt_fabricated=False,mandatory_STOP=True))
    print(__import__('json').dumps({k:audit[k] for k in ('status','SOURCE','signal_records_saved',
        'compile_workers_started','raw_partial_compile_task_results','raw_measured_axis_files','final_accounting_pass')}))


if __name__=='__main__':main()
