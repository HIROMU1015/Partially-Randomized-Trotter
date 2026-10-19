"""Audit saved H2 FP_run01 bytes and native receipts; never import science.

Does not decode NPZ, evaluate a signal, sample, compile, or compute cost statistics.
This is the author's completion audit, not a separate scientific review.
"""
import argparse
from pathlib import Path
import os

import h2_panels_contract as c
from prepare_h2_panels import preservation, row


def original_process_alive(pid, start_ticks):
    try:
        fields=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
        return int(fields[19])==start_ticks and fields[0]!='Z'
    except FileNotFoundError:
        return False


def audit():
    activation=c.PRIVATE/'FP_activation01'
    spec=c.read(activation/'FP_spec_EXECUTION.json')
    grant=c.read(activation/'FP_grant_APPROVED_PRIVATE.json')
    c.require(spec['stage']=='FP','FP_ONLY_AUDIT')
    c.authorize(spec,grant,runtime=False,interpreter=False)
    for record in c.read(activation/'activation_preservation_manifest.json'):
        c.require(c.sha(record['path'])==record['sha256'],'ACTIVATION_PRESERVED:'+record['role'])
    for record in c.read(c.STORE/'preparation_preservation.json'):
        c.require(c.sha(record['path'])==record['sha256'],'PREPARATION_PRESERVED:'+record['role'])
    before=preservation()
    attempt=c.owned(spec['attempt']); output=attempt/'output'
    result=c.read(output/'result.json'); terminal=c.read(output/'terminal.json')
    native=c.read(attempt/'native_exit.json'); started=c.read(attempt/'native_started.json')
    marker=c.read(attempt/'ONE_SHOT.json'); completion=c.read(attempt/'completion_audit.json')
    c.require(marker['SOURCE']==result['SOURCE']==completion['SOURCE']==spec['SOURCE'], 'FP_SOURCE')
    c.require(marker['spec_digest']==terminal['spec_digest']==c.digest(spec) and
              marker['grant_digest']==c.digest(grant),'FP_SPEC_GRANT_DIGEST')
    c.require(result['scope']==c.SCOPE and result['stage']==terminal['stage']==completion['stage']=='FP'
        and result['binding']==c.BINDING,'FP_BINDING')
    c.require(result['status']==terminal['status']==completion['status']=='COMPLETE_MANDATORY_STOP'
        and native['exit_code']==terminal['exit_code']==completion['native_exit_code']==0
        and native['same_parent_reaped'] and native['wait4_complete']
        and completion['same_parent_reaped'] and completion['wait4_complete'],'FP_NATIVE_COMPLETION')
    c.require(started['pid']==native['pid'] and started['parent_pid']==marker['parent_pid'],'NATIVE_SAME_PARENT')
    c.require(completion['result_sha256']==c.sha(output/'result.json'),'COMPLETED_RESULT_BYTES')
    c.require(not any(p.exists() for p in (attempt/'driver_STOP.json',output/'STOP.json')),'NO_SUCCESS_WITH_STOP')
    c.require(result['accounting'] is None and not result['FP_PR_cost_pooled_with_A']
        and not result['automatic_retry'] and not result['next_stage_authorized'],'FP_DIAGNOSTIC_STOP_SCOPE')
    cells=c.registrations()['FP']; byid={x['id']:x for x in cells}
    signals=result['signals']; c.selected(signals,cells)
    c.require(signals==c.read(output/'signals.json')['signals'],'SAVED_SIGNAL_COPY_IDENTITY')
    c.require([r['cell'] for r in signals]==cells,'EXACT_REGISTERED_SIGNAL_ORDER')
    records={r['cell']['id']:r for r in signals}
    measurements=result['measurements']; measured={r['cell_id']:r for r in measurements}
    c.require(len(measurements)==8 and set(measured)==set(byid),'EXACT_EIGHT_NATIVE_CONFIGURATIONS')
    reserved={k:0 for k in spec['calls']}
    for p in sorted(output.glob('reservation_*.json')):
        r=c.read(p); c.require(r['nonrefundable'] is True,'NONREFUNDABLE_RESERVATION')
        reserved[r['category']]+=r['count']
    c.require(reserved==result['calls_reserved']==completion['calls_reserved']
        and result['calls_maximum']==spec['calls'] and all(reserved[k]<=v for k,v in spec['calls'].items()),'CALL_LEDGER')
    primitive=c.read(output/'primitive_validation.json')
    c.require(primitive['actual_time_count']*primitive['probe_count']==reserved['primitive_probe']==111
        and primitive['all_registered_times'] and primitive['includes_negative_yoshida_times']
        and primitive['max_native_expm_error']<1e-9 and primitive['max_spectral_expm_error']<1e-9,
        'PRIMITIVE_PROBES')
    workers=[]; controls=[]; provenance=[]
    raw={x['id'] for x in cells if x['method']!='B0I'}
    for group,expected in [('signal_workers',raw),('first_control_workers',
            {x['id']+'_0' if x['R'] else x['id'] for x in cells})]:
        directory=output/group; receipt=c.read(directory/'accepted_native_receipt.json')
        c.require(receipt['status']=='PASS' and receipt['all_reaped'] and not receipt['cleanup_errors']
            and receipt['automatic_retry'] is False and receipt['first_error'] is None,'WORKER_COMPLETION')
        c.require(len(receipt['workers'])==len(expected) and {r['task_id'] for r in receipt['workers']}==expected,
            'EXACT_WORKER_TASKS')
        for worker in receipt['workers']:
            path=c.owned(worker['directory']); value=c.read(path/'task_result.json')
            c.require(path.parent==directory and worker['exit_code']==0 and worker['reaped']
                and c.sha(path/'task_result.json')==worker['result_sha256'],'WORKER_BYTES_AND_EXIT')
            identity=worker['identity']
            c.require(identity[0]==worker['pid'] and identity[2]==native['pid'] and identity[3]==os.getuid(),
                'OWNED_WORKER_PARENT_IDENTITY')
            c.require(not original_process_alive(identity[0],identity[1]),'ORIGINAL_WORKER_STILL_LIVE')
            workers.append(worker)
            if group=='signal_workers':
                c.require(value==records[value['cell']['id']],'SAVED_FORWARD_RECORD')
                continue
            cid=value['cell_id']; cell=byid[cid]; c.check_pair(value)
            c.require(value==measured[cid] and value['replica']==0 and value['cohort']=='FP_NATIVE_DIAGNOSTIC_N1',
                'MEASUREMENT_COPY_AND_COHORT')
            prep_key=cid.replace('B0I_','B0_',1)
            c.require(value['preparation_hash']==result['preparation_identities'][prep_key],'COMPILE_PREPARATION')
            control=c.read(path/'control.json')
            c.require(control['status']=='PASS' and control['max_error']<1e-9,'NATIVE_CONTROL')
            controls.append(control)
            for axis in ('cosine','sine'):
                c.require(c.read(path/('measured_'+axis+'.json'))==value['axes'][axis],'AXIS_COPY')
                r=c.read(path/('compile_'+axis+'_reserved.json'))
                c.require(r['fingerprint']==value['axes'][axis]['fingerprint']
                    and r['event_digest']==value['event_digest'],'COMPILE_FINGERPRINT_EVENT')
            if cell['R']:
                trajectory=c.read(path/'trajectory.json')
                c.require(not trajectory['conditional'] and not trajectory['quantum_weights_changed']
                    and trajectory['cohort']=='FP_NATIVE_DIAGNOSTIC_N1'
                    and trajectory['event_digest']==c.digest(trajectory['events'])==value['event_digest'],
                    'UNCONDITIONAL_PAIRED_TRAJECTORY')
                registry=c.read(spec['seed_registry'])['tasks']
                task=next(t for t in registry if t['cell']['id']==cid)
                c.require(task['seed']==trajectory['seed']==value['seed'] and
                    c.stats.event_stratum(trajectory['events'])==value['stratum'],'SEALED_SEED_STRATUM')
                provenance.append(dict(cell_id=cid,n=1,stratum=value['stratum'],
                    event_digest=value['event_digest'],trajectory_sha256=c.sha(path/'trajectory.json')))
    c.require(len(workers)==14 and len(controls)==8 and reserved['control_probe']==4*len(controls),
        'WORKER_AND_CONTROL_COVERAGE')
    c.require(reserved['wrapper_compile']==2*len(measurements) and reserved['trajectory']==len(provenance)==2,
        'COMPILE_AND_TRAJECTORY_CALLS')
    c.require(reserved['base_signal']==len(raw) and reserved['B0I_overlay']==2 and
        reserved['new_signed_shot_endpoints']==len(signals)*len(c.EPS),'SIGNAL_CALLS')
    c.require(not (c.PRIVATE/'attempts/D_run01').exists() and not (c.PRIVATE/'attempts/A_run01').exists(),
        'NO_AUTOMATIC_NEXT_STAGE')
    return dict(status='COMPLETE_MANDATORY_STOP',stage='FP',SOURCE=spec['SOURCE'],
        result_sha256=c.sha(output/'result.json'),calls_reserved=reserved,scientific_computation_repeated_by_auditor=False,
        NPZ_decoded_by_auditor=False,native_exit_code=0,same_parent_reaped=True,wait4_complete=True,
        worker_processes_reaped=len(workers),all_worker_result_hashes_match=True,
        native_wall_seconds=native['wall_seconds'],native_peak_RSS_bytes=native['peak_RSS_bytes'],
        primitive=primitive,control_configurations=len(controls),
        maximum_control_error=max(r['max_error'] for r in controls),PR_diagnostic_cohorts=provenance,
        F_records_preserved=before['records_checked'],preparation_preserved=True,activation_preserved=True,
        independent_scientific_review=False,D_A_S_STATS_P_executed=False,automatic_retry=False,
        next_stage_authorized=False,push_performed=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); path=c.owned(args.output)
    result=audit(); c.save(path,result)
    print(__import__('json').dumps(result,indent=2))


if __name__=='__main__': main()
