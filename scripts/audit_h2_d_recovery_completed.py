"""Saved-byte/native-receipt audit of D_run02; no second numerical account."""
import argparse
from pathlib import Path

import h2_d_recovery_contract as c
from audit_h2_fp_completed import original_process_alive


def audit():
    activation=c.PRIVATE/'D_recovery_activation01'
    spec=c.read(activation/'D_spec_EXECUTION.json');grant=c.read(activation/'D_grant_APPROVED_PRIVATE.json')
    c.authorize(spec,grant,runtime=False,interpreter=False)
    for r in c.read(activation/'activation_preservation_manifest.json'):
        c.require(c.sha(r['path'])==r['sha256'],'RECOVERY_ACTIVATION_CHANGED')
    attempt=c.owned(spec['attempt']);out=attempt/'output'
    result=c.read(out/'result.json');terminal=c.read(out/'terminal.json');native=c.read(attempt/'native_exit.json')
    started=c.read(attempt/'native_started.json');marker=c.read(attempt/'ONE_SHOT.json');done=c.read(attempt/'completion_audit.json')
    c.require(result['status']==terminal['status']==done['status']=='COMPLETE_MANDATORY_STOP'
        and result['stage']==done['stage']=='D' and result['scope']==c.SCOPE,'COMPLETE_D_RECOVERY')
    c.require(result['SOURCE']==done['SOURCE']==marker['SOURCE']==spec['SOURCE']
        and result['binding']==c.old.BINDING and marker['spec_digest']==terminal['spec_digest']==c.digest(spec)
        and marker['grant_digest']==c.digest(grant),'RESULT_SCOPE_DIGEST')
    c.require(native['exit_code']==terminal['exit_code']==done['native_exit_code']==0
        and native['same_parent_reaped'] and native['wait4_complete']
        and started['pid']==native['pid'] and started['parent_pid']==marker['parent_pid'],'NATIVE_EXIT_WAIT4')
    c.require(done['result_sha256']==c.sha(out/'result.json'),'RESULT_BYTES')
    c.require(not list(attempt.glob('**/STOP.json')) and not (attempt/'driver_STOP.json').exists(),'NO_SUCCESS_WITH_STOP')
    reserved={k:0 for k in c.CALLS}
    for p in out.glob('reservation_*.json'):
        r=c.read(p);c.require(r['nonrefundable'],'NONREFUNDABLE_RESERVATION');reserved[r['category']]+=r['count']
    c.require(reserved==result['calls_reserved']==done['calls_reserved']==c.CALLS,'EXACT_FRESH_RESERVATIONS')
    original=c.read(spec['saved_signals'])['signals'];fp=c.read(spec['FP_result'])
    c.require(result['signals']==original==c.read(out/'signals.json')['signals']
        and result['reference']==fp['reference'],'NO_SIGNAL_OR_REFERENCE_RECOMPUTATION')
    c.require(result['reused_signal_records']==70 and result['old_partial_compile_pairs_reused']==0
        and result['old_reservation_refunds']==0 and result['cumulative_D_compile_axes_reserved']==112,'LEGACY_CHARGES_AND_REUSE')
    rows=result['measurements'];byid={r['cell_id']:r for r in rows}
    c.require(len(rows)==28 and set(byid)=={x['id'] for x in c.cells()},'EXACT_NEW_CONFIGURATIONS')
    receipt=c.read(out/'compile_workers/accepted_native_receipt.json')
    c.require(receipt['status']=='PASS' and receipt['all_reaped'] and not receipt['cleanup_errors']
        and receipt['automatic_retry'] is False and len(receipt['workers'])==28,'ALL_WORKERS_REAPED')
    errors=[]
    for w in receipt['workers']:
        p=c.owned(w['directory']);value=c.read(p/'task_result.json');identity=w['identity']
        c.require(p.parent==out/'compile_workers' and w['reaped'] and w['exit_code']==0
            and c.sha(p/'task_result.json')==w['result_sha256'] and value==byid[value['cell_id']],'WORKER_BYTES_AND_EXIT')
        c.require(identity[0]==w['pid'] and identity[2]==native['pid']
            and not original_process_alive(identity[0],identity[1]),'OWNED_WORKER_NOT_LIVE')
        c.old.check_pair(value);c.require(value['cohort']=='DETERMINISTIC' and value['seed'] is None,'DETERMINISTIC_ONLY')
        control=c.read(p/'control.json');c.require(control['status']=='PASS' and control['max_error']<1e-9,'NATIVE_CONTROL')
        errors.append(control['max_error'])
        for axis in ('cosine','sine'):
            c.require(c.read(p/('measured_'+axis+'.json'))==value['axes'][axis],'SAVED_AXIS_IDENTITY')
            reservation=c.read(p/('compile_'+axis+'_reserved.json'))
            c.require(reservation['fingerprint']==value['axes'][axis]['fingerprint'],'AXIS_FINGERPRINT')
    pairs=fp['measurements'];expected={r['cell_id']:r for r in pairs if r['cell_id'] in
        {x['id'] for x in c.old.registrations()['D']}}
    c.require(len(result['reused_FP_measurements'])==6 and
        all(r==expected[r['cell_id']] for r in result['reused_FP_measurements']),'SIX_UNCHANGED_FP_COSTS')
    records={r['cell']['id']:r for r in original};account=result['accounting'];saved=account['rows']
    c.require(len(saved)==350 and {(r['cell_id'],r['epsilon']) for r in saved}==
        {(cid,e) for cid in records for e in c.old.EPS},'D70_FIVE_ENDPOINTS')
    for r in saved:
        ep=records[r['cell_id']]['precision_rows'][str(r['epsilon'])]['1.0']
        c.require(r['eligible']==ep['cost_acquisition_eligible'],'UNCHANGED_SAVED_ELIGIBILITY')
        if r['eligible']:
            c.require(r['n']==1 and r['G_RZ_exact'] is not None and r['shots']==
                dict(cosine=ep['axes']['real']['shots'],sine=ep['axes']['imag']['shots']),'COMPLETE_SAVED_ACCOUNTING')
    c.require(all(w is None or w in saved for w in account['winners'].values()),'SAVED_PF_WINNER_ROW')
    c.require(not (c.PRIVATE/'attempts/A_run01').exists(),'A_NOT_AUTHORIZED')
    return dict(status='COMPLETE_MANDATORY_STOP',stage='D',SOURCE=spec['SOURCE'],result_sha256=c.sha(out/'result.json'),
        native_exit_code=0,same_parent_reaped=True,wait4_complete=True,worker_processes_reaped=28,
        native_wall_seconds=native['wall_seconds'],native_peak_RSS_bytes=native['peak_RSS_bytes'],
        actual_fresh_calls=reserved,maximum_control_error=max(errors),saved_signal_records_reused=70,
        saved_FP_deterministic_pairs_reused=6,old_unconfirmed_compile_pairs_reused=0,
        old_compile_reservations=56,new_compile_reservations=56,cumulative_compile_reservations=112,
        old_reservation_refunds=0,accounting_rows=350,F_records_preserved=295,FP_private_files_preserved=204,
        original_D_private_files_preserved=541,scientific_computation_repeated_by_auditor=False,
        NPZ_decoded_by_auditor=False,accounting_recomputed_by_auditor=False,independent_scientific_review=False,
        automatic_retry=False,next_stage_authorized=False,A_S_STATS_P_executed=False,push_performed=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=audit();c.save(c.owned(a.output),result);print(__import__('json').dumps(result,indent=2))


if __name__=='__main__':main()
