"""Inventory saved outputs and native receipts; never recompute a scientific value.

Read only this known run, its fixed manifests and explicit preserved file paths.
Do not call before the supervisor native witness has stored its final STOP.
"""
import csv
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
META=HERE/'launch01';NS=HERE/'run01'
TECH=ROOT.parent/'non-df-h4-c-reference-fix-20261023'

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,sort_keys=True);f.write('\n')
def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def main():
    assert (META/'supervisor_native/STOP.json').exists(),'STILL_RUNNING_NO_COMPLETION_AUDIT'
    ctl=load('candidate_saved_control',TECH/'src/trotterlib/h4_certificate_control.py')
    contract=read(NS/'contract.json');seal=read(NS/'seal.json');grant=read(NS/'grant.json')
    ctl.validate_contract(contract,active=True);ctl.verify_seal(seal);ctl.validate_routing(contract,seal)
    assert seal['SOURCE']==contract['C_production_implementation_SOURCE']==read(META/'preflight.json')['SOURCE']
    assert grant['kind']=='C_SCIENCE_ONE_SHOT' and grant['one_shot'] is True
    assert grant['namespace']==contract['output_namespace']==str(NS)
    assert grant['approval_sha256']==contract['approval_sha256']
    assert grant['grant_id']==ctl.digest({k:v for k,v in grant.items() if k!='grant_id'})
    assert sha(NS/'contract.json')==grant['contract_sha256']
    assert sha(NS/'seal.json')==grant['seal_sha256']==contract['technical_seal_sha256']
    assert read(NS/'CONSUMED.json')['grant_id']==grant['grant_id']
    assert (NS/'LAUNCH_CLAIM.json').exists()
    assert read(NS/'LAUNCH_CLAIM.json')['grant_id']==grant['grant_id']
    ledger=read(NS/'call_ledger.json')
    assert ledger['limits']==ctl.LIMITS
    assert all(0<=ledger['used'][k]<=ctl.LIMITS[k] for k in ctl.LIMITS)
    assert len(ledger['events'])==sum(ledger['used'].values())
    counts=Counter(event['call'] for event in ledger['events'])
    assert all(counts[k]==ledger['used'][k] for k in ctl.LIMITS)
    assert all(event['reserved_before_call'] is True for event in ledger['events'])
    preserved=read(META/'preservation_before.json');changed=[]
    for name,digest in preserved['files'].items():
        if sha(Path(name))!=digest:changed.append(name)
    # A changed unrelated index would be reported separately, never kill workers.
    save(META/'preservation_after.json',dict(status='PASS_BYTE_IDENTICAL' if not changed else 'CHANGES_REQUIRE_CLASSIFICATION',files_checked=len(preserved['files']),changed=changed))
    old_table_name='artifacts/non_df_h4_pilot/2026-10-23/postprocessing_recovery01/precision_table.csv'
    old_table=Path(contract['legacy_root'])/old_table_name
    fixed_table=subprocess.check_output(['git','show','6892e95ffdfc78e75425e7432e9e4ccdb9958df1:'+old_table_name],cwd=contract['legacy_root'])
    assert old_table.read_bytes()==fixed_table,'OLD_PRECISION_TABLE_CHANGED'
    with old_table.open(newline='') as f:old_rows=list(csv.DictReader(f))
    assert len(old_rows)==20
    assert all(all(row[k]=='' for k in ('N_cosine','N_sine','N_total','G_mean')) and row['accounting_usable']=='False' for row in old_rows)
    old_precision=dict(path=str(old_table),commit='6892e95ffdfc78e75425e7432e9e4ccdb9958df1',SHA256=sha(old_table),rows=20,
                       N_G_fields='Saved CSV fields empty, preserving the documented JSON null interpretation; never zero',accounting_usable=False)
    receipts={};live=[]
    folders=[META/'supervisor_native']+[NS/(rep+suffix) for rep in ('pauli_jw','df_native') for suffix in ('_native','_bridge_native')]
    for folder in folders:
        record={}
        for name in ('worker_identity.json','worker_native_exit.json','STOP.json'):
            if (folder/name).exists():record[name]=read(folder/name)
        ident=record.get('worker_identity.json')
        if ident and ctl._proc_identity(ident['pid'])==ident:live.append(ident)
        record['native_wait_receipts']={p.name:read(p) for p in sorted(folder.glob('native_exit_*.json'))}
        if 'worker_native_exit.json' in record:
            final=record['worker_native_exit.json']
            assert final['reaped'] is True
            raw=record['native_wait_receipts']['native_exit_'+str(final['pid'])+'.json']
            assert raw['native_wait'] is True and raw['reaped'] is True
            assert raw['exit_code']==final['exit_code']
        receipts[str(folder.relative_to(ROOT))]=record
    assert not live,'NATIVE_PROCESS_STILL_LIVE'
    assert read(NS/'LAUNCH_CLAIM.json')['pid']==read(META/'supervisor_native/worker_identity.json')['pid']
    for rep in ('pauli_jw','df_native'):
        for suffix,folder in (('_worker_claim.json','_native'),('_bridge_claim.json','_bridge_native')):
            if (NS/(rep+suffix)).exists() and (NS/(rep+folder)/'worker_identity.json').exists():
                assert read(NS/(rep+suffix))['pid']==read(NS/(rep+folder)/'worker_identity.json')['pid']
    normal_native=all(r.get('worker_native_exit.json',{}).get('exit_code')==0
                      and r.get('STOP.json',{}).get('all_owned_reaped') is True
                      and r.get('STOP.json',{}).get('status')=='NATIVE_EXIT_REAP_RECORDED'
                      and bool(r['native_wait_receipts']) for r in receipts.values())
    native=dict(status='PASS_NEW_RUN_NATIVE_EXIT_REAP' if normal_native else 'INCOMPLETE_OR_FAILED_NEW_RUN_NATIVE',receipts=receipts,
                matching_owned_processes_remaining=live,original_H4_native_verified=False,original_supervisor_cause='UNKNOWN')
    save(HERE/'native_audit.json',native)
    rows=[];results={};normalizations={}
    for rep in ('pauli_jw','df_native'):
        path=NS/(rep+'_result.json')
        if not path.exists():results[rep]=dict(status='NO_FINAL_SAVED_RESULT');continue
        saved=read(path)
        result=saved['last'] if saved['status']=='UNDETERMINED_PRECISION_EXHAUSTED' else saved
        # No intervals are recomputed, narrowed or converted to floating points.
        assert result['formal_certificate_accepted'] is False and result['original_N_G_changed'] is False
        assert result['bits'] in contract['precision_schedule_bits']
        expected={(rep+'_'+m,eps,axis) for m in ('PF','PR') for eps in contract['epsilon_sig'] for axis in contract['axes']}
        assert len(result['axes'])==20 and {(a['case'],a['epsilon_sig'],a['axis']) for a in result['axes']}==expected
        for axis in result['axes']:
            assert axis['old_N'] is None and axis['old_G'] is None and axis['old_status']=='UNDETERMINED'
            assert axis['formal_certificate_accepted'] is False
            rows.append(dict(case=axis['case'],epsilon_sig=axis['epsilon_sig'],axis=axis['axis'],
                precision_bits=axis['precision_bits'],status=axis['status'],target_exact=axis['target_exact'],
                bias_distance_lower=axis['bias_distance']['lower'],bias_distance_upper=axis['bias_distance']['upper'],
                total_error_upper=axis['total_error_upper']['upper'],formal_certificate_accepted=False,old_N=None,old_G=None,old_status=axis['old_status'],
                saved_result_path=str(path.relative_to(ROOT)),saved_result_SHA256=sha(path)))
        results[rep]=dict(status=saved['status'],last_tier_status=result['status'],bits=result['bits'],axes=len(result['axes']),reference=result['reference'],
                          state_norm=result['state_norm'],state_norm_defect=result['state_norm_defect'],
                          operator_snapshot_residual_upper=result['operator_snapshot_residual_upper'],
                          result_path=str(path.relative_to(ROOT)),result_SHA256=sha(path))
        normalizations[rep]=result['normalization']
    rows.sort(key=lambda a:(contract['cases'].index(a['case']),contract['epsilon_sig'].index(a['epsilon_sig']),contract['axes'].index(a['axis'])))
    save(HERE/'axis_outcomes.json',dict(scope='Verbatim saved numerical outputs; no shot/G or new scientific reanalysis',rows=rows))
    if rows:
        with (HERE/'axis_outcomes.csv').open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save(HERE/'normalization_saved.json',dict(scope='Verbatim copy from each final result; formal acceptance remains false',representations=normalizations))
    met=sum(a['status']=='TARGET_MET' for a in rows)
    stop=read(NS/'STOP.json')
    result=dict(status='PASS_SAVED_CANDIDATE_STRUCTURE_AND_NEW_NATIVE' if normal_native and not changed and len(rows)==40 else 'PARTIAL_OR_UNRESOLVED_PRESERVE_NO_RETRY',
        SOURCE=seal['SOURCE'],grant_id=grant['grant_id'],scientific_dispatch_count=1,
        run_STOP=stop,results=results,saved_axes=len(rows),numerical_design_target_met_axes=met,
        formal_scientific_certificate_accepted_axes=0,formal_certificate_accepted=False,
        new_run_native_complete=normal_native,call_reservations=ledger['used'],call_limits=ledger['limits'],
        reservation_counts_are_not_completed_call_counts=True,old_files_checked=len(preserved['files']),old_files_changed=changed,
        original_precision_table=old_precision,
        old_N_G='20 rows NULL and40 axes UNDETERMINED unchanged',original_H4_native_verified=False,original_supervisor_cause='UNKNOWN',
        chemistry_sampling_compile_shot_G_calls=0,scientific_retry=0,old_grant_reused=False,push=False,
        literal_compiled_wrapper_transfer_certified=False,DF_exact_event_range_certified=False,
        PCG64_independence_proved=False,final_scientific_evidence_freeze=False,mandatory_STOP=True)
    save(HERE/'completion_audit.json',result)
    print(json.dumps({k:result[k] for k in ('status','SOURCE','saved_axes','numerical_design_target_met_axes','formal_scientific_certificate_accepted_axes','new_run_native_complete','call_reservations','old_files_checked','old_files_changed')},sort_keys=True))

if __name__=='__main__':main()
