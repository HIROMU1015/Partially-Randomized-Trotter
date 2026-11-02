#!/usr/bin/env python3
"""Read only the saved C attempt; structure, SHA, native wait and defect overlay.

No H4 engine, no signal arithmetic, no scientific retry or certificate/accounting.
"""
import argparse
from collections import Counter
from datetime import datetime,timezone
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
# HERE is artifacts/non_df_h4_c_evidence/2026-10-23, therefore root is parents[2].
META=HERE/'launch01';NS=HERE/'run01';TECH=ROOT/'artifacts/non_df_h4_c_production/2026-10-23'

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    with Path(path).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,sort_keys=True);f.write('\n')
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def check_saved():
    ctl=load('saved_C_authority_only',ROOT/'src/trotterlib/h4_certificate_control.py')
    prior=load('saved_C_preserve_only',TECH/'technical_audit.py')
    seal=read(NS/'seal.json');ctl.verify_seal(seal)
    original_contract=read(TECH/'C_execution_contract.json');active=read(NS/'contract.json')
    ctl.validate_contract(active,active=True);ctl.validate_routing(active,seal)
    assert active==read(META/'active_contract.json')
    assert original_contract['execution_authorized'] is original_contract['grant_valid'] is original_contract['dispatch_enabled'] is False
    assert sha(TECH/'SOURCE.json')==sha(NS/'seal.json')==active['technical_seal_sha256']
    grant=read(NS/'grant.json');assert grant['grant_id']==ctl.digest({k:v for k,v in grant.items() if k!='grant_id'})
    assert sha(NS/'contract.json')==grant['contract_sha256'] and sha(NS/'seal.json')==grant['seal_sha256']
    assert sha(META/'approval.json')==grant['approval_sha256']==active['approval_sha256']
    assert read(NS/'CONSUMED.json')['grant_id']==read(NS/'LAUNCH_CLAIM.json')['grant_id']==grant['grant_id']
    assert read(META/'dispatcher.json')['dispatch_count']==1
    before=read(META/'preservation_before.json');after=prior.preservation();assert before==after==read(TECH/'preservation_after.json')
    package=read(META/'technical_package_before.json')
    report_indexes={'PROJECT_MAP.md','docs/README.md','scripts/README.md','src/trotterlib/README.md','docs/research/研究概要・現状.md','artifacts/validation_manifest.json'}
    report_index_changes=[]
    for name,digest in package['files'].items():
        if sha(ROOT/name)==digest:continue
        assert name in report_indexes,('TECHNICAL_PACKAGE_CHANGED',name)
        oldbytes=subprocess.check_output(['git','show',package['base_commit']+':'+name],cwd=ROOT)
        assert hashlib.sha256(oldbytes).hexdigest()==digest
        if name=='artifacts/validation_manifest.json':
            original=json.loads(oldbytes);current=read(ROOT/name)
            assert all(current[k]==v for k,v in original.items())
            assert set(current)-set(original)=={'non_df_h4_c_candidate_run01_20261023'}
        else:
            current=(ROOT/name).read_bytes()
            assert current.startswith(oldbytes) or current.endswith(oldbytes),('NON_APPEND_REPORT_CHANGE',name)
        report_index_changes.append(name)
    assert sha(TECH/'result_inventory.json')==package['inventory_sha256']
    ledger=read(NS/'call_ledger.json');assert ledger['limits']==ctl.LIMITS
    for name,count in ledger['used'].items():assert count<=ledger['limits'][name]
    assert len(ledger['events'])==sum(ledger['used'].values())
    assert all(x['reserved_before_call'] for x in ledger['events'])
    stop=read(NS/'STOP.json');assert stop['status']=='FAILED_NO_RETRY'
    native={};expected={'pauli_jw_native':0,'df_native_native':-15,'pauli_jw_bridge_native':0,'df_native_bridge_native':0,'supervisor_native':1}
    for name,exit_code in expected.items():
        folder=(META if name=='supervisor_native' else NS)/name
        receipt=read(folder/'worker_native_exit.json');wstop=read(folder/'STOP.json');ident=read(folder/'worker_identity.json')
        assert receipt['exit_code']==exit_code and receipt['reaped'] is True and receipt['pid']==ident['pid']
        raw=read(folder/('native_exit_'+str(ident['pid'])+'.json'))
        assert raw['native_wait'] is True and raw['reaped'] is True and raw['exit_code']==exit_code
        assert wstop['all_owned_reaped'] is True and wstop['exit_code']==exit_code
        try:
            fields=Path('/proc',str(ident['pid']),'stat').read_text().rsplit(')',1)[1].split()
            assert int(fields[19])!=ident['start_ticks'],('OWNED_PROCESS_STILL_LIVE',name)
        except FileNotFoundError:pass
        native[name]=dict(native_exit=exit_code,reaped=True,witness_status=wstop['status'],identity=ident)
    result=read(NS/'pauli_jw_result.json');assert result==read(NS/'pauli_jw_tier_64.json')
    assert result['status']=='TARGET_NOT_MET' and result['bits']==64 and len(result['axes'])==20
    assert result['reference']==dict(real=dict(lower='0',upper='0'),imag=dict(lower='0',upper='0'))
    assert not (NS/'df_native_result.json').exists() and not list(NS.glob('df_native_tier_*.json'))
    counter=read(HERE/'diagnosis01/artificial_regression.json');assert counter['failures']==1 and counter['errors']==0 and counter['tests']==1
    diagnosis=read(HERE/'diagnosis01/source_diagnosis.json');assert diagnosis['patch_applied'] is False
    rows=[];mapping=read(active['method_map_path'])['rows']
    indexed={(x['case'],x['epsilon_sig'],x['axis']):x for x in result['axes']}
    for mapped in mapping:
        key=(mapped['case'],mapped['epsilon_sig'],mapped['axis']);saved=indexed.get(key)
        assert mapped['current_N'] is None and mapped['current_G'] is None and mapped['current_status']=='UNDETERMINED'
        rows.append(dict(case=key[0],epsilon_sig=key[1],axis=key[2],raw_result_saved=saved is not None,
            raw_axis_status=saved['status'] if saved else 'NO_FINAL_TIER_SAVED',
            evidence_status='REJECTED_REFERENCE_IMPLEMENTATION_DEFECT' if saved else 'INCOMPLETE_INTEGRITY_STOP',
            target_exact=mapped['design_total_error_target_exact'],formal_certificate_accepted=False,old_N=None,old_G=None,old_status='UNDETERMINED'))
    assert len(rows)==40 and len(set((x['case'],x['epsilon_sig'],x['axis']) for x in rows))==40
    record=dict(status='SAVED_ATTEMPT_AUDITED_NUMERICAL_BIAS_NOT_ACCEPTED_MANDATORY_STOP',UTC=datetime.now(timezone.utc).isoformat(),
        SOURCE=seal['SOURCE'],base_commit='c4497d8b49a3d796513df439dd025c72fb964a21',branch='codex/non-df-h4-c-evidence-run01-20261023',
        grant_id=grant['grant_id'],grant_consumed=True,dispatch_count=1,scientific_retry=0,
        execution_state='FAILED_NO_RETRY',scientific_acceptance='REJECTED_IMPLEMENTATION_DEFECT',
        raw_pauli_axes_saved=20,accepted_axes=0,DF_final_axes_saved=0,missing_axes_retained=20,
        native=native,call_reservations=ledger['used'],call_limits=ledger['limits'],
        completed_call_counts_not_inferred_from_reservations=True,forbidden_scientific_calls={k:0 for k in ctl.ZERO},
        source_code_unchanged=True,technical_package_unchanged=not report_index_changes,technical_evidence_unchanged=True,report_index_changes_append_only=report_index_changes,old_preservation_unchanged=True,
        old_SOURCE_files=738,old_result_claim_files=1147,old_investigation_files=17,BC_inventory_files=99,
        previous_technical_tests=dict(tests=110,failures=0,errors=0,skips=0,interpretation='Historical suite; inadequate reference-state coverage'),
        new_artificial_bug_regression=dict(tests=1,failures=1,errors=0,skips=0),
        old_N_G='20_ROWS_NULL_40_AXES_UNDETERMINED_UNCHANGED',formal_certificate_accepted=False,
        original_H4_native_completion_verified=False,original_supervisor_cause='UNKNOWN',
        patch_applied=False,push_performed=False,final_scientific_evidence_freeze=False,mandatory_STOP=True)
    return record,rows,after

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--record',action='store_true');args=parser.parse_args()
    result,rows,preservation=check_saved()
    if args.record:
        save(HERE/'completion_audit.json',result);save(HERE/'axis_outcomes.json',dict(rows=rows,scope='Saved-state inventory only; no numerical recomputation'))
        with (HERE/'axis_outcomes.csv').open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        save(META/'preservation_after.json',preservation)
        save(HERE/'STOP.json',dict(status='MANDATORY_STOP_CONFIRMED_IMPLEMENTATION_DEFECT',UTC=datetime.now(timezone.utc).isoformat(),
            scope='NEW_C_CANDIDATE_RUN_ONLY_NOT_ORIGINAL_H4',SOURCE=result['SOURCE'],grant_id=result['grant_id'],grant_consumed=True,
            dispatch_count=1,scientific_retry=0,accepted_axes=0,formal_certificate_accepted=False,new_N_G_accounting=0,
            all_owned_worker_native_exit_and_reap_recorded=True,source_unchanged=True,patch_applied=False,
            original_H4_native_completion_verified=False,original_supervisor_cause='UNKNOWN',next_requires_explicit_user_approval=True,push_performed=False))
    print(json.dumps({k:result[k] for k in ['status','execution_state','raw_pauli_axes_saved','accepted_axes','DF_final_axes_saved','scientific_retry','old_preservation_unchanged','grant_id']},sort_keys=True))
