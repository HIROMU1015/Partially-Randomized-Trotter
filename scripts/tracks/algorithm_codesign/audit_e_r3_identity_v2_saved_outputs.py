"""Saved JSON/bytes-only audit. No numerical, circuit, or runner imports."""
import csv
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess

ROOT=Path(__file__).resolve().parents[3]
OUT=Path('artifacts/track_b_e_r3_identity_diagnostic_result/2026-10-12/v2')
ART=Path('artifacts/track_b_e_r3_identity_diagnostic_preparation/2026-10-12/v2')
AUDIT=Path('artifacts/track_b_e_r3_identity_diagnostic_execution_audit/2026-10-12/v2')
S='b9519c29b5f1528ad5c8a6e398b2fa12062db8c6'
A='5e8baf475dd086e85e4b2611cc20b1f77161e7d5'

def sha(b):return hashlib.sha256(b).hexdigest()
def load(p):return json.loads((ROOT/p).read_bytes())
def identity(p):
    b=(ROOT/p).read_bytes();return {'path':str(p),'bytes':len(b),'sha256':sha(b)}
def write(name,value):
    with (ROOT/AUDIT/name).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
def differences(a,b,path=''):
    if type(a)!=type(b):return [{'path':path,'current':a,'legacy':b}]
    if isinstance(a,dict):
        out=[]
        for k in sorted(a.keys()|b.keys()):
            if k not in a or k not in b:out.append({'path':path+'/'+k,'current':a.get(k),'legacy':b.get(k)})
            else:out.extend(differences(a[k],b[k],path+'/'+k))
        return out
    if isinstance(a,list):
        if len(a)!=len(b):return [{'path':path+'/length','current':len(a),'legacy':len(b)}]
        return [d for i,(x,y) in enumerate(zip(a,b)) for d in differences(x,y,path+'/'+str(i))]
    return [] if a==b else [{'path':path,'current':a,'legacy':b}]

def main():
    checks=[]
    def check(label,ok):
        checks.append({'check':label,'passed':bool(ok)})
        if not ok:raise AssertionError(label)
    original=load(AUDIT/'original_run_inventory_v2.json')['files']
    for r in original:check('original runner bytes '+r['path'],identity(r['path'])==r)
    marker=load(OUT/'one_shot_marker.json');markerhash=identity(OUT/'one_shot_marker.json')['sha256']
    auth=load(ART/'authorization.json');contract=load(ART/'contract_v1.json');manifest=load(ART/'source_manifest_v1.json')
    git=lambda *args:subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
    check('authorization direct child',git('rev-list','--parents','-n','1',A).split()==[A,S])
    check('authorization only diff',set(git('diff','--name-only',S,A).splitlines())=={
        str(ART/'authorization.json'),'docs/tracks/algorithm_codesign/e_r3_identity_diagnostic_execution_authorization_v2.md'})
    check('exact approval scope',auth['source_commit']==S and auth['status']=='APPROVED_FOR_ONE_E_R3_IDENTITY_ONLY_DIAGNOSTIC'
          and auth['diagnostic_execution_authorized'] and not auth['science_execution_authorized'] and auth['runs']==1 and auth['retries']==0 and auth['mandatory_STOP'])
    expected={'source':S,'authorization':A,'authorization_sha256':identity(ART/'authorization.json')['sha256'],
              'contract_sha256':identity(ART/'contract_v1.json')['sha256'],
              'manifest_sha256':identity(ART/'source_manifest_v1.json')['sha256'],
              'branch':'track-b-e-r3-identity-only-v2-one-shot-execution-20261012','remote_SHA':A}
    check('marker provenance',all(marker[k]==v for k,v in expected.items()) and marker['runs']==1 and marker['retries']==0 and marker['mandatory_STOP'])
    check('auth binding',auth['contract_sha256']==expected['contract_sha256'] and auth['manifest_sha256']==expected['manifest_sha256'])
    started=load(OUT/'identity.started.json');capability=dict(expected,marker_sha256=markerhash)
    check('exclusive child provenance',all(started[k]==v for k,v in capability.items()) and started['runs']==1 and started['retries']==0)
    result=load(OUT/'identity_result.json');terminal=load(OUT/'terminal_status.json');process=load(OUT/'process_receipt.json');outer=load(OUT/'outer_process_receipt_v2.json')
    for name,saved in [('result',result),('terminal',terminal)]:
        check(name+' complete STOP',saved['status']=='E_R3_IDENTITY_DIAGNOSTIC_COMPLETE' and saved['retries']==0 and saved['mandatory_STOP'])
        check(name+' provenance',all(saved['provenance'][k]==v for k,v in capability.items()))
    check('outer exit complete',outer['exit_status']==0 and outer['runner_invocations']==1 and outer['retries']==0 and outer['mandatory_STOP'])
    check('outer source HEAD',outer['source']==S and outer['execution_HEAD']==A)
    check('inner receipt consistency',terminal['resource_receipt']==process and len(process['child_receipts'])==1
          and process['child_receipts'][0]['exit_status']==0 and process['last_phase']['failure'] is None)
    check('source CPU RSS within caps',process['CPU_seconds_parent_and_reaped_children']<=contract['caps']['CPU_seconds_total']
          and process['peak_sampled_aggregate_RSS_MiB']<=contract['caps']['aggregate_RSS_MiB']
          and result['_process_receipt']['peak_RSS_MiB']<=contract['caps']['worker_RSS_MiB'])
    for r in manifest['files']:check('protected source '+r['path'],identity(r['path'])==r)
    check('old contract bytes unchanged',(ROOT/ART/'contract_v1.json').read_bytes()==(ROOT/'artifacts/track_b_e_r3_identity_diagnostic_preparation/2026-10-12/contract_v1.json').read_bytes())
    old=load('artifacts/track_b_df_correction_comparison_execution_audit/2026-10-12/saved_output_audit_v1.json')['result_file_inventory']
    for r in old:check('old result unchanged '+r['path'],identity(r['path'])==r)
    paths={p:load(OUT/(p+'.json')) for p in ['current','legacy']}
    stages=['frame_0','frame_1','frame_2','Q_0','Q_1','Q_2','T','U','input_target','actual_gate_parameter'];table=[]
    job=load(contract['inputs']['saved_job']['path'])
    for label,record in paths.items():
        check(label+' provenance',all(record['provenance'][k]==v for k,v in capability.items()))
        check(label+' domain/order',not record['controlled'] and record['primitive_signs']==[1,-1,1] and record['provenance']['path_order']==['current','legacy'])
        check(label+' observation',record['phase']=='IDENTITY_OBSERVATION_SAVED' and record['observation']=='MISMATCH'
              and record['actual_fingerprint'] is not None and record['fingerprint_exception'] is None and record['observer_exception'] is None)
        check(label+' observer lifecycle',record['observer_lifecycle']['installation_verified']
              and record['observer_lifecycle']['restoration_verified'] and not record['observer_lifecycle']['exceptions'])
        check(label+' definition counts',record['definition_work']['recursive_payload_attempts']==1
              and record['definition_work']['UnitaryGate_define_entries']==1
              and record['definition_work']['qs_decomposition_python_entries']==5
              and record['definition_work']['native_transpile_blocked_entries']==0)
        check(label+' hash exact saved payload',record['actual_fingerprint']==sha(json.dumps(record['recursive_payload'],sort_keys=True,separators=(',',':')).encode()))
        check(label+' expected unchanged',record['historical_expected_fingerprint']==contract['historical_expected_fingerprint'])
        for stage in stages:
            arr=record['input_arrays'][stage];b=bytes.fromhex(arr['raw_bytes_hex'])
            check(label+' bytes '+stage,len(b)==arr['bytes']==math.prod(arr['shape'])*16 and sha(b)==arr['sha256'] and arr['dtype_str']=='<c16')
        for i,v in enumerate(job['frames']):
            expected_bytes=b''.join(struct.pack('<dd',float.fromhex(z[0]),float.fromhex(z[1])) for row in v for z in row)
            check(label+' saved frame byte representation '+str(i),expected_bytes.hex()==record['input_arrays']['frame_'+str(i)]['raw_bytes_hex'])
        for stage in ['input_target','actual_gate_parameter']:
            check(label+' input-target-binding '+stage,record['input_arrays'][stage]['raw_bytes_hex']==record['input_arrays']['U']['raw_bytes_hex'])
    equal={}
    for stage in stages:
        a=paths['current']['input_arrays'][stage];b=paths['legacy']['input_arrays'][stage]
        equal[stage]=all(a[k]==b[k] for k in ['dtype_str','shape','raw_bytes_hex'])
        table.append({'stage':stage,'dtype_str':a['dtype_str'],'shape':str(a['shape']),'bytes':a['bytes'],
                      'current_sha256':a['sha256'],'legacy_sha256':b['sha256'],'bytes_equal':equal[stage]})
    check('saved matrix comparison',equal==result['comparison']['matrix_stage_bytes_equal'] and all(equal.values()) and result['comparison']['first_different_stage'] is None)
    rawdiff=differences(paths['current']['recursive_payload'],paths['legacy']['recursive_payload'])
    saved=[{k:v for k,v in r.items() if k!='diagnostic_category'} for r in result['comparison']['payload_differences']]
    check('all payload differences preserved',rawdiff==saved and len(rawdiff)==4)
    check('auto-name-only saved categories',result['comparison']['automatic_names_only'] and all(r['diagnostic_category']=='AUTOGENERATED_OPERATION_NAME' for r in result['comparison']['payload_differences']))
    check('STOP no scientific winner',result['science_winner'] is None and not result['historical_cause_unique'] and not result['comparison']['acceptance_guard_changed'] and not result['comparison']['historical_execution_reproduced'])
    files=[identity(str(p.relative_to(ROOT))) for p in sorted((ROOT/OUT).rglob('*')) if p.is_file()]
    check('source run output cap',sum(r['bytes'] for r in files)<=contract['caps']['output_bytes'])
    check('log cap',sum(r['bytes'] for r in files if r['path'].endswith('.log'))<=contract['caps']['log_bytes'])
    check('all raw files GitHub review-sized',all(r['bytes']<=contract['caps']['review_file_bytes'] for r in files))
    summary={'kind':'E_R3_V2_SAVED_OUTPUT_TECHNICAL_AUDIT','checks':checks,'all_checks_passed':True,
             'status':result['status'],'source':S,'authorization':A,'marker_sha256':markerhash,
             'registered_runner_invocations':1,'retries':0,'paths_completed':2,'path_order':['current','legacy'],
             'matrix_bytes_equal':equal,'path_observations':result['path_observations'],
             'actual_fingerprints':{p:r['actual_fingerprint'] for p,r in paths.items()},
             'historical_expected_fingerprint':contract['historical_expected_fingerprint'],
             'payload_differences':result['comparison']['payload_differences'],
             'outer_process':outer,'source_guard_process':process,
             'protected_source_files':len(manifest['files']),'old_result_files_retained':len(old),
             'result_file_inventory':files,'result_output_bytes':sum(r['bytes'] for r in files),
             'audit_only_no_numpy_matrix_or_circuit_import':True,
             'new_registered_Q_T_U_or_payload_acquisition_during_audit':0,
             'CD_E_r6_controlled_Pauli_native_transpile_T_synthesis_sampling_trajectory_LP_DF_molecule_NPZ_GPU_calls':0,
             'same_current_runtime_legacy_reconstruction_only':True,'historical_cause_unique':False,
             'expected_or_hash_policy_modified':False,'science_winner':None,'next_stage_authorized':False,'mandatory_STOP':True}
    write('saved_output_audit_v2.json',summary)
    with (ROOT/AUDIT/'matrix_stage_identity_v2.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    write('payload_auto_name_differences_v2.json',{'differences':summary['payload_differences'],'raw_saved_payload_comparison_only':True,'no_acceptance_canonicalization':True})
    write('STOP.json',{'status':result['status'],'registered_runs':1,'retries':0,'marker_consumed':True,'marker_sha256':markerhash,'next_stage_authorized':False,'source_or_guard_repair_authorized':False,'mandatory_STOP':True})
    print(json.dumps({'status':result['status'],'checks':len(checks),'all_checks_passed':True,'matrix_bytes_equal':all(equal.values()),'payload_auto_name_differences':len(rawdiff),'source_files_unchanged':len(manifest['files']),'old_result_files_unchanged':len(old),'mandatory_STOP':True}))

if __name__=='__main__':main()
