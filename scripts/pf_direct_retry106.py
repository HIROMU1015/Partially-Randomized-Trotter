"""Closed 106-axis repair execution, reusing the existing owned-process runner."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pf_direct_compile_contract as c
import pf_direct_execution as old
from pf_direct_control_reference import policy_for,OVERLAY

ROOT=Path(__file__).resolve().parents[1]
BASE='636208ba6d74af13e7c405970c896b621022af65'
PREVIOUS_SOURCE='0c5fcc83ef55045c2c64746aadfaf3128667b058'
PREVIOUS_ROOT=ROOT.parent/'pf-direct-compile-retry216-20261020'
PREVIOUS_PRIVATE=PREVIOUS_ROOT/'.pf_direct_retry216_private'
PRIVATE=ROOT/'.pf_direct_retry106_private'
PUBLIC=ROOT/'artifacts/resource_applicability/track_a_pf_direct_compile_retry106/2026-10-20'
PRIOR_PACKET='artifacts/resource_applicability/track_a_pf_direct_compile_retry216/2026-10-20'
OLD_PACKET='artifacts/resource_applicability/track_a_pf_direct_compile_interrupted_audit/2026-10-20'
OLD_MEASUREMENTS_PATH=ROOT/PRIOR_PACKET/'results01/combined_observation_index.json'
GROUPS=('remaining_106',)
EXPECTED_NEW_AXES=106
EXPECTED_OLD_AXES=682
NEW_PROVENANCE='NEW_RETRY106_DIRECT_OBSERVATION'
SUCCESS_STATUS='PASS_RETRY106_COMPLETE_MANDATORY_STOP'
REUSE_FILENAME='reuse_682_receipt.json.gz'
PRESERVE_REUSED_ROWS=True
SOURCE_FILES=('scripts/pf_direct_retry106.py','scripts/pf_direct_control_reference.py',
 'scripts/pf_direct_production_worker.py','scripts/pf_direct_retry_support.py',
 'scripts/run_pf_direct_execution.py','scripts/observe_pf_direct_retry216.py',
 'scripts/launch_pf_direct_retry216.py','scripts/audit_pf_direct_retry216.py',
 'scripts/pf_direct_retry_contract.py','scripts/pf_direct_execution.py',
 'scripts/pf_direct_compile_contract.py','scripts/pf_direct_compile_adapter.py',
 'scripts/h8_strengthened_export.py','scripts/h8_strengthened_contract.py',
 'tests/test_pf_direct_retry106.py')
ADMISSION_POLICY=dict(policy='LIVE_PERMITTED_CPU_MEMORY_WITH_PREVIOUS_STRUCTURAL_ESTIMATE',
 CPU_RAM_worker_limits=None,PSI='observation only',no_automatic_retry=True,
 estimate='Existing native structural memory estimate; admission reservation only, no process RAM ceiling')
save=old.save
append=old.append
sha=old.sha
candidate_pairs=old.candidate_pairs
resource_snapshot=old.resource_snapshot
choose_admissions=old.choose_admissions


def pinned(name):
    raw=subprocess.check_output(['git','show',BASE+':'+name],cwd=ROOT)
    c.require((ROOT/name).read_bytes()==raw,'PRIOR_BYTES_CHANGED:'+name)
    return raw


def expected_tasks():
    status=list(csv.DictReader(io.StringIO(pinned(PRIOR_PACKET+'/results01/retry_axis_status.csv').decode())))
    c.require(len(status)==216 and sum(v['status']=='STOP_RETRY' for v in status)==106,'EXACT_PREVIOUS_OUTCOME')
    proposals,_=old.expected_tasks('current_4series_722')
    selected=c.select_pairs(proposals,[v['task_id'] for v in status if v['status']=='STOP_RETRY'])
    c.require(len(selected)==106 and {s:sum(v['system']==s for v in selected) for s in {v['system'] for v in selected}}
        ==dict(H2_DF3=46,H4_DF10=60),'ONLY_REMAINING_106')
    for task in selected:policy_for(task)
    return proposals,selected


def reference_bindings(selected):
    return {s+':'+cid:policy_for(pair[0]) for (s,cid),pair in candidate_pairs(selected).items()}


def authorize(spec,grant):
    c.require(grant.get('action')=='PF_DIRECT_RETRY106_SCIENTIFIC_EXECUTION'
        and grant.get('approved_by_user') is True and grant.get('runtime_authorization') is True,
        'EXPLICIT_RETRY106_GRANT_REQUIRED')
    c.require(grant.get('spec_digest')==c.digest(spec) and grant.get('SOURCE')==spec.get('SOURCE')
        and grant.get('contract_id')==spec.get('contract_id')==GROUPS[0],'EXACT_NEW_CONTRACT')
    proposals,selected=expected_tasks()
    c.require(spec.get('selected_task_ids')==[v['task_id'] for v in selected]
        and spec.get('call_caps')==c.call_caps(selected),'EXACT_106_AXES_AND_CALLS')
    c.require(spec.get('control_reference_bindings')==reference_bindings(selected),'SEALED_REFERENCE_BINDINGS')
    c.require(spec.get('base_commit')==BASE and spec.get('previous_SOURCE')==PREVIOUS_SOURCE
        and spec.get('sealed') is True and re.fullmatch('[0-9a-f]{40}',spec.get('SOURCE','')),
        'NEW_COMMITTED_SOURCE')
    c.require(spec.get('T')==.8 and spec.get('P')==0 and spec.get('control')=='symmetric_directional'
        and spec.get('compiler_hash')==c.COMPILER and spec.get('excluded_conditional_axes')==48,'FIXED_SCIENTIFIC_CONDITIONS')
    c.require(all(doc.get('retry') is False and doc.get('next_stage_authorized') is False for doc in (spec,grant)),
        'NO_FURTHER_AUTOMATIC_RETRY')
    approval=Path(grant['approval_receipt']);c.require(sha(approval)==grant['approval_sha256'],'USER_APPROVAL_BYTES')
    receipt=c.read_json(approval)
    c.require(receipt['verbatim_user_message']=='修正を行って再計算して' and receipt['role']=='user'
        and receipt['authorized_axes']==106 and receipt['scientific_execution_authorized'] is True,'NEW_106_APPROVAL')
    c.require(Path(spec['output_root'])==PRIVATE/'run01'/GROUPS[0],'OWN_EXCLUSIVE_NEW_OUTPUT')
    c.require(set(spec['source_hashes'])==set(SOURCE_FILES) and spec['bound_files'],'COMPLETE_SOURCE_CLOSURE')
    c.require(sha(spec['reuse_receipt'])==spec['reuse_receipt_sha256'],'REUSE_682_BYTES')
    return proposals,selected


def gate(spec,grant):
    authorize(spec,grant)
    for name,digest in spec['source_hashes'].items():
        c.require(sha(ROOT/name)==digest,'SOURCE_CHANGED:'+name)
        blob=subprocess.check_output(['git','show',spec['SOURCE']+':'+name],cwd=ROOT)
        c.require(hashlib.sha256(blob).hexdigest()==digest,'COMMITTED_SOURCE_BLOB:'+name)
    audit=old.hash_audit(spec['bound_files']);c.require(audit['status']=='PASS','BOUND_BYTE_CHANGE:'+str(audit['errors']))
    fresh=resource_snapshot()
    c.require(fresh['cpu_slots']>0 and fresh['effective_available_memory']>0 and fresh['disk_free']>0,'NO_PERMITTED_RESOURCE')
    return dict(status='PASS',SOURCE=spec['SOURCE'],spec_digest=c.digest(spec),byte_audit=audit,resource=fresh)


def validate_saved_worker(pair,files,witness,spec):
    key=pair[0]['system']+':'+pair[0]['candidate_id'];expected=spec['control_reference_bindings'][key]
    reference=files['control_reference.json'];diagnostics=files['control_diagnostics.json']
    c.require(all(reference[k]==v for k,v in expected.items()),'SAVED_REFERENCE_POLICY_BINDING')
    c.require(diagnostics['reference_policy']==expected and diagnostics['status']=='PASS'
        and diagnostics['all_errors_finite'] and max(diagnostics['errors'])<diagnostics['tolerance'],
        'SAVED_CONTROL_DIAGNOSTICS_PASS')
    for row in files['accepted_measurements.json']['rows']:
        c.require(row['control_reference_binding']==expected
            and row['requires_saved_classical_signal_reconciliation']==expected['single_term_S4_saved_signal_inconsistent'],
            'ROW_REFERENCE_QUALIFICATION')


def prepare(source):
    c.require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()==source,'COMMITTED_SOURCE_REQUIRED')
    target=PRIVATE/'sealed01';c.require(not target.exists(),'ONE_SHOT_SEAL_ALREADY_EXISTS')
    approval=PRIVATE/'user_approval01.json';c.require(approval.is_file(),'NEW_APPROVAL_RECEIPT_REQUIRED')
    _,selected=expected_tasks()
    previous_path=PREVIOUS_PRIVATE/'sealed01/current_remaining_216_spec.json'
    previous=c.read_json(previous_path);c.require(previous['SOURCE']==PREVIOUS_SOURCE,'PREVIOUS_SEALED_SOURCE')
    previous_audit=c.read_json(ROOT/PRIOR_PACKET/'results01/completion_audit.json')
    c.require(previous_audit['coordinator_summary']['all_workers_reaped']
        and previous_audit['post_execution_byte_gate']['status']=='PASS','PREVIOUS_EXIT_AND_BYTE_AUDIT')
    pinned(PRIOR_PACKET+'/results01/combined_observation_index.json')
    accepted=c.read_json(OLD_MEASUREMENTS_PATH)['rows']
    c.require(len(accepted)==682 and len({v['task_id'] for v in accepted})==682,'EXACT_REUSE_682')
    c.require({v['task_id'] for v in accepted}.isdisjoint(v['task_id'] for v in selected),'NO_RECOMPILING_682')
    bound=dict(previous['bound_files']);bound[str(previous_path)]=sha(previous_path)
    for name,digest in previous['source_hashes'].items():bound[str(PREVIOUS_ROOT/name)]=digest
    # Only this completed run's actual receipts/source/results are frozen, never unrelated HEAD/status.
    prior_files={}
    for directory,dirs,files in os.walk(PREVIOUS_PRIVATE):
        dirs[:]=[name for name in dirs if name not in ('tmp','numba','mpl','cache','__pycache__')]
        for name in files:
            path=Path(directory)/name
            if path.suffix in ('.json','.jsonl','.log'):prior_files[str(path)]=sha(path)
    bound.update(prior_files)
    for path in (ROOT/PRIOR_PACKET).rglob('*'):
        if path.is_file():
            name=str(path.relative_to(ROOT));pinned(name);bound[str(path)]=sha(path)
    systems={name:previous['systems'][name] for name in ('H2_DF3','H4_DF10')}
    oracles=c.read_json(previous['oracles'])
    for (system,cid),pair in candidate_pairs(selected).items():
        row=oracles[system][cid]['record'];policy=policy_for(pair[0])
        c.require(row['cell']['id']==cid and policy['field'] in row['signals']
            and policy['field'] in row['independent_forward_signals'],'EXISTING_SAVED_REFERENCE_FIELDS')
        if policy['method']=='B0I':
            c.require(row['allowance']['policy']==OVERLAY and row['scalar_receipt']['policy']=='EXACT_STORED_FULL_FOCK_TAIL_IDENTITY',
                'KNOWN_D_OVERLAY_SCHEMA')
    # Pin the source evidence for the correction, including the one-term iterator.
    diagnosis_files=(
      ROOT.parent/'h2-pr-resource-preparation-20261019/scripts/h2_d_recovery_science.py',
      ROOT.parent/'h4-tol-only-main-phase0-20261019/scripts/h4_tol_main_science.py',
      ROOT.parent/'h4-tol-only-main-phase0-20261019/scripts/h6_native_b0i_bridge_v1.py')
    for path in diagnosis_files:bound[str(path)]=sha(path)
    for name in ('argparse','csv','io','gzip','resource','signal','selectors','threading','sysconfig'):
        __import__(name)
    observer_files={str(Path(sys.executable).resolve()):sha(Path(sys.executable).resolve())}
    for module in list(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if path and Path(path).is_file() and not Path(path).resolve().is_relative_to(ROOT):
            observer_files[str(Path(path).resolve())]=sha(Path(path).resolve())
    bound.update(observer_files)
    audit=old.hash_audit(bound);c.require(audit['status']=='PASS','PRE_SEAL_BYTE_AUDIT')
    target.mkdir(parents=True,exist_ok=False)
    reuse=target/'reuse_682_receipt.json'
    save(reuse,dict(status='REUSE_682_SAVED_TASK_ACCEPTANCES',accepted_task_ids=[v['task_id'] for v in accepted],
        previous_completion_commit=BASE,previous_SOURCE=PREVIOUS_SOURCE,previous_new_110_final_gate='PASS',
        original_572_global_final_gate_still_missing=True,prior_receipt_files=prior_files,
        observed_input_compiler_runtime_profile_bytes_unchanged=True))
    bound[str(reuse)]=sha(reuse)
    spec=dict(schema='PF_DIRECT_RETRY106_SEALED_V1',base_commit=BASE,previous_SOURCE=PREVIOUS_SOURCE,
        SOURCE=source,sealed=True,contract_id=GROUPS[0],selected_task_ids=[v['task_id'] for v in selected],
        call_caps=c.call_caps(selected),T=.8,P=0,control='symmetric_directional',compiler_hash=c.COMPILER,
        excluded_conditional_axes=48,source_hashes={name:sha(ROOT/name) for name in SOURCE_FILES},
        bound_files=bound,systems=systems,oracles=previous['oracles'],missing_saved_oracles=[],
        control_reference_bindings=reference_bindings(selected),output_root=str(PRIVATE/'run01'/GROUPS[0]),
        observer_python=str(Path(sys.executable).absolute()),observer_runtime_hashes=observer_files,
        reuse_receipt=str(reuse),reuse_receipt_sha256=sha(reuse),one_shot=True,retry=False,
        next_stage_authorized=False,resource_limits=None)
    grant=dict(action='PF_DIRECT_RETRY106_SCIENTIFIC_EXECUTION',approved_by_user=True,runtime_authorization=True,
        SOURCE=source,contract_id=GROUPS[0],spec_digest=c.digest(spec),approval_receipt=str(approval),
        approval_sha256=sha(approval),retry=False,next_stage_authorized=False)
    authorize(spec,grant)
    save(target/(GROUPS[0]+'_spec.json'),spec);save(target/(GROUPS[0]+'_grant.json'),grant)
    save(target/'seal_audit.json',dict(status='PASS',SOURCE=source,byte_audit=audit,new_axes=106,
        reused_saved_axes=682,reference_policies=reference_bindings(selected),new_scientific_calls=0))
    print(json.dumps(dict(status='SEALED_106',SOURCE=source,axes=106,reuse=682,bound_files=len(bound))))


def command(action,spec,seal,directory=None):
    result=[spec['observer_python'],'-B',str(ROOT/'scripts/pf_direct_retry106.py'),'--action',action,
        '--sealed-directory',str(seal)]
    if directory is not None:result+=['--directory',str(directory)]
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=('prepare','launch','observer','run','audit','worker'),default='worker')
    parser.add_argument('--SOURCE');parser.add_argument('--sealed-directory',type=Path)
    parser.add_argument('--directory',type=Path)
    for name in ('spec','grant','system','candidate','output'):parser.add_argument('--'+name)
    args=parser.parse_args();api=sys.modules[__name__]
    if args.action=='prepare':prepare(args.SOURCE)
    elif args.action=='launch':
        from launch_pf_direct_retry216 import main as launch
        launch(args.sealed_directory,api=api)
    elif args.action=='observer':
        from observe_pf_direct_retry216 import main as observe
        return observe(args.sealed_directory,args.directory,api=api)
    elif args.action=='run':
        from run_pf_direct_execution import run
        run(args.sealed_directory,execution_api=api,worker_script='scripts/pf_direct_retry106.py')
    elif args.action=='audit':
        from audit_pf_direct_retry216 import audit
        return audit(args.sealed_directory,api=api)
    else:
        from pf_direct_production_worker import run
        return run(args.spec,args.grant,args.system,args.candidate,args.output,authorize=authorize)
    return 0


if __name__=='__main__':raise SystemExit(main())
