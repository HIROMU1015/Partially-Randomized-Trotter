"""Seal the authorized remaining 216 axes; never execute scientific kernels."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import pf_direct_compile_contract as c
import pf_direct_retry_contract as r
from run_pf_direct_execution import task_estimate


def seal(source):
    c.require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=r.ROOT).decode().strip()==source,'COMMITTED_NEW_SOURCE_REQUIRED')
    target=r.PRIVATE/'sealed01';c.require(not target.exists(),'NEW_SEAL_ALREADY_EXISTS')
    approval=r.PRIVATE/'user_approval01.json';c.require(approval.is_file(),'NEW_APPROVAL_RECEIPT_REQUIRED')
    proposals,selected=r.expected_tasks()
    oldspecpath=r.OLD_PRIVATE/'sealed01/current_4series_722_spec.json'
    oldspec=c.read_json(oldspecpath);c.require(oldspec['SOURCE']==r.OLD_SOURCE,'OLD_SCIENCE_SOURCE_IDENTITY')
    old_preserve=c.read_json(r.ROOT/r.OLD_PACKET/'original_run_preservation.json')['files']
    old_preserve.update(c.read_json(r.ROOT/r.OLD_PACKET/'additional_original_file_hashes.json')['files'])
    # Preserve all stopped-run facts without using any unrelated worktree HEAD/status as a gate.
    for name,digest in oldspec['source_hashes'].items():old_preserve[str(r.OLD_ROOT/name)]=digest
    preservation=r.old.hash_audit(old_preserve)
    c.require(preservation['status']=='PASS','OLD_RUN_OR_SOURCE_CHANGED')
    accepted=c.read_json(r.ROOT/r.OLD_PACKET/'accepted_measurements.json')['rows']
    c.require(len(accepted)==572 and len({row['task_id'] for row in accepted})==572,'EXACT_572_REUSE')
    c.require(not ({row['task_id'] for row in accepted}&{row['task_id'] for row in selected}),'NO_RECOMPILING_572')
    # Old task acceptance does not fill the missing global final gate.
    reuse=dict(status='REUSE_SAVED_TASK_ACCEPTANCE_572_WITH_ORIGINAL_GLOBAL_GATE_LIMIT',
        previous_SOURCE=r.OLD_SOURCE,previous_audit_commit=r.BASE,accepted_task_ids=[v['task_id'] for v in accepted],
        original_global_final_gate_still_missing=True,raw_partial_rows_not_reused=True,
        preserved_old_files=old_preserve,preservation=preservation,
        input_compiler_profile_condition='same exact bound bytes and compiler; native exit/result hashes already reconciled by pinned interruption audit')
    olddocs=(r.OLD_PACKET+'/accepted_measurements.json',r.OLD_PACKET+'/axis_status.csv',r.OLD_PACKET+'/interruption_audit.json')
    for name in olddocs:r.pinned_blob(name)
    systems={s:oldspec['systems'][s] for s in sorted({v['system'] for v in selected})}
    runtime=systems['H4_DF10']['config']['runtime_file_hashes']
    c.require(runtime==systems['H8_DF28']['config']['runtime_file_hashes'],'COMMON_SAME_RUNTIME')
    # Reuse of all 572 old accepted axes also requires their original input,
    # provider, compiler/runtime and profile dependencies to remain identical.
    bound=dict(oldspec['bound_files']);bound.update(runtime)
    legacy_path=r.OLD_PRIVATE/'sealed01/legacy_h4_df12_66_spec.json'
    legacy=c.read_json(legacy_path)
    c.require(legacy['SOURCE']==r.OLD_SOURCE,'OLD_LEGACY_SOURCE_IDENTITY')
    bound.update(legacy['bound_files']);bound[str(legacy_path)]=r.sha(legacy_path)
    bound.update({str(oldspecpath):r.sha(oldspecpath),oldspec['oracles']:r.sha(oldspec['oracles'])})
    for system,value in systems.items():
        bound.update(value['provider_hashes'])
        cfg=value['config'];needed=('tol_snapshot','tol_receipt') if system in ('H2_DF3','H4_DF10') else (
            'h8_input_snapshot.npz','snapshot_receipt.json','df_receipt.json','h8_preparation_export')
        exact={cfg['mapping'][key]:value['input_hashes'][cfg['mapping'][key]] for key in needed}
        value['input_hashes']=exact;bound.update(exact)
        for path,digest in value['provider_hashes'].items():
            relative=str(Path(path).relative_to(cfg['provider_root']))
            raw=subprocess.check_output(['git','show',value['provider_SOURCE']+':'+relative],cwd=cfg['provider_root'])
            c.require(hashlib.sha256(raw).hexdigest()==digest,'PROVIDER_GIT_BLOB')
        profile=value['profile'];reference=profile.get('reference',profile)
        if reference.get('commit'):
            raw=subprocess.check_output(['git','show',reference['commit']+':'+reference['path']],cwd=r.ROOT)
            c.require(hashlib.sha256(raw).hexdigest()==reference['sha256'],'SAME_PROFILE_BYTES')
    bound.update(old_preserve)
    for name in olddocs:bound[str(r.ROOT/name)]=r.sha(r.ROOT/name)
    anchor=r.OLD_PRIVATE/'run01/current_4series_722/H8_DF28__B0I_p17_q8_4th/owned_worker_exit.json'
    peak=c.read_json(anchor);c.require(peak['exit_code']==0 and peak['reaped'] and peak['peak_RSS_bytes']==5336395776,'CONFIRMED_MATCHED_RSS_ANCHOR')
    bound[str(anchor)]=r.sha(anchor)
    footprint={}
    for row in selected:
        if row['system']=='H8_DF28':
            ratio=(int(row['prefix'])+1)*int(row['q'])*(3 if row['PF']=='4th' else 1)/(18*8*3)
            footprint[row['task_id']]=math.ceil(.75*2**30+2*max(1,ratio)*peak['peak_RSS_bytes'])
        else:footprint[row['task_id']]=task_estimate(row)
    # Bind the stdlib observer runtime separately from the unchanged science venv.
    for name in ('hashlib','json','subprocess','selectors','threading','signal','resource','sysconfig'):
        __import__(name)
    observer_files={str(Path(sys.executable).resolve()):r.sha(Path(sys.executable).resolve())}
    for module in list(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if path and Path(path).is_file() and not Path(path).resolve().is_relative_to(r.ROOT):
            observer_files[str(Path(path).resolve())]=r.sha(Path(path).resolve())
    bound.update(observer_files)
    oracles=c.read_json(oldspec['oracles'])
    for row in selected:
        oracle=oracles[row['system']][row['candidate_id']]
        cell=oracle['record']['cell'];c.require(cell['id']==row['candidate_id'],'OWN_SAVED_ORACLE')
        if cell['method']=='B0I':
            receipt=oracle['record'].get('scalar_receipt') or oracle.get('saved_outer_scalar')
            c.require(isinstance(receipt,dict),'SAVED_SCALAR_METADATA_REQUIRED')
            if 'receipt' in receipt:
                c.require(receipt.get('policy')=='REUSED_SAVED_P21_FULL_FOCK_SCALAR','KNOWN_NESTED_POLICY')
                receipt=receipt['receipt']
            c.require('C' in receipt or 'scalar' in receipt,'SUPPORTED_SAVED_SCALAR_METADATA')
    check=r.old.hash_audit(bound);c.require(check['status']=='PASS','RETRY_BOUND_INPUT_CHANGED')
    target.mkdir(parents=True,exist_ok=False)
    r.save(target/'reuse_572_receipt.json',reuse)
    reusepath=target/'reuse_572_receipt.json';bound[str(reusepath)]=r.sha(reusepath)
    spec=dict(schema='PF_DIRECT_RETRY216_SEALED_V1',base_commit=r.BASE,previous_SOURCE=r.OLD_SOURCE,
        contract_id=r.GROUPS[0],SOURCE=source,sealed=True,selected_task_ids=[v['task_id'] for v in selected],
        call_caps=c.call_caps(selected),excluded_conditional_axes=48,compiler_hash=c.COMPILER,T=.8,P=0,control='symmetric_directional',
        source_hashes={name:r.sha(r.ROOT/name) for name in r.SOURCE_FILES},bound_files=bound,
        systems=systems,oracles=oldspec['oracles'],missing_saved_oracles=[],
        output_root=str(r.PRIVATE/'run01'/r.GROUPS[0]),observer_python=str(Path(sys.executable).absolute()),
        observer_runtime_hashes=observer_files,memory_admission_bytes=footprint,
        reuse_receipt=str(reusepath),reuse_receipt_sha256=r.sha(reusepath),
        one_shot=True,retry=False,next_stage_authorized=False,resource_limits=None)
    grant=dict(action='PF_DIRECT_RETRY216_SCIENTIFIC_EXECUTION',approved_by_user=True,runtime_authorization=True,
        contract_id=r.GROUPS[0],SOURCE=source,spec_digest=c.digest(spec),retry=False,next_stage_authorized=False,
        approval_receipt=str(approval),approval_sha256=r.sha(approval))
    r.authorize(spec,grant)
    r.save(target/(r.GROUPS[0]+'_spec.json'),spec);r.save(target/(r.GROUPS[0]+'_grant.json'),grant)
    r.save(target/'seal_audit.json',dict(status='PASS',SOURCE=source,selected_axes=216,reused_saved_axes=572,
        initial_bound_file_audit=check,old_preservation=preservation,matched_memory_anchor=dict(path=str(anchor),sha256=r.sha(anchor),witness=peak),
        actual_scientific_calls_in_preparation=0))
    print(json.dumps(dict(status='SEALED_RETRY216',SOURCE=source,axes=216,reuse=572,bound_files=len(bound))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--SOURCE',required=True)
    seal(p.parse_args().SOURCE)
