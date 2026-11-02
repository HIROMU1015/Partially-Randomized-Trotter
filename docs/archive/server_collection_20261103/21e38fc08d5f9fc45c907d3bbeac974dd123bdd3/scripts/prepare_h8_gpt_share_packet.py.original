"""Prepare a local-only, allowlisted H8 evidence packet. Never dispatch science."""
import ast
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/resource_applicability/track_a_h8_gpt_share_packet/2026-10-20'
CANDIDATE = OUT/'candidate'
LOCAL = OUT/'local_only'
SOURCE = '745a674546f2cb6258526fbfde1d9a00601689f8'
RESULT = 'e6eea5387a6e84d1e39a9adb3b61975acc74e200'
RECOVERY = '253356942845dddf05c2bb6f4582b026e0bce942'
PUBLIC_INDEX = '9f33ad0b11b4febe48b802000c38c5a0e3fd267c'
SCOPES = ['H8_CONFIRM_P8_Q2','H8_CONFIRM_P7_Q16']
RESULT_DIR = 'artifacts/resource_applicability/track_a_h8_new_pr_confirmation_results/2026-10-19/'
AXIS_FIELDS = ['RZ','CX','depth','RZ_depth','CX_depth','size','wall_seconds','compiler_hash','fingerprint','event_digest']
LEDGER_KEYS = {'kind','count','total','nonrefundable'}
artifacts, by_path, mappings, observations = [], {}, [], {}

def require(ok, message):
    if not ok:
        raise ValueError(message)

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()).hexdigest()

def stamp(path):
    s=path.stat()
    return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]

def original(path):
    p=Path(path)
    require(p.is_file() and not p.is_symlink(),'REGULAR_ORIGINAL_REQUIRED')
    before=stamp(p);raw=p.read_bytes()
    require(before==stamp(p),'ORIGINAL_CHANGED_DURING_READ')
    observations[str(p)]=dict(path=str(p),stat=before,sha256=digest(raw),bytes=len(raw))
    return raw

def read(path):
    return json.loads(original(path))

def evidence(path, role):
    p=Path(path);raw=original(p);key=str(p)
    if key not in by_path:
        eid='E'+str(len(artifacts)+1).zfill(5)
        by_path[key]=eid
        artifacts.append(dict(evidence_id=eid,role=role,original_sha256=digest(raw),original_bytes=len(raw),original_bytes_in_packet=False))
    return by_path[key]

def origin_ref(path, role, pointer=''):
    return dict(evidence_id=evidence(path,role),json_pointer=pointer)

def record_mapping(output, origin, keys, transformation):
    mappings.append(dict(packet_location=output,original=origin,selected_keys=keys,transformation=transformation))

def full_audit(path, role, allowed):
    raw=original(path);value=json.loads(raw)
    require(set(value)<=allowed,'UNEXPECTED_FULL_AUDIT_FIELD')
    require(not any(x in raw.decode() for x in ['/'+'home'+'/','/'+'tmp'+'/','password','credential']),'FULL_AUDIT_PRIVATE_CONTENT')
    eid=evidence(path,role)
    next(r for r in artifacts if r['evidence_id']==eid)['original_bytes_in_packet']=True
    return dict(evidence_id=eid,original_json_utf8=raw.decode(),value=value)

def selected(value, keys):
    return {k:value[k] for k in keys if k in value}

def git(ref,path):
    return subprocess.check_output(['git','--no-optional-locks','-C',str(ROOT),'show',ref+':'+path])

def write_json(name,value):
    (CANDIDATE/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def public_copy(path,name):
    raw=git(RESULT,path)
    (CANDIDATE/name).parent.mkdir(parents=True,exist_ok=True)
    (CANDIDATE/name).write_bytes(raw)
    return dict(packet_file=name,commit=RESULT,path=path,sha256=digest(raw),git_blob_sha=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest())

def public_excerpts():
    parts=['# 固定公開SOURCEの検査条件抜粋\n\nコードは実行していない。SOURCE中の条件と実行時receiptは別の証拠である。全文の固定URLとSHAを併記する。\n']
    requests={
        'scripts/h8_pr_confirmation.py':['hierarchy','derive_registry','validate_registry','technical_digest','verify','authorize','native_identity','accept'],
        'scripts/h8_pr_confirmation_parallel.py':['results','verify_descendants'],
        'scripts/prepare_h8_pr_confirmation_resume.py':['verify_manifest'],
    }
    for path,names in requests.items():
        raw=git(SOURCE,path);text=raw.decode();lines=text.splitlines();tree=ast.parse(text)
        parts.append('\n## '+path+'\n\nSOURCE `'+SOURCE+'`; file SHA256 `'+digest(raw)+'`. [全文](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/'+SOURCE+'/'+path+')\n')
        for name in names:
            node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
            parts.append('\n### '+name+' (lines '+str(node.lineno)+'–'+str(node.end_lineno)+')\n\n```python\n'+'\n'.join(lines[node.lineno-1:node.end_lineno])+'\n```\n')
    parts.append('\n`authorize`のCPU配置部分は実行時の条件である。本packetの過去grant照合はSOURCE/technical digest/承認receipt/scopeの結合を対象とし、現在のCPU配置で過去実行を再認可しない。registry導出の再検証は前回の原本照合に記録されている。ここではseed値を提供せず、新たなseed生成・event samplingをしない。\n')
    (CANDIDATE/'10_source_conditions.md').write_text(''.join(parts))

def main():
    require(not OUT.exists(),'FRESH_PACKET_DIRECTORY_REQUIRED')
    CANDIDATE.mkdir(parents=True);LOCAL.mkdir(mode=0o700)
    audit=read(PRIOR/'audit_receipt_PRIVATE.json')
    supplement=read(PRIOR/'provenance_supplement_PRIVATE.json')
    inventory=read(PRIOR/'evidence_inventory_PRIVATE.json')
    previous_by_path={r['path']:r for r in inventory['files']}
    remote=read(PRIOR/'public_read_receipt.json')
    require(audit['status']=='PASS_EVIDENCE_STRUCTURE_ONLY' and not audit['failures'],'PRIOR_RECOVERY_STATUS')
    for row in remote['files']:
        raw=git(row['commit'],row['path'])
        require(digest(raw)==row['sha256'] and len(raw)==row['bytes'],'FIXED_PUBLIC_BLOB')
    public_files=[public_copy(RESULT_DIR+'comparisons.csv','public/comparisons.csv'),public_copy(RESULT_DIR+'paired_metrics.csv','public/paired_metrics.csv'),public_copy('docs/research/track_a_h8_new_pr_confirmation_results_20261019.md','public/result_report.md')]
    comparisons=list(csv.DictReader(io.StringIO((CANDIDATE/'public/comparisons.csv').read_text())))
    batch_path=RUN/'execution01/batch_native_completion.json'
    batch=read(batch_path);batch_started=read(RUN/'execution01/batch_started.json')
    old_started=read(ORIGIN_RUN/'execution01/batch_started.json')
    execution=dict(batch=selected(batch,['SOURCE','mandatory_STOP','automatic_retry','next_stage_authorized']),batch_origin=origin_ref(batch_path,'continuation_batch_native'),scopes=[])
    native=dict(nodes=[],maps=[],identity_redaction='Native PID/starttime/PPID/UID/CPU/paths/timing/RSS removed; aliases preserve locally checked relationships, not independent raw identity verification')
    contracts=dict(scopes=[],ledger_semantics='Parent admission and task-local reservations describe the same calls; do not sum both layers or charge them again as scientific cost')
    tasks_packet=dict(tasks=[],axis_fields=AXIS_FIELDS,private_raw_excluded=['seeds','event arrays','RTE samples','amplitudes','matrices','state arrays','raw shots'])
    bindings=dict(scopes=[],runtime_verification_epoch=audit['checked_at'],runtime_note='Full runtime content hashes were checked in the preceding recovery. This preparation rechecks original preservation and extraction bytes; it does not rerun scientific or runtime initialization')
    result_rows=[];registry_view=None
    node_by_directory={}
    def node(handle,alias,parent,scope,attempt,kind,receipt_path,pointer):
        td=Path(handle['directory']);started_path=td/'started.json';started=read(started_path)
        ident=handle['identity']
        require(ident==started['identity'] and ident[0]==handle['pid'] and started['task']==handle['task_id'],'LOCAL_NATIVE_START_IDENTITY')
        if 'pid' in started:require(started['pid']==handle['pid'],'STARTED_PID_FIELD')
        require(ident[3]==os.getuid(),'LOCAL_NATIVE_UID')
        parent_node=next(n for n in native['nodes'] if n['node_id']==parent)
        parent_raw=parent_node.pop('_raw_pid',None)
        if parent_raw is not None:
            parent_node['_raw_pid']=parent_raw
            require(ident[2]==parent_raw,'LOCAL_PARENT_PID_LINK')
        row=dict(node_id=alias,parent_id=parent,scope=scope,attempt=attempt,kind=kind,task_id=handle['task_id'],reaped=handle['reaped'],exit_code=handle['exit_code'],started_identity_alias=alias,locally_checked_identity_equal=True,origin=origin_ref(receipt_path,'native_map',pointer),started_origin=origin_ref(started_path,'native_started'),_raw_pid=handle['pid'])
        if handle.get('result_sha256'):
            require(digest(original(td/'task_result.json'))==handle['result_sha256'],'NATIVE_RESULT_HASH')
            row.update(result_sha256=handle['result_sha256'],result_evidence_id=evidence(td/'task_result.json','native_task_result'))
        native['nodes'].append(row);node_by_directory[str(td)]=alias
        return row
    for scope in SCOPES:
        label='P8' if scope.endswith('P8_Q2') else 'P7'
        spec_path=RUN/'sealed01'/scope/'technical_seal.json';s=read(spec_path)
        out=Path(s['output']);resume=read(s['resume_manifest']);old=read(resume['origin_spec']);raw=read(out/'result.json');nr_path=out/'native_receipt.json';nr=read(nr_path)
        terminal=full_audit(out/'terminal.json','terminal',{'status','SOURCE','result_sha256','next_stage_authorized','retry'})
        require(terminal['value']['result_sha256']==digest(original(out/'result.json')),'RAW_TERMINAL_HASH')
        startup=read(out/'startup_gate.json');wait_path=RUN/'execution01'/(scope+'_native_completion.json');wait=read(wait_path)
        require(wait==batch['results'][scope] and wait['exit_code']==0 and wait['reaped'] is True and wait['pid']==startup['pid']==batch_started['children'][scope]['pid'],'DRIVER_COMPLETION')
        oldwait=read(resume['parent_wait'])
        driver=label+'.continuation.driver';old_driver=label+'.origin.driver'
        native['nodes'] += [dict(node_id=driver,parent_id=None,scope=scope,attempt='continuation',kind='driver',exit_code=wait['exit_code'],reaped=wait['reaped'],origin=origin_ref(wait_path,'driver_native'),_raw_pid=wait['pid']),dict(node_id=old_driver,parent_id=None,scope=scope,attempt='origin',kind='driver',exit_code=oldwait['exit_code'],reaped=oldwait['reaped'],origin=origin_ref(resume['parent_wait'],'origin_driver_native'),_raw_pid=oldwait['pid'])]
        require(old_started['children'][scope]['pid']==oldwait['pid'],'ORIGIN_DRIVER_STARTED')
        execution['scopes'].append(dict(scope=scope,SOURCE=s['SOURCE'],seal=dict(evidence_id=evidence(spec_path,'technical_seal'),technical_digest=s['technical_digest'],design_digest=s['design_digest'],n=s['n'],cohort=s['cohort'],binding=s['binding'],technical_sealed=s['technical_sealed'],approved=s['approved'],runtime_authorization=s['runtime_authorization'],retry=s['retry'],next_stage_authorized=s['next_stage_authorized']),startup=selected(startup,['status','SOURCE','technical_digest']),startup_origin=origin_ref(out/'startup_gate.json','startup_gate'),driver_node=driver,terminal=terminal,raw_result_evidence_id=evidence(out/'result.json','raw_result'),native_receipt_evidence_id=evidence(nr_path,'native_receipt'),all_reaped=nr['all_reaped'],reused_task_count=nr['reused_task_count']))
        old_map_path=Path(old['output'])/'workers/parallel_receipt.json';old_map=read(old_map_path)
        for group_no,(group,attempt,parent,map_path,pointer) in enumerate([(old_map,'origin',old_driver,old_map_path,'')]+[(g,'continuation',driver,nr_path,'/maps/'+str(i)) for i,g in enumerate(nr['maps'])]):
            aliases=[]
            for i,handle in enumerate(group['workers']):
                replica=int(handle['task_id'].rsplit('_',1)[1])
                alias=label+'.'+attempt+'.trajectory.'+str(replica).zfill(2)
                node(handle,alias,parent,scope,attempt,'trajectory',map_path,pointer+'/workers/'+str(i));aliases.append(alias)
            native['maps'].append(dict(map_id=label+'.'+attempt+'.map.'+str(group_no),scope=scope,attempt=attempt,parent_id=parent,workers=aliases,fields=selected(group,['status','all_reaped','cleanup_errors','automatic_retry','first_error']),origin=origin_ref(map_path,'native_map',pointer)))
        registry=read(s['registry']);seed_plan=read(Path(s['registry']).parent/'seed_plan.json')
        if registry_view is None:
            registry_view=dict(registry_origin=origin_ref(s['registry'],'seed_registry'),seed_plan_origin=origin_ref(Path(s['registry']).parent/'seed_plan.json','seed_plan'),declaration=selected(registry,['schema','SOURCE','seed_plan_digest','old_cohorts_not_reused','redraw_on_collision','actual_RTE_event_sampling']),excluded_digest=seed_plan['excluded_digest'],registered_tasks=[selected(t,['id','scope','replica','cell']) for t in registry['tasks']],prior_local_audit=dict(checked_at=audit['checked_at'],old_excluded_values=supplement['old_excluded_values'],A_registered_masters=supplement['A_registered_masters'],merged_excluded_values=supplement['merged_excluded_values'],registered_confirmation_tasks=supplement['registered_confirmation_tasks'],old_registration_checks=supplement['old_registration_checks'],origin=origin_ref(PRIOR/'provenance_supplement_PRIVATE.json','prior_exclusion_audit')),withheld=['actual master and derived seed values','excluded seed values','raw event arrays'],independent_recheck_limit='The reader can check task coverage and declared cohort/source relationships. Exact deterministic registry derivation, collision/exclusion checks and event digest calculation cannot be independently rerun without withheld values.',mathematical_independence_proven=False)
        task_specs={t['id']:t for t in registry['tasks'] if t['scope']==scope}
        handles=dict(resume['accepted'])
        for group in nr['maps']:
            for handle in group['workers']:handles[handle['task_id']]=handle
        require(len(handles)==48 and set(handles)==set(task_specs),'FINAL48_TASKS')
        grants=[]
        for attempt,sp,grantpath in [('origin',old,ORIGIN_RUN/'execution01'/(scope+'_grant_APPROVED_PRIVATE.json')),('continuation',s,RUN/'execution01'/(scope+'_grant_APPROVED_PRIVATE.json'))]:
            g=read(grantpath);approval=read(g['approval_receipt'])
            require(g['technical_digest']==sp['technical_digest'] and g['SOURCE']==sp['SOURCE'] and g['scope']==scope,'HISTORICAL_GRANT_BINDING')
            require(g['approval_receipt_sha256']==digest(original(g['approval_receipt'])),'APPROVAL_SHA')
            grants.append(dict(attempt=attempt,grant=selected(g,['scope','SOURCE','technical_digest','approved','approved_by_user','runtime_authorization','one_shot','retry','next_stage_authorized','explicit_resume_authorized','approval_receipt_sha256']),grant_origin=origin_ref(grantpath,'grant'),seal=selected(sp,['scope','SOURCE','technical_digest','design_digest','n','cohort','approved','runtime_authorization','technical_sealed','source_hashes']),seal_origin=origin_ref(resume['origin_spec'] if attempt=='origin' else spec_path,'technical_seal'),approval=selected(approval,['approved_by_user','repair_parallel_resume_authorized','design_digest','scopes','n_per_candidate','wrapper_compiles','seed_registry_sha256','reuse_only_full_exit0_native_tasks','old_STOP_and_reservations_preserved','automatic_further_retry','sample_extension','next_stage_authorized']),approval_origin=origin_ref(g['approval_receipt'],'approval_receipt'),marker=full_audit(sp['lock'],'one_shot_marker',{'SOURCE','scope','nonrefundable_attempt'})))
        carry=[]
        require(raw['carry_reservations']==resume['old_reservations'],'EXACT_CARRY_RECORDS')
        for r in resume['old_reservations']:
            require(digest(original(r['path']))==r['sha256'],'CARRY_ORIGINAL_SHA')
            carry.append(full_audit(r['path'],'old_nonrefundable_reservation',LEDGER_KEYS))
        parent_ledger=[full_audit(p,'parent_reservation',LEDGER_KEYS) for p in sorted(out.glob('*_reserved.json'))]
        cp=dict(scope=scope,grants=grants,origin_stop=full_audit(Path(old['output'])/'STOP.json','origin_STOP',{'reason','first_reason_preserved','retry'}),origin_native_map=selected(old_map,['status','all_reaped','cleanup_errors','automatic_retry','first_error']),resume_origin=origin_ref(s['resume_manifest'],'resume_manifest'),resume=selected(resume,['origin_SOURCE','seed_derivation_SOURCE','scope','registry_sha256','failed_or_unfinished_charge_refunded','original_STOP_preserved','quiescence_observations','manifest_digest']),accepted_task_ids=list(resume['accepted']),previous_event_digests=resume['previous_event_digests'],parent_calls=raw['calls'],parent_ledger=parent_ledger,carried_reservations=carry,result_carry_evidence_ids=[x['evidence_id'] for x in carry],old_marker_and_STOP_unmodified=True)
        contracts['scopes'].append(cp)
        raw_pairs={p['cell_id']+'_'+str(p['replica']):p for p in raw['paired_costs']}
        for task_id,handle in sorted(handles.items(),key=lambda item:task_specs[item[0]]['replica']):
            td=Path(handle['directory']);r=read(td/'task_result.json');trace=read(td/'trajectory.json');reused=task_id in resume['accepted'];attempt='origin' if reused else 'continuation'
            require(r==raw_pairs[task_id] and r['seed']==trace['seed']==task_specs[task_id]['seed'],'RAW_TASK_REGISTERED_BINDING')
            require(r['event_digest']==trace['event_digest'] and r['cohort']==trace['cohort']==s['cohort'],'TRACE_COHORT_DIGEST')
            task=dict(scope=scope,task_id=task_id,replica=r['replica'],candidate=r['cell_id'],node_id=node_by_directory[str(td)],measurement_SOURCE=r['SOURCE'],reused_complete_trajectory=reused,cohort=r['cohort'],synthetic=r['synthetic'],normalization_unchanged=r['normalization_unchanged'],trace_flags=selected(trace,['conditional','quantum_weights_changed']),event_digest=r['event_digest'],law_digest=canonical(trace['original_RTE_distribution']),expected_law_digest=s['target']['original_law_sha256'],task_result_origin=origin_ref(td/'task_result.json','task_result'),trace_origin=origin_ref(td/'trajectory.json','trajectory'),calls=r['calls'],ledger=[full_audit(p,'task_reservation',LEDGER_KEYS) for p in sorted(td.glob('*_reserved.json'))],axes={})
            for axis,metric in r['axes'].items():
                mp=td/('measured_'+axis+'.json');m=read(mp)
                require(m==metric and m['seed']==r['seed'],'MEASURED_AXIS_BINDING')
                task['axes'][axis]=dict(metrics=selected(metric,AXIS_FIELDS),origin=origin_ref(mp,'measured_axis'))
            if r['replica']==0:
                control=read(td/'control.json')
                task['control']=dict(fields=selected(control,['status','max_error','tolerance','control_probes','independent_event_oracles','full_Fock_scalar_once','reference_rotated','ordinary_ancilla_branch','parallel_native','native_receipt_sha256']),origin=origin_ref(td/'control.json','first_control'))
            for groupname in ['parallel_axes','parallel_control']:
                rp=td/groupname/'parallel_receipt.json'
                if not rp.exists():continue
                rec=read(rp);parent=task['node_id'];aliases=[]
                for i,ch in enumerate(rec['workers']):
                    alias=parent+'.'+groupname+'.'+ch['task_id']
                    nrw=node(ch,alias,parent,scope,attempt,'axis' if groupname=='parallel_axes' else 'control',rp,'/workers/'+str(i));aliases.append(alias)
                    if groupname=='parallel_axes':
                        child=read(Path(ch['directory'])/'task_result.json')
                        require(child==r['axes'][ch['task_id']],'NESTED_AXIS_METRIC_MATCH')
                        nrw['axis_metrics']=selected(child,AXIS_FIELDS)
                native['maps'].append(dict(map_id=parent+'.'+groupname,scope=scope,attempt=attempt,parent_id=parent,workers=aliases,fields=selected(rec,['status','all_reaped','cleanup_errors','automatic_retry','first_error']),origin=origin_ref(rp,'nested_native_map')))
            tasks_packet['tasks'].append(task)
            record_mapping('06_task_axis_links.json#/tasks/'+str(len(tasks_packet['tasks'])-1),task['task_result_origin'],list(r),'Allowlist metrics and audit flags; remove seed, raw events and paths; native identities become aliases')
        prefix=str(s['target']['cell']['prefix']);prep=read(out/'preparation_binding.json');restore=read(out/'preparation_receipt.json')
        runtime=read(s['runtime_receipt']);previous=next(x for x in audit['scopes'] if x['scope']==scope)
        payload=read(s['payload']);native_path=payload['native_adapter']['native_config'];native_config=read(native_path)
        minimal_prep=selected(prep,['saved_export_sha256','new_Gaussian_operations','new_SCF_DF_state','compiler_sha256','profile_sha256'])
        minimal_prep['prefixes']={k:selected(v,['preparation_hash','saved_exact_export_match']) for k,v in prep['prefixes'].items()}
        minimal_restore=dict(prefixes={k:selected(v,['preparation_hash','hamiltonian_hash']) for k,v in restore['prefixes'].items()},SCF_DF_state_solver=restore['SCF_DF_state_solver'])
        bindings['scopes'].append(dict(scope=scope,binding=s['binding'],source_hashes=s['source_hashes'],input_rows=[dict(input_id=scope+'.input.'+str(i).zfill(3),role=row['role'],bytes=row['bytes'],sha256=row['sha256'],origin_evidence_id=evidence(row['path'],'bound_input')) for i,row in enumerate(s['inputs'])],provider_closure=dict(expected_files=sum(len(v) for v in s['provider_closure'].values()),previous_content_hash_match=True,previous_check=next(x for x in audit['checks'] if x['check']==scope+':existing_verify')),runtime_closure=dict(expected_files=len(runtime['expected']),runtime_receipt_sha256=s['runtime_receipt_sha256'],runtime_receipt_origin=origin_ref(s['runtime_receipt'],'runtime_receipt'),python_sha256=s['python_sha256'],previous_content_hash_match=True,previous_check=next(x for x in audit['checks'] if x['check']==scope+':runtime_all_content_hashes')),preparation_binding=minimal_prep,preparation_receipt=minimal_restore,target_preparation_sha256=s['target']['preparation_sha256'],profile_reference=s['profile_reference'],prior_audit_origin=origin_ref(PRIOR/'audit_receipt_PRIVATE.json','prior_byte_audit'),independent_recheck_limit='Input/provider/runtime bytes and machine file names are withheld. This preserves expected hashes, roles, counts and prior verification conditions; a recipient cannot rerun original-byte verification from this packet alone.'))
        bindings['scopes'][-1].update(compiler_options=native_config['compiler_options'],compiler_config_origin=origin_ref(native_path,'native_config'))
        saved=[r for r in comparisons if r['scope']==scope]
        result_rows.append(dict(scope=scope,candidate=s['target']['cell'],n=48,axes=96,delta_literal='.4' if scope.endswith('P8_Q2') else '.05',result_commit=RESULT,science_SOURCE=s['SOURCE'],seed_derivation_SOURCE=s['seed_derivation_SOURCE'],raw_result_evidence_id=evidence(out/'result.json','raw_result'),saved_comparison_rows=saved,saved_result_claims={mode:selected(raw[mode],['cohort','population_conclusion','confidence_interval','p_value','old_samples_pooled','interpretation']) for mode in ['primary','secondary']},result_flags=selected(raw,['status','SOURCE','new_signal','new_PF_compile','new_fit','new_quantum_shots','automatic_retry','next_stage_authorized','unobserved_rare_costs_not_zero_filled']),reused_complete_trajectories=len(resume['accepted'])))
    for n in native['nodes']:n.pop('_raw_pid',None)
    write_json('01_results_and_source.json',dict(context=dict(system='linear H8',spacing_Angstrom='1.0',basis='STO-3G',DF_rank=28,rank_policy='tol_only_1e-8',spin_modes=16,sector_dimension=4900,T='.8',P=0,PR_formula='2nd',r=4,K=4,primary_bias='USUAL_SIGNED',allowance_factor=1,control='ordinary ancilla; symmetric_directional',full_Fock_scalar_once=True,reference_rotated=False),conditions=result_rows,statistics='Saved literal tokens copied only; no mean/SD/SE/cost/percentage/CI/p-value recomputation',fixed_PF_audit='Selected public deterministic PF receipts only; never a substitute for PR native execution audit'))
    # This digest is an actual deterministic seed-derivation input. It must not
    # accompany public source/cell/replica values even when seed values are absent.
    registry_view['declaration'].pop('seed_plan_digest',None)
    registry_view['seed_plan_derivation_key_withheld']=True
    registry_view['withheld'].append('seed-plan derivation-input digest; public code plus this value can reconstruct master seeds')
    write_json('02_execution_receipts.json',execution);write_json('03_worker_receipts.json',native)
    write_json('04_grant_ledger_marker_stop.json',contracts);write_json('05_seed_cohort_audit.json',registry_view)
    tasks_packet['tasks'].sort(key=lambda t:(SCOPES.index(t['scope']),t['replica']))
    write_json('06_task_axis_links.json',tasks_packet);write_json('07_input_runtime_bindings.json',bindings)
    public_excerpts()
    origin_source='1549b6f891a15ff48379996c34b344118ba2606c';origin_path='scripts/h8_pr_confirmation.py';origin_raw=git(origin_source,origin_path)
    origin_pin=dict(commit=origin_source,path=origin_path,sha256=digest(origin_raw),git_blob_sha=hashlib.sha1(b'blob '+str(len(origin_raw)).encode()+b'\0'+origin_raw).hexdigest(),url='https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/'+origin_source+'/'+origin_path,verification='Local fixed Git blob matched to origin seal source_hashes; no new remote fetch')
    write_json('11_public_sources.json',dict(fixed_commits=dict(local_recovery=RECOVERY,public_index=PUBLIC_INDEX,result=RESULT,science_SOURCE=SOURCE,seed_derivation_and_reused_measurement_SOURCE=origin_source),origin_source_pin=origin_pin,public_files_in_packet=public_files,previous_public_read_receipt=remote,network_access_in_this_preparation=False))
    # Full nonshared original locator remains only in this local branch.
    actual_locations=[dict(evidence_id=eid,**observations[path]) for path,eid in by_path.items()]
    for row in observations.values():
        previous=previous_by_path.get(row['path'])
        if previous:require(row['sha256']==previous['sha256'],'PREVIOUS_ORIGINAL_HASH_CHANGED')
        require(stamp(Path(row['path']))==row['stat'],'ORIGINAL_CHANGED_AFTER_EXTRACTION')
    # Read-only stat preservation of the previous known-file inventory, no broad scan.
    for row in inventory['files']:require(stamp(Path(row['path']))==row['stat'],'PRIOR_INVENTORY_STAT_CHANGED')
    write_json('08_provenance_and_redactions.json',dict(artifacts=artifacts,source_fields='All values are selected from existing originals or explicitly marked local checks; original paths are omitted',full_audit_bytes='Only allowlisted terminal/STOP/one-shot/reservation audit JSON are included as exact UTF-8 strings, enabling original SHA256 checks. No scientific raw is included.',redacted_projections='Other original SHA256 values cannot be recomputed from redacted fields. They are provenance links rather than authentication or standalone proof of original contents.',identity_limit='Aliases preserve locally checked graph relationships, but actual PID/starttime/PPID/UID consistency cannot be independently rechecked after redaction.',registry_limit='Registry and exclusion values are withheld. A recipient cannot repeat noncollision/exclusion or seed/event-hierarchy checks.',hash_limit='Hashes of private originals disclose correlation identifiers and byte lengths. They do not disclose raw contents or replace access to originals.',new_scientific_calls=0,new_sampling=0,new_compile=0,new_statistics=0,result_recalculation=False,scientific_validity_review_started=False,final_evidence_freeze=False,actual_external_sharing_authorized=False,actual_external_sharing_performed=False,status='MANDATORY_STOP_AFTER_PREPARATION'))
    index=dict(local_only=True,never_attach=True,packet_to_original=actual_locations,field_mappings=mappings,observed_originals=list(observations.values()),prior_preservation_inventory=str(PRIOR/'evidence_inventory_PRIVATE.json'),prior_known_files_stat_unchanged=len(inventory['files']),input_original_files_rehashed=len(observations),raw_identity_values_not_copied=True,raw_seed_values_not_copied=True)
    (LOCAL/'NONSHARE_original_index_PRIVATE.json').write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n')
    os.chmod(LOCAL/'NONSHARE_original_index_PRIVATE.json',0o600)
    (LOCAL/'README_NONSHARE.md').write_text('# 非共有原本索引\n\nこのdirectoryはZIPから除外する。絶対pathと原証拠の所在を含むためChatGPTへ添付しない。`NONSHARE_original_index_PRIVATE.json`でpacketのE番号から原本へ戻れる。原fileを変更していない。原seed値・認証情報はこの索引にも転記していない。branch全体のpushは禁止。\n')
    print(json.dumps(dict(status='EXTRACTIONS_PREPARED_LOCALLY',artifact_links=len(artifacts),tasks=len(tasks_packet['tasks']),native_nodes=len(native['nodes']),originals_rehashed=len(observations),prior_inventory_stat_preserved=len(inventory['files']),new_scientific_calls=0),indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior-dossier',type=Path,required=True)
    parser.add_argument('--continuation-run',type=Path,required=True)
    parser.add_argument('--origin-run',type=Path,required=True)
    parser.add_argument('--output-directory',type=Path,default=OUT)
    args=parser.parse_args()
    PRIOR,RUN,ORIGIN_RUN=args.prior_dossier,args.continuation_run,args.origin_run
    OUT=args.output_directory
    CANDIDATE,LOCAL=OUT/'candidate',OUT/'local_only'
    main()
