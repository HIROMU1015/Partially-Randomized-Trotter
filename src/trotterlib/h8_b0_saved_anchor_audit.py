"""Saved H8 anchor evidence and administrative Git checks; stdlib only.

This module has no scientific executor, imports no molecular libraries and never
fits costs. A successful audit does not change the original acquisition STOP.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

SOURCE='db559a518214fd27fa96e2f1451911df05c4f4a8'
RESULT_COMMIT='af10b55420d31e227905f7762971e4b39c6c6a05'
PROFILE='622ce4a8391002558b877240b48d831d9bb0a72aeabba2e4d73bfec64b9c9f12'
COMPILER='1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b'
PROVIDER='c31ef424175c02379eafebe8fc6c7548c1d9aabe'
RUNTIME_MAP='850568c8018943b91dc9dbb031198b684112f88f8870f5a217253c61ee5fb958'
PROVIDER_MAP='9382176df9a1353c9f02b842edec5943dfbeb06b223387f6a99542e53c878bc0'
ARTIFACT='artifacts/resource_applicability/track_a_h8_b0_anchor_run02_stop/2026-10-16'
PLAN='artifacts/resource_applicability/track_a_h8_b0_anchor_readiness/2026-10-16/acquisition_plan_UNAPPROVED_v1.json'
PLAN_SHA='c7755c88d9a6e465411292bc0d6196c8d8f05f8c5642994fe3d1291d93a27273'
GIT_OPTIONS=('--no-optional-locks','-c','index.threads=1','-c','core.preloadIndex=false',
             '-c','core.fsmonitor=false','-c','core.untrackedCache=false')


def git_bytes(root,*args):
    """Per-command settings only: no index refresh writes, preload or read threads."""
    return subprocess.check_output(['git',*GIT_OPTIONS,*args],cwd=root)


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def require(condition,reason):
    if not condition:raise ValueError(reason)


def relative_path(name):
    p=Path(name)
    require(not p.is_absolute() and '..' not in p.parts,'UNSAFE_EVIDENCE_PATH')
    return p


def validate_records(plan,bundle):
    """Validate existing values/receipts, never derive model or total costs."""
    driver=bundle['driver'];launcher=bundle['launcher'];native=bundle['native'];admission=bundle['admission']
    require(driver['status']==launcher['status']==admission['status']=='STOP','ORIGINAL_STOP_REQUIRED')
    require(launcher['first_stop']=='RuntimeError:PROCESS_THREADS' and
            driver['error']=='DRIVER_STOP_SIGNAL_OR_DEADLINE','EXPECTED_STOP_REASON')
    require(launcher['all_reaped'] and not launcher['cleanup_errors'] and launcher['native_exit_code']==-9,
            'CLEANUP_OR_DRIVER_EXIT')
    require(native['native_exit_code']==1 and native['wait_completed'],'LAUNCHER_NATIVE_WAIT')
    require(not bundle['final_gate_exists'],'ORIGINAL_FINAL_GATE_UNEXPECTED')
    require(driver['SOURCE']==SOURCE and driver['scope']==plan['scope'],'DRIVER_SCOPE_SOURCE')
    require(driver['unknown_reserved_compile_calls']==0 and driver['compile_entry_markers']==8,
            'UNKNOWN_COMPILE_RESERVATIONS')
    expected={(case['id'],axis):case for case in plan['cases'] for axis in ('cosine','sine')}
    rows=bundle['train']['axes']+bundle['holdout']['axes']
    require(len(rows)==8 and len(expected)==8 and {(r['case_id'],r['axis']) for r in rows}==set(expected),
            'EXACT_EIGHT_AXES')
    require(bundle['train']['role']=='TRAIN' and len(bundle['train']['axes'])==6 and
            bundle['holdout']['role']=='HOLDOUT' and len(bundle['holdout']['axes'])==2 and
            not bundle['train']['fit_performed'] and not bundle['holdout']['fit_performed'], 'FIXED_SPLIT_NO_FIT')
    reserved=driver['irrevocable_reservations'];settled=driver['settlements']
    require(set(reserved)==set(settled)=={c+'_'+a for c,a in expected} and
            all(reserved[k]==settled[k]=={'compile':1} for k in reserved),'EXACT_SETTLEMENTS')
    require(driver['completed_calls_lower_bound']==plan['call_caps'],'EXACT_CALL_COUNTS_NO_FORBIDDEN_CALLS')
    controls={v['case']['id']:v for v in bundle['controls']}
    require(len(bundle['controls'])==len(controls)==4,'FOUR_CONTROLS')
    workers={v['task_id']:v for wave in bundle['waves'] for v in wave['workers']}
    require(len(workers)==8 and len(bundle['waves'])==2 and
            all(len(wave['workers'])==4 for wave in bundle['waves']),'EIGHT_WORKERS_TWO_WAVES')
    require(all(v['status']=='PASS' and v['all_reaped'] and not v['cleanup_errors'] and
                v['automatic_retry'] is False for v in bundle['waves']),'WAVE_FAILURE')
    require([v['rows'] for v in bundle['accepted']]==[rows[:4],rows[4:]] and
            all(v['all_reaped'] and v['next_stage_authorized'] is False for v in bundle['accepted']),
            'ACCEPTED_ROWS_NOT_SAVED_ROWS')
    for row in rows:
        case=expected[(row['case_id'],row['axis'])];control=controls[case['id']]
        preparation=plan['binding']['prefix_preparations'][str(case['prefix'])]
        require(row['role']==case['role'] and row['SOURCE']==SOURCE and
                row['scope']==plan['scope'] and row['profile_sha256']==PROFILE and
                row['compiler_hash']==COMPILER,'AXIS_LINEAGE')
        require(row['control']==plan['control'] and row['scalar_policy']==plan['scalar_policy'] and
                row['preparation_binding']==preparation,'AXIS_PREPARATION_CONTROL_SCALAR')
        error=row['max_control_error']
        require(type(error) in (int,float) and math.isfinite(error) and 0<=error<1e-8,'CONTROL_TOLERANCE')
        require(control['case']==case and control['SOURCE']==SOURCE and control['profile_sha256']==PROFILE and
                control['max_control_error']==error and control['preparation_binding']==preparation and
                control['scalar_policy']==plan['scalar_policy'] and
                control['basis_observed_from_restored_live_operations'] is True and
                control['observed_ordered_block_description_sha256']==preparation['ordered_block_description_sha256'],
                'LIVE_PREPARATION_RECEIPT')
        require(row['fingerprint_before']==row['fingerprint_after']==control['wrapper_fingerprints'][row['axis']],
                'WRAPPER_FINGERPRINT')
        require(all(type(row[k]) is int and row[k]>0 for k in ('measured_RZ','CX','depth','size')),
                'POSITIVE_SAVED_COMPILED_COUNTS')
        worker=workers.get(case['id']+'_'+row['axis'])
        require(worker is not None and worker['reaped'] and worker['exit_code']==0 and row['reaped'] and
                row['native_exit_code']==0 and worker['identity']==row['worker_identity'],'WORKER_NATIVE_EXIT_REAP')
    require(not any(v.get('retry',v.get('automatic_retry',False)) or v.get('next_stage_authorized',False)
                    for v in (driver,launcher,native,admission)),'NO_RETRY_NO_NEXT_STAGE')
    return dict(status='SAVED_EIGHT_AXES_STRUCTURALLY_REUSABLE_PENDING_B1_APPROVAL',
        measured_axes_verified=8,TRAIN_axes=6,HOLDOUT_axes=2,all_compile_workers_exit0_reaped=True,
        original_acquisition_status='STOP',original_final_gate_replaced=False,
        historical_thread_violator='UNRESOLVED_NOT_RECORDED',historical_thread_compliance_certified=False,
        new_compile=0,new_signal=0,new_fit=0,new_residual=0,new_cost_reaccount=0,new_valid_grant=0,
        automatic_retry=False,B1_authorized=False,next_stage_authorized=False,mandatory_STOP=True)


def audit_saved(evidence,execution_worktree,published_worktree):
    """Rehash originals and old code; never mutate their directories or Git config."""
    e=Path(evidence);p=e/'prepared';o=p/'output01';execution=Path(execution_worktree);published=Path(published_worktree)
    inventory=read(published/ARTIFACT/'publication_files_v1.json')
    require(git_bytes(published,'show',RESULT_COMMIT+':'+ARTIFACT+'/publication_files_v1.json')==
            (published/ARTIFACT/'publication_files_v1.json').read_bytes(),'PINNED_PUBLIC_INVENTORY')
    for row in inventory['files']:
        f=published/relative_path(row['path'])
        require(f.stat().st_size==row['bytes'] and sha(f)==row['sha256'],'PUBLIC_EVIDENCE_BYTES')
    raw_inventory=read(published/ARTIFACT/'original_execution_file_hashes_v1.json')['files']
    for row in raw_inventory:
        f=e/relative_path(row['logical_path'])
        require(f.stat().st_size==row['bytes'] and sha(f)==row['sha256'],'ORIGINAL_RUN02_BYTES')
    binding=read(e/'execution_binding_PRIVATE.json');config=read(p/'config_UNAPPROVED.json')
    public_binding=read(published/ARTIFACT/'execution_binding_v1.json')
    require(sha(p/'config_UNAPPROVED.json')==binding['config_sha256']==public_binding['config_sha256'] and
            sha(p/'grant_APPROVED_ONCE.json')==binding['grant_sha256']==public_binding['grant_sha256'],
            'ORIGINAL_CONFIG_GRANT_BYTES')
    require(config['SOURCE']==SOURCE and config['provider_source']==PROVIDER and
            config['expected_profile_digest']==PROFILE,'PINNED_EXECUTION_PROFILE')
    require(len(config['source_hashes'])==15 and len(config['provider_hashes'])==265 and
            len(config['runtime_file_hashes'])==34273 and len(config['files'])==8,'PINNED_INVENTORY_COUNTS')
    require(digest(config['provider_hashes'])==PROVIDER_MAP and digest(config['runtime_file_hashes'])==RUNTIME_MAP,
            'PINNED_RUNTIME_PROVIDER_INVENTORY')
    for root,commit,hashes in [(execution,SOURCE,config['source_hashes']),
                               (Path(config['provider_root']),PROVIDER,config['provider_hashes'])]:
        for name,expected in hashes.items():
            require(sha(root/relative_path(name))==expected and
                    hashlib.sha256(git_bytes(root,'show',commit+':'+name)).hexdigest()==expected,
                    'ORIGINAL_SOURCE_OR_PROVIDER_CHANGED')
    for name,expected in config['runtime_file_hashes'].items():require(sha(name)==expected,'RUNTIME_CHANGED')
    for row in config['files']:
        require(Path(row['path']).stat().st_size==row['bytes'] and sha(row['path'])==row['sha256'],'INPUT_CHANGED')
    for row in config['protected_files']:
        require(Path(row['path']).stat().st_size==row['bytes'] and sha(row['path'])==row['sha256'],'PROTECTED_FILE_CHANGED')
    for row in config['protected_worktrees']:
        require(git_bytes(row['path'],'rev-parse','HEAD').decode().strip()==row['HEAD'] and
                hashlib.sha256(git_bytes(row['path'],'status','--porcelain=v1','-z')).hexdigest()==row['status_sha256'],
                'PROTECTED_WORKTREE_CHANGED')
    raw_plan=git_bytes(execution,'show',SOURCE+':'+PLAN)
    require(hashlib.sha256(raw_plan).hexdigest()==PLAN_SHA,'PINNED_ORIGINAL_PLAN')
    plan=json.loads(raw_plan)
    bundle=dict(driver=read(o/'terminal.json'),launcher=read(p/'control01/launcher_terminal.json'),
        admission=read(p/'launcher_admission01/terminal.json'),native=read(e/'launcher_native_exit_PRIVATE.json'),
        final_gate_exists=(p/'control01/acquisition_final_gate.json').exists(),
        train=read(o/'train_measurements.json'),holdout=read(o/'holdout_measurements.json'),
        controls=[read(o/(case['id']+'_control_preparation.json')) for case in plan['cases']],
        waves=[read(o/f'wave_{i}/parallel_receipt.json') for i in (1,2)],
        accepted=[read(o/f'wave_{i}_accepted.json') for i in (1,2)])
    result=validate_records(plan,bundle)
    for wave in bundle['waves']:
        for worker in wave['workers']:
            original=Path(worker['directory'])/'task_result.json'
            require(original.resolve().is_relative_to(o.resolve()),'WORKER_RESULT_LOCATION')
            saved=next(r for r in bundle['train']['axes']+bundle['holdout']['axes']
                       if r['case_id']+'_'+r['axis']==worker['task_id'])
            require(sha(original)==saved['result_sha256'],'RAW_TASK_RESULT_SHA')
            require(all(saved[k]==v for k,v in read(original).items()),'RAW_TASK_RESULT_FIELDS')
            if 'result_sha256' in worker:require(worker['result_sha256']==sha(original),'POOL_RESULT_SHA')
    stderr=(p/'control01/stderr.log').read_text()
    require('r.verify_files(config);r.protected(config)' in stderr and
            "['git','status','--porcelain=v1','-z']" in stderr,'FINAL_AUDIT_TRACEBACK')
    require(all(read(p/'launcher_admission01'/name)['status']=='PASS'
                for name in ('CPU_initial.json','CPU_driver_refresh.json')),'CPU_ADMISSION_RECEIPTS')
    # Recheck private originals after the audit itself. No new files are written here.
    for row in raw_inventory:
        require(sha(e/relative_path(row['logical_path']))==row['sha256'],'AUDIT_CHANGED_ORIGINAL')
    result.update(execution_SOURCE=SOURCE,saved_result_commit=RESULT_COMMIT,source_files_verified=15,
        provider_files_verified=265,runtime_files_verified=34273,input_files_verified=8,
        original_run02_files_verified=len(raw_inventory),public_evidence_files_verified=len(inventory['files']),
        original_protected_files_verified=len(config['protected_files']),
        original_worktrees_verified=len(config['protected_worktrees']),
        SOURCE_input_runtime_preparation_compiler_binding=True,
        original_config_grant_raw_unchanged=True,final_preservation_audit_passed_posthoc=True,
        posthoc_audit_retroactively_closes_contract=False,Git_cause_of_original_violation_proven=False,
        administrative_git_options=list(GIT_OPTIONS),subagents=0,independent_review=False)
    return result
