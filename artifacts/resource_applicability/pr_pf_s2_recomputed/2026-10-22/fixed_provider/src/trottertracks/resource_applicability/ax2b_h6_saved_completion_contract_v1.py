"""Stdlib-only saved DF input-completion plan, parent seal and fresh grant gate."""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import subprocess
from .ax2a_preparation import digest
from .ax2b_h6_df_diagnostic_contract_v1 import safe_path, file_hash, verify_input
from .ax2b_h6_input_generation_contract_v1 import environment, plan as old_plan
from .ax2b_h6_input_generation_audit_v1 import read_json, npz_bytes

DIAGNOSTIC_RESULT = '8a3189e69dd461724fa9e2c01ea08562c1c35f8d'
DIAGNOSTIC_SOURCE = 'ff24de4bc410234472a416186b773fc7875ae373'
RAW = 'artifacts/resource_applicability/track_a_ax2b_h6_df_diagnostic/2026-10-10/launch_v1/'
EVIDENCE = 'artifacts/resource_applicability/track_a_ax2b_h6_df_diagnostic_execution_v1/2026-10-10/'
FREEZE = 'artifacts/resource_applicability/track_a_ax2b_h6_df_diagnostic_preparation/2026-10-10/source_freeze_v1.json'
RUNNER = 'scripts/resource_applicability/run_track_a_h6_saved_completion_v1.py'
AUDITOR = 'scripts/resource_applicability/audit_track_a_h6_saved_completion_v1.py'
NAMESPACE = 'artifacts/resource_applicability/track_a_h6_saved_df_completion/2026-10-10/'
PHASES = ('saved_input','df_acceptance','state_snapshot')
KIND = 'H6_SAVED_DF_INPUT_COMPLETION'


def plan():
    return {'schema':'track_a_h6_saved_df_completion_plan_v1','kind':KIND,
        'target':old_plan()['target'],'state_policy':old_plan()['state_policy'],
        'policy_name':'WEIGHTED_HERMITIAN_PROJECTION_FROM_SAVED_RAW_V1',
        'projection_budget_hartree':1e-10,'decision_limit_hartree':9.9e-11,
        'actual_rank_required':19,'df_tol':1e-8,'coefficient_cutoff':0.,
        'diagnostic_result_commit':DIAGNOSTIC_RESULT,'diagnostic_source_commit':DIAGNOSTIC_SOURCE,
        'caps':{'phase_wall_seconds':dict(zip(PHASES,(60,120,900))),
            'total_wall_seconds':1080,'address_space_bytes':8*2**30,
            'output_bytes':32*2**20,'snapshot_expanded_bytes':16*2**20,
            'input_expanded_bytes':16*2**20,'raw_expanded_bytes':4*2**20,
            'log_bytes':65536,'diagnostics':256,'progress_records':256,
            'acceptance':1,'state_solver':1,'solver_matvec':10000,
            'integral_build':0,'df_decomposition':0,'signal':0,'trajectory':0,'occurrence':0,'compile':0},
        'retry':False,'resume':False,'gpu':False,'H6_status':'H6_NOT_AUTHORIZED',
        'contract_status':'DRAFT_NOT_AUTHORIZATION','mandatory_stop':True,'next_stage_authorized':False}


def preparation():
    return {'schema':'track_a_h6_saved_df_completion_preparation_v1','kind':KIND,'plan':plan(),
        'status':'H6_SAVED_DF_COMPLETION_NOT_AUTHORIZED','source_commit':None,'source_hashes':None,
        'input_identity':None,'environment':None,'assigned_resources':None,'exclusive_output':None,
        'execution_plan_sealed':False,'science_authorized':False,'launch_allowed':False,
        'H6_status':'H6_NOT_AUTHORIZED','contract_status':'DRAFT_NOT_AUTHORIZATION',
        'mandatory_stop':True,'next_stage_authorized':False}


def source_paths(root):
    root=Path(root)
    return sorted({str(p.relative_to(root)) for d in ('src/trotterlib','src/trottertracks')
        for p in (root/d).rglob('*.py')} | {RUNNER,AUDITOR})


def verify_sources(root,commit,hashes):
    if not isinstance(commit,str) or len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('COMPLETION_SOURCE_COMMIT')
    if not isinstance(hashes,dict) or set(hashes)!=set(source_paths(root)):
        raise ValueError('COMPLETION_SOURCE_CLOSURE')
    subprocess.run(['git','merge-base','--is-ancestor',commit,'HEAD'],cwd=root,check=True,stdout=subprocess.DEVNULL)
    for p,sha in hashes.items():
        if file_hash(safe_path(root,p))!=sha or hashlib.sha256(subprocess.check_output(['git','show',commit+':'+p],cwd=root)).hexdigest()!=sha:
            raise ValueError('COMPLETION_SOURCE_CHANGED:'+p)


def verify_parents(root):
    """Git blobs, hashes and NPY headers only. Never evaluates real arrays."""
    root=Path(root);integral=verify_input(root)
    inv=read_json(root/EVIDENCE/'execution_evidence_inventory_v1.json')
    if inv['source_commit']!=DIAGNOSTIC_SOURCE or inv['H6_input_accepted'] is not False or inv['one_shot_authorization_consumed'] is not True:
        raise ValueError('COMPLETION_DIAGNOSTIC_PARENT')
    for p,sha in inv['raw_file_hashes'].items():
        if file_hash(safe_path(root,p))!=sha or hashlib.sha256(subprocess.check_output(['git','show',DIAGNOSTIC_RESULT+':'+p],cwd=root)).hexdigest()!=sha:
            raise ValueError('COMPLETION_RAW_PARENT_CHANGED:'+p)
    freeze=read_json(root/FREEZE)
    for group in ('science_source_hashes','validation_source_hashes'):
        for p,sha in freeze[group].items():
            if file_hash(safe_path(root,p))!=sha or hashlib.sha256(subprocess.check_output(['git','show',DIAGNOSTIC_SOURCE+':'+p],cwd=root)).hexdigest()!=sha:
                raise ValueError('COMPLETION_PARENT_SOURCE_CHANGED:'+p)
    raw=read_json(root/RAW/'raw_decomposition_receipt.json')
    hyp=read_json(root/RAW/'hypothetical_receipt.json')
    members=npz_bytes(root/RAW/'raw_decomposition.npz',raw,cap=4*2**20)
    expected={'lambdas_raw':((19,),'<f8'),'g_matrices_raw':((19,12,12),'<c16'),
              'one_body_correction_raw':((12,12),'<c16'),'truncation_value_raw':((),'<f8')}
    if set(members)!=set(expected) or any(members[k][0]!={'shape':shape,'descr':dtype,'fortran_order':False} for k,(shape,dtype) in expected.items()):
        raise ValueError('COMPLETION_RAW_LAYOUT')
    npz_bytes(root/RAW/'hypothetical_hermitization.npz',hyp,cap=4*2**20)
    return {'integrals':integral,'diagnostic_result_commit':DIAGNOSTIC_RESULT,
        'diagnostic_source_commit':DIAGNOSTIC_SOURCE,'raw_path':RAW+'raw_decomposition.npz',
        'raw_sha256':raw['sha256'],'raw_receipt_sha256':file_hash(root/RAW/'raw_decomposition_receipt.json'),
        'hypothetical_sha256':hyp['sha256'],'hypothetical_receipt_sha256':file_hash(root/RAW/'hypothetical_receipt.json'),
        'summary_sha256':file_hash(root/RAW/'diagnostic_summary.json'),
        'diagnostic_inventory_sha256':file_hash(root/EVIDENCE/'execution_evidence_inventory_v1.json'),
        'parent_raw_count':len(inv['raw_file_hashes']),'raw_arrays':raw['arrays'],
        'hypothetical_arrays':hyp['arrays'],'historical_failed_raw_identity_claim':False}


def validate_launch(root,manifest,grant,output,*,requested=False,worker=False):
    if requested is not True or not isinstance(grant,dict) or grant.get('approved_by_user') is not True:
        raise ValueError('NEW_SAVED_DF_COMPLETION_GRANT_REQUIRED')
    if grant.get('schema')!='track_a_h6_saved_df_completion_authorization_v1':
        raise ValueError('COMPLETION_GRANT_SCHEMA')
    expected=preparation()
    for k in ('schema','kind','status','science_authorized','launch_allowed','H6_status','contract_status','mandatory_stop','next_stage_authorized'):
        if type(manifest.get(k)) is not type(expected[k]) or manifest[k]!=expected[k]:
            raise ValueError('COMPLETION_MANIFEST_FLAG:'+k)
    if manifest.get('execution_plan_sealed') is not True or digest(manifest.get('plan'))!=digest(plan()):
        raise ValueError('COMPLETION_FIXED_PLAN')
    if (grant.get('kind')!=KIND or grant.get('manifest_digest')!=digest(manifest)
            or grant.get('retry') is not False or grant.get('resume') is not False):
        raise ValueError('COMPLETION_GRANT_BINDING')
    cpu=grant.get('assigned_cpu')
    if type(cpu) is not int or cpu not in os.sched_getaffinity(0) or digest(manifest.get('assigned_resources'))!=digest({'assigned_cpu':cpu,'science_workers':1,'blas_threads':1}):
        raise ValueError('COMPLETION_CPU')
    intended=manifest.get('exclusive_output') or {};output=Path(output).resolve();name=intended.get('repository_path','')
    if (not name.startswith(NAMESPACE) or safe_path(root,name)!=output or intended.get('absolute_path')!=str(output)
            or grant.get('exclusive_output')!=str(output) or (not worker and output.exists())):
        raise ValueError('COMPLETION_EXCLUSIVE_OUTPUT')
    if worker:
        if read_json(output/'launch_binding.json')!={'manifest_digest':digest(manifest),'authorization_digest':digest(grant)} or (output/'worker_claim.json').exists():
            raise ValueError('COMPLETION_ONE_SHOT_BINDING')
    verify_sources(root,manifest.get('source_commit'),manifest.get('source_hashes'))
    if manifest.get('input_identity')!=verify_parents(root):
        raise ValueError('COMPLETION_PARENT_BINDING')
    if manifest.get('environment')!=environment():
        raise ValueError('COMPLETION_ENVIRONMENT')
    return cpu
