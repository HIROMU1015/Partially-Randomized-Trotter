"""Stdlib-only seal/gate for a separately authorized saved-integral DF diagnostic."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from .ax2a_preparation import digest
from .ax2b_h6_input_generation_audit_v1 import npz_bytes, read_json

OLD_RESULT = 'df3b1f694ceb72a198ab6e3e89706b239e56e1da'
OLD_SOURCE = '67312f3195aede26e8ba4f5727d89c236772f82e'
OLD_RUN = 'artifacts/resource_applicability/track_a_ax2b_h6_input_generation/2026-10-10/launch_v1/'
OLD_INVENTORY = 'artifacts/resource_applicability/track_a_ax2b_h6_input_generation_execution_v1/2026-10-10/stop_evidence_inventory_v1.json'
OLD_FREEZE = 'artifacts/resource_applicability/track_a_ax2b_h6_input_preparation_v1/2026-10-10/source_freeze_v1.json'
INPUT_SHA = 'edd0a618f86011757cacae481eff44dc637c11a3f64c55b0cbfb7ffbe637e51d'
NAMESPACE = 'artifacts/resource_applicability/track_a_ax2b_h6_df_diagnostic/2026-10-10/'
RUNNER = 'scripts/resource_applicability/run_track_a_h6_df_diagnostic_v1.py'
PHASES = ('saved_input', 'decomposition', 'diagnostics')

def file_hash(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def safe_path(root, name):
    root=Path(root).resolve(); p=(root/name).resolve()
    if not isinstance(name,str) or Path(name).is_absolute() or p==root or not p.is_relative_to(root):
        raise ValueError('DIAGNOSTIC_PATH')
    return p

def plan():
    return {'schema':'track_a_h6_df_diagnostic_plan_v1','kind':'H6_SAVED_INTEGRAL_DF_DIAGNOSTIC',
        'old_result_commit':OLD_RESULT,'old_source_commit':OLD_SOURCE,'input_path':OLD_RUN+'integrals.npz',
        'input_sha256':INPUT_SHA,'model':'linear_H6','geometry_angstrom':1.,'basis':'sto-3g',
        'n_spin_orbitals':12,'n_spatial_orbitals':6,'kwargs':{'truncation_threshold':1e-8},
        'implicit_defaults_not_passed':{'final_rank':None,'spin_basis':True},
        'hermitization_tolerance':1e-10,'coefficient_order':'returned decomposer order, unchanged',
        'coefficient_cutoff':0.,'failed_fragment_index':15,
        'raw_output':'raw_decomposition.npz before any Hermitization/diagnostic rejection',
        'projected_output':'hypothetical_hermitization.npz diagnostic only; never accepted as H6 input',
        'representation':'normal-order raw squared operators; canonical double antisymmetrization; coefficient norms only',
        'caps':{'phase_wall_seconds':dict(zip(PHASES,(60,300,120))),'total_wall_seconds':480,
                'address_space_bytes':8*2**30,'output_bytes':32*2**20,'log_bytes':65536,
                'input_expanded_bytes':16*2**20,'raw_expanded_bytes':4*2**20,
                'decomposition':1,'fragment_diagnostics':36,'progress_records':128,'diagnostics':128,
                'molecular_build':0,'state_solver':0,'signal':0,'trajectory':0,'occurrence':0,'compile':0},
        'retry':False,'resume':False,'gpu':False,'historical_raw_bytes_identity_claim':False,
        'policy_changed':False,'H6_status':'H6_NOT_AUTHORIZED','contract_status':'DRAFT_NOT_AUTHORIZATION',
        'mandatory_stop':True,'next_stage_authorized':False}

def preparation():
    return {'schema':'track_a_h6_df_diagnostic_preparation_v1','kind':plan()['kind'],'plan':plan(),
        'status':'H6_DF_DIAGNOSTIC_NOT_AUTHORIZED','source_commit':None,'source_hashes':None,
        'input_identity':None,'environment':None,'assigned_resources':None,'exclusive_output':None,
        'execution_plan_sealed':False,'science_authorized':False,'launch_allowed':False,
        'H6_status':'H6_NOT_AUTHORIZED','contract_status':'DRAFT_NOT_AUTHORIZATION',
        'mandatory_stop':True,'next_stage_authorized':False}

def environment():
    d=importlib.metadata.distribution('openfermion')
    return {'python':sys.version.split()[0],
        'packages':{n:importlib.metadata.version(n) for n in ('numpy','scipy','openfermion','pyscf','openfermionpyscf')},
        'decomposer_source_sha256':file_hash(d.locate_file('openfermion/circuits/low_rank.py')),
        'decomposer_config_sha256':file_hash(d.locate_file('openfermion/config.py'))}

def source_paths(root):
    root=Path(root)
    return sorted({str(p.relative_to(root)) for d in ('src/trotterlib','src/trottertracks')
                   for p in (root/d).rglob('*.py')}|{RUNNER})

def verify_sources(root, commit, hashes):
    if not isinstance(commit,str) or len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('DIAGNOSTIC_SOURCE_COMMIT')
    if not isinstance(hashes,dict) or set(hashes)!=set(source_paths(root)):
        raise ValueError('DIAGNOSTIC_SOURCE_CLOSURE')
    subprocess.run(['git','merge-base','--is-ancestor',commit,'HEAD'],cwd=root,check=True,stdout=subprocess.DEVNULL)
    for p,h in hashes.items():
        if file_hash(safe_path(root,p))!=h or hashlib.sha256(subprocess.check_output(['git','show',commit+':'+p],cwd=root)).hexdigest()!=h:
            raise ValueError('DIAGNOSTIC_SOURCE_BYTES:'+p)

def verify_input(root):
    """Existing file/header/old identity checks only; no numerical libraries."""
    root=Path(root); inventory=read_json(root/OLD_INVENTORY)
    if inventory['source_commit']!=OLD_SOURCE or inventory['worker_stop_reason']!='ValueError:HERMITIZATION_POLICY:fragment_15':
        raise ValueError('OLD_STOP_IDENTITY')
    for p,h in inventory['raw_file_hashes'].items():
        if file_hash(safe_path(root,p))!=h or hashlib.sha256(subprocess.check_output(['git','show',OLD_RESULT+':'+p],cwd=root)).hexdigest()!=h:
            raise ValueError('OLD_RAW_CHANGED:'+p)
    freeze=read_json(root/OLD_FREEZE)
    for p,h in freeze['science_source_hashes'].items():
        if file_hash(safe_path(root,p))!=h or hashlib.sha256(subprocess.check_output(['git','show',OLD_SOURCE+':'+p],cwd=root)).hexdigest()!=h:
            raise ValueError('OLD_SOURCE_CHANGED:'+p)
    receipt=read_json(root/(OLD_RUN+'integral_receipt.json')); inp=root/(OLD_RUN+'integrals.npz')
    if file_hash(inp)!=INPUT_SHA or receipt['sha256']!=INPUT_SHA:
        raise ValueError('SAVED_INTEGRAL_IDENTITY')
    members=npz_bytes(inp,receipt,cap=16*2**20)
    expected={'constant':((),'<f8'),'one_body':((12,12),'<c16'),'two_body':((12,)*4,'<c16'),
        'spatial_one_body':((6,6),'<c16'),'spatial_two_body':((6,)*4,'<c16'),'canonical_orbitals':((6,6),'<c16')}
    if set(members)!=set(expected) or any(members[k][0]!={'shape':shape,'descr':dtype,'fortran_order':False} for k,(shape,dtype) in expected.items()):
        raise ValueError('SAVED_INTEGRAL_LAYOUT')
    return {'input_path':OLD_RUN+'integrals.npz','input_sha256':INPUT_SHA,
        'receipt_sha256':file_hash(root/(OLD_RUN+'integral_receipt.json')),
        'old_manifest_sha256':file_hash(root/(OLD_RUN+'frozen_preparation.json')),
        'old_worker_terminal_sha256':file_hash(root/(OLD_RUN+'worker_terminal.json')),
        'integral_array_records':receipt['arrays'],
        'old_result_commit':OLD_RESULT,'old_source_commit':OLD_SOURCE,'old_raw_files_verified':len(inventory['raw_file_hashes']),
        'old_science_sources_verified':len(freeze['science_source_hashes']),'npz_headers_and_data_sha_checked':True}

def validate_launch(root, manifest, grant, output, *, requested=False, worker=False):
    if requested is not True or not isinstance(grant,dict) or grant.get('approved_by_user') is not True:
        raise ValueError('NEW_DIAGNOSTIC_GRANT_REQUIRED')
    if grant.get('schema')!='track_a_h6_df_diagnostic_authorization_v1':
        raise ValueError('DIAGNOSTIC_GRANT_SCHEMA')
    expected=preparation()
    for k in ('schema','kind','status','science_authorized','launch_allowed','H6_status','contract_status','mandatory_stop','next_stage_authorized'):
        if type(manifest.get(k)) is not type(expected[k]) or manifest[k]!=expected[k]:
            raise ValueError('DIAGNOSTIC_MANIFEST_FLAG:'+k)
    if manifest.get('execution_plan_sealed') is not True or digest(manifest.get('plan'))!=digest(plan()):
        raise ValueError('DIAGNOSTIC_SEAL_OR_PLAN')
    if (grant.get('kind')!=manifest['kind'] or grant.get('manifest_digest')!=digest(manifest)
            or grant.get('retry') is not False or grant.get('resume') is not False):
        raise ValueError('DIAGNOSTIC_GRANT_BINDING')
    cpu=grant.get('assigned_cpu')
    if type(cpu) is not int or cpu not in os.sched_getaffinity(0) or digest(manifest.get('assigned_resources'))!=digest({'assigned_cpu':cpu,'science_workers':1,'blas_threads':1}):
        raise ValueError('DIAGNOSTIC_CPU')
    intended=manifest.get('exclusive_output') or {};output=Path(output).resolve();name=intended.get('repository_path','')
    if (not name.startswith(NAMESPACE) or safe_path(root,name)!=output or intended.get('absolute_path')!=str(output)
            or grant.get('exclusive_output')!=str(output) or (not worker and output.exists())):
        raise ValueError('DIAGNOSTIC_EXCLUSIVE_OUTPUT')
    if worker:
        if read_json(output/'launch_binding.json')!={'manifest_digest':digest(manifest),'authorization_digest':digest(grant)} or (output/'worker_claim.json').exists():
            raise ValueError('DIAGNOSTIC_ONE_SHOT_BINDING')
    verify_sources(root,manifest.get('source_commit'),manifest.get('source_hashes'))
    if manifest.get('input_identity')!=verify_input(root):raise ValueError('DIAGNOSTIC_INPUT_CHANGED')
    if manifest.get('environment')!=environment():raise ValueError('DIAGNOSTIC_ENVIRONMENT_CHANGED')
    return cpu
