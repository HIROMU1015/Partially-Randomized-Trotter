"""New scope authorization before numerical imports or saved-array decoding."""
from pathlib import Path
from dataclasses import dataclass
import json,os,subprocess
from .qc_onebody_repartition_execution_v1 import digest,write_json,archive_headers,environment
from .qc_onebody_repartition_h_boundary_protocol_v1 import initial_grid

ARTIFACT='artifacts/qc_onebody_repartition_h_boundary_preparation_v1/2026-10-15'
OUTPUT='artifacts/qc_onebody_repartition_h_boundary_v1/2026-10-15/attempt_1'
SCOPE='QC_ONEBODY_REPARTITION_H_BOUNDARY_V1_20261015'

def preflight(root):
    root=Path(root);c=json.loads((root/ARTIFACT/'contract.json').read_text())
    if c['scope_id']!=SCOPE or c['registered_execution_authorized'] is not False:raise ValueError('CONTRACT_SCOPE')
    for path,x in c['fixed_files'].items():
        if digest(root/path)!=x['SHA256'] or (root/path).stat().st_size!=x['bytes']:raise ValueError('FIXED_IDENTITY_'+path)
    freeze=json.loads((root/ARTIFACT/'source_manifest.json').read_text())
    if freeze['scope_id']!=SCOPE or freeze['contract_SHA256']!=digest(root/ARTIFACT/'contract.json'):raise ValueError('FREEZE_SCOPE')
    for path,x in freeze['files'].items():
        if digest(root/path)!=x['SHA256'] or (root/path).stat().st_size!=x['bytes']:raise ValueError('SOURCE_CHANGED_'+path)
    if environment()!=freeze['environment']:raise ValueError('ENVIRONMENT_CHANGED')
    if any(os.environ.get(n)!='1' for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS')):raise ValueError('THREAD_SETTINGS')
    headers=archive_headers(root/c['saved_arrays'])
    if headers['state']['shape']!=[36] or headers['H']['shape']!=[36,36]:raise ValueError('SAVED_ARRAY_LAYOUT')
    old=json.loads((root/c['saved_raw']).read_text());seal=json.loads((root/c['saved_selector_seal']).read_text())
    if old['Lambda_common']!=c['schedule']['Lambda_common'] or old['h_grid'][0]!=c['schedule']['h_anchor']:raise ValueError('ANCHOR_METADATA')
    if seal['E0_accessed_by_generator'] or seal['finite_step_truth_accessed_by_generator']:raise ValueError('OLD_SELECTOR_TRUTH_LEAK')
    if old['result']['classification']!='NO_UNIFORM_REGISTERED_COMPONENT_GAIN':raise ValueError('PARENT_RESULT')
    grid,K=initial_grid(c['schedule']['h_anchor'],c['schedule']['Lambda_common'],c['schedule'])
    return {'status':'HEADER_SOURCE_IDENTITY_ONLY_PREFLIGHT_PASS','scope_id':SCOPE,'initial_h_points':len(grid),
            'maximum_h_points':c['schedule']['total_h_cap'],'phase_derivative_bound':K,
            'registered_arrays_deserialized':0,'new_registered_eigensolver_RDM_PF_compile_calls':0,
            'registered_execution_authorized':False,'mandatory_stop':True}

@dataclass(frozen=True)
class Context:
    root:Path
    output:Path
    contract:dict
    authorization_SHA256:str

def authorize(root,path):
    root=Path(root)
    if path is None or not Path(path).is_file():raise ValueError('NEW_EXPLICIT_EXECUTION_APPROVAL_REQUIRED')
    a=json.loads(Path(path).read_text())
    if a.get('approved') is not True or a.get('status')!='ISSUED_USER_EXPLICIT_EXECUTION':raise ValueError('UNISSUED_TARGET')
    if a.get('scope_id')!=SCOPE or a.get('attempts')!=1 or a.get('retry')!=0:raise ValueError('NEW_SCOPE_ONE_ATTEMPT_REQUIRED')
    preflight(root)
    if a.get('source_manifest_SHA256')!=digest(root/ARTIFACT/'source_manifest.json') or a.get('contract_SHA256')!=digest(root/ARTIFACT/'contract.json'):raise ValueError('APPROVAL_FIXED_SOURCE')
    if a.get('issuer')!='USER_EXPLICIT_SESSION_INSTRUCTION' or not a.get('user_instruction_verbatim'):raise ValueError('USER_INSTRUCTION_REQUIRED')
    c=json.loads((root/ARTIFACT/'contract.json').read_text())
    if a.get('fixed_files')!=c['fixed_files'] or a.get('limits')!=c['limits']:raise ValueError('APPROVAL_INPUT_OR_CAPS')
    if subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=all'],cwd=root).strip():raise ValueError('CLEAN_ISOLATED_WORKTREE_REQUIRED')
    output=root/OUTPUT;output.mkdir(parents=True,exist_ok=False)
    ah=digest(path)
    write_json(output/'attempt_consumed.json',{'scope_id':SCOPE,'attempt':1,'retry':0,'authorization_SHA256':ah,'science_started_at_gate':False,'mandatory_stop':True})
    return Context(root,output,c,ah)
