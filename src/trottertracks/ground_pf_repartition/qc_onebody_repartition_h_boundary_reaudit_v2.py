"""Stdlib-only gate for a NEW one-shot saved-data audit; no collection entry."""
from dataclasses import dataclass
from pathlib import Path
import hashlib,json,os,subprocess
from .qc_onebody_repartition_execution_v1 import environment,archive_headers,write_json

ARTIFACT='artifacts/qc_onebody_repartition_h_boundary_checker_v2_preparation/2026-10-15'
OUTPUT='artifacts/qc_onebody_repartition_h_boundary_reaudit_v2/2026-10-15/attempt_1'
SCOPE='QC_ONEBODY_REPARTITION_H_BOUNDARY_REAUDIT_V2_20261015'
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS')

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def check_files(root,files):
    for path,x in files.items():
        p=root/path
        if not p.is_file() or p.stat().st_size!=x['bytes'] or digest(p)!=x['SHA256']:raise ValueError('IDENTITY_'+path)

def preflight(root):
    root=Path(root);c=json.loads((root/ARTIFACT/'contract.json').read_text())
    if c['scope_id']!=SCOPE or c['registered_execution_authorized'] is not False:raise ValueError('CONTRACT_SCOPE')
    inputs=json.loads((root/ARTIFACT/'input_identity.json').read_text());check_files(root,inputs['files'])
    freeze=json.loads((root/ARTIFACT/'source_manifest.json').read_text())
    if freeze['scope_id']!=SCOPE or freeze['contract_SHA256']!=digest(root/ARTIFACT/'contract.json') or freeze['input_identity_SHA256']!=digest(root/ARTIFACT/'input_identity.json'):raise ValueError('FREEZE_BINDING')
    check_files(root,freeze['files'])
    if environment()!=freeze['environment']:raise ValueError('ENVIRONMENT_CHANGED')
    if any(os.environ.get(k)!='1' for k in THREADS):raise ValueError('SINGLE_THREAD_REQUIRED')
    old=json.loads((root/c['numerical_contract']).read_text())
    if digest(root/c['numerical_contract'])!=c['numerical_contract_SHA256']:raise ValueError('NUMERICAL_CONTRACT_CHANGED')
    if old['scope_id']!=c['original_scope_id']:raise ValueError('ORIGINAL_SCOPE')
    headers=archive_headers(root/old['saved_arrays'])
    if headers['H']['shape']!=[36,36] or headers['state']['shape']!=[36]:raise ValueError('PARENT_HEADERS')
    d=root/c['saved_attempt'];marker=json.loads((d/'attempt_consumed.json').read_text());failed=json.loads((d/'independent_audit.json').read_text())
    if marker['attempt']!=1 or marker['retry']!=0 or failed['reason']!='NUMERICAL_ALLOWANCE':raise ValueError('ORIGINAL_ATTEMPT_STATUS')
    if (root/OUTPUT).exists():raise ValueError('NEW_AUDIT_ALREADY_CONSUMED_NO_RETRY')
    return {'status':'SOURCE_INPUT_BYTE_HEADER_ONLY_PREFLIGHT_PASS','scope_id':SCOPE,
            'input_files_verified':len(inputs['files']),'source_files_verified':len(freeze['files']),
            'source_manifest_SHA256':digest(root/ARTIFACT/'source_manifest.json'),'registered_arrays_deserialized':0,
            'new_registered_numerical_diagnostics':0,'registered_execution_authorized':False,'mandatory_stop':True}

@dataclass(frozen=True)
class Context:
    root:Path
    output:Path
    saved_attempt:Path
    numerical_contract:dict
    contract:dict
    authorization_SHA256:str

def authorize(root,path):
    root=Path(root)
    if path is None or not Path(path).is_file():raise ValueError('NEW_EXPLICIT_REAUDIT_APPROVAL_REQUIRED')
    a=json.loads(Path(path).read_text())
    if a.get('approved') is not True or a.get('status')!='ISSUED_USER_EXPLICIT_EXECUTION':raise ValueError('UNISSUED_TARGET')
    if a.get('scope_id')!=SCOPE or a.get('attempts')!=1 or a.get('retry')!=0:raise ValueError('NEW_REAUDIT_SCOPE_REQUIRED')
    if a.get('issuer')!='USER_EXPLICIT_SESSION_INSTRUCTION' or not a.get('user_instruction_verbatim'):raise ValueError('USER_INSTRUCTION_REQUIRED')
    c=json.loads((root/ARTIFACT/'contract.json').read_text())
    for field,file in [('contract_SHA256','contract.json'),('source_manifest_SHA256','source_manifest.json'),('input_identity_SHA256','input_identity.json')]:
        if a.get(field)!=digest(root/ARTIFACT/file):raise ValueError('APPROVAL_'+field)
    if a.get('limits')!=c['limits']:raise ValueError('APPROVAL_CAPS')
    preflight(root)
    if subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=all'],cwd=root).strip():raise ValueError('CLEAN_ISOLATED_WORKTREE_REQUIRED')
    freeze=json.loads((root/ARTIFACT/'source_manifest.json').read_text())
    subprocess.run(['git','merge-base','--is-ancestor',freeze['source_commit'],'HEAD'],cwd=root,check=True)
    old=json.loads((root/c['numerical_contract']).read_text())
    # Only execution caps tighten; all original scientific thresholds stay exact.
    old['limits']['independent_PF_recompositions']=c['limits']['independent_PF_recompositions']
    old['limits']['independent_expm_calls']=c['limits']['independent_expm_calls']
    old['limits']['saved_cell_array_loads']=c['limits']['saved_cell_array_loads']
    output=root/OUTPUT;output.mkdir(parents=True,exist_ok=False)
    ah=digest(path)
    write_json(output/'attempt_consumed.json',{'scope_id':SCOPE,'attempt':1,'retry':0,'authorization_SHA256':ah,
        'source_manifest_SHA256':digest(root/ARTIFACT/'source_manifest.json'),'input_identity_SHA256':digest(root/ARTIFACT/'input_identity.json'),
        'old_authorization_reused':False,'science_started_at_gate':False,'mandatory_stop':True})
    return Context(root,output,root/c['saved_attempt'],old,c,ah)
