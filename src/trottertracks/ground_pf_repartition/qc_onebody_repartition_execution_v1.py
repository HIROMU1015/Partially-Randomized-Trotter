"""Stdlib-only identity preflight and one-attempt authorization boundary."""
from dataclasses import dataclass
from pathlib import Path
import ast, hashlib, importlib.metadata, json, os, subprocess, sys, zipfile

ARTIFACT = 'artifacts/qc_onebody_repartition_finite_step_preparation_v1/2026-10-15'
SCOPE = 'QC_ONEBODY_REPARTITION_FINITE_STEP_V1_20261015'
OUTPUT = 'artifacts/qc_onebody_repartition_finite_step_pilot_v1/2026-10-15/attempt_1'


class LaunchError(ValueError):
    pass


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def canonical_sha(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def archive_headers(path):
    """Read .npy headers only; do not deserialize any science array payload."""
    out={}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            with archive.open(name) as stream:
                prefix=stream.read(8)
                if prefix[:6]!=b'\x93NUMPY' or prefix[6] not in (1,2):
                    raise LaunchError('NPY_HEADER_VERSION')
                width=2 if prefix[6]==1 else 4
                size=int.from_bytes(stream.read(width),'little')
                if size>65536:
                    raise LaunchError('NPY_HEADER_TOO_LARGE')
                header=ast.literal_eval(stream.read(size).decode('latin1').strip())
                out[name[:-4]]={'shape':list(header['shape']),'dtype':header['descr'],'fortran':header['fortran_order']}
    return out


def environment():
    return {'python':sys.version.split()[0],**{x:importlib.metadata.version(x) for x in ['numpy','scipy']}}


def preflight(root, require_freeze=True):
    root=Path(root); contract_path=root/ARTIFACT/'contract.json'
    contract=json.loads(contract_path.read_text())
    if contract.get('scope_id')!=SCOPE or contract['registered_execution_authorized'] is not False:
        raise LaunchError('CONTRACT_SCOPE_OR_AUTH_FLAG')
    source=root/contract['input']['path']
    if digest(source)!=contract['input']['SHA256']:
        raise LaunchError('INPUT_SHA_MISMATCH')
    headers=archive_headers(source)
    expected={'constant':([], '<f8'),'one_body':([8,8],'<c16'),'lambdas':([12],'<f8'),
              'g_matrices':([12,8,8],'<c16'),'sector_basis_indices':([36],'<i8'),
              'state_vector':([256],'<c16'),'sector_state_vector':([36],'<c16')}
    if set(headers)!=set(expected)|{'metadata_json'}:
        raise LaunchError('INPUT_KEYS')
    for key,(shape,dtype) in expected.items():
        if headers[key]['shape']!=shape or headers[key]['dtype']!=dtype or headers[key]['fortran']:
            raise LaunchError('INPUT_LAYOUT_'+key)
    if headers['metadata_json']['shape']!=[] or not headers['metadata_json']['dtype'].startswith('<U'):
        raise LaunchError('METADATA_LAYOUT')
    freeze_path=root/ARTIFACT/'source_manifest.json'; freeze=None
    if freeze_path.exists():
        freeze=json.loads(freeze_path.read_text())
        if freeze['scope_id']!=SCOPE or freeze['contract_SHA256']!=digest(contract_path):
            raise LaunchError('SOURCE_MANIFEST_BINDING')
        for path,identity in freeze['files'].items():
            if digest(root/path)!=identity['SHA256'] or (root/path).stat().st_size!=identity['bytes']:
                raise LaunchError('SOURCE_CHANGED_'+path)
        if environment()!=freeze['environment']:
            raise LaunchError('ENVIRONMENT_CHANGED')
        for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:
            if os.environ.get(name)!='1':
                raise LaunchError('THREAD_SETTING_'+name)
    elif require_freeze:
        raise LaunchError('SOURCE_NOT_FROZEN')
    return {'status':'IDENTITY_HEADER_ONLY_PREFLIGHT_PASS','scope_id':SCOPE,
            'input_SHA256':contract['input']['SHA256'],'input_headers':headers,
            'source_manifest_SHA256':digest(freeze_path) if freeze else None,
            'science_array_payloads_deserialized':0,'eigensolver_RDM_PF_compile_calls':0,
            'registered_execution_authorized':False,'mandatory_stop':True}


@dataclass(frozen=True)
class AuthorizedContext:
    root: Path
    output: Path
    contract: dict
    authorization_SHA256: str


def authorize(root, authorization_path):
    root=Path(root)
    if authorization_path is None or not Path(authorization_path).is_file():
        raise LaunchError('SEPARATE_EXPLICIT_SCIENCE_AUTHORIZATION_REQUIRED')
    approval=json.loads(Path(authorization_path).read_text())
    if approval.get('status')!='ISSUED_USER_EXPLICIT_EXECUTION' or approval.get('approved') is not True:
        raise LaunchError('UNISSUED_AUTHORIZATION_TARGET')
    if approval.get('scope_id')!=SCOPE or approval.get('attempts')!=1 or approval.get('retry')!=0:
        raise LaunchError('AUTHORIZATION_SCOPE_OR_ATTEMPTS')
    preflight(root)
    if approval.get('contract_SHA256')!=digest(root/ARTIFACT/'contract.json') or approval.get('source_manifest_SHA256')!=digest(root/ARTIFACT/'source_manifest.json'):
        raise LaunchError('AUTHORIZATION_FIXED_IDENTITY_MISMATCH')
    if approval.get('input_SHA256')!=json.loads((root/ARTIFACT/'contract.json').read_text())['input']['SHA256']:
        raise LaunchError('AUTHORIZATION_INPUT_MISMATCH')
    if not approval.get('user_instruction_verbatim') or approval.get('issuer')!='USER_EXPLICIT_SESSION_INSTRUCTION':
        raise LaunchError('USER_EXECUTION_INSTRUCTION_REQUIRED')
    # Refuse dirty executable payload. Receipt/docs may be added in later commits.
    dirty=subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=all'],cwd=root).decode()
    if dirty.strip():
        raise LaunchError('EXECUTION_REQUIRES_CLEAN_ISOLATED_WORKTREE')
    output=root/OUTPUT
    output.mkdir(parents=True,exist_ok=False)  # consumed before science imports or input deserialization
    approval_hash=digest(authorization_path)
    write_json(output/'attempt_consumed.json',{'scope_id':SCOPE,'attempt':1,'retry':0,
               'authorization_SHA256':approval_hash,'science_started':False,'mandatory_stop':True})
    return AuthorizedContext(root,output,json.loads((root/ARTIFACT/'contract.json').read_text()),approval_hash)
