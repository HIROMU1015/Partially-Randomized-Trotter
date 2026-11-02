"""Standard-library launch barrier. Read-only preflight never imports PySCF."""
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import subprocess
import sys

ARTIFACT = 'artifacts/qc_mixed_access_source_v1/2026-10-13'
CONTRACT = 'artifacts/qc_mixed_access_feasibility/2026-10-13/feasibility_contract.json'
RUN_OUTPUT = 'artifacts/qc_mixed_access_feasibility_run_v1/2026-10-13/attempt_1'
PINNED = {'numpy':'1.26.4','scipy':'1.14.1','pyscf':'2.8.0','h5py':'3.11.0','setuptools':'75.8.0'}
THREAD_VARS = ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS')


class LaunchError(ValueError):
    pass


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_sources(root,manifest):
    root=Path(root).resolve()
    for name,expected in manifest['files'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or digest(path)!=expected:
            raise LaunchError('SOURCE_HASH:'+name)
    return True


def preflight(root):
    root=Path(root).resolve()
    if sys.version_info[:3]!=(3,12,3) or platform.python_implementation()!='CPython' or platform.machine()!='x86_64' or platform.system()!='Linux':
        raise LaunchError('FROZEN_INTERPRETER_OR_PLATFORM')
    manifest=json.loads((root/ARTIFACT/'source_manifest.json').read_text())
    verify_sources(root,manifest)
    try:
        versions={name:importlib.metadata.version(name) for name in PINNED}
    except importlib.metadata.PackageNotFoundError as error:
        raise LaunchError('DEPENDENCY_MISSING:'+str(error)) from error
    if versions!=PINNED: raise LaunchError('DEPENDENCY_VERSION')
    if digest(root/CONTRACT)!=manifest['contract_SHA256']: raise LaunchError('CONTRACT_HASH')
    cpus=len(os.sched_getaffinity(0))
    memory={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:','MemAvailable:'))}
    if cpus<2 or memory['MemAvailable']<2**32: raise LaunchError('HOST_RESOURCES')
    if (root/RUN_OUTPUT).exists(): raise LaunchError('ATTEMPT_ALREADY_CONSUMED')
    return {'status':'TECHNICAL_PREFLIGHT_PASS_SCIENCE_NOT_AUTHORIZED','versions':versions,
            'CPUs_affinity':cpus,'memory':memory,'source_manifest_SHA256':digest(root/ARTIFACT/'source_manifest.json'),
            'contract_SHA256':digest(root/CONTRACT),'molecular_jobs':0,'launch_allowed':False,
            'attempt_already_exists':(root/RUN_OUTPUT).exists()}


@dataclass(frozen=True)
class AuthorizedContext:
    root: Path
    output: Path
    contract: dict
    receipt: dict


def authorize(root,authorization):
    """Reject a missing/draft/mismatched approval before creating any output."""
    root=Path(root).resolve()
    if authorization is None: raise LaunchError('EXPLICIT_SCIENCE_APPROVAL_REQUIRED')
    approval=json.loads(Path(authorization).read_text())
    if approval.get('type')!='USER_EXPLICIT_SCIENCE_APPROVAL' or not approval.get('user_approval_reference'):
        raise LaunchError('DRAFT_OR_MISSING_USER_APPROVAL')
    if approval.get('scope_id')!='QC_MIXED_ACCESS_FEASIBILITY_V1_20261013' or approval.get('output')!=RUN_OUTPUT:
        raise LaunchError('SCOPE_OR_OUTPUT')
    if approval.get('attempts')!=1 or approval.get('retries')!=0:
        raise LaunchError('ONE_SHOT_CONTRACT')
    report=preflight(root)
    for key in ('contract_SHA256','source_manifest_SHA256'):
        if approval.get(key)!=report[key]: raise LaunchError('APPROVAL_HASH:'+key)
    if approval.get('dependency_lock_SHA256')!=digest(root/ARTIFACT/'requirements.lock'):
        raise LaunchError('APPROVAL_ENVIRONMENT_HASH')
    commit=approval.get('source_payload_commit','')
    if len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit): raise LaunchError('IMMUTABLE_SOURCE_COMMIT')
    manifest=json.loads((root/ARTIFACT/'source_manifest.json').read_text())
    for name,expected in manifest['files'].items():
        result=subprocess.run(['git','show',commit+':'+name],cwd=root,capture_output=True,check=True)
        if hashlib.sha256(result.stdout).hexdigest()!=expected: raise LaunchError('COMMIT_SOURCE_HASH:'+name)
    remote=subprocess.run(['git','remote','get-url','origin'],cwd=root,capture_output=True,text=True,check=True).stdout.strip()
    if remote not in ('git@github.com:HIROMU1015/Partially-Randomized-Trotter.git','https://github.com/HIROMU1015/Partially-Randomized-Trotter.git'):
        raise LaunchError('REPOSITORY_IDENTITY')
    if report['attempt_already_exists']: raise LaunchError('ATTEMPT_ALREADY_CONSUMED')
    for name in THREAD_VARS:
        os.environ[name]='1'
    receipt={**approval,'status':'AUTHORIZED_ATTEMPT_CONSUMED_BEFORE_INPUT_JOB','preflight':report,
             'mandatory_stop':True}
    output=root/RUN_OUTPUT
    output.mkdir(parents=True,exist_ok=False)
    (output/'launch_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return AuthorizedContext(root,output,json.loads((root/CONTRACT).read_text()),receipt)
