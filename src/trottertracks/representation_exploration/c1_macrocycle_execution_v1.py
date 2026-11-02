"""C1 launch identity and separate explicit approval gate; standard library only."""
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import subprocess
import sys

ARTIFACT='artifacts/qc_c1_macrocycle_preparation_v1/2026-10-13'
OUTPUT='artifacts/qc_c1_macrocycle_pilot_v1/2026-10-13/attempt_1'
SCOPE='QC_C1_FIXED_MACROCYCLE_PILOT_V1_20261013'
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
         'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS')


class LaunchError(ValueError): pass


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path): return json.loads(Path(path).read_text())


def check_approval(approval,expected):
    if not approval or approval.get('type')!='USER_EXPLICIT_SCIENCE_APPROVAL' or not approval.get('user_approval_reference'):
        raise LaunchError('SEPARATE_EXPLICIT_SCIENCE_APPROVAL_REQUIRED')
    if approval.get('scope_id')!=SCOPE or approval.get('attempts')!=1 or approval.get('retries')!=0:
        raise LaunchError('SCOPE_OR_ONE_SHOT')
    if approval.get('output')!=OUTPUT: raise LaunchError('OUTPUT_SCOPE')
    for key,value in expected.items():
        if approval.get(key)!=value: raise LaunchError('APPROVAL_IDENTITY:'+key)


def preflight(root):
    root=Path(root).resolve(); folder=root/ARTIFACT
    manifest=load(folder/'source_manifest.json'); plan=load(folder/'execution_plan.json')
    for name,digest in manifest['files'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha(path)!=digest:
            raise LaunchError('SOURCE_OR_INPUT:'+name)
        raw=subprocess.run(['git','show',plan['source_commit']+':'+name],cwd=root,
                           check=True,capture_output=True).stdout
        if hashlib.sha256(raw).hexdigest()!=digest: raise LaunchError('FIXED_COMMIT:'+name)
    if sys.version_info[:3]!=(3,12,3) or platform.machine()!='x86_64' or platform.system()!='Linux':
        raise LaunchError('INTERPRETER_PLATFORM')
    if importlib.metadata.version('numpy')!='1.26.4': raise LaunchError('NUMPY_VERSION')
    runtime=load(folder/'runtime_lock.json')
    if sha(sys.executable)!=runtime['CPython_executable_SHA256']: raise LaunchError('INTERPRETER_BYTES')
    for name,digest in runtime['numpy_distribution_files'].items():
        path=importlib.metadata.distribution('numpy').locate_file(name)
        if sha(path)!=digest: raise LaunchError('NUMPY_BYTES:'+name)
    if (root/OUTPUT).exists(): raise LaunchError('ATTEMPT_CONSUMED')
    if len(os.sched_getaffinity(0))<2: raise LaunchError('CPU_MINIMUM')
    memory={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
            if line.startswith(('MemTotal:','MemAvailable:'))}
    if memory['MemAvailable']<2**30: raise LaunchError('MEMORY_MINIMUM')
    remote=subprocess.run(['git','remote','get-url','origin'],cwd=root,check=True,
                          text=True,capture_output=True).stdout.strip()
    if remote not in ('git@github.com:HIROMU1015/Partially-Randomized-Trotter.git',
                      'https://github.com/HIROMU1015/Partially-Randomized-Trotter.git'):
        raise LaunchError('REMOTE_IDENTITY')
    identities={'source_commit':plan['source_commit'],
                'source_manifest_SHA256':sha(folder/'source_manifest.json'),
                'execution_plan_SHA256':sha(folder/'execution_plan.json'),
                'runtime_lock_SHA256':sha(folder/'runtime_lock.json')}
    return {'status':'TECHNICAL_PREFLIGHT_ONLY','identities':identities,'memory':memory,
            'launch_allowed':False,'scientific_evaluations':0,'numpy_imported':False}


def authorize(root,path):
    if path is None: raise LaunchError('SEPARATE_EXPLICIT_SCIENCE_APPROVAL_REQUIRED')
    root=Path(root).resolve(); approval=load(path)
    # Reject drafts before source/input processing, numerical imports or output creation.
    check_approval(approval,{})
    report=preflight(root); check_approval(approval,report['identities'])
    for key in THREADS: os.environ[key]='1'
    out=root/OUTPUT; out.mkdir(parents=True,exist_ok=False)
    (out/'launch_receipt.json').write_text(json.dumps({'approval':approval,'preflight':report,
        'attempt_consumed':True,'mandatory_stop':True},indent=2)+'\n')
    return root,out
