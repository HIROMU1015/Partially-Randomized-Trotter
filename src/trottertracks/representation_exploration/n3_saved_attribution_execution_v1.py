"""Identity-only gates for saved-data posthoc; no scientific producers."""
import hashlib
import json
import platform
from pathlib import Path
import subprocess

from .n3_saved_attribution_v1 import digest

SCOPE='n3_saved_attribution_v1_frozen_six_saved_inputs_posthoc'
BRANCH='n3-saved-attribution-v1-execution-20261013'
OUTPUT='artifacts/n3_saved_attribution_v1/2026-10-13/evaluation1'
PREPARATION='artifacts/n3_saved_attribution_v1/2026-10-13/preparation'
PLAN='docs/research/n3_saved_attribution_v1_execution_plan_20261013.json'
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','RAYON_NUM_THREADS')
RESOURCES={'workers':2,'threads_per_worker':1,'worker_AS_bytes':1024**3,
           'aggregate_sampled_RSS_bytes':6*1024**3,'minimum_available_bytes':8*1024**3,
           'input_total_bytes':64*1024**2,'input_file_bytes':8*1024**2,
           'output_total_bytes':32*1024**2,'output_file_bytes':4*1024**2,
           'log_bytes':4*1024**2,'failure_bytes':65536,'rational_bits':4096,
           'maximum_candidates':1000,'selected_proof_rechecks':3,
           'wall_time_cap_seconds':None,'cpu_time_cap_seconds':None}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def confined(root,relative):
    root=Path(root).resolve();p=root/relative
    if Path(relative).is_absolute() or not p.resolve().is_relative_to(root):raise ValueError('PATH_SCOPE')
    return p


def preflight(root,plan):
    root=Path(root)
    if plan['scope']!=SCOPE or plan['resources']!=RESOURCES or plan['python']!=platform.python_version():
        raise ValueError('FROZEN_POSTHOC_SCOPE')
    if plan['execution_branch']!=BRANCH or plan['output_path']!=OUTPUT or plan['attempts']!=1 or plan['retries']!=0:
        raise ValueError('POSTHOC_ONE_SHOT_CONTRACT')
    for p,h in plan['source_sha256'].items():
        if sha(confined(root,p))!=h:raise ValueError('SOURCE_BYTES')
        if (root/'.git').exists():
            b=subprocess.check_output(['git','show',plan['source_commit']+':'+p],cwd=root)
            if hashlib.sha256(b).hexdigest()!=h:raise ValueError('SOURCE_COMMIT_BINDING')
    if digest(plan['source_sha256'])!=plan['source_fingerprint']:raise ValueError('SOURCE_FINGERPRINT')
    regpath=confined(root,PREPARATION+'/saved_input_registry.json')
    if sha(regpath)!=plan['registry_sha256']:raise ValueError('REGISTRY_BINDING')
    registry=json.loads(regpath.read_bytes());total=0
    for p,h in registry['files'].items():
        path=confined(root,p);size=path.stat().st_size;total+=size
        if size>RESOURCES['input_file_bytes'] or total>RESOURCES['input_total_bytes']:raise ArithmeticError('INPUT_BYTES')
        if sha(path)!=h:raise ValueError('SAVED_INPUT_BYTES')
    if len(registry['case_ids'])!=6 or registry['case_ids']!=plan['case_ids']:raise ValueError('SAVED_SIX_INPUT_SCOPE')
    for p,h in plan['preparation_sha256'].items():
        if sha(confined(root,p))!=h:raise ValueError('PREPARATION_BINDING')
    return {'status':'SOURCE_SAVED_INPUTS_PLAN_IDENTITY_PASS_NO_NEW_NUMERIC_EVALUATION',
            'saved_input_files':len(registry['files']),'new_numeric_posthoc_evaluations':0,
            'new_PSD_certification_calls':0,'new_producer_calls':0,'new_solver_calls':0}


def authorization(auth,plan,plan_sha):
    keys={'scope','posthoc_execution_authorized','source_commit','source_fingerprint','plan_sha256',
          'registry_sha256','execution_branch','output_path','attempts','retries','mandatory_stop'}
    if set(auth)!=keys or auth['posthoc_execution_authorized'] is not True or auth['scope']!=SCOPE:
        raise ValueError('NEW_EXPLICIT_POSTHOC_AUTHORIZATION_REQUIRED')
    for k in ('source_commit','source_fingerprint','registry_sha256'):
        if auth[k]!=plan[k]:raise ValueError('AUTHORIZATION_IDENTITY')
    if auth['plan_sha256']!=plan_sha or auth['execution_branch']!=BRANCH or auth['output_path']!=OUTPUT:
        raise ValueError('AUTHORIZATION_SCOPE')
    if (type(auth['attempts']) is not int or auth['attempts']!=1 or type(auth['retries']) is not int
            or auth['retries']!=0 or auth['mandatory_stop'] is not True):raise ValueError('ONE_SHOT_RETRY_ZERO_STOP')
    return True


class Output:
    def __init__(self,path):self.path=Path(path);self.total=0
    def write(self,name,value,stop=False):
        if Path(name).name!=name:raise ValueError('OUTPUT_NAME')
        if stop and name not in ('result.json','MANDATORY_STOP.json'):raise ValueError('STOP_NAME')
        data=json.dumps(value,indent=2,allow_nan=False).encode()+b'\n'
        reserved=RESOURCES['log_bytes']+(0 if stop else RESOURCES['failure_bytes'])
        if len(data)>(RESOURCES['failure_bytes'] if stop else RESOURCES['output_file_bytes']) or self.total+len(data)>RESOURCES['output_total_bytes']-reserved:
            raise ArithmeticError('OUTPUT_BYTE_CAP')
        with (self.path/name).open('xb') as f:f.write(data)
        self.total+=len(data)
