"""Gates, resource watchdog and bounded output; no numerical imports.

Memory sampling follows run_n3_occupation_transfer_v1.MemoryWatch. It measures
sampled parent/descendant RSS, not between-sample instantaneous peaks.
"""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import threading
import time

from .n3_width_contract_v2 import CAPS,METHODS,canonical,decode_case,digest

THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','RAYON_NUM_THREADS')
RESOURCES={'workers':2,'threads_per_worker':1,'worker_AS_bytes':3*1024**3,
           'aggregate_sampled_RSS_bytes':6*1024**3,'minimum_available_bytes':8*1024**3,
           'total_output_bytes':128*1024**2,'output_shard_bytes':8*1024**2,'log_bytes':8*1024**2,
           'failure_bytes':65536,'wall_time_cap_seconds':None,'cpu_time_cap_seconds':None}
EXECUTION_BRANCH='n3-width-comparison-v2-execution-20261012'
OUTPUT_PATH='artifacts/n3_width_comparison_v2/2026-10-12/run1'
SCOPE='n3_width_comparison_v2_frozen_six_input_pilot'


def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preflight(root,plan):
    root=Path(root)
    if plan['scope']!=SCOPE or plan['methods']!=list(METHODS) or plan['caps']!=CAPS or plan['resources']!=RESOURCES:
        raise ValueError('FROZEN_COMPARISON_CONTRACT')
    for rel,sha in plan['source_sha256'].items():
        p=root/rel
        if not p.resolve().is_relative_to(root.resolve()) or file_sha(p)!=sha:raise ValueError('FROZEN_SOURCE_BYTES')
    if digest(plan['source_sha256'])!=plan['source_fingerprint']:raise ValueError('SOURCE_FINGERPRINT')
    for name,version in plan['environment_packages'].items():
        if importlib.metadata.version(name)!=version:raise ValueError('FROZEN_ENVIRONMENT:'+name)
    p=root/plan['input_path']
    if not p.resolve().is_relative_to(root.resolve()) or file_sha(p)!=plan['input_sha256']:raise ValueError('INPUT_PACK_SHA')
    pack=json.loads(p.read_bytes())
    if set(pack)!={'schema_version','recipe','cases'} or pack['schema_version']!=2:raise ValueError('INPUT_PACK_SCHEMA')
    if [c['id'] for c in pack['cases']]!=plan['input_ids']:raise ValueError('INPUT_IDS')
    for case in pack['cases']:
        decode_case(case)
        if digest(case)!=plan['case_sha256'][case['id']]:raise ValueError('CASE_SHA')
    return {'status':'IDENTITY_SCHEMA_PREFLIGHT_PASS_NO_SCIENCE','input_count':len(pack['cases']),
            'new_bound_calculations':0,'new_eigensolver_calls':0,'new_solver_calls':0,'new_certificate_searches':0}


def validate_authorization(auth,plan,plan_sha):
    keys={'scope','science_execution_authorized','source_commit','source_fingerprint','plan_sha256',
          'input_sha256','execution_branch','output_path','attempts','retries','mandatory_stop'}
    if set(auth)!=keys or auth['scope']!=SCOPE or auth['science_execution_authorized'] is not True:
        raise ValueError('SEPARATE_EXPLICIT_SCIENCE_AUTHORIZATION_REQUIRED')
    for key in ('source_commit','source_fingerprint','input_sha256'):
        if auth[key]!=plan[key]:raise ValueError('AUTHORIZATION_IDENTITY')
    if auth['plan_sha256']!=plan_sha or auth['execution_branch']!=EXECUTION_BRANCH or auth['output_path']!=OUTPUT_PATH:
        raise ValueError('AUTHORIZATION_SCOPE')
    if type(auth['attempts']) is not int or auth['attempts']!=1 or type(auth['retries']) is not int or auth['retries']!=0 or auth['mandatory_stop'] is not True:
        raise ValueError('ONE_ATTEMPT_RETRY_ZERO_STOP')
    return True


class Output:
    def __init__(self,root):self.root=Path(root);self.total=0
    def write(self,name,value,stop_record=False):
        if Path(name).name!=name:raise ValueError('OUTPUT_FILENAME')
        data=json.dumps(value,indent=2,allow_nan=False).encode()+b'\n'
        if stop_record and name not in ('result.json','MANDATORY_STOP.json','failure.json'):raise ValueError('STOP_RECORD_NAME')
        limit=RESOURCES['total_output_bytes']-RESOURCES['log_bytes']-(0 if stop_record else RESOURCES['failure_bytes'])
        shard=RESOURCES['failure_bytes'] if stop_record else RESOURCES['output_shard_bytes']
        if len(data)>shard or self.total+len(data)>limit:
            raise ArithmeticError('OUTPUT_BYTE_CAP')
        p=self.root/name
        with p.open('xb') as f:f.write(data)
        self.total+=len(data);return hashlib.sha256(data).hexdigest()


class MemoryWatch:
    def __init__(self):
        self.peak=0;self.violation=False;self.samples=0;self.last=None;self.max_gap=0.;self.stop_event=threading.Event();self.thread=None
    def sample(self):
        now=time.perf_counter()
        if self.last is not None:self.max_gap=max(self.max_gap,now-self.last)
        self.last=now;self.samples+=1;pending=[os.getpid()];seen=set();total=0
        while pending:
            pid=pending.pop()
            if pid in seen:continue
            seen.add(pid)
            try:
                s=Path(f'/proc/{pid}/status').read_text();total+=next((int(l.split()[1])*1024 for l in s.splitlines() if l.startswith('VmRSS:')),0)
                for p in Path(f'/proc/{pid}/task').glob('*/children'):
                    try:pending.extend(map(int,p.read_text().split()))
                    except OSError:pass
            except (OSError,ValueError):pass
        self.peak=max(self.peak,total)
        if total>RESOURCES['aggregate_sampled_RSS_bytes']:
            self.violation=True
            for pid in seen-{os.getpid()}:
                try:os.kill(pid,signal.SIGTERM)
                except OSError:pass
    def loop(self):
        while not self.stop_event.wait(.05):self.sample()
    def start(self):self.sample();self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()
    def stop(self):
        self.stop_event.set()
        if self.thread:self.thread.join()
        self.sample()
    def report(self):return {'sampled_aggregate_peak_RSS_bytes':self.peak,'samples':self.samples,'max_sample_gap':self.max_gap,
        'budget_violation':self.violation,'qualification':'Between-sample instantaneous peaks unmeasured'}
