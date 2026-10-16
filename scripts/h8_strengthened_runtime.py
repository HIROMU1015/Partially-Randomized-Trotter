"""Stdlib grant gate, immutable inputs and irrevocable task budgets for H8.

Nothing in this module imports the numerical adapter or launches science.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import time

import h8_strengthened_contract as contract

ROOT = contract.ROOT
THREAD_KEYS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
               'NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS','QISKIT_NUM_PROCS','RAYON_NUM_THREADS')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def read(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 32*2**20:
        raise ValueError('JSON_INPUT_PATH_OR_SIZE')
    def pairs(items):
        result = {}
        for k,v in items:
            if k in result:raise ValueError('DUPLICATE_JSON_KEY')
            result[k] = v
        return result
    def reject(value):raise ValueError('NONFINITE_JSON:'+value)
    return json.loads(path.read_text(),object_pairs_hook=pairs,parse_constant=reject)


def private_path(path):
    path = Path(path)
    if (not path.is_absolute() or path.resolve() != path or
            not path.is_relative_to('/home/AbeHiromu/projects/h6-handoff-evidence')):
        raise ValueError('EXCLUSIVE_PRIVATE_EVIDENCE_PATH')
    return path


def verify_grant(config,grant,fresh,*,config_sha256,source,now=None,affinity=None):
    p = contract.plan()
    if (grant.get('schema') != 'H8_STRENGTHENED_PHASE_A_GRANT_V1' or
            grant.get('approved_by_user') is not True or grant.get('runtime_authorization') is not True or
            not grant.get('approval_text') or grant.get('scope') != contract.SCOPE or
            grant.get('SOURCE') != source or config.get('SOURCE') != source or
            grant.get('config_sha256') != config_sha256 or
            grant.get('plan_sha256') != contract.digest(p) or
            config.get('plan_sha256') != contract.digest(p) or
            grant.get('resources') != p['resources'] or grant.get('call_caps') != p['call_caps'] or
            grant.get('one_shot') is not True or grant.get('retry') is not False or
            grant.get('resume') is not False or grant.get('next_stage_authorized') is not False or
            config.get('execution_sealed') is not True):
        raise ValueError('NEW_SEALED_H8_PHASE_A_USER_GRANT_REQUIRED')
    workers=grant.get('worker_cpus',[])
    expected=[grant.get('driver_cpu'),grant.get('monitor_cpu'),*workers]
    available=os.sched_getaffinity(0) if affinity is None else affinity
    if (len(workers)!=16 or any(type(x) is not int or x<0 for x in expected) or
            len(set(expected))!=18 or grant.get('allowed_cpus')!=expected or
            not set(expected)<=available):
        raise ValueError('EXPLICIT_DISJOINT_H8_CPU_ASSIGNMENT_REQUIRED')
    current=time.monotonic() if now is None else now
    stamp=fresh.get('monotonic_seconds')
    if (type(stamp) not in (int,float) or not math.isfinite(stamp) or
            not 0 <= current-stamp <=30 or
            fresh.get('source')!=source or fresh.get('config_sha256')!=config_sha256 or
            fresh.get('grant_sha256')!=contract.digest(grant) or
            fresh.get('allowed_cpus')!=expected or
            any(fresh.get(k) is not True for k in ('resources_idle_verified','profile_verified',
                'protected_jobs_unchanged','external_RSS_monitor_ready','physical_cores_verified',
                'quota_verified','source_input_bytes_verified'))):
        raise ValueError('FRESH_H8_RESOURCE_SOURCE_PROFILE_GATE_REQUIRED')
    for path in ('output','lock','control'):
        private_path(config[path])
    if len({config[k] for k in ('output','lock','control')})!=3:
        raise ValueError('DISJOINT_OUTPUT_LOCK_CONTROL')
    return True


def verify_files(config,source,*,runtime=True):
    for checkout,commit,hashes in ((ROOT,source,config['source_hashes']),
            (Path(config['provider_root']),contract.PROVIDER,config['provider_hashes'])):
        for name,expected in hashes.items():
            path=Path(name)
            if path.is_absolute() or '..' in path.parts:raise ValueError('SOURCE_PATH')
            fixed=subprocess.check_output(['git','show',commit+':'+name],cwd=checkout)
            if hashlib.sha256(fixed).hexdigest()!=expected or sha(checkout/path)!=expected:
                raise ValueError('SOURCE_OR_PROVIDER_CHANGED:'+name)
    for row in config['files']:
        path=Path(row['path'])
        if path.is_symlink() or path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:
            raise ValueError('IMMUTABLE_INPUT_CHANGED:'+row['logical_name'])
    if runtime:
        for path,expected in config['runtime_file_hashes'].items():
            if sha(path)!=expected:raise ValueError('RUNTIME_PROFILE_CHANGED')


class Writer:
    def __init__(self,directory,guard=lambda:None):self.directory,self.guard=Path(directory),guard
    def write(self,name,value):
        self.guard()
        if Path(name).name!=name:raise ValueError('OUTPUT_NAME')
        with (self.directory/name).open('x') as f:
            json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


class Permit:
    def __init__(self,config,grant,output,*,caps=None,started=None,child=False):
        self.config,self.grant,self.output=config,grant,Path(output)
        self.started=time.monotonic() if started is None else started
        self.used={};self.reserved={};self.settled={};self.parent_pid=os.getpid()
        self.call_caps=dict(grant['call_caps'] if caps is None else caps)
        self.child_role=child;self.child_CPU_seconds=0.;self.last_check=0.

    def require(self,scope):
        if scope!='signal' or self.grant['scope']!=contract.SCOPE:raise ValueError('H8_SIGNAL_SCOPE_ONLY')
        self.guard(force=True)

    def guard(self,force=False):
        now=time.monotonic()
        if not force and now-self.last_check<.25:return
        self.last_check=now;r=self.grant['resources']
        if now-self.started>r['wall_seconds']:raise ValueError('H8_WALL_CEILING')
        rss=r['worker_RSS_bytes'] if self.child_role else r['driver_RSS_bytes']
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>rss:raise ValueError('H8_RSS_CEILING')
        files=[p for p in self.output.rglob('*') if p.is_file()]
        if len(files)>r['output_files_max'] or sum(p.stat().st_size for p in files)>r['output_bytes']:
            raise ValueError('H8_OUTPUT_CEILING')

    def journal(self,value):
        with (self.output/'reservations.jsonl').open('a') as f:
            f.write(json.dumps(value,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())

    def take(self,name,count=1):
        self.guard()
        if type(count) is not int or count<0:raise ValueError('CALL_COUNT')
        value=self.used.get(name,0)+count
        reserved=sum(r.get(name,0) for r in self.reserved.values()) if not self.child_role else 0
        if name not in self.call_caps or value+reserved>self.call_caps[name]:raise ValueError('CALL_CAP:'+name)
        self.used[name]=value
        if not self.child_role:self.journal(dict(call=name,used=value,refund=False))

    def reserve(self,task):
        if os.getpid()!=self.parent_pid or task['id'] in self.reserved:raise ValueError('UNIQUE_PARENT_RESERVATION')
        for name,value in task['counts'].items():
            if (type(value) is not int or value<0 or self.used.get(name,0)+
                    sum(r.get(name,0) for r in self.reserved.values())+value>self.call_caps.get(name,0)):
                raise ValueError('RESERVED_CALL_CAP:'+name)
        self.journal(dict(task=task['id'],counts=task['counts'],refund=False))
        self.reserved[task['id']]=dict(task['counts'])  # Held forever, even after success.

    def settle(self,task_id,actual):
        budget=self.reserved[task_id]
        if task_id in self.settled or not set(actual)<=set(budget):raise ValueError('RESERVATION_SETTLEMENT')
        for key,value in budget.items():
            n=actual.get(key,0)
            if type(n) is not int or not 0<=n<=value or (key!='reference_matvec' and n!=value):
                raise ValueError('ACTUAL_RESERVED_CALL_MISMATCH:'+key)
        self.settled[task_id]=dict(actual)

    def child(self,directory,counts):
        return Permit(self.config,self.grant,directory,caps=counts,started=self.started,child=True)

    def phase(self,name):self.guard()
    def update(self,**kwargs):self.guard()
