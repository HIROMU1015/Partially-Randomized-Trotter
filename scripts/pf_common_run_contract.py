"""One-shot common-PF execution contracts; stdlib only before authorization."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
BASE='3728a618d7f09e67a070632ded1f27dab54b21a4'
FORMULAS=('2nd','4th','4th(m5_best)')
PIECES=dict(zip(FORMULAS,(1,3,11)))
EPS=(.05,.01,.005,.001,.0005)
COMPILER='1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b'
NATIVE='M5_TOL_ONLY_DF_EXPLICIT_GAUSSIAN_DIRECTIONAL_SCALAR_ONCE_V1'

def require(ok,label):
    if not ok:raise ValueError(label)
def read(p):return json.loads(Path(p).read_text())
def encoded(v):return (json.dumps(v,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
def digest(v):return hashlib.sha256(encoded(v)).hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:f.write(encoded(v));f.flush();os.fsync(f.fileno())
def seal(p,role):
    p=Path(p).resolve();return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),role=role)
def gitbytes(commit,path):return subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT)
def source():return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
def cid(row):return '_'.join(str(row[k]) for k in ('system','method','prefix','q','formula'))
def norm(system,row):
    return dict(system=system,method='FULL' if row['method']=='B1' else row['method'],prefix=int(row['prefix']),q=int(row['q']),formula=row['formula'])

def verify(plan):
    require(plan['approval']['execution_authorized'] is True and plan['approval']['no_intermediate_reapproval'] is True,'EXPLICIT_USER_APPROVAL')
    require(plan['retry'] is False and plan['resume'] is False and plan['system'] in ('H2','H4','H6'),'FIXED_SEPARATE_TRACK')
    root=Path(plan['private_root'])
    approval=read(root/'approval_receipt.json')
    require(read(root/'plan_APPROVED.json')==plan and approval['plan_sha256']==sha(root/'plan_APPROVED.json')
        and approval['SOURCE']==plan['SOURCE'] and approval['execution_authorized'] is True,'IMMUTABLE_APPROVED_PLAN')
    for e in plan['closure']:
        p=Path(e['path']);require(p.stat().st_size==e['bytes'] and sha(p)==e['sha256'],'SEALED_BYTES:'+e['role'])
        if e['role'].startswith('SOURCE:'):
            require(p.read_bytes()==gitbytes(plan['SOURCE'],e['role'][7:]),'COMMITTED_SOURCE')

class Stage:
    def __init__(self,private,kind,caps,dependencies=()):
        self.private=Path(private);self.plan=read(self.private/'plan_APPROVED.json');verify(self.plan)
        require(kind in ('SIGNAL','QUALIFY','COST','ACCOUNT'),'REGISTERED_STAGE')
        for dep in dependencies:require(read(self.private/dep/'completion.json')['status']=='PASS','DEPENDENCY:'+dep)
        self.root=self.private/kind
        require(not self.root.exists(),'ONE_SHOT_STAGE_ALREADY_ENTERED')
        self.root.mkdir();self.out=self.root/'output';self.out.mkdir();(self.root/'ledger').mkdir()
        self.caps=caps;self.used={}
        save(self.root/'reserved_counts.json',{k:0 for k in caps})
        grant=dict(SOURCE=self.plan['SOURCE'],stage=kind,system=self.plan['system'],caps=caps,
            plan_sha256=sha(self.private/'plan_APPROVED.json'),grant_id=kind+'_'+digest([self.plan['SOURCE'],str(self.root),caps]),
            approved_by_user=True,valid=True,retry=False,resume=False)
        save(self.root/'grant.json',grant);save(self.root/'CONSUMED.json',dict(grant=grant,state='CONSUMED_NONREFUNDABLE'))
        self.started=time.monotonic()
    def take(self,kind,task='context',count=1):
        # Each process/task reserves its own file. The end auditor aggregates
        # every nonrefundable reservation against global caps independently.
        require(kind in self.caps and type(count) is int and count>0,'CALL_KIND_COUNT:'+kind)
        index=self.used.get((kind,task),0);self.used[kind,task]=index+1
        require(count<=self.caps[kind],'SINGLE_RESERVATION_EXCEEDS_CAP')
        import fcntl
        with (self.root/'reservation.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            current=read(self.root/'reserved_counts.json')
            require(current[kind]+count<=self.caps[kind],'NONREFUNDABLE_GLOBAL_CAP:'+kind)
            save(self.root/'ledger'/f'{task}__{kind}__{index}.json',dict(kind=kind,count=count,task=task,entered_nonrefundable=True))
            current[kind]+=count
            temporary=self.root/'reserved_counts.partial';save(temporary,current);temporary.replace(self.root/'reserved_counts.json')
    def total(self):
        totals={k:0 for k in self.caps}
        for p in (self.root/'ledger').glob('*.json'):
            e=read(p);require(e['kind'] in totals,'UNREGISTERED_LEDGER');totals[e['kind']]+=e['count']
        require(all(totals[k]<=self.caps[k] for k in totals),'GLOBAL_CALL_CAP_EXCEEDED')
        return totals
    def finish(self,extra=None):
        verify(self.plan)
        result=dict(status='PASS',SOURCE=self.plan['SOURCE'],actual_calls=self.total(),caps=self.caps,
            wall_seconds=time.monotonic()-self.started,retry=False,next_stage_allowed_by_current_user_authorization=True,
            all_workers_reaped=True,**(extra or {}))
        save(self.root/'completion.json',result);return result
    def stop(self,e):
        import traceback
        save(self.root/'STOP.json',dict(status='MANDATORY_STOP',reason=str(e),traceback=traceback.format_exc(),
            retry=False,resume=False,actual_calls=self.total()))

def resources():
    """Read local affinity/cgroups and CPU-idle; no machine-wide changes."""
    import shutil
    def cpus():
        vals={}
        for line in Path('/proc/stat').read_text().splitlines():
            a=line.split()
            if a and a[0].startswith('cpu') and a[0][3:].isdigit():
                v=list(map(int,a[1:]));vals[int(a[0][3:])]=(sum(v),v[3]+v[4])
        return vals
    first=cpus();time.sleep(.3);last=cpus();aff=os.sched_getaffinity(0)
    idle=sum((last[k][1]-first[k][1])/max(1,last[k][0]-first[k][0]) for k in aff if k in first and k in last)
    mem=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
    slots=float(len(aff));path=Path('/sys/fs/cgroup')
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        if line.startswith('0::'):path=path/line[3:].lstrip('/')
    for p in [path,*path.parents]:
        if not str(p).startswith('/sys/fs/cgroup'):continue
        try:
            v=(p/'cpu.max').read_text().split()
            if v[0]!='max':slots=min(slots,int(v[0])/int(v[1]))
        except FileNotFoundError:pass
        try:
            lim=(p/'memory.max').read_text().strip()
            if lim!='max':mem=min(mem,max(0,int(lim)-int((p/'memory.current').read_text())))
        except FileNotFoundError:pass
    return dict(affinity=len(aff),idle_cpu=min(idle,slots),available_ram=mem,free_disk=shutil.disk_usage(ROOT).free,
        arbitrary_caps=None,shared_system_changes=False)

def workers(observation,pending,ram_proxy,disk_proxy):
    return max(0,min(pending,math.floor(observation['idle_cpu']),observation['available_ram']//ram_proxy,
                     observation['free_disk']//disk_proxy))
