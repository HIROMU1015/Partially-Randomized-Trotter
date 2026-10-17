"""New recovery authorization, saved B0 acceptance and nonrefundable two-axis ledger."""
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time

import h8_phase_b2_recovery_contract as c
import h8_strengthened_runtime as io
from h8_b0_saved_anchor_audit import git_bytes

sha=io.sha
read=io.read
private_path=io.private_path
Writer=io.Writer
THREAD_KEYS=io.THREAD_KEYS


def verify_authorization(config,g,*,config_sha,affinity=None):
    p=c.plan();source=config.get('SOURCE')
    if (g.get('schema')!='H8_PHASE_B2_RECOVERY_NEW_SCOPE_GRANT_V1' or
        g.get('approved_by_user') is not True or g.get('runtime_authorization') is not True or
        not g.get('approval_text') or g.get('scope')!=c.SCOPE or
        not isinstance(source,str) or len(source)!=40 or any(x not in '0123456789abcdef' for x in source) or
        g.get('SOURCE')!=source or config.get('execution_sealed') is not True or
        g.get('config_sha256')!=config_sha or g.get('plan_sha256')!=c.digest(p) or
        config.get('plan_sha256')!=c.digest(p) or g.get('resources')!=p['resources'] or
        g.get('call_caps')!=p['call_caps'] or config.get('resources')!=p['resources'] or
        any(g.get(k) is not v for k,v in [('one_shot',True),('retry',False),('resume',False),('next_stage_authorized',False),('C_authorized',False),('refit',False),('accept_saved_B0_2_axes',True),('accept_saved_cosine_and_sine_spool',True)])):
        raise ValueError('NEW_EXPLICIT_H8_B2_GRANT_REQUIRED')
    cpus=[g.get('driver_cpu'),g.get('monitor_cpu'),*g.get('worker_cpus',[]),g.get('witness_cpu')]
    available=os.sched_getaffinity(0) if affinity is None else affinity
    if (len(g.get('worker_cpus',[]))!=1 or len(cpus)!=4 or len(set(cpus))!=4 or any(type(x)is not int or x<0 for x in cpus)
            or g.get('allowed_cpus')!=cpus or not set(cpus)<=available):
        raise ValueError('FOUR_DISJOINT_EXPLICIT_CPUS_REQUIRED')
    paths=[private_path(config[k]) for k in ('output','control','lock')]
    if len(set(paths))!=3:raise ValueError('DISJOINT_PRIVATE_PATHS_REQUIRED')
    return cpus


def verify_grant(config,g,f,*,config_sha,now=None,affinity=None):
    cpus=verify_authorization(config,g,config_sha=config_sha,affinity=affinity)
    source=config['SOURCE'];now=time.monotonic() if now is None else now
    stamp=f.get('monotonic_seconds')
    if (type(stamp) not in (int,float) or not math.isfinite(stamp) or not 0<=now-stamp<=30 or
            f.get('source')!=source or f.get('config_sha256')!=config_sha or
            f.get('grant_sha256')!=c.digest(g) or f.get('allowed_cpus')!=cpus or
            any(f.get(k) is not True for k in ('profile_verified','source_input_bytes_verified',
                'physical_cores_verified','resources_idle_verified','protected_jobs_unchanged',
                'quota_verified','external_RSS_monitor_ready'))):raise ValueError('FRESH_B0_GATE_REQUIRED')


def verify_observer(f):
    from track_a_phase_a_parallel import identity
    current=identity(os.getppid())
    if current is None or list(current)!=f.get('observer_identity'):
        raise ValueError('LIVE_OWNED_OBSERVER_PARENT_REQUIRED')


def verify_files(config,*,runtime=True,git=True):
    if config['provider_source']!=c.PROVIDER or config['expected_profile_digest']!=c.PROFILE:
        raise ValueError('PINNED_PROVIDER_PROFILE')
    if set(config['source_hashes'])!=set(c.SOURCES) or len(config['provider_hashes'])!=265:
        raise ValueError('SOURCE_CLOSURE_INVENTORY')
    if len(config['runtime_file_hashes'])!=34273:raise ValueError('FULL_RUNTIME_INVENTORY_REQUIRED')
    if c.digest(config['runtime_file_hashes'])!=c.RUNTIME_MAP_SHA or c.digest(config['provider_hashes'])!=c.PROVIDER_MAP_SHA:
        raise ValueError('PINNED_INVENTORY_BYTES_REQUIRED')
    rows={row['logical_name']:row for row in config['files']}
    if len(rows)!=len(config['files']):raise ValueError('DUPLICATE_INPUT')
    for expected in c.plan()['binding']['inputs']:
        row=rows[expected['logical_name']]
        if any(row[k]!=expected[k] for k in ('bytes','sha256')):raise ValueError('FIXED_INPUT_RECEIPT')
    if config['input_mapping']!={k:row['path'] for k,row in rows.items()}:raise ValueError('INPUT_MAPPING')
    for root,commit,hashes in [(c.ROOT,config['SOURCE'],config['source_hashes']),
            (Path(config['provider_root']),c.PROVIDER,config['provider_hashes'])]:
        for name,expected in hashes.items():
            p=Path(name)
            if p.is_absolute() or '..' in p.parts:raise ValueError('SOURCE_PATH')
            if sha(root/p)!=expected:raise ValueError('SOURCE_DISK_CHANGED:'+name)
            if git and hashlib.sha256(git_bytes(root,'show',commit+':'+name)).hexdigest()!=expected:
                raise ValueError('SOURCE_GIT_CHANGED:'+name)
    for row in config['files']:
        p=Path(row['path'])
        if p.is_symlink() or p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:
            raise ValueError('INPUT_CHANGED:'+row['logical_name'])
    if runtime:
        for path,expected in config['runtime_file_hashes'].items():
            if sha(path)!=expected:raise ValueError('RUNTIME_CHANGED:'+path)


def protected(config):
    for row in config['protected_files']:
        if Path(row['path']).stat().st_size!=row['bytes'] or sha(row['path'])!=row['sha256']:
            raise ValueError('PROTECTED_FILE_CHANGED')
    for row in config['protected_worktrees']:
        head=git_bytes(row['path'],'rev-parse','HEAD').decode().strip()
        status=git_bytes(row['path'],'status','--porcelain=v1','-z')
        if head!=row['HEAD'] or hashlib.sha256(status).hexdigest()!=row['status_sha256']:
            raise ValueError('PROTECTED_WORKTREE_CHANGED')


def filesystem(parent):
    parent=Path(parent).resolve();stat=os.statvfs(parent)
    rows=[line.split() for line in Path('/proc/mounts').read_text().splitlines()]
    rows=[r for r in rows if len(r)>=4 and parent.is_relative_to(r[1])]
    if not rows:raise ValueError('MOUNT_UNKNOWN')
    m=max(rows,key=lambda r:len(r[1]));options=m[3].split(',')
    if m[2] not in ('ext4','xfs'):raise ValueError('QUOTA_FILESYSTEM_UNKNOWN')
    # Preserve the previous mount-option policy; enabled or unknown quotas STOP.
    if any(('quota' in x and not x.startswith('no')) or x in ('uquota','gquota','pquota') for x in options):
        raise ValueError('ENABLED_QUOTA_NEEDS_SEPARATE_VERIFIED_EFFECTIVE_LIMITS')
    return dict(bytes=stat.f_bavail*stat.f_frsize,inodes=stat.f_favail,
                mount=m[1],fstype=m[2],options=options,quota_verified=True)


def cpu_counters():
    return {int(p[0][3:]):list(map(int,p[1:9])) for line in Path('/proc/stat').read_text().splitlines()
            if (p:=line.split()) and p[0].startswith('cpu') and p[0][3:].isdigit()}


class CPUIdleError(ValueError):
    def __init__(self,receipt):
        super().__init__('PHYSICAL_CORE_BUSY')
        self.receipt=receipt


def idle_receipt(first,last,cpus):
    """Inspect every selected logical CPU, preserving the unchanged 5% limit."""
    rows=[]
    for cpu in sorted(cpus):
        before,after=first.get(cpu),last.get(cpu)
        valid=(isinstance(before,list) and isinstance(after,list) and len(before)==len(after)==8
               and all(type(x)is int and x>=0 for x in before+after))
        d=[b-a for a,b in zip(before,after)] if valid else None
        total=sum(d) if d is not None else 0
        valid=valid and min(d)>=0 and total>0
        busy=total-d[3]-d[4] if valid else None
        ok=valid and 20*busy<=total
        rows.append(dict(cpu=cpu,before=before,after=after,delta=d,total_ticks=total,
            busy_ticks=busy,busy_fraction=busy/total if valid else None,
            status='PASS' if ok else ('BUSY' if valid else 'UNKNOWN_COUNTERS')))
    if not rows:raise ValueError('CPU_OBSERVATION_EMPTY')
    return dict(status='PASS' if all(x['status']=='PASS' for x in rows) else 'STOP',
        maximum_busy_fraction='1/20',policy='ALL_SELECTED_LOGICAL_CPUS_INCLUDING_SMT_SIBLINGS',
        own_activity_subtracted=False,CPUs=rows,
        failing_CPUs=[x['cpu'] for x in rows if x['status']!='PASS'])


def check_idle(first,last,cpus):
    receipt=idle_receipt(first,last,cpus)
    if receipt['status']!='PASS':raise CPUIdleError(receipt)
    return receipt


def self_cpu():
    fields=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
    return dict(user_ticks=int(fields[11]),system_ticks=int(fields[12]),
        last_CPU=int(fields[36]),affinity=sorted(os.sched_getaffinity(0)))


def process_cpu(pid):
    """Only explicitly owned PIDs; start-time identity prevents PID reuse attribution."""
    try:
        path=Path('/proc')/str(pid);fields=(path/'stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=pid,start_ticks=int(fields[19]),uid=path.stat().st_uid,
            user_ticks=int(fields[11]),system_ticks=int(fields[12]),last_CPU=int(fields[36]),
            affinity=sorted(os.sched_getaffinity(pid)),available=True)
    except (FileNotFoundError,ProcessLookupError):return dict(pid=pid,available=False)


def process_cpu_delta(before,after):
    same=all(before.get(k)==after.get(k) for k in ('pid','start_ticks','uid'))
    known=before.get('available') is True and after.get('available') is True and same
    value=dict(before=before,after=after,identity_unchanged=known)
    if known:
        value.update(user_ticks_delta=after['user_ticks']-before['user_ticks'],
            system_ticks_delta=after['system_ticks']-before['system_ticks'])
    return value


def measure_idle(cpus,seconds=1.,*,process_pids=None):
    if seconds!=1.:raise ValueError('FIXED_ONE_SECOND_CPU_WINDOW_REQUIRED')
    started=time.monotonic();pids=sorted(set(process_pids or [os.getpid()]))
    process_first={pid:process_cpu(pid) for pid in pids}
    own_first=self_cpu();first=cpu_counters()
    time.sleep(seconds);last=cpu_counters();own_last=self_cpu()
    process_last={pid:process_cpu(pid) for pid in pids}
    receipt=idle_receipt(first,last,cpus)
    receipt.update(window_requested_seconds=seconds,elapsed_seconds=time.monotonic()-started,
        self_before=own_first,self_after=own_last,
        owned_processes=[process_cpu_delta(process_first[pid],process_last[pid]) for pid in pids],
        process_accounting_used_to_relax_gate=False,
        self_cpu_ticks_delta=own_last['user_ticks']+own_last['system_ticks']-own_first['user_ticks']-own_first['system_ticks'])
    return receipt


def topology(cpus):
    identities=[];siblings=set()
    for cpu in cpus:
        p=Path('/sys/devices/system/cpu')/('cpu'+str(cpu))/'topology'
        identities.append(((p/'physical_package_id').read_text(),(p/'core_id').read_text()))
        # Include SMT siblings in the idle check, without changing their affinity.
        for span in (p/'thread_siblings_list').read_text().strip().split(','):
            ends=list(map(int,span.split('-')));siblings.update(range(ends[0],ends[-1]+1))
    if len(identities)!=4 or len(set(identities))!=4:raise ValueError('FOUR_PHYSICAL_CORES_REQUIRED')
    relative=next(x.split(':',2)[2] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'))
    root=Path('/sys/fs/cgroup');group=root/relative.lstrip('/')
    while group.is_relative_to(root):
        quota=group/'cpu.max'
        if quota.exists():
            q,period=quota.read_text().split()
            if q!='max' and int(q)<4*int(period):raise ValueError('EFFECTIVE_CPU_QUOTA_BELOW_FOUR')
        if group==root:break
        group=group.parent
    return siblings


class Permit:
    def __init__(self,config,g,output):
        self.config,self.grant,self.output=config,g,Path(output)
        self.used={};self.reserved={};self.settled={};self.started=time.monotonic();self.last=0.
    def require(self,scope):
        if scope!=c.SCOPE or self.grant.get('scope')!=c.SCOPE or self.grant.get('runtime_authorization') is not True:
            raise ValueError('B2_COMPILE_VALIDATION_SCOPE_ONLY')
        self.guard()
    def guard(self):
        now=time.monotonic()
        if now-self.last<.25:return
        self.last=now;r=self.grant['resources']
        if now-self.started>r['wall_seconds']:raise ValueError('WALL_CEILING')
        status=dict(line.split(':',1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
        if int(status['VmHWM'].split()[0])*1024>r['driver_RSS_bytes']:raise ValueError('DRIVER_RSS_CEILING')
        all_files=[p for p in self.output.rglob('*') if p.is_file()]
        if sum(p.stat().st_size for p in all_files)>r['stage_storage_bytes']:raise ValueError('STAGE_STORAGE_CEILING')
        files=[p for p in all_files if 'scratch' not in p.relative_to(self.output).parts]
        if len(files)>r['output_files_max'] or sum(p.stat().st_size for p in files)>r['output_bytes']:
            raise ValueError('OUTPUT_CEILING')
    def journal(self,row):
        with (self.output/'reservations.jsonl').open('a') as f:
            f.write(json.dumps(row,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
    def take(self,name,count=1):
        self.guard();caps=self.grant['call_caps']
        held=sum(x.get(name,0) for x in self.reserved.values())
        if type(count)is not int or count<0 or name not in caps or self.used.get(name,0)+held+count>caps[name]:
            raise ValueError('CALL_CAP:'+name)
        self.used[name]=self.used.get(name,0)+count;self.journal(dict(call=name,used=self.used[name],refund=False))
    def reserve(self,task):
        if task['id'] in self.reserved:raise ValueError('TASK_ALREADY_RESERVED')
        if task.get('counts')!={'compile':1,'qpy_load':1}:raise ValueError('COMPILE_ONLY_TASK')
        self.take('compile',0)
        if self.used.get('compile',0)+sum(x['compile'] for x in self.reserved.values())+1>self.grant['call_caps']['compile']:
            raise ValueError('COMPILE_RESERVATION_CAP')
        self.reserved[task['id']]={'compile':1,'qpy_load':1};self.journal(dict(task=task['id'],compile=1,refund=False))
    def settle(self,task_id):
        if task_id not in self.reserved or task_id in self.settled:raise ValueError('UNIQUE_SETTLEMENT')
        self.settled[task_id]=dict(self.reserved[task_id])
    def actual(self):
        result={k:self.used.get(k,0) for k in self.grant['call_caps']}
        for counts in self.settled.values():
            for name,count in counts.items():result[name]+=count
        return result
