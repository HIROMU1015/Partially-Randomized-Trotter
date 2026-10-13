"""Closed six-call contract; preparation manifests never grant runtime access."""
import copy
import json
import math
import os
from pathlib import Path
import time
from ..h4_geometry.identity import Stop, require, fingerprint, hash_id
from ..h4_geometry.prelaunch_audit import private_path
from .adapter import METHOD, NAMESPACE, ORIGINAL_SOURCE, SEMANTICS, template

SELECTION=(('0.70',5),('0.80',5),('1.40',3))
CAPS=dict(new_compile_calls=6,workers=1,driver_AS_RSS=8*2**30,worker_AS_RSS=8*2**30,
          observer_AS=256*2**20,observer_RSS=64*2**20,admission_bytes=int(120.25*2**30),
          threads=1,observation_max_seconds=5,observer_period_seconds=1,
          stage_wall_seconds=3600,task_wall_seconds=600,
          stage_output_bytes=256*2**20,observer_output_bytes=32*2**20,
          storage_required_bytes=320*2**20,storage_required_inodes=256)
CARRY=dict(actual_invocations=73023,actual_invocation_cap=74805,
           charged_bytes=13615088338,charge_cap_bytes=17*2**30,
           recorded_wall_seconds=268318.34715263546,failed_calls_refunded=False)
FLAGS=dict(retry=False,automatic_scope_expansion=False,next_stage_authorized=False,mandatory_stop=True)


def digest(value):return fingerprint('h4-b0i-closed-contract-v1',value)


def validate_plan(plan):
    require(plan['schema_version']=='h4-b0i-pilot6-plan-v1' and plan['namespace']==NAMESPACE,'new six-call namespace/schema')
    require(plan['original_science_source']==ORIGINAL_SOURCE and plan['method']==METHOD and plan['semantics']==SEMANTICS,'source/method/control closure')
    require(plan['caps']==CAPS and plan['carry']==CARRY and
            all(type(plan['caps'][k]) is type(v) for k,v in CAPS.items()) and
            all(type(plan['carry'][k]) is type(v) for k,v in CARRY.items()),'fixed caps and historical carry')
    require(all(plan[k] is v for k,v in FLAGS.items()),'one-shot/STOP scope')
    require(type(plan['source_commit']) is str and len(plan['source_commit'])==40 and
            all(c in '0123456789abcdef' for c in plan['source_commit']),'actual SOURCE')
    tasks=plan['tasks'];require(len(tasks)==6,'six wrappers only')
    expected=[(g,p,a) for g,p in SELECTION for a in ('cosine','sine')]
    require([(t['geometry'],t['prefix'],t['axis']) for t in tasks]==expected,'exact representative registry and ordering')
    require(len({t['task_id'] for t in tasks})==6,'unique task IDs')
    for task in tasks:
        require(task['template']==template(task['prefix'],1),'deterministic S2/q1 contract')
        require(task['trajectory_seed'] is None and task['trajectory_index'] is None,'no trajectory generation/seed reuse')
        require(task['numeric_circuit_fingerprint'] is None and task['compiled_cost']=='UNMEASURED','not measured in preparation')
        require(type(task['tail_identity']) is float and math.isfinite(task['tail_identity']),'bound saved scalar')
        require(task['input_file']['bytes']>0,'frozen input bytes')
        hash_id(task['input_file']['sha256'])
        require(task['task_id']==digest({k:v for k,v in task.items() if k!='task_id'}),'task binding changed')
    require(plan['carry']['actual_invocations']+6<=plan['carry']['actual_invocation_cap'],'cumulative actual cap')
    require(plan['carry']['charged_bytes']+plan['caps']['stage_output_bytes']<=plan['carry']['charge_cap_bytes'],'cumulative charge cap')
    require(plan['output']!=plan['control'] and plan['output']!=plan['one_shot_lock'],'separate output/control/lock')
    for key in ('output','control','one_shot_lock'):private_path(plan[key])
    require(Path(plan['output']).name==NAMESPACE and Path(plan['control']).name==NAMESPACE+'-control','independent pilot output namespace')
    return plan


def authorize(plan, auth, review):
    # All current drafts fail here, before input/circuit/import/lock/process actions.
    require(auth.get('schema_version')=='h4-b0i-pilot6-authorization-v1' and
            auth.get('approved') is True and auth.get('runtime_authorization') is True,'new explicit approval required')
    validate_plan(plan)
    require(plan.get('approved') is True and plan.get('runtime_authorization') is True and
            plan.get('allowed_cpus')==auth['allowed_cpus'] and
            plan.get('absolute_launch_command')==auth['absolute_launch_command'],
            'approved plan/auth CPU and launch binding required')
    require(auth['plan_digest']==digest(plan) and auth['namespace']==NAMESPACE and
            auth['source_commit']==plan['source_commit'],'auth/plan/source binding')
    require(all(auth.get(k) is v for k,v in FLAGS.items()),'auth STOP scope')
    hash_id(auth['user_approval_evidence_sha256'])
    require(review.get('approved') is True and review.get('plan_digest')==digest(plan) and
            review.get('authorization_digest')==digest(auth),'final review binding')
    roles=auth['role_cpus'];require(set(roles)=={'driver','worker','observer'},'three owned roles')
    selected=[roles[r] for r in ('driver','worker','observer')]
    require(all(type(c) is int and c>=0 for c in selected) and len(set(selected))==3,'distinct explicitly approved CPUs')
    require(sorted(auth['allowed_cpus'])==sorted(selected),'explicit role CPU permissions')
    require(auth.get('absolute_launch_command') is not None,'approved launch binding')
    return selected


def fresh_gate(plan,auth,observation,*,now=None):
    now=time.monotonic() if now is None else now
    memory=observation['memory'];caps=plan['caps']
    require(0<=now-memory['observed_at']<=5 and observation['duration_seconds']<=5,'fresh admission within five seconds')
    require(memory['available']>=caps['admission_bytes'],'pilot effective memory admission')
    require(set(auth['allowed_cpus'])<=set(observation['scheduler_affinity']) & set(memory['process_cpus']) & set(observation['online_cpus']),'fresh CPU availability, no implied rights')
    topology={v['cpu']:v for v in observation['topology']}
    selected=[topology[c] for c in auth['allowed_cpus']]
    require(len({(v['package'],v['core']) for v in selected})==3 and
            all(v['busy_fraction'] is not None and v['busy_fraction']<=.2 for v in selected),
            'fresh distinct physical cores with <=20 percent passive load')
    require(memory['psi_full_by_scope']['host']<1 and
            all(v==0 for p,v in memory['psi_full_by_scope'].items() if p!='host'),'fresh host/cgroup pressure')
    filesystem=observation['filesystem']
    require(filesystem['available_bytes']>=caps['storage_required_bytes'] and
            filesystem['available_inodes']>=caps['storage_required_inodes'],'block/inode capacity')
    quota=observation['quota'];require(quota['status']=='KNOWN','quota unknown: fail closed')
    for item in quota['items']:
        require(item['status'] in ('ACTIVE','DISABLED'),'quota scope visibility')
        if item['status']=='ACTIVE':
            require(item['available_bytes'] is None or item['available_bytes']>=caps['storage_required_bytes'],'byte quota')
            require(item['available_inodes'] is None or item['available_inodes']>=caps['storage_required_inodes'],'inode quota')


def claim_lock(path, plan_digest):
    path=private_path(path);fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    try:
        data=json.dumps({'plan_digest':plan_digest,'pid':os.getpid(),'one_shot':True}).encode()+b'\n'
        require(os.write(fd,data)==len(data),'one-shot lock write');os.fsync(fd)
    finally:os.close(fd)
    from ..h4_geometry.resources import fsync_directory
    fsync_directory(path.parent)
    # Never unlink this consumed lock after COMPLETE or STOP.


class CallLedger:
    def __init__(self, tasks, *, reserve=lambda row:None):
        self.tasks={t['task_id'] for t in tasks};self.reserved=[];self.reserve=reserve
    def before_call(self,task_id):
        require(task_id in self.tasks and task_id not in self.reserved and len(self.reserved)<6,'unknown/repeated/seventh compile refused')
        row={'task_id':task_id,'call':len(self.reserved)+1,'status':'RESERVED_BEFORE_INVOCATION_NO_REFUND'}
        self.reserve(row);self.reserved.append(task_id)
        return row


class FirstStop:
    def __init__(self):self.reason=None
    def keep(self,reason):
        if self.reason is None:self.reason=str(reason)
        return self.reason


def cleanup(children,monitor,first):
    """Popen children paired with stable owned pidfds; continue after exit races."""
    complete=True
    for process,owner in children:
        try:
            if process.poll() is None:
                if owner is None:process.terminate()  # direct, unreaped Popen child during bootstrap only
                else:owner.terminate()
        except BaseException as exc:complete=False;first.keep('owned termination: '+str(exc))
        try:
            try:process.wait(timeout=2)
            except __import__('subprocess').TimeoutExpired:
                import signal
                if owner is None:process.kill()
                else:owner.terminate(signal.SIGKILL)
                process.wait(timeout=2)
        except BaseException as exc:complete=False;first.keep('cleanup: '+str(exc))
        finally:
            for stream in (process.stdin,process.stdout):
                try:
                    if stream is not None:stream.close()
                except BaseException as exc:complete=False;first.keep('pipe close: '+str(exc))
            try:
                if owner is not None:owner.close()
            except BaseException as exc:complete=False;first.keep('pidfd close: '+str(exc))
    try:
        if monitor is not None:monitor.close(abort=first.reason is not None)
    except BaseException as exc:complete=False;first.keep('observer cleanup: '+str(exc))
    return complete
