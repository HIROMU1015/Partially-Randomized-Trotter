"""Full comparison domain, conserved budget and verified six-result reuse."""
import math
from pathlib import Path
from ..h4_geometry.identity import require, fingerprint, hash_id
from ..h4_geometry.prelaunch_audit import private_path
from ..h4_b0i import adapter as science
from ..h4_b0i.contract import FLAGS, fresh_gate, claim_lock, FirstStop, cleanup

NAMESPACE = 'h4-b0i-map240-20261014'
GEOMETRIES = ('0.70','0.80','0.90','1.10','1.40','1.60')
PREFIXES = (3,4,5,6,9)
QS = (1,2,4,8)
PILOT_SOURCE = '8f6c1bc8ceeeffddbecf170ef6ddb0fe6d18b63b'
PILOT_RESULT_COMMIT = 'c377b074ab6cc442d6eb73ca4417bc56fa663d5c'
CAPS = dict(logical_wrappers=240,new_compile_calls=234,verified_reuses=6,workers=1,
    driver_AS_RSS=8*2**30,worker_AS_RSS=8*2**30,observer_AS=256*2**20,
    observer_RSS=64*2**20,admission_bytes=int(120.25*2**30),threads=1,
    observation_max_seconds=5,observer_period_seconds=1,stage_wall_seconds=14400,
    task_wall_seconds=600,stage_output_bytes=256*2**20,observer_output_bytes=160*2**20,
    storage_required_bytes=320*2**20,storage_required_inodes=1024,
    run_inode_cap=1024,journal_bytes=8*2**20,bootstrap_frame_bytes=8192)
CARRY = dict(actual_invocations=73029,actual_invocation_cap=74805,
    charged_bytes=13615239890,charge_cap_bytes=17*2**30,
    recorded_wall_seconds=268330.081695186,failed_calls_refunded=False)
PILOT_KEYS = {('0.70',5,1,a) for a in ('cosine','sine')} | {
    ('0.80',5,1,a) for a in ('cosine','sine')} | {
    ('1.40',3,1,a) for a in ('cosine','sine')}


def digest(value):
    return fingerprint('h4-b0i-map240-contract-v1', value)


def key(task):
    return task['geometry'],task['prefix'],task['template']['q'],task['axis']


def validate_plan(plan):
    require(plan['schema_version']=='h4-b0i-map240-plan-v1' and plan['namespace']==NAMESPACE,'full-map namespace/schema')
    require(plan['method']==science.METHOD and plan['semantics']==science.SEMANTICS and
            plan['original_science_source']==science.ORIGINAL_SOURCE,'unchanged B0+I science')
    require(plan['caps']==CAPS and plan['carry']==CARRY,'fixed proposed caps and conserved carry')
    require(all(type(plan['caps'][k]) is type(v) for k,v in CAPS.items()) and
            all(type(plan['carry'][k]) is type(v) for k,v in CARRY.items()),'budget value types')
    require(all(plan[k] is v for k,v in FLAGS.items()),'one-shot mandatory STOP')
    source=plan['source_commit']
    require(type(source) is str and len(source)==40 and all(c in '0123456789abcdef' for c in source),'fixed actual source')
    tasks=plan['tasks']
    expected=[(g,p,q,a) for g in GEOMETRIES for p in PREFIXES for q in QS for a in ('cosine','sine')]
    require(len(tasks)==240 and [key(t) for t in tasks]==expected,'complete domain and order')
    require(len({t['task_id'] for t in tasks})==240,'unique full-map tasks')
    for t in tasks:
        require(t['task_id']==digest({k:v for k,v in t.items() if k!='task_id'}),'task identity drift')
        require(t['template']==science.template(t['prefix'],t['template']['q']) and
                t['source_commit']==source and t['original_science_source']==science.ORIGINAL_SOURCE,'task source/science')
        require(t['action']==('VERIFY_REUSE' if key(t) in PILOT_KEYS else 'COMPILE'),'closed compile/reuse selection')
        require(t['trajectory_seed'] is None and t['trajectory_index'] is None and
                t['numeric_circuit_fingerprint'] is None and t['compiled_cost']=='UNMEASURED_IN_NEW_MAP','no sampling or invented new measurement')
        require(type(t['tail_identity']) is float and math.isfinite(t['tail_identity']) and
                t['phase_angle']==-t['template']['T']*t['tail_identity'],'fixed finite tail phase')
        require(t['input_file']['bytes']>0,'frozen input size');hash_id(t['input_file']['sha256'])
        require((t.get('reuse_receipt') is not None)==(t['action']=='VERIFY_REUSE'),'explicit reuse proof')
    require(plan['carry']['actual_invocations']+CAPS['new_compile_calls']<=plan['carry']['actual_invocation_cap'],'actual cap')
    require(plan['carry']['charged_bytes']+CAPS['stage_output_bytes']<=plan['carry']['charge_cap_bytes'],'charge cap')
    require(Path(plan['output']).name==NAMESPACE and Path(plan['control']).name==NAMESPACE+'-control','separate new run namespace')
    paths=[private_path(plan[k]) for k in ('output','control','one_shot_lock')]
    require(len(set(paths))==3 and len({p.parent for p in paths})==1,'separate output/control/lock')
    return plan


def authorize(plan,auth,review):
    require(auth.get('schema_version')=='h4-b0i-map240-authorization-v1' and
            auth.get('approved') is True and auth.get('runtime_authorization') is True,'new full-map explicit approval required')
    validate_plan(plan)
    require(plan.get('approved') is True and plan.get('runtime_authorization') is True and
            plan.get('allowed_cpus')==auth['allowed_cpus'] and
            plan.get('absolute_launch_command')==auth['absolute_launch_command'],'plan/auth CPU and command binding')
    require(auth['plan_digest']==digest(plan) and auth['source_commit']==plan['source_commit'] and
            auth['namespace']==NAMESPACE and all(auth.get(k) is v for k,v in FLAGS.items()),'new grant binding')
    hash_id(auth['user_approval_evidence_sha256'])
    roles=auth['role_cpus'];require(set(roles)=={'worker','driver','observer'},'three own roles')
    selected=list(roles.values())
    require(all(type(v) is int and v>=0 for v in selected) and len(set(selected))==3 and
            sorted(selected)==sorted(auth['allowed_cpus']),'explicit distinct CPU permission')
    require(type(auth['absolute_launch_command']) is str and bool(auth['absolute_launch_command']),'explicit command')
    require(review.get('approved') is True and review['plan_digest']==digest(plan) and
            review['authorization_digest']==digest(auth),'final full-map binding review')
    return selected


def bound_fingerprint(task,source,numeric):
    hash_id(numeric)
    return fingerprint('h4-b0i-map240-wrapper-v1',dict(namespace=NAMESPACE,
        method=science.METHOD,semantics=science.SEMANTICS,source=source,
        original_source=science.ORIGINAL_SOURCE,template=task['template'],
        input=task['input_identity'],input_file={k:task['input_file'][k] for k in ('bytes','sha256')},
        axis=task['axis'],control='ordinary',tail_scalar=task['tail_identity'],
        phase_angle=task['phase_angle'],numeric_circuit_fingerprint=numeric,
        trajectory_seed=None,trajectory_index=None,old_B0_cache_reuse=False))


def verify_reuse_match(task,old_task,receipt):
    """No compiler invocation; acceptance is conditional on fresh numeric FP too."""
    require(task['action']=='VERIFY_REUSE' and key(task) in PILOT_KEYS,'only the six pilot wrappers may be reused')
    for field in ('geometry','prefix','axis','template','input_file','input_identity','tail_identity','phase_angle','original_B0_template'):
        require(fingerprint('h4-b0i-reuse-field-v1',{field:task[field]})==
                fingerprint('h4-b0i-reuse-field-v1',{field:old_task[field]}),
                'pilot reuse exact science/input mismatch: '+field)
    require(old_task['source_commit']==PILOT_SOURCE and receipt['source_commit']==PILOT_SOURCE and
            receipt['task_id']==old_task['task_id'] and receipt['status']=='COMPLETE' and
            receipt['old_cache_reuse'] is False,'complete measured pilot receipt only')
    require(receipt['validation_wrapper_fingerprint']==science.bound_fingerprint(
        old_task['template'],old_task['tail_identity'],old_task['axis'],old_task['input_identity'],
        PILOT_SOURCE,receipt['numeric_circuit_fingerprint']),'pilot wrapper identity')
    require(task['reuse_receipt']['old_task_id']==old_task['task_id'] and
            task['reuse_receipt']['expected_numeric_fingerprint']==receipt['numeric_circuit_fingerprint'],'explicit namespace/source mapping')


class CallLedger:
    def __init__(self,tasks,*,reserve=lambda row:None):
        self.ids={t['task_id'] for t in tasks if t['action']=='COMPILE'}
        self.reserved=[];self.reserve=reserve
    def before_call(self,task_id):
        require(task_id in self.ids and task_id not in self.reserved and len(self.reserved)<234,'unknown/reuse/repeated/excess compile refused')
        row=dict(task_id=task_id,call=len(self.reserved)+1,status='RESERVED_BEFORE_INVOCATION_NO_REFUND')
        self.reserve(row);self.reserved.append(task_id)
        return row
