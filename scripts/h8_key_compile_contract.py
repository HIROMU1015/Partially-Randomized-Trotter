"""Six selected H8 axes. Metadata-only preparation never authorizes science."""
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
INPUTS = 'references/h8_key_compile_inputs_v1.json'
CLOSURE = 'references/h8_key_compile_closure_v1.json'
SCOPE = 'H8_THREE_SELECTED_CANDIDATES_SIX_NATIVE_AXES_PARALLEL_PREPARATION_V2'
PROFILE = '622ce4a8391002558b877240b48d831d9bb0a72aeabba2e4d73bfec64b9c9f12'
COMPILER = '1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b'
PROVIDER = 'c31ef424175c02379eafebe8fc6c7548c1d9aabe'
MODEL = '8420b9f247c63f47f10f40ba5e875f5ed971e8948ea6748b9845698fe6bdb8b0'
PROVIDER_MAP = '9382176df9a1353c9f02b842edec5943dfbeb06b223387f6a99542e53c878bc0'
RUNTIME_MAP = '850568c8018943b91dc9dbb031198b684112f88f8870f5a217253c61ee5fb958'
PYTHON = '/home/AbeHiromu/projects/h6-environments/20261012/native-dependency-v2/venv/bin/python'
GIT = ['git','--no-optional-locks','-c','index.threads=1','-c','core.preloadIndex=false',
       '-c','core.fsmonitor=false','-c','maintenance.auto=false','-c','gc.auto=0']
THREADS = ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS',
           'NUMBA_NUM_THREADS','QISKIT_NUM_PROCS','RAYON_NUM_THREADS')
CAPS = dict(snapshot_load=1,saved_preparation_decode=1,prefix_view=2,native_evolution_build=3,
    measured_wrapper_build=6,probe_wrapper_copy=6,control_probe=12,qpy_dump=6,qpy_load=6,
    compile=6,model_error_check=6,new_signal=0,new_reference=0,tail_scalar=0,
    SCF=0,DF=0,state_solver=0,PR_sampling=0,fit=0,cost_reaccount=0,extra_compile=0)
CASE_PREPARATION_CAPS = dict(native_evolution_build=1,measured_wrapper_build=2,
    probe_wrapper_copy=2,control_probe=4,qpy_dump=2)


def require(value, reason):
    if not value: raise ValueError(reason)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()


def read(path):
    def pairs(items):
        out={}
        for k,v in items:require(k not in out,'DUPLICATE_JSON_KEY');out[k]=v
        return out
    def number(s):
        v=float(s);require(math.isfinite(v),'NONFINITE_JSON');return v
    def reject(s):raise ValueError('NONFINITE_JSON:'+s)
    p=Path(path);require(not p.is_symlink(),'SYMLINK_JSON')
    return json.loads(p.read_bytes(),object_pairs_hook=pairs,parse_float=number,parse_constant=reject)


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())


def binding():return read(ROOT/INPUTS)
def source_paths():return read(ROOT/CLOSURE)['paths']


def plan():
    b=binding()
    return dict(schema='H8_KEY_COMPILE_PLAN_V2',scope=SCOPE,cases=b['cases'],axes=b['axes'],
        input_binding_sha256=sha(ROOT/INPUTS),source_closure_sha256=sha(ROOT/CLOSURE),
        science=dict(geometry='Linear H8 1.0 Angstrom STO-3G',modes=16,sector=4900,DF_rank=28,T=.8,
                     fragment_order='SAVED_SIGNED_GENERATION_ORDER',control='symmetric_directional',
                     S4='NEGATIVE_MIDDLE_NONMERGED_THREE_S2_YOSHIDA',reference='UNROTATED_SAVED_FULL_H'),
        scalar=dict(B0='ORIGINAL_ONCE',B0I='ORIGINAL_PLUS_OWN_PREFIX_SAVED_FULL_FOCK_IDENTITY_ONCE'),
        compiler_options=b['compiler_options'],compiler_hash=COMPILER,profile_sha256=PROFILE,
        provider_SOURCE=PROVIDER,model_sha256=MODEL,call_caps=CAPS,
        compile_concurrency=1,preparation_concurrency=6,max_cases_live=3,
        preparation=dict(case_build_concurrency=3,worker_pairs=3,workers_per_pair=2,
            wrapper_build='IN_AXIS_WORKER_WITH_VALIDATION_FINGERPRINT_AND_QPY',
            case_coordinator='SHARES_FIRST_CPU_OF_OWN_PAIR_WHILE_WAITING',
            call_reservations='PARENT_BEFORE_CASE_FORK_NONREFUNDABLE',
            per_case_caps=CASE_PREPARATION_CAPS,compile_admission='ALL_THREE_PREPARATIONS_PASS'),
        axis_order='PRIORITY_CASE_THEN_COSINE_SINE',check_before_next_compile=True,
        handover=dict(policy='VERIFIED_OLD_PREPARATION_AND_COMPLETED_PREFIX_ONLY',
            old_run_STOP='PRESERVE_WITH_NATIVE_EXIT_AND_CLEANUP',
            release='BOUND_APPEND_ONLY_AFTER_NEW_DRIVER_READY_AND_OLD_EXIT',
            accounting='NEW_ACTUAL_PLUS_REUSE_SEPARATE_PRIOR_RESERVATIONS_NONREFUNDABLE'),
        model_PASS='abs(prediction-measured)/measured < 1/50',control_PASS='finite error < 1e-8',
        resource_policy=dict(agent_imposed_CPU_RAM_CPUtime_wall_caps=None,OS_limits='INHERIT_WITHOUT_CHANGE',
            PSI='OBSERVE_AND_REESTABLISH_NO_TRANSIENT_STOP',unrelated_worktrees='OBSERVATION_ONLY',
            fork='SINGLE_THREAD_PARENT_REQUIRED',threads_per_worker=1,
            estimates=dict(aggregate_peak_RSS_bytes=3*64*2**30,headroom_bytes=16*2**30,
                           stage_bytes=2*2**30,inodes=1024,monitor_growth='APPEND_JSONL_BYTES_PER_SECOND_REPORTED',
                           wall='UNMEASURED_NO_DEADLINE')),
        mismatch='PRESERVE_RAW_THEN_STOP_NO_REFIT_NO_EXTRA_COMPILE',
        one_shot=True,automatic_retry=False,next_stage_authorized=False,cost_reaccount_authorized=False,
        full_grid_measured_optimum=False,rigorous_bias_guarantee=False,PR_population_guarantee=False,
        approved=False,runtime_authorization=False,allowed_cpus=[],absolute_launch_command=None,
        mandatory_STOP=True)


def axis_check(prediction, measured):
    require(type(measured)is int and measured>0,'POSITIVE_INTEGER_RZ_REQUIRED')
    predicted=Fraction(prediction);require(predicted>0,'POSITIVE_PREDICTION')
    error=abs(predicted-measured)/measured
    return dict(exact_prediction=str(predicted),measured_RZ=measured,exact_relative_error=str(error),
                relative_error_percent=float(error*100),PASS=error<Fraction(1,50))


def authorize(config, grant, *, affinity=None):
    source=config.get('SOURCE')
    require(type(source)is str and len(source)==40 and all(x in '0123456789abcdef' for x in source) and
        config.get('execution_sealed')is True and grant.get('schema')=='H8_KEY_COMPILE_GRANT_V2' and
        grant.get('approved')is True and grant.get('runtime_authorization')is True and
        grant.get('execution_sealed')is True and grant.get('approval_text') and
        grant.get('scope')==SCOPE and grant.get('SOURCE')==source and
        grant.get('plan_sha256')==config.get('plan_sha256')==digest(plan()) and
        grant.get('config_sha256')==digest(config) and grant.get('call_caps')==CAPS and
        grant.get('one_shot')is True and grant.get('automatic_retry')is False and
        grant.get('cost_reaccount_authorized')is False and grant.get('next_stage_authorized')is False,
        'NEW_EXPLICIT_SEALED_SIX_AXIS_GRANT_REQUIRED')
    cpus=grant.get('allowed_cpus');workers=grant.get('worker_cpus')
    available=os.sched_getaffinity(0) if affinity is None else affinity
    require(type(workers)is list and len(workers)==6 and cpus==[grant.get('driver_cpu'),*workers] and
        all(type(x)is int and x>=0 for x in cpus) and len(set(cpus))==7 and set(cpus)<=available,
        'EXPLICIT_SEVEN_DISJOINT_OWN_CPUS_REQUIRED')
    return cpus


def home(path):
    p=Path(path);require(p.is_absolute() and p.resolve()==p and p.is_relative_to('/home/AbeHiromu'),
                        'CANONICAL_HOME_PATH_REQUIRED');return p


def paths(config):
    out,control,lock=(home(config[k]) for k in ('output','control','lock'))
    require(len({out,control,lock})==3 and len({out.parent,control.parent,lock.parent})==1,'DISJOINT_STAGE_PATHS')
    require(not any(p.exists() for p in (out,control,lock)),'ONE_SHOT_PATH_USED')
    protected=[ROOT,home(config['provider_root']),*[home(r['path']) for r in config['files']]]
    require(all(not p.is_relative_to(q) and not q.is_relative_to(p) for p in (out,control,lock) for q in protected),
            'OUTPUT_OVERLAPS_DEPENDENCY')
    return out,control,lock


def verify(config, *, runtime=False, git=True):
    require(config['schema']=='H8_KEY_COMPILE_CONFIG_V2' and config['plan_sha256']==digest(plan()) and
        config['provider_source']==PROVIDER and config['expected_profile_digest']==PROFILE and
        config['python']==PYTHON and sha(config['python'])==config['python_sha256'],'PLAN_PROFILE_BINDING')
    require(digest(config['provider_hashes'])==PROVIDER_MAP and digest(config['runtime_file_hashes'])==RUNTIME_MAP,
            'PINNED_PROVIDER_RUNTIME_MAPS')
    require(set(config['source_hashes'])==set(source_paths()),'EXACT_SOURCE_CLOSURE')
    if git:require(subprocess.check_output(GIT+['rev-parse','HEAD'],cwd=ROOT).decode().strip()==config['SOURCE'],
                   'SOURCE_HEAD_CHANGED')
    for name,h in config['source_hashes'].items():
        p=Path(name);require(not p.is_absolute() and '..' not in p.parts and sha(ROOT/p)==h,'SOURCE_BYTES_CHANGED:'+name)
        if git:require(hashlib.sha256(subprocess.check_output(GIT+['show',config['SOURCE']+':'+name],cwd=ROOT)).hexdigest()==h,
                       'SOURCE_GIT_BYTES_CHANGED:'+name)
    specs=binding()['inputs'];by={x['logical_name']:x for x in config['files']}
    require(len(by)==len(config['files'])==len(specs) and set(by)=={x['logical_name'] for x in specs},'EXACT_INPUT_SET')
    require(config['input_mapping']=={k:r['path'] for k,r in by.items()},'EXACT_INPUT_MAPPING')
    for e in specs:
        r=by[e['logical_name']];p=home(r['path'])
        require((r['bytes'],r['sha256'])==(e['bytes'],e['sha256']) and p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],
                'INPUT_BYTES_CHANGED:'+e['logical_name'])
    if runtime:
        for name,h in config['provider_hashes'].items():
            require(sha(home(config['provider_root'])/name)==h,'PROVIDER_CHANGED:'+name)
        for name,h in config['runtime_file_hashes'].items():require(sha(Path(name))==h,'RUNTIME_CHANGED:'+name)
    return dict(status='PASS',source_files=len(source_paths()),inputs=len(by),full_runtime_checked=runtime,
                unrelated_worktrees_enforced=0)


class Permit:
    def __init__(self,config,grant,output,*,synthetic=False,affinity=None):
        if not synthetic:authorize(config,grant,affinity=affinity)
        self.config,self.grant,self.output=config,grant,Path(output)
        self.calls={k:0 for k in CAPS}
        self.new_calls={k:0 for k in CAPS};self.reused_calls={k:0 for k in CAPS}
    def require(self,scope):require(scope==SCOPE,'SCOPE_MISMATCH')
    def take(self,key):
        require(self.calls[key]<CAPS[key],'CALL_CAP:'+key);self.calls[key]+=1
        save(self.output/('reservation_'+key+'_'+str(self.calls[key])+'.json'),dict(call=key,ordinal=self.calls[key]))
        self.new_calls[key]+=1

    def reuse(self,key):
        require(self.calls[key]<CAPS[key],'REUSE_CALL_CAP:'+key);self.calls[key]+=1
        save(self.output/('reuse_'+key+'_'+str(self.calls[key])+'.json'),dict(call=key,ordinal=self.calls[key]))
        self.reused_calls[key]+=1


def complete(rows, calls, source, *, reused=()):
    axes=binding()['axes'];require(calls==CAPS and len(rows)==6,'EXACT_SIX_AXIS_COUNTS')
    for i,(row,axis) in enumerate(zip(rows,axes)):
        case=next(x for x in binding()['cases'] if x['id']==axis['case_id'])
        require((row['SOURCE']==source or (i<len(reused) and row==reused[i])) and row['preparation_binding']==binding()['prefix_preparations'][str(case['prefix'])] and
            (row['case_id'],row['axis'])==(axis['case_id'],axis['axis']) and row['native_exit_code']==0 and
            row['reaped']is True and row['compiler_hash']==COMPILER and row['profile_sha256']==PROFILE and
            row['model_sha256']==MODEL and row['fingerprint_before']==row['fingerprint_after'] and
            0<=row['max_control_error']<1e-8 and row['model_check']==axis_check(axis['exact_prediction'],row['measured_RZ']) and
            row['model_check']['PASS']is True,'COMPLETE_BINDING_OR_MODEL_ERROR')
    return dict(status='PASS_ONLY_SIX_REGISTERED_AXES_MANDATORY_STOP',full_grid_measured_optimum=False,
                cost_reaccount_authorized=False,next_stage_authorized=False)
