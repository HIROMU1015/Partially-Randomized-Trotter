"""Independent A/B/D one-shot gates. Invalid grants fail before input loading."""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/trotterlib'))
import pr_cost_integration as core
require,digest=core.require,core.digest
REGISTRY='references/pr_cost_integration_evidence_v1.json'
STAGES=('A_COST','D_ACCOUNTING','B_ANALYSIS')
SOURCES=('src/trotterlib/pr_cost_integration.py','scripts/pr_cost_contract.py',
         'scripts/pr_cost_native.py','scripts/pr_cost_saved.py','scripts/run_pr_cost_integration.py',
         'scripts/prepare_pr_cost_integration.py',REGISTRY,
         'tests/test_pr_cost_integration.py','tests/test_pr_cost_native_artificial.py',
         'src/trotterlib/h8_six_axis_reaccount.py','src/trotterlib/h8_phase_c_accounting.py')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def read(path):
    def pairs(items):
        out={}
        for k,v in items:
            require(k not in out,'DUPLICATE_JSON_KEY');out[k]=v
        return out
    def number(text):
        value=float(text);require(math.isfinite(value),'NONFINITE_JSON');return value
    path=Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size<=64*2**20,'BOUNDED_JSON_FILE')
    return json.loads(path.read_bytes(),object_pairs_hook=pairs,parse_float=number,
        parse_constant=lambda x:(_ for _ in ()).throw(ValueError('NONFINITE_JSON:'+x)))


def save(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def plan(system,stage):
    require(system in ('H4','H6','H8') and stage in STAGES,'SYSTEM_STAGE')
    registry=read(ROOT/REGISTRY);entry=registry['systems'][system]
    calls=entry['proposed_calls'] if stage=='A_COST' else dict(
        saved_scalar_pass=1,signal=0,trajectory=0,wrapper_compile=0,control_probe=0,fit=0,quantum_shots=0)
    return dict(schema='PR_COST_PREPARATION_V1',system=system,stage=stage,
        scope=system+'_'+stage+'_20261018',binding=entry['binding'],registry_sha256=sha(ROOT/REGISTRY),
        proposed_calls=calls,proposal_is_authorization=False,epsilons=list(core.EPS),
        bias_modes=list(core.MODES),allowance_factors=[1.,10.],P=0,
        A_acquisition_endpoints='saved multiplier1 flags; multiplier10 accounting only after separate authorization',
        cells=entry['cells'] if stage=='A_COST' else None,
        selected_cells=entry['acquisition']['selected'] if stage=='A_COST' else None,
        n=entry['n_proposed'] if stage=='A_COST' else None,
        original_n32_and_new_n8_separate=True,conditional_mean_not_unconditional=True,
        automatic_retry=False,next_stage_authorized=False,mandatory_STOP=True,
        resource_policy=dict(agent_imposed_CPU_RAM_worker_wall_caps=None,
            selection='fresh available resources and known concurrent jobs; exclusion of their assigned CPUs',
            internal_threads='1 per compile worker to avoid nested parallelism',PSI='observation only'),
        pending_technical_seal=True,absolute_launch_command=None)


def invalid_grant(system,stage):
    p=plan(system,stage)
    return dict(schema='INVALID_GRANT_DRAFT',scope=p['scope'],system=system,stage=stage,
        approved=False,approved_by_user=False,runtime_authorization=False,one_shot=True,
        automatic_retry=False,next_stage_authorized=False,SOURCE=None,
        plan_digest=digest(p),config_digest=None,allowed_cpus=[],absolute_launch_command=None)


def authorize(grant,system,stage):
    require(stage in STAGES and system in ('H4','H6','H8'),'SYSTEM_STAGE')
    require(grant.get('scope')==system+'_'+stage+'_20261018' and grant.get('system')==system
            and grant.get('stage')==stage,'EXACT_INDEPENDENT_A_B_D_SCOPE')
    require(all(grant.get(k) is True for k in ('approved','approved_by_user','runtime_authorization','one_shot'))
            and grant.get('automatic_retry') is False and grant.get('next_stage_authorized') is False,
            'EXPLICIT_SCOPED_EXECUTION_APPROVAL_REQUIRED')
    require(grant.get('user_approval_receipt_sha256') and len(grant.get('SOURCE',''))==40,
            'SOURCE_AND_USER_APPROVAL_RECEIPT')


def home_path(path):
    path=Path(path).resolve()
    require(path.is_relative_to(Path('/home/AbeHiromu')),'HOME_ONLY_OUTPUT')
    return path


def verify_byte_rows(rows):
    roles=set()
    for row in rows:
        require(row['role'] not in roles,'UNIQUE_INPUT_ROLE');roles.add(row['role'])
        path=Path(row['path'])
        require(not path.is_symlink() and path.is_file() and path.stat().st_size==row['bytes']
                and sha(path)==row['sha256'],'INPUT_BYTES_CHANGED:'+row['role'])
    return roles


def verify_source(root,source,hashes):
    for name,h in hashes.items():
        relative=Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts,'SOURCE_RELATIVE_PATH')
        require(sha(root/name)==h,'SOURCE_BYTES_CHANGED:'+name)
        value=subprocess.check_output(['git','--no-optional-locks','-c','maintenance.auto=false',
            '-c','gc.auto=0','-C',str(root),'show',source+':'+name])
        require(hashlib.sha256(value).hexdigest()==h,'SOURCE_GIT_BINDING:'+name)


def verify(config,grant,system,stage,*,root=ROOT,final=False):
    authorize(grant,system,stage)
    p=plan(system,stage)
    require(grant['plan_digest']==digest(p) and grant['config_digest']==digest(config),'SEALED_PLAN_CONFIG')
    require(config.get('technical_seal_complete') is True,'TECHNICAL_SEAL_REQUIRED')
    core.same_binding(config['binding'],p['binding'])
    require(set(config['source_hashes'])==set(SOURCES),'COMPLETE_OWN_SOURCE_CLOSURE')
    verify_source(root,grant['SOURCE'],config['source_hashes'])
    require(config.get('inputs') and config.get('runtime_hashes'),'SEALED_INPUT_RUNTIME_CLOSURE')
    roles=verify_byte_rows(config['inputs'])
    for name,h in config['runtime_hashes'].items():
        require(sha(name)==h,'RUNTIME_CHANGED')
    require({'user_approval_receipt','compiler_profile','accounting_payload'}<=roles
            if stage!='A_COST' else {'user_approval_receipt','compiler_profile','A_signal','native_config','seed_exclusions'}<=roles,
            'REQUIRED_PRIVATE_ROLES')
    approval=next(x for x in config['inputs'] if x['role']=='user_approval_receipt')
    require(approval['sha256']==grant['user_approval_receipt_sha256'],'APPROVAL_RECEIPT_BINDING')
    require(sha(config['python'])==config['python_sha256'],'PYTHON_BINARY_BINDING')
    if stage=='A_COST':
        require(config.get('provider_closure') and config.get('native_config') and config.get('preparation_identities'),
                'NATIVE_SOURCE_PREPARATION_CLOSURE')
        for base,files in config['provider_closure'].items():
            for name,h in files.items():
                relative=Path(name)
                require(not relative.is_absolute() and '..' not in relative.parts,'PROVIDER_RELATIVE_PATH')
                require(sha(Path(base)/relative)==h,'PROVIDER_BYTES_CHANGED')
        covered={(str(Path(x['path']).resolve()),x['sha256'],x['bytes']) for x in config['inputs']}
        old=read(config['native_config'])
        require(old.get('inputs'),'ORIGINAL_NATIVE_INPUT_CLOSURE')
        for row in old['inputs']:
            require((str(Path(row['path']).resolve()),row['sha256'],row['bytes']) in covered,
                    'UNCOVERED_ORIGINAL_NATIVE_INPUT:'+row['role'])
        for directory in config['import_paths']:
            require(any(Path(directory).resolve().is_relative_to(Path(base).resolve())
                        for base in config['provider_closure']),'UNCOVERED_PROVIDER_IMPORT_PATH')
        require(config.get('seed_exclusions_complete') is True,'OLD_MASTER_DERIVED_SEEDS_RECEIPT')
    allowed=grant['allowed_cpus']
    require(allowed and len(allowed)==len(set(allowed)) and set(allowed)<=os.sched_getaffinity(0), 'FRESH_ALLOWED_CPUS')
    require(set(allowed).isdisjoint(config['concurrent_job_cpus']),'CONCURRENT_JOB_CPU_EXCLUSION')
    output=home_path(config['output']);lock=home_path(config['one_shot_lock'])
    require(output!=lock,'SEPARATE_OUTPUT_ONE_SHOT')
    if not final:
        require(not output.exists() and not lock.exists(),'UNUSED_OUTPUT_ONE_SHOT')
    f=fresh(output.parent)
    require(final or (f['memory_available_bytes']>=config['memory_estimate_bytes']
            and f['disk_free_bytes']>=config['output_estimate_bytes']
            and f['free_inodes']>=config['inode_estimate']),'RESOURCE_ADMISSION_ESTIMATES')
    return dict(status='PASS',SOURCE=grant['SOURCE'],fresh=f)


def fresh(directory):
    directory=Path(directory)
    while not directory.exists():directory=directory.parent
    mem={line.split(':')[0]:int(line.split()[1])*1024 for line in
         Path('/proc/meminfo').read_text().splitlines() if len(line.split())>=2 and line.split()[1].isdigit()}
    st=os.statvfs(directory)
    return dict(time=time.time(),affinity=sorted(os.sched_getaffinity(0)),
        memory_available_bytes=mem['MemAvailable'],disk_free_bytes=shutil.disk_usage(directory).free,
        free_inodes=st.f_favail,memory_pressure=Path('/proc/pressure/memory').read_text(),
        transient_pressure_is_STOP=False,quota='requires fresh site-specific read-only receipt before seal')


class Budget:
    def __init__(self,directory,caps):self.directory=Path(directory);self.caps=caps;self.used={}
    def take(self,kind,count=1):
        require(type(count) is int and count>0 and kind in self.caps,'REGISTERED_CALL_KIND')
        value=self.used.get(kind,0)+count
        require(value<=self.caps[kind],'CALL_BUDGET_EXCEEDED:'+kind)
        # Each task owns its directory; no shared counter races across workers.
        save(self.directory/(kind+'_'+str(value)+'_reserved.json'),dict(kind=kind,count=count,total=value,nonrefundable=True))
        self.used[kind]=value


def first_stop(directory,reason):
    try:save(Path(directory)/'STOP.json',dict(reason=reason,first_reason_preserved=True,retry=False))
    except FileExistsError:pass


def claim_attempt(path,source,scope):
    """Exclusive, persistent attempt marker; never removed on STOP or exception."""
    save(path,dict(SOURCE=source,scope=scope,nonrefundable_attempt=True))
