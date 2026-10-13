"""Schema-only preflight and separate one-shot saved-posthoc authorization."""
from pathlib import Path
import hashlib,json,os,platform,resource,subprocess,sys,re
from .c1_saved_oracle_spec_v1 import (BASE_COMMIT,SCOPE,PREPARATION,OUTPUT,COMPARISONS,SEAL,
                                     LIMITS,THREADS,ApprovalError,InputError,SOURCE_FILES,OLD_SOURCE_LOCK,SUMMARY,PRIOR_STOP)
from .c1_saved_oracle_schema_v1 import validate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    path=Path(path)
    if path.stat().st_size>8*2**20:raise InputError('INPUT_FILE_SIZE_CAP')
    return json.loads(path.read_bytes(),parse_constant=lambda s:(_ for _ in ()).throw(InputError('NONFINITE_JSON_LITERAL')))

def approval(a,identities=None):
    if not isinstance(a,dict) or a.get('type')!='C1_SAVED_ORACLE_POSTHOC_APPROVAL_V1' or a.get('status')!='APPROVED':
        raise ApprovalError('SEPARATE_EXPLICIT_POSTHOC_APPROVAL_REQUIRED')
    if a.get('scope')!=SCOPE or a.get('limits')!=LIMITS or not isinstance(a.get('explicit_user_approval'),str) or not a['explicit_user_approval'].strip():
        raise ApprovalError('APPROVAL_SCOPE_LIMITS_OR_USER_INSTRUCTION')
    if identities is not None and a.get('identities')!=identities:
        raise ApprovalError('APPROVAL_SOURCE_INPUT_PLAN_IDENTITY')

def runtime_identity():
    return {'python_version':platform.python_version(),'python_executable_SHA256':sha(Path(sys.executable).resolve()),
            'backend':'stdlib JSON/Fraction,serial,1thread,no numerical libraries'}

def preflight(root):
    root=Path(root).resolve();folder=root/PREPARATION
    plan=read(folder/'execution_plan.json');sources=read(folder/'source_manifest.json')
    inputs=read(folder/'input_manifest.json');runtime=read(folder/'runtime_lock.json')
    if set(sources.get('files',{}))!=set(SOURCE_FILES):raise InputError('SOURCE_MANIFEST_COVERAGE')
    if inputs.get('base_commit')!=BASE_COMMIT:raise InputError('INPUT_BASE_COMMIT')
    legacy=read(root/OLD_SOURCE_LOCK)
    if len(legacy.get('files',{}))!=20 or set(inputs.get('files',{}))!=set(legacy['files'])|{COMPARISONS,SEAL,SUMMARY,PRIOR_STOP,OLD_SOURCE_LOCK}:
        raise InputError('INPUT_MANIFEST_COVERAGE')
    if plan.get('scope')!=SCOPE or plan.get('limits')!=LIMITS or plan.get('output')!=OUTPUT or plan.get('base_commit')!=BASE_COMMIT:
        raise InputError('EXECUTION_PLAN_CONTRACT')
    source_commit=plan.get('source_commit','')
    if not re.fullmatch('[0-9a-f]{40}',source_commit):raise InputError('SOURCE_COMMIT_NOT_FROZEN')
    subprocess.run(['git','merge-base','--is-ancestor',source_commit,'HEAD'],cwd=root,check=True,capture_output=True)
    origin=subprocess.check_output(['git','remote','get-url','origin'],cwd=root,text=True).strip()
    if origin not in ('git@github.com:HIROMU1015/Partially-Randomized-Trotter.git',
                      'https://github.com/HIROMU1015/Partially-Randomized-Trotter.git'):
        raise InputError('HIROMU1015_REMOTE_REQUIRED')
    for p,want in sources['files'].items():
        if sha(root/p)!=want:raise InputError('SOURCE_HASH:'+p)
        old=subprocess.check_output(['git','show',source_commit+':'+p],cwd=root)
        if hashlib.sha256(old).hexdigest()!=want:raise InputError('FROZEN_SOURCE_COMMIT:'+p)
    for p,record in inputs['files'].items():
        b=(root/p).read_bytes()
        if hashlib.sha256(b).hexdigest()!=record['SHA256'] or len(b)!=record['bytes']:
            raise InputError('SAVED_INPUT_HASH:'+p)
        old=subprocess.check_output(['git','show',BASE_COMMIT+':'+p],cwd=root)
        if b!=old:raise InputError('SAVED_INPUT_BASE_COMMIT:'+p)
    if runtime_identity()!=runtime['identity']:raise InputError('RUNTIME_IDENTITY')
    schema=validate(read(root/COMPARISONS),read(root/SEAL))
    if (root/OUTPUT).exists():raise InputError('ONE_SHOT_OUTPUT_ALREADY_EXISTS')
    memory={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
            if line.startswith(('MemTotal:','MemAvailable:'))}
    if memory['MemAvailable']<LIMITS['memory_bytes']*2:raise InputError('AVAILABLE_MEMORY')
    identities={'source_commit':source_commit,'source_manifest_SHA256':sha(folder/'source_manifest.json'),
                'input_manifest_SHA256':sha(folder/'input_manifest.json'),
                'execution_plan_SHA256':sha(folder/'execution_plan.json'),'runtime_lock_SHA256':sha(folder/'runtime_lock.json')}
    return {'status':'IDENTITY_SCHEMA_PREFLIGHT_ONLY_PASS','identities':identities,'schema':schema,
            'runtime':runtime_identity(),'CPUs_available':len(os.sched_getaffinity(0)),
            'memory':memory,'chosen_workers':1,'chosen_threads':1,'GPU_used':False,
            'new_registered_metrics_evaluated':0,'eligibility_recomputed':False,'authorization_issued':False,
            'launch_allowed':False,'mandatory_stop':True}

def safe_write(out,name,value):
    encoded=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False).encode()+b'\n'
    used=sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    ceiling=LIMITS['output_bytes'] if name.endswith('failure.json') else LIMITS['output_bytes']-8192
    if used+len(encoded)>ceiling:raise InputError('OUTPUT_TOTAL_CAP')
    (out/name).write_bytes(encoded)

def authorize(root,path):
    if path is None:raise ApprovalError('SEPARATE_EXPLICIT_POSTHOC_APPROVAL_REQUIRED')
    a=read(path);approval(a)
    report=preflight(root);approval(a,report['identities'])
    for key in THREADS:os.environ[key]='1'
    resource.setrlimit(resource.RLIMIT_AS,(LIMITS['memory_bytes'],LIMITS['memory_bytes']))
    out=Path(root)/OUTPUT;out.mkdir(parents=True,exist_ok=False)
    receipt={'scope':SCOPE,'authorization':a,'preflight':report,'attempt_consumed':True,'mandatory_stop':True}
    safe_write(out,'launch_receipt.json',receipt)
    return out,receipt

def checker_receipt(root):
    root=Path(root);out=root/OUTPUT;receipt=read(out/'launch_receipt.json')
    if receipt.get('scope')!=SCOPE or receipt.get('attempt_consumed') is not True:
        raise ApprovalError('CONSUMED_POSTHOC_RECEIPT_REQUIRED')
    # Reuse the same identity checks without permitting a second output creation.
    a=receipt['authorization'];approval(a)
    folder=root/PREPARATION;plan=read(folder/'execution_plan.json')
    identities={'source_commit':plan['source_commit'],'source_manifest_SHA256':sha(folder/'source_manifest.json'),
                'input_manifest_SHA256':sha(folder/'input_manifest.json'),'execution_plan_SHA256':sha(folder/'execution_plan.json'),
                'runtime_lock_SHA256':sha(folder/'runtime_lock.json')}
    approval(a,identities)
    if runtime_identity()!=read(folder/'runtime_lock.json')['identity']:raise InputError('AUDIT_RUNTIME_IDENTITY')
    for p,want in read(folder/'source_manifest.json')['files'].items():
        if sha(root/p)!=want:raise InputError('AUDIT_SOURCE_HASH:'+p)
    for p,record in read(folder/'input_manifest.json')['files'].items():
        if sha(root/p)!=record['SHA256']:raise InputError('AUDIT_INPUT_HASH:'+p)
    for key in THREADS:os.environ[key]='1'
    resource.setrlimit(resource.RLIMIT_AS,(LIMITS['memory_bytes'],LIMITS['memory_bytes']))
    return out,receipt


def claim_audit(out):
    out=Path(out)
    encoded=json.dumps({'independent_audit_attempt_consumed':True,'retry':0,'mandatory_stop':True}).encode()+b'\n'
    used=sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    if used+len(encoded)>LIMITS['output_bytes']-8192:raise InputError('AUDIT_OUTPUT_CAP')
    with (out/'audit_attempt_consumed.json').open('xb') as f:f.write(encoded)
