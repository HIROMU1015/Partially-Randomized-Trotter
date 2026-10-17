#!/usr/bin/env python3
"""Two independent, explicitly authorized one-shot saved-value science stages."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/trotterlib'))
from h8_rare_tail_selection import require, digest
from h8_saved_analysis import (STAGES, CANDIDATES, RARE, EPS, MODES,
                              validate_stat_contract, rare_endpoint, selection_analysis)
from h8_saved_analysis_adapter import MANIFEST, read_fixed, projection

FILES=('src/trotterlib/h8_saved_analysis.py','src/trotterlib/h8_rare_tail_selection.py',
       'scripts/run_h8_saved_analysis.py','scripts/h8_saved_analysis_adapter.py',
       'scripts/audit_h8_rare_tail_selection_evidence.py',
       'references/h8_saved_analysis_inputs_v1.json','references/h8_rare_tail_selection_inputs_v1.json',
       'tests/test_h8_saved_analysis.py','tests/test_h8_rare_tail_selection_readiness.py')


def sha(b):return hashlib.sha256(b).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def write_new(path,value):
    with path.open('x') as f:f.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def runtime():
    import fractions, decimal, random, json as jm, gzip, csv, _decimal, _random, math, _hashlib
    paths={str(Path(sys.executable).resolve())}
    modules=(fractions,decimal,random,jm,gzip,csv,_decimal,_random,math,_hashlib)
    builtins=[]
    for m in modules:
        if getattr(m,'__file__',None):paths.add(m.__file__)
        else:
            require(m.__spec__.origin=='built-in','UNIDENTIFIED_RUNTIME_MODULE')
            builtins.append(m.__name__)
    return dict(python=sys.version,files={p:sha(Path(p).read_bytes()) for p in sorted(paths)},
                modules_in_hashed_python_executable=sorted(builtins),
                numeric='EXACT_INTEGER_FRACTION; DECIMAL80_SQRT_AND_BOUNDARY; STDLIB_RANDOM_MT19937')


def authorization(config,grant,approval,closure,root=ROOT):
    # Reject UNAPPROVED drafts before opening science or Git input blobs.
    require(config.get('authorization_status')=='APPROVED' and
            config.get('runtime_authorization') is True and grant.get('valid') is True,
            'EXPLICIT_SCIENTIFIC_EXECUTION_APPROVAL_REQUIRED')
    stage=config.get('stage');require(stage in STAGES,'STAGE')
    require(grant.get('stage')==stage and approval.get('stage')==stage and
            grant.get('one_shot') is True and grant.get('config_sha256')==digest(config) and
            grant.get('approval_sha256')==digest(approval) and grant.get('SOURCE')==config['SOURCE'],
            'STAGE_SOURCE_CONFIG_APPROVAL_BINDING')
    require(approval.get('authorized') is True and approval.get('user_instruction')==
            '実際に実行するところまで行ってよい' and approval.get('scope_sha256')==digest(config['analysis'])
            and isinstance(approval.get('scope_confirmation'),str) and approval['scope_confirmation'],
            'USER_APPROVAL_SCOPE_BINDING')
    require(re.fullmatch('[0-9a-f]{40}',config['SOURCE']) and config['one_shot_id']==grant.get('one_shot_id')
            and re.fullmatch('[A-Z0-9_]{8,100}',config['one_shot_id']), 'SOURCE_ONESHOT_SCHEMA')
    require(digest(closure)==config['closure_sha256'] and closure['SOURCE']==config['SOURCE']
            and set(closure['files'])==set(FILES),'SOURCE_CLOSURE')
    for path,expected in closure['files'].items():
        b=subprocess.check_output(['git','--no-optional-locks','-C',str(root),'show',
                                   config['SOURCE']+':'+path],stderr=subprocess.PIPE)
        require(sha(b)==expected and sha((root/path).read_bytes())==expected,'SOURCE_DRIFT:'+path)
    require(config['input_manifest_sha256']==sha((root/MANIFEST).read_bytes()),'INPUT_MANIFEST_DRIFT')
    require(config['runtime']==runtime(),'PYTHON_RUNTIME_DRIFT')
    a=config['analysis']
    require(a['numeric_precision']=='FRACTION_EXACT_DECIMAL80' and
            a['new_sampling']==0 and a['new_compile']==0 and a['new_shot_calculation']==0,
            'SCIENCE_SCOPE')
    if stage=='RARE_SENSITIVITY':
        require(a['candidate_set']==list(RARE) and a['epsilon_scope']==list(EPS)
                and a['bias_mode']==list(MODES) and a['allowance_scope']==['1.0','10.0']
                and a['mode']=='TWO_VARIABLE_BOUNDARY' and a['unknown_cost_assumptions']=={}
                and a['comparator_scope']=='strengthened184','RARE_FIXED_BOUNDARY_SCOPE')
    else:
        require(a['candidate_set']==list(CANDIDATES) and a['epsilon_scope']==list(EPS[:4])
                and a['bias_mode']==[MODES[0]] and a['allowance_scope']==['1.0']
                and a['eligibility_mask']=='SAVED_FIXED_NO_POSTHOC_CANDIDATE_REMOVAL',
                'SELECTION_MINIMAL_SCOPE')
        validate_stat_contract(a)


def run(config,grant,approval,closure,*,projector=projection,reader=read_fixed,root=ROOT,
        execution_root=None,authorization_paths=None):
    authorization(config,grant,approval,closure,root)
    auth_bytes={str(p):Path(p).read_bytes() for p in (authorization_paths or ())}
    home=(root.parent/'execution') if execution_root is None else Path(execution_root)
    home.mkdir(parents=True,exist_ok=True)
    out=home/config['stage'].lower()/config['one_shot_id']
    require(not out.exists(),'OUTPUT_NAMESPACE_COLLISION')
    registry=home/'registry';registry.mkdir(exist_ok=True)
    lock=registry/(config['one_shot_id']+'.json')
    require(not lock.exists(),'ONE_SHOT_ALREADY_CONSUMED')
    write_new(lock,dict(stage=config['stage'],config_sha256=digest(config),created=now(),
                        status='RESERVED_NOT_REFUNDABLE'))
    out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();error=None;result=None;dispatch=0
    try:
        write_new(out/'authorization_binding.json',dict(config=config,grant=grant,approval=approval,
                                                       closure=closure,grant_is_cryptographic_signature=False))
        raw=reader()
        write_new(out/'fresh_gate.json',dict(status='PASS',SOURCE=config['SOURCE'],
            source_closure_sha256=digest(closure),input_files=len(raw),runtime=runtime(),
            effective_CPU_count=len(os.sched_getaffinity(0)),host_memory_observation=
                next((l.strip() for l in Path('/proc/meminfo').read_text().splitlines()
                      if l.startswith('MemAvailable:')),'UNAVAILABLE'),
            timestamp=now(),resource_competition_STOP=False,other_worktrees_frozen=False))
        data=projector(raw)
        scoped=[e for e in data['endpoints'] if e['epsilon'] in config['analysis']['epsilon_scope']
                and e['bias_kind'] in config['analysis']['bias_mode']
                and e['factor'] in config['analysis']['allowance_scope']]
        mask={e['id']:{c:v['eligible'] for c,v in e['candidates'].items()} for e in scoped}
        if 'saved_eligibility_mask_sha256' in config['analysis']:
            require(digest(mask)==config['analysis']['saved_eligibility_mask_sha256'],
                    'SAVED_ELIGIBILITY_MASK_DRIFT')
        dispatch=1
        write_new(out/'science_dispatch.json',dict(stage=config['stage'],count=1,time=now()))
        if config['stage']=='RARE_SENSITIVITY':
            rows=[]
            for e in data['endpoints']:
                for c in RARE:
                    ep=dict(e,candidate=c,**e['candidates'][c])
                    rows.append(rare_endpoint(data['laws'][c],data['by'][c],data['pilot'][c],ep,{}))
            result=dict(stage=config['stage'],status='COMPLETED_CONDITIONAL_BOUNDARIES',endpoints=rows,
                        unknown_costs_imputed=False,original_costs_reaccounted=False)
        else:
            a=config['analysis']
            endpoints=[e for e in data['endpoints'] if e['epsilon'] in a['epsilon_scope'] and
                       e['bias_kind'] in a['bias_mode'] and e['factor'] in a['allowance_scope']]
            require(len(endpoints)==4,'FOUR_ENDPOINTS')
            result=selection_analysis(data['by'],endpoints,a,data['fixed'])
            result['stage']=config['stage']
        result['provenance']=data['provenance'];result['SOURCE']=config['SOURCE']
        result['config_sha256']=digest(config)
        write_new(out/'result.json',result)
        # Post gate verifies own immutable sources, all public inputs, runtime and caller bytes.
        authorization(config,grant,approval,closure,root)
        require(all(Path(p).read_bytes()==b for p,b in auth_bytes.items()),'AUTHORIZATION_FILE_DRIFT')
        after=reader()
        require({k:sha(v) for k,v in raw.items()}=={k:sha(v) for k,v in after.items()},'POST_INPUT_DRIFT')
        write_new(out/'final_gate.json',dict(status='PASS_MANDATORY_STOP',science_dispatch_count=1,
            result_sha256=sha((out/'result.json').read_bytes()),SOURCE=config['SOURCE'],
            config_sha256=digest(config),input_files=len(after),new_sampling=0,new_compile=0,
            new_shot_calculation=0,next_stage_authorized_by_this_runner=False,first_STOP=None,time=now()))
    except BaseException as exc:
        error=f'{type(exc).__name__}:{exc}'
        first=out/'first_STOP.json'
        if not first.exists():write_new(first,dict(status='STOP',first_reason=error,time=now()))
    finally:
        write_new(out/'terminal.json',dict(status='STOP' if error else 'PASS_MANDATORY_STOP',
            first_reason=error,native_return_code=1 if error else 0,science_dispatch_count=dispatch,
            wall_seconds=time.monotonic()-start,pid=os.getpid(),time=now(),
            automatic_retry=0,automatic_next_stage=0,shared_environment_changes=0))
    if error:raise RuntimeError(error)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','grant','approval','closure'):p.add_argument('--'+name)
    p.add_argument('--execute',action='store_true');a=p.parse_args()
    if not a.execute:
        print(json.dumps(dict(stages=STAGES,authorization='EXPLICIT_SEPARATE_ONE_SHOT_REQUIRED',
                              science_dispatch_count=0)));return 0
    try:
        # Open authorization metadata first. No analysis input access on rejection.
        require(all(getattr(a,k) for k in ('config','grant','approval','closure')),'AUTHORIZATION_FILES_REQUIRED')
        c=json.loads(Path(a.config).read_text());g=json.loads(Path(a.grant).read_text())
        require(c.get('authorization_status')=='APPROVED' and g.get('valid') is True,
                'EXPLICIT_SCIENTIFIC_EXECUTION_APPROVAL_REQUIRED')
        ap=json.loads(Path(a.approval).read_text());cl=json.loads(Path(a.closure).read_text())
        def stopped(signum,frame):raise RuntimeError('OWN_PROCESS_SIGNAL:'+str(signum))
        signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
        out=run(c,g,ap,cl,authorization_paths=[Path(getattr(a,k)) for k in ('config','grant','approval','closure')])
        print(json.dumps(dict(status='PASS_MANDATORY_STOP',output=str(out))))
        return 0
    except Exception as exc:
        print(json.dumps(dict(status='STOP',reason=f'{type(exc).__name__}:{exc}')),file=sys.stderr)
        return 2


if __name__=='__main__':sys.exit(main())
