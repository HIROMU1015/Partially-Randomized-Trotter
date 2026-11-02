"""Guarded H4 dispatch, separate input/case interpreters, no automatic retry.

The parent imports stdlib/control only. Qiskit/molecular modules are loaded in
workers after the matching authority, claim and release records are checked.
Artificial mode uses an internal supplied-array fixture exclusively.
"""
from pathlib import Path
import json
import os
import sys
import time
import traceback

from .h4_execution_contract import ROOT, CASES, completed_contract, checked_pack, git
from .isolated_qiskit_execution import digest, read, save, sha
from .isolated_qiskit_supervisor import WatchPolicy, run_supervised, worker_ready

SCRIPT=ROOT/'scripts/run_h4_non_df_pilot.py'


def resources(path):
    fs=os.statvfs(path)
    return dict(affinity=len(os.sched_getaffinity(0)),load=list(os.getloadavg()),
        MemAvailable=next(l for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable')),
        disk_available_bytes=fs.f_bavail*fs.f_frsize,IO_PSI=Path('/proc/pressure/io').read_text(),
        CPU_PSI=Path('/proc/pressure/cpu').read_text(),OOM_observation='kernel managed',
        arbitrary_resource_caps=False,shared_environment_changes=False)


def synthetic_plan():
    p=completed_contract()
    p['MC']['trajectories_per_PR']=2
    p.update(scope='ARTIFICIAL_EIGHT_QUBIT_ONLY',molecular_execution=False)
    return p


def binding(pack,artificial):
    if artificial:
        cfg=read(Path(pack)/'artificial_config.json')
        if cfg['scope']!='ARTIFICIAL_EIGHT_QUBIT_ONLY' or cfg['plan']!=synthetic_plan():
            raise PermissionError('ARTIFICIAL_SCOPE_ONLY')
        return cfg['plan'],dict(SOURCE=cfg['SOURCE']),dict(grant_id='NO_SCIENTIFIC_GRANT_ARTIFICIAL_ONLY',
            claim_directory=str(Path(pack)/'artificial_claims'))
    return checked_pack(pack,allow_consumed=True)


def case_binding(pack,source,grant):
    data=Path(pack)/'input_run/input_generation/data'
    return dict(SOURCE=source['SOURCE'],grant_id=grant['grant_id'],
        contract_digest=digest(read(Path(pack)/'contract.json')) if (Path(pack)/'contract.json').exists() else digest(synthetic_plan()),
        receipt_sha256=sha(data/'receipt.json'),snapshot_sha256=sha(data/'snapshot.npz'))


def worker(pack,name,output,release,*,artificial=False):
    pack,output=Path(pack).absolute(),Path(output).absolute()
    plan,source,grant=binding(pack,artificial)
    stage='input_run' if name=='input_generation' else 'run'
    if name not in (*CASES,'input_generation') or output!=pack/stage/name:
        raise PermissionError('H4_WORKER_NAMESPACE_OR_CASE')
    marker=read(output/'marker.json')
    expected=dict(SOURCE=source['SOURCE'],grant_id=grant['grant_id'],contract_digest=digest(plan))
    if name!='input_generation':expected=case_binding(pack,source,grant)
    if marker['binding']!=expected or marker['case']!=name:
        raise PermissionError('H4_WORKER_CLAIM_BINDING')
    worker_ready(output,release)
    try:
        # Neither real molecules nor Qiskit are touched before authorization.
        from .h4_pilot_science import save_inputs,load_inputs,evaluate_h4_case,artificial_inputs
        if name=='input_generation':
            save(output/'reserved_scientific_calls.json',dict(scope='ARTIFICIAL_ONLY' if artificial else 'H4_INPUT',
                budget=plan['execution']['call_budget'],nonrefundable=True,automatic_retry=False))
            if artificial:
                p,d,s,r=artificial_inputs()
            else:
                from .h4_molecular_inputs import build_h4_inputs
                p,d,s,r=build_h4_inputs(output/'data',plan)
            save_inputs(output/'data',p,d,s,r,artificial=artificial)
            save(output/'status.json',dict(status='INPUTS_SEALED',artificial_only=artificial))
        else:
            p,d,s,r=load_inputs(pack/'input_run/input_generation/data',expected)
            if r['artificial_only']!=artificial:
                raise PermissionError('H4_ARTIFICIAL_MOLECULAR_INPUT_MIX')
            representation,method=name.rsplit('_',1)
            count=plan['MC']['trajectories_per_PR'] if method=='PR' else 1
            save(output/'reserved_scientific_calls.json',dict(paired_trajectories=count,
                wrapper_compile_axes=2*count,MC_production_events=count*plan['evolution']['q']*plan['evolution']['r'] if method=='PR' else 0,
                SOURCE=source['SOURCE'],scope='ARTIFICIAL_ONLY' if artificial else 'H4_FIXED_CASE',no_retry=True))
            evaluate_h4_case(p if representation=='pauli_jw' else d,s,representation,method,plan,r,expected,output)
    except BaseException as exc:
        save(output/'FAILED.json',dict(status='FAILED_NO_RETRY',reason=str(exc),traceback=traceback.format_exc(),
            case=name,SOURCE=source['SOURCE'],artificial_only=artificial))
        raise


def stage_jobs(pack,names,stage,bind,artificial):
    jobs=[]
    for name in names:
        out=Path(pack)/stage/name
        argv=[sys.executable,'-B',str(SCRIPT),'--worker',name,'--pack',str(pack),
            '--output',str(out),'--release',str(Path(pack)/stage/'release.json')]
        if artificial:argv.append('--artificial-worker')
        jobs.append(dict(case=name,case_id=name,output=out,binding=bind,argv=argv))
    return jobs


def run_pipeline(pack,*,artificial=False):
    pack=Path(pack).absolute()
    if artificial:
        pack.mkdir(parents=True,exist_ok=False)
        save(pack/'artificial_config.json',dict(scope='ARTIFICIAL_EIGHT_QUBIT_ONLY',plan=synthetic_plan(),SOURCE=git('rev-parse','HEAD')))
        plan,source,grant=binding(pack,True)
    else:
        plan,source,grant=checked_pack(pack)
    save(pack/'CONSUMED.json',dict(grant_id=grant['grant_id'],SOURCE=source['SOURCE'],
        resource_observation=resources(pack),one_shot=True,artificial_only=artificial))
    started=time.time()
    try:
        b=dict(SOURCE=source['SOURCE'],grant_id=grant['grant_id'],contract_digest=digest(plan))
        report=run_supervised(stage_jobs(pack,['input_generation'],'input_run',b,artificial),pack/'input_run',cwd=ROOT,
            policy=WatchPolicy(**plan['execution']['watch_policy']),claim_directory=grant['claim_directory'],
            stop_request=pack/'STOP.request')
        if report['status']!='COMPLETE_MANDATORY_STOP':
            raise RuntimeError('H4_INPUT_STAGE_FAILED_NO_CASE_DISPATCH')
        b=case_binding(pack,source,grant)
        report=run_supervised(stage_jobs(pack,CASES,'run',b,artificial),pack/'run',cwd=ROOT,
            policy=WatchPolicy(**plan['execution']['watch_policy']),claim_directory=grant['claim_directory'],
            stop_request=pack/'STOP.request')
        from .h4_pilot_science import integrate
        summary=integrate(pack/'run',plan,b)
        save(pack/'summary.json',summary)
        if report['status']!='COMPLETE_MANDATORY_STOP' or summary['status']!='FOUR_CASES_COMPLETE':
            raise RuntimeError('H4_PARTIAL_RESULTS_PRESERVED_NO_RETRY')
        final=dict(status='ARTIFICIAL_COMPLETE_MANDATORY_STOP' if artificial else 'H4_FIXED_PILOT_COMPLETE_MANDATORY_STOP',
            scope='ARTIFICIAL_ONLY' if artificial else 'H4_FIXED_FOUR_CASE_PILOT',
            SOURCE=source['SOURCE'],grant_id=grant['grant_id'],all_children_reaped=report['all_children_reaped'],
            elapsed_seconds=time.time()-started,automatic_retry=False,automatic_next_stage=False)
        save(pack/'STOP.json',final)
        return final
    except BaseException as exc:
        # Supervisor already retained its logs, partial events, exit/owned identities.
        if not (pack/'STOP.json').exists():
            save(pack/'STOP.json',dict(status='FAILED_NO_RETRY_MANDATORY_STOP',reason=str(exc),
                traceback=traceback.format_exc(),SOURCE=source['SOURCE'],grant_id=grant['grant_id'],
                artificial_only=artificial,automatic_retry=False,automatic_next_stage=False))
        raise
