#!/usr/bin/env python3
"""Observe ONE approved fast heavy retry, then execute registered seven once."""
import argparse,json,os,resource,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.resource_applicability.h8_fast_retry_contract_v1 import (
    RUNNER,PREPARER,verify_binding,verify_completed_heavy)
from trottertracks.resource_applicability.h8_fast_handoff_contract_v1 import read,inactive,private_path
from trottertracks.resource_applicability.ax2b_h4_contract_v5 import file_hash
from trottertracks.resource_applicability.h6_matched_execution_v2 import worker_environment
from trottertracks.resource_applicability.h6_server_resources_v1 import process_identity


def publish(path,data):
    with Path(path).open('x') as f:f.write(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def wait_existing(launch_path,directory,stop):
    launch=read(launch_path);expected=launch['identity'];terminal=directory/'execution_terminal.json';missing=None
    while not terminal.exists():
        if stop.exists():raise ValueError('USER_STOP_CONTINUATION')
        if inactive(expected):
            missing=time.monotonic() if missing is None else missing
            if time.monotonic()-missing>5:raise ValueError('RETRY_EXIT_WITHOUT_TERMINAL')
        time.sleep(.5)
    report=read(terminal)
    if report['first_stop_reason'] is not None or not report['all_dispatched_reaped'] or report['cleanup_errors']:
        raise ValueError('EXPLICIT_RETRY_STOP:'+str(report['first_stop_reason']))
    for _ in range(30):
        if inactive(expected):return report
        time.sleep(.2)
    raise ValueError('RETRY_COORDINATOR_STILL_ACTIVE')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--heavy-manifest',type=Path,required=True);p.add_argument('--heavy-grant',type=Path,required=True)
    p.add_argument('--manifest-sha256',required=True);p.add_argument('--grant-sha256',required=True)
    p.add_argument('--heavy-launch',type=Path);p.add_argument('--start-heavy',action='store_true')
    p.add_argument('--control',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.start_heavy==bool(a.heavy_launch):raise ValueError('CHOOSE_START_HEAVY_OR_EXISTING_LAUNCH')
    if file_hash(a.heavy_manifest)!=a.manifest_sha256 or file_hash(a.heavy_grant)!=a.grant_sha256:raise ValueError('RETRY_CONTINUATION_BOUND_BYTES')
    m,g=read(a.heavy_manifest),read(a.heavy_grant);verify_binding(ROOT,m,g)
    if m['scope']!='HEAVY_RETRY_FAST':raise ValueError('RETRY_FIRST_HEAVY_ONLY')
    control=private_path(a.control.resolve());control.mkdir(parents=True,exist_ok=False)
    resource.setrlimit(resource.RLIMIT_AS,(8*2**30,32*2**30));resource.setrlimit(resource.RLIMIT_FSIZE,(32*2**20,2*2**30))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));os.sched_setaffinity(0,{31})
    publish(control/'armed.json',dict(SOURCE=m['source_commit'],identity=process_identity(os.getpid()),
        status='START_APPROVED_FAST_HEAVY_RETRY' if a.start_heavy else 'WAITING_APPROVED_FAST_HEAVY_RETRY',remaining_cases=7,remaining_compile=18,
        cumulative_compile_cap=21,automatic_retry=False,H8_full_campaign_authorized=False))
    try:
        if a.start_heavy:
            cache=control/'heavy_cache';cache.mkdir();temp=cache/'tmp';temp.mkdir()
            env=worker_environment(1,cache);env.update(TMPDIR=str(temp),TEMP=str(temp),TMP=str(temp),XDG_CACHE_HOME=str(cache/'xdg'))
            argv=[sys.executable,'-I','-P','-B',str(ROOT/RUNNER),'--manifest',str(a.heavy_manifest.resolve()),
                '--grant',str(a.heavy_grant.resolve()),'--manifest-sha256',a.manifest_sha256,
                '--grant-sha256',a.grant_sha256,'--execute']
            with (control/'heavy_coordinator.log').open('xb') as log:
                child=subprocess.Popen(argv,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            identity=process_identity(child.pid)
            publish(control/'heavy_launch.json',dict(identity=identity,command=argv,SOURCE=m['source_commit']))
            code=child.wait()
            publish(control/'heavy_native_exit.json',dict(identity=identity,exit_code=code,wait_reaped=True,
                exit_observed_utc=datetime.now(timezone.utc).isoformat(),exact_exit_timestamp=None))
            if code!=0:raise ValueError('EXPLICIT_HEAVY_RETRY_EXIT:'+str(code))
        else:
            wait_existing(a.heavy_launch,Path(m['output']),control/'STOP_CONTINUATION')
        verify_binding(ROOT,m,g);verify_completed_heavy(ROOT,m['output'],m['source_commit'])
        bundle=control/'remaining_auth'
        argv=[sys.executable,'-I','-P','-B',str(ROOT/PREPARER),'--scope','REMAINING_AFTER_FAST_RETRY',
            '--source',m['source_commit'],'--parent',m['old_parent']['directory'],'--output',str(a.output.resolve()),
            '--bundle',str(bundle),'--completed-heavy',m['output']]
        with (control/'preparation.log').open('xb') as f:subprocess.run(argv,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        cache=control/'cache';cache.mkdir();temp=cache/'tmp';temp.mkdir()
        env=worker_environment(1,cache);env.update(TMPDIR=str(temp),TEMP=str(temp),TMP=str(temp),XDG_CACHE_HOME=str(cache/'xdg'))
        argv=[sys.executable,'-I','-P','-B',str(ROOT/RUNNER),'--manifest',str(bundle/'manifest.json'),
            '--grant',str(bundle/'grant.json'),'--manifest-sha256',file_hash(bundle/'manifest.json'),
            '--grant-sha256',file_hash(bundle/'grant.json'),'--execute']
        with (control/'coordinator.log').open('xb') as log:
            child=subprocess.Popen(argv,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        identity=process_identity(child.pid);publish(control/'launch.json',dict(identity=identity,command=argv,SOURCE=m['source_commit']))
        code=child.wait();publish(control/'native_exit.json',dict(identity=identity,exit_code=code,wait_reaped=True,
            exit_observed_utc=datetime.now(timezone.utc).isoformat(),exact_exit_timestamp=None))
        t=read(a.output/'execution_terminal.json') if (a.output/'execution_terminal.json').exists() else None
        complete=bool(code==0 and t and t['first_stop_reason'] is None and t['all_dispatched_reaped'] and t['completed_indices']==[0,1,2,4,5,6,7])
        publish(control/'terminal.json',dict(status='FAST_RETRY_EIGHT_CASES_COMPLETE_MANDATORY_STOP' if complete else 'FAST_RETRY_STOP',
            SOURCE=m['source_commit'],coordinator_exit_code=code,resource_terminal=t,
            output=str(a.output),automatic_retry=False,H8_full_campaign_authorized=False))
        return 0 if complete else 1
    except Exception as e:
        publish(control/'terminal.json',dict(status='FAST_RETRY_STOP',reason=type(e).__name__+':'+str(e)[:512],
            SOURCE=m['source_commit'],automatic_retry=False,H8_full_campaign_authorized=False))
        return 1


if __name__=='__main__':raise SystemExit(main())
