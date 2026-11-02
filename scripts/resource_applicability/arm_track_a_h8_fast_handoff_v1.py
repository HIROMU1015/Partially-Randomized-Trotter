#!/usr/bin/env python3
"""Wait for the unchanged heavy job; launch remaining-seven ONCE on success.

No old worker is stopped, no science is retried, and full H8 is never launched.
The conditional template is byte-pinned before waiting. Final heavy hashes,
plan/auth digest and fresh resources are resolved only after its native end.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.resource_applicability.h8_fast_handoff_contract_v1 import (
    AUTHORIZATION,RUNNER,read,private_path,verify_template,admit_heavy,finalized,verify_binding,inactive)
from trottertracks.resource_applicability.ax2a_preparation import digest
from trottertracks.resource_applicability.ax2b_h4_contract_v5 import file_hash
from trottertracks.resource_applicability.h6_matched_execution_v2 import worker_environment
from trottertracks.resource_applicability.h6_server_resources_v1 import process_identity,stop_owned_group


def publish(path,value):
    with Path(path).open('x') as f:f.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def validate_conditional(template,authorization):
    if (authorization.get('schema')!='h8_conditional_handoff_user_authorization_v1'
        or authorization.get('approved_by_user') is not True or authorization.get('authorization_basis')!=AUTHORIZATION
        or authorization.get('template_sha256')!=digest(template) or authorization.get('source_commit')!=template['source_commit']
        or authorization.get('output')!=template['output'] or authorization.get('one_shot') is not True
        or authorization.get('retry') is not False or authorization.get('H8_full_campaign_authorized') is not False
        or authorization.get('old_heavy_must_not_be_interrupted') is not True
        or (authorization.get('remaining_cases'),authorization.get('remaining_compile'),authorization.get('total_compile'))!=(7,18,20)):
        raise ValueError('CONDITIONAL_HANDOFF_AUTHORIZATION')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--template',type=Path,required=True);p.add_argument('--authorization',type=Path,required=True)
    p.add_argument('--template-sha256',required=True);p.add_argument('--authorization-sha256',required=True)
    p.add_argument('--control',type=Path,required=True)
    a=p.parse_args()
    if file_hash(a.template)!=a.template_sha256 or file_hash(a.authorization)!=a.authorization_sha256:
        raise ValueError('CONDITIONAL_BYTES_CHANGED')
    template=read(a.template);auth=read(a.authorization);validate_conditional(template,auth);verify_template(ROOT,template)
    control=private_path(a.control.resolve());control.mkdir(parents=True,exist_ok=False)
    resource.setrlimit(resource.RLIMIT_AS,(8*2**30,32*2**30))
    resource.setrlimit(resource.RLIMIT_FSIZE,(32*2**20,2*2**30));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    os.sched_setaffinity(0,{31})
    identity=process_identity(os.getpid())
    publish(control/'armed.json',dict(status='ARMED_WAITING_UNCHANGED_HEAVY',SOURCE=template['source_commit'],
        identity=identity,template_sha256=a.template_sha256,authorization_sha256=a.authorization_sha256,
        output=template['output'],remaining_cases=7,remaining_compile=18,total_compile=20,
        heavy_must_finish_without_interruption=True,old_heavy_source=template['old_parent']['source_commit'],
        retry=False,H8_full_campaign_authorized=False))
    parent=Path(template['old_parent']['directory']);child=None;child_identity=None
    try:
        missing_since=None
        while True:
            if (control/'STOP_ARM').exists():raise ValueError('USER_STOP_ARM')
            if file_hash(a.template)!=a.template_sha256 or file_hash(a.authorization)!=a.authorization_sha256:
                raise ValueError('ARMED_TEMPLATE_CHANGED')
            heavy=admit_heavy(parent)
            if heavy is not None:break
            if inactive(template['old_parent']['coordinator_identity']) and not (parent/'heavy01/execution_terminal.json').exists():
                missing_since=time.monotonic() if missing_since is None else missing_since
                if time.monotonic()-missing_since>5:raise ValueError('OLD_HEAVY_EXIT_WITHOUT_TERMINAL')
            time.sleep(.5)
        # A completed-and-reaped old case is the only accepted parent. Recheck
        # the stopped pending pipeline, inputs and exact source just before
        # creating the new one-shot manifest/grant; no old grant is reopened.
        verify_template(ROOT,template)
        m,g=finalized(template,heavy);verify_binding(ROOT,m,g)
        if Path(m['output']).exists() or (Path(m['output']).parent/(Path(m['output']).name+'.one_shot_lock')).exists():
            raise ValueError('NEW_OUTPUT_USED')
        publish(control/'manifest.json',m);publish(control/'grant.json',g)
        publish(control/'accepted_heavy_receipt.json',heavy)
        cache=control/'runtime_cache';cache.mkdir();temp=cache/'tmp';temp.mkdir()
        env=worker_environment(1,cache);env.update(TMPDIR=str(temp),TEMP=str(temp),TMP=str(temp),XDG_CACHE_HOME=str(cache/'xdg'))
        argv=[sys.executable,'-I','-P','-B',str(ROOT/RUNNER),'--manifest',str(control/'manifest.json'),
            '--grant',str(control/'grant.json'),'--manifest-sha256',file_hash(control/'manifest.json'),
            '--grant-sha256',file_hash(control/'grant.json'),'--execute']
        with (control/'coordinator.log').open('xb') as log:
            child=subprocess.Popen(argv,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        child_identity=process_identity(child.pid)
        publish(control/'launch.json',dict(identity=child_identity,command=argv,SOURCE=m['source_commit'],
            manifest_sha256=file_hash(control/'manifest.json'),grant_sha256=file_hash(control/'grant.json'),
            old_heavy_receipt_digest=digest(heavy),remaining_cases=7,retry=False))
        code=child.wait()
        publish(control/'coordinator_native_exit.json',dict(identity=child_identity,exit_code=code,wait_reap=True,
            exit_observed_utc=datetime.now(timezone.utc).isoformat(),exact_exit_timestamp=None))
        terminal=read(Path(m['output'])/'execution_terminal.json') if (Path(m['output'])/'execution_terminal.json').exists() else None
        complete=bool(code==0 and terminal and terminal['first_stop_reason'] is None and terminal['all_dispatched_reaped']
            and terminal['completed_indices']==[0,1,2,4,5,6,7] and not terminal['cleanup_errors'])
        if complete:
            combined=read(Path(m['output'])/'combined_feasibility_result.json')
            if (combined['total_wrappers'],combined['new_wrappers'],combined['calls_attempted']['compile'])!=(20,18,20):
                raise ValueError('COMBINED_HANDOFF_RESULT')
        publish(control/'terminal.json',dict(status='FAST_HANDOFF_COMPLETE_MANDATORY_STOP' if complete else 'FAST_HANDOFF_STOP',
            coordinator_exit_code=code,coordinator_wait_reaped=True,resource_terminal=terminal,
            SOURCE=m['source_commit'],output=m['output'],old_heavy_preserved=True,retry=False,
            H8_full_campaign_authorized=False))
        return 0 if complete else 1
    except Exception as exc:
        cleanup=None
        if child is not None and child.returncode is None:
            # Request the owned coordinator's normal fail-closed cleanup first.
            # Never signal the old heavy coordinator/worker or another run.
            try:
                out=Path(template['output'])
                if out.is_dir():
                    with (out/'STOP_REQUEST').open('x') as f:f.write('Armed handoff metadata failure\n')
                try:code=child.wait(timeout=15)
                except subprocess.TimeoutExpired:code=stop_owned_group(child,child_identity)
                cleanup={'new_coordinator_exit_code':code,'new_coordinator_reaped':True,
                         'old_heavy_signalled':False}
            except Exception as error:
                cleanup={'error':type(error).__name__+':'+str(error)[:256],
                         'new_coordinator_reaped':child.returncode is not None,
                         'old_heavy_signalled':False}
        publish(control/'terminal.json',dict(status='FAST_HANDOFF_STOP',reason=type(exc).__name__+':'+str(exc)[:512],
            SOURCE=template['source_commit'],old_heavy_preserved=True,remaining_started=child is not None,
            cleanup=cleanup,retry=False,H8_full_campaign_authorized=False))
        return 1


if __name__=='__main__':raise SystemExit(main())
