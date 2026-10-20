"""Detached observer: actual wait4 witness for coordinator and metadata auditor."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import pf_direct_compile_contract as c
import pf_direct_retry_contract as r


def observe_command(command,directory,label,*,heartbeat_seconds=5,environment=None):
    """Own one child. Heartbeat/timeout interval never limits scientific runtime."""
    directory=Path(directory)
    log=(directory/(label+'.log')).open('x')
    start=time.monotonic()
    child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
        env=environment,cwd=r.ROOT,start_new_session=True,close_fds=True)
    stat=Path(f'/proc/{child.pid}/stat').read_text();values=stat[stat.rfind(')')+2:].split()
    r.save(directory/(label+'_started.json'),dict(pid=child.pid,parent_pid=os.getpid(),
        session_id=os.getsid(child.pid),process_group=os.getpgid(child.pid),start_ticks=values[19],
        command=command,monotonic=start))
    last=0.
    while True:
        pid,status,usage=os.wait4(child.pid,os.WNOHANG)
        if pid:break
        now=time.monotonic()
        if now-last>=heartbeat_seconds:
            r.append(directory/'observer_heartbeat.jsonl',dict(observer_pid=os.getpid(),child_pid=child.pid,label=label,
                monotonic=now,elapsed_seconds=now-start))
            last=now
        time.sleep(min(.2,heartbeat_seconds))
    code=os.waitstatus_to_exitcode(status);child.returncode=code;log.close()
    witness=dict(pid=child.pid,label=label,exit_code=code,reaped=True,raw_wait_status=status,
        wall_seconds=time.monotonic()-start,user_CPU_seconds=usage.ru_utime,system_CPU_seconds=usage.ru_stime,
        peak_RSS_bytes=usage.ru_maxrss*1024,automatic_retry=False,cleanup_errors=[])
    r.save(directory/(label+'_native_exit.json'),witness)
    return witness


def main(seal,directory,*,api=r):
    r=api
    spec=c.read_json(seal/(r.GROUPS[0]+'_spec.json'));grant=c.read_json(seal/(r.GROUPS[0]+'_grant.json'))
    r.authorize(spec,grant)
    c.require(Path(directory)==r.PRIVATE/'observer01','OBSERVER_OUTPUT_IDENTITY')
    directory=Path(directory);r.gate(spec,grant)
    stat=Path('/proc/self/stat').read_text();parts=stat[stat.rfind(')')+2:].split()
    r.save(directory/'observer_started.json',dict(pid=os.getpid(),parent_pid=os.getppid(),session_id=os.getsid(0),
        process_group=os.getpgrp(),start_ticks=parts[19],SOURCE=spec['SOURCE'],spec_digest=c.digest(spec),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()))
    environment=os.environ.copy();environment['PYTHONDONTWRITEBYTECODE']='1'
    code=0
    try:
        command=r.command('run',spec,seal) if hasattr(r,'command') else [spec['observer_python'],'-B',str(r.ROOT/'scripts/run_pf_direct_retry216.py'),'--sealed-directory',str(seal)]
        coordinator=observe_command(command,directory,'coordinator',environment=environment)
        c.require(coordinator['exit_code']==0,'COORDINATOR_NATIVE_FAILURE')
        terminal=r.PRIVATE/'run01/terminal.json'
        c.require(terminal.is_file(),'COORDINATOR_TERMINAL_MISSING')
        r.save(directory/'coordinator_terminal_binding.json',dict(path=str(terminal),sha256=r.sha(terminal),
            native_exit_witness_sha256=r.sha(directory/'coordinator_native_exit.json')))
        command=r.command('audit',spec,seal) if hasattr(r,'command') else [spec['observer_python'],'-B',str(r.ROOT/'scripts/audit_pf_direct_retry216.py'),'--sealed-directory',str(seal)]
        auditor=observe_command(command,directory,'metadata_auditor',environment=environment)
        c.require(auditor['exit_code']==0,'METADATA_AUDIT_FAILURE')
    except BaseException as error:
        code=2
        r.save(directory/'STOP.json',dict(status='MANDATORY_STOP',reason=type(error).__name__+':'+str(error),
            traceback=traceback.format_exc(),automatic_retry=False,next_stage_authorized=False))
    r.save(directory/'observer_terminal.json',dict(status='COMPLETE_OBSERVER_MANDATORY_STOP' if code==0 else 'STOP',
        self_reported_exit_code=code,coordinator_exit_has_independent_wait4_witness=True,
        observer_exit_not_externally_witnessed=True,SOURCE=spec['SOURCE'],next_stage_authorized=False))
    return code


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sealed-directory',type=Path,required=True);p.add_argument('--directory',type=Path,required=True)
    args=p.parse_args();raise SystemExit(main(args.sealed_directory,args.directory))
