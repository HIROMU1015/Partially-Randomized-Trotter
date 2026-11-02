"""Linux pilot-only supervisor. All state/output scopes are supplied explicitly.

Accounting: unique process IDs plus supervisor own RSS (not physical unique
memory); shared pages can be counted across PIDs. wait4 with child-subreaper
collects terminated/reparented descendants. Cooperative descendants may not
escape the supervisor; no privileged cgroup/system environment changes.
"""
import ctypes
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

def stat(pid):
    try:
        data=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return {'pid':pid,'state':data[0],'ppid':int(data[1]),'pgrp':int(data[2]),
                'start_ticks':int(data[19]),'RSS_bytes':int(data[21])*os.sysconf('SC_PAGE_SIZE')}
    except (OSError,ValueError,IndexError):
        return None

def members(group, supervisor):
    records=[v for name in Path('/proc').iterdir() if name.name.isdigit()
             for v in [stat(int(name.name))] if v is not None]
    chosen={v['pid']:v for v in records if v['pgrp']==group or v['ppid']==supervisor}
    while True:
        extra={v['pid']:v for v in records if v['ppid'] in chosen and v['pid'] not in chosen}
        if not extra:break
        chosen.update(extra)
    return chosen

def output_bytes(roots):
    seen=set();size=0
    for root in map(Path,roots):
        paths=[root] if root.is_file() else root.rglob('*') if root.exists() else []
        for p in paths:
            try:
                if not p.is_file():continue
                s=p.stat();key=(s.st_dev,s.st_ino)
                if key not in seen:seen.add(key);size+=s.st_size
            except FileNotFoundError:pass
    return size

def supervise(spec):
    ledger=Path(spec['ledger']);ledger.mkdir(parents=True,exist_ok=True)
    stop=Path(spec['stop_path']);record={'id':spec['id'],'launched':False,'retries':0,
        'accounting':'UNIQUE_PID_RSS_SUM_CONSERVATIVE','CPU_accounting':'kernel wait4 with PR_SET_CHILD_SUBREAPER',
        'failure':None,'returncode':None,'samples':[],'wait4_records':[],'remaining_members':[]}
    result_path=ledger/f"{spec['id']}.result.json"
    if stop.exists() or result_path.exists() or (ledger/f"{spec['id']}.started.json").exists():
        return {'id':spec['id'],'launched':False,'failure':'RETRY_OR_AFTER_STOP_REFUSED','retries':0}
    with (ledger/f"{spec['id']}.started.json").open('x') as f:json.dump(spec,f)
    began=time.monotonic();remaining=spec['pilot_deadline_epoch']-time.time()
    effective=min(spec['wall_seconds'],remaining)
    record['effective_wall_seconds']=effective
    root_pid=None;known=set();cpu=0.;peak=0;largest_output=0
    libc=ctypes.CDLL(None,use_errno=True)
    subreaper=libc.prctl(36,1,0,0,0)==0
    record['subreaper_enabled']=subreaper
    if not subreaper:record['failure']='SUBREAPER_UNAVAILABLE'
    elif effective<=0:record['failure']='PILOT_WALL_CAP'
    elif output_bytes(spec['output_roots'])>=spec['output_bytes']:record['failure']='OUTPUT_CAP'
    else:
        def limits():
            os.setsid()
            resource.setrlimit(resource.RLIMIT_CORE,(0,0))
            resource.setrlimit(resource.RLIMIT_AS,(spec['address_space_bytes'],)*2)
            resource.setrlimit(resource.RLIMIT_FSIZE,(spec['output_bytes'],)*2)
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                 TMPDIR=spec['TMPDIR'],PYTHONDONTWRITEBYTECODE='1')
        Path(spec['TMPDIR']).mkdir(parents=True,exist_ok=True)
        stdout=Path(spec['stdout']);stderr=Path(spec['stderr'])
        stdout.parent.mkdir(parents=True,exist_ok=True);stderr.parent.mkdir(parents=True,exist_ok=True)
        with stdout.open('xb') as out,stderr.open('xb') as err:
            child=subprocess.Popen(spec['command'],cwd=spec['cwd'],env=env,stdout=out,stderr=err,preexec_fn=limits)
            root_pid=child.pid;record['launched']=True;record['root_pid']=root_pid
            kill_started=None
            while True:
                # Reap without Popen.poll()/wait() so direct-child rusage is retained.
                while True:
                    try:pid,status,usage=os.wait4(-1,os.WNOHANG)
                    except ChildProcessError:break
                    if pid==0:break
                    code=os.waitstatus_to_exitcode(status)
                    own_cpu=usage.ru_utime+usage.ru_stime;cpu+=own_cpu
                    record['wait4_records'].append({'pid':pid,'returncode':code,'CPU_seconds':own_cpu,
                                                    'ru_maxrss_KiB':usage.ru_maxrss})
                    if pid==root_pid:record['returncode']=code;child.returncode=code
                selected=members(root_pid,os.getpid());known.update(selected)
                own=stat(os.getpid());parts=dict(selected)
                if own is not None:parts[own['pid']]=own
                rss=sum(v['RSS_bytes'] for v in parts.values());peak=max(peak,rss)
                count=output_bytes(spec['output_roots']);largest_output=max(largest_output,count)
                record['samples'].append({'elapsed_seconds':time.monotonic()-began,
                    'parts':list(parts.values()),'PID_count':len(parts),'RSS_sum_bytes':rss,'output_bytes':count})
                if record['failure'] is None:
                    if time.monotonic()-began>=effective:record['failure']='WALL_CAP'
                    elif rss>=spec['RSS_bytes']:record['failure']='RSS_CAP'
                    elif count>=spec['output_bytes']:record['failure']='OUTPUT_CAP'
                if record['failure'] is not None:
                    if not stop.exists():
                        with stop.open('x') as f:json.dump({'id':spec['id'],'failure':record['failure'],'retry':0},f)
                    if kill_started is None:kill_started=time.monotonic()
                    try:os.killpg(root_pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    for pid in selected:
                        try:os.kill(pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                if record['returncode'] is not None and not selected:break
                if kill_started is not None and time.monotonic()-kill_started>3:
                    record['remaining_members']=list(selected.values());record['termination_failed']=True;break
                time.sleep(spec.get('sample_seconds',.025))
        if record['failure'] is None and record['returncode']!=0:record['failure']='PROCESS_EXIT_FAILURE'
    if record['failure'] is not None and not stop.exists():
        with stop.open('x') as f:json.dump({'id':spec['id'],'failure':record['failure'],'retry':0},f)
    record.update(wall_seconds=time.monotonic()-began,CPU_seconds=cpu,peak_RSS_sum_bytes=peak,
                  peak_output_bytes=largest_output,known_target_PIDs=sorted(known),
                  residual_processes=0 if not record['remaining_members'] else len(record['remaining_members']))
    result_path.write_text(json.dumps(record,indent=2)+'\n')
    record['final_output_bytes']=output_bytes(spec['output_roots'])
    result_path.write_text(json.dumps(record,indent=2)+'\n')
    return record

if __name__=='__main__':
    specification=json.loads(Path(sys.argv[1]).read_text())
    result=supervise(specification)
    print(json.dumps({k:v for k,v in result.items() if k not in ['samples','wait4_records']}))
    sys.exit(0 if result.get('failure') is None else 3)
