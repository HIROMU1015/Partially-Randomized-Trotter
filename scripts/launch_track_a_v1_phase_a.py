"""Grant-only V1 launcher; independent stdlib observer, owned process cleanup.

The CLI has no default execution path. It cannot issue or approve a grant.
Linux subreaping and parent-death signals cover driver/worker crash cascades.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

import track_a_phase_a_parallel as owned
import track_a_v1_runner as runner

ROOT=Path(__file__).resolve().parents[1]


def snapshot():
    mem=next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
             if line.startswith('MemAvailable:'))
    relative=next(line.split(':',2)[2] for line in Path('/proc/self/cgroup').read_text().splitlines()
                  if line.startswith('0::'))
    root=Path('/sys/fs/cgroup');group=root/relative.lstrip('/');available=[mem];events={}
    pressure={'host':Path('/proc/pressure/memory').read_text()}
    while group.is_relative_to(root):
        if (group/'memory.events').exists():
            events[str(group)]=dict(line.split() for line in (group/'memory.events').read_text().splitlines())
            pressure[str(group)]=(group/'memory.pressure').read_text()
            for name in ('memory.max','memory.high'):
                cap=(group/name).read_text().strip()
                if cap!='max':available.append(max(0,int(cap)-int((group/'memory.current').read_text())))
        if group==root:break
        group=group.parent
    return dict(effective_available_bytes=min(available),events=events,pressure=pressure)


def resource_reason(current,baseline,headroom):
    if current['effective_available_bytes']<headroom:return 'MEMORY_HEADROOM'
    for group,rows in current['events'].items():
        for name in ('oom','oom_kill','oom_group_kill'):
            if int(rows.get(name,0))>int(baseline['events'][group].get(name,0)):return 'OOM_INCREMENT'
    for group,raw in current['pressure'].items():
        for line in raw.splitlines():
            value=float(dict(piece.split('=') for piece in line.split()[1:])['avg10'])
            if value>=1 if group=='host' else value>0:return 'MEMORY_PRESSURE'
    return None


def process_record(pid):
    path=Path('/proc')/str(pid);ident=owned.identity(pid)
    if ident is None:return None
    fields=(path/'stat').read_text().rsplit(')',1)[1].split()
    if fields[0]=='Z':return dict(identity=ident,zombie=True)
    values=dict(line.split(':',1) for line in (path/'status').read_text().splitlines() if ':' in line)
    ticks=os.sysconf('SC_CLK_TCK')
    return dict(identity=ident,zombie=False,RSS_bytes=int(values['VmRSS'].split()[0])*1024,
        AS_bytes=int(values['VmSize'].split()[0])*1024,threads=int(values['Threads']),
        CPU_seconds=(int(fields[11])+int(fields[12]))/ticks,
        reaped_child_CPU_seconds=(int(fields[13])+int(fields[14]))/ticks)


def enforce_processes(rows,limits,cpu_seconds):
    live=[r for r in rows if not r['zombie']]
    if len(live)>limits['max_workers']+1:raise RuntimeError('PROCESS_COUNT_CEILING')
    if sum(r['RSS_bytes'] for r in live)>limits['aggregate_RSS_bytes']:raise RuntimeError('AGGREGATE_RSS_CEILING')
    for r in live:
        if r['RSS_bytes']>limits['RSS_bytes'] or r['AS_bytes']>limits['AS_bytes']:raise RuntimeError('PROCESS_MEMORY_CEILING')
        if r['threads']!=1:raise RuntimeError('PROCESS_THREADS')
    if cpu_seconds>limits['aggregate_CPU_seconds']:raise RuntimeError('AGGREGATE_CPU_CEILING')


def observer_memory(*,status=None,lifetime_peak_kib=None):
    """Current executable's peak; ru_maxrss can include an earlier executable.

    Linux resets /proc VmHWM at exec, but preserves getrusage's lifetime peak.
    Keep both values so a STOP never loses the measurement that triggered it.
    """
    if status is None:status=Path('/proc/self/status').read_text()
    fields=dict(line.split(':',1) for line in status.splitlines() if ':' in line)
    if lifetime_peak_kib is None:lifetime_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    row={name:int(fields[key].split()[0])*1024 for name,key in
         (('current_RSS_bytes','VmRSS'),('current_exec_peak_RSS_bytes','VmHWM'),('AS_bytes','VmSize'))}
    row['lifetime_peak_RSS_bytes']=lifetime_peak_kib*1024
    if min(row.values())<0 or row['current_RSS_bytes']>row['current_exec_peak_RSS_bytes']:
        raise RuntimeError('INVALID_OBSERVER_MEMORY_OBSERVATION')
    return row


def enforce_observer(row,limits):
    if row['current_exec_peak_RSS_bytes']>limits.get('observer_RSS_bytes',limits['RSS_bytes']):
        raise RuntimeError('OBSERVER_RSS_CEILING')
    if row['AS_bytes']>limits.get('observer_AS_bytes',limits['AS_bytes']):
        raise RuntimeError('OBSERVER_AS_CEILING')


def supervise(command,*,env,control,output,limits,allowed_cpus,probe=snapshot):
    """Internal owned-child supervisor; CLI validates the grant before calling."""
    control=Path(control);output=Path(output);control.mkdir(mode=0o700)
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(36,1,0,0,0)!=0:raise OSError(ctypes.get_errno(),'PR_SET_CHILD_SUBREAPER')
    start=time.monotonic();last=start;baseline=probe();handles={};driver=None;stop=None
    peak=0;gap_max=0;exitcode=None;cleanup_errors=[];last_observer_memory=None
    previous={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
    def stop_signal(*args):raise RuntimeError('LAUNCHER_STOP_SIGNAL')
    for sig in previous:signal.signal(sig,stop_signal)
    try:
        stop=resource_reason(baseline,baseline,limits['aggregate_RSS_bytes']+16*2**30)
        if stop:raise RuntimeError('FRESH_GATE:'+stop)
        parent=os.getpid()
        with (control/'stdout.log').open('xb') as out,(control/'stderr.log').open('xb') as err:
            mask=signal.pthread_sigmask(signal.SIG_BLOCK,set(previous))
            try:
                pid=os.fork()
                if pid==0:
                    try:
                        if libc.prctl(1,int(signal.SIGKILL),0,0,0)!=0 or os.getppid()!=parent:os._exit(125)
                        for sig in previous:signal.signal(sig,signal.SIG_DFL)
                        os.sched_setaffinity(0,set(allowed_cpus))
                        os.dup2(out.fileno(),1);os.dup2(err.fileno(),2)
                        signal.pthread_sigmask(signal.SIG_SETMASK,mask)
                        os.execve(command[0],command,env)
                    except BaseException:os._exit(126)
                driver=dict(pid=pid,identity=owned.identity(pid),reaped=False)
                if driver['identity'] is None or driver['identity'][2:]!=(parent,os.getuid()):
                    raise RuntimeError('DRIVER_IDENTITY_BINDING')
                handles[pid]=driver
                owned.atomic_json(control/'started.json',dict(driver=driver,command=command))
            finally:signal.pthread_sigmask(signal.SIG_SETMASK,mask)
            if 'observer_AS_bytes' in limits:
                # Apply only AFTER fork; the driver needs the separate 32GiB
                # allowance and must not inherit the observer's hard ceiling.
                resource.setrlimit(resource.RLIMIT_AS,(limits['observer_AS_bytes'],limits['observer_AS_bytes']))
                resource.setrlimit(resource.RLIMIT_CPU,(limits['CPU_seconds'],limits['CPU_seconds']))
            with (control/'observations.jsonl').open('x') as log:
                while True:
                    now=time.monotonic();gap=now-last;last=now;gap_max=max(gap_max,gap)
                    if gap>5:raise RuntimeError('EXTERNAL_MONITOR_STALENESS')
                    owned.wait_owned(driver)
                    if driver['reaped']:
                        exitcode=driver['exit_code'];break
                    if now-start>limits['wall_seconds']:raise RuntimeError('WALL_CEILING')
                    # /proc task children enumerates direct children only;
                    # workers never fork further. Discover only owned lineage.
                    children=(Path('/proc')/str(pid)/'task'/str(pid)/'children').read_text().split()
                    rows=[]
                    for number in [pid,*map(int,children)]:
                        try:row=process_record(number)
                        except (FileNotFoundError,KeyError,ProcessLookupError):row=None
                        if row is None:
                            old=handles.get(number)
                            if old and time.monotonic()-old.get('last_seen',start)>5:
                                raise RuntimeError('LIVE_RESOURCE_OBSERVATION_MISSING')
                            continue
                        ident=row['identity']
                        if number not in handles:
                            if ident[2:]!=(pid,os.getuid()):raise RuntimeError('WORKER_LINEAGE')
                            handles[number]=dict(pid=number,identity=ident,reaped=False)
                        if ident!=handles[number]['identity']:raise RuntimeError('PROCESS_IDENTITY_CHANGED')
                        if not os.sched_getaffinity(number)<=set(allowed_cpus):raise RuntimeError('CPU_ASSIGNMENT_CHANGED')
                        handles[number]['last_seen']=time.monotonic();rows.append(row)
                    self_usage=resource.getrusage(resource.RUSAGE_SELF)
                    total_cpu=sum(r.get('CPU_seconds',0)+r.get('reaped_child_CPU_seconds',0) for r in rows)
                    enforce_processes(rows,limits,total_cpu+self_usage.ru_utime+self_usage.ru_stime)
                    last_observer_memory=observer_memory(lifetime_peak_kib=self_usage.ru_maxrss)
                    enforce_observer(last_observer_memory,limits)
                    observer_rss=last_observer_memory['current_exec_peak_RSS_bytes']
                    if sum(r.get('RSS_bytes',0) for r in rows)+observer_rss>limits['aggregate_RSS_bytes']:
                        raise RuntimeError('AGGREGATE_RSS_CEILING')
                    current=probe();reason=resource_reason(current,baseline,16*2**30)
                    if reason:raise RuntimeError(reason)
                    fs=os.statvfs(control)
                    if fs.f_bavail*fs.f_frsize<2*2**30 or fs.f_favail<4096:raise RuntimeError('FILESYSTEM_HEADROOM')
                    size=sum(p.stat().st_size for d in (output,control) for p in d.rglob('*') if p.is_file())
                    if size>limits['output_bytes']:raise RuntimeError('OUTPUT_CEILING')
                    if any((control/n).stat().st_size>16*2**20 for n in ('stdout.log','stderr.log','observations.jsonl')):
                        raise RuntimeError('LOG_CEILING')
                    rss=sum(r.get('RSS_bytes',0) for r in rows)+observer_rss;peak=max(peak,rss)
                    log.write(json.dumps(dict(elapsed=now-start,interval=gap,aggregate_RSS_bytes=rss,
                        aggregate_CPU_seconds=total_cpu,processes=len(rows),observer_memory=last_observer_memory))+'\n');log.flush()
                    time.sleep(1)
            if exitcode!=0:raise RuntimeError('DRIVER_NATIVE_EXIT:'+str(exitcode))
            terminal=runner.read_json(output/'terminal.json')
            if terminal.get('status')!='COMPLETE_MANDATORY_STOP':raise RuntimeError('MISSING_COMPLETE_TERMINAL')
    except BaseException as exc:stop=type(exc).__name__+':'+str(exc)
    finally:
        mask=signal.pthread_sigmask(signal.SIG_BLOCK,set(previous))
        try:
            # Driver gets TERM first so it can reap workers and record STOP.
            # KILL cascades through PDEATHSIG; orphaned children are adopted.
            for sig,seconds in ((signal.SIGTERM,5.),(signal.SIGKILL,3.)):
                if driver is not None:
                    try:
                        owned.wait_owned(driver)
                        if not driver['reaped']:owned.send_owned(driver,sig)
                    except BaseException as exc:cleanup_errors.append(str(exc))
                if sig==signal.SIGKILL:
                    # Every direct child of this exclusive launcher is owned;
                    # the subreaper adopts workers when their driver dies.
                    own_children=Path('/proc/self/task')/str(os.getpid())/'children'
                    for text_pid in own_children.read_text().split():
                        number=int(text_pid);ident=owned.identity(number)
                        if ident is None:continue
                        if ident[2:]!=(os.getpid(),os.getuid()):
                            cleanup_errors.append('ADOPTED_OWNERSHIP_CHANGED');continue
                        h=handles.setdefault(number,dict(pid=number,reaped=False,adopted=True))
                        h['identity']=ident
                        try:owned.send_owned(h,sig)
                        except BaseException as exc:cleanup_errors.append(str(exc))
                until=time.monotonic()+seconds
                while time.monotonic()<until:
                    try:child,status,usage=os.wait4(-1,os.WNOHANG)
                    except ChildProcessError:break
                    if child:
                        if child not in handles:
                            handles[child]=dict(pid=child,identity=None,reaped=True,adopted=True)
                        handles[child].update(reaped=True,exit_code=os.waitstatus_to_exitcode(status))
                    else:time.sleep(.02)
            exitcode=driver.get('exit_code') if driver is not None else None
            alive=[int(x) for x in (Path('/proc/self/task')/str(os.getpid())/'children').read_text().split()]
            if alive:cleanup_errors.append('OWNED_PROCESS_STILL_PRESENT:'+str(alive))
            owned.atomic_json(control/'launcher_terminal.json',dict(
                status='COMPLETE_MANDATORY_STOP' if stop is None and not cleanup_errors else 'STOP',
                native_exit_code=exitcode,first_stop=stop,cleanup_errors=cleanup_errors,
                all_reaped=not alive,wall_seconds=time.monotonic()-start,
                peak_aggregate_RSS_bytes=peak,max_monitor_gap_seconds=gap_max,
                last_observer_memory=last_observer_memory,
                automatic_retry=False,next_stage_authorized=False))
        finally:
            for sig,handler in previous.items():signal.signal(sig,handler)
            signal.pthread_sigmask(signal.SIG_SETMASK,mask)
    if stop or cleanup_errors:raise RuntimeError(stop or 'OWNED_CLEANUP_UNCONFIRMED')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','grant','fresh-gate','control'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();config=runner.read_json(args.config);grant=runner.read_json(args.grant)
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    runner.verify_grant(config,grant,config_sha256=runner.sha(args.config),source=source,
                        fresh=runner.read_json(args.fresh_gate))
    if config.get('phase_a_parallel') is not True:raise ValueError('PHASE_A_ONLY')
    for name,h in config['source_hashes'].items():
        if runner.sha(ROOT/name)!=h:raise ValueError('LAUNCHER_SOURCE_CHANGED')
    for path in (args.config,args.grant,args.fresh_gate,args.control):
        if not path.is_absolute() or path.resolve()!=path or not path.is_relative_to('/home/AbeHiromu/projects'):
            raise ValueError('PRIVATE_ABSOLUTE_PATH_REQUIRED')
    if str(args.control)!=config['control'] or args.control.exists():raise ValueError('NEW_BOUND_CONTROL_REQUIRED')
    if Path(config['outputs']['signal']).exists() or Path(config['locks']['signal']).exists():
        raise ValueError('ONE_SHOT_ALREADY_SPENT')
    fs=os.statvfs(args.control.parent)
    if fs.f_bavail*fs.f_frsize<config['resources']['stage_storage_bytes']+2*2**30 or fs.f_favail<4096:
        raise ValueError('FRESH_STORAGE_GATE')
    env=os.environ.copy()
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS',
                'NUMBA_NUM_THREADS','QISKIT_NUM_PROCS','RAYON_NUM_THREADS'):env[key]='1'
    env['PYTHONDONTWRITEBYTECODE']='1'
    command=[config['python'],'-B',str(ROOT/'scripts/track_a_v1_runner.py'),'--execute-approved',
        '--config',str(args.config),'--grant',str(args.grant),'--fresh-gate',str(args.fresh_gate),
        '--output',config['outputs']['signal']]
    # Fork the driver before binding the observer's own CPU; the driver gets
    # the complete sealed allowed set and binds itself after verification.
    old=os.sched_getaffinity(0)
    try:
        os.sched_setaffinity(0,{grant['monitor_cpu']})
        supervise(command,env=env,control=args.control,output=config['outputs']['signal'],
            limits=grant['resources'],allowed_cpus=grant['allowed_cpus'])
    finally:os.sched_setaffinity(0,old)


if __name__=='__main__':main()
