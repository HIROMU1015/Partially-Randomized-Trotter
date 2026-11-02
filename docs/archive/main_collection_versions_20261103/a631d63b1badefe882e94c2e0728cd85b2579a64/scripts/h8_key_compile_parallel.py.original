"""Owned fork workers for the six-axis direct-compile scope. No scientific imports or launch.

The caller must validate a fresh user grant before invoking map_owned. Inputs
are inherited through copy-on-write; workers write exclusive private artifacts.
Historical RSS/AS/CPU/wall limits are intentionally not inherited.
Resource observations do not impose process-thread or PSI STOP thresholds.
Unlike a process pool, every attempt has its own PID/start identity and wait4.
"""
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import resource
import math
import signal
import sys
import time


def identity(pid):
    try:
        path=Path('/proc')/str(pid)
        fields=(path/'stat').read_text().rsplit(')',1)[1].split()
        return (pid,int(fields[19]),int(fields[1]),path.stat().st_uid)
    except (FileNotFoundError,ProcessLookupError):return None


def send_owned(handle,sig):
    if handle['reaped']:return
    current=identity(handle['pid'])
    if current is None:return
    if current!=handle['identity']:raise RuntimeError('OWNED_WORKER_IDENTITY_CHANGED')
    try:fd=os.pidfd_open(handle['pid'])
    except ProcessLookupError:return
    try:
        if identity(handle['pid'])!=handle['identity']:raise RuntimeError('PIDFD_IDENTITY_CHANGED')
        signal.pidfd_send_signal(fd,sig)
    except ProcessLookupError:pass
    finally:os.close(fd)


def wait_owned(h):
    if h['reaped']:return
    pid,status,usage=os.wait4(h['pid'],os.WNOHANG)
    if pid:
        h.update(reaped=True,exit_code=os.waitstatus_to_exitcode(status),
                 cpu_seconds=usage.ru_utime+usage.ru_stime,peak_RSS_bytes=usage.ru_maxrss*1024)


def worker_failure(h):
    """Keep the first failed child's cause across nested preparation maps."""
    path=Path(h['directory'])/'task_STOP.json';cause=None
    try:
        if path.is_file() and path.stat().st_size<=2*2**20:
            value=json.loads(path.read_bytes()).get('error')
            if isinstance(value,str):cause=value
    except (OSError,ValueError):pass
    return RuntimeError('WORKER_FAILED:'+h['task_id']+(':'+cause if cause else ''))


def atomic_json(path,value):
    raw=json.dumps(value,sort_keys=True,allow_nan=False).encode()
    if len(raw)>2*2**20:raise RuntimeError('WORKER_CONTROL_ARTIFACT_TOO_LARGE')
    temp=path.with_suffix('.tmp')
    with temp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    os.rename(temp,path)


def child_main(task,callback,directory,cpu,limits,parent,deadline,font_seed,old_mask):
    status=2
    try:
        if os.getppid()!=parent:raise RuntimeError('PARENT_EXITED_BEFORE_WORKER_START')
        libc=ctypes.CDLL(None,use_errno=True)
        if libc.prctl(1,int(signal.SIGKILL),0,0,0)!=0:raise OSError(ctypes.get_errno(),'PDEATHSIG')
        if os.getppid()!=parent:raise RuntimeError('PARENT_DEATH_RACE')
        signal.setitimer(signal.ITIMER_REAL,0)
        os.sched_setaffinity(0,{cpu})
        def stop(*args):raise RuntimeError('WORKER_STOP_SIGNAL_OR_DEADLINE')
        for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM):signal.signal(sig,stop)
        scratch=directory/'scratch';scratch.mkdir();mpl=scratch/'mpl';mpl.mkdir()
        for key,name in (('NUMBA_CACHE_DIR','numba'),('TMPDIR','tmp'),('XDG_CACHE_HOME','cache')):
            p=scratch/name;p.mkdir();os.environ[key]=str(p)
        os.environ['MPLCONFIGDIR']=str(mpl)
        if font_seed:
            raw=Path(font_seed['path']).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=font_seed['sha256']:raise RuntimeError('FONT_SEED_CHANGED')
            (mpl/font_seed['name']).write_bytes(raw)
        import tempfile
        tempfile.tempdir=os.environ['TMPDIR']
        for stream,fd in (('stdout',1),('stderr',2)):
            with (directory/(stream+'.log')).open('xb',buffering=0) as log:os.dup2(log.fileno(),fd)
        signal.pthread_sigmask(signal.SIG_SETMASK,old_mask)
        value=callback(task,directory)
        atomic_json(directory/'task_result.json',value)
        status=0
    except BaseException as exc:
        try:atomic_json(directory/'task_STOP.json',dict(error=type(exc).__name__+':'+str(exc)[:2000]))
        except BaseException:pass
    finally:
        try:sys.stdout.flush();sys.stderr.flush()
        finally:os._exit(status)


def map_owned(tasks,callback,*,cpus,limits,directory,deadline,guard=lambda:None,
              reserve=lambda task:None,font_seed=None,prior_child_CPU_seconds=0.,
              ready=lambda task:True,on_complete=lambda task,handle:None):
    """Run each task once. On failure terminate/reap only owned children.

    Results are accepted only after native exit 0 and same-parent wait4. No
    automatic retry, resumed partial reuse, shared result stream or pickle.
    """
    if (not cpus or len(cpus)!=len(set(cpus)) or any(type(x) is not int or x<0 for x in cpus)
            or len(cpus)>limits['max_workers']):raise ValueError('INVALID_WORKER_CPU_ASSIGNMENT')
    ids=[t['id'] for t in tasks]
    if len(ids)!=len(set(ids)) or any(not x or x in ('.','..') or Path(x).name!=x for x in ids):
        raise ValueError('UNIQUE_SAFE_TASK_IDS_REQUIRED')
    directory=Path(directory);directory.mkdir()
    parent=os.getpid();handles=[];queue=list(tasks);first_error=None
    by_id={t['id']:t for t in tasks}
    def completed(h):
        if h.get('completion_notified'):return
        if h['exit_code']!=0:raise worker_failure(h)
        result=Path(h['directory'])/'task_result.json'
        if not result.is_file():raise RuntimeError('MISSING_COMPLETE_WORKER_RESULT')
        h['result_sha256']=hashlib.sha256(result.read_bytes()).hexdigest()
        on_complete(by_id[h['task_id']],h)
        h['completion_notified']=True
    ticks=os.sysconf('SC_CLK_TCK');cpu_total=prior_child_CPU_seconds
    try:
        while queue or any(not h['reaped'] for h in handles):
            guard()
            active=[h for h in handles if not h['reaped']]
            while queue and len(active)<len(cpus):
                if len(list(Path('/proc/self/task').iterdir()))!=1:
                    raise RuntimeError('FORK_REQUIRES_SINGLE_THREAD_PARENT')
                position=next((i for i,t in enumerate(queue) if ready(t)),None)
                if position is None:break
                task=queue.pop(position);reserve(task)
                cpu=next(x for x in cpus if x not in {h['cpu'] for h in active})
                work=directory/task['id'];work.mkdir()
                h=dict(task_id=task['id'],cpu=cpu,directory=str(work),pid=None,identity=None,
                    reaped=False,exit_code=None,cpu_seconds=0.,peak_RSS_bytes=0,started=time.monotonic())
                handles.append(h)
                mask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGTERM,signal.SIGINT,signal.SIGALRM})
                try:
                    pid=os.fork()
                    if pid==0:
                        child_main(task,callback,work,cpu,limits,parent,deadline,font_seed,mask)
                    h['pid']=pid;h['identity']=identity(pid)
                    if h['identity'] is None or h['identity'][2:]!=(parent,os.getuid()):
                        raise RuntimeError('WORKER_PARENT_UID_BINDING')
                    atomic_json(work/'started.json',dict(pid=pid,identity=h['identity'],cpu=cpu,task=task['id']))
                finally:signal.pthread_sigmask(signal.SIG_SETMASK,mask)
                active.append(h)
            for h in handles:
                if h['reaped']:continue
                wait_owned(h)
                if h['reaped']:
                    completed(h)
                    continue
                current=identity(h['pid'])
                if current is None:continue
                if current!=h['identity']:raise RuntimeError('WORKER_IDENTITY_CHANGED')
                try:
                    proc=Path('/proc')/str(h['pid'])
                    fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
                    if fields[0]=='Z':continue
                    values=dict(line.split(':',1) for line in (proc/'status').read_text().splitlines() if ':' in line)
                    rss=int(values['VmRSS'].split()[0])*1024
                    h['peak_RSS_bytes']=max(h['peak_RSS_bytes'],rss)
                    h['cpu_seconds']=(int(fields[11])+int(fields[12]))/ticks
                    h['observed_threads']=int(values['Threads'])
                    if time.monotonic()-h['started']>5 and os.sched_getaffinity(h['pid'])!={h['cpu']}:
                        raise RuntimeError('WORKER_AFFINITY_CHANGED')
                    h['last_resource_observation']=time.monotonic()
                except (FileNotFoundError,ProcessLookupError,KeyError):
                    wait_owned(h)
                    h['observation_stale']=not h['reaped']
            usage=resource.getrusage(resource.RUSAGE_SELF)
            if queue and all(h['reaped'] for h in handles) and not any(ready(t) for t in queue):
                raise RuntimeError('UNSATISFIED_TASK_DEPENDENCY')
            cpu_total=prior_child_CPU_seconds+sum(h['cpu_seconds'] for h in handles)
            if queue or any(not h['reaped'] for h in handles):time.sleep(.05)
    except BaseException as exc:
        first_error=exc
    finally:
        # No Python TERM/alarm interruption may abandon cleanup. The caller
        # exits after a failed map; all old/unrelated process identities stay out.
        mask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGTERM,signal.SIGINT,signal.SIGALRM})
        cleanup_errors=[]
        try:
            for sig,seconds in ((signal.SIGTERM,limits.get('cleanup_TERM_seconds',2.)),(signal.SIGKILL,2.)):
                for h in handles:
                    if h['pid'] is None:continue
                    try:
                        wait_owned(h)
                        if not h['reaped']:send_owned(h,sig)
                    except BaseException as exc:cleanup_errors.append(str(exc))
                until=time.monotonic()+seconds
                while any(h['pid'] is not None and not h['reaped'] for h in handles) and time.monotonic()<until:
                    for h in handles:
                        if h['pid'] is not None:
                            try:wait_owned(h)
                            except BaseException as exc:cleanup_errors.append(str(exc))
                    time.sleep(.02)
            for h in handles:
                if h['pid'] is not None and not h['reaped']:
                    try:
                        pid,status,usage=os.wait4(h['pid'],0)
                        h.update(reaped=True,exit_code=os.waitstatus_to_exitcode(status),cpu_seconds=usage.ru_utime+usage.ru_stime,peak_RSS_bytes=usage.ru_maxrss*1024)
                    except BaseException as exc:cleanup_errors.append(str(exc))
            clean=all(h['pid'] is None or h['reaped'] for h in handles)
            # An exit can be reaped by the /proc-race path or cleanup rather
            # than the main completion branch. Normalize every native receipt.
            if first_error is None:
                try:
                    for h in handles:
                        if h['pid'] is None or not h['reaped'] or h['exit_code']!=0:
                            raise worker_failure(h)
                        completed(h)
                except BaseException as exc:first_error=exc
            receipt=dict(status='PASS' if first_error is None and clean and not cleanup_errors else 'STOP',
                all_reaped=clean,cleanup_errors=cleanup_errors,workers=handles,automatic_retry=False,
                first_error=str(first_error) if first_error else None,
                child_CPU_seconds=prior_child_CPU_seconds+sum(h['cpu_seconds'] for h in handles))
            atomic_json(directory/'parallel_receipt.json',receipt)
        finally:signal.pthread_sigmask(signal.SIG_SETMASK,mask)
    if first_error is not None:raise first_error
    if not clean or cleanup_errors:raise RuntimeError('OWNED_CLEANUP_UNCONFIRMED')
    return receipt
