"""Owned fork/COW scheduler. Global trajectory demand includes BOTH compile axes.

No CPU affinity, thread environment, OS scheduler or shared resource settings are changed.
Linux subreaper affects only this coordinator and its own descendants.
"""
import ctypes
import math
import os
from pathlib import Path
import signal
import time
import traceback
from .accounting import atomic,read
from .sampling import require


def identity(pid):
    try:
        p=Path('/proc')/str(pid);stat=p.joinpath('stat').read_text().rsplit(')',1)[1].split()
        return (pid,int(stat[19]),p.stat().st_uid,int(stat[1]))
    except (FileNotFoundError,ProcessLookupError):return None


def single_thread():return len(list(Path('/proc/self/task').iterdir()))==1


def map_fork(tasks,callback,directory,capacity,demand,*,new_group=True,cancel=lambda:False,observe=lambda rows:None,resources_pre_admitted=False):
    """capacity returns current CPU slots/private memory/I-O slots; no fixed W cap.

    callback(task, task_directory, axis_width) runs once. Nested maps have their own
    wait4 receipts. A failed parent cannot leave grandchildren behind: the outer
    coordinator adopts and reaps descendants in its owned process groups.
    """
    require(single_thread(),'UNSAFE_MULTITHREADED_FORK_PARENT')
    if new_group:
        require(ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)==0,'SUBREAPER_REQUIRED')
    root=Path(directory);root.mkdir(parents=True,exist_ok=False)
    require(len({t['id'] for t in tasks})==len(tasks),'DUPLICATE_TASK_ID')
    require(all(Path(t['id']).name==t['id'] and t['id'] not in ('.','..') for t in tasks),'TASK_PATH')
    pending=sorted(tasks,key=lambda t:t['id']);active={};done=[];adopted=[];error=None
    oldhandlers={}
    def interrupt(signum,frame):raise RuntimeError('OWN_COORDINATOR_SIGNAL_'+str(signum))
    for sig in (signal.SIGTERM,signal.SIGINT):oldhandlers[sig]=signal.signal(sig,interrupt)
    def finish(pid,status,usage):
        h=active.pop(pid);h.update(exit_code=os.waitstatus_to_exitcode(status),reaped=True,
            CPU_seconds=usage.ru_utime+usage.ru_stime,peak_RSS_bytes=usage.ru_maxrss*1024)
        done.append(h)
        if h['exit_code']!=0:raise RuntimeError('OWN_WORKER_FAILED:'+h['id'])
        value=read(Path(h['directory'])/'result.json');h['result']=value
    try:
        while pending or active:
            if cancel():raise RuntimeError('OWN_JOB_CANCELLED')
            for pid in list(active):
                got,status,usage=os.wait4(pid,os.WNOHANG)
                if got:finish(got,status,usage)
            observe(list(active.values()))
            slots=capacity()
            cpu_used=sum(h['cpu_demand'] for h in active.values())
            # Available memory is already net of current RSS; reserve future growth.
            growth=0
            for h in active.values():
                private=0
                p=Path('/proc')/str(h['pid'])/'smaps_rollup'
                try:
                    fields=dict(line.split(':',1) for line in p.read_text().splitlines() if ':' in line)
                    private=sum(int(fields.get(k,'0 kB').split()[0])*1024 for k in ('Private_Clean','Private_Dirty','Private_Hugetlb'))
                except (FileNotFoundError,ProcessLookupError,PermissionError):pass
                growth+=max(0,h['memory_demand']-private)
            while pending:
                task=pending[0];cost=demand(task,slots)
                require(cost['cpu']>0 and (cost['memory']>0 or resources_pre_admitted and cost['memory']==0) and cost['io']>0,'UNJUSTIFIED_RESOURCE_DEMAND')
                if cpu_used+cost['cpu']>slots['cpu'] or (not resources_pre_admitted and growth+cost['memory']>slots['memory']) or sum(h['io_demand'] for h in active.values())+cost['io']>slots['io']:break
                require(single_thread(),'FORK_PARENT_BECAME_MULTITHREADED')
                pending.pop(0);work=root/task['id'];work.mkdir()
                mask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGTERM,signal.SIGINT})
                try:
                    pid=os.fork()
                    if pid==0:
                        if new_group:os.setpgid(0,0)
                        signal.pthread_sigmask(signal.SIG_SETMASK,mask)
                        try:
                            value=callback(task,work,cost['axis_width']);atomic(work/'result.json',value);os._exit(0)
                        except BaseException:
                            atomic(work/'failure.json',{'error':traceback.format_exc()});os._exit(1)
                    if new_group:
                        try:os.setpgid(pid,pid)
                        except (PermissionError,ProcessLookupError):pass
                    active[pid]=dict(id=task['id'],pid=pid,identity=identity(pid),directory=str(work),
                        cpu_demand=cost['cpu'],memory_demand=cost['memory'],io_demand=cost['io'],axis_width=cost['axis_width'],
                        group=pid if new_group else os.getpgrp(),reaped=False)
                    atomic(work/'started.json',active[pid]);cpu_used+=cost['cpu'];growth+=cost['memory']
                finally:signal.pthread_sigmask(signal.SIG_SETMASK,mask)
            if pending or active:time.sleep(.02)
    except BaseException as exc:error=str(exc)
    finally:
        mask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGTERM,signal.SIGINT})
        try:
            owned=list(active.values())+done
            if error:
                # Groups are created by us; an extant child PID is checked before signaling.
                for sig in (signal.SIGTERM,signal.SIGKILL):
                    groups=set()
                    for h in active.values():
                        current=identity(h['pid'])
                        if current is not None and current==h['identity']:groups.add(h['group'])
                    children=Path('/proc/self/task')/str(os.getpid())/'children'
                    for token in children.read_text().split():
                        pid=int(token)
                        try:
                            group=os.getpgid(pid);current=identity(pid)
                            if current and current[2]==os.getuid() and group in {h['group'] for h in owned}:groups.add(group)
                        except ProcessLookupError:pass
                    for group in groups:
                        try:
                            if new_group:os.killpg(group,sig)
                            else:
                                for h in active.values():
                                    if identity(h['pid'])==h['identity']:os.kill(h['pid'],sig)
                        except ProcessLookupError:pass
                    if sig==signal.SIGTERM:time.sleep(.05)
            for pid,h in list(active.items()):
                got,status,usage=os.wait4(pid,0);h.update(reaped=True,exit_code=os.waitstatus_to_exitcode(status));done.append(h);active.pop(pid)
            if new_group:
                # Direct adopted descendants only; other groups are never signaled or waited.
                children=Path('/proc/self/task')/str(os.getpid())/'children'
                while True:
                    candidates=[]
                    for token in children.read_text().split():
                        pid=int(token)
                        try:group=os.getpgid(pid)
                        except ProcessLookupError:continue
                        if group in {h['group'] for h in owned}:candidates.append(pid)
                    if not candidates:break
                    if error is None:error='UNEXPECTED_ORPHAN_DESCENDANTS'
                    for pid in candidates:
                        require((Path('/proc')/str(pid)).stat().st_uid==os.getuid(),'ADOPTED_UID')
                        try:os.kill(pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                        got,status,usage=os.wait4(pid,0);adopted.append(dict(pid=got,exit_code=os.waitstatus_to_exitcode(status),reaped=True))
        finally:
            for sig,handler in oldhandlers.items():signal.signal(sig,handler)
            signal.pthread_sigmask(signal.SIG_SETMASK,mask)
    receipt=dict(status='STOP' if error else 'PASS',error=error,workers=sorted(done,key=lambda h:h['id']),
        adopted_descendants=adopted,unsubmitted=[t['id'] for t in pending],all_reaped=not active,
        retry=False,resume=False,shared_settings_changed=False,arbitrary_resource_cap=False,memory_admission_covered_by_outer=resources_pre_admitted)
    atomic(root/'native_receipt.json',receipt)
    if error:atomic(root/'STOP.json',dict(status='MANDATORY_STOP',reason=error,retry=False,resume=False))
    return receipt
