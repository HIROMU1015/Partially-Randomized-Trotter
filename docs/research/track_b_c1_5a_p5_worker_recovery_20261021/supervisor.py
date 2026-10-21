"""Exec children, record wait4 status/signals and preserve first failure. No retries."""
import os
from pathlib import Path
import signal
import subprocess
import time
from common import journal,save,new_marker,proc_info,cgroup,require

def supervise(tasks,directory,workers,*,exclusive=True):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    if exclusive:new_marker(directory/'supervisor_marker.json',dict(pid=os.getpid(),retry=0))
    require(workers>0 and len({t['id'] for t in tasks})==len(tasks),'UNIQUE_TASK_PLAN')
    began=time.monotonic();before=cgroup();save(directory/'cgroup_before.json',before)
    pending=list(tasks);active={};results=[];failure=None
    with (directory/'observations.jsonl').open('a') as observations:
        try:
            while pending or active:
                while pending and len(active)<workers and failure is None:
                    task=pending.pop(0);name=str(task['id'])
                    out=(directory/(name+'.stdout.log')).open('xb');err=(directory/(name+'.stderr.log')).open('xb')
                    try:
                        child=subprocess.Popen(task['cmd'],stdout=out,stderr=err,env=task.get('env'),
                            cwd=task.get('cwd'),start_new_session=True)
                    except BaseException:
                        out.close();err.close();raise
                    active[child.pid]=dict(child=child,task=task,out=out,err=err,started=time.monotonic())
                    journal(directory/'events.jsonl',event='START',task_id=task['id'],worker_pid=child.pid,command=task['cmd'])
                for pid,row in list(active.items()):
                    waited,status,usage=os.wait4(pid,os.WNOHANG)
                    if not waited:continue
                    code=os.waitstatus_to_exitcode(status);row['child'].returncode=code
                    row['out'].close();row['err'].close();del active[pid]
                    receipt=dict(task_id=row['task']['id'],worker_pid=pid,exit_code=code,
                        signal=(-code if code<0 else None),raw_wait_status=status,
                        wall_seconds=time.monotonic()-row['started'],user_CPU=usage.ru_utime,
                        system_CPU=usage.ru_stime,max_RSS_KiB=usage.ru_maxrss)
                    results.append(receipt);save(directory/(str(row['task']['id'])+'.exit.json'),receipt)
                    journal(directory/'events.jsonl',event='EXIT',**receipt)
                    if code!=0 and failure is None:
                        failure=receipt;save(directory/'first_failure.json',dict(**receipt,technical_failure_not_zero=True))
                observations.write(__import__('json').dumps(dict(monotonic=time.monotonic(),
                    processes=[proc_info(pid) for pid in active],cgroup=cgroup()))+'\n');observations.flush()
                if failure is not None:break
                if active:time.sleep(.2)
        except BaseException as exc:
            import traceback
            if failure is None:
                failure=dict(supervisor_exception=repr(exc),traceback=traceback.format_exc(),
                    technical_failure_not_zero=True)
                save(directory/'first_failure.json',failure)
        finally:
            if active:
                for pid in active:
                    try:os.killpg(pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                deadline=time.monotonic()+5
                for pid,row in list(active.items()):
                    while True:
                        waited,status,usage=os.wait4(pid,os.WNOHANG)
                        if waited:break
                        if time.monotonic()>=deadline:
                            try:os.killpg(pid,signal.SIGKILL)
                            except ProcessLookupError:pass
                            _,status,usage=os.wait4(pid,0);break
                        time.sleep(.05)
                    code=os.waitstatus_to_exitcode(status);row['child'].returncode=code
                    row['out'].close();row['err'].close()
                    receipt=dict(task_id=row['task']['id'],worker_pid=pid,exit_code=code,
                        signal=(-code if code<0 else None),raw_wait_status=status,
                        user_CPU=usage.ru_utime,system_CPU=usage.ru_stime,max_RSS_KiB=usage.ru_maxrss,
                        cancelled_by_own_supervisor=True)
                    results.append(receipt);save(directory/(str(row['task']['id'])+'.exit.json'),receipt)
                    journal(directory/'events.jsonl',event='CANCELLED_EXIT',**receipt)
    after=cgroup();save(directory/'cgroup_after.json',after)
    delta={}
    for scope,values in before.items():
        def counts(text):
            try:return {k:int(v) for k,v in (line.split() for line in text.splitlines())}
            except ValueError:return {}
        a=counts(values.get('memory.events',''));b=counts(after.get(scope,{}).get('memory.events',''))
        delta[scope]={k:b.get(k,0)-v for k,v in a.items()}
    summary=dict(complete=failure is None and not pending and len(results)==len(tasks),first_failure=failure,
        planned=[t['id'] for t in tasks],completed=[r['task_id'] for r in results if r['exit_code']==0],
        exits=results,not_completed=[t['id'] for t in tasks if t['id'] not in {r['task_id'] for r in results if r['exit_code']==0}],
        supervised_wall_seconds=time.monotonic()-began,memory_events_delta=delta,retry=0,failed_tasks_not_resubmitted=True)
    save(directory/'summary.json',summary)
    return summary
