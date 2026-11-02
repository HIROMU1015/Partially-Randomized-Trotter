"""H7-only exit-race fix. Live observation loss still fails closed.

Keep old H5 scientific/monitor sources immutable. No process is created at import.
"""
import json
import os
import time
from pathlib import Path
from ..h5_open_shell.execution import spawn,cleanup,write_json


def worker_snapshot(pid):
    memory={};threads=None
    for line in Path(f'/proc/{pid}/status').read_text().splitlines():
        key,_,value=line.partition(':')
        if key in ('VmRSS','VmSize','VmHWM','VmPeak'):memory[key+'_bytes']=int(value.split()[0])*1024
        elif key=='Threads':threads=int(value)
    if 'VmRSS_bytes' not in memory or 'VmSize_bytes' not in memory:raise ValueError('PROCESS_MEMORY_MISSING')
    if threads is None or threads<1:raise ValueError('PROCESS_THREADS_MISSING')
    if threads>1:raise RuntimeError('INTERNAL_THREAD_LIMIT')
    return dict(memory=memory,threads=threads)


def owned_snapshot(child,*,identity=None,snapshot=None,clock=time.monotonic,pause=time.sleep,grace=.2):
    from ..h6_server_resources_v1 import process_identity
    identity=identity or process_identity;snapshot=snapshot or worker_snapshot
    deadline=clock()+grace
    while True:
        if child.exited():return None
        try:
            current=identity(child.pid)
            if current['pid']!=child.pid or current['start_ticks']!=child.identity['start_ticks']:
                raise RuntimeError('OWN_IDENTITY_CHANGED')
            result=snapshot(child.pid)
            current=identity(child.pid)
            if current['start_ticks']!=child.identity['start_ticks']:raise RuntimeError('OWN_IDENTITY_CHANGED')
            return result
        except (FileNotFoundError,ValueError) as error:
            if isinstance(error,ValueError) and str(error) not in ('PROCESS_MEMORY_MISSING','PROCESS_THREADS_MISSING'):raise
            if child.exited():return None
            # Linux may release mm before waitid exposes the exit. Recheck only
            # this pinned, unreaped child for <=.2s; never synthesize a zero RSS.
            if clock()>=deadline:raise
            pause(.01)


def supervise(output,registered,function,cpus,limits,baseline,*,spawn_fn=spawn,observe=None):
    """Fixed registration only; completion or first STOP, never retry/resume."""
    from ..h6_server_resources_v1 import host_memory,process_memory,memory_reason,process_identity
    pending=list(enumerate(registered));active=[];completed=[];retired=[];first=None;errors=[]
    previous=time.monotonic()
    policy=dict(fresh_host_available_bytes=0,fresh_effective_available_bytes=0,
        running_host_available_bytes=limits['headroom_bytes'],running_effective_available_bytes=limits['headroom_bytes'],
        host_full_psi_stop_percent=1.,cgroup_full_psi_stop_percent=0.)
    observe=observe or host_memory
    log=Path(output)/'resource_observations.jsonl'
    try:
        with log.open('xb') as journal:
            while pending or active:
                start=time.monotonic();interval=start-previous;previous=start
                if interval>5:raise RuntimeError('OBSERVATION_STALE')
                memory=observe();reason=memory_reason(memory,baseline,policy)
                if reason:raise RuntimeError(reason)
                workers=[];aggregate=0
                for child in list(active):
                    if child.exited():
                        code=child.reap();active.remove(child);retired.append(dict(pid=child.pid,identity=child.identity,exit_code=code,reaped=True))
                        terminal=json.loads((child.directory/'terminal.json').read_bytes())
                        if code or terminal['status']!='COMPLETE':raise RuntimeError('WORKER_STOP:'+str(child.pid))
                        result=json.loads((child.directory/'result.json').read_bytes())
                        if result['task']!=child.task:raise RuntimeError('WORKER_TASK_BINDING')
                        completed.append(result);continue
                    sample=owned_snapshot(child)
                    if sample is None:
                        # Exit is proven by waitid, not inferred from missing RSS.
                        # The next pass performs the normal terminal/exit/reap path.
                        continue
                    usage=sample['memory']
                    aggregate+=usage['VmRSS_bytes'];workers.append(dict(identity=child.identity,memory=usage,threads=sample['threads']))
                    if usage['VmRSS_bytes']>limits['worker_RSS_bytes'] or usage['VmSize_bytes']>limits['AS_bytes']:
                        raise RuntimeError('WORKER_MEMORY_LIMIT')
                if aggregate>limits['aggregate_RSS_bytes']:raise RuntimeError('AGGREGATE_RSS_LIMIT')
                stat=os.statvfs(output)
                if stat.f_bavail*stat.f_frsize<limits['minimum_free_bytes'] or stat.f_favail<1000:raise RuntimeError('FILESYSTEM_HEADROOM')
                size=sum(p.stat().st_size for p in Path(output).rglob('*') if p.is_file())
                if size>limits['output_bytes']:raise RuntimeError('OUTPUT_LIMIT')
                duration=time.monotonic()-start
                if duration>5:raise RuntimeError('OBSERVATION_DURATION')
                record=dict(monotonic=start,interval_seconds=interval,observation_seconds=duration,
                    memory=memory,workers=workers,aggregate_RSS_bytes=aggregate,first_stop=None)
                raw=(json.dumps(record,allow_nan=False)+'\n').encode()
                if journal.tell()+len(raw)>64*2**20:raise RuntimeError('MONITOR_OUTPUT_CAP')
                journal.write(raw);journal.flush()
                occupied={child.task['_cpu'] for child in active}
                for cpu in cpus:
                    if cpu in occupied or not pending:continue
                    index,task=pending.pop(0);task=dict(task,_cpu=cpu)
                    child=spawn_fn(Path(output)/f'group_{index:03d}',task,function,cpu,limits);active.append(child)
                if active:time.sleep(max(.01,1-(time.monotonic()-start)))
    except BaseException as e:first=repr(e)
    finally:
        errors=cleanup(active)
        retired.extend(dict(pid=c.pid,identity=c.identity,exit_code=c.exit_code,reaped=c.reaped) for c in active)
        summary=dict(completed=completed,registered_count=len(registered),pending=len(pending),
            retired=retired,first_stop=first,cleanup_errors=errors,mandatory_stop=first is not None)
        write_json(Path(output)/'summary.json',summary)
    if first or errors:raise RuntimeError('H7_FAIL_CLOSED:'+str(first or errors))
    if len(completed)!=len(registered):raise RuntimeError('MISSING_COMPLETIONS')
    return summary
