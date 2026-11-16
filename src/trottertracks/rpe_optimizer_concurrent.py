"""Work-conserving execution around the unchanged per-configuration search rule.

Only the search's chosen answer is adopted. Out-of-order completions stay in
their source-bound job directories. Finite budgets include reservations before
launch; running scientific workers are never cancelled to discard lookahead.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace

from . import rpe_method_optimizer as O


def _write(path, value):
    with Path(path).open('x') as f:json.dump(value,f,default=str);f.write('\n')


def _tree_rss(pid):
    """Sample only this executor's process trees; no shared-server traversal."""
    total=0;todo=[pid];seen=set()
    while todo:
        p=todo.pop()
        if p in seen:continue
        seen.add(p)
        try:
            for line in Path('/proc/%s/status'%p).read_text().splitlines():
                if line.startswith('VmRSS:'):total+=int(line.split()[1]);break
            todo.extend(map(int,Path('/proc/%s/task/%s/children'%(p,p)).read_text().split()))
        except (OSError,ValueError):pass
    return total


class ConcurrentExecutor(O.Executor):
    supports_nonblocking_prefetch=True
    def __init__(self,*args,budgets=None,science_approved=True,dp_workers=None,prefix_cache_directory=None,prefix_checkpoint_rounds=None,**kwargs):
        super().__init__(*args,**kwargs)
        if dp_workers is not None and (type(dp_workers)is not int or dp_workers<1):
            raise ValueError('Positive integer dp_workers required')
        self.run_dir.mkdir(parents=True,exist_ok=True)
        driver=(self.run_dir/'FAST_DRIVER.lock').open('a')
        try:fcntl.flock(driver,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            driver.close();raise ValueError('A fast driver already owns this run directory')
        self.capacity=max(1,kwargs.get('max_concurrent',1));self.dp_workers=dp_workers
        self.prefix_cache_directory=None if prefix_cache_directory is None else str(Path(prefix_cache_directory).resolve())
        self.prefix_checkpoint_rounds=prefix_checkpoint_rounds
        existing={}
        for method in self.routes:
            base=self.run_dir/'jobs'/method
            existing[method]=sum((d/'LAUNCH.json').exists()or(d/'STARTING.json').exists()for d in base.iterdir())if base.exists()else 0
        self.state=SimpleNamespace(cv=threading.Condition(threading.RLock()),tasks={},active=0,closed=False,
            failure=None,owner_clock={},clock=0,budgets=budgets or {},approved=science_approved,existing=existing,
            reserved=Counter(),metrics=Counter(),by_method={},pids={},driver=driver,peak_RSS_KiB=0,peak_active=0,adopted_jobs=set(),external={},monitor_stop=threading.Event())
        self.invocation=str(os.getpid())+'-'+str(time.time_ns())
        self.log(event='EXECUTOR_START',capacity=self.capacity,dp_workers=dp_workers)
        for method in self.routes:
            base=self.run_dir/'jobs'/method
            for claim in base.glob('*/STARTING.json'):
                if not(claim.parent/'LAUNCH.json').exists()and not(claim.parent/'OUTCOME.json').exists():
                    fcntl.flock(driver,fcntl.LOCK_UN);driver.close()
                    raise ValueError('Unresolved start requires review; no retry: '+str(claim.parent))
            for marker in base.glob('*/LAUNCH.json'):
                job=marker.parent
                try:launch=json.loads(marker.read_text())
                except (OSError,ValueError):continue
                if not(job/'PROCESS_RETURN.json').exists()and self.alive(launch,job):
                    self.state.external[job]=launch;self.state.pids[job]=launch['pid'];self.state.active+=1
                    self.for_method(method).log(event='RESUME_RUNNING',job=job.name,pid=launch['pid'])
        self.state.peak_active=self.state.active
        self.dispatcher=threading.Thread(target=self._dispatch,name='rpe-job-dispatch',daemon=False)
        self.dispatcher.start()
        self.monitor=threading.Thread(target=self._monitor,name='rpe-owned-process-monitor',daemon=True)
        self.monitor.start()

    def for_method(self,method):
        view=ConcurrentExecutor.__new__(ConcurrentExecutor);view.__dict__.update(self.__dict__);view.method=method
        return view

    def log(self,**event):
        with self.lock:
            with(self.run_dir/'PROGRESS.jsonl').open('a')as f:
                f.write(json.dumps(dict(UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                    epoch=time.time(),monotonic=time.monotonic(),invocation=self.invocation,method=self.method,**event),default=str)+'\n')

    def request(self,proposal,job,rule):
        req=super().request(proposal,job,rule)
        req['execution_profile']=dict(dp_workers=self.dp_workers)
        if self.prefix_cache_directory is not None:
            req['execution_profile'].update(prefix_cache_directory=self.prefix_cache_directory,
                prefix_checkpoint_rounds=self.prefix_checkpoint_rounds)
        return req

    def prefetch_state(self):
        """Current capacity and completed speculation; in-flight work is not waste."""
        with self.state.cv:
            tasks=[t for t in self.state.tasks.values() if self.method is None or t.view.method==self.method]
            speculative=[t for t in tasks if t.started and getattr(t,'speculative',False)]
            return dict(capacity=self.capacity,active_jobs=self.state.active,
                speculative_launched=len(speculative),
                speculative_completed=sum(t.status=='DONE' for t in speculative),
                speculative_running=sum(t.status=='RUNNING' for t in speculative),
                speculative_adopted=sum(t.job in self.state.adopted_jobs for t in speculative))

    def launched(self,method):
        base=self.run_dir/'jobs'/method
        return sum((d/'LAUNCH.json').exists()or(d/'STARTING.json').exists()for d in base.iterdir())if base.exists()else 0

    def has_job(self,proposal):
        job=self.job_dir(proposal)
        with self.state.cv:
            task=self.state.tasks.get(job)
            queued=task is not None and task.status!='CANCELLED'
        return queued or super().has_job(proposal) or(job/'STARTING.json').exists()

    def _enqueue(self,proposal,demand=False):
        job=self.job_dir(proposal);s=self.state
        with s.cv:
            if s.failure:raise RuntimeError('Executor persistence/start failure; no new launches: '+s.failure)
            old=s.tasks.get(job)
            if old and old.status!='CANCELLED':
                if demand:old.demand=True;s.cv.notify_all()
                return old
            if(job/'OUTCOME.json').exists()or(job/'LAUNCH.json').exists()or(job/'STARTING.json').exists():return None
            rule=self.route(proposal['configuration_id'])
            if rule is None or self.prior_job(proposal) is not None:return None
            cap=s.budgets.get(self.method)
            cap=None if cap in(None,'UNLIMITED')else cap
            if not s.approved or(cap is not None and s.existing.get(self.method,0)+s.reserved[self.method]>=cap):return False
            if s.closed:raise RuntimeError('Executor is closed')
            task=SimpleNamespace(view=self,job=job,proposal=deepcopy(proposal),rule=rule,
                owner=(self.method,proposal['configuration_id']),demand=demand,status='QUEUED',
                event=threading.Event(),enqueue=time.monotonic(),error=None,started=False)
            s.tasks[job]=task;s.reserved[self.method]+=1;s.metrics['submitted']+=1
            try:self.log(event='ENQUEUE',job=job.name,configuration=proposal['configuration_id'],demand=demand, origin=proposal.get('origin',{}).get('kind'))
            except OSError as exc:
                self._stop_after_failure(exc);raise
            s.cv.notify_all();return task

    def prefetch(self,proposals,scope='CONFIGURATION'):
        """Nonblocking. Retire stale *queued* speculation, never a running job."""
        if scope not in ('CONFIGURATION','METHOD'):raise ValueError('Unknown prefetch scope')
        if not proposals and scope != 'METHOD':return
        desired={self.job_dir(p)for p in proposals}
        owner=(self.method,proposals[0]['configuration_id']) if proposals else None
        with self.state.cv:
            for t in self.state.tasks.values():
                same_scope=(t.view.method==self.method) if scope=='METHOD' else t.owner==owner
                if same_scope and t.status=='QUEUED'and not t.demand and t.job not in desired:self._cancel(t)
        for i,p in enumerate(proposals):self._enqueue(p,demand=(i==0))

    def _cancel(self,t):
        t.status='CANCELLED';self.state.reserved[t.view.method]-=1
        self.state.metrics['queued_cancelled']+=1;t.event.set()
        t.view.log(event='CANCEL_QUEUED',job=t.job.name)

    def _dispatch(self):
        s=self.state
        while True:
            with s.cv:
                queue=[t for t in s.tasks.values()if t.status=='QUEUED']
                if s.closed and not queue:return
                if s.failure or not queue or s.active>=self.capacity:
                    s.cv.wait(.1);continue
                t=min(queue,key=lambda t:(not t.demand,s.owner_clock.get(t.owner,-1),t.enqueue))
                s.clock+=1;s.owner_clock[t.owner]=s.clock;t.status='RUNNING';t.speculative=not t.demand;s.active+=1
                s.peak_active=max(s.peak_active,s.active)
            self._start(t)

    def _stop_after_failure(self,error):
        # Persistence/start failure stops new launches; existing children finish.
        with self.state.cv:
            self.state.failure=str(error)
            for task in self.state.tasks.values():
                if task.status=='QUEUED':
                    task.error=str(error)
                    try:self._cancel(task)
                    except OSError:pass  # the queue event is already released
            self.state.cv.notify_all()

    def _start(self,t):
        s=self.state;v=t.view;job=t.job;start=time.time();proc=None
        try:
            job.mkdir(parents=True,exist_ok=True)
            if not(job/'REQUEST.json').exists():_write(job/'REQUEST.json',v.request(t.proposal,job,t.rule))
            # A crash in the Popen -> LAUNCH interval is unresolved, never a retry.
            _write(job/'STARTING.json',dict(parent_pid=os.getpid(),epoch=start,automatic_retry=False))
            proc=v.run_adapter(job,t.rule,'execute');t.started=True
            with self.lock:self.uncounted.add(job)
            with s.cv:
                s.metrics['launched']+=1;s.by_method.setdefault(v.method,Counter())['launched']+=1
                s.pids[job]=proc.pid
                if t.speculative:s.metrics['speculative_launched']+=1
            _write(job/'LAUNCH.json',dict(pid=proc.pid,started=start,kind=t.rule['kind']))
            v.log(event='LAUNCH',job=job.name,configuration=t.proposal['configuration_id'],
                  pid=proc.pid,queue_wait_seconds=time.monotonic()-t.enqueue,active=s.active,demand=not t.speculative)
        except Exception as exc:
            t.error=repr(exc);self._stop_after_failure(t.error)
            if proc is None:self._finished(t,None,start);return
            # Popen succeeded: retain its slot and wait even if LAUNCH could not save.
        def wait():
            code=None
            try:
                code=proc.wait()
                _write(job/'PROCESS_RETURN.json',dict(returncode=code if code>=0 else None,signal=-code if code<0 else None,
                                                    wall_seconds=time.time()-start,automatic_retry=False))
            except Exception as exc:
                t.error=repr(exc);self._stop_after_failure(t.error)
            finally:self._finished(t,code,start)
        thread=threading.Thread(target=wait,name='rpe-wait-'+job.name,daemon=False)
        with self.lock:self.waiters.append(thread)
        thread.start()

    def _finished(self,t,code,start):
        s=self.state
        with s.cv:
            t.status='DONE';s.active-=1;s.pids.pop(t.job,None);s.metrics['finished']+=1
            try:t.view.log(event='EXIT',job=t.job.name,returncode=code,error=t.error,wall_seconds=time.time()-start)
            except OSError as exc:
                t.error=repr(exc);self._stop_after_failure(t.error)
            finally:t.event.set();s.cv.notify_all()

    def _monitor(self):
        s=self.state
        while True:
            with s.cv:
                pids=list(s.pids.values());closed=s.closed;external=list(s.external.items())
            if closed:return
            for job,launch in external:
                if(job/'PROCESS_RETURN.json').exists()or not self.alive(launch,job):
                    with s.cv:
                        if job in s.external:
                            del s.external[job];s.pids.pop(job,None);s.active-=1
                            self.for_method(job.parent.name).log(event='RESUME_EXIT',job=job.name)
                            s.cv.notify_all()
            rss=sum(_tree_rss(p)for p in pids)
            with s.cv:s.peak_RSS_KiB=max(s.peak_RSS_KiB,rss)
            s.monitor_stop.wait(.1)

    def answer(self,proposal,allow_new=True):
        job=self.job_dir(proposal);s=self.state
        with s.cv:
            task=s.tasks.get(job)
            if task and task.status=='CANCELLED':task=None
            if task:task.demand=True;s.cv.notify_all()
        if task and not task.event.is_set():
            self.log(event='DEPENDENCY_WAIT_START',job=job.name)
            try:task.event.wait()  # retain parent-exit gate even if OUTCOME appeared first
            finally:self.log(event='DEPENDENCY_WAIT_END',job=job.name)
        if task and task.error:
            return dict(status='UNKNOWN',reason=task.error,source='EXECUTOR_FAILURE',calls=self.take_count(job),
                        MSE_lower=None,MSE_upper=None,job=str(job))
        if(job/'OUTCOME.json').exists():answer=self.ingest(job,proposal,calls=self.take_count(job))
        else:
            prior = self.prior_job(proposal) if task is None and not (job/'LAUNCH.json').exists() else None
            if prior is not None:
                # The base handles completed, live and unresolved prior-owner
                # jobs read-only; do not ingest a LAUNCH without an OUTCOME.
                answer=super().answer(proposal,allow_new=False)
            elif task is None and(job/'LAUNCH.json').exists():
                answer=super().answer(proposal,allow_new=False)
            elif task is None and(job/'STARTING.json').exists():
                return dict(status='STALE_JOB',job=str(job))
            else:
                if not allow_new and task is None:return None
                task=self._enqueue(proposal,demand=True)
                if task is False:return dict(status='BUDGET_EXHAUSTED',job=None)
                if task is None:return dict(status='NO_ROUTE',job=None)
                self.log(event='DEPENDENCY_WAIT_START',job=job.name)
                try:task.event.wait()
                finally:self.log(event='DEPENDENCY_WAIT_END',job=job.name)
                if(job/'OUTCOME.json').exists():answer=self.ingest(job,proposal,calls=self.take_count(job))
                else:answer=dict(status='UNKNOWN',reason=task.error or'ADAPTER_EXIT_WITHOUT_OUTCOME',
                    source='NEW_Q3_JOB',calls=self.take_count(job),MSE_lower=None,MSE_upper=None,job=str(job))
        if answer is not None and answer['status']not in('IN_PROGRESS','STALE_JOB','NO_ROUTE'):
            with s.cv:
                first=job not in s.adopted_jobs
                if first:
                    s.adopted_jobs.add(job);s.metrics['adopted']+=1;s.by_method.setdefault(self.method,Counter())['adopted']+=1
                    if task and getattr(task,'speculative',False):s.metrics['speculative_adopted']+=1
            if first:self.log(event='ADOPT',job=job.name,configuration=proposal['configuration_id'],status=answer['status'])
        return answer

    def finish(self):
        s=self.state
        with s.cv:
            if s.closed and s.driver.closed:return
            for t in s.tasks.values():
                if t.status=='QUEUED'and(self.method is None or t.view.method==self.method):self._cancel(t)
            while (any(t.status=='RUNNING'and(self.method is None or t.view.method==self.method)for t in s.tasks.values())
                   or (self.wait_for_running and any(self.method is None or job.parent.name==self.method for job in s.external))):
                s.cv.wait(.1)
            if self.method is not None:return
            s.closed=True;s.cv.notify_all()
            s.monitor_stop.set()
        self.dispatcher.join();self.monitor.join()
        for t in self.waiters:t.join()
        try:self.log(event='EXECUTOR_STOP')
        finally:fcntl.flock(s.driver,fcntl.LOCK_UN);s.driver.close()

    def metrics(self):
        s=self.state
        with s.cv:
            launched={str(t.job)for t in s.tasks.values()if t.started}
            adopted=set();events=[]
            for line in (self.run_dir/'PROGRESS.jsonl').read_text().splitlines()if(self.run_dir/'PROGRESS.jsonl').exists()else []:
                e=json.loads(line)
                if e.get('invocation')!=self.invocation:continue
                events.append(e)
                if e['event']=='ADOPT':adopted.add(e['job'])
            active=set();queued=set();waiting=Counter();busy=idle_ready=idle_empty=idle_dependency=0.;last=events[0]['monotonic']
            for e in events:
                dt=max(0,e['monotonic']-last);last=e['monotonic'];busy+=len(active)*dt
                idle=max(0,self.capacity-len(active))*dt
                if queued:idle_ready+=idle
                elif sum(waiting.values()):idle_dependency+=idle
                else:idle_empty+=idle
                event=e['event'];job=e.get('job')
                if event=='DEPENDENCY_WAIT_START':waiting[job]+=1
                elif event=='DEPENDENCY_WAIT_END':waiting[job]-=1
                elif event=='ENQUEUE':queued.add(job)
                elif event=='LAUNCH':queued.discard(job);active.add(job)
                elif event=='CANCEL_QUEUED':queued.discard(job)
                elif event in('EXIT','RESUME_EXIT'):active.discard(job)
                elif event=='RESUME_RUNNING':active.add(job)
            elapsed=events[-1]['monotonic']-events[0]['monotonic']
            real=sum(t.rule['kind']!='artificial'and t.started for t in s.tasks.values())
            counts={k:s.metrics[k]for k in ('submitted','queued_cancelled','launched','finished','adopted','speculative_launched','speculative_adopted')}
            return dict(**counts,peak_active=s.peak_active,sampled_owned_tree_peak_RSS_KiB=s.peak_RSS_KiB,
                completed_not_adopted=sum(Path(j).name not in adopted for j in launched),
                by_method={m:dict(c)for m,c in s.by_method.items()},max_concurrent_jobs=self.capacity,
                dp_workers=self.dp_workers,new_real_route_launches=real,executor_failure=s.failure,elapsed_seconds=elapsed,
                busy_job_seconds=busy,idle_slot_seconds_no_queued_candidate=idle_empty+idle_dependency,
                idle_slot_seconds_waiting_selected_result=idle_dependency,
                idle_slot_seconds_no_queued_or_selected_result=idle_empty,
                idle_slot_seconds_with_queued_candidate=idle_ready,
                slot_utilization=busy/(elapsed*self.capacity)if elapsed else 0,
                speculative_adoption_fraction=s.metrics['speculative_adopted']/s.metrics['speculative_launched']if s.metrics['speculative_launched']else None,
                memory_scope='sampled sum of owned job process RSS, .1s interval; shared pages may be double-counted')


def optimize(root,problem,*,executor,methods=None,close_executor=True,**kwargs):
    """Parallel methods; each retains O.optimize/run_method's own adoption order."""
    methods=list(methods or problem['methods'])
    try:
        if kwargs.get('context') is None:kwargs['context']=O.prepare(root,problem,methods=methods)
        with ThreadPoolExecutor(max_workers=max(1,len(methods)))as pool:
            futures={m:pool.submit(O.optimize,root,problem,executor=executor.for_method(m),methods=[m],**kwargs)for m in methods}
            out={m:futures[m].result()[m]for m in methods}
    finally:
        if close_executor:executor.finish()
        else:
            for method in methods:executor.for_method(method).finish()
    metrics=executor.metrics()
    for m,result in out.items():
        counts=metrics['by_method'].get(m,{})
        result['adopted_new_calls']=result['new_calls']
        result['new_calls']=counts.get('launched',0)
        result['budget']['new_calls']=result['new_calls']
        result['executor_metrics']=dict(metrics,method_launched=counts.get('launched',0),method_adopted=counts.get('adopted',0))
    return out
