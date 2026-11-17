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
import subprocess
import sys
import signal

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
        if task and task.error == 'HOST_ELIGIBILITY_HOLD':
            return self._held_answer(proposal)
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
                if task.error == 'HOST_ELIGIBILITY_HOLD':
                    return self._held_answer(proposal)
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


class DistributedExecutor(ConcurrentExecutor):
    """The existing queue/adoption order, with persistent host-owned attempts.

    Every host, including this controller's local slots, uses the same adapter
    worker. Assignment never expires merely because transport is unavailable.
    """
    def __init__(self, *args, distributed, **kwargs):
        from .rpe_q3_store import JobStore
        from . import rpe_q3_coordinator as Q
        self.distributed = deepcopy(distributed)
        self.eligibility_holds = {}
        self.eligibility_policy_cache = {}
        run = Path(args[1]).resolve()
        self.store = JobStore(run/'distributed'/'QUEUE.sqlite3')
        self.hosts = {}
        self.input_hash_cache = {}
        self.local_worker = None
        self.local_worker_pid = None
        self.rpc_server = None
        super().__init__(*args, **kwargs)
        try:
            self.controller_source = self.optimizer_source
            self.computation_root = Path(distributed.get('computation_root', self.root)).resolve()
            self.computation_source = distributed.get('computation_source', self.optimizer_source)
            self.optimizer_source = self.computation_source
            self.worker_script = Path(distributed.get('worker_script', self.root/'scripts/run_rpe_q3_worker.py')).resolve()
            self.source_manifest = Q.source_manifest(self.computation_root, self.computation_source)
            contract_path = run/'distributed'/'SOURCE.json'
            fixed = dict(source=self.source_manifest, evaluation_contract=self.contract)
            if contract_path.exists() and json.loads(contract_path.read_text()) != fixed:
                raise ValueError('Distributed run SOURCE/evaluation contract changed; use a new run')
            if not contract_path.exists(): Q.atomic_json(contract_path, fixed)
            with self.state.cv:
                for row in self.store.rows():
                    if row['state'] == 'QUEUED':
                        decision = self._eligibility_for(row['request'], row['job_id'])
                        if not self.store.set_eligibility(row['job_id'], decision):
                            self.eligibility_holds[row['job_id']] = decision
                            continue
                        row = self.store.row(row['job_id'])
                    if row['state'] not in ('CANCELLED', 'ACCEPTED'):
                        self._restore_task(row)
                    elif row['state'] == 'ACCEPTED':
                        self._publish_transfer(row)
            self.rpc_server, self.rpc_thread = Q.serve(self, distributed.get('bind', '127.0.0.1'), distributed.get('port', 8763))
            address = self.rpc_server.server_address
            self.controller_url = 'http://127.0.0.1:'+str(address[1])
            Q.atomic_json(run/'distributed'/'ENDPOINT.json', dict(url=self.controller_url, pid=os.getpid(), source=self.optimizer_source))
            profile = distributed.get('local_worker')
            if profile and profile.get('slots', 0) > 0:
                profile = dict(profile)
                profile.setdefault('root', str(self.computation_root))
                profile_path = run/'distributed'/'LOCAL_WORKER.json'
                Q.atomic_json(profile_path, profile)
                daemon_path = run/'distributed'/'LOCAL_DAEMON.json'
                previous = json.loads(daemon_path.read_text()) if daemon_path.exists() else {}
                if self._local_daemon_alive(previous.get('pid'), profile_path):
                    self.local_worker_pid = previous['pid']
                else:
                    with (run/'distributed'/'LOCAL_WORKER.log').open('a') as log:
                        self.local_worker = subprocess.Popen([sys.executable, '-B', str(self.worker_script),
                            '--profile', str(profile_path), '--controller', self.controller_url, '--poll-seconds', '.1'],
                            cwd=self.root, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
                    self.local_worker_pid = self.local_worker.pid
                    Q.atomic_json(daemon_path, dict(pid=self.local_worker_pid, profile=str(profile_path)))
            self.log(event='DISTRIBUTED_START', url=self.controller_url, controller_source=self.controller_source,
                computation_source=self.computation_source, computation_root=str(self.computation_root))
        except BaseException:
            with self.state.cv:
                self.state.closed = True
                self.state.cv.notify_all()
                self.state.monitor_stop.set()
            self.dispatcher.join(); self.monitor.join()
            if self.rpc_server: self.rpc_server.shutdown(); self.rpc_server.server_close()
            fcntl.flock(self.state.driver, fcntl.LOCK_UN); self.state.driver.close()
            raise

    def for_method(self, method):
        view = DistributedExecutor.__new__(DistributedExecutor)
        view.__dict__.update(self.__dict__)
        view.method = method
        return view

    def alive(self, launch, job):
        if launch.get('attempt_id'): return False  # a remote PID has no controller-local meaning
        return super().alive(launch, job)

    def _local_daemon_alive(self, pid, profile_path):
        if type(pid) is not int: return False
        try:
            command = Path('/proc/%s/cmdline' % pid).read_bytes().decode(errors='replace')
            return str(profile_path) in command and str(self.worker_script) in command
        except OSError: return False

    def _dispatch(self):
        # RPC claims supply host slots; the original dispatcher order is used
        # in _poll rather than launching controller-local adapters directly.
        with self.state.cv:
            while not self.state.closed: self.state.cv.wait(.2)

    def _eligibility_for(self, request, job_id):
        """Bind a prepared host decision to this exact scientific request."""
        from . import rpe_q3_coordinator as Q
        policy = self.distributed.get('host_eligibility')
        if policy is None and self.store.artificial(request):
            return None
        result = dict(request_sha256=Q.digest(request), source_manifest_sha256=Q.digest(self.source_manifest),
                      eligible_host_ids=[], host_fingerprints={}, reasons=[])
        try:
            if isinstance(policy, (str, Path)):
                path = Path(policy)
                if not path.is_absolute(): path = self.root/path
                st = path.stat();signature = (str(path.resolve()), st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
                if self.eligibility_policy_cache.get('signature') != signature:
                    self.eligibility_policy_cache.update(signature=signature, policy=json.loads(path.read_text()))
                policy = self.eligibility_policy_cache['policy']
            if not isinstance(policy, dict) or policy.get('schema') != 'RPE_Q3_HOST_ELIGIBILITY_V1':
                raise ValueError('MISSING_HOST_ELIGIBILITY_POLICY')
            record = (policy.get('jobs') or {}).get(job_id)
            if not isinstance(record, dict): raise ValueError('NO_REQUEST_SPECIFIC_HOST_EVIDENCE')
            if record.get('decision_sha256') != Q.digest({k:v for k,v in record.items() if k != 'decision_sha256'}):
                raise ValueError('HOST_ELIGIBILITY_DECISION_CONTENT_CHANGED')
            current_source = Q.source_manifest(self.computation_root, self.computation_source)
            head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=self.computation_root, text=True).strip()
            if current_source != self.source_manifest or head != self.computation_source:
                raise ValueError('COMPUTATION_SOURCE_CHANGED')
            for key, expected in (('request_sha256', result['request_sha256']),
                                  ('source_manifest_sha256', result['source_manifest_sha256']),
                                  ('method', request['method'])):
                if record.get(key) != expected: raise ValueError('HOST_ELIGIBILITY_CHANGED_' + key.upper())
            if record.get('source_manifest') != self.source_manifest:
                raise ValueError('HOST_ELIGIBILITY_CHANGED_SOURCE_MANIFEST')
            by_input = lambda rows: sorted(rows, key=lambda r:(r['commit'], r['path'], r['sha256']))
            actual_inputs = Q.input_hashes(self.computation_root, request, self.input_hash_cache)
            if by_input(record.get('input_hashes', [])) != by_input(actual_inputs):
                raise ValueError('HOST_ELIGIBILITY_CHANGED_INPUT_HASHES')
            result['decision_sha256'] = record.get('decision_sha256')
            if not result['decision_sha256']: raise ValueError('NO_HOST_ELIGIBILITY_DECISION_DIGEST')
            result['host_decisions'] = {name: dict(status=cell.get('status', 'UNCHECKED'),
                reasons=cell.get('reasons', [])) for name, cell in (record.get('hosts') or {}).items()}
            result['execution_profiles'] = {}
            for host in record.get('eligible_host_ids', []):
                cell = (record.get('hosts') or {}).get(host, {})
                fingerprint = cell.get('current_host_fingerprint_sha256')
                if cell.get('status') == 'PASS' and isinstance(fingerprint, str) and fingerprint:
                    current = (self.hosts.get(host, {}).get('host_fingerprints') or {}).get(job_id)
                    if isinstance(current, dict) and Q.digest(current) != fingerprint:
                        result['reasons'].append('CURRENT_HOST_FINGERPRINT_CHANGED:' + host)
                        result['host_decisions'][host] = dict(status='UNCHECKED',
                            reasons=['CURRENT_HOST_FINGERPRINT_CHANGED'])
                        continue
                    result['eligible_host_ids'].append(host)
                    result['host_fingerprints'][host] = fingerprint
                    if cell.get('execution_profile') is not None:
                        result['execution_profiles'][host] = deepcopy(cell['execution_profile'])
            if not result['eligible_host_ids']: result['reasons'].append('ALL_HOSTS_FAIL_OR_UNCHECKED')
        except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
            result['reasons'].append(str(exc))
        return result

    def _held_answer(self, proposal):
        job = self.job_dir(proposal)
        return dict(status='HOST_ELIGIBILITY_HOLD', reason='NO_MATCHING_ELIGIBLE_HOST',
                    host_eligibility=self.eligibility_holds.get(job.name), source='HOST_READINESS',
                    calls=0, MSE_lower=None, MSE_upper=None, job=None)

    def _restore_task(self, row, proposal=None, demand=False):
        request = row['request']; method = row['method']
        job = self.run_dir/'jobs'/method/row['job_id']
        old = self.state.tasks.get(job)
        if old and old.status != 'CANCELLED':
            old.demand |= demand
            return old
        job.mkdir(parents=True, exist_ok=True)
        if not (job/'REQUEST.json').exists(): _write(job/'REQUEST.json', request)
        if proposal is None:
            proposal = dict(configuration_id=request['configuration_id'], vector=O.interleave(request['N_X'], request['N_Y']),
                            cost=O.frac(request['cost']), origin=request['origin'])
        running = row['state'] != 'QUEUED'
        task = SimpleNamespace(view=self.for_method(method), job=job, proposal=deepcopy(proposal), rule=request['route'],
            owner=(method, request['configuration_id']), demand=demand, status='RUNNING' if running else 'QUEUED',
            event=threading.Event(), enqueue=time.monotonic()-max(0, time.time()-row['enqueued_epoch']),
            error=None, started=running, speculative=not demand)
        self.state.tasks[job] = task
        if running:
            self.state.active += 1
            task.view.log(event='RESUME_RUNNING', job=job.name, host_id=row['host_id'], attempt_id=row['attempt_id'])
        else:
            self.state.reserved[method] += 1
        if row['state'] == 'UNKNOWN':
            task.error = 'WORKER_STATE_UNKNOWN_RECONCILIATION_REQUIRED'
            task.status = 'DONE'; task.event.set(); self.state.active -= 1
        self.state.peak_active = max(self.state.peak_active, self.state.active)
        return task

    def _enqueue(self, proposal, demand=False):
        from .rpe_q3_coordinator import prefix_key
        job = self.job_dir(proposal); s = self.state
        with s.cv:
            if s.failure: raise RuntimeError('Executor persistence failure: '+s.failure)
            task = s.tasks.get(job)
            if task and task.status != 'CANCELLED':
                task.demand |= demand
                return task
            row = self.store.row(job.name)
            if row and row['state'] == 'ACCEPTED': return None
            if row and row['state'] != 'CANCELLED':
                if row['state'] == 'QUEUED':
                    decision = self._eligibility_for(row['request'], row['job_id'])
                    if not self.store.set_eligibility(row['job_id'], decision):
                        self.eligibility_holds[job.name] = decision
                        return False
                    row = self.store.row(job.name)
                return self._restore_task(row, proposal, demand)
            if (job/'OUTCOME.json').exists() or (job/'LAUNCH.json').exists() or self.prior_job(proposal) is not None: return None
            rule = self.route(proposal['configuration_id'])
            if rule is None: return None
            if s.closed: raise RuntimeError('Executor is closed')
            request = self.request(proposal, job, rule)
            if row is not None: request = row['request']  # same cancelled scientific job; origin may have changed
            eligibility = self._eligibility_for(request, job.name)
            if not self.store.eligible(request, eligibility):
                self.eligibility_holds[job.name] = eligibility
                self.log(event='HOST_ELIGIBILITY_HOLD', job=job.name, eligibility=eligibility)
                return False
            self.eligibility_holds.pop(job.name, None)
            if not self.store.reserve(self.method, job.name, request, s.budgets.get(self.method), s.approved,
                                      contract=dict(evaluation=self.contract, prefix_key=prefix_key(request)),
                                      eligibility=eligibility): return False
            task = self._restore_task(self.store.row(job.name), proposal, demand)
            s.metrics['submitted'] += 1
            self.log(event='ENQUEUE', job=job.name, configuration=proposal['configuration_id'], demand=demand,
                     origin=proposal.get('origin', {}).get('kind'))
            s.cv.notify_all()
            return task

    def _cancel(self, task):
        if self.store.cancel(task.view.method, task.job.name): super()._cancel(task)

    def launched(self, method):
        return self.store.counts(method, started_only=True)

    def has_job(self, proposal):
        row = self.store.row(self.job_dir(proposal).name)
        return (row is not None and row['state'] != 'CANCELLED') or super().has_job(proposal)

    def _assignment(self, row, host_id=None):
        from .rpe_q3_coordinator import input_hashes, prefix_key, digest
        from .rpe_host_profiles import profile_core
        value = dict(job_id=row['job_id'], method=row['method'], attempt_id=row['attempt_id'], request=row['request'],
            request_sha256=row['request_sha256'], source_manifest=self.source_manifest,
            input_hashes=input_hashes(self.computation_root, row['request'], self.input_hash_cache), prefix_key=prefix_key(row['request']))
        # Eligibility can change only before claim. Assigned rows retain the
        # selected profile for replay/recovery even after policy replacement.
        fixed = self.store.row(row['job_id'])
        host = host_id or row.get('host_id')
        profile = ((fixed.get('eligibility') or {}).get('execution_profiles') or {}).get(host)
        if profile is not None:
            value['host_execution_profile'] = profile_core(profile)
            value['host_execution_profile_sha256'] = digest(value['host_execution_profile'])
        return value

    def rpc(self, path, value):
        from . import rpe_q3_coordinator as Q
        s = self.state
        with s.cv:
            host = value['host_id']
            if not isinstance(host, str) or not host: raise ValueError('Nonempty host_id required')
            if path == '/v1/poll':
                if value.get('source_sha') != self.optimizer_source: raise ValueError('Worker fixed SOURCE differs')
                slots = value['slots']
                if type(slots) is not int or slots < 0: raise ValueError('Nonnegative free slots required')
                fingerprints = value.get('host_fingerprints') or {}
                if not isinstance(fingerprints, dict): raise ValueError('Host fingerprints must map job IDs to current facts')
                self.hosts[host] = dict(epoch=time.time(), slots=slots, cached_prefixes=value.get('cached_prefixes', []),
                                       host_fingerprints=fingerprints)
                # A lost response is reconciled with its original owner even
                # if current readiness changed or another host is offline.
                assignments = [self._assignment(row) for row in self.store.rows()
                    if row['host_id'] == host and row['state'] == 'ASSIGNED'][:slots]
                required = self.distributed.get('required_hosts', [])
                if any(name not in self.hosts for name in required):
                    return dict(assignments=assignments, stop=False,
                                waiting_for_hosts=[name for name in required if name not in self.hosts])
                queue = sorted((t for t in s.tasks.values() if t.status == 'QUEUED'),
                    key=lambda t: (not t.demand, s.owner_clock.get(t.owner, -1), t.enqueue))
                if s.failure or s.closed: return dict(assignments=assignments, stop=False)
                eligible, challenges = [], []
                skipped = False
                for task in queue:
                    row = self.store.row(task.job.name)
                    decision = self._eligibility_for(row['request'], row['job_id'])
                    if not self.store.set_eligibility(row['job_id'], decision):
                        self.eligibility_holds[row['job_id']] = decision
                        task.error = 'HOST_ELIGIBILITY_HOLD'
                        ConcurrentExecutor._cancel(self, task)
                        skipped = True
                        continue
                    self.eligibility_holds.pop(row['job_id'], None)
                    if decision is not None and host not in decision['eligible_host_ids']:
                        skipped = True
                        continue
                    if not self.store.eligible(row['request'], decision, host, fingerprints, row['job_id']):
                        # Metadata challenge only: no attempt ID, reservation
                        # transition, LAUNCH or scientific execution occurs.
                        challenge = self._assignment(row, host_id=host)
                        challenge.pop('attempt_id', None)
                        challenges.append(challenge)
                        skipped = True
                        continue
                    conditional = task.proposal.get('origin', {}).get('kind') == 'CHAIN_LOOKAHEAD_IF_PREVIOUS_PASSES'
                    if skipped and conditional and not task.demand:
                        continue
                    eligible.append(task)
                # Prefix locality is considered only among eligible hosts for
                # the same first dispatchable task, never an unsupported host.
                if eligible and time.monotonic()-eligible[0].enqueue < self.distributed.get('affinity_wait_seconds', .2):
                    first = self.store.row(eligible[0].job.name)
                    key = Q.prefix_key(first['request'])
                    if key not in value.get('cached_prefixes', []):
                        preferred = [h for h, info in self.hosts.items() if h != host and info['slots'] > 0
                            and time.time()-info['epoch'] < 2 and key in info['cached_prefixes']
                            and self.store.eligible(first['request'], first['eligibility'], h,
                                                   info.get('host_fingerprints'), first['job_id'])]
                        if preferred: return dict(assignments=assignments, stop=False, eligibility_challenges=challenges)
                remaining = max(0, min(slots-len(assignments), self.capacity-s.active))
                for task in eligible:
                    if remaining <= 0: break
                    rows = self.store.claim(host, 1, value.get('cached_prefixes', []), [task.job.name],
                                            host_fingerprints=fingerprints)
                    if not rows: continue
                    row = rows[0]
                    # Assignment and reservation are committed before a worker
                    # receives anything. A lost HTTP response retains ownership.
                    Q.atomic_json(task.job/'LAUNCH.json', dict(host_id=host, attempt_id=row['attempt_id'],
                        pid=None, started=row['assigned_epoch'], kind=task.rule['kind'], transport='distributed'))
                    task.status = 'RUNNING'; task.started = True; task.speculative = not task.demand
                    s.clock += 1; s.owner_clock[task.owner] = s.clock; s.active += 1
                    s.peak_active = max(s.peak_active, s.active)
                    self.uncounted.add(task.job)
                    s.metrics['launched'] += 1
                    s.by_method.setdefault(row['method'], Counter())['launched'] += 1
                    if task.speculative: s.metrics['speculative_launched'] += 1
                    task.view.log(event='LAUNCH', job=row['job_id'], host_id=host, attempt_id=row['attempt_id'],
                        configuration=task.proposal['configuration_id'], queue_wait_seconds=time.monotonic()-task.enqueue,
                        active=s.active, demand=task.demand)
                    assignments.append(self._assignment(row))
                    remaining -= 1
                return dict(assignments=assignments, stop=False, eligibility_challenges=challenges)
            row = self.store.row(value['job_id'])
            if row is None or row['host_id'] != host or row['attempt_id'] != value['attempt_id']:
                raise ValueError('Different execution host/attempt')
            job = self.run_dir/'jobs'/row['method']/row['job_id']
            task = s.tasks.get(job)
            if path == '/v1/transfer-metric':
                self.for_method(row['method']).log(event='TRANSFER_TIMING', job=row['job_id'], host_id=host,
                    **value.get('timing', {}))
                return dict(ok=True)
            if path == '/v1/state':
                if row['state'] != 'ACCEPTED':
                    self.store.update(host, row['job_id'], value['attempt_id'], value['state'], value.get('detail'))
                    if value['state'] == 'UNKNOWN' and task and task.status == 'RUNNING':
                        task.error = 'WORKER_UNKNOWN:'+str(value.get('detail'))
                        self._finished(task, None, row['assigned_epoch'])
                return dict(ok=True)
            if path != '/v1/complete': raise ValueError('Unknown Q3 RPC endpoint')
            started = time.monotonic()
            if value['request_sha256'] != row['request_sha256']: raise ValueError('Completion request hash differs')
            transfer = self.run_dir/'distributed'/'transfers'/row['job_id']/row['attempt_id']
            transfer.mkdir(parents=True, exist_ok=True)
            Q.install_files(transfer, value['files'])
            if not (transfer/'REQUEST.json').exists(): _write(transfer/'REQUEST.json', row['request'])
            receipt = json.loads((transfer/'PROCESS_RETURN.json').read_text())
            outcome = json.loads((transfer/'OUTCOME.json').read_text())
            if outcome['status'] in (O.PASS, O.FAIL, 'INCONCLUSIVE') and (receipt.get('returncode') != 0 or receipt.get('signal') is not None):
                raise ValueError('Successful outcome without a clean adapter exit')
            proposal = task.proposal if task else dict(configuration_id=row['request']['configuration_id'],
                vector=O.interleave(row['request']['N_X'], row['request']['N_Y']))
            validation = self.for_method(row['method']).ingest(transfer, proposal, calls=0)
            if outcome['status'] in (O.PASS, O.FAIL, 'INCONCLUSIVE') and validation['status'] == 'UNKNOWN':
                raise ValueError('Transferred outcome identity mismatch: '+str(validation.get('reason')))
            accepted, duplicate = self.store.complete(host, row['job_id'], row['attempt_id'], value['request_sha256'], value['files'])
            self._publish_transfer(accepted)
            if not duplicate:
                self.for_method(row['method']).log(event='TRANSFER_COMPLETE', job=row['job_id'], host_id=host,
                    transfer_seconds=time.monotonic()-started, transfer_bytes=sum(len(x)*3//4-x.count('=') for x in value['files'].values()),
                    worker_transfer_seconds=value.get('transfer_seconds'))
            if task:
                task.error = None
                if task.status == 'RUNNING': self._finished(task, receipt.get('returncode'), row['assigned_epoch'])
                else: task.event.set()
            s.cv.notify_all()
            return dict(ok=True, duplicate=duplicate)

    def _publish_transfer(self, row):
        from . import rpe_q3_coordinator as Q
        import base64
        job = self.run_dir/'jobs'/row['method']/row['job_id']
        stage = self.run_dir/'distributed'/'transfers'/row['job_id']/row['attempt_id']
        if not stage.exists():
            if (job/'OUTCOME.json').exists(): return
            raise ValueError('Accepted completion is missing its transfer directory')
        job.mkdir(parents=True, exist_ok=True)
        files = {str(p.relative_to(stage)): base64.b64encode(p.read_bytes()).decode() for p in stage.rglob('*')
                 if p.is_file() and str(p.relative_to(stage)) not in ('REQUEST.json', 'OUTCOME.json')}
        Q.install_files(job, files)
        outcome = stage/'OUTCOME.json'
        if outcome.exists(): Q.install_files(job, {'OUTCOME.json': base64.b64encode(outcome.read_bytes()).decode()})

    def answer(self, proposal, allow_new=True):
        job = self.job_dir(proposal)
        row = self.store.row(job.name)
        # Readiness is an operational hold, not a failed scientific evaluation
        # or an exhausted budget. Already owned/complete jobs retain recovery.
        if (allow_new and (row is None or row['state'] in ('CANCELLED', 'QUEUED')) and
                not (job/'OUTCOME.json').exists() and not (job/'LAUNCH.json').exists() and self.prior_job(proposal) is None):
            rule = self.route(proposal['configuration_id'])
            if rule is not None:
                request = row['request'] if row is not None else self.request(proposal, job, rule)
                decision = self._eligibility_for(request, job.name)
                if not self.store.eligible(request, decision):
                    with self.state.cv:
                        self.eligibility_holds[job.name] = decision
                        if row is not None: self.store.set_eligibility(job.name, decision)
                        task = self.state.tasks.get(job)
                        if task and task.status == 'QUEUED':
                            task.error = 'HOST_ELIGIBILITY_HOLD'
                            ConcurrentExecutor._cancel(self, task)
                    return self._held_answer(proposal)
                self.eligibility_holds.pop(job.name, None)
        if row and row['state'] not in ('ACCEPTED', 'CANCELLED'):
            with self.state.cv: self._restore_task(row, proposal, demand=True)
        answer = super().answer(proposal, allow_new)
        if answer and answer.get('status') == 'BUDGET_EXHAUSTED' and job.name in self.eligibility_holds:
            return self._held_answer(proposal)
        if answer and row is not None and self.store.row(row['job_id'])['state'] == 'ACCEPTED':
            self.store.adopt(row['job_id'])
        elif answer:
            current = self.store.row(self.job_dir(proposal).name)
            if current and current['state'] == 'ACCEPTED': self.store.adopt(current['job_id'])
        return answer

    def ingest(self, job, proposal, calls):
        answer = super().ingest(job, proposal, calls)
        current = self.store.row(Path(job).name)
        own = current is not None and Path(job) == self.run_dir/'jobs'/current['method']/current['job_id']
        if calls == 0 and own and current['state'] == 'ACCEPTED':
            with self.state.cv:
                reused = getattr(self.state, 'reused_jobs', set())
                if current['job_id'] not in reused:
                    reused.add(current['job_id']); self.state.reused_jobs = reused
                    self.log(event='RESULT_REUSE', job=current['job_id'], host_id=current['host_id'])
        return answer

    def metrics(self):
        result = super().metrics()
        rows = self.store.rows(); by_host = {}
        for row in rows:
            if row['host_id'] is None: continue
            counts = by_host.setdefault(row['host_id'], dict(assigned=0, launched=0, completed=0, adopted=0, reused=0))
            counts['assigned'] += 1
            counts['launched'] += row['started_epoch'] is not None
            counts['completed'] += row['state'] == 'ACCEPTED'
            counts['adopted'] += row['adopted_epoch'] is not None
        for line in (self.run_dir/'PROGRESS.jsonl').read_text().splitlines():
            event = json.loads(line)
            if event.get('event') == 'RESULT_REUSE' and event.get('host_id') in by_host:
                by_host[event['host_id']]['reused'] += 1
        result.update(distributed=True, by_host=by_host, persistent_reserved={m:self.store.counts(m) for m in self.routes},
                      host_eligibility_holds=deepcopy(self.eligibility_holds),
                      actual_OS_host_count='recorded by integration harness; host IDs alone do not establish separate machines')
        return result

    def finish(self):
        if self.method is not None: return super().finish()
        if self.state.driver.closed: return
        super().finish()
        if self.local_worker is not None:
            self.local_worker.terminate()
            self.local_worker.wait(timeout=10)
        elif self._local_daemon_alive(self.local_worker_pid, self.run_dir/'distributed'/'LOCAL_WORKER.json'):
            os.kill(self.local_worker_pid, signal.SIGTERM)
        if self.rpc_server is not None:
            self.rpc_server.shutdown(); self.rpc_server.server_close(); self.rpc_thread.join()


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
