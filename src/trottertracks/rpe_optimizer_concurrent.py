"""Work-conserving execution around the unchanged per-configuration search rule.

The search controls adoption, including eligible out-of-order scaled results.
Finite budgets include reservations before
launch; running scientific workers are never cancelled to discard lookahead.
"""
from collections import Counter, defaultdict, deque
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
from .rpe_q3_timing import stamp

LIVE_READINESS_RETRY_SECONDS = 10
_CHECK_PENDING = object()


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
    def __init__(self,*args,budgets=None,science_approved=True,dp_workers=None,prefix_cache_directory=None,
                 prefix_checkpoint_rounds=None,prefix_cache_max_bytes=None,prefix_selective_checkpoint_rounds=(),
                 prefix_selective_window_seconds=None,compact_q3_evidence=False,parallel_dp_residual=False,
                 profile_dp_residual=False,**kwargs):
        super().__init__(*args,**kwargs)
        if dp_workers is not None and (type(dp_workers)is not int or dp_workers<1):
            raise ValueError('Positive integer dp_workers required')
        if prefix_cache_max_bytes is not None and (type(prefix_cache_max_bytes) is not int or prefix_cache_max_bytes < 1):
            raise ValueError('Positive integer prefix_cache_max_bytes required')
        if type(compact_q3_evidence) is not bool:
            raise ValueError('compact_q3_evidence must be a boolean')
        if type(parallel_dp_residual) is not bool:
            raise ValueError('parallel_dp_residual must be a boolean')
        if type(profile_dp_residual) is not bool or (profile_dp_residual and not parallel_dp_residual):
            raise ValueError('profile_dp_residual requires parallel_dp_residual')
        if not isinstance(prefix_selective_checkpoint_rounds,(list,tuple)) or any(
                type(m) is not int or m not in (prefix_checkpoint_rounds or ())
                for m in prefix_selective_checkpoint_rounds):
            raise ValueError('Selective prefix rounds must occur in prefix_checkpoint_rounds')
        if prefix_selective_window_seconds is not None and (
                type(prefix_selective_window_seconds) is not int or prefix_selective_window_seconds < 1):
            raise ValueError('Positive integer prefix_selective_window_seconds required')
        self.run_dir.mkdir(parents=True,exist_ok=True)
        driver=(self.run_dir/'FAST_DRIVER.lock').open('a')
        try:fcntl.flock(driver,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            driver.close();raise ValueError('A fast driver already owns this run directory')
        self.capacity=max(1,kwargs.get('max_concurrent',1));self.dp_workers=dp_workers
        self.prefix_cache_directory=None if prefix_cache_directory is None else str(Path(prefix_cache_directory).resolve())
        self.prefix_checkpoint_rounds=prefix_checkpoint_rounds
        self.prefix_cache_max_bytes=prefix_cache_max_bytes
        self.prefix_selective_checkpoint_rounds=list(prefix_selective_checkpoint_rounds)
        self.prefix_selective_window_seconds=prefix_selective_window_seconds
        self.compact_q3_evidence=compact_q3_evidence
        self.parallel_dp_residual=parallel_dp_residual
        self.profile_dp_residual=profile_dp_residual
        existing={}
        for method in self.routes:
            base=self.run_dir/'jobs'/method
            existing[method]=sum((d/'LAUNCH.json').exists()or(d/'STARTING.json').exists()for d in base.iterdir())if base.exists()else 0
        self.state=SimpleNamespace(cv=threading.Condition(threading.RLock()),tasks={},active=0,closed=False,
            failure=None,owner_clock={},clock=0,budgets=budgets or {},approved=science_approved,existing=existing,
            reserved=Counter(),metrics=Counter(),by_method={},pids={},driver=driver,peak_RSS_KiB=0,peak_active=0,adopted_jobs=set(),observations=defaultdict(list),external={},monitor_stop=threading.Event())
        self.state.scaled_completions = defaultdict(dict)
        self.state.scaled_recovered = set()
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
        # The normal factory decorates the owning executor's finish() to close
        # its optional price-validation service. A method view must use its
        # class finish(), which drains only that method and keeps the shared
        # queue alive for preparation continuations and subsequent batches.
        view.__dict__.pop('finish',None)
        return view

    def log(self,**event):
        job = event.get('job')
        if job:
            event['job'] = Path(str(job)).name
            marker = self.run_dir/'jobs'/str(self.method)/event['job']/'LAUNCH.json'
            try:
                owner = json.loads(marker.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                # An independent local launcher may still be writing the
                # receipt. Event annotations must not stop the evaluator.
                owner = {}
            for key in ('host_id', 'attempt_id'):
                if owner.get(key) is not None:event.setdefault(key, owner[key])
        with self.lock:
            with(self.run_dir/'PROGRESS.jsonl').open('a')as f:
                f.write(json.dumps(dict(UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                    **stamp(),invocation=self.invocation,method=self.method,**event),default=str)+'\n')

    def request(self,proposal,job,rule):
        from .rpe_host_profiles import PREFIX_PR_Q3_ROUTES, REVIEWED_PR_Q3_ROUTES
        projection = rule.get('horizon_projection')
        configurations = rule.get('configurations')
        if configurations is not None:
            request_id = proposal['configuration_id']
            source_id = projection['source_configuration_id'] if projection else request_id
            source = configurations.get(source_id)
            if source is None and rule.get('templates') and rule['kind'] in (
                    'pf_b_pf6', 'pauli_b_ld3', 'pauli_b_ld5', 'prdf_d2_split'):
                # A partial registry must not mask a case selected from the
                # saved templates. The fixed input hasher indexes this field
                # by request ID whenever it is present.
                rule = {key: value for key, value in rule.items() if key != 'configurations'}
            elif projection and source is not None:
                # The hasher uses the request ID. The worker uses the original
                # ID and validates the requested prefix against that source.
                rule = dict(rule, configurations={**configurations, request_id: source})
        req=super().request(proposal,job,rule)
        execution = req['execution_profile'] = dict(dp_workers=self.dp_workers)
        artificial = rule['kind'] == 'artificial'
        route = req['method'], rule['kind']
        if self.prefix_cache_directory is not None:
            execution.update(prefix_cache_directory=self.prefix_cache_directory,
                prefix_checkpoint_rounds=self.prefix_checkpoint_rounds)
        if artificial or route in REVIEWED_PR_Q3_ROUTES or (
                route in PREFIX_PR_Q3_ROUTES and self.prefix_selective_checkpoint_rounds):
            # Distributed hosts resolve their own prefix cache directory.
            if self.prefix_cache_max_bytes is not None:
                execution['prefix_cache_max_bytes'] = self.prefix_cache_max_bytes
            if self.prefix_checkpoint_rounds is not None:
                execution['prefix_checkpoint_rounds'] = self.prefix_checkpoint_rounds
            if self.prefix_selective_checkpoint_rounds:
                execution['prefix_selective_checkpoint_rounds'] = self.prefix_selective_checkpoint_rounds
                if self.prefix_selective_window_seconds is not None:
                    execution['prefix_selective_window_seconds'] = self.prefix_selective_window_seconds
            if self.parallel_dp_residual:
                execution['parallel_dp_residual'] = True
            if self.profile_dp_residual:
                execution['profile_dp_residual'] = True
        if self.compact_q3_evidence and (artificial or route in REVIEWED_PR_Q3_ROUTES):
            execution['compact_q3_evidence'] = True
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
            if self._user_stop_requested():return False
            if s.failure:raise RuntimeError('Executor persistence/start failure; no new launches: '+s.failure)
            if proposal['configuration_id']in self.execution_holds.get(self.method,{}):return None
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

    def retire_priced_out(self, configurations):
        """Retire only work still owned by this method's local queue."""
        with self.state.cv:
            self.standby = deque(p for p in getattr(self,'standby', ())
                                 if p['configuration_id'] not in configurations)
            for task in self.state.tasks.values():
                if task.status=='QUEUED' and task.owner[0]==self.method and task.owner[1] in configurations \
                        and task.rule['kind']!='preparation':
                    self._cancel(task)

    def prefetch_standby(self, choice, proposals):
        """Reserve the demand first and retain the rest as unreserved alternatives."""
        with self.state.cv:
            self.standby = deque(proposals)
            speculative = [t for t in self.state.tasks.values() if t.view.method == self.method
                           and t.status == 'QUEUED' and not t.demand]
            demand = self.job_dir(choice) if choice is not None else None
            if speculative:
                desired = {self.job_dir(p) for p in self.standby}
                for t in speculative:
                    if t.job not in desired and t.job != demand:
                        self._cancel(t)
        if choice is not None and self._enqueue(choice, demand=True) is False:
            with self.state.cv:
                # Queued reservations can be reassigned to demand. Running jobs retain theirs.
                t = (next((t for t in reversed(speculative) if t.status == 'QUEUED'
                           and not t.demand and t.job != demand), None) if self._budget_reserved() else None)
                if t is not None:
                    self._cancel(t)
            if t is not None:
                self._enqueue(choice, demand=True)
        self._fill_standby()

    def _budget_reserved(self):
        cap = self.state.budgets.get(self.method)
        return type(cap) is int and self.state.existing.get(self.method, 0)+self.state.reserved[self.method] >= cap

    def _standby_capacity(self):
        return self.capacity

    def _standby_headroom(self):
        return 1

    def _standby_can_fill(self):
        s = self.state
        return (getattr(self, 'standby', None) and not s.failure and not s.closed and
                s.active + sum(t.status == 'QUEUED' and t.rule['kind'] != 'preparation'
                               for t in s.tasks.values()) <
                self._standby_capacity() + self._standby_headroom())

    def _fill_standby(self):
        # Prepare the next jobs while the current ones run. Advance past
        # duplicates and holds so a completion never revisits the same work.
        before = self.state.metrics['submitted']
        while True:
            with self.state.cv:
                if not self._standby_can_fill():
                    break
                proposal = self.standby.popleft()
            if self._enqueue(proposal) is False:
                with self.state.cv:
                    if self._budget_reserved() or self._user_stop_requested():
                        self.standby.clear()
                        break
        added = self.state.metrics['submitted'] - before
        if added:
            self.log(event='Q3_STANDBY_REFILL', added=added, remaining=len(self.standby))

    def search_profile(self, configuration_id):
        route = self.route(configuration_id) or {}
        return (route.get('kind'), route.get('interpreter'), route.get('view_commit'),
                self.dp_workers, self.prefix_cache_directory)

    def search_profile_for_proposal(self, proposal):
        return self.search_profile(proposal['configuration_id'])

    def search_observations(self):
        with self.state.cv:
            return list(self.state.observations.get(self.method, ()))

    def _cancel(self,t):
        t.status='CANCELLED';self.state.reserved[t.view.method]-=1
        self.state.metrics['queued_cancelled']+=1;t.event.set()
        self.state.cv.notify_all()
        t.view.log(event='CANCEL_QUEUED',job=t.job.name)

    def hold_configuration(self,proposal,answer):
        super().hold_configuration(proposal,answer)
        with self.state.cv:
            for task in self.state.tasks.values():
                if task.status=='QUEUED'and task.owner==(self.method,proposal['configuration_id']):self._cancel(task)
            self.state.cv.notify_all()  # running workers finish and preserve their outcomes

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
                s.cv.notify_all()  # the dispatch may empty the queue with spare slots
            self._start(t)

    def _user_stop_requested(self):
        """Stop intake and release only unassigned work; preserve remote owners."""
        if not (self.run_dir/'STOP_REQUESTED.json').exists():return False
        with self.state.cv:
            for task in self.state.tasks.values():
                if task.status=='QUEUED':self._cancel(task)
            self.state.cv.notify_all()
        return True

    def _stop_after_failure(self,error):
        # Persistence/start failure stops new launches; existing children finish.
        notify=getattr(self.state,'failure_callback',None)
        if notify is not None:notify(error)
        with self.state.cv:
            if self.state.failure is None:self.state.failure=str(error)
            if isinstance(error,BaseException)and getattr(self.state,'primary_exception',None)is None:
                self.state.primary_exception=error
            for task in self.state.tasks.values():
                if task.status=='QUEUED':
                    task.error=str(error)
                    try:self._cancel(task)
                    except OSError:pass  # the queue event is already released
            self.state.cv.notify_all()

    def _start(self,t):
        s=self.state;v=t.view;job=t.job;start=time.time();proc=None
        t.started_monotonic = time.monotonic()
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
            t.error=repr(exc);self._stop_after_failure(exc)
            O.record_runtime_error(self.run_dir,exc,'Q3_START',job=job.name)
            if proc is None:self._finished(t,None,start);return
            # Popen succeeded: retain its slot and wait even if LAUNCH could not save.
        def wait():
            code=None
            try:
                code=proc.wait()
                _write(job/'PROCESS_RETURN.json',dict(returncode=code if code>=0 else None,signal=-code if code<0 else None,
                                                    wall_seconds=time.time()-start,automatic_retry=False))
            except Exception as exc:
                t.error=repr(exc);self._stop_after_failure(exc)
                O.record_runtime_error(self.run_dir,exc,'Q3_WAIT',job=job.name)
            finally:self._finished(t,code,start)
        thread=threading.Thread(target=wait,name='rpe-wait-'+job.name,daemon=False)
        with self.lock:self.waiters.append(thread)
        thread.start()

    def _finished(self,t,code,start):
        s=self.state
        with s.cv:
            t.status='DONE';s.active-=1;s.pids.pop(t.job,None);s.metrics['finished']+=1
            t.finished_at = time.monotonic()
            try:t.view.log(event='EXIT',job=t.job.name,returncode=code,error=t.error,wall_seconds=time.time()-start)
            except OSError as exc:
                t.error=repr(exc);self._stop_after_failure(exc)
                O.record_runtime_error(self.run_dir,exc,'Q3_EXIT_RECORD',job=t.job.name)
            finally:
                if code == 0 and not t.error:
                    self._note_scaled_completion(t.view.method, t.job, t.proposal['origin'])
                t.event.set()
                s.cv.notify_all()

    def _note_scaled_completion(self, method, job, origin):
        from .rpe_scaled_shots import RULE
        if origin.get('kind') == RULE and job not in self.state.adopted_jobs:
            self.state.scaled_completions[method][job] = None

    def scaled_completion_pending(self):
        return bool(self.state.scaled_completions[self.method])

    def read_scaled_completions(self, proposal_for):
        """Read only published completions for the selector's current allocations."""
        with self.state.cv:
            jobs = tuple(self.state.scaled_completions[self.method])
            self.state.scaled_completions[self.method].clear()
        for job in jobs:
            if not (job/'OUTCOME.json').is_file():
                continue
            request = O.load_json(job/'REQUEST.json')
            cid = request['configuration_id']
            key = cid, O.interleave(request['N_X'], request['N_Y'])
            proposal = proposal_for(key)
            if (proposal is None or self.job_dir(proposal) != job or not self.same_route(job, cid)
                    or request.get('cost') is None or O.frac(request['cost']) != proposal['cost']):
                continue
            receipt = O.load_json(job/'PROCESS_RETURN.json')
            if receipt.get('returncode') != 0 or receipt.get('signal') is not None:
                continue
            answer = O.Executor.ingest(self, job, proposal, calls=0)
            if answer['status'] not in (O.PASS, O.FAIL, 'INCONCLUSIVE'):
                continue
            lower, upper = answer['MSE_lower'], answer['MSE_upper']
            if lower is None or upper is None or not 0 <= lower <= upper:
                continue
            answer['calls'] = self.take_count(job)
            self._adopt_answer(job, proposal, answer)
            yield key, answer

    def _monitor(self):
        s=self.state
        check_owners = getattr(self, '_check_worker_contact', None)
        next_check = 0
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
            if check_owners is not None and time.monotonic() >= next_check:
                check_owners()
                next_check = time.monotonic() + 5
            s.monitor_stop.wait(.1)

    def has_spare_slot(self, *, allow_queued_speculation=False):
        with self.state.cv:
            return (not self.state.failure and self.state.active < self.capacity
                    and not any(t.status == 'QUEUED' and (t.demand or not allow_queued_speculation)
                                for t in self.state.tasks.values()))

    def _can_take_ready_candidates(self):
        return self.has_spare_slot()

    def _wait_for_task(self, task):
        """Wake for fresh work when a slot is available, or for completed results."""
        s = self.state
        ready_pending = getattr(self, 'ready_pending', None)
        completion_pending = getattr(self, 'completion_pending', None)
        self.log(event='DEPENDENCY_WAIT_START', job=task.job.name)
        try:
            if ready_pending is None and completion_pending is None and not getattr(self, 'standby', None):
                task.event.wait()
                return True

            def fresh_ready():
                return ready_pending is not None and ready_pending() and self._can_take_ready_candidates()

            def reselect():
                if completion_pending is not None and completion_pending():
                    return 'Q3_WAIT_YIELD_FOR_COMPLETION'
                if fresh_ready():
                    return 'Q3_WAIT_YIELD_FOR_READY'
                return None

            while True:
                with s.cv:
                    s.cv.wait_for(lambda: task.event.is_set() or reselect() or self._standby_can_fill())
                    if task.event.is_set():
                        return True
                    event = reselect()
                    if event:
                        break
                self._fill_standby()
            self.log(event=event, job=task.job.name)
            return False
        finally:
            self.log(event='DEPENDENCY_WAIT_END', job=task.job.name)

    def answer(self,proposal,allow_new=True):
        job=self.job_dir(proposal);s=self.state
        with s.cv:
            task=s.tasks.get(job)
            if task and task.status=='CANCELLED':task=None
            if task:task.demand=True;s.cv.notify_all()
        if task and not task.event.is_set() and not self._wait_for_task(task):
            return dict(status='RESELECT_READY', job=str(job), calls=0)
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
                if task is False:
                    # A user stop declines new work; it is not a numerical
                    # answer. Keep the priority search out of its MSE/call
                    # accounting and let the outer driver persist the stop.
                    if self._user_stop_requested():return None
                    return dict(status='BUDGET_EXHAUSTED',job=None)
                if task is None:return dict(status='NO_ROUTE',job=None)
                if not self._wait_for_task(task):
                    return dict(status='RESELECT_READY', job=str(job), calls=0)
                if task.error == 'HOST_ELIGIBILITY_HOLD':
                    return self._held_answer(proposal)
                if(job/'OUTCOME.json').exists():answer=self.ingest(job,proposal,calls=self.take_count(job))
                else:answer=dict(status='UNKNOWN',reason=task.error or'ADAPTER_EXIT_WITHOUT_OUTCOME',
                    source='NEW_Q3_JOB',calls=self.take_count(job),MSE_lower=None,MSE_upper=None,job=str(job))
        answer=self.scope_unknown(job,proposal,answer)
        if answer is not None and answer['status']not in('IN_PROGRESS','STALE_JOB','NO_ROUTE'):
            self._adopt_answer(job, proposal, answer)
        return answer

    def _adopt_answer(self, job, proposal, answer):
        s = self.state
        with s.cv:
            s.scaled_completions[self.method].pop(job, None)
            task = s.tasks.get(job)
            first=job not in s.adopted_jobs
            if first:
                s.adopted_jobs.add(job);s.metrics['adopted']+=1;s.by_method.setdefault(self.method,Counter())['adopted']+=1
                if task and getattr(task,'speculative',False):s.metrics['speculative_adopted']+=1
                if (getattr(self, 'standby', None) is not None and task and task.status == 'DONE'
                    and answer['status'] in (O.PASS, O.FAIL) and hasattr(task, 'started_monotonic')):
                    cid = proposal['configuration_id']
                    s.observations[self.method].append(dict(
                        id=job.name, kind='SHOT_Q3', method=self.method, configuration_id=cid,
                        profile=self.search_profile(cid), status=answer['status'],
                        finished_at=task.finished_at, seconds=task.finished_at-task.started_monotonic))
        if first:self.log(event='ADOPT',job=job.name,configuration=proposal['configuration_id'],status=answer['status'])

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
            active=set();queued=set();waiting=Counter();busy=idle_ready=idle_empty=idle_dependency=available=0.;last=events[0]['monotonic']
            interval_capacity=self.capacity
            for e in events:
                dt=max(0,e['monotonic']-last);last=e['monotonic'];busy+=len(active)*dt
                available+=interval_capacity*dt
                idle=max(0,interval_capacity-len(active))*dt
                if queued:idle_ready+=idle
                elif sum(waiting.values()):idle_dependency+=idle
                else:idle_empty+=idle
                event=e['event'];job=e.get('job')
                if event in ('EXECUTOR_START','AUTO_CAPACITY'):interval_capacity=e['capacity']
                elif event=='DEPENDENCY_WAIT_START':waiting[job]+=1
                elif event=='DEPENDENCY_WAIT_END':waiting[job]-=1
                elif event=='ENQUEUE':queued.add(job)
                elif event=='LAUNCH':queued.discard(job);active.add(job)
                elif event=='CANCEL_QUEUED':queued.discard(job)
                elif event in('EXIT','RESUME_EXIT'):active.discard(job)
                elif event=='RESUME_RUNNING':active.add(job)
            elapsed=events[-1]['monotonic']-events[0]['monotonic']
            real=sum(t.rule['kind'] not in ('artificial','preparation') and t.started for t in s.tasks.values())
            counts={k:s.metrics[k]for k in ('submitted','queued_cancelled','launched','finished','adopted','speculative_launched','speculative_adopted')}
            return dict(**counts,peak_active=s.peak_active,sampled_owned_tree_peak_RSS_KiB=s.peak_RSS_KiB,
                completed_not_adopted=sum(Path(j).name not in adopted for j in launched),
                by_method={m:dict(c)for m,c in s.by_method.items()},max_concurrent_jobs=self.capacity,
                dp_workers=self.dp_workers,new_real_route_launches=real,executor_failure=s.failure,elapsed_seconds=elapsed,
                busy_job_seconds=busy,idle_slot_seconds_no_queued_candidate=idle_empty+idle_dependency,
                idle_slot_seconds_waiting_selected_result=idle_dependency,
                idle_slot_seconds_no_queued_or_selected_result=idle_empty,
                idle_slot_seconds_with_queued_candidate=idle_ready,
                integrated_available_slot_seconds=available,
                slot_utilization=busy/available if available else 0,
                capacity_measurement='INTEGRATED_EXECUTOR_START_AND_AUTO_CAPACITY_INTERVALS',
                speculative_adoption_fraction=s.metrics['speculative_adopted']/s.metrics['speculative_launched']if s.metrics['speculative_launched']else None,
                memory_scope='sampled sum of owned job process RSS, .1s interval; shared pages may be double-counted')


class DistributedExecutor(ConcurrentExecutor):
    """The existing queue/adoption order, with persistent host-owned attempts.

    Every host, including this controller's local slots, uses the same adapter
    worker. Assignment never expires merely because transport is unavailable.
    """
    @property
    def capacity(self):
        # Method views copy __dict__, but share the controller state. Dynamic
        # capacity must therefore live there, not in a copied scalar.
        state = getattr(self, 'state', None)
        return getattr(state, 'allocation_capacity', getattr(self, '_capacity', 1))

    @capacity.setter
    def capacity(self, value):
        self._capacity = value

    def __init__(self, *args, distributed, **kwargs):
        from .rpe_q3_store import JobStore
        from . import rpe_q3_coordinator as Q
        self.distributed = deepcopy(distributed)
        self.ready_queue_waves = distributed.get('ready_queue_waves', 2 if distributed.get('work_conserving_search') else 1)
        if type(self.ready_queue_waves) is not int or self.ready_queue_waves < 1:
            raise ValueError('ready_queue_waves must be a positive integer')
        self.eligibility_holds = {}
        self.eligibility_policy_cache = {}
        self.live_readiness_inflight = {}
        self.live_readiness_partial = {}
        self.live_readiness_workers = max(1, len(kwargs.get('problem', args[4] if len(args)>4 else {}).get('methods', {})))
        self.live_readiness_pool = (ThreadPoolExecutor(max_workers=self.live_readiness_workers, thread_name_prefix='rpe-q3-readiness')
                                    if self.distributed.get('live_q3_readiness') is True else None)
        self.host_admission_cache = {}
        self.preparation_holds = {}
        self.preparation_dependencies = {}
        run = Path(args[1]).resolve()
        requested_capacity = kwargs.get('max_concurrent', 1)
        if requested_capacity != 'auto' and (type(requested_capacity) is not int or requested_capacity < 1):
            raise ValueError('Positive max_concurrent_jobs or auto required')
        settings_path = run/'distributed'/'ALLOCATION_SETTINGS.json'
        default_allocation = dict(priority_groups=[['server2', 'server1', 'local'], ['local0-wsl']],
                                  offer_ttl_seconds=2.0, discovery_grace_seconds=1.0)
        config = dict(default_allocation, **(distributed.get('auto_allocation') or {}))
        if (config.get('offer_ttl_seconds', 0) <= 0 or config.get('remote_offer_ttl_seconds', 1) <= 0
                or config.get('discovery_grace_seconds', -1) < 0):
            raise ValueError('Positive offer lifetime and nonnegative discovery grace required')
        if settings_path.exists():
            settings = json.loads(settings_path.read_text())
            if settings.get('schema') != 'RPE_Q3_ALLOCATION_SETTINGS_V1':
                raise ValueError('Unknown saved allocation settings')
            saved = settings['max_concurrent_jobs']
            if requested_capacity != 'auto' and requested_capacity != saved:
                raise ValueError('Saved run concurrency cannot change on resume')
            requested_capacity = saved
            config = settings['auto_allocation']
        else:
            if requested_capacity == 'auto' and (run/'distributed'/'QUEUE.sqlite3').exists():
                old_round = json.loads((run/'ROUND.json').read_text()) if (run/'ROUND.json').exists() else {}
                requested_capacity = old_round.get('executor', {}).get('max_concurrent_jobs', 1)
            settings = dict(schema='RPE_Q3_ALLOCATION_SETTINGS_V1',
                mode='auto' if requested_capacity == 'auto' else 'fixed',
                max_concurrent_jobs=requested_capacity, auto_allocation=config)
        self.allocation_settings = settings
        self.allocation_settings_path = settings_path
        self.allocation_config = config
        kwargs['max_concurrent'] = 1 if requested_capacity == 'auto' else requested_capacity
        self.store = JobStore(run/'distributed'/'QUEUE.sqlite3')
        from .rpe_search_continuation import verify_retired_preparation_attempts
        self.retired_preparation_attempts = set(verify_retired_preparation_attempts(run).values())
        from .rpe_preparation_artifacts import ArtifactCache
        self.artifact_cache = ArtifactCache(run/'distributed'/'artifacts')
        self.preparation_pending_path = run/'distributed'/'PREPARATION_PENDING.json'
        saved_preparation = json.loads(self.preparation_pending_path.read_text()) if self.preparation_pending_path.exists() else {}
        if saved_preparation and saved_preparation.get('schema') != 'RPE_PENDING_PREPARATION_READINESS_V1':
            raise ValueError('Unknown saved preparation readiness schema')
        self.preparation_pending = saved_preparation.get('tasks', {})
        # A resumed controller can recheck a pending request which never had
        # a queue reservation. Existing attempt ownership stays in JobStore.
        for job_id, record in self.preparation_pending.items():
            if self.store.row(job_id) is None:
                record['probe_started_epoch'] = time.time()
        self.preparation_probe_offsets = {}
        self.hosts = {}
        self.input_hash_cache = {}
        self.local_worker = None
        self.local_worker_pid = None
        self.pbs_manager = None
        self.rpc_server = None
        super().__init__(*args, **kwargs)
        try:
            self.state.allocation_auto = requested_capacity == 'auto'
            self.state.allocation_capacity = kwargs['max_concurrent']
            self.state.allocation_start = time.monotonic()
            self.state.allocation_epoch = time.time()
            self.state.allocation_waits = {}
            self.state.allocation_wait_totals = {}
            self.state.allocation_discovery = {}
            if not settings_path.exists(): Q.atomic_json(settings_path, settings)
            self.controller_source = self.optimizer_source
            self.computation_root = Path(distributed.get('computation_root', self.root)).resolve()
            self.computation_source = distributed.get('computation_source', self.optimizer_source)
            self.optimizer_source = self.computation_source
            self.worker_script = Path(distributed.get('worker_script', self.root/'scripts/run_rpe_q3_worker.py')).resolve()
            self.source_manifest = Q.source_manifest(self.computation_root, self.computation_source)
            self.release_manifest=None
            if distributed.get('release_manifest'):
                from .rpe_release_deployment import read,manifest_reference,ensure_components
                self.release_manifest=read(distributed['release_manifest'])
                reference=manifest_reference(self.release_manifest)
                if reference!=distributed.get('required_release'):
                    raise ValueError('Controller release/component binding changed')
                if distributed.get('local_worker')and distributed.get('local_deployment_layout'):
                    report=ensure_components(self.release_manifest,read(distributed['local_deployment_layout']),
                        components=['optimizer'])
                    if report['status']!='PASS':raise ValueError('Local optimizer deployment HOLD: '+str(report['holds']))
            contract_path = run/'distributed'/'SOURCE.json'
            fixed = dict(source=self.source_manifest, evaluation_contract=self.contract)
            if contract_path.exists() and json.loads(contract_path.read_text()) != fixed:
                raise ValueError('Distributed run SOURCE/evaluation contract changed; use a new run')
            if not contract_path.exists(): Q.atomic_json(contract_path, fixed)
            with self.state.cv:
                for row in self.store.rows():
                    if row['state'] == 'QUEUED':
                        decision = self._eligibility_for(row['request'], row['job_id'], blocking=False,
                                                         previous=row['eligibility'])
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
            self.start_controller_relay(address[1])
            if distributed.get('pbs'):
                from .rpe_pbs import PbsManager
                self.pbs_manager = PbsManager(self, distributed['pbs'])
                self.pbs_manager.start()
            profile = distributed.get('local_worker')
            if profile and profile.get('slots', 0) > 0:
                profile = dict(profile)
                profile.setdefault('root', str(self.computation_root))
                profile.setdefault('work_dir', str(run/'distributed'/'server2-worker'))
                profile_path = run/'distributed'/'LOCAL_WORKER.json'
                Q.atomic_json(profile_path, profile)
                daemon_path = run/'distributed'/'LOCAL_DAEMON.json'
                previous = json.loads(daemon_path.read_text()) if daemon_path.exists() else {}
                if self._local_daemon_alive(previous.get('pid'), profile_path):
                    self.local_worker_pid = previous['pid']
                else:
                    command = [sys.executable, '-B', str(self.worker_script), '--profile', str(profile_path),
                               '--controller', self.controller_url, '--poll-seconds', '.1']
                    if distributed.get('controller_queue_first'):
                        command = ['nice', '-n', '10', *command]
                    with (run/'distributed'/'LOCAL_WORKER.log').open('a') as log:
                        self.local_worker = subprocess.Popen(command,
                            cwd=self.root, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
                    self.local_worker_pid = self.local_worker.pid
                    Q.atomic_json(daemon_path, dict(pid=self.local_worker_pid, profile=str(profile_path)))
            self.log(event='DISTRIBUTED_START', url=self.controller_url, controller_source=self.controller_source,
                computation_source=self.computation_source, computation_root=str(self.computation_root),
                allocation_settings=settings)
        except BaseException:
            with self.state.cv:
                self.state.closed = True
                self.state.cv.notify_all()
                self.state.monitor_stop.set()
            self.dispatcher.join(); self.monitor.join()
            if self.live_readiness_pool: self.live_readiness_pool.shutdown(wait=False, cancel_futures=True)
            self.stop_controller_relay()
            if self.pbs_manager: self.pbs_manager.close()
            if self.rpc_server: self.rpc_server.shutdown(); self.rpc_server.server_close()
            fcntl.flock(self.state.driver, fcntl.LOCK_UN); self.state.driver.close()
            raise

    def log(self, **event):
        # Reuse the existing dependency-wait boundaries emitted by the normal
        # answer path. These durations share the controller's monotonic clock.
        state = getattr(self, 'state', None)
        if state is not None and hasattr(state, 'allocation_waits') and event.get('job'):
            job = Path(str(event['job'])).name
            with self.lock:
                if event.get('event') == 'DEPENDENCY_WAIT_START':
                    state.allocation_waits.setdefault(job, []).append(time.monotonic())
                elif event.get('event') == 'DEPENDENCY_WAIT_END':
                    starts = state.allocation_waits.get(job, [])
                    if starts:
                        elapsed = time.monotonic()-starts.pop()
                        state.allocation_wait_totals[job] = state.allocation_wait_totals.get(job, 0)+elapsed
        return super().log(**event)

    def start_controller_relay(self, port):
        """The ordinary controller owns its direct SSH mailbox polling too."""
        settings=self.distributed.get('relay')
        if not settings:return
        from .rpe_q3_coordinator import atomic_json
        directory=self.run_dir/'distributed'/'relay';directory.mkdir(parents=True,exist_ok=True)
        path=directory/'CONFIG.json'
        config=dict(deepcopy(settings),state_directory=str(directory),external_tunnel=True,controller=dict(local_port=port))
        # A crashed controller can leave its polling relay alive. Reuse only
        # that exact endpoint/configuration; never create a second poller.
        with (directory/'RELAY.lock').open('a')as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                if not path.exists()or json.loads(path.read_text())!=config:
                    raise ValueError('Existing run relay configuration differs; reconcile its owner before resume')
                if (directory/'STOP').exists():
                    raise ValueError('Existing run relay is still draining; resume after its mailbox transfer finishes')
                self.state.controller_relay_directory=directory
                self.log(event='CONTROLLER_RELAY_REUSED',configuration=str(path));return
            atomic_json(path,config);(directory/'STOP').unlink(missing_ok=True)
        command=[sys.executable,'-B',str(self.root/'scripts/run_rpe_q3_relay.py'),'--config',str(path),'--run']
        with (directory/'RELAY.log').open('a')as log:
            process=subprocess.Popen(command,cwd=self.root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        self.state.controller_relay_directory=directory
        self.state.controller_relay_process=process
        atomic_json(directory/'LAUNCH.json',dict(pid=process.pid,command=command))
        self.log(event='CONTROLLER_RELAY_STARTED',configuration=str(path),pid=process.pid)

    def stop_controller_relay(self):
        directory=getattr(self.state,'controller_relay_directory',None)
        if directory:
            (directory/'STOP').touch()
            self.log(event='CONTROLLER_RELAY_STOP_REQUESTED',received_worker_jobs_preserved=True)
            process=getattr(self.state,'controller_relay_process',None)
            if process is not None:
                try:process.wait(timeout=self.distributed['relay'].get('request_timeout_seconds',30)+2)
                except subprocess.TimeoutExpired:
                    self.log(event='CONTROLLER_RELAY_DRAIN_PENDING',received_worker_jobs_preserved=True)

    def for_method(self, method):
        view = DistributedExecutor.__new__(DistributedExecutor)
        view.__dict__.update(self.__dict__)
        view.__dict__.pop('finish', None)
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

    def _host_admission_policy(self):
        """Optional participation is independent of request-specific readiness.

        A configured file is replaced atomically by the operator. Failure to
        read it stops new claims, while previously owned attempts still recover.
        """
        policy = self.distributed.get('host_admission')
        if policy is None: return dict(hosts={}), None
        try:
            if isinstance(policy, (str, Path)):
                path = Path(policy)
                if not path.is_absolute(): path = self.root/path
                st = path.stat()
                signature = (str(path.resolve()), st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
                if self.host_admission_cache.get('signature') != signature:
                    loaded = json.loads(path.read_text())
                    self.host_admission_cache.update(signature=signature, policy=loaded)
                policy = self.host_admission_cache['policy']
            if (not isinstance(policy, dict) or
                    policy.get('schema', 'RPE_Q3_HOST_ADMISSION_V1') != 'RPE_Q3_HOST_ADMISSION_V1' or
                    not isinstance(policy.get('hosts'), dict)):
                raise ValueError('HOST_ADMISSION_POLICY_INVALID')
            for host, rule in policy['hosts'].items():
                if not isinstance(host, str) or not host or not isinstance(rule, dict):
                    raise ValueError('HOST_ADMISSION_HOST_INVALID')
                if type(rule.get('accepting')) is not bool:
                    raise ValueError('HOST_ADMISSION_ACCEPTING_REQUIRED:' + host)
                if 'optional' in rule and type(rule['optional']) is not bool:
                    raise ValueError('HOST_ADMISSION_OPTIONAL_INVALID:' + host)
                if 'max_slots' in rule and (type(rule['max_slots']) is not int or rule['max_slots'] < 0):
                    raise ValueError('HOST_ADMISSION_MAX_SLOTS_INVALID:' + host)
            return policy, None
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return dict(hosts={}), str(exc)

    def _host_admission(self, host, policy, error=None):
        execution_hosts = self.distributed.get('execution_hosts')
        if execution_hosts is not None and host not in execution_hosts:
            return dict(accepting=False, optional=True, max_slots=0,
                        reason='OUTSIDE_FIXED_EXECUTION_HOSTS')
        rule = policy['hosts'].get(host, {})
        return dict(accepting=rule.get('accepting', True) and error is None,
                    optional=rule.get('optional', False), max_slots=rule.get('max_slots'),
                    error=error)

    def _host_new_slots(self, host, slots, admission):
        if not admission['accepting']: return 0
        maximum = admission['max_slots']
        if maximum is None: return slots
        # UNKNOWN and completed-but-untransferred attempts still belong to this
        # host. A communication outage must not open another execution slot.
        owned = sum(row['host_id'] == host for row in self._owned_attempts())
        return min(slots, max(0, maximum-owned))

    def _local_calibration_pair_allows(self, host, request):
        trial = self.distributed.get('local_calibration_pair')
        if host != 'local' or trial is None:
            return True
        owned = [self.store.row(row['job_id'])['request'] for row in self._owned_attempts()
                 if row['host_id'] == host]
        if not owned:
            return True
        if len(owned) != 1:
            return False
        preparations = [(item.get('preparation') or {}) for item in [*owned, request]]
        pf_providers = {'PF_SIGNALS', 'PF_AUDIT', 'PF_HOLDOUT_AUDIT', 'PF_MODEL_FREEZE', 'PF_FINISH'}
        pf_pair = trial.get('pf_preparation') and all(p.get('provider') in pf_providers for p in preparations)
        calibration_pair = all(p.get('provider') == 'DF_CALIBRATION' and p.get('parallel_df_price_axes') is True
                               for p in preparations)
        if not (pf_pair or calibration_pair):
            return False
        from .rpe_q3_resources import resource_snapshot
        profile = self.distributed['local_worker']
        budget = resource_snapshot(profile)
        free = [budget[key] for key in ('machine_mem_available_bytes', 'remaining_group_memory_bytes')
                if budget.get(key) is not None]
        from .rpe_pbs_worker import request_cpu_footprint
        try:
            cpus = sum(request_cpu_footprint(profile, item) for item in [*owned, request])
        except (KeyError, ValueError):
            return False
        return (bool(free) and min(free) >= trial['memory_reserve_bytes']
                and (budget.get('allowed_cpu_count') or 0) >= cpus)

    def _wait_for_preparation_preference(self, host, row, policy, error):
        preference = self.distributed.get('preparation_host_preference')
        prep = row['request'].get('preparation') or {}
        if not preference or prep.get('provider') not in preference['providers']:
            return False
        preferred = preference['host']
        if (host == preferred or time.time() - row['enqueued_epoch'] >= preference['seconds']
                or not self._host_supports_stage(preferred, row['request'])
                or not self._offer_free(preferred, policy, error)):
            return False
        record = self.preparation_pending.get(row['job_id'], {})
        return preferred not in record.get('failures', {})

    def _owned_attempts(self):
        with self.store._connection() as database:
            rows = database.execute(
                "SELECT job_id,host_id,attempt_id,state,method,detail,assigned_epoch FROM jobs "
                "WHERE host_id IS NOT NULL AND state NOT IN ('QUEUED','CANCELLED','ACCEPTED')")
            # UNKNOWN still owns the request. Release only its capacity when
            # the worker proved it exited or a retired PBS allocation is verified.
            return [dict(row) for row in rows if not (
                row['state'] == 'UNKNOWN' and (
                    (row['method'].startswith('PREPARATION:')
                     and (row['job_id'], row['attempt_id']) in self.retired_preparation_attempts)
                    or (json.loads(row['detail']).get('resource_released') is True
                        and json.loads(row['detail']).get('automatic_retry') is False)))]

    def _check_worker_contact(self):
        """Yield waiters after lost contact while keeping the queue attempt owned."""
        state = self.state
        with state.cv:
            if state.closed or not any(task.status == 'RUNNING' for task in state.tasks.values()):
                return
            if not hasattr(state, 'allocation_epoch'):
                return  # DistributedExecutor startup has not reached the worker setup.
            now = time.time()
            relay = self.distributed.get('relay') or {}
            pbs = self.distributed.get('pbs') or {}
            grace = 3 * max(30, relay.get('request_timeout_seconds', 30),
                            pbs.get('request_timeout_seconds', 20),
                            self.allocation_config.get('remote_offer_ttl_seconds', 0))
            for row in self._owned_attempts():
                job = self.run_dir/'jobs'/row['method']/row['job_id']
                task = state.tasks.get(job)
                if task is None or task.status != 'RUNNING':
                    continue
                observed = self.hosts.get(row['host_id'], {}).get('epoch', state.allocation_epoch)
                assigned = row['assigned_epoch']
                if now - max(observed, assigned) <= grace:
                    continue
                task.error = 'WORKER_CONTACT_LOST'
                self._finished(task, None, assigned)
                task.view.log(event='WORKER_CONTACT_LOST', job=row['job_id'], host_id=row['host_id'],
                              attempt_id=row['attempt_id'], owner_preserved=True, automatic_retry=False)

    def _offer_free(self, host, policy, error=None):
        """Fresh unused capacity, never a lease on an assigned evaluation."""
        info = self.hosts.get(host)
        if not self._offer_fresh(host):
            return 0
        if info.get('accepting_new_jobs') is False: return 0
        admission = self._host_admission(host, policy, error)
        free = self._host_new_slots(host, info['reported_slots'], admission)
        owned = self._owned_attempts()
        current = {row['attempt_id'] for row in owned if row['host_id'] == host}
        # An offer cannot be reused by another poll after its free slot has
        # already been committed to an attempt by this controller.
        free = max(0, free-len(current-set(info.get('owned_at_offer', []))))
        capacity = info.get('capacity')
        if type(capacity) is int and capacity >= 0:
            free = min(free, max(0, capacity-len(current)))
        return free

    def _offer_fresh(self, host):
        ttl = self.allocation_config['offer_ttl_seconds']
        info = self.hosts.get(host, {})
        manager = self.pbs_manager
        if manager is not None and host in manager.configured:
            pbs = (info.get('resource_report') or {}).get('pbs') or {}
            if not manager.accepting_allocation(host, pbs.get('job_id')):
                return False
            # A PBS poll traverses two shared-spool exchanges, and can carry
            # several fixed-input checks. Use the gateway's transport window.
            ttl = max(ttl, manager.settings.get('request_timeout_seconds', 20))
        elif 'remote_offer_ttl_seconds' in self.allocation_config and any(
                w['host_id'] == host for w in (self.distributed.get('relay') or {}).get('workers', ())):
            ttl = self.allocation_config['remote_offer_ttl_seconds']
        return time.monotonic() - info.get('observed_monotonic', float('-inf')) <= ttl

    def _refresh_capacity(self, policy, error=None):
        if not self.state.allocation_auto: return
        owned = len(self._owned_attempts())
        queue = self._ready_queue()
        available = {host:self._offer_free(host, policy, error) for host in self.hosts}
        available = {host:min(free,len(queue)) for host,free in available.items() if free > 0}
        usable = len(self._match_ready_jobs(queue, available))
        capacity = max(1, owned+usable)
        if capacity != self.state.allocation_capacity:
            self.state.allocation_capacity = capacity
            self.log(event='AUTO_CAPACITY', capacity=capacity, unresolved_owned=owned,
                     ready_jobs=len(queue), fresh_free_slots=sum(available.values()),
                     request_compatible_slots=usable,
                     reason='MATCHED_READY_REQUESTS_TO_PARTICIPATING_FRESH_ELIGIBLE_SLOTS')

    def _ready_queue(self):
        def priority(task):
            preparation = task.rule['kind'] == 'preparation'
            conditional = (task.proposal.get('origin', {}).get('kind') == 'CHAIN_LOOKAHEAD_IF_PREVIOUS_PASSES'
                           and not task.demand)
            if self.distributed.get('controller_queue_first'):
                # Fill workers from ready independent steps across candidates.
                # A candidate's current Q3 does not outrank others' preparation.
                rank = int(conditional)
            else:
                rank = 2 if not task.demand else int(self.distributed.get('live_ready_refill', False) and preparation)
                if self.distributed.get('work_conserving_search') and not preparation and not task.demand and not conditional:
                    rank = 1
            return (getattr(task, 'optional_price_validation', False), rank,
                    self.state.owner_clock.get(task.owner, -1), task.enqueue)

        return sorted((task for task in self.state.tasks.values() if task.status == 'QUEUED'
                       and self._task_ready(task)), key=priority)

    def _budget_reserved(self):
        cap = self.state.budgets.get(self.method)
        return type(cap) is int and self.store.counts(self.method) >= cap

    def _ready_cpu_supply(self, ready, owned, policy, error):
        """Measure compatible waiting work against each fresh worker's CPU offer."""
        from .rpe_pbs import resource_need
        from .rpe_pbs_worker import runtime_cpu_footprint

        manager = self.pbs_manager
        rows = [self.store.row(task.job.name) for task in ready]
        owned_rows = [self.store.row(row['job_id']) for row in owned]
        supply = {}
        for host, info in self.hosts.items():
            slots = self._offer_free(host, policy, error)
            if not slots:
                continue
            report = info.get('resource_report') or {}
            limit = (report.get('resources') or {}).get('allowed_cpu_count')
            pool = (report.get('pbs') or {}).get('resource_pool')
            if pool is not None:
                limit = pool['cpu_count']
            profile = (manager.configured[host]['profile'] if manager and host in manager.configured
                       else self.distributed.get('local_worker') or {} if host == 'local' else {})

            def footprint(row):
                if pool is not None:
                    return resource_need(row['request'], profile, manager.settings, host)
                runtime = ((row['eligibility'] or {}).get('execution_profiles') or {}).get(host)
                if runtime is None:
                    return None
                return runtime_cpu_footprint(profile, runtime), 0

            try:
                running = [footprint(row) for row in owned_rows if row['host_id'] == host]
                if None in running:
                    continue
                remaining_ram = (pool['memory_bytes'] - (1 << 30) - sum(memory for _, memory in running)
                                 if pool is not None else None)
                by_stage = Counter()
                estimates = []
                unknown = 0
                for row in rows:
                    request = row['request']
                    if (not self._host_supports_stage(host, request) or not self._fits_host_walltime(host, request)
                            or not self.store.eligible(request, row['eligibility'], host,
                                                       info.get('host_fingerprints'), row['job_id'])):
                        continue
                    need = footprint(row)
                    if need is None:
                        unknown += 1
                        continue
                    cpu, memory = need
                    if (remaining_ram is not None and memory > remaining_ram) or (limit and cpu > limit):
                        continue
                    stage = (request.get('preparation') or {}).get('stage', 'q3')
                    by_stage[stage] += cpu
                    estimates.append(cpu)
                known_limit = type(limit) is int and limit > 0
                if not known_limit and report.get('mode') != 'FIXED':
                    continue
                free_cpu = max(0, limit - sum(cpu for cpu, _ in running)) if known_limit else slots * max(estimates, default=0)
                target = free_cpu * self.ready_queue_waves
                supply[host] = dict(available_cpu=free_cpu, ready_cpu=sum(by_stage.values()),
                                    capacity_source='RESOURCE_OFFER' if known_limit else 'FIXED_SLOTS_BY_READY_FOOTPRINT',
                                    ready_cpu_by_stage=dict(by_stage), unestimated_jobs=unknown,
                                    target_cpu=target, missing_cpu=max(0, target-sum(by_stage.values())))
            except (KeyError, ValueError, TypeError):
                continue
        return supply

    def preparation_needs_refill(self):
        """Keep producing while any accepting host lacks ready work."""
        with self.state.cv:
            supply = [row for host, row in getattr(self.state, 'ready_cpu_supply', {}).items()
                      if self._offer_fresh(host)]
            return not supply or any(not row['ready_cpu'] or row['missing_cpu'] or row['unestimated_jobs']
                                     for row in supply)

    def _standby_capacity(self):
        if not self.state.allocation_auto:
            return self.capacity
        policy, error = self._host_admission_policy()
        fresh = sum(self._offer_free(host, policy, error) for host in self.hosts)
        return max(self.capacity, self.state.active + fresh)

    def _standby_headroom(self):
        if not self.state.allocation_auto or not self.distributed.get('work_conserving_search'):
            return super()._standby_headroom()
        policy, error = self._host_admission_policy()
        targets = {}
        manager = self.pbs_manager
        potential = {}
        if (self.ready_queue_waves > 1 and manager is not None
                and manager.settings.get('allocation_sizing', {}).get('per_task_resources')):
            # Size the preparation of requests against the authorized pool. A
            # small first qsub must not become the ceiling for the next qsub.
            from .rpe_pbs import usable_memory
            from .rpe_pbs_worker import runtime_cpu_footprint
            settings = manager.settings
            for host, worker in manager.configured.items():
                profile = worker['profile']
                slots = profile.get('slot_policy', {})
                cpu = min(settings['resources']['ncpus'], slots.get('resource_limits', {}).get(
                    'cpu_count', settings['resources']['ncpus']))
                runtimes = [profile] + [r['profile'] for r in profile.get('route_profiles', ())]
                per_job = max(runtime_cpu_footprint(profile, rt) for rt in runtimes)
                memory = max(settings['allocation_sizing']['q3_startup_memory_bytes_per_task'],
                             (profile.get('pbs') or {}).get('task_memory_bytes') or 0)
                potential[host] = min(cpu // per_job, usable_memory(profile, settings) // memory,
                                      slots.get('max_slots') or cpu)
        for host in self.hosts.keys() | potential.keys():
            admission = self._host_admission(host, policy, error)
            offer = self.hosts.get(host, {})
            if not admission['accepting']:
                continue
            fresh = self._offer_fresh(host)
            if fresh and offer.get('accepting_new_jobs') is False:
                continue
            lanes = max(potential.get(host, 0), int(offer.get('capacity') or 0) if fresh else 0)
            if admission['max_slots'] is not None:
                lanes = min(lanes, admission['max_slots'])
            if lanes and self._host_supports_stage(host, {}):
                targets[host] = lanes * self.ready_queue_waves
        capacity = sum(targets.values())
        if self.ready_queue_waves > 1 and capacity:
            # A backlog usable only on one host can conceal a shortage on the
            # others. Look ahead by at most their missing waves as well.
            covered = Counter()
            for task in self.state.tasks.values():
                if task.status == 'QUEUED' and task.rule['kind'] != 'preparation':
                    allowed = task.supply_hosts
                    covered.update(targets.keys() if allowed is None else targets.keys() & allowed)
            deficits = {host: max(0, count - covered[host]) for host, count in targets.items()}
            cpu_supply = getattr(self.state, 'ready_cpu_supply', {})
            for host in targets.keys() & cpu_supply.keys():
                offer = cpu_supply[host]
                # CPU deficits supplement the lane allowance when waiting jobs
                # use fewer cores than the current host profile can allocate.
                if (self._offer_fresh(host) and not offer['unestimated_jobs']
                        and offer['ready_cpu_by_stage'].get('q3')):
                    cpu_per_q3 = max(1, offer['ready_cpu_by_stage']['q3'] // max(1, covered[host]))
                    deficits[host] = max(deficits[host],
                        (offer['missing_cpu'] + cpu_per_q3 - 1) // cpu_per_q3)
            capacity += sum(deficits.values())
        return max(1, capacity)

    def _can_take_ready_candidates(self):
        if self.ready_queue_waves > 1 and self.state.allocation_auto and self.distributed.get('work_conserving_search'):
            queued = sum(t.status == 'QUEUED' and t.rule['kind'] != 'preparation' for t in self.state.tasks.values())
            return not self._budget_reserved() and queued < self._standby_headroom()
        return super()._can_take_ready_candidates()

    def _standby_can_fill(self):
        return (getattr(self, '_awaiting_probe_revision', None) !=
                getattr(self.state, 'probe_revision', 0) and super()._standby_can_fill())

    def _fill_standby(self):
        if not (self.live_readiness_pool and self.ready_queue_waves > 1
                and self.distributed.get('work_conserving_search')):
            return super()._fill_standby()
        # A selector must be able to adopt its demand result while later
        # requests are still in dry-check. Leave pending proposals unreserved.
        waiting = []
        added = 0
        with self.state.cv:
            revision = getattr(self.state, 'probe_revision', 0)
        while True:
            with self.state.cv:
                if not self._standby_can_fill():
                    break
                if self._budget_reserved() or self._user_stop_requested():
                    self.standby.clear()
                    waiting.clear()
                    break
                proposal = self.standby.popleft()
            task = self._enqueue(proposal, defer_readiness=True)
            if task is _CHECK_PENDING:
                waiting.append(proposal)
                if len(waiting) >= self.live_readiness_workers:
                    break
            elif task:
                added += 1
        with self.state.cv:
            if waiting:
                self.standby.extendleft(reversed(waiting))
                self._awaiting_probe_revision = revision
        if added:
            self.log(event='Q3_STANDBY_REFILL', added=added, remaining=len(self.standby))

    def has_spare_slot(self, *, allow_queued_speculation=False):
        # Auto capacity is sized against already queued jobs. With one job
        # running it can equal one even when another worker has a free offer.
        with self.state.cv:
            if self.state.failure or self.state.closed:
                return False
            owned = max(self.state.active, len(self._owned_attempts()))
            if not self.hosts:
                return not owned
            policy, error = self._host_admission_policy()
            offers = {host: self._offer_free(host, policy, error) for host in self.hosts}
            offers = {host: count for host, count in offers.items() if count}
            free = sum(offers.values())
            if not self.state.allocation_auto:
                free = min(free, self.capacity - owned)
            if free <= 0:
                return False
            demand = [task for task in self._ready_queue() if task.demand and task.rule['kind'] != 'preparation']
            return free > len(self._match_ready_jobs(demand, offers))

    @staticmethod
    def _bound_timing_profile(request, host, profile):
        from .rpe_host_profiles import digest, profile_core
        route = request['route']
        bound = digest(profile_core(profile)) if profile is not None else None
        return host, route.get('kind'), route.get('view_commit'), bound

    def search_profile_for_proposal(self, proposal):
        route = self.route(proposal['configuration_id'])
        if route is None:
            return None
        measured = {row['profile'] for row in self.search_observations()
                    if row['configuration_id'] == proposal['configuration_id']
                    and row['profile'][1:3] == (route.get('kind'), route.get('view_commit'))}
        if not measured:
            return None
        request = {'route': route}
        with self.state.cv:
            policy, error = self._host_admission_policy()
            compatible = [profile for profile in measured if (host := profile[0]) in self.hosts
                        and self._host_admission(host, policy, error)['accepting']
                        and self.hosts[host]['accepting_new_jobs']
                        and self._offer_fresh(host)
                        and self._host_supports_stage(host, request) and self._fits_host_walltime(host, request)]
            if not compatible:
                return None
            return min(compatible, key=lambda item: (not self._offer_free(item[0], policy, error),
                                                      self._host_rank(item[0]), item[0], item[3] or ''))

    def _observe_q3_completion(self, row, transfer, validation):
        if validation['status'] not in (O.PASS, O.FAIL):
            return
        from .rpe_q3_timing import q3_body_spans
        try:
            _, _, spans = q3_body_spans(transfer)
        except (OSError, ValueError, KeyError, TypeError):
            return  # missing timing does not affect acceptance of a validated result
        profiles = (row.get('eligibility') or {}).get('execution_profiles', {})
        host = row['host_id']
        profile = profiles.get(host)
        if not spans or (profile is None and not self.store.artificial(row['request'])):
            return
        self.state.observations[row['method']].append(dict(
            id=row['job_id'], kind='SHOT_Q3', method=row['method'],
            configuration_id=row['request']['configuration_id'],
            profile=self._bound_timing_profile(row['request'], host, profile),
            status=validation['status'], finished_at=time.monotonic(), seconds=sum(spans)))

    def _host_rank(self, host):
        groups = self.allocation_config['priority_groups']
        if host == 'local' and self.distributed.get('controller_queue_first'):
            return len(groups)
        return next((index for index, group in enumerate(groups) if host in group), 0)

    def _preferred_ready_jobs(self, host, queue, policy, error=None):
        """Match existing READY jobs to immediately usable higher-priority slots.

        This is only a choice among free execution slots. It creates no job
        ownership, budget reservation or attempt ID. Augmenting paths preserve
        the original candidate order while avoiding a flexible job consuming
        the sole compatible slot of another independent job.
        """
        rank = self._host_rank(host)
        if rank == 0: return set()
        available = {h:self._offer_free(h, policy, error) for h in self.hosts
                     if self._host_rank(h) < rank}
        available = {h:min(count, len(queue)) for h,count in available.items() if count > 0}
        return self._match_ready_jobs(queue, available)

    def _match_ready_jobs(self, queue, available):
        """One request per usable host slot, preserving candidate order."""
        owners = {h:[] for h in available}
        compatible = {};skipped = {h:False for h in available}
        policy, error = (self._host_admission_policy() if self.distributed.get('preparation_host_preference')
                         else (None, None))
        for task in queue:
            row = self.store.row(task.job.name)
            compatible[task.job.name] = []
            for h in available:
                if not self._host_supports_stage(h, row['request']) or not self._fits_host_walltime(h, row['request']):
                    skipped[h] = True
                    continue
                if (self._wait_for_preparation_preference(h, row, policy, error)
                        or not self._local_calibration_pair_allows(h, row['request'])):
                    skipped[h] = True
                    continue
                if not self.store.eligible(row['request'], row['eligibility'], h,
                        self.hosts[h].get('host_fingerprints'), row['job_id']):
                    skipped[h] = True
                    continue
                conditional = task.proposal.get('origin', {}).get('kind') == 'CHAIN_LOOKAHEAD_IF_PREVIOUS_PASSES'
                if not (conditional and skipped[h] and not task.demand):
                    compatible[task.job.name].append(h)

        def assign(job, seen):
            for preferred in compatible[job]:
                if preferred in seen: continue
                seen.add(preferred)
                if len(owners[preferred]) < available[preferred]:
                    owners[preferred].append(job)
                    return True
                for offset, other in enumerate(list(owners[preferred])):
                    if assign(other, seen):
                        owners[preferred][offset] = job
                        return True
            return False

        matched = set()
        for task in queue:
            if assign(task.job.name, set()): matched.add(task.job.name)
        return matched

    def _host_supports_stage(self, host, request):
        capabilities = (self.distributed.get('host_stage_capabilities') or {}).get(host)
        preparation=request.get('preparation') or {}
        stage = preparation.get('stage', 'q3')
        provider = preparation.get('provider')
        providers = (self.distributed.get('preparation_provider_hosts') or {}).get(provider)
        if provider in ('PF_COMPILE', 'PF_HOLDOUT_COMPILE') and self.distributed.get('pf_legacy_compile_hosts'):
            adapter = 'private/pf-transport/scripts/run_pf_preparation_stage.py'
            if adapter not in preparation.get('inputs', {}):
                providers = self.distributed['pf_legacy_compile_hosts']
        return (capabilities is None or stage in capabilities) and (providers is None or host in providers)

    def _fits_host_walltime(self, host, request):
        from .rpe_pbs import fits_request
        return fits_request(request, self.hosts.get(host, {}).get('resource_report'),
                            self.distributed.get('pbs') or {}, host)

    def _capacity_feedback(self, host):
        with self.store._connection() as database:
            own = [dict(row) for row in database.execute(
                'SELECT job_id,state,completed_epoch,adopted_epoch FROM jobs WHERE host_id=?', (host,))]
        epoch = self.state.allocation_epoch
        now = time.monotonic()
        with self.lock:
            totals = dict(self.state.allocation_wait_totals)
            for job, starts in self.state.allocation_waits.items():
                if starts: totals[job] = totals.get(job, 0)+sum(now-start for start in starts)
        selected = [totals[row['job_id']] for row in own if row['job_id'] in totals]
        ready = 0
        for task in self.state.tasks.values():
            if task.status != 'QUEUED' or not self._task_ready(task): continue
            row = self.store.row(task.job.name)
            if row is not None and self._host_supports_stage(host, row['request']) and self.store.eligible(row['request'], row['eligibility'], host,
                    self.hosts.get(host, {}).get('host_fingerprints'), row['job_id']):
                ready += 1
        return dict(controller_invocation=self.invocation, measurement_seconds=now-self.state.allocation_start,
            controller_clock_id=stamp()['clock_id'],
            usable_results=sum(row['state'] == 'ACCEPTED' and (row['completed_epoch'] or 0) >= epoch for row in own),
            adopted_results=sum((row['adopted_epoch'] or 0) >= epoch for row in own),
            decision_wait_seconds=sum(selected) if selected else None,
            decision_wait_scope='SUM_SELECTED_JOB_DEPENDENCY_WAIT_SPANS_SAME_CONTROLLER_CLOCK',
            ready_jobs=ready, capacity=self.capacity)

    def _eligibility_for(self, request, job_id, *, blocking=True, previous=None, host=None, prefetch=False):
        """Bind a prepared host decision to this exact scientific request."""
        from . import rpe_q3_coordinator as Q
        if request.get('route', {}).get('kind') == 'preparation':
            return self._preparation_eligibility_for(request, job_id)
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
        if (self.distributed.get('live_q3_readiness') is True and not result['eligible_host_ids']
                and result['reasons'] in (['NO_REQUEST_SPECIFIC_HOST_EVIDENCE'],
                                          ['HOST_ELIGIBILITY_CHANGED_REQUEST_SHA256'])):
            return self._live_eligibility(request, job_id, result, blocking=blocking, previous=previous,
                                          host=host, prefetch=prefetch)
        return result

    def _live_eligibility(self, request, job_id, result, *, blocking, previous, host, prefetch=False):
        """Share a probe per REQUEST and admit hosts as their independent checks pass."""
        from .rpe_live_q3_readiness import probe
        from .rpe_q3_coordinator import digest

        key = (job_id, result['request_sha256'])
        with self.state.cv:
            cache = self.eligibility_policy_cache.setdefault('live_checks', {})
            cached = cache.get(key)
            seen = frozenset(name for name, report in self.hosts.items()
                              if report.get('accepting_new_jobs', True))
            eligible = (cached[2] if cached else None) or previous or result
            if not eligible['eligible_host_ids'] and previous:
                eligible = previous  # a claim still requires a fresh matching worker fingerprint
            eligible = self.live_readiness_partial.get(key, eligible)
            future = self.live_readiness_inflight.get(key)
            missing_host = host is not None and host in seen and host not in eligible['eligible_host_ids']
            if blocking or missing_host or prefetch:
                holds = (eligible.get('host_decisions') or {}).values()
                fixed_scope_hold = holds and all(h.get('reasons') in (
                    ['COUNT_ONLY_SCOPE_EXCEEDED'], ['OUTSIDE_VALIDATED_R3_SCALAR_EQUIVALENCE_DOMAIN'])
                    for h in holds)
                stale = not cached or (not eligible.get('controller_scope') and (
                    cached[1] != seen or not fixed_scope_hold
                    and (not eligible['eligible_host_ids'] or missing_host)
                    and time.monotonic() - cached[0] >= LIVE_READINESS_RETRY_SECONDS))
                if (stale and future is None and not self.state.closed and not self._user_stop_requested()
                        and (blocking or len(self.live_readiness_inflight) < self.live_readiness_workers)):
                    def publish(decision):
                        with self.state.cv:
                            if self.state.closed:
                                return
                            # An unobserved host keeps its prior PASS until this
                            # refresh reaches it. The claim still checks current facts.
                            observed = decision['host_decisions']
                            for name in eligible['eligible_host_ids']:
                                if name not in observed:
                                    decision['eligible_host_ids'].append(name)
                                    for section in ('host_fingerprints', 'execution_profiles', 'host_decisions'):
                                        if name in eligible.get(section, {}):
                                            decision[section][name] = eligible[section][name]
                            decision['eligible_host_ids'].sort()
                            decision.pop('decision_sha256', None)
                            decision['decision_sha256'] = digest(decision)
                            self.live_readiness_partial[key] = decision
                            self.state.probe_revision = getattr(self.state, 'probe_revision', 0) + 1
                            self.state.cv.notify_all()

                    def check():
                        decision = result
                        try:
                            decision = probe(self, request, job_id, deepcopy(result), publish=publish)
                        except (OSError, KeyError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                            decision['reasons'].append(str(exc))
                        finally:
                            with self.state.cv:
                                cache[key] = (time.monotonic(), seen, decision)
                                self.live_readiness_partial.pop(key, None)
                                self.live_readiness_inflight.pop(key, None)
                                self.state.probe_revision = getattr(self.state, 'probe_revision', 0) + 1
                                self.state.cv.notify_all()
                        return decision

                    future = self.live_readiness_pool.submit(check)
                    self.live_readiness_inflight[key] = future
            if not blocking or future is None or eligible['eligible_host_ids']:
                return deepcopy(eligible)
            while key in self.live_readiness_inflight and not self._user_stop_requested():
                if key in self.live_readiness_partial:
                    return deepcopy(self.live_readiness_partial[key])
                self.state.cv.wait(.25)
            if self._user_stop_requested():
                return deepcopy(result)
        return deepcopy(future.result())

    def prefetch_standby(self, choice, proposals):
        self._awaiting_probe_revision = None
        if self.distributed.get('controller_queue_first') and self.live_readiness_pool:
            # Start the fixed-input checks for the demand and independent next
            # candidates together. A check does not create a reservation.
            proposals = list(proposals)
            candidates = ([choice] if choice is not None else []) + proposals
            checked = 0
            for proposal in candidates:
                with self.state.cv:
                    if (self.state.closed or self._user_stop_requested() or self._budget_reserved()
                            or checked >= self.live_readiness_workers
                            or len(self.live_readiness_inflight) >= self.live_readiness_workers):
                        break
                if self.has_job(proposal) or not self._candidate_preparation_ready(proposal):
                    continue
                route = self.route(proposal['configuration_id'])
                if route is None:
                    continue
                job = self.job_dir(proposal)
                request = self.request(proposal, job, route)
                self._eligibility_for(request, job.name, blocking=False, prefetch=True)
                checked += 1
        return super().prefetch_standby(choice, proposals)

    @staticmethod
    def _preparation_request(request):
        return request.get('preparation') if request.get('route', {}).get('kind') == 'preparation' else None

    def _preparation_inputs(self, request):
        preparation = self._preparation_request(request)
        manifest = preparation.get('inputs', {})
        if isinstance(manifest, list): manifest = {item['path']:item['sha256'] for item in manifest}
        manifest = dict(manifest)
        # Named dependency outputs are bound to the same immutable predecessor
        # task, and resolved only after its validated completion is accepted.
        for predecessor, paths in preparation.get('dependency_inputs', {}).items():
            row = self.store.row(predecessor)
            if row is None or row['state'] != 'ACCEPTED': raise ValueError('Preparation dependency is not accepted: '+predecessor)
            outcome = json.loads((self.run_dir/'jobs'/row['method']/predecessor/'OUTCOME.json').read_text())
            if outcome.get('status') != 'PREPARATION_COMPLETE': raise ValueError('Preparation dependency did not complete: '+predecessor)
            for source, target in paths.items():
                sha = outcome['artifacts'][source]
                if target in manifest and manifest[target] != sha: raise ValueError('Dependency artifact binding conflicts: '+target)
                manifest[target] = sha
        return manifest

    def _task_ready(self, task):
        request = self.store.row(task.job.name)['request']
        preparation = self._preparation_request(request)
        if preparation is None: return True
        for predecessor in preparation.get('dependencies', []):
            row = self.store.row(predecessor)
            if row is None or row['state'] != 'ACCEPTED': return False
            outcome = self.run_dir/'jobs'/row['method']/predecessor/'OUTCOME.json'
            if not outcome.exists() or json.loads(outcome.read_text()).get('status') != 'PREPARATION_COMPLETE': return False
        try: return not self.artifact_cache.missing(self._preparation_inputs(request).values())
        except (KeyError, ValueError, OSError): return False

    def _preparation_eligibility_for(self, request, job_id):
        from . import rpe_q3_coordinator as Q
        preparation = self._preparation_request(request)
        policy = self.distributed.get('preparation_host_eligibility')
        pending = self.preparation_pending.get(job_id)
        if pending is not None: return deepcopy(pending['decision'])
        row = self.store.row(job_id)
        # Explicit submission decisions are persisted in the same durable row;
        # there is no implicit transfer of Q3's PASS to a different stage.
        if policy is None and row is not None: return row['eligibility']
        if policy is None and self.store.artificial(request): return None
        result = dict(request_sha256=Q.digest(request), eligible_host_ids=[], host_fingerprints={},
                      execution_profiles={}, reasons=[])
        try:
            if isinstance(policy, (str, Path)):
                path = Path(policy)
                if not path.is_absolute(): path = self.root/path
                policy = json.loads(path.read_text())
            if not isinstance(policy, dict) or policy.get('schema') != 'RPE_PREPARATION_HOST_ELIGIBILITY_V1':
                raise ValueError('MISSING_PREPARATION_STAGE_READINESS')
            record = (policy.get('jobs') or {}).get(job_id)
            if not isinstance(record, dict): raise ValueError('NO_EXACT_PREPARATION_REQUEST_EVIDENCE')
            if record.get('decision_sha256') != Q.digest({k:v for k,v in record.items() if k != 'decision_sha256'}):
                raise ValueError('PREPARATION_READINESS_DECISION_CHANGED')
            if record.get('request_sha256') != Q.digest(request): raise ValueError('PREPARATION_REQUEST_CHANGED')
            if record.get('source_manifest') != preparation.get('source'): raise ValueError('PREPARATION_SOURCE_CHANGED')
            if record.get('input_hashes', []) != preparation.get('input_hashes', []): raise ValueError('PREPARATION_INPUTS_CHANGED')
            result['decision_sha256'] = record['decision_sha256']
            result['host_decisions'] = {host:dict(status=cell.get('status', 'UNCHECKED'),reasons=cell.get('reasons', []))
                for host,cell in (record.get('hosts') or {}).items()}
            for host in record.get('eligible_host_ids', []):
                cell = record.get('hosts', {}).get(host, {})
                fp = cell.get('current_host_fingerprint_sha256')
                if cell.get('status') != 'PASS' or not isinstance(fp, str) or not fp: continue
                current = (self.hosts.get(host, {}).get('host_fingerprints') or {}).get(job_id)
                if current is not None and Q.digest(current) != fp:
                    result['reasons'].append('CURRENT_PREPARATION_FINGERPRINT_CHANGED:'+host)
                    continue
                result['eligible_host_ids'].append(host);result['host_fingerprints'][host] = fp
                if cell.get('execution_profile') is not None: result['execution_profiles'][host] = deepcopy(cell['execution_profile'])
            if not result['eligible_host_ids']: result['reasons'].append('ALL_PREPARATION_HOSTS_FAIL_OR_UNCHECKED')
        except (OSError, KeyError, ValueError, TypeError) as exc: result['reasons'].append(str(exc))
        return result

    def _configured_preparation_profile(self, host):
        configured = (self.distributed.get('preparation_profiles') or {}).get(host)
        if isinstance(configured, (str, Path)):
            path = Path(configured)
            if not path.is_absolute(): path = self.root/path
            configured = json.loads(path.read_text())
        if not isinstance(configured, dict) or configured.get('host_id') != host:
            raise ValueError('No configured preparation stage profiles for host: '+host)
        return configured

    def _save_preparation_pending(self):
        from .rpe_q3_coordinator import atomic_json
        atomic_json(self.preparation_pending_path,dict(schema='RPE_PENDING_PREPARATION_READINESS_V1',tasks=self.preparation_pending))

    def _pending_preparation_assignment(self, record, host):
        from .rpe_q3_coordinator import digest
        from .rpe_preparation_worker import select_preparation_profile,preparation_profile_core
        request=record['request'];prep=request['preparation']
        assignment=dict(job_id=request['request_id'],request=request,request_sha256=digest(request),
            source_manifest=prep['source'],input_hashes=prep.get('input_hashes',[]),prefix_key='preparation:'+digest(request))
        profile=preparation_profile_core(select_preparation_profile(self._configured_preparation_profile(host),assignment))
        assignment.update(host_execution_profile=profile,host_execution_profile_sha256=digest(profile))
        return assignment

    def _preparation_probe_targets(self):
        """Hosts whose current participation may resolve pending readiness.

        Busy hosts remain targets: a free execution slot is not required for a
        metadata probe. An optional paused, absent or stale host cannot turn
        established failures on participating hosts into a participation wait.
        """
        policy,error=self._host_admission_policy()
        targets=set()
        for host in self.distributed.get('preparation_profiles')or{}:
            admission=self._host_admission(host,policy,error)
            observed=self.hosts.get(host)
            if not admission['accepting']or admission['max_slots']==0:continue
            if observed is not None and observed.get('accepting_new_jobs')is False:continue
            if admission['optional'] and (observed is None or
                    not self._offer_fresh(host)):continue
            targets.add(host)
        return targets

    def _preparation_probe_hold(self, record):
        targets={host for host in self._preparation_probe_targets()
                 if self._host_supports_stage(host,record['request'])}
        if targets.intersection(record['decision']['eligible_host_ids']):return None
        if targets and not targets.issubset(record.get('failures',{})):return None
        reasons = deepcopy(record.get('failures', {}))
        request = record['request']
        policy, error = self._host_admission_policy()
        manager = self.pbs_manager
        for host in (self.distributed.get('preparation_profiles') or {}):
            if host in targets or host in reasons:
                continue
            admission = self._host_admission(host, policy, error)
            if not self._host_supports_stage(host, request):
                reasons[host] = 'STAGE_OR_PROVIDER_NOT_CONFIGURED'
            elif not admission['accepting'] or admission['max_slots'] == 0:
                reasons[host] = 'HOST_ADMISSION_HOLD'
            elif manager is not None and host in manager.configured:
                from .rpe_pbs import task_estimate
                if not (task_estimate(request, manager.settings, host) or 0) > 0:
                    reasons[host] = 'PBS_WALLTIME_ESTIMATE_UNAVAILABLE'
                    continue
                try:
                    self._pending_preparation_assignment(record, host)
                except (OSError, KeyError, ValueError, TypeError) as exc:
                    reasons[host] = str(exc)
                    continue
                # PBS may have to acquire a worker just to check this request.
                # Keep only this candidate's lane waiting for a bounded period.
                started = record.get('probe_started_epoch', 0)
                if 0 <= time.time() - started < manager.preparation_readiness_wait_seconds(host):
                    return None
                reasons[host] = 'PBS_PREPARATION_READINESS_WAIT_EXPIRED'
            else:
                reasons[host] = (((self.hosts.get(host, {}).get('resource_report') or {}).get('storage') or {}).get('reason')
                                 or 'HOST_HAS_NO_FRESH_ACCEPTING_OFFER')
        return dict(status='HOST_ELIGIBILITY_HOLD',
            reason='ALL_PARTICIPATING_PREPARATION_HOSTS_FAIL_OR_UNCHECKED'if targets else'NO_PARTICIPATING_PREPARATION_HOST',
            host_reasons=reasons,probe_host_ids=sorted(targets))

    def _pending_preparation_challenges(self, host, free_slots):
        if host not in self._preparation_probe_targets():return []
        candidates = []
        for job_id, record in self.preparation_pending.items():
            if not self._host_supports_stage(host,record['request']):continue
            row=self.store.row(job_id)
            if row is not None and row['state']not in('QUEUED','CANCELLED'):continue
            candidates.append((job_id, record))
        if not candidates:
            self.preparation_probe_offsets.pop(host, None)
            return []
        start = self.preparation_probe_offsets.get(host, 0) % len(candidates)
        # Fill the current offer. Even a busy host can check the next request
        # in advance. Advance past failures and rotate through later requests.
        target = max(1, free_slots)
        challenges = []
        for examined, (job_id, record) in enumerate(candidates[start:]+candidates[:start], 1):
            try:challenges.append(self._pending_preparation_assignment(record,host))
            except (OSError,KeyError,ValueError,TypeError)as exc:
                failure=str(exc)
                if record.setdefault('failures',{}).get(host)!=failure:
                    record['failures'][host]=failure;self._save_preparation_pending()
                hold=self._preparation_probe_hold(record)
                if hold is not None:self.preparation_holds[job_id]=hold
                continue
            if len(challenges) == target:
                break
        self.preparation_probe_offsets[host] = (start + examined) % len(candidates)
        return challenges

    def _refresh_pending_preparation(self, host, fingerprints, errors):
        from .rpe_q3_coordinator import digest
        from .rpe_preparation_worker import validate_preparation_fingerprint
        if host not in self._preparation_probe_targets():return
        for job_id,record in list(self.preparation_pending.items()):
            if not self._host_supports_stage(host,record['request']):continue
            old=self.store.row(job_id)
            if old is not None and old['state']not in('QUEUED','CANCELLED'):continue
            if job_id not in fingerprints and job_id not in errors:continue
            decision=deepcopy(record['decision']);changed=False
            try:
                assignment=self._pending_preparation_assignment(record,host)
                if job_id in errors:raise ValueError(errors[job_id])
                fingerprint=fingerprints[job_id]
                selected=validate_preparation_fingerprint(assignment,self._configured_preparation_profile(host),fingerprint)
                current=digest(fingerprint)
                changed=decision['host_fingerprints'].get(host)!=current
                if host not in decision['eligible_host_ids']:decision['eligible_host_ids'].append(host);changed=True
                decision['host_fingerprints'][host]=current;decision['execution_profiles'][host]=selected
                if host in record.setdefault('failures',{}):record['failures'].pop(host);changed=True
                record['reports'][host]=fingerprint
                decision['reasons']=[]
                record['decision']=decision
                if changed:self._save_preparation_pending()
                self.preparation_holds.pop(job_id,None)
                # A verified metadata report enables a normal durable reservation;
                # readiness itself consumes no attempt or scientific budget.
                if old is None or old['state']=='CANCELLED':
                    self.submit_preparation(record['request'],budget=record['budget'],approved=record['approved'],
                        contract=record['contract'],eligibility=decision)
                elif old['state']=='QUEUED':self.store.set_eligibility(job_id,decision)
                if changed:self.log(event='PREPARATION_HOST_READINESS',job=job_id,host_id=host,status='PASS',stage=record['request']['preparation']['stage'])
            except (OSError,KeyError,ValueError,TypeError)as exc:
                failure=str(exc);failures=record.setdefault('failures',{})
                changed=failures.get(host)!=failure;failures[host]=failure
                if host in decision['eligible_host_ids']:
                    decision['eligible_host_ids'].remove(host);decision['host_fingerprints'].pop(host,None);decision['execution_profiles'].pop(host,None);changed=True
                decision['reasons']=['NO_MATCHING_PREPARATION_STAGE_HOST']if not decision['eligible_host_ids']else[]
                record['decision']=decision
                if changed:
                    self._save_preparation_pending()
                    self.log(event='PREPARATION_HOST_READINESS',job=job_id,host_id=host,status='UNCHECKED',stage=record['request']['preparation']['stage'],reason=failure)
                if old is not None and old['state']=='QUEUED':self.store.set_eligibility(job_id,decision)
                hold=self._preparation_probe_hold(record)
                if hold is not None:self.preparation_holds[job_id]=hold
        self.state.cv.notify_all()

    def submit_preparation(self, request, *, budget='UNLIMITED', approved=False, contract=None,
                           eligibility=None, dependencies=(), expected_job_id=None):
        """Reserve one exact preparation, independent of candidate aliases/Q3 budget.

        Signal and compile callers may submit together; dependency-bearing work
        shares the queue and waits for accepted artifacts. This is an execution
        API, not an additional permission system: callers pass their existing
        stage contract's execution allowance and budget.
        """
        from . import rpe_q3_coordinator as Q
        from .rpe_preparation_queue import normalize_preparation_request
        request = normalize_preparation_request(request,source_manifest=self.source_manifest,dependencies=dependencies)
        preparation = request['preparation']
        optional = preparation.get('optional_price_validation') is True
        job_id = request['request_id']
        if expected_job_id is not None and job_id != expected_job_id:
            from .pr_candidate_completion import Held
            raise Held('PREPARATION_REQUEST_MISMATCH:'+expected_job_id+':'+job_id)
        method = 'PREPARATION:'+preparation['stage']
        task_contract = dict(evaluation=deepcopy(contract),dependencies=preparation['dependencies'])
        with self.state.cv:
            if self._user_stop_requested():
                self.preparation_holds[job_id]=dict(status='PREPARATION_HOLD',reason='USER_REQUESTED_STOP')
                return job_id
            if preparation.get('dry_check_only')is True:
                self.preparation_holds[job_id]=dict(status='PREPARATION_HOLD',reason='INSPECTION_ONLY_REQUEST_NOT_EXECUTABLE')
                self.log(event='PREPARATION_HOLD',job=job_id,stage=preparation['stage'],reason='INSPECTION_ONLY_REQUEST_NOT_EXECUTABLE')
                return job_id
            row = self.store.row(job_id)
            if row is not None and (row['request_sha256'] != Q.digest(request) or row['contract'] != task_contract):
                from .pr_candidate_completion import Held
                raise Held('PREPARATION_REQUEST_MISMATCH:'+job_id)
            if row is None or row['state'] == 'CANCELLED':
                if eligibility is None: eligibility = self._preparation_eligibility_for(request, job_id)
                if not self.store.eligible(request, eligibility):
                    if (approved is True and budget not in (0,) and self.distributed.get('preparation_profiles')
                            and not self.store.artificial(request) and job_id not in self.preparation_pending):
                        empty=dict(request_sha256=Q.digest(request),eligible_host_ids=[],host_fingerprints={},execution_profiles={},reasons=['PREPARATION_HOST_METADATA_PENDING'])
                        self.preparation_pending[job_id]=dict(request=request,budget=budget,approved=approved,contract=deepcopy(contract),decision=empty,reports={},failures={},probe_started_epoch=time.time())
                        self._save_preparation_pending()
                    self.eligibility_holds[job_id] = eligibility
                    self.preparation_holds[job_id] = dict(status='HOST_ELIGIBILITY_HOLD',reason='NO_MATCHING_PREPARATION_STAGE_HOST',host_eligibility=eligibility)
                    pending = self.preparation_pending.get(job_id)
                    hold = self._preparation_probe_hold(pending) if pending is not None else self.preparation_holds[job_id]
                    if hold is None:
                        self.log(event='PREPARATION_PROBE_PENDING',job=job_id,stage=preparation['stage'])
                    else:
                        self.log(event='PREPARATION_HOLD',job=job_id,stage=preparation['stage'],
                            reason=hold['reason'],host_reasons=hold.get('host_reasons', {}))
                    return job_id
                if not self.store.reserve(method, job_id, request, budget, approved, contract=task_contract,eligibility=eligibility):
                    stage = preparation['stage']
                    hold = dict(status='PREPARATION_BUDGET_OR_PERMISSION_HOLD',
                                reason='PREPARATION_STAGE_BUDGET_EXHAUSTED' if approved else 'PREPARATION_STAGE_NOT_APPROVED',
                                stage=stage, budget=budget, reserved=self.store.preparation_reservations().get(stage, 0))
                    if self.preparation_holds.get(job_id) != hold:
                        self.log(event='PREPARATION_HOLD', job=job_id, **hold)
                    self.preparation_holds[job_id] = hold
                    return job_id
                row = self.store.row(job_id)
                self.preparation_holds.pop(job_id, None)
                self.eligibility_holds.pop(job_id, None)
                self.state.metrics['preparation_submitted'] += 1
                self.for_method(method).log(event='ENQUEUE',job=job_id,configuration=job_id,stage=preparation['stage'],demand=not optional,
                    optional_price_validation=optional)
            if row['state'] != 'ACCEPTED': self._restore_task(row, demand=not optional)
            self.state.cv.notify_all()
        return job_id

    def preparation_result(self, job_id, *, wait=False, adopt=True):
        row = self.store.row(job_id)
        if wait and row is None and job_id in self.preparation_pending:
            self.log(event='DEPENDENCY_WAIT_START',job=job_id)
            try:
                with self.state.cv:
                    while row is None and not self.state.closed:
                        if self._user_stop_requested() or self.state.failure is not None:
                            return dict(status='PREPARATION_STOPPED_BEFORE_RESERVATION',
                                        job_id=job_id,calls=0,preparation_calls=0)
                        pending=self.preparation_pending[job_id]
                        if self._preparation_probe_hold(pending)is not None:break
                        if pending['decision']['eligible_host_ids'] and job_id in self.preparation_holds and self.preparation_holds[job_id].get('status')=='PREPARATION_BUDGET_OR_PERMISSION_HOLD':break
                        self.state.cv.wait(.1);row=self.store.row(job_id)
            finally:self.log(event='DEPENDENCY_WAIT_END',job=job_id)
        if row is None:
            if job_id in self.preparation_pending:
                pending=self.preparation_pending[job_id]
                if self._preparation_probe_hold(pending)is None and self.preparation_holds.get(job_id,{}).get('status')!='PREPARATION_BUDGET_OR_PERMISSION_HOLD':return None
            hold = self.preparation_holds.get(job_id)
            if job_id in self.preparation_pending and(hold is None or hold.get('status')=='HOST_ELIGIBILITY_HOLD'):
                hold=self._preparation_probe_hold(self.preparation_pending[job_id])
            return None if hold is None else dict(hold,job_id=job_id,calls=0,preparation_calls=0)
        if self._preparation_request(row['request']) is None: raise ValueError('Not a preparation task')
        task = self.state.tasks.get(self.run_dir/'jobs'/row['method']/job_id)
        if wait and row['state'] != 'ACCEPTED' and task is not None and not task.event.is_set():
            self.for_method(row['method']).log(event='DEPENDENCY_WAIT_START',job=job_id)
            try:
                while not task.event.wait(.1):
                    for dependency in row['request']['preparation'].get('dependencies', []):
                        result = self.preparation_result(dependency,adopt=False)
                        if result is not None and result.get('status') != 'PREPARATION_COMPLETE':
                            return dict(status='PREPARATION_DEPENDENCY_HOLD',reason='PREDECESSOR_NOT_COMPLETE',dependency=dependency,predecessor=result,job_id=job_id,calls=0,preparation_calls=0)
            finally: self.for_method(row['method']).log(event='DEPENDENCY_WAIT_END',job=job_id)
            row = self.store.row(job_id)
        if row['state'] == 'ACCEPTED':
            job = self.run_dir/'jobs'/row['method']/job_id
            outcome = json.loads((job/'OUTCOME.json').read_text())
            artifacts = job/'artifacts'
            manifest = outcome.get('artifacts', {})
            if not adopt and self.artifact_cache.missing(manifest.values()):
                return dict(status='PREPARATION_RESULT_ARTIFACTS_MISSING', job_id=job_id,
                            calls=0, preparation_calls=1)
            self.artifact_cache.materialize(manifest, artifacts, immutable=True)
            if adopt: self.adopt_preparation(job_id)
            return dict(outcome,job_id=job_id,job=str(job),artifact_directory=str(artifacts),calls=0,preparation_calls=1)
        if row['state'] == 'UNKNOWN' or (task and task.error):
            return dict(status='UNKNOWN',reason=task.error if task else 'PREPARATION_STATE_UNKNOWN',job_id=job_id,calls=0,preparation_calls=1)
        return None

    def adopt_preparation(self, job_id):
        row = self.store.row(job_id)
        if row is None or row['state'] != 'ACCEPTED' or self._preparation_request(row['request']) is None:
            raise ValueError('Only a received preparation result can be adopted')
        if self.store.adopt(job_id):
            job = self.run_dir/'jobs'/row['method']/job_id
            outcome = json.loads((job/'OUTCOME.json').read_text())
            self.for_method(row['method']).log(event='ADOPT',job=job_id,configuration=job_id,
                status=outcome['status'],stage=row['request']['preparation']['stage'])
            self.state.metrics['preparation_adopted'] += 1

    def await_preparation(self, job_id, *, adopt=True):
        return self.preparation_result(job_id,wait=True,adopt=adopt)

    def configure_preparation_graph(self, settings):
        """Submit an explicitly supplied graph; candidate selection stays outside."""
        from . import rpe_q3_coordinator as Q
        frozen = self.run_dir/'distributed'/'PREPARATION_GRAPH.json'
        if frozen.exists() and json.loads(frozen.read_text()) != settings:
            raise ValueError('Preparation graph changed on resume; use a new run')
        if not frozen.exists(): Q.atomic_json(frozen, settings)
        jobs, pending = {}, list(settings.get('tasks', []))
        while pending:
            ready = [item for item in pending if all(name in jobs for name in item.get('dependencies', []))]
            if not ready: raise ValueError('Preparation graph has missing or cyclic dependencies')
            for item in ready:
                request = deepcopy(item['request'])
                prep = request['preparation']
                if prep.get('dependency_inputs'):
                    prep['dependency_inputs'] = {jobs.get(name,name):paths for name,paths in prep['dependency_inputs'].items()}
                jobs[item['name']] = self.submit_preparation(request,budget=item.get('budget',0),approved=item.get('approved') is True,
                    contract=item.get('contract'),eligibility=item.get('eligibility'),dependencies=[jobs[name] for name in item.get('dependencies', [])])
                pending.remove(item)
        self.preparation_dependencies.update({configuration:[jobs[name] for name in names]
            for configuration,names in settings.get('candidate_dependencies', {}).items()})
        Q.atomic_json(self.run_dir/'distributed'/'PREPARATION_BINDINGS.json',dict(tasks=jobs,candidate_dependencies=self.preparation_dependencies))
        return jobs

    def _candidate_preparation_ready(self, proposal):
        for job_id in self.preparation_dependencies.get(proposal['configuration_id'], []):
            result = self.preparation_result(job_id,adopt=False)
            if result is None or result.get('status') != 'PREPARATION_COMPLETE': return False
        return True

    def _wait_candidate_preparation(self, proposal):
        for job_id in self.preparation_dependencies.get(proposal['configuration_id'], []):
            result = self.preparation_result(job_id,wait=True)
            if result is None or result.get('status') != 'PREPARATION_COMPLETE':
                return dict(status='HOST_ELIGIBILITY_HOLD',reason='CANDIDATE_PREPARATION_NOT_COMPLETE',preparation=result,host_eligibility=dict(preparation_task=job_id,result=result),job_id=job_id,calls=0,MSE_lower=None,MSE_upper=None,job=None)
        return None

    def _complete_preparation(self, row, value, transfer, receipt, outcome, task, started):
        from .rpe_completion_acceptance import MissingPreparationArtifacts, validate_preparation_result
        try:
            validate_preparation_result(row, receipt, outcome, self.artifact_cache)
        except MissingPreparationArtifacts:
            raise  # The original attempt can still upload the declared blobs.
        except ValueError as error:
            self._hold_result_transfer(row, task, str(error))
        accepted, duplicate = self.store.complete(row['host_id'],row['job_id'],row['attempt_id'],value['request_sha256'],value['files'])
        self._publish_transfer(accepted)
        if not duplicate:
            self.for_method(row['method']).log(event='TRANSFER_COMPLETE',job=row['job_id'],host_id=row['host_id'],stage=row['request']['preparation']['stage'],transfer_seconds=time.monotonic()-started)
        if task:
            task.error = None
            if task.status == 'RUNNING': self._finished(task,receipt.get('returncode'),row['assigned_epoch'])
            else: task.event.set()
        self.state.cv.notify_all()
        self._timing_queue()
        return dict(ok=True,duplicate=duplicate)

    def _held_answer(self, proposal):
        job = self.job_dir(proposal)
        return dict(status='HOST_ELIGIBILITY_HOLD', reason='NO_MATCHING_ELIGIBLE_HOST',
                    host_eligibility=self.eligibility_holds.get(job.name), source='HOST_READINESS',
                    calls=0, MSE_lower=None, MSE_upper=None, job=None)

    def _restore_task(self, row, proposal=None, demand=False):
        request = row['request']; method = row['method']
        optional = bool((self._preparation_request(request) or {}).get('optional_price_validation'))
        if optional: demand = False
        job = self.run_dir/'jobs'/method/row['job_id']
        old = self.state.tasks.get(job)
        if old and old.status != 'CANCELLED':
            old.demand |= demand
            return old
        job.mkdir(parents=True, exist_ok=True)
        if not (job/'REQUEST.json').exists(): _write(job/'REQUEST.json', request)
        if proposal is None and self._preparation_request(request):
            proposal = dict(configuration_id=row['job_id'], origin=dict(kind='PREPARATION'), cost=0)
        if proposal is None:
            proposal = dict(configuration_id=request['configuration_id'], vector=O.interleave(request['N_X'], request['N_Y']),
                            cost=O.frac(request['cost']), origin=request['origin'])
        running = row['state'] != 'QUEUED'
        task = SimpleNamespace(view=self.for_method(method), job=job, proposal=deepcopy(proposal), rule=request['route'],
            owner=(method, proposal['configuration_id']), demand=demand, status='RUNNING' if running else 'QUEUED',
            event=threading.Event(), enqueue=time.monotonic()-max(0, time.time()-row['enqueued_epoch']),
            error=None, started=running, speculative=not demand, optional_price_validation=optional)
        decision = row.get('eligibility')
        task.supply_hosts = None if decision is None else frozenset(decision['eligible_host_ids'])
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

    def replay_prior_q3(self):
        """Resolve exact allocations authorized in the successor's fixed round."""
        from .rpe_q3_coordinator import digest

        checked = []
        for hold in self.prior_q3_holds:
            if hold.get('replay_exact') is not True:
                continue
            method, job_id = hold['method'], hold['job_id']
            parent_job = Path(hold['parent_directory'])/'jobs'/method/job_id
            request = O.load_json(parent_job/'REQUEST.json')
            view = self.for_method(method)
            cid = request['configuration_id']
            proposal = dict(configuration_id=cid, vector=O.interleave(request['N_X'], request['N_Y']),
                            origin=request['origin'], cost=O.frac(request['cost']))
            row = self.store.row(job_id)
            if (digest(request) != hold['request_sha256']
                    or request['optimizer_source'] != self.computation_source
                    or view.job_dir(proposal).name != job_id or not view.same_route(parent_job, cid)
                    or view.prior_job(proposal, exclude=(parent_job.resolve(), view.job_dir(proposal).resolve())) is not None
                    or (row is not None and row['request'] != request)
                    or (parent_job/'OUTCOME.json').exists()):
                raise ValueError('Prior Q3 replay binding or ownership changed: '+job_id)
            checked.append((view, proposal, parent_job, request))

        # Dispatch the whole independent set before waiting for any result.
        for view, proposal, parent_job, request in checked:
            view._enqueue(proposal, demand=True, replay=(parent_job, request))
        for view, proposal, parent_job, request in checked:
            row = self.store.row(request['request_id'])
            if row is None or row['state'] == 'CANCELLED':
                if self._user_stop_requested():
                    return False
                raise RuntimeError('No eligible allocation for exact Q3 replay: '+parent_job.name)
            answer = view.answer(proposal, allow_new=False)
            if not answer or answer['status'] not in (O.PASS, O.FAIL, 'INCONCLUSIVE'):
                raise RuntimeError('Exact Q3 replay requires reconciliation: '+parent_job.name)
            view.log(event='PRIOR_Q3_REPLAY_VALIDATED', job=parent_job.name, parent_job=str(parent_job))
        self.prior_q3_holds = [hold for hold in self.prior_q3_holds if hold.get('replay_exact') is not True]
        return True

    def _enqueue(self, proposal, demand=False, *, replay=None, defer_readiness=False):
        from .rpe_q3_coordinator import prefix_key
        if not self._candidate_preparation_ready(proposal):
            return False
        job = self.job_dir(proposal)
        s = self.state

        def current_job():
            if self._user_stop_requested():
                return True, False, None
            if s.failure:
                raise RuntimeError('Executor persistence failure: ' + s.failure)
            task = s.tasks.get(job)
            if task and task.status != 'CANCELLED':
                task.demand |= demand
                s.cv.notify_all()
                return True, task, None
            row = self.store.row(job.name)
            if row and row['state'] == 'ACCEPTED':
                return True, None, row
            if row and row['state'] not in ('QUEUED', 'CANCELLED'):
                return True, self._restore_task(row, proposal, demand), row
            if proposal['configuration_id'] in self.execution_holds.get(self.method, {}):
                return True, None, row
            if s.closed:
                raise RuntimeError('Executor is closed')
            return False, None, row

        with s.cv:
            resolved, task, row = current_job()
            if resolved:
                return task
            if not row or row['state'] == 'CANCELLED':
                excluded = (replay[0].resolve(),) if replay else ()
                if (job/'OUTCOME.json').exists() or (job/'LAUNCH.json').exists() or self.prior_job(proposal, exclude=excluded) is not None:
                    return None
                rule = replay[1]['route'] if replay else self.route(proposal['configuration_id'])
                if rule is None:
                    return None
                if not s.approved or self._budget_reserved():
                    return False
            request = row['request'] if row else replay[1] if replay else self.request(proposal, job, rule)

        options = dict(blocking=False, prefetch=True) if defer_readiness else {}
        eligibility = self._eligibility_for(request, job.name, previous=row and row['eligibility'], **options)

        with s.cv:
            resolved, task, row = current_job()
            if resolved:
                return task
            if row and row['request'] != request:
                if row['state'] == 'QUEUED':
                    return self._restore_task(row, proposal, demand)
                raise ValueError('Scientific request changed during host readiness')
            if defer_readiness and eligibility and not eligibility['eligible_host_ids']:
                key = (job.name, eligibility['request_sha256'])
                if key in self.live_readiness_inflight:
                    return _CHECK_PENDING
                if eligibility['reasons'] in (['NO_REQUEST_SPECIFIC_HOST_EVIDENCE'],
                                              ['HOST_ELIGIBILITY_CHANGED_REQUEST_SHA256']):
                    cached = self.eligibility_policy_cache.get('live_checks', {}).get(key)
                    if cached is None:
                        return _CHECK_PENDING
                    # The asynchronous check may have finished outside the lock.
                    eligibility = cached[2]
            if not self.store.eligible(request, eligibility):
                if row and row['state'] == 'QUEUED':
                    self.store.set_eligibility(job.name, eligibility)
                self.eligibility_holds[job.name] = eligibility
                self.log(event='HOST_ELIGIBILITY_HOLD', job=job.name, eligibility=eligibility)
                return False
            self.eligibility_holds.pop(job.name, None)
            if row and row['state'] == 'QUEUED':
                self.store.set_eligibility(job.name, eligibility)
                return self._restore_task(self.store.row(job.name), proposal, demand)
            if not self.store.reserve(self.method, job.name, request, s.budgets.get(self.method), s.approved,
                                      contract=dict(evaluation=self.contract, prefix_key=prefix_key(request)),
                                      eligibility=eligibility):
                return False
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
        preparation = self._preparation_request(row['request'])
        value = dict(job_id=row['job_id'], method=row['method'], attempt_id=row['attempt_id'], request=row['request'],
            request_sha256=row['request_sha256'], source_manifest=preparation.get('source', self.source_manifest) if preparation else self.source_manifest,
            input_hashes=preparation.get('input_hashes', []) if preparation else input_hashes(self.computation_root, row['request'], self.input_hash_cache),
            prefix_key=('preparation:'+digest(row['request'])) if preparation else prefix_key(row['request']))
        if preparation:
            value['artifact_manifest'] = self._preparation_inputs(row['request'])
        # Eligibility can change only before claim. Assigned rows retain the
        # selected profile for replay/recovery even after policy replacement.
        fixed = self.store.row(row['job_id'])
        host = host_id or row.get('host_id')
        profile = ((fixed.get('eligibility') or {}).get('execution_profiles') or {}).get(host)
        if profile is not None:
            if preparation:
                from .rpe_preparation_worker import preparation_profile_core
                value['host_execution_profile'] = preparation_profile_core(profile)
            else: value['host_execution_profile'] = profile_core(profile)
            value['host_execution_profile_sha256'] = digest(value['host_execution_profile'])
        if self.distributed.get('required_release'):
            value['required_release']=deepcopy(self.distributed['required_release'])
            local=self.distributed.get('local_worker')or{}
            if host==local.get('host_id')and self.distributed.get('local_deployment_layout'):
                from .rpe_release_deployment import read,validate_assignment_release
                validate_assignment_release(self.release_manifest,
                    read(self.distributed['local_deployment_layout']),value,local)
        return value

    def _timing_queue(self):
        """Observe queue causes under the existing claim lock; never reorder."""
        s=self.state;reasons={};required=self.distributed.get('required_hosts',[])
        admission=None;admission_error=None
        # Use the newer optional-host policy when supplied by the integration;
        # do not copy or change its dispatch/admission implementation here.
        if hasattr(self,'_host_admission_policy'):
            admission,admission_error=self._host_admission_policy()
            required=[h for h in required if not self._host_admission(h,admission,admission_error)['optional']]
        # Compact operational columns only: do not deserialize every saved REQUEST.
        with self.store._connection() as db:
            snapshot={r['job_id']:dict(r)for r in db.execute(
                "SELECT job_id,host_id,state,eligibility FROM jobs WHERE state NOT IN ('CANCELLED','ACCEPTED')")}
        owned_attempts = self._owned_attempts()
        owned=Counter(r['host_id'] for r in owned_attempts)
        for task in s.tasks.values():
            if task.status!='QUEUED':continue
            row=snapshot.get(task.job.name)
            if row is None:continue
            decision=json.loads(row['eligibility'])
            allowed=decision.get('eligible_host_ids',[]) if decision else list(self.hosts)
            if admission is not None:
                allowed=[h for h in allowed if self._host_admission(h,admission,admission_error)['accepting']
                    and self._host_admission(h,admission,admission_error)['max_slots']!=0
                    and self.hosts.get(h,{}).get('accepting_new_jobs') is not False
                    and (not self._host_admission(h,admission,admission_error)['optional'] or
                         self._offer_fresh(h))]
            def capacity(h):
                value=self.hosts[h].get('capacity')
                maximum=None if admission is None else self._host_admission(h,admission,admission_error)['max_slots']
                return min(value,maximum)if value is not None and maximum is not None else value
            if self._preparation_request(self.store.row(task.job.name)['request']) and not self._task_ready(task):reason='PREPARATION_DEPENDENCY_OR_ARTIFACT_WAIT'
            elif admission_error or(admission is not None and not allowed):reason='HOST_ADMISSION_HOLD'
            elif any(h not in self.hosts for h in required):reason='REQUIRED_HOST_PARTICIPATION_WAIT'
            elif s.active>=self.capacity:reason='ALL_ELIGIBLE_SLOTS_BUSY'
            elif allowed and all(h in self.hosts and capacity(h) is not None and
                    owned[h]>=capacity(h) for h in allowed):reason='ALL_ELIGIBLE_SLOTS_BUSY'
            elif not allowed or any(h not in self.hosts for h in allowed):reason='ELIGIBLE_HOST_AVAILABILITY_UNKNOWN'
            else:reason='DISPATCH_READINESS_OR_POLL_WAIT'
            reasons[task.job.name]=reason
        payload=dict(reasons=reasons,active=s.active,capacity=self.capacity,
            hosts={h:dict(capacity=v.get('capacity'),slots=v['slots'])for h,v in self.hosts.items()})
        if self.distributed.get('live_ready_refill'):
            ready = self._ready_queue()
            reserved = self.store.preparation_reservations()
            budget_blocked = {}
            checking = False
            for job, record in self.preparation_pending.items():
                if job in snapshot or self.store.row(job) is not None:
                    continue
                stage = record['request']['preparation']['stage']
                budget = record.get('budget')
                if budget not in (None, 'UNLIMITED') and reserved.get(stage, 0) >= budget:
                    budget_blocked[stage] = budget_blocked.get(stage, 0) + 1
                else:
                    checking = True
            if budget_blocked:
                payload['preparation_budget_blocked'] = {
                    stage: dict(waiting=count, reserved=reserved.get(stage, 0))
                    for stage, count in budget_blocked.items()}
            waiting_preparation = (any(task.rule['kind'] == 'preparation' and task.status in ('QUEUED', 'RUNNING')
                                       for task in s.tasks.values()) or checking)
            work_conserving = self.distributed.get('work_conserving_search')
            if work_conserving:
                s.ready_cpu_supply = self._ready_cpu_supply(ready, owned_attempts, admission, admission_error)
            for host, observation in payload['hosts'].items():
                cpu = getattr(s, 'ready_cpu_supply', {}).get(host)
                if cpu is not None:
                    observation['cpu_supply'] = cpu
                free = self._offer_free(host, admission, admission_error)
                observation.update(owned=owned[host], fresh_free_slots=free)
                storage = (self.hosts[host].get('resource_report') or {}).get('storage')
                if storage is not None:
                    observation['storage'] = storage
                    if not storage['accepting_new_jobs']:
                        observation['idle_reason'] = storage['reason']
                if free:
                    compatible = self._match_ready_jobs(ready, {host: free})
                    if work_conserving:
                        observation['idle_reason'] = ('ASSIGNMENT_WAIT' if compatible else
                            'ELIGIBLE_HOST_SHORTAGE' if ready else
                            'PREPARATION_WAIT' if waiting_preparation else
                            'PREPARATION_BUDGET_HOLD' if budget_blocked else 'CANDIDATE_SHORTAGE')
                    else:
                        observation['idle_reason'] = ('NO_READY_TASK' if not ready else
                            'DISPATCH_READINESS_OR_POLL_WAIT' if compatible else 'NO_ELIGIBLE_READY_TASK')
        marker=json.dumps(payload,sort_keys=True)
        if marker!=getattr(s,'last_timing_queue',None):
            s.last_timing_queue=marker;self.log(event='QUEUE_WAIT_STATE',**payload)

    def rpc(self, path, value):
        from . import rpe_q3_coordinator as Q
        s = self.state
        host = value['host_id']
        if not isinstance(host, str) or not host: raise ValueError('Nonempty host_id required')
        # The content-addressed cache installs only complete hash-checked
        # files. Hashing and encoding large bundles need no queue lock.
        if path.startswith('/v1/artifacts/'):
            if path == '/v1/artifacts/missing': return dict(missing=self.artifact_cache.missing(value.get('hashes', [])))
            if path == '/v1/artifacts/get': return dict(blobs=self.artifact_cache.get(value.get('hashes', [])))
            if path == '/v1/artifacts/put': return dict(ok=True, **self.artifact_cache.put(value.get('blobs', {})))
            raise ValueError('Unknown artifact RPC endpoint')
        with s.cv:
            if path == '/v1/poll':
                if value.get('source_sha') != self.optimizer_source: raise ValueError('Worker fixed SOURCE differs')
                stopping = self._user_stop_requested()
                slots = value['slots']
                if type(slots) is not int or slots < 0: raise ValueError('Nonnegative free slots required')
                fingerprints = value.get('host_fingerprints') or {}
                if not isinstance(fingerprints, dict): raise ValueError('Host fingerprints must map job IDs to current facts')
                admission_policy, admission_error = self._host_admission_policy()
                admission = self._host_admission(host, admission_policy, admission_error)
                new_slots = self._host_new_slots(host, slots, admission)
                host_capacity = value.get('capacity')
                if host_capacity is not None and (type(host_capacity) is not int or host_capacity < 0):
                    raise ValueError('Nonnegative host capacity required')
                if host not in self.hosts:
                    self.log(event='HOST_FIRST_POLL',host_id=host,capacity=value.get('capacity'))
                self.hosts[host] = dict(epoch=time.time(), slots=new_slots, reported_slots=slots,
                                       capacity=host_capacity, host_admission=admission,
                                       accepting_new_jobs=value.get('accepting_new_jobs', True),
                                       observed_monotonic=time.monotonic(),
                                       owned_at_offer=[row['attempt_id'] for row in self._owned_attempts() if row['host_id'] == host],
                                       resource_report=value.get('resource_report'),
                                       cached_prefixes=value.get('cached_prefixes', []),
                                       host_fingerprints=fingerprints)
                self._refresh_pending_preparation(host,fingerprints,value.get('readiness_errors')or{})
                self._refresh_capacity(admission_policy, admission_error)
                s.cv.notify_all()
                new_slots = self._offer_free(host, admission_policy, admission_error)
                pending_challenges = self._pending_preparation_challenges(host, new_slots)
                def reply(**fields):
                    existing=fields.get('eligibility_challenges',[])
                    fields['eligibility_challenges']=list({item['job_id']:item for item in existing+pending_challenges}.values())
                    return dict(fields, capacity_feedback=self._capacity_feedback(host))
                # A lost response is reconciled with its original owner even
                # if current readiness changed or another host is offline.
                with self.store._connection() as database:
                    replay_ids = [row['job_id'] for row in database.execute(
                        "SELECT job_id FROM jobs WHERE host_id=? AND state='ASSIGNED' ORDER BY enqueued_epoch,job_id LIMIT ?",
                        (host, slots))]
                assignments = [self._assignment(self.store.row(job)) for job in replay_ids]
                if stopping:
                    return reply(assignments=assignments, stop=False,
                                 host_admission=dict(accepting=False,reason='USER_REQUESTED_STOP'))
                self._timing_queue()
                required = [name for name in self.distributed.get('required_hosts', [])
                            if not self._host_admission(name, admission_policy, admission_error)['optional']]
                if any(name not in self.hosts for name in required):
                    return reply(assignments=assignments, stop=False, host_admission=admission,
                                waiting_for_hosts=[name for name in required if name not in self.hosts])
                if not new_slots:
                    return reply(assignments=assignments, stop=False, host_admission=admission,
                                eligibility_challenges=[])
                queue = self._ready_queue()
                if s.failure or s.closed: return reply(assignments=assignments, stop=False, host_admission=admission)
                eligible, challenges = [], []
                skipped = False
                for task in queue:
                    row = self.store.row(task.job.name)
                    if not self._host_supports_stage(host, row['request']):
                        skipped = True
                        continue
                    if not self._fits_host_walltime(host, row['request']):
                        skipped = True
                        continue
                    decision = self._eligibility_for(row['request'], row['job_id'], blocking=False,
                                                     previous=row['eligibility'], host=host)
                    if not self.store.set_eligibility(row['job_id'], decision):
                        self.eligibility_holds[row['job_id']] = decision
                        task.error = 'HOST_ELIGIBILITY_HOLD'
                        ConcurrentExecutor._cancel(self, task)
                        skipped = True
                        continue
                    self.eligibility_holds.pop(row['job_id'], None)
                    task.supply_hosts = None if decision is None else frozenset(decision['eligible_host_ids'])
                    if decision is not None and host not in decision['eligible_host_ids']:
                        skipped = True
                        continue
                    if not self.store.eligible(row['request'], decision, host, fingerprints, row['job_id']):
                        # Metadata challenge only: no attempt ID, reservation
                        # transition, LAUNCH or scientific execution occurs.
                        if row['job_id'] not in self.preparation_pending:
                            challenge = self._assignment(row, host_id=host)
                            challenge.pop('attempt_id', None)
                            challenges.append(challenge)
                        skipped = True
                        continue
                    conditional = task.proposal.get('origin', {}).get('kind') == 'CHAIN_LOOKAHEAD_IF_PREVIOUS_PASSES'
                    if skipped and conditional and not task.demand:
                        continue
                    eligible.append(task)
                if value.get('observe_only') is True:
                    self._timing_queue()
                    return reply(assignments=[], stop=False, eligibility_challenges=challenges, host_admission=admission)
                if self.pbs_manager is not None:
                    starting = self.pbs_manager.startup_jobs(except_host=host)
                    eligible = [task for task in eligible if task.job.name not in starting]
                    starting = self.pbs_manager.startup_jobs()
                    queue = [task for task in queue if task.job.name not in starting]
                reserved_for_preferred = self._preferred_ready_jobs(host, queue, admission_policy, admission_error)
                if reserved_for_preferred:
                    skipped_preferred = False
                    remaining_tasks = []
                    for task in eligible:
                        if task.job.name in reserved_for_preferred:
                            skipped_preferred = True
                            continue
                        if (skipped_preferred and not task.demand and
                                task.proposal.get('origin', {}).get('kind') == 'CHAIN_LOOKAHEAD_IF_PREVIOUS_PASSES'):
                            continue
                        remaining_tasks.append(task)
                    eligible = remaining_tasks
                # Direct workers may poll before the other hosts are observed.
                # One bounded discovery interval allows simultaneous free
                # offers to arrive; it never waits for a busy host to finish.
                rank = self._host_rank(host)
                if s.allocation_auto and rank > 0 and eligible:
                    higher = [h for group in self.allocation_config['priority_groups'][:rank] for h in group]
                    unknown = [h for h in higher if h not in self.hosts]
                    relevant = any((self.store.row(task.job.name)['eligibility'] is None or
                        h in self.store.row(task.job.name)['eligibility']['eligible_host_ids'])
                        for h in unknown for task in eligible)
                    # Start once at this host's first eligible poll, rather
                    # than at controller startup (planning/placement can be
                    # slow). Any higher-group observation ends discovery:
                    # an absent third host cannot delay known surplus work.
                    now = time.monotonic()
                    since = s.allocation_discovery.setdefault(host, now)
                    if (relevant and not any(h in self.hosts for h in higher) and
                            now-since < self.allocation_config['discovery_grace_seconds']):
                        return reply(assignments=assignments, stop=False, eligibility_challenges=challenges,
                                     host_admission=admission, waiting_for_discovery=unknown)
                # Prefix locality is considered only among eligible hosts for
                # the same first dispatchable task, never an unsupported host.
                if eligible and not self._preparation_request(self.store.row(eligible[0].job.name)['request']) and time.monotonic()-eligible[0].enqueue < self.distributed.get('affinity_wait_seconds', .2):
                    first = self.store.row(eligible[0].job.name)
                    key = Q.prefix_key(first['request'])
                    if key not in value.get('cached_prefixes', []):
                        preferred = [h for h, info in self.hosts.items() if h != host and info['slots'] > 0
                            and self._host_rank(h) <= self._host_rank(host)
                            and self._host_new_slots(h, info.get('reported_slots', info['slots']),
                                self._host_admission(h, admission_policy, admission_error)) > 0
                            and time.time()-info['epoch'] < 2 and key in info['cached_prefixes']
                            and self._host_supports_stage(h, first['request'])
                            and self._fits_host_walltime(h, first['request'])
                            and self.store.eligible(first['request'], first['eligibility'], h,
                                                   info.get('host_fingerprints'), first['job_id'])]
                        if preferred: return reply(assignments=assignments, stop=False, eligibility_challenges=challenges, host_admission=admission)
                unresolved = len(self._owned_attempts())
                remaining = max(0, min(slots-len(assignments), new_slots,
                    self.capacity-max(s.active, unresolved)-len(reserved_for_preferred)))
                if host == 'local' and self.distributed.get('controller_queue_first'):
                    eligible.sort(key=lambda task: not (
                        task.rule['kind'] == 'preparation' and task.demand))
                resource_pool = ((value.get('resource_report') or {}).get('pbs') or {}).get('resource_pool')
                if resource_pool is not None and remaining:
                    from .rpe_pbs import GIB, resource_need
                    settings = self.distributed['pbs']
                    worker = next(w for w in settings['workers'] if w['host_id'] == host)
                    free_cpu = resource_pool['cpu_count']
                    free_memory = resource_pool['memory_bytes'] - GIB
                    for owned in self._owned_attempts():
                        if owned['host_id'] == host:
                            cpu, memory = resource_need(self.store.row(owned['job_id'])['request'],
                                                        worker['profile'], settings, host)
                            free_cpu -= cpu
                            free_memory -= memory
                for task in eligible:
                    if remaining <= 0: break
                    row = self.store.row(task.job.name)
                    if self._wait_for_preparation_preference(host, row, admission_policy, admission_error):
                        continue
                    if not self._local_calibration_pair_allows(host, row['request']):
                        continue
                    if resource_pool is not None:
                        try:
                            cpu, memory = resource_need(self.store.row(task.job.name)['request'],
                                                        worker['profile'], settings, host)
                        except (KeyError, ValueError):
                            continue
                        if cpu > free_cpu or memory > free_memory:
                            continue
                    rows = self.store.claim(host, 1, value.get('cached_prefixes', []), [task.job.name],
                                            host_fingerprints=fingerprints)
                    if not rows: continue
                    row = rows[0]
                    if resource_pool is not None:
                        free_cpu -= cpu
                        free_memory -= memory
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
                self._timing_queue()
                self._refresh_capacity(admission_policy, admission_error)
                return reply(assignments=assignments, stop=False, eligibility_challenges=challenges, host_admission=admission)
            row = self.store.row(value['job_id'])
            if row is None or row['host_id'] != host or row['attempt_id'] != value['attempt_id']:
                raise ValueError('Different execution host/attempt')
            job = self.run_dir/'jobs'/row['method']/row['job_id']
            task = s.tasks.get(job)
            if path == '/v1/transfer-metric':
                self.for_method(row['method']).log(event='TRANSFER_TIMING', job=row['job_id'], host_id=host,
                    attempt_id=value['attempt_id'], **value.get('timing', {}))
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
            self.for_method(row['method']).log(event='RESULT_RECEIVED',job=row['job_id'],host_id=host,attempt_id=row['attempt_id'])
            if value['request_sha256'] != row['request_sha256']: raise ValueError('Completion request hash differs')
            transfer = self.run_dir/'distributed'/'transfers'/row['job_id']/row['attempt_id']
            transfer.mkdir(parents=True, exist_ok=True)
            Q.install_files(transfer, value['files'])
            if not (transfer/'REQUEST.json').exists(): _write(transfer/'REQUEST.json', row['request'])
            receipt = json.loads((transfer/'PROCESS_RETURN.json').read_text())
            outcome = json.loads((transfer/'OUTCOME.json').read_text())
            preparation = self._preparation_request(row['request'])
            if preparation:
                return self._complete_preparation(row, value, transfer, receipt, outcome, task, started)
            if outcome['status'] in (O.PASS, O.FAIL, 'INCONCLUSIVE') and (receipt.get('returncode') != 0 or receipt.get('signal') is not None):
                self._hold_result_transfer(row, task, 'Successful outcome without a clean adapter exit')
            proposal = task.proposal if task else dict(configuration_id=row['request']['configuration_id'],
                vector=O.interleave(row['request']['N_X'], row['request']['N_Y']))
            owner = task.view if task and row['request']['route'].get('horizon_projection') else self.for_method(row['method'])
            validation = owner.ingest(transfer, proposal, calls=0)
            if outcome['status'] in (O.PASS, O.FAIL, 'INCONCLUSIVE') and validation['status'] == 'UNKNOWN':
                self._hold_result_transfer(row, task, 'Transferred outcome identity mismatch: '+str(validation.get('reason')))
            accepted, duplicate = self.store.complete(host, row['job_id'], row['attempt_id'], value['request_sha256'], value['files'])
            self._publish_transfer(accepted)
            if not duplicate:
                if self.distributed.get('work_conserving_search'):
                    self._observe_q3_completion(accepted, transfer, validation)
                self.for_method(row['method']).log(event='TRANSFER_COMPLETE', job=row['job_id'], host_id=host,
                    transfer_seconds=time.monotonic()-started, transfer_bytes=sum(len(x)*3//4-x.count('=') for x in value['files'].values()),
                    worker_transfer_seconds=value.get('transfer_seconds'))
            if task:
                task.error = None
                if task.status == 'RUNNING': self._finished(task, receipt.get('returncode'), row['assigned_epoch'])
                else: task.event.set()
            if accepted['adopted_epoch'] is None and receipt.get('returncode') == 0:
                self._note_scaled_completion(row['method'], job, row['request'].get('origin') or {})
            self._timing_queue()
            s.cv.notify_all()
            return dict(ok=True, duplicate=duplicate)

    def _hold_result_transfer(self, row, task, reason):
        # Keep the rejected transfer and its durable execution owner for review.
        if task and task.status == 'RUNNING':
            task.error = 'WORKER_RESULT_MISMATCH'
            self._finished(task, None, row['assigned_epoch'])
        self.for_method(row['method']).log(event='RESULT_TRANSFER_HOLD', job=row['job_id'],
            attempt_id=row['attempt_id'], host_id=row['host_id'], reason=reason)
        raise ValueError(reason)

    def _publish_transfer(self, row):
        from .rpe_q3_coordinator import publish_staged_files

        job = self.run_dir/'jobs'/row['method']/row['job_id']
        stage = self.run_dir/'distributed'/'transfers'/row['job_id']/row['attempt_id']
        if not stage.exists():
            if (job/'OUTCOME.json').exists(): return
            raise ValueError('Accepted completion is missing its transfer directory')
        publish_staged_files(stage, job)

    def recover_scaled_completions(self):
        with self.state.cv:
            if self.method in self.state.scaled_recovered:
                return
            for job_id in self.store.unadopted_results(self.method):
                self.state.scaled_completions[self.method][self.run_dir/'jobs'/self.method/job_id] = None
            self.state.scaled_recovered.add(self.method)

    def _adopt_answer(self, job, proposal, answer):
        current = self.store.row(job.name)
        if current and current['state'] == 'ACCEPTED':
            if not self.store.adopt(current['job_id']):
                with self.state.cv:
                    self.state.scaled_completions[self.method].pop(job, None)
                return
        super()._adopt_answer(job, proposal, answer)

    def answer(self, proposal, allow_new=True):
        blocked = self._wait_candidate_preparation(proposal)
        if blocked is not None: return blocked
        job = self.job_dir(proposal)
        row = self.store.row(job.name)
        if row and row['state'] not in ('ACCEPTED', 'CANCELLED'):
            with self.state.cv:
                row = self.store.row(job.name)
                if row and row['state'] not in ('ACCEPTED', 'CANCELLED'):
                    self._restore_task(row, proposal, demand=True)
        answer = super().answer(proposal, allow_new)
        if answer and answer.get('status') == 'RESELECT_READY':
            return answer  # a concurrent transfer is not adoption by this selector
        if answer and answer.get('status') == 'BUDGET_EXHAUSTED' and job.name in self.eligibility_holds:
            return self._held_answer(proposal)
        if answer:
            current = self.store.row(job.name)
            task_hold = answer.get('reason') in ('WORKER_CONTACT_LOST', 'WORKER_RESULT_MISMATCH')
            if (answer['status'] == 'UNKNOWN' and current
                    and not self.state.failure and current['attempt_id'] and current['host_id']
                    and (task_hold or
                         current['state'] == 'UNKNOWN' and current['detail'].get('automatic_retry') is False)):
                # The durable worker report identifies this unresolved attempt.
                # Retain its ownership and isolate proposals that depend on it.
                answer = dict(answer, failure_scope='CONFIGURATION',
                              reason=(answer['reason'] if task_hold
                                      else current['detail'].get('reason') or 'WORKER_STATE_UNKNOWN_RECONCILIATION_REQUIRED'))
        return answer

    def record_prior_result_reuse(self, job):
        job = Path(job)
        if job.parent.parent != self.run_dir/'jobs':
            return
        current = self.store.row(job.name)
        if current is not None and job.parent.name == current['method'] and current['state'] == 'ACCEPTED':
            with self.state.cv:
                reused = getattr(self.state, 'reused_jobs', set())
                if current['job_id'] not in reused:
                    reused.add(current['job_id']); self.state.reused_jobs = reused
                    self.log(event='RESULT_REUSE', job=current['job_id'], host_id=current['host_id'])

    def ingest(self, job, proposal, calls):
        answer = super().ingest(job, proposal, calls)
        if calls == 0:
            self.record_prior_result_reuse(job)
        return answer

    def metrics(self):
        # RPC updates hosts/tasks/preflight state under this same reentrant
        # condition. Early candidate completion may request metrics while an
        # optional host joins; observe one consistent controller snapshot.
        with self.state.cv:
            return self._metrics_locked()

    def _metrics_locked(self):
        result = super().metrics()
        rows = self.store.rows(); by_host = {}
        for row in rows:
            if row['host_id'] is None: continue
            counts = by_host.setdefault(row['host_id'], dict(assigned=0, launched=0, completed=0, adopted=0, reused=0))
            counts['assigned'] += 1
            counts['launched'] += row['started_epoch'] is not None
            counts['completed'] += row['state'] == 'ACCEPTED'
            counts['adopted'] += row['adopted_epoch'] is not None
        with self.lock:
            progress_lines=(self.run_dir/'PROGRESS.jsonl').read_text().splitlines()
        for line in progress_lines:
            event = json.loads(line)
            if event.get('event') == 'RESULT_REUSE' and event.get('host_id') in by_host:
                by_host[event['host_id']]['reused'] += 1
        admission_policy, admission_error = self._host_admission_policy()
        admissions = {host:self._host_admission(host, admission_policy, admission_error)
                      for host in sorted(set(self.hosts) | set(admission_policy['hosts']))}
        result.update(distributed=True, by_host=by_host, host_admission=admissions,
                      pbs=self.pbs_manager.metrics() if self.pbs_manager else None,
                      host_admission_error=admission_error,
                      allocation_settings=deepcopy(self.allocation_settings),
                      automatic_capacity=self.state.allocation_auto,
                      unresolved_owned_attempts=len(self._owned_attempts()),
                      unresolved_q3_attempts=[
                          {key: row[key] for key in ('method', 'job_id', 'attempt_id', 'host_id')}
                          for row in rows if not self._preparation_request(row['request'])
                          and (row['state'] == 'UNKNOWN' or
                               getattr(self.state.tasks.get(self.run_dir/'jobs'/row['method']/row['job_id']),
                                       'error', None) == 'WORKER_CONTACT_LOST')],
                      host_offers={host:dict(capacity=info.get('capacity'),
                          fresh_free_slots=self._offer_free(host, admission_policy, admission_error),
                          reported_free_slots=info['reported_slots'],
                          observed_age_seconds=time.monotonic()-info.get('observed_monotonic', time.monotonic()),
                          priority_group=self._host_rank(host), resource_report=info.get('resource_report'),
                          capacity_feedback=self._capacity_feedback(host))
                          for host, info in self.hosts.items()},
                      persistent_reserved={m:self.store.counts(m) for m in self.routes},
                      host_eligibility_holds=deepcopy(self.eligibility_holds),
                      preparation={stage:dict(reserved=self.store.counts('PREPARATION:'+stage),
                          launched=self.store.counts('PREPARATION:'+stage,started_only=True))for stage in ('signal','compile','audit')},
                      preparation_readiness_pending={job:dict(stage=value['request']['preparation']['stage'],eligible_host_ids=value['decision']['eligible_host_ids'],failures=value.get('failures',{}))for job,value in self.preparation_pending.items() if self.store.row(job) is None},
                      actual_OS_host_count='recorded by integration harness; host IDs alone do not establish separate machines')
        return result

    def finish(self):
        if self.method is not None: return super().finish()
        if self.state.driver.closed: return
        # Optional diagnostics never extend search shutdown. The durable queue
        # retains both queued requests and the original owner/attempt of work
        # already assigned. Workers supervise their children independently;
        # stopping a polling daemon below never signals scientific children.
        # A subsequent controller reconciles the same rows, without retry.
        with self.state.cv:
            for job, task in list(self.state.tasks.items()):
                if not getattr(task, 'optional_price_validation', False): continue
                if task.status == 'RUNNING': self.state.active -= 1
                del self.state.tasks[job]
                self.state.external.pop(job, None)
                self.state.pids.pop(job, None)
                self.log(event='OPTIONAL_VALIDATION_DETACHED',job=job.name,state=task.status,
                         owner_preserved=True,automatic_retry=False)
            self.state.cv.notify_all()
        super().finish()
        if self.live_readiness_pool:
            self.live_readiness_pool.shutdown(wait=False, cancel_futures=True)
        if self.pbs_manager:
            retain = (self.pbs_manager.retain_between_batches and not self.state.failure
                      and not self._user_stop_requested())
            self.pbs_manager.close(retain_allocation=retain)
        self.stop_controller_relay()
        try:
            self._finish_local_worker()
        finally:
            self._close_worker_rpc()

    def _close_worker_rpc(self):
        if self.rpc_server is not None:
            self.rpc_server.shutdown()
            self.rpc_server.server_close()
            self.rpc_thread.join()
            self.rpc_server = None

    def _local_attempts_settled(self, profile_path):
        try:
            profile = json.loads(profile_path.read_text())
            if any(row['host_id'] == profile['host_id'] for row in self._owned_attempts()):
                return False
            work = Path(profile['work_dir'])
            return work.is_dir() and all(
                json.loads(path.with_name('WORKER_STATE.json').read_text())['state'] == 'TRANSFERRED'
                for path in work.glob('*/*/ASSIGNMENT.json'))
        except (OSError, ValueError, KeyError):
            return False

    def _finish_local_worker(self):
        from . import rpe_q3_coordinator as Q

        profile_path = self.run_dir/'distributed'/'LOCAL_WORKER.json'
        if self.local_worker is None:
            if self._local_daemon_alive(self.local_worker_pid, profile_path):
                os.kill(self.local_worker_pid, signal.SIGTERM)
            return
        worker = self.local_worker
        worker.terminate()
        if self._local_attempts_settled(profile_path):
            # The last poll depends on this endpoint. Once receipts are
            # transferred, close intake before waiting for the daemon.
            self._close_worker_rpc()
        reason = 'GRACEFUL'
        try:
            worker.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if not self._local_attempts_settled(profile_path):
                raise
            # Only the owned polling Popen stops here, after both ledgers
            # confirm settlement. Scientific supervisors are detached.
            worker.kill()
            worker.wait(timeout=3)
            reason = 'IDLE_POLLER_STOPPED_AFTER_RPC_CLOSE'
        Q.atomic_json(self.run_dir/'distributed'/'LOCAL_WORKER_EXIT.json',
                      dict(pid=worker.pid, returncode=worker.returncode, reason=reason))


def optimize(root,problem,*,executor,methods=None,close_executor=True,**kwargs):
    """Parallel methods; each retains O.optimize/run_method's own adoption order."""
    methods=list(methods or problem['methods'])
    finalizers = {}
    try:
        if kwargs.get('context') is None:kwargs['context']=O.prepare(root,problem,methods=methods)
        with ThreadPoolExecutor(max_workers=max(1,len(methods)))as pool:
            ready_source = kwargs.get('ready_source')
            def report_failure(method,future):
                if future.cancelled():return
                error=future.exception()
                if error is None:
                    result=future.result().get(method,{})
                    if result.get('stop_reason')=='UNKNOWN_RESULT':
                        notify=getattr(executor.state,'failure_callback',None)
                        if notify is not None:notify(None)
                    return
                try:executor._stop_after_failure(error)
                finally:O.record_runtime_error(executor.run_dir,error,'METHOD_OPTIMIZE',method=method)
            def finished(method, future):
                try:
                    report_failure(method, future)
                finally:
                    with executor.state.cv:
                        executor.state.cv.notify_all()

            def submit(method):
                future = pool.submit(O.optimize, root, problem, executor=executor.for_method(method),
                                     methods=[method], scaled_finalizers=finalizers, **kwargs)
                future.add_done_callback(lambda done: finished(method, done))
                return future

            futures = {m: submit(m) for m in methods}
            if ready_source is None:
                out = {m: futures[m].result()[m] for m in methods}
            else:
                out = {}
                state = executor.state

                def idle_ready():
                    if not executor.has_spare_slot():
                        return []
                    return [m for m in out if m not in futures and ready_source.has_new(m)]

                while futures:
                    with state.cv:
                        state.cv.wait_for(lambda: any(f.done() for f in futures.values()) or idle_ready())
                        done = [m for m, f in futures.items() if f.done()]
                    for method in done:
                        out[method] = futures.pop(method).result()[method]
                    if not futures or any(r['stop_reason'] == 'UNKNOWN_RESULT' for r in out.values()):
                        # If one method failed, drain the other selector using the
                        # same failure callback and existing stop contract.
                        out.update((m, f.result()[m]) for m, f in futures.items())
                        break
                    with state.cv:
                        restart = idle_ready()
                    for method in restart:
                        futures[method] = submit(method)
    except BaseException:
        primary=getattr(executor.state,'primary_exception',None)
        if primary is not None:raise primary
        raise
    finally:
        if close_executor:executor.finish()
        else:
            for method in methods:executor.for_method(method).finish()
    for method, finalize in finalizers.items():
        finalize(out[method])
    metrics=executor.metrics()
    for m,result in out.items():
        counts=metrics['by_method'].get(m,{})
        result['adopted_new_calls']=result['new_calls']
        result['new_calls']=counts.get('launched',0)
        result['budget']['new_calls']=result['new_calls']
        result['executor_metrics']=dict(metrics,method_launched=counts.get('launched',0),method_adopted=counts.get('adopted',0))
    return out
