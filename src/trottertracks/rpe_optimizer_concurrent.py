"""Work-conserving execution around the unchanged per-configuration search rule.

The search controls adoption, including eligible out-of-order scaled results.
Finite budgets include reservations before
launch; running scientific workers are never cancelled to discard lookahead.
"""
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
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
                 profile_dp_residual=False,large_shot_bounds=False,**kwargs):
        super().__init__(*args,**kwargs)
        if dp_workers is not None and (type(dp_workers)is not int or dp_workers<1):
            raise ValueError('Positive integer dp_workers required')
        if prefix_cache_max_bytes is not None and (type(prefix_cache_max_bytes) is not int or prefix_cache_max_bytes < 1):
            raise ValueError('Positive integer prefix_cache_max_bytes required')
        if type(compact_q3_evidence) is not bool:
            raise ValueError('compact_q3_evidence must be a boolean')
        if type(large_shot_bounds) is not bool:
            raise ValueError('large_shot_bounds must be a boolean')
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
        self.large_shot_bounds=large_shot_bounds
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
        self.state.unmatched_scaled_completions = defaultdict(dict)
        self.state.scaled_recovered = set()
        self.state.reuse_lock = threading.Lock()
        self.state.reused_jobs = set()
        self.invocation=str(os.getpid())+'-'+str(time.time_ns())
        progress = self.run_dir/'PROGRESS.jsonl'
        self.progress_start = progress.stat().st_size if progress.exists() else 0
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
        if self.large_shot_bounds and req.get('origin', {}).get('kind') == 'RPE_SCALED_SHOTS_V1':
            execution['large_shot_bounds'] = True
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

    def _enqueue_demand(self, proposal):
        return self._enqueue(proposal, demand=True)

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
        if choice is not None and self._enqueue_demand(choice) is False:
            with self.state.cv:
                # Queued reservations can be reassigned to demand. Running jobs retain theirs.
                t = (next((t for t in reversed(speculative) if t.status == 'QUEUED'
                           and not t.demand and t.job != demand), None) if self._budget_reserved() else None)
                if t is not None:
                    self._cancel(t)
            if t is not None:
                self._enqueue_demand(choice)
        self._fill_standby()

    def _budget_reserved(self):
        cap = self.state.budgets.get(self.method)
        return type(cap) is int and self.state.existing.get(self.method, 0)+self.state.reserved[self.method] >= cap

    def _standby_capacity(self):
        return self.capacity

    def _standby_headroom(self):
        return 1

    def _standby_has_capacity(self):
        s = self.state
        return s.active + sum(t.status == 'QUEUED' and t.rule['kind'] != 'preparation'
                              for t in s.tasks.values()) < self._standby_capacity() + self._standby_headroom()

    def _standby_can_fill(self):
        return (getattr(self, 'standby', None) and not self.state.failure and not self.state.closed
                and self._standby_has_capacity())

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
        if (origin.get('kind') == RULE and job not in self.state.adopted_jobs
                and job not in self.state.unmatched_scaled_completions[method]):
            self.state.scaled_completions[method][job] = None

    def scaled_completion_pending(self):
        return bool(self.state.scaled_completions[self.method])

    def refresh_scaled_completions(self, configuration_ids):
        """Reoffer early results only when their candidate becomes visible."""
        with self.state.cv:
            deferred = self.state.unmatched_scaled_completions[self.method]
            for job, configuration in tuple(deferred.items()):
                if configuration in configuration_ids:
                    del deferred[job]
                    self.state.scaled_completions[self.method][job] = None

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
            if proposal is None:
                with self.state.cv:
                    if job not in self.state.adopted_jobs:
                        self.state.scaled_completions[self.method].pop(job, None)
                        self.state.unmatched_scaled_completions[self.method][job] = cid
                continue
            if (self.job_dir(proposal) != job or not self.same_route(job, cid)
                    or (not getattr(self, 'exploratory_cost', False) and
                        (request.get('cost') is None or O.frac(request['cost']) != proposal['cost']))):
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
            s.unmatched_scaled_completions[self.method].pop(job, None)
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
                if t.status=='QUEUED'and(self.method is None or t.view.method==self.method):
                    if (self.method is None and getattr(s, 'retain_queued_preparation', False)
                            and t.demand and t.rule['kind'] == 'preparation'):
                        continue
                    self._cancel(t)
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
            adopted=set()
            active=set();queued=set();waiting=Counter();busy=idle_ready=idle_empty=idle_dependency=available=0.
            first=last=None
            interval_capacity=self.capacity
            with self.lock, (self.run_dir/'PROGRESS.jsonl').open('rb') as progress:
                progress.seek(self.progress_start)
                for line in progress:
                    e=json.loads(line)
                    if e.get('invocation')!=self.invocation:continue
                    if first is None:first=last=e['monotonic']
                    if e['event']=='ADOPT':adopted.add(e['job'])
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
            elapsed=last-first if first is not None else 0.
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
        # Serialize retransmissions of the same attempt without blocking other hosts.
        self.transfer_locks = tuple(threading.Lock() for _ in range(32))
        self.eligibility_policy_cache = {}
        self.live_readiness_inflight = {}
        self.live_readiness_partial = {}
        self.live_readiness_planning = set()
        self.live_readiness_host_counts = Counter()
        self.live_readiness_host_pools = {}
        self.poll_probe_offsets = {}
        self.pbs_q3_check_queue = {}
        self.live_readiness_workers = max(1, len(kwargs.get('problem', args[4] if len(args)>4 else {}).get('methods', {})))
        self.controller_parallel = distributed.get('controller_parallel') is True
        self.live_readiness_max_workers = distributed.get('live_readiness_max_workers')
        if self.controller_parallel and self.live_readiness_max_workers is None:
            from .rpe_q3_resources import controller_cpu_count
            self.live_readiness_max_workers = controller_cpu_count()
        self.live_readiness_refill_seconds = distributed.get('live_readiness_refill_seconds', 5)
        if (self.live_readiness_max_workers is not None and
                (type(self.live_readiness_max_workers) is not int or self.live_readiness_max_workers < 1)):
            raise ValueError('Positive live readiness worker limit required')
        if self.live_readiness_refill_seconds <= 0:
            raise ValueError('Positive live readiness limits required')
        self.live_readiness_pool = (ThreadPoolExecutor(max_workers=self.live_readiness_max_workers or self.live_readiness_workers, thread_name_prefix='rpe-q3-readiness')
                                    if self.distributed.get('live_q3_readiness') is True else None)
        self.readiness_connections = None
        if self.live_readiness_pool is not None and self.controller_parallel:
            from .rpe_live_q3_readiness import ReadinessConnections
            self.readiness_connections = ReadinessConnections()
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
                        decision = (row['eligibility'] if row['detail'].get('prelaunch_recovery') is True
                                    else self._eligibility_for(row['request'], row['job_id'], blocking=False,
                                                               previous=row['eligibility']))
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
                if distributed.get('local_deployment_layout') and self.release_manifest:
                    profile['deployment'] = dict(manifest=distributed['release_manifest'],
                                                 layout=distributed['local_deployment_layout'])
                    continuation = distributed.get('local_pf_continuation')
                    if continuation:
                        release = continuation['references'][profile['host_id']]['id']
                        profile['task_deployments'] = {release: {key: continuation[key] for key in ('manifest','layout')}}
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
            self._close_live_readiness()
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
        relay = self.distributed.get('relay_script', str(self.root/'scripts/run_rpe_q3_relay.py'))
        command = [sys.executable, '-B', relay, '--config', str(path), '--run']
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

    def _host_new_slots(self, host, slots, admission, *, owned=None):
        if not admission['accepting']: return 0
        maximum = admission['max_slots']
        if maximum is None: return slots
        # UNKNOWN and completed-but-untransferred attempts still belong to this
        # host. A communication outage must not open another execution slot.
        owned = self._owned_attempts() if owned is None else owned
        count = sum(row['host_id'] == host for row in owned)
        return min(slots, max(0, maximum-count))

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

    def _offer_free(self, host, policy, error=None, *, owned=None):
        """Fresh unused capacity, never a lease on an assigned evaluation."""
        info = self.hosts.get(host)
        if not self._offer_fresh(host):
            return 0
        if info.get('accepting_new_jobs') is False: return 0
        admission = self._host_admission(host, policy, error)
        owned = self._owned_attempts() if owned is None else owned
        free = self._host_new_slots(host, info['reported_slots'], admission, owned=owned)
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
        owned = self._owned_attempts()
        queue = self._ready_queue()
        available = {host:self._offer_free(host, policy, error, owned=owned) for host in self.hosts}
        available = {host:min(free,len(queue)) for host,free in available.items() if free > 0}
        usable = len(self._match_ready_jobs(queue, available))
        capacity = max(1, len(owned)+usable)
        if capacity != self.state.allocation_capacity:
            self.state.allocation_capacity = capacity
            self.log(event='AUTO_CAPACITY', capacity=capacity, unresolved_owned=len(owned),
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
                       and (task.rule['kind'] != 'preparation' or self._task_ready(task))), key=priority)

    def _budget_reserved(self):
        cap = self.state.budgets.get(self.method)
        return type(cap) is int and self.store.counts(self.method) >= cap

    @staticmethod
    def _uses_pbs_task_resources(row):
        # Workers publish this state after the adapter and local collection
        # have both finished. Transfer still owns the attempt and result bytes.
        return row.get('state') != 'COMPLETED_PENDING_TRANSFER'

    def _ready_cpu_supply(self, ready, owned, policy, error):
        """Measure compatible waiting work against each fresh worker's CPU offer."""
        from .rpe_pbs import GIB, resource_need, task_estimate
        from .rpe_pbs_worker import runtime_cpu_footprint

        manager = self.pbs_manager
        snapshot = self.store.rows_for([task.job.name for task in ready] + [row['job_id'] for row in owned])
        rows = [snapshot[task.job.name] for task in ready]
        owned_rows = [snapshot[row['job_id']] for row in owned]
        supply = {}
        eligible = {}
        for host, info in self.hosts.items():
            if (not self._offer_fresh(host) or info.get('accepting_new_jobs') is False
                    or not self._host_admission(host, policy, error)['accepting']):
                continue
            slots = self._offer_free(host, policy, error, owned=owned)
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
                running = [footprint(row) for row in owned_rows if row['host_id'] == host
                           and (pool is None or self._uses_pbs_task_resources(row))]
                if None in running:
                    continue
                # A job that fits after a running task completes is useful
                # replacement work. Live claims still use the remaining RAM.
                memory_limit = pool['memory_bytes'] - GIB if pool is not None else None
                free_memory = memory_limit - sum(memory for _, memory in running) if pool is not None else None
                free_cpu = max(0, limit - sum(cpu for cpu, _ in running)) if limit else None
                compatible_jobs = 0
                fits_now = 0
                estimates = []
                q3_estimates = []
                unknown = 0
                host_jobs = {}
                for row in rows:
                    request = row['request']
                    if (not self._host_supports_stage(host, request) or not self._fits_host_walltime(host, request)
                            or not self.store.eligible(request, row['eligibility'], host,
                                                       info.get('host_fingerprints'), row['job_id'],
                                                       request_sha256=row['request_sha256'])):
                        continue
                    need = footprint(row)
                    if need is None:
                        unknown += 1
                        continue
                    cpu, memory = need
                    if (memory_limit is not None and memory > memory_limit) or (limit and cpu > limit):
                        continue
                    compatible_jobs += 1
                    if ((free_cpu is None or cpu <= free_cpu)
                            and (free_memory is None or memory <= free_memory)):
                        fits_now += 1
                    stage = (request.get('preparation') or {}).get('stage', 'q3')
                    seconds = task_estimate(request, manager.settings, host) if pool is not None else None
                    host_jobs[row['job_id']] = (cpu, stage, cpu * seconds if seconds else None)
                    estimates.append(cpu)
                    if stage == 'q3':
                        q3_estimates.append(cpu)
                known_limit = type(limit) is int and limit > 0
                if not known_limit and report.get('mode') != 'FIXED':
                    continue
                # Maintain a replacement wave even with zero free slots. A
                # current completion is too late to begin a remote preflight.
                spare = max(0, limit - sum(cpu for cpu, _ in running)) if known_limit else slots * max(estimates, default=0)
                footprint_hint = min(estimates, default=max(1, profile.get('dp_workers') or 1))
                lanes = max(1, info.get('capacity') or slots)
                next_wave = lanes * footprint_hint
                if known_limit:
                    next_wave = min(limit, next_wave)
                target = max(spare, next_wave) * self.ready_queue_waves
                supply[host] = dict(available_cpu=spare, ready_cpu=0,
                                    capacity_source='RESOURCE_OFFER' if known_limit else 'FIXED_SLOTS_BY_READY_FOOTPRINT',
                                    ready_cpu_by_stage={}, unestimated_jobs=unknown,
                                    compatible_ready_jobs=compatible_jobs, ready_jobs_fitting_free_resources=fits_now,
                                    ready_cpu_seconds_by_stage={}, unknown_duration_jobs=0,
                                    ready_cpu_seconds=0, target_cpu_seconds=(
                                        target * self.live_readiness_refill_seconds if pool is not None else None),
                                    q3_cpus_per_job=min(q3_estimates, default=footprint_hint),
                                    target_cpu=target, missing_cpu=target)
                for job, estimate in host_jobs.items():
                    eligible.setdefault(job, {})[host] = estimate
            except (KeyError, ValueError, TypeError):
                continue
        # Supply is a forecast, independent of actual assignment. Allocate
        # inflexible jobs first and count a flexible job only once.
        for options in sorted(eligible.values(), key=len):
            host = max(options, key=lambda h: (
                supply[h]['missing_cpu'] / max(1, supply[h]['target_cpu']),
                supply[h]['available_cpu'] > 0, -self._host_rank(h)))
            cpu, stage, cpu_seconds = options[host]
            row = supply[host]
            row['ready_cpu'] += cpu
            stages = row['ready_cpu_by_stage']
            stages[stage] = stages.get(stage, 0) + cpu
            row['missing_cpu'] = max(0, row['target_cpu'] - row['ready_cpu'])
            if cpu_seconds is None:
                row['unknown_duration_jobs'] += 1
            else:
                seconds_by_stage = row['ready_cpu_seconds_by_stage']
                seconds_by_stage[stage] = seconds_by_stage.get(stage, 0) + cpu_seconds
                row['ready_cpu_seconds'] += cpu_seconds
        return supply

    def preparation_needs_refill(self):
        """Keep producing while any accepting host lacks ready work."""
        with self.state.cv:
            supply = [row for host, row in getattr(self.state, 'ready_cpu_supply', {}).items()
                      if self._offer_fresh(host)]
            return not supply or any(
                not row['ready_cpu'] or row['missing_cpu'] or row['unestimated_jobs']
                or (row.get('target_cpu_seconds') is not None and not row['unknown_duration_jobs']
                    and row['ready_cpu_seconds'] < row['target_cpu_seconds'])
                for row in supply
            )

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
                if self._offer_fresh(host) and not offer['unestimated_jobs']:
                    cpu_per_q3 = offer['q3_cpus_per_job']
                    deficits[host] = max(deficits[host],
                        (offer['missing_cpu'] + cpu_per_q3 - 1) // cpu_per_q3)
            capacity += sum(deficits.values())
        return max(1, capacity)

    def _can_take_ready_candidates(self):
        if self.ready_queue_waves > 1 and self.state.allocation_auto and self.distributed.get('work_conserving_search'):
            queued = sum(t.status == 'QUEUED' and t.rule['kind'] != 'preparation' for t in self.state.tasks.values())
            return not self._budget_reserved() and queued < self._standby_headroom()
        return super()._can_take_ready_candidates()

    def _live_readiness_window(self):
        maximum = self.live_readiness_max_workers or self.live_readiness_workers
        if not self.state.allocation_auto:
            return maximum if self.controller_parallel else min(maximum, self.live_readiness_workers)
        if maximum <= self.live_readiness_workers:
            return maximum
        missing = 0
        for host, supply in getattr(self.state, 'ready_cpu_supply', {}).items():
            if not self._offer_fresh(host):
                continue
            per_job = max(1, supply.get('q3_cpus_per_job', 1))
            missing += (supply['missing_cpu'] + per_job - 1) // per_job
        if not missing and not getattr(self.state, 'ready_cpu_supply', {}):
            missing = self._standby_capacity()
        horizon = self.live_readiness_refill_seconds
        seconds = getattr(self.state, 'q3_readiness_seconds', horizon)
        from math import ceil
        return min(maximum, max(self.live_readiness_workers, ceil(missing * min(1., seconds / horizon))))

    def _standby_can_fill(self):
        return (getattr(self, '_awaiting_probe_revision', None) !=
                getattr(self.state, 'probe_revision', 0) and super()._standby_can_fill())

    def _host_readiness_limit(self, host):
        window = self._live_readiness_window()
        local = self.distributed.get('local_worker') or {}
        if self.readiness_connections is not None and host != local.get('host_id'):
            window = min(window, self.readiness_connections.channels_per_host)
        capacity = self.hosts.get(host, {}).get('capacity')
        return min(window, self.ready_queue_waves * capacity) if capacity else window

    def _live_readiness_has_room(self, host=None):
        if not self.controller_parallel:
            return len(self.live_readiness_inflight) < self._live_readiness_window()
        if len(self.live_readiness_planning) >= self._live_readiness_window():
            return False
        hosts = [host] if host else [name for name, row in self.hosts.items()
                                    if row.get('accepting_new_jobs', True)]
        return not hosts or any(self.live_readiness_host_counts[name] < self._host_readiness_limit(name)
                                for name in hosts)

    def _close_live_readiness(self):
        if self.live_readiness_pool:
            self.live_readiness_pool.shutdown(wait=False, cancel_futures=True)
        for pool in self.live_readiness_host_pools.values():
            pool.shutdown(wait=False, cancel_futures=True)
        if self.readiness_connections is not None:
            self.readiness_connections.close()

    def _dispatch_live_checks(self, key, participants, check, receive, finish):
        """Use separate bounded lanes and publish each host as it finishes."""
        with self.state.cv:
            self.live_readiness_planning.discard(key)
            if self.state.closed:
                finish(())
                return
            checking = {}
            deferred = set()
            for item in participants:
                host = item[0]
                if self.live_readiness_host_counts[host] >= self._host_readiness_limit(host):
                    deferred.add(host)
                    continue
                pool = self.live_readiness_host_pools.get(host)
                if pool is None:
                    pool = self.live_readiness_host_pools[host] = ThreadPoolExecutor(
                        max_workers=self.live_readiness_max_workers or self.live_readiness_workers,
                        thread_name_prefix='rpe-q3-readiness-'+host)
                self.live_readiness_host_counts[host] += 1
                checking[pool.submit(check, item)] = host

            def complete(future):
                with self.state.cv:
                    host = checking.pop(future)
                    self.live_readiness_host_counts[host] -= 1
                    try:
                        if not self.state.closed and not future.cancelled():
                            receive(*future.result())
                    except (OSError, KeyError, TypeError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                        self.log(event='Q3_HOST_READINESS_ERROR', job=key[0], host_id=host, reason=str(error))
                    finally:
                        self.state.probe_revision = getattr(self.state, 'probe_revision', 0) + 1
                        if not checking:
                            finish(deferred)
                        self.state.cv.notify_all()

            if not checking:
                finish(deferred)
            for future in tuple(checking):
                future.add_done_callback(complete)

    def _fill_standby(self):
        if not (self.live_readiness_pool and (self.controller_parallel or
                self.ready_queue_waves > 1 and self.distributed.get('work_conserving_search'))):
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
                if len(waiting) >= self._live_readiness_window():
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
        if not available:
            return set()
        owners = {h:[] for h in available}
        compatible = {};skipped = {h:False for h in available}
        policy, error = (self._host_admission_policy() if self.distributed.get('preparation_host_preference')
                         else (None, None))
        queue = list(queue)
        rows = self.store.rows_for(task.job.name for task in queue)
        for task in queue:
            row = rows[task.job.name]
            compatible[task.job.name] = []
            for h in available:
                if not self._host_supports_stage(h, row['request']) or not self._fits_host_walltime(h, row['request']):
                    skipped[h] = True
                    continue
                if (not self._fits_host_task_class(h, row)
                        or self._wait_for_preparation_preference(h, row, policy, error)
                        or not self._local_calibration_pair_allows(h, row['request'])):
                    skipped[h] = True
                    continue
                if not self.store.eligible(row['request'], row['eligibility'], h,
                        self.hosts[h].get('host_fingerprints'), row['job_id'], request_sha256=row['request_sha256']):
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
        continuation = self.distributed.get('local_pf_continuation')
        if (continuation and provider and provider.startswith('PF_')
                and preparation.get('source', {}).get('commit') == continuation['source']
                and host not in continuation['references']):
            return False
        if provider in ('PF_COMPILE', 'PF_HOLDOUT_COMPILE') and self.distributed.get('pf_legacy_compile_hosts'):
            adapter = 'private/pf-transport/scripts/run_pf_preparation_stage.py'
            if adapter not in preparation.get('inputs', {}):
                providers = self.distributed['pf_legacy_compile_hosts']
        return (capabilities is None or stage in capabilities) and (providers is None or host in providers)

    def _fits_host_walltime(self, host, request):
        from .rpe_pbs import fits_request
        return fits_request(request, self.hosts.get(host, {}).get('resource_report'),
                            self.distributed.get('pbs') or {}, host)

    @staticmethod
    def _task_resource_class(row, host):
        from .rpe_q3_resources import measured_resource_class, resource_class
        eligibility = row['eligibility'] or {}
        decision = eligibility.get('host_decisions', {}).get(host, {})
        runtime = eligibility.get('execution_profiles', {}).get(host)
        if runtime is not None:
            return measured_resource_class(row['request'], decision.get('evaluator'), runtime)
        return resource_class(row['request'], decision.get('evaluator'))

    def _fits_host_task_class(self, host, row):
        report = self.hosts.get(host, {}).get('resource_report') or {}
        classes = report.get('admitted_task_classes')
        return classes is None or self._task_resource_class(row, host) in classes

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
        classify = self.distributed.get('live_readiness_max_workers') is not None
        local = self.distributed.get('local_worker') or {}
        class_admission = (classify and host == local.get('host_id')
                           and (local.get('slot_policy') or {}).get('separate_task_resources'))
        ready = 0
        capacity_ready = 0
        ready_classes = set()
        class_order = []
        observed = self.hosts.get(host, {})
        fingerprints = observed.get('host_fingerprints') or {}
        rejected = observed.get('readiness_errors') or {}
        admission = observed.get('host_admission') or {}
        storage = (observed.get('resource_report') or {}).get('storage') or {}
        can_plan = (admission.get('accepting', True) and admission.get('max_slots') != 0
                    and observed.get('accepting_new_jobs', True) and storage.get('accepting_new_jobs', True)
                    and not self._user_stop_requested() and not self.state.failure and not self.state.closed)
        queue = self._ready_queue() if class_admission else self.state.tasks.values()
        pending = [task for task in queue if can_plan and task.status == 'QUEUED']
        rows = self.store.rows_for(task.job.name for task in pending)
        for task in pending:
            row = rows.get(task.job.name)
            if row is not None and task.rule['kind'] == 'preparation' and not self._task_ready(task, row):
                continue
            if row is None or not self._host_supports_stage(host, row['request']) or not self._fits_host_walltime(host, row['request']):
                continue
            eligibility = row['eligibility'] or {}
            fresh = self.store.eligible(row['request'], row['eligibility'], host, fingerprints, row['job_id'],
                                        request_sha256=row['request_sha256'])
            ready += int(fresh)
            # A request-specific dry-check establishes demand even while the
            # busy worker has no claim-time fingerprint in its current poll.
            if not fresh and (row['job_id'] in fingerprints or row['job_id'] in rejected
                              or host not in eligibility.get('eligible_host_ids', ())
                              or not eligibility.get('host_fingerprints', {}).get(host)):
                continue
            capacity_ready += 1
            if classify:
                kind = self._task_resource_class(row, host)
                ready_classes.add(kind)
                if class_admission:
                    class_order.append(kind)
        ready_now = ready
        if self.distributed.get('work_conserving_search'):
            recent = getattr(self.state, 'host_ready_backlog', {})
            self.state.host_ready_backlog = recent
            # Use the same horizon that determines how much readiness work
            # should already be in flight. Polling at a completion must not
            # discard demand that was observed just before it.
            horizon = self.live_readiness_refill_seconds
            observations = {count: at for count, at in recent.get(host, {}).items() if now - at <= horizon}
            if ready:
                observations[ready] = now
            recent[host] = observations
            ready = max(observations, default=0)
        return dict(controller_invocation=self.invocation, measurement_seconds=now-self.state.allocation_start,
            controller_clock_id=stamp()['clock_id'],
            usable_results=sum(row['state'] == 'ACCEPTED' and (row['completed_epoch'] or 0) >= epoch for row in own),
            adopted_results=sum((row['adopted_epoch'] or 0) >= epoch for row in own),
            decision_wait_seconds=sum(selected) if selected else None,
            decision_wait_scope='SUM_SELECTED_JOB_DEPENDENCY_WAIT_SPANS_SAME_CONTROLLER_CLOCK',
            ready_jobs=ready, ready_jobs_now=ready_now, capacity_ready_jobs=capacity_ready,
            ready_task_classes=sorted(ready_classes), capacity=self.capacity,
            **({'ready_task_class_order': class_order} if class_admission else {}))

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

    def _publish_queued_readiness(self, job_id, request, decision):
        """Make an additional host visible to dispatch and PBS demand immediately."""
        hosts = frozenset(decision['eligible_host_ids'])
        task = self.state.tasks.get(self.run_dir/'jobs'/request['method']/job_id)
        if (not hosts or task is None or task.status != 'QUEUED' or self.state.closed
                or self._user_stop_requested() or hosts <= (task.supply_hosts or frozenset())):
            return
        row = self.store.row(job_id)
        if (row is not None and row['state'] == 'QUEUED'
                and row['request_sha256'] == decision['request_sha256']
                and row['detail'].get('prelaunch_recovery') is not True
                and self.store.set_eligibility(job_id, decision)):
            task.supply_hosts = hosts

    def _plan_pbs_q3_checks(self, job_id, request, decision):
        """Let the allocated worker check shortlisted requests before reservation."""
        if self.pbs_manager is None or self.state.closed:
            return
        for host in self.pbs_manager.configured:
            if host not in decision['eligible_host_ids']:
                if host in decision.get('host_decisions', {}):
                    self.pbs_q3_check_queue.get(host, {}).pop(job_id, None)
                continue
            pending = self.pbs_q3_check_queue.setdefault(host, {})
            previous = pending.get(job_id)
            if previous and previous[2]['request_sha256'] == decision['request_sha256']:
                # Late eligibility from other hosts must not push an already
                # waiting request behind the next compute-node check batch.
                pending[job_id] = (previous[0], request, decision)
            else:
                pending[job_id] = (time.monotonic(), request, decision)
            supply = getattr(self.state, 'ready_cpu_supply', {}).get(host, {})
            per_job = max(1, supply.get('q3_cpus_per_job', 1))
            depth = max(self._live_readiness_window(), (supply.get('target_cpu', 0) + per_job - 1) // per_job)
            while len(pending) > depth:
                del pending[next(reversed(pending))]

    def _pbs_q3_challenges(self, host, fingerprints, existing, slots):
        from .rpe_pbs import Q3_WORKER_PROOF_SECONDS

        pending = self.pbs_q3_check_queue.get(host, {})
        limit = min(slots, self._live_readiness_window()) - len(existing)
        if not pending or limit <= 0:
            return []
        rows = self.store.rows_for(pending)
        now = time.monotonic()
        result = []
        unreserved = []
        # Already queued requests take precedence over speculative probes.
        for job_id in sorted(pending, key=lambda job: job not in rows):
            checked, request, decision = pending[job_id]
            row = rows.get(job_id)
            if (now - checked >= Q3_WORKER_PROOF_SECONDS
                    or row is not None and (row['state'] != 'QUEUED'
                        or row['request_sha256'] != decision['request_sha256'])):
                del pending[job_id]
                continue
            if (job_id in existing or not self._host_supports_stage(host, request)
                    or not self._fits_host_walltime(host, request)
                    or self.store.eligible(request, decision, host, fingerprints, job_id)):
                continue
            try:
                challenge = self._assignment(dict(job_id=job_id, method=request['method'],
                    request=request, request_sha256=decision['request_sha256'], attempt_id=None),
                    host_id=host, eligibility=decision)
            except (OSError, KeyError, ValueError, subprocess.SubprocessError) as error:
                del pending[job_id]
                self.log(event='PBS_Q3_PREFLIGHT_UNAVAILABLE', job=job_id, host_id=host, reason=str(error))
                continue
            challenge.pop('attempt_id', None)
            result.append(challenge)
            if row is None:
                unreserved.append(job_id)
            if len(result) == limit:
                break
        if unreserved:
            self.log(event='PBS_Q3_PREFLIGHT_BEFORE_RESERVATION', host_id=host, jobs=unreserved)
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
                deferred = set(eligible.get('deferred_hosts', ())) & seen
                targets = {host} if self.controller_parallel and missing_host and cached and eligible['eligible_host_ids'] else None
                if targets is None and deferred and not eligible['eligible_host_ids']:
                    targets = deferred
                holds = (eligible.get('host_decisions') or {}).values()
                fixed_scope_hold = holds and all(h.get('reasons') in (
                    ['COUNT_ONLY_SCOPE_EXCEEDED'], ['OUTSIDE_VALIDATED_R3_SCALAR_EQUIVALENCE_DOMAIN'])
                    for h in holds)
                stale = not cached or (not eligible.get('controller_scope') and (
                    bool(deferred.intersection(targets or seen)) or
                    cached[1] != seen or not fixed_scope_hold
                    and (not eligible['eligible_host_ids'] or missing_host)
                    and time.monotonic() - cached[0] >= LIVE_READINESS_RETRY_SECONDS))
                if (stale and future is None and not self.state.closed and not self._user_stop_requested()
                        and (blocking or self._live_readiness_has_room(host if targets == {host} else None))):
                    started = time.monotonic()
                    def merge(decision):
                        if not decision.get('controller_scope'):
                            # An unobserved host keeps its prior PASS until this
                            # refresh reaches it. The claim still checks current facts.
                            observed = set(decision.get('host_decisions', {}))
                            for name in eligible['eligible_host_ids']:
                                if name not in observed and name not in decision['eligible_host_ids']:
                                    decision['eligible_host_ids'].append(name)
                                    for section in ('host_fingerprints', 'execution_profiles', 'host_decisions'):
                                        if name in eligible.get(section, {}):
                                            decision.setdefault(section, {})[name] = eligible[section][name]
                            waiting = (deferred - observed) | set(decision.get('deferred_hosts', ()))
                            if waiting:
                                decision['deferred_hosts'] = sorted(waiting)
                            if decision['eligible_host_ids']:
                                decision['reasons'] = []
                        decision.pop('decision_sha256', None)
                        decision['decision_sha256'] = digest(decision)
                        return decision

                    def publish(decision):
                        with self.state.cv:
                            if key not in self.live_readiness_partial:
                                elapsed = time.monotonic() - started
                                previous_seconds = getattr(self.state, 'q3_readiness_seconds', elapsed)
                                self.state.q3_readiness_seconds = .75 * previous_seconds + .25 * elapsed
                            if self.state.closed:
                                return
                            decision['eligible_host_ids'].sort()
                            decision = merge(decision)
                            self.live_readiness_partial[key] = decision
                            self._publish_queued_readiness(job_id, request, decision)
                            self._plan_pbs_q3_checks(job_id, request, decision)
                            self.state.probe_revision = getattr(self.state, 'probe_revision', 0) + 1
                            self.state.cv.notify_all()

                    def finish(decision):
                        with self.state.cv:
                            decision = merge(decision)
                            cache[key] = (time.monotonic(), seen, decision)
                            self.live_readiness_partial.pop(key, None)
                            self.live_readiness_inflight.pop(key, None)
                            self.live_readiness_planning.discard(key)
                            if not decision.get('deferred_hosts') or decision['eligible_host_ids']:
                                self.state.probe_revision = getattr(self.state, 'probe_revision', 0) + 1
                            self.state.cv.notify_all()
                            self._publish_queued_readiness(job_id, request, decision)
                            self._plan_pbs_q3_checks(job_id, request, decision)

                    def dispatch(participants, check_host, receive, snapshot):
                        def finish_checks(deferred_hosts):
                            try:
                                decision = snapshot(deferred_hosts)
                            except OSError as error:
                                decision = deepcopy(self.live_readiness_partial.get(key, result))
                                if not decision['eligible_host_ids']:
                                    decision['reasons'] = [str(error)]
                            finish(decision)

                        self._dispatch_live_checks(key, participants, check_host, receive, finish_checks)

                    def check():
                        try:
                            options = dict(dispatch=dispatch, targets=targets) if self.controller_parallel else {}
                            decision = probe(self, request, job_id, deepcopy(result), publish=publish, **options)
                        except (OSError, KeyError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                            decision = dict(result, reasons=[*result['reasons'], str(exc)])
                        if decision is not None:
                            finish(decision)

                    future = self.live_readiness_pool.submit(check)
                    self.live_readiness_inflight[key] = future
                    if self.controller_parallel:
                        self.live_readiness_planning.add(key)
            if not blocking or future is None or eligible['eligible_host_ids']:
                return deepcopy(eligible)
            while key in self.live_readiness_inflight and not self._user_stop_requested():
                if key in self.live_readiness_partial:
                    return deepcopy(self.live_readiness_partial[key])
                self.state.cv.wait(.25)
            if self._user_stop_requested():
                return deepcopy(result)
            return deepcopy(cache[key][2])

    def _enqueue_demand(self, proposal):
        return self._enqueue(proposal, demand=True,
                             defer_readiness=self.controller_parallel and self.live_readiness_pool is not None)

    def prefetch_standby(self, choice, proposals):
        self._awaiting_probe_revision = None
        if (self.controller_parallel or self.distributed.get('controller_queue_first')) and self.live_readiness_pool:
            # Start the fixed-input checks for the demand and independent next
            # candidates together. A check does not create a reservation.
            proposals = list(proposals)
            candidates = ([choice] if choice is not None else []) + proposals
            checked = 0
            for proposal in candidates:
                with self.state.cv:
                    if (self.state.closed or self._user_stop_requested() or self._budget_reserved()
                            or checked >= self._live_readiness_window()
                            or not self._live_readiness_has_room()):
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

    def _task_ready(self, task, row=None):
        request = (self.store.row(task.job.name) if row is None else row)['request']
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
        reference = self._task_release(request, host)
        if reference is not None:
            assignment['required_release'] = reference
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
        states = self.store.preparation_states(self.preparation_pending)
        for job_id, record in self.preparation_pending.items():
            if not self._host_supports_stage(host, record['request']):
                continue
            if job_id in states and states[job_id][1] not in ('QUEUED', 'CANCELLED'):
                continue
            candidates.append((job_id, record))
        if not candidates:
            self.preparation_probe_offsets.pop(host, None)
            return []
        start = self.preparation_probe_offsets.get(host, 0) % len(candidates)
        # Fill the current offer. Even a busy host can check the next request
        # in advance. Advance past failures and rotate through later requests.
        target = max(1, free_slots)
        challenges = []
        changed = False
        for examined, (job_id, record) in enumerate(candidates[start:]+candidates[:start], 1):
            try:challenges.append(self._pending_preparation_assignment(record,host))
            except (OSError,KeyError,ValueError,TypeError)as exc:
                failure=str(exc)
                if record.setdefault('failures',{}).get(host)!=failure:
                    record['failures'][host] = failure
                    changed = True
                hold=self._preparation_probe_hold(record)
                if hold is not None:self.preparation_holds[job_id]=hold
                continue
            if len(challenges) == target:
                break
        if changed:
            self._save_preparation_pending()
        self.preparation_probe_offsets[host] = (start + examined) % len(candidates)
        return challenges

    def _refresh_pending_preparation(self, host, fingerprints, errors):
        from .rpe_q3_coordinator import digest
        from .rpe_preparation_worker import validate_preparation_fingerprint
        if host not in self._preparation_probe_targets():return
        reports = sorted(self.preparation_pending.keys() & (fingerprints.keys() | errors.keys()))
        if not reports:
            return
        rows = self.store.rows_for(reports)
        modified = False
        for job_id in reports:
            record = self.preparation_pending[job_id]
            if not self._host_supports_stage(host, record['request']):
                continue
            old = rows.get(job_id)
            if old is not None and (old['state'] not in ('QUEUED', 'CANCELLED')
                                    or old['detail'].get('prelaunch_recovery') is True):
                continue
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
                modified |= changed
                self.preparation_holds.pop(job_id,None)
                # A verified metadata report enables a normal durable reservation;
                # readiness itself consumes no attempt or scientific budget.
                if old is None or old['state']=='CANCELLED':
                    self.submit_preparation(record['request'],budget=record['budget'],approved=record['approved'],
                        contract=record['contract'],eligibility=decision)
                elif old['state']=='QUEUED':self.store.set_eligibility(job_id,decision)
                if changed:
                    preparation = record['request']['preparation']
                    started = record.get('probe_started_epoch')
                    self.log(event='PREPARATION_HOST_READINESS', job=job_id, host_id=host,
                        status='PASS', stage=preparation['stage'], provider=preparation.get('provider'),
                        seconds_since_probe_requested=max(0, time.time()-started) if started else None)
            except (OSError,KeyError,ValueError,TypeError)as exc:
                failure=str(exc);failures=record.setdefault('failures',{})
                changed=failures.get(host)!=failure;failures[host]=failure
                if host in decision['eligible_host_ids']:
                    decision['eligible_host_ids'].remove(host);decision['host_fingerprints'].pop(host,None);decision['execution_profiles'].pop(host,None);changed=True
                decision['reasons']=['NO_MATCHING_PREPARATION_STAGE_HOST']if not decision['eligible_host_ids']else[]
                record['decision']=decision
                modified |= changed
                if changed:
                    self.log(event='PREPARATION_HOST_READINESS',job=job_id,host_id=host,status='UNCHECKED',stage=record['request']['preparation']['stage'],reason=failure)
                if old is not None and old['state']=='QUEUED':self.store.set_eligibility(job_id,decision)
                hold=self._preparation_probe_hold(record)
                if hold is not None:self.preparation_holds[job_id]=hold
        if modified:
            self._save_preparation_pending()
        self.state.cv.notify_all()

    def _prelaunch_recovery_challenges(self, host, slots, fingerprints):
        """Use normal host checks and keep an ended attempt's reservation."""
        from .rpe_q3_coordinator import atomic_json, digest
        from .rpe_preparation_worker import validate_preparation_fingerprint

        challenges = []
        if not self.state.approved:
            return challenges
        for row in self.store.prelaunch_rejections():
            if self._user_stop_requested() or self.state.failure or self.state.closed:
                break
            job_id, request = row['job_id'], row['request']
            if not self._host_supports_stage(host, request) or not self._fits_host_walltime(host, request):
                continue
            job = self.run_dir/'jobs'/row['method']/job_id
            if (job/'OUTCOME.json').exists() or (job/'PROCESS_RETURN.json').exists():
                continue
            transfer = self.run_dir/'distributed'/'transfers'/job_id/row['attempt_id']
            if (transfer/'OUTCOME.json').exists() or (transfer/'PROCESS_RETURN.json').exists():
                continue
            prep = self._preparation_request(request)
            facts = fingerprints.get(job_id)
            try:
                if prep:
                    assignment = self._pending_preparation_assignment(dict(request=request), host)
                    decision = dict(request_sha256=row['request_sha256'], eligible_host_ids=[host],
                                    host_fingerprints={}, execution_profiles={})
                    if facts is not None:
                        profile = self._configured_preparation_profile(host)
                        selected = validate_preparation_fingerprint(assignment, profile, facts)
                        decision['host_fingerprints'][host] = digest(facts)
                        decision['execution_profiles'][host] = selected
                else:
                    decision = self._eligibility_for(request, job_id, blocking=False, host=host,
                                                     previous=row['eligibility'], prefetch=True)
                    if not decision or host not in decision['eligible_host_ids']:
                        continue
                    assignment = self._assignment(row, host_id=host, eligibility=decision, metadata_only=True)
                    assignment.pop('attempt_id', None)
                if facts is None:
                    if len(challenges) < slots:
                        challenges.append(assignment)
                    continue
                if (host == row['host_id'] and digest(facts) ==
                        ((row['eligibility'] or {}).get('host_fingerprints') or {}).get(host)):
                    continue
                # A queued fallback still requires a fresh fingerprint at claim.
                decision = dict(decision, eligible_host_ids=[host])
                launch = job/'LAUNCH.json'
                if launch.is_file():
                    history = job/'attempt-history'/row['attempt_id']/'LAUNCH.json'
                    if not history.is_file():
                        history.parent.mkdir(parents=True, exist_ok=True)
                        atomic_json(history, json.loads(launch.read_bytes()))
                if not self.store.requeue_prelaunch(job_id, row['attempt_id'], host, decision, facts):
                    continue
                original = self.state.tasks.get(job)
                demand = prep is not None or (original is not None and original.demand)
                self.state.tasks.pop(job, None)
                self._restore_task(self.store.row(job_id), demand=demand)
                self.preparation_holds.pop(job_id, None)
                self.eligibility_holds.pop(job_id, None)
                self.log(event='PRELAUNCH_REDISPATCH', job=job_id, host_id=host,
                         original_host_id=row['host_id'], original_attempt_id=row['attempt_id'])
                self.state.cv.notify_all()
                slots -= 1
                if not slots:
                    break
            except (OSError, KeyError, ValueError, TypeError, subprocess.SubprocessError):
                # Leave the same attempt available for safe re-observation.
                continue
        return challenges

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

    def ready_preparations(self, job_ids, *, wait, stop_requested):
        """Wait for a watched job's result or its terminal eligibility decision."""
        if not job_ids:
            return []
        with self.state.cv:
            while not (self.state.closed or self.state.failure or stop_requested()):
                states = self.store.preparation_states(job_ids)
                ready = []
                can_progress = False
                for job in job_ids:
                    method, state = states.get(job, (None, None))
                    task = self.state.tasks.get(self.run_dir/'jobs'/method/job) if method else None
                    if state == 'UNKNOWN' or (state == 'ACCEPTED' and
                            (self.run_dir/'jobs'/method/job/'OUTCOME.json').is_file()) or (task and task.error):
                        ready.append(job)
                    if state == 'QUEUED':
                        row = self.store.row(job)
                        if self._preparation_dependency_hold(row):
                            ready.append(job)
                        elif wait:
                            eligibility = (row.get('eligibility') or {}).get('eligible_host_ids')
                            if eligibility is None:  # Artificial queues may have no readiness policy.
                                can_progress = True
                            else:
                                policy, error = self._host_admission_policy()
                                age = time.time() - row['enqueued_epoch']
                                for host in eligibility:
                                    admission = self._host_admission(host, policy, error)
                                    offer = self.hosts.get(host, {})
                                    if (not admission['accepting'] or admission['max_slots'] == 0
                                            or offer.get('accepting_new_jobs') is False):
                                        continue
                                    if (self._offer_fresh(host) and self._fits_host_walltime(host, row['request'])
                                            or not offer and age < self.allocation_config['offer_ttl_seconds']):
                                        can_progress = True
                                        break
                    elif state not in (None, 'CANCELLED'):
                        can_progress = True
                    if state not in (None, 'CANCELLED'):
                        continue
                    pending = self.preparation_pending.get(job)
                    hold = self.preparation_holds.get(job, {})
                    if (hold.get('status') == 'PREPARATION_BUDGET_OR_PERMISSION_HOLD'
                            or (pending is not None and self._preparation_probe_hold(pending) is not None)
                            or (pending is None and hold)):
                        ready.append(job)
                    else:
                        can_progress |= pending is not None
                if ready or not wait or not can_progress:
                    return ready
                # Some admission deadlines and STOPs are set outside this
                # process. Result publication and local eligibility wake us.
                self.state.cv.wait(.5)
        return []

    def _preparation_dependency_hold(self, row):
        for dependency in row['request']['preparation'].get('dependencies', []):
            result = self.preparation_result(dependency, adopt=False, materialize=False)
            if result is not None and result.get('status') != 'PREPARATION_COMPLETE':
                return dict(status='PREPARATION_DEPENDENCY_HOLD', reason='PREDECESSOR_NOT_COMPLETE',
                            dependency=dependency, predecessor=result, job_id=row['job_id'],
                            calls=0, preparation_calls=0)
        return None

    def preparation_result(self, job_id, *, wait=False, adopt=True, materialize=True):
        row = self.store.row(job_id)
        if row is not None and row['state'] == 'CANCELLED':
            row = None
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
                        if row is not None and row['state'] == 'CANCELLED':
                            row = None
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
        if row['state'] == 'QUEUED' and (hold := self._preparation_dependency_hold(row)):
            return hold
        task = self.state.tasks.get(self.run_dir/'jobs'/row['method']/job_id)
        if wait and row['state'] != 'ACCEPTED' and task is not None and not task.event.is_set():
            self.for_method(row['method']).log(event='DEPENDENCY_WAIT_START',job=job_id)
            try:
                while not task.event.wait(.1):
                    if hold := self._preparation_dependency_hold(row):
                        return hold
            finally: self.for_method(row['method']).log(event='DEPENDENCY_WAIT_END',job=job_id)
            row = self.store.row(job_id)
        if row['state'] == 'ACCEPTED':
            job = self.run_dir/'jobs'/row['method']/job_id
            try:
                outcome = json.loads((job/'OUTCOME.json').read_text())
            except FileNotFoundError:
                return None  # ACCEPTED precedes transfer publication; OUTCOME is published last.
            if not materialize:
                return dict(status=outcome['status'], job_id=job_id)
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

    def _wait_candidate_preparation(self, proposal, *, wait=True):
        for job_id in self.preparation_dependencies.get(proposal['configuration_id'], []):
            result = self.preparation_result(job_id, wait=wait)
            if result is None and not wait:
                return dict(status='DISPATCH_PENDING', reason='CANDIDATE_PREPARATION_RUNNING',
                            job_id=job_id, calls=0)
            if result is None or result.get('status') != 'PREPARATION_COMPLETE':
                return dict(status='HOST_ELIGIBILITY_HOLD',reason='CANDIDATE_PREPARATION_NOT_COMPLETE',preparation=result,host_eligibility=dict(preparation_task=job_id,result=result),job_id=job_id,calls=0,MSE_lower=None,MSE_upper=None,job=None)
        return None

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
                            cost=None if request['cost'] is None else O.frac(request['cost']), origin=request['origin'])
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
                            origin=request['origin'], cost=None if request['cost'] is None else O.frac(request['cost']))
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
            if eligibility and 'input_hashes' in eligibility:
                # A faster host may have published before the reservation existed.
                key = (job.name, eligibility['request_sha256'])
                cached = self.eligibility_policy_cache.get('live_checks', {}).get(key)
                latest = self.live_readiness_partial.get(key) or (cached and cached[2])
                if latest and set(latest['eligible_host_ids']) > set(eligibility['eligible_host_ids']):
                    eligibility = latest
            if defer_readiness and eligibility and not eligibility['eligible_host_ids']:
                key = (job.name, eligibility['request_sha256'])
                if key in self.live_readiness_inflight:
                    return _CHECK_PENDING
                if eligibility.get('deferred_hosts'):
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
        row = self.store.row(task.job.name)
        if row and row['state'] == 'QUEUED' and row['detail'].get('prelaunch_recovery') is True:
            super()._cancel(task)  # preserve the original reservation for the next Resume
            return
        if self.store.cancel(task.view.method, task.job.name): super()._cancel(task)

    def launched(self, method):
        return self.store.counts(method, started_only=True)

    def has_job(self, proposal):
        row = self.store.row(self.job_dir(proposal).name)
        return (row is not None and row['state'] != 'CANCELLED') or super().has_job(proposal)

    def _task_release(self, request, host):
        reference = self.distributed.get('required_release')
        continuation = self.distributed.get('local_pf_continuation')
        preparation = self._preparation_request(request)
        if (reference and continuation and preparation and preparation['provider'].startswith('PF_')
                and preparation['source']['commit'] == continuation['source']):
            if (host not in continuation['references']
                    or continuation['manifest']['parent_release'] != reference):
                raise ValueError('PF continuation host or parent release differs')
            reference = continuation['references'][host]
        return deepcopy(reference)

    def _assignment(self, row, host_id=None, *, eligibility=None, metadata_only=False):
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
        eligibility = eligibility if eligibility is not None else fixed.get('eligibility') or {}
        profile = (eligibility.get('execution_profiles') or {}).get(host)
        evaluator = (eligibility.get('host_decisions') or {}).get(host, {}).get('evaluator')
        if evaluator:
            value['q3_evaluator'] = evaluator
        if profile is not None:
            if preparation:
                from .rpe_preparation_worker import preparation_profile_core
                value['host_execution_profile'] = preparation_profile_core(profile)
            else: value['host_execution_profile'] = profile_core(profile)
            value['host_execution_profile_sha256'] = digest(value['host_execution_profile'])
        reference = self._task_release(row['request'], host)
        if reference is not None:
            value['required_release'] = reference
            continuation = self.distributed.get('local_pf_continuation')
            local=self.distributed.get('local_worker')or{}
            if not metadata_only and host == local.get('host_id') and self.distributed.get('local_deployment_layout'):
                from .rpe_release_deployment import read,validate_assignment_release
                if continuation and reference != self.distributed['required_release']:
                    manifest, layout = continuation['manifest'], continuation['layout']
                else:
                    manifest, layout = self.release_manifest, read(self.distributed['local_deployment_layout'])
                validate_assignment_release(manifest,layout,value,local)
        return value

    def _timing_queue(self):
        """Observe queue causes under the existing claim lock; never reorder."""
        s=self.state;reasons={}
        started = time.monotonic()
        # Waiting work and CPU supply are sampled diagnostics. Several hosts
        # can poll in the same instant. Rebuilding the same report for each
        # poll delays claims and result acknowledgements behind the lock.
        if started < getattr(s, 'next_queue_timing', 0):
            return
        admission=None;admission_error=None
        # Use the newer optional-host policy when supplied by the integration;
        # do not copy or change its dispatch/admission implementation here.
        if hasattr(self,'_host_admission_policy'):
            admission,admission_error=self._host_admission_policy()
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
                free = self._offer_free(host, admission, admission_error, owned=owned_attempts)
                observation.update(owned=owned[host], fresh_free_slots=free)
                storage = (self.hosts[host].get('resource_report') or {}).get('storage')
                if storage is not None:
                    observation['storage'] = storage
                    if not storage['accepting_new_jobs']:
                        observation['idle_reason'] = storage['reason']
                if free:
                    compatible = self._match_ready_jobs(ready, {host: free})
                    if work_conserving:
                        blocked_by_resources = (cpu is not None and cpu['compatible_ready_jobs']
                                                and not cpu['ready_jobs_fitting_free_resources'])
                        observation['idle_reason'] = ('TASK_RESOURCE_WAIT' if blocked_by_resources else
                            'ASSIGNMENT_WAIT' if compatible else
                            'ELIGIBLE_HOST_SHORTAGE' if ready else
                            'PREPARATION_WAIT' if waiting_preparation else
                            'PREPARATION_BUDGET_HOLD' if budget_blocked else 'CANDIDATE_SHORTAGE')
                    else:
                        observation['idle_reason'] = ('NO_READY_TASK' if not ready else
                            'DISPATCH_READINESS_OR_POLL_WAIT' if compatible else 'NO_ELIGIBLE_READY_TASK')
        marker=json.dumps(payload,sort_keys=True)
        if marker!=getattr(s,'last_timing_queue',None):
            s.last_timing_queue=marker;self.log(event='QUEUE_WAIT_STATE', report_seconds=time.monotonic()-started,**payload)
        s.next_queue_timing = time.monotonic() + 2

    @contextmanager
    def _rpc_queue_lock(self, path, host):
        waiting = time.monotonic()
        with self.state.cv:
            acquired = time.monotonic()
            if acquired-waiting >= 1:
                self.log(event='DISTRIBUTED_RPC_LOCK_WAIT', endpoint=path, host_id=host, seconds=acquired-waiting)
            try:
                yield
            finally:
                held = time.monotonic()-acquired
        if held >= 1:
            self.log(event='DISTRIBUTED_RPC_LOCK_HELD', endpoint=path, host_id=host, seconds=held)

    def rpc(self, path, value):
        from . import rpe_q3_coordinator as Q
        if path == '/v1/complete':
            return self._receive_completion(value)
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
        with self._rpc_queue_lock(path, host):
            if path == '/v1/poll':
                if value.get('source_sha') != self.optimizer_source: raise ValueError('Worker fixed SOURCE differs')
                stopping = self._user_stop_requested()
                slots = value['slots']
                if type(slots) is not int or slots < 0: raise ValueError('Nonnegative free slots required')
                fingerprints = value.get('host_fingerprints') or {}
                if not isinstance(fingerprints, dict): raise ValueError('Host fingerprints must map job IDs to current facts')
                admission_policy, admission_error = self._host_admission_policy()
                admission = self._host_admission(host, admission_policy, admission_error)
                owned = self._owned_attempts()
                new_slots = self._host_new_slots(host, slots, admission, owned=owned)
                host_capacity = value.get('capacity')
                if host_capacity is not None and (type(host_capacity) is not int or host_capacity < 0):
                    raise ValueError('Nonnegative host capacity required')
                if host not in self.hosts:
                    self.log(event='HOST_FIRST_POLL',host_id=host,capacity=value.get('capacity'))
                self.hosts[host] = dict(epoch=time.time(), slots=new_slots, reported_slots=slots,
                                       capacity=host_capacity, host_admission=admission,
                                       accepting_new_jobs=value.get('accepting_new_jobs', True),
                                       observed_monotonic=time.monotonic(),
                                       owned_at_offer=[row['attempt_id'] for row in owned if row['host_id'] == host],
                                       resource_report=value.get('resource_report'),
                                       cached_prefixes=value.get('cached_prefixes', []),
                                       host_fingerprints=fingerprints, readiness_errors=value.get('readiness_errors'))
                self._refresh_pending_preparation(host,fingerprints,value.get('readiness_errors')or{})
                self._refresh_capacity(admission_policy, admission_error)
                s.cv.notify_all()
                new_slots = self._offer_free(host, admission_policy, admission_error)
                may_probe = (not stopping and not s.failure and not s.closed and admission['accepting']
                             and admission['max_slots'] != 0 and value.get('accepting_new_jobs', True)
                             and self._offer_fresh(host)
                             and ((value.get('resource_report') or {}).get('storage') or {}).get('accepting_new_jobs', True))
                pending_challenges = self._pending_preparation_challenges(host, new_slots) if may_probe else []
                if may_probe and new_slots:
                    pending_challenges.extend(self._prelaunch_recovery_challenges(host, new_slots, fingerprints))
                def reply(*, probe_rows=(), **fields):
                    existing=fields.get('eligibility_challenges',[])
                    challenges = {item['job_id']:item for item in existing+pending_challenges}
                    if probe_rows:
                        # Workers verify the exact release during their metadata
                        # check. Keep a slow host from holding the shared lock
                        # while the controller validates every waiting request.
                        start = self.poll_probe_offsets.get(host, 0) % len(probe_rows)
                        limit = min(len(probe_rows), max(new_slots, self._host_readiness_limit(host)))
                        for row in (probe_rows[start:]+probe_rows[:start])[:limit]:
                            try:
                                challenge = self._assignment(row, host_id=host, metadata_only=True)
                            except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
                                self.log(event='HOST_PROBE_INPUT_UNAVAILABLE', host_id=host,
                                         job=row['job_id'], reason=str(error))
                                continue
                            challenge.pop('attempt_id', None)
                            challenges[challenge['job_id']] = challenge
                        self.poll_probe_offsets[host] = (start+limit) % len(probe_rows)
                    if may_probe and new_slots and self.pbs_manager is not None and host in self.pbs_manager.configured:
                        for challenge in self._pbs_q3_challenges(host, fingerprints, challenges, new_slots):
                            challenges[challenge['job_id']] = challenge
                    fields['eligibility_challenges'] = list(challenges.values())
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
                if not new_slots:
                    self._timing_queue()
                    return reply(assignments=assignments, stop=False, host_admission=admission,
                                eligibility_challenges=[])
                queue = self._ready_queue()
                if s.failure or s.closed: return reply(assignments=assignments, stop=False, host_admission=admission)
                eligible, challenges = [], []
                skipped = False
                rows = self.store.rows_for(task.job.name for task in queue)
                for task in queue:
                    row = rows[task.job.name]
                    if not self._host_supports_stage(host, row['request']):
                        skipped = True
                        continue
                    if not self._fits_host_walltime(host, row['request']):
                        skipped = True
                        continue
                    saved = row['eligibility']
                    # Live evidence is bound to the immutable request at reservation
                    # and claim requires this host's current exact fingerprint.
                    # A host without a stored PASS still goes through live readiness.
                    if (row['detail'].get('prelaunch_recovery') is True
                            or (saved and 'input_hashes' in saved and host in saved['eligible_host_ids']
                                and fingerprints.get(row['job_id']) is not None
                                and Q.digest(fingerprints[row['job_id']]) == saved['host_fingerprints'].get(host))):
                        decision = saved
                    else:
                        decision = self._eligibility_for(row['request'], row['job_id'], blocking=False,
                                                         previous=saved, host=host)
                    unchanged = decision == saved and self.store.eligible(
                        row['request'], decision, request_sha256=row['request_sha256'])
                    if not unchanged and not self.store.set_eligibility(row['job_id'], decision):
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
                    if not self.store.eligible(row['request'], decision, host, fingerprints, row['job_id'],
                                               request_sha256=row['request_sha256']):
                        # Metadata challenge only: no attempt ID, reservation
                        # transition, LAUNCH or scientific execution occurs.
                        if row['job_id'] not in self.preparation_pending:
                            challenges.append(row)
                        skipped = True
                        continue
                    if not self._fits_host_task_class(host, dict(row, eligibility=decision)):
                        skipped = True
                        continue
                    conditional = task.proposal.get('origin', {}).get('kind') == 'CHAIN_LOOKAHEAD_IF_PREVIOUS_PASSES'
                    if skipped and conditional and not task.demand:
                        continue
                    eligible.append(task)
                if value.get('observe_only') is True:
                    self._timing_queue()
                    return reply(assignments=[], stop=False, probe_rows=challenges, host_admission=admission)
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
                    # Start once at this host's first eligible poll, rather
                    # than at controller startup (planning/placement can be
                    # slow). Any higher-group observation ends discovery:
                    # an absent third host cannot delay known surplus work.
                    now = time.monotonic()
                    since = s.allocation_discovery.setdefault(host, now)
                    if (unknown and not any(h in self.hosts for h in higher) and
                            now-since < self.allocation_config['discovery_grace_seconds']):
                        def needs_discovery(task):
                            decision = self.store.row(task.job.name)['eligibility']
                            return decision is None or bool(set(unknown).intersection(decision['eligible_host_ids']))

                        eligible = [task for task in eligible if not needs_discovery(task)]
                        if not eligible:
                            return reply(assignments=assignments, stop=False, probe_rows=challenges,
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
                            and self._fits_host_task_class(h, first)
                            and self.store.eligible(first['request'], first['eligibility'], h,
                                                   info.get('host_fingerprints'), first['job_id'])]
                        if preferred:
                            eligible = eligible[1:]
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
                        if owned['host_id'] == host and self._uses_pbs_task_resources(owned):
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
                    prepared_assignment = None
                    if (row['request'].get('preparation') and self.distributed.get('required_release')
                            and host == (self.distributed.get('local_worker') or {}).get('host_id')
                            and self.distributed.get('local_deployment_layout')):
                        try:
                            prepared_assignment = self._assignment(row, host_id=host)
                        except (ValueError, OSError, subprocess.CalledProcessError) as error:
                            reason = 'PREPARATION_RELEASE_HOLD:' + str(error)
                            if task.error != reason:
                                task.error = reason
                                self.log(event='PREPARATION_RELEASE_HOLD', job=row['job_id'], host=host, reason=reason)
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
                        active=s.active, demand=task.demand,
                        provider=(row['request'].get('preparation') or {}).get('provider'),
                        eligible_hosts=(row['eligibility'] or {}).get('eligible_host_ids'))
                    if prepared_assignment is None:
                        assignments.append(self._assignment(row))
                    else:
                        assignments.append(dict(prepared_assignment, attempt_id=row['attempt_id']))
                    remaining -= 1
                self._timing_queue()
                self._refresh_capacity(admission_policy, admission_error)
                return reply(assignments=assignments, stop=False, probe_rows=challenges, host_admission=admission)
            row = self.store.row(value['job_id'])
            attempt = value['attempt_id']
            if path == '/v1/state' and row is None:
                # Workers reuse a spool across runs. A repeat notification
                # for another run is harmless and creates no ownership.
                return dict(ok=True, disposition='STATE_OUTSIDE_CURRENT_RUN')
            if row is None or row['host_id'] != host or row['attempt_id'] != attempt:
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
            raise ValueError('Unknown Q3 RPC endpoint')

    def _receive_completion(self, value):
        """Verify independent transfers before taking the shared queue lock."""
        from . import rpe_q3_coordinator as Q
        from .rpe_completion_acceptance import MissingPreparationArtifacts, validate_preparation_result

        host, job_id, attempt = (value[k] for k in ('host_id', 'job_id', 'attempt_id'))
        with self.transfer_locks[hash((job_id, attempt)) % len(self.transfer_locks)]:
            row = self.store.row(job_id)
            if row is None or (host, attempt) != (row['host_id'], row['attempt_id']):
                raise ValueError('Different execution host/attempt')
            if value['request_sha256'] != row['request_sha256']:
                raise ValueError('Completion request hash differs')
            files = value['files']
            completion_sha = Q.digest(files)
            job = self.run_dir/'jobs'/row['method']/job_id
            if row['state'] == 'ACCEPTED':
                if row['completion_sha256'] != completion_sha:
                    raise ValueError('Duplicate completion changed artifacts')
                self._publish_transfer(row)
                receipt = json.loads((job/'PROCESS_RETURN.json').read_text())
                with self.state.cv:
                    self._finish_received_task(job, row, receipt)
                return dict(ok=True, duplicate=True)

            started = time.monotonic()
            view = self.for_method(row['method'])
            view.log(event='RESULT_RECEIVED', job=job_id, host_id=host, attempt_id=attempt)
            transfer = self.run_dir/'distributed'/'transfers'/job_id/attempt
            transfer.mkdir(parents=True, exist_ok=True)
            Q.install_files(transfer, files)
            if not (transfer/'REQUEST.json').exists():
                _write(transfer/'REQUEST.json', row['request'])
            receipt = json.loads((transfer/'PROCESS_RETURN.json').read_text())
            outcome = json.loads((transfer/'OUTCOME.json').read_text())
            preparation = self._preparation_request(row['request'])
            validation, rejection = None, None
            if preparation:
                try:
                    validate_preparation_result(row, receipt, outcome, self.artifact_cache)
                except MissingPreparationArtifacts:
                    raise  # Keep the original uploader able to supply missing blobs.
                except ValueError as error:
                    rejection = str(error)
            else:
                success = outcome['status'] in (O.PASS, O.FAIL, 'INCONCLUSIVE')
                if success and (receipt.get('returncode') != 0 or receipt.get('signal') is not None):
                    rejection = 'Successful outcome without a clean adapter exit'
                else:
                    with self.state.cv:
                        task = self.state.tasks.get(job)
                        proposal = task.proposal if task else dict(configuration_id=row['request']['configuration_id'],
                            vector=O.interleave(row['request']['N_X'], row['request']['N_Y']))
                        owner = task.view if task and row['request']['route'].get('horizon_projection') else view
                    validation = owner.ingest(transfer, proposal, calls=0)
                    if success and validation['status'] == 'UNKNOWN':
                        rejection = 'Transferred outcome identity mismatch: '+str(validation.get('reason'))
            transfer_bytes = sum(len(x)*3//4-x.count('=') for x in files.values())
            verified = time.monotonic()
            with self.state.cv:
                current = self.store.row(job_id)
                if (host, attempt, row['request_sha256']) != (
                        current['host_id'], current['attempt_id'], current['request_sha256']):
                    raise ValueError('Execution owner changed during result verification')
                task = self.state.tasks.get(job)
                if rejection is not None:
                    self._hold_result_transfer(current, task, rejection)
                accepted, duplicate = self.store.complete_hashed(
                    host, job_id, attempt, row['request_sha256'], completion_sha)
                self._publish_transfer(accepted)
                if not duplicate:
                    if validation is not None and self.distributed.get('work_conserving_search'):
                        self._observe_q3_completion(accepted, transfer, validation)
                    view.log(event='TRANSFER_COMPLETE', job=job_id, host_id=host,
                        attempt_id=attempt, stage=preparation['stage'] if preparation else 'q3',
                        transfer_seconds=time.monotonic()-started, validation_seconds=verified-started,
                        lock_wait_seconds=time.monotonic()-verified,
                        transfer_bytes=transfer_bytes,
                        worker_transfer_seconds=value.get('transfer_seconds'))
                self._finish_received_task(job, accepted, receipt)
                return dict(ok=True, duplicate=duplicate)

    def _finish_received_task(self, job, row, receipt):
        task = self.state.tasks.get(job)
        if task:
            task.error = None
            if task.status == 'RUNNING':
                self._finished(task, receipt.get('returncode'), row['assigned_epoch'])
            else:
                task.event.set()
        if (not self._preparation_request(row['request']) and row['adopted_epoch'] is None
                and receipt.get('returncode') == 0):
            self._note_scaled_completion(row['method'], job, row['request'].get('origin') or {})
        self.state.cv.notify_all()

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

    def scope_unknown(self, job, proposal, answer):
        answer = super().scope_unknown(job, proposal, answer)
        if (answer is None or answer['status'] != 'UNKNOWN' or self.state.failure
                or answer.get('failure_scope') == 'CONFIGURATION'):
            return answer
        # A transferred result is checked before publication. Only isolate a
        # failure at the job directory that owns the exact durable request.
        if job != self.job_dir(proposal):
            return answer
        row = self.store.row(job.name)
        if not row or not row['attempt_id'] or not row['host_id'] or row['method'] != self.method:
            return answer
        request = row['request']
        if (request['configuration_id'] != proposal['configuration_id']
                or request['optimizer_source'] != self.computation_source
                or O.interleave(request['N_X'], request['N_Y']) != tuple(proposal['vector'])
                or not self.same_route(job, proposal['configuration_id'], request=request)):
            return answer
        reason = answer.get('reason')
        if (row['state'] not in ('UNKNOWN', 'ACCEPTED')
                and reason not in ('WORKER_CONTACT_LOST', 'WORKER_RESULT_MISMATCH')):
            return answer
        return dict(answer, failure_scope='CONFIGURATION',
                    reason=reason or row['detail'].get('reason') or 'WORKER_STATE_UNKNOWN_RECONCILIATION_REQUIRED')

    def answer_now(self, proposal):
        """Advance one independent scaled allocation without waiting for its worker."""
        if self._user_stop_requested():
            return dict(status='STOP_REQUESTED', calls=0)
        blocked = self._wait_candidate_preparation(proposal, wait=False)
        if blocked is not None:
            return blocked
        if self.route(proposal['configuration_id']) is None:
            return dict(status='NO_ROUTE', calls=0)
        job = self.job_dir(proposal)
        with self.state.cv:
            revision = getattr(self.state, 'probe_revision', 0)
            if (not self.has_job(proposal) and not self._budget_reserved() and not self._standby_has_capacity()):
                return dict(status='DISPATCH_PENDING', reason='Q3_QUEUE_WINDOW_FULL', job_id=job.name, calls=0)
        task = self._enqueue_demand(proposal)
        if task is _CHECK_PENDING or (task and not task.event.is_set()):
            return dict(status='DISPATCH_PENDING', reason='Q3_PREFLIGHT_OR_RESULT_PENDING',
                        job_id=job.name, probe_revision=revision, calls=0)
        if task is False:
            if job.name in self.eligibility_holds:
                return self._held_answer(proposal)
            return dict(status='BUDGET_EXHAUSTED', calls=0)
        result = self.answer(proposal, allow_new=False)
        return result if result is not None else dict(status='TECHNICAL_HOLD', reason='Q3_NOT_AVAILABLE', calls=0)

    def await_dispatch(self, pending):
        """Wait for a publication or a host proof after checking independent work."""
        with self.state.cv:
            if self.state.closed or self.state.failure or self._user_stop_requested():
                return False
            ready_pending = getattr(self, 'ready_pending', lambda: False)
            if self.scaled_completion_pending() or ready_pending():
                return True
            q3 = [row for row in pending if row['reason'] == 'Q3_PREFLIGHT_OR_RESULT_PENDING']
            jobs = {row['job_id'] for row in q3}
            revision = getattr(self.state, 'probe_revision', 0)
            if any(revision != row['probe_revision'] for row in q3):
                return True
            queue_wait = any(row['reason'] == 'Q3_QUEUE_WINDOW_FULL' for row in pending)
            if queue_wait or (q3 and self.live_readiness_inflight) or any(
                    task.job.name in jobs and task.status in ('QUEUED', 'RUNNING', 'DONE')
                    for task in self.state.tasks.values()):
                self.state.cv.wait(.25)
                return True
        preparation = {row['job_id'] for row in pending if row['reason'] == 'CANDIDATE_PREPARATION_RUNNING'}
        wait_ended = lambda: self._user_stop_requested() or ready_pending() or self.scaled_completion_pending()
        return bool(self.ready_preparations(preparation, wait=True, stop_requested=wait_ended)
                    or ready_pending() or self.scaled_completion_pending())

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
        return self.scope_unknown(job, proposal, answer)

    def record_prior_result_reuse(self, job):
        job = Path(job)
        if job.parent.parent != self.run_dir/'jobs':
            return
        # This set only suppresses duplicate timing events. The caller has
        # already checked the current outcome, route and input evidence.
        with self.state.reuse_lock:
            if job.name in self.state.reused_jobs:
                return
        host = self.store.accepted_host(job.name, job.parent.name)
        if host is not None:
            with self.state.reuse_lock:
                if job.name in self.state.reused_jobs:
                    return
                self.state.reused_jobs.add(job.name)
            self.log(event='RESULT_REUSE', job=job.name, host_id=host)

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

    def _reuse_counts(self):
        # All method views share the progress log and this cursor. A fresh
        # controller rebuilds counts from the durable log on its first query.
        with self.lock, (self.run_dir/'PROGRESS.jsonl').open('rb') as progress:
            stat = os.fstat(progress.fileno())
            identity = (stat.st_dev, stat.st_ino)
            previous, offset, counts = getattr(self.state, 'reuse_cursor', (None, 0, None))
            if previous != identity or stat.st_size < offset:
                offset, counts = 0, Counter()
            progress.seek(offset)
            for line in progress:
                if not line.endswith(b'\n'):
                    break
                if b'RESULT_REUSE' in line:
                    row = json.loads(line)
                    if row.get('event') == 'RESULT_REUSE' and row.get('host_id'):
                        counts[row['host_id']] += 1
                offset += len(line)
            self.state.reuse_cursor = (identity, offset, counts)
            return counts

    def _metrics_locked(self):
        result = super().metrics()
        rows = self.store.metric_rows(); by_host = {}
        for row in rows:
            if row['host_id'] is None: continue
            counts = by_host.setdefault(row['host_id'], dict(assigned=0, launched=0, completed=0, adopted=0, reused=0))
            counts['assigned'] += 1
            counts['launched'] += row['started_epoch'] is not None
            counts['completed'] += row['state'] == 'ACCEPTED'
            counts['adopted'] += row['adopted_epoch'] is not None
        for host, count in self._reuse_counts().items():
            if host in by_host:
                by_host[host]['reused'] = count
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
                          for row in rows if not row['is_preparation']
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
            if self.state.failure or self._user_stop_requested():
                self.state.retain_queued_preparation = False
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
        self._close_live_readiness()
        if self.pbs_manager:
            retain = (self.pbs_manager.retain_between_batches and not self.state.failure
                      and not self._user_stop_requested())
            self.pbs_manager.close(retain_allocation=retain)
        self.stop_controller_relay()
        try:
            self._finish_local_worker()
        finally:
            self._close_worker_rpc()

    def queued_preparation_count(self):
        with self.store._connection() as db:
            return db.execute("SELECT count(*) FROM jobs WHERE state='QUEUED' "
                              "AND method LIKE 'PREPARATION:%' "
                              "AND NOT coalesce(json_extract(request,'$.preparation.optional_price_validation'),0)").fetchone()[0]

    def await_dispatchable_preparation(self):
        with self.state.cv:
            policy, error = self._host_admission_policy()
            offers = {host: self._offer_free(host, policy, error) for host in self.hosts}
            ready = [task for task in self._ready_queue() if task.rule['kind'] == 'preparation' and task.demand]
            jobs = self._match_ready_jobs(ready, {host: slots for host, slots in offers.items() if slots})
        return self.ready_preparations(jobs, wait=True, stop_requested=self._user_stop_requested)

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
    ready_source = kwargs.get('ready_source')
    keep_active = (not close_executor and ready_source is not None
                   and isinstance(executor, DistributedExecutor)
                   and executor.distributed.get('work_conserving_search'))
    try:
        if kwargs.get('context') is None:kwargs['context']=O.prepare(root,problem,methods=methods)
        with ThreadPoolExecutor(max_workers=max(1,len(methods)))as pool:
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
                options = dict(kwargs, close_executor=False) if keep_active else kwargs
                future = pool.submit(O.optimize, root, problem, executor=executor.for_method(method),
                                     methods=[method], scaled_finalizers=finalizers, **options)
                future.add_done_callback(lambda done: finished(method, done))
                return future

            futures = {m: submit(m) for m in methods}
            if ready_source is None:
                out = {m: futures[m].result()[m] for m in methods}
            else:
                out = {}
                state = executor.state

                def idle_ready():
                    if not keep_active and not executor.has_spare_slot():
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
        elif not keep_active or executor.state.failure or executor._user_stop_requested():
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
