"""PBS allocations attached to the existing controller's durable task queue.

PBS submission is not a scientific task. Only an allocated worker's live pull
offer participates in matching; queued/held PBS jobs contribute no free slots.
"""
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import queue
import shlex
import subprocess
import threading
import time
import uuid

from .rpe_q3_timing import stamp, append
from .rpe_q3_worker import atomic, digest, read


TERMINAL = {'F', 'X', 'FINISHED_SAVED_WORKER', 'SUBMIT_REJECTED'}


class Gateway:
    """A private persistent SSH JSON-lines connection; reconnect is read/replay."""
    def __init__(self, settings):
        self.settings = settings
        self.process = None
        self.responses = queue.Queue()

    def _start(self):
        config = self.settings
        remote = [config.get('gateway_python', '/usr/bin/python3'), '-B',
                  config['gateway_script'], '--root', config['remote_root']]
        command = config.get('gateway_command') or [
            'ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
            '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=2',
            config['ssh_alias'], shlex.join(remote)]
        self.responses = queue.Queue()
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)
        process, responses = self.process, self.responses

        def collect():
            try:
                for line in process.stdout:
                    responses.put(line)
            finally:
                responses.put(None)
        threading.Thread(target=collect, daemon=True).start()

    def call(self, payload):
        if self.process is None or self.process.poll() is not None:
            self._start()
        try:
            self.process.stdin.write(json.dumps(payload, separators=(',', ':'))+'\n')
            self.process.stdin.flush()
            line = self.responses.get(timeout=self.settings.get('request_timeout_seconds', 20))
            if line is None:
                raise OSError('PBS gateway SSH connection closed')
            value = json.loads(line)
            if not value.get('ok'):
                raise ValueError(value.get('error', 'PBS gateway failed'))
            return value['result']
        except (OSError, ValueError, queue.Empty):
            self.close()
            raise

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try: self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill(); self.process.wait()
            self.process = None


def task_estimate(request, settings):
    preparation = request.get('preparation') or {}
    stage = preparation.get('stage', 'q3')
    rules = settings.get('preparation_task_seconds')
    if rules is not None and preparation:
        provider = preparation.get('provider')
        source = (preparation.get('source') or {}).get('commit')
        matches = [rule['seconds'] for rule in rules if (rule['stage'], rule['provider'], rule['source'])
                   == (stage, provider, source)]
        return matches[0] if len(matches) == 1 else None
    return (settings.get('task_seconds') or {}).get(stage)


def fits_request(request, report, settings):
    """Require a declared conservative runtime before starting scientific work."""
    pbs = (report or {}).get('pbs')
    if pbs is None:
        return True
    artificial = request.get('route', {}).get('kind') == 'artificial' or request.get('preparation', {}).get('artificial')
    estimate = task_estimate(request, settings)
    if estimate is None and artificial:
        estimate = pbs.get('minimum_task_seconds', 1)
    if not isinstance(estimate, (int, float)) or estimate <= 0:
        return False
    return (pbs.get('accepting', False) and
            pbs.get('remaining_seconds', 0) >= estimate + settings.get('drain_seconds', 30))


class PbsManager:
    def __init__(self, executor, settings, *, gateway=None):
        self.executor = executor
        self.settings = deepcopy(settings)
        if self.settings.get('namespace_by_run', True):
            self.settings['remote_root'] = self.settings['remote_root'].rstrip('/')+'/'+digest(
                dict(controller_run=str(executor.run_dir.resolve())))[:20]
        self.directory = executor.run_dir/'distributed'/'pbs'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = (self.directory/'MANAGER.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fixed = self.directory/'SETTINGS.json'
        if fixed.exists() and read(fixed) != self.settings:
            self.lock.close()
            raise ValueError('Existing run retains its pinned PBS settings')
        if not fixed.exists(): atomic(fixed, self.settings)
        self.path = self.directory/'STATE.json'
        self.state = read(self.path) if self.path.exists() else dict(allocations={}, requests={})
        self.gateway = gateway or Gateway(self.settings)
        self.stop = threading.Event()
        self.thread = None
        self.allocations = []
        self.responses = {}
        self.controls = {}
        self.last_error = None
        self.configured = {worker['host_id']:worker for worker in settings['workers']}
        if len(self.configured) != len(settings['workers']):
            raise ValueError('PBS host IDs must be distinct')

    def log(self, event, **fields):
        append(self.directory/'EVENTS.jsonl', event, **fields)

    def save(self):
        atomic(self.path, self.state)

    def _dispatch(self, token, request):
        host = self.state['allocations'][token]['spec']['profile']['host_id']
        if request['payload'].get('host_id') != host:
            raise ValueError('PBS request host differs from its saved allocation')
        request_id = request['request_id']
        saved = self.directory/'responses'/token/(request_id+'.json')
        saved.parent.mkdir(parents=True, exist_ok=True)
        identity = digest(request)
        if saved.exists():
            record = read(saved)
            if record['request_sha256'] != identity:
                raise ValueError('Reused PBS RPC ID has changed content')
            return record['response']
        # Queue/attempt operations themselves are idempotent if the controller
        # crashes after RPC commit but before saving this response.
        try:
            payload = deepcopy(request['payload'])
            allocation = next((a for a in self.allocations if a['token'] == token), None)
            if request['endpoint'] == '/v1/poll' and allocation is not None:
                admission = (payload.get('resource_report') or {}).get('pbs')
                current = allocation.get('worker') or {}
                if admission is not None:
                    admission['remaining_seconds'] = min(admission.get('remaining_seconds',0), current.get('remaining_seconds',0))
                    admission['accepting'] = admission.get('accepting',False) and current.get('accepting',False)
            if request['endpoint'] == '/v1/poll' and (allocation is None or allocation['status'] != 'R' or
                    not (allocation.get('worker') or {}).get('accepting', False)):
                # A delayed file survives worker/PBS termination; it cannot
                # constitute a current free-slot offer.
                payload.update(slots=0, capacity=0, accepting_new_jobs=False)
            result = self.executor.rpc(request['endpoint'], payload)
            response = dict(ok=True, result=result)
        except (OSError, ValueError, KeyError) as error:
            response = dict(ok=False, error=type(error).__name__+': '+str(error))
        atomic(saved, dict(request_sha256=identity, response=response, clock=stamp()))
        return response

    def _harvest(self, allocation):
        token = allocation['token']
        if self.state['allocations'][token].get('harvested'):
            return
        data = self.gateway.call(dict(action='harvest', token=token))
        missing = self.executor.artifact_cache.missing(data.get('artifact_hashes', []))
        if missing:
            recovered = self.gateway.call(dict(action='harvest', token=token, hashes=missing))
            self.executor.artifact_cache.put(recovered['blobs'])
        for item in data['attempts']:
            assignment = item['assignment']
            row = self.executor.store.row(assignment['job_id'])
            if row is not None and row['state'] == 'ACCEPTED':
                continue
            # The original durable task/owner/attempt must exist; the normal
            # completion path checks exit receipt, identities and all outcomes.
            self.executor.rpc('/v1/complete', dict(host_id=data['host_id'],
                job_id=assignment['job_id'], attempt_id=assignment['attempt_id'],
                request_sha256=assignment['request_sha256'], files=item['files']))
        ended = allocation['status'] in TERMINAL or (allocation.get('worker') or {}).get('state') == 'EXITED'
        for item in data.get('incomplete', []) if ended else []:
            assignment = item['assignment']; row = self.executor.store.row(assignment['job_id'])
            if row is not None and row['state'] not in ('ACCEPTED','UNKNOWN','CANCELLED'):
                self.executor.rpc('/v1/state', dict(host_id=data['host_id'], job_id=assignment['job_id'],
                    attempt_id=assignment['attempt_id'], state='UNKNOWN',
                    detail=dict(reason='PBS_ENDED_WITHOUT_TASK_EXIT_RECEIPT', automatic_retry=False)))
        if ended:
            for row in self.executor._owned_attempts():
                if row['host_id'] == data['host_id'] and row['state'] != 'UNKNOWN':
                    self.executor.rpc('/v1/state',dict(host_id=row['host_id'], job_id=row['job_id'],
                        attempt_id=row['attempt_id'], state='UNKNOWN',detail=dict(
                            reason='PBS_ENDED_WITHOUT_COMPLETE_ORIGINAL_ATTEMPT',automatic_retry=False)))
        self.state['allocations'][token]['harvested'] = ended
        self.log('PBS_SAVED_ARTIFACTS_RECONCILED', token=token, attempts=len(data['attempts']))

    def _demand(self):
        with self.executor.state.cv:
            def relevant(request, eligibility):
                artificial = self.executor.store.artificial(request)
                if not artificial and not (task_estimate(request, self.settings) or 0) > 0:
                    return False
                return any(self.executor._host_supports_stage(host, request) and
                    (eligibility is None and artificial or host in (eligibility or {}).get('eligible_host_ids', []))
                    for host in self.configured)
            tasks = [task for task in self.executor.state.tasks.values()
                     if task.status == 'QUEUED' and self.executor._task_ready(task) and relevant(
                         self.executor.store.row(task.job.name)['request'],
                         self.executor.store.row(task.job.name)['eligibility'])]
            # Pending preparation metadata is also demand for a live compute
            # worker, but does not consume a scientific budget or reserve a task.
            pending = sum(self.executor.store.row(job) is None and
                (task_estimate(record['request'], self.settings) or 0) > 0 and
                any(self.executor._host_supports_stage(host, record['request']) for host in self.configured)
                for job,record in self.executor.preparation_pending.items())
            queued = len(tasks) + pending
            policy, error = self.executor._host_admission_policy()
            available = {host:self.executor._offer_free(host,policy,error) for host in self.executor.hosts if host not in self.configured}
            free = len(self.executor._match_ready_jobs(tasks, {host:count for host,count in available.items() if count > 0}))
            return max(0, queued-free), queued, free

    def _submit(self, host, worker, demand):
        token = uuid.uuid4().hex
        profile = deepcopy(worker['profile'])
        profile['host_id'] = host
        profile['work_dir'] = self.settings['remote_root']+'/workers/'+host
        profile.setdefault('prefix_cache_directory', self.settings['remote_root']+'/cache/'+host+'/dp')
        profile.setdefault('artifact_cache_directory', self.settings['remote_root']+'/cache/'+host+'/artifacts')
        profile['pbs'] = dict(profile.get('pbs') or {}, worker_source=self.settings['transport_source'],
            worker_root=self.settings['transport_root'], drain_seconds=self.settings.get('drain_seconds', 30))
        deployment=worker.get('deployment')or self.settings.get('deployment')
        if deployment:
            profile['pbs']['deployment']=deepcopy(deployment)
        spec = dict(token=token, profile=profile, resources=self.settings['resources'],
            python=self.settings['python'], transport_root=self.settings['transport_root'],
            run_id=str(self.executor.run_dir), reason=demand)
        # Save intent BEFORE making the remote call. On lost qsub/SSH response
        # the same token is reconciled, never replaced with a new submission.
        self.state['allocations'][token] = dict(spec=spec, status='INTENT', clock=stamp())
        self.save()
        self.log('PBS_SUBMISSION_INTENT', token=token, host_id=host, resources=spec['resources'], reason=demand)
        self.gateway.call(dict(action='ensure', spec=spec))

    def cycle(self):
        for token, record in self.state['allocations'].items():
            if record['status'] == 'INTENT':
                self.gateway.call(dict(action='ensure', spec=record['spec']))
        data = self.gateway.call(dict(action='exchange', responses=self.responses, controls=self.controls))
        self.responses, self.controls = {}, {}
        self.allocations = data['allocations']
        for allocation in self.allocations:
            token = allocation['token']
            if token not in self.state['allocations']:
                # Separate runs use different remote roots. Never adopt an
                # unrelated directory merely because it is accessible.
                raise ValueError('Unowned PBS allocation in this run namespace')
            record = self.state['allocations'][token]
            if record.get('status') != allocation['status']:
                self.log('PBS_ALLOCATION_STATE', token=token, job_id=allocation['job_id'],
                    status=allocation['status'], worker=allocation['worker'])
            record.update(status=allocation['status'], observed=allocation, clock=stamp())
            if allocation['status'] in TERMINAL or allocation['status'] == 'HISTORY_MISSING_RECONCILE_SPOOL':
                self._harvest(allocation)
        for token, requests in data['requests'].items():
            self.responses[token] = {request['request_id']:self._dispatch(token, request) for request in requests}
        shortage, queued, free = self._demand()
        active = [a for a in self.allocations if a['status'] not in TERMINAL]
        stopping = self.stop.is_set() or self.executor._user_stop_requested() or self.executor.state.closed
        for allocation in self.allocations:
            if allocation['status'] not in TERMINAL:
                self.controls[allocation['token']] = dict(stop=stopping, release_when_idle=queued == 0)
                if stopping and allocation['status'] in ('Q','H','W') and allocation.get('job_id') and not any(
                        row['host_id'] == allocation['host_id'] for row in self.executor._owned_attempts()):
                    result = self.gateway.call(dict(action='delete', token=allocation['token']))
                    self.log('PBS_UNUSED_ALLOCATION_CANCELLED', token=allocation['token'],
                        job_id=allocation['job_id'], result=result, reason='STOP_WITHOUT_OWNED_SCIENTIFIC_ATTEMPTS')
        if shortage and not stopping:
            owned_hosts = {a['host_id'] for a in active}
            # A lost remote response may leave an intent absent from this
            # snapshot. It still occupies its prospective allocation.
            owned_hosts.update(r['spec']['profile']['host_id'] for r in self.state['allocations'].values()
                               if r['status'] not in TERMINAL)
            covered = sum(max(1, self.settings['resources']['ncpus']//max(1,int(a['spec']['profile'].get('dp_workers',1))))
                          for a in self.state['allocations'].values() if a['status'] not in TERMINAL)
            for host, worker in self.configured.items():
                if shortage <= covered: break
                if host in owned_hosts: continue
                if any(record['spec']['profile']['host_id'] == host and record['status'] == 'SUBMIT_REJECTED'
                       for record in self.state['allocations'].values()): continue
                # UNKNOWN/transfer-pending original attempts prevent reuse.
                if any(row['host_id'] == host for row in self.executor._owned_attempts()): continue
                self._submit(host, worker, dict(ready_tasks=queued, existing_host_free_slots=free,
                    prospective_pbs_task_slots=covered, shortage=shortage))
                covered += max(1,self.settings['resources']['ncpus']//max(1,int(worker['profile'].get('dp_workers',1))))
        self.last_error = None
        self.state['heartbeat'] = stamp()
        self.save()

    def start(self):
        def run():
            try:
                while not self.stop.is_set():
                    try: self.cycle()
                    except Exception as error:
                        self.last_error = type(error).__name__+': '+str(error)
                        self.log('PBS_TRANSPORT_PENDING', error=self.last_error, automatic_scientific_retry=False)
                    self.stop.wait(self.settings.get('poll_seconds', .25))
            finally:
                self.gateway.close()
        self.thread = threading.Thread(target=run, name='rpe-pbs-manager', daemon=True)
        self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=self.settings.get('request_timeout_seconds',20)+4)
        if self.thread and self.thread.is_alive():
            self.gateway.close(); self.thread.join(timeout=4)
        try:
            controls = {token:dict(stop=True, release_when_idle=True) for token,record in self.state['allocations'].items()
                        if record['status'] not in TERMINAL}
            if controls or self.responses:
                self.gateway.call(dict(action='exchange', responses=self.responses, controls=controls))
            for allocation in self.allocations:
                if allocation['status'] in ('Q','H','W') and allocation.get('job_id') and not any(
                        row['host_id'] == allocation['host_id'] for row in self.executor._owned_attempts()):
                    result = self.gateway.call(dict(action='delete', token=allocation['token']))
                    self.log('PBS_UNUSED_ALLOCATION_CANCELLED', token=allocation['token'],
                        job_id=allocation['job_id'], result=result, reason='CONTROLLER_CLOSE_NO_OWNED_ATTEMPTS')
        except Exception as error:
            self.log('PBS_DRAIN_CONTROL_PENDING', error=str(error), recovery='Same run resume reconciles original allocations')
        finally:
            self.gateway.close(); self.lock.close()

    def metrics(self):
        return dict(allocations=[{key:value for key,value in a.items() if key != 'scheduler'} for a in self.allocations],
            error=self.last_error, science_budget_per_task=True)
