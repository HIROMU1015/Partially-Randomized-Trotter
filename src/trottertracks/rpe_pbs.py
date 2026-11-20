"""PBS allocations attached to the existing controller's durable task queue.

PBS submission is not a scientific task. Only an allocated worker's live pull
offer participates in matching; queued/held PBS jobs contribute no free slots.
"""
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import queue
import re
import shlex
import subprocess
import threading
from statistics import median
import time
import uuid

from .rpe_q3_timing import stamp, append
from .rpe_q3_worker import atomic, digest, read
from .rpe_pbs_worker import request_cpu_footprint, task_cpu_footprint
from .rpe_worker_storage import SAMPLE_SECONDS


TERMINAL = {'F', 'X', 'FINISHED_SAVED_WORKER', 'SUBMIT_REJECTED'}
GIB = 1024**3


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

    def storage_report(self, policy):
        config = self.settings
        remote = [config.get('gateway_python', '/usr/bin/python3'), '-B',
                  str(Path(config['transport_root'])/'scripts/run_rpe_q3_worker.py'), '--storage-limit-report']
        root = Path(config['remote_root'])
        for worker in config['workers']:
            remote.extend(('--storage-spool', str(root/'workers'/worker['host_id'])))
        if config.get('namespace_by_run', True):
            remote.extend(('--storage-pbs-runs', str(root.parent)))
        output = subprocess.check_output(
            ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5',
             config['ssh_alias'], shlex.join(remote)], input=json.dumps(policy), text=True, timeout=15)
        return json.loads(output)

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try: self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill(); self.process.wait()
            self.process = None


def q3_walltime_workers(rule, settings):
    """Resolve the runtime through the release's existing fixed worker profiles."""
    return [worker for worker in settings.get('workers', [])
            if any((route['route_kind'], route['view_commit'],
                    route['profile'].get('adapter_manifest', {}).get('commit'))
                   == (rule['route_kind'], rule['view_commit'], rule['adapter_source'])
                   for route in worker['profile'].get('route_profiles', []))]


def apply_walltime_policy(settings, policy, overrides, *, computation_source=None):
    """Resolve one release policy into the existing immutable PBS run settings."""
    if policy['schema'] != 'RPE_PBS_WALLTIME_POLICY_V1':
        raise ValueError('Unsupported PBS walltime policy')
    if (policy['computation_source'] != computation_source
            or policy['transport_source'] != settings.get('transport_source')):
        raise ValueError('PBS walltime policy SOURCE differs from selected release')
    q3 = deepcopy(policy['q3'])
    preparation = deepcopy(policy['preparation'])
    rules = q3 + preparation
    ids = {row['id'] for row in rules}
    if (len(ids) != len(rules) or set(overrides) - ids
            or any(type(value) is not int or value < 0 for value in overrides.values())):
        raise ValueError('PBS overrides require known rule IDs and nonnegative integer seconds')
    for rows, keys in ((q3, ('method', 'route_kind', 'view_commit')),
                       (preparation, ('stage', 'provider', 'source'))):
        if len({tuple(row[key] for key in keys) for row in rows}) != len(rows):
            raise ValueError('Ambiguous PBS walltime estimate')
    for row in rules:
        if type(row['seconds']) is not int or row['seconds'] <= 0:
            raise ValueError('Release PBS estimates must be positive integer seconds')
        if 'memory_bytes' in row and (type(row['memory_bytes']) is not int or row['memory_bytes'] <= 0):
            raise ValueError('Release PBS memory estimates must be positive integer bytes')
        row['seconds'] = overrides.get(row['id'], row['seconds'])
    for row in q3:
        row['hosts'] = [w['host_id'] for w in q3_walltime_workers(row, settings)]
        if not row['hosts']:
            raise ValueError('PBS walltime rule has no matching fixed route profile: ' + row['id'])
    settings.update(walltime_policy=dict(id=policy['id'], sha256=digest(policy)),
                    q3_task_seconds=q3, preparation_task_seconds=preparation, task_seconds={})
    if 'allocation_sizing' in policy:
        sizing = policy['allocation_sizing']
        memory = sizing['q3_startup_memory_bytes_per_task']
        if type(memory) is not int or memory <= 0:
            raise ValueError('PBS Q3 startup memory estimate must be positive integer bytes')
        settings['allocation_sizing'] = deepcopy(sizing)


def task_rule(request, settings, host_id=None):
    preparation = request.get('preparation') or {}
    stage = preparation.get('stage', 'q3')
    rules = settings.get('preparation_task_seconds')
    if rules is not None and preparation:
        provider = preparation.get('provider')
        source = (preparation.get('source') or {}).get('commit')
        matches = [rule for rule in rules if (rule['stage'], rule['provider'], rule['source'])
                   == (stage, provider, source)]
        return matches[0] if len(matches) == 1 else None
    if not preparation and 'q3_task_seconds' in settings:
        route = request.get('route') or {}
        values = (request.get('method'), route.get('kind'), route.get('view_commit'))
        matches = [rule for rule in settings['q3_task_seconds']
                   if values == (rule['method'], rule['route_kind'], rule['view_commit'])
                   and (host_id is None or host_id in rule['hosts'])]
        return matches[0] if len(matches) == 1 else None
    return dict(seconds=(settings.get('task_seconds') or {}).get(stage))


def task_estimate(request, settings, host_id=None):
    return (task_rule(request, settings, host_id) or {}).get('seconds')


def task_memory_estimate(request, settings, host_id=None):
    return (task_rule(request, settings, host_id) or {}).get('memory_bytes')


def usable_memory(profile, settings):
    amount, unit = re.fullmatch(r'(\d+)([kmgt]?b)?', settings['resources']['mem'].lower()).groups()
    memory = int(amount) * (1024 ** ('b', 'kb', 'mb', 'gb', 'tb').index(unit or 'b'))
    limit = profile.get('slot_policy', {}).get('resource_limits', {}).get('memory_bytes', memory)
    return min(memory, limit) // GIB * GIB - GIB


def resource_need(request, profile, settings, host):
    memory = task_memory_estimate(request, settings, host)
    if memory is None:
        # An unmeasured preparation stage runs alone from the memory pool.
        memory = usable_memory(profile, settings) if request.get('preparation') else settings['allocation_sizing']['q3_startup_memory_bytes_per_task']
    return (request_cpu_footprint(profile, request),
            max(1, memory, (profile.get('pbs') or {}).get('task_memory_bytes') or 0))


def packed_resources(requests, profile, settings, host, *, selected=None):
    resources = settings['resources']
    limits = profile.get('slot_policy', {}).get('resource_limits', {})
    cpu_limit = min(resources['ncpus'], limits.get('cpu_count', resources['ncpus']))
    memory_limit = usable_memory(profile, settings)
    cpus = memory = count = 0
    max_slots = profile.get('slot_policy', {}).get('max_slots') or cpu_limit
    # A preparation task without a memory estimate needs the whole pool. Give
    # tasks with a usable estimate a chance to share it before that fallback.
    ordered = sorted(requests, key=lambda request: bool(request.get('preparation'))
                     and task_memory_estimate(request, settings, host) is None)
    for request in ordered:
        if count >= max_slots:
            break
        try:
            cpu, ram = resource_need(request, profile, settings, host)
        except (KeyError, ValueError):
            continue
        if cpus + cpu <= cpu_limit and memory + ram <= memory_limit:
            cpus += cpu
            memory += ram
            count += 1
            if selected is not None:
                selected.append(request)
    return cpus, (memory + GIB - 1) // GIB + 1, count


def fits_request(request, report, settings, host_id=None):
    """Require a declared conservative runtime before starting scientific work."""
    pbs = (report or {}).get('pbs')
    if pbs is None:
        return True
    artificial = request.get('route', {}).get('kind') == 'artificial' or request.get('preparation', {}).get('artificial')
    estimate = task_estimate(request, settings, host_id)
    if estimate is None and artificial:
        estimate = pbs.get('minimum_task_seconds', 1)
    if not isinstance(estimate, (int, float)) or estimate <= 0:
        return False
    slot_memory = pbs.get('sized_task_memory_bytes')
    if slot_memory is not None:
        required = task_memory_estimate(request, settings, host_id)
        if required is None or required > slot_memory:
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
        self.storage_checks = {}
        self.last_error = None
        self.configured = {worker['host_id']:worker for worker in settings['workers']}
        if len(self.configured) != len(settings['workers']):
            raise ValueError('PBS host IDs must be distinct')

    def log(self, event, **fields):
        append(self.directory/'EVENTS.jsonl', event, **fields)

    def save(self):
        atomic(self.path, self.state)

    def readiness_wait_seconds(self):
        # PBS queue latency varies by queue and resource request. Keep the
        # wait bounded, including when the gateway or scheduler is unavailable.
        samples = [a['submit_to_ready_seconds'] for a in self.state['allocations'].values()
                   if 'submit_to_ready_seconds' in a]
        return max(15., min(30., median(samples[-10:]) if samples else 20.))

    def accepting_allocation(self, host, job_id):
        return any(allocation['host_id'] == host and allocation['status'] == 'R'
                   and allocation['job_id'] == job_id
                   and (allocation.get('worker') or {}).get('accepting')
                   and not self.state['allocations'][allocation['token']].get('resize_requested')
                   for allocation in self.allocations)

    def startup_jobs(self, *, except_host=None):
        """Offer a sized allocation a brief first chance at its checked work."""
        now = stamp()
        jobs = set()
        for record in list(self.state['allocations'].values()):
            spec = record['spec']
            submitted = record['submitted_clock']
            reason = spec['reason']
            if (record['status'] in ('INTENT', 'Q', 'R')
                    and spec['profile']['host_id'] != except_host
                    and not record.get('resize_requested')
                    and not record.get('worker_storage_hold')
                    and now['clock_id'] == submitted['clock_id']
                    and now['monotonic'] - submitted['monotonic'] < reason.get('startup_wait_seconds', 0)):
                jobs.update(reason.get('startup_jobs', ()))
        return jobs

    def preparation_readiness_wait_seconds(self, host):
        # Queue wait and worker-side preflight have different budgets. A live
        # worker may need several artifact RPCs before it can report readiness.
        # Bound the extended wait so a stalled controller cannot hold a lane.
        rpc_timeout = self.settings.get('request_timeout_seconds', 20)
        worker = self.configured[host]['profile']
        rpc_timeout = max(rpc_timeout, (worker.get('pbs') or {}).get('rpc_timeout_seconds', 15))
        base = self.readiness_wait_seconds() + 3 * rpc_timeout
        ready = any(allocation['host_id'] == host and allocation['status'] == 'R'
                    and (allocation.get('worker') or {}).get('accepting')
                    for allocation in self.allocations)
        if ready and self.last_error is None:
            return max(base, 300.)
        return base

    def _recent_preparation_completion(self, seconds):
        # A newly accepted, successful required stage may shortly deliver an
        # audit or a ready candidate. Never hold an allocation for a long stage
        # with no completion in sight or for optional price checks.
        with self.executor.store._connection() as database:
            rows = database.execute(
                "SELECT request, detail FROM jobs WHERE state='ACCEPTED' "
                "AND method LIKE 'PREPARATION:%' AND completed_epoch>=?",
                (time.time() - seconds,))
            return any(not json.loads(row[0])['preparation'].get('optional_price_validation')
                       and json.loads(row[1] or '{}').get('outcome_status') == 'PREPARATION_COMPLETE'
                       for row in rows)

    def _storage_accepting(self, worker):
        policy = worker['profile'].get('storage_limit')
        if policy is None:
            return True
        host = worker['host_id']
        sampled, report = self.storage_checks.get(host, (float('-inf'), None))
        if time.monotonic() - sampled >= SAMPLE_SECONDS:
            try:
                new = self.gateway.storage_report(policy)
                used = new['used_bytes']
                free = new.get('filesystem_free_bytes')
                minimum = new.get('filesystem_min_free_bytes')
                if used is None:
                    reason = 'STORAGE_USAGE_UNAVAILABLE'
                elif type(used) is not int or used < 0:
                    raise ValueError('Invalid PBS storage usage')
                elif used >= policy['limit_bytes']:
                    reason = 'STORAGE_LIMIT_REACHED'
                elif free is not None and minimum is not None and free < minimum:
                    reason = 'FILESYSTEM_SPACE_LOW'
                else:
                    reason = None
                if (new['limit_bytes'] != policy['limit_bytes']
                        or new['reason'] != reason
                        or new['accepting_new_jobs'] is not (reason is None)):
                    raise ValueError('PBS storage report differs from selected limit')
            except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
                new = dict(limit_bytes=policy['limit_bytes'], used_bytes=None,
                           accepting_new_jobs=False, reason='STORAGE_USAGE_UNAVAILABLE')
            if report is None or (report['accepting_new_jobs'], report['reason']) != (new['accepting_new_jobs'], new['reason']):
                self.log('PBS_STORAGE_ADMISSION', host_id=host, **new)
            report = new
            self.storage_checks[host] = (time.monotonic(), report)
            self.state.setdefault('storage', {})[host] = report
        return report['accepting_new_jobs']

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
                    spec = self.state['allocations'][token]['spec']
                    if spec['profile'].get('pbs', {}).get('per_task_resources'):
                        observed = current.get('resources') or {}
                        admission['resource_pool'] = dict(
                            cpu_count=min(spec['resources']['ncpus'], observed.get('cpu_count') or 0),
                            memory_bytes=min(int(spec['resources']['mem'][:-2]) * GIB,
                                             observed.get('memory_bytes') or 0))
                    else:
                        admission.pop('resource_pool', None)
            if request['endpoint'] == '/v1/poll' and (allocation is None or allocation['status'] != 'R' or
                    not (allocation.get('worker') or {}).get('accepting', False) or
                    self.state['allocations'][token].get('resize_requested')):
                # A delayed file survives worker/PBS termination; it cannot
                # constitute a current free-slot offer.
                payload.update(slots=0, capacity=0, accepting_new_jobs=False)
            result = self.executor.rpc(request['endpoint'], payload)
            if request['endpoint'] == '/v1/poll':
                record = self.state['allocations'][token]
                storage = (payload.get('resource_report') or {}).get('storage') or {}
                ready = payload['slots'] > 0 and payload.get('accepting_new_jobs', True)
                if ready:
                    previous = record.get('first_usable_offer_clock') or {}
                    now = stamp()
                    if previous.get('clock_id') != now['clock_id']:
                        record['first_usable_offer_clock'] = now
                    record['last_usable_offer_clock'] = now
                else:
                    record.pop('first_usable_offer_clock', None)
                    record.pop('last_usable_offer_clock', None)
                errors = payload.get('readiness_errors') or {}
                error_hash = digest(errors) if errors else None
                if (error_hash != record.get('worker_readiness_error_sha256') or
                        storage.get('reason') != record.get('worker_storage_hold')):
                    self.log('PBS_WORKER_ADMISSION', token=token, host_id=host,
                             storage_reason=storage.get('reason'), readiness_errors=errors)
                    record['worker_storage_hold'] = storage.get('reason')
                    record['worker_readiness_error_sha256'] = error_hash
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
            policy, error = self.executor._host_admission_policy()
            admission = {host: self.executor._host_admission(host, policy, error) for host in self.configured}
            hosts = [host for host, rule in admission.items() if rule['accepting'] and rule.get('max_slots') != 0]
            def compatible(request, eligibility, *, unchecked=False):
                artificial = self.executor.store.artificial(request)
                return [host for host in hosts if self.executor._host_supports_stage(host, request) and
                    (artificial or (task_estimate(request, self.settings, host) or 0) > 0) and
                    (unchecked or (eligibility is None and artificial) or
                     host in (eligibility or {}).get('eligible_host_ids', []))]

            # A task can already be reserved on another host before a PBS
            # allocation exists to check it. Retain that eligible demand if
            # there is no compatible free slot outside PBS.
            checks = {}
            budgets = self.executor.store.preparation_reservations()
            for job, record in self.executor.preparation_pending.items():
                request = record['request']
                row = self.executor.store.row(job)
                if row is not None and row['state'] not in ('QUEUED', 'CANCELLED'):
                    continue
                stage = request['preparation']['stage']
                budget = record.get('budget')
                needs_reservation = row is None or row['state'] == 'CANCELLED'
                if needs_reservation and (not record.get('approved', True) or
                        (budget not in (None, 'UNLIMITED') and budgets.get(stage, 0) >= budget)):
                    continue
                usable = [host for host in compatible(request, None, unchecked=True)
                          if host not in record.get('failures', {})]
                if usable:
                    checks[job] = (usable, request)
                    if needs_reservation and budget not in (None, 'UNLIMITED'):
                        budgets[stage] = budgets.get(stage, 0) + 1
            ready = []
            for task in self.executor._ready_queue():
                row = self.executor.store.row(task.job.name)
                usable = compatible(row['request'], row['eligibility'])
                unchecked, _ = checks.pop(task.job.name, ([], None))
                if usable or unchecked:
                    ready.append((task, usable, unchecked, row['request']))
            available = {host:self.executor._offer_free(host,policy,error) for host in self.executor.hosts if host not in self.configured}
            startup = self.startup_jobs()
            matched = self.executor._match_ready_jobs([task for task, _, _, _ in ready if task.job.name not in startup],
                {host:count for host,count in available.items() if count > 0})
            outstanding = [(usable, request) for task, usable, _, request in ready
                           if usable and task.job.name not in matched]
            pending = [(unchecked, request) for task, usable, unchecked, request in ready
                       if not usable and task.job.name not in matched]
            unreserved = [row for job,row in checks.items() if self.executor.store.row(job) is None]
            return outstanding, pending + unreserved, len(ready) + len(unreserved), len(matched)

    def _submit(self, host, worker, demand, requests=()):
        token = uuid.uuid4().hex
        profile = deepcopy(worker['profile'])
        resources = deepcopy(self.settings['resources'])
        sizing = self.settings.get('allocation_sizing')
        if sizing and sizing.get('per_task_resources'):
            packed = []
            cpus, gib, capacity = packed_resources(requests, profile, self.settings, host, selected=packed)
            if not capacity:
                raise ValueError('No PBS task fits the configured CPU and memory pool')
            resources.update(ncpus=cpus, mem=f'{gib}gb')
            profile.setdefault('pbs', {})['per_task_resources'] = True
            profile.setdefault('slot_policy', {})['initial_slots'] = capacity
            demand = dict(demand, requested_task_capacity=capacity, per_task_resources=True)
            # Hold only work that passed the PBS-specific input check and fit
            # this actual request. No queue owner, attempt or budget is changed.
            startup_jobs = []
            for request in packed:
                job = request.get('request_id')
                row = self.executor.store.row(job) if job else None
                if row and row['state'] == 'QUEUED' and host in (row.get('eligibility') or {}).get('eligible_host_ids', ()):
                    startup_jobs.append(job)
            if startup_jobs:
                samples = [a['submit_to_ready_seconds'] for a in self.state['allocations'].values()
                           if 'submit_to_ready_seconds' in a]
                demand.update(startup_jobs=startup_jobs, startup_wait_seconds=min(
                    120., max(samples[-6:], default=30.) + self.readiness_wait_seconds()))
        elif sizing:
            cost = task_cpu_footprint(profile)
            maximum = int(min(resources['ncpus'], (profile.get('slot_policy') or {}).get(
                'resource_limits', {}).get('cpu_count', resources['ncpus']))) // cost
            capacity = min(maximum, max(1, len(requests)))
            if capacity < 1:
                raise ValueError('PBS CPU limit does not fit one configured task')
            resources['ncpus'] = capacity * cost
            policy = profile.setdefault('slot_policy', {})
            if policy.get('max_slots') is not None:
                capacity = min(capacity, policy['max_slots'])
                resources['ncpus'] = capacity * cost
            if capacity < 1:
                raise ValueError('PBS slot limit does not fit one configured task')
            estimates = [task_memory_estimate(request, self.settings, host) for request in requests[:capacity]]
            if estimates and all(value is not None for value in estimates):
                amount, unit = re.fullmatch(r'(\d+)([kmgt]?b)?', resources['mem'].lower()).groups()
                maximum_memory = int(amount) * (1024 ** ('b', 'kb', 'mb', 'gb', 'tb').index(unit or 'b'))
                task_memory = max(*estimates, (profile.get('pbs') or {}).get('task_memory_bytes') or 0)
                unit = 1024**3
                task_gib = (task_memory + unit - 1) // unit
                overhead_gib = 1
                capacity = min(capacity, (maximum_memory // unit - overhead_gib) // task_gib)
                if capacity < 1:
                    raise ValueError('PBS memory limit does not fit one compatible task')
                resources.update(ncpus=capacity * cost, mem=f'{capacity * task_gib + overhead_gib}gb')
                policy['initial_slots'] = capacity
                profile.setdefault('pbs', {}).update(task_memory_bytes=task_gib * unit,
                                                     sized_task_memory_bytes=task_memory)
            elif requests and all(request.get('route', {}).get('kind') != 'preparation'
                                and task_estimate(request, self.settings, host) for request in requests):
                amount, unit = re.fullmatch(r'(\d+)([kmgt]?b)?', resources['mem'].lower()).groups()
                memory = int(amount) * (1024 ** ('b', 'kb', 'mb', 'gb', 'tb').index(unit or 'b'))
                per_task = sizing['q3_startup_memory_bytes_per_task']
                if (profile.get('pbs') or {}).get('task_memory_bytes'):
                    per_task = max(per_task, profile['pbs']['task_memory_bytes'])
                policy['initial_slots'] = max(1, min(capacity, memory // per_task))
            demand = dict(demand, requested_cpu_per_task=cost, requested_task_capacity=capacity,
                          initial_slots=policy.get('initial_slots', profile.get('slots', 1)))
        profile['host_id'] = host
        profile['work_dir'] = self.settings['remote_root']+'/workers/'+host
        profile.setdefault('prefix_cache_directory', self.settings['remote_root']+'/cache/'+host+'/dp')
        profile.setdefault('artifact_cache_directory', self.settings['remote_root']+'/cache/'+host+'/artifacts')
        if self.settings.get('namespace_by_run', True):
            pool = Path(self.settings['remote_root']).parent/'preparation-metadata-cache'
            storage = profile.get('storage_limit') or {}
            if any(pool.is_relative_to(Path(root)) for root in storage.get('roots', [])):
                profile.setdefault('preparation_metadata_cache_directory', str(pool))
        profile['pbs'] = dict(profile.get('pbs') or {}, worker_source=self.settings['transport_source'],
            worker_root=self.settings['transport_root'], drain_seconds=self.settings.get('drain_seconds', 30))
        profile['pbs'].setdefault('rpc_timeout_seconds', max(15., 2 * self.settings.get('request_timeout_seconds', 20)))
        deployment=worker.get('deployment')or self.settings.get('deployment')
        if deployment:
            profile['pbs']['deployment']=deepcopy(deployment)
        spec = dict(token=token, profile=profile, resources=resources,
            python=self.settings['python'], transport_root=self.settings['transport_root'],
            run_id=str(self.executor.run_dir), reason=demand)
        # Save intent BEFORE making the remote call. On lost qsub/SSH response
        # the same token is reconciled, never replaced with a new submission.
        started = stamp()
        self.state['allocations'][token] = dict(spec=spec, status='INTENT', clock=started, submitted_clock=started)
        self.save()
        self.log('PBS_SUBMISSION_INTENT', token=token, host_id=host, resources=spec['resources'], reason=demand)
        self.gateway.call(dict(action='ensure', spec=spec))
        return spec

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
            now = stamp()
            worker = allocation.get('worker') or {}
            started = record.get('submitted_clock') or {}
            if (allocation['status'] == 'R' and worker.get('accepting')
                    and 'submit_to_ready_seconds' not in record and now['clock_id'] == started.get('clock_id')):
                record['submit_to_ready_seconds'] = max(0., now['monotonic'] - started['monotonic'])
            if allocation['status'] == 'R' and worker.get('accepting'):
                first = record.get('first_accepting_clock') or {}
                if first.get('clock_id') != now['clock_id']:
                    record['first_accepting_clock'] = now
            record.update(status=allocation['status'], observed=allocation, clock=now)
            if allocation['status'] in TERMINAL or allocation['status'] == 'HISTORY_MISSING_RECONCILE_SPOOL':
                self._harvest(allocation)
        for token, requests in data['requests'].items():
            self.responses[token] = {request['request_id']:self._dispatch(token, request) for request in requests}
        outstanding, pending, queued, free = self._demand()
        waiting_tasks = outstanding + pending
        shortage = len(waiting_tasks)
        active = [a for a in self.allocations if a['status'] not in TERMINAL]
        stopping = self.stop.is_set() or self.executor._user_stop_requested() or self.executor.state.closed
        keep_for_preparation = (not (outstanding or pending) and not stopping and any(
            a['status'] == 'R' and (a.get('worker') or {}).get('accepting') for a in active)
            and self._recent_preparation_completion(self.readiness_wait_seconds()))
        hosts_with_attempts = {row['host_id'] for row in self.executor._owned_attempts()}
        for allocation in self.allocations:
            if allocation['status'] not in TERMINAL:
                token, host = allocation['token'], allocation['host_id']
                record = self.state['allocations'][token]
                spec = record['spec']
                current_cpus = spec['resources']['ncpus']
                requested_slots = sum(host in usable for usable, _ in waiting_tasks)
                cpu_per_task = task_cpu_footprint(spec['profile'])
                slot_memory = spec['profile'].get('pbs', {}).get('sized_task_memory_bytes')
                waiting = [request for hosts, request in waiting_tasks if host in hosts]
                estimates = [task_memory_estimate(request, self.settings, host) for request in waiting]
                memory_mismatch = slot_memory is not None and estimates and all(
                    estimate is None or estimate > slot_memory for estimate in estimates)
                if spec['profile'].get('pbs', {}).get('per_task_resources'):
                    base = self.configured[host]['profile']
                    requested_cpus, _, _ = packed_resources(waiting, base, self.settings, host)
                    ram = (int(spec['resources']['mem'][:-2]) - 1) * GIB
                    fits_current = False
                    for request in waiting:
                        try:
                            cpu, memory = resource_need(request, base, self.settings, host)
                        except (KeyError, ValueError):
                            continue
                        if cpu <= current_cpus and memory <= ram:
                            fits_current = True
                            break
                    memory_mismatch = requested_cpus > 0 and not fits_current
                else:
                    requested_cpus = min(self.settings['resources']['ncpus'], requested_slots * cpu_per_task)
                # Releasing the old idle allocation is required before another
                # allocation can take ownership under this PBS host identity.
                # Give a newly ready worker a chance to round-trip its input
                # checks and claim work before declaring the allocation idle.
                first = record.get('first_usable_offer_clock') or {}
                latest = record.get('last_usable_offer_clock') or {}
                now = record['clock']
                offered_long_enough = (first.get('clock_id') == now['clock_id']
                    and latest.get('clock_id') == now['clock_id']
                    and now['monotonic'] - latest['monotonic'] <= self.readiness_wait_seconds() and
                    now['monotonic'] - first['monotonic'] >= self.readiness_wait_seconds())
                if (not stopping and self.settings.get('allocation_sizing') and allocation['status'] == 'R'
                        and (allocation.get('worker') or {}).get('accepting') and not record.get('resize_requested')
                        and offered_long_enough
                        and (memory_mismatch or
                             requested_cpus >= 2 * current_cpus)
                        and host not in hosts_with_attempts):
                    record['resize_requested'] = True
                    self.log('PBS_IDLE_ALLOCATION_RESIZE_DRAIN', token=token, host_id=host,
                             previous_cpus=current_cpus, compatible_ready_jobs=requested_slots)
                self.controls[token] = dict(stop=stopping or record.get('resize_requested', False),
                    release_when_idle=not waiting and not keep_for_preparation and host not in hosts_with_attempts)
                if stopping and allocation['status'] in ('Q','H','W') and allocation.get('job_id') and host not in hosts_with_attempts:
                    result = self.gateway.call(dict(action='delete', token=allocation['token']))
                    self.log('PBS_UNUSED_ALLOCATION_CANCELLED', token=allocation['token'],
                        job_id=allocation['job_id'], result=result, reason='STOP_WITHOUT_OWNED_SCIENTIFIC_ATTEMPTS')
        if shortage and not stopping:
            owned_hosts = {a['host_id'] for a in active}
            # A lost remote response may leave an intent absent from this
            # snapshot. It still occupies its prospective allocation.
            owned_hosts.update(r['spec']['profile']['host_id'] for r in self.state['allocations'].values()
                               if r['status'] not in TERMINAL)
            covered = sum(a['spec'].get('reason', {}).get('requested_task_capacity',
                          max(1, a['spec']['resources']['ncpus']//task_cpu_footprint(a['spec']['profile'])))
                          for a in self.state['allocations'].values() if a['status'] not in TERMINAL)
            for host, worker in self.configured.items():
                if shortage <= covered: break
                if host in owned_hosts: continue
                if any(record['spec']['profile']['host_id'] == host and record['status'] == 'SUBMIT_REJECTED'
                       for record in self.state['allocations'].values()): continue
                # UNKNOWN/transfer-pending original attempts prevent reuse.
                if host in hosts_with_attempts: continue
                # Reserve room for independent preparation tasks whose worker
                # preflight can only run after PBS starts. Assignment still
                # requires that task's successful preflight.
                requests = [request for usable, request in waiting_tasks if host in usable]
                if not requests: continue
                if not self._storage_accepting(worker): continue
                spec = self._submit(host, worker, dict(ready_tasks=queued, existing_host_free_slots=free,
                    prospective_pbs_task_slots=covered, shortage=shortage), requests[:shortage-covered])
                covered += spec['reason'].get('requested_task_capacity',
                    max(1, spec['resources']['ncpus']//task_cpu_footprint(spec['profile'])))
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
