"""One PBS allocation hosting the existing durable, source-bound pull worker.

Only the worker runs on a compute node.  The login-node gateway copies durable
JSON messages; it never executes an assigned scientific task.  Allocation exit
leaves the ordinary worker spool and per-task receipts on persistent storage.
"""
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import uuid

from . import rpe_q3_resources as Resources
from .rpe_host_profiles import check_runtime_profile
from .rpe_q3_timing import stamp
from .rpe_q3_worker import Worker, atomic, event, head, host_fingerprints, read, save_state


THREAD_VARIABLES = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
                    'NUMBA_NUM_THREADS', 'RAYON_NUM_THREADS', 'QISKIT_NUM_PROCS')
TRANSPORT_FILES = ('src/trottertracks/rpe_pbs_worker.py',
                   'scripts/run_rpe_pbs_worker.py')


def _positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Positive finite ' + name + ' required')
    return value


def fixed_worker(profile):
    """Check only the pinned transport bytes; task SOURCE checks remain unchanged."""
    pbs = profile['pbs']
    root = Path(pbs.get('worker_root', profile.get('adapter_root', profile['root']))).resolve()
    expected = pbs['worker_source']
    if head(root) != expected:
        raise ValueError('PBS worker HEAD differs from fixed transport SOURCE')
    for name in TRANSPORT_FILES:
        saved = subprocess.check_output(['git', 'show', expected + ':' + name], cwd=root)
        if (root / name).read_bytes() != saved:
            raise ValueError('PBS worker SOURCE bytes differ: ' + name)
    return dict(root=str(root), source=expected,
                files={name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                       for name in TRANSPORT_FILES})


def runtime_profiles(profile):
    """All active runtime choices, without changing their scientific contracts."""
    yield 'default', profile
    for index, rule in enumerate(profile.get('route_profiles', [])):
        yield 'route-' + str(index), rule['profile']
    for index, rule in enumerate(profile.get('preparation_profiles', [])):
        yield 'preparation-' + str(index), rule['profile']


def task_cpu_footprint(profile):
    return max(runtime_cpu_footprint(profile, runtime) for _, runtime in runtime_profiles(profile))


def runtime_cpu_footprint(profile, runtime):
    threads = int(_positive(profile.get('pbs', {}).get('library_threads', 1), 'library_threads'))
    return max(int(runtime.get('dp_workers') or 1) * threads, int(runtime.get('task_cpu_count') or 1))


def request_cpu_footprint(profile, request):
    from .rpe_q3_worker import select_host_profile
    runtime = select_host_profile(profile, {'request': request})
    return runtime_cpu_footprint(profile, runtime)


def allocation_profile(profile, allocation, *, environ=None, snapshot=None):
    """Bind actual PBS limits, accounting DP workers and inner library threads.

    The maximum configured task footprint is used for every slot.  This is a
    conservative shared reservation across Q3/compile/signal/audit; no stage has
    a separate pool that could reserve the same CPU or memory twice.
    """
    env = os.environ if environ is None else environ
    job_id = env.get('PBS_JOBID')
    if not job_id or job_id != allocation.get('job_id'):
        raise ValueError('PBS_JOBID differs from the actual allocation record')
    if allocation.get('select_nodes') != 1:
        raise ValueError('PBS worker supports exactly one allocated node')
    ncpus = int(_positive(allocation['ncpus'], 'allocation ncpus'))
    memory = int(_positive(allocation['memory_bytes'], 'allocation memory_bytes'))
    _positive(allocation['walltime_seconds'], 'allocation walltime_seconds')
    resolved = deepcopy(profile)
    pbs = resolved['pbs']
    threads = int(_positive(pbs.get('library_threads', 1), 'library_threads'))
    weighted = pbs.get('per_task_resources') is True
    cost = 1 if weighted else task_cpu_footprint(resolved)
    policy = resolved.setdefault('slot_policy', {})
    declared = policy.setdefault('resource_limits', {})
    declared['cpu_count'] = min(ncpus, declared.get('cpu_count', ncpus))
    declared['memory_bytes'] = min(memory, declared.get('memory_bytes', memory))
    snapshot = Resources.resource_snapshot(resolved, environ=env) if snapshot is None else snapshot
    cpu_bound = min(ncpus, snapshot.get('allowed_cpu_count') or ncpus,
                    declared['cpu_count'])
    memory_bound = min(memory, snapshot.get('allowed_memory_bytes') or memory,
                       declared['memory_bytes'])
    cpu_slots = int(cpu_bound // cost)
    if cpu_slots < 1:
        raise ValueError('Configured DP/library/compiler CPU footprint exceeds PBS allocation')
    task_memory = None if weighted else pbs.get('task_memory_bytes')
    memory_slots = int(memory_bound // _positive(task_memory, 'task_memory_bytes')) if task_memory else cpu_slots
    maximum = min(cpu_slots, memory_slots, int(policy.get('max_slots', cpu_slots)))
    if maximum < 1:
        raise ValueError('Configured task memory footprint exceeds PBS allocation')
    policy['mode'] = 'fixed' if weighted else 'auto'
    policy['initial_slots'] = min(int(policy.get('initial_slots', resolved.get('slots', 1))), maximum)
    if policy['initial_slots'] < 1:
        raise ValueError('Positive initial_slots required')
    policy['max_slots'] = maximum
    if weighted:
        # This offer is a count ceiling. The controller reserves CPU and RAM
        # against each actual request before claiming it under the queue lock.
        resolved['slots'] = maximum
    # Existing auto-slot logic accounts dp_workers.  The CPU bound additionally
    # enforces library threads/compiler parallelism without changing task profiles.
    policy['resource_limits']['cpu_count'] = min(cpu_bound, maximum * cost)
    policy['resource_limits']['memory_bytes'] = memory_bound
    return resolved, dict(actual=deepcopy(allocation), snapshot=snapshot,
                          cpu_count=cpu_bound, memory_bytes=memory_bound,
                          library_threads=threads, task_cpu_count=cost,
                          cpu_slots=cpu_slots, maximum_slots=maximum,
                          task_memory_bytes=task_memory,
                          stages=sorted({'q3'} | {rule['stage'] for rule in resolved.get('preparation_profiles', [])}),
                          one_shared_slot_pool=True)


class SpoolTransport:
    """Durable request/response transport over the allocation's shared directory.

    A communication timeout never retries scientific execution.  Worker/JobStore
    reconciliation retains the same attempt.  Responses are consumed only when
    the request ID matches; the gateway persists its answer before delivery.
    """
    def __init__(self, directory, *, deadline, timeout_seconds=15.,
                 poll_seconds=.1, heartbeat=None):
        self.directory = Path(directory)
        self.requests = self.directory / 'requests'
        self.responses = self.directory / 'responses'
        self.requests.mkdir(parents=True, exist_ok=True)
        self.responses.mkdir(parents=True, exist_ok=True)
        self.deadline = deadline
        self.timeout_seconds = timeout_seconds
        self.poll_seconds = poll_seconds
        self.heartbeat = heartbeat

    def __call__(self, endpoint, payload):
        matching = []
        for path in self.requests.glob('*.json'):
            saved = read(path)
            if saved['endpoint'] == endpoint and (endpoint == '/v1/poll' or saved['payload'] == payload):
                matching.append((path, saved))
        # Exactly one unconsumed acquisition may exist across timeouts and
        # process restarts. A fresh UUID here could reserve another task before
        # the first reply has reached the worker.
        if endpoint == '/v1/poll' and len(matching) > 1:
            raise ValueError('Multiple unconsumed PBS acquisition requests require reconciliation')
        if matching:
            request, saved = matching[0]
            request_id = saved['request_id']
        else:
            request_id = uuid.uuid4().hex
            request = self.requests / (request_id + '.json')
            atomic(request, dict(request_id=request_id, endpoint=endpoint,
                                 payload=payload, clock=stamp()))
        response = self.responses / (request_id + '.json')
        limit = min(self.deadline, time.monotonic() + self.timeout_seconds)
        while True:
            if self.heartbeat:
                self.heartbeat()
            if response.exists():
                value = read(response)
                if value.get('request_id', request_id) != request_id:
                    raise ValueError('PBS spool response ID differs')
                request.unlink(missing_ok=True)
                response.unlink(missing_ok=True)
                if value.get('ok') is not True:
                    raise ValueError(value.get('error', 'PBS gateway rejected request'))
                return value['result']
            if time.monotonic() >= limit:
                raise TimeoutError('PBS gateway response remains pending: ' + request_id)
            time.sleep(min(self.poll_seconds, max(0, limit - time.monotonic())))


class AllocationWorker(Worker):
    """Add PBS admission metadata to the existing worker's unchanged RPC."""
    def __init__(self, profile, *, directory, deadline, resources, poll_seconds=1.):
        self.directory = Path(directory)
        self.deadline = deadline
        self.pbs_resources = resources
        self.drain_seconds = float(profile['pbs'].get('drain_seconds', 30.))
        self.minimum_task_seconds = float(profile['pbs'].get('minimum_task_seconds', 60.))
        self.idle_release_seconds = float(profile['pbs'].get('idle_release_seconds', 5.))
        self.idle_since = None
        if self.drain_seconds < 0 or self.minimum_task_seconds <= 0:
            raise ValueError('Nonnegative drain and positive task time required')
        if self.idle_release_seconds < 0:
            raise ValueError('Nonnegative idle release grace required')
        self.last_heartbeat = -1.
        self.draining = False
        self.exit_reason = None
        self.start_clock = stamp()
        transport = SpoolTransport(directory, deadline=deadline,
                                   timeout_seconds=float(profile['pbs'].get('rpc_timeout_seconds', 15.)),
                                   heartbeat=self.heartbeat)
        super().__init__(profile, 'http://127.0.0.1:1',
                         poll_seconds=poll_seconds, transport=transport)

    @property
    def capacity_state_dir(self):
        # Attempts stay in the shared spool. Throughput trials belong to the
        # allocation whose CPU and memory limits they were measured under.
        return self.directory

    def control(self):
        path = self.directory / 'CONTROL.json'
        return read(path) if path.exists() else {}

    def _validate_acquisition(self, assignment):
        deployment=self.profile['pbs'].get('deployment')
        if deployment:
            from .rpe_release_deployment import validate_assignment_release
            validate_assignment_release(read(deployment['manifest']),read(deployment['layout']),
                                        assignment,self.profile)

    def _check_challenges(self, challenges, report):
        if not challenges:
            return super()._check_challenges(challenges, report)
        started = time.monotonic()
        parallel = max(1, min(report['slots'], 4))
        result = host_fingerprints(self.profile, challenges, q3_parallel=parallel)
        event(self.profile, 'PBS_READINESS_CHECKED', jobs=[item['job_id'] for item in challenges],
              parallel=parallel, passed=len(result['host_fingerprints']),
              seconds=time.monotonic()-started)
        return result

    def _receive_assignment(self, assignment):
        # A later request's validation, the next status RPC, or a completed
        # upload must not delay a request whose exact assignment is durable.
        super()._receive_assignment(assignment)
        attempt = self.work/assignment['job_id']/assignment['attempt_id']
        if read(attempt/'WORKER_STATE.json')['state'] == 'ASSIGNED':
            self._launch(attempt)

    def resource_snapshot(self):
        snapshot = super().resource_snapshot()
        # This cluster does not expose a per-job cpuset/cgroup. Its qstat
        # Resource_List is nevertheless a finite scheduling boundary, already
        # injected into resource_limits. Preserve all tighter process limits;
        # missing cgroup observations remain reported separately.
        actual = self.pbs_resources['actual']
        snapshot['allowed_cpu_count'] = min(snapshot.get('allowed_cpu_count') or actual['ncpus'], actual['ncpus'])
        snapshot['allowed_memory_bytes'] = min(snapshot.get('allowed_memory_bytes') or actual['memory_bytes'], actual['memory_bytes'])
        snapshot['allocation_bounds_complete'] = True
        snapshot['boundary_detection'] = 'ACTUAL_PBS_SINGLE_NODE_CPU_AND_MEMORY_WITH_PROCESS_LIMITS'
        snapshot['pbs_allocation'] = dict(job_id=actual['job_id'], ncpus=actual['ncpus'],
                                          memory_bytes=actual['memory_bytes'])
        snapshot['memory_scope'] = 'BOUNDED_PBS_ALLOCATION'
        return snapshot

    def remaining(self):
        return max(0., self.deadline - time.monotonic())

    def accepting(self):
        return not self.draining and not self.stop.is_set() and self.remaining() > self.drain_seconds + self.minimum_task_seconds

    def heartbeat(self, *, state=None, force=False):
        now = time.monotonic()
        if not force and now - self.last_heartbeat < 1.:
            return
        self.last_heartbeat = now
        control = self.control()
        if control.get('stop') or self.remaining() <= self.drain_seconds + self.minimum_task_seconds:
            self.draining = True
        value = dict(stamp(), schema='RPE_PBS_WORKER_STATE_V1',
                     state=state or ('DRAINING' if self.draining else 'READY'),
                     ready=state != 'EXITED', accepting=self.accepting(),
                     host_id=self.host_id, job_id=self.pbs_resources['actual']['job_id'],
                     hostname=socket.gethostname(), pid=os.getpid(),
                     source_sha=self.profile['pbs']['worker_source'],
                     resources=self.pbs_resources, capacity=self.slots,
                     remaining_seconds=self.remaining(),
                     start_clock=self.start_clock,
                     exit_reason=self.exit_reason)
        atomic(self.directory / 'STATE.json', value)

    def _post(self, endpoint, payload):
        if endpoint == '/v1/poll':
            payload = deepcopy(payload)
            pbs = dict(job_id=self.pbs_resources['actual']['job_id'],
                       remaining_seconds=self.remaining(),
                       minimum_task_seconds=self.minimum_task_seconds,
                       drain_seconds=self.drain_seconds, accepting=self.accepting())
            if 'sized_task_memory_bytes' in self.profile['pbs']:
                pbs['sized_task_memory_bytes'] = self.profile['pbs']['sized_task_memory_bytes']
            if self.profile['pbs'].get('per_task_resources'):
                pbs['resource_pool'] = {key: self.pbs_resources[key] for key in ('cpu_count', 'memory_bytes')}
            payload.setdefault('resource_report', {})['pbs'] = pbs
            if not pbs['accepting']:
                payload['slots'] = 0
        return super()._post(endpoint, payload)

    def _launch(self, attempt):
        # A slow transport reply can arrive after the last admission check. Keep
        # its original attempt and make the non-launch explicit; never roll its
        # reservation back or silently create another scientific attempt.
        if self.control().get('stop') or not self.accepting():
            save_state(attempt, 'UNKNOWN', reason='PBS_ADMISSION_DEADLINE_BEFORE_LAUNCH',
                       scientific_execution_started=False, automatic_retry=False)
            event(self.profile, 'PBS_ADMISSION_HOLD', job_id=Path(attempt).parent.name,
                  attempt_id=Path(attempt).name, remaining_seconds=self.remaining())
            return
        super()._launch(attempt)

    def run_allocation(self):
        self.heartbeat(force=True)
        try:
            while self.remaining() > 0:
                control = self.control()
                if control.get('stop') or self.stop.is_set():
                    self.draining = True
                    self.exit_reason = self.exit_reason or 'CONTROLLER_STOP_DRAIN'
                if self.remaining() <= self.drain_seconds + self.minimum_task_seconds:
                    self.draining = True
                    self.exit_reason = self.exit_reason or 'WALLTIME_ADMISSION_CUTOFF'
                self.tick(acquire=self.accepting())
                self.heartbeat()
                if self.draining and not self.pending():
                    break
                if control.get('release_when_idle') and not self.pending():
                    if self.idle_since is None:
                        self.idle_since = time.monotonic()
                    if time.monotonic() - self.idle_since >= self.idle_release_seconds:
                        self.exit_reason = 'CONTROLLER_IDLE_RELEASE'
                        break
                else:
                    self.idle_since = None
                time.sleep(min(self.poll_seconds, self.remaining()))
            if self.exit_reason is None:
                self.exit_reason = 'ALLOCATION_WALLTIME_END'
        finally:
            self.draining = True
            self.heartbeat(state='EXITED', force=True)
            # No signal is sent to a scientific child; PBS owns the remaining
            # allocation lifetime. Original receipts/UNKNOWN policy is retained.
            self.close()


def run(profile_path, *, allocation_dir=None, poll_seconds=1.):
    profile = read(profile_path)
    deployment=profile['pbs'].get('deployment')
    if deployment:
        from .rpe_release_deployment import ensure_components
        layout=read(deployment['layout'])
        installed=ensure_components(read(deployment['manifest']),layout,
            components=layout.get('participation_components',['optimizer','pbs_worker','pbs_gateway']))
        if installed['status']!='PASS':
            raise ValueError('PBS release deployment HOLD: '+json.dumps(installed['holds']))
    if allocation_dir is not None:
        profile['pbs']['allocation_dir'] = str(Path(allocation_dir).resolve())
    identity = fixed_worker(profile)
    directory = Path(profile['pbs']['allocation_dir']).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    allocation = read(directory / 'ALLOCATION.json')
    remaining = min(float(allocation.get('remaining_seconds', allocation['walltime_seconds'])),
                    float(allocation['walltime_seconds']))
    deadline = time.monotonic() + remaining
    resolved, resources = allocation_profile(profile, allocation)
    for variable in THREAD_VARIABLES:
        os.environ[variable] = str(resources['library_threads'])
    runtime = {}
    for name, choice in runtime_profiles(resolved):
        runtime[name] = check_runtime_profile(choice)
        if runtime[name]['status'] != 'PASS':
            raise ValueError('PBS stage runtime checks failed: ' + name)
        for field, expected in (choice.get('runtime') or {}).items():
            if runtime[name]['runtime'].get(field) != expected:
                raise ValueError('PBS stage runtime contract differs: ' + name + '/' + field)
    atomic(directory / 'STARTUP.json', dict(stamp(), identity=identity, runtime=runtime,
                                           resources=resources, scientific_execution=False))
    worker = AllocationWorker(resolved, directory=directory,
                              deadline=deadline,
                              resources=resources, poll_seconds=poll_seconds)
    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, lambda *_: worker.stop.set())
    worker.run_allocation()
    return 0
