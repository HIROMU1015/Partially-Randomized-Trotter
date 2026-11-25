"""One PBS allocation hosting the existing durable, source-bound pull worker.

Only the worker runs on a compute node.  The login-node gateway copies durable
JSON messages; it never executes an assigned scientific task.  Allocation exit
leaves the ordinary worker spool and per-task receipts on persistent storage.
"""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from itertools import groupby
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
import threading
import time
import uuid

from . import rpe_q3_resources as Resources
from .rpe_host_profiles import check_runtime_profile
from .rpe_preparation_artifacts import worker_cache
from .rpe_preparation_worker import missing_metadata_bundle_hashes
from .rpe_q3_timing import stamp
from ._collection_q3_worker import Worker, atomic, digest, event, head, read, save_state, resource_state
from .rpe_q3_worker import host_fingerprints


THREAD_VARIABLES = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
                    'NUMBA_NUM_THREADS', 'RAYON_NUM_THREADS', 'QISKIT_NUM_PROCS')
TRANSPORT_FILES = ('src/trottertracks/rpe_pbs_worker.py',
                   'scripts/run_rpe_pbs_worker.py')
Q3_WORKER_PROOF_SECONDS = 120
READINESS_WORKERS = 4
MIN_PARALLEL_ACQUISITION_SECONDS = .01


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
                 poll_seconds=.1, heartbeat=None, bounded_results=False):
        self.directory = Path(directory)
        self.requests = self.directory / 'requests'
        self.responses = self.directory / 'responses'
        self.requests.mkdir(parents=True, exist_ok=True)
        self.responses.mkdir(parents=True, exist_ok=True)
        self.deadline = deadline
        self.timeout_seconds = timeout_seconds
        self.poll_seconds = poll_seconds
        self.heartbeat = heartbeat
        self.bounded_results = bounded_results

    def __call__(self, endpoint, payload):
        evidence_batch = endpoint == '/v1/complete' and 'pbs_evidence_status_batch' in payload
        completion_prefix = ('complete-' + digest([payload.get('job_id'), payload.get('attempt_id')])[:20] + '-'
                             if endpoint == '/v1/complete' else '')
        previous_prefix = ()
        if evidence_batch:
            completion_prefix = 'complete-evidence-'
        elif endpoint == '/v1/complete' and 'pbs_evidence' in payload:
            previous_prefix = (completion_prefix,)
            completion_prefix = 'complete-q3-' + completion_prefix[len('complete-'):]
        if self.bounded_results and endpoint == '/v1/artifacts/put':
            completion_prefix = 'artifact-put-' + digest(sorted(payload['blobs']))[:20] + '-'
        matching = []
        for path in self.requests.glob('*.json'):
            if path.name.startswith(('complete-', 'artifact-put-')) and (
                    not completion_prefix or not path.name.startswith((completion_prefix,) + previous_prefix)):
                continue
            try:
                saved = read(path)
            except FileNotFoundError:
                continue  # Another attempt consumed its reply.
            if saved['endpoint'] == endpoint and (endpoint == '/v1/poll' or evidence_batch
                                                  or saved['payload'] == payload):
                matching.append((path, saved))
        # Exactly one unconsumed acquisition may exist across timeouts and
        # process restarts. A fresh UUID here could reserve another task before
        # the first reply has reached the worker.
        if endpoint == '/v1/poll' and len(matching) > 1:
            raise ValueError('Multiple unconsumed PBS acquisition requests require reconciliation')
        if matching:
            request, saved = matching[0]
            request_id = saved['request_id']
            payload = saved['payload']
        else:
            request_id = completion_prefix + uuid.uuid4().hex
            request = self.requests / (request_id + '.json')
            atomic(request, dict(request_id=request_id, endpoint=endpoint,
                                 payload=payload, clock=stamp()))
        response = self.responses / (request_id + '.json')
        limit = min(self.deadline, time.monotonic() + self.timeout_seconds)
        service = getattr(self, 'service_during_poll', None) if endpoint == '/v1/poll' else None
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
                if endpoint == '/v1/poll':
                    # An overdue response acknowledges the durable request's
                    # proofs, which may differ from the caller's latest offer.
                    return dict(value['result'], submitted_readiness=payload.get('readiness_checked_monotonic', {}))
                return value['result']
            if time.monotonic() >= limit:
                raise TimeoutError('PBS gateway response remains pending: ' + request_id)
            if service is not None:
                service()
            time.sleep(min(self.poll_seconds, max(0, limit - time.monotonic())))


class AllocationWorker(Worker):
    inline_readiness_retry = False

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
                                   heartbeat=self.heartbeat, bounded_results=profile['pbs'].get('bounded_results', False))
        super().__init__(profile, 'http://127.0.0.1:1',
                         poll_seconds=poll_seconds, transport=transport)
        count = profile['pbs'].get('collection_workers', 0)
        if type(count) is not int or count < 0:
            self.close()
            raise ValueError('PBS collection_workers must be a nonnegative integer')
        self.collection = (ThreadPoolExecutor(max_workers=count, thread_name_prefix='pbs-collect')
                           if count else None)
        self.collection_workers = count
        self.collection_inflight = {}
        self.evidence_batch_inflight = None
        self.next_collection_scan = 0.
        self.launch_tracking = set(self.records)
        self.memory_backfill = self.profile['pbs'].get('q3_memory_backfill') is True
        self.memory_history_path = self.work/'Q3_MEMORY_HISTORY.json'
        self.memory_history = read(self.memory_history_path) if self.memory_backfill and self.memory_history_path.is_file() else {}
        self.memory_checked = set()
        self.backfill_head = None
        self.backfills_left = 0
        for attempt, assignment in (self.records.items() if self.memory_backfill else ()):
            if (attempt/'OWNED_RESOURCES.json').is_file() and (attempt/'PROCESS_RETURN.json').is_file():
                self._learn_q3_memory(attempt, assignment)
        if self.memory_backfill:
            event(self.profile, 'PBS_Q3_MEMORY_HISTORY', profiles=len(self.memory_history),
                  calibrated=sum(len(row['jobs']) == 2 for row in self.memory_history.values()))
        self.buffer_cpu = 0
        self.next_assignment_poll = 0.
        self.offline_search = None
        self.offline_active = False
        self.last_central_contact = time.monotonic()
        self.central_contact = None
        if reference := self.profile['pbs'].get('offline_plan'):
            from .rpe_pbs_offline import OfflineSearch
            try:
                (self.directory/'OFFLINE_ACTIVE').unlink(missing_ok=True)
                raw = Path(reference['path']).read_bytes()
                if hashlib.sha256(raw).hexdigest() != reference['sha256']:
                    raise ValueError('PBS continuation plan digest differs')
                self.offline_search = OfflineSearch(json.loads(raw), self.profile, self.directory, self.pbs_resources)
            except (OSError, KeyError, ValueError) as error:
                event(self.profile, 'PBS_OFFLINE_SEARCH_UNAVAILABLE', reason=str(error))
        transport.service_during_poll = self._pump_collection
        self.preparation_transfers = set()
        self.dispatch_wakeup = threading.Event()
        self._open_readiness()

    def _open_readiness(self):
        self.readiness_pool = ThreadPoolExecutor(max_workers=READINESS_WORKERS,
                                                thread_name_prefix='pbs-readiness')
        self.readiness_pending = {}
        self.readiness_running = {}
        self.readiness_ready = {}
        self.readiness_retry = {}
        self.readiness_sent = {}
        self.readiness_lock = threading.Lock()
        self.readiness_stop = threading.Event()
        self.readiness_thread = None
        self.readiness_offer = (0, False)
        self.readiness_requests = {}

    def _refill_readiness(self):
        while not self.readiness_stop.wait(.2):
            with self.readiness_lock:
                slots, accepting = self.readiness_offer
                if (accepting and self.accepting() and not self.offline_active
                        and not self.control().get('stop')):
                    self._advance_readiness(slots, True)

    def tick(self, acquire=True):
        if getattr(self, 'collection', None) is None:
            return super().tick(acquire=acquire)
        self._pump_collection()
        if (self.profile['pbs'].get('standby_waves') and self.buffer_cpu >= self.pbs_resources['cpu_count']//2
                and time.monotonic() < self.next_assignment_poll):
            return None  # Retain the normal completion lane while using the buffered work.
        result = self._poll_assignments(acquire)
        self.next_assignment_poll = time.monotonic() + max(2., self.poll_seconds*4)
        self._launch_buffered()
        return result

    def _offline_tick(self, control):
        offline = getattr(self, 'offline_search', None)
        if offline is None:
            return False
        now = time.monotonic()
        contact = control.get('contact')
        if contact is not None and contact != self.central_contact:
            self.last_central_contact, self.central_contact = now, contact
        disconnected = (now-self.last_central_contact >= 30 and self.accepting() and not control.get('stop'))
        if disconnected != self.offline_active:
            marker = self.directory/'OFFLINE_ACTIVE'
            if disconnected:
                marker.touch()
            else:
                marker.unlink(missing_ok=True)
            self.offline_active = disconnected
        if not disconnected and offline.process is None:
            return False
        self._launch_buffered()
        active = False
        for attempt in tuple(self.launch_tracking):
            state = read(attempt/'WORKER_STATE.json')
            if (state['state'] in ('STARTING', 'RUNNING', 'UNKNOWN')
                    and attempt not in self.collection_inflight):
                state = self._reconcile(attempt, self.records[attempt])
            active |= (state['state'] in ('ASSIGNED', 'STARTING', 'RUNNING')
                       or state['state'] == 'UNKNOWN'
                       and not state.get('detail', {}).get('resource_released')
                       and not self._known_decline(attempt, state))
        storage = (self.resource_report or {}).get('storage') or {}
        try:
            if disconnected and not active:
                offline.reserve_storage(storage)
            started = offline.advance(disconnected=disconnected,
                runnable=(not active and self.resource_report is not None
                          and storage.get('accepting_new_jobs', True)
                          and self.remaining() >= self.drain_seconds+2*self.minimum_task_seconds),
                batch=max(1, self.slots))
            return started or disconnected
        except (OSError, KeyError, ValueError, subprocess.SubprocessError) as error:
            offline.recheck_at = now+60
            event(self.profile, 'PBS_OFFLINE_SEARCH_PENDING', reason=str(error))
            return disconnected or offline.process is not None

    def _pump_collection(self):
        if getattr(self, 'collection', None) is None:
            return
        now = time.monotonic()
        if now < getattr(self, 'next_collection_scan', 0.):
            return
        self.next_collection_scan = now + .5
        self._launch_buffered()
        for attempt, future in list(self.collection_inflight.items()):
            if future.done():
                del self.collection_inflight[attempt]
                self.preparation_transfers.discard(attempt)
                future.result()
        batch = getattr(self, 'evidence_batch_inflight', None)
        if batch is not None and batch.done():
            batch.result()
            self.evidence_batch_inflight = batch = None
        # Prefer ready receipts. Every attempt still reports state, transfers,
        # and marks its own ACK in order using the ordinary worker code.
        available = self.collection_workers-len(self.collection_inflight)-int(batch is not None)
        if available:
            candidates = []
            evidence = []
            batched = self.profile.get('pbs', {}).get('batch_q3_completions') is True
            for attempt, assignment in self.records.items():
                if attempt in self.collection_inflight or attempt in self.transferred_attempts:
                    continue
                if now < getattr(self, 'pbs_evidence_checks', {}).get(attempt, 0):
                    continue
                state = read(attempt/'WORKER_STATE.json')
                if state['state'] == 'ASSIGNED' and assignment.get('pbs_launch_resources'):
                    continue
                if state['state'] == 'TRANSFERRED':
                    self.transferred_attempts.add(attempt)
                    continue
                if (state['state'] == 'RUNNING' and self.reported.get(attempt) == state and
                        not (attempt/'PROCESS_RETURN.json').exists()):
                    process = self.processes.get(attempt)
                    if process is not None and process.poll() is None:
                        continue
                complete = state['state'] == 'COMPLETED_PENDING_TRANSFER'
                if complete and batched and (attempt/'PBS_CORE_ACK.json').is_file():
                    evidence.append((attempt, assignment))
                    continue
                if complete and getattr(self, 'memory_backfill', False):
                    self._learn_q3_memory(attempt, assignment)
                candidates.append((not complete, attempt, assignment, state))
            if evidence and batch is None and (available > 1 or not candidates):
                evidence.sort(key=lambda entry: self.pbs_evidence_checks.get(entry[0], 0))
                self.evidence_batch_inflight = self.collection.submit(
                    self._check_deferred_evidence_batch, evidence[:max(1, self.slots)])
                available -= 1
            if not available:
                return
            for incomplete, attempt, assignment, state in sorted(candidates):
                preparation = 'preparation' in assignment['request']
                transfer = not preparation or (not incomplete and not self.preparation_transfers)
                if preparation and not transfer and not incomplete and self.reported.get(attempt) == state:
                    continue
                # One bulk transfer need not delay state notifications or the
                # launch of another preparation. Both use the same attempts.
                self.collection_inflight[attempt] = self.collection.submit(
                    self._service_attempt, attempt, assignment, transfer=transfer)
                if preparation and transfer:
                    self.preparation_transfers.add(attempt)
                available -= 1
                if available == 0:
                    break

    def close(self):
        self._close_readiness()
        self._close_collection()
        super().close()

    def _close_readiness(self):
        if getattr(self, 'readiness_thread', None) is not None:
            self.readiness_stop.set()
            self.readiness_thread.join()
            self.readiness_thread = None
        if getattr(self, 'readiness_pool', None) is not None:
            self.readiness_pool.shutdown(wait=True)
            self.readiness_pool = None

    def _close_collection(self):
        if getattr(self, 'collection', None) is not None:
            self.collection.shutdown(wait=True)
            self.collection = None

    @property
    def capacity_state_dir(self):
        # Attempts stay in the shared spool. Throughput trials belong to the
        # allocation whose CPU and memory limits they were measured under.
        return self.directory

    def control(self):
        path = self.directory / 'CONTROL.json'
        return read(path) if path.exists() else {}

    def _validate_acquisition(self, assignment):
        settings = self.profile['pbs']
        release = (assignment.get('required_release') or {}).get('id')
        deployment = settings.get('task_deployments', {}).get(release, settings.get('deployment'))
        if deployment:
            self._validate_release_assignment(deployment, assignment)

    def _validate_release_assignment(self, deployment, assignment):
        from .rpe_release_deployment import validate_assignment_release
        cache = getattr(self, 'batch_input_hashes', None)
        args = read(deployment['manifest']), read(deployment['layout']), assignment, self.profile
        if cache is None:
            validate_assignment_release(*args)
        else:
            validate_assignment_release(*args, input_hash_cache=cache)

    def _receive_assignments(self, assignments):
        # Reuse only immutable Git objects within one reply and compare the
        # resulting digest with each assignment, keeping the other checks.
        self.batch_input_hashes = {}
        try:
            # A known durable replay may be needed after a lost reply, but
            # should never sit in front of fresh work in the same batch.
            ordered = sorted(assignments, key=lambda item:
                self.work/item['job_id']/item['attempt_id'] in self.records)
            for is_q3, group in groupby(ordered, self._fresh_q3_assignment):
                if is_q3:
                    self._receive_fresh_q3(list(group))
                else:
                    for assignment in group:
                        self._receive_assignment(assignment)
        finally:
            self.batch_input_hashes = None

    def _fresh_q3_assignment(self, assignment):
        attempt = self.work / assignment['job_id'] / assignment['attempt_id']
        return 'preparation' not in assignment['request'] and not (attempt/'ASSIGNMENT.json').exists()

    def _receive_fresh_q3(self, assignments):
        # Start the priority task immediately. It also warms the batch cache
        # for immutable inputs shared with the rest of the reply.
        if len(assignments) <= 2:
            for assignment in assignments:
                self._receive_assignment(assignment)
            return
        check_seconds = 0.

        def check_first(assignment):
            nonlocal check_seconds
            started = time.monotonic()
            self._validate_acquisition(assignment)
            check_seconds = time.monotonic() - started

        self._receive_assignment(assignments[0], validate=check_first)
        if check_seconds < MIN_PARALLEL_ACQUISITION_SECONDS:
            for assignment in assignments[1:]:
                self._receive_assignment(assignment)
            return
        # Only the read-only host/release verification runs in the pool.
        # The poll thread alone commits ownership and starts supervisors.
        first_error = None
        with ThreadPoolExecutor(max_workers=READINESS_WORKERS) as checks:
            pending = {checks.submit(self._validate_acquisition, assignment): assignment
                       for assignment in assignments[1:]}
            while pending:
                complete, _ = wait(pending, timeout=.25, return_when=FIRST_COMPLETED)
                for verified in complete:
                    assignment = pending.pop(verified)
                    try:
                        verified.result()
                    except (OSError, ValueError, subprocess.CalledProcessError) as error:
                        if first_error is None:
                            first_error = ValueError(str(error))
                        continue
                    self._receive_assignment(assignment, validate=lambda _: verified.result())
                self._pump_collection()
        if first_error is not None:
            raise first_error

    @staticmethod
    def _readiness_key(challenge):
        return (challenge['request_sha256'], challenge.get('host_execution_profile_sha256'),
                digest({name: challenge.get(name) for name in
                        ('artifact_manifest', 'q3_evaluator', 'required_release')}))

    def _run_readiness_check(self, challenge, kind):
        started = time.monotonic()
        job = challenge['job_id']
        try:
            if kind == 'transfer':
                self._prepare_challenge_artifacts([challenge])
            checked = host_fingerprints(self.profile, [challenge])
            proof = checked['host_fingerprints'].get(job)
            error = checked['readiness_errors'].get(job)
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
            proof, error = None, f'{type(exc).__name__}: {exc}'
        return started, time.monotonic(), proof, error

    def _discard_readiness(self):
        self.readiness_pending.clear()
        self.readiness_ready.clear()
        # A result from before a pause belongs to the previous offer.
        self.readiness_running = {job: (None, kind, future)
                                  for job, (_, kind, future) in self.readiness_running.items()}

    def _poll_readiness(self, report, accepting):
        with self.readiness_lock:
            self.readiness_offer = report['slots'], accepting
            self._advance_readiness(report['slots'], accepting and self.accepting())
            facts, errors, checked_at = {}, {}, {}
            for job, (_, _, finished_at, proof, error) in self.readiness_ready.items():
                checked_at[job] = finished_at
                if error is None:
                    facts[job] = proof
                else:
                    errors[job] = error
            self.readiness_sent = checked_at
            return dict(host_fingerprints=facts, readiness_errors=errors,
                        readiness_checked_monotonic=checked_at,
                        pbs_cached_challenges={job: entry[0] for job, entry in self.readiness_requests.items()})

    def _advance_readiness(self, slots, accepting):
        # Keep the next check moving even while execution fills the allocation.
        limit = min(max(1, slots), READINESS_WORKERS)
        now = time.monotonic()
        self.readiness_retry = {key: until for key, until in self.readiness_retry.items() if until > now}
        if not accepting:
            self._discard_readiness()
        for job, (_, _, touched) in list(self.readiness_pending.items()):
            if job not in self.readiness_running and now-touched >= Q3_WORKER_PROOF_SECONDS:
                del self.readiness_pending[job]
                self.readiness_ready.pop(job, None)
        for job, (_, _, checked_at, _, _) in list(self.readiness_ready.items()):
            if now-checked_at >= Q3_WORKER_PROOF_SECONDS:
                del self.readiness_ready[job]
        finished = []
        for job, (key, _, future) in list(self.readiness_running.items()):
            if not future.done():
                continue
            del self.readiness_running[job]
            pending = self.readiness_pending.get(job)
            checked = future.result()
            if pending is not None and pending[0] == key and now-checked[1] < Q3_WORKER_PROOF_SECONDS:
                self.readiness_ready[job] = (key, *checked)
                finished.append(job)
        if finished:
            self.dispatch_wakeup.set()
            results = [self.readiness_ready[job] for job in finished]
            event(self.profile, 'PBS_READINESS_CHECKED', jobs=finished,
                  parallel=limit,
                  passed=sum(proof is not None for _, _, _, proof, _ in results),
                  seconds=round(max(end-start for _, start, end, _, _ in results), 6), background=True)

        preparation_limit = max(1, limit-1)
        preparation_running = sum(kind != 'q3' for _, kind, _ in self.readiness_running.values())
        transferring = any(kind == 'transfer' for _, kind, _ in self.readiness_running.values())
        cache = worker_cache(self.profile)
        started = {}
        for job, (key, challenge, _) in self.readiness_pending.items():
            if len(self.readiness_running) >= limit:
                break
            if job in self.readiness_ready or job in self.readiness_running or (job, key) in self.readiness_retry:
                continue
            kind = 'q3'
            if challenge['request'].get('route', {}).get('kind') == 'preparation':
                if preparation_running >= preparation_limit:
                    continue
                try:
                    needs_transfer = any(not cache.path(sha).is_file() for sha in missing_metadata_bundle_hashes(challenge, self.profile))
                except (OSError, ValueError, KeyError, TypeError):
                    # The full input check reports the malformed requirement.
                    needs_transfer = False
                if needs_transfer and transferring:
                    continue
                kind = 'transfer' if needs_transfer else 'cached'
                preparation_running += 1
                transferring |= needs_transfer
            future = self.readiness_pool.submit(self._run_readiness_check, challenge, kind)
            self.readiness_running[job] = (key, kind, future)
            started[job] = kind
        if started:
            event(self.profile, 'PBS_READINESS_STARTED', jobs=started, pending=len(self.readiness_pending))

    def _readiness_delivered(self, response):
        with self.readiness_lock:
            self._apply_readiness_response(response)

    def _apply_readiness_response(self, response):
        now = time.monotonic()
        assigned = {item['job_id'] for item in response.get('assignments', [])}
        if assigned:
            checking = sum(not future.done() for _, _, future in self.readiness_running.values())
            if checking:
                event(self.profile, 'PBS_ASSIGNED_DURING_READINESS', jobs=sorted(assigned),
                      checks_in_flight=checking)
        submitted = response.get('submitted_readiness', self.readiness_sent)
        reported = {job for job, checked_at in submitted.items()
                    if job in self.readiness_ready and self.readiness_ready[job][2] == checked_at}
        for job in reported:
            key, _, _, _, error = self.readiness_ready.pop(job)
            if error is not None:
                self.readiness_retry[job, key] = now+max(1., self.poll_seconds*2)
        for job in assigned | reported:
            self.readiness_pending.pop(job, None)
        if not self.accepting() or response.get('host_admission', {}).get('accepting') is False:
            self.readiness_offer = (0, False)
            self._discard_readiness()
            return
        for challenge in response.get('eligibility_challenges', []):
            job = challenge['job_id']
            if job in assigned:
                continue
            ref = challenge.get('pbs_challenge_ref')
            cached = self.readiness_requests.pop(job, None)
            if ref is not None:
                if cached is None or cached[0] != ref:
                    event(self.profile, 'PBS_READINESS_REFERENCE_MISS', job_id=job)
                    continue
                challenge = cached[1]
            if challenge['request'].get('route', {}).get('kind') != 'preparation':
                self.readiness_requests[job] = (ref or digest(challenge), challenge)
                if len(self.readiness_requests) > 256:
                    del self.readiness_requests[next(iter(self.readiness_requests))]
            key = self._readiness_key(challenge)
            self.readiness_pending[job] = (key, challenge, now)
            if job in self.readiness_ready and self.readiness_ready[job][0] != key:
                del self.readiness_ready[job]

    def _receive_assignment(self, assignment, *, validate=None):
        # A later request's validation, the next status RPC, or a completed
        # upload must not delay a request whose exact assignment is durable.
        self._acquire(assignment, validate=validate)
        attempt = self.work/assignment['job_id']/assignment['attempt_id']
        if hasattr(self, 'launch_tracking'):
            self.launch_tracking.add(attempt)
        if (not assignment.get('pbs_launch_resources')
                and read(attempt/'WORKER_STATE.json')['state'] == 'ASSIGNED'):
            self._launch(attempt)

    @staticmethod
    def _known_decline(attempt, state):
        detail = state.get('detail', {})
        return (detail.get('reason') == 'PBS_ADMISSION_DEADLINE_BEFORE_LAUNCH'
                and detail.get('scientific_execution_started') is False
                and detail.get('automatic_retry') is False
                and not any((attempt/name).exists() for name in (
                    'SUPERVISOR_STARTED.json', 'SUPERVISOR_LAUNCH.json',
                    'WORKER_LAUNCH.json', 'SCIENTIFIC_STARTING.json', 'PROCESS_RETURN.json')))

    @staticmethod
    def _q3_memory_key(assignment, profile):
        request = assignment['request']
        route = request.get('route') or {}
        resources = assignment.get('pbs_launch_resources') or {}
        if ('preparation' in request or not request.get('N_X') or not request.get('N_Y')
                or not resources.get('cpu') or not profile.get('interpreter')):
            return None
        return digest([request['method'], request['N_X'], request['N_Y'], request.get('optimizer_source'),
                       route.get('kind'), route.get('view_commit'), route.get('horizon_risk_source'),
                       assignment.get('q3_evaluator'), profile['interpreter'],
                       (profile.get('adapter_manifest') or {}).get('commit'), profile.get('dp_workers'),
                       resources['cpu']])

    def _learn_q3_memory(self, attempt, assignment):
        if attempt in self.memory_checked or 'preparation' in assignment['request']:
            return
        self.memory_checked.add(attempt)
        try:
            receipt = read(attempt/'PROCESS_RETURN.json')
            if receipt.get('returncode') != 0 or receipt.get('signal') is not None:
                return
            result = read(attempt/'OUTCOME.json')
            if result.get('status') not in ('MODEL_BOUND_MEETS_ADOPTED_THRESHOLD',
                    'MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD', 'INCONCLUSIVE'):
                return
            key = self._q3_memory_key(assignment, read(attempt/'WORKER_PROFILE.json'))
            peak = read(attempt/'OWNED_RESOURCES.json')['peak_rss_bytes']
            if key is None or type(peak) is not int or peak <= 0:
                return
            row = self.memory_history.setdefault(key, {'jobs': [], 'peak_rss_bytes': 0})
            job = assignment['job_id']
            if job not in row['jobs'] and len(row['jobs']) < 2:
                row['jobs'].append(job)
            row['peak_rss_bytes'] = max(peak, row['peak_rss_bytes'])
            atomic(self.memory_history_path, self.memory_history)
        except (OSError, KeyError, ValueError, TypeError):
            return  # Missing telemetry leaves the original reservation in force.

    def _q3_memory_reservation(self, attempt, resources):
        if getattr(self, 'memory_backfill', False) and self.memory_history:
            key = self._q3_memory_key(self.records[attempt], read(attempt/'WORKER_PROFILE.json'))
            row = self.memory_history.get(key, {})
            if len(row.get('jobs', ())) == 2:
                return max(1024**3, (3*row['peak_rss_bytes']+1)//2,
                           self.profile['pbs'].get('task_memory_bytes') or 0), True
        return resources['memory_bytes'], False

    def _launch_buffered(self):
        pending = []
        slots = self.slots
        cpus = self.pbs_resources['cpu_count']
        memory = self.pbs_resources['memory_bytes'] - 1024**3
        reserved_storage = 0
        for attempt in tuple(self.launch_tracking):
            assignment = self.records[attempt]
            state = read(attempt/'WORKER_STATE.json')
            status = state['state']
            resources = assignment.get('pbs_launch_resources')
            if status == 'ASSIGNED' and resources:
                pending.append((attempt, resources))
            elif status in ('ASSIGNED', 'STARTING', 'RUNNING', 'UNKNOWN') and (
                    not state.get('detail', {}).get('resource_released')
                    and not (status == 'UNKNOWN' and self._known_decline(attempt, state))
                    and resource_state(attempt, status) != 'EXITED'):
                slots -= 1
                reserved_storage += (resources or {}).get('storage_bytes', 0)
                cpus -= resources['cpu'] if resources else self.pbs_resources['cpu_count']
                if resources and getattr(self, 'memory_backfill', False):
                    reserved, _ = self._q3_memory_reservation(attempt, resources)
                    sample = attempt/'OWNED_RESOURCES.json'
                    memory -= max(reserved, read(sample).get('peak_rss_bytes', 0) if sample.is_file() else 0)
                else:
                    memory -= resources['memory_bytes'] if resources else self.pbs_resources['memory_bytes']
            else:
                self.launch_tracking.discard(attempt)
        self.buffer_cpu = sum(resources['cpu'] for _, resources in pending)
        if not pending:
            return
        paused_only=(getattr(self,'paused',False) and not self.draining and not self.stop.is_set()
                     and self.remaining() > self.drain_seconds+self.minimum_task_seconds)
        stopping = self.control().get('stop') or not (self.accepting() or paused_only)
        storage = (self.resource_report or {}).get('storage') or {}
        if not stopping and (self.resource_report is None or storage.get('accepting_new_jobs') is False):
            return
        # Non-buffered assignments retain priority. Independent buffered Q3s
        # can fill holes without waiting for the next gateway response.
        pending.sort(key=lambda item: (item[1]['standby'], item[1]['order']))
        blocked = None
        if not any(attempt == getattr(self, 'backfill_head', None) for attempt, _ in pending):
            self.backfill_head = None
        for attempt, resources in pending:
            if stopping or self.remaining() < self.drain_seconds + resources['seconds']:
                self._decline_before_launch(attempt)
                self.buffer_cpu -= resources['cpu']
                continue
            if slots <= 0 or resources['cpu'] > cpus:
                continue
            needed = resources.get('storage_bytes', 0)
            if needed and storage:
                used, limit = storage.get('used_bytes'), storage.get('limit_bytes')
                free, minimum = storage.get('filesystem_free_bytes'), storage.get('filesystem_min_free_bytes')
                if (type(used) is not int or type(limit) is not int or used + reserved_storage + needed > limit
                        or (type(free) is int and type(minimum) is int and free-reserved_storage-needed < minimum)):
                    continue
            request = self.records[attempt].get('request', {})
            q3 = getattr(self, 'memory_backfill', False) and 'preparation' not in request
            reserved, calibrated = self._q3_memory_reservation(attempt, resources)
            if reserved > memory:
                if q3 and blocked is None and reserved <= self.pbs_resources['memory_bytes']-1024**3:
                    blocked = request.get('configuration_id', attempt.parent.name)
                    if self.backfill_head != attempt:
                        self.backfill_head, self.backfills_left = attempt, slots
                continue
            if blocked is not None and q3 and (not calibrated or self.backfills_left == 0
                    or request.get('configuration_id', attempt.parent.name) == blocked):
                continue
            self._launch(attempt)
            self.buffer_cpu -= resources['cpu']
            slots -= 1
            cpus -= resources['cpu']
            memory -= reserved
            reserved_storage += needed
            if blocked is not None and q3:
                self.backfills_left -= 1
                event(self.profile, 'PBS_CALIBRATED_Q3_BACKFILL', job_id=attempt.parent.name,
                      attempt_id=attempt.name, memory_bytes=reserved, bypasses_left=self.backfills_left)
            if resources['standby']:
                event(self.profile, 'PBS_READY_BUFFER_LAUNCH', job_id=attempt.parent.name,
                      attempt_id=attempt.name, cpu=resources['cpu'], memory_bytes=resources['memory_bytes'])

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
        return not getattr(self,'paused',False) and not self.draining and not self.stop.is_set() and self.remaining() > self.drain_seconds + self.minimum_task_seconds

    def _sync_drain(self, control):
        """Admission pause is reversible; a real session STOP is not."""
        if self.stop.is_set():
            self.draining = True
            self.exit_reason = 'WORKER_STOP_DRAIN'
        elif self.remaining() <= self.drain_seconds + self.minimum_task_seconds:
            self.draining = True
            self.exit_reason = 'WALLTIME_ADMISSION_CUTOFF'
        elif control.get('stop') is True:
            self.draining = True
            self.exit_reason = self.exit_reason or 'CONTROLLER_STOP_DRAIN'
        elif not self.draining and type(control.get('pause')) is bool:
            self.paused=control['pause']

    def heartbeat(self, *, state=None, force=False):
        now = time.monotonic()
        if not force and now - self.last_heartbeat < 1.:
            return
        self.last_heartbeat = now
        control = self.control()
        if state != 'EXITED':self._sync_drain(control)
        value = dict(stamp(), schema='RPE_PBS_WORKER_STATE_V1',
                     state=state or ('DRAINING' if self.draining else 'PAUSED' if getattr(self,'paused',False) else 'READY'),
                     ready=state != 'EXITED', accepting=self.accepting() and not getattr(self, 'offline_active', False),
                     host_id=self.host_id, job_id=self.pbs_resources['actual']['job_id'],
                     hostname=socket.gethostname(), pid=os.getpid(),
                     source_sha=self.profile['pbs']['worker_source'],
                     resources=self.pbs_resources, capacity=self.slots,
                     remaining_seconds=self.remaining(),
                     start_clock=self.start_clock,
                     exit_reason=self.exit_reason)
        if getattr(self, 'offline_search', None) is not None:
            value['offline_revision'] = self.offline_search.revision
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
                pbs['launch_fit'] = self.collection is not None
                storage = payload['resource_report'].get('storage') or {}
                used, limit = storage.get('used_bytes'), storage.get('limit_bytes')
                pbs['ready_buffer'] = (self.collection is not None and storage.get('accepting_new_jobs') is not False
                                       and (limit is None or (used is not None and limit-used >= 1024**3)))
            # The central ledger uses these only to omit delivery replays;
            # resources remain charged against every unresolved owner.
            payload['durable_assignments'] = [
                [a['job_id'], a['attempt_id'], a['request_sha256']]
                for path, a in self.records.items() if path not in self.transferred_attempts
            ]
            payload.setdefault('resource_report', {})['pbs'] = pbs
            if not pbs['accepting']:
                payload['slots'] = 0
        result = super()._post(endpoint, payload)
        if endpoint == '/v1/state' and payload['state'] == 'COMPLETED_PENDING_TRANSFER':
            self.dispatch_wakeup.set()
        return result

    def _decline_before_launch(self, attempt):
        buffered = bool((self.records[attempt].get('pbs_launch_resources') or {}).get('standby'))
        save_state(attempt, 'UNKNOWN', reason='PBS_ADMISSION_DEADLINE_BEFORE_LAUNCH',
                   scientific_execution_started=False, automatic_retry=False, pbs_buffer_reserved=buffered)
        event(self.profile, 'PBS_ADMISSION_HOLD', job_id=attempt.parent.name,
              attempt_id=attempt.name, remaining_seconds=self.remaining())

    def _launch(self, attempt):
        # A slow transport reply can arrive after the last admission check. Keep
        # its original attempt and make the non-launch explicit; never roll its
        # reservation back or silently create another scientific attempt.
        if self.control().get('stop') or self.draining or self.stop.is_set() or self.remaining() <= self.drain_seconds+self.minimum_task_seconds:
            self._decline_before_launch(attempt)
            return
        super()._launch(attempt)

    def run_allocation(self):
        self.heartbeat(force=True)
        try:
            if getattr(self, 'readiness_pool', None) is not None:
                self.readiness_thread = threading.Thread(target=self._refill_readiness, name='pbs-readiness-refill')
                self.readiness_thread.start()
            while self.remaining() > 0:
                control = self.control()
                self._sync_drain(control)
                self.dispatch_wakeup.clear()
                offline_busy = self._offline_tick(control)
                if not offline_busy:
                    self.tick(acquire=self.accepting())
                self.heartbeat()
                if self.draining and not offline_busy and not self.pending():
                    break
                if control.get('release_when_idle') and not offline_busy and not self.pending():
                    if self.idle_since is None:
                        self.idle_since = time.monotonic()
                    if time.monotonic() - self.idle_since >= self.idle_release_seconds:
                        self.exit_reason = 'CONTROLLER_IDLE_RELEASE'
                        break
                else:
                    self.idle_since = None
                self.dispatch_wakeup.wait(min(self.poll_seconds, self.remaining()))
            if self.exit_reason is None:
                self.exit_reason = 'ALLOCATION_WALLTIME_END'
        finally:
            self.draining = True
            if getattr(self, 'offline_search', None) is not None:
                self.offline_search.request_stop()
            self._close_collection()
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
