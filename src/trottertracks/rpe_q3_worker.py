"""Persistent pull worker for the coordinator's unchanged, source-bound Q3 adapter.

A detached per-attempt supervisor preserves adapter exit evidence if the network
or polling daemon disappears. An ambiguous launch is never retried elsewhere.
"""
from copy import deepcopy
import base64
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request


# Match the existing adapter: exact DP bounds may contain very long integers.
# Each mailbox/supervisor is a fresh process and cannot inherit its setting.
if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)


STATES = {'ASSIGNED', 'STARTING', 'RUNNING', 'UNKNOWN', 'COMPLETED_PENDING_TRANSFER', 'TRANSFERRED'}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), default=str).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()) + '-' + str(threading.get_ident()))
    with temporary.open('w') as stream:
        json.dump(value, stream, default=str, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def exclusive(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, default=str, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def safe_id(value):
    if not isinstance(value, str) or not value or len(value) > 160 or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in value):
        raise ValueError('Unsafe job/attempt identifier')
    return value


def safe_relative(value):
    path = Path(value)
    if path.is_absolute() or not path.parts or any(p in ('.', '..') for p in path.parts):
        raise ValueError('Unsafe bundle/source path: ' + str(value))
    return path


def head(root):
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()


def process_alive(pid, marker):
    if type(pid) is not int or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        command = Path('/proc/%d/cmdline' % pid).read_bytes().decode(errors='replace')
        return str(marker) in command
    except (OSError, ProcessLookupError):
        return False


def event(profile, name, **detail):
    value = dict(event=name, epoch=time.time(), monotonic=time.monotonic(), host_id=profile['host_id'], **detail)
    path = Path(profile['work_dir']) / 'WORKER_EVENTS.jsonl'
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(json.dumps(value, default=str) + '\n'); stream.flush()


def save_state(attempt, state, **detail):
    if state not in STATES:
        raise ValueError('Invalid worker state')
    value = dict(state=state, epoch=time.time(), detail=detail)
    atomic(Path(attempt) / 'WORKER_STATE.json', value)
    return value


def validate_assignment(profile, assignment):
    """Verify scientific content before resolving the host's execution paths."""
    root = Path(profile['root']).resolve()
    request = assignment['request']
    if digest(request) != assignment['request_sha256']:
        raise ValueError('Request SHA256 differs')
    if request['request_id'] != assignment['job_id']:
        raise ValueError('Scientific job ID differs')
    if request['method'] != assignment['method']:
        raise ValueError('Method differs')
    manifest = assignment['source_manifest']
    if head(root) != manifest['commit']:
        raise ValueError('Worker HEAD differs from fixed SOURCE')
    for name, expected in manifest['files'].items():
        path = root / safe_relative(name)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Worker SOURCE hash differs: ' + name)
    for locator in assignment.get('input_hashes', []):
        safe_relative(locator['path'])
        blob = subprocess.check_output(['git', 'show', locator['commit'] + ':' + locator['path']], cwd=root)
        if hashlib.sha256(blob).hexdigest() != locator['sha256']:
            raise ValueError('Fixed input hash differs: ' + locator['path'])
    resolved = deepcopy(request)
    rule = resolved['route']
    rule['interpreter'] = profile['interpreter']
    if rule['kind'] != 'artificial':
        commit = rule['view_commit']
        view = Path(profile['route_views'][commit]).resolve()
        if head(view) != commit:
            raise ValueError('Route view HEAD differs')
        # Existing route check_source/metadata guards are additionally run by dry-check.
        rule['view'] = str(view)
        if rule['kind'] in ('prdf_d2_split', 'prdf_d2_common_entry'):
            # Host paths only, on the resolved copy. The saved caller declares
            # its required inner commit; the adapter checks that checkout.
            rule['route_views'] = dict(profile['route_views'])
    execution = resolved.setdefault('execution_profile', {})
    workers = profile.get('dp_workers')
    if workers is not None:
        if type(workers) is not int or workers < 1:
            raise ValueError('Positive integer host dp_workers required')
        execution['dp_workers'] = workers
    if profile.get('prefix_cache_directory') is not None:
        execution['prefix_cache_directory'] = str(Path(profile['prefix_cache_directory']).resolve())
    else:
        execution.pop('prefix_cache_directory', None)
    return resolved


def _adapter_command(profile, attempt, mode, output):
    return [profile['interpreter'], '-B', str(Path(profile['root']) / 'scripts/run_rpe_optimizer_q3_job.py'),
            '--request', str(Path(attempt) / 'RESOLVED_REQUEST.json'), '--output', str(output), '--mode', mode]


def _run_adapter(profile, attempt, mode, output):
    started = time.monotonic()
    with (Path(attempt) / ('WORKER_' + mode.upper() + '.log')).open('a') as log:
        process = subprocess.Popen(_adapter_command(profile, attempt, mode, output), cwd=profile['root'],
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), stdout=log, stderr=subprocess.STDOUT)
        launch_error = None
        if mode == 'execute':
            try:
                exclusive(Path(attempt) / 'WORKER_LAUNCH.json', dict(pid=process.pid, supervisor_pid=os.getpid(), epoch=time.time()))
                save_state(attempt, 'RUNNING', pid=process.pid, supervisor_pid=os.getpid())
                event(profile, 'LAUNCH', job_id=Path(attempt).parent.name, attempt_id=Path(attempt).name, pid=process.pid)
            except Exception as error:
                # A launched adapter still needs its exit receipt on marker/log failure.
                launch_error = error
        code = process.wait()
    receipt = dict(returncode=code if code >= 0 else None, signal=-code if code < 0 else None,
                   wall_seconds=time.monotonic() - started, automatic_retry=False)
    destination = 'PROCESS_RETURN.json' if mode == 'execute' else 'WORKER_' + mode.upper() + '_RETURN.json'
    exclusive(Path(attempt) / destination, receipt)
    if launch_error is not None:
        raise launch_error
    return code


def validate_completed(profile, assignment, attempt):
    """Revalidate stored artifacts without another scientific execution."""
    attempt = Path(attempt)
    receipt = read(attempt / 'PROCESS_RETURN.json')
    if not (type(receipt.get('returncode')) is int or type(receipt.get('signal')) is int):
        raise ValueError('No adapter exit code/signal evidence')
    outcome = attempt / 'OUTCOME.json'
    artificial = assignment['request']['route']['kind'] == 'artificial'
    # The artificial adapter writes OUTCOME with O_EXCL even in collect mode.
    # Existing OUTCOME is instead checked directly against ARTIFICIAL_RESULT.
    if artificial and outcome.exists():
        result = read(attempt / 'ARTIFICIAL_RESULT.json')
        observed = read(outcome)
        for name, value in result.items():
            if observed.get(name) != value:
                raise ValueError('Artificial worker artifact/outcome mismatch')
    elif not (attempt / 'WORKER_COLLECT_RETURN.json').exists():
        _run_adapter(profile, attempt, 'collect', attempt)
    if not outcome.exists():
        raise ValueError('No validated OUTCOME after collection')
    collected = attempt / 'COLLECTED_OUTCOME.json'
    if not artificial and collected.exists():
        observed = read(collected)
    else:
        observed = read(outcome)
    if not artificial:
        collection_return = read(attempt / 'WORKER_COLLECT_RETURN.json')
        if collection_return.get('returncode') not in (0, 2) or collection_return.get('signal') is not None:
            raise ValueError('Collection did not finish with a validated outcome')
    if observed.get('status') != 'UNKNOWN' and (receipt.get('returncode') != 0 or receipt.get('signal') is not None):
        raise ValueError('Nonzero adapter exit cannot authorize a successful outcome')
    if observed.get('request_id') != assignment['job_id'] or observed.get('route') != assignment['request']['route']['kind']:
        raise ValueError('Outcome job/route differs')
    # Identity and numerical contract are checked again by the controller's ingest.
    if observed.get('status') not in ('MODEL_BOUND_MEETS_ADOPTED_THRESHOLD', 'MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD', 'INCONCLUSIVE', 'UNKNOWN'):
        raise ValueError('Unsupported outcome status')
    return observed


def supervise(attempt):
    """One-shot child independent of polling/network; never retries execute."""
    attempt = Path(attempt).resolve()
    profile = read(attempt / 'WORKER_PROFILE.json')
    assignment = read(attempt / 'ASSIGNMENT.json')
    lock = (attempt / 'WORKER_PROCESS.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close(); return 3
    if (attempt / 'SUPERVISOR_STARTED.json').exists() or read(attempt / 'WORKER_STATE.json')['state'] != 'STARTING':
        lock.close(); return 3
    try:
        exclusive(attempt / 'SUPERVISOR_STARTED.json', dict(pid=os.getpid(), epoch=time.time()))
        resolved = validate_assignment(profile, assignment)
        exclusive(attempt / 'RESOLVED_REQUEST.json', resolved)
        dry = attempt / '_dry_check'; dry.mkdir()
        if _run_adapter(profile, attempt, 'dry-check', dry) != 0 or read(dry / 'DRY_CHECK.json').get('status') != 'DRY_CHECK_PASS':
            raise ValueError('Existing adapter dry-check failed')
        # Recheck the supplied fixed bytes immediately before the scientific launch.
        validate_assignment(profile, assignment)
        exclusive(attempt / 'SCIENTIFIC_STARTING.json', dict(epoch=time.time(), automatic_retry=False))
        _run_adapter(profile, attempt, 'execute', attempt)
        observed = validate_completed(profile, assignment, attempt)
        save_state(attempt, 'COMPLETED_PENDING_TRANSFER', outcome_status=observed['status'])
        event(profile, 'COMPLETE', job_id=assignment['job_id'], attempt_id=assignment['attempt_id'],
              status=observed['status'], process_return=read(attempt / 'PROCESS_RETURN.json'))
        return 0
    except Exception as error:
        save_state(attempt, 'UNKNOWN', reason=type(error).__name__ + ': ' + str(error), automatic_retry=False)
        event(profile, 'UNCERTAIN', job_id=assignment['job_id'], attempt_id=assignment['attempt_id'], reason=str(error))
        return 2
    finally:
        lock.close()


# Controller markers and assignment/profile files stay owned by their sender.
_TRANSFER_OMIT = {'ASSIGNMENT.json', 'WORKER_PROFILE.json', 'WORKER_STATE.json', 'WORKER_PROCESS.lock',
    'SUPERVISOR_STARTED.json', 'SCIENTIFIC_STARTING.json', 'WORKER_LAUNCH.json', 'WORKER_SUPERVISOR.log',
    'SUPERVISOR_LAUNCH.json', 'STARTING.json', 'LAUNCH.json', 'REQUEST.json'}


def result_bundle(attempt):
    files = {}
    for path in sorted(Path(attempt).rglob('*')):
        if path.is_symlink():
            raise ValueError('Symlink in result bundle')
        if path.is_file() and path.relative_to(attempt).parts[0] != '_dry_check' and str(path.relative_to(attempt)) not in _TRANSFER_OMIT:
            name = str(path.relative_to(attempt)); safe_relative(name)
            files[name] = base64.b64encode(path.read_bytes()).decode('ascii')
    if 'OUTCOME.json' not in files or 'PROCESS_RETURN.json' not in files:
        raise ValueError('Incomplete result bundle')
    return files


class Worker:
    def __init__(self, profile, controller, *, poll_seconds=1., transport=None):
        self.profile = dict(profile)
        self.root = Path(profile['root']).resolve()
        self.work = Path(profile['work_dir']).resolve(); self.work.mkdir(parents=True, exist_ok=True)
        self.profile.update(root=str(self.root), work_dir=str(self.work))
        self.host_id = safe_id(profile['host_id'])
        self.slots = profile.get('slots', 1)
        if type(self.slots) is not int or self.slots < 1:
            raise ValueError('Positive integer worker slots required')
        if not Path(profile['interpreter']).is_absolute():
            raise ValueError('Worker interpreter must be absolute')
        self.controller = controller.rstrip('/'); self.poll_seconds = poll_seconds
        if poll_seconds <= 0:
            raise ValueError('Positive poll interval required')
        self.transport = transport or self._http
        self.source_sha = head(self.root)
        self.stop = threading.Event(); self.processes = {}; self.reported = {}; self.records = {}
        self.lock = (self.work / 'WORKER.lock').open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close(); raise ValueError('Another polling worker owns this work_dir')
        for path in sorted(self.work.glob('*/*/ASSIGNMENT.json')):
            assignment = read(path)
            if assignment.get('host_id', self.host_id) != self.host_id:
                raise ValueError('Spool belongs to another host ID')
            self.records[path.parent] = assignment
        event(self.profile, 'WORKER_START', source_sha=self.source_sha, slots=self.slots)

    def _http(self, endpoint, payload):
        data = json.dumps(payload, allow_nan=False).encode()
        request = urllib.request.Request(self.controller + endpoint, data=data, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.load(response)

    def _post(self, endpoint, payload):
        return self.transport(endpoint, dict(host_id=self.host_id, **payload))

    def _acquire(self, assignment):
        safe_id(assignment['job_id']); safe_id(assignment['attempt_id'])
        if digest(assignment['request']) != assignment['request_sha256']:
            raise ValueError('Assignment request SHA256 differs')
        attempt = self.work / assignment['job_id'] / assignment['attempt_id']
        attempt.mkdir(parents=True, exist_ok=True)
        if (attempt / 'ASSIGNMENT.json').exists():
            before = read(attempt / 'ASSIGNMENT.json')
            if before != assignment:
                raise ValueError('Assignment changed for existing attempt')
            self.records[attempt] = before
            return
        # ASSIGNMENT becomes visible only after the exact request/profile are durable.
        atomic(attempt / 'ORIGINAL_REQUEST.json', assignment['request'])
        atomic(attempt / 'WORKER_PROFILE.json', self.profile)
        save_state(attempt, 'ASSIGNED')
        atomic(attempt / 'ASSIGNMENT.json', assignment)
        self.records[attempt] = assignment
        event(self.profile, 'ASSIGNED', job_id=assignment['job_id'], attempt_id=assignment['attempt_id'])

    def _launch(self, attempt):
        # Durable before Popen: an interrupted handoff is ambiguous, never retried.
        save_state(attempt, 'STARTING', automatic_retry=False)
        command = [self.profile['interpreter'], '-B', str(self.root / 'scripts/run_rpe_q3_worker.py'),
                   '--supervise-attempt', str(attempt)]
        with (attempt / 'WORKER_SUPERVISOR.log').open('a') as log:
            process = subprocess.Popen(command, cwd=self.root, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        exclusive(attempt / 'SUPERVISOR_LAUNCH.json', dict(pid=process.pid, epoch=time.time()))
        self.processes[attempt] = process

    def _reconcile(self, attempt, assignment):
        state = read(attempt / 'WORKER_STATE.json')
        process = self.processes.get(attempt)
        if process is not None:
            code = process.poll()
            if code is None:
                return read(attempt / 'WORKER_STATE.json')
            self.processes.pop(attempt)
            event(self.profile, 'SUPERVISOR_RETURN', job_id=assignment['job_id'], attempt_id=assignment['attempt_id'],
                  returncode=code if code >= 0 else None, signal=-code if code < 0 else None)
            state = read(attempt / 'WORKER_STATE.json')
        if state['state'] in ('STARTING', 'RUNNING'):
            for name in ('SUPERVISOR_STARTED.json', 'SUPERVISOR_LAUNCH.json'):
                marker = attempt / name
                if marker.exists() and process_alive(read(marker)['pid'], attempt):
                    return state
            # A process could have started before its first marker was saved.
            # Without a live known supervisor or receipt there is no safe retry.
            if (attempt / 'PROCESS_RETURN.json').exists():
                try:
                    observed = validate_completed(self.profile, assignment, attempt)
                    state = save_state(attempt, 'COMPLETED_PENDING_TRANSFER', outcome_status=observed['status'], recovered=True)
                    event(self.profile, 'RECOVER_COMPLETED', job_id=assignment['job_id'], attempt_id=assignment['attempt_id'])
                except Exception as error:
                    state = save_state(attempt, 'UNKNOWN', reason=str(error), automatic_retry=False)
            else:
                state = save_state(attempt, 'UNKNOWN', reason='LAUNCH_STATE_UNCERTAIN_NO_ADAPTER_EXIT_RECEIPT', automatic_retry=False)
        return state

    def tick(self, acquire=True):
        active = 0
        for attempt, assignment in list(self.records.items()):
            state = self._reconcile(attempt, assignment)
            if state['state'] == 'ASSIGNED':
                self._launch(attempt); state = read(attempt / 'WORKER_STATE.json')
            if state['state'] in ('ASSIGNED', 'STARTING', 'RUNNING'):
                active += 1
            payload = dict(job_id=assignment['job_id'], attempt_id=assignment['attempt_id'], state=state['state'], detail=state['detail'])
            if state['state'] != 'TRANSFERRED' and self.reported.get(attempt) != state:
                try:
                    self._post('/v1/state', payload); self.reported[attempt] = state
                except (OSError, ValueError, urllib.error.URLError) as error:
                    event(self.profile, 'STATE_OFFLINE', job_id=assignment['job_id'], reason=str(error))
            if state['state'] == 'COMPLETED_PENDING_TRANSFER':
                try:
                    started = time.monotonic(); files = result_bundle(attempt)
                    response = self._post('/v1/complete', dict(job_id=assignment['job_id'], attempt_id=assignment['attempt_id'],
                        request_sha256=assignment['request_sha256'], files=files))
                    if response.get('ok') is not True:
                        raise ValueError('Controller refused completion')
                    timing = dict(upload_roundtrip_seconds=time.monotonic()-started,
                                  bytes=sum(len(base64.b64decode(v)) for v in files.values()))
                    save_state(attempt, 'TRANSFERRED', duplicate=response.get('duplicate', False))
                    event(self.profile, 'TRANSFER_ACK', job_id=assignment['job_id'], attempt_id=assignment['attempt_id'],
                          seconds=timing['upload_roundtrip_seconds'], bytes=timing['bytes'],
                          duplicate=response.get('duplicate', False))
                    try:
                        self._post('/v1/transfer-metric', dict(job_id=assignment['job_id'], attempt_id=assignment['attempt_id'], timing=timing))
                    except (OSError, ValueError, urllib.error.URLError) as error:
                        # Optional telemetry cannot undo a durable completion acknowledgement.
                        event(self.profile, 'TRANSFER_METRIC_OFFLINE', job_id=assignment['job_id'], reason=str(error))
                except (OSError, ValueError, urllib.error.URLError) as error:
                    event(self.profile, 'TRANSFER_PENDING', job_id=assignment['job_id'], reason=str(error))
        if acquire and not self.stop.is_set() and active < self.slots:
            cached = sorted({a.get('prefix_key') for p, a in self.records.items()
                if a.get('prefix_key') and read(p / 'WORKER_STATE.json')['state'] in ('COMPLETED_PENDING_TRANSFER', 'TRANSFERRED')
                })
            try:
                response = self._post('/v1/poll', dict(slots=self.slots-active, cached_prefixes=cached, source_sha=self.source_sha))
                assignments = response.get('assignments', [])
                new_assignments = [a for a in assignments if
                    self.work / safe_id(a['job_id']) / safe_id(a['attempt_id']) not in self.records]
                if len(new_assignments) > self.slots-active:
                    raise ValueError('Controller overfilled announced slots')
                for assignment in assignments:
                    self._acquire(assignment)
                if response.get('stop'):
                    self.stop.set()
            except (OSError, ValueError, urllib.error.URLError) as error:
                event(self.profile, 'POLL_OFFLINE', reason=str(error))
        return active

    def pending(self):
        return any(read(p / 'WORKER_STATE.json')['state'] not in ('TRANSFERRED', 'UNKNOWN') for p in self.records)

    def run(self, *, once=False, drain=False):
        polled = False
        try:
            while True:
                acquire = not drain and not self.stop.is_set() and not (once and polled)
                self.tick(acquire=acquire)
                if acquire:
                    polled = True
                if (drain or once or self.stop.is_set()) and not self.pending():
                    break
                self.stop.wait(self.poll_seconds) if not self.stop.is_set() else time.sleep(self.poll_seconds)
        finally:
            # Stopping polling never signals the detached scientific process.
            event(self.profile, 'WORKER_STOP', pending=self.pending())
            self.lock.close()

    def close(self):
        self.lock.close()


def mailbox(profile, action, payload=None):
    """One SSH request against the same durable spool used by the HTTP daemon.

    submit detaches the one-shot supervisor; status transfers bytes only after
    saved adapter return/outcome validation. No shared filesystem is assumed.
    """
    payload = payload or {}
    worker = Worker(profile, 'http://127.0.0.1:1')
    try:
        if action == 'submit':
            assignment = payload.get('assignment', payload)
            validate_assignment(worker.profile, assignment)
            safe_id(assignment['job_id']); safe_id(assignment['attempt_id'])
            attempt = worker.work / assignment['job_id'] / assignment['attempt_id']
            duplicate = (attempt / 'ASSIGNMENT.json').exists()
            active = sum(worker._reconcile(path, saved)['state'] in ('ASSIGNED', 'STARTING', 'RUNNING')
                         for path, saved in worker.records.items())
            if not duplicate and active >= worker.slots:
                raise ValueError('No free worker slot')
            worker._acquire(assignment)
            state = read(attempt / 'WORKER_STATE.json')
            if state['state'] == 'ASSIGNED':
                worker._launch(attempt)
                threading.Thread(target=worker.processes[attempt].wait, daemon=True).start()
            return dict(ok=True, duplicate=duplicate)
        if action == 'ack':
            attempt = worker.work / safe_id(payload['job_id']) / safe_id(payload['attempt_id'])
            if attempt not in worker.records:
                raise ValueError('Unknown completion attempt')
            state = worker._reconcile(attempt, worker.records[attempt])
            if state['state'] not in ('COMPLETED_PENDING_TRANSFER', 'TRANSFERRED'):
                raise ValueError('Attempt is not complete')
            duplicate = state['state'] == 'TRANSFERRED'
            if not duplicate:
                save_state(attempt, 'TRANSFERRED', via='SSH_MAILBOX')
                event(worker.profile, 'TRANSFER_ACK', job_id=payload['job_id'], attempt_id=payload['attempt_id'], via='SSH_MAILBOX')
            return dict(ok=True, duplicate=duplicate)
        if action == 'status':
            attempts = []; active = 0; cached = set()
            for attempt, assignment in worker.records.items():
                state = worker._reconcile(attempt, assignment)
                if state['state'] == 'ASSIGNED':
                    worker._launch(attempt)
                    threading.Thread(target=worker.processes[attempt].wait, daemon=True).start()
                    state = read(attempt / 'WORKER_STATE.json')
                active += state['state'] in ('ASSIGNED', 'STARTING', 'RUNNING')
                row = dict(job_id=assignment['job_id'], attempt_id=assignment['attempt_id'],
                    state=state['state'], detail=state['detail'], request_sha256=assignment['request_sha256'])
                if state['state'] == 'COMPLETED_PENDING_TRANSFER':
                    row['files'] = result_bundle(attempt)
                if state['state'] in ('COMPLETED_PENDING_TRANSFER', 'TRANSFERRED') and assignment.get('prefix_key'):
                    cached.add(assignment['prefix_key'])
                attempts.append(row)
            return dict(host_id=worker.host_id, slots=max(0,worker.slots-active), source_sha=worker.source_sha,
                        cached_prefixes=sorted(cached), attempts=attempts)
        raise ValueError('Unknown mailbox action')
    finally:
        worker.close()
