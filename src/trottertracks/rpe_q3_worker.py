"""Persistent pull worker for the coordinator's unchanged, source-bound Q3 adapter.

A detached per-attempt supervisor preserves adapter exit evidence if the network
or polling daemon disappears. An ambiguous launch is never retried elsewhere.
"""
from copy import deepcopy
from .rpe_dry_check_recovery import dry_check_rejection, failed_dry_check_released
from .rpe_q3_timing import stamp, append as timing_append
from . import rpe_pbs_attempt_control as PBSAttempt
from . import rpe_q3_resources as Resources
from . import rpe_worker_storage as Storage
from .rpe_host_profiles import select_host_profile as _select_q3_profile, validate_adapter_profile, check_runtime_profile
from . import rpe_preparation_worker as Preparation
from .rpe_preparation_artifacts import worker_cache, input_manifest
import base64
from concurrent.futures import ThreadPoolExecutor
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
from uuid import UUID


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


def atomic(path, value, *, publish_once=False):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()) + '-' + str(threading.get_ident()))
    with temporary.open('w') as stream:
        json.dump(value, stream, default=str, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    if publish_once:
        try:
            os.link(temporary, path)
        finally:
            temporary.unlink()
    else:
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


def session_has_live_members(session_id, *, exclude=()):
    for stat in Path('/proc').glob('[0-9]*/stat'):
        if int(stat.parent.name) in exclude:
            continue
        try:
            fields = stat.read_text().rsplit(')', 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if int(fields[3]) == session_id and fields[0] not in ('Z', 'X'):
            return True
    return False


def supervisor_session_ended(attempt):
    """A matching detached session with no live members cannot consume a slot."""
    try:
        started = read(attempt / 'SUPERVISOR_STARTED.json')
        launched = read(attempt / 'SUPERVISOR_LAUNCH.json')
        with (attempt / 'TRANSPORT_TIMING.jsonl').open() as stream:
            clock = json.loads(stream.readline())['clock_id']
        pid = started['pid']
        if (clock != stamp()['clock_id'] or type(pid) is not int or pid <= 0
                or launched['pid'] != pid or started.get('session_id', pid) != pid):
            return False
        adapter = attempt / 'WORKER_LAUNCH.json'
        if adapter.exists() and read(adapter).get('supervisor_pid') != pid:
            return False
        with (attempt / 'WORKER_PROCESS.lock').open('rb') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return not session_has_live_members(pid)
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return False


def resource_state(attempt, state):
    """Keep recovery ownership separate from CPU occupancy with exit evidence."""
    if state != 'UNKNOWN':
        return state
    if (attempt / PBSAttempt.OWNER).is_file():
        detail = read(attempt/'WORKER_STATE.json').get('detail', {})
        return 'EXITED' if detail.get('resource_released') is True else state
    for name in ('SUPERVISOR_STARTED.json', 'SUPERVISOR_LAUNCH.json', 'WORKER_LAUNCH.json'):
        marker = attempt / name
        if marker.is_file() and process_alive(read(marker).get('pid'), attempt):
            return state
    if failed_dry_check_released(attempt):
        return 'EXITED'
    exit_receipt = attempt / 'PROCESS_RETURN.json'
    if exit_receipt.is_file():
        try:
            receipt = read(exit_receipt)
        except (OSError, ValueError):
            receipt = None  # An interrupted write cannot release a live slot.
        if isinstance(receipt, dict) and (
                type(receipt.get('returncode')) is int or type(receipt.get('signal')) is int):
            return 'EXITED'
    timing = attempt / 'TRANSPORT_TIMING.jsonl'
    if timing.is_file():
        try:
            with timing.open() as stream:
                original = json.loads(stream.readline()).get('clock_id', '')
            host, _, boot = original.partition(':')
            current_host, _, current_boot = stamp()['clock_id'].partition(':')
            if host == current_host and UUID(boot) != UUID(current_boot):
                return 'EXITED'
        except (ValueError, AttributeError):
            pass
    return 'EXITED' if supervisor_session_ended(attempt) else state


def event(profile, name, **detail):
    value = dict(stamp(), event=name, host_id=profile['host_id'], **detail)
    path = Path(profile['work_dir']) / 'WORKER_EVENTS.jsonl'
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(json.dumps(value, default=str) + '\n'); stream.flush()


def save_state(attempt, state, **detail):
    if state not in STATES:
        raise ValueError('Invalid worker state')
    value = dict(stamp(), state=state, detail=detail)
    atomic(Path(attempt) / 'WORKER_STATE.json', value)
    return value


def select_host_profile(profile, assignment):
    return Preparation.select_preparation_profile(profile, assignment) if Preparation.is_preparation(assignment) else _select_q3_profile(profile, assignment)


def validate_assignment(profile, assignment):
    """Verify scientific content before resolving the host's execution paths."""
    base_profile = profile
    profile = select_host_profile(profile, assignment)
    if Preparation.is_preparation(assignment):
        Preparation.current_preparation_fingerprint(assignment, base_profile)
        return deepcopy(assignment['request'])
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


def host_fingerprints(profile, challenges, *, q3_parallel=1, preparation_parallel=1, deployment=None):
    """Read current execution identities; never acquire or launch a job."""
    from .rpe_host_eligibility import current_host_fingerprint
    fingerprints, errors = {}, {}
    deployment = deployment or profile.get('pbs') or profile
    def check(assignment):
        job = assignment['job_id']
        try:
            if assignment.get('attempt_id') is None and assignment.get('required_release'):
                from .rpe_release_deployment import prepare_worker_assignment
                prepare_worker_assignment(profile, assignment, deployment)
            fact = (Preparation.current_preparation_fingerprint(assignment, profile, check_workspace=True)
                    if Preparation.is_preparation(assignment) else current_host_fingerprint(assignment, profile))
            return job, fact, None
        except Exception as error:
            return job, None, type(error).__name__ + ': ' + str(error)

    def collect(results):
        for job, fact, error in results:
            if error is None:
                fingerprints[job] = fact
            else:
                errors[job] = error

    q3, preparation = [], []
    for item in challenges:
        (preparation if Preparation.is_preparation(item) else q3).append(item)
    for batch, parallel in ((q3, q3_parallel), (preparation, preparation_parallel)):
        if parallel > 1 and len(batch) > 1:
            with ThreadPoolExecutor(max_workers=min(len(batch), parallel)) as pool:
                collect(pool.map(check, batch))
        else:
            collect(map(check, batch))
    return dict(host_fingerprints=fingerprints, readiness_errors=errors)


def _adapter_command(profile, attempt, mode, output):
    saved_assignment = Path(attempt) / 'ASSIGNMENT.json'
    if saved_assignment.is_file():
        assignment = read(saved_assignment)
        profile = select_host_profile(profile, assignment)
        if Preparation.is_preparation(assignment):
            # The worker release supplies the transport. The profile still
            # binds the provider wrapper, fixed sources and runtime inputs.
            root = Path(__file__).resolve().parents[2]
            return [profile['interpreter'], '-B', str(root / 'scripts/run_rpe_preparation_job.py'),
                    '--assignment', str(saved_assignment), '--profile', str(Path(attempt)/'WORKER_PROFILE.json'),
                    '--task-root', str(attempt), '--output', str(output), '--mode', mode]
    adapter = validate_adapter_profile(profile)
    if profile.get('runtime_checks') and check_runtime_profile(profile)['status'] != 'PASS':
        raise ValueError('Fixed execution runtime AST guard did not pass')
    root = Path(adapter['root'] if adapter else profile['root'])
    command = [profile['interpreter'], '-B', str(root / 'scripts/run_rpe_optimizer_q3_job.py'),
               '--request', str(Path(attempt) / 'RESOLVED_REQUEST.json'), '--output', str(output), '--mode', mode]
    if adapter:
        paths = Path(attempt) / 'ROUTE_VIEWS.json'
        if paths.exists():
            if read(paths) != profile['route_views']:
                raise ValueError('Assigned adapter route views changed')
        else:
            atomic(paths, profile['route_views'])
        command += ['--route-views', str(paths)]
        record = Path(attempt) / 'ADAPTER_SOURCE.json'
        if record.exists() and read(record) != adapter:
            raise ValueError('Assigned adapter manifest changed')
        if not record.exists():
            atomic(record, adapter)
    return command


def _run_adapter(profile, attempt, mode, output):
    started = time.monotonic()
    attempt = Path(attempt)
    owner = PBSAttempt.local_supervisor(attempt) if mode in ('dry-check', 'execute') else None
    PBSAttempt.before_launch(attempt, owner, exclusive)
    timing_append(attempt/'TRANSPORT_TIMING.jsonl','ADAPTER_PROCESS_START',mode=mode)
    with (Path(attempt) / ('WORKER_' + mode.upper() + '.log')).open('a') as log:
        process = subprocess.Popen(_adapter_command(profile, attempt, mode, output), cwd=profile.get('adapter_root', profile['root']),
            env=PBSAttempt.adapter_environment(owner), stdout=log, stderr=subprocess.STDOUT,
            **({'process_group': 0} if owner else {}))
        launch_error = None
        if mode == 'execute':
            try:
                exclusive(Path(attempt) / 'WORKER_LAUNCH.json', dict(pid=process.pid, supervisor_pid=os.getpid(), epoch=time.time()))
                save_state(attempt, 'RUNNING', pid=process.pid, supervisor_pid=os.getpid())
                event(profile, 'LAUNCH', job_id=Path(attempt).parent.name, attempt_id=Path(attempt).name, pid=process.pid)
            except Exception as error:
                # A launched adapter still needs its exit receipt on marker/log failure.
                launch_error = error
        code, stopped = PBSAttempt.wait_adapter(process, attempt, owner, exclusive)
    receipt = dict(returncode=code if code >= 0 else None, signal=-code if code < 0 else None,
                   wall_seconds=time.monotonic() - started, automatic_retry=False)
    destination = 'PROCESS_RETURN.json' if mode == 'execute' else 'WORKER_' + mode.upper() + '_RETURN.json'
    atomic(Path(attempt) / destination, receipt, publish_once=True)
    timing_append(Path(attempt)/'TRANSPORT_TIMING.jsonl','ADAPTER_PROCESS_END',mode=mode,returncode=code)
    if stopped:
        raise PBSAttempt.AttemptStopped
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
    if Preparation.is_preparation(assignment):
        if not (attempt / 'WORKER_COLLECT_RETURN.json').exists():
            _run_adapter(profile, attempt, 'collect', attempt)
        observed = Preparation.collect(assignment, profile, attempt)
        if observed['status'] == 'PREPARATION_COMPLETE' and (receipt.get('returncode') != 0 or receipt.get('signal') is not None):
            raise ValueError('Preparation success requires a clean adapter exit')
        return observed
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
        exclusive(attempt / 'SUPERVISOR_STARTED.json',
                  dict(pid=os.getpid(), session_id=os.getsid(0), epoch=time.time()))
        PBSAttempt.register(profile, attempt, assignment, exclusive)
        resolved = validate_assignment(profile, assignment)
        exclusive(attempt / 'RESOLVED_REQUEST.json', resolved)
        dry = attempt / '_dry_check'; dry.mkdir()
        if _run_adapter(profile, attempt, 'dry-check', dry) != 0 or read(dry / 'DRY_CHECK.json').get('status') != 'DRY_CHECK_PASS':
            raise ValueError('Existing adapter dry-check failed')
        # Recheck the supplied fixed bytes immediately before the scientific launch.
        validate_assignment(profile, assignment)
        PBSAttempt.before_launch(attempt, PBSAttempt.local_supervisor(attempt), exclusive)
        exclusive(attempt / 'SCIENTIFIC_STARTING.json', dict(epoch=time.time(), automatic_retry=False))
        _run_adapter(profile, attempt, 'execute', attempt)
        observed = validate_completed(profile, assignment, attempt)
        timing_append(attempt/'TRANSPORT_TIMING.jsonl','RESULT_BUNDLE_READY')
        save_state(attempt, 'COMPLETED_PENDING_TRANSFER', outcome_status=observed['status'])
        event(profile, 'COMPLETE', job_id=assignment['job_id'], attempt_id=assignment['attempt_id'],
              status=observed['status'], process_return=read(attempt / 'PROCESS_RETURN.json'))
        return 0
    except PBSAttempt.AttemptStopped:
        save_state(attempt, 'UNKNOWN', reason='OPERATOR_PBS_ATTEMPT_STOP', automatic_retry=False)
        return 2
    except Exception as error:
        reason = type(error).__name__ + ': ' + str(error)
        save_state(attempt, 'UNKNOWN', reason=reason, automatic_retry=False)
        if not (attempt/'SCIENTIFIC_STARTING.json').exists():
            # This receipt is written by the one-shot supervisor itself, not
            # inferred later from a missing output or a lost connection.
            exclusive(attempt/'SUPERVISOR_PRELAUNCH_FAILURE.json', dict(
                job_id=assignment['job_id'], attempt_id=assignment['attempt_id'],
                request_sha256=assignment['request_sha256'], supervisor_pid=os.getpid(),
                reason=reason, returncode=2, scientific_execution_started=False))
        event(profile, 'UNCERTAIN', job_id=assignment['job_id'], attempt_id=assignment['attempt_id'], reason=str(error))
        return 2
    finally:
        lock.close()


# Controller markers and assignment/profile files stay owned by their sender.
_TRANSFER_OMIT = {'ASSIGNMENT.json', 'WORKER_PROFILE.json', 'WORKER_STATE.json', 'WORKER_PROCESS.lock',
    'SUPERVISOR_STARTED.json', 'SCIENTIFIC_STARTING.json', 'WORKER_LAUNCH.json', 'WORKER_SUPERVISOR.log',
    'SUPERVISOR_LAUNCH.json', 'STARTING.json', 'LAUNCH.json', 'REQUEST.json'}
_TRANSFER_OMIT.update((PBSAttempt.OWNER, PBSAttempt.REQUEST, PBSAttempt.INTERRUPTED))


def result_bundle(attempt):
    attempt = Path(attempt)
    assignment_path = attempt/'ASSIGNMENT.json'
    if assignment_path.exists() and Preparation.is_preparation(read(assignment_path)):
        names = ('OUTCOME.json', 'PROCESS_RETURN.json', 'ARTIFACT_MANIFEST.json', 'PREPARATION_RECEIPT.json',
                 'JOB_STAGES.jsonl', 'TRANSPORT_TIMING.jsonl', 'WORKER_COLLECT_RETURN.json')
        files = {name:base64.b64encode((attempt/name).read_bytes()).decode('ascii')
                 for name in names if (attempt/name).is_file()}
        if 'OUTCOME.json' not in files or 'PROCESS_RETURN.json' not in files:raise ValueError('Incomplete preparation result bundle')
        return files
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
        self.slots = (profile.get('slot_policy') or {}).get('initial_slots', profile.get('slots', 1))
        if type(self.slots) is not int or self.slots < 1:
            raise ValueError('Positive integer worker slots required')
        if not Path(profile['interpreter']).is_absolute():
            raise ValueError('Worker interpreter must be absolute')
        self.controller = controller.rstrip('/'); self.poll_seconds = poll_seconds
        if poll_seconds <= 0:
            raise ValueError('Positive poll interval required')
        self.transport = transport or self._http
        self.source_sha = head(self.root)
        self.eligibility_challenges = []
        self.stop = threading.Event(); self.processes = {}; self.reported = {}; self.records = {}
        self.resource_report = None
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

    @property
    def capacity_state_dir(self):
        return self.work

    def _capacity_feedback(self, response):
        feedback = response.get('capacity_feedback')
        if feedback is not None and (self.profile.get('slot_policy') or {}).get('mode') == 'auto':
            atomic(self.capacity_state_dir / 'CAPACITY_FEEDBACK.json', feedback)

    def resource_snapshot(self):
        return Resources.resource_snapshot(self.profile)

    def _storage_offer(self, report):
        storage = Storage.report(self.profile.get('storage_limit'), spools=(str(self.work),),
                                 cache_directories=Storage.prefix_caches(self.profile))
        if storage is not None:
            report['storage'] = storage
            if not storage['accepting_new_jobs']:
                report['slots'] = 0
        self.resource_report = report
        return report

    def _resource_offer(self):
        if (self.profile.get('slot_policy') or {}).get('mode') != 'auto':
            jobs = []
            for path in self.records:
                saved = read(path/'WORKER_STATE.json')
                state = saved['state']
                if state == 'UNKNOWN' and saved.get('detail', {}).get('resource_released') is True:
                    state = 'EXITED'
                jobs.append(dict(state=state))
            _, report = Resources.decide_slots(self.profile, {}, {}, jobs)
            self.slots = report['capacity']
            return self._storage_offer(report)
        jobs = []
        preparation_observations = []
        for attempt, assignment in self.records.items():
            persisted = read(attempt / 'WORKER_STATE.json')
            state = persisted['state']
            saved = read(attempt / 'WORKER_PROFILE.json')
            sample = dict(rss_bytes=0, cpu_seconds=0)
            for name in ('SUPERVISOR_STARTED.json', 'SUPERVISOR_LAUNCH.json'):
                marker = attempt / name
                if marker.exists():
                    pid = read(marker).get('pid')
                    if process_alive(pid, attempt):
                        sample = Resources.owned_tree_sample(pid)
                        break
            path = attempt / 'OWNED_RESOURCES.json'
            previous = read(path) if path.exists() else {}
            peak = max(previous.get('peak_rss_bytes', 0), sample['rss_bytes'])
            observed = stamp()
            seconds = observed['monotonic']-previous.get('monotonic', observed['monotonic'])
            same_clock = previous.get('clock_id') == observed['clock_id']
            cores = max(0, sample['cpu_seconds']-previous.get('cpu_seconds', 0))/seconds if same_clock and seconds > 0 else 0
            resource = 'EXITED' if state == 'UNKNOWN' and persisted['detail'].get('resource_released') is True else resource_state(attempt, state)
            kind = Resources.measured_resource_class(assignment['request'], assignment.get('q3_evaluator'), saved, preparation_detail=True,
                                                     workload=assignment.get('preparation_workload'))
            # Previously assigned attempts may predate evaluator metadata. Their
            # samples stay in the conservative count-DP profile.
            job = dict(state=resource, resource_class=kind, dp_workers=max(saved.get('dp_workers') or 1,
                                                 saved.get('task_cpu_count') or 1),
                       peak_rss_bytes=peak, cpu_seconds=sample['cpu_seconds'],
                       observed_cpu_cores=max(cores, previous.get('observed_cpu_cores', 0)))
            if state in ('ASSIGNED', 'STARTING', 'RUNNING', 'UNKNOWN') and (self.profile.get('slot_policy') or {}).get('mode') == 'auto':
                atomic(path, dict(observed, **job))
            jobs.append(job)
            observation = Resources.owned_preparation_observation(attempt, assignment, saved, peak)
            if observation is not None:preparation_observations.append(observation)
        feedback_path = self.capacity_state_dir / 'CAPACITY_FEEDBACK.json'
        state_path = self.capacity_state_dir / 'SLOT_STATE.json'
        feedback = read(feedback_path) if feedback_path.exists() else {}
        previous = read(state_path) if state_path.exists() else {}
        resources = self.resource_snapshot() if (self.profile.get('slot_policy') or {}).get('mode') == 'auto' else {}
        state, report = Resources.decide_slots(self.profile, previous, resources, jobs, feedback)
        if preparation_observations:report['preparation_observations'] = preparation_observations
        report['preparation_class_version'] = 3
        self.slots = report['capacity']
        self._storage_offer(report)
        if report['mode'] == 'AUTO':
            if previous.get('target_slots') != report['capacity'] or previous.get('stage') != report['stage']:
                event(self.profile, 'SLOT_CAPACITY_DECISION', capacity=report['capacity'], stage=report['stage'],
                      reasons=report['reasons'], limits=report['limits'],
                      dp_workers_per_q3=report['dp_workers_per_q3'], feedback=report['feedback'])
            atomic(state_path, state)
            atomic(self.work / 'RESOURCE_OFFER.json', report)
        return report

    def _http(self, endpoint, payload):
        data = json.dumps(payload, allow_nan=False).encode()
        request = urllib.request.Request(self.controller + endpoint, data=data, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.load(response)

    def _post(self, endpoint, payload):
        return self.transport(endpoint, dict(host_id=self.host_id, **payload))

    def _validate_acquisition(self, assignment):
        """An allocation worker may also enforce a deployed release contract."""

    def _acquire(self, assignment):
        assignment = deepcopy(assignment)
        safe_id(assignment['job_id']); safe_id(assignment['attempt_id'])
        if digest(assignment['request']) != assignment['request_sha256']:
            raise ValueError('Assignment request SHA256 differs')
        incoming = assignment.pop('artifact_blobs', {})
        if Preparation.is_preparation(assignment):
            cache = worker_cache(self.profile)
            cache.put(incoming)
            hashes = list(Preparation.assigned_artifacts(assignment).values())
            hashes += Preparation.missing_metadata_bundle_hashes(assignment, self.profile)
            missing = cache.missing(hashes)
            if missing:
                cache.put(self._post('/v1/artifacts/get', dict(hashes=missing))['blobs'])
            if cache.missing(hashes):raise ValueError('Preparation input artifacts not available')
        # Fixed release validation also checks the preparation input metadata.
        # It needs the verified bundles but must precede durable ownership.
        self._validate_acquisition(assignment)
        attempt = self.work / assignment['job_id'] / assignment['attempt_id']
        attempt.mkdir(parents=True, exist_ok=True)
        if (attempt / 'ASSIGNMENT.json').exists():
            before = read(attempt / 'ASSIGNMENT.json')
            if before != assignment:
                raise ValueError('Assignment changed for existing attempt')
            self.records[attempt] = before
            return
        # ASSIGNMENT becomes visible only after the exact request/profile are durable.
        selected_profile = select_host_profile(self.profile, assignment)
        if 'pbs' in self.profile:
            selected_profile['pbs'] = deepcopy(self.profile['pbs'])
        atomic(attempt / 'ORIGINAL_REQUEST.json', assignment['request'])
        atomic(attempt / 'WORKER_PROFILE.json', selected_profile)
        save_state(attempt, 'ASSIGNED')
        atomic(attempt / 'ASSIGNMENT.json', assignment)
        self.records[attempt] = assignment
        event(self.profile, 'ASSIGNED', job_id=assignment['job_id'], attempt_id=assignment['attempt_id'])
        timing_append(attempt/'TRANSPORT_TIMING.jsonl','WORKER_ASSIGNMENT_DURABLE',
            job_id=assignment['job_id'],attempt_id=assignment['attempt_id'])

    def _launch(self, attempt):
        # Durable before Popen: an interrupted handoff is ambiguous, never retried.
        profile = select_host_profile(read(attempt / 'WORKER_PROFILE.json'), read(attempt / 'ASSIGNMENT.json'))
        save_state(attempt, 'STARTING', automatic_retry=False)
        # Supervising is transport work. A provider's interpreter can be a
        # narrowly scoped launcher that deliberately rejects worker scripts.
        command = [sys.executable, '-B', str(Path(__file__).resolve().parents[2] / 'scripts/run_rpe_q3_worker.py'),
                   '--supervise-attempt', str(attempt)]
        with (attempt / 'WORKER_SUPERVISOR.log').open('a') as log:
            process = subprocess.Popen(command, cwd=profile['root'], env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
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
        proof = state.get('detail', {}).get('pbs_attempt_exit')
        if state['state'] in ('STARTING', 'RUNNING', 'UNKNOWN') and (attempt / PBSAttempt.OWNER).is_file():
            if not proof:
                proof = PBSAttempt.exit_evidence(attempt, assignment, self.profile, supervisor_session_ended)
            if proof and proof['operator_stop']:
                if state['detail'].get('pbs_attempt_exit') != proof:
                    state = save_state(attempt, 'UNKNOWN', reason='OPERATOR_PBS_ATTEMPT_STOP',
                                       automatic_retry=False, resource_released=True, pbs_attempt_exit=proof)
                return state
        if state['state'] in ('STARTING', 'RUNNING') or (
                state['state'] == 'UNKNOWN' and not (attempt / PBSAttempt.INTERRUPTED).is_file()
                and (attempt / 'OUTCOME.json').is_file()
                and (attempt / 'PROCESS_RETURN.json').is_file()):
            for name in ('SUPERVISOR_STARTED.json', 'SUPERVISOR_LAUNCH.json'):
                marker = attempt / name
                if marker.exists() and process_alive(read(marker)['pid'], attempt):
                    return state
            # A process could have started before its first marker was saved.
            # Without a live known supervisor or receipt there is no safe retry.
            if (attempt / 'PROCESS_RETURN.json').exists():
                try:
                    observed = validate_completed(read(attempt / 'WORKER_PROFILE.json'), assignment, attempt)
                    state = save_state(attempt, 'COMPLETED_PENDING_TRANSFER', outcome_status=observed['status'], recovered=True)
                    event(self.profile, 'RECOVER_COMPLETED', job_id=assignment['job_id'], attempt_id=assignment['attempt_id'])
                except Exception as error:
                    state = save_state(attempt, 'UNKNOWN', reason=str(error), automatic_retry=False)
            else:
                state = save_state(attempt, 'UNKNOWN', reason='LAUNCH_STATE_UNCERTAIN_NO_ADAPTER_EXIT_RECEIPT', automatic_retry=False)
        if state['state'] == 'UNKNOWN':
            detail = state['detail']
            if detail.get('prelaunch_rejection') is None and (rejection := dry_check_rejection(attempt, assignment)):
                state = save_state(attempt, 'UNKNOWN', **dict(detail, resource_released=True, prelaunch_rejection=rejection))
            elif not detail.get('resource_released'):
                if proof:
                    state = save_state(attempt, 'UNKNOWN', **dict(
                        detail, resource_released=True, automatic_retry=False, pbs_attempt_exit=proof))
                elif not (attempt / PBSAttempt.OWNER).is_file() and resource_state(attempt, 'UNKNOWN') == 'EXITED':
                    state = save_state(attempt, 'UNKNOWN', **dict(detail, resource_released=True))
        return state

    def _prepare_challenge_artifacts(self, challenges):
        hashes=[]
        for assignment in challenges:
            if Preparation.is_preparation(assignment):hashes.extend(Preparation.missing_metadata_bundle_hashes(assignment,self.profile))
        if not hashes:return
        cache=worker_cache(self.profile)
        for sha in cache.missing(hashes):
            try:
                blobs = self._post('/v1/artifacts/get', dict(hashes=[sha]))['blobs']
            except urllib.error.HTTPError as error:
                raise ValueError(f'Cannot fetch preparation bundle {sha}: {error}') from error
            cache.put(blobs)

    def _upload_artifacts(self, attempt, assignment):
        if not Preparation.is_preparation(assignment):return
        manifest = read(attempt/'OUTCOME.json').get('artifacts', {})
        hashes = list(dict.fromkeys(manifest.values()))
        missing = self._post('/v1/artifacts/missing', dict(hashes=hashes))['missing']
        cache = worker_cache(self.profile)
        # Upload one missing immutable object at a time. Reconnect asks again,
        # so completed large objects are never retransmitted with every status.
        for sha in missing:
            self._post('/v1/artifacts/put', dict(blobs=cache.get([sha])))

    def _check_challenges(self, challenges, report):
        return host_fingerprints(self.profile, challenges)

    def _check_readiness_batch(self, challenges, report):
        ready, failed = [], {}
        for assignment in challenges:
            try:
                self._prepare_challenge_artifacts([assignment])
            except (OSError, ValueError) as error:
                failed[assignment['job_id']] = str(error)
            else:
                ready.append(assignment)
        result = self._check_challenges(ready, report)
        result.setdefault('readiness_errors', {}).update(failed)
        return result

    inline_readiness_retry = True

    def _poll_readiness(self, report, accepting):
        challenges = self.eligibility_challenges if accepting else []
        if getattr(self, '_native_readiness', None) is not None:
            result = {}
            pending = self._native_probe
            if pending is not None and pending[0].done():
                future, checked = pending
                self._native_probe = None
                try:
                    current = {item['job_id']: item for item in challenges}
                    valid = {item['job_id'] for item in checked
                             if accepting and current.get(item['job_id'], item) == item}
                    result = {key: {job: fact for job, fact in rows.items() if job in valid}
                              for key, rows in future.result().items()}
                    challenges = [item for item in challenges if item['job_id'] not in valid]
                except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
                    event(self.profile, 'READINESS_OFFLINE', reason=str(error))
            if self._native_probe is None and challenges:
                future = self._native_readiness.submit(self._check_readiness_batch, challenges, report)
                self._native_probe = (future, challenges)
            return result
        return self._check_readiness_batch(challenges, report)

    def _receive_assignment(self, assignment):
        self._acquire(assignment)

    def _service_attempt(self, attempt, assignment, *, transfer=True):
        state = self._reconcile(attempt, assignment)
        if state['state'] == 'ASSIGNED':
            self._launch(attempt); state = read(attempt / 'WORKER_STATE.json')
        payload = dict(job_id=assignment['job_id'], attempt_id=assignment['attempt_id'], state=state['state'], detail=state['detail'])
        if state['state'] != 'TRANSFERRED' and self.reported.get(attempt) != state:
            try:
                self._post('/v1/state', payload); self.reported[attempt] = state
            except (OSError, ValueError, urllib.error.URLError) as error:
                event(self.profile, 'STATE_OFFLINE', job_id=assignment['job_id'], reason=str(error))
        if transfer and state['state'] == 'COMPLETED_PENDING_TRANSFER':
            try:
                started = time.monotonic()
                self._upload_artifacts(attempt, assignment)
                files = result_bundle(attempt)
                response = self._post('/v1/complete', dict(job_id=assignment['job_id'], attempt_id=assignment['attempt_id'],
                    request_sha256=assignment['request_sha256'], files=files))
                if response.get('ok') is not True:
                    raise ValueError('Controller refused completion')
                timing = dict(upload_roundtrip_seconds=time.monotonic()-started,
                              bytes=sum(len(base64.b64decode(v)) for v in files.values()))
                save_state(attempt, 'TRANSFERRED', duplicate=response.get('duplicate', False))
                Storage.reclaim_transferred_attempt(self.profile.get('storage_limit'), attempt)
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

    def _poll_assignments(self, acquire):
        report = self._resource_offer()
        active = report['occupied']
        if acquire and not self.stop.is_set():
            cached = sorted({a.get('prefix_key') for p, a in self.records.items()
                if a.get('prefix_key') and read(p / 'WORKER_STATE.json')['state'] in ('COMPLETED_PENDING_TRANSFER', 'TRANSFERRED')
                })
            try:
                storage_accepting = report.get('storage', {}).get('accepting_new_jobs', True)
                payload = dict(slots=report['slots'], capacity=self.slots, resource_report=report, cached_prefixes=cached,
                               cached_artifact_sha256=worker_cache(self.profile).hashes(),
                               cached_metadata_bundle_sha256=Preparation.cached_metadata_bundle_hashes(self.profile), source_sha=self.source_sha,
                               accepting_new_jobs=storage_accepting, **self._poll_readiness(report, storage_accepting))
                response = self._post('/v1/poll', payload)
                self._capacity_feedback(response)
                self.eligibility_challenges = response.get('eligibility_challenges', [])
                if not self.inline_readiness_retry:
                    self._readiness_delivered(response)
                elif storage_accepting and not response.get('assignments') and self.eligibility_challenges:
                    payload.update(self._poll_readiness(report, storage_accepting))
                    response = self._post('/v1/poll', payload)
                    self._capacity_feedback(response)
                    self.eligibility_challenges = response.get('eligibility_challenges', self.eligibility_challenges)
                assignments = response.get('assignments', [])
                new_assignments = [a for a in assignments if
                    self.work / safe_id(a['job_id']) / safe_id(a['attempt_id']) not in self.records]
                if len(new_assignments) > report['slots']:
                    raise ValueError('Controller overfilled announced slots')
                for assignment in assignments:
                    self._receive_assignment(assignment)
                if response.get('stop'):
                    self.stop.set()
            except (OSError, ValueError, urllib.error.URLError) as error:
                event(self.profile, 'POLL_OFFLINE', reason=str(error))
        return active

    def tick(self, acquire=True):
        pool = getattr(self, '_native_collection', None)
        if pool is not None:
            for attempt, future in list(self._native_receipts.items()):
                if future.done():
                    del self._native_receipts[attempt]
                    future.result()
            for attempt, assignment in list(self.records.items()):
                if len(self._native_receipts) >= self._native_collection_width:
                    break
                if attempt in self._native_receipts or attempt in self._native_retired:
                    continue
                state = read(attempt/'WORKER_STATE.json')
                if state['state'] == 'TRANSFERRED':
                    self._native_retired.add(attempt)
                    continue
                if attempt in getattr(self, 'unowned_attempts', ()):
                    continue
                if (state['state'] == 'UNKNOWN' and self.reported.get(attempt) == state
                        and not (attempt/'OUTCOME.json').exists()):
                    continue
                if (state['state'] == 'RUNNING' and self.reported.get(attempt) == state
                        and not (attempt/'PROCESS_RETURN.json').exists()):
                    process = self.processes.get(attempt)
                    if process is not None and process.poll() is None:
                        continue
                self._native_receipts[attempt] = pool.submit(self._service_attempt, attempt, assignment)
            return self._poll_assignments(acquire)
        for attempt, assignment in list(self.records.items()):
            self._service_attempt(attempt, assignment)
        return self._poll_assignments(acquire)

    def pending(self):
        for attempt in self.records:
            state = read(attempt/'WORKER_STATE.json')
            if (state['state'] not in ('TRANSFERRED', 'UNKNOWN')
                    or PBSAttempt.unreleased_unknown(attempt, state)):
                return True
        return False

    def _open_native_lanes(self):
        # Keep at least one other receipt/launch moving during a bulk transfer.
        self._native_collection_width = max(2, self.slots + 1)
        self._native_collection = ThreadPoolExecutor(max_workers=self._native_collection_width,
                                                     thread_name_prefix='rpe-local-receipt')
        self._native_readiness = ThreadPoolExecutor(max_workers=1, thread_name_prefix='rpe-local-readiness')
        self._native_receipts, self._native_retired, self._native_probe = {}, set(), None
        self.inline_readiness_retry = False

    def _readiness_delivered(self, response):
        """The next poll sends any finished proof for its current challenges."""

    def run(self, *, once=False, drain=False):
        polled = False
        try:
            if type(self) is Worker and not once and not drain:
                self._open_native_lanes()
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
            self.close()

    def close(self):
        for name in ('_native_readiness', '_native_collection'):
            pool = getattr(self, name, None)
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
                setattr(self, name, None)
        self.lock.close()


def _mailbox_submit(worker, payload, *, offer=None):
    assignment = payload.get('assignment', payload)
    safe_id(assignment['job_id']); safe_id(assignment['attempt_id'])
    attempt = worker.work / assignment['job_id'] / assignment['attempt_id']
    duplicate = (attempt / 'ASSIGNMENT.json').exists()
    profile = read(attempt / 'WORKER_PROFILE.json') if duplicate else worker.profile
    validate_assignment(profile, assignment)
    offer = worker._resource_offer() if offer is None else worker._storage_offer(dict(offer))
    if not duplicate and not offer.get('storage', {}).get('accepting_new_jobs', True):
        raise ValueError(offer['storage']['reason'])
    occupied_states = Resources.OCCUPIED if (worker.profile.get('slot_policy') or {}).get('mode') == 'auto' else ('ASSIGNED','STARTING','RUNNING')
    active = sum(resource_state(path, worker._reconcile(path, saved)['state']) in occupied_states
                 for path, saved in worker.records.items())
    if not duplicate and active >= worker.slots:
        raise ValueError('No free worker slot')
    worker._acquire(assignment)
    state = read(attempt / 'WORKER_STATE.json')
    if state['state'] == 'ASSIGNED':
        worker._launch(attempt)
        threading.Thread(target=worker.processes[attempt].wait, daemon=True).start()
    return dict(ok=True, duplicate=duplicate)


def mailbox_submit_batch(profile, assignments):
    """Share the one spool owner and its loaded records across submissions."""
    replies = []
    worker = Worker(profile, 'http://127.0.0.1:1')
    try:
        offer = worker._resource_offer()
        for assignment in assignments:
            try:
                result = _mailbox_submit(worker, assignment, offer=offer)
            except Exception as error:
                result = dict(ok=False, error=type(error).__name__ + ': ' + str(error))
            replies.append(dict(job_id=assignment['job_id'], attempt_id=assignment['attempt_id'],
                                result=result))
        return replies
    finally:
        worker.close()


def mailbox(profile, action, payload=None):
    """One SSH request against the same durable spool used by the HTTP daemon.

    Submit detaches the one-shot supervisor. Metadata-only status advertises
    capacity and attempt state, and result retrieves a specific validated attempt.
    """
    payload = payload or {}
    if action in ('artifacts-missing', 'artifacts-get', 'artifacts-put'):
        cache = worker_cache(profile)
        if action == 'artifacts-missing':
            hashes = payload.get('hashes', [])
            if 'preparation_challenges' in payload and 'direct_hashes' in payload:
                challenges = payload['preparation_challenges']
                direct = set(payload['direct_hashes'])
                declared = {sha for row in challenges for sha in Preparation.metadata_bundle_hashes(row)}
                if set(hashes) != direct | declared:
                    raise ValueError('Different declared preparation artifact hashes')
                hashes = list(direct | {sha for row in challenges
                        for sha in Preparation.missing_metadata_bundle_hashes(row, profile)})
            return dict(missing=cache.missing(hashes))
        if action == 'artifacts-get':return dict(blobs=cache.get(payload.get('hashes', [])))
        return dict(ok=True, **cache.put(payload.get('blobs', {})))
    if action == 'readiness':
        return dict(host_id=profile['host_id'], **host_fingerprints(profile, payload.get('challenges', []),
                                                                 deployment=payload.get('deployment')))
    if action == 'result':
        job = safe_id(payload['job_id']); attempt_id = safe_id(payload['attempt_id'])
        attempt = Path(profile['work_dir']).resolve()/job/attempt_id
        assignment = read(attempt/'ASSIGNMENT.json')
        host = profile['host_id']
        if (payload.get('host_id', host) != host or
                (assignment['job_id'], assignment['attempt_id'], assignment.get('host_id', host),
                 assignment['request_sha256']) != (job, attempt_id, host, payload['request_sha256'])):
            raise ValueError('Completion identity differs from the saved assignment')
        state = read(attempt/'WORKER_STATE.json')['state']
        if state != 'COMPLETED_PENDING_TRANSFER':
            if (state != 'UNKNOWN' or not (attempt/'PROCESS_RETURN.json').is_file()
                    or not (attempt/'OUTCOME.json').is_file()):
                raise ValueError('Attempt has no pending completion')
            with (attempt/'WORKER_PROCESS.lock').open('rb') as lock:
                fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
                saved = read(attempt/'WORKER_PROFILE.json')
                validate_completed(saved, assignment, attempt)
        result = dict(files=result_bundle(attempt))
        if Preparation.is_preparation(assignment):
            result['artifact_manifest'] = read(attempt/'OUTCOME.json').get('artifacts', {})
        return result
    worker = Worker(profile, 'http://127.0.0.1:1')
    try:
        if action == 'resources':
            worker._capacity_feedback(payload)
            report = worker._resource_offer()
            return dict(ok=True, host_id=worker.host_id, capacity=worker.slots, resource_report=report)
        if action == 'submit':
            return _mailbox_submit(worker, payload)
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
            Storage.reclaim_transferred_attempt(worker.profile.get('storage_limit'), attempt)
            return dict(ok=True, duplicate=duplicate)
        if action == 'status':
            metadata_only = payload.get('metadata_only') is True
            attempts = []; cached = set()
            for attempt, assignment in worker.records.items():
                state = worker._reconcile(attempt, assignment)
                if state['state'] == 'ASSIGNED':
                    worker._launch(attempt)
                    threading.Thread(target=worker.processes[attempt].wait, daemon=True).start()
                    state = read(attempt / 'WORKER_STATE.json')
                if state['state'] in ('COMPLETED_PENDING_TRANSFER', 'TRANSFERRED') and assignment.get('prefix_key'):
                    cached.add(assignment['prefix_key'])
                if metadata_only and state['state'] == 'TRANSFERRED':
                    continue
                row = dict(job_id=assignment['job_id'], attempt_id=assignment['attempt_id'],
                    state=state['state'], detail=state['detail'], request_sha256=assignment['request_sha256'])
                if not metadata_only and state['state'] == 'COMPLETED_PENDING_TRANSFER':
                    row['files'] = result_bundle(attempt)
                    if Preparation.is_preparation(assignment):
                        row['artifact_manifest'] = read(attempt/'OUTCOME.json').get('artifacts', {})
                attempts.append(row)
            report = worker._resource_offer()
            return dict(host_id=worker.host_id, slots=report['slots'], capacity=worker.slots, resource_report=report, source_sha=worker.source_sha,
                        accepting_new_jobs=report.get('storage', {}).get('accepting_new_jobs', True),
                        cached_prefixes=sorted(cached), cached_artifact_sha256=worker_cache(profile).hashes(),
                        cached_metadata_bundle_sha256=Preparation.cached_metadata_bundle_hashes(profile),
                        attempts=attempts, metadata_only=metadata_only)
        raise ValueError('Unknown mailbox action')
    finally:
        worker.close()
