"""Attempt-local stop and occupancy evidence, independent of the PBS pilot."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from .rpe_q3_timing import stamp


OWNER = 'PBS_ATTEMPT_OWNER.json'
REQUEST = 'PBS_ATTEMPT_STOP_REQUEST.json'
INTERRUPTED = 'PBS_ATTEMPT_INTERRUPTED.json'
IDENTITY = ('job_id', 'attempt_id', 'request_sha256')
STOP_GRACE_SECONDS = 10
ATTEMPT_ENV = 'RPE_PBS_ATTEMPT_OWNER'


class AttemptStopped(Exception):
    """An operator stopped this attempt; it is never an automatic retry."""


def read(path):
    try:
        return json.loads(path.read_bytes())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def unreleased_unknown(attempt, state):
    return (state['state'] == 'UNKNOWN' and (attempt/OWNER).is_file()
            and state['detail'].get('resource_released') is not True)


def register(profile, attempt, assignment, write_once):
    """Called by the detached supervisor while it owns WORKER_PROCESS.lock."""
    pbs = profile.get('pbs') or {}
    if not pbs.get('allocation_dir') or not os.environ.get('PBS_JOBID'):
        return
    directory = Path(pbs['allocation_dir']).resolve()
    allocation = read(directory/'ALLOCATION.json') or {}
    pid = os.getpid()
    if os.getsid(0) != pid or allocation.get('job_id') != os.environ['PBS_JOBID']:
        return
    scope = {name: assignment[name] for name in IDENTITY}
    write_once(attempt/OWNER, dict(scope, allocation_dir=str(directory), token=directory.name,
        pbs_job_id=allocation['job_id'], clock_id=stamp()['clock_id'], supervisor_pid=pid))


def local_supervisor(attempt):
    owner = read(attempt/OWNER)
    if owner and owner['supervisor_pid'] == os.getpid() == os.getsid(0):
        return owner
    return None


def stop_requested(attempt, owner):
    request = read(attempt/REQUEST)
    return request == owner


def before_launch(attempt, owner, write_once):
    if owner and stop_requested(attempt, owner):
        write_once(attempt/INTERRUPTED, dict(owner, action='BEFORE_ADAPTER_LAUNCH', stopped_epoch=time.time()))
        raise AttemptStopped


def adapter_environment(owner, profile=None):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    if owner:
        env[ATTEMPT_ENV] = attempt_tag(owner)
    profile=profile or {}
    # Legacy native profiles retain their existing environment. New native
    # profiles may explicitly bind inner-library threads and their footprint.
    threads=profile.get('library_threads')
    if threads is not None:
        if (type(threads) is not int or threads<1 or
                (threads>1 and (type(profile.get('task_cpu_count')) is not int or
                 profile['task_cpu_count']<threads*max(1,profile.get('dp_workers') or 1)))):
            raise ValueError('Native library threads exceed the declared task CPU footprint')
        if (profile.get('pbs') or {}).get('library_threads',threads)!=threads:
            raise ValueError('Native library threads conflict with the PBS execution profile')
        from .rpe_pbs_worker import THREAD_VARIABLES
        env.update({key:str(threads) for key in THREAD_VARIABLES})
    return env


def rollover_owner_live(records,guard,alive):
    """A delayed backfill must still have a positively live original adapter."""
    if guard is None:return True
    owners={tuple(owner) for owner in guard.get('owners',())}
    for path,assignment in tuple(records.items()):
        if (assignment.get('job_id'),assignment.get('attempt_id')) not in owners:continue
        if (path/'PROCESS_RETURN.json').exists():continue
        state=read(path/'WORKER_STATE.json') or {}
        launch=read(path/'WORKER_LAUNCH.json') or {}
        if state.get('state')=='RUNNING' and alive(launch.get('pid'),path):return True
    return False


def attempt_tag(owner):
    return owner['token']+':'+owner['attempt_id']


def wait_adapter(process, attempt, owner, write_once):
    """Signal only the private group of this supervisor's unreaped child."""
    if not owner:
        return process.wait(), False
    while not stop_requested(attempt, owner):
        try:
            return process.wait(timeout=1), False
        except subprocess.TimeoutExpired:
            pass

    group = process.pid
    try:
        if os.getpgid(group) != group or os.getsid(group) != owner['supervisor_pid']:
            # An adapter that changed its isolation must finish normally.
            return process.wait(), False
    except ProcessLookupError:
        return process.wait(), False
    write_once(attempt/INTERRUPTED, dict(owner, action='SIGNAL_ADAPTER_GROUP', stopped_epoch=time.time()))
    try:
        os.killpg(group, signal.SIGTERM)
    except ProcessLookupError:
        return process.wait(), True

    # Keep the child unreaped until escalation or group exit. This pins the
    # group ID even if the adapter exits ahead of one of its children.
    deadline = time.monotonic() + STOP_GRACE_SECONDS
    while time.monotonic() < deadline:
        if not group_has_live_members(group):
            return process.wait(), True
        time.sleep(.25)
    try:
        os.killpg(group, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return process.wait(), True


def group_has_live_members(group):
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if int(fields[2]) == group and fields[0] not in ('Z', 'X'):
            return True
    return False


def detached_child_remains(owner):
    """A still-tagged descendant may have changed its process group or session."""
    target = (ATTEMPT_ENV+'='+attempt_tag(owner)).encode()
    for path in Path('/proc').glob('[0-9]*'):
        try:
            if path.stat().st_uid != os.getuid():
                continue
            if target in (path/'environ').read_bytes().split(b'\0'):
                return True
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return False


def exit_evidence(attempt, assignment, profile, session_ended):
    owner = read(attempt/OWNER)
    if not owner:
        return None
    started = read(attempt/'SUPERVISOR_STARTED.json') or {}
    current = (profile.get('pbs') or {}).get('allocation_dir')
    if (not current or any(owner[key] != assignment.get(key) for key in IDENTITY)
            or owner['supervisor_pid'] != started.get('pid')):
        return None
    if (str(Path(current).resolve()) == owner['allocation_dir']
            and owner['pbs_job_id'] == os.environ.get('PBS_JOBID')
            and owner['clock_id'] == stamp()['clock_id']):
        if not session_ended(attempt) or detached_child_remains(owner):
            return None
        return dict(owner, session_ended=True, operator_stop=(attempt/INTERRUPTED).is_file())
    # Managed keeper writes SESSION_END only after reaping its detached
    # descendants. Bind the receipt to the old allocation and job; neither a
    # missing receipt nor a different current job ID is sufficient evidence.
    prior = Path(owner['allocation_dir'])
    if prior.name != owner['token']:
        return None
    allocation = read(prior/'ALLOCATION.json') or {}
    terminal = read(prior/'ALLOCATION_END.json') or {}
    if (allocation.get('job_id') == owner['pbs_job_id']
            and terminal.get('token') == owner['token']
            and terminal.get('job_id') == owner['pbs_job_id']
            and terminal.get('job_state') in ('F','X')):
        return dict(owner, session_ended=True, operator_stop=(attempt/INTERRUPTED).is_file(),
                    scheduler_allocation_end=terminal)
    session = read(prior/'SESSION_START.json') or {}
    ended = read(prior/'SESSION_END.json') or {}
    if (allocation.get('job_id') != owner['pbs_job_id']
            or session.get('job_id') != owner['pbs_job_id']
            or ended.get('reason') not in ('WORKER_EXIT', 'WORKER_ERROR')
            or type(ended.get('returncode')) is not int
            or type(session.get('epoch')) not in (int, float)
            or type(ended.get('epoch')) not in (int, float)
            or ended['epoch'] < session['epoch']):
        return None
    return dict(owner, session_ended=True, operator_stop=(attempt/INTERRUPTED).is_file(),
                keeper_session_end=ended)
