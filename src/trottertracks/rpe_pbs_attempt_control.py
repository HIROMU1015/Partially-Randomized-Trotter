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


def adapter_environment(owner):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    if owner:
        env[ATTEMPT_ENV] = attempt_tag(owner)
    return env


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
    if (not current or str(Path(current).resolve()) != owner['allocation_dir']
            or owner['pbs_job_id'] != os.environ.get('PBS_JOBID')
            or owner['clock_id'] != stamp()['clock_id']
            or any(owner[key] != assignment.get(key) for key in IDENTITY)
            or owner['supervisor_pid'] != started.get('pid')
            or not session_ended(attempt) or detached_child_remains(owner)):
        return None
    return dict(owner, session_ended=True, operator_stop=(attempt/INTERRUPTED).is_file())
