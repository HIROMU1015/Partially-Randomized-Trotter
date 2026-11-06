"""Unconnected lifecycle prototype, exercised only with synthetic children.

No scientific runner CLI, retries, resource quotas or estimator data access.
Integration with any future N3 SOURCE/worker supervisor remains separate work.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from .shared_cpu import module

atomic_write_json = module('parallel_validation_executor').atomic_write_json


def event(path, kind, **fields):
    row = dict(event=kind, utc=datetime.now(timezone.utc).isoformat(),
               monotonic_ns=time.monotonic_ns(), pid=os.getpid(), **fields)
    with Path(path).open('a') as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def wait_status(returncode):
    if returncode is None:
        return dict(status='UNKNOWN', signal=None, raw_returncode=None)
    if returncode < 0:
        number = -returncode
        try:
            name = signal.Signals(number).name
        except ValueError:
            name = 'UNKNOWN_SIGNAL_NAME'
        return dict(status='SIGNALED', signal=number, signal_name=name,
                    raw_returncode=returncode, sender='UNKNOWN')
    return dict(status='EXITED', exit_code=returncode, signal=None,
                raw_returncode=returncode)


def process_identity(pid, proc=Path('/proc')):
    try:
        # stat fields3.., robust to spaces/parentheses in the comm field.
        values = (proc / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, ppid=int(values[1]), pgid=int(values[2]),
                    sid=int(values[3]), start_ticks=int(values[19]),
                    boot_id=(proc / 'sys/kernel/random/boot_id').read_text().strip(),
                    cgroup=(proc / str(pid) / 'cgroup').read_text().strip())
    except (OSError, ValueError, IndexError) as exc:
        return dict(pid=pid, identity='UNKNOWN', error_type=type(exc).__name__)


def memory_snapshot(cgroup=Path('/sys/fs/cgroup'), proc=Path('/proc'), *, cgroup_record=None):
    result = {}
    for name in ('meminfo', 'pressure/memory', 'pressure/cpu'):
        try:
            result[name] = (proc / name).read_text()
        except OSError:
            result[name] = 'UNKNOWN'
    try:
        record = (proc / 'self/cgroup').read_text() if cgroup_record is None else cgroup_record
        relative = record.strip().split('::', 1)[1]
        group = cgroup / relative.lstrip('/')
        result['cgroup_path'] = str(group)
        result['ancestor_cgroups'] = {}
        while group.is_relative_to(cgroup):
            values = {}
            for name in ('memory.current', 'memory.peak', 'memory.max', 'memory.high',
                         'memory.events', 'memory.events.local', 'pids.current', 'pids.events'):
                try:
                    values[name] = (group / name).read_text()
                except OSError:
                    values[name] = 'UNKNOWN'
            result['ancestor_cgroups'][str(group)] = values
            if group == cgroup / relative.lstrip('/'):
                result.update(values)
            if group == cgroup:
                break
            group = group.parent
    except (OSError, IndexError):
        result['cgroup_path'] = 'UNKNOWN'
    return result


@contextmanager
def traced_call(journal, ticket, name):
    event(journal, 'CALL_START', ticket=ticket, name=name)
    try:
        yield
    except BaseException as exc:
        event(journal, 'CALL_EXCEPTION', ticket=ticket, name=name,
              exception_type=type(exc).__name__)
        raise
    else:
        event(journal, 'CALL_RETURN', ticket=ticket, name=name)


def observe_child(argv, fresh_output, *, poll_seconds=0.25):
    """One owned direct child, no retry or deadline; no shell exit-code guess.

    poll_seconds is heartbeat cadence, not an execution-time or resource cap.
    This prototype cannot record its own SIGKILL, guarantee a durable manager,
    or reap grandchildren. It is not connected to the N3 driver.
    """
    out = Path(fresh_output)
    out.mkdir(parents=True, exist_ok=False)
    journal = out / 'LIFECYCLE.jsonl'
    event(journal, 'SUPERVISOR_START', identity=process_identity(os.getpid()),
          resources=memory_snapshot())
    with (out / 'child.log').open('xb') as log:
        try:
            child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as exc:
            event(journal, 'SPAWN_ERROR', exception_type=type(exc).__name__)
            raise
        identity = process_identity(child.pid)
        group_record = identity.get('cgroup', 'UNKNOWN')
        event(journal, 'CHILD_START', identity=identity, owned_direct_child=True,
              resources=memory_snapshot(cgroup_record=group_record))
        while True:
            try:
                code = child.wait(timeout=poll_seconds)
                break
            except subprocess.TimeoutExpired:
                identity = process_identity(child.pid)
                group_record = identity.get('cgroup', group_record)
                event(journal, 'HEARTBEAT', identity=identity,
                      resources=memory_snapshot(cgroup_record=group_record))
        log.flush()
        os.fsync(log.fileno())
    status = wait_status(code)
    event(journal, 'CHILD_WAIT_STATUS', child_pid=child.pid, **status,
          resources=memory_snapshot(cgroup_record=group_record))
    atomic_write_json(out / 'EXIT.json', status)
    return status
