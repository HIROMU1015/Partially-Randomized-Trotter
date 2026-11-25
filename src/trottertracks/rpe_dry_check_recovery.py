"""Read-only dry-check exit evidence shared by the worker and collectors."""
import fcntl
import json


def _read_saved(path):
    return json.loads(path.read_text())


def failed_dry_check_released(attempt):
    """Check the supervisor's original files, including from a shared PBS spool."""
    try:
        with (attempt / 'WORKER_PROCESS.lock').open('rb') as lock:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            state = _read_saved(attempt / 'WORKER_STATE.json')
            detail = state['detail']
            receipt = _read_saved(attempt / 'WORKER_DRY-CHECK_RETURN.json')
            pid = _read_saved(attempt / 'SUPERVISOR_STARTED.json')['pid']
            launch_pid = _read_saved(attempt / 'SUPERVISOR_LAUNCH.json')['pid']
            return (state['state'] == 'UNKNOWN' and detail.get('automatic_retry') is False
                    and detail.get('reason') == 'ValueError: Existing adapter dry-check failed'
                    and type(receipt['returncode']) is int and receipt['returncode'] > 0
                    and receipt.get('signal') is None
                    and type(pid) is int and pid > 0 and launch_pid == pid
                    and not any((attempt / name).exists() for name in (
                        'SCIENTIFIC_STARTING.json', 'WORKER_LAUNCH.json', 'WORKER_EXECUTE.log',
                        'WORKER_EXECUTE_RETURN.json', 'PROCESS_RETURN.json', 'OUTCOME.json',
                        'PREPARATION_STARTED.json', 'PREPARATION_BODY.log', 'PREPARATION_RECEIPT.json')))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def dry_check_rejection(attempt, assignment):
    """Identify an ended dry-check using the original attempt's durable files."""
    if not failed_dry_check_released(attempt):
        return supervisor_prelaunch_rejection(attempt, assignment)
    receipt = _read_saved(attempt/'WORKER_DRY-CHECK_RETURN.json')
    return dict(kind='FAILED_DRY_CHECK_BEFORE_EXECUTION', job_id=assignment['job_id'],
                attempt_id=assignment['attempt_id'], request_sha256=assignment['request_sha256'],
                returncode=receipt['returncode'], signal=receipt.get('signal'))


_SCIENCE_MARKERS = ('SCIENTIFIC_STARTING.json', 'WORKER_LAUNCH.json', 'WORKER_EXECUTE.log',
    'WORKER_EXECUTE_RETURN.json', 'PROCESS_RETURN.json', 'OUTCOME.json',
    'PREPARATION_STARTED.json', 'PREPARATION_BODY.log', 'PREPARATION_RECEIPT.json')


def supervisor_prelaunch_rejection(attempt, assignment):
    """The supervisor itself recorded an exception before scientific launch.

    A missing adapter exit is not proof. The receipt must bind the assignment,
    match the saved supervisor and state, and its process lock must be released.
    """
    try:
        with (attempt/'WORKER_PROCESS.lock').open('rb') as lock:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            receipt = _read_saved(attempt/'SUPERVISOR_PRELAUNCH_FAILURE.json')
            state = _read_saved(attempt/'WORKER_STATE.json')
            started = _read_saved(attempt/'SUPERVISOR_STARTED.json')
            launch = _read_saved(attempt/'SUPERVISOR_LAUNCH.json')
            if (state['state'] != 'UNKNOWN' or state['detail'].get('automatic_retry') is not False
                    or receipt.get('reason') != state['detail'].get('reason')
                    or receipt.get('returncode') != 2 or receipt.get('scientific_execution_started') is not False
                    or type(started['pid']) is not int or started['pid'] <= 0
                    or started['pid'] != launch['pid'] or started['pid'] != receipt.get('supervisor_pid')
                    or not all(receipt.get(k) == assignment[k] for k in ('job_id','attempt_id','request_sha256'))
                    or any((attempt/name).exists() for name in _SCIENCE_MARKERS)):
                return None
            return dict(kind='SUPERVISOR_FAILED_BEFORE_EXECUTION', job_id=assignment['job_id'],
                        attempt_id=assignment['attempt_id'], request_sha256=assignment['request_sha256'],
                        reason=receipt['reason'], returncode=2, signal=None, scientific_execution_started=False)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
