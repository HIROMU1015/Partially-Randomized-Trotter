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
        return None
    receipt = _read_saved(attempt/'WORKER_DRY-CHECK_RETURN.json')
    return dict(kind='FAILED_DRY_CHECK_BEFORE_EXECUTION', job_id=assignment['job_id'],
                attempt_id=assignment['attempt_id'], request_sha256=assignment['request_sha256'],
                returncode=receipt['returncode'], signal=receipt.get('signal'))
