"""Linux single numerical worker, metadata-only parent with hard wall cutoff."""
import os
import ctypes
import signal
import time


def supervise(worker, seconds):
    start = time.monotonic()
    parent = os.getpid()
    pid = os.fork()
    if pid == 0:
        try:
            # Linux PR_SET_PDEATHSIG: do not leave a science worker orphaned.
            libc = ctypes.CDLL(None, use_errno=True)
            if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0 or os.getppid() != parent:
                os._exit(3)
            status = worker()
            os._exit(0 if status == 'GAMMA_CONDITIONS_COMPLETE_AWAITING_SCIENTIFIC_JUDGMENT' else 2)
        except BaseException:
            os._exit(3)
    timed_out = False
    interruption = None
    def terminate(signum, frame):
        raise RuntimeError('supervisor terminated')
    old_term = signal.signal(signal.SIGTERM, terminate)
    try:
        while True:
            finished, wait_status = os.waitpid(pid, os.WNOHANG)
            if finished:
                break
            remaining = seconds - (time.monotonic()-start)
            if remaining <= 0:
                timed_out = True
                os.kill(pid, signal.SIGKILL)
                _, wait_status = os.waitpid(pid, 0)
                break
            time.sleep(min(0.02, remaining))
    except BaseException as exc:
        interruption = type(exc).__name__ + ': ' + str(exc)[:1024]
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        _, wait_status = os.waitpid(pid, 0)
    finally:
        signal.signal(signal.SIGTERM, old_term)
    exitcode = os.waitstatus_to_exitcode(wait_status)
    return {'worker_exitcode': exitcode, 'wall_cap_reached': timed_out,
            'elapsed_wall_seconds': time.monotonic()-start,
            'interruption': interruption,
            'status': ('COMPLETE_WORKER_EXIT' if not timed_out and interruption is None and exitcode == 0 else 'TECHNICAL_STOP'),
            'mandatory_STOP': True, 'retry': False}
