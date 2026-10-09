"""Independent Linux watchdog and counters; no numerical libraries."""
from __future__ import annotations

import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time


class CallBudget:
    def __init__(self, **limits):
        if any(type(v) is not int or v < 0 for v in limits.values()):
            raise ValueError('Integer nonnegative call caps required.')
        self.limits = dict(limits)
        self.used = dict.fromkeys(limits, 0)

    def take(self, name, count=1):
        if type(count) is not int or count < 0 or name not in self.limits:
            raise ValueError('Invalid budget request.')
        if self.used[name] + count > self.limits[name]:
            raise RuntimeError('CALL_BUDGET:' + name)
        self.used[name] += count


def exclusive_json(path, value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n'
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(payload)


def output_size(directory):
    return sum(p.stat().st_size for p in Path(directory).rglob('*') if p.is_file())


def install_worker_limits(cpu, address_space_bytes, output_bytes):
    """Called in the exec'd child before importing NumPy/SciPy/Qiskit."""
    if cpu not in os.sched_getaffinity(0):
        raise ValueError('ASSIGNED_CPU_UNAVAILABLE')
    os.sched_setaffinity(0, {cpu})
    resource.setrlimit(resource.RLIMIT_AS, (address_space_bytes, address_space_bytes))
    resource.setrlimit(resource.RLIMIT_FSIZE, (output_bytes, output_bytes))
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS', 'RAYON_NUM_THREADS', 'QISKIT_NUM_PROCS'):
        if os.environ.get(key) != '1':
            raise ValueError('WORKER_THREAD_ENV:' + key)


def supervise(command, output, *, total_wall_seconds, phase_wall_seconds,
              output_bytes, poll_seconds=0.05):
    """External watchdog works while native BLAS/transpilation is blocked.

    Only one child is started, with no retry or resume. Phase deadline changes
    must match the three monotonic registered phases; no timer resets on cells.
    Output has reserved room for the small parent terminal report. RLIMIT_FSIZE
    independently limits each child file; the parent polls aggregate bytes.
    """
    if min(total_wall_seconds, phase_wall_seconds, output_bytes, poll_seconds) <= 0 or poll_seconds > 1:
        raise ValueError('Invalid watchdog limits.')
    output = Path(output)
    reserve = min(65536, output_bytes // 4)
    env = dict(os.environ)
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS', 'RAYON_NUM_THREADS', 'QISKIT_NUM_PROCS'):
        env[key] = '1'
    env['QISKIT_PARALLEL'] = 'FALSE'
    phases = ('input_reference', 'correctness', 'wrapper_cost')
    start = phase_start = time.monotonic()
    phase_index = 0
    reason = None
    code = None
    log_bytes = 0
    log_cap = min(65536, output_bytes // 8)
    with (output / 'worker.log').open('xb', buffering=0) as log:
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       env=env, start_new_session=True)
        except OSError:
            process = None
            reason = 'WORKER_START_FAILED'
        if process is not None:
            os.set_blocking(process.stdout.fileno(), False)
        def drain_log():
            nonlocal log_bytes, reason
            try:
                data = os.read(process.stdout.fileno(), 65536)
            except BlockingIOError:
                return False
            if not data:
                return False
            room = max(0, min(log_cap - log_bytes, output_bytes - reserve - output_size(output)))
            kept = data[:room]
            log.write(kept)
            log_bytes += len(kept)
            if len(kept) != len(data):
                reason = 'WORKER_LOG_CAP'
            return True
        try:
            while process is not None:
                now = time.monotonic()
                drain_log()
                for index in range(phase_index + 1, len(phases)):
                    if (output / ('phase_' + phases[index] + '.json')).exists():
                        if index != phase_index + 1:
                            reason = 'INVALID_PHASE_ORDER'
                        else:
                            phase_index, phase_start = index, now
                        break
                if now - start > total_wall_seconds:
                    reason = 'TOTAL_WALL_CAP'
                elif now - phase_start > phase_wall_seconds:
                    reason = 'PHASE_WALL_CAP'
                elif output_size(output) > output_bytes - reserve:
                    reason = 'OUTPUT_CAP'
                code = process.poll()
                if code is not None:
                    while reason is None and drain_log():
                        pass
                if reason is not None or code is not None:
                    break
                time.sleep(poll_seconds)
        except Exception as error:
            reason = 'WATCHDOG_ERROR:' + type(error).__name__
        except BaseException:
            reason = 'WATCHDOG_INTERRUPTED'
        finally:
            if process is not None:
                # Kill any descendants as well, even if the direct worker
                # has exited. No second scientific worker is launched.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                code = process.wait()
                process.stdout.close()
    witness = output / 'worker_terminal.json'
    try:
        worker = json.loads(witness.read_text()) if witness.exists() and witness.stat().st_size <= min(8192, reserve // 2) else None
    except (ValueError, OSError):
        worker = None
    success = (reason is None and code == 0 and isinstance(worker, dict)
               and worker.get('status') == 'H4_TECHNICAL_PILOT_COMPLETE'
               and worker.get('completed_correctness_cells') == 8
               and worker.get('compiled_wrappers') == 28)
    report = {'status': 'H4_TECHNICAL_PILOT_COMPLETE' if success else 'H4_TECHNICAL_PILOT_STOP',
              'reason': reason or (None if success else 'WORKER_FAILED_OR_INCOMPLETE'),
              'worker_exit_code': code, 'worker_terminal': worker,
              'wall_seconds': time.monotonic() - start, 'output_bytes_before_terminal': output_size(output),
              'worker_log_bytes': log_bytes, 'watchdog_poll_seconds': poll_seconds,
              'mandatory_stop': True, 'next_stage_authorized': False, 'retry': False}
    exclusive_json(output / 'terminal_status.json', report)
    return report
