"""One-worker watchdog with H6-specific phase budgets; no science launcher."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import time

from .ax2b_h6_contract import PHASES
from .ax2b_h6_controller import BoundedWriter
from .ax2b_limits import output_size


def supervise_synthetic(command, output, *, total_wall_seconds, phase_wall_seconds,
                         output_bytes, log_bytes=65536, poll_seconds=0.02):
    """Synthetic worker integration only. Future science needs a sealed launcher.

    Wall clocks start at process creation (imports included); phase changes
    cannot reset the total timer. Log and aggregate output are separately
    bounded, and partial output survives STOP. Child/grandchildren are killed.
    """
    if set(phase_wall_seconds) != set(PHASES):
        raise ValueError("PHASE_SCHEMA")
    times = (total_wall_seconds, poll_seconds, *phase_wall_seconds.values())
    if any(isinstance(v, bool) or not isinstance(v, (float,int)) or not 0 < v < float("inf") for v in times) or poll_seconds > 1:
        raise ValueError("WALL_CAPS")
    if type(log_bytes) is not int or not 0 < log_bytes <= 65536:
        raise ValueError("LOG_CAP")
    output = Path(output)
    writer = BoundedWriter(output, byte_cap=output_bytes)
    start = phase_start = time.monotonic()
    index, log_size, reason, code = 0, 0, None, None
    process = None
    env = dict(os.environ)
    for name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","NUMBA_NUM_THREADS","QISKIT_NUM_PROCS"):
        env[name] = "1"
    env["QISKIT_PARALLEL"] = "FALSE"
    with (output / "worker.log").open("xb", buffering=0) as log:
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       env=env, start_new_session=True)
            os.set_blocking(process.stdout.fileno(), False)
            while True:
                now = time.monotonic()
                future = [i for i in range(index+1, len(PHASES)) if (output / ("phase_"+PHASES[i]+".json")).exists()]
                # A worker may pass several phases between polls. Preserve
                # order by requiring all intervening markers; no lost phase.
                if future:
                    last = max(future)
                    if any(not (output / ("phase_"+PHASES[i]+".json")).exists() for i in range(index, last+1)):
                        reason = "INVALID_PHASE_ORDER"
                    else:
                        index, phase_start = last, now
                try:
                    data = os.read(process.stdout.fileno(), 65536)
                except BlockingIOError:
                    data = b""
                if data:
                    room = max(0, min(log_bytes-log_size, output_bytes-writer.reserve-output_size(output)))
                    log.write(data[:room]); log_size += min(room, len(data))
                    if len(data) > room:
                        reason = "WORKER_LOG_CAP"
                if now-start > total_wall_seconds:
                    reason = "TOTAL_WALL_CAP"
                elif now-phase_start > phase_wall_seconds[PHASES[index]]:
                    reason = "PHASE_WALL_CAP:" + PHASES[index]
                elif output_size(output) > output_bytes-writer.reserve:
                    reason = "OUTPUT_CAP"
                code = process.poll()
                if code is not None and reason is None:
                    while True:
                        try:
                            tail = os.read(process.stdout.fileno(), 65536)
                        except BlockingIOError:
                            break
                        if not tail:
                            break
                        room = max(0, min(log_bytes-log_size, output_bytes-writer.reserve-output_size(output)))
                        log.write(tail[:room]); log_size += min(room,len(tail))
                        if len(tail) > room:
                            reason = "WORKER_LOG_CAP"
                            break
                if reason is not None or code is not None:
                    break
                time.sleep(poll_seconds)
        except Exception as error:
            reason = "WATCHDOG_ERROR:" + type(error).__name__
        except BaseException:
            reason = "WATCHDOG_INTERRUPTED"
        finally:
            if process is not None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                code = process.wait()
                process.stdout.close()
    witness = output / "worker_terminal.json"
    try:
        worker = json.loads(witness.read_text()) if witness.is_file() and witness.stat().st_size <= 8192 else None
    except (ValueError, OSError):
        worker = None
    success = (reason is None and code == 0 and isinstance(worker,dict)
               and worker.get("status") == "SYNTHETIC_CONTROLLER_COMPLETE"
               and worker.get("completed_correctness_cells") == 7
               and worker.get("compiled_wrappers") == 36
               and worker.get("synthetic_only") is True
               and worker.get("mandatory_stop") is True
               and worker.get("next_stage_authorized") is False)
    report = {"status": "SYNTHETIC_WATCHDOG_COMPLETE" if success else "SYNTHETIC_WATCHDOG_STOP",
              "reason": reason or (None if success else "WORKER_FAILED_OR_INCOMPLETE"),
              "worker_exit_code": code, "wall_seconds": time.monotonic()-start,
              "worker_log_bytes": log_size, "worker_terminal": worker,
              "synthetic_only": True, "H6_status": "H6_NOT_AUTHORIZED",
              "mandatory_stop": True, "next_stage_authorized": False, "retry": False}
    writer.write("terminal_status.json", report, terminal=True)
    return report
