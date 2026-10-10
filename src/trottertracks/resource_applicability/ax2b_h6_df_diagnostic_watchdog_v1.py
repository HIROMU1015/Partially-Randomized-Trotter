"""Stdlib supervision for a future saved-integral diagnostic; no science at import."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

from .ax2b_h6_df_diagnostic_contract_v1 import PHASES
from .ax2b_supplement_records_v1 import AtomicWriter
from .ax2b_limits import output_size


class DiagnosticProgress:
    def __init__(self, writer, *, cap, started=None):
        self.writer, self.cap = writer, cap
        self.started = time.monotonic() if started is None else started
        self.count = 0
        self.row = {'phase':None, 'point':'not_started', 'last_completed_record':None,
                    'calls_attempted':{}, 'calls_completed':{}}

    def phase(self, name):
        if name not in PHASES: raise ValueError('DIAGNOSTIC_PHASE')
        self.writer.write('phase_'+name+'.json', {'phase':name,'elapsed':time.monotonic()-self.started})
        self.update(phase=name, point='phase_started')

    def update(self, **values):
        if self.count >= self.cap:
            raise RuntimeError('INPUT_PROGRESS_CAP')
        self.row.update(values)
        self.row.update(progress_sequence=self.count, elapsed_seconds=time.monotonic()-self.started,
                        worker_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        self.writer.write('progress_%04d.json'%self.count, self.row, diagnostic=True)
        self.count += 1


def latest_progress(output):
    rows = sorted(Path(output).glob('progress_[0-9][0-9][0-9][0-9].json'))
    if not rows:
        return None
    if len(rows) > 256 or rows[-1].stat().st_size > 16384:
        raise ValueError('INPUT_PROGRESS_READ_CAP')
    return json.loads(rows[-1].read_text())


def supervise(command, output, *, caps, poll_seconds=.02):
    if set(caps['phase_wall_seconds']) != set(PHASES):
        raise ValueError('INPUT_PHASE_SCHEMA')
    times = (caps['total_wall_seconds'], poll_seconds, *caps['phase_wall_seconds'].values())
    if any(isinstance(v, bool) or not math.isfinite(v) or v <= 0 for v in times) or poll_seconds > 1:
        raise ValueError('INPUT_WALL_CAPS')
    output = Path(output)
    writer = AtomicWriter(output, byte_cap=caps['output_bytes'])
    start, index, transition, log_size = time.monotonic(), 0, 0., 0
    seen, reason, code, process = set(), None, None, None
    env = dict(os.environ)
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS',
                 'NUMBA_NUM_THREADS','RAYON_NUM_THREADS','QISKIT_NUM_PROCS'):
        env[name] = '1'
    env['QISKIT_PARALLEL'] = 'FALSE'
    scratch = output/'scratch'
    scratch.mkdir(exist_ok=False)
    for name in ('TMPDIR','TMP','TEMP','MPLCONFIGDIR','NUMBA_CACHE_DIR'):
        env[name] = str(scratch.resolve())
    def drain(log):
        nonlocal log_size, reason
        for _ in range(16):
            try:
                data = os.read(process.stdout.fileno(), 65536)
            except BlockingIOError:
                break
            if not data:
                break
            room = max(0, min(caps['log_bytes']-log_size,
                             caps['output_bytes']-writer.reserve-output_size(output)))
            log.write(data[:room]); log_size += min(len(data), room)
            if len(data) > room:
                reason = 'WORKER_LOG_OR_OUTPUT_CAP'; break
    try:
        with (output/'worker.log').open('xb', buffering=0) as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       env=env, start_new_session=True, cwd=scratch)
            os.set_blocking(process.stdout.fileno(), False)
            while True:
                elapsed = time.monotonic()-start
                found = {i for i, p in enumerate(PHASES) if (output/('phase_'+p+'.json')).exists()}
                if found and found != set(range(max(found)+1)):
                    reason = 'INVALID_PHASE_ORDER'
                for i in sorted(found-seen):
                    path = output/('phase_'+PHASES[i]+'.json')
                    if path.stat().st_size > 8192:
                        raise ValueError('PHASE_RECORD_SIZE')
                    row = json.loads(path.read_text()); stamp = row.get('elapsed')
                    if (row.get('phase') != PHASES[i] or isinstance(stamp, bool)
                            or not isinstance(stamp, (float,int)) or not math.isfinite(stamp)
                            or stamp < transition or stamp > elapsed+1):
                        raise ValueError('PHASE_TIMESTAMP')
                    if i > 0 and stamp-transition > caps['phase_wall_seconds'][PHASES[index]]:
                        reason = 'COMPLETED_PHASE_WALL_CAP:'+PHASES[index]
                    if i > 0:
                        transition = stamp
                    index = i; seen.add(i)
                drain(log)
                if elapsed > caps['total_wall_seconds']:
                    reason = 'TOTAL_WALL_CAP'
                elif elapsed-transition > caps['phase_wall_seconds'][PHASES[index]]:
                    reason = 'PHASE_WALL_CAP:'+PHASES[index]
                elif output_size(output) > caps['output_bytes']-writer.reserve:
                    reason = 'OUTPUT_CAP'
                code = process.poll()
                if code is not None:
                    drain(log)
                if reason is not None or code is not None:
                    break
                time.sleep(poll_seconds)
    except Exception as error:
        reason = 'WATCHDOG_ERROR:'+type(error).__name__+':'+str(error)[:128]
    except BaseException:
        reason = 'WATCHDOG_INTERRUPTED'
    finally:
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            code = process.wait(); process.stdout.close()
    try:
        path = output/'worker_terminal.json'
        worker = json.loads(path.read_text()) if path.stat().st_size <= 8192 else None
    except (OSError, ValueError):
        worker = None
    try:
        progress = latest_progress(output)
    except (OSError, ValueError) as error:
        progress = None; reason = reason or 'PROGRESS_READ_ERROR:'+type(error).__name__
    success = (reason is None and code == 0 and seen == {0,1,2} and isinstance(worker, dict)
               and worker.get('status') == 'H6_DF_DIAGNOSTIC_RECORDED'
               and worker.get('raw_saved') is True and worker.get('summary_saved') is True
               and worker.get('mandatory_stop') is True and worker.get('next_stage_authorized') is False
               and worker.get('H6_status') == 'H6_NOT_AUTHORIZED'
               and worker.get('N') is None and worker.get('G') is None
               and worker.get('numerical_allowance_certified') is False
               and worker.get('accuracy_eligibility') == 'UNDETERMINED'
               and worker.get('contract_status') == 'DRAFT_NOT_AUTHORIZATION')
    report = {'status':'H6_DF_DIAGNOSTIC_RECORDED' if success else 'H6_DF_DIAGNOSTIC_STOP',
              'reason':reason or (None if success else 'WORKER_FAILED_OR_INCOMPLETE'),
              'worker_exit_code':code,'wall_seconds':time.monotonic()-start,'worker_log_bytes':log_size,
              'worker_terminal':worker,'latest_progress':progress,
              'counter_scope':'last saved boundary only; interrupted remainder unknown',
              'N':None,'G':None,'numerical_allowance_certified':False,'accuracy_eligibility':'UNDETERMINED',
              'H6_status':'H6_NOT_AUTHORIZED','contract_status':'DRAFT_NOT_AUTHORIZATION',
              'mandatory_stop':True,'next_stage_authorized':False,'retry':False,'resume':False}
    writer.write('terminal_status.json', report, terminal=True)
    return report
