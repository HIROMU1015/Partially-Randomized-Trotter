"""Bounded single-worker supervision; terminal success is technical only."""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
from .ax2b_bound_launch_v2 import PHASES
from .ax2b_h6_controller import BoundedWriter
from .ax2b_limits import output_size


def supervise(command, output, *, caps, kind, poll_seconds=.02):
    if kind not in ('H4_LIMITED','H6_TECHNICAL') or set(caps['phase_wall_seconds']) != set(PHASES):
        raise ValueError('SUPERVISION_SCOPE')
    if any(isinstance(x,bool) or not math.isfinite(x) or x <= 0 for x in
           (caps['total_wall_seconds'],*caps['phase_wall_seconds'].values(),poll_seconds)):
        raise ValueError('WALL_CAPS')
    output = Path(output)
    writer = BoundedWriter(output,byte_cap=caps['output_bytes'])
    start, index, transition, log_size, reason = time.monotonic(),0,0.,0,None
    seen = set()
    env = dict(os.environ)
    for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS',
              'NUMBA_NUM_THREADS','RAYON_NUM_THREADS','QISKIT_NUM_PROCS'):
        env[k] = '1'
    env['QISKIT_PARALLEL'] = 'FALSE'
    process, code = None,None
    def drain(log):
        nonlocal log_size, reason
        for _ in range(16):
            try:
                data = os.read(process.stdout.fileno(),65536)
            except BlockingIOError:
                break
            if not data:
                break
            room = max(0,min(caps['log_bytes']-log_size,
                           caps['output_bytes']-writer.reserve-output_size(output)))
            log.write(data[:room]); log_size += min(room,len(data))
            if len(data) > room:
                reason = 'WORKER_LOG_OR_OUTPUT_CAP'; break
    try:
        with (output/'worker.log').open('xb',buffering=0) as log:
            process = subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            os.set_blocking(process.stdout.fileno(),False)
            while True:
                elapsed = time.monotonic()-start
                found = {i for i,p in enumerate(PHASES) if (output/('phase_'+p+'.json')).exists()}
                if found and found != set(range(max(found)+1)):
                    reason = 'INVALID_PHASE_ORDER'
                for i in sorted(found-seen):
                    path = output/('phase_'+PHASES[i]+'.json')
                    if path.stat().st_size > 8192:
                        raise ValueError('PHASE_RECORD_SIZE')
                    row = json.loads(path.read_text()); stamp = row.get('elapsed')
                    if (row.get('phase') != PHASES[i] or isinstance(stamp,bool)
                            or not isinstance(stamp,(int,float)) or not math.isfinite(stamp)
                            or stamp < transition or stamp > elapsed+1):
                        raise ValueError('PHASE_TIMESTAMP')
                    if i > 0 and stamp-transition > caps['phase_wall_seconds'][PHASES[index]]:
                        reason = 'COMPLETED_PHASE_WALL_CAP:'+PHASES[index]
                    # First phase includes startup/imports (parent origin).
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
                os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:
                pass
            code = process.wait(); process.stdout.close()
    try:
        path = output/'worker_terminal.json'
        terminal = json.loads(path.read_text()) if path.stat().st_size <= 8192 else None
    except (ValueError,OSError):
        terminal = None
    count,cost = (8,0) if kind == 'H4_LIMITED' else (7,36)
    success = (reason is None and code == 0 and seen == {0,1,2} and isinstance(terminal,dict)
               and terminal.get('status') == kind+'_COMPLETE' and terminal.get('completed_correctness_cells') == count
               and terminal.get('compiled_wrappers') == cost and terminal.get('mandatory_stop') is True
               and terminal.get('next_stage_authorized') is False and terminal.get('N') is None and terminal.get('G') is None
               and terminal.get('numerical_allowance_certified') is False and terminal.get('accuracy_eligibility') == 'UNDETERMINED')
    report = {'status':kind+'_COMPLETE' if success else kind+'_STOP','worker_terminal':terminal,
              'reason':reason or (None if success else 'WORKER_FAILED_OR_INCOMPLETE'),
              'worker_exit_code':code,'wall_seconds':time.monotonic()-start,'worker_log_bytes':log_size,
              'mandatory_stop':True,'next_stage_authorized':False,'N':None,'G':None,'retry':False}
    writer.write('terminal_status.json',report,terminal=True)
    return report
