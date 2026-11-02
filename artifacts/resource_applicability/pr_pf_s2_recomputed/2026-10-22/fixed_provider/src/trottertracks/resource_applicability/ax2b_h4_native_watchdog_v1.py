"""One H4-P worker; startup/imports and all cells share a single deadline."""
from __future__ import annotations

import math
import os
from pathlib import Path
import signal
import subprocess
import time

from .ax2a_preparation import digest
from .ax2b_h4_contract_v5 import file_hash
from .ax2b_h4_native_receipt_v1 import (
    KIND, FORBIDDEN, assemble_bounds, bounded_json, plan,
)
from .ax2b_h6_controller import BoundedWriter
from .ax2b_limits import output_size


def verify_terminal(output, manifest, static):
    output = Path(output)
    terminal = bounded_json(output/'worker_terminal.json', 8192)
    expected_calls = {'snapshot_loads': 1, 'native_preparation_calls': 8, **dict.fromkeys(FORBIDDEN, 0)}
    if (terminal.get('status') != KIND+'_COMPLETE' or terminal.get('reason') is not None
            or terminal.get('manifest_digest') != digest(manifest)
            or terminal.get('calls') != expected_calls or terminal.get('completed_preparations') != 8
            or terminal.get('synthetic_only') is not False
            or terminal.get('science_manifest_sealed') is not False):
        raise ValueError('WORKER_FAILED_OR_INCOMPLETE')
    receipt_path = output/'native_receipt.json'
    if file_hash(receipt_path) != terminal.get('receipt_sha256'):
        raise ValueError('NATIVE_RECEIPT_HASH')
    receipt = bounded_json(receipt_path)
    records = [bounded_json(output/(cell['id']+'_native.json')) for cell in plan()['cells']]
    bounds = assemble_bounds(records, static)
    if (receipt.get('schema') != 'track_a_ax2b_h4_native_receipt_v1'
            or receipt.get('kind') != KIND or receipt.get('manifest_digest') != digest(manifest)
            or receipt.get('source_commit') != manifest['source_commit']
            or receipt.get('input_binding') != manifest['input_binding']
            or receipt.get('environment') != manifest['environment']
            or receipt.get('cell_receipt_digests') != {r['cell_id']: digest(r) for r in records}
            or receipt.get('coverage_binding') != {'sealed': True, 'actual_rank': 12,
                'schedule_digest': digest(plan()['cells']), 'expected_bounds': bounds}
            or receipt.get('synthetic_only') is not False
            or receipt.get('science_manifest_sealed') is not False
            or receipt.get('structural_bounds_are_not_compiled_costs') is not True):
        raise ValueError('RECEIPT_BINDING_OR_COVERAGE')
    for row in (terminal, receipt):
        if ('N' not in row or 'G' not in row or row['N'] is not None or row['G'] is not None
                or row.get('numerical_allowance_certified') is not False
                or row.get('accuracy_eligibility') != 'UNDETERMINED'
                or row.get('mandatory_stop') is not True or row.get('next_stage_authorized') is not False
                or row.get('H6_status') != 'H6_NOT_AUTHORIZED'
                or row.get('contract_status') != 'DRAFT_NOT_AUTHORIZATION'):
            raise ValueError('MANDATORY_STOP_OR_SCIENCE_FLAGS')
    return terminal


def supervise(command, output, *, caps, manifest, static, poll_seconds=.02):
    if (not command or isinstance(caps['total_wall_seconds'], bool)
            or not math.isfinite(caps['total_wall_seconds']) or caps['total_wall_seconds'] <= 0
            or not 0 < poll_seconds <= 1 or caps['log_bytes'] <= 0):
        raise ValueError('WATCHDOG_CAPS')
    output = Path(output)
    writer = BoundedWriter(output, byte_cap=caps['output_bytes'])
    started, log_size, reason, code, process = time.monotonic(), 0, None, None, None
    env = dict(os.environ)
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS', 'RAYON_NUM_THREADS', 'QISKIT_NUM_PROCS'):
        env[name] = '1'
    env['QISKIT_PARALLEL'] = 'FALSE'

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
            log.write(data[:room])
            log_size += min(room, len(data))
            if len(data) > room:
                reason = 'WORKER_LOG_OR_OUTPUT_CAP'
                break

    try:
        with (output/'worker.log').open('xb', buffering=0) as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       env=env, start_new_session=True)
            os.set_blocking(process.stdout.fileno(), False)
            while True:
                drain(log)
                if time.monotonic()-started > caps['total_wall_seconds']:
                    reason = 'TOTAL_WALL_CAP'
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
            # Descendants also die on normal exit; no second worker/retry.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            code = process.wait()
            process.stdout.close()
    terminal = None
    if reason is None and code == 0:
        try:
            terminal = verify_terminal(output, manifest, static)
        except (ValueError, OSError, KeyError, TypeError, IndexError) as error:
            reason = 'TERMINAL_INVALID:'+str(error)[:256]
    else:
        reason = reason or 'WORKER_NONZERO_EXIT'
        try:
            terminal = bounded_json(output/'worker_terminal.json', 8192)
        except (ValueError, OSError):
            pass
    report = {'status': KIND+('_COMPLETE' if reason is None else '_STOP'),
        'reason': reason, 'worker_terminal': terminal, 'worker_exit_code': code,
        'wall_seconds': time.monotonic()-started, 'worker_log_bytes': log_size,
        'output_bytes_before_terminal': output_size(output), 'watchdog_poll_seconds': poll_seconds,
        'manifest_digest': digest(manifest), 'mandatory_stop': True,
        'next_stage_authorized': False, 'science_manifest_sealed': False,
        'N': None, 'G': None, 'numerical_allowance_certified': False,
        'accuracy_eligibility': 'UNDETERMINED', 'H6_status': 'H6_NOT_AUTHORIZED',
        'contract_status': 'DRAFT_NOT_AUTHORIZATION', 'retry': False, 'resume': False}
    writer.write('terminal_status.json', report, terminal=True)
    return report
