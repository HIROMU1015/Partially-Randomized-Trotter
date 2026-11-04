"""GDB Python: observe only its own new inferior, never attach to another job."""
import collections
import json
import os
from pathlib import Path
import sys

import gdb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from runtime_policy import REQUIRED, mode_supported, validate_probe, write_json

output = Path(os.environ['RPE_BRIDGE_OBSERVER_OUTPUT'])
counts, modes, threads = collections.Counter(), set(), set()
all_counts = collections.Counter()
phase, sequence, columns, failure, exit_code = 'startup', 0, 0, None, None
checkpoints = []


def mxcsr():
    value = int(gdb.parse_and_eval('$mxcsr'))
    if not mode_supported(value):
        raise ValueError('Unsupported MXCSR: '+hex(value))
    modes.add(value)
    threads.add(gdb.selected_thread().global_num)
    return value


def snapshot():
    original = gdb.selected_thread()
    states = []
    try:
        for thread in gdb.selected_inferior().threads():
            if not thread.is_valid() or thread.is_exited():
                continue
            thread.switch()
            states.append(mxcsr())
    finally:
        if original is not None and original.is_valid():
            original.switch()
    if not states:
        raise ValueError('No inspectable inferior threads')
    return states


def save(status):
    write_json(output/'runtime.json', {'status': status, 'sequence': sequence, 'phase': phase,
        'child_pid': gdb.selected_inferior().pid,
        'counts': dict(counts), 'observed_mxcsr': sorted(modes), 'observed_thread_count': len(threads),
        'all_phase_counts': dict(all_counts),
        'checkpoint_thread_counts': checkpoints, 'probe_output_columns': columns,
        'failure': failure, 'child_exit_code': exit_code})


class Observe(gdb.Breakpoint):
    def __init__(self, name):
        super().__init__(name, internal=True)
        self.name = name

    def stop(self):
        global columns, failure
        try:
            mxcsr()
            all_counts[self.name] += 1
            if phase == 'probe_start':
                counts[self.name] += 1
                if self.name == 'cblas_zgemv64_':
                    values = [int(gdb.parse_and_eval('$'+r)) for r in ('rdi', 'rsi', 'rdx', 'rcx')]
                    alpha = [float(gdb.parse_and_eval('((double*)$r8)['+str(i)+']')) for i in (0, 1)]
                    # GDB may skip the inspected push-rbp/mov-rsp-rbp prologue.
                    offset = int(gdb.parse_and_eval('$pc'))-int(gdb.parse_and_eval('(void*)&cblas_zgemv64_'))
                    stack = '$rsp+32' if offset == 0 else '$rbp+40' if offset == 4 else None
                    if stack is None:
                        raise ValueError('Unrecognized inspected CBLAS entry/prologue')
                    beta = [float(gdb.parse_and_eval('(*(double**)('+stack+'))['+str(i)+']')) for i in (0, 1)]
                    if values != [102, 112, 70, 70] or alpha != [1., 0.] or beta != [0., 0.]:
                        raise ValueError('Unsupported CBLAS probe invocation')
                if self.name == 'zgemv_t_PRESCOTT':
                    inner = int(gdb.parse_and_eval('$rdi')); width = int(gdb.parse_and_eval('$rsi'))
                    if inner != 70 or not 1 <= width <= 70:
                        raise ValueError('Unsupported GEMVT partition')
                    columns += width
                if self.name == 'zgemv_n_PRESCOTT':
                    raise ValueError('Unexpected GEMVN path')
            return False
        except BaseException as exc:
            failure = str(exc)
            save('RUNTIME_UNCONFIRMED_MANDATORY_STOP')
            return True


for command in ('set pagination off', 'set confirm off', 'set breakpoint pending on',
                'set disable-randomization off', 'set print thread-events off', 'handle SIGTRAP stop nopass'):
    gdb.execute(command)
for name in (*REQUIRED, 'zgemv_n_PRESCOTT', 'CDOUBLE_multiply', 'CDOUBLE_subtract'):
    Observe(name)


def exited(event):
    global exit_code
    exit_code = getattr(event, 'exit_code', None)


gdb.events.exited.connect(exited)
try:
    gdb.execute('run')
    while gdb.selected_inferior().pid and failure is None:
        request = json.loads((output/'runtime_request.json').read_text())
        if request['sequence'] != sequence+1:
            raise ValueError('Unexpected stop/checkpoint sequence')
        next_phase = request['phase']
        if next_phase == 'probe_complete':
            validate_probe(counts, columns)
        states = snapshot()
        checkpoints.append({'phase': next_phase, 'threads': len(states)})
        phase, sequence = next_phase, request['sequence']
        save('RUNTIME_CONDITIONS_CONFIRMED')
        gdb.execute('continue')
    if failure is not None:
        gdb.execute('kill')
    elif exit_code != 0 or phase != 'complete':
        raise ValueError('Child did not complete the monitored pipeline')
except BaseException as exc:
    failure = failure or str(exc)
    if gdb.selected_inferior().pid:
        gdb.execute('kill')
finally:
    save('RUNTIME_CONDITIONS_CONFIRMED' if failure is None else 'RUNTIME_UNCONFIRMED_MANDATORY_STOP')
gdb.execute('quit '+('0' if failure is None else '2'))
