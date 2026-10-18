"""Observe only the two sealed H8 D jobs; never control or launch science."""
import argparse
import json
import os
from pathlib import Path
import time


CASES = ('B0_p15_q8_2nd', 'B0I_p16_q4_4th')


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def process(pid, *, expected_start=None, driver=False):
    try:
        path = Path('/proc') / str(pid)
        fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        if expected_start is not None and int(fields[19]) != expected_start:
            return dict(observed=False, reason='PID_IDENTITY_CHANGED')
        if driver and b'run_h8_dba_direct.py\0' not in (path / 'cmdline').read_bytes():
            return dict(observed=False, reason='DRIVER_IDENTITY_NOT_CONFIRMED')
        return dict(observed=True, state=fields[0], start_ticks=int(fields[19]),
                    cpu_seconds=(int(fields[11]) + int(fields[12])) / os.sysconf('SC_CLK_TCK'),
                    RSS_bytes=int(fields[21]) * os.sysconf('SC_PAGE_SIZE'), threads=int(fields[17]))
    except (OSError, ValueError, IndexError):
        return dict(observed=False, reason='PROCESS_NOT_OBSERVABLE')


def snapshot(private):
    cases = {}
    for case in CASES:
        base = private / case
        out = base / 'output'
        started = read(base / 'native_started.json')
        native_exit = read(base / 'native_exit.json')
        terminal = read(out / 'terminal.json')
        result = read(out / 'result.json')
        stop = read(out / 'STOP.json')
        final_gate = read(out / 'final_gate.json')
        fresh_gate = read(out / 'fresh_gate.json')
        measured = [read(out / ('measured_' + axis + '.json')) for axis in ('cosine', 'sine')]
        measured = [x for x in measured if x is not None]
        valid = bool(started and native_exit and terminal and result and final_gate
                     and native_exit.get('exit_code') == 0
                     and native_exit.get('same_parent_reaped') is True
                     and native_exit.get('wait4_complete') is True
                     and terminal.get('exit_code') == 0
                     and terminal.get('status') == 'COMPLETE_MANDATORY_STOP'
                     and result.get('status') == 'COMPLETE_MANDATORY_STOP'
                     and final_gate.get('status') == 'PASS'
                     and len(measured) == 2
                     and all(x.get('model_check', {}).get('PASS') is True for x in measured)
                     and [x.get('axis') for x in measured] == ['cosine', 'sine']
                     and all(x.get('SOURCE') == started['SOURCE'] and x.get('case_id') == case
                             for x in measured)
                     and all(x.get('SOURCE') == started['SOURCE']
                             for x in (native_exit, terminal, result)))
        proc = process(started['pid'], driver=True) if started else dict(observed=False)
        status = ('COMPLETE_MANDATORY_STOP' if valid else 'STOP_OR_INCOMPLETE_TERMINAL') if native_exit else (
            'STOP_NATIVE_EXIT_PENDING' if stop else 'RUNNING' if proc.get('observed') else 'UNCONFIRMED')
        workers = []
        for directory in (case + '_directional_controls', case + '_axis_controls_and_fingerprints',
                          'compile_cosine', 'compile_sine'):
            root = out / directory
            if not root.exists():
                continue
            for work in root.iterdir():
                if not work.is_dir():
                    continue
                worker = read(work / 'started.json')
                if worker:
                    workers.append(dict(phase=directory, task=worker['task'], pid=worker['pid'],
                        process=process(worker['pid'], expected_start=worker['identity'][1]),
                        result_saved=(work / 'task_result.json').exists(),
                        STOP=read(work / 'task_STOP.json')))
        if (out / 'control_preparation.json').exists():
            phase = 'COMPILE_OR_AXIS_CHECK'
        elif (out / (case + '_axis_controls_and_fingerprints')).exists():
            phase = 'AXIS_CONTROLS_FINGERPRINTS_QPY'
        elif (out / (case + '_directional_controls')).exists():
            phase = 'DIRECTIONAL_CONTROLS'
        else:
            phase = 'EVOLUTION_BUILD_OR_INPUT_GATE'
        cases[case] = dict(status=status, phase=phase, SOURCE=started['SOURCE'] if started else None,
            driver=proc, workers=workers, fresh_gate_PASS=bool(fresh_gate and fresh_gate.get('status') == 'PASS'),
            final_gate_PASS=bool(final_gate and final_gate.get('status') == 'PASS'), measured_axes=measured,
            STOP_reason=stop.get('reason') if stop else None, native_exit=native_exit)
    return dict(time_unix=time.time(), cases=cases, observer_never_controls_workers=True,
                automatically_starts_next_stage=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--public', type=Path)
    args = parser.parse_args()
    while True:
        value = snapshot(args.private)
        if not args.watch:
            print(json.dumps(value, indent=2, allow_nan=False))
            return
        write(args.private / 'progress.json', value)
        with (args.private / 'progress_history.jsonl').open('a') as log:
            log.write(json.dumps(value, allow_nan=False) + '\n')
        if all(x['native_exit'] is not None for x in value['cases'].values()):
            write(args.private / 'completion_summary.json', value)
            if args.public:
                args.public.mkdir(parents=True, exist_ok=True)
                write(args.public / 'completion_status.json', value)
                for case in CASES:
                    out = args.private / case / 'output'
                    result = read(out / 'result.json')
                    # Public outputs contain no QPY paths or private configuration.
                    if result is not None and '/home/' not in json.dumps(result):
                        write(args.public / (case + '_result.json'), result)
                    write(args.public / (case + '_terminal.json'), read(out / 'terminal.json'))
            return
        time.sleep(10)


if __name__ == '__main__':
    main()
