"""Measure the normal readiness command with an empty then unchanged local cache."""
import argparse
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
INPUTS = ROOT / 'docs/research/rpe_distributed_h4_plan_20261117/inputs'


def run_once(name, base, interpreter):
    output = base / name
    command = [interpreter, '-B', str(ROOT / 'scripts/run_rpe_schedule_search.py'),
        '--problem', str(INPUTS / 'PROBLEM.json'), '--round', str(INPUTS / 'FULL_HISTORY_PLAN_ROUND.json'),
        '--run-dir', str(output), '--plan-only', '--prepare-readiness',
        '--preparation-cache', str(base / 'cache')]
    started, epoch = time.monotonic(), time.time()
    ready = report = None
    with (base / (name + '.COMMAND.log')).open('w') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end='', flush=True)
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not isinstance(row, dict):
                continue
            if row.get('stage') == 'CANDIDATES_AND_REMOTE_INPUTS_AVAILABLE' and row.get('event') == 'READY':
                ready = time.monotonic() - started
            if row.get('stage') == 'REPORT_COMPLETE' and row.get('event') == 'COMPLETE':
                report = time.monotonic() - started
        returncode = process.wait()
    timing = dict(command=command, cwd=str(ROOT), process_start_epoch=epoch,
        command_to_READY_observed_seconds=ready, command_to_REPORT_observed_seconds=report,
        command_to_process_exit_seconds=time.monotonic() - started, returncode=returncode,
        measured_from='Parent monotonic clock before Popen; includes interpreter startup',
        publication='Git publication of the report is measured separately')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'PROCESS_TIMING.json').write_text(json.dumps(timing, indent=2) + '\n')
    if returncode or ready is None or report is None:
        raise SystemExit('Preparation did not publish READY and REPORT successfully: ' + str(output))
    return timing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-base', type=Path, required=True)
    parser.add_argument('--interpreter', default='/home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python')
    args = parser.parse_args()
    base = args.output_base.resolve()
    base.mkdir(parents=True, exist_ok=False)
    timings = {name: run_once(name, base, args.interpreter) for name in ('first', 'second')}
    (base / 'BENCHMARK_TIMING.json').write_text(json.dumps(timings, indent=2) + '\n')
    print('BENCHMARK_COMPLETE ' + json.dumps(timings), flush=True)


if __name__ == '__main__':
    main()
