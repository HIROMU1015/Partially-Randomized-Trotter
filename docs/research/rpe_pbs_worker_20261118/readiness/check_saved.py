"""Compute-node metadata inspection of unchanged saved real requests only."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/home/AbeHiromu/projects/rpe-pbs-20261118/readiness')
PYTHON = '/home/AbeHiromu/projects/rpe-pbs-20261118/runtime/venv311rc1/bin/python'
SOURCE = ROOT / 'views/14a6102d4eefea8c37638e71838686a1782d5390'
PROFILE = ROOT / 'payload/PROFILE.json'
OUTPUT = ROOT / 'checks01'


def execute(method):
    command = [PYTHON, '-B', str(SOURCE / 'scripts/check_rpe_q3_host_readiness.py'),
               '--assignment', str(ROOT / 'payload/assignments' / (method + '.json')),
               '--profile', str(PROFILE), '--output', str(OUTPUT / method)]
    started = time.monotonic()
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (OUTPUT / (method + '.log')).write_text(result.stdout)
    return dict(method=method, returncode=result.returncode, duration_seconds=time.monotonic()-started)


def main():
    if not os.environ.get('PBS_JOBID'):
        raise ValueError('Saved real input inspections run only in an allocated PBS node')
    OUTPUT.mkdir(exist_ok=False)
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(execute, ['FULL_PF', 'TRUNCATED_PF', 'PR_PAULI', 'PR_DF']))
    command = [PYTHON, '-B', str(SOURCE / 'scripts/check_rpe_preparation_host_readiness.py'),
               '--tasks', str(ROOT / 'payload/TASKS.json'), '--profile', str(PROFILE),
               '--output', str(OUTPUT / 'preparation')]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (OUTPUT / 'preparation.log').write_text(result.stdout)
    summary = dict(schema='RPE_PBS_SAVED_INPUT_CHECKS_V1', pbs_job_id=os.environ['PBS_JOBID'],
                   hostname=os.uname().nodename, q3_dry_checks=rows,
                   preparation_checker_returncode=result.returncode,
                   same_clock_duration_seconds=time.monotonic()-started,
                   scientific_evaluation_calls=0, compile_calls=0,
                   q3_execution_calls=0, scope='SAVED_REAL_INPUT_METADATA_ONLY')
    (OUTPUT / 'SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
