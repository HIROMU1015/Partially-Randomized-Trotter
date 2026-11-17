"""Exact saved REQUEST metadata check on a PBS compute node; no Q3 execution."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/home/AbeHiromu/projects/rpe-pbs-20261118/portability-h4-q3-20261118')
CODE = Path('/home/AbeHiromu/projects/rpe-pbs-20261118/releases/925985b4e')
if not os.environ.get('PBS_JOBID'):
    raise RuntimeError('PBS compute allocation required')
sys.path.insert(0, str(CODE / 'src'))
from trottertracks.rpe_plan_readiness import run_server2_dry_checks
from trottertracks.rpe_q3_timing import stamp

started = stamp()
result = run_server2_dry_checks(
    CODE, {'FULL_PF': ROOT / 'ASSIGNMENT.json'}, ROOT / 'PBS_PROFILE.json',
    ROOT / 'dry-check', checker=CODE / 'scripts/check_rpe_q3_host_readiness.py')
(ROOT / 'DRY_CHECK_RUN.json').write_text(json.dumps(dict(
    schema='RPE_PBS_EXACT_SAVED_REQUEST_DRY_CHECK_V1', pbs_job_id=os.environ['PBS_JOBID'],
    hostname=os.uname().nodename, start=started, end=stamp(),
    monotonic_seconds=time.monotonic() - started['monotonic'], result=result,
    scientific_q3_execute_calls=0, signal_calls=0, compile_calls=0), indent=2) + '\n')
if result['methods']['FULL_PF']['status'] != 'REAL_HOST_DRY_CHECK_PASS':
    raise RuntimeError('Full saved REQUEST metadata check did not pass')
