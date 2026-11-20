"""Add resume preparation to the shared bootstrap without changing pinned controllers."""

import json
from pathlib import Path
import sqlite3
import subprocess
import sys


def preflight_with_archives(state, normal_preflight, request_error):
    """Run under the old Gateway's regular registry lock and HOLD checks."""
    restore = Path(__file__).with_name('restore_rpe_preparation_bundle.py')

    def preflight(run, problem, spec):
        try:
            completed = subprocess.run([sys.executable, '-B', str(restore), '--resume-run', str(run),
                                        '--state-directory', str(state)], capture_output=True, text=True, check=True)
            recovered = json.loads(completed.stdout)
        except subprocess.CalledProcessError as error:
            raise request_error('preparation_archives', error.stderr.strip()[-1500:]) from error
        report = normal_preflight(run, problem, spec)
        if any(recovered.values()):
            report['preparation_archive_recovery'] = recovered
        return report

    preflight.restores_preparation_archives = True
    return preflight


def describe_resume_hold(state, request):
    """Return central observations only. A RUNNING row is not a live worker probe."""
    state = Path(state)
    run_id = request.get('run_id')
    if run_id:
        entry = json.loads((state/'RUNS.json').read_bytes()).get(run_id)
    else:
        entry = json.loads((state/'CURRENT_TARGET.json').read_bytes())
        run_id = entry['run_id']
    if entry is None:
        return {}
    run = Path(entry['run_directory']).resolve()
    launch_path = run/'LAUNCH.json'
    exit_path = run/'EXIT.json'
    launch = json.loads(launch_path.read_bytes()) if launch_path.is_file() else {}
    finished = json.loads(exit_path.read_bytes()) if exit_path.is_file() else {}
    invocation = launch.get('invocation_id')
    result = dict(run_id=run_id, invocation_id=invocation,
                  matching_exit_receipt=bool(invocation and invocation == finished.get('invocation_id')))
    database = run/'distributed/QUEUE.sqlite3'
    if database.is_file():
        with sqlite3.connect(database.as_uri()+'?mode=ro', uri=True) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT job_id, method, state, host_id, attempt_id, detail FROM jobs "
                "WHERE state IN ('ASSIGNED','STARTING','RUNNING','COMPLETED_PENDING_TRANSFER','UNKNOWN') "
                "ORDER BY updated_epoch, job_id")
            result['owned_attempts'] = [
                dict(job_id=row['job_id'], method=row['method'], ledger_state=row['state'], host=row['host_id'],
                     attempt_id=row['attempt_id'], reason=json.loads(row['detail']).get('reason'))
                for row in rows]
    return result
