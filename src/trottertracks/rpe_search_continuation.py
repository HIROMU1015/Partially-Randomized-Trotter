"""Verify a technical-stop handoff without mutating its original ledger."""
import json
from pathlib import Path
import sqlite3

from .rpe_release_registry import fingerprint


def load(path):
    return json.loads(path.read_bytes())


def verify(run, problem, spec):
    from .rpe_local0_search import digest

    contract = spec['continuation']
    previous = Path(contract['parent_directory']).resolve()
    if previous == run.resolve() or previous.name != contract['parent_run_id']:
        raise ValueError('Continuation parent identity differs')
    selected = load(previous/'APPLIED_SCOPE.json')
    old = load(previous/'ROUND.requested.json')
    if (selected['code_source'] != contract['parent_code_source']
            or selected['signature'] != contract['parent_signature']
            or selected['round_requested_sha256'] != digest(old)
            or selected['problem_sha256'] != digest(problem)
            or load(previous/'PROBLEM.json') != problem):
        raise ValueError('Continuation parent SOURCE, problem or scope binding differs')
    for key in ('base_commit', 'methods', 'initial', 'method_overrides', 'search_overrides',
                'kappa_families', 'max_steps_per_method'):
        if spec.get(key) != old.get(key):
            raise ValueError('Continuation changes the adopted scope: '+key)
    if previous.as_posix() not in spec['executor'].get('prior_run_dirs', []):
        raise ValueError('Continuation must retain the parent in the duplicate audit')
    if (previous/'STOP_REQUESTED.json').exists():
        raise ValueError('Continuation preserves the previous explicit STOP')
    launch = load(previous/'LAUNCH.json')
    stopped = load(previous/'EXIT.json')
    end = load(previous/'KAPPA_STATE.json').get('stop') or {}
    if (stopped.get('invocation_id') != launch.get('invocation_id')
            or stopped.get('returncode') in (None, 0)
            or end.get('reason') != 'RUNTIME_EXCEPTION'
            or end.get('error_type') != 'TimeoutExpired'
            or str(previous/'distributed/LOCAL_WORKER.json') not in end.get('message', '')):
        raise ValueError('Continuation requires a reconciled polling-worker shutdown failure')
    reasons = {row['stop_reason'] for row in load(previous/'ROUND_RESULT.json')['methods'].values()}
    if reasons & {'UNKNOWN_RESULT', 'STALE_JOB_NEEDS_REVIEW', 'NO_ROUTE_ADAPTER', 'HOST_ELIGIBILITY_HOLD'}:
        raise ValueError('Continuation preserves the scientific STOP')
    for marker in ('LAUNCH.json', 'NORMAL_DRIVER.json', 'distributed/LOCAL_DAEMON.json',
                   'distributed/relay/LAUNCH.json'):
        path = previous/marker
        if path.exists():
            pid = load(path).get('pid')
            try:
                cmd = (Path('/proc')/str(pid)/'cmdline').read_bytes()
            except FileNotFoundError:
                continue
            if str(previous).encode() in cmd:
                raise ValueError('Continuation parent still has a live owner: '+marker)
    pbs = previous/'distributed/pbs/STATE.json'
    if pbs.exists() and load(pbs).get('allocations'):
        raise ValueError('Continuation requires reconciliation of the parent PBS allocation')
    ledger = previous/'distributed/QUEUE.sqlite3'
    with sqlite3.connect(ledger.as_uri()+'?mode=ro', uri=True) as sql:
        rows = sql.execute('SELECT job_id, method, state, host_id, attempt_id, request_sha256 FROM jobs ORDER BY job_id').fetchall()
    if any(row[2] not in ('ACCEPTED', 'CANCELLED') for row in rows):
        raise ValueError('Continuation parent has unresolved or unknown attempts')
    if fingerprint(rows) != contract['parent_queue_sha256']:
        raise ValueError('Continuation parent queue changed')
    counts = {}
    for _, method, status, *_ in rows:
        if status == 'ACCEPTED':
            counts[method] = counts.get(method, 0) + 1
    if counts != contract['consumed']:
        raise ValueError('Continuation consumed budget differs from the parent queue')
    if any(row[2] == 'ACCEPTED' and (not row[3] or not row[4]) for row in rows):
        raise ValueError('Continuation parent accepted job is missing its owner/attempt')
    old_budgets = old['executor']['distributed']['preparation']['budgets']
    new_budgets = spec['executor']['distributed']['preparation']['budgets']
    remaining = {k: n - counts.get('PREPARATION:'+k, 0) for k, n in old_budgets.items()}
    remaining_q3 = {k: n - counts.get(k, 0) for k, n in old['max_new_q3_per_method'].items()}
    if (remaining != new_budgets or remaining_q3 != spec['max_new_q3_per_method']
            or min(*remaining.values(), *remaining_q3.values()) < 0):
        raise ValueError('Continuation increases or resets the parent budget')
    return dict(parent=previous.name, consumed=counts, remaining=remaining, remaining_q3=remaining_q3,
                parent_queue_sha256=contract['parent_queue_sha256'])
