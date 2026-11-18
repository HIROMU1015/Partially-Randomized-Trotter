"""Verify a technical-stop handoff without mutating its original ledger."""
import json
from pathlib import Path
import sqlite3

from .rpe_release_registry import fingerprint


def load(path):
    return json.loads(path.read_bytes())


def verify_prior_q3_holds(root, run, problem, spec):
    """Keep an unresolved prior evaluation out of both reuse and new dispatch."""
    from .rpe_q3_coordinator import digest

    options = spec.get('executor', {})
    holds = {}
    for fixed in options.get('prior_q3_holds', []):
        parent = Path(fixed['parent_directory']).resolve()
        prior = {(root / path).resolve() for path in options.get('prior_run_dirs', [])}
        job_id, method = fixed['job_id'], fixed['method']
        if (parent == run.resolve() or parent not in prior or method not in spec['methods']
                or len(job_id) != 20 or any(c not in '0123456789abcdef' for c in job_id)):
            raise ValueError('Prior Q3 hold must refer to a job in the duplicate audit')
        if fixed.get('kind') == 'LOCAL_INTERRUPTED_WRITE':
            job = parent/'jobs'/method/job_id
            old_problem = load(parent.parent/'PROBLEM.json')
            request = load(job/'REQUEST.json')
            outcome = job/'OUTCOME.json'
            returned = job/'PROCESS_RETURN.json'
            owner = load(parent/'CONTROLLER_PID.json')
            if (any(problem[key] != old_problem[key] for key in ('problem','rpe','target','accuracy'))
                    or (request['request_id'],request['method']) != (job_id,method)
                    or digest(request) != fixed['request_sha256']
                    or not (job/'STARTING.json').is_file() or not (job/'LAUNCH.json').is_file()
                    or (outcome.exists() and outcome.stat().st_size)
                    or not returned.is_file() or returned.stat().st_size):
                raise ValueError('Prior local Q3 write hold changed: '+job_id)
            for marker, directory in ((owner,parent),(load(job/'LAUNCH.json'),job)):
                try:
                    command = (Path('/proc')/str(marker['pid'])/'cmdline').read_bytes()
                except FileNotFoundError:
                    continue
                if str(directory).encode() in command:
                    raise ValueError('Prior local Q3 still has a live owner: '+job_id)
            holds[str(job)] = dict(fixed, parent_directory=str(parent),
                                  configuration_id=request['configuration_id'], N_X=request['N_X'],
                                  N_Y=request['N_Y'], reason='INTERRUPTED_RESULT_PERSISTENCE_NO_RETRY')
            continue
        old_problem = load(parent/'PROBLEM.json')
        launch, end = load(parent/'LAUNCH.json'), load(parent/'EXIT.json')
        if (any(problem[key] != old_problem[key] for key in ('problem', 'rpe', 'target', 'accuracy'))
                or launch['invocation_id'] != end['invocation_id']
                or load(parent/'distributed/relay/STATUS.json').get('stopped') is not True):
            raise ValueError('Prior Q3 hold requires the same scientific problem and a stopped parent')
        job = parent/'jobs'/method/job_id
        request = load(job/'REQUEST.json')
        owner = load(job/'LAUNCH.json')
        queue = parent/'distributed/QUEUE.sqlite3'
        with sqlite3.connect(queue.as_uri()+'?mode=ro', uri=True) as db:
            row = db.execute('''SELECT state, method, host_id, attempt_id, request_sha256, request,
                detail, completed_epoch, adopted_epoch, completion_sha256 FROM jobs WHERE job_id=?''',
                (job_id,)).fetchone()
        if (not row or row[:5] != ('UNKNOWN', method, fixed['host_id'], fixed['attempt_id'], fixed['request_sha256'])
                or json.loads(row[5]) != request or digest(request) != fixed['request_sha256']
                or owner.get('host_id') != row[2] or owner.get('attempt_id') != row[3]
                or (request['request_id'], request['method']) != (job_id, method)
                or json.loads(row[6]).get('automatic_retry') is not False
                or any(value is not None for value in row[7:]) or (job/'OUTCOME.json').exists()):
            raise ValueError('Prior Q3 hold no longer matches the unresolved attempt: '+job_id)
        holds[str(job)] = dict(fixed, parent_directory=str(parent), method=method,
                               configuration_id=request['configuration_id'], N_X=request['N_X'], N_Y=request['N_Y'],
                               reason=json.loads(row[6])['reason'])
    return holds


def verify_budget_parents(run,problem,spec):
    """A new run can lift its Q3 ceiling while the old terminal runs stay fixed."""
    from .rpe_local0_search import digest

    prior={Path(path).resolve() for path in spec['executor']['prior_run_dirs']}
    report={}
    for fixed in spec['q3_budget_parents']:
        parent=Path(fixed['path']).resolve()
        method=fixed['method']
        if parent==run.resolve() or parent not in prior or spec['max_new_q3_per_method'].get(method)!='UNLIMITED':
            raise ValueError('Unbounded continuation must retain its capped parent in the duplicate audit')
        applied=load(parent/'APPLIED_SCOPE.json')
        old=load(parent/'ROUND.requested.json')
        previous_problem=load(parent/'PROBLEM.json')
        if (applied['signature']!=fixed['signature'] or applied['code_source']!=fixed['controller_source'] or
                applied['round_requested_sha256']!=digest(old) or
                applied['problem_sha256']!=digest(previous_problem) or
                any(problem[k]!=previous_problem[k] for k in ('problem','rpe','target','accuracy'))):
            raise ValueError('Unbounded continuation differs from the fixed parent inputs')
        final=load(parent/'EXIT.json')
        result=load(parent/'ROUND_RESULT.json')['methods'][method]
        if (final['invocation_id']!=load(parent/'LAUNCH.json')['invocation_id'] or
                result['stop_reason']!='Q3_BUDGET_EXHAUSTED'):
            raise ValueError('Parent did not settle at its Q3 ceiling')
        for name in ('LAUNCH.json','NORMAL_DRIVER.json','distributed/relay/STATUS.json'):
            path=parent/name
            if path.exists():
                pid=load(path).get('pid')
                try: command=(Path('/proc')/str(pid)/'cmdline').read_bytes()
                except FileNotFoundError:continue
                if str(parent).encode() in command or str(fixed['path']).encode() in command:
                    raise ValueError('Unbounded continuation still has a live parent owner')
        pbs=parent/'distributed/pbs/STATE.json'
        if pbs.exists() and load(pbs).get('allocations'):
            raise ValueError('Unbounded continuation has an unresolved parent PBS owner')
        queue=parent/'distributed/QUEUE.sqlite3'
        with sqlite3.connect(queue.as_uri()+'?mode=ro',uri=True) as db:
            rows=db.execute('SELECT job_id,method,state,host_id,attempt_id,request_sha256 FROM jobs ORDER BY job_id').fetchall()
        if fingerprint(rows)!=fixed['queue_sha256'] or any(row[2] not in ('ACCEPTED','CANCELLED') for row in rows):
            raise ValueError('Unbounded continuation parent has changed or unresolved attempts')
        accepted=sum(row[1]==method and row[2]=='ACCEPTED' and bool(row[3] and row[4]) for row in rows)
        cap=old['max_new_q3_per_method'][method]
        if type(cap) is not int or accepted!=cap:
            raise ValueError('Parent adopted Q3 count does not match its original ceiling')
        report[method]=dict(parent=parent.name,old_q3_ceiling=cap,accepted=accepted,new_q3_ceiling='UNLIMITED',
                           parent_queue_sha256=fixed['queue_sha256'])
    return report


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
