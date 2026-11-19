"""Verify a technical-stop handoff without mutating its original ledger."""
import hashlib
import json
from pathlib import Path
import sqlite3

from .rpe_release_registry import fingerprint


def load(path):
    return json.loads(path.read_bytes())


def verify_retired_preparation_attempts(run):
    """Bind inherited holds to the same request and ended parent allocation."""
    from .rpe_q3_coordinator import digest

    lineage = run/'RUN_LINEAGE.json'
    if not lineage.is_file():
        return {}
    handoff = load(lineage)
    carried = handoff.get('retired_preparation_attempts', {})
    if not carried:
        return {}
    def entry(directory, job):
        with sqlite3.connect((directory/'distributed/QUEUE.sqlite3').resolve().as_uri()+'?mode=ro', uri=True) as db:
            return db.execute('SELECT method, host_id, attempt_id, request_sha256, request, state, detail '
                              'FROM jobs WHERE job_id=?', (job,)).fetchone()

    verified = {}
    for job, hold in carried.items():
        row = entry(run, job)
        receipt = hold['allocation_receipt']
        if (not row or not row[0].startswith('PREPARATION:')
                or row[1:4] != ('pbs-0', hold['attempt_id'], hold['request_sha256'])
                or row[5] != 'UNKNOWN' or json.loads(row[6]).get('automatic_retry') is not False
                or digest(json.loads(row[4])) != row[3] or receipt['job_state'] not in ('F', 'X')):
            raise ValueError('Retired preparation ownership differs: '+job)
        parent = Path(handoff['parent_directory']).resolve()
        visited = {run.resolve()}
        while True:
            if (parent in visited or load(parent/'LAUNCH.json')['invocation_id'] != load(parent/'EXIT.json')['invocation_id']
                    or load(parent/'distributed/relay/STATUS.json').get('stopped') is not True):
                raise ValueError('Retired preparation attempts require a stopped parent')
            if entry(parent, job) != row:
                raise ValueError('Retired preparation ownership differs: '+job)
            verified[(parent/'jobs'/row[0]/job).resolve()] = (job, row[2])
            allocations = load(parent/'distributed/pbs/STATE.json')['allocations'].values()
            if any(a['spec']['profile']['host_id'] == 'pbs-0' and
                   a.get('observed', {}).get('job_id') == receipt['pbs_job_id'] for a in allocations):
                break
            # A successor can carry the same retired attempt across another
            # handoff. Its terminal allocation remains in the original run.
            lineage = parent/'RUN_LINEAGE.json'
            previous = load(lineage) if lineage.is_file() else {}
            if previous.get('retired_preparation_attempts', {}).get(job) != hold:
                raise ValueError('Retired preparation ownership differs: '+job)
            visited.add(parent)
            parent = Path(previous['parent_directory']).resolve()
    return verified


def preparation_rejected_before_science(job):
    """Verify a local preparation stopped before a scientific command ran."""
    from .rpe_q3_coordinator import digest
    from .rpe_release_deployment import manifest_reference

    if not job.parent.name.startswith('PREPARATION:'):
        return False
    parent = job.parents[2]
    queue = parent/'distributed/QUEUE.sqlite3'
    if not queue.is_file():
        return False
    with sqlite3.connect(queue.resolve().as_uri()+'?mode=ro', uri=True) as db:
        row = db.execute('''SELECT state, method, host_id, attempt_id, request_sha256, detail,
            started_epoch, completed_epoch, adopted_epoch, completion_sha256, request
            FROM jobs WHERE job_id=?''', (job.name,)).fetchone()
    if not row or row[0:2] != ('UNKNOWN', job.parent.name) or any(x is not None for x in row[7:10]):
        return False
    detail = json.loads(row[5])
    if detail.get('automatic_retry') is not False or detail.get('resource_released') is not True:
        return False
    if not all((parent/path).is_file() for path in ('EXIT.json', 'STOP_REQUESTED.json',
                                                   'distributed/relay/STATUS.json')):
        return False
    if (load(parent/'LAUNCH.json')['invocation_id'] != load(parent/'EXIT.json')['invocation_id']
            or load(parent/'distributed/relay/STATUS.json').get('stopped') is not True):
        return False
    request = load(job/'REQUEST.json')
    owner = load(job/'LAUNCH.json')
    dist = load(parent/'ROUND.requested.json')['executor']['distributed']
    worker = dist['local_worker']
    work_dir = Path(worker['work_dir'])
    if ((row[2], row[3], row[4]) != (worker['host_id'], owner['attempt_id'], digest(request))
            or json.loads(row[10]) != request
            or owner['host_id'] != row[2] or not work_dir.is_dir()
            or (parent/'distributed/transfers'/job.name).exists()
            or set(p.name for p in job.iterdir()) != {'REQUEST.json', 'LAUNCH.json'}):
        return False
    stage = request['preparation']['stage']
    source = request['preparation']['source']['commit']
    manifest = load(Path(dist['release_manifest']))
    if (row[1] != 'PREPARATION:'+stage or not dist.get('local_deployment_layout')
            or dist.get('required_release') != manifest_reference(manifest)):
        return False
    if detail['reason'] == 'PREPARATION_DISPATCH_VALIDATION_REJECTED_BEFORE_DELIVERY':
        return (row[6] is None and not (work_dir/job.name).exists()
                and not any(manifest['components'][name]['source'] == source
                            for name in manifest['stage_components'].get(stage, [])))
    spec = request['preparation']
    if spec.get('provider') != 'PAULI_COMPILE' or spec.get('publish_pauli_binding') is not True:
        return False
    # Older workers recorded this failure in stderr before they wrote a
    # preparation outcome. Bind that receipt to the original worker attempt.
    attempt = work_dir/job.name/row[3]
    if (row[6] is None or (attempt/'PREPARATION_BODY.log').exists()
            or not all((attempt/path).is_file() for path in ('ASSIGNMENT.json', 'PROCESS_RETURN.json',
                                                           'WORKER_STATE.json', 'WORKER_EXECUTE.log'))):
        return False
    assignment = load(attempt/'ASSIGNMENT.json')
    receipt = load(attempt/'PROCESS_RETURN.json')
    worker_state = load(attempt/'WORKER_STATE.json')
    return (assignment['request'] == request and assignment['attempt_id'] == row[3]
            and assignment['request_sha256'] == row[4] and receipt.get('returncode') == 1
            and receipt.get('signal') is None and receipt.get('automatic_retry') is False
            and worker_state.get('state') == 'UNKNOWN' and worker_state.get('detail') == detail
            and (attempt/'WORKER_EXECUTE.log').read_text().rstrip().endswith(
                'ValueError: Pauli publication base ref differs from the fixed provider'))


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
            if fixed.get('replay_exact'):
                raise ValueError('Prior distributed Q3 replay cannot use a local hold: '+job_id)
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
        terminal = fixed.get('kind') == 'TERMINAL_WORKER_UNKNOWN'
        if (not row or row[:5] != ('ACCEPTED' if terminal else 'UNKNOWN',
                                   method, fixed['host_id'], fixed['attempt_id'], fixed['request_sha256'])
                or json.loads(row[5]) != request or digest(request) != fixed['request_sha256']
                or owner.get('host_id') != row[2] or owner.get('attempt_id') != row[3]
                or (request['request_id'], request['method']) != (job_id, method)):
            raise ValueError('Prior Q3 hold no longer matches the unresolved attempt: '+job_id)
        detail = json.loads(row[6])
        if terminal:
            outcome_bytes = (job/'OUTCOME.json').read_bytes()
            receipt_bytes = (job/'PROCESS_RETURN.json').read_bytes()
            outcome, receipt = json.loads(outcome_bytes), json.loads(receipt_bytes)
            if (row[7] is None or row[8] is not None or row[9] != fixed['completion_sha256']
                    or detail.get('outcome_status') != 'UNKNOWN' or outcome.get('status') != 'UNKNOWN'
                    or outcome.get('request_id') != job_id or outcome.get('scientific_evaluation_calls') != 0
                    or type(receipt.get('returncode')) is not int or receipt['returncode'] == 0
                    or receipt.get('automatic_retry') is not False
                    or hashlib.sha256(outcome_bytes).hexdigest() != fixed['outcome_sha256']
                    or hashlib.sha256(receipt_bytes).hexdigest() != fixed['process_return_sha256']):
                raise ValueError('Prior terminal Q3 hold differs from its zero-call receipt: '+job_id)
            reason = outcome['reason']
        else:
            if (detail.get('automatic_retry') is not False or any(value is not None for value in row[7:])
                    or (job/'OUTCOME.json').exists()):
                raise ValueError('Prior Q3 hold no longer matches the unresolved attempt: '+job_id)
            reason = detail['reason']
        if fixed.get('replay_exact') is True:
            if (terminal or reason != 'LAUNCH_STATE_UNCERTAIN_NO_ADAPTER_EXIT_RECEIPT'
                    or (job/'PROCESS_RETURN.json').exists()
                    or _has_local_owner(launch.get('pid'), parent)):
                raise ValueError('Prior Q3 replay requires a stopped, interrupted attempt: '+job_id)
        holds[str(job)] = dict(fixed, parent_directory=str(parent), method=method,
                               configuration_id=request['configuration_id'], N_X=request['N_X'], N_Y=request['N_Y'],
                               reason=reason)
    return holds


def _has_local_owner(pid, run):
    if type(pid) is not int:
        return False
    try:
        return str(run).encode() in (Path('/proc')/str(pid)/'cmdline').read_bytes()
    except FileNotFoundError:
        return False


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
