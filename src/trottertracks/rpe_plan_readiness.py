"""Prepare plan-only Q3 requests and run or reuse explicit host dry checks.

No worker, controller, queue, execute or collect API is used here. The fixed
33928bf delivery is read through Git and is never changed or republished.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from . import rpe_method_optimizer as O
from . import rpe_q3_coordinator as Q

SOURCE = '28d976b99cdf8329adcc1b79327e329e947ef774'
BASELINE = '33928bf80cca18ce3f1fa3ddde88d27d7e2b7e5b'
BASE = 'docs/research/rpe_distributed_h4_plan_20261117'
METHODS = ('FULL_PF', 'TRUNCATED_PF', 'PR_PAULI', 'PR_DF')


def _git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root)


def _blob(root, commit, name):
    return _git(root, 'show', commit + ':' + name)


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + '\n')


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def baseline_delivery(root, commit=BASELINE, prefix=BASE):
    """Return immutable blob identities without opening old output directories."""
    raw = _git(root, 'ls-tree', '-r', '-z', commit, '--', prefix)
    files = []
    for line in raw.split(b'\0'):
        if not line:
            continue
        meta, name = line.split(b'\t', 1)
        mode, kind, oid = meta.decode().split()
        name = name.decode()
        if (name.startswith(prefix + '/inputs/') or name.startswith(prefix + '/requests/')
                or name.startswith(prefix + '/assignments/') or name.startswith(prefix + '/profiles/')
                or name.startswith(prefix + '/dry-check/') or name.rsplit('/', 1)[-1] in
                ('LOCAL0_DRY_CHECK.md', 'LOCAL0_DRY_CHECK.zip')):
            files.append(dict(commit=commit, path=name, git_blob=oid))
    return dict(commit=commit, files=files, access='READ_ONLY_GIT_BLOBS_NO_OLD_PATH_WRITES')


def build_readiness(root, context, problem, round_, result, output, *,
                    baseline_commit=BASELINE, baseline_prefix=BASE):
    """Build requests from a normal entry result, then write early READY.json.

    ``context['routes']`` may contain the already prepared effective routes;
    otherwise problem routes plus round routes_override are used. Exact baseline
    matches retain the original REQUEST/assignment bytes and remote ZIP bytes.
    New candidates get a separate versioned delivery, never the old delivery.
    """
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    if _git(root, 'rev-parse', 'HEAD').decode().strip() != SOURCE:
        raise ValueError('Readiness requests require the fixed computation SOURCE28d')
    routes = deepcopy(context.get('routes') or problem['execution_routes'])
    if 'routes' not in context:
        routes.update(deepcopy(round_.get('routes_override', {})))
    executor = O.Executor(root, output / 'unsubmitted', routes, context['expected'], problem,
                          optimizer_source=SOURCE, launch_disabled=True)
    manifest = Q.source_manifest(root, SOURCE)
    input_cache, assignments, rows, unchanged = {}, {}, [], True
    for method, summary in result['methods'].items():
        pending = summary.get('pending_request')
        if pending is None:
            rows.append(dict(method=method, status='NO_PENDING_REQUEST'))
            continue
        proposal = deepcopy(pending)
        proposal['cost'] = O.frac(proposal['cost'])
        method_executor = executor.for_method(method)
        rule = method_executor.route(proposal['configuration_id'])
        if rule is None:
            unchanged = False
            rows.append(dict(method=method, status='NO_ROUTE_ADAPTER', proposal=pending))
            continue
        job = method_executor.job_dir(proposal)
        request = method_executor.request(proposal, job, rule)
        request['execution_profile'] = dict(dp_workers=3, prefix_checkpoint_rounds=[5, 11])
        assignment = dict(job_id=job.name, attempt_id='READINESS_ONLY_NO_EXECUTION_' + job.name,
                          method=method, request=request, request_sha256=Q.digest(request),
                          source_manifest=manifest, input_hashes=Q.input_hashes(root, request, input_cache),
                          prefix_key=Q.prefix_key(request), scope='READINESS_ONLY_NEVER_SUBMIT')
        old_request_bytes = _blob(root, baseline_commit, baseline_prefix + '/requests/' + method + '.json')
        old_assignment_bytes = _blob(root, baseline_commit, baseline_prefix + '/assignments/' + method + '.json')
        same = request == json.loads(old_request_bytes) and assignment == json.loads(old_assignment_bytes)
        unchanged = unchanged and same
        request_path = output / 'requests' / (method + '.json')
        assignment_path = output / 'assignments' / (method + '.json')
        if same:
            request_path.parent.mkdir(parents=True, exist_ok=True)
            assignment_path.parent.mkdir(parents=True, exist_ok=True)
            request_path.write_bytes(old_request_bytes)
            assignment_path.write_bytes(old_assignment_bytes)
        else:
            _write(request_path, request)
            _write(assignment_path, assignment)
        assignments[method] = str(assignment_path)
        rows.append(dict(method=method, status='UNEVALUATED', job_id=job.name,
                         configuration_id=request['configuration_id'], N_X=request['N_X'], N_Y=request['N_Y'],
                         cost=request['cost'], origin=request['origin'], classification=pending.get('cls'),
                         request=str(request_path), assignment=str(assignment_path),
                         request_sha256=assignment['request_sha256'], baseline_byte_identical=same,
                         dimensions_before=dict(source='FIXED28D_CHECKED', input='CHECKED',
                                                route='NOT_DRY_CHECKED', runtime='NOT_DRY_CHECKED'),
                         dimensions_after=dict(source='FIXED28D_CHECKED', input='CHECKED',
                                               route='NOT_DRY_CHECKED', runtime='NOT_DRY_CHECKED')))
    unchanged = unchanged and set(assignments) == set(METHODS)
    delivery = baseline_delivery(root, baseline_commit, baseline_prefix)
    for name in ('server2.json', 'server1.json', 'local.json'):
        path = output / 'profiles' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_blob(root, baseline_commit, baseline_prefix + '/profiles/' + name))
    # Fixed remote commands, deployment paths and ZIP are preserved byte for byte.
    for name in ('LOCAL0_DRY_CHECK.md', 'LOCAL0_DRY_CHECK.zip'):
        (output / name).write_bytes(_blob(root, baseline_commit, baseline_prefix + '/' + name))
    delivery['remote_bundle_status'] = 'EXACT_BASELINE_COPY' if unchanged else 'BASELINE_COPY_NEW_CANDIDATES_REQUIRE_VERSIONED_BUNDLE'
    if not unchanged:
        import zipfile
        bundle = output / 'LOCAL0_DRY_CHECK_NEW_CANDIDATES.zip'
        with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as archive:
            for name in ('server1.json', 'local.json'):
                archive.write(output / 'profiles' / name, 'profiles/' + name)
            for method, path in assignments.items():
                archive.write(path, 'assignments/' + method + '.json')
            archive.writestr('scripts/check_rpe_q3_host_readiness.py',
                             _blob(root, baseline_commit, 'scripts/check_rpe_q3_host_readiness.py'))
        delivery['new_bundle'] = str(bundle)
        delivery['new_bundle_notice'] = 'NEW CANDIDATES: use a new owned remote directory; baseline commands and paths are historical and unchanged'
    _write(output / 'BASELINE_DELIVERY.json', delivery)
    queue_states = current_queue_states(round_.get('executor', {}).get('prior_run_dirs', ()))
    queue_unconfirmed = any(s['status'] != 'READ_ONLY_QUEUE_STATE' for s in queue_states)
    for row in rows:
        owners = [s for s in queue_states if s.get('job_id') == row.get('job_id') and s.get('state') != 'CANCELLED']
        row.update(existing_queue_owners=owners,
                   queue_dispatchable=not queue_unconfirmed and not owners,
                   dispatchable=False,
                   dispatchability=('HELD_QUEUE_STATE_UNCONFIRMED' if queue_unconfirmed else
                                    'HELD_EXISTING_OWNER' if owners else 'DRY_CHECK_PENDING'))
    ready = dict(schema='RPE_PLAN_READINESS_V1', status='PREPARED_NOT_DRY_CHECKED',
                 computation_source=SOURCE, candidates=rows, assignments=assignments,
                 server2_profile=str(output / 'profiles/server2.json'),
                 old_remote_delivery_unchanged=unchanged, baseline=str(output / 'BASELINE_DELIVERY.json'),
                 current_operational_queue_states=queue_states,
                 queue_state_confirmed=not queue_unconfirmed,
                 remote_dispatchability=('BLOCKED_QUEUE_STATE_UNCONFIRMED' if queue_unconfirmed else
                                         'AWAITING_REMOTE_HOST_DRY_CHECK'),
                 wall_seconds=time.monotonic() - started, scientific_calls=0, queue_operations=0,
                 output=str(output))
    _write(output / 'CANDIDATES.json', rows)
    _write(output / 'READY.json', ready)
    return ready


def _view_fingerprint(path):
    path = Path(path).resolve()
    head = _git(path, 'rev-parse', 'HEAD').decode().strip()
    dirty = _git(path, 'diff', 'HEAD', '--name-only').decode().splitlines()
    changed = {name: _sha((path / name).read_bytes()) if (path / name).is_file() else 'MISSING'
               for name in dirty}
    return dict(path=str(path), head=head, tracked_changes=changed)


def dry_check_fingerprint(root, assignment, profile):
    """Refresh source/input/route/runtime evidence before considering a cache."""
    root = Path(root)
    route = assignment['request']['route']
    view = profile['route_views'].get(route['view_commit'], '')
    runtime_code = ('import sys,json,platform,importlib.metadata as m; '
                    'd={}; '
                    '\nfor n in ("numpy","scipy","numba","python-flint"):'
                    '\n try: d[n]=m.version(n)'
                    '\n except m.PackageNotFoundError: d[n]=None'
                    '\nprint(json.dumps(dict(executable=sys.executable,version=sys.version,platform=platform.platform(),packages=d),sort_keys=True))')
    runtime = subprocess.check_output([profile['interpreter'], '-B', '-c', runtime_code])
    executable = Path(profile['interpreter']).resolve()
    stat = executable.stat()
    views = [_view_fingerprint(view)]
    # The pinned PR-DF caller loads this nested absolute view before runtime_check.
    if route['kind'].startswith('prdf_d2'):
        nested = Path('/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/d2-pr-df-ld-split-q3-source-20261115')
        views.append(_view_fingerprint(nested))
    return dict(source=Q.source_manifest(root, SOURCE), source_head=_git(root, 'rev-parse', 'HEAD').decode().strip(),
                input_hashes=Q.input_hashes(root, assignment['request']),
                request_sha256=Q.digest(assignment['request']), profile=profile, views=views,
                runtime=json.loads(runtime), interpreter_stat=dict(path=str(executable), size=stat.st_size,
                mtime_ns=stat.st_mtime_ns, inode=stat.st_ino))


def run_server2_dry_checks(root, assignments, profile, output, *, cache=None, checker=None):
    """Run all independent adapter dry checks concurrently, or reuse fresh cache.

    Cached failures remain failures. A cached result is labelled CACHED_RESULT_REUSED
    and keeps the original result locator/status; it is never called a fresh PASS.
    """
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    profile_path = Path(profile).resolve()
    profile_data = _read(profile_path)
    checker = Path(checker or Path(__file__).resolve().parents[2] / 'scripts/check_rpe_q3_host_readiness.py')
    if isinstance(assignments, dict) and 'assignments' in assignments:
        assignments = assignments['assignments']
    old = {}
    if cache:
        cache_path = Path(cache)
        if cache_path.is_dir():
            cache_path /= 'DRY_CACHE.json'
        if cache_path.is_file():
            old = _read(cache_path).get('methods', {})
    started = time.monotonic()

    def check(item):
        method, assignment_path = item
        assignment = _read(assignment_path)
        target = output / method
        error = None
        try:
            fingerprint = dry_check_fingerprint(root, assignment, profile_data)
        except Exception as exc:
            # Incomplete evidence disables cache, then real dry-check records the error.
            fingerprint = None
            error = dict(type=type(exc).__name__, reason=str(exc))
        prior = old.get(method) or {}
        previous_result = Path(prior.get('result', ''))
        previous_intact = previous_result.is_file() and _sha(previous_result.read_bytes()) == prior.get('result_sha256')
        if fingerprint is not None and fingerprint == prior.get('fingerprint') and previous_intact:
            original = _read(prior['result'])
            _write(target / 'READINESS.json', dict(status='CACHED_RESULT_REUSED',
                original_result=original.get('original_result', prior['result']), original_status=original.get('original_status', original['status']),
                fresh_dimensions_checked=['SOURCE', 'route', 'input', 'runtime'],
                fingerprint=Q.digest(fingerprint), scientific_evaluation_calls=0, queue_operations=0))
            mode = 'CACHED'
        else:
            target.mkdir()
            # The existing checker insists on a new directory, so remove the empty placeholder.
            target.rmdir()
            with (output / (method + '.log')).open('w') as log:
                subprocess.run([sys.executable, '-B', str(checker), '--assignment', str(assignment_path),
                    '--profile', str(profile_path), '--output', str(target)], cwd=root,
                    stdout=log, stderr=subprocess.STDOUT)
            mode = 'LIVE_DRY_CHECK'
        observed = _read(target / 'READINESS.json')
        return method, dict(mode=mode, status=observed['status'], original_status=observed.get('original_status'),
                            result=str(target / 'READINESS.json'), result_sha256=_sha((target / 'READINESS.json').read_bytes()),
                            fingerprint=fingerprint,
                            freshness_error=error,
                            dimensions_before=dict(source='CHECKED' if fingerprint else 'INCOMPLETE',
                                route='CHECKED' if fingerprint else 'INCOMPLETE', input='CHECKED' if fingerprint else 'INCOMPLETE',
                                runtime='CHECKED' if fingerprint else 'INCOMPLETE'),
                            dimensions_after=dict(source='CHECKED' if fingerprint else 'SEE_DRY_CHECK',
                                route=observed.get('original_status', observed['status']),
                                input=observed.get('original_status', observed['status']),
                                runtime=observed.get('original_status', observed['status'])))

    with ThreadPoolExecutor(max_workers=max(1, len(assignments))) as pool:
        checked = dict(pool.map(check, assignments.items()))
    summary = dict(schema='RPE_PLAN_DRY_CACHE_V1', methods=checked,
                   live_checks=sum(v['mode'] == 'LIVE_DRY_CHECK' for v in checked.values()),
                   cached_checks=sum(v['mode'] == 'CACHED' for v in checked.values()),
                   wall_seconds=time.monotonic() - started, scientific_calls=0, queue_operations=0)
    _write(output / 'DRY_CACHE.json', summary)
    return summary


def finish_readiness(prepared, dry_checks):
    """Publish completion separately from the early prepared-only READY event."""
    ready = deepcopy(prepared)
    ready.update(status='DRY_CHECKS_COMPLETED', server2=dry_checks)
    for row in ready['candidates']:
        check = dry_checks['methods'].get(row['method'])
        if check:
            row['dimensions_before'] = check['dimensions_before']
            row['dimensions_after'] = check['dimensions_after']
            passed = (check.get('original_status') or check['status']) == 'REAL_HOST_DRY_CHECK_PASS'
            row['dispatchable'] = row.get('queue_dispatchable', True) and passed
            if row.get('queue_dispatchable', True):
                row['dispatchability'] = ('DRY_CHECK_PASS_NO_EXISTING_QUEUE_OWNER_OBSERVED' if passed else
                                         'HELD_DRY_CHECK_NOT_PASS')
    _write(Path(ready['output']) / 'CANDIDATES.json', ready['candidates'])
    _write(Path(ready['output']) / 'READY.json', ready)
    return ready


def current_queue_states(prior_run_dirs):
    """Read existing controller queues through SQLite read-only URI connections.

    Do not instantiate JobStore: its constructor sets WAL and creates tables.
    Unknown schema/state or an unreadable database becomes an explicit hold.
    No database is created when a prior run has no distributed queue.
    """
    import sqlite3
    required = {'job_id', 'method', 'host_id', 'attempt_id', 'state'}
    known = {'QUEUED', 'ASSIGNED', 'STARTING', 'RUNNING', 'UNKNOWN',
             'COMPLETED_PENDING_TRANSFER', 'ACCEPTED', 'CANCELLED'}
    states, seen = [], set()
    for prior in prior_run_dirs:
        path = (Path(prior) / 'distributed/QUEUE.sqlite3').resolve()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        database = None
        try:
            database = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
            database.execute('PRAGMA query_only=ON')
            columns = {row[1] for row in database.execute('PRAGMA table_info(jobs)')}
            if not required.issubset(columns):
                raise ValueError('Unknown jobs schema; missing columns: ' + ','.join(sorted(required - columns)))
            rows = database.execute('SELECT job_id,method,host_id,attempt_id,state FROM jobs ORDER BY job_id').fetchall()
            observed = time.time()
            for row in rows:
                item = dict(zip(('job_id', 'method', 'host_id', 'attempt_id', 'state'), row))
                item.update(database=str(path), observed_epoch=observed,
                            status='READ_ONLY_QUEUE_STATE' if item['state'] in known else 'QUEUE_STATE_UNCONFIRMED',
                            mode='sqlite_uri_mode_ro_query_only')
                if item['state'] not in known:
                    item['error'] = 'Unknown queue state: ' + str(item['state'])
                states.append(item)
        except Exception as exc:
            states.append(dict(database=str(path), status='QUEUE_SCHEMA_OR_READ_UNCONFIRMED',
                               error_type=type(exc).__name__, error=str(exc),
                               dispatchable=False, observed_epoch=time.time()))
        finally:
            if database is not None:
                database.close()
    return states
