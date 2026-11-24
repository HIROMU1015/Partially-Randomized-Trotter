"""Request-specific dry-checks using the already deployed worker and adapter."""
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import threading
import time
import zlib

from . import rpe_q3_coordinator as Q
from .rpe_pbs import task_estimate


class ReadinessConnections:
    """One private OpenSSH master per target for short metadata checks."""

    # Leave room for the relay and gateway, even with the usual SSH limit of
    # ten sessions per connection. This bounds handshakes, not job CPU slots.
    channels_per_host = 8

    def __init__(self):
        self.sockets = tempfile.TemporaryDirectory(prefix='rpe-q3-probe-', dir='/tmp')
        self.locks = {}
        self.lock = threading.Lock()
        self.closed = False

    def options(self, ssh, alias):
        if Path(ssh[0]).name != 'ssh':
            return []
        token = hashlib.sha256(json.dumps([list(ssh), alias]).encode()).hexdigest()[:16]
        path = Path(self.sockets.name)/token
        with self.lock:
            target_lock = self.locks.setdefault(token, threading.Lock())
        with target_lock:
            if self.closed:
                raise OSError('Readiness connection pool closed')
            control = ['-o', 'ControlPath='+str(path)]
            flags = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=6']
            checked = path.is_socket() and subprocess.run([*ssh, *control, '-O', 'check', alias],
                stdin=subprocess.DEVNULL, capture_output=True, timeout=2).returncode == 0
            if not checked:
                subprocess.run([*ssh, *control, *flags, '-o', 'ControlMaster=yes', '-o', 'ControlPersist=15',
                                '-fNT', alias], stdin=subprocess.DEVNULL, capture_output=True, timeout=9, check=True)
            return [*control, '-o', 'ControlMaster=auto', '-o', 'ControlPersist=15']

    def close(self):
        with self.lock:
            self.closed = True
            self.sockets.cleanup()


@lru_cache(maxsize=8)
def relay_type(script):
    if script is None:
        from run_rpe_q3_relay import Relay
        return Relay
    spec = importlib.util.spec_from_file_location('_rpe_selected_q3_relay', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Relay

_REMOTE_PROBE = (
    'import base64,io,json,sys,zlib;v=json.load(sys.stdin);'
    'code=zlib.decompress(base64.b64decode(v.pop("probe_code")));'
    'sys.stdin=io.StringIO(json.dumps(v));exec(code)'
)

# Runs at the host. This only invokes the existing no-science readiness entry.
_HOST_PROBE = r"""
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

payload = json.load(sys.stdin)
root = Path(payload['worker_root'])
sys.path.insert(0, str(root/'src'))
from trottertracks import rpe_host_eligibility
from trottertracks.rpe_host_profiles import profile_core, select_host_profile

assignment = payload['assignment']
with tempfile.TemporaryDirectory(prefix='rpe-q3-live-') as directory:
    base = Path(directory)
    profile = Path(payload['profile_path']) if payload.get('profile_path') else base/'profile.json'
    if not payload.get('profile_path'):
        profile.write_text(json.dumps(payload['profile']))
    configured = json.loads(profile.read_text())
    if (assignment.get('required_release') and payload.get('deployment') and
            'input_objects' not in payload and 'input_pack' not in payload):
        try:
            from trottertracks import rpe_release_deployment
            prepare = getattr(rpe_release_deployment, 'prepare_worker_assignment', None)
            if prepare is not None:
                prepare(configured, assignment, payload['deployment'])
        except ImportError:
            pass  # An older worker can still check assets already placed for its fixed release.
        except (OSError, KeyError, ValueError, subprocess.SubprocessError) as error:
            print(json.dumps(dict(status='HOLD', reason='WORKER_DEPLOYMENT_HOLD: '+str(error))))
            raise SystemExit(0)
    effective = select_host_profile(configured, assignment)
    route = assignment['request']['route']
    repositories = {Path(effective['root']).resolve()}
    view = effective.get('route_views', {}).get(route.get('view_commit'))
    if view:
        repositories.add(Path(view).resolve())
    if route['kind'] in ('prdf_d2_split','prdf_d2_common_entry'):
        inner = effective.get('route_views', {}).get('27e70efe533862f38f4080aabb9cdd3ad1749fe9')
        if inner:
            repositories.add(Path(inner).resolve())

    def git(repo, *args, **kwargs):
        return subprocess.run(['git', *args], cwd=repo, capture_output=True,
                              env=dict(os.environ, GIT_NO_LAZY_FETCH='1', GIT_TERMINAL_PROMPT='0'), **kwargs)

    def unavailable_inputs(repo):
        rows = assignment['input_hashes'] + payload.get('input_dependencies', [])
        if not rows:
            return []
        locators = [os.fsencode(row['commit'] + ':' + row['path']) for row in rows]
        # Do one lookup per checkout for the common case. If Git cannot
        # enumerate a partial object store, preserve the per-locator fallback.
        if all(b'\n' not in locator and b'\x00' not in locator for locator in locators):
            batch = git(repo, 'cat-file', '--batch-check=%(objecttype)',
                        input=b'\n'.join(locators) + b'\n')
            reports = batch.stdout.split(b'\n')[:-1]
            if (not batch.returncode and len(reports) == len(rows)
                    and all(report in (b'blob', b'tree', b'commit', b'tag')
                            or report.endswith(b' missing') for report in reports)):
                return [row for row, report in zip(rows, reports) if report.endswith(b' missing')]
        return [row for row, locator in zip(rows, locators)
                if git(repo, 'cat-file', '-e', locator).returncode]

    # Input commits may be newer than this fixed scientific checkout.
    # Install their missing objects only; the checkout and its refs stay fixed.
    missing = []
    for repo in sorted(repositories):
        locators = unavailable_inputs(repo)
        if not locators:
            continue
        if 'input_pack' in payload:
            pack = base64.b64decode(payload['input_pack'], validate=True)
            if hashlib.sha256(pack).hexdigest() != payload['input_pack_sha256']:
                raise ValueError('Q3 input object pack hash differs')
            before = git(repo, 'rev-parse', 'HEAD', check=True).stdout
            git(repo, 'index-pack', '--stdin', '--keep=RPE fixed Q3 inputs', input=pack, check=True)
            if git(repo, 'rev-parse', 'HEAD', check=True).stdout != before:
                raise ValueError('Q3 checkout SOURCE changed during input installation')
            for row in locators:
                blob = git(repo, 'show', row['commit']+':'+row['path'], check=True).stdout
                if hashlib.sha256(blob).hexdigest() != row['sha256']:
                    raise ValueError('Installed Q3 input hash differs: '+row['path'])
        else:
            anchors = [commit for commit in payload.get('input_anchors', [])
                       if git(repo, 'cat-file', '-e', commit+'^{tree}').returncode == 0]
            objects = payload.get('input_objects', [])
            absent = []
            if objects:
                scanned = git(repo, 'cat-file', '--batch-check', input=('\n'.join(objects)+'\n').encode())
                if scanned.returncode:
                    # Git may abort batch lookup for a missing promisor object.
                    # Per-object existence checks still work in partial clones.
                    absent = [oid for oid in objects if git(repo, 'cat-file', '-e', oid).returncode]
                else:
                    absent = [row.split()[0] for row in scanned.stdout.decode().splitlines() if row.endswith(' missing')]
            missing.append(dict(commits=sorted({row['commit'] for row in locators}), anchors=anchors, objects=absent))
    if missing:
        print(json.dumps(dict(status='NEEDS_Q3_INPUT_OBJECTS', repositories=missing)))
        raise SystemExit(0)

    request = base/'ASSIGNMENT.json'
    request.write_text(json.dumps(assignment))
    output = base/'check'
    checked = subprocess.run([sys.executable, '-B', str(root/'scripts/check_rpe_q3_host_readiness.py'),
                               '--assignment', str(request), '--profile', str(profile), '--output', str(output)],
                              capture_output=True, text=True, timeout=90)
    result = json.loads((output/'READINESS.json').read_text()) if (output/'READINESS.json').exists() else {}
    if checked.returncode or result.get('status') != 'REAL_HOST_DRY_CHECK_PASS':
        dry = result.get('adapter_result') or {}
        print(json.dumps(dict(status='HOLD', reason=dry.get('reason') or result.get('error') or checked.stderr[-2000:])))
    elif result.get('scientific_evaluation_calls') != 0 or result.get('request_sha256') != assignment['request_sha256']:
        raise ValueError('Dry-check changed the request or executed a scientific call')
    else:
        checked_fingerprint = getattr(rpe_host_eligibility, 'fingerprint_after_dry_check', None)
        fingerprint = (checked_fingerprint(assignment, configured, result) if checked_fingerprint
                       else rpe_host_eligibility.current_host_fingerprint(assignment, configured))
        print(json.dumps(dict(status='PASS', request_sha256=assignment['request_sha256'],
                              fingerprint=fingerprint, profile=profile_core(effective),
                              evaluator=(result.get('adapter_result') or {}).get('evaluator'))))
"""


def _supply_inputs(executor, payload, invoke, response):
    """Reuse the host's existing input objects and send the missing delta."""
    repositories = response['repositories']
    anchors = set.intersection(*(set(row['anchors']) for row in repositories))
    if not anchors:
        return dict(status='HOLD', reason='Q3 input delivery requires a placed fixed Git base')
    commits = {commit for row in repositories for commit in row['commits']}
    inputs = payload['assignment']['input_hashes'] + payload.get('input_dependencies', [])
    if not anchors <= set(payload['input_anchors']) or not commits <= {row['commit'] for row in inputs}:
        raise ValueError('Worker Git input inventory is outside the fixed Q3 assignment')
    root = executor.computation_root
    # An exact Git locator needs its commit, path trees and blob. Checking
    # the whole ancestry sends large unrelated histories to sparse workers.
    locators = []
    for row in inputs:
        commit = row['commit']
        if commit in commits:
            path = Path(row['path'])
            locators.extend([commit, commit+'^{tree}',
                             *(commit+':'+p.as_posix() for p in (*path.parents, path) if p != Path('.'))])
    started = time.monotonic()
    objects = set(subprocess.check_output(['git', 'rev-parse', *locators], cwd=root, text=True).split())
    timing = dict(locator_seconds=time.monotonic()-started, offered_objects=len(objects))
    started = time.monotonic()
    try:
        inventory = invoke(dict(payload, input_objects=sorted(objects)))
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        return dict(status='HOLD', reason='Q3 input inventory: '+str(error)[-1500:],
                    input_transfer=dict(timing, inventory_seconds=time.monotonic()-started))
    timing['inventory_seconds'] = time.monotonic()-started
    if inventory['status'] != 'NEEDS_Q3_INPUT_OBJECTS':
        return inventory
    needed = sorted({oid for row in inventory['repositories'] for oid in row['objects']})
    if not needed:
        return dict(status='HOLD', reason='Q3 input locator missing despite available Git objects')
    if not set(needed) <= set(objects):
        raise ValueError('Worker requested objects outside the fixed Q3 input delta')
    started = time.monotonic()
    content = subprocess.check_output(['git', 'pack-objects', '--stdout', '--window=0'], cwd=root,
                                      input=('\n'.join(needed)+'\n').encode())
    sha256 = hashlib.sha256(content).hexdigest()
    timing.update(objects=len(needed), bytes=len(content), sha256=sha256, pack_seconds=time.monotonic()-started)
    started = time.monotonic()
    try:
        result = invoke(dict(payload, input_pack=base64.b64encode(content).decode(), input_pack_sha256=sha256))
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        result = dict(status='HOLD', reason='Q3 input delivery and check: '+str(error)[-1500:])
    return dict(result, input_transfer=dict(timing, delivery_check_seconds=time.monotonic()-started))


def controller_input_hashes(executor, request):
    """Place newer provider receipts in the fixed controller-side Q3 object store."""
    target=executor.computation_root
    try:
        return Q.input_hashes(target,request,executor.input_hash_cache)
    except subprocess.CalledProcessError:
        if target==executor.root:
            raise
    expected=Q.input_hashes(executor.root,request)
    commits=sorted({row['commit']for row in expected})
    before=subprocess.check_output(['git','rev-parse','HEAD'],cwd=target)
    subprocess.run(['git','fetch','--no-tags','--no-write-fetch-head',str(executor.root),*commits],
                   cwd=target,check=True,capture_output=True)
    observed=Q.input_hashes(target,request)
    if observed!=expected or subprocess.check_output(['git','rev-parse','HEAD'],cwd=target)!=before:
        raise ValueError('Controller Q3 input placement changed fixed SOURCE or input bytes')
    executor.input_hash_cache.update({(row['commit'],row['path']):row for row in observed})
    return observed


def q3_input_dependencies(executor, request):
    """Find the raw signal blobs read by a saved PR-DF signal packet."""
    route = request['route']
    if route['kind'] not in ('prdf_d2_split', 'prdf_d2_common_entry'):
        return []
    binding = route.get('configurations', {}).get(request.get('configuration_id'), route)
    locator = binding.get('signal_packet')
    if not locator:
        return []
    key = ('prdf_signal_inputs', Q.digest(locator))
    cache = executor.input_hash_cache
    if key not in cache:
        packet = json.loads(subprocess.check_output(
            ['git', 'show', locator['commit']+':'+locator['path']], cwd=executor.computation_root))
        for selector in locator.get('selector', ()):
            packet = packet[selector]
        means = [row['source'] for row in packet.get('means', ())]
        # These raw rows are transport dependencies of the fixed packet.
        # Preserve the request hash and the existing assignment input identity.
        cache[key] = controller_input_hashes(executor, dict(route=dict(kind=route['kind'], sources=means)))
    return cache[key]


_SCOPE_CHECK = """
import json
from pathlib import Path
import runpy
import sys

payload = json.load(sys.stdin)
adapter = runpy.run_path(payload['entry'])
request, views = payload['request'], payload['views']
route = request['route']
view = Path(views[route['view_commit']])
horizon = adapter['horizon_of'](request)
supported_horizon = adapter.get('horizon_supported')
if not (supported_horizon(request, horizon) if supported_horizon else horizon is None or horizon <= 12):
    print(json.dumps(dict(reason='OUTSIDE_VALIDATED_R3_SCALAR_EQUIVALENCE_DOMAIN', M=horizon)))
else:
    case, inputs = adapter['scoped_case'](view, request)
    implementation = adapter['route'](view, route['kind'], views)
    bounds = implementation.bounds(case['allocation']['shots'])
    supported = adapter.get('finite_support_requested')
    if not adapter['in_scope'](bounds, inputs['options']) and not (supported and supported(request, case)):
        print(json.dumps(dict(reason='COUNT_ONLY_SCOPE_EXCEEDED', bounds=bounds,
                              limits={key: inputs['options'][key] for key in
                                      ('maximum_round_outcomes', 'maximum_verification_pairs')},
                              transition_limit=inputs['options']['total_state_refinement']['maximum_transition_pairs'])))
    else:
        print('null')
"""


def controller_scope_hold(request, profile):
    """Use the fixed route's own count guard without evaluating the case."""
    if not profile or request['route']['kind'] == 'artificial':
        return None
    from .rpe_host_profiles import select_host_profile, validate_adapter_profile

    try:
        profile = select_host_profile(profile, {'request': request})
        adapter = validate_adapter_profile(profile)
        if adapter is None:
            return None
        views = profile['route_views']
        risk = request['route'].get('horizon_risk_source')
        if risk is not None and risk['commit'] not in views:
            return None  # A legacy local profile cannot disqualify a remote extension.
        commit = request['route']['view_commit']
        view = Path(views[commit])
        if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=view, text=True).strip() != commit:
            return None
        if subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', 'scripts', 'src'], cwd=view).returncode:
            return None
        entry = 'scripts/run_rpe_optimizer_q3_job.py'
        result = json.loads(subprocess.check_output(
            [profile['interpreter'], '-B', '-c', _SCOPE_CHECK],
            input=json.dumps(dict(entry=str(Path(adapter['root']) / entry), request=request, views=views)),
            cwd=adapter['root'], text=True, stderr=subprocess.PIPE, timeout=20))
        if result is not None:
            return dict(result, adapter_commit=adapter['commit'], adapter_sha256=adapter['files'][entry],
                        route_commit=commit, scientific_evaluation_calls=0)
    except (OSError, KeyError, ValueError, subprocess.SubprocessError):
        # An unavailable controller view does not establish a route-wide hold.
        return None
    return None


def accepts_probe(executor, host, policy, error):
    admission = executor._host_admission(host, policy, error)
    if not admission['accepting'] or admission.get('max_slots') == 0:
        return False
    manager = getattr(executor, 'pbs_manager', None)
    if manager is not None and host in manager.configured and not any(
            row['host_id'] == host and row['status'] == 'R' for row in manager.allocations):
        # A drained PBS session's last negative offer cannot prevent the
        # request check needed to attach its successor to the reservation.
        return True
    observed = executor.hosts.get(host, {})
    return (observed.get('accepting_new_jobs', True)
            and ((observed.get('resource_report') or {}).get('storage') or {}).get('accepting_new_jobs', True))


def probe(executor, request, job_id, decision, *, publish=None, dispatch=None, targets=None):
    """Obtain real-input dry checks for new jobs before reserving a slot."""
    assignment = dict(job_id=job_id, method=request['method'], request=request,
                      request_sha256=Q.digest(request), source_manifest=executor.source_manifest)
    if executor.distributed.get('required_release'):
        assignment['required_release'] = executor.distributed['required_release']
    local = executor.distributed.get('local_worker') or {}
    policy, error = executor._host_admission_policy()
    hold = controller_scope_hold(request, local)
    horizon_can_differ = (hold and hold['reason']=='OUTSIDE_VALIDATED_R3_SCALAR_EQUIVALENCE_DOMAIN'
                          and request['route'].get('horizon_risk_source'))
    if hold is not None and not horizon_can_differ:
        decision.update(reasons=[hold['reason']], execution_profiles={}, host_decisions={}, controller_scope=hold)
        record = executor.run_dir/'distributed'/'q3-readiness'/job_id/'READINESS.json'
        # Keep host observations already saved by an earlier controller version.
        with executor.state.cv:
            if not executor.state.closed:
                record.parent.mkdir(parents=True, exist_ok=True)
                Q.atomic_json(record.with_name('CONTROLLER_SCOPE.json'), dict(assignment=assignment, scope=hold))
        return dict(decision, decision_sha256=Q.digest(decision))
    assignment['input_hashes'] = controller_input_hashes(executor, request)
    existing = {(row['commit'], row['path']) for row in assignment['input_hashes']}
    dependencies = [row for row in q3_input_dependencies(executor, request)
                    if (row['commit'], row['path']) not in existing]
    participants = []
    pbs = executor.distributed.get('pbs') or {}
    def accepting(host):
        return accepts_probe(executor, host, policy, error)

    pbs_workers = {worker['host_id']: worker for worker in pbs.get('workers', [])
                   if pbs.get('walltime_policy') and accepting(worker['host_id'])}
    if local and accepting(local['host_id']):
        participants.append((local['host_id'], None))
    participants.extend((host, None) for host in pbs_workers)
    relay = executor.run_dir/'distributed'/'relay'/'CONFIG.json'
    bridge = None
    connections = getattr(executor, 'readiness_connections', None)
    if relay.is_file():
        bridge = relay_type(executor.distributed.get('relay_script'))(relay)
        bridge.readiness_connections = connections
        bridge.config['request_timeout_seconds'] = 120
        participants.extend((worker['host_id'], worker) for worker in bridge.config.get('workers', [])
                            if worker.get('enabled', True)
                            and accepting(worker['host_id'])
                            and (not executor._host_admission(worker['host_id'], policy, error)['optional']
                                 or executor.hosts.get(worker['host_id'], {}).get('accepting_new_jobs') is True))
    if targets is not None:
        participants = [item for item in participants if item[0] in targets]

    def check_host(item):
        host, worker = item
        try:
            anchors = [executor.computation_source, (request.get('route') or {}).get('view_commit')]
            provider = executor.distributed.get('live_q3_input_base')
            if provider:
                anchors.append(provider)
            payload = dict(assignment=assignment, input_anchors=[value for value in anchors if value])
            if dependencies:
                payload['input_dependencies'] = dependencies
            if host in pbs_workers:
                if not (task_estimate(request, pbs, host) or 0) > 0:
                    return host, dict(status='HOLD', reason='PBS_WALLTIME_ESTIMATE_UNAVAILABLE')
                # The gateway and compute nodes share the deployed fixed views.
                # The worker repeats the normal fingerprint on the allocated node.
                declared=pbs_workers[host]
                payload.update(profile=declared['profile'], worker_root=pbs['transport_root'],
                    deployment=dict(deployment=declared.get('deployment') or pbs.get('deployment'),
                                    task_deployments=declared.get('task_deployments',{})))
                reuse = connections.options(['ssh'], pbs['ssh_alias']) if connections else []
                command = ['ssh', *reuse, '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                           pbs['ssh_alias'], shlex.join([pbs['gateway_python'], '-B', '-c', _HOST_PROBE])]
                def invoke(data):
                    checked = subprocess.run(command, input=json.dumps(data), text=True,
                                             capture_output=True, check=True, timeout=120)
                    return json.loads(checked.stdout)
            elif worker is None:
                payload.update(profile=local, worker_root=str(executor.worker_script.parent.parent))
                distributed=executor.distributed
                if distributed.get('local_deployment_layout') and distributed.get('release_manifest'):
                    payload['deployment']=dict(deployment=dict(manifest=distributed['release_manifest'],
                                                              layout=distributed['local_deployment_layout']))
                def invoke(data):
                    return json.loads(subprocess.check_output([sys.executable, '-B', '-c', _HOST_PROBE],
                                                              input=json.dumps(data), text=True, timeout=120))
            else:
                # Keep Windows argv short even as the metadata checker grows.
                payload.update(worker_root=worker['root'], profile_path=worker['profile'],
                               probe_code=base64.b64encode(zlib.compress(_HOST_PROBE.encode())).decode())
                payload['deployment']={key:worker[key] for key in ('deployment','task_deployments') if key in worker}
                def invoke(data):
                    return bridge.remote_command(worker, [worker.get('python', 'python3'), '-B', '-c', _REMOTE_PROBE],
                                                 'q3-readiness', data)[0]
            result = invoke(payload)
            if result['status'] == 'NEEDS_Q3_INPUT_OBJECTS':
                result = _supply_inputs(executor, payload, invoke, result)
            return host, result
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            return host, dict(status='HOLD', reason=str(exc)[-2000:])

    def check(item):
        started = time.monotonic()
        host, result = check_host(item)
        return host, dict(result, readiness_seconds=time.monotonic() - started)

    def snapshot():
        value = deepcopy(decision)
        value['eligible_host_ids'].sort()
        value['reasons'] = [] if value['eligible_host_ids'] else ['ALL_LIVE_DRY_CHECKS_HELD']
        return dict(value, decision_sha256=Q.digest(value))

    observations = {}
    record = executor.run_dir/'distributed'/'q3-readiness'/job_id/'READINESS.json'
    previous = {}
    if targets is not None and record.is_file():
        saved = json.loads(record.read_bytes())
        if saved.get('assignment') == assignment:
            previous = saved.get('observations', {})

    def save(complete):
        with executor.state.cv:
            if not executor.state.closed:
                record.parent.mkdir(parents=True, exist_ok=True)
                Q.atomic_json(record, dict(assignment=assignment, observations={**previous, **observations},
                                           complete=complete))

    decision.update(input_hashes=assignment['input_hashes'], execution_profiles={}, host_decisions={})

    def receive(host, result):
        observations[host] = result
        fingerprint = result.get('fingerprint') or {}
        current = result.get('status') == 'PASS' and result.get('request_sha256') == assignment['request_sha256']
        if current and fingerprint.get('source_manifest') == executor.source_manifest:
            decision['eligible_host_ids'].append(host)
            decision['host_fingerprints'][host] = Q.digest(fingerprint)
            decision['execution_profiles'][host] = result['profile']
            decision['host_decisions'][host] = dict(status='PASS', reasons=[])
            if result.get('evaluator'):
                decision['host_decisions'][host]['evaluator'] = result['evaluator']
            if publish is not None:
                save(complete=False)
                publish(snapshot())
        else:
            decision['host_decisions'][host] = dict(status='HOLD', reasons=[result.get('reason', 'Fingerprint differs')])

    def finish(deferred=()):
        save(complete=not deferred)
        return dict(snapshot(), deferred_hosts=sorted(deferred)) if deferred else snapshot()

    if dispatch is not None:
        dispatch(participants, check, receive, finish)
        return None
    with ThreadPoolExecutor(max_workers=max(1, len(participants))) as pool:
        for future in as_completed(pool.submit(check, item) for item in participants):
            receive(*future.result())
    return finish()
