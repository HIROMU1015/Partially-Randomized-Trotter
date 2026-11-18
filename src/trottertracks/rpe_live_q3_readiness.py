"""Request-specific dry-checks using the already deployed worker and adapter."""
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import zlib

from . import rpe_q3_coordinator as Q
from .rpe_pbs import task_estimate

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
from trottertracks.rpe_host_eligibility import current_host_fingerprint
from trottertracks.rpe_host_profiles import profile_core, select_host_profile

assignment = payload['assignment']
with tempfile.TemporaryDirectory(prefix='rpe-q3-live-') as directory:
    base = Path(directory)
    profile = Path(payload['profile_path']) if payload.get('profile_path') else base/'profile.json'
    if not payload.get('profile_path'):
        profile.write_text(json.dumps(payload['profile']))
    configured = json.loads(profile.read_text())
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

    # Input commits may be newer than this fixed scientific checkout.
    # Install their missing objects only; the checkout and its refs stay fixed.
    missing = []
    for repo in sorted(repositories):
        locators = []
        for row in assignment['input_hashes']:
            if git(repo, 'cat-file', '-e', row['commit']+':'+row['path']).returncode:
                locators.append(row)
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
        print(json.dumps(dict(status='HOLD', reason=result.get('error') or checked.stderr[-2000:])))
    elif result.get('scientific_evaluation_calls') != 0 or result.get('request_sha256') != assignment['request_sha256']:
        raise ValueError('Dry-check changed the request or executed a scientific call')
    else:
        fingerprint = current_host_fingerprint(assignment, configured)
        print(json.dumps(dict(status='PASS', request_sha256=assignment['request_sha256'],
                              fingerprint=fingerprint, profile=profile_core(effective))))
"""


def _supply_inputs(executor, payload, invoke, response):
    """Reuse the host's existing input objects and send the missing delta."""
    repositories = response['repositories']
    anchors = set.intersection(*(set(row['anchors']) for row in repositories))
    if not anchors:
        return dict(status='HOLD', reason='Q3 input delivery requires a placed fixed Git base')
    commits = {commit for row in repositories for commit in row['commits']}
    if not anchors <= set(payload['input_anchors']) or not commits <= {
            row['commit'] for row in payload['assignment']['input_hashes']}:
        raise ValueError('Worker Git input inventory is outside the fixed Q3 assignment')
    root = executor.computation_root
    objects = set(subprocess.check_output(['git', 'rev-list', '--objects', '--no-object-names',
                                          *sorted(commits), '--not', *sorted(anchors)], cwd=root, text=True).split())
    # Sparse worker stores can lack trees or blobs shared with an anchor.
    # Include only the fixed locators' commits and their path components.
    locators = []
    for row in payload['assignment']['input_hashes']:
        commit = row['commit']
        if commit in commits:
            path = Path(row['path'])
            locators.extend([commit, commit+'^{tree}',
                             *(commit+':'+p.as_posix() for p in (*path.parents, path) if p != Path('.'))])
    if locators:
        objects.update(subprocess.check_output(['git', 'rev-parse', *locators], cwd=root, text=True).split())
    inventory = invoke(dict(payload, input_objects=sorted(objects)))
    if inventory['status'] != 'NEEDS_Q3_INPUT_OBJECTS':
        return inventory
    needed = sorted({oid for row in inventory['repositories'] for oid in row['objects']})
    if not needed:
        return dict(status='HOLD', reason='Q3 input locator missing despite available Git objects')
    if not set(needed) <= set(objects):
        raise ValueError('Worker requested objects outside the fixed Q3 input delta')
    content = subprocess.check_output(['git', 'pack-objects', '--stdout', '--window=0'], cwd=root,
                                      input=('\n'.join(needed)+'\n').encode())
    sha256 = hashlib.sha256(content).hexdigest()
    result = invoke(dict(payload, input_pack=base64.b64encode(content).decode(), input_pack_sha256=sha256))
    return dict(result, input_transfer=dict(objects=len(needed), bytes=len(content), sha256=sha256))


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


def probe(executor, request, job_id, decision):
    """Obtain real-input dry checks for new jobs before reserving a slot."""
    assignment = dict(job_id=job_id, method=request['method'], request=request,
                      request_sha256=Q.digest(request), source_manifest=executor.source_manifest,
                      input_hashes=controller_input_hashes(executor,request))
    local = executor.distributed.get('local_worker') or {}
    policy, error = executor._host_admission_policy()
    participants = []
    pbs = executor.distributed.get('pbs') or {}
    pbs_workers = {worker['host_id']: worker for worker in pbs.get('workers', [])
                   if pbs.get('walltime_policy') and executor._host_admission(worker['host_id'], policy, error)['accepting']}
    if local and executor._host_admission(local['host_id'], policy, error)['accepting']:
        participants.append((local['host_id'], None))
    participants.extend((host, None) for host in pbs_workers)
    relay = executor.run_dir/'distributed'/'relay'/'CONFIG.json'
    bridge = None
    if relay.is_file():
        from run_rpe_q3_relay import Relay
        bridge = Relay(relay)
        bridge.config['request_timeout_seconds'] = 120
        participants.extend((worker['host_id'], worker) for worker in bridge.config.get('workers', [])
                            if worker.get('enabled', True)
                            and executor._host_admission(worker['host_id'], policy, error)['accepting']
                            and (not executor._host_admission(worker['host_id'], policy, error)['optional']
                                 or executor.hosts.get(worker['host_id'], {}).get('accepting_new_jobs') is True))

    def check(item):
        host, worker = item
        try:
            anchors = [executor.computation_source, (request.get('route') or {}).get('view_commit')]
            provider = executor.distributed.get('live_q3_input_base')
            if provider:
                anchors.append(provider)
            payload = dict(assignment=assignment, input_anchors=[value for value in anchors if value])
            if host in pbs_workers:
                if not (task_estimate(request, pbs, host) or 0) > 0:
                    return host, dict(status='HOLD', reason='PBS_WALLTIME_ESTIMATE_UNAVAILABLE')
                # The gateway and compute nodes share the deployed fixed views.
                # The worker repeats the normal fingerprint on the allocated node.
                payload.update(profile=pbs_workers[host]['profile'], worker_root=pbs['transport_root'])
                command = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                           pbs['ssh_alias'], shlex.join([pbs['gateway_python'], '-B', '-c', _HOST_PROBE])]
                def invoke(data):
                    checked = subprocess.run(command, input=json.dumps(data), text=True,
                                             capture_output=True, check=True, timeout=120)
                    return json.loads(checked.stdout)
            elif worker is None:
                payload.update(profile=local, worker_root=str(executor.worker_script.parent.parent))
                def invoke(data):
                    return json.loads(subprocess.check_output([sys.executable, '-B', '-c', _HOST_PROBE],
                                                              input=json.dumps(data), text=True, timeout=120))
            else:
                # Keep Windows argv short even as the metadata checker grows.
                payload.update(worker_root=worker['root'], profile_path=worker['profile'],
                               probe_code=base64.b64encode(zlib.compress(_HOST_PROBE.encode())).decode())
                def invoke(data):
                    return bridge.remote_command(worker, [worker.get('python', 'python3'), '-B', '-c', _REMOTE_PROBE],
                                                 'q3-readiness', data)[0]
            result = invoke(payload)
            if result['status'] == 'NEEDS_Q3_INPUT_OBJECTS':
                result = _supply_inputs(executor, payload, invoke, result)
            return host, result
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            return host, dict(status='HOLD', reason=str(exc)[-2000:])

    with ThreadPoolExecutor(max_workers=max(1, len(participants))) as pool:
        observations = dict(pool.map(check, participants))
    record = executor.run_dir/'distributed'/'q3-readiness'/job_id/'READINESS.json'
    record.parent.mkdir(parents=True, exist_ok=True)
    Q.atomic_json(record, dict(assignment=assignment, observations=observations))
    decision.update(input_hashes=assignment['input_hashes'], execution_profiles={}, host_decisions={})
    for host, result in observations.items():
        fingerprint = result.get('fingerprint') or {}
        current = result.get('status') == 'PASS' and result.get('request_sha256') == assignment['request_sha256']
        if current and fingerprint.get('source_manifest') == executor.source_manifest:
            decision['eligible_host_ids'].append(host)
            decision['host_fingerprints'][host] = Q.digest(fingerprint)
            decision['execution_profiles'][host] = result['profile']
            decision['host_decisions'][host] = dict(status='PASS', reasons=[])
        else:
            decision['host_decisions'][host] = dict(status='HOLD', reasons=[result.get('reason', 'Fingerprint differs')])
    decision['reasons'] = [] if decision['eligible_host_ids'] else ['ALL_LIVE_DRY_CHECKS_HELD']
    decision['decision_sha256'] = Q.digest(decision)
    return decision
