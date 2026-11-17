"""Request-specific dry-checks using the already deployed worker and adapter."""
import base64
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import zlib

from . import rpe_q3_coordinator as Q


# Runs at the host. This only invokes the existing no-science readiness entry.
_HOST_PROBE = r"""
import json
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
        configured = json.loads(profile.read_text())
        effective = select_host_profile(configured, assignment)
        fingerprint = current_host_fingerprint(assignment, configured)
        print(json.dumps(dict(status='PASS', request_sha256=assignment['request_sha256'],
                              fingerprint=fingerprint, profile=profile_core(effective))))
"""


def probe(executor, request, job_id, decision):
    """Obtain real-input dry checks for new jobs before reserving a slot."""
    assignment = dict(job_id=job_id, method=request['method'], request=request,
                      request_sha256=Q.digest(request), source_manifest=executor.source_manifest,
                      input_hashes=Q.input_hashes(executor.computation_root, request, executor.input_hash_cache))
    local = executor.distributed.get('local_worker') or {}
    policy, error = executor._host_admission_policy()
    participants = []
    if local and executor._host_admission(local['host_id'], policy, error)['accepting']:
        participants.append((local['host_id'], None))
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
            if worker is None:
                payload = dict(assignment=assignment, profile=local,
                               worker_root=str(executor.worker_script.parent.parent))
                response = subprocess.check_output([sys.executable, '-B', '-c', _HOST_PROBE],
                                                   input=json.dumps(payload), text=True, timeout=120)
                return host, json.loads(response)
            code = base64.b64encode(zlib.compress(_HOST_PROBE.encode())).decode()
            command = 'import base64,zlib;exec(zlib.decompress(base64.b64decode(' + repr(code) + ')))'
            payload = dict(assignment=assignment, worker_root=worker['root'], profile_path=worker['profile'])
            result, _ = bridge.remote_command(worker, [worker.get('python', 'python3'), '-B', '-c', command],
                                              'q3-readiness', payload)
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
