#!/usr/bin/env python3
"""Collect current metadata for an already completed official real-input dry-check.

No worker, queue, adapter, simulation or scientific computation is started here.
Run in a fresh interpreter, with the fixed transport checkout supplied explicitly.
The existing eligibility derivation checks the resulting runtime index unchanged.
"""
import argparse
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2) + '\n').encode()
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError('Existing proof bytes differ; preserved: ' + str(path))
    else:
        with path.open('xb') as stream:
            stream.write(raw)


def bind_runtime(profile, fingerprint, actual, observed):
    """Bind the actual selected interpreter, never the checker's top-level Python."""
    result = deepcopy(fingerprint)
    runtime = actual['runtime']
    if actual.get('status') != 'PASS':
        raise ValueError('Actual selected-runtime metadata/precheck did not pass')
    for key in ('executable', 'version'):
        if result['runtime'].get(key) != runtime.get(key):
            raise ValueError('Current runtime metadata changed: ' + key)
    if runtime['executable'] != profile['interpreter']:
        for key in ('interpreter_invocation', 'launcher_sha256', 'actual_executable_realpath', 'actual_binary_sha256'):
            if not runtime.get(key):
                raise ValueError('Actual launcher identity is incomplete: ' + key)
            result['runtime'][key] = runtime[key]
    if profile.get('runtime_checks'):
        if observed.get('runtime_precheck') != actual:
            raise ValueError('Official runtime precheck differs from the currently selected runtime')
    guard = (observed.get('adapter_result') or {}).get('runtime')
    if guard:
        if guard.get('executable') != runtime['executable'] or guard.get('version') != runtime['version']:
            raise ValueError('Official adapter runtime differs from actual selected interpreter')
        if guard.get('guard_modified') is not False or guard.get('constructor') != 'PASS_UNCHANGED_GUARD':
            raise ValueError('Original adapter runtime/AST guard did not pass unchanged')
    return result


def collect(transport_root, assignment_path, profile_path, readiness_path, output_directory):
    root = Path(transport_root).resolve()
    # Use the fixed transport implementation, not the scientific validation
    # module imported in a separate checker process from the original 28d root.
    existing = sys.modules.get('trottertracks')
    if existing is not None and not Path(existing.__file__).resolve().is_relative_to(root):
        raise ValueError('Use a fresh interpreter; another transport package is already imported')
    sys.path.insert(0, str(root / 'src'))
    H = importlib.import_module('trottertracks.rpe_host_profiles')
    R = importlib.import_module('trottertracks.rpe_plan_readiness')
    E = importlib.import_module('trottertracks.rpe_host_eligibility')
    W = importlib.import_module('trottertracks.rpe_q3_worker')
    Q = importlib.import_module('trottertracks.rpe_q3_coordinator')
    for module in (H, R, E, W, Q):
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise ValueError('Metadata helper module is outside the fixed transport checkout')
    assignment, base = read(assignment_path), read(profile_path)
    profile = H.select_host_profile(base, assignment)
    adapter = H.validate_adapter_profile(profile)
    if not adapter or Path(adapter['root']).resolve() != root:
        raise ValueError('Selected timed adapter is not the fixed transport checkout')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    if head != adapter['commit']:
        raise ValueError('Transport HEAD differs from adapter source binding')
    path = Path(readiness_path).resolve(); observed = read(path)
    if observed.get('status') not in ('REAL_HOST_DRY_CHECK_PASS', 'REAL_HOST_DRY_CHECK_MISSING_OR_INVALID_INPUT'):
        raise ValueError('Expected the original official real-input dry-check result')
    if observed.get('scientific_evaluation_calls') != 0 or observed.get('queue_operations') != 0:
        raise ValueError('Observation was not a dry-check-only result')
    expected = dict(host_id=profile['host_id'], job_id=assignment['job_id'],
                    source=assignment['source_manifest']['commit'], request_sha256=assignment['request_sha256'])
    if any(observed.get(k) != v for k, v in expected.items()):
        raise ValueError('Official observation identity differs from the supplied assignment/profile')
    if Q.digest(assignment['request']) != assignment['request_sha256']:
        raise ValueError('Assignment request bytes differ from their declared identity')
    resolved = W.validate_assignment(profile, assignment)
    resolved_path = path.parent / 'RESOLVED_REQUEST.json'
    if read(resolved_path) != resolved or sha(resolved_path) != observed.get('resolved_request_sha256'):
        raise ValueError('Official resolved request differs from actual fixed-source validation')
    command = observed.get('command') or []
    if (len(command) < 3 or command[0] != profile['interpreter'] or
            command[2] != str(root / 'scripts/run_rpe_optimizer_q3_job.py') or
            command.count('--mode') != 1 or command[command.index('--mode') + 1] != 'dry-check'):
        raise ValueError('Official command is not this timed adapter in explicit dry-check mode')
    saved_adapter = observed.get('adapter_source') or {}
    if (saved_adapter.get('manifest') != adapter or saved_adapter.get('root') != str(root) or
            saved_adapter.get('original_assignment_manifest_rewritten') is not False):
        raise ValueError('Official adapter provenance differs from its selected fixed source')
    maps = path.parent / 'ROUTE_VIEWS.json'
    if read(maps) != profile['route_views']:
        raise ValueError('Official route mapping differs from the selected profile')
    if command.count('--route-views') != 1 or command[command.index('--route-views') + 1] != str(maps):
        raise ValueError('Official command used another route mapping')
    fingerprint = R.dry_check_fingerprint(profile['root'], assignment, base)
    actual = H.check_runtime_profile(profile)  # pure runtime/AST metadata, no numerical import
    fingerprint = bind_runtime(profile, fingerprint, actual, observed)
    resolution = (observed.get('adapter_result') or {}).get('source_resolution')
    if resolution:
        if (resolution.get('frozen_sources_modified') is not False or
                profile['route_views'].get(resolution.get('commit')) != resolution.get('view')):
            raise ValueError('Official inner SOURCE resolution is not the selected unchanged mapping')
    output = Path(output_directory).resolve()
    put(output / 'SELECTED_PROFILE.json', profile)
    put(output / 'ACTUAL_RUNTIME.json', actual)
    cache = dict(schema='RPE_PLAN_DRY_CACHE_V1', methods={assignment['method']: dict(
        result=str(path), result_sha256=sha(path), fingerprint=fingerprint,
        evidence_mode='CURRENT_METADATA_BOUND_TO_FRESH_OFFICIAL_TIMED_ADAPTER_DRY_CHECK')},
        scientific_calls=0, queue_operations=0)
    put(output / 'DRY_CACHE.json', cache)
    descriptor = dict(kind='local', root=str(path.parent), runtime_cache=str(output / 'DRY_CACHE.json'))
    policy = E.derive_host_eligibility([assignment], [base], {profile['host_id']: descriptor},
                                    root=profile['root'])
    cell = policy['jobs'][assignment['job_id']]['hosts'][profile['host_id']]
    if observed['status'] == 'REAL_HOST_DRY_CHECK_PASS':
        if cell['status'] != 'PASS':
            raise ValueError('Existing eligibility derivation did not accept the fresh proof: ' + repr(cell['reasons']))
        current = E.current_host_fingerprint(assignment, base)
        if current != cell['current_host_fingerprint']:
            raise ValueError('Derived saved fingerprint differs from the actual worker poll fingerprint')
        put(output / 'CURRENT_HOST_FINGERPRINT.json', current)
    summary = dict(schema='RPE_Q3_DEPLOYMENT_READINESS_COLLECTION_V1', status=cell['status'],
        host_id=profile['host_id'], method=assignment['method'], job_id=assignment['job_id'],
        computation_source=assignment['source_manifest']['commit'], transport_source=head,
        official_readiness_sha256=sha(path), actual_adapter_interpreter=profile['interpreter'],
        actual_runtime=actual['runtime'], worker_fingerprint_matches=cell['status'] == 'PASS',
        scientific_calls=0, queue_operations=0, reasons=cell.get('reasons', []))
    put(output / 'COLLECTION.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transport-root', type=Path, required=True)
    parser.add_argument('--assignment', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--readiness', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.transport_root, args.assignment, args.profile, args.readiness, args.output)
    print(json.dumps({k: result[k] for k in ('status', 'host_id', 'method', 'job_id', 'worker_fingerprint_matches', 'scientific_calls')}))
    return 0 if result['status'] == 'PASS' else 2


if __name__ == '__main__':raise SystemExit(main())
