"""Derive per-request host eligibility from saved, received dry-check evidence.

This module does not start a controller or a scientific process. Saved PASS is
useful only for the same full request and a matching current host fingerprint.
Incomplete or inconsistent evidence is UNCHECKED; a bound dry-check failure
remains FAIL. There are no method names or candidate IDs in this policy.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath

from . import rpe_q3_coordinator as Q


SCHEMA = 'RPE_Q3_HOST_ELIGIBILITY_V1'


def _read(value):
    return json.loads(Path(value).read_text()) if isinstance(value, (str, Path)) else deepcopy(value)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _path(value, base=None):
    value = Path(value)
    return value if value.is_absolute() else Path(base or '.') / value


def _inputs(rows):
    return sorted(rows, key=lambda row: (row['commit'], row['path'], row['sha256']))


def _effective_profile(profile, assignment):
    route = assignment['request']['route']
    return dict(host_id=profile['host_id'], root=profile['root'], interpreter=profile['interpreter'],
                route_view=profile['route_views'][route['view_commit']],
                dp_workers=profile.get('dp_workers'),
                prefix_cache_directory=profile.get('prefix_cache_directory'))


def _resolved_request(assignment, profile):
    """The path-only transformation in the existing worker validator.

    We cannot call validate_assignment on the controller for a remote filesystem.
    Its fixed-source/input checks are instead bound by saved evidence and the
    worker's current_host_fingerprint at each poll.
    """
    request = deepcopy(assignment['request'])
    request['route']['interpreter'] = profile['interpreter']
    request['route']['view'] = profile['route_views'][request['route']['view_commit']]
    execution = request.setdefault('execution_profile', {})
    if profile.get('dp_workers') is not None:
        execution['dp_workers'] = profile['dp_workers']
    if profile.get('prefix_cache_directory') is not None:
        execution['prefix_cache_directory'] = profile['prefix_cache_directory']
    else:
        execution.pop('prefix_cache_directory', None)
    return request


def _fingerprint(assignment, profile, runtime, views=None, actual=None):
    route = assignment['request']['route']
    selected = profile['route_views'][route['view_commit']]
    if actual is None:
        source = assignment['source_manifest']
        source_head = source['commit']
        inputs = assignment.get('input_hashes', [])
    else:
        source, source_head, inputs = actual['source'], actual['source_head'], actual['input_hashes']
    normalized_views = []
    for view in views or []:
        normalized_views.append(dict(path=view['path'], commit=view.get('head', view.get('commit')),
                                     clean_tracked=not view.get('tracked_changes', {})))
    if not normalized_views:
        normalized_views.append(dict(path=selected, commit=route['view_commit'], clean_tracked=True))
    return dict(schema='RPE_Q3_CURRENT_HOST_FINGERPRINT_V1',
                request_sha256=Q.digest(assignment['request']),
                source_manifest=source, source_head=source_head, input_hashes=_inputs(inputs),
                profile=_effective_profile(profile, assignment),
                views=sorted(normalized_views, key=lambda view: view['path']),
                runtime=dict(executable=runtime['executable'], version=runtime['version']))


def current_host_fingerprint(assignment, profile):
    """Read actual source, input, route and interpreter metadata for a poll.

    Reuse the existing readiness fingerprint, including its known nested route
    checks. The subprocess only reads Python/runtime metadata; no adapter/Q3 is
    invoked. Errors deliberately propagate so the caller can withhold new claims.
    """
    from .rpe_plan_readiness import dry_check_fingerprint
    assignment, profile = _read(assignment), _read(profile)
    raw = dry_check_fingerprint(profile['root'], assignment, profile)
    # The older readiness helper labels its fixed source as 28d. Actual HEAD is
    # separate and must be used for a newly fixed worker source as well.
    raw['source'] = dict(raw['source'], commit=raw['source_head'])
    return _fingerprint(assignment, profile, raw['runtime'], raw['views'], actual=raw)


def _receipt_check(descriptor, base):
    """Verify the existing transfer/receipt mechanism; invent no new authority."""
    if descriptor.get('kind') == 'local':
        return dict(status='MATCH', kind='DIRECT_LOCAL_OBSERVATION')
    manifest_path = _path(descriptor['manifest'], base)
    receipt_path = _path(descriptor['receipt'], base)
    manifest, receipt = _read(manifest_path), _read(receipt_path)
    manifest_root = _path(descriptor.get('manifest_root', descriptor['root']), base).resolve()
    if 'transfer_manifest_sha256' in receipt:
        if receipt['transfer_manifest_sha256'] != _sha(manifest_path):
            raise ValueError('Received transfer manifest hash differs')
        if receipt.get('verified_transferred_files') != len(manifest):
            raise ValueError('Received file count differs from transfer manifest')
    elif receipt.get('received') is True and descriptor.get('archive'):
        if receipt.get('archive_sha256') != _sha(_path(descriptor['archive'], base)):
            raise ValueError('Received evidence archive hash differs')
        if receipt.get('verified_files') != len(manifest):
            raise ValueError('Received evidence file count differs')
    else:
        raise ValueError('No matching existing receipt for this manifest')
    for name, expected in manifest.items():
        part = PurePosixPath(name)
        if part.is_absolute() or '..' in part.parts:
            raise ValueError('Unsafe received evidence path')
        # Verify the files used by each cell below. A missing unrelated log must
        # not erase another candidate's independently hash-bound dry-check.
    return dict(status='MATCH', kind='HASH_VERIFIED_RECEIVED_EVIDENCE',
                manifest=str(manifest_path), manifest_sha256=_sha(manifest_path),
                receipt=str(receipt_path), receipt_sha256=_sha(receipt_path),
                manifest_root=str(manifest_root), files=manifest)


def _covered(path, receipt):
    if receipt['kind'] == 'DIRECT_LOCAL_OBSERVATION':
        return
    relative = Path(path).resolve().relative_to(Path(receipt['manifest_root'])).as_posix()
    if relative not in receipt['files'] or _sha(path) != receipt['files'][relative]:
        raise ValueError('Evidence is not covered by the received manifest: ' + relative)


def _host_evidence(descriptor, base):
    descriptor = deepcopy(descriptor)
    root = _path(descriptor['root'], base)
    receipt = _receipt_check(descriptor, base)
    facts = None
    facts_path = _path(descriptor.get('host_facts', str(root / 'HOST_FACTS.json')), base)
    if facts_path.is_file():
        _covered(facts_path, receipt)
        facts = _read(facts_path)
    cache = {}
    if descriptor.get('runtime_cache'):
        cache = _read(_path(descriptor['runtime_cache'], base)).get('methods', {})
    records = []
    for path in sorted(root.rglob('READINESS.json')):
        observed = {}
        try:
            observed = _read(path)
            _covered(path, receipt)
            original_path = path
            runtime_item = next((item for item in cache.values()
                                 if item.get('result_sha256') == _sha(path)), None)
            if observed.get('status') == 'CACHED_RESULT_REUSED':
                if runtime_item is None:
                    raise ValueError('Cached dry-check wrapper is not hash-bound by the index')
                original_path = Path(observed['original_result'])
                if observed.get('original_result_sha256') != _sha(original_path):
                    raise ValueError('Cached dry-check original hash differs or is missing')
                _covered(original_path, receipt)
                observed = _read(original_path)
            elif runtime_item is None and descriptor.get('kind') == 'local':
                raise ValueError('Local dry-check record is not hash-bound by the runtime index')
            records.append(dict(path=original_path, readiness=observed, runtime_item=runtime_item))
        except Exception as exc:
            records.append(dict(path=path, readiness=observed, error=str(exc)))
    return dict(root=root, receipt=receipt, facts=facts, records=records)


def _fact_views(assignment, profile, facts):
    """Read both already published HOST_FACTS schemas."""
    route = assignment['request']['route']
    selected = profile['route_views'][route['view_commit']]
    if 'source_files' in facts:
        head = facts.get('source_head') or {}
        if head.get('returncode') != 0 or head.get('stdout', '').strip() != assignment['source_manifest']['commit']:
            raise ValueError('Received SOURCE HEAD differs')
        for name, expected in assignment['source_manifest']['files'].items():
            check = facts['source_files'].get(name) or {}
            if check.get('expected') != expected or check.get('actual') != expected or check.get('match') is not True:
                raise ValueError('Received SOURCE file verification differs: ' + name)
        saved = facts.get('route_views', {}).get(route['view_commit']) or {}
        head, diff = saved.get('head', {}), saved.get('tracked_source_diff', {})
        if saved.get('path') != selected or head.get('returncode') != 0 or head.get('stdout', '').strip() != route['view_commit']:
            raise ValueError('Received route view HEAD/path differs')
        if diff.get('returncode') != 0 or diff.get('stdout', '').strip():
            raise ValueError('Received route view has unknown or changed tracked SOURCE')
        views = [dict(path=saved.get('realpath', selected), commit=route['view_commit'])]
        if route['kind'].startswith('prdf_d2'):
            inner = facts.get('pr_df_inner_source') or {}
            if inner.get('exists') and inner.get('head', {}).get('returncode') == 0 and inner.get('tracked_source_diff', {}).get('returncode') == 0:
                views.append(dict(path=inner.get('realpath', inner['required_path']),
                                  commit=inner['head']['stdout'].strip(),
                                  tracked_changes={} if not inner['tracked_source_diff'].get('stdout', '').strip() else {'UNKNOWN': True}))
        return views
    source_view = facts.get('views', {}).get(profile['root']) or {}
    route_view = facts.get('views', {}).get(selected) or {}
    if source_view.get('commit') != assignment['source_manifest']['commit'] or source_view.get('checked_files', 0) < len(assignment['source_manifest']['files']):
        raise ValueError('No received matching SOURCE file verification')
    if route_view.get('commit') != route['view_commit'] or route_view.get('checked_files', 0) < 1:
        raise ValueError('No received matching route file verification')
    return [dict(path=route_view.get('realpath', selected), commit=route_view['commit'])]


def _cell(assignment, profile, evidence, *, current_inputs=None, current_source=None):
    dimensions = {name: 'UNCHECKED' for name in ('request', 'source', 'input', 'route', 'runtime', 'receipt')}
    result = dict(status='UNCHECKED', dimensions=dimensions, reasons=[], evidence=[])
    receipt = evidence['receipt']
    dimensions['receipt'] = 'MATCH'
    matching = [item for item in evidence['records'] if
                item['readiness'].get('job_id') == assignment['job_id'] and
                item['readiness'].get('request_sha256') == assignment['request_sha256']]
    if not matching:
        result['reasons'].append('No received dry-check for this full current request/job identity')
        return result
    checked = []
    for item in matching:
        row = deepcopy(result)
        dims = row['dimensions']
        observed, path = item['readiness'], item['path']
        try:
            if item.get('error'):
                raise ValueError(item['error'])
            if Q.digest(assignment['request']) != assignment['request_sha256']:
                raise ValueError('Current request hash differs from its assignment')
            if observed.get('host_id') != profile['host_id']:
                raise ValueError('Saved host ID differs from the current host profile')
            if observed.get('scientific_evaluation_calls') != 0 or observed.get('queue_operations') != 0:
                raise ValueError('Saved result does not establish a dry-check-only observation')
            dims['request'] = 'MATCH'
            if observed.get('source') != assignment['source_manifest']['commit']:
                raise ValueError('Saved SOURCE commit differs from the current assignment')
            if current_source is not None and current_source != assignment['source_manifest']:
                raise ValueError('Current computation SOURCE files differ from the assignment')
            dims['source'] = 'MATCH'
            if _inputs(observed.get('input_hashes', [])) != _inputs(assignment.get('input_hashes', [])):
                raise ValueError('Saved input hashes differ from current request inputs')
            if current_inputs is not None and _inputs(current_inputs) != _inputs(assignment.get('input_hashes', [])):
                raise ValueError('Current fixed input blobs differ from the assignment')
            dims['input'] = 'MATCH'
            resolved_path = path.parent / 'RESOLVED_REQUEST.json'
            _covered(resolved_path, receipt)
            resolved = _read(resolved_path)
            if _sha(resolved_path) != observed.get('resolved_request_sha256'):
                raise ValueError('Saved resolved request bytes differ from the dry-check record')
            expected_resolved = _resolved_request(assignment, profile)
            if resolved != expected_resolved or observed.get('route') != expected_resolved['route']:
                raise ValueError('Saved resolved route/profile/request differs from the current host contract')
            dims['route'] = 'MATCH'
            command = observed.get('command') or []
            if not command or command[0] != profile['interpreter'] or command[-2:] != ['--mode', 'dry-check']:
                raise ValueError('Saved adapter interpreter or explicit dry-check command differs')
            facts = evidence['facts']
            raw = (item['runtime_item'] or {}).get('fingerprint')
            if raw:
                if (raw.get('request_sha256') != assignment['request_sha256'] or
                        _effective_profile(raw.get('profile', {}), assignment) != _effective_profile(profile, assignment)):
                    raise ValueError('Saved local runtime fingerprint is for another request/profile')
                if raw.get('source') != assignment['source_manifest'] or raw.get('source_head') != assignment['source_manifest']['commit']:
                    raise ValueError('Saved local SOURCE fingerprint differs')
                if _inputs(raw.get('input_hashes', [])) != _inputs(assignment.get('input_hashes', [])):
                    raise ValueError('Saved local input fingerprint differs')
                runtime, views = raw['runtime'], raw['views']
            else:
                if facts is None or facts.get('python') != profile['interpreter'] or facts.get('python_version') != observed.get('python_version'):
                    raise ValueError('No matching actual interpreter/runtime observation')
                views = _fact_views(assignment, profile, facts)
                runtime = dict(executable=facts['python'], version=facts['python_version'])
            if runtime.get('executable') != profile['interpreter'] or not runtime.get('version'):
                raise ValueError('Actual interpreter identity/version is missing or mismatched')
            expected_runtime = profile.get('runtime') or {}
            for key, expected in expected_runtime.items():
                if runtime.get(key) != expected:
                    raise ValueError('Saved runtime differs from the current host runtime requirement: ' + key)
            if any(view.get('tracked_changes') for view in views):
                raise ValueError('Saved route had tracked SOURCE changes')
            dims['runtime'] = 'MATCH'
            status = observed.get('status')
            if status == 'REAL_HOST_DRY_CHECK_PASS':
                adapter_path = path.parent / 'adapter/DRY_CHECK.json'
                _covered(adapter_path, receipt)
                adapter = _read(adapter_path)
                if (adapter != observed.get('adapter_result') or adapter.get('status') != 'DRY_CHECK_PASS'
                        or adapter.get('mode') != 'dry-check' or adapter.get('request_id') != assignment['job_id']
                        or adapter.get('route') != assignment['request']['route']['kind']):
                    raise ValueError('Saved adapter PASS evidence differs or is incomplete')
                row['status'] = 'PASS'
            elif status == 'REAL_HOST_DRY_CHECK_MISSING_OR_INVALID_INPUT':
                row['status'] = 'FAIL'
                row['reasons'].append(observed.get('error') or observed.get('error_type') or 'Existing real-input dry-check failed')
            else:
                raise ValueError('No established real-input dry-check status')
            fingerprint = _fingerprint(assignment, profile, runtime, views)
            row.update(current_host_fingerprint=fingerprint,
                       current_host_fingerprint_sha256=Q.digest(fingerprint),
                       host_profile_sha256=Q.digest(_effective_profile(profile, assignment)))
            row['evidence'].append(dict(readiness=str(path), sha256=_sha(path), receipt=receipt.get('receipt'),
                                        manifest=receipt.get('manifest'), runtime_source='LOCAL_FINGERPRINT' if raw else 'RECEIVED_HOST_FACTS'))
        except Exception as exc:
            row['status'] = 'UNCHECKED'
            row['reasons'].append(str(exc))
        checked.append(row)
    # Do not turn an established failure into PASS because another stale record exists.
    return next((row for row in checked if row['status'] == 'FAIL'),
                next((row for row in checked if row['status'] == 'PASS'), checked[-1]))


def _items(values, field):
    if isinstance(values, (str, Path)):
        path = Path(values)
        if path.is_dir():
            values = list(path.glob('*.json'))
        else:
            values = _read(path)
    if isinstance(values, dict):
        if field in values:
            return [values]
        values = values.values()
    return [_read(value) for value in values]


def derive_host_eligibility(assignments, profiles, evidence_by_host, *, root=None, base_directory=None):
    """Derive a fresh policy from full current requests and saved evidence only.

    Descriptors use root, kind=local/received, optional runtime_cache, and for
    received evidence manifest/receipt/manifest_root. Relative descriptor paths
    are resolved against base_directory. No evidence file is modified.
    """
    assignments = _items(assignments, 'request')
    profiles = {profile['host_id']: profile for profile in _items(profiles, 'host_id')}
    prepared_evidence = {}
    for host, descriptor in evidence_by_host.items():
        try:
            prepared_evidence[host] = _host_evidence(descriptor, base_directory)
        except Exception as exc:
            prepared_evidence[host] = dict(error=str(exc))
    jobs = {}
    for assignment in assignments:
        hosts = {}
        current_inputs = current_source = None
        current_error = None
        if root is not None:
            try:
                current_inputs = Q.input_hashes(root, assignment['request'])
                current_source = Q.source_manifest(root, assignment['source_manifest']['commit'])
                import subprocess
                current_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
                if current_head != assignment['source_manifest']['commit']:
                    raise ValueError('Controller computation SOURCE HEAD differs from current assignment')
            except Exception as exc:
                current_error = str(exc)
        for host, profile in profiles.items():
            evidence = prepared_evidence.get(host) or dict(error='No saved received evidence descriptor for this host')
            if current_error or evidence.get('error'):
                hosts[host] = dict(status='UNCHECKED', reasons=[current_error or evidence['error']],
                    dimensions={key: 'UNCHECKED' for key in ('request', 'source', 'input', 'route', 'runtime', 'receipt')}, evidence=[])
            else:
                try:
                    hosts[host] = _cell(assignment, profile, evidence, current_inputs=current_inputs, current_source=current_source)
                except Exception as exc:
                    hosts[host] = dict(status='UNCHECKED', reasons=[str(exc)],
                        dimensions={key: 'UNCHECKED' for key in ('request', 'source', 'input', 'route', 'runtime', 'receipt')}, evidence=[])
        row = dict(method=assignment['method'], request_sha256=assignment['request_sha256'],
                   source_manifest=assignment['source_manifest'], source_manifest_sha256=Q.digest(assignment['source_manifest']),
                   input_hashes=assignment.get('input_hashes', []), hosts=hosts,
                   eligible_host_ids=sorted(host for host, status in hosts.items() if status['status'] == 'PASS'))
        row['decision_sha256'] = Q.digest(row)
        jobs[assignment['job_id']] = row
    return dict(schema=SCHEMA, scope='SAVED_DRY_CHECK_EVIDENCE_REQUIRES_CURRENT_POLL_FINGERPRINT',
                jobs=jobs, scientific_evaluation_calls=0, queue_operations=0)
