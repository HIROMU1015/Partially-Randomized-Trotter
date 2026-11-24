"""Reuse committed PF prices with their original C implementation admission."""
from collections import defaultdict
from copy import deepcopy
from functools import lru_cache
import gzip
import hashlib
import json
import os
from pathlib import Path
from contextlib import closing
import sqlite3
import subprocess


def receipt_locators(waves, physical_ids=None, *, index=False):
    """Read only outputs of the existing audited PF stage in this search."""
    from itertools import chain

    if isinstance(waves, Path):
        waves = (waves,)
    result = []
    by_physical = defaultdict(list)
    for candidate_root in chain.from_iterable(map(candidate_roots, waves)):
        provider = '*/pf_auto/pf_completion/provider_workspace/docs/research/distributed_selected_pf/*/'
        paths = chain(candidate_root.glob(provider+'*AUDIT.RESULT.json'),
                      candidate_root.glob(provider+'recovered_prices/*/PF_AUDIT.RESULT.json'))
        for path in sorted(paths):
            try:
                audit = json.loads(path.read_bytes())
                if audit['status'] == 'AUDIT_PASS':
                    for row in audit['prices']:
                        physical = row.get('physical_id')
                        if row['gate'] == 'C_IMPLEMENTATION_GUARD_PASS' and (
                                physical_ids is None or physical in physical_ids):
                            locator = row['price_evidence']['source']
                            if index:
                                by_physical[physical].append(locator)
                            else:
                                result.append(locator)
            except (OSError, KeyError, TypeError, ValueError):
                continue
    if index:
        return by_physical
    return list({json.dumps(loc, sort_keys=True): loc for loc in result}.values())


def candidate_roots(waves):
    return [waves/'candidates', waves/'price-recovery'/'candidates',
            *sorted((waves/'price-recovery').glob('recheck-*/candidates'))]


def prior_waves(run, spec):
    return [p/'pf_auto' for path in spec.get('executor', {}).get('prior_run_dirs', ())
            if (p := Path(path).resolve()) != run.resolve() and (p/'pf_auto'/'WAVES.json').is_file()]


def price_indexes(waves, settings, physical_ids=None, *, audits=None):
    locators = list(settings.get('saved_price_receipts', ()))
    recovered = settings.get('recovered_price_receipts_by_physical_id', {})
    for physical in recovered.keys() if physical_ids is None else physical_ids:
        locators.extend(recovered.get(physical, ()))
    if audits is None:
        locators.extend(receipt_locators(waves, physical_ids))
    else:
        for physical in physical_ids or audits.keys():
            locators.extend(audits.get(physical, ()))
    unique = {json.dumps(loc, sort_keys=True): loc for loc in locators}
    return [unique[key] for key in sorted(unique)]


def completed_partial(row, leaf, configuration_id):
    """Only an ended, audited subset may be replanned in a different leaf."""
    provider = leaf/'provider_workspace/docs/research/distributed_selected_pf'/configuration_id
    try:
        response = json.loads((leaf/'RESPONSE.json').read_bytes())['items']
        item = next(item for item in response if item['configuration_id'] == configuration_id)
        prices = item.get('audited_independent_prices')
        if item['status'] != 'PRICE_EVIDENCE_REQUIRED' or prices is None:
            return None
        if any(provider.rglob('STOP*.json')):
            return None
        markers = (*provider.glob('*.DISTRIBUTED.json'), *provider.glob('*.STARTED.json'))
        stages = {marker.name.split('.')[0] for marker in markers}
        if prices and ('PF_AUDIT' not in stages or
                       ('PF_COMPILE' not in stages and not (provider/'SAVED_CALIBRATION_REUSE.json').is_file())):
            return None
        for marker in markers:
            stage = marker.name.split('.')[0]
            ended = json.loads((provider/(stage+'.RETURN.json')).read_bytes())
            if (ended['returncode'] != 0 or ended.get('signal') is not None or
                    (marker.name.endswith('.DISTRIBUTED.json') and
                     ended.get('job_id') != json.loads(marker.read_bytes())['job_id'])):
                return None
            if stage not in ('PF_COMPILE', 'PF_AUDIT', 'PF_SIGNALS'):
                return None
        if 'PF_SIGNALS' in stages:
            signal = json.loads((provider/'PF_SIGNALS.RESULT.json').read_bytes())
            if signal['status'] != 'SIGNALS_READY' or not signal.get('publication'):
                return None
        if prices:
            saved = json.loads((provider/'PF_AUDIT.RESULT.json').read_bytes())
            if saved['status'] != 'AUDIT_PASS' or saved['prices'] != prices:
                return None
        axes = {(p['physical_id'], p['axis']) for p in prices
                if p['gate'] == 'C_IMPLEMENTATION_GUARD_PASS'}
        if len(axes) != len(prices):
            return None
        return [list(axis) for axis in sorted(
            axes | {tuple(p) for p in row.get('previously_audited_price_axes', ())})]
    except (OSError, KeyError, StopIteration, TypeError, ValueError):
        return None


def live_price_owners(item):
    """Observe only preflight owners whose actual command still matches."""
    pending = {}
    findings = (hit for axes in item.get('deferred_price_rounds', {}).values()
                for axis in axes for hit in axis.get('findings', ()))
    for hit in findings:
        job = hit.get('running_job')
        if not job or type(job.get('pid')) is not int or job['pid'] <= 0:
            continue
        try:
            args = Path('/proc', str(job['pid']), 'cmdline').read_bytes().split(b'\0')
        except OSError:
            continue
        binding = job.get('job') or job.get('caller')
        if binding and (os.fsencode(binding) in args or
                        job.get('caller') and any(Path(os.fsdecode(a)).name == binding for a in args)):
            pending[job['pid']] = binding
    return sorted(pending.items())


def reconciliation_locators(waves, settings, row, *, audits=None):
    item = row.get('price_reconciliation', {})
    deferred = item.get('deferred_price_rounds', {})
    physical_ids = {axis['physical_id'] for axes in deferred.values() for axis in axes}
    physical_ids.update(axis[0] for axis in row.get('previously_audited_price_axes', ()))
    physical_ids.update(axis[0] for axis in item.get('missing_audited_axes', ()))
    references = [ref for axes in deferred.values() for axis in axes
                  for field in ('checked_receipts', 'unavailable_receipts')
                  for ref in axis.get(field, ())]
    for axes in deferred.values():
        for axis in axes:
            for finding in axis.get('findings', ()):
                if (origin := finding.get('original_compile')) is None:
                    continue
                raw = Path(origin['path']).parent
                # Arrival of a previously unreadable raw publication may
                # make the already-ended compile eligible for an audit.
                references.extend(dict(commit=origin['commit'], path=path.as_posix()) for path in (
                    raw.parent.parent/'PLAN.json', raw/'RESULT.json', raw/'PRICES.json',
                    raw/'SIDECARS_UNBOUND.json', raw/'IMPLEMENTATION_EVIDENCE.json'))
    return unique_locators([*price_indexes(waves, settings, physical_ids or None, audits=audits), *references])


def receipt_objects(root, paths):
    # An existing locator may become readable when the original object arrives.
    paths = list(dict.fromkeys(paths))
    if not paths:
        return {}
    result = subprocess.run(['git', 'cat-file', '--batch-check'], cwd=root,
                            env=dict(os.environ, GIT_NO_LAZY_FETCH='1'),
                            input=''.join(p+'\n' for p in paths), text=True,
                            capture_output=True, check=True).stdout
    return dict(zip(paths, result.splitlines(keepends=True), strict=True))


def reconciliation_snapshot(root, row, locators, *, objects=None, scoped=False):
    """Revisit final decisions when an applicable receipt or owner changes."""
    paths = [loc['commit']+':'+loc['path'] for loc in locators]
    objects = receipt_objects(root, paths) if objects is None else objects
    item = row.get('price_reconciliation', {})
    owner_paths = set(item.get('original_paths', ()))
    for directory in item.get('original_owner_directories', ()):
        base = Path(directory)
        markers = (path for pattern in ('*.json', '*/*/*.json', 'recovered_prices/*/*.json')
                   for path in base.glob(pattern))
        if scoped:
            owner_paths.update(str(path) for path in markers if (
                path.name.startswith(('PF_', 'STOP', 'COMPLETION'))
                and any(token in path.name for token in ('COMPLETION', 'RESULT', 'RETURN', 'STOP', 'DISTRIBUTED', 'STARTED'))))
        else:
            owner_paths.update(str(path) for path in markers if any(
                part in path.name for part in ('COMPLETION', 'RESULT', 'RETURN', 'STOP')))
    owner = []
    for raw in sorted(owner_paths):
        path = Path(raw)
        try:
            stat = path.stat()
            owner.append((raw, stat.st_size, stat.st_mtime_ns))
            if scoped and path.name.endswith('.DISTRIBUTED.json'):
                stage = path.name.split('.')[0]
                if stage in OWNER_STAGES:
                    mark = owner_document(path)
                    state, _ = distributed_owner_return(path.parent, mark, stage)
                    owner.append((raw, state))
        except (OSError, KeyError, TypeError, ValueError, sqlite3.Error):
            owner.append((raw, None, None))
    return hashlib.sha256(json.dumps([locators, ''.join(objects[path] for path in paths),
                                     owner, live_price_owners(item)],
                                     sort_keys=True).encode()).hexdigest()


def price_axis_actions(item):
    """Report the next operation and the new-computation decision separately."""
    actions = []
    audited = {(row['physical_id'], row['axis']) for row in item.get('audited_independent_prices', ())}
    raw_audit = (item.get('original_calibration') or {}).get('axes_by_round', {})
    for axes in item.get('deferred_price_rounds', {}).values():
        for axis in axes:
            findings = axis.get('findings', ())
            owner_states = {hit.get('original_owner_state') for hit in findings}
            if (axis['physical_id'], axis['axis']) in audited:
                action, reason = 'REUSE_AUDITED_PRICE', 'The central controller received a guarded price'
            elif 'AUDIT_ORIGINAL_COMPILE' in owner_states or axis['axis'] in raw_audit.get(str(axis['m']), ()):
                action, reason = 'AUDIT_ORIGINAL_COMPILE', 'Original compile matched the fixed plan'
            elif 'STOP' in owner_states:
                action, reason = 'NO_NEW_CALCULATION_UNEVALUATED', 'The original STOP or no-retry condition applies to this axis'
            elif owner_states - {None}:
                action, reason = 'RECONCILE_ORIGINAL_ATTEMPT', ', '.join(sorted(owner_states - {None}))
            elif any(hit.get('running_job') for hit in findings):
                action, reason = 'RECONCILE_ORIGINAL_ATTEMPT', 'Original owner appeared in the duplicate check'
            elif any(hit.get('prior_admitted_sources') for hit in findings):
                action, reason = 'CHECK_ADMITTED_RECEIPT', 'Exact receipt is available to the fixed price planner'
            elif axis['action'] == 'NEW_COMPILE_MISSING_CONFIRMED':
                action, reason = 'MISSING_AXIS', 'Duplicate check cleared this axis for the selected provider'
            elif any(hit.get('trial_files') or hit.get('existing_route_result') or hit.get('trial')
                     for hit in findings):
                action, reason = 'CHECK_ORIGINAL_COMPILE', 'Check the raw compile and bind it with the existing C guard'
            else:
                action, reason = 'LOCATE_ORIGINAL_RECEIPT', 'Catalog or aggregate evidence requires the original axis receipt'
            recovery = {key: axis[key] for key in ('checked_receipts', 'unavailable_receipts',
                                                   'checked_legacy_evidence') if key in axis}
            if recovery and action in ('CHECK_ORIGINAL_COMPILE', 'LOCATE_ORIGINAL_RECEIPT'):
                action = 'NO_NEW_CALCULATION_UNEVALUATED'
                reason = ('Registered evidence inspected without finding an admissible price '
                          'or proving that the fixed duplicate check permits a new compile')
            actions.append(dict(m=axis['m'], axis=axis['axis'], physical_id=axis['physical_id'],
                                action=action, reason=reason,
                                new_calculation=action == 'MISSING_AXIS', **recovery))
    return actions


def legacy_origins(value):
    """Read provenance links in legacy JSON evidence without treating it as a price row."""
    if isinstance(value, list):
        for item in value:
            yield from legacy_origins(item)
    elif isinstance(value, dict):
        commit, path = value.get('commit'), value.get('path')
        if isinstance(commit, str) and isinstance(path, str) and len(commit) == 40 and (
                all(c in '0123456789abcdef' for c in commit)) and (
                path and not Path(path).is_absolute() and '..' not in Path(path).parts):
            yield {key: value[key] for key in ('commit', 'path', 'selector', 'key', 'result_sha256')
                   if key in value}
        for name in ('source', 'carrier', 'receipt_trace', 'evidence', 'original_trial', 'receipt',
                     'receipts', 'references', 'proof', 'X', 'Y', 'cosine', 'sine'):
            if name in value:
                yield from legacy_origins(value[name])


def catalog_receipt_locators(root, rows):
    """Follow fixed original-result and admission references exposed by duplicate evidence."""
    @lru_cache(maxsize=16)
    def original(commit, path):
        try:
            raw = subprocess.check_output(['git', 'show', commit+':'+path], cwd=root,
                                          stderr=subprocess.DEVNULL,
                                          env=dict(os.environ, GIT_NO_LAZY_FETCH='1'))
            return json.loads(gzip.decompress(raw) if path.endswith('.gz') else raw)
        except (OSError, ValueError, subprocess.CalledProcessError):
            return None

    @lru_cache(maxsize=1)
    def m5_dataset():
        import csv
        import io

        # D's saved m3 provenance links this public, measured axis inventory.
        commit = '5764aacb0714c6be42cb6dc7c7d8fdf1c8d98bdb'
        path = ('artifacts/resource_applicability/track_a_h46_m5_det_cost_run01_results/'
                '2026-10-20/direct_measured_axes_92.csv')
        try:
            raw = subprocess.check_output(['git', 'show', commit+':'+path], cwd=root,
                                          stderr=subprocess.DEVNULL,
                                          env=dict(os.environ, GIT_NO_LAZY_FETCH='1'))
        except (OSError, subprocess.CalledProcessError):
            return {}, {}
        rows = csv.DictReader(io.StringIO(raw.decode()))
        return dict(commit=commit, path=path), {
            (r['candidate_id'], r['axis']): (i, r) for i, r in enumerate(rows)
        }

    def references(value):
        if isinstance(value, list):
            for item in value:
                yield from references(item)
        if not isinstance(value, dict):
            return
        if isinstance(value.get('path'), str) and isinstance(value.get('commit'), str):
            yield value
        for key in ('source', 'receipt', 'receipts', 'price_reference', 'price_evidence', 'references',
                    'PRICES.json', 'original_trial', 'origin', 'proof', 'receipt_trace', 'evidence',
                    'carrier', 'X', 'Y', 'cosine', 'sine'):
            if key in value:
                yield from references(value[key])

    def locate(ref, key=None, seen=frozenset()):
        path, commit = ref['path'], ref['commit']
        identity = (commit, path, json.dumps([key, ref.get('selector')], sort_keys=True))
        if identity in seen or len(seen) >= 8:
            return [], []
        parsed = original(commit, path)
        if parsed is None:
            return [], [dict(commit=commit, path=path)]
        if Path(path).name == 'PRICES.json':
            if 'selector' in ref:
                return [dict(commit=commit, path=path, selector=ref['selector'])], []
            entries = range(len(parsed)) if isinstance(parsed, list) else ()
            return [dict(commit=commit, path=path, selector=[i]) for i in entries], []
        try:
            if key is not None:
                parsed = parsed[key]
            for selector in ref.get('selector', ()):
                parsed = parsed[selector]
        except (KeyError, IndexError, TypeError):
            return [], []
        found = []
        missing = []
        # Guard admission below still checks the actual price and sidecar.
        for nested in [*references(parsed), *legacy_origins(parsed)]:
            if Path(nested['path']).name == 'PRICES.json' or (
                    nested.get('key') is not None or nested.get('selector')):
                refs, unavailable = locate(nested, nested.get('key'), seen | {identity})
                found.extend(refs)
                missing.extend(unavailable)
        return found, missing

    by_axis = {}
    producer_head = None
    for row in rows:
        key = row['physical_id'], row['axis']
        found, missing = [], []
        for hit in row['findings']:
            if hit.get('existing_route_result'):
                try:
                    relative = Path(hit['existing_route_result']).resolve().relative_to(Path(root).resolve())
                except ValueError:
                    relative = None
                if relative is not None:
                    if producer_head is None:
                        producer_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                            cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
                    refs, unavailable = locate(dict(commit=producer_head,
                                                   path=relative.with_name('PRICES.json').as_posix()))
                    found.extend(refs)
                    missing.extend(unavailable)
            if hit.get('key') is not None and isinstance(hit.get('source'), dict):
                refs, unavailable = locate(hit['source'], hit['key'])
                found.extend(refs)
                missing.extend(unavailable)
            for ref in references(hit.get('receipts', [])):
                refs, unavailable = locate(ref)
                found.extend(refs)
                missing.extend(unavailable)
            for ref in legacy_origins(hit.get('receipts', [])):
                if not Path(ref['path']).name.endswith(('.json', '.json.gz')):
                    continue
                refs, unavailable = locate(ref, ref.get('key'))
                found.extend(refs)
                missing.extend(unavailable)
        if row.get('findings'):
            evidence = []
            for hit in row['findings']:
                refs = list(legacy_origins(hit))
                for ref in refs:
                    value = original(ref['commit'], ref['path']) if ref['path'].endswith(('.json', '.json.gz')) else None
                    field = ref.get('key', hit.get('key') if ref == hit.get('source') else None)
                    if field is not None and isinstance(value, dict):
                        value = value.get(field)
                    if value is None:
                        continue
                    evidence.append(dict(ref, axis=row['axis'], status='ORIGIN_REQUIRES_C_ADMISSION'))
                    wrapper = {'X': 'cosine', 'Y': 'sine'}[row['axis']]
                    for proof in value.get('proof', ()) if isinstance(value, dict) else ():
                        if proof.get('source') != 'm5_dataset' or proof.get('axis') != wrapper:
                            continue
                        saved = proof['receipt']
                        catalog, index = m5_dataset()
                        match = index.get((saved['candidate_id'], wrapper))
                        if match is None:
                            continue
                        line, observed = match
                        if not (observed['cost_type'] == 'DIRECT_MEASURED_NEW_RUN01'
                                and observed['result_sha256'] == saved['result_sha256']
                                and observed['fingerprint'] == saved['fingerprint']
                                and observed['compiler_settings_hash'] == saved['compiler_settings_hash']
                                and observed['preparation_hash'] == saved['preparation_hash']
                                and observed['cost_receipt_sha256'] == saved['receipt_sha256']
                                and observed['rz_count'] == str(saved['rz_count'])):
                            continue
                        evidence.append(dict(catalog, axis=row['axis'], csv_data_row=line,
                                             status='ORIGINAL_AXIS_MATCH_REQUIRES_C_ADMISSION'))
                for path in hit.get('trial_files') or ([hit['trial']] if hit.get('trial') else []):
                    if Path(path).is_absolute():
                        evidence.append(dict(path=path, axis=row['axis'], status='ORIGINAL_HOST_RECEIPT'))
            if evidence:
                row['checked_legacy_evidence'] = unique_locators(evidence)
        by_axis[key] = (unique_locators(found), unique_locators(missing))
    return by_axis


def unique_locators(locators):
    unique = {json.dumps(loc, sort_keys=True): loc for loc in locators}
    return [unique[key] for key in sorted(unique)]


OWNER_STAGES = {
    'PF_SIGNALS': 'signal', 'PF_COMPILE': 'compile', 'PF_HOLDOUT_COMPILE': 'compile',
    'PF_AUDIT': 'audit', 'PF_HOLDOUT_AUDIT': 'audit', 'PF_MODEL_FREEZE': 'model',
    'PF_FINISH': 'model',
}
OWNER_MARKERS = ('.DISTRIBUTED.json', '.STARTED.json', '.STARTING.json', '.LAUNCH.json',
                 '.RETURN.json', '.ORIGINAL_TRANSPORT_ATTEMPT.json', '.LEGACY_UNSTARTED.json')


def owner_document(path):
    return json.loads(path.read_bytes()) if path.is_file() else None


@lru_cache(maxsize=16)
def accepted_signal_publications(run, revision):
    """Index the accepted signal heads of one fixed prior run."""
    ledger = run/'distributed/QUEUE.sqlite3'
    with closing(sqlite3.connect(ledger.as_uri()+'?mode=ro', uri=True, timeout=2)) as db:
        rows = db.execute("SELECT job_id, request FROM jobs WHERE method='PREPARATION:signal' "
                          "AND state='ACCEPTED' AND json_extract(request, '$.preparation.provider')='PF_SIGNALS'"
                          ).fetchall()
    publications = {}
    for job, request in rows:
        try:
            outcome = owner_document(run/'jobs/PREPARATION:signal'/job/'OUTCOME.json')
            if not outcome or outcome.get('status') != 'PREPARATION_COMPLETE':
                continue
            command = json.loads(request)['preparation']['command']
            owner = Path(command[command.index('--output')+1]).parent
            for bundle in outcome.get('git_bundles', ()):
                if bundle.get('workspace', 'workspace') == 'workspace':
                    publications[bundle['head']] = str(owner)
        except (OSError, KeyError, IndexError, TypeError, ValueError):
            continue
    return publications


def archived_signal_owner(waves, commit):
    """Find the owner of a legacy locator even if its Git object was pruned."""
    for wave in waves:
        run = wave.parent
        ledger = run/'distributed/QUEUE.sqlite3'
        try:
            info = ledger.stat()
            wal = ledger.with_name(ledger.name+'-wal')
            revision = (info.st_mtime_ns, wal.stat().st_mtime_ns if wal.is_file() else None)
            owner = accepted_signal_publications(run, revision).get(commit)
            if owner:
                return owner
        except (OSError, KeyError, IndexError, ValueError, sqlite3.Error):
            continue
    return None


def distributed_owner_return(directory, marker, stage):
    """Reconcile a submitted request against the owning queue in read-only mode."""
    job = marker['job_id']
    expected = deepcopy(marker['request'])
    if (expected.pop('request_id', job) != job or expected['preparation']['provider'] != stage
            or expected['preparation']['source']['commit'] != marker['source']):
        return 'UNKNOWN', None
    expected['preparation'].pop('dependencies', None)
    for base in directory.parents:
        ledger = base/'distributed/QUEUE.sqlite3'
        if not ledger.is_file():
            continue
        with closing(sqlite3.connect(ledger.resolve().as_uri()+'?mode=ro', uri=True, timeout=2)) as db:
            row = db.execute('SELECT request, request_sha256, method, state, attempt_id, completion_sha256 '
                             'FROM jobs WHERE job_id=?', (job,)).fetchone()
        if row is None:
            continue
        stored, sha, method, state, attempt, completion = row
        request = json.loads(stored)
        body = json.dumps(request, sort_keys=True, separators=(',', ':')).encode()
        if hashlib.sha256(body).hexdigest() != sha or request.pop('request_id', None) != job:
            return 'UNKNOWN', None
        request['preparation'].pop('dependencies', None)
        if request != expected:
            return 'UNKNOWN', None
        if state == 'CANCELLED' and attempt is None:
            return 'UNSTARTED', None
        if state != 'ACCEPTED' or not attempt or not completion:
            return 'UNKNOWN', None
        output = base/'jobs'/method/job
        outcome = owner_document(output/'OUTCOME.json')
        returned = owner_document(output/'PROCESS_RETURN.json')
        if (not outcome or outcome.get('request_sha256') != sha
                or outcome.get('request_id') != job or not returned):
            return 'UNKNOWN', None
        if outcome.get('status') == 'PREPARATION_FAILED':
            return 'STOP', None  # The original distributed contract forbids automatic retry.
        if (outcome.get('status') == 'PREPARATION_COMPLETE' and returned.get('returncode') == 0
                and returned.get('signal') is None):
            bundles = [bundle for bundle in outcome.get('git_bundles', ())
                       if bundle.get('workspace', 'workspace') == 'workspace'
                       and bundle['base'] == marker['source']]
            heads = [bundle['head'] for bundle in bundles]
            collected = owner_document(directory/(stage+'.RETURN.json'))
            if collected:
                if (collected.get('job_id') != job or collected.get('source') != marker['source']
                        or collected.get('returncode') != 0 or collected.get('signal') is not None
                        or set(collected.get('git_heads', ())) - set(heads)):
                    return 'UNKNOWN', None
                if (directory/(stage+'.RESULT.json')).is_file():
                    return 'ENDED', dict(collected, git_heads=heads, accepted_job=str(output))
            # A prior run may have received the immutable result before it
            # wrote the producer-local result. Read that exact cached output.
            controller_root = Path(expected['preparation']['controller_root'])
            command = expected['preparation']['command']
            output_path = Path(command[command.index('--output')+1]).relative_to(controller_root)
            sha = outcome.get('artifacts', {}).get('workspace/'+output_path.as_posix(), '')
            cached = None
            if len(sha) == 64 and all(ch in '0123456789abcdef' for ch in sha):
                path = base/'distributed/artifacts'/sha[:2]/sha
                if path.is_file():
                    raw = path.read_bytes()
                    if hashlib.sha256(raw).hexdigest() == sha:
                        cached = json.loads(raw)
            return 'ENDED', dict(job_id=job, source=marker['source'], cached_result=cached,
                                 git_heads=heads, accepted_job=str(output))
        return 'UNKNOWN', None
    return 'UNKNOWN', None


def owner_stage_axes(payload, stage, all_axes):
    kind = OWNER_STAGES[stage]
    if kind == 'signal':
        return set()
    if kind == 'model' or payload is None:
        return all_axes
    selected = payload if kind == 'compile' else payload['compiled']
    return {(m, axis) for m in selected['rounds']
            for axis in selected.get('axes_by_round', {}).get(str(m), ('X', 'Y'))} & all_axes


def owner_stop_axes(fields, all_axes):
    if 'rounds' in fields:
        all_axes = owner_stage_axes(fields, 'PF_COMPILE', all_axes)
    elif 'axes_by_round' in fields:
        all_axes = {(m, axis) for m, axis in all_axes if axis in fields['axes_by_round'].get(str(m), ())}
    return {(m, axis) for m, axis in all_axes
            if m == fields.get('m', m) and axis == fields.get('axis', axis)
            and ('axes' not in fields or axis in fields['axes'])}


def bound_owner_input(marker, path):
    preparation = marker['request']['preparation']
    command = preparation['command']
    original = Path(command[command.index('--input')+1]).relative_to(preparation['controller_root'])
    return preparation['inputs'].get('workspace/'+original.as_posix()) == hashlib.sha256(path.read_bytes()).hexdigest()


def scoped_original_owners(plan, directories):
    """Index scoped original attempts and their immutable publications.

    RESULT and STATUS are observations, not ownership. An unavailable or
    conflicting receipt never clears a started request. The native provider's
    own duplicate check runs separately against its legacy price sources.
    """
    output = dict(holds=[], signals=[], prices=[], compiles=[])
    cfg = plan['configuration']
    all_axes = {(row['m'], axis) for row in plan['prepared']['pending']['requests'] for axis in ('X', 'Y')}

    def hold(path, kind, axes, state):
        output['holds'].append(dict(path=str(path), stage=kind, axes=sorted(axes), state=state))

    for directory in sorted(set(map(Path, directories))):
        try:
            original = owner_document(directory/'PLAN.json')
            if original and original['configuration']['id'] != cfg['id']:
                continue
        except (OSError, KeyError, TypeError, ValueError):
            pass  # The stage markers still determine which work is protected.
        leaves = [directory, *sorted((directory/'recovered_prices').glob('*/'))]
        for leaf in leaves:
            for path in sorted({*leaf.glob('STOP*.json'), *leaf.glob('PF_*.STOP.json')}):
                try:
                    fields = owner_document(path)
                    if fields.get('configuration_id', cfg['id']) != cfg['id']:
                        continue
                    stage = fields.get('stage', fields.get('phase', path.name.split('.')[0]))
                    kind = OWNER_STAGES.get(stage, stage)
                    kind = kind if kind in OWNER_STAGES.values() else 'all'
                    keys = owner_stop_axes(fields, all_axes)
                except (OSError, AttributeError, TypeError, ValueError):
                    kind, keys = 'all', all_axes
                hold(path, kind, keys, 'STOP')
            for stage, kind in OWNER_STAGES.items():
                markers = [leaf/(stage+suffix) for suffix in OWNER_MARKERS if (leaf/(stage+suffix)).is_file()]
                if not markers:
                    continue
                axes, marker = all_axes, None
                try:
                    input_path = leaf/(stage+'.INPUT.json')
                    marker = owner_document(leaf/(stage+'.DISTRIBUTED.json'))
                    if marker is not None and not bound_owner_input(marker, input_path):
                        hold(markers[0], kind, all_axes, 'UNKNOWN')
                        continue
                    payload = owner_document(input_path)
                    if payload is not None and payload['value']['configuration']['id'] != cfg['id']:
                        continue
                    axes = owner_stage_axes(payload, stage, all_axes)
                    exact = (payload is not None and payload['value']['configuration'] == cfg
                             and payload['value']['prepared']['pending']['requests'] == plan['prepared']['pending']['requests'])
                    if marker is not None:
                        state, returned = distributed_owner_return(leaf, marker, stage)
                        for archived in markers:
                            if archived.suffixes[-2] not in ('.ORIGINAL_TRANSPORT_ATTEMPT', '.LEGACY_UNSTARTED'):
                                continue
                            old_state, _ = distributed_owner_return(leaf, owner_document(archived), stage)
                            if old_state == 'UNKNOWN':
                                hold(archived, kind, axes, old_state)
                        if returned and (returned.get('job_id') != marker['job_id']
                                         or returned.get('source') != marker['source']):
                            state = 'UNKNOWN'
                    else:
                        returned = owner_document(leaf/(stage+'.RETURN.json'))
                        state = ('ENDED' if returned and returned.get('returncode') == 0
                                 and returned.get('signal') is None else 'UNKNOWN')
                        if returned and returned.get('returncode') not in (0, None):
                            state = 'STOP'
                    if state == 'UNSTARTED':
                        continue
                    if state != 'ENDED' or not exact:
                        hold(markers[0], kind, axes, 'UNKNOWN' if state == 'ENDED' else state)
                        continue
                    result = (returned or {}).get('cached_result')
                    if result is None:
                        result = owner_document(leaf/(stage+'.RESULT.json'))
                    # A collected, ended stage must also point at its own
                    # immutable publication before we replace missing work.
                    heads = (returned or {}).get('git_heads', ())
                    if kind == 'signal' and result and result.get('status') == 'SIGNALS_READY' and result.get('publication') in heads:
                        output['signals'].append(dict(commit=result['publication'], owner=str(leaf)))
                    elif kind == 'compile' and result and result.get('status') == 'COMPILE_RECEIPTS_READY' and result.get('raw_commit') in heads:
                        if owner_stage_axes(result, stage, all_axes) != axes:
                            hold(markers[0], kind, axes, 'UNKNOWN')
                        else:
                            output['compiles'].append(result)
                    elif kind == 'audit' and result and result.get('status') == 'AUDIT_PASS' and all(
                            p['gate'] == 'C_IMPLEMENTATION_GUARD_PASS' and p['price_evidence']['source']['commit'] in heads
                            for p in result['prices']):
                        output['prices'].extend(p['price_evidence']['source'] for p in result['prices'])
                        hold(markers[0], kind, axes, 'WAITING_FOR_PRIOR_AUDIT_ADMISSION')
                    elif kind != 'model':
                        hold(markers[0], kind, axes, 'WAITING_FOR_RECEIPT')
                except (OSError, KeyError, IndexError, TypeError, ValueError, sqlite3.Error):
                    hold(markers[0], kind, all_axes if marker is None else axes, 'UNKNOWN')
        # Legacy native markers carry stage ownership even without a queue.
        for name, kind, modern in (('signal_run01', 'signal', 'PF_SIGNALS'),
                                   ('calibration/run01', 'compile', 'PF_COMPILE'),
                                   ('holdout/run01', 'compile', 'PF_HOLDOUT_COMPILE')):
            native = directory/name
            tracked = any((directory/(modern+suffix)).is_file() for suffix in OWNER_MARKERS)
            legacy_markers = () if tracked else (*native.glob('*STARTED*.json'), *native.glob('JOB.json'))
            for path in (*native.glob('STOP*.json'), *legacy_markers):
                try:
                    data = owner_document(path)
                    keys = {(c['m'], axis) for c in data['cells'] for axis in c.get('axes', ('X', 'Y'))}
                except (OSError, KeyError, TypeError, ValueError):
                    keys = all_axes
                hold(path, kind, keys & all_axes, 'STOP' if path.name.startswith('STOP') else 'UNKNOWN')
            status = native/'STATUS.json'
            if not tracked and kind == 'signal' and status.is_file():
                try:
                    active = owner_document(status).get('status') != 'COMPLETE_THIRTEEN_ROUNDS'
                except (OSError, ValueError, AttributeError):
                    active = True
                if active or plan.get('new_signal_means'):
                    hold(status, kind, set(), 'WAITING_FOR_RECEIPT')
    return output


def raw_compile_documents(root, plan, compiled):
    """Verify original compile completion and retain the four raw C inputs."""
    relative = Path(compiled['relative'])
    commit = compiled['raw_commit']
    if (relative.is_absolute() or '..' in relative.parts or relative.name != 'run01'
            or relative.parent.name != compiled['phase'] or len(commit) != 40
            or any(ch not in '0123456789abcdef' for ch in commit)):
        raise ValueError('Invalid original PF compile locator')

    def blob(path):
        return json.loads(subprocess.check_output(['git', 'show', commit+':'+path.as_posix()],
            cwd=root, stderr=subprocess.DEVNULL, env=dict(os.environ, GIT_NO_LAZY_FETCH='1')))

    old = blob(relative.parents[1]/'PLAN.json')
    result, prices, sides, proofs = (blob(relative/name) for name in
        ('RESULT.json', 'PRICES.json', 'SIDECARS_UNBOUND.json', 'IMPLEMENTATION_EVIDENCE.json'))
    axes = owner_stage_axes(compiled, 'PF_COMPILE', {
        (r['m'], a) for r in plan['prepared']['pending']['requests'] for a in ('X', 'Y')})
    wrappers = {(row['cell']['m'], {'cosine': 'X', 'sine': 'Y'}[axis])
                for row in result['results'] for axis in row['wrappers']}
    if (old['configuration'] != plan['configuration']
            or old['prepared']['pending']['requests'] != plan['prepared']['pending']['requests']
            or result['child_exit_code'] != 0 or result.get('child_signal') is not None
            or not all(r['ok'] for r in result['results']) or not axes or axes != wrappers
            or axes != {(row['m'], row['axis']) for row in prices} or len(prices) != len(axes)
            or len(sides) != len(prices)
            or any(price['axis'] != side['axis'] or price['physical_context'] != side['physical_context']
                   or price['source'] != side['price_source'] for price, side in zip(prices, sides, strict=True))):
        raise ValueError('Original compile does not match the fixed PF plan')
    return result, prices, sides, proofs


def reconcile_original_prices(root, plan, decisions, owners):
    """Add the registered ledger's authority to the native duplicate decisions."""
    rows = {(r['m'], r['axis']): r for group in decisions.values() for r in group}
    wanted = {(cell['m'], axis) for cell in plan['new_compile_cells'] for axis in cell['axes']}
    physical = {job['m']: job['physical_candidate']['id'] for job in plan['prepared']['pending']['requests']}

    def block(key, state, **evidence):
        row = rows.setdefault(key, dict(m=key[0], axis=key[1], physical_id=physical[key[0]], findings=[]))
        row.update(action='HELD', new_calculation=False)
        row['findings'].append(dict(original_owner_state=state, **evidence))

    held = set()
    for item in owners['holds']:
        if item['stage'] != 'signal' and item['stage'] != 'model':
            for key in wanted & {tuple(axis) for axis in item['axes']}:
                block(key, item['state'], original_owner=item['path'])
                held.add(key)
    matching = defaultdict(dict)
    for receipt in owners['compiles']:
        axes = owner_stage_axes(receipt, 'PF_COMPILE', wanted) - held
        if not axes:
            continue
        origin = dict(commit=receipt['raw_commit'], path=receipt['relative']+'/PRICES.json')
        try:
            _, price_rows, sides, proofs = raw_compile_documents(root, plan, receipt)
            proofs_by_axis = {}
            for row, side in zip(price_rows, sides, strict=True):
                check = proofs
                for selector in side['implementation_evidence'].get('selector', ()):
                    check = check[selector]
                proofs_by_axis[row['m'], row['axis']] = check
        except ValueError:
            # The original attempt is known to have ended. A complete but
            # incompatible receipt cannot supply the required price. The
            # original native duplicate gate still decides whether to compile.
            continue
        except (OSError, KeyError, IndexError, TypeError, subprocess.CalledProcessError):
            for key in axes:
                block(key, 'WAITING_FOR_RECEIPT', original_compile=origin)
                held.add(key)
            continue
        for row, side in zip(price_rows, sides, strict=True):
            key = row['m'], row['axis']
            if key not in axes:
                continue
            signature = json.dumps([row, side, proofs_by_axis[key]], sort_keys=True)
            identity = receipt['raw_commit'], receipt['relative']
            matching[key].setdefault(signature, (identity, receipt))
    audits = {}
    for key in sorted(matching.keys() - held):
        choices = matching[key]
        if len(choices) != 1:
            block(key, 'CONFLICTING_RECEIPTS')
            continue
        # The exact compile source now owns pricing for this axis. If a native
        # owner is still live, defer the audit until its request reconciles.
        if any(hit.get('running_job') for hit in rows.get(key, {}).get('findings', ())):
            continue
        identity, receipt = next(iter(choices.values()))
        audit = audits.setdefault(identity, dict(compiled=receipt, axes_by_round={}))
        audit['axes_by_round'].setdefault(str(key[0]), []).append(key[1])
        block(key, 'AUDIT_ORIGINAL_COMPILE', original_compile=dict(
            commit=identity[0], path=identity[1]+'/PRICES.json'))
    reduced = {r['m'] for r in rows.values() if r['action'] != 'NEW_COMPILE_MISSING_CONFIRMED'}
    return {m: [r for r in rows.values() if r['m'] == m] for m in sorted(reduced)}, list(audits.values())


def original_calibration(root, plan, deferred, relative):
    """Find a committed raw calibration for this exact plan, ready for C audit."""
    cells = [cell for cell in plan['new_compile_cells'] if cell['role'] != 'HOLDOUT']
    if not cells:
        return None
    relative = Path(relative)
    raw_folder = relative/'calibration/run01'
    candidates = {r['commit'] for rows in deferred.values() for row in rows
                  for r in row.get('checked_receipts', ())
                  if r['path'] == (raw_folder/'PRICES.json').as_posix()}
    expected = {str(cell['m']): cell.get('axes', ['X', 'Y']) for cell in cells}
    axes = {(int(m), axis) for m, names in expected.items() for axis in names}
    matches = {}
    for commit in sorted(candidates):
        def blob(path):
            return subprocess.check_output(['git', 'show', commit+':'+path.as_posix()],
                                           cwd=root, stderr=subprocess.DEVNULL,
                                           env=dict(os.environ, GIT_NO_LAZY_FETCH='1'))
        try:
            prior = json.loads(blob(relative/'PLAN.json'))
            result = json.loads(blob(raw_folder/'RESULT.json'))
            if (prior['configuration'] != plan['configuration'] or
                    prior['prepared']['pending']['requests'] != plan['prepared']['pending']['requests'] or
                    result['child_exit_code'] != 0 or result.get('child_signal') is not None):
                continue
            wrappers = {(row['cell']['m'], {'cosine': 'X', 'sine': 'Y'}[wrapper])
                        for row in result['results'] if row['ok'] for wrapper in row['wrappers']}
            if wrappers != axes or not all(row['ok'] for row in result['results']):
                continue
            raw = [blob(raw_folder/name) for name in
                   ('RESULT.json', 'PRICES.json', 'SIDECARS_UNBOUND.json', 'IMPLEMENTATION_EVIDENCE.json')]
            prices = json.loads(raw[1])
            if len(prices) != len(axes) or {(p['m'], p['axis']) for p in prices} != axes:
                continue
            signature = tuple(hashlib.sha256(data).hexdigest() for data in raw)
            original_method = plan['configuration']['spec']['split']['original_PF_method']
            matches[signature] = dict(status='COMPILE_RECEIPTS_READY', phase='calibration',
                rounds=[cell['m'] for cell in cells], raw_commit=commit, relative=raw_folder.as_posix(),
                route='TRUNCATED_PF' if original_method == 'B0I' else 'FULL_PF', axes_by_round=expected)
        except (OSError, KeyError, TypeError, ValueError, subprocess.CalledProcessError):
            continue
    return next(iter(matches.values())) if len(matches) == 1 else None


def audited_prices(root, locators, evidence):
    """Read the original, committed price row and its sibling guard report."""
    @lru_cache(maxsize=128)
    def document(commit, path):
        result = subprocess.run(['git', 'show', commit+':'+path], cwd=root,
                                env=dict(os.environ, GIT_NO_LAZY_FETCH='1'),
                                check=True, capture_output=True)
        return json.loads(result.stdout)

    def read(locator):
        row = document(locator['commit'], locator['path'])
        for key in locator.get('selector', ()):
            row = row[key]
        return row

    receipts = defaultdict(list)
    for loc in locators:
        try:
            row = read(loc)
            guard_loc = evidence.ref(loc['commit'], str(Path(loc['path']).with_name('GUARD_CHECK.json')))
            guard_document = read(guard_loc)
            guards = (guard_document if isinstance(guard_document, list)
                      else [guard_document.get('guard', guard_document)])
            axis, physical = row['axis'], row['physical_context']['physical_candidate']
            matches = []
            for n, guard in enumerate(guards):
                if (guard['status'] != 'IMPLEMENTATION_POLICY_CHECKED_NOT_SCIENTIFIC_QUALIFICATION'
                        or guard['policy']['id'] not in ('PF_SELECTED_SAME_PHYSICAL_WRAPPER_V1',
                                                        'C_SAVED_M3_METADATA_COMPATIBILITY_ONLY')):
                    continue
                for check in guard['checked']:
                    if check['m'] != row['m'] or check['axis'] != axis:
                        continue
                    side = read(check['evidence'])
                    if (side['physical_context'] == row['physical_context'] and side['axis'] == axis
                            and side['cohort'] == row['cohort'] and side['price_source'] == row['source']
                            and side['identity'] == row['identity'] and side['law_binding'] == row['law_binding']):
                        matches.append(n)
            if len(matches) == 1:
                admission = dict(guard_loc, selector=[matches[0]]) if isinstance(guard_document, list) else guard_loc
                receipts[evidence.physical_key(physical), axis].append(dict(value=row, source=loc, admission=admission))
        except (OSError, KeyError, IndexError, TypeError, ValueError, subprocess.CalledProcessError):
            continue
    return receipts


def matching_receipts(receipts, request):
    fields = ('physical_context', 'axis', 'measurement_axis', 'observable',
              'outcome_convention', 'quantity', 'units', 'P')
    profile = request['physical_context']['profile']
    return [hit for hit in receipts if all(hit['value'].get(k) == request[k] for k in fields)
            and all(hit['value'].get('price_profile', {}).get(k) == v for k,v in profile.items())
            and hit['value'].get('price_profile', {}).get('lowering_id') == 'H46_LONG_TIME_NATIVE_COMPOSE'
            and hit['value'].get('classification') == 'DIRECT_COMPILED_DETERMINISTIC'
            and hit['value'].get('expectation') == 'DETERMINISTIC']


def use_saved_prices(index):
    """Extend the pinned producer's price pool and leave its admission tests in use."""
    from trottertracks import pf_evidence_preparation as evidence

    data = json.loads(Path(index).read_bytes())
    extra = audited_prices(Path(data['root']), data['locators'], evidence)
    original = evidence.price_pool
    accept_direct = evidence.accepted_direct

    def accepted(repo, prepared):
        pool = dict(original(str(repo)))
        for request in evidence.I.price_requests(repo, prepared):
            key = evidence.physical_key(request['physical_context']['physical_candidate']), request['axis']
            if key in pool:
                continue
            exact = matching_receipts(extra.get(key, ()), request)
            if exact and all(hit['value']['mean'] == exact[0]['value']['mean'] for hit in exact):
                pool[key] = exact[0]
        # The pinned gate also checks compiler, wrapper, native lowering and class.
        evidence.price_pool = lambda _root: pool
        try:
            return accept_direct(repo, prepared)
        finally:
            evidence.price_pool = original

    evidence.accepted_direct = accepted
    return extra


def unresolved_rounds(root, plan, output, extra, *, discover=False):
    """Retain duplicate evidence and identify independent, whole-round work."""
    import run_pf_m0_prices as native
    from trottertracks import pf_evidence_preparation as evidence

    rounds = {row['m']: row.get('axes', ('X', 'Y')) for row in plan['new_compile_cells']}
    targets = [dict(configuration_id=plan['configuration']['id'], physical_id=job['physical_candidate']['id'],
                    m=job['m'], axis=axis, physical_candidate=job['physical_candidate'])
               for job in plan['prepared']['pending']['requests'] for axis in rounds.get(job['m'], ())]
    if not targets:
        return {}
    # The fixed legacy checker visits every other row for native JOB/RESULT
    # discovery. Supply its original X/Y pairs even when only one axis is needed.
    pairs = {row['m']: row for row in targets}
    wanted = {(row['m'], row['axis']) for row in targets}
    audit = native.duplicate_check((1, 2), selected=[
        dict(row, axis=axis) for row in pairs.values() for axis in ('X', 'Y')])
    audit['rows'] = [row for row in audit['rows'] if (row['m'], row['axis']) in wanted]
    requests = {(r['m'], r['axis']): r for r in evidence.I.price_requests(root, plan['prepared'])}
    discovered = catalog_receipt_locators(root, audit['rows']) if discover else {}
    located = audited_prices(root, unique_locators(loc for refs, _ in discovered.values()
                                                   for loc in refs), evidence)
    admitted = []
    for row in audit['rows']:
        key = evidence.physical_key(row['physical_candidate']), row['axis']
        refs, unavailable = discovered.get((row['physical_id'], row['axis']), ([], []))
        if discover and row['findings']:
                row.update(checked_receipts=refs, unavailable_receipts=unavailable)
        compatible = matching_receipts([*extra.get(key, ()), *located.get(key, ())],
                                       requests[row['m'], row['axis']])
        if compatible:
            row['action'] = 'HELD'
            row['findings'].append(dict(prior_admitted_sources=[hit['source'] for hit in compatible]))
            if discover and all(hit['value']['mean'] == compatible[0]['value']['mean'] for hit in compatible):
                admitted.extend(hit['source'] for hit in compatible)
                extra[key] = compatible
        row['new_calculation'] = row['action'] == 'NEW_COMPILE_MISSING_CONFIRMED'
    if discover:
        audit['discovered_admitted_receipts'] = unique_locators(admitted)
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')
    deferred = {row['m'] for row in audit['rows'] if row['action'] != 'NEW_COMPILE_MISSING_CONFIRMED'}
    return {m: [row for row in audit['rows'] if row['m'] == m] for m in sorted(deferred)}


def independent_plan(plan, deferred):
    selected = deepcopy(plan)
    blocked = {int(m) for m in deferred}
    if plan.get('selected_axis_compile'):
        held_axes = {(int(m), row['axis']) for m, rows in deferred.items() for row in rows
                     if row['action'] != 'NEW_COMPILE_MISSING_CONFIRMED'}
        for row in selected['new_compile_cells']:
            row['axes'] = [axis for axis in row['axes'] if (row['m'], axis) not in held_axes]
        selected['new_compile_cells'] = [row for row in selected['new_compile_cells'] if row['axes']]
    else:
        selected['new_compile_cells'] = [row for row in selected['new_compile_cells'] if row['m'] not in blocked]
    selected['maximum_new_compile'] = sum(len(row.get('axes', ('X', 'Y'))) for row in selected['new_compile_cells'])
    return selected
