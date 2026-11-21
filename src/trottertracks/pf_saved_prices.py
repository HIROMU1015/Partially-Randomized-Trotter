"""Reuse committed PF prices with their original C implementation admission."""
from collections import defaultdict
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import subprocess


def receipt_locators(waves, physical_ids=None, *, index=False):
    """Read only outputs of the existing audited PF stage in this search."""
    from itertools import chain

    if isinstance(waves, Path):
        waves = (waves,)
    result = []
    by_physical = defaultdict(list)
    for candidate_root in chain.from_iterable(map(candidate_roots, waves)):
        for path in sorted(candidate_root.glob(
                '*/pf_auto/pf_completion/provider_workspace/docs/research/distributed_selected_pf/*/*AUDIT.RESULT.json')):
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
        if prices and not {'PF_COMPILE', 'PF_AUDIT'} <= stages:
            return None
        for marker in markers:
            stage = marker.name.split('.')[0]
            ended = json.loads((provider/(stage+'.RETURN.json')).read_bytes())
            if (ended['returncode'] != 0 or ended.get('signal') is not None or
                    (marker.name.endswith('.DISTRIBUTED.json') and
                     ended.get('job_id') != json.loads(marker.read_bytes())['job_id'])):
                return None
            if stage not in ('PF_COMPILE', 'PF_AUDIT'):
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
    return price_indexes(waves, settings, physical_ids or None, audits=audits)


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


def reconciliation_snapshot(root, row, locators, *, objects=None):
    """Revisit final decisions when an applicable receipt or owner changes."""
    paths = [loc['commit']+':'+loc['path'] for loc in locators]
    objects = receipt_objects(root, paths) if objects is None else objects
    item = row.get('price_reconciliation', {})
    owner_paths = set(item.get('original_paths', ()))
    for directory in item.get('original_owner_directories', ()):
        base = Path(directory)
        owner_paths.update(str(path) for pattern in ('*.json', '*/*/*.json')
                           for path in base.glob(pattern)
                           if any(part in path.name for part in ('COMPLETION', 'RESULT', 'RETURN', 'STOP')))
    owner = []
    for raw in sorted(owner_paths):
        path = Path(raw)
        try:
            stat = path.stat()
            owner.append((raw, stat.st_size, stat.st_mtime_ns))
        except OSError:
            owner.append((raw, None, None))
    return hashlib.sha256(json.dumps([locators, ''.join(objects[path] for path in paths),
                                     owner, live_price_owners(item)],
                                     sort_keys=True).encode()).hexdigest()


def price_axis_actions(item):
    """Describe the next action using the fixed owner's duplicate evidence."""
    actions = []
    for axes in item.get('deferred_price_rounds', {}).values():
        for axis in axes:
            findings = axis.get('findings', ())
            if any(hit.get('running_job') for hit in findings):
                action, reason = 'RECONCILE_ORIGINAL_ATTEMPT', 'Original owner appeared in the duplicate check'
            elif any(hit.get('prior_admitted_sources') for hit in findings):
                action, reason = 'CHECK_ADMITTED_RECEIPT', 'Exact receipt is available to the fixed price planner'
            elif axis['action'] == 'NEW_COMPILE_MISSING_CONFIRMED':
                action, reason = 'MISSING_AXIS', 'This axis is clear but the fixed native worker requires a pair'
            elif any(hit.get('trial_files') or hit.get('existing_route_result') or hit.get('trial')
                     for hit in findings):
                action, reason = 'CHECK_ORIGINAL_COMPILE', 'Check the raw compile and bind it with the existing C guard'
            else:
                action, reason = 'LOCATE_ORIGINAL_RECEIPT', 'Catalog or aggregate evidence requires the original axis receipt'
            actions.append(dict(m=axis['m'], axis=axis['axis'], physical_id=axis['physical_id'],
                                action=action, reason=reason))
    return actions


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
            guard = read(guard_loc)['guard']
            axis, physical = row['axis'], row['physical_context']['physical_candidate']
            if (guard['status'] != 'IMPLEMENTATION_POLICY_CHECKED_NOT_SCIENTIFIC_QUALIFICATION'
                    or guard['policy']['id'] != 'PF_SELECTED_SAME_PHYSICAL_WRAPPER_V1'):
                continue
            check = next((item for item in guard['checked']
                          if item['m'] == row['m'] and item['axis'] == axis), None)
            if check is None:
                continue
            side = read(check['evidence'])
            if (side['physical_context'] != row['physical_context'] or side['axis'] != axis
                    or side['cohort'] != row['cohort'] or side['price_source'] != row['source']
                    or side['identity'] != row['identity'] or side['law_binding'] != row['law_binding']):
                continue
            receipts[evidence.physical_key(physical), axis].append(
                dict(value=row, source=loc, admission=guard_loc))
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


def unresolved_rounds(root, plan, output, extra):
    """Retain duplicate evidence and identify independent, whole-round work."""
    import run_pf_m0_prices as native
    from trottertracks import pf_evidence_preparation as evidence

    rounds = {row['m'] for row in plan['new_compile_cells']}
    targets = [dict(configuration_id=plan['configuration']['id'], physical_id=job['physical_candidate']['id'],
                    m=job['m'], axis=axis, physical_candidate=job['physical_candidate'])
               for job in plan['prepared']['pending']['requests'] for axis in ('X', 'Y') if job['m'] in rounds]
    if not targets:
        return {}
    audit = native.duplicate_check((1, 2), selected=targets)
    requests = {(r['m'], r['axis']): r for r in evidence.I.price_requests(root, plan['prepared'])}
    for row in audit['rows']:
        prior = extra.get((evidence.physical_key(row['physical_candidate']), row['axis']))
        compatible = matching_receipts(prior or (), requests[row['m'], row['axis']])
        if compatible:
            row['action'] = 'HELD'
            row['findings'].append(dict(prior_admitted_sources=[hit['source'] for hit in compatible]))
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')
    deferred = {row['m'] for row in audit['rows'] if row['action'] != 'NEW_COMPILE_MISSING_CONFIRMED'}
    return {m: [row for row in audit['rows'] if row['m'] == m] for m in sorted(deferred)}


def independent_plan(plan, deferred):
    selected = deepcopy(plan)
    blocked = {int(m) for m in deferred}
    selected['new_compile_cells'] = [row for row in plan['new_compile_cells'] if row['m'] not in blocked]
    selected['maximum_new_compile'] = 2*len(selected['new_compile_cells'])
    return selected
