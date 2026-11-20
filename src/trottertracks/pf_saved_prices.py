"""Reuse committed PF prices with their original C implementation admission."""
from collections import defaultdict
from copy import deepcopy
from functools import lru_cache
import json
import os
from pathlib import Path
import subprocess


def receipt_locators(waves):
    """Read only outputs of the existing audited PF stage in this search."""
    result = []
    for candidate_root in (waves/'candidates', waves/'price-recovery'/'candidates'):
        for path in sorted(candidate_root.glob(
                '*/pf_auto/pf_completion/provider_workspace/docs/research/distributed_selected_pf/*/*AUDIT.RESULT.json')):
            try:
                audit = json.loads(path.read_bytes())
                if audit['status'] == 'AUDIT_PASS':
                    result.extend(row['price_evidence']['source'] for row in audit['prices']
                                  if row['gate'] == 'C_IMPLEMENTATION_GUARD_PASS')
            except (OSError, KeyError, TypeError, ValueError):
                continue
    return list({json.dumps(loc, sort_keys=True): loc for loc in result}.values())


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
            fields = ('physical_context', 'axis', 'measurement_axis', 'observable',
                      'outcome_convention', 'quantity', 'units', 'P')
            exact = [hit for hit in extra.get(key, ()) if all(hit['value'].get(k) == request[k] for k in fields)]
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
    for row in audit['rows']:
        prior = extra.get((evidence.physical_key(row['physical_candidate']), row['axis']))
        if prior:
            row['action'] = 'HELD'
            row['findings'].append(dict(prior_admitted_sources=[hit['source'] for hit in prior]))
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')
    deferred = {row['m'] for row in audit['rows'] if row['action'] != 'NEW_COMPILE_MISSING_CONFIRMED'}
    return {m: [row for row in audit['rows'] if row['m'] == m] for m in sorted(deferred)}


def independent_plan(plan, deferred):
    selected = deepcopy(plan)
    blocked = {int(m) for m in deferred}
    selected['new_compile_cells'] = [row for row in plan['new_compile_cells'] if row['m'] not in blocked]
    selected['maximum_new_compile'] = 2*len(selected['new_compile_cells'])
    return selected
