"""Connection to A's PR-Pauli candidate preparation entry (metadata only).

A's entry (``scripts/prepare_pauli_candidate.py``, fixed SOURCE in the problem's ``generation.preparation`` block, run as a
subprocess on a pinned view) takes the same ``kind/items/base/changes`` request as B2's generator, plus items without a
saved base (``configuration: {L_D}``).  It returns, per item, either a complete configuration (``GENERATED``: id, body,
source, 13-round schedule, signal plan, D2 price hand-off) or a preparation request (``NEEDS_SPLIT_PREPARATION``: the
P0 split work still needed).  It never starts means, prices, compile or Q3.

Here:
  * candidates the search range holds without a configuration (new L_D) are sent to the entry BEFORE generate_variants
    narrows to configurations with an id;
  * a complete configuration is registered as a candidate even when its 26-axis price is missing (price, signal and Q3
    route stay separate needs; it is never searched until they exist);
  * an incomplete one stays as a preparation request with A's next actions;
  * kappa proposals of the search rule can be prepared the same way (base = the saved configuration of that L_D).
"""
import json
from pathlib import Path
import subprocess

PREP_SCHEMA = 'A_PAULI_CANDIDATE_PREPARATION_V1'


def check_view(prep):
    """The entry runs only from a view at the fixed SOURCE with untouched src/scripts (artificial entries excepted)."""
    if prep.get('kind') == 'artificial':
        return
    view = Path(prep['view'])
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=view, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--', 'src', 'scripts'], cwd=view, text=True).strip()
    if head != prep['view_commit'] or dirty:
        raise ValueError('PREPARATION_VIEW_NOT_AT_FIXED_SOURCE:%s:%s' % (head, 'dirty' if dirty else 'clean'))


def run_preparation(root, prep, items, out_dir, label):
    """One call of the entry; returns the parsed response (or a failure record) and where it was written."""
    out_dir = Path(out_dir).resolve()  # the entry runs with the view as cwd: every path handed over is absolute
    import hashlib
    # one directory per distinct item list: the entry refuses to overwrite PREPARATION.<i>.CONFIGURATION.json with another body
    tag = hashlib.sha256(json.dumps(items, sort_keys=True, default=str).encode()).hexdigest()[:12]
    out = out_dir / (label + '_' + tag);out.mkdir(parents=True, exist_ok=True)
    request = dict(kind=prep.get('request_kind', 'pauli_native'), view=prep.get('view'), view_commit=prep.get('view_commit'),
                   items=items, saved_packets=prep.get('saved_packets', []), saved_configurations=prep.get('saved_configurations', []))
    if prep.get('kind') == 'artificial':
        request['artificial_responses'] = prep.get('artificial_responses', {})
    (out / 'REQUEST.json').write_text(json.dumps(request, indent=1, ensure_ascii=False) + '\n')
    check_view(prep)
    if prep.get('kind') == 'artificial':
        entry, cwd = Path(root) / prep['entry'], Path(root)
    else:
        entry, cwd = Path(prep['view']) / prep.get('entry', 'scripts/prepare_pauli_candidate.py'), Path(prep['view'])
    cmd = [prep['interpreter'], '-B', str(entry), '--request', str(out / 'REQUEST.json'), '--output', str(out / 'PREPARATION.json'),
           '--cache', str(out_dir / 'SPLITS.json')]
    with (out / 'ENTRY.log').open('w') as log:
        code = subprocess.run(cmd, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT,
                              env=dict(__import__('os').environ, PYTHONDONTWRITEBYTECODE='1')).returncode
    if code != 0 or not (out / 'PREPARATION.json').exists():
        return dict(status='PREPARATION_CALL_FAILED', returncode=code, items=[], log=str(out / 'ENTRY.log'),
                    log_tail=(out / 'ENTRY.log').read_text()[-2000:]), out
    return json.loads((out / 'PREPARATION.json').read_text()), out


def summary(item, call_dir):
    plan = item.get('signal_plan') or {};price = item.get('price_handoff') or {}
    return dict(item_id=item.get('item_id'), status=item.get('status'), preparation_status=item.get('preparation_status'),
                L_D=item.get('L_D'), readiness=item.get('readiness'), next_actions=item.get('next_actions'),
                missing_inputs=item.get('missing_inputs'), configuration_id=item.get('configuration_id'),
                configuration_source=item.get('configuration_source'), configuration_file=item.get('configuration_file'),
                requested_changes=item.get('requested_changes'), descriptor_status=(item.get('descriptor') or {}).get('status'),
                dictionary_source=item.get('dictionary_source'), lambda_P=item.get('lambda_P'), lambda_DF=item.get('lambda_DF'),
                signal_counts=plan.get('counts'), signal_request_rounds=plan.get('request_rounds'),
                signal_unsupported_rounds=plan.get('unsupported_rounds'), signal_split_validation=plan.get('split_signal_validation'),
                price_handoff=dict(implementation_status=price.get('implementation_status'), needs=price.get('needs'),
                                   next_action=price.get('next_action'), quote_executed=price.get('quote_executed'),
                                   registry_entry=(price.get('evidence') or {}).get('registry_entry')) if price else None,
                scientific_request=item.get('scientific_request'), call=str(call_dir))


def needs_of(item):
    plan = item.get('signal_plan') or {};needs = []
    if item.get('configuration_source') is None:
        needs.append('CONFIGURATION_PUBLICATION')
    rounds = sorted(set(plan.get('request_rounds') or []) | set(plan.get('unsupported_rounds') or []))
    if rounds:
        needs.append('SIGNAL_ROUNDS:' + ','.join(map(str, rounds)))
    if len(rounds) < 13:
        needs.append('SIGNAL_PACKET_ASSEMBLY_FROM_REUSED_ROUNDS')
    needs.append('PRICE_AXES:pending_quote')  # replaced by the D2 PriceRules quote result
    needs.append('Q3_ROUTE_TEMPLATE_FOR_NEW_CONFIGURATION')
    return needs


def register(name, cand, item, candidates, call_dir, generated_from=None, key=None):
    """Complete configuration -> registered candidate (even without prices); otherwise ``cand`` stays a preparation request.
    Returns the disposition."""
    info = summary(item, call_dir)
    complete = (item.get('status') == 'GENERATED' and (item.get('readiness') or {}).get('configuration_complete')
                and item.get('configuration') is not None and item.get('configuration_id'))
    if not complete:
        if cand is not None:
            acts = item.get('next_actions') or []
            cand['needs'] = ['SPLIT_PREPARATION:' + ','.join(acts) if acts else 'PREPARATION_' + str(item.get('status'))]
            cand['needs'] += ['PREPARATION_INPUT_MISSING:' + str(m) for m in item.get('missing_inputs') or []]
            cand['ready'] = False;cand['preparation'] = info;cand['preparation_request'] = item.get('scientific_request')
        return 'PREPARATION_REQUEST'
    cid = item['configuration_id']
    from . import rpe_method_optimizer as O  # lazy: O imports this module
    config = item['configuration'];keys = O.schedule_keys(config) if config.get('rounds') else None
    kappa = str(O.dec_kappa(config)) if (config.get('spec') or {}).get('kappa') else None
    existing = next((c for c in candidates if c['configuration_id'] == cid), None)
    if existing is None and keys is not None:  # same physical schedule (kappa acts only through R): reuse its evaluations
        twin = next((c for c in candidates if c['method'] == name and c.get('schedule_keys') == keys), None)
        if twin is not None:
            twin.setdefault('physical_aliases', []).append(dict(configuration_id=cid, preparation=info, kappa_value=kappa))
            if cand is not None and cand in candidates and cand['configuration_id'] is None:
                candidates.remove(cand)
            return 'PHYSICALLY_IDENTICAL_TO:' + twin['configuration_id']
    if existing is not None:
        existing.setdefault('preparation_matches', []).append(info)
        if cand is not None and cand is not existing and cand in candidates and cand['configuration_id'] is None:
            candidates.remove(cand)
        return 'ALREADY_REGISTERED'
    ch = item.get('requested_changes') or {}
    k = dict(key or {});k.setdefault('L_D', int(item['L_D']) if item.get('L_D') is not None else None)
    for field in ('formula', 'delta_over_h', 'kappa_multiplier'):
        if field in ch:
            k[field] = str(ch[field])
    caps = cand['caps'] if cand is not None else next(c['caps'] for c in candidates if c['method'] == name)
    new = dict(method=name, family='PR', key=k, configuration_id=cid, index_signal=None, caps=caps, generated=True,
               generated_from=generated_from, labels=[], signal_source=None, prices=None, price_status='NOT_QUOTED_YET', saved_points=[],
               needs=needs_of(item), ready=False, generated_configuration=item['configuration'],
               configuration_source=item.get('configuration_source'), preparation=info, prepared_by='A_PAULI_CANDIDATE_PREPARATION',
               kappa_value=kappa, schedule_keys=keys)
    new['name'] = name + ':' + ','.join('%s=%s' % kv for kv in k.items()) + ':' + cid[:8]
    if cand is not None and cand in candidates:
        candidates[candidates.index(cand)] = new
    else:
        candidates.append(new)
    return 'REGISTERED_WITHOUT_PRICE_REQUIREMENT'


def prepare_unregistered(root, name, prep, candidates, baseline, out_dir):
    """Search-range candidates without a configuration -> A's entry (called before the id filter of generate_variants)."""
    todo = [c for c in candidates if c['method'] == name and c['configuration_id'] is None and c['family'] == 'PR']
    if not todo:
        return []
    changes = dict(formula=baseline.get('formula'), delta_over_h=baseline.get('delta_over_h'),
                   kappa_multiplier=baseline.get('kappa_multiplier'), K=prep.get('K', 2))
    items = [dict(item_id='%s:L_D=%s' % (name, c['key']['L_D']), configuration=dict(L_D=c['key']['L_D']), changes=changes) for c in todo]
    response, call = run_preparation(root, prep, items, out_dir, 'unregistered_L_D')
    rows = []
    by_id = {it.get('item_id'): it for it in response.get('items', [])}
    for c, req in zip(todo, items):
        item = by_id.get(req['item_id'])
        if item is None:
            item = dict(item_id=req['item_id'], status=response.get('status', 'NO_RESPONSE_ITEM'),
                        missing_inputs=response.get('missing_inputs') or [x for x in [response.get('log_tail', '')[-300:].strip()] if x])
        disposition = register(name, c, item, candidates, call)
        rows.append(dict(item_id=req['item_id'], L_D=c['key']['L_D'], status=item.get('status'), disposition=disposition,
                         configuration_id=item.get('configuration_id'), next_actions=item.get('next_actions'),
                         readiness=item.get('readiness'), missing_inputs=item.get('missing_inputs')))
    return rows


def _kappa(config):
    v = config['spec']['kappa']['value']
    from fractions import Fraction
    return Fraction(int(v['numerator']), int(v['denominator'])) if isinstance(v, dict) else Fraction(str(v))


def width_tag(fam):
    """Requested width of a family as given: delta_pf keeps its JSON representation (the string '1/5' is the exact rational,
    the number 0.2 is its binary64 value; they are different widths and never rounded into one another)."""
    if fam.get('delta_pf') is not None:
        return 'delta_pf=' + json.dumps(fam['delta_pf'])
    return 'delta_over_h=%s' % fam['delta_over_h']


def kappa_item(name, fam, base, mult, tag, K=2):
    """A-entry item of one kappa point: the requested width goes to A's mapper unchanged (q_m and T_m/q_m come back)."""
    changes = dict(formula=fam['formula'], kappa_multiplier=str(mult), K=K)
    if fam.get('delta_pf') is not None:
        changes['delta_pf'] = fam['delta_pf']
    else:
        changes['delta_over_h'] = fam['delta_over_h']
    wid = '' if fam.get('delta_pf') is None else ':' + width_tag(fam)
    return dict(item_id='%s:L_D=%s%s:%s' % (name, fam['L_D'], wid, tag), base=base['source'], changes=changes)


def prepare_kappa(root, name, prep, candidates, proposals, base_sources, out_dir, label='kappa_proposals'):
    """Kappa proposals of the search rule -> A's entry with the saved configuration of that L_D as base."""
    from fractions import Fraction
    items, keep = [], []
    for p in proposals:
        fam = p['family'];base = base_sources.get(fam['L_D'])
        if base is None:
            keep.append(dict(proposal=p, disposition='NO_SAVED_BASE_FOR_L_D'));continue
        if p.get('kappa') is not None:  # kappa itself: multiplier = target / the base's own spec.kappa.value, exactly
            target = Fraction(str(p['kappa']));mult = target / Fraction(str(base['kappa_value']))
            tag = 'kappa=%s' % target
        else:
            target, mult = None, Fraction(str(p['kappa_multiplier']));tag = 'kappa_x=%s' % mult
        items.append(kappa_item(name, fam, base, mult, tag, K=prep.get('K', 2)))
        keep.append(dict(proposal=p, base=base['configuration_id'], item_id=items[-1]['item_id'], kappa=None if target is None else str(target),
                         kappa_base=None if base.get('kappa_value') is None else str(base['kappa_value']), kappa_multiplier=str(mult)))
    if not items:
        return keep
    response, call = run_preparation(root, prep, items, out_dir, label)
    by_id = {it.get('item_id'): it for it in response.get('items', [])}
    for row in keep:
        if 'item_id' not in row:
            continue
        item = by_id.get(row['item_id']) or dict(item_id=row['item_id'], status=response.get('status', 'NO_RESPONSE_ITEM'),
                                                 missing_inputs=response.get('missing_inputs') or [])
        fam = row['proposal']['family']
        key = dict(L_D=fam['L_D'], formula=fam['formula'], delta_over_h=fam.get('delta_over_h'))
        if fam.get('delta_pf') is not None:
            key['delta_pf'] = fam['delta_pf']
        if row['kappa'] is not None:
            key['kappa'] = row['kappa']
            got = item.get('configuration') and Fraction(str(_kappa(item['configuration'])))
            row['kappa_exact_check'] = dict(target=row['kappa'], generated=None if got is None else str(got), equal=got == Fraction(row['kappa']))
            if item.get('status') == 'GENERATED' and got != Fraction(row['kappa']):
                row.update(disposition='KAPPA_VALUE_MISMATCH', status=item.get('status'), configuration_id=item.get('configuration_id'));continue
        else:
            key['kappa_multiplier'] = row['proposal']['kappa_multiplier']
        row['disposition'] = register(name, None, item, candidates, call, generated_from=row['base'], key=key)
        reg = next((c for c in candidates if c['configuration_id'] == item.get('configuration_id')), None)
        if reg is not None and row['kappa'] is not None:
            reg.update(kappa_base=row['kappa_base'], kappa_multiplier_exact=row['kappa_multiplier'])
        row.update(status=item.get('status'), configuration_id=item.get('configuration_id'), readiness=item.get('readiness'),
                   signal_counts=(item.get('signal_plan') or {}).get('counts'), next_actions=item.get('next_actions'))
    return keep
