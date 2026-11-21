"""Deterministic PF evidence -> the existing optimizer (metadata and saved I/O).

The registry publishes configuration, signal, exact per-axis prices and a Q3
template.  It does not choose shots, fit prices, compute signals, or certify
unobserved prices.  A normal optimizer round may explicitly request its finite
``selected`` candidates from D's existing completion entry before preparation.
"""
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys

from .saved_rpe_cost_catalog import frac

SCHEMA = 'D_PF_EVIDENCE_REGISTRY_V1'
METHODS = ('FULL_PF', 'TRUNCATED_PF')
AXES = ('X', 'Y')


def read_source(root, reader, source):
    if isinstance(source, dict) and 'commit' in source and 'path' in source:
        return reader.read(source)
    if isinstance(source, (str, Path)):
        path = Path(source)
        return json.loads((path if path.is_absolute() else Path(root) / path).read_text())
    return deepcopy(source)


def load_registry(root, reader, source=None, *, allow_artificial=False):
    registry = read_source(root, reader, source) if source is not None else dict(schema=SCHEMA, configurations={})
    if registry.get('schema') != SCHEMA or not isinstance(registry.get('configurations'), dict):
        raise ValueError('Unsupported deterministic PF evidence registry')
    if registry.get('artificial') is True and not allow_artificial:
        raise ValueError('Artificial PF evidence requires an explicit artificial fixture context')
    return registry


def physical_ids(root, config):
    """Existing B/D physical ID rules; no preparation, signal or circuit call."""
    from .baseline_signal_pipeline import published_api
    api = published_api(str(root));B, D = api.B, api.D
    B._validate(config)
    method = B.comparison_method(config['spec'])
    if method == 'FULL_PF':
        physical = D.physical_schedule(config['spec'])
        D._check_against_b(physical, config)
        return {r['m']: r['physical_candidate']['id'] for r in physical['rounds']}
    if method != 'TRUNCATED_PF' or config['spec']['split']['original_PF_method'] != 'B0I':
        raise ValueError('PF registry supports Full PF or B0I deterministic PF only')
    s = config['spec'];result = {}
    for r in config['rounds']:
        T = float(B.a.exact_scalar(r['T']))
        c = dict(system=s['binding']['system'], method='B0I', prefix=s['L_D'], formula=s['formula'],
                 T=T, q=r['q_PF'], delta=T / r['q_PF'], R_outer=None, K=None)
        result[r['m']] = B.p.candidate_id(c)
    return result


def configuration_metadata(config):
    s = config['spec'];split = s.get('split') or {}
    method = 'TRUNCATED_PF' if split.get('original_PF_method') == 'B0I' else 'FULL_PF'
    p = s['L_D'] if method == 'TRUNCATED_PF' else s['binding']['rank']
    return dict(method=method, p=p, formula=s['formula'], delta_over_h=frac(s['delta_pf']) / frac(s['h']))


def width_values(spec):
    """Return an explicit finite width grid, or None for the legacy interval."""
    values = spec.get('delta_over_h_values')
    if values is None:
        return None
    if not isinstance(values, list) or not values:
        raise ValueError('PF delta_over_h_values must be a nonempty exact width list')
    if any(isinstance(value, bool) for value in values):
        raise ValueError('PF discrete widths must be positive exact values')
    widths = tuple(dict.fromkeys(frac(value) for value in values))
    if any(width <= 0 for width in widths):
        raise ValueError('PF discrete widths must be positive exact values')
    return widths


def width_inside(value, spec):
    widths = width_values(spec)
    if widths is not None:
        return frac(value) in widths
    return Fraction(spec['delta_over_h_range'][0]) <= frac(value) <= Fraction(spec['delta_over_h_range'][1])


def _inside(entry, spec):
    return (entry['formula'] in spec['formulas']
            and width_inside(entry['delta_over_h'], spec)
            and (spec.get('p_values') is None or entry['p'] in spec['p_values']))


def enumerate_prefixes(root, reader, problem, candidates, methods=None):
    """Expose requested p as missing candidates before any acquisition choice.

    Formula/width choices come from the existing fixed PF index and method
    scope.  Only canonical configuration metadata and saved T1/T2 priority
    evidence are generated; no scientific input is completed by enumeration.
    A registry can then hydrate any selected subset.  p10 is represented by the
    existing Full PF method, never a second truncated-PF candidate.
    """
    scoped=[(m,s)for m,s in problem['methods'].items()if s.get('family')=='PF'
            and (s.get('p_values') is not None or s.get('delta_over_h_values') is not None)
            and (methods is None or m in methods)]
    if not scoped:
        return candidates
    from . import pf_candidate_space as P
    index=reader.document(problem['evidence']['pf_candidate_index']['commit'],
                          problem['evidence']['pf_candidate_index']['path'])
    by_id={c['configuration_id']:c for c in candidates}
    for method,spec in scoped:
        values=spec.get('p_values')
        if values is None:
            values=[problem['problem']['hamiltonian_data']['expected']['fragment_count']] if method=='FULL_PF' else [7]
        if (len(set(values))!=len(values) or any(type(p)is not int or p not in range(11)for p in values)
                or (method=='FULL_PF' and any(p!=10 for p in values))):
            raise ValueError('PF p_values must be unique legal prefixes; Full PF is p10')
        explicit = width_values(spec)
        widths={f:list(explicit or ()) for f in spec['formulas']}
        if explicit is None:
            for row in index['candidates']:
                width=row.get('requested_delta_over_h');formula=row['formula']
                if (row['method']==spec['index_method'] and formula in widths and width is not None
                        and width_inside(width, spec)):
                    if frac(width) not in widths[formula]:
                        widths[formula].append(frac(width))
        for meta in P.candidates(root,formulas=spec['formulas'],widths=widths,p_values=values):
            # P's p10 canonical alias is already enumerated under Full PF.
            if meta['method']!=method:
                continue
            cid=meta['configuration_id']
            if spec.get('configuration_ids') is not None and cid not in spec['configuration_ids']:
                continue
            c=by_id.get(cid)
            if c is None:
                c=dict(method=method,family='PF',configuration_id=cid,caps=spec['caps'],labels=[],saved_points=[],
                       prices=None,signal_source=None,index_signal=None,price_status='NO_SAVED_26_AXIS_PRICE',
                       needs=['SIGNAL_13ROUND_PACKET','PRICE_26AXIS','Q3_ROUTE_TEMPLATE_FOR_NEW_CONFIGURATION'],
                       ready=False,outside_index=True)
                candidates.append(c);by_id[cid]=c
            elif c['method']!=method:
                raise ValueError('PF prefix metadata has a conflicting method/configuration identity')
            c['key']=dict(p=meta['p'],formula=meta['formula'],delta_over_h=str(frac(meta['delta_over_h'])))
            c['name']=method+':'+','.join('%s=%s'%kv for kv in c['key'].items())+':'+cid[:8]
            c['configuration_body']=deepcopy(meta['configuration'])
            c['priority_evidence']=deepcopy(meta['priority_evidence'])
            c['p10_alias_full_pf']=meta.get('p10_alias_full_pf',False)
    return candidates


def hydrate(root, reader, problem, candidates, registry, facts=None, *, allow_artificial=False):
    """Join only each configuration's own saved signal and axis prices.

    Partial entries stay explicit missing inputs.  A different prefix, physical
    schedule, axis, or canonical configuration cannot borrow another price.
    Existing catalog candidates are hydrated too; no ``generated`` exception.
    """
    if registry.get('artificial') is True and not allow_artificial:
        raise ValueError('Artificial PF evidence cannot enter a saved scientific catalog')
    expected = {}
    for cid, entry in registry['configurations'].items():
        method = entry['method']
        if method not in METHODS:
            raise ValueError('Non-PF method in deterministic PF registry')
        rank = (facts or {}).get('fragment_count', 10)
        if type(entry['p']) is not int or not 0 <= entry['p'] <= rank or (entry['p'] == rank) != (method == 'FULL_PF'):
            raise ValueError('Full PF must use the saved rank; B0I must retain a strict prefix')
        if method not in problem['methods']:
            continue
        spec = problem['methods'][method]
        if spec['family'] != 'PF' or not _inside(entry, spec):
            continue
        allowed = spec.get('configuration_ids')
        if allowed is not None and cid not in allowed and entry.get('horizon_source_configuration_id') not in allowed:
            continue
        config = read_source(root, reader, entry['configuration'])
        if (config.get('id') or config.get('configuration_id')) != cid:
            raise ValueError('PF registry configuration ID mismatch')
        meta = configuration_metadata(config)
        if any(meta[k] != (frac(entry[k]) if k == 'delta_over_h' else entry[k]) for k in meta):
            raise ValueError('PF registry prefix/formula/width differs from configuration')
        horizon = config['spec'].get('M', 12)
        extension = entry.get('horizon_risk_source')
        parent_id = entry.get('horizon_source_configuration_id')
        if horizon == 13:
            if problem.get('search', {}).get('rule') != 'RPE_SCALED_SHOTS_V1':
                continue
            parent = entry.get('horizon_parent') or {}
            if not extension or not parent_id or set(parent) != {'configuration', 'signal_packet'}:
                raise ValueError('Extended PF needs its fixed evaluation SOURCE and saved parent')
            original = read_source(root, reader, parent['configuration'])
            old_packet = read_source(root, reader, parent['signal_packet'])
            if (original['id'] != parent_id or old_packet['configuration_id'] != parent_id
                    or dict(config['spec'], M=12) != original['spec']):
                raise ValueError('Extended PF changes the scoped parent configuration')
        elif horizon != 12 or parent_id is not None or extension is not None:
            raise ValueError('PF registry requires a fixed M=12 or M=13 configuration')
        count = horizon + 1
        rounds = {r['m']: r for r in config['rounds']}
        if len(config['rounds']) != count or set(rounds) != set(range(count)):
            raise ValueError('PF registry needs the complete canonical schedule')
        signal_need, price_need = f'SIGNAL_{count}ROUND_PACKET', f'PRICE_{2*count}AXIS'
        if facts:
            for key, fk in [('snapshot', 'snapshot_sha256'), ('hamiltonian', 'hamiltonian_hash'), ('state', 'state_hash')]:
                if config['spec']['binding'][key] != facts[fk]:
                    raise ValueError('PF registry physical input mismatch: ' + key)
        hits = [c for c in candidates if c['configuration_id'] == cid]
        if len(hits) > 1 or (hits and hits[0]['method'] != method):
            raise ValueError('PF registry configuration belongs to a different candidate')
        if hits:
            c = hits[0]
        else:
            c = dict(method=method, family='PF', configuration_id=cid, caps=spec['caps'], labels=[], saved_points=[],
                     prices=None, signal_source=None, needs=[signal_need, price_need],
                     price_status=f'NO_SAVED_{2*count}_AXIS_PRICE', outside_index=True)
            candidates.append(c)
        c['key'] = dict(p=entry['p'], formula=entry['formula'], delta_over_h=str(frac(entry['delta_over_h'])))
        c['name'] = method + ':' + ','.join('%s=%s' % kv for kv in c['key'].items()) + ':' + cid[:8]
        c['pf_evidence'] = deepcopy(entry)
        c['configuration_source'] = deepcopy(entry['configuration'])
        if horizon == 13:
            c.update(M=horizon, source_configuration_id=parent_id)
        ids = None
        if entry.get('signal_packet'):
            packet = read_source(root, reader, entry['signal_packet'])
            means = packet.get('means') or []
            if packet.get('configuration_id') != cid or len(means) != count or {m['m'] for m in means} != set(rounds):
                raise ValueError('PF registry signal packet/configuration/round mismatch')
            ids = physical_ids(root, config)
            for row in means:
                r = rounds[row['m']]
                if row.get('identity') != r['identity'] or row.get('law_binding') != r['law_binding']:
                    raise ValueError('PF registry signal identity/law mismatch')
                if row.get('physical_candidate', {}).get('id') != ids[row['m']]:
                    raise ValueError('PF registry signal physical schedule mismatch')
                if row.get('status') != 'SIGNAL_AVAILABLE_NOT_RMSE_QUALIFIED':
                    raise ValueError('PF registry signal is not an available saved mean')
            c['signal_source'] = deepcopy(entry['signal_packet'])
            c['needs'] = [n for n in c['needs'] if n != signal_need]
        if entry.get('prices'):
            table = read_source(root, reader, entry['prices']);rows = table['axes']
            ids = ids or physical_ids(root, config)
            units = {}
            for row in rows:
                key = (row['m'], row['axis'])
                if key in units or key[0] not in rounds or key[1] not in AXES or row['physical_id'] != ids[key[0]]:
                    raise ValueError('PF registry price axis/physical schedule mismatch')
                if row.get('configuration_id', cid) != cid or not row.get('kind') or not row.get('price_evidence'):
                    raise ValueError('PF registry price lacks own configuration/provenance')
                if row['kind'].startswith('STRUCTURAL'):
                    raise ValueError('Structural price is not compiled RZ price')
                units[key] = frac(row['unit_price_RZ'])
            keys = [(m, a) for m in sorted(rounds) for a in AXES]
            missing = [dict(m=m, axis=a) for m, a in keys if (m, a) not in units]
            partial = table.get('coverage') == 'PARTIAL_DIRECT_PRICE_EVIDENCE'
            if any(v <= 0 for v in units.values()) or (missing and not partial):
                raise ValueError('PF registry needs positive own-axis prices and explicit partial coverage')
            if c['prices'] is not None and any(units[key] != c['prices'][i]
                    for i, key in enumerate(keys) if key in units):
                raise ValueError('PF registry conflicts with existing accepted unit prices')
            if partial and missing and c['prices'] is None:
                c.update(price_status='SOURCE_BOUND_PF_PARTIAL_AXIS_PRICES', price_axes=deepcopy(rows),
                         price_source=deepcopy(entry['prices']), price_needs=missing)
                c['needs'] = list(dict.fromkeys([*c['needs'], price_need]))
            elif not missing:
                c.update(prices=[units[key] for key in keys], price_status=f'SOURCE_BOUND_PF_{2*count}_AXIS_PRICES',
                         price_scope=entry.get('price_scope'), price_axes=deepcopy(rows),
                         price_source=deepcopy(entry['prices']))
                c['needs'] = [n for n in c['needs'] if n != price_need]
                c.pop('price_needs', None)
        if entry.get('q3_inputs'):
            inputs = read_source(root, reader, entry['q3_inputs'])
            cases = [r for r in inputs['cases'] if (r.get('configuration_id') or r.get('original_configuration_id')) == cid]
            if len(cases) != 1:
                raise ValueError('PF registry needs one matching Q3 template case')
            case = cases[0]
            if horizon == 13 and len(inputs['options']['phase_radii']) != count:
                raise ValueError('Extended PF Q3 template needs 14 fixed phase radii')
            for key in ('total_configuration_id', 'binding_id'):
                # Existing PF cases call the same canonical total ID
                # ``total_view_id``; the executor uses total_configuration_id.
                actual = case.get(key) or (case.get('total_view_id') if key == 'total_configuration_id' else None)
                if not entry.get(key) or actual != entry[key]:
                    raise ValueError('PF registry Q3 template binding mismatch: ' + key)
            if case.get('signal_packet') != entry.get('signal_packet') or case.get('configuration') != entry['configuration']:
                raise ValueError('PF registry Q3 template sources differ from published candidate')
            c['needs'] = [n for n in c['needs'] if n != 'Q3_ROUTE_TEMPLATE_FOR_NEW_CONFIGURATION']
        elif not c['saved_points'] and 'Q3_ROUTE_TEMPLATE_FOR_NEW_CONFIGURATION' not in c['needs']:
            c['needs'].append('Q3_ROUTE_TEMPLATE_FOR_NEW_CONFIGURATION')
        c['ready'] = not c['needs'] and c['prices'] is not None and c['signal_source'] is not None
        expected[cid] = {k: entry.get(k) for k in ('total_configuration_id', 'binding_id')}
        expected[cid]['signal_source'] = c['signal_source']
        if horizon == 13:
            expected[cid]['M'] = horizon
    return expected


def registry_routes(routes, registry):
    """Use the existing pf_b_pf6 saved-template route; no new worker pipeline."""
    routes = deepcopy(routes)
    for cid, entry in registry['configurations'].items():
        if not entry.get('q3_inputs'):
            continue
        method_rules=routes.get(entry['method'],[])
        if registry.get('artificial') is True:
            if method_rules and all(r.get('kind')=='artificial'for r in method_rules):
                continue  # fixture worker uses hydrate's expected IDs, not PF templates
            raise ValueError('Artificial PF registry needs artificial-only method routes')
        rules = [r for r in method_rules if r.get('kind') == 'pf_b_pf6']
        if not rules:
            raise ValueError('Published PF candidate has no existing pf_b_pf6 route')
        base_rules = [rule for rule in rules if not rule.get('horizon_risk_source')]
        if not base_rules:
            raise ValueError('Published PF candidate has no base route')
        rule = next((r for r in base_rules if r.get('configuration_ids') in (None, '*')
                     or cid in r['configuration_ids']), base_rules[0])
        loc = deepcopy(entry['q3_inputs'])
        if loc.get('selector'):
            loc['inputs_selector'] = loc.pop('selector')
        if entry.get('horizon_risk_source'):
            if any(r.get('horizon_risk_source') and r.get('configuration_ids') == [cid] for r in rules):
                continue
            method_rules.insert(0, dict(rule, templates=[loc], configurations={}, configuration_ids=[cid],
                horizon_risk_source=deepcopy(entry['horizon_risk_source']), horizon_parent=deepcopy(entry['horizon_parent'])))
            rule = method_rules[0]
        if loc not in rule.setdefault('templates', []):
            rule['templates'].append(loc)
        rule.setdefault('configurations', {})[cid] = {k: deepcopy(entry[k]) for k in
                                                     ('configuration', 'signal_packet', 'total_configuration_id', 'binding_id')}
        if isinstance(rule.get('configuration_ids'), list) and cid not in rule['configuration_ids']:
            rule['configuration_ids'].append(cid)
    return routes


class CompletionRequestMismatch(ValueError):
    """The candidate's workspace is already bound to a different request."""

    def __init__(self, path, saved, requested):
        super().__init__('PF completion directory belongs to another selected request')
        self.path = path
        self.fields = sorted(key for key in saved.keys() | requested.keys()
                             if key not in saved or key not in requested or saved[key] != requested[key])


def inherited_completion_request(provider_entry, run, saved, requested, prior_runs):
    """Continue an already selected provider workspace with its frozen paths."""
    metadata = {'registry_file', 'output_root', 'recover_saved_price_rounds'}
    if ({k: v for k, v in saved.items() if k not in metadata} !=
            {k: v for k, v in requested.items() if k not in metadata}):
        return False
    marker = run/'PROVIDER_SOURCE.json'
    local_request = run/'provider_workspace/docs/research/distributed_selected_pf/REQUEST.json'
    if (len(saved.get('selected', ())) != 1 or not marker.is_file() or not local_request.is_file()
            or not all((run/name).is_file() for name in ('PLAN_RESPONSE.json',
                                                        'ORIGINAL_OWNER_REQUEST.json', 'ORIGINAL_OWNER_CHECK.json'))):
        return False
    provider = json.loads(marker.read_text())
    if Path(provider['root']).resolve() != provider_entry.resolve().parents[1]:
        return False
    workspace_request = json.loads(local_request.read_text())
    if workspace_request.get('selected') != saved['selected']:
        return False
    [candidate] = saved['selected']
    cid = candidate['configuration_body']['id']
    [item] = json.loads((run/'PLAN_RESPONSE.json').read_text())['items']
    if item.get('configuration_id') != cid:
        return False
    producer_path = Path('docs/research/distributed_selected_pf')/cid
    # Both saved paths must name the same selected leaf of a declared prior
    # run. An earlier request's recovery options never change its saved plan.
    for prior in prior_runs:
        leaf = Path(prior).resolve()/'pf_auto/candidates'/cid/'pf_auto'
        original = leaf/'pf_completion/provider_workspace'/producer_path
        local_plan = run/'provider_workspace'/producer_path/'PLAN.json'
        if (Path(saved['output_root']) == leaf/'provider_output' and
                Path(saved['registry_file']) == leaf/'REGISTRY.json' and
                item.get('output') == str(original) and local_plan.is_file()
                and (original/'PLAN.json').is_file()
                and local_plan.read_bytes() == (original/'PLAN.json').read_bytes()):
            return True
    return False


def complete_selected(root, spec, run_dir, *, stage_executor=None):
    """One explicitly selected finite provider call before normal O.prepare.

    A saved response resumes without invoking the provider again.  Missing or
    UNKNOWN results stay visible; this hook never retries science implicitly.
    """
    completion = spec.get('pf_completion')
    if not completion:
        return None
    selected = completion.get('selected')
    mode=completion.get('mode','complete')
    if mode not in ('plan','complete'):
        raise ValueError('PF completion mode must be plan or complete')
    if not selected or (mode=='complete' and spec.get('science_approved') is not True):
        raise ValueError('PF completion needs selected candidates; complete also requires science_approved')
    if any(m not in METHODS for m in spec['methods']):
        raise ValueError('Deterministic PF completion cannot request a PR method')
    root = Path(root);run = Path(run_dir) / 'pf_completion';run.mkdir(parents=True, exist_ok=True)
    request = {k: deepcopy(v) for k, v in completion.items() if k not in ('entry', 'interpreter')}
    for key in ('registry_file', 'output_root'):
        path = Path(request[key]);request[key] = str(path if path.is_absolute() else root / path)
    req, out = run / 'REQUEST.json', run / 'RESPONSE.json'
    entry = Path(completion.get('entry', 'scripts/prepare_pf_candidate.py'))
    entry = entry if entry.is_absolute() else root / entry
    result_registry = request['registry_file']
    inherited = False
    if req.exists():
        saved = json.loads(req.read_text())
        if saved != request:
            prior = spec.get('executor', {}).get('prior_run_dirs', ())
            if (stage_executor is None or mode != 'complete' or not
                    inherited_completion_request(entry, run, saved, request, prior)):
                raise CompletionRequestMismatch(req, saved, request)
            request = saved
            inherited = True
    if not req.exists():
        req.write_text(json.dumps(request, ensure_ascii=False, indent=2) + '\n')
    if not out.exists():
        cmd = [completion.get('interpreter', sys.executable), '-B', str(entry), '--request', str(req.resolve()),
               '--output', str(out.resolve()), '--mode', mode]
        if stage_executor is not None and mode=='complete':
            from .pf_distributed_preparation import complete_selected as distributed_complete
            options = {'inherited_registry': result_registry} if inherited else {}
            distributed_complete(root,spec,run,request,entry,out,stage_executor,**options)
        else:subprocess.run(cmd, cwd=root, check=True)
    response = json.loads(out.read_text())
    return dict(registry_file=response.get('registry_file', request['registry_file']), response=response,
                request=str(req), output=str(out),mode=mode)
