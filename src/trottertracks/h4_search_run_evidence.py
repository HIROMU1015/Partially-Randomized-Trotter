"""Published optimizer metadata for the H4 search map; never executes a producer.

Summary outcomes count as observed evaluations, with their summary provenance.
Only explicitly NEW_Q3_JOB records count as new work in a legacy run. Reused
records, child-worker copies, and plan-only targets are kept distinct.
"""
from collections import defaultdict
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import PurePosixPath
import re
import subprocess

from .saved_rpe_cost_catalog import ref

METHODS = {'FULL_PF', 'TRUNCATED_PF', 'PR_PAULI', 'PR_DF'}
RESULT_NAMES = {'ROUND_RESULT.json', 'ROUND_RESULT_INTERRUPTED.json'}
SCOPE_NAMES = {'SCOPE_METADATA.json', 'SEARCH_STATE.json', 'STATE.json'}


def _git(reader, *args):
    return subprocess.check_output(['git', *args], cwd=reader.root, text=True).strip()


def _allowed(path):
    parts = PurePosixPath(path).parts
    return not any(p in {'jobs', 'checks', 'load_check', 'tests', 'fixed_reference', 'local_plan', 'fixtures'} or
                   p.startswith(('artificial', 'executor_acceptance_', 'acceptance_',
                                 'plan_only', 'preparation_connection_')) for p in parts)


def _ident(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()[:24]


def _source(reader, head, path, selector=()):
    fixed = _git(reader, 'log', '-1', '--format=%H', head, '--', path)
    return ref(fixed or head, path, *selector)


def _read(reader, head, path):
    try:
        return reader.document(head, path)
    except (subprocess.CalledProcessError, KeyError, TypeError, ValueError):
        return None


def _attrs(row):
    """Display attributes only, preserving the original width representation."""
    key = row.get('configuration') or row.get('key') or {}
    if not isinstance(key, dict):
        key = {}
    result = {k: row.get(k, key.get(k)) for k in ('formula', 'p', 'L_D', 'h')}
    width = row.get('requested_delta_pf', row.get('requested_width'))
    if width is None:
        width = key.get('requested_delta_pf', key.get('requested_width'))
    if width is None and key.get('delta_over_h') is not None:
        width = {'delta_over_h': key['delta_over_h']}
    if width is None and row.get('delta_over_h') is not None:
        width = {'delta_over_h': row['delta_over_h']}
    result['requested_width'] = width
    result['schedule'] = row.get('schedule')
    result['kappa'] = row.get('kappa_value', row.get('kappa', row.get('metadata_kappa')))
    # Older display names are the published search_range's only scope evidence.
    name = row.get('name', row.get('candidate', ''))
    for field in ('formula', 'p', 'L_D', 'delta_over_h'):
        match = re.search(r'(?:[:,|]|^)' + field + r'=([^,:|]+)', name)
        if match:
            val = match.group(1)
            if field in ('p', 'L_D'):
                try: val = int(val)
                except ValueError: pass
            if field == 'delta_over_h' and result['requested_width'] is None:
                result['requested_width'] = {'delta_over_h': val}
            elif field in result and result[field] is None:
                result[field] = val
    return result


def _configuration_attrs(reader, locator, configuration_id):
    """Resolve an explicitly published locator, never a producer's local path."""
    if not (isinstance(locator, dict) and locator.get('commit') and locator.get('path')):
        return None
    try:
        value = reader.read(locator)
    except (subprocess.CalledProcessError, KeyError, TypeError, ValueError, IndexError):
        return None
    candidates = [value]
    if isinstance(value, dict):
        candidates += [value.get('configuration')]
        candidates += [p.get('configuration') for p in value.get('packets', []) if isinstance(p, dict)]
    config = next((c for c in candidates if isinstance(c, dict) and
                   c.get('id') == configuration_id and isinstance(c.get('spec'), dict)), None)
    if config is None:
        return None
    spec = config['spec']; split = spec.get('split') or {}
    schedule = []
    for item in config.get('rounds', []):
        # Canonical rounds embed the full Hamiltonian binding repeatedly. The
        # map needs the saved schedule, not thousands of copies of that binding.
        step = {k: deepcopy(item[k]) for k in ('m', 'T', 'q_RPE', 'q_PF', 'R_outer',
            'actual_delta_pf', 'structure') if k in item}
        candidate = item.get('candidate') or {}
        if candidate.get('id'):
            step['physical_id'] = candidate['id']
        schedule.append(step)
    return dict(formula=spec.get('formula'), L_D=spec.get('L_D', split.get('L_D')),
                physical_binding=spec.get('binding'),profile=spec.get('profile'),
                p=split.get('prefix', split.get('p')), h=spec.get('h'),
                requested_width=spec.get('delta_pf', spec.get('requested_delta_pf')),
                schedule=schedule, kappa=deepcopy(spec.get('kappa')),
                split={k: deepcopy(split[k]) for k in ('prefix', 'p', 'L_D', 'scalar', 'randomized_scope',
                    'deterministic_scope', 'original_PF_method') if k in split},
                law=deepcopy(spec.get('law')),
                configuration_source=deepcopy(locator), metadata_verification='EXACT_CONFIGURATION_ID_AT_FIXED_LOCATOR')


def _resolved_attrs(row):
    attrs = _attrs(row)
    for key, value in (row.get('_resolved_attrs') or {}).items():
        if value is not None:
            # Immutable public metadata is shared across allocations. The map
            # consumer owns its output copy; no solver mutates these objects.
            attrs[key] = value
    return attrs


def _bound_summary(value, source, field):
    """Long exact fractions stay at their immutable original selector."""
    if isinstance(value, dict) and 'numerator' in value:
        if len(str(value['numerator'])) + len(str(value.get('denominator', ''))) > 700:
            return dict(exact_source=dict(source, selector=list(source.get('selector', [])) + [field]),
                        display='元結果のexact上下界を参照')
    return value


def _target(row, method, source):
    return dict(_resolved_attrs(row), method=method,
                configuration_id=row.get('configuration_id'),
                name=row.get('name', row.get('candidate', row.get('family'))),
                source=source, status=row.get('status', 'TARGETED'),
                missing=row.get('needs', row.get('missing', [])))


def _eval(row, method, source, attrs=None):
    result = dict(attrs or _attrs(row))
    tag = row.get('source', row.get('answer_source'))
    new = True if tag == 'NEW_Q3_JOB' else False if tag in {
        'SAVED', 'PRIOR_ROUND_JOB', 'RESUMED', 'SAVED_RESULT'} else None
    result.update(method=method, configuration_id=row.get('configuration_id'),
                  N_X=row.get('N_X'), N_Y=row.get('N_Y'),
                  q3_status=row.get('status'), MSE_lower=_bound_summary(row.get('MSE_lower'), source, 'MSE_lower'),
                  MSE_upper=_bound_summary(row.get('MSE_upper'), source, 'MSE_upper'), conditional_cost=row.get('cost'),
                  source=source, configuration_source=row.get('configuration_source') or result.get('configuration_source'),
                  new_evaluation=new, origin=tag or row.get('origin'),
                  job=row.get('job'), outcome_sha256=row.get('outcome_sha256'),
                  signal_source=row.get('signal_source'),
                  adopted_evaluation=row.get('adopted_evaluation') if isinstance(row.get('adopted_evaluation'), dict) else None,
                  controller_adopted=row.get('controller_adopted', row.get('adopted_evaluation') is not False),
                  price_evidence=row.get('price_quote', row.get('price_evidence')),
                  hosts=[], issues=[], evidence_kind='PUBLISHED_SUMMARY',
                  result_sources=row.get('result_sources', []))
    if not result['N_X'] or not result['N_Y']:
        result['issues'].append('SAVED_ALLOCATION_MISSING')
    if result['adopted_evaluation'] is None:
        result['issues'].append('EVALUATION_CONTRACT_NOT_IN_SUMMARY')
    return result


def _scope_from_round(spec, source):
    """Declare explicit finite grids only; no κ or science candidates generated."""
    targets = []
    widths = spec.get('search_overrides', {}).get('requested_widths') or {}
    for method in spec.get('methods', []):
        if method not in METHODS:
            continue
        info = spec.get('method_overrides', {}).get(method, {})
        gen = info.get('generation') or {}
        formulas = gen.get('formulas') or [None]
        indices = info.get('p_values' if method == 'TRUNCATED_PF' else 'L_D_values') or [None]
        values = widths.get(method) or [dict(delta_over_h=v) for v in gen.get('delta_over_h_values', [])] or [None]
        for formula in formulas:
            for index in indices:
                for width in values:
                    target = dict(method=method, configuration_id=None, formula=formula,
                                  p=index if method == 'TRUNCATED_PF' else None,
                                  L_D=index if method.startswith('PR_') else None,
                                  requested_width=width, h=None, source=source,
                                  status='DECLARED_TARGET', missing=[], schedule=None, kappa=None)
                    targets.append(target)
    return targets


def _context(reader, head, directory, paths, spec=None):
    """Sibling public documents, never private filesystem/run traversal."""
    needs_path = directory + '/NEEDS.json'
    needs = _read(reader, head, needs_path) if needs_path in paths else None
    meta = {}
    if isinstance(needs, dict):
        for i, row in enumerate(needs.get('candidates', [])):
            key = row.get('configuration_id') or row.get('name')
            if key: meta[key] = (deepcopy(row), _source(reader, head, needs_path, ['candidates', i]))
    # Ordinary route publication embeds immutable configuration locators, not
    # just labels. This covers old saved configurations missing from NEEDS.
    for method, routes in (spec or {}).get('routes_override', {}).items():
        if method not in METHODS:
            continue
        for route in routes:
            for cid, item in (route.get('configurations') or {}).items():
                locator = item.get('configuration', item.get('configuration_source'))
                attrs = _configuration_attrs(reader, locator, cid)
                if attrs:
                    if cid not in meta:
                        meta[cid] = (dict(method=method, configuration_id=cid, name=item.get('label')),
                                     dict(locator))
                    meta[cid][0]['_resolved_attrs'] = attrs
    for cid, (row, _) in meta.items():
        if row.get('_resolved_attrs'):
            continue
        prep = row.get('preparation') or {}
        for locator in (row.get('configuration_source'), prep.get('configuration_source'), row.get('signal_source')):
            attrs = _configuration_attrs(reader, locator, cid)
            if attrs:
                row['_resolved_attrs'] = attrs
                break
    return meta


def _scaled_horizon_directions(rows):
    """Compare adjacent horizons of one explicitly linked physical candidate."""
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        cid = row.get('source_configuration_id') or row['configuration_id']
        if cid and type(row['M']) is int:
            groups[cid][row['M']].append(row)

    def feasible_cost(row):
        best = row.get('best') or {}
        cost = best.get('cost')
        if best.get('status') != 'MODEL_BOUND_MEETS_ADOPTED_THRESHOLD' or cost is None:
            return None
        try:
            if isinstance(cost, dict):
                return Fraction(int(cost['numerator']), int(cost['denominator']))
            return Fraction(cost) if type(cost) is int or isinstance(cost, str) else None
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return None

    for cid, horizons in sorted(groups.items()):
        pairs = sorted(horizons.items())
        for (m, a), (n, b) in zip(pairs, pairs[1:]):
            if len(a) != 1 or len(b) != 1:
                continue
            left, right = a[0], b[0]
            lo, hi = feasible_cost(left), feasible_cost(right)
            delta = None if lo is None or hi is None else hi - lo
            yield dict(source_configuration_id=cid, from_M=m, to_M=n, from_source=left['source'],
                to_source=right['source'], from_status=left['minimum_status'], to_status=right['minimum_status'],
                direction='UNRESOLVED' if delta is None else 'COST_DOWN' if delta < 0 else 'COST_UP' if delta > 0 else 'EQUAL',
                delta_RZ=None if delta is None else dict(numerator=str(delta.numerator), denominator=str(delta.denominator)))


def _run_result(reader, head, path, document, paths):
    source = _source(reader, head, path)
    spec = document.get('round') or {}
    directory = str(PurePosixPath(path).parent)
    run_id = spec.get('round_id') or directory
    # A published checkpoint copy is not another controller execution. Alias
    # it only when the normal sibling result exists with the same saved round.
    if PurePosixPath(directory).name == 'first_snapshot':
        sibling=str(PurePosixPath(directory).parent/'run01'/'ROUND_RESULT.json')
        if sibling in paths:
            normal=_read(reader,head,sibling)or{}
            if (normal.get('round')or{}).get('round_id')==run_id:
                directory=str(PurePosixPath(sibling).parent)
    run = dict(id=_ident([run_id, directory]), label=run_id, source=source,
               directory=directory, status={}, hosts=[], scope_sources=[source],
               targets=_scope_from_round(spec, dict(source, selector=['round'])),
               evaluations=[], branches=[], snapshot_source_commit=head)
    metadata = _context(reader, head, directory, paths, spec)
    for row, locator in metadata.values():
        if row.get('method') in METHODS:
            run['targets'].append(_target(row, row['method'], locator))
    for method, value in (document.get('methods') or {}).items():
        if method not in METHODS or not isinstance(value, dict):
            continue
        run['status'][method] = value.get('stop_reason')
        if (value.get('search_range') or {}).get('rule') == 'RPE_SCALED_SHOTS_V1' and value.get('scaled_shots'):
            # Keep the controller's conditional quote and actual Q3 verdict
            # separate from C's existing price admission/certification.
            summaries = []
            for i, row in enumerate(value.get('by_configuration') or []):
                item = {k: deepcopy(row.get(k)) for k in ('configuration_id', 'source_configuration_id',
                    'M', 'configuration', 'needs', 'minimum_status', 'unresolved_competitors', 'required_rounds',
                    'best', 'comparison_best', 'comparison_delta_RZ', 'mse_feasible', 'cost_status', 'price_source')}
                item['source'] = dict(source, selector=['methods', method, 'by_configuration', i])
                item['allocation_states'] = {}
                item['allocation_reasons'] = {}
                for alloc in row.get('allocations') or []:
                    state = alloc.get('status')
                    item['allocation_states'][state] = item['allocation_states'].get(state, 0) + 1
                    reason = alloc.get('reason')
                    if reason:
                        item['allocation_reasons'][reason] = item['allocation_reasons'].get(reason, 0) + 1
                summaries.append(item)
            run.setdefault('scaled_shots', {})[method] = dict(
                summary=deepcopy(value['scaled_shots']), configurations=summaries,
                stop_reason=run['status'][method], horizon_directions=list(_scaled_horizon_directions(summaries)))
        for i, row in enumerate(value.get('not_ready_candidates', [])):
            run['targets'].append(_target(row, method, dict(source, selector=['methods', method, 'not_ready_candidates', i])))
        for i, name in enumerate((value.get('search_range') or {}).get('candidates', [])):
            item = {'candidate': name} if isinstance(name, str) else name
            run['targets'].append(_target(item, method, dict(source, selector=['methods', method, 'search_range', 'candidates', i])))
        for i, row in enumerate(value.get('evaluated', [])):
            cmeta = metadata.get(row.get('configuration_id'))
            attrs = _resolved_attrs(cmeta[0]) if cmeta else _attrs(row)
            e = _eval(row, method, dict(source, selector=['methods', method, 'evaluated', i]), attrs)
            if e['new_evaluation'] is True:
                job = str(e.get('job') or '')
                if not (job.startswith(directory + '/jobs/') or ('/' + directory + '/jobs/') in job):
                    e['new_evaluation'] = None
                    e['issues'].append('NEW_JOB_OWNER_NOT_RESOLVED_TO_THIS_RUN')
            if cmeta:
                e['candidate_metadata_source'] = cmeta[1]
                e['signal_source'] = e['signal_source'] or cmeta[0].get('signal_source')
                e['price_evidence'] = e['price_evidence'] or cmeta[0].get('price_quote')
            if cmeta and e.get('conditional_cost') is not None:
                quote = e.get('price_evidence') or {}
                if quote.get('complete') is not True:
                    e['issues'].append('PRICE_BINDING_NOT_COMPLETE_IN_PUBLISHED_CANDIDATE')
            run['evaluations'].append(e)
    for candidate in (directory + '/STATUS.json', str(PurePosixPath(directory).parent) + '/STATUS.json'):
        if candidate in paths:
            status = _read(reader, head, candidate) or {}
            if status.get('host'):
                run['hosts'].append(status['host'])
                run.setdefault('host_sources', []).append(_source(reader, head, candidate))
    for e in run['evaluations']:
        e['hosts'] = list(run['hosts'])
    return run


def _scope_metadata(reader, head, path, document):
    source = _source(reader, head, path)
    directory = str(PurePosixPath(path).parent)
    status_doc = _read(reader, head, directory + '/STATUS.json') or {}
    round_doc = _read(reader, head, directory + '/ROUND.json') or {}
    run_id = status_doc.get('round_id') or round_doc.get('round_id') or directory
    run = dict(id=_ident([run_id, directory + '/run01']), label=run_id, source=source,
               directory=directory + '/run01', status=status_doc.get('state', 'SCOPE_ONLY'),
               hosts=[status_doc['host']] if status_doc.get('host') else [],
               scope_sources=[source], targets=[], evaluations=[], branches=[],
               snapshot_source_commit=head)
    for i, row in enumerate(document.get('families', [])):
        if row.get('method') in METHODS:
            run['targets'].append(_target(row, row['method'], dict(source, selector=['families', i])))
    if not run['targets']:
        run['targets'] = _scope_from_round(round_doc, _source(reader, head, directory + '/ROUND.json'))
    if status_doc.get('valid_outcomes') and not run['evaluations']:
        run['issues'] = ['STATUS_REPORTS_OUTCOMES_BUT_PER_ALLOCATION_CONTROLLER_RESULT_NOT_PUBLISHED']
        for target in run['targets']:
            target['missing'] = list(target.get('missing') or []) + ['CONTROLLER_ALLOCATION_RESULTS_NOT_PUBLISHED']
    return run


def _validation_metadata(reader, head, path, run):
    """Read a controller-published diagnostic sidecar without changing admission.

    Coefficients and observation sources remain in the producer's fixed record;
    a display status cannot qualify a price, Q3 result or scientific comparison.
    """
    value = reader.document(head, path)
    if value.get('schema') != 'RPE_PRICE_MODEL_VALIDATIONS_V1':
        raise ValueError('Unsupported price model validation schema')
    models = value.get('models', [])
    if not isinstance(models, list) or any(not isinstance(m, dict) or not m.get('model_id') for m in models):
        raise ValueError('Price validation models require explicit model IDs')
    if len({m['model_id'] for m in models}) != len(models):
        raise ValueError('Duplicate price validation model ID')
    known = {m['model_id'] for m in models}
    applications = value.get('applications', [])
    if not isinstance(applications, list) or any(not isinstance(a, dict) or not a.get('configuration_id')
            or a.get('model_id') not in known for a in applications):
        raise ValueError('Price validation application lacks configuration/model binding')
    run['price_model_validations'] = value
    run['price_validation_source'] = _source(reader, head, path)
    return run


def _publication(reader, head, path, doc, paths):
    """A producer chooses the public controller files; validate bytes only."""
    if doc.get('role') != 'CONTROLLER':
        raise ValueError('Only controller-adopted results may register evaluations')
    declared = doc.get('files', {})
    if not isinstance(declared, dict):
        raise ValueError('Publication files must be a name to locator mapping')
    verified = {}; optional_issues = []
    for name, item in declared.items():
        if not isinstance(item, dict) or not item.get('path') or not item.get('sha256'):
            raise ValueError('Publication file locator/hash missing: ' + name)
        p = item['path']
        try:
            if p not in paths or not _allowed(p):
                raise ValueError('Publication references absent or non-controller file: ' + p)
            raw = subprocess.check_output(['git', 'show', head + ':' + p], cwd=reader.root)
            if hashlib.sha256(raw).hexdigest() != item['sha256']:
                raise ValueError('Publication SHA256 mismatch: ' + p)
            verified[PurePosixPath(p).name] = p
        except (ValueError, subprocess.CalledProcessError) as exc:
            if PurePosixPath(p).name != 'PRICE_MODEL_VALIDATIONS.json':
                raise
            optional_issues.append('PRICE_VALIDATION_METADATA_INCOMPLETE:' + str(exc))
    result_path = next((verified[n] for n in RESULT_NAMES if n in verified), None)
    if result_path:
        result = reader.document(head, result_path)
        if result.get('schema') != 'RPE_OPTIMIZER_ROUND_RESULT_V1':
            raise ValueError('Unsupported controller result schema')
        run = _run_result(reader, head, result_path, result, paths)
    elif 'SCOPE_METADATA.json' in verified:
        p = verified['SCOPE_METADATA.json']
        run = _scope_metadata(reader, head, p, reader.document(head, p))
    else:
        raise ValueError('Publication lacks supported controller result or scope metadata')
    if 'PRICE_MODEL_VALIDATIONS.json' in verified:
        try:
            _validation_metadata(reader, head, verified['PRICE_MODEL_VALIDATIONS.json'], run)
        except (ValueError, TypeError, KeyError, subprocess.CalledProcessError) as exc:
            optional_issues.append('PRICE_VALIDATION_METADATA_INCOMPLETE:' + str(exc))
    if optional_issues:
        run.setdefault('issues', []).extend(optional_issues)
    directory = doc.get('run_directory', run['directory'])
    label = doc.get('run_id', run['label'])
    run.update(id=_ident([label, directory]), label=label, directory=directory,
               publication_source=_source(reader, head, path),
               hosts=doc.get('execution_hosts', []), producer_role='CONTROLLER')
    progress = doc.get('search_progress')
    if isinstance(progress, dict) and progress.get('schema') == 'H4_SEARCH_PROGRESS_V1':
        run['search_progress'] = dict(progress, source=dict(run['publication_source'], selector=['search_progress']))
    return run


def _unique(values):
    seen = set()
    for value in values:
        key = json.dumps(value, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            yield value


def _coalesce(reader, runs):
    """A run may publish several snapshots and be carried by many branches."""
    grouped = {}
    for run in runs:
        grouped.setdefault(run['id'], []).append(run)
    out = []
    for run_id, snapshots in grouped.items():
        chosen = snapshots[0]
        unordered = False
        for view in snapshots[1:]:
            # A validation sidecar can advance without a new Q3 result.
            # The common publication is the immutable snapshot boundary.
            a = (chosen.get('publication_source') or chosen['source'])['commit']
            b = (view.get('publication_source') or view['source'])['commit']
            if a == b:
                continue
            if subprocess.run(['git', 'merge-base', '--is-ancestor', a, b], cwd=reader.root,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
                chosen = view
            elif subprocess.run(['git', 'merge-base', '--is-ancestor', b, a], cwd=reader.root,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
                unordered = True
        run = dict(chosen)
        run['snapshot_history'] = list(_unique(dict(source=s['source'], status=s['status'],
            publication_source=s.get('publication_source'), price_validation_source=s.get('price_validation_source'),
            branches=s['branches']) for s in snapshots))
        run['snapshot_selection'] = 'UNORDERED_BRANCH_VIEWS' if unordered else 'SOURCE_ANCESTRY_LATEST'
        run['branches'] = sorted({b for s in snapshots for b in s['branches']})
        run['hosts'] = list(_unique(h for s in snapshots for h in s.get('hosts', [])))
        run['scope_sources'] = list(_unique(v for s in snapshots for v in s.get('scope_sources', [])))
        run['targets'] = list(_unique(v for s in snapshots for v in s['targets']))
        evaluations = {}
        for snap in snapshots:
            for e in snap['evaluations']:
                key = _ident([e['method'], e['configuration_id'], e['N_X'], e['N_Y'], e.get('adopted_evaluation')])
                if key not in evaluations:
                    evaluations[key] = dict(e)
                    evaluations[key]['source_history'] = [e['source']]
                else:
                    previous = evaluations[key]
                    previous['source_history'] = list(_unique(previous['source_history'] + [e['source']]))
                    if e['new_evaluation'] is True:
                        previous['new_evaluation'] = True
                    if any(previous.get(k) != e.get(k) for k in ('q3_status', 'conditional_cost')):
                        previous['issues'] = list(_unique(previous['issues'] + ['PUBLIC_SNAPSHOT_OUTCOME_CONFLICT']))
                        previous.setdefault('outcome_variants', []).append(dict(source=e['source'],
                            q3_status=e['q3_status'], conditional_cost=e['conditional_cost']))
        run['evaluations'] = list(evaluations.values())
        out.append(run)
    return out


def _propagate_fixed_configuration_metadata(runs):
    """Reuse exact-ID metadata across saved runs, independently of allocation."""
    known = {}
    fields = ('formula', 'p', 'L_D', 'h', 'requested_width', 'schedule', 'kappa',
              'split', 'law', 'physical_binding','profile','configuration_source', 'metadata_verification')
    for run in runs:
        for row in run['targets'] + run['evaluations']:
            if row.get('metadata_verification') == 'EXACT_CONFIGURATION_ID_AT_FIXED_LOCATOR':
                known.setdefault(row.get('configuration_id'), row)
    for run in runs:
        for row in run['targets'] + run['evaluations']:
            saved = known.get(row.get('configuration_id'))
            if saved:
                for field in fields:
                    # Fixed canonical input is stronger than a display label.
                    if saved.get(field) is not None:
                        row[field] = saved[field]
                row['configuration_metadata_reused_by_exact_id'] = True
    return runs


def discover_runs(reader, heads, *, legacy_heads=None):
    """Read all unique public controller result snapshots and scope declarations.

    Branch names/host local paths do not establish execution identity. Histories
    carried into several branches collapse by path+blob before normalization.
    """
    found, issues, trees = {}, [], {}
    for branch, head in sorted(heads.items()):
        try:
            tree = _git(reader, 'rev-parse', head + ':docs/research')
        except subprocess.CalledProcessError:
            continue
        if tree not in trees:
            paths = {}
            for line in _git(reader, 'ls-tree', '-r', head, '--', 'docs/research').splitlines():
                meta, path = line.split('\t', 1)
                paths[path] = meta.split()[2]
            trees[tree] = paths
        paths = trees[tree]
        # An explicit publication owns its referenced result paths, even when
        # malformed: a bad hash or worker role must not fall back to implicit
        # controller discovery of the same raw file.
        covered = set()
        for p in paths:
            if PurePosixPath(p).name != 'SEARCH_PUBLICATION.json' or not _allowed(p):
                continue
            publication = _read(reader, head, p)
            if isinstance(publication, dict) and publication.get('schema') == 'H4_SEARCH_PUBLICATION_V1':
                for item in (publication.get('files') or {}).values():
                    if isinstance(item, dict) and item.get('path'):
                        covered.add(item['path'])
        for path, blob in sorted(paths.items()):
            name = PurePosixPath(path).name
            if name not in RESULT_NAMES | SCOPE_NAMES | {'SEARCH_PUBLICATION.json'} or not _allowed(path):
                continue
            if legacy_heads is not None and branch not in legacy_heads and name != 'SEARCH_PUBLICATION.json':
                continue
            if path in covered:
                continue
            key = (path, blob)
            if key in found:
                found[key]['branches'].append(branch)
                continue
            try:
                doc = reader.document(head, path)
                if not isinstance(doc, dict) or doc.get('artificial_only') is True or doc.get('artificial') is True:
                    continue
                if name == 'SEARCH_PUBLICATION.json' and doc.get('schema') == 'H4_SEARCH_PUBLICATION_V1':
                    run = _publication(reader, head, path, doc, paths)
                elif name in RESULT_NAMES and doc.get('schema') == 'RPE_OPTIMIZER_ROUND_RESULT_V1':
                    run = _run_result(reader, head, path, doc, paths)
                elif name == 'SCOPE_METADATA.json' and isinstance(doc.get('families'), list):
                    run = _scope_metadata(reader, head, path, doc)
                else:
                    continue
                run['branches'] = [branch]
                found[key] = run
            except (ValueError, TypeError, KeyError, subprocess.CalledProcessError) as exc:
                issues.append(dict(source=ref(head, path), reason=type(exc).__name__ + ': ' + str(exc)[:200]))
    return _propagate_fixed_configuration_metadata(_coalesce(reader, list(found.values()))), issues
