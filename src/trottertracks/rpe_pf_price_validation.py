"""Expose supplied deterministic-PF price models to optional validation.

No predictor, fitter, compiler or scientific eligibility logic. Frozen model
records, actual schedules and comparison/receipt locators remain authoritative.
"""
from copy import deepcopy
import re

from .rpe_price_validation import canonical, digest, enc, frac

_HISTORIC_MODELS = dict(commit='12372fa6b53c878dc5dd124e036572ba0cd82061',
    path='docs/research/pf_stage1_model_freeze_20261115/run01/FROZEN_MODELS.json')
_HISTORIC_CHECKS = (
    (6, dict(commit='a55813373b84b9128cbd10c6812c0d26143a4280',
        path='docs/research/d_pf_stage2_holdout_20261115/run01/RESULT.json', selector=['rows'])),
    (9, dict(commit='0a35101914c6833244b6e0bad7bc2767a49b5a47',
        path='docs/research/d_pf_m9_holdout_20261115/run01/RESULT.json', selector=['rows'])))


def _locator(value):
    return isinstance(value, dict) and bool(value.get('commit') and value.get('path'))


def _read(reader, value):
    return reader.read(value) if _locator(value) else deepcopy(value)


def _model_reference(row):
    for value in (row.get('frozen_model'), (row.get('model') or {}).get('source'),
                  (row.get('price_evidence') or {}).get('model'),
                  (row.get('price_evidence') or {}).get('source')):
        if _locator(value):
            return value
    return None


def _primary_price_row(reader, row):
    seen = set()
    while _model_reference(row) is None:
        provenance = row.get('source') or {}
        loc = row.get('price_handoff_source') or (provenance.get('C_price_table') if isinstance(provenance, dict) else None)
        if not _locator(loc):
            return row
        key = digest(loc)
        if key in seen:
            raise ValueError('PF_PRICE_SOURCE_CYCLE')
        seen.add(key)
        original = reader.read(loc)
        if not isinstance(original, dict):
            raise ValueError('PF_PRIMARY_PRICE_ROW_REQUIRED')
        for field in ('m', 'axis', 'physical_id'):
            if row.get(field) is not None and original.get(field) is not None and row[field] != original[field]:
                raise ValueError('PF_PRIMARY_PRICE_ROW_IDENTITY_DIFFERS:' + field)
        row = dict(original, **{k: v for k, v in row.items() if k != 'source'})
        row['source'] = original.get('source')
    return row


def _axis_model(value, axis):
    if value.get('coefficients') is not None:
        return value
    model = (value.get('models') or {}).get(axis)
    if not isinstance(model, dict) or model.get('coefficients') is None:
        raise ValueError('PF_FROZEN_AXIS_COEFFICIENTS_MISSING:' + axis)
    return model


def _same_compiled_step(row, step, models):
    canonical_step = frac(row['actual_delta_pf'])
    if canonical_step == frac(step):
        return True
    # Existing PF physical bridge constructs T as binary64, then T/q. This
    # representation is distinct from the canonical exact rational T/q; it is
    # not a tolerance or a modified physical schedule.
    lowering = {(m.get('domain', {}).get('implementation') or {}).get('lowering_id') for m in models.values()}
    return lowering == {'H46_LONG_TIME_NATIVE_COMPOSE'} and frac(float(frac(row['T'])) / row['q_PF']) == frac(step)


def describe_candidate(root, candidate, reader):
    """Describe only model-priced rows of an already accepted PF candidate.

    The existing optimizer supplies checked per-axis prices. DIRECT-only
    candidates return []; a model row with unknown provenance is rejected.
    """
    if candidate.get('method') not in ('FULL_PF', 'TRUNCATED_PF'):
        return []
    evidence = candidate.get('pf_evidence') or {}
    cfg_source = candidate.get('configuration_source') or evidence.get('configuration')
    cfg = candidate.get('configuration_body') or candidate.get('generated_configuration')
    if cfg is None and cfg_source is not None:
        cfg = _read(reader, cfg_source)
    table_source = candidate.get('price_source') or evidence.get('prices') or (candidate.get('price_evidence') or {}).get('source')
    table = _read(reader, table_source) if table_source is not None else {}
    table = table if isinstance(table, dict) else {}
    axes = candidate.get('price_axes') or table.get('axes') or (candidate.get('price_evidence') or {}).get('price_axes') or []
    models = {}
    for axis in axes:
        row = deepcopy(axis)
        kind = (row.get('kind') or row.get('price_kind') or row.get('classification') or '')
        if str(kind).startswith('DIRECT'):
            continue
        if row.get('price_source'):
            original = reader.read(row['price_source'])
            if isinstance(original, dict):
                row = dict(original, **row)
                kind = (row.get('kind') or row.get('price_kind') or row.get('classification') or '')
        if str(kind).startswith('DIRECT'):
            continue
        if not any(x in str(kind).upper() for x in ('MODEL', 'CONDITIONAL', 'CLOSURE', 'AFFINE')):
            continue
        row = _primary_price_row(reader, row)
        ref = _model_reference(row)
        table_model = table.get('model') or table.get('model_source')
        if ref is None and _locator(table_model):
            ref = table_model
        if ref is None:
            raise ValueError('PF_MODEL_PROVENANCE_MISSING')
        axis_name = row.get('axis')
        if axis_name not in ('X', 'Y'):
            raise ValueError('PF_MODEL_AXIS_MISSING')
        raw_model = reader.read(ref)
        axis_model = _axis_model(raw_model, axis_name)
        domain = axis_model.get('domain') or {}
        if domain.get('axis', axis_name) != axis_name or domain.get('route', candidate['method']) != candidate['method']:
            raise ValueError('PF_FROZEN_MODEL_DOMAIN_DIFFERS')
        if raw_model.get('configuration_id', candidate['configuration_id']) != candidate['configuration_id']:
            raise ValueError('PF_FROZEN_MODEL_CONFIGURATION_DIFFERS')
        if any(x.get('configuration_id', candidate['configuration_id']) != candidate['configuration_id']
               for x in axis_model.get('input_locators', [])):
            raise ValueError('PF_FROZEN_MODEL_CALIBRATION_CONFIGURATION_DIFFERS')
        model_group = axis_model.get('model_group')
        if isinstance(model_group, list) and len(model_group) == 4 and cfg is not None:
            if model_group[1] != cfg['spec']['formula'] or model_group[3] != axis_name:
                raise ValueError('PF_FROZEN_MODEL_GROUP_DIFFERS')
        step = row.get('actual_delta_pf', row.get('actual_delta'))
        if step is None and cfg is not None:
            step = next(r['actual_delta_pf'] for r in cfg['rounds'] if r['m'] == row['m'])
        if step is None:
            raise ValueError('PF_MODEL_ACTUAL_STEP_MISSING')
        source = {k: ref[k] for k in ('commit', 'path')}
        key = digest([source, enc(step)])
        group = models.setdefault(key, dict(source=source, step=enc(step), axis_models={}, model_sources={},
            calibration_rounds=raw_model.get('calibration_rounds', []), holdout_round=raw_model.get('holdout_round')))
        if axis_name in group['axis_models'] and group['axis_models'][axis_name] != axis_model:
            raise ValueError('PF_CONFLICTING_FROZEN_AXIS_MODELS')
        group['axis_models'][axis_name] = axis_model
        group['model_sources'][axis_name] = deepcopy(ref)
    if not models:
        return []
    if cfg is None or cfg.get('id') != candidate['configuration_id']:
        raise ValueError('PF_MODEL_CANONICAL_CONFIGURATION_MISSING_OR_DIFFERENT')
    prices = candidate.get('prices')
    if not prices or len(prices) != 2 * len(cfg['rounds']):
        raise ValueError('PF_MODEL_SUPPLIED_PRICE_VECTOR_MISSING')
    result = []
    spec = cfg['spec']
    for group in models.values():
        if set(group['axis_models']) != {'X', 'Y'}:
            raise ValueError('PF_MODEL_BOTH_AXIS_CONTRACTS_REQUIRED')
        family = dict(method=candidate['method'], p=(candidate.get('key') or {}).get('p', spec.get('L_D')),
            formula=spec['formula'], binding=spec.get('binding'), split=spec.get('split'),
            profile=spec.get('profile'), domains={a: m.get('domain') for a, m in group['axis_models'].items()})
        identity = canonical(dict(family=family, actual_step=group['step'], mode='DETERMINISTIC_PF_AFFINE',
            coefficients={a: m['coefficients'] for a, m in group['axis_models'].items()}, model_source=group['source']))
        calibration = []
        for model in group['axis_models'].values():
            for entry in model.get('input_locators') or model.get('calibration') or []:
                m = entry.get('m')
                if m is None:
                    match = re.search(r':m([0-9]+)$', entry.get('cell_id', ''))
                    if match: m = int(match.group(1))
                calibration.append(dict(m=m, q=entry.get('q'), R=0,
                    source=deepcopy(entry.get('price_source') or entry.get('source'))))
        calibration += [dict(m=m, R=0) for m in group['calibration_rounds']]
        axis_groups = {a: m.get('model_group_id', (group['model_sources'][a].get('selector') or [None])[-1])
                       for a, m in group['axis_models'].items()}
        sources = []
        if group['source'] == _HISTORIC_MODELS:
            sources += [dict(source=source, role='INDEPENDENT_HOLDOUT', format='PF_HOLDOUT',
                            m=m, R=0, configuration_id=cfg['id'], model_groups=axis_groups) for m, source in _HISTORIC_CHECKS]
        table_model = table.get('model') or table.get('model_source')
        if table.get('holdout') and _locator(table_source) and table_model and all(
                table_model.get(k) == group['source'][k] for k in ('commit', 'path')):
            sources.append(dict(source=dict(table_source, selector=list(table_source.get('selector') or []) + ['holdout']),
                role='INDEPENDENT_HOLDOUT', format='PF_HOLDOUT', R=0, configuration_id=cfg['id'], model_source=group['source']))
        result.append(dict(model_id=digest(identity), identity=identity, registry='PF:' + candidate['name'],
            calibration_sources=list(group['model_sources'].values()), calibration_points=calibration,
            evidence_sources=sources, configuration_id=cfg['id'], configuration_source=deepcopy(cfg_source),
            points=[dict(m=r['m'], q=r['q_PF'], R=0, physical_id=r['candidate']['id'],
                T=r['T'], actual_step=group['step'], canonical_actual_step=r['actual_delta_pf'],
                actual_step_rule='EXISTING_H46_BINARY64_T_DIV_Q' if frac(r['actual_delta_pf']) != frac(group['step']) else 'CANONICAL_EXACT_T_DIV_Q',
                round=deepcopy(r),
                predicted_RZ={a: enc(prices[2*r['m']+i]) for i, a in enumerate(('X', 'Y'))})
                for r in cfg['rounds'] if _same_compiled_step(r, group['step'], group['axis_models'])],
            qualified_for_candidate=False))
    return result
