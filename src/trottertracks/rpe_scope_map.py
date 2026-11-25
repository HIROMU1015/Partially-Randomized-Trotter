"""Metadata-only finite scope coverage and exact PR requested-width planning.

The proposed max-width mapper is deliberately distinct from the frozen fixed
grid evaluator. A metadata mapping is not evidence of numerical qualification.
"""
from collections import Counter
from fractions import Fraction
from itertools import product
from math import ceil

MAX_WIDTH_MAPPER = 'PR_DF_REQUESTED_MAX_WIDTH_V1'


def exact(value):
    return Fraction(value['numerator'], value['denominator']) if isinstance(value, dict) else Fraction(value)


def encode(value):
    value = exact(value)
    return dict(numerator=value.numerator, denominator=value.denominator)


def max_width_schedule(h, width, M, *, lambda_R=None, kappa=None, pieces=1):
    """Exact requested max width; retain the separate numerical realization."""
    h, width = exact(h), exact(width)
    if h <= 0 or width <= 0 or type(M) is not int or M < 0 or type(pieces) is not int or pieces < 1:
        raise ValueError('Positive h/width/piece count and nonnegative M required')
    if (lambda_R is None) != (kappa is None):
        raise ValueError('Both lambda_R and kappa are required for event counts')
    if lambda_R is not None and (exact(lambda_R) <= 0 or exact(kappa) <= 0):
        raise ValueError('Positive lambda_R and kappa required')
    rows = []
    for m in range(M + 1):
        T = h * 2**m
        q = max(1, ceil(T/width))
        actual = T/q
        R = None if lambda_R is None else max(pieces, ceil(exact(kappa)*exact(lambda_R)**2*T**2/q))
        rows.append(dict(m=m, T=encode(T), q_PF=q, actual_delta_pf=encode(actual),
                         actual_delta_binary64=float(actual).hex(), R_outer=R,
                         event_count=None if R is None else q*R))
    return dict(mapper=MAX_WIDTH_MAPPER, h=encode(h), requested_delta_pf=encode(width), rows=rows)


def scope_axes(method_scope, horizons):
    scope = method_scope or {}
    generation = scope.get('generation') or {}
    axes = {'M': horizons}
    if scope.get('configuration_ids') is not None:
        axes['source_configuration_id'] = scope['configuration_ids']
    else:
        for field, values in (
            ('p', scope.get('p_values')), ('L_D', scope.get('L_D_values')),
            ('formula', scope.get('formulas') or generation.get('formulas')),
            ('delta_pf', scope.get('requested_delta_pf_values')),
            ('delta_over_h', None if scope.get('requested_delta_pf_values') else
             scope.get('delta_over_h_values') or generation.get('delta_over_h_values')),
            ('absolute_kappa', scope.get('absolute_kappa_values')),
        ):
            if values:
                axes[field] = values
    return {key: tuple(dict.fromkeys(normalized(key, v) for v in vals)) for key, vals in axes.items()}


def normalized(key, value):
    if key in ('absolute_kappa', 'delta_over_h', 'delta_pf') and value is not None:
        return str(exact(value))
    return value


def coordinate(axes, values):
    return tuple(normalized(key, values.get(key)) for key in axes)


def row_coordinates(row):
    return dict(row.get('configuration') or {}, M=row['M'],
                source_configuration_id=row.get('source_configuration_id') or row.get('configuration_id'))


def _outcome_state(row):
    states = Counter(a['status'] for a in row.get('allocations', ()))
    price = row.get('cost_status')
    feasible = bool(row.get('mse_feasible') or row.get('best'))
    if row.get('minimum_status') == 'OUTSIDE_QUALIFIED_EVALUATOR_DOMAIN':
        status = 'EVALUATOR_UNSUPPORTED'
    elif states['PENDING'] or states['IN_PROGRESS']:
        status = 'EVALUATION_PENDING'
    elif states['UNKNOWN'] or states['TECHNICAL_HOLD']:
        status = 'EVALUATION_HELD'
    elif states['INPUT_WAITING']:
        status = 'INPUTS_REQUIRED'
    elif feasible and price == 'UNAVAILABLE':
        status = 'MSE_PASS_PRICE_REQUIRED'
    elif states['INCONCLUSIVE'] and row.get('unresolved_competitors', 1):
        status = 'EVALUATION_INCONCLUSIVE'
    elif feasible:
        status = 'EVALUATED_FEASIBLE'
    elif states and set(states) == {'MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD'}:
        status = 'EVALUATED_NO_PASS'
    else:
        status = 'NOT_EVALUATED'
    return dict(status=status, configuration_id=row.get('configuration_id'),
                allocations=dict(states), needs=row.get('needs', ()),
                minimum_status=row.get('minimum_status'), price_status=row.get('price_status'),
                mse_pass=feasible, price_required=feasible and price == 'UNAVAILABLE')


def _mapper_reason(family):
    reason = str(family.get('reason') or family['status'])
    if 'NONEMPTY' in reason:
        return dict(status='MAPPER_UNSUPPORTED', reason=reason, reason_kind='ENDPOINT_PROVIDER_UNSUPPORTED')
    if 'Nonintegral PF grid' in reason or 'REJECTED_REQUESTED_STEP_WIDTH' in reason:
        return dict(status='MAPPER_UNSUPPORTED', reason=reason, reason_kind='FIXED_GRID_POLICY', first_round=0)
    if 'NOT_EXACT_BINARY64' in reason:
        return dict(status='MAPPER_UNSUPPORTED', reason=reason, reason_kind='REQUESTED_WIDTH_REPRESENTATION')
    return dict(status='MAPPER_UNSUPPORTED', reason=reason, reason_kind='PROVIDER_GENERATION_UNSUPPORTED')


def build_scope_map(methods, horizons, *, families=None, points=(), results=None, mapper_version=None):
    """Exactly one row per logical coordinate; physical aliases remain visible."""
    families = families or {}
    results = results or {}
    methods_map = {}
    for method, spec in methods.items():
        axes = scope_axes(spec, horizons)
        rows = results.get(method, {}).get('by_configuration', ())
        indexed = {coordinate(axes, row_coordinates(r)): r for r in rows if r.get('configuration_id')}
        by_id = {(cid, r['M']): r for r in rows for cid in
                 (r.get('configuration_id'), r.get('source_configuration_id')) if cid}
        family_axes = tuple(k for k in axes if k not in ('M', 'absolute_kappa'))
        family_index = {}
        points_index = {}
        split_bases = {}
        for key, family in families.items():
            if family.get('method') != method:
                continue
            values = dict(family, delta_pf=family.get('requested_delta_pf', family.get('delta_pf')))
            family_index[coordinate(family_axes, values)] = (key, family)
            if family.get('status') == 'READY_METADATA' and family.get('configuration'):
                split_bases[(family.get('L_D'), family.get('formula'))] = family['configuration']
        for point in points:
            if point.get('method') == method:
                points_index[(point['family'], normalized('absolute_kappa', point.get('kappa')))] = point
        entries = []
        for key in product(*axes.values()):
            values = dict(zip(axes, key))
            record = dict(coordinates=values, status='NOT_GENERATED')
            row = indexed.get(key)
            family_entry = family_index.get(coordinate(family_axes, values))
            if family_entry is not None:
                family_key, family = family_entry
                record['family'] = family_key
                if family['status'] == 'UNSUPPORTED_MAPPER':
                    record.update(_mapper_reason(family))
                    if method == 'PR_DF' and 'delta_pf' in values and values.get('absolute_kappa') is not None:
                        base = split_bases.get((values.get('L_D'), values.get('formula')))
                        if base is not None:
                            spec = base['spec']
                            pieces = len(base['rounds'][0]['structure']['signed_weights'])
                            proposal = max_width_schedule(spec['h'], values['delta_pf'], values['M'],
                                lambda_R=spec['split']['lambda_R'], kappa=values['absolute_kappa'], pieces=pieces)
                            record['proposed_mapper'] = dict(proposal, status='METADATA_ONLY_NEW_PROVIDER_REQUIRED')
                        elif values.get('L_D') == 0:
                            record['proposed_mapper'] = dict(mapper=MAX_WIDTH_MAPPER,
                                status='ENDPOINT_DESCRIPTOR_AND_PROVIDER_REQUIRED',
                                one_body_policy='DETERMINISTIC_ONE_BODY_NOT_FULL_H_RTE')
                elif family['status'] != 'READY_METADATA':
                    record.update(status='FAMILY_PREPARATION_REQUIRED', reason=family.get('reason') or family['status'])
                else:
                    record['status'] = 'SCHEDULE_GENERATION_REQUIRED'
                    point = points_index.get((family_key, values.get('absolute_kappa')))
                    if point is not None:
                        logical = point.get('configuration_id') or (point.get('completion') or {}).get('configuration_id')
                        physical = point.get('evaluation_configuration_id') or logical
                        record.update(logical_configuration_id=logical, evaluation_configuration_id=physical,
                                      status=point['status'], reason=point.get('reason') or (point.get('completion') or {}).get('reason'))
                        if physical:
                            row = by_id.get((physical, values['M'])) or row
            if row is not None:
                record.update(_outcome_state(row))
            entries.append(record)
        counts = Counter(r['status'] for r in entries)
        reasons=Counter(r['reason_kind'] for r in entries if r.get('reason_kind'))
        resolved=counts['EVALUATED_FEASIBLE']+counts['EVALUATED_NO_PASS']
        methods_map[method] = dict(axes=list(axes), requested=len(entries), counts=dict(counts),
            resolved=resolved,unresolved=len(entries)-resolved,mapper_reason_counts=dict(reasons),
            mse_pass_price_required=sum(bool(r.get('price_required')) for r in entries),rows=entries)
    return dict(schema='RPE_SCOPE_MAP_V1', mapper_version=mapper_version, methods=methods_map)
