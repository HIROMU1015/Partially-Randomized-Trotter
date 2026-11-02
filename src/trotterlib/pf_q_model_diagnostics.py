"""Exact diagnostics of already saved PF prices; never evaluates a model."""
from collections import defaultdict
from fractions import Fraction

THRESHOLD = Fraction(1, 50)
GROUPINGS = (
    (), ('system',), ('system', 'q'), ('system', 'prefix'),
    ('system', 'PF'), ('system', 'method'), ('system', 'axis'),
    ('system', 'PF', 'method'), ('system', 'PF', 'q'),
    ('system', 'PF', 'prefix'), ('system', 'PF', 'method', 'q'),
    ('system', 'PF', 'method', 'prefix'),
    ('system', 'PF', 'method', 'axis'),
    ('system', 'model_point_class'), ('system', 'direct_cohort'),
)


def axis_errors(prediction_literal, measured):
    if type(measured) is not int or measured <= 0:
        raise ValueError('POSITIVE_INTEGER_MEASURED_RZ_REQUIRED_NO_ZERO_FILL')
    predicted = Fraction(prediction_literal)
    if predicted <= 0:
        raise ValueError('POSITIVE_SAVED_PREDICTION_REQUIRED')
    difference = predicted - measured
    relative = difference / measured
    return dict(predicted_RZ_exact=str(predicted), measured_RZ=measured,
                difference_RZ_exact=str(difference), absolute_error_RZ_exact=str(abs(difference)),
                signed_relative_error_exact=str(relative), absolute_relative_error_exact=str(abs(relative)),
                predicted_RZ=float(predicted), difference_RZ=float(difference),
                absolute_error_RZ=float(abs(difference)), signed_relative_error_percent=float(100*relative),
                absolute_relative_error_percent=float(100*abs(relative)),
                strictly_below_2_percent=abs(relative) < THRESHOLD)


def point_class(roles):
    if any(r in {'TRAIN', 'TRAIN_ANCHOR'} for r in roles):
        return 'TRAINING_USED'
    if roles:
        return 'VALIDATION_USED'
    return 'OTHER_COMPARISON_POINT'


def quantile(values, probability):
    """Type-7 empirical quantile, exact linear interpolation; no CI/inference."""
    if not values:
        raise ValueError('EMPTY_GROUP_NO_ZERO_FILL')
    values = sorted(Fraction(v) for v in values)
    probability = Fraction(probability)
    if not 0 <= probability <= 1:
        raise ValueError('QUANTILE_PROBABILITY')
    position = probability * (len(values)-1)
    lo = position.numerator // position.denominator
    hi = min(lo+1, len(values)-1)
    return values[lo] + (position-lo)*(values[hi]-values[lo])


def summarize(rows):
    if not rows:
        raise ValueError('EMPTY_GROUP_NO_ZERO_FILL')
    errors = [Fraction(r['difference_RZ_exact']) for r in rows]
    absolute = [Fraction(r['absolute_error_RZ_exact']) for r in rows]
    signed = [Fraction(r['signed_relative_error_exact']) for r in rows]
    relative = [Fraction(r['absolute_relative_error_exact']) for r in rows]
    n = len(rows)
    result = dict(n_axes=n, below_2_percent=sum(r['strictly_below_2_percent'] for r in rows),
                  at_or_above_2_percent=sum(not r['strictly_below_2_percent'] for r in rows),
                  original_final_gate_missing_axes=sum(r['original_global_final_gate_missing'] for r in rows))
    scalar = {
        'mean_signed_error_RZ': sum(errors)/n, 'mean_absolute_error_RZ': sum(absolute)/n,
        'mean_signed_relative_error_percent': 100*sum(signed)/n,
        'mean_absolute_relative_error_percent': 100*sum(relative)/n,
        'median_absolute_relative_error_percent': 100*quantile(relative, Fraction(1,2)),
        'p95_absolute_relative_error_percent': 100*quantile(relative, Fraction(19,20)),
        'max_absolute_relative_error_percent': 100*max(relative),
        'min_signed_relative_error_percent': 100*min(signed),
        'max_signed_relative_error_percent': 100*max(signed),
    }
    for name, value in scalar.items():
        result[name] = float(value)
        result[name+'_exact'] = str(value)
    result['max_error_task_ids'] = [r['task_id'] for r in rows
                                    if Fraction(r['absolute_relative_error_exact']) == max(relative)]
    return result


def aggregate(rows):
    result = []
    for dimensions in GROUPINGS:
        groups = defaultdict(list)
        for row in rows:
            groups[tuple(row[d] for d in dimensions)].append(row)
        for key, selected in sorted(groups.items(), key=lambda item: str(item[0])):
            result.append(dict(grouping='+'.join(dimensions) or 'ALL',
                               group=dict(zip(dimensions, key)), **summarize(selected)))
    return result


def diagnose(correspondence, predictions):
    """Select only the authorized saved 450 points; refuse gaps or ambiguity."""
    saved = defaultdict(list)
    for record in predictions:
        saved[record['task_id']].append(record)
    selected = [r for r in correspondence if r['comparison_class'] == 'CONDITIONAL_COMPARISON']
    if len(selected) != 450 or len({r['task_id'] for r in selected}) != 450:
        raise ValueError('EXACT_AUTHORIZED_450_AXES_REQUIRED')
    result = []
    for row in selected:
        literals = row['saved_prediction_literals']
        records = saved[row['task_id']]
        if len(literals) != 1 or not records or {r['saved_prediction_literal'] for r in records} != set(literals):
            raise ValueError('MISSING_OR_AMBIGUOUS_SAVED_PREDICTION_NO_EVALUATION')
        direct = int(row['direct_RZ_literal'])
        if str(direct) != row['direct_RZ_literal']:
            raise ValueError('INTEGER_DIRECT_LITERAL_REQUIRED')
        result.append(dict(row, prior_residuals_computed=row['residuals_computed'],
                           model_point_class=point_class(row['prior_model_point_roles']),
                           saved_prediction_provenance=records,
                           **axis_errors(literals[0], direct)))
    return result
