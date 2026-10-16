"""Metadata-only H8 B1 readiness guards. No fit or scientific entry point.

Model equations are hypotheses. This module never evaluates them on real costs.
"""
from fractions import Fraction


METHODS = ('B0', 'B0I')
AXES = ('cosine', 'sine')
PROFILE = '622ce4a8391002558b877240b48d831d9bb0a72aeabba2e4d73bfec64b9c9f12'
COMPILER = '1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b'
REQUIRED = ('input_bound', 'compiler_bound', 'runtime_profile_bound',
            'cost_preparation_bound', 'control_scalar_native_bound',
            'successful_measured_wrapper')


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def eligible(row):
    return (all(row['evidence'].get(k) is True for k in REQUIRED)
            and row['runtime_profile_sha256'] == PROFILE
            and row['compiler_hash'] == COMPILER)


def inspect_anchors(rows):
    """Count provenance-complete anchors; never compute design rank or residuals."""
    seen = set()
    counts = {m + '/' + a: 0 for m in METHODS for a in AXES}
    for row in rows:
        key = (row['method'], row['prefix'], row['q'], row['axis'])
        require(key not in seen, 'DUPLICATE_ANCHOR')
        seen.add(key)
        require(row['method'] in METHODS and row['axis'] in AXES, 'STRATUM')
        require(row['formula'] == '2nd' and row['prefix'] in (14, 21)
                and row['q'] in (1, 2, 4, 8), 'ANCHOR_GRID')
        require(type(row['saved_RZ']) is int and row['saved_RZ'] > 0, 'MEASURED_RZ')
        require(row['control'] == 'symmetric_directional', 'CONTROL')
        if eligible(row):
            counts[row['method'] + '/' + row['axis']] += 1
    expected = {(m, p, q, a) for m in METHODS for p in (14, 21)
                for q in (1, 2, 4, 8) for a in AXES}
    require(seen == expected, 'ANCHOR_INVENTORY')
    # Preserve the published pre-registration: >=3 non-collinear anchors.
    # Here we check only its count prerequisite, never actual design rank.
    complete = all(n >= 3 for n in counts.values())
    return dict(status='PROVENANCE_READY_PENDING_APPROVAL_AND_DESIGN_CHECK' if complete
                else 'STOP_MISSING_REQUIRED_ANCHOR_PROVENANCE',
                measured_axes=len(rows), provenance_eligible_axes=sum(counts.values()),
                counts=counts, actual_design_rank_evaluated=False,
                actual_fit_performed=False)


def require_full_b1_provenance(rows):
    result = inspect_anchors(rows)
    require(result['status'] == 'PROVENANCE_READY_PENDING_APPROVAL_AND_DESIGN_CHECK',
            'STOP_MISSING_REQUIRED_ANCHOR_PROVENANCE')
    return result


def select_b2(cells):
    """Apply the previously registered rule to saved eligibility booleans only."""
    require(len({c['id'] for c in cells}) == len(cells), 'DUPLICATE_S4_CELL')
    selected = []
    for method in METHODS:
        candidates = [c for c in cells if c['method'] == method and c['eligible'] is True]
        for c in candidates:
            require(c['formula'] == '4th' and type(c['prefix']) is int
                    and 15 <= c['prefix'] <= 21 and c['q'] in (1, 2, 4, 8), 'B2_GRID')
            require(c['id'] == f"{method}_p{c['prefix']}_q{c['q']}_4th", 'B2_ID')
        if not candidates:
            continue
        choose = min if method == 'B0' else max
        cell = choose(candidates, key=lambda c: (c['prefix'], c['q']))
        selected.extend(dict(candidate_id=cell['id'], axis=a, predicted_RZ=None,
                             model_sha256=None, compile_authorized=False) for a in AXES)
    return selected


def b2_axis_passes(predicted, measured):
    """Exact artificial-test guard: equality at 2% fails, measured denominator."""
    require(type(measured) is int and measured > 0, 'MEASURED_RZ')
    prediction = Fraction(predicted)
    require(prediction > 0, 'PREDICTION')
    return abs(prediction - measured) / measured < Fraction(1, 50)


def validate_draft(plan):
    a = plan['authorization']
    require(a['approved'] is False and a['runtime_authorization'] is False
            and a['grant_issued'] is False and a['SOURCE'] is None
            and a['allowed_cpus'] == [] and a['command'] is None, 'DRAFT_NOT_EXECUTABLE')
    require(a['mandatory_stop'] is True and a['retry'] is False
            and a['next_stage_authorized'] is False, 'PHASE_BOUNDARY')
    require(plan['models'] == dict(S2='a+b*q+d*prefix*q',
            S4='a+3*b*q+3*d*prefix*q', coefficients=None,
            H6_coefficients_transferred=False), 'MODEL_HYPOTHESES_CHANGED')
    require(plan['required_strata'] == ['B0/cosine', 'B0/sine',
            'B0I/cosine', 'B0I/sine'], 'NO_PARTIAL_FIT_SUBSTITUTION')
    require(plan['B2']['max_axes'] == 4 and plan['B2']['compile_concurrency'] == 1
            and plan['B2']['error_stop'] == 'abs(predicted-measured)/measured >= 1/50'
            and plan['B2']['retry'] is False and plan['B2']['refit'] is False
            and plan['B2']['separate_approval'] is True, 'B2_STOP_POLICY')
    require(plan['C']['separate_approval'] is True, 'C_BOUNDARY')
    require(plan['implementation']['real_fit_runner_sealed'] is False,
            'FIT_RUNNER_NOT_PREPARED')
    return True
