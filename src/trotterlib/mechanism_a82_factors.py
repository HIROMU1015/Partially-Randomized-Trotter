"""One A82 comparison's factors from already validated sides; no side evaluation."""
import copy
import math
import mechanism_explanation as equations

SCHEMA = 'TRACK_A_A82_SAVED_SIDE_FACTORS_V1'
SCOPE = 'TRACK_A_A82_COMPARISON_FACTORS_ONLY'
TARGET = 'A82_OLD_PR'
AXES = equations.AXES
FACTORS = ('nominal_normalization_squared', 'normalization_margin_squared',
           'normalization_upper_squared', 'headroom_squared', 'alpha_log',
           'continuous_shot', 'rounding_factor')
DEFINITION_SOURCE = 'b6716396acc5760fde7cf7cc6f2481b183a08f10'
SIDE_SOURCE = 'a8961c45d7fd0310c1e34c50ffd004dd58898247'
PRIOR_RESULT = '182d7044fb3e61d68d8c7a5dd86d84c0e961ed6c'
IDENTITY = dict(system='H8', geometry='linear_1_Angstrom', basis='STO-3G',
    DF_rank=28, DF_policy='tol-only', T='0.8', P=0, epsilon='0.05',
    bias_mode='USUAL_SIGNED', allowance_factor='1',
    PR_id='B2_p7_q2_2nd_r4_K2', PF_id='B0I_p11_q1_2nd',
    PR_cohort='CONFIRMATION_UNION_POST_SELECTION', n=32)
CALLS = dict(saved_side_factor_passes=1, comparison_axes=2,
    PR_side_evaluation=0, PF_side_evaluation=0, core_arithmetic=0,
    other31_comparison_evaluation=0, statistics=0, sampling=0, signal=0,
    compile=0, ranking=0)
require = equations.require
exp_safe = equations.exp_safe


def validate_saved_lineage(bundle, delta, core, archived):
    """Metadata and saved JSON equality only; no equation or factor arithmetic."""
    require(bundle['schema'] == SCHEMA and bundle['target_ids'] == [TARGET] and
            bundle['axes'] == list(AXES) and bundle['factor_names'] == list(FACTORS),
            'EXACT_ONE_COMPARISON_TWO_AXES_SEVEN_FACTORS')
    require(bundle['identity'] == IDENTITY and bundle['side_source'] == SIDE_SOURCE and
            bundle['prior_result_commit'] == PRIOR_RESULT and
            bundle['definition_source'] == DEFINITION_SOURCE, 'FIXED_IDENTITY_LINEAGE')
    ids = [r['contrast_id'] for r in core]
    require(len(ids) == len(set(ids)) == 32 and
            ids == [r['contrast_id'] for r in archived] and TARGET in ids,
            'UNCHANGED_32_ARCHIVE_JOIN')
    require(all(r['core_status'] == 'CORE_COMPLETE' and
                r['saved_G_exact_match'] is True and
                r['exact_decompositions_match'] is True for r in core), 'SAVED_CORE_COMPLETE')
    old = archived[ids.index(TARGET)]
    require(old['explanation']['sides']['PR']['status'] == 'EXPLANATORY_INPUT_MISSING' and
            old['explanation']['sides']['PF']['status'] == 'EXPLANATION_COMPLETE' and
            old['explanation']['status'] == 'EXPLANATORY_INPUT_MISSING' and
            old['explanation']['ratios'] == {}, 'HISTORICAL_MISSING_RETAINED')
    for row in archived:
        require(row['core_reused_without_recalculation'] is True, 'CORE_NOT_RECALCULATED')
        if row['contrast_id'] != TARGET:
            require(row['explanation']['status'] == 'EXPLANATION_COMPLETE' and
                    set(row['explanation']['ratios']) == set(AXES) and
                    all(row['explanation']['sides'][m]['status'] == 'EXPLANATION_COMPLETE'
                        for m in ('PR', 'PF')), 'OTHER31_SAVED_COMPLETE')
    require(delta['schema'] == 'TRACK_A_A82_PR_EXPLANATION_DELTA_V1' and
            delta['target_side_id'] == TARGET+'.PR' and
            delta['candidate_id'] == IDENTITY['PR_id'] and delta['epsilon'] == '0.05' and
            delta['historical_PR_status'] == 'EXPLANATORY_INPUT_MISSING' and
            delta['old_archives_modified'] is False and
            delta['new_comparison_factors_computed'] is False and
            delta['next_stage_authorized'] is False, 'VALIDATED_A82_DELTA_IDENTITY')
    require(delta['inherited_PF_side'] == old['explanation']['sides']['PF'], 'PF_SAVED_COPY_CHANGED')
    sides = dict(PR=delta['new_PR_side'], PF=delta['inherited_PF_side'])
    for side in sides.values():
        require(side['status'] == 'EXPLANATION_COMPLETE' and set(side['axes']) == set(AXES),
                'SAVED_SIDE_NOT_COMPLETE')
        require(all(k in side['normalization'] for k in
                    ('log_B_nominal', 'log_B_margin', 'log_B_upper')), 'NORMALIZATION_INPUT_MISSING')
        for axis in AXES:
            require(all(k in side['axes'][axis] and side['axes'][axis][k] is not None
                        for k in ('headroom', 'alpha_axis', 'log_Nstar', 'log_kappa')) and
                    side['axes'][axis]['integer_status'] == 'BINARY64_DIAGNOSTIC',
                    'SAVED_AXIS_INPUT_MISSING_OR_ROUTE_CHANGED')
    return sides


def saved_side_factors(sides):
    """Verbatim AST of the existing explain() factor block, no explain_side()."""
    ratios = {}
    pnorm, dnorm = [sides[m]['normalization'] for m in ('PR', 'PF')]
    for axis in AXES:
        p, d = [sides[m]['axes'][axis] for m in ('PR', 'PF')]
        logs = dict(nominal_normalization_squared=2*(pnorm['log_B_nominal']-dnorm['log_B_nominal']),
            normalization_margin_squared=2*(pnorm['log_B_margin']-dnorm['log_B_margin']),
            normalization_upper_squared=2*(pnorm['log_B_upper']-dnorm['log_B_upper']),
            headroom_squared=2*(math.log(d['headroom'])-math.log(p['headroom'])),
            alpha_log=math.log(math.log(2/p['alpha_axis']))-math.log(math.log(2/d['alpha_axis'])),
            continuous_shot=p['log_Nstar']-d['log_Nstar'])
        if p['log_kappa'] is not None and d['log_kappa'] is not None:
            logs['rounding_factor'] = p['log_kappa']-d['log_kappa']
        ratios[axis] = dict(log_factors=logs,
            factors={k: exp_safe(v) for k, v in logs.items()})
    return ratios


def validate_result(result):
    require(result['schema'] == SCHEMA and result['contrast_id'] == TARGET and
            result['identity'] == IDENTITY and result['status'] == 'EXPLANATION_COMPLETE' and
            result['definition_source'] == DEFINITION_SOURCE and
            result['validated_side_source'] == SIDE_SOURCE and
            result['historical_PR_status'] == 'EXPLANATORY_INPUT_MISSING' and
            result['certification'] == 'EMPIRICAL_CONDITIONAL_ONLY' and
            result['calls'] == CALLS and result['side_results_recomputed'] is False and
            result['old_archives_modified'] is False and result['next_stage_authorized'] is False,
            'FACTOR_DELTA_OUTPUT_SCOPE')
    require(set(result['ratios']) == set(AXES), 'OUTPUT_AXES')
    for axis in AXES:
        for key in ('factors', 'log_factors'):
            data = result['ratios'][axis][key]
            require(set(data) == set(FACTORS), 'OUTPUT_FACTOR_SCHEMA')
            require(all(type(v) in (int, float) and math.isfinite(v) and
                        (key != 'factors' or v > 0) for v in data.values()),
                    'NONFINITE_OR_UNREPRESENTABLE_FACTOR_STOP')


def analyze_factors(bundle, delta, core, archived, permit):
    require(bundle['kind'] == permit.kind and
            bundle.get('artificial') is (permit.kind == 'ARTIFICIAL'),
            'EXPLICIT_NUMERICAL_PERMISSION_REQUIRED')
    sides = validate_saved_lineage(bundle, delta, core, archived)
    ratios = saved_side_factors(sides)  # the only new scientific operation
    result = dict(schema=SCHEMA, contrast_id=TARGET, identity=copy.deepcopy(IDENTITY),
        status='EXPLANATION_COMPLETE', ratios=ratios, calls=copy.deepcopy(CALLS),
        definition_source=DEFINITION_SOURCE, validated_side_source=SIDE_SOURCE,
        input_result_digests=copy.deepcopy(bundle['input_result_digests']),
        historical_PR_status='EXPLANATORY_INPUT_MISSING',
        side_results_recomputed=False, old_archives_modified=False,
        certification='EMPIRICAL_CONDITIONAL_ONLY', next_stage_authorized=False)
    validate_result(result)
    return result
