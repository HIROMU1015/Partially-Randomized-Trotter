"""A82 PR-side delta only. Reuses fixed explanation equations unchanged."""
import copy
import mechanism_explanation as equations

SCHEMA = 'TRACK_A_A82_PR_EXPLANATION_DELTA_V1'
SCOPE = 'TRACK_A_A82_OLD_PR_SIDE_ONLY'
TARGET = 'A82_OLD_PR'
SIDE = TARGET + '.PR'
CANDIDATE = 'B2_p7_q2_2nd_r4_K2'
AXES = ('cosine', 'sine')
CALLS = dict(explain_side=1, PR_axes=2, PF_evaluation=0, core_arithmetic=0,
             other_side_evaluation=0, ratio_or_factor_calculation=0,
             statistics=0, sampling=0, signal=0, compile=0, ranking=0)
require = equations.require


def validate_inputs(bundle, saved_core, saved_explanations, permit):
    require(bundle['schema'] == SCHEMA and bundle['kind'] == permit.kind, 'A82_SCHEMA_KIND')
    row = bundle['comparison']
    equations.original.check_permission(row, permit)
    require(row['contrast_id'] == TARGET and row['system'] == 'H8' and
            row['epsilon'] == '0.05' and row['PR_id'] == CANDIDATE and row['PF_id']=='B0I_p11_q1_2nd' and
            row['PR_role'] == 'OLD_PR' and row['PR_cohort'] == 'CONFIRMATION_UNION_POST_SELECTION',
            'A82_FIXED_ENDPOINT')
    require(row['binding']['rank'] == 28 and row['binding']['T'] == '0.8', 'A82_DF_T_IDENTITY')
    equations.validate_identity(row)  # metadata only, evaluates no equation
    ids = [r['contrast_id'] for r in saved_core]
    require(len(ids) == len(set(ids)) == 32 and
            ids == [r['contrast_id'] for r in saved_explanations] and TARGET in ids,
            'FIXED_32_ARCHIVE_IDENTITY')
    original = saved_explanations[ids.index(TARGET)]
    require(original['explanation']['sides']['PR']['status'] == 'EXPLANATORY_INPUT_MISSING' and
            original['explanation']['sides']['PF']['status'] == 'EXPLANATION_COMPLETE',
            'HISTORICAL_A82_STATUS')
    for record in saved_explanations:
        require(record['core_reused_without_recalculation'] is True, 'CORE_INHERITANCE_FLAG')
        for method in ('PR', 'PF'):
            expected = 'EXPLANATORY_INPUT_MISSING' if (record['contrast_id'], method) == (TARGET, 'PR') else 'EXPLANATION_COMPLETE'
            require(record['explanation']['sides'][method]['status'] == expected, 'ARCHIVED_63_SIDE_STATUS')
    target_core = saved_core[ids.index(TARGET)]
    require(target_core['core_status'] == 'CORE_COMPLETE' and
            target_core['saved_G_exact_match'] is True and
            target_core['exact_decompositions_match'] is True, 'FIXED_CORE_STATUS')
    return row, original


def analyze_delta(bundle, saved_core, saved_explanations, permit):
    row, archived = validate_inputs(bundle, saved_core, saved_explanations, permit)
    # Exactly one equation call. Never call explain(), analyze_bundle(), or PF equations.
    result = equations.explain_side(copy.deepcopy(row['sides']['PR']), row['epsilon'])
    require(result['status'] == 'EXPLANATION_COMPLETE' and set(result['axes']) == set(AXES),
            'A82_NUMERIC_MISMATCH_OR_MISSING_STOP')
    return dict(schema=SCHEMA, target_side_id=SIDE, candidate_id=CANDIDATE,
        epsilon='0.05', new_PR_side=result,
        inherited_PF_side=copy.deepcopy(archived['explanation']['sides']['PF']),
        historical_PR_status='EXPLANATORY_INPUT_MISSING',
        old_archives_modified=False, new_comparison_factors_computed=False,
        calls=copy.deepcopy(CALLS), next_stage_authorized=False)
