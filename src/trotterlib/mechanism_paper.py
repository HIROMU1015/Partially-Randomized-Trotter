"""Copy saved mechanism evidence into manuscript tables without arithmetic.

This module intentionally does not import the scientific analysis runners,
Fraction, math, numpy, or a statistics library. Numeric cells remain strings.
"""
from collections import Counter
import copy


AXES = ('cosine', 'sine')
MAIN_EPSILON = ('0.05', '0.005', '0.0005')
CONFIRMATION_IDS = ('H8_CONFIRM_1', 'H8_CONFIRM_3')
MISSING_ID = 'A82_OLD_PR'
H4_D_COMMIT = '4e8ad1936363266d1b55bb689c36846e6c3dfcdb'
CORE_COLUMNS = ('G_PR', 'G_PF', 'R_G', 'R_N', 'R_Ceff',
                'reduction_percent_exact', 'saved_SE')
RATIO_COLUMNS = ('R_G', 'R_N', 'R_Ceff', 'reduction_percent_exact', 'saved_SE')
FACTOR_COLUMNS = ('nominal_normalization_squared', 'normalization_margin_squared',
                  'normalization_upper_squared', 'headroom_squared', 'alpha_log',
                  'continuous_shot', 'rounding_factor')
AXIS_COLUMNS = ('R_N_axis', 'R_C_axis', 'w_PF', 'weighted_term')
H4_P_COMMIT = '8a18b277663bf2f8998f79e9b8c0a9e61d2b184c'
H4_P_SOURCE = '73ec1964501f14390d037a378c3149110e273635'


class TranscriptionError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise TranscriptionError(message)


def index(rows, keys):
    indexed = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        require(key not in indexed, 'DUPLICATE_' + '_'.join(keys))
        indexed[key] = row
    return indexed


def value(row, key):
    item = row[key]
    require(isinstance(item, str) and item != '', 'SAVED_STRING_REQUIRED_' + key)
    return item


def build_tables(metadata, core_ratios, core_totals, core_axes,
                 comparison_status, factors, h4_winners):
    """Join fixed saved rows by identity; never evaluate a scientific formula."""
    ids = [r['contrast_id'] for r in metadata]
    require(len(ids) == len(set(ids)) == 32, 'FIXED_32_UNIQUE_IDS')
    for rows in (core_ratios, core_totals, comparison_status):
        require([r['contrast_id'] for r in rows] == ids, 'FIXED_ROW_ORDER_JOIN')
    axes = index(core_axes, ('contrast_id', 'axis'))
    expected_axes = {(cid, axis) for cid in ids for axis in AXES}
    require(set(axes) == expected_axes, 'EXACT_64_AXIS_JOIN')
    factor_index = index(factors, ('contrast_id', 'axis'))
    require(MISSING_ID in ids, 'KNOWN_MISSING_CONTRAST_REQUIRED')
    require(set(factor_index) == {(cid, axis) for cid in ids if cid != MISSING_ID for axis in AXES},
            'EXACT_62_EXPLANATION_JOIN_KEEP_A82_MISSING')
    require(len(h4_winners) == 10 and
            all(w['saved_subset_winner_changed'] is False for w in h4_winners),
            'H4_SAVED_D_10_UNCHANGED')
    h4 = index(h4_winners, ('bias_mode', 'epsilon'))
    output, axis_output, factor_output, plan = [], [], [], []
    for meta, core, total, status in zip(metadata, core_ratios, core_totals, comparison_status):
        cid = meta['contrast_id']
        for saved in (core, total, status):
            require(saved['system'] == meta['system'] and saved['H4_stage'] == meta['H4_stage'],
                    'SYSTEM_STAGE_IDENTITY')
        require(core['core_status'] == total['core_status'] == 'CORE_COMPLETE', 'SAVED_CORE_COMPLETE')
        require(core['PF_cost_class'] == total['PF_cost_class'] == meta['PF_cost_class'], 'PF_CLASS_IDENTITY')
        require(meta['PF_cost_class'] in ('DIRECT_MEASURED', 'VALIDATED_MODEL_ESTIMATE'), 'UNKNOWN_COST_CLASS')
        for key in RATIO_COLUMNS:
            require(value(core, key) == value(total, key), 'SAVED_VALUE_MISMATCH_' + key)
        for key in ('epsilon', 'PR_id', 'PF_id', 'PR_role'):
            require(meta[key] == status[key], 'EXPLANATION_METADATA_JOIN_' + key)
        require(status['core_reused_without_recalculation'] == 'True', 'CORE_REUSE_FLAG')
        require(meta['cohorts_pooled'] == 'False', 'COHORT_POOLING_REJECTED')
        expected_status = 'EXPLANATORY_INPUT_MISSING' if cid == MISSING_ID else 'EXPLANATION_COMPLETE'
        require(status['explanation_status'] == expected_status, 'HISTORICAL_EXPLANATION_STATUS')
        if meta['system'] == 'H4':
            require(meta['H4_stage'] == 'H4_PRE_D_P_VALIDATION', 'KEEP_ORIGINAL_H4_STAGE')
            winner = h4[('USUAL_SIGNED', meta['epsilon'])]
            require(winner['best']['cell_id'] == meta['PF_id'] and
                    winner['best']['G_exact'] == total['G_PF'], 'H4_D_SAVED_CANDIDATE_COST_JOIN')
        if cid in CONFIRMATION_IDS:
            require(meta['system'] == 'H8' and meta['n'] == '48', 'N48_COHORT_IDENTITY')
        row = copy.deepcopy(meta)
        row.update({k: value(total, k) for k in CORE_COLUMNS})
        row.update(core_status=core['core_status'],
                   historical_core_explanation_status=core['explanation_status'],
                   explanation_status=status['explanation_status'],
                   population_conclusion=core['population_conclusion'],
                   H4_new_provenance='D_COMPLETE_P_PENDING' if meta['system'] == 'H4' else 'NOT_APPLICABLE',
                   H4_D_result_commit=H4_D_COMMIT if meta['system'] == 'H4' else '',
                   saved_SE_use='TRANSCRIPTION_ONLY_NO_CI')
        output.append(row)
        common = {k: meta[k] for k in ('contrast_id','system','epsilon','PR_role','PR_id','PF_id',
                                      'PR_cohort','n','PF_cost_class','H4_stage')}
        for axis in AXES:
            original = axes[(cid, axis)]
            require(original['system'] == meta['system'], 'AXIS_SYSTEM_JOIN')
            axis_output.append(dict(common, axis=axis, **{k:value(original, k) for k in AXIS_COLUMNS}))
            if cid != MISSING_ID:
                original = factor_index[(cid, axis)]
                for key in ('system','epsilon','PR_role','PR_id','PF_id'):
                    require(original[key] == meta[key], 'FACTOR_METADATA_JOIN_' + key)
                columns = FACTOR_COLUMNS + tuple('log_' + k for k in FACTOR_COLUMNS)
                factor_output.append(dict(common, axis=axis, **{k:value(original,k) for k in columns}))
        group = ('H8_INDEPENDENT_N48_EXAMPLE' if cid in CONFIRMATION_IDS else
                 'MAIN_FIXED_EPSILON' if meta['epsilon'] in MAIN_EPSILON else 'SUPPLEMENT_RETAINED')
        plan.append(dict(common, display_group=group, explanation_status=status['explanation_status'],
                         missing_display='GAP_WITH_LABEL' if cid == MISSING_ID else 'NONE',
                         PF_marker_fill='filled' if meta['PF_cost_class']=='DIRECT_MEASURED' else 'open'))
    require(set(CONFIRMATION_IDS).issubset(ids), 'BOTH_N48_EXAMPLES_REQUIRED')
    return dict(main_table=output, core_axes=axis_output, explanation_factors=factor_output,
                display_plan=plan, counts=dict(contrasts=len(output), axes=len(axis_output),
                explanation_factor_axes=len(factor_output),
                explanation_status=dict(Counter(r['explanation_status'] for r in output))),
                scientific_arithmetic_performed=False, numeric_cells='EXACT_SAVED_STRINGS')


def add_h4_p_provenance(main_table, saved_p, audit):
    """Append P lineage; P costs/SE remain a separate cohort table."""
    require(saved_p['SOURCE'] == audit['SOURCE'] == H4_P_SOURCE, 'H4_P_SOURCE')
    require(saved_p['scientific_result_sha256'] == audit['scientific_result_sha256'], 'H4_P_RAW_RESULT_SHA')
    require(saved_p['old_exploration_samples_pooled'] is False and
            saved_p['population_mean_certified'] is False, 'H4_P_COHORT_SCOPE')
    require(audit['status']=='COMPLETE_MANDATORY_STOP' and audit['primary_comparisons']==10 and
            audit['diagnostic_comparisons']==4 and audit['actual_calls']['trajectory']==4185 and
            audit['actual_calls']['wrapper_compile']==8370, 'H4_P_FIXED_COMPLETION')
    require(saved_p['n_fixed']==audit['n_fixed'] and len(saved_p['comparisons'])==14, 'H4_P_FIXED_N')
    primary={}
    separate=[]
    for position,record in enumerate(saved_p['comparisons']):
        ep=record['endpoint']
        require(record['status']=='REPRODUCED' and record['population_mean_certified'] is False, 'H4_P_SAVED_STATUS')
        key=(ep['cell_id'],str(ep['epsilon']))
        if ep['role']=='PRIMARY':
            require(key not in primary,'H4_P_DUPLICATE_ENDPOINT');primary[key]=(position,record)
        separate.append(dict(P_endpoint_index=str(position),candidate_id=ep['cell_id'],epsilon=str(ep['epsilon']),
            role=ep['role'],n=str(saved_p['n_fixed'][ep['cell_id']]),cohort='H4_INDEPENDENT_P_FIXED_N_LEGACY_DF12',
            PR_G_exact=record['G_exact'],saved_paired_SE=str(record['paired_SE']),
            PF_id=ep['PF_id'],PF_G_exact=ep['PF_G_exact'],status=record['status'],
            population_mean_certified='False',SOURCE=H4_P_SOURCE,result_commit=H4_P_COMMIT))
    require(len(primary)==10,'H4_P_PRIMARY_10')
    output=copy.deepcopy(main_table)
    for row in output:
        if row['system']=='H4':
            position,record=primary[(row['PR_id'],row['epsilon'])]
            require(row['H4_stage']=='H4_PRE_D_P_VALIDATION' and row['H4_new_provenance']=='D_COMPLETE_P_PENDING',
                    'KEEP_H4_HISTORICAL_PROVENANCE')
            require(row['PF_id']==record['endpoint']['PF_id'] and row['G_PF']==record['endpoint']['PF_G_exact'], 'H4_P_FIXED_PF_JOIN')
            row.update(H4_latest_provenance='D_AND_P_COMPLETE_LEGACY_DF12',H4_P_result_commit=H4_P_COMMIT,
                H4_P_endpoint_index=str(position),H4_P_saved_status=record['status'],
                H4_P_cohort='SEPARATE_INDEPENDENT_FIXED_N_NO_POOLING')
        else:
            row.update(H4_latest_provenance='NOT_APPLICABLE',H4_P_result_commit='',H4_P_endpoint_index='',
                H4_P_saved_status='',H4_P_cohort='')
    return output,separate


def figure_bundle(main_table, axes, factors, display_plan):
    """Select the predeclared display; numeric tokens are copied unchanged."""
    require(len(main_table)==len(display_plan)==32,'FIGURE_32_ROWS')
    metas=index(main_table,('contrast_id',));ax=index(axes,('contrast_id','axis'));fac=index(factors,('contrast_id','axis'))
    require(len(ax)==64 and len(fac)==62,'FIGURE_64_62_AXES')
    require({k for k in ax}=={(cid,a) for cid, in metas for a in AXES},'FIGURE_AXIS_IDENTITY')
    require({k for k in fac}=={(cid,a) for cid, in metas if cid!=MISSING_ID for a in AXES},'A82_EXPLANATION_GAP')
    require(metas[(MISSING_ID,)]['explanation_status']=='EXPLANATORY_INPUT_MISSING','KEEP_A82_MISSING')
    plan=index(display_plan,('contrast_id',));require(set(plan)==set(metas),'FIXED_DISPLAY_JOIN')
    rows=[];examples=[]
    for meta in main_table:
        cid=meta['contrast_id'];common=copy.deepcopy(meta)
        for axis in AXES:
            source=ax[(cid,axis)]
            require(all(source[k]==meta[k] for k in ('system','epsilon','PR_id','PF_id','PR_cohort','n','PF_cost_class')),
                    'FIGURE_AXIS_METADATA')
        if cid in CONFIRMATION_IDS:
            require(meta['n']=='48' and meta['PR_cohort']=='INDEPENDENT_FIXED_CONFIRMATION_N48','FIGURE_N48')
            common['factors']={a:{k:value(fac[(cid,a)],k) for k in FACTOR_COLUMNS} for a in AXES}
            examples.append(common)
        elif meta['epsilon'] in MAIN_EPSILON:
            require(plan[(cid,)]['display_group']=='MAIN_FIXED_EPSILON','FIXED_EPSILON_RULE')
            common['R_C_axis']={a:value(ax[(cid,a)],'R_C_axis') for a in AXES}
            common['R_N_axis']={a:value(ax[(cid,a)],'R_N_axis') for a in AXES}
            rows.append(common)
    require(len(rows)==18 and len(examples)==2,'FIGURE_18_PLUS_2')
    return dict(origin='SAVED_FIXED_EVIDENCE',rows=rows,examples=examples,
        missing_annotation='A82 explanation missing in archived result; recovered input not yet numerically validated.',
        H4_annotation='Legacy DF12: D and independent P complete; plotted costs retain historical n8/n32 cohorts.')
