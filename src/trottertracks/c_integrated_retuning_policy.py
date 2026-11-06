"""Selection-only adapter for the proposed integrated research-C experiment.

Inputs are already qualified, accounted selection-cohort summaries. This module
does no molecular loading, sampling, PF/RPE evaluation, accounting or compilation.
The executor and scientific execution remain separate work/permission boundaries.
"""
import math

R_VALUES = (16, 32, 64)
TIE_ORDER = (32, 16, 64)
SELECTION_SAMPLES = 32
FINAL_SAMPLES = 64


def select_r(rows, candidate_sha256, *, r_values=R_VALUES, tie_order=TIE_ORDER):
    """Choose the least selection-cohort total RZ among all eligible fixed r.

    Missing policy opportunities and correctness failures are errors. Scientific
    ineligibility is retained and may yield no selection. Final-cohort data must
    never be supplied here. Tie preference keeps the common-r32 baseline first.
    """
    rows = list(rows)
    if (len(set(r_values)) != len(r_values) or set(tie_order) != set(r_values)
            or any(type(r) is not int or r < 1 for r in r_values)):
        raise ValueError('FIXED_R_GRID_AND_TIE_ORDER')
    if len(rows) != len(r_values) or {x['r'] for x in rows} != set(r_values):
        raise ValueError('ALL_THREE_FIXED_R_REQUIRED')
    eligible = []
    for row in rows:
        if row['candidate_sha256'] != candidate_sha256:
            raise ValueError('MIXED_CANDIDATE')
        if row['split'] != 'selection' or row['samples'] != SELECTION_SAMPLES:
            raise ValueError('SELECTION_COHORT_ONLY')
        if row['correctness'] != 'PASS':
            raise ValueError('CORRECTNESS_FAILURE')
        if type(row['eligible']) is not bool:
            raise ValueError('ELIGIBILITY_BOOLEAN_REQUIRED')
        if not row['eligible']:
            if row['total_rz'] is not None:
                raise ValueError('INELIGIBLE_COST_MUST_BE_NULL')
            continue
        value = row['total_rz']
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError('INVALID_ACCOUNTED_TOTAL')
        eligible.append(row)
    if not eligible:
        return {'status': 'UNRESOLVED_NO_ELIGIBLE_R', 'selected_r': None}
    best = min(eligible, key=lambda x: (x['total_rz'], tie_order.index(x['r'])))
    return {'status': 'SELECTION_FIXED', 'selected_r': best['r'],
            'selection_total_rz': best['total_rz'], 'candidate_sha256': candidate_sha256}


def final_pairs(fixed_candidate, tuned_candidate, selected_r):
    """Deduplicate identical candidate/r pairs for independent final cohorts.

    Fixed-r and retuned strength selections may choose different candidates. A
    common r value therefore does not imply reusable final data. When the exact
    pair overlaps it is one shared cohort, not two independent replications.
    """
    if isinstance(selected_r, bool) or selected_r not in R_VALUES:
        raise ValueError('REGISTERED_R_ONLY')
    if not all(isinstance(x, str) and x for x in (fixed_candidate, tuned_candidate)):
        raise ValueError('CANDIDATE_IDENTITY_REQUIRED')
    fixed = (fixed_candidate, 32)
    tuned = (tuned_candidate, selected_r)
    return (fixed,) if fixed == tuned else (fixed, tuned)
