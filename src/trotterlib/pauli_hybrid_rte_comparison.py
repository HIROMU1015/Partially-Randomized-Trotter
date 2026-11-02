"""Prospective saved-value class comparison. Real data use needs execution permission.

The existing finite_minimum convention uses exact Fraction and preserves ties,
but requires guard ranges missing from the H4 CSV. This small variant selects
only saved, eligible finite G; it neither invents guard ranges nor recomputes N.
Importing this module performs no analysis or scientific work.
"""
from fractions import Fraction


def saved_class_minimum(rows, *, molecule, epsilon, group, cohort):
    if group not in ('RTE', 'Hybrid') or cohort not in ('exploration', 'confirmation'):
        raise ValueError('CLASS_OR_COHORT')
    matches = [r for r in rows if (r['molecule'], r['epsilon_sig'], r['group'], r['cohort'])
               == (molecule, epsilon, group, cohort)]
    eligible = [r for r in matches if r['eligibility'] == 'CONDITIONAL_GUARD_ELIGIBLE'
                and r['G_exact'] is not None]
    unresolved = [r['candidate_id'] for r in matches if r['eligibility'] != 'BIAS_INELIGIBLE'
                  and r['G_exact'] is None]
    if not eligible:
        return dict(selected=None, ties=[], G_exact=None, unresolved_ids=unresolved,
                    status='CONFIRMATION_MISSING' if cohort == 'confirmation'
                    else 'NUMERICALLY_OR_ACCOUNTING_UNDETERMINED' if unresolved
                    else 'NO_ELIGIBLE_REGISTERED_CANDIDATE',
                    missing_reason=('CONFIRMED_SET_NO_ELIGIBLE_FINITE_G' if cohort == 'confirmation'
                                    else 'EXPLORATION_SET_NO_ELIGIBLE_FINITE_G') if matches else 'NO_SAVED_ROWS')
    minimum = min(Fraction(r['G_exact']) for r in eligible)
    tied = sorted([r for r in eligible if Fraction(r['G_exact']) == minimum],
                  key=lambda r: r['candidate_id'])
    return dict(selected=tied[0], ties=[r['candidate_id'] for r in tied],
                G_exact=tied[0]['G_exact'], unresolved_ids=unresolved,
                status='FINITE_SAVED_COHORT_OBSERVED_MINIMUM',
                all_registered_parameters_resolved=not unresolved,
                post_selection=True, population_winner_guaranteed=False)


def hybrid_over_rte(hybrid, rte):
    if hybrid['selected'] is None or rte['selected'] is None:
        return None
    h, r = hybrid['selected'], rte['selected']
    if (h['group'], r['group']) != ('Hybrid', 'RTE'):
        raise ValueError('RATIO_ORIENTATION')
    if (h['molecule'], h['epsilon_sig'], h['cohort']) != (r['molecule'], r['epsilon_sig'], r['cohort']):
        raise ValueError('MIXED_MOLECULE_PRECISION_OR_COHORT')
    if Fraction(r['G_exact']) <= 0:
        raise ValueError('NO_POSITIVE_FINITE_DENOMINATOR')
    return str(Fraction(h['G_exact']) / Fraction(r['G_exact']))
