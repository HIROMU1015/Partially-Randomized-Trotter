"""H6 D1 input binding and supplied-policy connection; no work on import.

No saved files are loaded here. Numerical entries require their separately
permitted scope. They reuse the H4 geometry/shot/accounting implementation.
"""
from copy import deepcopy
import math

from . import h4_df10_rpe_connection as common

BINDING = dict(system='H6', rank=19, rank_policy='tol_only_1e-8',
    hamiltonian='223628b06c794f4a2a7a84db532e19ed60915f7e86898de66347129a5aeaffa9',
    snapshot='99440a59d903dfd6330e786d84a956f1dd8a687a592295a1cd771c839769b005',
    state='664c63350c2510518b0ec561f804641447d39f75e68b960192c31d37540265f3')
D1_CANDIDATES = {
    '76601611270ed9f0677d': (.8, 'FULL', 19, None, None),
    '2606efdab20e3b8522e5': (.8, 'PR', 5, 32, 2),
    'a4f5ffc0f320644dadfa': (1.6, 'FULL', 19, None, None),
    '2d2af6ca830dff14a695': (1.6, 'PR', 7, 16, 2),
}


def _check_candidate(point):
    c = point['candidate']
    if c['system'] != 'H6' or any(point['binding'].get(k) != v for k, v in BINDING.items()):
        raise ValueError('H6 D1 scientific binding differs')
    observed = tuple(c[k] for k in ('T', 'method', 'prefix', 'R_outer', 'K'))
    if (D1_CANDIDATES.get(c['id']) != observed or c['q'] != 1
            or c['delta'] != c['T'] or c['formula'] != '4th(m5_best)'):
        raise ValueError('Candidate differs from the fixed H6 D1 scope')


def inspect_saved_point(point):
    """Read-only inspection of the four fixed D1 candidates, without N/G."""
    result = common.inspect_saved_point(point, expected_system='H6',
        expected_rank=19, expected_binding=BINDING)
    _check_candidate(result)
    return result


def bind_saved_round(point, *, target_id, m, base_time):
    inspect_saved_point(point)
    if base_time not in (.8, .1):
        raise ValueError('D1 base-time candidates are .8 and .1; no automatic search')
    return common.bind_saved_round(point, target_id=target_id, m=m,
        base_time=base_time, expected_system='H6', expected_rank=19,
        expected_binding=BINDING)


def prepare_conditions(*, base_time, energy_ha, alpha_total, beta_total,
                       beta_stat, energy_prior=None, prior_provenance=None):
    """Use an explicit external prior and the existing horizon function.

    No E0 input or prior fitting. Real-target calls are scientific evaluation;
    tests supply artificial energy precision and prior. Missing prior stays
    unresolved before any horizon calculation, with no silent h/beta fallback.
    """
    for key, value in dict(base_time=base_time, energy_ha=energy_ha,
            alpha_total=alpha_total, beta_total=beta_total, beta_stat=beta_stat).items():
        common._finite(value, key, positive=True)
    if not 0 < beta_stat < beta_total < math.pi/3 or alpha_total >= 1:
        raise ValueError('Strict phase and total failure budgets required')
    result = dict(base_time=base_time, energy_ha=energy_ha,
                  alpha_total=alpha_total, beta_total=beta_total,
                  beta_stat=beta_stat, horizon=None, alpha_axis=None,
                  full_rpe_energy_success=False)
    if (energy_prior is None or not isinstance(prior_provenance, str)
            or not prior_provenance.strip()):
        return dict(result, status='UNRESOLVED_ENERGY_PRIOR')
    if (len(energy_prior) != 2 or any(isinstance(v, bool) or not math.isfinite(v)
            for v in energy_prior) or energy_prior[0] >= energy_prior[1]):
        raise ValueError('Finite ordered external energy interval required')
    prior = tuple(energy_prior)
    if max(abs(v) for v in prior)*base_time + beta_total >= math.pi:
        return dict(result, status='ALIAS_WINDOW_EXCEEDED', energy_prior=prior,
                    prior_provenance=prior_provenance)
    horizon = common.existing._legacy('rpe_target_round_horizon_validation').required_rpe_round_horizon(
        target_energy_precision=energy_ha, beta_rpe=beta_total,
        delta_time=base_time)
    axes = 2*horizon['round_count_M_plus_one']
    alpha = alpha_total/axes
    while math.fsum([alpha]*axes) > alpha_total:
        alpha = math.nextafter(alpha, 0.)
    return dict(result, status='CONDITIONAL_RPE_POLICY', horizon=horizon,
                alpha_axis=alpha, energy_prior=prior,
                prior_provenance=prior_provenance,
                qualification='CONDITIONAL_ON_SUPPLIED_EXTERNAL_PRIOR')


def assess_supplied_round(row, *, conditions, law_model, **evidence):
    """Reuse H4 supplied-disk geometry; no disk is inferred from saved bias."""
    inspect_saved_point(row)
    if conditions['status'] != 'CONDITIONAL_RPE_POLICY':
        return dict(status=conditions['status'], m=row['m'],
                    target_id=row['target_id'], N_X=None, N_Y=None)
    if (row['base_time'] != conditions['base_time']
            or not 0 <= row['m'] <= conditions['horizon']['maximum_round_index_M']
            or row['q_RPE'] != 1 << row['m'] or row['q_PF'] != row['candidate']['q']
            or math.ldexp(row['base_time'], row['m']) != row['candidate']['T']):
        raise ValueError('Round differs from the supplied common RPE task')
    policy = dict(beta_total=conditions['beta_total'], beta_stat=conditions['beta_stat'],
        alpha_X=conditions['alpha_axis'], alpha_Y=conditions['alpha_axis'],
        law_model=law_model)
    result = common.assess_supplied_round(row, policy=policy,
        df_accounting='own_DF_target_no_DF_energy_shift', **evidence)
    return dict(result, energy_target='saved_DF_H_ground_energy',
                energy_ha=conditions['energy_ha'], prior_provenance=conditions['prior_provenance'])


def supplied_price_axes(row, *, cohort):
    """Select one frozen cohort and copy its X/Y prices; never average/pool."""
    inspect_saved_point(row)
    matches = [p for p in row['prices'] if p['cohort'] == cohort]
    if len(matches) != 1:
        raise ValueError('Choose one explicit saved cohort, without pooling')
    p = matches[0]
    if p['classification'] not in common.PRICE_KINDS[:2]:
        raise ValueError('H6 D1 uses saved direct prices only')
    return [dict(deepcopy(p), m=row['m'], axis=axis,
                 mean=deepcopy(p['mean_metrics'][saved]),
                 peak_qubits=p.get('peak_qubits'))
            for axis, saved in zip(common.AXES, ('cosine', 'sine'))]


def aggregate_partial(plans, prices, *, conditions, compiler):
    """Cost of saved .8/1.6 rounds, explicitly marking all missing rounds."""
    plans, prices = tuple(plans), tuple(prices)
    if conditions['status'] != 'CONDITIONAL_RPE_POLICY':
        return dict(status=conditions['status'], totals=None)
    if conditions['base_time'] not in (.8, .1):
        raise ValueError('D1 base-time candidates are .8 and .1')
    expected = (0, 1) if conditions['base_time'] == .8 else (3, 4)
    if len({p['candidate']['method'] for p in plans}) > 1:
        raise ValueError('PF and PR need separate partial cost evaluations')
    for p in plans:
        _check_candidate(p)
        if (p['base_time'] != conditions['base_time']
                or p['energy_ha'] != conditions['energy_ha']
                or p['alpha_X'] != conditions['alpha_axis']
                or p['alpha_Y'] != conditions['alpha_axis']):
            raise ValueError('Partial plan differs from the uniform full-task allocation')
    result = common.aggregate_supplied_axes(plans, prices, expected_rounds=expected,
        alpha_total=conditions['alpha_total'], compiler=compiler, partial_rounds=True)
    missing = tuple(m for m in range(conditions['horizon']['maximum_round_index_M']+1)
                    if m not in expected)
    return dict(result, partial_only=True, missing_rounds=missing, full_horizon_evaluated=False,
                full_rpe_energy_success=False)
