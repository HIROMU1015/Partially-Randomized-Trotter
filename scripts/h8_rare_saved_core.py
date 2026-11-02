"""Fixed two-cell saved-cost integration. No molecular/runtime imports or RNG."""
from decimal import Decimal, localcontext
from fractions import Fraction as F
from h8_rare_c05.accounting import sample_statistics

STRATA = ('S0_ONLY', 'S2_SINGLE', 'S2_MULTI', 'S4_PRESENT')
CELLS = {'H8_0.05': 'B2_p8_q2_2nd_r4_K4',
         'H8_0.0005': 'B2_p7_q16_2nd_r4_K4'}
SHARES = (F(0), F(1, 4), F(1, 2), F(3, 4), F(1))


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def dec(x):
    x = F(x)
    with localcontext() as ctx:
        ctx.prec = 100
        return str(Decimal(x.numerator) / Decimal(x.denominator))


def root(x):
    x = F(x)
    with localcontext() as ctx:
        ctx.prec = 100
        return str((Decimal(x.numerator) / Decimal(x.denominator)).sqrt())


def near(a, b, tolerance=F(1, 10**85)):
    return abs(F(a) - F(b)) <= tolerance * max(F(1), abs(F(a)), abs(F(b)))


def integrate(probabilities, zero_values, single_values, pf, previous):
    """Exact algebra on the stored decimal masses, not a new RTE law evaluation."""
    p = [F(probabilities[s]) for s in STRATA]
    require(all(x > 0 for x in p) and abs(sum(p)-1) < F(1, 10**90), 'STORED_MASS_DOMAIN')
    stats = [sample_statistics(v) for v in (zero_values, single_values)]
    require(all(s['n'] >= 2 for s in stats), 'INSUFFICIENT_KNOWN_N')
    means = [F(s['mean']) for s in stats]
    contributions = [p[j]*means[j] for j in (0, 1)]
    var_terms = [p[j]**2*F(stats[j]['variance'])/stats[j]['n'] for j in (0, 1)]
    known, vknown = sum(contributions), sum(var_terms)
    se = F(root(vknown))  # 100-digit numerical sqrt, not an exact algebraic number.
    gap = F(pf)-known
    residual = p[2]+p[3]
    old_mass = p[1]+residual
    old_gap = F(pf)-contributions[0]
    require(near(previous['observed_plugin_G'], contributions[0]) and
            near(previous['unknown_mass'], old_mass) and
            near(previous['unknown_mixture_mean_threshold_G'], old_gap/old_mass),
            'STAGE_B_REPRODUCTION')
    known_layers = []
    for j in (0, 1):
        known_layers.append(dict(stratum=STRATA[j], probability=probabilities[STRATA[j]],
                                 **stats[j], mean_decimal=dec(means[j]),
                                 contribution_G=dec(contributions[j]),
                                 contribution_SE_G=root(var_terms[j]),
                                 estimated_variance_of_contribution=dec(var_terms[j])))
    endpoints = {STRATA[j]: gap/p[j] for j in (2, 3)}
    boundary = [dict(multi_budget_share=str(t),
                     mu_MULTI_G=dec(t*endpoints['S2_MULTI']),
                     mu_S4_G=dec((1-t)*endpoints['S4_PRESENT']),
                     mu_MULTI_over_S0=dec(t*endpoints['S2_MULTI']/means[0]),
                     mu_S4_over_S0=dec((1-t)*endpoints['S4_PRESENT']/means[0]),
                     hypothetical_boundary_coordinate=True) for t in SHARES]
    shifts = []
    for k in (0, 1, 2, 3):
        lo, hi = gap-k*se, gap+k*se
        shifts.append(dict(k=k, coverage_probability=None,
                           K_known_low=dec(known-k*se), K_known_high=dec(known+k*se),
                           residual_budget_low=dec(lo), residual_budget_high=dec(hi),
                           mixture_threshold_low=dec(lo/residual), mixture_threshold_high=dec(hi/residual),
                           multi_intercept_low=dec(lo/p[2]), multi_intercept_high=dec(hi/p[2]),
                           S4_intercept_low=dec(lo/p[3]), S4_intercept_high=dec(hi/p[3])))
    return dict(known_layers=known_layers,
                K_known_G=dec(known), estimated_variance_K_known_G=dec(vknown),
                SE_K_known_G=root(vknown), known_SE_over_PF=dec(se/F(pf)),
                remaining_budget_D_G=dec(gap), headroom_in_known_SE=dec(gap/se),
                K_known_over_PF=dec(known/F(pf)),
                residual_mass=dec(residual), residual_means={'S2_MULTI': None, 'S4_PRESENT': None},
                residual_variances={'S2_MULTI': None, 'S4_PRESENT': None},
                residual_mixture_mean_threshold_G=dec(gap/residual),
                residual_mixture_threshold_over_S0=dec(gap/residual/means[0]),
                two_layer_boundary=dict(equation='p_MULTI*mu_MULTI + p_S4*mu_S4 = D',
                                        p_MULTI=probabilities['S2_MULTI'], p_S4=probabilities['S4_PRESENT'],
                                        D=dec(gap), multi_intercept_G=dec(endpoints['S2_MULTI']),
                                        S4_intercept_G=dec(endpoints['S4_PRESENT']),
                                        multi_intercept_over_S0=dec(endpoints['S2_MULTI']/means[0]),
                                        S4_intercept_over_S0=dec(endpoints['S4_PRESENT']/means[0]),
                                        p_MULTI_over_p_S4=dec(p[2]/p[3]), points=boundary),
                known_error_sensitivity=shifts,
                stage_B_comparison=dict(previous_unknown_mass=dec(old_mass),
                                        newly_estimated_mass=probabilities['S2_SINGLE'],
                                        fraction_of_previous_unknown_mass_measured=dec(p[1]/old_mass),
                                        unknown_mass_reduction_factor=dec(old_mass/residual),
                                        previous_mixture_threshold_G=previous['unknown_mixture_mean_threshold_G'],
                                        previous_threshold_over_S0=previous['threshold_over_observed_S0_mean'],
                                        acquired_single_contribution_G=dec(contributions[1]),
                                        fraction_of_previous_headroom_used=dec(contributions[1]/old_gap),
                                        single_mean_over_S0=dec(means[1]/means[0]),
                                        single_contribution_SE_over_known_SE=dec(F(root(var_terms[1]))/se),
                                        reduction_in_full_mean_uncertainty=None),
                fixed_PF_G_exact=str(pf),
                full_population_mean_G=None, full_population_variance_G=None,
                full_population_mean_SE=None, law_transfer_bias_bound=None,
                population_ranking_guaranteed=False, mathematical_cost_upper_bound=None,
                unknown_cost_zero_filled=False, cohorts_pooled=False,
                inference='EMPIRICAL_KNOWN_MEANS_CONDITIONAL_RESIDUAL_SENSITIVITY')


def analyze_cell(b, pilot, saved_pilot, law_gate):
    cid, candidate = b['condition'], b['candidate']
    require(cid in CELLS and candidate == CELLS[cid] == pilot['candidate'], 'FIXED_CELL')
    require(b['system']=='H8' and b['rank']==28 and b['r']==4 and b['K']==4 and b['n']==48,
            'FIXED_S0_IDENTITY')
    require(pilot['condition']==cid and pilot['shots']==b['shots'], 'SHOTS_COHORT_IDENTITY')
    require(law_gate['status']=='PASS' and law_gate['diagnostic']['engineering_gate_passed'] and
            not law_gate['diagnostic']['compiled_cost_bias_bounded'] and law_gate['law']==b['law']['saved_law'],
            'OLD_LAW_EVIDENCE')
    require(law_gate['q']==b['q'] and law_gate['r']==4 and law_gate['K']==4 and law_gate['T']==0.8,
            'LAW_CELL')
    zero = []
    for row in b['tasks']:
        require(row['stratum']=='S0_ONLY' and row['order2']==row['order4']==0, 'S0_STRATUM')
        g = sum(b['shots'][a]*row['RZ_'+a] for a in ('cosine', 'sine'))
        require(F(row['G_exact'])==g and row['candidate']==candidate, 'S0_PAIRED_ACCOUNTING')
        zero.append(g)
    single, seen = [], set()
    for row in pilot['rows']:
        require(row['task_id'] not in seen and row['candidate']==candidate and
                row['stratum']=='S2_SINGLE' and not row['scientific_mock'] and not row['normalization_changed'],
                'SINGLE_IDENTITY_OR_DUPLICATE')
        seen.add(row['task_id'])
        require(row['orders'].count(2)==1 and row['orders'].count(4)==0 and
                len(row['orders'])==b['q']*4 and set(row['orders'])<={0,2}, 'SINGLE_ORDER_SUPPORT')
        require(all(row['axes'][a]['event_digest']==row['event_digest'] for a in ('cosine', 'sine')),
                'SINGLE_PAIRED_EVENT')
        single.append(sum(b['shots'][a]*row['axes'][a]['metrics']['RZ'] for a in ('cosine', 'sine')))
    require(len(zero)==48 and len(single)==43, 'FIXED_48_43')
    probabilities = {g['stratum']: g['probability'] for g in b['groups']}
    out = integrate(probabilities, zero, single, b['fixed_PF_G_exact'], b['conditional_expectation'])
    for s, saved, keys in [(out['known_layers'][0], b['groups'][0]['G'], ('mean_exact','sample_variance_exact')),
                           (out['known_layers'][1], saved_pilot['groups']['S2_SINGLE'], ('mean','variance'))]:
        require(F(s['mean'])==F(saved[keys[0]]) and F(s['variance'])==F(saved[keys[1]]) and
                near(s['SE'], saved['SE']), 'SAVED_STATISTICS_EXACT_REPRODUCTION')
    out.update(condition=cid, candidate=candidate, probabilities=probabilities,
               fixed_PF=b['fixed_PF'], shots=b['shots'],
               S0_cohort=b['cohort'], single_cohort='C1_CONDITIONAL_S2_SINGLE_PILOT_N43',
               stored_law_evidence=law_gate,
               assumptions=['Within-stratum iid conditional costs and finite variance',
                            'Independence between the old unconditional P draws conditioned on n0=48 and new pilot streams',
                            'Seed noncollision is engineering evidence, not a proof of PRNG independence',
                            'Canonical layer masses fixed to saved Stage B decimals; component-law transfer bias not bounded',
                            'Fixed analytic shots/PF accounting/compiler/input identities; no candidate reselection'],
               statistical_coverage_probability=None,
               true_mean_expression='E[G] = K_known_hat + p_MULTI*mu_MULTI + p_S4*mu_S4 + delta_K',
               delta_K_definition='p_S0*(true_canonical_mu_S0-mean_hat_S0) + p_SINGLE*(true_canonical_mu_SINGLE-mean_hat_SINGLE)',
               covariance_if_not_independent='Var(Khat)=p0^2*Var(mean0)+p1^2*Var(mean1)+2*p0*p1*Cov(mean0,mean1); covariance unknown')
    return out
