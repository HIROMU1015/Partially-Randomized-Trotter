"""Saved-trajectory sensitivity and descriptive selection diagnostics.

No sampling of quantum trajectories, compiler, shot calculation, or refit.
Integer/Fraction cost arithmetic; Decimal(80) for probabilities and diagnostics.
The caller must validate authorization before supplying scientific inputs.
"""
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import math
import random

from h8_rare_tail_selection import require, digest, paired_cost

STAGES = ('RARE_SENSITIVITY', 'SELECTION_SAVED')
EPS = ('0.05', '0.01', '0.005', '0.001', '0.0005')
MODES = ('USUAL_SIGNED_SAVED', 'EMPIRICAL_NONCANCELLATION')
CANDIDATES = tuple(f'B2_p7_q{q}_2nd_r4_K{k}' for q in (2, 4, 8) for k in (2, 4))
FIXED = dict(zip(EPS[:4], (CANDIDATES[0], CANDIDATES[2], CANDIDATES[2], CANDIDATES[4])))
RARE = (CANDIDATES[3], CANDIDATES[5])
PRECISION = 80


def D(value):
    if isinstance(value, Fraction):
        return Decimal(value.numerator) / Decimal(value.denominator)
    out = Decimal(str(value))
    require(out.is_finite(), 'NONFINITE_VALUE')
    return out


def nonnegative(value):
    out = D(value)
    require(out >= 0, 'NEGATIVE_COST')
    return out


def summary(values):
    require(len(values) >= 2, 'AT_LEAST_TWO_TRAJECTORIES')
    require(all(isinstance(v, (int, Fraction)) and not isinstance(v, bool) and v >= 0
                for v in values), 'COST_INTEGER_OR_FRACTION')
    n = len(values)
    mean = sum(values, Fraction()) / n
    variance = sum(((v - mean)**2 for v in values), Fraction()) / (n - 1)
    with localcontext() as ctx:
        ctx.prec = PRECISION
        return dict(n=n, mean_exact=str(mean), mean=str(D(mean)),
                    sample_variance_exact=str(variance), sample_SD=str(D(variance).sqrt()),
                    sample_SE=str((D(variance)/n).sqrt()),
                    population_mean_guarantee=False)


def weighted_mean(rows, shots):
    require(rows, 'MISSING_LAYER_SAMPLES')
    require(all(all(type(r.get(a)) is int and r[a]>=0 for a in ('cosine','sine')) for r in rows),
            'MISSING_NONFINITE_LAYER_COST')
    return sum((Fraction(paired_cost(r, shots)) for r in rows), Fraction())/len(rows)


def probabilities(law):
    """Preserve saved masses; sum positive saved multi count cells, no renormalization."""
    p = [D(law[k]) for k in ('saved_pi0', 'saved_pi2', 'saved_pi4')]
    require(all(0 <= x <= 1 for x in p) and abs(sum(p)-1) <= Decimal('1e-12'),
            'STRATUM_NORMALIZATION')
    counts = law['conditional_N2_counts']
    vals = [D(v) for v in law['conditional_N2_probabilities']]
    require(counts == list(range(1, law['R']+1)) and len(vals) == len(counts)
            and all(0 <= x <= 1 for x in vals)
            and abs(sum(vals)-1) <= Decimal('1e-12'), 'CONDITIONAL_NORMALIZATION')
    multi = sum(vals[1:], Decimal(0))
    # Evaluate the declared (1-w) decomposition; retain saved rounding residual.
    single = 1-multi
    return dict(pi0=p[0], pi2=p[1], pi4=p[2], single_given_s2=single,
                w=multi, single_mass=p[1]*single, multi_mass=p[1]*multi,
                saved_mass_residual=sum(p)-1, count_mass_residual=sum(vals)-1,
                saved_single_given_s2=vals[0])


def threshold(budget, coefficient, other_coefficient, assumed_other):
    if coefficient == 0:
        return dict(status='ZERO_COEFFICIENT_NO_DIVISION', value=None)
    if assumed_other is None:
        return dict(status='UNKNOWN_OTHER_COST_NO_SINGLE_THRESHOLD', value=None)
    value = (budget - other_coefficient*nonnegative(assumed_other))/coefficient
    return dict(status='NEGATIVE_THRESHOLD_ALREADY_REVERSED_FOR_NONNEGATIVE_COSTS'
                if value < 0 else 'CONDITIONAL_THRESHOLD', value=str(value),
                assumed_other=str(assumed_other), equality='TIE; ABOVE_THRESHOLD_PR_MORE_COSTLY')


def rare_endpoint(law, rows0, rows_single, endpoint, assumptions):
    """Return a two-variable boundary without filling unknown layer means."""
    base = dict(candidate=endpoint['candidate'], endpoint=endpoint['id'],
                bias_kind=endpoint['bias_kind'], epsilon=endpoint['epsilon'],
                factor=endpoint['factor'], shots=endpoint.get('shots'),
                comparator=endpoint['comparator'], saved_PR_winner=endpoint['saved_PR_winner'],
                candidate_is_saved_PR_winner=endpoint['candidate']==endpoint['saved_PR_winner'],
                unobserved_s2_multi_mean=None, unobserved_s4_mean=None,
                rigorous_bias_guarantee=None, population_cost_guarantee=False)
    if not endpoint['eligible']:
        return dict(**base, status='INELIGIBLE_SAVED_ENDPOINT', boundary=None)
    if endpoint.get('shots') is None:
        return dict(**base, status='MISSING_SAVED_SHOTS', boundary=None)
    if endpoint['comparator'].get('cost_exact') is None:
        return dict(**base, status='COMPARATOR_UNAVAILABLE', boundary=None)
    with localcontext() as ctx:
        ctx.prec = PRECISION
        p = probabilities(law)
        mu0 = weighted_mean(rows0, endpoint['shots'])
        single = weighted_mean(rows_single, endpoint['shots'])
        det = nonnegative(Fraction(endpoint['comparator']['cost_exact']))
        baseline = p['pi0']*D(mu0) + p['single_mass']*D(single)
        budget = det - baseline
        am, a4 = p['multi_mass'], p['pi4']
        # Intercepts are coordinates of the boundary, NOT measured zero-cost layers.
        intercepts = dict(multi_at_s4_coordinate_zero=None if am==0 else str(budget/am),
                          s4_at_multi_coordinate_zero=None if a4==0 else str(budget/a4))
        relation = ('BASELINE_ALREADY_EXCEEDS_COMPARATOR' if budget < 0 else
                    'BASELINE_TIES_COMPARATOR; POSITIVE_WEIGHTED_UNKNOWN_COST_REVERSES' if budget == 0
                    else 'PR_LOWER_ONLY_BELOW_BOUNDARY_IN_NONNEGATIVE_COST_DOMAIN')
        return dict(**base, status='BOUNDARY_EVALUATED',
            probabilities={k:str(v) for k,v in p.items()},
            s0=dict(n=len(rows0), mean_G_exact=str(mu0), evidence='ALL_ZERO_ORDER_N32_FINITE_SAMPLE'),
            s2_single=dict(n=len(rows_single), mean_G_exact=str(single),
                evidence='S2_CONDITIONAL_N8_ALL_OBSERVED_N2_EQ1; NOT_DESIGNED_SINGLE_ONLY'),
            known_baseline_G=str(baseline),
            boundary=dict(formula='multi_mass*mu2_multi + pi4*mu4 = remaining_G',
                          multi_coefficient=str(am), s4_coefficient=str(a4), remaining_G=str(budget),
                          unknown_cost_domain='BOTH_NONNEGATIVE_SHOT_WEIGHTED_LAYER_MEANS',
                          intercepts=intercepts, interpretation=relation,
                          unknown_axis_costs_assumed_equal=False),
            mu4_threshold=threshold(budget, a4, am, assumptions.get('mu2_multi')),
            mu2_multi_threshold=threshold(budget, am, a4, assumptions.get('mu4')),
            assumptions=dict(assumptions),
            limitations=['FINITE_SAMPLE_S0_AND_SINGLE_PLUGINS_NOT_POPULATION_MEANS',
                         'UNKNOWN_MULTI_AND_S4_NOT_IMPUTED', 'COMPARATOR_MAY_USE_FIXED_MODEL_COST'])


def validate_trajectories(by, candidates, *, expected_n, exploration_seeds):
    require(set(by) == set(candidates) and len(set(candidates)) == len(candidates), 'CANDIDATE_SET')
    occupied = set()
    for c, rows in by.items():
        require(len(rows) == expected_n, 'CONFIRMATION_SAMPLE_COUNT')
        require(sorted(r['replica'] for r in rows) == list(range(expected_n)), 'TRAJECTORY_REPLICAS')
        for r in rows:
            require(r['candidate']==c and r['phase']=='confirmation' and r['law']=='unconditional',
                    'NO_EXPLORATION_OR_CONDITIONAL_POOLING')
            require(r['seed'] not in occupied and r['seed'] not in exploration_seeds, 'SEED_OVERLAP')
            occupied.add(r['seed'])
            require(r['event_digest_cosine']==r['event_digest_sine'], 'AXIS_PAIRING')
            require(all(type(r[a]) is int and r[a]>=0 for a in ('cosine','sine')), 'SAVED_NATIVE_COST')


def validate_stat_contract(cfg):
    require(cfg['statistical_method'] in ('DESCRIPTIVE_ONLY', 'EMPIRICAL_PRODUCT_BOOTSTRAP'),
            'STATISTICAL_METHOD')
    require(cfg['cross_candidate_pairing'] is False, 'CROSS_CANDIDATE_PAIRING_UNPROVED')
    require(cfg['population_guarantee'] is False, 'NO_ALL_SUPPORT_BOUND_OR_VALID_MARGINAL_INTERVAL')
    require(cfg['family_alpha'] is None, 'NO_CERTIFIED_INTERVAL_FAMILY_ALPHA')
    require(cfg['interval_kind'] == 'EXPLORATORY_EMPIRICAL_PERCENTILES_NOT_CONFIDENCE_INTERVAL',
            'INTERVAL_KIND')
    require(0 < D(cfg['diagnostic_tail_fraction']) < Decimal('.5'), 'DIAGNOSTIC_TAIL')
    require(type(cfg['bootstrap_replicates']) is int and cfg['bootstrap_replicates'] >= 0,
            'BOOTSTRAP_REPLICATES')
    if cfg['statistical_method']=='EMPIRICAL_PRODUCT_BOOTSTRAP':
        require(cfg['bootstrap_replicates']>=2 and cfg['dependence_assumption']==
                'PRODUCT_OF_CANDIDATE_EMPIRICAL_DISTRIBUTIONS_NOT_MEASURED_INDEPENDENCE',
                'BOOTSTRAP_ASSUMPTION')
    else:
        require(cfg['bootstrap_replicates']==0, 'DESCRIPTIVE_BOOTSTRAP_MUST_BE_ZERO')
    require(set(cfg['candidate_set'])==set(CANDIDATES), 'FIXED_UNION6_REQUIRED')


def quantile(sorted_values, probability):
    """Linear empirical quantile, exact rational interpolation (type 7)."""
    require(sorted_values and 0 <= probability <= 1, 'QUANTILE_DOMAIN')
    x = Fraction(probability)*(len(sorted_values)-1)
    i = x.numerator//x.denominator
    return sorted_values[i] if i==len(sorted_values)-1 else (
        sorted_values[i]*(1-(x-i)) + sorted_values[i+1]*(x-i))


def selection_analysis(by, endpoints, cfg, fixed):
    validate_stat_contract(cfg)
    require(len({e['id'] for e in endpoints})==len(endpoints), 'ENDPOINT_IDS')
    output, eligible, values = {}, {}, {}
    for e in endpoints:
        require(set(e['candidates']) == set(by), 'ENDPOINT_CANDIDATE_MASK')
        mask = [c for c in sorted(by) if e['candidates'][c]['eligible']]
        eligible[e['id']] = mask
        summaries = {}
        for c in mask:
            shots = e['candidates'][c]['shots']
            vals = [paired_cost(r, shots) for r in by[c]]
            summaries[c] = summary(vals)
            values[e['id'], c] = vals
            saved=e['candidates'][c].get('saved_G_exact')
            if saved is not None:
                require(Fraction(summaries[c]['mean_exact'])==Fraction(saved), 'SAVED_MEAN_REPRODUCTION')
        chosen = min(mask, key=lambda c:(Fraction(summaries[c]['mean_exact']),c)) if mask else None
        # No retroactive original exploration winner at a new epsilon.
        jfixed = fixed.get(e['epsilon'])
        require(e['epsilon']!='0.0005' or jfixed is None, 'RETROACTIVE_EPSILON_SELECTION')
        fixed_status = ('NOT_IN_ORIGINAL_FOUR_EPSILON_EXPLORATION' if jfixed is None else
                        'SAVED_FIXED_CANDIDATE_INELIGIBLE' if jfixed not in mask else
                        'ORIGINAL_EXPLORATION_FIXED_CONFIRMATION_EVALUATION'
                        if e['bias_kind']==MODES[0] and e['factor']=='1.0' else
                        'ORIGINAL_CANDIDATE_TRANSFERRED_TO_NEW_ENDPOINT_NOT_ORIGINALLY_SELECTED_THERE')
        output[e['id']] = dict(endpoint=e['id'], epsilon=e['epsilon'], bias_kind=e['bias_kind'],
            factor=e['factor'], eligibility_mask={c:c in mask for c in sorted(by)},
            candidate_summaries=summaries, shots={c:e['candidates'][c]['shots'] for c in mask},
            exploration_fixed_candidate=jfixed, exploration_fixed_status=fixed_status,
            fixed_candidate_confirmation=summaries.get(jfixed), post_confirmation_point_selected=chosen,
            post_confirmation_point_min=None if chosen is None else summaries[chosen]['mean_exact'],
            fixed_minus_postselected_point_exact=None if jfixed not in mask or chosen is None else
                str(Fraction(summaries[jfixed]['mean_exact'])-Fraction(summaries[chosen]['mean_exact'])),
            finite_union_population_min=None, selected_candidate_population_mean=None,
            selection_procedure_population_performance=None, rigorous_population_interval=None,
            candidate_cross_covariance=None, bootstrap=None)
    reps=cfg['bootstrap_replicates']
    if reps:
        # Cache per-replicate axis sums once; share the SAME draw at all endpoints.
        rng={c:random.Random(int(digest([cfg['resampling_seed_policy'],c]),16)) for c in sorted(by)}
        dist={e['id']:{c:[] for c in eligible[e['id']]} for e in endpoints}
        minima={e['id']:[] for e in endpoints}
        selected_dist={e['id']:[] for e in endpoints}
        fixed_minus_min={e['id']:[] for e in endpoints}
        fixed_minus_selected={e['id']:[] for e in endpoints}
        optimism={e['id']:[] for e in endpoints}
        frequencies={e['id']:{c:0 for c in eligible[e['id']]} for e in endpoints}
        index_hash=hashlib.sha256()
        for b in range(reps):
            sums={}
            for c, rows in sorted(by.items()):
                indices=[rng[c].randrange(len(rows)) for _ in rows]
                index_hash.update((digest([b,c,indices])+'\n').encode())
                sums[c]=(sum(rows[i]['cosine'] for i in indices),sum(rows[i]['sine'] for i in indices))
            for e in endpoints:
                key=e['id'];means={}
                for c in eligible[key]:
                    shots=e['candidates'][c]['shots'];a,baxis=sums[c]
                    means[c]=Fraction(shots['cosine']*a+shots['sine']*baxis,len(by[c]))
                    dist[key][c].append(means[c])
                if means:
                    chosen=min(means,key=lambda c:(means[c],c))
                    frequencies[key][chosen]+=1;minima[key].append(means[chosen])
                    # Distribution of the observed-selected candidate with its ID fixed.
                    selected_dist[key].append(means[output[key]['post_confirmation_point_selected']])
                    jfixed=output[key]['exploration_fixed_candidate']
                    if jfixed in means:
                        fixed_minus_min[key].append(means[jfixed]-means[chosen])
                        fixed_minus_selected[key].append(means[jfixed]-
                            means[output[key]['post_confirmation_point_selected']])
                    empirical=Fraction(output[key]['candidate_summaries'][chosen]['mean_exact'])
                    optimism[key].append(empirical-means[chosen])
        tail=Fraction(cfg['diagnostic_tail_fraction'])
        def interval(v):
            if not v:return None
            v=sorted(v)
            return dict(lower_exact=str(quantile(v,tail)),upper_exact=str(quantile(v,1-tail)),
                        kind='EXPLORATORY_EMPIRICAL_PERCENTILES_NOT_CONFIDENCE_INTERVAL')
        for e in endpoints:
            key=e['id']
            output[key]['bootstrap']=dict(replicates=reps,
                candidate_empirical_intervals={c:interval(v) for c,v in dist[key].items()},
                resampled_point_min_interval=interval(minima[key]),
                observed_selected_ID_fixed_interval=interval(selected_dist[key]),
                fixed_minus_resampled_min_interval=interval(fixed_minus_min[key]),
                fixed_minus_observed_selected_ID_fixed_interval=interval(fixed_minus_selected[key]),
                empirical_selection_optimism_mean_exact=None if not optimism[key] else
                    str(sum(optimism[key],Fraction())/reps),
                empirical_selection_optimism_estimand='ORIGINAL_EMPIRICAL_MEAN_OF_RESAMPLED_WINNER_MINUS_ITS_RESAMPLED_MEAN; NOT_TRUE_POPULATION_OPTIMISM',
                winner_counts=frequencies[key], winner_frequency_interpretation=
                    'EMPIRICAL_RESAMPLING_FREQUENCY_NOT_PROBABILITY_OF_POPULATION_BEST',
                population_coverage_certified=False, selection_adjusted_coverage_certified=False,
                missing_tail_recovered=False)
    else:
        index_hash=None
    return dict(status='COMPLETED_DESCRIPTIVE_EXPLORATORY_ONLY', endpoints=list(output.values()),
        analysis_family=cfg['analysis_family'], marginal_count=sum(len(v) for v in eligible.values()),
        bootstrap_replicates=reps, shared_resampling_indices_digest=None if index_hash is None else index_hash.hexdigest(),
        cross_candidate_dependence='PRODUCT_EMPIRICAL_RESAMPLING_ASSUMPTION' if reps else 'NOT_ESTIMATED',
        covariance_measured_zero=False, simultaneous_coverage_certified=False,
        all_support_bound=None, strict_bias_guarantee=None, independent_review=False)
