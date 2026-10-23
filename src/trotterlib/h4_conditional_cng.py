"""Finite saved-H4 adapter. No chemistry, state, sampler or compiler imports.

Fractions bind binary64 inputs exactly. Decimal ln/exp are correctly rounded;
one adjacent decimal on each side encloses those scalar operations (96 digits).
All following bound/guard/ceil arithmetic is rational. This scalar safeguard
does not certify the unknown compiled-mean residual or a population cost tail.
"""
import ast
import csv
from decimal import Decimal, localcontext, ROUND_FLOOR, ROUND_CEILING
from fractions import Fraction as F
import importlib.util
import itertools
import math
from pathlib import Path

AXES = ('cosine', 'sine')
THETA = ('0', '1/4', '1/2', '3/4', '1')
G = F(1, 100)
ROOT = Path(__file__).resolve().parents[2]


def extract(path, name, namespace):
    tree = ast.parse((ROOT/path).read_bytes())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(nodes) != 1:
        raise ValueError('PURE_FUNCTION_NOT_UNIQUE:'+name)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT/path), 'exec'), namespace)
    return namespace[name]


def reuse():
    import numpy as np
    spec = importlib.util.spec_from_file_location('cng_saved_accounting', ROOT/'src/trotterlib/fixed_signal_accounting.py')
    acc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(acc)
    stats = extract('src/trotterlib/fixed_signal_evaluation.py', '_statistics',
                    dict(np=np, math=math, Fraction=F))
    from .h4_certificate_control import digest
    join = extract('src/trotterlib/rte_event_audit.py', 'verify_join', dict(digest=digest))
    return acc.fixed_signal_shots, stats, join


def trans_bounds(kind, value):
    """Enclose exp/ln of an exact rational, including conversion rounding."""
    value = F(value)
    ends = []
    with localcontext() as ctx:
        ctx.prec = 96
        for rounding, side in ((ROUND_FLOOR, -1), (ROUND_CEILING, 1)):
            ctx.rounding = rounding
            x = Decimal(value.numerator)/Decimal(value.denominator)
            y = x.exp() if kind == 'exp' else x.ln()
            y = y.next_minus() if side < 0 else y.next_plus()
            ends.append(F(y))
    return tuple(ends)


def scale_bounds(saved_B, saved_log):
    elo, ehi = trans_bounds('exp', F(saved_log))
    blo, bhi = max(F(1), F(saved_B), elo), max(F(1), F(saved_B), ehi)
    llo, lhi = trans_bounds('ln', F(2)/F(.025))
    return dict(B_saved_exact=str(F(saved_B)), log_B_upper_exact=str(F(saved_log)),
                B_up_lower=str(blo), B_up_upper=str(bhi),
                A_lower=str(2*blo*blo*llo), A_upper=str(2*bhi*bhi*lhi))


def base_axis(eps, bias, uenc, scale):
    # Preserve the original floating epsilon allocation; do not reallocate it.
    ea = F(float(eps)/math.sqrt(2))
    s = ea-abs(F(bias))
    u = F(uenc)
    if u < 0:
        raise ValueError('NEGATIVE_SAVED_DISTANCE_UPPER')
    cap = G*s/(1+G)-u
    return dict(epsilon_axis_exact=str(ea), signed_bias_exact=str(F(bias)),
                u_enc_exact=str(u), s_exact=str(s), rho_guard_exact=str(cap),
                rho_guard=float(cap), domain_nonempty=s > 0 and cap >= 0,
                actual_rho=None, scale=scale)


def shot(axis, rho):
    if rho is None:
        return dict(N=None, reason='BASE_GUARD_DOMAIN_EMPTY_OR_UNKNOWN_RHO')
    r = F(rho)
    if r < 0:
        raise ValueError('NEGATIVE_RESIDUAL')
    s, u = F(axis['s_exact']), F(axis['u_enc_exact'])+r
    h = s-u
    if h <= 0 or u/h > G:
        return dict(N=None, reason='FIXED_GUARD_FAILED', rho_exact=str(r), headroom_exact=str(h))
    lo = F(axis['scale']['A_lower'])/h**2
    hi = F(axis['scale']['A_upper'])/h**2
    n = max(1, math.ceil(hi))
    if n >= 2**52:
        return dict(N=None, reason='SAFE_INTEGER_LIMIT', bound_upper_exact=str(hi))
    assert F(n)*h*h >= F(axis['scale']['A_upper'])
    return dict(N=n, rho_exact=str(r), headroom_exact=str(h), u_over_h_exact=str(u/h),
                bound_lower_exact=str(lo), bound_upper_exact=str(hi),
                ceiling_interval_width=math.ceil(hi)-math.ceil(lo),
                sufficient_inequality_verified=True, reason='CONDITIONAL_ONLY')


def paired(prices, weights, stats):
    z = [sum(weights[a]*p[a] for a in AXES) for p in prices]
    out = stats(z)
    if len(prices) == 1:
        out.update(standard_error=0., nominal95_halfwidth=0.)
    assert F(out['mean_exact']) == sum(F(stats([p[a] for p in prices])['mean_exact'])*weights[a] for a in AXES)
    return out


def cost_summary(prices, stats):
    n = len(prices)
    axes = {a:stats([p[a] for p in prices]) for a in AXES}
    mu = {a:F(axes[a]['mean_exact']) for a in AXES}
    cov = {a:{b:sum((F(p[a])-mu[a])*(F(p[b])-mu[b]) for p in prices)/(n-1) if n > 1 else F(0)
              for b in AXES} for a in AXES}
    if n == 1:
        for a in AXES:
            axes[a].update(standard_error=0., nominal95_halfwidth=0.)
    return dict(n=n, axes=axes, covariance_exact={a:{b:str(v) for b,v in row.items()} for a,row in cov.items()},
                observed_min={a:min(p[a] for p in prices) for a in AXES},
                observed_max={a:max(p[a] for p in prices) for a in AXES},
                population_mean_upper=None, nominal_t95_is_tail_bound=False)


def component_key(component):
    return (component.get('basis', {}).get('basis_hash'), tuple(component['support']),
            F(component['coefficient_abs']), component['sign'])


def rare_law(model, normalization, joins):
    """Only metadata/event joins and saved probabilities; no RNG or operator work."""
    c = model['cases']['PR']
    comps = c['components']
    eff = list(map(F, normalization['component_effective_probabilities']))
    order = list(map(F, normalization['order_effective_probabilities']))
    assert len(eff) == len(comps) and sum(eff) == 1 and sum(order) == 1
    assert normalization['IID_uniform_word_assumption'] is True
    assert c['law']['finite_distribution'] == joins[0]['distribution']
    assert all(j['distribution'] == joins[0]['distribution'] and j['q'] == 8 and j['r'] == 4 for j in joins)
    keys = {component_key(x):i for i,x in enumerate(comps)} if model['representation'] == 'df_native' else None
    if keys is not None:
        assert len(keys) == len(comps)
    observed = set()
    counts = {0:0, 2:0, 4:0}
    strata = []
    id_map = {}
    for j in joins:
        has4 = False
        for e in j['events']:
            k = e['taylor_order']; counts[k] += 1; has4 |= k == 4
            assert e['selected_component_ids'] == [a['component_id'] for a in e['application_sequence']]
            for app in e['application_sequence']:
                if keys is None:
                    idx = int(app['component_id'])
                else:
                    idx = keys[(app['basis_hash'], tuple(app['diagonal_pauli_support']), F(app['coefficient_abs']), app['coefficient_sign'])]
                comp = comps[idx]
                assert F(comp['coefficient_abs']) == F(app['coefficient_abs']) and comp['sign'] == app['coefficient_sign']
                assert eff[idx] > 0
                assert app['component_id'] not in id_map or id_map[app['component_id']] == idx
                id_map[app['component_id']] = idx
                observed.add(idx)
        strata.append('any_order4' if has4 else 'remainder')
    unseen = sorted(set(range(len(comps)))-observed)
    mu = sum((eff[i] for i in unseen), F(0))
    M = 32
    p4 = 1-(1-order[2])**M
    no4_noU = (sum(order[i]*(1-mu)**(k+1) for i,k in enumerate((0,2))))**M
    pu = (1-order[2])**M-no4_noU
    probs = dict(any_order4=p4, no4_with_unseen=pu, remainder=no4_noU)
    assert all(p >= 0 for p in probs.values()) and sum(probs.values()) == 1
    nominal_order = c['law']['order_probabilities']
    nominal_components = c['law']['component_probabilities']
    return dict(implemented_probability={k:float(p) for k,p in probs.items()},
                implemented_probability_exact={k:str(p) for k,p in probs.items()},
                implemented_order_mass_exact=list(map(str,order)),
                nominal_order_probability_saved=nominal_order,
                nominal_component_probability_saved=nominal_components,
                unseen_component_indices=unseen, unseen_component_mass_exact=str(mu),
                unseen_selection='DESCRIPTIVE_SAME_DATA_SELECTED_NOT_COVERAGE',
                order_event_counts=counts, trajectory_strata=strata,
                observed_stratum_count={k:strata.count(k) for k in probs},
                stratum_population_mean={k:None for k in probs},
                valid_whole_wrapper_upper=None, PCG64_independence_proved=False,
                probability_assumption='saved effective IID uniform-word product law, not a new RNG proof')


def comparison(left, right, means):
    out = dict(epsilon_sig=left['epsilon_sig'], left=left['case'], right=right['case'],
               primary=left['case'].rsplit('_',1)[0] == right['case'].rsplit('_',1)[0],
               residual_variables='four independent axis residuals in the two guard boxes',
               exact_partition='left cheaper iff sum_a ceil(A_left/h_left_a^2)*mu_left_a < sum_a ceil(A_right/h_right_a^2)*mu_right_a; equality and > are separate regions',
               population_mu_default=None, unconditional_order='UNDETERMINED')
    if not left['domain_nonempty'] or not right['domain_nonempty']:
        return dict(out, status='BASE_GUARD_DOMAIN_EMPTY', empirical_order=None, reversal='NO_JOINT_ADMISSIBLE_DOMAIN')
    def bounds(f):
        low, high = [], []
        for a in AXES:
            ax = f['axes'][a]
            low.append(shot(ax, F(0))['N'])
            # This is the analytic guard face h=s/(1+g), not an extra scenario.
            high.append(shot(ax, F(ax['rho_guard_exact']))['N'])
        if any(n is None for n in low+high):
            return None
        return [sum(F(means[f['case']][a])*v[i] for i,a in enumerate(AXES)) for v in (low,high)]
    lb, rb = bounds(left), bounds(right)
    if lb is None or rb is None:
        return dict(out, status='NUMERICAL_BOUND_UNRESOLVED', empirical_order=None)
    if lb[1] < rb[0]:
        order, reversal = 'LEFT_LOWER_THROUGHOUT_INDEPENDENT_BOXES', 'NONE_WITH_FIXED_EMPIRICAL_MEANS_AND_GUARD'
    elif rb[1] < lb[0]:
        order, reversal = 'RIGHT_LOWER_THROUGHOUT_INDEPENDENT_BOXES', 'NONE_WITH_FIXED_EMPIRICAL_MEANS_AND_GUARD'
    else:
        order, reversal = 'CEIL_PARTITION_DEPENDENT', 'SEE_EXACT_PARTITION_INEQUALITY'
    return dict(out, status='CONDITIONAL_EMPIRICAL_REGION', empirical_order=order, reversal=reversal,
                left_G_range_exact=list(map(str,lb)), right_G_range_exact=list(map(str,rb)),
                left_right_ratio_enclosure=[float(lb[0]/rb[1]),float(lb[1]/rb[0])],
                population_reversal_condition='Replace empirical mu by independent population mu (or stratum mixture); reverse the exact_partition inequality. No finite tail envelope is supplied.')


def csv_save(path, rows):
    import json
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('x', newline='') as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)


def figures(ns, costs, functions, scenarios, contract):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'svg.fonttype':'none'})
    cases=contract['cases']; eps=contract['epsilon_sig']
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,a in zip(axs,AXES):
        ax.bar(range(4),[costs[c]['axes'][a]['mean'] for c in cases],
               yerr=[costs[c]['axes'][a]['nominal95_halfwidth'] for c in cases],capsize=4)
        ax.set_xticks(range(4),cases,rotation=20);ax.set_ylabel('Compiled wrapper RZ / shot');ax.set_title(a+'; PR: nominal t95, PF: deterministic')
    for ext in ('png','svg','pdf'):fig.savefig(ns/('cost_comparison.'+ext),dpi=160)
    plt.close(fig)
    fig,axs=plt.subplots(1,5,figsize=(16,4),layout='constrained')
    for ax,e in zip(axs,eps):
        for c in cases:
            rows=[r for r in scenarios if r['case']==c and r['epsilon_sig']==e and r['G'] is not None]
            if rows:ax.plot([float(F(r['theta'])) for r in rows],[r['G'] for r in rows],marker='.',label=c)
        ax.set_title('epsilon='+e);ax.set_xlabel('Hypothetical theta');ax.set_yscale('log');ax.grid(alpha=.2)
    axs[0].set_ylabel('Conditional empirical G (RZ)');axs[-1].legend(fontsize=7)
    fig.suptitle('Unknown real rho; curves show per-case common-rho slices, absent curves = empty guard domain')
    for ext in ('png','svg','pdf'):fig.savefig(ns/('conditional_G.'+ext),dpi=160)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    data=np.array([[min(float(F(f['axes'][a]['rho_guard_exact'])) for a in AXES)
                    for e in eps for f in functions if f['case']==c and f['epsilon_sig']==e] for c in cases])
    im=ax.imshow(data,aspect='auto',cmap='coolwarm')
    ax.set_xticks(range(5),eps);ax.set_yticks(range(4),cases);ax.set_xlabel('epsilon_sig')
    for i in range(4):
        for j in range(5):ax.text(j,i,format(data[i,j],'.2e')+(' EMPTY' if data[i,j]<0 else ''),ha='center',va='center',fontsize=8)
    fig.colorbar(im,ax=ax,label='min axis rho_guard (signal units)')
    ax.set_title('Residual guard budget; positive budget does not bound the actual residual')
    for ext in ('png','svg','pdf'):fig.savefig(ns/('residual_guard.'+ext),dpi=160)
    plt.close(fig)


def run(namespace, contract, budget):
    from .h4_certificate_control import read, save, hashes, require
    ns = Path(namespace)
    budget.take('saved_input_binding_batches')
    plan = ROOT/'artifacts/non_df_h4_conditional_cng_plan/2026-10-24'
    manifest = read(plan/'inputs.json'); roles = manifest['roles']
    hashes({str(ROOT/k):v for k,v in manifest['files'].items()})
    old = {c:read(ROOT/roles[c+'_old']) for c in contract['cases']}
    models = {r:read(ROOT/roles[r+'_C_model']) for r in ('pauli_jw','df_native')}
    cresults = {r:read(ROOT/roles[r+'_C_result']) for r in models}
    fixed, stats, verify_join = reuse()
    prices, joins = {}, {}
    for case, result in old.items():
        samples = result['circuit_cost']['samples']
        require(len(samples) == (256 if case.endswith('_PR') else 1), 'FIXED_COHORT')
        prices[case] = [s['RZ'] for s in samples]; joins[case] = []
        require(len({s['trajectory_id'] for s in samples}) == len(samples), 'DUPLICATE_TRAJECTORY')
        for s in samples:
            p = Path(roles[case+'_old']).parent/'trajectories'/s['trajectory_id']/'price_join.json'
            require(str(p) in manifest['files'], 'UNBOUND_PRICE')
            j = read(ROOT/p); verify_join(j, expected_case=case)
            require(j['binding']['SOURCE'] == contract['scientific_SOURCE']['original'], 'OLD_SOURCE_JOIN')
            require(j['compiled_RZ'] == s['RZ'] and j['trajectory_record_hash'] == s['trajectory_record_hash'] and j['seed'] == s['seed'], 'PRICE_JOIN')
            require(all(type(x) is int and x >= 0 for x in s['RZ'].values()), 'PRICE_INTEGER')
            joins[case].append(j)
        require(all(r['shots']['N_total'] is None and r['total_RZ'] is None for r in read(ROOT/roles[case+'_precision'])), 'OLD_NULL_CHANGED')
    save(ns/'input_verification.json',dict(status='PASS',files=len(manifest['files']),joins=514,old_N_G_null_preserved=True))
    budget.take('conditional_package_batches')
    budget.take('base_error_and_scale_batches')
    catalog = read(plan/'axis_conditions.json')['axes']
    axisrows, functions = [], []
    for case in contract['cases']:
        rep, method = case.rsplit('_',1)
        result = old[case]
        scale = scale_bounds(result['normalization']['B'], result['normalization']['log_B_upper'])
        for eps in contract['epsilon_sig']:
            axes = {}
            for a,key in zip(AXES,('real','imag')):
                mapping = next(x for x in catalog if (x['case'],x['epsilon_sig'],x['axis']) == (case,eps,a))
                item = cresults[rep]['axes'][mapping['C_axis_pointer'][1]]
                require((item['case'],str(item['epsilon_sig']),item['axis']) == (case,eps,a), 'AXIS_POINTER')
                ax = base_axis(eps, result['signal']['signed_bias'][key], item['bias_distance']['upper'], scale)
                axes[a] = ax; axisrows.append(dict(case=case,epsilon_sig=eps,axis=a,**ax))
            functions.append(dict(case=case,epsilon_sig=eps,axes=axes,domain_nonempty=all(x['domain_nonempty'] for x in axes.values()),
                actual_N=None,actual_G=None,formal_certificate_accepted=False,
                formula='N_a=ceil(A/h_a(rho_a)^2), A=2*B_up^2*ln(2/alpha); G_population=sum_a N_a*mu_C_a',
                interpretation='CONDITIONAL_ONLY_UNKNOWN_REAL_RESIDUAL',residual_default=None))
    save(ns/'conditional_functions.json', functions); csv_save(ns/'guard_thresholds.csv',axisrows)
    budget.take('cost_summary_batches')
    for case in contract['cases']:
        if case.endswith('_PR'):
            budget.take('PR_cost_uncertainty_batches')
    costs = {c:cost_summary(p,stats) for c,p in prices.items()}
    save(ns/'cost_summary.json',costs)
    csv_save(ns/'cost_summary.csv',[dict(case=c,**d) for c,d in costs.items()])
    for case in contract['cases']:
        if case.endswith('_PR'):
            require(costs[case]['n'] == 256,'PR_COHORT')
    scenarios, evaluations = [], []
    for f in functions:
        case, eps, axes = f['case'], f['epsilon_sig'], f['axes']
        cap = min(F(ax['rho_guard_exact']) for ax in axes.values()) if f['domain_nonempty'] else None
        for theta in THETA:
            rho = F(theta)*cap if cap is not None else None
            ev = {a:shot(axes[a],rho) for a in AXES}
            diag = fixed(epsilon_sig=float(eps), bias={a:abs(float(F(axes[a]['signed_bias_exact']))) for a in AXES},
                         allowance={a:float(F(axes[a]['u_enc_exact'])+rho) if rho is not None else None for a in AXES},
                         log_B_upper=old[case]['normalization']['log_B_upper'], evidence_kind='EMPIRICAL', evidence_ref='C_SAVED_PLUS_HYPOTHETICAL_RHO_ONLY')
            for a in AXES:
                evaluations.append(dict(case=case,epsilon_sig=eps,theta=theta,axis=a,**ev[a],old_float_evaluator_N=diag['axes'][a]['shots'],
                                        new_minus_float_N=ev[a]['N']-diag['axes'][a]['shots'] if ev[a]['N'] is not None and diag['axes'][a]['shots'] is not None else None))
            nsamp = {a:ev[a]['N'] for a in AXES}
            row = dict(case=case,epsilon_sig=eps,theta=theta,hypothetical_rho_exact=str(rho) if rho is not None else None,
                       domain_nonempty=f['domain_nonempty'],N_cos=nsamp['cosine'],N_sin=nsamp['sine'],
                       N_total=sum(nsamp.values()) if all(n is not None for n in nsamp.values()) else None,
                       G=None, actual_rho=None, unconditional_N_G=None, status='BASE_GUARD_DOMAIN_EMPTY',hypothetical_not_estimate=True)
            if row['N_total'] is not None:
                gz = paired(prices[case],nsamp,stats)
                cov = costs[case]['covariance_exact']; m = len(prices[case])
                var = sum(nsamp[a]*nsamp[b]*F(cov[a][b]) for a in AXES for b in AXES)/m
                require(math.isclose(gz['standard_error']**2,float(var),rel_tol=2e-12,abs_tol=1e-12), 'PAIRED_COVARIANCE_IDENTITY')
                row.update(G=gz['mean'],G_exact=gz['mean_exact'],G_SE=gz['standard_error'],G_nominal95_halfwidth=gz['nominal95_halfwidth'],
                           paired_variance_of_mean_exact=str(var),status='CONDITIONAL_EMPIRICAL_ONLY')
            scenarios.append(row)
    save(ns/'conditional_results.json',scenarios); csv_save(ns/'conditional_results.csv',scenarios); csv_save(ns/'axis_results.csv',evaluations)
    budget.take('comparison_region_batches')
    means = {c:{a:d['axes'][a]['mean_exact'] for a in AXES} for c,d in costs.items()}
    regions = [comparison(x,y,means) for eps in contract['epsilon_sig'] for x,y in itertools.combinations([f for f in functions if f['epsilon_sig']==eps],2)]
    save(ns/'comparison_regions.json',regions); csv_save(ns/'comparison_regions.csv',regions)
    rare = {}
    for rep in models:
        budget.take('PR_rare_stratum_mass_and_envelope_batches')
        case, pf = rep+'_PR', rep+'_PF'
        d = rare_law(models[rep], cresults[rep]['normalization']['PR'], joins[case])
        d['observed_stratum_cost'] = {s:cost_summary([p for p,k in zip(prices[case],d['trajectory_strata']) if k == s],stats)
            if d['observed_stratum_count'][s] else None for s in d['observed_stratum_count']}
        d['population_G_formula'] = 'sum_h p_h * (N_cos*mu_h_cos+N_sin*mu_h_sin); every mu_h unknown without further assumptions'
        d['rank_flip_sensitivity'] = []
        # Fixed theta=0 already evaluated; no further shot calls. These are cost-only sensitivities.
        for eps in contract['epsilon_sig']:
            prrow = next(r for r in scenarios if r['case']==case and r['epsilon_sig']==eps and r['theta']=='0')
            pfrow = next(r for r in scenarios if r['case']==pf and r['epsilon_sig']==eps and r['theta']=='0')
            out = dict(epsilon_sig=eps,theta='0',status='NO_JOINT_DOMAIN',rare_mean_threshold=None)
            if prrow['G'] is not None and pfrow['G'] is not None:
                pp = {k:F(p) for k,p in d['implemented_probability_exact'].items()}; rarep = pp['any_order4']+pp['no4_with_unseen']
                regular = d['observed_stratum_cost']['remainder']
                regular_z = sum(prrow['N_cos' if a=='cosine' else 'N_sin']*F(regular['axes'][a]['mean_exact']) for a in AXES) if regular else None
                threshold = (F(pfrow['G_exact'])-F(d['implemented_probability_exact']['remainder'])*regular_z)/F(rarep) if regular_z is not None and rarep > 0 else None
                out.update(status='HYPOTHETICAL_REMAINDER_MEAN_SET_TO_OBSERVED_NOT_CERTIFIED',
                           rare_probability=float(rarep),regular_mean_Z_exact=str(regular_z) if regular_z is not None else None,
                           rare_mean_threshold=float(threshold) if threshold is not None else None,
                           threshold_units='weighted whole-trajectory Z RZ, conditional on PR N at rho=0',
                           reversal='PR population G >= PF if mean_Z_in_rare_union >= threshold, conditional on regular mean equaling observed and saved IID masses',
                           threshold_over_empirical_PR_G=float(threshold/F(prrow['G_exact'])) if threshold is not None else None)
            d['rank_flip_sensitivity'].append(out)
        rare[case] = d
    save(ns/'rare_cost_sensitivity.json',rare)
    figures(ns,costs,functions,scenarios,contract)
    budget.take('scalar_rounding_verification_batches')
    require(len(functions)==20 and len(scenarios)==100 and len(evaluations)==200 and len(axisrows)==40 and len(regions)==30 and sum(r['primary'] for r in regions)==10, 'OUTPUT_CARDINALITY')
    hashes({str(ROOT/k):v for k,v in manifest['files'].items()})
    summary = dict(status='CONDITIONAL_REANALYSIS_COMPLETE',nonempty_conditions=sum(f['domain_nonempty'] for f in functions),
        conditional_functions=20,scenario_rows=100,axis_evaluations=200,guard_thresholds=40,comparison_regions=30,primary_regions=10,
        price_joins=514,PR_tail_batches=2,finite_scenario_rows=sum(r['G'] is not None for r in scenarios),
        scalar_sufficient_inequalities_verified=all(e.get('sufficient_inequality_verified') for e in evaluations if e['N'] is not None),
        max_scalar_ceiling_interval_width=max((e.get('ceiling_interval_width',0) for e in evaluations),default=0),
        paired_covariance_verified=True,input_preservation='PASS',original_N_G_null=True,formal_certificate_accepted=False,
        C_L='COMBINATION_BLOCKED',compiler='MISSING_EVIDENCE',original_H4_native='UNKNOWN_UNCHANGED',forbidden_calls=contract['forbidden_calls'],
        mandatory_STOP=True)
    save(ns/'result_summary.json', summary)
    return summary
