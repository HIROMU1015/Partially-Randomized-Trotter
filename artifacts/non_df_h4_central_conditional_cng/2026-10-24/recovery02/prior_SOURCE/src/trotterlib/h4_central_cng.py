"""Finite central-price adapter; no quantum, chemistry, RNG or circuit imports.

Reuse saved C/bias/scale functions, the original rational shot safeguard and
paired statistics. DF shots/G and guard faces are copied, never reevaluated.
"""
import ast
import copy
from fractions import Fraction as F
import inspect
import math
from pathlib import Path

from . import h4_conditional_cng as P
from . import h4_certificate_control as C

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'artifacts/non_df_h4_central_conditional_cng/2026-10-24'
OLD = 'artifacts/non_df_h4_conditional_cng_results/2026-10-24'
AXES = P.AXES
EPS = ('0.05', '0.01', '0.005', '0.001', '0.0005')
CASES = ('pauli_jw_PF', 'pauli_jw_PR')
PAIRS = (('pauli_jw_PR', 'pauli_jw_PF'),
         ('pauli_jw_PF', 'df_native_PF'),
         ('pauli_jw_PR', 'df_native_PR'))


def bind_prices(rows, samples):
    """Join by method/index AND trajectory/event hashes, never index alone."""
    prices = {}
    for case in CASES:
        method = case.rsplit('_', 1)[1]
        selected = sorted((r for r in rows if r['case'] == method), key=lambda r:r['index'])
        expected = samples[case]
        C.require(len(selected) == len(expected) == (1 if method == 'PF' else 256), 'COHORT_SIZE')
        C.require([r['index'] for r in selected] == list(range(len(expected))), 'COHORT_INDEX')
        C.require(len({r['trajectory_id'] for r in selected}) == len(selected), 'DUPLICATE_ID')
        for r, old in zip(selected, expected):
            C.require(all(r[k] == old[k] for k in ('trajectory_id', 'trajectory_record_hash')), 'TRAJECTORY_JOIN')
            C.require(r['generic_RZ'] == old['RZ'], 'GENERIC_PRICE_JOIN')
            C.require(all(type(r[a]['RZ']) is int and r[a]['RZ'] >= 0 for a in AXES), 'PRICE_INTEGER')
        prices[case] = [{a:r[a]['RZ'] for a in AXES} for r in selected]
    C.require(len(rows) == 257, 'EXTRA_PRICE_ROWS')
    return prices


def cached_comparison(left, right, ranges):
    """Reuse original comparison verbatim, replacing only its nested bounds IO.

    Original bounds() calls shot on both methods. That would recalculate DF.
    Substitute already-evaluated Pauli bounds and saved DF bounds instead.
    The original interval predicates, ratios and interpretation are unchanged.
    """
    tree = ast.parse(inspect.getsource(P.comparison))
    node = tree.body[0]
    bound = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == 'bounds')
    replacement = ast.parse("def bounds(f):\n    return ranges[(f['case'], f['epsilon_sig'])]\n").body[0]
    node.body[node.body.index(bound)] = replacement
    ast.fix_missing_locations(tree)
    env = dict(P.comparison.__globals__, ranges=ranges)
    exec(compile(tree, '<reused comparison with saved bounds>', 'exec'), env)
    return env['comparison'](left, right, {})


def guard_range(evaluations, means):
    if any(evaluations[a][k]['N'] is None for a in AXES for k in ('rho_zero', 'guard_face')):
        return None
    return [sum(F(means[a])*evaluations[a][k]['N'] for a in AXES)
            for k in ('rho_zero', 'guard_face')]


def figure(ns, rows, references, comparisons):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype':'none'})
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    groups = [('Pauli central PF', rows, 'pauli_jw_PF'),
              ('Pauli central PR', rows, 'pauli_jw_PR'),
              ('Pauli generic PF', references, 'pauli_jw_PF'),
              ('Pauli generic PR', references, 'pauli_jw_PR'),
              ('DF native PF', references, 'df_native_PF'),
              ('DF native PR', references, 'df_native_PR')]
    for label, data, case in groups:
        rr = [next(r for r in data if r['case'] == case and r['epsilon_sig'] == e) for e in EPS]
        axs[0].plot(range(5), [r['G'] for r in rr], marker='.', label=label)
    axs[0].set_yscale('log'); axs[0].set_ylabel('Conditional saved-mean G [RZ]')
    axs[0].set_title('Hypothetical rho = 0; actual rho unknown')
    axs[0].legend(fontsize=7)
    for pair in PAIRS:
        rr = [next(r for r in comparisons if (r['left'], r['right'], r['epsilon_sig']) == (*pair, e)) for e in EPS]
        mid = [r['rho_zero_ratio'] for r in rr]
        low = [r['left_right_ratio_enclosure'][0] for r in rr]
        high = [r['left_right_ratio_enclosure'][1] for r in rr]
        axs[1].errorbar(range(5), mid, yerr=[[m-l for m,l in zip(mid,low)], [h-m for h,m in zip(high,mid)]],
                        marker='.', capsize=4, label=pair[0]+' / '+pair[1])
    axs[1].axhline(1, color='grey', linestyle=':'); axs[1].set_ylim(bottom=0)
    axs[1].set_ylabel('G ratio; bars: independent guard boxes')
    axs[1].set_title('Fixed empirical means; population tails unresolved'); axs[1].legend(fontsize=7)
    for ax in axs:
        ax.set_xticks(range(5), EPS); ax.set_xlabel('Signal epsilon'); ax.grid(alpha=.2)
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(ns/('conditional_resources.'+ext), dpi=160)
    plt.close(fig)


def run(ns, contract, budget):
    ns = Path(ns)
    budget.take('input_binding_batch')
    manifest = C.read(BASE/'inputs.json')
    C.hashes({str(ROOT/k):v for k,v in manifest['files'].items()})
    roles = C.read(ROOT/'artifacts/non_df_h4_conditional_cng_plan/2026-10-24/inputs.json')['roles']
    original = {case:C.read(ROOT/roles[case+'_old']) for case in (*CASES, 'df_native_PF', 'df_native_PR')}
    oldrows = C.read(ROOT/OLD/'run01/conditional_results.json')
    functions = C.read(ROOT/OLD/'run01/conditional_functions.json')
    oldregions = C.read(ROOT/OLD/'run01/comparison_regions.json')
    central_stats = C.read(ROOT/'artifacts/non_df_h4_a_axes/2026-10-24/run01/statistics.json')
    price_rows = C.read(ROOT/'artifacts/non_df_h4_a_axes/2026-10-24/run01/paired_costs.json')['rows']
    prices = bind_prices(price_rows, {case:original[case]['circuit_cost']['samples'] for case in CASES})
    _, stats, verify_join = P.reuse()
    # Same original join verifier; no generation, replay, operator or compiler.
    for r in price_rows:
        case = 'pauli_jw_'+r['case']
        p = Path(roles[case+'_old']).parent/'trajectories'/r['trajectory_id']/'price_join.json'
        j = C.read(ROOT/p); verify_join(j, expected_case=case)
        C.require(j['event_record_hash'] == r['event_record_hash'], 'EVENT_HASH_JOIN')
        C.require(j['trajectory_record_hash'] == r['trajectory_record_hash'], 'EVENT_TRAJECTORY_JOIN')
    cref = C.read(ROOT/roles['pauli_jw_C_result'])
    catalog = C.read(ROOT/'artifacts/non_df_h4_conditional_cng_plan/2026-10-24/axis_conditions.json')['axes']
    selected = [copy.deepcopy(f) for f in functions if f['case'] in CASES]
    C.require(len(selected) == 10 and len({(f['case'],f['epsilon_sig']) for f in selected}) == 10, 'TEN_FUNCTIONS')
    scales = {}; means = {}; covariances = {}
    for case in CASES:
        rr = original[case]
        C.require(all(x['shots']['N_total'] is None and x['total_RZ'] is None for x in C.read(ROOT/roles[case+'_precision'])), 'OLD_NULL_CHANGED')
        C.require(rr['state'] == original['df_native_PF']['state'] == original['df_native_PR']['state'], 'SHARED_STATE')
        C.require(rr['reference'] == original['df_native_PF']['reference'] == original['df_native_PR']['reference'], 'SHARED_REFERENCE')
        scales[case] = dict(state=rr['state'], reference=rr['reference'], signed_bias=rr['signal']['signed_bias'],
                            normalization=rr['normalization'], C_reference=cref['reference'], C_state_norm=cref['state_norm'],
                            compiled_rho_new=None, old_generic_residual_transferred=False,
                            semantic_transfer='same ideal D2 control, saved ray/event law/B_saved signal scale; actual compiled error unknown')
        if case.endswith('_PF'):
            means[case] = {a:str(central_stats['PF'][a]['RZ']) for a in AXES}
            covariances[case] = {a:{b:'0' for b in AXES} for a in AXES}
        else:
            means[case] = {a:central_stats['PR']['axes'][a]['mean_exact'] for a in AXES}
            cv = central_stats['PR']['sample_covariance_RZ_exact']
            # Saved all-row equality is checked; not assumed beyond this cohort.
            C.require(all(p['cosine'] == p['sine'] for p in prices[case]), 'SAVED_COHORT_EQUALITY')
            covariances[case] = {a:{b:cv for b in AXES} for a in AXES}
        for a in AXES:
            C.require(sum(p[a] for p in prices[case]) == F(means[case][a])*len(prices[case]), 'SAVED_PRICE_MEAN')
    budget.take('pauli_10_conditions_20_axes_with_guard_faces')
    axes_out = []; bounds = {}; evaluations = {}
    for f in selected:
        case, eps = f['case'], f['epsilon_sig']; ev = {}
        for a, key in zip(AXES, ('real', 'imag')):
            ax = f['axes'][a]
            pointer = next(x for x in catalog if (x['case'],x['epsilon_sig'],x['axis']) == (case,eps,a))['C_axis_pointer'][1]
            item = cref['axes'][pointer]
            C.require((item['case'],str(item['epsilon_sig']),item['axis']) == (case,eps,a), 'C_POINTER')
            C.require(F(ax['u_enc_exact']) == F(item['bias_distance']['upper']), 'C_BIAS_DISTANCE_ONLY')
            C.require(F(ax['signed_bias_exact']) == F(original[case]['signal']['signed_bias'][key]), 'SIGNED_BIAS')
            C.require(F(ax['scale']['B_saved_exact']) == F(original[case]['normalization']['B']) and
                      F(ax['scale']['log_B_upper_exact']) == F(original[case]['normalization']['log_B_upper']), 'SAVED_NORMALIZATION')
            # Saved scalar enclosure is reused; no C/L or scale recalculation.
            C.require(P.base_axis(eps, F(ax['signed_bias_exact']), F(ax['u_enc_exact']), ax['scale']) == ax, 'UNCHANGED_ACCOUNTING_BASIS')
            ev[a] = dict(rho_zero=P.shot(ax, 0), guard_face=P.shot(ax, F(ax['rho_guard_exact'])) if ax['domain_nonempty'] else dict(N=None, reason='EMPTY_DOMAIN'))
            axes_out.append(dict(case=case, epsilon_sig=eps, axis=a, **ax, **ev[a], actual_rho_new=None))
        evaluations[case,eps] = ev
        bounds[case,eps] = guard_range(ev, means[case])
    budget.take('paired_G_statistics_batch')
    results = []; trajectory_G = []
    for f in selected:
        case, eps = f['case'], f['epsilon_sig']
        ev = evaluations[case,eps]; weights = {a:ev[a]['rho_zero']['N'] for a in AXES}
        row = dict(case=case, implementation='Pauli central D2', epsilon_sig=eps, hypothetical_rho_new='0',
                   actual_rho_new=None, actual_N=None, actual_G=None, formal_certificate_accepted=False,
                   N_cos=weights['cosine'], N_sin=weights['sine'], N_total=None, G=None,
                   prices_exact=means[case], guard_domain_nonempty=f['domain_nonempty'], status='CONDITIONAL_UNDETERMINED')
        if all(n is not None for n in weights.values()):
            gz = P.paired(prices[case], weights, stats)
            n = len(prices[case]); cov = covariances[case]
            var = sum(weights[a]*weights[b]*F(cov[a][b]) for a in AXES for b in AXES)/n
            C.require(math.isclose(gz['standard_error']**2,float(var),rel_tol=2e-12,abs_tol=1e-12), 'PAIRED_COVARIANCE_IDENTITY')
            old = next(r for r in oldrows if (r['case'],r['epsilon_sig'],r['theta']) == (case,eps,'0'))
            C.require((old['N_cos'], old['N_sin']) == (weights['cosine'],weights['sine']), 'UNCHANGED_IDEAL_BASIS_RHO_ZERO_N')
            values = [sum(weights[a]*p[a] for a in AXES) for p in prices[case]]
            trajectory_G.append(dict(case=case, epsilon_sig=eps, trajectory_ids=[r['trajectory_id'] for r in price_rows if r['case']==case.rsplit('_',1)[1]], G_i=values))
            row.update(N_total=sum(weights.values()), G=gz['mean'], G_exact=gz['mean_exact'],
                       G_SD=gz['standard_error']*math.sqrt(n), G_SE=gz['standard_error'], G_nominal95_halfwidth=gz['nominal95_halfwidth'],
                       G_nominal95=[gz['mean']-gz['nominal95_halfwidth'],gz['mean']+gz['nominal95_halfwidth']],
                       paired_variance_of_mean_exact=str(var), cohort_n=n, status='HYPOTHETICAL_RHO_ZERO_EMPIRICAL_ONLY',
                       generic_G_saved=old['G'], generic_G_exact_saved=old['G_exact'], central_over_generic_exact=str(F(gz['mean_exact'])/F(old['G_exact'])),
                       G_guard_range_exact=list(map(str,bounds[case,eps])))
        results.append(row)
    references = [copy.deepcopy(r) for r in oldrows if r['theta'] == '0']
    # Only saved DF aggregate guard ranges; no DF shot()/paired()/G reaccounting.
    for e in EPS:
        saved = next(r for r in oldregions if (r['epsilon_sig'],r['left'],r['right']) == (e,'df_native_PF','df_native_PR'))
        for method, side in (('PF','left'), ('PR','right')):
            bounds['df_native_'+method,e] = list(map(F,saved[side+'_G_range_exact']))
    budget.take('comparisons_15_with_guard_boxes')
    fdict = {(f['case'],f['epsilon_sig']):f for f in functions}
    lookup = {(r['case'],r['epsilon_sig']):r for r in references if r['case'].startswith('df_')}
    lookup.update({(r['case'],r['epsilon_sig']):r for r in results})
    comparisons = []
    for e in EPS:
        for l, r in PAIRS:
            out = cached_comparison(fdict[l,e], fdict[r,e], bounds)
            lr, rr = lookup[l,e], lookup[r,e]
            ratio = F(lr['G_exact'])/F(rr['G_exact']) if lr.get('G_exact') and rr.get('G_exact') else None
            out.update(rho_zero_ratio_exact=str(ratio) if ratio is not None else None, rho_zero_ratio=float(ratio) if ratio is not None else None,
                       price_pairing='within Pauli axes only; no Pauli/DF index pairing', actual_residual_guard_membership='UNKNOWN',
                       population_order='UNDETERMINED_NO_TAIL_ENVELOPE', DF_N_G_recomputed=False)
            comparisons.append(out)
    C.require(len(results)==10 and len(axes_out)==20 and len(comparisons)==15, 'FINITE_PACKAGE_CARDINALITY')
    for name, value in (('conditional_results',results), ('axis_results',axes_out), ('comparison_regions',comparisons)):
        C.save(ns/(name+'.json'), value); P.csv_save(ns/(name+'.csv'),value)
    C.save(ns/'reused_reference_rows.json', references)
    C.save(ns/'paired_G_values.json', trajectory_G)
    C.save(ns/'error_scale_provenance.json', scales)
    C.save(ns/'input_verification.json', dict(status='PASS', paired_rows=257, price_joins=257, input_files=len(manifest['files']),
        unchanged_DF_rows=10, reused_DF_guard_boxes=10, actual_rho_unknown=True, old_N_G_null=True))
    figure(ns, results, references, comparisons)
    C.hashes({str(ROOT/k):v for k,v in manifest['files'].items()})
    C.save(ns/'result_summary.json',dict(status='CENTRAL_CONDITIONAL_CNG_COMPLETE_MANDATORY_STOP',
        conditions=10, primary_axes=20, analytic_guard_faces=20, comparisons=15, paired_G_statistics_batches=1,
        nonempty_guard_conditions=sum(f['domain_nonempty'] for f in selected), finite_hypothetical_rows=sum(r['G'] is not None for r in results),
        scalar_sufficiency=all(e[k].get('sufficient_inequality_verified',False) for e in axes_out for k in ('rho_zero','guard_face')),
        max_ceil_interval_width=max(e[k].get('ceiling_interval_width',0) for e in axes_out for k in ('rho_zero','guard_face')),
        DF_N_G_recomputed=False, forbidden_calls=contract['forbidden_calls'],
        historical_PR_hash='FAIL_CAUSE_UNKNOWN', original_native='UNKNOWN_UNCHANGED', original_N_G_null=True,
        C_L='COMBINATION_BLOCKED', old_compiler='MISSING_EVIDENCE', formal_certificate_accepted=False,
        real_rho_new=None, population_cost_upper=None, unconditional_N_G=None, mandatory_STOP=True))
