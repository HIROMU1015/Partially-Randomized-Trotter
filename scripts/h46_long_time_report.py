"""Accounting, fixed independent confirmation design, and standalone figures."""
import argparse
import csv
from fractions import Fraction
import json
import math
from pathlib import Path
import statistics

import h46_long_time_signal as study
ROOT,PUBLIC,PRIVATE=study.ROOT,study.PUBLIC,study.PRIVATE
read,save=study.read,study.save


def fraction(record):
    return Fraction(record['numerator'],record['denominator'])


def cost_for(endpoint,cost):
    row=endpoint['candidate'];nc,ns=endpoint['N_cos'],endpoint['N_sin']
    if row['method']!='PR':
        if 'unit_statistics' in cost:
            stats=cost['unit_statistics'];a=fraction(stats['cosine']['mean_exact']);b=fraction(stats['sine']['mean_exact'])
        else:
            a,b=Fraction(cost['cosine']),Fraction(cost['sine'])
        value=nc*a+ns*b
        return dict(G=float(value),G_exact=str(value),G_SE=None,n=1,C_cos=float(a),C_sin=float(b),
                    cost_kind='DETERMINISTIC_INTEGER',cohort='DETERMINISTIC')
    stats=cost['unit_statistics'];n=stats['cosine']['n']
    a=fraction(stats['cosine']['mean_exact']);b=fraction(stats['sine']['mean_exact']);value=nc*a+ns*b
    variance=None
    if stats['cosine']['variance_exact'] is not None:
        variance=nc*nc*fraction(stats['cosine']['variance_exact'])+ns*ns*fraction(stats['sine']['variance_exact'])+2*nc*ns*fraction(stats['covariance_exact'])
    return dict(G=float(value),G_exact=str(value),G_SE=None if variance is None else math.sqrt(float(variance/n)),n=n,
                C_cos=float(a),C_sin=float(b),cost_kind='FINITE_SAMPLE_PAIRED_EXPECTATION_ESTIMATE',
                cohort=cost.get('cohort','SAVED_EXPLORATION'),population_guarantee=False)


def endpoints():
    values=read(PUBLIC/'signal_endpoints.json.gz')
    path=PUBLIC/'extension_signal_endpoints.json.gz'
    if path.exists():
        values+=read(path)
    return values


def costs():
    result={}
    # All compatible saved prices remain useful even when the new acquisition
    # screen did not select that candidate. Do not narrow an existing baseline.
    p=PUBLIC/'reused_compiled_prices.json.gz'
    if p.exists():
        result.update(read(p))
    for name in ('exploration_costs.json.gz','extension_exploration_costs.json.gz','extension_parallel_costs.json.gz'):
        p=PUBLIC/name
        if p.exists():
            result.update(read(p))
    p=PUBLIC/'saved_secondary_metrics.json'
    if p.exists():
        for identity,value in read(p).items():
            if identity in result and 'mean_metrics' not in result[identity]:
                result[identity]['mean_metrics']=value['mean_metrics']
    return result


def feasibility():
    values=endpoints();rows=[]
    for s in study.PREFIX:
        for T in study.Q:
            for e in study.EPS:
                for family in ('PF','PR'):
                    for f in study.FORMULAS:
                        domain=[v for v in values if v['candidate']['system']==s and v['candidate']['T']==T
                            and v['epsilon']==e and v['candidate']['formula']==f
                            and (v['candidate']['method']=='PR')==(family=='PR')]
                        eligible=[v for v in domain if v['status']=='ELIGIBLE']
                        q=min((v['candidate']['q'] for v in eligible),default=None)
                        selected=[v for v in eligible if v['candidate']['q']==q]
                        rows.append(dict(system=s,T=T,epsilon=e,family=family,formula=f,
                            minimum_eligible_q=q,maximum_eligible_delta=None if q is None else T/q,
                            eligible_candidates=len(eligible),evaluated_candidates=len(domain),
                            min_log_B_at_minimum_q=min((v['log_B'] for v in selected),default=None),
                            max_log_B_at_minimum_q=max((v['log_B'] for v in selected),default=None),
                            meaning='EMPIRICAL_SIGNAL_ELIGIBILITY_ONLY; not cost optimum or a strict error guarantee'))
    write_csv(PUBLIC/'signal_feasibility.csv',rows)
    return dict(rows=len(rows),complete_registered_signal_domain=True,compiled_cost_optimality=False)


def write_csv(path,rows,fields=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=fields or list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)


def check_mean_and_trial_binding(prices):
    """Saved metadata only: native law/scalar must match the priced mean.

    No old simulation, compile, numerical signal or statistical test is replayed.
    """
    import numpy as np
    signals=read(PUBLIC/'signal_records.json.gz')
    signals.update(read(PUBLIC/'extension_signal_records.json.gz'))
    contexts={s:read(PRIVATE/(s+'_context.json')) for s in study.PREFIX}
    constants={}
    cfg=study.origin_config()
    for s,info in cfg['systems'].items():
        item=next(v for k,v in info['inputs'].items() if k.endswith('_snapshot.npz'))
        with np.load(item['path'],allow_pickle=False) as z:
            constants[s]=float(z['constant'])
    selected=dict(prices)
    if (PUBLIC/'confirmation_costs.json.gz').exists():
        selected.update({'confirmation_'+k:v for k,v in read(PUBLIC/'confirmation_costs.json.gz').items()})
    seen=set();count=0;log_error=0.;scalar_error=0.
    for group in selected.values():
        for path in group.get('trial_files',[]):
            trial=read(path);r=trial['candidate'];key=(trial['cohort'],r['id'],trial['index'])
            if key in seen:
                raise ValueError('DUPLICATE_TRIAL_IN_ACCOUNTING')
            seen.add(key);count+=1
            if trial['status']!='PASS' or trial['conditional']:
                raise ValueError('UNACCEPTED_OR_CONDITIONAL_COST_SAMPLE')
            error=abs(trial['log_B']-signals[r['id']]['signal']['log_B']);log_error=max(log_error,error)
            if error>1e-10:
                raise ValueError('NATIVE_AND_SIGNAL_NORMALIZATION_DISAGREEMENT')
            expected=constants[r['system']]
            if r['method'] in ('PR','B0I'):
                expected+=contexts[r['system']]['prefixes'][str(r['prefix'])]['scalar']
            scalar_error=max(scalar_error,abs(trial['scalar']-expected))
            if abs(trial['scalar']-expected)>1e-12:
                raise ValueError('NATIVE_AND_SIGNAL_SCALAR_DISAGREEMENT')
            if r['method']=='PR' and sum(trial['orders'].values())!=r['q']*r['R_outer']:
                raise ValueError('INCOMPLETE_RTE_TRIAL_ACCOUNTING')
    occupied=set();seed_counts={}
    for cohort in ('EXPLORATION','EXTENSION_EXPLORATION','CONFIRMATION'):
        p=PRIVATE/(cohort+'_seeds.json.gz')
        if not p.exists():
            continue
        values=read(p)
        if len(set(values))!=len(values) or occupied.intersection(values):
            raise ValueError('NEW_COHORT_SEED_COLLISION')
        occupied.update(values);seed_counts[cohort]=len(values)
    result=dict(checked_new_trials=count,maximum_log_B_disagreement=log_error,
        maximum_scalar_disagreement=scalar_error,complete_RTE_trial_counts=True,
        seed_counts=seed_counts,new_seed_collisions=0,old_scientific_runs_replayed=False,
        scientific_certificate=False,check_kind='RECORDED_MEAN_NATIVE_BINDING_ONLY')
    save(PUBLIC/('final_mean_trial_binding.json' if 'CONFIRMATION' in seed_counts else 'exploration_mean_trial_binding.json'),result)
    return result


def account():
    data=[];prices=costs();ep=endpoints()
    check_mean_and_trial_binding(prices)
    signals=read(PUBLIC/'signal_records.json.gz')
    if (PUBLIC/'extension_signal_records.json.gz').exists():
        signals.update(read(PUBLIC/'extension_signal_records.json.gz'))
    for endpoint in ep:
        row=endpoint['candidate'];identity=row['id']
        signal=signals[identity]['signal'];bias=signal['total_signed_bias']
        out=dict(**row,epsilon=endpoint['epsilon'],status=endpoint['status'],
                 N_cos=endpoint['N_cos'],N_sin=endpoint['N_sin'],u=endpoint['u'],log_B=endpoint['log_B'],
                 bias_cos=abs(bias['real']),bias_sin=abs(bias['imag']),
                 signed_bias_cos=bias['real'],signed_bias_sin=bias['imag'],
                 reference_cos=signal['reference']['real'],reference_sin=signal['reference']['imag'],
                 corrected_cos=signal['corrected']['real'],corrected_sin=signal['corrected']['imag'],
                 raw_cos=None if signal.get('raw') is None else signal['raw']['real'],
                 raw_sin=None if signal.get('raw') is None else signal['raw']['imag'],
                 headroom_cos=endpoint['axes']['real']['headroom'],headroom_sin=endpoint['axes']['imag']['headroom'],
                 native_cost_status='ACQUIRED' if identity in prices else 'NOT_ACQUIRED',
                 G=None,G_SE=None,G_exact=None)
        if endpoint['status']=='ELIGIBLE' and identity in prices:
            out.update(cost_for(endpoint,prices[identity]))
        data.append(out)
    best=[]
    for s in study.PREFIX:
        for T in study.Q:
            for e in study.EPS:
                for family in ('PF','PR'):
                    for formula in (*study.FORMULAS,'ALL'):
                        domain=[v for v in data if v['system']==s and v['T']==T and v['epsilon']==e
                            and (v['method']=='PR')==(family=='PR') and (formula=='ALL' or v['formula']==formula)]
                        priced=[v for v in domain if v['G'] is not None]
                        winner=min(priced,key=lambda v:Fraction(v['G_exact'])) if priced else None
                        eligible=sum(v['status']=='ELIGIBLE' for v in domain)
                        best.append(dict(system=s,T=T,epsilon=e,family=family,formula_group=formula,
                            best=winner,priced_eligible=len(priced),eligible=eligible,
                            complete_compiled_domain=len(priced)==eligible,
                            interpretation='BEST_AMONG_PRICED_ELIGIBLE_CANDIDATES',population_guarantee=False))
    save(PUBLIC/'accounting.json.gz',data);save(PUBLIC/'best_configurations.json',best)
    write_csv(PUBLIC/'candidate_precision_accounting.csv',data)
    rows=[]
    for value in best:
        row={k:v for k,v in value.items() if k!='best'}
        if value['best']:
            row.update({k:value['best'].get(k) for k in ('id','method','formula','q','delta','prefix','R_outer','K','N_cos','N_sin','log_B','G','G_SE','n','cohort')})
        rows.append(row)
    write_csv(PUBLIC/'best_configurations.csv',rows)
    # All r/K values retain measured signal/shot information; missing circuit
    # costs remain empty rather than being silently predicted or zero-filled.
    write_csv(PUBLIC/'r_K_sensitivity.csv',[r for r in data if r['method']=='PR'])
    common=[]
    for r in data:
        if r['method']=='PR' and r['R_outer']==32 and r['K']==4:
            common.append({k:r.get(k) for k in ('system','T','epsilon','q','delta','prefix','formula','log_B','N_cos','N_sin','G','G_SE','native_cost_status')})
    write_csv(PUBLIC/'same_q_prefix_R32_K4.csv',common)
    metric_rows=[]
    for identity,cost in prices.items():
        for axis in ('cosine','sine'):
            values=cost.get('mean_metrics',{}).get(axis,{})
            row=dict(id=identity,axis=axis,**values)
            if 'rz_count' not in row:
                row['rz_count']=(cost['unit_statistics'][axis]['mean'] if 'unit_statistics' in cost else cost[axis])
            row['missing_non_RZ_metrics']=not bool(values)
            metric_rows.append(row)
    write_csv(PUBLIC/'per_circuit_metrics.csv',metric_rows)
    # Cost changes within each method family: C_eff and N identify whether
    # lower shot demand offsets a larger per-circuit price.
    decompositions=[]
    for s in study.PREFIX:
        for T in study.Q:
            for e in study.EPS:
                for family in ('PF','PR'):
                    selected={v['formula_group']:v['best'] for v in best if v['system']==s and v['T']==T and v['epsilon']==e and v['family']==family}
                    low=selected['2nd']
                    for f in study.FORMULAS[1:]:
                        high=selected[f]
                        if low is None or high is None:
                            continue
                        nl=low['N_cos']+low['N_sin'];nh=high['N_cos']+high['N_sin']
                        decompositions.append(dict(system=s,T=T,epsilon=e,family=family,higher_formula=f,
                            higher_over_S2_G=high['G']/low['G'],higher_over_S2_N=nh/nl,
                            higher_over_S2_C_eff=(high['G']/nh)/(low['G']/nl),
                            S2_q=low['q'],higher_q=high['q'],S2_prefix=low['prefix'],higher_prefix=high['prefix'],
                            finite_sample_point_comparison=True))
    write_csv(PUBLIC/'C_N_G_decomposition.csv',decompositions)
    summary=dict(endpoints=len(data),registered_candidates=len({r['id'] for r in data}),
        priced_candidates=len(prices),eligible_endpoints=sum(r['status']=='ELIGIBLE' for r in data),
        priced_eligible_endpoints=sum(r['G'] is not None for r in data),
        minima_are_priced_subset_only=True,unobserved_rare_costs_not_imputed=True)
    save(PUBLIC/'accounting_summary.json',summary);return summary


def old_confirmation():
    from h246_saved_cost_evaluation import paired
    folder=ROOT/'artifacts/resource_applicability/track_a_h246_reconstructed_confirmation/2026-10-22/fixed_evaluation'
    samples=list(csv.DictReader((folder/'paired_samples_118.csv').open()))
    result={}
    for system in study.PREFIX:
        value=read(folder/(system+'_fixed_comparison.json'))
        for group in value['groups']:
            row=dict(group['candidate'],T=.8,R_outer=32,K=4)
            if row['method']!='PR':
                continue
            subset=[v for v in samples if v['system']==system and v['candidate_id']==row['id']]
            statistics_=paired([(int(v['cosine_RZ']),int(v['sine_RZ'])) for v in subset])
            row['id']=study.cid(row)
            result[row['id']]=dict(candidate=row,cohort='SAVED_INDEPENDENT_CONFIRMATION',n=len(subset),
                unit_statistics=statistics_,origin=str(folder.relative_to(ROOT)),
                population_guarantee=False,rare_tail_bound=None)
    return result


def confirmation_plan():
    best=read(PUBLIC/'best_configurations.json');prices=costs();selected={};comparisons=[]
    def find(s,T,e,family,f):
        return next(v['best'] for v in best if v['system']==s and v['T']==T and v['epsilon']==e and v['family']==family and v['formula_group']==f)
    for s in study.PREFIX:
        for T in study.Q:
            for e in study.EPS:
                pf,pr=find(s,T,e,'PF','ALL'),find(s,T,e,'PR','ALL')
                if pf and pr and pr['G']<pf['G']:
                    selected[pr['id']]=pr
                    comparisons.append(dict(system=s,T=T,epsilon=e,kind='PR_VS_PF',left=pr['id'],right=pf['id']))
                for family in ('PF','PR'):
                    s2=find(s,T,e,family,'2nd')
                    for f in study.FORMULAS[1:]:
                        high=find(s,T,e,family,f)
                        if s2 and high and high['G']<s2['G'] and family=='PR':
                            selected[s2['id']]=s2;selected[high['id']]=high
                            comparisons.append(dict(system=s,T=T,epsilon=e,kind=f+'_PR_VS_S2_PR',left=high['id'],right=s2['id']))
    reusable=old_confirmation();reuse={};fresh=[];counts={}
    for identity,row in selected.items():
        if identity in reusable:
            reuse[identity]=reusable[identity];continue
        stats=prices[identity]['unit_statistics']
        a,b=stats['cosine']['mean'],stats['sine']['mean'];mean=a+b
        variance=stats['cosine']['variance']+stats['sine']['variance']+2*stats['covariance']
        # Fixed design before drawing confirmation samples: 2% nominal MCSE,
        # variance inflation4, minimum33. No post-confirmation top-up.
        n=max(33,math.ceil(4*variance/(.02*mean)**2))
        counts[identity]=n
        fresh.append({k:row[k] for k in ('id','system','T','q','delta','formula','method','prefix','R_outer','K')})
    plan=dict(candidates=fresh,n_by_candidate=counts,reused_candidates=list(reuse),
        fixed_comparisons=comparisons,selection='FIXED_BEFORE_INDEPENDENT_CONFIRMATION',
        nominal_relative_MCSE=.02,variance_inflation=4,minimum_n=33,
        approximate_interval='Marginal normal 95%, and Bonferroni family sensitivity; neither is a tail guarantee',
        adaptive_confirmation_top_up=False,exploration_pooling=False,
        trajectories=sum(counts.values()),confirmation_target_candidates=len(selected))
    save(PUBLIC/'confirmation_selection.json',plan);save(PUBLIC/'reused_confirmation_costs.json.gz',reuse)
    return {k:v for k,v in plan.items() if k not in ('candidates','fixed_comparisons','n_by_candidate')}


def confirm():
    import h46_long_time_native as native
    return native.run_cost('CONFIRMATION',read(PUBLIC/'confirmation_selection.json'))


def confirmation_account():
    check_mean_and_trial_binding(costs())
    candidates={r['id']:r for r in read(PUBLIC/'accounting.json.gz') if r['G'] is not None}
    # Cost and eligibility depend on epsilon, so recover full endpoint lookup.
    ep={(r['candidate']['id'],r['epsilon']):r for r in endpoints()}
    exploratory={(r['id'],r['epsilon']):r for r in read(PUBLIC/'accounting.json.gz') if r['G'] is not None}
    confirmation=read(PUBLIC/'reused_confirmation_costs.json.gz')
    p=PUBLIC/'confirmation_costs.json.gz'
    if p.exists():
        confirmation.update(read(p))
    plan=read(PUBLIC/'confirmation_selection.json');out=[];family=len(plan['fixed_comparisons'])
    zfamily=statistics.NormalDist().inv_cdf(1-.05/(2*max(1,family)))
    for comparison in plan['fixed_comparisons']:
        e=comparison['epsilon'];left,right=comparison['left'],comparison['right']
        a=cost_for(ep[left,e],confirmation[left]) if left in confirmation else None
        if candidates[right]['method']=='PR':
            b=cost_for(ep[right,e],confirmation[right]) if right in confirmation else None
        else:
            b=exploratory[right,e]
        row=dict(comparison,status='MISSING_INDEPENDENT_COST',population_guarantee=False)
        if a and b:
            diff=b['G']-a['G'];se=math.sqrt((a['G_SE'] or 0.)**2+(b['G_SE'] or 0.)**2)
            row.update(status='INDEPENDENT_FINITE_SAMPLE_POINT_COMPARISON',left_G=a['G'],right_G=b['G'],
                right_minus_left=diff,difference_SE=se,marginal_95_low=diff-1.96*se,marginal_95_high=diff+1.96*se,
                family_95_low=diff-zfamily*se,family_95_high=diff+zfamily*se,left_n=a['n'],right_n=b.get('n',1),
                lower_cost_point_reproduced=diff>0,zero_observed_variance_warning=se==0,
                approximate_interval_not_population_tail_guarantee=True)
        out.append(row)
    save(PUBLIC/'independent_comparisons.json',out);write_csv(PUBLIC/'independent_comparisons.csv',out)
    return dict(comparisons=len(out),complete=sum(r['status']!='MISSING_INDEPENDENT_COST' for r in out),
                point_direction_reproduced=sum(r.get('lower_cost_point_reproduced',False) for r in out))


def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    best=read(PUBLIC/'best_configurations.json')
    colors=plt.cm.viridis([0.,.23,.46,.7,.93])
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
    for si,s in enumerate(study.PREFIX):
        for fi,family in enumerate(('PF','PR')):
            ax=axes[si,fi]
            for e,color in zip(study.EPS,colors):
                values=[v['best'] for v in best if v['system']==s and v['epsilon']==e and v['family']==family and v['formula_group']=='ALL' and v['best']]
                values.sort(key=lambda v:v['T'])
                ax.errorbar([v['T'] for v in values],[v['G'] for v in values],
                    yerr=[v.get('G_SE') or 0. for v in values],marker='o',color=color,label=str(e),capsize=2)
            ax.set_title(s+' '+family+' (priced subset)');ax.set_yscale('log');ax.set_ylabel('Analytic-shot weighted compiled RZ')
            ax.grid(alpha=.25);ax.set_xlabel('Total time T');ax.legend(title='epsilon',fontsize=8)
    fig.tight_layout()
    for suffix in ('png','svg','pdf'):
        fig.savefig(PUBLIC/('cost_vs_time.'+suffix),dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
    for si,s in enumerate(study.PREFIX):
        for fi,family in enumerate(('PF','PR')):
            ax=axes[si,fi]
            for e,color in zip(study.EPS,colors):
                values=[v['best'] for v in best if v['system']==s and v['epsilon']==e and v['family']==family and v['formula_group']=='ALL' and v['best']]
                values.sort(key=lambda v:v['T'])
                ax.plot([v['T'] for v in values],[v['delta'] for v in values],marker='o',color=color,label=str(e))
            ax.set_title(s+' '+family+' selected delta');ax.set_ylabel('delta = T/q');ax.set_xlabel('Total time T');ax.grid(alpha=.25)
            ax.legend(title='epsilon',fontsize=8)
    fig.tight_layout()
    for suffix in ('png','svg','pdf'):
        fig.savefig(PUBLIC/('selected_delta_vs_time.'+suffix),dpi=160)
    plt.close(fig)
    import numpy as np
    from matplotlib.colors import ListedColormap
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    labels={'2nd':0,'4th':1,'4th(m5_best)':2}
    short={'2nd':'S2','4th':'Y4','4th(m5_best)':'m5'}
    for si,s in enumerate(study.PREFIX):
        for fi,family in enumerate(('PF','PR')):
            ax=axes[si,fi];matrix=np.full((5,4),np.nan)
            for ei,e in enumerate(study.EPS):
                for ti,T in enumerate(study.Q):
                    value=next(v['best'] for v in best if v['system']==s and v['family']==family and v['T']==T
                               and v['epsilon']==e and v['formula_group']=='ALL')
                    if value:
                        matrix[ei,ti]=labels[value['formula']]
                        label=short[value['formula']]+' / q'+str(value['q'])
                        if family=='PR':
                            label+='\nR'+str(value['R_outer'])+' K'+str(value['K'])
                        ax.text(ti,ei,label,ha='center',va='center',fontsize=8)
            ax.imshow(matrix,cmap=ListedColormap(['#b8d8ef','#e0edbc','#f5c6a5']),vmin=0,vmax=2,aspect='auto')
            ax.set_xticks(range(4),list(study.Q));ax.set_yticks(range(5),[str(e) for e in study.EPS])
            ax.set_xlabel('Total time T');ax.set_ylabel('epsilon');ax.set_title(s+' '+family+' lowest acquired point cost')
    fig.tight_layout()
    for suffix in ('png','svg','pdf'):
        fig.savefig(PUBLIC/('selected_formula_grid.'+suffix),dpi=160)
    plt.close(fig)
    return dict(figures=['cost_vs_time','selected_delta_vs_time','selected_formula_grid'],formats=['png','svg','pdf'],
                interpretation='Finite-sample points and 1 SE; empirical signal allowance; evaluated cost subset')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['feasibility','account','confirmation-plan','confirm','confirmation-account','figures']);args=parser.parse_args()
    study.install()
    functions={'feasibility':feasibility,'account':account,'confirmation-plan':confirmation_plan,'confirm':confirm,
               'confirmation-account':confirmation_account,'figures':figures}
    print(json.dumps(functions[args.command](),ensure_ascii=False))
