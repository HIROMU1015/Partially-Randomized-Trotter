"""Accounting and reporting for the new B3 study, with saved H46 baselines."""
from fractions import Fraction
import json
import math
import statistics
from pathlib import Path
import h46_b3 as b


def setup():
    b.setup()
    from h46_long_time_report import cost_for, write_csv
    return cost_for,write_csv


def baseline():
    """Accept the fixed study only after science/compiler/shot bindings match."""
    provenance=b.read(b.BASE_PUBLIC/'scientific_provenance.json')
    contexts={s:b.read(b.PRIVATE/(s+'_context.json')) for s in b.signal.PREFIX}
    for s in b.signal.PREFIX:
        own=contexts[s]
        if own['binding']!=provenance['inputs'][s]['binding']:
            raise ValueError('BASELINE_INPUT_BINDING')
    reused={'scripts/h46_long_time_signal.py','scripts/h46_long_time_native.py',
        'scripts/h46_new2_core.py','scripts/h46_new2_native.py','scripts/h46_m5_tol_native.py',
        'scripts/run_pf_common_run.py','scripts/h246_saved_cost_evaluation.py',
        'scripts/d0_frozen_shot_rule_v1.py','src/trotterlib/h46_m5_plan.py'}
    for name,expected in provenance['source_files'].items():
        if name not in reused: continue
        if b.signal.sha(b.ROOT/name)!=expected:
            raise ValueError('REUSED_SCIENTIFIC_SOURCE_CHANGED:'+name)
    data=b.read(b.BASE_PUBLIC/'accounting.json.gz')
    error=0.
    for r in data:
        refs=contexts[r['system']]['reference'][str(r['T'])]
        error=max(error,abs(r['reference_cos']-refs['real']),abs(r['reference_sin']-refs['imag']))
    if error>1e-10: raise ValueError('BASELINE_REFERENCE')
    check=dict(input_state_H_scalar_geometry_basis_rank=True,compiler_hash=provenance['compiler']['hash'],
        control='symmetric_directional',scalar_once=True,actual_adjoint=True,
        shot_policy='UNCHANGED_qualify_record_u1e-9_alpha_axis025_headroom_1percent',
        maximum_reference_difference=error,baseline_commit='155529e20a8fac39e00786c40b57830b71203c4f',
        reused_scientific_source_files=sorted(reused),
        old_results_read_only=True,old_PR_vs_PF_envelope_cells=33,total_cells=40)
    b.save(b.PUBLIC/'comparison_binding.json',check)
    return data


def winner(values):
    priced=[v for v in values if v.get('G') is not None]
    return min(priced,key=lambda v:Fraction(v['G_exact'])) if priced else None


def account():
    cost_for,csv=setup();old=baseline()
    records=b.read(b.PUBLIC/'signal_records.json.gz');eps=b.read(b.PUBLIC/'signal_endpoints.json.gz')
    prices=b.read(b.PUBLIC/'b3_exploration_costs.json.gz')
    data=[];seen=set();maxlog=0.;maxscalar=0.;trials=0
    for identity,group in prices.items():
        for name in group['trial_files']:
            t=b.read(name);r=t['candidate'];key=(t['cohort'],identity,t['index'])
            if key in seen or t['conditional'] or t['status']!='PASS': raise ValueError('DUPLICATE_OR_BAD_TRIAL')
            seen.add(key);trials+=1
            err=abs(t['log_B']-records[identity]['signal']['log_B']);maxlog=max(maxlog,err)
            ctx=b.read(b.PRIVATE/(r['system']+'_context.json'))
            scalar=ctx['constant']+ctx['prefixes']['0']['scalar']
            maxscalar=max(maxscalar,abs(t['scalar']-scalar))
            if err>1e-9 or abs(t['scalar']-scalar)>1e-12 or sum(t['orders'].values())!=r['s']:
                raise ValueError('MEAN_NATIVE_BINDING')
    b.save(b.PUBLIC/'mean_native_binding.json',dict(trials=trials,duplicate_samples=0,
        maximum_log_B_error=maxlog,maximum_scalar_error=maxscalar,event_counts_complete=True))
    for ep in eps:
        r=ep['candidate'];sig=records[r['id']]['signal']
        out=dict(r,epsilon=ep['epsilon'],status=ep['status'],N_cos=ep['N_cos'],N_sin=ep['N_sin'],
            log_B=ep['log_B'],B=math.exp(ep['log_B']) if ep['log_B']<709 else None,
            B_overflow=ep['log_B']>=709,u=ep['u'],log_B_margin=ep['log_B_margin'],
            headroom_cos=ep['axes']['real']['headroom'],headroom_sin=ep['axes']['imag']['headroom'],
            native_cost_status='ACQUIRED' if r['id'] in prices else 'NOT_ACQUIRED',G=None,G_SE=None,G_exact=None)
        for label,key in [('reference','reference'),('corrected','corrected'),('raw','raw'),
                          ('bias','total_signed_bias'),('finite_RTE_bias','finite_RTE_bias')]:
            for axis,component in [('cos','real'),('sin','imag')]:
                out[label+'_'+axis]=sig[key][component]
        if ep['status']=='ELIGIBLE' and r['id'] in prices: out.update(cost_for(ep,prices[r['id']]))
        data.append(out)
    b.save(b.PUBLIC/'accounting.json.gz',data);csv(b.PUBLIC/'candidate_precision_accounting.csv',data)
    best=[];comparisons=[];byq=[];fixed=[]
    for s in b.signal.PREFIX:
        for T in b.signal.Q:
            for e in b.signal.EPS:
                domain=[v for v in data if v['system']==s and v['T']==T and v['epsilon']==e]
                bd=[v for v in old if v['system']==s and v['T']==T and v['epsilon']==e]
                winners={name:winner([v for v in bd if v['method'] in methods]) for name,methods in
                    [('Full_PF',('FULL',)),('B0',('B0','B0I')),('B2',('PR',)),('PF_envelope',('FULL','B0','B0I'))]}
                winners['B3']=winner(domain)
                expanded=min((name for name in ('Full_PF','B0','B2','B3') if winners[name]),key=lambda name:Fraction(winners[name]['G_exact']))
                comparison=dict(system=s,T=T,epsilon=e,expanded_best_family=expanded,
                    old_PR_PF_scope='SAVED_CANDIDATE_SET_33_OF_40_UNCHANGED',finite_sample_only=True)
                for name,v in winners.items():
                    comparison[name+'_G']=None if v is None else v['G']
                    comparison[name+'_id']=None if v is None else v['id']
                for name in ('B2','Full_PF','PF_envelope'):
                    comparison['B3_over_'+name]=winners['B3']['G']/winners[name]['G'] if winners['B3'] and winners[name] else None
                comparisons.append(comparison)
                for f in (*b.signal.FORMULAS,'ALL'):
                    dom=[v for v in domain if f=='ALL' or v['formula']==f]
                    best.append(dict(system=s,T=T,epsilon=e,formula_group=f,best=winner(dom),
                        eligible=sum(v['status']=='ELIGIBLE' for v in dom),priced=sum(v['G'] is not None for v in dom),
                        complete_compiled_domain=all(v['status']!='ELIGIBLE' or v['G'] is not None for v in dom)))
                    for q in sorted({v['q'] for v in dom}):
                        w=winner([v for v in dom if v['q']==q])
                        if w: byq.append(dict(w,formula_group=f))
                for name,dom in [('B3',domain),('B2',[v for v in bd if v['method']=='PR']),
                                  ('PF_envelope',[v for v in bd if v['method']!='PR'])]:
                    w=winner([v for v in dom if abs(v['delta']-.2)<1e-12])
                    fixed.append(dict(system=s,T=T,epsilon=e,family=name,best=w))
    b.save(b.PUBLIC/'best_configurations.json',best);b.save(b.PUBLIC/'comparisons.json',comparisons)
    b.save(b.PUBLIC/'fixed_delta.json',fixed)
    csv(b.PUBLIC/'best_configurations.csv',[dict({k:v for k,v in item.items() if k!='best'},**(item['best'] or {})) for item in best])
    csv(b.PUBLIC/'comparisons.csv',comparisons);csv(b.PUBLIC/'best_by_q.csv',byq)
    csv(b.PUBLIC/'R_K_sensitivity.csv',data)
    csv(b.PUBLIC/'fixed_delta.csv',[dict({k:v for k,v in item.items() if k!='best'},**(item['best'] or {})) for item in fixed])
    # Matched q/R/K controls retain missing prices as null.
    matched=[]
    lookup={}
    for r in old:
        if r['method']=='PR': lookup.setdefault((r['system'],r['T'],r['epsilon'],r['formula'],r['q'],r['R_outer'],r['K']),[]).append(r)
    for r in data:
        if r['G'] is None: continue
        key=tuple(r[k] for k in ('system','T','epsilon','formula','q','R_outer','K'))
        for previous in lookup.get(key,[]):
            matched.append(dict(system=r['system'],T=r['T'],epsilon=r['epsilon'],formula=r['formula'],q=r['q'],
                R_outer=r['R_outer'],K=r['K'],B2_prefix=previous['prefix'],B2_id=previous['id'],B3_id=r['id'],
                B2_status=previous['status'],B2_G=previous['G'],B3_G=r['G'],
                B3_over_B2=None if previous['G'] is None else r['G']/previous['G'],
                B2_log_B=previous['log_B'],B3_log_B=r['log_B']))
    csv(b.PUBLIC/'matched_B2_B3_controls.csv',matched)
    summary=dict(signal_candidates=len(records),endpoints=len(data),priced_candidates=len(prices),
        eligible_endpoints=sum(v['status']=='ELIGIBLE' for v in data),
        B3_below_B2=sum(v['B3_over_B2'] is not None and v['B3_over_B2']<1 for v in comparisons),
        B3_below_Full_PF=sum(v['B3_over_Full_PF'] is not None and v['B3_over_Full_PF']<1 for v in comparisons),
        B3_below_PF_envelope=sum(v['B3_over_PF_envelope'] is not None and v['B3_over_PF_envelope']<1 for v in comparisons),
        B3_expanded_best=sum(v['expanded_best_family']=='B3' for v in comparisons),cells=len(comparisons),
        complete_compiled_domain=False,population_guarantee=False)
    b.save(b.PUBLIC/'accounting_summary.json',summary)
    schedules={}
    from h46_m5_contract import core
    from h46_new2_core import allocation
    for rec in records.values():
        r=rec['candidate'];key=str((r['T'],r['q'],r['formula'],r['R_outer']))
        coeff=core.weights(r['formula'])
        schedules[key]=dict(T=r['T'],q=r['q'],formula=r['formula'],R_outer=r['R_outer'],
            weights=coeff,signed_times=[r['T']*w/r['q'] for w in coeff],steps_per_piece=allocation(coeff,r['R_outer']),
            repeat_outers=r['q'],allocation='ABS_TIME_PER_OUTER')
    b.save(b.PUBLIC/'signed_schedules.json.gz',schedules)
    return summary


def plan():
    setup();best=b.read(b.PUBLIC/'best_configurations.json');prices=b.read(b.PUBLIC/'b3_exploration_costs.json.gz')
    selected={};nominees=[]
    for item in best:
        if item['formula_group']=='ALL' or item['epsilon']==min(b.signal.EPS):
            r=item['best']
            if r:
                selected[r['id']]={k:r[k] for k in b.read(b.PUBLIC/'candidate_inventory.json.gz')[0]}
                nominees.append(dict(system=item['system'],T=item['T'],epsilon=item['epsilon'],formula_group=item['formula_group'],id=r['id']))
    counts={}
    for identity in selected:
        st=prices[identity]['unit_statistics'];mean=st['cosine']['mean']+st['sine']['mean']
        variance=st['cosine']['variance']+st['sine']['variance']+2*st['covariance']
        counts[identity]=max(33,math.ceil(4*variance/(.02*mean)**2))
    result=dict(candidates=list(selected.values()),n_by_candidate=counts,nominees=nominees,
        fixed_before_sampling=True,minimum_n=33,variance_inflation=4,nominal_relative_MCSE=.02,
        trajectories=sum(counts.values()),exploration_pooling=False,post_confirmation_top_up=False,
        decision='Report fixed nominee direction and normal95/Bonferroni sensitivity; no population-tail guarantee')
    path=b.PUBLIC/'confirmation_selection.json'
    if path.exists() and b.read(path)!=result: raise ValueError('FIXED_CONFIRMATION_PLAN_CHANGED')
    b.save(path,result)
    return dict(candidates=len(selected),trajectories=sum(counts.values()))


def confirm_account():
    cost_for,csv=setup();own=b.read(b.PUBLIC/'b3_confirmation_costs.json.gz')
    old=b.read(b.BASE_PUBLIC/'confirmation_costs.json.gz');old.update(b.read(b.BASE_PUBLIC/'reused_confirmation_costs.json.gz'))
    original=b.read(b.BASE_PUBLIC/'accounting.json.gz');comparisons=b.read(b.PUBLIC/'comparisons.json')
    eps={(ep['candidate']['id'],ep['epsilon']):ep for ep in b.read(b.PUBLIC/'signal_endpoints.json.gz')}
    oldlookup={(r['id'],r['epsilon']):r for r in original}
    result=[];rows=[]
    for comp in comparisons:
        identity=comp['B3_id'];e=comp['epsilon'];a=cost_for(eps[identity,e],own[identity])
        rows.append(dict(system=comp['system'],T=comp['T'],epsilon=e,id=identity,**a))
        for family in ('B2','Full_PF','PF_envelope'):
            r=oldlookup[comp[family+'_id'],e]
            independent=r['method']!='PR' or r['id'] in old
            ep=dict(candidate=r,N_cos=r['N_cos'],N_sin=r['N_sin'])
            previous=cost_for(ep,old[r['id']]) if r['id'] in old and r['method']=='PR' else r
            diff=previous['G']-a['G'];se=math.sqrt((a['G_SE'] or 0.)**2+(previous['G_SE'] or 0.)**2)
            z=statistics.NormalDist().inv_cdf(1-.05/(2*3*len(comparisons)))
            result.append(dict(system=comp['system'],T=comp['T'],epsilon=e,family=family,B3_id=identity,
                baseline_id=r['id'],B3_G=a['G'],baseline_G=previous['G'],baseline_minus_B3=diff,difference_SE=se,
                marginal95_low=diff-1.96*se,marginal95_high=diff+1.96*se,
                family95_low=diff-z*se,family95_high=diff+z*se,B3_n=a['n'],baseline_n=previous['n'],
                baseline_uses_independent_confirmation=independent,B3_lower=diff>0,
                fixed_exploration_nominee=True,population_tail_guarantee=False))
    b.save(b.PUBLIC/'independent_comparisons.json',result);csv(b.PUBLIC/'independent_comparisons.csv',result)
    b.save(b.PUBLIC/'fixed_nominee_confirmation.json',rows);csv(b.PUBLIC/'fixed_nominee_confirmation.csv',rows)
    # Formula comparisons keep the pre-fixed strict-precision nominees.
    nominees=b.read(b.PUBLIC/'confirmation_selection.json')['nominees'];formula=[]
    for s in b.signal.PREFIX:
        for T in b.signal.Q:
            ids={n['formula_group']:n['id'] for n in nominees if n['system']==s and n['T']==T and n['epsilon']==min(b.signal.EPS)}
            if '2nd' not in ids: continue
            e=min(b.signal.EPS);s2=cost_for(eps[ids['2nd'],e],own[ids['2nd']])
            for f in b.signal.FORMULAS[1:]:
                if f not in ids: continue
                a=cost_for(eps[ids[f],e],own[ids[f]])
                formula.append(dict(system=s,T=T,epsilon=e,formula=f,S2_id=ids['2nd'],high_id=ids[f],
                    high_over_S2=a['G']/s2['G'],high_G=a['G'],S2_G=s2['G'],
                    difference_SE=math.hypot(a['G_SE'] or 0,s2['G_SE'] or 0),fixed_nominees=True))
    csv(b.PUBLIC/'formula_confirmation.csv',formula);b.save(b.PUBLIC/'formula_confirmation.json',formula)
    return dict(comparisons=len(result),B3_below_B2=sum(v['B3_lower'] for v in result if v['family']=='B2'),
        B3_below_PF_envelope=sum(v['B3_lower'] for v in result if v['family']=='PF_envelope'))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('command',choices=['account','plan','confirm_account'])
    args=p.parse_args();print(json.dumps(globals()[args.command]()))
