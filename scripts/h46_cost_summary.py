"""Finite fixed-panel synthesis. No new shots, samples, fits or winner updates."""
from fractions import Fraction as F
import csv
import json


def rational(v):
    if isinstance(v,dict):return F(int(v['numerator']),int(v['denominator']))
    return F(str(v))


def ratios(lo,hi,observed):
    lo,hi,y=map(rational,(lo,hi,observed))
    if not 0<lo<=hi or y<=0:raise ValueError('INVALID_POSITIVE_COST_INTERVAL')
    return dict(rho_lower=str(y/hi),rho_upper=str(y/lo),residual_lower=str((y-hi)/y),residual_upper=str((y-lo)/y))


def summarize(x,plan,output):
    inp=plan['inputs'];rows=x.read(inp['h4_theory']['path'])+x.read(inp['h6_mixed']['path'])+x.read(inp['h6_uniform']['path'])
    d0=x.read(inp['saved_accounting']['path']);expected=[];paired=[];shots=[];totals=[];means={}
    lookup={(r['system'],r['candidate_id'],r['model'],r['axis']):r for r in rows}
    for job in plan['jobs']:
        system=job['system'];cid=job['candidate']['id'];scope='H4_NATIVE_FIXED16' if system=='H4' else 'H6_EXPLORATION92'
        group=[r for r in d0[scope] if r['cell_id']==cid]
        if len(group)!=10:raise ValueError('TEN_PRESERVED_ACCOUNTING_CONDITIONS')
        complete=[r for r in group if r.get('metrics') and r['metrics'].get('RZ')]
        if not complete:raise ValueError('REGISTERED_SAVED_UNIT_COST_MISSING')
        first=complete[0]
        for metric in ('RZ','CX'):
            for a in complete:
                if a['metrics'][metric]['axis_means']!=first['metrics'][metric]['axis_means']:
                    raise ValueError('SAVED_UNIT_COST_REUSE_CONFLICT')
            means[system,cid,metric]=first['metrics'][metric]['axis_means']
        for axis in ('cosine','sine'):
            native=x.read(output/(system+'_'+cid)/('wrapper_'+axis+'.json'))
            theory_pair=x.read(output/(system+'_'+cid)/'paired_theory.json')
            for model in ('A','B','NATIVE_MIXED'):
                t=lookup[system,cid,model,axis]
                single=next(r for r in theory_pair if r['model']==model and r['axis']==axis)
                for metric in ('RZ','CX'):
                    lo=rational(t['lower'][metric.lower()]);hi=rational(t['upper'][metric.lower()])
                    saved=means[system,cid,metric][axis]
                    expected.append(dict(system=system,candidate_id=cid,model=model,axis=axis,metric=metric,
                        theory_lower=str(lo),theory_upper=str(hi),compiled_saved_mean=saved,sample_size=first['sample_count'],
                        sample_phase=first['sample_phase'],alignment='CONDITIONAL_SAVED_MEAN_NOT_COMPLETE_NUMERIC_ENVELOPE',
                        estimand='EXPECTATION_VS_FINITE_SAMPLE' if cid.startswith('B2') else 'DETERMINISTIC_FIXED_CIRCUIT',
                        attribution='UNATTRIBUTED',**ratios(lo,hi,saved)))
                    tc=single['cost'][metric.lower()];observed=native['metrics'][metric]
                    raw=native['raw_native']['cost'];raw=None if raw is None else raw[metric.lower()]
                    paired.append(dict(system=system,candidate_id=cid,model=model,axis=axis,metric=metric,
                        theory_same_event=tc,native_explicit_lowered=raw,compiled_same_event=observed,
                        compiled_origin=native['compiled_origin'],event_digest=native['event_digest'],
                        preparation_hash=native['preparation_hash'],fingerprint=native['fingerprint'],compiler_hash=native['compiler_hash'],
                        alignment='SAME_EVENT_AND_CONSTRUCTED_WRAPPER_BOUND' if cid.startswith('B2') else 'SAME_FIXED_CIRCUIT_BOUND',
                        sample_size=1,population_mean_certified=False,old_compiler_runtime_bit_equivalence_certified=False,
                        attribution='EXPLICIT_LOWERING_VS_QISKIT_LOWERING_AND_OPT1' if raw==tc and model=='NATIVE_MIXED' else 'UNATTRIBUTED',
                        **ratios(tc,tc,observed)))
        for row in group:
            accounting=row.get('accounting') or {};axes=accounting.get('axes',{})
            good=accounting.get('cost_acquisition_eligible') is True and all(axes.get(a,{}).get('eligible') is True for a in ('real','imag'))
            entry=dict(system=system,candidate_id=cid,epsilon=row['epsilon'],factor=row['factor'],status='ELIGIBLE_SAVED_EMPIRICAL_SHOTS' if good else 'INELIGIBLE_SAVED_RULE',
                sample_count=row['sample_count'],sample_phase=row['sample_phase'],shots=None,N_total=row.get('N_total'),no_new_bias_or_shot_rule=True)
            if good:entry['shots']=dict(cosine=axes['real']['shots'],sine=axes['imag']['shots'])
            shots.append(entry)
            for model in ('A','B','NATIVE_MIXED'):
                for metric in ('RZ','CX'):
                    r=dict(system=system,candidate_id=cid,epsilon=row['epsilon'],factor=row['factor'],model=model,metric=metric,status=entry['status'])
                    if good:
                        pair={a:lookup[system,cid,model,a] for a in ('cosine','sine')};N=entry['shots']
                        r['theory_G_lower']=str(sum(N[a]*rational(pair[a]['lower'][metric.lower()]) for a in N))
                        r['theory_G_upper']=str(sum(N[a]*rational(pair[a]['upper'][metric.lower()]) for a in N))
                        compiled=sum(N[a]*rational(means[system,cid,metric][a]) for a in N)
                        original=row['metrics'][metric]['G_exact']
                        if compiled!=rational(original):raise ValueError('PRESERVED_COMPILED_G_EXACT_MISMATCH')
                        r['compiled_G_saved']=str(compiled)
                    totals.append(r)
    comparisons=[]
    for system in ('H4','H6'):
        ids=x.PANEL[system];prs=[c for c in ids if c.startswith('B2')];pfs=[c for c in ids if c.startswith('B1')]
        table={(r['candidate_id'],r['epsilon'],r['factor'],r['model'],r['metric']):r for r in totals if r['system']==system}
        conditions=sorted({(r['epsilon'],r['factor'],r['model'],r['metric']) for r in totals if r['system']==system})
        for pr in prs:
            for pf in pfs:
                for eps,factor,model,metric in conditions:
                    a=table[pr,eps,factor,model,metric];b=table[pf,eps,factor,model,metric]
                    record=dict(system=system,PR=pr,PF=pf,epsilon=eps,factor=factor,model=model,metric=metric,
                        status='NO_COMMON_SAVED_ELIGIBILITY',scope='FIXED_PANEL_ONLY_CONDITIONAL_EMPIRICAL_SHOTS',population_mean_certified=False)
                    if 'compiled_G_saved' in a and 'compiled_G_saved' in b:
                        record.update(status='CONDITIONAL_SAVED_COST_COMPARISON',
                            theory_PR_PF_lower=str(F(a['theory_G_lower'])/F(b['theory_G_upper'])),
                            theory_PR_PF_upper=str(F(a['theory_G_upper'])/F(b['theory_G_lower'])),
                            compiled_PR_PF=str(F(a['compiled_G_saved'])/F(b['compiled_G_saved'])))
                    comparisons.append(record)
    x.write(output/'expectation_vs_saved.json',expected);x.write(output/'same_event_comparisons.json',paired)
    x.write(output/'preserved_shots.json',shots);x.write(output/'shot_weighted_totals.json',totals)
    x.write(output/'pf_pr_comparisons.json',comparisons)
    for name,data in [('expectation_vs_saved',expected),('same_event_comparisons',paired),('pf_pr_comparisons',comparisons)]:
        fields=list(dict.fromkeys(k for r in data for k in r))
        with (output/(name+'.csv')).open('x',newline='') as f:
            w=csv.DictWriter(f,fields);w.writeheader();w.writerows(data)
    return dict(expectation_comparison_rows=len(expected),same_event_comparison_rows=len(paired),
                pf_pr_comparison_rows=len(comparisons),preserved_shot_conditions=len(shots),
                new_shot_rules=0,new_population_claims=0)
