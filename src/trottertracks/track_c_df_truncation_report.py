"""Read frozen results, check accounting, and export the comparison tables.

No molecular generation, sampler, compiler, or quantum simulation is called.
"""
import csv
import hashlib
import json
import math
from fractions import Fraction as F
from pathlib import Path
import numpy as np
from . import track_c_df_truncation as d

OUT=d.OUT
PRIOR=d.ROOT/'artifacts/track_c_closed_p3_joint_opt/2026-11-08/extended/FINAL.json'


def write_csv(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def old_policy(method,family):
    p=json.loads(PRIOR.read_text())['methods'][method]
    assert p['all_source_native_closed'] and p['mean_branch_pass']
    return {**p,'n':4,'family':family,'status':'QUALIFIED_REUSED','trials':[r['trials'] for r in p['rounds']],
            'origin':f'{d.BASE}:extended/FINAL.json:{method}','legacy':True,
            'mean_decoder':dict(absolute_error_validation_only=p['mean_energy_error'])}


def load_samples(phase='holdout'):
    data={}
    for n in (2,6):
        for path in (OUT/f'H{n}/pr_{phase}').glob('*.json'):
            r=json.loads(path.read_text());cell=(n,r['ld'],r['h'],r['r']);data.setdefault(cell,{}).setdefault(r['rare'],[]).append(r)
    return data


def row_costs(p,m,samples):
    point=p['selected'][m];ns=np.array(p['trials'][m]);weight=int(ns.sum())
    if p.get('legacy'):
        row=p['rounds'][m]
        return np.array(row['total']),np.array(row['D']),np.array(row['R']),np.array(row['wrapper']),None
    row=point['row'];groups=row['groups'];steps=point['signal']['steps']
    if p['family']=='PR':
        D=np.array(point['D']);over=np.array(point['overhead']);cell=tuple(point['cell'])
        R=steps*sum((sub[0]['probability']*np.array([x['metrics'] for x in sub]).mean(axis=0) for sub in samples[cell].values()),np.zeros(4))
        mean=D+R+over
    else:
        D=np.array(groups['first'])+np.array(groups['last'])+(steps-1)*np.array(groups['middle']);R=np.zeros(4)
        mean=np.array(point['mean']);over=mean-D
    total=(mean*ns[:,None]).sum(axis=0)
    return total,weight*D,weight*R,(over*ns[:,None]).sum(axis=0),mean


def summarize(ordering=False):
    det=json.loads((OUT/'DETERMINISTIC_FINAL.json').read_text());pr=json.loads((OUT/'PR_FINAL.json').read_text())
    target=OUT/'ordering' if ordering else OUT
    if ordering:
        audit=json.loads((target/'FINAL.json').read_text());all_policies=[{**p,'origin':'ordering/FINAL.json'} for p in audit['policies']]
        assert hashlib.sha256((target/'POLICY.json').read_bytes()).hexdigest()==audit['policy_sha256']
    else:all_policies=[{**p,'origin':'DETERMINISTIC_FINAL.json'} for p in det['policies']]+[{**p,'origin':'PR_FINAL.json'} for p in pr['policies']]
    all_policies+=[old_policy('C_Full','Full'),old_policy('B_closed','PR')]
    policies=[]
    for n in (2,4,6):
        for family in ('Full','Truncated','PR'):
            choices=[p for p in all_policies if p['n']==n and p['family']==family and p['status'].startswith('QUALIFIED')]
            policies.append(min(choices,key=lambda p:p['totals'][0]) if choices else dict(n=n,family=family,status='UNRESOLVED'))
    summary=[];rounds=[];mechanisms=[];budgets=[];samples=load_samples('ordering_holdout' if ordering else 'holdout');validations=[]
    for p in policies:
        n=p['n'];family=p['family'];name=f'H{n}_{family}'
        if p['status']=='UNRESOLVED':summary.append(dict(n=n,method=family,status=p['status']));continue
        alpha=np.array(p['alpha']);ns=np.array(p['trials']);alpha_sum=sum(F.from_float(float(x)) for x in alpha.flat)
        assert alpha_sum<=F(1,20) and int(ns.sum())==p['trials_total']
        rho=[];beta=[];total=np.zeros(4);Dtotal=total.copy();Rtotal=total.copy();Wtotal=total.copy()
        for m,point in enumerate(p['selected']):
            q=point['qualification'];assert q['closed'];rho.append(q['rho_lower']);beta.append(q['beta_stat_lower'])
            budgets.append({**{k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in q.items()},'molecule':f'H{n}','method':family,'m':m})
            for a,N in zip(alpha[m],ns[m]):assert d.rigorous_shots(q['rho_lower'],q['beta_stat_lower'],a)<=N
            cost,dc,rc,wc,mean=row_costs(p,m,samples);total+=cost;Dtotal+=dc;Rtotal+=rc;Wtotal+=wc
            signal=point['signal'];microsteps=signal.get('r',sum(x['r'] for x in signal.get('recipe',[])))
            row=dict(molecule=f'H{n}',method=family,m=m,case=signal.get('name'),h=signal['h'],steps=signal.get('steps'),microsteps_per_D_step=microsteps,legacy_resolution_rate=signal.get('rate'),
                N_X=int(ns[m,0]),N_Y=int(ns[m,1]),alpha_X=float(alpha[m,0]),alpha_Y=float(alpha[m,1]),rho=q['rho_lower'],beta=q['beta_stat_lower'],
                total_phase_upper=q.get('total_truncation_PF_phase_upper',q.get('PF_phase_upper')),log_range=q.get('log_range',q.get('RA_log_Wmax_upper',signal.get('log_range',0.))),
                G_RZ=cost[0],G_CX=cost[1],G_serial_depth_upper=cost[2],G_size=cost[3],D_RZ=dc[0],R_RZ=rc[0],wrapper_RZ=wc[0],shot_weighted_mean_unit_RZ=cost[0]/int(ns[m].sum()),origin=p['origin'])
            if mean is not None:row.update(unit_RZ_X=mean[0,0],unit_RZ_Y=mean[1,0])
            rounds.append(row)
            if family=='Truncated':
                candidate=point['row']['candidate'];pilot=json.loads((OUT/f'H{n}/pilot/{candidate["name"]}.json').read_text());t=.02*2**m
                zkeep=complex(*pilot['exact_signal'][m]);zpf=complex(*q['signal']);energy=d.model(n)['energy']
                trunc=float(np.angle(zkeep*np.exp(1j*energy*t)));pf=float(np.angle(zpf*np.conj(zkeep)));combined=float(np.angle(zpf*np.exp(1j*energy*t)))
                mechanisms.append(dict(molecule=f'H{n}',m=m,candidate=candidate['name'],removed_coefficients=len(candidate['drop']),retained_fragment_prefix=candidate.get('retained_fragments'),identity_retained=candidate.get('identity_retained'),h=signal['h'],formula=point['row']['formula'],
                    drop_operator_norm_observed=pilot['drop_operator_norm'],ground_energy_difference_observed=pilot['ground_energy_difference'],dominant_energy_bias_observed=pilot['dominant_energy_bias'],dominant_overlap_observed=pilot['dominant_overlap'],minimum_exact_radius_observed=pilot['minimum_exact_radius'],
                    truncation_phase_signed_observed=trunc,PF_phase_signed_observed=pf,combined_phase_signed_observed=combined,total_phase_upper=q['total_truncation_PF_phase_upper'],rho=q['rho_lower'],beta=q['beta_stat_lower'],N_X=int(ns[m,0]),N_Y=int(ns[m,1])))
        assert np.allclose(total,p['totals'],rtol=1e-12,atol=.01)
        assert np.allclose(Dtotal+Rtotal+Wtotal,total,rtol=1e-12,atol=.01)
        energy_error=p['mean_decoder']['absolute_error_validation_only'];assert energy_error<1.5936e-4
        summary.append(dict(n=n,method=family,status=p['status'],trials=p['trials_total'],RZ=p['totals'][0],CX=p['totals'][1],serial_depth_upper=p['totals'][2],size=p['totals'][3],RZ_MC_SE=p['MC_SE'][0],CX_MC_SE=p['MC_SE'][1],depth_MC_SE=p['MC_SE'][2],size_MC_SE=p['MC_SE'][3],
            minimum_rho=min(rho),minimum_beta=min(beta),alpha_sum_exact=str(alpha_sum),mean_decoder_error=energy_error,D_RZ=float(Dtotal[0]),R_RZ=float(Rtotal[0]),wrapper_RZ=float(Wtotal[0]),origin=p['origin']))
        validations.append(dict(method=name,all_18_closed=True,alpha_sum_exact=str(alpha_sum),all_36_sufficient_trials=True,mean_decoder_error=energy_error,no_E0_decoder_argument=True,origin=p['origin']))
    comparisons=[]
    for n in (2,4,6):
        methods=sorted((p for p in summary if p['n']==n and p['status']!='UNRESOLVED'),key=lambda p:p['RZ'])
        if len(methods)!=3:comparisons.append(dict(n=n,winner='UNDETERMINED'));continue
        best,second=methods[:2];se=math.hypot(best['RZ_MC_SE'],second['RZ_MC_SE']);gap=second['RZ']-best['RZ']
        comparisons.append(dict(n=n,winner_point_estimate=best['method'],second=second['method'],RZ_gap=gap,nominal_MC_SE_of_gap=se,gap_in_nominal_SE=gap/se if se else None,interpretation='finite candidate policies; nominal price-MC comparison, not a global optimum or hardware claim'))
    phases=('training','holdout','ordering_training') if ordering else ('training',)
    training=[json.loads(path.read_text()) for phase in phases for n in (2,6) for path in (OUT/f'H{n}/pr_{phase}').glob('*.json')]
    holdout=[r for cell in samples.values() for sub in cell.values() for r in sub]
    assert not ({r['seed'] for r in training}&{r['seed'] for r in holdout})
    microcerts=[json.loads(path.read_text()) for n in (2,6) for path in (OUT/f'H{n}/pr_cells').glob('*.json')]
    assert all(x['domain'] and x['log_Wmax'][0]>=0 for x in microcerts)
    for phase in ('PR_POLICY.json','DETERMINISTIC_POLICY.json'):
        final=pr if phase.startswith('PR_') else det
        assert hashlib.sha256((OUT/phase).read_bytes()).hexdigest()==final['policy_sha256']
    catalogue=[]
    for n in (2,4,6):
        qualified=[json.loads(path.read_text()) for path in (OUT/f'H{n}/qualified_repriced').glob('*.json')]
        if ordering:qualified+=[dict(row=x['row'],m=x['signal']['m'],qualification=x['qualification']) for path in (target/'qualified').glob(f'H{n}_*.json') if (x:=json.loads(path.read_text()))['family']!='PR']
        selected=next(p for p in policies if p['n']==n and p['family']=='Truncated')
        for path in sorted((OUT/f'H{n}/pilot').glob('*.json')):
            x=json.loads(path.read_text());candidate=x['candidate'];name=candidate['name'];closed=sorted({r['m'] for r in qualified if r['row']['candidate']['name']==name and r['qualification']['closed']})
            chosen=[p['signal']['m'] for p in selected.get('selected',[]) if p['row']['candidate']['name']==name]
            catalogue.append(dict(molecule=f'H{n}',candidate=name,retained_fragments=candidate.get('retained_fragments'),identity_retained=candidate.get('identity_retained'),removed_coefficients=len(candidate['drop']),removed_L1_exact=candidate.get('coefficient_L1_exact'),retained_blocks=x['retained_blocks'],retained_nonidentity_terms=x['retained_nonidentity_terms'],drop_operator_norm_observed=x['drop_operator_norm'],ground_energy_difference_observed=x['ground_energy_difference'],dominant_energy_bias_observed=x['dominant_energy_bias'],dominant_overlap_observed=x['dominant_overlap'],minimum_exact_radius_observed=x['minimum_exact_radius'],nonresonant_18round_bound=x['nonresonant_bound'],screen_eligible=x['pilot_eligible'],directed_closed_rounds=','.join(map(str,closed)),final_selected_rounds=','.join(map(str,chosen)),status='QUALIFIED_AT_LISTED_ROUNDS' if closed else ('PILOT_ONLY' if x['pilot_eligible'] else 'OUTSIDE_SUFFICIENT_PILOT_SCREEN')))
    write_csv(target/'CANDIDATES.csv',catalogue)
    write_csv(target/'ERROR_BUDGET.csv',budgets)
    write_csv(target/'COMPARISON.csv',summary);write_csv(target/'ROUNDS.csv',rounds);write_csv(target/'TRUNCATION_MECHANISMS.csv',mechanisms)
    d.save_json(target/'RESULTS.json',dict(source=d.source(),methods=summary,comparisons=comparisons,metrics=d.METRICS,scope='fixed saved original Hermitian H; exact original ground; ideal native; fresh independent outcomes/bits; state preparation excluded',
        menu='finite per-molecule h/order/retention/split menus and frozen qualified shortlists; H4 Full also includes the prior strong frozen policy',
        training_samples=len(training),holdout_samples=len(holdout),old_H4_PR='reused c5a3c102 extended B_closed; not pooled with new molecules',quantum_backend_shots=0))
    d.save_json(target/'VALIDATION.json',dict(source=d.source(),methods=validations,training_holdout_seed_overlap=0,policy_hashes_preserved=True,all_component_accounts_close=True,all_G7_envelopes_at_least_one=True,minimum_G7_log_W_lower=min(x['log_Wmax'][0] for x in microcerts),nominal_energy_horizon_bound=.4/(.02*2**17),energy_tolerance=1.5936e-4,failure_upper_per_method_and_molecule=.05,
        comparison_scope='N and the success bound concern one algorithm run. The .05 budget is not a joint success statement over all nine table entries. Price MC SE is separate from RPE failure.'))
    print(json.dumps(dict(methods=summary,comparisons=comparisons),indent=2))


if __name__=='__main__':
    import sys
    summarize(ordering='--ordering' in sys.argv[1:])
