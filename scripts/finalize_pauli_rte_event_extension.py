#!/usr/bin/env python3
"""Saved-cohort comparison, scientific plots and bounded-run verification."""
import csv,gzip,hashlib,importlib.util,io,json,math
from pathlib import Path
from fractions import Fraction as Q
ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('event_final_driver',ROOT/'scripts/run_pauli_rte_event_extension.py')
D=importlib.util.module_from_spec(s);s.loader.exec_module(D)
OUT,PREP=D.OUT,D.PREP
read,save=D.read,D.save
def csvsave(p,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('x',newline='') as f:w=csv.DictWriter(f,fields,lineterminator='\n');w.writeheader();w.writerows(rows)
def flat(x):
    cid=x['candidate_id'];c=D.spec(cid);st=x['price_statistics'];t=read(D.case(cid)/'target.json')
    r=dict(molecule=x['molecule'],epsilon_sig=x['epsilon_sig'],candidate_id=cid,L_D=0,q=8,r=x['r'],s=x['s'],K=x['K'],
        cohort=x['cohort'],price_n=st['n'] if st else None,eligibility=x['eligibility'],B_exact=x['B_exact'],
        G_exact=x['G_exact'],G_SE=x['G_SE'],G_nominal95_halfwidth=x.get('G_nominal95_halfwidth'),
        source_file=x['target_path'],source_commit=read(D.current_source())['commit'],hypothetical_rho='0',actual_rho=None,population_tail='UNGUARANTEED',
        G_guard_range_exact=json.dumps(x['G_guard_range_exact']))
    baseline=D.csvrows(PREP/'saved_baseline_quotes.csv')
    for group in ['RTE','Hybrid']:
        b=next(z for z in baseline if z['molecule']==x['molecule'] and z['epsilon_sig']==x['epsilon_sig'] and z['group']==group)
        r['G_over_saved_'+group+'_exact']=ratio(x,b)
    for a,k in [('cosine','cos'),('sine','sin')]:
        r.update({f'N_{k}':str(x[f'N_{k}']) if x[f'N_{k}'] is not None else None,
                  f'bias_{k}_exact':t['axes'][a]['signed_bias_exact'],f'u_{k}_exact':t['axes'][a]['u_enc_exact'],
                  f'C_{k}_exact':st['mean_exact'][a] if st else None,f'C_{k}_SE':st['SE'][a] if st else None,
                  f'shot_{k}_reason':x['axes'][a]['rho_zero']['reason'],f'rho_guard_{k}_exact':x['axes'][a]['rho_guard_exact']})
    return r
def normalize_old(x):
    r=dict(x);r.update(q=int(r['q']),r=int(r['r']),K=int(r['K']),L_D=0,s=int(r['q'])*int(r['r']))
    r['G_SE']=float(r['G_SE']) if r.get('G_SE') is not None else None
    r['price_n']=int(r['price_n']) if r.get('price_n') is not None else None
    r['origin']='saved prior result, quoted without reaccounting'
    return r
def ratio(a,b):return None if a is None or b is None or not D.finite(a) or not D.finite(b) else str(Q(a['G_exact'])/Q(b['G_exact']))
def classification(h,r):
    if not h or not r or not D.finite(h) or not D.finite(r):return 'UNRESOLVED'
    if D.nearest(h,r):return 'NEAR_TIE_OR_UNRESOLVED'
    return 'RTE_LOWER_EMPIRICAL_G' if Q(r['G_exact'])<Q(h['G_exact']) else 'HYBRID_LOWER_EMPIRICAL_G'

def verify(data):
    from qiskit import qpy
    sources=sorted(OUT.glob('SOURCE_*.json'))
    current=read(D.current_source())
    for p,h in {**current['code_sha256'],**current['input_sha256']}.items():assert D.sha(ROOT/p)==h,p
    native=[r for p in OUT.glob('**/native.json') for r in read(p)['results']]
    assert all(x['exit_code']==0 and x['reaped'] for x in native)
    trajectories=0;targets=0;qpy_sha=0;decoded=0;observed={}
    for c in D.registry():
        cid=c['candidate_id'];d=D.case(cid);t=read(d/'target.json');model,ref=D.model_ref(c['molecule'])
        assert not t['state_renormalized'] and t['B_applied_count']==1
        nu=Q(t['raw_norm_exact'])
        if c['molecule']=='H6':assert nu==Q(ref['raw_norm_exact'])
        else:assert Q(ref['state_norm']['lower'])<=nu<=Q(ref['state_norm']['upper'])
        calls=read(d/'target_call_starts.json');assert calls==t['attempted_bits'];targets+=len(calls)
        for cohort in ['exploration','confirmation']:
            files=sorted((d/'prices'/cohort).glob('*/result.json'))
            if not files:continue
            assert len(files)==8 if cohort=='exploration' else len(files) in (48,96,192,384)
            counts={k:0 for k in [0,2,4]}
            for i,p in enumerate(files):
                x=read(p);assert x['index']==i and x['seed']==D.E.trajectory_seed(c['molecule'],c['r'],c['K'],cohort,i)
                assert x['event_count']==len(x['orders'])==c['s']==len(x['events'])
                assert all(k in [0,2,4] and k<=c['K'] for k in x['orders'])
                for k in counts:counts[k]+=x['orders'].count(k)
                for a in D.AXES:
                    for kind in ['input','compiled']:
                        b=(p.parent/(a+'_'+kind+'.qpy.gz')).read_bytes();meta=x['axes'][a][kind+'_qpy']
                        assert hashlib.sha256(b).hexdigest()==meta['gzip_sha256'];raw=gzip.decompress(b)
                        assert hashlib.sha256(raw).hexdigest()==meta['raw_sha256'];qpy_sha+=1
                        if kind=='compiled' and i in (0,len(files)-1):
                            circuit=qpy.load(io.BytesIO(raw))[0];ops=circuit.count_ops()
                            assert circuit.num_qubits=={'H4':9,'H6':13}[c['molecule']]
                            assert ops.get('rz',0)==x['axes'][a]['rz_count'] and ops.get('cx',0)==x['axes'][a]['cx_count']
                            assert set(ops)<=set(('rz','sx','x','cx','measure','barrier'));decoded+=1
                trajectories+=1
            observed[cid+':'+cohort]=dict(n=len(files),event_order_counts=counts)
    for x in data:
        if x['N_total'] is not None:
            for a in D.AXES:
                s=x['axes'][a];shot=s['rho_zero'];assert Q(shot['N'])*Q(shot['headroom_exact'])**2>=Q(s['scale']['A_upper'])
        if x['G_exact'] is not None:
            st=x['price_statistics']
            for a in D.AXES:assert Q(st['mean_exact'][a])==Q(sum(st['values'][a]),st['n'])
            assert Q(x['G_exact'])==sum(Q(x['N_cos' if a=='cosine' else 'N_sin'])*Q(st['mean_exact'][a]) for a in D.AXES)
    usage=read(OUT/'usage.json');limits=read(OUT/'active_scope.json')['limits']
    assert len(D.registry())<=limits['total_new_candidates_max'] and targets<=limits['total_target_state_actions_max']
    assert trajectories==usage['exploration_trajectories']+usage['confirmation_trajectories']<=limits['total_new_trajectories_max']
    assert len(native)==len(D.registry())+trajectories
    return dict(status='PASS',new_candidates=len(D.registry()),target_state_actions=targets,trajectories=trajectories,axis_compiles=2*trajectories,
        native_children=len(native),all_native_exit_zero_and_reaped=True,all_new_QPY_byte_SHA_checked=qpy_sha,representative_compiled_QPY_decoded=decoded,
        seed_event_count_and_norm_verified=True,exact_mean_G_and_sufficient_shot_inequalities_verified=True,prepared_and_prior_inputs_unchanged=True,
        new_Hamiltonian_state_reference=0,new_H2_or_Hybrid_scientific_calls=0,observed_orders=observed,
        actual_rho_guaranteed=False,population_tail_guaranteed=False,formal_certificate_accepted=False,
        checking_scope='Primary Codex technical checks; representative compiled structure, not independent scientific review or a compiler-wide proof')

def figures(trends,baselines):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype':'none'})
    folder=OUT/'figures';folder.mkdir(exist_ok=False)
    for m in ['H6','H4']:
        rs=[r for r in trends if r['molecule']==m and r['q']==8]
        fig,axs=plt.subplots(1,5,figsize=(17,4),layout='constrained')
        for ax,e in zip(axs,D.EPS):
            for K,color in [(2,'tab:blue'),(4,'tab:orange')]:
                for cohort,marker in [('exploration','x'),('confirmation','o')]:
                    selected=sorted([r for r in rs if r['epsilon_sig']==e and r['K']==K and r['cohort']==cohort and D.finite(r)],key=lambda r:r['s'])
                    if selected:
                        ax.errorbar([r['s'] for r in selected],[float(Q(r['G_exact'])) for r in selected],yerr=[2*float(r['G_SE']) for r in selected],marker=marker,color=color,linestyle='--' if cohort=='exploration' else '-',label=f'K{K} {cohort}; 2SE')
            h=next(r for r in baselines if r['molecule']==m and r['epsilon_sig']==e and r['group']=='Hybrid')
            ax.axhline(float(Q(h['G_exact'])),color='black',linestyle=':',label='Saved Hybrid confirmation')
            ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_title('epsilon='+e);ax.set_xlabel('s=q*r, q=8');ax.grid(alpha=.2)
        axs[0].set_ylabel('Conditional G (compiled RZ x shot)');axs[-1].legend(fontsize=6)
        fig.suptitle(m+': hypothetical rho=0; finite prices; cohorts separate; lines are visual guides')
        for ext in ['png','svg','pdf']:fig.savefig(folder/(m+'_G_vs_s.'+ext),dpi=160)
        plt.close(fig)
        fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained')
        for K,color in [(2,'tab:blue'),(4,'tab:orange')]:
            unique={}
            for r in rs:
                if r['K']==K and (r.get('B_exact') or r.get('B_saved_decimal')):unique.setdefault(r['s'],r)
            bb=sorted(unique.values(),key=lambda r:r['s'])
            if bb:axs[0].plot([r['s'] for r in bb],[float(Q(r.get('B_exact') or r['B_saved_decimal'])) for r in bb],marker='o',color=color,label=f'K{K}')
            for cohort,marker in [('exploration','x'),('confirmation','o')]:
                selected=sorted([r for r in rs if r['epsilon_sig']=='.05' or r['epsilon_sig']=='0.05'],key=lambda r:r['s'])
                selected=[r for r in selected if r['K']==K and r['cohort']==cohort and r.get('C_cos_exact') is not None]
                if selected:axs[1].errorbar([r['s'] for r in selected],[float(Q(r['C_cos_exact'])) for r in selected],yerr=[2*float(r['C_cos_SE']) for r in selected],marker=marker,color=color,linestyle='--' if cohort=='exploration' else '-',label=f'K{K} {cohort} cosine; 2SE')
        for e in D.EPS:
            selected=[]
            for s in sorted({r['s'] for r in rs}):
                candidates=[r for r in rs if r['s']==s and r['epsilon_sig']==e and r['K']==2 and r.get('N_cos') is not None and r.get('N_sin') is not None]
                if candidates:selected.append(candidates[0])
            if selected:axs[2].plot([r['s'] for r in selected],[int(r['N_cos'])+int(r['N_sin']) for r in selected],marker='.',label='K2 eps='+e)
        for ax,title,ylabel in zip(axs,['Normalization','Single-wrapper RZ','Conditional sufficient shots'],['B_saved','C_cos (RZ / shot)','N_cos+N_sin']):
            ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_title(title);ax.set_xlabel('s=q*r, q=8');ax.set_ylabel(ylabel);ax.grid(alpha=.2);ax.legend(fontsize=6)
        fig.suptitle(m+': B/N/C from the saved per-candidate accounting; no price tail guarantee')
        for ext in ['png','svg','pdf']:fig.savefig(folder/(m+'_B_N_C_vs_s.'+ext),dpi=160)
        plt.close(fig)

def main():
    if (OUT/'result_summary.json').exists():raise ValueError('FINAL_RESULTS_ALREADY_COMPLETE')
    D.check();data=D.rows()
    save(OUT/'analysis_SOURCE.json',dict(scientific_SOURCE=read(D.current_source())['commit'],analysis_code_sha256=D.sha(Path(__file__)),candidate_results_input_sha256=D.sha(OUT/'candidate_results.json')))
    verification=verify(data);new=[dict(flat(x),origin='new event-extension') for x in data]
    registered={c['candidate_id']:c for c in D.registry()}
    csvsave(OUT/'candidate_status.csv',[dict(candidate_id=c['candidate_id'],molecule=c['molecule'],q=c['q'],r=c['r'],K=c['K'],s=c['s'],initial_or_conditional=c['stage'],
        target_status='ACQUIRED' if c['candidate_id'] in registered else 'NOT_EXECUTED_TRIGGER_NOT_MET',
        scientific_SOURCE=read(D.current_source())['commit'] if c['candidate_id'] in registered else None) for c in read(OUT/'active_scope.json')['candidates']])
    prior_h4=[normalize_old(x) for x in D.csvrows(ROOT/read(OUT/'input_roles.json')['prior_inventory']) if x['molecule']=='H4' and x['group']=='RTE']
    prior_h6=[normalize_old(x) for x in D.csvrows(ROOT/read(OUT/'input_roles.json')['H6_RTE_inventory'])]
    trends=prior_h4+prior_h6+new;baseline=D.csvrows(PREP/'saved_baseline_quotes.csv')
    csvsave(OUT/'candidate_precision_table.csv',new);csvsave(OUT/'event_cost_trends.csv',trends)
    nominees=read(OUT/'confirmation_nominations.json')['nominees_by_precision'];results=[];details=[]
    for m in ['H4','H6']:
        for e in D.EPS:
            h=next(x for x in baseline if x['molecule']==m and x['epsilon_sig']==e and x['group']=='Hybrid')
            old=next(x for x in baseline if x['molecule']==m and x['epsilon_sig']==e and x['group']=='RTE')
            rr=[x for x in trends if x['molecule']==m and x['epsilon_sig']==e]
            ce=D.minimum([x for x in rr if x['cohort']=='confirmation']);ex=D.minimum([x for x in rr if x['cohort']=='exploration'])
            nominee_id=nominees.get(m+':'+e);nom=next((x for x in new if x['candidate_id']==nominee_id and x['epsilon_sig']==e and x['cohort']=='confirmation'),None)
            ss=sorted({x['s'] for x in rr if D.finite(x)});maxnew=max((x['s'] for x in new if x['molecule']==m),default=64)
            unresolved=[x['candidate_id'] for x in rr if x['cohort']=='exploration' and x['eligibility']!='BIAS_INELIGIBLE' and x['G_exact'] is None]
            row=dict(molecule=m,epsilon_sig=e,Hybrid_id=h['candidate_id'],Hybrid_G_exact=h['G_exact'],Hybrid_G_SE=h['G_SE'],Hybrid_n=h['price_n'],
                RTE_id=ce['candidate_id'] if ce else None,RTE_s=ce['s'] if ce else None,RTE_K=ce['K'] if ce else None,RTE_G_exact=ce['G_exact'] if ce else None,RTE_G_SE=ce['G_SE'] if ce else None,RTE_n=ce['price_n'] if ce else None,
                Hybrid_over_RTE_exact=ratio(h,ce),RTE_over_Hybrid_exact=ratio(ce,h),classification=classification(h,ce),
                old_RTE_id=old['candidate_id'],old_RTE_G_exact=old['G_exact'],new_RTE_over_old_exact=ratio(ce,old),
                exploration_RTE_id=ex['candidate_id'] if ex else None,exploration_RTE_G_exact=ex['G_exact'] if ex else None,
                frozen_nominee_id=nominee_id,nominee_confirmation_G_exact=nom['G_exact'] if nom else None,
                upper_tested_s=maxnew,confirmed_min_on_upper_boundary=ce['s']==maxnew if ce else None,
                unresolved_ids=';'.join(sorted(set(unresolved))),new_matched_Hybrid=None,comparison_type='FAMILY_COMPARISON_TO_SAVED_HYBRID',
                hypothetical_rho='0',population_tail='UNGUARANTEED',actual_rho=None,global_optimality_proved=False)
            results.append(row);details.append(dict(comparison=row,Hybrid=h,RTE=ce,frozen_nominee=nom,prior_RTE=old))
    csvsave(OUT/'best_by_precision.csv',results);save(OUT/'comparison_details.json',details)
    csvsave(OUT/'selected_cost_breakdown.csv',[dict(x[key],comparison_molecule=x['comparison']['molecule'],comparison_epsilon=x['comparison']['epsilon_sig'],role=key) for x in details for key in ['Hybrid','RTE']])
    figures(trends,baseline);save(OUT/'verification.json',verification)
    save(OUT/'result_summary.json',dict(status='CONDITIONAL_EVENT_EXTENSION_COMPLETE',comparisons=results,counts={k:verification[k] for k in ['new_candidates','target_state_actions','trajectories','axis_compiles','native_children']},
        existing_H2_unchanged=True,actual_rho_guaranteed=False,population_tail_guaranteed=False,formal_certificate_accepted=False,important_claim_adopted=False,mandatory_STOP=True))
    save(OUT/'STUDY_STOP.json',dict(mandatory_STOP=True,new_scientific_run_may_not_resume=True,science_complete=True,counts={k:verification[k] for k in ['target_state_actions','trajectories','axis_compiles']},no_formal_certificate_or_major_claim=True))
    print(json.dumps(dict(status='COMPLETE',counts=verification['trajectories'],comparisons=[{k:x[k] for k in ['molecule','epsilon_sig','RTE_s','Hybrid_over_RTE_exact','classification']} for x in results]),indent=2))
if __name__=='__main__':main()
