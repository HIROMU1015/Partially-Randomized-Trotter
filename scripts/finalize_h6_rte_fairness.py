#!/usr/bin/env python3
"""H6-only saved-value comparison and artifact checks after fixed batches."""
import csv, gzip, hashlib, importlib, importlib.util, io, json, math
from fractions import Fraction as Q
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('h6_fair_fixed_driver',ROOT/'scripts/run_h6_rte_fairness.py')
D=importlib.util.module_from_spec(s);s.loader.exec_module(D)
C=importlib.import_module('hx_study.pauli_hybrid_rte_comparison')
OUT,BASE,PREP=D.OUT,D.BASE,D.PREP
read,save=D.read,D.save

def csvsave(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fields,lineterminator='\n');w.writeheader();w.writerows(rows)

def flatten(x):
    st=x['price_statistics'];cfg=x['config']
    row=dict(molecule='H6',epsilon_sig=x['epsilon_sig'],candidate_id=x['candidate_id'],group='RTE',cohort=x['cohort'],
        eligibility=x['eligibility'],L_D=str(cfg['L_D']),q=str(cfg['q']),r=str(cfg['r']),K=str(cfg['K']),
        G_exact=x['G_exact'],G_SE=str(x['G_SE']) if x['G_SE'] is not None else None,
        B_exact=x['B_exact'],price_n=str(st['n']) if st else None,
        source_commit=read(OUT/'SOURCE.json')['commit'],source_file='artifacts/h6_rte_fairness/2026-10-25/new_candidate_results.json',
        target_path=x['target_path'],hypothetical_rho='0',actual_compiled_rho='UNKNOWN',population_price_tail='UNGUARANTEED')
    for a,k in [('cosine','cos'),('sine','sin')]:
        row.update({f'N_{k}':str(x[f'N_{k}']) if x[f'N_{k}'] is not None else None,
                    f'C_{k}_exact':st['mean_exact'][a] if st else None,f'C_{k}_SE':str(st['SE'][a]) if st else None,
                    f'bias_{k}_exact':x['bias'][a]['signed_bias_exact'],f'u_{k}_exact':x['bias'][a]['u_enc_exact'],
                    f'shot_{k}_reason':x['axes'][a]['rho_zero']['reason']})
    return row

def minimum(rows,e,group,cohort):
    return C.saved_class_minimum(rows,molecule='H6',epsilon=e,group=group,cohort=cohort)

def ratio(a,b):
    return None if a is None or b is None or a['G_exact'] is None or b['G_exact'] is None else str(Q(a['G_exact'])/Q(b['G_exact']))

def observed_order(h,r):
    if h is None or r is None or h['G_exact'] is None or r['G_exact'] is None:return 'UNRESOLVED'
    gap=Q(h['G_exact'])-Q(r['G_exact'])
    if h['G_SE'] is None or r['G_SE'] is None:return 'UNCERTAINTY_MISSING'
    if abs(float(gap))<=2*math.hypot(float(h['G_SE']),float(r['G_SE'])):return 'NEAR_TIE_OR_UNRESOLVED'
    return 'HYBRID_LOWER_EMPIRICAL_G' if gap<0 else 'RTE_LOWER_EMPIRICAL_G'

def check_raw():
    from qiskit import qpy
    source=read(OUT/'SOURCE.json')
    for p,h in {**source['code_sha256'],**source['input_sha256']}.items():assert D.sha(ROOT/p)==h,p
    reference=read(OUT/'H6/inputs/reference.json');count=0;order={};targets=0
    old_ids={x['candidate_id'] for x in read(OUT/'old_rte_accounting_extended.json')}
    for c in D.registry():
        assert c['candidate_id'] not in old_ids
        target=read(ROOT/c['target_path']);assert target['raw_norm_exact']==reference['raw_norm_exact']
        assert target['B_applied_count']==1 and not target['state_renormalized'];targets+=1
        for cohort in ['exploration','confirmation']:
            files=sorted((ROOT/c['case_path']/'prices'/cohort).glob('*/result.json'))
            if not files:continue
            assert len(files)==8 if cohort=='exploration' else len(files) in (48,96,192,384)
            totals={k:0 for k in (0,2,4)}
            for i,p in enumerate(files):
                x=read(p);assert x['index']==i and x['seed']==D.R.seeds(c,cohort,i)
                assert len(x['orders'])==c['config']['q']*c['config']['r']
                assert all(k in (0,2,4) and k<=c['config']['K'] for k in x['orders'])
                for k in totals:totals[k]+=x['orders'].count(k)
                assert x['qiskit_version']=='1.3.0'
                for a in D.R.AXES:
                    for kind in ['input','compiled']:
                        b=(p.parent/(a+'_'+kind+'.qpy.gz')).read_bytes();meta=x['axes'][a][kind+'_qpy']
                        assert hashlib.sha256(b).hexdigest()==meta['gzip_sha256']
                        raw=gzip.decompress(b);assert hashlib.sha256(raw).hexdigest()==meta['raw_sha256']
                        if kind=='compiled':
                            circuit=qpy.load(io.BytesIO(raw))[0];ops=circuit.count_ops()
                            assert circuit.num_qubits==13
                            assert ops.get('rz',0)==x['axes'][a]['rz_count'] and ops.get('cx',0)==x['axes'][a]['cx_count']
                            assert set(ops)<=set(('rz','sx','x','cx','measure','barrier'))
                count+=1
            order[c['candidate_id']+':'+cohort]=dict(n=len(files),event_order_counts=totals)
    native=[x for p in OUT.glob('**/native.json') for x in read(p)['results']]
    assert len(native)==targets+count
    assert all(x['exit_code']==0 and x['reaped'] for x in native)
    assert targets==17 and count<=136+2688
    fresh=read(OUT/'new_candidate_results.json')
    for x in fresh:
        if x['N_total'] is not None:
            for a in D.R.AXES:
                v=x['axes'][a]['rho_zero'];n=v['N'];h=Q(v['headroom_exact'])
                assert Q(n)*h*h>=Q(x['axes'][a]['scale']['A_upper'])
        if x['G_exact'] is not None:
            st=x['price_statistics']
            for a in D.R.AXES:assert Q(st['mean_exact'][a])==Q(sum(st['values'][a]),st['n'])
            assert Q(x['G_exact'])==sum(Q(x['N_cos' if a=='cosine' else 'N_sin'])*Q(st['mean_exact'][a]) for a in D.R.AXES)
    return dict(status='PASS',new_target_state_actions=targets,new_trajectories=count,new_axis_compiles=2*count,
        all_native_exit_zero_and_reaped=True,native_child_count=len(native),QPY_SHA_RZ_CX_wires_verified=True,
        exact_G_and_sufficient_inequalities_verified=True,same_raw_norm_B_once=True,source_and_old_inputs_unchanged=True,
        observed_orders=order,new_Hamiltonian_state_reference=0,formal_certificate_accepted=False,
        actual_rho_guaranteed=False,population_tail_guaranteed=False,technical_check_by_primary_Codex_not_independent_review=True)

def main():
    if (OUT/'result_summary.json').exists():raise ValueError('FINAL_ANALYSIS_ALREADY_COMPLETE')
    D.source_check()
    assert all(read(p)['completed'] for p in (OUT/'H6').glob('batch*/BATCH_STOP.json'))
    assert (OUT/'confirmations/cohort_1/COHORT_STOP.json').exists()
    save(OUT/'analysis_SOURCE.json',dict(scientific_kernel_SOURCE=read(OUT/'SOURCE.json')['commit'],
         analysis_code_sha256=D.sha(Path(__file__)),input_sha256={str(p.relative_to(ROOT)):D.sha(p) for p in [OUT/'new_candidate_results.json',OUT/'old_rte_accounting_extended.json',OUT/'confirmation_nominees.json']}))
    verification=check_raw()
    old_hybrid=[x for x in D.csvrows(PREP/'candidate_evidence_inventory.csv') if x['molecule']=='H6' and x['group']=='Hybrid']
    old_rte=[flatten(x) for x in read(OUT/'old_rte_accounting_extended.json')]
    for x in old_rte:x['source_file']='artifacts/h6_rte_fairness/2026-10-25/old_rte_accounting_extended.json'
    new=[flatten(x) for x in read(OUT/'new_candidate_results.json')]
    prior_confirmation=D.csvrows(BASE/'new_candidate_accounting.csv')
    prior_confirmation=[dict(x,source_commit='90c1c60820d52ca7ddcd4fb5315e9d21bb4a86c3',source_file='artifacts/pauli_hybrid_rte_comparison/2026-10-25/new_candidate_accounting.csv') for x in prior_confirmation if x['molecule']=='H6']
    rte=old_rte+new+prior_confirmation;allrows=old_hybrid+rte
    assert len({(x['candidate_id'],x['epsilon_sig'],x['cohort']) for x in rte})==len(rte)
    exploration=[x for x in rte if x['cohort']=='exploration'];ids={x['candidate_id'] for x in exploration}
    assert len(ids)==24 and len(exploration)==120
    assert {(int(x['q']),int(x['r']),int(x['K'])) for x in exploration}=={(q,r,K) for q in (1,2,4,8) for r in (2,4,8) for K in (2,4)}
    csvsave(OUT/'expanded_RTE_inventory.csv',rte)
    registered=[]
    for cid in sorted(ids):
        x=next(x for x in exploration if x['candidate_id']==cid)
        registered.append(dict(candidate_id=cid,L_D=x['L_D'],q=x['q'],r=x['r'],K=x['K'],origin='reused old r4' if cid in {x['candidate_id'] for x in old_rte} else 'new',target_path=x.get('target_path'),source_file=x['source_file']))
    csvsave(OUT/'registered_RTE_candidates.csv',registered)
    base={x['epsilon_sig']:x for x in read(BASE/'comparison_by_precision.json') if x['molecule']=='H6'}
    with (BASE/'comparison_by_precision.csv').open() as f:baseline={x['epsilon_sig']:x for x in csv.DictReader(f) if x['molecule']=='H6'}
    result=[];wide=[];detail=[];unresolved=[];matched=[]
    nominees=read(OUT/'confirmation_nominees.json')['family_nominees_before_confirmation']
    for e in D.R.EPS:
        exph=minimum(allrows,e,'Hybrid','exploration');expr=minimum(allrows,e,'RTE','exploration')
        ch=minimum(allrows,e,'Hybrid','confirmation');cr=minimum(allrows,e,'RTE','confirmation');h=ch['selected'];r=cr['selected']
        mid='PR_p0000_q1_r8_K4' if e=='0.05' else 'PR_p0000_q8_r8_K2'
        m=next((x for x in rte if (x['candidate_id'],x['epsilon_sig'],x['cohort'])==(mid,e,'confirmation')),None)
        assert m is not None and all(h[k]==m[k] for k in ['q','r','K'])
        nominated=next(x for x in rte if (x['candidate_id'],x['epsilon_sig'],x['cohort'])==(nominees[e]['candidate_id'],e,'confirmation'))
        b=baseline[e]
        row=dict(epsilon_sig=e,Hybrid_id=h['candidate_id'],Hybrid_G_exact=h['G_exact'],Hybrid_G_SE=h['G_SE'],Hybrid_n=h['price_n'],
             matched_RTE_id=mid,matched_RTE_G_exact=m['G_exact'],matched_RTE_G_SE=m['G_SE'],matched_RTE_n=m['price_n'],
             matched_Hybrid_over_RTE_exact=ratio(h,m),matched_classification=observed_order(h,m),
             expanded_RTE_id=r['candidate_id'],expanded_RTE_G_exact=r['G_exact'],expanded_RTE_G_SE=r['G_SE'],expanded_RTE_n=r['price_n'],
             family_Hybrid_over_RTE_exact=ratio(h,r),family_classification=observed_order(h,r),
             exploration_Hybrid_id=exph['selected']['candidate_id'],exploration_Hybrid_G_exact=exph['G_exact'],
             exploration_RTE_id=expr['selected']['candidate_id'],exploration_RTE_G_exact=expr['G_exact'],exploration_Hybrid_over_RTE_exact=C.hybrid_over_rte(exph,expr),
             frozen_family_nominee_id=nominated['candidate_id'],nominee_confirmation_G_exact=nominated['G_exact'],
             old_r4_RTE_id=b['confirmation_RTE_candidate_id'],old_r4_RTE_G_exact=b['confirmation_RTE_G_exact'],old_r4_Hybrid_over_RTE_exact=b['confirmation_Hybrid_over_RTE_exact'],
             expanded_RTE_over_old_r4_RTE_exact=str(Q(r['G_exact'])/Q(b['confirmation_RTE_G_exact'])),
             best_deterministic_id=b['deterministic_id'],best_deterministic_G_exact=b['deterministic_G_exact'],
             unresolved_RTE_ids=';'.join(expr['unresolved_ids']),hypothetical_rho='0',population_tail='UNGUARANTEED',
             minimum_scope='COMPUTABLE_FINITE_SUBSET; exploration/confirmation separate; confirmation min post-selection')
        wide.append(row);matched.append({k:v for k,v in row.items() if k.startswith(('epsilon','Hybrid','matched','unresolved','hypothetical','population','minimum'))})
        result.append(dict(epsilon_sig=e,matched_Hybrid=h,matched_RTE=m,exploration_Hybrid=exph,exploration_RTE=expr,confirmation_Hybrid=ch,confirmation_RTE=cr,comparison=row,frozen_nominee_confirmation=nominated))
        for role,x in [('Hybrid_confirmation',h),('matched_RTE_confirmation',m),('RTE_family_confirmation_min',r),('RTE_family_exploration_min',expr['selected'])]:detail.append(dict(role=role,**x))
        for cid in expr['unresolved_ids']:
            x=next(x for x in exploration if x['epsilon_sig']==e and x['candidate_id']==cid)
            # A bound for the SAME FIXED EMPIRICAL price; never a population C bound.
            sc=D.F.scale(x['B_exact']);ea=Q(float(e)/math.sqrt(2));nlo=math.ceil(Q(sc['A_lower'])/ea**2)
            price_known=all(x.get('C_'+k+'_exact') is not None for k in ['cos','sin'])
            glb=nlo*(Q(x['C_cos_exact'])+Q(x['C_sin_exact'])) if price_known else None
            unresolved.append(dict(epsilon_sig=e,candidate_id=cid,eligibility=x['eligibility'],reason_cos=x['shot_cos_reason'],reason_sin=x['shot_sin_reason'],
                exact_N_G_remain_null=True,N_per_axis_lower_if_feasible=str(nlo),fixed_observed_price_G_lower_exact=str(glb) if glb is not None else None,
                exceeds_finite_exploration_min=glb>Q(expr['G_exact']) if glb is not None else None,
                scope='ONLY_SAME_FIXED_OBSERVED_PRICE_NOT_A_POPULATION_PRICE_LOWER_BOUND; h<=epsilon_axis, A>=A_lower'))
    csvsave(OUT/'comparison_by_precision.csv',wide);csvsave(OUT/'matched_comparison.csv',matched)
    csvsave(OUT/'selected_cost_breakdown.csv',detail);csvsave(OUT/'unresolved_candidates.csv',unresolved)
    save(OUT/'comparison_by_precision.json',result)
    investigation=read(OUT/'integer_limit_investigation.json')
    summary=dict(status='CONDITIONAL_H6_FAIRNESS_COMPARISON_COMPLETE',registered_RTE_candidates=24,reused_RTE_candidates=7,new_RTE_candidates=17,
        new_target_state_actions=verification['new_target_state_actions'],new_trajectories=verification['new_trajectories'],new_axis_compiles=verification['new_axis_compiles'],
        recovered_old_safe_integer_rows=sum(x['old_G_exact'] is None and x['new_G_exact'] is not None for x in investigation),
        unresolved_RTE_rows=len(unresolved),matched_results=[x['matched_classification'] for x in wide],family_finite_subset_results=[x['family_classification'] for x in wide],
        ratio_family_exact=[x['family_Hybrid_over_RTE_exact'] for x in wide],ratio_matched_exact=[x['matched_Hybrid_over_RTE_exact'] for x in wide],
        actual_rho_guaranteed=False,population_tail_guaranteed=False,whole_registered_population_winner=False,formal_certificate_accepted=False,mandatory_STOP=True)
    save(OUT/'verification.json',verification);save(OUT/'result_summary.json',summary)
    save(OUT/'STUDY_STOP.json',dict(mandatory_STOP=True,new_science_may_not_resume=True,science_complete=True,no_new_Hamiltonian_state_reference=True,
         new_target_state_actions=verification['new_target_state_actions'],new_trajectories=verification['new_trajectories'],new_axis_compiles=verification['new_axis_compiles']))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
