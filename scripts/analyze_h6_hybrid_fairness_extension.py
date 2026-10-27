#!/usr/bin/env python3
"""Approved H6 saved-statistical design and fixed-four accounting only."""
import csv,gzip,hashlib,importlib.util,json,math,subprocess
from fractions import Fraction as Q
from pathlib import Path
import numpy as np
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('fixed_four_worker',ROOT/'scripts/run_h6_hybrid_fairness_extension.py')
W=importlib.util.module_from_spec(s);s.loader.exec_module(W)
R,E,F,OUT=W.R,W.E,W.F,W.OUT
read,save,sha=W.read,W.save,W.sha
EPS=R.EPS;AXES=R.AXES
OLD='8fd1ceb283309a7e515ff4a533a253260bdd0f78'
PREP='25d248a1f2ed88de62b0e621c2d1cea8532804c4'
EXT='10cb3e5cca7834ebf75301bd91bd093ca51d4a6b'
def frozen_analysis():
    p=OUT/'ANALYSIS_SOURCE.json'
    files=['scripts/analyze_h6_hybrid_fairness_extension.py','scripts/run_h6_hybrid_fairness_extension.py']
    if not p.exists():save(p,dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),code_sha256={x:sha(ROOT/x) for x in files}))
    for path,h in read(p)['code_sha256'].items():assert sha(ROOT/path)==h,path
def gitjson(ref,path):return json.loads(W.gitbytes(ref,path))
def csvsave(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def stats_fn():return R.A.A.extract('src/trotterlib/fixed_signal_evaluation.py','_statistics',dict(np=np,math=math,Fraction=Q))
def price_summary(cid,cohort):
    ps=sorted((W.case(cid)/'prices'/cohort).glob('*/result.json'))
    if not ps:return None
    raw=[read(p) for p in ps];values=[{a:r['axes'][a]['rz_count'] for a in AXES} for r in raw]
    return R.A.A.cost_summary(values,stats_fn()),values,raw
def candidate_rows(cid,cohort):
    spec=read(W.case(cid)/'spec.json');target=read(W.case(cid)/'target.json');st=price_summary(cid,cohort)
    prices={a:st[0]['axes'][a]['mean_exact'] if st else None for a in AXES};out=[]
    for eps in EPS:
        x=E.account_target(target,prices,eps,spec['law']);status=R.A.joint_status(target,eps)
        x.update(candidate_id=cid,cohort=cohort,eligibility=status,price_n=st[0]['n'] if st else None,config=spec['config'],
                 B_exact=target['B_saved_exact'],raw_norm_exact=target['raw_norm_exact'],G_SE=None,
                 C_cos_exact=prices['cosine'],C_sin_exact=prices['sine'])
        if st and x['N_total'] is not None:
            z=R.A.A.paired(st[1],dict(zip(AXES,[x['N_cos'],x['N_sin']])),stats_fn())
            assert Q(z['mean_exact'])==Q(x['G_exact']);x.update(G_SE=z['standard_error'],G_nominal95_halfwidth=z['nominal95_halfwidth'])
        for a,short in zip(AXES,['cos','sin']):x['bias_'+short+'_exact']=target['axes'][a]['signed_bias_exact'];x['u_'+short+'_exact']=target['axes'][a]['u_enc_exact']
        out.append(x)
    return out
def baseline():
    if (OUT/'saved_baselines.json').exists():return read(OUT/'saved_baselines.json')
    old=gitjson(PREP,'artifacts/pauli_hybrid_rte_preparation/2026-10-25/evidence/H6/confirmation01/results.json')
    detail=gitjson(EXT,'artifacts/pauli_rte_event_extension/2026-10-26/comparison_details.json');out=[]
    for e in EPS:
        refs=[]
        for cid in ['PR_p0078_q8_r8_K2','PR_p0684_q1_r8_K4']:
            x=next(x for x in old if x['candidate_id']==cid and x['epsilon_sig']==e)
            refs.append(dict(candidate_id=cid,epsilon_sig=e,G_exact=x['G_exact'],G_SE=x['G_SE'],price_n=x['price_statistics']['n'],N_cos=x['N_cos'],N_sin=x['N_sin'],source_commit=PREP))
        d=next(x for x in detail if x['comparison']['molecule']=='H6' and x['comparison']['epsilon_sig']==e)
        r=d['RTE'];refs.append({k:r[k] for k in ['candidate_id','epsilon_sig','G_exact','G_SE','price_n','N_cos','N_sin']}|dict(source_commit=EXT))
        out.append(dict(epsilon_sig=e,refs=refs,saved_Hybrid_id=d['comparison']['Hybrid_id']))
    save(OUT/'saved_baselines.json',out);return out
def design():
    W.check();frozen_analysis();p=OUT/'sampling_design.json'
    if p.exists():raise ValueError('DESIGN_ALREADY_FIXED')
    old=gitjson(PREP,'artifacts/pauli_hybrid_rte_preparation/2026-10-25/evidence/H6/confirmation01/results.json');oldstats={};cvs=[]
    for cid in ['PR_p0078_q8_r8_K2','PR_p0684_q1_r8_K4']:
        x=next(x for x in old if x['candidate_id']==cid);st=x['price_statistics'];vals=[{a:st['values'][a][i] for a in AXES} for i in range(st['n'])]
        cs=R.A.A.cost_summary(vals,stats_fn());oldstats[cid]=cs
        for axis in AXES:
            cv2=float(Q(cs['covariance_exact'][axis][axis]))/float(Q(cs['axes'][axis]['mean_exact']))**2;cvs.append(cv2)
    save(OUT/'saved_paired_covariance_reanalysis.json',dict(source_commit=PREP,statistics=oldstats,actual_compiled_rho=None,population_tail='UNGUARANTEED'))
    eligible=[];rarity={};signal=[]
    for cid in E.CANDIDATES:
        rr=candidate_rows(cid,'exploration');signal+=rr
        if not any(x['eligibility']=='CONDITIONAL_GUARD_ELIGIBLE' for x in rr):continue
        eligible.append(cid);sp=read(W.case(cid)/'spec.json');law=sp['law'];cfg=sp['config']
        nonzero=float(sum(Q(p) for k,p in zip(law['orders'],law['order_mapping']['exact_masses']) if k>0));s=cfg['q']*cfg['r']
        any_nonzero=-math.expm1(s*math.log1p(-nonzero))
        needed=math.ceil(math.log(.1)/math.log1p(-any_nonzero)) if 0<any_nonzero<1 else 1 if any_nonzero==1 else None
        rarity[cid]=dict(nonzero_event_mass=nonzero,probability_any_nonzero_per_trajectory=any_nonzero,n_for_nominal90pct_at_least_one=needed,
                         not_a_tail_or_variance_bound=True,K=cfg['K'])
    c=float(t.ppf(.975,47));cv_requirement=math.ceil(c*c*max(cvs)/.01**2)
    # Bounded exploratory budget, separate from hardware resource limits. Rare
    # event detection is a design diagnostic, never a population-cost proof.
    rare_requirement=max([v['n_for_nominal90pct_at_least_one'] or 128 for v in rarity.values()] or [16])
    n_E=min(128,max(16,cv_requirement,rare_requirement))
    plan=dict(status='FIXED_BEFORE_EXPLORATION',candidates=eligible,n_E=n_E,n_E_max=128,n_C_min=64,n_C_max=1024,
              nominal_relative_halfwidth=.01,confirmation_variance_coefficient='Student_t_0.975_df_nE_minus1',
              confirmation_rule='All jointly eligible candidates; n_C=max over eligible eps and finite fixed comparators of ceil(v/(D^2/c^2-SE_old^2)), clipped[64,1024]. Nonpositive denominator keeps unresolved floor; also require own 1% mean precision. Freeze n_C once before confirmation. No confirmation top-up.',
              nominee_rule='Per-epsilon finite eligible exploration minimum, exact G then candidate ID tie-break; freeze before confirmation. Keep post-confirmation minimum separate.',
              legacy_cv2_max=max(cvs),cv_based_n_E_requirement=cv_requirement,rarity=rarity,
              maximum_new_trajectories=4*(128+1024),maximum_axis_compiles=8*(128+1024),
              actual_trajectory_cap=len(eligible)*(n_E+1024),new_target_max=4,new_law_max=4,
              independent_cohorts=True,confirmation_top_up=False,old_cohort_top_up=False,unseen_rare_event_contribution_zero=False)
    save(p,plan);save(OUT/'signal_only_results.json',signal);baseline()
    # All planned addresses must be checked before any draw. Old candidates
    # have different r; exploration/confirmation tags differ.
    seeds=[E.trajectory_seed(cid,cohort,i) for cid in eligible for cohort,n in [('exploration',n_E),('confirmation',1024)] for i in range(n)]
    assert len(seeds)==len(set(seeds));save(OUT/'seed_address_check.json',dict(planned_addresses=len(seeds),collisions=0,scheme='20261025,6,L_D,q,r,K,101_or_202,index',not_PRNG_independence_proof=True))
    print(json.dumps(dict(candidates=eligible,n_E=n_E,cv_requirement=cv_requirement,rarity_requirements={k:v['n_for_nominal90pct_at_least_one'] for k,v in rarity.items()},signal=[dict(candidate_id=x['candidate_id'],epsilon=x['epsilon_sig'],status=x['eligibility'],B=x['B_exact'],N=x['N_total']) for x in signal])),flush=True)
def nominate():
    W.check();frozen_analysis();plan=read(OUT/'sampling_design.json');out=OUT/'confirmation_plan.json'
    if out.exists():raise ValueError('CONFIRMATION_ALREADY_FIXED')
    bas=baseline();allrows=[];candidates={};nE=plan['n_E'];c=float(t.ppf(.975,nE-1))
    for cid in plan['candidates']:
        rr=candidate_rows(cid,'exploration');allrows+=rr;summary,prices,_=price_summary(cid,'exploration');requirements=[];floors=[]
        for x in rr:
            if x['G_exact'] is None:continue
            z=[x['N_cos']*p['cosine']+x['N_sin']*p['sine'] for p in prices];variance=float(np.var(z,ddof=1));own_d=.01*float(Q(x['G_exact']))
            own=math.ceil(c*c*variance/own_d**2);requirements.append(own)
            for ref in next(b['refs'] for b in bas if b['epsilon_sig']==x['epsilon_sig']):
                if ref['G_exact'] is None:continue
                denom=(.01*float(Q(ref['G_exact']))/c)**2-float(ref['G_SE'])**2
                if denom<=0:floors.append(dict(epsilon=x['epsilon_sig'],reference=ref['candidate_id'],reason='OLD_SE_FLOOR_TARGET_UNREACHABLE'));continue
                requirements.append(math.ceil(variance/denom))
        raw_n=max(requirements or [64]);n_C=max(64,min(1024,raw_n));candidates[cid]=dict(n_C=n_C,unclipped_required=raw_n,capped=raw_n>1024,floor_limited=floors,zero_exploratory_variance=all(float(Q(v))==0 for row in summary['covariance_exact'].values() for v in row.values()))
    noms={}
    for e in EPS:
        pool=[x for x in allrows if x['epsilon_sig']==e and x['G_exact'] is not None]
        noms[e]=min(pool,key=lambda x:(Q(x['G_exact']),x['candidate_id']))['candidate_id'] if pool else None
    save(out,dict(status='FROZEN_BEFORE_ANY_CONFIRMATION',candidates=candidates,nominees=noms,exploration_n=nE,confirmation_top_up=False,planned_confirmation_trajectories=sum(x['n_C'] for x in candidates.values()),exploration_result_sha256={cid:hashlib.sha256(json.dumps(candidate_rows(cid,'exploration'),sort_keys=True).encode()).hexdigest() for cid in candidates}))
    save(OUT/'exploration_results.json',allrows);print(json.dumps(read(out)),flush=True)
def finite_min(rows):return min((x for x in rows if x['G_exact'] is not None),key=lambda x:(Q(x['G_exact']),x['candidate_id']),default=None)
def compare(x,ref):
    if x is None or ref is None or x.get('G_exact') is None or ref.get('G_exact') is None:return dict(classification='INCONCLUSIVE_MISSING',ratio=None)
    delta=Q(x['G_exact'])-Q(ref['G_exact']);se=math.hypot(float(x['G_SE']),float(ref['G_SE']))
    return dict(classification='INCONCLUSIVE' if abs(float(delta))<=2*se else 'HYBRID_LOWER_EMPIRICAL_G' if delta<0 else 'REFERENCE_LOWER_EMPIRICAL_G',difference_exact=str(delta),combined_SE=se,new_over_reference_exact=str(Q(x['G_exact'])/Q(ref['G_exact'])))
def finish():
    W.check();frozen_analysis();rows=[];price_stats={};law_events={};native=[]
    for cid in read(OUT/'sampling_design.json')['candidates']:
        for cohort in ['exploration','confirmation']:
            rr=candidate_rows(cid,cohort);rows+=rr;summary,values,raw=price_summary(cid,cohort)
            price_stats[cid+'/'+cohort]=summary;orders={str(k):sum(r['orders'].count(k) for r in raw) for k in [0,2,4]};law_events[cid+'/'+cohort]=dict(n=len(raw),event_orders=orders)
    fields=['candidate_id','cohort','epsilon_sig','eligibility','price_n','B_exact','N_cos','N_sin','N_total','C_cos_exact','C_sin_exact','G_exact','G_SE','G_nominal95_halfwidth','bias_cos_exact','bias_sin_exact','u_cos_exact','u_sin_exact','raw_norm_exact']
    csvsave(OUT/'CANDIDATE_RESULTS.csv',[{k:r.get(k) for k in fields} for r in rows]);save(OUT/'candidate_results.json',rows);save(OUT/'price_statistics.json',price_stats);save(OUT/'event_orders.json',law_events)
    cp=read(OUT/'confirmation_plan.json');comp=[];details=[]
    for e in EPS:
        new=finite_min([x for x in rows if x['cohort']=='confirmation' and x['epsilon_sig']==e]);b=next(x for x in baseline() if x['epsilon_sig']==e);rte=b['refs'][-1];h=next(x for x in b['refs'] if x['candidate_id']==b['saved_Hybrid_id'])
        merged=finite_min([new,h] if new else [h]);nom=next((x for x in rows if x['cohort']=='confirmation' and x['epsilon_sig']==e and x['candidate_id']==cp['nominees'][e]),None);d=compare(merged,rte)
        comp.append(dict(epsilon_sig=e,new_Hybrid_id=new['candidate_id'] if new else None,new_Hybrid_G_exact=new['G_exact'] if new else None,new_Hybrid_G_SE=new['G_SE'] if new else None,new_Hybrid_n=new['price_n'] if new else None,
                         expanded_Hybrid_id=merged['candidate_id'],expanded_Hybrid_G_exact=merged['G_exact'],expanded_Hybrid_G_SE=merged['G_SE'],RTE_id=rte['candidate_id'],RTE_G_exact=rte['G_exact'],RTE_G_SE=rte['G_SE'],RTE_n=rte['price_n'],
                         classification=d['classification'],Hybrid_over_RTE_exact=d.get('new_over_reference_exact'),combined_SE=d.get('combined_SE'),nominee_id=cp['nominees'][e],nominee_confirmation_G_exact=nom['G_exact'] if nom else None,
                         minimum_at_r64_boundary=new['config']['r']==64 if new else None,hypothetical_rho='0',population_tail='UNGUARANTEED'))
        primary=[]
        for x in [x for x in rows if x['cohort']=='confirmation' and x['epsilon_sig']==e]:
            anchor=next(z for z in b['refs'] if z['candidate_id']==('PR_p0684_q1_r8_K4' if x['config']['L_D']==684 else 'PR_p0078_q8_r8_K2'))
            primary.append(dict(candidate_id=x['candidate_id'],versus_own_r8=compare(x,anchor),versus_RTE=compare(x,rte)))
        details.append(dict(epsilon_sig=e,new_set_min=new,expanded_set_min=merged,nominee=nom,RTE=rte,saved_Hybrid=h,primary_comparisons=primary))
    csvsave(OUT/'COMPARISON_RESULTS.csv',comp);save(OUT/'comparison_details.json',details)
    for p in sorted((OUT/'stages').glob('*/native.json')):
        d=read(p)
        for x in d['results']:
            assert x['exit_code']==0 and x['reaped'];native.append(dict(stage=p.parent.name,pid=x['pid'],exit_code=0,reaped=True,elapsed_seconds=x['elapsed_seconds']))
    assert len(native)==4+sum(v['n'] for v in price_stats.values())
    save(OUT/'PUBLIC_NATIVE_AUDIT.json',dict(records=native,all_exit_zero=True,all_reaped=True,private_commands_omitted=True))
    summary=dict(status='COMPLETE_CONDITIONAL_FINITE_CANDIDATE_COMPARISON',new_laws=4,new_target_state_actions=4,new_paired_trajectories=sum(v['n'] for v in price_stats.values()),new_axis_compiles=2*sum(v['n'] for v in price_stats.values()),native_children=len(native),all_native_exit0_reaped=True,artificial_test_count_phaseA=27,actual_rho=None,population_tail='UNGUARANTEED',old_14_of_15_unchanged=True,confirmation_top_up=False,science_SOURCE=read(OUT/'SOURCE.json')['commit'],analysis_SOURCE=read(OUT/'ANALYSIS_SOURCE.json')['commit'])
    save(OUT/'EXECUTION_SUMMARY.json',summary);save(OUT/'STUDY_STOP.json',dict(mandatory_STOP=True,science_complete=True,new_science_automatic=False));print(json.dumps(dict(summary=summary,comparison=comp)),flush=True)
if __name__=='__main__':
    import sys
    dict(design=design,nominate=nominate,finish=finish)[sys.argv[1]]()
