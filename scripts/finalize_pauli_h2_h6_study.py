#!/usr/bin/env python3
"""Independent confirmation extensions and saved-value final comparisons."""
import csv,hashlib,importlib.util,json,math,sys
from fractions import Fraction as Q
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('hx_batch_runner',ROOT/'scripts/pauli_h2_h6_study.py');R=importlib.util.module_from_spec(s);s.loader.exec_module(R)
def extend(n):
 from scipy.stats import t
 reg={c['candidate_id']:c for c in R.registry(n)};folder=R.OUT/f'H{n}/confirmation01';plan=R.read(folder/'plan.json')
 for level in [96,192,384]:
  results=[]
  for cid in plan['finalists']:results.extend(R.accounting(reg[cid],'confirmation'))
  selected=set();reasons=[]
  det=R.read(R.OUT/f'H{n}/best_exploration.json')
  for e in R.EPS:
   rows=[x for x in results if x['epsilon_sig']==e and x['G_exact'] is not None]
   d=next(x for x in det if x['epsilon_sig']==e)['deterministic']
   for x in rows:
    ns=x['price_statistics']['n'];G=float(Q(x['G_exact']));se=x['G_SE'] or 0
    half=float(t.ppf(.975,ns-1))*se
    close_d=d is not None and abs(G-float(Q(d['G_exact'])))<=2*se
    close_candidates=[y['candidate_id'] for y in rows if y['candidate_id']!=x['candidate_id'] and abs(G-float(Q(y['G_exact'])))<=2*math.hypot(se,y['G_SE'] or 0)]
    if ns<level and half>.01*G and (close_d or close_candidates):
     selected.add(x['candidate_id']);reasons.append(dict(epsilon=e,candidate=x['candidate_id'],n=ns,relative_halfwidth=half/G,close_to_deterministic=close_d,close_candidates=close_candidates))
  if not selected:break
  d=folder/f'extension_{level}';d.mkdir(exist_ok=False)
  R.save(d/'plan.json',dict(molecule=n,finalists=sorted(selected),new_total_n=level,reasons=reasons,
     SOURCE=R.source(n),analysis_sha256=R.sha(Path(__file__)),no_pooling=True,nominal_CI_not_sequential_valid=True))
  jobs=[]
  for cid in sorted(selected):
   c=reg[cid];ns=R.statistics(c,'confirmation')['n']
   for i in range(ns,level):jobs.append(([R.COMP,str(ROOT/'scripts/pauli_h2_h6_study.py'),'price',str(ROOT/c['batch']),cid,'confirmation',str(i)],d/'native'/f'{cid}_{i}'))
  R.run_jobs(jobs,d/'native.json')
 results=[]
 for cid in plan['finalists']:results.extend(R.accounting(reg[cid],'confirmation'))
 R.save(folder/'results.json',results)
 print('independent confirmation finished',n,[(c,R.statistics(reg[c],'confirmation')['n']) for c in plan['finalists']],flush=True)

def final(n):
 from scipy.stats import t
 deterministic=R.read(R.OUT/f'H{n}/best_exploration.json');confirmation=R.read(R.OUT/f'H{n}/confirmation01/results.json');plan=R.read(R.OUT/f'H{n}/confirmation01/plan.json');allrows=R.read(R.OUT/f'H{n}/candidate_results.json')
 result=[]
 for old in deterministic:
  e=old['epsilon_sig'];p=R.A.finite_minimum([x for x in confirmation if x['epsilon_sig']==e]);d=old['deterministic']
  nominee_id=plan['nominees'][e][0] if plan['nominees'][e] else None
  nominee=next((x for x in confirmation if x['epsilon_sig']==e and x['candidate_id']==nominee_id),None)
  ratio=None if not p or not d else Q(p['G_exact'])/Q(d['G_exact'])
  rr=dict(molecule=f'H{n}',epsilon_sig=e,deterministic=d,B0=old['B0'],PF=old['PF'],
          PR_exploration=old['PR_exploration'],PR_confirmed_nominee=nominee,PR_confirmation_set_minimum=p,
          PR_over_D_exact=None if ratio is None else str(ratio),PR_over_D=None if ratio is None else float(ratio),
          selection_scope='MINIMUM_OF_INDEPENDENTLY_CONFIRMED_FROZEN_FINALISTS; post-confirmation selection explicit',
          independent_rho_guard_comparison=R.A.compare_ranges(d['independent_guard_range_exact'] if d else None,p['independent_guard_range_exact'] if p else None),
          actual_rho=None,PR_population_tail_bound=None,formal_certificate_accepted=False)
  if p:
   selected=p['selected'];ns=selected['price_statistics']['n'];se=selected['G_SE'];half=float(t.ppf(.975,ns-1))*se
   rr['nominal_G_t95']=[float(Q(p['G_exact']))-half,float(Q(p['G_exact']))+half]
   rr['nominal_PR_D_t95']=None if not d else [x/float(Q(d['G_exact'])) for x in rr['nominal_G_t95']]
   rr['nominal_CI_scope']='paired finite-sample t; no population-tail, selection or sequential-coverage guarantee'
  rr['registered_counts']={method:sum(x['config']['method']==method and x['epsilon_sig']==e for x in allrows) for method in ['B0','PF','PR']}
  rr['eligibility_counts']={status:sum(x['eligibility']==status and x['epsilon_sig']==e for x in allrows) for status in ['CONDITIONAL_GUARD_ELIGIBLE','BIAS_INELIGIBLE','NUMERICALLY_UNDETERMINED']}
  result.append(rr)
 R.save(R.OUT/f'H{n}/final_results.json',result)
 return result

def tables():
 rows=[]
 for n in [2,6]:
  for x in final(n):
   for family,minimum in [('B0',x['B0']),('full_PF',x['PF']),('PR_confirmed_best',x['PR_confirmation_set_minimum'])]:
    z=minimum['selected'] if minimum else None;st=z['price_statistics'] if z else None;cfg=z['config'] if z else {}
    rows.append(dict(molecule=f'H{n}',epsilon_sig=x['epsilon_sig'],family=family,candidate_id=z['candidate_id'] if z else None,
      L_D=cfg.get('L_D'),q=cfg.get('q'),r=cfg.get('r') if family.startswith('PR') else None,K=cfg.get('K') if family.startswith('PR') else None,
      C_cos_exact=st['mean_exact']['cosine'] if st else None,C_sin_exact=st['mean_exact']['sine'] if st else None,
      price_n=st['n'] if st else None,C_cos_SE=st['SE']['cosine'] if st else None,C_sin_SE=st['SE']['sine'] if st else None,
      bias_cos_exact=z['target']['axes']['cosine']['signed_bias_exact'] if z else None,bias_sin_exact=z['target']['axes']['sine']['signed_bias_exact'] if z else None,
      u_cos_exact=z['target']['axes']['cosine']['u_enc_exact'] if z else None,u_sin_exact=z['target']['axes']['sine']['u_enc_exact'] if z else None,
      B_exact=z['target']['B_saved_exact'] if z else None,N_cos=z['N_cos'] if z else None,N_sin=z['N_sin'] if z else None,
      G_exact=z['G_exact'] if z else None,G_SE=z['G_SE'] if z else None,eligibility=z['eligibility'] if z else 'NO_ELIGIBLE_REGISTERED_CANDIDATE',
      PR_over_D_exact=x['PR_over_D_exact'] if family.startswith('PR') else None,
      cohort='independent_confirmation' if family.startswith('PR') else 'deterministic',source=z['case_path'] if z else None,
      actual_rho=None,PR_population_tail_bound=None))
 R.save(R.OUT/'H2_H6_best_comparison.json',rows)
 with (R.OUT/'H2_H6_best_comparison.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 combined=[]
 for n in [2,6]:
  for x in R.read(R.OUT/f'H{n}/final_results.json'):
   d=x['deterministic'];p=x['PR_confirmation_set_minimum'];ds=d['selected'] if d else None;ps=p['selected'] if p else None
   combined.append(dict(molecule=f'H{n}',epsilon_sig=x['epsilon_sig'],deterministic_candidate=ds['candidate_id'] if ds else None,
     deterministic_G_exact=d['G_exact'] if d else None,PR_candidate=ps['candidate_id'] if ps else None,PR_G_exact=p['G_exact'] if p else None,
     PR_over_D_exact=x['PR_over_D_exact'],PR_over_D=x['PR_over_D'],PR_scope='L_D/q/r/K finite search + independent confirmation',
     H4_PR_optimized=False,actual_rho=None,PR_population_tail_bound=None,source=f'H{n}/final_results.json'))
 # H4 is copied from saved finite-set results; its PR was a fixed candidate.
 h4=R.read(ROOT/'artifacts/non_df_h4_pauli_b0_comparison/2026-10-25/result_summary.json')
 R.save(R.OUT/'H4_saved_summary_reference.json',dict(path='artifacts/non_df_h4_pauli_b0_comparison/2026-10-25/result_summary.json',
   commit='fbfbfd21a252a4a32f52e3b40e50acd91b3c626a',sha256=R.sha(ROOT/'artifacts/non_df_h4_pauli_b0_comparison/2026-10-25/result_summary.json'),summary_top_level_keys=list(h4)))
 # Exact stored values, not a repeated signal/shot/price calculation.
 with (ROOT/'artifacts/non_df_h4_pauli_b0_comparison/2026-10-25/best_by_precision.csv').open() as f:h4rows=list(csv.DictReader(f))
 R.save(R.OUT/'H4_saved_best_rows.json',h4rows)
 return combined
if __name__=='__main__':
 if sys.argv[1]=='extend':extend(int(sys.argv[2]))
 elif sys.argv[1]=='tables':R.save(R.OUT/'H2_H6_comparison_core.json',tables())
 else:raise ValueError(sys.argv[1])
