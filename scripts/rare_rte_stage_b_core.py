"""Stage B saved-data analysis. No sampling, simulation, circuit or compiler API."""
from decimal import Decimal,localcontext
from fractions import Fraction
from math import factorial,comb
STRATA=('S0_ONLY','S2_SINGLE','S2_MULTI','S4_PRESENT')
PRECISION=100
D=Decimal

def dec(v):
 if isinstance(v,Fraction):return D(v.numerator)/D(v.denominator)
 if isinstance(v,float):return D.from_float(v)
 return D(v)

def finite_law(tau,K):
 assert K in (2,4)
 t=abs(dec(tau));weights={k:(D(1) if k==0 else t**k)/D(factorial(k))*(1+t*t/D((k+1)**2)).sqrt() for k in range(0,K+1,2)}
 B=sum(weights.values());p={k:weights[k]/B for k in weights}
 return p,B,weights

def masses(p,R):
 p0,p2,p4=(p.get(k,D(0)) for k in (0,2,4))
 return dict(zip(STRATA,[p0**R,D(R)*p2*p0**(R-1),
  sum(D(comb(R,j))*p2**j*p0**(R-j) for j in range(2,R+1)),
  sum(D(comb(R,j))*p4**j*(p0+p2)**(R-j) for j in range(1,R+1))]))

def stratum(events):
 orders=[e['taylor_order'] for step in events for e in step]
 assert all(x in (0,2,4) for x in orders)
 n2=orders.count(2);n4=orders.count(4)
 return ('S4_PRESENT' if n4 else 'S2_MULTI' if n2>=2 else 'S2_SINGLE' if n2 else 'S0_ONLY'),orders

def stats(values):
 n=len(values)
 if n==0:return dict(n=0,mean_exact=None,SE=None,sample_variance_exact=None,sample_min=None,sample_max=None)
 mean=Fraction(sum(values),n)
 variance=sum((Fraction(v)-mean)**2 for v in values)/(n-1) if n>1 else None
 return dict(n=n,mean_exact=str(mean),SE=str((dec(variance)/n).sqrt()) if variance is not None else None,
  sample_variance_exact=str(variance) if variance is not None else None,sample_min=min(values),sample_max=max(values))

def analyze(c):
 with localcontext() as ctx:
  ctx.prec=PRECISION
  saved=c['law'];p,B,w=finite_law(saved['dimensionless_step_time'],c['K']);R=c['q']*c['r'];pi=masses(p,R)
  assert abs(sum(pi.values())-1)<D('1e-90')
  saved_p={k:dec(saved['order_probabilities'][i]) for i,k in enumerate(saved['orders'])};saved_sum=sum(saved_p.values());saved_simplex={k:v/saved_sum for k,v in saved_p.items()};saved_pi=masses(saved_simplex,R)
  mass_roundoff=max(abs(pi[s]-saved_pi[s]) for s in STRATA);assert mass_roundoff<D('1e-12')
  p_errors=[abs(p[k]-dec(saved['order_probabilities'][i])) for i,k in enumerate(saved['orders'])]
  assert max(p_errors)<D('1e-12'),'CANONICAL_SAVED_PROBABILITY_MISMATCH'
  assert abs(B-dec(saved['exact_finite_distribution']))<D('1e-12')*max(D(1),B)
  logB=R*B.ln();assert abs(logB-dec(c['saved_log_B']))<D('1e-12')*max(D(1),abs(logB)),'SAVED_SIGNAL_NORMALIZATION_MISMATCH'
  grouped={s:[] for s in STRATA};axis_grouped={s:{a:[] for a in ('cosine','sine')} for s in STRATA};hist={str(k):0 for k in (0,2,4)};task_rows=[];all_costs=[];lambda_first=None;lambda_error=D(0)
  for t in c['tasks']:
   s,orders=stratum(t['events']);assert len(orders)==R
   if c['K']==2:assert 4 not in orders
   if t.get('saved_stratum') is not None:assert s==t['saved_stratum']
   g=sum(c['shots'][a]*t['axes'][a]['RZ'] for a in ('cosine','sine'));grouped[s].append(g);all_costs.append(g)
   for a in ('cosine','sine'):axis_grouped[s][a].append(t['axes'][a]['RZ'])
   for k in (0,2,4):hist[str(k)]+=orders.count(k)
   # Source _make_event: for order0 c_m=a0*p_l, p_l=|h_l|/lambda.
   # Recover the effective lambda from RECORDED coefficients, with source normalization rounding.
   for step in t['events']:
    for e in step:
     if e['taylor_order']==0:
      lam=dec(e['application_sequence'][0]['coefficient_abs'])*dec(saved['unnormalized_order_weights'][0])/dec(e['event_coefficient'])
      if lambda_first is None:lambda_first=lam
      tau=dec(float(.8/c['q']))*lam/c['r'];err=abs(tau-dec(saved['dimensionless_step_time']))/max(abs(dec(saved['dimensionless_step_time'])),D('1e-100'))
      lambda_error=max(lambda_error,err)
   task_rows.append(dict(candidate=c['candidate'],replica=t['replica'],stratum=s,order2=orders.count(2),order4=orders.count(4),G_exact=str(g),RZ_cosine=t['axes']['cosine']['RZ'],RZ_sine=t['axes']['sine']['RZ'],event_digest=t['event_digest'],trace_sha256=t['trace_sha256']))
  assert lambda_first is not None and lambda_error<D('1e-12'),'RECORDED_COMPONENT_LAMBDA_TAU_MISMATCH'
  overall=stats(all_costs);assert Fraction(overall['mean_exact'])==Fraction(c['mean_exact']),'SAVED_MEAN_NOT_REPRODUCED'
  assert abs(D(overall['SE'])-D(str(c['paired_SE'])))<D('1e-12')*max(D(1),abs(D(overall['SE']))),'SAVED_SE_NOT_REPRODUCED'
  groups=[];known=D(0);unknown=[]
  for s in STRATA:
   st=stats(grouped[s]);supported=pi[s]>0
   if st['n']:
    assert supported;known+=pi[s]*dec(Fraction(st['mean_exact']))
   elif supported:unknown.append(dict(stratum=s,coefficient=str(pi[s]),mean_G=None,variance_G=None))
   groups.append(dict(stratum=s,probability=str(pi[s]),support_status='IN_SUPPORT' if supported else 'OUTSIDE_K2_SUPPORT',
    G=st,RZ_by_axis={a:stats(axis_grouped[s][a]) for a in ('cosine','sine')},conditional_population_mean_G=None,
    observation_status='OBSERVED' if st['n'] else 'UNOBSERVED' if supported else 'NOT_APPLICABLE_SUPPORT_ZERO'))
  U=sum(D(x['coefficient']) for x in unknown);pf=dec(Fraction(c['PF_G_exact']));budget=pf-known
  if U:
   threshold=budget/U
   kind='PR_LOWER_IFF_UNKNOWN_MIXTURE_MEAN_BELOW_THRESHOLD' if budget>0 else 'PF_LOWER_FOR_ALL_NONNEGATIVE_UNKNOWN_MEANS_GIVEN_OBSERVED_PLUGIN' if budget<0 else 'BOUNDARY_AT_ZERO_UNKNOWN_CONTRIBUTION'
  else:threshold=None;kind='PLUGIN_PR_LOWER' if budget>0 else 'PLUGIN_PF_LOWER' if budget<0 else 'PLUGIN_TIE'
  s0=next(g for g in groups if g['stratum']=='S0_ONLY')['G']['mean_exact']
  return dict(condition=c['condition'],candidate=c['candidate'],system=c['system'],rank=c['rank'],epsilon=c['epsilon'],q=c['q'],r=c['r'],K=c['K'],R=R,n=c['n'],SOURCE=c['SOURCE'],result_commit=c['result_commit'],cohort=c['cohort'],shots=c['shots'],fixed_PF=c['fixed_PF'],fixed_PF_G_exact=c['PF_G_exact'],
   law=dict(definition='CANONICAL_FINITE_RTE_AT_EXACT_SAVED_BINARY64_TAU',tau_saved_repr=repr(saved['dimensionless_step_time']),tau_exact_binary64=str(dec(saved['dimensionless_step_time'])),saved_law=saved,canonical_p={str(k):str(v) for k,v in p.items()},canonical_B=str(B),canonical_B_power_R=str(B**R),canonical_log_B_total=str(logB),saved_log_B=c['saved_log_B'],saved_probability_sum_minus_one=str(saved_sum-1),saved_simplex_roundoff_diagnostic={s:str(v) for s,v in saved_pi.items()},maximum_mass_roundoff_diagnostic=str(mass_roundoff),saved_law_modified=False,maximum_saved_p_difference=str(max(p_errors)),
    lambda_R_recovered=str(lambda_first),lambda_provenance='RECOVERED_FROM_RECORDED_ORDER0_EVENT_COEFFICIENT_AND_ABS_COMPONENT; NOT_INDEPENDENT_FULL_TAIL_NORM_AUDIT',maximum_lambda_tau_relative_discrepancy=str(lambda_error),binary64_rng_bit_law_certified=False,interval_certificate=False),
   saved_finite_sample=overall,event_order_counts=hist,groups=groups,conditional_expectation=dict(definition='sum observed pi_s*empirical_mean_s + sum unobserved pi_s*unknown_population_mean_s',observed_plugin_G=str(known),unknown_terms=unknown,unknown_mass=str(U),PF_minus_observed_plugin=str(budget),unknown_mixture_definition='sum_unknown pi_s*mu_s / sum_unknown pi_s; NOT equal-cost assumption',unknown_mixture_mean_threshold_G=str(threshold) if threshold is not None else None,threshold_over_observed_S0_mean=str(threshold/dec(Fraction(s0))) if threshold is not None and s0 else None,conditional_direction=kind,unconditional_population_mean_G=None,mathematically_guaranteed_cost_upper_bound=None,confidence_interval=None,zero_filled=False,known_means_are_empirical=True),tasks=task_rows)
