"""Independent numerical route: rational square roots, 4-state probability DP,
integer raw moments; never calls the primary law/mass/statistics functions.
"""
from fractions import Fraction as F
from math import factorial,isqrt
from decimal import Decimal,localcontext
S=('S0_ONLY','S2_SINGLE','S2_MULTI','S4_PRESENT')
def decimal(f):return Decimal(f.numerator)/Decimal(f.denominator)
def root_rational(x):
 scale=10**130
 return F(isqrt(x.numerator*scale*scale//x.denominator),scale)
def verify(c,result):
 with localcontext() as ctx:
  ctx.prec=110;t=F(c['law']['dimensionless_step_time']);weights={k:abs(t)**k/F(factorial(k))*root_rational(1+t*t/F((k+1)**2)) for k in range(0,c['K']+1,2)};B=sum(weights.values());p={k:v/B for k,v in weights.items()}
  dist=[F(1),F(0),F(0),F(0)]
  for _ in range(c['q']*c['r']):
   a,b,m,f=dist;p0=p[0];p2=p[2];p4=p.get(4,F(0))
   dist=[a*p0,b*p0+a*p2,m*(p0+p2)+b*p2,f+(a+b+m)*p4]
  assert sum(dist)==1
  mass_error=max(abs(decimal(dist[i])-Decimal(result['groups'][i]['probability'])) for i in range(4));assert mass_error<Decimal('1e-85')
  group={s:[] for s in S};allvals=[];counts={str(k):0 for k in (0,2,4)}
  for task in c['tasks']:
   n2=n4=0
   for step in task['events']:
    for event in step:
     k=event['taylor_order'];counts[str(k)]+=1;n2+=int(k==2);n4+=int(k==4)
   idx=3 if n4>0 else 2 if n2>1 else 1 if n2==1 else 0
   value=c['shots']['cosine']*task['axes']['cosine']['RZ']+c['shots']['sine']*task['axes']['sine']['RZ'];group[S[idx]].append(value);allvals.append(value)
  assert counts==result['event_order_counts']
  for name,vals,reported in [(s,group[s],result['groups'][i]['G']) for i,s in enumerate(S)]+[('ALL',allvals,result['saved_finite_sample'])]:
   n=len(vals);assert n==reported['n']
   if not n:assert reported['mean_exact'] is None and reported['SE'] is None;continue
   total=sum(vals);assert F(total,n)==F(reported['mean_exact'])
   if n==1:assert reported['SE'] is None;continue
   variance=F(n*sum(x*x for x in vals)-total*total,n*(n-1));assert variance==F(reported['sample_variance_exact'])
   se=(decimal(variance)/n).sqrt();assert abs(se-Decimal(reported['SE']))<Decimal('1e-85')*max(Decimal(1),abs(se))
  known=sum(dist[i]*F(result['groups'][i]['G']['mean_exact']) for i in range(4) if group[S[i]])
  unknown=sum(dist[i] for i in range(4) if not group[S[i]])
  budget=F(c['PF_G_exact'])-known;threshold=budget/unknown if unknown else None
  err=abs(decimal(known)-Decimal(result['conditional_expectation']['observed_plugin_G']));assert err<Decimal('1e-85')*max(Decimal(1),abs(decimal(known)))
  if threshold is not None:assert abs(decimal(threshold)-Decimal(result['conditional_expectation']['unknown_mixture_mean_threshold_G']))<Decimal('1e-85')*max(Decimal(1),abs(decimal(threshold)))
  return dict(status='PASS_INDEPENDENT_ARITHMETIC_NOT_POPULATION_CERTIFICATE',probability_route='rational integer-isqrt at 130 decimal places + exact Fraction 4-state DP',statistics_route='integer raw moments then exact Fraction, independent centered-moment route',maximum_mass_difference=str(mass_error),mean_SE_and_threshold_match=True,uncertified_rounding_inputs=True)
