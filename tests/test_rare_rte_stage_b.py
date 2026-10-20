"""Synthetic fixtures only. Real-input numeric analysis is restricted to one-shot."""
import unittest,sys
from pathlib import Path
from decimal import Decimal,localcontext
from fractions import Fraction
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import rare_rte_stage_b_core as c
import rare_rte_stage_b_verify as v
class Synthetic(unittest.TestCase):
 def test_multievent_mass_independent_DP(self):
  with localcontext() as ctx:
   ctx.prec=100
   for K in (2,4):
    for tau in ('0.1','-0.25','1e-20'):
     p,B,w=c.finite_law(Decimal(tau),K)
     for R in (1,2,8,64):
      mass=c.masses(p,R);self.assertLess(abs(sum(mass.values())-1),Decimal('1e-90'))
      self.assertTrue(all(x>=0 for x in mass.values()))
      self.assertEqual(mass['S4_PRESENT']==0,K==2)
 def test_tau_zero_support(self):
  with localcontext() as x:
   x.prec=100;p,_,_=c.finite_law(Decimal(0),4);self.assertEqual(c.masses(p,8)['S0_ONLY'],1)
 def test_positive_tiny_mass_not_zero_filled(self):
  with localcontext() as x:
   x.prec=100;p,_,_=c.finite_law(Decimal('1e-40'),4);self.assertGreater(c.masses(p,64)['S4_PRESENT'],0)
 def test_missing_single_SE(self):
  self.assertIsNone(c.stats([])['mean_exact']);self.assertIsNone(c.stats([23])['SE'])
 def test_paired_variance_includes_covariance(self):
  with localcontext() as x:
   x.prec=100
   a=c.stats([100,400,900]);self.assertEqual(Fraction(a['mean_exact']),Fraction(1400,3));self.assertEqual(Fraction(a['sample_variance_exact']),Fraction(490000,3))
 def test_strata_are_disjoint(self):
  for orders,expected in [([0,0],c.STRATA[0]),([2,0],c.STRATA[1]),([2,2],c.STRATA[2]),([2,4],c.STRATA[3])]:
   self.assertEqual(c.stratum([[dict(taylor_order=k) for k in orders]])[0],expected)
 def test_unknown_formula_no_cost_assumption(self):
  # Aggregate unknown mean is a definition, not an equal-cost or zero-cost prior.
  p0,p1=Fraction(1,5),Fraction(3,10);m0,m1=17,31
  self.assertEqual((p0*m0+p1*m1)/(p0+p1),Fraction(127,5))
 def test_independent_square_root(self):
  f=v.root_rational(Fraction(2));self.assertLess(abs(f*f-2),Fraction(1,10**128))
 def test_order4_support_and_multi_position(self):
  self.assertEqual(c.stratum([[dict(taylor_order=4),dict(taylor_order=2)]])[0],'S4_PRESENT')
 def test_illegal_order_rejected(self):
  with self.assertRaises(AssertionError):c.stratum([[dict(taylor_order=3)]])
 def fixture(self,orders):
  with localcontext() as ctx:
   ctx.prec=100;p,B,w=c.finite_law(.2,4)
   law=dict(dimensionless_step_time=.2,finite_taylor_order=4,orders=[0,2,4],unnormalized_order_weights=[float(w[k]) for k in (0,2,4)],order_probabilities=[float(p[k]) for k in (0,2,4)],exact_finite_distribution=float(B))
   traces=[]
   for i,ks in enumerate(orders):
    es=[dict(taylor_order=k,application_sequence=[dict(coefficient_abs=1.)],event_coefficient=float(w[0])) for k in ks]
    traces.append(dict(replica=i,events=[es],axes={a:dict(RZ=50*(i+1)) for a in ('cosine','sine')},event_digest='synthetic',trace_sha256='synthetic',saved_stratum=None))
   n=len(orders);mean=Fraction(100*(n+1),2);variance=Fraction(10000*n*(n+1),12)
   return dict(condition='SYNTHETIC',candidate='fixture',system='toy',rank=1,epsilon=.05,q=1,r=4,K=4,n=n,SOURCE='synthetic',result_commit='synthetic',cohort='SYNTHETIC_ONLY',shots=dict(cosine=1,sine=1),fixed_PF='fixed',PF_G_exact='1000',mean_exact=str(mean),paired_SE=str((c.dec(variance)/n).sqrt()),law=law,saved_log_B=float(4*B.ln()),tasks=traces)
 def test_complete_primary_and_independent_route(self):
  x=self.fixture([[0,0,0,0],[0,0,0,2],[0,0,2,2],[0,0,0,4]])
  y=c.analyze(x);self.assertEqual(y['conditional_expectation']['unknown_terms'],[]);self.assertTrue(v.verify(x,y)['mean_SE_and_threshold_match'])
 def test_unobserved_means_remain_unknown(self):
  x=self.fixture([[0,0,0,0],[0,0,0,0]]);y=c.analyze(x);self.assertEqual(len(y['conditional_expectation']['unknown_terms']),3)
  self.assertTrue(all(z['mean_G'] is None for z in y['conditional_expectation']['unknown_terms']));self.assertTrue(v.verify(x,y)['mean_SE_and_threshold_match'])
 def test_negative_boundary_kept(self):
  x=self.fixture([[0,0,0,0],[0,0,0,0]]);x['PF_G_exact']='1';y=c.analyze(x)
  self.assertLess(Decimal(y['conditional_expectation']['unknown_mixture_mean_threshold_G']),0);self.assertTrue(v.verify(x,y)['mean_SE_and_threshold_match'])
 def test_saved_mean_drift_rejected(self):
  x=self.fixture([[0,0,0,0],[0,0,0,0]]);x['mean_exact']='1'
  with self.assertRaises(AssertionError):c.analyze(x)
if __name__=='__main__':unittest.main()
