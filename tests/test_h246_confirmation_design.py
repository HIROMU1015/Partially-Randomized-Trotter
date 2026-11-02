import importlib.util
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h246_confirmation_design as d

class DesignTests(unittest.TestCase):
    def sample(self,mean=100,variance=100):return dict(total_RZ=mean,total_RZ_variance=variance)
    def test_marginal_closed_form(self):self.assertEqual(d.marginal_n(100,100,.02,4,9),(100,False))
    def test_zero_variance_unresolved(self):self.assertEqual(d.marginal_n(100,0,.02,4,9),(9,True))
    def test_inflation_and_precision_monotonic(self):
        a,b=self.sample(),self.sample(60,64)
        base=d.pair_plan(a,b)
        self.assertTrue(all(x>=y for x,y in zip(d.pair_plan(a,b,factor=8)['n_C'],base['n_C'])))
        self.assertTrue(all(x>=y for x,y in zip(d.pair_plan(a,b,rho=.01)['n_C'],base['n_C'])))
    def test_close_gap_requires_more(self):
        a=self.sample();far=d.pair_plan(a,self.sample(50))['n_C'];near=d.pair_plan(a,self.sample(99))['n_C']
        self.assertGreater(sum(near),sum(far))
    def test_nominal_goals_hold(self):
        a,b=self.sample(),self.sample(80)
        p=d.pair_plan(a,b)
        self.assertLessEqual(p['planned_normal_halfwidth_over_observed_gap'],.2)
    def test_reversed_and_unresolved_not_reselected(self):
        r=d.classify_confirmation(dict(mean=90,SE=10),dict(mean=100,SE=10),z=2.4)
        self.assertEqual(r['point_status'],'REVERSED');self.assertEqual(r['approximate_direction'],'UNRESOLVED')
        self.assertFalse(r['reselection']);self.assertEqual(r['population_rank'],'NOT_CERTIFIED')
    def test_bad_direction_or_variance_refused(self):
        for b in (self.sample(101),dict(total_RZ=50,total_RZ_variance=None)):
            with self.assertRaises(ValueError):d.pair_plan(self.sample(),b)

if __name__=='__main__':unittest.main()
