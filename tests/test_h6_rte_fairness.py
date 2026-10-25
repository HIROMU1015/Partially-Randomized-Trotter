"""Artificial integer-boundary and unchanged-guard checks; no molecule runs."""
import importlib, sys, types, unittest
from fractions import Fraction as Q
from pathlib import Path
p=types.ModuleType('fair_test');p.__path__=[str(Path(__file__).resolve().parents[1]/'src/trotterlib')];sys.modules[p.__name__]=p
F=importlib.import_module(p.__name__+'.h6_rte_fairness')

def axis(n, s='1', u='0'):
    return dict(s_exact=s,u_enc_exact=u,scale=dict(A_lower=str(n),A_upper=str(n)))

class Checks(unittest.TestCase):
    def test_large_exact_and_fractional_ceil(self):
        for bound in [Q(2**52),Q(2**100)+Q(1,3),Q(10**80)-Q(1,7)]:
            a=axis(bound);out=F.integer_shot(a,0)
            self.assertEqual(out['N'],-(-bound.numerator//bound.denominator))
            self.assertTrue(out['sufficient_inequality_verified'])
            self.assertGreaterEqual(Q(out['N']),bound)
    def test_small_original_unchanged(self):
        for bound in [Q(1,3),Q(100),Q(2**52)-1]:
            a=axis(bound);self.assertEqual(F.integer_shot(a,0),F.A.shot(a,0))
    def test_guard_failures_not_promoted(self):
        for a,rho in [(axis(2**100,u='1/2'),0),(axis(2**100,s='-1'),0),(axis(2**100),None)]:
            self.assertEqual(F.integer_shot(a,rho),F.A.shot(a,rho))
            self.assertIsNone(F.integer_shot(a,rho)['N'])
    def test_headroom_and_allowance_preserved(self):
        a=axis(2**80,u='1/1000');x=F.integer_shot(a,Q(1,1000));h=Q(998,1000)
        self.assertEqual(Q(x['headroom_exact']),h)
        self.assertGreaterEqual(x['N']*h*h,2**80)
    def test_mean_G_exact_at_large_integer(self):
        n=10**50+1
        target=dict(axes={a:dict(signed_bias_exact='0',u_enc_exact='0') for a in F.A.AXES})
        out=F.account(target,{a:'7/3' for a in F.A.AXES},'.05',scale=dict(A_lower=str(n),A_upper=str(n)))
        self.assertEqual(Q(out['G_exact']),Q(7,3)*out['N_total'])
        self.assertTrue(all(v['rho_zero']['sufficient_inequality_verified'] for v in out['axes'].values()))

if __name__=='__main__':unittest.main()
