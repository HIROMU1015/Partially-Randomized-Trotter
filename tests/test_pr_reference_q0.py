"""Small artificial tests for Q0; no saved scientific input evaluation."""
import importlib.util
from pathlib import Path
from fractions import Fraction as F
import math
import ast
import unittest

spec=importlib.util.spec_from_file_location('q0',Path(__file__).resolve().parents[1]/'scripts/diagnose_pr_reference_q0.py')
q=importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)
q.I=q.setup()


class ArtificialQ0(unittest.TestCase):
    def test_pinned_hoeffding_inversion(self):
        tree=ast.parse(q.raw(q.BASE,q.REUSE[1]))
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_hoeffding_shots')
        ns={'math':math}
        exec(compile(ast.Module(body=[node],type_ignores=[]),q.REUSE[1],'exec'),ns)
        n=ns['_hoeffding_shots'](.25,.01)
        self.assertEqual(q.verdict(q.epsilon(n,F(1,100)),q.I('.25')),'PASS')
        self.assertEqual(q.verdict(q.epsilon(n-1,F(1,100)),q.I('.25')),'FAIL')

    def test_both_sine_branches_in_independent_preimage(self):
        check_spec=importlib.util.spec_from_file_location('q0_check',Path(q.__file__).with_name('check_pr_reference_q0.py'))
        check=importlib.util.module_from_spec(check_spec)
        check_spec.loader.exec_module(check)
        width=check.sine_preimage(q.I('.9'),q.I('.9'),q.I(1),q.I(0),q.I('2.2'),q.I(0))
        expected=q.mp.iv.pi-2*q.asin(q.I('.9'))
        self.assertTrue(check.overlap(width,expected))

    def test_equal_radius_inverse(self):
        for s in ('-.3','0','.4'):
            a=q.asin(q.I(s))
            self.assertTrue(q.endpoints(q.mp.iv.sin(a))[0] <= q.mp.mpf(s) <= q.endpoints(q.mp.iv.sin(a))[1])

    def test_xy_sign_without_circuit(self):
        z=complex(.3,-.4)
        for phi,expected in [(0,.3),(math.pi/2,-.4)]:
            projected=(complex(math.cos(phi),-math.sin(phi))*z).real
            self.assertAlmostEqual(projected,expected)

    def test_all_small_counts_envelope(self):
        for rho in ('1','.6'):
            for D in ('0','.02','.2'):
                for n in range(1,41):
                    a=q.worst(n,q.I(rho),q.I(D),q.I('.8'))[0]
                    b=q.worst(n,q.I(rho),q.I(D),q.I('.8'),full=True)[0]
                    self.assertLessEqual(max(q.endpoints(a)[0],q.endpoints(b)[0]),min(q.endpoints(a)[1],q.endpoints(b)[1]))

    def test_nuisance_expansion(self):
        for k in range(21):
            a=q.hull(20,k,q.I('.8'),q.I('.01'),q.I('.8'))
            b=q.hull(20,k,q.I('.5'),q.I('.03'),q.I('.8'))
            if a is not None:
                self.assertIsNotNone(b)
                self.assertLessEqual(q.endpoints(b[0])[0],q.endpoints(a[0])[1])
                self.assertGreaterEqual(q.endpoints(b[1])[1],q.endpoints(a[1])[0])

    def test_empty_and_reflected_counts(self):
        eps=q.I('.01')
        self.assertIsNone(q.hull(10,10,q.I('.6'),q.I('.01'),q.I('.2'),eps))
        a=q.hull(10,4,q.I('.6'),q.I('.01'),q.I('.8'),eps)
        b=q.hull(10,6,q.I('.6'),q.I('.01'),q.I('.8'),eps)
        self.assertIsNotNone(a)
        self.assertLessEqual(max(q.endpoints(a[2])[0],q.endpoints(b[2])[0]),min(q.endpoints(a[2])[1],q.endpoints(b[2])[1]))

    def test_exact_failure_allocation(self):
        self.assertEqual(34*F(1,720)+F(1,360),F(1,20))
        for n in range(2,16):
            for nx in range(1,n):
                ax=F(1,360)*F(n-nx,n)
                ay=F(1,360)*F(nx,n)
                self.assertEqual(ax+ay,F(1,360))

    def test_alias_intersection_never_widens(self):
        lo,hi=F(-3,5),F(4,5)
        self.assertLessEqual(min(hi,F(7,10))-max(lo,F(-1,2)),hi-lo)

    def test_nonmonotone_domain_rejected_for_future_use(self):
        # The production model only uses a monotone sine branch. An interval
        # crossing pi/2 must not silently use a single inverse branch.
        self.assertEqual(q.verdict(q.I('1.7'),q.mp.iv.pi/2),'FAIL')


if __name__=='__main__':
    unittest.main()
