"""B0 artificial fixtures; no saved-input scientific campaign here."""
import importlib.util
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import pr_reference_feedback_q3b as b
q1=b.q1;q=q1._q0();F=b.F
spec=importlib.util.spec_from_file_location('b0_old_fixtures',ROOT/'tests/test_pr_reference_feedback_q1.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)


def history(theta=0.,n=4096):
    # Reuse the original artificial count generator; bind Q1's same data type.
    return tuple(q1.RoundCounts(r.m,q1.AxisCounts(r.X.n,r.X.plus),q1.AxisCounts(r.Y.n,r.Y.plus))
                 for r in old.artificial_history(theta,n))


class IntervalBranchTests(unittest.TestCase):
    def test_positive_negative_and_legacy_diagnostic(self):
        for theta in (.0123,-.0317):
            p=b.prefix_from_counts(history(theta))
            self.assertEqual(p.legacy_result['status'],'AVAILABLE')
            self.assertFalse(p.legacy_result['basis_for_selection'])
            self.assertTrue(all(c['unique'] for c in p.branch_certificates))
            error=q1._circular_distance(p.base_phase_center-q.I(theta))
            self.assertLessEqual(q.endpoints(error)[1],q.endpoints(q.iv(q1.CAP)/(1<<16))[0])

    def test_strict_integer_cells_and_ties(self):
        for j in (-10,-1,0,1,10):
            self.assertEqual(b.unique_integer_cell(q.iv(F(j)+F(1,8))),j)
        for value in (q.iv(F(1,2)),q.iv(F(-1,2)),q.I(['.49','.51'])):
            with self.assertRaises(q1.UnqualifiedPrefix):b.unique_integer_cell(value)

    def test_period_cut_with_small_base_phase(self):
        for theta in (math.pi/64,-math.pi/64):
            h=history(theta,1<<16)
            self.assertEqual(h[6].Y.mean,0)
            p=b.prefix_from_counts(h,include_legacy=False)
            self.assertLess(b.width(p.energy_center),b.KAPPA_LIMIT)
            f=b.feedback_from_prefix(p)
            self.assertLess(f.arithmetic_error_upper,b.ARITHMETIC_ANGLE_LIMIT)

    def test_zero_ambiguous_and_wrong_alias_rejected(self):
        h=list(history());h[0]=q1.RoundCounts(0,q1.AxisCounts(2,1),q1.AxisCounts(2,1))
        with self.assertRaises(q1.UnqualifiedPrefix):b.prefix_from_counts(h)
        h=list(history());h[1]=q1.RoundCounts(1,q1.AxisCounts(2,0),q1.AxisCounts(2,1))
        with self.assertRaises(q1.UnqualifiedPrefix):b.prefix_from_counts(h)
        with self.assertRaises(q1.UnqualifiedPrefix):b.prefix_from_counts(history(math.pi))

    def test_legacy_failure_cannot_choose_branch(self):
        with patch.object(q1,'_legacy_decoder',side_effect=RuntimeError('injected diagnostic failure')):
            p=b.prefix_from_counts(history())
        self.assertEqual(p.legacy_result['status'],'UNAVAILABLE')
        self.assertEqual(b.bounds(p.base_phase_center),(F(0),F(0)))

    def test_both_arm_binding_feedback_and_decoder(self):
        for arm in ('Bprime','Reference_B1'):
            with patch.object(q1,'_saved_prefix_shots',return_value=(4096,)*17):
                p=b.prefix_from_counts(history(),saved_arm=arm,include_legacy=False)
            f=b.feedback_from_prefix(p)
            model=q1.SignalModel(F(3,4),F(1,100),saved_arm=arm)
            result=b.decode_counts(p,f,170,model,additional_feedback_error=F(1,10000))
            self.assertEqual(result['status'],'CONDITIONAL_WIDTH_SUFFICIENT')
            self.assertFalse(result['formal_RPE_or_resource_qualification'])
            with patch.object(q1,'_saved_prefix_shots',return_value=(4095,)*17):
                with self.assertRaises(q1.UnqualifiedPrefix):b.prefix_from_counts(history(),saved_arm=arm)


if __name__=='__main__':unittest.main()
