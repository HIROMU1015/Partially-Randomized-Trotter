"""Q3-A proof-obligation fixtures only; no saved science/count evaluation.

Import Q1's artificial fixtures without running its old test suite. Fault
injections establish the need for numerical contracts, not counterexamples
to the unchanged production library on the fixed scientific domain.
"""
import copy
import importlib.util
import math
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('q3a_q1_fixtures',ROOT/'tests/test_pr_reference_feedback_q1.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
q1,q=old.q1,old.q
F=q1.F


class GuaranteeConnectionTests(unittest.TestCase):
    def test_near_cap_artificial_history_is_accepted(self):
        theta=F(1,16);n=1<<16;history=[]
        for m in range(17):
            phase=float((1<<m)*theta)+(.39 if m%2 else -.39)
            kx=round(n*(1+.75*math.cos(phase))/2)
            ky=round(n*(1+.75*math.sin(phase))/2)
            row=q1.RoundCounts(m,q1.AxisCounts(n,kx),q1.AxisCounts(n,ky))
            distance=q1._circular_distance(q1._count_phase(row)-q.iv((1<<m)*theta))
            self.assertLess(q.endpoints(distance)[1],q.endpoints(q.iv(q1.CAP))[0])
            history.append(row)
        p=q1.prefix_from_counts(history,h=F(1,8))
        error=q1._circular_distance(p.base_phase_center-q.iv(theta))
        self.assertLessEqual(q.endpoints(error)[1],q.endpoints(q.iv(q1.CAP)/p.q_last)[0])
        self.assertTrue(all(r['nearest_branch_verified'] for r in p.branch_certificates))

    def test_full_prefix_period_cuts(self):
        for theta in (math.pi/64,-math.pi/64):
            history=old.artificial_history(theta,n=1<<16)
            self.assertEqual(history[6].Y.mean,0)
            self.assertLess(history[6].X.mean,0)
            p=q1.prefix_from_counts(history,h=F(1,8))
            self.assertTrue(all(r['nearest_branch_verified'] for r in p.branch_certificates))
            self.assertLess(abs(float(sum(q.endpoints(p.base_phase_center))/2)-theta),float(q1.CAP)/p.q_last)

    def test_wrong_binary_branch_is_not_silently_accepted(self):
        history=old.artificial_history()
        p=q1.prefix_from_counts(history,h=F(1,8))
        wrong=copy.deepcopy(p.legacy_result)
        wrong['rounds'][-1]['selected_branch']=1
        wrong['rounds'][-1]['base_phase_estimate']=2*math.pi/(1<<16)
        with patch.object(q1,'_legacy_decoder',return_value=lambda *args:wrong):
            with self.assertRaises(q1.UnqualifiedPrefix):
                q1.prefix_from_counts(history,h=F(1,8))

    def test_containment_without_a_width_bound_does_not_prove_acceptance(self):
        history=old.artificial_history();original=q1._count_phase
        def wide(row):
            return q.I([-4,4]) if row.m==16 else original(row)
        with patch.object(q1,'_count_phase',side_effect=wide):
            with self.assertRaises(q1.UnqualifiedPrefix):
                q1.prefix_from_counts(history,h=F(1,8))

    def test_binary_error_enclosure_against_artificial_high_precision_oracle(self):
        n=1<<32
        for kx,ky in ((n,n//2),(0,n//2),(n,n),(0,n//2-1),(0,n//2+1)):
            row=q1.RoundCounts(16,q1.AxisCounts(n,kx),q1.AxisCounts(n,ky))
            f=q1._feedback_phase(q1._count_phase(row))
            self.assertLess(f.arithmetic_error_upper,F(1,1<<50))
            # Only this artificial point oracle uses 384 bits. The production
            # adapter stays at 192 bits before and after every comparison.
            with q.mp.workprec(384):
                def exact(v):return q.mp.mpf(v.numerator)/v.denominator
                x,y=row.X.mean,row.Y.mean
                phase=-q.mp.pi if not y and x<0 else q.mp.atan2(exact(y),exact(x))
                target=2*phase+q.mp.pi/2+2*f.period_shift*q.mp.pi
                self.assertLessEqual(abs(q.mp.mpf(f.parameter)-target),exact(f.arithmetic_error_upper))
            self.assertEqual((q.mp.mp.prec,q.mp.iv.prec),(192,192))


if __name__=='__main__':unittest.main()
