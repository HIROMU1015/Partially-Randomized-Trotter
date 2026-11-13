"""Artificial exact-arithmetic checks; no saved scientific inputs."""
from fractions import Fraction as F
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from trottertracks import rpe_cost_preliminary as m


class ArithmeticTests(unittest.TestCase):
    def cells(self):
        return [dict(m=r,axis=a,N_nominal=n,S=s,lower=s-1,upper=s+1,T=2**r)
                for r,n,s in [(0,2,3),(1,1,8)] for a in ('X','Y')]

    def test_exact_weighted_sum_shares_and_box(self):
        out=m.structural(self.cells(),tails=(1,))
        self.assertEqual(out['total'],28)
        self.assertEqual(out['tails'][0]['share'],F(4,7))
        self.assertEqual(sum(v['share'] for v in out['rounds']),1)
        self.assertEqual(out['numerical_enclosure'],[22,34])
        lo,hi=out['tails'][0]['share_enclosure']
        for early in (8,16):
            for late in (14,18):self.assertTrue(lo<=F(late,early+late)<=hi)
        self.assertIsNone(out['G']);self.assertIsNone(out['winner'])

    def test_missing_and_zero_denominator_are_unknown(self):
        cells=self.cells();cells[0]['S']=None
        out=m.structural(cells,tails=(1,))
        self.assertIsNone(out['total']);self.assertIsNone(out['rounds'][1]['share'])
        self.assertIsNone(m.ratio(F(1),F(0)))
        self.assertIsNone(m.ratio(None,F(1)))
        with self.assertRaises(ValueError):m.exact(float('nan'))

    def test_exact_source_rational_strings_and_huge_integer(self):
        self.assertEqual(m.exact('375/2'),F(375,2))
        huge=2**1200+1
        encoded=m.number(F(huge,3))
        self.assertEqual(m.exact(encoded['exact']),F(huge,3))
        self.assertIsNone(encoded['display'])

    def test_pilot_uses_only_existing_summary_and_does_not_make_CI(self):
        row=dict(m=0,cohort='D_P2_DIRECT_PILOT_V1',role='DEVELOPMENT',expectation='FINITE_COHORT_ONLY',n_paths=4,
            C=dict(mean=['9','9']),S=dict(mean=['6','6']),X=dict(mean_D='3'),Y=dict(mean_D='3'),
            XY_covariance_D=None,uncertainty=dict(law_distance=None))
        ratios=m.pilot_ratios([row])
        self.assertEqual(ratios[0]['C_over_S'],F(3,2))
        self.assertEqual(ratios[0]['D_over_S'],F(1,2));self.assertIsNone(ratios[0]['CI'])
        row['role']='REPLAY'
        with self.assertRaises(ValueError):m.pilot_ratios([row])


if __name__=='__main__':unittest.main(verbosity=2)
