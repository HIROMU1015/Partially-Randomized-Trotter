"""Artificial accounting checks; no Hamiltonian/price producer is called."""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.price_extrapolation_analysis import analyze, break_even, number


def fixture():
    p=dict(point_id='toy',configuration_id='cfg',M=1,saved_total_RZ=100,
        allocation=[dict(m=0,N_X=1,N_Y=1),dict(m=1,N_X=2,N_Y=2)],axes=[])
    for m in range(2):
        for a in ('X','Y'):
            p['axes'].append(dict(m=m,axis=a,N=m+1,configuration_id='cfg',physical_id='toy'+str(m),
                source=dict(commit='ARTIFICIAL',path='toy.json'),unit_price_RZ=10 if m==0 else 20,
                tier='DIRECT' if m==0 else 'EXTRAPOLATION',components=[] if m==0 else
                [dict(id='shared',value=5,feature=4)]))
    return p


class Accounting(unittest.TestCase):
    def test_exact_total_tiers_rounds_and_correlated_parameter(self):
        r=analyze(fixture())
        self.assertEqual(number(r['total_RZ']),100)
        self.assertEqual(number(r['evidence_totals']['DIRECT']['cost_RZ']),20)
        self.assertEqual(number(r['evidence_totals']['EXTRAPOLATION']['cost_RZ']),80)
        self.assertEqual(number(r['shared_coefficients']['shared']['sensitivity']),16)
        self.assertEqual(r['rounds'][1]['d_total_per_same_shift_on_both_axes'],4)
        self.assertEqual(number(r['rounds'][1]['one_percent_both_axis_price_shift_RZ']),Fraction(4,5))
        self.assertIsNone(r['uncertainty_bound'])

    def test_missing_duplicate_and_missing_value_are_rejected(self):
        for change in ('missing','duplicate','null'):
            p=fixture()
            if change=='missing':p['axes'].pop()
            elif change=='duplicate':p['axes'][-1]=deepcopy(p['axes'][0])
            else:p['axes'][-1]['unit_price_RZ']=None
            with self.assertRaises(ValueError):analyze(p)

    def test_identity_source_allocation_and_saved_total_are_enforced(self):
        for key,value in [('configuration_id','wrong'),('source',None),('physical_id',None),('N',3)]:
            p=fixture();p['axes'][0][key]=value
            with self.assertRaises(ValueError):analyze(p)
        p=fixture();p['saved_total_RZ']=99
        with self.assertRaises(ValueError):analyze(p)

    def test_direct_price_and_differing_shared_values_are_not_silently_perturbed(self):
        p=fixture();p['axes'][0]['components']=[dict(id='shared',value=5,feature=2)]
        with self.assertRaisesRegex(ValueError,'direct'):analyze(p)
        p=fixture();p['axes'][-1]['components'][0].update(value=4,feature=5)
        with self.assertRaisesRegex(ValueError,'inconsistent'):analyze(p)

    def test_fixed_shot_break_even_keeps_direct_prices_fixed(self):
        lower=analyze(fixture());p=fixture();p['saved_total_RZ']=120
        for row in p['axes']:
            if row['m']==1:row['unit_price_RZ']=25;row['components'][0]['feature']=5
        r=break_even(lower,analyze(p))
        self.assertEqual(number(r['gap_RZ']),20)
        self.assertEqual(number(r['direct_prices_fixed']['lower_only_model_price_increase']),Fraction(1,4))
        self.assertEqual(number(r['direct_prices_fixed']['equal_adverse_relative_changes']),Fraction(1,9))
        self.assertIsNone(r['rank_probability'])

    def test_binary64_and_exact_rational_inputs_stay_distinct(self):
        self.assertNotEqual(number(.2),number('1/5'))
        self.assertEqual(number(dict(numerator=1,denominator=5)),Fraction(1,5))


if __name__=='__main__':unittest.main(verbosity=2)
