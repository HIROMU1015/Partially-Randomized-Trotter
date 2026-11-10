"""Artificial-only tests for integer precision. No molecular signals."""
from copy import deepcopy
from decimal import Decimal,localcontext
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'scripts')]
import verify_h4_df10_rpe_shot_integers as v


class IntegerTests(unittest.TestCase):
    def test_large_artificial_N_is_upward_sufficient_and_never_reduced(self):
        changed=0
        for epsilon in (.1,1e-9,2e-30,1e-38):
            alpha=.002;raw=math.ceil(2/(epsilon*epsilon)*math.log(2/alpha));check=v.integer_check(epsilon,alpha,raw)
            final=int(check['N_handoff_decimal']);self.assertGreaterEqual(final,raw);changed+=check['changed']
            with localcontext() as c:
                c.prec=180;e=Decimal.from_float(epsilon);a=Decimal.from_float(alpha)
                self.assertGreaterEqual(Decimal(final)*e*e/2,(Decimal(2)/a).ln())
        self.assertGreater(changed,0)

    def test_source_formula_drift_and_invalid_inputs_are_not_corrected(self):
        raw=math.ceil(2/.01*math.log(1000))
        with self.assertRaises(ValueError):v.integer_check(.1,.002,raw+100)
        for e,a,n in [(0.,.01,3),(float('nan'),.01,3),(.1,1.,3),(.1,.01,3.)]:
            with self.assertRaises(ValueError):v.integer_check(e,a,n)

    def test_saved_m3_is_copied_without_any_N_arithmetic(self):
        n=dict(identity=dict(m=3,candidate=dict(id='ARTIFICIAL')),plan=dict(status='CONDITIONAL_ROUND_PASS',N_X=11,N_Y=12),actual=dict(N_X=None,N_Y=None),diagnostics=dict(normalization={}),source=dict(path='MOCK'))
        with patch.object(v,'integer_check',side_effect=AssertionError('No m3 reanalysis')):
            out=v.checked_record(n,{},'MOCK')
        self.assertEqual(out['plan'],n['plan']);self.assertEqual(out['actual'],n['actual']);self.assertEqual(out['source'],n['source'])

    def test_failed_or_inconsistent_axis_is_not_promoted(self):
        n=dict(identity=dict(m=0),plan=dict(status='FAIL'),diagnostics={})
        with self.assertRaises(ValueError):v.checked_record(n,{},'MOCK')
        n['plan']=dict(status='CONDITIONAL_ROUND_PASS',N_X=10,axes=dict(X=dict(shots=11)))
        with self.assertRaises(ValueError):v.checked_record(n,{},'MOCK')


if __name__=='__main__':unittest.main()
