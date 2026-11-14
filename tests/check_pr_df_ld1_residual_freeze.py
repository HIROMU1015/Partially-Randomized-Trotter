"""Artificial paired residual fits only. Real inputs are metadata-only elsewhere."""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch
import unittest

import freeze_pr_df_ld1_residual_models as F


def mock_inputs():
    paths=[]
    for m,(pid,q,R) in F.FOCUS.items():
        for axis in ('X','Y'):
            # Different axes; signed residuals/coefficients are legal quantities.
            theta=(Fraction(1,4),Fraction(2),Fraction(3)) if axis=='X' else (Fraction(-2),Fraction(-1),Fraction(0))
            mean=theta[0]+theta[1]*q+theta[2]*q*R
            # X has a quarter-integer mean, built from four integer observations.
            base=int(mean);deltas=([0,0,0,1] if axis=='X' else [0,0,0,0])
            for i,delta in enumerate(deltas):
                paths.append(dict(m=m,axis=axis,q=q,R=R,physical_id=pid,index=i,S=100000,C=100000+base+delta))
    saved=dict(quantity='FULL_WRAPPER_MEAN_STRUCTURAL_RZ_COUNT',expectation='IDEAL_FINITE_K_SOURCE_LAW',
        mean_S=1.25,numerical_enclosure=[1.24,1.26],law_binding={'source':'artificial_I'},kernel_source='mock',source_probability_factorization_relative_bound=0.)
    return dict(paths=paths,structural_m4={a:dict(source={'fixture':True},saved=deepcopy(saved),admission_identity={'fixture':True}) for a in ('X','Y')},
        holdout_metadata={'fixture':True},holdout_source={'fixture':True},holdout_rule=dict(n_paths=4,tolerance=None,model_verdict='NOT_ASSESSED_NO_ADOPTED_TOLERANCE'))


class Tests(unittest.TestCase):
    def test_exact_rank_coefficients_and_two_forecasts(self):
        d=F.calculate(mock_inputs(),'ARTIFICIAL_ONLY')
        self.assertEqual(d['models']['X']['coefficients'],dict(one=F.enc(Fraction(1,4)),q=F.enc(2),qR=F.enc(3)))
        self.assertEqual(d['models']['Y']['coefficients'],dict(one=F.enc(-2),q=F.enc(-1),qR=F.enc(0)))
        for model in d['models'].values():
            self.assertEqual(model['rank'],3);self.assertEqual(model['lack_of_fit_DOF'],0)
            self.assertTrue(all(v['numerator']==0 for v in model['calibration_arithmetic_residuals']))
        self.assertEqual(d['models']['X']['m4_prediction']['predicted_RZ'],F.enc(Fraction(6278,4)))
        self.assertEqual(d['models']['Y']['m4_prediction']['predicted_RZ'],F.enc(Fraction(-67,4)))

    def test_four_paths_required_no_missing_or_replay_pool(self):
        d=mock_inputs();d['paths'][0]['index']=1
        with self.assertRaisesRegex(ValueError,'missing/duplicate'):F.calculate(d,'mock')
        d=mock_inputs();d['paths'].pop()
        with self.assertRaisesRegex(ValueError,'24 calibration'):F.calculate(d,'mock')

    def test_other_physical_point_rejected(self):
        d=mock_inputs();d['paths'][0]['physical_id']='LD2'
        with self.assertRaisesRegex(ValueError,'scope'):F.calculate(d,'mock')

    def test_m4_receipt_read_fails_before_git(self):
        with patch.object(F.G.C,'blob',side_effect=AssertionError('No holdout read')):
            with self.assertRaisesRegex(ValueError,'M4_OBSERVATION_READ_FORBIDDEN'):
                F.raw_receipt(Path('.'),dict(commit='mock',path='results/m4/price.json'))

    def test_no_tolerance_no_model_pass_and_no_law_qualification(self):
        d=F.calculate(mock_inputs(),'mock')
        self.assertIsNone(d['comparison_rule']['tolerance']);self.assertIsNone(d['qualified_E_C'])
        for model in d['models'].values():
            p=model['m4_prediction'];self.assertFalse(model['qualified_for_candidate'])
            for key in ('observed_RZ','errors','law_distance','finite_cohort_uncertainty','CI'):self.assertIsNone(p[key])
            self.assertEqual(p['model_verdict'],'NOT_ASSESSED_NO_ADOPTED_TOLERANCE')


if __name__=='__main__':unittest.main()
