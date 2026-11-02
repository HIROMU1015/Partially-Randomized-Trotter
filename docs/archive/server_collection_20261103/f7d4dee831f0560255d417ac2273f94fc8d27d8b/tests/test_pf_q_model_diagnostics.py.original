"""Synthetic arithmetic, scope and descriptive-summary tests only."""
from fractions import Fraction
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/trotterlib'))
sys.path.insert(0,str(ROOT/'scripts'))
import pf_q_model_diagnostics as d
from h8_phase_b2_contract import axis_check


class DiagnosticsTests(unittest.TestCase):
    def test_sign_and_measured_denominator(self):
        r=d.axis_errors('110',100)
        self.assertEqual(r['difference_RZ_exact'],'10')
        self.assertEqual(r['signed_relative_error_exact'],'1/10')
        self.assertEqual(r['absolute_relative_error_percent'],10)
        r=d.axis_errors('90',100)
        self.assertEqual(r['signed_relative_error_exact'],'-1/10')
        self.assertEqual(r['absolute_error_RZ_exact'],'10')

    def test_strict_threshold_exactly_two_percent_fails(self):
        for value,passed in [('102',False),('98',False),('10199/100',True),('9801/100',True),('100',True)]:
            r=d.axis_errors(value,100)
            self.assertEqual(r['strictly_below_2_percent'],passed)
            self.assertEqual(axis_check(value,100)['PASS'],passed)
            self.assertEqual(axis_check(value,100)['exact_relative_error'],r['absolute_relative_error_exact'])

    def test_rational_prediction_is_not_rounded(self):
        r=d.axis_errors('301/3',100)
        self.assertEqual(r['difference_RZ_exact'],'1/3')
        self.assertEqual(r['absolute_relative_error_exact'],'1/300')

    def test_nonpositive_or_missing_is_rejected_without_zero_fill(self):
        for value in (0,-1,None,True,1.5):
            with self.assertRaises((ValueError,TypeError)):d.axis_errors('100',value)
        for value in ('0','-1','invalid',None):
            with self.assertRaises((ValueError,TypeError)):d.axis_errors(value,100)

    def test_quantile_uses_declared_type7(self):
        self.assertEqual(d.quantile([0,2,4,6],Fraction(1,2)),3)
        self.assertEqual(d.quantile([0,2,4,6],Fraction(19,20)),Fraction(57,10))
        self.assertEqual(d.quantile([5],Fraction(19,20)),5)
        with self.assertRaises(ValueError):d.quantile([],Fraction(1,2))

    def test_training_role_takes_precedence_and_is_not_independent(self):
        self.assertEqual(d.point_class(['TRAIN_ANCHOR','HOLDOUT']),'TRAINING_USED')
        self.assertEqual(d.point_class(['PRIOR_DIRECT_VALIDATION']),'VALIDATION_USED')
        self.assertEqual(d.point_class([]),'OTHER_COMPARISON_POINT')

    def test_unweighted_summary_keeps_axis_counts(self):
        rows=[dict(task_id=str(i),original_global_final_gate_missing=i==0,**d.axis_errors(v,100))
              for i,v in enumerate(('101','103'))]
        r=d.summarize(rows)
        self.assertEqual(r['n_axes'],2)
        self.assertEqual(r['mean_absolute_relative_error_percent_exact'],'2')
        self.assertEqual(r['below_2_percent'],1)
        self.assertEqual(r['at_or_above_2_percent'],1)
        self.assertEqual(r['original_final_gate_missing_axes'],1)
        self.assertEqual(r['max_error_task_ids'],['1'])
        with self.assertRaises(ValueError):d.summarize([])

    def fixtures(self):
        rows=[];pred=[]
        for i in range(450):
            rows.append(dict(task_id=str(i),comparison_class='CONDITIONAL_COMPARISON',
                saved_prediction_literals=['101'],direct_RZ_literal='100',prior_model_point_roles=[],residuals_computed=False))
            pred.append(dict(task_id=str(i),saved_prediction_literal='101'))
        return rows,pred

    def test_scope_refuses_missing_and_duplicate_axes(self):
        rows,pred=self.fixtures()
        with self.assertRaises(ValueError):d.diagnose(rows[:-1],pred)
        rows[-1]['task_id']='0'
        with self.assertRaises(ValueError):d.diagnose(rows,pred)

    def test_prediction_ambiguity_and_missing_are_rejected(self):
        rows,pred=self.fixtures();rows[0]['saved_prediction_literals']=[]
        with self.assertRaises(ValueError):d.diagnose(rows,pred)
        rows,pred=self.fixtures();pred[0]['saved_prediction_literal']='102'
        with self.assertRaises(ValueError):d.diagnose(rows,pred)

    def test_excluded_rows_are_not_evaluated(self):
        rows,pred=self.fixtures()
        rows.append(dict(task_id='excluded',comparison_class='NOT_COMPARABLE',direct_RZ_literal='invalid'))
        result=d.diagnose(rows,pred)
        self.assertEqual(len(result),450)
        self.assertNotIn('excluded',{r['task_id'] for r in result})


if __name__=='__main__':unittest.main()
