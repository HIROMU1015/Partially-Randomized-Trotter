"""Artificial caller/threshold checks; never real H4 risk."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from check_rpe_total_binding import pauli,MOCK,T
spec=importlib.util.spec_from_file_location('_scale_three_quarter',ROOT/'scripts/run_pauli_q3_three_quarter.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)


class ThreeQuarterTests(unittest.TestCase):
    def test_exact_vector_and_finite_support_bounds(self):
        original=[64,60,56,52,47,43,38,34,30,25,21,17,12]
        vector=[max(1,(3*n+3)//4)for n in original]
        self.assertEqual(vector,[48,45,42,39,36,33,29,26,23,19,16,13,9])
        b=M.finite_bounds(vector)
        self.assertLessEqual(b['maximum_round_outcomes'],4225)
        self.assertLessEqual(b['verification_pairs_upper'],55043173)
        self.assertLessEqual(b['transition_pairs_upper'],141559036)

    def test_model_classifications_are_not_physical_qualification(self):
        contract=dict(MSE_upper=1,epsilon_E_decimal='1')
        for lo,hi,status in [(0,1,'MODEL_BOUND_MEETS_ADOPTED_THRESHOLD'),
            (2,3,'MODEL_BOUND_EXCEEDS_ADOPTED_THRESHOLD'),(.5,2,'INCONCLUSIVE')]:
            out=M.assess(dict(MSE_lower=lo,MSE_upper=hi),contract,MOCK)
            self.assertEqual(out['status'],status)
            self.assertFalse(out['q3']['physical_total_error_guarantee'])
            self.assertIsNone(out['sufficient_N']);self.assertIsNone(out['G_qualified'])
        self.assertEqual(M.assess(dict(MSE_lower=None),contract,MOCK)['status'],'UNKNOWN')
        with self.assertRaises(ValueError):M.assess(dict(MSE_lower=2,MSE_upper=1),contract,MOCK)

    def test_reused_accelerated_worker_on_artificial_means_only(self):
        args,_=pauli();bridge=T.create(ROOT,args['config'],original_source=MOCK)
        case=dict(label='LD1',original_source=MOCK,binding_id=bridge['id'],allocation=args['allocations'][0],
            evaluation_target=args['evaluation_target'])
        inputs=dict(statistical_contract=args['statistical_contract'],options=args['options'],
            acceleration=dict(workers_per_configuration=2,outcome_batch=1,source=MOCK),
            adopted_evaluation=dict(MSE_upper=999,epsilon_E_decimal='ARTIFICIAL_ONLY'))
        with tempfile.TemporaryDirectory()as directory,patch.object(M.R.L.A,'accept_saved',return_value=dict(
            results=[dict(configuration=args['config'],signal_packet=args['signal_packet'])])):
            result=M.worker(case,inputs,MOCK,directory)
            self.assertEqual(result['status'],'MODEL_BOUND_MEETS_ADOPTED_THRESHOLD')
            self.assertEqual(result['scientific_evaluation_calls'],1)
            self.assertTrue(result['target_precision_filter_performed'])
            self.assertFalse(result['prior_control_recomputed'])
            self.assertEqual(result['price_calls'],0)
            self.assertIsNone(result['sufficient_N']);self.assertIsNone(result['G'])


if __name__=='__main__':unittest.main(verbosity=2)
