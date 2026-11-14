"""Tiny artificial caller tests. Never reads or evaluates real Pauli means."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from check_rpe_total_binding import pauli, MOCK, T

spec=importlib.util.spec_from_file_location('_pauli_runner_test',ROOT/'scripts/run_pauli_total_q3_fixed.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def inputs(self):
        args,_=pauli();binding=T.create(ROOT,args['config'],original_source=MOCK)
        item=dict(configuration=args['config'],signal_packet=args['signal_packet'])
        case=dict(label='LD1',original_source=MOCK,binding_id=binding['id'],
            allocation=args['allocations'][0],evaluation_target=args['evaluation_target'])
        inputs=dict(statistical_contract=args['statistical_contract'],options=args['options'])
        return case,inputs,item

    def test_synthetic_one_call_preserves_nulls_and_partial_evidence(self):
        case,inputs,item=self.inputs()
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(runner.A,'accept_saved',return_value=dict(results=[item])), \
             patch.object(T,'evaluate',wraps=T.evaluate) as evaluate:
            result=runner.worker(case,inputs,MOCK,directory)
            self.assertEqual(result['status'],'CONDITIONAL_OPT_IN_TOTAL_MODEL_BOUND_SAVED')
            self.assertEqual(evaluate.call_count,1)
            self.assertEqual(result['scientific_evaluation_calls'],1)
            self.assertEqual(result['old_partial_DP_calls'],0)
            self.assertTrue((Path(directory)/'PARTIAL_TOTAL_DP_LD1.json').exists())
            for key in ('sufficient_N','global_RMSE_bound','G','G_qualified'):
                self.assertIsNone(result[key])

    def test_closure_failure_stops_before_total_DP_without_retry(self):
        case,inputs,item=self.inputs()
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(runner.A,'accept_saved',return_value=dict(results=[item])), \
             patch.object(T.Q.D,'verify_good_transitions',return_value=dict(status='UNKNOWN',reason='ARTIFICIAL_FAIL')), \
             patch.object(T.Q,'total_state_dp',side_effect=AssertionError('DP must not run')):
            result=runner.worker(case,inputs,MOCK,directory)
            self.assertEqual(result['status'],'UNKNOWN_STOP_NO_RETRY')
            self.assertIn('RHO_CLOSURE_UNKNOWN',result['reason'])
            self.assertEqual(result['scientific_evaluation_calls'],1)
            self.assertFalse((Path(directory)/'PARTIAL_TOTAL_DP_LD1.json').exists())

    def test_source_mismatch_stops_before_evaluation(self):
        case,inputs,item=self.inputs();case['binding_id']='OTHER_LD'
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(runner.A,'accept_saved',return_value=dict(results=[item])), \
             patch.object(T,'evaluate',side_effect=AssertionError('must not evaluate')):
            result=runner.worker(case,inputs,MOCK,directory)
            self.assertEqual(result['scientific_evaluation_calls'],0)
            self.assertEqual(result['status'],'UNKNOWN_STOP_NO_RETRY')


if __name__=='__main__':unittest.main(verbosity=2)
