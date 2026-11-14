"""Artificial replacement-caller smoke tests; no real signal risk evaluated."""
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
from check_rpe_total_binding import pauli,MOCK,T
spec=importlib.util.spec_from_file_location('_accelerated_runner',ROOT/'scripts/run_pauli_total_q3_accelerated.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def test_artificial_parallel_worker_matches_original_risk(self):
        args,_=pauli();binding=T.create(ROOT,args['config'],original_source=MOCK)
        case=dict(label='LD1',original_source=MOCK,binding_id=binding['id'],allocation=args['allocations'][0],
            evaluation_target=args['evaluation_target'])
        inputs=dict(statistical_contract=args['statistical_contract'],options=args['options'],
            acceleration=dict(workers_per_configuration=2,outcome_batch=1,source=MOCK))
        expected=T.Q.evaluate(**dict(args,integration=T.C.interfaces(str(ROOT)).pipeline))[0]['diagnostic']
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(runner.L.A,'accept_saved',return_value=dict(results=[dict(
                 configuration=args['config'],signal_packet=args['signal_packet'])])):
            result=runner.worker(case,inputs,MOCK,directory)
            self.assertEqual(result['status'],'CONDITIONAL_OPT_IN_TOTAL_MODEL_BOUND_SAVED')
            self.assertEqual(result['record']['diagnostic'],expected)
            self.assertEqual(result['scientific_evaluation_calls'],1)
            self.assertEqual(result['old_partial_DP_calls'],0)
            self.assertTrue(any(r['workers']==2 for r in result['acceleration']['rounds']))
            self.assertIsNone(result['G']);self.assertIsNone(result['sufficient_N'])
            self.assertGreater(result['performance']['child_cpu_seconds'],0)

    def test_reference_scientific_conditions_are_identical(self):
        old=json.loads(runner.blob(runner.REFERENCE,runner.REFERENCE_INPUT))
        new=json.loads((ROOT/runner.INPUT).read_bytes())
        self.assertEqual(new['cases'],old['cases'])
        self.assertEqual(new['statistical_contract'],old['statistical_contract'])
        self.assertEqual({k:v for k,v in new['options'].items() if k!='total_dp_acceleration_source'},old['options'])
        self.assertEqual(new['acceleration']['workers_per_configuration'],8)


if __name__=='__main__':unittest.main(verbosity=2)
