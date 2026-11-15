"""Saved-integer metadata, one-case dispatch, and actual toy PF persistence."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tests')]
import run_p7_y4_coarse_a_shape_q3 as A
import check_p7_y4_coarse_q3_recovery as Previous
R,O,F=A.R,A.O,Previous.F


class AShapeTests(unittest.TestCase):
    def test_saved_integers_signal_and_contract_only_change_the_case(self):
        inputs=A.fixed_inputs(F);case=inputs['cases'][0];before=deepcopy(case)
        base=O.read(A.BASE);old=O.read(A.INTEGER_SOURCE)
        self.assertEqual(case['allocation']['shots'],old['shots'])
        self.assertNotEqual(case['allocation']['allocation_id'],old['allocation_id'])
        self.assertEqual(case['signal_packet'],R.PACKET)
        self.assertEqual(case['configuration_id'],O.CID)
        for key in ('kernel_source','C_source','statistical_contract','options','adopted_evaluation','acceleration'):
            self.assertEqual(inputs[key],base[key])
        for mutate in (lambda c:c.update(configuration_id=O.CURRENT),
                       lambda c:c['allocation']['shots'][3].update(N_Y=30),
                       lambda c:c['allocation'].update(allocation_id=old['allocation_id']),
                       lambda c:c.update(signal_packet=A.BASE),
                       lambda c:c['allocation']['shots'].reverse()):
            wrong=deepcopy(case);mutate(wrong)
            with self.assertRaises(ValueError):A.validate_case(wrong)
        self.assertEqual(case,before)

    def test_execute_dispatches_one_fixed_case_to_original_shared_supervisor(self):
        inputs={'cases':[{'label':'TOY'}],'maximum_scientific_calls':1};env={'toy':True}
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            (out/'INPUTS.json').write_text(json.dumps(dict(inputs=inputs,runtime=env)))
            (out/'DELTA_CHECK.json').write_text(json.dumps(dict(equivalent_results=[],matching_jobs=[],unresolved=[])))
            with patch.object(A,'OUT',out),patch.object(O,'q3_runtime',return_value=F),\
                 patch.object(O,'source',return_value=dict(commit='TOY',path='TOY')),\
                 patch.object(O,'blob',side_effect=lambda commit,path:(ROOT/path).read_bytes()),\
                 patch.object(A,'fixed_inputs',return_value=inputs),patch.object(R,'runtime_check',return_value=env),\
                 patch.object(R,'run_case',return_value=0) as run:
                self.assertEqual(A.execute(),0)
                args=run.call_args.args
                self.assertEqual(args[0],F);self.assertEqual(args[1],inputs)
                self.assertEqual(args[2]['caller'],A.CALLER)
                self.assertEqual(args[2]['integer_allocation_source'],A.INTEGER_SOURCE)
                self.assertEqual(args[3],out/'run01');self.assertEqual(args[4],A.PREVIOUS)
                self.assertEqual(run.call_count,1)

    def test_shared_one_case_worker_saves_toy_API_and_refuses_second_entry(self):
        case,inputs,binding,view=Previous.artificial()
        inputs.update(cases=[case],maximum_scientific_calls=1,concurrent_configurations=1)
        with tempfile.TemporaryDirectory() as folder,patch.object(F,'metadata',return_value=(binding,view)):
            out=Path(folder)/'run01'
            self.assertEqual(R.run_case(F,inputs,Previous.MOCK,out,Previous.MOCK),0)
            complete=json.loads((out/'COMPLETION.json').read_text())
            self.assertEqual(complete['status'],'ONE_FIXED_Q3_SAVED')
            self.assertEqual(complete['receipts'][0]['scientific_evaluation_calls'],1)
            saved=out/case['label'];record=json.loads((saved/'RESULT.json').read_text())
            self.assertEqual(json.loads((saved/'API_RETURN.json').read_text()),[record['record']])
            with self.assertRaises(FileExistsError):R.run_case(F,inputs,Previous.MOCK,out,Previous.MOCK)


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(AShapeTests)
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(Previous.RecoveryWorkerTests))
    for name in ['test_full_original_DP_output_exactly_matches_all_artificial_cases',
                 'test_process_counts_and_task_partition_leave_outputs_identical',
                 'test_scope_boundary_uses_exact_same_state_prefix_and_residual']:
        suite.addTest(Previous.ParallelDPTests(name))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
