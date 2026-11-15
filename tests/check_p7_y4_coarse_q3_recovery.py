"""Actual PF worker + untouched AST guard, synthetic DP and persistence only."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_p7_y4_coarse_q3_recovery as R
F=R.O.q3_runtime()
sys.path.insert(0,str(R.O.RUNTIME/'tests'))
from check_pf_shortlist_q3_caller import artificial,MOCK
from check_rpe_total_dp_parallel import ParallelDPTests


class RecoveryWorkerTests(unittest.TestCase):
    def test_real_runtime_constructor_and_original_saved_metadata(self):
        env=R.runtime_check(F);inputs=R.fixed_inputs(F)
        self.assertEqual(env['actual_AST_hash'],env['expected_AST_hash'])
        self.assertFalse(env['guard_modified']);self.assertEqual(len(inputs['cases']),1)

    def test_actual_PF_process_worker_constructor_evaluation_and_persistence(self):
        case,inputs,binding,view=artificial();before=deepcopy((case,inputs,view))
        inputs['concurrent_configurations']=1
        with tempfile.TemporaryDirectory() as folder,patch.object(F,'metadata',return_value=(binding,view)):
            output=Path(folder);(output/case['label']).mkdir()
            # Same M.supervise -> F.worker -> original PF worker -> Observed
            # constructor -> T.evaluate -> API/partials/resource persistence.
            # Only intake metadata is replaced by invented toy signals.
            with F.installed_workers(output):result=F.M.supervise([case],inputs,MOCK,output,poll_seconds=.05)
            self.assertFalse(result['errors']);self.assertFalse(result['monitor_errors'])
            self.assertEqual(len(result['receipts']),1)
            receipt=result['receipts'][0];self.assertEqual(receipt['scientific_evaluation_calls'],1)
            saved=output/case['label'];r=json.loads((saved/'RESULT.json').read_text())
            self.assertEqual(json.loads((saved/'API_RETURN.json').read_text()),[r['record']])
            for name in ['PARTIAL_COARSE.json','PARTIAL_LEMMA.json','PARTIAL_TOTAL_DP.json',
                         'RESOURCE_FINAL.json','PERSISTED.json','SUPERVISOR_RECEIPT.json']:
                self.assertTrue((saved/name).exists(),name)
            exits=[c for p in output.glob('PROCESS_EXITS_*.jsonl') for line in p.read_text().splitlines()
                   for c in json.loads(line)['children']]
            self.assertTrue(exits);self.assertTrue(all(c['exitcode']==0 for c in exits))
            self.assertEqual(r['status'],'MODEL_BOUND_MEETS_ADOPTED_THRESHOLD')
        self.assertEqual((case,{k:v for k,v in inputs.items() if k!='concurrent_configurations'},view),before)


if __name__=='__main__':
    # Reuse three original exact-DP tests: all synthetic laws, task partition,
    # and scope/residual boundaries. No molecular or saved scientific means.
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(RecoveryWorkerTests)
    for name in ['test_full_original_DP_output_exactly_matches_all_artificial_cases',
                 'test_process_counts_and_task_partition_leave_outputs_identical',
                 'test_scope_boundary_uses_exact_same_state_prefix_and_residual']:
        suite.addTest(ParallelDPTests(name))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
