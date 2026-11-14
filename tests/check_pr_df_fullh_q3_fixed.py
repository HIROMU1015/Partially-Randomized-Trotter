"""Artificial persistence/monitor tests and saved metadata; no real H4 risk."""
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
from check_pf_shortlist_q3_caller import artificial,MOCK
spec=importlib.util.spec_from_file_location('_df_fullh_caller',ROOT/'scripts/run_pr_df_fullh_total_q3_fixed.py')
M=importlib.util.module_from_spec(spec);sys.modules[spec.name]=M;spec.loader.exec_module(M)


class FixedCallerTests(unittest.TestCase):
    def test_saved_routes_and_exact_count_scope_without_risk(self):
        inputs=json.loads((ROOT/M.INPUT).read_text())
        self.assertEqual([c['label']for c in inputs['cases']],M.LABELS)
        with patch.object(M.T,'evaluate',side_effect=AssertionError('No real risk')):
            for case in inputs['cases']:
                binding,view=M.metadata(case)
                self.assertEqual(binding['total_configuration']['id'],case['total_configuration_id'])
                self.assertEqual(len(view['signal_packet']['means']),13)
                self.assertEqual([r['N_X']for r in case['allocation']['shots']],
                                 [64,60,56,52,47,43,38,34,30,25,21,17,12])
        self.assertLessEqual(inputs['finite_scope_preflight']['transition_pairs_upper'],141559036)
        self.assertEqual(inputs['finite_scope_preflight']['first_round_full_support_states'],2593)
        self.assertEqual(inputs['statistical_contract']['simulation_policy']['target_law'],'ideal_finite_RTE_law')

    def test_worker_persists_untouched_API_partials_resources_before_return(self):
        case,inputs,binding,view=artificial();before=deepcopy((case,inputs,view))
        with tempfile.TemporaryDirectory()as folder,patch.object(M,'metadata',return_value=(binding,view)),\
                patch.object(M,'check_source'):
            receipt=M.worker(case,inputs,MOCK,folder);path=Path(folder)
            result=json.loads((path/'RESULT.json').read_text());api=json.loads((path/'API_RETURN.json').read_text())
            self.assertEqual(api,[result['record']])
            self.assertTrue(receipt['api_return_saved']);self.assertTrue(receipt['resource_saved'])
            for file in ['PARTIAL_COARSE.json','PARTIAL_LEMMA.json','PARTIAL_TOTAL_DP.json','RESOURCE_FINAL.json','PERSISTED.json']:
                self.assertTrue((path/file).exists(),file)
            usage=json.loads((path/'RESOURCE_FINAL.json').read_text())
            self.assertIsNotNone(usage['process_tree_CPU_seconds']);self.assertGreater(usage['worker_peak_RSS_KiB'],0)
            self.assertEqual(result['scientific_evaluation_calls'],1)
            self.assertIsNone(result['sufficient_N']);self.assertIsNone(result['G'])
        self.assertEqual((case,inputs,view),before)

    def test_parent_monitor_failure_keeps_started_worker_API_and_metrics(self):
        case,inputs,binding,view=artificial();inputs['concurrent_configurations']=1
        context=M.mp.get_context('fork');ready=context.Event();release=context.Event()
        original=M.T.evaluate
        def blocked_evaluate(*args,**kwargs):
            ready.set()
            if not release.wait(10):raise AssertionError('Artificial parent did not release worker')
            return original(*args,**kwargs)
        def monitor_error(*args):
            if not ready.wait(10):raise AssertionError('Artificial worker did not start')
            release.set();raise RuntimeError('ARTIFICIAL_PARENT_MONITOR_FAILURE')
        with tempfile.TemporaryDirectory()as folder,patch.object(M,'metadata',return_value=(binding,view)),\
                patch.object(M,'check_source'),patch.object(M.T,'evaluate',side_effect=blocked_evaluate):
            path=Path(folder);(path/case['label']).mkdir()
            result=M.supervise([case],inputs,MOCK,path,monitor=monitor_error,poll_seconds=.01)
            self.assertEqual(len(result['monitor_errors']),1)
            self.assertFalse(result['errors'])
            receipt=result['receipts'][0]
            self.assertEqual(receipt['status'],'MODEL_BOUND_MEETS_ADOPTED_THRESHOLD')
            self.assertEqual(receipt['scientific_evaluation_calls'],1)
            for file in ['API_RETURN.json','RESULT.json','PARTIAL_TOTAL_DP.json','RESOURCE_FINAL.json','PERSISTED.json']:
                self.assertTrue((path/case['label']/file).exists(),file)
            self.assertTrue((path/'MONITOR_ERROR.json').exists())

    def test_failure_flag_blocks_not_started_call_but_saves_status_and_usage(self):
        case,inputs,binding,view=artificial();flag=M.mp.get_context('fork').Event();flag.set()
        with tempfile.TemporaryDirectory()as folder,patch.object(M,'metadata',return_value=(binding,view)),\
                patch.object(M,'check_source'),patch.object(M,'FAILURE',flag),patch.object(M.P,'FAILURE',flag),\
                patch.object(M.T,'evaluate',side_effect=AssertionError('Must not call'))as call:
            receipt=M.worker(case,inputs,MOCK,folder)
            self.assertEqual(receipt['status'],'UNKNOWN');self.assertEqual(receipt['scientific_evaluation_calls'],0)
            self.assertEqual(call.call_count,0);self.assertFalse(receipt['api_return_saved'])
            self.assertTrue((Path(folder)/'RESOURCE_FINAL.json').exists())

    def test_foreign_full_h_configuration_or_R448_ID_rejected_before_risk(self):
        case=deepcopy(json.loads((ROOT/M.INPUT).read_text())['cases'][-1])
        case['original_configuration_id']='cfa6aee3260de664de9acd444a628faae388c459a2eaf68a786c707573029d1c'
        with patch.object(M.T,'evaluate',side_effect=AssertionError('No real risk'))as call:
            with self.assertRaisesRegex(ValueError,'Original ID'):M.metadata(case)
        self.assertEqual(call.call_count,0)


if __name__=='__main__':unittest.main(verbosity=2)
