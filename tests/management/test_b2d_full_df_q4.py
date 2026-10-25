"""Opaque q4 execution/IO tests, sharing the q2 mock and import barrier."""
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'),str(Path(__file__).resolve().parent)]
import test_b2d_full_df as old
from trottertracks.b2d_evaluator import full_df_q4 as f


class Q4Tests(unittest.TestCase):
    def setUp(self):
        guard = old.Barrier();sys.meta_path.insert(0,guard)
        self.addCleanup(sys.meta_path.remove,guard)

    def test_scope_derived_from_signed_four_block_q4_schedule_and_q2_preserved(self):
        scope = f.scope()
        self.assertEqual([r['case_id'] for r in f.cases()],['FD5','FD6'])
        for row in f.cases():
            stages = f.core.schedule(row,validator=f.validate_case)
            self.assertEqual(len(stages),96)
            self.assertEqual([s.term for s in stages[:8]],[0,1,2,3,3,2,1,0])
            self.assertTrue(any(s.time*float(row['total_time'])<0 for s in stages))
            for block in range(4):
                self.assertTrue(math.isclose(sum(s.time for s in stages if s.term==block),
                                             float(row['total_time']),abs_tol=1e-15))
        self.assertEqual((scope['block_probes'],scope['exponential_keys'],len(scope['calls'])),(32,22,138))
        self.assertEqual((scope['operator_gates'],scope['expectations'],scope['wrappers']),(44,8,4))
        saved = json.loads((ROOT/f.Q2_PATH/'binding.json').read_text())['scope']
        self.assertEqual(json.loads(f.core._encode(f.core.scope())),saved)
        for key,value in (('q',2),('q',8),('total_time',.25),('formula','2nd'),
                          ('case_id','FD3'),('control_policy','symmetric_directional'),('n_d',1)):
            with self.subTest(key=key,value=value),self.assertRaises(f.Stage1Failure):
                f.core.schedule({**f.cases()[0],key:value},validator=f.validate_case)
        with self.assertRaises(f.Stage1Failure):f.core.schedule(f.cases()[0])

    def mock_run(self,d,fail=None):
        run = dict(root=ROOT,output=Path(d)/'fresh',scope=f.scope(),source='MOCK',inputs={})
        status = old.run_prepared(run,lambda r,l,e:f.execute(r,l,e,
            native_factory=lambda r,l:old.MockNative(r,l,fail)),mock_only=True)
        return run,status

    def test_opaque_one_shot_exact_scope_correctness_first_and_fresh_output(self):
        with tempfile.TemporaryDirectory() as d:
            run,status = self.mock_run(d)
            self.assertEqual(status['status'],'COMPLETED_DESCRIPTIVE_MANDATORY_STOP',status)
            self.assertEqual(status['calls']['completed'],run['scope']['boundaries'])
            result = json.loads((run['output']/'result.json').read_text())
            self.assertEqual(len(result['rows']),2);self.assertEqual(result['compile_requests'],4)
            self.assertIsNone(result['G']);self.assertIsNone(result['gamma'])
            events = [json.loads(x) for x in (run['output']/'events.jsonl').read_text().splitlines()]
            last = max(i for i,x in enumerate(events) if x['event']=='complete' and
                (x['boundary'].endswith('_gate') or x['boundary'].endswith('_expectation') or
                 x['boundary']=='full_df.saved_comparisons'))
            first = min(i for i,x in enumerate(events) if x['event']=='start' and
                x['boundary'] in ('full_df.signals','full_df.measure'))
            self.assertLess(last,first)
            self.assertTrue((run['output']/'STOP.json').exists())
            self.assertEqual(old.run_prepared(run,lambda *a:self.fail('no resume'),mock_only=True)['status'],
                             'FRESH_OUTPUT_FAILED_MANDATORY_STOP')

    def test_failure_preserves_partial_stop_and_no_signal_cost_or_retry(self):
        for kind in ('input_gate','definition_gate','block_gate','system_gate','controlled_gate',
                     'raw_wrapper_gate','raw_expectation','compiled_gate','compiled_expectation','saved_comparisons'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
                run,status = self.mock_run(d,kind)
                self.assertEqual(status['status'],'FAILED_MANDATORY_STOP')
                self.assertNotIn('full_df.measure',status['calls']['attempted'])
                self.assertNotIn('full_df.signals',status['calls']['attempted'])
                self.assertTrue((run['output']/'partial_results.json').exists())
                self.assertTrue((run['output']/'STOP.json').exists())
        ledger = old.FollowupLedger(f.scope())
        with self.assertRaises(f.Stage1Failure):ledger.call('full_df.compile',lambda:None,identity='FD3:cosine')

    def test_readonly_source_snapshot_and_profile_preflight_no_science(self):
        out = ROOT/f.OUTPUT_PARENT/'__mock_only_unused__'
        with patch('trottertracks.b2d_evaluator.followup_io._check_source') as check, \
             patch('importlib.metadata.version',return_value='1.3.0'), \
             patch.object(f,'Native',side_effect=AssertionError('science forbidden')):
            run = f.preflight(ROOT,out,'MOCK')
            check.assert_called_once_with(ROOT.resolve(),'MOCK',output=out.resolve())
            self.assertEqual(run['scope'],f.scope());self.assertFalse(out.exists())
            self.assertEqual(len(run['saved_q2_b1_blobs']),6)
        with self.assertRaises(ValueError):f.preflight(ROOT,out,'0'*40)
        with patch.object(f.core,'preflight',return_value={'inputs':{}}), \
             patch.object(f.subprocess,'check_output',side_effect=['expected','different']):
            with self.assertRaises(f.Stage1Failure):f.preflight(ROOT,out,'MOCK')

    def test_observed_budget_only_with_unknown_nonfinite_and_outside_not_failure(self):
        for bias in (0.,1e-10,5e-10):
            self.assertEqual(f.observed_bias_classification(bias),'WITHIN_OBSERVED_BIAS_BUDGET')
        self.assertEqual(f.observed_bias_classification(5.1e-10),'OUTSIDE_OBSERVED_BIAS_BUDGET')
        for bias in (None,float('nan'),float('inf'),-1.):
            self.assertEqual(f.observed_bias_classification(bias),'INCONCLUSIVE_NUMERICAL_IDENTIFIABILITY')
        native = object.__new__(f.Native)
        with patch.object(f.core.Native,'signals',return_value={
                'physical_complex_coherent_signal_absolute_error':1e-8,'u_num':None,'gamma':None}):
            result = native.signals(f.cases()[0],None)
            self.assertEqual(result['observed_bias_classification'],'OUTSIDE_OBSERVED_BIAS_BUDGET')
            self.assertIsNone(result['u_num']);self.assertIsNone(result['gamma'])

    def test_saved_comparisons_quote_fixed_cells_without_recomputing_statistics(self):
        native = object.__new__(f.Native)
        native.run = {'root':ROOT}
        with patch.object(f.core.Native,'saved_comparisons',return_value={}):
            saved = native.saved_comparisons()
        self.assertEqual(len(saved['high_precision_pr']),6)
        self.assertEqual(len(saved['full_y4_q2']),2)
        self.assertEqual(len(saved['b1_statistics']),72)
        self.assertTrue(all(r['sample_count']=='256' for r in saved['b1_statistics']))

    def test_cli_requires_explicit_execution_flag_and_metadata_imports_only(self):
        command = [sys.executable,'-I','-B','-S',str(ROOT/'scripts/run_b2d_full_df_y4_q4.py')]
        p = subprocess.run(command+['--inspect'],capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(p.stdout)['scope']['new_cases'],2)
        self.assertNotEqual(subprocess.run(command,capture_output=True).returncode,0)


if __name__=='__main__':unittest.main()
