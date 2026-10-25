"""Full DF-PF metadata/IO tests; no numerical imports or real fixture builds."""
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'),str(Path(__file__).resolve().parent)]
from test_b2d_evaluator import Barrier
from test_b2d_b0_native import Circuit,Opaque

guard = Barrier();sys.meta_path.insert(0,guard)
try:
    from trottertracks.b2d_evaluator import full_df as f
    from trottertracks.b2d_evaluator.followup_io import run_prepared,FollowupLedger
finally:sys.meta_path.remove(guard)


class MockNative(f.Native):
    def __init__(self,run,ledger,fail=None):
        self.run,self.ledger,self.fail = run,ledger,fail
        self.cache = {};self.compile_cache = {}
        self.input_diagnostics = {};self.full_diagnostics = {}
        for k in ('fixture','split','prepare','full_inputs','input_gate','full_prepare','definition_gate'):
            self.call(k,'input',lambda:None)
    def call(self,kind,key,fn):
        def wrapped():
            if kind==self.fail:raise f.Stage1Failure('MOCK_FAILURE')
            return fn()
        return super().call(kind,key,wrapped)
    def exp(self,label,time):
        return self.cached('exponential',f.b0.exp_key(label,time),lambda:Opaque())
    def block_probe(self,index,time,controlled):
        return self.cached('block_build',f.b0.block_key(f.LABELS[index],time,controlled),lambda:Circuit(3 if controlled else 2))
    def system(self,row):return Circuit(2)
    def controlled_system(self,row):return Circuit(3)
    def full_reference(self,row):return Opaque()
    def controlled(self,u):return Opaque()
    def circuit_gate(self,*a,**k):return 0.
    def wrapper(self,*a):return Circuit(3)
    def wrapper_reference(self,*a):return Opaque()
    def expectation(self,*a):return dict(mock=True)
    def compile(self,qc):return qc
    def measure(self,qc):return dict(mock=True)
    def signals(self,*a):return dict(mock=True,normalization=1)
    def saved_comparisons(self):return dict(mock=True)
    def provenance(self):return dict(mock=True)


class FullDFTests(unittest.TestCase):
    def setUp(self):
        guard = Barrier();sys.meta_path.insert(0,guard)
        self.addCleanup(sys.meta_path.remove,guard)

    def test_fixed_cases_all_blocks_signed_stage_order_and_no_split_parameters(self):
        rows = f.cases()
        self.assertEqual([r['case_id'] for r in rows],['FD1','FD2','FD3','FD4'])
        for row in rows:
            stages = f.schedule(row)
            count = 16 if row['formula']=='2nd' else 48
            self.assertEqual(len(stages),count)
            self.assertEqual([s.term for s in stages[:8]],[0,1,2,3,3,2,1,0])
            self.assertEqual({s.term for s in stages},{0,1,2,3})
            self.assertNotIn('n_d',row);self.assertNotIn('rte_parameters',row)
            self.assertEqual(row['control_policy'],'ordinary')
            if row['formula']=='4th':self.assertTrue(any(s.time*float(row['total_time'])<0 for s in stages))
        scope = f.scope()
        self.assertEqual((scope['wrappers'],scope['compile_requests'],scope['cost_requests']),(8,8,8))
        self.assertEqual(scope['scalar_policy'],'original_c0_once')
        self.assertFalse(any(n.startswith(('rte.','exact_r.','b0.')) for n in scope['boundaries']))

    def test_out_of_scope_candidate_q_T_control_and_extra_split_rejected(self):
        for key,value in (('case_id','FD5'),('q',4),('total_time',.25),('formula','8th'),
                          ('control_policy','symmetric_directional'),('method','D_R_split'),('n_d',1)):
            with self.subTest(key=key),self.assertRaises(f.Stage1Failure):f.schedule({**f.cases()[0],key:value})

    def mock_run(self,d,fail=None):
        run = dict(root=ROOT,output=Path(d)/'fresh',scope=f.scope(),source='MOCK',inputs={})
        status = run_prepared(run,lambda r,l,e:f.execute(r,l,e,native_factory=lambda r,l:MockNative(r,l,fail)),mock_only=True)
        return run,status

    def test_full_opaque_exact_scope_and_all_gates_before_comparisons_costs(self):
        with tempfile.TemporaryDirectory() as d:
            run,status = self.mock_run(d)
            self.assertEqual(status['status'],'COMPLETED_DESCRIPTIVE_MANDATORY_STOP',status)
            self.assertEqual(status['calls']['completed'],run['scope']['boundaries'])
            result = json.loads((run['output']/'result.json').read_text())
            self.assertEqual(len(result['rows']),4);self.assertIsNone(result['G']);self.assertIsNone(result['shots'])
            self.assertTrue(all(s['mode']=='ORDINARY' for r in result['rows'] for s in r['append_schedule']))
            events = [json.loads(x) for x in (run['output']/'events.jsonl').read_text().splitlines()]
            end = max(i for i,x in enumerate(events) if x['event']=='complete' and
                      (x['boundary'].endswith('_gate') or x['boundary'].endswith('_expectation') or x['boundary']=='full_df.saved_comparisons'))
            start = min(i for i,x in enumerate(events) if x['event']=='start' and x['boundary'] in ('full_df.signals','full_df.measure'))
            self.assertLess(end,start)
            self.assertEqual(run_prepared(run,lambda *a:self.fail('resume forbidden'),mock_only=True)['status'],
                             'FRESH_OUTPUT_FAILED_MANDATORY_STOP')

    def test_failure_partial_and_stop_no_metrics_or_automatic_retry(self):
        for kind in ('input_gate','definition_gate','block_gate','system_gate','controlled_gate',
                     'raw_wrapper_gate','compile','compiled_gate','saved_comparisons'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
                run,status = self.mock_run(d,kind)
                self.assertEqual(status['status'],'FAILED_MANDATORY_STOP')
                self.assertNotIn('full_df.measure',status['calls']['attempted'])
                self.assertNotIn('full_df.signals',status['calls']['attempted'])
                partial = json.loads((run['output']/'partial_results.json').read_text())
                self.assertTrue(any(x.get('error_type') for x in partial['rows']))
                self.assertTrue((run['output']/'STOP.json').exists())
        ledger = FollowupLedger(f.scope())
        with self.assertRaises(f.Stage1Failure):ledger.call('full_df.compile',lambda:None,identity='FD5:cosine')

    def test_readonly_preflight_source_fresh_output_and_versions_without_science(self):
        out = ROOT/f.OUTPUT_PARENT/'__mock_only_unused__'
        self.assertFalse(out.exists())
        with patch('trottertracks.b2d_evaluator.followup_io._check_source') as check, \
             patch('importlib.metadata.version',return_value='1.3.0'), \
             patch.object(f,'Native',side_effect=AssertionError('science forbidden')):
            run = f.preflight(ROOT,out,'MOCK')
            check.assert_called_once_with(ROOT.resolve(),'MOCK',output=out.resolve())
            self.assertEqual(run['scope'],f.scope());self.assertFalse(out.exists())
        with self.assertRaises(ValueError):f.preflight(ROOT,out,'0'*40)
        with patch.object(f,'base_preflight',return_value={'versions':{'qiskit':'wrong'}}):
            with self.assertRaises(ValueError):f.preflight(ROOT,out,'MOCK')

    def test_cli_metadata_and_no_execution_without_explicit_flag(self):
        command = [sys.executable,'-I','-B','-S',str(ROOT/'scripts/run_b2d_full_df_pf.py')]
        p = subprocess.run(command+['--inspect'],capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(p.stdout)['status'],'METADATA_ONLY_NO_SCIENCE')
        self.assertNotEqual(subprocess.run(command,capture_output=True).returncode,0)
        with patch.object(f,'preflight',return_value=dict(source='MOCK',versions={},scope={})), \
             patch.object(f,'run_prepared',side_effect=AssertionError('no science')), \
             patch.object(sys,'argv',['pilot','--preflight','--source-commit','MOCK','--output','MOCK']), \
             patch('sys.stdout',new=io.StringIO()):
            self.assertEqual(f.main(),0)


if __name__=='__main__':unittest.main()
