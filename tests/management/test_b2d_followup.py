"""A0/B0 preparation tests: no real arrays, mpmath, circuits or compiler."""
from dataclasses import replace
from fractions import Fraction as F
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"src"),str(Path(__file__).resolve().parent)]
from test_b2d_evaluator import Barrier,FORBIDDEN
guard=Barrier();sys.meta_path.insert(0,guard)
try:
    from trottertracks.b2d_evaluator import a0_reference as a0
    from trottertracks.b2d_evaluator import followup_io as io_parts
    from trottertracks.b2d_evaluator.launcher import Evidence
finally:
    sys.meta_path.remove(guard)


class FollowupTests(unittest.TestCase):
    def setUp(self):
        guard=Barrier();sys.meta_path.insert(0,guard)
        self.addCleanup(sys.meta_path.remove,guard)

    def test_a0_fixed_cases_precisions_five_targets_and_scope(self):
        nodes,rows=a0.plan();s=a0.scope()
        self.assertEqual(len(rows),16)
        self.assertEqual({r['candidate_id'] for r in rows},set(a0.IDS))
        self.assertEqual({r['digits'] for r in rows},{90,128})
        self.assertEqual({r['case']['total_time'] for r in rows},{F(9,50),F(-9,50)})
        self.assertEqual(s['boundaries']['a0.target'],80)
        self.assertEqual(s['boundaries']['a0.signal'],80)
        for i,n in enumerate(nodes):self.assertTrue(all(d<i for d in n.dependencies))
        self.assertEqual(s['expm_methods'],{'90':'taylor','128':'pade'})

    def test_exact_binary_lift_uses_ratio_not_short_decimal(self):
        self.assertEqual(a0.binary_fraction(.1),F(*(.1).as_integer_ratio()))
        self.assertNotEqual(a0.binary_fraction(.1),F(1,10))
        for bad in (float('nan'),float('inf')):
            with self.assertRaises(ValueError):a0.binary_fraction(bad)

    def test_a0_polynomial_and_signed_exponential_metadata(self):
        nodes,_=a0.plan()
        self.assertEqual({n.parameters[3] for n in nodes if n.kind=='polynomial_term'},{2,4})
        self.assertTrue(any(n.parameters[2]<0 for n in nodes if n.kind=='exponential'))
        self.assertEqual({n.parameters[-1] for n in nodes if n.kind=='polynomial_term' and n.parameters[3]==4},{1,2,3,4,5})
        self.assertEqual({n.parameters[1] for n in nodes if n.kind=='exponential'}, {'physical','D','R','D_0','D_1','D_2'})

    def test_snapshot_metadata_checks_bytes_without_scientific_imports(self):
        metadata,blobs=io_parts.snapshot_metadata(ROOT)
        self.assertEqual(set(blobs),set(io_parts.SNAPSHOT_FILES))
        self.assertEqual(metadata['sector_basis_indices'],[1,2])
        self.assertFalse(FORBIDDEN.intersection(n.split('.')[0] for n in sys.modules))
        with patch.object(io_parts.subprocess,'check_output',side_effect=['a','b']):
            with self.assertRaises(io_parts.Stage1Failure):io_parts.snapshot_metadata(ROOT)

    def test_existing_recorder_rejects_unregistered_and_duplicate_calls(self):
        ledger=io_parts.FollowupLedger(a0.scope())
        with self.assertRaises(io_parts.Stage1Failure):ledger.call('unknown',lambda:self.fail('not reached'),identity='x')
        ledger.call('a0.snapshot',lambda:'OPAQUE',identity='input')
        with self.assertRaises(io_parts.Stage1Failure):ledger.call('a0.snapshot',lambda:self.fail('retry'),identity='input')
        with self.assertRaises(io_parts.Stage1Failure):ledger.finish()

    def test_shared_io_mock_complete_failure_stop_and_collision(self):
        s=io_parts.registered_scope('test',[('test.call','once')])
        def body(run,ledger,evidence):
            ledger.call('test.call',lambda:'OPAQUE',identity='once')
            ledger.rows.append(dict(mock=True,u_num=None,certified=False))
            return dict(rows=ledger.rows)
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'fresh'
            run=dict(output=output,source='MOCK',scope=s,inputs={})
            status=io_parts.run_prepared(run,body,mock_only=True)
            self.assertEqual(status['status'],'COMPLETED_DESCRIPTIVE_MANDATORY_STOP')
            stop=(output/'STOP.json').read_bytes()
            self.assertEqual(io_parts.run_prepared(run,body,mock_only=True)['status'],'FRESH_OUTPUT_FAILED_MANDATORY_STOP')
            self.assertEqual((output/'STOP.json').read_bytes(),stop)
            self.assertFalse(json.loads(stop)['resume_allowed'])
        def fail(run,ledger,evidence):
            ledger.rows.append(dict(partial=True));raise io_parts.Stage1Failure('MOCK_GATE_FAILURE')
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'fail';run=dict(output=output,source='MOCK',scope=s,inputs={})
            status=io_parts.run_prepared(run,fail,mock_only=True)
            self.assertEqual(status['status'],'FAILED_MANDATORY_STOP')
            self.assertEqual(json.loads((output/'partial_results.json').read_text())['rows'],[dict(partial=True)])
            self.assertTrue((output/'STOP.json').exists())

    def test_a0_all_nodes_dispatch_with_opaque_values(self):
        nodes,rows=a0.plan();ledger=io_parts.FollowupLedger(a0.scope())
        ledger.call('a0.snapshot',lambda:'OPAQUE',identity='input')
        ledger.call('a0.initialize',lambda:'OPAQUE_REFERENCE',identity='reference')
        values=[]
        for i,n in enumerate(nodes):values.append(ledger.call('a0.'+n.kind,lambda:'OPAQUE',identity=str(i)))
        for row in rows:
            if row['digits']==128:ledger.call('a0.compare',lambda:dict(u_num=None,certified=False),identity=row['key'])
        ledger.finish()
        self.assertEqual(ledger.completed,ledger.scope['boundaries'])

    def test_a0_cli_missing_execute_and_inspect_are_inert(self):
        for args,expected in (([],2),(['--execute'],2),(['--inspect'],0)):
            with patch.object(sys,'argv',['run_b2d_outer_a0.py',*args]),patch('sys.stdout',new_callable=io.StringIO),patch('sys.stderr'),patch.object(a0,'execute') as work:
                with self.assertRaises(SystemExit) as status:
                    runpy.run_path(str(ROOT/'scripts/run_b2d_outer_a0.py'),run_name='__main__')
                self.assertEqual(status.exception.code,expected);work.assert_not_called()

    def test_a0_execute_full_path_uses_only_artificial_scalars_and_stubs(self):
        class Fake:
            def __init__(self,*args):
                self.contexts={d:NS(mpc=lambda re,im:complex(float(re),float(im)),mpf=float) for d in (90,128)}
                self.exact=lambda ctx,x:float(x)
            def perform(self,kind,args,params):
                return dict(value=0.,stages=[]) if kind=='bound_total' else 0.
            def encode(self,v,d):
                if isinstance(v,dict):return {k:self.encode(x,d) for k,x in v.items()}
                if isinstance(v,list):return []
                return dict(real=str(v.real),imag=str(v.imag))
        _,rows=a0.plan()
        old=[]
        for row in rows[:8]:
            outputs={name:0. for name in ('outer_operator_error','ideal_tail_physical_error','corrected_physical_error',
                     'raw_physical_error','D_approximation_error','finite_RTE_error')}
            outputs.update(signals={t:dict(X=0.,Y=0.) for t in a0.TARGETS},taylor_residual=dict(complex_signal_bound=0.))
            outputs['signals']['corrected_physical_signal_error']=0.
            old.append(dict(candidate_id=row['candidate_id'],case=dict(total_time=str(row['case']['total_time'])),outputs=outputs))
        with tempfile.TemporaryDirectory() as directory:
            run=dict(output=Path(directory)/'fresh',root=ROOT,source='MOCK',scope=a0.scope(),inputs={})
            def body(r,l,e):return a0.execute(r,l,e,reference_factory=Fake,snapshot_loader=lambda:'OPAQUE',previous_rows=old)
            status=io_parts.run_prepared(run,body,mock_only=True)
            self.assertEqual(status['status'],'COMPLETED_DESCRIPTIVE_MANDATORY_STOP',status)
            result=json.loads((run['output']/'result.json').read_text())
            self.assertEqual(len(result['rows']),16);self.assertEqual(len(result['comparisons']),8)
            self.assertIsNone(result['u_num']);self.assertFalse(result['certified'])

    def test_precision_methods_and_spectral_norm_use_explicit_stub_APIs(self):
        calls=[]
        class Label:
            def __mul__(self,other):return self
            __rmul__=__mul__
            def __neg__(self):return self
        ref=a0.Reference.__new__(a0.Reference)
        ref.exact=lambda ctx,value:Label()
        ref.data={d:dict(physical=Label()) for d in (90,128)}
        ref.contexts={d:NS(j=Label(),expm=lambda a,method:calls.append(method) or 'OPAQUE') for d in (90,128)}
        for d in (90,128):self.assertEqual(ref.perform('exponential',(),(d,'physical',F(9,50))),'OPAQUE')
        self.assertEqual(calls,['taylor','pade'])
        self.assertEqual(ref.norm(NS(svd=lambda matrix,compute_uv:[3,1]),'OPAQUE_MATRIX'),3)


if __name__=='__main__':unittest.main()
