"""B0 opaque renderer/launcher tests. No Qiskit or real matrix operations."""
import ast
from fractions import Fraction as F
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(Path(__file__).resolve().parent)]
from test_b2d_evaluator import Barrier,FORBIDDEN
guard=Barrier();sys.meta_path.insert(0,guard)
try:
    from trottertracks.b2d_evaluator import b0_native as b0
    from trottertracks.b2d_evaluator.followup_io import FollowupLedger,run_prepared,Stage1Failure
    from trottertracks.b2d_evaluator.outer import DStage
finally:sys.meta_path.remove(guard)


class Opaque:
    def __matmul__(self,other):return self
    def __rmul__(self,other):return self
    def copy(self):return self


class Circuit:
    def __init__(self,n,*args):self.num_qubits=n;self.num_clbits=0;self.global_phase=0.;self.log=[]
    def compose(self,other,inplace=True):self.log.append(('compose',other));return self
    def p(self,phase,qubit):self.log.append(('p',phase,qubit))
    def x(self,q):self.log.append(('x',q))
    def h(self,q):self.log.append(('h',q))
    def sdg(self,q):self.log.append(('sdg',q))
    def measure(self,q,c):self.log.append(('measure',q,c))
    def remove_final_measurements(self,inplace=False):return self


class MockNative(b0.Native):
    def __init__(self,run,ledger,fail=None):
        self.run,self.ledger,self.fail=run,ledger,fail;self.cache={};self.compile_cache={}
        self.context=b0.candidate_graph('MOCK').settings.context
        self.scalar=0.;self.QC=Circuit;self.eye=Opaque();self.np=NS(exp=lambda x:1.)
        self.Operator=lambda c:NS(data=Opaque())
        for kind in ('fixture','split','prepare','full_inputs','input_gate'):
            self.call(kind,'input',lambda:'OPAQUE_INPUT')
        self.input_diagnostics={'mock':True}
    def call(self,kind,key,fn):
        def wrapped():
            if self.fail==kind:raise Stage1Failure('MOCK_GATE_FAILURE')
            return fn()
        return self.ledger.call('b0.'+kind,wrapped,identity=key)
    def config(self,spec):return self.cached('config',str(spec),lambda:'OPAQUE_CONFIG')
    def event(self,spec,descriptor):
        def event():self.config(spec);return 'OPAQUE_EVENT'
        return self.cached('event',b0.event_key(spec,descriptor),event)
    def event_matrix(self,spec,descriptor):return self.cached('event_operator',b0.event_key(spec,descriptor),lambda:Opaque())
    def exp(self,label,time):return self.cached('exponential',b0.exp_key(label,time),lambda:Opaque())
    def block(self,label,time,controlled):
        return self.cached('block_build',b0.block_key(label,time,controlled),lambda:Circuit(3 if controlled else 2))
    def rte(self,spec,word,controlled):
        return self.cached('rte_build',b0.rte_key(spec,word,controlled),lambda:Circuit(3 if controlled else 2))
    def controlled(self,u):return Opaque()
    def wrapper_reference(self,u,axis):return Opaque()
    def wrapper(self,c,axis):return Circuit(3)
    def compile(self,c):return c
    def expectation(self,c,r):return dict(mock=True)
    def check(self,a,b,code='MOCK'):return 0.
    def measure(self,c):return dict(mock=True,rz_count='OPAQUE_NO_PRICE')
    def provenance(self):return dict(mock=True,no_numeric_input_used=True)
    def single_mean(self,*args):return dict(mock=True)
    def analytic_mean(self,row):
        # Reuse actual reference traversal to exercise every registered exp key.
        plan=b0.outer_plan(row)
        for kind,w in b0.mathematical_stages(plan.case.method):
            time=plan.case.total_time*w/plan.case.step_count;self.exp(kind,time)
        self.exp('physical',plan.case.total_time)
        rs=[s for s in plan.body if not isinstance(s,DStage)]
        for time in dict.fromkeys(s.evolution_time for s in rs):
            spec=next(s.payload for s in rs if s.evolution_time==time)
            self.cached('moments',str(spec),lambda:'OPAQUE_MOMENTS')
        return dict(mock=True,gamma=None)


class B0Tests(unittest.TestCase):
    def setUp(self):
        guard=Barrier();sys.meta_path.insert(0,guard);self.addCleanup(sys.meta_path.remove,guard)

    def test_fixed_cases_and_laws(self):
        rows=b0.cases();self.assertEqual(len(rows),12)
        self.assertEqual({r['candidate_id'] for r in rows},set(b0.IDS))
        self.assertEqual({r['case']['total_time'] for r in rows},{F(9,50),F(-9,50)})
        for row in rows:
            plan=b0.outer_plan(row)
            for variant in range(4):
                words=b0.trajectory(row,variant,plan)
                self.assertEqual(len(words),row['cost']['R_occurrences'])
                self.assertTrue(all(len(word)==spec.rte_steps for spec,word in words))
                self.assertTrue(all(order in range(0,spec.cutoff+1,2) and len(indices)==order+1
                    and all(i in range(3) for i in indices) for spec,word in words for order,indices in word))
            if row['candidate_id']=='F1':self.assertEqual(plan.case.step_count,16)

    def test_representatives_cover_central_sign_flip_both_T_and_high_order(self):
        for row in b0.cases():
            selected=b0.selected_stages(row)[1]
            a=b0.trajectory(row,2);b=b0.trajectory(row,3)
            self.assertEqual(a[0][1][0][0],row['rte_parameters']['K'])
            self.assertEqual(b[selected][1][-1][0],2)
            if row['case']['method']=='outer-Yoshida4':
                self.assertLess(b[selected][0].duration*row['case']['total_time'],0)
                self.assertNotEqual(a[0][0].duration,b[selected][0].duration)

    def test_scope_counts_include_leaf_builds_and_1692_mean_terms(self):
        s=b0.scope();self.assertEqual((s['cases'],s['trajectories'],s['compile_requests']),(12,48,96))
        self.assertEqual((s['operator_gates'],s['expectations'],s['single_means'],s['event_matrix_terms']),(288,192,24,1692))
        self.assertEqual(s['boundaries']['b0.config'],76)
        self.assertEqual(s['boundaries']['b0.exponential'],206)
        self.assertEqual(s['compiler']['approximation_degree'],1.)
        self.assertFalse(s['compiler']['qubits_initially_zero'])
        self.assertEqual(s['compiler']['seed_transpiler'],17)

    def test_identity_keys_include_duration_law_control_and_event_sequence(self):
        spec,word=b0.trajectory(b0.cases()[0],0)[0]
        from dataclasses import replace
        keys={b0.event_key(s,word[0]) for s in (spec,replace(spec,cutoff=4),replace(spec,rte_steps=4),replace(spec,duration=-spec.duration))}
        self.assertEqual(len(keys),4)
        self.assertNotEqual(b0.rte_key(spec,word,False),b0.rte_key(spec,word,True))

    def test_full_mock_uses_actual_renderer_reference_traversal_and_exact_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            run=dict(root=ROOT,output=Path(directory)/'fresh',scope=b0.scope(),source='MOCK',inputs={})
            status=run_prepared(run,lambda r,l,e:b0.execute(r,l,e,native_factory=MockNative),mock_only=True)
            self.assertEqual(status['status'],'COMPLETED_DESCRIPTIVE_MANDATORY_STOP',status)
            result=json.loads((run['output']/'result.json').read_text())
            self.assertEqual(len(result['rows']),96)
            self.assertTrue(all(r['metrics']['mock'] for r in result['rows']))
            self.assertIsNone(result['expected_cost']);self.assertIsNone(result['shots'])
            events=[json.loads(x) for x in (run['output']/'events.jsonl').read_text().splitlines()]
            metrics=[i for i,e in enumerate(events) if e.get('event')=='start' and e.get('boundary')=='b0.measure']
            gates=[i for i,e in enumerate(events) if e.get('event')=='complete' and (e.get('boundary','').endswith('_gate') or e.get('boundary')=='b0.compiled_expectation')]
            self.assertGreater(min(metrics),max(gates))

    def test_gate_failure_stops_before_any_cost_measurement(self):
        for fail in ('input_gate','single_mean','system_gate','controlled_gate','compiled_gate','compiled_expectation'):
            with self.subTest(fail=fail),tempfile.TemporaryDirectory() as directory:
                run=dict(root=ROOT,output=Path(directory)/'fail',scope=b0.scope(),source='MOCK',inputs={})
                factory=lambda r,l:MockNative(r,l,fail=fail)
                status=run_prepared(run,lambda r,l,e:b0.execute(r,l,e,native_factory=factory),mock_only=True)
                self.assertEqual(status['status'],'FAILED_MANDATORY_STOP')
                self.assertNotIn('b0.measure',status['calls']['attempted'])
                self.assertTrue((run['output']/'STOP.json').is_file())

    def test_first_diagnostic_exact_scope_and_no_costs(self):
        with tempfile.TemporaryDirectory() as directory:
            run=dict(root=ROOT,output=Path(directory)/'diagnostic',scope=b0.scope(diagnostic=True),source='MOCK',inputs={})
            status=run_prepared(run,lambda r,l,e:b0.execute(r,l,e,native_factory=MockNative),mock_only=True)
            self.assertEqual(status['status'],'COMPLETED_DESCRIPTIVE_MANDATORY_STOP',status)
            self.assertNotIn('b0.measure',status['calls']['attempted'])
            self.assertEqual(status['calls']['completed']['b0.compile'],1)
            self.assertEqual(status['calls']['completed']['b0.system_build'],1)

    def test_failed_check_details_are_retained_without_scientific_import(self):
        class Matrix:
            def __sub__(self,other):return self
            def __iter__(self):return iter([[complex(1)]])
        native=object.__new__(b0.Native)
        native.np=NS(linalg=NS(norm=lambda *a,**kw:2.),isfinite=lambda x:True)
        native.ledger=FollowupLedger(dict(boundaries={'b0.compiled_gate':1},calls=[['b0.compiled_gate','first']],max_event_records=4))
        with self.assertRaises(Stage1Failure):
            native.call('compiled_gate','first',lambda:native.check(Matrix(),Matrix()))
        row=native.ledger.rows[0]
        self.assertEqual(row['diagnostics']['spectral_error'],2.)
        self.assertEqual(row['diagnostics']['atol'],1e-12)
        self.assertIsNone(row['metrics'])
        self.assertFalse(row['diagnostics']['phase_fit'])

    def test_shared_wrapper_functions_execute_only_against_stub_circuit(self):
        # Extract only stdlib-free function definitions; never import Qiskit.
        source=ast.parse((ROOT/'src/trotterlib/rpe_hadamard_interrogation.py').read_text())
        functions=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in ('append_hadamard_readout','build_prepared_hadamard_wrapper')]
        namespace={'QuantumCircuit':Circuit}
        exec(compile(ast.Module(body=functions,type_ignores=[]),'opaque-wrapper','exec'),namespace)
        for axis in ('cosine','sine'):
            wrapper=namespace['build_prepared_hadamard_wrapper'](Circuit(3),ancilla=2,axis=axis,occupied_qubits=(0,))
            names=[op[0] for op in wrapper.log]
            self.assertEqual(names,['x','h','compose']+(['sdg'] if axis=='sine' else [])+['h','measure'])
            self.assertEqual(wrapper.log[0],('x',0))

    def test_shared_block_glue_does_not_pretend_to_be_partial_S2(self):
        source=ast.parse((ROOT/'src/trotterlib/df_partial_s2.py').read_text())
        cls=next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name=='QiskitDFPartialS2CircuitBuilder')
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='append_block_evolution')
        namespace={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'opaque-block','exec'),namespace)
        calls=[];builder=NS(_append_block=lambda *args,**kw:calls.append((args,kw)))
        namespace['append_block_evolution'](builder,'OPAQUE_QC','OPAQUE_BLOCK',-.125,controlled=True,ancilla_qubit=2)
        self.assertTrue(calls[0][0][2].controlled);self.assertEqual(calls[0][1]['evolution_time'],-.125)

    def test_inspect_CLI_never_builds_and_no_science_modules_are_imported(self):
        with patch.object(sys,'argv',['run_b2d_outer_b0.py','--inspect']),patch('sys.stdout',new_callable=io.StringIO) as out,patch.object(b0,'execute') as work:
            with self.assertRaises(SystemExit) as code:runpy.run_path(str(ROOT/'scripts/run_b2d_outer_b0.py'),run_name='__main__')
            self.assertEqual(code.exception.code,0);self.assertEqual(json.loads(out.getvalue())['scope']['trajectories'],48);work.assert_not_called()
        self.assertFalse(FORBIDDEN.intersection(n.split('.')[0] for n in sys.modules))


if __name__=='__main__':unittest.main()
