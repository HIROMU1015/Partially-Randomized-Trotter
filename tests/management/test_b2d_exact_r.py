"""Opaque Exact-R scope/IO tests. No scientific imports or fixed-fixture builds."""
from dataclasses import replace
from fractions import Fraction as F
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(Path(__file__).resolve().parent)]
from test_b2d_evaluator import Barrier
from test_b2d_b0_native import Circuit, Opaque

guard = Barrier()
sys.meta_path.insert(0, guard)
try:
    from trottertracks.b2d_evaluator import exact_r as e
    from trottertracks.b2d_evaluator.followup_io import run_prepared, FollowupLedger
finally:
    sys.meta_path.remove(guard)


class MockNative(e.Native):
    def __init__(self, run, ledger, fail=None):
        self.run, self.ledger, self.fail = run, ledger, fail
        self.cache, self.compile_cache = {}, {}
        self.context = e.b0.candidate_graph('MOCK').settings.context
        self.QC = Circuit
        self.scalar = .37
        self.Operator = lambda c: NS(data=Opaque())
        for k in ('fixture', 'split', 'prepare', 'full_inputs', 'input_gate', 'tail_prepare', 'tail_identity_gate'):
            self.call(k, 'input', lambda: None)
        self.input_diagnostics, self.tail_diagnostics = {}, {}

    def call(self, kind, key, function):
        def checked():
            if kind == self.fail:
                raise e.Stage1Failure('MOCK_FAILURE')
            return function()
        return super().call(kind, key, checked)

    def block(self, label, time, controlled):
        return self.cached('block_build', e.b0.block_key(label, time, controlled),
                           lambda: Circuit(3 if controlled else 2))

    def tail(self, time, controlled):
        return self.cached('tail_build', e.b0.block_key('R', time, controlled),
                           lambda: Circuit(3 if controlled else 2))

    def exp(self, label, time):
        return self.cached('exponential', e.b0.exp_key(label, time), lambda: Opaque())

    def exact_reference(self, row):
        return Opaque()

    def controlled(self, u): return Opaque()
    def circuit_gate(self, *a, **k): return 0.
    def wrapper(self, *a): return Circuit(3)
    def wrapper_reference(self, *a): return Opaque()
    def expectation(self, *a): return dict(mock=True)
    def compile(self, c): return c
    def measure(self, c): return dict(mock=True)
    def signals(self, *a): return dict(mock=True, normalization=1)
    def saved_baseline(self, *a): return dict(mock=True)
    def provenance(self): return dict(mock=True)


class ExactRTests(unittest.TestCase):
    def setUp(self):
        guard = Barrier()
        sys.meta_path.insert(0, guard)
        self.addCleanup(sys.meta_path.remove, guard)

    def test_fixed_grid_and_scope_exclusions(self):
        rows = e.cases()
        self.assertEqual([r['case_id'] for r in rows], ['E1', 'E2', 'E3', 'E4'])
        self.assertEqual([r['baseline_candidate_id'] for r in rows], ['B3', 'B3', 'B4', 'B4'])
        s = e.scope()
        self.assertEqual((s['new_exact_r_cases'], s['wrappers'], s['compile_requests']), (4, 8, 8))
        self.assertEqual((s['normalization'], s['phase_fit'], s['sampling']), (1, False, False))
        self.assertFalse(any(k.startswith(('rte.', 'b0.')) for k in s['boundaries']))
        self.assertFalse(s['retry_allowed']); self.assertFalse(s['resume_allowed'])

    def test_scope_rejects_candidate_time_mode_or_inner_changes(self):
        for k, value in (('case_id', 'E5'), ('tail_mode', 'finite-rte'), ('baseline_candidate_id', 'D1')):
            bad = {**e.cases()[0], k: value}
            with self.assertRaises(e.Stage1Failure): e.plan(bad)
        for k, value in (('total_time', F(1, 4)), ('n_d', 2), ('macro_steps', 4), ('inner_arm', 'A3')):
            bad = dict(e.cases()[0]); bad['case'] = {**bad['case'], k: value}
            with self.assertRaises(e.Stage1Failure): e.plan(bad)

    def test_descriptor_signed_time_and_centered_phase(self):
        block = NS(original_fragment_index=7, block_kind='df_fragment')
        for time in (F(2, 7), F(-2, 7)):
            stage = e.ExactRStage(7, 'TOY_TAIL', block, time, .43)
            for controlled in (False, True):
                calls = []
                builder = NS(append_block_evolution=lambda *a, **kw: calls.append((a, kw)))
                qc = Circuit(3 if controlled else 2)
                e.append_exact_r(qc, stage, builder, controlled=controlled, ancilla_qubit=2 if controlled else None)
                self.assertEqual(calls[0][0][2], float(time))
                self.assertEqual(calls[0][1]['controlled'], controlled)
                if controlled: self.assertEqual(qc.log, [('p', float(time)*.43, 2)])
                else: self.assertEqual(qc.global_phase, float(time)*.43)
            with self.assertRaises(ValueError): replace(stage, fragment_index=2)
            with self.assertRaises(ValueError): replace(stage, tail_mode='finite-rte')
            with self.assertRaises(ValueError): e.append_exact_r(Circuit(3), stage, builder, controlled=True)
        with self.assertRaises(ValueError): replace(stage, duration=0)

    def test_exact_renderer_preserves_every_D_R_position_and_scalar_once(self):
        for row in e.cases():
            native = object.__new__(e.Native)
            native.QC = Circuit
            native.context = e.b0.candidate_graph('TOY_METADATA').settings.context
            native.scalar = .37
            native.block = lambda label,t,c: ('D',label,t,c)
            native.tail = lambda t,c: ('R',t,c)
            word = e.plan(row, native.context)
            expected = []
            for stage in word.body:
                if isinstance(stage, e.DStage):
                    expected.extend(('D','D_'+str(native.context.block_ids.index(op.block_id)),op.evolution_time,True)
                                    for op in stage.append_operations())
                else: expected.append(('R',stage.evolution_time,True))
            qc = native.render_exact(row, True)
            self.assertEqual([x[1] for x in qc.log if x[0]=='compose'], expected)
            self.assertEqual([x for x in qc.log if x[0]=='p'], [('p',-float(word.case.total_time)*.37,2)])
            self.assertEqual(sum(x[0]=='R' for x in expected), word.cost_metadata['R_occurrences'])

    def mock_run(self, directory, fail=None):
        run = dict(root=ROOT, output=Path(directory)/'fresh', scope=e.scope(), source='MOCK', inputs={})
        status = run_prepared(run, lambda r,l,v: e.execute(r,l,v,
            native_factory=lambda r,l: MockNative(r,l,fail)), mock_only=True)
        return run, status

    def test_full_mock_exact_scope_all_gates_before_signal_or_cost(self):
        with tempfile.TemporaryDirectory() as d:
            run, status = self.mock_run(d)
            self.assertEqual(status['status'], 'COMPLETED_DESCRIPTIVE_MANDATORY_STOP', status)
            self.assertEqual(status['calls']['completed'], run['scope']['boundaries'])
            result = json.loads((run['output']/'result.json').read_text())
            self.assertEqual(len(result['rows']), 4)
            self.assertEqual(result['normalization'], 1)
            events = [json.loads(x) for x in (run['output']/'events.jsonl').read_text().splitlines()]
            gate_ends = [i for i,x in enumerate(events) if x.get('event')=='complete' and
                         (x['boundary'].endswith('_gate') or x['boundary'].endswith('_expectation'))]
            comparisons = [i for i,x in enumerate(events) if x.get('event')=='start' and
                           x['boundary'] in ('exact_r.signals','exact_r.measure')]
            self.assertLess(max(gate_ends), min(comparisons))
            duplicate = run_prepared(run, lambda *args: self.fail('no resume'), mock_only=True)
            self.assertEqual(duplicate['status'], 'FRESH_OUTPUT_FAILED_MANDATORY_STOP')

    def test_failure_keeps_partial_diagnostics_and_no_comparison_or_cost(self):
        for fail in ('input_gate','tail_identity_gate','tail_gate','system_gate','controlled_gate',
                     'raw_wrapper_gate','compile','compiled_gate','compiled_expectation'):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as d:
                run, status = self.mock_run(d, fail)
                self.assertEqual(status['status'], 'FAILED_MANDATORY_STOP', status)
                self.assertNotIn('exact_r.measure', status['calls']['attempted'])
                self.assertNotIn('exact_r.signals', status['calls']['attempted'])
                partial = json.loads((run['output']/'partial_results.json').read_text())
                self.assertTrue(any(x.get('error_type') for x in partial['rows']))
                self.assertTrue((run['output']/'STOP.json').exists())

    def test_unregistered_or_repeated_calls_rejected(self):
        ledger = FollowupLedger(e.scope())
        with self.assertRaises(e.Stage1Failure): ledger.call('b0.config', lambda: None, identity='input')
        ledger.call('exact_r.fixture', lambda: None, identity='input')
        with self.assertRaises(e.Stage1Failure): ledger.call('exact_r.fixture', lambda: None, identity='input')

    def test_saved_join_schema_and_case_mismatch_use_only_stub_numbers(self):
        row = e.cases()[0]
        old = dict(candidate_id='B3',case={**row['case'],'total_time':'9/50'},
                   rte_parameters={'r_rte':2,'K':2},outputs={'signals':{
                       'physical':{'X':1,'Y':0},'ideal_tail':{'X':1,'Y':0}}})
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/e.SNAPSHOT/'result.json'; path.parent.mkdir(parents=True)
            path.write_text(e._encode({'rows':[old]}))
            native = object.__new__(e.Native)
            native.run = {'root':Path(d)}; native.state = Opaque()
            native.np = NS(vdot=lambda *a:1,asarray=lambda x: x)
            native.exp = lambda *a: Opaque(); native.check = lambda *a:0.
            joined = native.saved_baseline(row,Opaque())
            self.assertEqual(set(joined['saved_signal_consistency']),{'ideal_tail','physical'})
            old['rte_parameters']['K'] = 4
            path.write_text(e._encode({'rows':[old]}))
            with self.assertRaises(e.Stage1Failure): native.saved_baseline(row,Opaque())

    def test_readonly_preflight_preserves_source_fresh_output_and_profile_checks(self):
        output = ROOT/e.OUTPUT_PARENT/'__mock_preflight_unused__'
        self.assertFalse(output.exists())
        with patch('trottertracks.b2d_evaluator.followup_io._check_source') as source, \
             patch('trottertracks.b2d_evaluator.followup_io.snapshot_metadata', return_value=({},{})), \
             patch('importlib.metadata.version', return_value='1.3.0'), \
             patch.object(e, 'Native', side_effect=AssertionError('science forbidden')):
            run = e.preflight(ROOT,output,'MOCK_SOURCE')
            source.assert_called_once_with(ROOT.resolve(),'MOCK_SOURCE',output=output.resolve())
            self.assertEqual(run['scope'],e.scope())
            self.assertFalse(output.exists())
        with self.assertRaises(ValueError):
            e.preflight(ROOT,output,'0'*40)  # Real SOURCE check rejects before science.
        with patch.object(e, 'followup_preflight', return_value={'versions':{'qiskit':'WRONG'}}):
            with self.assertRaises(ValueError): e.preflight(ROOT,output,'MOCK')

    def test_inspect_and_cli_need_no_approval_artifacts_or_science(self):
        runner = ROOT/'scripts/run_b2d_exact_r.py'
        command = [sys.executable, '-I', '-B', '-S', str(runner)]
        output = subprocess.run(command+['--inspect'], check=True, capture_output=True, text=True)
        self.assertEqual(json.loads(output.stdout)['status'], 'METADATA_ONLY_NO_SCIENCE')
        self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
        with patch.object(e, 'preflight', return_value=dict(source='TOY',versions={},scope={})) as preflight, \
             patch.object(e, 'run_prepared', side_effect=AssertionError('science forbidden')), \
             patch.object(sys, 'argv', ['pilot','--preflight','--source-commit','TOY','--output','TOY']), \
             patch('sys.stdout', new=io.StringIO()):
            self.assertEqual(e.main(), 0)
            preflight.assert_called_once()


if __name__ == '__main__': unittest.main()
