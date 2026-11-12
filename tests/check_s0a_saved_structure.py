"""S0a artificial reader/law/path tests. Never load molecular inputs or raw."""
import ast
from copy import deepcopy
import importlib.util
import math
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from trottertracks import s0a_saved_structure as s
from trottertracks import r_price_phase1 as c1
from trottertracks.s0a_structure_context import context_body, Context, no_circuits
from check_c1_structural_stream import fixture, H4Accounting
from trottertracks.r_price_phase1_science import structural


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'events.json'
        self.blocks,self.candidate,self.alloc,self.events=fixture(q=2)
        self.cell=dict(system='H4',id='ARTIFICIAL_H4',candidate_id='MOCK',role='VALIDATION',cohort='MOCK',seeds=[[17,18]],
            request=dict(identity=dict(candidate=self.candidate,structure=dict(allocation=self.alloc))))
        self.laws=[dict(steps=3,orders=[0],probabilities=[1.])]*2
        self.raw=dict(events=self.events,seeds=[17,18],laws=self.laws,conditional=False)
        self.record=dict(S=[structural(H4Accounting,self.blocks,self.candidate,.3,a,self.events)['structural_rz'] for a in c1.AXES],event_digest=c1.digest(self.events))
        self.write()
        self.context=SimpleNamespace(Accounting=H4Accounting,blocks=self.blocks,scalar=.3,
            event_validator=lambda cell:(self.laws,lambda j,e:None))

    def write(self):self.path.write_text(json.dumps(self.raw,ensure_ascii=False,indent=2))
    def reader(self,chunk=7):return s.SavedEvents(self.path,2,self.alloc,self.record['event_digest'],lambda j,e:None,chunk=chunk)

    def test_reader_preserves_order_binary64_and_digest_at_small_chunks(self):
        self.events[0][0]['rotation_angle']=-0.0
        self.events[0][2]['rotation_angle']=float.fromhex('0x1.0000000000001p-3')
        self.record['event_digest']=c1.digest(self.events);self.write()
        for chunk in [1,7,4096]:
            reader=self.reader(chunk);actual=[list(batch) for batch in reader.batches()]
            self.assertEqual(actual,self.events)
            self.assertEqual(actual[0][0]['rotation_angle'].hex(),'-0x0.0p+0')
            self.assertEqual(reader.digest,self.record['event_digest']);s.check_metadata(reader,self.cell,self.laws)

    def test_real_consumer_on_artificial_path_matches_original_saved_baseline(self):
        result=s.parity_one(self.context,self.cell,self.record,self.path)
        self.assertTrue(result['parity']);self.assertEqual(result['S_stream'],self.record['S'])
        self.assertFalse(result['qualified_for_candidate']);self.assertIsNone(result['mean_compiled_price'])

    def test_missing_extra_events_and_tails_fail(self):
        for change in ('missing_event','extra_event','missing_tail','extra_tail'):
            self.raw['events']=deepcopy(self.events)
            if change=='missing_event':self.raw['events'][0].pop()
            if change=='extra_event':self.raw['events'][0].append(self.raw['events'][0][0])
            if change=='missing_tail':self.raw['events'].pop()
            if change=='extra_tail':self.raw['events'].append(self.raw['events'][0])
            self.write()
            with self.assertRaises(ValueError):[list(b) for b in self.reader().batches()]

    def test_digest_law_seed_conditional_and_incomplete_tail_rejected_before_S(self):
        for field,value in [('seeds',[18,17]),('laws',[]),('conditional',True),('events',list(reversed(self.events)))]:
            original=deepcopy(self.raw);self.raw[field]=value;self.write()
            with patch.object(self.context,'Accounting',side_effect=AssertionError('S must not start')):
                with self.assertRaises(ValueError):s.parity_one(self.context,self.cell,self.record,self.path)
            self.raw=original
        self.write();batches=self.reader().batches();next(batches)
        with self.assertRaises(ValueError):next(batches)

    def test_duplicate_keys_nonfinite_trailing_and_missing_file_rejected(self):
        valid=self.path.read_text()
        for text in ['{"events":[],"events":[]}',valid+'false',valid.replace('0.125','NaN',1),valid[:-1]]:
            self.path.write_text(text)
            with self.assertRaises(ValueError):[list(b) for b in self.reader().batches()]
        self.path.unlink()
        with self.assertRaises(FileNotFoundError):[list(b) for b in self.reader().batches()]

    def test_law_validation_uses_pinned_components_basis_phase_and_signed_time(self):
        # Existing toy tail and exact _make_event; no sampling, circuits or molecule.
        spec=importlib.util.spec_from_file_location('_s0a_pure_rte',ROOT/'src/trotterlib/rte.py')
        rte=importlib.util.module_from_spec(spec);sys.modules[spec.name]=rte;spec.loader.exec_module(rte)
        tail=SimpleNamespace(tail_id='ARTIFICIAL',tail_hash='ARTIFICIAL',lambda_r=1.,components=(
            rte.RTEComponent('a',.7,.7,1,basis_id='a',basis_hash='a',diagonal_pauli_support=(0,)),
            rte.RTEComponent('b',.3,.3,-1,basis_id='b',basis_hash='b',diagonal_pauli_support=(1,))))
        prep=SimpleNamespace(rte_preparation=SimpleNamespace(symbolic_tail=tail),num_system_qubits=8)
        ctx=Context.__new__(Context);ctx.ctx=SimpleNamespace(prep=prep);ctx.rte=rte;ctx.tolerance=lambda tau,K:math.nextafter(rte.step_taylor_truncation_residual_bound(tau,K),math.inf)
        cell=dict(seeds=[[17,18]],request=dict(identity=dict(candidate=dict(q=1,T=.4,K=2,system='H4'),
            structure=dict(allocation=[2,3],signed_weights=[1.,-.5]))))
        laws,validator=ctx.event_validator(cell)
        for j,(time,r) in enumerate([(.4,2),(-.2,3)]):
            _,dist=rte.make_rte_config(tail,evolution_time=time,rte_steps=r,truncation_tolerance=ctx.tolerance(time/r,2),finite_taylor_order=2,seed=17+j)
            event=rte._make_event([0],tail.components,dist,0).to_dict();validator(j,event)
            rare=rte._make_event([0,1,0],tail.components,dist,1).to_dict();validator(j,rare)
            self.assertEqual(rare['phase'],dict(real=-1.,imag=0.))
            for field,value in [('rotation_angle',1.),('event_normalization',1.),('phase',dict(real=0.,imag=1.))]:
                bad=deepcopy(event);bad[field]=value
                with self.assertRaises(ValueError):validator(j,bad)
            bad=deepcopy(event);bad['application_sequence'][0]['basis_hash']='WRONG'
            with self.assertRaises(ValueError):validator(j,bad)
        self.assertEqual([l['steps'] for l in laws],[2,3])

    def test_context_guard_refuses_build_and_transpile_before_they_run(self):
        import qiskit
        with no_circuits():
            with self.assertRaisesRegex(RuntimeError,'BUILD_FORBIDDEN'):qiskit.QuantumCircuit(1)
            with self.assertRaisesRegex(RuntimeError,'TRANSPILE_FORBIDDEN'):qiskit.transpile(None)

    def test_context_excludes_compiler_and_sampler_invocations(self):
        body=context_body(ROOT)
        text=ast.unparse(body)
        for forbidden in ['ctx.compile','costs.transpile','transpile_and_measure_cost','sample_requests(','compose(','QuantumCircuit(']:
            self.assertNotIn(forbidden,text)
        self.assertIn('h[\'prepare\']',text);self.assertIn('counter_path_local',text)
        self.assertIn('scalar=ham.constant + prep.extracted_identity_coefficient',text)

    def test_prepare_and_unapproved_execution_never_import_context_or_count(self):
        with patch.object(s,'selected',return_value=({'runtime':{}},[])),patch.dict(sys.modules,{'trottertracks.s0a_structure_context':None}),patch('trottertracks.mean_compiled_price.stream_structure',side_effect=AssertionError('No S')):
            p=s.prepare(ROOT,'MOCK_SOURCE')
            self.assertIsNone(p['mean_compiled_price']);self.assertEqual(p['new_stream_S'],8)
            with self.assertRaises(ValueError):s.execute(ROOT,p,Path(self.tmp.name)/'out')
            self.assertFalse((Path(self.tmp.name)/'out').exists())

    def test_mock_runner_executes_only_eight_stream_counts_and_stops(self):
        plan=dict(source='MOCK',paths=[dict(raw_exists=True)]*4)
        rows=[]
        for i in range(4):
            cell=deepcopy(self.cell);cell['id']='MOCK_'+str(i)
            rows.append(dict(cell=cell,record=self.record,raw=str(self.path)))
        out=Path(self.tmp.name)/'out'
        from trottertracks import mean_compiled_price as mean
        with patch.object(s,'prepare',return_value=plan),patch.object(s,'selected',return_value=({},rows)),patch.object(s.c1,'blob',side_effect=lambda root,source,name:(ROOT/name).read_bytes()),patch.object(mean,'stream_structure',wraps=mean.stream_structure) as count:
            result=s.execute(ROOT,plan,out,execute=True,context_factory=lambda cell:self.context)
            self.assertEqual(count.call_count,8);self.assertEqual(result['new_stream_S'],8)
            self.assertTrue(json.loads((out/'STOP.json').read_text())['mandatory_stop'])
            with self.assertRaises(FileExistsError):s.execute(ROOT,plan,out,execute=True,context_factory=lambda cell:self.context)
        self.assertEqual(len(json.loads((out/'RESULTS.json').read_text())['paths']),4)

    def test_mock_failure_writes_STOP_and_preserves_missing_result(self):
        plan=dict(source='MOCK',paths=[dict(raw_exists=True)]*4)
        out=Path(self.tmp.name)/'failed'
        with patch.object(s,'prepare',return_value=plan),patch.object(s,'selected',return_value=({},[dict(cell=self.cell,record=self.record,raw=str(self.path))])),patch.object(s.c1,'blob',side_effect=lambda root,source,name:(ROOT/name).read_bytes()):
            with self.assertRaises(ValueError):s.execute(ROOT,plan,out,execute=True,context_factory=lambda cell:(_ for _ in ()).throw(ValueError('MOCK_MISSING_INPUT')))
        self.assertFalse((out/'RESULTS.json').exists())
        self.assertEqual(json.loads((out/'STOP.json').read_text())['status'],'S0A_FAILED_NO_AUTOMATIC_RETRY')

    def test_saved_S_mismatch_stops_without_model_fallback(self):
        record=deepcopy(self.record);record['S'][0]+=1
        with self.assertRaisesRegex(ValueError,'SAVED_S_PARITY'):s.parity_one(self.context,self.cell,record,self.path)


if __name__=='__main__':unittest.main(verbosity=2)
