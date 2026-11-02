"""Artificial circuits/metadata only; molecular state files are never read."""
import ast
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from qiskit import QuantumCircuit, qpy
from qiskit.circuit.library import XXPlusYYGate

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import h6_m5_phase_repair_contract as c
import h6_m5_phase_repair_native as native
import run_h6_m5_phase_repair as runner
import m5_phase_repair_prototype as prototype
import h6_m5_phase_engine as engine
import inspect_h6_m5_phase_structure as structure


def fake_success(cfg,event):
    for stage in c.STAGES:c.reserve(cfg,stage)
    c.save(Path(cfg['output'])/'report.json',dict(status='REPAIR_COSINE_SINGLE_STATE_PASS_MANDATORY_STOP',SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],checks=[dict(PASS=True,strictly_below=1e-9,raw_error=0.) for _ in range(3)],artifacts={}))


def fake_failure(cfg,event):
    c.reserve(cfg,c.STAGES[0]);c.save(Path(cfg['output'])/'STOP.json',dict(reason='ARTIFICIAL_STOP'))


def minimal(base):
    root=base/'run'
    return dict(SOURCE='a'*40,plan_sha256='b'*64,pipeline_identity='c'*64,
        runtime={'python':sys.executable},grant={'valid':True},run_root=str(root),
        output=str(root/'output'),ledger=str(root/'ledger'),marker=str(root/'CONSUMED.json'),resources={})


class ContractTests(unittest.TestCase):
    def test_draft_CLI_before_science(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'draft.json';path.write_text(json.dumps(dict(authorized=False,grant={'valid':False})))
            for mode in ('--execute','--supervise'):
                r=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/run_h6_m5_phase_repair.py'),mode,'--config',str(path)],capture_output=True,text=True)
                self.assertEqual(r.returncode,2);self.assertIn('EXPLICIT_H6_REPAIR',r.stdout)
            self.assertEqual(list(Path(tmp).iterdir()),[path])

    def test_old_grant_or_expanded_compile_rejected(self):
        cfg=dict(authorized=True,SOURCE='a'*40,plan_sha256='b'*64,pipeline_identity='c'*64,grant=dict(valid=True,scope_id='OLD'))
        with self.assertRaisesRegex(ValueError,'EXACT_NEW_REPAIR_GRANT'):c.authorize(cfg)
        g=dict(scope_id=c.SCOPE,SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],pipeline_identity=cfg['pipeline_identity'],maximum_new_compile=2,maximum_statevector_replays=2,sine=0,sampling=0,retry=False,accepted_cost_rows=0,next_phase_authorized=False,grant_id='ARTIFICIAL',valid=True)
        cfg['grant']=g
        with self.assertRaisesRegex(ValueError,'EXACT_NEW_REPAIR_GRANT'):c.authorize(cfg)

    def test_stage_order_and_reentry(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=minimal(Path(tmp));Path(cfg['ledger']).mkdir(parents=True)
            c.save(cfg['marker'],dict(state='CONSUMED_NO_RETRY_NO_REFUND',grant=cfg['grant']))
            with self.assertRaisesRegex(ValueError,'STAGE_ORDER'):c.reserve(cfg,'COMPILE_COSINE')
            c.reserve(cfg,'LOAD_SAVED_INPUT')
            with self.assertRaisesRegex(ValueError,'STAGE_ORDER'):c.reserve(cfg,'LOAD_SAVED_INPUT')

    def test_scope_preserves_no_sine_no_cost_old_STOP(self):
        s=c.scope();self.assertEqual(s['maximum_new_compile'],1);self.assertEqual(s['maximum_statevector_replays'],2)
        self.assertEqual(s['raw_strictly_below'],1e-9);self.assertEqual(s['sine'],0)
        self.assertEqual(s['accepted_cost_rows'],0);self.assertTrue(s['old_STOP_remains_STOP'])
        self.assertFalse(s['phase_alignment_for_acceptance']);self.assertFalse(s['retry'])


class NumericalPortTests(unittest.TestCase):
    def test_port_math_AST_identical_to_verified_prototype(self):
        def get(path,name):return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
        old=get(ROOT/'scripts/m5_phase_repair_prototype.py','lower_native_once')
        new=get(ROOT/'scripts/h6_m5_phase_repair_native.py','_lower_native_core')
        self.assertEqual([ast.dump(n) for n in old.body[2:]], [ast.dump(n) for n in new.body[1:]])

    def test_port_matches_prototype_gate_bits_and_scalar(self):
        from test_m5_phase_repair_prototype import toy,bytes_metadata
        a,ra=prototype.lower_native_once(toy(),artificial=True);b,rb=native._lower_native_core(toy())
        self.assertEqual(bytes_metadata(a),bytes_metadata(b));self.assertEqual(ra,rb)

    def test_raw_boundary_nonfinite_no_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=np.zeros(1,complex)
            self.assertTrue(native.gate_error('small',a,np.array([.9e-9]),tmp)['PASS'])
            for name,value in [('edge',1e-9),('nan',float('nan')),('phase',2e-9j)]:
                with self.assertRaisesRegex(ValueError,'NUMERIC_DISAGREEMENT'):native.gate_error(name,a,np.array([value]),tmp)
                self.assertFalse(c.read(Path(tmp)/(name+'.json'))['PASS'])


class SciencePathTests(unittest.TestCase):
    def artificial(self,base,bad_reference=False):
        cfg=minimal(base);Path(cfg['output']).mkdir(parents=True);Path(cfg['ledger']).mkdir()
        c.save(cfg['marker'],dict(state='CONSUMED_NO_RETRY_NO_REFUND',grant=cfg['grant']))
        qc=QuantumCircuit(13,1);qc.h(12);qc.append(XXPlusYYGate(.23,.9),[0,1]);qc.p(-.4,12);qc.h(12);qc.measure(12,0)
        low=np.zeros(8192,complex);low[0]=1.;before=engine.ordinary(qc,low)
        if bad_reference:before=before.copy();before[0]+=.001
        inputs={}
        path=base/'wrapper.qpy'
        with path.open('wb') as f:qpy.dump(qc,f)
        inputs['wrapper_qpy']=c.seal(path,'wrapper')
        path=base/'metadata.json';c.save(path,dict(circuits={'WRAPPER':structure.summarize(qc,'WRAPPER',13)}));inputs['static_metadata']=c.seal(path,'metadata')
        for role,arrays in [('initial_npz',dict(low=low)),('ordinary_vector',dict(vector=before)),('saved_vectors_npz',dict(low=low,before=before,after=before,independent_expected_high=low[:4096]))]:
            path=base/(role+'.npz')
            with path.open('wb') as f:np.savez(f,**arrays)
            inputs[role]=c.seal(path,role)
        import qiskit,mpmath
        cfg.update(inputs=inputs,sealed_files=[dict(path=str(Path(m.__file__).resolve())) for m in (np,qiskit,mpmath)],resources={'candidate_instruction_bound':100})
        return cfg

    def test_artificial_13q_full_body_one_compile_two_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=self.artificial(Path(tmp))
            with patch.object(c,'authorize'),patch.object(c,'verify_seals'):
                result=native.run_science(cfg,mp.get_context('spawn').Event())
            self.assertEqual(result['status'],'REPAIR_COSINE_SINGLE_STATE_PASS_MANDATORY_STOP')
            self.assertEqual(len(result['checks']),3);self.assertTrue(all(row['PASS'] for row in result['checks']))
            self.assertEqual({p.stem for p in Path(cfg['ledger']).glob('*.json')},set(c.STAGES))
            self.assertEqual(result['new_compile'],1);self.assertEqual(result['statevector_replays'],2)
            self.assertEqual(result['accepted_cost_rows'],0)
            for row in result['artifacts'].values():self.assertEqual(c.sha(Path(cfg['output'])/row['file']),row['sha256'])

    def test_lowered_mismatch_suppresses_compile(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=self.artificial(Path(tmp),bad_reference=True)
            with patch.object(c,'authorize'),patch.object(c,'verify_seals'),patch('qiskit.transpile',side_effect=AssertionError('NO_COMPILE')):
                with self.assertRaisesRegex(ValueError,'lowered_vs_saved'):native.run_science(cfg,mp.get_context('spawn').Event())
            self.assertFalse((Path(cfg['ledger'])/'COMPILE_COSINE.json').exists())
            self.assertTrue((Path(cfg['output'])/'lowered_vector.npz').exists())


class LauncherTests(unittest.TestCase):
    def test_spawn_native_exit_reap_and_one_shot(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=minimal(Path(tmp))
            with patch.object(c,'authorize'),patch.object(c,'verify_seals'),patch.object(runner.observed,'fresh_resources',return_value={'workers':1}):
                self.assertEqual(runner.execute(cfg,target=fake_success),0)
                with self.assertRaisesRegex(ValueError,'UNUSED_REPAIR_ROOT'):runner.execute(cfg,target=fake_success)
            end=c.read(Path(cfg['run_root'])/'completion_witness.json')
            self.assertEqual(end['worker_exit'],0);self.assertTrue(end['worker_reaped'])
            self.assertTrue(end['mandatory_STOP']);self.assertFalse(end['next_phase_started'])

    def test_scientific_STOP_is_not_native_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=minimal(Path(tmp))
            with patch.object(c,'authorize'),patch.object(c,'verify_seals'),patch.object(runner.observed,'fresh_resources',return_value={'workers':1}):
                self.assertEqual(runner.execute(cfg,target=fake_failure),1)
            end=c.read(Path(cfg['run_root'])/'completion_witness.json')
            self.assertEqual(end['status'],'STOP');self.assertTrue(end['worker_reaped'])
            self.assertEqual(end['stage_entries'],['LOAD_SAVED_INPUT'])


if __name__=='__main__':unittest.main()
