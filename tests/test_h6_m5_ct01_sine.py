"""Artificial circuits/metadata/processes only; no saved molecular arrays loaded."""
import ast
import copy
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h6_m5_ct01_sine_contract as c
import h6_m5_ct01_sine_native as native
import run_h6_m5_ct01_sine as runner


def fixture(root):
    artifact=lambda label:dict(sha256=c.digest(label),bytes=10)
    inputs={k:artifact(k) for k in ('wrapper_qpy','compiled_cosine_qpy','initial_npz','saved_vectors_npz')}
    error=1.504043707445665e-9
    checks=[dict(check=k,error=v) for k,v in zip(('normalization_log_B','native_low','native_leakage','independent_occupation_vector','hadamard_cosine','compiled_vector_cosine'),(0.,8e-14,0.,2e-12,1e-13,error))]
    fp='da9afa1d776bf6da829119b2f4ff888042b243ba7d4b98a037134516ed9cf83c'
    data=dict(old_config=dict(SOURCE=c.RECOVERY_SOURCE,plan_sha256='ffe5d880ecdebdc6616a0e4a41bd3432115503e2feba16f2414a081a2ac098da'),
      old_stop=dict(task=c.old.TASK['id'],lane=c.old.LANE,status='STOP',reason='NUMERIC_DISAGREEMENT:compiled_vector_cosine',checks=checks),
      diagnostics=dict(raw_error=error,raw_gate_PASS=False,acceptance_override=False),
      wrapper_receipt=dict(SOURCE=c.RECOVERY_SOURCE,plan_sha256='ffe5d880ecdebdc6616a0e4a41bd3432115503e2feba16f2414a081a2ac098da',axis='cosine',wrapper_fingerprint=fp,circuit=inputs['wrapper_qpy'],state=inputs['initial_npz']),
      compiled_receipt=dict(pretranspile_fingerprint=fp,compiler_settings_hash=c.COMPILER_HASH,layout='None',circuit=inputs['compiled_cosine_qpy'],metrics=dict(rz_count=831662,cx_count=367008)),
      vectors_receipt=dict(shape=[8192],dtype='complex128',vectors=inputs['saved_vectors_npz']),
      old_completion=dict(SOURCE=c.RECOVERY_SOURCE,status='STOP',worker_exit=0,worker_reaped=True),
      old_exit=dict(native_exit=1,reaped=True))
    for key,value in data.items():
        p=root/(key+'.json');c.save(p,value);inputs[key]=dict(path=str(p))
    return inputs


def fake_worker_PASS(cfg,stop):
    for stage in c.STAGES:c.reserve(cfg,stage)
    artifact=Path(cfg['output'])/'synthetic.txt';artifact.write_text('ARTIFICIAL ONLY\n')
    c.save(Path(cfg['output'])/'cost_sine.json',dict(status='PILOT_AXIS_ACCEPTED_CT01'))
    c.save(Path(cfg['output'])/'report.json',dict(status='CT01_SINE_PASS_MANDATORY_STOP',SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],pipeline_identity=cfg['pipeline_identity'],
      new_compile=1,compiled_state_actions=1,new_cosine_compile=0,new_uncompiled_full_replays=0,new_sampling=0,compiler_policy='ORIGINAL_PRESET_OPT1_UNMODIFIED',
      reused_cosine=cfg['cosine_acceptance'],pilot_axis_receipts=2,cost_accounting=False,ranking=False,main_table_adoption=False,
      checks=[dict(PASS=True,raw_error=0.,strictly_below=t) for t in (1e-9,1e-7)],mandatory_STOP=True,retry=False,next_phase_started=False,AD03='DEFERRED',
      artifacts=dict(toy=dict(file=artifact.name,bytes=artifact.stat().st_size,sha256=c.sha(artifact)))))


def fake_worker_STOP(cfg,stop):
    c.reserve(cfg,c.STAGES[0]);c.save(Path(cfg['output'])/'STOP.json',dict(status='STOP',reason='ARTIFICIAL_MISMATCH'))


class ContractTests(unittest.TestCase):
    def test_metadata_acceptance_preserves_old_FAIL(self):
        with tempfile.TemporaryDirectory() as d:
            inputs=fixture(Path(d));before=c.read(inputs['old_stop']['path']);r=c.assess_saved_cosine(inputs)
            self.assertTrue(r['new_PASS']);self.assertFalse(r['old_PASS'])
            self.assertEqual(r['metrics']['rz_count'],831662);self.assertEqual(r['new_compile'],0)
            self.assertEqual(before,c.read(inputs['old_stop']['path']));self.assertFalse(r['main_table_adoption'])

    def test_wrong_prior_checks_or_provenance_refused(self):
        mutations=[('old_stop',lambda x:x['checks'][1].update(error=2e-9)),
          ('old_stop',lambda x:x.update(status='PASS')),('old_stop',lambda x:x['checks'].pop()),
          ('diagnostics',lambda x:x.update(raw_error=1e-7)),('diagnostics',lambda x:x.update(acceptance_override=True)),
          ('wrapper_receipt',lambda x:x.update(wrapper_fingerprint='wrong')),
          ('compiled_receipt',lambda x:x.update(compiler_settings_hash='wrong')),
          ('compiled_receipt',lambda x:x['circuit'].update(sha256='wrong')),
          ('vectors_receipt',lambda x:x.update(shape=[4096])),
          ('old_completion',lambda x:x.update(worker_reaped=False)),('old_exit',lambda x:x.update(native_exit=0))]
        for role,mutate in mutations:
            with self.subTest(role=role),tempfile.TemporaryDirectory() as d:
                inputs=fixture(Path(d));p=inputs[role]['path'];x=c.read(p);mutate(x);Path(p).unlink();c.save(p,x)
                with self.assertRaises(ValueError):c.assess_saved_cosine(inputs)

    def test_draft_refuses_before_science_import(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'config.json';c.save(p,dict(authorized=False,grant=dict(valid=False)))
            for mode in ('--execute','--supervise'):
                r=subprocess.run([sys.executable,'-B',str(c.ROOT/'scripts/run_h6_m5_ct01_sine.py'),mode,'--config',str(p)],capture_output=True,text=True)
                self.assertEqual(r.returncode,2);self.assertIn('EXPLICIT_CT01_SINE_EXECUTION_APPROVAL_REQUIRED',r.stdout)
            self.assertEqual(len(list(Path(d).iterdir())),1)

    def test_expanded_or_old_grant_refused(self):
        cfg=dict(authorized=True,SOURCE='a'*40,plan_sha256='b'*64,pipeline_identity='c'*64)
        grant=dict(valid=True,grant_id='artificial',scope_id=c.SCOPE,SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],pipeline_identity=cfg['pipeline_identity'],
          maximum_new_compile=1,maximum_compiled_state_actions=1,new_cosine_compile=0,new_sampling=0,compile_vector_strictly_below=1e-7,other_checks_strictly_below=1e-9,retry=False,next_phase_authorized=False)
        for key,value in [('scope_id','OLD'),('maximum_new_compile',2),('new_cosine_compile',1),('new_sampling',1),('retry',True),('compile_vector_strictly_below',1e-6),('other_checks_strictly_below',1e-7)]:
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'EXACT_CT01_SINE_GRANT'):c.authorize(dict(cfg,grant=dict(grant,**{key:value})))

    def test_stage_order_and_no_reentry(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);(r/'ledger').mkdir();cfg=dict(marker=str(r/'CONSUMED.json'),ledger=str(r/'ledger'),grant={},SOURCE='a'*40,plan_sha256='b'*64)
            c.save(cfg['marker'],dict(state='CONSUMED_NO_RETRY_NO_REFUND',grant={}))
            with self.assertRaises(ValueError):c.reserve(cfg,c.STAGES[1])
            for s in c.STAGES:c.reserve(cfg,s)
            for s in (c.STAGES[0],'COMPILE_COSINE'):
                with self.assertRaises(ValueError):c.reserve(cfg,s)

    def test_raw_strict_tolerance_and_nonfinite(self):
        for v,threshold,passed in [(1.5e-9,1e-7,True),(1e-7,1e-7,False),(2e-7,1e-7,False),(float('nan'),1e-7,False),(float('inf'),1e-7,False),(-1.,1e-7,False),(1.5e-9,1e-9,False)]:
            with self.subTest(v=v,threshold=threshold),tempfile.TemporaryDirectory() as d:
                if passed:native.check_error('artificial',v,threshold,d)
                else:
                    with self.assertRaises(ValueError):native.check_error('artificial',v,threshold,d)
                r=c.read(Path(d)/'artificial.json');self.assertEqual(r['PASS'],passed);self.assertFalse(r['phase_alignment'])

    def test_spawn_reap_one_shot_PASS_and_STOP(self):
        for target,expected in [(fake_worker_PASS,0),(fake_worker_STOP,1)]:
            with self.subTest(worker=target.__name__),tempfile.TemporaryDirectory() as d:
                r=Path(d)/'run01';cfg=dict(SOURCE='a'*40,plan_sha256='b'*64,pipeline_identity='c'*64,runtime=dict(python=sys.executable),run_root=str(r),output=str(r/'output'),ledger=str(r/'ledger'),marker=str(r/'CONSUMED.json'),grant={},cosine_acceptance=dict(ARTIFICIAL=True))
                with patch.object(c,'authorize'),patch.object(c,'verify_seals'),patch.object(runner.observed,'fresh_resources',return_value=dict(workers=1)):
                    self.assertEqual(runner.execute(cfg,target=target),expected)
                    w=c.read(r/'completion_witness.json');self.assertTrue(w['worker_reaped']);self.assertEqual(w['worker_exit'],0)
                    self.assertEqual(w['status']=='STOP',expected==1)
                    with self.assertRaisesRegex(ValueError,'UNUSED_CT01_RUN_ROOT'):runner.execute(cfg,target=target)

    def test_one_original_compile_one_action_no_rebuild(self):
        t=ast.parse(Path(native.__file__).read_text());calls=[n.func.id for n in ast.walk(t) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
        self.assertEqual(calls.count('transpile_and_measure_cost'),1);self.assertEqual(calls.count('simulate_statevector'),1)
        text=Path(native.__file__).read_text()
        for forbidden in ('algebraic_zero','phase_repair_prototype','original.context(','phase_alignment=True'):self.assertNotIn(forbidden,text)
        s=c.scope();self.assertEqual(s['maximum_new_compile'],1);self.assertFalse(s['cost_accounting']);self.assertEqual(s['AD03'],'DEFERRED')


class CircuitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        provider=Path('/home/AbeHiromu/projects/h46-higher-pf-acquire/20261018/private/provider')
        sys.path.insert(0,str(provider/'src'))
        import numpy as np
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector,Operator
        from trottertracks.resource_applicability.ax2b_native_df_v5 import NativeEvolution,build_native_hadamard_wrapper
        from trottertracks.resource_applicability.ax2b_stream_fingerprint_v5 import canonical_qiskit_circuit_fingerprint
        cls.np=np;cls.QC=QuantumCircuit;cls.Statevector=Statevector;cls.Operator=Operator
        cls.Evolution=NativeEvolution;cls.builder=staticmethod(build_native_hadamard_wrapper);cls.fp=staticmethod(canonical_qiskit_circuit_fingerprint)

    def wrappers(self,n,phase):
        ev=self.QC(n+1);ev.ry(.37,0);ev.cx(0,n);ev.rz(-.41,n);ev.p(.2,n);ev.global_phase=phase
        e=self.Evolution(ev,n,.8,2,'4th(m5_best)','symmetric_directional',len(ev.data),self.fp(ev))
        return tuple(self.builder(e,axis=a,max_instructions=100) for a in ('cosine','sine'))

    def test_wrapper_matches_original_provider_with_global_phase(self):
        for n in (1,2):
            for phase in (0.,.3,-.9):
                with self.subTest(n=n,phase=phase):
                    cosine,sine=self.wrappers(n,phase);derived=native.sine_wrapper(cosine)
                    self.assertEqual(self.fp(derived),self.fp(sine));self.assertEqual(len(derived.data),len(cosine.data)+1)
                    delta=self.Operator(derived.remove_final_measurements(inplace=False)).data-self.Operator(sine.remove_final_measurements(inplace=False)).data
                    self.assertLess(self.np.linalg.norm(delta),1e-14)

    def test_terminal_transform_matches_direct_artificial_state_action(self):
        for n in (1,2):
            cosine,sine=self.wrappers(n,.37);size=1<<(n+1)
            rng=self.np.random.default_rng(123+n);v=rng.normal(size=size)+1j*rng.normal(size=size);v/=self.np.linalg.norm(v)
            # Statevector.evolve may apply global_phase to the supplied ndarray.
            # Each independent reference action must own its input array.
            before=self.Statevector(v.copy()).evolve(cosine.remove_final_measurements(inplace=False)).data
            direct=self.Statevector(v.copy()).evolve(sine.remove_final_measurements(inplace=False)).data
            self.assertLess(self.np.linalg.norm(native.expected_sine_from_cosine(before)-direct),2e-15)
            # Replacing Sdg by S is observably different for this example.
            half=size//2;a=(before[:half]+before[half:])/math.sqrt(2);b=(before[:half]-before[half:])/math.sqrt(2)
            wrong=self.np.concatenate(((a+1j*b)/math.sqrt(2),(a-1j*b)/math.sqrt(2)))
            self.assertGreater(self.np.linalg.norm(wrong-direct),.1)

    def test_raw_common_phase_not_aligned(self):
        v=self.np.array([1.+0j,0j,0j,0j]);w=v*self.np.exp(5e-8j)
        raw=float(self.np.linalg.norm(w-v))
        with tempfile.TemporaryDirectory() as d:
            r=native.check_error('common_phase',raw,1e-7,d);self.assertGreater(r['raw_error'],4e-8);self.assertFalse(r['phase_alignment'])

    def test_native_adapter_artificial_end_to_end_and_numeric_STOP(self):
        from qiskit import qpy,transpile
        import numpy as np,qiskit
        import h46_m5_native as original
        import trotterlib.df_trotter.circuit as simulation
        import trotterlib.rte_compiled_cost as costs
        ev=self.QC(13);ev.h(0);ev.cx(0,12);ev.p(.2,12);ev.rz(.13,0);ev.global_phase=.37
        e=self.Evolution(ev,12,.8,2,'4th(m5_best)','symmetric_directional',len(ev.data),self.fp(ev))
        cosine,sine=tuple(self.builder(e,axis=a,max_instructions=100) for a in ('cosine','sine'))
        low=np.zeros(8192,complex);low[0]=1
        before=simulation.simulate_statevector(cosine.remove_final_measurements(inplace=False),low.copy())
        expected=simulation.simulate_statevector(sine.remove_final_measurements(inplace=False),low.copy())
        half=len(expected)//2;signal=float(np.vdot(expected[:half],expected[:half]).real-np.vdot(expected[half:],expected[half:]).real)
        def artificial_compile(qc,compiler,actual_circuit_fingerprint):
            compiled=transpile(qc,basis_gates=['rz','sx','x','cx'],optimization_level=1,seed_transpiler=17,num_processes=1)
            counts=compiled.count_ops()
            return SimpleNamespace(transpiled_circuit=compiled,compiler_settings_hash=c.COMPILER_HASH,actual_circuit_fingerprint=actual_circuit_fingerprint,
              rz_count=int(counts['rz']),cx_count=int(counts['cx']),rz_depth=1,cx_depth=1,total_depth=compiled.depth(),circuit_size=len(compiled.data))
        simulate=simulation.simulate_statevector
        for fail in (False,True):
            with self.subTest(fail=fail),tempfile.TemporaryDirectory() as d:
                root=Path(d);out=root/'output';out.mkdir();(root/'ledger').mkdir()
                with (root/'wrapper.qpy').open('wb') as f:qpy.dump(cosine,f)
                np.savez(root/'initial.npz',low=low);np.savez(root/'vectors.npz',low=low,before=before,after=before,independent_expected_high=before)
                c.save(root/'signal.json',dict(raw=dict(imag=signal)))
                inputs={k:dict(path=str(root/name)) for k,name in [('wrapper_qpy','wrapper.qpy'),('initial_npz','initial.npz'),('saved_vectors_npz','vectors.npz'),('saved_signal','signal.json')]}
                cfg=dict(SOURCE='a'*40,plan_sha256='b'*64,pipeline_identity='c'*64,inputs=inputs,output=str(out),ledger=str(root/'ledger'),marker=str(root/'CONSUMED.json'),grant={},
                  sealed_files=[dict(path=str(Path(m.__file__).resolve())) for m in (np,qiskit)],cosine_acceptance=dict(wrapper_fingerprint=self.fp(cosine)))
                c.save(cfg['marker'],dict(state='CONSUMED_NO_RETRY_NO_REFUND',grant={}))
                with (patch.object(c,'authorize'),patch.object(c,'verify_seals'),patch.object(original,'install'),patch.object(costs,'transpile_and_measure_cost',side_effect=artificial_compile) as compiler,
                  patch.object(simulation,'simulate_statevector',side_effect=lambda qc,v:simulate(qc,v)*(np.exp(2e-7j) if fail else 1)) as action):
                    if fail:
                        with self.assertRaisesRegex(ValueError,'NUMERIC_DISAGREEMENT:compiled_vector_sine'):native.run_science(cfg,threading.Event())
                    else:
                        r=native.run_science(cfg,threading.Event());self.assertEqual(r['pilot_axis_receipts'],2)
                    self.assertEqual(compiler.call_count,1);self.assertEqual(action.call_count,1)
                self.assertEqual((out/'cost_sine.json').exists(),not fail)
                self.assertTrue((out/'compiled_sine.qpy').exists());self.assertTrue((out/'compiled_sine_vector.npz').exists())

    def test_invalid_terminal_or_vector_refused(self):
        cosine,sine=self.wrappers(1,.3)
        with self.assertRaises(ValueError):native.sine_wrapper(sine)
        cosine.data[-2]=cosine.data[1]
        with self.assertRaises(ValueError):native.sine_wrapper(cosine)
        for v in (self.np.ones(3,dtype=complex),self.np.ones(4,dtype=float),self.np.full(4,complex('nan'))):
            with self.assertRaises(ValueError):native.expected_sine_from_cosine(v)


if __name__=='__main__':unittest.main()
