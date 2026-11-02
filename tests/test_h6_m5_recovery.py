"""Artificial vectors/circuits and isolated child processes; no molecular input."""
import ast
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import h6_m5_recovery_contract as c
import h6_m5_recovery_native as n
import launch_h6_m5_recovery as launcher


def artificial_child(cfg, stop):
    taskroot=Path(cfg['output'])/c.TASK['id'];taskroot.mkdir()
    for axis in ('cosine','sine'):
        c.save(Path(cfg['ledger'])/(axis+'.json'),{})
        c.save(taskroot/('cost_'+axis+'.json'),{})
    result=dict(status='PASS',
        lane=c.LANE,task=c.TASK['id'],SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],
        checks=[dict(error=0.) for _ in range(8)])
    c.save(Path(cfg['run_root'])/'worker_return_PRIVATE.json',result)
    c.save(taskroot/'result.json',result)


def artificial_failed_child(cfg, stop):
    c.save(Path(cfg['ledger'])/'cosine.json',{})
    c.save(Path(cfg['run_root'])/'worker_return_PRIVATE.json',dict(status='STOP',reason='ARTIFICIAL_STOP'))


class ContractTests(unittest.TestCase):
    def test_scope_exactly_one_full_point(self):
        s=c.scope();self.assertEqual(len(s['tasks']),1)
        g=s['tasks'][0]['group']
        self.assertEqual((g['system'],g['method'],g['prefix'],g['q'],g['formula']),('H6','FULL',19,2,'4th(m5_best)'))
        self.assertEqual(s['compile_axes'],2);self.assertEqual(s['PR_trajectories'],0)
        self.assertEqual(s['full_H_reference_jobs'],0);self.assertEqual(s['numeric_tolerance'],1e-9)

    def test_draft_refuses_before_seals(self):
        with patch.object(c,'verify_seals',side_effect=AssertionError('too late')):
            with self.assertRaisesRegex(ValueError,'EXPLICIT_FIXED_POINT'):
                c.authorize(dict(authorized=False,grant={'valid':False}))

    def test_cli_draft_refusal_no_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'draft.json';path.write_text(json.dumps(dict(authorized=False,grant={'valid':False})))
            p=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/launch_h6_m5_recovery.py'),'--execute','--config',str(path)],capture_output=True,text=True)
            self.assertEqual(p.returncode,2);self.assertIn('EXPLICIT_FIXED_POINT',p.stdout)
            self.assertEqual(list(Path(tmp).iterdir()),[path])

    def test_old_scientific_statements_unchanged_after_removing_hooks(self):
        old=ast.parse((ROOT/'scripts/h46_m5_native.py').read_text())
        old_fn=next(x for x in old.body if isinstance(x,ast.FunctionDef) and x.name=='_run_task')
        class Strip(ast.NodeTransformer):
            def visit_Expr(self,node):
                if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id in {'capture_wrapper','capture_compiled','capture_vectors'}:return None
                return self.generic_visit(node)
            def visit_Subscript(self,node):
                if isinstance(node.value,ast.Name) and node.value.id=='config' and isinstance(node.slice,ast.Constant) and node.slice.value=='origin_SOURCE':node.slice.value='SOURCE'
                return self.generic_visit(node)
        new=Strip().visit(n.derive_task_ast()).body[0]
        self.assertEqual(ast.dump(old_fn,include_attributes=False),ast.dump(new,include_attributes=False))

    def test_sealed_old_native_bytes_match_original(self):
        for rel in c.base.SOURCE_PATHS:
            self.assertEqual((ROOT/rel).read_bytes(),subprocess.check_output(['git','show',c.ORIGIN_SOURCE+':'+rel],cwd=ROOT))

    def test_sine_cannot_reserve_before_cosine_acceptance(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=dict(output=tmp,ledger=tmp,SOURCE='a'*40,plan_sha256='b'*64)
            with self.assertRaisesRegex(ValueError,'COSINE_PASS'):
                c.reserve(cfg,c.TASK,'sine','toy')

    def test_atomic_binary_exclusive_and_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'data';n.atomic_binary(p,lambda f:f.write(b'original'))
            with self.assertRaises(FileExistsError):n.atomic_binary(p,lambda f:f.write(b'overwrite'))
            self.assertEqual(p.read_bytes(),b'original')
            other=Path(tmp)/'failed'
            def fail(f):f.write(b'partial');raise OSError('ARTIFICIAL_DISK_FAILURE')
            with self.assertRaises(OSError):n.atomic_binary(other,fail)
            self.assertFalse(other.exists())


class ProcessTests(unittest.TestCase):
    def execute_artificial(self, worker):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'new'
            cfg=dict(run_root=str(root),output=str(root/'output'),ledger=str(root/'ledger'),marker=str(root/'marker'),
                grant={'valid':True},SOURCE='a'*40,origin_SOURCE=c.ORIGIN_SOURCE,plan_sha256='b'*64,
                runtime={'python':sys.executable},RSS_proxy_bytes=1,disk_free_planning_proxy_bytes=1)
            with patch.object(c,'authorize'),patch.object(c,'verify_seals'),patch.object(c,'consumed'),patch.object(launcher,'fresh_resources',return_value={'workers':1}):
                code=launcher.execute(cfg,worker=worker)
            return code,c.read(root/'completion_witness.json')

    def test_spawn_success_and_reap(self):
        code,w=self.execute_artificial(artificial_child)
        self.assertEqual(code,0);self.assertEqual(w['status'],'COMPLETE_MANDATORY_STOP')
        self.assertTrue(w['worker_reaped']);self.assertEqual(w['worker_exit'],0)
        self.assertFalse(w['next_phase_started'])

    def test_spawn_stop_and_reap(self):
        code,w=self.execute_artificial(artificial_failed_child)
        self.assertEqual(code,1);self.assertEqual(w['compile_entries'],1)
        self.assertEqual(w['accepted_cost_axes'],0);self.assertTrue(w['worker_reaped'])
        self.assertFalse(w['retry'])


@unittest.skipUnless(os.environ.get('M5_TEST_PROVIDER'),'synthetic-only provider must be explicit')
class ArtificialNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0,str(Path(os.environ['M5_TEST_PROVIDER'])/'src'))

    def artificial_task(self, failure=None):
        import numpy as np
        from qiskit import QuantumCircuit
        from trotterlib.df_trotter.circuit import simulate_statevector
        import trotterlib.df_trotter.circuit as simulator
        import trotterlib.rte_compiled_cost as costs
        from trottertracks.resource_applicability.ax2b_native_df_v5 import NativeEvolution
        from trottertracks.resource_applicability.ax2b_stream_fingerprint_v5 import canonical_qiskit_circuit_fingerprint as fp
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'out';out.mkdir();ledger=root/'ledger';ledger.mkdir();ref=root/'ref';ref.mkdir()
            (ref/'reference.npz').write_bytes(b'artificial opaque reference bytes')
            c.save(ref/'receipt.json',dict(status='PASS',SOURCE=c.ORIGIN_SOURCE,snapshot='toy',full_preparation='prep',cut_preparation='prep',reference_sha256=c.sha(ref/'reference.npz')))
            prep=SimpleNamespace(preparation_hash='prep')
            oracle=SimpleNamespace(initial=np.array([1.,0.],complex),qindices=np.array([0,1]),specs=[(np.zeros(2),np.eye(2),np.eye(2)) for _ in range(20)])
            context=SimpleNamespace(np=np,qstate=np.array([1.,0.],complex),oracle=oracle,fullprep=prep,cut=prep,scalar=0.,tail_identity=0.)
            cfg=dict(lane=c.LANE,SOURCE='a'*40,origin_SOURCE=c.ORIGIN_SOURCE,plan_sha256='b'*64,
                binding={'snapshot':'toy'},output=str(out),ledger=str(ledger),reference_root=str(ref))
            def compose(*args):
                qc=QuantumCircuit(2)
                qc.cp(.123,1,0)  # Nontrivial toy action outside the initial |0> system state.
                return NativeEvolution(qc,1,.8,2,'4th(m5_best)','symmetric_directional',1,fp(qc))
            calls=[]
            actual_transpile=costs.transpile_and_measure_cost
            def fake_transpile(wrapper,compiler,actual_circuit_fingerprint):
                calls.append('cosine' if not calls else 'sine');qc=wrapper.copy()
                if failure=='real_compile':return actual_transpile(wrapper,compiler,actual_circuit_fingerprint=actual_circuit_fingerprint)
                if failure==calls[-1]:qc.global_phase+=2e-9
                if failure=='nonfinite':qc.name='ARTIFICIAL_NONFINITE'
                return SimpleNamespace(transpiled_circuit=qc,actual_circuit_fingerprint=actual_circuit_fingerprint,
                    compiler_settings_hash=c.COMPILER_HASH,rz_count=1,cx_count=0,rz_depth=1,cx_depth=0,total_depth=2,circuit_size=2)
            def simulate(qc,state):
                if qc.name=='ARTIFICIAL_NONFINITE':return np.full_like(state,np.nan)
                return simulate_statevector(qc,state)
            with (patch.dict(n._namespace,context=lambda cfg:context,compose=compose,signal=lambda ctx,g:dict(log_B=0.,kind='ARTIFICIAL')),
                 patch.object(c,'verify_seals'),patch.object(costs,'transpile_and_measure_cost',side_effect=fake_transpile),patch.object(simulator,'simulate_statevector',side_effect=simulate)):
                if failure=='disk':
                    with patch.object(n,'atomic_binary',side_effect=OSError('ARTIFICIAL_DISK_FAILURE')):
                        result=n._run_task(cfg,c.TASK,threading.Event(),fp)
                else:result=n._run_task(cfg,c.TASK,threading.Event(),fp)
            taskout=out/c.TASK['id']
            diagnostics={p.parent.name:c.read(p) for p in taskout.glob('quarantine/*/diagnostics.json')}
            qpy_files=list(taskout.glob('quarantine/*/*.qpy'))
            if qpy_files:
                from qiskit import qpy
                for p in qpy_files:
                    with p.open('rb') as f:self.assertEqual(len(qpy.load(f)),1)
            return dict(result=result,calls=calls,ledger=len(list(ledger.glob('*.json'))),
                accepted=len(list(taskout.glob('cost_*.json'))),diagnostics=diagnostics,qpy=len(qpy_files),
                vector_files=len(list(taskout.glob('quarantine/*/vectors.npz'))))

    def test_paired_success_keeps_evidence(self):
        v=self.artificial_task();self.assertEqual(v['result']['status'],'PASS',v['result'])
        self.assertEqual((v['ledger'],v['accepted'],v['qpy'],v['vector_files']),(2,2,4,2))
        self.assertEqual(len(v['result']['checks']),8)

    def test_real_toy_transpile_and_qpy_paired_path(self):
        v=self.artificial_task('real_compile')
        self.assertEqual(v['result']['status'],'PASS',v['result'])
        self.assertEqual((v['accepted'],v['qpy'],v['vector_files']),(2,4,2))

    def test_phase_only_raw_failure_does_not_accept_or_run_sine(self):
        v=self.artificial_task('cosine');self.assertEqual(v['result']['status'],'STOP')
        self.assertEqual(v['calls'],['cosine']);self.assertEqual((v['ledger'],v['accepted']),(1,0))
        d=v['diagnostics']['cosine'];self.assertGreaterEqual(d['raw_error'],1e-9)
        self.assertLess(d['phase_aligned_error_DIAGNOSTIC_ONLY'],1e-12)
        self.assertFalse(d['acceptance_override']);self.assertEqual(v['qpy'],2)

    def test_sine_failure_preserves_cosine_acceptance_only(self):
        v=self.artificial_task('sine');self.assertEqual(v['result']['status'],'STOP')
        self.assertEqual((v['ledger'],v['accepted']),(2,1))

    def test_disk_failure_no_compiler_or_accepted_cost(self):
        v=self.artificial_task('disk');self.assertEqual(v['result']['status'],'STOP')
        self.assertEqual(v['calls'],[]);self.assertEqual(v['accepted'],0)

    def test_nonfinite_persists_and_stops(self):
        v=self.artificial_task('nonfinite');self.assertEqual(v['result']['status'],'STOP')
        self.assertEqual(v['accepted'],0);self.assertEqual(v['vector_files'],1)
        self.assertEqual(v['diagnostics']['cosine']['status'],'STOP_NONFINITE_VECTOR')


if __name__=='__main__':unittest.main()
