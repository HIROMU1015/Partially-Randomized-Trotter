"""Synthetic gates, vectors, one-shot paths and spawned workers. No H6 loads."""
import copy
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from qiskit import QuantumCircuit, qpy
import qiskit
from qiskit.circuit import Gate, Parameter
from qiskit.circuit.library import HGate, PhaseGate, RZGate, RZZGate, CXGate, XXPlusYYGate, SXGate, XGate, IGate

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import h6_m5_phase_contract as c
import h6_m5_phase_engine as e
import h6_m5_phase_adapter as a
import run_h6_m5_saved_phase as r


def synthetic_worker(cfg,job,stop):
    folder=Path(cfg['output'])/job;folder.mkdir()
    c.save(folder/'result.json',dict(job=job,SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],ordinary_reproduced=True))


def synthetic_failure_worker(cfg,job,stop):
    if job=='WRAPPER_ORDINARY':raise ValueError('ARTIFICIAL_WORKER_FAILURE')
    synthetic_worker(cfg,job,stop)


def synthetic_nonreproduction_worker(cfg,job,stop):
    folder=Path(cfg['output'])/job;folder.mkdir()
    c.save(folder/'result.json',dict(job=job,SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],ordinary_reproduced=False))


def toy_inputs(folder,n=3):
    low=np.zeros(2**n,complex);low[:4]=np.array([1,2j,-2,3j])/np.sqrt(18)
    circuits={}
    for kind in ('WRAPPER','COMPILED'):
        qc=QuantumCircuit(n,1);qc.rz(.31,0);qc.cx(0,n-1);qc.rz(-.21,n-1)
        qc.global_phase=.29+(2e-9 if kind=='COMPILED' else 0)
        qc.measure(n-1,0);circuits[kind]=qc
    before=e.ordinary(circuits['WRAPPER'],low);after=e.ordinary(circuits['COMPILED'],low)
    inputs={}
    def row(role,path):inputs[role]=c.seal(path,role)
    for kind,qc in circuits.items():
        path=folder/(kind+'.qpy')
        with path.open('wb') as f:qpy.dump(qc,f)
        row(kind.lower()+'_qpy',path)
        path=folder/(kind+'_receipt.json');c.save(path,dict(global_phase=float(qc.global_phase)))
        row(kind.lower()+'_receipt',path)
    for name,values in [('initial_npz',dict(low=low)),('vectors_npz',dict(low=low,before=before,after=after,independent_expected_high=low[:2**(n-1)]))]:
        path=folder/(name+'.npz');np.savez(path,**values);row(name,path)
    path=folder/'diagnostics.json';d=e.comparison(before,after,expected_hadamard=0.)
    # A tiny artificial compiled common phase exceeds the unchanged raw threshold.
    d=e.comparison(before,after,expected_hadamard=0.);d['status']='DIAGNOSTICS_ONLY_NO_ACCEPTANCE_OVERRIDE'
    c.save(path,d);row('saved_diagnostics',path)
    return dict(inputs=inputs),circuits,low


class EngineTests(unittest.TestCase):
    def test_all_analytic_gates_all_wire_orders(self):
        rng=np.random.default_rng(451);low=rng.normal(size=8)+1j*rng.normal(size=8);low/=np.linalg.norm(low)
        gates=[HGate(),PhaseGate(.137),RZGate(-.413),RZZGate(.237),CXGate(),XXPlusYYGate(-.738,.319),SXGate(),XGate(),IGate()]
        for gate in gates:
            for qs in ((0,),(1,),(2,)) if gate.num_qubits==1 else ((0,1),(1,0),(0,2),(2,0),(1,2),(2,1)):
                with self.subTest(gate=gate.name,qs=qs):
                    qc=QuantumCircuit(3,1);qc.append(gate,qs);qc.global_phase=.197;qc.measure(2,0)
                    actual=e.extended(qc,low)
                    self.assertEqual(actual.dtype,np.dtype(np.clongdouble))
                    self.assertLess(np.linalg.norm(actual-e.ordinary(qc,low)),2e-15)

    def test_gate_inverses_and_global_phase_once(self):
        low=np.array([1.,2j,3.,-4j],complex)/np.sqrt(30)
        qc=QuantumCircuit(2,1)
        for g in (RZGate(.71),PhaseGate(-.22)):
            qc.append(g,[0]);qc.append(g.inverse(),[0])
        g=XXPlusYYGate(.913,-.471);qc.append(g,[1,0]);qc.append(g.inverse(),[1,0])
        qc.global_phase=.357;qc.measure(1,0)
        self.assertLess(e.norm(e.extended(qc,low)-e.cis(float(qc.global_phase))*low.astype(np.clongdouble)),1e-18)

    def test_extended_does_not_use_binary64_matrix_fallback(self):
        qc=QuantumCircuit(2,1);qc.append(XXPlusYYGate(.31,.21),[0,1]);qc.measure(1,0)
        with patch.object(XXPlusYYGate,'to_matrix',side_effect=AssertionError('binary64 fallback')):
            out=e.extended(qc,np.array([0,1,0,0],complex))
        self.assertTrue(np.isfinite(out).all())

    def test_mantissa_and_actual_extra_precision(self):
        p=e.precision();self.assertGreater(p['extended_mantissa_bits'],p['binary64_mantissa_bits'])
        self.assertNotEqual(e.cis(np.longdouble(1)),np.clongdouble(np.exp(1j)))

    def test_precision_fallback_refused(self):
        original=np.finfo
        with patch.object(e.np,'finfo',side_effect=lambda dtype:original(np.float64)):
            with self.assertRaisesRegex(ValueError,'NOT_WIDER'):e.precision()

    def test_parameters_unknown_nonfinite_and_nonbinary64(self):
        for value in (float('inf'),float('nan'),True,1j,Parameter('p')):
            with self.subTest(value=str(value)),self.assertRaises(ValueError):e.parameter(value)
        with self.assertRaisesRegex(ValueError,'BINARY64_ONLY'):e.parameter(np.longdouble(1)+np.finfo(np.longdouble).eps)
        with self.assertRaisesRegex(ValueError,'UNKNOWN_ANALYTIC'):e.kernel('unknown',())

    def test_cancelled_replay_refuses(self):
        qc=QuantumCircuit(2,1);qc.h(0);qc.measure(1,0)
        with self.assertRaisesRegex(ValueError,'OWN_REPLAY_SIGNAL'):e.extended(qc,np.array([1,0,0,0],complex),cancelled=lambda:True)

    def test_schema_good_and_bad(self):
        qc=QuantumCircuit(3,1);qc.h(2);qc.rzz(.3,0,2);qc.measure(2,0)
        self.assertEqual(e.validate_circuit(qc,'WRAPPER',3)['ancilla'],2)
        bad=[]
        q=QuantumCircuit(3,1);q.x(0);q.measure(2,0);bad.append(q)
        q=QuantumCircuit(3,1);q.append(Gate('h',1,[]),[0]);q.measure(2,0);bad.append(q)
        q=QuantumCircuit(3,1);q.rz(Parameter('x'),0);q.measure(2,0);bad.append(q)
        q=QuantumCircuit(3,1);q.measure(2,0);q.h(0);bad.append(q)
        q=QuantumCircuit(3,1);q.h(0);q.measure(0,0);bad.append(q)
        q=QuantumCircuit(3,1);q.h(0);bad.append(q)
        q=qc.copy();q._layout=object();bad.append(q)
        for q in bad:
            with self.subTest(data=str(q.data)),self.assertRaises(ValueError):e.validate_circuit(q,'WRAPPER',3)

    def test_phase_relative_phase_and_norm_are_diagnostic_only(self):
        v=np.array([1,0,1,0],complex)/np.sqrt(2);snapshot=v.copy()
        phase=e.comparison(v,np.exp(2e-9j)*v,extended_metrics=True)
        self.assertFalse(phase['raw_gate_PASS']);self.assertLess(phase['phase_aligned_error_DIAGNOSTIC_ONLY'],1e-16)
        rel=v.copy();rel[2]*=np.exp(2e-4j)
        self.assertGreater(e.comparison(v,rel)['phase_aligned_error_DIAGNOSTIC_ONLY'],1e-5)
        norm=e.comparison(v,1.01*v);self.assertGreater(norm['raw_error'],.009)
        self.assertFalse(phase['acceptance_override']);np.testing.assert_array_equal(v,snapshot)

    def test_nonfinite_vectors_and_zero_overlap(self):
        with self.assertRaisesRegex(ValueError,'FINITE_VECTOR'):e.comparison(np.array([np.nan,0]),np.zeros(2))
        self.assertIsNone(e.comparison(np.array([1,0]),np.array([0,1]))['common_phase'])


class AdapterTests(unittest.TestCase):
    def test_qpy_roundtrip_and_saved_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg,qcs,low=toy_inputs(Path(tmp))
            scope=dict(c.scope(),nqubits=3)
            with patch.object(c,'scope',return_value=scope):
                self.assertEqual(a.saved_gate(cfg)['status'],'REPRODUCED')
                for kind in qcs:
                    self.assertEqual(a.circuit_gate(cfg,kind)['global_phase'],float(qcs[kind].global_phase))
                    restored=a.load_circuit(cfg,kind)
                    np.testing.assert_array_equal(e.ordinary(restored,low),e.ordinary(qcs[kind],low))

    def test_duplicate_low_and_metric_tamper_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg,_,_=toy_inputs(Path(tmp));scope=dict(c.scope(),nqubits=3)
            p=Path(cfg['inputs']['saved_diagnostics']['path']);d=c.read(p);d['raw_error']+=.1;p.write_text(json.dumps(d))
            with patch.object(c,'scope',return_value=scope),self.assertRaisesRegex(ValueError,'SAVED_METRIC_REPRODUCTION'):a.saved_gate(cfg)
            np.savez(cfg['inputs']['initial_npz']['path'],low=np.zeros(8,complex))
            with patch.object(c,'scope',return_value=scope),self.assertRaisesRegex(ValueError,'DUPLICATED_LOW'):a.load_vectors(cfg)

    def test_replay_result_publication_is_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'v.npz';receipt=a.save_vector(p,np.ones(4,np.clongdouble))
            self.assertEqual(c.sha(p),receipt['sha256'])
            with self.assertRaises(FileExistsError):a.save_vector(p,np.zeros(4))


class ContractTests(unittest.TestCase):
    def test_grant_cannot_expand_scope_or_reuse_an_old_lane(self):
        grant=dict(valid=True,approval_reference='ARTIFICIAL TEST ONLY',grant_id='ARTIFICIAL',scope_id=c.SCOPE,
            SOURCE='a'*40,plan_sha256='b'*64,maximum_replays=4,maximum_saved_initial_states=1,
            new_compile=0,new_sampling=0,new_state_generation=0,accounting=False,retry=False,
            acceptance_override=False,next_phase_authorized=False)
        cfg=dict(authorized=True,SOURCE='a'*40,plan_sha256='b'*64,grant=grant)
        for key,value in [('scope_id','OLD_COMPILE_LANE'),('maximum_replays',5),('maximum_saved_initial_states',2),
            ('new_compile',1),('new_sampling',1),('new_state_generation',1),('accounting',True),
            ('retry',True),('acceptance_override',True),('next_phase_authorized',True)]:
            bad=copy.deepcopy(cfg);bad['grant'][key]=value
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'GRANT_EXACT_SCOPE'):c.authorize(bad)

    def test_scope_no_compile_or_new_state_or_acceptance(self):
        s=c.scope();self.assertEqual(s['maximum_work']['statevector_replays'],4)
        for key in ('new_compile','new_sampling','new_SCF_DF_state_generation','shot_cost_reaccounting'):
            self.assertEqual(s['maximum_work'][key],0)
        self.assertFalse(s['acceptance_override']);self.assertFalse(s['next_phase_authorized'])

    def test_draft_refuses_before_input_or_science(self):
        with patch.object(c,'verify_seals',side_effect=AssertionError('late')):
            with self.assertRaisesRegex(ValueError,'EXPLICIT_SAVED_NUMERICAL'):c.authorize({'authorized':False,'grant':{'valid':False}})
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'draft.json';p.write_text(json.dumps(dict(authorized=False,grant={'valid':False})))
            result=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/run_h6_m5_saved_phase.py'),'--execute','--config',str(p)],capture_output=True,text=True)
            self.assertEqual(result.returncode,2);self.assertIn('EXPLICIT_SAVED_NUMERICAL',result.stdout)
            self.assertEqual(list(Path(tmp).iterdir()),[p])

    def test_scheduler_stops_new_dispatch_and_prioritizes_ordinary(self):
        self.assertEqual(r.decisions([],[],None,1),['WRAPPER_ORDINARY'])
        self.assertEqual(r.decisions(['WRAPPER_ORDINARY'],[],'failed',4),[])
        self.assertEqual(r.decisions([],c.JOBS[:2],None,2),list(c.JOBS[2:]))

    def test_exclusive_ledger_and_no_unregistered_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=dict(ledger=tmp,marker=str(Path(tmp)/'marker'),grant={'grant_id':'new'},SOURCE='a'*40,plan_sha256='b'*64)
            c.save(cfg['marker'],dict(state='CONSUMED_NO_RETRY_NO_REFUND',grant=cfg['grant']))
            c.reserve(cfg,c.JOBS[0])
            with self.assertRaises(FileExistsError):c.reserve(cfg,c.JOBS[0])
            with self.assertRaisesRegex(ValueError,'REGISTERED'):c.reserve(cfg,'SINE')

    def test_no_transpile_sampling_or_synthesis_import(self):
        import ast
        for rel in c.SOURCE_PATHS[:5]:
            tree=ast.parse((ROOT/rel).read_text())
            for node in ast.walk(tree):
                if isinstance(node,ast.Call):
                    name=getattr(node.func,'id',getattr(node.func,'attr',''))
                    self.assertNotIn(name,('transpile','decompose','to_matrix','sample','normalize','prepare_pyscf_system'))


class ProcessTests(unittest.TestCase):
    def test_external_wait4_observes_native_exit_without_real_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);child=root/'artificial_child'
            child.write_text('#!/bin/sh\nexit 7\n');child.chmod(0o700)
            paths=dict(entered=str(root/'entered.json'),log=str(root/'native.log'),exit=str(root/'exit.json'))
            cfg=dict(SOURCE='a'*40,plan_sha256='b'*64,grant={'grant_id':'ARTIFICIAL'},
                run_root=str(root/'unused'),supervisor=paths,runtime={'python':str(child)})
            path=root/'config.json';c.save(path,cfg)
            with patch.object(c,'authorize'):
                self.assertEqual(r.supervise(cfg,path),7)
                with self.assertRaisesRegex(ValueError,'UNUSED_SUPERVISOR'):r.supervise(cfg,path)
            receipt=c.read(paths['exit'])
            self.assertEqual(receipt['native_exit'],7);self.assertTrue(receipt['reaped'])
            self.assertFalse(receipt['next_phase_started']);self.assertFalse((root/'unused').exists())

    def test_real_adapter_engine_report_on_synthetic_13_qubits(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent=Path(tmp);cfg,_,_=toy_inputs(parent,13);root=parent/'run'
            cfg.update(run_root=str(root),output=str(root/'output'),ledger=str(root/'ledger'),marker=str(root/'marker'),
                authorized=True,grant={'valid':True},SOURCE='a'*40,plan_sha256='b'*64,runtime={'python':sys.executable},
                resources={'per_worker_RSS_proxy_bytes':1},
                sealed_files=[dict(path=str(Path(m.__file__).resolve())) for m in (np,qiskit)])
            # Only synthetic authorization/seals/resources substituted. All array,
            # schema, replay, publication and final diagnostic gates run normally.
            with patch.object(c,'authorize'),patch.object(c,'verify_seals'),patch.object(r,'fresh_resources',return_value={'workers':4}):
                self.assertEqual(r.execute(cfg),0)
            report=c.read(root/'diagnostic_report.json');w=c.read(root/'completion_witness.json')
            self.assertFalse(report['ordinary']['raw_gate_PASS']);self.assertFalse(report['extended']['raw_gate_PASS'])
            self.assertTrue(report['old_STOP_remains_STOP']);self.assertTrue(w['all_reaped'])
            self.assertEqual(w['replay_entries'],4);self.assertEqual(w['accepted_new_cost_axes'],0)
            self.assertEqual(len(report['result_sha256']),4)

    def artificial_run(self,worker,capacity=4):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'fresh'
            cfg=dict(run_root=str(root),output=str(root/'output'),ledger=str(root/'ledger'),marker=str(root/'marker'),
                authorized=True,grant={'valid':True},SOURCE='a'*40,plan_sha256='b'*64,runtime={'python':sys.executable},
                resources={'per_worker_RSS_proxy_bytes':1},
                sealed_files=[dict(path=str(Path(m.__file__).resolve())) for m in (np,qiskit)])
            with patch.object(c,'authorize'),patch.object(c,'verify_seals'),patch.object(c,'consumed'),\
                patch.object(r,'fresh_resources',return_value={'workers':capacity}),\
                patch.object(a,'saved_gate',return_value={'status':'REPRODUCED'}),\
                patch.object(a,'circuit_gate',return_value={}),\
                patch.object(a,'report',return_value={'status':'ARTIFICIAL_ONLY'}):
                code=r.execute(cfg,worker=worker)
                with self.assertRaisesRegex(ValueError,'UNUSED_RUN_ROOT'):r.execute(cfg,worker=worker)
            return code,c.read(root/'completion_witness.json')

    def test_four_spawn_workers_exit_and_reap(self):
        code,w=self.artificial_run(synthetic_worker)
        self.assertEqual(code,0);self.assertEqual(w['replay_entries'],4)
        self.assertEqual(len(w['native_workers']),4);self.assertTrue(w['all_reaped'])
        self.assertFalse(w['next_phase_started']);self.assertEqual(w['new_compile'],0)

    def test_failed_process_stops_dispatch_reaps_and_never_retries(self):
        code,w=self.artificial_run(synthetic_failure_worker,1)
        self.assertEqual(code,1);self.assertEqual(w['replay_entries'],1)
        self.assertTrue(w['all_reaped']);self.assertFalse(w['retry'])

    def test_nonreproduction_stops_new_extended_jobs(self):
        code,w=self.artificial_run(synthetic_nonreproduction_worker,1)
        self.assertEqual(code,1);self.assertEqual(w['replay_entries'],1)
        self.assertIn('DID_NOT_REPRODUCE',w['reason'])


if __name__=='__main__':unittest.main()
