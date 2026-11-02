"""Artificial preflight for the live diagnostic glue; never calls load_model()."""
from fractions import Fraction as F
from pathlib import Path
import tempfile
import unittest
import json
import gzip
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate
from qiskit.quantum_info import Operator
from support import HERE, prior_adapter, Timers
from diagnostic import Bits, Sampler, generate_path, logical_resources, build_tail, METHODS
from trotterlib.df_rte_tail import DFBasisRegistry
from trottertracks.algorithm_codesign.g6_f3_contract import Evidence, _unpack

P=dict(root_bits=256,probability_bits=160,eta=F(1,10**12),rho=F(1,10**12),
       angle_tau=F(1,10**12),angle_root_bits=256,angle_terms=64)

class Fixed:
    def __init__(self,values): self.values=iter(values);self.draws=[]
    def bits(self,n):
        value=next(self.values,0);self.draws.append(dict(width=n,value=value));return value

class Technical(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=HERE);self.path=Path(self.temp.name)
        reg=DFBasisRegistry();rows=[]
        for i in range(3):
            d=reg.register(((RYGate(.1*(i+1)),(i%2,)),),num_system_qubits=2)
            rows.append(dict(label=i,component_id='artificial:'+str(i),sign=(-1 if i==1 else 1),
                support=[i%2],basis_id=d.basis_id,basis_hash=d.basis_hash))
        adapter=prior_adapter().DFAdapter(rows,(F(1,5),F(3,10),F(1,2)),reg,2,'ARTIFICIAL')
        self.model=dict(adapter=adapter,p=adapter.p,x=F(1,2),scalar=F(-3,11),labels_id=adapter.labels_id)
    def tearDown(self): self.temp.cleanup()
    def sampler(self,method):return Sampler(method,self.model,'ARTIFICIAL',self.path,precision=P)

    def test_01_bit_mapping_order_independence_and_exact_width(self):
        for width in (1,160,2048):
            a,b=Bits('ordinary',7,2),Bits('ordinary',7,2)
            self.assertEqual([a.bits(width) for _ in range(3)],[b.bits(width) for _ in range(3)])
            self.assertTrue(all(0<=d['value']<1<<width for d in a.draws))
        self.assertNotEqual(Bits('ordinary',7,2).bits(2048),Bits('full_return',7,2).bits(2048))

    def test_02_all_three_paths_no_measurements(self):
        for method in METHODS:
            s=self.sampler(method);row=generate_path(s,0,Timers())
            d=_unpack(row['exact'])
            self.assertEqual(len(d['microsteps']),4)
            self.assertFalse(d['measurement_performed'])
            self.assertIsNone(d['coherent_signal_estimator'])
            rec=Evidence.create(d);self.assertEqual(rec.digest,row['exact_digest'])
            if row['logical']:
                self.assertEqual(row['logical']['workspace_qubits'],1)
                self.assertEqual(row['logical']['child_rotations'],4)
                expected=F(1)
                for step in d['microsteps']:
                    raw=gzip.decompress((self.path/'events'/(step['event_id']+'.json.gz')).read_bytes())
                    event=Evidence(raw.decode(),step['event_id']).read()['event']
                    expected*=event['weight']
                self.assertEqual(expected,d['weight'])

    def test_03_full_both_zero_reasons_and_actual_weights(self):
        s=self.sampler('full_return');width=1<<160
        order_two=int(s.g.order[0]*width)
        raw=s.sample(1,0,Fixed([order_two,0,0]))
        coin=s.sample(1,1,Fixed([0,width-1]))
        event=s.sample(1,2,Fixed([]))
        self.assertEqual(raw['header']['reason'],'RAW_NONREDUCED')
        self.assertEqual(coin['header']['reason'],'COIN_REJECT')
        e=event['event'];self.assertEqual(e['proposal']*e['weight'],e['coefficient'])

    def test_04_raw_reduction_retains_phase(self):
        s=self.sampler('ordinary');e=s.g.event(1,(0,0),1)
        self.assertEqual(e['word'],());self.assertEqual(e['phase_i_power'],2)
        self.assertEqual(e['circuit_time_word'],())

    def test_05_technical_failure_is_terminal(self):
        s=self.sampler('ordinary')
        with self.assertRaises(Exception): s.sample(0,0,Fixed([-1]))
        self.assertIsNotNone(s.failed)
        with self.assertRaises(Exception): s.sample(0,1,Fixed([]))

    def test_06_angle_cache_and_same_full_p5_event(self):
        f,p=self.sampler('full_return'),self.sampler('closed_P5_full')
        ef,ep=f.g.event((),0),p.g.event(0,(),0)
        self.assertEqual(ef['ratio'],ep['ratio'])
        self.assertEqual(f.angle(ef,Timers())[0],p.angle(ep,Timers())[0])
        f.angle(ef,Timers());self.assertEqual(f.angle_hits,1)

    def test_07_counting_is_raw_logical_not_native(self):
        c=QuantumCircuit(3);c.h(0);c.cx(2,0);c.ccx(2,0,1)
        d=logical_resources(c,[dict(word=(0,1))])
        self.assertEqual((d['logical_1Q'],d['logical_2Q'],d['logical_multi_qubit']),(1,1,1))
        self.assertEqual(d['instructions'],3);self.assertEqual(d['native_T_CX'],'UNKNOWN')

    def test_08_live_gate_requires_sealed_authorization_and_marker(self):
        from run import coordinator
        # No authorization file is created by this test; no model read is reached.
        with self.assertRaises(FileNotFoundError): coordinator('ordinary')

    def test_09_zero_still_generates_all_four_steps(self):
        s=self.sampler('full_return');old=s.sample;calls=[];width=1<<160
        start=int(s.g.order[0]*width)
        def fixed(trial,k):
            calls.append(k)
            return old(trial,k,Fixed([start,0,0] if k==0 else []))
        s.sample=fixed
        row=generate_path(s,2,Timers());d=_unpack(row['exact'])
        self.assertEqual(calls,[0,1,2,3]);self.assertEqual(d['weight'],0)
        self.assertIsNone(row['logical']);self.assertEqual(d['kind'],'ZERO')

    def test_10_incomplete_trials_cannot_complete(self):
        from run import summarize
        with self.assertRaises(ValueError): summarize(self.path,[],{})

    def test_11_artificial_fork_batch_storage_roundtrip(self):
        import run
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        s=self.sampler('ordinary');s.abort=mp.get_context('fork').Event()
        old=run.verify_frozen
        try:
            # Dependency guard is separately sealed in the live entrance; no real files opened here.
            run.verify_frozen=lambda all_files=True: None
            run.G=(s,self.path,dict(batch_trials=2,trials_per_method=4))
            with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('fork')) as pool:
                rows=list(pool.map(run.batch,[0,1]))
            self.assertEqual(sum(r['completed'] for r in rows),4)
            seen=[]
            for r in rows:
                for line in gzip.decompress((HERE/r['path']).read_bytes()).splitlines():
                    row=json.loads(line);d=_unpack(row['exact'])
                    self.assertEqual(Evidence.create(d).digest,row['exact_digest'])
                    seen.append(d['trial'])
            self.assertEqual(sorted(seen),[0,1,2,3])
        finally:
            run.verify_frozen=old

    def test_12_whole_controlled_operator_phase_order_scalar(self):
        import math
        s=self.sampler('full_return');q=[]
        for row in self.model['adapter'].rows:
            c=QuantumCircuit(2)
            for gate,qubits in self.model['adapter'].registry.definition(row['basis_id']).runtime_operations:
                c.append(gate,qubits)
            b=Operator(c).data
            diagonal=np.diag([row['sign']*(-1)**sum((k>>i)&1 for i in row['support']) for k in range(4)])
            q.append(b@diagonal@b.conj().T)
        events=[s.g.event((0,2),1),s.g.event((),0),s.g.event((2,0),1),s.g.event((),2)]
        angles=[s.angle(e,Timers())[0] for e in events]
        actual=Operator(build_tail(self.model['adapter'],self.model['scalar'],events,angles)).data
        expected=np.eye(4,dtype=complex)
        for e in events:
            word=np.eye(4,dtype=complex)
            for i in e['raw_word']: word=word@q[i]
            phi=math.atan(float(e['ratio']))
            v=(-1j)**len(e['raw_word'])*(math.cos(phi)*np.eye(4)-1j*math.sin(phi)*q[e['child']])@word
            expected=v@expected
        expected*=np.exp(-1j*float(F(4,5)*self.model['scalar']))
        full=np.eye(8,dtype=complex);full[4:,4:]=expected
        self.assertLess(float(np.linalg.norm(actual-full,'fro')),1e-10)

if __name__=='__main__': unittest.main(verbosity=2)
