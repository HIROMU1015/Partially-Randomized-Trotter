"""Deterministic artificial fixtures only. No molecular data, RNG trials or shots."""
from fractions import Fraction as F
import importlib.util
import math
from math import factorial
import unittest
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate, RZGate, XXPlusYYGate
from qiskit.quantum_info import Operator, Statevector
from support import PRIOR, source_rows
from family import Family, METHODS, POLICY, require, compare_contracts
from whole_path import WholePath, AllTrials
from native_receipt import key, check, synthetic_receipt, path_price, PRICE_FIELDS, ideal_event_id
from trottertracks.algorithm_codesign.g6_f3_contract import Evidence
from trottertracks.algorithm_codesign.g7_reference import reference_events
from trotterlib.df_rte_tail import DFBasisRegistry

spec = importlib.util.spec_from_file_location('prior_adapter_readonly', PRIOR / 'adapter.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)

class Bits:
    def __init__(self, values=()):
        self.values = iter(values)
    def bits(self, width):
        return next(self.values, 0)

def reference(event, q, sigma=1):
    out = np.eye(4, dtype=complex)
    for i in event['raw_word']:
        out = out @ q[i]
    phi = math.atan(float(event['ratio']))
    return (-1j * sigma)**len(event['raw_word']) * (
        math.cos(phi)*np.eye(4) - 1j*sigma*math.sin(phi)*q[event['child']]) @ out

def controlled(active):
    out = np.eye(8, dtype=complex)
    out[4:, 4:] = active
    return out

class Synthetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p, cls.x = (F(1,5), F(3,10), F(1,2)), F(1,2)
        registry, rows, q = DFBasisRegistry(), [], []
        sequences = [((RYGate(.37),(0,)), (RZGate(.29),(0,))),
                     ((XXPlusYYGate(.61,.23),(0,1)), (RZGate(-.43),(1,))),
                     ((RYGate(-.81),(1,)), (XXPlusYYGate(.42,-.31),(0,1)))]
        for i, ops in enumerate(sequences):
            d = registry.register(ops, num_system_qubits=2)
            c = QuantumCircuit(2)
            for gate, qubits in ops:
                c.append(gate, qubits)
            b = Operator(c).data
            support, sign = ([0,1] if i==2 else [i%2]), (-1 if i==1 else 1)
            diag = np.diag([sign*(-1)**sum((k>>j)&1 for j in support) for k in range(4)])
            q.append(b @ diag @ b.conj().T)
            rows.append(dict(label=i, component_id='synthetic:'+str(i), sign=sign, support=support,
                             basis_id=d.basis_id, basis_hash=d.basis_hash))
        cls.adapter = prior.DFAdapter(rows, cls.p, registry, 2, 'synthetic:c1-45')
        cls.q = q
        cls.families = {method: cls.family(method) for method in METHODS}
        cls.path = WholePath([cls.families['full_return']]*2, schedule_id='synthetic:S2-two-step',
                            independence_assumption='fresh IID bits across steps/trials; independent measurement')

    @classmethod
    def family(cls, method='full_return', sigma=1):
        return Family(method, cls.p, cls.x, cls.adapter.model_id, cls.adapter.labels_id,
                      sigma=sigma, scope='SYNTHETIC_ONLY')

    def path_record(self, tid, zero=None, prefix=None):
        f = self.families['full_return']; width = 1 << POLICY['probability_bits']
        start_two = int(f.g.order[0] * width)
        first = f.sample(Bits([start_two,0,0] if zero=='raw' else [0,width-1] if zero=='coin' else []), tid+':0')
        second = f.sample(Bits(), tid+':1')
        return self.path.assemble(tid, [first,second], presampled=prefix is None, spent_prefix=prefix)

    def test_01_full_denominator_both_zeros(self):
        a = AllTrials(self.path, ['a','r','c'], 'X')
        e = self.path_record('a')
        a.add(e,-1); a.add(self.path_record('r','raw')); a.add(self.path_record('c','coin'))
        d = a.evidence().read()
        self.assertEqual(d['valid_zeros'],2)
        self.assertEqual(d['full_trial_mean'],-e.read()['weight']/3)
        self.assertTrue(d['complete'])

    def test_02_failure_not_zero_and_terminal(self):
        a = AllTrials(self.path,['a','b'],'X'); a.add(self.path_record('a'),1)
        a.fail('provider technical failure')
        self.assertIsNone(a.evidence().read()['full_trial_mean'])
        with self.assertRaises(ValueError): a.add(self.path_record('b','coin'))
        self.assertEqual(a.evidence().read()['completed'],1)

    def test_03_actual_weight_and_proposal_product(self):
        d = self.path_record('p').read(); first,second = d['microsteps']
        self.assertEqual(d['weight'],first['event']['weight']*second['event']['weight'])
        self.assertEqual(d['proposal'],first['event']['proposal']*second['event']['proposal'])

    def test_04_phase_order_scalar_halves_and_XY(self):
        width = 1 << POLICY['probability_bits']; errors=[]
        state = np.array([1,2j,-3,.7j],complex); state /= np.linalg.norm(state)
        for sigma in (1,-1):
            f = self.family(sigma=sigma)
            pth = WholePath([f,f],schedule_id='synthetic:halves',independence_assumption='ASSUMED_IID')
            start_two=int(f.g.order[0]*width); start_label1=int(f.g.labels[0]*width)
            r0=f.sample(Bits([start_two,0,start_label1,0,width-1]),'o:0')
            r1=f.sample(Bits(),'o:1')
            r=pth.assemble('o',[r0,r1])
            before=QuantumCircuit(3); before.cry(.31,2,0)
            after=QuantumCircuit(3); after.crz(-.47,2,1)
            phase=F(-6,77)
            c=pth.circuit(r,[self.adapter]*2,before=before,after=after,scalar_phase=phase)
            active=Operator(after).data[4:,4:] @ reference(r1.read()['event'],self.q,sigma) @ \
                   reference(r0.read()['event'],self.q,sigma) @ Operator(before).data[4:,4:]
            active *= np.exp(-1j*float(phase))
            errors.append(float(np.linalg.norm(Operator(c).data-controlled(active),'fro')))
            self.assertLess(errors[-1],1e-10)
            signal=np.vdot(state,active@state)
            for axis,expected in [('X',signal.real),('Y',signal.imag)]:
                v=Statevector(np.r_[state,np.zeros(4)]).evolve(prior.hadamard(c,2,axis)).data
                observed=np.sum(abs(v[:4])**2)-np.sum(abs(v[4:])**2)
                self.assertAlmostEqual(float(observed),float(expected),places=10)
        self.assertTrue(errors)

    def test_05_XY_signed_accumulators(self):
        r=self.path_record('one'); w=r.read()['weight']
        for axis,sign in [('X',1),('Y',-1)]:
            a=AllTrials(self.path,['one'],'Y' if axis=='Y' else 'X');a.add(r,sign)
            self.assertEqual(a.evidence().read()['full_trial_mean'],sign*w)

    def test_06_identity_and_digest_tampering(self):
        r=self.path_record('id'); d=r.read(); d['context_id']='corrupt'
        with self.assertRaises(ValueError): self.path.validate(Evidence.create(d))
        with self.assertRaises(Exception): Evidence(r.document+' ',r.digest).read()
        f=self.family('ordinary'); receipt=f.sample(Bits(),'z')
        for field in ('source_id','policy_id','target_id'):
            d=receipt.read(); d['context'][field]='corrupt'
            with self.assertRaises(ValueError): f.validate(Evidence.create(d))

    def test_07_incomplete_duplicate_and_midfailure(self):
        a=AllTrials(self.path,['a','b'],'X');a.add(self.path_record('a'),1)
        self.assertFalse(a.evidence().read()['complete'])
        self.assertIsNone(a.evidence().read()['full_trial_mean'])
        with self.assertRaises(ValueError): a.add(self.path_record('a'),1)
        self.assertIsNotNone(a.evidence().read()['failure'])

    def test_08_conditioning_and_formal_queries_rejected(self):
        f=self.families['full_return']; e=f.formal(f.g.event((),0),'q:0')
        with self.assertRaises(ValueError): self.path.assemble('q',[e,f.sample(Bits(),'q:1')])
        d=self.path_record('q').read();d['conditioning']='ACCEPTED_ONLY'
        with self.assertRaises(ValueError): AllTrials(self.path,['q'],'X').add(Evidence.create(d),1)

    def test_09_common_target_different_laws(self):
        full=self.families['full_return']; p5=self.families['closed_P5_full']
        d=compare_contracts(full,p5)
        self.assertTrue(d['common_target'] and d['full_p5_same_ideal_accepted_ensemble'])
        self.assertFalse(d['identical_digital_law'])
        wrong=self.family('ordinary'); wrong.target['x']=F(1,3)
        with self.assertRaises(ValueError): wrong.sample(Bits(),'bad')

    def test_10_provider_fairness_and_unknowns(self):
        spec={k:'synthetic:'+k for k in ('source','model','basis','labels','angle','precision','profile',
                'control_phase','actual_adjoint','scalar_policy','event')}
        r=synthetic_receipt(spec)
        prices=[check(r,spec)['price'] for method in METHODS]
        self.assertTrue(all(p==prices[0] for p in prices))
        with self.assertRaises(ValueError): check(r,spec,real=True)
        with self.assertRaises(ValueError): check(r,spec,require_full_support=True)
        for field,value in [('phase_metric','UP_TO_PHASE'),('coverage','UNKNOWN'),('actual_adjoint',False)]:
            d=r.read();d[field]=value
            with self.assertRaises(ValueError): check(Evidence.create(d),spec)
        for field in PRICE_FIELDS:
            d=r.read();d['price'][field]=None
            with self.assertRaises(ValueError): check(Evidence.create(d),spec)
        wrong=dict(spec,profile='different')
        with self.assertRaises(ValueError): check(r,wrong)
        out=path_price([r,r],[spec,spec],prep_readout={k:F(2) for k in PRICE_FIELDS})
        self.assertEqual(out['price']['T'],4)
        self.assertEqual(out['price']['workspace'],5)
        full=self.families['full_return'].g.event((),0)
        p5=self.families['closed_P5_full'].g.event(0,(),0)
        self.assertEqual(ideal_event_id(full),ideal_event_id(p5))
        self.assertEqual(key(dict(spec,event=ideal_event_id(full))),key(dict(spec,event=ideal_event_id(p5))))
        wrong=self.families['full_return'].sample(Bits(),'tamper')
        d=wrong.read(); d['event']['coefficient']*=2; d['event']['weight']*=2
        with self.assertRaises(ValueError): self.families['full_return'].validate(Evidence.create(d))

    def test_11_prequantum_zero_and_spent_prefix(self):
        r=self.path_record('zero','raw')
        self.assertIsNone(self.path.circuit(r,[]))
        r=self.path_record('prefix','coin',prefix={'CX':F(7)})
        self.assertFalse(r.read()['prequantum_zero_avoids_all_quantum_calls'])
        a=AllTrials(self.path,['prefix'],'X');a.add(r)
        self.assertEqual(a.evidence().read()['spent_prefix']['CX'],7)
        with self.assertRaises(ValueError): self.path.circuit(r,[])

    def test_12_all_five_finite_means_and_provider_connections(self):
        a=sum(float(pi)*q for pi,q in zip(self.p,self.q))
        target=sum((-1j*float(self.x))**j/factorial(j)*np.linalg.matrix_power(a,j) for j in range(6))
        details=[]
        for method,f in self.families.items():
            events=list(f.g.reference_events() if method=='closed_P5_full' else reference_events(f.g))
            mean=sum(float(e['proposal']*e['weight'])*reference(e,self.q) for e in events)
            error=float(np.linalg.norm(mean-target,2));self.assertLess(error,1e-10)
            self.assertLess(float(np.linalg.norm(mean@mean-target@target,2)),1e-10)
            rec=f.sample(Bits(),'connection'); c=f.build(rec,self.adapter)
            self.assertLess(float(np.linalg.norm(Operator(c).data-controlled(reference(rec.read()['event'],self.q)),'fro')),1e-10)
            self.assertTrue(all(e['proposal']*e['weight']==e['coefficient'] for e in events))
            details.append(dict(method=method,events=len(events),digital_mean_error_diagnostic=error,
                                certificate=f.context['certificate']))
        from support import save
        save('evidence/family_checks.json',dict(scope='SYNTHETIC_ONLY',rows=details,
            finite_support_enumeration='TEST_ORACLE_ONLY_L3',new_H4_science=0,native_calls=0,shots=0))

    def test_13_reduced_raw_phase_is_preserved(self):
        f=self.families['ordinary']; event=f.g.event(1,(0,0),1)
        self.assertEqual(event['word'],())
        self.assertEqual(event['phase_i_power'],2)
        r=f.formal(event,'formal')
        self.assertLess(float(np.linalg.norm(Operator(f.build(r,self.adapter)).data-
                        controlled(reference(event,self.q)),'fro')),1e-10)

    def test_14_invalid_bits_and_real_entrance_closed(self):
        f=self.family('ordinary')
        with self.assertRaises(ValueError): f.sample(Bits([-1]),'bad')
        self.assertIsNotNone(f.failed)
        with self.assertRaises(ValueError): f.sample(Bits(),'retry')
        with self.assertRaises(ValueError): Family('full_return',self.p,self.x,'H4',self.adapter.labels_id,
                                                  scope='LIVE_H4')

    def test_15_integrated_sampler_failure_is_terminal(self):
        f=self.family('ordinary')
        pth=WholePath([f,f],schedule_id='synthetic:integrated',independence_assumption='ASSUMED_IID')
        a=AllTrials(pth,['good','bad'],'X')
        r=a.prepare('good',[Bits(),Bits()]); a.add(r,1)
        with self.assertRaises(ValueError): a.prepare('bad',[Bits([-1]),Bits()])
        d=a.evidence().read()
        self.assertIsNotNone(d['failure']);self.assertFalse(d['complete'])
        self.assertEqual(d['valid_zeros'],0);self.assertIsNone(d['full_trial_mean'])
        with self.assertRaises(ValueError): a.prepare('bad',[Bits(),Bits()])

if __name__ == '__main__':
    unittest.main(verbosity=2)
