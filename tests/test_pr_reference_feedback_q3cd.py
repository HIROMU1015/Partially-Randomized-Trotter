"""Artificial circuit/control-flow checks only; no H4 cache/event generation."""
import math
from fractions import Fraction as F
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch, Mock
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from trottertracks import pr_reference_feedback_q3cd as c


def session(arm):
    n = 4096
    history = tuple(c.q1.RoundCounts(m,c.q1.AxisCounts(n,n),c.q1.AxisCounts(n,n//2))
                    for m in range(17))
    with patch.object(c.q1,'_saved_prefix_shots',return_value=(n,)*17):
        return c.AdaptiveReadout.from_counts(arm,history)


class ArtificialTests(unittest.TestCase):
    def test_one_angle_fresh_calls_and_real_Q_wrapper(self):
        for arm in ('Bprime','Reference_B1'):
            s = session(arm)
            prep = NS(rte_preparation=NS(symbolic_tail=NS(tail_id='artificial',tail_hash='fake')),
                      exact_rte_lambda_r=.5)
            data = {'preps':{3:prep}}
            inputs = c.consumer.FixedInputs(None,c.consumer.CACHE_SHA256,data,prep)
            runtime = NS(common=lambda:data,circuit=QuantumCircuit,
                         compile_reference=Mock(side_effect=lambda q:q.copy()))
            class Events:
                def __len__(self):return 131072*32
            class Source(c.producer_module.CanonicalFiniteBitProducer):
                def __init__(self):self.calls=[]
                def __call__(self,meta,prep):
                    events=Events();self.calls.append(events)
                    cfg=NS(tail_id='artificial',tail_hash='fake',lambda_r=.5)
                    return c.consumer.DeclaredCanonicalEvents(meta,prep,cfg,None,events,
                        c.consumer.CANONICAL_LAW,'artificial continuing source')
            source=Source();width=9 if arm=='Bprime' else 17
            core=QuantumCircuit(width);core.p(.375,width-1)
            with patch.object(c.consumer,'_runtime',return_value=runtime), \
                 patch.object(c,'_bprime_core',return_value=core), \
                 patch.object(c.schedule,'assemble_reference_for_saved_round',return_value={'core':core}) as assemble:
                shots=[s.physical_shot(inputs,producer=source) for _ in range(2)]
            self.assertEqual(len(source.calls),2)
            self.assertIsNot(source.calls[0],source.calls[1])
            self.assertEqual(runtime.compile_reference.call_count,2)
            for call in runtime.compile_reference.call_args_list:
                raw=call.args[0]
                self.assertEqual([x.operation.name for x in raw.data],['h','p','p','h','measure'])
                self.assertEqual(float(raw.data[-3].operation.params[0]),-s.feedback.parameter)
            self.assertEqual([x['axis'] for x in shots],['Q','Q'])
            self.assertEqual(shots[0]['feedback_parameter_hex'],shots[1]['feedback_parameter_hex'])
            self.assertEqual([i.operation.name for i in shots[0]['circuit'].data[:8]],
                             ['reset']*8 if arm=='Reference_B1' else ['h','p','p','h','measure'])
            if arm=='Reference_B1':
                self.assertTrue(all(x.kwargs['mode']=='exact_binary_m17' for x in assemble.call_args_list))
            else:self.assertEqual(assemble.call_count,0)

    def test_binding_and_saved_decoder(self):
        s=session('Bprime')
        with self.assertRaises(ValueError):c.AdaptiveReadout('Reference_B1',s.prefix,s.feedback)
        model=c.q1.SignalModel(F(3,4),F(1,100),saved_arm='Bprime')
        with patch.object(c.q1,'saved_q0_model',return_value=model):
            result=s.decode(170)
        self.assertEqual(result['status'],'CONDITIONAL_WIDTH_SUFFICIENT')
        self.assertEqual(result['n'],340)
        self.assertFalse(result['formal_RPE_or_resource_qualification'])
        with self.assertRaises(TypeError):s.physical_shot(None,producer=object())

    def test_virtual_state_identity_and_general_disk_units(self):
        # Fixed artificial entangled joint state and imperfect readout H.
        v=np.array([1,2j,3,1-1j],complex);v/=np.linalg.norm(v)
        H=np.array([[1,1],[1,-1]],complex)/math.sqrt(2)
        qc=QuantumCircuit(1);qc.h(0);qc.ry(.0003,0)
        Hn=Operator(qc).data
        for phi in (0.,.213,-.719,math.pi):
            R=np.diag([np.exp(.5j*phi),np.exp(-.5j*phi)])
            physical=(Hn@R@v.reshape(2,2)).reshape(-1)
            virtual=(R.conj().T@H@Hn@R@v.reshape(2,2)).reshape(-1)
            z=2*np.vdot(virtual[:2],virtual[2:])
            mu=np.vdot(physical[:2],physical[:2]).real-np.vdot(physical[2:],physical[2:]).real
            self.assertLess(abs(mu-(np.exp(-1j*phi)*z).real),1e-14)
            self.assertLessEqual(abs(z),1+1e-14)
            original=2*np.vdot(v[:2],v[2:])
            self.assertLessEqual(abs(z-original),2*math.sqrt(2)*np.linalg.norm(Hn-H,2)+1e-14)


if __name__=='__main__':unittest.main()
