"""Q1 artificial fixtures only. No H4 cache, sampling, compile or Q0 campaign."""
import ast
from dataclasses import replace
from fractions import Fraction as F
import importlib.util
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import XXPlusYYGate
from qiskit.quantum_info import Operator

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('feedback_q1',ROOT/'src/trottertracks/pr_reference_feedback_q1.py')
q1=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=q1
spec.loader.exec_module(q1)
q=q1._q0()


def pure_definitions(path,names,namespace):
    raw=subprocess.check_output(['git','show',q1.Q0_COMMIT+':'+path],cwd=ROOT)
    nodes=[n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),path,'exec'),namespace)
    return namespace


def artificial_history(theta=0.,n=4096):
    rows=[]
    for m in range(17):
        angle=math.remainder((1<<m)*theta,2*math.pi)
        kx=round(n*(1+math.cos(angle))/2)
        ky=round(n*(1+math.sin(angle))/2)
        rows.append(q1.RoundCounts(m,q1.AxisCounts(n,kx),q1.AxisCounts(n,ky)))
    return tuple(rows)


def overlap(a,b):
    return max(q.endpoints(a)[0],q.endpoints(b)[0])<=min(q.endpoints(a)[1],q.endpoints(b)[1])


class PrefixAndDecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.history=artificial_history()
        cls.prefix=q1.prefix_from_counts(cls.history,h=F(1,8))
        cls.feedback=q1.feedback_from_prefix(cls.prefix)
        cls.model=q1.SignalModel(F(3,4),F(1,100),F(1,100))

    def test_exact_count_branch_and_energy_sign(self):
        for theta in (.0123,-.0317):
            p=q1.prefix_from_counts(artificial_history(theta),h=F(1,8))
            lo,hi=q.endpoints(p.energy_interval)
            self.assertLessEqual(lo,-theta/(1/8))
            self.assertGreaterEqual(hi,-theta/(1/8))
            self.assertTrue(all(x['nearest_branch_verified'] for x in p.branch_certificates))
            self.assertGreater(max(c['branch'] for c in p.branch_certificates),0)

    def test_feedback_uses_m16_not_base_phase(self):
        p=q1.prefix_from_counts(artificial_history(.0123),h=F(1,8))
        angle=q1.feedback_from_prefix(p)
        # Exact identity: the lift differs by whole periods after doubling.
        target=2*p.q_last*p.base_phase_center+q.mp.iv.pi/2
        diff=q1._circular_distance(q.I(angle.parameter)-target)
        self.assertLessEqual(q.endpoints(diff)[1],q.endpoints(q.iv(angle.arithmetic_error_upper))[1]+q.mp.mpf('1e-48'))
        wrong=2*p.legacy_result['final_phase_estimate']+math.pi/2
        self.assertGreater(abs(math.remainder(angle.parameter-wrong,2*math.pi)),.01)

    def test_period_cuts_and_nonzero_rounding_enclosure(self):
        for phase in (q.mp.iv.pi,q.mp.iv.pi-q.I('1e-12'),-q.mp.iv.pi+q.I('1e-12'),
                      q.mp.iv.pi/4,-q.mp.iv.pi/4,q.I('.123'),q.I('-.456')):
            f=q1._feedback_phase(phase)
            err=abs(q.I(f.parameter)-f.reduced_ideal_interval)
            self.assertLessEqual(q.endpoints(err)[1],q.endpoints(q.iv(f.arithmetic_error_upper))[1])
            self.assertGreater(f.arithmetic_error_upper,0)
            self.assertLess(abs(f.parameter),math.pi+1e-12)

    def test_zero_sample_vector_and_ambiguous_branch_rejected(self):
        bad=list(self.history)
        bad[0]=q1.RoundCounts(0,q1.AxisCounts(2,1),q1.AxisCounts(2,1))
        with self.assertRaises(q1.UnqualifiedPrefix):q1.prefix_from_counts(bad)
        bad=list(self.history)
        bad[1]=q1.RoundCounts(1,q1.AxisCounts(2,0),q1.AxisCounts(2,1))
        with self.assertRaises(q1.UnqualifiedPrefix):q1.prefix_from_counts(bad)

    def test_alias_edge_does_not_become_linear_interval(self):
        bad=list(self.history)
        bad[0]=q1.RoundCounts(0,q1.AxisCounts(4096,0),q1.AxisCounts(4096,2048))
        with self.assertRaises(q1.UnqualifiedPrefix):q1.prefix_from_counts(bad)

    def test_saved_shot_binding_without_scientific_data(self):
        with patch.object(q1,'_saved_prefix_shots',return_value=(4096,)*17):
            p=q1.prefix_from_counts(self.history,saved_arm='Bprime')
            self.assertEqual(p.saved_arm,'Bprime')
        with patch.object(q1,'_saved_prefix_shots',return_value=(4095,)*17):
            with self.assertRaises(q1.UnqualifiedPrefix):q1.prefix_from_counts(self.history,saved_arm='Bprime')

    def test_count_decoder_reuses_q0_hull(self):
        error=F(1,10000)
        D=q.iv(self.model.bias_upper+self.feedback.arithmetic_error_upper+error)
        for k in (90,140,170,200,250):
            result=q1.decode_counts(self.prefix,self.feedback,k,self.model,additional_feedback_error=error)
            expected=q.hull(340,k,q.iv(self.model.rho_lower),D,q.iv(2*q1.CAP))
            self.assertIsNotNone(expected)
            self.assertEqual(result['phase_diameter'],q.pack(expected[2]))
            self.assertFalse(result['formal_RPE_or_resource_qualification'])

    def test_energy_hull_contains_artificial_consistent_point(self):
        for u in (-.5,0.,.5):
            mu=.8*math.sin(u)
            k=round(340*(1+mu)/2)
            result=q1.decode_counts(self.prefix,self.feedback,k,self.model,additional_feedback_error=F(1,10000))
            a,b=map(F,result['energy_hull']['exact_interval'])
            center=sum(q.endpoints(self.prefix.energy_center))/2
            truth=center-q.mp.mpf(u)/(2*self.prefix.q_last*(1/8))
            self.assertLessEqual(q.mp.mpf(a.numerator)/a.denominator,truth)
            self.assertGreaterEqual(q.mp.mpf(b.numerator)/b.denominator,truth)
            estimate=F(result['energy_estimate_rational'])
            self.assertLessEqual(a,estimate)
            self.assertLessEqual(estimate,b)

    def test_unknown_error_empty_counts_and_wrong_n(self):
        self.assertEqual(q1.decode_counts(self.prefix,self.feedback,170,self.model)['status'],'UNQUALIFIED_FEEDBACK_ERROR')
        for k in (0,340):
            r=q1.decode_counts(self.prefix,self.feedback,k,self.model,additional_feedback_error=F(0))
            self.assertEqual(r['status'],'INCONSISTENT_COUNTS')
            self.assertIsNone(r['estimate'])
        with self.assertRaises(ValueError):q1.decode_counts(self.prefix,self.feedback,170,self.model,n=339)
        with self.assertRaises(ValueError):q1.AxisCounts(340,341)
        with self.assertRaises(ValueError):q1.AxisCounts(True,0)

    def test_no_silent_branch_drop_or_prefix_substitution(self):
        broad=q1.SignalModel(F(3,4),F(1,100),F(2))
        result=q1.decode_counts(self.prefix,self.feedback,170,broad,additional_feedback_error=F(1))
        self.assertEqual(result['status'],'UNQUALIFIED_MULTIPLE_SINE_BRANCHES')
        with self.assertRaises(ValueError):q1.decode_counts(replace(self.prefix),self.feedback,170,self.model,additional_feedback_error=F(0))
        saved=replace(self.model,saved_arm='Reference_B1')
        self.assertEqual(q1.decode_counts(self.prefix,self.feedback,170,saved,additional_feedback_error=F(0))['status'],'UNQUALIFIED_PREFIX_SCHEDULE')

    def test_failure_allocation_and_no_target_oracle(self):
        self.assertEqual(34*F(1,720)+F(1,360),F(1,20))
        with self.assertRaises(TypeError):q1.prefix_from_counts(self.history,true_energy=0)

    def test_precision_context_is_not_silently_changed(self):
        old=q.mp.iv.prec
        try:
            q.mp.iv.prec=80
            with self.assertRaises(RuntimeError):q1.feedback_from_prefix(self.prefix)
        finally:
            q.mp.iv.prec=old


class ReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ns=pure_definitions('src/trottertracks/pr_reference_interference_pilot.py',
                            ('reference_core','wrapper','vacuum_phase'),{'QuantumCircuit':QuantumCircuit})
        ns=pure_definitions('src/trottertracks/pr_reference_bprime.py',('bprime_kernel',),ns)
        cls.old=ns
        frame=QuantumCircuit(2);frame.append(XXPlusYYGate(.375,.125),[0,1])
        f=frame.copy();f.rzz(.125,0,1);f.rz(.0625,0);f.compose(frame.inverse(),inplace=True);f.global_phase=.03125
        g=frame.copy();g.rzz(.125,0,1);g.z(1);g.compose(frame.inverse(),inplace=True);g.global_phase=-.015625
        cls.V=Operator(f).data@Operator(g).data
        b=QuantumCircuit(3);b.compose(frame,qubits=[0,1],inplace=True)
        b.compose(ns['bprime_kernel'](.25),qubits=[2,0,1],inplace=True)
        b.crz(.0625,2,0);b.cz(2,1);b.p(.015625,2)
        b.compose(frame.inverse(),qubits=[0,1],inplace=True)
        ref=ns['reference_core'](f,g,ns['vacuum_phase'](f)+ns['vacuum_phase'](g))
        cls.cores={'Bprime':b,'Reference_B1':ref}

    @staticmethod
    def angle(phi):
        return q1.FeedbackAngle(float(phi),q.I(float(phi)),F(0),0)

    @staticmethod
    def observable(circuit):
        u=Operator(circuit.remove_final_measurements(inplace=False)).data[:,:4]
        n=circuit.num_qubits
        signs=1-2*((np.arange(1<<n)>>(n-1))&1)
        return u.conj().T@(signs[:,None]*u)

    def test_all_input_observable_and_xy_signs(self):
        with patch('qiskit.transpile',side_effect=AssertionError('No compiler in Q1')):
            for family,core in self.cores.items():
                for phi in (0.,math.pi/2,-math.pi/2,math.pi/4,.213,-.719):
                    circuit=q1.feedback_readout(core,self.angle(phi),family=family)
                    target=(np.exp(-1j*phi)*self.V+np.exp(1j*phi)*self.V.conj().T)/2
                    self.assertLess(np.linalg.norm(self.observable(circuit)-target),2e-12)

    def test_old_xy_wrappers_unchanged_observable(self):
        for family,core in self.cores.items():
            for axis,phi in (('X',0.),('Y',math.pi/2)):
                old=self.old['wrapper'](core,axis)
                new=q1.feedback_readout(core,self.angle(phi),family=family)
                self.assertLess(np.linalg.norm(self.observable(old)-self.observable(new)),2e-12)

    def test_common_and_relative_phase_are_different(self):
        for family,core in self.cores.items():
            baseline=q1.feedback_readout(core,self.angle(.213),family=family)
            common=core.copy();common.global_phase+=.375
            shifted=q1.feedback_readout(common,self.angle(.213),family=family)
            self.assertLess(np.linalg.norm(self.observable(baseline)-self.observable(shifted)),2e-12)
            relative=core.copy();relative.p(.25,core.num_qubits-1)
            changed=q1.feedback_readout(relative,self.angle(.213),family=family)
            self.assertGreater(np.linalg.norm(self.observable(baseline)-self.observable(changed)),1e-3)

    def test_layout_parameter_and_core_preserved(self):
        for family,core in self.cores.items():
            before=core.copy()
            result=q1.feedback_readout(core,self.angle(-.719),family=family)
            self.assertEqual(core,before)
            self.assertEqual((result.num_qubits,result.num_clbits),(core.num_qubits,1))
            tail=result.data[-3:]
            self.assertEqual([i.operation.name for i in tail],['p','h','measure'])
            self.assertEqual(tail[0].operation.params[0],.719)
            self.assertEqual(result.find_bit(tail[-1].qubits[0]).index,core.num_qubits-1)
            self.assertEqual(result.find_bit(tail[-1].clbits[0]).index,0)

    def test_reuse_consumer_reset_order_with_mock_compiler(self):
        ns=pure_definitions('src/trottertracks/pr_reference_v6q_consumer.py',('_compile_and_reset',),{})
        events=[]
        class FakeCircuit:
            def __init__(self,n,c):self.num_qubits=n;self.num_clbits=c
            def reset(self,wires):events.append(('reset',tuple(wires)))
            def compose(self,native,inplace):events.append(('compose',native))
        raw=NS(num_qubits=17,num_clbits=1,label='artificial Q readout')
        def compile_mock(circuit):events.append(('compile',circuit));return circuit
        runtime=NS(compile_reference=compile_mock,circuit=FakeCircuit)
        result=ns['_compile_and_reset'](raw,runtime)
        self.assertEqual([r[0] for r in events],['compile','reset','compose'])
        self.assertEqual(events[1][1],tuple(range(8,16)))
        self.assertEqual(result.num_qubits,17)

    def test_invalid_readout_rejected(self):
        with self.assertRaises(ValueError):q1.feedback_readout(QuantumCircuit(3,1),self.angle(0),family='Bprime')
        with self.assertRaises(ValueError):q1.feedback_readout(QuantumCircuit(4),self.angle(0),family='Reference_B1')
        with self.assertRaises(ValueError):q1.feedback_readout(QuantumCircuit(3),self.angle(float('nan')),family='Bprime')


if __name__=='__main__':
    unittest.main()
