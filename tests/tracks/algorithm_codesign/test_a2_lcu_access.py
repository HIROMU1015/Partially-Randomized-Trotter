"""Only invented 1/2-qubit states and circuits; no files, RNG or compilation.

Operator/Statevector below are independent small references, not algorithmic
state/norm/probability oracles. The adapter itself never imports them.
"""
import ast
from fractions import Fraction as F
from itertools import product
import math
from pathlib import Path
import unittest

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Parameter
from qiskit.quantum_info import Operator, Statevector

from trottertracks.algorithm_codesign import a2_lcu_access as a
from trottertracks.pr_dictionary_validation.gates import matrix_gate


def fixtures(n=1):
    fine, coarse = QuantumCircuit(n), QuantumCircuit(n)
    fine.rx(-.37, 0); fine.rz(.23, 0); fine.global_phase = .41
    coarse.ry(.19, 0); coarse.global_phase = -.17
    if n == 2:
        fine.cx(0, 1); coarse.rzz(-.28, 0, 1)
    return fine, coarse


def hadamard_state(unitary, psi):
    return np.concatenate((psi, unitary @ psi)) / np.sqrt(2.)


def statistics(plan, psi, weight=F(1)):
    # Algorithm inputs have a,b,h=0. No normalized success-state oracle is used.
    initial = np.zeros(1 << plan.circuit.num_qubits, complex)
    initial[:len(psi)] = psi
    probabilities = abs(Statevector(initial).evolve(plan.circuit).data)**2
    mean = second = fourth = success = 0.
    for i, probability in enumerate(probabilities):
        bits = tuple((i >> q) & 1 for q in
                     (plan.herald_ancilla, plan.block_ancilla, plan.signal_ancilla))
        value = plan.scale.outcome(*bits, common_weight=weight)
        mean += probability * value
        second += probability * value**2
        fourth += probability * value**4
        success += probability * (bits[0] == 0)
    return mean, second, fourth, success


class LCUAccessTests(unittest.TestCase):
    def test_success_and_failure_operator_blocks_on_every_input(self):
        for n, zeta in product((1, 2), (.5, 1., 3.)):
            fine, coarse = fixtures(n); plan = a.build_access(fine, coarse, zeta=zeta)
            m = Operator(plan.circuit).data; half = len(m)//2
            # Existing Hadamard convention, independently constructed reference.
            vf = QuantumCircuit(n+1); vf.h(n); vf.append(fine.to_gate().control(), [n,*range(n)])
            vc = QuantumCircuit(n+1); vc.h(n); vc.append(coarse.to_gate().control(), [n,*range(n)])
            f, c = Operator(vf).data, Operator(vc).data
            s, beta = plan.scale.s, plan.scale.beta
            block = np.array([[-zeta, -1.], [1., -zeta]])/s
            left = np.kron(np.eye(2), f); right = np.kron(block, c)
            expected = (zeta*left+s*right)/beta
            failure = math.sqrt(zeta*s)/beta*(right-left)
            matrix_gate(m[:half,:half], expected, 3e-12, label='SUCCESS_BLOCK')
            matrix_gate(m[half:,:half], failure, 3e-12, label='FAILURE_BLOCK')
            matrix_gate(m.conj().T@m, np.eye(len(m)), 3e-12, label='UNITARY_DILATION')

    def test_success_probability_and_zero_fill_both_axes(self):
        psi = np.array([3.,4j])/5.
        fine, coarse = fixtures();f,c=Operator(fine).data,Operator(coarse).data
        sf,sc=hadamard_state(f,psi),hadamard_state(c,psi)
        d2=float(np.vdot(sf-sc,sf-sc).real)
        for zeta,axis,weight in product((.5,1.,3.),('X','Y'),(F(1),F(-2,3))):
            plan=a.build_access(fine,coarse,zeta=zeta,axis=axis)
            mean,m2,m4,p=statistics(plan,psi,weight)
            signal=np.vdot(psi,(f-c)@psi)
            expected=signal.real if axis=='X' else signal.imag
            self.assertAlmostEqual(mean,float(weight)*expected,places=11)
            self.assertAlmostEqual(p,(1+zeta*zeta*d2)/plan.scale.beta2,places=12)
            self.assertAlmostEqual(m2,float(weight)**2*plan.scale.beta2*(d2+zeta**-2),places=10)
            self.assertLessEqual(m4,plan.scale.absolute_range(weight)**2*m2+1e-8)
            self.assertTrue(0<p<=1)

    def test_identical_paths_retain_all_attempt_noise_floor(self):
        fine,_=fixtures();psi=np.array([1.,0.],complex)
        for zeta,axis in product((.5,1.,3.),('X','Y')):
            plan=a.build_access(fine,fine,zeta=zeta,axis=axis)
            mean,m2,_,p=statistics(plan,psi)
            self.assertAlmostEqual(mean,0.,places=12)
            self.assertAlmostEqual(p,1/plan.scale.beta2,places=12)
            self.assertAlmostEqual(m2,plan.scale.beta2/zeta**2,places=10)
            self.assertGreater(m2,4.)

    def test_scalar_phase_is_coherent_and_Y_sign_is_preserved(self):
        fine,coarse=QuantumCircuit(1),QuantumCircuit(1)
        fine.global_phase=-.43
        for axis in ('X','Y'):
            plan=a.build_access(fine,coarse,zeta=1.,axis=axis)
            mean,_,_,_=statistics(plan,np.array([1.,0.],complex))
            expected=math.cos(.43)-1 if axis=='X' else -math.sin(.43)
            self.assertAlmostEqual(mean,expected,places=12)
        self.assertNotEqual(mean,0.) # A channel-only/global-phase drop would fail.

    def test_gate_order_and_two_system_qubit_layout(self):
        fine,coarse=fixtures(2);psi=np.array([1.,1j,-1.,1.])/2.
        signal=np.vdot(psi,(Operator(fine).data-Operator(coarse).data)@psi)
        for axis in ('X','Y'):
            plan=a.build_access(fine,coarse,zeta=1.,axis=axis)
            self.assertEqual((plan.signal_ancilla,plan.block_ancilla,plan.herald_ancilla),(2,3,4))
            self.assertAlmostEqual(statistics(plan,psi)[0],signal.real if axis=='X' else signal.imag,places=11)

    def test_measurement_register_mapping_and_failure_accounting(self):
        fine,coarse=fixtures();plan=a.build_access(fine,coarse,zeta=1.,axis='Y',measure=True)
        mapping=[(plan.circuit.find_bit(x.qubits[0]).index,plan.circuit.find_bit(x.clbits[0]).index)
                 for x in plan.circuit.data if x.operation.name=='measure']
        self.assertEqual(mapping,[(3,0),(2,1),(1,2)])
        for b,c in product((0,1),repeat=2):self.assertEqual(plan.scale.outcome(1,b,c),0.)
        self.assertEqual(plan.circuit.metadata['fine_system_calls'],1)
        self.assertEqual(plan.circuit.metadata['coarse_system_calls'],1)
        self.assertFalse(plan.numerically_certified)
        self.assertEqual(plan.circuit.metadata['numerical_error_bound'],'UNKNOWN')
        self.assertAlmostEqual(plan.scale.outcome(0,0,0),-plan.scale.outcome(0,0,1))

    def test_exact_toy_pushforward_marginal_without_sampling(self):
        labels={'a':F(1,4),'b':F(3,4)};order={0:F(3,4),2:F(1,4)}
        marginal={label:F(0) for label in labels}
        for n,pn in order.items():
            for ids in product(labels,repeat=n+1):
                coarse=a.order0_projection(n,ids);p=pn
                for label in ids:p*=labels[label]
                marginal[coarse[1][0]]+=p
        self.assertEqual(marginal,labels)
        original=['a','b','a'];self.assertEqual(a.order0_projection(2,original),(0,('a',)))
        self.assertEqual(original,['a','b','a'])

    def test_nonphysical_or_unspecified_inputs_stop(self):
        fine,coarse=fixtures()
        for zeta in (0.,-1.,float('nan'),float('inf'),1e308,1e-308,True):
            with self.assertRaises(ValueError):a.build_access(fine,coarse,zeta=zeta)
        bad=[]
        c=QuantumCircuit(1);c.reset(0);bad.append(c)
        c=QuantumCircuit(1,1);c.measure(0,0);bad.append(c)
        c=QuantumCircuit(1);c.rz(Parameter('t'),0);bad.append(c)
        c=QuantumCircuit(1);c.append(Gate('free_state_oracle',1,[]),[0]);bad.append(c)
        c=QuantumCircuit(1);c.append(Gate('rz',1,[.2]),[0]);bad.append(c)
        for c in bad:
            with self.assertRaises(ValueError):a.build_access(c,coarse,zeta=1.)
        with self.assertRaises(ValueError):a.build_access(fine,QuantumCircuit(2),zeta=1.)
        with self.assertRaises(ValueError):a.build_access(fine,coarse,zeta=1.,measure=True)
        with self.assertRaises(ValueError):a.order0_projection(4,['a']*5)
        with self.assertRaises(ValueError):a.LCUScale.from_zeta(1.).outcome(0,0,0,common_weight=.5)

    def test_adapter_has_no_probability_state_rng_or_science_loader(self):
        tree=ast.parse(Path(a.__file__).read_text())
        forbidden={'Statevector','Operator','SavedEngine','from_fixed_bundle','qualify_fixed_bundle',
                   'transpile','random','default_rng','read_text','open','evolve','initialize'}
        names={node.id for node in ast.walk(tree) if isinstance(node,ast.Name)}
        attrs={node.attr for node in ast.walk(tree) if isinstance(node,ast.Attribute)}
        self.assertFalse(forbidden&(names|attrs))


if __name__=='__main__':unittest.main()
