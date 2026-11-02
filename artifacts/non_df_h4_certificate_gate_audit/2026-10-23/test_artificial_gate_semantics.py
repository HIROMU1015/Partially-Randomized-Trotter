"""Artificial 1/2-qubit deterministic semantics only; no sampling or compile.

These are author-run technical diagnostics, not interval proofs or certificates.
Supplied components and event indices never come from any real H4 saved model.
"""
import contextlib
from fractions import Fraction as Q
import importlib.util
import json
import math
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate,RZGate
from qiskit.quantum_info import Operator,Statevector

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
# Select the frozen package tree without executing its broad convenience init.
pkg=types.ModuleType('trotterlib');pkg.__path__=[str(ROOT/'src/trotterlib')];sys.modules['trotterlib']=pkg
from trotterlib.fixed_signal_evaluation import hadamard_wrapper
from trotterlib.df_rte_tail import DFBasisRegistry
from trotterlib.df_rte_circuit import DFRTEComponentCircuitSpec,DFRTEEventCircuitRequest,DFRTEIdentityCircuitSpec
from trotterlib.df_rte_qiskit import QiskitDFRTEEventCircuitBuilder
from trotterlib.pauli_signal_adapter import append_pauli_event
from trotterlib.rte import RTEComponent,_make_event,finite_rte_distribution


def fixture(identity=False):
    reg=DFBasisRegistry();definition=reg.register(((RYGate(.23),(0,)),(RZGate(-.17),(0,))),num_system_qubits=1,basis_id='ARTIFICIAL')
    common=dict(df_fragment_id='supplied',basis_id=definition.basis_id,basis_hash=definition.basis_hash,
                basis_change_operations=definition.metadata.operations)
    component=RTEComponent('signed',1.,.125,-1,is_identity=identity,diagonal_pauli_support=() if identity else (0,),**common)
    if identity:
        spec=DFRTEIdentityCircuitSpec(component_id='signed',coefficient_abs=.125,coefficient_sign=-1,num_system_qubits=1,
            df_fragment_id='supplied',basis_id=definition.basis_id,basis_hash=definition.basis_hash)
    else:
        spec=DFRTEComponentCircuitSpec(component_id='signed',coefficient_abs=.125,coefficient_sign=-1,
            num_system_qubits=1,diagonal_pauli_support=(0,),**common)
    basis=QuantumCircuit(1)
    for gate,qubits in definition.runtime_operations:basis.append(gate,list(qubits))
    U=Operator(basis).data
    P=-np.eye(2) if identity else -U@np.diag([1.,-1.])@U.conj().T
    return reg,component,spec,P

def build(order_index=1,controlled=False,reuse=True,identity=False):
    reg,comp,spec,P=fixture(identity)
    distribution=finite_rte_distribution(.03125,4)
    order=distribution.orders[order_index]
    # Fixed manually supplied component indices, not a draw or RNG replay.
    event=_make_event([0]*(order+1),(comp,),distribution,order_index)
    request=DFRTEEventCircuitRequest(event,(spec,),controlled=controlled,
        ancilla_qubit=1 if controlled else None,cancel_adjacent_equal_bases=reuse)
    circuit=QiskitDFRTEEventCircuitBuilder(basis_registry=reg).build_event(request).circuit
    expected=event.phase*(math.cos(event.rotation_angle)*np.eye(2)-1j*math.sin(event.rotation_angle)*P)@np.linalg.matrix_power(P,order)
    return circuit,expected,event

class ArtificialSemantics(unittest.TestCase):
    def assert_close(self,a,b):np.testing.assert_allclose(a,b,rtol=0,atol=2e-14)

    def test_df_signed_products_orders_0_2_4(self):
        for index in range(3):
            with self.subTest(index=index):
                circuit,expected,_=build(index)
                self.assert_close(Operator(circuit).data,expected)

    def test_df_controlled_relative_phases_orders_0_2_4(self):
        for index in range(3):
            with self.subTest(index=index):
                circuit,expected,_=build(index,True)
                self.assert_close(Operator(circuit).data,np.block([[np.eye(2),np.zeros((2,2))],[np.zeros((2,2)),expected]]))

    def test_df_identity_relative_phase(self):
        for index in range(3):
            circuit,expected,_=build(index,True,identity=True)
            self.assert_close(Operator(circuit).data,np.block([[np.eye(2),np.zeros((2,2))],[np.zeros((2,2)),expected]]))

    def test_df_basis_reuse_preserves_supplied_semantics(self):
        reused,_,_=build(reuse=True);explicit,_,_=build(reuse=False)
        self.assert_close(Operator(reused).data,Operator(explicit).data)

    def test_df_ideal_events_unitary(self):
        for index in range(3):
            circuit,_,_=build(index)
            U=Operator(circuit).data;self.assert_close(U.conj().T@U,np.eye(2))

    def test_pauli_signed_event_and_phase(self):
        comp=RTEComponent('p',1.,.125,-1)
        dist=finite_rte_distribution(.03125,4)
        for i,k in enumerate(dist.orders):
            event=_make_event([0]*(k+1),(comp,),dist,i)
            circuit=QuantumCircuit(1);append_pauli_event(circuit,event,{'p':'X'})
            P=-np.array([[0.,1.],[1.,0.]])
            wanted=event.phase*(math.cos(event.rotation_angle)*np.eye(2)-1j*math.sin(event.rotation_angle)*P)@np.linalg.matrix_power(P,k)
            self.assert_close(Operator(circuit).data,wanted)

    def test_cosine_sine_wrapper_probabilities(self):
        circuit,_,_=build()
        state=np.array([.6,.8j]);U=Operator(circuit).data;z=np.vdot(state,U@state)
        initial=Statevector(np.concatenate([state,np.zeros_like(state)]))
        for axis,target in [('cosine',z.real),('sine',z.imag)]:
            wrapper=hadamard_wrapper(circuit,axis,measure=False)
            probability=initial.evolve(wrapper).probabilities([1])
            self.assertAlmostEqual(probability[0]-probability[1],target,places=13)

    def test_native_controlled_wrapper_agrees(self):
        circuit,_,_=build();controlled,_,_=build(controlled=True)
        state=Statevector(np.array([.6,.8j,0.,0.]))
        for axis in ('cosine','sine'):
            generic=hadamard_wrapper(circuit,axis,measure=False)
            native=hadamard_wrapper(controlled,axis,measure=False,already_controlled=True)
            self.assert_close(state.evolve(generic).data,state.evolve(native).data)

    def test_evolution_global_phase_becomes_observable_relative_phase(self):
        phased=QuantumCircuit(1);phased.global_phase=.4
        wrapper=hadamard_wrapper(phased,'sine',measure=False)
        p=Statevector([1.,0.,0.,0.]).evolve(wrapper).probabilities([1])
        self.assertAlmostEqual(p[0]-p[1],math.sin(.4),places=13)

    def test_measured_register_and_no_state_preparation(self):
        circuit=QuantumCircuit(1);circuit.x(0)
        wrapped=hadamard_wrapper(circuit,'cosine')
        self.assertEqual(wrapped.num_qubits,2);self.assertEqual(wrapped.num_clbits,1)
        final=wrapped.data[-1];self.assertEqual(final.operation.name,'measure')
        self.assertEqual(wrapped.find_bit(final.qubits[0]).index,1)
        self.assertNotIn('initialize',wrapped.count_ops())

    def test_rounded_matrix_not_an_exact_involution(self):
        # Exact rational counterexample for the inference, not H4 evidence.
        a=Q(1025,1024);p00=a*a
        self.assertNotEqual(p00*p00,1)
        self.assertGreater(p00,1)

    def test_binary_readout_range_does_not_require_encoded_surrogate_unitarity(self):
        B=Q(7,3)
        self.assertEqual({B*(1-2*bit) for bit in (0,1)},{-B,B})
        # All convex mixtures preserve the range regardless of event weights.
        for probability in (Q(0),Q(1,7),Q(1)):
            mean=B*(2*probability-1);self.assertTrue(-B<=mean<=B)

    def test_wrong_controlled_global_phase_location_is_detected(self):
        right=QuantumCircuit(2);right.p(.4,1)
        wrong=QuantumCircuit(2);wrong.global_phase=.4
        state=Statevector([1/math.sqrt(2),0.,1/math.sqrt(2),0.])
        self.assertFalse(np.allclose(state.evolve(right).data,state.evolve(wrong).data))

    def test_no_production_sampler_or_compiler_needed(self):
        # Exercise the same tiny construction with every RNG/compile entrypoint
        # blocked, making accidental broad test discovery immediately fail.
        with (patch('numpy.random.PCG64',side_effect=AssertionError('NO_RNG')),
              patch('qiskit.transpile',side_effect=AssertionError('NO_COMPILE'))):
            circuit,expected,_=build(0)
            self.assert_close(Operator(circuit).data,expected)

def main():
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ArtificialSemantics)
    with (HERE/'artificial_tests.log').open('x') as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    record=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),
        scope='Manually supplied one-system-qubit events and two-qubit wrappers, float diagnostics and rational counterexample only',
        real_H4_calls=0,high_precision_or_interval_calls=0,sampling_calls=0,RNG_replay=0,compile_calls=0,
        scientific_reanalysis=False,new_certificate=False)
    with (HERE/'artificial_tests.json').open('x') as f:json.dump(record,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(record));sys.exit(0 if result.wasSuccessful() else 1)

if __name__=='__main__':main()
