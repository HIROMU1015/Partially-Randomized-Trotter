"""Artificial-only backend/cache equivalence; no molecular input or compile."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import run_track_d_c7_3_q32 as bootstrap
bootstrap.c2.old().bootstrap(os.environ['TRACK_D_ORIGIN'])
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.circuit.library import CXGate, XXPlusYYGate
from qiskit.quantum_info import Operator
from trottertracks.c1_h6_state_action import simulation_circuit as original_conversion
from trottertracks import track_d_c7_3_acceleration as fast


def artificial(n,layers=16):
    q=QuantumCircuit(n);q.global_phase=-.371
    for i in range(layers):
        a,b=i%n,(i+1)%n
        q.rz([.23,-.17,0.,1e-14][i%4],a);q.sx(b);q.cx(a,b)
        if i%3==0:q.append(XXPlusYYGate(.28,-.13),[a,b])
        if i%4==0:q.crz(-.19,b,a)
    block=QuantumCircuit(n);block.global_phase=.193;block.p(.07,0);block.cx(0,n-1)
    q.append(block.to_gate(),range(n));return q


class FastTests(unittest.TestCase):
    def test_cache_conversion_phase_zero_small_signed_and_composite_match_original(self):
        for n in (3,5):
            q=artificial(n);before=Operator(q).data.copy()
            converted=fast.cached_simulation_circuit(q)
            np.testing.assert_allclose(Operator(converted).data,Operator(original_conversion(q)).data,atol=1e-12,rtol=0)
            np.testing.assert_array_equal(Operator(q).data,before)
        self.assertGreater(fast.cache_statistics()['hits'],0)
        self.assertNotEqual(fast._key(__import__('qiskit').circuit.library.RZGate(0.)),fast._key(__import__('qiskit').circuit.library.RZGate(1e-14)))

    def test_cache_control_state_and_arbitrary_same_name_do_not_alias(self):
        self.assertNotEqual(fast._key(CXGate(ctrl_state=0)),fast._key(CXGate(ctrl_state=1)))
        q=QuantumCircuit(2)
        for angle in (.29,-.43):
            definition=QuantumCircuit(1);definition.ry(angle,0)
            gate=Gate('rz',1,[]);gate.definition=definition
            self.assertIsNone(fast._key(gate));q.append(gate,[0])
        q.append(CXGate(ctrl_state=0),[0,1]);q.cx(0,1)
        np.testing.assert_allclose(Operator(fast.cached_simulation_circuit(q)).data,Operator(q).data,atol=1e-12,rtol=0)

    def test_complete_small_unitary_all_entries_phase_and_adjoint(self):
        for n in (3,5,9):
            q=artificial(n);expected=Operator(q).data
            for threads in (1,4):
                engine=fast.SmallUnitary(threads);value=engine(q)
                np.testing.assert_allclose(value,expected,atol=1e-12,rtol=0)
                np.testing.assert_allclose(engine(q.inverse()),expected.conj().T,atol=1e-12,rtol=0)
                np.testing.assert_allclose(value.conj().T@value,np.eye(1<<n),atol=1e-12,rtol=0)
                self.assertEqual(engine.last_metadata['method'],'unitary')

    def test_reference_batched_state_action_reuses_cache_without_input_mutation(self):
        q=artificial(3);inputs=np.vstack([np.eye(8,dtype=complex),np.eye(8,dtype=complex)[::-1]])
        before=Operator(q).data.copy()
        with patch.object(fast.batched,'simulation_circuit',fast.cached_simulation_circuit):
            action=fast.batched.BatchedStateActions(3)
            states=action.apply(q,inputs)
        np.testing.assert_allclose(states,inputs@before.T,atol=1e-12,rtol=0)
        np.testing.assert_array_equal(Operator(q).data,before)

    def test_nonunitary_unbound_and_seventeen_qubit_unitary_remain_forbidden(self):
        engine=fast.SmallUnitary(2)
        with self.assertRaises(ValueError):engine(QuantumCircuit(17))
        for q in [QuantumCircuit(2,1),QuantumCircuit(2)]:
            if not q.num_clbits:q.reset(0)
            with self.assertRaises(ValueError):fast.cached_simulation_circuit(q)

    def test_long_artificial_identity_at_unchanged_scientific_allowances(self):
        # Fixed before execution: 196608 artificial operations, no molecule.
        # RZ(t),X,RZ(t),X and CX,CX are exactly identity, including phase.
        q=QuantumCircuit(9);q.global_phase=.371
        for i in range(32768):
            a,b=i%9,(i+1)%9;angle=(.23,-.17,0.,1e-14)[i%4]
            q.rz(angle,a);q.x(a);q.rz(angle,a);q.x(a);q.cx(a,b);q.cx(a,b)
        value=fast.SmallUnitary(4)(q);expected=np.exp(1j*.371)*np.eye(512)
        self.assertLessEqual(float(np.max(abs(value-expected))),1e-10)
        self.assertLessEqual(float(np.linalg.norm(value-expected)),2e-10)


if __name__=='__main__':unittest.main()
