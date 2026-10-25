"""Artificial compiler regressions: compare absolute operators, no phase fit."""
from pathlib import Path
import math
import sys
import unittest
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import XXPlusYYGate
from qiskit.quantum_info import Operator

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.pr_dictionary_validation.fixture import CompilerProfile
from trottertracks.pr_dictionary_validation.gates import compiled_wrapper,stable_official_lowering


class CompilerPhaseTests(unittest.TestCase):
    def check_compile(self, circuit):
        target=Operator(circuit).data
        compiled=compiled_wrapper(circuit,CompilerProfile())
        np.testing.assert_allclose(Operator(compiled).data,target,rtol=0,atol=5e-10)
        self.assertLessEqual(set(compiled.count_ops()),{'rz','sx','x','cx'})
        return compiled

    def test_single_rotations_preserve_phase_and_original(self):
        for angle in (-0.185,0.37,math.pi):
            for name in ('rx','ry'):
                with self.subTest(name=name,angle=angle):
                    circuit=QuantumCircuit(1,global_phase=0.23)
                    getattr(circuit,name)(angle,0)
                    self.check_compile(circuit)
                    self.assertEqual(circuit.data[0].operation.name,name)
                    self.assertEqual(circuit.global_phase,0.23)

    def test_gaussian_primitive_preserves_phase(self):
        for theta,beta in ((0.37,math.pi/2),(-0.61,-0.42)):
            with self.subTest(theta=theta,beta=beta):
                circuit=QuantumCircuit(2)
                circuit.append(XXPlusYYGate(theta,beta),[0,1])
                self.check_compile(circuit)

    def test_controlled_rotation_preserves_relative_phase(self):
        circuit=QuantumCircuit(2,global_phase=0.19)
        circuit.h(0)
        circuit.append(XXPlusYYGate(0.37,math.pi/2),[0,1])
        circuit.cp(-0.23,0,1)
        circuit.cry(0.43,0,1)
        circuit.h(0)
        self.check_compile(circuit)

    def test_many_official_definition_phases_use_one_accurate_sum(self):
        # Artificial cancellation; no scientific fixture or stored arrays.
        circuit=QuantumCircuit(1,global_phase=.23)
        piece=QuantumCircuit(1,global_phase=math.pi/4);piece.rz(.00031,0)
        gate=piece.to_gate()
        for _ in range(3000):circuit.append(gate,[0])
        lowered=stable_official_lowering(circuit,['rz','sx','x','cx','u'])
        self.assertEqual(lowered.global_phase,.23) # 3000*pi/4 = 375*2pi
        # Analytic scalar/gate construction, no phase fit.
        expected=QuantumCircuit(1,global_phase=.23)
        for _ in range(3000):expected.rz(.00031,0)
        self.assertLessEqual(np.linalg.norm(Operator(lowered).data-Operator(expected).data,ord=2),1e-12)
        self.assertEqual(circuit.data[0].operation,gate)

    def test_stable_lowering_controlled_official_definitions(self):
        circuit=QuantumCircuit(3,global_phase=.19)
        circuit.append(XXPlusYYGate(.37,.21),[0,1]);circuit.cry(-.27,2,0);circuit.cp(.013,2,1)
        compiled=compiled_wrapper(circuit,CompilerProfile(),stable_phase=True)
        self.assertLessEqual(np.linalg.norm(Operator(compiled).data-Operator(circuit).data,ord=2),1e-12)

    def test_native_U_and_phase_identities_without_operator_fit(self):
        for theta,phi,lam in ((.01,.2,-.3),(-.7,1.9,.31),(0.,0.,0.)):
            circuit=QuantumCircuit(1);circuit.u(theta,phi,lam,0);circuit.p(-.019,0)
            lowered=stable_official_lowering(circuit,['rz','sx','x','cx','u'])
            self.assertLessEqual(np.linalg.norm(Operator(lowered).data-Operator(circuit).data,ord=2),1e-14)
            self.assertLessEqual(set(lowered.count_ops()),{'rz','sx','x','cx'})
        for name in ('h','s','sdg','t','tdg','sxdg'):
            circuit=QuantumCircuit(1);getattr(circuit,name)(0)
            lowered=stable_official_lowering(circuit,['rz','sx','x','cx','u'])
            self.assertLessEqual(np.linalg.norm(Operator(lowered).data-Operator(circuit).data,ord=2),1e-14)


if __name__=='__main__':unittest.main()
