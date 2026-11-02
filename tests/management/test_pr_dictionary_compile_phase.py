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
from trottertracks.pr_dictionary_validation.gates import compiled_wrapper


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


if __name__=='__main__':unittest.main()
