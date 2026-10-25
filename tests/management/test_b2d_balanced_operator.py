"""Artificial bit-order/product checks; no saved scientific arrays or fixture."""
from pathlib import Path
import runpy
import unittest
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

ROOT=Path(__file__).resolve().parents[2]
balanced=runpy.run_path(str(ROOT/'scripts/diagnose_b2d_b0_compiler.py'))['balanced_operator']

class BalancedTests(unittest.TestCase):
    def test_bit_order_noncommuting_products_and_global_phase(self):
        for n in (1,2,3):
            c=QuantumCircuit(n,global_phase=.23)
            for q in range(n):c.rx(.13*(q+1),q);c.rz(-.41,q)
            if n>1:c.cx(n-1,0);c.cry(-.29,0,n-1)
            if n>2:c.ccx(0,2,1)
            a=balanced(c)
            self.assertEqual(a.dtype,np.complex128)
            self.assertLess(np.linalg.norm(a-Operator(c).data,ord=2),1e-14)

    def test_odd_tree_empty_and_terminal_measurement(self):
        for length in (0,1,3,501):
            c=QuantumCircuit(3,1)
            for j in range(length):c.sx(j%3);c.rz(.01*(j%7-3),j%3)
            c.h(2);c.measure(2,0)
            self.assertLess(np.linalg.norm(balanced(c)-Operator(c.remove_final_measurements(inplace=False)).data,ord=2),1e-12)

if __name__=='__main__':unittest.main()
