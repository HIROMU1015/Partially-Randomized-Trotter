"""Synthetic gate/CDF verification, without molecular input or old runs."""
import unittest
from fractions import Fraction
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RZGate,XXPlusYYGate
from qiskit.quantum_info import Operator
from trotterlib.calpha_interval_bounds import IntervalBounds,endpoint
from trotterlib.calpha_sampling_law import (
    finite_cdf_masses,describe_operations,basis_gate_interval,centered_head_error)


class SamplingLawTests(unittest.TestCase):
    def test_grid_boundaries_and_rare_component(self):
        p=[.25,2**-55,.75-2**-55]
        masses,cdf=finite_cdf_masses(p,bits=4)
        self.assertEqual(sum(masses),1)
        for k in range(16):
            picked=int(np.searchsorted(cdf,k/16,side='right'))
            self.assertGreater(masses[picked],0)
        self.assertEqual(masses[0],Fraction(1,4))

    def test_matches_numpy_cdf(self):
        p=np.array([.13,.27,.19,.41])
        _,cdf=finite_cdf_masses(p)
        target=p.cumsum();target/=target[-1]
        self.assertEqual(cdf,target.tolist())

    def test_gate_and_inverse_semantics_full_and_number_block(self):
        operations=[(RZGate(.31),(0,)),(RZGate(-.27),(2,)),
                    (XXPlusYYGate(.62,-.18),(1,0)),(XXPlusYYGate(-.4,.71),(1,2))]
        described=describe_operations(operations)
        qc=QuantumCircuit(3)
        for gate,qubits in operations:qc.append(gate,list(qubits))
        target=Operator(qc).data*np.exp(.5j*(.31-.27))
        b=IntervalBounds()
        full=basis_gate_interval(b,described,list(range(8)),3)
        actual=np.array([[complex(float(z.real.a),float(z.imag.a)) for z in row]
                         for row in full.tolist()])
        np.testing.assert_allclose(actual,target,atol=3e-16)
        states=[1,2,4]
        block=basis_gate_interval(b,described,states,3)
        small=np.array([[complex(float(z.real.a),float(z.imag.a)) for z in row]
                        for row in block.tolist()])
        np.testing.assert_allclose(small,target[np.ix_(states,states)],atol=3e-16)

    def test_centered_head_bound_covers_nonunitary_reference(self):
        b=IntervalBounds();c=b.c
        angle=.003
        a=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        literal=np.eye(2)*(1+2e-8)
        error=b.real(float(np.linalg.norm(a-literal)))
        gram=b.real(float(np.linalg.norm(literal.T@literal-np.eye(2))))
        norm=b.real(float(np.linalg.norm(literal,2)))
        native=[b.real(.03),b.real(.04)]
        rec=[b.real(.03+1e-9),b.real(.04)]
        upper,_,_=centered_head_error(b,error,norm,gram,native,rec)
        exact=np.linalg.norm(a@np.diag(np.exp(1j*np.array([.03,.04])))@a.T
            -literal@np.diag(np.exp(1j*np.array([.03+1e-9,.04])))@literal.T,2)
        self.assertLess(exact,float(upper))
        self.assertLess(float(upper),float(error)/10)


if __name__=='__main__':unittest.main()
