"""Structural toy circuits only; no saved H6 inputs or gate evaluation."""
from pathlib import Path
import sys,unittest
from unittest.mock import patch
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import inspect_h6_m5_phase_structure as s


class StructureTests(unittest.TestCase):
    def circuit(self):
        q=QuantumCircuit(3,1);q.h(2);q.p(.37,0);q.p(-.37,1);q.cx(0,2);q.p(.21,2);q.h(2);q.measure(2,0)
        return q

    def test_counts_positions_and_opposite_bits(self):
        r=s.summarize(self.circuit(),'WRAPPER',3)
        self.assertEqual(r['gate_counts'],dict(h=2,p=3,cx=1,measure=1))
        self.assertEqual(r['adjacent_P_opposite_binary64_on_distinct_qubits'],1)
        self.assertIn(dict(gate='p',qubits=[2],count=1),r['per_location_counts'])

    def test_order_digest_not_statevector_or_matrix(self):
        with patch.object(Statevector,'evolve',side_effect=AssertionError('forbidden')):
            q=self.circuit();a=s.summarize(q,'WRAPPER',3);b=s.summarize(q.copy(),'WRAPPER',3)
        self.assertEqual(a['ordered_metadata_sha256'],b['ordered_metadata_sha256'])
        self.assertFalse(a['phase_parameter_sums_computed']);self.assertFalse(a['determinant_ledger_evaluated'])
        self.assertEqual(a['statevector_replays'],0)

    def test_unknown_symbolic_and_nonfinite_rejected(self):
        q=self.circuit();q.data[0].operation # synthetic input only
        bad=QuantumCircuit(3,1);bad.x(0);bad.measure(2,0)
        with self.assertRaisesRegex(ValueError,'UNKNOWN_SAVED_GATE'):s.summarize(bad,'WRAPPER',3)
        bad=QuantumCircuit(3,1);bad.rz(Parameter('x'),0);bad.measure(2,0)
        with self.assertRaisesRegex(ValueError,'SYMBOLIC'):s.summarize(bad,'WRAPPER',3)
        with self.assertRaisesRegex(ValueError,'FINITE_SERIALIZED'):s.numeric_bits(float('nan'))

    def test_nonterminal_or_wrong_measurement_rejected(self):
        for wrong in (True,False):
            q=QuantumCircuit(3,1);q.measure(0 if wrong else 2,0)
            if not wrong:q.h(0)
            with self.assertRaises(ValueError):s.summarize(q,'WRAPPER',3)

    def test_no_science_or_angle_sum_calls_in_inspector(self):
        import ast
        for node in ast.walk(ast.parse((ROOT/'scripts/inspect_h6_m5_phase_structure.py').read_text())):
            if isinstance(node,ast.Call):
                name=getattr(node.func,'id',getattr(node.func,'attr',''))
                self.assertNotIn(name,('sum','fsum','to_matrix','det','norm','evolve','transpile','decompose','load_vectors'))


class SymbolicLedgerProposalTests(unittest.TestCase):
    """Exact symbolic toy identities; no real QPY parameter sums."""
    def test_gate_determinants_and_phase_weights(self):
        import sympy as sp
        t,b=sp.symbols('t b',real=True);i=sp.I
        c,sn=sp.cos(t/2),sp.sin(t/2)
        xy=sp.eye(4);xy[1:3,1:3]=sp.Matrix([[c,-i*sn*sp.exp(-i*b)],[-i*sn*sp.exp(i*b),c]])
        gates=[(sp.Matrix([[1,1],[1,-1]])/sp.sqrt(2),-1),
            (sp.Matrix([[0,1],[1,0]]),-1),
            (sp.Matrix([[1+i,1-i],[1-i,1+i]])/2,i),
            (sp.diag(1,sp.exp(i*t)),sp.exp(i*t)),
            (sp.diag(sp.exp(-i*t/2),sp.exp(i*t/2)),1),
            (sp.diag(sp.exp(-i*t/2),sp.exp(i*t/2),sp.exp(i*t/2),sp.exp(-i*t/2)),1),
            (sp.Matrix([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]]),-1),(xy,1)]
        for gate,det in gates:self.assertEqual(sp.simplify(gate.det()-det),0)

    def test_tensor_lift_determinant_exponent(self):
        import sympy as sp
        t=sp.symbols('t',real=True);p=sp.diag(1,sp.exp(sp.I*t))
        self.assertEqual(sp.simplify(sp.kronecker_product(sp.eye(4),p).det()-sp.exp(4*sp.I*t)),0)

    def test_determinant_alias_does_not_certify_operator_equality(self):
        import sympy as sp
        phase=sp.exp(sp.I*sp.pi/4);unitary=phase*sp.eye(8)
        self.assertEqual(sp.simplify(unitary.det()),1)
        self.assertFalse(unitary.equals(sp.eye(8)))


if __name__=='__main__':unittest.main()
