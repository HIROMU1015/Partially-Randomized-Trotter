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


if __name__=='__main__':unittest.main()
