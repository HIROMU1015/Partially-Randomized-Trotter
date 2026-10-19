"""Synthetic checks of the H6 wrapper; no molecular file is loaded."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
import numpy as np
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag,dag_to_circuit
from qiskit.quantum_info import Operator,Statevector
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import m5_algebraic_zero_compiler as c
from run_m5_algebraic_oneq_artificial import fixtures


class AlgebraicZeroTests(unittest.TestCase):
    def test_same_planner_and_raw_operators_as_artificial_source(self):
        from m5_algebraic_oneq_prototype import AlgebraicPlanner
        self.assertIs(c.AlgebraicPlanner,AlgebraicPlanner)
        for name,q in fixtures().items():
            with self.subTest(name=name):
                rows=[];out=dag_to_circuit(c.AlgebraicZeroRounding(rows.append).run(circuit_to_dag(q)))
                self.assertLess(np.linalg.norm(Operator(q).data-Operator(out).data),1e-12)
                self.assertEqual(rows[0]['spent_after']['numerator'],'0')
                self.assertEqual(rows[0]['policy'],c.POLICY)
                self.assertEqual(float(q.global_phase).hex(),float(out.global_phase).hex())

    def test_full_13_qubit_synthetic_measurement_path(self):
        q=QuantumCircuit(13,1);q.global_phase=.37;q.sx(0);q.rz(.125,0);q.rz(.375,0);q.cx(0,12);q.measure(12,0)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'policy.jsonl'
            out=c.compile_with_algebraic_zero(q,phase_journal=path,basis_gates=['rz','sx','x','cx'],optimization_level=1,seed_transpiler=17,num_processes=1)
            a=Statevector.from_instruction(q.remove_final_measurements(inplace=False)).data
            b=Statevector.from_instruction(out.remove_final_measurements(inplace=False)).data
            self.assertLess(np.linalg.norm(a-b),1e-12)
            self.assertEqual(out.count_ops()['measure'],1);self.assertEqual(out.count_ops()['rz'],1)
            rows=[json.loads(s) for s in path.read_text().splitlines()]
            self.assertTrue(all(r['spent_after']['numerator']=='0' and not r['Euler_resynthesis'] and not r['angle_wrap'] for r in rows))

    def test_profile_and_wrong_scope_rejected(self):
        q=QuantumCircuit(13,1);q.measure(12,0)
        with self.assertRaises(ValueError):c.compile_with_algebraic_zero(q,phase_journal='unused',basis_gates=['rz','sx','x','cx'],optimization_level=1,seed_transpiler=18,num_processes=1)
        with self.assertRaises(ValueError):c.compile_with_algebraic_zero(QuantumCircuit(1),phase_journal='unused',basis_gates=['rz','sx','x','cx'],optimization_level=1,seed_transpiler=17,num_processes=1)

    def test_no_journal_overwrite(self):
        q=QuantumCircuit(13,1);q.measure(12,0)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'kept';p.write_text('preserved')
            with self.assertRaises(FileExistsError):c.compile_with_algebraic_zero(q,phase_journal=p,basis_gates=['rz','sx','x','cx'],optimization_level=1,seed_transpiler=17,num_processes=1)
            self.assertEqual(p.read_text(),'preserved')


if __name__=='__main__':unittest.main()
