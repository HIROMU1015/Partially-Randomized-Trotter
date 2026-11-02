"""Artificial raw operators for the conservative one-qubit control."""
from pathlib import Path
import sys
import json
import tempfile
import unittest
import numpy as np
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag,dag_to_circuit
from qiskit.quantum_info import Operator
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import m5_preserve_oneq_compiler as compiler


def without_barriers(circuit):
    q=QuantumCircuit(circuit.num_qubits);q.global_phase=circuit.global_phase
    for item in circuit.data:
        if item.operation.name!='barrier':q.append(item.operation,[circuit.find_bit(bit).index for bit in item.qubits])
    return q


class PreservationTests(unittest.TestCase):
    def test_replacement_returns_identical_DAG_and_phase(self):
        q=QuantumCircuit(2);q.global_phase=.37;q.rz(2.**-40,0);q.sx(1);q.cx(0,1)
        dag=circuit_to_dag(q);rows=[]
        result=compiler.PreserveOneQubitRuns(rows.append).run(dag)
        self.assertIs(result,dag)
        self.assertEqual(np.linalg.norm(Operator(q).data-Operator(dag_to_circuit(result)).data),0.)
        self.assertFalse(rows[0]['oneq_resynthesis_executed'])
        self.assertFalse(rows[0]['scalar_phase_generated'])

    def test_full_pipeline_retains_small_turns_with_raw_operators(self):
        for angle in (2.**-55,2.**-40,.2,-.7):
            q=QuantumCircuit(2);q.global_phase=.37
            for _ in range(16):q.rz(angle,0);q.barrier(0)
            q.sx(1);q.sx(1);q.cx(0,1);q.cx(0,1)
            with self.subTest(angle=angle),tempfile.TemporaryDirectory() as d:
                path=Path(d)/'policy.jsonl'
                out=compiler.compile_with_preserve_oneq(q,phase_journal=path,basis_gates=['rz','sx','x','cx'],optimization_level=1,seed_transpiler=17,num_processes=1)
                self.assertEqual(out.count_ops()['rz'],16)
                self.assertEqual([float(i.operation.params[0]).hex() for i in out.data if i.operation.name=='rz'],[angle.hex()]*16)
                self.assertLess(np.linalg.norm(Operator(q).data-Operator(out).data),2e-14)
                self.assertLess(np.linalg.norm(Operator(without_barriers(q).to_gate().control()).data-Operator(without_barriers(out).to_gate().control()).data),2e-13)
                rows=[json.loads(r) for r in path.read_text().splitlines()]
                self.assertGreater(len(rows),0)
                self.assertTrue(all(not r['oneq_resynthesis_executed'] and r['retained_same_DAG'] for r in rows))

    def test_other_opt1_tasks_still_execute(self):
        q=QuantumCircuit(2);q.cx(0,1);q.cx(0,1);q.rz(.2,0);q.rz(.3,0)
        with tempfile.TemporaryDirectory() as d:
            out=compiler.compile_with_preserve_oneq(q,phase_journal=Path(d)/'policy.jsonl',basis_gates=['rz','sx','x','cx'],optimization_level=1,seed_transpiler=17,num_processes=1)
            self.assertEqual(out.count_ops().get('cx',0),0)
            self.assertEqual(out.count_ops()['rz'],2)

    def test_profile_and_nonbasis_refused(self):
        with self.assertRaises(ValueError):compiler.patched_pass_manager(recorder=lambda r:None,basis_gates=['u','cx'],optimization_level=1,seed_transpiler=17)
        q=QuantumCircuit(1);q.h(0)
        with self.assertRaises(ValueError):compiler.compile_with_preserve_oneq(q,phase_journal='unused',basis_gates=['rz','sx','x','cx'],optimization_level=1,seed_transpiler=17,num_processes=1)

    def test_no_journal_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'kept';p.write_text('preserved')
            with self.assertRaises(FileExistsError):compiler.compile_with_preserve_oneq(QuantumCircuit(1),phase_journal=p,basis_gates=['rz','sx','x','cx'],optimization_level=1,seed_transpiler=17,num_processes=1)
            self.assertEqual(p.read_text(),'preserved')


if __name__=='__main__':unittest.main()
