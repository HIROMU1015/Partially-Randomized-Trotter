"""Synthetic-only whole-operator, gate identity and exact accumulation tests."""
from fractions import Fraction
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import Operator
from qiskit.transpiler.passes import Optimize1qGatesDecomposition

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import m5_phase_batch_compiler as batch


def gate_identity(qc):
    return [(item.operation.name,tuple(qc.find_bit(q).index for q in item.qubits),
        tuple(float(v).hex() for v in item.operation.params)) for item in qc.data]


class BatchTests(unittest.TestCase):
    def test_scalar_exactness_and_no_early_float(self):
        ledger=batch.PhaseLedger();x=batch.ExactPhase(Fraction.from_float(1.7),ledger)
        for _ in range(1024):x=x+2.**-54
        x=x+(-1024*2.**-54)
        self.assertEqual(x.exact_value,Fraction.from_float(1.7))
        self.assertEqual(ledger.count,1025)
        self.assertEqual(list(x.parameters),[])
        with self.assertRaises(ValueError):float(x)

    def test_nonfinite_or_nonfloat_increment_refused(self):
        for value in (float('nan'),float('inf'),1,Fraction(1,2)):
            with self.subTest(value=value),self.assertRaises(ValueError):batch.PhaseLedger().append(value)

    def test_actual_kernel_dispatch_and_gate_identity(self):
        qc=QuantumCircuit(3);qc.rz(.21,0);qc.rz(.31,0);qc.sx(0);qc.sx(0);qc.rz(.11,0);qc.cx(0,1)
        old=Optimize1qGatesDecomposition(basis=['rz','sx','x','cx']);a=old.run(circuit_to_dag(qc))
        records=[];new=batch.PhaseBatchOptimize1qGatesDecomposition(old,records.append)
        b=new.run(circuit_to_dag(qc))
        self.assertEqual(gate_identity(dag_to_circuit(a)),gate_identity(dag_to_circuit(b)))
        self.assertGreater(records[0]['phase_increment_count'],0)
        self.assertTrue(records[0]['accelerated_kernel_unchanged'])
        self.assertEqual(type(b.global_phase),float)

    def test_full_opt1_raw_and_controlled_operators(self):
        for angle in (.2,-.7,2*math.pi):
            qc=QuantumCircuit(3);qc.global_phase=.37
            for i in range(5):
                qc.rz(angle,0);qc.rz(-.13,0);qc.sx(0);qc.sx(0);qc.cx(0,1);qc.cx(0,1)
                qc.rz(.11,1);qc.sx(1);qc.rz(.17,1)
            with self.subTest(angle=angle),tempfile.TemporaryDirectory() as d:
                compiled=batch.compile_with_phase_batch(qc,phase_journal=Path(d)/'phase.jsonl',
                    basis_gates=['rz','sx','x','cx'],optimization_level=1,seed_transpiler=17,num_processes=1)
                original=transpile(qc,basis_gates=['rz','sx','x','cx'],optimization_level=1,seed_transpiler=17,num_processes=1)
                self.assertEqual(gate_identity(compiled),gate_identity(original))
                self.assertLess(np.linalg.norm(Operator(qc).data-Operator(compiled).data),1e-12)
                self.assertLess(np.linalg.norm(Operator(qc.to_gate().control()).data-Operator(compiled.to_gate().control()).data),1e-12)
                records=[json.loads(s) for s in (Path(d)/'phase.jsonl').read_text().splitlines()]
                self.assertEqual([r['pass_ordinal'] for r in records],list(range(len(records))))
                self.assertTrue(all(r['phase_rounding_assignments_at_boundary']==1 and not r['postcompile_correction'] for r in records))

    def test_wrong_profile_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'phase.jsonl';p.write_text('preserved')
            qc=QuantumCircuit(3);qc.rz(.1,0);qc.rz(.2,0)
            with self.assertRaises(FileExistsError):batch.compile_with_phase_batch(qc,phase_journal=p,basis_gates=['rz','sx','x','cx'],optimization_level=1,seed_transpiler=17,num_processes=1)
            self.assertEqual(p.read_text(),'preserved')
        with self.assertRaises(ValueError):batch.patched_pass_manager(recorder=lambda r:None,basis_gates=['u','cx'],optimization_level=1,seed_transpiler=17)

    def test_no_symbolic_or_nonbasis_input(self):
        qc=QuantumCircuit(3);qc.h(0)
        with self.assertRaises(ValueError):batch.compile_with_phase_batch(qc,phase_journal='unused.jsonl',basis_gates=['rz','sx','x','cx'],optimization_level=1,seed_transpiler=17,num_processes=1)


if __name__=='__main__':unittest.main()
