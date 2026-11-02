#!/usr/bin/env python3
"""Fixed artificial contention workload; never uses molecular inputs."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import threading
import traceback

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PauliEvolutionGate, RZZGate
from qiskit.quantum_info import SparsePauliOp
from trotterlib.isolated_qiskit_execution import save


def run(destination):
    destination.mkdir()
    barrier = threading.Barrier(4)

    def task(index):
        completed = 0
        try:
            barrier.wait()
            for iteration in range(64):
                circuit = QuantumCircuit(5)
                for j in range(16):
                    if index % 2:
                        circuit.append(RZZGate(.01*(j+1)).control(), [4,j%4,(j+1)%4])
                    else:
                        circuit.append(PauliEvolutionGate(SparsePauliOp("XYZX"), time=.001*(j+1)), range(4))
                transpile(circuit, basis_gates=["rz","sx","x","cx"], optimization_level=1, seed_transpiler=17)
                completed += 1
            return dict(worker=index, status="PASS", completed_iterations=completed)
        except BaseException as exc:
            return dict(worker=index, status="FAILED", completed_iterations=completed,
                        reason=str(exc), traceback=traceback.format_exc())

    with ThreadPoolExecutor(max_workers=4) as pool:
        reports = list(pool.map(task, range(4)))
    save(destination/"diagnostic.json", dict(artificial_only=True, threads=4, maximum_iterations=64,
        reports=reports, borrow_exception_reproduced=any("borrow" in r.get("reason","").lower() for r in reports),
        global_Qiskit_modified=False, scientific_calls=0))


if __name__=="__main__":
    run(Path(sys.argv[1]))
