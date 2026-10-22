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
from trotterlib.isolated_qiskit_execution import save, read, run_processes, await_release


def task(index, barrier=None):
        completed = 0
        try:
            if barrier is not None:
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


def run(destination):
    destination.mkdir()
    barrier = threading.Barrier(4)
    with ThreadPoolExecutor(max_workers=4) as pool:
        reports = list(pool.map(lambda index:task(index,barrier), range(4)))
    save(destination/"diagnostic.json", dict(artificial_only=True, threads=4, maximum_iterations=64,
        reports=reports, borrow_exception_reproduced=any("borrow" in r.get("reason","").lower() for r in reports),
        global_Qiskit_modified=False, scientific_calls=0))


def processes(destination):
    destination.mkdir()
    gate = destination/"release.json"
    jobs=[]
    for i in range(4):
        out=destination/str(i)
        out.mkdir()
        jobs.append(dict(case=str(i), output=out, argv=[sys.executable, __file__, "--worker", str(i), str(out), str(gate)]))
    reports=run_processes(jobs,gate,cwd=Path.cwd())
    results=[read(destination/str(i)/"result.json") for i in range(4)]
    if any(r["exit_code"]!=0 for r in reports) or any(r["status"]!="PASS" for r in results):
        raise RuntimeError("ISOLATED_CONTENTION_FAILED")
    save(destination/"diagnostic.json",dict(status="PASS", artificial_only=True, reports=results,
        processes=reports, same_workload_as_thread_diagnostic=True, scientific_calls=0))


if __name__=="__main__":
    if sys.argv[1]=="--worker":
        out=Path(sys.argv[3])
        await_release(out,sys.argv[4])
        report=task(int(sys.argv[2]))
        save(out/"result.json",report)
        if report["status"]!="PASS": sys.exit(1)
    elif sys.argv[1]=="--processes":
        processes(Path(sys.argv[2]))
    else:
        run(Path(sys.argv[1]))
