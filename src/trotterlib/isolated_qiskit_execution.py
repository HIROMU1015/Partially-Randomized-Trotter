"""Fresh-interpreter case execution. No shared Qiskit compiler/session objects.

The parent orchestrates processes and files only. There is no retry, pool reuse,
forked Python state, global environment change or implicit scientific task.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def await_release(destination, gate):
    save(Path(destination)/"ready.json", dict(pid=os.getpid(), parent_pid=os.getppid(), time=time.time()))
    while not Path(gate).exists():
        time.sleep(.05)


def run_processes(jobs, gate, *, cwd):
    """Launch each declared job exactly once, release together, reap every child.

    Each job supplies argv and an existing unique output directory. Child stdout,
    stderr, PID, exit status and elapsed time are retained even on worker failure.
    Already launched jobs settle; failed jobs are never re-submitted.
    """
    outputs = [str(Path(j["output"]).resolve()) for j in jobs]
    if len(set(outputs)) != len(outputs):
        raise ValueError("OUTPUT_COLLISION")
    if Path(gate).exists():
        raise FileExistsError(gate)
    children, reports = [], []
    try:
        for job in jobs:
            out = Path(job["output"])
            stdout = (out/"stdout.log").open("x")
            try:
                stderr = (out/"stderr.log").open("x")
            except BaseException:
                stdout.close()
                raise
            try:
                p = subprocess.Popen(job["argv"], cwd=cwd, stdout=stdout, stderr=stderr)
            except BaseException:
                stdout.close()
                stderr.close()
                raise
            children.append((job, p, stdout, stderr, time.time()))
            save(out/"launch.json", dict(pid=p.pid, argv=job["argv"], fresh_interpreter=True))
        while not all((Path(j["output"])/"ready.json").exists() for j, *_ in children):
            if any(p.poll() is not None for _, p, *_ in children):
                raise RuntimeError("WORKER_EXITED_BEFORE_RELEASE")
            time.sleep(.05)
        save(gate, dict(release_time=time.time(), child_pids=[p.pid for _,p,*_ in children]))
        for job, p, _, _, started in children:
            code = p.wait()
            report = dict(case=job["case"], pid=p.pid, exit_code=code,
                          output=str(job["output"]), elapsed_seconds=time.time()-started)
            save(Path(job["output"])/"process.json", report)
            reports.append(report)
        return reports
    finally:
        # Only children owned by this invocation are touched, e.g. on interrupt
        # or startup failure. Ordinary case failures above settle without retry.
        for job, p, stdout, stderr, started in children:
            if p.poll() is None:
                p.terminate()
            p.wait()
            stdout.close()
            stderr.close()
            if not (Path(job["output"])/"process.json").exists():
                save(Path(job["output"])/"process.json", dict(case=job["case"], pid=p.pid,
                    exit_code=p.returncode, aborted=True, elapsed_seconds=time.time()-started))


def artificial_case(name, destination):
    """Two-qubit supplied algebraic Hamiltonians, no molecular generation."""
    import numpy as np
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import PauliEvolutionGate, RZZGate
    from qiskit.quantum_info import Operator, SparsePauliOp
    from .df_hamiltonian import DFHamiltonian
    from .h2_non_df_pilot import contract, evaluate_case
    from .h2_non_df_pilot_validation import validate_case, validate_df_primitives
    from .pauli_signal_adapter import PauliHamiltonian

    start = time.time()
    representation, method = name.rsplit("_", 1)
    h = (PauliHamiltonian.from_terms([("II", .1), ("ZI", .4), ("XZ", -.2), ("YY", .1)])
         if representation == "pauli_jw" else
         DFHamiltonian(.1, np.array([[.2,.03],[.03,-.1]]), np.array([.4,-.2]),
                       (np.array([[.5,.1],[.1,.2]]), np.diag([.1,.6])), {"basis":"artificial"}))
    plan = contract()
    plan["evaluation"].update(T=.2, q=2, PR_cost_samples=4)
    psi = np.array([0.,1.,1j,0.])/np.sqrt(2)
    c, a, evidence, result, rows = evaluate_case(h, psi, representation, method, plan, {"SOURCE":"artificial"})
    validation = validate_case(a, psi, c, result, plan)
    if representation == "df_native" and method == "PF":
        validate_df_primitives(a, plan)
    # Explicitly exercise both former exception paths in every synthetic worker.
    probe = QuantumCircuit(3)
    probe.append(PauliEvolutionGate(SparsePauliOp("XY"), time=.13).control(), [0,1,2])
    probe.append(RZZGate(.21).control(), [0,1,2])
    compiled = transpile(probe, basis_gates=plan["compiler"]["basis_gates"],
                         optimization_level=1, seed_transpiler=17)
    error = float(np.linalg.norm(Operator(compiled).data-Operator(probe).data, 2))
    if error > 1e-9:
        raise ValueError("ARTIFICIAL_CONTROL_PHASE")
    operator = Operator(a.trajectory(q=c.q, seed=result["circuit_cost"]["samples"][0]["seed"])).data
    semantic = dict(config=result["config"], signal=result["signal"], reference=result["reference"],
        normalization=result["normalization"], samples=result["circuit_cost"]["samples"],
        probe_RZ=compiled.count_ops().get("rz",0), operator_real=operator.real.tolist(),
        operator_imag=operator.imag.tolist())
    report = dict(status="PASS", artificial_only=True, pid=os.getpid(), started=start, ended=time.time(),
        semantic=semantic, validation=validation, probe_error=error,
        compiled_probability_error=result["circuit_cost"]["compiled_signal_max_error"])
    save(Path(destination)/"result.json", report)
    return report


def compare_artificial(sequential, parallel):
    import numpy as np
    checks = []
    for name in sequential:
        a, b = sequential[name]["semantic"], parallel[name]["semantic"]
        for field in ["config", "samples", "probe_RZ"]:
            if a[field] != b[field]:
                raise ValueError("ARTIFICIAL_COUNT_OR_SEED_MISMATCH:"+name+"/"+field)
        errors = {}
        for field in ["operator_real", "operator_imag"]:
            errors[field] = float(np.max(np.abs(np.array(a[field])-np.array(b[field]))))
        for field in ["corrected", "signed_bias", "raw"]:
            errors[field] = max(abs(a["signal"][field][x]-b["signal"][field][x]) for x in ["real","imag"])
        errors["log_B"] = abs(a["normalization"]["log_B"]-b["normalization"]["log_B"])
        errors["reference"] = max(abs(a["reference"]["signal"][x]-b["reference"]["signal"][x]) for x in ["real","imag"])
        if max(errors.values()) > 1e-12:
            raise ValueError("ARTIFICIAL_SEMANTIC_MISMATCH:"+name)
        checks.append(dict(case=name, errors=errors, gate_counts_and_seeds_identical=True, tolerance=1e-12))
    pids = {v["pid"] for v in parallel.values()}
    overlap = min(v["ended"] for v in parallel.values())-max(v["started"] for v in parallel.values())
    if len(pids) != 4 or overlap <= 0:
        raise ValueError("FOUR_WAY_CONCURRENT_PREFLIGHT_NOT_ESTABLISHED")
    return dict(status="PASS", cases=checks, distinct_worker_pids=sorted(pids), overlap_seconds=overlap,
                all_four_actually_concurrent=True, no_new_molecular_work=True)
