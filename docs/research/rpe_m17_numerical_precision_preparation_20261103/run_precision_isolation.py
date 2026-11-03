"""Task A only. Default --describe checks bytes/headers, never real arrays."""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
import zipfile

import numpy as np
import precision_adapter as adapter

ROOT = Path(__file__).resolve().parents[3]
ORIGINAL = "cd749066904d3f29b28b03c19ab69370372b157f"
ORIGINAL_SOURCE = "128e9a5bf2630cd0333ec5f69b81b25def78eb79"
B_COMMIT = "e22d7718c47358ff650dbc04e8e72f6ab1c8bf41"
RESULT_PATH = "artifacts/rpe_m17_mean_validation/2026-11-03/run01/RESULT.json"
OPERATORS_PATH = "artifacts/rpe_m17_mean_validation/2026-11-03/run01/operators.npz"
STATE_PATH = "artifacts/rpe_fixed_request_measurement_b/2026-11-03/run01/physical_state.npz"
INPUTS = ((ORIGINAL, RESULT_PATH, "79fdf7e84f90c48230857b13f39f9becb30f19892e2e9c09d80805d346432a4a"),
          (ORIGINAL, OPERATORS_PATH, "e1a65238e9d9b61a6d0aa542ca7f509827b0a761fc751615f1f2018a0e7020df"),
          (B_COMMIT, STATE_PATH, "793300c86417f52569289ce86980c5f9a87292e130199355a0f4f7056225c9c4"))
OPERATOR_KEYS = {"A_probability", "A_coefficient", "forward", "reverse", "reference_tail", "law_event_mean", "law_tail_mean"}
Q = 131072

sys.path.insert(0, str(ROOT/"docs/research/rpe_m17_mean_validation_preparation_20261103"))
import run_m17_mean_validation as original_runner
if Path(original_runner.__file__).resolve() != ROOT/"docs/research/rpe_m17_mean_validation_preparation_20261103/run_m17_mean_validation.py":
    raise ValueError("Wrong original helper module")


def archive_headers(path):
    headers = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            with archive.open(name) as stream:
                if stream.read(6) != b"\x93NUMPY":
                    raise ValueError("Invalid NPY member")
                version = tuple(stream.read(2))
                if version not in ((1, 0), (2, 0), (3, 0)):
                    raise ValueError("Unsupported NPY header")
                length = struct.unpack("<H" if version == (1, 0) else "<I", stream.read(2 if version == (1, 0) else 4))[0]
                headers[name.removesuffix(".npy")] = ast.literal_eval(stream.read(length).decode("utf-8" if version == (3, 0) else "latin1"))
    return headers


def fixed_metadata():
    inputs = []
    for commit, path, expected in INPUTS:
        raw = (ROOT/path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected or raw != original_runner.fixed_bytes(commit, path):
            raise ValueError("Fixed input byte mismatch: "+path)
        inputs.append({"commit": commit, "path": path, "sha256": expected})
    source_path = "docs/research/rpe_m17_mean_validation_preparation_20261103/run_m17_mean_validation.py"
    if (ROOT/source_path).read_bytes() != original_runner.fixed_bytes(ORIGINAL_SOURCE, source_path):
        raise ValueError("Reused helper SOURCE changed")
    headers = archive_headers(ROOT/OPERATORS_PATH)
    if set(headers) != OPERATOR_KEYS or any(h["shape"] != (70, 70) or h["descr"] != "<c16" for h in headers.values()):
        raise ValueError("Original operator schema mismatch")
    original = json.loads((ROOT/RESULT_PATH).read_text())
    if original["SOURCE"]["commit"] != ORIGINAL_SOURCE or original["evaluation"]["numeric_diagnostic_status"] != "INCONSISTENT":
        raise ValueError("Original result identity/status mismatch")
    literals = original["fixed_literals"]
    if tuple(literals[k] for k in ("q", "r", "K", "delta")) != ("131072", "32", "2", "0.02"):
        raise ValueError("Fixed physical tuple mismatch")
    return {"inputs": inputs, "operator_headers": headers, "old_result": original,
            "precision_bits": adapter.precision_ladder(70, Q)}


def describe():
    meta = fixed_metadata()
    return {"status": "PREPARED_NOT_SCIENTIFICALLY_EXECUTED", "inputs": meta["inputs"],
            "operator_headers": meta["operator_headers"], "precision_bits": meta["precision_bits"],
            "modes": ["fixed_step", "fixed_factors only if the converged fixed-step gap remains above 1e-10"],
            "q": Q, "original_numeric_diagnostic_status": "INCONSISTENT", "u_num_bound": None,
            "RPE_tolerance_certified": False, "scientific_status": "UNDETERMINED", "mandatory_STOP": True}


def load_arrays():
    """First real numeric payload access: only after separate --run permission."""
    with np.load(ROOT/OPERATORS_PATH, allow_pickle=False) as saved:
        arrays = {key: adapter.validated_array(saved[key]) for key in ("forward", "reverse", "law_tail_mean", "reference_tail")}
    with np.load(ROOT/STATE_PATH, allow_pickle=False) as saved:
        psi = adapter.validated_array(saved["state"], vector=True)
    if psi.shape != (70,):
        raise ValueError("Fixed physical state dimension mismatch")
    for array in (*arrays.values(), psi):
        array.setflags(write=False)
    factors = {path: {"forward": arrays["forward"], "reverse": arrays["reverse"], "tail": arrays[key]}
               for path, key in (("law", "law_tail_mean"), ("reference", "reference_tail"))}
    return factors, psi


def build_jobs(mode, bits, factors, psi, steps):
    return [{"mode": mode, "path": path, "bits": precision, "q": Q, "psi": psi,
             **({"step": steps[path]} if mode == "fixed_step" else factors[path])}
            for precision in bits for path in ("law", "reference")]


def evaluate_batch(jobs, pool, record, output):
    futures = [pool.submit(adapter.evaluate_job, job) for job in jobs]
    completed = []
    try:
        for future in as_completed(futures):
            item = future.result()
            completed.append(item)
            record["high_precision_jobs"].append(item)
            original_runner.write_result(output, record)
    except BaseException:
        for future in futures:
            future.cancel()
        raise
    return completed


def run_ladder(mode, factors, psi, steps, bits, pool, record, output):
    results = evaluate_batch(build_jobs(mode, bits[:2], factors, psi, steps), pool, record, output)
    by_bits = {b: {r["path"]: r for r in results if r["bits"] == b} for b in bits[:2]}
    check = adapter.convergence(by_bits[bits[0]], by_bits[bits[1]])
    checks = [check]
    best = by_bits[bits[1]]
    if not check["converged"]:
        more = evaluate_batch(build_jobs(mode, bits[2:], factors, psi, steps), pool, record, output)
        best = {r["path"]: r for r in more}
        check = adapter.convergence(by_bits[bits[1]], best)
        checks.append(check)
    return {"mode": mode, "precision_bits_used": list(bits[:len(checks)+1]),
            "convergence_checks": checks, "convergence": check, "best_pair": best}


def source_metadata():
    paths = [Path(__file__).resolve(), Path(adapter.__file__).resolve()]
    for path in paths:
        if path.read_bytes() != original_runner.fixed_bytes("HEAD", path.relative_to(ROOT).as_posix()):
            raise ValueError("Freeze this SOURCE version before --run")
    return {"commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]).decode().strip(),
            "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}


def execute(output):
    source = source_metadata()
    meta = fixed_metadata()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    record = {"execution_status": "STARTED", "SOURCE": source, "mandatory_STOP": True,
              "original_numeric_diagnostic_status": "INCONSISTENT", "u_num_bound": None,
              "RPE_tolerance_certified": False, "scientific_status": "UNDETERMINED", "high_precision_jobs": [],
              "inputs": meta["inputs"], "precision_bits": meta["precision_bits"], "q": Q,
              "original_fixed_literals": meta.get("old_result", {}).get("fixed_literals"),
              "stage": "load_fixed_arrays", "sampling_compile_H_state_DF_generation": 0}
    try:
        original_runner.write_result(output, record)
        factors, psi = load_arrays()
        record["stage"] = "float_baseline"
        baseline, steps = adapter.float_baseline(factors, psi, Q)
        record["baseline"] = baseline
        old_means = meta["old_result"]["evaluation"]["means"]
        drift = {path: original_runner.residual(complex(baseline[path]['real'],baseline[path]['imag']),
                    complex(old_means[path]['real'],old_means[path]['imag'])) for path in ("law", "reference")}
        reproduced = all(v["complex_abs"] <= adapter.DIAGNOSTIC_ATOL for v in drift.values())
        record["baseline_drift"] = drift
        record["baseline_reproduced"] = reproduced
        np.savez_compressed(output/"frozen_steps.npz", **steps)
        record["frozen_steps_npz_sha256"] = hashlib.sha256((output/"frozen_steps.npz").read_bytes()).hexdigest()
        if not reproduced:
            record["interpretation"] = {"candidate_labels": ["INSUFFICIENT_EVIDENCE"], "reason": "Float baseline did not reproduce original means within original 1e-10"}
        else:
            available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
            workers = min(available, 4)  # Four independent path/precision jobs, not a CPU cap.
            record["parallelism"] = {"workers": workers, "available_CPUs": available, "start_method": "spawn",
                "numpy_version": np.__version__, "mpmath_version": importlib.metadata.version("mpmath"),
                "BLAS_environment": {n: os.environ.get(n) for n in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}}
            record["stage"] = "fixed_step"
            original_runner.write_result(output, record)
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                step = run_ladder("fixed_step", factors, psi, steps, meta["precision_bits"], pool, record, output)
                record["fixed_step"] = step
                factor = None
                if step["convergence"]["converged"] and adapter.gap_relation(step["best_pair"])["relation_to_original_atol"] == "ABOVE":
                    record["stage"] = "fixed_factors"
                    original_runner.write_result(output, record)
                    factor = run_ladder("fixed_factors", factors, psi, steps, meta["precision_bits"], pool, record, output)
                    record["fixed_factors"] = factor
            record["interpretation"] = adapter.interpret(reproduced, baseline, step, factor)
        record["execution_status"] = "COMPLETED_MANDATORY_STOP"
        record["stage"] = "completed"
    except BaseException as exc:
        record["execution_status"] = "INTERRUPTED_OR_INVALID_MANDATORY_STOP"
        record["error"] = {"type": type(exc).__name__, "message": str(exc).replace(str(ROOT), "<repository>")}
        raise
    finally:
        record["wall_seconds"] = time.monotonic()-started
        original_runner.write_result(output, record)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--run", action="store_true", help="New numerical scientific evaluation after separate user permission")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if not args.run:
        print(json.dumps(describe(), ensure_ascii=False, indent=2))
        return
    if args.output_dir is None:
        parser.error("--run requires fresh --output-dir")
    execute(args.output_dir)


if __name__ == "__main__":
    main()
