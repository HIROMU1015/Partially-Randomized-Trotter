"""Preparation default: saved bytes, JSON and NPY headers only. --run needs separate permission."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import time
import zipfile

import numpy as np
import b1_adapter as adapter

precision, task_a = adapter.precision, adapter.task_a
ROOT = adapter.ROOT
TASK_A = "a5304864ec5e1412a36c31574bcb95fb13f80202"
TASK_A_SOURCE = "4479a1c9cf150250254e2c2650f70fb43841fed1"
TASK_A_RESULT = "artifacts/rpe_m17_precision_isolation/2026-11-03/run01/RESULT.json"
TASK_A_STEPS = "artifacts/rpe_m17_precision_isolation/2026-11-03/run01/frozen_steps.npz"
EXTRA_INPUTS = ((TASK_A, TASK_A_RESULT, "e5c3cb1cd455d102c8a1e74d7556ec95b26ab79e72f0727e172f4483162f362e"),
                (TASK_A, TASK_A_STEPS, "2250e10bcae320c2fe1ec30662eb9ce6e14df50798f3db2d78f60c28dcfd7c30"))
Q, R = 131072, 32
PATHS = ("law", "reference")
TAIL_KEYS = {"law": "law_tail_mean", "reference": "reference_tail"}
STATUS = {"original_numeric_diagnostic_status": "INCONSISTENT", "u_num_bound": None,
          "RPE_tolerance_certified": False, "scientific_status": "UNDETERMINED", "mandatory_STOP": True}


def fixed_metadata():
    meta = task_a.fixed_metadata()  # Byte/hash/schema checks only; no np.load.
    for commit, path, expected in EXTRA_INPUTS:
        raw = (ROOT/path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected or raw != task_a.original_runner.fixed_bytes(commit, path):
            raise ValueError("Task A fixed input mismatch: "+path)
        meta["inputs"].append({"commit": commit, "path": path, "sha256": expected})
    for path in (Path(task_a.__file__), Path(precision.__file__)):
        if path.read_bytes() != task_a.original_runner.fixed_bytes(TASK_A_SOURCE, path.relative_to(ROOT).as_posix()):
            raise ValueError("Task A helper SOURCE changed")
    rte_path = "src/trotterlib/rte.py"
    if (ROOT/rte_path).read_bytes() != task_a.original_runner.fixed_bytes("4591b8d75a2a6a55c74a51225f0c61648023c8ee", rte_path):
        raise ValueError("Finite RTE carrier changed")
    prior = json.loads((ROOT/TASK_A_RESULT).read_text())
    if (prior["SOURCE"]["commit"] != TASK_A_SOURCE or prior["execution_status"] != "COMPLETED_MANDATORY_STOP"
            or not prior["fixed_factors"]["convergence"]["converged"]):
        raise ValueError("Task A baseline unavailable")
    meta["task_a"] = prior
    return meta


def describe():
    meta = fixed_metadata()
    return {"status": "PREPARED_NOT_SCIENTIFICALLY_EXECUTED", **STATUS, "inputs": meta["inputs"],
            "operator_headers": meta["operator_headers"], "precision_bits": meta["precision_bits"],
            "plan": "B-1a only; conditional replay-tail baseline if replay differs; no B-1b/B-2",
            "new_real_numeric_calls": 0, "max_future_HP_mean_calls": 12}


def load_arrays():
    with np.load(ROOT/task_a.OPERATORS_PATH, allow_pickle=False) as saved:
        arrays = {key: precision.validated_array(saved[key]) for key in task_a.OPERATOR_KEYS}
    with np.load(ROOT/task_a.STATE_PATH, allow_pickle=False) as saved:
        psi = precision.validated_array(saved["state"], vector=True)
    if psi.shape != (70,) or any(a.shape != (70, 70) for a in arrays.values()):
        raise ValueError("Fixed physical shape mismatch")
    for a in (*arrays.values(), psi):
        a.setflags(write=False)
    return arrays, psi


def jobs(mode, bits, arrays, psi, scalars, replay, paths=PATHS):
    return [{"mode": mode, "bits": b, "path": p, "q": Q, "psi": psi,
             "forward": arrays["forward"], "reverse": arrays["reverse"],
             "saved_tail": arrays[TAIL_KEYS[p]], "replay_tail": replay[TAIL_KEYS[p]],
             "A": arrays["A_probability" if p == "law" else "A_coefficient"], "scalars": scalars}
            for b in bits for p in paths]


def compact_pair(pair):
    return {p: {k: v for k, v in item.items() if k != "operators"} for p, item in pair.items()}


def run_ladder(mode, arrays, psi, scalars, replay, bits, pool, record, output, archive, paths=PATHS):
    def batch(selected_bits):
        pairs = {b: {} for b in selected_bits}
        futures = [pool.submit(adapter.evaluate_job, job) for job in jobs(mode, selected_bits, arrays, psi, scalars, replay, paths)]
        try:
            for future in as_completed(futures):
                item = future.result()
                name = f'{mode}/{item["path"]}/{item["bits"]}.json'
                archive.writestr(name, json.dumps(item["operators"], allow_nan=False))
                item["operators_archive_member"] = name
                pairs[item["bits"]][item["path"]] = item
                record["high_precision_jobs"].append({k: v for k, v in item.items() if k != "operators"})
                task_a.original_runner.write_result(output, record)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
        return pairs
    pairs = batch(bits[:2])
    check = adapter.convergence(pairs[bits[0]], pairs[bits[1]], Q)
    checks, used, best = [check], list(bits[:2]), pairs[bits[1]]
    if not check["converged"]:
        more = batch(bits[2:])
        best = more[bits[2]]
        check = adapter.convergence(pairs[bits[1]], best, Q)
        checks.append(check)
        used.append(bits[2])
    ctx = precision.scalar_context()
    tails = [adapter.restore_matrix(ctx, best[p]["operators"]["tail"]) for p in paths]
    return {"mode": mode, "precision_bits_used": used, "convergence_checks": checks,
            "convergence": check, "best_pair": compact_pair(best),
            "tail_law_vs_reference_frobenius": ctx.nstr(ctx.norm(tails[0]-tails[1]), 80) if len(tails) == 2 else None}


def source_metadata():
    paths = [Path(__file__), Path(adapter.__file__)]
    for path in paths:
        if path.read_bytes() != task_a.original_runner.fixed_bytes("HEAD", path.relative_to(ROOT).as_posix()):
            raise ValueError("Freeze this scientific SOURCE before --run")
    return {"commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]).decode().strip(),
            "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}


def execute(output):
    source, meta = source_metadata(), fixed_metadata()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    record = {"execution_status": "STARTED", **STATUS, "SOURCE": source, "inputs": meta["inputs"],
              "high_precision_jobs": [], "stage": "load_fixed_arrays", "original_fixed_literals": meta["old_result"]["fixed_literals"],
              "sampling_compile_H_state_DF_component_generation": 0}
    try:
        task_a.original_runner.write_result(output, record)
        arrays, psi = load_arrays()
        record["stage"] = "float_SOURCE_replay"
        scalars = adapter.replay_scalars(meta["old_result"]["context"]["distribution"], R)
        record["scalar_trace"] = adapter.scalar_trace(scalars)
        replay = adapter.float_replay(arrays, scalars)
        if any(not np.isfinite(a).all() for a in replay.values()):
            raise ValueError("Nonfinite float SOURCE replay")
        np.savez_compressed(output/"float_source_replay.npz", **replay)
        record["float_replay_comparison"] = {key: {"array_equal": bool(np.array_equal(replay[key], arrays[key])),
            "frobenius": float(np.linalg.norm(replay[key]-arrays[key], "fro"))}
            for key in ("law_event_mean", "law_tail_mean", "reference_tail")}
        changed_paths = tuple(p for p in PATHS if not record["float_replay_comparison"][TAIL_KEYS[p]]["array_equal"])
        prior = meta["task_a"]["fixed_factors"]
        record["task_a_saved_factor_baseline"] = prior  # Literal reuse; never recalculate old-tail means.
        available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
        record["parallelism"] = {"workers": min(available, 4), "available_CPUs": available, "start_method": "spawn",
            "numpy_version": np.__version__, "BLAS_environment": {n: os.environ.get(n) for n in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}}
        task_a.original_runner.write_result(output, record)
        with zipfile.ZipFile(output/"high_precision_operators.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
            with ProcessPoolExecutor(max_workers=min(available, 4), mp_context=multiprocessing.get_context("spawn")) as pool:
                baseline = prior
                record["baseline_origin"] = "saved Task A (float replay tails identical)"
                if changed_paths:
                    record["stage"] = "conditional_replay_tail_baseline"
                    replay_stage = run_ladder("replay_tail_baseline", arrays, psi, scalars, replay, meta["precision_bits"], pool, record, output, archive, changed_paths)
                    record["replay_tail_baseline"] = replay_stage
                    baseline = {"best_pair": {**prior["best_pair"], **replay_stage["best_pair"]}, "convergence": replay_stage["convergence"]}
                    record["baseline_origin"] = "changed paths use new SOURCE replay tails, not original saved tails; unchanged paths reuse Task A"
                    record["baseline_recomputed_paths"] = changed_paths
                    record["replay_vs_task_a_mean"] = {p: precision.complex_difference(baseline["best_pair"][p]["mean"], prior["best_pair"][p]["mean"]) for p in PATHS}
                if baseline["convergence"]["converged"]:
                    record["stage"] = "b1a"
                    stage = run_ladder("b1a", arrays, psi, scalars, replay, meta["precision_bits"], pool, record, output, archive)
                    record["b1a"] = stage
                    if stage["convergence"]["converged"]:
                        record["b1a_diagnostic"] = precision.gap_relation(stage["best_pair"])
                        record["construction_mean_change"] = {p: precision.complex_difference(stage["best_pair"][p]["mean"], baseline["best_pair"][p]["mean"]) for p in PATHS}
                        record["b1a_vs_task_a_mean"] = {p: precision.complex_difference(stage["best_pair"][p]["mean"], prior["best_pair"][p]["mean"]) for p in PATHS}
                        relation = record["b1a_diagnostic"]["relation_to_original_atol"]
                        record["candidate_labels"] = (["CONSTRUCTION_ARITHMETIC_DEPENDENCE_SUPPORTED"] if relation == "BELOW" else
                                                      ["FROZEN_INPUT_OR_DEFINITION_DIFFERENCE_POSSIBLE"] if relation == "ABOVE" else ["INSUFFICIENT_EVIDENCE"])
                        record["interpretation"] = "Tail-construction arithmetic changed as a group; scalar versus matrix causes are not uniquely separated. Frozen input representation/definitions remain possible."
                    else:
                        record["interpretation"] = "INSUFFICIENT_EVIDENCE: B-1a did not converge by 224 bit"
                else:
                    record["interpretation"] = "INSUFFICIENT_EVIDENCE: replay-tail baseline did not converge; B-1a skipped"
        record["output_sha256"] = {name: hashlib.sha256((output/name).read_bytes()).hexdigest()
                                   for name in ("float_source_replay.npz", "high_precision_operators.zip")}
        record["execution_status"], record["stage"] = "COMPLETED_MANDATORY_STOP", "completed"
    except BaseException as exc:
        record["execution_status"] = "INTERRUPTED_OR_INVALID_MANDATORY_STOP"
        record["error"] = {"type": type(exc).__name__, "message": str(exc).replace(str(ROOT), "<repository>")}
        raise
    finally:
        record["wall_seconds"] = time.monotonic()-started
        task_a.original_runner.write_result(output, record)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--run", action="store_true", help="Separately permitted B-1a scientific evaluation")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if not args.run:
        print(json.dumps(describe(), ensure_ascii=False, indent=2))
    elif args.output_dir is None:
        parser.error("--run requires a fresh --output-dir")
    else:
        execute(args.output_dir)


if __name__ == "__main__":
    main()
