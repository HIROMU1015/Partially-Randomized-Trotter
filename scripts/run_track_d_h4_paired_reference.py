#!/usr/bin/env python3
"""Stage A only: saved 32 paths, all prechecks, then <=128 P0 wrappers, STOP."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import importlib.util
from importlib.metadata import version
import json
import multiprocessing
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
ORIGIN = "ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7"
SAVED = "3649b75af191916daf44abb91e14d34c43b83e8f"
STATIC = "523efe89fde52a14a8e0a4eba4ea6b70e3f512e1"
OLD_RUN = "scripts/run_track_d_h4_reference.py"
NEW = ("src/trottertracks/track_d_h4_paired_reference.py", __file__.split(str(ROOT)+"/")[-1],
       "tests/check_track_d_h4_paired_reference.py",
       "docs/research/track_d_h4_paired32_reference_20261106/PLAN.md")
CONTRACT = "artifacts/research_c_fixed_word_pilot/2026-11-04/run01/contract.json"
MANIFEST = "artifacts/research_c_fixed_word_pilot/2026-11-04/run01/SOURCE.json"
LAW = "artifacts/research_c_near_practical_r32/2026-11-03/run01/PR/A/law.json"
METRICS = ("rz_count", "cx_count", "rz_depth", "cx_depth", "total_depth", "qubit_count", "circuit_size")
_STOP = None
_CONTEXT = None


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def old_runner():
    if "paired_old_runner" not in sys.modules:
        load_module("paired_old_runner", ROOT / OLD_RUN)
    return sys.modules["paired_old_runner"]


def write(path, value):
    old_runner().write(path, value)


def preflight(origin, saved):
    """Byte/metadata checks only; no scientific module imported."""
    source = git("rev-parse", "HEAD").decode().strip()
    old = old_runner()
    files = tuple(dict.fromkeys((*NEW, OLD_RUN, *old.NEW_FILES, *old.REUSED)))
    for path in files:
        if (ROOT / path).read_bytes() != git("show", source + ":" + path):
            raise ValueError("UNCOMMITTED_EXECUTION_SOURCE:" + path)
    c = json.loads((ROOT / CONTRACT).read_text())
    manifest = json.loads((ROOT / MANIFEST).read_text())
    for label, table in (("SOURCE", manifest["files"]), ("INPUT", c["saved_inputs"])):
        for path, expected in table.items():
            if digest((origin / path).read_bytes()) != expected:
                raise ValueError(label + ":" + path)
    if (origin / LAW).read_bytes() != git("show", ORIGIN + ":" + LAW):
        raise ValueError("SAVED_LAW_BINDING")
    packages = {p: version(p) for p in c["packages"]}
    if packages != c["packages"]:
        raise ValueError("PINNED_PACKAGE_VERSIONS")
    bindings = json.loads((saved / "identity.json").read_text())
    expected = {f"{kind}/sample{i:02d}{suffix}" for i in range(32)
                for kind, suffix in (("samples", "_events.json"), ("references", ".npz"), ("circuits", "_raw.qpy"))}
    if set(bindings) != expected:
        raise ValueError("EXACT_SAVED_32_BINDINGS_REQUIRED")
    for leaf, item in bindings.items():
        data = (saved / leaf).read_bytes()
        if (item["commit"] != SAVED or data != git("show", SAVED + ":" + item["path"])
                or digest(data) != item["sha256"]):
            raise ValueError("SAVED_RESULT:" + leaf)
    return c, dict(execution_source=source, static_report=STATIC, original_science_source=ORIGIN,
        saved_result_carrier=SAVED, original_source_files=manifest["files"],
        original_saved_inputs=c["saved_inputs"], extra_saved_law=dict(path=LAW, sha256=digest((origin / LAW).read_bytes())),
        new_and_reused={p: digest((ROOT / p).read_bytes()) for p in files},
        saved_cohort_bindings=bindings, packages=packages, python=sys.version.split()[0])


def initialize(stop):
    global _STOP
    _STOP = stop


def guard():
    if _STOP.is_set():
        raise RuntimeError("CANCELLED_AFTER_OTHER_PATH_FAILURE")


def precheck_path(task):
    global _CONTEXT
    sample, origin, saved, output, threads = task
    directory = Path(output) / f"sample{sample:02d}"
    started = time.monotonic(); records = []
    try:
        guard()
        directory.mkdir(); (directory / "circuits").mkdir()
        os.environ["MPLCONFIGDIR"] = str(Path(output) / "matplotlib-cache" / str(os.getpid()))
        old = old_runner(); base = old.bootstrap(origin)
        from trottertracks import track_d_h4_paired_reference as extension
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            if _CONTEXT is None:
                c = json.loads((ROOT / CONTRACT).read_text())
                _CONTEXT = extension.prepare_common(origin, c)
            guard()
            bundle = extension.prepare_path(_CONTEXT, saved, sample)
            write(directory / "conditions.json", bundle.identity)
            write(directory / "fusion.json", bundle.fusion)
            for (arm, axis), circuit in bundle.wrappers.items():
                base.save_qpy(directory / "circuits" / f"{arm}_{axis}_raw.qpy", circuit)
            for name, circuit in (("A", bundle.later), ("B", bundle.earlier), ("A_actual_adjoint", bundle.later_adjoint)):
                base.save_qpy(directory / "circuits" / (name + ".qpy"), circuit)
            checks = old.load_file("paired_h4_checks", ROOT / "tests/track_d_h4_reference_checks.py")
            guard()
            report = extension.run_prechecks(bundle, checks, directory, records, threads)
        report.update(sample=sample, wall_seconds=time.monotonic()-started,
                      peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write(directory / "precompile_checks.json", report)
        return dict(sample=sample, status="PASS", checks=len(records), wall_seconds=report["wall_seconds"])
    except BaseException as error:
        _STOP.set()
        result = dict(sample=sample, status="FAIL", error=repr(error),
                      traceback=traceback.format_exc(), completed_checks=records)
        directory.mkdir(exist_ok=True)
        write(directory / "precompile_checks.json", result)
        return result


def compile_verify_path(task):
    sample, origin, saved, output, threads = task
    directory = Path(output) / f"sample{sample:02d}"
    started = time.monotonic(); rows = {}
    try:
        guard()
        old = old_runner(); base = old.bootstrap(origin)
        from threadpoolctl import threadpool_limits
        import numpy as np
        from trottertracks.c1_stage3_molecular import component_check
        tasks = [(origin, str(directory), arm, axis, threads if arm == "reference" else 1)
                 for arm in ("symmetric_directional", "reference") for axis in ("X", "Y")]
        for item in tasks:
            guard()
            with threadpool_limits(limits=1):
                old.compile_one(item)
        for item in tasks:
            guard()
            key, row = old.verify_one(item)
            rows[key] = row
        paired = {}
        for axis in ("X", "Y"):
            with np.load(directory / f"reference_{axis}_observable.npz", allow_pickle=False) as z:
                reference = z["observable"]
            with np.load(directory / f"symmetric_directional_{axis}_observable.npz", allow_pickle=False) as z:
                symmetric = z["observable"]
            row = component_check(reference, symmetric, base.ATOL, label="compiled_paired_" + axis)
            if row["frobenius"] > base.SIGNAL_ATOL:
                raise ValueError("COMPILED_PAIRED_OBSERVABLE_SIGNAL_ALLOWANCE")
            paired[axis] = row
        result = dict(sample=sample, status="PASS", compiled=rows, paired_observables=paired,
                      wall_seconds=time.monotonic()-started)
        write(directory / "result.json", result)
        return result
    except BaseException as error:
        _STOP.set()
        result = dict(sample=sample, status="FAIL", error=repr(error), traceback=traceback.format_exc(), compiled=rows)
        write(directory / "result.json", result)
        return result


def stage(pool, function, tasks, output, name, stop):
    """Bound pending work to active workers; a failure prevents new submissions."""
    iterator = iter(tasks); futures = {}; results = {}
    for _ in range(pool._max_workers):
        task = next(iterator, None)
        if task is not None:
            futures[pool.submit(function, task)] = task[0]
    while futures:
        future = next(as_completed(futures)); sample = futures.pop(future)
        row = future.result(); results[sample] = row
        write(output / "progress.json", dict(stage=name, completed=sorted(results),
              pass_count=sum(r["status"] == "PASS" for r in results.values()), failure_observed=stop.is_set()))
        print(json.dumps(dict(stage=name, sample=sample, status=row["status"], completed=len(results))), flush=True)
        if not stop.is_set():
            task = next(iterator, None)
            if task is not None:
                futures[pool.submit(function, task)] = task[0]
    return results


def summarize(rows, output):
    """Descriptive finite-cohort paired values; no population CI or RPE shots."""
    table = []
    for sample in range(32):
        for axis in ("X", "Y"):
            cells = rows[sample]["compiled"]
            symmetric = cells["symmetric_directional_" + axis]["metrics"]
            reference = cells["reference_" + axis]["metrics"]
            row = dict(sample=sample, axis=axis)
            for metric in METRICS:
                row["symmetric_" + metric] = symmetric[metric]
                row["reference_" + metric] = reference[metric]
                row["delta_" + metric] = reference[metric] - symmetric[metric]
            table.append(row)
    with (output / "paired_resources.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    summary = {}
    for axis in ("X", "Y"):
        subset = [r for r in table if r["axis"] == axis]; summary[axis] = {}
        for metric in METRICS:
            differences = [r["delta_" + metric] for r in subset]
            summary[axis][metric] = dict(n_paths=32, symmetric_mean=statistics.mean(r["symmetric_" + metric] for r in subset),
                reference_mean=statistics.mean(r["reference_" + metric] for r in subset),
                delta_mean=statistics.mean(differences), delta_min=min(differences), delta_max=max(differences),
                delta_cohort_population_sd=statistics.pstdev(differences),
                reference_lower=sum(x < 0 for x in differences), equal=sum(x == 0 for x in differences),
                reference_higher=sum(x > 0 for x in differences))
    write(output / "finite_cohort_summary.json", dict(estimand="saved finite cohort only; X/Y paired, not 64 independent paths",
          axes=summary, population_expectation=None, confidence_interval=None, RPE_total=None))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("origin", "saved", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(); origin = args.origin.resolve(); saved = args.saved.resolve(); output = args.output.resolve()
    sys.dont_write_bytecode = True
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["QISKIT_PARALLEL"] = "false"; os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    _, identity = preflight(origin, saved)
    if not args.execute:
        print(json.dumps(dict(status="STATIC_PREFLIGHT_PASS", source=identity["execution_source"], scientific_execution=False)))
        return 0
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); cpus = len(os.sched_getaffinity(0))
    mem = {k: int(v.split()[0])*1024 for k, v in (line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
           if k in ("MemTotal", "MemAvailable")}
    # Old observed Reference worker peak is about 6 GiB; use a 1.5x planning
    # estimate, not an OS/resource limit. Share available CPU across paths.
    workers = min(32, max(1, cpus//4), max(1, mem["MemAvailable"]//(9*1024**3)))
    threads = max(1, cpus//workers)
    result = dict(status="RUNNING", source_commit=identity["execution_source"],
        samples=32, maximum_wrapper_compiles=128, sampling_calls=0, ground_solves=0, DF_refits=0,
        quantum_shots=0, RPE_shot_evaluations=0, RPE_rounds_evaluated=0, retry=0, resume=0,
        whole_law_qualified=False, RPE_total_cost_evaluated=False, scientific_GO=False, mandatory_stop=True,
        resources=dict(available_cpus=cpus, memory_before_bytes=mem, path_workers=workers, numerical_threads_per_path=threads,
                       small_matrix_BLAS_threads=1, resource_caps=None, dense_17qubit_Operator=False))
    write(output / "SOURCE.json", identity)
    context = multiprocessing.get_context("spawn"); stop = context.Event(); code = 1
    tasks = [(i, str(origin), str(saved), str(output), threads) for i in range(32)]
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=initialize, initargs=(stop,)) as pool:
            pre = stage(pool, precheck_path, tasks, output, "precompile_correctness", stop)
            result["precompile"] = pre
            if stop.is_set() or len(pre) != 32:
                raise ValueError("PRECOMPILE_STAGE_FAIL_NO_COMPILES")
            compiled = stage(pool, compile_verify_path, tasks, output, "compile_and_postcheck", stop)
            result["compiled_paths"] = compiled
            if stop.is_set() or len(compiled) != 32:
                raise ValueError("COMPILE_OR_POSTCHECK_FAIL")
        result["finite_cohort"] = summarize(compiled, output)
        _, after = preflight(origin, saved)
        if after != identity:
            raise ValueError("SOURCE_INPUT_CHANGED_DURING_EXECUTION")
        result.update(status="TECHNICAL_PASS_MANDATORY_STOP", source_input_preserved=True)
        code = 0
    except BaseException as error:
        stop.set(); result.update(status="FAIL_MANDATORY_STOP", error=repr(error), traceback=traceback.format_exc())
    finally:
        result.update(compile_calls_started=len(list(output.glob("sample*/*_compile.json"))),
            wall_seconds=time.monotonic()-started, parent_CPU_seconds=time.process_time(),
            parent_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write(output / "result.json", result)
        write(output / "STOP.json", dict(status=result["status"], mandatory_stop=True, retry=0, resume=0,
             compile_calls_started=result["compile_calls_started"], next_stage_authorized=False))
        inventory = {str(p.relative_to(output)):dict(bytes=p.stat().st_size, sha256=digest(p.read_bytes()))
                     for p in output.rglob("*") if p.is_file() and "matplotlib-cache" not in p.parts}
        write(output / "inventory.json", inventory)
    print(json.dumps({k:result[k] for k in ("source_commit", "status", "compile_calls_started", "wall_seconds")}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
