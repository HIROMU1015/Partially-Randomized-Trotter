#!/usr/bin/env python3
"""One fixed H4 word/cut: correctness, then six full-wrapper compiles and STOP."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import importlib.util
from importlib.metadata import version
import json
import multiprocessing
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
ORIGIN = "ba0e2bf2d1d6f0dc78616d3392f3ff5b6ba7d5b7"
RESULT = "3649b75af191916daf44abb91e14d34c43b83e8f"
STATIC = "a9a74b1386ba713ed3648658dd94be4d992a492e"
NEW_FILES = ("src/trottertracks/track_d_h4_reference.py", "scripts/run_track_d_h4_reference.py",
    "tests/track_d_h4_reference_checks.py", "tests/check_track_d_h4_reference_adapter.py",
    "docs/research/track_d_h4_reference_validation_20261106/PLAN.md")
REUSED = ("src/trotterlib/track_d_minimal_reference.py", "src/trotterlib/native_df_gate_model.py",
    "src/trottertracks/resource_applicability/ax2b_native_df_v5.py",
    "src/trottertracks/resource_applicability/ax2b_stream_fingerprint_v5.py",
    "src/trottertracks/resource_applicability/ax2a_control_plan.py")
CONTRACT = "artifacts/research_c_fixed_word_pilot/2026-11-04/run01/contract.json"


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def sha(data): return hashlib.sha256(data).hexdigest()


def write(path, value):
    path = Path(path); temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    temp.replace(path)


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module); return module


def bootstrap(origin):
    # trottertracks is a namespace package: original C modules win; the new
    # adapter and reused AX2B helpers come from this frozen Track D branch.
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(Path(origin)/"src"), str(ROOT/"src")]
    import trotterlib
    extra = str(ROOT/"src/trotterlib")
    if extra not in trotterlib.__path__: trotterlib.__path__.append(extra)
    from trottertracks import track_d_h4_reference as m
    return m


def preflight(origin, saved):
    source = git("rev-parse", "HEAD").decode().strip()
    if subprocess.check_output(["git", "-C", str(origin), "rev-parse", "HEAD"]).decode().strip() != ORIGIN:
        raise ValueError("WRONG_ORIGINAL_SOURCE")
    for p in (*NEW_FILES, *REUSED):
        if (ROOT/p).read_bytes() != git("show", source+":"+p):
            raise ValueError("UNCOMMITTED_EXECUTION_SOURCE:"+p)
    c = json.loads((ROOT/CONTRACT).read_text())
    original_manifest = json.loads((ROOT/"artifacts/research_c_fixed_word_pilot/2026-11-04/run01/SOURCE.json").read_text())
    for label, table in (("SOURCE", original_manifest["files"]), ("INPUT", c["saved_inputs"])):
        for p, want in table.items():
            if sha((origin/p).read_bytes()) != want: raise ValueError(label+":"+p)
    for p, want in c["packages"].items():
        if version(p) != want: raise ValueError("ENVIRONMENT:"+p)
    bindings = json.loads((saved/"identity.json").read_text())
    for leaf, item in bindings.items():
        data = (saved/leaf).read_bytes()
        if item["commit"] != RESULT or data != git("show", RESULT+":"+item["path"]) or sha(data) != item["sha256"]:
            raise ValueError("SAVED_RESULT:"+leaf)
    return c, dict(execution_source=source, static_report=STATIC, original_science_source=ORIGIN,
        original_result=RESULT, new_and_reused={p:sha((ROOT/p).read_bytes()) for p in (*NEW_FILES,*REUSED)},
        original_source_files=original_manifest["files"], original_saved_inputs=c["saved_inputs"],
        saved_word_bindings=bindings, packages={p:version(p) for p in c["packages"]})


def compile_one(task):
    started = time.monotonic()
    origin, output, arm, axis, threads = task
    m = bootstrap(origin)
    from trottertracks.c1_h4_native_fair import compile_checked
    from trottertracks.pr_dictionary_validation.fixture import CompilerProfile
    raw = m.load_qpy(Path(output)/"circuits"/(arm+"_"+axis+"_raw.qpy"))
    profile = CompilerProfile(); profile.validate()
    write(Path(output)/(arm+"_"+axis+"_compile.json"), dict(status="STARTED", arm=arm, axis=axis, compile_calls=1))
    compiled = compile_checked(raw, profile)  # exactly one P0 transpile here
    m.save_qpy(Path(output)/"circuits"/(arm+"_"+axis+"_compiled.qpy"), compiled)
    write(Path(output)/(arm+"_"+axis+"_compile.json"), dict(status="PASS", arm=arm, axis=axis,
        compile_calls=1, compiler=asdict(profile), wall_seconds=time.monotonic()-started))
    return arm, axis


def verify_one(task):
    started = time.monotonic()
    origin, output, arm, axis, threads = task
    m = bootstrap(origin)
    import numpy as np
    from threadpoolctl import threadpool_limits
    checks = load_file("track_d_h4_checks", ROOT/"tests/track_d_h4_reference_checks.py")
    compiled = m.load_qpy(Path(output)/"circuits"/(arm+"_"+axis+"_compiled.qpy"))
    with np.load(Path(output)/"references.npz", allow_pickle=False) as archive:
        arrays = {k:archive[k] for k in archive.files}
    with threadpool_limits(limits=threads):
        obs, report = checks.check_compiled(compiled, arm, axis, arrays, m, threads)
    np.savez_compressed(Path(output)/(arm+"_"+axis+"_observable.npz"), observable=obs)
    report.update(arm=arm, axis=axis, wall_seconds=time.monotonic()-started,
        peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        metrics=m.artificial.metrics(compiled), state_action_threads=threads)
    write(Path(output)/(arm+"_"+axis+"_check.json"), report)
    return arm+"_"+axis, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", type=Path, required=True)
    parser.add_argument("--saved", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    origin, saved, output = args.origin.resolve(), args.saved.resolve(), args.output.resolve()
    # Process-local pools only. Independent compile workers use one BLAS thread;
    # numerical phases select their task-specific parallel budget below.
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["QISKIT_PARALLEL"] = "false"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["MPLCONFIGDIR"] = str(output/"matplotlib-cache")
    c, identity = preflight(origin, saved)
    output.mkdir(parents=True, exist_ok=False)  # one shot, never retry/resume
    (output/"circuits").mkdir()
    write(output/"SOURCE.json", identity)
    started = time.monotonic(); records = []
    result = dict(status="RUNNING", source_commit=identity["execution_source"],
        precompile_checks=dict(status="RUNNING", checks=records), compile_calls_started=0, compile_tasks_submitted=0,
        sampling_calls=0, ground_solves=0, DF_refits=0, shots=0, retry=0, resume=0,
        whole_law_qualified=False, RPE_total_cost_evaluated=False, scientific_GO=False,
        mandatory_stop=True)
    cpus = len(os.sched_getaffinity(0))
    workers = min(6, cpus)  # six independent tasks, no artificial resource cap
    reference_threads = max(1, (cpus-min(4,cpus))//2)
    mem = {l.split(':')[0]:l.split(':')[1].strip() for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith(('MemTotal:', 'MemAvailable:'))}
    result["resources"] = dict(available_cpus=cpus, memory_before=mem, compile_workers=workers,
        direct_postcheck_threads=1, reference_postcheck_threads=reference_threads,
        raw_reference_threads=cpus, GPU="driver unavailable; existing Aer CPU path",
        resource_caps=None, parallel_policy="six compiler processes; two reference state-action batches and four small direct matrices",
        dense_reference_Operator=False, original_source_read_only=True)
    exit_code = 1
    try:
        m = bootstrap(origin)
        from threadpoolctl import threadpool_limits
        from trottertracks.pr_dictionary_validation.fixture import CompilerProfile
        write(output/"progress.json", dict(stage="prepare_fixed_word"))
        # Small eigensystems/256-512 matrices run efficiently with coordinated
        # BLAS; the only large raw state action uses all available Aer threads.
        with threadpool_limits(limits=1):
            bundle = m.prepare(origin, saved, c)
            result["conditions"] = bundle.identity
            write(output/"conditions.json", bundle.identity)
            write(output/"fusion.json", bundle.fusion)
            for (arm, axis), qc in bundle.wrappers.items():
                m.save_qpy(output/"circuits"/(arm+"_"+axis+"_raw.qpy"), qc)
            for name, qc in (("A",bundle.later),("B",bundle.earlier),("A_actual_adjoint",bundle.later_adjoint)):
                m.save_qpy(output/"circuits"/(name+".qpy"), qc)
            checks = load_file("track_d_h4_checks", ROOT/"tests/track_d_h4_reference_checks.py")
            write(output/"progress.json", dict(stage="precompile_correctness"))
            result["precompile_checks"] = checks.run_checks(bundle, m, output, records, cpus)
        write(output/"precompile_checks.json", result["precompile_checks"])
        write(output/"progress.json", dict(stage="six_wrapper_compiles"))
        tasks = [(str(origin),str(output),arm,axis,reference_threads if arm=="reference" else 1)
            for arm in m.ARMS for axis in m.AXES]
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            futures = []
            for task in tasks:
                futures.append(pool.submit(compile_one,task));result["compile_tasks_submitted"] += 1
            for future in as_completed(futures):
                try: future.result()
                except BaseException:
                    for other in futures: other.cancel()
                    raise
            write(output/"progress.json", dict(stage="compiled_correctness", compiles=6))
            futures = [pool.submit(verify_one,task) for task in tasks]
            rows = {}
            for future in as_completed(futures):
                try:
                    key, report = future.result(); rows[key] = report
                    write(output/"progress.json", dict(stage="compiled_correctness", complete=sorted(rows)))
                except BaseException:
                    for other in futures: other.cancel()
                    raise
        result.update(status="TECHNICAL_PASS_MANDATORY_STOP", compiled=rows,
            compiler=asdict(CompilerProfile()), phase_policy="qiskit_P0_mpfr160_declared_phase_v1",
            qualification="one saved word, fixed cut, all system input columns, numerical observable agreement; no rigorous enclosure")
        # Byte preservation after execution (no original runners/grants used).
        _, after = preflight(origin, saved)
        if after != identity: raise ValueError("SOURCE_INPUT_CHANGED_DURING_EXECUTION")
        result["source_input_preserved"] = True
        exit_code = 0
    except BaseException as error:
        result.update(status="FAIL_MANDATORY_STOP", error=repr(error), traceback=traceback.format_exc())
    finally:
        result["compile_calls_started"] = len(list(output.glob("*_compile.json")))
        result.update(wall_seconds=time.monotonic()-started,
            parent_CPU_seconds=time.process_time(), parent_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write(output/"result.json", result)
        write(output/"STOP.json", dict(status=result["status"], mandatory_stop=True, retry=0, resume=0,
            compile_calls_started=result["compile_calls_started"], next_stage_authorized=False))
        inventory={str(p.relative_to(output)):dict(bytes=p.stat().st_size,sha256=sha(p.read_bytes()))
            for p in output.rglob('*') if p.is_file() and 'matplotlib-cache' not in p.parts}
        write(output/"inventory.json",inventory)
    print(json.dumps({k:result[k] for k in ("source_commit","status","compile_calls_started","wall_seconds")}))
    return exit_code


if __name__ == "__main__": raise SystemExit(main())
