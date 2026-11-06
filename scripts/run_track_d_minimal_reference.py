#!/usr/bin/env python3
"""Run the frozen Track D artificial checks, then exactly six wrapper compiles."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib
from importlib.metadata import version
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = (
    "src/trotterlib/track_d_minimal_reference.py",
    "scripts/run_track_d_minimal_reference.py",
    "tests/track_d_minimal_reference_checks.py",
    "docs/research/track_d_minimal_reference_validation_20261106/PLAN.md",
)
REUSED_PATHS = (
    "scripts/native_df_qiskit_validation_support.py",
    "src/trotterlib/native_df_gate_model.py",
    "src/trotterlib/df_partial_s2.py",
    "src/trotterlib/df_rte_circuit.py",
    "src/trotterlib/df_rte_qiskit.py",
    "src/trotterlib/rte.py",
    "src/trotterlib/df_trotter/ops.py",
    "src/trottertracks/pr_dictionary_validation/gates.py",
    "src/trottertracks/pr_dictionary_validation/fixture.py",
)


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = git("rev-parse", "HEAD").decode().strip()
    for path in (*SOURCE_PATHS, *REUSED_PATHS):
        if (ROOT / path).read_bytes() != git("show", source + ":" + path):
            raise ValueError("Uncommitted execution SOURCE: " + path)
    if version("qiskit") != "1.3.0" or version("qiskit-nature") != "0.7.2":
        raise ValueError("Frozen artificial Qiskit environment is unavailable")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)  # fresh results, no retry/resume
    cpus = len(os.sched_getaffinity(0))
    workers = min(6, cpus)  # six fixed tasks; no additional scientific work
    # Tiny (<=32x32) numerical operations: avoid nested pools under six compiles.
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["QISKIT_PARALLEL"] = "false"
    os.environ["MPLCONFIGDIR"] = str(output / "matplotlib-cache")
    started = time.monotonic()
    result = dict(schema="track_d_minimal_reference_v1", source_commit=source,
                  fixture="native-partial-closed:B", status="RUNNING", mandatory_stop=True,
                  compile_calls_started=0, compiled_wrappers=[],
                  source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCE_PATHS},
                  reused_source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in REUSED_PATHS},
                  python=sys.version, packages={p: version(p) for p in
                    ("qiskit", "qiskit-nature", "numpy", "scipy", "openfermion")},
                  available_logical_cpus=cpus, compile_workers=workers,
                  workload="one fixed two-mode word; three arms; X/Y; no sampling",
                  acceleration="shared immutable fixture; six independent compile tasks; small dense observables",
                  threads={k: os.environ[k] for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                      "MKL_NUM_THREADS", "RAYON_NUM_THREADS", "QISKIT_PARALLEL")},
                  started_utc=datetime.now(timezone.utc).isoformat())
    write_json(output / "result.json", result)
    exit_code = 1
    try:
        support = load_file("track_d_existing_support", ROOT / "scripts/native_df_qiskit_validation_support.py")
        native = support.bootstrap()
        implementation = importlib.import_module("trotterlib.track_d_minimal_reference")
        support.package("track_d_existing_compile", ROOT / "src/trottertracks/pr_dictionary_validation")
        compile_gates = importlib.import_module("track_d_existing_compile.gates")
        profile_type = importlib.import_module("track_d_existing_compile.fixture").CompilerProfile
        profile = profile_type()
        result["compiler_profile"] = asdict(profile)
        result["native_source"] = support.REFERENCE
        fixture = implementation.build_fixture(support, native)
        checks = load_file("track_d_scientific_checks", ROOT / "tests/track_d_minimal_reference_checks.py")
        result["correctness"] = checks.run_checks(fixture, implementation, compile_gates.matrix_gate)
        result["phase_radians"] = fixture.phase
        result["physical_operator"] = implementation.complex_array(fixture.physical_reference)
        result["input_definition"] = dict(T=implementation.T, q=implementation.Q,
            scalar=implementation.SCALAR, tau=implementation.TAU, cutoff=0,
            system_modes=2, events=[e.to_dict() for e in fixture.events],
            scientific_H_schedule_certified=False, ground_state_generated=False,
            sharing="same-basis tail B; same-block D0 step-boundary ordinary fusion in both direct arms",
            reference_cut="between complete steps; scalar assigned to A; cut fusion loss retained")
        write_json(output / "result.json", result)
        from qiskit import qpy
        from qiskit.quantum_info import Operator

        def compile_one(key):
            arm, axis = key
            start = time.monotonic()
            raw = fixture.wrappers[key]
            compiled = implementation.compile_wrapper(raw, compile_gates, profile)
            observable = implementation.induced_observable(compiled)
            agreement = compile_gates.matrix_gate(observable,
                implementation.expected_observable(fixture.physical_reference, axis),
                implementation.TOLERANCE, label=arm + ":" + axis + ":compiled_observable")
            # Same-qubit raw/compiled whole unitary comparison retains global phase.
            raw_u = Operator(raw.remove_final_measurements(inplace=False)).data
            compiled_u = Operator(compiled.remove_final_measurements(inplace=False)).data
            whole = compile_gates.matrix_gate(compiled_u, raw_u, implementation.TOLERANCE,
                                              label=arm + ":" + axis + ":compiled_unitary")
            for label, circuit in (("raw", raw), ("compiled", compiled)):
                with (output / (arm + "_" + axis + "_" + label + ".qpy")).open("wb") as stream:
                    qpy.dump(circuit, stream)
            return dict(arm=arm, axis=axis, metrics=implementation.metrics(compiled),
                        raw_metrics=implementation.metrics(raw), observable_agreement=agreement,
                        global_phase_preserving_unitary_agreement=whole,
                        compile_and_check_wall_seconds=time.monotonic() - start)

        keys = [(arm, axis) for arm in implementation.ARMS for axis in implementation.AXES]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for key in keys:
                result["compile_calls_started"] += 1
                futures[executor.submit(compile_one, key)] = key
            for future in as_completed(futures):
                result["compiled_wrappers"].append(future.result())
                write_json(output / "result.json", result)
        result["compiled_wrappers"].sort(key=lambda r: (r["arm"], r["axis"]))
        if len(result["compiled_wrappers"]) != 6 or result["compile_calls_started"] != 6:
            raise ValueError("Six-wrapper scope incomplete")
        result["status"] = "TECHNICAL_PASS_MANDATORY_STOP"
        exit_code = 0
    except Exception as error:
        result["status"] = "FAILED_MANDATORY_STOP"
        result["error"] = str(error)
        result["traceback"] = traceback.format_exc()
    finally:
        result["wall_seconds"] = time.monotonic() - started
        result["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        result["next_stage_executed"] = False
        result["quantum_shots"] = 0
        result["sampling_calls"] = 0
        result["molecular_calculations"] = 0
        result["scientific_GO"] = False
        result["end_to_end_cost_evaluated"] = False
        write_json(output / "result.json", result)
    print(json.dumps({k: result[k] for k in ("source_commit", "status", "compile_calls_started", "wall_seconds")}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
