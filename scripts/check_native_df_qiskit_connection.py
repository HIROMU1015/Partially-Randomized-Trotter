#!/usr/bin/env python3
"""Run the fixed-source artificial Qiskit connection suite, never molecular work."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata as metadata
import importlib.util
import inspect
import io
import json
import os
from pathlib import Path
import platform
import resource
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load_test(name):
    path = ROOT / "tests" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.receipt and args.receipt.exists(): parser.error("Preserve existing receipt; use a new path")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "RAYON_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["QISKIT_PARALLEL"] = "false"
    with tempfile.TemporaryDirectory(prefix="native-df-qiskit-local-cache-") as cache:
        os.environ["MPLCONFIGDIR"] = cache
        packages = {}
        for key in ("qiskit", "qiskit-nature", "numpy", "scipy", "openfermion"):
            try: packages[key] = metadata.version(key)
            except metadata.PackageNotFoundError: packages[key] = None
        have_environment = all(packages.values())
        modules = [load_test("test_native_df_gate_model")]
        if have_environment: modules.append(load_test("test_native_df_qiskit_connection"))
        suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(module) for module in modules)
        stream = io.StringIO(); start = time.monotonic(); cpu = time.process_time()
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        sources, observations, gate_definitions = [], [], []
        if have_environment:
            import native_df_qiskit_validation_support as support
            sources, observations = support.receipt_sources(), support.OBSERVATIONS
            from qiskit.circuit import library as lib
            from qiskit_nature.second_q.circuit.library import BogoliubovTransform
            for cls in (lib.RZGate, lib.RZZGate, lib.CRZGate, lib.XXPlusYYGate, lib.HGate,
                        lib.PhaseGate, lib.CXGate, BogoliubovTransform):
                path = Path(inspect.getfile(cls))
                gate_definitions.append({"class": cls.__name__, "package_path": "/".join(path.parts[-5:]),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        own = ["src/trotterlib/native_df_gate_model.py", "tests/test_native_df_gate_model.py",
               "tests/test_native_df_qiskit_connection.py", "scripts/native_df_qiskit_validation_support.py",
               "scripts/check_native_df_qiskit_connection.py"]
        receipt = {"schema_version": 1, "scope": "LOCAL_ARTIFICIAL_QISKIT_CONNECTION_ONLY",
            "base_commit": "1ea42fbf1bd1c8341ffd410b8067467b4ba4e170",
            "native_reference_commit": "87b6b9f46b7e8ba9abe132ee21478c114b6f46f7",
            "python": platform.python_version(), "packages": packages,
            "qiskit_1_3_0_version_match": packages["qiskit"] == "1.3.0",
            "full_experimental_environment_compatibility_certified": False,
            "status": "QISKIT_CONNECTION_SYNTHETIC_PASS" if have_environment and result.wasSuccessful() else
                      "PARTIAL_STDLIB_ONLY" if not have_environment and result.wasSuccessful() else "TEST_FAILURE",
            "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
            "skipped": len(result.skipped), "wall_seconds": time.monotonic() - start,
            "process_cpu_seconds": time.process_time() - cpu,
            "cumulative_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "threads": {k: os.environ[k] for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "RAYON_NUM_THREADS", "QISKIT_PARALLEL")},
            "parallelization": "SERIAL_TINY_CASES_WITH_REUSED_METADATA_AND_GATE_COST_CACHE",
            "phase_or_total_wall_cap": None,
            "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in own},
            "loaded_source_closure": sources, "installed_gate_definitions": gate_definitions,
            "observations": observations, "transcript": stream.getvalue(),
            "general_identity_phase_expectation": "REMAINS_BOUND_0_TO_1_P_PER_OCCURRENCE",
            "real_H6_H8_data_loaded": False, "molecular_computation": 0, "trajectory_sampling": 0,
            "transpile_compile": 0, "new_grants": 0, "server_connections": 0,
            "independent_scientific_review": False, "next_stage_authorized": False,
            "status_after_task": "MANDATORY_STOP"}
        if args.receipt:
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            with args.receipt.open("x", encoding="utf-8") as f:
                json.dump(receipt, f, ensure_ascii=False, indent=2); f.write("\n")
        print(stream.getvalue(), end="")
        print("Validation status:", receipt["status"])
        return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
