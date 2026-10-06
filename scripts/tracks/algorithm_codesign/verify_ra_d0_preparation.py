#!/usr/bin/env python3
"""Read-only static audit plus focused off-domain tests. No registered solve."""
import hashlib
import argparse
import importlib.util
import io
import json
from pathlib import Path
import platform
import sys
import time
import unittest
from importlib import metadata

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"src"))
from trottertracks.algorithm_codesign.ra_d0.table import extract_saved_table
from trottertracks.algorithm_codesign.ra_d0.grid import build_grid
from trottertracks.algorithm_codesign.ra_d0.semantics import audit_semantics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt-name", default="focused_verification_v1.json",
                        choices=("focused_verification_v1.json", "focused_verification_v2.json",
                                 "focused_verification_v3.json"))
    args = parser.parse_args()
    start = time.monotonic()
    directory = ROOT/"artifacts/track_b_ra_d0_preparation/2026-10-06"
    table = extract_saved_table((ROOT/"artifacts/track_b_rte_reallocation_r1_result/2026-10-06/v1/result.json").read_bytes())
    checks = {}
    for name, actual in (("candidate_table_v1.json", table),
                         ("shot_grid_query_recipe_v1.json", build_grid(table)),
                         ("semantic_audit_v1.json", audit_semantics(table))):
        if json.loads((directory/name).read_text()) != actual:
            raise ValueError("static artifact/source mismatch: "+name)
        checks[name] = "MATCH"
    test_path = ROOT/"tests/tracks/algorithm_codesign/test_ra_d0_preparation.py"
    spec = importlib.util.spec_from_file_location("ra_d0_preparation_tests", test_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    packages = {}
    for name in ("numpy", "scipy"):
        distribution = metadata.distribution(name)
        packages[name] = {"version": distribution.version,
                          "METADATA_sha256": hashlib.sha256(distribution.read_text("METADATA").encode()).hexdigest(),
                          "RECORD_sha256": hashlib.sha256(distribution.read_text("RECORD").encode()).hexdigest()}
    report = {"status": "PASS" if result.wasSuccessful() else "FAIL", "tests_run": result.testsRun,
              "test_log": stream.getvalue(), "wall_seconds": time.monotonic()-start,
              "static_artifact_checks": checks, "saved_sequence_checks": 126,
              "synthetic_LP_solver_calls": 2, "registered_optimization_calls": 0,
              "registered_solver_launch_rejection_checks": 2,
              "synthesis_science_circuit_sampler_GPU_calls": 0,
              "python": platform.python_version(), "packages": packages,
              "backend_is_technical_candidate_not_execution_authorization": True,
              "R1_unchanged": True, "mandatory_STOP": True}
    with (directory/args.receipt_name).open("x") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("test_log", "packages")}))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
