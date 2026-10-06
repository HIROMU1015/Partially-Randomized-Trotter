#!/usr/bin/env python3
"""Local focused/static v2 verification; never runs registered optimization."""
import argparse
import hashlib
from importlib import metadata, util
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"src"))
from trottertracks.algorithm_codesign.ra_d0.table import extract_saved_table
from trottertracks.algorithm_codesign.ra_d0.semantics import audit_semantics
from trottertracks.algorithm_codesign.ra_d0 import lp, backend

BASE = "0ddf67756516e08f85fed1b987459a5e862676b7"
OLD = "artifacts/track_b_ra_d0_preparation/2026-10-06"
NEW = "artifacts/track_b_ra_d0_source_review_v2/2026-10-07"


def fixed_blob(path):
    if path.lower().endswith(".npz"):
        raise PermissionError("molecular identity access forbidden")
    return subprocess.check_output(["git", "-C", str(ROOT), "show", BASE+":"+path])


def static_audit():
    old_audit = json.loads((ROOT/OLD/"source_review_audit_v1.json").read_text())
    hashes = dict(old_audit["protected_files_sha256"])
    manifest_path = "artifacts/track_b_rte_reallocation_r1_source/2026-10-06/source_manifest_v1.json"
    hashes.update(json.loads(fixed_blob(manifest_path))["critical_sha256"])
    hashes[manifest_path] = hashlib.sha256(fixed_blob(manifest_path)).hexdigest()
    old_inventory = json.loads((ROOT/OLD/"evidence_manifest_v1.json").read_text())["file_inventory"]
    protected_old = {p: i["sha256"] for p, i in old_inventory.items()
                     if p.startswith(OLD+"/") or p == "tests/tracks/algorithm_codesign/test_ra_d0_preparation.py"}
    protected_old[OLD+"/evidence_manifest_v1.json"] = hashlib.sha256(fixed_blob(OLD+"/evidence_manifest_v1.json")).hexdigest()
    hashes.update(protected_old)
    for path, expected in hashes.items():
        value = (ROOT/path).read_bytes() if (ROOT/path).exists() else fixed_blob(path)
        if hashlib.sha256(value).hexdigest() != expected:
            raise ValueError("protected identity changed: "+path)
    raw = (ROOT/"artifacts/track_b_rte_reallocation_r1_result/2026-10-06/v1/result.json").read_bytes()
    table = extract_saved_table(raw)
    if table != json.loads((ROOT/OLD/"candidate_table_v1.json").read_text()):
        raise ValueError("saved table/source mismatch")
    semantic = audit_semantics(table)
    if semantic != json.loads((ROOT/OLD/"semantic_audit_v1.json").read_text()):
        raise ValueError("old ideal semantic artifact changed")
    grid = json.loads((ROOT/OLD/"shot_grid_query_recipe_v1.json").read_text())
    counts = {x: {"points": len(g["points"]), "anchors": len(g["anchor_shots"]),
                  "coverage": sum(p["tag"] == "COVERAGE_GRID" for p in g["points"])} for x, g in grid["grids"].items()}
    if sum(g["points"] for g in counts.values()) != 737 or any(g["anchors"] != 9 for g in counts.values()):
        raise ValueError("fixed grid changed")
    indexes = json.loads((ROOT/NEW/"index_prefix_identity_v2.json").read_text())
    for path, receipt in indexes["indexes"].items():
        current, old = (ROOT/path).read_bytes(), fixed_blob(path)
        if current[receipt["prefix_bytes"]:] != old or hashlib.sha256(old).hexdigest() != receipt["base_body_sha256"]:
            raise ValueError("index historical body changed: "+path)
    ancestry = {}
    for label, commit in (("R1", "24bfeb84a4ce87b56985d174dd98d1d5e1702a2b"),
                          ("R1p5", "af3d014d0a0cfcbbd25bb544f6544652fec92942"),
                          ("math_audit", "8a04c148a66d23dbc1f045086a95a5e19a6372dc")):
        subprocess.check_call(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", commit, BASE])
        ancestry[label] = True
    return {"protected_sha256": hashes, "fixed_inputs_in_base_ancestry": ancestry,
            "candidate_columns_per_x": {x: len(t["columns"]) for x, t in table["tables"].items()},
            "sign_pairs": sum(len(t["sign_checks"]) for t in table["tables"].values()),
            "saved_sequence_identity_checks": 126, "grid_counts": counts,
            "old_35_tests_byte_exact": True, "historical_index_bodies_byte_exact": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-write", action="store_true", help="technical verification without replacing a receipt")
    args = parser.parse_args()
    start = time.monotonic()
    audit = static_audit()
    import scipy.optimize
    original_linprog = scipy.optimize.linprog
    stack = []
    counts = {"synthetic_solver_calls": 0, "registered_solver_calls": 0, "registered_rejection_attempts": 0}
    def tracked_linprog(*a, **kw):
        if stack and stack[-1] != "SYNTHETIC":
            counts["registered_solver_calls"] += 1
            raise AssertionError("registered optimization is forbidden in source verification")
        counts["synthetic_solver_calls"] += 1
        return original_linprog(*a, **kw)
    def tracking(function):
        def wrapped(problem, *a, **kw):
            stack.append(problem.domain)
            if problem.domain != "SYNTHETIC":
                counts["registered_rejection_attempts"] += 1
            try:
                return function(problem, *a, **kw)
            finally:
                stack.pop()
        return wrapped
    old_nominal, old_synthetic = backend.nominal, lp.solve_synthetic
    scipy.optimize.linprog = tracked_linprog
    backend.nominal, lp.solve_synthetic = tracking(old_nominal), tracking(old_synthetic)
    try:
        suite, groups = unittest.TestSuite(), {}
        for name in ("test_ra_d0_preparation", "test_ra_d0_source_review_v2"):
            spec = util.spec_from_file_location(name, ROOT/"tests/tracks/algorithm_codesign"/(name+".py"))
            module = util.module_from_spec(spec); spec.loader.exec_module(module)
            tests = unittest.defaultTestLoader.loadTestsFromModule(module)
            groups[name] = tests.countTestCases()
            suite.addTests(tests)
        stream = io.StringIO()
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    finally:
        scipy.optimize.linprog = original_linprog
        backend.nominal, lp.solve_synthetic = old_nominal, old_synthetic
    packages = {}
    for name in ("numpy", "scipy"):
        dist = metadata.distribution(name)
        packages[name] = {"version": dist.version, "RECORD_sha256": hashlib.sha256(dist.read_text("RECORD").encode()).hexdigest(),
                          "METADATA_sha256": hashlib.sha256(dist.read_text("METADATA").encode()).hexdigest()}
    tested_paths = sorted([*ROOT.glob("src/trottertracks/algorithm_codesign/ra_d0/*.py"),
                          *[ROOT/"tests/tracks/algorithm_codesign"/(g+".py") for g in groups],
                          ROOT/"scripts/tracks/algorithm_codesign/run_ra_d0_one_shot.py",
                          Path(__file__).resolve()])
    report = {"schema": "ra_d0_source_review_focused_verification_v2",
              "status": "PASS" if result.wasSuccessful() and counts["registered_solver_calls"] == 0 else "FAIL",
              "tests_run": result.testsRun, "test_groups": groups, "test_log": stream.getvalue(),
              "wall_seconds": time.monotonic()-start, **counts, "static_audit": audit,
              "runtime_identity": {"python": platform.python_version(), "packages": packages},
              "tested_source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in tested_paths},
              "authorization_created": False, "registered_budget_or_witness_acquired": False,
              "new_synthesis_science_circuit_matrix_trajectory_GPU_calls": 0, "NPZ_access": 0,
              "evidence_scope": "local off-domain technical tests/static saved-value identity audit; not CI/external reproduction/scientific result",
              "mandatory_STOP": True}
    if not args.no_write:
        with (ROOT/NEW/"focused_verification_v2.json").open("x") as out:
            json.dump(report, out, ensure_ascii=False, indent=2); out.write("\n")
    print(json.dumps({k: report[k] for k in ("status", "tests_run", "test_groups", "synthetic_solver_calls", "registered_solver_calls")}))
    if report["status"] != "PASS":
        print(stream.getvalue()); raise SystemExit(1)


if __name__ == "__main__":
    main()
