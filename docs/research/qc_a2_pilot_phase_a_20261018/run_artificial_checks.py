"""Record direct stdlib unittest + static checks. Artificial test modules only.

This is a development check harness, not a scientific runner or authorization.
Run with Python -B from any directory in this isolated checkout.
"""

from pathlib import Path
import ast
import copy
import datetime
import hashlib
import importlib.util
import json
import os
import platform
import resource
import sys
import time
import unittest

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT / "src"))


def identity(path):
    data = path.read_bytes()
    return {"path": str(path.relative_to(ROOT)), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


class TrackingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records, self.subcounts = [], {}

    def startTest(self, test):
        super().startTest(test)
        self.subcounts[test.id()] = {"pass": 0, "fail": 0}

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        self.subcounts[test.id()]["pass" if err is None else "fail"] += 1

    def stopTest(self, test):
        super().stopTest(test)
        bad = any(t is test or getattr(t, "test_case", None) is test for t, _ in self.failures + self.errors)
        self.records.append({"id": test.id(), "status": "FAIL" if bad else "PASS",
                             "subtests": self.subcounts[test.id()]})


def main():
    modules, suites = [], []
    for name in ("test_a2_spec01_03_synthetic", "test_a2_spec04_07_synthetic", "test_a2_pilot_phase_a_synthetic"):
        spec = importlib.util.spec_from_file_location(name, ROOT / "tests" / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
        suites.append(unittest.defaultTestLoader.loadTestsFromModule(module))
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    wall, cpu = time.perf_counter(), time.process_time()
    with (OUT / "synthetic_test_log.txt").open("w") as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2, resultclass=TrackingResult).run(unittest.TestSuite(suites))
    wall, cpu = time.perf_counter() - wall, time.process_time() - cpu
    new_sources = sorted((ROOT / "src/trottercert/a2_pilot").glob("*.py")) + [ROOT / "scripts/run_qc_a2_fixed_df_pilot.py"]
    old_sources = sorted((ROOT / "src/trottercert/a2_spec01_03").glob("*.py")) + sorted((ROOT / "src/trottercert/a2_spec04_07").glob("*.py"))
    test_sources = [ROOT / "tests" / (m.__name__ + ".py") for m in modules]
    rows = result.records
    report = {"schema": "a2_pilot_phase_a_artificial_checks_v1", "scope": "ARTIFICIAL_TECHNICAL_ONLY",
              "started_at_utc": started, "completed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "status": "PASS" if result.wasSuccessful() else "FAIL", "tests_run": result.testsRun,
              "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
              "subtests_passed": sum(x["pass"] for x in result.subcounts.values()),
              "subtests_failed": sum(x["fail"] for x in result.subcounts.values()),
              "suites": [{"module": m.__name__, "tests": sum(r["id"].startswith(m.__name__ + ".") for r in rows),
                          "subtests_passed": sum(r["subtests"]["pass"] for r in rows if r["id"].startswith(m.__name__ + "."))}
                         for m in modules],
              "test_results": rows, "source_and_test_identities": [identity(p) for p in new_sources + old_sources + test_sources],
              "python": platform.python_version(), "wall_seconds_measured": format(wall, ".6f"),
              "CPU_seconds_measured": format(cpu, ".6f"),
              "process_peak_RSS_KiB_including_harness": str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
              "resources_observed": {"logical_cpus": os.cpu_count(), "affinity_cpus": len(os.sched_getaffinity(0))},
              "resource_policy": "No CPU/RAM/worker/wall caps imposed. Short exact-integer tests use one runner for monkeypatch isolation. No shared settings changed; no measured speedup claimed.",
              "scientific_inputs_evaluated": 0, "scientific_certificates_generated": 0,
              "saved_scientific_reanalysis": 0, "simulation_or_native_compile": 0,
              "authorization_grant_marker_issued": 0, "SPEC08_implemented": False, "physical_scientific_markers_created": 0, "test_authority_and_claims": "IN_MEMORY_ARTIFICIAL_ONLY_PRODUCTION_REJECTS_THE_TEST_SCHEMA",
              "conftest_loaded": False, "input_scope": "Three named artificial modules only; no repository scientific arrays read by tests"}
    (OUT / "synthetic_test_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if not result.wasSuccessful():
        print((OUT / "synthetic_test_log.txt").read_text())
        return 1
    static = []
    forbidden = {"numpy", "scipy", "pyscf", "qiskit", "openfermion", "trotterlib"}
    for path in new_sources + test_sources + [Path(__file__).resolve()]:
        syntax = ast.parse(path.read_text(), filename=str(path))
        imports = [node for node in ast.walk(syntax) if isinstance(node, (ast.Import, ast.ImportFrom))]
        names = [a.name.split(".")[0] for node in imports if isinstance(node, ast.Import) for a in node.names]
        names += [(node.module or "").split(".")[0] for node in imports if isinstance(node, ast.ImportFrom)]
        no_science_import = not (set(names) & forbidden)
        no_producer_import = True
        if path.name in ("core01_checker.py", "core04_checker.py", "verifier_algebra.py") and path.parent.name == "a2_pilot":
            no_producer_import = not any("producer" in (getattr(node, "module", "") or "") or
                    any("producer" in alias.name or alias.name == "generator" for alias in node.names) for node in imports)
        no_float = True
        if path in new_sources:
            no_float = not any(isinstance(node, ast.Constant) and type(node.value) in (float, complex) for node in ast.walk(syntax))
        static.append({**identity(path), "syntax": "PASS", "no_scientific_backend_import": no_science_import,
                       "checker_no_producer_import": no_producer_import, "source_no_float_or_complex_literal": no_float})
    static_report = {"schema": "a2_pilot_phase_a_static_v1", "checks": static,
                     "status": "PASS" if all(r["no_scientific_backend_import"] and r["checker_no_producer_import"] and r["source_no_float_or_complex_literal"] for r in static) else "FAIL",
                     "limitations": "AST inspection and artificial tests only; subprocess is used exclusively for pinned Git blob metadata. No real input/certificate/marker/authorization, no native compilation or pycache files."}
    (OUT / "static_inspection.json").write_text(json.dumps(static_report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "tests_run", "failures", "errors", "subtests_passed", "subtests_failed", "suites", "wall_seconds_measured")}))
    print("Static:", static_report["status"])
    return 0 if static_report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
