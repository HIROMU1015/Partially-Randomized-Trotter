#!/usr/bin/env python3
"""Run only the standalone artificial gate-model suite; emit a technical receipt."""
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import platform
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.receipt and args.receipt.exists():
        parser.error("Receipt already exists; preserve prior evidence")
    path = ROOT / "tests/test_native_df_gate_model.py"
    spec = importlib.util.spec_from_file_location("test_native_df_gate_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stream = io.StringIO()
    start = time.monotonic()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(module))
    files = ("src/trotterlib/native_df_gate_model.py", "tests/test_native_df_gate_model.py",
             "scripts/check_native_df_gate_model_synthetic.py")
    receipt = {"schema_version": 1, "scope": "LOCAL_STDLIB_SYNTHETIC_ONLY",
               "python": platform.python_version(), "profile": module.m.PROFILE,
               "tests_run": result.testsRun, "failures": len(result.failures),
               "errors": len(result.errors), "skipped": len(result.skipped),
               "successful": result.wasSuccessful(), "wall_seconds": time.monotonic() - start,
               "source_sha256": {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files},
               "test_transcript": stream.getvalue(),
               "molecular_data_loaded": False, "qiskit_imported": "qiskit" in sys.modules,
               "scientific_calculations": 0, "sampling": 0, "compile": 0,
               "next_stage_authorized": False, "status_after_task": "MANDATORY_STOP"}
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        with args.receipt.open("x", encoding="utf-8") as f:
            json.dump(receipt, f, ensure_ascii=False, indent=2)
            f.write("\n")
    print(stream.getvalue(), end="")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
