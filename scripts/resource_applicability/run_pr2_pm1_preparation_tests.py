#!/usr/bin/env python3
"""Run ONLY the approved synthetic/saved-text tests with a data-path barrier.

The barrier is installed BEFORE importing pytest or numerical dependencies.
It rejects NPZ/NPY/pickle opens and stat/lstat, including Path.resolve paths;
and runtime paths. It is a local diagnostic safeguard, not an OS sandbox.
"""
import builtins
from datetime import datetime, timezone
import io
import json
import os
import sys

TESTS = (
    "tests/tracks/resource_applicability/test_pm1_discard.py",
    "tests/tracks/resource_applicability/test_pm0_evidence_attribution.py",
    "tests/test_df_partial_s2_repeated.py",
    "tests/test_df_partial_s2_repeated_cost.py",
    "tests/test_rpe_hadamard_interrogation.py",
    "tests/test_df_rpe_hadamard_compiled_cost.py",
    "tests/test_rpe_hadamard_compiled_cost_benchmark.py",
)


def install_boundary():
    counter = {"blocked_protected_access_attempts": 0}

    def wrap(function):
        def guarded(path, *args, **kwargs):
            if not isinstance(path, int):
                name = os.fsdecode(os.fspath(path))
                parts = name.replace("\\", "/").split("/")
                if name.lower().endswith((".npz", ".npy", ".pkl", ".pickle")) or ".runtime" in parts:
                    counter["blocked_protected_access_attempts"] += 1
                    raise AssertionError("PM-1 preparation attempted protected scientific data access")
            return function(path, *args, **kwargs)
        return guarded

    builtins.open, io.open = wrap(builtins.open), wrap(io.open)
    os.stat, os.lstat = wrap(os.stat), wrap(os.lstat)
    return counter


def main():
    started = datetime.now(timezone.utc).isoformat()
    counter = install_boundary()
    import pytest
    code = pytest.main(["-q", "-rs", "-p", "no:cacheprovider", *TESTS])
    print("PM1_PREPARATION_DATA_BOUNDARY:", counter)
    print("PM1_TEST_RUN_AUDIT:", json.dumps({"started_utc": started,
          "finished_utc": datetime.now(timezone.utc).isoformat(), "exit_code": code, **counter}))
    return code if counter["blocked_protected_access_attempts"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
