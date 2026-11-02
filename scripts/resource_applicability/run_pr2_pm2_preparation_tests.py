#!/usr/bin/env python3
"""Run only PM-2 contract tests under a pre-import protected-path barrier."""
import builtins
from datetime import datetime, timezone
import io
import json
import os
import sys


def install_boundary():
    counter = {"protected_access_attempts": 0}

    def wrap(function):
        def guarded(path, *args, **kwargs):
            if not isinstance(path, int):
                name = os.fsdecode(os.fspath(path))
                parts = name.replace("\\", "/").split("/")
                if (name.lower().endswith((".npz", ".npy", ".pkl", ".pickle")) or
                        ".runtime" in parts or any(p.endswith("_registry") for p in parts)):
                    counter["protected_access_attempts"] += 1
                    raise AssertionError("PM-2 preparation attempted protected data access")
            return function(path, *args, **kwargs)
        return guarded

    builtins.open, io.open = wrap(builtins.open), wrap(io.open)
    os.stat, os.lstat = wrap(os.stat), wrap(os.lstat)
    return counter


def main():
    started = datetime.now(timezone.utc).isoformat()
    counter = install_boundary()
    import pytest
    code = pytest.main(["-q", "-rs", "-p", "no:cacheprovider",
                       "tests/tracks/resource_applicability/test_pm2_precision_contract.py"])
    print("PM2_PREPARATION_TEST_AUDIT:", json.dumps({"started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(), "exit_code": code,
        "python_executable": sys.executable, "python_version": sys.version,
        "suite": "PM2 contract only; no old science tests", **counter}))
    return code if counter["protected_access_attempts"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
