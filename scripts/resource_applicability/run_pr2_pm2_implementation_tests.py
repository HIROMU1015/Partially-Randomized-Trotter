#!/usr/bin/env python3
"""PM-2 synthetic implementation tests; real evidence and science are forbidden."""
from datetime import datetime, timezone
import json
import sys


def main():
    from trottertracks.resource_applicability.pm2_precision_analysis import install_boundary
    counter = install_boundary(forbid_saved_evidence=True)
    started = datetime.now(timezone.utc).isoformat()
    import pytest
    code = pytest.main(["-q", "-rs", "-p", "no:cacheprovider",
                       "tests/tracks/resource_applicability/test_pm2_precision_analysis.py"])
    print("PM2_IMPLEMENTATION_TEST_AUDIT:", json.dumps({"started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(), "exit_code": code,
        "python_executable": sys.executable, "python_version": sys.version,
        "test_scope": "synthetic only; real saved evidence/NPZ/runtime/registries forbidden", **counter}))
    return code if not any(counter.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
