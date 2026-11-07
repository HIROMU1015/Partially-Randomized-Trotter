#!/usr/bin/env python3
"""Authorized saved-four-wrapper numerical repair. No raw build or compile."""
import argparse
from pathlib import Path
import sys
import time

import run_track_d_c8_1_diagnostics as r
ROOT = r.ROOT
OUTPUT = "artifacts/track_d_c8_1_q64/2026-11-08/run03_nofusion_repair"
FILES = ("src/trottertracks/track_d_c8_1_nofusion.py", "scripts/run_track_d_c8_1_nofusion.py",
         "tests/check_track_d_c8_1_nofusion.py", "docs/research/track_d_c8_1_q64_20261108/NOFUSION_REPAIR_PLAN.md",
         "artifacts/track_d_c8_1_q64/2026-11-08/nofusion_technical/synthetic_tests.json")
ORIGINAL_WORKER = r.worker


def repaired_worker(prior, origin, arm, axis, threads, output):
    r.boot.bootstrap(origin)
    from trottertracks import track_d_c7_3_acceleration as fast
    from trottertracks import track_d_c8_1_nofusion as repair
    existing = fast.install
    fast.install = lambda budget: repair.install(budget, existing_install=existing)
    try:
        return ORIGINAL_WORKER(prior, origin, arm, axis, threads, output)
    finally:
        fast.install = existing


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--diagnostics-output",type=Path,required=True)
    args, remaining = parser.parse_known_args()
    previous = args.diagnostics_output.resolve()
    if not previous.exists() or "--execute" not in remaining:
        raise ValueError("EXISTING_DIAGNOSTICS_AND_EXPLICIT_EXECUTE_REQUIRED")
    print("waiting for existing four-check diagnostic batch; no science dispatched",flush=True)
    while not (previous/"STOP.json").exists():
        time.sleep(10)
    import json
    result=json.loads((previous/"result.json").read_text())
    if result["status"]!="DIAGNOSTICS_COMPLETE" or result["completed"]!=4 or not result["prior_preserved"]:
        raise ValueError("COMPLETE_PRESERVED_DIAGNOSTICS_REQUIRED")
    if json.loads((previous/"SOURCE.json").read_text())["execution_source"]!="46ce57e18b7d8cb606c0446e562ca98be63b4f5d":
        raise ValueError("DIAGNOSTIC_SOURCE")
    r.OUTPUT=OUTPUT
    r.FILES=(*r.FILES,*FILES)
    r.BACKEND=dict(backend_changed=True,simulation_fusion_enabled=False,precision_changed=False,
        prior_diagnostic_source="46ce57e18b7d8cb606c0446e562ca98be63b4f5d",
        prior_diagnostic_evidence={p.name:r.sha(p) for p in sorted(previous.glob("*.json"))})
    r.worker=repaired_worker
    sys.argv=[sys.argv[0],*remaining]
    # The reused runner performs exactly four saved-circuit postchecks and STOP.
    # It has no path construction, sampling, compiler or remaining-path dispatch.
    return r.main()


if __name__=="__main__":
    raise SystemExit(main())
