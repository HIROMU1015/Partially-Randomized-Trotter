#!/usr/bin/env python3
"""Dedicated metadata-only preflight; no switch can launch saved-value analysis."""
import argparse
from datetime import datetime, timezone
import importlib.abc
import json
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from trottertracks.resource_applicability.ax1b_contract import FLAGS, Stop, digest, require, sha256
from trottertracks.resource_applicability.ax1b_execution import _limits
from trottertracks.resource_applicability.ax1b_preflight import run_preflight


class NumericalBoundary(importlib.abc.MetaPathFinder):
    """Detect and deny accidental numerical/scientific imports or function calls."""
    def __init__(self):
        self.import_attempts = self.call_attempts = 0

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"numpy", "scipy", "qiskit", "trotterlib", "openfermion", "pyscf"} or fullname.endswith(".ax1b_analysis"):
            self.import_attempts += 1
            raise Stop("IMPLEMENTATION", "numerical/scientific import forbidden in preflight")

    def profile(self, frame, event, arg):
        if event == "call" and frame.f_globals.get("__name__", "").startswith("trottertracks.resource_applicability.ax1b_"):
            if frame.f_code.co_name in {"analyze", "fit_cost", "project_saved", "features_from_signal", "finite_normalization",
                                       "reference_shots", "selection", "common_support_selection", "paired_statistics", "cost_error",
                                       "error_summary", "rank_index", "conditional_work", "paired_total", "complexity_gate", "actions", "execute"}:
                self.call_attempts += 1
                raise Stop("IMPLEMENTATION", "scientific-value calculation forbidden in preflight")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-only-preflight", action="store_true")
    parser.add_argument("--request-manifest", type=Path)
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args(argv)
    input_audit = []
    boundary = NumericalBoundary()
    audit = dict(schema_version="track_a_ax1b_metadata_preflight_audit_v1", scope="READ_ONLY_HASH_SCHEMA_IDENTITY_FEATURE_SHAPE",
                 started_utc=datetime.now(timezone.utc).isoformat(), analysis_launch_count=0, model_fit_count=0,
                 performance_or_regret_evaluations=0, science_values_recalculated=0, **FLAGS)
    code = 2
    try:
        require(args.metadata_only_preflight and args.request_manifest is not None and args.audit_output is not None,
                "AUTHORIZATION", "explicit metadata-only request and new audit destination required")
        destination = args.audit_output.resolve()
        permitted = destination.is_relative_to(Path("/tmp")) or destination.is_relative_to(
            ROOT / "artifacts/resource_applicability/track_a_ax1b_identity_compatibility")
        require(permitted and not destination.exists(), "OUTPUT_COLLISION", "audit must use a new preparation-only path")
        request_raw = args.request_manifest.read_bytes()
        request = json.loads(request_raw)
        require(all(request.get(k) is v for k, v in FLAGS.items()) and request.get("metadata_only_preflight_authorized") is True,
                "AUTHORIZATION", "analysis authorization forbidden in preflight request")
        observed_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
        require(observed_head == request["tested_worktree_base_commit"], "IMPLEMENTATION", "preflight base HEAD differs")
        resources = dict(assigned_cpu_cores=1, cpu_affinity=[0], processes=1, blas_threads=1, ram_limit_bytes=8589934592,
                         wall_time_limit_seconds=300, output_disk_limit_bytes=536870912)
        require(request["resources"] == resources and request.get("gpu_prohibited") is True,
                "BUDGET", "preflight resource caps differ")
        audit.update(request_manifest_sha256=sha256(request_raw), request_manifest_canonical_sha256=digest(request),
                     observed_base_commit=observed_head, source_files=request["source_files"], environment=request["environment"], resources=resources)
        _limits(dict(resources=resources))
        sys.meta_path.insert(0, boundary)
        sys.setprofile(boundary.profile)
        audit.update(run_preflight(ROOT, request, input_audit))
        audit["status"] = "AX1B_IDENTITY_COMPATIBILITY_PRECHECK_COMPLETE_EXECUTION_NOT_AUTHORIZED"
        code = 0
    except (Stop, OSError, ValueError, KeyError, TypeError) as exc:
        audit.update(precheck_status="PRECHECK_FAIL", status=getattr(exc, "status", "AX1B_STOP_IMPLEMENTATION"), reason=str(exc))
    finally:
        sys.setprofile(None)
        if boundary in sys.meta_path:
            sys.meta_path.remove(boundary)
        signal.setitimer(signal.ITIMER_REAL, 0)
        audit.update(input_audit=input_audit, inputs_verified=len(input_audit),
                     numerical_import_attempts=boundary.import_attempts, forbidden_calculation_attempts=boundary.call_attempts,
                     finished_utc=datetime.now(timezone.utc).isoformat())
    if args.audit_output is not None:
        destination = args.audit_output.resolve()
        if (destination.is_relative_to(Path("/tmp")) or destination.is_relative_to(
                ROOT / "artifacts/resource_applicability/track_a_ax1b_identity_compatibility")) and not destination.exists():
            raw = (json.dumps(audit, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
            require(len(raw) <= 536870912, "BUDGET", "preflight audit disk cap")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                stream.write(raw)
    print(json.dumps({k: audit.get(k) for k in ("status", "precheck_status", "reason", "inputs_verified", "analysis_launch_count", "model_fit_count")}, sort_keys=True))
    return code


if __name__ == "__main__":
    sys.exit(main())
