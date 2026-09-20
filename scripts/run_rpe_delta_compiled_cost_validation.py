#!/usr/bin/env python3
"""Project compiled central-RTE-block costs for shortlisted delta schedules."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import qiskit

from trotterlib.rpe_delta_compiled_cost_validation import (
    validate_rpe_delta_compiled_cost,
    write_rpe_delta_compiled_cost_validation,
)
from trotterlib.rte_connected_cluster_cost_validation import (
    load_connected_cluster_calibration,
)


DEFAULT_SCHEDULE = Path(
    "artifacts/rpe_delta_round_schedule_validation/2026-09-20/"
    "h4_sto3g_d100_rank12_ld3_executed_delta_ca_over_10_round_schedule_v1.json"
)
DEFAULT_CALIBRATION = Path(
    "artifacts/rte_connected_cluster_cost_validation/"
    "h4_sto3g_d100_rank12_ld3_dt0p08_ref4_k2_connected_"
    "pilot20_max3000_target4-6_calibration_v2.json"
)
DEFAULT_ANGLE = Path(
    "artifacts/rpe_delta_compiled_cost_validation/2026-09-20/"
    "h4_ld3_shortstep_0p02_to_0p000390625_angle_invariance_n20_v1.json"
)
DEFAULT_BASELINE_TRANSFER = Path(
    "artifacts/rte_connected_cluster_cost_validation/"
    "h4_ld3_s0020_transfer_l4_l6_zero1500_rare500_seed20260913_v2.json"
)
DEFAULT_MIDDLE_TRANSFER = Path(
    "artifacts/rpe_delta_compiled_cost_validation/2026-09-20/"
    "h4_ld3_s0020_transfer_l8_zero300_rare2_v1.json"
)
DEFAULT_LONG_TRANSFER = Path(
    "artifacts/rpe_delta_compiled_cost_validation/2026-09-20/"
    "h4_ld3_s0020_transfer_l16_l32_zero300_rare2_v1.json"
)
DEFAULT_K4_VALIDATION = Path(
    "artifacts/rpe_delta_compiled_cost_validation/2026-09-20/"
    "h4_ld3_s0020_k4_n100_l16_l32_validation_v1.json"
)
DEFAULT_K4_ANGLE = Path(
    "artifacts/rpe_delta_compiled_cost_validation/2026-09-20/"
    "h4_ld3_shortstep_0p02_to_0p000390625_k4_angle_invariance_n10_v1.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/rpe_delta_compiled_cost_validation/2026-09-20/"
    "h4_ld3_delta_0p01_0p0125_0p02_round_schedule_rte_block_cost_v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(command: list[str]) -> str | list[str] | None:
    result = subprocess.run(
        ["git", *command], check=False, capture_output=True, text=True
    )
    if result.returncode != 0:
        return None
    lines = result.stdout.splitlines()
    return lines[0] if len(lines) == 1 else lines


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--angle-invariance", type=Path, default=DEFAULT_ANGLE)
    parser.add_argument(
        "--transfer",
        type=Path,
        action="append",
        default=None,
        help="Connected-cluster transfer artifact; may be supplied repeatedly.",
    )
    parser.add_argument("--model-relative-tolerance", type=float, default=0.05)
    parser.add_argument("--k4-validation", type=Path, default=DEFAULT_K4_VALIDATION)
    parser.add_argument("--k4-angle-invariance", type=Path, default=DEFAULT_K4_ANGLE)
    parser.add_argument("--k4-minimum-event-count", type=int, default=32)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    transfer_paths = (
        args.transfer
        if args.transfer is not None
        else [
            DEFAULT_BASELINE_TRANSFER,
            DEFAULT_MIDDLE_TRANSFER,
            DEFAULT_LONG_TRANSFER,
        ]
    )
    schedule = _load(args.schedule)
    calibration = load_connected_cluster_calibration(args.calibration)
    angle = _load(args.angle_invariance)
    transfers = [_load(path) for path in transfer_paths]
    k4_validation = _load(args.k4_validation)
    k4_angle = _load(args.k4_angle_invariance)
    sources = (
        Path("src/trotterlib/rpe_delta_compiled_cost_validation.py"),
        Path("scripts/run_rpe_delta_compiled_cost_validation.py"),
    )
    inputs = [
        args.schedule,
        args.calibration,
        args.angle_invariance,
        *transfer_paths,
        args.k4_validation,
        args.k4_angle_invariance,
    ]
    payload = validate_rpe_delta_compiled_cost(
        schedule,
        calibration,
        angle,
        transfer_validations=transfers,
        k4_validation=k4_validation,
        k4_angle_invariance=k4_angle,
        k4_minimum_event_count=args.k4_minimum_event_count,
        model_relative_tolerance=args.model_relative_tolerance,
        provenance={
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git(["rev-parse", "HEAD"]),
            "git_worktree_status_before_generation": _git(["status", "--short"]),
            "evidence_status": "local_worktree_validation_not_immutable_ci",
            "command": shlex.join(
                [
                    ".venv311/bin/python",
                    "scripts/run_rpe_delta_compiled_cost_validation.py",
                    *sys.argv[1:],
                ]
            ),
            "python_version": platform.python_version(),
            "qiskit_version": qiskit.__version__,
            "inputs": {
                str(path): {"sha256": _sha256(path)} for path in inputs
            },
            "source_sha256": {str(path): _sha256(path) for path in sources},
        },
    )
    write_rpe_delta_compiled_cost_validation(payload, args.output)
    print(args.output)
    print(payload["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
