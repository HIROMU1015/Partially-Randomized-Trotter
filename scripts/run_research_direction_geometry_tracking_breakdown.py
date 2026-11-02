#!/usr/bin/env python3
"""Run the preregistered P-C geometry tracking and breakdown validation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import tempfile
import time

import numpy as np
import scipy

from trotterlib.parallel_validation_executor import file_sha256
from trotterlib.research_direction_geometry_tracking_breakdown import (
    build_geometry_bundles,
    evaluate_geometry_tracking_breakdown,
    expected_task_manifest_body,
    finalize_expected_task_manifest,
    finalize_geometry_tracking_breakdown_artifact,
    read_json_object,
    validate_expected_task_manifest,
    write_json_nonoverwriting,
)


DEFAULT_EXPECTED = Path(
    "artifacts/research_direction_geometry_tracking_breakdown/2026-09-25/"
    "pc_tracking_breakdown_expected_tasks_v1.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/research_direction_geometry_tracking_breakdown/2026-09-25/"
    "pc_tracking_breakdown_validation_v1.json"
)
DEFAULT_PRIOR = Path(
    "artifacts/research_direction_geometry_energy_difference_pilot/2026-09-25/"
    "pc_h4_geometry_signed_error_v1_corrected_energy.json"
)
SOURCE_PATHS = (
    Path(
        "src/trotterlib/"
        "research_direction_geometry_tracking_breakdown.py"
    ),
    Path("scripts/run_research_direction_geometry_tracking_breakdown.py"),
)


def _git(command: list[str]) -> str | list[str] | None:
    result = subprocess.run(
        ["git", *command],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    lines = result.stdout.splitlines()
    return lines[0] if len(lines) == 1 else lines


def _source_hashes(root: Path) -> dict[str, str]:
    return {
        str(path): file_sha256(root / path)
        for path in SOURCE_PATHS
    }


def _provenance(
    root: Path,
    *,
    command: str,
    mode: str,
    elapsed_seconds: float | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_worktree_status_before_generation": _git(["status", "--short"]),
        "evidence_status": "local_dirty_worktree_not_externally_reproduced",
        "command": command,
        "mode": mode,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "source_sha256": _source_hashes(root),
        "cpu_count_visible": os.cpu_count(),
    }
    if elapsed_seconds is not None:
        payload["elapsed_seconds"] = float(elapsed_seconds)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--expected", type=Path, default=DEFAULT_EXPECTED)
    parser.add_argument("--prior-artifact", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = _parser().parse_args()
    root = args.root.resolve()
    expected_path = (
        args.expected
        if args.expected.is_absolute()
        else root / args.expected
    )
    prior_path = (
        args.prior_artifact
        if args.prior_artifact.is_absolute()
        else root / args.prior_artifact
    )
    output_path = (
        args.output if args.output.is_absolute() else root / args.output
    )
    command = shlex.join(
        [
            ".venv311/bin/python",
            "scripts/run_research_direction_geometry_tracking_breakdown.py",
            *sys.argv[1:],
        ]
    )

    if args.dry_run:
        expected = finalize_expected_task_manifest(
            expected_task_manifest_body(),
            provenance=_provenance(
                root,
                command=command,
                mode="compile_before_results_expected_tasks",
            ),
        )
        write_json_nonoverwriting(expected, expected_path)
        print(expected_path)
        print(expected["content_fingerprint"])
        print(expected["task_count"])
        return 0

    if not expected_path.is_file():
        raise FileNotFoundError(
            "Run --dry-run before the P-C tracking computation."
        )
    if not prior_path.is_file():
        raise FileNotFoundError(f"Missing prior P-C artifact: {prior_path}")
    expected = read_json_object(expected_path)
    validate_expected_task_manifest(expected)
    frozen_hashes = expected["provenance"]["source_sha256"]
    current_hashes = _source_hashes(root)
    if frozen_hashes != current_hashes:
        raise ValueError(
            "P-C tracking module or runner changed after dry-run."
        )

    prior = read_json_object(prior_path)
    started = time.perf_counter()
    print("P-C tracking: building 8 fixed H4 geometries", flush=True)
    with tempfile.TemporaryDirectory(
        prefix="pc_geometry_tracking_",
        dir="/tmp",
    ) as temporary_directory:
        bundles = build_geometry_bundles(Path(temporary_directory))
        print("P-C tracking: evaluating independent/tracked PF policies", flush=True)
        body = evaluate_geometry_tracking_breakdown(
            bundles,
            prior,
            expected,
        )
    elapsed = time.perf_counter() - started
    provenance = _provenance(
        root,
        command=command,
        mode="preregistered_full_computation",
        elapsed_seconds=elapsed,
    )
    artifact = finalize_geometry_tracking_breakdown_artifact(
        body,
        provenance=provenance,
        source_evidence=(
            {
                "path": str(args.expected),
                "sha256": file_sha256(expected_path),
                "content_fingerprint": expected["content_fingerprint"],
                "role": "compile_before_expected_tasks",
            },
            {
                "path": str(args.prior_artifact),
                "sha256": file_sha256(prior_path),
                "content_fingerprint": prior["content_fingerprint"],
                "role": "prior_pc_signed_error_pilot",
            },
        ),
    )
    write_json_nonoverwriting(artifact, output_path)
    print(output_path)
    print(artifact["content_fingerprint"])
    print(artifact["decision"]["status"])
    print(f"elapsed_seconds={elapsed:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
