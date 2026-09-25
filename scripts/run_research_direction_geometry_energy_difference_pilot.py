#!/usr/bin/env python3
"""Analyze the fixed five-geometry H4 P-C signed-error pilot."""

from __future__ import annotations

import argparse
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from trotterlib.parallel_validation_executor import file_sha256
from trotterlib.research_direction_geometry_energy_difference_pilot import (
    EXPECTED_GEOMETRIES,
    evaluate_geometry_energy_difference_pilot,
    finalize_geometry_energy_difference_pilot_artifact,
    read_json_object,
    write_geometry_energy_difference_pilot_artifact,
)


DEFAULT_INPUT_DIRECTORY = Path(
    "artifacts/research_direction_geometry_energy_difference_pilot/"
    "2026-09-25/inputs"
)
DEFAULT_OUTPUT = Path(
    "artifacts/research_direction_geometry_energy_difference_pilot/"
    "2026-09-25/pc_h4_geometry_signed_error_v1_corrected_energy.json"
)


def _git(command: list[str]) -> str | list[str] | None:
    result = subprocess.run(
        ["git", *command], check=False, capture_output=True, text=True
    )
    if result.returncode != 0:
        return None
    lines = result.stdout.splitlines()
    return lines[0] if len(lines) == 1 else lines


def _distance_label(distance: float) -> str:
    return f"{int(round(100 * distance)):03d}"


def _input_paths(directory: Path) -> list[Path]:
    return [
        directory
        / f"h4_sto3g_d{_distance_label(distance)}_rank12_ld3_v5.json"
        for distance in EXPECTED_GEOMETRIES
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-directory", type=Path, default=DEFAULT_INPUT_DIRECTORY
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    input_paths = _input_paths(args.input_directory)
    missing = [str(path) for path in input_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing P-C input artifacts: {missing}")
    payloads = [read_json_object(path) for path in input_paths]
    body = evaluate_geometry_energy_difference_pilot(payloads)
    body["source_evidence"] = [
        {
            "path": str(path),
            "sha256": file_sha256(path),
            "validation_fingerprint": payload["validation_fingerprint"],
            "geometry_angstrom": float(
                payload["hamiltonian"]["metadata"]["distance"]
            ),
        }
        for path, payload in zip(input_paths, payloads)
    ]
    source_paths = (
        Path(
            "src/trotterlib/"
            "research_direction_geometry_energy_difference_pilot.py"
        ),
        Path("scripts/run_research_direction_geometry_energy_difference_pilot.py"),
    )
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_worktree_status_before_generation": _git(["status", "--short"]),
        "evidence_status": "local_dirty_worktree_not_externally_reproduced",
        "command": shlex.join([sys.executable, *sys.argv]),
        "python_version": platform.python_version(),
        "source_sha256": {
            str(path): file_sha256(path) for path in source_paths
        },
        "input_count": len(input_paths),
        "discarded_preflight": {
            "geometry_angstrom": 0.9,
            "purpose": "runtime_and_execution_path_check_only",
            "included_in_evidence": False,
        },
    }
    artifact = finalize_geometry_energy_difference_pilot_artifact(
        body, provenance=provenance
    )
    write_geometry_energy_difference_pilot_artifact(artifact, args.output)
    print(args.output)
    print(artifact["content_fingerprint"])
    print(artifact["decision"]["status"])


if __name__ == "__main__":
    main()
