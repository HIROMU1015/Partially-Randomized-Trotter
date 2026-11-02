#!/usr/bin/env python3
"""Run P-B by reanalyzing the fixed H4 PF bias/weight/signal artifacts."""

from __future__ import annotations

import argparse
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from trotterlib.parallel_validation_executor import file_sha256
from trotterlib.research_direction_signal_weight_pilot import (
    EXPECTED_LD_VALUES,
    evaluate_signal_weight_pilot,
    finalize_signal_weight_pilot_artifact,
    read_json_object,
    write_signal_weight_pilot_artifact,
)


DEFAULT_INPUT_DIRECTORY = Path("artifacts/pf_delta_validation")
DEFAULT_OUTPUT = Path(
    "artifacts/research_direction_signal_weight_pilot/2026-09-25/"
    "pb_h4_s2_prefix_grid_v1.json"
)


def _git(command: list[str]) -> str | list[str] | None:
    result = subprocess.run(
        ["git", *command], check=False, capture_output=True, text=True
    )
    if result.returncode != 0:
        return None
    lines = result.stdout.splitlines()
    return lines[0] if len(lines) == 1 else lines


def _input_paths(directory: Path) -> list[Path]:
    return [
        directory / f"h4_sto3g_d100_rank12_ld{ld}_v5.json"
        for ld in EXPECTED_LD_VALUES
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
        raise FileNotFoundError(f"Missing P-B input artifacts: {missing}")
    payloads = [read_json_object(path) for path in input_paths]
    body = evaluate_signal_weight_pilot(payloads)
    body["source_evidence"] = [
        {
            "path": str(path),
            "sha256": file_sha256(path),
            "validation_fingerprint": payload["validation_fingerprint"],
            "ld": int(payload["request"]["ld"]),
        }
        for path, payload in zip(input_paths, payloads)
    ]
    source_paths = (
        Path("src/trotterlib/research_direction_signal_weight_pilot.py"),
        Path("scripts/run_research_direction_signal_weight_pilot.py"),
    )
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_worktree_status_before_generation": _git(["status", "--short"]),
        "evidence_status": "local_worktree_analysis_not_externally_reproduced",
        "command": shlex.join([sys.executable, *sys.argv]),
        "python_version": platform.python_version(),
        "source_sha256": {
            str(path): file_sha256(path) for path in source_paths
        },
        "input_count": len(input_paths),
    }
    artifact = finalize_signal_weight_pilot_artifact(
        body, provenance=provenance
    )
    write_signal_weight_pilot_artifact(artifact, args.output)
    print(args.output)
    print(artifact["content_fingerprint"])
    print(artifact["decision"]["status"])


if __name__ == "__main__":
    main()
