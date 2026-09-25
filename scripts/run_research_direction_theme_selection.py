#!/usr/bin/env python3
"""Select a provisional research theme from the completed P-B/P-C/P-A pilots."""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from trotterlib.parallel_validation_executor import file_sha256
from trotterlib.research_direction_theme_selection import (
    evaluate_theme_selection,
    finalize_theme_selection_artifact,
    write_theme_selection_artifact,
)


DEFAULTS = {
    "pb": Path(
        "artifacts/research_direction_signal_weight_pilot/2026-09-25/"
        "pb_h4_s2_prefix_grid_v1.json"
    ),
    "pc": Path(
        "artifacts/research_direction_geometry_energy_difference_pilot/2026-09-25/"
        "pc_h4_geometry_signed_error_v1.json"
    ),
    "pa": Path(
        "artifacts/research_direction_joint_synthesis_pilot/2026-09-25/"
        "pa_h4_interval_union_joint_synthesis_v1.json"
    ),
}
DEFAULT_OUTPUT = Path(
    "artifacts/research_direction_theme_selection/2026-09-25/"
    "theme_selection_pb_pc_pa_v1.json"
)


def _git(command: list[str]) -> str | list[str] | None:
    result = subprocess.run(
        ["git", *command], check=False, capture_output=True, text=True
    )
    if result.returncode != 0:
        return None
    lines = result.stdout.splitlines()
    return lines[0] if len(lines) == 1 else lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in DEFAULTS.items():
        parser.add_argument(f"--{name}", type=Path, default=default)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    paths = {name: getattr(args, name) for name in DEFAULTS}
    inputs = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
    }
    body = evaluate_theme_selection(**inputs)
    body["source_evidence"] = {
        name: {
            "path": str(path),
            "sha256": file_sha256(path),
            "content_fingerprint": inputs[name]["content_fingerprint"],
        }
        for name, path in paths.items()
    }
    sources = (
        Path("src/trotterlib/research_direction_theme_selection.py"),
        Path("scripts/run_research_direction_theme_selection.py"),
    )
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_worktree_status_before_generation": _git(["status", "--short"]),
        "evidence_status": "local_dirty_worktree_not_externally_reproduced",
        "command": shlex.join([sys.executable, *sys.argv]),
        "python_version": platform.python_version(),
        "source_sha256": {str(path): file_sha256(path) for path in sources},
    }
    artifact = finalize_theme_selection_artifact(body, provenance=provenance)
    write_theme_selection_artifact(artifact, args.output)
    print(args.output)
    print(artifact["content_fingerprint"])
    print(artifact["decision"]["status"])


if __name__ == "__main__":
    main()
