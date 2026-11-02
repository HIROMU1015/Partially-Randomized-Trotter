#!/usr/bin/env python3
"""Reanalyze the frozen P-D S1 artifact without new scientific computation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import platform
import shlex
import subprocess
import sys

from trotterlib.research_direction_energy_tail_pareto import (
    read_json_object,
    write_json_nonoverwriting,
)
from trotterlib.research_direction_pd_s1_posthoc import (
    SOURCE_RESULT_FILE_SHA256,
    finalize_result,
    reanalyze_s1,
)


DEFAULT_INPUT = Path(
    "artifacts/research_direction_pd_fair_comparison/2026-09-26/"
    "pd_s1_fair_comparison_v2.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/research_direction_pd_fair_comparison/2026-09-26/"
    "pd_s1_posthoc_reanalysis_v1.json"
)
SOURCE_PATHS = (
    Path("pd_s1_review_5c331f0.md"),
    Path("docs/research/pd_s1_posthoc_reanalysis_plan.md"),
    Path("src/trotterlib/research_direction_pd_s1_posthoc.py"),
    Path("scripts/run_research_direction_pd_s1_posthoc.py"),
    DEFAULT_INPUT,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(command: list[str]) -> str | list[str] | None:
    result = subprocess.run(
        ["git", *command], check=False, capture_output=True, text=True
    )
    if result.returncode != 0:
        return None
    lines = result.stdout.splitlines()
    return lines[0] if len(lines) == 1 else lines


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def main() -> int:
    args = _parser().parse_args()
    root = args.root.resolve()
    input_path = _resolve(root, args.input)
    output_path = _resolve(root, args.output)
    if _sha256(input_path) != SOURCE_RESULT_FILE_SHA256:
        raise ValueError("Frozen P-D S1 input file SHA-256 changed.")
    source = read_json_object(input_path)
    body = reanalyze_s1(source)
    command = shlex.join(
        [
            ".venv311/bin/python",
            "scripts/run_research_direction_pd_s1_posthoc.py",
            *sys.argv[1:],
        ]
    )
    artifact = finalize_result(
        body,
        provenance={
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git(["rev-parse", "HEAD"]),
            "git_worktree_status_before_generation": _git(["status", "--short"]),
            "evidence_status": "local_dirty_worktree_not_externally_reproduced",
            "command": command,
            "mode": "posthoc_reanalysis_of_frozen_s1_artifact",
            "python_version": platform.python_version(),
        },
        source_evidence=[
            {
                "path": str(path),
                "sha256": _sha256(root / path),
                "role": (
                    "frozen_source_result"
                    if path == DEFAULT_INPUT
                    else "posthoc_plan_or_analysis_source"
                ),
            }
            for path in SOURCE_PATHS
        ],
    )
    write_json_nonoverwriting(artifact, output_path)
    print(output_path)
    print(artifact["content_fingerprint"])
    print(artifact["posthoc_interpretation"]["selection_label"])
    print(artifact["decision"]["pd_active_development_status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
