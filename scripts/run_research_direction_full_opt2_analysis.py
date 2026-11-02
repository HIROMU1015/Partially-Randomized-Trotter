#!/usr/bin/env python3
"""Analyze a complete M06-F batch and run coherent opt2 reoptimization."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from trotterlib.parallel_validation_executor import file_sha256
from trotterlib.research_direction_full_opt2 import (
    create_full_opt2_task_manifest,
    evaluate_full_opt2_analysis,
    extension_cells_from_analysis,
    finalize_full_opt2_analysis_artifact,
    write_full_opt2_analysis_artifact,
)


def _git(*args: str) -> str | list[str] | None:
    result = subprocess.run(
        ["git", *args], check=False, capture_output=True, text=True
    )
    if result.returncode != 0:
        return None
    lines = result.stdout.splitlines()
    return lines[0] if len(lines) == 1 else lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--compute-output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extension-manifest", type=Path)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    body = evaluate_full_opt2_analysis(
        project_root=project_root,
        manifest_path=args.task_manifest,
        compute_output_dir=args.compute_output_dir,
        progress=lambda message: print(message, flush=True),
    )
    sources = (
        Path("src/trotterlib/research_direction_full_opt2.py"),
        Path("scripts/run_research_direction_full_opt2_analysis.py"),
    )
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_status": _git("status", "--short"),
        "command": shlex.join([sys.executable, *sys.argv]),
        "python_version": platform.python_version(),
        "qiskit_version": importlib.metadata.version("qiskit"),
        "numpy_version": importlib.metadata.version("numpy"),
        "scipy_version": importlib.metadata.version("scipy"),
        "source_sha256": {
            str(path): file_sha256(project_root / path) for path in sources
        },
        "evidence_status": "local_dirty_worktree_analysis_not_immutable_ci",
    }
    artifact = finalize_full_opt2_analysis_artifact(body, provenance=provenance)
    write_full_opt2_analysis_artifact(artifact, args.output)
    extension_path = None
    cells = extension_cells_from_analysis(artifact)
    if cells:
        if args.extension_manifest is None:
            raise SystemExit(
                "Proxy criteria failed; rerun with --extension-manifest to prepare "
                "the fresh 32-trajectory cells."
            )
        extension = create_full_opt2_task_manifest(
            args.extension_manifest,
            project_root=project_root,
            batch_id="wp11-all-r-opt2-fresh-32-extension",
            extension_cells=cells,
            include_deterministic=False,
        )
        extension_path = {
            "path": str(args.extension_manifest),
            "manifest_fingerprint": extension["manifest_fingerprint"],
            "task_count": len(extension["tasks"]),
        }
    print(
        json.dumps(
            {
                "output": str(args.output),
                "content_fingerprint": artifact["content_fingerprint"],
                "status": artifact["status"],
                "overall_pass": artifact["overall_pass"],
                "extension_manifest": extension_path,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
