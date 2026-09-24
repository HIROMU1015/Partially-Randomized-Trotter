#!/usr/bin/env python3
"""Audit the completed M06-F files and emit a gated analysis artifact."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from trotterlib.parallel_validation_executor import file_sha256
from trotterlib.research_direction_full_opt2_completion import (
    build_completion_audit,
    finalize_completion_audit,
    write_completion_audit,
)


def _run(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, check=False, capture_output=True, text=True)


def _git(root: Path, *args: str) -> str | list[str] | None:
    result = _run(["git", *args], cwd=root)
    if result.returncode != 0:
        return None
    lines = result.stdout.splitlines()
    return lines[0] if len(lines) == 1 else lines


def _matching_processes(patterns: tuple[str, ...]) -> list[dict[str, object]]:
    current = os.getpid()
    parent = os.getppid()
    rows: list[dict[str, object]] = []
    proc = Path("/proc")
    for child in proc.iterdir():
        if not child.name.isdigit() or int(child.name) in (current, parent):
            continue
        try:
            command = (child / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (OSError, UnicodeDecodeError):
            continue
        if command and any(pattern in command for pattern in patterns):
            rows.append({"pid": int(child.name), "command": command.strip()})
    return sorted(rows, key=lambda row: int(row["pid"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-manifest", type=Path, required=True)
    parser.add_argument("--compute-output-dir", type=Path, required=True)
    parser.add_argument("--initial-analysis", type=Path, required=True)
    parser.add_argument("--extension-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tmux-session", default="wp11_all_r_opt2")
    parser.add_argument(
        "--original-main-worktree",
        type=Path,
        default=Path("/home/AbeHiromu/projects/partially-randomized-trotter"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]

    tmux = _run(["tmux", "has-session", "-t", args.tmux_session])
    process_patterns = (
        "run_research_direction_full_opt2_compute.py",
        "wp11_all_r_opt2_initial_20260924_023012",
        "wp11-all-r-opt2-fresh-32-extension",
    )
    runtime = {
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "tmux_session": args.tmux_session,
        "tmux_session_present": tmux.returncode == 0,
        "matching_processes": _matching_processes(process_patterns),
    }
    original_status = _git(args.original_main_worktree, "status", "--short", "--branch")
    gpu_doc = args.original_main_worktree / "docs/gpu_execution_environment.md"
    original = {
        "path": str(args.original_main_worktree),
        "git_status": original_status,
        "status_matches_expected": original_status
        == ["## main...origin/main", "?? docs/gpu_execution_environment.md"],
        "gpu_execution_environment_sha256": (
            file_sha256(gpu_doc) if gpu_doc.is_file() else None
        ),
    }
    body = build_completion_audit(
        project_root=root,
        initial_manifest_path=args.initial_manifest,
        compute_output_dir=args.compute_output_dir,
        initial_analysis_path=args.initial_analysis,
        extension_manifest_path=args.extension_manifest,
        runtime_observation=runtime,
        original_main_observation=original,
    )
    sources = (
        Path("src/trotterlib/research_direction_full_opt2_completion.py"),
        Path("scripts/run_research_direction_full_opt2_completion.py"),
    )
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git(root, "rev-parse", "HEAD"),
        "git_status": _git(root, "status", "--short"),
        "command": shlex.join([sys.executable, *sys.argv]),
        "python_version": platform.python_version(),
        "qiskit_version": importlib.metadata.version("qiskit"),
        "numpy_version": importlib.metadata.version("numpy"),
        "scipy_version": importlib.metadata.version("scipy"),
        "source_sha256": {
            str(path): file_sha256(root / path) for path in sources
        },
        "evidence_status": "local_dirty_worktree_analysis_not_immutable_ci",
    }
    artifact = finalize_completion_audit(body, provenance=provenance)
    write_completion_audit(artifact, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "content_fingerprint": artifact["content_fingerprint"],
                "status": artifact["status"],
                "initial_compute_integrity_pass": artifact[
                    "initial_compute_integrity_pass"
                ],
                "workflow_counts": artifact["execution"][
                    "preregistered_workflow"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
