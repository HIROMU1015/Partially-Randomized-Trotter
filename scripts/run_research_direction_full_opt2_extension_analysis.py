#!/usr/bin/env python3
"""Audit M06-F fresh-32 results and run coherent opt2 reoptimization."""

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
from trotterlib.research_direction_full_opt2_extension_analysis import (
    evaluate_coherent_extension_analysis,
    evaluate_extension_audit,
    finalize_extension_analysis,
    finalize_extension_audit,
    write_extension_artifact,
)


def _git(root: Path, *args: str) -> str | list[str] | None:
    result = subprocess.run(
        ["git", *args], cwd=root, check=False, capture_output=True, text=True
    )
    if result.returncode != 0:
        return None
    lines = result.stdout.splitlines()
    return lines[0] if len(lines) == 1 else lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-manifest", type=Path, required=True)
    parser.add_argument("--initial-output-dir", type=Path, required=True)
    parser.add_argument("--extension-manifest", type=Path, required=True)
    parser.add_argument("--extension-output-dir", type=Path, required=True)
    parser.add_argument("--initial-analysis", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--analysis-output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sources = (
        Path("src/trotterlib/research_direction_full_opt2_extension_analysis.py"),
        Path("scripts/run_research_direction_full_opt2_extension_analysis.py"),
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
    common = {
        "project_root": root,
        "initial_manifest_path": args.initial_manifest,
        "initial_output_dir": args.initial_output_dir,
        "extension_manifest_path": args.extension_manifest,
        "extension_output_dir": args.extension_output_dir,
        "initial_analysis_path": args.initial_analysis,
    }
    audit_body = evaluate_extension_audit(**common)
    audit = finalize_extension_audit(audit_body, provenance=provenance)
    write_extension_artifact(audit, args.audit_output)
    if not audit["gate_summary"]["all_proxy_cells_pass"]:
        print(json.dumps({"audit": str(args.audit_output), "status": audit["status"]}))
        return 2
    analysis_body = evaluate_coherent_extension_analysis(
        **common,
        progress=lambda message: print(message, flush=True),
    )
    analysis = finalize_extension_analysis(analysis_body, provenance=provenance)
    write_extension_artifact(analysis, args.analysis_output)
    print(
        json.dumps(
            {
                "audit": str(args.audit_output),
                "audit_fingerprint": audit["content_fingerprint"],
                "analysis": str(args.analysis_output),
                "analysis_fingerprint": analysis["content_fingerprint"],
                "status": analysis["status"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
