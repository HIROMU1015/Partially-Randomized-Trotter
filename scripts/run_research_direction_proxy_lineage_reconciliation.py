#!/usr/bin/env python3
"""Run A0: reconcile fresh q=1,2 proxy with legacy opt2 q=16,32 holdouts."""

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
from trotterlib.research_direction_proxy_lineage_reconciliation import (
    evaluate_proxy_lineage_reconciliation,
    finalize_proxy_lineage_reconciliation_artifact,
    read_json_object,
    write_proxy_lineage_reconciliation_artifact,
)


DEFAULT_LATEST = Path(
    "artifacts/research_direction_full_opt2/2026-09-25/"
    "wp11_all_r_opt2_coherent_analysis_20260925_065827.json"
)
DEFAULT_LEGACY = Path(
    "artifacts/research_direction_compiler_transfer/2026-09-23/"
    "m06_l08_opt2_same_trajectory_compute_v1.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/research_direction_full_opt2/2026-09-25/"
    "m06f_a0_proxy_lineage_reconciliation_v1.json"
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
    parser.add_argument("--latest", type=Path, default=DEFAULT_LATEST)
    parser.add_argument("--legacy", type=Path, default=DEFAULT_LEGACY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    latest = read_json_object(args.latest)
    legacy = read_json_object(args.legacy)
    body = evaluate_proxy_lineage_reconciliation(latest, legacy)
    body["source_evidence"] = {
        "latest_fresh_proxy": {
            "path": str(args.latest),
            "sha256": file_sha256(args.latest),
            "content_fingerprint": latest["content_fingerprint"],
        },
        "legacy_fixed_holdouts": {
            "path": str(args.legacy),
            "sha256": file_sha256(args.legacy),
            "content_fingerprint": legacy["content_fingerprint"],
        },
    }
    source_paths = (
        Path("src/trotterlib/research_direction_proxy_lineage_reconciliation.py"),
        Path("scripts/run_research_direction_proxy_lineage_reconciliation.py"),
    )
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_worktree_status_before_generation": _git(["status", "--short"]),
        "evidence_status": "local_worktree_analysis_not_externally_reproduced",
        "command": shlex.join(
            [
                ".venv311/bin/python",
                "scripts/run_research_direction_proxy_lineage_reconciliation.py",
                *sys.argv[1:],
            ]
        ),
        "python_version": platform.python_version(),
        "source_sha256": {
            str(path): file_sha256(path) for path in source_paths
        },
    }
    artifact = finalize_proxy_lineage_reconciliation_artifact(
        body, provenance=provenance
    )
    write_proxy_lineage_reconciliation_artifact(artifact, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "content_fingerprint": artifact["content_fingerprint"],
                "summary": artifact["summary"],
                "decision": artifact["decision"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
