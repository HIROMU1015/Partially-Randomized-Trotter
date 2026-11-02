#!/usr/bin/env python3
"""Analyze and reaggregate the focused M06/L08 compiler-transfer run."""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from trotterlib.research_direction_compiler_transfer_analysis import (
    evaluate_compiler_transfer_analysis,
    finalize_compiler_transfer_analysis_artifact,
    write_compiler_transfer_analysis_artifact,
)
from trotterlib.research_direction_prevalidation import file_sha256


DEFAULT_RAW = Path(
    "artifacts/research_direction_compiler_transfer/2026-09-23/"
    "m06_l08_opt2_same_trajectory_compute_v1.json"
)
DEFAULT_WP05BR = Path(
    "artifacts/research_direction_full_scope_replication/2026-09-22/"
    "wp05br_r32_32trajectory_replication_v1.json"
)
DEFAULT_M08 = Path(
    "artifacts/research_direction_proxy_precision/2026-09-22/"
    "m08_late_round_q16_q32_proxy_precision_v1.json"
)
DEFAULT_WP01D = Path(
    "artifacts/research_direction_decision_cost/2026-09-22/"
    "wp01d_c07_full_scope_optimization_compute_v2.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/research_direction_compiler_transfer/2026-09-23/"
    "m06_l08_opt2_focused_analysis_reaggregation_v1.json"
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
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--wp05br", type=Path, default=DEFAULT_WP05BR)
    parser.add_argument("--m08", type=Path, default=DEFAULT_M08)
    parser.add_argument("--wp01d", type=Path, default=DEFAULT_WP01D)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    paths = {
        "raw": args.raw,
        "wp05br": args.wp05br,
        "m08": args.m08,
        "wp01d": args.wp01d,
    }
    inputs = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
    }
    body = evaluate_compiler_transfer_analysis(
        inputs["raw"], inputs["wp05br"], inputs["m08"], inputs["wp01d"]
    )
    body["source_evidence"] = {
        name: {
            "path": str(path),
            "sha256": file_sha256(path),
            "content_fingerprint": inputs[name]["content_fingerprint"],
        }
        for name, path in paths.items()
    }
    sources = (
        Path("src/trotterlib/research_direction_compiler_transfer_analysis.py"),
        Path("scripts/run_research_direction_compiler_transfer_analysis.py"),
    )
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_worktree_status_before_generation": _git(["status", "--short"]),
        "evidence_status": "local_worktree_analysis_not_externally_reproduced",
        "command": shlex.join(
            [
                ".venv311/bin/python",
                "scripts/run_research_direction_compiler_transfer_analysis.py",
                *sys.argv[1:],
            ]
        ),
        "python_version": platform.python_version(),
        "source_sha256": {str(path): file_sha256(path) for path in sources},
    }
    artifact = finalize_compiler_transfer_analysis_artifact(
        body, provenance=provenance
    )
    write_compiler_transfer_analysis_artifact(artifact, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "content_fingerprint": artifact["content_fingerprint"],
                "summary": artifact["summary"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
