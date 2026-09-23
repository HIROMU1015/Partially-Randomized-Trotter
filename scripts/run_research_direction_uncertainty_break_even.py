#!/usr/bin/env python3
"""Run N07 uncertainty-ledger and P03 preparation break-even analysis."""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from trotterlib.research_direction_prevalidation import file_sha256
from trotterlib.research_direction_uncertainty_break_even import (
    evaluate_uncertainty_break_even,
    finalize_uncertainty_break_even_artifact,
    write_uncertainty_break_even_artifact,
)


DEFAULT_WP01D = Path(
    "artifacts/research_direction_decision_cost/2026-09-22/"
    "wp01d_c07_full_scope_optimization_compute_v2.json"
)
DEFAULT_M08_REAGGREGATION = Path(
    "artifacts/research_direction_proxy_precision/2026-09-22/"
    "wp01d_c07_m08_measured_discrepancy_reaggregation_v1.json"
)
DEFAULT_COMPILER_TRANSFER = Path(
    "artifacts/research_direction_compiler_transfer/2026-09-23/"
    "m06_l08_opt2_focused_analysis_reaggregation_v1.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/research_direction_uncertainty_break_even/2026-09-23/"
    "n07_p03_uncertainty_break_even_v1.json"
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
    parser.add_argument("--wp01d", type=Path, default=DEFAULT_WP01D)
    parser.add_argument(
        "--m08-reaggregation", type=Path, default=DEFAULT_M08_REAGGREGATION
    )
    parser.add_argument(
        "--compiler-transfer", type=Path, default=DEFAULT_COMPILER_TRANSFER
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    paths = {
        "wp01d": args.wp01d,
        "m08_reaggregation": args.m08_reaggregation,
        "compiler_transfer": args.compiler_transfer,
    }
    inputs = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
    }
    body = evaluate_uncertainty_break_even(
        inputs["wp01d"],
        inputs["m08_reaggregation"],
        inputs["compiler_transfer"],
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
        Path("src/trotterlib/research_direction_uncertainty_break_even.py"),
        Path("scripts/run_research_direction_uncertainty_break_even.py"),
    )
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_worktree_status_before_generation": _git(["status", "--short"]),
        "evidence_status": "local_worktree_analysis_not_externally_reproduced",
        "command": shlex.join(
            [
                ".venv311/bin/python",
                "scripts/run_research_direction_uncertainty_break_even.py",
                *sys.argv[1:],
            ]
        ),
        "python_version": platform.python_version(),
        "source_sha256": {str(path): file_sha256(path) for path in sources},
    }
    artifact = finalize_uncertainty_break_even_artifact(
        body, provenance=provenance
    )
    write_uncertainty_break_even_artifact(artifact, args.output)
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
