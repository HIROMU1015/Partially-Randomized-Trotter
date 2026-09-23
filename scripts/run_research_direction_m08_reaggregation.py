#!/usr/bin/env python3
"""Reaggregate WP01-D/C07 intervals with measured M08 discrepancy."""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from trotterlib.research_direction_m08_reaggregation import (
    evaluate_m08_reaggregation,
    finalize_m08_reaggregation_artifact,
    write_m08_reaggregation_artifact,
)
from trotterlib.research_direction_prevalidation import file_sha256


DEFAULT_COMPUTE = Path(
    "artifacts/research_direction_decision_cost/2026-09-22/"
    "wp01d_c07_full_scope_optimization_compute_v2.json"
)
DEFAULT_SYNTHESIS = Path(
    "artifacts/research_direction_decision_cost/2026-09-22/"
    "wp01d_c07_conditional_interval_synthesis_v1.json"
)
DEFAULT_G08 = Path(
    "artifacts/research_direction_round_dominance/2026-09-22/"
    "g08_round_cost_risk_proxy_dominance_v1.json"
)
DEFAULT_M08 = Path(
    "artifacts/research_direction_proxy_precision/2026-09-22/"
    "m08_late_round_q16_q32_proxy_precision_v1.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/research_direction_proxy_precision/2026-09-22/"
    "wp01d_c07_m08_measured_discrepancy_reaggregation_v1.json"
)


def _git(command: list[str]) -> str | list[str] | None:
    result = subprocess.run(
        ["git", *command], check=False, capture_output=True, text=True
    )
    if result.returncode != 0:
        return None
    lines = result.stdout.splitlines()
    return lines[0] if len(lines) == 1 else lines


def _provenance() -> dict[str, object]:
    sources = (
        Path("src/trotterlib/research_direction_m08_reaggregation.py"),
        Path("scripts/run_research_direction_m08_reaggregation.py"),
    )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_worktree_status_before_generation": _git(["status", "--short"]),
        "evidence_status": "local_worktree_validation_not_immutable_ci",
        "command": shlex.join(
            [
                ".venv311/bin/python",
                "scripts/run_research_direction_m08_reaggregation.py",
                *sys.argv[1:],
            ]
        ),
        "python_version": platform.python_version(),
        "source_sha256": {str(path): file_sha256(path) for path in sources},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compute", type=Path, default=DEFAULT_COMPUTE)
    parser.add_argument("--synthesis", type=Path, default=DEFAULT_SYNTHESIS)
    parser.add_argument("--g08", type=Path, default=DEFAULT_G08)
    parser.add_argument("--m08", type=Path, default=DEFAULT_M08)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    inputs = {
        "compute": json.loads(args.compute.read_text(encoding="utf-8")),
        "synthesis": json.loads(args.synthesis.read_text(encoding="utf-8")),
        "g08": json.loads(args.g08.read_text(encoding="utf-8")),
        "m08": json.loads(args.m08.read_text(encoding="utf-8")),
    }
    body = evaluate_m08_reaggregation(
        inputs["compute"],
        inputs["synthesis"],
        inputs["g08"],
        inputs["m08"],
    )
    paths = {
        "compute": args.compute,
        "synthesis": args.synthesis,
        "g08": args.g08,
        "m08": args.m08,
    }
    body["source_evidence"] = {
        name: {
            "path": str(path),
            "sha256": file_sha256(path),
            "content_fingerprint": inputs[name]["content_fingerprint"],
        }
        for name, path in paths.items()
    }
    artifact = finalize_m08_reaggregation_artifact(
        body, provenance=_provenance()
    )
    write_m08_reaggregation_artifact(artifact, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "content_fingerprint": artifact["content_fingerprint"],
                "overall_pass": artifact["overall_pass"],
                "summary": artifact["summary"],
                "decision": artifact["decision"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
