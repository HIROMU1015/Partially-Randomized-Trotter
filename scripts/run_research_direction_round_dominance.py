#!/usr/bin/env python3
"""Run G08 round-wise cost, risk, and proxy-uncertainty analysis."""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from trotterlib.research_direction_decision_cost import (
    validate_wp01d_compute_artifact,
)
from trotterlib.research_direction_decision_synthesis import (
    validate_wp01d_synthesis_artifact,
)
from trotterlib.research_direction_prevalidation import file_sha256
from trotterlib.research_direction_round_dominance import (
    evaluate_g08_round_dominance,
    finalize_g08_artifact,
    write_g08_artifact,
)


DEFAULT_COMPUTE = Path(
    "artifacts/research_direction_decision_cost/2026-09-22/"
    "wp01d_c07_full_scope_optimization_compute_v2.json"
)
DEFAULT_SYNTHESIS = Path(
    "artifacts/research_direction_decision_cost/2026-09-22/"
    "wp01d_c07_conditional_interval_synthesis_v1.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/research_direction_round_dominance/2026-09-22/"
    "g08_round_cost_risk_proxy_dominance_v1.json"
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
        Path("src/trotterlib/research_direction_round_dominance.py"),
        Path("scripts/run_research_direction_round_dominance.py"),
    )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_worktree_status_before_generation": _git(["status", "--short"]),
        "evidence_status": "local_worktree_validation_not_immutable_ci",
        "command": shlex.join(
            [
                ".venv311/bin/python",
                "scripts/run_research_direction_round_dominance.py",
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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    compute = json.loads(args.compute.read_text(encoding="utf-8"))
    validate_wp01d_compute_artifact(compute)
    synthesis = json.loads(args.synthesis.read_text(encoding="utf-8"))
    validate_wp01d_synthesis_artifact(synthesis)
    body = evaluate_g08_round_dominance(compute, synthesis)
    body["source_evidence"] = {
        "compute": {
            "path": str(args.compute),
            "sha256": file_sha256(args.compute),
            "content_fingerprint": compute["content_fingerprint"],
        },
        "synthesis": {
            "path": str(args.synthesis),
            "sha256": file_sha256(args.synthesis),
            "content_fingerprint": synthesis["content_fingerprint"],
        },
    }
    artifact = finalize_g08_artifact(body, provenance=_provenance())
    write_g08_artifact(artifact, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "content_fingerprint": artifact["content_fingerprint"],
                "overall_pass": artifact["overall_pass"],
                "summary": artifact["summary"],
                "m08_target": artifact["m08_target"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
