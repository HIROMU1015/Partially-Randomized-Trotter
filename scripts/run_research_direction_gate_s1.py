#!/usr/bin/env python3
"""Synthesize the completed screening work packages into Gate S1."""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from trotterlib.research_direction_gate_s1 import (
    build_gate_s1_body,
    finalize_gate_s1_artifact,
    write_gate_s1_artifact,
)
from trotterlib.research_direction_prevalidation import file_sha256


DEFAULT_WP00 = Path(
    "artifacts/research_direction_prevalidation/2026-09-21/"
    "wp00_comparison_contract_v1.json"
)
DEFAULT_WP02 = Path(
    "artifacts/research_direction_prevalidation/2026-09-21/"
    "wp02_round_horizon_coverage_v1.json"
)
DEFAULT_WP01 = Path(
    "artifacts/research_direction_prevalidation/2026-09-21/"
    "wp01s_model_conditional_screening_v1.json"
)
DEFAULT_WP04 = Path(
    "artifacts/research_direction_ablation/2026-09-21/"
    "wp04_finite_rte_statistical_ablation_v1.json"
)
DEFAULT_WP03 = Path(
    "artifacts/research_direction_pf_sensitivity/2026-09-22/"
    "wp03_pf_coefficient_selection_sensitivity_v1.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/research_direction_gate_s1/2026-09-22/"
    "gate_s1_research_direction_decision_v1.json"
)


def _git(command: list[str]) -> str | list[str] | None:
    result = subprocess.run(
        ["git", *command], check=False, capture_output=True, text=True
    )
    if result.returncode != 0:
        return None
    lines = result.stdout.splitlines()
    return lines[0] if len(lines) == 1 else lines


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _ref(path: Path, payload: dict) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "content_fingerprint": payload["content_fingerprint"],
    }


def _provenance() -> dict[str, object]:
    sources = (
        Path("src/trotterlib/research_direction_gate_s1.py"),
        Path("scripts/run_research_direction_gate_s1.py"),
    )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_worktree_status_before_generation": _git(["status", "--short"]),
        "evidence_status": "local_worktree_validation_not_immutable_ci",
        "command": shlex.join(
            [
                ".venv311/bin/python",
                "scripts/run_research_direction_gate_s1.py",
                *sys.argv[1:],
            ]
        ),
        "python_version": platform.python_version(),
        "source_sha256": {str(path): file_sha256(path) for path in sources},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wp00", type=Path, default=DEFAULT_WP00)
    parser.add_argument("--wp02", type=Path, default=DEFAULT_WP02)
    parser.add_argument("--wp01", type=Path, default=DEFAULT_WP01)
    parser.add_argument("--wp04", type=Path, default=DEFAULT_WP04)
    parser.add_argument("--wp03", type=Path, default=DEFAULT_WP03)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    wp00 = _load(args.wp00)
    wp02 = _load(args.wp02)
    wp01 = _load(args.wp01)
    wp04 = _load(args.wp04)
    wp03 = _load(args.wp03)
    body = build_gate_s1_body(
        wp00=wp00,
        wp02=wp02,
        wp01s=wp01,
        wp04=wp04,
        wp03=wp03,
    )
    body["source_evidence"] = {
        "WP00": _ref(args.wp00, wp00),
        "WP02": _ref(args.wp02, wp02),
        "WP01-S": _ref(args.wp01, wp01),
        "WP04": _ref(args.wp04, wp04),
        "WP03": _ref(args.wp03, wp03),
    }
    artifact = finalize_gate_s1_artifact(body, provenance=_provenance())
    write_gate_s1_artifact(artifact, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "content_fingerprint": artifact["content_fingerprint"],
                "overall_pass": artifact["overall_pass"],
                "summary": artifact["summary"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
