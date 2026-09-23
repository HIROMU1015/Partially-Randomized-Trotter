#!/usr/bin/env python3
"""Run WP11 scoped direction synthesis and select one follow-up."""

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
from trotterlib.research_direction_wp11_synthesis import (
    evaluate_wp11_synthesis,
    finalize_wp11_artifact,
    write_wp11_artifact,
)


DEFAULTS = {
    "gate_s1": Path(
        "artifacts/research_direction_gate_s1/2026-09-22/"
        "gate_s1_research_direction_decision_v1.json"
    ),
    "wp06a": Path(
        "artifacts/research_direction_structure_pilot/2026-09-22/"
        "wp06a_circuit_structure_pilot_v1.json"
    ),
    "wp06b": Path(
        "artifacts/research_direction_sequence_policy/2026-09-22/"
        "wp06b_sequence_policy_proxy_bridge_v1.json"
    ),
    "wp05a": Path(
        "artifacts/research_direction_full_scope/2026-09-22/"
        "wp05a_full_controlled_interrogation_connection_v1.json"
    ),
    "wp05b": Path(
        "artifacts/research_direction_full_scope_extension/2026-09-22/"
        "wp05b_q8_delta_0p01_full_scope_extension_v1.json"
    ),
    "wp05br": Path(
        "artifacts/research_direction_full_scope_replication/2026-09-22/"
        "wp05br_r32_32trajectory_replication_v1.json"
    ),
    "wp01d": Path(
        "artifacts/research_direction_decision_cost/2026-09-22/"
        "wp01d_c07_full_scope_optimization_compute_v2.json"
    ),
    "g08": Path(
        "artifacts/research_direction_round_dominance/2026-09-22/"
        "g08_round_cost_risk_proxy_dominance_v1.json"
    ),
    "m08_reaggregation": Path(
        "artifacts/research_direction_proxy_precision/2026-09-22/"
        "wp01d_c07_m08_measured_discrepancy_reaggregation_v1.json"
    ),
    "compiler_transfer": Path(
        "artifacts/research_direction_compiler_transfer/2026-09-23/"
        "m06_l08_opt2_focused_analysis_reaggregation_v1.json"
    ),
    "n07_p03": Path(
        "artifacts/research_direction_uncertainty_break_even/2026-09-23/"
        "n07_p03_uncertainty_break_even_v1.json"
    ),
}
DEFAULT_OUTPUT = Path(
    "artifacts/research_direction_wp11_synthesis/2026-09-23/"
    "wp11_scoped_direction_synthesis_v1.json"
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
    for name, default in DEFAULTS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, default=default)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    paths = {name: getattr(args, name) for name in DEFAULTS}
    inputs = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
    }
    body = evaluate_wp11_synthesis(**inputs)
    body["source_evidence"] = {
        name: {
            "path": str(path),
            "sha256": file_sha256(path),
            "content_fingerprint": inputs[name]["content_fingerprint"],
        }
        for name, path in paths.items()
    }
    sources = (
        Path("src/trotterlib/research_direction_wp11_synthesis.py"),
        Path("scripts/run_research_direction_wp11_synthesis.py"),
    )
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_worktree_status_before_generation": _git(["status", "--short"]),
        "evidence_status": "local_worktree_synthesis_not_externally_reproduced",
        "command": shlex.join(
            [
                ".venv311/bin/python",
                "scripts/run_research_direction_wp11_synthesis.py",
                *sys.argv[1:],
            ]
        ),
        "python_version": platform.python_version(),
        "source_sha256": {str(path): file_sha256(path) for path in sources},
    }
    artifact = finalize_wp11_artifact(body, provenance=provenance)
    write_wp11_artifact(artifact, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "content_fingerprint": artifact["content_fingerprint"],
                "summary": artifact["summary"],
                "current_decision": artifact["current_decision"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
