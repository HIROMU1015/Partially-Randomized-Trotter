#!/usr/bin/env python3
"""Run the WP01-D/C07 full-scope numerical optimization stage."""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import qiskit

from trotterlib.df_partial_randomized_pf import split_df_hamiltonian_by_ld
from trotterlib.df_partial_s2 import prepare_df_partial_s2
from trotterlib.research_direction_decision_cost import (
    evaluate_wp01d_c07_compute,
    finalize_wp01d_compute_artifact,
    write_wp01d_compute_artifact,
)
from trotterlib.research_direction_prevalidation import file_sha256
from trotterlib.rte_connected_cluster_cost_validation import (
    load_connected_cluster_hamiltonian_snapshot,
)


DEFAULT_SNAPSHOT = Path(
    "artifacts/rte_connected_cluster_cost_validation/"
    "h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k2_connected_"
    "pilot30_max1500_hold1500_rare375_v1.hamiltonian.npz"
)
DEFAULT_WP01 = Path(
    "artifacts/research_direction_prevalidation/2026-09-21/"
    "wp01s_model_conditional_screening_v1.json"
)
DEFAULT_WP05A = Path(
    "artifacts/research_direction_full_scope/2026-09-22/"
    "wp05a_full_controlled_interrogation_connection_v1.json"
)
DEFAULT_WP05B = Path(
    "artifacts/research_direction_full_scope_extension/2026-09-22/"
    "wp05b_q8_delta_0p01_full_scope_extension_v1.json"
)
DEFAULT_WP05BR = Path(
    "artifacts/research_direction_full_scope_replication/2026-09-22/"
    "wp05br_r32_32trajectory_replication_v1.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/research_direction_decision_cost/2026-09-22/"
    "wp01d_c07_full_scope_optimization_compute_v2.json"
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
        Path("src/trotterlib/research_direction_ablation.py"),
        Path("src/trotterlib/research_direction_decision_cost.py"),
        Path("scripts/run_research_direction_decision_cost.py"),
    )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_worktree_status_before_generation": _git(["status", "--short"]),
        "evidence_status": "local_worktree_validation_not_immutable_ci",
        "command": shlex.join(
            [
                ".venv311/bin/python",
                "scripts/run_research_direction_decision_cost.py",
                *sys.argv[1:],
            ]
        ),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "qiskit_version": qiskit.__version__,
        "source_sha256": {str(path): file_sha256(path) for path in sources},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--wp01", type=Path, default=DEFAULT_WP01)
    parser.add_argument("--wp05a", type=Path, default=DEFAULT_WP05A)
    parser.add_argument("--wp05b", type=Path, default=DEFAULT_WP05B)
    parser.add_argument("--wp05br", type=Path, default=DEFAULT_WP05BR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    inputs = {
        "wp01": json.loads(args.wp01.read_text(encoding="utf-8")),
        "wp05a": json.loads(args.wp05a.read_text(encoding="utf-8")),
        "wp05b": json.loads(args.wp05b.read_text(encoding="utf-8")),
        "wp05br": json.loads(args.wp05br.read_text(encoding="utf-8")),
    }
    hamiltonian = load_connected_cluster_hamiltonian_snapshot(args.snapshot)
    preparations = {
        ld: prepare_df_partial_s2(
            hamiltonian,
            split_df_hamiltonian_by_ld(hamiltonian, ld),
            identity_policy="extract_identity_phase",
        )
        for ld in (3, 12)
    }
    body = evaluate_wp01d_c07_compute(
        preparations,
        inputs["wp01"],
        inputs["wp05a"],
        inputs["wp05b"],
        inputs["wp05br"],
        progress=lambda message: print(message, flush=True),
    )
    body["source_evidence"] = {
        "snapshot": {
            "path": str(args.snapshot),
            "sha256": file_sha256(args.snapshot),
        },
        **{
            name: {
                "path": str(path),
                "sha256": file_sha256(path),
                "content_fingerprint": inputs[name]["content_fingerprint"],
            }
            for name, path in (
                ("wp01", args.wp01),
                ("wp05a", args.wp05a),
                ("wp05b", args.wp05b),
                ("wp05br", args.wp05br),
            )
        },
    }
    artifact = finalize_wp01d_compute_artifact(
        body, provenance=_provenance()
    )
    write_wp01d_compute_artifact(artifact, args.output)
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
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
