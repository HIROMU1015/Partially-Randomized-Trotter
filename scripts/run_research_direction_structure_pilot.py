#!/usr/bin/env python3
"""Run the WP06-a representative DF circuit-structure pilot."""

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
from trotterlib.research_direction_prevalidation import file_sha256
from trotterlib.research_direction_structure_pilot import (
    evaluate_wp06a_structure_pilot,
    finalize_wp06a_artifact,
    write_wp06a_artifact,
)
from trotterlib.rte import CompilerSettings
from trotterlib.rte_connected_cluster_cost_validation import (
    load_connected_cluster_hamiltonian_snapshot,
)


DEFAULT_SNAPSHOT = Path(
    "artifacts/rte_connected_cluster_cost_validation/"
    "h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k2_connected_"
    "pilot30_max1500_hold1500_rare375_v1.hamiltonian.npz"
)
DEFAULT_GATE_S1 = Path(
    "artifacts/research_direction_gate_s1/2026-09-22/"
    "gate_s1_research_direction_decision_v1.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/research_direction_structure_pilot/2026-09-22/"
    "wp06a_circuit_structure_pilot_v1.json"
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
        Path("src/trotterlib/research_direction_structure_pilot.py"),
        Path("scripts/run_research_direction_structure_pilot.py"),
    )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_worktree_status_before_generation": _git(["status", "--short"]),
        "evidence_status": "local_worktree_validation_not_immutable_ci",
        "command": shlex.join(
            [
                ".venv311/bin/python",
                "scripts/run_research_direction_structure_pilot.py",
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
    parser.add_argument("--gate-s1", type=Path, default=DEFAULT_GATE_S1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    gate_s1 = json.loads(args.gate_s1.read_text(encoding="utf-8"))
    rule = gate_s1["next_action"]["preregistered_decision_rule"]
    eta = float(rule["eta_decision_relative_rz"])
    if gate_s1["next_action"]["work_package"] != "WP06-a":
        raise ValueError("Gate S1 does not select WP06-a as the next action.")

    hamiltonian = load_connected_cluster_hamiltonian_snapshot(args.snapshot)
    preparation = prepare_df_partial_s2(
        hamiltonian,
        split_df_hamiltonian_by_ld(hamiltonian, 3),
        identity_policy="extract_identity_phase",
    )
    compiler = CompilerSettings(
        basis_gates=("rz", "sx", "x", "cx"),
        backend_name=None,
        coupling_map=None,
        optimization_level=1,
        layout_method=None,
        routing_method=None,
        transpiler_seed=17,
        qiskit_version=qiskit.__version__,
    )
    body = evaluate_wp06a_structure_pilot(
        hamiltonian,
        preparation,
        compiler,
        delta_time=0.02,
        eta_decision_relative_rz=eta,
    )
    body["source_evidence"] = {
        "snapshot": {
            "path": str(args.snapshot),
            "sha256": file_sha256(args.snapshot),
        },
        "gate_s1": {
            "path": str(args.gate_s1),
            "sha256": file_sha256(args.gate_s1),
            "content_fingerprint": gate_s1["content_fingerprint"],
        },
    }
    artifact = finalize_wp06a_artifact(body, provenance=_provenance())
    write_wp06a_artifact(artifact, args.output)
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
