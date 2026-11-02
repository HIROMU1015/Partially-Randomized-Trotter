#!/usr/bin/env python3
"""Run the P-A interval-aware DF sequence synthesis pilot."""

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
from trotterlib.parallel_validation_executor import file_sha256
from trotterlib.research_direction_joint_synthesis_pilot import (
    evaluate_joint_synthesis_pilot,
    finalize_joint_synthesis_pilot_artifact,
    write_joint_synthesis_pilot_artifact,
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
DEFAULT_WP06B = Path(
    "artifacts/research_direction_sequence_policy/2026-09-22/"
    "wp06b_sequence_policy_proxy_bridge_v1.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/research_direction_joint_synthesis_pilot/2026-09-25/"
    "pa_h4_interval_union_joint_synthesis_v1.json"
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
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--wp06b", type=Path, default=DEFAULT_WP06B)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    wp06b = json.loads(args.wp06b.read_text(encoding="utf-8"))
    if wp06b["decision"]["selected_policy"] != "support_run_le_1":
        raise ValueError("P-A requires the fixed WP06-b current policy.")
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
    body = evaluate_joint_synthesis_pilot(
        hamiltonian, preparation, compiler
    )
    body["source_evidence"] = {
        "snapshot": {
            "path": str(args.snapshot),
            "sha256": file_sha256(args.snapshot),
        },
        "wp06b": {
            "path": str(args.wp06b),
            "sha256": file_sha256(args.wp06b),
            "content_fingerprint": wp06b["content_fingerprint"],
        },
    }
    source_paths = (
        Path("src/trotterlib/research_direction_joint_synthesis_pilot.py"),
        Path("scripts/run_research_direction_joint_synthesis_pilot.py"),
    )
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_worktree_status_before_generation": _git(["status", "--short"]),
        "evidence_status": "local_dirty_worktree_not_externally_reproduced",
        "command": shlex.join([sys.executable, *sys.argv]),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "qiskit_version": qiskit.__version__,
        "source_sha256": {
            str(path): file_sha256(path) for path in source_paths
        },
    }
    artifact = finalize_joint_synthesis_pilot_artifact(
        body, provenance=provenance
    )
    write_joint_synthesis_pilot_artifact(artifact, args.output)
    print(args.output)
    print(artifact["content_fingerprint"])
    print(artifact["decision"]["status"])


if __name__ == "__main__":
    main()
