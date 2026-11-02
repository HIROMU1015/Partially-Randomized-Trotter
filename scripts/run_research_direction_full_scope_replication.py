#!/usr/bin/env python3
"""Run the focused WP05-bR r=32 replication in a detached session."""

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
from trotterlib.research_direction_full_scope_extension import (
    validate_wp05b_artifact,
)
from trotterlib.research_direction_full_scope_replication import (
    evaluate_wp05br_replication,
    finalize_wp05br_artifact,
    write_wp05br_artifact,
)
from trotterlib.research_direction_prevalidation import file_sha256
from trotterlib.research_direction_sequence_policy import validate_wp06b_artifact
from trotterlib.rte import CompilerSettings
from trotterlib.rte_connected_cluster_cost_validation import (
    load_connected_cluster_hamiltonian_snapshot,
)


DEFAULT_SNAPSHOT = Path(
    "artifacts/rte_connected_cluster_cost_validation/"
    "h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k2_connected_"
    "pilot30_max1500_hold1500_rare375_v1.hamiltonian.npz"
)
DEFAULT_WP05B = Path(
    "artifacts/research_direction_full_scope_extension/2026-09-22/"
    "wp05b_q8_delta_0p01_full_scope_extension_v1.json"
)
DEFAULT_WP06B = Path(
    "artifacts/research_direction_sequence_policy/2026-09-22/"
    "wp06b_sequence_policy_proxy_bridge_v1.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/research_direction_full_scope_replication/2026-09-22/"
    "wp05br_r32_32trajectory_replication_v1.json"
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
        Path("src/trotterlib/research_direction_full_scope_extension.py"),
        Path("src/trotterlib/research_direction_full_scope_replication.py"),
        Path("scripts/run_research_direction_full_scope_replication.py"),
    )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_worktree_status_before_generation": _git(["status", "--short"]),
        "evidence_status": "local_worktree_validation_not_immutable_ci",
        "command": shlex.join(
            [
                ".venv311/bin/python",
                "scripts/run_research_direction_full_scope_replication.py",
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
    parser.add_argument("--wp05b", type=Path, default=DEFAULT_WP05B)
    parser.add_argument("--wp06b", type=Path, default=DEFAULT_WP06B)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    wp05b = json.loads(args.wp05b.read_text(encoding="utf-8"))
    validate_wp05b_artifact(wp05b)
    wp06b = json.loads(args.wp06b.read_text(encoding="utf-8"))
    validate_wp06b_artifact(wp06b)
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
    body = evaluate_wp05br_replication(
        hamiltonian,
        preparation,
        compiler,
        wp05b,
        wp06b,
        progress=lambda message: print(message, flush=True),
    )
    body["source_evidence"] = {
        "snapshot": {
            "path": str(args.snapshot),
            "sha256": file_sha256(args.snapshot),
        },
        "wp05b": {
            "path": str(args.wp05b),
            "sha256": file_sha256(args.wp05b),
            "content_fingerprint": wp05b["content_fingerprint"],
        },
        "wp06b": {
            "path": str(args.wp06b),
            "sha256": file_sha256(args.wp06b),
            "content_fingerprint": wp06b["content_fingerprint"],
        },
    }
    artifact = finalize_wp05br_artifact(body, provenance=_provenance())
    write_wp05br_artifact(artifact, args.output)
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
