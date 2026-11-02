#!/usr/bin/env python3
"""Run raw opt-level-2 computation for the focused M06/L08 validation."""

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
from trotterlib.research_direction_compiler_transfer_compute import (
    compute_compiler_transfer_raw,
    finalize_compiler_transfer_compute_artifact,
    write_compiler_transfer_compute_artifact,
)
from trotterlib.research_direction_prevalidation import file_sha256
from trotterlib.rte import CompilerSettings
from trotterlib.rte_connected_cluster_cost_validation import (
    load_connected_cluster_hamiltonian_snapshot,
)


DEFAULT_SNAPSHOT = Path(
    "artifacts/rte_connected_cluster_cost_validation/"
    "h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k2_connected_"
    "pilot30_max1500_hold1500_rare375_v1.hamiltonian.npz"
)
DEFAULT_WP05BR = Path(
    "artifacts/research_direction_full_scope_replication/2026-09-22/"
    "wp05br_r32_32trajectory_replication_v1.json"
)
DEFAULT_WP06B = Path(
    "artifacts/research_direction_sequence_policy/2026-09-22/"
    "wp06b_sequence_policy_proxy_bridge_v1.json"
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
    "artifacts/research_direction_compiler_transfer/2026-09-23/"
    "m06_l08_opt2_same_trajectory_compute_v1.json"
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
    parser.add_argument("--wp05br", type=Path, default=DEFAULT_WP05BR)
    parser.add_argument("--wp06b", type=Path, default=DEFAULT_WP06B)
    parser.add_argument("--g08", type=Path, default=DEFAULT_G08)
    parser.add_argument("--m08", type=Path, default=DEFAULT_M08)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    inputs = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in (
            ("wp05br", args.wp05br),
            ("wp06b", args.wp06b),
            ("g08", args.g08),
            ("m08", args.m08),
        )
    }
    hamiltonian = load_connected_cluster_hamiltonian_snapshot(args.snapshot)
    ld3 = prepare_df_partial_s2(
        hamiltonian,
        split_df_hamiltonian_by_ld(hamiltonian, 3),
        identity_policy="extract_identity_phase",
    )
    ld12 = prepare_df_partial_s2(
        hamiltonian,
        split_df_hamiltonian_by_ld(hamiltonian, 12),
        identity_policy="extract_identity_phase",
    )
    compiler = CompilerSettings(
        basis_gates=("rz", "sx", "x", "cx"),
        backend_name=None,
        coupling_map=None,
        optimization_level=2,
        layout_method=None,
        routing_method=None,
        transpiler_seed=17,
        qiskit_version=qiskit.__version__,
    )
    body = compute_compiler_transfer_raw(
        hamiltonian,
        ld3,
        ld12,
        compiler,
        inputs["wp05br"],
        inputs["wp06b"],
        inputs["g08"],
        inputs["m08"],
        progress=lambda message: print(message, flush=True),
    )
    sources = (
        Path("src/trotterlib/research_direction_compiler_transfer_compute.py"),
        Path("scripts/run_research_direction_compiler_transfer_compute.py"),
    )
    input_paths = {
        "snapshot": args.snapshot,
        "wp05br": args.wp05br,
        "wp06b": args.wp06b,
        "g08": args.g08,
        "m08": args.m08,
    }
    body["source_evidence"] = {
        name: {
            "path": str(path),
            "sha256": file_sha256(path),
            **(
                {"content_fingerprint": inputs[name]["content_fingerprint"]}
                if name in inputs
                else {}
            ),
        }
        for name, path in input_paths.items()
    }
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_worktree_status_before_generation": _git(["status", "--short"]),
        "evidence_status": "local_worktree_compute_not_yet_interpreted",
        "command": shlex.join(
            [
                ".venv311/bin/python",
                "scripts/run_research_direction_compiler_transfer_compute.py",
                *sys.argv[1:],
            ]
        ),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "qiskit_version": qiskit.__version__,
        "source_sha256": {str(path): file_sha256(path) for path in sources},
    }
    artifact = finalize_compiler_transfer_compute_artifact(
        body, provenance=provenance
    )
    write_compiler_transfer_compute_artifact(artifact, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "content_fingerprint": artifact["content_fingerprint"],
                "status": "raw_compute_complete_analysis_pending",
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
