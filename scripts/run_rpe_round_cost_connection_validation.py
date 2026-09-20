#!/usr/bin/env python3
"""Run the representative H4 short-round cost-connection validation."""

from __future__ import annotations

import argparse
import hashlib
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import qiskit

from trotterlib.df_hamiltonian import PhysicalSector
from trotterlib.rpe_round_cost_connection_validation import (
    validate_rpe_round_cost_connection,
    write_rpe_round_cost_connection_validation,
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
DEFAULT_OUTPUT = Path(
    "artifacts/rpe_round_cost_connection_validation/"
    "h4_sto3g_d100_rank12_ld3_dt0p1_r4_k2_q1_q2_q4_mc8_v1.json"
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(command: list[str]) -> str | list[str] | None:
    result = subprocess.run(
        ["git", *command],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    lines = result.stdout.splitlines()
    return lines[0] if len(lines) == 1 else lines


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--compiled-cost-seed", type=int, default=20260901)
    return parser


def main() -> int:
    args = _parser().parse_args()
    hamiltonian = load_connected_cluster_hamiltonian_snapshot(args.snapshot)
    sector = PhysicalSector.number_sector(
        n_qubits=hamiltonian.n_qubits,
        n_electrons=4,
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
    payload = validate_rpe_round_cost_connection(
        hamiltonian,
        sector,
        compiler,
        ld=3,
        delta_time=0.1,
        q_values=(1, 2, 4),
        rte_steps_per_occurrence=4,
        finite_taylor_order=2,
        compiled_cost_sample_count=args.samples,
        compiled_cost_seed=args.compiled_cost_seed,
        provenance={
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git(["rev-parse", "HEAD"]),
            "git_worktree_status_before_generation": _git(["status", "--short"]),
            "evidence_status": "local_worktree_validation_not_immutable_ci",
            "command": shlex.join(
                [
                    ".venv311/bin/python",
                    "scripts/run_rpe_round_cost_connection_validation.py",
                    *sys.argv[1:],
                ]
            ),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "qiskit_version": qiskit.__version__,
            "snapshot": str(args.snapshot),
            "snapshot_sha256": _file_sha256(args.snapshot),
            "source_sha256": {
                "src/trotterlib/rpe_round_cost_connection_validation.py": (
                    _file_sha256(
                        Path(
                            "src/trotterlib/"
                            "rpe_round_cost_connection_validation.py"
                        )
                    )
                ),
                "scripts/run_rpe_round_cost_connection_validation.py": (
                    _file_sha256(
                        Path("scripts/run_rpe_round_cost_connection_validation.py")
                    )
                ),
            },
        },
    )
    write_rpe_round_cost_connection_validation(payload, args.output)
    print(
        f"wrote {args.output}; overall_pass={payload['summary']['overall_pass']}; "
        f"elapsed={payload['performance']['elapsed_seconds']:.3f}s"
    )
    return 0 if payload["summary"]["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
