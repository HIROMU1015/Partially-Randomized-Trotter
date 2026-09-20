#!/usr/bin/env python3
"""Run the H4 q=1,2,4,8 physical-signal and RPE branch validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import qiskit

from trotterlib.df_hamiltonian import PhysicalSector
from trotterlib.rpe_four_round_phase_validation import (
    validate_rpe_four_round_phase_reconstruction,
    write_rpe_four_round_phase_validation,
)
from trotterlib.rte_connected_cluster_cost_validation import (
    load_connected_cluster_hamiltonian_snapshot,
)


DEFAULT_SNAPSHOT = Path(
    "artifacts/rte_connected_cluster_cost_validation/"
    "h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k2_connected_"
    "pilot30_max1500_hold1500_rare375_v1.hamiltonian.npz"
)
DEFAULT_ACCOUNTING = Path(
    "artifacts/rpe_four_round_accounting_validation/2026-09-18/"
    "h4_sto3g_d100_rank12_ld3_dt0p1_r4_k2_q1_q2_q4_q8_limited_v1.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/rpe_four_round_phase_validation/2026-09-20/"
    "h4_sto3g_d100_rank12_ld3_dt0p1_r4_k2_q1_q2_q4_q8_"
    "physical_branch_v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(command: list[str]) -> str | list[str] | None:
    result = subprocess.run(
        ["git", *command], check=False, capture_output=True, text=True
    )
    if result.returncode != 0:
        return None
    lines = result.stdout.splitlines()
    return lines[0] if len(lines) == 1 else lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--accounting", type=Path, default=DEFAULT_ACCOUNTING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--marginal-repetitions", type=int, default=100_000)
    parser.add_argument("--marginal-seed", type=int, default=20260920)
    parser.add_argument("--explicit-trajectory-seed", type=int, default=20260921)
    args = parser.parse_args()

    hamiltonian = load_connected_cluster_hamiltonian_snapshot(args.snapshot)
    accounting = json.loads(args.accounting.read_text(encoding="utf-8"))
    sources = (
        Path("src/trotterlib/rpe_four_round_phase_validation.py"),
        Path("scripts/run_rpe_four_round_phase_validation.py"),
    )
    payload = validate_rpe_four_round_phase_reconstruction(
        hamiltonian,
        PhysicalSector.number_sector(
            n_qubits=hamiltonian.n_qubits, n_electrons=4
        ),
        accounting,
        marginal_repetitions=args.marginal_repetitions,
        marginal_seed=args.marginal_seed,
        explicit_trajectory_seed=args.explicit_trajectory_seed,
        provenance={
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git(["rev-parse", "HEAD"]),
            "git_worktree_status_before_generation": _git(["status", "--short"]),
            "evidence_status": "local_worktree_validation_not_immutable_ci",
            "command": shlex.join(
                [
                    ".venv311/bin/python",
                    "scripts/run_rpe_four_round_phase_validation.py",
                    *sys.argv[1:],
                ]
            ),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "qiskit_version": qiskit.__version__,
            "inputs": {
                "snapshot": {
                    "path": str(args.snapshot),
                    "sha256": _sha256(args.snapshot),
                },
                "accounting": {
                    "path": str(args.accounting),
                    "sha256": _sha256(args.accounting),
                },
            },
            "source_sha256": {str(path): _sha256(path) for path in sources},
        },
    )
    write_rpe_four_round_phase_validation(payload, args.output)
    marginal = payload["marginal_branch_monte_carlo"]
    q8 = payload["physical_signals_and_exact_probabilities"]["rounds"][-1]
    print(
        f"wrote {args.output}; overall_pass={payload['summary']['overall_pass']}; "
        f"q8_radius={q8['observed_signal_radius']:.12g}; "
        f"branch_failure={marginal['branch_failure_rate']:.6g}; "
        f"final_failure={marginal['final_failure_rate']:.6g}; "
        f"elapsed={payload['performance']['elapsed_seconds']:.3f}s"
    )
    return 0 if payload["summary"]["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
