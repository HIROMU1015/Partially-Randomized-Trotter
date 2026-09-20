#!/usr/bin/env python3
"""Run the target-precision RPE round-horizon validation."""

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

from trotterlib.config import CA, TARGET_ERROR
from trotterlib.df_hamiltonian import PhysicalSector
from trotterlib.rpe_target_round_horizon_validation import (
    validate_rpe_target_round_horizon,
    write_rpe_target_round_horizon_validation,
)
from trotterlib.rte_connected_cluster_cost_validation import (
    load_connected_cluster_hamiltonian_snapshot,
)


DEFAULT_SNAPSHOT = Path(
    "artifacts/rte_connected_cluster_cost_validation/"
    "h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k2_connected_"
    "pilot30_max1500_hold1500_rare375_v1.hamiltonian.npz"
)
DEFAULT_FOUR_ROUND = Path(
    "artifacts/rpe_four_round_phase_validation/2026-09-20/"
    "h4_sto3g_d100_rank12_ld3_dt0p1_r4_k2_q1_q2_q4_q8_"
    "physical_branch_v1.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/rpe_target_round_horizon_validation/2026-09-20/"
    "h4_sto3g_d100_rank12_ld3_dt0p1_r4_k2_ca_over_10_horizon_v1.json"
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
    parser.add_argument("--four-round", type=Path, default=DEFAULT_FOUR_ROUND)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--target-energy-precision", type=float, default=TARGET_ERROR
    )
    parser.add_argument("--chemical-accuracy", type=float, default=CA)
    args = parser.parse_args()

    hamiltonian = load_connected_cluster_hamiltonian_snapshot(args.snapshot)
    four_round = json.loads(args.four_round.read_text(encoding="utf-8"))
    sources = (
        Path("src/trotterlib/rpe_target_round_horizon_validation.py"),
        Path("scripts/run_rpe_target_round_horizon_validation.py"),
    )
    payload = validate_rpe_target_round_horizon(
        hamiltonian,
        PhysicalSector.number_sector(
            n_qubits=hamiltonian.n_qubits, n_electrons=4
        ),
        four_round,
        target_energy_precision=args.target_energy_precision,
        chemical_accuracy=args.chemical_accuracy,
        provenance={
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git(["rev-parse", "HEAD"]),
            "git_worktree_status_before_generation": _git(["status", "--short"]),
            "evidence_status": "local_worktree_validation_not_immutable_ci",
            "command": shlex.join(
                [
                    ".venv311/bin/python",
                    "scripts/run_rpe_target_round_horizon_validation.py",
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
                "four_round": {
                    "path": str(args.four_round),
                    "sha256": _sha256(args.four_round),
                },
            },
            "source_sha256": {str(path): _sha256(path) for path in sources},
        },
    )
    write_rpe_target_round_horizon_validation(payload, args.output)
    summary = payload["summary"]
    target = payload["fixed_schedule_matrix_diagnostics"][-1]
    print(
        f"wrote {args.output}; overall_pass={summary['overall_pass']}; "
        f"M={summary['required_maximum_round_index_M']}; "
        f"q_max={summary['required_q_max']}; "
        f"fixed_schedule_accepted={summary['fixed_schedule_target_q_accepted']}; "
        f"target_attenuation={target['attenuation']:.12g}; "
        f"target_pf_phase_error={target['pf_phase_error']:.12g}; "
        f"elapsed={payload['performance']['elapsed_seconds']:.3f}s"
    )
    return 0 if summary["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
