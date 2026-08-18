#!/usr/bin/env python3
"""Generate representative H-chain finite-RTE signal validation artifacts.

When several ``L_D`` values are requested, the molecular integrals and DF
Hamiltonian are built once and reused.  Besides being faster, this keeps the
whole split comparison on one common numerical DF representation.
"""

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
import scipy

from trotterlib.df_hamiltonian import PhysicalSector, build_df_h_d_from_molecule
from trotterlib.finite_rte_signal_validation import (
    validate_finite_rte_signals,
    write_finite_rte_signal_validation,
)


def _int_tuple(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integers.") from exc
    if not result:
        raise argparse.ArgumentTypeError("At least one integer is required.")
    return result


def _float_tuple(value: str) -> tuple[float, ...]:
    try:
        result = tuple(
            float(item.strip()) for item in value.split(",") if item.strip()
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated numbers.") from exc
    if not result:
        raise argparse.ArgumentTypeError("At least one number is required.")
    return result


def _git_head() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _git_worktree_status() -> list[str]:
    completed = subprocess.run(
        ["git", "status", "--short"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return ["git_status_unavailable"]
    return [line for line in completed.stdout.splitlines() if line]


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--molecule", type=int, default=4)
    parser.add_argument("--distance", type=float, default=1.0)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--df-rank", type=int, default=12)
    parser.add_argument("--ld", type=int, default=3)
    parser.add_argument(
        "--ld-values",
        type=_int_tuple,
        help="Comma-separated L_D values; supersedes --ld and reuses one Hamiltonian.",
    )
    parser.add_argument("--delta-time", type=float)
    parser.add_argument(
        "--delta-time-values",
        type=_float_tuple,
        help=(
            "Comma-separated explicit delta values; supersedes --delta-time and "
            "reuses one Hamiltonian for the complete sensitivity grid."
        ),
    )
    parser.add_argument("--maximum-delta-time", type=float, default=0.1)
    parser.add_argument("--q-values", type=_int_tuple, default=(1, 2, 4))
    parser.add_argument("--r-values", type=_int_tuple, default=(1, 2, 4, 8))
    parser.add_argument("--k-values", type=_int_tuple, default=(0, 2, 4, 6))
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--coefficient-atol", type=float, default=1e-12)
    parser.add_argument(
        "--output",
        type=Path,
        help="Single-output path; only valid when one L_D value is requested.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("artifacts/finite_rte_signal_validation"),
    )
    return parser


def _default_output_path(
    args: argparse.Namespace,
    ld: int,
    delta_time: float | None,
    *,
    include_delta_label: bool,
) -> Path:
    basis_label = str(args.basis).lower().replace("-", "").replace(" ", "_")
    distance_label = int(round(100 * float(args.distance)))
    delta_label = ""
    if include_delta_label:
        delta_label = (
            "_dtauto"
            if delta_time is None
            else f"_dt{format(float(delta_time), '.8g').replace('.', 'p')}"
        )
    return args.output_directory / (
        f"h{args.molecule}_{basis_label}_d{distance_label}_"
        f"rank{args.df_rank}_ld{ld}{delta_label}_v1.json"
    )


def main() -> int:
    args = _parser().parse_args()
    ld_values = args.ld_values if args.ld_values is not None else (args.ld,)
    if args.delta_time_values is not None and args.delta_time is not None:
        raise ValueError("--delta-time and --delta-time-values are mutually exclusive.")
    delta_time_values = (
        args.delta_time_values
        if args.delta_time_values is not None
        else (args.delta_time,)
    )
    if len(set(ld_values)) != len(ld_values):
        raise ValueError("L_D values must not contain duplicates.")
    if len(set(delta_time_values)) != len(delta_time_values):
        raise ValueError("delta-time values must not contain duplicates.")
    if any(ld < 0 for ld in ld_values):
        raise ValueError("L_D values must be non-negative.")
    if any(
        value is not None and (not np.isfinite(value) or value <= 0.0)
        for value in delta_time_values
    ):
        raise ValueError("delta-time values must be finite and positive.")
    combination_count = len(ld_values) * len(delta_time_values)
    if args.output is not None and combination_count != 1:
        raise ValueError("--output can only be used with one (L_D, delta) pair.")

    hamiltonian, molecule_sector = build_df_h_d_from_molecule(
        args.molecule,
        distance=args.distance,
        basis=args.basis,
        df_rank=args.df_rank,
    )
    if molecule_sector.n_electrons is None:
        raise RuntimeError("The molecular sector is missing its electron count.")
    validation_sector = PhysicalSector.number_sector(
        n_qubits=hamiltonian.n_qubits,
        n_electrons=molecule_sector.n_electrons,
    )
    reproducible_command = shlex.join(
        [
            ".venv311/bin/python",
            "scripts/run_finite_rte_signal_validation.py",
            *sys.argv[1:],
        ]
    )
    common_provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_head(),
        "git_worktree_status_before_generation": _git_worktree_status(),
        "evidence_status": "local_worktree_validation_not_immutable_ci",
        "command": reproducible_command,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "qiskit_version": qiskit.__version__,
        "platform": platform.platform(),
        "shared_hamiltonian_generation": True,
        "batch_ld_values": list(ld_values),
        "batch_delta_time_values": list(delta_time_values),
        "source_sha256": {
            "src/trotterlib/finite_rte_signal_validation.py": _file_sha256(
                Path("src/trotterlib/finite_rte_signal_validation.py")
            ),
            "scripts/run_finite_rte_signal_validation.py": _file_sha256(
                Path("scripts/run_finite_rte_signal_validation.py")
            ),
        },
    }
    results = []
    hamiltonian_hash: str | None = None
    all_pass = True
    for ld in ld_values:
        for delta_time in delta_time_values:
            output = args.output or _default_output_path(
                args,
                ld,
                delta_time,
                include_delta_label=combination_count > 1,
            )
            provenance = dict(common_provenance)
            provenance["batch_position"] = len(results)
            payload = validate_finite_rte_signals(
                hamiltonian,
                validation_sector,
                ld=ld,
                delta_time=delta_time,
                maximum_delta_time=args.maximum_delta_time,
                q_values=args.q_values,
                rte_step_values=args.r_values,
                finite_taylor_orders=args.k_values,
                seed=args.seed,
                coefficient_atol=args.coefficient_atol,
                provenance=provenance,
            )
            current_hash = payload["hamiltonian"]["hamiltonian_hash"]
            if hamiltonian_hash is None:
                hamiltonian_hash = current_hash
            elif current_hash != hamiltonian_hash:
                raise RuntimeError("A reused Hamiltonian produced inconsistent hashes.")
            write_finite_rte_signal_validation(payload, output)
            all_pass = all_pass and payload["summary"]["overall_pass"]
            results.append(
                {
                    "ld": ld,
                    "delta_time": payload["request"]["delta_time"],
                    "output": str(output),
                    "validation_fingerprint": payload["validation_fingerprint"],
                    "hamiltonian_hash": current_hash,
                    "summary": payload["summary"],
                    "performance": payload["performance"],
                }
            )
    print(json.dumps({"results": results}, indent=2, sort_keys=True))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
