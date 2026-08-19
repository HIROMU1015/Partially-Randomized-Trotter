#!/usr/bin/env python3
"""Generate the representative H4 Product Formula delta-grid validation."""

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
from trotterlib.pf_delta_validation import (
    validate_pf_delta_grid,
    write_pf_delta_validation,
)


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


def _int_tuple(value: str) -> tuple[int, ...]:
    try:
        result = tuple(
            int(item.strip()) for item in value.split(",") if item.strip()
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integers.") from exc
    if not result:
        raise argparse.ArgumentTypeError("At least one integer is required.")
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
    return completed.stdout.splitlines()


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
        help=(
            "Comma-separated L_D values; supersedes --ld and reuses one "
            "Hamiltonian for all direct-tail validations."
        ),
    )
    parser.add_argument(
        "--calibration-times",
        type=_float_tuple,
        default=(0.01, 0.02, 0.04, 0.08),
    )
    parser.add_argument(
        "--validation-delta-times",
        type=_float_tuple,
        default=(0.0125, 0.025, 0.05, 0.1, 0.2, 0.4),
    )
    parser.add_argument("--q-values", type=_int_tuple, default=(1, 2, 4))
    parser.add_argument("--beta-pf-budget", type=float, default=0.08)
    parser.add_argument("--relative-tolerance", type=float, default=0.25)
    parser.add_argument("--slope-min", type=float, default=1.5)
    parser.add_argument("--slope-max", type=float, default=2.5)
    parser.add_argument("--coefficient-atol", type=float, default=1e-12)
    parser.add_argument("--paper-d6-minimum-sine-abs", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument(
        "--output",
        type=Path,
        help="Single-output path; only valid when one L_D value is requested.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("artifacts/pf_delta_validation"),
    )
    return parser


def _default_output_path(args: argparse.Namespace, ld: int) -> Path:
    basis_label = str(args.basis).lower().replace("-", "").replace(" ", "_")
    distance_label = int(round(100 * float(args.distance)))
    return args.output_directory / (
        f"h{args.molecule}_{basis_label}_d{distance_label}_"
        f"rank{args.df_rank}_ld{ld}_v5.json"
    )


def main() -> int:
    args = _parser().parse_args()
    ld_values = args.ld_values if args.ld_values is not None else (args.ld,)
    if len(set(ld_values)) != len(ld_values):
        raise ValueError("L_D values must not contain duplicates.")
    if any(ld < 0 for ld in ld_values):
        raise ValueError("L_D values must be non-negative.")
    if args.output is not None and len(ld_values) != 1:
        raise ValueError("--output can only be used with one L_D value.")
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
    command = shlex.join(
        [
            ".venv311/bin/python",
            "scripts/run_pf_delta_validation.py",
            *sys.argv[1:],
        ]
    )
    common_provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_head(),
        "git_worktree_status_before_generation": _git_worktree_status(),
        "evidence_status": "local_worktree_validation_not_immutable_ci",
        "command": command,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "qiskit_version": qiskit.__version__,
        "platform": platform.platform(),
        "shared_hamiltonian_generation": True,
        "batch_ld_values": list(ld_values),
        "source_sha256": {
            "src/trotterlib/pf_delta_validation.py": _file_sha256(
                Path("src/trotterlib/pf_delta_validation.py")
            ),
            "src/trotterlib/finite_rte_signal_validation.py": _file_sha256(
                Path("src/trotterlib/finite_rte_signal_validation.py")
            ),
            "scripts/run_pf_delta_validation.py": _file_sha256(
                Path("scripts/run_pf_delta_validation.py")
            ),
        },
    }
    results = []
    hamiltonian_hash: str | None = None
    all_pass = True
    for ld in ld_values:
        output = args.output or _default_output_path(args, ld)
        provenance = dict(common_provenance)
        provenance["batch_position"] = len(results)
        payload = validate_pf_delta_grid(
            hamiltonian,
            validation_sector,
            ld=ld,
            surrogate_calibration_times=args.calibration_times,
            validation_delta_times=args.validation_delta_times,
            q_values=args.q_values,
            beta_pf_budget=args.beta_pf_budget,
            surrogate_relative_tolerance=args.relative_tolerance,
            scaling_slope_interval=(args.slope_min, args.slope_max),
            coefficient_atol=args.coefficient_atol,
            paper_d6_minimum_sine_abs=args.paper_d6_minimum_sine_abs,
            seed=args.seed,
            provenance=provenance,
        )
        current_hash = payload["hamiltonian"]["hamiltonian_hash"]
        if hamiltonian_hash is None:
            hamiltonian_hash = current_hash
        elif current_hash != hamiltonian_hash:
            raise RuntimeError("A reused Hamiltonian produced inconsistent hashes.")
        write_pf_delta_validation(payload, output)
        all_pass = all_pass and payload["summary"]["overall_pass"]
        results.append(
            {
                "ld": ld,
                "output": str(output),
                "validation_fingerprint": payload["validation_fingerprint"],
                "hamiltonian_hash": current_hash,
                "summary": payload["summary"],
                "surrogate": payload["surrogate"],
                "performance": payload["performance"],
            }
        )
    print(json.dumps({"results": results}, indent=2, sort_keys=True))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
