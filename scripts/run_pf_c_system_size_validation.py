#!/usr/bin/env python3
"""Validate operational partial-S2 PF coefficients across H-chain sizes."""

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

from trotterlib.config import DEFAULT_BASIS, DEFAULT_DISTANCE
from trotterlib.df_hamiltonian import PhysicalSector, build_df_h_d_from_molecule
from trotterlib.pf_c_system_size_validation import (
    configured_qiskit_delta_times,
    make_system_size_payload,
    summarize_size_result,
    validate_state_action_coefficient,
    write_system_size_validation,
)
from trotterlib.pf_delta_validation import (
    validate_pf_delta_grid,
    write_pf_delta_validation,
)


def _int_tuple(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
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


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--molecule-sizes", type=_int_tuple, default=(2, 3, 4, 5))
    parser.add_argument("--distance", type=float, default=DEFAULT_DISTANCE)
    parser.add_argument("--basis", default=DEFAULT_BASIS)
    parser.add_argument("--paper-d6-minimum-sine-abs", type=float, default=0.1)
    parser.add_argument("--paper-d6-relative-tolerance", type=float, default=0.02)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("artifacts/pf_c_system_size_validation"),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if len(set(args.molecule_sizes)) != len(args.molecule_sizes):
        raise ValueError("molecule sizes must not contain duplicates.")
    if any(size < 2 for size in args.molecule_sizes):
        raise ValueError("molecule sizes must be at least 2.")
    command = shlex.join(
        [
            ".venv311/bin/python",
            "scripts/run_pf_c_system_size_validation.py",
            *sys.argv[1:],
        ]
    )
    source_paths = (
        Path("src/trotterlib/config.py"),
        Path("src/trotterlib/finite_rte_signal_validation.py"),
        Path("src/trotterlib/pf_c_system_size_validation.py"),
        Path("src/trotterlib/pf_delta_validation.py"),
        Path("scripts/run_pf_c_system_size_validation.py"),
    )
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_head(),
        "evidence_status": "local_worktree_validation_not_immutable_ci",
        "command": command,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "qiskit_version": qiskit.__version__,
        "platform": platform.platform(),
        "source_sha256": {
            str(path): _file_sha256(path) for path in source_paths
        },
    }
    size_results = []
    systems = {}
    for size in args.molecule_sizes:
        hamiltonian, molecule_sector = build_df_h_d_from_molecule(
            size,
            distance=args.distance,
            basis=args.basis,
        )
        if molecule_sector.n_electrons is None:
            raise RuntimeError("The molecular sector is missing its electron count.")
        sector = PhysicalSector.number_sector(
            n_qubits=hamiltonian.n_qubits,
            n_electrons=molecule_sector.n_electrons,
        )
        ld = hamiltonian.n_blocks // 2
        systems[size] = (hamiltonian, sector, ld)
        delta_times = configured_qiskit_delta_times(size)
        core_path = args.output_directory / (
            f"h{size}_{str(args.basis).replace('-', '').lower()}_"
            f"d{int(round(100 * args.distance))}_rank{hamiltonian.n_blocks}_"
            f"ld{ld}_pf_d6_validation_v1.json"
        )
        core_payload = validate_pf_delta_grid(
            hamiltonian,
            sector,
            ld=ld,
            validation_delta_times=delta_times,
            q_values=(1,),
            maximum_dense_reference_qubits=max(8, hamiltonian.n_qubits),
            paper_d6_minimum_sine_abs=(
                args.paper_d6_minimum_sine_abs
            ),
            provenance={
                **provenance,
                "system_size_validation_molecule_type": size,
            },
        )
        write_pf_delta_validation(core_payload, core_path)
        result = summarize_size_result(
            core_payload,
            molecule_type=size,
            core_artifact_path=str(core_path),
            paper_d6_minimum_sine_abs=args.paper_d6_minimum_sine_abs,
            paper_d6_relative_tolerance=args.paper_d6_relative_tolerance,
        )
        size_results.append(result)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)

    state_action_results = []
    state_action_sizes = tuple(dict.fromkeys((*args.molecule_sizes, 6)))
    for size in state_action_sizes:
        if size in systems:
            hamiltonian, sector, ld = systems[size]
        else:
            hamiltonian, molecule_sector = build_df_h_d_from_molecule(
                size,
                distance=args.distance,
                basis=args.basis,
            )
            if molecule_sector.n_electrons is None:
                raise RuntimeError(
                    "The molecular sector is missing its electron count."
                )
            sector = PhysicalSector.number_sector(
                n_qubits=hamiltonian.n_qubits,
                n_electrons=molecule_sector.n_electrons,
            )
            ld = hamiltonian.n_blocks // 2
        exact_reference = None
        if size in args.molecule_sizes:
            core_path = args.output_directory / (
                f"h{size}_{str(args.basis).replace('-', '').lower()}_"
                f"d{int(round(100 * args.distance))}_rank{hamiltonian.n_blocks}_"
                f"ld{ld}_pf_d6_validation_v1.json"
            )
            exact_reference = json.loads(core_path.read_text(encoding="utf-8"))
        state_result = validate_state_action_coefficient(
            hamiltonian,
            sector,
            molecule_type=size,
            ld=ld,
            exact_reference_payload=exact_reference,
            paper_d6_relative_tolerance=args.paper_d6_relative_tolerance,
            paper_d6_minimum_sine_abs=(
                args.paper_d6_minimum_sine_abs
            ),
        )
        state_action_results.append(state_result)
        print(json.dumps(state_result, indent=2, sort_keys=True), flush=True)

    index_payload = make_system_size_payload(
        size_results,
        state_action_results=state_action_results,
        request={
            "molecule_sizes": list(args.molecule_sizes),
            "distance": float(args.distance),
            "basis": str(args.basis),
            "df_rank_policy": "project_config_selected_rank",
            "ld_policy": "floor(df_rank_actual/2)",
            "delta_policy": "configured_lower_order_qiskit_execution_window",
            "delta_step": 0.002,
            "q_values": [1],
            "paper_d6_minimum_sine_abs": float(
                args.paper_d6_minimum_sine_abs
            ),
            "paper_d6_relative_tolerance": float(
                args.paper_d6_relative_tolerance
            ),
            "operational_coefficient_policy": (
                "maximum of dominant-eigenphase and paper-Eq.-D6 "
                "point coefficients over the executable delta window; "
                "shift-invariant estimator retained as a diagnostic"
            ),
        },
        provenance=provenance,
    )
    index_path = args.output_directory / "h2_h6_paper_d6_c_v1.json"
    write_system_size_validation(index_payload, index_path)
    print(json.dumps(index_payload["summary"], indent=2, sort_keys=True))
    print(f"wrote {index_path}")
    validation_ok = bool(
        index_payload["summary"]["all_paper_d6_estimator_validation_pass"]
        and index_payload["summary"]["all_state_action_validation_pass"]
    )
    return 0 if validation_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
