#!/usr/bin/env python3
"""Validate a fixed connected-cluster calibration on unused full circuits."""

from __future__ import annotations

import argparse
import hashlib
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import qiskit

from trotterlib.rte import CompilerSettings
from trotterlib.rte_connected_cluster_cost_validation import (
    load_connected_cluster_calibration,
    load_connected_cluster_hamiltonian_snapshot,
    validate_connected_cluster_calibration_holdout,
    write_connected_cluster_transfer_validation,
)


def _git_head() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _git_status() -> list[str]:
    completed = subprocess.run(
        ["git", "status", "--short"],
        check=False,
        capture_output=True,
        text=True,
    )
    return (
        completed.stdout.splitlines()
        if completed.returncode == 0
        else ["git_status_unavailable"]
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("calibration", type=Path)
    parser.add_argument("hamiltonian_snapshot", type=Path)
    parser.add_argument("--holdout-lengths", default="4,6,8")
    parser.add_argument("--holdout-zero-samples", type=int, default=200)
    parser.add_argument("--holdout-single-order2-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--maximum-workers", type=int, default=3)
    parser.add_argument(
        "--sample-chunk-size",
        type=int,
        default=128,
        help="Trajectories per resumable worker chunk; use 0 to disable.",
    )
    parser.add_argument("--checkpoint-directory", type=Path)
    parser.add_argument("--persistent-cache", type=Path)
    parser.add_argument("--no-chunk-patterns", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def _compiler(payload: dict) -> CompilerSettings:
    coupling = payload.get("coupling_map")
    return CompilerSettings(
        basis_gates=tuple(payload["basis_gates"]),
        backend_name=payload.get("backend_name"),
        coupling_map=(
            None if coupling is None else tuple(tuple(edge) for edge in coupling)
        ),
        optimization_level=int(payload["optimization_level"]),
        layout_method=payload.get("layout_method"),
        routing_method=payload.get("routing_method"),
        transpiler_seed=int(payload["transpiler_seed"]),
        qiskit_version=str(payload["qiskit_version"]),
    )


def main() -> int:
    args = _parser().parse_args()
    sample_chunk_size = None if args.sample_chunk_size == 0 else args.sample_chunk_size
    calibration = load_connected_cluster_calibration(args.calibration)
    hamiltonian = load_connected_cluster_hamiltonian_snapshot(
        args.hamiltonian_snapshot
    )
    lengths = tuple(
        int(value.strip())
        for value in str(args.holdout_lengths).split(",")
        if value.strip()
    )
    output = args.output or args.calibration.with_name(
        f"{args.calibration.stem}_transfer_seed{args.seed}_v2.json"
    )
    checkpoint_directory = args.checkpoint_directory or output.with_suffix(
        ".checkpoints"
    )
    source_paths = (
        Path("src/trotterlib/rte_connected_cluster_cost_validation.py"),
        Path("src/trotterlib/rte_compiled_cost.py"),
        Path("src/trotterlib/rte.py"),
        Path("src/trotterlib/df_rte_circuit.py"),
        Path("src/trotterlib/df_rte_qiskit.py"),
        Path("scripts/run_rte_connected_cluster_transfer_validation.py"),
    )
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_head(),
        "git_worktree_status_before_generation": _git_status(),
        "command": shlex.join(
            [
                ".venv311/bin/python",
                "scripts/run_rte_connected_cluster_transfer_validation.py",
                *sys.argv[1:],
            ]
        ),
        "python_version": platform.python_version(),
        "qiskit_version": qiskit.__version__,
        "platform": platform.platform(),
        "evidence_status": "local_worktree_validation_not_immutable_ci",
        "calibration_path": str(args.calibration),
        "calibration_sha256": _file_sha256(args.calibration),
        "hamiltonian_snapshot": str(args.hamiltonian_snapshot),
        "hamiltonian_snapshot_sha256": _file_sha256(
            args.hamiltonian_snapshot
        ),
        "checkpoint_directory": str(checkpoint_directory),
        "sample_chunk_size": sample_chunk_size,
        "source_sha256": {str(path): _file_sha256(path) for path in source_paths},
    }
    payload = validate_connected_cluster_calibration_holdout(
        calibration,
        hamiltonian,
        compiler=_compiler(calibration["configuration"]["compiler"]),
        holdout_lengths=lengths,
        holdout_zero_sample_count=args.holdout_zero_samples,
        holdout_single_rare_sample_count=args.holdout_single_order2_samples,
        seed=args.seed,
        maximum_workers=args.maximum_workers,
        checkpoint_directory=checkpoint_directory,
        persistent_cache_path=args.persistent_cache,
        chunk_patterns=not args.no_chunk_patterns,
        sample_chunk_size=sample_chunk_size,
        provenance=provenance,
    )
    write_connected_cluster_transfer_validation(payload, output)
    print(output)
    print(payload["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
