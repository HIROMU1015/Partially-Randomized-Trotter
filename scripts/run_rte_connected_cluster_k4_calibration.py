#!/usr/bin/env python3
"""Calibrate independent paired K4 coefficients against fixed transfer holdouts."""

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

import qiskit

from trotterlib.rte import CompilerSettings
from trotterlib.rte_connected_cluster_cost_validation import (
    calibrate_and_validate_connected_cluster_k4,
    load_connected_cluster_hamiltonian_snapshot,
    validate_connected_cluster_transfer_payload,
    write_connected_cluster_k4_calibration,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _git_status() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short"], capture_output=True, text=True, check=False
    )
    return result.stdout.splitlines() if result.returncode == 0 else []


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("transfer", type=Path)
    parser.add_argument("hamiltonian_snapshot", type=Path)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--maximum-workers", type=int, default=3)
    parser.add_argument("--sample-chunk-size", type=int, default=15)
    parser.add_argument("--checkpoint-directory", type=Path)
    parser.add_argument("--persistent-cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    transfer = json.loads(args.transfer.read_text(encoding="utf-8"))
    validate_connected_cluster_transfer_payload(transfer)
    hamiltonian = load_connected_cluster_hamiltonian_snapshot(
        args.hamiltonian_snapshot
    )
    raw = transfer["calibration_reference"]["condition"]["compiler"]
    compiler = CompilerSettings(
        basis_gates=tuple(raw["basis_gates"]),
        backend_name=raw["backend_name"],
        coupling_map=(
            None
            if raw["coupling_map"] is None
            else tuple(tuple(edge) for edge in raw["coupling_map"])
        ),
        optimization_level=int(raw["optimization_level"]),
        layout_method=raw["layout_method"],
        routing_method=raw["routing_method"],
        transpiler_seed=int(raw["transpiler_seed"]),
        qiskit_version=raw["qiskit_version"],
    )
    sources = (
        Path("src/trotterlib/rte_connected_cluster_cost_validation.py"),
        Path("scripts/run_rte_connected_cluster_k4_calibration.py"),
    )
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_head(),
        "git_worktree_status_before_generation": _git_status(),
        "evidence_status": "local_worktree_validation_not_immutable_ci",
        "command": shlex.join(
            [
                ".venv311/bin/python",
                "scripts/run_rte_connected_cluster_k4_calibration.py",
                *sys.argv[1:],
            ]
        ),
        "python_version": platform.python_version(),
        "qiskit_version": qiskit.__version__,
        "platform": platform.platform(),
        "source_sha256": {str(path): _sha256(path) for path in sources},
        "source_transfer": str(args.transfer),
        "source_transfer_sha256": _sha256(args.transfer),
        "hamiltonian_snapshot": str(args.hamiltonian_snapshot),
        "hamiltonian_snapshot_sha256": _sha256(args.hamiltonian_snapshot),
    }
    payload = calibrate_and_validate_connected_cluster_k4(
        transfer,
        hamiltonian,
        compiler=compiler,
        sample_count_per_pattern=args.samples,
        seed=args.seed,
        maximum_workers=args.maximum_workers,
        persistent_cache_path=args.persistent_cache,
        checkpoint_directory=args.checkpoint_directory,
        sample_chunk_size=(
            None if args.sample_chunk_size == 0 else args.sample_chunk_size
        ),
        provenance=provenance,
    )
    write_connected_cluster_k4_calibration(payload, args.output)
    print(args.output, flush=True)
    print(payload["summary"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
