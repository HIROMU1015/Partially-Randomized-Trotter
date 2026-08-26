#!/usr/bin/env python3
"""Add an independent precision holdout to a connected-cluster validation."""

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

from trotterlib.df_hamiltonian import build_df_h_d_from_molecule
from trotterlib.df_partial_randomized_pf import df_hamiltonian_hash
from trotterlib.rte import CompilerSettings
from trotterlib.rte_connected_cluster_cost_validation import (
    load_connected_cluster_hamiltonian_snapshot,
    supplement_connected_cluster_holdout_precision,
    write_connected_cluster_supplement_validation,
)


DEFAULT_SOURCE = Path(
    "artifacts/rte_connected_cluster_cost_validation/"
    "h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k2_connected_"
    "pilot30_max1500_hold1000_rare250_v1.json"
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
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--hamiltonian-snapshot", type=Path)
    parser.add_argument("--molecule", type=int, default=4)
    parser.add_argument("--distance", type=float, default=1.0)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--df-rank", type=int, default=12)
    parser.add_argument("--additional-zero-samples", type=int, default=500)
    parser.add_argument("--additional-single-order2-samples", type=int, default=125)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--maximum-workers", type=int, default=3)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("artifacts/rte_connected_cluster_cost_validation"),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    source_payload = json.loads(args.source.read_text(encoding="utf-8"))
    source_configuration = source_payload["configuration"]
    compiler_data = source_configuration["compiler"]
    compiler = CompilerSettings(
        basis_gates=tuple(compiler_data["basis_gates"]),
        backend_name=compiler_data["backend_name"],
        coupling_map=(
            None
            if compiler_data["coupling_map"] is None
            else tuple(tuple(edge) for edge in compiler_data["coupling_map"])
        ),
        optimization_level=int(compiler_data["optimization_level"]),
        layout_method=compiler_data["layout_method"],
        routing_method=compiler_data["routing_method"],
        transpiler_seed=int(compiler_data["transpiler_seed"]),
        qiskit_version=compiler_data["qiskit_version"],
    )
    if compiler.qiskit_version != qiskit.__version__:
        raise RuntimeError("Installed Qiskit version differs from the source artifact.")
    snapshot_value = args.hamiltonian_snapshot or source_payload.get(
        "provenance", {}
    ).get("hamiltonian_snapshot")
    snapshot_path = None if snapshot_value is None else Path(snapshot_value)
    if snapshot_path is not None and snapshot_path.exists():
        expected_sha256 = source_payload.get("provenance", {}).get(
            "hamiltonian_snapshot_sha256"
        )
        if expected_sha256 is not None and _file_sha256(snapshot_path) != expected_sha256:
            raise RuntimeError("Hamiltonian snapshot SHA-256 differs from the source.")
        hamiltonian = load_connected_cluster_hamiltonian_snapshot(snapshot_path)
        hamiltonian_source = "exact_source_snapshot"
    else:
        hamiltonian, _sector = build_df_h_d_from_molecule(
            args.molecule,
            distance=args.distance,
            basis=args.basis,
            df_rank=args.df_rank,
        )
        hamiltonian_source = "numerically_checked_molecular_rebuild"
    source_paths = (
        Path("src/trotterlib/rte_connected_cluster_cost_validation.py"),
        Path("src/trotterlib/rte_compiled_cost.py"),
        Path("src/trotterlib/rte.py"),
        Path("src/trotterlib/df_rte_circuit.py"),
        Path("src/trotterlib/df_rte_qiskit.py"),
        Path("scripts/run_rte_connected_cluster_holdout_supplement.py"),
    )
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_head(),
        "git_worktree_status_before_generation": _git_status(),
        "evidence_status": "local_worktree_validation_not_immutable_ci",
        "command": shlex.join(
            [
                ".venv311/bin/python",
                "scripts/run_rte_connected_cluster_holdout_supplement.py",
                *sys.argv[1:],
            ]
        ),
        "python_version": platform.python_version(),
        "qiskit_version": qiskit.__version__,
        "platform": platform.platform(),
        "parent_hamiltonian_hash": df_hamiltonian_hash(hamiltonian),
        "source_artifact": str(args.source),
        "source_artifact_sha256": _file_sha256(args.source),
        "hamiltonian_source": hamiltonian_source,
        "hamiltonian_snapshot": (
            None if snapshot_path is None else str(snapshot_path)
        ),
        "source_sha256": {str(path): _file_sha256(path) for path in source_paths},
    }
    payload = supplement_connected_cluster_holdout_precision(
        source_payload,
        hamiltonian,
        compiler=compiler,
        additional_zero_sample_count=args.additional_zero_samples,
        additional_single_rare_sample_count=(
            args.additional_single_order2_samples
        ),
        seed=args.seed,
        maximum_workers=args.maximum_workers,
        provenance=provenance,
    )
    output = args.output_directory / (
        f"{args.source.stem}_supplement_hold{args.additional_zero_samples}_"
        f"rare{args.additional_single_order2_samples}_seed{args.seed}_v1.json"
    )
    write_connected_cluster_supplement_validation(payload, output)
    print(output)
    print(payload["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
