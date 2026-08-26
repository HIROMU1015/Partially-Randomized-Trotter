#!/usr/bin/env python3
"""Run the operational K=2 connected-cluster cost validation."""

from __future__ import annotations

import argparse
import hashlib
import math
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
    calibrate_connected_cluster_cost_model,
    load_connected_cluster_hamiltonian_snapshot,
    validate_operational_connected_cluster_cost_model,
    write_connected_cluster_hamiltonian_snapshot,
    write_connected_cluster_calibration,
    write_connected_cluster_validation,
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
    parser.add_argument(
        "--mode", choices=("validation", "calibration"), default="validation"
    )
    parser.add_argument("--molecule", type=int, default=4)
    parser.add_argument("--distance", type=float, default=1.0)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--df-rank", type=int, default=12)
    parser.add_argument("--ld", type=int, default=3)
    parser.add_argument("--reference-delta-time", type=float, default=0.1)
    parser.add_argument("--reference-rte-steps", type=int, default=4)
    parser.add_argument("--pilot-samples", type=int, default=30)
    parser.add_argument("--minimum-production-samples", type=int, default=30)
    parser.add_argument("--maximum-production-samples", type=int, default=1500)
    parser.add_argument("--prediction-relative-se-target", type=float, default=0.01)
    parser.add_argument("--allocation-safety-factor", type=float, default=1.5)
    parser.add_argument(
        "--target-event-counts",
        default="4,6,8",
        help="Comma-separated event counts used to allocate calibration precision.",
    )
    parser.add_argument("--holdout-zero-samples", type=int, default=1000)
    parser.add_argument("--holdout-single-order2-samples", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--maximum-workers", type=int, default=3)
    parser.add_argument(
        "--sample-chunk-size",
        type=int,
        default=128,
        help="Trajectories per resumable worker chunk; use 0 to disable.",
    )
    parser.add_argument(
        "--adaptive-production-rounds",
        type=int,
        default=2,
        help="Maximum incremental reallocations after checking realized precision.",
    )
    parser.add_argument("--optimization-level", type=int, default=1)
    parser.add_argument("--transpiler-seed", type=int, default=17)
    parser.add_argument("--checkpoint-directory", type=Path)
    parser.add_argument("--persistent-cache", type=Path)
    parser.add_argument("--hamiltonian-snapshot", type=Path)
    parser.add_argument("--equal-cost-allocation", action="store_true")
    parser.add_argument("--no-chunk-patterns", action="store_true")
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("artifacts/rte_connected_cluster_cost_validation"),
    )
    return parser


def _slug(args: argparse.Namespace) -> str:
    basis = str(args.basis).lower().replace("-", "").replace(" ", "_")
    distance = int(round(100 * float(args.distance)))
    delta = format(float(args.reference_delta_time), ".8g").replace(".", "p")
    common = (
        f"h{args.molecule}_{basis}_d{distance}_rank{args.df_rank}_ld{args.ld}_"
        f"dt{delta}_ref{args.reference_rte_steps}_k2_connected_"
        f"pilot{args.pilot_samples}_max{args.maximum_production_samples}_"
    )
    if args.mode == "calibration":
        normalized_targets = "-".join(
            str(int(value.strip()))
            for value in str(args.target_event_counts).split(",")
            if value.strip()
        )
        target_suffix = (
            "" if normalized_targets == "4-6-8" else f"target{normalized_targets}_"
        )
        precision_suffix = ""
        if not math.isclose(args.prediction_relative_se_target, 0.01):
            precision = format(
                float(args.prediction_relative_se_target), ".8g"
            ).replace(".", "p")
            precision_suffix = f"rse{precision}_"
        return common + target_suffix + precision_suffix + "calibration_v2.json"
    return common + (
        f"hold{args.holdout_zero_samples}_"
        f"rare{args.holdout_single_order2_samples}_v1.json"
    )


def main() -> int:
    args = _parser().parse_args()
    sample_chunk_size = None if args.sample_chunk_size == 0 else args.sample_chunk_size
    output = args.output_directory / _slug(args)
    hamiltonian_snapshot = args.hamiltonian_snapshot or output.with_suffix(
        ".hamiltonian.npz"
    )
    checkpoint_directory = args.checkpoint_directory or (
        args.output_directory / f"{output.stem}_checkpoints"
    )
    source_paths = (
        Path("src/trotterlib/rte_connected_cluster_cost_validation.py"),
        Path("src/trotterlib/rte_order_stratified_cost_validation.py"),
        Path("src/trotterlib/rte_compiled_cost.py"),
        Path("src/trotterlib/rte.py"),
        Path("src/trotterlib/df_rte_circuit.py"),
        Path("src/trotterlib/df_rte_qiskit.py"),
        Path("scripts/run_rte_connected_cluster_cost_validation.py"),
    )
    if hamiltonian_snapshot.exists():
        hamiltonian = load_connected_cluster_hamiltonian_snapshot(
            hamiltonian_snapshot
        )
        hamiltonian_source = "existing_exact_snapshot"
    else:
        hamiltonian, _sector = build_df_h_d_from_molecule(
            args.molecule,
            distance=args.distance,
            basis=args.basis,
            df_rank=args.df_rank,
        )
        write_connected_cluster_hamiltonian_snapshot(
            hamiltonian, hamiltonian_snapshot
        )
        hamiltonian_source = "new_exact_snapshot"
    expected_metadata = {
        "molecule_type": args.molecule,
        "distance": args.distance,
        "basis": args.basis,
        "df_rank_requested": args.df_rank,
    }
    for key, expected in expected_metadata.items():
        actual = hamiltonian.metadata.get(key)
        if isinstance(expected, float):
            matches = abs(float(actual) - expected) <= 1e-12
        else:
            matches = actual == expected
        if not matches:
            raise RuntimeError(f"Hamiltonian snapshot differs at metadata {key}.")
    compiler = CompilerSettings(
        basis_gates=("rz", "sx", "x", "cx"),
        backend_name=None,
        coupling_map=None,
        optimization_level=args.optimization_level,
        layout_method=None,
        routing_method=None,
        transpiler_seed=args.transpiler_seed,
        qiskit_version=qiskit.__version__,
    )
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_head(),
        "git_worktree_status_before_generation": _git_status(),
        "evidence_status": "local_worktree_validation_not_immutable_ci",
        "command": shlex.join(
            [
                ".venv311/bin/python",
                "scripts/run_rte_connected_cluster_cost_validation.py",
                *sys.argv[1:],
            ]
        ),
        "python_version": platform.python_version(),
        "qiskit_version": qiskit.__version__,
        "platform": platform.platform(),
        "parent_hamiltonian_hash": df_hamiltonian_hash(hamiltonian),
        "hamiltonian_snapshot": str(hamiltonian_snapshot),
        "hamiltonian_snapshot_sha256": _file_sha256(hamiltonian_snapshot),
        "hamiltonian_source": hamiltonian_source,
        "checkpoint_directory": str(checkpoint_directory),
        "sample_chunk_size": sample_chunk_size,
        "adaptive_production_rounds": args.adaptive_production_rounds,
        "source_sha256": {
            str(path): _file_sha256(path) for path in source_paths
        },
    }
    common_arguments = {
        "ld": args.ld,
        "reference_delta_time": args.reference_delta_time,
        "reference_rte_steps": args.reference_rte_steps,
        "compiler": compiler,
        "pilot_sample_count": args.pilot_samples,
        "minimum_production_sample_count": args.minimum_production_samples,
        "maximum_production_sample_count": args.maximum_production_samples,
        "prediction_relative_standard_error_target": (
            args.prediction_relative_se_target
        ),
        "allocation_safety_factor": args.allocation_safety_factor,
        "seed": args.seed,
        "maximum_workers": args.maximum_workers,
        "persistent_cache_path": args.persistent_cache,
        "checkpoint_directory": checkpoint_directory,
        "provenance": provenance,
    }
    if args.mode == "calibration":
        target_event_counts = tuple(
            int(value.strip())
            for value in str(args.target_event_counts).split(",")
            if value.strip()
        )
        payload = calibrate_connected_cluster_cost_model(
            hamiltonian,
            **common_arguments,
            target_event_counts=target_event_counts,
            cost_aware_allocation=not args.equal_cost_allocation,
            chunk_patterns=not args.no_chunk_patterns,
            sample_chunk_size=sample_chunk_size,
            adaptive_production_rounds=args.adaptive_production_rounds,
        )
        write_connected_cluster_calibration(payload, output)
        summary = {
            "condition_fingerprint": payload["condition_fingerprint"],
            "calibration_fingerprint": payload["calibration_fingerprint"],
            "total_seconds": payload["performance"]["total_seconds"],
        }
    else:
        payload = validate_operational_connected_cluster_cost_model(
            hamiltonian,
            **common_arguments,
            holdout_zero_sample_count=args.holdout_zero_samples,
            holdout_single_rare_sample_count=args.holdout_single_order2_samples,
        )
        write_connected_cluster_validation(payload, output)
        summary = payload["summary"]
    print(output)
    print(payload["allocation"])
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
