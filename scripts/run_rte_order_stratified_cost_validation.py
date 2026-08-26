#!/usr/bin/env python3
"""Run the K=2 Taylor-order-stratified RTE cost validation."""

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

from trotterlib.df_hamiltonian import build_df_h_d_from_molecule
from trotterlib.df_partial_randomized_pf import df_hamiltonian_hash
from trotterlib.rte import CompilerSettings
from trotterlib.rte_connected_cluster_cost_validation import (
    load_connected_cluster_hamiltonian_snapshot,
)
from trotterlib.rte_order_stratified_cost_validation import (
    validate_paired_order_stratified_k2_cluster_model,
    validate_order_stratified_k2_cost_model,
    write_paired_cluster_validation,
    write_order_stratified_cost_validation,
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
    parser.add_argument("--reference-delta-time", type=float, default=0.1)
    parser.add_argument("--reference-rte-steps", type=int, default=4)
    parser.add_argument("--common-samples", type=int, default=500)
    parser.add_argument("--single-order2-samples", type=int, default=100)
    parser.add_argument("--multi-order2-samples", type=int, default=2)
    parser.add_argument("--paired-common-samples", type=int, default=100)
    parser.add_argument("--paired-single-order2-samples", type=int, default=25)
    parser.add_argument(
        "--paired-local-residual",
        action="store_true",
        help="Validate the fixed local-window formula with paired trajectories.",
    )
    parser.add_argument(
        "--shared-both",
        action="store_true",
        help="Run independent and paired validations on one shared Hamiltonian.",
    )
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--maximum-workers", type=int, default=3)
    parser.add_argument("--optimization-level", type=int, default=1)
    parser.add_argument("--transpiler-seed", type=int, default=17)
    parser.add_argument("--hamiltonian-snapshot", type=Path)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("artifacts/rte_order_stratified_cost_validation"),
    )
    return parser


def _slug(args: argparse.Namespace, *, mode: str | None = None) -> str:
    basis = str(args.basis).lower().replace("-", "").replace(" ", "_")
    distance = int(round(100 * float(args.distance)))
    delta = format(float(args.reference_delta_time), ".8g").replace(".", "p")
    if mode is None:
        mode = "paired" if args.paired_local_residual else "independent"
    common_samples = (
        args.paired_common_samples if mode == "shared_paired" else args.common_samples
    )
    single_samples = (
        args.paired_single_order2_samples
        if mode == "shared_paired"
        else args.single_order2_samples
    )
    return (
        f"h{args.molecule}_{basis}_d{distance}_rank{args.df_rank}_ld{args.ld}_"
        f"dt{delta}_ref{args.reference_rte_steps}_k2_{mode}_"
        f"ncommon{common_samples}_nsingle{single_samples}_"
        f"nmulti{args.multi_order2_samples}_v1.json"
    )


def main() -> int:
    args = _parser().parse_args()
    source_paths = (
        Path("src/trotterlib/rte_order_stratified_cost_validation.py"),
        Path("src/trotterlib/rte_compiled_cost.py"),
        Path("src/trotterlib/rte.py"),
        Path("src/trotterlib/df_rte_circuit.py"),
        Path("src/trotterlib/df_rte_qiskit.py"),
        Path("scripts/run_rte_order_stratified_cost_validation.py"),
    )
    if args.hamiltonian_snapshot is None:
        hamiltonian, _sector = build_df_h_d_from_molecule(
            args.molecule,
            distance=args.distance,
            basis=args.basis,
            df_rank=args.df_rank,
        )
        hamiltonian_source = "molecular_rebuild"
    else:
        hamiltonian = load_connected_cluster_hamiltonian_snapshot(
            args.hamiltonian_snapshot
        )
        hamiltonian_source = "existing_exact_snapshot"
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
    command = shlex.join(
        [
            ".venv311/bin/python",
            "scripts/run_rte_order_stratified_cost_validation.py",
            *sys.argv[1:],
        ]
    )
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_head(),
        "git_worktree_status_before_generation": _git_status(),
        "evidence_status": "local_worktree_validation_not_immutable_ci",
        "command": command,
        "python_version": platform.python_version(),
        "qiskit_version": qiskit.__version__,
        "platform": platform.platform(),
        "parent_hamiltonian_hash": df_hamiltonian_hash(hamiltonian),
        "hamiltonian_source": hamiltonian_source,
        "hamiltonian_snapshot": (
            None
            if args.hamiltonian_snapshot is None
            else str(args.hamiltonian_snapshot)
        ),
        "hamiltonian_snapshot_sha256": (
            None
            if args.hamiltonian_snapshot is None
            else _file_sha256(args.hamiltonian_snapshot)
        ),
        "source_sha256": {
            str(path): _file_sha256(path) for path in source_paths
        },
    }
    if args.shared_both:
        independent = validate_order_stratified_k2_cost_model(
            hamiltonian,
            ld=args.ld,
            reference_delta_time=args.reference_delta_time,
            reference_rte_steps=args.reference_rte_steps,
            compiler=compiler,
            common_sample_count=args.common_samples,
            single_rare_sample_count=args.single_order2_samples,
            multi_rare_sample_count=args.multi_order2_samples,
            seed=args.seed,
            maximum_workers=args.maximum_workers,
            provenance={**provenance, "shared_batch_mode": "independent"},
        )
        independent_output = args.output_directory / _slug(
            args, mode="shared_independent"
        )
        write_order_stratified_cost_validation(independent, independent_output)
        print(independent_output, flush=True)
        print(independent["summary"], flush=True)
        paired = validate_paired_order_stratified_k2_cluster_model(
            hamiltonian,
            ld=args.ld,
            reference_delta_time=args.reference_delta_time,
            reference_rte_steps=args.reference_rte_steps,
            compiler=compiler,
            common_sample_count=args.paired_common_samples,
            single_rare_sample_count=args.paired_single_order2_samples,
            seed=args.seed,
            maximum_workers=args.maximum_workers,
            provenance={**provenance, "shared_batch_mode": "paired"},
        )
        if (
            independent["hamiltonian"]["preparation_hash"]
            != paired["hamiltonian"]["preparation_hash"]
        ):
            raise RuntimeError("Shared-batch preparation hashes do not match.")
        paired_output = args.output_directory / _slug(args, mode="shared_paired")
        write_paired_cluster_validation(paired, paired_output)
        print(paired_output, flush=True)
        print(paired["summary"], flush=True)
        print(
            "shared_preparation_hash="
            f"{paired['hamiltonian']['preparation_hash']}",
            flush=True,
        )
        return 0
    if args.paired_local_residual:
        payload = validate_paired_order_stratified_k2_cluster_model(
            hamiltonian,
            ld=args.ld,
            reference_delta_time=args.reference_delta_time,
            reference_rte_steps=args.reference_rte_steps,
            compiler=compiler,
            common_sample_count=args.common_samples,
            single_rare_sample_count=args.single_order2_samples,
            multi_rare_sample_count=args.multi_order2_samples,
            seed=args.seed,
            maximum_workers=args.maximum_workers,
            provenance=provenance,
        )
    else:
        payload = validate_order_stratified_k2_cost_model(
            hamiltonian,
            ld=args.ld,
            reference_delta_time=args.reference_delta_time,
            reference_rte_steps=args.reference_rte_steps,
            compiler=compiler,
            common_sample_count=args.common_samples,
            single_rare_sample_count=args.single_order2_samples,
            multi_rare_sample_count=args.multi_order2_samples,
            seed=args.seed,
            maximum_workers=args.maximum_workers,
            provenance=provenance,
        )
    output = args.output_directory / _slug(args)
    if args.paired_local_residual:
        write_paired_cluster_validation(payload, output)
    else:
        write_order_stratified_cost_validation(payload, output)
    print(output)
    if not args.paired_local_residual:
        print(payload["configuration"]["distribution"])
    print(payload["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
