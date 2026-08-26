#!/usr/bin/env python3
"""Run high-statistics boundary-cost replications from one H-chain payload."""

from __future__ import annotations

import argparse
import hashlib
import multiprocessing
import platform
import shlex
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import qiskit

from trotterlib.df_hamiltonian import build_df_h_d_from_molecule
from trotterlib.df_partial_randomized_pf import df_hamiltonian_hash
from trotterlib.rte import CompilerSettings
from trotterlib.rte_boundary_cost_validation import (
    validate_rte_boundary_cost_model,
    write_rte_boundary_cost_validation,
)
from trotterlib.rte_boundary_pair_validation import (
    validate_rte_boundary_pairs,
    write_rte_boundary_pair_validation,
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
    parser.add_argument("--finite-taylor-order", type=int, default=0)
    parser.add_argument("--replication-samples", type=int, default=1_000)
    parser.add_argument(
        "--replication-seeds",
        type=_int_tuple,
        default=(20260823, 20261823),
    )
    parser.add_argument("--pair-calibration-samples", type=int, default=1_500)
    parser.add_argument("--pair-holdout-samples", type=int, default=1_500)
    parser.add_argument("--pair-calibration-seed", type=int, default=20263823)
    parser.add_argument("--pair-holdout-seed", type=int, default=20263824)
    parser.add_argument("--optimization-level", type=int, default=1)
    parser.add_argument("--transpiler-seed", type=int, default=17)
    parser.add_argument("--maximum-workers", type=int, default=3)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("artifacts/rte_boundary_cost_validation"),
    )
    parser.add_argument(
        "--pair-output-directory",
        type=Path,
        default=Path("artifacts/rte_boundary_pair_validation"),
    )
    return parser


def _slug(args: argparse.Namespace) -> str:
    basis = str(args.basis).lower().replace("-", "").replace(" ", "_")
    distance = int(round(100 * float(args.distance)))
    delta = format(float(args.reference_delta_time), ".8g").replace(".", "p")
    return (
        f"h{args.molecule}_{basis}_d{distance}_rank{args.df_rank}_ld{args.ld}_"
        f"dt{delta}_ref{args.reference_rte_steps}_k{args.finite_taylor_order}"
    )


def _run_cluster(task: dict[str, Any]) -> dict[str, Any]:
    seed = task["seed"]
    provenance = dict(task["provenance"])
    provenance.update(
        {
            "worker_started_at_utc": datetime.now(timezone.utc).isoformat(),
            "replication_kind": "full_cluster_calibration_and_holdout",
            "replication_seed": seed,
        }
    )
    payload = validate_rte_boundary_cost_model(
        task["hamiltonian"],
        ld=task["ld"],
        reference_delta_time=task["reference_delta_time"],
        reference_rte_steps=task["reference_rte_steps"],
        finite_taylor_order=task["finite_taylor_order"],
        compiler=task["compiler"],
        calibration_sample_count=task["sample_count"],
        holdout_sequence_lengths=(4, 6),
        holdout_sample_count=task["sample_count"],
        seed=seed,
        maximum_samples=task["sample_count"],
        cache_maximum_entries=8_192,
        provenance=provenance,
    )
    output = task["output"]
    write_rte_boundary_cost_validation(payload, output)
    return {
        "kind": "cluster",
        "output": str(output),
        "summary": payload["summary"],
        "preparation_hash": payload["hamiltonian"]["preparation_hash"],
    }


def _run_pair(task: dict[str, Any]) -> dict[str, Any]:
    provenance = dict(task["provenance"])
    provenance.update(
        {
            "worker_started_at_utc": datetime.now(timezone.utc).isoformat(),
            "replication_kind": "fragment_pair_stratified_calibration_and_holdout",
        }
    )
    payload = validate_rte_boundary_pairs(
        task["hamiltonian"],
        ld=task["ld"],
        reference_delta_time=task["reference_delta_time"],
        reference_rte_steps=task["reference_rte_steps"],
        finite_taylor_order=task["finite_taylor_order"],
        compiler=task["compiler"],
        calibration_sample_count=task["calibration_sample_count"],
        holdout_sample_count=task["holdout_sample_count"],
        calibration_seed=task["calibration_seed"],
        holdout_seed=task["holdout_seed"],
        maximum_samples=max(
            task["calibration_sample_count"],
            task["holdout_sample_count"],
        ),
        provenance=provenance,
    )
    output = task["output"]
    write_rte_boundary_pair_validation(payload, output)
    return {
        "kind": "pair",
        "output": str(output),
        "summary": payload["summary"],
        "preparation_hash": payload["hamiltonian"]["preparation_hash"],
    }


def main() -> int:
    args = _parser().parse_args()
    if len(set(args.replication_seeds)) != len(args.replication_seeds):
        raise ValueError("replication-seeds must be unique.")
    if args.pair_calibration_seed == args.pair_holdout_seed:
        raise ValueError("Pair calibration and holdout seeds must differ.")

    source_paths = (
        Path("src/trotterlib/rte_compiled_cost.py"),
        Path("src/trotterlib/rte_boundary_cost_validation.py"),
        Path("src/trotterlib/rte_boundary_pair_validation.py"),
        Path("scripts/run_rte_boundary_cost_replication.py"),
    )
    hamiltonian, _sector = build_df_h_d_from_molecule(
        args.molecule,
        distance=args.distance,
        basis=args.basis,
        df_rank=args.df_rank,
    )
    parent_hamiltonian_hash = df_hamiltonian_hash(hamiltonian)
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
                "scripts/run_rte_boundary_cost_replication.py",
                *sys.argv[1:],
            ]
        ),
        "python_version": platform.python_version(),
        "qiskit_version": qiskit.__version__,
        "platform": platform.platform(),
        "shared_parent_hamiltonian_payload": True,
        "parent_hamiltonian_hash": parent_hamiltonian_hash,
        "parallel_worker_count": min(
            args.maximum_workers,
            len(args.replication_seeds) + 1,
        ),
        "source_sha256": {
            str(path): _file_sha256(path) for path in source_paths
        },
    }
    slug = _slug(args)
    common = {
        "hamiltonian": hamiltonian,
        "compiler": compiler,
        "ld": args.ld,
        "reference_delta_time": args.reference_delta_time,
        "reference_rte_steps": args.reference_rte_steps,
        "finite_taylor_order": args.finite_taylor_order,
        "provenance": provenance,
    }
    tasks: list[tuple[Any, dict[str, Any]]] = []
    for seed in args.replication_seeds:
        tasks.append(
            (
                _run_cluster,
                {
                    **common,
                    "seed": seed,
                    "sample_count": args.replication_samples,
                    "output": args.output_directory
                    / f"{slug}_n{args.replication_samples}_seed{seed}_v1.json",
                },
            )
        )
    tasks.append(
        (
            _run_pair,
            {
                **common,
                "calibration_sample_count": args.pair_calibration_samples,
                "holdout_sample_count": args.pair_holdout_samples,
                "calibration_seed": args.pair_calibration_seed,
                "holdout_seed": args.pair_holdout_seed,
                "output": args.pair_output_directory
                / (
                    f"{slug}_ncal{args.pair_calibration_samples}_"
                    f"nhold{args.pair_holdout_samples}_v1.json"
                ),
            },
        )
    )

    results = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(args.maximum_workers, len(tasks)),
        mp_context=context,
    ) as executor:
        futures = [executor.submit(function, task) for function, task in tasks]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(result["output"], flush=True)
            print(result["summary"], flush=True)

    preparation_hashes = {result["preparation_hash"] for result in results}
    if len(preparation_hashes) != 1:
        raise RuntimeError("Worker preparation hashes do not match.")
    print(f"shared_preparation_hash={next(iter(preparation_hashes))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
