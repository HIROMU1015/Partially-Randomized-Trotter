#!/usr/bin/env python3
"""Run the three held-out validations for the hierarchical cost model."""

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
from trotterlib.hierarchical_cost_validation import (
    validate_controlled_repetition_extension,
    validate_rte_cluster_extension,
    write_hierarchical_cost_validation,
)
from trotterlib.rte import CompilerSettings


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
    parser.add_argument("--l8-calibration-samples", type=int, default=2_000)
    parser.add_argument("--l8-holdout-samples", type=int, default=2_000)
    parser.add_argument("--k2-calibration-samples", type=int, default=500)
    parser.add_argument("--k2-holdout-samples", type=int, default=500)
    parser.add_argument("--controlled-calibration-samples", type=int, default=300)
    parser.add_argument("--controlled-holdout-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--maximum-workers", type=int, default=3)
    parser.add_argument("--optimization-level", type=int, default=1)
    parser.add_argument("--transpiler-seed", type=int, default=17)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("artifacts/hierarchical_cost_validation"),
    )
    return parser


def _slug(args: argparse.Namespace) -> str:
    basis = str(args.basis).lower().replace("-", "").replace(" ", "_")
    distance = int(round(100 * float(args.distance)))
    delta = format(float(args.reference_delta_time), ".8g").replace(".", "p")
    return (
        f"h{args.molecule}_{basis}_d{distance}_rank{args.df_rank}_ld{args.ld}_"
        f"dt{delta}_ref{args.reference_rte_steps}"
    )


def _run_cluster(task: dict[str, Any]) -> dict[str, Any]:
    provenance = dict(task["provenance"])
    provenance.update(
        {
            "worker_started_at_utc": datetime.now(timezone.utc).isoformat(),
            "validation_label": task["label"],
        }
    )
    payload = validate_rte_cluster_extension(
        task["hamiltonian"],
        ld=task["ld"],
        reference_delta_time=task["reference_delta_time"],
        reference_rte_steps=task["reference_rte_steps"],
        finite_taylor_order=task["finite_taylor_order"],
        compiler=task["compiler"],
        calibration_sample_count=task["calibration_sample_count"],
        holdout_sequence_lengths=task["holdout_sequence_lengths"],
        holdout_sample_count=task["holdout_sample_count"],
        seed=task["seed"],
        maximum_exact_events=1_000,
        maximum_samples=max(
            task["calibration_sample_count"],
            task["holdout_sample_count"],
        ),
        cache_maximum_entries=8_192,
        provenance=provenance,
    )
    write_hierarchical_cost_validation(payload, task["output"])
    return {
        "label": task["label"],
        "output": str(task["output"]),
        "summary": payload["summary"],
        "preparation_hash": payload["hamiltonian"]["preparation_hash"],
    }


def _run_controlled(task: dict[str, Any]) -> dict[str, Any]:
    provenance = dict(task["provenance"])
    provenance.update(
        {
            "worker_started_at_utc": datetime.now(timezone.utc).isoformat(),
            "validation_label": task["label"],
        }
    )
    payload = validate_controlled_repetition_extension(
        task["hamiltonian"],
        ld=task["ld"],
        reference_delta_time=task["reference_delta_time"],
        reference_rte_steps=task["reference_rte_steps"],
        finite_taylor_order=0,
        compiler=task["compiler"],
        calibration_sample_count=task["calibration_sample_count"],
        holdout_sample_count=task["holdout_sample_count"],
        seed=task["seed"],
        maximum_samples=max(
            task["calibration_sample_count"],
            task["holdout_sample_count"],
        ),
        cache_maximum_entries=8_192,
        provenance=provenance,
    )
    write_hierarchical_cost_validation(payload, task["output"])
    return {
        "label": task["label"],
        "output": str(task["output"]),
        "summary": payload["summary"],
        "preparation_hash": payload["hamiltonian"]["preparation_hash"],
    }


def main() -> int:
    args = _parser().parse_args()
    source_paths = (
        Path("src/trotterlib/hierarchical_cost_validation.py"),
        Path("src/trotterlib/rte_compiled_cost.py"),
        Path("src/trotterlib/df_partial_s2_repeated_cost.py"),
        Path("scripts/run_hierarchical_cost_validation.py"),
    )
    hamiltonian, _sector = build_df_h_d_from_molecule(
        args.molecule,
        distance=args.distance,
        basis=args.basis,
        df_rank=args.df_rank,
    )
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
                "scripts/run_hierarchical_cost_validation.py",
                *sys.argv[1:],
            ]
        ),
        "python_version": platform.python_version(),
        "qiskit_version": qiskit.__version__,
        "platform": platform.platform(),
        "shared_parent_hamiltonian_payload": True,
        "parent_hamiltonian_hash": df_hamiltonian_hash(hamiltonian),
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
        "provenance": provenance,
    }
    tasks: list[tuple[Any, dict[str, Any]]] = [
        (
            _run_cluster,
            {
                **common,
                "label": "k0_l8_holdout",
                "finite_taylor_order": 0,
                "calibration_sample_count": args.l8_calibration_samples,
                "holdout_sample_count": args.l8_holdout_samples,
                "holdout_sequence_lengths": (8,),
                "seed": args.seed,
                "output": args.output_directory
                / (
                    f"{slug}_k0_l8_ncal{args.l8_calibration_samples}_"
                    f"nhold{args.l8_holdout_samples}_v1.json"
                ),
            },
        ),
        (
            _run_cluster,
            {
                **common,
                "label": "k2_internal_cluster_holdout",
                "finite_taylor_order": 2,
                "calibration_sample_count": args.k2_calibration_samples,
                "holdout_sample_count": args.k2_holdout_samples,
                "holdout_sequence_lengths": (4, 6),
                "seed": args.seed + 10_000,
                "output": args.output_directory
                / (
                    f"{slug}_k2_l4l6_ncal{args.k2_calibration_samples}_"
                    f"nhold{args.k2_holdout_samples}_v1.json"
                ),
            },
        ),
        (
            _run_controlled,
            {
                **common,
                "label": "controlled_q4_affine_holdout",
                "calibration_sample_count": args.controlled_calibration_samples,
                "holdout_sample_count": args.controlled_holdout_samples,
                "seed": args.seed + 20_000,
                "output": args.output_directory
                / (
                    f"{slug}_k0_controlled_q4_"
                    f"ncal{args.controlled_calibration_samples}_"
                    f"nhold{args.controlled_holdout_samples}_v1.json"
                ),
            },
        ),
    ]

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
