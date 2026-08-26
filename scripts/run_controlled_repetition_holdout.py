#!/usr/bin/env python3
"""Run a fixed-snapshot controlled q=1,2 affine holdout at an unused q."""

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

from trotterlib.hierarchical_cost_validation import (
    validate_controlled_repetition_extension,
    write_hierarchical_cost_validation,
)
from trotterlib.rte import CompilerSettings
from trotterlib.rte_connected_cluster_cost_validation import (
    load_connected_cluster_hamiltonian_snapshot,
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
    parser.add_argument("hamiltonian_snapshot", type=Path)
    parser.add_argument("--ld", type=int, default=3)
    parser.add_argument("--reference-delta-time", type=float, default=0.1)
    parser.add_argument("--reference-rte-steps", type=int, default=4)
    parser.add_argument("--finite-taylor-order", type=int, default=0)
    parser.add_argument("--calibration-samples", type=int, default=150)
    parser.add_argument("--holdout-samples", type=int, default=150)
    parser.add_argument("--holdout-q", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--optimization-level", type=int, default=1)
    parser.add_argument("--transpiler-seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    hamiltonian = load_connected_cluster_hamiltonian_snapshot(
        args.hamiltonian_snapshot
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
    sources = (
        Path("src/trotterlib/hierarchical_cost_validation.py"),
        Path("scripts/run_controlled_repetition_holdout.py"),
    )
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_head(),
        "git_worktree_status_before_generation": _git_status(),
        "evidence_status": "local_worktree_validation_not_immutable_ci",
        "command": shlex.join(
            [
                ".venv311/bin/python",
                "scripts/run_controlled_repetition_holdout.py",
                *sys.argv[1:],
            ]
        ),
        "python_version": platform.python_version(),
        "qiskit_version": qiskit.__version__,
        "platform": platform.platform(),
        "source_sha256": {str(path): _sha256(path) for path in sources},
        "hamiltonian_snapshot": str(args.hamiltonian_snapshot),
        "hamiltonian_snapshot_sha256": _sha256(args.hamiltonian_snapshot),
    }
    payload = validate_controlled_repetition_extension(
        hamiltonian,
        ld=args.ld,
        reference_delta_time=args.reference_delta_time,
        reference_rte_steps=args.reference_rte_steps,
        finite_taylor_order=args.finite_taylor_order,
        compiler=compiler,
        calibration_sample_count=args.calibration_samples,
        holdout_sample_count=args.holdout_samples,
        holdout_repetition_count=args.holdout_q,
        seed=args.seed,
        maximum_samples=max(args.calibration_samples, args.holdout_samples),
        provenance=provenance,
    )
    write_hierarchical_cost_validation(payload, args.output)
    print(args.output, flush=True)
    print(payload["summary"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
