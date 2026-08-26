#!/usr/bin/env python3
"""Calibrate and hold out-test compiled RTE boundary-cost corrections."""

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
from trotterlib.rte import CompilerSettings
from trotterlib.rte_boundary_cost_validation import (
    validate_rte_boundary_cost_model,
    write_rte_boundary_cost_validation,
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
    parser.add_argument("--calibration-samples", type=int, default=300)
    parser.add_argument("--holdout-lengths", type=_int_tuple, default=(4, 6))
    parser.add_argument("--holdout-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--coefficient-atol", type=float, default=1e-12)
    parser.add_argument("--maximum-exact-events", type=int, default=1_000)
    parser.add_argument("--maximum-samples", type=int, default=10_000)
    parser.add_argument("--cache-maximum-entries", type=int, default=4_096)
    parser.add_argument("--optimization-level", type=int, default=1)
    parser.add_argument("--transpiler-seed", type=int, default=17)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("artifacts/rte_boundary_cost_validation"),
    )
    parser.add_argument("--output", type=Path)
    return parser


def _default_output(args: argparse.Namespace) -> Path:
    basis = str(args.basis).lower().replace("-", "").replace(" ", "_")
    distance = int(round(100 * float(args.distance)))
    delta = format(float(args.reference_delta_time), ".8g").replace(".", "p")
    return args.output_directory / (
        f"h{args.molecule}_{basis}_d{distance}_rank{args.df_rank}_ld{args.ld}_"
        f"dt{delta}_ref{args.reference_rte_steps}_"
        f"k{args.finite_taylor_order}_v1.json"
    )


def main() -> int:
    args = _parser().parse_args()
    module_path = Path("src/trotterlib/rte_boundary_cost_validation.py")
    script_path = Path("scripts/run_rte_boundary_cost_validation.py")
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_head(),
        "git_worktree_status_before_generation": _git_status(),
        "evidence_status": "local_worktree_validation_not_immutable_ci",
        "command": shlex.join(
            [
                ".venv311/bin/python",
                "scripts/run_rte_boundary_cost_validation.py",
                *sys.argv[1:],
            ]
        ),
        "python_version": platform.python_version(),
        "qiskit_version": qiskit.__version__,
        "platform": platform.platform(),
        "source_sha256": {
            str(module_path): _file_sha256(module_path),
            str(script_path): _file_sha256(script_path),
        },
    }
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
    payload = validate_rte_boundary_cost_model(
        hamiltonian,
        ld=args.ld,
        reference_delta_time=args.reference_delta_time,
        reference_rte_steps=args.reference_rte_steps,
        finite_taylor_order=args.finite_taylor_order,
        compiler=compiler,
        calibration_sample_count=args.calibration_samples,
        holdout_sequence_lengths=args.holdout_lengths,
        holdout_sample_count=args.holdout_samples,
        seed=args.seed,
        coefficient_atol=args.coefficient_atol,
        maximum_exact_events=args.maximum_exact_events,
        maximum_samples=args.maximum_samples,
        cache_maximum_entries=args.cache_maximum_entries,
        provenance=provenance,
    )
    output = args.output or _default_output(args)
    write_rte_boundary_cost_validation(payload, output)
    print(output)
    print(payload["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
