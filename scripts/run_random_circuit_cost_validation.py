#!/usr/bin/env python3
"""Run one H-chain pilot for the randomized compiled-cost model."""

from __future__ import annotations

import argparse
import hashlib
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import qiskit

from trotterlib.df_hamiltonian import build_df_h_d_from_molecule
from trotterlib.random_circuit_cost_validation import (
    validate_random_circuit_cost_model,
    write_random_circuit_cost_validation,
)
from trotterlib.rte import CompilerSettings


def _int_tuple(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integers.") from exc
    if not result:
        raise argparse.ArgumentTypeError("At least one integer is required.")
    return result


def _float_tuple(value: str) -> tuple[float, ...]:
    try:
        result = tuple(
            float(item.strip()) for item in value.split(",") if item.strip()
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated numbers.") from exc
    if not result:
        raise argparse.ArgumentTypeError("At least one number is required.")
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
    parser.add_argument("--ld-values", type=_int_tuple)
    parser.add_argument("--delta-time", type=float, default=0.1)
    parser.add_argument("--delta-time-values", type=_float_tuple)
    parser.add_argument("--rte-steps", type=int, default=1)
    parser.add_argument("--rte-step-values", type=_int_tuple)
    parser.add_argument("--finite-taylor-order", type=int, default=0)
    parser.add_argument("--finite-taylor-order-values", type=_int_tuple)
    parser.add_argument("--sample-counts", type=_int_tuple, default=(30, 100))
    parser.add_argument(
        "--scope",
        choices=("both", "partial-s2", "rte-occurrence"),
        default="both",
    )
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--coefficient-atol", type=float, default=1e-12)
    parser.add_argument("--maximum-exact-event-sequences", type=int, default=1_000)
    parser.add_argument("--maximum-samples", type=int, default=10_000)
    parser.add_argument("--cache-maximum-entries", type=int, default=4_096)
    parser.add_argument("--optimization-level", type=int, default=1)
    parser.add_argument("--transpiler-seed", type=int, default=17)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("artifacts/random_circuit_cost_validation"),
    )
    return parser


def _default_output(
    args: argparse.Namespace,
    *,
    ld: int,
    delta_time: float,
    rte_steps: int,
    finite_taylor_order: int,
) -> Path:
    basis = str(args.basis).lower().replace("-", "").replace(" ", "_")
    distance = int(round(100 * float(args.distance)))
    delta = format(float(delta_time), ".8g").replace(".", "p")
    return args.output_directory / (
        f"h{args.molecule}_{basis}_d{distance}_rank{args.df_rank}_ld{ld}_"
        f"dt{delta}_r{rte_steps}_k{finite_taylor_order}_v1.json"
    )


def main() -> int:
    args = _parser().parse_args()
    ld_values = args.ld_values or (args.ld,)
    delta_time_values = args.delta_time_values or (args.delta_time,)
    rte_step_values = args.rte_step_values or (args.rte_steps,)
    cutoff_values = args.finite_taylor_order_values or (
        args.finite_taylor_order,
    )
    combinations = tuple(
        product(ld_values, delta_time_values, rte_step_values, cutoff_values)
    )
    if args.output is not None and len(combinations) != 1:
        raise ValueError("--output is only valid for one validation point.")
    module_path = Path("src/trotterlib/random_circuit_cost_validation.py")
    script_path = Path("scripts/run_random_circuit_cost_validation.py")
    common_provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_head(),
        "git_worktree_status_before_generation": _git_status(),
        "evidence_status": "local_worktree_validation_not_immutable_ci",
        "command": shlex.join(
            [
                ".venv311/bin/python",
                "scripts/run_random_circuit_cost_validation.py",
                *sys.argv[1:],
            ]
        ),
        "python_version": platform.python_version(),
        "qiskit_version": qiskit.__version__,
        "platform": platform.platform(),
        "shared_hamiltonian_generation": True,
        "batch_ld_values": list(ld_values),
        "batch_delta_time_values": list(delta_time_values),
        "batch_rte_step_values": list(rte_step_values),
        "batch_finite_taylor_order_values": list(cutoff_values),
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
    for position, (ld, delta_time, rte_steps, cutoff) in enumerate(combinations):
        output = args.output or _default_output(
            args,
            ld=ld,
            delta_time=delta_time,
            rte_steps=rte_steps,
            finite_taylor_order=cutoff,
        )
        provenance = dict(common_provenance)
        provenance["batch_position"] = position
        payload = validate_random_circuit_cost_model(
            hamiltonian,
            ld=ld,
            delta_time=delta_time,
            rte_steps=rte_steps,
            finite_taylor_order=cutoff,
            monte_carlo_sample_counts=args.sample_counts,
            compiler=compiler,
            evaluation_scopes={
                "both": ("partial_s2", "rte_occurrence"),
                "partial-s2": ("partial_s2",),
                "rte-occurrence": ("rte_occurrence",),
            }[args.scope],
            seed=args.seed,
            coefficient_atol=args.coefficient_atol,
            maximum_exact_event_sequences=args.maximum_exact_event_sequences,
            maximum_samples=args.maximum_samples,
            cache_maximum_entries=args.cache_maximum_entries,
            provenance=provenance,
        )
        write_random_circuit_cost_validation(payload, output)
        print(output)
        print(payload["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
