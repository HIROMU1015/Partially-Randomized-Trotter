#!/usr/bin/env python3
"""Check compiled local RTE metrics across numeric short-step angles."""

from __future__ import annotations

import argparse
from pathlib import Path

import qiskit

from trotterlib.rte import CompilerSettings
from trotterlib.rte_connected_cluster_cost_validation import (
    load_connected_cluster_hamiltonian_snapshot,
)
from trotterlib.rte_cost_angle_invariance_validation import (
    validate_rte_cost_angle_invariance,
    write_rte_cost_angle_invariance_validation,
)


def _float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def _int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("hamiltonian_snapshot", type=Path)
    parser.add_argument("--ld", type=int, required=True)
    parser.add_argument("--short-step-times", type=_float_tuple, required=True)
    parser.add_argument("--cluster-lengths", type=_int_tuple, default=(1, 2, 3))
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--optimization-level", type=int, default=1)
    parser.add_argument("--transpiler-seed", type=int, default=17)
    parser.add_argument("--persistent-cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
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
    payload = validate_rte_cost_angle_invariance(
        hamiltonian,
        ld=args.ld,
        short_step_times=args.short_step_times,
        compiler=compiler,
        sample_count_per_pattern=args.samples,
        seed=args.seed,
        cluster_lengths=args.cluster_lengths,
        persistent_cache_path=args.persistent_cache,
        provenance={
            "evidence_status": "local_worktree_validation_not_immutable_ci",
            "hamiltonian_snapshot": str(args.hamiltonian_snapshot),
        },
    )
    write_rte_cost_angle_invariance_validation(payload, args.output)
    print(args.output)
    print(payload["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
