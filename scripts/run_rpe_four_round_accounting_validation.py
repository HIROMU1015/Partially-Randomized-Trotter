#!/usr/bin/env python3
"""Validate the selected H4 q=1,2,4,8 limited resource aggregation."""

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

import numpy as np
import qiskit

from trotterlib.rpe_four_round_accounting_validation import (
    validate_rpe_four_round_accounting,
    write_rpe_four_round_accounting,
)
from trotterlib.rpe_hadamard_compiled_cost_proxy import (
    RPEHadamardCompiledCostProxyValidationResult,
)
from trotterlib.rte import CompilerSettings
from trotterlib.rte_connected_cluster_cost_validation import (
    load_connected_cluster_hamiltonian_snapshot,
)


DEFAULT_SNAPSHOT = Path(
    "artifacts/rte_connected_cluster_cost_validation/"
    "h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k2_connected_"
    "pilot30_max1500_hold1500_rare375_v1.hamiltonian.npz"
)
DEFAULT_ALLOCATION = Path(
    "artifacts/rpe_allocation_sensitivity_validation/2026-09-01/"
    "h4_sto3g_d100_rank12_ld3_dt0p1_r4_k2_q1_q2_q4_q8_"
    "beta_alpha_sensitivity_v1.json"
)
DEFAULT_DIRECT = Path(
    "artifacts/rpe_round_cost_connection_validation/"
    "h4_sto3g_d100_rank12_ld3_dt0p1_r4_k2_q1_q2_q4_mc8_v1.json"
)
DEFAULT_PROXY = Path(
    "artifacts/rpe_hadamard_proxy_resource_validation/2026-09-01/"
    "h4_sto3g_d100_rank12_ld3_dt0p1_r4_k2_cal_q1_q2_q4_"
    "hold_q8_mc8_v1.proxy_validation.json"
)
DEFAULT_FAILURE = Path(
    "artifacts/rpe_hadamard_failure_validation/"
    "h4_sto3g_d100_rank12_ld3_dt0p1_r4_k2_q1_q2_q4_"
    "marginal100000_fresh_v1.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/rpe_four_round_accounting_validation/2026-09-18/"
    "h4_sto3g_d100_rank12_ld3_dt0p1_r4_k2_q1_q2_q4_q8_limited_v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(command: list[str]) -> str | list[str] | None:
    result = subprocess.run(
        ["git", *command], check=False, capture_output=True, text=True
    )
    if result.returncode != 0:
        return None
    lines = result.stdout.splitlines()
    return lines[0] if len(lines) == 1 else lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--allocation", type=Path, default=DEFAULT_ALLOCATION)
    parser.add_argument("--direct", type=Path, default=DEFAULT_DIRECT)
    parser.add_argument("--proxy-validation", type=Path, default=DEFAULT_PROXY)
    parser.add_argument("--failure", type=Path, default=DEFAULT_FAILURE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    hamiltonian = load_connected_cluster_hamiltonian_snapshot(args.snapshot)
    allocation = json.loads(args.allocation.read_text(encoding="utf-8"))
    direct = json.loads(args.direct.read_text(encoding="utf-8"))
    proxy = RPEHadamardCompiledCostProxyValidationResult.read_json(
        args.proxy_validation
    )
    failure = json.loads(args.failure.read_text(encoding="utf-8"))
    compiler = CompilerSettings(
        basis_gates=("rz", "sx", "x", "cx"),
        backend_name=None,
        coupling_map=None,
        optimization_level=1,
        layout_method=None,
        routing_method=None,
        transpiler_seed=17,
        qiskit_version=qiskit.__version__,
    )
    sources = (
        Path("src/trotterlib/rpe_four_round_accounting_validation.py"),
        Path("scripts/run_rpe_four_round_accounting_validation.py"),
    )
    inputs = {
        "snapshot": args.snapshot,
        "allocation": args.allocation,
        "direct": args.direct,
        "proxy_validation": args.proxy_validation,
        "previous_failure": args.failure,
    }
    payload = validate_rpe_four_round_accounting(
        hamiltonian,
        compiler,
        allocation,
        direct,
        proxy,
        failure,
        provenance={
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git(["rev-parse", "HEAD"]),
            "git_worktree_status_before_generation": _git(["status", "--short"]),
            "evidence_status": "local_worktree_validation_not_immutable_ci",
            "command": shlex.join(
                [
                    ".venv311/bin/python",
                    "scripts/run_rpe_four_round_accounting_validation.py",
                    *sys.argv[1:],
                ]
            ),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "qiskit_version": qiskit.__version__,
            "inputs": {
                name: {"path": str(path), "sha256": _sha256(path)}
                for name, path in inputs.items()
            },
            "source_sha256": {str(path): _sha256(path) for path in sources},
        },
    )
    write_rpe_four_round_accounting(payload, args.output)
    print(
        f"wrote {args.output}; overall_pass={payload['summary']['overall_pass']}; "
        f"RZ={payload['limited_aggregation']['total_rz_cost']:.6f}; "
        f"union={payload['limited_aggregation']['total_axis_alpha_union_bound']:.17g}"
    )
    return 0 if payload["summary"]["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
