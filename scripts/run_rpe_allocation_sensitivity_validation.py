#!/usr/bin/env python3
"""Run the H4 beta/alpha allocation-sensitivity validation."""

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

from trotterlib.rpe_allocation_sensitivity_validation import (
    validate_rpe_allocation_sensitivity,
    write_rpe_allocation_sensitivity_validation,
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
DEFAULT_DIRECT = Path(
    "artifacts/rpe_round_cost_connection_validation/"
    "h4_sto3g_d100_rank12_ld3_dt0p1_r4_k2_q1_q2_q4_mc8_v1.json"
)
DEFAULT_PROXY_VALIDATION = Path(
    "artifacts/rpe_hadamard_proxy_resource_validation/2026-09-01/"
    "h4_sto3g_d100_rank12_ld3_dt0p1_r4_k2_cal_q1_q2_q4_"
    "hold_q8_mc8_v1.proxy_validation.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/rpe_allocation_sensitivity_validation/2026-09-01/"
    "h4_sto3g_d100_rank12_ld3_dt0p1_r4_k2_q1_q2_q4_q8_"
    "beta_alpha_sensitivity_v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(command: list[str]) -> str | list[str] | None:
    result = subprocess.run(
        ["git", *command],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    lines = result.stdout.splitlines()
    return lines[0] if len(lines) == 1 else lines


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--direct", type=Path, default=DEFAULT_DIRECT)
    parser.add_argument(
        "--proxy-validation", type=Path, default=DEFAULT_PROXY_VALIDATION
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = _parser().parse_args()
    hamiltonian = load_connected_cluster_hamiltonian_snapshot(args.snapshot)
    direct_payload = json.loads(args.direct.read_text(encoding="utf-8"))
    proxy_validation = RPEHadamardCompiledCostProxyValidationResult.read_json(
        args.proxy_validation
    )
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
    source_paths = (
        Path("src/trotterlib/rpe_allocation_sensitivity_validation.py"),
        Path("scripts/run_rpe_allocation_sensitivity_validation.py"),
    )
    payload = validate_rpe_allocation_sensitivity(
        hamiltonian,
        compiler,
        direct_payload,
        proxy_validation,
        provenance={
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git(["rev-parse", "HEAD"]),
            "git_worktree_status_before_generation": _git(["status", "--short"]),
            "evidence_status": "local_worktree_validation_not_immutable_ci",
            "command": shlex.join(
                [
                    ".venv311/bin/python",
                    "scripts/run_rpe_allocation_sensitivity_validation.py",
                    *sys.argv[1:],
                ]
            ),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "qiskit_version": qiskit.__version__,
            "inputs": {
                "snapshot": str(args.snapshot),
                "snapshot_sha256": _sha256(args.snapshot),
                "direct_connection": str(args.direct),
                "direct_connection_sha256": _sha256(args.direct),
                "proxy_validation": str(args.proxy_validation),
                "proxy_validation_sha256": _sha256(args.proxy_validation),
            },
            "source_sha256": {str(path): _sha256(path) for path in source_paths},
        },
    )
    write_rpe_allocation_sensitivity_validation(payload, args.output)
    print(
        f"wrote {args.output}; overall_pass={payload['summary']['overall_pass']}; "
        f"selected={payload['selection']['selected_scenario_id']}; "
        f"reduction={payload['summary']['selected_relative_cost_reduction']:.6%}; "
        f"elapsed={payload['performance']['elapsed_seconds']:.3f}s"
    )
    return 0 if payload["summary"]["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
