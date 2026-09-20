#!/usr/bin/env python3
"""Run the executed-delta and round-specific finite-RTE schedule validation."""

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

from trotterlib.df_hamiltonian import PhysicalSector
from trotterlib.rpe_allocation_sensitivity_validation import (
    validate_rpe_allocation_sensitivity_payload,
)
from trotterlib.rpe_delta_round_schedule_validation import (
    validate_rpe_delta_round_schedule,
    write_rpe_delta_round_schedule_validation,
)
from trotterlib.rpe_target_round_horizon_validation import (
    validate_rpe_target_round_horizon_payload,
)
from trotterlib.rte_connected_cluster_cost_validation import (
    load_connected_cluster_hamiltonian_snapshot,
)


DEFAULT_SNAPSHOT = Path(
    "artifacts/rte_connected_cluster_cost_validation/"
    "h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k2_connected_"
    "pilot30_max1500_hold1500_rare375_v1.hamiltonian.npz"
)
DEFAULT_PF_DELTA = Path("artifacts/pf_delta_validation/h4_sto3g_d100_rank12_ld3_v5.json")
DEFAULT_ALLOCATION = Path(
    "artifacts/rpe_allocation_sensitivity_validation/2026-09-01/"
    "h4_sto3g_d100_rank12_ld3_dt0p1_r4_k2_q1_q2_q4_q8_"
    "beta_alpha_sensitivity_v1.json"
)
DEFAULT_HORIZON = Path(
    "artifacts/rpe_target_round_horizon_validation/2026-09-20/"
    "h4_sto3g_d100_rank12_ld3_dt0p1_r4_k2_ca_over_10_horizon_v1.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/rpe_delta_round_schedule_validation/2026-09-20/"
    "h4_sto3g_d100_rank12_ld3_executed_delta_ca_over_10_"
    "round_schedule_v1.json"
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
    parser.add_argument("--pf-delta", type=Path, default=DEFAULT_PF_DELTA)
    parser.add_argument("--allocation", type=Path, default=DEFAULT_ALLOCATION)
    parser.add_argument("--horizon", type=Path, default=DEFAULT_HORIZON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    hamiltonian = load_connected_cluster_hamiltonian_snapshot(args.snapshot)
    pf_delta = json.loads(args.pf_delta.read_text(encoding="utf-8"))
    allocation = json.loads(args.allocation.read_text(encoding="utf-8"))
    horizon = json.loads(args.horizon.read_text(encoding="utf-8"))
    validate_rpe_allocation_sensitivity_payload(allocation)
    validate_rpe_target_round_horizon_payload(horizon)

    allocation_config = allocation["configuration"]
    selected_beta_label = allocation["selection"]["selected_beta_profile"]
    selected_beta = next(
        item
        for item in allocation_config["beta_profiles"]
        if item["label"] == selected_beta_label
    )
    horizon_summary = horizon["summary"]
    pf_request = pf_delta["request"]
    calibration_deltas = tuple(pf_request["surrogate_calibration_times"])
    validation_deltas = tuple(pf_request["validation_delta_times"])
    sources = (
        Path("src/trotterlib/rpe_delta_round_schedule_validation.py"),
        Path("scripts/run_rpe_delta_round_schedule_validation.py"),
    )
    payload = validate_rpe_delta_round_schedule(
        hamiltonian,
        PhysicalSector.number_sector(
            n_qubits=hamiltonian.n_qubits,
            n_electrons=int(horizon["system"]["sector_n_electrons"]),
        ),
        ld=int(horizon["system"]["ld"]),
        target_energy_precision=float(
            horizon_summary["provisional_target_energy_precision_ha"]
        ),
        delta_candidates=tuple(sorted({*calibration_deltas, *validation_deltas})),
        calibration_delta_values=calibration_deltas,
        disjoint_validation_delta_values=validation_deltas,
        pf_coefficient=float(allocation_config["pf_coefficient"]),
        pf_coefficient_source=str(allocation_config["pf_coefficient_source"]),
        beta_rpe=float(allocation_config["beta_rpe"]),
        beta_pf_budget=float(selected_beta["beta_pf"]),
        beta_rte_budget=float(selected_beta["beta_rte"]),
        beta_stat_budget=float(selected_beta["beta_stat"]),
        alpha_total=float(allocation_config["alpha_total"]),
        provenance={
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git(["rev-parse", "HEAD"]),
            "git_worktree_status_before_generation": _git(["status", "--short"]),
            "evidence_status": "local_worktree_validation_not_immutable_ci",
            "command": shlex.join(
                [
                    ".venv311/bin/python",
                    "scripts/run_rpe_delta_round_schedule_validation.py",
                    *sys.argv[1:],
                ]
            ),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "qiskit_version": qiskit.__version__,
            "inputs": {
                "snapshot": {
                    "path": str(args.snapshot),
                    "sha256": _sha256(args.snapshot),
                },
                "pf_delta": {
                    "path": str(args.pf_delta),
                    "sha256": _sha256(args.pf_delta),
                    "validation_fingerprint": pf_delta["validation_fingerprint"],
                },
                "allocation": {
                    "path": str(args.allocation),
                    "sha256": _sha256(args.allocation),
                    "content_fingerprint": allocation["content_fingerprint"],
                },
                "horizon": {
                    "path": str(args.horizon),
                    "sha256": _sha256(args.horizon),
                    "content_fingerprint": horizon["content_fingerprint"],
                },
            },
            "source_sha256": {str(path): _sha256(path) for path in sources},
        },
    )
    write_rpe_delta_round_schedule_validation(payload, args.output)
    summary = payload["summary"]
    print(
        f"wrote {args.output}; overall_pass={summary['overall_pass']}; "
        f"pf_pass={summary['pf_screen_passing_deltas']}; "
        f"proxy_best_delta={summary['proxy_best_delta']}; "
        f"near_tie={summary['near_tie_deltas_within_provisional_5_percent']}; "
        f"min_radius={summary['minimum_matrix_observed_radius_over_selected_schedules']:.12g}; "
        f"elapsed={payload['performance']['elapsed_seconds']:.3f}s"
    )
    return 0 if summary["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
