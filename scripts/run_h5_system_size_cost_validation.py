#!/usr/bin/env python3
"""Run the resumable H5 paired connected-cost system-size validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import qiskit

from trotterlib.df_hamiltonian import build_df_h_d_from_molecule
from trotterlib.rte import CompilerSettings
from trotterlib.rte_connected_cluster_cost_validation import (
    load_connected_cluster_hamiltonian_snapshot,
    write_connected_cluster_hamiltonian_snapshot,
)
from trotterlib.rte_system_size_cost_validation import (
    validate_system_size_paired_cluster_models,
    validate_system_size_paired_payload,
    write_system_size_paired_validation,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/rte_cost_system_size_h5/2026-08-25"),
    )
    parser.add_argument("--common-samples", type=int, default=100)
    parser.add_argument("--single-order2-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--maximum-workers", type=int, default=3)
    return parser


def main() -> int:
    args = _parser().parse_args()
    root = args.root.resolve()
    outputs = root / "outputs"
    checkpoints = root / "checkpoints"
    cache = root / "cache"
    snapshots = root / "snapshots"
    for directory in (outputs, checkpoints, cache, snapshots):
        directory.mkdir(parents=True, exist_ok=True)
    for marker in (root / "DONE", root / "FAILED"):
        marker.unlink(missing_ok=True)

    snapshot_path = snapshots / "h5_sto3g_d100_rank9_df_snapshot_v1.npz"
    output = outputs / (
        "h5_sto3g_d100_rank9_ld4_s0025_k2_paired_"
        f"l4_l6_l8_zero{args.common_samples}_single{args.single_order2_samples}_v1.json"
    )
    command = shlex.join(
        [
            ".venv311/bin/python",
            "scripts/run_h5_system_size_cost_validation.py",
            *sys.argv[1:],
        ]
    )
    total_pattern_tasks = 2 * sum(length + 1 for length in (4, 6, 8))
    plan = {
        "schema_version": "h5_system_size_cost_plan_v1",
        "created_at_utc": _utc_now(),
        "working_directory": str(Path.cwd()),
        "command": command,
        "output": str(output),
        "hamiltonian_snapshot": str(snapshot_path),
        "checkpoint_directory": str(checkpoints),
        "persistent_cache": str(cache / "compiled_cost.sqlite"),
        "configuration": {
            "molecule": "H5 chain",
            "distance_angstrom": 1.0,
            "basis": "STO-3G",
            "df_rank_policy": "project_config_selected_rank_expected_9",
            "ld": 4,
            "reference_delta_time": 0.1,
            "reference_rte_steps": 4,
            "short_step_time": 0.025,
            "finite_taylor_order": 2,
            "sequence_lengths": [4, 6, 8],
            "cluster_lengths": [3, 4],
            "common_samples": args.common_samples,
            "single_order2_samples_per_position": args.single_order2_samples,
            "maximum_workers": args.maximum_workers,
            "seed": args.seed,
        },
        "total_pattern_tasks": total_pattern_tasks,
    }
    _write_json_atomic(root / "plan.json", plan)
    state = {
        "schema_version": "h5_system_size_cost_status_v1",
        "pid": os.getpid(),
        "status": "running",
        "stage": "preparing_h5_snapshot",
        "started_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
        "output": str(output),
        "completed_pattern_checkpoints": len(list(checkpoints.glob("*.json"))),
        "total_pattern_checkpoints": total_pattern_tasks,
    }
    _write_json_atomic(root / "status.json", state)
    started = time.perf_counter()
    try:
        if snapshot_path.exists():
            hamiltonian = load_connected_cluster_hamiltonian_snapshot(snapshot_path)
            reused_snapshot = True
        else:
            hamiltonian, _sector = build_df_h_d_from_molecule(
                5, distance=1.0, basis="sto-3g"
            )
            if hamiltonian.n_qubits != 10 or hamiltonian.n_blocks != 9:
                raise RuntimeError(
                    "The configured H5 system is not the expected 10-qubit rank-9 model."
                )
            write_connected_cluster_hamiltonian_snapshot(hamiltonian, snapshot_path)
            reused_snapshot = False
        if hamiltonian.n_qubits != 10 or hamiltonian.n_blocks != 9:
            raise RuntimeError("The saved H5 snapshot has unexpected dimensions.")

        state.update(
            {
                "stage": "paired_k1_k3_then_k1_k4",
                "updated_at_utc": _utc_now(),
                "reused_snapshot": reused_snapshot,
                "hamiltonian_snapshot_sha256": _sha256(snapshot_path),
            }
        )
        _write_json_atomic(root / "status.json", state)
        if output.exists():
            payload = json.loads(output.read_text(encoding="utf-8"))
            validate_system_size_paired_payload(payload)
            reused_output = True
        else:
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
                Path("src/trotterlib/rte_order_stratified_cost_validation.py"),
                Path("src/trotterlib/rte_system_size_cost_validation.py"),
                Path("src/trotterlib/rte_compiled_cost.py"),
                Path("scripts/run_h5_system_size_cost_validation.py"),
            )
            provenance = {
                "generated_at_utc": _utc_now(),
                "git_commit": _git_head(),
                "git_worktree_status_before_generation": _git_status(),
                "evidence_status": "local_worktree_validation_not_immutable_ci",
                "command": command,
                "python_version": platform.python_version(),
                "qiskit_version": qiskit.__version__,
                "platform": platform.platform(),
                "source_sha256": {str(path): _sha256(path) for path in sources},
                "hamiltonian_snapshot": str(snapshot_path),
                "hamiltonian_snapshot_sha256": _sha256(snapshot_path),
            }
            payload = validate_system_size_paired_cluster_models(
                hamiltonian,
                ld=4,
                reference_delta_time=0.1,
                reference_rte_steps=4,
                compiler=compiler,
                common_sample_count=args.common_samples,
                single_rare_sample_count=args.single_order2_samples,
                seed=args.seed,
                sequence_lengths=(4, 6, 8),
                cluster_lengths=(3, 4),
                maximum_workers=args.maximum_workers,
                persistent_cache_path=cache / "compiled_cost.sqlite",
                checkpoint_directory=checkpoints,
                provenance=provenance,
            )
            write_system_size_paired_validation(payload, output)
            reused_output = False
        state.update(
            {
                "status": "completed",
                "stage": "completed",
                "finished_at_utc": _utc_now(),
                "updated_at_utc": _utc_now(),
                "elapsed_seconds": float(time.perf_counter() - started),
                "reused_output": reused_output,
                "completed_pattern_checkpoints": len(list(checkpoints.glob("*.json"))),
                "model_summaries": payload["model_summaries"],
                "decision": payload["decision"],
            }
        )
        _write_json_atomic(root / "status.json", state)
        (root / "DONE").write_text(_utc_now() + "\n", encoding="utf-8")
        print(output, flush=True)
        print(payload["model_summaries"], flush=True)
        print(payload["decision"], flush=True)
        return 0
    except BaseException as exc:
        state.update(
            {
                "status": "failed",
                "stage": "failed",
                "finished_at_utc": _utc_now(),
                "updated_at_utc": _utc_now(),
                "elapsed_seconds": float(time.perf_counter() - started),
                "completed_pattern_checkpoints": len(list(checkpoints.glob("*.json"))),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        _write_json_atomic(root / "status.json", state)
        (root / "FAILED").write_text(_utc_now() + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
