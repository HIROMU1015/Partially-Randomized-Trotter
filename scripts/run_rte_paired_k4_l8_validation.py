#!/usr/bin/env python3
"""Run the resumable paired L=8 K1--K4 residual validation."""

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

from trotterlib.rte import CompilerSettings
from trotterlib.rte_connected_cluster_cost_validation import (
    load_connected_cluster_hamiltonian_snapshot,
    validate_connected_cluster_transfer_payload,
)
from trotterlib.rte_order_stratified_cost_validation import (
    validate_paired_k4_l8_payload,
    validate_paired_k4_l8_residual,
    write_paired_k4_l8_validation,
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
        default=Path("artifacts/rte_cost_paired_k4_l8/2026-08-25"),
    )
    parser.add_argument(
        "--hamiltonian-snapshot",
        type=Path,
        default=Path(
            "artifacts/rte_connected_cluster_cost_validation/"
            "h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k2_connected_"
            "pilot30_max1500_hold1500_rare375_v1.hamiltonian.npz"
        ),
    )
    parser.add_argument(
        "--transfer",
        type=Path,
        default=Path(
            "artifacts/rte_connected_cluster_cost_validation/"
            "h4_ld6_s0025_transfer_l4_l6_zero1500_rare500_seed20260912_v2.json"
        ),
    )
    parser.add_argument("--common-samples", type=int, default=100)
    parser.add_argument("--single-order2-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--maximum-workers", type=int, default=3)
    return parser


def main() -> int:
    args = _parser().parse_args()
    root = args.root.resolve()
    outputs = root / "outputs"
    checkpoints = root / "checkpoints"
    cache = root / "cache"
    for directory in (outputs, checkpoints, cache):
        directory.mkdir(parents=True, exist_ok=True)
    for marker in (root / "DONE", root / "FAILED"):
        marker.unlink(missing_ok=True)

    snapshot_path = args.hamiltonian_snapshot.resolve()
    transfer_path = args.transfer.resolve()
    transfer = json.loads(transfer_path.read_text(encoding="utf-8"))
    validate_connected_cluster_transfer_payload(transfer)
    hamiltonian = load_connected_cluster_hamiltonian_snapshot(snapshot_path)
    condition = transfer["calibration_reference"]["condition"]
    raw = condition["compiler"]
    compiler = CompilerSettings(
        basis_gates=tuple(raw["basis_gates"]),
        backend_name=raw["backend_name"],
        coupling_map=(
            None
            if raw["coupling_map"] is None
            else tuple(tuple(edge) for edge in raw["coupling_map"])
        ),
        optimization_level=int(raw["optimization_level"]),
        layout_method=raw["layout_method"],
        routing_method=raw["routing_method"],
        transpiler_seed=int(raw["transpiler_seed"]),
        qiskit_version=raw["qiskit_version"],
    )
    reference_rte_steps = 4
    reference_delta_time = float(condition["short_step_time"]) * reference_rte_steps
    output = outputs / (
        "h4_ld6_s0025_k2_paired_k1_k4_l8_"
        f"zero{args.common_samples}_single{args.single_order2_samples}_v1.json"
    )
    command = shlex.join(
        [
            ".venv311/bin/python",
            "scripts/run_rte_paired_k4_l8_validation.py",
            *sys.argv[1:],
        ]
    )
    plan = {
        "schema_version": "rte_paired_k4_l8_plan_v1",
        "created_at_utc": _utc_now(),
        "working_directory": str(Path.cwd()),
        "command": command,
        "output": str(output),
        "checkpoint_directory": str(checkpoints),
        "persistent_cache": str(cache / "compiled_cost.sqlite"),
        "configuration": {
            "ld": int(condition["ld"]),
            "short_step_time": float(condition["short_step_time"]),
            "finite_taylor_order": 2,
            "sequence_length": 8,
            "maximum_cluster_length": 4,
            "common_samples": args.common_samples,
            "single_order2_samples_per_position": args.single_order2_samples,
            "maximum_workers": args.maximum_workers,
            "seed": args.seed,
        },
    }
    _write_json_atomic(root / "plan.json", plan)
    state = {
        "schema_version": "rte_paired_k4_l8_status_v1",
        "pid": os.getpid(),
        "status": "running",
        "started_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
        "output": str(output),
        "completed_pattern_checkpoints": len(list(checkpoints.glob("pattern_*.json"))),
        "total_pattern_checkpoints": 9,
    }
    _write_json_atomic(root / "status.json", state)
    started = time.perf_counter()
    try:
        if output.exists():
            existing = json.loads(output.read_text(encoding="utf-8"))
            validate_paired_k4_l8_payload(existing)
            payload = existing
            reused_output = True
        else:
            sources = (
                Path("src/trotterlib/rte_order_stratified_cost_validation.py"),
                Path("src/trotterlib/rte_compiled_cost.py"),
                Path("scripts/run_rte_paired_k4_l8_validation.py"),
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
                "source_transfer": str(transfer_path),
                "source_transfer_sha256": _sha256(transfer_path),
                "hamiltonian_snapshot": str(snapshot_path),
                "hamiltonian_snapshot_sha256": _sha256(snapshot_path),
            }
            payload = validate_paired_k4_l8_residual(
                hamiltonian,
                ld=int(condition["ld"]),
                reference_delta_time=reference_delta_time,
                reference_rte_steps=reference_rte_steps,
                compiler=compiler,
                common_sample_count=args.common_samples,
                single_rare_sample_count=args.single_order2_samples,
                seed=args.seed,
                maximum_workers=args.maximum_workers,
                persistent_cache_path=cache / "compiled_cost.sqlite",
                checkpoint_directory=checkpoints,
                provenance=provenance,
            )
            write_paired_k4_l8_validation(payload, output)
            reused_output = False
        state.update(
            {
                "status": "completed",
                "finished_at_utc": _utc_now(),
                "updated_at_utc": _utc_now(),
                "elapsed_seconds": float(time.perf_counter() - started),
                "reused_output": reused_output,
                "completed_pattern_checkpoints": len(
                    list(checkpoints.glob("pattern_*.json"))
                ),
                "summary": payload["summary"],
            }
        )
        _write_json_atomic(root / "status.json", state)
        (root / "DONE").write_text(_utc_now() + "\n", encoding="utf-8")
        print(output, flush=True)
        print(payload["summary"], flush=True)
        return 0
    except BaseException as exc:
        state.update(
            {
                "status": "failed",
                "finished_at_utc": _utc_now(),
                "updated_at_utc": _utc_now(),
                "elapsed_seconds": float(time.perf_counter() - started),
                "completed_pattern_checkpoints": len(
                    list(checkpoints.glob("pattern_*.json"))
                ),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        _write_json_atomic(root / "status.json", state)
        (root / "FAILED").write_text(_utc_now() + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
