#!/usr/bin/env python3
"""Run the detached compiled-cost data-collection queue with durable status."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from trotterlib.hierarchical_cost_validation import (
    validate_hierarchical_cost_payload,
)
from trotterlib.rte_connected_cluster_cost_validation import (
    validate_connected_cluster_k4_calibration_payload,
)
from trotterlib.rte_order_stratified_cost_validation import (
    validate_order_stratified_cost_payload,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        default=Path("artifacts/rte_cost_data_batch/2026-08-25"),
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
        "--ld6-transfer",
        type=Path,
        default=Path(
            "artifacts/rte_connected_cluster_cost_validation/"
            "h4_ld6_s0025_transfer_l4_l6_zero1500_rare500_seed20260912_v2.json"
        ),
    )
    return parser


def _validate_output(path: Path, validator: Callable[[Any], None]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    validator(payload)


def main() -> int:
    args = _parser().parse_args()
    root = args.root.resolve()
    outputs = root / "outputs"
    logs = root / "logs"
    checkpoints = root / "checkpoints"
    cache = root / "cache"
    for directory in (outputs, logs, checkpoints, cache):
        directory.mkdir(parents=True, exist_ok=True)
    for marker in (root / "DONE", root / "FAILED"):
        marker.unlink(missing_ok=True)
    snapshot = args.hamiltonian_snapshot.resolve()
    transfer = args.ld6_transfer.resolve()
    if not snapshot.exists() or not transfer.exists():
        raise FileNotFoundError("The fixed snapshot or LD6 transfer is missing.")

    # Keep the virtual-environment launcher path.  Resolving its symlink would
    # replace it with the system interpreter and lose the venv site-packages.
    python = sys.executable
    output_a = outputs / (
        "h4_sto3g_d100_rank12_ld0_dt0p1_ref4_k2_independent_"
        "ncommon50_nsingle50_nmulti20_v1.json"
    )
    output_b = outputs / "h4_ld6_s0025_independent_k4_n30_v1.json"
    output_c = outputs / "h4_ld3_dt0p1_ref4_k0_controlled_q8_n150_v1.json"
    jobs = {
        "batch_a_multi_order2": {
            "phase": 1,
            "output": output_a,
            "validator": validate_order_stratified_cost_payload,
            "command": [
                python,
                "scripts/run_rte_order_stratified_cost_validation.py",
                "--molecule",
                "4",
                "--distance",
                "1.0",
                "--basis",
                "sto-3g",
                "--df-rank",
                "12",
                "--ld",
                "0",
                "--reference-delta-time",
                "0.1",
                "--reference-rte-steps",
                "4",
                "--common-samples",
                "50",
                "--single-order2-samples",
                "50",
                "--multi-order2-samples",
                "20",
                "--seed",
                "20261001",
                "--maximum-workers",
                "3",
                "--hamiltonian-snapshot",
                str(snapshot),
                "--output-directory",
                str(outputs),
            ],
        },
        "batch_b_independent_k4": {
            "phase": 1,
            "output": output_b,
            "validator": validate_connected_cluster_k4_calibration_payload,
            "command": [
                python,
                "scripts/run_rte_connected_cluster_k4_calibration.py",
                str(transfer),
                str(snapshot),
                "--samples",
                "30",
                "--seed",
                "20261002",
                "--maximum-workers",
                "3",
                "--sample-chunk-size",
                "15",
                "--checkpoint-directory",
                str(checkpoints / "batch_b_k4"),
                "--persistent-cache",
                str(cache / "batch_b_k4.sqlite"),
                "--output",
                str(output_b),
            ],
        },
        "batch_c_controlled_q8": {
            "phase": 2,
            "output": output_c,
            "validator": validate_hierarchical_cost_payload,
            "command": [
                python,
                "scripts/run_controlled_repetition_holdout.py",
                str(snapshot),
                "--ld",
                "3",
                "--reference-delta-time",
                "0.1",
                "--reference-rte-steps",
                "4",
                "--finite-taylor-order",
                "0",
                "--calibration-samples",
                "150",
                "--holdout-samples",
                "150",
                "--holdout-q",
                "8",
                "--seed",
                "20261003",
                "--output",
                str(output_c),
            ],
        },
    }
    plan = {
        "schema_version": "rte_cost_data_batch_plan_v1",
        "created_at_utc": _utc_now(),
        "working_directory": str(Path.cwd()),
        "hamiltonian_snapshot": str(snapshot),
        "ld6_transfer": str(transfer),
        "jobs": {
            name: {
                "phase": job["phase"],
                "output": str(job["output"]),
                "log": str(logs / f"{name}.log"),
                "command": job["command"],
            }
            for name, job in jobs.items()
        },
    }
    _write_json_atomic(root / "plan.json", plan)
    state = {
        "schema_version": "rte_cost_data_batch_status_v1",
        "batch_pid": os.getpid(),
        "started_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
        "status": "running",
        "jobs": {
            name: {"status": "pending", "output": str(job["output"])}
            for name, job in jobs.items()
        },
    }
    _write_json_atomic(root / "status.json", state)

    failures = []
    for phase in (1, 2):
        running: dict[str, tuple[subprocess.Popen[Any], Any, float]] = {}
        for name, job in jobs.items():
            if job["phase"] != phase:
                continue
            output = job["output"]
            try:
                if output.exists():
                    _validate_output(output, job["validator"])
                    state["jobs"][name].update(
                        {"status": "reused", "finished_at_utc": _utc_now()}
                    )
                    continue
            except Exception:
                pass
            log_handle = (logs / f"{name}.log").open("a", encoding="utf-8")
            log_handle.write(f"\n[{_utc_now()}] starting\n")
            log_handle.flush()
            process = subprocess.Popen(
                job["command"],
                cwd=Path.cwd(),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            running[name] = (process, log_handle, time.perf_counter())
            state["jobs"][name].update(
                {
                    "status": "running",
                    "pid": process.pid,
                    "started_at_utc": _utc_now(),
                }
            )
        state["updated_at_utc"] = _utc_now()
        _write_json_atomic(root / "status.json", state)
        while running:
            for name, (process, log_handle, started) in tuple(running.items()):
                returncode = process.poll()
                if returncode is None:
                    continue
                log_handle.write(
                    f"[{_utc_now()}] finished returncode={returncode}\n"
                )
                log_handle.close()
                job = jobs[name]
                try:
                    if returncode != 0:
                        raise RuntimeError(f"subprocess exit code {returncode}")
                    _validate_output(job["output"], job["validator"])
                    status = "completed"
                    error = None
                except Exception as exc:
                    status = "failed"
                    error = f"{type(exc).__name__}: {exc}"
                    failures.append(name)
                state["jobs"][name].update(
                    {
                        "status": status,
                        "returncode": returncode,
                        "finished_at_utc": _utc_now(),
                        "elapsed_seconds": time.perf_counter() - started,
                        "error": error,
                    }
                )
                running.pop(name)
            state["updated_at_utc"] = _utc_now()
            _write_json_atomic(root / "status.json", state)
            if running:
                time.sleep(2.0)

    state["status"] = "failed" if failures else "completed"
    state["finished_at_utc"] = _utc_now()
    state["updated_at_utc"] = _utc_now()
    _write_json_atomic(root / "status.json", state)
    marker = root / ("FAILED" if failures else "DONE")
    marker.write_text(
        ("failed jobs: " + ", ".join(failures) if failures else "completed")
        + "\n",
        encoding="utf-8",
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
