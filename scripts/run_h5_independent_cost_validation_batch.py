#!/usr/bin/env python3
"""Run resumable H5 independent calibration and full-circuit holdout."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from trotterlib.rte_connected_cluster_cost_validation import (
    validate_connected_cluster_calibration_payload,
    validate_connected_cluster_transfer_payload,
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


def _validate_output(path: Path, validator: Callable[[Any], None]) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    validator(payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/rte_cost_system_size_h5_independent/2026-08-26"),
    )
    parser.add_argument(
        "--hamiltonian-snapshot",
        type=Path,
        default=Path(
            "artifacts/rte_cost_system_size_h5/2026-08-25/snapshots/"
            "h5_sto3g_d100_rank9_df_snapshot_v1.npz"
        ),
    )
    parser.add_argument("--calibration-seed", type=int, default=20261015)
    parser.add_argument("--holdout-seed", type=int, default=20261016)
    parser.add_argument("--maximum-workers", type=int, default=3)
    parser.add_argument("--maximum-production-samples", type=int, default=2000)
    parser.add_argument("--holdout-zero-samples", type=int, default=500)
    parser.add_argument("--holdout-single-order2-samples", type=int, default=125)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.calibration_seed == args.holdout_seed:
        raise ValueError("Calibration and holdout seeds must differ.")
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
    if not snapshot.exists():
        raise FileNotFoundError(snapshot)
    calibration = outputs / (
        "h5_sto3g_d100_rank9_ld4_dt0p1_ref4_k2_connected_"
        f"pilot30_max{args.maximum_production_samples}_calibration_v2.json"
    )
    holdout = outputs / (
        "h5_sto3g_d100_rank9_ld4_s0025_k2_independent_"
        f"l4_l6_l8_zero{args.holdout_zero_samples}_"
        f"single{args.holdout_single_order2_samples}_v2.json"
    )
    python = sys.executable
    shared_cache = cache / "compiled_cost.sqlite"
    jobs = [
        {
            "name": "independent_calibration_k1_k3",
            "output": calibration,
            "validator": validate_connected_cluster_calibration_payload,
            "command": [
                python,
                "scripts/run_rte_connected_cluster_cost_validation.py",
                "--mode",
                "calibration",
                "--molecule",
                "5",
                "--distance",
                "1.0",
                "--basis",
                "sto-3g",
                "--df-rank",
                "9",
                "--ld",
                "4",
                "--reference-delta-time",
                "0.1",
                "--reference-rte-steps",
                "4",
                "--pilot-samples",
                "30",
                "--minimum-production-samples",
                "30",
                "--maximum-production-samples",
                str(args.maximum_production_samples),
                "--prediction-relative-se-target",
                "0.01",
                "--allocation-safety-factor",
                "1.5",
                "--target-event-counts",
                "4,6,8",
                "--seed",
                str(args.calibration_seed),
                "--maximum-workers",
                str(args.maximum_workers),
                "--sample-chunk-size",
                "128",
                "--adaptive-production-rounds",
                "2",
                "--checkpoint-directory",
                str(checkpoints / "calibration"),
                "--persistent-cache",
                str(shared_cache),
                "--hamiltonian-snapshot",
                str(snapshot),
                "--output-directory",
                str(outputs),
            ],
        },
        {
            "name": "independent_full_holdout",
            "output": holdout,
            "validator": validate_connected_cluster_transfer_payload,
            "command": [
                python,
                "scripts/run_rte_connected_cluster_transfer_validation.py",
                str(calibration),
                str(snapshot),
                "--holdout-lengths",
                "4,6,8",
                "--holdout-zero-samples",
                str(args.holdout_zero_samples),
                "--holdout-single-order2-samples",
                str(args.holdout_single_order2_samples),
                "--seed",
                str(args.holdout_seed),
                "--maximum-workers",
                str(args.maximum_workers),
                "--sample-chunk-size",
                "128",
                "--checkpoint-directory",
                str(checkpoints / "holdout"),
                "--persistent-cache",
                str(shared_cache),
                "--output",
                str(holdout),
            ],
        },
    ]
    plan = {
        "schema_version": "h5_independent_cost_batch_plan_v1",
        "created_at_utc": _utc_now(),
        "working_directory": str(Path.cwd()),
        "hamiltonian_snapshot": str(snapshot),
        "acceptance_policy": {
            "primary_metric": "rz_count",
            "maximum_absolute_relative_point_error": 0.05,
            "maximum_prediction_relative_95_half_width": 0.02,
        },
        "jobs": [
            {
                "name": job["name"],
                "output": str(job["output"]),
                "log": str(logs / f"{job['name']}.log"),
                "command": job["command"],
            }
            for job in jobs
        ],
    }
    _write_json_atomic(root / "plan.json", plan)
    state = {
        "schema_version": "h5_independent_cost_batch_status_v1",
        "batch_pid": os.getpid(),
        "status": "running",
        "started_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
        "current_job": jobs[0]["name"],
        "jobs": {
            job["name"]: {"status": "pending", "output": str(job["output"])}
            for job in jobs
        },
    }
    _write_json_atomic(root / "status.json", state)
    batch_started = time.perf_counter()
    try:
        final_payload = None
        for job in jobs:
            name = job["name"]
            output = job["output"]
            state["current_job"] = name
            state["updated_at_utc"] = _utc_now()
            job_state = state["jobs"][name]
            try:
                payload = _validate_output(output, job["validator"])
                job_state.update(
                    {
                        "status": "reused",
                        "finished_at_utc": _utc_now(),
                        "elapsed_seconds": 0.0,
                    }
                )
                _write_json_atomic(root / "status.json", state)
                final_payload = payload
                continue
            except (FileNotFoundError, json.JSONDecodeError, ValueError):
                pass
            log_path = logs / f"{name}.log"
            job_state.update({"status": "running", "started_at_utc": _utc_now()})
            _write_json_atomic(root / "status.json", state)
            job_started = time.perf_counter()
            with log_path.open("a", encoding="utf-8") as log_handle:
                log_handle.write(f"\n[{_utc_now()}] starting\n")
                log_handle.flush()
                completed = subprocess.run(
                    job["command"],
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                log_handle.write(
                    f"[{_utc_now()}] finished returncode={completed.returncode}\n"
                )
            if completed.returncode != 0:
                raise RuntimeError(f"{name} failed with return code {completed.returncode}")
            payload = _validate_output(output, job["validator"])
            job_state.update(
                {
                    "status": "completed",
                    "finished_at_utc": _utc_now(),
                    "elapsed_seconds": float(time.perf_counter() - job_started),
                }
            )
            if name == "independent_full_holdout":
                job_state["summary"] = payload["summary"]
            else:
                job_state["maximum_realized_rz_relative_standard_error"] = (
                    payload["allocation"]["adaptive_history"][-1][
                        "maximum_realized_rz_relative_standard_error"
                    ]
                )
                job_state["adaptive_stop_reason"] = payload["allocation"][
                    "adaptive_stop_reason"
                ]
            state["updated_at_utc"] = _utc_now()
            _write_json_atomic(root / "status.json", state)
            final_payload = payload

        if final_payload is None:
            raise RuntimeError("The H5 independent batch produced no payload.")
        final_summary = final_payload["summary"]
        accepted = bool(
            final_summary["primary_point_tolerance_passed"]
            and final_summary["primary_prediction_precision_passed"]
        )
        state.update(
            {
                "status": "completed",
                "current_job": None,
                "finished_at_utc": _utc_now(),
                "updated_at_utc": _utc_now(),
                "elapsed_seconds": float(time.perf_counter() - batch_started),
                "decision": {
                    "independent_k1_k3_accepted": accepted,
                    "point_tolerance_passed": final_summary[
                        "primary_point_tolerance_passed"
                    ],
                    "prediction_precision_passed": final_summary[
                        "primary_prediction_precision_passed"
                    ],
                },
            }
        )
        _write_json_atomic(root / "status.json", state)
        (root / "DONE").write_text(_utc_now() + "\n", encoding="utf-8")
        print(holdout, flush=True)
        print(final_summary, flush=True)
        print(state["decision"], flush=True)
        return 0
    except BaseException as exc:
        state.update(
            {
                "status": "failed",
                "finished_at_utc": _utc_now(),
                "updated_at_utc": _utc_now(),
                "elapsed_seconds": float(time.perf_counter() - batch_started),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        _write_json_atomic(root / "status.json", state)
        (root / "FAILED").write_text(_utc_now() + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
