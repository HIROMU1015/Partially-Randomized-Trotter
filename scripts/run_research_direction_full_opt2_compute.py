#!/usr/bin/env python3
"""Prepare and execute resumable M06-F optimization-level-2 cell tasks."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import psutil
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from trotterlib.parallel_validation_executor import (
    ParallelValidationExecutor,
    atomic_write_json,
    dry_run_report,
    inspect_batch_status,
    load_task_manifest,
    plan_resources,
)
from trotterlib.research_direction_full_opt2 import (
    THREAD_ENVIRONMENT,
    create_full_opt2_task_manifest,
    manifest_workload_summary,
)


def _git(*args: str) -> str | list[str] | None:
    result = subprocess.run(
        ["git", *args], check=False, capture_output=True, text=True
    )
    if result.returncode != 0:
        return None
    lines = result.stdout.splitlines()
    return lines[0] if len(lines) == 1 else lines


def _parse_cell(value: str) -> tuple[float, int, int]:
    try:
        delta_text, r_text, q_text = value.split(",")
        return float(delta_text), int(r_text), int(q_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "cell must be DELTA,R,Q (for example 0.01,1,1)"
        ) from exc


def _scheduler_record() -> dict[str, object]:
    commands = {
        name: shutil.which(name)
        for name in ("sbatch", "srun", "qsub", "qstat", "bsub", "bjobs")
    }
    allocation = {
        name: os.environ.get(name)
        for name in ("SLURM_JOB_ID", "PBS_JOBID", "LSB_JOBID")
        if os.environ.get(name)
    }
    return {
        "detected_commands": {key: value for key, value in commands.items() if value},
        "active_allocation": allocation,
    }


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _unique_hashes(manifest: dict[str, object], key: str) -> dict[str, str]:
    merged: dict[str, str] = {}
    for task in manifest["tasks"]:
        for path, digest in task[key].items():
            existing = merged.setdefault(path, digest)
            if existing != digest:
                raise RuntimeError(f"Conflicting {key} value for {path}")
    return dict(sorted(merged.items()))


def _require_runtime_environment() -> dict[str, str]:
    values = {name: os.environ.get(name, "") for name in THREAD_ENVIRONMENT}
    if any(value != "1" for value in values.values()):
        raise SystemExit(
            "OMP/OPENBLAS/MKL/NUMEXPR/NUMBA thread variables must all equal 1."
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise SystemExit("CUDA_VISIBLE_DEVICES must be explicitly empty.")
    return values


def _write_runtime_provenance(
    *,
    manifest: dict[str, object],
    output_dir: Path,
    resource_plan: object,
) -> None:
    path = output_dir / "run_provenance.json"
    if path.exists():
        raise RuntimeError(f"Refusing to replace runtime provenance: {path}")
    thread_environment = _require_runtime_environment()
    payload = {
        "schema_version": "research_direction_full_opt2_run_provenance_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_status": _git("status", "--short"),
        "git_branch": _git("branch", "--show-current"),
        "command": shlex.join([sys.executable, *sys.argv]),
        "python_version": platform.python_version(),
        "qiskit_version": importlib.metadata.version("qiskit"),
        "numpy_version": importlib.metadata.version("numpy"),
        "scipy_version": importlib.metadata.version("scipy"),
        "cpu": {
            "model": _cpu_model(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
        },
        "memory": {
            "total_bytes": psutil.virtual_memory().total,
            "available_bytes_at_start": psutil.virtual_memory().available,
        },
        "scheduler": _scheduler_record(),
        "resource_plan": resource_plan.to_dict(),
        "thread_environment": thread_environment,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tmpdir": os.environ.get("TMPDIR"),
        "mplconfigdir": os.environ.get("MPLCONFIGDIR"),
        "numba_cache_dir": os.environ.get("NUMBA_CACHE_DIR"),
        "manifest_path": manifest["manifest_path"],
        "manifest_fingerprint": manifest["manifest_fingerprint"],
        "workload": manifest_workload_summary(manifest),
        "input_sha256": _unique_hashes(manifest, "input_sha256"),
        "source_sha256": _unique_hashes(manifest, "source_sha256"),
        "evidence_status": "local_dirty_worktree_compute_in_progress",
        "scientific_verdict_included": False,
    }
    atomic_write_json(path, payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--create-manifest", type=Path)
    parser.add_argument("--task-manifest", type=Path)
    parser.add_argument("--batch-id", default="wp11-all-r-opt2-initial")
    parser.add_argument(
        "--cell",
        type=_parse_cell,
        action="append",
        help="Limit a newly created manifest to DELTA,R,Q; repeatable.",
    )
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--exclude-deterministic", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--dry-run", action="store_true")
    actions.add_argument("--run", action="store_true")
    actions.add_argument("--status", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--memory-budget-gib", type=float, default=32.0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    manifest_path = args.task_manifest
    if args.create_manifest is not None:
        manifest = create_full_opt2_task_manifest(
            args.create_manifest,
            project_root=project_root,
            batch_id=args.batch_id,
            initial_cells=args.cell,
            initial_sample_count=args.samples,
            include_deterministic=not args.exclude_deterministic,
        )
        manifest_path = args.create_manifest
        print(
            json.dumps(
                {
                    "manifest_path": str(args.create_manifest),
                    "manifest_fingerprint": manifest["manifest_fingerprint"],
                    "workload": manifest_workload_summary(manifest),
                },
                indent=2,
            ),
            flush=True,
        )
        if not (args.dry_run or args.run or args.status):
            return 0
    if manifest_path is None:
        raise SystemExit("--task-manifest or --create-manifest is required.")
    manifest = load_task_manifest(manifest_path, project_root=project_root)
    if args.output_dir is None:
        raise SystemExit("--output-dir is required for dry-run, run, or status.")
    resource_plan = plan_resources(
        manifest["tasks"],
        max_workers=args.max_workers,
        memory_budget_gib=args.memory_budget_gib,
        gpu_ids=(),
    )
    if args.dry_run:
        report = dry_run_report(
            manifest=manifest,
            output_dir=args.output_dir,
            project_root=project_root,
            resource_plan=resource_plan,
        )
        report["workload"] = manifest_workload_summary(manifest)
        report["thread_environment_required"] = {
            name: "1" for name in THREAD_ENVIRONMENT
        }
        report["cuda_visible_devices_required"] = ""
        print(json.dumps(report, indent=2), flush=True)
        return 0
    if args.status:
        print(
            json.dumps(
                inspect_batch_status(
                    manifest=manifest,
                    output_dir=args.output_dir,
                    project_root=project_root,
                ),
                indent=2,
            ),
            flush=True,
        )
        return 0
    if not args.run:
        raise SystemExit("Choose --dry-run, --run, or --status.")
    scheduler = _scheduler_record()
    if scheduler["detected_commands"] and not scheduler["active_allocation"]:
        raise SystemExit(
            "A scheduler is installed but no allocation is active; refusing direct run."
        )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        _write_runtime_provenance(
            manifest=manifest,
            output_dir=output_dir,
            resource_plan=resource_plan,
        )
    executor = ParallelValidationExecutor(
        manifest=manifest,
        output_dir=output_dir,
        project_root=project_root,
        resource_plan=resource_plan,
        resume=args.resume,
    )
    summary = executor.run()
    print(
        json.dumps(
            {
                **summary,
                "workload": manifest_workload_summary(manifest),
                "launched_pids": executor.launched_pids,
                "output_dir": str(output_dir),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0 if summary["state"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
