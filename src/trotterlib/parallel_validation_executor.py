"""Bounded, resumable subprocess execution for validation tasks.

This module owns orchestration only.  Scientific adapters call the existing
validation implementation and return JSON-serializable compute records; they
do not decide whether a scientific validation passes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import FrameType
from typing import Any, Callable, Iterable, Mapping, Sequence


TASK_MANIFEST_SCHEMA_VERSION = "parallel_validation_task_manifest_v1"
TASK_SPEC_SCHEMA_VERSION = "parallel_validation_task_spec_v1"
CHECKPOINT_SCHEMA_VERSION = "parallel_validation_checkpoint_v1"
WORKER_RESULT_SCHEMA_VERSION = "parallel_validation_worker_result_v1"
AGGREGATE_SCHEMA_VERSION = "parallel_validation_compute_aggregate_v1"
BATCH_STATUS_SCHEMA_VERSION = "parallel_validation_batch_status_v1"

_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
_METRIC_NAMES = (
    "rz_count",
    "rz_depth",
    "cx_count",
    "cx_depth",
    "total_depth",
    "circuit_size",
)
_ADAPTER_SOURCE_PATHS = {
    "rpe_hadamard_full_wrapper": (
        "src/trotterlib/df_rpe_hadamard_compiled_cost.py",
        "src/trotterlib/rpe_hadamard_compiled_cost_benchmark.py",
        "src/trotterlib/df_partial_s2.py",
        "src/trotterlib/rte.py",
    ),
}


class ParallelValidationError(RuntimeError):
    """Base exception for executor validation and persistence errors."""


class ManifestError(ParallelValidationError):
    """Raised when a task manifest is invalid or internally inconsistent."""


class CheckpointError(ParallelValidationError):
    """Raised when a checkpoint is corrupt or belongs to another task."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(payload: Any) -> str:
    """Return the canonical JSON representation used for all fingerprints."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def fingerprint_payload(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()



def trajectory_seed(master_seed: int, trajectory_index: int) -> int:
    """Derive one stable, non-negative trajectory seed from a task seed."""
    payload = {
        "scheme": "parallel_validation_trajectory_seed_v1",
        "master_seed": _require_int(master_seed, name="master_seed", minimum=0),
        "trajectory_index": _require_int(
            trajectory_index,
            name="trajectory_index",
            minimum=0,
        ),
    }
    return int.from_bytes(
        hashlib.sha256(canonical_json(payload).encode("utf-8")).digest()[:8],
        "big",
    ) % (2**63)


def _json_ready(payload: Any, *, name: str) -> Any:
    try:
        return json.loads(canonical_json(payload))
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"{name} must be finite JSON data: {exc}") from exc


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write JSON through a sibling temporary file and atomic ``os.replace``."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp.{os.getpid()}.{time.time_ns()}"
    )
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointError(f"Existing immutable file is unreadable: {path}") from exc
        if existing != payload:
            raise CheckpointError(f"Refusing to replace mismatched immutable file: {path}")
        return
    atomic_write_json(path, payload)


def _installed_qiskit_version() -> str:
    try:
        return importlib.metadata.version("qiskit")
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _resolve_repo_file(
    value: str | Path,
    *,
    project_root: Path,
    name: str,
) -> Path:
    candidate = Path(value)
    path = candidate if candidate.is_absolute() else project_root / candidate
    resolved = path.resolve(strict=True)
    root = project_root.resolve(strict=True)
    if resolved != root and root not in resolved.parents:
        raise ManifestError(f"{name} must stay inside the project: {value}")
    if not resolved.is_file():
        raise ManifestError(f"{name} is not a file: {value}")
    return resolved


def _relative_path(path: Path, project_root: Path) -> str:
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def validate_output_directory(path: str | Path, project_root: str | Path) -> Path:
    """Require all runtime output to stay below the repository's artifacts/ tree."""
    root = Path(project_root).resolve(strict=True)
    artifacts = (root / "artifacts").resolve(strict=True)
    raw = Path(path)
    candidate = raw if raw.is_absolute() else root / raw
    resolved = candidate.resolve(strict=False)
    if resolved != artifacts and artifacts not in resolved.parents:
        raise ManifestError(
            f"output_dir must be inside {artifacts}; received {candidate}"
        )
    return resolved


def _require_int(value: Any, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestError(f"{name} must be an integer.")
    if minimum is not None and value < minimum:
        raise ManifestError(f"{name} must be >= {minimum}.")
    return value


def _require_float(value: Any, *, name: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestError(f"{name} must be a number.")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ManifestError(f"{name} must be {qualifier}.")
    return number


def _task_source_paths(
    adapter: str,
    manifest_paths: Sequence[Any],
    task_paths: Sequence[Any],
    *,
    project_root: Path,
) -> tuple[Path, ...]:
    values: list[str | Path] = [
        "src/trotterlib/parallel_validation_executor.py",
        *_ADAPTER_SOURCE_PATHS.get(adapter, ()),
        *manifest_paths,
        *task_paths,
    ]
    paths = {
        _resolve_repo_file(value, project_root=project_root, name="source_path")
        for value in values
    }
    return tuple(sorted(paths, key=lambda item: _relative_path(item, project_root)))


def _normalize_task(
    payload: Mapping[str, Any],
    *,
    manifest_source_paths: Sequence[Any],
    project_root: Path,
    qiskit_version: str,
) -> dict[str, Any]:
    validation_id = payload.get("validation_id")
    adapter = payload.get("adapter")
    if not isinstance(validation_id, str) or not validation_id.strip():
        raise ManifestError("task.validation_id must be a non-empty string.")
    if adapter not in ("synthetic", "python_callable", "rpe_hadamard_full_wrapper"):
        raise ManifestError(f"Unsupported task adapter: {adapter!r}.")
    resource = payload.get("resource", "cpu")
    if resource not in ("cpu", "gpu"):
        raise ManifestError("task.resource must be 'cpu' or 'gpu'.")
    if adapter == "rpe_hadamard_full_wrapper" and resource != "cpu":
        raise ManifestError("The full-wrapper transpile adapter is CPU-only.")

    ld = _require_int(payload.get("ld"), name="task.ld", minimum=0)
    delta = _require_float(payload.get("delta"), name="task.delta", positive=True)
    r_value = _require_int(payload.get("r"), name="task.r", minimum=0)
    k_value = _require_int(payload.get("k"), name="task.k", minimum=0)
    q_value = _require_int(payload.get("q"), name="task.q", minimum=1)
    trajectory_index = _require_int(
        payload.get("trajectory_index"),
        name="task.trajectory_index",
        minimum=0,
    )
    seed = _require_int(payload.get("seed"), name="task.seed", minimum=0)
    estimated_memory_gib = _require_float(
        payload.get("estimated_memory_gib", 1.0),
        name="task.estimated_memory_gib",
        positive=True,
    )
    compiler_settings = _json_ready(
        payload.get("compiler_settings", {}), name="task.compiler_settings"
    )
    if not isinstance(compiler_settings, dict):
        raise ManifestError("task.compiler_settings must be an object.")
    parameters = _json_ready(payload.get("parameters", {}), name="task.parameters")
    if not isinstance(parameters, dict):
        raise ManifestError("task.parameters must be an object.")

    input_values = payload.get("input_paths", [])
    source_values = payload.get("source_paths", [])
    if not isinstance(input_values, list) or not isinstance(source_values, list):
        raise ManifestError("task input_paths/source_paths must be arrays.")
    input_paths = tuple(
        sorted(
            {
                _resolve_repo_file(
                    value,
                    project_root=project_root,
                    name="input_path",
                )
                for value in input_values
            },
            key=lambda item: _relative_path(item, project_root),
        )
    )
    source_paths = _task_source_paths(
        adapter,
        manifest_source_paths,
        source_values,
        project_root=project_root,
    )
    input_hashes = {
        _relative_path(path, project_root): file_sha256(path) for path in input_paths
    }
    source_hashes = {
        _relative_path(path, project_root): file_sha256(path) for path in source_paths
    }
    identity = {
        "validation_id": validation_id.strip(),
        "adapter": adapter,
        "resource": resource,
        "ld": ld,
        "delta": delta,
        "r": r_value,
        "k": k_value,
        "q": q_value,
        "trajectory_index": trajectory_index,
        "seed": seed,
        "compiler_settings": compiler_settings,
        "qiskit_version": qiskit_version,
        "parameters": parameters,
        "input_sha256": input_hashes,
        "source_sha256": source_hashes,
    }
    task_fingerprint = fingerprint_payload(identity)
    supplied_task_id = payload.get("task_id")
    if supplied_task_id is not None and supplied_task_id != task_fingerprint:
        raise ManifestError(
            "Supplied task_id does not match the task inputs and provenance."
        )
    return {
        "schema_version": TASK_SPEC_SCHEMA_VERSION,
        "task_id": task_fingerprint,
        "task_fingerprint": task_fingerprint,
        **identity,
        "estimated_memory_gib": estimated_memory_gib,
        "input_paths": list(input_hashes),
        "source_paths": list(source_hashes),
    }


def load_task_manifest(
    path: str | Path,
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    """Load, validate, hash, and deterministically order a task manifest."""
    manifest_path = Path(path).resolve(strict=True)
    root = Path(project_root).resolve(strict=True)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"Task manifest is not valid JSON: {manifest_path}") from exc
    if not isinstance(payload, dict):
        raise ManifestError("Task manifest must be a JSON object.")
    if payload.get("schema_version") != TASK_MANIFEST_SCHEMA_VERSION:
        raise ManifestError("Unsupported task manifest schema_version.")
    batch_id = payload.get("batch_id")
    if not isinstance(batch_id, str) or not batch_id.strip():
        raise ManifestError("Manifest batch_id must be a non-empty string.")
    source_paths = payload.get("source_paths", [])
    tasks = payload.get("tasks")
    if not isinstance(source_paths, list) or not isinstance(tasks, list) or not tasks:
        raise ManifestError("Manifest needs source_paths array and non-empty tasks array.")
    qiskit_version = _installed_qiskit_version()
    normalized = [
        _normalize_task(
            task,
            manifest_source_paths=source_paths,
            project_root=root,
            qiskit_version=qiskit_version,
        )
        for task in tasks
        if isinstance(task, dict)
    ]
    if len(normalized) != len(tasks):
        raise ManifestError("Every manifest task must be a JSON object.")
    ids = [task["task_id"] for task in normalized]
    if len(set(ids)) != len(ids):
        raise ManifestError("Manifest contains duplicate task identities.")
    normalized.sort(key=lambda task: task["task_id"])
    manifest_identity = {
        "schema_version": TASK_MANIFEST_SCHEMA_VERSION,
        "batch_id": batch_id.strip(),
        "tasks": normalized,
    }
    return {
        **manifest_identity,
        "manifest_path": str(manifest_path),
        "manifest_fingerprint": fingerprint_payload(manifest_identity),
    }


def detected_cpu_ids() -> tuple[int, ...]:
    if hasattr(os, "sched_getaffinity"):
        return tuple(sorted(os.sched_getaffinity(0)))
    count = os.cpu_count() or 1
    return tuple(range(count))


def parse_cpu_affinity(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    cpus: set[int] = set()
    for piece in value.split(","):
        item = piece.strip()
        if not item:
            raise ManifestError("cpu-affinity contains an empty item.")
        try:
            if "-" in item:
                left, right = item.split("-", 1)
                start = _require_int(int(left), name="cpu-affinity", minimum=0)
                stop = _require_int(int(right), name="cpu-affinity", minimum=start)
                cpus.update(range(start, stop + 1))
            else:
                cpus.add(_require_int(int(item), name="cpu-affinity", minimum=0))
        except ValueError as exc:
            raise ManifestError(
                f"Invalid cpu-affinity item: {item!r}."
            ) from exc
    if not cpus:
        raise ManifestError("cpu-affinity must select at least one CPU.")
    return tuple(sorted(cpus))


@dataclass(frozen=True)
class ResourcePlan:
    worker_count: int
    detected_cpu_count: int
    detected_cpu_ids: tuple[int, ...]
    affinity_cpu_ids: tuple[int, ...] | None
    memory_budget_gib: float
    maximum_task_memory_gib: float
    estimated_peak_memory_gib: float
    memory_limited_worker_count: int
    gpu_ids: tuple[str, ...]
    task_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "detected_cpu_ids": list(self.detected_cpu_ids),
            "affinity_cpu_ids": (
                None if self.affinity_cpu_ids is None else list(self.affinity_cpu_ids)
            ),
            "gpu_ids": list(self.gpu_ids),
        }


def plan_resources(
    tasks: Sequence[Mapping[str, Any]],
    *,
    max_workers: int | None = None,
    memory_budget_gib: float = 32.0,
    cpu_affinity: Sequence[int] | None = None,
    gpu_ids: Sequence[str] = (),
    allow_more_than_16_workers: bool = False,
    available_cpu_ids: Sequence[int] | None = None,
) -> ResourcePlan:
    """Build a conservative CPU/memory/GPU plan without starting work."""
    if not tasks:
        raise ManifestError("At least one task is required.")
    available = tuple(
        sorted(set(detected_cpu_ids() if available_cpu_ids is None else available_cpu_ids))
    )
    if not available:
        raise ManifestError("No CPUs are available to this process.")
    budget = _require_float(memory_budget_gib, name="memory_budget_gib", positive=True)
    maximum_memory = max(
        _require_float(
            task.get("estimated_memory_gib", 1.0),
            name="estimated_memory_gib",
            positive=True,
        )
        for task in tasks
    )
    memory_limit = int(math.floor(budget / maximum_memory))
    if memory_limit < 1:
        raise ManifestError(
            "memory budget is smaller than the largest task estimate "
            f"({budget:g} < {maximum_memory:g} GiB)."
        )
    normalized_affinity: tuple[int, ...] | None = None
    if cpu_affinity is not None:
        normalized_affinity = tuple(sorted(set(int(item) for item in cpu_affinity)))
        unavailable = sorted(set(normalized_affinity).difference(available))
        if not normalized_affinity or unavailable:
            raise ManifestError(
                f"cpu-affinity includes unavailable CPUs: {unavailable or 'none selected'}"
            )
    cpu_capacity = len(normalized_affinity or available)
    default_workers = min(
        8,
        max(1, len(available) // 4),
        cpu_capacity,
        memory_limit,
        len(tasks),
    )
    if max_workers is None:
        workers = default_workers
    else:
        workers = _require_int(max_workers, name="max_workers", minimum=1)
        if workers > 16 and not allow_more_than_16_workers:
            raise ManifestError(
                "More than 16 workers requires --allow-more-than-16-workers."
            )
        if workers > memory_limit:
            raise ManifestError(
                f"max_workers={workers} exceeds the memory limit {memory_limit}."
            )
        if workers > cpu_capacity:
            raise ManifestError(
                f"max_workers={workers} exceeds selected CPU capacity {cpu_capacity}."
            )
        if len(available) > 1 and workers >= len(available):
            raise ManifestError("Refusing to occupy every detected CPU.")
        workers = min(workers, len(tasks))
    normalized_gpus = tuple(str(item).strip() for item in gpu_ids)
    if any(not item for item in normalized_gpus) or len(set(normalized_gpus)) != len(
        normalized_gpus
    ):
        raise ManifestError("gpu-ids must be non-empty and unique.")
    if any(task.get("resource", "cpu") == "gpu" for task in tasks) and not normalized_gpus:
        raise ManifestError("GPU tasks require explicit --gpu-ids.")
    return ResourcePlan(
        worker_count=workers,
        detected_cpu_count=len(available),
        detected_cpu_ids=available,
        affinity_cpu_ids=normalized_affinity,
        memory_budget_gib=budget,
        maximum_task_memory_gib=maximum_memory,
        estimated_peak_memory_gib=workers * maximum_memory,
        memory_limited_worker_count=memory_limit,
        gpu_ids=normalized_gpus,
        task_count=len(tasks),
    )


def _checkpoint_payload(
    *,
    task: Mapping[str, Any],
    status: str,
    attempt: int,
    pid: int,
    started_at_utc: str,
    finished_at_utc: str,
    assigned_gpu_id: str | None,
    affinity_cpu_id: int | None,
    result: Mapping[str, Any] | None,
    error: Mapping[str, Any] | None,
) -> dict[str, Any]:
    body = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "artifact_kind": "execution_compute_checkpoint",
        "scientific_verdict_included": False,
        "task_id": task["task_id"],
        "task_fingerprint": task["task_fingerprint"],
        "status": status,
        "attempt": attempt,
        "pid": pid,
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "assigned_gpu_id": assigned_gpu_id,
        "affinity_cpu_id": affinity_cpu_id,
        "provenance": {
            "validation_id": task["validation_id"],
            "adapter": task["adapter"],
            "resource": task["resource"],
            "input_sha256": task["input_sha256"],
            "source_sha256": task["source_sha256"],
            "compiler_settings": task["compiler_settings"],
            "qiskit_version": task["qiskit_version"],
            "seed": task["seed"],
        },
        "result": result,
        "error": error,
    }
    return {**body, "checkpoint_fingerprint": fingerprint_payload(body)}


def read_checkpoint(path: str | Path, *, task: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint_path = Path(path)
    try:
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"Corrupt checkpoint: {checkpoint_path}") from exc
    if not isinstance(payload, dict):
        raise CheckpointError(f"Checkpoint is not an object: {checkpoint_path}")
    stored_fingerprint = payload.get("checkpoint_fingerprint")
    body = {key: value for key, value in payload.items() if key != "checkpoint_fingerprint"}
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointError(f"Unsupported checkpoint schema: {checkpoint_path}")
    if stored_fingerprint != fingerprint_payload(body):
        raise CheckpointError(f"Checkpoint fingerprint mismatch: {checkpoint_path}")
    if payload.get("task_id") != task["task_id"] or payload.get(
        "task_fingerprint"
    ) != task["task_fingerprint"]:
        raise CheckpointError(f"Checkpoint belongs to a different task: {checkpoint_path}")
    if payload.get("status") not in ("completed", "failed", "interrupted"):
        raise CheckpointError(f"Unsupported checkpoint status: {checkpoint_path}")
    return payload


def _numeric_metrics(result: Mapping[str, Any]) -> dict[str, float]:
    metrics = result.get("metrics", {})
    if not isinstance(metrics, Mapping):
        raise CheckpointError("Completed result metrics must be an object.")
    values: dict[str, float] = {}
    for name, value in metrics.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        number = float(value)
        if not math.isfinite(number):
            raise CheckpointError(f"Metric {name!r} is not finite.")
        values[str(name)] = number
    return values


def build_deterministic_aggregate(
    checkpoints: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate completed results in task-id order with stable summation."""
    completed = sorted(
        (dict(item) for item in checkpoints if item.get("status") == "completed"),
        key=lambda item: item["task_id"],
    )
    rows = []
    for item in completed:
        metrics = _numeric_metrics(item["result"])
        rows.append(
            {
                "task_id": item["task_id"],
                "task_fingerprint": item["task_fingerprint"],
                "result_fingerprint": fingerprint_payload({"metrics": metrics}),
                "metrics": metrics,
            }
        )
    metric_names = sorted({name for row in rows for name in row["metrics"]})
    summaries: dict[str, dict[str, float | int | None]] = {}
    for name in metric_names:
        values = [row["metrics"][name] for row in rows if name in row["metrics"]]
        count = len(values)
        mean = math.fsum(values) / count
        if count < 2:
            standard_error = None
        else:
            squared = math.fsum((value - mean) ** 2 for value in values)
            standard_error = math.sqrt(squared / (count - 1)) / math.sqrt(count)
        summaries[name] = {
            "count": count,
            "mean": mean,
            "standard_error": standard_error,
        }
    deterministic = {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "artifact_kind": "execution_compute_aggregate",
        "scientific_verdict_included": False,
        "task_count": len(rows),
        "task_ids": [row["task_id"] for row in rows],
        "records": rows,
        "metric_summaries": summaries,
    }
    return {
        **deterministic,
        "aggregate_fingerprint": fingerprint_payload(deterministic),
    }


def _synthetic_adapter(task: Mapping[str, Any], *, attempt: int) -> dict[str, Any]:
    parameters = task["parameters"]
    sleep_seconds = _require_float(
        parameters.get("sleep_seconds", 0.0),
        name="sleep_seconds",
    )
    if sleep_seconds < 0.0:
        raise ManifestError("sleep_seconds must be non-negative.")
    if sleep_seconds:
        time.sleep(sleep_seconds)
    fail_until = _require_int(
        parameters.get("fail_until_attempt", 0),
        name="fail_until_attempt",
        minimum=0,
    )
    if attempt <= fail_until:
        raise RuntimeError(f"intentional synthetic failure for attempt {attempt}")
    value = _require_float(parameters.get("value", 0.0), name="value")
    metrics = parameters.get("metrics", {"value": value})
    if not isinstance(metrics, dict):
        raise ManifestError("synthetic metrics must be an object.")
    return {
        "adapter": "synthetic",
        "metrics": _json_ready(metrics, name="synthetic metrics"),
        "value": value,
        "pid": os.getpid(),
        "thread_environment": {
            name: os.environ.get(name) for name in _THREAD_ENVIRONMENT
        },
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "affinity_cpu_ids": (
            sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
        ),
    }


def _python_callable_adapter(task: Mapping[str, Any], *, attempt: int) -> dict[str, Any]:
    target = task["parameters"].get("callable")
    if not isinstance(target, str) or ":" not in target:
        raise ManifestError("python_callable needs parameters.callable='module:function'.")
    module_name, function_name = target.split(":", 1)
    if not module_name.startswith(("trotterlib.", "scripts.")):
        raise ManifestError("python_callable is limited to trotterlib.* or scripts.*.")
    function = getattr(importlib.import_module(module_name), function_name)
    result = function(dict(task), attempt=attempt)
    if not isinstance(result, Mapping):
        raise TypeError("Validation callable must return a mapping.")
    return _json_ready(dict(result), name="python callable result")


def _compiler_from_task(task: Mapping[str, Any]):
    from .rte import CompilerSettings

    payload = dict(task["compiler_settings"])
    required = {
        "basis_gates",
        "backend_name",
        "coupling_map",
        "optimization_level",
        "layout_method",
        "routing_method",
        "transpiler_seed",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ManifestError(f"compiler_settings is missing: {', '.join(missing)}")
    payload["basis_gates"] = tuple(payload["basis_gates"])
    if payload["coupling_map"] is not None:
        payload["coupling_map"] = tuple(tuple(edge) for edge in payload["coupling_map"])
    payload["qiskit_version"] = task["qiskit_version"]
    return CompilerSettings(**payload)


def _rpe_hadamard_full_wrapper_adapter(
    task: Mapping[str, Any], *, attempt: int
) -> dict[str, Any]:
    """Adapt one trajectory task to the existing paired full-wrapper path."""
    from .df_partial_randomized_pf import split_df_hamiltonian_by_ld
    from .df_partial_s2 import prepare_df_partial_s2
    from .rpe_hadamard_compiled_cost_benchmark import (
        RPEHadamardCompiledCostBenchmarkRequest,
        generate_rpe_hadamard_compiled_cost_benchmark_dataset,
    )
    from .rte import finite_rte_distribution, make_rte_config
    from .rte_compiled_cost import TranspiledCircuitCostCache
    from .rte_connected_cluster_cost_validation import (
        load_connected_cluster_hamiltonian_snapshot,
    )

    parameters = task["parameters"]
    snapshot_value = parameters.get("snapshot_path")
    if not isinstance(snapshot_value, str):
        raise ManifestError("full-wrapper adapter needs parameters.snapshot_path.")
    project_root = Path(task["project_root"])
    snapshot = _resolve_repo_file(
        snapshot_value,
        project_root=project_root,
        name="snapshot_path",
    )
    normalized_snapshot = _relative_path(snapshot, project_root)
    if normalized_snapshot not in task["input_sha256"]:
        raise ManifestError("snapshot_path must also appear in task.input_paths.")
    hamiltonian = load_connected_cluster_hamiltonian_snapshot(snapshot)
    preparation = prepare_df_partial_s2(
        hamiltonian,
        split_df_hamiltonian_by_ld(hamiltonian, int(task["ld"])),
        identity_policy="extract_identity_phase",
    )
    r_value = int(task["r"])
    k_value = int(task["k"])
    delta = float(task["delta"])
    task_trajectory_seed = trajectory_seed(
        int(task["seed"]),
        int(task["trajectory_index"]),
    )
    if preparation.is_deterministic_only:
        if r_value or k_value:
            raise ManifestError("A deterministic tail requires r=K=0.")
        rte_config = None
        rte_distribution = None
        method = "exact"
        sample_count = None
        master_seed = None
    else:
        if r_value < 1 or k_value % 2:
            raise ManifestError("A randomized tail requires r>=1 and even K.")
        tau = preparation.exact_rte_lambda_r * delta / r_value
        distribution = finite_rte_distribution(tau, k_value)
        rte_config, rte_distribution = make_rte_config(
            preparation.rte_preparation.symbolic_tail,
            evolution_time=delta,
            rte_steps=r_value,
            truncation_tolerance=max(
                distribution.step_truncation_residual_bound,
                math.ulp(0.0),
            ),
            finite_taylor_order=k_value,
            seed=task_trajectory_seed,
        )
        method = "monte_carlo"
        sample_count = 1
        master_seed = task_trajectory_seed
    partition = parameters.get("partition", "calibration")
    if partition not in ("calibration", "holdout"):
        raise ManifestError("partition must be calibration or holdout.")
    q_value = int(task["q"])
    q_tuple = (q_value,)
    request = RPEHadamardCompiledCostBenchmarkRequest(
        preparation=preparation,
        delta_time=delta,
        calibration_repetition_counts=q_tuple if partition == "calibration" else (),
        holdout_repetition_counts=q_tuple if partition == "holdout" else (),
        rte_steps_per_occurrence=r_value,
        finite_taylor_order=k_value,
        rte_config=rte_config,
        rte_distribution=rte_distribution,
        compiler=_compiler_from_task(task),
        evaluation_method=method,
        sample_count=sample_count,
        seed=master_seed,
        generation_id=f"parallel-task-{task['task_id']}",
        maximum_repetition_count=max(
            q_value,
            int(parameters.get("maximum_repetition_count", q_value)),
        ),
        maximum_trajectories=int(parameters.get("maximum_trajectories", 10_000)),
        maximum_samples=1,
        maximum_untranspiled_circuit_size=int(
            parameters.get("maximum_untranspiled_circuit_size", 100_000)
        ),
        maximum_retained_trajectory_records=1,
        maximum_build_requests=int(parameters.get("maximum_build_requests", 8)),
        maximum_transpile_requests=int(parameters.get("maximum_transpile_requests", 8)),
        maximum_planned_instruction_applications=int(
            parameters.get("maximum_planned_instruction_applications", 10_000_000)
        ),
        construction_policy=parameters.get("construction_policy", "boundary_optimized"),
        cache=TranspiledCircuitCostCache(),
    )
    benchmark = generate_rpe_hadamard_compiled_cost_benchmark_dataset(request)
    dataset = benchmark.dataset
    if not dataset.complete:
        raise RuntimeError(
            "full-wrapper benchmark failed: " + "; ".join(dataset.incomplete_reasons)
        )
    dataset_payload = dataset.to_dict()
    metrics: dict[str, float] = {}
    for record in dataset_payload["records"]:
        for metric in _METRIC_NAMES:
            metrics[f"{record['axis']}_{metric}"] = float(record["cost"][metric])
    return {
        "adapter": "rpe_hadamard_full_wrapper",
        "attempt": attempt,
        "requested_trajectory_index": task["trajectory_index"],
        "derived_trajectory_seed": task_trajectory_seed,
        "paired_axes": ["cosine", "sine"],
        "metrics": metrics,
        "dataset": dataset_payload,
        "scientific_verdict_included": False,
    }


_ADAPTERS: dict[str, Callable[..., dict[str, Any]]] = {
    "synthetic": _synthetic_adapter,
    "python_callable": _python_callable_adapter,
    "rpe_hadamard_full_wrapper": _rpe_hadamard_full_wrapper_adapter,
}


def execute_worker_spec(spec_path: str | Path, result_path: str | Path) -> int:
    """Execute exactly one immutable task spec in a child process."""
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    task = spec["task"]
    if spec.get("schema_version") != TASK_SPEC_SCHEMA_VERSION:
        raise ManifestError("Worker spec has an unsupported schema.")
    if task.get("task_fingerprint") != fingerprint_payload(
        {
            key: task[key]
            for key in (
                "validation_id",
                "adapter",
                "resource",
                "ld",
                "delta",
                "r",
                "k",
                "q",
                "trajectory_index",
                "seed",
                "compiler_settings",
                "qiskit_version",
                "parameters",
                "input_sha256",
                "source_sha256",
            )
        }
    ):
        raise ManifestError("Worker task fingerprint mismatch.")
    affinity_cpu_id = spec.get("affinity_cpu_id")
    if affinity_cpu_id is not None:
        if not hasattr(os, "sched_setaffinity"):
            raise RuntimeError("CPU affinity was requested but is unavailable.")
        os.sched_setaffinity(0, {int(affinity_cpu_id)})
    attempt = int(spec["attempt"])
    started = _utc_now()
    try:
        result = _ADAPTERS[task["adapter"]](task, attempt=attempt)
    except BaseException as exc:
        payload = {
            "schema_version": WORKER_RESULT_SCHEMA_VERSION,
            "task_id": task["task_id"],
            "task_fingerprint": task["task_fingerprint"],
            "status": "failed",
            "attempt": attempt,
            "pid": os.getpid(),
            "started_at_utc": started,
            "finished_at_utc": _utc_now(),
            "result": None,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        }
        atomic_write_json(result_path, payload)
        return 1
    payload = {
        "schema_version": WORKER_RESULT_SCHEMA_VERSION,
        "task_id": task["task_id"],
        "task_fingerprint": task["task_fingerprint"],
        "status": "completed",
        "attempt": attempt,
        "pid": os.getpid(),
        "started_at_utc": started,
        "finished_at_utc": _utc_now(),
        "result": _json_ready(result, name="adapter result"),
        "error": None,
    }
    atomic_write_json(result_path, payload)
    return 0


@dataclass
class _RunningTask:
    task: dict[str, Any]
    process: subprocess.Popen[bytes]
    log_stream: Any
    log_path: Path
    log_temporary_path: Path
    spec_path: Path
    result_path: Path
    attempt: int
    assigned_gpu_id: str | None
    affinity_cpu_id: int | None
    started_at_utc: str


class ParallelValidationExecutor:
    """Run validated tasks as bounded child processes with atomic checkpoints."""

    def __init__(
        self,
        *,
        manifest: Mapping[str, Any],
        output_dir: str | Path,
        project_root: str | Path,
        resource_plan: ResourcePlan,
        resume: bool = False,
        poll_interval_seconds: float = 0.05,
        termination_grace_seconds: float = 3.0,
    ) -> None:
        self.manifest = dict(manifest)
        self.tasks = [dict(item) for item in manifest["tasks"]]
        self.project_root = Path(project_root).resolve(strict=True)
        self.output_dir = validate_output_directory(output_dir, self.project_root)
        self.resource_plan = resource_plan
        self.resume = bool(resume)
        self.poll_interval_seconds = poll_interval_seconds
        self.termination_grace_seconds = termination_grace_seconds
        self._running: dict[int, _RunningTask] = {}
        self._stop_requested = False
        self._interrupted_signal: int | None = None
        self.launched_pids: list[int] = []
        self.terminated_pids: list[int] = []

    @property
    def checkpoint_dir(self) -> Path:
        return self.output_dir / "checkpoints"

    def _checkpoint_path(self, task: Mapping[str, Any]) -> Path:
        return self.checkpoint_dir / f"{task['task_id']}.json"

    def _existing_checkpoints(self) -> dict[str, dict[str, Any]]:
        task_by_id = {task["task_id"]: task for task in self.tasks}
        if not self.checkpoint_dir.exists():
            return {}
        found: dict[str, dict[str, Any]] = {}
        for path in sorted(self.checkpoint_dir.glob("*.json")):
            task = task_by_id.get(path.stem)
            if task is None:
                raise CheckpointError(
                    f"Output contains a checkpoint not present in this manifest: {path}"
                )
            found[path.stem] = read_checkpoint(path, task=task)
        return found

    def _has_runtime_state(self) -> bool:
        if any(
            (self.output_dir / name).exists()
            for name in ("batch_status.json", "aggregate.json")
        ):
            return True
        for name in ("tasks", "worker_results", "logs"):
            directory = self.output_dir / name
            if directory.exists() and any(directory.iterdir()):
                return True
        return False

    def _pending_tasks(self) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        existing = self._existing_checkpoints()
        if not self.resume and (existing or self._has_runtime_state()):
            raise CheckpointError(
                "Existing runtime state requires --resume; refusing to replace results."
            )
        pending = [
            task
            for task in self.tasks
            if existing.get(task["task_id"], {}).get("status") != "completed"
        ]
        return pending, existing

    def _attempt_for(
        self, task: Mapping[str, Any], existing: Mapping[str, Mapping[str, Any]]
    ) -> int:
        checkpoint = existing.get(task["task_id"])
        attempts = [0 if checkpoint is None else int(checkpoint["attempt"])]
        pattern = "{}.attempt-*.json".format(task["task_id"])
        directories = (
            self.output_dir / "tasks",
            self.output_dir / "worker_results",
        )
        for directory in directories:
            for path in directory.glob(pattern) if directory.exists() else ():
                try:
                    attempts.append(int(path.stem.rsplit("-", 1)[1]))
                except (IndexError, ValueError) as exc:
                    raise CheckpointError(
                        f"Malformed attempt file name: {path}"
                    ) from exc
        log_directory = self.output_dir / "logs"
        log_pattern = "{}.attempt-*.log".format(task["task_id"])
        if log_directory.exists():
            for path in log_directory.glob(log_pattern):
                try:
                    attempts.append(int(path.stem.rsplit("-", 1)[1]))
                except (IndexError, ValueError) as exc:
                    raise CheckpointError(
                        f"Malformed attempt log name: {path}"
                    ) from exc
        return max(attempts) + 1

    def request_stop(self, signum: int = signal.SIGINT) -> None:
        """Stop scheduling and terminate only PIDs launched by this executor."""
        self._stop_requested = True
        self._interrupted_signal = int(signum)
        for pid, running in tuple(self._running.items()):
            if running.process.poll() is None:
                try:
                    running.process.terminate()
                except ProcessLookupError:
                    continue
                if pid not in self.terminated_pids:
                    self.terminated_pids.append(pid)

    def _signal_handler(self, signum: int, _frame: FrameType | None) -> None:
        self.request_stop(signum)

    def _write_batch_status(
        self,
        *,
        state: str,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        payload = {
            "schema_version": BATCH_STATUS_SCHEMA_VERSION,
            "artifact_kind": "execution_status",
            "scientific_verdict_included": False,
            "batch_id": self.manifest["batch_id"],
            "manifest_fingerprint": self.manifest["manifest_fingerprint"],
            "state": state,
            "updated_at_utc": _utc_now(),
            "resource_plan": self.resource_plan.to_dict(),
            "active_pids": sorted(self._running),
            "launched_pids": self.launched_pids,
            "terminated_pids": self.terminated_pids,
            **dict(extra or {}),
        }
        atomic_write_json(self.output_dir / "batch_status.json", payload)

    def _choose_gpu(self) -> str | None:
        used = {
            running.assigned_gpu_id
            for running in self._running.values()
            if running.assigned_gpu_id is not None
        }
        return next((gpu for gpu in self.resource_plan.gpu_ids if gpu not in used), None)

    def _choose_affinity_cpu(self) -> int | None:
        candidates = self.resource_plan.affinity_cpu_ids
        if candidates is None:
            return None
        used = {
            running.affinity_cpu_id
            for running in self._running.values()
            if running.affinity_cpu_id is not None
        }
        return next((cpu for cpu in candidates if cpu not in used), None)

    def _can_launch(self, task: Mapping[str, Any]) -> bool:
        if len(self._running) >= self.resource_plan.worker_count:
            return False
        if task["resource"] == "gpu" and self._choose_gpu() is None:
            return False
        if (
            self.resource_plan.affinity_cpu_ids is not None
            and self._choose_affinity_cpu() is None
        ):
            return False
        return True

    def _launch(self, task: dict[str, Any], *, attempt: int) -> None:
        assigned_gpu = self._choose_gpu() if task["resource"] == "gpu" else None
        affinity_cpu = self._choose_affinity_cpu()
        task_with_root = {**task, "project_root": str(self.project_root)}
        spec = {
            "schema_version": TASK_SPEC_SCHEMA_VERSION,
            "task": task_with_root,
            "attempt": attempt,
            "assigned_gpu_id": assigned_gpu,
            "affinity_cpu_id": affinity_cpu,
        }
        task_dir = self.output_dir / "tasks"
        result_dir = self.output_dir / "worker_results"
        log_dir = self.output_dir / "logs"
        spec_path = task_dir / f"{task['task_id']}.attempt-{attempt:04d}.json"
        result_path = result_dir / f"{task['task_id']}.attempt-{attempt:04d}.json"
        log_path = log_dir / f"{task['task_id']}.attempt-{attempt:04d}.log"
        log_temporary_path = log_path.with_name(
            f".{log_path.name}.tmp.{os.getpid()}.{time.time_ns()}"
        )
        _write_immutable_json(spec_path, spec)
        result_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        if result_path.exists() or log_path.exists():
            raise CheckpointError(
                f"Attempt files already exist without a usable checkpoint: {task['task_id']}"
            )
        log_stream = log_temporary_path.open("x", encoding="utf-8")
        environment = os.environ.copy()
        environment.update(_THREAD_ENVIRONMENT)
        environment["CUDA_VISIBLE_DEVICES"] = assigned_gpu or ""
        source_path = str(self.project_root / "src")
        inherited_python_path = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            source_path
            if not inherited_python_path
            else os.pathsep.join((source_path, inherited_python_path))
        )
        command = [
            sys.executable,
            "-m",
            "trotterlib.parallel_validation_executor",
            "--worker-spec",
            str(spec_path),
            "--worker-result",
            str(result_path),
        ]
        try:
            process = subprocess.Popen(
                command,
                cwd=self.project_root,
                env=environment,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
            )
        except BaseException:
            log_stream.close()
            raise
        running = _RunningTask(
            task=task,
            process=process,
            log_stream=log_stream,
            log_path=log_path,
            log_temporary_path=log_temporary_path,
            spec_path=spec_path,
            result_path=result_path,
            attempt=attempt,
            assigned_gpu_id=assigned_gpu,
            affinity_cpu_id=affinity_cpu,
            started_at_utc=_utc_now(),
        )
        self._running[process.pid] = running
        self.launched_pids.append(process.pid)
        self._write_batch_status(state="running")

    def _finalize(self, pid: int, running: _RunningTask) -> dict[str, Any]:
        return_code = running.process.wait()
        running.log_stream.flush()
        os.fsync(running.log_stream.fileno())
        running.log_stream.close()
        if running.log_path.exists():
            raise CheckpointError(
                f"Refusing to replace existing task log: {running.log_path}"
            )
        os.replace(running.log_temporary_path, running.log_path)
        if running.result_path.exists():
            try:
                worker = json.loads(running.result_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise CheckpointError(
                    f"Worker result is corrupt: {running.result_path}"
                ) from exc
            if (
                worker.get("schema_version") != WORKER_RESULT_SCHEMA_VERSION
                or worker.get("task_id") != running.task["task_id"]
                or worker.get("task_fingerprint")
                != running.task["task_fingerprint"]
            ):
                raise CheckpointError(
                    f"Worker result identity mismatch: {running.result_path}"
                )
            status = worker["status"]
            result = worker.get("result")
            error = worker.get("error")
            started_at = worker.get("started_at_utc", running.started_at_utc)
            finished_at = worker.get("finished_at_utc", _utc_now())
        else:
            status = "interrupted" if self._stop_requested else "failed"
            result = None
            error = {
                "type": "WorkerProcessExit",
                "message": f"worker exited with code {return_code} without a result",
            }
            started_at = running.started_at_utc
            finished_at = _utc_now()
        checkpoint = _checkpoint_payload(
            task=running.task,
            status=status,
            attempt=running.attempt,
            pid=pid,
            started_at_utc=started_at,
            finished_at_utc=finished_at,
            assigned_gpu_id=running.assigned_gpu_id,
            affinity_cpu_id=running.affinity_cpu_id,
            result=result,
            error=error,
        )
        atomic_write_json(self._checkpoint_path(running.task), checkpoint)
        return checkpoint

    def _terminate_lingering(self) -> None:
        deadline = time.monotonic() + self.termination_grace_seconds
        while any(item.process.poll() is None for item in self._running.values()):
            if time.monotonic() >= deadline:
                break
            time.sleep(min(self.poll_interval_seconds, 0.05))
        for pid, running in tuple(self._running.items()):
            if running.process.poll() is None:
                try:
                    running.process.kill()
                except ProcessLookupError:
                    continue
                if pid not in self.terminated_pids:
                    self.terminated_pids.append(pid)

    def run(self) -> dict[str, Any]:
        """Execute pending/failed tasks and return a deterministic batch summary."""
        pending, existing = self._pending_tasks()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        queue = list(pending)
        old_handlers: dict[int, Any] = {}
        if threading_main_process():
            for signum in (signal.SIGINT, signal.SIGTERM):
                old_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, self._signal_handler)
        self._write_batch_status(
            state="running",
            extra={"pending_task_count": len(queue)},
        )
        try:
            while queue or self._running:
                launched = True
                while queue and launched and not self._stop_requested:
                    launched = False
                    for index, task in enumerate(queue):
                        if self._can_launch(task):
                            queue.pop(index)
                            self._launch(
                                task,
                                attempt=self._attempt_for(task, existing),
                            )
                            launched = True
                            break
                finished = [
                    pid
                    for pid, running in self._running.items()
                    if running.process.poll() is not None
                ]
                for pid in finished:
                    running = self._running.pop(pid)
                    checkpoint = self._finalize(pid, running)
                    existing[running.task["task_id"]] = checkpoint
                    self._write_batch_status(state="running")
                if self._stop_requested and self._running:
                    self._terminate_lingering()
                if self._stop_requested and not self._running:
                    break
                if not finished:
                    time.sleep(self.poll_interval_seconds)
        finally:
            if self._running:
                self.request_stop(self._interrupted_signal or signal.SIGTERM)
                self._terminate_lingering()
                for pid in tuple(self._running):
                    running = self._running.pop(pid)
                    checkpoint = self._finalize(pid, running)
                    existing[running.task["task_id"]] = checkpoint
            for signum, previous in old_handlers.items():
                signal.signal(signum, previous)

        checkpoints = self._existing_checkpoints()
        aggregate = build_deterministic_aggregate(checkpoints.values())
        atomic_write_json(self.output_dir / "aggregate.json", aggregate)
        counts = {
            status: sum(item["status"] == status for item in checkpoints.values())
            for status in ("completed", "failed", "interrupted")
        }
        counts["pending"] = len(self.tasks) - sum(counts.values())
        state = (
            "interrupted"
            if self._stop_requested
            else "completed"
            if counts["completed"] == len(self.tasks)
            else "failed"
        )
        summary = {
            "state": state,
            "counts": counts,
            "aggregate_fingerprint": aggregate["aggregate_fingerprint"],
            "signal": self._interrupted_signal,
        }
        self._write_batch_status(state=state, extra=summary)
        return summary


def threading_main_process() -> bool:
    """Return true when Python signal handlers may be installed."""
    import threading

    return threading.current_thread() is threading.main_thread()


def inspect_batch_status(
    *,
    manifest: Mapping[str, Any],
    output_dir: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    output = validate_output_directory(output_dir, project_root)
    checkpoint_dir = output / "checkpoints"
    task_by_id = {task["task_id"]: task for task in manifest["tasks"]}
    statuses: dict[str, str] = {}
    if checkpoint_dir.exists():
        for path in sorted(checkpoint_dir.glob("*.json")):
            if path.stem not in task_by_id:
                raise CheckpointError(f"Unknown checkpoint in output: {path}")
            checkpoint = read_checkpoint(path, task=task_by_id[path.stem])
            statuses[path.stem] = checkpoint["status"]
    counts = {
        status: sum(value == status for value in statuses.values())
        for status in ("completed", "failed", "interrupted")
    }
    counts["pending"] = len(task_by_id) - len(statuses)
    return {
        "batch_id": manifest["batch_id"],
        "manifest_fingerprint": manifest["manifest_fingerprint"],
        "output_dir": str(output),
        "counts": counts,
        "tasks": [
            {"task_id": task_id, "status": statuses.get(task_id, "pending")}
            for task_id in sorted(task_by_id)
        ],
    }


def dry_run_report(
    *,
    manifest: Mapping[str, Any],
    output_dir: str | Path,
    project_root: str | Path,
    resource_plan: ResourcePlan,
) -> dict[str, Any]:
    return {
        "dry_run": True,
        "batch_id": manifest["batch_id"],
        "manifest_fingerprint": manifest["manifest_fingerprint"],
        "output_dir": str(validate_output_directory(output_dir, project_root)),
        "resource_plan": resource_plan.to_dict(),
        "tasks": [
            {
                "task_id": task["task_id"],
                "validation_id": task["validation_id"],
                "resource": task["resource"],
                "estimated_memory_gib": task["estimated_memory_gib"],
                "input_sha256": task["input_sha256"],
                "source_sha256": task["source_sha256"],
            }
            for task in manifest["tasks"]
        ],
    }


def create_hadamard_task_manifest(
    path: str | Path,
    *,
    project_root: str | Path,
    batch_id: str,
    validation_id: str,
    snapshot_path: str | Path,
    ld: int,
    delta: float,
    r: int,
    k: int,
    q_values: Sequence[int],
    trajectory_count: int,
    seed: int,
    compiler_settings: Mapping[str, Any],
    estimated_memory_gib: float = 1.0,
    partition: str = "calibration",
) -> dict[str, Any]:
    """Create one immutable full-wrapper task manifest without running tasks."""
    root = Path(project_root).resolve(strict=True)
    output = validate_output_directory(path, root)
    if output.exists():
        raise ManifestError(f"Refusing to replace existing manifest: {output}")
    snapshot = _resolve_repo_file(
        snapshot_path,
        project_root=root,
        name="snapshot_path",
    )
    snapshot_relative = _relative_path(snapshot, root)
    ld_value = _require_int(ld, name="ld", minimum=0)
    delta_value = _require_float(delta, name="delta", positive=True)
    r_value = _require_int(r, name="r", minimum=0)
    k_value = _require_int(k, name="k", minimum=0)
    count = _require_int(
        trajectory_count,
        name="trajectory_count",
        minimum=1,
    )
    master_seed = _require_int(seed, name="seed", minimum=0)
    memory = _require_float(
        estimated_memory_gib,
        name="estimated_memory_gib",
        positive=True,
    )
    if partition not in ("calibration", "holdout"):
        raise ManifestError("partition must be calibration or holdout.")
    normalized_q = tuple(sorted(set(q_values)))
    if not normalized_q:
        raise ManifestError("At least one q value is required.")
    for q_value in normalized_q:
        _require_int(q_value, name="q", minimum=1)
        if q_value & (q_value - 1):
            raise ManifestError("Every q value must be a positive power of two.")
    compiler = _json_ready(compiler_settings, name="compiler_settings")
    if not isinstance(compiler, dict):
        raise ManifestError("compiler_settings must be an object.")
    if not isinstance(batch_id, str) or not batch_id.strip():
        raise ManifestError("batch_id must be a non-empty string.")
    if not isinstance(validation_id, str) or not validation_id.strip():
        raise ManifestError("validation_id must be a non-empty string.")
    tasks = [
        {
            "validation_id": validation_id,
            "adapter": "rpe_hadamard_full_wrapper",
            "resource": "cpu",
            "ld": ld_value,
            "delta": delta_value,
            "r": r_value,
            "k": k_value,
            "q": q_value,
            "trajectory_index": trajectory_index,
            "seed": master_seed,
            "compiler_settings": compiler,
            "input_paths": [snapshot_relative],
            "source_paths": [],
            "estimated_memory_gib": memory,
            "parameters": {
                "snapshot_path": snapshot_relative,
                "partition": partition,
            },
        }
        for q_value in normalized_q
        for trajectory_index in range(count)
    ]
    qiskit_version = _installed_qiskit_version()
    for task in tasks:
        _normalize_task(
            task,
            manifest_source_paths=(),
            project_root=root,
            qiskit_version=qiskit_version,
        )
    payload = {
        "schema_version": TASK_MANIFEST_SCHEMA_VERSION,
        "batch_id": batch_id,
        "source_paths": [],
        "tasks": tasks,
    }
    atomic_write_json(output, payload)
    return load_task_manifest(output, project_root=root)


def _default_compiler_settings() -> dict[str, Any]:
    return {
        "basis_gates": ["rz", "sx", "x", "cx"],
        "backend_name": None,
        "coupling_map": None,
        "optimization_level": 1,
        "layout_method": None,
        "routing_method": None,
        "transpiler_seed": 17,
    }


def _comma_separated_ints(value: str, *, name: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ManifestError(f"{name} must be a comma-separated integer list.") from exc
    if not values:
        raise ManifestError(f"{name} must not be empty.")
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run bounded, resumable validation tasks from a JSON manifest."
    )
    parser.add_argument("--task-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--memory-budget-gib", type=float, default=32.0)
    parser.add_argument("--cpu-affinity")
    parser.add_argument("--gpu-ids", default="")
    parser.add_argument("--allow-more-than-16-workers", action="store_true")
    parser.add_argument("--create-hadamard-manifest", type=Path)
    parser.add_argument("--batch-id", default="parallel-validation-batch")
    parser.add_argument("--validation-id", default="rpe-hadamard-full-wrapper")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--ld", type=int, default=3)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--r", type=int, default=4)
    parser.add_argument("--finite-taylor-order", type=int, default=2)
    parser.add_argument("--q-values", default="1")
    parser.add_argument("--trajectory-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--estimated-task-memory-gib", type=float, default=1.0)
    parser.add_argument(
        "--partition",
        choices=("calibration", "holdout"),
        default="calibration",
    )
    parser.add_argument("--compiler-settings-json", type=Path)
    parser.add_argument("--worker-spec", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    return parser


def cli_main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.worker_spec is not None or args.worker_result is not None:
        if args.worker_spec is None or args.worker_result is None:
            raise SystemExit("Both --worker-spec and --worker-result are required.")
        return execute_worker_spec(args.worker_spec, args.worker_result)
    project_root = Path(__file__).resolve().parents[2]
    if args.create_hadamard_manifest is not None:
        if args.snapshot is None:
            raise SystemExit("--snapshot is required with --create-hadamard-manifest.")
        compiler = _default_compiler_settings()
        if args.compiler_settings_json is not None:
            compiler_path = _resolve_repo_file(
                args.compiler_settings_json,
                project_root=project_root,
                name="compiler_settings_json",
            )
            compiler = json.loads(compiler_path.read_text(encoding="utf-8"))
        manifest = create_hadamard_task_manifest(
            args.create_hadamard_manifest,
            project_root=project_root,
            batch_id=args.batch_id,
            validation_id=args.validation_id,
            snapshot_path=args.snapshot,
            ld=args.ld,
            delta=args.delta,
            r=args.r,
            k=args.finite_taylor_order,
            q_values=_comma_separated_ints(args.q_values, name="q-values"),
            trajectory_count=args.trajectory_count,
            seed=args.seed,
            compiler_settings=compiler,
            estimated_memory_gib=args.estimated_task_memory_gib,
            partition=args.partition,
        )
        print(
            json.dumps(
                {
                    "created_manifest": manifest["manifest_path"],
                    "manifest_fingerprint": manifest["manifest_fingerprint"],
                    "task_count": len(manifest["tasks"]),
                    "task_ids": [task["task_id"] for task in manifest["tasks"]],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.task_manifest is None:
        raise SystemExit("--task-manifest is required.")
    manifest = load_task_manifest(args.task_manifest, project_root=project_root)
    output_dir = args.output_dir or Path(
        "artifacts/parallel_validation_execution"
    ) / manifest["batch_id"]
    if args.status:
        print(
            json.dumps(
                inspect_batch_status(
                    manifest=manifest,
                    output_dir=output_dir,
                    project_root=project_root,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    affinity = parse_cpu_affinity(args.cpu_affinity)
    gpu_ids = tuple(item.strip() for item in args.gpu_ids.split(",") if item.strip())
    plan = plan_resources(
        manifest["tasks"],
        max_workers=args.max_workers,
        memory_budget_gib=args.memory_budget_gib,
        cpu_affinity=affinity,
        gpu_ids=gpu_ids,
        allow_more_than_16_workers=args.allow_more_than_16_workers,
    )
    report = dry_run_report(
        manifest=manifest,
        output_dir=output_dir,
        project_root=project_root,
        resource_plan=plan,
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return 0
    executor = ParallelValidationExecutor(
        manifest=manifest,
        output_dir=output_dir,
        project_root=project_root,
        resource_plan=plan,
        resume=args.resume,
    )
    summary = executor.run()
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["state"] == "interrupted":
        return 130 if summary["signal"] == signal.SIGINT else 143
    return 0 if summary["state"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(cli_main())
