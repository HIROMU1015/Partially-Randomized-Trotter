from __future__ import annotations

import json
import math
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import trotterlib.parallel_validation_executor as executor_module
from trotterlib.parallel_validation_executor import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointError,
    ManifestError,
    ParallelValidationExecutor,
    TASK_MANIFEST_SCHEMA_VERSION,
    atomic_write_json,
    build_deterministic_aggregate,
    create_hadamard_task_manifest,
    fingerprint_payload,
    load_task_manifest,
    plan_resources,
    read_checkpoint,
    trajectory_seed,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
H4_SNAPSHOT = Path(
    "artifacts/rte_connected_cluster_cost_validation/"
    "h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k2_connected_"
    "pilot30_max1500_hold1500_rare375_v1.hamiltonian.npz"
)
ADAPTER_SOURCES = (
    "src/trotterlib/parallel_validation_executor.py",
    "src/trotterlib/df_rpe_hadamard_compiled_cost.py",
    "src/trotterlib/rpe_hadamard_compiled_cost_benchmark.py",
    "src/trotterlib/df_partial_s2.py",
    "src/trotterlib/rte.py",
)


def _make_project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    for relative in ADAPTER_SOURCES:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        source = (
            Path(executor_module.__file__)
            if relative.endswith("parallel_validation_executor.py")
            else REPO_ROOT / relative
        )
        shutil.copy2(source, destination)
    (root / "artifacts").mkdir()
    (root / "input.txt").write_text("stable input\n", encoding="utf-8")
    return root


def _task(
    index: int,
    *,
    value: float = 1.0,
    sleep_seconds: float = 0.0,
    fail_until_attempt: int = 0,
    resource: str = "cpu",
) -> dict[str, object]:
    return {
        "validation_id": "tiny-regression",
        "adapter": "synthetic",
        "resource": resource,
        "ld": 1,
        "delta": 0.01,
        "r": 1,
        "k": 2,
        "q": 1,
        "trajectory_index": index,
        "seed": 1000,
        "compiler_settings": {"name": "none"},
        "input_paths": ["input.txt"],
        "estimated_memory_gib": 0.25,
        "parameters": {
            "value": value,
            "sleep_seconds": sleep_seconds,
            "fail_until_attempt": fail_until_attempt,
        },
    }


def _write_manifest(
    root: Path,
    tasks: list[dict[str, object]],
    *,
    batch_id: str = "tiny-batch",
    name: str = "manifest.json",
) -> Path:
    path = root / name
    path.write_text(
        json.dumps(
            {
                "schema_version": TASK_MANIFEST_SCHEMA_VERSION,
                "batch_id": batch_id,
                "source_paths": [],
                "tasks": tasks,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _load(root: Path, path: Path) -> dict[str, object]:
    return load_task_manifest(path, project_root=root)


def _run(
    root: Path,
    manifest: dict[str, object],
    output_name: str,
    *,
    workers: int = 1,
    resume: bool = False,
    gpu_ids: tuple[str, ...] = (),
) -> tuple[ParallelValidationExecutor, dict[str, object]]:
    plan = plan_resources(
        manifest["tasks"],
        max_workers=workers,
        memory_budget_gib=4.0,
        gpu_ids=gpu_ids,
        available_cpu_ids=tuple(range(8)),
    )
    executor = ParallelValidationExecutor(
        manifest=manifest,
        output_dir=root / "artifacts" / output_name,
        project_root=root,
        resource_plan=plan,
        resume=resume,
        poll_interval_seconds=0.01,
        termination_grace_seconds=0.5,
    )
    return executor, executor.run()


def _checkpoint(root: Path, output_name: str, task_id: str) -> Path:
    return root / "artifacts" / output_name / "checkpoints" / f"{task_id}.json"


def test_task_fingerprint_is_deterministic_and_binds_inputs(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    first = _load(root, _write_manifest(root, [_task(1), _task(0)]))
    second = _load(
        root,
        _write_manifest(
            root,
            [_task(0), _task(1)],
            name="manifest-reordered.json",
        ),
    )
    assert first["manifest_fingerprint"] == second["manifest_fingerprint"]
    assert [task["task_id"] for task in first["tasks"]] == [
        task["task_id"] for task in second["tasks"]
    ]

    (root / "input.txt").write_text("changed input\n", encoding="utf-8")
    changed = _load(
        root,
        _write_manifest(root, [_task(0), _task(1)], name="manifest-changed.json"),
    )
    assert {task["task_id"] for task in first["tasks"]}.isdisjoint(
        task["task_id"] for task in changed["tasks"]
    )
    assert trajectory_seed(1000, 0) != trajectory_seed(1000, 1)


def test_atomic_json_write_leaves_no_temporary_file(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    atomic_write_json(path, {"value": 1})
    atomic_write_json(path, {"value": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 2}
    assert list(tmp_path.glob(".checkpoint.json.tmp.*")) == []


def test_resume_skips_completed_task(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    manifest = _load(root, _write_manifest(root, [_task(0)]))
    first_executor, first = _run(root, manifest, "resume")
    task_id = manifest["tasks"][0]["task_id"]
    checkpoint_path = _checkpoint(root, "resume", task_id)
    checkpoint_before = checkpoint_path.read_bytes()
    assert first["state"] == "completed"
    assert first_executor.launched_pids
    log_dir = root / "artifacts/resume/logs"
    assert len(list(log_dir.glob("*.log"))) == 1
    assert list(log_dir.glob(".*.tmp.*")) == []

    second_executor, second = _run(root, manifest, "resume", resume=True)
    assert second["state"] == "completed"
    assert second_executor.launched_pids == []
    assert checkpoint_path.read_bytes() == checkpoint_before


def test_corrupt_checkpoint_is_rejected_without_overwrite(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    manifest = _load(root, _write_manifest(root, [_task(0)]))
    _run(root, manifest, "corrupt")
    task_id = manifest["tasks"][0]["task_id"]
    checkpoint_path = _checkpoint(root, "corrupt", task_id)
    checkpoint_path.write_text("{broken", encoding="utf-8")

    with pytest.raises(CheckpointError, match="Corrupt checkpoint"):
        _run(root, manifest, "corrupt", resume=True)
    assert checkpoint_path.read_text(encoding="utf-8") == "{broken"


def test_mismatched_manifest_checkpoint_is_rejected(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    first = _load(root, _write_manifest(root, [_task(0, value=1.0)]))
    _run(root, first, "mismatch")
    second = _load(
        root,
        _write_manifest(root, [_task(0, value=2.0)], name="manifest-second.json"),
    )
    with pytest.raises(CheckpointError, match="not present in this manifest"):
        _run(root, second, "mismatch", resume=True)


def _complete_checkpoint(task_id: str, value: float) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "artifact_kind": "execution_compute_checkpoint",
        "scientific_verdict_included": False,
        "task_id": task_id,
        "task_fingerprint": task_id,
        "status": "completed",
        "attempt": 1,
        "pid": 1,
        "started_at_utc": "2026-01-01T00:00:00+00:00",
        "finished_at_utc": "2026-01-01T00:00:01+00:00",
        "assigned_gpu_id": None,
        "affinity_cpu_id": None,
        "provenance": {},
        "result": {"metrics": {"value": value}, "runtime_pid": value + 10},
        "error": None,
    }
    return {**body, "checkpoint_fingerprint": fingerprint_payload(body)}


def test_deterministic_aggregate_ignores_completion_and_runtime_order() -> None:
    first = _complete_checkpoint("a" * 64, 1.0)
    second = _complete_checkpoint("b" * 64, 3.0)
    forward = build_deterministic_aggregate([first, second])
    reverse = build_deterministic_aggregate([second, first])
    assert forward == reverse
    summary = forward["metric_summaries"]["value"]
    assert summary["mean"] == 2.0
    assert summary["standard_error"] == 1.0


def test_resource_guard_limits_workers_memory_and_gpu_use() -> None:
    tasks = [_task(index) for index in range(40)]
    plan = plan_resources(
        tasks,
        memory_budget_gib=2.0,
        available_cpu_ids=tuple(range(40)),
    )
    assert plan.worker_count == 8
    assert plan.worker_count <= len(plan.detected_cpu_ids) // 4
    assert plan.estimated_peak_memory_gib == 2.0
    pinned = plan_resources(
        tasks,
        memory_budget_gib=32.0,
        cpu_affinity=(2, 3),
        available_cpu_ids=tuple(range(40)),
    )
    assert pinned.worker_count == 2
    with pytest.raises(ManifestError, match="More than 16"):
        plan_resources(
            tasks,
            max_workers=17,
            memory_budget_gib=32.0,
            available_cpu_ids=tuple(range(40)),
        )
    with pytest.raises(ManifestError, match="memory budget"):
        plan_resources(
            [{**_task(0), "estimated_memory_gib": 4.0}],
            memory_budget_gib=2.0,
            available_cpu_ids=tuple(range(8)),
        )
    with pytest.raises(ManifestError, match="explicit --gpu-ids"):
        plan_resources(
            [_task(0, resource="gpu")],
            memory_budget_gib=2.0,
            available_cpu_ids=tuple(range(8)),
        )


def test_failed_task_is_the_only_task_retried(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    tasks = [_task(0), _task(1, fail_until_attempt=1)]
    manifest = _load(root, _write_manifest(root, tasks))
    first_executor, first = _run(root, manifest, "retry", workers=2)
    assert first["state"] == "failed"
    assert len(first_executor.launched_pids) == 2

    second_executor, second = _run(root, manifest, "retry", workers=2, resume=True)
    assert second["state"] == "completed"
    assert len(second_executor.launched_pids) == 1
    checkpoints = [
        read_checkpoint(_checkpoint(root, "retry", task["task_id"]), task=task)
        for task in manifest["tasks"]
    ]
    assert sorted(checkpoint["attempt"] for checkpoint in checkpoints) == [1, 2]


def test_sequential_and_parallel_tiny_results_match(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    tasks = [_task(index, value=float(index + 1)) for index in range(4)]
    manifest = _load(root, _write_manifest(root, tasks))
    _, sequential = _run(root, manifest, "sequential", workers=1)
    _, parallel = _run(root, manifest, "parallel", workers=2)
    assert sequential["state"] == parallel["state"] == "completed"
    assert sequential["aggregate_fingerprint"] == parallel["aggregate_fingerprint"]


def test_gpu_dispatcher_requires_ids_and_exposes_one_selected_id(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    manifest = _load(
        root,
        _write_manifest(
            root,
            [
                _task(0, resource="gpu", sleep_seconds=0.05),
                _task(1, resource="gpu", sleep_seconds=0.05),
            ],
        ),
    )
    _, summary = _run(root, manifest, "gpu-dispatch", workers=2, gpu_ids=("3",))
    assert summary["state"] == "completed"
    for task in manifest["tasks"]:
        checkpoint = read_checkpoint(
            _checkpoint(root, "gpu-dispatch", task["task_id"]),
            task=task,
        )
        assert checkpoint["assigned_gpu_id"] == "3"
        assert checkpoint["result"]["cuda_visible_devices"] == "3"


def test_sigint_stops_only_executor_children(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    manifest = _load(
        root,
        _write_manifest(root, [_task(0, sleep_seconds=5.0), _task(1)]),
    )
    plan = plan_resources(
        manifest["tasks"],
        max_workers=1,
        memory_budget_gib=4.0,
        available_cpu_ids=tuple(range(8)),
    )
    executor = ParallelValidationExecutor(
        manifest=manifest,
        output_dir=root / "artifacts/signal",
        project_root=root,
        resource_plan=plan,
        poll_interval_seconds=0.01,
        termination_grace_seconds=0.2,
    )
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    def send_sigint() -> None:
        deadline = time.monotonic() + 3.0
        while not executor.launched_pids and time.monotonic() < deadline:
            time.sleep(0.01)
        os.kill(os.getpid(), signal.SIGINT)

    interrupter = threading.Thread(target=send_sigint, daemon=True)
    interrupter.start()
    try:
        summary = executor.run()
        interrupter.join(timeout=1.0)
        assert summary["state"] == "interrupted"
        assert summary["counts"]["pending"] == 1
        assert unrelated.poll() is None
        assert executor.terminated_pids == executor.launched_pids
        assert all(not Path(f"/proc/{pid}").exists() for pid in executor.launched_pids)
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=2.0)


@pytest.mark.slow
def test_h4_q1_full_wrapper_single_trajectory(tmp_path: Path) -> None:
    snapshot_source = REPO_ROOT / H4_SNAPSHOT
    if not snapshot_source.exists():
        pytest.skip("The tracked H4 Hamiltonian snapshot is unavailable.")
    root = tmp_path / "h4-project"
    (root / "artifacts").mkdir(parents=True)
    for relative in ADAPTER_SOURCES:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / relative, destination)
    snapshot = root / H4_SNAPSHOT
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(snapshot_source, snapshot)
    task = {
        "validation_id": "h4-q1-full-wrapper-regression",
        "adapter": "rpe_hadamard_full_wrapper",
        "resource": "cpu",
        "ld": 3,
        "delta": 0.1,
        "r": 4,
        "k": 2,
        "q": 1,
        "trajectory_index": 0,
        "seed": 20260923,
        "compiler_settings": {
            "basis_gates": ["rz", "sx", "x", "cx"],
            "backend_name": None,
            "coupling_map": None,
            "optimization_level": 1,
            "layout_method": None,
            "routing_method": None,
            "transpiler_seed": 17,
        },
        "input_paths": [H4_SNAPSHOT.as_posix()],
        "estimated_memory_gib": 1.0,
        "parameters": {
            "snapshot_path": H4_SNAPSHOT.as_posix(),
            "partition": "calibration",
        },
    }
    manifest = _load(root, _write_manifest(root, [task], batch_id="h4-q1"))
    _, summary = _run(root, manifest, "h4-q1", workers=1)
    assert summary["state"] == "completed"
    task_spec = manifest["tasks"][0]
    checkpoint = read_checkpoint(
        _checkpoint(root, "h4-q1", task_spec["task_id"]),
        task=task_spec,
    )
    assert checkpoint["result"]["paired_axes"] == ["cosine", "sine"]
    assert set(checkpoint["result"]["metrics"]) == {
        f"{axis}_{metric}"
        for axis in ("cosine", "sine")
        for metric in (
            "rz_count",
            "rz_depth",
            "cx_count",
            "cx_depth",
            "total_depth",
            "circuit_size",
        )
    }
    assert all(
        math.isfinite(value) for value in checkpoint["result"]["metrics"].values()
    )


def test_hadamard_manifest_creation_is_bounded_and_immutable(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    output = root / "artifacts/generated-manifest.json"
    compiler = {
        "basis_gates": ["rz", "sx", "x", "cx"],
        "backend_name": None,
        "coupling_map": None,
        "optimization_level": 1,
        "layout_method": None,
        "routing_method": None,
        "transpiler_seed": 17,
    }
    manifest = create_hadamard_task_manifest(
        output,
        project_root=root,
        batch_id="generated",
        validation_id="generated-full-wrapper",
        snapshot_path="input.txt",
        ld=3,
        delta=0.1,
        r=4,
        k=2,
        q_values=(1, 2),
        trajectory_count=2,
        seed=1234,
        compiler_settings=compiler,
        estimated_memory_gib=1.5,
    )
    assert len(manifest["tasks"]) == 4
    assert all(task["resource"] == "cpu" for task in manifest["tasks"])
    with pytest.raises(ManifestError, match="Refusing to replace"):
        create_hadamard_task_manifest(
            output,
            project_root=root,
            batch_id="generated",
            validation_id="generated-full-wrapper",
            snapshot_path="input.txt",
            ld=3,
            delta=0.1,
            r=4,
            k=2,
            q_values=(1,),
            trajectory_count=1,
            seed=1234,
            compiler_settings=compiler,
        )


def test_resume_preserves_orphan_attempt_and_uses_next_number(tmp_path: Path) -> None:
    root = _make_project(tmp_path)
    manifest = _load(root, _write_manifest(root, [_task(0)]))
    task_id = manifest["tasks"][0]["task_id"]
    task_dir = root / "artifacts/orphan/tasks"
    task_dir.mkdir(parents=True)
    orphan = task_dir / f"{task_id}.attempt-0001.json"
    orphan.write_text("{\"orphan\": true}\n", encoding="utf-8")

    with pytest.raises(CheckpointError, match="requires --resume"):
        _run(root, manifest, "orphan")
    _, summary = _run(root, manifest, "orphan", resume=True)
    assert summary["state"] == "completed"
    assert orphan.read_text(encoding="utf-8") == "{\"orphan\": true}\n"
    assert (task_dir / f"{task_id}.attempt-0002.json").exists()
