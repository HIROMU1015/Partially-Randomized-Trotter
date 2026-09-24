"""M06-F all-r coherent optimization-level-2 compute and analysis support."""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import qiskit

from .df_partial_randomized_pf import split_df_hamiltonian_by_ld
from .df_partial_s2 import prepare_df_partial_s2
from .df_partial_s2_repeated_cost import (
    make_monte_carlo_df_partial_s2_repeated_trajectory_stream,
)
from .parallel_validation_executor import (
    TASK_MANIFEST_SCHEMA_VERSION,
    _compiler_from_task,
    atomic_write_json,
    file_sha256,
    load_task_manifest,
    read_checkpoint,
    validate_output_directory,
)
from .research_direction_compiler_transfer_analysis import (
    validate_compiler_transfer_analysis_artifact,
)
from .research_direction_compiler_transfer_compute import (
    _compile_deterministic_point,
    validate_compiler_transfer_compute_artifact,
)
from .research_direction_decision_cost import (
    _candidate_configuration,
    _grid_search,
    _pair_model,
)
from .research_direction_full_scope import (
    AXES,
    METRICS,
    POLICY_LABEL,
    _compile_paired_requests,
    _derived_seed,
    _finite_inputs,
    _materialize_stream,
    fingerprint,
    validate_wp05a_artifact,
)
from .research_direction_full_scope_extension import (
    fit_affine_holdouts,
    validate_wp05b_artifact,
)
from .research_direction_prevalidation import validate_artifact
from .research_direction_sequence_policy import (
    register_support_restricted_bases,
    validate_wp06b_artifact,
)
from .research_direction_wp11_synthesis import validate_wp11_artifact
from .rte_connected_cluster_cost_validation import (
    load_connected_cluster_hamiltonian_snapshot,
)


SCHEMA_VERSION = "research_direction_full_opt2_analysis_v1"
METHOD = "m06f_all_r_coherent_optimization_level_2_reoptimization_v1"
EXPECTED_WP11_FINGERPRINT = (
    "45def7f696eddba574878cc7530837dfdfc5c6e9c2767ee3e115cf4a5f1ac092"
)
EXPECTED_COMPILER_RAW_FINGERPRINT = (
    "87f66c8944dedfaa9fe5f0edf864d2a2a07cb82e23245af4f5944f90bda83ad6"
)
EXPECTED_SNAPSHOT_SHA256 = (
    "13e4b10d2347ed900fe4aa4b2238128ebe735a48988e321b9e4fabac4d092a3b"
)
EXPECTED_WP06B_FINGERPRINT = (
    "6b9ec62e620f58ff5654de42a33085744e6b52efc49969a90eb58f001482f31a"
)
SNAPSHOT_PATH = (
    "artifacts/rte_connected_cluster_cost_validation/"
    "h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k2_connected_"
    "pilot30_max1500_hold1500_rare375_v1.hamiltonian.npz"
)
WP01_PATH = (
    "artifacts/research_direction_prevalidation/2026-09-21/"
    "wp01s_model_conditional_screening_v1.json"
)
WP05A_PATH = (
    "artifacts/research_direction_full_scope/2026-09-22/"
    "wp05a_full_controlled_interrogation_connection_v1.json"
)
WP05B_PATH = (
    "artifacts/research_direction_full_scope_extension/2026-09-22/"
    "wp05b_q8_delta_0p01_full_scope_extension_v1.json"
)
WP06B_PATH = (
    "artifacts/research_direction_sequence_policy/2026-09-22/"
    "wp06b_sequence_policy_proxy_bridge_v1.json"
)
WP11_PATH = (
    "artifacts/research_direction_wp11_synthesis/2026-09-23/"
    "wp11_scoped_direction_synthesis_v1.json"
)
COMPILER_RAW_PATH = (
    "artifacts/research_direction_compiler_transfer/2026-09-23/"
    "m06_l08_opt2_same_trajectory_compute_v1.json"
)
COMPILER_ANALYSIS_PATH = (
    "artifacts/research_direction_compiler_transfer/2026-09-23/"
    "m06_l08_opt2_focused_analysis_reaggregation_v1.json"
)
RTE_STEPS = (1, 2, 4, 8, 16, 32)
CALIBRATION_Q = (1, 2)
HOLDOUT_Q = (8,)
INITIAL_SAMPLE_COUNT = 8
EXTENSION_SAMPLE_COUNT = 32
EXTENSION_MASTER_SEED = 2026092402
EXPECTED_COMPILER = {
    "basis_gates": ["rz", "sx", "x", "cx"],
    "backend_name": None,
    "coupling_map": None,
    "optimization_level": 2,
    "layout_method": None,
    "routing_method": None,
    "transpiler_seed": 17,
}
THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)
SOURCE_PATHS = (
    "src/trotterlib/research_direction_full_opt2.py",
    "src/trotterlib/research_direction_full_scope.py",
    "src/trotterlib/research_direction_decision_cost.py",
    "src/trotterlib/research_direction_sequence_policy.py",
    "src/trotterlib/parallel_validation_executor.py",
    "src/trotterlib/df_partial_s2_repeated.py",
    "src/trotterlib/df_partial_s2_repeated_cost.py",
    "src/trotterlib/rpe_hadamard_compiled_cost_benchmark.py",
    "src/trotterlib/rte_compiled_cost.py",
    "scripts/run_research_direction_full_opt2_compute.py",
)
INPUT_PATHS = (
    SNAPSHOT_PATH,
    WP05A_PATH,
    WP05B_PATH,
    WP06B_PATH,
    WP11_PATH,
    COMPILER_RAW_PATH,
)


def _read_json(root: Path, relative: str) -> dict[str, Any]:
    return json.loads((root / relative).read_text(encoding="utf-8"))


def _compiler_without_version(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: payload.get(key) for key in EXPECTED_COMPILER}


def load_and_validate_full_opt2_evidence(
    project_root: str | Path,
) -> dict[str, dict[str, Any]]:
    """Validate the fixed WP11 handoff and every reused compute input."""
    root = Path(project_root).resolve(strict=True)
    evidence = {
        "wp05a": _read_json(root, WP05A_PATH),
        "wp05b": _read_json(root, WP05B_PATH),
        "wp06b": _read_json(root, WP06B_PATH),
        "wp11": _read_json(root, WP11_PATH),
        "compiler_raw": _read_json(root, COMPILER_RAW_PATH),
    }
    validate_wp05a_artifact(evidence["wp05a"])
    validate_wp05b_artifact(evidence["wp05b"])
    validate_wp06b_artifact(evidence["wp06b"])
    validate_wp11_artifact(evidence["wp11"])
    validate_compiler_transfer_compute_artifact(evidence["compiler_raw"])
    if evidence["wp11"]["content_fingerprint"] != EXPECTED_WP11_FINGERPRINT:
        raise ValueError("WP11 content fingerprint does not match the handoff.")
    if evidence["wp11"]["current_decision"]["selected_next_followup"] != (
        "all_r_coherent_opt2_reoptimization"
    ):
        raise ValueError("WP11 did not select the expected next follow-up.")
    if evidence["wp06b"]["content_fingerprint"] != EXPECTED_WP06B_FINGERPRINT:
        raise ValueError("WP06-b policy artifact fingerprint changed.")
    raw = evidence["compiler_raw"]
    if raw["content_fingerprint"] != EXPECTED_COMPILER_RAW_FINGERPRINT:
        raise ValueError("Existing optimization-level-2 artifact changed.")
    if file_sha256(root / SNAPSHOT_PATH) != EXPECTED_SNAPSHOT_SHA256:
        raise ValueError("H4 Hamiltonian snapshot SHA-256 mismatch.")
    if raw["source_evidence"]["snapshot"]["sha256"] != EXPECTED_SNAPSHOT_SHA256:
        raise ValueError("Existing opt2 artifact used a different snapshot.")
    if raw["source_evidence"]["wp06b"]["content_fingerprint"] != (
        EXPECTED_WP06B_FINGERPRINT
    ):
        raise ValueError("Existing opt2 artifact used a different basis policy.")
    for record in raw["source_evidence"].values():
        path = root / record["path"]
        if file_sha256(path) != record["sha256"]:
            raise ValueError(f"Reused input SHA-256 changed: {record['path']}")
    config = raw["configuration"]
    if (
        _compiler_without_version(config["compiler"]) != EXPECTED_COMPILER
        or config["compiler"]["qiskit_version"] != "1.3.0"
        or float(config["delta_time"]) != 0.02
        or int(config["rte_steps"]) != 32
        or list(config["q_values"]) != [1, 2, 16, 32]
    ):
        raise ValueError("Existing r=32 opt2 artifact is not reusable.")
    if qiskit.__version__ != "1.3.0":
        raise ValueError("M06-F requires Qiskit 1.3.0.")
    wp05a_config = evidence["wp05a"]["configuration"]
    wp05b_config = evidence["wp05b"]["configuration"]
    if (
        int(wp05a_config["master_seed"]) != 2026092205
        or int(wp05a_config["sample_count_per_randomized_point"]) != 8
        or int(wp05b_config["master_seed"]) != 2026092206
        or int(wp05b_config["sample_count_per_randomized_point"]) != 8
    ):
        raise ValueError("WP05 trajectory seed contract changed.")
    return evidence


def initial_randomized_cells() -> tuple[tuple[float, int, int], ...]:
    cells = []
    for delta, rte_steps_values in (
        (0.01, RTE_STEPS),
        (0.02, RTE_STEPS[:-1]),
    ):
        for rte_steps in rte_steps_values:
            for q_m in (*CALIBRATION_Q, *HOLDOUT_Q):
                cells.append((delta, rte_steps, q_m))
    return tuple(cells)


def _matched_cell_seed(
    evidence: Mapping[str, Mapping[str, Any]],
    *,
    delta: float,
    rte_steps: int,
    q_m: int,
) -> tuple[int, str]:
    if delta == 0.02 and q_m in CALIBRATION_Q:
        master = int(evidence["wp05a"]["configuration"]["master_seed"])
        return (
            _derived_seed(master, "ld3", rte_steps, q_m),
            "WP05-a_same_physical_trajectory_stream",
        )
    master = int(evidence["wp05b"]["configuration"]["master_seed"])
    return (
        _derived_seed(master, "wp05b", delta, rte_steps, q_m),
        "WP05-b_same_physical_trajectory_stream",
    )


def build_full_opt2_task_definitions(
    evidence: Mapping[str, Mapping[str, Any]],
    *,
    extension_cells: Sequence[tuple[float, int, int]] | None = None,
    initial_cells: Sequence[tuple[float, int, int]] | None = None,
    initial_sample_count: int = INITIAL_SAMPLE_COUNT,
    include_deterministic: bool = True,
) -> list[dict[str, Any]]:
    """Build immutable cell tasks; one task preserves one canonical stream."""
    extension = extension_cells is not None
    if extension and initial_cells is not None:
        raise ValueError("Initial and extension cell selections are mutually exclusive.")
    cells = (
        tuple(extension_cells or ())
        if extension
        else tuple(initial_cells or initial_randomized_cells())
    )
    sample_count = (
        EXTENSION_SAMPLE_COUNT if extension else int(initial_sample_count)
    )
    if sample_count < 1:
        raise ValueError("initial_sample_count must be positive.")
    tasks: list[dict[str, Any]] = []
    for delta, rte_steps, q_m in cells:
        if extension:
            seed = _derived_seed(
                EXTENSION_MASTER_SEED,
                "m06f-fresh-extension",
                delta,
                rte_steps,
                q_m,
            )
            seed_source = "M06-F_fresh_32_trajectory_extension"
        else:
            seed, seed_source = _matched_cell_seed(
                evidence,
                delta=delta,
                rte_steps=rte_steps,
                q_m=q_m,
            )
        tasks.append(
            {
                "validation_id": "m06f-all-r-coherent-opt2-randomized-cell",
                "adapter": "python_callable",
                "resource": "cpu",
                "ld": 3,
                "delta": delta,
                "r": rte_steps,
                "k": 2,
                "q": q_m,
                "trajectory_index": 0,
                "seed": seed,
                "compiler_settings": EXPECTED_COMPILER,
                "input_paths": list(INPUT_PATHS),
                "source_paths": [],
                "estimated_memory_gib": 4.0,
                "parameters": {
                    "callable": (
                        "trotterlib.research_direction_full_opt2:"
                        "compute_full_opt2_randomized_cell"
                    ),
                    "snapshot_path": SNAPSHOT_PATH,
                    "wp06b_path": WP06B_PATH,
                    "sample_count": sample_count,
                    "partition": (
                        "calibration" if q_m in CALIBRATION_Q else "holdout"
                    ),
                    "seed_source": seed_source,
                    "expected_transpile_count": sample_count * len(AXES) * 2,
                    "maximum_repetition_count": 8,
                    "extension": extension,
                },
            }
        )
    if not extension and include_deterministic:
        for q_m in (*CALIBRATION_Q, *HOLDOUT_Q):
            tasks.append(
                {
                    "validation_id": "m06f-coherent-opt2-deterministic-cell",
                    "adapter": "python_callable",
                    "resource": "cpu",
                    "ld": 12,
                    "delta": 0.01,
                    "r": 0,
                    "k": 0,
                    "q": q_m,
                    "trajectory_index": 0,
                    "seed": 0,
                    "compiler_settings": EXPECTED_COMPILER,
                    "input_paths": list(INPUT_PATHS),
                    "source_paths": [],
                    "estimated_memory_gib": 4.0,
                    "parameters": {
                        "callable": (
                            "trotterlib.research_direction_full_opt2:"
                            "compute_full_opt2_deterministic_cell"
                        ),
                        "snapshot_path": SNAPSHOT_PATH,
                        "partition": (
                            "calibration" if q_m in CALIBRATION_Q else "holdout"
                        ),
                        "expected_transpile_count": len(AXES),
                        "maximum_repetition_count": 8,
                        "extension": False,
                    },
                }
            )
    return tasks


def create_full_opt2_task_manifest(
    path: str | Path,
    *,
    project_root: str | Path,
    batch_id: str,
    extension_cells: Sequence[tuple[float, int, int]] | None = None,
    initial_cells: Sequence[tuple[float, int, int]] | None = None,
    initial_sample_count: int = INITIAL_SAMPLE_COUNT,
    include_deterministic: bool = True,
) -> dict[str, Any]:
    root = Path(project_root).resolve(strict=True)
    destination = validate_output_directory(path, root)
    if destination.exists():
        raise ValueError(f"Refusing to replace existing manifest: {destination}")
    evidence = load_and_validate_full_opt2_evidence(root)
    payload = {
        "schema_version": TASK_MANIFEST_SCHEMA_VERSION,
        "batch_id": batch_id,
        "source_paths": list(SOURCE_PATHS),
        "tasks": build_full_opt2_task_definitions(
            evidence,
            extension_cells=extension_cells,
            initial_cells=initial_cells,
            initial_sample_count=initial_sample_count,
            include_deterministic=include_deterministic,
        ),
    }
    atomic_write_json(destination, payload)
    return load_task_manifest(destination, project_root=root)


def manifest_workload_summary(manifest: Mapping[str, Any]) -> dict[str, int]:
    tasks = list(manifest["tasks"])
    return {
        "cell_task_count": len(tasks),
        "randomized_cell_task_count": sum(int(task["ld"]) == 3 for task in tasks),
        "deterministic_cell_task_count": sum(
            int(task["ld"]) == 12 for task in tasks
        ),
        "direct_transpile_count": sum(
            int(task["parameters"]["expected_transpile_count"])
            for task in tasks
        ),
    }


def _require_worker_environment() -> dict[str, str]:
    values = {name: os.environ.get(name, "") for name in THREAD_ENVIRONMENT}
    if any(value != "1" for value in values.values()):
        raise RuntimeError("Every scientific worker thread limit must equal 1.")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("M06-F workers require CUDA_VISIBLE_DEVICES to be empty.")
    return values


def _worker_preparation(
    task: Mapping[str, Any], *, ld: int
) -> tuple[Any, Any, Any, Path]:
    root = Path(task["project_root"]).resolve(strict=True)
    snapshot = root / str(task["parameters"]["snapshot_path"])
    if file_sha256(snapshot) != EXPECTED_SNAPSHOT_SHA256:
        raise ValueError("Worker snapshot SHA-256 mismatch.")
    hamiltonian = load_connected_cluster_hamiltonian_snapshot(snapshot)
    preparation = prepare_df_partial_s2(
        hamiltonian,
        split_df_hamiltonian_by_ld(hamiltonian, ld),
        identity_policy="extract_identity_phase",
    )
    compiler = _compiler_from_task(task)
    if _compiler_without_version(task["compiler_settings"]) != EXPECTED_COMPILER:
        raise ValueError("Worker compiler context changed.")
    return hamiltonian, preparation, compiler, root


def _dependency_versions() -> dict[str, str]:
    return {
        "python": os.sys.version.split()[0],
        "qiskit": importlib.metadata.version("qiskit"),
        "numpy": importlib.metadata.version("numpy"),
        "scipy": importlib.metadata.version("scipy"),
    }


def compute_full_opt2_randomized_cell(
    task: Mapping[str, Any], *, attempt: int
) -> dict[str, Any]:
    """Compile one (delta,r,q) cell while preserving its canonical stream."""
    started = time.monotonic()
    thread_environment = _require_worker_environment()
    hamiltonian, preparation, compiler, root = _worker_preparation(task, ld=3)
    wp06b = _read_json(root, str(task["parameters"]["wp06b_path"]))
    validate_wp06b_artifact(wp06b)
    if wp06b["content_fingerprint"] != EXPECTED_WP06B_FINGERPRINT:
        raise ValueError("Worker policy artifact fingerprint mismatch.")
    support_definitions, proof_records = register_support_restricted_bases(
        hamiltonian, preparation
    )
    delta = float(task["delta"])
    rte_steps = int(task["r"])
    q_m = int(task["q"])
    sample_count = int(task["parameters"]["sample_count"])
    config, distribution = _finite_inputs(
        preparation,
        delta_time=delta,
        rte_steps=rte_steps,
        finite_taylor_order=2,
    )
    stream = make_monte_carlo_df_partial_s2_repeated_trajectory_stream(
        preparation,
        delta,
        q_m,
        config,
        distribution,
        sample_count=sample_count,
        seed=int(task["seed"]),
        maximum_samples=sample_count,
        controlled=True,
        ancilla_qubit=preparation.num_system_qubits,
        construction_policy="boundary_optimized",
    )
    point, _probe = _compile_paired_requests(
        _materialize_stream(stream),
        support_definitions,
        compiler,
        training_fingerprint=str(
            wp06b["training_selection"]["training_fingerprint"]
        ),
        maximum_repetition_count=int(
            task["parameters"]["maximum_repetition_count"]
        ),
        operator_probe_requested=False,
        retain_trajectory_records=True,
    )
    trajectory_seeds = [
        int(record["trajectory_seed"]) for record in point["trajectory_records"]
    ]
    if len(trajectory_seeds) != sample_count or len(set(trajectory_seeds)) != sample_count:
        raise RuntimeError("Trajectory records are missing or duplicated.")
    proof_residual = max(
        float(record["preserved_columns_max_abs_residual"])
        for record in proof_records
    )
    if proof_residual > 1e-10:
        raise RuntimeError("Support-restricted basis certificate failed.")
    return {
        "adapter": "m06f_all_r_coherent_opt2_randomized_cell",
        "attempt": attempt,
        "cell": {
            "ld": 3,
            "delta": delta,
            "r": rte_steps,
            "k": 2,
            "q": q_m,
            "partition": task["parameters"]["partition"],
            "sample_count": sample_count,
            "stream_seed": int(task["seed"]),
            "seed_source": task["parameters"]["seed_source"],
            "trajectory_seed_policy": "monte_carlo_master_trajectory_step_v1",
        },
        "compiler": {**dict(task["compiler_settings"]), "qiskit_version": "1.3.0"},
        "point": point,
        "support_basis_max_abs_residual": proof_residual,
        "dependency_versions": _dependency_versions(),
        "thread_environment": thread_environment,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "input_sha256": dict(task["input_sha256"]),
        "source_sha256": dict(task["source_sha256"]),
        "elapsed_seconds": time.monotonic() - started,
        "scientific_verdict_included": False,
    }


def compute_full_opt2_deterministic_cell(
    task: Mapping[str, Any], *, attempt: int
) -> dict[str, Any]:
    started = time.monotonic()
    thread_environment = _require_worker_environment()
    _hamiltonian, preparation, compiler, _root = _worker_preparation(task, ld=12)
    q_m = int(task["q"])
    point = _compile_deterministic_point(
        preparation,
        compiler,
        delta_time=float(task["delta"]),
        q_m=q_m,
        maximum_repetition_count=int(
            task["parameters"]["maximum_repetition_count"]
        ),
    )
    return {
        "adapter": "m06f_coherent_opt2_deterministic_cell",
        "attempt": attempt,
        "cell": {
            "ld": 12,
            "delta": float(task["delta"]),
            "r": 0,
            "k": 0,
            "q": q_m,
            "partition": task["parameters"]["partition"],
            "sample_count": 1,
        },
        "compiler": {**dict(task["compiler_settings"]), "qiskit_version": "1.3.0"},
        "point": point,
        "dependency_versions": _dependency_versions(),
        "thread_environment": thread_environment,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "input_sha256": dict(task["input_sha256"]),
        "source_sha256": dict(task["source_sha256"]),
        "elapsed_seconds": time.monotonic() - started,
        "scientific_verdict_included": False,
    }


def _point_metric(
    point: Mapping[str, Any], *, axis: str, metric: str, policy: str | None
) -> Mapping[str, Any]:
    if policy is None:
        return point["axes"][axis][metric]
    return point["axes"][axis]["policies"][policy][metric]


def evaluate_proxy_cell(
    points: Mapping[int, Mapping[str, Any]],
    *,
    holdout_q: Sequence[int],
) -> dict[str, Any]:
    selected = fit_affine_holdouts(
        points, policy=POLICY_LABEL, holdout_q=holdout_q
    )
    full = fit_affine_holdouts(
        points, policy="full_basis_shared", holdout_q=holdout_q
    )
    rows: list[dict[str, Any]] = []
    for policy, models in ((POLICY_LABEL, selected), ("full_basis_shared", full)):
        for axis in AXES:
            for metric in METRICS:
                for q_text, holdout in models[axis][metric]["holdouts"].items():
                    rows.append(
                        {
                            "policy": policy,
                            "axis": axis,
                            "metric": metric,
                            "q": int(q_text),
                            "absolute_relative_error": float(
                                holdout["absolute_relative_error"]
                            ),
                        }
                    )
    relative_se_rows = []
    for q_m, point in points.items():
        for policy in (POLICY_LABEL, "full_basis_shared"):
            for axis in AXES:
                record = _point_metric(
                    point, axis=axis, metric="rz_count", policy=policy
                )
                mean = float(record["mean"])
                relative_se_rows.append(
                    {
                        "policy": policy,
                        "axis": axis,
                        "q": q_m,
                        "relative_standard_error": (
                            0.0
                            if mean == 0.0 and float(record["standard_error"]) == 0.0
                            else abs(float(record["standard_error"]) / mean)
                        ),
                    }
                )
    selected_rz = max(
        row["absolute_relative_error"]
        for row in rows
        if row["policy"] == POLICY_LABEL and row["metric"] == "rz_count"
    )
    selected_all = max(
        row["absolute_relative_error"]
        for row in rows
        if row["policy"] == POLICY_LABEL
    )
    full_rz = max(
        row["absolute_relative_error"]
        for row in rows
        if row["policy"] == "full_basis_shared" and row["metric"] == "rz_count"
    )
    direct_rz_se = max(
        row["relative_standard_error"] for row in relative_se_rows
    )
    checks = {
        "selected_rz_holdout_within_5_percent": selected_rz <= 0.05,
        "selected_all_metrics_holdout_within_5_percent": selected_all <= 0.05,
        "direct_rz_relative_se_within_2_percent": direct_rz_se <= 0.02,
    }
    return {
        "holdout_q": list(holdout_q),
        "maximum_selected_rz_holdout_error": selected_rz,
        "maximum_selected_all_metrics_holdout_error": selected_all,
        "maximum_full_basis_rz_holdout_error": full_rz,
        "maximum_direct_rz_relative_standard_error": direct_rz_se,
        "holdout_rows": rows,
        "direct_rz_relative_standard_error_rows": relative_se_rows,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _collect_completed_points(
    manifest: Mapping[str, Any], output_dir: Path
) -> dict[tuple[int, float, int], dict[int, Any]]:
    points: dict[tuple[int, float, int], dict[int, Any]] = {}
    checkpoint_dir = output_dir / "checkpoints"
    for task in manifest["tasks"]:
        path = checkpoint_dir / f"{task['task_id']}.json"
        if not path.exists():
            raise RuntimeError(f"Compute is incomplete; missing checkpoint: {path.name}")
        checkpoint = read_checkpoint(path, task=task)
        if checkpoint["status"] != "completed":
            raise RuntimeError(
                f"Compute is incomplete; task status is {checkpoint['status']}."
            )
        key = (int(task["ld"]), float(task["delta"]), int(task["r"]))
        points.setdefault(key, {})[int(task["q"])] = checkpoint["result"]["point"]
    return points


def _reused_points(
    raw: Mapping[str, Any],
) -> dict[tuple[int, float, int], dict[int, Any]]:
    return {
        (3, 0.02, 32): {
            int(q): point for q, point in raw["direct_points"]["3"].items()
        },
        (12, 0.02, 0): {
            int(q): point for q, point in raw["direct_points"]["12"].items()
        },
    }


def _deterministic_diagnostic(
    points: Mapping[int, Mapping[str, Any]], *, holdout_q: Sequence[int]
) -> dict[str, Any]:
    models = fit_affine_holdouts(points, policy=None, holdout_q=holdout_q)
    errors = [
        float(models[axis][metric]["holdouts"][str(q_m)]["absolute_relative_error"])
        for axis in AXES
        for metric in METRICS
        for q_m in holdout_q
    ]
    return {
        "holdout_q": list(holdout_q),
        "maximum_all_metrics_holdout_error": max(errors),
        "maximum_rz_holdout_error": max(
            float(models[axis]["rz_count"]["holdouts"][str(q_m)]["absolute_relative_error"])
            for axis in AXES
            for q_m in holdout_q
        ),
    }


def _measured_model_half_width(
    best: Mapping[str, Any],
    diagnostics: Mapping[str, Mapping[str, Any]],
    *,
    policy: str,
    delta: float,
) -> float:
    field = (
        "maximum_selected_rz_holdout_error"
        if policy == POLICY_LABEL
        else "maximum_full_basis_rz_holdout_error"
    )
    return math.fsum(
        float(row["compiled_rz_point_estimate"])
        * float(diagnostics[f"delta{delta:g}:r{int(row['r_m'])}"][field])
        for row in best["rounds"]
    )


def _interval(cost: float, half_width: float) -> list[float]:
    return [max(0.0, cost - half_width), cost + half_width]


def _comparison(
    ld3_best: Mapping[str, Any],
    ld12_best: Mapping[str, Any],
    diagnostics: Mapping[str, Mapping[str, Any]],
    deterministic: Mapping[str, Mapping[str, Any]],
    *,
    delta3: float,
    delta12: float,
) -> dict[str, Any]:
    cost3 = float(ld3_best["total_compiled_rz_point_estimate"])
    cost12 = float(ld12_best["total_compiled_rz_point_estimate"])
    sampling3 = float(ld3_best["conservative_calibration_95_half_width"])
    sampling12 = float(ld12_best["conservative_calibration_95_half_width"])
    selected_model3 = _measured_model_half_width(
        ld3_best, diagnostics, policy=POLICY_LABEL, delta=delta3
    )
    full_model3 = _measured_model_half_width(
        ld3_best, diagnostics, policy="full_basis_shared", delta=delta3
    )
    deterministic_error = float(
        deterministic[f"delta{delta12:g}"]["maximum_rz_holdout_error"]
    )
    model12 = cost12 * deterministic_error
    scenarios = {}
    for name, half3, half12 in (
        (
            "per_r_selected_measured_discrepancy",
            sampling3 + selected_model3,
            sampling12 + model12,
        ),
        (
            "per_r_full_basis_measured_discrepancy",
            sampling3 + full_model3,
            sampling12 + model12,
        ),
        ("local_5_percent", sampling3 + 0.05 * cost3, sampling12 + 0.05 * cost12),
        ("transfer_25_percent", sampling3 + 0.25 * cost3, sampling12 + 0.25 * cost12),
    ):
        interval3 = _interval(cost3, half3)
        interval12 = _interval(cost12, half12)
        scenarios[name] = {
            "ld3_interval": interval3,
            "ld12_interval": interval12,
            "ld3_total_half_width": half3,
            "ld12_total_half_width": half12,
            "intervals_overlap": max(interval3[0], interval12[0]) <= min(
                interval3[1], interval12[1]
            ),
        }
    return {
        "ld3_total_compiled_rz_point_estimate": cost3,
        "ld12_total_compiled_rz_point_estimate": cost12,
        "ld3_over_ld12_point_estimate_ratio": cost3 / cost12,
        "point_preference": "L_D=3" if cost3 < cost12 else "L_D=12",
        "sampling_uncertainty": {"ld3_95_half_width": sampling3, "ld12_95_half_width": sampling12},
        "measured_model_discrepancy_half_width": {
            "ld3_selected_per_r": selected_model3,
            "ld3_full_basis_per_r": full_model3,
            "ld12_deterministic": model12,
        },
        "scenarios": scenarios,
    }


def _preparation_break_even(
    comparison: Mapping[str, Any], *, shots3: int, shots12: int
) -> dict[str, Any]:
    denominator = shots3 - shots12
    cost3 = float(comparison["ld3_total_compiled_rz_point_estimate"])
    cost12 = float(comparison["ld12_total_compiled_rz_point_estimate"])
    point = None if denominator <= 0 else (cost12 - cost3) / denominator
    scenarios = {}
    for name, row in comparison["scenarios"].items():
        robust = None
        if denominator > 0:
            robust = (float(row["ld12_interval"][0]) - float(row["ld3_interval"][1])) / denominator
        scenarios[name] = {
            "common_per_shot_point_break_even_rz_equivalent": point,
            "common_per_shot_robust_interval_break_even_rz_equivalent": robust,
            "intervals_overlap_at_zero_preparation": bool(row["intervals_overlap"]),
        }
    return {
        "ld3_total_shots": shots3,
        "ld12_total_shots": shots12,
        "ld3_minus_ld12_shots": denominator,
        "scenarios": scenarios,
    }


def evaluate_full_opt2_analysis(
    *,
    project_root: str | Path,
    manifest_path: str | Path,
    compute_output_dir: str | Path,
    progress: Any = None,
) -> dict[str, Any]:
    """Analyze only a complete initial batch and then rerun coherent optimization."""
    root = Path(project_root).resolve(strict=True)
    evidence = load_and_validate_full_opt2_evidence(root)
    manifest = load_task_manifest(manifest_path, project_root=root)
    points = _collect_completed_points(manifest, Path(compute_output_dir))
    points.update(_reused_points(evidence["compiler_raw"]))
    diagnostics: dict[str, Any] = {}
    for delta in (0.01, 0.02):
        for rte_steps in RTE_STEPS:
            holdout = (16, 32) if delta == 0.02 and rte_steps == 32 else HOLDOUT_Q
            diagnostics[f"delta{delta:g}:r{rte_steps}"] = evaluate_proxy_cell(
                points[(3, delta, rte_steps)], holdout_q=holdout
            )
    deterministic = {
        "delta0.01": _deterministic_diagnostic(
            points[(12, 0.01, 0)], holdout_q=HOLDOUT_Q
        ),
        "delta0.02": _deterministic_diagnostic(
            points[(12, 0.02, 0)], holdout_q=(16, 32)
        ),
    }
    all_proxy_pass = all(row["pass"] for row in diagnostics.values())
    checks = {
        "complete_initial_batch_loaded": True,
        "all_selected_r_have_coherent_opt2_points": len(diagnostics) == 12,
        "all_initial_proxy_cells_pass": all_proxy_pass,
        "coherent_reoptimization_completed": False,
        "focused_and_coherent_results_are_separate": True,
        "final_total_cost_and_scientific_superiority_not_claimed": True,
    }
    body: dict[str, Any] = {
        "configuration": {
            "molecule": "H4_chain",
            "geometry_angstrom": 1.0,
            "basis": "STO-3G",
            "n_qubits": 8,
            "df_rank": 12,
            "candidate_ld_values": [3, 12],
            "delta_values": [0.01, 0.02],
            "r_values": list(RTE_STEPS),
            "finite_taylor_order": 2,
            "calibration_q": list(CALIBRATION_Q),
            "new_holdout_q": list(HOLDOUT_Q),
            "compiler": {**EXPECTED_COMPILER, "qiskit_version": "1.3.0"},
        },
        "compute_manifest_fingerprint": manifest["manifest_fingerprint"],
        "proxy_diagnostics": diagnostics,
        "deterministic_diagnostics": deterministic,
        "extension_required_cells": [
            key for key, row in diagnostics.items() if not row["pass"]
        ],
        "checks": checks,
        "scope": {
            "coherent_optimization_level_2_context": True,
            "state_preparation_included": False,
            "backend_execution_included": False,
            "q_above_32_directly_validated": False,
            "final_total_cost_evaluation_performed": False,
            "scientific_superiority_claimed": False,
        },
    }
    if not all_proxy_pass:
        body["overall_pass"] = False
        body["status"] = "requires_fresh_32_trajectory_extension"
        return body

    wp01 = _read_json(root, WP01_PATH)
    validate_artifact(wp01)
    hamiltonian = load_connected_cluster_hamiltonian_snapshot(root / SNAPSHOT_PATH)
    preparations = {
        ld: prepare_df_partial_s2(
            hamiltonian,
            split_df_hamiltonian_by_ld(hamiltonian, ld),
            identity_policy="extract_identity_phase",
        )
        for ld in (3, 12)
    }
    models: dict[tuple[int, float], tuple[Any, ...]] = {}
    for delta in (0.01, 0.02):
        models[(3, delta)] = tuple(
            _pair_model(
                points[(3, delta, rte_steps)],
                rte_steps=rte_steps,
                finite_taylor_order=2,
                policy=POLICY_LABEL,
                source_fingerprint=manifest["manifest_fingerprint"],
                source_label=f"M06-F_opt2_delta{delta:g}_r{rte_steps}",
            )
            for rte_steps in RTE_STEPS
        )
        models[(12, delta)] = (
            _pair_model(
                points[(12, delta, 0)],
                rte_steps=0,
                finite_taylor_order=0,
                policy=None,
                source_fingerprint=manifest["manifest_fingerprint"],
                source_label=f"M06-F_opt2_deterministic_delta{delta:g}",
            ),
        )
    candidates = {}
    for ld in (3, 12):
        for delta in (0.01, 0.02):
            maximum_round_index, pf_coefficient = _candidate_configuration(
                wp01, ld=ld, delta_time=delta
            )
            key = f"ld{ld}:delta{delta:g}"
            candidates[key] = _grid_search(
                preparations[ld],
                models[(ld, delta)],
                ld=ld,
                delta_time=delta,
                maximum_round_index=maximum_round_index,
                pf_coefficient=pf_coefficient,
                beta_rpe=0.4,
                alpha_total=0.05,
                rte_seed=2026092208,
                progress=progress,
            )
    best_by_ld = {
        str(ld): min(
            (row for row in candidates.values() if int(row["ld"]) == ld),
            key=lambda row: row["best"]["total_compiled_rz_point_estimate"],
        )
        for ld in (3, 12)
    }
    comparison = _comparison(
        best_by_ld["3"]["best"],
        best_by_ld["12"]["best"],
        diagnostics,
        deterministic,
        delta3=float(best_by_ld["3"]["delta_time"]),
        delta12=float(best_by_ld["12"]["delta_time"]),
    )
    break_even = _preparation_break_even(
        comparison,
        shots3=int(best_by_ld["3"]["best"]["total_shots"]),
        shots12=int(best_by_ld["12"]["best"]["total_shots"]),
    )
    focused = _read_json(root, COMPILER_ANALYSIS_PATH)
    validate_compiler_transfer_analysis_artifact(focused)
    body.update(
        {
            "coherent_reoptimization": {
                "candidates": candidates,
                "best_by_ld": best_by_ld,
                "comparison": comparison,
            },
            "preparation_break_even": break_even,
            "focused_mixed_compiler_reference": {
                "content_fingerprint": focused["content_fingerprint"],
                "comparison": focused["focused_fixed_plan_reaggregation"]["comparison"],
                "label": "mixed_opt1_opt2_focused_not_coherent",
            },
            "status": "coherent_opt2_reoptimization_complete",
        }
    )
    checks["coherent_reoptimization_completed"] = True
    body["overall_pass"] = all(checks.values())
    return body


def extension_cells_from_analysis(
    payload: Mapping[str, Any]
) -> tuple[tuple[float, int, int], ...]:
    cells = []
    for key in payload.get("extension_required_cells", []):
        delta_text, r_text = str(key).split(":")
        delta = float(delta_text.removeprefix("delta"))
        rte_steps = int(r_text.removeprefix("r"))
        for q_m in (*CALIBRATION_Q, *HOLDOUT_Q):
            cells.append((delta, rte_steps, q_m))
    return tuple(cells)


def finalize_full_opt2_analysis_artifact(
    body: Mapping[str, Any], *, provenance: Mapping[str, Any]
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "stage": "M06-F-analysis",
        **dict(body),
        "provenance": dict(provenance),
    }
    payload["content_fingerprint"] = fingerprint(payload)
    validate_full_opt2_analysis_artifact(payload)
    return payload


def validate_full_opt2_analysis_artifact(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("method") != METHOD:
        raise ValueError("Unsupported M06-F analysis schema or method.")
    unsigned = dict(payload)
    observed = unsigned.pop("content_fingerprint", None)
    if observed != fingerprint(unsigned):
        raise ValueError("M06-F analysis fingerprint mismatch.")
    checks = payload.get("checks", {})
    if payload.get("overall_pass") != (bool(checks) and all(checks.values())):
        raise ValueError("M06-F overall status does not match checks.")
    scope = payload.get("scope", {})
    if scope.get("final_total_cost_evaluation_performed") is not False:
        raise ValueError("M06-F cannot claim final total cost.")
    if scope.get("scientific_superiority_claimed") is not False:
        raise ValueError("M06-F cannot claim scientific superiority.")


def write_full_opt2_analysis_artifact(
    payload: Mapping[str, Any], path: str | Path
) -> None:
    validate_full_opt2_analysis_artifact(payload)
    output = Path(path)
    if output.exists():
        raise ValueError(f"Refusing to replace existing analysis artifact: {output}")
    atomic_write_json(output, payload)
