"""Completion audit and gated synthesis for the M06-F opt2 workflow."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .parallel_validation_executor import (
    TASK_SPEC_SCHEMA_VERSION,
    WORKER_RESULT_SCHEMA_VERSION,
    atomic_write_json,
    build_deterministic_aggregate,
    file_sha256,
    fingerprint_payload,
    load_task_manifest,
    read_checkpoint,
)
from .research_direction_compiler_transfer_analysis import (
    validate_compiler_transfer_analysis_artifact,
)
from .research_direction_full_opt2 import (
    EXPECTED_COMPILER,
    POLICY_LABEL,
    RTE_STEPS,
    _collect_completed_points,
    _reused_points,
    load_and_validate_full_opt2_evidence,
    validate_full_opt2_analysis_artifact,
)
from .research_direction_uncertainty_break_even import (
    validate_uncertainty_break_even_artifact,
)
from .research_direction_wp11_synthesis import validate_wp11_artifact


SCHEMA_VERSION = "research_direction_full_opt2_completion_audit_v1"
METHOD = "m06f_all_r_coherent_opt2_completion_audit_v1"
THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)
MIXED_ANALYSIS_PATH = (
    "artifacts/research_direction_compiler_transfer/2026-09-23/"
    "m06_l08_opt2_focused_analysis_reaggregation_v1.json"
)
N07_PATH = (
    "artifacts/research_direction_uncertainty_break_even/2026-09-23/"
    "n07_p03_uncertainty_break_even_v1.json"
)
WP11_PATH = (
    "artifacts/research_direction_wp11_synthesis/2026-09-23/"
    "wp11_scoped_direction_synthesis_v1.json"
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _attempt_name(task_id: str, attempt: int = 1) -> str:
    return f"{task_id}.attempt-{attempt:04d}.json"


def _direct_rz_rows(
    points: Mapping[tuple[int, float, int], Mapping[int, Mapping[str, Any]]],
    *,
    initial_keys: set[tuple[int, float, int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (ld, delta, rte_steps), q_points in sorted(points.items()):
        for q_m, point in sorted(q_points.items()):
            partition = "calibration" if q_m in (1, 2) else "holdout"
            for axis in ("cosine", "sine"):
                if ld == 3:
                    policy_rows = point["axes"][axis]["policies"]
                else:
                    policy_rows = {"deterministic": point["axes"][axis]}
                for policy, metrics in sorted(policy_rows.items()):
                    record = metrics["rz_count"]
                    rows.append(
                        {
                            "ld": ld,
                            "delta": delta,
                            "r": rte_steps,
                            "q": q_m,
                            "partition": partition,
                            "axis": axis,
                            "policy": policy,
                            "sample_count": int(point["sample_count"]),
                            "mean": float(record["mean"]),
                            "standard_error": float(record["standard_error"]),
                            "minimum": float(record["minimum"]),
                            "maximum": float(record["maximum"]),
                            "source": (
                                "initial_batch_checkpoint"
                                if (ld, delta, rte_steps) in initial_keys
                                else "validated_existing_opt2_raw_artifact"
                            ),
                        }
                    )
    return rows


def _proxy_summary(analysis: Mapping[str, Any]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for key, diagnostic in sorted(analysis["proxy_diagnostics"].items()):
        rows[key] = {
            "pass": bool(diagnostic["pass"]),
            "holdout_q": list(diagnostic["holdout_q"]),
            "maximum_selected_rz_holdout_error": float(
                diagnostic["maximum_selected_rz_holdout_error"]
            ),
            "maximum_selected_all_metrics_holdout_error": float(
                diagnostic["maximum_selected_all_metrics_holdout_error"]
            ),
            "maximum_full_basis_rz_holdout_error": float(
                diagnostic["maximum_full_basis_rz_holdout_error"]
            ),
            "maximum_direct_rz_relative_standard_error": float(
                diagnostic["maximum_direct_rz_relative_standard_error"]
            ),
            "checks": dict(diagnostic["checks"]),
        }
    return {
        "cells": rows,
        "passing_cell_count": sum(row["pass"] for row in rows.values()),
        "failing_cell_count": sum(not row["pass"] for row in rows.values()),
        "maximum_selected_rz_holdout_error": max(
            row["maximum_selected_rz_holdout_error"] for row in rows.values()
        ),
        "maximum_selected_all_metrics_holdout_error": max(
            row["maximum_selected_all_metrics_holdout_error"]
            for row in rows.values()
        ),
        "maximum_direct_rz_relative_standard_error": max(
            row["maximum_direct_rz_relative_standard_error"]
            for row in rows.values()
        ),
    }


def _load_upstream(root: Path) -> dict[str, dict[str, Any]]:
    mixed = _read_json(root / MIXED_ANALYSIS_PATH)
    n07 = _read_json(root / N07_PATH)
    wp11 = _read_json(root / WP11_PATH)
    validate_compiler_transfer_analysis_artifact(mixed)
    validate_uncertainty_break_even_artifact(n07)
    validate_wp11_artifact(wp11)
    return {"mixed": mixed, "n07": n07, "wp11": wp11}


def build_completion_audit(
    *,
    project_root: str | Path,
    initial_manifest_path: str | Path,
    compute_output_dir: str | Path,
    initial_analysis_path: str | Path,
    extension_manifest_path: str | Path,
    runtime_observation: Mapping[str, Any],
    original_main_observation: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit completed files and stop synthesis at the preregistered gate."""
    root = Path(project_root).resolve(strict=True)
    output_dir = Path(compute_output_dir).resolve(strict=True)
    manifest = load_task_manifest(initial_manifest_path, project_root=root)
    extension = load_task_manifest(extension_manifest_path, project_root=root)
    evidence = load_and_validate_full_opt2_evidence(root)
    analysis_path = Path(initial_analysis_path).resolve(strict=True)
    analysis = _read_json(analysis_path)
    validate_full_opt2_analysis_artifact(analysis)
    upstream = _load_upstream(root)

    invalid_json: list[str] = []
    json_paths = sorted(output_dir.rglob("*.json"))
    for path in json_paths:
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            invalid_json.append(_relative(path, root))

    tasks = manifest["tasks"]
    task_ids = [task["task_id"] for task in tasks]
    task_id_set = set(task_ids)
    checkpoints: list[dict[str, Any]] = []
    missing: list[str] = []
    partial: list[str] = []
    failed: list[str] = []
    mismatched: list[str] = []
    for task in tasks:
        task_id = task["task_id"]
        checkpoint_path = output_dir / "checkpoints" / f"{task_id}.json"
        spec_path = output_dir / "tasks" / _attempt_name(task_id)
        worker_path = output_dir / "worker_results" / _attempt_name(task_id)
        log_path = output_dir / "logs" / _attempt_name(task_id).replace(
            ".json", ".log"
        )
        for kind, path in (
            ("checkpoint", checkpoint_path),
            ("task_spec", spec_path),
            ("worker_result", worker_path),
            ("log", log_path),
        ):
            if not path.is_file():
                missing.append(f"{kind}:{task_id}")
        if any(not path.is_file() for path in (checkpoint_path, spec_path, worker_path)):
            continue
        checkpoint = read_checkpoint(checkpoint_path, task=task)
        spec = _read_json(spec_path)
        worker = _read_json(worker_path)
        checkpoints.append(checkpoint)
        if checkpoint["status"] != "completed" or checkpoint["result"] is None:
            partial.append(task_id)
        if checkpoint["status"] == "failed" or checkpoint["error"] is not None:
            failed.append(task_id)
        spec_task = dict(spec.get("task", {}))
        spec_task.pop("project_root", None)
        if (
            spec.get("schema_version") != TASK_SPEC_SCHEMA_VERSION
            or spec.get("attempt") != 1
            or spec_task != task
            or worker.get("schema_version") != WORKER_RESULT_SCHEMA_VERSION
            or worker.get("task_id") != task_id
            or worker.get("task_fingerprint") != task["task_fingerprint"]
            or worker.get("attempt") != 1
            or worker.get("status") != "completed"
            or worker.get("error") is not None
            or worker.get("result") != checkpoint.get("result")
        ):
            mismatched.append(task_id)

    checkpoint_ids_on_disk = {
        path.stem for path in (output_dir / "checkpoints").glob("*.json")
    }
    extra_checkpoint_ids = sorted(checkpoint_ids_on_disk - task_id_set)
    duplicate_task_ids = sorted(
        {task_id for task_id in task_ids if task_ids.count(task_id) > 1}
    )
    cell_keys = [
        (
            int(task["ld"]),
            float(task["delta"]),
            int(task["r"]),
            int(task["q"]),
        )
        for task in tasks
    ]
    duplicate_cells = [
        list(cell)
        for cell in sorted(
            {cell for cell in cell_keys if cell_keys.count(cell) > 1}
        )
    ]

    aggregate_path = output_dir / "aggregate.json"
    status_path = output_dir / "batch_status.json"
    provenance_path = output_dir / "run_provenance.json"
    aggregate = _read_json(aggregate_path)
    status = _read_json(status_path)
    run_provenance = _read_json(provenance_path)
    rebuilt_aggregate = build_deterministic_aggregate(checkpoints)

    expected_randomized = {
        (3, delta, rte_steps, q_m)
        for delta, values in ((0.01, RTE_STEPS), (0.02, RTE_STEPS[:-1]))
        for rte_steps in values
        for q_m in (1, 2, 8)
    }
    expected_deterministic = {(12, 0.01, 0, q_m) for q_m in (1, 2, 8)}
    observed_cells = set(cell_keys)
    required_r_coverage = all(
        (3, delta, rte_steps, q_m) in observed_cells
        for delta in (0.01, 0.02)
        for rte_steps in (1, 2, 4, 8, 16)
        for q_m in (1, 2, 8)
    )
    partitions_valid = all(
        task["parameters"]["partition"]
        == ("calibration" if int(task["q"]) in (1, 2) else "holdout")
        for task in tasks
    )
    compiler_valid = all(
        task["compiler_settings"] == EXPECTED_COMPILER
        and checkpoint["result"]["compiler"]
        == {**EXPECTED_COMPILER, "qiskit_version": "1.3.0"}
        for task, checkpoint in zip(tasks, checkpoints, strict=True)
    )
    thread_and_gpu_valid = all(
        checkpoint["assigned_gpu_id"] is None
        and checkpoint["result"]["cuda_visible_devices"] == ""
        and checkpoint["result"]["thread_environment"]
        == {name: "1" for name in THREAD_ENVIRONMENT}
        for checkpoint in checkpoints
    )
    provenance_valid = all(
        checkpoint["provenance"]["seed"] == task["seed"]
        and checkpoint["result"]["input_sha256"] == task["input_sha256"]
        and checkpoint["result"]["source_sha256"] == task["source_sha256"]
        and checkpoint["result"]["dependency_versions"]
        == {
            "python": "3.12.3",
            "qiskit": "1.3.0",
            "numpy": "1.26.4",
            "scipy": "1.14.1",
        }
        for task, checkpoint in zip(tasks, checkpoints, strict=True)
    )
    status_valid = (
        status["state"] == "completed"
        and status["counts"]
        == {"completed": len(tasks), "failed": 0, "interrupted": 0, "pending": 0}
        and status["manifest_fingerprint"] == manifest["manifest_fingerprint"]
        and status["aggregate_fingerprint"] == aggregate["aggregate_fingerprint"]
        and status["active_pids"] == []
        and status["terminated_pids"] == []
    )
    aggregate_valid = aggregate == rebuilt_aggregate
    seed_values = [int(task["seed"]) for task in tasks if int(task["ld"]) == 3]
    provenance_recorded = (
        isinstance(run_provenance.get("git_commit"), str)
        and isinstance(run_provenance.get("git_status"), list)
        and run_provenance.get("input_sha256")
        and run_provenance.get("source_sha256")
        and len(seed_values) == len(set(seed_values))
    )
    initial_checks = {
        "tmux_session_absent": not runtime_observation["tmux_session_present"],
        "parent_and_worker_processes_absent": not runtime_observation[
            "matching_processes"
        ],
        "runner_exit_code_inferred_zero": status_valid,
        "expected_equals_completed": len(checkpoints) == len(tasks),
        "no_failed_missing_partial_or_duplicate_tasks": not any(
            (
                failed,
                missing,
                partial,
                mismatched,
                duplicate_task_ids,
                duplicate_cells,
                extra_checkpoint_ids,
            )
        ),
        "all_compute_json_readable": not invalid_json,
        "checkpoint_worker_and_aggregate_consistent": aggregate_valid
        and not mismatched,
        "required_r_1_2_4_8_16_present": required_r_coverage,
        "calibration_and_holdout_distinct": partitions_valid,
        "single_opt2_compiler_context": compiler_valid,
        "seed_dependencies_hashes_commit_dirty_recorded": provenance_valid
        and provenance_recorded,
        "gpu_not_used_and_worker_threads_one": thread_and_gpu_valid
        and run_provenance["resource_plan"]["gpu_ids"] == [],
        "initial_cell_scope_exact": observed_cells
        == expected_randomized | expected_deterministic,
        "original_main_observed_unchanged": bool(
            original_main_observation["status_matches_expected"]
        ),
    }

    initial_points = _collect_completed_points(manifest, output_dir)
    all_points = dict(initial_points)
    all_points.update(_reused_points(evidence["compiler_raw"]))
    direct_rows = _direct_rz_rows(
        all_points, initial_keys=set(initial_points)
    )
    proxy = _proxy_summary(analysis)

    extension_output_candidates = sorted(
        str(path)
        for path in (root / "artifacts/research_direction_full_opt2").rglob("*")
        if path.is_dir() and "extension" in path.name and path != output_dir
    )
    extension_completed = 0
    workflow_expected = len(tasks) + len(extension["tasks"])
    workflow_completed = len(checkpoints) + extension_completed
    extension_missing = len(extension["tasks"]) - extension_completed

    generated = datetime.fromisoformat(run_provenance["generated_at_utc"])
    finished = datetime.fromisoformat(status["updated_at_utc"])
    mixed_comparison = upstream["mixed"]["focused_fixed_plan_reaggregation"][
        "comparison"
    ]
    n07_scenario = upstream["n07"]["p03_break_even_scenarios"][
        "opt2_focused_selected_q_le_32"
    ]
    direction_rows = {
        row["direction_id"]: row["wp11_decision"]
        for row in upstream["wp11"]["direction_decisions"]
        if row["direction_id"] in ("T1", "T2", "T4", "T5", "T6", "T7")
    }

    body = {
        "status": "initial_batch_complete_fresh32_extension_missing",
        "requested_coherent_reoptimization_complete": False,
        "initial_compute_integrity_pass": all(initial_checks.values()),
        "configuration": dict(analysis["configuration"]),
        "execution": {
            "initial_batch": {
                "expected": len(tasks),
                "completed": len(checkpoints),
                "failed": len(failed),
                "missing": len(missing),
                "partial": len(partial),
                "duplicate": len(duplicate_task_ids) + len(duplicate_cells),
                "started_at_utc": run_provenance["generated_at_utc"],
                "finished_at_utc": status["updated_at_utc"],
                "elapsed_seconds": (finished - generated).total_seconds(),
                "exit_code": {
                    "value": 0 if status_valid else None,
                    "basis": "inferred_from_completed_runner_contract",
                    "independent_shell_exit_status_recorded": False,
                },
            },
            "preregistered_workflow": {
                "expected": workflow_expected,
                "completed": workflow_completed,
                "failed": len(failed),
                "missing_extension": extension_missing,
            },
            "runtime_observation": dict(runtime_observation),
            "original_main_observation": dict(original_main_observation),
            "run_provenance": run_provenance,
        },
        "integrity": {
            "checks": initial_checks,
            "invalid_json": invalid_json,
            "missing": missing,
            "partial": partial,
            "failed": failed,
            "mismatched": mismatched,
            "duplicate_task_ids": duplicate_task_ids,
            "duplicate_cells": duplicate_cells,
            "extra_checkpoint_ids": extra_checkpoint_ids,
            "json_file_count": len(json_paths),
            "aggregate_fingerprint": aggregate["aggregate_fingerprint"],
            "manifest_fingerprint": manifest["manifest_fingerprint"],
        },
        "direct_compiled_rz_measurements": direct_rows,
        "calibration_holdout_diagnostics": proxy,
        "extension_gate": {
            "required": True,
            "manifest_fingerprint": extension["manifest_fingerprint"],
            "task_count": len(extension["tasks"]),
            "completed_task_count": extension_completed,
            "output_directories": extension_output_candidates,
            "failing_initial_cells": list(analysis["extension_required_cells"]),
            "reason": (
                "five cells exceed the preregistered 2 percent direct-RZ "
                "relative-standard-error threshold"
            ),
        },
        "analysis_items": {
            "1_direct_measured_compiled_rz": "complete",
            "2_calibration_and_holdout_prediction_error": "complete",
            "3_beta_alpha_shots_schedule_by_r": "blocked_by_fresh32_gate",
            "4_ld3_vs_ld12": "blocked_by_coherent_reoptimization",
            "5_delta_0p01_vs_0p02": "blocked_by_coherent_reoptimization",
            "6_mixed_vs_coherent_opt2": "mixed_reference_recorded_coherent_missing",
            "7_compiler_transfer_uncertainty": (
                "direct_r1_2_4_8_16_coverage_added_but_not_resolved"
            ),
            "8_preparation_break_even": (
                "existing_mixed_reference_recorded_coherent_update_blocked"
            ),
            "9_external_instance_pilot": "remain_deferred",
            "10_priority_effect": "no_change_until_extension_and_reoptimization",
        },
        "mixed_compiler_reference": {
            "content_fingerprint": upstream["mixed"]["content_fingerprint"],
            "comparison": mixed_comparison,
        },
        "preparation_break_even_reference": {
            "content_fingerprint": upstream["n07"]["content_fingerprint"],
            "decision": upstream["n07"]["decision"],
            "synthesis": upstream["n07"]["synthesis"],
            "selected_scenario": n07_scenario,
        },
        "research_decision": {
            "wp11_content_fingerprint": upstream["wp11"]["content_fingerprint"],
            "wp11_decision_changed": False,
            "external_instance_pilot": "defer",
            "direction_priorities": direction_rows,
            "reason": (
                "the initial compute is complete, but the preregistered "
                "precision extension and coherent optimization are not complete"
            ),
        },
        "scope": {
            "molecule": "H4_chain",
            "geometry_angstrom": 1.0,
            "basis": "STO-3G",
            "n_qubits": 8,
            "df_rank": 12,
            "c_use_is_rigorous_upper_bound": False,
            "q_above_32_conclusion": False,
            "h12_extrapolation_conclusion": False,
            "state_preparation_mixed_with_pf_coefficient": False,
            "final_total_cost_evaluation_performed": False,
            "scientific_superiority_claimed": False,
        },
        "source_artifacts": {
            "initial_manifest": {
                "path": _relative(Path(initial_manifest_path), root),
                "fingerprint": manifest["manifest_fingerprint"],
            },
            "initial_analysis": {
                "path": _relative(analysis_path, root),
                "fingerprint": analysis["content_fingerprint"],
            },
            "extension_manifest": {
                "path": _relative(Path(extension_manifest_path), root),
                "fingerprint": extension["manifest_fingerprint"],
            },
            "compute_output": {
                "path": _relative(output_dir, root),
                "aggregate_fingerprint": aggregate["aggregate_fingerprint"],
            },
        },
    }
    return body


def finalize_completion_audit(
    body: Mapping[str, Any], *, provenance: Mapping[str, Any]
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "stage": "M06-F-completion-audit",
        **dict(body),
        "provenance": dict(provenance),
    }
    payload["content_fingerprint"] = fingerprint_payload(payload)
    validate_completion_audit(payload)
    return payload


def validate_completion_audit(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get(
        "method"
    ) != METHOD:
        raise ValueError("Unsupported M06-F completion audit schema or method.")
    unsigned = dict(payload)
    observed = unsigned.pop("content_fingerprint", None)
    if observed != fingerprint_payload(unsigned):
        raise ValueError("M06-F completion audit fingerprint mismatch.")
    if not payload.get("initial_compute_integrity_pass"):
        raise ValueError("M06-F initial compute integrity audit did not pass.")
    if payload.get("requested_coherent_reoptimization_complete") is not False:
        raise ValueError("Audit must not claim an unexecuted coherent optimization.")
    if payload["scope"]["final_total_cost_evaluation_performed"] is not False:
        raise ValueError("Audit cannot claim final total-cost completion.")
    if payload["scope"]["scientific_superiority_claimed"] is not False:
        raise ValueError("Audit cannot claim scientific superiority.")


def write_completion_audit(payload: Mapping[str, Any], path: str | Path) -> None:
    validate_completion_audit(payload)
    output = Path(path)
    if output.exists():
        raise ValueError(f"Refusing to replace existing audit artifact: {output}")
    atomic_write_json(output, payload)
