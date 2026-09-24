"""Audit M06-F fresh-32 evidence and run the gated coherent analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .df_partial_randomized_pf import split_df_hamiltonian_by_ld
from .df_partial_s2 import prepare_df_partial_s2
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
from .research_direction_decision_cost import (
    _candidate_configuration,
    _grid_search,
    _pair_model,
)
from .research_direction_full_opt2 import (
    CALIBRATION_Q,
    COMPILER_ANALYSIS_PATH,
    EXPECTED_COMPILER,
    HOLDOUT_Q,
    POLICY_LABEL,
    RTE_STEPS,
    SNAPSHOT_PATH,
    THREAD_ENVIRONMENT,
    WP01_PATH,
    _collect_completed_points,
    _comparison,
    _deterministic_diagnostic,
    _preparation_break_even,
    _read_json,
    _reused_points,
    evaluate_proxy_cell,
    load_and_validate_full_opt2_evidence,
    validate_full_opt2_analysis_artifact,
)
from .research_direction_full_scope import AXES, fingerprint
from .research_direction_prevalidation import validate_artifact
from .rte_connected_cluster_cost_validation import (
    load_connected_cluster_hamiltonian_snapshot,
)


AUDIT_SCHEMA_VERSION = "research_direction_full_opt2_extension_audit_v1"
ANALYSIS_SCHEMA_VERSION = "research_direction_full_opt2_extension_analysis_v1"
METHOD = "m06f_fresh32_gated_coherent_opt2_reoptimization_v1"


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _extension_keys(payload: Mapping[str, Any]) -> set[tuple[int, float, int]]:
    keys = set()
    for value in payload["extension_required_cells"]:
        delta_text, r_text = str(value).split(":")
        keys.add(
            (
                3,
                float(delta_text.removeprefix("delta")),
                int(r_text.removeprefix("r")),
            )
        )
    return keys


def _load_combined_points(
    *,
    root: Path,
    initial_manifest: Mapping[str, Any],
    initial_output_dir: Path,
    extension_manifest: Mapping[str, Any],
    extension_output_dir: Path,
    initial_analysis: Mapping[str, Any],
    evidence: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[tuple[int, float, int], dict[int, Any]], set[tuple[int, float, int]]]:
    initial = _collect_completed_points(initial_manifest, initial_output_dir)
    extension = _collect_completed_points(extension_manifest, extension_output_dir)
    expected_extension = _extension_keys(initial_analysis)
    if set(extension) != expected_extension:
        raise ValueError("Fresh-32 cells differ from the preregistered failing cells.")
    if any(set(points) != {1, 2, 8} for points in extension.values()):
        raise ValueError("Every fresh-32 cell must contain q=1,2,8.")

    combined = {key: dict(value) for key, value in initial.items()}
    for key, value in _reused_points(evidence["compiler_raw"]).items():
        combined.setdefault(key, {}).update(value)
    # A fresh cell replaces its complete eight-trajectory predecessor. This also
    # prevents the reused delta=0.02,r=32 q=1,2 points from masking fresh q=1,2,8.
    combined.update({key: dict(value) for key, value in extension.items()})
    return combined, expected_extension


def _diagnostics(
    points: Mapping[tuple[int, float, int], Mapping[int, Mapping[str, Any]]]
) -> dict[str, Any]:
    rows = {}
    for delta in (0.01, 0.02):
        for rte_steps in RTE_STEPS:
            rows[f"delta{delta:g}:r{rte_steps}"] = evaluate_proxy_cell(
                points[(3, delta, rte_steps)], holdout_q=HOLDOUT_Q
            )
    return rows


def _model_points(points: Mapping[int, Any]) -> dict[str, Any]:
    """Adapt compute checkpoint q keys to the cost model's artifact format."""
    return {str(q_m): point for q_m, point in points.items()}


def _rz_statistics(record: Mapping[str, Any]) -> dict[str, float]:
    return {
        key: float(record[key])
        for key in ("mean", "standard_error", "minimum", "maximum")
    }


def _direct_rz_measurements(
    points: Mapping[tuple[int, float, int], Mapping[int, Mapping[str, Any]]],
    *,
    extension_keys: set[tuple[int, float, int]],
) -> dict[str, Any]:
    cells = {}
    for (ld, delta, rte_steps), q_points in sorted(points.items()):
        q_rows = {}
        for q_m, point in sorted(q_points.items()):
            if ld == 3:
                policies = {
                    label: {
                        axis: _rz_statistics(
                            point["axes"][axis]["policies"][label]["rz_count"]
                        )
                        for axis in AXES
                    }
                    for label in (POLICY_LABEL, "full_basis_shared")
                }
            else:
                policies = {
                    "deterministic_tail_free": {
                        axis: _rz_statistics(point["axes"][axis]["rz_count"])
                        for axis in AXES
                    }
                }
            q_rows[str(q_m)] = {
                "role": (
                    "calibration"
                    if q_m in CALIBRATION_Q
                    else "fixed_holdout"
                    if q_m in HOLDOUT_Q
                    else "additional_holdout"
                ),
                "sample_count": int(point["sample_count"]),
                "compiled_rz": policies,
            }
        key = f"ld{ld}:delta{delta:g}:r{rte_steps}"
        cells[key] = {
            "source_partition": (
                "fresh32_extension"
                if (ld, delta, rte_steps) in extension_keys
                else "initial_or_validated_reuse"
            ),
            "q": q_rows,
        }
    return cells


def _batch_integrity(
    manifest: Mapping[str, Any], output_dir: Path
) -> dict[str, Any]:
    checkpoints = []
    missing = []
    mismatched = []
    for task in manifest["tasks"]:
        task_id = task["task_id"]
        checkpoint_path = output_dir / "checkpoints" / f"{task_id}.json"
        attempt_name = f"{task_id}.attempt-0001.json"
        spec_path = output_dir / "tasks" / attempt_name
        worker_path = output_dir / "worker_results" / attempt_name
        log_path = output_dir / "logs" / attempt_name.replace(".json", ".log")
        for label, path in (
            ("checkpoint", checkpoint_path),
            ("task_spec", spec_path),
            ("worker_result", worker_path),
            ("log", log_path),
        ):
            if not path.is_file():
                missing.append(f"{label}:{task_id}")
        if not checkpoint_path.is_file():
            continue
        checkpoint = read_checkpoint(checkpoint_path, task=task)
        checkpoints.append(checkpoint)
        if not spec_path.is_file() or not worker_path.is_file():
            continue
        spec = _read_object(spec_path)
        worker = _read_object(worker_path)
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

    invalid_json = []
    for path in sorted(output_dir.rglob("*.json")):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            invalid_json.append(str(path))
    aggregate = _read_object(output_dir / "aggregate.json")
    status = _read_object(output_dir / "batch_status.json")
    rebuilt = build_deterministic_aggregate(checkpoints)
    expected_ids = {task["task_id"] for task in manifest["tasks"]}
    observed_ids = {
        path.stem for path in (output_dir / "checkpoints").glob("*.json")
    }
    return {
        "expected": len(manifest["tasks"]),
        "completed": sum(row["status"] == "completed" for row in checkpoints),
        "failed": sum(row["status"] == "failed" for row in checkpoints),
        "partial": sum(row["status"] != "completed" for row in checkpoints),
        "missing": missing,
        "mismatched": mismatched,
        "invalid_json": invalid_json,
        "extra_checkpoint_ids": sorted(observed_ids - expected_ids),
        "missing_checkpoint_ids": sorted(expected_ids - observed_ids),
        "aggregate_matches": aggregate == rebuilt,
        "aggregate_fingerprint": aggregate["aggregate_fingerprint"],
        "batch_state": status["state"],
        "batch_counts": status["counts"],
        "active_pids": status["active_pids"],
    }


def _seed_audit(
    batches: Sequence[tuple[str, Mapping[str, Any], Path]]
) -> dict[str, Any]:
    task_seeds: dict[str, list[int]] = {}
    trajectory_seeds: dict[str, list[int]] = {}
    for label, manifest, output_dir in batches:
        task_seeds[label] = [
            int(task["seed"]) for task in manifest["tasks"] if int(task["ld"]) == 3
        ]
        values = []
        for task in manifest["tasks"]:
            if int(task["ld"]) != 3:
                continue
            checkpoint = read_checkpoint(
                output_dir / "checkpoints" / f"{task['task_id']}.json",
                task=task,
            )
            values.extend(
                int(row["trajectory_seed"])
                for row in checkpoint["result"]["point"]["trajectory_records"]
            )
        trajectory_seeds[label] = values
    initial_tasks = set(task_seeds["initial"])
    extension_tasks = set(task_seeds["extension"])
    initial_trajectories = set(trajectory_seeds["initial"])
    extension_trajectories = set(trajectory_seeds["extension"])
    return {
        "initial_randomized_task_seed_count": len(task_seeds["initial"]),
        "extension_randomized_task_seed_count": len(task_seeds["extension"]),
        "initial_task_seeds_unique": len(initial_tasks) == len(task_seeds["initial"]),
        "extension_task_seeds_unique": len(extension_tasks)
        == len(task_seeds["extension"]),
        "cross_batch_task_seed_overlap": sorted(initial_tasks & extension_tasks),
        "initial_trajectory_seed_count": len(trajectory_seeds["initial"]),
        "extension_trajectory_seed_count": len(trajectory_seeds["extension"]),
        "initial_trajectory_seeds_unique": len(initial_trajectories)
        == len(trajectory_seeds["initial"]),
        "extension_trajectory_seeds_unique": len(extension_trajectories)
        == len(trajectory_seeds["extension"]),
        "cross_batch_trajectory_seed_overlap": sorted(
            initial_trajectories & extension_trajectories
        ),
    }


def evaluate_extension_audit(
    *,
    project_root: str | Path,
    initial_manifest_path: str | Path,
    initial_output_dir: str | Path,
    extension_manifest_path: str | Path,
    extension_output_dir: str | Path,
    initial_analysis_path: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve(strict=True)
    initial_output = Path(initial_output_dir).resolve(strict=True)
    extension_output = Path(extension_output_dir).resolve(strict=True)
    initial_manifest = load_task_manifest(initial_manifest_path, project_root=root)
    extension_manifest = load_task_manifest(extension_manifest_path, project_root=root)
    initial_analysis = _read_object(Path(initial_analysis_path).resolve(strict=True))
    validate_full_opt2_analysis_artifact(initial_analysis)
    evidence = load_and_validate_full_opt2_evidence(root)
    points, extension_keys = _load_combined_points(
        root=root,
        initial_manifest=initial_manifest,
        initial_output_dir=initial_output,
        extension_manifest=extension_manifest,
        extension_output_dir=extension_output,
        initial_analysis=initial_analysis,
        evidence=evidence,
    )
    diagnostics = _diagnostics(points)
    initial_integrity = _batch_integrity(initial_manifest, initial_output)
    extension_integrity = _batch_integrity(extension_manifest, extension_output)
    seeds = _seed_audit(
        (
            ("initial", initial_manifest, initial_output),
            ("extension", extension_manifest, extension_output),
        )
    )
    wrapper = _read_object(extension_output / "wrapper_status.json")
    extension_provenance = _read_object(extension_output / "run_provenance.json")
    expected_cells = {
        (delta, rte_steps, q_m)
        for _, delta, rte_steps in extension_keys
        for q_m in (*CALIBRATION_Q, *HOLDOUT_Q)
    }
    observed_cells = {
        (float(task["delta"]), int(task["r"]), int(task["q"]))
        for task in extension_manifest["tasks"]
    }
    compiler_ok = all(
        task["compiler_settings"] == EXPECTED_COMPILER
        for task in initial_manifest["tasks"] + extension_manifest["tasks"]
    )
    worker_environment_ok = all(
        checkpoint["result"]["cuda_visible_devices"] == ""
        and checkpoint["assigned_gpu_id"] is None
        and checkpoint["result"]["thread_environment"]
        == {name: "1" for name in THREAD_ENVIRONMENT}
        for manifest, output in (
            (initial_manifest, initial_output),
            (extension_manifest, extension_output),
        )
        for task in manifest["tasks"]
        for checkpoint in [
            read_checkpoint(
                output / "checkpoints" / f"{task['task_id']}.json", task=task
            )
        ]
    )
    source_hashes_unchanged = all(
        file_sha256(root / path) == digest
        for path, digest in extension_provenance["source_sha256"].items()
    )
    all_proxy_pass = all(row["pass"] for row in diagnostics.values())
    integrity_checks = {
        "initial_36_complete": initial_integrity["completed"] == 36,
        "extension_15_complete": extension_integrity["completed"] == 15,
        "combined_51_complete": initial_integrity["completed"]
        + extension_integrity["completed"]
        == 51,
        "no_failed_partial_missing_mismatched_or_extra": not any(
            (
                initial_integrity["failed"],
                initial_integrity["partial"],
                initial_integrity["missing"],
                initial_integrity["mismatched"],
                initial_integrity["invalid_json"],
                initial_integrity["extra_checkpoint_ids"],
                initial_integrity["missing_checkpoint_ids"],
                extension_integrity["failed"],
                extension_integrity["partial"],
                extension_integrity["missing"],
                extension_integrity["mismatched"],
                extension_integrity["invalid_json"],
                extension_integrity["extra_checkpoint_ids"],
                extension_integrity["missing_checkpoint_ids"],
            )
        ),
        "both_aggregates_rebuild_exactly": initial_integrity["aggregate_matches"]
        and extension_integrity["aggregate_matches"],
        "extension_cells_exactly_preregistered": observed_cells == expected_cells,
        "randomized_task_seeds_unique_and_disjoint": seeds[
            "initial_task_seeds_unique"
        ]
        and seeds["extension_task_seeds_unique"]
        and not seeds["cross_batch_task_seed_overlap"],
        "trajectory_seeds_unique_and_disjoint": seeds[
            "initial_trajectory_seeds_unique"
        ]
        and seeds["extension_trajectory_seeds_unique"]
        and not seeds["cross_batch_trajectory_seed_overlap"],
        "same_opt2_compiler_context": compiler_ok,
        "cpu_only_and_threads_one": worker_environment_ok,
        "wrapper_exit_code_zero": wrapper["exit_code"] == 0,
        "compute_sources_unchanged": source_hashes_unchanged,
        "direct_rz_relative_se_gate_pass": all(
            row["checks"]["direct_rz_relative_se_within_2_percent"]
            for row in diagnostics.values()
        ),
        "holdout_5_percent_gate_pass": all(
            row["checks"]["selected_rz_holdout_within_5_percent"]
            and row["checks"]["selected_all_metrics_holdout_within_5_percent"]
            for row in diagnostics.values()
        ),
    }
    return {
        "status": "extension_complete_gates_pass" if all_proxy_pass else "extension_complete_gate_failed",
        "overall_pass": all(integrity_checks.values()) and all_proxy_pass,
        "integrity_checks": integrity_checks,
        "initial_batch": initial_integrity,
        "extension_batch": extension_integrity,
        "combined_counts": {
            "expected": 51,
            "completed": initial_integrity["completed"]
            + extension_integrity["completed"],
            "failed": initial_integrity["failed"] + extension_integrity["failed"],
        },
        "seed_audit": seeds,
        "wrapper_status": wrapper,
        "proxy_diagnostics": diagnostics,
        "gate_summary": {
            "all_proxy_cells_pass": all_proxy_pass,
            "maximum_direct_rz_relative_standard_error": max(
                row["maximum_direct_rz_relative_standard_error"]
                for row in diagnostics.values()
            ),
            "maximum_selected_rz_holdout_error": max(
                row["maximum_selected_rz_holdout_error"]
                for row in diagnostics.values()
            ),
            "maximum_selected_all_metrics_holdout_error": max(
                row["maximum_selected_all_metrics_holdout_error"]
                for row in diagnostics.values()
            ),
            "failing_cells": [
                key for key, row in diagnostics.items() if not row["pass"]
            ],
        },
        "direct_compiled_rz_measurements": _direct_rz_measurements(
            points, extension_keys=extension_keys
        ),
        "source_artifacts": {
            "initial_manifest_fingerprint": initial_manifest["manifest_fingerprint"],
            "extension_manifest_fingerprint": extension_manifest[
                "manifest_fingerprint"
            ],
            "initial_analysis_fingerprint": initial_analysis["content_fingerprint"],
            "initial_aggregate_fingerprint": initial_integrity[
                "aggregate_fingerprint"
            ],
            "extension_aggregate_fingerprint": extension_integrity[
                "aggregate_fingerprint"
            ],
            "extension_output_dir": _relative(extension_output, root),
        },
        "scope": {
            "molecule": "H4_chain",
            "basis": "STO-3G",
            "n_qubits": 8,
            "df_rank": 12,
            "final_total_cost_evaluation_performed": False,
            "scientific_superiority_claimed": False,
        },
    }


def evaluate_coherent_extension_analysis(
    *,
    project_root: str | Path,
    initial_manifest_path: str | Path,
    initial_output_dir: str | Path,
    extension_manifest_path: str | Path,
    extension_output_dir: str | Path,
    initial_analysis_path: str | Path,
    progress: Any = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve(strict=True)
    initial_manifest = load_task_manifest(initial_manifest_path, project_root=root)
    extension_manifest = load_task_manifest(extension_manifest_path, project_root=root)
    initial_analysis = _read_object(Path(initial_analysis_path).resolve(strict=True))
    validate_full_opt2_analysis_artifact(initial_analysis)
    evidence = load_and_validate_full_opt2_evidence(root)
    points, extension_keys = _load_combined_points(
        root=root,
        initial_manifest=initial_manifest,
        initial_output_dir=Path(initial_output_dir).resolve(strict=True),
        extension_manifest=extension_manifest,
        extension_output_dir=Path(extension_output_dir).resolve(strict=True),
        initial_analysis=initial_analysis,
        evidence=evidence,
    )
    diagnostics = _diagnostics(points)
    if not all(row["pass"] for row in diagnostics.values()):
        raise ValueError("Fresh-32 proxy gates did not all pass; refusing optimization.")
    deterministic = {
        "delta0.01": _deterministic_diagnostic(
            points[(12, 0.01, 0)], holdout_q=HOLDOUT_Q
        ),
        "delta0.02": _deterministic_diagnostic(
            points[(12, 0.02, 0)], holdout_q=(16, 32)
        ),
    }
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
    compute_fingerprint = fingerprint_payload(
        {
            "method": METHOD,
            "initial_manifest": initial_manifest["manifest_fingerprint"],
            "extension_manifest": extension_manifest["manifest_fingerprint"],
            "reused_opt2": evidence["compiler_raw"]["content_fingerprint"],
        }
    )
    models: dict[tuple[int, float], tuple[Any, ...]] = {}
    for delta in (0.01, 0.02):
        models[(3, delta)] = tuple(
            _pair_model(
                _model_points(points[(3, delta, rte_steps)]),
                rte_steps=rte_steps,
                finite_taylor_order=2,
                policy=POLICY_LABEL,
                source_fingerprint=compute_fingerprint,
                source_label=f"M06-F_fresh32_opt2_delta{delta:g}_r{rte_steps}",
            )
            for rte_steps in RTE_STEPS
        )
        models[(12, delta)] = (
            _pair_model(
                _model_points(points[(12, delta, 0)]),
                rte_steps=0,
                finite_taylor_order=0,
                policy=None,
                source_fingerprint=compute_fingerprint,
                source_label=f"M06-F_opt2_deterministic_delta{delta:g}",
            ),
        )
    candidates = {}
    for ld in (3, 12):
        for delta in (0.01, 0.02):
            maximum_round_index, pf_coefficient = _candidate_configuration(
                wp01, ld=ld, delta_time=delta
            )
            candidates[f"ld{ld}:delta{delta:g}"] = _grid_search(
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
    focused_comparison = focused["focused_fixed_plan_reaggregation"]["comparison"]
    coherent_ld3 = float(comparison["ld3_total_compiled_rz_point_estimate"])
    focused_ld3 = float(focused_comparison["ld3_total_compiled_rz_point_estimate"])
    mixed_to_coherent = {
        "ld3_compiled_rz_absolute_change": coherent_ld3 - focused_ld3,
        "ld3_compiled_rz_relative_change": coherent_ld3 / focused_ld3 - 1.0,
        "mixed_ld3_advantage_fraction": 1.0
        - float(focused_comparison["ld3_over_ld12_point_estimate_ratio"]),
        "coherent_ld3_advantage_fraction": 1.0
        - float(comparison["ld3_over_ld12_point_estimate_ratio"]),
    }
    mixed_to_coherent["advantage_fraction_change"] = (
        mixed_to_coherent["coherent_ld3_advantage_fraction"]
        - mixed_to_coherent["mixed_ld3_advantage_fraction"]
    )
    return {
        "status": "coherent_opt2_reoptimization_complete",
        "overall_pass": True,
        "configuration": dict(initial_analysis["configuration"]),
        "compute_fingerprint": compute_fingerprint,
        "proxy_diagnostics": diagnostics,
        "deterministic_diagnostics": deterministic,
        "direct_compiled_rz_measurements": _direct_rz_measurements(
            points, extension_keys=extension_keys
        ),
        "coherent_reoptimization": {
            "candidates": candidates,
            "best_by_ld": best_by_ld,
            "comparison": comparison,
        },
        "preparation_break_even": break_even,
        "focused_mixed_compiler_reference": {
            "content_fingerprint": focused["content_fingerprint"],
            "comparison": focused_comparison,
            "label": "mixed_opt1_opt2_focused_not_coherent",
        },
        "mixed_to_coherent_difference": mixed_to_coherent,
        "checks": {
            "complete_51_task_evidence_loaded": True,
            "all_proxy_cells_pass": True,
            "coherent_reoptimization_completed": True,
            "focused_and_coherent_results_are_separate": True,
            "final_total_cost_and_scientific_superiority_not_claimed": True,
        },
        "scope": {
            "coherent_optimization_level_2_context": True,
            "state_preparation_included": False,
            "backend_execution_included": False,
            "q_above_32_directly_validated": False,
            "final_total_cost_evaluation_performed": False,
            "scientific_superiority_claimed": False,
        },
    }


def _finalize(
    body: Mapping[str, Any],
    *,
    schema_version: str,
    stage: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": schema_version,
        "method": METHOD,
        "stage": stage,
        **dict(body),
        "provenance": dict(provenance),
    }
    payload["content_fingerprint"] = fingerprint(payload)
    validate_extension_artifact(payload, schema_version=schema_version)
    return payload


def finalize_extension_audit(
    body: Mapping[str, Any], *, provenance: Mapping[str, Any]
) -> dict[str, Any]:
    return _finalize(
        body,
        schema_version=AUDIT_SCHEMA_VERSION,
        stage="M06-F-fresh32-audit",
        provenance=provenance,
    )


def finalize_extension_analysis(
    body: Mapping[str, Any], *, provenance: Mapping[str, Any]
) -> dict[str, Any]:
    return _finalize(
        body,
        schema_version=ANALYSIS_SCHEMA_VERSION,
        stage="M06-F-fresh32-coherent-analysis",
        provenance=provenance,
    )


def validate_extension_artifact(
    payload: Mapping[str, Any], *, schema_version: str
) -> None:
    if payload.get("schema_version") != schema_version or payload.get("method") != METHOD:
        raise ValueError("Unsupported M06-F extension artifact schema or method.")
    unsigned = dict(payload)
    observed = unsigned.pop("content_fingerprint", None)
    if observed != fingerprint(unsigned):
        raise ValueError("M06-F extension artifact fingerprint mismatch.")
    if payload.get("overall_pass") is not True:
        raise ValueError("M06-F extension artifact did not pass its checks.")
    scope = payload.get("scope", {})
    if scope.get("final_total_cost_evaluation_performed") is not False:
        raise ValueError("M06-F extension cannot claim final total cost.")
    if scope.get("scientific_superiority_claimed") is not False:
        raise ValueError("M06-F extension cannot claim scientific superiority.")


def write_extension_artifact(payload: Mapping[str, Any], path: str | Path) -> None:
    output = Path(path)
    if output.exists():
        raise ValueError(f"Refusing to replace existing artifact: {output}")
    atomic_write_json(output, payload)
