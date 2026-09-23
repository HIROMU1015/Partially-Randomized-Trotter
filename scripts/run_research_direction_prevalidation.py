#!/usr/bin/env python3
"""Run WP00, WP02, and the model-conditional WP01-S screening."""

from __future__ import annotations

import argparse
import json
import math
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import qiskit

from trotterlib.config import CA
from trotterlib.df_partial_randomized_pf import split_df_hamiltonian_by_ld
from trotterlib.df_partial_s2 import prepare_df_partial_s2
from trotterlib.research_direction_prevalidation import (
    affine_prediction_with_standard_error,
    build_horizon_audit,
    file_sha256,
    finalize_artifact,
    select_analytic_round_schedule,
    write_artifact,
)
from trotterlib.rpe_hadamard_compiled_cost_benchmark import (
    RPEHadamardCompiledCostBenchmarkDataset,
    RPEHadamardCompiledCostBenchmarkRequest,
    generate_rpe_hadamard_compiled_cost_benchmark_dataset,
)
from trotterlib.rpe_hadamard_compiled_cost_proxy import (
    RPEHadamardCompiledCostProxy,
    RPEHadamardCompiledCostProxyFitRequest,
    fit_rpe_hadamard_compiled_cost_proxy,
)
from trotterlib.rte import CompilerSettings, finite_rte_distribution, make_rte_config
from trotterlib.rte_compiled_cost import TranspiledCircuitCostCache
from trotterlib.rte_connected_cluster_cost_validation import (
    load_connected_cluster_hamiltonian_snapshot,
)


DEFAULT_SNAPSHOT = Path(
    "artifacts/rte_connected_cluster_cost_validation/"
    "h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k2_connected_"
    "pilot30_max1500_hold1500_rare375_v1.hamiltonian.npz"
)
DEFAULT_OUTPUT_DIR = Path(
    "artifacts/research_direction_prevalidation/2026-09-21"
)
ALLOCATION = Path(
    "artifacts/rpe_allocation_sensitivity_validation/2026-09-01/"
    "h4_sto3g_d100_rank12_ld3_dt0p1_r4_k2_q1_q2_q4_q8_"
    "beta_alpha_sensitivity_v1.json"
)
HORIZON = Path(
    "artifacts/rpe_target_round_horizon_validation/2026-09-20/"
    "h4_sto3g_d100_rank12_ld3_dt0p1_r4_k2_ca_over_10_horizon_v1.json"
)
SCHEDULE = Path(
    "artifacts/rpe_delta_round_schedule_validation/2026-09-20/"
    "h4_sto3g_d100_rank12_ld3_executed_delta_ca_over_10_"
    "round_schedule_v1.json"
)
CENTRAL_RTE_COST = Path(
    "artifacts/rpe_delta_compiled_cost_validation/2026-09-20/"
    "h4_ld3_delta_0p01_0p0125_0p02_round_schedule_rte_block_cost_v1.json"
)
HADAMARD_CONNECTION = Path(
    "artifacts/rpe_hadamard_proxy_resource_validation/2026-09-01/"
    "h4_sto3g_d100_rank12_ld3_dt0p1_r4_k2_cal_q1_q2_q4_"
    "hold_q8_mc8_v1.connection.json"
)
LEGACY_PF_LD3 = Path(
    "artifacts/pf_delta_validation/h4_sto3g_d100_rank12_ld3_v5.json"
)


def _git(command: list[str]) -> str | list[str] | None:
    result = subprocess.run(
        ["git", *command], check=False, capture_output=True, text=True
    )
    if result.returncode != 0:
        return None
    lines = result.stdout.splitlines()
    return lines[0] if len(lines) == 1 else lines


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _ref(path: Path, **extra: object) -> dict:
    return {"path": str(path), "sha256": file_sha256(path), **extra}


def _provenance() -> dict:
    sources = (
        Path("src/trotterlib/research_direction_prevalidation.py"),
        Path("scripts/run_research_direction_prevalidation.py"),
    )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_worktree_status_before_generation": _git(["status", "--short"]),
        "evidence_status": "local_worktree_validation_not_immutable_ci",
        "command": shlex.join(
            [
                ".venv311/bin/python",
                "scripts/run_research_direction_prevalidation.py",
                *sys.argv[1:],
            ]
        ),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "qiskit_version": qiskit.__version__,
        "source_sha256": {str(path): file_sha256(path) for path in sources},
    }


def _pf_path(output_dir: Path, ld: int) -> Path:
    return output_dir / "pf_delta_same_snapshot" / (
        f"h4_sto3g_d100_rank12_ld{ld}_v5.json"
    )


def _wp00(snapshot: Path, output_dir: Path) -> dict:
    pf = {ld: _json(_pf_path(output_dir, ld)) for ld in (0, 3, 12)}
    legacy_pf_ld3 = _json(LEGACY_PF_LD3)
    allocation = _json(ALLOCATION)
    schedule = _json(SCHEDULE)
    snapshot_hash = schedule["system"]["hamiltonian_hash"]
    hash_checks = {
        f"ld{ld}_pf_matches_fixed_snapshot": (
            item["hamiltonian"]["hamiltonian_hash"] == snapshot_hash
        )
        for ld, item in pf.items()
    }
    body = {
        "status": "complete_for_representative_h4_screening_contract",
        "comparison_contract": {
            "model": "H4 linear chain",
            "geometry_angstrom": 1.0,
            "basis": "STO-3G",
            "df_rank": 12,
            "n_qubits": 8,
            "physical_sector_n_electrons": 4,
            "physical_sector_dimension": 70,
            "input_state": "exact physical-sector ground state assumed available",
            "product_formula": "second_order_partial_S2",
            "candidate_ld_values": [0, 3, 12],
            "delta_candidates": [0.01, 0.0125, 0.02],
            "precision_scenarios_ha": {
                "CA": CA,
                "CA_over_10": CA / 10.0,
                "CA_over_100": CA / 100.0,
            },
            "provisional_primary_precision": "CA_over_10",
            "beta_rpe": 0.4,
            "phase_allocation": {
                "beta_pf": 0.02,
                "beta_rte": 0.02,
                "beta_stat": 0.36,
            },
            "alpha_total": 0.05,
            "alpha_policy": "uniform_across_all_round_axes_per_candidate",
            "circuit_scope": (
                "single_hadamard_interrogation_without_state_preparation"
            ),
            "control_convention": "ordinary_controlled_diag_I_U",
            "compiler": {
                "qiskit_version": qiskit.__version__,
                "basis_gates": ["rz", "sx", "x", "cx"],
                "optimization_level": 1,
                "transpiler_seed": 17,
                "backend": None,
                "coupling_map": None,
            },
            "primary_cost_metric": "rz_count",
            "secondary_cost_metrics": [
                "rz_depth",
                "cx_count",
                "cx_depth",
                "total_depth",
                "circuit_size",
            ],
            "state_preparation_included": False,
            "backend_execution_included": False,
            "quantum_shots_executed": 0,
        },
        "fixed_instance": {
            "snapshot": _ref(snapshot),
            "hamiltonian_hash": snapshot_hash,
            "candidate_fingerprints": {
                str(ld): {
                    "partition_hash": item["hamiltonian"]["partition_hash"],
                    "preparation_hash": item["hamiltonian"]["preparation_hash"],
                    "pf_validation_fingerprint": item["validation_fingerprint"],
                    "pf_artifact_overall_pass": item["summary"]["overall_pass"],
                    "paper_d6_estimator_validation_pass": item["summary"][
                        "paper_d6_estimator_validation_pass"
                    ],
                    "scalable_pf_coefficient": item["summary"][
                        "scalable_pf_fixed_second_order_coefficient"
                    ],
                    "coefficient_is_rigorous_bound": False,
                }
                for ld, item in pf.items()
            },
        },
        "evidence_registry": {
            "same_snapshot_pf": {
                str(ld): _ref(
                    _pf_path(output_dir, ld),
                    validation_fingerprint=item["validation_fingerprint"],
                )
                for ld, item in pf.items()
            },
            "allocation": _ref(
                ALLOCATION,
                content_fingerprint=allocation["content_fingerprint"],
            ),
            "target_horizon": _ref(HORIZON),
            "round_schedule": _ref(
                SCHEDULE, content_fingerprint=schedule["content_fingerprint"]
            ),
            "central_rte_cost": _ref(CENTRAL_RTE_COST),
            "short_q_full_hadamard_connection": _ref(HADAMARD_CONNECTION),
        },
        "legacy_join_diagnostic": {
            "legacy_pf_ld3": _ref(
                LEGACY_PF_LD3,
                validation_fingerprint=legacy_pf_ld3["validation_fingerprint"],
            ),
            "legacy_hamiltonian_hash": legacy_pf_ld3["hamiltonian"][
                "hamiltonian_hash"
            ],
            "fixed_snapshot_hamiltonian_hash": snapshot_hash,
            "fingerprints_match": (
                legacy_pf_ld3["hamiltonian"]["hamiltonian_hash"]
                == snapshot_hash
            ),
            "coefficient_relative_difference_after_same_snapshot_regeneration": (
                abs(
                    float(
                        legacy_pf_ld3["summary"][
                            "scalable_pf_fixed_second_order_coefficient"
                        ]
                    )
                    - float(
                        pf[3]["summary"][
                            "scalable_pf_fixed_second_order_coefficient"
                        ]
                    )
                )
                / float(
                    pf[3]["summary"][
                        "scalable_pf_fixed_second_order_coefficient"
                    ]
                )
            ),
            "action": (
                "do_not_join_legacy_PF_artifact_by_fingerprint;_use_the_"
                "same_snapshot_regeneration_for_this_contract"
            ),
        },
        "source_version_table": {
            "repository_commit": _git(["rev-parse", "HEAD"]),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "qiskit": qiskit.__version__,
            "pf_artifact_schema": pf[3]["schema_version"],
            "schedule_artifact_schema": schedule["schema_version"],
        },
        "coverage": {
            "A01_A03_B01_B05_B08_N03": {
                "current_status": "satisfied_for_representative_H4_contract",
                "reused_evidence": list(
                    (_ref(_pf_path(output_dir, ld))["path"] for ld in (0, 3, 12))
                ),
                "covered_domain": "one H4 rank-12 snapshot and three L_D values",
                "remaining_delta": (
                    "cross-instance portability and immutable external reproduction"
                ),
                "blocked_by": None,
            }
        },
        "checks": {
            **hash_checks,
            "all_pf_inputs_match_fixed_snapshot": all(hash_checks.values()),
            "allocation_horizon_and_schedule_paths_exist": all(
                path.exists()
                for path in (ALLOCATION, HORIZON, SCHEDULE, CENTRAL_RTE_COST)
            ),
        },
        "known_exclusions": [
            "stale_DF_screening_values",
            "prose_only_UWC_values",
            "state_preparation_cost",
            "noise_and_backend_execution",
            "final_total_cost_claim",
        ],
    }
    body["overall_pass"] = all(body["checks"].values())
    return finalize_artifact(stage="WP00", body=body, provenance=_provenance())


def _wp02(output_dir: Path) -> dict:
    pf = _json(_pf_path(output_dir, 3))
    schedule = _json(SCHEDULE)
    coefficient = float(
        pf["summary"]["scalable_pf_fixed_second_order_coefficient"]
    )
    rows = build_horizon_audit(
        precision_scenarios=(
            ("CA", CA),
            ("CA_over_10", CA / 10.0),
            ("CA_over_100", CA / 100.0),
        ),
        delta_values=(0.01, 0.0125, 0.02),
        beta_rpe=0.4,
        beta_pf_budget=0.02,
        pf_coefficient=coefficient,
        direct_wrapper_q_max=8,
        existing_schedule_precision=CA / 10.0,
        existing_schedule_deltas=(0.01, 0.0125, 0.02),
    )
    by_precision = {}
    for label in ("CA", "CA_over_10", "CA_over_100"):
        selected = [row for row in rows if row["precision_label"] == label]
        by_precision[label] = {
            "q_max_range": [
                min(int(row["q_max"]) for row in selected),
                max(int(row["q_max"]) for row in selected),
            ],
            "all_empirical_pf_screens_pass": all(
                row["empirical_pf_screen_pass"] for row in selected
            ),
            "exact_existing_schedule_matches": sum(
                bool(row["existing_round_schedule_exact_condition_match"])
                for row in selected
            ),
            "first_constraint": (
                "cost_proxy_and_failure_allocation_revalidation"
                if label == "CA"
                else (
                    "none_within_existing_CA_over_10_matrix_scope"
                    if label == "CA_over_10"
                    else "empirical_PF_budget_and_unvalidated_long_q_cost_domain"
                )
            ),
        }
    body = {
        "status": "complete_as_coverage_audit",
        "scope": {
            "new_matrix_validation_performed": False,
            "new_q_greater_than_8_circuit_compilation_performed": False,
            "existing_CA_over_10_matrix_checks_reused": 56,
            "final_total_cost_evaluation_performed": False,
        },
        "configuration": {
            "beta_rpe": 0.4,
            "beta_pf_budget": 0.02,
            "pf_coefficient": coefficient,
            "pf_coefficient_source": (
                "same_snapshot_paper_D6_empirical_coefficient"
            ),
            "pf_coefficient_is_rigorous_bound": False,
            "direct_full_wrapper_q_domain": [1, 8],
            "central_rte_proxy_ld_domain": [3],
            "central_rte_proxy_r_domain": [1, 32],
            "central_rte_proxy_short_step_time_domain": [0.000390625, 0.02],
            "central_rte_proxy_includes_control_and_outer_boundaries": False,
        },
        "round_horizon_matrix": rows,
        "precision_summary": by_precision,
        "checks": {
            "all_horizons_are_minimal_powers_of_two": all(
                row["minimal_power_of_two_horizon"] for row in rows
            ),
            "all_CA_over_10_rows_match_existing_schedules": all(
                row["existing_round_schedule_exact_condition_match"]
                for row in rows
                if row["precision_label"] == "CA_over_10"
            ),
            "CA_over_100_gap_is_detected": not any(
                row["empirical_pf_screen_pass"]
                for row in rows
                if row["precision_label"] == "CA_over_100"
            ),
            "reused_schedule_passed_original_checks": schedule["summary"][
                "overall_pass"
            ],
        },
        "interpretation": (
            "CA_over_10_is_covered_by_existing_schedule_and_matrix_evidence;_"
            "CA_needs_reallocated_shots_before_cost_use;_CA_over_100_requires_"
            "smaller_delta_and_new_schedule_before_cost_compilation"
        ),
    }
    body["overall_pass"] = all(body["checks"].values())
    return finalize_artifact(stage="WP02", body=body, provenance=_provenance())


def _compiler() -> CompilerSettings:
    return CompilerSettings(
        basis_gates=("rz", "sx", "x", "cx"),
        backend_name=None,
        coupling_map=None,
        optimization_level=1,
        layout_method=None,
        routing_method=None,
        transpiler_seed=17,
        qiskit_version=qiskit.__version__,
    )


def _calibration(
    preparation,
    *,
    ld: int,
    r_m: int,
    k_m: int,
    sample_count: int,
    output_dir: Path,
) -> tuple[RPEHadamardCompiledCostBenchmarkDataset, RPEHadamardCompiledCostProxy, dict]:
    evaluation_label = (
        "exact" if preparation.is_deterministic_only else f"mc{sample_count}"
    )
    stem = f"ld{ld}_dt0p02_r{r_m}_k{k_m}_q1_q2_{evaluation_label}"
    dataset_path = output_dir / "wp01s_calibrations" / f"{stem}.dataset.json"
    proxy_path = output_dir / "wp01s_calibrations" / f"{stem}.proxy.json"
    if dataset_path.exists() and proxy_path.exists():
        dataset = RPEHadamardCompiledCostBenchmarkDataset.read_json(dataset_path)
        proxy = RPEHadamardCompiledCostProxy.read_json(proxy_path)
    else:
        if preparation.is_deterministic_only:
            config = None
            distribution = None
            evaluation_method = "exact"
            samples = None
            seed = None
        else:
            distribution = finite_rte_distribution(
                preparation.exact_rte_lambda_r * 0.02 / r_m,
                k_m,
            )
            config, distribution = make_rte_config(
                preparation.rte_preparation.symbolic_tail,
                evolution_time=0.02,
                rte_steps=r_m,
                truncation_tolerance=max(
                    distribution.step_truncation_residual_bound,
                    math.ulp(0.0),
                ),
                finite_taylor_order=k_m,
                seed=20260818,
            )
            evaluation_method = "monte_carlo"
            samples = sample_count
            seed = 20260921 + 100 * ld + 10 * r_m + k_m
        result = generate_rpe_hadamard_compiled_cost_benchmark_dataset(
            RPEHadamardCompiledCostBenchmarkRequest(
                preparation=preparation,
                delta_time=0.02,
                calibration_repetition_counts=(1, 2),
                holdout_repetition_counts=(),
                rte_steps_per_occurrence=r_m,
                finite_taylor_order=k_m,
                rte_config=config,
                rte_distribution=distribution,
                compiler=_compiler(),
                evaluation_method=evaluation_method,
                sample_count=samples,
                seed=seed,
                generation_id=(
                    f"research-direction-WP01-S-{stem}-2026-09-21"
                ),
                maximum_repetition_count=2,
                maximum_samples=max(2, sample_count),
                maximum_retained_trajectory_records=max(2, sample_count),
                maximum_untranspiled_circuit_size=500_000,
                maximum_planned_instruction_applications=500_000_000,
                cache=TranspiledCircuitCostCache(),
            )
        )
        dataset = result.dataset
        if not dataset.complete:
            failures = [
                record.failure_reason
                for record in dataset.records
                if record.status == "failed"
            ]
            raise RuntimeError(f"Incomplete WP01-S calibration: {failures}")
        proxy = fit_rpe_hadamard_compiled_cost_proxy(
            RPEHadamardCompiledCostProxyFitRequest(dataset=dataset)
        )
        dataset.write_json(dataset_path)
        proxy.write_json(proxy_path)
    return dataset, proxy, {
        "dataset": _ref(
            dataset_path, dataset_fingerprint=dataset.dataset_fingerprint
        ),
        "proxy": _ref(proxy_path, proxy_fingerprint=proxy.proxy_fingerprint),
    }


def _axis_stats(dataset, axis: str) -> dict[int, tuple[float, float]]:
    result = {}
    for point in dataset.records:
        if point.partition != "calibration" or point.axis != axis:
            continue
        stats = dict(point.metric_statistics)["rz_count"]
        result[point.q_m] = (
            float(stats.mean),
            0.0 if stats.standard_error is None else float(stats.standard_error),
        )
    return result


def _cost_schedule(schedule: dict, calibration_map: dict) -> dict:
    total = 0.0
    conservative_se = 0.0
    rows = []
    for row in schedule["rounds"]:
        key = (int(row["r_m"]), int(row["K_m"]))
        dataset, proxy, _reference = calibration_map[key]
        axis_rows = {}
        round_total = 0.0
        round_se = 0.0
        for axis in ("cosine", "sine"):
            stats = _axis_stats(dataset, axis)
            prediction = proxy.model(axis, "rz_count").predict(int(row["q_m"]))
            _same_prediction, standard_error = affine_prediction_with_standard_error(
                q_m=int(row["q_m"]),
                q1_mean=stats[1][0],
                q2_mean=stats[2][0],
                q1_standard_error=stats[1][1],
                q2_standard_error=stats[2][1],
            )
            if prediction < 0.0:
                raise RuntimeError("Affine WP01-S cost prediction became negative.")
            shots = int(row[f"{axis}_shots"])
            round_total += shots * prediction
            round_se += shots * standard_error
            axis_rows[axis] = {
                "shots": shots,
                "predicted_rz_count_per_interrogation": prediction,
                "propagated_calibration_standard_error": standard_error,
            }
        total += round_total
        conservative_se += round_se
        rows.append(
            {
                "round_index": row["round_index"],
                "q_m": row["q_m"],
                "r_m": row["r_m"],
                "K_m": row["K_m"],
                "axes": axis_rows,
                "round_total_rz_count_point_estimate": round_total,
            }
        )
    statistical_half_width = 1.96 * conservative_se
    return {
        "rounds": rows,
        "total_rz_count_point_estimate": total,
        "conservative_calibration_95_half_width": statistical_half_width,
        "scenario_intervals": {
            "local_5_percent_plus_calibration": [
                max(0.0, 0.95 * total - statistical_half_width),
                1.05 * total + statistical_half_width,
            ],
            "transfer_25_percent_plus_calibration": [
                max(0.0, 0.75 * total - statistical_half_width),
                1.25 * total + statistical_half_width,
            ],
        },
    }


def _wp01s(snapshot: Path, output_dir: Path, samples: int) -> dict:
    hamiltonian = load_connected_cluster_hamiltonian_snapshot(snapshot)
    preparations = {
        ld: prepare_df_partial_s2(
            hamiltonian,
            split_df_hamiltonian_by_ld(hamiltonian, ld),
            identity_policy="extract_identity_phase",
        )
        for ld in (0, 3, 12)
    }
    coefficients = {
        ld: float(
            _json(_pf_path(output_dir, ld))["summary"][
                "scalable_pf_fixed_second_order_coefficient"
            ]
        )
        for ld in (0, 3, 12)
    }
    schedules = {
        ld: {
            delta: select_analytic_round_schedule(
                preparations[ld],
                delta_time=delta,
                target_energy_precision=CA / 10.0,
                pf_coefficient=coefficients[ld],
                **(
                    {
                        "rte_step_values": tuple(
                            1 << exponent for exponent in range(17)
                        ),
                        "finite_taylor_orders": tuple(range(0, 18, 2)),
                    }
                    if ld == 0
                    else {}
                ),
            )
            for delta in (0.01, 0.0125, 0.02)
        }
        for ld in (0, 3, 12)
    }

    calibration_maps = {}
    calibration_refs = {}
    for ld in (3, 12):
        pairs = sorted(
            {
                (int(row["r_m"]), int(row["K_m"]))
                for schedule in schedules[ld].values()
                for row in schedule["rounds"]
            }
        )
        calibration_maps[ld] = {}
        calibration_refs[str(ld)] = {}
        for r_m, k_m in pairs:
            dataset, proxy, reference = _calibration(
                preparations[ld],
                ld=ld,
                r_m=r_m,
                k_m=k_m,
                sample_count=samples,
                output_dir=output_dir,
            )
            calibration_maps[ld][(r_m, k_m)] = (dataset, proxy, reference)
            calibration_refs[str(ld)][f"r{r_m}_k{k_m}"] = reference

    candidates = []
    for ld in (0, 3, 12):
        for delta, schedule in schedules[ld].items():
            record = {
                "ld": ld,
                "delta_time": delta,
                "schedule": schedule,
                "pf_coefficient": coefficients[ld],
                "pf_coefficient_is_rigorous_bound": False,
            }
            if ld == 0:
                record.update(
                    {
                        "screening_status": (
                            "screened_out_before_compiled_cost_by_expanded_"
                            "component_application_proxy"
                        ),
                        "compiled_cost_evaluated": False,
                        "no_prep_total_rz_count": None,
                    }
                )
            else:
                record.update(
                    {
                        "screening_status": "model_conditional_cost_available",
                        "compiled_cost_evaluated": True,
                        "no_prep_total_rz_count": _cost_schedule(
                            schedule, calibration_maps[ld]
                        ),
                    }
                )
            candidates.append(record)

    costed = [item for item in candidates if item["compiled_cost_evaluated"]]
    best_by_ld = {}
    for ld in (3, 12):
        selected = min(
            (item for item in costed if item["ld"] == ld),
            key=lambda item: item["no_prep_total_rz_count"][
                "total_rz_count_point_estimate"
            ],
        )
        best_by_ld[str(ld)] = {
            "delta_time": selected["delta_time"],
            **selected["no_prep_total_rz_count"],
        }
    best = min(
        best_by_ld,
        key=lambda ld: best_by_ld[ld]["total_rz_count_point_estimate"],
    )
    runner_up = "12" if best == "3" else "3"
    best_interval = best_by_ld[best]["scenario_intervals"][
        "transfer_25_percent_plus_calibration"
    ]
    runner_interval = best_by_ld[runner_up]["scenario_intervals"][
        "transfer_25_percent_plus_calibration"
    ]
    overlap = max(best_interval[0], runner_interval[0]) <= min(
        best_interval[1], runner_interval[1]
    )
    local_ld3 = best_by_ld["3"]["scenario_intervals"][
        "local_5_percent_plus_calibration"
    ]
    local_ld12 = best_by_ld["12"]["scenario_intervals"][
        "local_5_percent_plus_calibration"
    ]
    local_overlap = max(local_ld3[0], local_ld12[0]) <= min(
        local_ld3[1], local_ld12[1]
    )
    ld3_point = float(best_by_ld["3"]["total_rz_count_point_estimate"])
    ld12_point = float(best_by_ld["12"]["total_rz_count_point_estimate"])
    ld0_best = min(
        (item for item in candidates if item["ld"] == 0),
        key=lambda item: float(
            item["schedule"][
                "total_shot_weighted_randomized_component_application_proxy"
            ]
        ),
    )
    ld3_best_schedule = next(
        item["schedule"]
        for item in candidates
        if item["ld"] == 3
        and item["delta_time"] == best_by_ld["3"]["delta_time"]
    )
    ld0_component_proxy = float(
        ld0_best["schedule"][
            "total_shot_weighted_randomized_component_application_proxy"
        ]
    )
    ld3_component_proxy = float(
        ld3_best_schedule[
            "total_shot_weighted_randomized_component_application_proxy"
        ]
    )
    body = {
        "status": "model_conditional_screening_complete",
        "scope": {
            "comparison_task": "WP00 fixed H4 CA_over_10 task",
            "circuit_cost_scope": (
                "single_hadamard_interrogation_without_state_preparation"
            ),
            "short_q_direct_calibration": [1, 2],
            "long_q_cost_method": "axis_metric_affine_q1_q2_extrapolation",
            "holdout_used_for_this_schedule_family": False,
            "angle_transfer_from_delta_0p02": True,
            "state_preparation_included": False,
            "backend_execution_included": False,
            "quantum_shots_executed": 0,
            "final_total_cost_evaluation_performed": False,
            "decision_grade": False,
        },
        "configuration": {
            "candidate_ld_values": [0, 3, 12],
            "delta_candidates": [0.01, 0.0125, 0.02],
            "target_energy_precision_ha": CA / 10.0,
            "beta_rpe": 0.4,
            "beta_pf_budget": 0.02,
            "beta_rte_budget": 0.02,
            "beta_stat_budget": 0.36,
            "alpha_total": 0.05,
            "compiler": {
                "basis_gates": ["rz", "sx", "x", "cx"],
                "optimization_level": 1,
                "transpiler_seed": 17,
                "qiskit_version": qiskit.__version__,
                "backend": None,
                "coupling_map": None,
            },
            "randomized_calibration_sample_count": samples,
            "ld0_expanded_analytic_grid": {
                "rte_step_values": [1 << exponent for exponent in range(17)],
                "finite_taylor_orders": list(range(0, 18, 2)),
            },
        },
        "calibrations": calibration_refs,
        "candidates": candidates,
        "summary": {
            "ld0_best_delta_by_component_application_proxy": ld0_best[
                "delta_time"
            ],
            "ld0_total_shots_at_proxy_optimum": ld0_best["schedule"][
                "total_shots"
            ],
            "ld0_minimum_radius_lower_bound_at_proxy_optimum": ld0_best[
                "schedule"
            ]["minimum_conservative_radius_lower_bound"],
            "ld0_component_application_proxy_at_optimum": ld0_component_proxy,
            "ld3_component_application_proxy_at_cost_optimum": (
                ld3_component_proxy
            ),
            "ld0_over_ld3_component_application_proxy_ratio": (
                ld0_component_proxy / ld3_component_proxy
            ),
            "ld0_selected_r_grid_boundary_hit": ld0_best["schedule"][
                "selected_r_grid_boundary_hit"
            ],
            "ld0_selected_K_grid_boundary_hit": ld0_best["schedule"][
                "selected_K_grid_boundary_hit"
            ],
            "best_costed_candidate_ld": int(best),
            "runner_up_ld": int(runner_up),
            "best_by_ld": best_by_ld,
            "ld3_over_ld12_point_estimate_ratio": ld3_point / ld12_point,
            "ld12_reduction_relative_to_ld3_point_estimate": (
                (ld3_point - ld12_point) / ld3_point
            ),
            "local_5_percent_intervals_overlap": local_overlap,
            "conservative_intervals_overlap": overlap,
            "directional_result": (
                "undetermined_between_intermediate_and_deterministic_endpoint"
                if overlap
                else f"conditional_point_and_interval_favor_ld{best}"
            ),
            "ld0_directional_result": (
                "strongly_disfavored_within_declared_analytic_component_"
                "application_screen"
            ),
            "next_required_validation": (
                "WP04_then_WP03;_full_scope_WP05_is_required_before_WP01-D"
            ),
        },
        "limitations": [
            "PF coefficients are empirical rather than rigorous upper bounds.",
            (
                "The q=1,2 affine cost fits have no unused holdout for these "
                "exact schedule families."
            ),
            (
                "The delta=0.02 compiled calibration is transferred to "
                "delta=0.01 and 0.0125."
            ),
            "The 5% and 25% widths are sensitivity scenarios, not confidence bounds.",
            "Fresh IID RTE trajectories per quantum shot are assumed, not executed.",
            (
                "The L_D=0 screen compares analytic component applications with "
                "L_D=3, not compiled RZ cost."
            ),
            (
                "No state preparation, noise, backend run, or RPE phase "
                "reconstruction is included."
            ),
        ],
    }
    return finalize_artifact(stage="WP01-S", body=body, provenance=_provenance())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("all", "wp00", "wp02", "wp01s"), default="all"
    )
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--samples", type=int, default=2)
    args = parser.parse_args()
    if args.samples < 2:
        raise ValueError("--samples must be at least two.")
    outputs = []
    stages = ("wp00", "wp02", "wp01s") if args.stage == "all" else (args.stage,)
    for stage in stages:
        if stage == "wp00":
            payload = _wp00(args.snapshot, args.output_dir)
            output = args.output_dir / "wp00_comparison_contract_v1.json"
        elif stage == "wp02":
            payload = _wp02(args.output_dir)
            output = args.output_dir / "wp02_round_horizon_coverage_v1.json"
        else:
            payload = _wp01s(args.snapshot, args.output_dir, args.samples)
            output = args.output_dir / "wp01s_model_conditional_screening_v1.json"
        write_artifact(payload, output)
        outputs.append(
            {
                "stage": payload["stage"],
                "output": str(output),
                "overall_pass": payload.get("overall_pass"),
                "status": payload.get("status"),
                "content_fingerprint": payload["content_fingerprint"],
            }
        )
        print(json.dumps(outputs[-1], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
