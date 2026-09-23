"""WP05-bR focused replication of the failed r=32 q=8 proxy holdout."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping

from .df_hamiltonian import DFHamiltonian
from .df_partial_s2 import DFPartialS2Preparation
from .research_direction_full_scope import AXES, METRICS, POLICY_LABEL, fingerprint
from .research_direction_full_scope_extension import (
    _compile_randomized_delta,
    fit_affine_holdouts,
    validate_wp05b_artifact,
)
from .research_direction_sequence_policy import register_support_restricted_bases
from .rte import CompilerSettings


SCHEMA_VERSION = "research_direction_full_scope_replication_v1"
METHOD = "wp05br_r32_32trajectory_replication_v1"
DELTA_TIME = 0.02
RTE_STEPS = 32
Q_VALUES = (1, 2, 4, 8)
SAMPLE_COUNT = 32


def _errors(
    models: Mapping[str, Any],
    *,
    policy: str,
    q_m: int,
    metric: str | None,
) -> list[float]:
    metrics = METRICS if metric is None else (metric,)
    return [
        float(
            models[axis][name]["holdouts"][str(q_m)][
                "absolute_relative_error"
            ]
        )
        for axis in AXES
        for name in metrics
    ]


def _maximum_error_record(
    models: Mapping[str, Any],
    *,
    policy: str,
    q_m: int,
    metric: str | None,
    point: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = METRICS if metric is None else (metric,)
    candidates = []
    for axis in AXES:
        for name in metrics:
            holdout = models[axis][name]["holdouts"][str(q_m)]
            direct_se = float(
                point["axes"][axis]["policies"][policy][name][
                    "standard_error"
                ]
            )
            prediction_se = float(holdout["prediction_standard_error"])
            difference = abs(
                float(holdout["prediction"]) - float(holdout["direct_mean"])
            )
            combined_se = math.hypot(prediction_se, direct_se)
            candidates.append(
                {
                    "axis": axis,
                    "metric": name,
                    **dict(holdout),
                    "direct_standard_error": direct_se,
                    "combined_standard_error": combined_se,
                    "absolute_difference_over_combined_standard_error": (
                        math.inf if combined_se == 0.0 else difference / combined_se
                    ),
                }
            )
    return max(candidates, key=lambda row: row["absolute_relative_error"])


def evaluate_wp05br_replication(
    hamiltonian: DFHamiltonian,
    preparation: DFPartialS2Preparation,
    compiler: CompilerSettings,
    wp05b: Mapping[str, Any],
    wp06b: Mapping[str, Any],
    *,
    sample_count: int = SAMPLE_COUNT,
    master_seed: int = 2026092207,
    accuracy_threshold: float = 0.05,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Repeat the sole failed WP05-b cell with fresh larger samples."""
    validate_wp05b_artifact(wp05b)
    if wp05b["overall_pass"]:
        raise ValueError("WP05-bR expects the documented failed WP05-b input.")
    if preparation.ld != 3:
        raise ValueError("WP05-bR requires the fixed L_D=3 preparation.")
    if wp05b["physical_instance"]["hamiltonian_hash"] != (
        preparation.hamiltonian_hash
    ):
        raise ValueError("WP05-b and WP05-bR Hamiltonian hashes differ.")
    expected_failed = {
        "full_basis_q8_rz_holdout_within_5_percent",
        "selected_policy_q8_all_metrics_within_5_percent",
        "selected_policy_q8_rz_holdout_within_5_percent",
    }
    observed_failed = {
        name for name, passed in wp05b["checks"].items() if not passed
    }
    if observed_failed != expected_failed:
        raise ValueError("WP05-b failure set differs from the replication target.")

    support_definitions, proof_records = register_support_restricted_bases(
        hamiltonian, preparation
    )
    training_fingerprint = str(
        wp06b["training_selection"]["training_fingerprint"]
    )
    compiled, _probe = _compile_randomized_delta(
        preparation,
        support_definitions,
        compiler,
        delta_time=DELTA_TIME,
        q_values=Q_VALUES,
        schedule_rte_steps=(RTE_STEPS,),
        finite_taylor_order=2,
        sample_count=sample_count,
        master_seed=master_seed,
        training_fingerprint=training_fingerprint,
        probe=False,
        progress=progress,
    )
    points = {
        int(q): point for q, point in compiled[str(RTE_STEPS)]["points"].items()
    }
    models = {
        policy: fit_affine_holdouts(points, policy=policy)
        for policy in ("full_basis_shared", POLICY_LABEL)
    }
    selected_q8_rz = _errors(
        models[POLICY_LABEL], policy=POLICY_LABEL, q_m=8, metric="rz_count"
    )
    selected_q8_all = _errors(
        models[POLICY_LABEL], policy=POLICY_LABEL, q_m=8, metric=None
    )
    full_q8_rz = _errors(
        models["full_basis_shared"],
        policy="full_basis_shared",
        q_m=8,
        metric="rz_count",
    )
    selected_q4_all = _errors(
        models[POLICY_LABEL], policy=POLICY_LABEL, q_m=4, metric=None
    )
    selected_not_worse = True
    reductions = []
    for q_m, point in points.items():
        for axis in AXES:
            full = float(
                point["axes"][axis]["policies"]["full_basis_shared"][
                    "rz_count"
                ]["mean"]
            )
            selected = float(
                point["axes"][axis]["policies"][POLICY_LABEL]["rz_count"][
                    "mean"
                ]
            )
            relative = 0.0 if full == 0.0 else (selected - full) / full
            reductions.append(relative)
            selected_not_worse = selected_not_worse and relative <= 0.0
    proof_residual = max(
        float(row["preserved_columns_max_abs_residual"]) for row in proof_records
    )
    checks = {
        "fresh_seed_differs_from_wp05b": master_seed
        != int(wp05b["configuration"]["master_seed"]),
        "all_four_q_values_have_32_fresh_trajectories": all(
            int(points[q]["sample_count"]) == sample_count for q in Q_VALUES
        )
        and sample_count == SAMPLE_COUNT,
        "support_basis_certificates_pass": proof_residual <= 1e-10,
        "selected_policy_q8_rz_holdout_within_5_percent": (
            max(selected_q8_rz) <= accuracy_threshold
        ),
        "selected_policy_q8_all_metrics_within_5_percent": (
            max(selected_q8_all) <= accuracy_threshold
        ),
        "full_basis_q8_rz_holdout_within_5_percent": (
            max(full_q8_rz) <= accuracy_threshold
        ),
        "selected_policy_q4_all_metrics_within_5_percent": (
            max(selected_q4_all) <= accuracy_threshold
        ),
        "selected_policy_mean_rz_not_worse_at_direct_points": selected_not_worse,
        "production_default_remains_full_basis": True,
        "reoptimization_and_final_total_cost_not_claimed": True,
    }
    overall_pass = all(checks.values())
    next_action = (
        "WP01-D_C07_reoptimize_alpha_shots_and_schedules"
        if overall_pass
        else "fit_q1_q2_q4_weighted_proxy_then_use_a_second_fresh_q8_holdout"
    )
    q8_point = points[8]
    return {
        "configuration": {
            "molecule": "H4_chain",
            "geometry_angstrom": 1.0,
            "basis": "STO-3G",
            "n_qubits": hamiltonian.n_qubits,
            "df_rank": len(hamiltonian.lambdas),
            "ld": 3,
            "delta_time": DELTA_TIME,
            "finite_taylor_order": 2,
            "rte_steps": RTE_STEPS,
            "q_values": list(Q_VALUES),
            "calibration_q": [1, 2],
            "diagnostic_q": [4],
            "holdout_q": [8],
            "fresh_sample_count_per_q": sample_count,
            "master_seed": master_seed,
            "accuracy_threshold": accuracy_threshold,
            "compiler": dict(wp05b["configuration"]["compiler"]),
        },
        "physical_instance": dict(wp05b["physical_instance"]),
        "policy_input": {
            "policy": POLICY_LABEL,
            "wp06b_training_fingerprint": training_fingerprint,
            "production_default_changed": False,
            "maximum_preserved_columns_residual": proof_residual,
        },
        "upstream_wp05b": {
            "content_fingerprint": wp05b["content_fingerprint"],
            "failed_checks": sorted(observed_failed),
            "data_reused_in_replication_fit": False,
        },
        "direct_randomized_ld3": compiled[str(RTE_STEPS)],
        "affine_models": models,
        "diagnostics": {
            "maximum_selected_q8_rz": _maximum_error_record(
                models[POLICY_LABEL],
                policy=POLICY_LABEL,
                q_m=8,
                metric="rz_count",
                point=q8_point,
            ),
            "maximum_selected_q8_all_metrics": _maximum_error_record(
                models[POLICY_LABEL],
                policy=POLICY_LABEL,
                q_m=8,
                metric=None,
                point=q8_point,
            ),
            "maximum_full_q8_rz": _maximum_error_record(
                models["full_basis_shared"],
                policy="full_basis_shared",
                q_m=8,
                metric="rz_count",
                point=q8_point,
            ),
            "selected_minus_full_rz_relative_change_range": [
                min(reductions),
                max(reductions),
            ],
        },
        "decision": {
            "status": (
                "WP05bR_replication_passed"
                if overall_pass
                else "WP05bR_replication_failed"
            ),
            "next_action": next_action,
        },
        "scope": {
            "focused_r32_replication": True,
            "fresh_trajectories_only": True,
            "q8_directly_transpiled": True,
            "alpha_reoptimized": False,
            "shot_counts_reoptimized": False,
            "round_schedule_reoptimized": False,
            "decision_grade": False,
            "final_total_cost_evaluation_performed": False,
        },
        "limitations": [
            "This is a focused H4 delta=0.02 r=32 replication in one compiler context.",
            "The q=1,2 affine formula is retained without adapting it to the observed WP05-b q=8 result.",
            "No allocation, shot, or schedule reoptimization and no final total-cost or superiority claim is made.",
        ],
        "checks": checks,
        "overall_pass": overall_pass,
        "summary": {
            "status": (
                "WP05bR_replication_passed"
                if overall_pass
                else "WP05bR_replication_failed"
            ),
            "maximum_selected_policy_q8_rz_relative_error": max(selected_q8_rz),
            "maximum_selected_policy_q8_all_metric_relative_error": max(
                selected_q8_all
            ),
            "maximum_full_basis_q8_rz_relative_error": max(full_q8_rz),
            "maximum_selected_policy_q4_all_metric_relative_error": max(
                selected_q4_all
            ),
            "next_action": next_action,
        },
    }


def finalize_wp05br_artifact(
    body: Mapping[str, Any], *, provenance: Mapping[str, Any]
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "stage": "WP05-bR",
        **dict(body),
        "provenance": dict(provenance),
    }
    payload["content_fingerprint"] = fingerprint(payload)
    validate_wp05br_artifact(payload)
    return payload


def validate_wp05br_artifact(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported WP05-bR schema.")
    if payload.get("method") != METHOD or payload.get("stage") != "WP05-bR":
        raise ValueError("Unsupported WP05-bR method or stage.")
    unsigned = dict(payload)
    observed = unsigned.pop("content_fingerprint", None)
    if observed != fingerprint(unsigned):
        raise ValueError("WP05-bR content_fingerprint mismatch.")
    checks = payload.get("checks", {})
    if payload.get("overall_pass") != (bool(checks) and all(checks.values())):
        raise ValueError("WP05-bR overall status does not match its checks.")
    scope = payload.get("scope", {})
    if scope.get("fresh_trajectories_only") is not True:
        raise ValueError("WP05-bR requires fresh trajectories.")
    if scope.get("final_total_cost_evaluation_performed") is not False:
        raise ValueError("WP05-bR cannot claim final total cost.")
    if scope.get("decision_grade") is not False:
        raise ValueError("WP05-bR remains non-decision-grade.")


def write_wp05br_artifact(payload: Mapping[str, Any], path: str | Path) -> None:
    validate_wp05br_artifact(payload)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
