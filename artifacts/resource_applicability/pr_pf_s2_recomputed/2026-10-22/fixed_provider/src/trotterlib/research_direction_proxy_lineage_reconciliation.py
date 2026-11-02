"""Reconcile the latest fresh q=1,2 proxy with legacy opt2 q=16,32 holdouts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from .parallel_validation_executor import atomic_write_json
from .research_direction_compiler_transfer_compute import (
    validate_compiler_transfer_compute_artifact,
)
from .research_direction_full_opt2 import (
    EXPECTED_COMPILER,
    EXPECTED_COMPILER_RAW_FINGERPRINT,
    EXPECTED_SNAPSHOT_SHA256,
    POLICY_LABEL,
)
from .research_direction_full_opt2_extension_analysis import (
    ANALYSIS_SCHEMA_VERSION,
    validate_extension_artifact,
)
from .research_direction_full_scope import AXES, fingerprint
from .research_direction_prevalidation import (
    affine_prediction_with_standard_error,
)


SCHEMA_VERSION = "research_direction_proxy_lineage_reconciliation_v1"
METHOD = "a0_fresh_q1_q2_legacy_q16_q32_fixed_holdout_v1"
STAGE = "A0-proxy-lineage-reconciliation"
EXPECTED_LATEST_FINGERPRINT = (
    "5ce368a94daa39680b4edc0cfb59b30168d8bc159ad2538b29cb67b928e3cdba"
)
TARGET_CELL = "ld3:delta0.02:r32"
TARGET_DELTA = 0.02
TARGET_R = 32
TARGET_K = 2
CALIBRATION_Q = (1, 2)
FRESH_HOLDOUT_Q = (8,)
LEGACY_HOLDOUT_Q = (16, 32)
POLICIES = (POLICY_LABEL, "full_basis_shared")
RELATIVE_ERROR_LIMIT = 0.05
DIRECT_RZ_RELATIVE_SE_LIMIT = 0.02


def _rz_record(
    point: Mapping[str, Any], *, axis: str, policy: str
) -> Mapping[str, Any]:
    return point["axes"][axis]["policies"][policy]["rz_count"]


def _require_matching_context(
    latest: Mapping[str, Any], legacy: Mapping[str, Any]
) -> dict[str, Any]:
    latest_config = latest["configuration"]
    legacy_config = legacy["configuration"]
    shared_fields = (
        "molecule",
        "geometry_angstrom",
        "basis",
        "n_qubits",
        "df_rank",
        "finite_taylor_order",
    )
    field_matches = {
        field: latest_config[field] == legacy_config[field]
        for field in shared_fields
    }
    checks = {
        **{f"same_{field}": value for field, value in field_matches.items()},
        "same_compiler": latest_config["compiler"] == legacy_config["compiler"],
        "expected_compiler": {
            key: latest_config["compiler"].get(key) for key in EXPECTED_COMPILER
        }
        == EXPECTED_COMPILER,
        "qiskit_1p3p0": latest_config["compiler"].get("qiskit_version")
        == "1.3.0",
        "target_delta_present": TARGET_DELTA in latest_config["delta_values"]
        and float(legacy_config["delta_time"]) == TARGET_DELTA,
        "target_r_present": TARGET_R in latest_config["r_values"]
        and int(legacy_config["rte_steps"]) == TARGET_R,
        "target_k_matches": int(latest_config["finite_taylor_order"]) == TARGET_K
        and int(legacy_config["finite_taylor_order"]) == TARGET_K,
        "legacy_snapshot_matches": legacy["source_evidence"]["snapshot"][
            "sha256"
        ]
        == EXPECTED_SNAPSHOT_SHA256,
    }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise ValueError(f"A0 compiler/snapshot context mismatch: {failed}")
    return checks


def _source_contract(
    latest: Mapping[str, Any], legacy: Mapping[str, Any]
) -> dict[str, Any]:
    direct = latest["direct_compiled_rz_measurements"][TARGET_CELL]
    old_points = legacy["direct_points"]["3"]
    checks = {
        "latest_cell_is_fresh32_extension": direct["source_partition"]
        == "fresh32_extension",
        "latest_calibration_q_exact": set(CALIBRATION_Q).issubset(
            {int(q) for q in direct["q"]}
        ),
        "latest_calibration_sample_count_32": all(
            int(direct["q"][str(q)]["sample_count"]) == 32
            for q in CALIBRATION_Q
        ),
        "legacy_holdout_q_exact": set(LEGACY_HOLDOUT_Q).issubset(
            {int(q) for q in old_points}
        ),
        "legacy_holdouts_are_m08_fresh": all(
            old_points[str(q)]["partition"] == "m08_fresh_holdout"
            for q in LEGACY_HOLDOUT_Q
        ),
        "legacy_holdout_sample_count_8": all(
            int(old_points[str(q)]["sample_count"]) == 8
            for q in LEGACY_HOLDOUT_Q
        ),
        "fresh_and_legacy_source_partitions_separate": (
            direct["source_partition"] == "fresh32_extension"
            and all(
                old_points[str(q)]["partition"] == "m08_fresh_holdout"
                for q in LEGACY_HOLDOUT_Q
            )
        ),
        "policies_and_axes_present": all(
            policy in direct["q"][str(q)]["compiled_rz"]
            and all(
                axis in direct["q"][str(q)]["compiled_rz"][policy]
                for axis in AXES
            )
            for q in CALIBRATION_Q
            for policy in POLICIES
        )
        and all(
            policy in old_points[str(q)]["axes"][axis]["policies"]
            for q in LEGACY_HOLDOUT_Q
            for axis in AXES
            for policy in POLICIES
        ),
    }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise ValueError(f"A0 source-lineage mismatch: {failed}")
    return checks


def evaluate_proxy_lineage_reconciliation(
    latest: Mapping[str, Any], legacy: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply only the latest fresh q=1,2 fit to fixed legacy q=16,32 points."""
    validate_extension_artifact(latest, schema_version=ANALYSIS_SCHEMA_VERSION)
    validate_compiler_transfer_compute_artifact(legacy)
    if latest["content_fingerprint"] != EXPECTED_LATEST_FINGERPRINT:
        raise ValueError("A0 latest coherent-analysis fingerprint changed.")
    if legacy["content_fingerprint"] != EXPECTED_COMPILER_RAW_FINGERPRINT:
        raise ValueError("A0 legacy opt2 holdout fingerprint changed.")

    context_checks = _require_matching_context(latest, legacy)
    lineage_checks = _source_contract(latest, legacy)
    direct = latest["direct_compiled_rz_measurements"][TARGET_CELL]
    old_points = legacy["direct_points"]["3"]

    models: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for policy in POLICIES:
        models[policy] = {}
        for axis in AXES:
            q1 = direct["q"]["1"]["compiled_rz"][policy][axis]
            q2 = direct["q"]["2"]["compiled_rz"][policy][axis]
            model_source = {
                "source_content_fingerprint": latest["content_fingerprint"],
                "cell": TARGET_CELL,
                "policy": policy,
                "axis": axis,
                "metric": "rz_count",
                "calibration": {"1": dict(q1), "2": dict(q2)},
            }
            model = {
                "formula": "(2-q)*mean_q1+(q-1)*mean_q2",
                "slope": float(q2["mean"]) - float(q1["mean"]),
                "intercept": 2.0 * float(q1["mean"]) - float(q2["mean"]),
                "q1_mean": float(q1["mean"]),
                "q1_standard_error": float(q1["standard_error"]),
                "q2_mean": float(q2["mean"]),
                "q2_standard_error": float(q2["standard_error"]),
                "model_fingerprint": fingerprint(model_source),
            }
            models[policy][axis] = model
            for q_m in LEGACY_HOLDOUT_Q:
                prediction, prediction_se = affine_prediction_with_standard_error(
                    q_m=q_m,
                    q1_mean=model["q1_mean"],
                    q2_mean=model["q2_mean"],
                    q1_standard_error=model["q1_standard_error"],
                    q2_standard_error=model["q2_standard_error"],
                )
                observed = _rz_record(
                    old_points[str(q_m)], axis=axis, policy=policy
                )
                direct_mean = float(observed["mean"])
                direct_se = float(observed["standard_error"])
                residual = prediction - direct_mean
                combined_se = math.hypot(prediction_se, direct_se)
                rows.append(
                    {
                        "policy": policy,
                        "axis": axis,
                        "metric": "rz_count",
                        "q_m": q_m,
                        "model_fingerprint": model["model_fingerprint"],
                        "prediction": prediction,
                        "prediction_standard_error": prediction_se,
                        "direct_mean": direct_mean,
                        "direct_standard_error": direct_se,
                        "direct_relative_standard_error": abs(
                            direct_se / direct_mean
                        ),
                        "signed_residual": residual,
                        "absolute_residual": abs(residual),
                        "signed_relative_error": residual / direct_mean,
                        "absolute_relative_error": abs(residual) / direct_mean,
                        "combined_standard_error": combined_se,
                        "absolute_standardized_residual": (
                            abs(residual) / combined_se
                            if combined_se > 0.0
                            else 0.0
                        ),
                        "within_5_percent": abs(residual) / direct_mean
                        <= RELATIVE_ERROR_LIMIT,
                        "within_1p96_combined_standard_errors": (
                            abs(residual) <= 1.96 * combined_se
                        ),
                        "holdout_partition": old_points[str(q_m)]["partition"],
                        "holdout_sample_count": int(
                            old_points[str(q_m)]["sample_count"]
                        ),
                    }
                )

    selected_rows = [row for row in rows if row["policy"] == POLICY_LABEL]
    full_rows = [row for row in rows if row["policy"] == "full_basis_shared"]
    maximum_selected = max(
        selected_rows, key=lambda row: row["absolute_relative_error"]
    )
    maximum_full = max(full_rows, key=lambda row: row["absolute_relative_error"])
    maximum_direct_se = max(
        rows, key=lambda row: row["direct_relative_standard_error"]
    )
    maximum_standardized = max(
        selected_rows, key=lambda row: row["absolute_standardized_residual"]
    )
    checks = {
        "input_artifacts_validate": True,
        "fixed_context_matches": all(context_checks.values()),
        "source_lineage_matches": all(lineage_checks.values()),
        "legacy_q16_q32_excluded_from_fit": True,
        "selected_policy_q16_q32_within_5_percent": all(
            row["within_5_percent"] for row in selected_rows
        ),
        "legacy_direct_rz_relative_se_within_2_percent": (
            maximum_direct_se["direct_relative_standard_error"]
            <= DIRECT_RZ_RELATIVE_SE_LIMIT
        ),
    }
    return {
        "configuration": {
            "molecule": latest["configuration"]["molecule"],
            "geometry_angstrom": latest["configuration"]["geometry_angstrom"],
            "basis": latest["configuration"]["basis"],
            "n_qubits": latest["configuration"]["n_qubits"],
            "df_rank": latest["configuration"]["df_rank"],
            "ld": 3,
            "delta": TARGET_DELTA,
            "r": TARGET_R,
            "k": TARGET_K,
            "compiler": latest["configuration"]["compiler"],
            "metric": "rz_count",
            "calibration_q": list(CALIBRATION_Q),
            "fresh_holdout_q_already_in_latest_artifact": list(FRESH_HOLDOUT_Q),
            "legacy_fixed_holdout_q": list(LEGACY_HOLDOUT_Q),
            "selected_policy": POLICY_LABEL,
            "relative_error_limit": RELATIVE_ERROR_LIMIT,
            "direct_rz_relative_se_limit": DIRECT_RZ_RELATIVE_SE_LIMIT,
        },
        "context_checks": context_checks,
        "lineage_checks": lineage_checks,
        "affine_models": models,
        "fixed_holdout_rows": rows,
        "summary": {
            "maximum_selected_policy_absolute_relative_error": dict(
                maximum_selected
            ),
            "maximum_full_basis_absolute_relative_error": dict(maximum_full),
            "maximum_direct_rz_relative_standard_error": dict(
                maximum_direct_se
            ),
            "maximum_selected_policy_absolute_standardized_residual": dict(
                maximum_standardized
            ),
            "selected_policy_q16_q32_pass_5_percent": all(
                row["within_5_percent"] for row in selected_rows
            ),
            "full_basis_q16_q32_pass_5_percent": all(
                row["within_5_percent"] for row in full_rows
            ),
        },
        "applicable_domain": {
            "selected_policy_compiled_rz": {
                "cell": TARGET_CELL,
                "fresh_calibration_q": list(CALIBRATION_Q),
                "fresh_direct_holdout_q": list(FRESH_HOLDOUT_Q),
                "legacy_fixed_holdout_q_reconciled": list(LEGACY_HOLDOUT_Q),
                "maximum_directly_reconciled_q": 32,
                "five_percent_criterion_passed": all(
                    row["within_5_percent"] for row in selected_rows
                ),
            },
            "full_basis_compiled_rz": {
                "legacy_fixed_holdout_q_reconciled": list(LEGACY_HOLDOUT_Q),
                "five_percent_criterion_passed": all(
                    row["within_5_percent"] for row in full_rows
                ),
                "maximum_q_passing_individually": max(
                    row["q_m"] for row in full_rows if row["within_5_percent"]
                ),
            },
            "all_metrics_latest_fresh_holdout_max_q": 8,
            "q_above_32_validated": False,
        },
        "decision": {
            "status": "selected_policy_rz_q16_q32_reconciled_within_5_percent",
            "selected_policy_domain_update": (
                "For the exact H4 L_D=3 delta=0.02 r=32 K=2 opt2 cell, "
                "the latest fresh q=1,2 RZ proxy is now fixed-holdout checked "
                "against legacy q=16,32 measurements."
            ),
            "full_basis_diagnostic": (
                "The q=32 full-basis RZ residual exceeds 5%; this non-gating "
                "diagnostic is not promoted to a passing q=32 domain."
            ),
            "next_action_changed": False,
        },
        "checks": checks,
        "overall_pass": all(checks.values()),
        "scope": {
            "new_compilation_performed": False,
            "legacy_holdouts_refit": False,
            "all_metric_q16_q32_reconciliation_performed": False,
            "schedule_or_cost_reoptimization_performed": False,
            "external_instance_included": False,
            "state_preparation_included": False,
            "backend_execution_included": False,
            "q_above_32_directly_validated": False,
            "final_total_cost_evaluation_performed": False,
            "scientific_superiority_claimed": False,
        },
    }


def finalize_proxy_lineage_reconciliation_artifact(
    body: Mapping[str, Any], *, provenance: Mapping[str, Any]
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "stage": STAGE,
        **dict(body),
        "provenance": dict(provenance),
    }
    payload["content_fingerprint"] = fingerprint(payload)
    validate_proxy_lineage_reconciliation_artifact(payload)
    return payload


def validate_proxy_lineage_reconciliation_artifact(
    payload: Mapping[str, Any],
) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported A0 proxy-lineage schema.")
    if payload.get("method") != METHOD or payload.get("stage") != STAGE:
        raise ValueError("Unsupported A0 proxy-lineage method or stage.")
    unsigned = dict(payload)
    observed = unsigned.pop("content_fingerprint", None)
    if observed != fingerprint(unsigned):
        raise ValueError("A0 proxy-lineage artifact fingerprint mismatch.")
    checks = payload.get("checks", {})
    if payload.get("overall_pass") != (bool(checks) and all(checks.values())):
        raise ValueError("A0 proxy-lineage status does not match checks.")
    scope = payload.get("scope", {})
    if scope.get("new_compilation_performed") is not False:
        raise ValueError("A0 cannot claim new compilation.")
    if scope.get("legacy_holdouts_refit") is not False:
        raise ValueError("A0 legacy q=16,32 points must remain fixed holdouts.")
    if scope.get("final_total_cost_evaluation_performed") is not False:
        raise ValueError("A0 cannot claim final total cost.")
    if scope.get("scientific_superiority_claimed") is not False:
        raise ValueError("A0 cannot claim scientific superiority.")


def write_proxy_lineage_reconciliation_artifact(
    payload: Mapping[str, Any], path: str | Path
) -> None:
    validate_proxy_lineage_reconciliation_artifact(payload)
    output = Path(path)
    if output.exists():
        raise ValueError(f"Refusing to replace existing artifact: {output}")
    atomic_write_json(output, payload)


def read_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload
