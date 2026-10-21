"""M08 targeted q>8 proxy-precision holdout for dominant late rounds."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .df_hamiltonian import DFHamiltonian
from .df_partial_s2 import DFPartialS2Preparation
from .df_partial_s2_repeated_cost import (
    make_monte_carlo_df_partial_s2_repeated_trajectory_stream,
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
)
from .research_direction_full_scope_extension import fit_affine_holdouts
from .research_direction_full_scope_replication import validate_wp05br_artifact
from .research_direction_round_dominance import validate_g08_artifact
from .research_direction_sequence_policy import register_support_restricted_bases
from .rte import CompilerSettings


SCHEMA_VERSION = "research_direction_proxy_precision_v1"
METHOD = "m08_late_round_q16_q32_proxy_precision_v1"


def _compile_holdouts(
    preparation: DFPartialS2Preparation,
    support_definitions: Mapping[Any, Any],
    compiler: CompilerSettings,
    *,
    delta_time: float,
    rte_steps: int,
    q_values: Sequence[int],
    sample_count: int,
    master_seed: int,
    training_fingerprint: str,
    progress: Callable[[str], None] | None,
) -> dict[int, Any]:
    config, distribution = _finite_inputs(
        preparation,
        delta_time=delta_time,
        rte_steps=rte_steps,
        finite_taylor_order=2,
    )
    output: dict[int, Any] = {}
    for q_m in q_values:
        if progress is not None:
            progress(
                f"M08 delta={delta_time:g} r={rte_steps} q={int(q_m)} "
                f"samples={sample_count}"
            )
        seed = _derived_seed(
            master_seed, "m08", delta_time, rte_steps, int(q_m)
        )
        stream = make_monte_carlo_df_partial_s2_repeated_trajectory_stream(
            preparation,
            delta_time,
            int(q_m),
            config,
            distribution,
            sample_count=sample_count,
            seed=seed,
            maximum_samples=sample_count,
            controlled=True,
            ancilla_qubit=preparation.num_system_qubits,
            construction_policy="boundary_optimized",
        )
        compiled, _probe = _compile_paired_requests(
            _materialize_stream(stream),
            support_definitions,
            compiler,
            training_fingerprint=training_fingerprint,
            maximum_repetition_count=max(int(q) for q in q_values),
            operator_probe_requested=False,
        )
        compiled.update(
            {
                "q_m": int(q_m),
                "partition": "m08_fresh_holdout",
                "master_seed": seed,
            }
        )
        output[int(q_m)] = compiled
    return output


def _maximum_error_record(
    models: Mapping[str, Any],
    points: Mapping[int, Mapping[str, Any]],
    *,
    policy: str,
    metrics: Sequence[str],
) -> dict[str, Any]:
    records = []
    for axis in AXES:
        for metric in metrics:
            for q_label, holdout in models[axis][metric][
                "holdouts"
            ].items():
                q_m = int(q_label)
                direct = points[q_m]["axes"][axis]["policies"][policy][metric]
                direct_se = float(direct["standard_error"])
                prediction_se = float(holdout["prediction_standard_error"])
                difference = abs(
                    float(holdout["prediction"])
                    - float(holdout["direct_mean"])
                )
                combined_se = math.hypot(direct_se, prediction_se)
                records.append(
                    {
                        "axis": axis,
                        "metric": metric,
                        "q_m": q_m,
                        **dict(holdout),
                        "direct_standard_error": direct_se,
                        "direct_relative_standard_error": (
                            0.0
                            if float(holdout["direct_mean"]) == 0.0
                            else direct_se / float(holdout["direct_mean"])
                        ),
                        "combined_standard_error": combined_se,
                        "absolute_difference_over_combined_standard_error": (
                            math.inf
                            if combined_se == 0.0
                            else difference / combined_se
                        ),
                    }
                )
    return max(records, key=lambda row: row["absolute_relative_error"])


def evaluate_m08_proxy_precision(
    hamiltonian: DFHamiltonian,
    preparation: DFPartialS2Preparation,
    compiler: CompilerSettings,
    wp05br: Mapping[str, Any],
    wp06b: Mapping[str, Any],
    g08: Mapping[str, Any],
    *,
    sample_count: int = 8,
    master_seed: int = 2026092211,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Directly score q=16/32 against the unchanged q=1,2 affine proxy."""
    validate_wp05br_artifact(wp05br)
    validate_g08_artifact(g08)
    if not wp05br["overall_pass"] or not g08["overall_pass"]:
        raise ValueError("M08 requires passing WP05-bR and G08 inputs.")
    target = g08["m08_target"]
    delta_time = float(target["delta_time"])
    rte_steps = int(target["rte_steps"][0])
    q_values = tuple(int(q) for q in target["direct_holdout_q_values"])
    if q_values != (16, 32) or rte_steps != 32 or delta_time != 0.02:
        raise ValueError("Unsupported M08 target selected by G08.")
    if preparation.ld != 3:
        raise ValueError("M08 requires the fixed L_D=3 preparation.")

    support_definitions, proof_records = register_support_restricted_bases(
        hamiltonian, preparation
    )
    training_fingerprint = str(
        wp06b["training_selection"]["training_fingerprint"]
    )
    holdout_points = _compile_holdouts(
        preparation,
        support_definitions,
        compiler,
        delta_time=delta_time,
        rte_steps=rte_steps,
        q_values=q_values,
        sample_count=sample_count,
        master_seed=master_seed,
        training_fingerprint=training_fingerprint,
        progress=progress,
    )
    calibration_points = wp05br["direct_randomized_ld3"]["points"]
    all_points = {
        1: calibration_points["1"],
        2: calibration_points["2"],
        **holdout_points,
    }
    models = {
        policy: fit_affine_holdouts(
            all_points, policy=policy, holdout_q=q_values
        )
        for policy in ("full_basis_shared", POLICY_LABEL)
    }
    selected_rz = _maximum_error_record(
        models[POLICY_LABEL],
        holdout_points,
        policy=POLICY_LABEL,
        metrics=("rz_count",),
    )
    selected_all = _maximum_error_record(
        models[POLICY_LABEL],
        holdout_points,
        policy=POLICY_LABEL,
        metrics=METRICS,
    )
    full_rz = _maximum_error_record(
        models["full_basis_shared"],
        holdout_points,
        policy="full_basis_shared",
        metrics=("rz_count",),
    )
    maximum_direct_rz_relative_se = max(
        float(
            holdout_points[q]["axes"][axis]["policies"][policy][
                "rz_count"
            ]["standard_error"]
        )
        / float(
            holdout_points[q]["axes"][axis]["policies"][policy][
                "rz_count"
            ]["mean"]
        )
        for q in q_values
        for axis in AXES
        for policy in ("full_basis_shared", POLICY_LABEL)
    )
    accuracy_threshold = float(target["accuracy_threshold"])
    separation_limit = float(target["separation_limit"])
    proof_residual = max(
        float(row["preserved_columns_max_abs_residual"]) for row in proof_records
    )
    checks = {
        "g08_target_executed_without_scope_expansion": True,
        "fresh_seed_differs_from_wp05br": (
            master_seed != int(wp05br["configuration"]["master_seed"])
        ),
        "q16_q32_have_requested_fresh_trajectories": all(
            int(holdout_points[q]["sample_count"]) == sample_count
            for q in q_values
        ),
        "support_basis_certificates_pass": proof_residual <= 1e-10,
        "selected_policy_q16_q32_rz_within_5_percent": (
            float(selected_rz["absolute_relative_error"])
            <= accuracy_threshold
        ),
        "selected_policy_q16_q32_all_metrics_within_5_percent": (
            float(selected_all["absolute_relative_error"])
            <= accuracy_threshold
        ),
        "full_basis_q16_q32_rz_within_5_percent": (
            float(full_rz["absolute_relative_error"]) <= accuracy_threshold
        ),
        "selected_policy_rz_error_below_local_separation_limit": (
            float(selected_rz["absolute_relative_error"]) <= separation_limit
        ),
        "direct_rz_relative_standard_error_within_2_percent": (
            maximum_direct_rz_relative_se <= 0.02
        ),
        "production_default_remains_full_basis": True,
        "final_total_cost_and_robust_superiority_not_claimed": True,
    }
    overall_pass = all(checks.values())
    return {
        "configuration": {
            "molecule": "H4_chain",
            "geometry_angstrom": 1.0,
            "basis": "STO-3G",
            "n_qubits": hamiltonian.n_qubits,
            "df_rank": len(hamiltonian.lambdas),
            "ld": 3,
            "delta_time": delta_time,
            "finite_taylor_order": 2,
            "rte_steps": rte_steps,
            "calibration_q_values": [1, 2],
            "fresh_holdout_q_values": list(q_values),
            "fresh_sample_count_per_q": sample_count,
            "master_seed": master_seed,
            "accuracy_threshold": accuracy_threshold,
            "local_interval_separation_limit": separation_limit,
            "compiler": dict(wp05br["configuration"]["compiler"]),
        },
        "policy_input": {
            "policy": POLICY_LABEL,
            "wp06b_training_fingerprint": training_fingerprint,
            "production_default_changed": False,
            "maximum_preserved_columns_residual": proof_residual,
        },
        "direct_holdout_points": {
            str(q): point for q, point in holdout_points.items()
        },
        "affine_models": models,
        "diagnostics": {
            "maximum_selected_policy_rz_error": selected_rz,
            "maximum_selected_policy_all_metric_error": selected_all,
            "maximum_full_basis_rz_error": full_rz,
            "maximum_direct_rz_relative_standard_error": (
                maximum_direct_rz_relative_se
            ),
        },
        "decision": {
            "m08_local_q32_proxy_adequacy": (
                "passed" if overall_pass else "requires_refinement"
            ),
            "validated_direct_q_maximum": 32,
            "schedule_q_maximum": int(g08["candidates"]["3"]["q_max"]),
            "robust_directional_result_remains": (
                "undetermined_under_transfer_sensitivity"
            ),
            "next_action": (
                "reaggregate_wp01d_with_measured_q16_q32_discrepancy"
                if overall_pass
                else "refine_proxy_or_add_targeted_q64_holdout"
            ),
        },
        "scope": {
            "fresh_q16_q32_complete_wrappers_transpiled": True,
            "state_preparation_included": False,
            "backend_execution_included": False,
            "noise_included": False,
            "q_above_32_directly_validated": False,
            "final_total_cost_evaluation_performed": False,
            "scientific_superiority_claimed": False,
        },
        "limitations": [
            "Direct validation stops at q=32 while the selected schedule reaches q=131072.",
            "The result tests affine curvature over an expanded local q domain but cannot prove linearity at all late-round q values.",
            "The comparison remains H4-only, no-state-preparation, and one compiler context.",
        ],
        "checks": checks,
        "overall_pass": overall_pass,
        "summary": {
            "maximum_selected_policy_q16_q32_rz_relative_error": float(
                selected_rz["absolute_relative_error"]
            ),
            "maximum_selected_policy_q16_q32_all_metric_relative_error": float(
                selected_all["absolute_relative_error"]
            ),
            "maximum_full_basis_q16_q32_rz_relative_error": float(
                full_rz["absolute_relative_error"]
            ),
            "maximum_direct_rz_relative_standard_error": (
                maximum_direct_rz_relative_se
            ),
            "status": (
                "M08_q16_q32_pilot_passed"
                if overall_pass
                else "M08_q16_q32_pilot_requires_refinement"
            ),
        },
    }


def finalize_m08_artifact(
    body: Mapping[str, Any], *, provenance: Mapping[str, Any]
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "stage": "M08",
        **dict(body),
        "provenance": dict(provenance),
    }
    payload["content_fingerprint"] = fingerprint(payload)
    validate_m08_artifact(payload)
    return payload


def validate_m08_artifact(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported M08 schema.")
    if payload.get("method") != METHOD or payload.get("stage") != "M08":
        raise ValueError("Unsupported M08 method or stage.")
    unsigned = dict(payload)
    observed = unsigned.pop("content_fingerprint", None)
    if observed != fingerprint(unsigned):
        raise ValueError("M08 content_fingerprint mismatch.")
    checks = payload.get("checks", {})
    if payload.get("overall_pass") != (bool(checks) and all(checks.values())):
        raise ValueError("M08 overall status does not match checks.")
    scope = payload.get("scope", {})
    if scope.get("q_above_32_directly_validated") is not False:
        raise ValueError("M08 cannot claim direct q>32 validation.")
    if scope.get("final_total_cost_evaluation_performed") is not False:
        raise ValueError("M08 cannot claim final total cost.")
    if scope.get("scientific_superiority_claimed") is not False:
        raise ValueError("M08 cannot claim scientific superiority.")


def write_m08_artifact(payload: Mapping[str, Any], path: str | Path) -> None:
    validate_m08_artifact(payload)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
