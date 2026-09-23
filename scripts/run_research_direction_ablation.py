#!/usr/bin/env python3
"""Run the WP04 schedule, beta, alpha, and provider ablation."""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import qiskit

from trotterlib.df_hamiltonian import PhysicalSector
from trotterlib.df_partial_randomized_pf import split_df_hamiltonian_by_ld
from trotterlib.df_partial_s2 import prepare_df_partial_s2
from trotterlib.finite_rte_signal_validation import (
    validate_finite_rte_signal_payload,
    validate_finite_rte_signals,
    write_finite_rte_signal_validation,
)
from trotterlib.research_direction_ablation import (
    AffineAxisCostModel,
    PairCompiledCostModel,
    build_wp04_ablation_body,
    deterministic_physical_signal_diagnostic,
    finalize_wp04_artifact,
    selected_matrix_signal_diagnostic,
    statistical_bound_diagnostic,
    write_wp04_artifact,
)
from trotterlib.research_direction_prevalidation import (
    file_sha256,
    validate_artifact,
)
from trotterlib.rpe_hadamard_compiled_cost_benchmark import (
    RPEHadamardCompiledCostBenchmarkDataset,
)
from trotterlib.rpe_hadamard_compiled_cost_proxy import (
    RPEHadamardCompiledCostProxy,
)
from trotterlib.rte_connected_cluster_cost_validation import (
    load_connected_cluster_hamiltonian_snapshot,
)


DEFAULT_SNAPSHOT = Path(
    "artifacts/rte_connected_cluster_cost_validation/"
    "h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k2_connected_"
    "pilot30_max1500_hold1500_rare375_v1.hamiltonian.npz"
)
DEFAULT_WP01 = Path(
    "artifacts/research_direction_prevalidation/2026-09-21/"
    "wp01s_model_conditional_screening_v1.json"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/research_direction_ablation/2026-09-21")


def _git(command: list[str]) -> str | list[str] | None:
    result = subprocess.run(
        ["git", *command], check=False, capture_output=True, text=True
    )
    if result.returncode != 0:
        return None
    lines = result.stdout.splitlines()
    return lines[0] if len(lines) == 1 else lines


def _ref(path: Path, **extra: object) -> dict[str, object]:
    return {"path": str(path), "sha256": file_sha256(path), **extra}


def _provenance() -> dict[str, object]:
    sources = (
        Path("src/trotterlib/research_direction_ablation.py"),
        Path("scripts/run_research_direction_ablation.py"),
    )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_worktree_status_before_generation": _git(["status", "--short"]),
        "evidence_status": "local_worktree_validation_not_immutable_ci",
        "command": shlex.join(
            [
                ".venv311/bin/python",
                "scripts/run_research_direction_ablation.py",
                *sys.argv[1:],
            ]
        ),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "qiskit_version": qiskit.__version__,
        "source_sha256": {str(path): file_sha256(path) for path in sources},
    }


def _axis_model(
    dataset: RPEHadamardCompiledCostBenchmarkDataset,
    axis: str,
) -> AffineAxisCostModel:
    statistics = {}
    for record in dataset.records:
        if record.partition != "calibration" or record.axis != axis:
            continue
        metric = dict(record.metric_statistics)["rz_count"]
        statistics[int(record.q_m)] = (
            float(metric.mean),
            0.0 if metric.standard_error is None else float(metric.standard_error),
        )
    if set(statistics) != {1, 2}:
        raise ValueError("WP04 requires exact q=1,2 calibration records.")
    return AffineAxisCostModel(
        q1_mean=statistics[1][0],
        q2_mean=statistics[2][0],
        q1_standard_error=statistics[1][1],
        q2_standard_error=statistics[2][1],
    )


def _load_cost_models(
    wp01: dict,
) -> tuple[dict[int, tuple[PairCompiledCostModel, ...]], dict[str, object]]:
    models: dict[int, tuple[PairCompiledCostModel, ...]] = {}
    references: dict[str, object] = {}
    for ld in (3, 12):
        candidate_models = []
        references[str(ld)] = {}
        for pair_label, refs in sorted(wp01["calibrations"][str(ld)].items()):
            dataset_path = Path(refs["dataset"]["path"])
            proxy_path = Path(refs["proxy"]["path"])
            dataset = RPEHadamardCompiledCostBenchmarkDataset.read_json(
                dataset_path
            )
            proxy = RPEHadamardCompiledCostProxy.read_json(proxy_path)
            r_label, k_label = pair_label.split("_")
            model = PairCompiledCostModel(
                rte_steps=int(r_label[1:]),
                finite_taylor_order=int(k_label[1:]),
                cosine=_axis_model(dataset, "cosine"),
                sine=_axis_model(dataset, "sine"),
                dataset_fingerprint=dataset.dataset_fingerprint,
                proxy_fingerprint=proxy.proxy_fingerprint,
            )
            for axis in ("cosine", "sine"):
                axis_model = model.cosine if axis == "cosine" else model.sine
                for q_m in (1, 2):
                    prediction, _standard_error = axis_model.predict(q_m)
                    proxy_prediction = proxy.model(axis, "rz_count").predict(q_m)
                    if not np.isclose(prediction, proxy_prediction, atol=1e-12):
                        raise ValueError("Raw q=1,2 affine model differs from proxy.")
            candidate_models.append(model)
            references[str(ld)][pair_label] = {
                "dataset": _ref(
                    dataset_path,
                    dataset_fingerprint=dataset.dataset_fingerprint,
                ),
                "proxy": _ref(
                    proxy_path,
                    proxy_fingerprint=proxy.proxy_fingerprint,
                ),
            }
        models[ld] = tuple(candidate_models)
    return models, references


def _scenario(body: dict, scenario_id: str) -> dict:
    matches = [
        item for item in body["scenarios"] if item["scenario_id"] == scenario_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Scenario {scenario_id!r} is missing or duplicated.")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--wp01-artifact", type=Path, default=DEFAULT_WP01)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    wp01 = json.loads(args.wp01_artifact.read_text(encoding="utf-8"))
    validate_artifact(wp01)
    hamiltonian = load_connected_cluster_hamiltonian_snapshot(args.snapshot)
    sector = PhysicalSector.number_sector(
        n_qubits=hamiltonian.n_qubits,
        n_electrons=4,
    )
    preparations = {
        ld: prepare_df_partial_s2(
            hamiltonian,
            split_df_hamiltonian_by_ld(hamiltonian, ld),
            identity_policy="extract_identity_phase",
        )
        for ld in (3, 12)
    }
    coefficients = {
        ld: float(
            next(
                item["pf_coefficient"]
                for item in wp01["candidates"]
                if item["ld"] == ld
            )
        )
        for ld in (3, 12)
    }
    cost_models, calibration_refs = _load_cost_models(wp01)
    body = build_wp04_ablation_body(
        preparations,
        pf_coefficients=coefficients,
        cost_models=cost_models,
    )

    full_ids = body["full_setting"]["scenario_ids"]
    full_ld3 = _scenario(body, full_ids["3"])
    full_ld12 = _scenario(body, full_ids["12"])
    q_values = tuple(int(row["q_m"]) for row in full_ld3["rounds"])
    selected_r = tuple(sorted({int(row["r_m"]) for row in full_ld3["rounds"]}))
    selected_k = tuple(sorted({int(row["K_m"]) for row in full_ld3["rounds"]}))

    matrix_payload = validate_finite_rte_signals(
        hamiltonian,
        sector,
        ld=3,
        delta_time=0.02,
        q_values=q_values,
        rte_step_values=selected_r,
        finite_taylor_orders=selected_k,
        beta_rpe=0.4,
        beta_pf_budget=0.015,
        beta_rte_budget=0.005,
        beta_stat_budget=0.38,
        alpha_total=0.05,
        seed=20260818,
        provenance={
            **_provenance(),
            "role": "WP04_LD3_full_setting_selected_schedule_signal_grid",
        },
    )
    matrix_output = (
        args.output_dir / "wp04_ld3_full_schedule_signal_grid_v1.json"
    )
    write_finite_rte_signal_validation(matrix_payload, matrix_output)
    validate_finite_rte_signal_payload(matrix_payload)
    ld3_signal = selected_matrix_signal_diagnostic(matrix_payload, full_ld3)
    ld12_signal = deterministic_physical_signal_diagnostic(
        hamiltonian,
        sector,
        ld=12,
        delta_time=0.02,
        q_values=q_values,
    )
    statistical = {
        "3": statistical_bound_diagnostic(full_ld3, ld3_signal),
        "12": statistical_bound_diagnostic(full_ld12, ld12_signal),
    }
    body["signal_diagnostics"] = {
        "3": ld3_signal,
        "12": ld12_signal,
    }
    body["statistical_bound_diagnostics"] = statistical
    body["source_evidence"] = {
        "snapshot": _ref(args.snapshot),
        "wp01_screening": _ref(
            args.wp01_artifact,
            content_fingerprint=wp01["content_fingerprint"],
        ),
        "compiled_cost_calibrations": calibration_refs,
        "ld3_selected_schedule_signal_grid": _ref(
            matrix_output,
            validation_fingerprint=matrix_payload["validation_fingerprint"],
        ),
    }
    body["checks"].update(
        {
            "ld3_matrix_signal_grid_passed": bool(
                matrix_payload["summary"]["overall_pass"]
            ),
            "ld3_selected_radius_bounds_pass": bool(
                ld3_signal["all_radius_bounds_pass"]
            ),
            "all_exact_binomial_failures_within_allocated_alpha": all(
                item["all_exact_failures_within_allocated_alpha"]
                for item in statistical.values()
            ),
            "all_exact_binomial_union_bounds_within_alpha_total": all(
                item["exact_union_bound_at_hoeffding_shots"]
                <= item["selected_alpha_total"] + 1e-15
                for item in statistical.values()
            ),
            "deterministic_unit_radius_model_deviation_below_1e_8": bool(
                ld12_signal["maximum_unit_radius_model_absolute_error"] < 1e-8
            ),
        }
    )
    body["overall_pass"] = all(body["checks"].values())

    ld3_full_cost = float(full_ld3["total_compiled_rz_point_estimate"])
    ld12_full_cost = float(full_ld12["total_compiled_rz_point_estimate"])
    body["summary"] = {
        "status": "WP04_model_conditional_ablation_complete",
        "ld3_full_setting_rz_point_estimate": ld3_full_cost,
        "ld12_full_setting_rz_point_estimate": ld12_full_cost,
        "ld12_reduction_relative_to_ld3_full_setting": (
            (ld3_full_cost - ld12_full_cost) / ld3_full_cost
        ),
        "directional_result": body["full_setting"]["directional_result"],
        "dominant_common_factor": "beta_reallocation_then_alpha_reallocation",
        "partial_specific_round_schedule_contribution": (
            "small_after_compiled_cost_aligned_selection"
        ),
        "component_schedule_objective_reverses_rz_direction": body[
            "cost_provider_diagnostic"
        ]["schedule_objective_changes_direction"],
        "next_required_validation": "WP03_PF_coefficient_selection_sensitivity",
    }
    body["limitations"] = [
        (
            "The long-q compiled costs are q=1,2 affine extrapolations "
            "without an unused holdout for these schedule families."
        ),
        (
            "The beta profiles are a predeclared diagnostic grid, not a "
            "continuous global optimization."
        ),
        (
            "The exact-binomial counterfactual uses small-sector matrix "
            "signals and does not replace full RPE branch reconstruction."
        ),
        (
            "PF coefficients remain empirical and are held fixed here; "
            "WP03 changes them next."
        ),
        (
            "No state preparation, backend execution, noise model, or "
            "final total-cost evaluation is included."
        ),
    ]

    payload = finalize_wp04_artifact(body, provenance=_provenance())
    output = args.output_dir / "wp04_finite_rte_statistical_ablation_v1.json"
    write_wp04_artifact(payload, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "content_fingerprint": payload["content_fingerprint"],
                "overall_pass": payload["overall_pass"],
                "ld3_full_rz": ld3_full_cost,
                "ld12_full_rz": ld12_full_cost,
                "ld12_reduction": payload["summary"][
                    "ld12_reduction_relative_to_ld3_full_setting"
                ],
                "directional_result": payload["summary"]["directional_result"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
