#!/usr/bin/env python3
"""Run WP03 PF-coefficient selection and regret sensitivity."""

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
from trotterlib.df_partial_randomized_pf import (
    fit_df_cgs_with_perturbation,
    split_df_hamiltonian_by_ld,
)
from trotterlib.df_partial_s2 import prepare_df_partial_s2
from trotterlib.pf_delta_validation import validate_pf_delta_payload
from trotterlib.research_direction_ablation import (
    AffineAxisCostModel,
    PairCompiledCostModel,
    validate_wp04_artifact,
)
from trotterlib.research_direction_pf_sensitivity import (
    evaluate_wp03_sensitivity,
    extract_pf_coefficient_audit,
    finalize_wp03_artifact,
    write_wp03_artifact,
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
DEFAULT_WP04 = Path(
    "artifacts/research_direction_ablation/2026-09-21/"
    "wp04_finite_rte_statistical_ablation_v1.json"
)
DEFAULT_PF_DIR = Path(
    "artifacts/research_direction_prevalidation/2026-09-21/"
    "pf_delta_same_snapshot"
)
DEFAULT_OUTPUT = Path(
    "artifacts/research_direction_pf_sensitivity/2026-09-22/"
    "wp03_pf_coefficient_selection_sensitivity_v1.json"
)
COMMON_COEFFICIENT_WINDOW = (0.05, 0.1, 0.2, 0.4)


def _git(command: list[str]) -> str | list[str] | None:
    result = subprocess.run(
        ["git", *command], check=False, capture_output=True, text=True
    )
    if result.returncode != 0:
        return None
    lines = result.stdout.splitlines()
    return lines[0] if len(lines) == 1 else lines


def _provenance() -> dict[str, object]:
    sources = (
        Path("src/trotterlib/research_direction_pf_sensitivity.py"),
        Path("scripts/run_research_direction_pf_sensitivity.py"),
    )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_worktree_status_before_generation": _git(["status", "--short"]),
        "evidence_status": "local_worktree_validation_not_immutable_ci",
        "command": shlex.join(
            [
                ".venv311/bin/python",
                "scripts/run_research_direction_pf_sensitivity.py",
                *sys.argv[1:],
            ]
        ),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "qiskit_version": qiskit.__version__,
        "source_sha256": {str(path): file_sha256(path) for path in sources},
    }


def _ref(path: Path, **extra: object) -> dict[str, object]:
    return {"path": str(path), "sha256": file_sha256(path), **extra}


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
        raise ValueError("WP03 requires exact q=1,2 calibration records.")
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
                    expected = proxy.model(axis, "rz_count").predict(q_m)
                    if not np.isclose(prediction, expected, atol=1e-12):
                        raise ValueError("Raw q=1,2 model differs from proxy.")
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


def _common_window_hd_fits(hamiltonian) -> dict[int, dict[str, object]]:
    sector = PhysicalSector.number_sector(
        n_qubits=hamiltonian.n_qubits,
        n_electrons=4,
    )
    fits = {}
    for ld in (0, 3, 12):
        result = fit_df_cgs_with_perturbation(
            hamiltonian,
            sector,
            split_df_hamiltonian_by_ld(hamiltonian, ld),
            "2nd",
            t_values=COMMON_COEFFICIENT_WINDOW,
            evolution_backend="cpu",
            matrix_free_backend="auto",
            parallel_times=False,
            use_ground_state_cache=False,
            require_usable_estimate=False,
        )
        fits[ld] = {
            "delta_values": list(COMMON_COEFFICIENT_WINDOW),
            "fixed_second_order_coefficient": float(
                result.fit_coeff_fixed_order
            ),
            "free_fit_slope": (
                None if result.fit_slope is None else float(result.fit_slope)
            ),
            "free_fit_coefficient": (
                None if result.fit_coeff is None else float(result.fit_coeff)
            ),
            "signed_energy_biases": [
                float(value) for value in result.signed_phase_biases
            ],
            "absolute_energy_biases": [
                float(value) for value in result.perturbation_errors
            ],
            "relative_overlap_magnitudes": [
                float(value) for value in result.relative_overlap_magnitudes
            ],
            "estimator_status": result.estimator_status,
            "screening_usable": bool(
                result.metadata.get("screening_usable", False)
            ),
            "fit_window_relative_spread": float(
                result.metadata.get("fit_window_relative_spread", 0.0)
            ),
            "ground_state_residual_norm": (
                None
                if result.metadata.get("ground_state_residual_norm") is None
                else float(result.metadata["ground_state_residual_norm"])
            ),
            "is_rigorous_bound": False,
        }
    return fits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--wp01-artifact", type=Path, default=DEFAULT_WP01)
    parser.add_argument("--wp04-artifact", type=Path, default=DEFAULT_WP04)
    parser.add_argument("--pf-directory", type=Path, default=DEFAULT_PF_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    wp01 = json.loads(args.wp01_artifact.read_text(encoding="utf-8"))
    validate_artifact(wp01)
    wp04 = json.loads(args.wp04_artifact.read_text(encoding="utf-8"))
    validate_wp04_artifact(wp04)
    hamiltonian = load_connected_cluster_hamiltonian_snapshot(args.snapshot)
    preparations = {
        ld: prepare_df_partial_s2(
            hamiltonian,
            split_df_hamiltonian_by_ld(hamiltonian, ld),
            identity_policy="extract_identity_phase",
        )
        for ld in (3, 12)
    }
    pf_paths = {
        ld: args.pf_directory / f"h4_sto3g_d100_rank12_ld{ld}_v5.json"
        for ld in (0, 3, 12)
    }
    pf_payloads = {
        ld: json.loads(path.read_text(encoding="utf-8"))
        for ld, path in pf_paths.items()
    }
    for payload in pf_payloads.values():
        validate_pf_delta_payload(payload)
    common_window_fits = _common_window_hd_fits(hamiltonian)
    coefficient_audit = extract_pf_coefficient_audit(
        pf_payloads,
        common_window_hd_fits=common_window_fits,
    )
    cost_models, calibration_refs = _load_cost_models(wp01)
    body = evaluate_wp03_sensitivity(
        preparations,
        cost_models=cost_models,
        coefficient_audit=coefficient_audit,
    )
    body["checks"]["coefficient_comparison_uses_common_delta_window"] = bool(
        coefficient_audit["common_delta_window_comparison"]
        and coefficient_audit["common_delta_window"]
        == list(COMMON_COEFFICIENT_WINDOW)
    )
    body["overall_pass"] = all(body["checks"].values())

    if wp04["configuration"]["delta_time"] != 0.02:
        raise ValueError("WP03 expected the WP04 delta=0.02 full setting.")
    if wp04["configuration"]["alpha_total"] != body["configuration"][
        "alpha_total"
    ]:
        raise ValueError("WP03 alpha_total differs from WP04.")

    body["source_evidence"] = {
        "snapshot": _ref(args.snapshot),
        "wp01_screening": _ref(
            args.wp01_artifact,
            content_fingerprint=wp01["content_fingerprint"],
        ),
        "wp04_ablation": _ref(
            args.wp04_artifact,
            content_fingerprint=wp04["content_fingerprint"],
        ),
        "pf_coefficient_artifacts": {
            str(ld): _ref(
                path,
                validation_fingerprint=pf_payloads[ld][
                    "validation_fingerprint"
                ],
            )
            for ld, path in pf_paths.items()
        },
        "compiled_cost_calibrations": calibration_refs,
    }

    d6_comparison = body["ld3_vs_ld12"]["paper_d6"]
    d6_regret = {
        row["candidate_id"]: row["relative_regret_vs_policy_best"]
        for row in body["candidate_regret"]["paper_d6"]
    }
    audit_rows = body["coefficient_audit"]["rows"]
    costed_d6_eigen_gap = max(
        abs(
            float(
                audit_rows[str(ld)][
                    "paper_d6_relative_difference_vs_dominant_eigenphase"
                ]
            )
        )
        for ld in (3, 12)
    )
    selected_headrooms = {
        str(ld): next(
            row["relative_coefficient_increase_to_pf_budget_boundary"]
            for row in body["scenarios"]
            if row["coefficient_policy"] == "paper_d6"
            and int(row["ld"]) == ld
            and float(row["delta_time"]) == 0.02
        )
        for ld in (3, 12)
    }
    body["summary"] = {
        "status": "WP03_pf_coefficient_selection_sensitivity_complete",
        "selection_changed_by_coefficient_policy": not body[
            "selection_sensitivity"
        ]["ld_and_delta_selection_invariant"],
        "selected_candidate_for_all_policies": body["selections"][
            "paper_d6"
        ]["candidate_id"],
        "paper_d6_ld3_point_regret_vs_ld12": d6_regret["ld3_delta0.02"],
        "paper_d6_local_5_percent_relative_gap_interval": d6_comparison[
            "local_5_percent_relative_gap_interval"
        ],
        "paper_d6_transfer_25_percent_relative_gap_interval": d6_comparison[
            "transfer_25_percent_relative_gap_interval"
        ],
        "costed_max_paper_d6_vs_dominant_coefficient_relative_gap": (
            costed_d6_eigen_gap
        ),
        "paper_d6_relative_headroom_to_pf_boundary_at_delta_0p02": (
            selected_headrooms
        ),
        "ld0_hd_surrogate_is_zero_while_paper_d6_is_nonzero": bool(
            audit_rows["0"]["coefficients"]["hd_surrogate"] == 0.0
            and audit_rows["0"]["coefficients"]["paper_d6"] > 0.0
        ),
        "directional_result": body["selection_sensitivity"][
            "directional_result"
        ],
        "gate_s1_status": "all_required_wp_results_available_for_synthesis",
        "next_required_action": "Gate_S1_research_direction_synthesis",
    }
    body["limitations"] = [
        (
            "C_D is an H_D-only screening surrogate, not a bound; at L_D=0 "
            "it is zero while the full partial-S2 coefficients are nonzero."
        ),
        (
            "Paper-D6 and dominant-eigenphase signed values use opposite "
            "recorded sign conventions, so magnitude agreement cannot be "
            "interpreted as signed commutator cancellation."
        ),
        (
            "The source artifacts do not decompose individual mixed or tail "
            "commutators; C_D-to-full gaps aggregate those omitted effects."
        ),
        (
            "The coefficient set is a discrete sensitivity family rather "
            "than a statistical confidence distribution."
        ),
        (
            "Long-q costs remain q=1,2 affine extrapolations without an "
            "unused holdout for these schedule families."
        ),
        (
            "No state preparation, backend execution, noise model, full-"
            "scope holdout, or final total-cost evaluation is included."
        ),
    ]

    payload = finalize_wp03_artifact(body, provenance=_provenance())
    write_wp03_artifact(payload, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "content_fingerprint": payload["content_fingerprint"],
                "overall_pass": payload["overall_pass"],
                "selected_candidate": payload["summary"][
                    "selected_candidate_for_all_policies"
                ],
                "selection_changed": payload["summary"][
                    "selection_changed_by_coefficient_policy"
                ],
                "ld3_point_regret": payload["summary"][
                    "paper_d6_ld3_point_regret_vs_ld12"
                ],
                "directional_result": payload["summary"][
                    "directional_result"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
