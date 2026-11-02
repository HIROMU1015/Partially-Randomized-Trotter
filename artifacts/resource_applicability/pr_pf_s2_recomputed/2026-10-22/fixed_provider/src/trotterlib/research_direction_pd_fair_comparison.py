"""Preregistered P-D S1 fair product-formula comparison.

This is a dense H4 screening diagnostic, not a compiled-circuit or full-RPE
cost calculation.  Every formula is compared at a common physical time and
phase-error budget.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import research_direction_energy_tail_pareto as pd
from . import research_direction_pd_realization as realization
from .rte import finite_rte_distribution


EXPECTED_SCHEMA = "research_direction_pd_fair_comparison_expected_v1"
RESULT_SCHEMA = "research_direction_pd_fair_comparison_v1"
METHOD = "pd_s1_fixed_time_fair_pf_reoptimization_v1"

LD = 3
N_ELECTRONS = 4
TOTAL_TIME = 0.8
DELTAS = (0.1, 0.2, 0.4)
INNER_SUBSTEPS = (8, 16, 32, 64)
RTE_TOTAL_STEPS = (16, 32, 64, 128)
INNER_BOUNDARY_EXTENSION = 128
RTE_BOUNDARY_EXTENSION = 256
PRIMARY_K = 2
SENSITIVITY_K = 4
PHASE_BUDGET = 8.0e-7
DECISION_REGRET = 0.05
CONSTRUCTIONS = ("nested", "native")

CONFIGURATION = {
    "molecule": "H4 linear chain",
    "geometry_angstrom": 1.0,
    "basis": "STO-3G",
    "n_qubits": 8,
    "n_electrons": N_ELECTRONS,
    "df_rank": 12,
    "ld": LD,
    "total_physical_time_au": TOTAL_TIME,
    "deltas": list(DELTAS),
    "outer_step_counts": [int(round(TOTAL_TIME / value)) for value in DELTAS],
    "inner_hd_substeps": list(INNER_SUBSTEPS),
    "rte_short_steps_per_outer_step": list(RTE_TOTAL_STEPS),
    "primary_finite_taylor_order": PRIMARY_K,
    "restricted_sensitivity_finite_taylor_order": SENSITIVITY_K,
    "total_phase_error_budget_rad": PHASE_BUDGET,
    "decision_regret_threshold": DECISION_REGRET,
    "formula_labels": list(pd.FORMULA_LABELS),
    "constructions": list(CONSTRUCTIONS),
    "allocation_policy": "absolute_time_proportional_largest_remainder",
    "inner_hd_formula": "second_order_symmetric_fixed_df_prefix_order",
    "boundary_extension": {
        "inner_hd_substeps": INNER_BOUNDARY_EXTENSION,
        "rte_short_steps_per_outer_step": RTE_BOUNDARY_EXTENSION,
        "maximum_extension_rounds": 1,
    },
}


def _slug(label: str) -> str:
    return (
        label.lower()
        .replace("(", "_")
        .replace(")", "")
        .replace(" ", "_")
        .replace("/", "_")
    )


def _outer_step_count(delta: float) -> int:
    value = TOTAL_TIME / float(delta)
    rounded = int(round(value))
    if not math.isclose(value, rounded, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("Every S1 delta must divide the fixed total time.")
    return rounded


def _deterministic_task_id(
    construction: str,
    formula_label: str,
    delta: float,
    inner_substeps: int | None,
) -> str:
    inner = "na" if inner_substeps is None else str(int(inner_substeps))
    return f"{construction}_{_slug(formula_label)}_d{delta:g}_m{inner}"


def expected_task_manifest_body() -> dict[str, Any]:
    tasks: list[dict[str, Any]] = []
    for construction in CONSTRUCTIONS:
        for label in pd.FORMULA_LABELS:
            for delta in DELTAS:
                substeps_values: Sequence[int | None] = (
                    INNER_SUBSTEPS if construction == "nested" else (None,)
                )
                for substeps in substeps_values:
                    tasks.append(
                        {
                            "task_id": _deterministic_task_id(
                                construction, label, delta, substeps
                            ),
                            "construction": construction,
                            "formula_label": label,
                            "delta": float(delta),
                            "outer_step_count": _outer_step_count(delta),
                            "inner_hd_substeps": substeps,
                        }
                    )
    return {
        "schema_version": EXPECTED_SCHEMA,
        "method": METHOD,
        "configuration": dict(CONFIGURATION),
        "deterministic_tasks": tasks,
        "primary_finite_task_count": len(tasks) * len(RTE_TOTAL_STEPS),
        "adaptive_rules": {
            "k4_trigger": [
                "b2_selected_setting_per_construction",
                "provisional_b4_k2_selected_setting_per_construction",
                "b2_setting_for_formula_with_no_k2_feasible_point",
                "selected_setting_at_primary_r_upper_boundary",
            ],
            "one_step_inner_boundary_extension": INNER_BOUNDARY_EXTENSION,
            "one_step_r_boundary_extension": RTE_BOUNDARY_EXTENSION,
            "extension_rounds": 1,
        },
        "selection_models": ["B0", "B1a", "B1b", "B2", "B4"],
        "classification_priority": ["C", "D", "B", "A"],
        "mandatory_stop_after_s1": True,
        "scope": {
            "fixed_h4_dense_screening": True,
            "common_total_time_and_phase_budget": True,
            "nested_and_native_construction_axis": True,
            "finite_rte_operator_sampled": False,
            "compiled_circuit_cost_evaluated": False,
            "full_rpe_total_cost_evaluated": False,
            "h12_evaluated": False,
            "backend_or_noise_evaluated": False,
            "scientific_superiority_claimed": False,
        },
    }


def finalize_expected_task_manifest(
    body: Mapping[str, Any], *, provenance: Mapping[str, Any]
) -> dict[str, Any]:
    payload = {**dict(body), "provenance": dict(provenance)}
    payload["content_fingerprint"] = pd.fingerprint(payload)
    validate_expected_task_manifest(payload)
    return payload


def validate_expected_task_manifest(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != EXPECTED_SCHEMA:
        raise ValueError("Unexpected P-D S1 expected-task schema.")
    unsigned = dict(payload)
    observed = unsigned.pop("content_fingerprint", None)
    if observed != pd.fingerprint(unsigned):
        raise ValueError("P-D S1 expected-task fingerprint mismatch.")
    canonical = expected_task_manifest_body()
    for key in (
        "configuration",
        "deterministic_tasks",
        "primary_finite_task_count",
        "adaptive_rules",
        "selection_models",
        "classification_priority",
        "mandatory_stop_after_s1",
        "scope",
    ):
        if payload.get(key) != canonical[key]:
            raise ValueError(f"P-D S1 expected-task field changed: {key}")


def _phase_distance(left: complex, right: complex) -> float:
    if abs(left) == 0.0 or abs(right) == 0.0:
        return math.pi
    return float(abs(np.angle(left * np.conj(right))))


def _signal(unitary: np.ndarray, state: np.ndarray, steps: int) -> complex:
    return complex(np.vdot(state, np.linalg.matrix_power(unitary, int(steps)) @ state))


def _evolution(
    eigensystem: tuple[np.ndarray, np.ndarray], signed_time: float
) -> np.ndarray:
    values, vectors = eigensystem
    return (
        vectors * np.exp(-1.0j * float(signed_time) * values)[np.newaxis, :]
    ) @ vectors.conj().T


def _internal_hd_unitary(
    terms: Sequence[np.ndarray],
    eigensystems: Sequence[tuple[np.ndarray, np.ndarray]],
    signed_time: float,
    substeps: int,
) -> np.ndarray:
    one_step = pd._pf_unitary(
        terms,
        "2nd",
        float(signed_time) / int(substeps),
        eigensystems=eigensystems,
    )
    return np.linalg.matrix_power(one_step, int(substeps))

def _nested_outer_unitary(
    hd_terms: Sequence[np.ndarray],
    hd_term_eigensystems: Sequence[tuple[np.ndarray, np.ndarray]],
    hr_eigensystem: tuple[np.ndarray, np.ndarray],
    formula_label: str,
    delta: float,
    substeps: int,
) -> np.ndarray:
    dimension = int(hd_terms[0].shape[0])
    result = np.eye(dimension, dtype=np.complex128)
    cache: dict[float, np.ndarray] = {}
    for term_index, weight in pd.iter_pf_steps(2, pd._get_w_list(formula_label)):
        signed_time = float(delta) * float(weight)
        if term_index == 0:
            if signed_time not in cache:
                cache[signed_time] = _internal_hd_unitary(
                    hd_terms,
                    hd_term_eigensystems,
                    signed_time,
                    substeps,
                )
            factor = cache[signed_time]
        else:
            factor = _evolution(hr_eigensystem, signed_time)
        result = factor @ result
    return result


def _tail_coefficients(formula_label: str, num_terms: int) -> tuple[float, ...]:
    return tuple(
        float(weight)
        for term_index, weight in pd.iter_pf_steps(
            int(num_terms), pd._get_w_list(formula_label)
        )
        if term_index == int(num_terms) - 1
    )


def _deterministic_row(
    *,
    construction: str,
    formula_label: str,
    delta: float,
    inner_substeps: int | None,
    exact_target_signal: complex,
    ground_state: np.ndarray,
    hd: np.ndarray,
    hr: np.ndarray,
    exact_outer_eigensystems: Sequence[tuple[np.ndarray, np.ndarray]],
    hd_terms: Sequence[np.ndarray],
    hd_term_eigensystems: Sequence[tuple[np.ndarray, np.ndarray]],
    hr_eigensystem: tuple[np.ndarray, np.ndarray],
) -> dict[str, Any]:
    q = _outer_step_count(delta)
    exact_outer = pd._pf_unitary(
        (hd, hr),
        formula_label,
        delta,
        eigensystems=exact_outer_eigensystems,
    )
    exact_outer_signal = _signal(exact_outer, ground_state, q)

    if construction == "nested":
        if inner_substeps is None:
            raise ValueError("nested construction requires inner substeps.")
        unitary = _nested_outer_unitary(
            hd_terms,
            hd_term_eigensystems,
            hr_eigensystem,
            formula_label,
            delta,
            int(inner_substeps),
        )
        signal = _signal(unitary, ground_state, q)
        outer_phase = _phase_distance(exact_outer_signal, exact_target_signal)
        inner_phase = _phase_distance(signal, exact_outer_signal)
        conservative_phase = outer_phase + inner_phase
        num_terms = 2
        steps = list(pd.iter_pf_steps(2, pd._get_w_list(formula_label)))
        deterministic_occurrences = sum(index == 0 for index, _ in steps)
        inner_stages = int(inner_substeps) * (2 * len(hd_terms) - 1)
        deterministic_actions = q * deterministic_occurrences * inner_stages
    elif construction == "native":
        if inner_substeps is not None:
            raise ValueError("native construction has no inner substep parameter.")
        native_terms = tuple(hd_terms) + (hr,)
        native_eigensystems = tuple(hd_term_eigensystems) + (hr_eigensystem,)
        unitary = pd._pf_unitary(
            native_terms,
            formula_label,
            delta,
            eigensystems=native_eigensystems,
        )
        signal = _signal(unitary, ground_state, q)
        outer_phase = _phase_distance(signal, exact_target_signal)
        inner_phase = 0.0
        conservative_phase = outer_phase
        num_terms = len(native_terms)
        steps = list(pd.iter_pf_steps(num_terms, pd._get_w_list(formula_label)))
        deterministic_occurrences = sum(
            index != num_terms - 1 for index, _ in steps
        )
        inner_stages = None
        deterministic_actions = q * deterministic_occurrences
    else:
        raise ValueError(f"Unknown construction: {construction}")

    tail_coefficients = _tail_coefficients(formula_label, num_terms)
    return {
        "deterministic_task_id": _deterministic_task_id(
            construction, formula_label, delta, inner_substeps
        ),
        "construction": construction,
        "formula_label": formula_label,
        "delta": float(delta),
        "outer_step_count": q,
        "inner_hd_substeps": inner_substeps,
        "tail_coefficients": list(tail_coefficients),
        "gamma_r": float(math.fsum(abs(value) for value in tail_coefficients)),
        "tail_occurrence_count": len(tail_coefficients),
        "outer_stage_count_total": q * len(steps),
        "deterministic_occurrence_count_per_outer_step": (
            deterministic_occurrences
        ),
        "inner_stages_per_hd_occurrence": inner_stages,
        "deterministic_component_actions_total": int(deterministic_actions),
        "outer_phase_error_rad": float(outer_phase),
        "inner_phase_error_rad": float(inner_phase),
        "deterministic_phase_bound_rad": float(conservative_phase),
        "deterministic_actual_phase_error_rad": _phase_distance(
            signal, exact_target_signal
        ),
        "deterministic_signal_radius": float(abs(signal)),
        "deterministic_signal_phase_rad": float(np.angle(signal)),
        "deterministic_feasible": bool(conservative_phase <= PHASE_BUDGET),
    }


def _finite_candidate(
    deterministic: Mapping[str, Any],
    *,
    lambda_r: float,
    total_rte_steps: int,
    finite_order: int,
) -> dict[str, Any]:
    coefficients = tuple(float(value) for value in deterministic["tail_coefficients"])
    q = int(deterministic["outer_step_count"])
    delta = float(deterministic["delta"])
    gamma = float(deterministic["gamma_r"])
    deterministic_actions = int(
        deterministic["deterministic_component_actions_total"]
    )
    c1_b2 = deterministic_actions + q * int(total_rte_steps)
    ell_b2 = (
        q
        * float(lambda_r) ** 2
        * delta**2
        * gamma**2
        / int(total_rte_steps)
    )
    shot_b2 = float(math.exp(2.0 * ell_b2))
    g_b2 = float(c1_b2 * shot_b2)
    candidate_id = (
        f"{deterministic['deterministic_task_id']}_r{int(total_rte_steps)}"
        f"_k{int(finite_order)}"
    )
    leading_payload = {
        **dict(deterministic),
        "candidate_id": candidate_id,
        "rte_total_short_steps_per_outer_step": int(total_rte_steps),
        "finite_taylor_order": int(finite_order),
        "b2_log_normalization_total": float(ell_b2),
        "b2_shot_factor": shot_b2,
        "b2_c1shot_proxy": int(c1_b2),
        "b2_objective": g_b2,
    }
    if int(total_rte_steps) < len(coefficients):
        return {
            **leading_payload,
            "integer_allocation": None,
            "allocation_feasible": False,
            "allocation_infeasibility_reason": (
                "fewer_short_steps_than_tail_occurrences"
            ),
            "b4_log_normalization_total": None,
            "b4_shot_factor": None,
            "b4_c1shot_expected_component_actions": None,
            "b4_objective": None,
            "finite_signal_error_bound": None,
            "finite_phase_error_bound_rad": None,
            "b4_total_phase_error_bound_rad": None,
            "b4_feasible": False,
            "b2_objective_relative_error_vs_b4": None,
            "b2_shot_factor_relative_error_vs_b4": None,
            "occurrences": [],
        }

    allocation = pd._integer_allocation(
        coefficients,
        total_steps=int(total_rte_steps),
        policy="absolute_time_proportional",
    )
    ell_one_outer = 0.0
    residual_one_outer = 0.0
    expected_tail_actions_one_outer = 0.0
    occurrence_rows: list[dict[str, Any]] = []
    for coefficient, rte_steps in zip(coefficients, allocation, strict=True):
        tau = float(lambda_r) * delta * coefficient / int(rte_steps)
        distribution = finite_rte_distribution(tau, int(finite_order))
        expected_actions = math.fsum(
            probability * (int(order) + 1)
            for order, probability in zip(
                distribution.orders,
                distribution.order_probabilities,
                strict=True,
            )
        )
        ell_one_outer += int(rte_steps) * math.log(
            distribution.exact_finite_distribution
        )
        residual_one_outer += (
            int(rte_steps) * distribution.step_truncation_residual_bound
        )
        expected_tail_actions_one_outer += int(rte_steps) * expected_actions
        occurrence_rows.append(
            {
                "tail_coefficient": coefficient,
                "rte_steps": int(rte_steps),
                "dimensionless_short_step_time": float(tau),
                "finite_distribution_normalization": float(
                    distribution.exact_finite_distribution
                ),
                "step_truncation_residual_bound": float(
                    distribution.step_truncation_residual_bound
                ),
                "expected_component_actions_per_short_step": float(
                    expected_actions
                ),
            }
        )
    ell_b4 = q * ell_one_outer
    residual_b4 = q * residual_one_outer
    shot_b4 = float(math.exp(2.0 * ell_b4))
    c1_b4 = float(
        deterministic_actions + q * expected_tail_actions_one_outer
    )
    g_b4 = float(c1_b4 * shot_b4)
    radius = float(deterministic["deterministic_signal_radius"])
    finite_phase = (
        float(math.asin(residual_b4 / radius))
        if radius > 0.0 and residual_b4 < radius
        else None
    )
    deterministic_bound = float(deterministic["deterministic_phase_bound_rad"])
    total_phase = (
        deterministic_bound + finite_phase if finite_phase is not None else None
    )
    finite_feasible = bool(
        total_phase is not None and total_phase <= PHASE_BUDGET
    )
    return {
        **leading_payload,
        "integer_allocation": list(allocation),
        "allocation_feasible": True,
        "allocation_infeasibility_reason": None,
        "b4_log_normalization_total": float(ell_b4),
        "b4_shot_factor": shot_b4,
        "b4_c1shot_expected_component_actions": c1_b4,
        "b4_objective": g_b4,
        "finite_signal_error_bound": float(residual_b4),
        "finite_phase_error_bound_rad": finite_phase,
        "b4_total_phase_error_bound_rad": total_phase,
        "b4_feasible": finite_feasible,
        "b2_objective_relative_error_vs_b4": float(g_b2 / g_b4 - 1.0),
        "b2_shot_factor_relative_error_vs_b4": float(
            shot_b2 / shot_b4 - 1.0
        ),
        "occurrences": occurrence_rows,
    }

_OBJECTIVE_FIELD = {
    "B0": "deterministic_actual_phase_error_rad",
    "B1a": "outer_stage_count_total",
    "B1b": "b2_c1shot_proxy",
    "B2": "b2_objective",
    "B4": "b4_objective",
}


def _eligible(row: Mapping[str, Any], model: str) -> bool:
    return bool(row["b4_feasible"] if model == "B4" else row["deterministic_feasible"])


def _selection_key(row: Mapping[str, Any], model: str) -> tuple[Any, ...]:
    inner = row["inner_hd_substeps"]
    return (
        float(row[_OBJECTIVE_FIELD[model]]),
        float(row["deterministic_phase_bound_rad"]),
        int(row["b2_c1shot_proxy"]),
        str(row["formula_label"]),
        float(row["delta"]),
        -1 if inner is None else int(inner),
        int(row["rte_total_short_steps_per_outer_step"]),
        int(row["finite_taylor_order"]),
    )


def _select(
    rows: Sequence[Mapping[str, Any]], model: str
) -> Mapping[str, Any] | None:
    eligible = [row for row in rows if _eligible(row, model)]
    return min(eligible, key=lambda row: _selection_key(row, model)) if eligible else None


def _selection_summary(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "candidate_id": row["candidate_id"],
        "deterministic_task_id": row["deterministic_task_id"],
        "construction": row["construction"],
        "formula_label": row["formula_label"],
        "delta": row["delta"],
        "outer_step_count": row["outer_step_count"],
        "inner_hd_substeps": row["inner_hd_substeps"],
        "rte_total_short_steps_per_outer_step": row[
            "rte_total_short_steps_per_outer_step"
        ],
        "finite_taylor_order": row["finite_taylor_order"],
        "deterministic_phase_bound_rad": row[
            "deterministic_phase_bound_rad"
        ],
        "finite_phase_error_bound_rad": row["finite_phase_error_bound_rad"],
        "b2_objective": row["b2_objective"],
        "b4_objective": row["b4_objective"],
        "b4_feasible": row["b4_feasible"],
    }


def _selection_block(
    rows: Sequence[Mapping[str, Any]], construction: str | None
) -> dict[str, Any]:
    scoped = [
        row
        for row in rows
        if construction is None or row["construction"] == construction
    ]
    chosen = {model: _select(scoped, model) for model in _OBJECTIVE_FIELD}
    reference = chosen["B4"]
    reference_value = (
        float(reference["b4_objective"]) if reference is not None else None
    )
    regret: dict[str, Any] = {}
    for model, row in chosen.items():
        if row is None or reference_value is None:
            regret[model] = {"false_acceptance": None, "regret": None}
        elif not row["b4_feasible"]:
            regret[model] = {"false_acceptance": True, "regret": None}
        else:
            regret[model] = {
                "false_acceptance": False,
                "regret": float(row["b4_objective"] / reference_value - 1.0),
            }
    return {
        "scope": "combined" if construction is None else construction,
        "candidate_count": len(scoped),
        "b4_feasible_count": sum(bool(row["b4_feasible"]) for row in scoped),
        "selections": {
            model: _selection_summary(row) for model, row in chosen.items()
        },
        "b4_reference_objective": reference_value,
        "regret_against_b4": regret,
    }


def _selected_rows(
    rows: Sequence[Mapping[str, Any]], blocks: Sequence[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    by_id = {str(row["candidate_id"]): row for row in rows}
    output: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for block in blocks:
        for summary in block["selections"].values():
            if summary is None:
                continue
            candidate_id = str(summary["candidate_id"])
            if candidate_id not in seen:
                output.append(by_id[candidate_id])
                seen.add(candidate_id)
    return output


def _classify(
    blocks: Mapping[str, Mapping[str, Any]],
    k4: Mapping[str, Any],
    *,
    unresolved_boundary: bool,
) -> dict[str, Any]:
    b2_bad = False
    b1_bad = False
    false_acceptance = False
    for block in (blocks[name] for name in (*CONSTRUCTIONS, "combined")):
        for model in ("B1a", "B1b", "B2"):
            record = block["regret_against_b4"][model]
            false_acceptance = false_acceptance or record["false_acceptance"] is True
            regret = record["regret"]
            is_bad = bool(
                record["false_acceptance"] is True
                or (regret is not None and regret > DECISION_REGRET)
            )
            if model == "B2":
                b2_bad = b2_bad or is_bad
            else:
                b1_bad = b1_bad or is_bad

    k4_relevant = bool(k4.get("decision_relevant", False))
    nested_choice = blocks["nested"]["selections"]["B4"]
    native_choice = blocks["native"]["selections"]["B4"]
    construction_difference = bool(
        nested_choice is not None
        and native_choice is not None
        and nested_choice["formula_label"] != native_choice["formula_label"]
    )
    if b2_bad or k4_relevant:
        case = "C"
        reason = "finite_RTE_correction_is_decision_relevant"
    elif construction_difference:
        case = "D"
        reason = "nested_native_construction_changes_the_selected_formula"
    elif b1_bad:
        case = "B"
        reason = "absolute_tail_time_model_is_needed_but_finite_correction_is_not"
    else:
        case = "A"
        reason = "simple_stage_or_leading_tail_models_are_sufficient_on_s1"
    status = {
        "A": "stop_pd_after_s1_simple_model_sufficient",
        "B": "stop_and_audit_novel_applicability_condition_before_any_s2",
        "C": "stop_after_s1_and_redesign_finite_rte_aware_research",
        "D": "stop_after_s1_and_narrow_to_nested_precision_allocation",
    }[case]
    if unresolved_boundary:
        status = "stop_s1_undetermined_boundary_no_go_decision"
    return {
        "primary_case": case,
        "reason": reason,
        "decision_regret_threshold": DECISION_REGRET,
        "b1_decision_relevant": b1_bad,
        "b2_finite_model_failure": b2_bad,
        "k4_decision_relevant": k4_relevant,
        "nested_native_formula_difference": construction_difference,
        "any_false_acceptance": false_acceptance,
        "undetermined_boundary": unresolved_boundary,
        "status": status,
        "mandatory_stop_before_s2": True,
    }

def evaluate_s1(
    snapshot_path: str | Path,
    expected_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    validate_expected_task_manifest(expected_manifest)
    hamiltonian = pd.load_connected_cluster_hamiltonian_snapshot(snapshot_path)
    if hamiltonian.n_qubits != 8 or hamiltonian.n_blocks != 12:
        raise ValueError("P-D S1 requires the fixed H4 rank-12 snapshot.")
    sector = pd.PhysicalSector.number_sector(
        n_qubits=hamiltonian.n_qubits,
        n_electrons=N_ELECTRONS,
    )
    full_hamiltonian = np.asarray(
        pd.dense_df_operator_in_sector(
            hamiltonian,
            sector,
            matrix_free_backend="python",
        ),
        dtype=np.complex128,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(full_hamiltonian)
    ground_energy = float(eigenvalues[0])
    ground_state = np.asarray(eigenvectors[:, 0], dtype=np.complex128)
    exact_target_signal = complex(np.exp(-1.0j * ground_energy * TOTAL_TIME))

    hd, partition = realization._dense_hd(hamiltonian, sector, LD)
    hr = np.asarray(full_hamiltonian - hd, dtype=np.complex128)
    hd_terms, reconstruction_residual = realization._fragment_terms(
        hamiltonian, sector, LD
    )
    hd_term_eigensystems = tuple(np.linalg.eigh(term) for term in hd_terms)
    exact_outer_eigensystems = (np.linalg.eigh(hd), np.linalg.eigh(hr))
    hr_eigensystem = exact_outer_eigensystems[1]
    preparation = pd.prepare_df_partial_s2(
        hamiltonian,
        partition,
        identity_policy="extract_identity_phase",
        coefficient_atol=1.0e-12,
    )
    lambda_r = float(preparation.exact_rte_lambda_r)

    deterministic_rows: list[dict[str, Any]] = []
    deterministic_by_id: dict[str, dict[str, Any]] = {}

    def add_deterministic(
        construction: str,
        label: str,
        delta: float,
        substeps: int | None,
    ) -> dict[str, Any]:
        task_id = _deterministic_task_id(construction, label, delta, substeps)
        if task_id in deterministic_by_id:
            return deterministic_by_id[task_id]
        row = _deterministic_row(
            construction=construction,
            formula_label=label,
            delta=delta,
            inner_substeps=substeps,
            exact_target_signal=exact_target_signal,
            ground_state=ground_state,
            hd=hd,
            hr=hr,
            exact_outer_eigensystems=exact_outer_eigensystems,
            hd_terms=hd_terms,
            hd_term_eigensystems=hd_term_eigensystems,
            hr_eigensystem=hr_eigensystem,
        )
        deterministic_rows.append(row)
        deterministic_by_id[task_id] = row
        return row

    for task in expected_manifest["deterministic_tasks"]:
        add_deterministic(
            str(task["construction"]),
            str(task["formula_label"]),
            float(task["delta"]),
            task["inner_hd_substeps"],
        )

    primary_rows: list[dict[str, Any]] = []
    finite_by_id: dict[str, dict[str, Any]] = {}

    def add_finite(
        deterministic: Mapping[str, Any], rte_steps: int, finite_order: int
    ) -> dict[str, Any]:
        candidate_id = (
            f"{deterministic['deterministic_task_id']}_r{int(rte_steps)}"
            f"_k{int(finite_order)}"
        )
        if candidate_id in finite_by_id:
            return finite_by_id[candidate_id]
        row = _finite_candidate(
            deterministic,
            lambda_r=lambda_r,
            total_rte_steps=int(rte_steps),
            finite_order=int(finite_order),
        )
        finite_by_id[candidate_id] = row
        if int(finite_order) == PRIMARY_K:
            primary_rows.append(row)
        return row

    for deterministic in list(deterministic_rows):
        for rte_steps in RTE_TOTAL_STEPS:
            add_finite(deterministic, rte_steps, PRIMARY_K)

    def make_blocks(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
        return {
            name: _selection_block(rows, None if name == "combined" else name)
            for name in (*CONSTRUCTIONS, "combined")
        }

    provisional_blocks = make_blocks(primary_rows)
    provisional_selected = _selected_rows(
        primary_rows, list(provisional_blocks.values())
    )
    boundary_actions: list[dict[str, Any]] = []
    for row in provisional_selected:
        if (
            row["construction"] == "nested"
            and row["inner_hd_substeps"] == max(INNER_SUBSTEPS)
        ):
            extended = add_deterministic(
                "nested",
                str(row["formula_label"]),
                float(row["delta"]),
                INNER_BOUNDARY_EXTENSION,
            )
            for rte_steps in RTE_TOTAL_STEPS:
                add_finite(extended, rte_steps, PRIMARY_K)
            boundary_actions.append(
                {
                    "axis": "inner_hd_substeps",
                    "source_candidate_id": row["candidate_id"],
                    "extension_value": INNER_BOUNDARY_EXTENSION,
                }
            )
        if row["rte_total_short_steps_per_outer_step"] == max(RTE_TOTAL_STEPS):
            extension = add_finite(row, RTE_BOUNDARY_EXTENSION, PRIMARY_K)
            boundary_actions.append(
                {
                    "axis": "rte_short_steps_per_outer_step",
                    "source_candidate_id": row["candidate_id"],
                    "extension_value": RTE_BOUNDARY_EXTENSION,
                    "extension_candidate_id": extension["candidate_id"],
                }
            )

    blocks_after_inner = make_blocks(primary_rows)
    for row in _selected_rows(primary_rows, list(blocks_after_inner.values())):
        if row["rte_total_short_steps_per_outer_step"] == max(RTE_TOTAL_STEPS):
            extension = add_finite(row, RTE_BOUNDARY_EXTENSION, PRIMARY_K)
            if not any(
                action["axis"] == "rte_short_steps_per_outer_step"
                and action["source_candidate_id"] == row["candidate_id"]
                for action in boundary_actions
            ):
                boundary_actions.append(
                    {
                        "axis": "rte_short_steps_per_outer_step",
                        "source_candidate_id": row["candidate_id"],
                        "extension_value": RTE_BOUNDARY_EXTENSION,
                        "extension_candidate_id": extension["candidate_id"],
                    }
                )

    primary_blocks = make_blocks(primary_rows)
    trigger_ids: set[str] = set()
    trigger_reasons: dict[str, set[str]] = {}

    def trigger(task_id: str, reason: str) -> None:
        trigger_ids.add(task_id)
        trigger_reasons.setdefault(task_id, set()).add(reason)

    for construction in CONSTRUCTIONS:
        block = primary_blocks[construction]
        for model in ("B2", "B4"):
            selected = block["selections"][model]
            if selected is not None:
                task_id = str(selected["deterministic_task_id"])
                trigger(task_id, f"{model}_selected")
                if selected["rte_total_short_steps_per_outer_step"] == max(
                    RTE_TOTAL_STEPS
                ):
                    trigger(task_id, f"{model}_selected_at_r_upper_boundary")
        construction_rows = [
            row for row in primary_rows if row["construction"] == construction
        ]
        for label in pd.FORMULA_LABELS:
            formula_rows = [
                row
                for row in construction_rows
                if row["formula_label"] == label
            ]
            if formula_rows and not any(row["b4_feasible"] for row in formula_rows):
                b2_choice = _select(formula_rows, "B2")
                if b2_choice is not None:
                    trigger(
                        str(b2_choice["deterministic_task_id"]),
                        "formula_has_no_k2_finite_feasible_point",
                    )

    sensitivity_rows: list[dict[str, Any]] = []
    for task_id in sorted(trigger_ids):
        deterministic = deterministic_by_id[task_id]
        for rte_steps in (*RTE_TOTAL_STEPS, RTE_BOUNDARY_EXTENSION):
            sensitivity_rows.append(
                add_finite(deterministic, rte_steps, SENSITIVITY_K)
            )

    k4_by_construction: dict[str, Any] = {}
    k4_decision_relevant = False
    for construction in CONSTRUCTIONS:
        scoped = [
            row for row in sensitivity_rows if row["construction"] == construction
        ]
        k4_choice = _select(scoped, "B4")
        k2_summary = primary_blocks[construction]["selections"]["B4"]
        k4_choice_summary = _selection_summary(k4_choice)
        setting_changed = bool(
            k2_summary is not None
            and k4_choice_summary is not None
            and any(
                k2_summary[key] != k4_choice_summary[key]
                for key in (
                    "formula_label",
                    "delta",
                    "inner_hd_substeps",
                    "rte_total_short_steps_per_outer_step",
                )
            )
        )
        formula_changed = bool(
            setting_changed
            and k2_summary["formula_label"] != k4_choice_summary["formula_label"]
        )
        relative_change = None
        if k2_summary is not None and k4_choice_summary is not None:
            relative_change = float(
                k4_choice_summary["b4_objective"]
                / k2_summary["b4_objective"]
                - 1.0
            )
        relevant = bool(
            setting_changed
            and relative_change is not None
            and abs(relative_change) > DECISION_REGRET
        )
        k4_decision_relevant = k4_decision_relevant or relevant
        k4_by_construction[construction] = {
            "triggered_deterministic_task_ids": sorted(
                task_id
                for task_id in trigger_ids
                if deterministic_by_id[task_id]["construction"] == construction
            ),
            "selected_k2": k2_summary,
            "selected_restricted_k4": k4_choice_summary,
            "formula_changed": formula_changed,
            "setting_changed": setting_changed,
            "relative_objective_change": relative_change,
            "decision_relevant": relevant,
        }
    k4_summary = {
        "is_restricted_not_exhaustive": True,
        "trigger_reasons": {
            key: sorted(values) for key, values in sorted(trigger_reasons.items())
        },
        "evaluated_candidate_count": len(sensitivity_rows),
        "by_construction": k4_by_construction,
        "decision_relevant": k4_decision_relevant,
    }

    final_selected = _selected_rows(primary_rows, list(primary_blocks.values()))
    unresolved_boundary = any(
        row["inner_hd_substeps"] == INNER_BOUNDARY_EXTENSION
        or row["rte_total_short_steps_per_outer_step"] == RTE_BOUNDARY_EXTENSION
        for row in final_selected
    )
    classification = _classify(
        primary_blocks,
        k4_summary,
        unresolved_boundary=unresolved_boundary,
    )

    return {
        "schema_version": RESULT_SCHEMA,
        "method": METHOD,
        "expected_task_fingerprint": expected_manifest["content_fingerprint"],
        "configuration": dict(CONFIGURATION),
        "hamiltonian": {
            "molecule": "H4 linear chain",
            "geometry_angstrom": 1.0,
            "basis": "STO-3G",
            "n_qubits": int(hamiltonian.n_qubits),
            "n_electrons": N_ELECTRONS,
            "df_rank": int(hamiltonian.n_blocks),
            "ld": LD,
            "hamiltonian_hash": pd.df_hamiltonian_hash(hamiltonian),
            "ground_energy_hartree": ground_energy,
            "sector_dimension": int(sector.dimension),
            "exact_rte_lambda_r": lambda_r,
            "fragment_term_count": len(hd_terms),
            "fragment_reconstruction_residual_frobenius": float(
                reconstruction_residual
            ),
        },
        "deterministic_rows": deterministic_rows,
        "primary_k2_candidates": primary_rows,
        "primary_selection": primary_blocks,
        "boundary_extensions": boundary_actions,
        "restricted_k4_sensitivity_candidates": sensitivity_rows,
        "restricted_k4_sensitivity": k4_summary,
        "classification": classification,
        "scope": dict(expected_manifest["scope"]),
        "limitations": [
            "S1 is a fixed-H4 dense screening calculation, not a general molecular conclusion.",
            "B1/B2/B4 use analytic component-action and shot-inflation proxies, not compiled gate counts or measured shots.",
            "The finite RTE term is a conservative analytic signal-error bound; an H4 sampled operator is not executed.",
            "K=4 is a preregistered restricted sensitivity analysis rather than an exhaustive grid.",
            "The native construction is a diagnostic axis, not an implementation-optimal circuit claim.",
            "H12, long RPE, backend noise, and final total cost are outside S1.",
        ],
    }

def finalize_result(
    body: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any],
    source_evidence: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload = {
        **dict(body),
        "provenance": dict(provenance),
        "source_evidence": [dict(row) for row in source_evidence],
    }
    payload["content_fingerprint"] = pd.fingerprint(payload)
    validate_result(payload)
    return payload


def validate_result(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != RESULT_SCHEMA:
        raise ValueError("Unexpected P-D S1 result schema.")
    unsigned = dict(payload)
    observed = unsigned.pop("content_fingerprint", None)
    if observed != pd.fingerprint(unsigned):
        raise ValueError("P-D S1 result fingerprint mismatch.")
    if payload.get("configuration") != CONFIGURATION:
        raise ValueError("P-D S1 result changed frozen configuration.")
    classification = payload.get("classification", {})
    if classification.get("primary_case") not in ("A", "B", "C", "D"):
        raise ValueError("P-D S1 result lacks the preregistered case classification.")
    if classification.get("mandatory_stop_before_s2") is not True:
        raise ValueError("P-D S1 must stop before S2.")
    scope = payload.get("scope", {})
    forbidden_true = (
        "finite_rte_operator_sampled",
        "compiled_circuit_cost_evaluated",
        "full_rpe_total_cost_evaluated",
        "h12_evaluated",
        "backend_or_noise_evaluated",
        "scientific_superiority_claimed",
    )
    if any(scope.get(key) is not False for key in forbidden_true):
        raise ValueError("P-D S1 result overstates its scope.")
    if not payload.get("primary_k2_candidates"):
        raise ValueError("P-D S1 result has no primary finite candidates.")
