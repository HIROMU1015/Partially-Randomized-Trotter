from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import BETA, DEFAULT_BASIS, DEFAULT_DISTANCE, PFLabel
from .grouped_uwc_comparison import GroupedPFQPERow, compare_grouped_uwc_pf_qpe
from .uwc import UWCConfig


DEFAULT_THETA_SWEEP_VALUES: tuple[float, ...] = (
    0.0,
    0.001,
    0.003,
    0.01,
    0.03,
    0.1,
    0.3,
    1.0,
)


@dataclass(frozen=True)
class GroupedUWCThetaSweepResult:
    config: dict[str, Any]
    rows: tuple[dict[str, Any], ...]
    summary: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "rows": list(self.rows),
            "summary": list(self.summary),
        }


def _as_jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_as_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_as_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _as_jsonable(item) for key, item in value.items()}
    return value


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_as_jsonable(value), sort_keys=True)
    return value


def _sector_check_passed(check: Mapping[str, Any] | None) -> bool:
    if not check:
        return True
    if check.get("checked") is False:
        return True
    tolerance = float(check.get("tolerance", 0.0) or 0.0)
    max_abs_shift = check.get("max_abs_shift_on_sector")
    if max_abs_shift is not None and float(max_abs_shift) > tolerance:
        return False
    ground_difference = check.get("ground_energy_difference")
    if ground_difference is not None and float(ground_difference) > tolerance:
        return False
    bound = check.get("ground_energy_difference_bound")
    if bound is not None and float(bound) > tolerance:
        return False
    return True


def _explicit_diagonalization_skipped(check: Mapping[str, Any] | None) -> bool:
    if not check or check.get("checked") is False:
        return False
    method = str(check.get("ground_energy_check_method", ""))
    return bool(method and method != "explicit_diagonalization")


def _flatten_row(
    row: GroupedPFQPERow,
    *,
    theta: float,
    power: str,
    baseline_row: GroupedPFQPERow,
    uwc_row: GroupedPFQPERow,
) -> dict[str, Any]:
    row_metadata = row.metadata or {}
    uwc_metadata = (uwc_row.metadata or {}).get("uwc_metadata", {})
    uwc_parameters = dict(uwc_metadata.get("uwc_parameters", {}))
    sector_check = uwc_metadata.get("sector_preservation_check")
    baseline_metadata = baseline_row.metadata or {}
    uwc_row_metadata = uwc_row.metadata or {}
    baseline_group_sizes = tuple(baseline_metadata.get("group_sizes", ()))
    uwc_group_sizes = tuple(uwc_row_metadata.get("uwc_group_sizes", ()))
    warnings = tuple(uwc_metadata.get("warnings", ())) if row.method == "uwc_grouped" else tuple()

    alpha_backend = row_metadata.get("alpha_fit_backend")
    if alpha_backend is None and str(row.alpha_source).startswith("grouped_artifact"):
        alpha_backend = "artifact"

    return {
        "molecule": row.molecule,
        "molecule_type": int(row.molecule_type),
        "pf_label": row.pf_label,
        "order": int(row.order),
        "theta": float(theta),
        "power": str(power),
        "method": row.method,
        "uwc_method": row.uwc_method,
        "uwc_objective": row.uwc_objective,
        "alpha": float(row.alpha),
        "baseline_alpha": float(baseline_row.alpha),
        "alpha_ratio_vs_grouped_baseline": float(row.alpha_ratio_vs_grouped_baseline),
        "qpe_iteration_factor": float(row.qpe_iteration_factor),
        "step_pauli_rotations": int(row.step_pauli_rotations),
        "total_pauli_rotations": float(row.total_pauli_rotations),
        "step_rz_layers": int(row.step_rz_layers),
        "total_rz_layers": float(row.total_rz_layers),
        "step_t_depth": float(row.step_t_depth),
        "total_t_depth": float(row.total_t_depth),
        "cost_metric": row.cost_metric,
        "g_total": float(row.g_total),
        "baseline_g_total": float(baseline_row.g_total),
        "cost_ratio_vs_grouped_baseline": float(row.cost_ratio_vs_grouped_baseline),
        "step_cost_ratio_vs_grouped_baseline": float(row.step_cost_ratio_vs_grouped_baseline),
        "num_groups": int(row.num_groups),
        "num_pauli_terms": int(row.num_pauli_terms),
        "grouping_rule": row.grouping_rule,
        "target_error": float(row.target_error),
        "qpe_beta": float(row.qpe_beta),
        "alpha_backend": alpha_backend,
        "alpha_requested_backend": row_metadata.get("alpha_requested_backend"),
        "alpha_source": row.alpha_source,
        "time_grid": tuple(float(value) for value in row.time_grid),
        "sector_preservation_check": sector_check if row.method == "uwc_grouped" else None,
        "sector_check_passed": _sector_check_passed(sector_check) if row.method == "uwc_grouped" else True,
        "uwc_hamiltonian_hash": row.uwc_hamiltonian_hash,
        "warnings": warnings,
        "target_particle_number": uwc_parameters.get("target_particle_number") if row.method == "uwc_grouped" else None,
        "target_particle_number_source": uwc_parameters.get("target_particle_number_source") if row.method == "uwc_grouped" else None,
        "max_abs_shift_on_sector": None if not sector_check else sector_check.get("max_abs_shift_on_sector"),
        "ground_energy_difference": None if not sector_check else sector_check.get("ground_energy_difference"),
        "ground_energy_difference_bound": None if not sector_check else sector_check.get("ground_energy_difference_bound"),
        "sector_dimension": None if not sector_check else sector_check.get("sector_dimension"),
        "sector_check_method": None if not sector_check else sector_check.get("ground_energy_check_method"),
        "explicit_diagonalization_skipped": _explicit_diagonalization_skipped(sector_check) if row.method == "uwc_grouped" else False,
        "baseline_num_groups": int(baseline_row.num_groups),
        "uwc_num_groups": int(uwc_row.num_groups),
        "baseline_num_pauli_terms": int(baseline_row.num_pauli_terms),
        "uwc_num_pauli_terms": int(uwc_row.num_pauli_terms),
        "baseline_group_sizes": baseline_group_sizes,
        "uwc_group_sizes": uwc_group_sizes,
        "baseline_step_pauli_rotations": int(baseline_row.step_pauli_rotations),
        "uwc_step_pauli_rotations": int(uwc_row.step_pauli_rotations),
        "baseline_step_rz_layers": int(baseline_row.step_rz_layers),
        "uwc_step_rz_layers": int(uwc_row.step_rz_layers),
    }


def _group_structure_changed(row: Mapping[str, Any]) -> bool:
    return bool(
        row.get("baseline_num_groups") != row.get("uwc_num_groups")
        or row.get("baseline_num_pauli_terms") != row.get("uwc_num_pauli_terms")
        or tuple(row.get("baseline_group_sizes") or ()) != tuple(row.get("uwc_group_sizes") or ())
    )


def _step_cost_changed(row: Mapping[str, Any]) -> bool:
    return bool(
        row.get("baseline_step_pauli_rotations") != row.get("uwc_step_pauli_rotations")
        or row.get("baseline_step_rz_layers") != row.get("uwc_step_rz_layers")
    )


def _summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline_alpha_source: str,
    theta_zero_tolerance: float,
) -> tuple[dict[str, Any], ...]:
    uwc_rows = [row for row in rows if row.get("method") == "uwc_grouped"]
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in uwc_rows:
        groups.setdefault((str(row["molecule"]), str(row["pf_label"])), []).append(row)

    summary: list[dict[str, Any]] = []
    for (molecule, pf_label), group_rows in sorted(groups.items()):
        best = min(group_rows, key=lambda row: float(row["cost_ratio_vs_grouped_baseline"]))
        theta_zero = next(
            (row for row in group_rows if math.isclose(float(row["theta"]), 0.0, abs_tol=1e-15)),
            None,
        )
        theta_zero_passed: bool | None = None
        if baseline_alpha_source == "fit" and theta_zero is not None:
            theta_zero_passed = bool(
                abs(float(theta_zero["alpha_ratio_vs_grouped_baseline"]) - 1.0)
                <= theta_zero_tolerance
                and abs(float(theta_zero["cost_ratio_vs_grouped_baseline"]) - 1.0)
                <= theta_zero_tolerance
                and abs(float(theta_zero["step_cost_ratio_vs_grouped_baseline"]) - 1.0)
                <= theta_zero_tolerance
            )
        summary.append(
            {
                "molecule": molecule,
                "molecule_type": int(best["molecule_type"]),
                "pf_label": pf_label,
                "order": int(best["order"]),
                "power": best["power"],
                "cost_metric": best["cost_metric"],
                "baseline_alpha_source": baseline_alpha_source,
                "best_theta": float(best["theta"]),
                "best_cost_ratio": float(best["cost_ratio_vs_grouped_baseline"]),
                "alpha_ratio": float(best["alpha_ratio_vs_grouped_baseline"]),
                "step_cost_ratio": float(best["step_cost_ratio_vs_grouped_baseline"]),
                "best_alpha": float(best["alpha"]),
                "best_g_total": float(best["g_total"]),
                "sector_check_passed": bool(best["sector_check_passed"]),
                "all_sector_checks_passed": all(bool(row["sector_check_passed"]) for row in group_rows),
                "group_structure_changed": _group_structure_changed(best),
                "any_group_structure_changed": any(_group_structure_changed(row) for row in group_rows),
                "step_cost_changed": _step_cost_changed(best),
                "any_step_cost_changed": any(_step_cost_changed(row) for row in group_rows),
                "theta_zero_sanity_passed": theta_zero_passed,
            }
        )
    return tuple(summary)


def run_grouped_uwc_theta_sweep(
    molecule_types: Sequence[int],
    *,
    pf_labels: Sequence[PFLabel],
    theta_values: Sequence[float] = DEFAULT_THETA_SWEEP_VALUES,
    power: str = "quadratic",
    target_error: float,
    distance: float = DEFAULT_DISTANCE,
    basis: str = DEFAULT_BASIS,
    qpe_beta: float = BETA,
    cost_metric: str = "rz_layers",
    t_values: Sequence[float] | None = None,
    baseline_alpha_source: str = "artifact",
    use_original_grouped_artifact: bool = False,
    use_reference_rz_layers: bool = True,
    alpha_backend: str = "cpu",
    alpha_gpu_ids: Sequence[str] = ("0",),
    alpha_parallel_processes: int | None = None,
    alpha_chunk_splits: int = 1,
    alpha_gpu_optimization_level: int = 0,
    alpha_gpu_debug: bool = False,
    uwc_objective: str = "l1_norm",
    uwc_target_ld: int | None = None,
    uwc_target_particle_number: int | None = None,
    uwc_sector_energy_tolerance: float = 1e-8,
    uwc_sector_energy_check: str = "warn",
    uwc_max_sector_dimension_for_check: int = 256,
    include_baseline_rows: bool = True,
    theta_zero_tolerance: float = 1e-8,
) -> GroupedUWCThetaSweepResult:
    if not molecule_types:
        raise ValueError("molecule_types must not be empty.")
    if not pf_labels:
        raise ValueError("pf_labels must not be empty.")
    if not theta_values:
        raise ValueError("theta_values must not be empty.")
    if str(power) not in {"linear", "quadratic"}:
        raise ValueError("power must be 'linear' or 'quadratic'.")
    if baseline_alpha_source not in {"artifact", "fit"}:
        raise ValueError("baseline_alpha_source must be 'artifact' or 'fit'.")

    flat_rows: list[dict[str, Any]] = []
    for molecule_type in molecule_types:
        for pf_label in pf_labels:
            for theta in theta_values:
                parameters: dict[str, Any] = {
                    "theta": float(theta),
                    "power": str(power),
                }
                if uwc_target_particle_number is not None:
                    parameters["target_particle_number"] = int(uwc_target_particle_number)
                    parameters["target_particle_number_source"] = "user"
                uwc_config = UWCConfig(
                    enabled=True,
                    method="bliss",
                    objective=uwc_objective,
                    target_ld=uwc_target_ld,
                    parameters=parameters,
                    sector_energy_tolerance=uwc_sector_energy_tolerance,
                    sector_energy_check=uwc_sector_energy_check,
                    max_sector_dimension_for_check=uwc_max_sector_dimension_for_check,
                )
                comparison = compare_grouped_uwc_pf_qpe(
                    int(molecule_type),
                    pf_label=pf_label,
                    uwc_config=uwc_config,
                    target_error=target_error,
                    distance=distance,
                    basis=basis,
                    qpe_beta=qpe_beta,
                    cost_metric=cost_metric,
                    t_values=t_values,
                    baseline_alpha_source=baseline_alpha_source,
                    use_original_grouped_artifact=use_original_grouped_artifact,
                    use_reference_rz_layers=use_reference_rz_layers,
                    alpha_backend=alpha_backend,
                    alpha_gpu_ids=alpha_gpu_ids,
                    alpha_parallel_processes=alpha_parallel_processes,
                    alpha_chunk_splits=alpha_chunk_splits,
                    alpha_gpu_optimization_level=alpha_gpu_optimization_level,
                    alpha_gpu_debug=alpha_gpu_debug,
                )
                baseline_row, uwc_row = comparison.rows
                if include_baseline_rows:
                    flat_rows.append(
                        _flatten_row(
                            baseline_row,
                            theta=float(theta),
                            power=str(power),
                            baseline_row=baseline_row,
                            uwc_row=uwc_row,
                        )
                    )
                flat_rows.append(
                    _flatten_row(
                        uwc_row,
                        theta=float(theta),
                        power=str(power),
                        baseline_row=baseline_row,
                        uwc_row=uwc_row,
                    )
                )

    config = {
        "molecule_types": [int(value) for value in molecule_types],
        "pf_labels": [str(value) for value in pf_labels],
        "theta_values": [float(value) for value in theta_values],
        "power": str(power),
        "target_error": float(target_error),
        "distance": float(distance),
        "basis": basis,
        "qpe_beta": float(qpe_beta),
        "cost_metric": cost_metric,
        "baseline_alpha_source": baseline_alpha_source,
        "use_original_grouped_artifact": bool(use_original_grouped_artifact),
        "use_reference_rz_layers": bool(use_reference_rz_layers),
        "alpha_backend": alpha_backend,
        "alpha_gpu_ids": [str(value) for value in alpha_gpu_ids],
        "alpha_parallel_processes": alpha_parallel_processes,
        "alpha_chunk_splits": int(alpha_chunk_splits),
        "alpha_gpu_optimization_level": int(alpha_gpu_optimization_level),
        "uwc_objective": uwc_objective,
        "uwc_target_ld": uwc_target_ld,
        "uwc_target_particle_number": uwc_target_particle_number,
        "uwc_sector_energy_tolerance": float(uwc_sector_energy_tolerance),
        "uwc_sector_energy_check": uwc_sector_energy_check,
        "uwc_max_sector_dimension_for_check": int(uwc_max_sector_dimension_for_check),
        "include_baseline_rows": bool(include_baseline_rows),
        "theta_zero_tolerance": float(theta_zero_tolerance),
    }
    summary = _summarize_rows(
        flat_rows,
        baseline_alpha_source=baseline_alpha_source,
        theta_zero_tolerance=theta_zero_tolerance,
    )
    return GroupedUWCThetaSweepResult(
        config=config,
        rows=tuple(_as_jsonable(row) for row in flat_rows),
        summary=summary,
    )


def save_grouped_uwc_theta_sweep(
    result: GroupedUWCThetaSweepResult,
    json_path: str | Path,
    *,
    csv_path: str | Path | None = None,
) -> tuple[Path, Path]:
    json_output = Path(json_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    csv_output = Path(csv_path) if csv_path is not None else json_output.with_suffix(".csv")
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    rows = list(result.rows)
    fieldnames = sorted({key for row in rows for key in row})
    with csv_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})
    return json_output, csv_output
