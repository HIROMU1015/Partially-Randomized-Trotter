from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trotterlib.config import BETA, TARGET_ERROR  # noqa: E402
from trotterlib.grouped_uwc_comparison import (  # noqa: E402
    GroupedUWCComparisonResult,
    compare_grouped_uwc_pf_qpe,
)
from trotterlib.uwc import UWCConfig  # noqa: E402


def _parse_json_object(raw: str | None, *, option_name: str) -> dict[str, object]:
    if raw is None:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"{option_name} is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(f"{option_name} must be a JSON object.")
    return dict(parsed)


def _parse_t_values(raw: str | None) -> list[float] | None:
    if raw is None:
        return None
    return [float(token.strip()) for token in raw.split(",") if token.strip()]


def _parse_gpu_ids(raw: str) -> tuple[str, ...]:
    values = tuple(token.strip() for token in raw.split(",") if token.strip())
    return values or ("0",)


def _parse_molecule_types(raw: str | None) -> tuple[int, ...]:
    if raw is None:
        return tuple(range(2, 7))
    values = tuple(int(token.strip()) for token in raw.split(",") if token.strip())
    if not values:
        raise argparse.ArgumentTypeError("--molecule-types must not be empty.")
    return values


def _default_gpu_ids() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    return visible if visible else "0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit grouped PF Trotter alpha for UWC-applied H2-H6 Hamiltonians "
            "using the existing multi-GPU statevector path."
        ),
    )
    parser.add_argument(
        "--molecule-types",
        type=str,
        default=None,
        help="Comma-separated molecule sizes. Default: 2,3,4,5,6.",
    )
    parser.add_argument("--pf-label", type=str, default="2nd")
    parser.add_argument("--target-error", type=float, default=TARGET_ERROR)
    parser.add_argument("--distance", type=float, default=1.0)
    parser.add_argument("--basis", type=str, default="sto-3g")
    parser.add_argument("--qpe-beta", type=float, default=BETA)
    parser.add_argument(
        "--cost-metric",
        choices=("rz_layers", "pauli_rotations"),
        default="rz_layers",
    )
    parser.add_argument(
        "--t-values",
        type=str,
        default=None,
        help="Comma-separated fit times. Default uses the repository time grid.",
    )
    parser.add_argument(
        "--baseline-alpha-source",
        choices=("artifact", "fit"),
        default="artifact",
        help="Use existing grouped coefficient artifacts or recompute baseline alpha.",
    )
    parser.add_argument(
        "--use-original-grouped-artifact",
        action="store_true",
        help="Read baseline alpha from trotter_expo_coeff_gr_original.",
    )
    parser.add_argument(
        "--alpha-backend",
        choices=("cpu", "gpu", "auto"),
        default="gpu",
        help="Backend used when fitting the UWC grouped alpha.",
    )
    parser.add_argument(
        "--alpha-gpu-ids",
        type=str,
        default=_default_gpu_ids(),
        help="Comma-separated GPU ids. Defaults to CUDA_VISIBLE_DEVICES or 0.",
    )
    parser.add_argument("--alpha-parallel-processes", type=int, default=None)
    parser.add_argument("--alpha-chunk-splits", type=int, default=1)
    parser.add_argument("--alpha-gpu-optimization-level", type=int, default=0)
    parser.add_argument("--alpha-gpu-debug", action="store_true")
    parser.add_argument(
        "--reuse-original-ground-state-for-uwc",
        action="store_true",
        help="Use the grouped baseline state/energy for UWC alpha fitting.",
    )
    parser.add_argument(
        "--solve-uwc-ground-state",
        action="store_true",
        help="Solve the UWC Hamiltonian ground state for UWC alpha fitting.",
    )
    parser.add_argument(
        "--no-reference-rz-layer-regroup",
        action="store_true",
        help="Use generic qubit-support RZ layering for UWC.",
    )
    parser.add_argument(
        "--uwc-method",
        choices=(
            "none",
            "simple_shift",
            "test_shift",
            "bliss",
            "orbital_optimization",
            "orbital_optimization_bliss",
        ),
        default="simple_shift",
    )
    parser.add_argument(
        "--uwc-objective",
        choices=("l1_norm", "lambda_r", "estimated_total_cost"),
        default="l1_norm",
    )
    parser.add_argument("--uwc-target-ld", type=int, default=None)
    parser.add_argument("--uwc-max-iterations", type=int, default=0)
    parser.add_argument("--uwc-seed", type=int, default=None)
    parser.add_argument("--uwc-optimizer-json", type=str, default=None)
    parser.add_argument("--uwc-parameters-json", type=str, default=None)
    parser.add_argument("--uwc-simple-shift", type=float, default=0.01)
    parser.add_argument("--uwc-simple-shift-qubit", type=int, default=0)
    parser.add_argument("--uwc-coefficient-scale", type=float, default=None)
    parser.add_argument("--uwc-bliss-theta", type=float, default=None)
    parser.add_argument("--uwc-target-particle-number", type=int, default=None)
    parser.add_argument(
        "--uwc-bliss-power",
        choices=("linear", "quadratic"),
        default=None,
    )
    parser.add_argument("--uwc-sector-energy-tol", type=float, default=1e-8)
    parser.add_argument(
        "--uwc-sector-energy-check",
        choices=("warn", "error", "off"),
        default="warn",
    )
    parser.add_argument("--uwc-max-sector-dimension-for-check", type=int, default=256)
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first molecule that fails.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )
    return parser


def _build_uwc_config(
    args: argparse.Namespace,
) -> UWCConfig:
    optimizer_settings = _parse_json_object(
        args.uwc_optimizer_json,
        option_name="--uwc-optimizer-json",
    )
    parameters = _parse_json_object(
        args.uwc_parameters_json,
        option_name="--uwc-parameters-json",
    )
    if args.uwc_simple_shift is not None:
        parameters["shift"] = args.uwc_simple_shift
        parameters["qubit"] = args.uwc_simple_shift_qubit
    if args.uwc_coefficient_scale is not None:
        parameters["coefficient_scale"] = args.uwc_coefficient_scale
    if args.uwc_bliss_theta is not None:
        parameters["theta"] = args.uwc_bliss_theta
    if args.uwc_target_particle_number is not None:
        parameters["target_particle_number"] = args.uwc_target_particle_number
    if args.uwc_bliss_power is not None:
        parameters["power"] = args.uwc_bliss_power

    return UWCConfig(
        enabled=args.uwc_method != "none",
        method=args.uwc_method,
        objective=args.uwc_objective,
        target_ld=args.uwc_target_ld,
        optimizer_settings=optimizer_settings,
        max_iterations=args.uwc_max_iterations,
        seed=args.uwc_seed,
        parameters=parameters,
        sector_energy_tolerance=args.uwc_sector_energy_tol,
        sector_energy_check=args.uwc_sector_energy_check,
        max_sector_dimension_for_check=args.uwc_max_sector_dimension_for_check,
    )


def _default_output_path(args: argparse.Namespace) -> Path:
    molecule_token = (args.molecule_types or "H2-H6").replace(",", "_")
    suffix = (
        f"{molecule_token}_{args.pf_label}_grouped_uwc_alpha_"
        f"{args.uwc_method}_{args.alpha_backend}.json"
    )
    return PROJECT_ROOT / "artifacts" / "grouped_uwc_pf_qpe" / suffix


def _summary_rows(result: GroupedUWCComparisonResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in result.rows:
        rows.append(
            {
                "molecule": row.molecule,
                "method": row.method,
                "uwc_method": row.uwc_method,
                "uwc_objective": row.uwc_objective,
                "pf_label": row.pf_label,
                "order": row.order,
                "grouping_rule": row.grouping_rule,
                "num_groups": row.num_groups,
                "num_pauli_terms": row.num_pauli_terms,
                "alpha": row.alpha,
                "qpe_iteration_factor": row.qpe_iteration_factor,
                "step_pauli_rotations": row.step_pauli_rotations,
                "total_pauli_rotations": row.total_pauli_rotations,
                "step_rz_layers": row.step_rz_layers,
                "total_rz_layers": row.total_rz_layers,
                "cost_ratio_vs_grouped_baseline": row.cost_ratio_vs_grouped_baseline,
                "alpha_ratio_vs_grouped_baseline": row.alpha_ratio_vs_grouped_baseline,
                "step_cost_ratio_vs_grouped_baseline": (
                    row.step_cost_ratio_vs_grouped_baseline
                ),
                "alpha_backend": None
                if row.metadata is None
                else row.metadata.get("alpha_fit_backend"),
                "alpha_requested_backend": None
                if row.metadata is None
                else row.metadata.get("alpha_requested_backend"),
            }
        )
    return rows


def _print_summary(rows: list[dict[str, Any]]) -> None:
    print(
        "molecule method alpha alpha_backend num_groups "
        "step_pauli_rotations step_rz_layers cost_ratio alpha_ratio"
    )
    for row in rows:
        if row.get("status") == "failed":
            print(f"{row['molecule']} failed {row['error']}")
            continue
        print(
            f"{row['molecule']} {row['method']} {row['alpha']:.12e} "
            f"{row.get('alpha_backend')} {row['num_groups']} "
            f"{row['step_pauli_rotations']} {row['step_rz_layers']} "
            f"{row['cost_ratio_vs_grouped_baseline']:.8g} "
            f"{row['alpha_ratio_vs_grouped_baseline']:.8g}"
        )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.reuse_original_ground_state_for_uwc and args.solve_uwc_ground_state:
        parser.error(
            "--reuse-original-ground-state-for-uwc and --solve-uwc-ground-state "
            "are mutually exclusive."
        )
    try:
        molecule_types = _parse_molecule_types(args.molecule_types)
        _build_uwc_config(args)
    except (argparse.ArgumentTypeError, ValueError) as exc:
        parser.error(str(exc))

    reuse_ground: bool | None = None
    if args.reuse_original_ground_state_for_uwc:
        reuse_ground = True
    if args.solve_uwc_ground_state:
        reuse_ground = False

    summary_rows: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    failures = 0
    for molecule_type in molecule_types:
        try:
            uwc_config = _build_uwc_config(args)
            result = compare_grouped_uwc_pf_qpe(
                molecule_type,
                pf_label=args.pf_label,
                uwc_config=uwc_config,
                target_error=args.target_error,
                distance=args.distance,
                basis=args.basis,
                qpe_beta=args.qpe_beta,
                cost_metric=args.cost_metric,
                t_values=_parse_t_values(args.t_values),
                baseline_alpha_source=args.baseline_alpha_source,
                use_original_grouped_artifact=args.use_original_grouped_artifact,
                reuse_original_ground_state_for_uwc=reuse_ground,
                use_reference_rz_layers=not args.no_reference_rz_layer_regroup,
                alpha_backend=args.alpha_backend,
                alpha_gpu_ids=_parse_gpu_ids(args.alpha_gpu_ids),
                alpha_parallel_processes=args.alpha_parallel_processes,
                alpha_chunk_splits=args.alpha_chunk_splits,
                alpha_gpu_optimization_level=args.alpha_gpu_optimization_level,
                alpha_gpu_debug=args.alpha_gpu_debug,
            )
        except Exception as exc:
            failures += 1
            failure_row = {
                "molecule": f"H{int(molecule_type)}",
                "status": "failed",
                "error": str(exc),
                "alpha_backend": args.alpha_backend,
            }
            summary_rows.append(failure_row)
            print(f"H{int(molecule_type)} failed: {exc}", file=sys.stderr)
            if args.fail_fast:
                break
            continue
        comparisons.append(result.to_dict())
        summary_rows.extend(_summary_rows(result))

    output = {
        "config": {
            "molecule_types": list(molecule_types),
            "pf_label": args.pf_label,
            "target_error": args.target_error,
            "qpe_beta": args.qpe_beta,
            "cost_metric": args.cost_metric,
            "alpha_backend": args.alpha_backend,
            "alpha_gpu_ids": list(_parse_gpu_ids(args.alpha_gpu_ids)),
            "alpha_parallel_processes": args.alpha_parallel_processes,
            "alpha_chunk_splits": args.alpha_chunk_splits,
            "alpha_gpu_optimization_level": args.alpha_gpu_optimization_level,
            "uwc_config_template": {
                "method": args.uwc_method,
                "objective": args.uwc_objective,
                "target_ld": args.uwc_target_ld,
                "max_iterations": args.uwc_max_iterations,
                "seed": args.uwc_seed,
                "parameters_json": args.uwc_parameters_json,
                "simple_shift": args.uwc_simple_shift,
                "simple_shift_qubit": args.uwc_simple_shift_qubit,
                "coefficient_scale": args.uwc_coefficient_scale,
                "bliss_theta": args.uwc_bliss_theta,
                "target_particle_number": args.uwc_target_particle_number,
                "target_particle_number_inference": (
                    "grouped_reference_state"
                    if args.uwc_target_particle_number is None
                    and args.uwc_method in {"bliss", "orbital_optimization_bliss"}
                    else None
                ),
            },
        },
        "summary_rows": summary_rows,
        "comparisons": comparisons,
    }
    output_path = args.output if args.output is not None else _default_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    _print_summary(summary_rows)
    print(f"Saved H2-H6 grouped UWC alpha results to {output_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
