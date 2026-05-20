from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trotterlib.config import BETA, TARGET_ERROR  # noqa: E402
from trotterlib.grouped_uwc_comparison import (  # noqa: E402
    compare_grouped_uwc_pf_qpe,
    save_grouped_uwc_comparison,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare grouped baseline PF+QPE against UWC-applied grouped PF+QPE.",
    )
    parser.add_argument("--molecule-type", type=int, required=True)
    parser.add_argument("--pf-label", type=str, required=True)
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
        help=(
            "Use generic qubit-support RZ layering for UWC instead of the "
            "existing grouped reference RZ-layer rule plus UWC deltas."
        ),
    )
    parser.add_argument(
        "--alpha-backend",
        choices=("cpu", "gpu", "auto"),
        default="cpu",
        help="Backend used when fitting grouped Trotter alpha.",
    )
    parser.add_argument(
        "--alpha-gpu-ids",
        type=str,
        default="0",
        help="Comma-separated GPU ids used for alpha fitting.",
    )
    parser.add_argument(
        "--alpha-parallel-processes",
        type=int,
        default=None,
        help="Number of parallel time-grid workers for GPU alpha fitting.",
    )
    parser.add_argument("--alpha-chunk-splits", type=int, default=1)
    parser.add_argument("--alpha-gpu-optimization-level", type=int, default=0)
    parser.add_argument("--alpha-gpu-debug", action="store_true")
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
        default="none",
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
    parser.add_argument("--uwc-simple-shift", type=float, default=None)
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
    parser.add_argument(
        "--uwc-max-sector-dimension-for-check",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )
    return parser


def _build_uwc_config(args: argparse.Namespace) -> UWCConfig:
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
    suffix = (
        f"H{args.molecule_type}_{args.pf_label}_eps_{args.target_error:.3e}"
        f"_grouped_uwc_{args.uwc_method}_{args.cost_metric}.json"
    )
    return PROJECT_ROOT / "artifacts" / "grouped_uwc_pf_qpe" / suffix


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.reuse_original_ground_state_for_uwc and args.solve_uwc_ground_state:
        parser.error(
            "--reuse-original-ground-state-for-uwc and --solve-uwc-ground-state "
            "are mutually exclusive."
        )
    try:
        uwc_config = _build_uwc_config(args)
    except (argparse.ArgumentTypeError, ValueError) as exc:
        parser.error(str(exc))

    reuse_ground: bool | None = None
    if args.reuse_original_ground_state_for_uwc:
        reuse_ground = True
    if args.solve_uwc_ground_state:
        reuse_ground = False

    result = compare_grouped_uwc_pf_qpe(
        args.molecule_type,
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

    output_path = args.output if args.output is not None else _default_output_path(args)
    save_grouped_uwc_comparison(result, output_path)
    print(json.dumps(result.to_dict()["rows"], indent=2))
    print(f"Saved grouped UWC comparison to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
