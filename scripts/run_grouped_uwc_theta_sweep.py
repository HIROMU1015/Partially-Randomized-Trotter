from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trotterlib.config import BETA, TARGET_ERROR  # noqa: E402
from trotterlib.grouped_uwc_theta_sweep import (  # noqa: E402
    DEFAULT_THETA_SWEEP_VALUES,
    run_grouped_uwc_theta_sweep,
    save_grouped_uwc_theta_sweep,
)


def _parse_float_list(raw: str) -> tuple[float, ...]:
    values = tuple(float(token.strip()) for token in raw.split(",") if token.strip())
    if not values:
        raise argparse.ArgumentTypeError("value list must not be empty.")
    return values


def _parse_molecule_token(token: str) -> int:
    token = token.strip()
    if token.lower().startswith("h"):
        token = token[1:]
    return int(token)


def _parse_int_ranges(raw: str | None) -> tuple[int, ...]:
    if raw is None:
        return tuple(range(2, 7))
    values: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            start = _parse_molecule_token(left)
            stop = _parse_molecule_token(right)
            if stop < start:
                raise argparse.ArgumentTypeError("molecule ranges must be increasing.")
            values.extend(range(start, stop + 1))
        else:
            values.append(_parse_molecule_token(token))
    if not values:
        raise argparse.ArgumentTypeError("--molecule-types must not be empty.")
    return tuple(dict.fromkeys(values))


def _parse_pf_labels(raw: str) -> tuple[str, ...]:
    values = tuple(token.strip() for token in raw.split(",") if token.strip())
    if not values:
        raise argparse.ArgumentTypeError("--pf-labels must not be empty.")
    return values


def _parse_gpu_ids(raw: str) -> tuple[str, ...]:
    values = tuple(token.strip() for token in raw.split(",") if token.strip())
    return values or ("0",)


def _parse_t_values(raw: str | None) -> tuple[float, ...] | None:
    if raw is None:
        return None
    return _parse_float_list(raw)


def _default_gpu_ids() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    return visible if visible else "0"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a standard BLISS theta sweep for grouped UWC PF+QPE costs.",
    )
    parser.add_argument(
        "--molecule-types",
        type=str,
        default=None,
        help="Comma-separated molecule sizes or ranges. Default: 2-6.",
    )
    parser.add_argument("--pf-labels", type=str, default="2nd")
    parser.add_argument(
        "--theta-values",
        type=str,
        default=",".join(str(value) for value in DEFAULT_THETA_SWEEP_VALUES),
    )
    parser.add_argument("--power", choices=("linear", "quadratic"), default="quadratic")
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
        "--baseline-alpha-source",
        choices=("artifact", "fit"),
        default="artifact",
    )
    parser.add_argument("--use-original-grouped-artifact", action="store_true")
    parser.add_argument(
        "--alpha-backend",
        choices=("cpu", "gpu", "auto"),
        default="cpu",
    )
    parser.add_argument("--alpha-gpu-ids", type=str, default=_default_gpu_ids())
    parser.add_argument("--alpha-parallel-processes", type=int, default=None)
    parser.add_argument("--alpha-chunk-splits", type=int, default=1)
    parser.add_argument("--alpha-gpu-optimization-level", type=int, default=0)
    parser.add_argument("--alpha-gpu-debug", action="store_true")
    parser.add_argument(
        "--t-values",
        type=str,
        default=None,
        help="Optional comma-separated alpha fit time grid.",
    )
    parser.add_argument(
        "--no-reference-rz-layer-regroup",
        action="store_true",
        help="Use generic support-layer counting instead of reference grouped RZ layers.",
    )
    parser.add_argument("--uwc-objective", choices=("l1_norm", "lambda_r", "estimated_total_cost"), default="l1_norm")
    parser.add_argument("--uwc-target-ld", type=int, default=None)
    parser.add_argument("--uwc-target-particle-number", type=int, default=None)
    parser.add_argument("--uwc-sector-energy-tol", type=float, default=1e-8)
    parser.add_argument(
        "--uwc-sector-energy-check",
        choices=("warn", "error", "off"),
        default="warn",
    )
    parser.add_argument("--uwc-max-sector-dimension-for-check", type=int, default=256)
    parser.add_argument(
        "--no-baseline-rows",
        action="store_true",
        help="Store only UWC rows in the flat rows table.",
    )
    parser.add_argument("--theta-zero-tol", type=float, default=1e-8)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--csv-output", type=Path, default=None)
    return parser


def _default_output_path(args: argparse.Namespace) -> Path:
    molecules = (args.molecule_types or "H2-H6").replace(",", "_").replace("-", "_to_")
    pf = args.pf_labels.replace(",", "_").replace("/", "_")
    suffix = (
        f"theta_sweep_{molecules}_{pf}_bliss_{args.power}_"
        f"{args.cost_metric}_{args.baseline_alpha_source}_{args.alpha_backend}.json"
    )
    return PROJECT_ROOT / "artifacts" / "grouped_uwc_pf_qpe" / suffix


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        molecule_types = _parse_int_ranges(args.molecule_types)
        pf_labels = _parse_pf_labels(args.pf_labels)
        theta_values = _parse_float_list(args.theta_values)
        t_values = _parse_t_values(args.t_values)
    except (argparse.ArgumentTypeError, ValueError) as exc:
        parser.error(str(exc))

    result = run_grouped_uwc_theta_sweep(
        molecule_types,
        pf_labels=pf_labels,
        theta_values=theta_values,
        power=args.power,
        target_error=args.target_error,
        distance=args.distance,
        basis=args.basis,
        qpe_beta=args.qpe_beta,
        cost_metric=args.cost_metric,
        t_values=t_values,
        baseline_alpha_source=args.baseline_alpha_source,
        use_original_grouped_artifact=args.use_original_grouped_artifact,
        use_reference_rz_layers=not args.no_reference_rz_layer_regroup,
        alpha_backend=args.alpha_backend,
        alpha_gpu_ids=_parse_gpu_ids(args.alpha_gpu_ids),
        alpha_parallel_processes=args.alpha_parallel_processes,
        alpha_chunk_splits=args.alpha_chunk_splits,
        alpha_gpu_optimization_level=args.alpha_gpu_optimization_level,
        alpha_gpu_debug=args.alpha_gpu_debug,
        uwc_objective=args.uwc_objective,
        uwc_target_ld=args.uwc_target_ld,
        uwc_target_particle_number=args.uwc_target_particle_number,
        uwc_sector_energy_tolerance=args.uwc_sector_energy_tol,
        uwc_sector_energy_check=args.uwc_sector_energy_check,
        uwc_max_sector_dimension_for_check=args.uwc_max_sector_dimension_for_check,
        include_baseline_rows=not args.no_baseline_rows,
        theta_zero_tolerance=args.theta_zero_tol,
    )

    output_path = args.output if args.output is not None else _default_output_path(args)
    json_path, csv_path = save_grouped_uwc_theta_sweep(
        result,
        output_path,
        csv_path=args.csv_output,
    )
    print(json.dumps(result.to_dict()["summary"], indent=2))
    print(f"Saved theta sweep JSON to {json_path}")
    print(f"Saved theta sweep CSV to {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
