from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trotterlib.partial_randomized_pf import (  # noqa: E402
    analyze_partial_randomized_pf,
    save_kappa_sweep_csv,
    save_partial_randomized_result,
)


def _parse_pf_labels(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    return [label.strip() for label in raw.split(",") if label.strip()]


def _parse_ld_values(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    return [int(token.strip()) for token in raw.split(",") if token.strip()]


def _parse_kappa_grid(raw: str | None) -> list[float] | None:
    if raw is None:
        return None
    return [float(token.strip()) for token in raw.split(",") if token.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the simplified partially randomized PF scan with kappa-aware randomized prefactors.",
    )
    parser.add_argument("--molecule-type", type=int, required=True, help="H-chain size.")
    parser.add_argument(
        "--epsilon-total",
        type=float,
        required=True,
        help="Target total error epsilon in eps^2 = eps_qpe^2 + eps_trot^2.",
    )
    parser.add_argument("--distance", type=float, default=1.0)
    parser.add_argument(
        "--pf-labels",
        type=str,
        default=None,
        help="Comma-separated PF labels. Default uses the built-in scan list.",
    )
    parser.add_argument(
        "--ld-values",
        type=str,
        default=None,
        help="Comma-separated L_D values. Default scans the full range.",
    )
    parser.add_argument(
        "--ld-step",
        type=int,
        default=1,
        help="Stride for the default L_D sweep when --ld-values is omitted.",
    )
    parser.add_argument(
        "--kappa-mode",
        choices=("fixed", "optimize", "sweep"),
        default="optimize",
        help="Use kappa=const, optimize kappa, or sweep a kappa grid.",
    )
    parser.add_argument(
        "--kappa-value",
        type=float,
        default=2.0,
        help="Reference kappa value for --kappa-mode fixed.",
    )
    parser.add_argument(
        "--kappa-min",
        type=float,
        default=1.0,
        help="Lower bound for kappa optimization.",
    )
    parser.add_argument(
        "--kappa-max",
        type=float,
        default=32.0,
        help="Upper bound for kappa optimization.",
    )
    parser.add_argument(
        "--kappa-grid",
        type=str,
        default=None,
        help="Comma-separated kappa values for sensitivity mode. Default is 1,2,4,8,16,32.",
    )
    parser.add_argument(
        "--randomized-method",
        choices=("qdrift", "rte"),
        default="qdrift",
        help="Randomized-side method used in B0.",
    )
    parser.add_argument(
        "--g-rand",
        type=float,
        default=1.0,
        help="Randomized-side per-step cost proxy G_rand used in B0.",
    )
    parser.add_argument(
        "--matrix-free-backend",
        choices=("auto", "numba", "python"),
        default="auto",
        help="Ground-state matrix-free backend used for C_gs fits.",
    )
    parser.add_argument(
        "--matrix-free-threads",
        type=int,
        default=None,
        help="Number of numba threads for matrix-free matvec. Default auto-detects available CPUs. Use 0 for auto.",
    )
    parser.add_argument(
        "--ground-state-ncv",
        type=int,
        default=None,
        help="ARPACK ncv for ground-state eigsh. Lower values reduce memory use.",
    )
    parser.add_argument(
        "--export-kappa-sweep-csv",
        nargs="?",
        const="__AUTO__",
        default=None,
        help="In sweep mode, optionally export the kappa sensitivity table to CSV.",
    )
    parser.add_argument(
        "--random-prefactor",
        type=float,
        default=None,
        help="Deprecated legacy path: use a fixed external B instead of B(kappa).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. Defaults to artifacts/partial_randomized_pf/.",
    )
    return parser


def _default_output_path(args: argparse.Namespace) -> Path:
    suffix = f"H{args.molecule_type}_eps_{args.epsilon_total:.3e}_partial_randomized_pf_{args.kappa_mode}.json"
    return PROJECT_ROOT / "artifacts" / "partial_randomized_pf" / suffix


def _default_sweep_csv_path(json_path: Path) -> Path:
    return json_path.with_name(f"{json_path.stem}_kappa_sweep.csv")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    result = analyze_partial_randomized_pf(
        args.molecule_type,
        epsilon_total=args.epsilon_total,
        distance=args.distance,
        pf_labels=_parse_pf_labels(args.pf_labels),
        ld_values=_parse_ld_values(args.ld_values),
        ld_step=args.ld_step,
        random_prefactor=args.random_prefactor,
        kappa_mode=args.kappa_mode,
        kappa_value=args.kappa_value,
        kappa_min=args.kappa_min,
        kappa_max=args.kappa_max,
        kappa_grid=_parse_kappa_grid(args.kappa_grid),
        randomized_method=args.randomized_method,
        g_rand=args.g_rand,
        matrix_free_backend=args.matrix_free_backend,
        matrix_free_threads=args.matrix_free_threads,
        ground_state_ncv=args.ground_state_ncv,
    )

    output_path = args.output if args.output is not None else _default_output_path(args)
    save_partial_randomized_result(result, output_path)

    csv_path: Path | None = None
    if args.export_kappa_sweep_csv is not None:
        if args.kappa_mode != "sweep":
            parser.error("--export-kappa-sweep-csv can only be used with --kappa-mode sweep.")
        csv_path = (
            _default_sweep_csv_path(output_path)
            if args.export_kappa_sweep_csv == "__AUTO__"
            else Path(args.export_kappa_sweep_csv)
        )
        save_kappa_sweep_csv(result, csv_path)

    print("Best candidate")
    print(json.dumps(result.to_dict()["best"], indent=2))
    print(f"Saved full result to {output_path}")
    if csv_path is not None:
        print(f"Saved kappa sweep CSV to {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
