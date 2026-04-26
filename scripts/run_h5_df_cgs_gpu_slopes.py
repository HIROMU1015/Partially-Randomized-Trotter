from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trotterlib.df_hamiltonian import build_df_h_d_from_molecule  # noqa: E402
from trotterlib.df_partial_randomized_pf import (  # noqa: E402
    get_or_compute_cached_df_cgs_fit,
    load_df_cgs_json_cache,
    rank_df_fragments,
    split_df_hamiltonian_by_ld,
)


def _parse_ld_values(value: str, n_blocks: int) -> tuple[int, ...]:
    if value == "all":
        return tuple(range(n_blocks + 1))
    return tuple(int(item) for item in value.split(",") if item.strip())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit DF-native H5 C_gs,D on GPU and print slopes."
    )
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--ld-values", default="all")
    parser.add_argument("--chunk-splits", type=int, default=1)
    parser.add_argument("--optimization-level", type=int, default=0)
    parser.add_argument("--processes", type=int, default=None)
    parser.add_argument(
        "--no-parameterized-template",
        action="store_true",
        help="Build and transpile each t circuit separately instead of reusing a template.",
    )
    parser.add_argument(
        "--no-ground-state-cache",
        action="store_true",
        help="Solve the DF H_D ground state every time instead of using the cache.",
    )
    parser.add_argument(
        "--no-parallel-times",
        action="store_true",
        help="Run t_values sequentially even when multiple GPU ids are provided.",
    )
    parser.add_argument("--ground-state-tol", type=float, default=1e-10)
    parser.add_argument("--matrix-free-threads", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    gpu_ids = tuple(item.strip() for item in args.gpu_ids.split(",") if item.strip())
    pf_labels = ("2nd", "4th", "8th(Morales)")
    hamiltonian, sector = build_df_h_d_from_molecule(5)
    ranked = rank_df_fragments(hamiltonian)
    ld_values = _parse_ld_values(args.ld_values, hamiltonian.n_blocks)
    cache_document = load_df_cgs_json_cache()

    rows = []
    for ld in ld_values:
        partition = split_df_hamiltonian_by_ld(
            hamiltonian,
            ld,
            ranked_fragments=ranked,
        )
        for pf_label in pf_labels:
            result = get_or_compute_cached_df_cgs_fit(
                hamiltonian=hamiltonian,
                sector=sector,
                partition=partition,
                pf_label=pf_label,
                cache_document=cache_document,
                evolution_backend="gpu",
                gpu_ids=gpu_ids,
                chunk_splits=args.chunk_splits,
                optimization_level=args.optimization_level,
                parallel_times=not args.no_parallel_times,
                processes=args.processes,
                use_parameterized_template=not args.no_parameterized_template,
                use_ground_state_cache=not args.no_ground_state_cache,
                ground_state_tol=args.ground_state_tol,
                matrix_free_threads=args.matrix_free_threads,
                debug=args.debug,
            )
            row = {
                "molecule_type": 5,
                "representation_type": result.representation_type,
                "df_rank_actual": result.df_rank_actual,
                "pf_label": result.pf_label,
                "order": result.order,
                "ld": result.ld,
                "lambda_r": result.lambda_r,
                "c_gs_d": result.coeff,
                "fit_slope": result.fit_slope,
                "fit_coeff": result.fit_coeff,
                "fixed_order_coeff": result.fit_coeff_fixed_order,
                "t_values": list(result.t_values),
                "perturbation_errors": list(result.perturbation_errors),
                "evolution_backend": result.evolution_backend,
                "gpu_ids": list(result.gpu_ids),
                "chunk_splits": result.chunk_splits,
                "use_parameterized_template": result.metadata.get(
                    "use_parameterized_template"
                ),
                "ground_state_cache": result.metadata.get("ground_state_cache"),
                "df_step_cost": result.metadata.get("df_step_cost"),
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    out_path = (
        PROJECT_ROOT
        / "artifacts"
        / "partial_randomized_pf"
        / "H5_df_cgs_gpu_slopes.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
