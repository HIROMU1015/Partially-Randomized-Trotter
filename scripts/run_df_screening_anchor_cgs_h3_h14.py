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


def _parse_gpu_ids(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _parse_pf_labels(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compute DF screening-anchor C_gs,D for H3-H14 at "
            "L_anchor=floor(DF rank/2)."
        )
    )
    parser.add_argument("--molecule-min", type=int, default=3)
    parser.add_argument("--molecule-max", type=int, default=14)
    parser.add_argument(
        "--pf-labels",
        default="2nd,4th,8th(Morales)",
        help="Comma-separated PF labels to compute.",
    )
    parser.add_argument(
        "--output-stem",
        default="H3_H14_df_screening_anchor_cgs",
        help="Output filename stem under artifacts/partial_randomized_pf.",
    )
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--processes", type=int, default=None)
    parser.add_argument("--optimization-level", type=int, default=0)
    parser.add_argument("--ground-state-tol", type=float, default=1e-10)
    parser.add_argument("--matrix-free-threads", type=int, default=None)
    parser.add_argument(
        "--cache-path",
        type=Path,
        help="Optional DF Cgs cache path for this run.",
    )
    parser.add_argument(
        "--no-parameterized-template",
        action="store_true",
        help="Build and transpile one concrete circuit per t instead of one parameterized template.",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    gpu_ids = _parse_gpu_ids(args.gpu_ids)
    pf_labels = _parse_pf_labels(args.pf_labels)
    out_dir = PROJECT_ROOT / "artifacts" / "partial_randomized_pf"
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / f"{args.output_stem}.json"
    partial_path = out_dir / f"{args.output_stem}.partial.json"

    cache_document = (
        load_df_cgs_json_cache(args.cache_path)
        if args.cache_path is not None
        else load_df_cgs_json_cache()
    )
    cache_kwargs = (
        {"cache_path": args.cache_path}
        if args.cache_path is not None
        else {}
    )
    rows: list[dict[str, object]] = []
    if partial_path.exists():
        try:
            loaded = json.loads(partial_path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                rows = [dict(item) for item in loaded if isinstance(item, dict)]
        except json.JSONDecodeError:
            rows = []
    done = {
        (int(row["molecule_type"]), str(row["pf_label"]))
        for row in rows
        if "molecule_type" in row and "pf_label" in row
    }

    for molecule_type in range(int(args.molecule_min), int(args.molecule_max) + 1):
        hamiltonian, sector = build_df_h_d_from_molecule(molecule_type)
        ranked = rank_df_fragments(hamiltonian)
        ld_anchor = hamiltonian.n_blocks // 2
        partition = split_df_hamiltonian_by_ld(
            hamiltonian,
            ld_anchor,
            ranked_fragments=ranked,
        )
        print(
            json.dumps(
                {
                    "event": "molecule_start",
                    "molecule_type": molecule_type,
                    "df_rank_actual": hamiltonian.n_blocks,
                    "ld_anchor": ld_anchor,
                    "gpu_ids": list(gpu_ids),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        for pf_label in pf_labels:
            if (molecule_type, pf_label) in done:
                print(
                    json.dumps(
                        {
                            "event": "skip_existing",
                            "molecule_type": molecule_type,
                            "pf_label": pf_label,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                continue
            result = get_or_compute_cached_df_cgs_fit(
                hamiltonian=hamiltonian,
                sector=sector,
                partition=partition,
                pf_label=pf_label,
                cache_document=cache_document,
                **cache_kwargs,
                evolution_backend="gpu",
                gpu_ids=gpu_ids,
                optimization_level=int(args.optimization_level),
                parallel_times=True,
                processes=args.processes,
                use_parameterized_template=not bool(args.no_parameterized_template),
                use_ground_state_cache=True,
                ground_state_tol=float(args.ground_state_tol),
                matrix_free_threads=args.matrix_free_threads,
                debug=bool(args.debug),
            )
            df_step_cost = result.metadata.get("df_step_cost", {})
            ground_state_cache = result.metadata.get("ground_state_cache", {})
            row = {
                "molecule_type": molecule_type,
                "df_rank_actual": result.df_rank_actual,
                "df_rank_requested": result.df_rank_requested,
                "df_tol_requested": result.df_tol_requested,
                "ld_anchor": ld_anchor,
                "ld": result.ld,
                "lambda_r": result.lambda_r,
                "pf_label": result.pf_label,
                "order": result.order,
                "c_gs_d": result.coeff,
                "fit_slope": result.fit_slope,
                "fit_coeff": result.fit_coeff,
                "fixed_order_coeff": result.fit_coeff_fixed_order,
                "t_values": list(result.t_values),
                "perturbation_errors": list(result.perturbation_errors),
                "gpu_ids": list(result.gpu_ids),
                "parallel_times": result.parallel_times,
                "processes": result.processes,
                "use_parameterized_template": result.metadata.get(
                    "use_parameterized_template"
                ),
                "ground_state_cache": ground_state_cache,
                "df_step_cost": df_step_cost,
                "total_ref_rz_depth": (
                    df_step_cost.get("total_ref_rz_depth")
                    if isinstance(df_step_cost, dict)
                    else None
                ),
                "template_transpile_s": (
                    result.metadata.get("parameterized_template_profile") or {}
                ).get("transpile_s"),
            }
            rows.append(row)
            partial_path.write_text(
                json.dumps(rows, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            print(json.dumps(row, sort_keys=True), flush=True)

    final_path.write_text(
        json.dumps(rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"wrote {final_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
