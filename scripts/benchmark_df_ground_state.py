from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trotterlib.df_hamiltonian import (  # noqa: E402
    build_df_h_d_from_molecule,
    solve_df_ground_state,
)


def _parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_optional_int_list(raw: str) -> list[int | None]:
    values: list[int | None] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(None if item.lower() in {"none", "null", "auto"} else int(item))
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark DF ground-state solver settings.")
    parser.add_argument("--molecule-type", type=int, default=10)
    parser.add_argument("--distance", type=float, default=None)
    parser.add_argument("--df-rank", type=int, default=None)
    parser.add_argument("--df-tol", type=float, default=None)
    parser.add_argument("--threads", default="32,64,96,128")
    parser.add_argument("--tols", default="1e-8,1e-10")
    parser.add_argument("--ncvs", default="none")
    parser.add_argument(
        "--block-chunk-sizes",
        default="none",
        help="Comma-separated DF block chunk sizes; use none for full fused/auto behavior.",
    )
    parser.add_argument("--maxiter", type=int, default=None)
    parser.add_argument("--backend", choices=("auto", "numba", "python"), default="numba")
    parser.add_argument("--solver", choices=("eigsh", "lobpcg"), default="eigsh")
    parser.add_argument("--lobpcg-block-size", type=int, default=4)
    parser.add_argument("--no-preconditioner", action="store_true")
    parser.add_argument(
        "--warm-start",
        action="store_true",
        help="Feed each result vector into the next solve. Disabled by default for fair sweeps.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    build_start = time.perf_counter()
    hamiltonian, sector = build_df_h_d_from_molecule(
        args.molecule_type,
        distance=args.distance,
        df_rank=args.df_rank,
        df_tol=args.df_tol,
    )
    print(
        json.dumps(
            {
                "event": "built",
                "elapsed_s": time.perf_counter() - build_start,
                "molecule_type": args.molecule_type,
                "n_qubits": sector.n_qubits,
                "sector_dim": sector.dimension,
                "df_rank_actual": hamiltonian.n_blocks,
                "metadata": hamiltonian.metadata,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    previous_sector_state = None
    for tol in _parse_float_list(args.tols):
        for ncv in _parse_optional_int_list(args.ncvs):
            for block_chunk_size in _parse_optional_int_list(args.block_chunk_sizes):
                for threads in _parse_int_list(args.threads):
                    result = solve_df_ground_state(
                        hamiltonian,
                        sector,
                        matrix_free_backend=args.backend,
                        matrix_free_threads=threads,
                        matrix_free_block_chunk_size=block_chunk_size,
                        solver=args.solver,
                        use_preconditioner=not args.no_preconditioner,
                        lobpcg_block_size=args.lobpcg_block_size,
                        tol=tol,
                        maxiter=args.maxiter,
                        ncv=ncv,
                        v0=previous_sector_state,
                        expand_state=False,
                    )
                    if args.warm_start:
                        previous_sector_state = result.sector_state_vector
                    print(
                        json.dumps(
                            {
                                "event": "result",
                                "threads": threads,
                                "tol": tol,
                                "ncv": ncv,
                                "block_chunk_size": block_chunk_size,
                                "elapsed_s": result.elapsed_s,
                                "energy": result.energy,
                                "residual_norm": result.residual_norm,
                                "matvec_count": result.matvec_count,
                                "converged": result.converged,
                                "solver": result.solver,
                                "message": result.message,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
