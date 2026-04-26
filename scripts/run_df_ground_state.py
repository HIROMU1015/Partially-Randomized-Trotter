from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trotterlib.df_hamiltonian import (  # noqa: E402
    build_df_h_d_from_molecule,
    solve_df_ground_state,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Solve a DF-based deterministic H_D ground state matrix-free.",
    )
    parser.add_argument("--molecule-type", type=int, default=2)
    parser.add_argument("--distance", type=float, default=None)
    parser.add_argument("--df-rank", type=int, default=None)
    parser.add_argument("--df-tol", type=float, default=None)
    parser.add_argument(
        "--matrix-free-backend",
        choices=("auto", "numba", "python"),
        default="auto",
    )
    parser.add_argument(
        "--matrix-free-threads",
        type=int,
        default=None,
        help="Number of numba threads for DF matrix-free matvec. Use 0 for auto.",
    )
    parser.add_argument(
        "--matrix-free-block-chunk-size",
        type=int,
        default=None,
        help="Optional DF block chunk size for fused numba matvec.",
    )
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--maxiter", type=int, default=None)
    parser.add_argument("--ncv", type=int, default=None)
    parser.add_argument("--solver", choices=("eigsh", "lobpcg"), default="eigsh")
    parser.add_argument("--lobpcg-block-size", type=int, default=4)
    parser.add_argument(
        "--no-preconditioner",
        action="store_true",
        help="Disable the diagonal preconditioner for LOBPCG.",
    )
    parser.add_argument(
        "--no-expand-state",
        action="store_true",
        help="Do not materialize the full 2^N state vector in the result object.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    hamiltonian, sector = build_df_h_d_from_molecule(
        args.molecule_type,
        distance=args.distance,
        df_rank=args.df_rank,
        df_tol=args.df_tol,
    )
    result = solve_df_ground_state(
        hamiltonian,
        sector,
        matrix_free_backend=args.matrix_free_backend,
        matrix_free_threads=args.matrix_free_threads,
        matrix_free_block_chunk_size=args.matrix_free_block_chunk_size,
        solver=args.solver,
        use_preconditioner=not args.no_preconditioner,
        lobpcg_block_size=args.lobpcg_block_size,
        tol=args.tol,
        maxiter=args.maxiter,
        ncv=args.ncv,
        expand_state=not args.no_expand_state,
    )
    print(
        json.dumps(
            {
                "energy": result.energy,
                "converged": result.converged,
                "residual_norm": result.residual_norm,
                "matvec_count": result.matvec_count,
                "solver": result.solver,
                "message": result.message,
                "elapsed_s": result.elapsed_s,
                "n_qubits": sector.n_qubits,
                "sector_dim": sector.dimension,
                "n_electrons": sector.n_electrons,
                "nelec_alpha": sector.nelec_alpha,
                "nelec_beta": sector.nelec_beta,
                "df_rank_actual": hamiltonian.n_blocks,
                "matrix_free_backend": args.matrix_free_backend,
                "matrix_free_threads": args.matrix_free_threads,
                "matrix_free_block_chunk_size": args.matrix_free_block_chunk_size,
                "metadata": hamiltonian.metadata,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
