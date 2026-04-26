from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trotterlib.config import (  # noqa: E402
    PARTIAL_RANDOMIZED_DEFAULT_G_RAND,
    PARTIAL_RANDOMIZED_DEFAULT_KAPPA,
    PARTIAL_RANDOMIZED_DEFAULT_RANDOMIZED_METHOD,
    PARTIAL_RANDOMIZED_KAPPA_MAX,
    PARTIAL_RANDOMIZED_KAPPA_MIN,
)
from trotterlib.df_screening_cost import (  # noqa: E402
    DEFAULT_DF_CGS_COST_TABLE,
    DEFAULT_DF_SCREENING_COST_OUTPUT,
    optimize_df_screening_cost,
    save_df_screening_cost_result,
)


def _parse_pf_labels(raw: str | None) -> tuple[str, ...] | None:
    if raw is None:
        return None
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate DF reduced-screening cost minimization using anchor Cgs values."
    )
    parser.add_argument("--cgs-table", type=Path, default=DEFAULT_DF_CGS_COST_TABLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_DF_SCREENING_COST_OUTPUT)
    parser.add_argument("--epsilon-total", type=float, default=1e-4)
    parser.add_argument("--molecule-min", type=int, default=None)
    parser.add_argument("--molecule-max", type=int, default=None)
    parser.add_argument("--pf-labels", type=str, default=None)
    parser.add_argument("--kappa-mode", choices=("fixed", "optimize"), default="optimize")
    parser.add_argument("--kappa-value", type=float, default=PARTIAL_RANDOMIZED_DEFAULT_KAPPA)
    parser.add_argument("--kappa-min", type=float, default=PARTIAL_RANDOMIZED_KAPPA_MIN)
    parser.add_argument("--kappa-max", type=float, default=PARTIAL_RANDOMIZED_KAPPA_MAX)
    parser.add_argument(
        "--randomized-method",
        choices=("qdrift", "rte"),
        default=PARTIAL_RANDOMIZED_DEFAULT_RANDOMIZED_METHOD,
    )
    parser.add_argument("--g-rand", type=float, default=PARTIAL_RANDOMIZED_DEFAULT_G_RAND)
    args = parser.parse_args()

    result = optimize_df_screening_cost(
        cgs_table_path=args.cgs_table,
        epsilon_total=float(args.epsilon_total),
        molecule_min=args.molecule_min,
        molecule_max=args.molecule_max,
        pf_labels=_parse_pf_labels(args.pf_labels),
        kappa_mode=args.kappa_mode,
        kappa_value=float(args.kappa_value),
        kappa_min=float(args.kappa_min),
        kappa_max=float(args.kappa_max),
        randomized_method=args.randomized_method,
        g_rand=float(args.g_rand),
        progress_callback=lambda best: print(json.dumps(best, sort_keys=True), flush=True),
    )
    save_df_screening_cost_result(result, args.output)
    print(f"wrote {args.output} ({len(result['candidates'])} candidates)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
