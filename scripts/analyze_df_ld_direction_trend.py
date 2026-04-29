from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trotterlib.partial_randomized_pf import optimize_error_budget_and_kappa  # noqa: E402


def _load_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [dict(item) for item in data if isinstance(item, dict)]
    if isinstance(data, dict) and isinstance(data.get("entries"), list):
        return [dict(item) for item in data["entries"] if isinstance(item, dict)]
    raise ValueError(f"Unsupported input JSON: {path}")


def _row_key(row: dict[str, Any]) -> tuple[int, str, int]:
    return (int(row["molecule_type"]), str(row["pf_label"]), int(row["ld"]))


def _cost_for_row(row: dict[str, Any], *, epsilon_total: float) -> dict[str, Any]:
    step_cost = int(row["total_ref_rz_depth"])
    budget = optimize_error_budget_and_kappa(
        epsilon_total=float(epsilon_total),
        order=int(row["order"]),
        deterministic_step_cost_value=step_cost,
        c_gs=float(row["c_gs_d"]),
        lambda_r=float(row["lambda_r"]),
        kappa_mode="optimize",
    )
    return {
        "molecule_type": int(row["molecule_type"]),
        "pf_label": str(row["pf_label"]),
        "ld": int(row["ld"]),
        "order": int(row["order"]),
        "lambda_r": float(row["lambda_r"]),
        "c_gs_d": float(row["c_gs_d"]),
        "total_ref_rz_depth": step_cost,
        "q_opt": budget.q_ratio,
        "eps_qpe_opt": budget.eps_qpe,
        "eps_trot_opt": budget.eps_trot,
        "kappa_opt": budget.kappa,
        "b_opt": budget.b_value,
        "g_det": budget.g_det,
        "g_rand": budget.g_rand,
        "g_total": budget.g_total,
        "boundary_hit_q": budget.boundary_hit_q,
        "boundary_hit_kappa": budget.boundary_hit_kappa,
        "source_kind": str(row.get("source_kind", "unknown")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recompute explicit-LD DF cost trends after direction refinement."
    )
    parser.add_argument("--decision-file", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epsilon-total", type=float, default=1e-4)
    args = parser.parse_args()

    rows_by_key: dict[tuple[int, str, int], dict[str, Any]] = {}
    for input_path in args.input:
        for row in _load_rows(input_path):
            if not all(key in row for key in ("molecule_type", "pf_label", "ld")):
                continue
            if row.get("c_gs_d") is None or row.get("total_ref_rz_depth") is None:
                continue
            rows_by_key[_row_key(row)] = row

    decision_doc = json.loads(args.decision_file.read_text(encoding="utf-8"))
    decisions_out: list[dict[str, Any]] = []
    for raw in decision_doc.get("decisions", []):
        decision = dict(raw)
        molecule_type = int(decision["molecule_type"])
        pf_label = str(decision["pf_label"])
        lds = {int(cost["ld"]) for cost in decision.get("costs", [])}
        if decision.get("extra_ld") is not None:
            lds.add(int(decision["extra_ld"]))
        costs = []
        missing = []
        for ld in sorted(lds):
            row = rows_by_key.get((molecule_type, pf_label, ld))
            if row is None:
                missing.append(ld)
                continue
            costs.append(_cost_for_row(row, epsilon_total=float(args.epsilon_total)))
        best = min(costs, key=lambda item: float(item["g_total"])) if costs else None
        decision["post_extension_costs"] = costs
        decision["post_extension_missing_ld"] = missing
        decision["post_extension_best"] = best
        if decision.get("extra_ld") is not None:
            decision["extra_ld_available"] = int(decision["extra_ld"]) not in missing
        decisions_out.append(decision)

    out_doc = {
        "schema_version": 1,
        "epsilon_total": float(args.epsilon_total),
        "decision_file": str(args.decision_file),
        "inputs": [str(path) for path in args.input],
        "num_decisions": len(decisions_out),
        "decisions": decisions_out,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out_doc, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
