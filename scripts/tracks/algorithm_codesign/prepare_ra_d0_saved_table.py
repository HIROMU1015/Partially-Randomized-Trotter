#!/usr/bin/env python3
"""Generate static RA-D0 preparation only. No optimization command exists."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"src"))
from trottertracks.algorithm_codesign.ra_d0.table import extract_saved_table
from trottertracks.algorithm_codesign.ra_d0.grid import build_grid
from trottertracks.algorithm_codesign.ra_d0.semantics import audit_semantics
from trottertracks.algorithm_codesign.ra_d0.exact import log_interval
from trottertracks.algorithm_codesign.ra_d0.lp import build_lp

INPUT = "artifacts/track_b_rte_reallocation_r1_result/2026-10-06/v1/result.json"
OUT = "artifacts/track_b_ra_d0_preparation/2026-10-06"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-static-artifacts", action="store_true", required=True)
    parser.parse_args()
    table = extract_saved_table((ROOT/INPUT).read_bytes())
    grid = build_grid(table)
    semantics = audit_semantics(table)
    models = {}
    for xs, data in table["tables"].items():
        n = grid["grids"][xs]["anchor_shots"][0]
        models[xs] = {}
        for baseline, robust in (("B2", False), ("B3", True)):
            lp = build_lp(data, n, "T", log_interval(10560)[1], baseline, robust=robust)
            models[xs][baseline] = {"variables": len(lp.c), "inequalities": len(lp.A),
                                   "equalities": len(lp.H), "domain": lp.domain,
                                   "solved": False, "anchor_n_for_schema_only": n}
    num_points = sum(len(g["points"]) for g in grid["grids"].values())
    summary = {"status": "SOURCE_PREPARATION_REQUIRES_MINIMAL_CONTRACT_AMENDMENTS",
               "candidate_columns": {x: v["distinct_columns"] for x, v in table["tables"].items()},
               "sign_pairs_checked": 18, "grid_points": num_points,
               "max_paired_queries": 300*num_points,
               "max_LP_calls_including_B2_minima": 603*num_points,
               "LP_schema_checks": models,
               "budget_values_and_witnesses": "NOT_ACQUIRED",
               "registered_optimization_calls": 0, "synthesis_calls": 0,
               "science_runs": 0, "sampler_or_circuit_calls": 0,
               "R1_classification_changed": False, "RUN_READY": False,
               "development_execution_authorized": False,
               "mandatory_STOP": True}
    artifacts = {"candidate_table_v1.json": table, "shot_grid_query_recipe_v1.json": grid,
                 "semantic_audit_v1.json": semantics, "static_preparation_summary_v1.json": summary}
    destination = ROOT/OUT
    destination.mkdir(parents=True, exist_ok=True)
    for name, data in artifacts.items():
        # Explicitly refuse replacing any saved preparation output.
        with (destination/name).open("x") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
