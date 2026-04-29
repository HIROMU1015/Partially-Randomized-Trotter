from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

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


ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "partial_randomized_pf"
DEFAULT_OUTPUT_DIR = ARTIFACT_DIR / "df_cgs" / "refinement"


def _parse_csv(raw: str | None) -> tuple[str, ...] | None:
    if raw is None:
        return None
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_entries(document: Any) -> list[dict[str, Any]]:
    if isinstance(document, list):
        return [dict(item) for item in document if isinstance(item, dict)]
    if isinstance(document, dict) and isinstance(document.get("entries"), list):
        return [dict(item) for item in document["entries"] if isinstance(item, dict)]
    return []


def _load_existing_cgs_keys(paths: tuple[Path, ...]) -> set[tuple[int, str, int]]:
    keys: set[tuple[int, str, int]] = set()
    for path in paths:
        if not path.exists():
            continue
        for row in _iter_entries(_load_json(path)):
            if "molecule_type" not in row or "pf_label" not in row or "ld" not in row:
                continue
            if row.get("c_gs_d") is None and row.get("coeff") is None:
                continue
            keys.add((int(row["molecule_type"]), str(row["pf_label"]), int(row["ld"])))
    return keys


def _candidate_is_finite(candidate: dict[str, Any]) -> bool:
    value = candidate.get("g_total")
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _best_ld_by_system_pf(
    screening_result_path: Path,
    *,
    molecule_min: int | None,
    molecule_max: int | None,
    pf_labels: tuple[str, ...] | None,
) -> dict[tuple[int, str], dict[str, Any]]:
    document = _load_json(screening_result_path)
    candidates = document.get("candidates", []) if isinstance(document, dict) else []
    pf_filter = None if pf_labels is None else set(pf_labels)
    best: dict[tuple[int, str], dict[str, Any]] = {}
    for raw in candidates:
        if not isinstance(raw, dict) or not _candidate_is_finite(raw):
            continue
        molecule_type = int(raw["molecule_type"])
        pf_label = str(raw["pf_label"])
        if molecule_min is not None and molecule_type < int(molecule_min):
            continue
        if molecule_max is not None and molecule_type > int(molecule_max):
            continue
        if pf_filter is not None and pf_label not in pf_filter:
            continue
        key = (molecule_type, pf_label)
        current = best.get(key)
        if current is None or float(raw["g_total"]) < float(current["g_total"]):
            best[key] = dict(raw)
    return best


def _make_targets(
    best_by_system_pf: dict[tuple[int, str], dict[str, Any]],
    *,
    ld_window: int,
    existing_cgs_keys: set[tuple[int, str, int]],
) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for (molecule_type, pf_label), best in sorted(best_by_system_pf.items()):
        rank = int(best["df_rank_actual"])
        best_ld = int(best["ld"])
        lo = max(0, best_ld - int(ld_window))
        hi = min(rank, best_ld + int(ld_window))
        for ld in range(lo, hi + 1):
            target = {
                "molecule_type": molecule_type,
                "pf_label": pf_label,
                "ld": ld,
                "ld_screening_best": best_ld,
                "ld_window": int(ld_window),
                "df_rank_actual": rank,
                "screening_g_total": float(best["g_total"]),
                "screening_q_opt": best.get("q_opt"),
                "screening_kappa_opt": best.get("kappa_opt"),
                "skip_existing_cgs": (molecule_type, pf_label, ld) in existing_cgs_keys,
            }
            targets.append(target)
    return targets


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute DF C_gs,D only near anchor-screened LD optima."
    )
    parser.add_argument("--screening-result", type=Path)
    parser.add_argument(
        "--target-file",
        type=Path,
        help="Explicit target JSON to run instead of deriving targets from a screening result.",
    )
    parser.add_argument(
        "--existing-cgs-table",
        type=Path,
        action="append",
        default=[],
        help="Compact/detailed Cgs tables whose exact (H,PF,LD) entries should be skipped.",
    )
    parser.add_argument("--molecule-min", type=int, default=3)
    parser.add_argument("--molecule-max", type=int, default=13)
    parser.add_argument(
        "--pf-labels",
        default="2nd,4th,4th(new_2),8th(Morales)",
        help="Comma-separated PF labels.",
    )
    parser.add_argument("--ld-window", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-stem", default="H3_H13_df_ld_window_cgs")
    parser.add_argument("--gpu-ids", default="3,4,5,6,7")
    parser.add_argument("--processes", type=int, default=None)
    parser.add_argument("--optimization-level", type=int, default=0)
    parser.add_argument(
        "--cache-path",
        type=Path,
        help="Optional DF Cgs cache path for this run.",
    )
    parser.add_argument("--ground-state-tol", type=float, default=1e-10)
    parser.add_argument("--matrix-free-threads", type=int, default=None)
    parser.add_argument(
        "--no-parameterized-template",
        action="store_true",
        help="Build one concrete circuit per t instead of a parameterized template.",
    )
    parser.add_argument(
        "--no-skip-existing-cgs",
        action="store_true",
        help="Recompute targets even when an exact entry exists in --existing-cgs-table.",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    gpu_ids = tuple(item.strip() for item in args.gpu_ids.split(",") if item.strip())
    pf_labels = _parse_csv(args.pf_labels)
    existing_keys = (
        set()
        if args.no_skip_existing_cgs
        else _load_existing_cgs_keys(tuple(args.existing_cgs_table))
    )
    if args.target_file is not None:
        loaded_targets = _load_json(args.target_file)
        if not isinstance(loaded_targets, list):
            raise ValueError("--target-file must contain a JSON list")
        targets = [dict(item) for item in loaded_targets if isinstance(item, dict)]
        for target in targets:
            key = (
                int(target["molecule_type"]),
                str(target["pf_label"]),
                int(target["ld"]),
            )
            target["skip_existing_cgs"] = (
                False
                if args.no_skip_existing_cgs
                else bool(target.get("skip_existing_cgs", key in existing_keys))
            )
    else:
        if args.screening_result is None:
            parser.error("--screening-result is required unless --target-file is used")
        best = _best_ld_by_system_pf(
            args.screening_result,
            molecule_min=args.molecule_min,
            molecule_max=args.molecule_max,
            pf_labels=pf_labels,
        )
        targets = _make_targets(
            best,
            ld_window=int(args.ld_window),
            existing_cgs_keys=existing_keys,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_path = args.output_dir / f"{args.output_stem}.json"
    partial_path = args.output_dir / f"{args.output_stem}.partial.json"
    target_path = args.output_dir / f"{args.output_stem}.targets.json"
    target_path.write_text(
        json.dumps(targets, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    rows: list[dict[str, Any]] = []
    if partial_path.exists():
        loaded = _load_json(partial_path)
        if isinstance(loaded, list):
            rows = [dict(item) for item in loaded if isinstance(item, dict)]
    done = {
        (int(row["molecule_type"]), str(row["pf_label"]), int(row["ld"]))
        for row in rows
        if "molecule_type" in row and "pf_label" in row and "ld" in row
    }

    print(
        json.dumps(
            {
                "event": "target_summary",
                "screening_result": (
                    None if args.screening_result is None else str(args.screening_result)
                ),
                "target_file": (
                    None if args.target_file is None else str(args.target_file)
                ),
                "output": str(final_path),
                "target_count": len(targets),
                "skip_existing_count": sum(
                    1 for target in targets if target["skip_existing_cgs"]
                ),
                "compute_count": sum(
                    1 for target in targets if not target["skip_existing_cgs"]
                ),
                "gpu_ids": list(gpu_ids),
            },
            sort_keys=True,
        ),
        flush=True,
    )

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
    current_molecule: int | None = None
    hamiltonian = None
    sector = None
    ranked = None
    for target in targets:
        key = (
            int(target["molecule_type"]),
            str(target["pf_label"]),
            int(target["ld"]),
        )
        if target["skip_existing_cgs"]:
            print(
                json.dumps({"event": "skip_existing_cgs", **target}, sort_keys=True),
                flush=True,
            )
            continue
        if key in done:
            print(json.dumps({"event": "skip_done", **target}, sort_keys=True), flush=True)
            continue
        molecule_type = int(target["molecule_type"])
        if current_molecule != molecule_type:
            hamiltonian, sector = build_df_h_d_from_molecule(molecule_type)
            ranked = rank_df_fragments(hamiltonian)
            current_molecule = molecule_type
            print(
                json.dumps(
                    {
                        "event": "molecule_loaded",
                        "molecule_type": molecule_type,
                        "df_rank_actual": hamiltonian.n_blocks,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        assert hamiltonian is not None
        assert sector is not None
        assert ranked is not None
        partition = split_df_hamiltonian_by_ld(
            hamiltonian,
            int(target["ld"]),
            ranked_fragments=ranked,
        )
        print(json.dumps({"event": "target_start", **target}, sort_keys=True), flush=True)
        result = get_or_compute_cached_df_cgs_fit(
            hamiltonian=hamiltonian,
            sector=sector,
            partition=partition,
            pf_label=str(target["pf_label"]),
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
        row = {
            "molecule_type": molecule_type,
            "df_rank_actual": result.df_rank_actual,
            "df_rank_requested": result.df_rank_requested,
            "df_tol_requested": result.df_tol_requested,
            "ld_screening_best": int(target["ld_screening_best"]),
            "ld_window": int(target["ld_window"]),
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
            "ground_state_cache": result.metadata.get("ground_state_cache", {}),
            "df_step_cost": df_step_cost,
            "total_ref_rz_depth": (
                df_step_cost.get("total_ref_rz_depth")
                if isinstance(df_step_cost, dict)
                else None
            ),
            "template_transpile_s": (
                result.metadata.get("parameterized_template_profile") or {}
            ).get("transpile_s"),
            "screening_g_total": target["screening_g_total"],
            "screening_q_opt": target["screening_q_opt"],
            "screening_kappa_opt": target["screening_kappa_opt"],
            "source_kind": str(target.get("source_kind", "ld_window_refinement")),
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
