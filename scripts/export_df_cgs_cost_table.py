from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "partial_randomized_pf"
DEFAULT_INPUTS = (
    ARTIFACT_DIR / "H3_H14_df_screening_anchor_cgs.json",
    ARTIFACT_DIR / "H3_H14_df_screening_anchor_cgs.partial.json",
)
DEFAULT_OUTPUT = ARTIFACT_DIR / "df_cgs_cost_table.json"
DEFAULT_SPLIT_DIR = ARTIFACT_DIR / "df_cgs_cost_tables"


def _load_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [dict(item) for item in data if isinstance(item, dict)]
    if isinstance(data, dict) and isinstance(data.get("entries"), list):
        return [dict(item) for item in data["entries"] if isinstance(item, dict)]
    raise ValueError(f"Unsupported Cgs input format: {path}")


def _entry_key(entry: dict[str, Any]) -> tuple[int, str, int, str]:
    return (
        int(entry["molecule_type"]),
        str(entry["pf_label"]),
        int(entry["ld"]),
        str(entry.get("source_kind", "unknown")),
    )


def _compact_entry(row: dict[str, Any], *, source_path: Path) -> dict[str, Any]:
    df_step_cost = row.get("df_step_cost")
    if not isinstance(df_step_cost, dict):
        df_step_cost = {}
    ld = int(row["ld"])
    ld_anchor = row.get("ld_anchor")
    is_anchor = ld_anchor is not None and ld == int(ld_anchor)
    return {
        "molecule": f"H{int(row['molecule_type'])}",
        "molecule_type": int(row["molecule_type"]),
        "pf_label": str(row["pf_label"]),
        "order": int(row["order"]),
        "ld": ld,
        "ld_anchor": int(ld_anchor) if ld_anchor is not None else None,
        "is_screening_anchor": bool(is_anchor),
        "source_kind": "screening_anchor" if is_anchor else "explicit_ld",
        "df_rank_actual": int(row["df_rank_actual"]),
        "df_rank_requested": row.get("df_rank_requested"),
        "lambda_r": float(row["lambda_r"]),
        "c_gs_d": float(row["c_gs_d"]),
        "fit_slope": row.get("fit_slope"),
        "fit_coeff": row.get("fit_coeff"),
        "fixed_order_coeff": row.get("fixed_order_coeff"),
        "t_values": list(row.get("t_values", ())),
        "total_ref_rz_depth": row.get("total_ref_rz_depth"),
        "total_ref_rz_count": df_step_cost.get("total_ref_rz_count"),
        "u_ref_rz_depth": df_step_cost.get("u_ref_rz_depth"),
        "d_ref_rz_depth": df_step_cost.get("d_ref_rz_depth"),
        "cost_definition": df_step_cost.get("cost_definition"),
        "d_cost_method": (
            (df_step_cost.get("d_only_cost") or {}).get("cost_method")
            if isinstance(df_step_cost.get("d_only_cost"), dict)
            else None
        ),
        "source_path": str(source_path.relative_to(PROJECT_ROOT)),
    }


def _default_input() -> Path:
    for path in DEFAULT_INPUTS:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No default Cgs input found. Pass --input explicitly."
    )


def _pf_slug(pf_label: str) -> str:
    return (
        str(pf_label)
        .replace("(", "_")
        .replace(")", "")
        .replace("/", "_")
        .replace(" ", "_")
    )


def _write_split_tables(entries: list[dict[str, Any]], split_dir: Path) -> None:
    index_entries: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for entry in entries:
        family = "h_chain"
        grouped.setdefault(
            (family, int(entry["molecule_type"]), str(entry["pf_label"])),
            [],
        ).append(entry)

    for (family, molecule_type, pf_label), group_entries in sorted(grouped.items()):
        molecule = f"H{molecule_type}"
        rel_path = (
            Path(family)
            / molecule
            / f"{_pf_slug(pf_label)}.json"
        )
        out_path = split_dir / rel_path
        group_entries = sorted(
            group_entries,
            key=lambda item: (
                int(item["ld"]),
                str(item["source_kind"]),
            ),
        )
        document = {
            "schema_version": 1,
            "table_type": "df_cgs_cost_input_by_system_pf",
            "molecule_family": family,
            "molecule": molecule,
            "molecule_type": molecule_type,
            "pf_label": pf_label,
            "order": int(group_entries[0]["order"]),
            "entries": group_entries,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(document, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        index_entries.append(
            {
                "molecule_family": family,
                "molecule": molecule,
                "molecule_type": molecule_type,
                "pf_label": pf_label,
                "order": int(group_entries[0]["order"]),
                "path": str(rel_path),
                "num_entries": len(group_entries),
                "screening_anchor_ld": next(
                    (
                        int(item["ld"])
                        for item in group_entries
                        if item.get("is_screening_anchor")
                    ),
                    None,
                ),
            }
        )

    index = {
        "schema_version": 1,
        "table_type": "df_cgs_cost_input_index",
        "description": "Index of compact DF Cgs tables split by molecule family, system size, and PF label.",
        "entries": index_entries,
    }
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export a compact DF Cgs table for cost optimization."
    )
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        help="Detailed Cgs JSON to import. Can be passed multiple times.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=DEFAULT_SPLIT_DIR,
        help="Directory for per-system/per-PF compact tables.",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Only write the aggregate output JSON.",
    )
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="Keep entries already present in the output unless replaced by input rows.",
    )
    args = parser.parse_args()

    input_paths = tuple(args.input) if args.input else (_default_input(),)
    entries_by_key: dict[tuple[int, str, int, str], dict[str, Any]] = {}

    output = args.output
    if args.merge_existing and output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        for entry in existing.get("entries", []):
            if isinstance(entry, dict):
                entries_by_key[_entry_key(entry)] = dict(entry)

    for input_path in input_paths:
        input_path = input_path.resolve()
        for row in _load_rows(input_path):
            entry = _compact_entry(row, source_path=input_path)
            entries_by_key[_entry_key(entry)] = entry

    entries = sorted(
        entries_by_key.values(),
        key=lambda item: (
            int(item["molecule_type"]),
            int(item["order"]),
            int(item["ld"]),
            str(item["source_kind"]),
        ),
    )
    document = {
        "schema_version": 1,
        "table_type": "df_cgs_cost_input",
        "description": (
            "Compact Cgs table for DF partial-randomized PF cost optimization. "
            "For the reduced screening model, use entries with "
            "is_screening_anchor=true as LD-independent Cgs values."
        ),
        "entries": entries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {output} ({len(entries)} entries)")
    if not args.no_split:
        _write_split_tables(entries, args.split_dir)
        print(f"wrote split tables under {args.split_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
