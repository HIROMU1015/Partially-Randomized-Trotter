from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trotterlib.config import (  # noqa: E402
    DECOMPO_NUM,
    PARTIAL_RANDOMIZED_ARTIFACTS_DIR,
    PF_RZ_LAYER,
    PFLabel,
    pf_order,
)
from trotterlib.rz_layers import (  # noqa: E402
    DEFAULT_PF_RZ_LABELS,
    RZ_LAYER_DIR,
    calculate_pf_rz_layer_from_group_layers,
)


DEFAULT_OUTPUT_DIR = PARTIAL_RANDOMIZED_ARTIFACTS_DIR / "grouping_step_costs"


def _parse_csv(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _parse_molecules(raw: str | None) -> tuple[int, ...]:
    if not raw:
        return tuple(sorted(RZ_LAYER_DIR))
    values: list[int] = []
    for item in _parse_csv(raw):
        token = item.removeprefix("H")
        if "-" in token:
            lo_raw, hi_raw = token.split("-", 1)
            lo = int(lo_raw.removeprefix("H"))
            hi = int(hi_raw.removeprefix("H"))
            step = 1 if hi >= lo else -1
            values.extend(range(lo, hi + step, step))
        else:
            values.append(int(token))
    return tuple(dict.fromkeys(values))


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _csv_fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    fieldnames: dict[str, None] = {}
    for row in rows:
        for key in row:
            fieldnames.setdefault(str(key), None)
    return list(fieldnames)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _csv_fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, default=_json_default)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def _available_pf_labels(raw: str | None) -> tuple[str, ...]:
    if raw:
        return _parse_csv(raw)
    labels = tuple(str(label) for label in DEFAULT_PF_RZ_LABELS)
    return tuple(label for label in labels if any(label in row for row in PF_RZ_LAYER.values()))


def _build_grouping_step_cost_rows(
    *,
    molecules: Sequence[int],
    pf_labels: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for molecule_type in molecules:
        group_layers = tuple(int(value) for value in RZ_LAYER_DIR[int(molecule_type)])
        molecule = f"H{int(molecule_type)}"
        config_rz = PF_RZ_LAYER.get(molecule, {})
        config_pauli = DECOMPO_NUM.get(molecule, {})
        for pf_label in pf_labels:
            if pf_label not in config_rz and pf_label not in config_pauli:
                continue
            step_rz_recomputed = calculate_pf_rz_layer_from_group_layers(
                group_layers,
                pf_label,  # type: ignore[arg-type]
            )
            step_rz_config = config_rz.get(pf_label)
            step_pauli_config = config_pauli.get(pf_label)
            rows.append(
                {
                    "molecule": molecule,
                    "molecule_type": int(molecule_type),
                    "pf_label": str(pf_label),
                    "order": int(pf_order(str(pf_label))),
                    "num_groups": len(group_layers),
                    "group_rz_layers": list(group_layers),
                    "group_rz_layer_sum": int(sum(group_layers)),
                    "group_rz_layer_max": int(max(group_layers, default=0)),
                    "step_rz_layers": int(step_rz_recomputed),
                    "step_rz_layers_recomputed": int(step_rz_recomputed),
                    "step_rz_layers_config": (
                        None if step_rz_config is None else int(step_rz_config)
                    ),
                    "rz_layer_config_match": (
                        None
                        if step_rz_config is None
                        else int(step_rz_recomputed) == int(step_rz_config)
                    ),
                    "step_pauli_rotations": (
                        None if step_pauli_config is None else int(step_pauli_config)
                    ),
                    "step_pauli_rotations_config": (
                        None if step_pauli_config is None else int(step_pauli_config)
                    ),
                    "primary_grouping_step_cost_unit": "rz_layers",
                    "secondary_grouping_step_cost_unit": "pauli_rotations",
                    "cost_definition": "grouping_step_cost_from_static_group_rz_layers_v1",
                    "rz_layer_source": "trotterlib.rz_layers.RZ_LAYER_DIR",
                    "rz_layer_reference_source": "trotterlib.config.PF_RZ_LAYER",
                    "pauli_rotation_reference_source": "trotterlib.config.DECOMPO_NUM",
                }
            )
    rows.sort(
        key=lambda row: (
            int(row["molecule_type"]),
            int(row["order"]),
            str(row["pf_label"]),
        )
    )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Cache deterministic grouping one-step costs for DF/partial-randomized/"
            "grouping comparisons."
        )
    )
    parser.add_argument(
        "--molecules",
        default=None,
        help="Comma/range list such as H3-H13. Default: every H-chain in RZ_LAYER_DIR.",
    )
    parser.add_argument(
        "--pf-labels",
        default=None,
        help=(
            "Comma list of PF labels. Default: labels available in the grouped "
            "RZ-layer table."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--name", default="grouping_step_costs")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any recomputed RZ-layer entry differs from PF_RZ_LAYER.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    molecules = _parse_molecules(args.molecules)
    pf_labels = _available_pf_labels(args.pf_labels)
    rows = _build_grouping_step_cost_rows(molecules=molecules, pf_labels=pf_labels)
    mismatches = [
        row
        for row in rows
        if row.get("rz_layer_config_match") not in (None, True)
    ]
    if args.strict and mismatches:
        examples = ", ".join(
            f"{row['molecule']} {row['pf_label']}" for row in mismatches[:5]
        )
        raise SystemExit(f"RZ-layer mismatches found: {examples}")

    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cache_kind": "grouping_step_costs",
        "description": (
            "Deterministic grouping one-step costs for comparison with DF-native "
            "and partially randomized cost records. The primary unit is grouped "
            "RZ-layer depth per PF step; Pauli-rotation count is included as a "
            "secondary unit."
        ),
        "molecules": [f"H{molecule_type}" for molecule_type in molecules],
        "pf_labels": list(pf_labels),
        "primary_unit": "grouping_rz_layers_per_pf_step",
        "secondary_unit": "pauli_rotations_per_pf_step",
        "source_tables": {
            "group_rz_layers": "trotterlib.rz_layers.RZ_LAYER_DIR",
            "pf_rz_layer_reference": "trotterlib.config.PF_RZ_LAYER",
            "pauli_rotation_reference": "trotterlib.config.DECOMPO_NUM",
        },
        "summary": {
            "num_rows": len(rows),
            "num_molecules": len(molecules),
            "num_pf_labels": len(pf_labels),
            "num_rz_layer_mismatches": len(mismatches),
        },
        "rows": rows,
    }

    output_dir = Path(args.output_dir)
    json_path = output_dir / f"{args.name}.json"
    csv_path = output_dir / f"{args.name}.csv"
    _write_json(json_path, payload)
    _write_csv(csv_path, rows)
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    if mismatches:
        print(f"warning: {len(mismatches)} RZ-layer mismatches against PF_RZ_LAYER")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
