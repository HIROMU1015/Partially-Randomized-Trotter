"""Shared strict raw/schema plumbing. Contains no floating-point decoder."""

from __future__ import annotations

import re

from .common import (
    CONVENTIONS, MODEL_SEMANTICS, UNITS, canonical_bytes, integer,
    keys_exact, loads_strict, object_ref, require,
)


def _labels(value, size: int, path: str) -> None:
    require(type(value) is list and len(value) == size, "INVENTORY", path, "Wrong label count")
    require(all(type(s) is str and re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*", s) for s in value),
            "MODE_OR_BLOCK_LABEL", path, "Expected distinct ASCII labels")
    require(len(set(value)) == len(value), "INVENTORY", path, "Duplicate labels")


def _array(value, shape: list[str], axes: list[list[str]], unit: str,
           allowed_dtypes: tuple[str, ...], path: str) -> None:
    keys_exact(value, ("shape", "axes", "dtype", "endian", "order", "unit", "data_hex"), path)
    require(value["shape"] == shape, "SHAPE", path, "Declared shape does not match model")
    require(value["axes"] == axes, "AXIS_ORDER", path, "Array axes differ from mode/block order")
    require(value["dtype"] in allowed_dtypes, "DTYPE", path, "Unsupported dtype; no casts")
    require(value["endian"] in ("little", "big"), "ENDIAN", path, "Explicit endian required")
    require(value["order"] == "C", "ARRAY_ORDER", path, "Only declared C order supported")
    require(value["unit"] == unit, "UNIT", path, "Wrong unit or factor scale")
    data = value["data_hex"]
    require(type(data) is str and re.fullmatch(r"[0-9a-f]*", data) is not None,
            "RAW_HEX", path, "Expected canonical lowercase hex")
    count = 1
    for dim in shape:
        count *= integer(dim, path + ".shape", natural=True)
    width = 8 if value["dtype"] == "float64" else 16
    require(len(data) == count * width * 2, "RAW_LENGTH", path, "Payload length does not match dtype/shape")


def validate_raw(raw) -> dict:
    """No path or NPZ adapter. Synthetic, explicitly full-value DF envelopes only."""
    if type(raw) is str:
        raw = loads_strict(raw)
    else:
        # Detach caller containers; reject numbers/custom objects before comparisons.
        raw = loads_strict(canonical_bytes(raw).decode("utf-8"))
    keys_exact(raw, ("schema", "semantics", "n_modes", "mode_order", "conventions",
                     "units", "block_inventory", "arrays", "provenance"), "raw")
    require(raw["schema"] == "A2_RAW_DF_V1" and raw["semantics"] == MODEL_SEMANTICS,
            "REPRESENTATION", "raw", "Only full DFHamiltonian values are accepted")
    n = integer(raw["n_modes"], "raw.n_modes", natural=True)
    _labels(raw["mode_order"], n, "raw.mode_order")
    require(raw["conventions"] == CONVENTIONS, "ORDER_CONVENTION", "raw.conventions", "Unsupported CAR/JW/bit convention")
    require(raw["units"] == UNITS, "UNIT", "raw.units", "No implicit rescaling")
    inv = raw["block_inventory"]
    require(type(inv) is list, "INVENTORY", "raw.block_inventory", "Expected full ordered block inventory")
    _labels(inv, len(inv), "raw.block_inventory")
    prov = raw["provenance"]
    keys_exact(prov, ("input_class", "representation", "constant_role", "one_body_role",
                      "df_truncation_value_role", "full_block_inventory", "original_molecular_correspondence",
                      "transform_history", "source_refs", "metadata"), "raw.provenance")
    required = {
        "input_class": "SYNTHETIC_TECHNICAL",
        "representation": "DFHAMILTONIAN_FULL",
        "constant_role": "FULL_CONSTANT_ONCE",
        "one_body_role": "FULL_ONE_BODY_ONCE",
        "df_truncation_value_role": "METADATA_ONLY_NOT_SCALAR",
        "full_block_inventory": True,
        "original_molecular_correspondence": "NOT_CERTIFIED",
    }
    for key, expected in required.items():
        require(prov[key] == expected, "INPUT_SCOPE_OR_CONVENTION", "raw.provenance." + key,
                "Synthetic full-value input required; molecular correspondence not certified")
    require(type(prov["transform_history"]) is list and type(prov["metadata"]) is dict,
            "PROVENANCE", "raw.provenance", "Expected history list and metadata object")
    require(type(prov["source_refs"]) is list, "PROVENANCE", "raw.provenance.source_refs", "Expected refs")
    for i, ref in enumerate(prov["source_refs"]):
        object_ref(ref, f"raw.provenance.source_refs[{i}]")
    a = raw["arrays"]
    keys_exact(a, ("constant", "one_body", "lambdas", "g_blocks"), "raw.arrays")
    modes = raw["mode_order"]
    _array(a["constant"], [], [], "HA", ("float64",), "raw.arrays.constant")
    _array(a["one_body"], [str(n), str(n)], [modes, modes], "HA", ("float64", "complex128"), "raw.arrays.one_body")
    _array(a["lambdas"], [str(len(inv))], [inv], "HA", ("float64",), "raw.arrays.lambdas")
    blocks = a["g_blocks"]
    require(type(blocks) is list and len(blocks) == len(inv), "INVENTORY", "raw.arrays.g_blocks", "Missing or extra block")
    for i, entry in enumerate(blocks):
        path = f"raw.arrays.g_blocks[{i}]"
        keys_exact(entry, ("block_id", "array"), path)
        require(entry["block_id"] == inv[i], "INVENTORY_ORDER", path, "g blocks and lambda inventory must agree")
        _array(entry["array"], [str(n), str(n)], [modes, modes], "DIMENSIONLESS",
               ("float64", "complex128"), path + ".array")
    return raw
