"""Bit-preserving legacy NPZ adapter. No NumPy, casts, Hermitization or algebra.

inspect_snapshot is metadata/byte identity only. adapt_snapshot is a scientific
input-conversion call and must be downstream of the adopted-contract runner guard.
"""

import ast
from hashlib import sha256
from io import BytesIO
import json
import re
import zipfile

from ..a2_spec01_03.common import (
    CONVENTIONS, MODEL_SEMANTICS, UNITS, content_id, integer, keys_exact, require,
)
from .raw import validate_raw

SCHEMA = "connected_cluster_df_hamiltonian_snapshot_v1"
MEMBERS = ("schema_version", "hamiltonian_hash", "constant", "one_body", "lambdas", "g_matrices", "metadata_json")


def _npy(blob):
    require(type(blob) is bytes and blob[:6] == b"\x93NUMPY", "NPY", "magic", "Expected NPY member")
    version = blob[6:8]
    require(version in (b"\x01\x00", b"\x02\x00"), "NPY", "version", "Only explicit NPY v1/v2 supported")
    length_bytes = 2 if version[0] == 1 else 4
    require(len(blob) >= 8 + length_bytes, "NPY", "header", "Truncated header length")
    length = int.from_bytes(blob[8:8 + length_bytes], "little")
    offset = 8 + length_bytes + length
    require(offset <= len(blob), "NPY", "header", "Truncated header")
    try:
        header = ast.literal_eval(blob[8 + length_bytes:offset].decode("latin1"))
    except (ValueError, SyntaxError, UnicodeError) as exc:
        raise ValueError("Malformed non-executable NPY header") from exc
    keys_exact(header, ("descr", "fortran_order", "shape"), "NPY.header")
    require(type(header["fortran_order"]) is bool and not header["fortran_order"],
            "ARRAY_ORDER", "NPY", "C order required; no implicit reorder")
    shape = header["shape"]
    require(type(shape) is tuple and all(type(n) is int and n >= 0 for n in shape),
            "SHAPE", "NPY", "Nonnegative tuple dimensions required")
    descr = header["descr"]
    match = re.fullmatch(r"([<>])([fcUi])(8|16|[1-9][0-9]*)", descr) if type(descr) is str else None
    require(match is not None, "DTYPE", "NPY", "Reject object/structured/native/ambiguous dtype")
    endian, kind, size = match.groups()
    size = int(size)
    require((kind, size) in (("f", 8), ("c", 16), ("i", 8)) or kind == "U",
            "DTYPE", "NPY", "Unsupported dtype")
    width = 4 * size if kind == "U" else size
    count = 1
    for dim in shape:
        count *= dim
    data = blob[offset:]
    require(len(data) == count * width, "RAW_LENGTH", "NPY", "Header and data length disagree")
    return {"shape": [str(n) for n in shape], "descr": descr, "endian": "little" if endian == "<" else "big",
            "kind": kind, "width_bytes": str(width), "data_offset": str(offset)}, data


def _members(blob):
    require(type(blob) is bytes, "INPUT_TYPE", "snapshot", "Exact container bytes required")
    with zipfile.ZipFile(BytesIO(blob)) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        require(len(names) == len(MEMBERS) and set(names) == {s + ".npy" for s in MEMBERS},
                "INVENTORY", "NPZ", "Missing, duplicate, extra or unsafe member")
        require(all(not (info.flag_bits & 1) for info in infos), "NPZ", "encryption", "Encrypted input not supported")
        return {name: archive.read(name + ".npy") for name in MEMBERS}


def _unicode_scalar(member):
    header, data = _npy(member)
    require(header["shape"] == [] and header["kind"] == "U", "NPY", "metadata", "Unicode scalar required")
    value = "".join(chr(int.from_bytes(data[k:k + 4], header["endian"])) for k in range(0, len(data), 4)).rstrip("\x00")
    require("\x00" not in value, "NPY", "metadata", "Embedded NUL not supported")
    return value


def inspect_snapshot(blob):
    """Hash file/member/array bytes and parse headers/metadata, never coefficients."""
    members = _members(blob)
    require(_unicode_scalar(members["schema_version"]) == SCHEMA, "REPRESENTATION", "schema", "Wrong snapshot writer/schema")
    legacy_hash = _unicode_scalar(members["hamiltonian_hash"])
    require(re.fullmatch(r"[0-9a-f]{64}", legacy_hash) is not None, "HASH", "legacy", "Malformed legacy metadata hash")
    metadata_text = _unicode_scalar(members["metadata_json"])
    require(type(json.loads(metadata_text)) is dict, "PROVENANCE", "metadata_json", "Metadata object required")
    rows = []
    for name in MEMBERS:
        header, data = _npy(members[name])
        rows.append({"name": name, "npy_bytes": str(len(members[name])), "npy_sha256": sha256(members[name]).hexdigest(),
                     "header": header, "array_bytes": str(len(data)), "array_sha256": sha256(data).hexdigest()})
    return {"schema": "A2_NPZ_BYTE_MANIFEST_V1", "snapshot_schema": SCHEMA,
            "container": {"bytes": str(len(blob)), "sha256": sha256(blob).hexdigest()}, "members": rows,
            "legacy_hash_metadata_only": legacy_hash, "legacy_hash_recomputed": False,
            "metadata_json_sha256": sha256(metadata_text.encode()).hexdigest(),
            "coefficient_values_decoded": False, "coefficient_algebra_performed": False}


def adapt_snapshot(blob, expected_manifest, plan):
    """Bit-preserving mapping only. Guarded by caller; never used on real Phase-A input."""
    manifest = inspect_snapshot(blob)
    require(manifest == expected_manifest, "INPUT_IDENTITY", "manifest", "Raw/member/array identity mismatch")
    fields = ("n_modes", "mode_order", "block_inventory", "sector_N", "mapping", "representation",
              "constant_role", "one_body_role", "df_truncation_value_role", "full_block_inventory")
    keys_exact(plan, fields, "input_mapping")
    require(plan["mapping"] == "STORED_AXES_AS_DECLARED_NO_PERMUTATION"
            and plan["representation"] == "DFHAMILTONIAN_FULL"
            and plan["constant_role"] == "FULL_CONSTANT_ONCE"
            and plan["one_body_role"] == "FULL_ONE_BODY_ONCE"
            and plan["df_truncation_value_role"] == "METADATA_ONLY_NOT_SCALAR"
            and plan["full_block_inventory"] is True,
            "CONVENTION", "input_mapping", "No correction-only, missing blocks or scalar reinterpretation")
    n, r = integer(plan["n_modes"], natural=True), len(plan["block_inventory"])
    require(integer(plan["sector_N"], natural=True) <= n, "SECTOR", "N", "N outside 0..n")
    members = _members(blob)

    def numeric(name, shape, axes, unit, kinds):
        header, data = _npy(members[name])
        require(header["shape"] == shape and header["kind"] in kinds,
                "SHAPE_OR_DTYPE", name, "Declared logical axes/dtype do not match stored array")
        return {"shape": shape, "axes": axes, "dtype": "float64" if header["kind"] == "f" else "complex128",
                "endian": header["endian"], "order": "C", "unit": unit, "data_hex": data.hex()}

    modes, inv = plan["mode_order"], plan["block_inventory"]
    const = numeric("constant", [], [], "HA", "f")
    q = numeric("one_body", [str(n)] * 2, [modes, modes], "HA", "fc")
    weights = numeric("lambdas", [str(r)], [inv], "HA", "f")
    g = numeric("g_matrices", [str(r), str(n), str(n)], [inv, modes, modes], "DIMENSIONLESS", "fc")
    block_hex_size = n * n * (16 if g["dtype"] == "complex128" else 8) * 2
    blocks = []
    for i, label in enumerate(inv):
        desc = {**g, "shape": [str(n)] * 2, "axes": [modes, modes],
                "data_hex": g["data_hex"][i * block_hex_size:(i + 1) * block_hex_size]}
        blocks.append({"block_id": label, "array": desc})
    raw = {"schema": "A2_RAW_DF_V1", "semantics": MODEL_SEMANTICS, "n_modes": plan["n_modes"],
           "mode_order": modes, "block_inventory": inv, "conventions": dict(CONVENTIONS), "units": dict(UNITS),
           "arrays": {"constant": const, "one_body": q, "lambdas": weights, "g_blocks": blocks},
           "provenance": {"input_class": "FIXED_DF_PILOT", "representation": plan["representation"],
                          "constant_role": plan["constant_role"], "one_body_role": plan["one_body_role"],
                          "df_truncation_value_role": plan["df_truncation_value_role"],
                          "full_block_inventory": True, "original_molecular_correspondence": "NOT_CERTIFIED",
                          "transform_history": ["EXACT_BYTE_AXIS_MAPPING_NO_NUMERICAL_TRANSFORM"],
                          "source_refs": [{"kind": "BYTES_SHA256", **manifest["container"]}],
                          "metadata": {"manifest_id": content_id("a2_input_manifest_v1", manifest),
                                       "mapping_id": content_id("a2_input_mapping_v1", plan),
                                       "legacy_hash_metadata_only": manifest["legacy_hash_metadata_only"],
                                       "original_metadata_json_sha256": manifest["metadata_json_sha256"]}}}
    return validate_raw(raw)
