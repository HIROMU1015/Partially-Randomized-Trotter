"""Shared representation/identity only: no IEEE decode, caps, shifts or PF math."""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re

VERSION = "A2_SPEC01_03_V1"
SCOPE = "SPEC01_03_SYNTHETIC_TECHNICAL_ONLY"
MODEL_SEMANTICS = "DFHAMILTONIAN_FULL_SIGNED_SQUARES_EXACT_BINARY64_V1"
CAP_METHOD = "ENTRYWISE_RE_IM_ABS_SUM_V1"
CONVENTIONS = {
    "CAR": "A_ADAGGER_DELTA",
    "occupation_basis": "CREATION_ASCENDING",
    "occupation_bits": "MODE0_LSB",
    "one_body_operator": "ADAGGER_P_A_Q",
    "mode_axes": "AS_DECLARED_MODE_ORDER",
}
UNITS = {
    "constant": "HA", "one_body": "HA", "lambdas": "HA",
    "g_matrices": "DIMENSIONLESS",
}
NOT_VERIFIED = [
    "U", "W", "K_SOS", "Cbar2", "RF_EW2_step", "ground_state_access",
    "quantum_circuits", "native_cost", "original_molecular_correspondence",
]


class ContractError(ValueError):
    """A structured rejection, never an approximate repair."""

    def __init__(self, code: str, path: str, message: str):
        self.code, self.path, self.message = code, path, message
        super().__init__(f"{code} at {path}: {message}")


def require(condition: bool, code: str, path: str, message: str) -> None:
    if not condition:
        raise ContractError(code, path, message)


def keys_exact(value, keys, path: str) -> None:
    require(type(value) is dict, "SCHEMA", path, "Expected an object")
    require(set(value) == set(keys), "SCHEMA", path, "Missing or unregistered keys")


def integer(value, path: str = "Int", *, natural: bool = False) -> int:
    pattern = r"0|[1-9][0-9]*" if natural else r"0|-?[1-9][0-9]*"
    require(type(value) is str and re.fullmatch(pattern, value) is not None,
            "INTEGER_ENCODING", path, "Expected a canonical decimal integer string")
    # Avoid changing Python's process-wide decimal digit limit.
    negative = value.startswith("-")
    digits = value[1:] if negative else value
    result = 0
    for start in range(0, len(digits), 9):
        chunk = digits[start:start + 9]
        result = result * 10 ** len(chunk) + int(chunk)
    return -result if negative else result


def _decimal(value: int) -> str:
    if value == 0:
        return "0"
    negative = value < 0
    value = -value if negative else value
    parts = []
    while value:
        value, remainder = divmod(value, 10 ** 9)
        parts.append(remainder)
    return ("-" if negative else "") + str(parts[-1]) + "".join(f"{v:09d}" for v in reversed(parts[:-1]))


def rat(value: Fraction | int) -> dict:
    require(type(value) in (Fraction, int), "RATIONAL_TYPE", "Rat", "Exact input only")
    q = Fraction(value)
    return {"num": _decimal(q.numerator), "den": _decimal(q.denominator)}


def read_rat(value, path: str = "Rat") -> Fraction:
    keys_exact(value, ("num", "den"), path)
    n = integer(value["num"], path + ".num")
    d = integer(value["den"], path + ".den", natural=True)
    require(d > 0, "RATIONAL_ENCODING", path, "Denominator must be positive")
    q = Fraction(n, d)
    require(_decimal(q.numerator) == value["num"] and _decimal(q.denominator) == value["den"],
            "RATIONAL_ENCODING", path, "Rational must be reduced; zero is 0/1")
    return q


def complex_rat(re_part: Fraction, im_part: Fraction = Fraction(0)) -> dict:
    return {"re": rat(re_part), "im": rat(im_part)}


def read_complex(value, path: str = "ComplexRat") -> tuple[Fraction, Fraction]:
    keys_exact(value, ("re", "im"), path)
    return read_rat(value["re"], path + ".re"), read_rat(value["im"], path + ".im")


def _tree(value, active: set[int]) -> None:
    if value is None or type(value) in (str, bool):
        return
    require(type(value) in (dict, list), "CANONICAL_TYPE", "$",
            "No JSON numbers, floats, custom objects or implicit conversions")
    require(id(value) not in active, "CANONICAL_CYCLE", "$", "Cyclic object")
    active.add(id(value))
    if type(value) is dict:
        for key, child in value.items():
            require(type(key) is str and key.isascii(), "CANONICAL_KEY", "$",
                    "Canonical object keys must be ASCII strings")
            _tree(child, active)
    else:
        for child in value:
            _tree(child, active)
    active.remove(id(value))


def canonical_bytes(value) -> bytes:
    try:
        _tree(value, set())
        return json.dumps(value, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (RecursionError, UnicodeError) as exc:
        raise ContractError("CANONICAL_ENCODING", "$", str(exc)) from exc


def loads_strict(text: str) -> dict:
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "DUPLICATE_KEY", key, "Duplicate JSON key")
            out[key] = value
        return out

    def number(_):
        raise ContractError("JSON_NUMBER", "$", "Use canonical integer/Rat strings")

    require(type(text) is str, "SCHEMA", "$", "Expected JSON text")
    try:
        result = json.loads(text, object_pairs_hook=pairs, parse_int=number,
                            parse_float=number, parse_constant=number)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ContractError("JSON_ENCODING", "$", str(exc)) from exc
    canonical_bytes(result)
    require(type(result) is dict, "SCHEMA", "$", "Top-level object required")
    return result


def content_id(namespace: str, payload) -> str:
    return hashlib.sha256(canonical_bytes({"namespace": namespace, "payload": payload})).hexdigest()


def node(namespace: str, payload) -> dict:
    return {"id": content_id(namespace, payload), "payload": payload}


def digest(value, path: str = "Digest") -> str:
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            "DIGEST", path, "Expected lowercase SHA256")
    return value


def object_ref(value, path: str = "ObjectRef") -> None:
    require(type(value) is dict, "SCHEMA", path, "Expected ObjectRef")
    if value.get("kind") == "BYTES_SHA256":
        keys_exact(value, ("kind", "sha256", "bytes"), path)
        digest(value["sha256"], path + ".sha256")
        integer(value["bytes"], path + ".bytes", natural=True)
    elif value.get("kind") == "GIT_BLOB":
        keys_exact(value, ("kind", "repository", "object_format", "commit", "blob", "path", "sha256"), path)
        require(value["object_format"] == "sha1", "GIT_FORMAT", path, "This version pins SHA1 Git objects")
        for field in ("commit", "blob"):
            require(type(value[field]) is str and re.fullmatch(r"[0-9a-f]{40}", value[field]) is not None,
                    "GIT_OID", path + "." + field, "Expected 40-digit SHA1 Git object")
        for field in ("repository", "path"):
            require(type(value[field]) is str and bool(value[field]), "SCHEMA", path, "Missing ref text")
        digest(value["sha256"], path + ".sha256")
    else:
        raise ContractError("OBJECT_REF_KIND", path, "Unsupported reference kind")


def matrix_ref(descriptor: dict, exact_table) -> dict:
    return {
        "shape": descriptor["shape"], "dtype": descriptor["dtype"],
        "endian": descriptor["endian"], "order": descriptor["order"],
        "axes": descriptor["axes"], "unit": descriptor["unit"],
        "raw_sha256": hashlib.sha256(bytes.fromhex(descriptor["data_hex"])).hexdigest(),
        "coefficient_id": content_id("exact_coefficient_table_v1", exact_table),
        "semantics": MODEL_SEMANTICS,
    }


def submission_identity(value) -> dict:
    """Retain provenance even if validation subsequently rejects an input."""
    try:
        data = value.encode("utf-8") if type(value) is str else canonical_bytes(value)
    except (ContractError, UnicodeError):
        return {"status": "UNAVAILABLE_NONCANONICAL_INPUT", "sha256": None}
    return {
        "status": "INPUT_BYTES_HASHED_NOT_VALIDATED",
        "encoding": "JSON_TEXT_UTF8" if type(value) is str else "CANONICAL_JSON",
        "bytes": _decimal(len(data)), "sha256": hashlib.sha256(data).hexdigest(),
    }


def source_identity() -> dict:
    """Hash only this isolated package's SOURCE; no dataset/file-loader inputs."""
    root = Path(__file__).parent
    names = ("__init__.py", "common.py", "raw.py", "generator.py", "checker.py")
    return {"version": VERSION, "modules": {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names
    }}


def rejection(exc: ContractError, stage: str, partial: dict, *, verdict: str) -> dict:
    return {
        "schema": VERSION, "scope": SCOPE, "verdict": verdict,
        "stage": stage, "reason": {"code": exc.code, "path": exc.path, "message": exc.message},
        "partial_evidence": partial, "verified_quantities": [],
        "not_verified": list(NOT_VERIFIED),
    }
