"""Scope-versioned SPEC-01--03 producer; frozen arithmetic reused. Output is explicitly unverified."""

from __future__ import annotations

from fractions import Fraction

from .scope01 import (
    CAP_METHOD, CONVENTIONS, MODEL_SEMANTICS, NOT_VERIFIED, SCOPE, UNITS, VERSION,
    ContractError, complex_rat, content_id, integer, matrix_ref, node, rat,
    read_complex, read_rat, rejection, require, source_identity, submission_identity,
)
from .raw import validate_raw


def decode_binary64(data: bytes, endian: str) -> Fraction:
    """Decode integer sign/exponent/mantissa, without creating a Python float."""
    require(type(data) is bytes and len(data) == 8, "RAW_LENGTH", "binary64", "Eight bytes required")
    require(endian in ("little", "big"), "ENDIAN", "binary64", "Explicit endian required")
    word = int.from_bytes(data, endian)
    sign = -1 if word >> 63 else 1
    exponent = (word >> 52) & 2047
    fraction = word & ((1 << 52) - 1)
    require(exponent != 2047, "NONFINITE", "binary64", "NaN and infinity are not exact finite inputs")
    if exponent == 0:
        mantissa, power = fraction, -1074
    else:
        mantissa, power = (1 << 52) + fraction, exponent - 1075
    numerator = sign * mantissa
    return Fraction(numerator << power, 1) if power >= 0 else Fraction(numerator, 1 << -power)


def _real_array(descriptor: dict) -> list[Fraction]:
    data = bytes.fromhex(descriptor["data_hex"])
    return [decode_binary64(data[i:i + 8], descriptor["endian"]) for i in range(0, len(data), 8)]


def _matrix(descriptor: dict, n: int) -> list:
    parts = _real_array(descriptor)
    if descriptor["dtype"] == "float64":
        flat = [complex_rat(q) for q in parts]
    else:
        flat = [complex_rat(parts[i], parts[i + 1]) for i in range(0, len(parts), 2)]
    table = [flat[i * n:(i + 1) * n] for i in range(n)]
    for i in range(n):
        for j in range(n):
            a, b = read_complex(table[i][j])
            c, d = read_complex(table[j][i])
            require(a == c and b == -d, "NON_HERMITIAN", "matrix", "Exact conjugate-transpose equality required")
    return table


def _intake(raw: dict) -> dict:
    n = integer(raw["n_modes"], natural=True)
    a = raw["arrays"]
    c = rat(_real_array(a["constant"])[0])
    q = _matrix(a["one_body"], n)
    lambdas = _real_array(a["lambdas"])
    squares, refs = [], {
        "constant": matrix_ref(a["constant"], c),
        "one_body": matrix_ref(a["one_body"], q),
        "lambdas": matrix_ref(a["lambdas"], [rat(v) for v in lambdas]),
        "g_blocks": [],
    }
    for index, block in enumerate(a["g_blocks"]):
        g = _matrix(block["array"], n)
        squares.append({"block_id": block["block_id"], "lambda": rat(lambdas[index]), "g": g})
        refs["g_blocks"].append({"block_id": block["block_id"], "ref": matrix_ref(block["array"], g)})
    core = {
        "semantics": MODEL_SEMANTICS, "target": "FIXED_DF_OPERATOR",
        "n_modes": raw["n_modes"], "mode_order": raw["mode_order"],
        "conventions": dict(CONVENTIONS), "units": dict(UNITS),
        "block_inventory": raw["block_inventory"],
        "constant": c, "one_body": q, "squares": squares,
    }
    return {
        "raw_identity_kind": "CANONICAL_RAW_ENVELOPE_WITH_EXACT_ARRAY_BYTES",
        "raw_input_id": content_id("raw_df_v1", raw),
        "model": node("model_v1", core), "array_refs": refs,
        "provenance": raw["provenance"],
    }


def intake_model(raw) -> dict:
    """SPEC-01 only; accepted intake is not a mathematical certificate."""
    partial = {"submitted_input": submission_identity(raw)}
    try:
        accepted = _intake(validate_raw(raw))
        return {"schema": VERSION, "scope": SCOPE,
                "verdict": "INTAKE_ACCEPTED_EXACT_NOT_CERTIFICATE", "intake": accepted}
    except ContractError as exc:
        return rejection(exc, "SPEC-01", partial, verdict="INVALID_MODEL_FOR_GUARANTEE")


def _entry_cap(table: list) -> Fraction:
    total = Fraction(0)
    for row in table:
        for item in row:
            re_part, im_part = read_complex(item)
            total += abs(re_part) + abs(im_part)
    return total


def generate_spec01_03(raw, *, particle_number: str,
                       fragment_order: list[str] | None = None) -> dict:
    """Produce exact L/f/Lambda evidence only, from explicit fixed-DF in-memory bytes."""
    partial, stage = {"submitted_input": submission_identity(raw)}, "SPEC-01"
    try:
        raw = validate_raw(raw)
        intake = _intake(raw)
        partial["intake"] = intake
        model = intake["model"]["payload"]
        model_id = intake["model"]["id"]
        n = integer(model["n_modes"], natural=True)
        stage = "SPEC-02"
        particles = integer(particle_number, "particle_number", natural=True)
        require(particles <= n, "PARTICLE_NUMBER", "particle_number", "Fixed N outside 0..n")
        sector = node("sector_v1", {
            "model_id": model_id, "n_modes": model["n_modes"], "N": particle_number,
            "mode_order": model["mode_order"], "conventions": dict(CONVENTIONS),
            "conservation_grammar": "IDENTITY_NUMBER_CONSERVING_ONE_BODY_AND_SQUARES_V1",
        })
        partial["sector"] = sector
        source_order = ["one_body"] + ["df:" + b for b in model["block_inventory"]]
        order = list(source_order) if fragment_order is None else fragment_order
        require(type(order) is list and all(type(v) is str for v in order)
                and len(order) == len(source_order) and set(order) == set(source_order),
                "FRAGMENT_ORDER", "fragment_order", "Must be a full permutation, including zero fragments")
        stage = "SPEC-03"
        s_q = _entry_cap(model["one_body"])
        u_caps = [{"block_id": b["block_id"], "u": rat(_entry_cap(b["g"]))} for b in model["squares"]]
        cap_witness = node("cap_witness_v1", {
            "model_id": model_id, "method": CAP_METHOD,
            "q_coefficient_id": intake["array_refs"]["one_body"]["coefficient_id"],
            "s_q": rat(s_q), "u_caps": u_caps,
            "g_coefficient_ids": [r["ref"]["coefficient_id"] for r in intake["array_refs"]["g_blocks"]],
        })
        partial["cap_witness"] = cap_witness
        by_id = {"one_body": {
            "fragment_id": "one_body", "base_kind": "ONE_BODY",
            "coefficient_id": intake["array_refs"]["one_body"]["coefficient_id"],
            "weight": rat(1), "scalar_shift": rat(s_q), "f": rat(2 * s_q),
            "psd_rule": "ONE_BODY_LOWER_SHIFT_V1", "unit": "HA",
        }}
        lower = read_rat(model["constant"]) - s_q
        contributions = [
            {"origin": "FULL_CONSTANT_ONCE", "value": model["constant"]},
            {"origin": "ONE_BODY_LOWER_SHIFT", "value": rat(-s_q)},
        ]
        for index, b in enumerate(model["squares"]):
            lam = read_rat(b["lambda"])
            u = read_rat(u_caps[index]["u"])
            f = abs(lam) * u * u
            shift = f if lam < 0 else Fraction(0)
            lower -= shift
            by_id["df:" + b["block_id"]] = {
                "fragment_id": "df:" + b["block_id"], "base_kind": "SIGNED_ONE_BODY_SQUARE",
                "coefficient_id": intake["array_refs"]["g_blocks"][index]["ref"]["coefficient_id"],
                "weight": b["lambda"], "scalar_shift": rat(shift), "f": rat(f),
                "psd_rule": "NEGATIVE_SQUARE_CAP_SHIFT_V1" if lam < 0 else "NONNEGATIVE_SQUARE_V1",
                "unit": "HA",
            }
            contributions.append({"origin": "DF_SQUARE_LOWER_SHIFT", "block_id": b["block_id"], "value": rat(-shift)})
        fragments = [by_id[k] for k in order]
        partition = node("partition_v1", {
            "model_id": model_id, "sector_id": sector["id"], "cap_method": CAP_METHOD,
            "q_minus": rat(-s_q), "q_plus": rat(s_q), "u_caps": u_caps,
            "fragment_order": order, "fragments": fragments, "L": rat(lower),
        })
        partial["partition"] = partition
        display = [{"fragment_id": k, "time": rat(Fraction(1, 2))} for k in order[:-1]]
        display += [{"fragment_id": order[-1], "time": rat(1)}]
        display += [{"fragment_id": k, "time": rat(Fraction(1, 2))} for k in reversed(order[:-1])]
        shifts = [{"fragment_id": f["fragment_id"], "s": f["scalar_shift"], "origin": f["psd_rule"]} for f in fragments]
        sum_s = sum((read_rat(s["s"]) for s in shifts), Fraction(0))
        ledger = {
            "constant": model["constant"], "L": rat(lower), "terms": shifts,
            "sum_s": rat(sum_s), "restored_constant": rat(lower + sum_s),
            "unit": "HA", "frame": "ORIGINAL_BASE_BLOCKS_PLUS_FULL_CONSTANT_ONCE",
            "controlled_scalar": "ANCILLA_RELATIVE_PHASE_OR_EXPLICIT_FRAME_RESTORATION",
        }
        pf = node("pf_v1", {
            "partition_id": partition["id"], "fragment_order": order,
            "display_stages": display, "time_stages": list(reversed(display)),
            "product_convention": "DISPLAY_LEFT_TO_RIGHT_TIME_RIGHT_TO_LEFT",
            "formula": "POSITIVE_PALINDROMIC_S2_V1", "h_unit": "INV_HA",
            "phase_sign": "EXP_MINUS_I_H_ENERGY", "scalar_ledger": ledger,
        })
        parents = {"model_id": model_id, "sector_id": sector["id"],
                   "partition_id": partition["id"], "pf_id": pf["id"]}
        lower_cert = node("lower_certificate_v1", {
            "kind": "LowerCertificate", "parents": parents,
            "raw_input_id": intake["raw_input_id"], "L": rat(lower),
            "scalar_contributions": contributions, "witness_refs": [cap_witness["id"]],
            "rule": "K0_6_1_SIGNED_OUTER_SCALAR_SHIFT", "unit": "HA", "source_version": VERSION,
        })
        total_cap = sum((read_rat(f["f"]) for f in fragments), Fraction(0))
        cap_cert = node("fragment_cap_certificate_v1", {
            "kind": "FragmentCapCertificate", "parents": parents,
            "raw_input_id": intake["raw_input_id"], "fragments": fragments,
            "Lambda": rat(total_cap), "witness_refs": [cap_witness["id"]],
            "unit": "HA", "source_version": VERSION,
        })
        evidence = node("spec01_03_evidence_v1", {
            "schema": VERSION, "scope": SCOPE, "completion": "SPEC01_03_ONLY",
            "source": source_identity(), "context": parents,
            "intake": intake, "sector": sector, "cap_witness": cap_witness,
            "partition": partition, "pf": pf, "lower_certificate": lower_cert,
            "fragment_cap_certificate": cap_cert,
        })
        return {"schema": VERSION, "scope": SCOPE,
                "verdict": "EVIDENCE_GENERATED_UNVERIFIED", "evidence": evidence,
                "not_verified": list(NOT_VERIFIED)}
    except ContractError as exc:
        return rejection(exc, stage, partial, verdict="GENERATION_REJECTED")
    except OSError as exc:
        return rejection(ContractError("SOURCE_IO", "source", str(exc)), stage, partial,
                         verdict="TECHNICAL_ERROR_OR_INTERRUPTED")
