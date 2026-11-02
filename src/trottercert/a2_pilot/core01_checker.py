"""Independent raw decode, Hermitian-pair cap and scalar/PF verification.

Does not import or call generator.py. Shared code is schema, rational encoding,
canonical hashing, raw container validation and SOURCE provenance only.
"""

from __future__ import annotations

from fractions import Fraction

from .scope01 import (
    CAP_METHOD, CONVENTIONS, MODEL_SEMANTICS, NOT_VERIFIED, SCOPE, UNITS, VERSION,
    ContractError, canonical_bytes, complex_rat, content_id, digest, integer,
    keys_exact, loads_strict, matrix_ref, node, rat, read_rat, rejection,
    require, source_identity, submission_identity,
)
from .raw import validate_raw


def _decode64(data: bytes, endian: str) -> Fraction:
    # Deliberately different from generator's mantissa/bit-shift implementation.
    binary = "".join(f"{b:08b}" for b in (data if endian == "big" else data[::-1]))
    sign = -1 if binary[0] == "1" else 1
    exponent = int(binary[1:12], 2)
    require(exponent != 2047, "NONFINITE", "raw coefficient", "Nonfinite coefficient")
    significand = Fraction(int(binary[12:], 2), 2 ** 52)
    if exponent:
        significand += 1
        power = exponent - 1023
    else:
        power = -1022
    scale = Fraction(2 ** power) if power >= 0 else Fraction(1, 2 ** (-power))
    return sign * significand * scale


def _scalar_at(desc: dict, position: int) -> Fraction:
    data = bytes.fromhex(desc["data_hex"])
    return _decode64(data[position * 8:position * 8 + 8], desc["endian"])


def _read_matrix(desc: dict, n: int) -> tuple[list, list]:
    data = bytes.fromhex(desc["data_hex"])
    width = 16 if desc["dtype"] == "complex128" else 8
    values = []
    for row in range(n):
        line = []
        for col in range(n):
            offset = (row * n + col) * width
            re_part = _decode64(data[offset:offset + 8], desc["endian"])
            im_part = _decode64(data[offset + 8:offset + 16], desc["endian"]) if width == 16 else Fraction(0)
            line.append((re_part, im_part))
        values.append(line)
    for row in range(n):
        require(values[row][row][1] == 0, "NON_HERMITIAN", "diagonal", "Diagonal imaginary part must vanish exactly")
        for col in range(row):
            r1, i1 = values[row][col]
            r2, i2 = values[col][row]
            require(r1 == r2 and i1 + i2 == 0, "NON_HERMITIAN", "off_diagonal", "Exact Hermitian pairs required")
    encoded = [[complex_rat(r, i) for r, i in row] for row in values]
    return values, encoded


def _reconstruct(raw: dict) -> tuple[dict, Fraction, list, list]:
    n = integer(raw["n_modes"], natural=True)
    arrays = raw["arrays"]
    c = _scalar_at(arrays["constant"], 0)
    q_values, q = _read_matrix(arrays["one_body"], n)
    lam = [_scalar_at(arrays["lambdas"], i) for i in range(len(raw["block_inventory"]))]
    squares, gs = [], []
    refs = {
        "constant": matrix_ref(arrays["constant"], rat(c)),
        "one_body": matrix_ref(arrays["one_body"], q),
        "lambdas": matrix_ref(arrays["lambdas"], [rat(a) for a in lam]),
        "g_blocks": [],
    }
    for i, entry in enumerate(arrays["g_blocks"]):
        matrix, encoded = _read_matrix(entry["array"], n)
        gs.append((entry["block_id"], lam[i], matrix))
        squares.append({"block_id": entry["block_id"], "lambda": rat(lam[i]), "g": encoded})
        refs["g_blocks"].append({"block_id": entry["block_id"], "ref": matrix_ref(entry["array"], encoded)})
    model = {
        "semantics": MODEL_SEMANTICS, "target": "FIXED_DF_OPERATOR",
        "n_modes": raw["n_modes"], "mode_order": raw["mode_order"],
        "conventions": dict(CONVENTIONS), "units": dict(UNITS),
        "block_inventory": raw["block_inventory"], "constant": rat(c),
        "one_body": q, "squares": squares,
    }
    return {
        "raw_identity_kind": "CANONICAL_RAW_ENVELOPE_WITH_EXACT_ARRAY_BYTES",
        "raw_input_id": content_id("raw_df_v1", raw), "model": node("model_v1", model),
        "array_refs": refs, "provenance": raw["provenance"],
    }, c, q_values, gs


def _hermitian_pair_cap(matrix: list) -> Fraction:
    # Diagonal plus twice each upper-triangular pair: not generator's all-entry sum.
    diagonal = sum((abs(matrix[i][i][0]) for i in range(len(matrix))), Fraction(0))
    pairs = Fraction(0)
    for row in range(len(matrix)):
        for col in range(row + 1, len(matrix)):
            real, imaginary = matrix[row][col]
            pairs += abs(real) + abs(imaginary)
    return diagonal + 2 * pairs


def _payload(record, path: str) -> dict:
    require(type(record) is dict and {"id", "payload"} <= set(record),
            "INCOMPLETE_CERTIFICATE", path, "Missing identity/payload")
    keys_exact(record, ("id", "payload"), path)
    digest(record["id"], path + ".id")
    require(type(record["payload"]) is dict, "SCHEMA", path, "Object payload required")
    return record["payload"]


def _same(actual, expected, code: str, path: str) -> None:
    require(actual == expected, code, path, "Evidence does not reproduce from raw bits and declared context")


def check_spec01_03(raw, evidence, *, expected_particle_number: str,
                    expected_fragment_order: list[str] | None = None,
                    expected_model_id: str | None = None) -> dict:
    """Recheck only SPEC-01--03. Caller supplies the independently intended N/PF."""
    stage = "RAW_INTAKE"
    partial = {"submitted_input": submission_identity(raw),
               "submitted_evidence": submission_identity(evidence)}
    try:
        raw = validate_raw(raw)
        intake, c, q_values, squares = _reconstruct(raw)
        model_id = intake["model"]["id"]
        if expected_model_id is not None:
            digest(expected_model_id, "expected_model_id")
            _same(model_id, expected_model_id, "CONTEXT_MISMATCH", "model_id")
        partial["raw_input_id"] = intake["raw_input_id"]
        partial["model_id"] = model_id
        stage = "EVIDENCE_STRUCTURE"
        evidence = loads_strict(evidence) if type(evidence) is str else loads_strict(canonical_bytes(evidence).decode("utf-8"))
        body = _payload(evidence, "evidence")
        fields = {"schema", "scope", "completion", "source", "context", "intake", "sector",
                  "cap_witness", "partition", "pf", "lower_certificate", "fragment_cap_certificate"}
        require(fields <= set(body), "INCOMPLETE_CERTIFICATE", "evidence", "Partial evidence cannot be fully accepted")
        keys_exact(body, fields, "evidence")
        _same((body["schema"], body["scope"], body["completion"]),
              (VERSION, SCOPE, "SPEC01_03_ONLY"), "SCOPE", "evidence")
        stage = "EXACT_MODEL"
        _same(body["intake"], intake, "RAW_EXACT_MISMATCH", "intake")
        n = integer(raw["n_modes"], natural=True)
        particles = integer(expected_particle_number, "expected_particle_number", natural=True)
        require(particles <= n, "PARTICLE_NUMBER", "expected_particle_number", "N outside 0..n")
        sector_payload = {
            "model_id": model_id, "n_modes": raw["n_modes"], "N": expected_particle_number,
            "mode_order": raw["mode_order"], "conventions": dict(CONVENTIONS),
            "conservation_grammar": "IDENTITY_NUMBER_CONSERVING_ONE_BODY_AND_SQUARES_V1",
        }
        stage = "SECTOR"
        _same(_payload(body["sector"], "sector"), sector_payload, "CONTEXT_MISMATCH", "sector")
        sector_id = content_id("sector_v1", sector_payload)
        _same(body["sector"]["id"], sector_id, "CONTEXT_MISMATCH", "sector.id")
        order = ["one_body"] + ["df:" + b for b in raw["block_inventory"]]
        requested_order = order if expected_fragment_order is None else expected_fragment_order
        require(type(requested_order) is list and all(type(v) is str for v in requested_order)
                and len(requested_order) == len(order) and set(requested_order) == set(order),
                "FRAGMENT_ORDER", "expected_fragment_order", "Expected complete inventory")
        stage = "CAP_RECOMPUTATION"
        s_q = _hermitian_pair_cap(q_values)
        us = [(name, lam, _hermitian_pair_cap(g)) for name, lam, g in squares]
        u_caps = [{"block_id": name, "u": rat(u)} for name, _, u in us]
        witness_payload = {
            "model_id": model_id, "method": CAP_METHOD,
            "q_coefficient_id": intake["array_refs"]["one_body"]["coefficient_id"],
            "s_q": rat(s_q), "u_caps": u_caps,
            "g_coefficient_ids": [item["ref"]["coefficient_id"] for item in intake["array_refs"]["g_blocks"]],
        }
        _same(_payload(body["cap_witness"], "cap_witness"), witness_payload,
              "CAP_RECOMPUTATION", "cap_witness")
        witness_id = content_id("cap_witness_v1", witness_payload)
        partition = _payload(body["partition"], "partition")
        keys_exact(partition, ("model_id", "sector_id", "cap_method", "q_minus", "q_plus",
                               "u_caps", "fragment_order", "fragments", "L"), "partition")
        _same((partition["model_id"], partition["sector_id"]), (model_id, sector_id), "CONTEXT_MISMATCH", "partition.parents")
        _same(partition["cap_method"], CAP_METHOD, "CAP_METHOD", "partition")
        _same(partition["q_minus"], rat(-s_q), "CAP_RECOMPUTATION", "q_minus")
        _same(partition["q_plus"], rat(s_q), "CAP_RECOMPUTATION", "q_plus")
        _same(partition["u_caps"], u_caps, "CAP_RECOMPUTATION", "u_caps")
        _same(partition["fragment_order"], requested_order, "CONTEXT_MISMATCH", "fragment_order")
        fragments = partition["fragments"]
        require(type(fragments) is list and len(fragments) == len(order), "INVENTORY", "fragments", "Full inventory required")
        stage = "PSD_AND_OPERATOR_IDENTITY"
        expected_bases = {"one_body": (
            "ONE_BODY", intake["array_refs"]["one_body"]["coefficient_id"], Fraction(1),
            s_q, s_q + s_q, "ONE_BODY_LOWER_SHIFT_V1",
        )}
        for index, (name, lam, u) in enumerate(us):
            if lam < 0:
                scalar = (-lam * u) * u
                cap = scalar
                rule = "NEGATIVE_SQUARE_CAP_SHIFT_V1"
            else:
                scalar = Fraction(0)
                cap = (lam * u) * u
                rule = "NONNEGATIVE_SQUARE_V1"
            expected_bases["df:" + name] = (
                "SIGNED_ONE_BODY_SQUARE", intake["array_refs"]["g_blocks"][index]["ref"]["coefficient_id"],
                lam, scalar, cap, rule,
            )
        checked_scalars, checked_caps = {}, {}
        for item, label in zip(fragments, requested_order, strict=True):
            keys_exact(item, ("fragment_id", "base_kind", "coefficient_id", "weight", "scalar_shift", "f", "psd_rule", "unit"), "fragment")
            kind, coeff, weight, scalar, cap, rule = expected_bases[label]
            _same((item["fragment_id"], item["base_kind"], item["coefficient_id"]),
                  (label, kind, coeff), "OPERATOR_IDENTITY", label)
            _same(read_rat(item["weight"]), weight, "SIGNED_WEIGHT", label)
            _same(read_rat(item["scalar_shift"]), scalar, "SCALAR_IDENTITY", label)
            _same(read_rat(item["f"]), cap, "FRAGMENT_CAP", label)
            _same((item["psd_rule"], item["unit"]), (rule, "HA"), "PSD_RULE", label)
            require(cap >= 0 and scalar >= 0, "PSD_RULE", label, "Nonnegative verified caps/shifts required")
            checked_scalars[label], checked_caps[label] = scalar, cap
        # Independent scalar reconstruction: c minus ALL fragment scalar shifts.
        restored_shift = sum(checked_scalars.values(), Fraction(0))
        lower = c - restored_shift
        _same(read_rat(partition["L"]), lower, "LOWER_IDENTITY", "partition.L")
        partition_id = content_id("partition_v1", partition)
        stage = "PF_AND_SCALAR_LEDGER"
        pf = _payload(body["pf"], "pf")
        keys_exact(pf, ("partition_id", "fragment_order", "display_stages", "time_stages", "product_convention", "formula", "h_unit", "phase_sign", "scalar_ledger"), "pf")
        _same(pf["partition_id"], partition_id, "CONTEXT_MISMATCH", "pf.partition_id")
        _same(pf["fragment_order"], requested_order, "CONTEXT_MISMATCH", "pf.fragment_order")
        stages = pf["display_stages"]
        m = len(order)
        require(type(stages) is list and len(stages) == 2 * m - 1, "PF_STAGES", "pf", "Wrong S2 stage count")
        totals = {label: Fraction(0) for label in order}
        for position, st in enumerate(stages):
            keys_exact(st, ("fragment_id", "time"), "stage")
            index = position if position < m else (2 * m - 2 - position)
            wanted_time = Fraction(1) if position == m - 1 else Fraction(1, 2)
            _same((st["fragment_id"], read_rat(st["time"])),
                  (requested_order[index], wanted_time), "PF_STAGES", "stage")
            totals[st["fragment_id"]] += read_rat(st["time"])
        require(stages == list(reversed(stages)) and all(v == 1 for v in totals.values()),
                "PF_STAGES", "pf", "Palindrome and full unit time for every fragment required")
        _same(pf["time_stages"], list(reversed(stages)), "PF_TIME_ORDER", "pf.time_stages")
        _same((pf["product_convention"], pf["formula"], pf["h_unit"], pf["phase_sign"]),
              ("DISPLAY_LEFT_TO_RIGHT_TIME_RIGHT_TO_LEFT", "POSITIVE_PALINDROMIC_S2_V1", "INV_HA", "EXP_MINUS_I_H_ENERGY"),
              "PF_CONVENTION", "pf")
        ledger = pf["scalar_ledger"]
        expected_terms = [{"fragment_id": f["fragment_id"], "s": f["scalar_shift"], "origin": f["psd_rule"]} for f in fragments]
        expected_ledger = {
            "constant": rat(c), "L": rat(lower), "terms": expected_terms,
            "sum_s": rat(restored_shift), "restored_constant": rat(c), "unit": "HA",
            "frame": "ORIGINAL_BASE_BLOCKS_PLUS_FULL_CONSTANT_ONCE",
            "controlled_scalar": "ANCILLA_RELATIVE_PHASE_OR_EXPLICIT_FRAME_RESTORATION",
        }
        _same(ledger, expected_ledger, "SCALAR_IDENTITY", "pf.scalar_ledger")
        pf_id = content_id("pf_v1", pf)
        parents = {"model_id": model_id, "sector_id": sector_id, "partition_id": partition_id, "pf_id": pf_id}
        _same(body["context"], parents, "CONTEXT_MISMATCH", "context")
        stage = "LOWER_AND_CAP_CERTIFICATES"
        low = _payload(body["lower_certificate"], "lower_certificate")
        scalar_contributions = [
            {"origin": "FULL_CONSTANT_ONCE", "value": rat(c)},
            {"origin": "ONE_BODY_LOWER_SHIFT", "value": rat(-checked_scalars["one_body"])},
        ]
        for label in raw["block_inventory"]:
            scalar_contributions.append({"origin": "DF_SQUARE_LOWER_SHIFT", "block_id": label,
                                         "value": rat(-checked_scalars["df:" + label])})
        _same(low, {
            "kind": "LowerCertificate", "parents": parents, "raw_input_id": intake["raw_input_id"],
            "L": rat(lower), "scalar_contributions": scalar_contributions,
            "witness_refs": [witness_id], "rule": "K0_6_1_SIGNED_OUTER_SCALAR_SHIFT", "unit": "HA", "source_version": VERSION,
        }, "LOWER_CERTIFICATE", "lower_certificate")
        cap = _payload(body["fragment_cap_certificate"], "fragment_cap_certificate")
        require("Lambda" in cap, "INCOMPLETE_CERTIFICATE", "Lambda", "Missing total cap")
        total = sum(checked_caps.values(), Fraction(0))
        _same(read_rat(cap["Lambda"]), total, "TOTAL_CAP", "Lambda")
        _same(cap, {
            "kind": "FragmentCapCertificate", "parents": parents,
            "raw_input_id": intake["raw_input_id"], "fragments": fragments,
            "Lambda": rat(total), "witness_refs": [witness_id], "unit": "HA", "source_version": VERSION,
        }, "FRAGMENT_CAP_CERTIFICATE", "fragment_cap_certificate")
        stage = "HASH_AND_SOURCE_IDENTITIES"
        for key, namespace in (
            ("sector", "sector_v1"), ("cap_witness", "cap_witness_v1"),
            ("partition", "partition_v1"), ("pf", "pf_v1"),
            ("lower_certificate", "lower_certificate_v1"),
            ("fragment_cap_certificate", "fragment_cap_certificate_v1"),
        ):
            record = body[key]
            _same(record["id"], content_id(namespace, record["payload"]), "HASH_MISMATCH", key)
        _same(body["source"], source_identity(), "SOURCE_IDENTITY", "source")
        _same(evidence["id"], content_id("spec01_03_evidence_v1", body), "HASH_MISMATCH", "evidence")
        return {
            "schema": VERSION, "scope": SCOPE, "verdict": "SPEC01_03_VERIFIED_EXACT_FIXED_DF",
            "evidence_id": evidence["id"], "context": parents, "raw_input_id": intake["raw_input_id"],
            "source": source_identity(), "L": rat(lower),
            "f_j": [{"fragment_id": label, "f": rat(checked_caps[label])} for label in requested_order],
            "Lambda": rat(total),
            "verified_quantities": ["exact_model_bits", "N_sector_grammar", "PSD_split", "L", "f_j", "Lambda", "S2_order", "scalar_identity"],
            "not_verified": list(NOT_VERIFIED),
        }
    except ContractError as exc:
        if exc.code == "CONTEXT_MISMATCH":
            verdict = "IDENTITY_MISMATCH_INCOMPARABLE"
        elif stage == "RAW_INTAKE":
            verdict = "INVALID_MODEL_FOR_GUARANTEE"
        elif exc.code == "INCOMPLETE_CERTIFICATE":
            verdict = "INCOMPLETE_CERTIFICATE"
        else:
            verdict = "INVALID_CERTIFICATE"
        return rejection(exc, stage, partial, verdict=verdict)
    except OSError as exc:
        return rejection(ContractError("SOURCE_IO", "source", str(exc)), stage, partial,
                         verdict="TECHNICAL_ERROR_OR_INTERRUPTED")
