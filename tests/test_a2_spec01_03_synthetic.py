"""Artificial raw bits only. No scientific fixtures, integrations or datasets.

Run directly with Python/unittest to avoid the repository's integration conftest.
The exact occupation-bit oracle below is test-only, not a new runtime capability.
"""

from __future__ import annotations

import copy
from fractions import Fraction as F
import itertools
import random
import struct
import unittest
from unittest.mock import patch

from trottercert.a2_spec01_03 import (
    canonical_bytes, check_spec01_03, content_id, decode_binary64,
    generate_spec01_03, intake_model, loads_strict, read_rat,
)
from trottercert.a2_spec01_03 import checker, generator
from trottercert.a2_spec01_03.common import (
    CONVENTIONS, MODEL_SEMANTICS, UNITS, ContractError, complex_rat, integer,
    object_ref, rat, read_complex,
)


def word(value, endian="little"):
    """Test construction only; independently compare IEEE decode to struct oracle."""
    return struct.pack("<d" if endian == "little" else ">d", value).hex()


def array(values, shape, axes, *, unit="HA", dtype="float64", endian="little"):
    data = "".join(word(v, endian) for v in values)
    return {"shape": [str(v) for v in shape], "axes": copy.deepcopy(axes),
            "dtype": dtype, "endian": endian, "order": "C", "unit": unit, "data_hex": data}


def fixture(*, zero=False, endian="little", complex_values=False):
    """Two invented modes; positive, negative, zero lambda and zero-cap blocks."""
    modes = ["m0", "m1"]
    ids = ["positive", "negative", "zero_weight", "zero_cap"]
    q = [0., 0., 0., 0.] if zero else [1., .5, .5, -1.]
    gs = [[1., 0., 0., 0.], [0., 1., 1., 0.], [1., 0., 0., -1.], [0., 0., 0., 0.]]
    if complex_values:
        q_data = [0., 0., 0., .5, 0., -.5, 1., 0.]
        q_desc = array(q_data, [2, 2], [modes, modes], dtype="complex128", endian=endian)
    else:
        q_desc = array(q, [2, 2], [modes, modes], endian=endian)
    raw = {
        "schema": "A2_RAW_DF_V1", "semantics": MODEL_SEMANTICS,
        "n_modes": "2", "mode_order": modes, "conventions": dict(CONVENTIONS),
        "units": dict(UNITS), "block_inventory": ids,
        "arrays": {
            "constant": array([2.], [], [], endian=endian), "one_body": q_desc,
            "lambdas": array([.5, -.25, 0., -1.], [4], [ids], endian=endian),
            "g_blocks": [{"block_id": name, "array": array(g, [2, 2], [modes, modes], unit="DIMENSIONLESS", endian=endian)}
                         for name, g in zip(ids, gs, strict=True)],
        },
        "provenance": {
            "input_class": "SYNTHETIC_TECHNICAL", "representation": "DFHAMILTONIAN_FULL",
            "constant_role": "FULL_CONSTANT_ONCE", "one_body_role": "FULL_ONE_BODY_ONCE",
            "df_truncation_value_role": "METADATA_ONLY_NOT_SCALAR", "full_block_inventory": True,
            "original_molecular_correspondence": "NOT_CERTIFIED", "transform_history": [],
            "source_refs": [], "metadata": {"fixture": "HAND_CONSTRUCTED_NOT_MOLECULAR"},
        },
    }
    if complex_values:
        raw["arrays"]["g_blocks"][1]["array"] = array(
            [0., 0., 0., 1., 0., -1., 0., 0.], [2, 2], [modes, modes],
            unit="DIMENSIONLESS", dtype="complex128", endian=endian,
        )
    return raw


def one_body_only(n=2):
    raw = fixture()
    modes = ["m" + str(i) for i in range(n)]
    raw.update(n_modes=str(n), mode_order=modes, block_inventory=[])
    raw["arrays"]["one_body"] = array([0.] * (n * n), [n, n], [modes, modes])
    raw["arrays"]["lambdas"] = array([], [0], [[]])
    raw["arrays"]["g_blocks"] = []
    return raw


def successful(raw, number="1", order=None):
    result = generate_spec01_03(raw, particle_number=number, fragment_order=order)
    if result["verdict"] != "EVIDENCE_GENERATED_UNVERIFIED":
        raise AssertionError(result)
    evidence = result["evidence"]
    report = check_spec01_03(raw, evidence, expected_particle_number=number,
                            expected_fragment_order=order)
    if report["verdict"] != "SPEC01_03_VERIFIED_EXACT_SYNTHETIC":
        raise AssertionError(report)
    return evidence, report


def at(value, *path):
    for part in path:
        value = value[part]
    return value


def replace(value, path, replacement):
    parent = at(value, *path[:-1])
    parent[path[-1]] = replacement


def rehash(evidence):
    """Hash repair only, never repair semantic parents: exercise semantic checks."""
    body = evidence["payload"]
    for key, namespace in [("sector", "sector_v1"), ("partition", "partition_v1"),
                           ("cap_witness", "cap_witness_v1"), ("pf", "pf_v1"),
                           ("lower_certificate", "lower_certificate_v1"),
                           ("fragment_cap_certificate", "fragment_cap_certificate_v1")]:
        body[key]["id"] = content_id(namespace, body[key]["payload"])
    evidence["id"] = content_id("spec01_03_evidence_v1", body)


class EncodingTests(unittest.TestCase):
    def test_ieee_known_words(self):
        cases = [
            (0, F(0)), (1 << 63, F(0)), (0x3ff0000000000000, F(1)),
            (0xc004000000000000, F(-5, 2)), (1, F(1, 2 ** 1074)),
            (0x000fffffffffffff, F(2 ** 52 - 1, 2 ** 1074)),
            (0x0010000000000000, F(1, 2 ** 1022)),
            (0x7fefffffffffffff, F((2 ** 53 - 1) * 2 ** 971)),
        ]
        for bits, expected in cases:
            for endian in ["little", "big"]:
                with self.subTest(bits=hex(bits), endian=endian):
                    b = bits.to_bytes(8, endian)
                    self.assertEqual(decode_binary64(b, endian), expected)
                    self.assertEqual(checker._decode64(b, endian), expected)

    def test_random_ieee_against_stdlib_float_oracle(self):
        rng = random.Random(991018)
        for index in range(512):
            bits = rng.getrandbits(64)
            if (bits >> 52) & 2047 == 2047:
                bits ^= 1 << 52
            for endian in ["little", "big"]:
                with self.subTest(case=index, endian=endian):
                    b = bits.to_bytes(8, endian)
                    value = struct.unpack("<d" if endian == "little" else ">d", b)[0]
                    expected = F.from_float(value)  # Test-only independent oracle.
                    self.assertEqual(decode_binary64(b, endian), expected)
                    self.assertEqual(checker._decode64(b, endian), expected)

    def test_nonfinite_rejected_in_both_decoders(self):
        for bits in [0x7ff0000000000000, 0xfff0000000000000,
                     0x7ff8000000000001, 0x7ff0000000000001]:
            for endian in ["little", "big"]:
                with self.subTest(bits=hex(bits), endian=endian):
                    data = bits.to_bytes(8, endian)
                    with self.assertRaises(ContractError):
                        decode_binary64(data, endian)
                    with self.assertRaises(ContractError):
                        checker._decode64(data, endian)

    def test_decoder_shape_type_endian(self):
        for data, endian in [(b"", "little"), (bytes(9), "little"),
                             (bytearray(8), "little"), (bytes(8), "native")]:
            with self.subTest(data=str(data), endian=endian):
                with self.assertRaises(ContractError):
                    decode_binary64(data, endian)

    def test_canonical_and_rat_types(self):
        self.assertEqual(canonical_bytes({"z": [], "a": rat(-3)}),
                         canonical_bytes({"a": rat(-3), "z": []}))
        self.assertNotEqual(content_id("first", {}), content_id("second", {}))
        self.assertEqual(read_complex(complex_rat(F(1, 2), F(-3, 4))), (F(1, 2), F(-3, 4)))
        for value in [{"num": "2", "den": "4"}, {"num": "0", "den": "2"},
                      {"num": "-0", "den": "1"}, {"num": "+1", "den": "1"},
                      {"num": "01", "den": "1"}, {"num": "1", "den": "0"},
                      {"num": "1", "den": "-1"}, {"num": True, "den": "1"}]:
            with self.subTest(value=value), self.assertRaises(ContractError):
                read_rat(value)
        huge = 10 ** 5000 + 7
        self.assertEqual(read_rat(rat(huge)), huge)
        self.assertEqual(integer("-" + "9" * 5000), -(10 ** 5000 - 1))

    def test_strict_json_rejects_duplicates_numbers_and_cycles(self):
        for text in ['{"a":"x","a":"y"}', '{"a":1}', '{"a":1.0}',
                     '{"a":NaN}', '{"a":Infinity}', '[]', '{']:
            with self.subTest(text=text), self.assertRaises(ContractError):
                loads_strict(text)
        cyclic = []
        cyclic.append(cyclic)
        for value in [{"a": 1}, {"a": 0.5}, {"a": ()}, {"a": object()},
                      {"非ASCII": "value"}, {"a": cyclic}]:
            with self.subTest(value=type(value)), self.assertRaises(ContractError):
                canonical_bytes(value)

    def test_object_ref_encoding(self):
        object_ref({"kind": "BYTES_SHA256", "sha256": "a" * 64, "bytes": "7"})
        object_ref({"kind": "GIT_BLOB", "repository": "HIROMU1015/example",
                    "object_format": "sha1", "commit": "1" * 40, "blob": "2" * 40,
                    "path": "example", "sha256": "3" * 64})
        for ref in [{"kind": "unknown"}, {"kind": "BYTES_SHA256", "sha256": "A" * 64, "bytes": "1"}]:
            with self.subTest(ref=ref), self.assertRaises(ContractError):
                object_ref(ref)


class PositiveTests(unittest.TestCase):
    def test_hand_calculated_signed_split(self):
        evidence, report = successful(fixture())
        self.assertEqual(read_rat(report["L"]), -2)
        self.assertEqual([read_rat(f["f"]) for f in report["f_j"]], [6, F(1, 2), 1, 0, 0])
        self.assertEqual(read_rat(report["Lambda"]), F(15, 2))
        ledger = evidence["payload"]["pf"]["payload"]["scalar_ledger"]
        self.assertEqual(read_rat(ledger["sum_s"]), 4)
        self.assertEqual(read_rat(ledger["restored_constant"]), 2)
        self.assertNotIn("U", report)
        self.assertIn("U", report["not_verified"])

    def test_fixed_N_endpoints(self):
        for n in ["0", "1", "2"]:
            with self.subTest(N=n):
                successful(fixture(), n)

    def test_one_body_M1_and_vacuum(self):
        for n in [0, 1, 2]:
            for number in sorted({0, n}):
                with self.subTest(modes=n, N=number):
                    e, report = successful(one_body_only(n), str(number))
                    pf = e["payload"]["pf"]["payload"]
                    self.assertEqual(len(pf["display_stages"]), 1)
                    self.assertEqual(read_rat(pf["display_stages"][0]["time"]), 1)
                    self.assertEqual(read_rat(report["Lambda"]), 0)

    def test_complex_and_endian(self):
        for endian, complex_values in itertools.product(["little", "big"], [False, True]):
            with self.subTest(endian=endian, complex_values=complex_values):
                successful(fixture(endian=endian, complex_values=complex_values))

    def test_raw_identity_separate_from_model(self):
        little, _ = successful(fixture(endian="little"))
        big, _ = successful(fixture(endian="big"))
        a, b = little["payload"]["intake"], big["payload"]["intake"]
        self.assertEqual(a["model"]["id"], b["model"]["id"])
        self.assertNotEqual(a["raw_input_id"], b["raw_input_id"])
        raw = fixture()
        raw["provenance"]["metadata"]["df_truncation_value"] = rat(F(9, 8))
        e, report = successful(raw)
        self.assertEqual(e["payload"]["intake"]["model"]["id"], a["model"]["id"])
        self.assertEqual(read_rat(report["L"]), -2)

    def test_signed_zero_model_identity(self):
        a, b = one_body_only(), one_body_only()
        a["arrays"]["constant"]["data_hex"] = (0).to_bytes(8, "little").hex()
        b["arrays"]["constant"]["data_hex"] = (1 << 63).to_bytes(8, "little").hex()
        ea, _ = successful(a)
        eb, _ = successful(b)
        self.assertEqual(ea["payload"]["intake"]["model"]["id"], eb["payload"]["intake"]["model"]["id"])
        self.assertNotEqual(ea["payload"]["intake"]["raw_input_id"], eb["payload"]["intake"]["raw_input_id"])

    def test_zero_fragments_retained_and_PF_reorder_has_new_identity(self):
        raw = fixture()
        e, _ = successful(raw)
        order = list(reversed(e["payload"]["partition"]["payload"]["fragment_order"]))
        other, _ = successful(raw, order=order)
        self.assertEqual(len(other["payload"]["partition"]["payload"]["fragments"]), 5)
        self.assertNotEqual(e["payload"]["partition"]["id"], other["payload"]["partition"]["id"])
        self.assertNotEqual(e["payload"]["pf"]["id"], other["payload"]["pf"]["id"])
        self.assertEqual(e["payload"]["intake"]["model"]["id"], other["payload"]["intake"]["model"]["id"])

    def test_source_context_and_witness_graph(self):
        e, report = successful(fixture())
        body = e["payload"]
        self.assertEqual(body["lower_certificate"]["payload"]["parents"], report["context"])
        self.assertEqual(body["fragment_cap_certificate"]["payload"]["witness_refs"], [body["cap_witness"]["id"]])
        self.assertEqual(len(body["source"]["modules"]), 5)
        self.assertNotIn("bundle_id", body)
        self.assertNotIn("comparison_id", body)

    def test_non_diagonal_complex_g_is_exactly_retained(self):
        e, _ = successful(fixture(complex_values=True))
        g = e["payload"]["intake"]["model"]["payload"]["squares"][1]["g"]
        self.assertEqual(read_complex(g[0][1]), (F(0), F(1)))
        self.assertEqual(read_complex(g[1][0]), (F(0), F(-1)))

    def test_json_roundtrip(self):
        raw = fixture()
        e, report = successful(canonical_bytes(raw).decode())
        serialized = canonical_bytes(e).decode()
        self.assertEqual(check_spec01_03(raw, serialized, expected_particle_number="1"), report)
        self.assertEqual(intake_model(raw)["verdict"], "INTAKE_ACCEPTED_EXACT_NOT_CERTIFICATE")


class NegativeRawTests(unittest.TestCase):
    def test_raw_invalid_table(self):
        cases = [
            (("arrays", "constant", "dtype"), "complex128"),
            (("arrays", "lambdas", "dtype"), "int64"),
            (("arrays", "one_body", "endian"), "native"),
            (("arrays", "one_body", "shape"), ["2", "3"]),
            (("arrays", "one_body", "axes"), [["m1", "m0"], ["m0", "m1"]]),
            (("arrays", "one_body", "order"), "F"),
            (("arrays", "constant", "unit"), "DIMENSIONLESS"),
            (("arrays", "constant", "data_hex"), "00"),
            (("arrays", "constant", "data_hex"), "0G" * 8),
            (("arrays", "constant", "data_hex"), (0x7ff0000000000000).to_bytes(8, "little").hex()),
            (("arrays", "lambdas", "data_hex"), (0x7ff8000000000001).to_bytes(8, "little").hex() + word(1.) * 3),
            (("provenance", "input_class"), "MOLECULAR"),
            (("provenance", "representation"), "DFMODEL_CORRECTION_ONLY"),
            (("provenance", "constant_role"), "CONSTANT_ADDED_TWICE"),
            (("provenance", "one_body_role"), "CORRECTION_ONLY"),
            (("provenance", "df_truncation_value_role"), "SCALAR_CORRECTION"),
            (("provenance", "full_block_inventory"), False),
            (("mode_order",), ["m0", "m0"]),
            (("block_inventory",), ["negative", "positive", "zero_weight", "zero_cap"]),
            (("arrays", "g_blocks"), []),
            (("conventions", "occupation_bits"), "MODE0_MSB"),
            (("n_modes",), "-1"),
            (("semantics",), "DFMODEL"),
        ]
        good, _ = successful(fixture())
        for path, value in cases:
            with self.subTest(path=path, value=value):
                raw = fixture()
                replace(raw, path, value)
                self.assertEqual(intake_model(raw)["verdict"], "INVALID_MODEL_FOR_GUARANTEE")
                self.assertNotEqual(generate_spec01_03(raw, particle_number="1")["verdict"], "EVIDENCE_GENERATED_UNVERIFIED")
                self.assertEqual(check_spec01_03(raw, good, expected_particle_number="1")["verdict"], "INVALID_MODEL_FOR_GUARANTEE")

    def test_non_Hermitian_and_imaginary_diagonal(self):
        for q in [[1., 1., 0., 1.], [1., 0., 0., 0., 0., 0., 1., 1.]]:
            raw = fixture()
            raw["arrays"]["one_body"] = array(q, [2, 2], [raw["mode_order"]] * 2,
                                                dtype="complex128" if len(q) == 8 else "float64")
            with self.subTest(q=q):
                self.assertEqual(intake_model(raw)["reason"]["code"], "NON_HERMITIAN")
        raw = fixture(complex_values=True)
        raw["arrays"]["g_blocks"][1]["array"]["data_hex"] = "".join(word(v) for v in [0., 0., 0., 1., 0., 1., 0., 0.])
        self.assertEqual(intake_model(raw)["reason"]["code"], "NON_HERMITIAN")

    def test_malformed_raw_fields_are_structured_rejections(self):
        raw = fixture()
        for field in list(raw):
            for value in [None, [], {}, True, "bad"]:
                with self.subTest(field=field, value=value):
                    changed = copy.deepcopy(raw)
                    changed[field] = value
                    result = intake_model(changed)
                    self.assertEqual(result["verdict"], "INVALID_MODEL_FOR_GUARANTEE")
                    self.assertIn("code", result["reason"])
        for field in raw["arrays"]["one_body"]:
            for value in [None, [], {}, True]:
                with self.subTest(array_field=field, value=value):
                    changed = copy.deepcopy(raw)
                    changed["arrays"]["one_body"][field] = value
                    self.assertEqual(intake_model(changed)["verdict"], "INVALID_MODEL_FOR_GUARANTEE")

    def test_invalid_N_and_fragment_permutations(self):
        for number in ["-1", "3", "01", True, 1]:
            with self.subTest(N=number):
                r = generate_spec01_03(fixture(), particle_number=number)
                self.assertEqual(r["verdict"], "GENERATION_REJECTED")
                self.assertIn("intake", r["partial_evidence"])
        for order in [[], ["one_body"], ["one_body"] * 5, [None] * 5]:
            with self.subTest(order=order):
                self.assertEqual(generate_spec01_03(fixture(), particle_number="1", fragment_order=order)["reason"]["code"], "FRAGMENT_ORDER")

    def test_raw_extra_fields_and_json_numbers(self):
        for value in [1, 1.0, True]:
            raw = fixture()
            raw["arrays"]["constant"] = value
            with self.subTest(value=value):
                self.assertEqual(intake_model(raw)["verdict"], "INVALID_MODEL_FOR_GUARANTEE")
        raw = fixture()
        raw["constant_correction"] = rat(1)
        self.assertEqual(intake_model(raw)["reason"]["code"], "SCHEMA")


class MutationTests(unittest.TestCase):
    def test_evidence_mutations_are_semantically_rejected(self):
        raw = fixture()
        good, _ = successful(raw)
        body = ("payload",)
        cases = [
            (body + ("cap_witness", "payload", "s_q"), rat(0)),
            (body + ("partition", "payload", "q_minus"), rat(0)),
            (body + ("partition", "payload", "u_caps", 1, "u"), rat(1)),
            (body + ("partition", "payload", "L"), rat(100)),
            (body + ("partition", "payload", "fragments", 2, "weight"), rat(F(1, 4))),
            (body + ("partition", "payload", "fragments", 2, "scalar_shift"), rat(0)),
            (body + ("partition", "payload", "fragments", 2, "f"), rat(0)),
            (body + ("partition", "payload", "fragments", 2, "psd_rule"), "NONNEGATIVE_SQUARE_V1"),
            (body + ("fragment_cap_certificate", "payload", "Lambda"), rat(0)),
            (body + ("lower_certificate", "payload", "L"), rat(0)),
            (body + ("lower_certificate", "payload", "witness_refs"), ["0" * 64]),
            (body + ("lower_certificate", "payload", "parents", "sector_id"), "0" * 64),
            (body + ("context", "partition_id"), "0" * 64),
            (body + ("sector", "payload", "N"), "0"),
            (body + ("pf", "payload", "display_stages", 0, "time"), rat(1)),
            (body + ("pf", "payload", "time_stages", 0, "time"), rat(1)),
            (body + ("pf", "payload", "fragment_order"), list(reversed(good["payload"]["pf"]["payload"]["fragment_order"]))),
            (body + ("pf", "payload", "scalar_ledger", "constant"), rat(4)),
            (body + ("pf", "payload", "scalar_ledger", "restored_constant"), rat(4)),
            (body + ("intake", "model", "payload", "one_body", 0, 0, "re"), rat(99)),
            (body + ("intake", "model", "payload", "squares", 0, "lambda"), rat(-1)),
            (body + ("intake", "model", "payload", "squares", 0, "g", 0, 0, "re"), rat(3)),
            (body + ("intake", "model", "payload", "mode_order"), ["m1", "m0"]),
            (body + ("intake", "raw_input_id"), "0" * 64),
            (body + ("source", "modules", "generator.py"), "0" * 64),
        ]
        for path, value in cases:
            for repair_hashes in [False, True]:
                with self.subTest(path=path, repaired_hashes=repair_hashes):
                    changed = copy.deepcopy(good)
                    replace(changed, path, value)
                    if repair_hashes:
                        rehash(changed)
                    report = check_spec01_03(raw, changed, expected_particle_number="1")
                    self.assertNotEqual(report["verdict"], "SPEC01_03_VERIFIED_EXACT_SYNTHETIC")
                    self.assertEqual(report["verified_quantities"], [])

    def test_raw_bits_mutations_rejected_even_if_Hermitian(self):
        raw = fixture()
        e, _ = successful(raw)
        for path, data in [
            (("arrays", "constant", "data_hex"), word(3.)),
            (("arrays", "lambdas", "data_hex"), word(-.5) + word(-.25) + word(0.) + word(-1.)),
            (("arrays", "g_blocks", 0, "array", "data_hex"), word(2.) + word(0.) * 3),
        ]:
            with self.subTest(path=path):
                changed = copy.deepcopy(raw)
                replace(changed, path, data)
                self.assertEqual(check_spec01_03(changed, e, expected_particle_number="1")["reason"]["code"], "RAW_EXACT_MISMATCH")

    def test_fully_resigned_scalar_tampering_fails_mathematical_reconstruction(self):
        raw = fixture()
        original, _ = successful(raw)
        for delta in [F(-1), F(1)]:
            with self.subTest(delta=str(delta)):
                changed = copy.deepcopy(original)
                b = changed["payload"]
                # Keep every recorded scalar equation and every parent hash coherent,
                # but violate the actual K0 cap/shift rule behind that equation.
                f = b["partition"]["payload"]["fragments"][2]
                f["scalar_shift"] = rat(read_rat(f["scalar_shift"]) + delta)
                b["partition"]["payload"]["L"] = rat(read_rat(b["partition"]["payload"]["L"]) - delta)
                b["partition"]["id"] = content_id("partition_v1", b["partition"]["payload"])
                b["pf"]["payload"]["partition_id"] = b["partition"]["id"]
                ledger = b["pf"]["payload"]["scalar_ledger"]
                ledger["L"] = b["partition"]["payload"]["L"]
                ledger["terms"][2]["s"] = f["scalar_shift"]
                ledger["sum_s"] = rat(read_rat(ledger["sum_s"]) + delta)
                b["pf"]["id"] = content_id("pf_v1", b["pf"]["payload"])
                b["context"]["partition_id"] = b["partition"]["id"]
                b["context"]["pf_id"] = b["pf"]["id"]
                b["lower_certificate"]["payload"]["L"] = ledger["L"]
                contribution = b["lower_certificate"]["payload"]["scalar_contributions"][3]
                contribution["value"] = rat(read_rat(contribution["value"]) - delta)
                # Objects share some references; reset certificate fragments explicitly.
                b["fragment_cap_certificate"]["payload"]["fragments"] = copy.deepcopy(b["partition"]["payload"]["fragments"])
                for key in ["lower_certificate", "fragment_cap_certificate"]:
                    b[key]["payload"]["parents"] = copy.deepcopy(b["context"])
                rehash(changed)
                result = check_spec01_03(raw, changed, expected_particle_number="1")
                self.assertEqual(result["reason"]["code"], "SCALAR_IDENTITY")

    def test_fully_resigned_cap_tampering_is_not_accepted_by_hash_alone(self):
        raw = fixture()
        changed, _ = successful(raw)
        body = changed["payload"]
        body["cap_witness"]["payload"]["u_caps"][1]["u"] = rat(1)
        body["partition"]["payload"]["u_caps"][1]["u"] = rat(1)
        rehash(changed)
        result = check_spec01_03(raw, changed, expected_particle_number="1")
        self.assertEqual(result["reason"]["code"], "CAP_RECOMPUTATION")

    def test_complete_different_context_requires_explicit_expected_context(self):
        raw = fixture()
        e, _ = successful(raw, "0")
        self.assertEqual(check_spec01_03(raw, e, expected_particle_number="1")["verdict"], "IDENTITY_MISMATCH_INCOMPARABLE")
        e, _ = successful(raw)
        order = list(reversed(e["payload"]["partition"]["payload"]["fragment_order"]))
        reordered, _ = successful(raw, order=order)
        self.assertEqual(check_spec01_03(raw, reordered, expected_particle_number="1")["verdict"], "IDENTITY_MISMATCH_INCOMPARABLE")
        changed = copy.deepcopy(raw)
        changed["arrays"]["constant"]["data_hex"] = word(4.)
        other, _ = successful(changed)
        model_id = e["payload"]["intake"]["model"]["id"]
        self.assertEqual(check_spec01_03(changed, other, expected_particle_number="1", expected_model_id=model_id)["verdict"], "IDENTITY_MISMATCH_INCOMPARABLE")

    def test_hash_witness_and_partial_rejection(self):
        e, _ = successful(fixture())
        for key in ["sector", "partition", "pf", "cap_witness", "lower_certificate", "fragment_cap_certificate"]:
            with self.subTest(key=key):
                changed = copy.deepcopy(e)
                changed["payload"][key]["id"] = "0" * 64
                self.assertNotEqual(check_spec01_03(fixture(), changed, expected_particle_number="1")["verdict"], "SPEC01_03_VERIFIED_EXACT_SYNTHETIC")
        changed = copy.deepcopy(e)
        del changed["payload"]["lower_certificate"]
        self.assertEqual(check_spec01_03(fixture(), changed, expected_particle_number="1")["verdict"], "INCOMPLETE_CERTIFICATE")
        changed = copy.deepcopy(e)
        changed["payload"]["U"] = rat(0)
        self.assertEqual(check_spec01_03(fixture(), changed, expected_particle_number="1")["verdict"], "INVALID_CERTIFICATE")

    def test_malformed_evidence_fails_closed(self):
        good, _ = successful(fixture())
        for value in [None, [], {}, "{}", "{", {"id": "0" * 64, "payload": []}]:
            with self.subTest(value=value):
                self.assertNotEqual(check_spec01_03(fixture(), value, expected_particle_number="1")["verdict"], "SPEC01_03_VERIFIED_EXACT_SYNTHETIC")
        for key in good["payload"]:
            for value in [None, [], "wrong", True, {}]:
                with self.subTest(key=key, value=value):
                    changed = copy.deepcopy(good)
                    changed["payload"][key] = value
                    self.assertNotEqual(check_spec01_03(fixture(), changed, expected_particle_number="1")["verdict"], "SPEC01_03_VERIFIED_EXACT_SYNTHETIC")

    def test_generator_functions_are_not_checker_dependencies(self):
        raw = fixture()
        e, _ = successful(raw)
        with patch.object(generator, "decode_binary64", side_effect=AssertionError("generator called")), \
             patch.object(generator, "_entry_cap", side_effect=AssertionError("generator called")), \
             patch.object(generator, "_intake", side_effect=AssertionError("generator called")), \
             patch.object(generator, "generate_spec01_03", side_effect=AssertionError("generator called")):
            self.assertEqual(check_spec01_03(raw, e, expected_particle_number="1")["verdict"], "SPEC01_03_VERIFIED_EXACT_SYNTHETIC")

    def test_wrong_generator_cap_still_fails_independent_checker(self):
        with patch.object(generator, "_entry_cap", return_value=F(0)):
            bad = generate_spec01_03(fixture(), particle_number="1")["evidence"]
        self.assertEqual(check_spec01_03(fixture(), bad, expected_particle_number="1")["reason"]["code"], "CAP_RECOMPUTATION")

    def test_technical_partial_evidence_on_SOURCE_read_error(self):
        with patch.object(generator, "source_identity", side_effect=OSError("synthetic I/O failure")):
            report = generate_spec01_03(fixture(), particle_number="1")
        self.assertEqual(report["verdict"], "TECHNICAL_ERROR_OR_INTERRUPTED")
        self.assertIn("partition", report["partial_evidence"])

    def test_generator_success_flag_is_not_an_input_to_checker(self):
        producer_result = generate_spec01_03(fixture(), particle_number="1")
        producer_result["verdict"] = "SUCCESS_ALREADY_CERTIFIED"
        rejected = check_spec01_03(fixture(), producer_result, expected_particle_number="1")
        self.assertEqual(rejected["verdict"], "INCOMPLETE_CERTIFICATE")

    def test_invalid_input_retains_digest_without_marking_it_valid(self):
        raw = fixture()
        raw["semantics"] = "wrong"
        failed = generate_spec01_03(raw, particle_number="1")
        saved = failed["partial_evidence"]["submitted_input"]
        self.assertEqual(saved["status"], "INPUT_BYTES_HASHED_NOT_VALIDATED")
        self.assertEqual(len(saved["sha256"]), 64)
        rejected = check_spec01_03(raw, {}, expected_particle_number="1")
        self.assertIn("submitted_evidence", rejected["partial_evidence"])
        self.assertEqual(rejected["verified_quantities"], [])


# A test-only occupation-bit oracle: no SOURCE helper, cap, shift or decode reuse.
def cadd(a, b):
    return (a[0] + b[0], a[1] + b[1])


def cmul(a, b):
    return (a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])


def conjugate(value):
    return (value[0], -value[1])


def exact_psd_oracle(matrix):
    """Test-only rational complex Schur elimination, no SOURCE caps/eigensolvers."""
    m = copy.deepcopy(matrix)
    size = len(m)
    for i in range(size):
        for j in range(size):
            if m[i][j] != conjugate(m[j][i]):
                return False
    for pivot in range(size):
        real, imaginary = m[pivot][pivot]
        if imaginary != 0 or real < 0:
            return False
        if real == 0:
            if any(m[i][pivot] != (0, 0) for i in range(pivot + 1, size)):
                return False
            continue
        for i in range(pivot + 1, size):
            for j in range(pivot + 1, size):
                correction = cmul(m[i][pivot], m[pivot][j])
                m[i][j] = cadd(m[i][j], (-correction[0] / real, -correction[1] / real))
    return True


def one_body_action(matrix, bits):
    n = len(matrix)
    out = {}
    for p in range(n):
        for q in range(n):
            if not bits & (1 << q):
                continue
            sign = -1 if (bits & ((1 << q) - 1)).bit_count() % 2 else 1
            removed = bits ^ (1 << q)
            if removed & (1 << p):
                continue
            sign *= -1 if (removed & ((1 << p) - 1)).bit_count() % 2 else 1
            target = removed | (1 << p)
            x = matrix[p][q]
            out[target] = cadd(out.get(target, (F(0), F(0))), (sign * x[0], sign * x[1]))
    return out


class PropertyAndOracleTests(unittest.TestCase):
    def test_artificial_diagonal_full_sector_psd_and_identity(self):
        rng = random.Random(1803)
        for case in range(40):
            n = 3
            modes = ["m0", "m1", "m2"]
            q = [F(rng.randint(-6, 6), 2) for _ in range(n)]
            gs = [[F(rng.randint(-4, 4), 2) for _ in range(n)] for _ in range(3)]
            lambdas = [F(1, 2), F(-3, 4), F(0)]
            raw = one_body_only(n)
            raw["block_inventory"] = ["a", "b", "z"]
            raw["arrays"]["one_body"] = array([float(q[i]) if i == j else 0. for i in range(n) for j in range(n)], [n, n], [modes, modes])
            raw["arrays"]["lambdas"] = array([float(v) for v in lambdas], [3], [raw["block_inventory"]])
            raw["arrays"]["g_blocks"] = [{"block_id": raw["block_inventory"][k], "array": array([float(gs[k][i]) if i == j else 0. for i in range(n) for j in range(n)], [n, n], [modes, modes], unit="DIMENSIONLESS")} for k in range(3)]
            for number in range(n + 1):
                with self.subTest(case=case, N=number):
                    e, report = successful(raw, str(number))
                    fragments = e["payload"]["partition"]["payload"]["fragments"]
                    lower = read_rat(report["L"])
                    for occupied in itertools.combinations(range(n), number):
                        h = F(2) + sum((q[i] for i in occupied), F(0))
                        bases = [sum((q[i] for i in occupied), F(0))]
                        for lam, g in zip(lambdas, gs, strict=True):
                            contribution = lam * sum((g[i] for i in occupied), F(0)) ** 2
                            h += contribution
                            bases.append(contribution)
                        shifted = [base + read_rat(f["scalar_shift"]) for base, f in zip(bases, fragments, strict=True)]
                        self.assertEqual(lower + sum(shifted), h)
                        for value, fragment in zip(shifted, fragments, strict=True):
                            self.assertGreaterEqual(value, 0)
                            self.assertLessEqual(value, read_rat(fragment["f"]))

    def test_non_diagonal_complex_exact_operator_identity(self):
        raw = fixture(complex_values=True)
        e, report = successful(raw)
        # Hand-written independent coefficient matrices, not extracted from evidence.
        q = [[(F(0), F(0)), (F(0), F(1, 2))], [(F(0), F(-1, 2)), (F(1), F(0))]]
        gs = [
            [[(F(1), F(0)), (F(0), F(0))], [(F(0), F(0)), (F(0), F(0))]],
            [[(F(0), F(0)), (F(0), F(1))], [(F(0), F(-1)), (F(0), F(0))]],
            [[(F(1), F(0)), (F(0), F(0))], [(F(0), F(0)), (F(-1), F(0))]],
            [[(F(0), F(0)), (F(0), F(0))], [(F(0), F(0)), (F(0), F(0))]],
        ]
        lam = [F(1, 2), F(-1, 4), F(0), F(-1)]
        fragments = e["payload"]["partition"]["payload"]["fragments"]
        for bits in range(4):
            base_actions = [one_body_action(q, bits)]
            for weight, g in zip(lam, gs, strict=True):
                result = {}
                for mid, first in one_body_action(g, bits).items():
                    for end, second in one_body_action(g, mid).items():
                        product = cmul(first, second)
                        result[end] = cadd(result.get(end, (F(0), F(0))), (weight * product[0], weight * product[1]))
                base_actions.append(result)
            direct = {bits: (F(2), F(0))}
            restored = {bits: (read_rat(report["L"]), F(0))}
            for action, fragment in zip(base_actions, fragments, strict=True):
                for target, val in action.items():
                    self.assertEqual(target.bit_count(), bits.bit_count())
                    direct[target] = cadd(direct.get(target, (F(0), F(0))), val)
                    restored[target] = cadd(restored.get(target, (F(0), F(0))), val)
                restored[bits] = cadd(restored[bits], (read_rat(fragment["scalar_shift"]), F(0)))
            self.assertEqual(direct, restored)

    def test_complex_fragments_psd_and_caps_by_exact_small_matrix_oracle(self):
        raw = fixture(complex_values=True)
        e, _ = successful(raw)
        q = [[(F(0), F(0)), (F(0), F(1, 2))], [(F(0), F(-1, 2)), (F(1), F(0))]]
        gs = [
            [[(F(1), F(0)), (F(0), F(0))], [(F(0), F(0)), (F(0), F(0))]],
            [[(F(0), F(0)), (F(0), F(1))], [(F(0), F(-1)), (F(0), F(0))]],
            [[(F(1), F(0)), (F(0), F(0))], [(F(0), F(0)), (F(-1), F(0))]],
            [[(F(0), F(0)), (F(0), F(0))], [(F(0), F(0)), (F(0), F(0))]],
        ]
        weights = [F(1, 2), F(-1, 4), F(0), F(-1)]
        fragments = e["payload"]["partition"]["payload"]["fragments"]
        matrices = [[[(F(0), F(0)) for _ in range(4)] for _ in range(4)] for _ in fragments]
        for col in range(4):
            for row, value in one_body_action(q, col).items():
                matrices[0][row][col] = value
            for block, (weight, g) in enumerate(zip(weights, gs, strict=True), start=1):
                for mid, first in one_body_action(g, col).items():
                    for row, second in one_body_action(g, mid).items():
                        r, i = cmul(second, first)
                        matrices[block][row][col] = cadd(matrices[block][row][col], (weight * r, weight * i))
        for block, (matrix, fragment) in enumerate(zip(matrices, fragments, strict=True)):
            for i in range(4):
                matrix[i][i] = cadd(matrix[i][i], (read_rat(fragment["scalar_shift"]), F(0)))
            cap = read_rat(fragment["f"])
            complement = [[((-matrix[i][j][0] + (cap if i == j else 0)), -matrix[i][j][1]) for j in range(4)] for i in range(4)]
            with self.subTest(block=block):
                self.assertTrue(exact_psd_oracle(matrix))
                self.assertTrue(exact_psd_oracle(complement))

    def test_subnormal_cap_does_not_underflow(self):
        raw = one_body_only(1)
        raw["arrays"]["one_body"]["data_hex"] = (1).to_bytes(8, "little").hex()
        _, report = successful(raw, "1")
        self.assertEqual(read_rat(report["L"]), F(2) - F(1, 2 ** 1074))
        self.assertEqual(read_rat(report["Lambda"]), F(1, 2 ** 1073))


if __name__ == "__main__":
    unittest.main(verbosity=2)
