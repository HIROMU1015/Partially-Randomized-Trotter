"""Invented exact inputs only. Direct unittest, no scientific data/conftest."""

import copy
from fractions import Fraction as F
import importlib.util
import itertools
from pathlib import Path
import random
import unittest
from unittest.mock import patch

from trottercert.a2_spec01_03 import generate_spec01_03
from trottercert.a2_spec01_03.common import (
    ContractError, canonical_bytes, complex_rat, content_id, rat, read_complex, read_rat,
)
from trottercert.a2_spec04_07 import check_spec04_07, generate_spec04_07
from trottercert.a2_spec04_07 import checker, generator
from trottercert.a2_spec04_07 import producer_algebra as pa, verifier_algebra as va
from trottercert.a2_spec04_07 import producer_steps as ps, verifier_steps as vs
from trottercert.a2_spec04_07.common import NAMESPACE, policy_validate, source_identity

_spec = importlib.util.spec_from_file_location("a2_old_synthetic_fixtures", Path(__file__).with_name("test_a2_spec01_03_synthetic.py"))
old = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(old)


def policy(**changes):
    result = {"d": rat(F(1, 2)), "alpha_spec": rat(F(1, 10)),
              "root_width": rat(F(1, 2 ** 16)), "max_bisections": "128"}
    result.update(changes)
    return result


def prepare(raw=None, number="1", occupied=None, order=None, pol=None, bits="16"):
    raw = old.fixture() if raw is None else raw
    occupied = ["0"] if occupied is None else occupied
    base = generate_spec01_03(raw, particle_number=number, fragment_order=order)
    if base["verdict"] != "EVIDENCE_GENERATED_UNVERIFIED":
        raise AssertionError(base)
    base = base["evidence"]
    result = generate_spec04_07(raw, base, particle_number=number, occupied_modes=occupied,
                               sqrt_bits=bits, comparison_policy=pol, fragment_order=order)
    if result["verdict"] != "EVIDENCE_GENERATED_UNVERIFIED":
        raise AssertionError(result)
    return raw, base, result["evidence"]


def checked(case, number="1", occupied=None, order=None, pol=None, bits="16"):
    raw, base, evidence = case
    return check_spec04_07(raw, base, evidence, expected_particle_number=number,
                           expected_occupied_modes=["0"] if occupied is None else occupied,
                           expected_sqrt_bits=bits, expected_comparison_policy=pol,
                           expected_fragment_order=order)


def data(evidence, key):
    return evidence["payload"][key]["payload"]["data"]


def rehash(evidence):
    body = evidence["payload"]
    for key, space in (("upper_certificate", "upper_certificate_v1"), ("ksos_certificate", "ksos_certificate_v1"),
                       ("commutator_certificate", "commutator_certificate_v1")):
        if key in body:
            body[key]["id"] = content_id(space, body[key]["payload"])
    if "step_comparison" in body:
        body["step_comparison"]["payload"]["certificate_refs"] = [body[k]["id"] for k in
                ("upper_certificate", "ksos_certificate", "commutator_certificate") if k in body]
        body["step_comparison"]["id"] = content_id("step_comparison_v1", body["step_comparison"]["payload"])
    evidence["id"] = content_id(NAMESPACE, body)


def cancellation_fixture():
    raw = old.one_body_only()
    modes, labels = raw["mode_order"], ["plus", "minus"]
    raw["arrays"]["one_body"] = old.array([1., 0., 0., -1.], [2, 2], [modes, modes])
    raw["block_inventory"] = labels
    raw["arrays"]["lambdas"] = old.array([1., -1.], [2], [labels])
    g = [1., .5, .5, 0.]
    raw["arrays"]["g_blocks"] = [{"block_id": k, "array": old.array(g, [2, 2], [modes, modes], unit="DIMENSIONLESS")} for k in labels]
    return raw


def random_fixture(n, seed):
    rng = random.Random(seed)
    raw = old.one_body_only(n)
    modes = raw["mode_order"]

    def hermitian(unit):
        entries = [[(0., 0.) for _ in range(n)] for _ in range(n)]
        for i in range(n):
            entries[i][i] = (rng.randint(-2, 2) / 2, 0.)
            for j in range(i):
                r, im = rng.randint(-2, 2) / 2, rng.randint(-2, 2) / 2
                entries[i][j], entries[j][i] = (r, im), (r, -im)
        flat = [v for row in entries for pair in row for v in pair]
        return old.array(flat, [n, n], [modes, modes], dtype="complex128", unit=unit)

    raw["arrays"]["one_body"] = hermitian("HA")
    labels = ["a", "b"]
    raw["block_inventory"] = labels
    raw["arrays"]["lambdas"] = old.array([.5, -.25], [2], [labels])
    raw["arrays"]["g_blocks"] = [{"block_id": k, "array": hermitian("DIMENSIONLESS")} for k in labels]
    return raw


class UpperTests(unittest.TestCase):
    def test_hand_signed_expectations(self):
        for occ, expected in [(["0"], F(13, 4)), (["1"], F(3, 4))]:
            with self.subTest(occ=occ):
                case = prepare(occupied=occ)
                report = checked(case, occupied=occ)
                self.assertEqual(report["verdict"], "SPEC04_07_VERIFIED_EXACT_SYNTHETIC")
                self.assertEqual(read_rat(report["U"]), expected)
                self.assertEqual(read_rat(report["W"]), expected + 2)

    def test_complex_signed_expectations(self):
        for occ, expected in [(["0"], F(9, 4)), (["1"], F(11, 4))]:
            with self.subTest(occ=occ):
                case = prepare(old.fixture(complex_values=True), occupied=occ)
                report = checked(case, occupied=occ)
                self.assertEqual(read_rat(report["U"]), expected)

    def test_vacuum_and_full_occupation(self):
        for number, occ, expected in [("0", [], F(2)), ("2", ["0", "1"], F(5, 2))]:
            with self.subTest(N=number):
                case = prepare(number=number, occupied=occ)
                self.assertEqual(read_rat(checked(case, number=number, occupied=occ)["U"]), expected)

    def test_zero_width_is_known_not_advantage(self):
        raw = old.one_body_only()
        case = prepare(raw, pol=policy())
        report = checked(case, pol=policy())
        self.assertEqual(read_rat(report["W"]), 0)
        self.assertEqual(report["step_comparison"]["status"], "TRIVIAL_KNOWN_ENERGY")

    def test_empty_mode_formal_endpoint(self):
        case = prepare(old.one_body_only(0), number="0", occupied=[], pol=policy())
        report = checked(case, number="0", occupied=[], pol=policy())
        self.assertEqual(report["verdict"], "SPEC04_07_VERIFIED_EXACT_SYNTHETIC")
        self.assertEqual(read_rat(report["Lambda"]), 0)
        self.assertEqual(report["step_comparison"]["status"], "TRIVIAL_KNOWN_ENERGY")

    def test_invalid_trial_rejected(self):
        raw, base, _ = prepare()
        for occupied in [["0", "0"], ["2"], ["-1"], [], ["1", "0"], [0], [True], "0"]:
            with self.subTest(occupied=occupied):
                out = generate_spec04_07(raw, base, particle_number="1", occupied_modes=occupied, sqrt_bits="8")
                self.assertNotEqual(out["verdict"], "EVIDENCE_GENERATED_UNVERIFIED")

    def test_expected_trial_context_independent(self):
        case = prepare()
        self.assertEqual(checked(case, occupied=["1"])["stage"], "SPEC04_UPPER")

    def test_transition_sign_and_sector(self):
        n = 3
        matrix = [[(F(0), F(0)) for _ in range(n)] for _ in range(n)]
        matrix[0][2] = matrix[2][0] = (F(1), F(0))
        self.assertEqual(pa.action_column(matrix, 6), {3: (F(-1), F(0))})
        self.assertEqual(va.occupation_matrix(matrix)[3, 6], (F(-1), F(0)))

    def test_random_columns_all_occupations(self):
        rng = random.Random(418)
        for n in (1, 2, 3, 4):
            matrix = [[(F(rng.randint(-3, 3), 2), F(rng.randint(-3, 3), 4)) for _ in range(n)] for _ in range(n)]
            ref = va.occupation_matrix(matrix)
            for ket in range(2 ** n):
                with self.subTest(n=n, ket=ket):
                    self.assertEqual(pa.action_column(matrix, ket), {r: v for (r, c), v in ref.items() if c == ket})
                    self.assertTrue(all(k.bit_count() == ket.bit_count() for k in pa.action_column(matrix, ket)))

    def test_generator_fault_not_inherited_by_checker(self):
        case = prepare()
        with patch.object(generator.algebra, "upper", side_effect=AssertionError("generator unavailable")):
            self.assertEqual(checked(case)["verdict"], "SPEC04_07_VERIFIED_EXACT_SYNTHETIC")


class KSOSTests(unittest.TestCase):
    def test_two_fragment_hand_B_rows(self):
        record = pa.k_sos([F(2), F(8)], ["a", "b"], 8)
        self.assertEqual([read_rat(v) for v in record["row_sums_upper"]], [F(13, 3), F(17, 3)])
        self.assertEqual(va.verify_k_sos(record, [F(2), F(8)], ["a", "b"], 8), F(17, 3))
        self.assertEqual(record["raw_count"], "2")

    def test_single_fragment_zero_K(self):
        rec = pa.k_sos([F(3)], ["only"], 8)
        self.assertEqual(read_rat(rec["K_SOS_upper"]), 0)
        self.assertEqual(rec["dictionary"], [])
        self.assertEqual(va.verify_k_sos(rec, [F(3)], ["only"], 8), 0)

    def test_raw_dictionary_counts(self):
        for m in range(1, 9):
            with self.subTest(M=m):
                caps = [F(j, 2) for j in range(m)]
                rec = pa.k_sos(caps, [str(j) for j in range(m)], 12)
                expected = m * (m - 1) // 2 + m * (m - 1) * (2 * m - 1) // 6
                self.assertEqual(int(rec["raw_count"]), expected)
                self.assertEqual(va.verify_k_sos(rec, caps, [str(j) for j in range(m)], 12), read_rat(rec["K_SOS_upper"]))

    def test_sqrt_upper_zero_rational_irrational(self):
        for a, b in [(0, 7), (2, 8), (2, 3), (F(1, 3), F(7, 11)), (F(1, 2 ** 40), F(3))]:
            for bits in [1, 8, 64]:
                with self.subTest(a=str(a), b=str(b), bits=bits):
                    rec = pa.k_sos([F(a), F(b)], ["a", "b"], bits)
                    va.verify_k_sos(rec, [F(a), F(b)], ["a", "b"], bits)
                    for w in rec["sqrt_witnesses"]:
                        z, u = read_rat(w["z"]), read_rat(w["u"])
                        self.assertGreaterEqual(u * u, z)
                        self.assertLess(u * u - z, (2 * u + F(1, 2 ** bits)) / 2 ** bits)

    def test_root_witness_bad_upper_and_nonminimal(self):
        rec = pa.k_sos([F(2), F(3)], ["a", "b"], 8)
        for w in range(len(rec["sqrt_witnesses"])):
            for change in ["negative", "under", "over", "z", "offgrid", "position"]:
                with self.subTest(w=w, change=change):
                    bad = copy.deepcopy(rec)
                    item = bad["sqrt_witnesses"][w]
                    if change == "negative": item["u"] = rat(-1)
                    elif change == "under": item["u"] = rat(0)
                    elif change == "over": item["u"] = rat(read_rat(item["u"]) + 1)
                    elif change == "z": item["z"] = rat(999)
                    elif change == "position": item["positions"] = ["0", "0"]
                    else: item["u"] = rat(read_rat(item["u"]) + F(1, 3))
                    with self.assertRaises(ContractError):
                        va.verify_k_sos(bad, [F(2), F(3)], ["a", "b"], 8)

    def test_precision_invalid_not_defaulted(self):
        raw, base, _ = prepare()
        for bits in ["0", "-1", "01", "1.5", 8, True, None]:
            with self.subTest(bits=bits):
                out = generate_spec04_07(raw, base, particle_number="1", occupied_modes=["0"], sqrt_bits=bits)
                self.assertNotEqual(out["verdict"], "EVIDENCE_GENERATED_UNVERIFIED")

    def test_checker_does_not_call_generator_rows(self):
        case = prepare()
        with patch.object(pa, "k_sos", side_effect=AssertionError("no producer")):
            self.assertEqual(checked(case)["verdict"], "SPEC04_07_VERIFIED_EXACT_SYNTHETIC")


class CommutatorTests(unittest.TestCase):
    def test_two_fragment_hand_nested_commutators(self):
        raw = old.one_body_only()
        modes, labels = raw["mode_order"], ["a"]
        raw["arrays"]["one_body"] = old.array([1., 0., 0., -1.], [2, 2], [modes, modes])
        raw["block_inventory"] = labels
        raw["arrays"]["lambdas"] = old.array([1.], [1], [labels])
        raw["arrays"]["g_blocks"] = [{"block_id": "a", "array": old.array([1., 1., 1., 0.], [2, 2], [modes, modes], unit="DIMENSIONLESS")}]
        case = prepare(raw)
        cb = data(case[2], "commutator_certificate")
        self.assertEqual(read_rat(cb["groups"][0]["first"]["structure_l1"]), 4)
        self.assertEqual(read_rat(cb["groups"][0]["second"]["structure_l1"]), 6)
        self.assertEqual(read_rat(checked(case)["Cbar2"]), F(2, 3))

    def test_JW_vs_independent_trace_all_one_body_units(self):
        for n in (1, 2, 3):
            for p in range(n):
                for q in range(n):
                    for coefficient in [(F(1), F(0)), (F(0), F(1))]:
                        with self.subTest(n=n, p=p, q=q, coefficient=str(coefficient)):
                            matrix = [[(F(0), F(0)) for _ in range(n)] for _ in range(n)]
                            matrix[p][q] = coefficient
                            self.assertEqual(pa.pauli_record(pa.one_body_pauli(matrix)), va.trace_pauli(va.occupation_matrix(matrix), n))

    def test_grouped_tail_cancellation_not_triangle_of_triples(self):
        case = prepare(cancellation_fixture(), pol=policy())
        report = checked(case, pol=policy())
        cb = data(case[2], "commutator_certificate")
        self.assertEqual(read_rat(report["Cbar2"]), 0)
        self.assertGreater(read_rat(cb["groups"][0]["first"]["center_cap"]), 0)
        self.assertEqual(cb["groups"][0]["first"]["pauli"], [])
        self.assertEqual(cb["groups"][0]["second"]["pauli"], [])
        self.assertEqual(report["step_comparison"]["status"], "COMMUTATOR_ZERO_EXACT_S2")

    def test_nonzero_hermitian_coefficients_and_min(self):
        case = prepare()
        cb = data(case[2], "commutator_certificate")
        self.assertGreater(read_rat(cb["Cbar2"]), 0)
        total = F(0)
        for group in cb["groups"]:
            for name, weight in [("first", F(1, 24)), ("second", F(1, 12))]:
                v = group[name]
                self.assertTrue(all(read_complex(t["coefficient"])[1] == 0 for t in v["pauli"]))
                self.assertEqual(read_rat(v["selected"]), min(read_rat(v["structure_l1"]), read_rat(v["center_cap"])))
                total += weight * read_rat(v["selected"])
        self.assertEqual(total, read_rat(cb["Cbar2"]))

    def test_random_complex_full_bundle(self):
        for n in (1, 2, 3):
            for seed in range(4):
                with self.subTest(n=n, seed=seed):
                    case = prepare(random_fixture(n, seed), occupied=[str(n - 1)], pol=policy())
                    self.assertEqual(checked(case, occupied=[str(n - 1)], pol=policy())["verdict"], "SPEC04_07_VERIFIED_EXACT_SYNTHETIC")

    def test_exact_JW_products_vs_matrix_squares(self):
        rng = random.Random(618)
        for n in (2, 3):
            for sample in range(6):
                with self.subTest(n=n, sample=sample):
                    matrix = [[(F(rng.randrange(-2, 3)), F(rng.randrange(-2, 3))) for _ in range(n)] for _ in range(n)]
                    poly, mat = pa.one_body_pauli(matrix), va.occupation_matrix(matrix)
                    self.assertEqual(pa.pauli_record(pa.product(poly, poly)), va.trace_pauli(va.matrix_product(mat, mat), n))

    def test_identity_external_scalar_invariance(self):
        a, b = old.fixture(), old.fixture()
        b["arrays"]["constant"] = old.array([1024.], [], [])
        ca, cb = prepare(a), prepare(b)
        self.assertEqual(data(ca[2], "commutator_certificate"), data(cb[2], "commutator_certificate"))
        self.assertEqual(data(ca[2], "upper_certificate")["W"], data(cb[2], "upper_certificate")["W"])
        self.assertNotEqual(ca[2]["payload"]["context"], cb[2]["payload"]["context"])

    def test_permutations_preserve_actual_order(self):
        labels = ["one_body", "df:positive", "df:negative", "df:zero_weight", "df:zero_cap"]
        for order in itertools.permutations(labels[:3]):
            full = list(order) + labels[3:]
            with self.subTest(order=full):
                case = prepare(order=full)
                self.assertEqual(checked(case, order=full)["verdict"], "SPEC04_07_VERIFIED_EXACT_SYNTHETIC")
                self.assertEqual(data(case[2], "ksos_certificate")["position_to_fragment"][0]["fragment_id"], full[0])

    def test_checker_does_not_call_JW_generator(self):
        case = prepare()
        with patch.object(pa, "one_body_pauli", side_effect=AssertionError("no JW")):
            self.assertEqual(checked(case)["verdict"], "SPEC04_07_VERIFIED_EXACT_SYNTHETIC")


class StepTests(unittest.TestCase):
    def test_normal_bundle_comparison(self):
        case = prepare(pol=policy())
        report = checked(case, pol=policy())
        self.assertEqual(report["verdict"], "SPEC04_07_VERIFIED_EXACT_SYNTHETIC")
        self.assertEqual(report["step_comparison"]["status"], "ENCLOSED")
        self.assertEqual([r["route_id"] for r in report["step_comparison"]["routes"]], ["RF0", "RF1", "EW2", "CENTERED_CBAR2"])

    def test_policy_none_no_implicit_execution(self):
        report = checked(prepare())
        self.assertEqual(report["step_comparison"]["status"], "NOT_EVALUATED_PENDING_POLICY")
        self.assertNotIn("step_predicates_and_enclosures", report["verified_quantities"])

    def test_direct_vs_cleared_predicates(self):
        rng = random.Random(718)
        for sample in range(160):
            lam, width, k, c = [F(rng.randrange(1, 20), rng.randrange(1, 9)) for _ in range(4)]
            h = F(rng.randrange(0, 300), 100) / lam
            for route in ps.ROUTES:
                with self.subTest(sample=sample, route=route):
                    args = (route, h, lam, width, k, c, F(3, 7), F(1, 20))
                    self.assertEqual(ps.predicate(*args), vs.predicate(*args))

    def test_strict_domain_edges(self):
        for route, edge in [("RF0", F(1, 4)), ("RF1", F(1, 4)), ("EW2", F(1, 4)), ("CENTERED_CBAR2", F(2))]:
            for fun in (ps.predicate, vs.predicate):
                with self.subTest(route=route, fun=fun.__module__):
                    self.assertEqual(fun(route, edge, F(1), F(1), F(1), F(1), F(1), F(1, 10))[0], "DOMAIN_EDGE_OPEN")
                    self.assertEqual(fun(route, F(0), F(1), F(1), F(1), F(1), F(1), F(1, 10))[0], "ZERO_ANCHOR")

    def test_eta_one_excluded(self):
        # At h=1/8, eta0=2/3. h=3/16 gives eta0>1 yet h<1/4.
        for fun in (ps.predicate, vs.predicate):
            self.assertEqual(fun("RF0", F(3, 16), F(1), F(1), F(1), F(1), F(1), F(1, 10))[0], "ETA_DOMAIN_EXCLUDED")

    def test_K_zero_quartic_not_dropped(self):
        for fun in (ps.predicate, vs.predicate):
            self.assertEqual(fun("RF1", F(1, 8), F(1), F(1), F(0), F(1), F(1), F(1, 100))[0], "REJECT_PROBABILITY")

    def test_unknown_route_is_not_silently_EW2(self):
        for fun in (ps.predicate, vs.predicate):
            with self.assertRaises(ContractError):
                fun("UNKNOWN", F(1, 8), F(1), F(1), F(1), F(1), F(1), F(1, 10))

    def test_counter_not_restricted_to_RF_domain(self):
        for fun in (ps.predicate, vs.predicate):
            self.assertEqual(fun("CENTERED_CBAR2", F(1, 2), F(1), F(1), F(1), F(1, 1000), F(1), F(1, 10))[0], "ACCEPT")
            self.assertEqual(fun("RF0", F(1, 2), F(1), F(1), F(1), F(1, 1000), F(1), F(1, 10))[0], "DOMAIN_EDGE_OPEN")

    def test_zero_counter_exact_all_h(self):
        for fun in (ps.compare, vs.compare):
            record = fun(F(2), F(1), F(1), F(0), policy())
            self.assertEqual(record["status"], "COMMUTATOR_ZERO_EXACT_S2")
            self.assertEqual(record["step_comparison"], "NO_RELATIVE_ADVANTAGE_OVER_EXACT_COUNTER")
            self.assertFalse(record["routes"][-1]["endpoint_attained"])

    def test_zero_lambda_and_width_no_division(self):
        for fun in (ps.compare, vs.compare):
            for lam, width in [(F(0), F(2)), (F(2), F(0)), (F(0), F(0))]:
                with self.subTest(fun=fun.__module__, lam=str(lam), width=str(width)):
                    self.assertEqual(fun(lam, width, F(0), F(0), policy())["status"], "TRIVIAL_KNOWN_ENERGY")

    def test_invalid_policy(self):
        for changes in [{"d": rat(0)}, {"d": rat(-1)}, {"alpha_spec": rat(0)}, {"alpha_spec": rat(1)},
                        {"alpha_spec": rat(-1)}, {"root_width": rat(0)}, {"max_bisections": "0"}, {"max_bisections": "-1"},
                        {"unknown": "x"}, {"alpha_spec": 0.05}]:
            with self.subTest(changes=changes), self.assertRaises(ContractError):
                policy_validate(policy(**changes))

    def test_negative_certified_bounds(self):
        for fun in (ps.compare, vs.compare):
            for i in range(4):
                args = [F(1)] * 4
                args[i] = F(-1)
                with self.subTest(fun=fun.__module__, i=i), self.assertRaises(ContractError):
                    fun(*args, policy())

    def test_budget_exhaustion_no_retry(self):
        p = policy(root_width=rat(F(1, 2 ** 128)), max_bisections="1")
        case = prepare(pol=p)
        report = checked(case, pol=p)
        self.assertEqual(report["step_comparison"]["status"], "UNRESOLVED_ARITHMETIC")
        self.assertTrue(all(r["bisections"] == "1" for r in report["step_comparison"]["routes"]))

    def test_overlapping_brackets_do_not_claim_equality(self):
        p = policy(root_width=rat(8))
        rec = ps.compare(F(1), F(1), F(1), F(1), p)
        self.assertEqual(rec["step_comparison"], "ORDER_UNRESOLVED")
        self.assertIsNone(rec["routes"][0]["accepted_probe"])
        self.assertEqual(rec, vs.compare(F(1), F(1), F(1), F(1), p))

    def test_exact_rational_probability_endpoint(self):
        # Counter root h=1/2: b=(1/4)/(3/4)=1/3, alpha=1/9, d=1.
        p = policy(d=rat(1), alpha_spec=rat(F(1, 9)))
        a = ps.compare(F(1), F(1), F(1), F(1), p)
        b = vs.compare(F(1), F(1), F(1), F(1), p)
        self.assertEqual(a, b)
        self.assertEqual(a["routes"][-1]["endpoint_bracket"], {"lo": rat(F(1, 2)), "hi": rat(F(1, 2))})
        self.assertTrue(a["routes"][-1]["endpoint_attained"])

    def test_equal_requires_exact_shared_endpoint(self):
        p = policy(d=rat(1), alpha_spec=rat(F(1, 36)))
        args = (F(1), F(1), F(1184, 45), F(124, 3), p)
        record = ps.compare(*args)
        self.assertEqual(record, vs.compare(*args))
        self.assertEqual(record["step_comparison"], "EQUAL_SUFFICIENT_STEP")
        for route in record["routes"][2:]:
            self.assertEqual(route["endpoint_bracket"], {"lo": rat(F(1, 16)), "hi": rat(F(1, 16))})

    def test_both_order_directions_and_no_best_envelope(self):
        for cbar, expected in [(F(1, 100000), "COUNTER_LARGER_SUFFICIENT_STEP"),
                               (F(100000), "RELATIVE_LARGER_SUFFICIENT_STEP_AGAINST_DECLARED_COUNTER")]:
            with self.subTest(cbar=str(cbar)):
                args = (F(1), F(1), F(1), cbar, policy())
                self.assertEqual(ps.compare(*args), vs.compare(*args))
                self.assertEqual(ps.compare(*args)["step_comparison"], expected)

    def test_monotone_acceptance_on_exact_grid(self):
        for route in ps.ROUTES:
            seen_reject = False
            for k in range(1, 65):
                status, _ = vs.predicate(route, F(k, 64), F(1), F(2), F(3), F(4), F(1, 2), F(1, 10))
                with self.subTest(route=route, k=k):
                    if status == "ACCEPT": self.assertFalse(seen_reject)
                    else: seen_reject = True

    def test_checker_uses_no_producer_step_code(self):
        case = prepare(pol=policy())
        with patch.object(ps, "predicate", side_effect=AssertionError("not checker")), patch.object(ps, "compare", side_effect=AssertionError("not checker")):
            self.assertEqual(checked(case, pol=policy())["verdict"], "SPEC04_07_VERIFIED_EXACT_SYNTHETIC")


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = prepare(pol=policy())

    def test_json_roundtrip_and_no_input_mutation(self):
        case = copy.deepcopy(self.case)
        before = canonical_bytes({"raw": case[0], "base": case[1], "evidence": case[2]})
        raw, base, e = case
        report = check_spec04_07(canonical_bytes(raw).decode(), canonical_bytes(base).decode(), canonical_bytes(e).decode(),
                                 expected_particle_number="1", expected_occupied_modes=["0"], expected_sqrt_bits="16", expected_comparison_policy=policy())
        self.assertEqual(report["verdict"], "SPEC04_07_VERIFIED_EXACT_SYNTHETIC")
        self.assertEqual(before, canonical_bytes({"raw": case[0], "base": case[1], "evidence": case[2]}))

    def test_each_required_record_missing(self):
        for key in self.case[2]["payload"]:
            with self.subTest(key=key):
                case = copy.deepcopy(self.case)
                del case[2]["payload"][key]
                self.assertNotEqual(checked(case, pol=policy())["verdict"], "SPEC04_07_VERIFIED_EXACT_SYNTHETIC")

    def test_raw_bit_change_and_base_tamper_rejected(self):
        for path in ["raw", "base"]:
            with self.subTest(path=path):
                case = copy.deepcopy(self.case)
                if path == "raw": case[0]["arrays"]["constant"] = old.array([3.], [], [])
                else: case[1]["payload"]["lower_certificate"]["payload"]["L"] = rat(999)
                self.assertEqual(checked(case, pol=policy())["stage"], "SPEC01_03_PREREQUISITE")

    def test_identity_mismatch_incomparable(self):
        case = copy.deepcopy(self.case)
        case[2]["payload"]["context"]["model_id"] = "f" * 64
        rehash(case[2])
        self.assertEqual(checked(case, pol=policy())["verdict"], "IDENTITY_MISMATCH_INCOMPARABLE")
        self.assertEqual(checked(self.case, pol=policy(alpha_spec=rat(F(1, 20))))["verdict"], "IDENTITY_MISMATCH_INCOMPARABLE")

    def test_wrong_expected_model_and_order(self):
        raw, base, e = self.case
        kwargs = dict(expected_particle_number="1", expected_occupied_modes=["0"], expected_sqrt_bits="16", expected_comparison_policy=policy())
        self.assertEqual(check_spec04_07(raw, base, e, **kwargs, expected_model_id="0" * 64)["verdict"], "IDENTITY_MISMATCH_INCOMPARABLE")
        order = list(reversed([f["fragment_id"] for f in base["payload"]["partition"]["payload"]["fragments"]]))
        self.assertNotEqual(check_spec04_07(raw, base, e, **kwargs, expected_fragment_order=order)["verdict"], "SPEC04_07_VERIFIED_EXACT_SYNTHETIC")

    def test_upper_value_column_and_context_mutations(self):
        mutations = [("U", rat(999)), ("W", rat(-1)), ("L", rat(0)), ("unit", "DIMENSIONLESS"),
                     ("constant", rat(0)), ("q_expectation", rat(0)), ("columns", []), ("square_contributions", [])]
        for key, value in mutations:
            with self.subTest(key=key):
                case = copy.deepcopy(self.case)
                data(case[2], "upper_certificate")[key] = value
                rehash(case[2])
                report = checked(case, pol=policy())
                self.assertEqual(report["stage"], "SPEC04_UPPER")
                self.assertNotEqual(report["verdict"], "SPEC04_07_VERIFIED_EXACT_SYNTHETIC")

    def test_signed_lambda_not_abs(self):
        case = copy.deepcopy(self.case)
        term = data(case[2], "upper_certificate")["square_contributions"][1]
        term["lambda"] = rat(abs(read_rat(term["lambda"])))
        term["weighted_expectation"] = rat(abs(read_rat(term["weighted_expectation"])))
        rehash(case[2])
        self.assertEqual(checked(case, pol=policy())["stage"], "SPEC04_UPPER")

    def test_k_dictionary_row_cap_mutations(self):
        for change in ["sign", "omit", "rows", "maximum", "raw_count", "sqrt_digest", "positions", "unit"]:
            with self.subTest(change=change):
                case = copy.deepcopy(self.case)
                k = data(case[2], "ksos_certificate")
                if change == "sign": k["dictionary"][0]["kappa"] = rat(-read_rat(k["dictionary"][0]["kappa"]))
                elif change == "omit": k["dictionary"].pop()
                elif change == "rows": k["row_sums_upper"][0] = rat(0)
                elif change == "maximum": k["K_SOS_upper"] = rat(0)
                elif change == "positions": k["position_to_fragment"].reverse()
                elif change == "raw_count": k["raw_count"] = "0"
                elif change == "unit": k["unit"] = "HA"
                else: k["sqrt_witness_digest"] = "0" * 64
                rehash(case[2])
                self.assertEqual(checked(case, pol=policy())["stage"], "SPEC05_KSOS")

    def test_commutator_structure_center_weights_mutations(self):
        for change in ["Cbar", "cap", "selected", "sign", "support", "imag", "remove", "weights", "tail"]:
            with self.subTest(change=change):
                case = copy.deepcopy(self.case)
                cb = data(case[2], "commutator_certificate")
                group = next(g for g in cb["groups"] if g["first"]["pauli"])
                if change == "Cbar": cb["Cbar2"] = rat(0)
                elif change == "cap": group["first"]["center_cap"] = rat(0)
                elif change == "selected": group["first"]["selected"] = rat(0)
                elif change == "sign": group["first"]["pauli"][0]["coefficient"]["re"] = rat(999)
                elif change == "support": group["first"]["pauli"][0]["pauli_mode0_left"] = "II"
                elif change == "imag": group["first"]["pauli"][0]["coefficient"]["im"] = rat(1)
                elif change == "remove": group["first"]["pauli"].pop()
                elif change == "weights": cb["weights"]["first"], cb["weights"]["second"] = cb["weights"]["second"], cb["weights"]["first"]
                else: group["tail_fragment_ids"] = []
                rehash(case[2])
                self.assertEqual(checked(case, pol=policy())["stage"], "SPEC06_COMMUTATOR")

    def test_step_probe_bracket_policy_flags_mutations(self):
        for change in ["bracket", "edge", "probe", "attained", "order", "status", "state", "ground"]:
            with self.subTest(change=change):
                case = copy.deepcopy(self.case)
                cp = case[2]["payload"]["step_comparison"]["payload"]
                row = cp["data"]["routes"][0]
                if change == "bracket": row["endpoint_bracket"]["lo"] = rat(1)
                elif change == "edge": row["domain_edge"] = rat(1)
                elif change == "probe": row["probes"][0]["status"] = "ACCEPT"
                elif change == "attained": row["endpoint_attained"] = True
                elif change == "order": cp["data"]["step_comparison"] = "SCIENTIFIC_GO"
                elif change == "status": cp["data"]["status"] = "CERTIFIED_PLATFORM"
                elif change == "state": cp["conditional_on_exact_ground_access"] = False
                else: cp["trial_is_ground_state"] = True
                rehash(case[2])
                self.assertEqual(checked(case, pol=policy())["stage"], "SPEC07_STEPS")

    def test_hash_and_source_and_limit_mutations(self):
        for change in ["hash", "source", "adoption", "cost", "unknown"]:
            with self.subTest(change=change):
                case = copy.deepcopy(self.case)
                e = case[2]
                if change == "hash": e["id"] = "0" * 64
                elif change == "source": e["payload"]["source"]["version"] = "another"
                elif change == "adoption": e["payload"]["limitations"]["A2_status"] = "ADOPTED"
                elif change == "cost": e["payload"]["cost_ledger"]["checker_cost"] = rat(0)
                else: e["payload"]["scientific_go"] = True
                self.assertNotEqual(checked(case, pol=policy())["verdict"], "SPEC04_07_VERIFIED_EXACT_SYNTHETIC")

    def test_malformed_strict_encodings(self):
        raw, base, _ = self.case
        for evidence in ['{"id":"a","id":"b"}', '{"payload":1}', 'null', '{}', [], None, {"id": "0" * 64, "payload": []}]:
            with self.subTest(evidence=str(evidence)):
                self.assertNotEqual(check_spec04_07(raw, base, evidence, expected_particle_number="1", expected_occupied_modes=["0"], expected_sqrt_bits="16", expected_comparison_policy=policy())["verdict"], "SPEC04_07_VERIFIED_EXACT_SYNTHETIC")

    def test_scientific_tag_is_rejected_not_relabelled(self):
        raw = old.fixture()
        raw["provenance"]["input_class"] = "MOLECULAR_SCIENTIFIC"
        self.assertNotEqual(generate_spec01_03(raw, particle_number="1")["verdict"], "EVIDENCE_GENERATED_UNVERIFIED")

    def test_source_inventory_and_no_authority(self):
        source = source_identity()
        self.assertEqual(len(source["modules"]), 8)
        report = checked(self.case, pol=policy())
        self.assertEqual(report["limitations"]["A2_status"], "NOT_ADOPTED")
        self.assertEqual(report["limitations"]["A1_status"], "CLOSED")
        self.assertEqual(report["limitations"]["K1_F_status"], "UNAPPROVED")
        self.assertFalse(report["limitations"]["execution_authorization_issued"])

    def test_io_failure_preserves_partial_proposals(self):
        raw, base, _ = self.case
        with patch.object(generator, "source_identity", side_effect=OSError("artificial source read failure")):
            out = generate_spec04_07(raw, base, particle_number="1", occupied_modes=["0"], sqrt_bits="16", comparison_policy=policy())
        self.assertEqual(out["verdict"], "TECHNICAL_ERROR_OR_INTERRUPTED")
        self.assertIn("upper_certificate", out["partial"])
        self.assertIn("ksos_certificate", out["partial"])
        self.assertIn("commutator_certificate", out["partial"])
        self.assertEqual(out["verified_quantities"], [])


if __name__ == "__main__":
    unittest.main()
