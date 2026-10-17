"""Artificial scalar/metadata tests only. Never load saved H4/DF/signal results."""
import copy
from decimal import Decimal, localcontext
from fractions import Fraction as F
import json
from multiprocessing import Pool
from pathlib import Path
import unittest
from unittest.mock import patch

from jsonschema import Draft202012Validator

from trottertracks.algorithm_codesign.m1s_stage_s_v1 import spec, control
from trottertracks.algorithm_codesign.m1s_stage_s_v1.arithmetic import Bounds, BoundLimit
from trottertracks.algorithm_codesign.m1s_stage_s_v1.kernels import (
    open_cap_integer, classify_integer, margin_at_cap, sufficient_at_cap, compare_cap)
from trottertracks.algorithm_codesign.m1s_stage_s_v1.adapter import RationalReader, keyed, prepare_saved
from trottertracks.algorithm_codesign.m1s_stage_s_v1.analysis import cell_job, worker_count, evaluate

ROOT = Path(__file__).resolve().parents[3]


def bounds():
    return Bounds(128, dict(spec.RESOURCES))


def decimal_sine(x):
    """Separate 100-digit Taylor reference for artificial small scalar arguments."""
    with localcontext() as ctx:
        ctx.prec = 100
        z = Decimal(x.numerator) / Decimal(x.denominator)
        term = total = z
        for k in range(1, 120):
            term = -term * z * z / Decimal((2*k)*(2*k+1))
            total += term
        return F(total)


class OpenEndpointTests(unittest.TestCase):
    def test_exact_integer_endpoint_requires_extra_one(self):
        self.assertEqual(open_cap_integer((F(7), F(7)))["value"], 8)

    def test_noninteger_endpoint(self):
        self.assertEqual(open_cap_integer((F(71, 10), F(71, 10)))["value"], 8)

    def test_upper_touching_integer_is_unresolved(self):
        self.assertFalse(open_cap_integer((F(69, 10), F(7)))["resolved"])

    def test_lower_integer_same_floor_is_resolved(self):
        self.assertEqual(open_cap_integer((F(7), F(71, 10)))["value"], 8)

    def test_interval_spanning_multiple_integers(self):
        self.assertEqual(open_cap_integer((F(5), F(8)))["interval"], [6, 9])

    def test_negative_or_reversed_rejected(self):
        for pair in [(F(-1), F(1)), (F(3), F(2))]:
            with self.subTest(pair=pair), self.assertRaises(ValueError):
                open_cap_integer(pair)

    def test_strict_witness(self):
        self.assertEqual(classify_integer(open_cap_integer((F(4), F(4))),
                                          open_cap_integer((F(5), F(5)))),
                         "CAP_EDGE_STRICT_INTEGER_WITNESS")

    def test_tie_is_not_global_impossibility(self):
        n = open_cap_integer((F(4), F(4)))
        self.assertEqual(classify_integer(n, n), "CAP_EDGE_NO_WITNESS_GLOBAL_UNDETERMINED")

    def test_reversed_order_is_only_cap_no_witness(self):
        self.assertEqual(classify_integer(open_cap_integer((F(5), F(5))),
                                          open_cap_integer((F(4), F(4)))),
                         "CAP_EDGE_NO_WITNESS_GLOBAL_UNDETERMINED")

    def test_even_separated_integer_intervals_remain_unresolved(self):
        self.assertEqual(classify_integer(open_cap_integer((F(2), F(3))),
                                          open_cap_integer((F(5), F(5)))),
                         "ARITHMETIC_UNRESOLVED")


class ArtificialArithmeticTests(unittest.TestCase):
    def test_outward_pack_contains_exact_fraction(self):
        B = bounds()
        for x in [F(1, 3), F(-1, 7), F(5, 17)]:
            lo, hi = B.pack((x, x))
            self.assertLessEqual(lo, x)
            self.assertGreaterEqual(hi, x)

    def test_sine_encloses_separate_decimal_reference(self):
        for x in [F(1, 16), F(1, 3), F(1), F(4, 3)]:
            lo, hi = bounds().sin_positive(x)
            ref = decimal_sine(x)
            self.assertLessEqual(lo, ref)
            self.assertGreaterEqual(hi, ref)

    def test_log_encloses_decimal_ln_on_artificial_scalars(self):
        for x in [F(3, 2), F(2), F(17), F(80)]:
            lo, hi = bounds().log_above_one(x)
            with localcontext() as ctx:
                ctx.prec = 100
                ref = F((Decimal(x.numerator)/Decimal(x.denominator)).ln())
            self.assertLessEqual(lo, ref)
            self.assertGreaterEqual(hi, ref)

    def test_sine_and_log_domains(self):
        for x in [F(-1), F(2)]:
            with self.assertRaises(ValueError):
                bounds().sin_positive(x)
        with self.assertRaises(ValueError):
            bounds().log_above_one(F(1))

    def test_instruction_and_series_caps(self):
        r = dict(spec.RESOURCES, max_bound_steps=1)
        B = Bounds(128, r)
        with self.assertRaises(BoundLimit):
            B.sin_positive(F(1))
        r = dict(spec.RESOURCES, max_series_terms_per_call=1)
        with self.assertRaises(BoundLimit):
            Bounds(128, r).sin_positive(F(1))

    def test_rational_bit_limit(self):
        with self.assertRaises(BoundLimit):
            Bounds(128, dict(spec.RESOURCES, max_rational_bits=4)).check(F(1, 100))

    def test_margin_interval_monotone_endpoints(self):
        cap = (F(1), F(1001, 1000))
        lo, hi = margin_at_cap(F(3, 4), F(1, 8), cap, bounds())
        self.assertLessEqual(lo, F(3, 4)*decimal_sine(cap[0]-F(1, 8)))
        self.assertGreaterEqual(hi, F(3, 4)*decimal_sine(cap[1]-F(1, 8)))

    def test_nonpositive_margin_or_invalid_cap_rejected(self):
        for A, bias, cap in [(F(0), F(0), (F(1), F(1))),
                             (F(1), F(1), (F(1), F(1))),
                             (F(1), F(0), (F(3, 2), F(3, 2)))]:
            with self.assertRaises(ValueError):
                margin_at_cap(A, bias, cap, bounds())

    def test_sufficient_interval_division_orientation(self):
        lo, hi = sufficient_at_cap((F(1, 2), F(3, 4)), (F(2), F(3)), bounds())
        self.assertLessEqual(lo, F(2)/F(3, 4)**2)
        self.assertGreaterEqual(hi, F(3)/F(1, 2)**2)

    def test_reversed_tradeoff_is_not_silently_swapped(self):
        with self.assertRaises(ValueError):
            compare_cap(F(1, 2), F(1, 8), F(3, 4), F(1, 16),
                        (F(1, 2), F(1, 2)), (F(1, 2), F(1, 2)), (F(5), F(5)), bounds())

    def test_ratio_orientation_and_fields(self):
        B = bounds()
        ms = margin_at_cap(F(1), F(1, 8), (F(1), F(1)), B)
        mc = margin_at_cap(F(3, 4), F(1, 16), (F(1), F(1)), B)
        out = compare_cap(F(1), F(1, 8), F(3, 4), F(1, 16), ms, mc, (F(5), F(5)), B)
        lo, hi = [F(x) for x in out["R_cont_cap_interval"]]
        ref = (decimal_sine(F(7, 8))/(F(3, 4)*decimal_sine(F(15, 16))))**2
        self.assertLessEqual(lo, ref)
        self.assertGreaterEqual(hi, ref)
        self.assertFalse(out["all_gamma_impossibility_established"])


class MetadataAndGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.candidate = json.loads((ROOT/spec.ART/"execution_contract_candidate.json").read_text())

    def test_candidate_is_unadopted(self):
        control.validate_contract(self.candidate)
        self.assertIsNone(self.candidate["selected_science"])

    def test_candidate_rejected_before_git_or_science(self):
        with patch.object(control, "git", side_effect=AssertionError("Git/input reached")):
            with self.assertRaises(control.ExecutionClosed):
                control.authorize(ROOT, spec.ART+"/execution_contract_candidate.json", None, None, None)

    def test_one_flag_cannot_open_candidate(self):
        for key in ["adopted", "coverage_model_adopted"]:
            c = copy.deepcopy(self.candidate)
            c[key] = True
            with self.assertRaises(control.ExecutionClosed):
                control.validate_contract(c, adopted=True)

    def test_formal_contract_still_requires_separate_authority(self):
        c = copy.deepcopy(self.candidate)
        c.update(adopted=True, coverage_model_adopted=True,
                 status="ADOPTED_FOR_SEPARATE_EXECUTION_APPROVAL",
                 selected_science=dict(spec.PROPOSED_SCIENCE))
        with patch.object(control, "read_json", return_value=c):
            with self.assertRaises(control.ExecutionClosed):
                control.authorize(ROOT, "nonexistent-contract.json", None, None, None)

    def test_run_adapter_and_evaluate_require_permit_first(self):
        with self.assertRaises(control.ExecutionClosed):
            control.run(ROOT, None)
        with self.assertRaises(control.ExecutionClosed):
            prepare_saved(None, None, None)
        with self.assertRaises(control.ExecutionClosed):
            evaluate(None, None, None, None, None)

    def test_output_path_escape_rejected(self):
        for p in ["/tmp/forbidden", "artifacts/../outside", "different-prefix/new"]:
            with self.assertRaises(control.ExecutionClosed):
                control.inside(ROOT, p, spec.OUTPUT_PREFIX)

    def test_rational_reader_rejects_unbounded_or_invalid_text(self):
        reader = RationalReader(dict(spec.RESOURCES, max_rational_text_chars=8))
        for text in ["1/0", "nan", "1.0", "1234567/11"]:
            with self.assertRaises(ValueError):
                reader.parse(text)

    def test_metadata_inventory_duplicates_and_missing(self):
        with self.assertRaises(ValueError):
            keyed([{"id": "a"}, {"id": "a"}], ["id"], [("a",), ("b",)])
        with self.assertRaises(ValueError):
            keyed([{"id": "a"}], ["id"], [("a",), ("b",)])

    def test_no_artificial_cpu_ram_or_worker_caps(self):
        for key in ["cpu_time_cap_seconds", "wall_time_cap_seconds", "cpu_core_cap",
                    "worker_cap", "gpu_cap", "memory_cap_bytes"]:
            self.assertIsNone(spec.RESOURCES[key])

    def test_worker_selection_follows_workload_and_affinity(self):
        with patch("os.sched_getaffinity", return_value=set(range(32))):
            self.assertEqual(worker_count(12), 12)
        with patch("os.sched_getaffinity", return_value={0, 1, 2}):
            self.assertEqual(worker_count(12), 3)

    def test_candidate_and_result_schemas_well_formed(self):
        for name in ["execution_contract_candidate.schema.json", "result_schema.json"]:
            Draft202012Validator.check_schema(json.loads((ROOT/spec.ART/name).read_text()))

    def test_candidate_schema_rejects_adoption_and_changed_alpha(self):
        v = Draft202012Validator(json.loads((ROOT/spec.ART/"execution_contract_candidate.schema.json").read_text()))
        self.assertTrue(v.is_valid(self.candidate))
        c = copy.deepcopy(self.candidate)
        c["adopted"] = True
        self.assertFalse(v.is_valid(c))
        c = copy.deepcopy(self.candidate)
        c["proposed_science"]["alpha_row"] = "unapproved"
        self.assertFalse(v.is_valid(c))

    def test_carry_rows_forbid_new_shot_values(self):
        schema = json.loads((ROOT/spec.ART/"result_schema.json").read_text())
        row = {"split": "G9", "R": 2, "delta": "1/8", "T": "1", "comparator": "ordinary",
               "old_pi3_gate_status": "G9_ABSOLUTE_GAIN_UNRESOLVED_NOT_TESTED",
               "old_absolute_contrast_status": "UNRESOLVED_ABSOLUTE_CONTRAST",
               "new_stage_s_status": "G9_ABSOLUTE_GAIN_UNRESOLVED_NOT_TESTED"}
        v = Draft202012Validator(schema["properties"]["comparisons"]["items"])
        self.assertTrue(v.is_valid(row))
        row["R_cont_cap_interval"] = ["1/1", "1/1"]
        self.assertFalse(v.is_valid(row))

    def test_runtime_result_validation_preserves_full_metadata_prefix(self):
        # Only an unevaluated metadata template, no mock scientific observations.
        schema = json.loads((ROOT/spec.ART/"result_schema.json").read_text())
        result = {"schema": "m1s_stage_s_v1_result", "status": "TECHNICAL_STOP",
                  "evidence": spec.EVIDENCE, "coverage_scope": "SAVED_CONDITIONAL_RADII_IDEAL_MEAN_ONLY",
                  "provenance": {}, "failure": None, "scientific_GO": None, "RPE_success": None,
                  "native_costs": None, "total_resources": None, "global_integer_impossibility": False,
                  "one_shot_consumed": True, "retry_count": 0, "mandatory_STOP": True,
                  "comparisons": [dict(zip(["split", "R", "delta", "T", "comparator"], k),
                                       old_pi3_gate_status=spec.old_status(k),
                                       old_absolute_contrast_status=None,
                                       new_stage_s_status="NOT_EVALUATED") for k in spec.contrast_keys()]}
        control.validate_result(result, schema)
        altered = copy.deepcopy(result)
        altered["comparisons"][1] = copy.deepcopy(altered["comparisons"][0])
        with self.assertRaises(ValueError):
            control.validate_result(altered, schema)
        altered = copy.deepcopy(result)
        altered["status"] = "STAGE_S_COMPLETE_AWAITING_USER_JUDGMENT"
        with self.assertRaises(Exception):
            control.validate_result(altered, schema)

    def test_parallel_and_serial_artificial_cell_equivalence(self):
        # Arbitrary scalar fixture; no H4 method rows or saved numerical inputs.
        methods = {"shared_pm": (F(1), F(1, 8)),
                   "ordinary": (F(3, 4), F(1, 16)),
                   "independent_pm": (F(2, 3), F(1, 32))}
        payloads = [(("TOY", i, "TOY", "TOY"), methods, (F(1), F(1)),
                     (F(5), F(5)), 128, dict(spec.RESOURCES)) for i in range(2)]
        serial = [cell_job(p) for p in payloads]
        with Pool(processes=2) as pool:
            parallel = pool.map(cell_job, payloads)
        self.assertEqual([x[:2] for x in serial], [x[:2] for x in parallel])


if __name__ == "__main__":
    unittest.main()
