"""Artificial-only tests for metadata copying and post-approval decoding."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "tests")]
import bind_pr_alpha_saved_inputs as binder
import pr_alpha_followup_runtime as runtime
from test_pr_alpha_followup import toy

codec = binder.codec


def fixture(method="B2"):
    r = toy(method)
    signal = dict(cell=dict(id=r["cell_id"], method=method), signals=dict(corrected=dict(real=0., imag=0.)),
        reference=dict(real=0., imag=0.), log_B=0.,
        allowance=dict(real=0., imag=0., log_B_margin=0., kind="EMPIRICAL", certified=False),
        error_decomposition=dict(total_signed=dict(real=-.125, imag=.25),
            signed_parts=dict(PF=dict(real=-.25, imag=.5), discard=dict(real=.125, imag=-.25))))
    if method == "B2":
        signal["signals"]["raw"] = dict(real=0., imag=0.)
        signal["error_decomposition"]["signed_parts"] = dict(outer_PF=dict(real=-.25, imag=.5), finite_RTE=dict(real=.125, imag=-.25))
    return dict(system="H6", cell_id=r["cell_id"], epsilon=r["epsilon"], mode="USUAL_SIGNED_SAVED",
        signal=signal, baseline=r["baseline_accounting"], cost=r["cost"], lineage=dict(signal="SYNTHETIC"))


class CodecTests(unittest.TestCase):
    def test_preparation_does_not_evaluate_scalar_recipes(self):
        f = fixture()
        with patch.object(codec.math, "exp", side_effect=AssertionError("exp forbidden")), \
             patch.object(codec, "abs", create=True, side_effect=AssertionError("abs forbidden")), \
             patch.object(codec, "sum", create=True, side_effect=AssertionError("sum forbidden")):
            out = codec.make_record(**f)
        self.assertNotIn("bias", out); self.assertNotIn("B_corr", out); self.assertNotIn("B_budget", out)
        self.assertEqual(out["baseline_accounting"], f["baseline"])

    def test_copy_is_not_alias_and_does_not_change_saved_N(self):
        f = fixture(); out = codec.make_record(**f)
        out["baseline_accounting"]["N_total"] = 123
        self.assertNotEqual(out["baseline_accounting"], f["baseline"])

    def test_usual_bias_decodes_abs_of_saved_total(self):
        r = codec.decode_record(codec.make_record(**fixture()))
        self.assertEqual(r["bias"], dict(real=.125, imag=.25))

    def test_noncancellation_sums_original_parts_after_approval(self):
        f = fixture(); f["mode"] = "EMPIRICAL_NONCANCELLATION"
        r = codec.decode_record(codec.make_record(**f))
        self.assertEqual(r["bias"], dict(real=.375, imag=.75))

    def test_saved_noncancellation_magnitude_is_copied(self):
        f = fixture(); f.update(mode="EMPIRICAL_NONCANCELLATION", noncancellation=dict(real=.4, imag=.8))
        r = codec.decode_record(codec.make_record(**f))
        self.assertEqual(r["bias"], dict(real=.4, imag=.8))

    def test_correction_and_budget_decode_separately(self):
        f = fixture(); f["signal"].update(log_B=math.log(2.)); f["signal"]["allowance"]["log_B_margin"] = math.log(1.5)
        r = codec.decode_binomial_normalization(codec.decode_record(codec.make_record(**f)))
        self.assertEqual(r["B_corr"], 2.); self.assertAlmostEqual(r["B_budget"], 3.)

    def test_B2_log_domain_does_not_evaluate_exp(self):
        f = fixture(); f["signal"]["log_B"] = 1000.
        with patch.object(codec.math, "exp", side_effect=AssertionError("exp forbidden")):
            r = codec.decode_record(codec.make_record(**f))
        self.assertNotIn("B_budget", r)

    def test_deterministic_raw_identity_requires_B_one(self):
        f = fixture("B1"); r = codec.make_record(**f)
        self.assertEqual(r["raw"], r["corrected"])
        self.assertEqual(r["raw_policy"], "DETERMINISTIC_IDENTITY_B_EQUALS_ONE")
        f["signal"]["log_B"] = .1
        with self.assertRaisesRegex(ValueError, "MISSING_PR_RAW"): codec.make_record(**f)

    def test_missing_PR_raw_is_not_reconstructed(self):
        f = fixture(); del f["signal"]["signals"]["raw"]
        with self.assertRaisesRegex(ValueError, "MISSING_PR_RAW"): codec.make_record(**f)

    def test_missing_baseline_or_signed_parts_stays_missing(self):
        f = fixture(); f["baseline"] = None
        self.assertIn("saved_baseline_accounting", codec.decode_record(codec.make_record(**f))["missing_fields"])
        f = fixture(); f["mode"] = "EMPIRICAL_NONCANCELLATION"; f["signal"]["error_decomposition"]["signed_parts"] = None
        self.assertIn("saved_signed_decomposition", codec.make_record(**f)["missing_fields"])

    def test_unknown_recipe_and_nonfinite_log_rejected(self):
        r = codec.make_record(**fixture()); r["bias_recipe"]["kind"] = "UNAPPROVED_OPTIMIZATION"
        with self.assertRaisesRegex(ValueError, "UNKNOWN_BIAS_RECIPE"): codec.decode_record(r)
        r = codec.make_record(**fixture()); r["log_B"] = math.nan
        with self.assertRaisesRegex(ValueError, "INVALID_SAVED_LOG"): codec.decode_record(r)


class CopyBindingTests(unittest.TestCase):
    def rows(self):
        return [dict(cell_id="B2_SYNTHETIC", phase="confirmation", control="symmetric_directional", replica=str(i),
            seed=str(i+1), axis=axis, event_digest=str(i), compiler_hash="a"*64, RZ=str(i+2))
            for i in range(32) for axis in ("cosine", "sine")]

    def test_paired_cost_is_copied_without_average(self):
        r = binder.pair_costs(self.rows(), ["B2_SYNTHETIC"])["B2_SYNTHETIC"]
        self.assertEqual(r["sample_count"], 32); self.assertEqual(r["pairs"][0]["real"], 2)
        self.assertNotIn("mean", r); self.assertNotIn("SE", r)

    def test_pair_identity_conflict_fails_closed(self):
        rows = self.rows(); rows[0]["seed"] = "999"
        with self.assertRaisesRegex(ValueError, "PAIRED_PROVENANCE"): binder.pair_costs(rows, ["B2_SYNTHETIC"])

    def test_measured_model_missing_costs_are_preserved(self):
        r = binder.deterministic_cost(dict(cosine=dict(measured=True, RZ="2"), sine=dict(measured=False, RZ="3/2")))
        self.assertEqual(r["kind"], "DETERMINISTIC_MIXED"); self.assertEqual(r["imag"], "3/2")
        self.assertEqual(binder.deterministic_cost(None)["kind"], "MISSING")

    def test_aggregate_PR_means_do_not_enter_PF_unit_index(self):
        pf = dict(cell_id="B1_SYNTHETIC", axis="cosine", RZ="2", measured=True)
        pr = dict(cell_id="B2_SYNTHETIC", axis="cosine", RZ="3", measured=True)
        units, _ = binder.deterministic_axis_index([pf, pr, dict(pr, RZ="4")])
        self.assertEqual(set(units), {("H6", "B1_SYNTHETIC")})
        with self.assertRaisesRegex(ValueError, "DUPLICATE_H6_UNIT_AXIS"):
            binder.deterministic_axis_index([pf, pf])

    def test_equal_unit_prices_keep_mode_specific_metadata(self):
        a = dict(cosine=dict(RZ="2", measured=True, status="USUAL"), sine=dict(RZ="3", measured=False, model_formula="saved"))
        b = deepcopy(a); b["cosine"]["status"] = "NONCANCELLATION"
        self.assertEqual(binder.unit_signature(a), binder.unit_signature(b))
        self.assertNotEqual(binder.deterministic_cost(a)["original_axis_costs"], binder.deterministic_cost(b)["original_axis_costs"])
        b["cosine"]["RZ"] = "99"
        self.assertNotEqual(binder.unit_signature(a), binder.unit_signature(b))

    def test_string_and_numeric_factor_encoding_join_equally(self):
        for value in (1, 1., "1", "1.0"):
            self.assertTrue(binder.factor_one(value))
        for value in (10, 10., "10", "10.0"):
            self.assertFalse(binder.factor_one(value))
        for value in (True, "nan", "2"):
            with self.assertRaises(ValueError): binder.factor_one(value)

    def test_stage_dependencies_stay_independent(self):
        closure = [dict(path=p) for p in binder.EXECUTION_FILES]
        b1 = {r['path'] for r in binder.stage_source_files(closure, 'B1')}
        b2 = {r['path'] for r in binder.stage_source_files(closure, 'B2')}
        self.assertNotIn('scripts/run_pr_alpha_sensitivity.py', b1)
        self.assertNotIn('scripts/run_pr_alpha_binomial.py', b2)
        original = dict(python_version=sys.version.split()[0], python_executable_sha256=runtime.digest_bytes(Path(sys.executable).read_bytes()), packages=[dict(name='SYNTHETIC_SCIPY')])
        profile = binder.stage_runtime_profile(original, 'B2')
        self.assertEqual(profile['packages'], [])
        with patch.object(runtime.importlib.metadata, 'distribution', side_effect=AssertionError('B2 has no SciPy dependency')):
            runtime.runtime_gate(profile)
        profile['stdlib_files'][0]['sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'USED_STDLIB_RUNTIME_CHANGED'): runtime.runtime_gate(profile)

    def test_private_input_requires_exact_existing_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "synthetic.json"; raw = b'{"fixture":"ARTIFICIAL"}'; p.write_bytes(raw)
            receipt = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()); store = binder.Store()
            store.private("SYNTHETIC", p, receipt); store.post_check()
            p.write_bytes(b'{}')
            with self.assertRaisesRegex(ValueError, "USED_PRIVATE_INPUT_CHANGED"): store.post_check()
            with self.assertRaisesRegex(ValueError, "FIXED_PRIVATE_BYTES"): store.private("SYNTHETIC", p, receipt)

    def test_V2_loader_only_decodes_after_admission(self):
        with tempfile.TemporaryDirectory() as d:
            r = codec.make_record(**fixture()); data = dict(schema=codec.SCHEMA, payload_kind="SYNTHETIC", records=[r],
                original_source_receipts=[dict(origin="synthetic://fixture", sha256="a"*64, projection_fields=["records"])])
            raw = runtime.canonical(data); p = Path(d)/"input.json"; p.write_bytes(raw)
            plan = dict(stage="B2", payload_kind="SYNTHETIC", input=dict(path=str(p), bytes=len(raw), sha256=runtime.digest_bytes(raw)))
            loaded = runtime.load_bound_input(plan)
            self.assertNotIn("bias", loaded["records"][0])
            decoded = runtime.decode_bound_records(plan, loaded)
            self.assertEqual(decoded[0]["bias"], dict(real=.125, imag=.25))
            with patch.object(runtime, "load_bound_input") as loader, self.assertRaises(ValueError):
                runtime.execute({}, {}, "B2", loader=loader)
            loader.assert_not_called()

    def test_recipes_decode_after_marker_consumed(self):
        from test_pr_alpha_followup import AdmissionTests
        with tempfile.TemporaryDirectory() as d:
            plan, grant = AdmissionTests().plan_grant(d, "B2")
            data = dict(schema=codec.SCHEMA, records=[codec.make_record(**fixture())])
            def calculate(plan, loaded):
                self.assertTrue(Path(grant["marker"]).is_file())
                self.assertEqual(runtime.decode_bound_records(plan, loaded)[0]["bias"]["real"], .125)
                return dict(fixture="ARTIFICIAL")
            result = runtime.execute(plan, grant, "B2", source_check=lambda _: None,
                                     loader=lambda _: data, calculator=calculate)
            self.assertFalse(result["real_scientific_stage_executed"])


if __name__ == "__main__":
    unittest.main()
