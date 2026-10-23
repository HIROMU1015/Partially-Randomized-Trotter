"""Static saved-record joins, artificial rejection cases and frozen T regressions.

No saved endpoint is evaluated/ordered/added, and no scientific code is imported.
"""
from copy import deepcopy
import builtins
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("binding_tests_module", ROOT / "src/trotterlib/h4_evidence_binding.py")
b = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(b)
MANIFEST = ROOT / "artifacts/non_df_h4_proof_binding/2026-10-23/evidence_manifest.json"


class StaticBindings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = b.T.read_json_bytes(MANIFEST.read_bytes())
        cls.store = b.EvidenceStore(ROOT, cls.manifest)
        cls.output = b.produce_bindings(cls.store)

    def test_40_axes_14_compatible_gates(self):
        self.assertEqual(len(self.output["rows"]), 40)
        for row in self.output["rows"]:
            self.assertEqual(set(row["gates"]), set(b.T.GATE_SCOPES))
            for gate, record in row["gates"].items():
                self.assertEqual(record["scope"], b.T.GATE_SCOPES[gate])
                self.assertIn(record["proof_status"], b.T.CLAIMS)

    def test_produce_verify_roundtrip(self):
        result = b.verify_bindings(self.store, self.output)
        self.assertEqual(result["axes"], 40)
        self.assertFalse(result["scientific_certificate_accepted"])

    def test_typed_endpoints_are_verbatim_not_new_bounds(self):
        gate = self.output["rows"][0]["gates"]["numerical_enclosure"]
        for fact in gate["facts"]:
            self.assertEqual(fact, self.store.fact(fact["role"], fact["pointer"]))
        self.assertEqual(gate["proof_status"], "EVIDENCE_EXISTS_UNCHECKED")
        self.assertFalse(gate["numerical_inequality_evaluated"])

    def test_endpoint_sign_order_not_silently_analysed(self):
        # Deliberately reversed/negative artificial values remain valid tokens.
        # This structural checker must not pretend to prove the inequality.
        b.endpoint_tokens(dict(lower="99", upper="-1/2"))

    def test_bad_endpoint_schema_rejected(self):
        for value in (dict(lower=0.1, upper="1/2"), dict(lower="nan", upper="1"),
                      dict(lower="0", upper="1/0"), dict(lower="0", upper="1", midpoint="0")):
            with self.assertRaises(b.T.GateValidationError):
                b.endpoint_tokens(value)

    def test_actual_reference_is_not_recomputed(self):
        gate = self.output["rows"][0]["gates"]["reference_encoded_full_H"]
        self.assertEqual(gate["proof_status"], "EVIDENCE_EXISTS_UNCHECKED")
        self.assertTrue(gate["obligations"])

    def test_DF_does_not_assume_unitarity(self):
        for row in self.output["rows"]:
            if row["target"]["case"].startswith("df_native"):
                gate = row["gates"]["encoded_event_norm"]
                self.assertEqual(gate["proof_status"], "MISSING_EVIDENCE")
                self.assertIsNone(gate["lemma"])

    def test_binary_range_does_not_close_wrapper(self):
        for row in self.output["rows"]:
            self.assertEqual(row["gates"]["measurement_binary_range"]["proof_status"], "VERIFIED")
            self.assertEqual(row["gates"]["wrapper_transfer"]["proof_status"], "MISSING_EVIDENCE")

    def test_PF_normalization_and_PR_saved_numeric_distinct(self):
        for row in self.output["rows"]:
            gate = row["gates"]["normalization_nominal_log_B"]
            expected = "VERIFIED" if row["target"]["case"].endswith("_PF") else "EVIDENCE_EXISTS_UNCHECKED"
            self.assertEqual(gate["proof_status"], expected)

    def test_compiler_blocked_for_all40(self):
        for row in self.output["rows"]:
            self.assertEqual(row["gates"]["compiler_transfer"]["proof_status"], "MISSING_EVIDENCE")
            self.assertEqual(row["target"]["compiler_target"], "HISTORICAL_PROGRAM")
            self.assertIsNone(row["target"]["compiled_program_sha256"])

    def test_scope_tampering_rejected(self):
        changed = deepcopy(self.output)
        changed["rows"][0]["gates"]["encoded_event_norm"]["scope"] = "BINARY_MEASUREMENT"
        with self.assertRaisesRegex(b.T.GateValidationError, "RECORD_OR_SCOPE_OR_TARGET"):
            b.verify_bindings(self.store, changed)

    def test_cross_axis_proof_reuse_rejected(self):
        changed = deepcopy(self.output)
        changed["rows"][1]["gates"]["wrapper_transfer"] = changed["rows"][0]["gates"]["wrapper_transfer"]
        with self.assertRaises(b.T.GateValidationError):
            b.verify_bindings(self.store, changed)

    def test_cross_case_proof_reuse_rejected(self):
        changed = deepcopy(self.output)
        changed["rows"][10]["gates"] = changed["rows"][0]["gates"]
        with self.assertRaises(b.T.GateValidationError):
            b.verify_bindings(self.store, changed)

    def test_missing_gate_proof_rejected(self):
        changed = deepcopy(self.output)
        del changed["rows"][0]["gates"]["bias_encoded_saved"]
        with self.assertRaises(b.T.GateValidationError):
            b.verify_bindings(self.store, changed)

    def test_mathematical_acceptance_cannot_be_injected(self):
        for key in ("formal_certificate_accepted", "execution_authorized", "grant_valid", "dispatch_enabled", "N_G_authorized"):
            changed = deepcopy(self.output)
            changed[key] = True
            with self.assertRaises(b.T.GateValidationError):
                b.verify_bindings(self.store, changed)

    def test_manifest_scientific_flags_rejected(self):
        for key in ("execution_authorized", "grant_valid", "dispatch_enabled", "formal_certificate_accepted", "N_G_authorized"):
            changed = deepcopy(self.manifest); changed[key] = True
            with self.assertRaisesRegex(b.T.GateValidationError, "NO_AUTHORITY"):
                b.validate_manifest(changed)

    def test_artificial_manifest_mixed_into_H4_rejected(self):
        changed = deepcopy(self.manifest);changed["domain"] = "ARTIFICIAL"
        with self.assertRaisesRegex(b.T.GateValidationError, "H4_DOMAIN"):
            b.validate_manifest(changed)

    def test_artificial_reference_mixed_into_H4_rejected(self):
        changed = deepcopy(self.manifest);changed["files"]["pauli_jw_result"]["domain"] = "ARTIFICIAL"
        with self.assertRaisesRegex(b.T.GateValidationError, "ARTIFICIAL_EVIDENCE"):
            b.validate_manifest(changed)

    def test_duplicate_reference_path_rejected(self):
        changed = deepcopy(self.manifest)
        changed["files"]["duplicate"] = changed["files"]["C_seal"]
        with self.assertRaisesRegex(b.T.GateValidationError, "DUPLICATE_EVIDENCE_PATH"):
            b.validate_manifest(changed)

    def test_manifest_SOURCE_mismatch_rejected(self):
        changed = deepcopy(self.manifest);changed["commits"]["C"] = "f" * 40
        with self.assertRaisesRegex(b.T.GateValidationError, "FIXED_SOURCE"):
            b.validate_manifest(changed)

    def test_manifest_file_SHA_mismatch_rejected(self):
        changed = deepcopy(self.manifest);changed["files"]["T_gate_source"]["sha256"] = "f" * 64
        with self.assertRaisesRegex(b.T.GateValidationError, "HASH_MISMATCH"):
            b.EvidenceStore(ROOT, changed)

    def test_provenance_binding_does_not_claim_old_native_PASS(self):
        self.assertNotIn("native_PASS", self.output)
        self.assertEqual(self.output["rows"][0]["gates"]["provenance"]["proof_status"], "VERIFIED")

    def test_error_semantic_term_already_covered(self):
        row = self.output["rows"][0]
        self.assertEqual(b.validate_error_terms(row["error_terms"]), ["C_bias_distance", "C_circuit_semantic_allowance"])

    def test_double_counted_semantic_allowance_rejected(self):
        terms = deepcopy(self.output["rows"][0]["error_terms"])
        terms.append(dict(id="duplicate_semantic", primitive_terms=["C_circuit_semantic_allowance"],
                          fact={}, newly_evaluated=False))
        with self.assertRaisesRegex(b.T.GateValidationError, "DOUBLE_COUNTED"):
            b.validate_error_terms(terms)

    def test_duplicate_error_proof_rejected(self):
        terms = deepcopy(self.output["rows"][0]["error_terms"])
        terms.append(terms[0])
        with self.assertRaisesRegex(b.T.GateValidationError, "DUPLICATE_ERROR"):
            b.validate_error_terms(terms)

    def test_new_upper_evaluation_rejected(self):
        terms = deepcopy(self.output["rows"][0]["error_terms"])
        terms[0]["newly_evaluated"] = True
        with self.assertRaisesRegex(b.T.GateValidationError, "NEW_BOUND"):
            b.validate_error_terms(terms)

    def test_missing_error_evidence_rejected(self):
        with self.assertRaisesRegex(b.T.GateValidationError, "ERROR_TERMS_REQUIRED"):
            b.validate_error_terms([])

    def test_IID_condition_retained(self):
        for row in self.output["rows"]:
            gate = row["gates"]["conditional_statistics"]
            self.assertEqual(gate["proof_status"], "ASSUMPTION_DEPENDENT")
            self.assertIn("FRESH_INDEPENDENT_SHOTS", gate["assumptions"])
            self.assertNotIn("PCG64_UNCONDITIONAL_INDEPENDENCE", gate["assumptions"])

    def test_unsupported_compiler_pass_rejected(self):
        with self.assertRaisesRegex(b.T.GateValidationError, "UNSUPPORTED_COMPILER_PASS"):
            b.inspect_pass_proofs([dict(pass_instance="Euler", rule="FLOAT_EULER", proof={})], ["Euler"])

    def test_missing_compiler_pass_stays_missing(self):
        result = b.inspect_pass_proofs([], ["BasisTranslator", "Optimize1qGatesDecomposition"])
        self.assertEqual(len(result["missing_pass_instances"]), 2)
        self.assertFalse(result["whole_compiler_equivalence_proved"])

    def test_word_equality_does_not_prove_whole_compiler(self):
        w = b.T.identity_word(1)
        proof = dict(pass_instance="inverse0", rule="WORD_EQUALITY",
                     proof=dict(rule="WORD_EQUALITY", data=dict(before=w, after=w)))
        result = b.inspect_pass_proofs([proof], ["inverse0"])
        self.assertEqual(result["missing_pass_instances"], [])
        self.assertEqual(result["pass_to_historical_IR_binding"], "NOT_PROVED")
        self.assertFalse(result["whole_compiler_equivalence_proved"])

    def test_duplicate_pass_proof_rejected(self):
        w = b.T.identity_word(1)
        p = dict(pass_instance="p", rule="WORD_EQUALITY", proof=dict(rule="WORD_EQUALITY", data=dict(before=w, after=w)))
        with self.assertRaisesRegex(b.T.GateValidationError, "DUPLICATE_PASS"):
            b.inspect_pass_proofs([p, p], ["p"])

    def test_no_scientific_imports(self):
        original = builtins.__import__
        def guard(name, *args, **kwargs):
            self.assertNotIn(name.split(".")[0], {"numpy", "qiskit", "scipy", "flint", "pyscf", "openfermion", "trotterlib"})
            return original(name, *args, **kwargs)
        spec = importlib.util.spec_from_file_location("fresh_binding_test", ROOT / "src/trotterlib/h4_evidence_binding.py")
        with patch("builtins.__import__", guard):
            module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


if __name__ == "__main__":
    unittest.main(verbosity=2)
