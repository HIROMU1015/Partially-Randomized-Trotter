"""Artificial algebra/metadata tests only; no H4 numerical input evaluation."""
import builtins
from copy import deepcopy
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


g = load("gate_test_module", ROOT / "src/trotterlib/h4_certificate_gates.py")
cli = load("gate_cli_test_module", ROOT / "scripts/verify_h4_certificate_gates.py")


def word(*atoms, n=1, phase=0, phases=None):
    return dict(n=n, phase_quarters=phase, phase_symbols=phases or {}, factors=list(atoms))


def unitary(name="U", inverse=False):
    return dict(kind="UNITARY_SYMBOL", name=name, inverse=inverse)


def pauli(label="Z", inverse=False):
    return dict(kind="PAULI", label=label, inverse=inverse)


def rule(name, **data):
    return dict(rule=name, data=data)


def store(root, relative, value):
    p = root / relative
    p.parent.mkdir(parents=True, exist_ok=True)
    raw = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True).encode()
    p.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def bundle_fixture(root, case="df_native_PR"):
    hashes = {i: store(root, i + ".json", {"opaque": i})
              for i in ("audit_matrix", "C_result", "C_seal")}
    target = dict(case=case, epsilon_sig="0.001", axis="sine", C_SOURCE=g.C_SOURCE,
                  original_SOURCE=g.ORIGINAL_SOURCE, C_result_sha256=hashes["C_result"],
                  C_seal_sha256=hashes["C_seal"], audit_matrix_sha256=hashes["audit_matrix"],
                  compiler_target="UNRESOLVED", compiled_program_sha256=None)
    return dict(schema=g.SCHEMA, domain="ARTIFICIAL", target=target,
                assumptions=["FRESH_INDEPENDENT_SHOTS"] + (["IID_UNIFORM_WORDS"] if case.endswith("_PR") else []),
                evidence=[dict(id=i, path=i + ".json", sha256=h, source_commit=g.C_SOURCE)
                          for i, h in hashes.items()],
                gates={k: dict(claim="VERIFIED", scope=v,
                               evidence_ids=["audit_matrix"], lemma=None) for k, v in g.GATE_SCOPES.items()},
                execution_authorized=False, formal_certificate_accepted=False, N_G_authorized=False)


class SymbolicSemantics(unittest.TestCase):
    def test_exact_Gaussian_integer_Pauli_oracle(self):
        # Exhaustive small artificial words; all entries and operations are
        # Gaussian integers in {0,+/-1,+/-i}. No numerical matrix package.
        matrices = {"I": ((1, 0), (0, 1)), "X": ((0, 1), (1, 0)),
                    "Y": ((0, -1j), (1j, 0)), "Z": ((1, 0), (0, -1))}
        def multiply(a, b):
            return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(2))
                               for j in range(2)) for i in range(2))
        def evaluate(labels, phase):
            result = matrices["I"]
            for label in labels:
                result = multiply(result, matrices[label])
            return tuple(tuple((1j ** (phase % 4)) * x for x in row) for row in result)
        count = 0
        for length in range(5):
            for labels in itertools.product("XYZ", repeat=length):
                for phase in range(4):
                    w = word(*(pauli(label) for label in labels), phase=phase)
                    normal, _ = g.normal_form(w)
                    self.assertEqual(evaluate(labels, phase),
                                     evaluate([token[1] for token in normal[3]], normal[1]))
                    count += 1
        self.assertEqual(count, 484)

    def test_signed_pauli_is_unitary(self):
        r = g.check_symbolic_rule(rule("WORD_UNITARITY", word=word(pauli("XYZ"), n=3, phase=2)))
        self.assertEqual(r["assumptions"], [])
        self.assertFalse(r["formal_certificate_accepted"])

    def test_symbolic_unitary_keeps_hypothesis(self):
        r = g.check_symbolic_rule(rule("WORD_UNITARITY", word=word(unitary("DFbasis"))))
        self.assertEqual(r["assumptions"], ["EXACT_UNITARY_SYMBOL:DFbasis"])
        self.assertEqual(r["binding_to_saved_H4"], "NOT_PROVED")

    def test_rounded_matrix_not_promoted(self):
        with self.assertRaisesRegex(g.GateValidationError, "NO_MATRIX_UNITARITY"):
            g.normal_form(word(dict(kind="ROUNDED_MATRIX", inverse=False, hash="0" * 64)))

    def test_finite_polynomial_not_unitary_atom(self):
        with self.assertRaisesRegex(g.GateValidationError, "UNSUPPORTED_ATOM"):
            g.normal_form(word(dict(kind="FINITE_P5", inverse=False)))

    def test_complex_coefficient_not_unitary_atom(self):
        with self.assertRaisesRegex(g.GateValidationError, "UNSUPPORTED_ATOM"):
            g.normal_form(word(dict(kind="SCALAR_WEIGHT", inverse=False)))

    def test_adjoint_reverses_noncommuting_product(self):
        w = word(unitary("A"), unitary("B"), phases={"phi": 1}, phase=1)
        self.assertEqual(g.normal_form(g.product(g.adjoint_word(w), w))[0],
                         g.normal_form(word())[0])

    def test_unreversed_inverse_is_rejected(self):
        before = word(unitary("A", True), unitary("B", True), unitary("A"), unitary("B"))
        with self.assertRaisesRegex(g.GateValidationError, "NOT_DERIVED"):
            g.check_symbolic_rule(rule("WORD_EQUALITY", before=before, after=word()))

    def test_control_keeps_quarter_phase(self):
        evolution = word(pauli(), phase=1)
        r = g.check_symbolic_rule(rule("CONTROLLED_WORD", evolution=evolution,
                                      zero_branch=word(), one_branch=evolution))
        self.assertIn("WITH_PHASE", r["conclusion"])

    def test_control_rejects_dropped_global_phase(self):
        with self.assertRaisesRegex(g.GateValidationError, "NOT_DERIVED"):
            g.check_symbolic_rule(rule("CONTROLLED_WORD", evolution=word(pauli(), phase=2),
                                      zero_branch=word(), one_branch=word(pauli())))

    def test_control_rejects_wrong_zero_branch_phase(self):
        with self.assertRaisesRegex(g.GateValidationError, "NOT_DERIVED"):
            g.check_symbolic_rule(rule("CONTROLLED_WORD", evolution=word(pauli()),
                                      zero_branch=word(phase=2), one_branch=word(pauli())))

    def test_control_keeps_general_real_phase(self):
        evolution = word(phases={"phi": 1})
        r = g.check_symbolic_rule(rule("CONTROLLED_WORD", evolution=evolution,
                                      zero_branch=word(), one_branch=evolution))
        self.assertIn("REAL_PHASE:phi", r["assumptions"])

    def test_phase_is_exact_not_float_tolerance(self):
        with self.assertRaisesRegex(g.GateValidationError, "EXACT_PHASE"):
            g.normal_form(word(phase=1.0))

    def test_basis_control_cancels_only_exact_adjoint(self):
        basis, diagonal = word(unitary("B1"), unitary("B2")), word(pauli())
        active = g.product(g.adjoint_word(basis), diagonal, basis)
        r = g.check_symbolic_rule(rule("BASIS_CONTROL", basis=basis, diagonal=diagonal,
                                      zero_branch=word(), one_branch=active))
        self.assertEqual(len(r["assumptions"]), 2)

    def test_basis_control_wrong_active_branch_rejected(self):
        with self.assertRaisesRegex(g.GateValidationError, "NOT_DERIVED"):
            g.check_symbolic_rule(rule("BASIS_CONTROL", basis=word(unitary("B")),
                                      diagonal=word(pauli()), zero_branch=word(), one_branch=word(pauli())))

    def test_cosine_wrapper(self):
        r = g.check_symbolic_rule(rule("HADAMARD_WRAPPER", axis="cosine",
                                      operations=["H", "CONTROLLED_U", "H"]))
        self.assertIn("REAL_EXPECTATION", r["conclusion"])

    def test_sine_wrapper(self):
        r = g.check_symbolic_rule(rule("HADAMARD_WRAPPER", axis="sine",
                                      operations=["H", "CONTROLLED_U", "SDG", "H"]))
        self.assertIn("IMAG_EXPECTATION", r["conclusion"])

    def test_sine_wrong_S_sign(self):
        with self.assertRaisesRegex(g.GateValidationError, "PHASE_OR_ORDER"):
            g.check_symbolic_rule(rule("HADAMARD_WRAPPER", axis="sine",
                                      operations=["H", "CONTROLLED_U", "S", "H"]))

    def test_sine_wrong_operation_order(self):
        with self.assertRaisesRegex(g.GateValidationError, "PHASE_OR_ORDER"):
            g.check_symbolic_rule(rule("HADAMARD_WRAPPER", axis="sine",
                                      operations=["H", "CONTROLLED_U", "H", "SDG"]))

    def test_binary_readout_independent_of_DF_norm(self):
        r = g.check_symbolic_rule(rule("BINARY_READOUT", scale_symbol="B_saved",
                                      bit_values=[0, 1], estimator="B*(1-2*b)"))
        self.assertEqual(r["gate"], "measurement_binary_range")
        self.assertEqual(r["assumptions"], ["FINITE_NONNEGATIVE_SCALE:B_saved"])

    def test_binary_bool_is_not_exact_bit_schema(self):
        with self.assertRaisesRegex(g.GateValidationError, "BINARY_BITS"):
            g.check_symbolic_rule(rule("BINARY_READOUT", scale_symbol="B",
                                      bit_values=[False, True], estimator="B*(1-2*b)"))

    def test_binary_wrong_scaling_rejected(self):
        with self.assertRaisesRegex(g.GateValidationError, "ESTIMATOR"):
            g.check_symbolic_rule(rule("BINARY_READOUT", scale_symbol="B",
                                      bit_values=[0, 1], estimator="2*B*(1-2*b)"))

    def test_compiler_HH_and_RZ_inverse_rules(self):
        h = dict(kind="H", wire=0, inverse=False)
        rz = dict(kind="RZ", wire=0, angle="theta", inverse=False)
        before = word(h, h, rz, dict(rz, inverse=True))
        r = g.check_symbolic_rule(rule("WORD_EQUALITY", before=before, after=word()))
        self.assertEqual(r["assumptions"], ["REAL_ANGLE:theta"])

    def test_compiler_unsupported_valid_commutation_not_guessed(self):
        a, b = dict(kind="H", wire=0, inverse=False), dict(kind="H", wire=1, inverse=False)
        with self.assertRaisesRegex(g.GateValidationError, "NOT_DERIVED"):
            g.check_symbolic_rule(rule("WORD_EQUALITY", before=word(a, b, n=2), after=word(b, a, n=2)))

    def test_RTE_order_zero_two_four_formal_signed_products(self):
        for order in (0, 2, 4):
            event = word(*([pauli("X"), pauli("Z")] * (order // 2)), unitary("rotation"),
                         phase=order)
            self.assertEqual(g.check_symbolic_rule(rule("WORD_UNITARITY", word=event))["status"],
                             "FORMAL_LEMMA_CHECKED")

    def test_noncommuting_RTE_event_order_rejected(self):
        with self.assertRaisesRegex(g.GateValidationError, "NOT_DERIVED"):
            g.equal_words(word(pauli("X"), pauli("Z")), word(pauli("Z"), pauli("X")))

    def test_dimension_mismatch_rejected(self):
        with self.assertRaisesRegex(g.GateValidationError, "DIMENSION_MISMATCH"):
            g.product(word(n=1), word(n=2))

    def test_unsupported_proof_rejected(self):
        with self.assertRaisesRegex(g.GateValidationError, "UNSUPPORTED_PROOF_RULE"):
            g.check_symbolic_rule(rule("EMPIRICAL_MAX_DIFFERENCE"))


class IntegrityAndScope(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bundle = bundle_fixture(self.root)

    def check(self):
        return g.validate_bundle(self.bundle, self.root)

    def test_all_VERIFIED_labels_cannot_accept(self):
        result = self.check()
        self.assertFalse(result["all_gate_proofs_verified"])
        self.assertFalse(result["formal_certificate_accepted"])
        self.assertIsNone(result["N"])
        self.assertIsNone(result["G"])
        self.assertEqual(len(result["gates"]), 14)

    def test_TARGET_MET_and_CERTIFIED_are_not_proofs(self):
        for claim in ("TARGET_MET", "CERTIFIED", True):
            self.bundle["gates"]["numerical_enclosure"]["claim"] = claim
            with self.assertRaisesRegex(g.GateValidationError, "CLAIM_VOCABULARY"):
                self.check()

    def test_unknown_gate_cannot_be_ignored(self):
        self.bundle["gates"]["fabricated"] = self.bundle["gates"]["encoded_event_norm"]
        with self.assertRaisesRegex(g.GateValidationError, "COMPLETE_GATE_SET"):
            self.check()

    def test_missing_gate_rejected(self):
        del self.bundle["gates"]["compiler_transfer"]
        with self.assertRaisesRegex(g.GateValidationError, "COMPLETE_GATE_SET"):
            self.check()

    def test_readout_cannot_replace_encoded_norm(self):
        self.bundle["gates"]["encoded_event_norm"]["scope"] = "BINARY_MEASUREMENT"
        with self.assertRaisesRegex(g.GateValidationError, "SCOPE_MISMATCH"):
            self.check()

    def test_changed_reference_bytes_rejected(self):
        (self.root / "C_result.json").write_bytes(b"changed")
        with self.assertRaisesRegex(g.GateValidationError, "HASH_MISMATCH"):
            self.check()

    def test_reference_target_hash_mismatch(self):
        self.bundle["target"]["C_result_sha256"] = "f" * 64
        with self.assertRaisesRegex(g.GateValidationError, "TARGET_EVIDENCE_BINDING"):
            self.check()

    def test_wrong_scientific_SOURCE_rejected(self):
        self.bundle["target"]["C_SOURCE"] = "a" * 40
        with self.assertRaisesRegex(g.GateValidationError, "FIXED_SCIENTIFIC_SOURCE"):
            self.check()

    def test_wrong_reference_SOURCE_rejected(self):
        self.bundle["evidence"][1]["source_commit"] = "a" * 40
        with self.assertRaisesRegex(g.GateValidationError, "SOURCE_EVIDENCE_BINDING"):
            self.check()

    def test_fresh_shot_assumption_required_even_PF(self):
        self.bundle = bundle_fixture(self.root, "pauli_jw_PF")
        self.bundle["assumptions"] = []
        with self.assertRaisesRegex(g.GateValidationError, "ASSUMPTIONS_NOT_RECORDED"):
            self.check()

    def test_IID_uniform_word_assumption_required_PR(self):
        self.bundle["assumptions"] = ["FRESH_INDEPENDENT_SHOTS"]
        with self.assertRaisesRegex(g.GateValidationError, "ASSUMPTIONS_NOT_RECORDED"):
            self.check()

    def test_unconditional_PCG_proof_not_new_requirement(self):
        self.assertFalse(self.check()["unconditional_PCG64_independence_required"])

    def test_no_arbitrary_extra_assumptions(self):
        self.bundle["assumptions"].append("ROUNDED_DF_IS_EXACT_UNITARY")
        with self.assertRaisesRegex(g.GateValidationError, "EXISTING_ASSUMPTIONS_ONLY"):
            self.check()

    def test_execution_acceptance_and_NG_flags_cannot_be_enabled(self):
        for flag in ("execution_authorized", "formal_certificate_accepted", "N_G_authorized"):
            for value in (True, 0, 1):
                changed = deepcopy(self.bundle)
                changed[flag] = value
                with self.assertRaisesRegex(g.GateValidationError, "TECHNICAL_SCOPE_ONLY"):
                    g.validate_bundle(changed, self.root)

    def attach_readout(self, gate="measurement_binary_range"):
        self.bundle["gates"][gate]["lemma"] = dict(target_digest=g.digest(self.bundle["target"]),
            proof=rule("BINARY_READOUT", scale_symbol="B_saved", bit_values=[0, 1], estimator="B*(1-2*b)"))

    def test_lemma_checked_without_laundering_H4_binding(self):
        self.attach_readout()
        r = self.check()["gates"]["measurement_binary_range"]
        self.assertEqual(r["checked_lemma"]["status"], "FORMAL_LEMMA_CHECKED")
        self.assertFalse(r["proof_gate_satisfied"])
        self.assertIn("BINDING", r["blocker"])

    def test_lemma_cannot_move_between_case_axis_precision(self):
        self.attach_readout()
        for key, value in (("case", "df_native_PF"), ("axis", "cosine"), ("epsilon_sig", "0.0005")):
            changed = deepcopy(self.bundle)
            changed["target"][key] = value
            with self.assertRaisesRegex(g.GateValidationError, "PROOF_TARGET_MISMATCH"):
                g.validate_bundle(changed, self.root)

    def test_readout_lemma_cannot_satisfy_norm_gate(self):
        self.attach_readout("encoded_event_norm")
        with self.assertRaisesRegex(g.GateValidationError, "LEMMA_GATE_SCOPE"):
            self.check()

    def test_positive_claim_without_evidence_rejected(self):
        self.bundle["gates"]["encoded_event_norm"]["evidence_ids"] = []
        with self.assertRaisesRegex(g.GateValidationError, "NEEDS_EVIDENCE"):
            self.check()

    def test_unknown_reference_id_rejected(self):
        self.bundle["gates"]["encoded_event_norm"]["evidence_ids"] = ["missing"]
        with self.assertRaisesRegex(g.GateValidationError, "UNBOUND_EVIDENCE"):
            self.check()

    def test_duplicate_reference_id_rejected(self):
        self.bundle["evidence"].append(self.bundle["evidence"][0])
        with self.assertRaisesRegex(g.GateValidationError, "DUPLICATE_EVIDENCE"):
            self.check()

    def test_failed_condition_not_erased(self):
        self.bundle["gates"]["encoded_event_norm"]["claim"] = "FAILED_CONDITION"
        self.assertIn("FAILED_SAVED_CLAIM", self.check()["gates"]["encoded_event_norm"]["blocker"])

    def test_new_allowance_cannot_enter_technical_bundle(self):
        self.bundle["allowance"] = 1e-6
        with self.assertRaisesRegex(g.GateValidationError, "BUNDLE_SCHEMA"):
            self.check()

    def test_precompiled_fingerprint_not_compiled_program(self):
        self.bundle["target"]["compiled_program_sha256"] = "1" * 64
        with self.assertRaisesRegex(g.GateValidationError, "UNRESOLVED_TARGET"):
            self.check()

    def test_program_hash_without_program_bytes_rejected(self):
        self.bundle["target"].update(compiler_target="HISTORICAL_PROGRAM", compiled_program_sha256="1" * 64)
        with self.assertRaisesRegex(g.GateValidationError, "POSTCOMPILED_PROGRAM_BYTES"):
            self.check()

    def test_unsafe_relative_paths_rejected(self):
        for path in ("../outside", "/tmp/outside", "C_result.json/../C_result.json", "./C_result.json"):
            with self.assertRaisesRegex(g.GateValidationError, "UNSAFE_EVIDENCE"):
                g.checked_file(self.root, path, "0" * 64)

    def test_symlink_parent_rejected(self):
        (self.root / "link").symlink_to(self.root, target_is_directory=True)
        expected = self.bundle["target"]["C_result_sha256"]
        with self.assertRaisesRegex(g.GateValidationError, "SYMLINK"):
            g.checked_file(self.root, "link/C_result.json", expected)

    def test_duplicate_JSON_keys_rejected(self):
        with self.assertRaisesRegex(g.GateValidationError, "DUPLICATE_JSON"):
            g.read_json_bytes(b'{"scope":"A","scope":"B"}')

    def test_nonfinite_JSON_constants_and_overflow_rejected(self):
        for value in (b"NaN", b"Infinity", b"1e9999"):
            with self.assertRaisesRegex(g.GateValidationError, "NONFINITE_JSON"):
                g.read_json_bytes(b'{"x":' + value + b'}')


def saved_fixture(root):
    audit = "artifacts/non_df_h4_certificate_gate_audit/2026-10-23/"
    seal_path = "artifacts/non_df_h4_c_reference_evidence/2026-10-23/run01/seal.json"
    seal_sha = store(root, seal_path, {"opaque_saved_seal": True})
    result_path = "artificial_opaque_result.bin"
    result_sha = store(root, result_path, b"opaque numerical payload, intentionally not JSON or evaluated")
    rows = [dict(case=c, epsilon_sig=float(e), axis=a, formal_certificate_accepted=False,
                 old_N=None, old_G=None, status="TARGET_MET", bits=64, fixed_C_result=result_path,
                 fixed_C_result_SHA256=result_sha, gates={k: "MISSING_EVIDENCE" for k in g.GATE_SCOPES})
            for c in g.CASES for e in g.EPSILONS for a in g.AXES]
    matrix = dict(schema="H4_existing_evidence_certificate_gate_audit_v1", axis_count=40,
                  rows=rows, formal_certificate_accepted=False, formal_certificate_generated=False,
                  C_SOURCE=g.C_SOURCE)
    files = {audit + "certificate_gate_matrix.json": store(root, audit + "certificate_gate_matrix.json", matrix),
             audit + "gate_catalog.json": store(root, audit + "gate_catalog.json", {"gates": [{"id": k} for k in g.GATE_SCOPES]}),
             audit + "SOURCE.json": store(root, audit + "SOURCE.json", dict(fixed_C_seal_SHA256=seal_sha,
                                                        original_scientific_SOURCE=g.ORIGINAL_SOURCE))}
    inventory = dict(kind="STATIC_AUDIT_DELIVERY_BYTE_INVENTORY", files=files, audit_SOURCE="1" * 40)
    inventory_sha = store(root, cli.INVENTORY, inventory)
    source_hashes = {p: store(root, p, b"artificial source bytes; not executed") for p in cli.SOURCE_FILES}
    executable = Path(sys.executable).resolve()
    seal = dict(kind="H4_GATE_PHASE_T_SOURCE_NO_SCIENTIFIC_AUTHORITY", technical_SOURCE="2" * 40,
                base_audit_commit="0369fa2a05aac7820e1c0f7aa7e55f9d048a0efa", files=source_hashes,
                audit_inventory_sha256=inventory_sha,
                runtime=dict(executable=str(executable), sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
                             version=sys.version), scientific_execution_authorized=False,
                certificate_acceptance_authorized=False, N_G_authorized=False)
    store(root, "technical_seal.json", seal)
    return matrix, seal


class SavedStructureAndCLI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.matrix, self.seal = saved_fixture(self.root)
        self.output = self.root / cli.OUTPUT_NAMESPACE / "artificial_check"
        self.args = SimpleNamespace(seal=str(self.root / "technical_seal.json"), output=str(self.output))

    def fake_git(self, args, **kwargs):
        self.assertEqual(args[:2], ["git", "show"])
        return (self.root / args[2].split(":", 1)[1]).read_bytes()

    def test_all40_structural_joins_do_not_read_numerical_payload(self):
        bundles = g.bundles_from_saved_audit(self.root, cli.INVENTORY)
        self.assertEqual(len(bundles), 40)
        for b in bundles:
            self.assertFalse(g.validate_bundle(b, self.root)["formal_certificate_accepted"])

    def test_duplicate_axis_not_covered_by_row_count(self):
        path = "artifacts/non_df_h4_certificate_gate_audit/2026-10-23/certificate_gate_matrix.json"
        self.matrix["rows"][1] = self.matrix["rows"][0]
        new_sha = store(self.root, path, self.matrix)
        p = self.root / cli.INVENTORY
        inv = json.loads(p.read_text()); inv["files"][path] = new_sha; store(self.root, cli.INVENTORY, inv)
        with self.assertRaisesRegex(g.GateValidationError, "AXIS_DUPLICATE"):
            g.bundles_from_saved_audit(self.root, cli.INVENTORY)

    def test_changed_audit_matrix_rejected(self):
        path = self.root / "artifacts/non_df_h4_certificate_gate_audit/2026-10-23/certificate_gate_matrix.json"
        path.write_text("{}")
        with self.assertRaisesRegex(g.GateValidationError, "HASH_MISMATCH"):
            g.bundles_from_saved_audit(self.root, cli.INVENTORY)

    def test_new_cli_artificial_end_to_end_and_exclusive_output(self):
        with patch.object(cli, "ROOT", self.root), patch.object(cli.subprocess, "check_output", self.fake_git):
            cli.run(self.args)
            before = (self.output / "preflight.json").read_bytes()
            result = json.loads(before)
            self.assertEqual(result["axes_checked"], 40)
            self.assertEqual(result["scientific_calls"], 0)
            self.assertTrue((self.output / "STOP.json").exists())
            with self.assertRaises(FileExistsError):
                cli.run(self.args)
            self.assertEqual((self.output / "preflight.json").read_bytes(), before)

    def test_old_evidence_namespace_cannot_receive_outputs(self):
        self.args.output = str(self.root / "artifacts/non_df_h4_pilot/new_path")
        with patch.object(cli, "ROOT", self.root), patch.object(cli.subprocess, "check_output", self.fake_git):
            with self.assertRaisesRegex(cli.g.GateValidationError, "OWN_FRESH_TECHNICAL_NAMESPACE"):
                cli.run(self.args)
        self.assertFalse(Path(self.args.output).exists())

    def test_bad_source_seal_leaves_no_output(self):
        self.seal["files"]["src/trotterlib/h4_certificate_gates.py"] = "f" * 64
        store(self.root, "technical_seal.json", self.seal)
        with patch.object(cli, "ROOT", self.root), patch.object(cli.subprocess, "check_output") as git_read:
            with self.assertRaisesRegex(cli.g.GateValidationError, "HASH_MISMATCH"):
                cli.run(self.args)
            git_read.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_runtime_mismatch_rejected(self):
        self.seal["runtime"]["sha256"] = "f" * 64
        with patch.object(cli.subprocess, "check_output", self.fake_git):
            with self.assertRaisesRegex(cli.g.GateValidationError, "RUNTIME_MISMATCH"):
                cli.validate_seal(self.seal, self.root)

    def test_source_commit_bytes_must_match(self):
        with patch.object(cli.subprocess, "check_output", return_value=b"other source bytes"):
            with self.assertRaisesRegex(cli.g.GateValidationError, "SOURCE_COMMIT_MISMATCH"):
                cli.validate_seal(self.seal, self.root)

    def test_science_authority_cannot_enter_seal(self):
        self.seal["scientific_execution_authorized"] = True
        with self.assertRaisesRegex(cli.g.GateValidationError, "NO_SCIENCE_AUTHORITY"):
            cli.validate_seal(self.seal, self.root)

    def test_source_set_cannot_omit_checker(self):
        del self.seal["files"]["src/trotterlib/h4_certificate_gates.py"]
        with self.assertRaisesRegex(cli.g.GateValidationError, "SOURCE_FILE_SET"):
            cli.validate_seal(self.seal, self.root)

    def test_import_is_standard_library_only(self):
        original = builtins.__import__
        forbidden = {"trotterlib", "numpy", "scipy", "qiskit", "flint", "pyscf", "openfermion"}
        def checked(name, *args, **kwargs):
            if name.split(".")[0] in forbidden:
                self.fail("Forbidden scientific import: " + name)
            return original(name, *args, **kwargs)
        with patch("builtins.__import__", checked):
            load("stdlib_gate_import_test", ROOT / "src/trotterlib/h4_certificate_gates.py")
            load("stdlib_gate_cli_import_test", ROOT / "scripts/verify_h4_certificate_gates.py")


if __name__ == "__main__":
    unittest.main(verbosity=2)
