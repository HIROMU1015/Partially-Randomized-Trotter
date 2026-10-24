"""Artificial inputs, fixed-source bytes and operations mocks; no science."""
from dataclasses import asdict, replace
import importlib.abc
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_b2d_evaluator import settings, mock_approval, MockBackend, Barrier, FORBIDDEN

guard = Barrier()
sys.meta_path.insert(0, guard)
try:
    from trottertracks.b2d_evaluator.input_package import (
        InputRecipe, connect_prepared, load_candidate, read_fixture_definition, unresolved_identity,
    )
    from trottertracks.b2d_evaluator.evaluation import build_graph, evaluate
    from trottertracks.b2d_evaluator.dense_backend import DenseBackend
    from trottertracks.b2d_evaluator.preflight import (
        STAGE, SCOPE_GROUPS, graph_layout, scope_proposal, validate_scope, verify_execution,
        source_inventory,
    )
    from trottertracks.pr_dictionary_validation.authorization import _canonical, file_sha
finally:
    sys.meta_path.remove(guard)


def no_phase_settings():
    return replace(settings(), phase_diagnostics=False, radius_floor=None, branch_margin=None)


def preparation():
    return NS(preparation_hash="MOCK_PREPARATION", hamiltonian_hash="MOCK_H",
        tail_extraction=NS(tail_hash="MOCK_TAIL"),
        deterministic_block_order=("one", "df-ranked-a", "df-ranked-b"),
        num_system_qubits=2, ld=2, identity_policy="extract_identity_phase", coefficient_atol=0,
        constant_coefficient=0.17, extracted_identity_coefficient=0.23,
        rte_preparation=NS(symbolic_tail=NS(tail_hash="MOCK_TAIL", lambda_r=2.0)))


class MockAuthority:
    is_mock = True
    def verify_approval(self, packet, approval, grant_digest, contract_digest):
        return approval == b"MOCK_RECORD_NOT_ACTUAL_USER_APPROVAL"


class InputPreflightTests(unittest.TestCase):
    def setUp(self):
        barrier = Barrier()
        sys.meta_path.insert(0, barrier)
        self.addCleanup(sys.meta_path.remove, barrier)

    def test_fixed_fixture_literals_without_importing_helper(self):
        d = read_fixture_definition(ROOT)
        self.assertEqual(d["constant"], .17)
        self.assertEqual(d["metadata"]["name"], "nontrivial-repeated-bases")
        self.assertEqual(len(d["g_matrices"]), 3)
        self.assertEqual(d["one_body"][0][1], .13)
        self.assertEqual(InputRecipe().weight_rule, "lambda_frobenius_squared")

    def test_changed_fixture_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / InputRecipe().fixture_path
            path.parent.mkdir()
            path.write_text("def _nontrivial_basis_case(): pass\n")
            with self.assertRaises(ValueError):
                read_fixture_definition(directory)

    def test_unresolved_identity_stays_null(self):
        d = unresolved_identity()
        for key in ("hamiltonian_id", "input_data_sha256", "ordered_d_block_ids", "state_id"):
            self.assertIsNone(d[key])

    def test_no_automatic_sector_fixture_or_split_change(self):
        for values in (dict(n_electrons=2), dict(occupied_modes=(1,)), dict(ld=1),
                       dict(weight_rule="generation_prefix"), dict(coefficient_atol=1)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                replace(InputRecipe(), **values).validate()

    def test_opaque_input_assembly_preserves_order_scalar_and_tail(self):
        p = preparation()
        blocks = {name: object() for name in p.deterministic_block_order}
        physical, tail, state = object(), object(), object()
        context, inputs = connect_prepared(p, blocks=blocks, physical_hamiltonian=physical,
            normalized_tail=tail, state=state, sector_basis_indices=(1, 2))
        self.assertEqual(tuple(inputs.blocks), context.block_ids)
        self.assertIs(inputs.physical_hamiltonian, physical)
        self.assertIs(inputs.normalized_tail, tail)
        self.assertIs(inputs.state, state)
        self.assertIs(inputs.symbolic_tail, p.rte_preparation.symbolic_tail)
        self.assertEqual(tuple(t.label for t in context.scalar_terms),
                         ("hamiltonian_constant", "extracted_tail_identity"))

    def test_bad_order_sector_or_identity_policy_rejected(self):
        p = preparation()
        good = dict(blocks={k: object() for k in p.deterministic_block_order},
                    physical_hamiltonian=object(), normalized_tail=object(), state=object(),
                    sector_basis_indices=(1, 2))
        for changes in (dict(blocks=dict(reversed(tuple(good["blocks"].items())))),
                        dict(sector_basis_indices=(2, 1)), dict(sector_basis_indices=(0, 3))):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                connect_prepared(p, **{**good, **changes})
        p.identity_policy = "faithful_identity_in_tail"
        with self.assertRaises(ValueError):
            connect_prepared(p, **good)

    def test_phase_disabled_needs_no_invented_thresholds(self):
        s = no_phase_settings()
        self.assertEqual(len(build_graph(s).rows), 36)
        for changes in (dict(radius_floor=.1), dict(branch_margin=.1), dict(numerical_atol=None)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                build_graph(replace(s, **changes))

    def test_phase_disabled_signal_schema_uses_opaque_complex_labels(self):
        # No backend constructor, matrix, state or scientific signal evaluation.
        value = NS(np=None, s=no_phase_settings(), i=NS(state=object()),
                   signal=lambda label, state: label, phase_distance=lambda *x: self.fail("phase called"))
        result = DenseBackend._signals(value, (1+0j, .8+.1j, .7+.2j, .6+.3j))
        self.assertEqual(result["physical"]["phase_status"], "NOT_REQUESTED")
        self.assertIsNone(result["corrected_physical_phase_distance"])
        self.assertIn("corrected_physical_signal_error", result)

    def test_all_correctness_gates_precede_every_comparison(self):
        graph = build_graph(no_phase_settings())
        backend = MockBackend()
        evaluate(graph, lambda _: backend, mock_approval(graph))
        kinds = [call[0] for call in backend.calls]
        gate_positions = [i for i, kind in enumerate(kinds) if kind == "a4_reference_check"]
        self.assertEqual(len(gate_positions), 4)
        self.assertLess(max(gate_positions), kinds.index("operator_error"))
        self.assertLess(max(gate_positions), kinds.index("signals"))
        self.assertEqual(len(kinds), 941)

    def test_correctness_failure_prevents_all_performance_calls(self):
        graph = build_graph(no_phase_settings())
        class FailingBackend(MockBackend):
            def perform(self, kind, args, params):
                if kind == "a4_reference_check":
                    raise ValueError("MOCK_CORRECTNESS_FAILURE")
                self.assert_no_comparison = kind not in ("operator_error", "signals")
                if not self.assert_no_comparison:
                    raise AssertionError("Comparison preceded correctness")
                return super().perform(kind, args, params)
        with self.assertRaisesRegex(ValueError, "MOCK_CORRECTNESS_FAILURE"):
            evaluate(graph, lambda _: FailingBackend(), mock_approval(graph))

    def test_unapproved_science_paths_reject_before_scientific_import(self):
        graph = build_graph(no_phase_settings())
        with self.assertRaises(PermissionError):
            evaluate(graph, lambda _: self.fail("factory reached"), replace(mock_approval(graph), mock_only=False))
        with self.assertRaises(PermissionError):
            DenseBackend(graph.settings, object())
        with self.assertRaises(PermissionError):
            load_candidate(ROOT, execution="caller approval text")
        with self.assertRaises(PermissionError):
            verify_execution("unused", ROOT, "unused", graph)

    def test_scope_proposal_is_not_941_total_and_cannot_authorize(self):
        graph = build_graph(no_phase_settings())
        proposal = scope_proposal(graph)
        self.assertFalse(proposal["fully_resolved"])
        self.assertEqual(proposal["dispatch_total"], 941)
        self.assertEqual(set(proposal["groups"]), set(SCOPE_GROUPS))
        self.assertGreater(proposal["known_interface_counts"]["backend_unique_expm_keys"], 410)
        with self.assertRaises(PermissionError):
            validate_scope(proposal, graph)

    def test_layout_binds_grid_not_uncreated_array_hashes(self):
        a = build_graph(no_phase_settings())
        b = build_graph(replace(a.settings, input_data_sha256="DIFFERENT_MOCK_ARRAYS"))
        self.assertEqual(graph_layout(a), graph_layout(b))
        self.assertNotEqual(a.fingerprint, b.fingerprint)
        self.assertNotEqual(graph_layout(a), graph_layout(build_graph(replace(a.settings, q_values=(2,)))))

    def _packet(self, directory):
        graph = build_graph(no_phase_settings())
        root = Path(directory)
        packet = root / "packet"
        packet.mkdir()
        output = root / "artifacts/research_b_b2d_stage1/mock/run"
        output.parent.mkdir(parents=True)
        scope = scope_proposal(graph)
        # Artificial ledger schema only: these counts are NOT science adoption.
        scope.update(fully_resolved=True, unresolved=[], groups={name: [dict(
            operation="MOCK_OPERATION", count=0, dependencies=[])] for name in SCOPE_GROUPS})
        contract = dict(stage=STAGE, status="USER_ADOPTED_STAGE1", retry_allowed=False,
            mandatory_stop=True, recipe_fingerprint=InputRecipe().fingerprint,
            input_resolution="REGISTERED_RECIPE_THEN_BACKEND_CORRECTNESS",
            numerical_atol=graph.settings.numerical_atol, graph_layout=graph_layout(graph),
            call_scope=scope, source_commit=graph.settings.source_commit,
            source_files={"MOCK_SOURCE": "MOCK_BYTES"}, dependency_versions={"mock": "1"})
        approval = b"MOCK_RECORD_NOT_ACTUAL_USER_APPROVAL"
        grant = dict(stage=STAGE, authority="USER_EXPLICIT_B2D_STAGE1", status="APPROVED_BOUND_ONE_SHOT",
            retry_allowed=False, mandatory_stop=True, approval_sha256=file_sha(approval),
            fresh_output_namespace=str(output), source_commit=contract["source_commit"],
            source_closure_sha256=file_sha(_canonical(contract["source_files"])),
            grant_id="b2d_MOCK_NOT_A_GRANT", approval_reference="MOCK_RECORD")
        def write():
            cb = _canonical(contract)
            grant["contract_sha256"] = file_sha(cb)
            (packet / "contract.json").write_bytes(cb)
            (packet / "authorization.json").write_bytes(_canonical(grant))
            (packet / "user_approval.txt").write_bytes(approval)
        write()
        return graph, packet, output, contract, grant, write

    def test_read_only_mock_packet_never_claims_and_used_marker_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            graph, packet, output, contract, grant, write = self._packet(directory)
            with patch("trottertracks.b2d_evaluator.preflight._check_source"), patch(
                "trottertracks.b2d_evaluator.preflight.read_fixture_definition"), patch(
                "importlib.metadata.version", return_value="1"):
                verified = verify_execution(packet, directory, output, graph, authority=MockAuthority())
                self.assertFalse(verified.consumption_path.exists())
                self.assertFalse(output.exists())
                with self.assertRaises(PermissionError):
                    verified.for_existing_claim()
                with self.assertRaises(PermissionError):
                    verified.after_existing_claim()
                # Artificial marker in isolated tmp only, not an operational grant.
                verified.consumption_path.write_text("MOCK_ALREADY_USED")
                with self.assertRaises(PermissionError):
                    verify_execution(packet, directory, output, graph, authority=MockAuthority())

    def test_packet_rejects_draft_source_grid_scope_output_and_false_record(self):
        mutations = [lambda c, g: c.update(status="PROPOSAL"),
            lambda c, g: c.update(source_commit="wrong"),
            lambda c, g: c.update(graph_layout="wrong"),
            lambda c, g: c["call_scope"].update(dispatch_counts={}),
            lambda c, g: c["call_scope"].update(fully_resolved=False),
            lambda c, g: g.update(stage="F1"),
            lambda c, g: g.update(fresh_output_namespace="wrong"),
            lambda c, g: g.update(retry_allowed=True)]
        for mutate in mutations:
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as directory:
                graph, packet, output, contract, grant, write = self._packet(directory)
                mutate(contract, grant)
                write()
                with patch("trottertracks.b2d_evaluator.preflight._check_source"), patch(
                    "trottertracks.b2d_evaluator.preflight.read_fixture_definition"), patch(
                    "importlib.metadata.version", return_value="1"), self.assertRaises(PermissionError):
                    verify_execution(packet, directory, output, graph, authority=MockAuthority())
        class Deny(MockAuthority):
            def verify_approval(self, *args): return False
        with tempfile.TemporaryDirectory() as directory:
            graph, packet, output, *_ = self._packet(directory)
            with self.assertRaises(PermissionError):
                verify_execution(packet, directory, output, graph, authority=Deny())

    def test_actual_source_check_rejects_changed_commit_without_science(self):
        from trottertracks.b2d_evaluator.preflight import _check_source
        with self.assertRaises(PermissionError):
            _check_source(ROOT, dict(source_commit="WRONG_COMMIT", source_files={}))
        inventory = source_inventory(ROOT)
        self.assertIn(InputRecipe().fixture_path, inventory)
        self.assertIn("src/trotterlib/df_hamiltonian.py", inventory)
        self.assertFalse(FORBIDDEN.intersection(name.split(".")[0] for name in sys.modules))

    def test_source_closure_and_dirty_checkout_rejected(self):
        from trottertracks.b2d_evaluator.preflight import _check_source
        inventory = source_inventory(ROOT)
        contract = dict(source_commit="MOCK_HEAD", source_files=inventory)
        with patch("subprocess.check_output", side_effect=["MOCK_HEAD\n", ""]):
            _check_source(ROOT, contract)
        for dirty, files in ((" M src/changed.py\n", inventory), ("", {})):
            with self.subTest(dirty=dirty), patch("subprocess.check_output", side_effect=["MOCK_HEAD\n", dirty]), self.assertRaises(PermissionError):
                _check_source(ROOT, dict(source_commit="MOCK_HEAD", source_files=files))


if __name__ == "__main__":
    unittest.main()
