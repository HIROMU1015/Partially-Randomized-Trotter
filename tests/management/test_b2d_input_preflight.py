"""Artificial inputs, fixed-source bytes and operations mocks; no science."""
from dataclasses import asdict, replace
import importlib.abc
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS, ModuleType
import unittest
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_b2d_evaluator import settings, MockBackend, Barrier, FORBIDDEN

guard = Barrier()
sys.meta_path.insert(0, guard)
try:
    from trottertracks.b2d_evaluator.input_package import (
        InputRecipe, connect_prepared, load_candidate, read_fixture_definition, unresolved_identity,
    )
    from trottertracks.b2d_evaluator.evaluation import build_graph, evaluate
    from trottertracks.b2d_evaluator.dense_backend import DenseBackend
    from trottertracks.b2d_evaluator.preflight import (
        graph_layout, validate_scope, verify_execution,
    )
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

    def test_real_loader_uses_scientific_context_for_leakage_and_source_metadata(self):
        # Exercise the actual loader, with every scientific import/function
        # replaced by opaque stubs. No Hamiltonian, matrix or circuit is made.
        from trottertracks.b2d_evaluator.scope import candidate_graph,Stage1Failure
        graph=candidate_graph('MOCK_SOURCE_NO_SCIENCE')
        prep=NS(deterministic_fragment_indices=(0,1),randomized_block_indices=(2,),
            deterministic_blocks=[NS(block_id='D'+str(i),block_kind='one_body' if i==0 else 'df_fragment',
                original_fragment_index=None if i==0 else i-1,basis_id='MOCK_BASIS',
                basis_hash='MOCK_HASH',basis_change_operations=()) for i in range(3)],
            tail_extraction=NS(basis_definitions=[NS(basis_id='MOCK_BASIS',basis_hash='MOCK_HASH',operations=())],
                components=[NS(component_id='MOCK_COMPONENT',df_fragment_id='df-fragment-2',
                    basis_id='MOCK_BASIS',diagonal_pauli_support=(0,))]),exact_rte_lambda_r='MOCK_LAMBDA')
        parent=ModuleType('trotterlib');parent.__path__=[]
        modules={name:ModuleType(name) for name in ('trotterlib.df_hamiltonian',
            'trotterlib.df_partial_randomized_pf','trotterlib.df_partial_s2',
            'trotterlib.finite_rte_signal_validation')}
        modules['trotterlib']=parent
        modules['numpy']=MagicMock(name='MOCK_NUMPY_NO_SCIENCE')
        h=NS(n_qubits=2,metadata={},one_body=MagicMock(),lambdas=MagicMock(),g_matrices=(None,None,None))
        modules['trotterlib.df_hamiltonian'].DFHamiltonian=MagicMock(return_value=h)
        modules['trotterlib.df_hamiltonian'].PhysicalSector=NS(number_sector=lambda **kw:NS(
            basis_indices=(1,2),dimension=2))
        modules['trotterlib.df_partial_randomized_pf'].split_df_hamiltonian_by_ld=lambda *a,**kw:'MOCK_SPLIT'
        modules['trotterlib.df_partial_s2'].prepare_df_partial_s2=lambda *a,**kw:prep
        numerical=modules['trotterlib.finite_rte_signal_validation']
        numerical.dense_df_operator_in_sector=lambda *a:MagicMock(name='MOCK_MATRIX')
        numerical._qiskit_to_openfermion_sector_permutation=lambda *a:'MOCK_PERMUTATION'
        numerical._array_fingerprint=lambda *a:'MOCK_ARRAY_HASH'
        context=NS(block_ids=('D0','D1','D2'),scalar_terms=())
        inputs=NS(hamiltonian_id='MOCK_H',preparation_id='MOCK_PREP',tail_id='MOCK_TAIL',
            sector_id='MOCK_N1',state_id='MOCK_STATE')
        for leakage in (0.0,1e-6):
            execution=NS(source_commit=graph.settings.source_commit,scope=NS(graph=graph),
                begin_input=MagicMock(),record_resolution=MagicMock(),bind_input=MagicMock(),
                science_call=lambda name,function,**kw:function())
            numerical._normalized_symbolic_tail_in_sector=lambda *a:NS(
                matrix=MagicMock(name='MOCK_TAIL'),leakage_frobenius_norm=leakage)
            with self.subTest(leakage=leakage),patch.dict(sys.modules,modules),patch(
                    'trottertracks.b2d_evaluator.input_package.connect_prepared',return_value=(context,inputs)):
                if leakage:
                    with self.assertRaisesRegex(Stage1Failure,'SECTOR_BASIS_LEAKAGE'):
                        load_candidate(ROOT,execution=execution)
                    execution.bind_input.assert_not_called()
                else:
                    _,_,provenance=load_candidate(ROOT,execution=execution)
                    self.assertEqual(provenance['source_commit'],'MOCK_SOURCE_NO_SCIENCE')
                    execution.bind_input.assert_called_once()

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
        evaluate(graph, lambda _: backend, mock_only=True)
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
            evaluate(graph, lambda _: FailingBackend(), mock_only=True)

    def test_backend_requires_scientific_input_scope_context(self):
        graph = build_graph(no_phase_settings())
        with self.assertRaises(ValueError):
            DenseBackend(graph.settings, object())

    def test_registry_rejects_incomplete_scientific_scope(self):
        from trottertracks.b2d_evaluator.scope import candidate_graph,registry,Stage1Failure
        graph = candidate_graph("MOCK")
        scope = registry(graph)
        self.assertEqual(scope["dispatch_total"],941)
        self.assertEqual(scope["boundaries"]["backend.expm"],448)
        scope["boundaries"] = {}
        with self.assertRaises(Stage1Failure): validate_scope(scope,graph)

    def test_layout_binds_grid_not_uncreated_array_hashes(self):
        a = build_graph(no_phase_settings())
        b = build_graph(replace(a.settings, input_data_sha256="DIFFERENT_MOCK_ARRAYS"))
        self.assertEqual(graph_layout(a), graph_layout(b))
        self.assertNotEqual(a.fingerprint, b.fingerprint)
        self.assertNotEqual(graph_layout(a), graph_layout(build_graph(replace(a.settings, q_values=(2,)))))

    def test_actual_source_check_rejects_changed_commit_without_science(self):
        from trottertracks.b2d_evaluator.preflight import _check_source
        with self.assertRaises(ValueError):
            _check_source(ROOT,"0"*40)
        self.assertFalse(FORBIDDEN.intersection(name.split(".")[0] for name in sys.modules))

    def test_clean_commit_and_dirty_checkout_checks_without_source_hash_packet(self):
        from trottertracks.b2d_evaluator.preflight import _check_source
        with patch("subprocess.check_output",side_effect=["MOCK_HEAD\n", ""]):
            _check_source(ROOT,"MOCK_HEAD")
        for head,dirty in (("WRONG", ""),("MOCK_HEAD", " M src/changed.py\n")):
            with self.subTest(head=head,dirty=dirty),patch("subprocess.check_output",side_effect=[head,dirty]),self.assertRaises(ValueError):
                _check_source(ROOT,"MOCK_HEAD")


if __name__ == "__main__": unittest.main()
