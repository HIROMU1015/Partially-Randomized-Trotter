"""Artificial-only P0 tolerance localization; never load molecular inputs.

The installed OpenFermion default is observed, never patched.  These fixtures
demonstrate possible mechanisms; they do not identify which terms were lost in
the saved LD7/8/9 runs.  No production gate, dictionary or STOP is modified.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np
from openfermion import FermionOperator, QubitOperator, jordan_wigner
from openfermion.config import EQ_TOLERANCE

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks import df_head_pauli_tail_p0 as P0


def fixed_one_body():
    """The same frozen M1 nested function extracted by square_factor_dictionary."""
    parent = next(node for node in ast.parse(P0.blob(ROOT, P0.M1,
        'src/trotterlib/o1_m1_matched_split.py')).body
        if isinstance(node, ast.FunctionDef) and node.name == 'prepare_matched_a')
    node = next(node for node in parent.body
        if isinstance(node, ast.FunctionDef) and node.name == 'one_body')
    scope = dict(np=np, Any=object, FermionOperator=FermionOperator)
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'fixed_M1_one_body', 'exec'), scope)
    return scope['one_body']


class SymbolicTolerance(unittest.TestCase):
    def setUp(self):
        self.no_load = patch.object(np, 'load', side_effect=AssertionError('No molecular array load'))
        self.no_load.start()
        self.addCleanup(self.no_load.stop)
        self.assertEqual(EQ_TOLERANCE, 1e-8)

    def test_frozen_sources_used_by_saved_P0_are_unchanged(self):
        for commit, path in (
            ('fd111f989b9cd2f052d7189e4b7b6fab10aa1a9f', 'src/trottertracks/df_head_pauli_tail_p0.py'),
            ('75fb0ba114ac518bfe5d65d0164cee07d7469bb4', 'src/trotterlib/pauli_signal_adapter.py')):
            self.assertEqual((ROOT / path).read_bytes(), P0.blob(ROOT, commit, path))

    def test_first_possible_loss_is_one_body_iadd_not_matrix_nonzero_filter(self):
        tiny = .75 * EQ_TOLERANCE
        g = np.diag([tiny, 0.])
        self.assertEqual(np.argwhere(g != 0).tolist(), [[0, 0]])
        direct = FermionOperator(((0, 1), (0, 0)), tiny)
        self.assertEqual(len(direct.terms), 1)  # constructor retains this value
        with patch.object(FermionOperator, 'compress', side_effect=AssertionError('No explicit compress')):
            self.assertEqual(fixed_one_body()(g).terms, {})  # += applies default threshold

    def test_weighted_residual_iadd_can_discard_before_aggregate_is_large(self):
        tiny = .75 * EQ_TOLERANCE
        a = fixed_one_body()(np.diag([1., 0.]))
        part = tiny * (a * a)
        self.assertEqual(len(part.terms), 1)  # product/scalar multiplication retained it
        total = FermionOperator()
        with patch.object(FermionOperator, 'compress', side_effect=AssertionError('No explicit compress')):
            total += part
            total += part
        self.assertEqual(total.terms, {})
        combined_first = (2 * tiny) * (a * a)
        self.assertGreater(abs(next(iter(combined_first.terms.values()))), EQ_TOLERANCE)

    def test_JW_partial_accumulation_can_remove_identity_and_nonidentity(self):
        coefficient = 1.5 * EQ_TOLERANCE
        f = FermionOperator(((0, 1), (0, 0)), coefficient)
        self.assertGreater(abs(next(iter(f.terms.values()))), EQ_TOLERANCE)
        # n0 = (I-Z0)/2. Each JW output coefficient falls below the default.
        self.assertLess(coefficient / 2, EQ_TOLERANCE)
        with patch.object(QubitOperator, 'compress', side_effect=AssertionError('No explicit compress')):
            self.assertEqual(jordan_wigner(f).terms, {})
        control = jordan_wigner(FermionOperator(((0, 1), (0, 0)), 3 * EQ_TOLERANCE))
        self.assertIn((), control.terms)
        self.assertIn(((0, 'Z'),), control.terms)

    def test_frozen_square_factor_path_reproduces_loss_on_two_mode_toy(self):
        weight = 1.5 * EQ_TOLERANCE
        ham = SimpleNamespace(lambdas=[weight], g_matrices=[np.eye(2)])
        with patch.object(FermionOperator, 'compress', side_effect=AssertionError('No explicit compress')), \
             patch.object(QubitOperator, 'compress', side_effect=AssertionError('No explicit compress')):
            actual = P0.square_factor_dictionary(ROOT, ham, 0)
        self.assertEqual(actual.terms, {})
        # Analytical toy w*(n0+n1)^2, not a saved H4 matrix or reanalysis.
        target = np.diag([0., weight, weight, 4 * weight])
        self.assertGreater(np.linalg.norm(target), P0.TOL)
        self.assertGreater(np.trace(target) / 4, EQ_TOLERANCE)

    def test_adapter_and_sparse_matrix_keep_tiny_supplied_identity_and_terms(self):
        tiny = .75 * EQ_TOLERANCE
        operator = QubitOperator(((0, 'Z'),), tiny)
        operator.terms[()] = tiny  # supplied already-collected data; no += cutoff
        model = P0.pauli_api().PauliHamiltonian.from_openfermion(operator, num_qubits=1)
        self.assertEqual(model.constant, tiny)
        self.assertEqual(model.terms, (('Z', tiny),))
        self.assertEqual(model.metadata['coefficient_cutoff'], 0.)
        np.testing.assert_array_equal(model.sparse_operator().toarray(), np.diag([2 * tiny, 0.]))
        np.testing.assert_array_equal(model.sparse_operator(constant=0.).toarray(), np.diag([tiny, -tiny]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
