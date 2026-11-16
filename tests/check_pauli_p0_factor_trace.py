"""Artificial-only tests for local default/no-filter factor/JW tracing."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np
from openfermion import FermionOperator, QubitOperator, jordan_wigner
from openfermion.config import EQ_TOLERANCE

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import pauli_p0_factor_trace as T


class FactorTrace(unittest.TestCase):
    def setUp(self):
        no_load=patch.object(np,'load',side_effect=AssertionError('Artificial tests do not load molecular arrays'))
        no_load.start();self.addCleanup(no_load.stop)
        self.addition_f=FermionOperator.__iadd__;self.addition_q=QubitOperator.__iadd__

    def tearDown(self):
        self.assertIs(FermionOperator.__iadd__,self.addition_f)
        self.assertIs(QubitOperator.__iadd__,self.addition_q)
        self.assertEqual(EQ_TOLERANCE,1e-8)

    def test_tracked_default_exactly_matches_frozen_path_with_complex_signed_factors(self):
        g=np.array([[.2, .1j, 1e-9],[-.1j,-.3,.04],[1e-9,.04,.1]])
        h=SimpleNamespace(lambdas=[.4,-2e-7,.75e-8],g_matrices=[g,np.eye(3),np.diag([1.,0.,0.])])
        got=T.trace(ROOT,h,0)
        fixed=T.P0.square_factor_dictionary(ROOT,h,0)
        self.assertEqual(list(got['default_jw'].terms.items()),list(fixed.terms.items()))
        self.assertGreater(len(got['evidence']['removed_nonzero_terms']),0)
        json.dumps(got['evidence'],allow_nan=False)
        self.assertEqual(got['evidence']['factor_indices'],[0,1,2])

    def test_weighted_fermion_partial_is_first_divergence_and_is_retained_in_shadow(self):
        h=SimpleNamespace(lambdas=[.75e-8,.75e-8],g_matrices=[np.diag([1.,0.])]*2)
        got=T.trace(ROOT,h,0);e=got['evidence']
        self.assertEqual(e['first_stage_difference']['stage'],'fermion_residual')
        self.assertEqual(e['first_removed_nonzero_term']['context'],
                         dict(stage='fermion_residual_addition',factor_index=0))
        self.assertEqual(got['default_jw'].terms,{})
        self.assertEqual(got['bridge_jw'].terms,{})
        self.assertAlmostEqual(got['shadow_jw'].terms[()],.75e-8)
        self.assertAlmostEqual(got['shadow_jw'].terms[((0,'Z'),)],-.75e-8)
        cut=e['removed_nonzero_terms'][0]
        self.assertEqual(cut['partial_before_removal']['real'],.75e-8)
        self.assertIn('real_hex',cut['incoming'])

    def test_JW_only_loss_and_bridge_exposes_complete_nonzero_target(self):
        h=SimpleNamespace(lambdas=[1.5e-8],g_matrices=[np.eye(2)])
        got=T.trace(ROOT,h,0);e=got['evidence']
        self.assertEqual(e['first_stage_difference']['stage'],'JW')
        self.assertEqual(e['first_removed_nonzero_term']['context']['stage'],'JW_addition')
        self.assertEqual(got['default_jw'].terms,{})
        self.assertEqual(got['bridge_jw'].terms,got['shadow_jw'].terms)
        self.assertAlmostEqual(got['shadow_jw'].terms[()],2.25e-8)
        self.assertTrue(any(c['term']==[] for c in e['removed_nonzero_terms']))
        self.assertTrue(any(c['source_fermion_term'] is not None for c in e['removed_nonzero_terms']))

    def test_one_body_loss_is_separate_from_product_and_weight_stages(self):
        h=SimpleNamespace(lambdas=[1.],g_matrices=[np.diag([.75e-8,1.])])
        got=T.trace(ROOT,h,0)
        self.assertEqual(got['evidence']['first_stage_difference']['stage'],'one_body')
        self.assertEqual(got['evidence']['first_removed_nonzero_term']['context'],
                         dict(stage='one_body_addition',factor_index=0))
        stages=got['evidence']['factors'][0]
        self.assertEqual(stages['weight']['real'],1.)
        self.assertEqual(stages['nonzero_input'][0]['indices'],[0,0])
        self.assertFalse(stages['square']['identical'])
        self.assertFalse(stages['weighted']['identical'])

    def test_no_filter_shadow_uses_same_addition_order_not_sorted_reduction(self):
        # Cancellation and reappearance exercise insertion order and bridge conversion.
        h=SimpleNamespace(lambdas=[.1,-.1,3e-8],g_matrices=[np.eye(2)]*3)
        got=T.trace(ROOT,h,0)
        fixed=T.P0.square_factor_dictionary(ROOT,h,0)
        self.assertEqual(list(got['default_jw'].terms.items()),list(fixed.terms.items()))
        self.assertEqual(got['evidence']['factor_indices'],[0,1,2])
        for row in got['evidence']['default_fermion_residual']:
            self.assertEqual(float.fromhex(row['real_hex']),row['real'])

    def test_invalid_or_nonfinite_factors_rejected(self):
        h=SimpleNamespace(lambdas=[1.],g_matrices=[np.eye(2)])
        for prefix in (-1,1,True):
            with self.assertRaises(ValueError):T.trace(ROOT,h,prefix)
        h.g_matrices=[np.diag([1.,np.nan])]
        with self.assertRaisesRegex(ValueError,'Nonfinite'):T.trace(ROOT,h,0)


if __name__=='__main__':unittest.main(verbosity=2)
