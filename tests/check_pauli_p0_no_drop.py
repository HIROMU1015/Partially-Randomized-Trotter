"""Local no-drop P0 tests: artificial inputs only; no molecular array loads."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from openfermion import FermionOperator, QubitOperator
from openfermion.config import EQ_TOLERANCE

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from trottertracks import pauli_p0_no_drop as N
from trottertracks import pauli_p0_factor_trace as T
from check_df_head_pauli_tail_p0 import fixture


def tiny_fixture():
    ham,sector,full,state,meta=fixture()
    ham.lambdas[-1]=1.5e-8;ham.g_matrices=(*ham.g_matrices[:-1],np.eye(3))
    # Exact full trace of w*(n0+n1+n2)^2: E[N^2]=3 in full 3-mode space.
    meta['saved_tail_descriptors']['2']['scalar']=4.5e-8
    return ham,sector,full,state,meta


class NoDrop(unittest.TestCase):
    def setUp(self):
        self.fadd=FermionOperator.__iadd__;self.qadd=QubitOperator.__iadd__
        self.guard=patch.object(np,'load',side_effect=AssertionError('No molecular arrays'))
        self.guard.start();self.addCleanup(self.guard.stop)

    def tearDown(self):
        self.assertIs(FermionOperator.__iadd__,self.fadd)
        self.assertIs(QubitOperator.__iadd__,self.qadd)
        self.assertEqual(EQ_TOLERANCE,1e-8)
        self.assertEqual(N.P0.TOL,1e-9)

    def test_fixed_original_P0_and_trace_helpers_are_unchanged(self):
        for commit,path in ((N.P0_CODE,N.P0_MODULE),
                            (N.TRACE_CODE,'src/trottertracks/pauli_p0_factor_trace.py')):
            self.assertEqual((ROOT/path).read_bytes(),N.P0.blob(ROOT,commit,path))

    def test_local_builder_matches_shadow_with_signed_complex_and_tiny_factors(self):
        g=np.array([[.2,.1j,1e-9],[-.1j,-.3,.04],[1e-9,.04,.1]])
        h=SimpleNamespace(lambdas=[.4,-2e-7,.75e-8],g_matrices=[g,np.eye(3),np.diag([1.,0.,0.])])
        actual=N.square_factor_dictionary(ROOT,h,0)
        shadow=T.trace(ROOT,h,0)['shadow_jw']
        self.assertEqual(list(actual.terms.items()),list(shadow.terms.items()))
        self.assertGreater(len(actual.terms),len(N.P0.square_factor_dictionary(ROOT,h,0).terms))

    def test_nonzero_final_tiny_terms_are_retained_without_a_new_cutoff(self):
        h=SimpleNamespace(lambdas=[.75e-8],g_matrices=[np.diag([1.,0.])])
        actual=N.square_factor_dictionary(ROOT,h,0)
        self.assertAlmostEqual(actual.terms[()],.375e-8)
        self.assertAlmostEqual(actual.terms[((0,'Z'),)],-.375e-8)
        self.assertEqual(N.P0.square_factor_dictionary(ROOT,h,0).terms,{})

    def test_all_original_P0_checks_pass_for_formerly_failing_tiny_tail(self):
        args=tiny_fixture();products={};result={}
        N.diagnose(ROOT,*args,result,lambda k,v:products.update({k:deepcopy(v)}),prefix=2,source='ARTIFICIAL_A')
        self.assertEqual([r['prefix']for r in result['prefixes']],[2])
        self.assertEqual(len(result['checks']),38)  # 3 blocks + one-body common checks, plus one split
        self.assertTrue(all(r['PASS']for r in result['checks']))
        self.assertTrue(all(r['threshold']==1e-9 for r in result['checks']if not r['name'].startswith('saved_')))
        self.assertFalse(any(r['name'].startswith('LD1_')for r in result['checks']))
        payload=products['LD2_DICTIONARY.json'];record=result['prefixes'][0]
        self.assertEqual(payload['builder']['id'],N.BUILDER_ID)
        self.assertEqual(payload['builder_source'],dict(commit='ARTIFICIAL_A',path=N.MODULE))
        self.assertEqual(payload['builder']['final_coefficient_cutoff'],0.)
        self.assertEqual(record['dictionary_sha256'],hashlib.sha256(json.dumps(payload,
            sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest())
        self.assertIn('LD2_Pauli_tail_saved_scalar_fro',[c['name']for c in result['checks']])
        self.assertIn('LD2_Pauli_tail_sector_fro',[c['name']for c in result['checks']])
        products_b={};other={}
        N.diagnose(ROOT,*args,other,lambda k,v:products_b.update({k:deepcopy(v)}),prefix=2,source='ARTIFICIAL_B')
        self.assertNotEqual(other['prefixes'][0]['dictionary_sha256'],record['dictionary_sha256'])
        self.assertEqual(products_b['LD2_DICTIONARY.json']['terms'],payload['terms'])
        self.assertAlmostEqual(record['relative_full_residual'],
            record['identity_metrics']['Pauli_tail_full_fro']/record['operator_scale']['tail_frobenius'])
        self.assertTrue(record['dictionary_eligible_for_configuration'])

    def test_all_66_fixed_checks_on_eight_mode_ten_block_artificial_input(self):
        weights=np.array([.3,-.2,.11,.04,-.02,.01,.006,.002,.001,1.5e-8])
        gs=tuple(np.eye(8)*(.1*(j+1))for j in range(9))+(np.eye(8),)
        ham=SimpleNamespace(n_qubits=8,n_blocks=10,constant=.17,
            one_body=np.diag(np.arange(8)*.01),lambdas=weights,g_matrices=gs)
        sector=SimpleNamespace(basis_indices=np.array([1<<j for j in range(8)]))
        state=np.arange(1.,9.).astype(complex);state/=np.linalg.norm(state)
        full=np.zeros(256,complex);full[sector.basis_indices]=state
        scalar=float(weights[-1]*((np.trace(gs[-1])**2+np.trace(gs[-1]@gs[-1]))/4))
        meta=dict(binding=dict(system='ARTIFICIAL_8_MODE_ONLY'),
            saved_tail_descriptors={'9':dict(scalar=scalar,lambda_R=1.)})
        result={};N.diagnose(ROOT,ham,sector,full,state,meta,result,lambda *a:None,prefix=9,source='ARTIFICIAL')
        self.assertEqual(len(result['checks']),66)
        self.assertTrue(all(c['PASS']for c in result['checks']))
        names={c['name']for c in result['checks']}
        self.assertTrue({'saved_sector_state_norm','saved_full_state_norm','saved_state_full_lift',
            'full_H_projection_fro','LD9_full_H_split_fro','LD9_Pauli_tail_full_fro',
            'LD9_Pauli_tail_saved_scalar_fro','LD9_Pauli_tail_sector_fro',
            'LD9_identity_full_trace_difference','LD9_probability_sum_error'}<=names)
        for kind in ('head','tail','Pauli_nonidentity'):
            for metric in ('hermiticity_fro','sector_escape_fro','sector_return_fro'):
                self.assertIn('LD9_'+kind+'_'+metric,names)
        for block in range(11):
            self.assertIn('block_%d_projection_fro'%block,names)

    def execute_fixture(self,out,*,bad_scalar=False):
        args=tiny_fixture();meta=deepcopy(args[-1])
        descriptor=deepcopy(meta['saved_tail_descriptors']['2'])
        if bad_scalar:descriptor['scalar']+=.01
        context=out.parent/'fake_context.json';context.write_text('{}')
        meta.update(inputs={},context_source=dict(path=str(context),sha256=N.P0.sha(context)))
        with patch.object(N,'source_records',return_value=[dict(artificial=True)]), \
             patch.object(N.P0,'metadata',return_value=meta), \
             patch.object(N.P0,'load_saved',return_value=(*args[:4],dict(artificial=True))):
            return N.execute_p0(ROOT,out,'ARTIFICIAL_SOURCE',2,descriptor,dict(artificial=True))

    def test_real_execute_api_single_prefix_fresh_products_and_runtime(self):
        with tempfile.TemporaryDirectory()as tmp:
            out=Path(tmp)/'P0'
            result=self.execute_fixture(out)
            self.assertEqual(result['status'],N.success(2),result.get('error'))
            self.assertEqual(result['scope']['Pauli_dictionaries'],1)
            self.assertEqual(result['scope']['saved_input_loads'],1)
            self.assertEqual(result['scope']['mean_evaluations'],0)
            self.assertNotIn('control',result)
            self.assertEqual(result['baseline_control']['status'],'HISTORICAL_ONLY_NOT_RECOMPUTED_FOR_NEW_BUILDER')
            self.assertEqual(result['runtime']['openfermion'],'1.6.1')
            self.assertTrue(result['dictionary_eligible_for_configuration'])
            self.assertTrue((out/'COMMON_OPERATORS.npz').exists())
            self.assertFalse((out/'LD1_DICTIONARY.json').exists())
            self.assertEqual(json.loads((out/'result.json').read_text())['status'],N.success(2))
            with self.assertRaisesRegex(ValueError,'FRESH'):self.execute_fixture(out)

    def test_scalar_failure_keeps_gate_partial_evidence_and_runtime_without_retry(self):
        with tempfile.TemporaryDirectory()as tmp:
            out=Path(tmp)/'P0';result=self.execute_fixture(out,bad_scalar=True)
            self.assertEqual(result['status'],'STOP_P0_LD2_INPUT_OR_NUMERICAL_FAILURE')
            self.assertIn('Pauli_tail_saved_scalar_fro',result['error']['message'])
            self.assertFalse(result['checks'][-1]['PASS'])
            self.assertEqual(result['checks'][-1]['threshold'],1e-9)
            self.assertEqual(result['coefficient_policy']['openfermion_EQ_TOLERANCE'],1e-8)
            self.assertIn('runtime',result)
            self.assertFalse(result['dictionary_eligible_for_configuration'])
            self.assertTrue((out/'LD2_DICTIONARY.json').exists())
            self.assertTrue(result['mandatory_stop']);self.assertFalse(result['automatic_retry'])


if __name__=='__main__':unittest.main(verbosity=2)
