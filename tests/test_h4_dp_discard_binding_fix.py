"""Discard binding regression. Molecular input reads and transpile are forbidden.

Real frozen builder tests use a 2-mode, 3-fragment artificial Hamiltonian.
Set H4_DP_FROZEN_PROVIDER to the preserved provider root; no H4 snapshot is read.
"""
from copy import copy
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import h4_dp_science as s
import run_h4_dp_verification as runner


def fake_prep():
    return SimpleNamespace(hamiltonian_hash='synthetic_truncated', num_system_qubits=2, ld=2,
        deterministic_fragment_indices=(0,1), randomized_block_indices=(),
        constant_coefficient=.25, exact_rte_lambda_r=0.)


class DiscardBinding(unittest.TestCase):
    def test_worker_context_does_not_duplicate_full_manifests(self):
        original=dict(stage='D',output='synthetic_output',native_config='synthetic_config',
            runtime_hashes={'synthetic_library':'a'*64},tasks=[dict(id='synthetic')],inputs=['synthetic'])
        minimal=runner.worker_context_spec(original)
        self.assertEqual(minimal,dict(stage='D',output='synthetic_output',native_config='synthetic_config'))
        self.assertIn('runtime_hashes',original)

    def test_p_worker_retains_fixed_cells_and_adapter(self):
        original=dict(stage='P',output='synthetic_output',native_config='synthetic_config',
            A_native_adapter='synthetic_adapter',cells=[dict(id='synthetic')],runtime_hashes={'unused':'hash'})
        minimal=runner.worker_context_spec(original)
        self.assertEqual(minimal['cells'],original['cells']);self.assertEqual(minimal['A_native_adapter'],'synthetic_adapter')
        self.assertNotIn('runtime_hashes',minimal)

    def test_valid_truncated_binding(self):
        s.validate_discard_preparation(fake_prep(), 'synthetic_truncated', 2, 2, .25)

    def test_full_hash_cannot_replace_truncated_hash(self):
        prep=fake_prep();prep.hamiltonian_hash='synthetic_full'
        with self.assertRaisesRegex(ValueError,'DISCARD_HAMILTONIAN_BINDING'):
            s.validate_discard_preparation(prep,'synthetic_truncated',2,2,.25)

    def test_changed_prefix_order_rejected(self):
        prep=fake_prep();prep.deterministic_fragment_indices=(1,0)
        with self.assertRaisesRegex(ValueError,'DISCARD_PARTITION_BINDING'):
            s.validate_discard_preparation(prep,'synthetic_truncated',2,2,.25)

    def test_nonempty_random_tail_rejected(self):
        prep=fake_prep();prep.randomized_block_indices=(2,)
        with self.assertRaisesRegex(ValueError,'DISCARD_PARTITION_BINDING'):
            s.validate_discard_preparation(prep,'synthetic_truncated',2,2,.25)

    def test_changed_scalar_rejected(self):
        with self.assertRaisesRegex(ValueError,'DISCARD_SCALAR_OR_EMPTY_TAIL_BINDING'):
            s.validate_discard_preparation(fake_prep(),'synthetic_truncated',2,2,.5)

    def test_nonzero_tail_norm_rejected(self):
        prep=fake_prep();prep.exact_rte_lambda_r=.01
        with self.assertRaisesRegex(ValueError,'DISCARD_SCALAR_OR_EMPTY_TAIL_BINDING'):
            s.validate_discard_preparation(prep,'synthetic_truncated',2,2,.25)


@unittest.skipUnless(os.environ.get('H4_DP_FROZEN_PROVIDER'),'frozen provider not selected')
class FrozenBuilderArtificialRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0,str(Path(os.environ['H4_DP_FROZEN_PROVIDER'])/'src'))
        import numpy as np
        from trotterlib.df_hamiltonian import DFHamiltonian
        from trotterlib.pr2_matched_accuracy_m1_execution import _prepare_discard
        from trotterlib.df_partial_randomized_pf import df_hamiltonian_hash
        cls.np=np;cls.hash=staticmethod(df_hamiltonian_hash);cls.builder=staticmethod(_prepare_discard)
        cls.ham=DFHamiltonian(constant=.25,one_body=np.array([[.3,.07],[.07,-.2]]),
            lambdas=np.array([.7,-.2,.11]),g_matrices=(np.array([[1.,.1],[.1,.3]]),
                np.array([[.2,-.15],[-.15,-.4]]),np.array([[.8,.05],[.05,.2]])),
            metadata={'purpose':'ARTIFICIAL_2_MODE_DISCARD_BINDING_FIX'})

    def test_frozen_builder_uses_truncated_hamiltonian_hash(self):
        import qiskit
        expected=self.hash(self.ham.select_blocks((0,1)))
        with (patch.object(self.np,'load',side_effect=AssertionError('NO_MOLECULAR_ARRAY_READ')),
              patch.object(qiskit,'transpile',side_effect=AssertionError('NO_TRANSPILE'))):
            prep=self.builder(self.ham,2)
        self.assertEqual(prep.hamiltonian_hash,expected)
        self.assertNotEqual(prep.hamiltonian_hash,self.hash(self.ham))
        s.validate_discard_preparation(prep,expected,2,2,.25)

    def test_prefix_cache_prepares_once_and_preserves_full_input(self):
        import qiskit
        before=self.hash(self.ham);calls=[]
        def build(ham,prefix):
            calls.append(prefix);return self.builder(ham,prefix)
        context=SimpleNamespace(ham=self.ham,m=SimpleNamespace(_prepare_discard=build),preparations={})
        with (patch.object(self.np,'load',side_effect=AssertionError('NO_MOLECULAR_ARRAY_READ')),
              patch.object(qiskit,'transpile',side_effect=AssertionError('NO_TRANSPILE'))):
            first=s.discard_preparation(context,2);second=s.discard_preparation(context,2)
        self.assertIs(first,second);self.assertEqual(calls,[2]);self.assertEqual(self.hash(self.ham),before)


if __name__=='__main__':unittest.main()
