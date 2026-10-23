"""Artificial 1-qubit regression only; no H4 data, chemistry or exponential.

This test intentionally checks the scientific reference contraction that the
sealed SOURCE implements. Its failure is preserved as a bug reproduction.
"""
import importlib.util
from pathlib import Path
import unittest
from flint import acb_mat
ROOT=Path(__file__).resolve().parents[4]
spec=importlib.util.spec_from_file_location('sealed_engine_reference_test',ROOT/'src/trotterlib/h4_ball_evidence.py')
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)

class ReferenceContraction(unittest.TestCase):
    def test_nonvacuum_state_identity_reference(self):
        state=acb_mat([[0],[1]])
        E=acb_mat([[1,0],[0,1]])
        expected=(e.dagger(state)*E*state)[0,0]
        actual=e.signal(state,E)  # Exact call shape used by sealed _tier.
        self.assertTrue(expected.contains(1))
        self.assertTrue(actual.contains(expected),
            'Sealed reference call omits E*state and selects the first column of state_dagger*E')

if __name__=='__main__':unittest.main(verbosity=2)
