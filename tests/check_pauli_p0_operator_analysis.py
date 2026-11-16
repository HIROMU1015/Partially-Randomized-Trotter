"""Artificial matrices only: identity/traceless split and vector attribution."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from trottertracks import pauli_p0_operator_analysis as A


class Analysis(unittest.TestCase):
    def test_absolute_relative_identity_and_traceless_residual(self):
        target=np.diag([2.,4.]);approx=target+np.diag([3.,1.])
        d=A.error_metrics(approx,target)
        self.assertAlmostEqual(d['frobenius'],np.sqrt(10))
        self.assertAlmostEqual(d['identity_frobenius'],np.sqrt(8))
        self.assertAlmostEqual(d['nonidentity_frobenius'],np.sqrt(2))
        self.assertAlmostEqual(d['relative_frobenius'],np.sqrt(.5))
        self.assertLess(d['orthogonal_squared_norm_closure'],1e-13)
        self.assertFalse(d['within_original_absolute_gate'])
        self.assertEqual(d['absolute_gate'],1e-9)

    def test_saved_scalar_control_does_not_erase_traceless_error(self):
        target=np.diag([3.,1.]);nonidentity=np.diag([.75,-.75])
        d=A.saved_analysis(target,nonidentity,0.,2.)
        self.assertGreater(d['using_JW_identity']['frobenius'],d['using_saved_DF_scalar']['frobenius'])
        self.assertGreater(d['using_saved_DF_scalar']['nonidentity_frobenius'],0)

    def test_telescope_preserves_cancellation_not_scalar_norm_sum(self):
        target=np.diag([1.,2.]);shadow=target.copy();bridge=target+np.eye(2);default=target-.5*np.eye(2)
        d=A.attribute(default,bridge,shadow,target)
        self.assertEqual(d['shadow']['frobenius'],0.)
        self.assertLess(d['vector_telescope_closure_frobenius'],1e-15)
        self.assertGreater(d['JW_partial_sum_loss']['frobenius']+d['pre_JW_Fermion_loss']['frobenius'],
                           d['default']['frobenius'])
        self.assertFalse(d['production_dictionary_adopted'])

    def test_zero_target_and_nonfinite_rejected_without_fake_relative_value(self):
        self.assertIsNone(A.error_metrics(np.eye(2),np.zeros((2,2)))['relative_frobenius'])
        with self.assertRaises(ValueError):A.decomposition(np.array([[np.inf]]))
        with self.assertRaises(ValueError):A.error_metrics(np.eye(3),np.eye(2))


if __name__=='__main__':unittest.main(verbosity=2)
