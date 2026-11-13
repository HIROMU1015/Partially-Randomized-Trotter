"""Artificial cached-reference validation only; no saved numerical inputs."""
from copy import deepcopy
import unittest
from unittest.mock import patch
import numpy as np
from scipy.linalg import expm
from check_pf_effective_hamiltonian import evaluate, OP, SOURCE, X, Z


def fixture():
    h=.3*X+.5*Z;energies,basis=np.linalg.eigh(h)
    cache=dict(source=SOURCE,hamiltonian=OP['target_hamiltonian'],eigenvalues_ha=energies.tolist(),
        eigenvectors=dict(real=basis.real.tolist(),imag=basis.imag.tolist()),state_weights=['IGNORED_CACHE_WEIGHTS'])
    return h,basis[:,0],cache


class CacheTests(unittest.TestCase):
    def test_cache_reuses_basis_without_eigh_and_recomputes_current_state_weights(self):
        h,state,cache=fixture()
        with patch('scipy.linalg.eigh',side_effect=AssertionError('No repeated original-H solve')) as solver:
            r=evaluate(expm(-.2j*h),.2,state=state,reference_hamiltonian=h,reference_eigensystem=cache)
        solver.assert_not_called();self.assertEqual(r['status'],'EFFECTIVE_HAMILTONIAN_DIAGNOSTIC_AVAILABLE')
        c=r['reference_comparison'];self.assertEqual(c['eigensystem_origin'],'SUPPLIED_SOURCE_BOUND_CACHE')
        self.assertEqual(c['supplied_eigensystem_source'],SOURCE)
        np.testing.assert_allclose(c['state_weights'],[1.,0.],atol=1e-14)
        self.assertFalse(r['ground_identification_qualified']);self.assertIsNone(r['rpe_bias_ha'])

    def test_original_matrix_source_and_sorted_complete_basis_are_required(self):
        h,state,cache=fixture();bad=[]
        for field,value in [('source',None),('hamiltonian','OTHER_H'),('eigenvalues_ha',list(reversed(cache['eigenvalues_ha']))),
                ('eigenvectors',dict(real=[[1.]],imag=[[0.]]))]:
            item=deepcopy(cache);item[field]=value;bad.append(item)
        for item in bad:
            r=evaluate(expm(-.1j*h),.1,state=state,reference_hamiltonian=h,reference_eigensystem=item)
            self.assertEqual(r['status'],'UNKNOWN');self.assertIsNone(r['reference_comparison'])
        r=evaluate(expm(-.1j*h),.1,reference_eigensystem=cache)
        self.assertEqual(r['status'],'UNKNOWN');self.assertIn('requires its original reference H',r['reason'])

    def test_bad_cached_numerical_residual_does_not_fallback_or_repair(self):
        h,state,cache=fixture();cache['eigenvalues_ha'][0]-=.1
        with patch('scipy.linalg.eigh',side_effect=AssertionError('No fallback')) as solver:
            r=evaluate(expm(-.1j*h),.1,state=state,reference_hamiltonian=h,reference_eigensystem=cache)
        solver.assert_not_called();self.assertEqual(r['status'],'UNKNOWN')
        self.assertGreater(r['diagnostics']['reference_eigen_residual_fro'],.09)
        self.assertIsNone(r['global_RMSE_bound'])


if __name__=='__main__':unittest.main()
