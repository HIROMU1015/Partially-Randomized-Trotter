"""Additional artificial reference/state diagnostics; no molecular input IO."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
from scipy.linalg import expm

from check_pf_effective_hamiltonian import evaluate, matrix, OP, REF, X, Z
from trottertracks.pf_power_stability import expm_step


class ReferenceTests(unittest.TestCase):
    def test_original_and_effective_state_measures_keep_declared_proxy(self):
        h=.3*X+.5*Z
        energies,basis=np.linalg.eigh(h)
        state=.8*basis[:,0]+.6j*basis[:,1]
        reference=dict(REF,energy=-.5)
        r=evaluate(expm_step([.3*X,.5*Z],(1.,),.2),.2,
            state=state,reference=reference,reference_hamiltonian=h)
        self.assertEqual(r['status'],'EFFECTIVE_HAMILTONIAN_DIAGNOSTIC_AVAILABLE')
        c=r['reference_comparison'];v=matrix(c['eigenvectors'])
        np.testing.assert_allclose(c['state_weights'],[.64,.36],atol=1e-14)
        np.testing.assert_allclose((v*np.array(c['eigenvalues_ha']))@v.conj().T,h,atol=1e-14)
        self.assertEqual(r['reference']['energy'],-.5)
        self.assertFalse(c['reference_ground_energy_replaces_declared_reference'])
        self.assertIsNone(c['matched_effective_group'])
        d=r['spectral_shift_diagnostics']
        self.assertAlmostEqual(d['declared_reference_minus_matrix_minimum_ha'],-.5-energies[0])
        self.assertAlmostEqual(d['minimum_branched_energy_minus_reference_matrix_minimum_ha'],
            r['spectrum']['energies_ha'][0]-energies[0])
        for key in ('reference_eigen_residual_fro','reference_eigenvector_gram_defect_fro','reference_weight_sum_error'):
            self.assertLess(r['diagnostics'][key],1e-13)
        self.assertIsNone(r['rpe_bias_ha'])

    def test_degenerate_ground_projector_is_not_a_matched_effective_ground(self):
        h=np.diag([-1.,-1.,.4,.9]);heff=np.diag([-.7,-.9,.5,1.])
        r=evaluate(expm(-.1j*heff),.1,reference_hamiltonian=h)
        c=r['reference_comparison']
        self.assertEqual(c['energy_groups'][0]['multiplicity'],2)
        self.assertAlmostEqual(c['energy_groups'][0]['weight'],1.)
        np.testing.assert_allclose(c['reference_lowest_group_to_effective_groups_trace_overlap'],[1.,1.,0.,0.],atol=1e-14)
        self.assertTrue(c['trace_overlap_is_not_a_normalized_state_probability'])
        self.assertAlmostEqual(sum(c['reference_lowest_group_to_effective_groups_trace_overlap']),2.)
        self.assertAlmostEqual(r['spectrum']['minimum_energy_group_weight'],0.)
        self.assertIsNone(c['matched_effective_group'])
        self.assertFalse(r['ground_identification_qualified'])

    def test_method_adoption_is_separate_from_individual_rpe_qualification(self):
        for kind in ('FULL_PF','TRUNCATED_PF','PR_HEAD','PR_EXACT_TAIL'):
            op=deepcopy(OP);op['kind']=kind
            r=evaluate(expm(-.1j*Z),.1,operator=op)
            expected=('ADOPTED_MAIN_DETERMINISTIC_PF_ERROR_ROUTE' if kind in ('FULL_PF','TRUNCATED_PF')
                else 'NOT_ADOPTED_FOR_TOTAL_PR_ERROR')
            self.assertEqual(r['scientific_method_status'],expected)
            self.assertFalse(r['physical_alias_qualified']);self.assertFalse(r['ground_identification_qualified'])
            for key in ('rpe_bias_ha','sufficient_N','global_RMSE_bound','G'):
                self.assertIsNone(r[key])

    def test_unresolved_reference_eigensystem_keeps_failure_diagnostics(self):
        with patch('scipy.linalg.eigh',return_value=(np.array([-1.,1.]),np.ones((2,2)))):
            r=evaluate(expm(-.1j*Z),.1,reference_hamiltonian=Z)
        self.assertEqual(r['status'],'UNKNOWN')
        self.assertGreater(r['diagnostics']['reference_eigenvector_gram_defect_fro'],1.)
        self.assertIsNone(r['reference_comparison']);self.assertIsNone(r['spectral_shift_diagnostics'])
        self.assertIsNone(r['rpe_bias_ha'])


if __name__=='__main__':unittest.main()
