"""Synthetic validation for archived-prefix extraction and independent PR paths."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))


class Signal(unittest.TestCase):
    def test_nested_sequence_occupation_matrix(self):
        import numpy as np
        from scipy.linalg import expm
        from types import SimpleNamespace
        from h6_dba_signal import outer_step
        result=outer_step([[.3,.1],[.1,-.2]],.05,SimpleNamespace(np=np,expm=expm))
        self.assertLess(np.linalg.norm(result.conj().T@result-np.eye(2)),1e-14)

    def test_saved_new_time_gate_is_exactly_scoped(self):
        from h6_dba_signal import validate_saved_primitive_gate
        gate=dict(status='PASS',new_time=[dict(block=i,times=[.025,-.025],probes_per_time=3,
                                             max_error=1e-14,status='PASS') for i in range(6)])
        self.assertEqual(validate_saved_primitive_gate(gate),gate)
        gate['new_time'][0]['times']=[.05,-.05]
        with self.assertRaises(ValueError):validate_saved_primitive_gate(gate)

    def test_forward_polynomial_agrees_with_frozen_horner(self):
        import numpy as np
        from trottertracks.resource_applicability.h6_matched_port_v1 import compact_signal
        from trottertracks.resource_applicability.ax2a_state_action import ActionBudget
        from h6_dba_signal import forward_finite
        import h6_dba_contract as c
        state=np.array([1.,1j])/np.sqrt(2);tail=np.array([[.4,.2],[.2,-.1]])
        actions=[lambda v,t:v*np.exp(-1j*t*np.array([.3,-.4]))]
        for K in (2,4):
            cell=c.cell(0,2,3,K)
            a=compact_signal(state,actions,lambda v:tail@v,cell=cell,T=.8,scalar=.11,lambda_r=.7,
                             budget=ActionBudget(1000,1000))
            b=forward_finite(state,actions,tail,cell,.11,.7,np)
            self.assertLess(abs(a['log_B']-b['log_B']),1e-14)
            for path in ('corrected','raw'):
                self.assertLess(np.linalg.norm(a['vectors'][path]-b['vectors'][path]),1e-13)
            self.assertLess(abs(b['signals']['raw']*np.exp(b['log_B'])-b['signals']['corrected']),1e-13)

    def test_frozen_prefix_matches_original_small_preparation(self):
        import numpy as np
        from types import SimpleNamespace
        from trotterlib.df_hamiltonian import DFHamiltonian
        from trotterlib.pr2_matched_accuracy_m1_execution import _prepare
        from h6_dba_signal import prefix_view
        ham=DFHamiltonian(.17,np.array([[.2,.04],[.04,-.3]]),np.array([.4,-.2]),
            (np.array([[.6,.1],[.1,-.2]]),np.array([[.3,.07],[.07,.5]])),{})
        whole=_prepare(ham,2);expected=_prepare(ham,1)
        restored=prefix_view(whole,ham,1,SimpleNamespace(np=np))
        self.assertNotEqual(restored.preparation_hash,expected.preparation_hash)
        self.assertEqual(restored.deterministic_blocks,whole.deterministic_blocks[:2])
        self.assertEqual(restored.randomized_block_indices,(1,))
        self.assertAlmostEqual(restored.exact_rte_lambda_r,expected.exact_rte_lambda_r,places=14)
        for a,b in zip(restored.tail_extraction.components,expected.tail_extraction.components):
            self.assertEqual(a.component_id,b.component_id)
            self.assertEqual(a.coefficient,b.coefficient)
            self.assertEqual(a.basis_hash,b.basis_hash)


if __name__=='__main__':unittest.main()
