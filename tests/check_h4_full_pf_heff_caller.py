"""Fixed metadata plus noncommuting 2x2 tests; never load saved H/state arrays."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tests'),str(ROOT/'src')]
import numpy as np
import run_h4_full_pf_heff as R
from check_pf_effective_hamiltonian import TOL, OP, REF, X, Z


class CallerTests(unittest.TestCase):
    def test_pinned_metadata_has_no_numerical_calls(self):
        with patch.object(R,'numerical_api',side_effect=AssertionError('No numerical call')):
            plan,config,reference,operator,tol=R.metadata()
        self.assertEqual(plan['proposed_signal_rounds'],[0,12])
        self.assertEqual([config['rounds'][m]['q_PF'] for m in (0,12)],[1,4096])
        self.assertEqual(operator['kind'],'FULL_PF')
        self.assertFalse(reference['ground_state_certified'])
        path='src/trottertracks/baseline_signal_numerics.py'
        self.assertEqual(R.blob(R.C_SOURCE,path),R.blob(plan['sources']['numerics']['commit'],path))

    def test_frozen_C_context_noncommuting_two_time_scalar_and_spectrum(self):
        api=R.numerical_api();state=np.array([.8,.6j]);constant=.17
        saved=SimpleNamespace(state=state,matrices=[.3*X,.5*Z],ham=SimpleNamespace(constant=constant))
        ctx=api.Context(ROOT,saved,{'rank':1})
        rows=[]
        for m in (0,12):
            q=1<<m
            c=dict(prefix=1,formula='2nd',T=.1*q,q=q,id='ARTIFICIAL_ONLY')
            rows.append(dict(m=m,q_RPE=q,candidate_id='ARTIFICIAL_ONLY',physical_candidate=c))
        plan=dict(step_key=dict(actual_delta=dict(numerator=3602879701896397,denominator=36028797018963968)),
            numerical_policy=dict(signal_difference_absolute=1e-9))
        observed={}
        diag,records=R.diagnose(ctx,.3*X+.5*Z+constant*np.eye(2),rows,deepcopy(REF),deepcopy(OP),deepcopy(TOL),
            plan,lambda name,value:observed.__setitem__(name,value))
        self.assertEqual(len(records),2);self.assertTrue(all(r['gate_pass'] for r in records))
        self.assertTrue(all(r['raw_equals_corrected_at_B1'] for r in records))
        self.assertIsNone(diag['rpe_bias_ha']);self.assertIsNone(diag['global_RMSE_bound'])
        self.assertAlmostEqual(diag['operator']['scalar_constant_ha'],constant)
        self.assertEqual(set(observed),{'OPERATORS.npz','EIGENSYSTEM.json','round_0.json','round_12.json'})

    def test_nonunitary_failure_is_saved_before_any_power(self):
        ctx=SimpleNamespace(state=np.array([1.,0.]),constant=0.,pf_step=lambda *args:1.01*np.eye(2))
        plan=dict(step_key=dict(actual_delta=dict(numerator=1,denominator=10)),numerical_policy=dict(signal_difference_absolute=1e-9))
        observed={};row=dict(physical_candidate=dict(prefix=1,T=.1,q=1))
        with patch('trottertracks.pf_power_stability.power_diagnostics',side_effect=AssertionError('No power after STOP')) as power:
            with self.assertRaisesRegex(ValueError,'H_eff numerical/branch gate'):
                R.diagnose(ctx,Z,[row],REF,OP,TOL,plan,lambda n,v:observed.__setitem__(n,v))
        power.assert_not_called()
        self.assertEqual(observed['EIGENSYSTEM.json']['status'],'UNKNOWN')
        self.assertIn('unitarity_defect_fro',observed['EIGENSYSTEM.json']['diagnostics'])
        self.assertEqual(set(observed),{'OPERATORS.npz','EIGENSYSTEM.json'})

    def test_existing_output_is_never_reused(self):
        with tempfile.TemporaryDirectory() as out:
            with patch.object(R,'metadata',side_effect=AssertionError('No execution')) as metadata:
                with self.assertRaisesRegex(ValueError,'Fresh output'):R.execute(out,'UNUSED')
            metadata.assert_not_called()


if __name__=='__main__':unittest.main()
