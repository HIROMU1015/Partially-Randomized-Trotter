"""Artificial window/caller and fixed metadata checks; no real H4 arrays."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'tests'),str(ROOT/'src')]
import numpy as np
from scipy.linalg import expm
import run_h4_full_pf_followup as R
from check_pf_effective_hamiltonian import evaluate,TOL,SOURCE,REF
from check_h4_full_pf_heff_scan_caller import toy,cell

BRANCH=dict(mode='ENERGY_WINDOW',purpose='EFFECTIVE_SPECTRUM_WINDOW',window=[-2.5,1.],source=SOURCE)


def saved_fixture():
    h=np.diag([-2.2,.7])
    return evaluate(expm(-.8j*h),.8,reference_hamiltonian=h)


class FollowupCallerTests(unittest.TestCase):
    def test_fixed_five_scope_excludes_held_and_preserves_exact_rounds_without_loader(self):
        with patch.object(R.previous,'numerical_api',side_effect=AssertionError('No science leaf')):
            plan,cells,ref,op,tol,policy=R.load_plan()
        self.assertEqual(len(cells),5)
        self.assertEqual([c['first_check']['m'] for _,c in cells],[4,3,4,5,4])
        self.assertEqual([c['last_check']['q_PF'] for _,c in cells],[1792,1536,768,640,256])
        self.assertEqual(cells[-1][1]['branch']['window'],[-2.5,1.])
        self.assertEqual(plan['metadata_counts']['held_new_actual_steps'],1)
        self.assertEqual(policy['signal_difference_absolute'],1e-9)

    def test_saved_window_preserves_groups_weights_and_reference_independence(self):
        saved=saved_fixture();before=deepcopy(saved)
        with patch('scipy.linalg.schur',side_effect=AssertionError('No new saved-spectrum solve')):
            result=R.check_saved_window(saved,BRANCH,.8,TOL)
        self.assertEqual(result['status'],'SAVED_WINDOW_COMPATIBILITY_DIAGNOSTIC_PASS')
        self.assertEqual(saved,before);self.assertTrue(result['state_weights_unchanged'])
        self.assertLess(result['metrics']['energy_lift_difference_max_ha'],1e-12)
        saved['reference']['energy']=900.
        self.assertEqual(result,R.check_saved_window(saved,BRANCH,.8,TOL))

    def test_saved_nonfinite_numerical_gate_and_wrong_window_stay_unknown(self):
        base=saved_fixture()
        for key in ('nonfinite','numerical','group','window'):
            saved=deepcopy(base);branch=deepcopy(BRANCH)
            if key=='nonfinite':saved['spectrum']['state_weights'][0]=float('nan')
            if key=='numerical':saved['diagnostics']['unitarity_defect_fro']=1.
            if key=='group':saved['spectrum']['energy_groups'][0]['indices']=[1]
            if key=='window':branch['window']=[-1.,1.]
            result=R.check_saved_window(saved,branch,.8,TOL)
            self.assertEqual(result['status'],'UNKNOWN',key)
            self.assertIsNone(result['global_RMSE_bound']);self.assertTrue(result['errors'])

    def test_explicit_branch_reuses_noncommuting_caller_without_eigh(self):
        ctx,h,cache,op=toy();observed={}
        with patch('scipy.linalg.eigh',side_effect=AssertionError('No new original-H solve')):
            summary=R.existing.diagnose_cell(ctx,h,cell('4th(m5_best)'),reference=REF,operator=op,
                tolerances=TOL,cache=cache,policy=dict(signal_difference_absolute=1e-9),budget=.001,
                branch=BRANCH,save=lambda n,v:observed.__setitem__(n,v))
        self.assertEqual(observed['EIGENSYSTEM.json']['branch'],BRANCH)
        self.assertTrue(summary['numerical_gates_pass']);self.assertEqual(len(summary['round_checks']),2)
        self.assertIsNone(summary['global_RMSE_bound'])

    def test_existing_directory_is_preserved_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(R,'load_plan',side_effect=AssertionError('No run')) as loader:
                with self.assertRaisesRegex(ValueError,'Fresh output directory'):R.execute(directory,'UNUSED')
            loader.assert_not_called()


if __name__=='__main__':unittest.main()
