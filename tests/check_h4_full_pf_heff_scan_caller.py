"""Synthetic noncommuting checks for the fixed scan caller; no saved arrays."""
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
import run_h4_full_pf_heff_scan as R
from check_pf_effective_hamiltonian import TOL,OP,REF,SOURCE,X,Z


def toy():
    api=R.previous.numerical_api();constant=.17;h=.3*X+.5*Z+constant*np.eye(2)
    state=np.array([.8,.6j]);energies,basis=np.linalg.eigh(h)
    saved=SimpleNamespace(state=state,matrices=[.3*X,.5*Z],ham=SimpleNamespace(constant=constant))
    ctx=api.Context(ROOT,saved,{'rank':1})
    cache=dict(source=SOURCE,hamiltonian=OP['target_hamiltonian'],eigenvalues_ha=energies.tolist(),
        eigenvectors=dict(real=basis.real.tolist(),imag=basis.imag.tolist()))
    operator=deepcopy(OP);operator['identity']=dict(binding={'fixture':'ARTIFICIAL_ONLY'})
    return ctx,h,cache,operator


def cell(formula):
    rows=[]
    for m in (0,3):
        q=1<<m
        rows.append(dict(m=m,q_RPE=q,canonical_candidate_id='ARTIFICIAL_ONLY',
            physical_candidate=dict(id='ARTIFICIAL_ONLY',prefix=1,formula=formula,T=.1*q,q=q)))
    return dict(label='ARTIFICIAL '+formula,formula=formula,
        actual_delta_pf=dict(numerator=3602879701896397,denominator=36028797018963968),
        first_check=rows[0],last_check=rows[1],signed_weights=list(R.previous.P._schedule().weights(formula)),uses=[])


class ScanCallerTests(unittest.TestCase):
    def test_fixed_plan_and_budget_are_metadata_only(self):
        with patch.object(R.previous,'numerical_api',side_effect=AssertionError('No real loader')):
            plan,reference,operator,tolerances,policy=R.load_plan()
        self.assertEqual(len(plan['step_cells']),15)
        self.assertEqual(sum(c['saved_source'] is None for c in plan['step_cells']),14)
        self.assertEqual(policy['signal_difference_absolute'],1e-9)
        self.assertFalse(reference['ground_state_certified'])

    def test_three_formulas_cache_scalar_and_budget_exceed_does_not_stop_diagnostics(self):
        ctx,h,cache,operator=toy()
        for formula in ('2nd','4th','4th(m5_best)'):
            observed={}
            with patch('scipy.linalg.eigh',side_effect=AssertionError('No new original-H solve')) as eig:
                result=R.diagnose_cell(ctx,h,cell(formula),reference=REF,operator=operator,
                    tolerances=TOL,cache=cache,policy=dict(signal_difference_absolute=1e-9),budget=1e-30,
                    save=lambda name,value:observed.__setitem__(name,value))
            eig.assert_not_called();self.assertEqual(len(result['round_checks']),2)
            self.assertTrue(result['numerical_gates_pass']);self.assertIsNone(result['global_RMSE_bound'])
            self.assertEqual(observed['EIGENSYSTEM.json']['reference_comparison']['eigensystem_origin'],'SUPPLIED_SOURCE_BOUND_CACHE')
            self.assertEqual(set(observed),{'OPERATORS.npz','EIGENSYSTEM.json','round_0.json','round_3.json'})
            if formula=='2nd':self.assertEqual(result['local_spectral_budget_screen'],'EXCEEDS_DECLARED_SPECTRAL_BUDGET_DIAGNOSTIC_ONLY')

    def test_failed_first_power_is_preserved_without_second_power(self):
        ctx,h,cache,operator=toy();observed={}
        with patch('trottertracks.pf_power_stability.power_diagnostics',return_value=(np.eye(2),{'gate_pass':False})) as power:
            with self.assertRaisesRegex(ValueError,'Two-time power/signal numerical gate'):
                R.diagnose_cell(ctx,h,cell('2nd'),reference=REF,operator=operator,
                    tolerances=TOL,cache=cache,policy=dict(signal_difference_absolute=1e-9),budget=.01,
                    save=lambda n,v:observed.__setitem__(n,v))
        self.assertEqual(power.call_count,1)
        self.assertIn('round_0.json',observed);self.assertNotIn('round_3.json',observed)
        self.assertFalse(observed['round_0.json']['gate_pass'])

    def test_existing_output_cannot_start_a_scientific_run(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(R,'load_plan',side_effect=AssertionError('No run')) as loader:
                with self.assertRaisesRegex(ValueError,'Fresh directory'):R.execute(directory,'UNUSED')
            loader.assert_not_called()


if __name__=='__main__':unittest.main()
