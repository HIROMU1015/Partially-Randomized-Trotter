"""Artificial 2/4 dimensional inputs only; no saved molecular arrays/results."""
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scipy.linalg import expm,logm
from trottertracks import pf_effective_hamiltonian as E
from trottertracks import pf_power_stability as P
from trottertracks.pf_pr_rpe_pipeline import _schedule

SOURCE=dict(fixture='ARTIFICIAL_ONLY')
X=np.array([[0.,1.],[1.,0.]],complex)
Y=np.array([[0.,-1j],[1j,0.]],complex)
Z=np.diag([1.,-1.]).astype(complex)
TOL=dict(unitarity=1e-9,reconstruction=1e-9,generator_hermiticity_ha=1e-9,
    phase_guard_rad=1e-8,energy_cluster_ha=1e-8,source=SOURCE)
OP=dict(kind='FULL_PF',source=SOURCE,identity={'candidate':'ARTIFICIAL','state':'ARTIFICIAL'},
    target_hamiltonian='TOY_FULL',retained_hamiltonian='TOY_FULL',scalar_policy='INCLUDED_ONCE_AT_STEP_TIME')
REF=dict(energy=-1.,definition='ARTIFICIAL_DECLARED_ENERGY',hamiltonian='TOY_FULL',source=SOURCE,
    ground_state_certified=False)
BRANCH=dict(mode='PRINCIPAL',source=SOURCE)


def matrix(payload):return np.array(payload['real'])+1j*np.array(payload['imag'])


def evaluate(u,delta,**kwargs):
    n=len(u);default=dict(state=np.eye(n,dtype=complex)[:,0],operator=deepcopy(OP),reference=deepcopy(REF),
        branch=deepcopy(BRANCH),tolerances=deepcopy(TOL))
    default.update(kwargs)
    return E.evaluate(u,delta,**default)


class EffectiveTests(unittest.TestCase):
    def available(self,r):
        self.assertEqual(r['status'],'EFFECTIVE_HAMILTONIAN_DIAGNOSTIC_AVAILABLE',r.get('reason'))
        for k in ('rpe_bias_ha','sufficient_N','global_RMSE_bound','G'):self.assertIsNone(r[k])
        self.assertFalse(r['physical_alias_qualified']);self.assertFalse(r['unitary_repaired'])

    def test_known_H_2_and_4_reconstructs_without_matrix_log_reimplementation(self):
        for h in (.3*X+.5*Z,np.kron(.3*X+.5*Z,np.eye(2))+.1*np.kron(np.eye(2),Z)):
            for delta in (.05,.4,1.):
                u=expm(-1j*delta*h);r=evaluate(u,delta,reference_hamiltonian=h);self.available(r)
                np.testing.assert_allclose(matrix(r['H_eff']),h,atol=3e-14)
                np.testing.assert_allclose(matrix(r['H_eff']),1j*logm(u)/delta,atol=3e-14)
                self.assertLess(r['diagnostics']['exponential_reconstruction_fro'],1e-13)

    def test_reuses_all_three_signed_PF_formulas_and_scalar(self):
        for formula in ('2nd','4th','4th(m5_best)'):
            weights=_schedule().weights(formula)
            u=P.expm_step([.2*X,.3*Z],weights,.1)*np.exp(-1j*.4*.1)
            r=evaluate(u,.1);self.available(r)
            self.assertLess(np.linalg.norm(matrix(r['H_eff'])-(.2*X+.3*Z+.4*np.eye(2))),1e-4)
            if formula!='2nd':self.assertTrue(any(w<0 for w in weights))

    def test_Strang_known_BCH_coefficient(self):
        a,b=.7,.4;h=a*X+b*Z
        # SU(2) Strang series: H_eff-H = d^2[-a*b^2 X/3 + a^2*b Z/6]+O(d^4).
        coefficient=-a*b*b*X/3+a*a*b*Z/6;errors=[]
        for d in (.16,.08,.04):
            r=evaluate(P.expm_step([a*X,b*Z],(1.,),d),d);self.available(r)
            errors.append(np.linalg.norm((matrix(r['H_eff'])-h)/d**2-coefficient))
        self.assertLess(errors[-1],1e-5)
        self.assertTrue(all(3.8<x/y<4.2 for x,y in zip(errors,errors[1:])))

    def test_explicit_window_recovers_shifted_spectrum_without_energy_oracle(self):
        h=np.diag([7.2,8.1]);u=expm(-1j*h)
        principal=evaluate(u,1.,reference_hamiltonian=h);self.available(principal)
        self.assertFalse(principal['diagnostics']['reference_norm_less_than_pi'])
        self.assertGreater(np.linalg.norm(matrix(principal['H_eff'])-h),6.)
        branch=dict(mode='ENERGY_WINDOW',purpose='EFFECTIVE_SPECTRUM_WINDOW',window=[7.,8.3],source=SOURCE)
        lifted=evaluate(u,1.,branch=branch,reference_hamiltonian=h);self.available(lifted)
        self.assertTrue(lifted['diagnostics']['centered_reference_norm_less_than_pi'])
        self.assertFalse(lifted['diagnostics']['reference_norm_less_than_pi'])
        np.testing.assert_allclose(matrix(lifted['H_eff']),h,atol=1e-13)
        other=evaluate(u,1.,branch=branch,reference=dict(REF,energy=900.))
        self.assertEqual(lifted['H_eff'],other['H_eff'])
        self.assertEqual(lifted['spectrum']['branch_integers'],[1,1])

    def test_cut_window_boundary_and_multiple_aliases_are_unknown(self):
        for value in (math.pi,math.pi+1e-11,math.pi-1e-11):
            r=evaluate(np.diag([np.exp(-1j*value),1.]),1.)
            self.assertEqual(r['status'],'UNKNOWN');self.assertIsNone(r['H_eff'])
        for window in ((-4.,4.),(2.,3.),(-.1,.5)):
            r=evaluate(expm(-1j*np.diag([-.1,.3])),1.,branch=dict(mode='ENERGY_WINDOW',
                purpose='EFFECTIVE_SPECTRUM_WINDOW',window=window,source=SOURCE))
            self.assertEqual(r['status'],'UNKNOWN')
        r=evaluate(expm(-1j*np.diag([math.pi,3.2])),1.,branch=dict(mode='ENERGY_WINDOW',
            purpose='EFFECTIVE_SPECTRUM_WINDOW',window=[3.,3.4],source=SOURCE))
        self.available(r)

    def test_lowest_eigenvalue_is_not_the_state_signal_or_RPE_bias(self):
        h=np.diag([-1.,1.]);r=evaluate(expm(-.2j*h),.2,state=np.array([0.,1.]))
        self.available(r)
        self.assertAlmostEqual(r['spectrum']['minimum_energy_group_weight'],0.)
        self.assertAlmostEqual(r['spectral_shift_diagnostics']['minimum_branched_energy_minus_declared_reference_ha'],0.)
        self.assertAlmostEqual(max(r['spectrum']['state_weights']),1.)
        self.assertIsNone(r['rpe_bias_ha'])

    def test_mixed_state_q_power_matches_full_spectral_measure(self):
        h=.3*X+.5*Z;psi=np.array([.6,.8j]);u=expm(-.2j*h)
        r=evaluate(u,.2,state=psi);self.available(r)
        for q in (1,2,7,64):
            powered,gate=P.power_diagnostics(u,q,psi);self.assertTrue(gate['gate_pass'])
            expected=np.vdot(psi,powered@psi)
            spectral=sum(w*np.exp(-1j*e*.2*q) for e,w in zip(r['spectrum']['energies_ha'],r['spectrum']['state_weights']))
            self.assertLess(abs(expected-spectral),4e-14)

    def test_degenerate_subspace_group_weight_not_arbitrary_basis_assignment(self):
        h=np.diag([-.5,-.5,.6,.6]);psi=np.array([.5,.5,.5,.5])
        r=evaluate(expm(-.3j*h),.3,state=psi,reference_hamiltonian=h);self.available(r)
        self.assertEqual([g['multiplicity'] for g in r['spectrum']['energy_groups']],[2,2])
        np.testing.assert_allclose([g['weight'] for g in r['spectrum']['energy_groups']],[.5,.5])
        self.assertTrue(r['reference_comparison']['degenerate_basis_correspondence_not_unique'])

    def test_truncated_H_and_identity_scalar_keep_original_reference(self):
        full=.4*X+.6*Z;retained=.4*X+.15*np.eye(2)
        reference=dict(REF,energy=-math.sqrt(.4**2+.6**2))
        operator=dict(OP,kind='TRUNCATED_PF',retained_hamiltonian={'method':'B0I','source':SOURCE})
        r=evaluate(expm(-.2j*retained),.2,operator=operator,reference=reference,reference_hamiltonian=full)
        self.available(r)
        self.assertAlmostEqual(r['spectral_shift_diagnostics']['minimum_branched_energy_minus_declared_reference_ha'],
            -.4+.15+math.sqrt(.4**2+.6**2))
        self.assertEqual(r['reference'],reference)
        r=evaluate(expm(-.2j*retained),.2,operator=operator,reference=dict(reference,hamiltonian='TRUNCATED'))
        self.assertEqual(r['status'],'UNKNOWN')

    def test_nonunitary_and_finite_K_operators_are_not_Hermitian_logs(self):
        r=evaluate(np.diag([1.,.98]),.1);self.assertEqual(r['status'],'UNKNOWN')
        self.assertGreater(r['diagnostics']['unitarity_defect_fro'],1e-9)
        for kind in ('PR_CORRECTED','PR_RAW','FULL_H_RTE'):
            r=evaluate(np.eye(2),.1,operator=dict(OP,kind=kind))
            self.assertEqual(r['status'],'NOT_APPLICABLE');self.assertIsNone(r['H_eff'])

    def test_original_H_norm_condition_alone_is_not_PF_branch_qualification(self):
        a=4*Z;b=-4*Z+.2*X;h=a+b;u=P.expm_step([a,b],(1.,),1.)
        r=evaluate(u,1.,reference_hamiltonian=h);self.available(r)
        self.assertTrue(r['diagnostics']['reference_norm_less_than_pi'])
        self.assertGreater(r['reference_comparison']['H_eff_minus_original_H_operator_norm'],.05)
        self.assertFalse(r['physical_alias_qualified'])

    def test_PR_exact_tail_reuses_existing_macro_and_ignores_R_K_only_for_control(self):
        spec=importlib.util.spec_from_file_location('artificial_PR',ROOT/'tests/check_pr_rpe_signal_provider.py')
        old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        batch,ctx=old.setup(M=0);job=batch.pending['jobs'][0];c=job['physical_context']['physical_candidate']
        controls=[]
        for R,K in ((2,2),(8,4)):
            candidate=dict(c,R_outer=R,K=K);candidate['id']=old.P.common.candidate_id(candidate)
            macro,_=old.P._macros(ctx,candidate,old.P.common.structure(candidate))
            u=macro['operators'][0]*np.exp(-1j*.1*(ctx.constant+ctx.tails[c['prefix']]['scalar']))
            controls.append(u)
            r=evaluate(u,.1,state=ctx.state,operator=dict(OP,kind='PR_EXACT_TAIL'))
            self.available(r)
        np.testing.assert_allclose(controls[0],controls[1],atol=1e-15)

    def test_common_step_metadata_does_not_qualify_varying_grid_or_finite_K(self):
        key=dict(kind='PR_EXACT_TAIL',binding='TOY',formula='2nd',prefix=1,scalar=.2,source=SOURCE,actual_delta=.1)
        rows=[dict(m=m,T=.1*2**m,q_PF=2**m,step_key=deepcopy(key),R_outer=2**m,K=2) for m in range(3)]
        r=E.common_step_schedule(rows);self.assertEqual(r['status'],'COMMON_STEP_DECLARED')
        self.assertFalse(r['common_step_qualified'])
        rows[-1]['step_key']['actual_delta']=.2;rows[-1]['q_PF']=2
        self.assertEqual(E.common_step_schedule(rows)['status'],'UNKNOWN_NO_COMMON_STEP')
        rows[-1]['step_key']['kind']='PR_CORRECTED'
        with self.assertRaises(ValueError):E.common_step_schedule(rows)

    def test_invalid_numeric_inputs_and_bad_Schur_do_not_get_repaired(self):
        for d in (0.,-1.,math.nan):
            r=evaluate(np.eye(2),d);self.assertEqual(r['status'],'UNKNOWN');json.dumps(r,allow_nan=False)
        r=evaluate(np.eye(2),.1,state=np.array([2.,0.]));self.assertEqual(r['status'],'UNKNOWN')
        r=evaluate(np.full((2,2),np.inf),.1);self.assertEqual(r['status'],'UNKNOWN')
        op=deepcopy(OP);op.pop('scalar_policy')
        self.assertEqual(evaluate(np.eye(2),.1,operator=op)['status'],'UNKNOWN')
        with patch('scipy.linalg.schur',return_value=(np.eye(2),np.eye(2)*2)):
            r=evaluate(np.eye(2),.1)
        self.assertEqual(r['status'],'UNKNOWN');self.assertGreater(r['diagnostics']['eigenvector_gram_defect_fro'],1.)

    def test_passive_import_has_no_scientific_libraries_or_loaders(self):
        code="import sys;sys.path.insert(0,'src');import trottertracks.pf_effective_hamiltonian;assert not any(v in sys.modules for v in ('numpy','scipy','qiskit','pyscf'))"
        subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,check=True)


if __name__=='__main__':unittest.main()
