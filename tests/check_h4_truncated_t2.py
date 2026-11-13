"""Artificial + metadata checks for the T2 caller. No H4 arrays, no saved science values."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
spec = importlib.util.spec_from_file_location('t2', ROOT/'scripts/run_h4_truncated_t2.py')
T2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(T2)
from trottertracks import baseline_signal_numerics as N  # noqa: E402
K = N.kernels(str(ROOT))
POLICY = dict(generator_hermiticity_and_reference_eigen_residual_ha=1e-9, reconstruction_fro=1e-9,
              energy_cluster_ha=1e-8, signal_difference_absolute=1e-9)
TOL = dict(unitarity=1e-9, reconstruction=1e-9, generator_hermiticity_ha=1e-9, energy_cluster_ha=1e-8,
           phase_guard_rad=1e-8, source='TOY')
PRINCIPAL = dict(mode='PRINCIPAL', source='TOY')


class ToyCtx:
    def __init__(self, mats, constant):
        self.matrices = mats; self.constant = constant
        w, v = eigh(constant*np.eye(len(mats[0]))+sum(mats)); self.state = v[:, 0]

    def pf_step(self, c, weights):
        u = np.eye(len(self.state), dtype=complex)
        for w in weights:
            u = K.expm_step(self.matrices[:c['prefix']+1], (1.,), c['T']*w/c['q'], {}) @ u
        return u


def toy(seed=5, dim=6, scales=(1., .5, .2, .05, .01)):
    rng = np.random.default_rng(seed); mats = []
    for s in scales:
        a = rng.normal(size=(dim, dim))+1j*rng.normal(size=(dim, dim)); h = (a+a.conj().T)/2
        mats.append(s*h/np.linalg.norm(h, 2))
    return mats


def run_cell(mats, constant, p, delta, weights=(1.,), lam=None, q_first=1, q12=8):
    ctx = ToyCtx(mats, constant); dim = len(ctx.state)
    full = constant*np.eye(dim)+sum(mats); wf, vf = eigh(full)
    groups = T2.T1.groups_function()(wf, abs(vf.conj().T@ctx.state)**2, 1e-8)
    tail = sum(mats[p+1:], np.zeros((dim, dim), complex)); c_tail = float(np.trace(tail).real/dim)
    hr = (constant+c_tail)*np.eye(dim)+sum(mats[:p+1]); wr, vr = eigh(hr)
    trunc = float(wr[0]-wf[0])
    cand = lambda q: dict(system='TOY', method='B0I', prefix=p, formula='toy', T=delta*q, q=q, delta=delta, id='toy%d' % q)
    cell = dict(prefix=p, formula='toy', actual_delta_pf=dict(numerator=int(delta*1000), denominator=1000),
                actual_delta_over_h='toy', signed_weights=list(weights),
                first_check=dict(m=0, physical_candidate=cand(q_first)), m12_check=dict(m=12, physical_candidate=cand(q12)))
    ref = dict(energy=float(wf[0]), definition='TOY', hamiltonian='TOY_FULL', source='TOY')
    op = dict(kind='TRUNCATED_PF', identity={}, source='TOY', target_hamiltonian='TOY_FULL',
              retained_hamiltonian=dict(construction='B0I'), scalar_policy='INCLUDED_ONCE_AT_STEP_TIME')
    cache = dict(source='TOY', hamiltonian='TOY_FULL', eigenvalues_ha=wf.tolist(),
                 eigenvectors=dict(real=vf.real.tolist(), imag=vf.imag.tolist()))
    saved = {}
    out = T2.diagnose_cell(cell, ctx=ctx, kernels=K, full_h=full, cache=cache, full_basis=vf, full_ground=groups[0],
        retained=(wr, vr, 0), c_tail=c_tail, lam_R=lam if lam is not None else float(np.linalg.norm(tail-c_tail*np.eye(dim), 2)),
        delta_trunc=trunc, reference=ref, operator=op, tolerances=TOL, branch=PRINCIPAL, policy=POLICY,
        budgets=dict(eps_trot=1e-3, eps_E=2e-3), save=lambda n, v: saved.__setitem__(n, v))
    return out, saved, trunc


class T2Core(unittest.TestCase):
    def test_decomposition_two_time_checks_and_files(self):
        mats = toy()
        for p in (1, 2, 3):
            out, saved, trunc = run_cell(mats, -1.2, p, .2, q_first=1, q12=40)
            self.assertEqual(out['status'], 'HEFF_CELL_AVAILABLE', out['reason'])
            self.assertLess(out['decomposition_identity_error_ha'], 1e-12)
            self.assertAlmostEqual(out['delta_E_total_ha'], trunc+out['delta_E_PF_ret_ha'], places=13)
            self.assertTrue(all(r['gate_pass'] for r in out['two_time_checks']))
            self.assertTrue({'OPERATORS.npz', 'SPECTRUM.npz', 'EIGENSYSTEM_SUMMARY.json',
                             'round_first_check.json', 'round_m12_check.json'} <= set(saved))
            self.assertNotIn('eigenvectors', saved['EIGENSYSTEM_SUMMARY.json']['spectrum'])

    def test_classification_labels(self):
        c = T2.classify(dict(delta_E_trunc_ha=-6e-4, delta_E_PF_ret_ha=5e-4, delta_E_total_ha=-1e-4),
                        lam_R=2e-3, eps_trot=8e-4, eps_E=1.6e-3)
        self.assertEqual(c['label'], 'CANCELLATION_DEPENDENT'); self.assertFalse(c['within_eps_trot']['L1'])
        c = T2.classify(dict(delta_E_trunc_ha=1e-5, delta_E_PF_ret_ha=5e-4, delta_E_total_ha=5.1e-4),
                        lam_R=3e-4, eps_trot=8e-4, eps_E=1.6e-3)
        self.assertEqual(c['label'], 'WITHIN_L2_ROBUST'); self.assertTrue(c['within_eps_trot']['L1'])
        c = T2.classify(dict(delta_E_trunc_ha=1e-5, delta_E_PF_ret_ha=9e-4, delta_E_total_ha=9.1e-4),
                        lam_R=3e-4, eps_trot=8e-4, eps_E=1.6e-3)
        self.assertEqual(c['label'], 'EXCEEDS_EPS_TROT_L3')

    def test_evaluator_gate_failure_is_unknown_not_stop(self):
        mats = toy(); bad = [m.copy() for m in mats]; bad[1] = bad[1]+0.3j*np.eye(6)  # non-Hermitian -> nonunitary step
        out, _, _ = run_cell(bad, -1., 2, .2)
        self.assertEqual(out['status'], 'UNKNOWN'); self.assertIsNotNone(out['reason'])

    def test_level_crossing_is_unknown(self):
        m0 = np.diag([.1, 0, 1, 2, 3, 4]).astype(complex); m1 = np.diag([-.3, 0, 0, 0, 0, 0]).astype(complex)
        z = np.zeros((6, 6), complex)
        out, _, _ = run_cell([m0, m1, z], 0., 0, .2)
        # retained p=0 ground is a different state: T1 would already be UNKNOWN; here the caller must not
        # silently match minima either (retained index 0 is not the full-H ground image).
        self.assertEqual(out['status'], 'UNKNOWN')


class T2Metadata(unittest.TestCase):
    """Published generators only; no H4 matrices, context or numerical leaves."""
    @classmethod
    def setUpClass(cls):
        from trottertracks import rpe_optimization_pipeline as I
        api = I.providers(str(ROOT)); base = T2.T1.read(dict(commit='a9abf8de75bf3432fcd714e868e5c14c8884b8f8',
            path='docs/research/pr_rpe_first_signal_menu_20261113/CONFIGURATION_LD1.json'))
        target = T2.T1.read(dict(commit='91349e703e0d09ef702ccd108745dfaf1acc29f7',
            path='docs/research/pr_rpe_first_signal_results_20261113/B_MEANS_LD1.json'))['target']
        phys = dict(commit='bba150834ca30bdb186fbfd58c55a8d51177b855',
                    path='docs/research/h4_pr_physical_error_contract_20261113/PHYSICAL_INPUTS.json')
        cls.cells = T2.plan_cells(api, I.C, base['spec'], target, base['spec']['binding'], phys)
        cls.follow = T2.T1.read(T2.A_FOLLOWUP_PLAN)

    def test_42_cells_with_checks_and_branch_rules(self):
        self.assertEqual(len(self.cells), 42)
        for c in self.cells:
            self.assertEqual(c['m12_check']['m'], 12)
            self.assertEqual(round(c['m12_check']['physical_candidate']['T'], 9), 409.6)
            b = T2.branch_for(c, self.follow)
            if c['formula'] == '4th(m5_best)' and c['actual_delta_over_h'] == '16':
                self.assertEqual((b['mode'], b['window']), ('ENERGY_WINDOW', [-2.5, 1.0]))
            else:
                self.assertEqual(b['mode'], 'PRINCIPAL')
        x = [c for c in self.cells if (c['prefix'], c['formula'], c['actual_delta_over_h']) == (7, '2nd', '1')][0]
        self.assertEqual(x['first_check']['physical_candidate']['id'], 'd2da7de820833a4c6334')  # saved B0I m0 ID
        self.assertEqual(sum(1 for u in x['uses'] if u['requested_delta_over_h'] == '1'), 13)


if __name__ == '__main__':
    unittest.main(verbosity=2)
