"""Artificial checks for the T1 caller core. No H4 input, no saved science values."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('t1', ROOT/'scripts/run_h4_truncated_t1.py')
T1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(T1)
POLICY = dict(generator_hermiticity_and_reference_eigen_residual_ha=1e-9, reconstruction_fro=1e-9, energy_cluster_ha=1e-8)
GROUPS = T1.groups_function()


def random_toy(seed=3, dim=6, scales=(1., .5, .2, .05, .01)):
    rng = np.random.default_rng(seed); mats = []
    for s in scales:
        a = rng.normal(size=(dim, dim))+1j*rng.normal(size=(dim, dim)); h = (a+a.conj().T)/2
        mats.append(s*h/np.linalg.norm(h, 2))
    return mats


def run(mats, constant, p, *, c_tail=None, lam=None, state=None):
    dim = len(mats[0]); full = constant*np.eye(dim)+sum(mats)
    w, v = eigh(full); psi = v[:, 0] if state is None else state
    groups = GROUPS(w, abs(v.conj().T@psi)**2, POLICY['energy_cluster_ha'])
    tail = sum(mats[p+1:], np.zeros((dim, dim), complex))
    c = float(np.trace(tail).real/dim) if c_tail is None else c_tail
    lam = sum(np.linalg.norm(m-np.trace(m).real/dim*np.eye(dim), 2) for m in mats[p+1:])+abs(c-np.trace(tail).real/dim) if lam is None else lam
    return T1.prefix_record(p, matrices=mats, constant=constant, state=psi, full_evals=w, full_basis=v,
        full_ground=groups[0], c_tail=c, lam_R=lam, groups=GROUPS, policy=POLICY), full, c


class T1Core(unittest.TestCase):
    def test_shift_matches_direct_eigh_and_b0_is_constant_offset(self):
        mats = random_toy()
        for p in range(len(mats)-1):
            rec, full, c = run(mats, -1.3, p)
            retained = (-1.3+c)*np.eye(6)+sum(mats[:p+1])
            expected = eigh(retained)[0][rec['matched_group_index']]-eigh(full)[0][0]
            self.assertAlmostEqual(rec['delta_E_trunc_B0I_ha'], expected, places=13)
            self.assertAlmostEqual(rec['delta_E_trunc_B0_ha'], rec['delta_E_trunc_B0I_ha']-c, places=14)
            self.assertTrue(rec['weyl']['abs_shift_le_sector_norm'] and rec['weyl']['sector_norm_le_lambda_R'])
            self.assertIn(rec['status'], ('TRUNCATION_SHIFT_AVAILABLE', 'UNKNOWN'))
        self.assertEqual(run(mats, -1.3, 3)[0]['status'], 'TRUNCATION_SHIFT_AVAILABLE')

    def test_level_crossing_is_unknown_not_minimum_matching(self):
        m0 = np.diag([.1, 0, 1, 2, 3, 4]).astype(complex); m1 = np.diag([-.3, 0, 0, 0, 0, 0]).astype(complex)
        rec, _, _ = run([m0, m1, np.zeros((6, 6), complex)], 0., 0, c_tail=0., lam=.3)
        self.assertEqual(rec['status'], 'UNKNOWN')
        self.assertFalse(rec['matched_group_is_minimum'])
        self.assertAlmostEqual(rec['delta_E_trunc_B0I_ha'], .1-(-.2), places=14)  # matched, not minimum (0-(-.2))

    def test_numerical_gate_failure_is_unknown(self):
        mats = random_toy(); bad = [m.copy() for m in mats]; bad[1][0, 1] += .5  # non-Hermitian retained block
        rec, _, _ = run(bad, -1., 2)
        self.assertEqual(rec['status'], 'UNKNOWN'); self.assertIn('gate', rec['reason'])

    def test_inconsistent_lambda_is_unknown(self):
        rec, _, _ = run(random_toy(), -1., 1, lam=1e-6)
        self.assertEqual(rec['status'], 'UNKNOWN'); self.assertIn('Weyl', rec['reason'])

    def test_saved_coefficients_compared_and_used(self):
        extractor = lambda ham, p: (float(p)/10, float(10-p))
        saved = {'1': dict(scalar=.1, lambda_R=9.), '7': dict(scalar=.7, lambda_R=3.)}
        rows = T1.tail_coefficients(None, saved, extractor)
        self.assertTrue(rows[1]['exact_match']); self.assertEqual(rows[4]['c_tail_source'], 'EXISTING_EXTRACTOR')
        self.assertEqual(rows[7]['used_c_tail'], .7)
        with self.assertRaises(ValueError):
            T1.tail_coefficients(None, {'7': dict(scalar=.71, lambda_R=3.)}, extractor)

    def test_cache_mismatch_stops(self):
        mats = random_toy(); full = -1.*np.eye(6)+sum(mats); w, v = eigh(full)
        cache = dict(eigenvalues_ha=w.tolist(), eigenvectors=dict(real=v.real.tolist(), imag=v.imag.tolist()))
        T1.check_full_eigensystem(full, cache, POLICY)
        with self.assertRaises(ValueError):
            T1.check_full_eigensystem(full+1e-6*np.eye(6), cache, POLICY)


if __name__ == '__main__':
    unittest.main(verbosity=2)
