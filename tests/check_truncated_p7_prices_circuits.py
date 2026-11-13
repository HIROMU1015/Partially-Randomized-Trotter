"""Artificial deterministic circuits for the T3 truncated price builder (provider-root process).

Toy 4-qubit DF Hamiltonian with three fragments; original prepare(), occupation
oracle and long-time native cost_trial with a counting compile_cached. No H4
input, transpile or price.
"""
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import h46_long_time_signal as signal  # noqa: E402
signal.install()
import run_truncated_p7_prices as T3  # noqa: E402

import numpy as np  # noqa: E402


def toy():
    from trotterlib.df_hamiltonian import DFHamiltonian
    from h46_new2_native import OccupationOracle, prepare
    rng = np.random.default_rng(3)
    a = rng.normal(size=(4, 4)); g = []
    for _ in range(3):
        b = rng.normal(size=(4, 4)); g.append(.3 * (b + b.T))
    ham = DFHamiltonian(.41, .3 * (a + a.T), np.array([.4, .3, .2]), tuple(g), {})
    full = np.zeros(16, complex)
    for i in range(16):
        if bin(i).count('1') == 2:
            full[i] = rng.normal() + 1j * rng.normal()
    full /= np.linalg.norm(full)
    oracle = OccupationOracle(ham, full, prefix=ham.n_blocks)
    qstate = np.zeros(16, complex); qstate[oracle.qindices] = oracle.initial
    return SimpleNamespace(ham=ham, oracle=oracle, qstate=qstate, preparations={p: prepare(ham, p) for p in (1, 2, 3)})


def row(method, prefix, formula, q=2, T=.4):
    return dict(system='TOY', method=method, prefix=prefix, formula=formula, T=T, q=q, delta=T / q,
                R_outer=None, K=None, id='TOY_%s_%d_%s_%d' % (method, prefix, formula, q))


class Tests(unittest.TestCase):
    def setUp(self):
        import h46_long_time_native as native
        self.native = native
        self.tmp = Path(tempfile.mkdtemp(prefix='t3_artificial_')); self.addCleanup(shutil.rmtree, self.tmp, True)
        native.PUBLIC, native.PRIVATE = self.tmp / 'public', self.tmp / 'private'
        (self.tmp / 'private').mkdir()
        self.ctx = toy(); native.CONTEXT['TOY'] = self.ctx
        self.calls = []

        def fake(wrapper, fingerprint, representative):
            self.calls.append(fingerprint)
            return dict(metrics={k: 1 for k in native.METRICS}), None, False
        native.compile_cached = fake

    def test_provider_modules_pinned(self):
        self.assertTrue(T3.provider_checks())

    def test_b0i_prefix_blocks_scalar_and_signed_pieces(self):
        for formula, pieces in (('2nd', 1), ('4th', 3), ('4th(m5_best)', 11)):
            trial, facts, ok = T3.checked_trial(self.native, self.ctx, row('B0I', 2, formula), 'TOY')
            self.assertTrue(ok, facts)
            self.assertEqual((facts['retained_blocks'], facts['fragment_order']), (3, [0, 1]))
            self.assertEqual(facts['signed_pieces'], 2 * pieces)
            self.assertEqual(facts['negative_pieces'] > 0, formula != '2nd')
            self.assertNotEqual(facts['tail_identity'], 0.)
            self.assertEqual(facts['scalar'], self.ctx.ham.constant + facts['tail_identity'])
        self.assertEqual(len(self.calls), 6)

    def test_full_pf_same_path_without_tail_identity(self):
        trial, facts, ok = T3.checked_trial(self.native, self.ctx, row('FULL', 3, '4th'), 'TOY')
        self.assertTrue(ok, facts)
        self.assertEqual(trial['scalar'], self.ctx.ham.constant)
        self.assertEqual(facts['retained_blocks'], 4)

    def test_saved_tail_scalar_mismatch_and_wrong_blocks_stop(self):
        prep = self.ctx.preparations[2]
        with self.assertRaises(ValueError):
            T3.checked_trial(self.native, self.ctx, row('B0I', 2, '2nd'), 'TOY',
                             expected_tail_scalar=prep.extracted_identity_coefficient + 1e-12)
        # A mismatched retained Hamiltonian is caught by the original index-0 oracle gate.
        self.ctx.preparations[2] = self.ctx.preparations[1]
        with self.assertRaises(ValueError):
            self.native.cost_trial((row('B0I', 2, '2nd'), 'TOY_BAD', 0))

    def test_scalar_once_in_composed_circuit(self):
        prep = self.ctx.preparations[2]; r = row('B0I', 2, '4th')
        scalar = self.ctx.ham.constant + prep.extracted_identity_coefficient
        from trottertracks.resource_applicability.h6_hash_reuse_v1 import use_fingerprint_reuse
        with use_fingerprint_reuse() as fp:
            evo = self.native.compose(r, prep, [], scalar, fp.fingerprint)
        last = evo.circuit.data[-1]
        self.assertEqual(last.operation.name, 'p'); self.assertEqual(float(last.operation.params[0]), -scalar * r['T'])
        same = [i for i in evo.circuit.data if i.operation.name == 'p' and float(i.operation.params[0]) == -scalar * r['T']]
        self.assertEqual(len(same), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
