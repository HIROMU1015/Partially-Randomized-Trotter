"""Artificial loader checks for the S-1 kappa2 caller.

The real schedule/plan is metadata only. Execution checks replace the saved
H4 context by a toy 4-dimensional Context (11 artificial Hermitian matrices,
real binding/constant metadata), the saved reference reader by a toy pair, and
the finite-K normalizer by an artificial function, so no H4 input file is read
and no real kappa2 normalizer, mean or logB is computed.
"""
from copy import deepcopy
import csv
import importlib.util
import json
import math
from pathlib import Path
import sys
import shutil
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('s1_caller', ROOT / 'scripts/run_full_h_kappa2_signals.py')
S1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(S1)
from trottertracks import baseline_signal_numerics as N  # noqa: E402

SOURCE = dict(commit=S1.NUMERICAL_SOURCE, caller=S1.CALLER, test='ARTIFICIAL_LOADER_ONLY')
PREPARED = S1.prepare(ROOT, SOURCE)


def toy_context(seed=7):
    import numpy as np
    rng = np.random.default_rng(seed); mats = []
    for _ in range(11):
        a = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)); mats.append(.05 * (a + a.conj().T))
    state = rng.normal(size=4) + 1j * rng.normal(size=4); state /= np.linalg.norm(state)
    binding = PREPARED['manifest']['shared']['binding']
    saved = SimpleNamespace(state=state, matrices=mats, ham=SimpleNamespace(constant=2.29310124732))
    return N.Context(str(ROOT), saved, binding, dict(source='ARTIFICIAL_TOY_CONTEXT'))


def toy_reader(bad_m=None):
    def reader(descriptor):
        z = dict(real=.6, imag=.8)
        other = dict(real=.6, imag=.8 + (1e-6 if descriptor['m'] == bad_m else 0.))
        return dict(deepcopy(descriptor), reference=z, independent_reference=other, signal_difference=0.)
    return reader


class ToyNormalizer:
    """Artificial finite normalizer: b(tau)=exp(f*tau^2) (f<=1 obeys the envelope)."""
    def __init__(self, factor=.9, ulp_above=False):
        self.factor, self.ulp_above, self.calls = factor, ulp_above, 0

    def __call__(self, tau, K):
        self.calls += 1
        b = math.exp(self.factor * tau * tau)
        if self.ulp_above:
            b = math.nextafter(math.exp(tau * tau), math.inf)
        return SimpleNamespace(exact_finite_distribution=b)


class Tests(unittest.TestCase):
    def setUp(self):
        self.real_kernels = N.kernels
        self.normalizer = ToyNormalizer()
        real = self.real_kernels(str(ROOT))
        fake = SimpleNamespace(**vars(real)); fake.finite_rte_distribution = self.normalizer
        N.kernels = lambda root: fake
        self.load = N.load_shared_h4
        N.load_shared_h4 = lambda *a, **k: (_ for _ in ()).throw(AssertionError('Saved H4 input must not be read'))

    def tearDown(self):
        N.kernels = self.real_kernels; N.load_shared_h4 = self.load

    def run_batch(self, *, reader=None, context=None):
        out = Path(tempfile.mkdtemp(prefix='s1_artificial_'))
        self.addCleanup(shutil.rmtree, out, True)
        return out, S1.run(ROOT, PREPARED, out, context_factory=context or (lambda: toy_context()),
                           reader=reader or toy_reader())

    def test_metadata_plan_is_fixed_kappa2_and_disjoint_from_r448(self):
        S1.numerical_sources(ROOT)
        plan = PREPARED['plan']
        self.assertEqual(plan['configuration']['id'], S1.CONFIGURATION_ID)
        reqs = plan['pending']['requests']
        self.assertEqual([r['physical_candidate']['q'] for r in reqs], [1 << m for m in range(13)])
        self.assertEqual(reqs[12]['physical_candidate']['R_outer'], 36223)
        self.assertEqual({r['comparison_method'] for r in reqs}, {'FULL_H_RTE'})
        r448 = {r['physical_candidate']['id'] for r in S1.read(S1.R448_PLAN)['requests']}
        self.assertFalse(r448 & {r['physical_candidate']['id'] for r in reqs})
        self.assertEqual(PREPARED['plan']['configuration']['spec']['kappa']['value'], dict(numerator=2, denominator=1))

    def test_toy_batch_complete_with_envelope_gate_and_packet(self):
        out, result = self.run_batch()
        self.assertEqual(result['status'], S1.COMPLETE, result)
        self.assertEqual((result['new_mean_evaluations'], result['saved_copies']), (13, 0))
        self.assertEqual(self.normalizer.calls, 13)
        self.assertTrue(all(c['gate'] == 'PASS' for c in result['post_gates']))
        packet = json.loads((out / S1.LABEL / 'B_SIGNAL_INPUT.json').read_text())
        self.assertEqual(packet['configuration_id'], S1.CONFIGURATION_ID)
        self.assertEqual(len(packet['means']), 13)
        for mean, row in zip(packet['means'], PREPARED['rows']):
            self.assertEqual(mean['law_binding']['kappa']['value'], dict(numerator=2, denominator=1))
            self.assertEqual(mean['physical_candidate']['id'], row['physical_candidate_id'])
            self.assertLessEqual(mean['normalization']['log_B'], .9 * row['paper_logB_envelope_float'] * (1 + 1e-8))
        with (out / 'SIGNALS.csv').open() as f:
            self.assertEqual(len(list(csv.DictReader(f))), 13)

    def test_one_ulp_allowance_is_needed_and_kept(self):
        self.normalizer.ulp_above = True
        out, result = self.run_batch()
        self.assertEqual(result['status'], S1.COMPLETE, result)
        strict = [c for c in result['post_gates'] if c['log_B'] > math.nextafter(c['envelope'], math.inf)]
        self.assertTrue(strict, 'artificial b above rounded exp(tau^2) should exceed the bare envelope')

    def test_envelope_violation_stops_without_packet(self):
        self.normalizer.factor = 1.5
        out, result = self.run_batch()
        self.assertEqual(result['status'], 'STOP_POST_GATE_UNRESOLVED')
        self.assertFalse((out / S1.LABEL / 'B_SIGNAL_INPUT.json').exists())
        self.assertTrue(all(c['gate'] == 'FAIL' for c in result['post_gates']))

    def test_first_numerical_failure_stops_without_retry(self):
        out, result = self.run_batch(reader=toy_reader(bad_m=3))
        self.assertEqual(result['status'], 'STOP_NUMERICAL_UNRESOLVED')
        self.assertEqual(result['new_mean_evaluations'], 4)
        produced = json.loads((out / S1.LABEL / 'RESULT.json').read_text())
        self.assertEqual([m['status'] for m in produced['means']][4:],
                         ['NOT_STARTED_AFTER_NUMERICAL_STOP'] * 9)
        self.assertFalse((out / S1.LABEL / 'B_SIGNAL_INPUT.json').exists())

    def test_context_failure_is_caller_stop(self):
        def broken():
            raise ValueError('toy input unavailable')
        out, result = self.run_batch(context=broken)
        self.assertNotEqual(result['status'], S1.COMPLETE)
        self.assertEqual(result['new_mean_evaluations'], 0)

    def test_tampered_candidate_or_r448_overlap_rejected(self):
        original = S1.read
        def tampered(d):
            value = original(d)
            if d == S1.CANDIDATES:
                value['candidates'][0]['rounds'][5]['physical_candidate_id'] = 'FULL_H_tampered'
            if d == S1.R448_PLAN:
                value['requests'][0]['physical_candidate']['id'] = PREPARED['rows'][0]['physical_candidate_id']
            return value
        for target in (S1.CANDIDATES, S1.R448_PLAN):
            def one(d, target=target):
                return tampered(d) if d == target else original(d)
            S1.read = one
            try:
                with self.assertRaises(ValueError):
                    S1.prepare(ROOT, SOURCE)
            finally:
                S1.read = original


if __name__ == '__main__':
    unittest.main(verbosity=2)
