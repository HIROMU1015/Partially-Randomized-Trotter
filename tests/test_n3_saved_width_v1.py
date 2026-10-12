"""Synthetic exact-algebra, provenance and failure-mode tests; no eight-input widths."""
import json
import copy
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction as F
import hashlib
import multiprocessing
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from trottertracks.representation_exploration.n3_saved_width_v1 import (
    Arithmetic, BITS, check_interval, compare, decode, interval, minimum,
)
from trottertracks.representation_exploration.n3_saved_evidence_v1 import Store, pointer, SHARD_CAP

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from run_n3_saved_width_v1 import authorization
from verify_n3_saved_width_v1 import verify_case


def synthetic_evaluation(payload):
    from trottertracks.representation_exploration.n3_saved_width_analysis_v1 import evaluate_case
    case, primary, secondary, fingerprint, seal = payload
    result = evaluate_case(case, primary, secondary, fingerprint, seal)
    if 'numpy' in sys.modules or 'trottertracks.representation_exploration.n3_occupation_transfer_v1' in sys.modules:
        raise AssertionError('unexpected numerical producer import')
    return result


class FixedWidthTests(unittest.TestCase):
    def test_positive_exact_root(self):
        w = interval(F(3), F(0), F(4))
        self.assertEqual(w['low'], [1, 1])
        self.assertEqual(w['high'], [1, 1])
        self.assertTrue(check_interval(3, 0, 4, w))

    def test_nondyadic_exact_root(self):
        self.assertEqual(interval(F(1, 3), 0, F(2, 9))['low'], [1, 3])

    def test_box_dominates_root(self):
        w = interval(2, 7, 1)
        self.assertEqual(w['low'], [5, 1])
        self.assertTrue(check_interval(2, 7, 1, w))

    def test_negative_q_and_zero_infimum(self):
        w = interval(3, 2, -100)
        self.assertTrue(w['infimum_zero'])
        self.assertGreater(decode(w['positive_proof_delta']), 0)
        self.assertTrue(check_interval(3, 2, -100, w))

    def test_negative_q_with_positive_box(self):
        self.assertEqual(interval(3, 5, -100)['low'], [2, 1])

    def test_irrational_enclosure(self):
        w = interval(1, 0, 1)
        lo, hi = decode(w['low']), decode(w['high'])
        self.assertLess(lo * (1 + lo), 1)
        self.assertGreater(hi * (1 + hi), 1)
        self.assertLessEqual(hi - lo, F(1, 1 << (BITS + 1)))
        self.assertTrue(check_interval(1, 0, 1, w))

    def test_rational_parameter_boundary_grid(self):
        # Independent polynomial certificate checks on synthetic rational triples.
        for g in (F(1, 7), F(1), F(5, 2)):
            for k in (F(0), F(1, 3), F(4)):
                for q in (F(-3), F(0), F(1, 101), F(9)):
                    with self.subTest(g=g, k=k, q=q):
                        self.assertTrue(check_interval(g, k, q, interval(g, k, q)))

    def test_energy_translation_keeps_q(self):
        # h->h+tI, U->U+nu*t, c->c+nu*t, s->s+nu*k*t.
        k, U, s, nu, t = F(2, 5), F(-3, 7), F(-1, 9), 2, F(13, 11)
        self.assertEqual(k * U - s, k * (U + nu * t) - (s + nu * k * t))

    def test_energy_scaling(self):
        w = interval(3, 0, 4)
        scaled = interval(6, 0, 16)
        self.assertEqual(decode(scaled['low']), 2 * decode(w['low']))

    def test_kappa_is_frozen(self):
        # A large fixed k forces a positive width even if the quadratic is free.
        self.assertEqual(interval(1, 2, 0)['low'], [1, 1])

    def test_norm_and_old_embeddings(self):
        nu, k, mu, U, beta2 = 2, F(1, 5), F(-1, 3), F(-1), F(2, 7)
        self.assertEqual(interval(3, 0, beta2), interval(3, 0, 0 * U - (-beta2)))
        self.assertEqual(interval(3, k, k * U - nu * mu), interval(3, k, k * U - (nu * mu - 0)))

    def test_bad_domains_and_floats_rejected(self):
        for args in ((0, 1, 1), (1, -1, 1), (1.0, 0, 1), (True, 0, 1)):
            with self.assertRaises(ValueError): interval(*args)
        for raw in ([1, 0], [True, 1], [1.0, 2]):
            with self.assertRaises(ValueError): decode(raw)

    def test_rational_cap_rejected(self):
        with self.assertRaises(ValueError): Arithmetic().q(F(1 << 4097))

    def test_unsafe_upper_rejected(self):
        w = interval(1, 0, 1); w['high'] = [1, 10]
        with self.assertRaises(ValueError): check_interval(1, 0, 1, w)

    def test_false_lower_rejected(self):
        w = interval(3, 0, 4); w['low'] = [2, 1]; w['high'] = [2, 1]; w['positive_proof_delta'] = [2, 1]
        with self.assertRaises(ValueError): check_interval(3, 0, 4, w)

    def test_strict_and_overlapping_comparisons(self):
        self.assertEqual(compare({'low': [1, 1], 'high': [1, 1]}, {'low': [2, 1], 'high': [2, 1]}), 'OCC_STRICTLY_NARROWER_IN_SAVED_SET')
        w = interval(1, 0, 1)
        self.assertEqual(compare(w, w), 'OVERLAPPING_INTERVALS_UNDETERMINED')
        self.assertEqual(compare(None, w), 'MISSING_SAVED_COVERAGE')
        self.assertEqual(compare(interval(3, 0, 4), interval(3, 0, 4)), 'EXACT_EQUAL_WIDTH')

    def test_saved_set_minimum(self):
        rows = [{'id': 'box', 'width': interval(1, 5, 0)}, {'id': 'root', 'width': interval(3, 0, 4)}]
        result = minimum(rows)
        self.assertEqual(result['upper_witness_id'], 'root')
        self.assertEqual(result['low'], [1, 1])
        self.assertIsNone(minimum([]))


class ExtractionTests(unittest.TestCase):
    def test_lossless_large_nested_values(self):
        value = {'matrix': [[[i * 101 + j, 103] for j in range(12)] for i in range(12)],
                 'repeated': [{'tau': [j, 7], 'Z': list(range(500))} for j in range(10)]}
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory); ref = store.put(value)
            store.flush()
            self.assertEqual(store.get(ref), value)
            self.assertTrue(all(p.stat().st_size <= SHARD_CAP for p in Path(directory).rglob('*.json')))
            self.assertEqual(store.put(value), ref)

    def test_pointer_escaping(self):
        self.assertEqual(pointer({'a/b': {'x~y': [7]}}, '/a~1b/x~0y/0'), 7)

    def test_tampered_fragment_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory); ref = store.put({'kappa': [1, 3]})
            store.flush()
            p = Path(directory) / store.locations[ref]
            raw = json.loads(p.read_bytes()); raw[ref]['value']['kappa'] = [2, 3]; p.write_text(json.dumps(raw))
            with self.assertRaises(ValueError): store.get(ref)


class ScopeAndSavedAuditTests(unittest.TestCase):
    def test_missing_authorization_denied_before_plan_load(self):
        r = subprocess.run([sys.executable, str(ROOT / 'scripts/run_n3_saved_width_v1.py'), '--evaluate'], capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        self.assertIn('separate explicit saved-width evaluation authorization required', r.stderr)
        self.assertNotIn('FileNotFoundError', r.stderr)
        self.assertFalse((ROOT / 'artifacts/n3_saved_width_v1/2026-10-12/evaluation1').exists())

    def test_old_batch_authorization_cannot_be_reused(self):
        old = json.loads((ROOT / 'docs/research/n3_occupation_transfer_v1_execution_authorization_20261012.json').read_bytes())
        with self.assertRaises(ValueError): authorization(old, {}, 'unused')

    def test_authorization_requires_each_binding(self):
        plan = {'source_commit': 's', 'source_fingerprint': 'f', 'original_payload_commit': 'p', 'ledger_sha256': 'l', 'input_sha256': 'i'}
        auth = dict(plan, scope='n3_saved_width_v1_one_posthoc_evaluation', evaluation_authorized=True,
                    plan_sha256='plan', attempts=1, retries=0, execution_branch='n3-saved-width-v1-evaluation-20261012',
                    output_path='artifacts/n3_saved_width_v1/2026-10-12/evaluation1', next_stage_authorized=False)
        authorization(auth, plan, 'plan')
        for key in ('source_commit', 'source_fingerprint', 'original_payload_commit', 'ledger_sha256', 'input_sha256', 'plan_sha256'):
            bad = dict(auth); bad[key] = 'wrong'
            with self.assertRaises(ValueError): authorization(bad, plan, 'plan')
        bad = dict(auth); bad['evaluation_authorized'] = False
        with self.assertRaises(ValueError): authorization(bad, plan, 'plan')

    def test_synthetic_two_worker_pipeline_and_independent_audit(self):
        directory = ROOT / 'artifacts/n3_occupation_transfer_v1/2026-10-12/preparation'
        pack = json.loads((directory / 'synthetic_pack.json').read_bytes())
        raw = (directory / 'synthetic_primary_ledger.json').read_bytes()
        primary = json.loads(raw)
        secondary = json.loads((directory / 'synthetic_secondary_ledger.json').read_bytes())
        seal = hashlib.sha256(raw).hexdigest()
        self.assertEqual(len(pack['cases']), 3)
        self.assertTrue(all(case['id'].startswith('synthetic_') for case in pack['cases']))
        payloads = [(case, p, s, p['source_fingerprint'], seal) for case, p, s in zip(pack['cases'], primary, secondary)]
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as pool:
            results = list(pool.map(synthetic_evaluation, payloads))
        for payload, result in zip(payloads, results):
            audit = verify_case(payload[0], payload[1], payload[2], result, payload[3], payload[4])
            self.assertEqual(audit['status'], 'SAVED_POSTHOC_CASE_POLYNOMIAL_AND_PSD_PASS')
            self.assertEqual(result['new_certificate_proposals'], 0)
        bad = copy.deepcopy(results[0]); bad['candidates'].pop()
        with self.assertRaises(ValueError): verify_case(*payloads[0][:3], bad, *payloads[0][3:])
        bad = copy.deepcopy(results[0]); bad['candidates'][0]['q'] = [99, 1]
        with self.assertRaises(ValueError): verify_case(*payloads[0][:3], bad, *payloads[0][3:])


if __name__ == '__main__':
    unittest.main()
