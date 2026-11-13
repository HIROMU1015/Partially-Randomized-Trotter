"""Pauli LD1/m0 pilot, parent side: frozen plan, saved-P0 bit order, C guard PR_PAULI_TAIL sidecar.

C's route-aware guard is loaded from its pinned blob (cfc0fae6) without merging.
Fictional prices only; the artificial circuits run in their own provider-root process.
"""
from copy import deepcopy
import importlib.util
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import run_pauli_ld1_m0_pilot as PILOT  # noqa: E402
P = PILOT.connection()
GUARD_COMMIT = 'cfc0fae6f1399bbc46689b4d88998b415fa47781'


def guard():
    name = 'trottertracks._price_guard_cfc0fae6'
    if name not in sys.modules:
        path = 'src/trottertracks/price_implementation_guard.py'
        mod = types.ModuleType(name); mod.__package__ = 'trottertracks'; sys.modules[name] = mod
        exec(compile(PILOT.blob(GUARD_COMMIT, path), GUARD_COMMIT + ':' + path, 'exec'), mod.__dict__)
    return sys.modules[name]


class Tests(unittest.TestCase):
    def test_frozen_plan_and_existing_seed_family(self):
        meta = PILOT.preflight()
        self.assertEqual(meta['configuration_id'], 'd905d2197d4b01df3074b3844c0f4a2b990c21b9c3e1a0e3b6ea13a488b1fd27')
        c = meta['legacy_physical_candidate']
        self.assertEqual((c['T'], c['q'], c['R_outer'], c['K'], c['prefix']), (.1, 1, 3, 2, 1))
        fn = P.existing_trial_seed()
        self.assertEqual(meta['seeds'], [fn(dict(id=meta['canonical_candidate']), meta['cohort'], 0, 0, 0)])
        self.assertEqual(meta['scalar'], 2.29310124732 + 0.4177175120626939)

    def test_saved_P0_bit_order_for_A(self):
        spec = importlib.util.spec_from_file_location('p0check', ROOT / 'docs/research/d_pauli_pilot_full_pf_prices_20261114/CHECK_P0_BIT_ORDER.py')
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        out = mod.main()
        self.assertTrue(out['pass'])
        for v in out['prefixes'].values():
            self.assertTrue(v['labels_reversed']); self.assertGreater(v['unconverted_mismatch'], 1.)

    def test_sidecar_passes_C_PR_PAULI_TAIL_guard(self):
        G = guard()
        meta = PILOT.preflight()
        result = dict(prices={a: dict(rz_count=500 + i) for i, a in enumerate(('cosine', 'sine'))}, scalar=meta['scalar'])
        source = dict(fixture='ARTIFICIAL', commit='5ccc1dfc341236d3fce2f745e7e9023447a027a7')
        prices, sides, proofs = PILOT.records(ROOT, result, source=source, evidence_commit='a' * 40, evidence_path='ARTIFICIAL.json')
        handoff = __import__('json').loads(PILOT.blob(P.PACKETS['commit'], P.PACKETS['path']))
        config = next(p for p in handoff['packets'] if p['prefix'] == 1)['configuration']
        cohorts = {(config['rounds'][0]['candidate']['id'], a): meta['cohort'] for a in ('X', 'Y')}
        policy = dict(id='ARTIFICIAL_PAULI', source=dict(fixture='ARTIFICIAL'), profile=deepcopy(config['spec']['profile']),
                      by_method=dict(PR_PAULI_TAIL=dict(native_policy=P.NATIVE_POLICY, rte_rotation_lowering=G.PAULI_D2,
                                                        scalar_quantity=P.SCALAR_QUANTITY)))
        self.assertEqual(P.LOWERING_ID, G.PAULI_D2)
        with patch.object(G, '_document', return_value=proofs):
            out = G.check(ROOT, config, prices, cohorts=cohorts, bindings=sides, policy=policy)
        self.assertEqual(len(out['checked']), 2)
        for mutate in (lambda f: f.pop('route'), lambda f: f.update(rte_rotation_lowering=G.NEW_ZZ),
                       lambda f: f.update(resolved_scalar=0.)):
            bad, badp = deepcopy(sides), deepcopy(proofs)
            for s in bad:
                mutate(s['implementation'])
            for k, v in badp.items():
                v['implementation'] = deepcopy(bad[0]['implementation'] if k.endswith('X') else bad[1]['implementation'])
            with patch.object(G, '_document', return_value=badp), self.assertRaises(ValueError):
                G.check(ROOT, config, prices, cohorts=cohorts, bindings=bad, policy=policy)
        dfpolicy = deepcopy(policy); dfpolicy['by_method'] = dict(PR=policy['by_method']['PR_PAULI_TAIL'])
        with patch.object(G, '_document', return_value=proofs), self.assertRaises(ValueError):
            G.check(ROOT, config, prices, cohorts=cohorts, bindings=sides, policy=dfpolicy)

    def test_artificial_circuits_in_provider_root_process(self):
        out = subprocess.run([sys.executable, '-B', '-W', 'ignore', str(ROOT / 'tests/check_pauli_ld1_m0_pilot_circuits.py')],
                             cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr[-3000:])
        self.assertIn('Ran 5 tests', out.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
