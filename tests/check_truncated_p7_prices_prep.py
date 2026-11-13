"""T3 parent-side checks: fixed p7 configurations/cells, C-guard sidecar format.

Metadata and fictional prices only; the circuit checks run in their own
provider-root process like the real child.
"""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import run_truncated_p7_prices as T3  # noqa: E402
from trottertracks import price_implementation_guard as G  # noqa: E402

PLANS = T3.plans()


class Tests(unittest.TestCase):
    def test_three_fixed_configurations_and_nine_cells(self):
        self.assertEqual([p['configuration_id'][:12] for p in PLANS], ['a6e557770a2e', 'defc5e88ff4d', '424eecbfdad0'])
        cells = [c for p in PLANS for c in p['cells']]
        self.assertEqual(len(cells), 9)
        self.assertEqual({(c['row']['method'], c['row']['prefix']) for c in cells}, {('B0I', 7)})
        pieces = {p['formula']: [c['signed_pieces'] for c in p['cells']] for p in PLANS}
        self.assertEqual(pieces, {'2nd': [7, 14, 28], '4th': [9, 18, 36], '4th(m5_best)': [11, 22, 44]})
        early = {p['formula']: p['plan']['configuration']['rounds'][3]['actual_delta_pf'] for p in PLANS}
        self.assertTrue(all(early[f] != PLANS[i]['plan']['configuration']['rounds'][6]['actual_delta_pf']
                            for i, f in enumerate(early)))

    def test_sidecar_passes_unchanged_c_guard_and_rte_route_rejected(self):
        p = PLANS[1]; c = p['cells'][0]
        trial = dict(scalar=2.29310124732 + 4.5302064712666724e-05,
                     prices={a: dict(rz_count=1000 + i) for i, a in enumerate(('cosine', 'sine'))})
        prices, sides, proofs = T3.price_bindings(T3.ROOT, p['plan'], c, trial, price_source=dict(fixture='ARTIFICIAL'),
            evidence_commit='a' * 40, evidence_path='ARTIFICIAL.json')
        config = p['plan']['configuration']
        cohorts = {(config['rounds'][c['m']]['candidate']['id'], a): T3.COHORT for a in ('X', 'Y')}
        policy = dict(id='ARTIFICIAL_PF', source=dict(fixture='ARTIFICIAL'), profile=deepcopy(config['spec']['profile']),
                      by_method=dict(TRUNCATED_PF=dict(native_policy=T3.NATIVE_POLICY, rte_rotation_lowering=G.NO_RTE,
                                                       scalar_quantity=T3.SCALAR_QUANTITY)))
        self.assertEqual({(x['classification'], x['expectation']) for x in prices}, {('DIRECT_COMPILED_DETERMINISTIC', 'DETERMINISTIC')})
        self.assertEqual({x['diagnostic']['record_role'] for x in prices}, {'EXACT_DETERMINISTIC_CIRCUIT_PRICE'})
        with patch.object(G, '_document', return_value=proofs):
            self.assertEqual(len(G.check(T3.ROOT, config, prices, cohorts=cohorts, bindings=sides, policy=policy)['checked']), 2)
        bad, badp = deepcopy(sides), deepcopy(proofs)
        for s in bad:
            s['implementation'].update(rte_rotation_lowering=G.NEW_ZZ, rte_lowering_source=dict(commit='01e068af7e06d9f1f5621a5b3800c30eb32ecb06',
                                       path='src/trottertracks/efficient_controlled_zz.py'))
        for v in badp.values():
            v['implementation'].update(bad[0]['implementation'])
        with patch.object(G, '_document', return_value=badp), self.assertRaises(ValueError):
            G.check(T3.ROOT, config, prices, cohorts=cohorts, bindings=bad, policy=policy)

    def test_circuit_checks_in_provider_root_process(self):
        out = subprocess.run([sys.executable, '-B', '-W', 'ignore', str(ROOT / 'tests/check_truncated_p7_prices_circuits.py')],
                             cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr[-3000:])
        self.assertIn('Ran 5 tests', out.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
