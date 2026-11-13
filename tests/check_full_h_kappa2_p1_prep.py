"""P-1 parent-side checks: fixed row/seeds, pinned SOURCE, C-guard sidecar format.

Metadata and fictional price values only. The circuit checks run in their own
process (provider root first), exactly like the real child.
"""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import run_full_h_kappa2_p1 as P1  # noqa: E402
from trottertracks import price_implementation_guard as G  # noqa: E402

OLD = 'CONTROLLED_ZZ_GENERIC_RZZ_CONTROL1_PROVIDER_C31EF424'


class Tests(unittest.TestCase):
    def test_fixed_row_seeds_and_pinned_sources(self):
        P1.source_checks()
        prepared, req, row, seeds = P1.plan_row()
        self.assertEqual((row['q'], row['R_outer'], row['s'], row['K'], row['T'], row['prefix']), (8, 71, 568, 2, .8, 0))
        self.assertEqual(row['id'], 'FULL_H_4de6130f0d3a66e5d695')
        self.assertEqual(len(set(seeds)), 8)

    def test_price_sidecar_format_passes_unchanged_c_guard(self):
        prepared, req, row, seeds = P1.plan_row()
        trial = dict(scalar=-0.33147781341681615, prices={a: dict(rz_count=100 + i) for i, a in enumerate(('cosine', 'sine'))})
        prices, sides, proofs = P1.price_bindings(ROOT, prepared, trial, price_source=dict(fixture='ARTIFICIAL'),
            evidence_commit='a' * 40, evidence_path='ARTIFICIAL_IMPLEMENTATION_EVIDENCE.json')
        self.assertEqual([p['axis'] for p in prices], ['X', 'Y'])
        self.assertEqual({p['expectation'] for p in prices}, {'NOT_A_LAW_MEAN'})
        config = prepared['plan']['configuration']
        cohorts = {(config['rounds'][P1.M]['candidate']['id'], a): P1.COHORT for a in ('X', 'Y')}
        policy = dict(id='ARTIFICIAL_NEW_ZZ', source=dict(fixture='ARTIFICIAL'), profile=deepcopy(config['spec']['profile']),
                      by_method=dict(FULL_H_RTE=dict(native_policy='H46_LONG_TIME_NATIVE_COMPOSE',
                                                     rte_rotation_lowering=G.NEW_ZZ, scalar_quantity=P1.SCALAR_QUANTITY)))
        with patch.object(G, '_document', return_value=proofs):
            out = G.check(ROOT, config, prices, cohorts=cohorts, bindings=sides, policy=policy)
        self.assertEqual(len(out['checked']), 2)
        bad, bad_proofs = deepcopy(sides), deepcopy(proofs)
        for s in bad:
            s['implementation']['rte_rotation_lowering'] = OLD
        for p in bad_proofs.values():
            p['implementation']['rte_rotation_lowering'] = OLD
        with patch.object(G, '_document', return_value=bad_proofs), self.assertRaises(ValueError):
            G.check(ROOT, config, prices, cohorts=cohorts, bindings=bad, policy=policy)
        wrong = deepcopy(sides); wrong[0]['implementation']['resolved_scalar'] = 0.
        with patch.object(G, '_document', return_value=proofs), self.assertRaises(ValueError):
            G.check(ROOT, config, prices, cohorts=cohorts, bindings=wrong, policy=policy)

    def test_circuit_checks_in_provider_root_process(self):
        out = subprocess.run([sys.executable, '-B', '-W', 'ignore', str(ROOT / 'tests/check_full_h_kappa2_p1_circuits.py')],
                             cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr[-3000:])
        self.assertIn('Ran 4 tests', out.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
