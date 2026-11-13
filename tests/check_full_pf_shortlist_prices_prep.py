"""Full PF shortlist price caller, parent side: fixed cells, pairing with Truncated p7, guard sidecar, C fit inputs.

Metadata and fictional prices only (no compile, no saved price value read).
"""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(ROOT / 'tests')]
import run_full_pf_shortlist_prices as FP  # noqa: E402
import run_truncated_p7_prices as T3  # noqa: E402
from check_pauli_ld1_m0_pilot_prep import guard  # noqa: E402  (C cfc0fae6 guard from its blob)

PLANS = FP.plans()


def fake_trial(scalar, base):
    return dict(scalar=scalar, prices={a: dict(rz_count=base + i) for i, a in enumerate(('cosine', 'sine'))})


class Tests(unittest.TestCase):
    def test_cells_ids_roles_and_saved_receipt_exclusion(self):
        self.assertEqual([p['configuration_id'][:12] for p in PLANS], ['11cf8ca6677a', 'a799697fc8ca', '70f7b31744ec'])
        cells = [c for p in PLANS for c in p['cells']]
        self.assertEqual([c['row']['id'] for c in cells], ['b7c4e73538ec19070cde', '8df73ab164fac57bfd6c', '4f3984c6d381fa0ca848',
            '3cf8eddfb3b78e96f5d2', '72383d110445e479155d', '7727a3fec25e033629f1',
            '0cd1527c80f7b9841a83', 'c63a835541462a1c264b', '8db0bd3c1a5e1eeb541f'])
        self.assertEqual([c['action'] for c in cells].count('NEW_COMPILE'), 8)
        saved = [c for c in cells if c['action'] != 'NEW_COMPILE']
        self.assertEqual((saved[0]['row']['formula'], saved[0]['m'], len(saved[0]['saved_receipts'])), ('4th(m5_best)', 4, 2))
        self.assertEqual({(c['m'], c['role']) for c in cells}, {(4, 'CALIBRATION'), (5, 'CALIBRATION'), (6, 'HOLDOUT')})
        self.assertEqual({(c['row']['method'], c['row']['prefix']) for c in cells}, {('FULL', 10)})

    def test_pairs_with_truncated_p7_on_same_width_q_and_pieces(self):
        trunc = {(p['formula'], c['m']): c for p in T3.plans() for c in p['cells']}
        for p in PLANS:
            for c in p['cells']:
                t = trunc[(p['formula'], c['m'])]
                self.assertEqual((t['row']['q'], t['row']['T'], t['row']['delta'], t['signed_pieces']),
                                 (c['row']['q'], c['row']['T'], c['row']['delta'], c['signed_pieces']))
                self.assertEqual((t['row']['prefix'], t['row']['method']), (7, 'B0I'))

    def test_sidecars_pass_route_guard_for_full_and_truncated_together(self):
        G = guard()
        full, trunc = PLANS[1], T3.plans()[1]
        pairs = [(full, FP.price_bindings, fake_trial(2.29310124732, 900), FP.SCALAR_QUANTITY, 'FULL_PF'),
                 (trunc, T3.price_bindings, fake_trial(2.29310124732 + 4.5302064712666724e-05, 700), T3.SCALAR_QUANTITY, 'TRUNCATED_PF')]
        policy = dict(id='ARTIFICIAL_PF_PAIR', source=dict(fixture='ARTIFICIAL'), profile=deepcopy(full['plan']['configuration']['spec']['profile']),
                      by_method={m: dict(native_policy=T3.NATIVE_POLICY, rte_rotation_lowering=G.NO_RTE, scalar_quantity=q)
                                 for _, _, _, q, m in pairs})
        for plan, bind, trial, _, method in pairs:
            c = plan['cells'][0]; config = plan['plan']['configuration']
            prices, sides, proofs = bind(ROOT, plan['plan'], c, trial, price_source=dict(fixture='ARTIFICIAL'),
                                         evidence_commit='a' * 40, evidence_path='ARTIFICIAL.json')
            cohorts = {(config['rounds'][c['m']]['candidate']['id'], a): prices[0]['cohort'] for a in ('X', 'Y')}
            with patch.object(G, '_document', return_value=proofs):
                self.assertEqual(len(G.check(ROOT, config, prices, cohorts=cohorts, bindings=sides, policy=policy)['checked']), 2)
            self.assertEqual({p['quantity'] for p in prices}, {'FULL_WRAPPER_MEAN_COMPILED_PRICE'})
            self.assertEqual(prices[0]['price_profile'], dict(config['spec']['profile'], lowering_id=T3.NATIVE_POLICY))
        bad = deepcopy(policy); bad['by_method']['FULL_PF']['scalar_quantity'] = T3.SCALAR_QUANTITY
        prices, sides, proofs = FP.price_bindings(ROOT, full['plan'], full['cells'][0], fake_trial(2.29310124732, 900),
            price_source=dict(fixture='ARTIFICIAL'), evidence_commit='a' * 40, evidence_path='ARTIFICIAL.json')
        cohorts = {(full['plan']['configuration']['rounds'][4]['candidate']['id'], a): FP.COHORT for a in ('X', 'Y')}
        with patch.object(G, '_document', return_value=proofs), self.assertRaises(ValueError):
            G.check(ROOT, full['plan']['configuration'], prices, cohorts=cohorts, bindings=sides, policy=bad)

    def test_C_fit_inputs_from_full_and_truncated_records(self):
        V = FP.c_model()
        records = []
        for plan, bind, route in ((PLANS[0], FP.price_bindings, 'FULL_PF'), (T3.plans()[0], T3.price_bindings, 'TRUNCATED_PF')):
            got = []
            for i, c in enumerate(plan['cells']):
                got += bind(ROOT, plan['plan'], c, fake_trial(2.3, 100 * (i + 1)), price_source=dict(fixture='ARTIFICIAL'),
                            evidence_commit='a' * 40, evidence_path='ARTIFICIAL.json')[0]
            fits = FP.fit_inputs(got, route=route, formula=plan['formula'], width_over_h=plan['width_over_h'])
            self.assertEqual([f['role'] for f in fits], ['CALIBRATION'] * 4 + ['HOLDOUT'] * 2)
            self.assertEqual({f['record'] for f in fits}, {V.RECORD})
            records.append(fits)
        # Interface only: C's exact solver accepts the X-axis calibration pair (fictional values).
        cal = [f for f in records[0] if f['role'] == 'CALIBRATION' and f['axis'] == 'X']
        model = V.fit(cal, features=('one', 'q'), design_source=dict(fixture='ARTIFICIAL_INTERFACE_ONLY'))
        self.assertEqual((model['method'], model['qualified_for_candidate']), ('EXACT_SQUARE_SOLVE', False))

    def test_artificial_circuits_in_provider_root_process(self):
        out = subprocess.run([sys.executable, '-B', '-W', 'ignore', str(ROOT / 'tests/check_full_pf_shortlist_prices_circuits.py')],
                             cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr[-3000:])
        self.assertIn('Ran 2 tests', out.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
