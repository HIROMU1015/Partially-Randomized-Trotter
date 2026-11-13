"""Full PF saved-evidence inventory: pinned metadata plus artificial edits only.

No Hamiltonian/state load, evolution, reference, N/RMSE, compile or fit. Saved
values are compared for identity only; nothing is recomputed.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests')]
from trottertracks import full_pf_saved_inventory as I
from trottertracks import full_baseline_v2_requests as D

SCIENCE_MODULES = ('h46_long_time_signal', 'h46_long_time_native', 'qiskit', 'pyscf', 'h46_new2_native', 'h46_m5_tol_native')


class InventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = {s: hashlib.sha256((I.PRIVATE/(s+'_context.json')).read_bytes()).hexdigest()
                      for s in ('H4', 'H6') if (I.PRIVATE/(s+'_context.json')).exists()}
        cls.k = I.load_carriers(); cls.cov = I.coverage(cls.k)
        cls.rows = {r['physical_candidate_id']: r for r in cls.cov['rows']}

    def pick(self, **kw):
        return next(r for r in self.cov['rows'] if all(r[k] == v for k, v in kw.items()))

    def test_01_DF10_is_not_mixed_with_old_DF12(self):
        c = deepcopy(self.pick(system='H4', formula='2nd', T=.8, q_PF=1)['physical_candidate'])
        old = dict(c, prefix=12); old['id'] = D.p.candidate_id(old)
        self.assertIn('PREFIX_NOT_FULL_RANK', I.check_physical(old, self.k))
        k = deepcopy(self.k); k['science']['inputs']['H4']['binding'].update(rank=12, rank_policy='rank_12')
        self.assertIn('STUDY_BINDING_NOT_TOL_ONLY_DF10', I.check_physical(c, k))
        self.assertEqual(I.classify(c, k)['group'], 'PHYSICAL_MISMATCH')
        binding = self.k['science']['inputs']['H4']['binding']
        self.assertEqual((binding['rank'], binding['series']), (10, 'H4_TOL_ONLY_DF10'))
        prep = D.saved_full_h_preparation('H4')['binding']
        self.assertTrue(D.check_binding(binding, prep))

    def test_02_truncated_PF_is_not_full(self):
        inventory = self.k['files']['candidate_inventory.json.gz']
        b0 = next(r for r in inventory if r['method'] == 'B0' and r['system'] == 'H4' and r['prefix'] == 7)
        c = {f: b0[f] for f in I.CANDIDATE_FIELDS}
        reasons = I.check_physical(c, self.k)
        self.assertIn('NOT_FULL_PF_METHOD_B0', reasons); self.assertIn('PREFIX_NOT_FULL_RANK', reasons)
        self.assertTrue(all(r['physical_candidate']['method'] == 'FULL' and r['prefix'] == D.saved_grid()['rank'][r['system']]
                            for r in self.cov['rows']))

    def test_03_standard_4th_and_m5_best_are_separate(self):
        a = self.pick(system='H4', formula='4th', T=.8, q_PF=1); b = self.pick(system='H4', formula='4th(m5_best)', T=.8, q_PF=1)
        self.assertNotEqual(a['physical_candidate_id'], b['physical_candidate_id'])
        self.assertNotEqual(a['formula_id'], b['formula_id'])
        k = deepcopy(self.k); reused = k['files']['reused_compiled_prices.json.gz']
        reused[a['physical_candidate_id']] = deepcopy(reused[b['physical_candidate_id']])
        moved = I.classify(a['physical_candidate'], k)
        self.assertNotEqual(moved['X'], I.EXACT)   # an m5_best receipt cannot certify a 4th candidate

    def test_04_same_T_q_different_SOURCE_is_not_identical(self):
        r = self.pick(system='H4', formula='2nd', T=1.2, q_PF=2)
        self.assertEqual(r['X'], I.EXACT)
        k = deepcopy(self.k); k['science']['compiler']['hash'] = 'OTHER_COMPILER'
        self.assertEqual(I.classify(r['physical_candidate'], k)['X'], I.MISMATCH)
        k = deepcopy(self.k); k['files']['exploration_costs.json.gz'][r['physical_candidate_id']]['candidate']['prefix'] = 7
        self.assertEqual(I.classify(r['physical_candidate'], k)['X'], I.MISMATCH)
        k = deepcopy(self.k); k['files']['signal_records.json.gz'][r['physical_candidate_id']]['origin'] = 'OTHER_STUDY'
        self.assertEqual(I.classify(r['physical_candidate'], k)['signal']['status'], I.CONDITIONAL)

    def test_05_signal_reference_and_price_are_judged_separately(self):
        unpriced = self.pick(system='H4', formula='4th(m5_best)', T=1.2, q_PF=2)
        self.assertEqual(unpriced['group'], 'SIGNAL_ONLY_PRICE_MISSING')
        self.assertEqual(unpriced['signal']['status'], I.EXACT); self.assertEqual(unpriced['X'], I.MISSING)
        self.assertEqual(unpriced['reference']['status'], I.EXACT)
        reused = self.pick(system='H6', formula='4th', T=1.6, q_PF=8)
        self.assertEqual(reused['signal']['status'], I.CONDITIONAL)
        self.assertIn('RAW_SIGNAL_MISSING', reused['signal']['reasons'])
        self.assertEqual(reused['X'], I.EXACT)
        t2 = self.pick(system='H6', formula='2nd', T=2.0, q_PF=4)
        self.assertEqual(t2['reference']['independent_pair_status'], I.EXACT)
        self.assertEqual(self.pick(system='H6', formula='2nd', T=1.2, q_PF=2)['reference']['independent_pair_status'], I.MISSING)

    def test_06_X_and_Y_prices_are_not_filled_from_each_other(self):
        r = self.pick(system='H4', formula='2nd', T=1.2, q_PF=2)
        k = deepcopy(self.k); entry = k['files']['exploration_costs.json.gz'][r['physical_candidate_id']]
        entry['mean_metrics'].pop('sine')
        out = I.classify(r['physical_candidate'], k)
        self.assertEqual((out['X'], out['Y']), (I.EXACT, I.MISSING))
        self.assertIsNone(out['prices'][0]['axes']['Y']['rz_count'])

    def test_07_compiler_or_cost_policy_mismatch_is_detected(self):
        r = self.pick(system='H6', formula='4th', T=2.0, q_PF=4)
        k = deepcopy(self.k); k['science']['cost_policy'] = 'D_ONLY_UNCONTROLLED_STEP_COST'
        self.assertEqual(I.classify(r['physical_candidate'], k)['X'], I.MISMATCH)
        reused = self.pick(system='H4', formula='4th(m5_best)', T=.8, q_PF=1)
        k = deepcopy(self.k)
        for proof in k['files']['reused_compiled_prices.json.gz'][reused['physical_candidate_id']]['proof']:
            proof['receipt']['compiler_settings_hash'] = 'OTHER'
        self.assertEqual(I.classify(reused['physical_candidate'], k)['X'], I.MISMATCH)

    def test_08_saved_MODEL_is_not_a_direct_price(self):
        other = I.other_saved_series()
        model = [r for r in other['rows'] if r['kind'] == 'MODEL_PREDICTION']
        direct = [r for r in other['rows'] if r['kind'] == 'DIRECT_COMPILED_DETERMINISTIC']
        self.assertTrue(model and direct)
        self.assertTrue(all(r['status'] == 'MODEL_NOT_DIRECT_PRICE' for r in model))
        self.assertTrue(all(r['status'] == I.CONDITIONAL for r in direct))
        self.assertEqual({r['axis'] for r in direct}, {'X', 'Y'})

    def test_09_saved_FAIL_or_ineligible_is_not_promoted(self):
        statuses = {s for r in self.cov['rows'] for s in r['numerical']['statuses']}
        self.assertIn('INELIGIBLE', statuses)
        self.assertTrue(all(r['numerical']['rmse_policy_qualified'] is False for r in self.cov['rows']))
        r = next(r for r in self.cov['rows'] if 'INELIGIBLE' in r['numerical']['statuses'])
        self.assertIn('INELIGIBLE', r['numerical']['by_epsilon'].values())
        k = deepcopy(self.k); k['files']['signal_records.json.gz'][r['physical_candidate_id']]['signal'].pop('allowance_evidence')
        self.assertNotEqual(I.classify(r['physical_candidate'], k)['signal']['status'], I.EXACT)

    def test_10_V2_IDs_from_B_for_all_formulas_and_join_by_physical_ID(self):
        from check_full_baseline_v2_requests import pf_spec
        for formula in ('2nd', '4th', '4th(m5_best)'):
            packet = D.bind(pf_spec(formula, M=4))
            joined = I.for_packet(packet, self.cov)
            self.assertTrue(all(j['V2_candidate_id'] == v['candidate']['id'] and j['b_commit'] == D.B_LATEST
                                for j, v in zip(joined, packet['v2_configuration']['rounds'])))
            self.assertEqual({j['V2_formula_id'] for j in joined}, {D.FORMULAS[formula]['formula_id']})
            m3 = joined[3]
            self.assertEqual(m3['physical_candidate_id'], self.pick(system='H4', formula=formula, T=.8, q_PF=8)['physical_candidate_id'])
            self.assertEqual(joined[0]['group'], 'OUTSIDE_SAVED_GRID_ALL_MISSING')
            self.assertEqual(joined[4]['physical_candidate_id'], self.pick(system='H4', formula=formula, T=1.6, q_PF=16)['physical_candidate_id'])

    def test_11_no_science_and_values_are_copied(self):
        self.assertFalse([m for m in sys.modules if m.split('.')[0] in SCIENCE_MODULES])
        r = self.pick(system='H4', formula='2nd', T=1.2, q_PF=2)
        saved = self.k['files']['exploration_costs.json.gz'][r['physical_candidate_id']]['mean_metrics']
        self.assertEqual(r['prices'][0]['axes']['X']['rz_count'], saved['cosine']['rz_count'])
        self.assertFalse(self.cov['values_recomputed']); self.assertEqual(self.cov['new_compile'], 0)

    def test_12_sources_and_private_carriers_are_byte_preserved(self):
        self.assertTrue(all(v['status'] in ('MATCH', 'NOT_RECORDED') for v in self.k['sha'].values()))
        self.assertEqual(sum(v['status'] == 'MATCH' for v in self.k['sha'].values()), 8)
        after = {s: hashlib.sha256((I.PRIVATE/(s+'_context.json')).read_bytes()).hexdigest() for s in self.before}
        self.assertEqual(after, self.before)
        if 'H4' in self.before:
            self.assertEqual(self.before['H4'], '8396605fa0c5ee34f29b436ee16e0f731cfaa6cd94f3304033106a85e9f5d52f')
        for commit, path in I.PF_COMMON.values():
            subprocess.check_call(['git', 'cat-file', '-e', commit + ':' + path + 'signal_records.json.gz'], cwd=ROOT)
        self.assertEqual(len(self.cov['rows']), 136)


if __name__ == '__main__':
    unittest.main(verbosity=2)
