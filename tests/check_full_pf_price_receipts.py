"""Receipt trace of the reused T=0.8 Full PF prices: pinned metadata plus artificial edits only.

No Hamiltonian/state load, evolution, compile/transpile, RZ recount, fit or G.
Edits below are applied to in-memory copies of saved metadata to show that a
mismatch is detected; nothing on disk or in Git is changed.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts'), str(ROOT/'tests')]
from trottertracks import full_pf_saved_inventory as I
from trottertracks import full_pf_price_receipts as R

D = I.D
SCIENCE_MODULES = ('h46_long_time_signal', 'h46_long_time_native', 'h46_m5_native', 'pf_common_run_native', 'qiskit', 'pyscf',
                   'h46_new2_native', 'h46_m5_tol_native')
OLD = 'docs/research/d_full_pf_saved_evidence_coverage_20261113/COVERAGE.json'
REUSED = 'reused_compiled_prices.json.gz'


class ReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = {s: hashlib.sha256((I.PRIVATE/(s+'_context.json')).read_bytes()).hexdigest()
                      for s in ('H4', 'H6') if (I.PRIVATE/(s+'_context.json')).exists()}
        cls.k = I.load_carriers(); cls.ctx = R.load(cls.k); cls.sources = R.source_checks(cls.ctx, cls.k)
        cls.old = I.coverage(cls.k); cls.new = R.update(cls.old, cls.k, cls.ctx)
        cls.old_rows = {r['physical_candidate_id']: r for r in cls.old['rows']}
        cls.new_rows = {r['physical_candidate_id']: r for r in cls.new['rows']}
        cls.target = sorted(i for i, r in cls.old_rows.items() if I.CONDITIONAL in (r['X'], r['Y']))

    def pick(self, **kw):
        return next(r for r in self.new['rows'] if all(r[k] == v for k, v in kw.items()))

    def edited(self, system):
        """Shallow copies with private deep copies of the parts a test edits."""
        k = dict(self.k); k['files'] = dict(self.k['files']); k['files'][REUSED] = deepcopy(self.k['files'][REUSED])
        ctx = dict(self.ctx); ctx['systems'] = dict(self.ctx['systems'])
        s = ctx['systems'][system] = dict(self.ctx['systems'][system])
        s['prices'] = deepcopy(s['prices']); s['by_sha'] = dict(s['by_sha']); s['inventory'] = deepcopy(s['inventory'])
        return k, ctx

    @staticmethod
    def rekey(k, ctx, c, record):
        """Store an edited native record under its own canonical SHA everywhere it is referenced."""
        s = ctx['systems'][c['system']]; sha = R.canonical_sha(record)
        s['by_sha'][sha] = record; s['inventory'].append(dict(candidate=record['candidate'], result_sha256=sha))
        k['files'][REUSED][c['id']]['result_sha256'] = sha; s['prices'][c['id']]['result_sha256'] = sha

    def test_01_seventeen_conditional_prices_trace_to_their_saved_results(self):
        self.assertEqual(len(self.target), 17)
        kinds = {}
        for cid in self.target:
            row = self.new_rows[cid]
            self.assertEqual((row['X'], row['Y']), (I.EXACT, I.EXACT))
            trace = next(t for t in self.new['receipt_traces'] if t['candidate_id'] == cid)
            kinds[cid] = sorted({e['kind'] for e in trace['evidence']})
            for e in trace['evidence']:
                if e['kind'] == 'PF_COMMON_NATIVE_RESULT':
                    record = self.ctx['systems'][row['system']]['by_sha'][e['source']['result_sha256']]
                    self.assertEqual(R.canonical_sha(record), self.k['files'][REUSED][cid]['result_sha256'])
                    self.assertEqual(record['candidate'], {f: row['physical_candidate'][f] for f in R.CANDIDATE_KEYS})
                if e['kind'] == 'M5_PILOT_RESULT':
                    self.assertEqual((e['source']['pilot_group']['q'], e['source']['pilot_group']['formula']), (row['q_PF'], row['formula']))
        flat = [x for v in kinds.values() for x in v]
        self.assertEqual(flat.count('PF_COMMON_NATIVE_RESULT'), 12)
        self.assertEqual(flat.count('M5_PILOT_RESULT'), 5)
        self.assertEqual(self.sources['status'], I.EXACT)
        self.assertTrue(self.sources['items']['deterministic_lowering_shared_with_long_time_compose'])

    def test_02_X_and_Y_are_judged_separately(self):
        c = self.pick(system='H4', formula='2nd', T=.8, q_PF=2)['physical_candidate']        # pilot proof
        k, ctx = self.edited('H4')
        for v in (k['files'][REUSED][c['id']], ctx['systems']['H4']['prices'][c['id']]):
            next(a for a in v['proof'][0]['axes'] if a['axis'] == 'sine')['fingerprint'] = 'OTHER_FINGERPRINT'
        out = R.trace(c, k, ctx, self.sources)['axes']
        self.assertEqual((out['X']['status'], out['Y']['status']), (I.EXACT, I.MISMATCH))
        c = self.pick(system='H4', formula='4th', T=.8, q_PF=4)['physical_candidate']        # native result
        k, ctx = self.edited('H4')
        record = deepcopy(ctx['systems']['H4']['by_sha'][k['files'][REUSED][c['id']]['result_sha256']])
        record['prices']['sine']['compiler'] = 'OTHER_COMPILER'; self.rekey(k, ctx, c, record)
        out = R.trace(c, k, ctx, self.sources)['axes']
        self.assertEqual((out['X']['status'], out['Y']['status']), (I.EXACT, I.MISMATCH))
        self.assertIn('COMPILER_DIFFERS', out['Y']['reasons'])

    def test_03_conditional_is_not_promoted_without_a_candidate_proof(self):
        c = self.pick(system='H6', formula='2nd', T=.8, q_PF=1)['physical_candidate']
        k, ctx = self.edited('H6')
        for v in (k['files'][REUSED][c['id']], ctx['systems']['H6']['prices'][c['id']]):
            v['result_sha256'] = '0' * 64                                   # a SHA alone, not in the results
        out = R.trace(c, k, ctx, self.sources)['axes']
        self.assertEqual({v['status'] for v in out.values()}, {R.UNAVAILABLE})
        k, ctx = self.edited('H6')
        for v in (k['files'][REUSED][c['id']], ctx['systems']['H6']['prices'][c['id']]):
            v.pop('result_sha256')                                          # RZ count only
        out = R.trace(c, k, ctx, self.sources)['axes']
        self.assertEqual({v['status'] for v in out.values()}, {I.CONDITIONAL})
        # Another candidate's result with the same compiler is rejected.
        other = self.pick(system='H6', formula='4th', T=.8, q_PF=1)['physical_candidate']
        k, ctx = self.edited('H6')
        sha = k['files'][REUSED][other['id']]['result_sha256']
        for v in (k['files'][REUSED][c['id']], ctx['systems']['H6']['prices'][c['id']]):
            v['result_sha256'] = sha
        out = R.trace(c, k, ctx, self.sources)['axes']
        self.assertEqual(out['X']['status'], I.MISMATCH); self.assertIn('CANDIDATE_DIFFERS', out['X']['reasons'])
        # A pilot proof naming another formula's pilot group is rejected.
        c = self.pick(system='H4', formula='2nd', T=.8, q_PF=2)['physical_candidate']
        k, ctx = self.edited('H4')
        for v in (k['files'][REUSED][c['id']], ctx['systems']['H4']['prices'][c['id']]):
            v['proof'][0]['source'] = 'pilot_' + R.pilot_group_function()('H4', 'FULL', '4th')['id'] + '_result'
        out = R.trace(c, k, ctx, self.sources)['axes']
        self.assertEqual(out['X']['status'], I.MISMATCH); self.assertIn('PILOT_GROUP_ID_DIFFERS_FROM_CANDIDATE', out['X']['reasons'])
        # The long-time carrier must equal the pf_common price it cites.
        k, ctx = self.edited('H4'); k['files'][REUSED][c['id']]['cosine'] += 1
        self.assertIn('CARRIER_PRICE_DIFFERS_FROM_PF_COMMON', R.trace(c, k, ctx, self.sources)['axes']['Y']['reasons'])
        # An edited saved result no longer matches its recorded SHA.
        c = self.pick(system='H4', formula='2nd', T=.8, q_PF=1)['physical_candidate']
        k, ctx = self.edited('H4'); s = ctx['systems']['H4']; sha = k['files'][REUSED][c['id']]['result_sha256']
        s['by_sha'][sha] = dict(s['by_sha'][sha], status='PASS', wall_seconds=0)
        self.assertIn('RESULT_SHA_DIFFERS', R.trace(c, k, ctx, self.sources)['axes']['X']['reasons'])

    def test_04_binding_wrapper_compiler_and_SOURCE_are_required(self):
        c = self.pick(system='H6', formula='4th', T=.8, q_PF=8)['physical_candidate']
        k, ctx = self.edited('H6')
        ctx['systems']['H6']['identity'] = deepcopy(ctx['systems']['H6']['identity'])
        ctx['systems']['H6']['identity']['binding']['snapshot'] = 'OTHER_SNAPSHOT'
        bad = R.source_checks(ctx, k)
        self.assertEqual(bad['status'], I.MISMATCH)
        self.assertEqual(bad['items']['pf_common_recorded_SOURCE_sha']['H6']['binding'], ['snapshot'])
        self.assertIn('NATIVE_SOURCE_TEXT_CHECK_FAILED', R.trace(c, k, ctx, bad)['axes']['X']['reasons'])
        items = self.sources['items']
        self.assertTrue(items['pilot_cost_text'] and all(items['pf_common_cost_text'].values()))   # wrapper with measurement, opt1/seed17
        self.assertTrue(items['compose_identical_across_pilot_and_pf_common'] and items['formula_source_identical'])
        for system in ('H4', 'H6'):
            self.assertEqual(self.ctx['systems'][system]['identity']['binding']['T'], .8)
            self.assertEqual(self.ctx['systems'][system]['price_conflicts'], [])

    def test_05_other_EXACT_and_missing_rows_are_unchanged(self):
        changed = {c['physical_candidate_id'] for c in self.new['changes']}
        self.assertEqual(changed, set(self.target))
        for cid, old in self.old_rows.items():
            if cid not in changed:
                self.assertEqual((old['X'], old['Y'], old['group']), (self.new_rows[cid]['X'], self.new_rows[cid]['Y'], self.new_rows[cid]['group']))
        self.assertEqual(self.new['totals']['X'], {I.EXACT: 73, I.MISSING: 63})
        self.assertEqual(self.new['totals']['Y'], {I.EXACT: 73, I.MISSING: 63})
        # The seven candidate-receipt EXACT reuses were traced without contradiction.
        receipts = [t for t in self.new['receipt_traces'] if t['candidate_id'] not in changed]
        self.assertEqual(len(receipts), 7)
        self.assertTrue(all(v['status'] == I.EXACT for t in receipts for v in t['axes'].values()))
        # The published 92ff17de coverage file is not rewritten.
        self.assertEqual((ROOT/OLD).read_bytes(), D._blob('92ff17dededea6e1ee273b2eee6a08e292ee9e5b', OLD))
        old_saved = {r['physical_candidate_id']: r for r in json.loads((ROOT/OLD).read_text())['rows']}
        self.assertEqual({i for i, r in old_saved.items() if I.CONDITIONAL in (r['X'], r['Y'])}, set(self.target))
        self.assertEqual(len(self.new['rows']), 136)

    def test_06_saved_MODEL_is_never_a_direct_price(self):
        kinds = {s['kind'] for r in self.new['rows'] for s in r['prices']}
        self.assertTrue(all(kind.startswith('DIRECT_COMPILED_DETERMINISTIC') for kind in kinds))
        model = [r for r in I.other_saved_series()['rows'] if r['kind'] == 'MODEL_PREDICTION']
        self.assertTrue(model and all(r['status'] == 'MODEL_NOT_DIRECT_PRICE' for r in model))
        self.assertFalse({r['physical_candidate_id'] for r in model} & set(self.target))

    def test_07_V2_join_keeps_physical_and_V2_identity_separate(self):
        from trace_full_pf_price_receipts import join_example
        out = join_example(self.new)
        self.assertEqual(out['status'], 'ARTIFICIAL_TECHNICAL_JOIN_NOT_A_MENU_OR_ADOPTED_CONDITION')
        for item in out['full_pf']:
            m3, m4 = item['saved_join'][3], item['saved_join'][4]
            for j, (T, q) in ((m3, (.8, 8)), (m4, (1.6, 16))):
                row = self.new_rows[j['physical_candidate_id']]
                self.assertEqual((row['system'], row['formula'], row['T'], row['q_PF']), (item['system'], item['formula'], T, q))
                self.assertTrue(j['V2_candidate_id'].startswith('RMSE:') and j['V2_candidate_id'] != j['physical_candidate_id'])
                self.assertEqual(j['mapping_source']['v2']['commit'], D.B_LATEST)
                self.assertEqual(j['mapping_source']['newly_mapped_by_B_formula_hook'], item['formula'] == '4th')
            self.assertEqual(m3['saved']['X'], I.EXACT)
            self.assertEqual(item['saved_join'][0]['group'], 'OUTSIDE_SAVED_GRID_ALL_MISSING')
        light = {v['system']: v for v in out['full_h_rte']}
        self.assertEqual(light['H4']['identity_map'][3]['physical_candidate_id'], 'FULL_H_8d6f23210529d68452dd')
        self.assertEqual(light['H6']['identity_map'][3]['physical_candidate_id'], 'FULL_H_78f4b605c921e6d3f491')
        self.assertTrue(all(v['saved_asset_status'][3] == 'SAVED_LIGHT_STUDY_CANDIDATE_FINITE_COHORT' for v in light.values()))
        self.assertEqual({v['event_mapping_kind'] for v in light.values()}, {D.B.EXPLICIT_LAYOUT})

    def test_08_no_science_compile_or_recount_and_bytes_preserved(self):
        self.assertFalse([m for m in sys.modules if m.split('.')[0] in SCIENCE_MODULES])
        self.assertFalse(self.new['values_recomputed']); self.assertEqual((self.new['new_compile'], self.new['rz_recount']), (0, 0))
        for cid in self.target:
            price = self.new_rows[cid]['prices'][0]['axes']
            self.assertEqual((price['X']['rz_count'], price['Y']['rz_count']),
                             (self.k['files'][REUSED][cid]['cosine'], self.k['files'][REUSED][cid]['sine']))
        after = {s: hashlib.sha256((I.PRIVATE/(s+'_context.json')).read_bytes()).hexdigest() for s in self.before}
        self.assertEqual(after, self.before)
        for commit, path in list(I.PF_COMMON.values()) + [R.PILOT]:
            subprocess.check_call(['git', 'cat-file', '-e', commit + ':' + path], cwd=ROOT)


if __name__ == '__main__':
    unittest.main(verbosity=2)
