"""Metadata/artificial checks of the saved Full PF / B0I p7 shortlist staging (no science).

Uses only pinned saved blobs and the existing C baseline path; nothing is computed.
"""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from trottertracks import rpe_optimization_pipeline as I  # noqa: E402
from trottertracks import saved_pf_mean_staging as S  # noqa: E402

SOURCE = dict(fixture='ARTIFICIAL_STAGING_SOURCE', kind=S.KIND)
STAGED = {c['label']: S.stage(ROOT, c, I=I, staging_source=SOURCE) for c in S.CANDIDATES}
PAIRS = S.git_json(ROOT, S.PHYSICS_REF)


def blob_json(locator):
    return json.loads(subprocess.check_output(['git', 'show', locator['commit'] + ':' + locator['path']], cwd=ROOT))


class Tests(unittest.TestCase):
    def test_1_saved_means_map_one_to_one_to_their_source_files(self):
        for st in STAGED.values():
            seen = set()
            for mean in st['packet']['means']:
                if mean['raw'] is None:
                    continue
                origin = mean['provider_binding']['input_source']['saved_mean']
                key = (origin['commit'], origin['path'])
                self.assertNotIn(key, seen); seen.add(key)
                saved = blob_json(origin)
                self.assertEqual(saved['direct_PF_mean'], mean['raw'])
                self.assertEqual(saved['physical_candidate'], mean['physical_candidate'])

    def test_2_configuration_and_physical_ids_match_price_plan_and_coverage(self):
        plan = S.git_json(ROOT, S.CELL_PLAN)
        cov = S.git_json(ROOT, S.COVERAGE)['rounds']
        for cand in S.CANDIDATES:
            st = STAGED[cand['label']]
            self.assertTrue(st['configuration_id'].startswith(cand['id_prefix']))
            cells = [x for x in plan if x['configuration_id'] == st['configuration_id']]
            self.assertEqual(len(cells), 6)          # m4/m5/m6 x X/Y
            for x in cells:
                self.assertEqual(x['physical_candidate']['id'], st['rounds'][x['m']]['physical_id'])
            rows = {r['m']: r for r in cov if r['candidate'] == cand['coverage']}
            for r in st['rounds']:
                self.assertEqual(rows[r['m']]['physical_id'], r['physical_id'])

    def test_3_raw_corrected_normalization_reference(self):
        for st in STAGED.values():
            for mean in st['packet']['means']:
                if mean['raw'] is None:
                    continue
                self.assertEqual(mean['raw'], mean['corrected'])
                n = mean['normalization']
                self.assertEqual((n['log_B'], n['B'], n['attenuation']), (0.0, 1.0, 1.0))
                pair = PAIRS['rounds'][mean['m']]['diagnostics']['reference_pair']
                self.assertEqual(mean['reference'], pair['reference'])
                self.assertEqual(n['numerical']['reference_pair_difference'], pair['signal_difference'])
                self.assertIsNone(mean['bias']['total_signed_bias'])        # not derived in staging
                self.assertEqual(mean['law_binding']['definition']['model'], 'deterministic')

    def test_4_full_pf_and_b0i_scalars_are_not_confused(self):
        full = STAGED['FULL_PF_S2_16h_over_7']; b0i = STAGED['B0I_P7_S2_16h_over_7']
        f_mean = next(m for m in full['packet']['means'] if m['raw'] is not None)
        b_mean = next(m for m in b0i['packet']['means'] if m['raw'] is not None)
        fs = f_mean['provider_binding']['input_source']['scalar']; bs = b_mean['provider_binding']['input_source']['scalar']
        self.assertEqual(fs['quantity'], 'HAMILTONIAN_CONSTANT'); self.assertNotIn('tail_identity', fs)
        self.assertEqual(bs['quantity'], 'HAMILTONIAN_CONSTANT_PLUS_B0I_TAIL_IDENTITY_P7')
        self.assertEqual(bs['tail_identity']['selector'], ['prefixes', '7', 'scalar'])
        retained = lambda m: m['provider_binding']['physical_context']['split']['retained_hamiltonian']
        self.assertEqual((retained(f_mean)['construction'], retained(b_mean)['construction']), ('FULL_H', 'B0I'))
        # a Full PF saved mean cannot be bound into the B0I request of the same round
        cand = next(c for c in S.CANDIDATES if c['label'] == 'B0I_P7_S2_16h_over_7')
        config = S.plan_request(ROOT, cand, I=I, staging_source=SOURCE)
        req = config['pending']['requests'][f_mean['m']]
        hit = dict(source=f_mean['provider_binding']['input_source']['saved_mean']['commit'] + ':'
                   + f_mean['provider_binding']['input_source']['saved_mean']['path'], note='A_FOLLOWUP_DIRECT_MEAN')
        pair = PAIRS['rounds'][f_mean['m']]['diagnostics']['reference_pair']
        with self.assertRaisesRegex(ValueError, 'physical candidate differs'):
            S.stage_record(ROOT, cand, None, req, hit, pair, SOURCE)

    def test_5_formula_width_round_cannot_be_swapped(self):
        a = STAGED['FULL_PF_Y4_16h_over_3']; cand = next(c for c in S.CANDIDATES if c['label'] == 'FULL_PF_M5_16h')
        plan = S.plan_request(ROOT, cand, I=I, staging_source=SOURCE)
        for m_src, m_dst in ((4, 4), (12, 12), (4, 12)):
            src = a['packet']['means'][m_src]['provider_binding']['input_source']['saved_mean']
            hit = dict(source=src['commit'] + ':' + src['path'], note='A_FOLLOWUP_DIRECT_MEAN')
            pair = PAIRS['rounds'][m_dst]['diagnostics']['reference_pair']
            with self.assertRaisesRegex(ValueError, 'physical candidate differs'):
                S.stage_record(ROOT, cand, None, plan['pending']['requests'][m_dst], hit, pair, SOURCE)

    def test_6_s2l_is_not_promoted(self):
        for st in STAGED.values():
            for r, mean in zip(st['rounds'], st['packet']['means']):
                if r['coverage_status'].startswith('S2L_'):
                    self.assertEqual((r['state'], mean['raw'], mean['status']), ('S2L_QUALIFICATION_PENDING', None, S.S2L))
                    self.assertTrue(r['legacy_sources'])
        self.assertFalse(set(S.LEGACY) & {n for v in S.CURRENT.values() for n in v})

    def test_7_s3_is_never_filled(self):
        n = 0
        for st in STAGED.values():
            for r, mean in zip(st['rounds'], st['packet']['means']):
                if r['state'].startswith('S3'):
                    n += 1
                    self.assertEqual(mean['status'], S.S3)
                    for key in ('raw', 'corrected', 'normalization', 'bias', 'uncertainty'):
                        self.assertIsNone(mean[key])
        self.assertEqual(n, 44)

    def test_8_incomplete_packet_is_not_a_risk_input(self):
        for st in STAGED.values():
            self.assertFalse(st['complete']); self.assertFalse(st['risk_input_allowed'])
            with self.assertRaisesRegex(ValueError, 'Incomplete staged signal packet'):
                S.require_complete(st['packet'])
        # the gate itself accepts a complete artificial packet (not real H4 data)
        st = STAGED['FULL_PF_M5_16h']; ready = next(m for m in st['packet']['means'] if m['raw'] is not None)
        fake = deepcopy(st['packet'])
        fake['means'] = [dict(deepcopy(ready), m=i, raw=dict(real=.25, imag=-.5)) for i in range(13)]
        self.assertIs(S.require_complete(fake), fake)

    def test_10_sources_unchanged_and_no_new_numbers(self):
        dirty = subprocess.run(['git', 'status', '--porcelain', '--', 'docs/research/h4_full_pf_followup_results_20261114',
                                'docs/research/truncated_pf_t2_results_20261114', 'src/trottertracks/baseline_signal_pipeline.py',
                                'src/trottertracks/rpe_optimization_pipeline.py'], cwd=ROOT, capture_output=True, text=True).stdout
        self.assertEqual(dirty, '')
        counts = sum(len([r for r in st['rounds'] if r['state'].startswith('READY')]) for st in STAGED.values())
        self.assertEqual(counts, 30)
        for st in STAGED.values():
            for mean in st['packet']['means']:
                if mean['raw'] is not None:
                    self.assertEqual(mean['normalization']['numerical']['staging'], 'SAVED_VALUE_BOUND_NO_RECOMPUTATION')
                    self.assertEqual(mean['source'], SOURCE)


if __name__ == '__main__':
    unittest.main(verbosity=2)
