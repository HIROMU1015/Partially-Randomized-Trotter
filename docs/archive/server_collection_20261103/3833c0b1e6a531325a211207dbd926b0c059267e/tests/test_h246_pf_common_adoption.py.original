"""Author's metadata guards; no scientific kernels, rankings or private inputs."""
import copy
import csv
import io
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import adopt_h246_pf_common_results as adoption
from h46_higher_pf_dataset import DatasetError


class AdoptionGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog, cls.saved, cls.raw = adoption.sources()

    def reject(self, mutation):
        data = copy.deepcopy(self.saved)
        mutation(data)
        with self.assertRaises(DatasetError):
            adoption.assemble(data)

    def test_all_saved_fields_transcribed_without_reanalysis(self):
        data = adoption.assemble(self.saved)
        self.assertEqual(len(data['candidates']), 588)
        for row in data['candidates']:
            system = row['candidate_status_saved']['system']
            cid = row['candidate_status_saved']['id']
            self.assertEqual(row['signal_record_saved'], self.saved[system + '_signal'][cid])
            for endpoint in row['endpoints_saved']:
                self.assertIn(endpoint, self.saved[system + '_endpoints'])

    def test_missing_candidate_rejected_instead_of_subset_best(self):
        self.reject(lambda d: d['H6_status'].pop())

    def test_missing_precision_rejected(self):
        self.reject(lambda d: d['H4_endpoints'].pop())

    def test_legacy_DF12_join_rejected(self):
        self.reject(lambda d: d['H4_identity']['binding'].update(rank=12))

    def test_confirmation_pooling_rejected(self):
        self.reject(lambda d: d['comparisons'][0].update(independent_confirmation=True))

    def test_old_PR_prefix_cannot_be_substituted(self):
        self.reject(lambda d: d['comparisons'][0]['PR_saved'].update(prefix='9'))

    def test_original_exit_status_must_stay_unknown(self):
        self.reject(lambda d: d['summary'].update(H6_original_parent_exit_reap=0))
        self.reject(lambda d: d['H6_interrupted_controls']['STOP'].update(original_native_exit_codes=[0]))

    def test_historical_paper_CSV_prefix_bytes_preserved(self):
        cem, ledger = adoption.evidence_tables(self.saved, self.catalog, self.raw)
        for actual, key, count in [(cem, 'prior_CEM', 21), (ledger, 'prior_ledger', 391)]:
            self.assertTrue(actual.startswith(self.raw[key]))
            self.assertEqual(len(list(csv.DictReader(io.StringIO(actual.decode())))), count)

    def test_optional_H6_lane_does_not_break_complete_CSV(self):
        files, _ = adoption.expected_files()
        rows = list(csv.DictReader(io.StringIO(files['candidate_index.csv'].decode())))
        self.assertEqual(len(rows), 588)
        self.assertEqual(rows[0]['signal_acquisition_lane'], '')
        h6 = [r for r in rows if r['system'] == 'H6']
        self.assertTrue(all(r['signal_acquisition_lane'] for r in h6))


if __name__ == '__main__':
    unittest.main()
