"""Read fixed LD2 evidence through the publication consumer; no acquisition."""
from pathlib import Path
import unittest
from trottertracks.saved_rpe_cost_catalog import Reader
from trottertracks.h4_search_run_evidence import discover_runs
from trottertracks.h4_search_map import build_index
from trottertracks.h4_search_map_html import render

ROOT=Path(__file__).resolve().parents[1]
SOURCE='92b3e617a0a0ede94d90ac2582b058f5bf9f67a3'

class SavedLD2(unittest.TestCase):
    def test_normal_parser_saved_measurements_and_receipt_links(self):
        reader=Reader(ROOT)
        runs,issues=discover_runs(reader,{'validation-producer':SOURCE},legacy_heads=set())
        runs=[r for r in runs if r['directory'].endswith('h4_price_validation_integration_20261117/saved_entry_example')]
        self.assertEqual(len(runs),1)
        empty=dict(comparison={},candidates=[])
        index=build_index(empty,runs)
        self.assertEqual(len(index['price_validation_models']),1)
        model=next(iter(index['price_validation_models'].values()))
        m5=next(p for p in model['validation_points']if p['m']==5)
        self.assertEqual((m5['q'],m5['R'],m5['independent_paths']),(32,33,2))
        self.assertEqual(len(m5['observations']),4)
        for axis in ['X','Y']:
            self.assertEqual(m5['observed_mean_RZ'][axis],dict(numerator=68264,denominator=1))
            self.assertEqual(m5['signed_relative_difference'][axis],dict(numerator=2751307649,denominator=6167206567816))
            self.assertEqual({o['path_id']for o in m5['observations']if o['axis']==axis},{'0','1'})
        page=render(index)
        for observation in m5['observations']:
            receipt=reader.read(observation['source'])
            self.assertEqual(receipt['request']['q'],32)
            self.assertEqual(receipt['request']['R'],33)
            self.assertEqual(receipt['axis'],observation['axis'])
            self.assertEqual(receipt['record']['metrics']['rz_count'],observation['compiled_RZ'])
            self.assertIn('/blob/'+observation['source']['commit']+'/'+observation['source']['path'],page)
        self.assertIn('function percent(x)',page)
        self.assertEqual(build_index(empty,runs),index)
        self.assertEqual(len(index['points']),2)
        self.assertTrue(all(p['price_validations']for p in index['points']))
        self.assertEqual(index['new_scientific_calls'],0)

if __name__=='__main__':unittest.main(verbosity=2)
