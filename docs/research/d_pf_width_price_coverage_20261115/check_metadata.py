"""Small consistency and rejection checks; stdlib, no science/provider imports."""
from copy import deepcopy
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('width_metadata',HERE/'build_metadata.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.d=json.loads((HERE/'PRICE_COVERAGE.json').read_text())

    def test_scope_and_reference_only_current_table(self):
        d=self.d
        self.assertEqual((len(d['candidates']),len(d['rows'])),(12,312))
        self.assertEqual((d['current_156_reference']['direct_axes'],d['current_156_reference']['conditional_axes']),(96,60))
        self.assertEqual(len({(r['configuration_id'],r['physical_id'],r['m'],r['axis']) for r in d['rows']}),312)
        self.assertEqual(d['counts'],{'DIRECT_SAME_PHYSICAL_EVIDENCE':100,'MISSING_VALIDATION_OR_ACQUISITION':212})
        self.assertEqual(len({(r['physical_id'],r['axis']) for r in d['rows'] if r['price_status']=='DIRECT_SAME_PHYSICAL_EVIDENCE'}),68)
        self.assertTrue(all(n==0 for n in d['scientific_calls'].values()))

    def test_physical_equality_rejects_q_only_and_rounded_width(self):
        c=self.d['rows'][0]['physical_candidate']
        for key,value in [('T',.2),('delta',.2),('method','B0'),('prefix',10),('formula','4th')]:
            bad=dict(c);bad[key]=value
            self.assertNotEqual(M.physical(c),M.physical(bad))
        bad=dict(c);bad['delta']=float.fromhex(c['delta'].hex())+2**-56
        self.assertNotEqual(M.physical(c),M.physical(bad))
        self.assertEqual(M.physical(c),M.physical(dict(c,id='OLD_ALIAS')))

    def test_all_schedules_and_axis_specific_sources(self):
        for row in self.d['rows']:
            saved=M.resolve(row['schedule_source']);c=row['physical_candidate']
            self.assertEqual((saved['m'],saved['q_PF'],saved['physical_candidate_id']),(row['m'],c['q'],c['id']))
            self.assertEqual(float(M.H*Fraction(saved['actual_delta_over_h'])).hex(),c['delta'].hex())
            self.assertEqual(float(M.H*2**row['m']).hex(),c['T'].hex())
            if row['price_status']=='DIRECT_SAME_PHYSICAL_EVIDENCE':
                self.assertTrue(row['price_evidence']);self.assertIsNotNone(row['saved_compiled_RZ'])
                known=row['price_evidence'][0]
                if 'receipt' in known:
                    original=M.resolve(known['proof_of_current_admission'])
                    self.assertEqual((original['physical_id'],original['m'],original['axis']),(c['id'],row['m'],row['axis']))
            else:self.assertIsNone(row['saved_compiled_RZ'])

    def test_no_cross_width_model_adoption_and_finite_stop(self):
        self.assertTrue(all(not r['conditional_model_applicable'] and not r['price_computed'] for r in self.d['rows']))
        self.assertTrue(all(not c['all_round_prices_available'] for c in self.d['candidates']))
        p=json.loads((HERE/'FIRST_FINITE_PLAN.json').read_text())
        self.assertEqual((len(p['targets']),p['reuse_axes'],p['new_compile_maximum']),(4,6,4))
        self.assertEqual({(r['m'],r['axis']) for r in p['targets']},{(1,'X'),(1,'Y'),(2,'X'),(2,'Y')})
        self.assertFalse(p['scientific_retry'])
        self.assertTrue(self.d['mandatory_stop'])


if __name__=='__main__': unittest.main(verbosity=2)
