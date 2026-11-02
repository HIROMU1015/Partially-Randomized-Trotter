"""Invented arithmetic/identity controls; no registered science fixtures."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from fractions import Fraction as F

ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('saved_mechanisms',ROOT/'scripts/tracks/algorithm_codesign/analyze_d3_saved_mechanisms.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class SavedMechanismsTests(unittest.TestCase):
    def test_acceptance_and_two_steps(self):
        self.assertEqual(m.whole_incidence(F(3,2),2,F(2,3)),F(4,3))

    def test_zero_trials(self):
        self.assertEqual(m.whole_incidence(4,1,0),0)

    def test_signed_key_prices_and_shots(self):
        d=m.contributions({'a':F(2,3),'b':F(1,2)}, {'a':7,'b':10}, {'a':10,'b':8}, 9)
        self.assertEqual(d,{'a':F(36),'b':F(-18)})
        self.assertEqual(sum(d.values()),F(18))

    def test_symmetric_gap_exact(self):
        d=m.symmetric_product_gap(7,F(8,3),5,F(13,3))
        self.assertEqual(d['total'],2*5*F(13,3)-2*7*F(8,3))
        self.assertEqual(d['shot_term']+d['unit_term'],d['total'])

    def test_affine_tie(self):
        d=m.affine_relation((17,4),(20,2))
        self.assertEqual(d['tie_h'],F(3,2))
        self.assertEqual(d['B_minus_A_intercept']+d['B_minus_A_slope']*d['tie_h'],0)

    def test_single_key_symbolic_tie(self):
        d=m.single_key_tie(F(1,3),F(5,2),11)
        self.assertEqual(F(1,3)+F(5,2)*d['displacement'],0)
        self.assertIsNone(m.single_key_tie(1,0,11)['tie_price'])

    def test_identity_reader_tamper_and_undeclared(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'fake.json';data=b'{"invented": 2}\n';p.write_bytes(data)
            receipt={'result_commit':m.RESULT_COMMIT,'all_identities_PASS':True,'files':[
                {'path':'fake.json','bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}]}
            reader=m.Inputs(t,receipt);self.assertEqual(reader.read('fake.json'),{'invented':2})
            with self.assertRaises(KeyError):reader.read('undeclared.json')
            p.write_text('{"invented": 3}\n')
            with self.assertRaises(ValueError):reader.read('fake.json')

    def test_wrong_commit_refused(self):
        with self.assertRaises(ValueError):m.Inputs('/tmp',{'result_commit':'other','all_identities_PASS':True})

    def test_exact_json_fraction_roundtrip(self):
        d=json.loads(m.dumps({'v':F(1,7)}));self.assertEqual(F(d['v']),F(1,7))


if __name__=='__main__':unittest.main()
