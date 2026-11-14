"""Synthetic metadata mutation/exact-boundary tests; no H4 context or circuits."""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import run_pf_m9_prices as M


class M9Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = M.predictions()
        cls.templates = {r: m.plans(ROOT) for r, m in M.ROUTES.items()}
        cls.alignment = M.read(cls.doc['rows'][0]['alignment_source'])
        cls.policy = M.read(M.POLICY)
        cls.models = M.read(M.M6.FROZEN)['models']

    def validate(self, doc):
        return M.validate(doc, self.templates, self.alignment, self.policy, self.models)

    def test_exact_twelve(self):
        self.assertEqual({r: len(v) for r, v in self.validate(self.doc).items()}, {'FULL_PF': 6, 'TRUNCATED_PF': 6})

    def test_foreign_metadata_rejected(self):
        changes = [('axis', 'Y'), ('m', 6), ('q', 225), ('T', 51.3), ('physical_id', 'wrong'),
                   ('configuration_id', 'wrong'), ('canonical_id', 'wrong'), ('actual_delta_binary64_hex', 'wrong'),
                   ('actual_delta', .2), ('fragment_order', 'reverse'), ('observed_RZ', 1), ('decision', 'PASS')]
        for key, value in changes:
            with self.subTest(key=key):
                doc=deepcopy(self.doc);doc['rows'][0][key]=value
                with self.assertRaises(ValueError): self.validate(doc)

    def test_scalar_and_structure_rejected(self):
        for key in ('scalar', 'input_binding', 'split', 'signed_structure', 'coefficients', 'domain'):
            with self.subTest(key=key):
                doc=deepcopy(self.doc);doc['rows'][0][key]={}
                with self.assertRaises(ValueError): self.validate(doc)

    def test_duplicate_missing_and_old_reader(self):
        doc=deepcopy(self.doc);doc['rows'][1]=deepcopy(doc['rows'][0])
        with self.assertRaises(ValueError): self.validate(doc)
        doc=deepcopy(self.doc);doc['rows'].pop()
        with self.assertRaises(ValueError): self.validate(doc)
        self.assertEqual(M.S.STAGES, dict(CALIBRATION=(4,5), HOLDOUT=(6,)))
        with self.assertRaises(ValueError): M.M6.validate_predictions(self.doc, M.S.plan_rows(ROOT))

    def test_exact_judgement_boundaries(self):
        row={'predicted_RZ':dict(numerator=102,denominator=1)}
        self.assertEqual(M.M6.judge_row(row,dict(observed_RZ=100))['decision'],'PASS')
        row['predicted_RZ']=dict(numerator=102001,denominator=1000)
        r=M.M6.judge_row(row,dict(observed_RZ=100))
        self.assertEqual(r['decision'],'FAIL')
        self.assertEqual(r['observed_RZ'],100)  # retain valid direct observation on model FAIL
        self.assertEqual(M.M6.exact(r['relative_error']),Fraction(2001,100000))
        self.assertEqual(M.M6.judge_row(row,dict(observed_RZ=0))['decision'],'UNKNOWN')
        self.assertEqual(M.M6.judge_row(row,problem='Synthetic missing m9 receipt')['decision'],'UNKNOWN')

    def test_route_namespaces_separate(self):
        names=[M.paths(r) for r in M.ROUTES]
        self.assertNotEqual(names[0],names[1])
        self.assertTrue(all('/m9_holdout/run01' in a for a,b in names))
        self.assertEqual(set(self.policy['physical_ids']),{v[0] for t in M.TARGETS.values() for v in t.values()})


if __name__=='__main__':unittest.main()
