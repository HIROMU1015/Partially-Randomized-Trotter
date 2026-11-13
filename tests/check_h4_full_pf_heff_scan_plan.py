"""Metadata-only fixed-menu tests; no mean/spectrum/price computation."""
from fractions import Fraction
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import prepare_h4_full_pf_heff_scan as P


class PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.plan=P.build()

    def test_existing_B_identity_and_finite_science_scope(self):
        p=self.plan
        self.assertEqual(p['metadata_counts']['configurations'],15)
        self.assertEqual(p['metadata_counts']['new_step_cells'],14)
        self.assertEqual(p['proposed_scientific_scope']['original_H_eigh'],0)
        self.assertEqual(p['proposed_scientific_scope']['direct_PF_complex_means'],28)
        c=next(x for x in p['configurations'] if x['formula']=='2nd' and P.exact(x['requested_delta_pf'])==Fraction(.1))
        self.assertEqual(c['configuration_id'],'f2d58f320ed0ebbeff8e6d4c0fbd5de14d6338b493c92015c966935201fd93ef')
        self.assertEqual(p['actual_scientific_calls'],0);self.assertIsNone(p['selected'])

    def test_actual_widths_early_exceptions_and_negative_time_pieces(self):
        p=self.plan;c=next(x for x in p['configurations'] if x['formula']=='4th' and P.exact(x['requested_delta_pf'])==Fraction(.8))
        self.assertEqual([float(P.exact(r['actual_delta_pf'])) for r in c['rounds'][:4]],[.1,.2,.4,.8])
        self.assertEqual(c['rounds'][12]['q_RPE'],4096);self.assertEqual(c['rounds'][12]['q_PF'],512)
        for cell in p['step_cells']:
            self.assertEqual(cell['signed_piece_count'],{'2nd':1,'4th':3,'4th(m5_best)':11}[cell['formula']])
            if cell['formula']!='2nd':self.assertTrue(any(v<0 for v in cell['signed_weights']))
            self.assertIsNotNone(cell['last_check']);self.assertFalse(cell['qualified']);self.assertIsNone(cell['delta_E_ha'])
        self.assertEqual(sum(x['saved_source'] is not None for x in p['step_cells']),1)

    def test_standalone_builder_imports_no_scientific_libraries(self):
        code="import sys;sys.path.insert(0,'scripts');import prepare_h4_full_pf_heff_scan as p;p.build();assert not any(x in sys.modules for x in ('numpy','scipy','qiskit','pyscf'))"
        subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,check=True)


if __name__=='__main__':unittest.main()
