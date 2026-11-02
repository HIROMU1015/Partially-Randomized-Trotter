"""Small saved-arithmetic checks, no science imports or molecular inputs."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from summarize_h6_fixed_delta_v1 import paired_metrics, account, signed_parts


class SavedArithmetic(unittest.TestCase):
    def test_paired_mean_and_SE(self):
        rows=[dict(replica=i,axis=a,metrics={'RZ':x}) for i,p in enumerate(((2,7),(4,3)))
            for a,x in zip(('cosine','sine'),p)]
        r=paired_metrics(rows,dict(real=3,imag=5),'RZ')
        self.assertEqual(r['G'],34);self.assertAlmostEqual(r['SE'],7)
        self.assertEqual(r['G_exact'],dict(numerator='34',denominator='1'))
    def test_missing_axis_rejected(self):
        with self.assertRaises(ValueError):paired_metrics([dict(replica=0,axis='cosine',metrics={'RZ':1})],None,'RZ')
    def test_duplicate_axis_rejected(self):
        r=dict(replica=0,axis='cosine',metrics={'RZ':1})
        with self.assertRaises(ValueError):paired_metrics([r,r],None,'RZ')
    def test_negative_metric_rejected(self):
        with self.assertRaises(ValueError):paired_metrics([dict(replica=0,axis='cosine',metrics={'RZ':-1})],None,'RZ')
    def test_ineligible_is_null(self):
        r=paired_metrics([dict(replica=0,axis=a,metrics={'RZ':3}) for a in ('cosine','sine')],None,'RZ')
        self.assertIsNone(r['G']);self.assertIsNone(r['SE'])
    def test_normalization_and_allowance_not_removed(self):
        s=dict(cell={'id':'toy'},error_decomposition={'total_signed':dict(real=.001,imag=.002)},
            allowance=dict(real=1e-6,imag=1e-6,log_B_margin=1e-6),log_B=.1)
        a=account(s,.01,1.);b=account(s,.01,10.)
        self.assertGreaterEqual(b['N_total'],a['N_total'])
        self.assertEqual(a['evidence_kind'],'EMPIRICAL')
        s['error_decomposition']['total_signed']['real']=1.
        self.assertIsNone(account(s,.01,1.)['N_total'])
    def test_signed_cancellation_not_absolute_addition(self):
        s=dict(cell={'method':'B0'},signals=dict(corrected=dict(real=.9,imag=0),
            exact_truncated=dict(real=.8,imag=0)),reference=dict(real=1,imag=0))
        r=signed_parts(s)
        self.assertAlmostEqual(r['total'],.1)
        self.assertAlmostEqual(sum(r['parts_abs'].values()),.3)


if __name__=='__main__':unittest.main(verbosity=2)
