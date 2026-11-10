"""Synthetic regression for saved-value corrections; no molecular input."""
import sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import h6_df19_price_v2 as m


def pair(b,y):return dict(q=1,basis_groups=b,actual=[100+y,100+2*y],baseline=[100,100])


class Tests(unittest.TestCase):
    def test_reused_multiresponse_fit_and_rank(self):
        helper=m.reuse(ROOT);ps=[pair(b,3+2*b) for b in [1,2,3,4]]
        f=m.calibrate(ps,'Mbasis',helper['fit']);np.testing.assert_allclose(f['coefficients'],[[3,6],[2,4]],atol=1e-12)
        self.assertEqual(f['rank'],2);self.assertFalse(f['q_slope_identified'])
        self.assertEqual(m.calibrate([pair(2,3)]*3,'Mbasis',helper['fit'])['status'],'EVALUATION_UNAVAILABLE')
    def test_offset_is_constant_not_q_slope(self):
        ps=[dict(pair(4,3),q=q) for q in [1,2,4]]
        self.assertEqual(m.design(ps,'Moffset').tolist(),[[1],[1],[1]])
    def test_pf_has_no_trajectory_covariance(self):
        f=m.calibrate([pair(None,7)],'Moffset',m.reuse(ROOT)['fit'])
        self.assertIsNone(f['nominal_OLS_coefficient_covariance']);self.assertEqual(f['degrees_of_freedom'],0)
    def test_axis_pairing_keeps_family_identity(self):
        rows=[]
        for axis in m.AXES:rows.append(dict(family='PR_A',cohort='TOY',candidate_id='TOY',q_PF=1,T=1.,path_index=3,axis=axis,compiled_rz=120,prediction=dict(model=m.V1_MODEL,structural_rz=100,features={'tail_basis_groups':4})))
        ps=m.pairs(rows,m.reuse(ROOT));self.assertEqual(len(ps),1);self.assertEqual(ps[0]['family'],'PR_A')
        rows[1]['prediction']['features']['tail_basis_groups']=5
        with self.assertRaises(ValueError):m.pairs(rows,m.reuse(ROOT))


if __name__=='__main__':unittest.main(verbosity=2)
