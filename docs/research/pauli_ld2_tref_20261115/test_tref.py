"""Artificial checks for exact T_ref arithmetic, saved-value reuse and withholding rules."""
from copy import deepcopy
from fractions import Fraction
import importlib.util
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_ld2_tref',HERE/'tref.py')
M=importlib.util.module_from_spec(spec);sys.modules[spec.name]=M;spec.loader.exec_module(M)


def coeffs(b0,b1,b2):
    return {'one':M.enc(b0),'pf_step_count':M.enc(b1),'event_count':M.enc(b2)}


def inputs(cx=(Fraction(1,3),2,Fraction(1,7)),cy=None,saved=None,baselines=None):
    cy=cy or cx;rows=[]
    for m in range(13):
        q=2**m;R=q+1
        for a in M.AXES:
            f={k:M.enc(v)for k,v in M.features(q,R).items()}
            s=(saved or {}).get((m,a))
            rows.append(dict(m=m,axis=a,q=q,R=R,N=M.N[m],features=f,
                value_source='SAVED_FROZEN_PREDICTION'if s is not None else'NEW_DIAGNOSTIC_APPLICATION',
                saved_predicted_RZ=M.enc(s)if s is not None else None))
    return dict(axes=rows,issues=[],model=dict(coefficients={'X':coeffs(*cx),'Y':coeffs(*cy)}),
                baselines=baselines or {'A':10**12,'B':10})


def expected(cx,cy):
    t=Fraction(0)
    for m in range(13):
        q=2**m;R=q+1
        for c in (cx,cy):t+=M.N[m]*(Fraction(c[0])+Fraction(c[1])*q+Fraction(c[2])*q*R)
    return t


class TRef(unittest.TestCase):
    def test_exact_sum_xy_separate_and_boundary_sign(self):
        cx=(Fraction(1,3),2,Fraction(1,7));cy=(5,Fraction(3,11),1)
        rows,res,issues=M.compute(inputs(cx,cy))
        self.assertEqual(issues,[]);self.assertEqual(len(rows),26)
        T=expected(cx,cy);self.assertEqual(M.frac(res['T_ref']),T)
        self.assertNotEqual(T.denominator,1)  # no rounding to integer RZ
        b={x['baseline']:M.frac(x['D_PF'])for x in res['boundaries']}
        self.assertEqual(b['A'],10**12-T);self.assertGreater(b['A'],0);self.assertLess(b['B'],0)

    def test_saved_predictions_used_as_is(self):
        saved={(5,'X'):Fraction(123456,7),(9,'Y'):Fraction(1)}
        rows,res,_=M.compute(inputs(saved=saved))
        r=next(r for r in rows if (r['m'],r['axis'])==(5,'X'));self.assertEqual(M.frac(r['C_ref']),Fraction(123456,7))
        r=next(r for r in rows if (r['m'],r['axis'])==(9,'Y'));self.assertEqual(M.frac(r['N_times_C_ref']),M.N[9])

    def test_negative_or_unresolved_rows_withhold_total(self):
        rows,res,issues=M.compute(inputs(cx=(-10**9,0,0),cy=(1,1,1)))
        self.assertIsNone(res);self.assertTrue(all(i['reason']=='NEGATIVE_MODEL_REFERENCE'for i in issues))
        self.assertEqual(len(issues),13)  # only X axes negative, listed per row
        x=inputs();x['axes'][3]['value_source']='UNRESOLVED';x['issues']=[dict(m=1,axis='Y',reason='input mismatch')]
        rows,res,issues=M.compute(x);self.assertIsNone(res);self.assertEqual(len(rows),25)

    def test_display_is_not_used_for_arithmetic(self):
        self.assertEqual(M.display(Fraction(1,3)),'0.333333');self.assertEqual(M.display(Fraction(-2,3)),'-0.666667')
        rows,res,_=M.compute(inputs());self.assertEqual(M.frac(res['T_ref']),expected((Fraction(1,3),2,Fraction(1,7)),(Fraction(1,3),2,Fraction(1,7))))

    def test_saved_fixed_predictions_consistent_with_frozen_formula(self):
        """Identity check of the 6 published values against the frozen coefficients (fixed blobs)."""
        frozen=M.doc(M.FROZEN);m9=M.doc(M.M9PRED)
        for key,s in M.saved_predictions(frozen,m9).items():
            p=s['p'];f={k:M.frac(v)for k,v in p['features'].items()}
            self.assertEqual(M.model_value(frozen['models'][p['axis']]['coefficients'],f),M.frac(p['predicted_RZ']),key)


if __name__=='__main__':
    unittest.main(verbosity=2)
