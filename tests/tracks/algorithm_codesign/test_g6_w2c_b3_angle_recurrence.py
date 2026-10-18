"""Synthetic rational precision tests only; never G7/session/scientific input."""
from fractions import Fraction as F
import importlib
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
R=importlib.import_module('trottertracks.algorithm_codesign.g6_f3_precision')
P=importlib.import_module('trottertracks.algorithm_codesign.g6_w2c_b3_angle_recurrence')
CONTEXT={'ARTIFICIAL_ONLY':True,'fixture':'rational equivalence, no G7 event'}

class Endpoints(unittest.TestCase):
    def test_exact_grid(self):
        values=[F(0),F(1,7),F(3,4),F(1)-F(1,2**80),F(-1,3),F(2**256+1,2**257+9)]
        for z in values:
            for j in (0,1,2,3,4,8,16,32,64):
                with self.subTest(z=z,j=j):self.assertEqual(R._atan_endpoint(z,j),P.atan_endpoint_recurrence(z,j))
    def test_large_operands(self):
        for bits in (64,256,1024,4096):
            z=F(2**bits+57,2**(bits+1)+19)
            self.assertEqual(R._atan_endpoint(z,32),P.atan_endpoint_recurrence(z,32))
    def test_no_float_path(self):
        for z,j in ((0.25,32),(F(1,4),True),(F(1,4),-1),(1,2)):
            with self.assertRaises(ValueError):P.atan_endpoint_recurrence(z,j)

class Enclosures(unittest.TestCase):
    def compare(self,r,**kw):
        arguments=dict(tau=F(1,1000000),root_bits=32,terms=32,sigma=1,context=CONTEXT);arguments.update(kw)
        def outcome(f):
            try:
                e=f(r,**arguments);return 'PASS',e.document,e.digest,e.read()
            except R.F3Failure as exc:return 'FAIL',exc.to_dict()
        reference=outcome(R.enclose_atan)
        self.assertEqual(reference,outcome(P.enclose_atan_artificial))
        self.assertEqual(reference,outcome(lambda r,**kw:P.enclose_atan_artificial(r,series='reference',**kw)))
        return reference
    def test_artificial_only_gate(self):
        with self.assertRaisesRegex(ValueError,'ARTIFICIAL_INPUT_CONTEXT_REQUIRED'):
            P.enclose_atan_artificial(F(1,7),tau=F(1,1000000),root_bits=32,terms=32)
    def test_signs(self):
        for sign in (-1,1):self.compare(F(7,13),sigma=sign)
    def test_root_outward_and_exact_square(self):
        for r in (F(3,4),F(1),F(7,13)):
            result=self.compare(r);self.assertEqual(result[0],'PASS')
            lo,hi=result[3]['sqrt_interval'];self.assertLessEqual(lo*lo,1+r*r);self.assertGreaterEqual(hi*hi,1+r*r)
    def test_vary_J_K_tau(self):
        for j in (1,2,3,8,16,32,64):
            for k in (1,8,32,64):self.compare(F(5,11),terms=j,root_bits=k,tau=F(1,1000000))
    def test_exact_radius_boundary(self):
        data=self.compare(F(5,11))[3];radius=data['representative_error_upper']
        self.assertEqual(self.compare(F(5,11),tau=radius)[0],'PASS')
        self.assertEqual(self.compare(F(5,11),tau=radius/2)[0],'FAIL')
    def test_precision_failure_exact_details(self):self.assertEqual(self.compare(F(7,13),terms=1,root_bits=1,tau=F(1,10**30))[0],'FAIL')
    def test_fixed_functions_not_replaced(self):
        before=(R._atan_endpoint,R.enclose_atan,R.root_interval)
        self.compare(F(2,9));self.assertEqual(before,(R._atan_endpoint,R.enclose_atan,R.root_interval))
    def test_no_sampler_session_imported(self):
        for n in ('g7_generator','g7_provider','g6_f3_generator','g9_native'):
            self.assertNotIn('trottertracks.algorithm_codesign.'+n,sys.modules)

for name,r in {'zero':F(0),'tiny':F(1,2**200),'boundary_two':F(2),
               'below_two':F(2)-F(1,2**160),'large_256':F(2**256+57,2**255+99),
               'large_1024':F(2**1024+21,2**1023+99)}.items():
    def make(v):
        def test(self):self.compare(v)
        return test
    setattr(Enclosures,'test_ratio_'+name,make(r))

for name,kw in {'negative_ratio':{'r':F(-1,7)},'above_domain':{'r':F(2)+F(1,2**100)},
 'zero_tau':{'tau':F(0)},'zero_J':{'terms':0},'negative_J':{'terms':-1},'boolean_J':{'terms':True},
 'zero_K':{'root_bits':0},'bad_sign':{'sigma':0},'float_ratio':{'r':0.5},
 'float_tau':{'tau':0.000001},'nonexact_context':{'context':{'ARTIFICIAL_ONLY':True,'bad':0.5}}}.items():
    def make(k):
        def test(self):
            args=dict(k);r=args.pop('r',F(7,13));self.assertEqual(self.compare(r,**args)[0],'FAIL')
        return test
    setattr(Enclosures,'test_failure_'+name,make(kw))

if __name__=='__main__':unittest.main()
