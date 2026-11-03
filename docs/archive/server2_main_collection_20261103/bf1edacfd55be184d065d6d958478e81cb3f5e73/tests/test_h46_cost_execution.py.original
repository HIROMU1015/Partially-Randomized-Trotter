"""Additional artificial tests; no real arrays, trajectories or transpile."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
from fractions import Fraction as F

ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('h46_exec',ROOT/'scripts/h46_cost_execution.py')
x=importlib.util.module_from_spec(s);sys.modules[s.name]=x;s.loader.exec_module(x)


class AdapterTests(unittest.TestCase):
    def fixture(self):
        b=x.m.Basis('b','sha',(x.m.Gate('rz',(0,),.31),))
        p=dict(n=1,preparation_hash='fixture',blocks=[dict(one=True,basis=b,diagonal=(.2,),lam=None)],
               components=[dict(component_id='Z',probability=1.,coefficient_sign=1,is_identity=False,support=(0,),basis=b)],
               constant=.1,extracted=.05,lambda_r=.2)
        c=dict(id='fake_PR',method='B2',q=2,r=2,K=2,formula='2nd',prefix=0)
        primitive=lambda *a:types.SimpleNamespace(global_phase=0.,rz=((0,.2),),rzz=())
        rte=types.SimpleNamespace(finite_rte_distribution=lambda tau,k:types.SimpleNamespace(orders=(0,2),order_probabilities=(.9,.1),to_dict=lambda:dict(orders=[0,2],order_probabilities=[.9,.1])))
        return p,c,primitive,rte

    def test_grouped_mixed_is_prefix_A_tail_B(self):
        p,c,f,rte=self.fixture()
        with patch.object(x,'factory',return_value=f):
            rows,law=x.evaluate(p,c,['A','B','NATIVE_MIXED'],None,rte)
        components,orders,tau,_=x.law(p,c,rte)
        expected=x.lib.native_expectation(model='NATIVE_MIXED',bases=(p['blocks'][0]['basis'],),primitive_factory=f,
            T=.8,q=2,ancilla=1,scalar_phase=-.15*.8,components=components,orders=orders,r=2,tau=tau)
        mixed=[r for r in rows if r['model']=='NATIVE_MIXED']
        for r in mixed:self.assertEqual(r['lower'],x.packed_cost(expected[r['axis']].lower))
        self.assertTrue(law['all_positive_orders_included'])

    def test_grouped_support_includes_rare_order(self):
        p,c,_,rte=self.fixture()
        rte.finite_rte_distribution=lambda t,k:types.SimpleNamespace(orders=(0,2,4),order_probabilities=(1.,1e-9,1e-18),to_dict=lambda:{})
        comps,orders,_,_=x.law(p,c,rte)
        self.assertGreater(orders[4],0);self.assertEqual(sum(orders.values()),1)
        self.assertEqual(sum(c.probability for c in comps),1)

    def test_pf_native_matches_A(self):
        p,c,f,rte=self.fixture();c.update(method='B1',formula='4th',r=0,K=0)
        with patch.object(x,'factory',return_value=f):rows,_=x.evaluate(p,c,['A','B','NATIVE_MIXED'],None,rte)
        for axis in ('cosine','sine'):
            rr={r['model']:r for r in rows if r['axis']==axis}
            self.assertEqual(rr['A']['lower'],rr['NATIVE_MIXED']['lower'])

    def test_invalid_grant_before_source_or_input(self):
        with patch.object(x.subprocess,'check_output',return_value='1'*40):
            with self.assertRaises(ValueError):x.verify(dict(sealed=False),dict(approved=False))

    def test_descriptions_retain_signed_zero(self):
        p,_,_,_=self.fixture();p['blocks'][0]['diagonal']=(-0.,)
        self.assertEqual(x.describe(p)['blocks'][0]['diagonal_hex'],['-0x0.0p+0'])

    def test_stage2_fixed_panel_cannot_expand(self):
        plan=dict(SOURCE='1'*40,sealed=True,mandatory_stop=True,stage=2,candidates=[dict(id='H8')])
        grant=dict(SOURCE='1'*40,approved=True,runtime_authorization=True,status='APPROVED',one_shot=True,automatic_retry=False,stage=2,plan_digest=x.lib.digest(plan))
        with patch.object(x.subprocess,'check_output',return_value='1'*40):
            with self.assertRaisesRegex(ValueError,'FIXED_SEVEN_PANEL'):x.verify(plan,grant)

    def test_saved_law_bit_drift_is_rejected(self):
        a=dict(native_distribution={'probability':.1},tau_hex=(.2).hex(),lambda_r_hex=(.3).hex())
        x.check_saved_law(a,dict(a))
        with self.assertRaisesRegex(ValueError,'PINNED_NATIVE_LAW'):
            x.check_saved_law(a,dict(a,tau_hex=(.20000000000000004).hex()))


if __name__=='__main__':unittest.main()
