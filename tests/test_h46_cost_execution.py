"""Additional artificial tests; no real arrays, trajectories or transpile."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
from fractions import Fraction as F
import tempfile

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

    def test_native_reuse_requires_all_bound_fields(self):
        same=x.load('_fake_same',ROOT/'scripts/h46_same_event.py')
        v=dict(cell_id='c',axis='cosine',control='symmetric_directional',fingerprint='f',event_digest='e',compiler_hash='h')
        kw=dict(candidate='c',axis='cosine',fingerprint='f',event_digest='e',compiler='h')
        self.assertTrue(same.reusable(v,**kw))
        for k in v:self.assertFalse(same.reusable(dict(v,**{k:'DIFFERENT'}),**kw))

    def test_attempt_cap_preserves_prior_reservations(self):
        same=x.load('_fake_same_cap',ROOT/'scripts/h46_same_event.py')
        with tempfile.TemporaryDirectory(dir='/home/AbeHiromu',prefix='h46-artificial-budget-') as d:
            p=Path(d)
            for i in range(8):same.take(x,p,'compile','H4:fake',str(i))
            with self.assertRaisesRegex(ValueError,'FIXED_SYSTEM_COMPILE_CAP'):same.take(x,p,'compile','H4:fake')
            self.assertEqual(len((p/'calls.jsonl').read_text().splitlines()),8)

    def test_ratio_interval_retains_endpoints(self):
        summary=x.load('_fake_summary',ROOT/'scripts/h46_cost_summary.py')
        r=summary.ratios('10','20','15')
        self.assertEqual(r['rho_lower'],'3/4');self.assertEqual(r['rho_upper'],'3/2')
        with self.assertRaises(ValueError):summary.ratios(0,20,15)

    def test_full_panel_synthesis_artificial_only(self):
        summary=x.load('_fake_panel_summary',ROOT/'scripts/h46_cost_summary.py')
        with tempfile.TemporaryDirectory(dir='/home/AbeHiromu',prefix='h46-artificial-synthesis-') as d:
            root=Path(d);out=root/'output';out.mkdir();theory=[];jobs=[]
            d0={'H4_NATIVE_FIXED16':[],'H6_EXPLORATION92':[]}
            for system in ('H4','H6'):
                for cid in x.PANEL[system]:
                    jobs.append(dict(system=system,candidate=dict(id=cid)))
                    own=out/(system+'_'+cid);own.mkdir()
                    paired=[]
                    for axis in ('cosine','sine'):
                        x.write(own/('wrapper_'+axis+'.json'),dict(metrics={'RZ':6,'CX':4},raw_native={'cost':{'rz':'9','cx':'9'}},compiled_origin='FAKE',event_digest='event',preparation_hash='p',fingerprint='f',compiler_hash='c'))
                        for model,value in [('A','10'),('B','8'),('NATIVE_MIXED','9')]:
                            cost={'rz':value,'cx':value};theory.append(dict(system=system,candidate_id=cid,model=model,axis=axis,lower=cost,upper=cost));paired.append(dict(model=model,axis=axis,cost=cost))
                    x.write(own/'paired_theory.json',paired)
                    for epsilon in (.05,.01,.005,.001,.0005):
                        for factor in ('1.0','10.0'):
                            d0['H4_NATIVE_FIXED16' if system=='H4' else 'H6_EXPLORATION92'].append(dict(cell_id=cid,epsilon=epsilon,factor=factor,sample_count=1 if cid.startswith('B1') else 8,sample_phase='fake',N_total=3,
                                accounting=dict(cost_acquisition_eligible=True,axes={'real':{'eligible':True,'shots':1},'imag':{'eligible':True,'shots':2}}),
                                metrics={metric:dict(axis_means={'cosine':v,'sine':v},G_exact={'numerator':str(3*v),'denominator':'1'}) for metric,v in [('RZ',6),('CX',4)]}))
            inputs={}
            for name,data in [('h4_theory',theory),('h6_mixed',[]),('h6_uniform',[]),('saved_accounting',d0)]:
                path=root/(name+'.json');x.write(path,data);inputs[name]={'path':str(path)}
            result=summary.summarize(x,dict(inputs=inputs,jobs=jobs),out)
            self.assertEqual(result['expectation_comparison_rows'],84)
            self.assertEqual(result['same_event_comparison_rows'],84)
            self.assertEqual(result['pf_pr_comparison_rows'],360)
            self.assertEqual(result['preserved_shot_conditions'],70)


if __name__=='__main__':unittest.main()
