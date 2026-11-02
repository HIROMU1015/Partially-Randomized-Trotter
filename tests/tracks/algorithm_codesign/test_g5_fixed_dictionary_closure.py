"""Off-domain rational tests only. No registered table/law read or runner call."""
import copy
import importlib.util
from fractions import Fraction as F
from pathlib import Path
import unittest

P=Path(__file__).resolve().parents[3]/'scripts/tracks/algorithm_codesign/g5_fixed_dictionary_closure.py'
spec=importlib.util.spec_from_file_location('g5',P)
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)

def fixture():
    x='1/3';ideal,ls,signs=g.ideal_cts_intervals(x);tiny=F(1,10**60)
    ratio=(ls.lo+ls.hi)/2;events=[];costs=[];mids=[]
    for label,z in ideal.items():
        real=label.startswith('real_minus_');axis=label.removeprefix('real_minus_').removeprefix('rotation_')
        coefficient={'lo':str(z.lo-tiny),'hi':str(z.hi+tiny)}
        angle=F(0) if real else 2*max(abs(ratio-ls.lo),abs(ratio-ls.hi))+tiny
        e={'label':label,'axis':axis,'real_event':real,'phase_i_power':2 if real else 0,
           'rotation_sign':0 if real else signs[axis],'ideal_coefficient':coefficient,'angle_error_upper':str(angle)}
        midpoint=(z.lo+z.hi)/2;mids.append(midpoint)
        native={'T':1,'CX':1,'1Q':1,'strict_event_error_upper':'0'}
        eps='exact' if real else 'synthetic'
        events.append({**e,'epsilon':eps,'coefficient_mid':str(midpoint),'native_cost':native})
        costs.append({'x':x,'label':label,'epsilon':eps,'native_cost':native})
    q=[F(1,8)]*5+[F(3,8)];w=[a/p for a,p in zip(mids,q)]
    coef=sum(max(abs(a-F(e['ideal_coefficient'][k])) for k in ('lo','hi')) for a,e in zip(mids,events))
    bias=2*sum(a*F(e['angle_error_upper']) for a,e in zip(mids,events));n=10**8
    law={'x':x,'optimized_axis':'1Q','event_source':events,'q_exact':list(map(str,q)),
        'weights_exact':list(map(str,w)),'sampler_denominator':str(2**60),
        'precision':{k:'synthetic' for k in signs},'shots_per_axis':n,
        'resource_total':{'T':str(2*n),'CX':str(2*n),'1Q':str(7*n)},'workspace_peak':1,
        'm2_exact':str(sum(p*z*z for p,z in zip(q,w))),'L_exact':str(max(w)),
        'coefficient_error':str(coef),'synthesis_and_angle_bias':str(bias),'remaining':str(g.v.EPS-coef-bias)}
    definition={'events':[{k:v for k,v in e.items() if k not in ('epsilon','coefficient_mid','native_cost')} for e in events],
                'fixed_rational_rotation_ratio':str(ratio)}
    return law,definition,costs

class TestG5(unittest.TestCase):
    def test_all_six_vertices_preserve_offdomain_mean(self):
        D,t=g.degree_system('1/3')
        for point in g.vertices('1/3').values():
            z=[point.get(name,F(0)) for name in g.ORDER]
            self.assertEqual([sum(a*b for a,b in zip(row,z)) for row in D],t)
    def test_full_precision_count(self):
        self.assertEqual(sum(3**len(z) for z in g.vertices('1/3').values()),252)
    def test_singular_coefficient_basis(self):
        self.assertIsNone(g.solve_square([[F(1),F(1)],[F(2),F(2)]],[F(1),F(2)]))
    def test_exact_coefficient_elimination(self):
        self.assertEqual(g.solve_square([[F(1),F(2)],[F(3),F(1)]],[F(5),F(5)]),[F(1),F(2)])
    def test_price_averages_roots_not_root_of_average(self):
        # O0's synthetic conditional costs 1 and 9 have E sqrt(C)=2.
        cols={'O0:test':{'events':[{'label_probability':'1/2','native_cost':{'T':1,'strict_event_error_upper':'0'}},
                                            {'label_probability':'1/2','native_cost':{'T':9,'strict_event_error_upper':'0'}}]}}
        p=g.price('1/3','synthetic',{'O0':F(1)},{'O0':'test'},cols,'T')
        lo,hi=map(F,(p['K']['lo'],p['K']['hi']))
        self.assertLessEqual(lo*lo,F(40,9));self.assertGreaterEqual(hi*hi,F(40,9))
        self.assertLess(hi*hi,F(50,9))
    def test_bridge_boundary(self):
        col={'x':{'id':'synthetic','events':[{'source_label':'a','native_cost':{'T':4,'strict_event_error_upper':'1/4'}}]}}
        self.assertTrue(g.bridge(col,'T',F(4))['pass'])
        self.assertFalse(g.bridge(col,'T',F(3))['pass'])
    def test_bridge_rejects_false_squared_positive_d_over_one(self):
        col={'x':{'id':'synthetic','events':[{'source_label':'a','native_cost':{'T':1,'strict_event_error_upper':'1'}}]}}
        self.assertFalse(g.bridge(col,'T',F(10))['pass'])
    def test_cts_offdomain_full_mean_and_confidence(self):
        law,d,costs=fixture();self.assertTrue(g.verify_cts(law,d,costs,expected_x='1/3')['pass'])
    def test_cts_wrong_phase_even_if_definition_also_changed(self):
        law,d,costs=fixture();law['event_source'][0]['phase_i_power']=0;d['events'][0]['phase_i_power']=0
        with self.assertRaisesRegex(ValueError,'independent CTS direction'):g.verify_cts(law,d,costs,expected_x='1/3')
    def test_cts_missing_support(self):
        law,d,costs=fixture();law['event_source'][0]=copy.deepcopy(law['event_source'][1])
        with self.assertRaisesRegex(ValueError,'support identity'):g.verify_cts(law,d,costs,expected_x='1/3')
    def test_cts_wrong_saved_native_price(self):
        law,d,costs=fixture();law['event_source'][0]['native_cost']=dict(law['event_source'][0]['native_cost'],T=2)
        with self.assertRaisesRegex(ValueError,'cost binding'):g.verify_cts(law,d,costs,expected_x='1/3')
    def test_cts_nominal_interval_is_not_certificate(self):
        law,d,costs=fixture();z=law['event_source'][2]['coefficient_mid']
        law['event_source'][2]['ideal_coefficient']={'lo':z,'hi':z};d['events'][2]['ideal_coefficient']={'lo':z,'hi':z}
        with self.assertRaisesRegex(ValueError,'independent ideal enclosure'):g.verify_cts(law,d,costs,expected_x='1/3')
    def test_cts_insufficient_shots(self):
        law,d,costs=fixture();law['shots_per_axis']=1
        with self.assertRaisesRegex(ValueError,'confidence'):g.verify_cts(law,d,costs,expected_x='1/3')
    def test_log_bound_with_rational_remainder(self):
        self.assertLess(g.v.exp_upper(F(37,4)),10560)
        self.assertGreater(g.v.log_enclosure().lo,F(37,4))

if __name__=='__main__':unittest.main()
