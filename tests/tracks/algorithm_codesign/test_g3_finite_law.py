"""Synthetic weighted-event laws; no saved science-profile scoring or synthesis."""
from copy import deepcopy
from fractions import Fraction as F
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[3]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
r=load('g3','scripts/tracks/algorithm_codesign/g3_finite_law.py')
fixtures=load('g2_fixtures','tests/tracks/algorithm_codesign/test_g2_saved_diagnostic.py')


def events():
    table,_=fixtures.synthetic()
    cols={c['id']:c for c in table['tables']['1/3']['columns']}
    w=r.g.vertices(F(1,3))['J1'];precision={k:'1e-4' for k,v in w.items() if v}
    return r.make_events('1/3',w,precision,cols)


def valid():
    es=events();rule,q=r.proposals(es,'T')[-1]
    return r.candidate(es,q,'1/3','synthetic','T',rule)


class FiniteLaw(unittest.TestCase):
    def test_dyadic_positive_sum_and_denominator(self):
        q=r.dyadic([F(1,7),F(3,7),F(3,7)])
        self.assertEqual(sum(q),1);self.assertGreater(min(q),0)
        self.assertTrue(all(r.N%v.denominator==0 for v in q))

    def test_lost_support_rejected(self):
        with self.assertRaises(ValueError):r.dyadic([F(1,10**100),1])

    def test_zero_probability_rejected(self):
        with self.assertRaises(ValueError):r.dyadic([0,1])

    def test_rational_coefficients_and_independent_certificate(self):
        v=valid();proof=r.certify(v)
        self.assertTrue(proof['certified'])
        self.assertLess(F(proof['degree_mean_residual_upper']),F(1,10**12))

    def test_conditional_law_preserved_before_coefficient_rounding(self):
        v=valid();groups={}
        for e in v['event_source']:
            val=F(e['coefficient_mid'])/F(e['conditional_probability'])
            if e['column_id'] in groups:self.assertEqual(val,groups[e['column_id']])
            groups[e['column_id']]=val

    def test_source_phase_binding_tamper_rejected(self):
        t,_=fixtures.synthetic();cols={c['id']:c for c in t['tables']['1/3']['columns']}
        v=valid();r.certify(v,source_columns=cols)
        v['event_source'][0]['phase_i_power']=1
        with self.assertRaises(ValueError):r.certify(v,source_columns=cols)

    def test_all_zero_cost_not_replaced(self):
        es=events()
        for e in es:e['native_cost']['T']=0
        rule,q=r.proposals(es,'T')[0]
        c=r.candidate(es,q,'1/3','synthetic','T',rule)
        self.assertEqual(F(c['resource_total']['T']),0)
        self.assertGreater(F(c['resource_total']['1Q']),0)
        r.certify(c)

    def test_zero_cost_full_support_no_epsilon_cost(self):
        es=events();before=deepcopy(es)
        for rule,q in r.proposals(es,'T'):
            self.assertGreater(min(q),0);self.assertEqual(sum(q),1)
            self.assertGreater(max(F(e['coefficient_mid'])/p for e,p in zip(es,q)),0)
        self.assertEqual(es,before)

    def test_weight_tamper_rejected(self):
        v=valid();v['weights_exact'][0]=str(F(v['weights_exact'][0])+1)
        with self.assertRaises(ValueError):r.certify(v)

    def test_shots_tamper_rejected(self):
        v=valid();v['shots_per_axis']=1
        with self.assertRaises(ValueError):r.certify(v)

    def test_resource_tamper_rejected(self):
        v=valid();v['resource_total']['CX']='0'
        with self.assertRaises(ValueError):r.certify(v)

    def test_external_nonobjective_cap_rejected(self):
        with self.assertRaises(ValueError):r.certify(valid(),{'CX':'1'})

    def test_tight_valid_caps(self):
        v=valid();self.assertTrue(r.certify(v,v['resource_total'])['certified'])

    def test_confidence_uses_actual_max_weight(self):
        v=valid();w=list(map(F,v['weights_exact']));q=list(map(F,v['q_exact']))
        self.assertEqual(F(v['L_exact']),max(w))
        self.assertEqual(F(v['m2_exact']),sum(p*z*z for p,z in zip(q,w)))

    def test_direct_readout_count(self):
        v=valid();self.assertEqual(F(v['resource_total']['1Q']),v['shots_per_axis']*(2*F(v['expected_cost']['1Q'])+5))

    def test_profile_scope_count(self):
        w=r.g.vertices(F(1,3));self.assertEqual(sum(3**sum(v>0 for v in w[a].values()) for a in r.ARMS),144)

    def test_m2_range_finite_for_zero_cost(self):
        v=valid();self.assertGreater(F(v['L_exact']),0);self.assertGreater(F(v['m2_exact']),0)
        self.assertLess(v['shots_per_axis'],10**9)


if __name__=='__main__':unittest.main()
