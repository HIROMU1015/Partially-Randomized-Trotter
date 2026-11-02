"""Off-domain CTS semantics and certificate mutants; no synthesis acquisition."""
from copy import deepcopy
from fractions import Fraction as F
from pathlib import Path
import importlib.util
import sys
import unittest
import mpmath as mp

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('g4_cts',ROOT/'scripts/tracks/algorithm_codesign/g4_matched_cts.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
spec2=importlib.util.spec_from_file_location('off_domain_matrix',Path(__file__).with_name('test_r1_rte_reallocation_semantics.py'))
mx=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(mx)

def fixture():
    d=r.cts.definition('1/3');cost=[];precision={k:'1e-4' for k in ('ZI','IZ','XY','YX')}
    for e in d['events']:
        cost.append({'x':'1/3','label':e['label'],'epsilon':'exact' if e['real_event'] else '1e-4',
            'native_cost':{'T':0 if e['real_event'] else 10,'CX':0 if e['axis']=='II' else 2,
                '1Q':1 if e['axis']=='II' else 20,'strict_event_error_upper':'0' if e['real_event'] else '1/1000000'}})
    es=r.events(d,precision,cost);rule,q=r.proposals(es,'T')[0]
    c=r.make_law('1/3',es,q,precision,'T',rule)
    return c,d,cost

class MatchedCTS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):r.configure(80)
    def test_exact_selected_quartic_field_identities(self):
        c,s=r.cts.C,r.cts.S
        self.assertEqual(c*c+s*s,1);self.assertEqual(c*s,c*c-F(1,2))
        self.assertEqual(c*c*c*c,c*c-F(1,8))
    def test_pauli_phase_and_order(self):
        self.assertEqual(r.cts.pauli_multiply('ZI','XY'),('YY',1))
        self.assertEqual(r.cts.pauli_multiply('XY','ZI'),('YY',3))
        self.assertEqual(r.cts.poly_product({('ZI',0):r.cts.alg(1)},{('XY',0):r.cts.alg(1)}),{('YY',1):r.cts.alg(1)})
    def test_off_domain_collected_P3_matches_full_matrix(self):
        x=mp.mpf(1)/3;R=mp.mpf(3)/4*mx.qmatrix('distinct_basis',0)+mp.mpf(1)/4*mx.qmatrix('distinct_basis',1)
        target=mp.eye(4)-mp.j*x*R-x*x*R*R/2+mp.j*x**3*R*R*R/6
        collected=mp.zeros(4)
        for (axis,p),a in r.cts.collected_target('1/3').items():
            lo,hi=a.interval().lo,a.interval().hi;z=(lo+hi)/2
            collected+=(mp.mpf(z.numerator)/z.denominator)*mp.j**p*mx.qmatrix('distinct_basis',axis)
        self.assertLess(mx.max_error(target,collected),mp.mpf('1e-60'))
    def test_literal_identity_is_separate_from_negative_correction(self):
        d=r.cts.definition('1/3');self.assertTrue(d['real_identity_correction_not_fused_into_rotation'])
        self.assertEqual(d['events'][0]['axis'],'II');self.assertEqual(d['events'][0]['phase_i_power'],2)
    def test_controlled_event_lowering_preserves_each_operator(self):
        d=r.cts.definition('1/3');q=F(d['fixed_rational_rotation_ratio']);t=mp.mpf(q.numerator)/q.denominator
        for e in d['events']:
            P=mx.qmatrix('distinct_basis',e['axis'])
            U=-P if e['real_event'] else (mp.eye(4)-mp.j*e['rotation_sign']*t*P)/mp.sqrt(1+t*t)
            self.assertLess(mx.max_error(mx.matrix(r.lower(d,e),3),mx.controlled(U)),mp.mpf('1e-60'))
    def test_negative_identity_phase_cannot_be_discarded(self):
        d=r.cts.definition('1/3');e=d['events'][0]
        self.assertGreater(mx.max_error(mx.matrix([],3),mx.matrix(r.lower(d,e),3)),mp.mpf('1'))
    def test_static_registered_keys_are_twelve_no_cost_acquisition(self):
        defs={x:r.cts.definition(x) for x in ('1/8','1/4')};keys=r.planned_keys(defs)
        self.assertEqual(len(keys),12);self.assertEqual({eps for _,eps in keys.values()},set(r.PRECISIONS))
        self.assertEqual({a.scale for a,_ in keys.values()},{F(-1),F(1)})
    def test_off_domain_full_operator_finite_certificate(self):
        c,d,cost=fixture();self.assertTrue(r.certify(c,d,cost)['operator_not_only_channel_mean'])
    def test_control_phase_mutant_rejected(self):
        c,d,cost=fixture();c['event_source'][0]['phase_i_power']=0
        with self.assertRaises(ValueError):r.certify(c,d,cost)
    def test_rotation_sign_mutant_rejected(self):
        c,d,cost=fixture();c['event_source'][2]['rotation_sign']*=-1
        with self.assertRaises(ValueError):r.certify(c,d,cost)
    def test_weight_mutant_rejected(self):
        c,d,cost=fixture();c['weights_exact'][0]='0'
        with self.assertRaises(ValueError):r.certify(c,d,cost)
    def test_shot_mutant_rejected(self):
        c,d,cost=fixture();c['shots_per_axis']=1
        with self.assertRaises(ValueError):r.certify(c,d,cost)
    def test_cost_mutant_rejected(self):
        c,d,cost=fixture();cost=deepcopy(cost);cost[0]['native_cost']['1Q']=0
        with self.assertRaises(ValueError):r.certify(c,d,cost)
    def test_resource_mutant_rejected(self):
        c,d,cost=fixture();c['resource_total']['T']='0'
        with self.assertRaises(ValueError):r.certify(c,d,cost)
    def test_saved_moment_mutant_rejected(self):
        c,d,cost=fixture();c['m2_exact']='0'
        with self.assertRaises(ValueError):r.certify(c,d,cost)
    def test_zero_T_event_has_positive_support_under_every_proposal(self):
        c,d,cost=fixture();es=c['event_source']
        for _,q in r.proposals(es,'T'):
            self.assertEqual(sum(q),1);self.assertGreater(min(q),0)
    def test_strict_guard_rejects_hidden_scalar_phase(self):
        from trottertracks.algorithm_codesign.rte_reallocation.numeric import strict_guard
        self.assertGreater(strict_guard('W',r.Angle('pi',F(0))),F(1,2))

if __name__=='__main__':unittest.main()
