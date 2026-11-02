"""Return algebra/native control on off-domain x=1/3 only; no synthesis."""
from copy import deepcopy
from fractions import Fraction as F
import importlib.util
from pathlib import Path
import unittest
import mpmath as mp

ROOT=Path(__file__).resolve().parents[3]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
r=load('g3_return','scripts/tracks/algorithm_codesign/g3_return_comparator.py')
fixture=load('g3_fixture','tests/tracks/algorithm_codesign/test_g2_saved_diagnostic.py')
s=load('r1_semantic_helpers','tests/tracks/algorithm_codesign/test_r1_rte_reallocation_semantics.py')

def columns():
    table,_=fixture.synthetic()
    old=next(c for c in table['tables']['1/3']['columns'] if c['id']=='O2:1e-4')
    costs=[{'x':'1/3','epsilon':'1e-4','label':i,'native_cost':deepcopy(old['events'][i]['native_cost'])} for i in range(2)]
    c0=r.event_column('1/3','1e-4',costs);c2=r.offdiagonal_column('1/3',old)
    return {c['id']:c for c in (c0,c2)},old

class ReturnSemantics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):r.configure(100)
    def test_exact_return_degree_identity_off_domain(self):
        r.return_mean_check('1/3');aa,b=r.return_ab('1/3')
        self.assertEqual(aa+F(5,8)*F(1,3)**2/2,1)
        self.assertEqual(b+F(5,8)*F(1,3)**3/6,F(1,3))
    def test_key_inventory_fixed_without_synthesizer(self):
        self.assertEqual(len(r.keys()),12)
        self.assertTrue(all(angle.scale in (-1,1) for angle,eps in r.keys().values()))
    def test_offdiagonal_law_and_IR_reuse(self):
        cols,old=columns();c=cols['RET2:1e-4']
        self.assertEqual(sum(F(e['label_probability']) for e in c['events']),1)
        for e in c['events']:
            orig=next(z for z in old['events'] if z['source_label']==e['source_label'])
            self.assertEqual(e['native_cost'],orig['native_cost'])
            self.assertEqual(e['phase_i_power'],2)
            self.assertEqual(e['word'],orig['word'])
            self.assertEqual(F(e['label_probability']),F(orig['label_probability'])/F(3,8))
    def test_offdiagonal_low_return_coefficients_positive(self):
        cols,old=columns();c=cols['RET2:1e-4']
        self.assertEqual(list(map(F,c['D_intervals'][0])),[F(5,3)*F(z) for z in old['D_intervals'][2]])
        self.assertEqual(list(map(F,c['D_intervals'][1])),[F(5,3)*F(z) for z in old['D_intervals'][3]])
    def test_whole_mean_against_literal_noncommuting_matrices(self):
        x=F(1,3);aa,b=r.return_ab(x);mean=mp.zeros(4)
        for i,p in enumerate((F(3,4),F(1,4))):
            e=r.Event('known_return','zero',aa,b,p,rotation=i)
            mean+=mp.sqrt(s.mp.mpf(e.norm_square.numerator)/e.norm_square.denominator)*s.mp.mpf(p.numerator)/p.denominator*s.literal_event('distinct_basis',e)
        # Original O2 only; unequal adjacent words, no extra chi multiplier.
        for e in s.events('distinct_basis',x,1,'ordinary'):
            if len(e.word)==2 and e.word[0]!=e.word[1]:
                mean+=mp.sqrt(mp.mpf(e.norm_square.numerator)/e.norm_square.denominator)*mp.mpf(e.label_probability.numerator)/e.label_probability.denominator*s.literal_event('distinct_basis',e)
        R=mp.mpf(3)/4*s.qmatrix('distinct_basis',0)+mp.mpf(1)/4*s.qmatrix('distinct_basis',1)
        z=mp.mpf(1)/3;target=mp.eye(4)-mp.j*z*R-z*z/2*R*R+mp.j*z**3/6*R*R*R
        self.assertLess(s.max_error(mean,target),mp.mpf('1e-80'))
    def test_returned_control_preserves_relative_phase_and_basis(self):
        aa,b=r.return_ab('1/3')
        for i in range(2):
            e=r.Event('known_return','zero',aa,b,F(1,2),rotation=i)
            self.assertLess(s.max_error(s.matrix(r.lower_event('distinct_basis',e,True),3),s.controlled(s.literal_event('distinct_basis',e))),mp.mpf('1e-80'))
    def test_strict_phase_guard_rejects_global_phase(self):
        from trottertracks.algorithm_codesign.rte_reallocation.numeric import strict_guard
        angle=r.Angle('pi',0)
        self.assertLess(strict_guard('',angle),F(1,10**80))
        self.assertGreater(strict_guard('WWWW',angle),2)
    def test_finite_weighted_return_certificate(self):
        cols,_=columns();es=r.return_events('1/3',{'RET0':'1e-4','RET2':'1e-4'},cols)
        for axis in r.a.AXES:
            rule,q=r.a.proposals(es,axis)[-1]
            c=r.a.candidate(es,q,'1/3','synthetic return',axis,rule)
            self.assertTrue(r.a.certify(c,source_columns=cols)['certified'])
    def test_return_phase_mutation_rejected(self):
        cols,_=columns();es=r.return_events('1/3',{'RET0':'1e-4','RET2':'1e-4'},cols)
        rule,q=r.a.proposals(es,'T')[0];c=r.a.candidate(es,q,'1/3','synthetic return','T',rule)
        c['event_source'][0]['phase_i_power']=2
        with self.assertRaises(ValueError):r.a.certify(c,source_columns=cols)
    def test_offdiagonal_source_not_mutated(self):
        _,old=columns();before=deepcopy(old);r.offdiagonal_column('1/3',old)
        self.assertEqual(before,old)

if __name__=='__main__':unittest.main()
