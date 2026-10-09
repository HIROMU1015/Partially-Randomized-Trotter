"""Independent off-domain fixtures; no saved profile execution or old helper."""
from copy import deepcopy
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import importlib.util
import unittest

ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('g4',ROOT/'scripts/tracks/algorithm_codesign/g4_independent_certificate.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)

def fixture():
    x='1/3';cols={};es=[]
    for name,gamma in r.weights(x,'J1').items():
        k,a,b=r.prototypes(x)[name];norm=r.sqrt_enclosure(a*a+b*b);events=[]
        direction=[(r.Interval(a if j==k else b if j==k+1 else 0)/norm).json() for j in range(4)]
        D=[[z['lo'],z['hi']] for z in direction]
        for labels in product((0,1),repeat=k+int(b!=0)):
            p=F(1)
            for i in labels:p*=(F(3,4),F(1,4))[i]
            e={'source_label':str(k)+':'+str(labels),'label_probability':str(p),'word':list(labels[1:] if b else labels),
                'rotation':labels[0] if b else None,'rotation_sign':1,'phase_i_power':(-k)%4,'complement':name=='A1',
                'native_cost':{'T':1,'CX':1,'1Q':10,'strict_event_error_upper':'1/10000000'}}
            events.append(e);mid=(norm.lo+norm.hi)/2
            es.append({'id':name+':1e-4/'+e['source_label'],'column_id':name+':1e-4','conditional_probability':str(p),
                'coefficient_mid':str(gamma*p*mid),'coefficient_lo':str(gamma*p*norm.lo),'coefficient_hi':str(gamma*p*norm.hi),
                'D_intervals':D,**{z:e[z] for z in ('word','rotation','rotation_sign','phase_i_power','complement','native_cost')}})
        cols[name+':1e-4']={'prototype':name,'D_intervals':D,'events':events}
    N=2**60;base=N//len(es);q=[F(base+int(j<N%len(es)),N) for j in range(len(es))]
    weights=[F(e['coefficient_mid'])/p for e,p in zip(es,q)];coef=sum(max(abs(F(e['coefficient_mid'])-F(e['coefficient_lo'])),abs(F(e['coefficient_hi'])-F(e['coefficient_mid']))) for e in es)
    bias=2*sum(F(e['coefficient_mid'])*F(e['native_cost']['strict_event_error_upper']) for e in es)
    s=r.EPS-coef-bias;m2=sum(p*w*w for p,w in zip(q,weights));L=max(weights)
    n=(r.log_enclosure().hi*(2*m2/(s*s)+F(4,3)*L/s)).__ceil__()
    c={'x':x,'id':'off-domain-fixture','event_source':es,'q_exact':list(map(str,q)),'weights_exact':list(map(str,weights)),
        'shots_per_axis':n,'resource_total':{k:str(n*(2*sum(p*F(e['native_cost'][k]) for p,e in zip(q,es))+(5 if k=='1Q' else 0))) for k in ('T','CX','1Q')},'workspace_peak':1}
    return c,cols

class IndependentCertificate(unittest.TestCase):
    def test_dyadic_root_encloses_rational_square(self):
        for v in (F(0),F(4),F(7,13),F(1234,123456)):
            z=r.sqrt_enclosure(v);self.assertLessEqual(z.lo*z.lo,v);self.assertGreaterEqual(z.hi*z.hi,v)
    def test_negative_root_rejected(self):
        with self.assertRaises(ValueError):r.sqrt_enclosure(-1)
    def test_log_lower_is_rationally_certified(self):
        self.assertLess(r.exp_upper(F(37,4)),10560);self.assertGreater(r.log_enclosure().lo,F(37,4))
    def test_invalid_exp_tail_rejected(self):
        with self.assertRaises(ValueError):r.exp_upper(62)
    def test_signed_interval_arithmetic(self):
        z=r.Interval(-3,2)*r.Interval(4,5);self.assertEqual((z.lo,z.hi),(F(-15),F(10)))
    def test_denominator_crossing_zero_rejected(self):
        with self.assertRaises(ZeroDivisionError):r.Interval(1)/r.Interval(-1,1)
    def test_all_formal_means_off_domain(self):
        for arm in ('ordinary','PTSC_K0','A','J1'):r.ideal_mean('1/3',arm)
    def test_independent_finite_certificate_off_domain(self):
        c,cols=fixture();self.assertTrue(r.finite_law_certificate(c,cols)['pass'])
    def test_changed_phase_rejected(self):
        c,cols=fixture();c['event_source'][0]['phase_i_power']=2
        with self.assertRaises(ValueError):r.finite_law_certificate(c,cols)
    def test_bad_shot_certificate_rejected(self):
        c,cols=fixture();c['shots_per_axis']=1
        with self.assertRaises(ValueError):r.finite_law_certificate(c,cols)
    def test_bad_mean_interval_rejected(self):
        c,cols=fixture();c['event_source'][0]['coefficient_lo']='2'
        with self.assertRaises(ValueError):r.finite_law_certificate(c,cols)
    def test_false_total_rejected(self):
        c,cols=fixture();c['resource_total']['T']='0'
        with self.assertRaises(ValueError):r.finite_law_certificate(c,cols)
    def test_l1_bridge_condition_is_necessary_for_this_proof(self):
        # Ideal K>=r*s; without h+rd<=r, tiny cost-bearing mass can be removed.
        r0=F(10);h=F(100);c=F(1);eps=F(10);ct=F(9,10);e=abs(ct-c)
        self.assertGreaterEqual(h*c,r0*eps)
        self.assertLess(h*ct,r0*(eps-e))

if __name__=='__main__':unittest.main()
