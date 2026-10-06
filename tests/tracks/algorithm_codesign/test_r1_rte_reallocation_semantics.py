"""Independent off-domain small-matrix semantics; no R1 registered acquisition."""
from fractions import Fraction as F
import unittest
import mpmath as mp
from trottertracks.algorithm_codesign.rte_reallocation.model import events,CONTEXTS,ARMS,P
from trottertracks.algorithm_codesign.rte_reallocation.native import lower_event,planned_angles,Gate,Angle,simplify
from trottertracks.algorithm_codesign.rte_reallocation.numeric import configure,angle_value,strict_guard

X=mp.matrix([[0,1],[1,0]]);Y=mp.matrix([[0,-mp.j],[mp.j,0]]);Z=mp.diag([1,-1]);I=mp.eye(2)


def tensor(a,b):
    return mp.matrix([[a[i//b.rows,j//b.cols]*b[i%b.rows,j%b.cols]
                       for j in range(a.cols*b.cols)] for i in range(a.rows*b.rows)])


def qmatrix(context,ref):
    if isinstance(ref,str):return tensor({'I':I,'X':X,'Y':Y,'Z':Z}[ref[0]],{'I':I,'X':X,'Y':Y,'Z':Z}[ref[1]])
    if ref==0:return tensor(Z,I)
    if context=='pauli_commuting':return tensor(I,Z)
    if context=='pauli_noncommuting':return tensor(X,X)
    xx=tensor(X,X);v=mp.cos(mp.pi/16)*mp.eye(4)-mp.j*mp.sin(mp.pi/16)*xx
    return v.transpose_conj()*tensor(I,Z)*v


def embed(g,n):
    dim=2**n
    if g.name=='GLOBAL':return mp.j**g.phase*mp.eye(dim)
    if g.name=='CX':
        out=mp.zeros(dim);c,t=g.wires
        for j in range(dim):out[j^(1<<(n-1-t)) if j&(1<<(n-1-c)) else j,j]=1
        return out
    if g.name=='RZ':
        theta=angle_value(g.angle,mp.mp);u=mp.diag([mp.exp(-mp.j*theta/2),mp.exp(mp.j*theta/2)])
    else:u={'H':mp.matrix([[1,1],[1,-1]])/mp.sqrt(2),'X':X,'Y':Y,'Z':Z,
            'S':mp.diag([1,mp.j]),'Sdg':mp.diag([1,-mp.j])}[g.name]
    out=mp.matrix([[1]])
    for wire in range(n):out=tensor(out,u if wire==g.wires[0] else I)
    return out


def matrix(circuit,n):
    u=mp.eye(2**n)
    for g in circuit:u=embed(g,n)*u
    return u


def literal_event(context,e):
    u=mp.eye(4)
    for ref in e.word:u=qmatrix(context,ref)*u
    if e.b:
        a,b=mp.mpf(e.a.numerator)/e.a.denominator,mp.mpf(e.b.numerator)/e.b.denominator
        u=(a*mp.eye(4)-mp.j*e.rotation_sign*b*qmatrix(context,e.rotation))*u/mp.sqrt(a*a+b*b)
    return mp.j**e.phase*u


def controlled(u):return tensor(mp.eye(4),mp.diag([1,0]))+tensor(u,mp.diag([0,1]))


def max_error(a,b):return max(abs(a[i,j]-b[i,j]) for i in range(a.rows) for j in range(a.cols))


class Semantics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):configure(80)
    def test_every_off_domain_event_preserves_phase_basis_and_complement(self):
        # x=1/3 is not a registered R1 x. No counts/synthesis/resource result.
        for context in CONTEXTS:
            for sigma in (-1,1):
                for arm in ARMS+(('CTS_collected',) if context!='distinct_basis' else ()):
                    for e in events(context,F(1,3),sigma,arm):
                        literal=literal_event(context,e)
                        for ctrl in (False,True):
                            with self.subTest(context=context,sigma=sigma,arm=arm,label=e.label,controlled=ctrl):
                                target=controlled(literal) if ctrl else literal
                                self.assertLess(max_error(matrix(lower_event(context,e,ctrl),3 if ctrl else 2),target),mp.mpf('1e-60'))
    def test_off_domain_all_arm_finite_means_match_same_polynomial(self):
        x=mp.mpf(1)/3
        for context in CONTEXTS:
            R=mp.mpf(3)/4*qmatrix(context,0)+mp.mpf(1)/4*qmatrix(context,1)
            for sigma in (-1,1):
                target=mp.eye(4)-mp.j*sigma*x*R-x*x/2*R*R+mp.j*sigma*x**3/6*R*R*R
                for arm in ARMS+(('CTS_collected',) if context!='distinct_basis' else ()):
                    mean=mp.zeros(4)
                    for e in events(context,F(1,3),sigma,arm):
                        c=mp.sqrt(mp.mpf(e.norm_square.numerator)/e.norm_square.denominator)
                        p=mp.mpf(e.label_probability.numerator)/e.label_probability.denominator
                        mean+=c*p*literal_event(context,e)
                    self.assertLess(max_error(mean,target),mp.mpf('1e-60'))
    def test_discarding_odd_controlled_phase_is_detectably_wrong(self):
        e=next(e for e in events('distinct_basis',F(1,3),1,'A') if e.complement)
        circuit=lower_event('distinct_basis',e,True)
        mutant=[g for g in circuit if not(g.name=='Z' and g.wires==(2,))]
        self.assertGreater(max_error(matrix(mutant,3),controlled(literal_event('distinct_basis',e))),mp.mpf('0.5'))
    def test_missing_basis_inverse_is_detectably_wrong(self):
        e=next(e for e in events('distinct_basis',F(1,3),1,'ordinary') if e.rotation==1 and not e.word)
        circuit=lower_event('distinct_basis',e,True)
        self.assertGreater(max_error(matrix(circuit[:-7],3),controlled(literal_event('distinct_basis',e))),mp.mpf('0.05'))
    def test_off_domain_lowering_keys_are_in_same_static_angle_rule(self):
        available=planned_angles(['1/3'])
        for context in CONTEXTS:
            for sigma in (-1,1):
                for arm in ARMS+(('CTS_collected',) if context!='distinct_basis' else ()):
                    for e in events(context,F(1,3),sigma,arm):
                        for ctrl in (False,True):
                            for g in lower_event(context,e,ctrl):
                                if g.angle:self.assertIn(g.angle.key,available)
    def test_identity_and_zero_support_avoid_division_by_rho(self):
        for arm in ARMS:
            es=events('distinct_basis',0,1,arm)
            self.assertEqual(len(es),1);self.assertEqual(lower_event('distinct_basis',es[0],True),[])
    def test_CTS_is_not_available_with_distinct_basis_I0(self):
        with self.assertRaises(ValueError):events('distinct_basis',F(1,3),1,'CTS_collected')
    def test_common_simplification_preserves_signed_RZ_and_global_phase(self):
        angle=Angle('pi',F(1,8))
        circuit=[Gate('RZ',(0,),angle),Gate('RZ',(0,),angle.neg()),Gate('GLOBAL',phase=1)]
        self.assertEqual(simplify(circuit),[Gate('GLOBAL',phase=1)])
    def test_strict_guard_does_not_hide_synthesizer_scalar_phase(self):
        zero=Angle('pi',F(0))
        self.assertLess(strict_guard('',zero),F(1,10**60))
        self.assertGreater(strict_guard('W',zero),F(1,2))
    def test_wrong_angle_sign_is_detected_by_strict_guard(self):
        self.assertGreater(strict_guard('',Angle('atan',F(1,3))),F(1,10))


if __name__=='__main__':unittest.main()
