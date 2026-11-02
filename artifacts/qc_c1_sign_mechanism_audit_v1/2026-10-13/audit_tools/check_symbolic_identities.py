"""Formal polynomial/Pauli identities only; no numeric Hamiltonian or pilot data.

No dense matrices, parameter-point substitutions, ground solver, cost evaluation,
performance probes or saved-result statistics. Coefficients remain formal symbols.
"""
from fractions import Fraction as F
import json
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(root/'src'))
from trottertracks.representation_exploration import c1_macrocycle_algebra_v1 as free

TABLE={('I','I'):('I',0),('I','X'):('X',0),('I','Y'):('Y',0),('I','Z'):('Z',0),
       ('X','I'):('X',0),('Y','I'):('Y',0),('Z','I'):('Z',0),
       ('X','X'):('I',0),('Y','Y'):('I',0),('Z','Z'):('I',0),
       ('X','Y'):('Z',1),('Y','Z'):('X',1),('Z','X'):('Y',1),
       ('Y','X'):('Z',3),('Z','Y'):('X',3),('X','Z'):('Y',3)}
PHASE=((F(1),F(0)),(F(0),F(1)),(F(-1),F(0)),(F(0),F(-1)))


def cmul(a,b):return (a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0])


def padd(*polys):
    out={}
    for poly in polys:
        for word,(r,i) in poly.items():
            x,y=out.get(word,(F(0),F(0)));out[word]=(x+r,y+i)
    return {w:c for w,c in out.items() if c!=(0,0)}


def pscale(poly,c):return {w:cmul(v,(F(c),F(0))) for w,v in poly.items() if c}


def pmul(left,right):
    out={}
    for x,a in left.items():
        for y,b in right.items():
            word=tuple(sorted(x+y));u,v=out.get(word,(F(0),F(0)));r,i=cmul(a,b)
            out[word]=(u+r,v+i)
    return {w:c for w,c in out.items() if c!=(0,0)}


def atom(pauli,variables=(),real=1,imag=0):
    return {pauli:{tuple(sorted(variables)):(F(real),F(imag))}}


def add(*ops):
    out={}
    for op in ops:
        for p,poly in op.items():out[p]=padd(out.get(p,{}),poly)
    return {p:x for p,x in out.items() if x}


def scale(op,c):return {p:pscale(poly,c) for p,poly in op.items() if c}


def mul(left,right):
    out={}
    for a,x in left.items():
        for b,y in right.items():
            chars=[];phase=0
            for p,q in zip(a,b):
                r,k=TABLE[p,q];chars.append(r);phase+=k
            word=''.join(chars);coeff={w:cmul(c,PHASE[phase%4]) for w,c in pmul(x,y).items()}
            out[word]=padd(out.get(word,{}),coeff)
    return {p:x for p,x in out.items() if x}


def comm(a,b):return add(mul(a,b),scale(mul(b,a),-1))


def substitute(poly,ops):
    n=len(next(iter(ops['A'])));out={}
    for word,c in poly.items():
        term=atom('I'*n,real=c)
        for name in word:term=mul(term,ops[name])
        out=add(out,term)
    return out


def encode(poly):
    return [{'variables':list(w),'real':str(r),'imag':str(i)} for w,(r,i) in sorted(poly.items())]


def check():
    checks=[]
    def same(name,a,b):
        assert a==b,name;checks.append(name)
    A=add(atom('ZI',('a',),-1),atom('IZ',('b',),-1),atom('ZZ',('j',),-1))
    B=add(atom('XI',('u',)),atom('IX',('v',)))
    C=add(atom('XX',('p',)),atom('YY',('q',)))
    same('BC',comm(B,C),add(atom('ZY',('q','u'),0,2),atom('YZ',('q','v'),0,2)))
    same('BBC',comm(B,comm(B,C)),add(atom('YY',('q','u','u'),4),atom('YY',('q','v','v'),4),atom('ZZ',('q','u','v'),-8)))
    same('CBC',comm(C,comm(B,C)),add(atom('XI',('q','q','u'),-4),atom('IX',('q','q','v'),-4)))
    same('ABC_triple',comm(A,comm(B,C)),add(atom('ZX',('q','b','u'),-4),atom('XZ',('q','a','v'),-4),atom('IX',('q','j','u'),-4),atom('XI',('q','j','v'),-4)))
    same('AAC',comm(A,comm(A,C)),add(atom('XX',('a','a','p'),4),atom('XX',('b','b','p'),4),atom('XX',('a','b','q'),-8),atom('YY',('a','a','q'),4),atom('YY',('b','b','q'),4),atom('YY',('a','b','p'),-8)))
    same('CAC',comm(C,comm(A,C)),add(atom('ZI',('a','p','p'),4),atom('ZI',('a','q','q'),4),atom('ZI',('b','p','q'),-8),atom('IZ',('b','p','p'),4),atom('IZ',('b','q','q'),4),atom('IZ',('a','p','q'),-8)))
    for order in free.REPRESENTATIVES:
        D=substitute(free.kernel_error(order),{'A':A,'B':B,'C':C})
        assert 'II' not in D and D;checks.append('generic_D_traceless_nonzero_'+''.join(order))
        same('free_log_equals_recursive_'+''.join(order),free.kernel_error(order),free.nested_error(order))
    fa,fb,fc=map(free.symbol,('A','B','C'));H=free.add(fa,fb,fc)
    difference=free.add(free.kernel_error(('A','B','C')),free.scale(free.kernel_error(('A','C','B')),-1))
    same('ordering_difference',difference,free.scale(free.commutator(free.add(fb,fc),free.commutator(fb,fc)),F(1,8)))
    outer=free.add(free.scale(free.commutator(fa,free.commutator(fa,H)),F(1,24)),free.scale(free.commutator(free.add(fb,fc),free.commutator(fa,H)),F(1,12)))
    same('outer_ground_gauge',free.add(outer,free.scale(free.commutator(fa,free.commutator(fa,H)),F(1,24))),free.scale(free.commutator(H,free.commutator(fa,H)),F(1,12)))
    # A symbolic counterfamily outside the pilot Hamiltonian/native-fragment class.
    # tau is never given a number; <I>=1,<Z>=-1,<X>=<Y>=0 is a Pauli identity.
    ca=add(atom('X',real=-1),atom('Z',('tau',)),atom('Z',real=-1))
    counter={};expectations={}
    for order,target in [(('A','B','C'),padd({('tau',):(F(1,6),F(0))},{():(F(-1,6),F(0))})),
                         (('A','C','B'),padd({('tau',):(F(1,6),F(0))},{():(F(1,3),F(0))})),
                         (('B','A','C'),padd({('tau',):(F(1,6),F(0))},{():(F(-1,6),F(0))}))]:
        D=substitute(free.kernel_error(order),{'A':ca,'B':atom('X'),'C':atom('Z')})
        expectation=padd(D.get('I',{}),pscale(D.get('Z',{}),-1))
        same('formal_counterfamily_'+''.join(order),expectation,target)
        expectations[''.join(order)]=encode(expectation)
    ratios=[]
    for a,wanted in [(F(1,8),F(-1,108)),(F(1,4),F(-1,4)),(F(1,3),F(-2)),(F(3,8),F(-27,4))]:
        ratio=-2*(a/(1-2*a))**3;same('fixed_time_ratio_'+str(a),ratio,wanted)
        ratios.append({'a':str(a),'required_dk_over_dj':str(ratio)})
    return {'status':'EXACT_SYMBOLIC_IDENTITIES_PASS','identities':checks,
            'formal_variables':['a','b','j','u','v','p','q','tau'],
            'fixed_weight_ratio_conditions':ratios,'counterfamily_d_polynomials':expectations,
            'counterfamily_scope':'general three-fragment dictionary only;not pilot family/native commuting-Pauli template;no tau value assigned',
            'pilot_input_files_read':0,'numeric_Hamiltonians_generated':0,'parameter_point_substitutions':0,
            'dense_matrices_or_eigensolvers':0,'new_posthoc_statistics_or_physical_quantities':0,
            'weak_coupling_remainder_proof':'documented manual analytic proof;not numerically calibrated on saved inputs',
            'formal_algorithm_or_research_adoption':False,'mandatory_stop':True}


if __name__=='__main__':
    result=check();dest=Path(__file__).resolve().parents[1]/'symbolic_identity_checks.json'
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('identities','counterfamily_d_polynomials')},ensure_ascii=False))
