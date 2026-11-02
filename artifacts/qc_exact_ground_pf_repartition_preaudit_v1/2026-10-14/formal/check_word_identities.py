"""Exact abstract-word checks only: no Hamiltonians, states, numeric expectations or PF runs."""
from fractions import Fraction as F
from pathlib import Path
import hashlib,json,sys

def clean(p): return {w:c for w,c in p.items() if c}
def add(*ps):
    out={}
    for p in ps:
        for w,c in p.items():out[w]=out.get(w,F(0))+c
    return clean(out)
def scale(p,c):return clean({w:v*c for w,v in p.items()})
def mul(p,q,cap=99):
    out={}
    for w,c in p.items():
        for v,d in q.items():
            if len(w+v)<=cap:out[w+v]=out.get(w+v,F(0))+c*d
    return clean(out)
def comm(p,q):return add(mul(p,q),scale(mul(q,p),-1))
def symbol(s):return {(s,):F(1)}
def exp3(p):
    power={():F(1)};out=power
    for k in range(1,4):
        power=scale(mul(power,p,3),F(1,k));out=add(out,power)
    return out
def log3(p):
    z=add(p,{():F(-1)})
    return add(z,scale(mul(z,z,3),F(-1,2)),scale(mul(mul(z,z,3),z,3),F(1,3)))
def log_series(fragments):
    slots=[scale(x,F(1,2)) for x in fragments[:-1]]+[fragments[-1]]+[scale(x,F(1,2)) for x in reversed(fragments[:-1])]
    out={():F(1)}
    for x in slots:out=mul(out,exp3(x),3)
    return log3(out)
def bch_recursive(fragments):
    out={}
    for i,a in enumerate(fragments[:-1]):
        b=add(*fragments[i+1:]);c=comm(a,b)
        out=add(out,scale(comm(a,c),F(1,24)),scale(comm(b,c),F(1,12)))
    return out
def substitute_t(p,labels):
    # Input is a free-word polynomial; output is a polynomial in a central t.
    out={}
    for word,c in p.items():
        terms={(0,()):c}
        for letter in word:
            new={}
            for (degree,w),coef in terms.items():
                for d,op,v in labels.get(letter,[(0,letter,F(1))]):
                    key=(degree+d,w+(op,));new[key]=new.get(key,F(0))+coef*v
            terms=new
        for key,v in terms.items():out[key]=out.get(key,F(0))+v
    return clean(out)
def coeff(p,k):return clean({w:c for (d,w),c in p.items() if d==k})
def commute_y_b(p,b):
    # Only relation used: Y commutes with chosen density fragment b.
    out={}
    for word,c in p.items():
        w=list(word)
        changed=True
        while changed:
            changed=False
            for i in range(len(w)-1):
                if w[i]==b and w[i+1]=='Y':w[i],w[i+1]=w[i+1],w[i];changed=True
        key=tuple(w);out[key]=out.get(key,F(0))+c
    return clean(out)
def eq(p,q):assert add(p,scale(q,-1))=={}

checks=[]
for n in range(2,7):
    fs=[symbol('F'+str(i)) for i in range(n)]
    direct=log_series(fs);dc={w:c for w,c in direct.items() if len(w)==3};D=bch_recursive(fs)
    eq(scale(dc,-1),D)
    eq({w:c for w,c in direct.items() if len(w)==1},add(*fs))
    assert not {w:c for w,c in direct.items() if len(w)==2}
    checks.append({'id':f'EXP_LOG_VERSUS_RECURSIVE_D_N{n}','status':'PASS'})
    for k in range(1,n):
        t=substitute_t(D,{'F0':[(0,'F0',F(1)),(1,'Y',F(1))],f'F{k}':[(0,f'F{k}',F(1)),(1,'Y',F(-1))]})
        assert coeff(t,3)=={}
        q=scale(comm(symbol('Y'),comm(symbol('Y'),add(fs[0],fs[k],scale(add(*fs[1:k]),3)))),F(-1,24))
        eq(coeff(t,2),q)
        checks.append({'id':f'LOCAL_QUADRATIC_AND_CLOSED_Q_N{n}_K{k}','status':'PASS'})
        if k==1:
            y=symbol('Y');a=fs[0];H=add(*fs)
            reduced=add(scale(comm(a,comm(a,y)),F(1,12)),scale(comm(H,comm(y,a)),F(1,8)))
            eq(commute_y_b(coeff(t,1),'F1'),commute_y_b(reduced,'F1'))
            eq(commute_y_b(coeff(t,2),'F1'),scale(comm(y,comm(a,y)),F(1,24)))
            checks.append({'id':f'FIRST_DENSITY_RESPONSE_MOD_GROUND_COMMUTATOR_N{n}','status':'PASS'})
a,b,y=symbol('A'),symbol('B'),symbol('Y');H=add(a,b);D=bch_recursive([a,b])
eq(add(D,scale(comm(a,comm(a,H)),F(1,24))),scale(comm(H,comm(a,b)),F(1,12)))
checks.append({'id':'TWO_FRAGMENT_GROUND_DOUBLE_COMMUTATOR_REDUCTION','status':'PASS'})
multi=substitute_t(bch_recursive([symbol('A'),symbol('B'),symbol('C')]),{'A':[(0,'A',F(1)),(1,'X',F(1)),(1,'Y',F(1))],'B':[(0,'B',F(1)),(1,'X',F(-1))],'C':[(0,'C',F(1)),(1,'Y',F(-1))]})
assert coeff(multi,3)
checks.append({'id':'MULTI_DIFFERENT_DIRECTION_CUBIC_OBSTRUCTION','status':'PASS','nonzero_cubic_word_terms':len(coeff(multi,3))})
out={'status':'ABSTRACT_RATIONAL_WORD_IDENTITY_PASS','checks':checks,'check_count':len(checks),
     'source_SHA256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
     'domain':'Free associative words over abstract fragments;central formal t;only Y-F1 commutation where explicitly declared',
     'certifies':['Degree-three formal log signs and recursive BCH identity','Local one-shared-direction quadratic degree and closed Q','Adjacent first-density ground-commutator-equivalent response'],
     'does_not_certify':['PSD of any numeric data','Actual ground state or energy','Finite-h PF performance','RDM numerical contraction','Gate/circuit resources','Prior-art novelty'],
     'Hamiltonians_generated':0,'eigensolver_calls':0,'numerical_expectation_calls':0,'numerical_optimization_calls':0,'PF_performance_calls':0,'compile_calls':0,'sampling_calls':0}
Path(sys.argv[1]).write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':out['status'],'check_count':len(checks),'science_numeric_calls':0}))
