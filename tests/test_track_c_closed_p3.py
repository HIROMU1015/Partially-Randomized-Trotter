"""Artificial noncommuting finite-mean and digital correction checks."""
from trottertracks.track_c_closed_p3_validation import synthetic

def test_closed_p3_operator_and_exact_proposal():
    rows=synthetic()
    assert len(rows)==4 and max(r['operator_error'] for r in rows)<1e-12


def test_independent_ordered_microstep_means():
    from fractions import Fraction as F
    import itertools
    import numpy as np
    from trottertracks.track_c_closed_p3 import CachedCanonical,BITS,ETA,RHO,angle
    from trotterlib.rte import finite_taylor_operator
    X=np.array([[0,1],[1,0]],complex);Z=np.diag([1.,-1.]);ops=[X,Z]
    def events(p,x):
        g=CachedCanonical(p,x,3,'closed_P3_tail',root_bits=BITS,probability_bits=BITS,eta=ETA,rho=RHO);out=[]
        for gi,group in enumerate(g.groups):
            words=[()] if group.degree==0 else [(group.first,1-group.first)]
            for word in words:
                for child in range(2):
                    if word and child==word[0]:continue
                    e=g.event(gi,tuple(word),child);theta,_=angle(e['ratio']);v=np.eye(2,dtype=complex)
                    for j in reversed(word):v=ops[j]@v
                    v=1j**e['phase_i_power']*(np.cos(theta)*np.eye(2)-1j*np.sin(theta)*ops[child])@v
                    out.append((e,v))
        return out
    p1=(F(1,3),F(2,3));p2=(F(3,4),F(1,4));x1=F(1,5);x2=F(1,7)
    e1=events(p1,x1);e2=events(p2,x2);mean=np.zeros((2,2),complex)
    for (a,u),(b,v) in itertools.product(e1,e2):
        coefficient=a['proposal']*b['proposal']*a['weight']*b['weight']
        assert coefficient==a['coefficient']*b['coefficient']
        mean+=float(coefficient)*(v@u)
    m1=finite_taylor_operator(sum(float(p)*q for p,q in zip(p1,ops)),float(x1),2)
    m2=finite_taylor_operator(sum(float(p)*q for p,q in zip(p2,ops)),float(x2),2)
    assert np.linalg.norm(mean-m2@m1)<1e-12
    assert np.linalg.norm(mean-m1@m2)>1e-4


def test_compressed_interval_path_weight():
    from fractions import Fraction as F
    from trottertracks.track_c_closed_p3_weights import compressed_weight
    ws=[F(11,10)]*100+[F(13,12)]*30
    value,error,data=compressed_weight(ws)
    exact_rounded=F.from_float(float(ws[0]))**100*F.from_float(float(ws[-1]))**30
    assert abs(F.from_float(value)-exact_rounded)<=error
    assert data['distinct_binary_weights']==2 and data['events']==130
