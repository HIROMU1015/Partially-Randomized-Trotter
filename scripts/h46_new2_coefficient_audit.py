"""High precision toy convergence and exact saved-decimal moment audit."""
import math
from decimal import Decimal, localcontext
import mpmath as mp


def audit():
    with localcontext() as ctx:
        ctx.prec=80
        a,b=Decimal('.42008729'),Decimal('.40899193')
        w=(b,a,1-2*(a+b),a,b)
        moments={str(k):str(sum(x**k for x in w)) for k in (1,3)}
    rows=[]
    with mp.workdps(70):
        x,z,T=mp.mpf('.73'),mp.mpf('-.42'),mp.mpf('.8')
        X=mp.matrix([[0,1],[1,0]]);Z=mp.matrix([[1,0],[0,-1]]);I=mp.eye(2)
        def exp_pauli(P,t):return mp.cos(t)*I-1j*mp.sin(t)*P
        reference=mp.expm(-1j*T*(x*X+z*Z))
        previous=None
        for q in (1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576):
            local=I
            for weight in w:
                t=T*mp.mpf(str(weight))/q
                s=exp_pauli(X,x*t/2)*exp_pauli(Z,z*t)*exp_pauli(X,x*t/2)
                local=s*local
            error=mp.norm(local**q-reference)
            order=mp.log(previous/error,2) if previous is not None else None
            rows.append(dict(q=q,operator_frobenius_error=mp.nstr(error,32),
                observed_order=float(order) if order is not None else None))
            previous=error
    return dict(schema='new2_saved_coefficient_order_audit_v1',
        coefficients=[str(x) for x in w],exact_decimal_moments=moments,
        strict_fourth_order_condition_satisfied=moments['3']=='0',
        coefficient_repair_performed=False,precision_digits=70,
        model='noncommuting .73 X - .42 Z; symmetric S2; T=.8',records=rows,
        interpretation='rounded coefficient cubic defect produces an asymptotic second-order term; practical fourth-order range is empirical')
