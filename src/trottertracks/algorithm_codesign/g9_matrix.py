"""Small-system reference only. Floating matrices never set statistical budgets."""
from fractions import Fraction as F
from functools import lru_cache
import numpy as np
from .g9_native import provider_polynomials

I=np.eye(2,dtype=complex);X=np.array([[0,1],[1,0]],complex)
Y=np.array([[0,-1j],[1j,0]],complex);Z=np.diag([1,-1]).astype(complex)
H=(X+Z)/np.sqrt(2);S=np.diag([1,1j]);T=np.diag([1,np.exp(1j*np.pi/4)])
SINGLE={'I':I,'X':X,'Y':Y,'Z':Z,'H':H,'S':S,'s':S.conj().T,'T':T,'t':T.conj().T,
        'W':np.exp(1j*np.pi/4)*I,'w':np.exp(-1j*np.pi/4)*I}


def pauli(axis):
    out=np.ones((1,1),complex)
    for p in axis:out=np.kron(out,SINGLE[p])
    return out


def sequence(s):
    out=I.copy()
    for c in s:out=out@SINGLE[c]
    return out


def expand(u,wire,n):
    out=np.ones((1,1),complex)
    for i in range(n):out=np.kron(out,u if i==wire else I)
    return out


@lru_cache(None)
def fixed(g,n):
    if g[0] in ('CX','CZ'):
        _,control,target=g;out=np.zeros((2**n,2**n),complex)
        for i in range(2**n):
            active=(i>>(n-1-control))&1
            if g[0]=='CX':out[i^(active<<(n-1-target)),i]=1
            else:out[i,i]=-1 if active and (i>>(n-1-target))&1 else 1
        return out
    return expand(SINGLE[g[0]],g[1],n)


def circuit(gates,cache,n=4):
    out=np.eye(2**n,dtype=complex)
    for g in gates:
        if g[0]=='R':
            if cache is None:
                theta=np.arctan(float(F(g[2])));u=np.diag([np.exp(-1j*theta/2),np.exp(1j*theta/2)])
            else:u=sequence(cache[g[2]]['sequence'])
            if g[3]==-1:u=u.conj().T
            gate=expand(u,g[1],n)
        else:gate=fixed(tuple(g),n)
        out=gate@out
    return out


def Q_matrices():
    # Independent analytic rotations from the review, not Pauli collection output.
    rot=lambda axis:np.cos(np.pi/8)*np.eye(8)-1j*np.sin(np.pi/8)*pauli(axis)
    v1=rot('XXI');v2=rot('IXX')@rot('ZZI')
    return [pauli('ZII'),v1.conj().T@pauli('IZI')@v1,v2.conj().T@pauli('IIZ')@v2]


def target(p,x):
    q=Q_matrices();R=sum(float(pi)*qi for pi,qi in zip(p,q));P=np.eye(8,dtype=complex);term=P.copy()
    for n in range(1,6):term=term@(-1j*float(x)*R)/n;P+=term
    return P


def event_operator(e):
    q=Q_matrices();out=np.eye(8,dtype=complex)
    if 'pauli' in e:
        P=pauli(e['pauli'])
        if not e['rotation_sign']:out=P
        else:
            phi=np.arctan(float(e['ratio']));out=np.cos(phi)*out-1j*e['rotation_sign']*np.sin(phi)*P
    else:
        for i in e['word']:out=out@q[i]
        phi=np.arctan(float(e['ratio']));out=(np.cos(phi)*np.eye(8)-1j*np.sin(phi)*q[e['child']])@out
    return (1j**e['phase_i_power'])*out


def controlled(u):
    # qubit3 is the least-significant outer wire after system0,1,2.
    out=np.zeros((16,16),complex);out[::2,::2]=np.eye(8);out[1::2,1::2]=u
    return out


def circuit_error(e,gates,cache,helper=False):
    wanted=controlled(event_operator(e));actual=circuit(gates,cache,5 if helper else 4)
    if not helper:return float(np.linalg.norm(actual-wanted,2))
    isometry=np.zeros((32,16),complex);isometry[::2,:]=np.eye(16)
    return float(np.linalg.norm(actual@isometry-isometry@wanted,2))
