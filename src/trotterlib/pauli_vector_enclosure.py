"""Matrix-free L2 enclosures around binary64 vector centers.

Arb encloses all scalar coefficients. The vector error uses the standard
IEEE-754 gamma_k bound, with k=8 times the number of complex terms (covering
real multiply/add operations and arbitrary reduction order). No fast-math or
float32 is used. Signed Pauli permutations are exact. See the report for the
roundoff/contraction proof; this does not bound compiled-circuit rho.
"""
from fractions import Fraction as Q
import math
import numpy as np
from . import h4_ball_evidence as C

U=2.**-53
def up(x):return math.nextafter(float(x),math.inf)
def add(x,y):return up(x+y)
def mul(x,y):return up(x*y)
def gamma(k):return up(k*U/(1-k*U))
def upper(a):return up(float(a.abs_upper()))

def coefficient(z):
    z=C.acb(z);x=float(z.real.mid());y=float(z.imag.mid())
    # L1 over the two scalar coordinates also bounds complex absolute value.
    err=upper(z.real-C.real(x))+upper(z.imag-C.real(y))
    return complex(x,y),up(abs(x)+abs(y)),up(err)

def mapping(label,sign=1):
    n=len(label);col=np.arange(1<<n,dtype=np.int64);row=col.copy();turn=np.zeros(len(col),dtype=np.int8)
    if sign==-1:turn+=2
    for j,s in enumerate(reversed(label)):
        bit=(col>>j)&1
        if s in 'XY':row^=1<<j
        if s=='Y':turn+=(1-2*bit).astype(np.int8)
        elif s=='Z':turn+=(2*bit).astype(np.int8)
    inv=np.empty_like(row);inv[row]=col
    return inv,(turn[inv]%4)

def permute(v,pm):
    inv,t=pm;w=v[inv].copy()
    # Multiplication by +/-1 or +/-i is a sign/swap, without rounded products.
    r=w.real.copy();i=w.imag.copy()
    w.real=np.where(t==0,r,np.where(t==1,-i,np.where(t==2,-r,i)))
    w.imag=np.where(t==0,i,np.where(t==1,r,np.where(t==2,-i,-r)))
    return w

class Vector:
    def __init__(self,v,error,norm):self.v=v;self.e=error;self.norm=norm
    def pauli(self,pm):return Vector(permute(self.v,pm),self.e,self.norm)

def linear(terms):
    """terms=(Arb coefficient, enclosed vector); full error retained."""
    out=np.zeros_like(terms[0][1].v);e=0.;work=0.;norm=0.
    for coef,v in terms:
        center,a,d=coefficient(coef);nc=add(v.norm,v.e)
        out+=center*v.v
        e=add(e,add(mul(add(a,d),v.e),mul(d,v.norm)))
        work=add(work,mul(a,nc));norm=add(norm,mul(add(a,d),v.norm))
    e=add(e,add(mul(gamma(8*len(terms)),work),len(out)*len(terms)*2.**-1000))
    if not np.isfinite(out).all() or not math.isfinite(e):raise ValueError('NONFINITE_VECTOR')
    return Vector(out,e,norm)

def rotation(v,pm,angle):
    a=C.real(angle);out=linear([(C.acb(a.cos()),v),(C.acb(0,-a.sin()),v.pauli(pm))])
    out.norm=v.norm # exact Pauli exponential is unitary
    return out

def initial(raw):
    nu=sum((C.rational(x)**2+C.rational(y)**2 for x,y in raw),Q(0))
    if nu<=0:raise ValueError('NONPOSITIVE_RAW_NORM')
    v=np.array([complex(x,y) for x,y in raw],np.complex128)
    return Vector(v,0.,upper(C.real(nu).sqrt())),nu

def signal(raw,out,nu):
    bra=np.array([complex(x,y) for x,y in raw],np.complex128)
    z=complex(np.sum(bra.conjugate()*out.v,dtype=np.complex128))
    sn=upper(C.real(nu).sqrt())
    e=add(mul(sn,out.e),mul(gamma(8*len(bra)),mul(sn,add(out.norm,out.e))))
    e=add(e,len(bra)*2.**-1000)
    return C.acb(C.real(z.real)+C.arb(0,C.real(e).upper()),C.real(z.imag)+C.arb(0,C.real(e).upper()))

def reference_enclosure(model):
    """Duhamel: ||(exp(-iHT)-exp(-iET))psi|| <= T||(H-E)psi||.

    Does not assume the saved raw state is an exact eigenvector. All residual
    components and the exact raw norm are enclosed on the supplied Hamiltonian.
    """
    with C.ctx.workprec(128):
        raw=model['state'];v,nu=initial(raw);h=model['hamiltonian'];energy=model['energy']
        terms=[(C.real(Q(h['constant'])-Q(energy)),v)]
        terms.extend((C.real(c),v.pauli(mapping(s))) for s,c in h['terms'])
        residual=linear(terms)
        square=sum((C.real(float(z.real))**2+C.real(float(z.imag))**2 for z in residual.v),C.arb(0))
        ru=upper(square.sqrt());ru=add(ru,residual.e)
        phase=C.phase(C.real(-Q(energy)*Q(.8)))
        radius=C.real(Q(.8))*C.real(ru)/C.real(nu).sqrt()
        physical=C.acb(phase.real+C.arb(0,radius.upper()),phase.imag+C.arb(0,radius.upper()))
        return dict(reference=C.serialize(physical*C.real(nu)),physical_reference=C.serialize(physical),
                    raw_norm_exact=str(nu),state_norm=C.endpoints(C.real(nu)),energy_binary64=energy,
                    residual_l2_upper=ru,reference_radius_upper=upper(radius),
                    method='ARBITRARY_RAW_STATE_DUHAMEL_RESIDUAL',precision_bits=128,
                    exact_eigenstate_assumed=False,Hamiltonian_hash=model['Hamiltonian_hash'])

def evaluate(model,reference,cfg,law=None):
    from .h4_pauli_b0_execution import _interval,center_radius
    with C.ctx.workprec(128):
        raw=model['state'];v,nu=initial(raw);norm=v.norm;n=model['n_qubits'];h=model['hamiltonian']
        retained=h['terms'][:cfg['L_D']];delta=Q(.8/cfg['q'])
        maps={s:mapping(s) for s,c in retained}
        if law:
            ps=list(map(Q,law['component_mapping']['exact_masses']));pis=list(map(Q,law['order_mapping']['exact_masses']))
            if sum(ps)!=1 or sum(pis)!=1 or min(ps+pis)<0:raise ValueError('NOT_CONVEX')
            pms=[mapping(c['label'],c['sign']) for c in law['components']]
            def M(x):
                w=linear([(C.real(p),x.pauli(pm)) for p,pm in zip(ps,pms)])
                w.norm=x.norm # convex combination of unitaries
                return w
            scalar=C.phase(C.real(law['scalar_step_binary64']))
        else:scalar=C.phase(C.real(-Q(h['constant'])*delta))
        for _ in range(cfg['q']):
            for reverse in (False,True):
                for s,c in (reversed(retained) if reverse else retained):v=rotation(v,maps[s],Q(c)*delta/2)
                if law and not reverse:
                    for _ in range(cfg['r']):
                        powers=[v]
                        for j in range(cfg['K']+1):powers.append(M(powers[-1]))
                        pieces=[]
                        for k,pi,a in zip(law['orders'],pis,law['angles']):
                            angle=C.real(a);w=C.real(pi*((-1)**(k//2)))
                            pieces.extend([(C.acb(w*angle.cos()),powers[k]),(C.acb(0,-w*angle.sin()),powers[k+1])])
                        v=linear(pieces);v.norm=norm # expectation of exact event unitaries is a contraction
            v=linear([(scalar,v)]);v.norm=norm
        raw_mean=signal(raw,v,nu);B=Q(law['B_saved_binary64']) if law else Q(1)
        corrected=raw_mean*C.real(B)
        ref=C.acb(_interval(reference['reference']['real'],C),_interval(reference['reference']['imag'],C))
        bias=(corrected-ref)/C.real(nu)
        result=dict(state_actions=1,scalar_precision_bits=128,vector_center_precision_bits=53,
                    physical_corrected_mean=C.serialize(corrected/C.real(nu)),bias=C.serialize(bias),axes={},
                    raw_norm_exact=str(nu),state_renormalized=False,state_l2_error_upper=v.e,
                    B_saved_exact=str(B),B_applied_count=1,actual_rho=None,formal_certificate_accepted=False,
                    error_model='IEEE_BINARY64_GAMMA_L2_WITH_ARB_SCALARS',reference_method=reference['method'])
        for axis,key in [('cosine','real'),('sine','imag')]:
            center,radius=center_radius(result['bias'][key])
            result['axes'][axis]=dict(signed_bias_exact=str(center),u_enc_exact=str(radius),interval_width_exact=str(2*radius),bias_interval=result['bias'][key])
        return result
