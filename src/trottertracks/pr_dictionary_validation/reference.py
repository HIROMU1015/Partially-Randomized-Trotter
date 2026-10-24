"""Independent small-register CAR/exterior-minor/high-precision references.

Does NOT import candidate algebra, canonicalization, router or moment helpers.
No matrices or high-precision contexts are instantiated on import.
"""
from __future__ import annotations
from fractions import Fraction
import itertools
import re


_SAVED_UNSIGNED = r'(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?'
_SAVED_REAL = r'[+-]?' + _SAVED_UNSIGNED
_SAVED_COMPLEX = re.compile(r'\((' + _SAVED_REAL + r') ([+-]) ('
                            + _SAVED_UNSIGNED + r')j\)')


def saved_scalar_parts(text):
    """Strict ctx.nstr grammar, preserving decimal text without float/eval."""
    if not isinstance(text, str):
        raise ValueError('Saved reference scalar must be text')
    if re.fullmatch(_SAVED_REAL, text):
        return text, '0'
    match = _SAVED_COMPLEX.fullmatch(text)
    if not match:
        raise ValueError('Unknown/nonfinite saved reference scalar')
    real, sign, imaginary = match.groups()
    return real, ('-' if sign == '-' else '') + imaginary


def validate_saved_reference(payload, *, full):
    """Metadata/string-only validation; no high-precision or matrix operations."""
    shapes = dict(h=16, d=16, r=16, operator=16, wrapper_X=32, wrapper_Y=32) if full else dict(controlled=32)
    expected = set(shapes) | ({'signal'} if full else set())
    if (set(payload) != {'digits', 'values'} or type(payload['digits']) is not int
            or payload['digits'] != 128 or not isinstance(payload['values'], dict)
            or set(payload['values']) != expected):
        raise ValueError('Saved reference schema/digits/keys mismatch')
    for key, size in shapes.items():
        matrix = payload['values'][key]
        if (not isinstance(matrix, list) or len(matrix) != size
                or any(not isinstance(row, list) or len(row) != size for row in matrix)):
            raise ValueError('Saved reference shape mismatch: '+key)
        for row in matrix:
            for text in row:
                saved_scalar_parts(text)
    if full:
        saved_scalar_parts(payload['values']['signal'])
    return payload


def restore_saved_reference(payload, *, full):
    """Explicit science entry: restore only selected saved 128-digit targets.

    No reference generation or precision re-comparison. Real/imag strings enter
    the high-precision context before the caller's existing binary64 conversion.
    """
    validate_saved_reference(payload, full=full)
    ctx = precision_context(128)
    def scalar(text):
        real, imaginary = saved_scalar_parts(text)
        value = ctx.mpc(ctx.mpf(real), ctx.mpf(imaginary))
        if not ctx.isfinite(value):
            raise ValueError('Nonfinite restored reference')
        return value
    keys = ('operator', 'wrapper_X', 'wrapper_Y') if full else ('controlled',)
    values = {key: ctx.matrix([[scalar(v) for v in row] for row in payload['values'][key]])
              for key in keys}
    if full:
        values['signal'] = scalar(payload['values']['signal'])
    return dict(ctx=ctx, **values)


def precision_context(digits: int):
    from mpmath import mp
    if digits not in (90,128): raise ValueError('Only predeclared reference precisions')
    ctx=mp.clone(); ctx.dps=digits
    return ctx


def exact(ctx, text):
    f=Fraction(text)
    return ctx.mpf(f.numerator)/f.denominator


def car_annihilator(ctx, modes, p):
    a=ctx.matrix(1<<modes,1<<modes)
    for col in range(1<<modes):
        if (col>>p)&1:
            sign=-1 if (col&((1<<p)-1)).bit_count()%2 else 1
            a[col^(1<<p),col]=sign
    return a


def orbital_frame(ctx, modes, fragment):
    # Independent implementation, reading the original rotation specification.
    v=ctx.eye(modes)
    for item in fragment.rotations:
        g=ctx.eye(modes); c=exact(ctx,item.cosine); s=exact(ctx,item.sine)
        g[item.p,item.p]=c; g[item.q,item.q]=c
        g[item.p,item.q]=-s; g[item.q,item.p]=s
        v=v*g
    return v


def exterior_frame(ctx, v):
    n=v.rows; dim=1<<n; b=ctx.matrix(dim,dim)
    occupied=[tuple(p for p in range(n) if (index>>p)&1) for index in range(dim)]
    for row,rs in enumerate(occupied):
        for col,cs in enumerate(occupied):
            if len(rs)!=len(cs): continue
            b[row,col]=1 if not rs else ctx.det(ctx.matrix([[v[i,j] for j in cs] for i in rs]))
    return b


def original_operators(ctx, spec):
    dim=1<<spec.modes; eye=ctx.eye(dim)
    ns=[]
    for p in range(spec.modes):
        a=car_annihilator(ctx,spec.modes,p); ns.append(a.H*a)
    d=ctx.matrix(dim,dim)
    for text,n in zip(spec.deterministic,ns,strict=True): d+=exact(ctx,text)*n
    fs=[]; frames={}
    for f in spec.fragments:
        b=exterior_frame(ctx,orbital_frame(ctx,spec.modes,f)); frames[f.fragment_id]=b
        diagonal=ctx.matrix(dim,dim)
        for text,n in zip(f.diagonal,ns,strict=True): diagonal+=exact(ctx,text)*n
        fs.append(b*diagonal*diagonal*b.H)
    residual=ctx.matrix(dim,dim)
    for f in fs: residual+=f
    h=exact(ctx,spec.scalar)*eye+d+residual
    return h,d,residual,tuple(fs),frames


def pauli_operator(ctx, word):
    # Explicit bit action; no Qiskit Pauli or candidate JW multiplication.
    n=len(word); result=ctx.matrix(1<<n,1<<n)
    for col in range(1<<n):
        row=col; phase=ctx.mpc(1)
        for p,ch in enumerate(word):
            bit=(col>>p)&1
            if ch=='X': row^=1<<p
            elif ch=='Y': row^=1<<p; phase*=(-ctx.j if bit else ctx.j)
            elif ch=='Z': phase*=(-1 if bit else 1)
            elif ch!='I': raise ValueError('Invalid Pauli word')
        result[row,col]=phase
    return result


def unsigned_atom_operator(ctx, atom, original_frames):
    q=pauli_operator(ctx,atom.word)
    if atom.kind=='FRAME_Z':
        b=original_frames[atom.frame_id]; return b*q*b.H
    return q


def dictionary_operator(ctx, atoms, original_frames):
    if not atoms: raise ValueError('Dimension required for empty dictionary')
    n=1<<len(atoms[0].word); out=ctx.matrix(n,n)
    for a in atoms: out+=exact(ctx,str(a.coefficient))*unsigned_atom_operator(ctx,a,original_frames)
    return out


def independent_law(ctx, atoms, micro_time, cutoff):
    lam=sum((abs(exact(ctx,str(a.coefficient))) for a in atoms),ctx.mpf(0))
    if not lam: raise ValueError('Deterministic-only law is not randomized')
    tau=lam*exact(ctx,str(micro_time))
    orders=tuple(range(0,cutoff+1,2))
    weights=tuple(abs(tau)**n/ctx.factorial(n)*ctx.sqrt(1+(tau/(n+1))**2) for n in orders)
    norm=sum(weights,ctx.mpf(0))
    probs=tuple(abs(exact(ctx,str(a.coefficient)))/lam for a in atoms)
    return lam,tau,orders,weights,norm,probs


def event_operator(ctx, event, by_id, original_frames, tau):
    n=event.taylor_order
    child=by_id[event.rotation_component_id]
    dim=1<<len(child.word); result=ctx.eye(dim)
    for aid in event.product_component_ids:
        a=by_id[aid]; q=unsigned_atom_operator(ctx,a,original_frames)
        result=(1 if a.coefficient>0 else -1)*q*result
    q=unsigned_atom_operator(ctx,child,original_frames)
    theta=ctx.atan(tau/(n+1))*(1 if child.coefficient>0 else -1)
    rotation=ctx.cos(theta)*ctx.eye(dim)-ctx.j*ctx.sin(theta)*q
    return ((-1)**(n//2))*rotation*result


def polynomial(ctx, residual, time, degree):
    result=ctx.eye(residual.rows); term=ctx.eye(residual.rows)
    for k in range(1,degree+1):
        term=term*(-ctx.j*time*residual)/k; result+=term
    return result


def finite_target(ctx,spec,context):
    _h,d,r,_fs,_frames=original_operators(ctx,spec)
    delta=exact(ctx,str(context.delta)); micro=exact(ctx,str(context.micro_time))
    method='taylor' if ctx.dps==90 else 'pade'
    half=ctx.expm(-ctx.j*delta*d/2,method=method)
    one=half*(polynomial(ctx,r,micro,context.cutoff+1)**context.r)*half
    return ctx.exp(-ctx.j*exact(ctx,spec.scalar)*exact(ctx,context.total_time))*(one**context.q)


def physical_target(ctx,spec,time):
    h,*_=original_operators(ctx,spec)
    return ctx.expm(-ctx.j*exact(ctx,str(time))*h,method='taylor' if ctx.dps==90 else 'pade')


def trajectory_operator(ctx,spec,context,macros,atoms):
    _h,d,_r,_fs,frames=original_operators(ctx,spec)
    _,tau,_,_,_,_=independent_law(ctx,atoms,context.micro_time,context.cutoff)
    half=ctx.expm(-ctx.j*exact(ctx,str(context.delta))*d/2,
                  method='taylor' if ctx.dps==90 else 'pade')
    result=ctx.eye(1<<spec.modes); by_id={a.atom_id:a for a in atoms}
    for events in macros:
        step=ctx.eye(result.rows)
        for e in events: step=event_operator(ctx,e,by_id,frames,tau)*step
        result=half*step*half*result
    return ctx.exp(-ctx.j*exact(ctx,spec.scalar)*exact(ctx,context.total_time))*result


def controlled_operator(ctx, u):
    n=u.rows; result=ctx.eye(2*n)
    for i in range(n):
        for j in range(n): result[n+i,n+j]=u[i,j]
    return result


def signal(ctx,operator,occupied):
    index=sum(1<<p for p in occupied)
    # Pure computational-basis input, no ground-state/energy oracle.
    return operator[index,index]


def wrapper_operator(ctx,u,occupied,axis):
    if axis not in ('X','Y'): raise ValueError('Unknown axis')
    modes=(u.rows-1).bit_length(); total=2*u.rows
    prep=pauli_operator(ctx,tuple('X' if p in occupied else 'I' for p in range(modes)))
    preparation=ctx.matrix(total,total)
    for branch in range(2):
        for i in range(u.rows):
            for j in range(u.rows): preparation[branch*u.rows+i,branch*u.rows+j]=prep[i,j]
    had=ctx.matrix(total,total); read=ctx.eye(total)
    for i in range(u.rows):
        for a,b,val in ((0,0,1),(0,1,1),(1,0,1),(1,1,-1)):
            had[a*u.rows+i,b*u.rows+i]=val/ctx.sqrt(2)
        if axis=='Y': read[u.rows+i,u.rows+i]=-ctx.j
    return had*read*controlled_operator(ctx,u)*had*preparation


def to_numpy(matrix):
    import numpy as np
    return np.array([[complex(matrix[i,j]) for j in range(matrix.cols)]
                     for i in range(matrix.rows)],dtype=complex)
