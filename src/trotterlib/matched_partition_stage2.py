"""P3 Stage 2 matrix-only nominal diagnostic. No circuits or sampled laws."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from decimal import Decimal, localcontext
from fractions import Fraction as F
from math import factorial, fsum, log, sqrt
from threading import Lock
from types import SimpleNamespace

import numpy as np
from scipy.linalg import expm, eigh
from scipy.sparse import csr_matrix, eye

from .matched_partition_stage1 import (correction_from_two_body, _native_one_body,
    spectral, tolerance, classify)
from .df_hamiltonian import PhysicalSector, _build_excitation_table
from .h2_non_df_pilot import reverse_bits
from .pauli_signal_adapter import PauliHamiltonian, paired_polynomial_action
from .df_trotter.decompose import diag_hermitian
from .df_rte_tail import exact_df_diagonal_coefficients
from .rte import finite_rte_distribution, finite_taylor_operator

T=F.from_float(.8); Q=8; R=4; K=4; DELTA=T/Q; H=DELTA/R
VBAR=4.679666433618258
EPS=np.finfo(float).eps
OPERATIONS=Counter(); _LOCK=Lock()


def bump(name,n=1):
    with _LOCK: OPERATIONS[name]+=n


def norm(a):
    bump('spectral_svd'); return spectral(a)


def fro(a):
    bump('frobenius_norm')
    return float(np.linalg.norm(a.data if hasattr(a,'tocsr') else a))


def mm(a,b):
    bump('explicit_matrix_products'); return a@b


def gamma(m): return m*EPS/(1-m*EPS)


def check(a,b):
    d=len(a); residual=norm(a-b); scale=max(1.,norm(a),norm(b)); tau=tolerance(d,scale)
    return dict(residual=residual,frobenius=fro(a-b),tolerance=tau,
                pass_threshold=tau/4,fail_threshold=4*tau,status=classify(residual,tau))


def scalar_check(error,d,scale=1.):
    tau=tolerance(d,scale)
    return dict(residual=float(error),tolerance=tau,status=classify(error,tau))


def native_context(n):
    d=1<<n; table=_build_excitation_table(PhysicalSector(n,np.arange(d)))
    perm=reverse_bits(n)
    def native(a):
        bump('native_one_body_assemblies')
        m=_native_one_body(a,table,d)
        return m[perm,:][:,perm].tocsr()
    return native


def canonical(data,native):
    t=native(data['df_one_body']); d=t.shape[0]; v=csr_matrix((d,d),dtype=complex)
    for lam,g in zip(data['lambdas'],data['g_matrices'],strict=True):
        a=native(g); v=v+float(lam)*mm(a,a)
    return t.toarray(),v.toarray()


def pauli_dictionary(data):
    from openfermion import InteractionOperator,get_fermion_operator,jordan_wigner
    d=correction_from_two_body(data['spin_two'])
    jw=jordan_wigner(get_fermion_operator(InteractionOperator(0.,-d,data['spin_two'])))
    imaginary_l1=fsum(abs(complex(c).imag) for c in jw.terms.values())
    model=PauliHamiltonian.from_openfermion(jw,num_qubits=len(d))
    ops=[model.sparse_operator([(s,1.)],constant=0.) for s,c in model.terms]
    return SimpleNamespace(matrix=model.sparse_operator(constant=0.).toarray(),
        coefficients=[c for s,c in model.terms],ops=ops,identity=model.constant,
        descriptors=[dict(id='pauli:'+s,label=s,coefficient=c) for s,c in model.terms],
        basis=[],checks={'imaginary_coefficient_l1':scalar_check(imaginary_l1,1<<len(d),1.)})


def df_fragment(index,lam,g,native):
    n=len(g); d=1<<n; ident=eye(d,format='csr',dtype=complex)
    tests={'G_Hermiticity':check(g,g.conj().T)}
    if tests['G_Hermiticity']['status']!='PASS':raise ValueError('DF_G_HERMITICITY')
    u,eta=diag_hermitian(g,sort='descending_abs',assume_hermitian=True)
    bump('DF_G_eigendecompositions')
    tests['basis_orthogonality']=check(u.conj().T@u,np.eye(n))
    tests['G_reconstruction']=check((u*eta)@u.conj().T,g)
    zs=[ident-2*native(np.outer(u[:,i],u[:,i].conj())) for i in range(n)]
    coeff=exact_df_diagonal_coefficients(eta,float(lam))
    v=csr_matrix((d,d),dtype=complex); ops=[]; values=[]; descriptions=[]
    identity=0.; max_inv=0.; max_herm=0.
    for support,c in coeff:
        if not support:identity=float(c);continue
        if c==0.:continue
        op=zs[support[0]] if len(support)==1 else mm(zs[support[0]],zs[support[1]])
        max_inv=max(max_inv,fro(mm(op,op)-ident))
        max_herm=max(max_herm,fro(op-op.conj().T))
        v=v+float(c)*op;ops.append(op);values.append(float(c))
        descriptions.append(dict(id=f'df:{index}:basis:{index}:'+','.join(map(str,support)),
            fragment=index,basis=index,support=list(support),coefficient=float(c)))
    # Frobenius upper estimates suffice here when below the spectral PASS threshold.
    tests['component_involution_F_upper']=scalar_check(max_inv,d)
    tests['component_Hermiticity_F_upper']=scalar_check(max_herm,d)
    return dict(matrix=v,ops=ops,values=values,descriptors=descriptions,identity=identity,
        basis=dict(fragment=index,lambda_factor=float(lam),U=u,eta=eta),checks=tests)


def df_dictionary(data,native,workers):
    args=list(enumerate(zip(data['lambdas'],data['g_matrices'],strict=True)))
    with ThreadPoolExecutor(max_workers=min(workers,len(args))) as pool:
        rows=list(pool.map(lambda item:df_fragment(item[0],*item[1],native),args))
    # Preserve saved fragment order even when the workers complete out of order.
    d=1<<len(data['df_one_body']); v=csr_matrix((d,d),dtype=complex)
    for row in rows:v=v+row['matrix']
    return SimpleNamespace(matrix=v.toarray(),coefficients=[c for x in rows for c in x['values']],
        ops=[p for x in rows for p in x['ops']],identity=fsum(x['identity'] for x in rows),
        descriptors=[p for x in rows for p in x['descriptors']],basis=[x['basis'] for x in rows],
        checks={f'fragment_{i}/{k}':v for i,x in enumerate(rows) for k,v in x['checks'].items()})


def build(data,workers=1):
    native=native_context(len(data['df_one_body']))
    with ThreadPoolExecutor(max_workers=3) as pool:
        p=pool.submit(pauli_dictionary,data)
        df=pool.submit(df_dictionary,data,native,workers)
        tv=pool.submit(canonical,data,native)
        t,v=tv.result(); arms={'P':p.result(),'DF':df.result()}
    v0=v-VBAR*np.eye(len(v)); tests={}
    for name,a in [('T',t),('V0',v0)]:tests[name+'_Hermiticity']=check(a,a.conj().T)
    for name,arm in arms.items():
        tests[name+'/residual_sum']=check(arm.matrix,v0)
        tests[name+'/Hermiticity']=check(arm.matrix,arm.matrix.conj().T)
        tests.update({name+'/'+k:v for k,v in arm.checks.items()})
    return t,v0,arms,tests


def normalization(coefficients,h=H,q=Q,r=R):
    exact=sum((F.from_float(abs(c)) for c in coefficients),F(0))
    lam=fsum(abs(c) for c in coefficients)
    if not np.isfinite(lam) or lam<=0:raise ValueError('NONPOSITIVE_LAMBDA')
    a=lam*float(h); dist=finite_rte_distribution(a,4)
    beta=dist.exact_finite_distribution; logb=q*r*log(beta); b=float(np.exp(logb))
    with localcontext() as ctx:
        ctx.prec=50
        dec=lambda f:Decimal(f.numerator)/Decimal(f.denominator)
        da=dec(exact)*dec(h)
        weights=[da**k/Decimal(factorial(k))*(1+(da/Decimal(k+1))**2).sqrt() for k in (0,2,4)]
        db=sum(weights); dlb=q*r*db.ln(); dB=db**(q*r)
        delta_beta=float(abs(Decimal.from_float(beta)-db))
        delta_B=float(abs(Decimal.from_float(b)-dB))
        check50=dict(lambda_exact=str(exact),a=str(da),weights=list(map(str,weights)),
            beta=str(db),B=str(dB),log_B=str(dlb),beta_abs_difference=delta_beta,
            B_abs_difference=delta_B,log_B_abs_difference=float(abs(Decimal.from_float(logb)-dlb)))
    return dict(lambda_R=lam,lambda_exact=str(exact),lambda_rounding_exact=str(F(lam)-exact),
        dimensionless_step_time=a,weights=list(dist.unnormalized_order_weights),
        order_probabilities=list(dist.order_probabilities),orders=[0,2,4],beta=beta,B=b,log_B=logb,
        decimal50=check50,beta_uncertainty=delta_beta+gamma(16)*abs(beta),
        B_uncertainty=delta_B+gamma(16*q*r)*abs(b),
        evolution_time=float(h*r),rte_steps=r,step_time=float(h),finite_taylor_order=4)


def product_roundoff(a,b):
    # This is the contract's gamma_(8d) || |A||B| ||_F, not an empirical floor.
    return gamma(8*a.shape[-1])*fro(mm(np.abs(a),np.abs(b)))


def product_error(a,b,ea,eb):
    return norm(a)*eb+norm(b)*ea+ea*eb+product_roundoff(a,b)


def polynomial_error(v,h):
    d=len(v); xnorm=abs(h)*norm(v); xf=abs(h)*fro(v)
    term_error=0.; total_error=0.; total_f=sqrt(d)
    for j in range(1,6):
        prev_f=sqrt(d)*xnorm**(j-1)/factorial(j-1)
        term_f=sqrt(d)*xnorm**j/factorial(j)
        term_error=(xnorm*term_error+gamma(8*d)*xf*prev_f)/j+gamma(2)*term_f
        total_error+=term_error+gamma(2)*(total_f+term_f)
        total_f+=term_f
    return total_error


def fourth_power(p,error):
    p2=mm(p,p); e2=product_error(p,p,error,error)
    p4=mm(p2,p2); e4=product_error(p2,p2,e2,e2)
    return p4,e4


def exponential(a,t):
    e=expm((-1j*t)*a); bump('common_expm')
    vals,u=eigh(a); bump('common_spectral_eigendecompositions')
    es=mm(u*np.exp(-1j*t*vals),u.conj().T)
    difference=fro(e-es); ra=fro(mm(u*vals,u.conj().T)-a)
    ru=fro(mm(u.conj().T,u)-np.eye(len(a)))
    herm=fro(a-a.conj().T)
    error=16*(difference+abs(t)*(ra+herm)+ru*(2+ru)+gamma(8*len(a))*(1+abs(t)*norm(a)))
    return e,error,dict(expm_vs_spectral_F=difference,reconstruction_F=ra,
        orthogonality_F=ru,Hermiticity_F=herm,engineering_error=error,time=t)


def raw_event_mean(arm,bundle):
    d=len(arm.matrix); ident=np.eye(d); lam=bundle['lambda_R']
    probabilities=[abs(c)/lam for c in arm.coefficients]
    w=csr_matrix((d,d),dtype=complex)
    for c,p,op in zip(arm.coefficients,probabilities,arm.ops,strict=True):w=w+p*np.sign(c)*op
    w=w.toarray(); probsum=fsum(probabilities)
    # Independently sum component moments, rather than dividing a computed polynomial by beta.
    ew=gamma(4*len(probabilities))*fsum(abs(p)*fro(op) for p,op in zip(probabilities,arm.ops))
    nw=norm(w); w2=mm(w,w); e2=product_error(w,w,ew,ew)
    w4=mm(w2,w2); e4=product_error(w2,w2,e2,e2)
    result=np.zeros_like(w); error=0.; rotation_scalars=[]
    for k,power,ep,pi in zip((0,2,4),(ident,w2,w4),(0.,e2,e4),bundle['order_probabilities']):
        x=bundle['dimensionless_step_time']/(k+1); co=1/sqrt(1+x*x); si=x*co
        rot=co*probsum*ident-1j*si*w
        er=abs(si)*ew+gamma(16)*(abs(co)*sqrt(d)+abs(si)*fro(w))
        term=mm(rot,power); et=product_error(rot,power,er,ep)
        oldf=fro(result);result+=((-1)**(k//2))*pi*term
        error+=abs(pi)*et+gamma(4)*(oldf+abs(pi)*fro(term))
        rotation_scalars.append(dict(k=k,cos=co,sin=si))
    # Include lambda/beta conversion and scalar formula error, not finite-CDF error.
    error+=gamma(64)*max(1.,fro(result))+bundle['beta_uncertainty']/bundle['beta']*fro(result)
    return result,error,dict(component_probabilities=probabilities,component_probability_sum=probsum,
        weighted_moment_vs_V_over_lambda=norm(w-arm.matrix/lam),rotation_scalars=rotation_scalars,
        event_order='products then rotation; (-1)^(k/2)',finite_bit_law_evaluated=False)


def exact_norm(state):
    return sum((F(float(z.real))**2+F(float(z.imag))**2 for z in state),F(0))


def physical_reference(raw,nu):
    from .pauli_mechanism_bias import enclosure
    interval={a:{s:str(F(raw[a][s])/nu) for s in ('lower','upper')} for a in ('real','imag')}
    return interval,enclosure(interval)


def signal_propagation(et,tail,phi,nu,phase,*,q=Q,repetitions=1,
                       et_error=0.,tail_error=0.,phase_error=0.,paired=None,paired_norm=None):
    out=phi.copy(); err=0.; net=norm(et)
    nt=norm(tail) if paired is None else paired_norm
    def action(a,na,ea,v,e):
        bump('signal_dense_matvecs'); new=a@v
        roundoff=gamma(8*len(v))*float(np.linalg.norm(np.abs(a)@np.abs(v)))
        return new,na*e+ea*float(np.linalg.norm(v))+roundoff
    for _ in range(q):
        out,err=action(et,net,et_error,out,err)
        if paired is None:
            for _ in range(repetitions):out,err=action(tail,nt,tail_error,out,err)
        else:
            vin=float(np.linalg.norm(out))
            out=paired_polynomial_action(paired,out,float(DELTA),R,K,normalization=1.)
            bump('paired_polynomial_matvecs',R*(K+1))
            err=nt*err+tail_error*vin+gamma(8*len(out))*sqrt(len(out))*nt*vin
        out,err=action(et,net,et_error,out,err)
    before=float(np.linalg.norm(out));out=phase*out
    err+=phase_error*before+gamma(4)*before
    n=float(nu); inner=np.vdot(phi,out);z=complex(inner/n)
    bump('physical_complex_signals')
    un=(float(np.linalg.norm(phi))*err+gamma(8*len(phi))*float(np.dot(np.abs(phi),np.abs(out))))/n
    un+=abs(z)*(abs(float(F(n)-nu))/n+gamma(4))
    if not np.isfinite([z.real,z.imag,un]).all():raise ValueError('NONFINITE_SIGNAL')
    return dict(real=z.real,imag=z.imag,u_num=un,u_bind=0.,u_ref_transfer=0.,
        precision='complex128; engineering uncertainty, not rigorous',norm_divisions=1,
        scalar_phase_applications=1,B_multiplications=0)


def value(z):return complex(z['real'],z['imag'])


def closure(left,right,extra=0.):
    diff=value(left)-value(right)
    u=sum(z['u_num']+z.get('u_bind',0.) for z in (left,right))+extra
    status='PASS' if abs(diff)<=u else ('UNRESOLVED' if abs(diff)<=4*u else 'FAIL')
    return dict(real=diff.real,imag=diff.imag,absolute=abs(diff),uncertainty=u,status=status)


def bias(left,right,*,bind=0.,ref_width=None,transfer=0.):
    diff=value(left)-value(right);out={}
    for axis,num in [('real',diff.real),('imag',diff.imag)]:
        un=left['u_num']+right.get('u_num',0.)
        ur=(ref_width or {}).get(axis,0.)+transfer
        total=un+bind+ur
        out[axis]=dict(value=num,u_num=un,u_bind=bind,u_ref_transfer=ur,
            uncertainty=total,status='RESOLVED_ENGINEERING' if abs(num)>total else 'UNRESOLVED')
    return out


def evaluate(t,v0,arms,phi,nu,constant,saved_h,reference_raw,reference_residual,checkpoint=None):
    d=len(t);h=float(H);dt=float(DELTA)
    et,ee,et_info=exponential(t,dt/2);ev,ve,ev_info=exponential(v0,dt)
    angle=F.from_float(float(constant))*T+F.from_float(VBAR)*T
    fa=float(angle);phase=np.exp(-1j*fa)
    phase_error=float(abs(F(fa)-angle))+gamma(8)*(1+abs(fa))
    pc=finite_taylor_operator(v0,h,K);bump('finite_polynomial_matrix_products',5)
    ec=polynomial_error(v0,h);pc4,ec4=fourth_power(pc,ec);npc=norm(pc)
    sig={};normal={};law={};closures={};bindings={}
    def save():
        if checkpoint:checkpoint(dict(signals=sig,normalization=normal,ideal_moments=law,
            closures=closures,bindings=bindings,common_exponentials=dict(T_half=et_info,V_full=ev_info)))
    sig['split']=signal_propagation(et,ev,phi,nu,phase,et_error=ee,tail_error=ve,phase_error=phase_error)
    save()
    for name,arm in arms.items():
        bundle=normalization(arm.coefficients);normal[name]=bundle;bump('normalization_bundles')
        epsilon=norm(arm.matrix-v0)+gamma(2)*(fro(arm.matrix)+fro(v0))
        rr=h*max(norm(arm.matrix),norm(v0));lip=sum(rr**j/factorial(j) for j in range(5))
        onebind=h*lip*epsilon
        one_norm=npc+ec+onebind
        tailbind=R*one_norm**(R-1)*onebind
        whole=Q*max(1.,norm(et)**2*max(norm(pc4),one_norm**R))**(Q-1)*norm(et)**2*tailbind
        bindings[name]=dict(residual_operator_difference=epsilon,polynomial_step_sensitivity=onebind,
            repeated_signal_uncertainty=whole,identity_defect=arm.identity-VBAR)
        ep=polynomial_error(arm.matrix,h)
        if name=='P':
            eb=R*(one_norm+ep)**(R-1)*ep
            sig['nom_P']=signal_propagation(et,None,phi,nu,phase,et_error=ee,tail_error=eb,
                phase_error=phase_error,paired=csr_matrix(arm.matrix),paired_norm=one_norm**R)
        else:
            w=arm.matrix/bundle['lambda_R'];a=bundle['dimensionless_step_time']
            pd=finite_taylor_operator(w,a,K);bump('finite_polynomial_matrix_products',5)
            ep+=lip*norm(a*w-h*arm.matrix)
            pd4,ed4=fourth_power(pd,ep)
            sig['nom_DF']=signal_propagation(et,pd4,phi,nu,phase,et_error=ee,tail_error=ed4,phase_error=phase_error)
        sig['nom_'+name]['u_bind']=whole
        save()
    sig['nom_star']=signal_propagation(et,pc4,phi,nu,phase,et_error=ee,tail_error=ec4,phase_error=phase_error)
    for name in arms:closures['nom_'+name+'_vs_star']=closure(sig['nom_'+name],sig['nom_star'])
    closures['nom_P_vs_DF']=closure(sig['nom_P'],sig['nom_DF']);save()
    if any(x['status']!='PASS' for x in closures.values()):raise ValueError('NOMINAL_COMPARISON_NONPASS')
    for name,arm in arms.items():
        raw,er,info=raw_event_mean(arm,normal[name]);law[name]=info
        sig['raw_'+name]=signal_propagation(et,raw,phi,nu,phase,repetitions=R,
            et_error=ee,tail_error=er,phase_error=phase_error)
        b=normal[name]['B'];zb=b*value(sig['raw_'+name])
        corrected=dict(real=zb.real,imag=zb.imag,u_num=b*sig['raw_'+name]['u_num'],u_bind=0.)
        closures['B_raw_vs_nom_'+name]=closure(corrected,sig['nom_'+name],
            extra=normal[name]['B_uncertainty']*abs(value(sig['raw_'+name])))
        save()
        if closures['B_raw_vs_nom_'+name]['status']!='PASS':raise ValueError('RAW_NORMALIZATION_CLOSURE_NONPASS')
    interval,ref=physical_reference(reference_raw,nu)
    new_h=t+v0+float(constant+VBAR)*np.eye(d)
    difference=norm(new_h-saved_h)
    transfer=abs(float(T))*(difference+float(reference_residual)+gamma(2)*(fro(new_h)+fro(saved_h)))
    refpoint=dict(real=ref['real'],imag=ref['imag'],u_num=0.)
    biases={'outer':bias(sig['split'],refpoint,ref_width=ref['axis_resolution'],transfer=transfer)}
    for name in arms:biases['finite_'+name]=bias(sig['nom_'+name],sig['split'],bind=sig['nom_'+name]['u_bind'])
    good=all(x['status']=='PASS' for x in closures.values())
    unresolved=any(x['status']=='UNRESOLVED' for row in biases.values() for x in row.values())
    stop=('STOP_STAGE2_CORRECTNESS_NONPASS_NO_RETRY' if not good else
        'STOP_STAGE2_UNRESOLVED_COMPONENTS' if unresolved else
        'STOP_STAGE2_NOMINAL_DIAGNOSTIC_COMPLETE_NO_AUTOMATIC_FOLLOWUP')
    return dict(signals=sig,normalization=normal,ideal_moments=law,closures=closures,biases=biases,
        bindings=bindings,common_exponentials=dict(T_half=et_info,V_full=ev_info),
        reference=dict(physical_interval=interval,display=ref,H_new_vs_saved=difference,
            saved_H_C_vs_snapshot_upper=str(reference_residual),transfer_uncertainty=transfer,
            newly_generated=False,raw_reference_norm_divisions=1),
        phase=dict(exact_angle=str(angle),binary64_angle=fa,engineering_error=phase_error),
        numerical_status='PASS' if good else 'NONPASS',stop=stop)
